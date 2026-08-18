"""证据检索节点。

三种模式：
- demo / llm：使用 MockSearchProvider
- full：使用 TavilySearchProvider 真实搜索 + LLM 提取

支持并发搜索（最多 3 个线程）和查询去重。
支持第二轮补充检索（由 workflow 调用时传入 claim_queries_override）。
支持工作流 deadline 检查，超时后停止开启新请求。
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.config import settings
from src.models import AgentState, Claim, Evidence
from src.tools.search_tool import (
    SearchProvider,
    SearchResult,
    _validate_url,
    choose_topic_for_claim,
    deduplicate_by_url,
    get_search_provider,
)


class _SearchDeadline(Exception):
    """搜索阶段超过工作流 deadline 时抛出。"""
    pass


def _check_search_deadline(state: AgentState) -> None:
    """检查工作流 deadline，超时则抛 _SearchDeadline。"""
    deadline = state.metadata.get("_workflow_deadline")
    if deadline is not None and time.time() > deadline:
        raise _SearchDeadline(
            f"搜索超时：工作流已运行 {time.time() - state.metadata.get('_workflow_start', time.time()):.0f}s"
        )


def _get_search_queries(state: AgentState, claim_id: str, fallback_text: str) -> list[str]:
    """从核查计划中提取搜索语句，带查询去重。"""
    claim = next((c for c in state.claims if c.claim_id == claim_id), None)
    prefer: list[str] = []
    if claim is not None:
        prefer = list(getattr(claim, "search_keywords", None) or [])

    plan_queries: list[str] = []
    for plan in state.verification_plan:
        if isinstance(plan, dict):
            if plan.get("claim_id") == claim_id:
                plan_queries = list(plan.get("search_queries") or plan.get("keywords") or [])
                break
        elif hasattr(plan, "claim_id") and plan.claim_id == claim_id:
            plan_queries = list(getattr(plan, "search_queries", None) or [])
            break
    if isinstance(state.verification_plan, dict) and not plan_queries:
        plan = state.verification_plan.get(claim_id, {})
        plan_queries = list(plan.get("keywords") or plan.get("search_queries") or [])

    merged: list[str] = []
    for q in prefer + plan_queries:
        if q and q not in merged:
            merged.append(q)
    if merged:
        return merged[:settings.SEARCH_MAX_QUERIES_PER_CLAIM]
    return [fallback_text[:20]]


def search_evidence(
    state: AgentState,
    claim_queries_override: dict[str, list[str]] | None = None,
    round_label: int = 1,
) -> AgentState:
    """根据核查计划检索每个主张的证据。支持并发搜索和 deadline 检查。"""
    try:
        if state.mode == "full":
            return _search_evidence_run(state, claim_queries_override, round_label, full=True)
        return _search_evidence_run(state, claim_queries_override, round_label, full=False)
    except _SearchDeadline:
        raise RuntimeError("搜索阶段超时：已超过工作流时间上限")


def _grade_search_result_publisher(r: SearchResult) -> str:
    p = (r.publisher or "")
    high = ["日报", "新闻", "通讯社", "周刊", "电视台", "集团官网", "官方", "政府", "教育部",
            "应急管理局", "轨道交通", "教育局", "经纪公司", "时代峰峻"]
    if any(k in p for k in high):
        return "A"
    medium = ["媒体", "时报", "晨报", "晚报", "新闻网"]
    if any(k in p for k in medium):
        return "B"
    if "搬运" in p or "粉丝" in p or "爆料" in p or "匿名" in p or "论坛" in p:
        return "D"
    if "网络用户" in p or "社交" in p or "剪辑" in p:
        return "E"
    return "C"


def _independence_group(r: SearchResult, query: str) -> str:
    title = r.title or ""
    content = getattr(r, "content", "") or ""
    markers = ["原始爆料组-A", "转载原始爆料-A", "原始爆料组", "转载原始爆料"]
    for m in markers:
        if m in title or m in content:
            return f"repost-group::{m}"
    return f"{r.domain or ''}::{title[:40]}::{query[:30]}"


def _stance_from_content(claim: Claim, r: SearchResult) -> tuple[str, bool]:
    text = (r.content or "")
    ct = (claim.text or "")
    if any(w in text for w in ["未发布", "未提及", "未接到", "并非", "不是", "澄清说明",
                                "未得到", "待核实", "真实性存疑"]):
        return "refutes", ("官方" in (r.publisher or "") or "政府" in (r.publisher or ""))
    if any(w in ct for w in ["情侣", "恋爱", "私人关系"]):
        if any(w in text for w in ["未得到双方本人", "未得到", "匿名", "粉丝解读", "二次创作"]):
            return "unclear", False
        return "partial", False
    return "supports", ("官方" in (r.publisher or ""))


def _choose_excerpt_matching_body(claim: Claim, r: SearchResult, full_body: str) -> str:
    body = full_body or r.content or ""
    if not body:
        return ""
    keywords = [claim.text or ""]
    keywords += list(getattr(claim, "entities", None) or [])
    for ent, aliases in (getattr(claim, "entity_aliases", None) or {}).items():
        keywords.append(ent)
        keywords.extend(list(aliases or []))
    snippet = ""
    for kw in keywords:
        if not kw:
            continue
        i = body.lower().find(kw.lower())
        if i >= 0:
            start = max(0, i - 60)
            end = min(len(body), i + 240)
            snippet = body[start:end].strip().replace("\n", " ")
            break
    if not snippet:
        snippet = body[:300].replace("\n", " ")
    return snippet[:300]


def _merge_evidence_maps(existing, new):
    out = {k: list(v) for k, v in (existing or {}).items()}
    seen_urls: set[tuple[str, str]] = set()
    for vs in out.values():
        for ev in vs:
            seen_urls.add((ev.claim_id, ev.source_url or ""))
    for cid, vs in (new or {}).items():
        cur = out.setdefault(cid, [])
        existing_count = len(cur)
        for j, ev in enumerate(vs):
            key = (cid, ev.source_url or "")
            if key in seen_urls:
                continue
            seen_urls.add(key)
            try:
                new_id = f"{cid}-r2e{existing_count + j + 1}"
                ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else {**ev.__dict__}
                ev_dict["evidence_id"] = new_id
                from src.models import Evidence as _E
                cur.append(_E(**ev_dict))
            except Exception:
                cur.append(ev)
    return out


def _search_evidence_run(
    state: AgentState,
    claim_queries_override: dict[str, list[str]] | None,
    round_label: int,
    full: bool,
) -> AgentState:
    override = claim_queries_override or {}
    search_start = time.time()

    provider: SearchProvider = get_search_provider("full" if full else "demo")
    rounds = state.metadata.setdefault("_search_rounds", {})
    for c in state.claims:
        rounds.setdefault(c.claim_id, 0)

    evidence_this_round: dict[str, list[Evidence]] = {}
    total_fetched = 0
    total_queries = 0
    failed_queries = 0
    seen_queries: set[str] = set()

    # Collect all (claim_id, queries) pairs
    claim_query_pairs: list[tuple[str, list[str], str]] = []
    for claim in state.claims:
        if override and claim.claim_id not in override:
            continue
        rounds[claim.claim_id] = max(rounds[claim.claim_id], round_label)

        if claim.claim_id in override:
            queries = list(override[claim.claim_id])
        else:
            queries = _get_search_queries(state, claim.claim_id, claim.text)
        queries = queries[: settings.SEARCH_MAX_QUERIES_PER_CLAIM]
        topic = choose_topic_for_claim(claim.text, claim.claim_type)
        claim_query_pairs.append((claim.claim_id, queries, topic))

    # Execute searches concurrently
    def _execute_single_search(query: str, topic: str) -> tuple[str, list[SearchResult], Optional[str], float]:
        """Execute a single search with deduplication cache."""
        cache_key = f"{query}::{topic}"
        if cache_key in seen_queries:
            return query, [], "duplicate", 0.0
        seen_queries.add(cache_key)
        max_per_query = max(1, min(settings.SEARCH_MAX_RESULTS_PER_QUERY, 5))
        results, rt, err = provider.search(query=query, max_results=max_per_query, topic=topic)
        return query, results, err, rt

    # Flatten all queries for concurrent execution
    all_queries: list[tuple[str, str, str]] = []  # (claim_id, query, topic)
    for cid, queries, topic in claim_query_pairs:
        for q in queries:
            all_queries.append((cid, q, topic))

    # Run searches with limited concurrency
    search_results_map: dict[str, list[SearchResult]] = {}  # (claim_id, query) -> results
    query_errors: dict[str, list[str]] = {}  # claim_id -> [errors]

    max_workers = min(settings.SEARCH_MAX_CONCURRENT, max(1, len(all_queries)))

    if full and len(all_queries) > 1:
        # 分批提交，每批之间检查 deadline
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {}
            search_timed_out = False
            for idx, (cid, q, topic) in enumerate(all_queries):
                try:
                    _check_search_deadline(state)
                except _SearchDeadline:
                    search_timed_out = True
                    break
                future = executor.submit(_execute_single_search, q, topic)
                future_to_meta[future] = (cid, q)

            for future in as_completed(future_to_meta):
                cid, q = future_to_meta[future]
                try:
                    query, results, err, rt = future.result()
                    key = f"{cid}::{q}"
                    if err == "duplicate":
                        continue
                    if err:
                        failed_queries += 1
                        query_errors.setdefault(cid, []).append(f"q:{q[:20]}:{err[:60]}")
                    else:
                        total_queries += 1
                        search_results_map[key] = results
                        total_fetched += len(results)
                except Exception as e:
                    failed_queries += 1
                    query_errors.setdefault(cid, []).append(f"q:{q[:20]}:{str(e)[:60]}")
    else:
        search_timed_out = False
        for cid, q, topic in all_queries:
            try:
                _check_search_deadline(state)
            except _SearchDeadline:
                search_timed_out = True
                break
            query, results, err, rt = _execute_single_search(q, topic)
            key = f"{cid}::{q}"
            if err == "duplicate":
                continue
            total_queries += 1
            if err:
                failed_queries += 1
                query_errors.setdefault(cid, []).append(f"q:{q[:20]}:{err[:60]}")
            else:
                search_results_map[key] = results
                total_fetched += len(results)

    # Now process claims with collected results
    for cid, queries, topic in claim_query_pairs:
        claim = next((c for c in state.claims if c.claim_id == cid), None)
        if claim is None:
            continue

        claim_results: list[SearchResult] = []
        for q in queries:
            key = f"{cid}::{q}"
            if key in search_results_map:
                claim_results.extend(search_results_map[key])

        claim_results = deduplicate_by_url(claim_results)

        # Content extraction
        extract_urls: list[str] = []
        for r in claim_results[:2]:
            raw_content = ""
            if isinstance(r.raw_response, dict):
                raw_content = str(r.raw_response.get("raw_content") or r.raw_response.get("content") or "")
            if not raw_content:
                raw_content = r.content or ""
            if len(raw_content.strip()) < 120 and r.url:
                extract_urls.append(r.url)
        extracted_map: dict[str, str] = {}
        if extract_urls:
            try:
                extracted_map = provider.extract(extract_urls) or {}
            except Exception:
                extracted_map = {}

        # Build Evidence objects
        now = datetime.now()
        evidences: list[Evidence] = []
        for ri, r in enumerate(claim_results, start=1):
            raw = ""
            if isinstance(r.raw_response, dict):
                raw = str(r.raw_response.get("raw_content") or "")
            if not raw:
                raw = extracted_map.get(r.url or "", "")
            if not raw:
                raw = r.content or ""

            excerpt = _choose_excerpt_matching_body(claim, r, raw)
            if not excerpt and r.content:
                excerpt = r.content[:260]
            raw_excerpt = raw[:400] if raw else excerpt

            grade = _grade_search_result_publisher(r)
            stance, is_primary = _stance_from_content(claim, r)
            ig = _independence_group(r, queries[0] if queries else "")

            ct = (claim.text or "")
            if any(k in ct for k in ["情侣", "恋爱", "私人关系", "私生活"]):
                if grade in ("C", "D", "E") or any(k in (r.publisher or r.title or "")
                                                      for k in ["粉丝", "匿名", "爆料", "搬运",
                                                                "剪辑", "论坛", "网络用户", "CP向"]):
                    if grade == "C":
                        grade = "D"
                    is_primary = False

            evidence_id = f"{claim.claim_id}-{'r2' if round_label > 1 else ''}e{ri}"

            if not _validate_url(r.url):
                continue

            ev = Evidence(
                evidence_id=evidence_id,
                claim_id=claim.claim_id,
                source_title=r.title or "",
                source_url=r.url or "",
                publisher=r.publisher or "",
                published_at=r.published_at or now,
                retrieved_at=now,
                evidence_summary=(r.content or "")[:200],
                relevant_excerpt=excerpt,
                source_type=("官方一手" if grade == "A" else
                             "权威署名媒体" if grade == "B" else
                             "一般媒体/转载" if grade == "C" else
                             "粉丝/自媒体/匿名爆料" if grade == "D" else "未经验证"),
                source_grade=grade,
                source_domain=r.domain,
                source_content=r.content or "",
                supports_or_refutes=stance,
                is_primary_source=is_primary,
                reliability_reason=(
                    "来自官方/署名来源" if grade in ("A", "B") else
                    "来自粉丝/匿名/搬运类来源，证据力有限" if grade in ("D", "E") else
                    "一般转载/非署名内容"
                ),
                search_query=queries[0] if queries else "",
                extraction_status="success" if excerpt else "insufficient_content",
                evidence_stance=stance,
                independence_group=ig,
            )
            evidences.append(ev)

        evidence_this_round[claim.claim_id] = evidences

    # Merge into state
    state.evidence = _merge_evidence_maps(state.evidence or {}, evidence_this_round)
    pool: list[Evidence] = list(state.evidence_pool or [])
    for vs in evidence_this_round.values():
        pool.extend(vs)
    state.evidence_pool = pool
    state.metadata["_search_rounds"] = rounds

    total_round_evs = sum(len(v) for v in evidence_this_round.values())
    elapsed = time.time() - search_start

    state.search_stats[f"round{round_label}_queries"] = total_queries
    state.search_stats[f"round{round_label}_evidence"] = total_round_evs
    state.search_stats[f"round{round_label}_failed_queries"] = failed_queries
    state.search_stats["total_results_fetched"] = (
        int(state.search_stats.get("total_results_fetched", 0) or 0) + total_fetched
    )
    state.search_stats["valid_evidence_count"] = (
        int(state.search_stats.get("valid_evidence_count", 0) or 0) + total_round_evs
    )

    if query_errors:
        for cid, errs in query_errors.items():
            merged_err = f"主张 {cid} 搜索错误: {'; '.join(errs)}"
            state.errors.append(merged_err)

    state.log(
        step="search" if round_label == 1 else "sufficiency",
        action=(f"第{round_label}轮检索：共 {total_queries} 个查询，"
                f"{total_round_evs} 条证据，耗时 {elapsed:.1f}s，"
                f"{failed_queries} 个查询失败"),
        status="completed" if round_label == 1 else "running",
        details={
            "round": round_label,
            "total_queries": total_queries,
            "round_evidence": total_round_evs,
            "failed_queries": failed_queries,
            "elapsed_s": round(elapsed, 1),
        },
    )
    if search_timed_out:
        state.metadata["_search_timed_out"] = True
    return state
