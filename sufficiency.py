"""证据充分性评估节点：第一轮检索后判断是否需要自动发起第二轮补充检索。

不使用真实 API；与 search_tool、search 节点协同，生成"补充检索计划"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.models import AgentState, Claim, Evidence

MAX_ROUNDS_PER_CLAIM = 2
MAX_QUERIES_PER_ROUND = 3


@dataclass
class ClaimSufficiency:
    claim_id: str
    sufficient: bool
    reasons: list[str] = field(default_factory=list)
    round: int = 1
    follow_up_queries: list[str] = field(default_factory=list)


def _stance_of(e: Evidence) -> str:
    """统一从 Evidence.evidence_stance 读取（兼容旧字段 .stance 若存在）。"""
    return getattr(e, "evidence_stance", None) or getattr(e, "stance", "context") or "context"


def _has_related_evidence(claim_id: str, evs: list[Evidence]) -> bool:
    rel = [e for e in evs if e.claim_id == claim_id and _stance_of(e) in ("supports", "refutes", "context")]
    return len(rel) > 0


def _high_quality_count(claim_id: str, evs: list[Evidence]) -> int:
    grades = {"A", "B"}
    cnt = 0
    for e in evs:
        if e.claim_id != claim_id:
            continue
        if (e.source_grade or "").upper() in grades:
            cnt += 1
    return cnt


def _count_independent_sources(claim_id: str, evs: list[Evidence]) -> int:
    """独立来源：按 independence_group 去重；同组只算 1（同一原始爆料被转载不重复计数）。"""
    groups: set[str] = set()
    for e in evs:
        if e.claim_id != claim_id:
            continue
        if _stance_of(e) not in ("supports", "refutes"):
            continue
        g = getattr(e, "independence_group", None) or f"{getattr(e, 'source_domain', '')}::{e.source_title}"
        groups.add(g)
    return len(groups)


def _entity_disambiguation_done(claim: Claim, evs: list[Evidence]) -> bool:
    """context/身份核查主张：需要出现过同时包含"人名+所属机构"的证据内容或关键词。"""
    if not claim.needs_background_verification:
        return True
    alias_kw: list[str] = []
    for ent, aliases in (claim.entity_aliases or {}).items():
        alias_kw.append(ent)
        alias_kw.extend(list(aliases or []))
    if not alias_kw:
        return True
    for e in evs:
        if e.claim_id != claim.claim_id:
            continue
        raw_excerpt = (
            getattr(e, "source_content", "") or
            getattr(e, "raw_content_excerpt", "") or
            ""
        )
        haystack = f"{(e.source_title or '')} {(getattr(e, 'relevant_excerpt', '') or '')} {raw_excerpt}".lower()
        # 至少命中 2 个不同关键词（人名+所属机构），才算"消歧完成"
        hits = sum(1 for k in set(alias_kw) if k and k.lower() in haystack)
        if hits >= 2:
            return True
    return False


def _time_conflict_found(claim: Claim, evs: list[Evidence]) -> bool:
    times = [e.published_at for e in evs if e.claim_id == claim.claim_id and getattr(e, "published_at", None)]
    # 如果主张带时间（2020），但证据时间集中在多年后，算冲突
    if not (getattr(claim, "time_reference", None) or getattr(claim, "location", None)):
        return False
    _ = times  # 保留引用，后续可扩展
    return False


def _evidence_from_primary_sources_only_fan_media(claim: Claim, evs: list[Evidence]) -> bool:
    rel = [e for e in evs if e.claim_id == claim.claim_id and _stance_of(e) in ("supports", "refutes")]
    if not rel:
        return False
    return all((e.source_grade or "").upper() in {"D", "E"} for e in rel)


def _generate_follow_up_queries(claim: Claim, evs: list[Evidence]) -> list[str]:
    """针对缺失的信息定向生成第二轮查询语句（最多 3 条，不额外调用 API）。"""
    queries: list[str] = []
    claim_text_lower = (claim.text or "").lower()

    # 1. 身份类：加入机构名+官方介绍+主流媒体人物资料
    if claim.needs_background_verification or claim.claim_role == "context":
        for ent, aliases in (claim.entity_aliases or {}).items():
            base = " ".join([ent] + list(aliases or [])).strip()
            if base:
                queries.append(f"{base} 官方 个人介绍")
                queries.append(f"{base} 权威媒体 人物资料")
        if claim.search_keywords:
            queries.extend([q for q in claim.search_keywords if q not in queries])
    else:
        if claim.search_keywords:
            queries.extend(list(claim.search_keywords))

    # 2. 高价值结果原文正文回溯：在搜索语句后加"正文"或原始来源
    if not _entity_disambiguation_done(claim, evs):
        for ent, aliases in (claim.entity_aliases or {}).items():
            extra = " ".join([ent] + list(aliases or [])).strip() + " 原始公告 官方名单"
            if extra and extra not in queries:
                queries.append(extra)

    # 3. 针对时间细节：追加年份关键词
    if claim.time_reference and claim.claim_role in ("causal_or_detail", "core"):
        time_ref = str(claim.time_reference)
        # 从已有 search_keywords 中每条补一个年份版本
        for k in claim.search_keywords or [claim.text]:
            if time_ref not in k:
                with_time = f"{k} {time_ref}"
                if with_time not in queries:
                    queries.append(with_time)

    # 4. 转载追溯原始来源（如果目前都是低质量）
    if _evidence_from_primary_sources_only_fan_media(claim, evs):
        if claim.search_keywords:
            base = claim.search_keywords[0]
            extra = f"{base} 原始报道 主流媒体 核实"
            if extra not in queries:
                queries.append(extra)

    # 去重、截断
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        if not q or q in seen:
            continue
        seen.add(q)
        result.append(q)
        if len(result) >= MAX_QUERIES_PER_ROUND:
            break
    return result


def assess_claim_sufficiency(claim: Claim, evs: list[Evidence], round_so_far: int) -> ClaimSufficiency:
    reasons: list[str] = []
    sufficient = True

    if not _has_related_evidence(claim.claim_id, evs):
        reasons.append("未找到相关证据")
        sufficient = False

    if _high_quality_count(claim.claim_id, evs) < 1:
        reasons.append("未找到一手或高质量来源（A/B 级）")
        sufficient = False

    if _count_independent_sources(claim.claim_id, evs) < 2:
        reasons.append("独立来源不足 2 个（存在同一爆料被重复转载）")
        sufficient = False

    if claim.needs_background_verification and not _entity_disambiguation_done(claim, evs):
        reasons.append("人物/机构身份未完成消歧（需要含机构上下文的结果）")
        sufficient = False

    if _time_conflict_found(claim, evs):
        reasons.append("存在公开时间与主张时间冲突")
        sufficient = False

    if _evidence_from_primary_sources_only_fan_media(claim, evs):
        reasons.append("当前结果仅来自粉丝、自媒体或匿名转载")
        sufficient = False

    queries = []
    if not sufficient and round_so_far < MAX_ROUNDS_PER_CLAIM:
        queries = _generate_follow_up_queries(claim, evs)

    return ClaimSufficiency(
        claim_id=claim.claim_id,
        sufficient=sufficient,
        reasons=reasons,
        round=min(round_so_far + 1 if not sufficient else round_so_far, MAX_ROUNDS_PER_CLAIM),
        follow_up_queries=queries,
    )


def assess_sufficiency(state: AgentState, search_evidence_fn=None) -> AgentState:
    """在 state.claims 上执行证据充分性评估，将结果写入 state.metadata["sufficiency"]。

    Args:
        search_evidence_fn: 可选，用于测试中计数/断言的回调；真实执行时由 workflow 显式调用 search_evidence。

    写入：
      state.metadata["follow_up"] = {claim_id: {"queries": [...], "round": 2}}
    """
    from src.workflow import _PHASE_SEARCH as SEARCH_PHASE  # local import 防循环
    _ = SEARCH_PHASE
    _ = search_evidence_fn  # 仅用于测试（兼容旧版单测传参调用）

    claims = state.claims or []
    evs = state.evidence_pool or []

    # 确定每条主张已经执行过的检索轮次（最多记 1 轮在第一轮搜索之后）
    rounds = state.metadata.setdefault("_search_rounds", {})

    per_claim: dict[str, ClaimSufficiency] = {}
    follow_up: dict[str, dict] = {}
    need_second_round = False

    for c in claims:
        r_so_far = int(rounds.get(c.claim_id, 1))
        cs = assess_claim_sufficiency(c, evs, r_so_far)
        per_claim[c.claim_id] = cs
        if not cs.sufficient and cs.follow_up_queries and r_so_far < MAX_ROUNDS_PER_CLAIM:
            follow_up[c.claim_id] = {
                "queries": cs.follow_up_queries,
                "round": 2,
                "reasons": cs.reasons,
            }
            need_second_round = True

    if need_second_round:
        # 关键日志：显式说明启动第 2 轮补充检索（符合用户要求，不静默）
        state.log(
            step="sufficiency",
            action="因证据不足，已启动第2轮补充检索",
            status="running",
            details={
                "claims_needing_round2": list(follow_up.keys()),
                "per_claim_reasons": {k: v["reasons"] for k, v in follow_up.items()},
            },
        )
    else:
        state.log(
            step="sufficiency",
            action="证据充分性评估完成，无需补充检索",
            status="success",
            details={"claims_total": len(claims)},
        )

    state.metadata["sufficiency"] = {
        "per_claim": {k: v.__dict__ for k, v in per_claim.items()},
    }
    state.metadata["follow_up"] = follow_up
    state.metadata["_search_rounds"] = rounds  # 第二轮实际执行后由 search 节点递增
    return state


def has_background_context(claim: Claim, evs: list[Evidence]) -> bool:
    """背景身份主张：需要 entity_aliases 或证据中包含至少一个"别名+所属机构"的匹配。

    说明：本函数当前实现不依赖 evs（保留 evs 参数兼容历史签名），主要看 claim.entity_aliases。
    """
    _ = evs
    aliases = claim.entity_aliases or {}
    # aliases 是 dict[str, list[str]]（或老版 list[str]）
    any_good = False
    if isinstance(aliases, dict):
        for ent, vs in aliases.items():
            # 只要某个实体有 ≥1 个机构上下文别名，就算"背景身份上下文已提供"
            if ent and isinstance(vs, list) and any(v for v in vs if v):
                any_good = True
                break
    elif isinstance(aliases, list):
        any_good = any(isinstance(x, str) and x.strip() for x in aliases)
    return any_good


__all__ = [
    "MAX_ROUNDS_PER_CLAIM",
    "MAX_QUERIES_PER_ROUND",
    "ClaimSufficiency",
    "assess_claim_sufficiency",
    "assess_sufficiency",
    "has_background_context",
]
