"""专业新闻核查 Agent：有边界的决策流程。

固定流程：
接收主张 → 查询历史核查记忆 → Tavily 首次搜索 → Agent 评估证据是否充分
→ 选择"结束核查"或"改写关键词并补搜一次" → 形成最终结论 → 保存核查记忆

限制：
- Tavily 最多调用 2 次
- LLM 最多调用 2 次（决策 + 最终判断）
- 只允许一次补搜
- 总时间不超过 75 秒
- 不允许无限循环
- 外部服务失败时必须正常降级结束
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from src.config import settings
from src.llm.client import LLMClient, LLMError
from src.memory_store import MemoryStore
from src.models import (
    AgentDecision,
    AgentState,
    CheckResult,
    CHECK_STATUS_SUCCESS,
    CHECK_STATUS_PARTIAL,
    CHECK_STATUS_UNAVAILABLE,
    Claim,
    ClaimResult,
    Evidence,
    FactCheckReport,
    TimelineEvent,
    build_failure_report,
    build_no_evidence_report,
    create_claim_result,
)
from src.search_cache import SearchCache
from src.tools.search_tool import (
    SearchResult,
    TavilySearchProvider,
    _validate_url,
)
from src.claim_decomposer import (
    decompose_claims_llm,
    claims_to_claim_objects,
)
from src.session_cache import get_session_cache, SessionCache


# 专业判定标准（只允许这 6 种结论）
PROFESSIONAL_VERDICTS = "基本属实／部分属实／存在错误／已证伪／证据不足／暂无法核查"

# 全局缓存实例
_cache: Optional[SearchCache] = None


def _get_cache() -> SearchCache:
    """获取全局缓存实例。"""
    global _cache
    if _cache is None:
        _cache = SearchCache()
    return _cache


# ===== 智能查询词提取 =====

# 时间模式：4位年份，可选"年"后缀
import re
_YEAR_RE = re.compile(r"((?:19|20)\d{2})年?")

# 中文数字转换
_CN_NUM_MAP = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def _extract_query_components(claim_text: str) -> dict:
    """从主张中提取核心组件：年份、主体、事件、属性、数值。

    返回 dict:
        year: 年份字符串（如 "2024"）
        subjects: 主体关键词列表
        events: 事件关键词列表
        attributes: 待验证属性关键词列表
        values: 目标数值列表
    """
    text = claim_text.strip()

    # 提取年份
    year_match = _YEAR_RE.search(text)
    year = year_match.group(1) if year_match else ""

    # 提取数值（阿拉伯数字 + 中文数字）
    values = []
    # 阿拉伯数字
    for m in re.finditer(r'\d+(?:\.\d+)?', text):
        val = m.group(0)
        if val != year:
            values.append(val)
    # 中文数字（如 "40枚"）
    for m in re.finditer(r'[零一二三四五六七八九十百]+(?:枚|个|次|人|项|张|座|艘|辆|套|种|类|台|部|篇|幅|例|起|宗|项)?', text):
        val = m.group(0)
        if len(val) >= 2:  # 至少两位才有效
            values.append(val)

    # 提取主体关键词（人物/组织/国家）
    subjects = []
    subject_patterns = [
        r'[\u4e00-\u9fff]{2,6}代表团',  # XX代表团
        r'[\u4e00-\u9fff]{2,6}队',  # XX队
        r'[\u4e00-\u9fff]{2,4}政府',  # XX政府
        r'[\u4e00-\u9fff]{2,6}明星',  # XX明星
        r'[\u4e00-\u9fff]{2,6}公司',  # XX公司
        r'[\u4e00-\u9fff]{2,6}机构',  # XX机构
        r'[\u4e00-\u9fff]{2,4}官方',  # XX官方
    ]
    for pat in subject_patterns:
        for m in re.finditer(pat, text):
            subjects.append(m.group(0))

    # 提取事件关键词
    events = []
    event_patterns = [
        r'[\u4e00-\u9fff]{2,6}(?:奥运会|世锦赛|世界杯|全运会|亚运会|青奥会)',
        r'[\u4e00-\u9fff]{2,6}(?:发布会|听证会|庭审|审判|峰会|论坛)',
        r'[\u4e00-\u9fff]{2,6}(?:签署|签署了|发布|宣布|声明|通告|通报)',
        r'[\u4e00-\u9fff]{2,6}(?:发生|出现|爆发|引发|导致)',
    ]
    for pat in event_patterns:
        for m in re.finditer(pat, text):
            events.append(m.group(0))

    # 提取属性关键词（待验证的内容）
    attributes = []
    attr_patterns = [
        r'[\u4e00-\u9fff]{0,4}(?:金牌|银牌|铜牌|奖牌|名次|排名|成绩|票数|支持率|收视率|票房)',
        r'[\u4e00-\u9fff]{0,4}(?:获得|取得|拿下|夺得|赢得|取得了|收获)',
        r'[\u4e00-\u9fff]{0,4}(?:宣布|官宣|公布|披露|透露)',
        r'[\u4e00-\u9fff]{0,4}(?:上涨|下跌|增长|下降|飙升|暴跌|突破|达到|超过|降至)',
        r'[\u4e00-\u9fff]{0,4}(?:停运|关闭|取消|推迟|延期|终止)',
    ]
    for pat in attr_patterns:
        for m in re.finditer(pat, text):
            attributes.append(m.group(0))

    # 去重
    subjects = list(dict.fromkeys(subjects))
    events = list(dict.fromkeys(events))
    attributes = list(dict.fromkeys(attributes))
    values = list(dict.fromkeys(values))

    return {
        "year": year,
        "subjects": subjects,
        "events": events,
        "attributes": attributes,
        "values": values,
    }


def _build_search_query(claim_text: str, supplementary: bool = False) -> str:
    """构建完整检索词，确保不会只用年份或单个宽泛词搜索。

    规则：
    1. 必须包含主体 + 事件/属性 + 数值（如有）
    2. 禁止只使用年份、人名或单个宽泛词
    3. 如果无法提取足够组件，回退到截断文本（至少20字）
    4. supplementary=True 时使用更精确的限定词
    """
    comp = _extract_query_components(claim_text)

    parts: list[str] = []

    # 年份 + 事件
    if comp["year"] and comp["events"]:
        parts.append(f'{comp["year"]}{comp["events"][0]}')
    elif comp["year"]:
        parts.append(comp["year"])

    # 主体
    if comp["subjects"]:
        parts.append(comp["subjects"][0])

    # 属性 + 数值
    if comp["attributes"]:
        attr = comp["attributes"][0]
        val_str = " ".join(comp["values"]) if comp["values"] else ""
        if val_str:
            parts.append(f'{attr} {val_str}'.strip())
        else:
            parts.append(attr)

    # 如果提取到了有效组件，用空格连接
    if len(parts) >= 2:
        query = " ".join(parts)
        # 追加 "官方" 提升权威性
        if not any(kw in query for kw in ["官方", "正式", "官方数据", "官方统计"]):
            query += " 官方"
        # 补充搜索时更精确
        if supplementary:
            if comp["year"] and comp["events"]:
                query += f' site:olympic.org site:chinabasketball.org'
        return query

    # 回退：确保最少20字符的查询，避免只有 "2024" 这样的短查询
    fallback = claim_text[:80].strip()
    if len(fallback) < 20:
        fallback = claim_text
    return fallback


# ===== 搜索结果相关性过滤 =====

# 来源等级评分：A=4, B=3, C=2, D=1
_SOURCE_GRADE_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}

# 必须匹配的关键词组（按优先级）
_RELEVANCE_CRITERIA = {
    "event": [  # 必须匹配事件关键词
        "奥运会", "世锦赛", "世界杯", "全运会", "亚运会", "青奥会",
        "发布会", "听证会", "庭审", "审判", "峰会", "论坛",
        "签署", "发布", "宣布", "声明", "通告", "通报",
        "发生", "出现", "爆发", "引发", "导致",
    ],
    "attribute": [  # 必须匹配属性关键词
        "金牌", "银牌", "铜牌", "奖牌", "名次", "排名", "成绩",
        "获得", "取得", "拿下", "夺得", "赢得",
        "宣布", "官宣", "公布", "披露", "透露",
        "上涨", "下跌", "增长", "下降", "飙升", "暴跌", "突破",
        "停运", "关闭", "取消", "推迟", "延期",
    ],
}


def _score_result_relevance(result: dict, claim_text: str) -> float:
    """计算单条搜索结果与主张的相关性评分（0.0-1.0）。

    评分维度（必须先满足2个核心要素，才能获得权威来源加分）：
    - 主体匹配：0.25
    - 事件/对象匹配：0.20
    - 行为/待验证属性匹配：0.20
    - 关键数值匹配：0.20
    - 时间/年份匹配：0.05
    - 权威来源加分：0.10（需≥2个核心要素匹配）

    硬性否决：如果网页只与主张共享年份，但没有覆盖主体、事件、行为/属性、对象/数值中的任何核心要素，直接返回 0.0。
    """
    title = result.get("title", "")
    content = result.get("content", "")
    publisher = result.get("publisher", "")

    comp = _extract_query_components(claim_text)

    # 组合标题和内容用于关键词匹配
    combined_text = title + " " + content

    # --- 核心要素匹配 ---
    core_match_count = 0

    # 1. 主体匹配 (0.25)
    subject_score = 0.0
    if comp["subjects"]:
        for subj in comp["subjects"]:
            if subj in combined_text:
                subject_score = 0.25
                core_match_count += 1
                break

    # 2. 事件/对象匹配 (0.20)
    event_score = 0.0
    if comp["events"]:
        for evt in comp["events"]:
            if evt in combined_text:
                event_score = 0.20
                core_match_count += 1
                break

    # 3. 行为/待验证属性匹配 (0.20)
    attr_score = 0.0
    if comp["attributes"]:
        for attr in comp["attributes"]:
            if attr in combined_text:
                attr_score = 0.20
                core_match_count += 1
                break

    # 4. 关键数值匹配 (0.20)
    value_score = 0.0
    if comp["values"]:
        for val in comp["values"]:
            if val in combined_text:
                value_score = 0.20
                core_match_count += 1
                break

    # --- 次要维度 ---
    # 5. 时间/年份匹配 (0.05)
    time_score = 0.0
    if comp["year"]:
        if comp["year"] in combined_text:
            time_score = 0.05

    # 6. 权威来源加分 (0.10) —— 需要至少2个核心要素匹配
    source_score = 0.0
    if core_match_count >= 2:
        source_grade = _classify_source_grade(publisher)
        if source_grade == "A":
            source_score = 0.10
        elif source_grade == "B":
            source_score = 0.08
        elif source_grade == "C":
            source_score = 0.04
        # D 级不给额外加分

    total = subject_score + event_score + attr_score + value_score + time_score + source_score

    # --- 硬性否决：只有年份匹配，没有任何核心要素匹配 ---
    if comp["year"] and core_match_count == 0:
        has_year_match = comp["year"] in combined_text
        if has_year_match:
            return 0.0

    return min(total, 1.0)


def _filter_relevant_results(
    results: list[dict],
    claim_text: str,
    min_score: float = 0.60,
) -> tuple[list[dict], int, int]:
    """过滤搜索结果，只保留与主张相关的证据。

    规则：
    - 仅保留 relevance_score >= 0.60 的网页
    - 没有合格网页时允许有效证据为 0 条
    - 不凑数：低分结果不会被重新放回
    - 按相关性分数降序排列

    返回: (filtered_results, total_count, low_score_count)
    """
    if not results:
        return [], 0, 0

    total_count = len(results)
    low_score_count = 0
    scored: list[tuple[dict, float]] = []
    for r in results:
        score = _score_result_relevance(r, claim_text)
        if score >= min_score:
            r_copy = dict(r)
            r_copy["_relevance_score"] = round(score, 3)
            scored.append((r_copy, score))
        else:
            low_score_count += 1

    # 按相关性分数降序排列
    scored.sort(key=lambda x: x[1], reverse=True)

    filtered = [item[0] for item in scored]
    return filtered, total_count, low_score_count


# ===== 机械清理（仅删除明显无效，不做相关性判断） =====

def _mechanically_clean_results(results: list[dict]) -> tuple[list[dict], int]:
    """对 Tavily 返回的 5 条结果做三项机械清理：
    1. URL 为空的删除
    2. 标题和摘要同时为空的删除
    3. 完全相同 URL 去重

    不根据关键词、年份或固定分数删除结果。
    """
    if not results:
        return [], 0

    seen_urls: set[str] = set()
    cleaned: list[dict] = []
    removed = 0

    for r in results:
        url = r.get("url", "").strip()
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()

        # 规则1: URL 为空
        if not url:
            removed += 1
            continue

        # 规则2: 标题和摘要同时为空
        if not title and not content:
            removed += 1
            continue

        # 规则3: URL 去重
        norm_url = url.rstrip("/").lower()
        if norm_url in seen_urls:
            removed += 1
            continue
        seen_urls.add(norm_url)

        cleaned.append(r)

    return cleaned, removed


# ===== LLM 相关性判断 =====

LLM_RELEVANCE_PROMPT = """你是事实核查的相关性判断专家。请判断以下网页是否与主张相关。

主张：{claim}

网页列表（共 {count} 条）：
{webpages}

对每条网页输出 JSON 格式判断：
{{
  "judgments": [
    {{
      "source_index": 0,
      "relevant": true或false,
      "stance": "support"或"refute"或"context",
      "reason": "一句话理由"
    }}
  ]
}}

判断规则：
- relevant=true: 网页直接讨论主张中的主体和事件，即使不包含具体数字或结论
- stance=support: 网页内容支持主张
- stance=refute: 网页内容反驳主张
- stance=context: 网页提供背景信息但不直接支持或反驳
- 只要网页讨论的是主张涉及的主体和事件，就应标记为 relevant=true
- 不因网页缺少某个关键词就判定为不相关
- 新闻页面、官方网站、百科条目只要讨论了相关事件就算相关"""


def _llm_judge_relevance(
    claim_text: str,
    cleaned_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """使用 LLM 判断哪些搜索结果与主张相关。

    返回 (relevant_results, all_judgments)：
    - relevant_results: LLM 判定 relevant=true 的结果列表（带 stance/reason）
    - all_judgments: 所有判断结果的完整列表
    """
    if not cleaned_results:
        return [], []

    # 构建网页列表文本
    webpage_lines = []
    for i, r in enumerate(cleaned_results):
        title = r.get("title", "")[:100]
        url = r.get("url", "")[:150]
        content = r.get("content", "")[:200]
        publisher = r.get("publisher", "")
        webpage_lines.append(
            f"[{i}] 标题：{title}\n"
            f"    来源：{publisher}\n"
            f"    URL：{url}\n"
            f"    摘要：{content}"
        )
    webpage_text = "\n".join(webpage_lines)

    prompt = LLM_RELEVANCE_PROMPT.format(
        claim=claim_text[:200],
        count=len(cleaned_results),
        webpages=webpage_text,
    )

    from pydantic import BaseModel, Field
    from typing import List as TypingList

    class RelevanceJudgment(BaseModel):
        source_index: int = Field(description="网页索引")
        relevant: bool = Field(description="是否相关")
        stance: str = Field(description="support/refute/context")
        reason: str = Field(description="判断理由")

    class RelevanceResult(BaseModel):
        judgments: TypingList[RelevanceJudgment] = Field(default_factory=list)

    try:
        client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 20))
        result = client.chat_json(
            system_prompt="你是严谨的相关性判断专家，只输出 JSON。",
            user_prompt=prompt,
            output_model=RelevanceResult,
            temperature=0.1,
        )
    except LLMError:
        # LLM 失败时保守处理：保留所有结果作为 context
        all_judgments = [
            RelevanceJudgment(
                source_index=i,
                relevant=True,
                stance="context",
                reason="LLM 判断超时，默认保留",
            )
            for i in range(len(cleaned_results))
        ]
    else:
        all_judgments = result.judgments

    # 将判断结果映射回原始结果
    relevant_results: list[dict] = []
    judgments_by_index: dict[int, dict] = {}
    for j in all_judgments:
        judgments_by_index[j.source_index] = {
            "relevant": j.relevant,
            "stance": j.stance,
            "reason": j.reason,
        }

    for i, r in enumerate(cleaned_results):
        judgment = judgments_by_index.get(i, {"relevant": True, "stance": "context", "reason": "默认保留"})
        r_copy = dict(r)
        r_copy["_llm_relevant"] = judgment["relevant"]
        r_copy["_llm_stance"] = judgment["stance"]
        r_copy["_llm_reason"] = judgment["reason"]
        if judgment["relevant"]:
            relevant_results.append(r_copy)

    return relevant_results, [dict(j.model_dump()) for j in all_judgments]


# ===== LLM 生成补充搜索查询 =====

LLM_SUPPLEMENT_PROMPT = """你是搜索查询生成专家。首次搜索结果中没有找到与主张直接相关的来源，请生成一个更精确的补充搜索查询。

主张：{claim}

要求：
- 必须保留主张中的核心主体、事件和待核查属性
- 使用中文搜索词，不超过 20 个词
- 不要使用 site: 限制（会过滤掉有效来源）
- 返回一个查询字符串即可"""


def _llm_generate_supplement_query(claim_text: str) -> str:
    """让 LLM 根据主张生成补充搜索查询。"""
    from pydantic import BaseModel, Field

    class SupplementQuery(BaseModel):
        query: str = Field(description="补充搜索查询词")

    prompt = LLM_SUPPLEMENT_PROMPT.format(claim=claim_text[:200])

    try:
        client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 15))
        result = client.chat_json(
            system_prompt="你是搜索查询生成专家，只输出 JSON。",
            user_prompt=prompt,
            output_model=SupplementQuery,
            temperature=0.1,
        )
        query = result.query.strip()
        # 确保查询不为空且不是单个词
        if query and len(query.split()) >= 2:
            return query
    except LLMError:
        pass

    # 回退：使用原始主张作为补充查询
    return claim_text.strip()


def _search_once(query: str, use_cache: bool = True) -> tuple[list[dict], str | None, str]:
    """单次 Tavily 搜索，带缓存支持。

    返回 (results, error, cache_info)：
    - results: 搜索结果列表
    - error: 错误信息，None 表示成功
    - cache_info: 缓存信息，如 "hit_24h", "fallback_72h", "live", "none"
    """
    cache = _get_cache()

    # 1. 检查 24h 直接命中
    if use_cache:
        hit_entry = cache.get_hit(query)
        if hit_entry:
            results = SearchCache.entry_to_results(hit_entry)
            dicts = _results_to_dicts(results)
            if dicts:
                print(f"[CACHE] 24h 命中缓存，共 {len(dicts)} 条结果")
                return dicts, None, "hit_24h"

    # 2. 执行实时搜索
    provider = TavilySearchProvider(timeout=settings.TAVILY_TIMEOUT)
    search_results, response_time, err = provider.search(
        query=query,
        max_results=5,
        topic="general",
    )

    if err:
        # 3. 实时搜索失败，尝试 72h 兜底缓存
        if use_cache:
            fallback_entry = cache.get_fallback(query)
            if fallback_entry:
                results = SearchCache.entry_to_results(fallback_entry)
                dicts = _results_to_dicts(results)
                if dicts:
                    age = cache.age_hours(fallback_entry)
                    print(f"[CACHE] 实时搜索失败，使用 {age:.1f}h 前的兜底缓存，共 {len(dicts)} 条结果")
                    return dicts, None, "fallback_72h"
        # 无缓存可用，返回错误
        return [], err, "none"

    # 4. 实时搜索成功，写入缓存
    if search_results and use_cache:
        try:
            cache.save(query, search_results)
            print(f"[CACHE] 搜索成功，已写入缓存，共 {len(search_results)} 条结果")
        except Exception as e:
            print(f"[CACHE] 写入缓存失败: {e}")

    dicts = _results_to_dicts(search_results)
    return dicts, None, "live"


def _results_to_dicts(results: list[SearchResult]) -> list[dict]:
    """将 SearchResult 列表转为 dict 列表。"""
    out: list[dict] = []
    for r in results:
        url = r.url.strip()
        if not _validate_url(url):
            continue
        out.append({
            "title": r.title.strip(),
            "url": url,
            "content": (r.content or "")[:500],
            "publisher": r.publisher or "",
            "score": r.score,
        })
    return out


def _pick_top3(results: list[dict]) -> list[dict]:
    """从搜索结果中去重并保留最相关 3 条。"""
    sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in sorted_results:
        url = r["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(r)
        if len(unique) >= 3:
            break
    return unique


def _classify_source_grade(publisher: str, source_type: str = "", url: str = "") -> str:
    """专业评价来源等级：A/B/C/D。

    A级：官方机构、当事人正式声明、原始文件、国际组织
    B级：正规媒体原创报道、专业机构
    C级：百科、聚合媒体、普通转载
    D级：论坛、自媒体、匿名爆料
    """
    publisher_lower = publisher.lower()
    source_type_lower = source_type.lower()
    url_lower = url.lower()

    # 从 URL 域名判断
    from urllib.parse import urlparse
    domain = ""
    if url:
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            pass

    # A 级判断（官方/国际组织/官方媒体）
    a_keywords = [
        "官方", "gov", "政府", "公安", "法院", "检察院", "正式声明", "公告", "通报",
        "奥委会", "olympic", "国际", "委员会", "联合会", "协会",
    ]
    a_domains = [
        "gov.cn", "gov.mo", "gov.hk",  # 中国政府
        "xinhuanet.com", "news.cn", "people.com.cn", "cctv.com",  # 央媒
        "chinadaily.com.cn", "cri.cn",  # 官方媒体
        "stats.gov.cn", "pbc.gov.cn", "csrc.gov.cn",  # 官方机构
    ]
    for kw in a_keywords:
        if kw.lower() in publisher_lower or kw.lower() in source_type_lower:
            return "A"
    for domain_key in a_domains:
        if domain_key in domain:
            return "A"

    # B 级判断（权威媒体）
    b_keywords = [
        "新华社", "人民日报", "央视", "财经", "路透", "bloomberg", "nature", "science",
        "bbc", "cnn", "nyt", "reuters", "ap",
    ]
    b_domains = [
        "sina.com.cn", "sohu.com", "163.com", "qq.com",  # 门户网站
        "thepaper.cn", "jiemian.com", "caixin.com", "yicai.com",  # 财经媒体
        "zaobao.com", "scmp.com",  # 境外权威中文媒体
        "36kr.com", "huxiu.com", "ifeng.com",  # 科技/商业媒体
        "newsweek.com", "time.com", "wsj.com", "ft.com",  # 国际媒体
    ]
    for kw in b_keywords:
        if kw.lower() in publisher_lower:
            return "B"
    for domain_key in b_domains:
        if domain_key in domain:
            return "B"

    # D 级判断（自媒体/低质量）
    d_keywords = ["微博", "微信", "抖音", "快手", "bbs", "论坛", "博客", "自媒体", "小红书", "youtube", "twitter"]
    d_domains = [
        "weibo.com", "weixin.qq.com", "douyin.com", "kuaishou.com",
        "zhihu.com", "bilibili.com", "zhihu.com",  # 问答/视频
    ]
    for kw in d_keywords:
        if kw.lower() in publisher_lower:
            return "D"
    for domain_key in d_domains:
        if domain_key in domain:
            return "D"

    # 默认 C 级（百科/聚合）
    return "C"


def _build_evidence_list(top3: list[dict], search_query: str = "") -> list[Evidence]:
    """将 Top3 搜索结果转为 Evidence 对象，带专业来源评价。"""
    now = datetime.now()
    evidences: list[Evidence] = []

    # 检测独立来源分组（转载相同内容的归为一组）
    from collections import defaultdict
    domain_groups: dict[str, list[str]] = defaultdict(list)
    for r in top3:
        from urllib.parse import urlparse
        domain = urlparse(r["url"]).netloc
        domain_groups[domain].append(r["url"])

    # 简单的转载检测：如果多个来源的标题高度相似，归为一组
    title_groups: dict[str, list[str]] = defaultdict(list)
    for r in top3:
        title_key = r["title"][:20]  # 取前20字符作为分组键
        title_groups[title_key].append(r["url"])

    for i, r in enumerate(top3, start=1):
        publisher = r.get("publisher", "") or "未知来源"
        url = r["url"]
        source_grade = _classify_source_grade(publisher, "", url)
        is_independent = True
        for group_urls in title_groups.values():
            if url in group_urls and len(group_urls) > 1:
                is_independent = False
                break

        # 检查是否直接支持主张（初始假设，后续由 LLM 评估）
        directly_supports = False  # 需要 LLM 后续判断

        evidences.append(Evidence(
            evidence_id=f"E{i}",
            claim_id="main",
            source_title=r["title"],
            source_url=url,
            publisher=publisher,
            published_at=now,
            retrieved_at=now,
            evidence_summary=r.get("content", "")[:200],
            summary="",  # 留空，后续由 LLM 填充
            source_type="搜索来源",
            source_grade=source_grade,
            supports_or_refutes="unclear",
            is_primary_source=(source_grade == "A"),
            reliability_reason=f"{source_grade}级来源：{publisher}",
            directly_supports=directly_supports,
            is_independent=is_independent,
            evidence_key_points=[],
            search_query=search_query,
        ))
    return evidences


# ========== Agent 决策 Prompt ==========
AGENT_DECISION_PROMPT = """你是一个严谨的事实核查 Agent。请根据当前搜索证据，评估是否足够形成结论。

用户主张：{claim}

当前证据（共 {evidence_count} 条）：
{evidence_text}

请进行结构化评估，输出 JSON：
{{
  "normalized_claim": "标准化后的主张（20字以内）",
  "claim_type": "事实/观点/预测/私人传闻",
  "sensitivity": "低/中/高",
  "evidence_requirement": "形成结论需要什么证据（一句话）",
  "evidence_sufficient": true或false,
  "missing_evidence": ["缺少的证据类型1", "缺少的证据类型2"],
  "action": "STOP" 或 "SEARCH_AGAIN",
  "supplemental_query": "补搜关键词（仅当action=SEARCH_AGAIN时）",
  "action_reason": "一句话解释为什么停止或补搜"
}}

判断原则：
- 如果证据来自 A 级来源（官方、原始文件）且直接支持主张，通常足够
- 如果只有 C/D 级来源或证据矛盾，可能需要补搜
- 对于私人传闻、敏感信息，需要 A 级来源才能判定
- 不要因为证据不充分就编造结论，选择 SEARCH_AGAIN 时必须有具体的补搜方向"""


# ========== 最终判断 Prompt ==========
FINAL_JUDGE_PROMPT = """你是一个事实核查助手。请根据以下证据，对主张做出专业判定。

用户主张：{claim}

搜索证据：
{evidence_text}

请从以下 6 个结论中选择一个最恰当的：
- 基本属实：有 A/B 级来源支持，核心内容正确
- 部分属实：部分内容有证据，部分未证实
- 存在错误：有可靠来源指出具体错误
- 已证伪：权威来源明确否认，或可靠事实直接矛盾
- 证据不足：没有足够证据支持或反驳（不等于已证伪）
- 暂无法核查：信息过于模糊或缺少可验证线索

严格规则：
- "没有证据支持"只能判为"证据不足"，不能判为"已证伪"
- "已证伪"必须有 A 级来源明确否认
- 私人关系、怀孕、违法、健康等敏感信息，没有 A 级来源时默认"证据不足"

同时输出：
1. 为每条证据写一句不超过 50 字的概括 —— 优先使用"原文摘录"中的内容，不要编造
2. 标注每条证据是否直接支持主张
3. 提取与主张直接相关的事件时间（最多 4 个）
4. 评估传播风险等级和原因

重要：evidence_summaries 中的摘要必须基于"原文摘录"字段，不要添加原文没有的信息！

输出 JSON：
{{
  "verdict": "六选一",
  "reason": "核心理由（50字以内）",
  "cited_evidence": ["E1", "E2"],
  "evidence_summaries": {{"E1": "基于原文摘录的一句话概括", "E2": "基于原文摘录的一句话概括"}},
  "evidence_support": {{"E1": true, "E2": false}},
  "timeline": [{{"date": "YYYY-MM-DD", "event": "与主张相关的事件"}}],
  "risk_level": "高/中/低/不确定",
  "risk_reason": "风险原因",
  "risk_factors": ["因素1"]
}}"""


def _agent_decide(claim_text: str, evidences: list[Evidence]) -> AgentDecision:
    """Agent 决策：评估证据是否充分，决定 STOP 或 SEARCH_AGAIN。"""
    if not evidences:
        return AgentDecision(
            normalized_claim=claim_text[:20],
            claim_type="事实",
            sensitivity="中",
            evidence_requirement="需要至少一个相关来源",
            evidence_sufficient=False,
            missing_evidence=["公开来源"],
            action="SEARCH_AGAIN",
            supplemental_query=claim_text[:20],
            action_reason="未检索到任何来源，需要补充搜索",
        )

    evidence_lines = []
    for ev in evidences:
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}.get(ev.source_grade, "")
        evidence_lines.append(
            f"[{ev.evidence_id}] ({grade_desc}级) {ev.source_title}\n"
            f"    摘要：{ev.evidence_summary[:100]}\n"
            f"    发布者：{ev.publisher}"
        )
    evidence_text = "\n".join(evidence_lines)

    prompt = AGENT_DECISION_PROMPT.format(
        claim=claim_text[:100],
        evidence_count=len(evidences),
        evidence_text=evidence_text,
    )

    client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 25))

    from pydantic import BaseModel, Field
    from typing import List, Optional as Opt

    class DecisionResult(BaseModel):
        normalized_claim: str = Field(default="", description="标准化主张")
        claim_type: str = Field(default="事实", description="主张类型")
        sensitivity: str = Field(default="中", description="敏感度")
        evidence_requirement: str = Field(default="", description="所需证据")
        evidence_sufficient: bool = Field(default=False, description="证据是否充分")
        missing_evidence: List[str] = Field(default_factory=list, description="缺少的证据")
        action: str = Field(default="STOP", description="STOP 或 SEARCH_AGAIN")
        supplemental_query: Opt[str] = Field(default=None, description="补搜关键词")
        action_reason: str = Field(default="", description="行动理由")

    try:
        result = client.chat_json(
            system_prompt="你是严谨的事实核查 Agent，只输出 JSON。",
            user_prompt=prompt,
            output_model=DecisionResult,
            temperature=0.1,
        )
        return AgentDecision(
            normalized_claim=result.normalized_claim or claim_text[:20],
            claim_type=result.claim_type if result.claim_type in ("事实", "观点", "预测", "私人传闻") else "事实",
            sensitivity=result.sensitivity if result.sensitivity in ("低", "中", "高") else "中",
            evidence_requirement=result.evidence_requirement or "需要更多证据",
            evidence_sufficient=result.evidence_sufficient,
            missing_evidence=result.missing_evidence or [],
            action=result.action if result.action in ("STOP", "SEARCH_AGAIN") else "STOP",
            supplemental_query=result.supplemental_query,
            action_reason=result.action_reason or "证据已足够",
        )
    except LLMError:
        # LLM 决策失败，保守地选择 STOP（不补搜）
        return AgentDecision(
            normalized_claim=claim_text[:20],
            claim_type="事实",
            sensitivity="中",
            evidence_requirement="需要至少一个相关来源",
            evidence_sufficient=len(evidences) > 0,
            missing_evidence=[],
            action="STOP",
            action_reason="决策超时，基于现有证据判断",
        )


def _final_judge(claim_text: str, evidences: list[Evidence]) -> dict:
    """最终判断：基于所有证据做出专业判定。

    优先使用原文摘录作为证据摘要，避免 LLM 幻觉。
    """
    if not evidences:
        return {
            "verdict": "证据不足",
            "reason": "未检索到有效来源",
            "cited_evidence": [],
            "evidence_summaries": {},
            "evidence_support": {},
            "timeline": [],
            "risk_level": "不确定",
            "risk_reason": "",
            "risk_factors": [],
        }

    # 构建证据文本，强调使用原文摘录
    evidence_lines = []
    # 预先生成原文摘录的证据摘要（优先使用）
    excerpt_summaries: Dict[str, str] = {}
    for ev in evidences:
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}.get(ev.source_grade, "")

        # 使用原文摘录作为摘要（如果有的话）
        if ev.evidence_summary and len(ev.evidence_summary.strip()) > 10:
            excerpt = ev.evidence_summary[:100].strip()
            # 清理摘录文本
            excerpt = ' '.join(excerpt.split())
            excerpt_summaries[ev.evidence_id] = excerpt
        else:
            excerpt_summaries[ev.evidence_id] = ev.source_title

        evidence_lines.append(
            f"[{ev.evidence_id}] ({grade_desc}级) {ev.source_title}\n"
            f"    原文摘录：{excerpt_summaries[ev.evidence_id]}\n"
            f"    链接：{ev.source_url}"
        )
    evidence_text = "\n".join(evidence_lines)

    prompt = FINAL_JUDGE_PROMPT.format(
        claim=claim_text[:100],
        evidence_text=evidence_text,
    )

    client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 30))

    from pydantic import BaseModel, Field
    from typing import List, Dict

    class TimelineItem(BaseModel):
        date: str = Field(default="", description="日期")
        event: str = Field(default="", description="事件描述")

    class FinalResult(BaseModel):
        verdict: str = Field(description="基本属实/部分属实/存在错误/已证伪/证据不足/暂无法核查")
        reason: str = Field(description="核心理由")
        cited_evidence: List[str] = Field(default_factory=list)
        evidence_summaries: Dict[str, str] = Field(default_factory=dict)
        evidence_support: Dict[str, bool] = Field(default_factory=dict)
        timeline: List[TimelineItem] = Field(default_factory=list)
        risk_level: str = Field(default="不确定")
        risk_reason: str = Field(default="")
        risk_factors: List[str] = Field(default_factory=list)

    try:
        result = client.chat_json(
            system_prompt="你是严谨的事实核查助手，只输出 JSON。",
            user_prompt=prompt,
            output_model=FinalResult,
            temperature=0.2,
        )
        # 优先使用原文摘录作为证据摘要，避免 LLM 幻觉
        # 如果 LLM 生成的摘要为空或不可靠，使用原文摘录
        final_evidence_summaries = {}
        for eid, llm_summary in result.evidence_summaries.items():
            # 检查 LLM 生成的摘要是否合理（长度在10-100字之间）
            if llm_summary and 10 <= len(llm_summary) <= 100:
                final_evidence_summaries[eid] = llm_summary
            # 否则使用原文摘录
            elif eid in excerpt_summaries:
                final_evidence_summaries[eid] = excerpt_summaries[eid]

        # 确保所有证据都有摘要
        for eid, excerpt in excerpt_summaries.items():
            if eid not in final_evidence_summaries:
                final_evidence_summaries[eid] = excerpt

        return {
            "verdict": result.verdict,
            "reason": result.reason,
            "cited_evidence": result.cited_evidence,
            "evidence_summaries": final_evidence_summaries,  # 使用混合策略的摘要
            "evidence_support": result.evidence_support,
            "timeline": [item.model_dump() for item in result.timeline],
            "risk_level": result.risk_level,
            "risk_reason": result.risk_reason,
            "risk_factors": result.risk_factors,
        }
    except LLMError:
        return {
            "verdict": "证据不足",
            "reason": "模型判断超时或失败",
            "cited_evidence": [],
            "evidence_summaries": {},
            "evidence_support": {},
            "timeline": [],
            "risk_level": "不确定",
            "risk_reason": "",
            "risk_factors": [],
        }


def _build_key_evidence_cards(evidences: list[Evidence], evidence_support: dict) -> list[dict]:
    """构建关键证据卡片。"""
    cards = []
    for ev in evidences:
        support = evidence_support.get(ev.evidence_id, False)
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}.get(ev.source_grade, "")
        cards.append({
            "id": ev.evidence_id,
            "title": ev.source_title,
            "url": ev.source_url,
            "publisher": ev.publisher,
            "grade": ev.source_grade,
            "grade_desc": grade_desc,
            "summary": ev.summary or ev.evidence_summary[:50],
            "directly_supports": support,
            "is_independent": ev.is_independent,
        })
    return cards


def run_professional_fact_check(original_text: str) -> FactCheckReport:
    """有边界的专业新闻核查 Agent。

    固定流程：
    接收主张 → 查询历史记忆 → Tavily 搜索 → Agent 评估 → 决定补搜或结束 → 最终判断 → 保存

    限制：
    - 总时间不超过 75 秒
    - Tavily 最多 2 次，LLM 最多 2 次
    - 只允许一次补搜
    """
    start_time = time.monotonic()
    deadline = start_time + 75

    # ===== 初始化会话级短期缓存 =====
    session_cache = get_session_cache()
    print(f"[SESSION_CACHE] 初始化，统计: {session_cache.get_stats()}")

    # ===== Step 1: 接收主张 =====
    state = AgentState(original_text=original_text, mode="full")
    state.mark_step_started("receive", "接收主张")
    state.mark_step_completed("receive", "主张已接收")

    # 记录决策轨迹
    decision_trace: list[dict] = []
    tool_calls_count = 0
    did_supplemental_search = False

    # ===== Step 1.5: 主张拆分（新功能）=====
    state.mark_step_started("decompose", "主张拆分")
    decomposed_claims_data = decompose_claims_llm(original_text, max_claims=5)
    claims_list = claims_to_claim_objects(decomposed_claims_data)

    print(f"[DECOMPOSE] 拆分出 {len(claims_list)} 条主张:")
    for i, c in enumerate(claims_list):
        print(f"  {i+1}. [{c.claim_type}] {c.text[:50]}...")

    decision_trace.append({
        "step": "主张拆分",
        "action": "LLM 智能拆分",
        "result": f"从原文中拆分出 {len(claims_list)} 条独立事实主张",
        "claims": [{"text": c.text[:60], "type": c.claim_type, "priority": c.verification_priority} for c in claims_list],
    })
    state.mark_step_completed("decompose", f"拆分完成：{len(claims_list)}条主张")

    # ===== Step 2: 查询历史核查记忆 =====
    memory = MemoryStore()
    historical_matches = []
    try:
        historical_matches = memory.search_similar(original_text)
        if historical_matches:
            print(f"[MEMORY] 发现 {len(historical_matches)} 条历史核查记录")
    except Exception as e:
        print(f"[MEMORY] 查询历史记忆失败: {e}")

    state.mark_step_started("memory", "查询历史记忆")
    if historical_matches:
        decision_trace.append({
            "step": "查询历史记忆",
            "action": "检索相似历史案例",
            "result": f"发现 {len(historical_matches)} 条历史记录",
        })
    else:
        decision_trace.append({
            "step": "查询历史记忆",
            "action": "检索相似历史案例",
            "result": "无匹配历史记录",
        })
    state.mark_step_completed("memory", "历史记忆查询完成")

    # 使用拆分后的主张列表
    state.claims = claims_list
    main_claim = claims_list[0] if claims_list else Claim(
        claim_id="main",
        text=original_text[:200],
        claim_type="事件陈述",
        entities=[],
        verification_priority=1,
    )

    # ===== 多主张循环核查 =====
    if len(claims_list) > 1:
        print(f"[MULTI_CLAIM] 检测到 {len(claims_list)} 条主张，开始循环核查")
        state.mark_step_started("search", "多主张循环核查")

        all_claim_results = []
        total_evidences = []
        max_claims_to_check = min(len(claims_list), 5)  # 最多核查5条

        for i, claim in enumerate(claims_list[:max_claims_to_check]):
            if time.monotonic() > deadline - 15:
                print(f"[MULTI_CLAIM] 时间不足，停止核查剩余主张")
                # 剩余主张标记为证据不足
                for remaining in claims_list[i:]:
                    result = create_claim_result(
                        claim=remaining,
                        verdict="证据不足",
                        confidence=0.2,
                        reasoning="时间不足，未完成核查",
                        evidence=[],
                        missing_information="需要更多时间核查",
                    )
                    all_claim_results.append(result)
                break

            print(f"[MULTI_CLAIM] 核查第 {i+1}/{max_claims_to_check} 条主张: {claim.text[:40]}...")
            claim_result, trace_adds, tool_adds, did_search = _verify_claim_single(
                claim=claim,
                deadline=deadline,
                session_cache=session_cache,
                memory=memory,
                decision_trace=decision_trace,
                tool_calls_count=tool_calls_count,
            )
            tool_calls_count += tool_adds
            decision_trace.extend(trace_adds)
            all_claim_results.append(claim_result)
            total_evidences.extend(claim_result.evidence)

            print(f"[MULTI_CLAIM] 主张{i+1}核查完成: {claim_result.verdict}, 置信度: {claim_result.confidence}")

        # 汇总所有主张结果
        state.mark_step_completed("search", f"多主张核查完成，共处理 {len(all_claim_results)} 条")

        # 确定总体结论
        verdicts = [r.verdict for r in all_claim_results]
        if all(v in ("基本属实", "部分属实") for v in verdicts):
            final_verdict = "基本属实"
            reason = "所有主张均有证据支持"
        elif any(v == "已证伪" for v in verdicts):
            final_verdict = "存在错误"
            reason = "部分主张被证伪"
        elif any(v in ("存在错误", "部分属实") for v in verdicts):
            final_verdict = "部分属实"
            reason = "部分主张存在问题"
        else:
            final_verdict = "证据不足"
            reason = "多数主张缺乏证据"

        # 构建报告
        all_evidence_support = {}
        for r in all_claim_results:
            for ev in r.evidence:
                all_evidence_support[ev.evidence_id] = ev.directly_supports or False

        key_cards = _build_key_evidence_cards(total_evidences, all_evidence_support)

        # 风险评估
        risk_factors = []
        if any(v in ("已证伪", "存在错误") for v in verdicts):
            risk_level = "高"
            risk_reason = "包含已证伪或存在错误的信息"
            risk_factors.append("部分内容不实")
        elif any(v == "证据不足" for v in verdicts):
            risk_level = "中"
            risk_reason = "部分内容缺乏证据支持"
            risk_factors.append("部分内容存疑")
        else:
            risk_level = "低"
            risk_reason = "所有主张均有可靠证据"

        propagation_risk = f"{risk_level}风险：{risk_reason}"

        # 计算总体可信度
        confidences = [r.confidence for r in all_claim_results]
        credibility = int(sum(confidences) / len(confidences) * 100) if confidences else 20

        recommendation = "可谨慎参考" if risk_level in ("中", "高") else "可放心引用"

        state.mark_step_started("output", "生成报告")
        state.mark_step_completed("output", "报告已生成")

        report = FactCheckReport(
            original_text=original_text,
            overall_verdict=final_verdict,
            overall_summary=reason,
            claim_results=all_claim_results,
            timeline=[],
            propagation_risk=propagation_risk,
            risk_level=risk_level,
            risk_reason=risk_reason,
            risk_factors=risk_factors,
            unresolved_questions=["部分主张需要更多证据验证"] if risk_level != "低" else [],
            execution_log=state.execution_log,
            decision_trace=decision_trace,
            agent_decision=None,
            did_supplemental_search=True,
            tool_calls_count=tool_calls_count,
            historical_matches=[],
            key_evidence_cards=key_cards,
            credibility_score=credibility,
            recommendation=recommendation,
            current_step="completed",
            completed_steps=["receive", "decompose", "memory", "search", "analyze", "output"],
            skipped_steps=[],
            progress_percent=100,
            workflow_completed=True,
            workflow_error=None,
            generated_at=datetime.now(),
        )

        try:
            memory.save_report(report)
        except Exception as e:
            print(f"[MEMORY] 保存核查记忆失败: {e}")

        return report

    # ===== 单主张模式：继续原有流程 =====
    # main_claim 已在上面赋值

    # ===== Step 3: Tavily 首次搜索 =====
    if time.monotonic() > deadline:
        return build_failure_report(original_text, "搜索前已超时", "timeout")

    state.mark_step_started("search", "首次搜索")
    # 使用第一条主张作为主要搜索查询
    first_search_query = main_claim.text[:100].strip()
    print(f"[WORKFLOW] 构建检索词（主主张）: {first_search_query}")

    # 检查会话缓存
    cached_search = session_cache.get_search_results(first_search_query)
    if cached_search:
        print(f"[SESSION_CACHE] 命中搜索缓存，跳过搜索调用")
        search_results = cached_search
        search_err = None
        cache_info = "session_cache_hit"
        decision_trace.append({
            "step": "首次搜索",
            "action": "会话缓存命中",
            "query": first_search_query,
            "result": f"使用缓存的{len(search_results)}条结果",
        })
    else:
        search_results, search_err, cache_info = _search_once(first_search_query)
        tool_calls_count += 1
        # 缓存搜索结果
        if not search_err and search_results:
            session_cache.set_search_results(first_search_query, search_results)
            print(f"[SESSION_CACHE] 缓存搜索结果: {len(search_results)}条")
    search_elapsed = time.monotonic() - start_time

    if search_err:
        state.mark_step_completed("search", "首次搜索",
                                  details={"status": "no_evidence", "error": search_err})
        decision_trace.append({
            "step": "首次搜索",
            "query": first_search_query,
            "result": f"搜索失败：{search_err}",
            "cache": cache_info,
        })
        print(f"[WORKFLOW] 搜索失败且无缓存可用: {search_err}")
        report = build_no_evidence_report(original_text, reason="搜索服务暂时不可用")
        report.decision_trace = decision_trace
        report.tool_calls_count = tool_calls_count
        return report

    # ===== 机械清理 + LLM 相关性判断 =====
    cleaned_results, removed_count = _mechanically_clean_results(search_results)
    relevant_results, all_judgments = _llm_judge_relevance(main_claim.text, cleaned_results)
    print(f"[WORKFLOW] 搜索{len(search_results)}条 → 机械清理后{len(cleaned_results)}条 → LLM判定相关{len(relevant_results)}条")

    cache_detail = {
        "results": len(search_results),
        "cleaned_results": len(cleaned_results),
        "relevant_results": len(relevant_results),
        "cache": cache_info,
    }
    if cache_info == "hit_24h":
        cache_detail["note"] = "使用 24 小时内的缓存结果"
    elif cache_info == "fallback_72h":
        cache_detail["note"] = "搜索服务暂不可用，使用 72 小时内的缓存结果"

    state.mark_step_completed("search", "首次搜索", details=cache_detail)

    decision_trace.append({
        "step": "首次搜索",
        "query": first_search_query,
        "result": f"搜索{len(search_results)}条，机械清理{len(cleaned_results)}条，LLM判定相关{len(relevant_results)}条",
        "cache": cache_info,
    })

    if not relevant_results:
        decision_trace.append({
            "step": "证据评估",
            "result": "LLM判定无相关来源，触发补充搜索",
        })
        state.mark_step_started("analyze", "分析证据")
        evidences: list[Evidence] = []
        supplement_query = _llm_generate_supplement_query(main_claim.text)
        agent_decision = AgentDecision(
            normalized_claim=main_claim.text[:20],
            claim_type="事实",
            sensitivity="中",
            evidence_requirement="需要至少一个相关来源",
            evidence_sufficient=False,
            missing_evidence=["公开来源"],
            action="SEARCH_AGAIN",
            supplemental_query=supplement_query,
            action_reason="首次搜索无相关来源，LLM生成补充查询",
        )
    else:
        # ===== Step 4: 保留 LLM 判定相关的 Top3 证据并构建证据列表 =====
        top3 = _pick_top3(relevant_results)
        evidences = _build_evidence_list(top3, first_search_query)

        # 缓存证据列表到会话缓存
        if evidences:
            session_cache.set_cached_evidences(main_claim.text, evidences)
            print(f"[SESSION_CACHE] 缓存 {len(evidences)} 条证据")

        # ===== Step 5: Agent 评估证据是否充分 =====
        if time.monotonic() > deadline:
            return build_failure_report(original_text, "Agent 评估前已超时", "timeout")

        state.mark_step_started("plan", "Agent 评估证据充分性")
        agent_decision = _agent_decide(main_claim.text[:200], evidences)
        tool_calls_count += 1

        decision_trace.append({
            "step": "Agent 证据评估",
            "claim_type": agent_decision.claim_type,
            "sensitivity": agent_decision.sensitivity,
            "evidence_sufficient": agent_decision.evidence_sufficient,
            "action": agent_decision.action,
            "reason": agent_decision.action_reason,
        })

        state.mark_step_completed("plan", "Agent 评估完成")

    # ===== Step 6: 决定是否补搜（自动触发或 Agent 决策） =====
    if (agent_decision.action == "SEARCH_AGAIN"
            and agent_decision.supplemental_query
            and not did_supplemental_search
            and time.monotonic() + 20 <= deadline):

        # 执行补搜
        did_supplemental_search = True
        state.mark_step_started("search", "补充搜索")

        supplement_query = agent_decision.supplemental_query
        print(f"[WORKFLOW] 决定补搜: {supplement_query}")

        supplement_results, supplement_err, supplement_cache = _search_once(supplement_query)
        tool_calls_count += 1

        if not supplement_err and supplement_results:
            # 机械清理 + LLM 相关性判断补充搜索结果
            cleaned_supplement, _ = _mechanically_clean_results(supplement_results)
            relevant_supplement, _ = _llm_judge_relevance(main_claim.text, cleaned_supplement)
            if relevant_supplement:
                new_top3 = _pick_top3(relevant_supplement)
                new_evidences = _build_evidence_list(new_top3, supplement_query)

                # 合并去重
                existing_urls = {e.source_url for e in evidences}
                for ne in new_evidences:
                    if ne.source_url not in existing_urls:
                        evidences.append(ne)
                        existing_urls.add(ne.source_url)

                decision_trace.append({
                    "step": "补充搜索",
                    "query": supplement_query,
                    "result": f"搜索{len(supplement_results)}条，机械清理{len(cleaned_supplement)}条，LLM判定相关{len(relevant_supplement)}条",
                })
            else:
                decision_trace.append({
                    "step": "补充搜索",
                    "query": supplement_query,
                    "result": f"搜索{len(supplement_results)}条，LLM判定无相关结果",
                })
        else:
            decision_trace.append({
                "step": "补充搜索",
                "query": supplement_query,
                "result": supplement_err or "无新结果",
            })

        state.mark_step_completed("search", "补充搜索完成")
    else:
        decision_trace.append({
            "step": "补充搜索决策",
            "action": agent_decision.action,
            "reason": agent_decision.action_reason,
        })

    # ===== Step 7: 最终判断 =====
    if time.monotonic() > deadline:
        return build_failure_report(original_text, "最终判断前已超时", "timeout")

    state.mark_step_started("analyze", "最终判断")

    judge_result = _final_judge(main_claim.text[:200], evidences)
    tool_calls_count += 1

    state.mark_step_completed("analyze", "判断完成")

    # 缓存判断结果到会话缓存
    session_cache.set_cached_verdict(main_claim.text, judge_result)
    print(f"[SESSION_CACHE] 缓存判断结果")

    verdict = judge_result.get("verdict", "证据不足")
    reason = judge_result.get("reason", "判断失败")
    evidence_summaries = judge_result.get("evidence_summaries", {})
    evidence_support = judge_result.get("evidence_support", {})
    timeline_data = judge_result.get("timeline", [])
    risk_level = judge_result.get("risk_level", "不确定")
    risk_reason = judge_result.get("risk_reason", "")
    risk_factors = judge_result.get("risk_factors", [])

    # 填充证据的一句话概括和支持判断
    for ev in evidences:
        if ev.evidence_id in evidence_summaries:
            ev.summary = evidence_summaries[ev.evidence_id]
        if ev.evidence_id in evidence_support:
            ev.directly_supports = evidence_support[ev.evidence_id]

    # 构建时间线
    report_timeline: list[TimelineEvent] = []
    for item in timeline_data:
        date_str = item.get("date", "")
        event_desc = item.get("event", "")
        event_time = None
        if date_str:
            try:
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"]:
                    try:
                        event_time = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        if not event_time and date_str and event_desc:
            event_desc = f"{date_str}：{event_desc}"
        if event_desc:
            report_timeline.append(TimelineEvent(
                event_time=event_time,
                description=event_desc,
                source_url=evidences[0].source_url if evidences else None,
            ))

    dated = [e for e in report_timeline if e.event_time]
    undated = [e for e in report_timeline if not e.event_time]
    dated.sort(key=lambda e: e.event_time)
    report_timeline = dated + undated
    report_timeline = report_timeline[:4]

    if not report_timeline:
        report_timeline.append(TimelineEvent(
            event_time=None,
            description="公开来源未提供可靠时间线信息",
        ))

    # 专业 verdict 映射
    verdict_map = {
        "基本属实": "基本属实",
        "部分属实": "部分属实",
        "存在错误": "存在错误",
        "已证伪": "已证伪",
        "证据不足": "证据不足",
        "暂无法核查": "暂无法核查",
    }
    final_verdict = verdict_map.get(verdict, "证据不足")

    # 可信度评估 (0-100 整数) - 改进版
    credibility = 20  # 基础分
    a_count = sum(1 for e in evidences if e.source_grade == "A")
    b_count = sum(1 for e in evidences if e.source_grade == "B")
    c_count = sum(1 for e in evidences if e.source_grade == "C")
    total_evidence = len(evidences)

    # 基础分：根据最高等级
    if a_count > 0:
        credibility = 75  # 有A级来源
    elif b_count > 0:
        credibility = 60  # 有B级来源
    elif c_count > 0:
        credibility = 40  # 只有C级来源
    else:
        credibility = 20  # 无有效证据

    # 数量加成：证据越多越可信（最多加15分）
    if total_evidence >= 5:
        credibility += 15
    elif total_evidence >= 3:
        credibility += 10
    elif total_evidence >= 1:
        credibility += 5

    # 正反对比加成：有支持也有反驳，说明经过充分验证
    support_count = sum(1 for e in evidences if e.supports_or_refutes == "supports")
    refute_count = sum(1 for e in evidences if e.supports_or_refutes == "refutes")
    if support_count > 0 and refute_count > 0:
        credibility += 5  # 有正反两方证据

    # 补充搜索加成
    if did_supplemental_search:
        credibility += 5

    # 限制在 0-100
    credibility = min(100, max(0, credibility))

    # 传播建议
    if final_verdict in ("证据不足", "暂无法核查") or credibility < 40:
        recommendation = "不建议继续传播"
    elif final_verdict == "已证伪" or risk_level == "高":
        recommendation = "不建议继续传播"
    else:
        recommendation = "可谨慎参考"

    # 构建关键证据卡片
    key_cards = _build_key_evidence_cards(evidences, evidence_support)

    # 构建 ClaimResult（通过工厂函数，确保字段契约一致）
    missing_info = None if final_verdict not in ("证据不足", "暂无法核查") else "需要更多独立来源验证"
    claim_result = create_claim_result(
        claim=main_claim,
        verdict=final_verdict,
        confidence=credibility / 100.0,  # 0-100 转为 0.0-1.0
        reasoning=reason,
        evidence=evidences,
        missing_information=missing_info,
    )
    state.claim_results = [claim_result]

    # 未解决的问题
    uq_raw = claim_result.missing_information
    if uq_raw and isinstance(uq_raw, str):
        uq = [uq_raw]
    else:
        uq = []

    # 传播风险描述
    if risk_reason:
        propagation_risk = f"{risk_level}风险：{risk_reason}"
    else:
        propagation_risk = f"{risk_level}风险"

    # 生成报告
    state.mark_step_started("output", "生成报告")
    state.mark_step_completed("output", "报告已生成")

    report = FactCheckReport(
        original_text=original_text,
        overall_verdict=final_verdict,
        overall_summary=reason,
        claim_results=[claim_result],
        timeline=report_timeline,
        propagation_risk=propagation_risk,
        risk_level=risk_level,
        risk_reason=risk_reason,
        risk_factors=risk_factors,
        unresolved_questions=uq,
        execution_log=state.execution_log,
        decision_trace=decision_trace,
        agent_decision=agent_decision,
        did_supplemental_search=did_supplemental_search,
        tool_calls_count=tool_calls_count,
        historical_matches=historical_matches,  # 填充历史匹配
        key_evidence_cards=key_cards,
        credibility_score=credibility,
        recommendation=recommendation,
        current_step="completed",
        completed_steps=["receive", "decompose", "memory", "search", "plan", "analyze", "output"],
        skipped_steps=[],
        progress_percent=100,
        workflow_completed=True,
        workflow_error=None,
        generated_at=datetime.now(),
    )

    # ===== Step 8: 保存核查记忆（异步，不阻塞返回）=====
    try:
        memory.save_report(report)
    except Exception as e:
        print(f"[MEMORY] 保存核查记忆失败: {e}")

    return report


# 兼容旧接口
def run_quick_fact_check(original_text: str) -> FactCheckReport:
    """兼容旧接口，内部调用新的专业核查流程。"""
    return run_professional_fact_check(original_text)


# ===== 统一入口：所有核查请求必须通过此函数 =====
def run_fact_check(original_text: str) -> CheckResult:
    """统一核查入口，返回 CheckResult 结构。

    确保无论外部 API 是否成功，都返回统一结构：
    - status: success / partial / unavailable
    - report: FactCheckReport（永远有值）
    - error_message: 中文友好消息（不含 Python traceback）

    验收标准：无论外部 API 是否成功，都不抛异常到页面层。
    """
    try:
        report = run_professional_fact_check(original_text)

        # 根据报告状态确定 status
        if report.current_step == "failed" or report.workflow_error:
            return CheckResult(
                status=CHECK_STATUS_UNAVAILABLE,
                report=report,
                error_message=_friendly_status_message(report),
            )
        elif report.overall_verdict == "暂无法核查":
            return CheckResult(
                status=CHECK_STATUS_PARTIAL,
                report=report,
                error_message="搜索服务暂时不可用，部分核查未能完成",
            )
        else:
            return CheckResult(
                status=CHECK_STATUS_SUCCESS,
                report=report,
                error_message="",
            )
    except Exception as e:
        print(f"[FACT_CHECK] 未预期异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

        # 兜底：任何异常都生成失败报告，绝不抛到页面层
        report = build_failure_report(
            original_text,
            "核查服务暂时不可用，请稍后重试",
            "init",
        )
        return CheckResult(
            status=CHECK_STATUS_UNAVAILABLE,
            report=report,
            error_message="核查服务暂时不可用，请稍后重试",
        )


# ============================================================================
# ReAct 循环实现 —— Agent 自主思考-行动-观察
# ============================================================================

# ---- ReAct 状态 ----

class ReActState:
    """ReAct 循环运行时状态。"""

    def __init__(self, original_text: str, deadline: float):
        self.original_text: str = original_text
        self.deadline: float = deadline
        self.iteration: int = 0
        self.max_iterations: int = 5
        self.search_calls: int = 0
        self.max_search_calls: int = 3
        self.evidences: list[Evidence] = []
        self.decision_log: list[dict] = []
        self.tool_calls: int = 0
        self.done: bool = False
        self.final_report: Optional[FactCheckReport] = None

    @property
    def remaining_time(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def can_continue(self) -> bool:
        if self.iteration >= self.max_iterations:
            return False
        if self.remaining_time < 15:
            return False
        if self.done:
            return False
        return True


# ---- ReAct 系统 Prompt ----

REACT_SYSTEM_PROMPT = """你是一个自主的事实核查 Agent (溯真)。你的任务是核查用户的主张是否属实。

你必须使用 Thought + Action + Observation 的循环方式工作：
1. Thought：分析当前掌握的信息，判断还需要什么
2. Action：选择一个可用工具并执行
3. Observation：阅读工具返回的结果
4. 重复以上步骤，直到你认为证据充分，然后调用 finalize 结束

你可用的工具：
- search_web(query, max_results=5)：搜索网页。适用于首次查找或补充搜索。
- evaluate_evidence(claim, evidence_ids)：评估当前证据是否足够形成结论。
- finalize(verdict, reason, cited_evidence_ids, evidence_summaries, risk_level, risk_reason)：结束核查并输出最终报告。

决策原则：
- 先用 search_web 找相关网页，再用 evaluate_evidence 判断是否足够。
- 如果证据不足，换一个关键词再搜索，但搜索次数不超过3次。
- 对冲突信息要交叉验证，优先 A 级来源（官方、原始文件）。
- 不要编造信息，没有证据就诚实地说"证据不足"。
- 你最多可以进行 5 轮思考-行动循环。

重要：你必须输出严格的 JSON 格式，不要包含任何解释、前言或 Markdown 代码围栏。
JSON 格式如下：
{
  "thought": "你这一轮的思考：当前有什么信息？缺什么？",
  "action": "search_web 或 evaluate_evidence 或 finalize",
  "action_input": {
    "query": "搜索关键词（仅 search_web 需要）",
    "evidence_ids": ["证据ID列表（仅 evaluate_evidence 需要）"],
    "verdict": "最终结论（仅 finalize 需要）",
    "reason": "核心理由（仅 finalize 需要）",
    "cited_evidence_ids": ["引用的证据ID（仅 finalize 需要）"],
    "evidence_summaries": {"证据ID": "一句话概括（仅 finalize 需要）"},
    "risk_level": "风险等级（仅 finalize 需要）",
    "risk_reason": "风险原因（仅 finalize 需要）"
  }
}"""

REACT_USER_PROMPT_TEMPLATE = """## 主张
{claim}

## 第 {iteration} 轮 思考

## 当前证据（{evidence_count} 条）：
{evidence_text}

## 历史行动：
{history_text}

## 可用工具：
1. search_web(query, max_results=5) — 搜索网页
2. evaluate_evidence(claim, evidence_ids) — 评估证据充分性
3. finalize(verdict, reason, cited_evidence_ids, evidence_summaries, risk_level, risk_reason) — 结束核查

请输出严格的 JSON 格式，包含 thought、action 和 action_input 字段。不要输出任何其他内容。"""

# ---- ReAct LLM 决策 Schema ----

from pydantic import BaseModel, Field
from typing import List


class ReActDecision(BaseModel):
    """ReAct 每轮的 LLM 决策。"""
    thought: str = Field(description="你这一轮的思考：当前有什么信息？缺什么？")
    action: str = Field(description="行动：search_web / evaluate_evidence / finalize")
    action_input: dict = Field(description="行动参数")


class ReActFinalize(BaseModel):
    """finalize 行动的参数。"""
    verdict: str = Field(
        description="最终结论：基本属实/部分属实/存在错误/已证伪/证据不足/暂无法核查"
    )
    reason: str = Field(description="核心理由，100字以内")
    cited_evidence_ids: List[str] = Field(default_factory=list, description="引用的证据ID列表")
    evidence_summaries: dict = Field(default_factory=dict, description="每条证据的一句话概括")
    risk_level: str = Field(default="不确定", description="风险等级：低/中/高/不确定")
    risk_reason: str = Field(default="", description="风险原因")


class ReActEvaluateResult(BaseModel):
    """evaluate_evidence 行动的 LLM 输出。"""
    sufficient: bool = Field(description="证据是否充分")
    reason: str = Field(description="判断理由")
    missing: List[str] = Field(default_factory=list, description="缺少的证据类型")


# ---- LLM stance → Evidence supports_or_refutes 映射 ----

_LLM_STANCE_MAP = {
    "support": "supports",
    "refute": "refutes",
    "context": "partial",
    "unclear": "unclear",
}

# ---- 工具执行 ----

def _tool_search_web(
    state: ReActState,
    query: str,
    max_results: int = 5,
) -> dict:
    """执行 search_web 工具。"""
    query = query.strip()
    if not query:
        return {"error": "查询词不能为空", "results": []}

    search_results, search_err, cache_info = _search_once(query)

    if search_err:
        return {
            "error": search_err,
            "results": [],
            "cache": cache_info,
            "note": "搜索失败，可能是网络问题或服务不可用",
        }

    cleaned, removed = _mechanically_clean_results(search_results)
    llm_relevant, _ = _llm_judge_relevance(state.original_text, cleaned)

    new_evidences: list[Evidence] = []
    for i, r in enumerate(llm_relevant):
        raw_stance = r.get("_llm_stance", "unclear")
        mapped_stance = _LLM_STANCE_MAP.get(raw_stance, "unclear")
        ev = Evidence(
            evidence_id=f"S{state.search_calls + 1}_{i + 1}",
            claim_id="main",
            source_title=r.get("title", "") or "未知标题",
            source_url=r.get("url", ""),
            publisher=r.get("publisher", "") or "未知来源",
            published_at=datetime.now(),
            retrieved_at=datetime.now(),
            evidence_summary=(r.get("content", "") or "")[:200],
            summary="",
            source_type="搜索来源",
            source_grade=_classify_source_grade(r.get("publisher", "") or ""),
            supports_or_refutes=mapped_stance,
            is_primary_source=False,
            reliability_reason=f"{_classify_source_grade(r.get('publisher', '') or '')}级来源",
            directly_supports=raw_stance == "support",
            is_independent=True,
            evidence_key_points=[],
            search_query=query,
        )
        new_evidences.append(ev)

    existing_urls = {e.source_url for e in state.evidences}
    added = 0
    for ne in new_evidences:
        if ne.source_url not in existing_urls:
            state.evidences.append(ne)
            existing_urls.add(ne.source_url)
            added += 1

    state.search_calls += 1
    state.tool_calls += 1

    return {
        "query": query,
        "total_results": len(search_results),
        "after_cleaning": len(cleaned),
        "llm_relevant": len(llm_relevant),
        "new_evidences_added": added,
        "total_evidences": len(state.evidences),
        "cache": cache_info,
        "evidences": [
            {
                "id": e.evidence_id,
                "title": e.source_title[:60],
                "url": e.source_url,
                "grade": e.source_grade,
                "publisher": e.publisher[:30],
                "summary": e.evidence_summary[:80],
                "stance": e.supports_or_refutes,
            }
            for e in state.evidences
        ],
    }


def _tool_evaluate_evidence(
    state: ReActState,
    evidence_ids: list[str],
) -> dict:
    """执行 evaluate_evidence 工具。"""
    target = [e for e in state.evidences if e.evidence_id in evidence_ids]
    if not target:
        target = state.evidences

    if not target:
        return {
            "sufficient": False,
            "reason": "当前没有任何证据",
            "missing": ["需要至少搜索一次获取相关来源"],
            "suggestion": "调用 search_web 开始搜索",
        }

    evidence_lines = []
    for ev in target:
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}.get(ev.source_grade, "")
        evidence_lines.append(
            f"[{ev.evidence_id}] ({grade_desc}级) {ev.source_title[:60]}\n"
            f"    摘要：{ev.evidence_summary[:100]}\n"
            f"    链接：{ev.source_url}\n"
            f"    立场：{ev.supports_or_refutes}"
        )
    evidence_text = "\n".join(evidence_lines)

    prompt = f"""请评估以下证据是否足以对主张做出判断。

主张：{state.original_text[:200]}

当前证据：
{evidence_text}

请结构化输出评估结果。"""

    try:
        client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 20))
        result = client.chat_json(
            system_prompt="你是严谨的证据评估专家。",
            user_prompt=prompt,
            output_model=ReActEvaluateResult,
            temperature=0.1,
        )
        return {
            "sufficient": result.sufficient,
            "reason": result.reason,
            "missing": result.missing,
            "suggestion": (
                "可以做出结论" if result.sufficient
                else "需要补充搜索，尝试使用不同的关键词"
            ),
            "evidence_ids_evaluated": [e.evidence_id for e in target],
        }
    except LLMError:
        has_a = any(e.source_grade == "A" for e in target)
        has_b = any(e.source_grade == "B" for e in target)
        if has_a or (has_b and len(target) >= 2):
            return {
                "sufficient": True,
                "reason": "有权威来源支持，证据充分",
                "missing": [],
                "suggestion": "可以做出结论",
                "evidence_ids_evaluated": [e.evidence_id for e in target],
            }
        return {
            "sufficient": False,
            "reason": "LLM 评估失败，保守判断为证据不足",
            "missing": ["需要更多独立来源"],
            "suggestion": "尝试补充搜索或降低结论力度",
            "evidence_ids_evaluated": [e.evidence_id for e in target],
        }


def _build_evidence_context(state: ReActState) -> str:
    """构建证据上下文供 LLM 阅读。"""
    if not state.evidences:
        return "（当前无证据）"

    lines = []
    for e in state.evidences:
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}.get(e.source_grade, "")
        stance_desc = {"support": "支持", "refute": "反驳", "context": "背景", "unclear": "立场不明"}.get(
            e.supports_or_refutes, "立场不明"
        )
        lines.append(
            f"- [{e.evidence_id}] ({grade_desc}级, {stance_desc}) "
            f"{e.source_title[:80]}"
        )
        lines.append(f"  摘要：{e.evidence_summary[:100]}")
        lines.append(f"  链接：{e.source_url}")
    return "\n".join(lines)


def _build_history_text(state: ReActState) -> str:
    """构建历史行动记录。"""
    if not state.decision_log:
        return "（首轮，尚无历史行动）"

    lines = []
    for i, log in enumerate(state.decision_log[-4:], start=1):
        lines.append(f"第{log['iteration']}轮：Thought={log.get('thought', '')[:60]}... → Action={log['action']}")
        if "observation" in log:
            obs = log["observation"]
            if isinstance(obs, dict):
                summary_parts = []
                for k in ("query", "total_results", "reason", "sufficient"):
                    if k in obs:
                        summary_parts.append(f"{k}={obs[k]}")
                lines.append(f"  Observation: {', '.join(summary_parts)}")
    return "\n".join(lines)


# ---- 主 ReAct 循环 ----

def _run_react_loop(
    original_text: str,
    deadline: float,
    historical_matches: list[dict],
) -> tuple[ReActState, list[dict]]:
    """运行 ReAct 循环，返回最终状态和决策轨迹。"""

    state = ReActState(original_text, deadline)
    decision_trace: list[dict] = []

    decision_trace.append({
        "step": "ReAct 初始化",
        "claim": original_text[:50],
        "historical_matches": len(historical_matches),
    })

    # 首轮自动搜索
    print(f"[ReAct] 第 0 轮：自动首次搜索")
    initial_search = _tool_search_web(state, original_text.strip(), max_results=5)
    decision_trace.append({
        "step": "首轮搜索",
        "action": "search_web",
        "query": original_text.strip()[:50],
        "observation": f"搜索{initial_search.get('total_results', 0)}条，LLM判定{initial_search.get('llm_relevant', 0)}条相关",
        "evidence_count": len(state.evidences),
    })
    state.iteration = 0

    if not state.evidences:
        # 首次搜索无结果，尝试一次 LLM 生成的补充搜索
        supplement_query = _llm_generate_supplement_query(original_text)
        if state.search_calls < state.max_search_calls and state.remaining_time > 20:
            print(f"[ReAct] 补充搜索（由 LLM 生成）：{supplement_query}")
            supplement_result = _tool_search_web(state, supplement_query, max_results=5)
            decision_trace.append({
                "step": "补充搜索",
                "action": "search_web",
                "query": supplement_query[:50],
                "observation": f"搜索{supplement_result.get('total_results', 0)}条，新增{supplement_result.get('new_evidences_added', 0)}条",
                "evidence_count": len(state.evidences),
            })

    # 检查是否直接可以结束
    if state.evidences and state.remaining_time > 15:
        eval_result = _tool_evaluate_evidence(state, [e.evidence_id for e in state.evidences])
        decision_trace.append({
            "step": "初步评估",
            "action": "evaluate_evidence",
            "observation": f"充分={eval_result['sufficient']}，理由={eval_result.get('reason', '')[:60]}",
        })

    # 进入 ReAct 思考-行动循环（剩余轮次）
    max_react_rounds = state.max_iterations - 1  # 首轮已用
    for round_idx in range(1, max_react_rounds + 1):
        if not state.can_continue():
            print(f"[ReAct] 达到停止条件：iteration={state.iteration}, remaining_time={state.remaining_time:.0f}s")
            break

        if state.remaining_time < 15:
            print(f"[ReAct] 剩余时间不足 {state.remaining_time:.0f}s，提前结束")
            break

        state.iteration = round_idx
        state.tool_calls += 1

        evidence_ctx = _build_evidence_context(state)
        history_ctx = _build_history_text(state)

        user_prompt = REACT_USER_PROMPT_TEMPLATE.format(
            claim=original_text[:200],
            iteration=round_idx,
            evidence_count=len(state.evidences),
            evidence_text=evidence_ctx,
            history_text=history_ctx,
        )

        try:
            client = LLMClient(timeout=min(settings.LLM_TIMEOUT, 20))
            decision = client.chat_json(
                system_prompt=REACT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_model=ReActDecision,
                temperature=0.1,
            )
        except LLMError:
            print(f"[ReAct] 第 {round_idx} 轮 LLM 失败，跳过重试")
            decision_trace.append({
                "step": f"ReAct 第 {round_idx} 轮",
                "action": "LLM_ERROR",
                "observation": "LLM 调用失败",
            })
            # 失败时保守处理：如果有证据就尝试 finalize
            if state.evidences:
                break
            # 无证据则继续下一轮
            continue

        thought = decision.thought
        action = decision.action
        action_input = decision.action_input or {}

        print(f"[ReAct] 第 {round_idx} 轮：Thought={thought[:50]}... → Action={action}")

        # ---- 执行动作 ----
        if action == "search_web":
            query = action_input.get("query", "").strip()
            if state.search_calls >= state.max_search_calls:
                observation = {"error": "搜索次数已达上限", "suggestion": "用现有证据做出判断"}
            else:
                if not query:
                    query = _llm_generate_supplement_query(original_text)
                observation = _tool_search_web(state, query, max_results=5)

        elif action == "evaluate_evidence":
            eids = action_input.get("evidence_ids", [])
            if not eids or eids == "all":
                eids = [e.evidence_id for e in state.evidences]
            if isinstance(eids, str):
                eids = [e.strip() for e in eids.split(",") if e.strip()]
            observation = _tool_evaluate_evidence(state, eids)

        elif action == "finalize":
            observation = {"message": "finalize 行动已确认，准备结束循环"}
            state.decision_log.append({
                "iteration": round_idx,
                "thought": thought,
                "action": action,
                "observation": observation,
            })
            decision_trace.append({
                "step": f"ReAct 第 {round_idx} 轮",
                "action": action,
                "thought": thought,
                "observation": observation.get("message", ""),
            })
            break  # finalize 直接退出循环

        else:
            observation = {"error": f"未知行动: {action}"}

        # ---- 记录本轮 ----
        state.decision_log.append({
            "iteration": round_idx,
            "thought": thought,
            "action": action,
            "observation": observation,
        })

        decision_trace.append({
            "step": f"ReAct 第 {round_idx} 轮",
            "action": action,
            "thought": thought,
            "observation": str(observation)[:200],
            "evidence_count": len(state.evidences),
        })

        # 如果评估结果为充分，且 LLM 没选择 finalize，提示下一轮结束
        if action == "evaluate_evidence" and observation.get("sufficient", False):
            print(f"[ReAct] 评估结果为充分，下一轮建议 finalize")

    return state, decision_trace


# ---- ReAct 最终报告生成 ----

def _build_react_report(
    state: ReActState,
    historical_matches: list[dict],
    decision_trace: list[dict],
) -> FactCheckReport:
    """将 ReAct 最终状态转为 FactCheckReport。"""

    now = datetime.now()
    original_text = state.original_text

    # 判断报告状态
    if state.evidences:
        # 有证据，调用 LLM 做最终判断
        judge_result = _final_judge(original_text[:200], state.evidences)
    else:
        judge_result = {
            "verdict": "证据不足",
            "reason": "未能检索到相关公开来源",
            "cited_evidence": [],
            "evidence_summaries": {},
            "evidence_support": {},
            "timeline": [],
            "risk_level": "不确定",
            "risk_reason": "",
            "risk_factors": [],
        }

    verdict = judge_result.get("verdict", "证据不足")
    reason = judge_result.get("reason", "判断失败")
    evidence_summaries = judge_result.get("evidence_summaries", {})
    evidence_support = judge_result.get("evidence_support", {})
    timeline_data = judge_result.get("timeline", [])
    risk_level = judge_result.get("risk_level", "不确定")
    risk_reason = judge_result.get("risk_reason", "")
    risk_factors = judge_result.get("risk_factors", [])

    # 填充证据详情
    for ev in state.evidences:
        if ev.evidence_id in evidence_summaries:
            ev.summary = evidence_summaries[ev.evidence_id]
        if ev.evidence_id in evidence_support:
            ev.directly_supports = evidence_support[ev.evidence_id]

    # 构建时间线
    report_timeline: list[TimelineEvent] = []
    for item in timeline_data:
        date_str = item.get("date", "")
        event_desc = item.get("event", "")
        event_time = None
        if date_str:
            try:
                for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y年%m月%d日"]:
                    try:
                        event_time = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue
            except Exception:
                pass
        if not event_time and date_str and event_desc:
            event_desc = f"{date_str}：{event_desc}"
        if event_desc:
            report_timeline.append(TimelineEvent(
                event_time=event_time,
                description=event_desc,
                source_url=state.evidences[0].source_url if state.evidences else None,
            ))

    dated = [e for e in report_timeline if e.event_time]
    undated = [e for e in report_timeline if not e.event_time]
    dated.sort(key=lambda e: e.event_time)
    report_timeline = dated + undated
    report_timeline = report_timeline[:4]

    if not report_timeline:
        report_timeline.append(TimelineEvent(
            event_time=None,
            description="公开来源未提供可靠时间线信息",
        ))

    # 可信度 - 改进版
    credibility = 20
    a_count = sum(1 for e in state.evidences if e.source_grade == "A")
    b_count = sum(1 for e in state.evidences if e.source_grade == "B")
    c_count = sum(1 for e in state.evidences if e.source_grade == "C")
    total_evidence = len(state.evidences)

    if a_count > 0:
        credibility = 75
    elif b_count > 0:
        credibility = 60
    elif c_count > 0:
        credibility = 40

    # 数量加成
    if total_evidence >= 5:
        credibility += 15
    elif total_evidence >= 3:
        credibility += 10
    elif total_evidence >= 1:
        credibility += 5

    # 正反对比加成
    support_count = sum(1 for e in state.evidences if e.supports_or_refutes == "supports")
    refute_count = sum(1 for e in state.evidences if e.supports_or_refutes == "refutes")
    if support_count > 0 and refute_count > 0:
        credibility += 5

    credibility = min(100, max(0, credibility))

    # 传播建议
    if verdict in ("证据不足", "暂无法核查") or credibility < 40:
        recommendation = "不建议继续传播"
    elif verdict == "已证伪" or risk_level == "高":
        recommendation = "不建议继续传播"
    else:
        recommendation = "可谨慎参考"

    # 关键证据卡片
    key_cards = _build_key_evidence_cards(state.evidences, evidence_support)

    # ClaimResult
    claim = Claim(
        claim_id="main",
        text=original_text[:200],
        claim_type="事件陈述",
        entities=[],
        verification_priority=1,
    )
    missing_info = None if verdict not in ("证据不足", "暂无法核查") else "需要更多独立来源验证"
    claim_result = create_claim_result(
        claim=claim,
        verdict=verdict,
        confidence=credibility / 100.0,
        reasoning=reason,
        evidence=state.evidences,
        missing_information=missing_info,
    )

    # Action 记录
    final_action = AgentDecision(
        normalized_claim=original_text[:20],
        claim_type="事实",
        sensitivity="中",
        evidence_requirement="基于 Agent 自主判断",
        evidence_sufficient=bool(state.evidences),
        missing_evidence=[],
        action="STOP",
        supplemental_query=None,
        action_reason=f"ReAct 循环 {state.iteration} 轮后结束，共 {len(state.evidences)} 条证据",
    )

    report = FactCheckReport(
        original_text=original_text,
        overall_verdict=verdict,
        overall_summary=reason,
        claim_results=[claim_result],
        timeline=report_timeline,
        propagation_risk=f"{risk_level}风险：{risk_reason}" if risk_reason else f"{risk_level}风险",
        risk_level=risk_level,
        risk_reason=risk_reason,
        risk_factors=risk_factors,
        unresolved_questions=[],
        execution_log=[],
        decision_trace=decision_trace,
        agent_decision=final_action,
        did_supplemental_search=state.search_calls > 1,
        tool_calls_count=state.tool_calls,
        historical_matches=historical_matches,
        key_evidence_cards=key_cards,
        credibility_score=credibility,
        recommendation=recommendation,
        current_step="completed",
        completed_steps=["receive", "memory", "search", "plan", "analyze", "output"],
        skipped_steps=[],
        progress_percent=100,
        workflow_completed=True,
        workflow_error=None,
        generated_at=now,
    )

    # 保存记忆
    try:
        memory = MemoryStore()
        memory.save_report(report)
    except Exception as e:
        print(f"[MEMORY] 保存核查记忆失败: {e}")

    return report


# ---- ReAct 入口 ----

def run_react_fact_check(original_text: str) -> FactCheckReport:
    """基于 ReAct 循环的 Agent 核查入口。

    Agent 自主思考、选择工具、观察结果、最终形成结论。
    最多 5 轮思考-行动循环，最多 3 次搜索调用，总时间 75 秒。
    """
    start_time = time.monotonic()
    deadline = start_time + 75

    # Step 1: 接收主张
    state = AgentState(original_text=original_text, mode="react")
    state.mark_step_started("receive", "接收主张")
    state.mark_step_completed("receive", "主张已接收")

    # Step 2: 查询历史记忆
    memory = MemoryStore()
    historical_matches = []
    try:
        historical_matches = memory.search_similar(original_text)
        if historical_matches:
            print(f"[ReAct] 发现 {len(historical_matches)} 条历史核查记录")
    except Exception as e:
        print(f"[ReAct] 查询历史记忆失败: {e}")

    # Step 3: ReAct 循环
    react_state, decision_trace = _run_react_loop(
        original_text, deadline, historical_matches
    )

    if react_state.remaining_time < 5:
        print(f"[ReAct] 剩余时间不足，快速生成报告")
        if not react_state.evidences:
            report = build_no_evidence_report(original_text, reason="核查时间不足，未能获取足够证据")
        else:
            report = _build_react_report(react_state, historical_matches, decision_trace)
    else:
        report = _build_react_report(react_state, historical_matches, decision_trace)

    return report


# ---- 切换默认入口到 ReAct ----

def run_fact_check(original_text: str) -> CheckResult:
    """统一核查入口（ReAct Agent 模式）。

    使用 ReAct 循环让 Agent 自主决定每一步行动。
    """
    try:
        report = run_react_fact_check(original_text)

        if report.current_step == "failed" or report.workflow_error:
            return CheckResult(
                status=CHECK_STATUS_UNAVAILABLE,
                report=report,
                error_message=_friendly_status_message(report),
            )
        elif report.overall_verdict == "暂无法核查":
            return CheckResult(
                status=CHECK_STATUS_PARTIAL,
                report=report,
                error_message="搜索服务暂时不可用，部分核查未能完成",
            )
        else:
            return CheckResult(
                status=CHECK_STATUS_SUCCESS,
                report=report,
                error_message="",
            )
    except Exception as e:
        import traceback
        error_tb = traceback.format_exc()
        print(f"[FACT_CHECK] 未预期异常: {type(e).__name__}: {e}")
        print(error_tb)

        try:
            with open("_debug_error.log", "w", encoding="utf-8") as f:
                f.write(f"Exception: {type(e).__name__}: {e}\n")
                f.write(error_tb)
        except Exception:
            pass

        report = build_failure_report(
            original_text,
            "核查服务暂时不可用，请稍后重试",
            "init",
        )
        return CheckResult(
            status=CHECK_STATUS_UNAVAILABLE,
            report=report,
            error_message="核查服务暂时不可用，请稍后重试",
        )


def _friendly_status_message(report: FactCheckReport) -> str:
    """根据报告的 workflow_error 生成友好消息。"""
    err = report.workflow_error or ""
    msg_map = {
        "search": "搜索服务连接临时中断",
        "timeout": "核查超过时间上限",
        "analyze": "分析阶段发生异常",
        "init": "核查启动失败",
    }
    return msg_map.get(err, "核查过程发生异常")


# ============================================================================
# 多主张循环核查辅助函数
# ============================================================================

def _verify_claim_single(
    claim: Claim,
    deadline: float,
    session_cache: SessionCache,
    memory: MemoryStore,
    decision_trace: list[dict],
    tool_calls_count: int,
) -> tuple[ClaimResult, list[dict], int, bool]:
    """核查单条主张。

    Args:
        claim: 要核查的主张
        deadline: 截止时间
        session_cache: 会话缓存
        memory: 长期记忆
        decision_trace: 决策轨迹列表
        tool_calls_count: 工具调用计数

    Returns:
        (claim_result, trace_additions, tool_calls_added, did_search)
    """
    trace_additions = []
    did_search = False

    # 检查会话缓存
    cached_evidences = session_cache.get_cached_evidences(claim.text)
    if cached_evidences:
        print(f"[VERIFY] 主张 '{claim.text[:30]}...' 使用缓存的 {len(cached_evidences)} 条证据")
        evidences = cached_evidences
        trace_additions.append({
            "step": f"主张核查: {claim.text[:30]}",
            "action": "使用会话缓存",
            "result": f"复用 {len(evidences)} 条已有证据",
        })
    else:
        # 搜索证据
        if time.monotonic() > deadline - 10:
            # 时间不足，直接返回证据不足
            claim_result = create_claim_result(
                claim=claim,
                verdict="证据不足",
                confidence=0.2,
                reasoning="时间不足，无法完成核查",
                evidence=[],
                missing_information="需要更多时间完成核查",
            )
            return claim_result, trace_additions, tool_calls_count, False

        search_query = claim.text[:80].strip()
        print(f"[VERIFY] 搜索主张: {search_query}")

        search_results, search_err, cache_info = _search_once(search_query)
        tool_calls_count += 1
        did_search = True

        if search_err:
            trace_additions.append({
                "step": f"主张核查: {claim.text[:30]}",
                "action": "搜索",
                "result": f"搜索失败: {search_err}",
            })
            # 搜索失败，返回证据不足
            claim_result = create_claim_result(
                claim=claim,
                verdict="证据不足",
                confidence=0.2,
                reasoning="搜索服务不可用",
                evidence=[],
                missing_information="搜索失败，无法获取证据",
            )
            return claim_result, trace_additions, tool_calls_count, did_search

        # 机械清理 + LLM 相关性判断
        cleaned_results, _ = _mechanically_clean_results(search_results)
        relevant_results, _ = _llm_judge_relevance(claim.text, cleaned_results)

        if relevant_results:
            top3 = _pick_top3(relevant_results)
            evidences = _build_evidence_list(top3, search_query)
            # 缓存证据
            session_cache.set_cached_evidences(claim.text, evidences)
        else:
            evidences = []

        trace_additions.append({
            "step": f"主张核查: {claim.text[:30]}",
            "action": "搜索",
            "query": search_query,
            "result": f"搜索{len(search_results)}条 → 相关{len(relevant_results)}条",
        })

    # 判断
    if evidences:
        judge_result = _final_judge(claim.text[:200], evidences)
        tool_calls_count += 1

        verdict = judge_result.get("verdict", "证据不足")
        reason = judge_result.get("reason", "判断失败")
        evidence_summaries = judge_result.get("evidence_summaries", {})
        evidence_support = judge_result.get("evidence_support", {})

        # 填充证据摘要
        for ev in evidences:
            if ev.evidence_id in evidence_summaries:
                ev.summary = evidence_summaries[ev.evidence_id]
            if ev.evidence_id in evidence_support:
                ev.directly_supports = evidence_support[ev.evidence_id]

        # 计算可信度 - 改进版
        a_count = sum(1 for e in evidences if e.source_grade == "A")
        b_count = sum(1 for e in evidences if e.source_grade == "B")
        c_count = sum(1 for e in evidences if e.source_grade == "C")
        total_evidence = len(evidences)

        if a_count > 0:
            credibility = 0.75
        elif b_count > 0:
            credibility = 0.6
        elif c_count > 0:
            credibility = 0.4
        else:
            credibility = 0.2

        # 数量加成
        if total_evidence >= 5:
            credibility += 0.15
        elif total_evidence >= 3:
            credibility += 0.10
        elif total_evidence >= 1:
            credibility += 0.05

        # 正反对比加成
        support_count = sum(1 for e in evidences if e.supports_or_refutes == "supports")
        refute_count = sum(1 for e in evidences if e.supports_or_refutes == "refutes")
        if support_count > 0 and refute_count > 0:
            credibility += 0.05

        credibility = min(1.0, max(0.0, credibility))

        missing_info = None if verdict not in ("证据不足", "暂无法核查") else "需要更多独立来源验证"

        claim_result = create_claim_result(
            claim=claim,
            verdict=verdict,
            confidence=credibility,
            reasoning=reason,
            evidence=evidences,
            missing_information=missing_info,
        )

        # 缓存判断结果
        session_cache.set_cached_verdict(claim.text, judge_result)

    else:
        # 无证据
        claim_result = create_claim_result(
            claim=claim,
            verdict="证据不足",
            confidence=0.2,
            reasoning="未检索到有效证据",
            evidence=[],
            missing_information="需要更多独立来源验证",
        )

    return claim_result, trace_additions, tool_calls_count, did_search
