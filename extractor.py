"""证据提取工具。

使用大模型从搜索到的网页内容中提取与主张相关的证据。
核心防幻觉机制：
1. 模型必须只根据 source_content 分析，不允许使用自身知识
2. relevant_excerpt 必须能在 source_content 中找到（规范化字符串匹配）
3. 找不到匹配时，extraction_status 设为 invalid_excerpt，该证据不参与最终判断
"""

import re
import unicodedata
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from src.llm.client import LLMClient, LLMError
from src.models import Claim, Evidence, EVIDENCE_STANCE, EVIDENCE_DIRECTNESS
from src.prompts.system_prompts import (
    EXTRACT_EVIDENCE_SYSTEM_PROMPT,
    GRADE_SOURCE_SYSTEM_PROMPT,
)
from src.tools.search_tool import SearchResult


# ============ LLM 输出校验模型 ============


class _ExtractOutput(BaseModel):
    """LLM 证据提取输出的校验模型。"""

    evidence_summary: str = Field(default="")
    relevant_excerpt: Optional[str] = Field(None)
    evidence_stance: EVIDENCE_STANCE = Field(default="context")
    directness: EVIDENCE_DIRECTNESS = Field(default="unclear")
    is_primary_source: bool = Field(default=False)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    source_type_guess: str = Field(default="")


class _GradeOutput(BaseModel):
    """LLM 来源分级输出的校验模型。"""

    source_grade: str = Field(default="D")
    reliability_reason: str = Field(default="")
    source_type: str = Field(default="")


# ============ 防幻觉：原文片段校验 ============


def _normalize_text(text: str) -> str:
    """规范化文本以便进行子串匹配。

    处理：
    - 全角/半角空格统一
    - 连续空白合并为单个空格
    - 去除首尾空白
    - NFKC Unicode 规范化（全角字符转半角等）
    """
    if not text:
        return ""
    # NFKC 规范化：全角→半角，兼容字符→标准形式
    normalized = unicodedata.normalize("NFKC", text)
    # 连续空白合并为单个空格
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def verify_excerpt_in_content(excerpt: str, content: str) -> bool:
    """校验 excerpt 是否能在 content 中找到（规范化匹配）。

    匹配策略：
    1. 直接子串匹配
    2. 规范化后子串匹配
    3. 截取 excerpt 前 50 字符进行匹配（模型可能截断）
    4. 截取 excerpt 中间连续 30 字符进行匹配
    """
    if not excerpt or not content:
        return False

    excerpt_norm = _normalize_text(excerpt)
    content_norm = _normalize_text(content)

    if not excerpt_norm or not content_norm:
        return False

    # 1. 直接子串匹配
    if excerpt_norm in content_norm:
        return True

    # 2. 截取前 50 字符匹配（模型可能截断长片段）
    if len(excerpt_norm) > 50:
        prefix = excerpt_norm[:50]
        if prefix in content_norm:
            return True

    # 3. 截取中间 30 字符匹配
    if len(excerpt_norm) > 30:
        mid_start = (len(excerpt_norm) - 30) // 2
        mid = excerpt_norm[mid_start : mid_start + 30]
        if mid in content_norm:
            return True

    return False


# ============ 证据提取与来源分级 ============


def extract_evidence_from_result(
    client: LLMClient,
    claim: Claim,
    result: SearchResult,
    query: str,
    evidence_id: str,
) -> Evidence:
    """对一条搜索结果使用 LLM 提取证据，并校验原文片段。

    失败时返回 extraction_status 为 failed 或 invalid_excerpt 的证据，
    不会抛出异常，保证流程继续。
    """
    now = result.published_at

    # 构造用户提示，包含 source_content 供模型分析
    content_for_model = (result.content or "")[:8000]  # 截断避免超长
    if not content_for_model.strip():
        return _build_failed_evidence(
            claim, result, query, evidence_id,
            status="insufficient_content",
            reason="网页内容为空",
        )

    user_prompt = (
        f"【需要核查的主张】\n{claim.text}\n\n"
        f"【主张类型】\n{claim.claim_type}\n\n"
        f"【网页标题】\n{result.title}\n\n"
        f"【网页 URL】\n{result.url}\n\n"
        f"【网页内容 source_content】\n{content_for_model}\n\n"
        f"请根据上述网页内容提取与该主张相关的证据。"
    )

    try:
        extract_out: _ExtractOutput = client.chat_json(
            system_prompt=EXTRACT_EVIDENCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=_ExtractOutput,
            temperature=0.0,
        )
    except LLMError as e:
        return _build_failed_evidence(
            claim, result, query, evidence_id,
            status="failed",
            reason=f"LLM 提取失败：{e}",
            content=result.content,
        )

    # 防幻觉：校验 relevant_excerpt 是否能在 source_content 中找到
    excerpt = extract_out.relevant_excerpt or ""
    if excerpt and not verify_excerpt_in_content(excerpt, result.content or ""):
        # 原文校验未通过：模型可能在编造片段
        return _build_failed_evidence(
            claim, result, query, evidence_id,
            status="invalid_excerpt",
            reason="证据原文校验未通过：模型返回的 relevant_excerpt 未能在网页内容中找到",
            content=result.content,
            summary=extract_out.evidence_summary,
            stance=extract_out.evidence_stance,
        )

    # 来源分级（再调一次 LLM）
    grade_out = _grade_source(client, result, claim)

    # evidence_stance 映射到旧版 supports_or_refutes
    stance_map = {
        "supports": "supports",
        "refutes": "refutes",
        "context": "partial",
        "irrelevant": "unclear",
    }
    supports_or_refutes = stance_map.get(extract_out.evidence_stance, "unclear")

    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim.claim_id,
        source_title=result.title,
        source_url=result.url,
        publisher=result.publisher,
        published_at=now,
        retrieved_at=datetime.now(),
        evidence_summary=extract_out.evidence_summary or result.content[:200],
        source_type=grade_out.source_type if grade_out else result.publisher,
        source_grade=grade_out.source_grade if grade_out else "D",
        supports_or_refutes=supports_or_refutes,
        is_primary_source=extract_out.is_primary_source,
        reliability_reason=grade_out.reliability_reason if grade_out else "未分级",
        search_query=query,
        source_domain=result.domain,
        source_content=result.content,
        relevant_excerpt=excerpt or None,
        relevance_score=extract_out.relevance_score,
        evidence_stance=extract_out.evidence_stance,
        directness=extract_out.directness,
        independence_group=None,  # 由 evaluate 节点统一分组
        extraction_status="success",
    )


def _grade_source(client: LLMClient, result: SearchResult, claim: Claim) -> Optional[_GradeOutput]:
    """使用 LLM 评估来源等级。失败时返回 None，由调用方使用默认值。"""
    user_prompt = (
        f"【网页标题】\n{result.title}\n\n"
        f"【网页 URL】\n{result.url}\n\n"
        f"【发布者】\n{result.publisher}\n\n"
        f"【域名】\n{result.domain}\n\n"
        f"【网页内容片段】\n{(result.content or '')[:2000]}\n\n"
        f"【关联主张】\n{claim.text}\n\n"
        f"请评估该来源的可信度等级。"
    )
    try:
        return client.chat_json(
            system_prompt=GRADE_SOURCE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=_GradeOutput,
            temperature=0.0,
        )
    except LLMError:
        return None


def _build_failed_evidence(
    claim: Claim,
    result: SearchResult,
    query: str,
    evidence_id: str,
    status: str,
    reason: str,
    content: Optional[str] = None,
    summary: str = "",
    stance: str = "irrelevant",
) -> Evidence:
    """构造一个提取失败的证据对象（不参与最终判断）。"""
    grade_map = {"success": "D", "insufficient_content": "E", "invalid_excerpt": "E", "failed": "E"}
    return Evidence(
        evidence_id=evidence_id,
        claim_id=claim.claim_id,
        source_title=result.title,
        source_url=result.url,
        publisher=result.publisher,
        published_at=result.published_at,
        retrieved_at=datetime.now(),
        evidence_summary=summary or reason[:200],
        source_type="提取失败",
        source_grade=grade_map.get(status, "E"),
        supports_or_refutes="unclear",
        is_primary_source=False,
        reliability_reason=reason,
        search_query=query,
        source_domain=result.domain,
        source_content=content,
        relevant_excerpt=None,
        relevance_score=0.0,
        evidence_stance=stance,
        directness="unclear",
        independence_group=None,
        extraction_status=status,
    )
