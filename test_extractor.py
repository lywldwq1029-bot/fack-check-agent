"""证据提取工具单元测试。

测试 LLM 证据提取、防幻觉校验（原文片段验证）、来源分级。
所有测试 mock LLM，不真实消耗 API 额度。
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.models import Claim, Evidence
from src.tools.extractor import (
    _normalize_text,
    extract_evidence_from_result,
    verify_excerpt_in_content,
)
from src.tools.search_tool import SearchResult


# ============ 原文片段校验 ============


def test_verify_excerpt_direct_match():
    """测试原文片段直接匹配成功。"""
    content = "2024年6月25日，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆。"
    excerpt = "嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆"
    assert verify_excerpt_in_content(excerpt, content) is True


def test_verify_excerpt_normalized_match():
    """测试规范化后匹配（全角空格、连续空白）。"""
    content = "嫦娥六号 返回舱 在 内蒙古 着陆场 成功着陆。"
    excerpt = "嫦娥六号　返回舱　在　内蒙古　着陆场　成功着陆"
    # 全角空格应被规范化为半角
    assert verify_excerpt_in_content(excerpt, content) is True


def test_verify_excerpt_prefix_match():
    """测试长片段前 50 字符匹配。"""
    content = "国家航天局今日发布公告，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆，标志着任务圆满完成。"
    excerpt = "国家航天局今日发布公告，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆，标志着任务圆满完成。后续将开展样品分析工作。"
    # 片段比原文长，但前 50 字符应能匹配
    assert verify_excerpt_in_content(excerpt, content) is True


def test_verify_excerpt_not_found_rejects():
    """测试模型虚构的片段不在原文中应被拒绝。"""
    content = "嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆。"
    excerpt = "嫦娥六号在海南文昌着陆场成功着陆"  # 地点虚构
    assert verify_excerpt_in_content(excerpt, content) is False


def test_verify_excerpt_empty_excerpt():
    """空片段应返回 False。"""
    assert verify_excerpt_in_content("", "内容") is False


def test_verify_excerpt_empty_content():
    """空内容应返回 False。"""
    assert verify_excerpt_in_content("片段", "") is False


def test_verify_excerpt_both_empty():
    """两者皆空应返回 False。"""
    assert verify_excerpt_in_content("", "") is False


def test_normalize_text_handles_whitespace():
    """测试规范化文本处理空白。"""
    assert _normalize_text("  多个  空格  ") == "多个 空格"
    assert _normalize_text("") == ""
    assert _normalize_text(None) == ""


# ============ 证据提取：正常情况 ============


def _build_claim() -> Claim:
    return Claim(
        claim_id="c1",
        text="嫦娥六号于2024年6月返回地球",
        claim_type="事件陈述",
        entities=["嫦娥六号"],
        time_reference="2024年6月",
        risk_level="low",
    )


def _build_result() -> SearchResult:
    return SearchResult(
        title="国家航天局：嫦娥六号返回舱成功着陆",
        url="https://www.cnsa.gov.cn/article/2024/06/change6.html",
        content="2024年6月25日，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆，标志着嫦娥六号任务圆满完成。",
        publisher="国家航天局",
        published_at=datetime(2024, 6, 25, 10, 0),
        score=0.95,
    )


def test_extract_evidence_success():
    """测试正常证据提取流程。"""
    claim = _build_claim()
    result = _build_result()

    extract_output = MagicMock()
    extract_output.evidence_summary = "嫦娥六号返回舱于2024年6月25日成功着陆"
    extract_output.relevant_excerpt = "嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆"
    extract_output.evidence_stance = "supports"
    extract_output.directness = "direct"
    extract_output.is_primary_source = True
    extract_output.relevance_score = 0.95
    extract_output.source_type_guess = "官方通报"

    grade_output = MagicMock()
    grade_output.source_grade = "A"
    grade_output.reliability_reason = "国家航天局为事件责任主体，原始公告"
    grade_output.source_type = "官方通报"

    mock_client = MagicMock()
    mock_client.chat_json.side_effect = [extract_output, grade_output]

    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="嫦娥六号 返回 地球",
        evidence_id="c1-e1",
    )

    assert evidence.evidence_id == "c1-e1"
    assert evidence.claim_id == "c1"
    assert evidence.extraction_status == "success"
    assert evidence.evidence_stance == "supports"
    assert evidence.source_grade == "A"
    assert evidence.is_primary_source is True
    assert evidence.relevance_score == 0.95
    assert "嫦娥六号返回舱" in evidence.relevant_excerpt


# ============ 防幻觉：模型虚构片段被拒绝 ============


def test_extract_evidence_hallucinated_excerpt_rejected():
    """测试模型虚构的原文片段被拒绝，标记为 invalid_excerpt。"""
    claim = _build_claim()
    result = _build_result()

    # 模型返回了一个不存在的片段（虚构了地点）
    extract_output = MagicMock()
    extract_output.evidence_summary = "嫦娥六号在海南文昌成功着陆"
    extract_output.relevant_excerpt = "嫦娥六号返回舱在海南文昌着陆场成功着陆"  # 原文是内蒙古四子王旗
    extract_output.evidence_stance = "supports"
    extract_output.directness = "direct"
    extract_output.is_primary_source = True
    extract_output.relevance_score = 0.9
    extract_output.source_type_guess = "官方通报"

    mock_client = MagicMock()
    mock_client.chat_json.return_value = extract_output

    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="嫦娥六号 返回 地球",
        evidence_id="c1-e1",
    )

    assert evidence.extraction_status == "invalid_excerpt"
    assert evidence.source_grade == "E"  # 失败证据降级为 E
    assert "校验未通过" in evidence.reliability_reason
    # 不应该参与最终判断
    assert evidence.evidence_stance != "supports" or evidence.extraction_status != "success"


def test_extract_evidence_excerpt_in_different_part_of_content():
    """测试片段在内容其他部分时仍能匹配。"""
    claim = _build_claim()
    result = SearchResult(
        title="新华社报道",
        url="https://www.xinhuanet.com/science/2024-06/change6.html",
        content="（新华社记者报道）根据国家航天局消息，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆。本次任务实现了人类首次月球背面采样返回。",
        publisher="新华社",
        published_at=datetime(2024, 6, 25),
        score=0.88,
    )

    extract_output = MagicMock()
    extract_output.evidence_summary = "嫦娥六号返回舱成功着陆"
    extract_output.relevant_excerpt = "嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆"
    extract_output.evidence_stance = "supports"
    extract_output.directness = "direct"
    extract_output.is_primary_source = False
    extract_output.relevance_score = 0.85
    extract_output.source_type_guess = "权威媒体"

    grade_output = MagicMock()
    grade_output.source_grade = "B"
    grade_output.reliability_reason = "新华社有署名报道"
    grade_output.source_type = "权威媒体"

    mock_client = MagicMock()
    mock_client.chat_json.side_effect = [extract_output, grade_output]

    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="嫦娥六号",
        evidence_id="c1-e1",
    )

    assert evidence.extraction_status == "success"
    assert evidence.source_grade == "B"


# ============ 空内容处理 ============


def test_extract_evidence_empty_content():
    """测试网页内容为空时返回 insufficient_content。"""
    claim = _build_claim()
    result = SearchResult(
        title="空网页",
        url="https://example.com/empty",
        content="",
        publisher="未知",
        published_at=None,
        score=0.0,
    )

    mock_client = MagicMock()
    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="测试",
        evidence_id="c1-e1",
    )

    assert evidence.extraction_status == "insufficient_content"
    assert "为空" in evidence.reliability_reason
    # 不应调用 LLM
    mock_client.chat_json.assert_not_called()


def test_extract_evidence_whitespace_only_content():
    """测试网页内容只有空白时返回 insufficient_content。"""
    claim = _build_claim()
    result = SearchResult(
        title="空白网页",
        url="https://example.com/ws",
        content="   \n\t  \n  ",
        publisher="未知",
        published_at=None,
        score=0.0,
    )

    mock_client = MagicMock()
    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="测试",
        evidence_id="c1-e1",
    )

    assert evidence.extraction_status == "insufficient_content"


# ============ LLM 调用失败 ============


def test_extract_evidence_llm_failure():
    """测试 LLM 调用失败时返回 failed 状态。"""
    from src.llm.client import LLMError

    claim = _build_claim()
    result = _build_result()

    mock_client = MagicMock()
    mock_client.chat_json.side_effect = LLMError("模拟 LLM 调用失败")

    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="嫦娥六号",
        evidence_id="c1-e1",
    )

    assert evidence.extraction_status == "failed"
    assert "LLM 提取失败" in evidence.reliability_reason


# ============ 来源分级失败容错 ============


def test_extract_evidence_grade_failure_uses_default():
    """测试来源分级调用失败时使用默认值 D。"""
    from src.llm.client import LLMError

    claim = _build_claim()
    result = _build_result()

    extract_output = MagicMock()
    extract_output.evidence_summary = "嫦娥六号返回舱成功着陆"
    extract_output.relevant_excerpt = "嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆"
    extract_output.evidence_stance = "supports"
    extract_output.directness = "direct"
    extract_output.is_primary_source = True
    extract_output.relevance_score = 0.9
    extract_output.source_type_guess = "官方通报"

    mock_client = MagicMock()
    # 第一次（提取）成功，第二次（分级）失败
    mock_client.chat_json.side_effect = [extract_output, LLMError("分级失败")]

    evidence = extract_evidence_from_result(
        client=mock_client,
        claim=claim,
        result=result,
        query="嫦娥六号",
        evidence_id="c1-e1",
    )

    # 提取本身成功
    assert evidence.extraction_status == "success"
    # 分级失败时使用默认值 D
    assert evidence.source_grade == "D"
