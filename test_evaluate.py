"""评估与交叉验证节点单元测试。

测试规则模式、LLM 交叉验证模式、强制约束（高风险主张保守结论、C/D 级证据不得判为已证实）。
所有测试 mock LLM，不真实消耗 API 额度。
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.models import AgentState, Claim, ClaimResult, Evidence
from src.nodes.evaluate import (
    _CrossValidateOutput,
    _evaluate_single_claim_llm,
    _evaluate_single_claim_rule,
    evaluate_claims,
)


# ============ 规则模式测试 ============


def _build_claim(text: str = "某市因暴雨导致地铁全线停运", risk_level: str = "medium") -> Claim:
    return Claim(
        claim_id="c1",
        text=text,
        claim_type="事件陈述",
        entities=["某市", "地铁"],
        time_reference="当前",
        location="某市",
        risk_level=risk_level,
    )


def _build_evidence(
    claim_id: str = "c1",
    grade: str = "B",
    stance: str = "supports",
    domain: str = "news1.com",
    extraction_status: str = "success",
) -> Evidence:
    return Evidence(
        evidence_id=f"{claim_id}-e1",
        claim_id=claim_id,
        source_title="测试来源",
        source_url=f"https://{domain}/article",
        publisher="测试发布者",
        published_at=datetime(2024, 6, 25),
        retrieved_at=datetime.now(),
        evidence_summary="测试证据摘要",
        source_type="媒体",
        source_grade=grade,
        supports_or_refutes=stance,
        is_primary_source=False,
        reliability_reason="测试",
        source_domain=domain,
        relevant_excerpt="测试原文片段",
        relevance_score=0.8,
        evidence_stance=stance,
        directness="direct",
        extraction_status=extraction_status,
    )


def test_evaluate_rule_mode_no_evidence():
    """规则模式：无证据时返回证据不足。"""
    claim = _build_claim()
    result = _evaluate_single_claim_rule(claim, [])
    assert result.verdict == "证据不足"
    assert result.confidence <= 0.5


def test_evaluate_rule_mode_metro_claim():
    """规则模式：地铁停运主张返回部分属实。"""
    claim = _build_claim(text="某市因暴雨导致地铁全线停运")
    evidence = _build_evidence()
    result = _evaluate_single_claim_rule(claim, [evidence])
    assert result.verdict == "部分属实"


def test_evaluate_rule_mode_missing_claim():
    """规则模式：未匹配关键词的主张返回证据不足。"""
    claim = _build_claim(text="某公司发布财报")
    evidence = _build_evidence()
    result = _evaluate_single_claim_rule(claim, [evidence])
    assert result.verdict == "证据不足"


def test_evaluate_demo_mode_uses_rule():
    """演示模式应使用规则评估。"""
    state = AgentState(
        original_text="网传某市因暴雨导致地铁全线停运。",
        mode="demo",
    )
    state.claims = [_build_claim()]
    state.evidence = {"c1": [_build_evidence()]}
    state = evaluate_claims(state)

    assert len(state.claim_results) == 1
    assert state.claim_results[0].verdict in (
        "已证实", "基本属实", "部分属实", "证据不足", "存在误导", "已证伪", "仍在发展"
    )


# ============ LLM 交叉验证：强制约束测试 ============


def _mock_settings_full_configured():
    """临时伪造完整配置（LLM + Tavily），返回原值用于恢复。"""
    orig_llm_key = settings.LLM_API_KEY
    orig_llm_model = settings.LLM_MODEL
    orig_tavily_key = settings.TAVILY_API_KEY
    settings.LLM_API_KEY = "fake-key-for-testing"
    settings.LLM_MODEL = "fake-model-for-testing"
    settings.TAVILY_API_KEY = "fake-tavily-key"
    return orig_llm_key, orig_llm_model, orig_tavily_key


def _restore_settings(llm_key: str, llm_model: str, tavily_key: str) -> None:
    settings.LLM_API_KEY = llm_key
    settings.LLM_MODEL = llm_model
    settings.TAVILY_API_KEY = tavily_key


def test_llm_cross_validate_normal_case():
    """LLM 交叉验证：低风险主张正常返回结论。"""
    claim = _build_claim(text="嫦娥六号于2024年6月返回地球", risk_level="low")
    evidences = [
        _build_evidence(grade="A", stance="supports", domain="cnsa.gov.cn"),
        _build_evidence(grade="B", stance="supports", domain="xinhuanet.com"),
    ]

    cross_output = _CrossValidateOutput(
        verdict="已证实",
        confidence=0.95,
        reasoning="多个权威来源一致支持该主张。证据1为国家航天局公告，证据2为新华社报道。",
        missing_information=None,
        independent_sources_count=2,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    assert result.verdict == "已证实"
    assert result.confidence == 0.95


def test_llm_high_risk_without_a_grade_downgraded():
    """强制约束：高风险名誉指控没有 A 级证据时，结论不得高于证据不足。"""
    claim = _build_claim(
        text="分手原因是鹿晗男女关系混乱",
        risk_level="high",
    )
    # 只有 B 级证据，且只有 1 个独立来源
    evidences = [_build_evidence(grade="B", stance="supports", domain="entnews.com")]

    cross_output = _CrossValidateOutput(
        verdict="已证实",  # 模型试图判为已证实
        confidence=0.8,
        reasoning="根据报道...",
        missing_information=None,
        independent_sources_count=1,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    # 应被降级为证据不足
    assert result.verdict == "证据不足"
    assert "强制约束" in result.reasoning


def test_llm_high_risk_with_a_grade_keeps_verdict():
    """高风险主张有 A 级证据且多个独立来源时保持原结论。"""
    claim = _build_claim(
        text="分手原因是鹿晗男女关系混乱",
        risk_level="high",
    )
    evidences = [
        _build_evidence(grade="A", stance="supports", domain="official1.com"),
        _build_evidence(grade="B", stance="supports", domain="news1.com"),
        _build_evidence(grade="B", stance="supports", domain="news2.com"),
    ]

    cross_output = _CrossValidateOutput(
        verdict="已证实",
        confidence=0.9,
        reasoning="多个独立高质量来源支持。",
        missing_information=None,
        independent_sources_count=3,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    # 满足条件，不应被降级
    assert result.verdict == "已证实"


def test_llm_only_c_d_grade_cannot_be_verified():
    """强制约束：只有 C/D 级证据时不得判为已证实。"""
    claim = _build_claim(text="某网红发布产品测评", risk_level="low")
    evidences = [
        _build_evidence(grade="C", stance="supports", domain="blog1.com"),
        _build_evidence(grade="D", stance="supports", domain="blog2.com"),
    ]

    cross_output = _CrossValidateOutput(
        verdict="已证实",  # 模型试图判为已证实
        confidence=0.7,
        reasoning="根据多个博客报道...",
        missing_information=None,
        independent_sources_count=2,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    assert result.verdict == "证据不足"
    assert "C/D/E 级" in result.reasoning


def test_llm_no_valid_evidence_returns_insufficient():
    """LLM 模式：无有效证据时直接返回证据不足。"""
    claim = _build_claim()
    # 所有证据都是 invalid_excerpt
    evidences = [
        _build_evidence(extraction_status="invalid_excerpt"),
        _build_evidence(stance="irrelevant", extraction_status="success"),
    ]

    mock_client = MagicMock()
    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    assert result.verdict == "证据不足"
    # 不应调用 LLM
    mock_client.chat_json.assert_not_called()


def test_llm_call_failure_returns_conservative():
    """LLM 交叉验证失败时返回保守结论。"""
    from src.llm.client import LLMError

    claim = _build_claim()
    evidences = [_build_evidence(grade="B", stance="supports")]

    mock_client = MagicMock()
    mock_client.chat_json.side_effect = LLMError("模拟交叉验证失败")

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    assert result.verdict == "证据不足"
    assert "LLM 交叉验证失败" in result.reasoning


def test_llm_irrelevant_evidence_filtered():
    """LLM 模式：irrelevant 证据被过滤，不参与判断。"""
    claim = _build_claim()
    evidences = [
        _build_evidence(stance="irrelevant", extraction_status="success"),
        _build_evidence(stance="supports", grade="B", extraction_status="success"),
    ]

    cross_output = _CrossValidateOutput(
        verdict="基本属实",
        confidence=0.7,
        reasoning="根据证据1...",
        missing_information=None,
        independent_sources_count=1,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    # 只有 1 条有效证据（supports 的那条），结论应正常返回
    assert result.verdict == "基本属实"


def test_llm_reasoning_cites_evidence_id():
    """LLM 输出的推理应引用证据编号（用户提示中已要求）。"""
    claim = _build_claim()
    evidences = [
        _build_evidence(grade="A", stance="supports", domain="cnsa.gov.cn"),
        _build_evidence(grade="B", stance="supports", domain="xinhuanet.com"),
    ]

    cross_output = _CrossValidateOutput(
        verdict="已证实",
        confidence=0.95,
        reasoning="证据1为国家航天局公告，证据2为新华社报道，两者一致支持。",
        missing_information=None,
        independent_sources_count=2,
    )

    mock_client = MagicMock()
    mock_client.chat_json.return_value = cross_output

    state = AgentState(original_text="测试", mode="full")
    result = _evaluate_single_claim_llm(mock_client, claim, evidences, 1, state)

    # 推理中应引用了证据编号
    assert "证据1" in result.reasoning or "证据2" in result.reasoning


# ============ 配置缺失时降级 ============


def test_full_mode_without_llm_config_falls_back_to_rule():
    """完整真实模式未配置 LLM 时应降级为规则模式。"""
    orig_llm_key = settings.LLM_API_KEY
    orig_llm_model = settings.LLM_MODEL
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""

    try:
        state = AgentState(
            original_text="网传某市因暴雨导致地铁全线停运。",
            mode="full",
        )
        state.claims = [_build_claim()]
        state.evidence = {"c1": [_build_evidence()]}
        state = evaluate_claims(state)

        assert len(state.claim_results) == 1
        # 应记录错误
        assert any("LLM" in err for err in state.errors)
    finally:
        settings.LLM_API_KEY = orig_llm_key
        settings.LLM_MODEL = orig_llm_model
