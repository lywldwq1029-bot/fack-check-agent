"""工作流模式与配置测试。"""

import pytest

from src.config import settings
from src.models import AgentState, Claim, VerificationPlan
from src.workflow import run_fact_check_workflow


def test_demo_mode_still_works():
    """演示模式原有测试仍然通过。"""
    text = "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
    report = run_fact_check_workflow(text, mode="demo")
    assert report is not None
    assert report.original_text == text
    assert len(report.claim_results) >= 3


def test_workflow_default_mode_is_demo():
    """默认模式为 demo。"""
    text = "测试文本"
    state = AgentState(original_text=text)
    assert state.mode == "demo"


def test_workflow_llm_mode_without_config_returns_error_report():
    """真实 LLM 模式未配置密钥时应返回带错误的报告。"""
    # 确保未配置
    original_key = settings.LLM_API_KEY
    original_model = settings.LLM_MODEL
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""

    try:
        text = "网传某市因暴雨导致地铁全线停运。"
        report = run_fact_check_workflow(text, mode="llm")
        # 应该返回报告，但报告应反映错误
        assert report is not None
        # 由于未配置，拆解失败，主张应为空
        assert len(report.claim_results) == 0
    finally:
        settings.LLM_API_KEY = original_key
        settings.LLM_MODEL = original_model


def test_llm_configured_check():
    """测试 llm_configured 方法。"""
    original_key = settings.LLM_API_KEY
    original_model = settings.LLM_MODEL

    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    assert settings.llm_configured() is False

    settings.LLM_API_KEY = "fake-key"
    settings.LLM_MODEL = ""
    assert settings.llm_configured() is False

    settings.LLM_API_KEY = "fake-key"
    settings.LLM_MODEL = "fake-model"
    assert settings.llm_configured() is True

    settings.LLM_API_KEY = original_key
    settings.LLM_MODEL = original_model


def test_verification_plan_in_workflow():
    """演示模式下工作流应生成 VerificationPlan 列表。"""
    text = "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
    # 通过 workflow 内部状态验证，这里直接测试 plan 节点
    from src.nodes.decompose import decompose_claims
    from src.nodes.plan import plan_verification

    state = AgentState(original_text=text, mode="demo")
    state = decompose_claims(state)
    state = plan_verification(state)

    assert len(state.verification_plan) > 0
    assert all(isinstance(p, VerificationPlan) for p in state.verification_plan)
    # 每条计划应有搜索语句
    for plan in state.verification_plan:
        assert len(plan.search_queries) >= 1


# ============ 完整真实模式配置预检测试 ============


def test_full_mode_blocked_without_tavily_key():
    """完整真实模式缺少 TAVILY_API_KEY 时应阻止运行并返回错误报告。"""
    original_llm_key = settings.LLM_API_KEY
    original_llm_model = settings.LLM_MODEL
    original_tavily_key = settings.TAVILY_API_KEY

    # 配置 LLM 但不配置 Tavily
    settings.LLM_API_KEY = "fake-llm-key"
    settings.LLM_MODEL = "fake-model"
    settings.TAVILY_API_KEY = ""

    try:
        text = "网传某市因暴雨导致地铁全线停运。"
        report = run_fact_check_workflow(text, mode="full")
        # 应返回报告，但报告应反映配置缺失错误
        assert report is not None
        # 由于配置缺失，不应有主张被核查
        assert len(report.claim_results) == 0
        # 总体摘要应提及配置缺失
        assert "TAVILY_API_KEY" in report.overall_summary or "配置" in report.overall_summary
    finally:
        settings.LLM_API_KEY = original_llm_key
        settings.LLM_MODEL = original_llm_model
        settings.TAVILY_API_KEY = original_tavily_key


def test_full_mode_blocked_without_llm_config():
    """完整真实模式缺少 LLM 配置时应阻止运行并返回错误报告。"""
    original_llm_key = settings.LLM_API_KEY
    original_llm_model = settings.LLM_MODEL
    original_tavily_key = settings.TAVILY_API_KEY

    # 配置 Tavily 但不配置 LLM
    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    settings.TAVILY_API_KEY = "fake-tavily-key"

    try:
        text = "网传某市因暴雨导致地铁全线停运。"
        report = run_fact_check_workflow(text, mode="full")
        assert report is not None
        assert len(report.claim_results) == 0
        assert "LLM_API_KEY" in report.overall_summary or "LLM_MODEL" in report.overall_summary or "配置" in report.overall_summary
    finally:
        settings.LLM_API_KEY = original_llm_key
        settings.LLM_MODEL = original_llm_model
        settings.TAVILY_API_KEY = original_tavily_key


def test_full_mode_blocked_without_all_configs():
    """完整真实模式缺少全部配置时应阻止运行。"""
    original_llm_key = settings.LLM_API_KEY
    original_llm_model = settings.LLM_MODEL
    original_tavily_key = settings.TAVILY_API_KEY

    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    settings.TAVILY_API_KEY = ""

    try:
        text = "网传某市因暴雨导致地铁全线停运。"
        report = run_fact_check_workflow(text, mode="full")
        assert report is not None
        assert len(report.claim_results) == 0
        # 应同时列出所有缺失项
        summary = report.overall_summary
        assert "LLM_API_KEY" in summary
        assert "LLM_MODEL" in summary
        assert "TAVILY_API_KEY" in summary
    finally:
        settings.LLM_API_KEY = original_llm_key
        settings.LLM_MODEL = original_llm_model
        settings.TAVILY_API_KEY = original_tavily_key


def test_search_configured_check():
    """测试 search_configured 方法。"""
    original_tavily_key = settings.TAVILY_API_KEY

    settings.TAVILY_API_KEY = ""
    assert settings.search_configured() is False

    settings.TAVILY_API_KEY = "fake-tavily-key"
    assert settings.search_configured() is True

    settings.TAVILY_API_KEY = original_tavily_key


def test_full_real_configured_check():
    """测试 full_real_configured 方法。"""
    original_llm_key = settings.LLM_API_KEY
    original_llm_model = settings.LLM_MODEL
    original_tavily_key = settings.TAVILY_API_KEY

    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    settings.TAVILY_API_KEY = ""
    assert settings.full_real_configured() is False

    settings.LLM_API_KEY = "fake-key"
    settings.LLM_MODEL = "fake-model"
    settings.TAVILY_API_KEY = ""
    assert settings.full_real_configured() is False

    settings.LLM_API_KEY = "fake-key"
    settings.LLM_MODEL = "fake-model"
    settings.TAVILY_API_KEY = "fake-tavily-key"
    assert settings.full_real_configured() is True

    settings.LLM_API_KEY = original_llm_key
    settings.LLM_MODEL = original_llm_model
    settings.TAVILY_API_KEY = original_tavily_key


def test_missing_configs_returns_all_missing():
    """测试 missing_configs 方法返回所有缺失项。"""
    original_llm_key = settings.LLM_API_KEY
    original_llm_model = settings.LLM_MODEL
    original_tavily_key = settings.TAVILY_API_KEY

    settings.LLM_API_KEY = ""
    settings.LLM_MODEL = ""
    settings.TAVILY_API_KEY = ""

    try:
        missing = settings.missing_configs()
        assert "LLM_API_KEY" in missing
        assert "LLM_MODEL" in missing
        assert "TAVILY_API_KEY" in missing
        assert len(missing) == 3
    finally:
        settings.LLM_API_KEY = original_llm_key
        settings.LLM_MODEL = original_llm_model
        settings.TAVILY_API_KEY = original_tavily_key
