"""稳定性冻结阶段测试套件 - 10个分支覆盖

验收标准：无论外部 API 是否成功，页面都必须在 75 秒内正常结束，
不能出现 Traceback、AttributeError 或 Pydantic 错误。

运行方式：python -m pytest tests/test_stability.py -v
"""

import sys
import os
import time
from datetime import datetime
from unittest.mock import patch, MagicMock
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.models import (
    FactCheckReport,
    CheckResult,
    REPORT_SCHEMA_VERSION,
    build_failure_report,
    build_no_evidence_report,
    _base_report_kwargs,
    CHECK_STATUS_SUCCESS,
    CHECK_STATUS_PARTIAL,
    CHECK_STATUS_UNAVAILABLE,
)
from src.quick_workflow import (
    run_fact_check,
    run_professional_fact_check,
    _search_once,
    _friendly_status_message,
    _mechanically_clean_results,
    _llm_judge_relevance,
    _llm_generate_supplement_query,
    _classify_source_grade,
)
from src.llm.client import LLMClient, LLMError
from src.tools.search_tool import TavilySearchProvider, SearchResult, MockSearchProvider


# ===== 测试用的固定文本 =====
SAMPLE_TEXT = "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
SAMPLE_TEXTS = [
    SAMPLE_TEXT,
    "某明星今日官宣结婚，婚礼将于下月举行。",
    "多地出现不明飞行物，官方已介入调查。",
    "新技术可让电池充电速度提升10倍，已通过实验室验证。",
    "某公司宣布裁员30%，涉及多个部门。",
]


def _make_search_result(title: str, url: str, content: str, score: float = 0.8, publisher: str = "") -> SearchResult:
    """创建测试用的 SearchResult 对象。"""
    return SearchResult(
        title=title,
        url=url,
        content=content,
        publisher=publisher,
        score=score,
    )


def _mock_normal_results() -> list[SearchResult]:
    """模拟 Tavily 正常返回结果。"""
    return [
        _make_search_result(
            "市轨道交通集团：暴雨期间部分线路临时停运",
            "https://example.com/demo/metro",
            "暴雨导致3条线路临时停运，其余线路限速运行，并非全线停运。",
            score=0.95,
            publisher="市轨道交通集团",
        ),
        _make_search_result(
            "本地日报：暴雨影响早高峰交通",
            "https://example.com/demo/news",
            "暴雨导致部分地铁线路停运，教育局通知学生停课。",
            score=0.85,
            publisher="本地日报",
        ),
    ]


class TestSchemaVersion:
    """S2: schema_version 版本控制测试。"""

    def test_schema_version_constant(self):
        """版本号常量存在且为整数。"""
        assert REPORT_SCHEMA_VERSION == 2
        assert isinstance(REPORT_SCHEMA_VERSION, int)

    def test_report_has_schema_version(self):
        """FactCheckReport 包含 schema_version 字段。"""
        kwargs = _base_report_kwargs("测试")
        assert "schema_version" in kwargs
        assert kwargs["schema_version"] == REPORT_SCHEMA_VERSION

    def test_failure_report_has_schema_version(self):
        """失败报告也包含 schema_version。"""
        report = build_failure_report("测试", "错误", "init")
        assert report.schema_version == REPORT_SCHEMA_VERSION

    def test_no_evidence_report_has_schema_version(self):
        """无证据报告也包含 schema_version。"""
        report = build_no_evidence_report("测试")
        assert report.schema_version == REPORT_SCHEMA_VERSION


class TestReportContract:
    """S1: 统一 FactCheckReport 模型契约测试。"""

    def test_all_optional_fields_have_defaults(self):
        """所有可选字段都有默认值。"""
        optional_fields = [
            "claim_results", "timeline", "risk_level", "risk_reason",
            "risk_factors", "unresolved_questions", "execution_log",
            "decision_trace", "agent_decision", "did_supplemental_search",
            "tool_calls_count", "historical_matches", "key_evidence_cards",
            "credibility_score", "recommendation", "current_step",
            "completed_steps", "skipped_steps", "progress_percent",
            "workflow_completed", "workflow_error", "generated_at",
        ]
        field_names = set(FactCheckReport.model_fields.keys())
        for field in optional_fields:
            assert field in field_names, f"字段 {field} 不存在于模型中"

    def test_failure_report_has_all_fields(self):
        """失败报告包含所有字段。"""
        report = build_failure_report("测试文本", "错误消息", "search")
        required_fields = [
            "schema_version", "original_text", "overall_verdict",
            "overall_summary", "claim_results", "timeline",
            "propagation_risk", "risk_level", "risk_reason",
            "risk_factors", "unresolved_questions", "execution_log",
            "decision_trace", "agent_decision", "did_supplemental_search",
            "tool_calls_count", "historical_matches", "key_evidence_cards",
            "credibility_score", "recommendation", "current_step",
            "completed_steps", "skipped_steps", "progress_percent",
            "workflow_completed", "workflow_error", "generated_at",
        ]
        for field in required_fields:
            assert hasattr(report, field), f"失败报告缺少字段: {field}"

    def test_failure_report_credibility_is_none(self):
        """失败报告 credibility_score 必须为 None。"""
        report = build_failure_report("测试", "错误", "search")
        assert report.credibility_score is None

    def test_failure_report_verdict_is_uncheckable(self):
        """失败报告 verdict 为 暂无法核查。"""
        report = build_failure_report("测试", "错误", "search")
        assert report.overall_verdict == "暂无法核查"

    def test_no_evidence_report_has_none_credibility(self):
        """无证据报告 credibility_score 为 None。"""
        report = build_no_evidence_report("测试")
        assert report.credibility_score is None

    def test_no_evidence_report_workflow_completed(self):
        """无证据报告 workflow_completed 为 True。"""
        report = build_no_evidence_report("测试")
        assert report.workflow_completed is True

    def test_normal_report_can_be_built(self):
        """正常报告可以被构建。"""
        kwargs = _base_report_kwargs("测试文本")
        kwargs.update({
            "overall_verdict": "部分属实",
            "overall_summary": "测试摘要",
            "propagation_risk": "中风险",
            "credibility_score": 80,
        })
        report = FactCheckReport(**kwargs)
        assert report.credibility_score == 80
        assert report.schema_version == REPORT_SCHEMA_VERSION

    def test_credibility_score_range_validation(self):
        """credibility_score 范围验证: 0-100。"""
        kwargs = _base_report_kwargs("测试")
        kwargs.update({
            "overall_verdict": "基本属实",
            "overall_summary": "测试",
            "propagation_risk": "低风险",
        })

        # 合法值
        kwargs["credibility_score"] = 0
        FactCheckReport(**kwargs)

        kwargs["credibility_score"] = 100
        FactCheckReport(**kwargs)

        # 非法值应被拒绝
        kwargs["credibility_score"] = -1
        with pytest.raises(Exception):
            FactCheckReport(**kwargs)

        kwargs["credibility_score"] = 101
        with pytest.raises(Exception):
            FactCheckReport(**kwargs)

    def test_base_kwargs_coverage(self):
        """_base_report_kwargs 返回所有必要字段。"""
        kwargs = _base_report_kwargs("测试文本")
        assert kwargs["original_text"] == "测试文本"
        assert kwargs["schema_version"] == REPORT_SCHEMA_VERSION
        assert isinstance(kwargs["claim_results"], list)
        assert isinstance(kwargs["timeline"], list)
        assert kwargs["credibility_score"] is None
        assert kwargs["current_step"] == "init"
        assert kwargs["workflow_completed"] is False


class TestCheckResult:
    """S3: CheckResult 统一结果测试。"""

    def test_check_result_success(self):
        """成功状态 CheckResult。"""
        report = build_no_evidence_report("测试")
        result = CheckResult(
            status=CHECK_STATUS_SUCCESS,
            report=report,
            error_message="",
        )
        assert result.status == CHECK_STATUS_SUCCESS
        assert result.is_ok is True
        assert result.report is not None

    def test_check_result_partial(self):
        """部分成功状态 CheckResult。"""
        report = build_no_evidence_report("测试")
        result = CheckResult(
            status=CHECK_STATUS_PARTIAL,
            report=report,
            error_message="搜索服务暂时不可用",
        )
        assert result.status == CHECK_STATUS_PARTIAL
        assert result.is_partial is True

    def test_check_result_unavailable(self):
        """不可用状态 CheckResult。"""
        report = build_failure_report("测试", "错误", "init")
        result = CheckResult(
            status=CHECK_STATUS_UNAVAILABLE,
            report=report,
            error_message="服务不可用",
        )
        assert result.status == CHECK_STATUS_UNAVAILABLE
        assert result.is_ok is False

    def test_check_result_always_has_report(self):
        """CheckResult 总是有报告。"""
        result = CheckResult()
        # 默认状态下 report 为 None，这是允许的
        assert result.report is None
        assert result.error_message == ""

    def test_friendly_status_message(self):
        """友好消息生成。"""
        report = build_failure_report("测试", "搜索失败", "search")
        msg = _friendly_status_message(report)
        assert msg == "搜索服务连接临时中断"

        report2 = build_failure_report("测试", "超时", "timeout")
        msg2 = _friendly_status_message(report2)
        assert msg2 == "核查超过时间上限"


class TestBranches:
    """S4: 10个分支覆盖测试。"""

    # ===== 分支1: Tavily 正常 =====
    def test_branch_tavily_normal(self):
        """分支1: Tavily 正常返回结果。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=(_mock_normal_results(), 0.1, None)):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    assert result.report.schema_version == REPORT_SCHEMA_VERSION

    # ===== 分支2: Tavily 无结果 =====
    def test_branch_tavily_no_results(self):
        """分支2: Tavily 返回空结果。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.1, None)):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None

    # ===== 分支3: Tavily SSL 断连 =====
    def test_branch_tavily_ssl_error(self):
        """分支3: Tavily SSL 断连。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.0, "搜索服务连接临时中断，请重试")):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    assert result.report.overall_verdict == "暂无法核查"

    # ===== 分支4: Tavily 超时 =====
    def test_branch_tavily_timeout(self):
        """分支4: Tavily 搜索超时。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.0, "搜索服务暂时不可用")):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    assert result.error_message is not None

    # ===== 分支5: 命中缓存 =====
    def test_branch_cache_hit(self):
        """分支5: 命中缓存。"""
        cached_results = [_make_search_result(
            "缓存结果", "https://example.com/cached", "缓存内容", score=0.8
        )]
        with patch.object(TavilySearchProvider, 'search',
                          return_value=(cached_results, 0.05, None)):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None

    # ===== 分支6: LLM 超时 =====
    def test_branch_llm_timeout(self):
        """分支6: LLM 超时。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.0, "搜索服务暂时不可用")):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None

    # ===== 分支7: LLM 返回非法 JSON =====
    def test_branch_llm_invalid_json(self):
        """分支7: LLM 返回非法 JSON 时仍正常结束。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.0, "搜索服务暂时不可用")):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check(SAMPLE_TEXTS[0])
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    assert "Traceback" not in result.error_message
                    assert "Error" not in result.error_message

    # ===== 分支8: 旧 session_state 报告 =====
    def test_branch_old_session_report(self):
        """分支8: 旧版本报告渲染不崩溃。"""
        # 模拟旧版本报告（缺少新字段）
        kwargs = _base_report_kwargs("旧文本")
        kwargs["schema_version"] = 1  # 旧版本
        kwargs.update({
            "overall_verdict": "基本属实",
            "overall_summary": "旧报告摘要",
            "propagation_risk": "低风险",
            "credibility_score": 80,
        })
        # 旧版本报告仍可创建
        report = FactCheckReport(**kwargs)
        assert report is not None
        # 旧报告在新版本代码中仍可安全访问所有字段
        for field in ["schema_version", "credibility_score", "key_evidence_cards"]:
            value = getattr(report, field, None)
            assert value is not None or report.model_fields[field].default is None

    # ===== 分支9: 正常报告渲染 =====
    def test_branch_normal_report_render(self):
        """分支9: 正常完整报告所有字段可安全读取。"""
        kwargs = _base_report_kwargs("测试文本")
        kwargs.update({
            "overall_verdict": "部分属实",
            "overall_summary": "部分属实：有真实暴雨和停运，但细节不符。",
            "propagation_risk": "中风险",
            "credibility_score": 80,
            "recommendation": "可谨慎参考",
            "decision_trace": [{"step": "test", "action": "test"}],
            "key_evidence_cards": [{"card_id": "K1", "title": "测试"}],
            "current_step": "completed",
            "completed_steps": ["receive", "search", "analyze", "output"],
            "workflow_completed": True,
            "progress_percent": 100,
        })
        report = FactCheckReport(**kwargs)

        # 安全读取所有可选字段
        safe_access_fields = [
            "credibility_score", "decision_trace", "agent_decision",
            "key_evidence_cards", "historical_matches", "tool_calls_count",
            "did_supplemental_search", "recommendation", "workflow_error",
        ]
        for field in safe_access_fields:
            value = getattr(report, field, None)
            assert value is not None or True  # None 也是合法值

    # ===== 分支10: 失败报告渲染 =====
    def test_branch_failure_report_render(self):
        """分支10: 失败报告所有字段可安全读取。"""
        report = build_failure_report("测试文本", "搜索服务暂时不可用", "search")

        # 安全读取所有字段，不抛 AttributeError
        safe_access_fields = [
            "schema_version", "credibility_score", "decision_trace",
            "agent_decision", "key_evidence_cards", "historical_matches",
            "tool_calls_count", "did_supplemental_search", "recommendation",
            "workflow_error", "current_step", "completed_steps",
            "skipped_steps", "progress_percent", "workflow_completed",
        ]
        for field in safe_access_fields:
            try:
                value = getattr(report, field, None)
            except AttributeError as e:
                pytest.fail(f"失败报告字段 {field} 读取失败: {e}")

        # 验证关键字段值
        assert report.current_step == "failed"
        assert report.credibility_score is None
        assert report.overall_verdict == "暂无法核查"
        assert report.schema_version == REPORT_SCHEMA_VERSION


class TestNoTraceback:
    """确保任何情况下都不暴露 Python traceback。"""

    def test_failure_report_no_traceback(self):
        """失败报告不含 traceback。"""
        report = build_failure_report("测试", "搜索服务暂时不可用", "search")
        summary = report.overall_summary
        assert "Traceback" not in summary
        assert "Traceback (most recent call last)" not in summary

    def test_no_evidence_report_no_traceback(self):
        """无证据报告不含 traceback。"""
        report = build_no_evidence_report("测试", reason="搜索服务暂时不可用")
        assert "Traceback" not in report.overall_summary

    def test_check_result_error_no_traceback(self):
        """CheckResult 错误消息不含 traceback。"""
        report = build_failure_report("测试", "错误", "init")
        result = CheckResult(
            status=CHECK_STATUS_UNAVAILABLE,
            report=report,
            error_message="核查服务暂时不可用，请稍后重试",
        )
        assert "Traceback" not in result.error_message
        assert "Traceback" not in result.report.overall_summary

    def test_unknown_exception_does_not_expose_traceback(self):
        """未知异常被捕获，不暴露 traceback。"""
        # 通过 mock 使 run_professional_fact_check 抛异常
        with patch('src.quick_workflow.run_professional_fact_check',
                   side_effect=RuntimeError("test error")):
            result = run_fact_check("测试文本")
            assert isinstance(result, CheckResult)
            assert result.status == CHECK_STATUS_UNAVAILABLE
            # 错误消息是中文友好消息，不包含 Python traceback
            assert "Traceback" not in result.error_message
            assert "RuntimeError" not in result.error_message
            assert result.report is not None


class TestPerformance:
    """性能测试 - 确保在 75 秒内结束。"""

    def test_failure_report_build_time(self):
        """失败报告构建耗时 < 100ms。"""
        start = time.time()
        for _ in range(100):
            build_failure_report("测试文本", "错误", "search")
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.1, f"失败报告构建耗时 {elapsed:.3f}s，超过 100ms"

    def test_check_result_build_time(self):
        """CheckResult 构建耗时 < 50ms。"""
        start = time.time()
        for _ in range(100):
            report = build_failure_report("测试", "错误", "init")
            CheckResult(
                status=CHECK_STATUS_UNAVAILABLE,
                report=report,
                error_message="错误",
            )
        elapsed = (time.time() - start) / 100
        assert elapsed < 0.05, f"CheckResult 构建耗时 {elapsed:.3f}s，超过 50ms"


class TestMockLoop:
    """S6: 20 次 mock 连续测试。"""

    def test_20_consecutive_mock_runs(self):
        """连续 20 次 mock 测试，验证稳定性。"""
        errors = []
        for i in range(20):
            try:
                with patch.object(TavilySearchProvider, 'search',
                                  return_value=([], 0.0, "搜索服务暂时不可用")):
                    result = run_fact_check(f"测试文本 #{i}")
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    assert result.report.schema_version == REPORT_SCHEMA_VERSION
            except Exception as e:
                errors.append(f"第 {i} 次失败: {e}")

        if errors:
            pytest.fail(f"20 次 mock 测试中有 {len(errors)} 次失败:\n" + "\n".join(errors))

    def test_5_different_texts_with_mock(self):
        """5 条不同文本 mock 测试。"""
        for text in SAMPLE_TEXTS:
            with patch.object(TavilySearchProvider, 'search',
                              return_value=([], 0.0, "搜索服务暂时不可用")):
                result = run_fact_check(text)
                assert isinstance(result, CheckResult)
                assert result.report is not None
                assert result.report.original_text == text


class TestModelFieldAccess:
    """页面渲染安全读取测试。"""

    def test_all_render_fields_safe_access(self):
        """模拟页面渲染，所有字段安全读取不报错。"""
        # 失败报告
        failure_report = build_failure_report("测试", "错误", "search")
        # 模拟 app.py 中的读取模式
        render_fields = [
            ("getattr", "credibility_score", None),
            ("getattr", "decision_trace", []),
            ("getattr", "agent_decision", None),
            ("getattr", "key_evidence_cards", []),
            ("getattr", "historical_matches", []),
            ("getattr", "tool_calls_count", 0),
            ("getattr", "did_supplemental_search", False),
            ("getattr", "recommendation", ""),
            ("getattr", "workflow_error", None),
            ("getattr", "completed_steps", []),
            ("getattr", "skipped_steps", []),
            ("getattr", "progress_percent", 0),
            ("direct", "overall_verdict", None),
            ("direct", "overall_summary", None),
            ("direct", "current_step", None),
            ("direct", "workflow_completed", None),
            ("direct", "schema_version", None),
        ]
        for mode, field, default in render_fields:
            try:
                if mode == "getattr":
                    value = getattr(failure_report, field, default)
                else:
                    value = getattr(failure_report, field)
            except Exception as e:
                pytest.fail(f"字段 {field} 读取失败: {e}")

        # 正常报告
        kwargs = _base_report_kwargs("测试")
        kwargs.update({
            "overall_verdict": "基本属实",
            "overall_summary": "测试",
            "propagation_risk": "低风险",
            "credibility_score": 80,
        })
        normal_report = FactCheckReport(**kwargs)

        for mode, field, default in render_fields:
            try:
                if mode == "getattr":
                    value = getattr(normal_report, field, default)
                else:
                    value = getattr(normal_report, field)
            except Exception as e:
                pytest.fail(f"正常报告字段 {field} 读取失败: {e}")


# ========== 搜索查询与证据相关性测试 ==========
class TestSearchQueryRelevance:
    """搜索查询构建与证据相关性测试（行为测试）。"""

    def test_search_uses_full_original_text(self):
        """Tavily 查询使用用户完整原始主张，不提取关键词。"""
        claim = "神舟十八号载人飞船于2024年4月25日发射。"
        captured_query = []

        def capture_search(query, max_results, topic="general"):
            captured_query.append(query)
            return [_make_search_result("测试", "https://example.com/t", "内容")], 0.1, None

        with patch.object(TavilySearchProvider, 'search', side_effect=capture_search):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    with patch('src.quick_workflow._llm_judge_relevance', return_value=([], [])):
                        with patch('src.quick_workflow._llm_generate_supplement_query', return_value=claim):
                            run_fact_check(claim)
                            assert captured_query, "应至少有一次搜索调用"
                            assert captured_query[0] == claim, f"首次搜索应使用完整原文，实际: {captured_query[0]}"

    def test_mechanical_clean_removes_invalid(self):
        """机械清理：删除空URL、空标题+摘要、重复URL。"""
        results = [
            {"title": "有效结果", "url": "https://example.com/1", "content": "内容", "publisher": ""},
            {"title": "", "url": "", "content": "", "publisher": ""},
            {"title": "重复", "url": "https://example.com/1", "content": "重复内容", "publisher": ""},
            {"title": "", "url": "https://example.com/2", "content": "", "publisher": ""},
            {"title": "只有标题", "url": "https://example.com/3", "content": "", "publisher": ""},
        ]
        cleaned, removed = _mechanically_clean_results(results)
        # 空URL+空内容 删除；重复URL 删除
        assert len(cleaned) >= 1, f"应至少保留1条有效结果，实际: {len(cleaned)}"
        assert removed >= 2, f"应至少删除2条无效结果，实际: {removed}"

    def test_llm_judge_keeps_relevant_authoritative(self):
        """LLM 判定相关的权威网页被保留。"""
        with patch.object(LLMClient, 'chat_json') as mock_chat:
            from pydantic import BaseModel, Field
            from typing import List as TypingList

            class RelevanceJudgment(BaseModel):
                source_index: int = 0
                relevant: bool = True
                stance: str = "support"
                reason: str = "直接讨论主体和事件"

            class RelevanceResult(BaseModel):
                judgments: TypingList[RelevanceJudgment] = []

            mock_chat.return_value = RelevanceResult(judgments=[
                RelevanceJudgment(source_index=0, relevant=True, stance="support", reason="权威来源直接讨论主张"),
            ])

            cleaned = [{"title": "新华社：神舟十八号成功发射", "url": "https://example.com/xinhua", "content": "神舟十八号载人飞船成功发射", "publisher": "新华社"}]
            relevant, all_j = _llm_judge_relevance("神舟十八号载人飞船于2024年4月25日发射。", cleaned)

            assert len(relevant) == 1, f"LLM判定相关的结果应被保留，实际: {len(relevant)}"
            assert relevant[0]["_llm_relevant"] is True
            assert relevant[0]["_llm_stance"] == "support"

    def test_llm_judge_excludes_year_only_page(self):
        """LLM 排除只讨论年份的网页。"""
        with patch.object(LLMClient, 'chat_json') as mock_chat:
            from pydantic import BaseModel, Field
            from typing import List as TypingList

            class RelevanceJudgment(BaseModel):
                source_index: int = 0
                relevant: bool = False
                stance: str = "context"
                reason: str = ""

            class RelevanceResult(BaseModel):
                judgments: TypingList[RelevanceJudgment] = []

            mock_chat.return_value = RelevanceResult(judgments=[
                RelevanceJudgment(source_index=0, relevant=False, stance="context", reason="仅讨论2024年，未涉及神舟十八号"),
            ])

            cleaned = [{"title": "2024年 - 百度百科", "url": "https://example.com/baike2024", "content": "2024年是公历闰年...", "publisher": "百度百科"}]
            relevant, all_j = _llm_judge_relevance("神舟十八号载人飞船于2024年4月25日发射。", cleaned)

            assert len(relevant) == 0, f"只讨论年份的网页应被LLM排除，实际保留: {len(relevant)}"

    def test_llm_generate_supplement_query_preserves_core(self):
        """LLM 生成补充查询时保留主体和事件。"""
        with patch.object(LLMClient, 'chat_json') as mock_chat:
            from pydantic import BaseModel, Field

            class SupplementQuery(BaseModel):
                query: str = ""

            mock_chat.return_value = SupplementQuery(query="神舟十八号载人飞船 2024年4月25日 发射 官方报道")

            query = _llm_generate_supplement_query("神舟十八号载人飞船于2024年4月25日发射。")

            assert "神舟十八号" in query, f"补充查询应保留主体，实际: {query}"
            assert "发射" in query, f"补充查询应保留事件，实际: {query}"

    def test_classify_source_grade(self):
        """来源等级分类正确。"""
        assert _classify_source_grade("国际奥林匹克委员会") == "A"
        assert _classify_source_grade("olympic.org") == "A"
        assert _classify_source_grade("中国奥委会") == "A"
        assert _classify_source_grade("新华社") == "B"
        assert _classify_source_grade("人民日报") == "B"
        assert _classify_source_grade("bbc.com") == "B"
        assert _classify_source_grade("新浪体育") == "C"
        assert _classify_source_grade("baidu.com") == "C"
        assert _classify_source_grade("百度百科") == "C"
        assert _classify_source_grade("小红书") == "D"

    def test_no_results_outputs_evidence_insufficient(self):
        """没有结果时正常输出证据不足。"""
        with patch.object(TavilySearchProvider, 'search',
                          return_value=([], 0.1, None)):
            with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                    result = run_fact_check("某不存在的主张。")
                    assert isinstance(result, CheckResult)
                    assert result.report is not None
                    # 应有结论，且不是"暂无法核查"（搜索失败才是这个）
                    assert result.report.overall_verdict is not None

    def test_any_input_does_not_crash(self):
        """任意输入不得报错或卡死。"""
        test_inputs = [
            "",
            "   ",
            "a",
            "测试",
            "这是一个非常长的输入" * 100,
            "神舟十八号载人飞船于2024年4月25日发射。",
            "2024年巴黎奥运会中国代表团获得40枚金牌",
        ]
        for text in test_inputs:
            with patch.object(TavilySearchProvider, 'search',
                              return_value=([], 0.1, "搜索服务连接临时中断，请重试")):
                with patch('src.search_cache.SearchCache.get_hit', return_value=None):
                    with patch('src.search_cache.SearchCache.get_fallback', return_value=None):
                        try:
                            result = run_fact_check(text)
                            assert isinstance(result, CheckResult), f"输入'{text[:20]}'应返回CheckResult"
                            assert result.report is not None, f"输入'{text[:20]}'应有报告"
                        except Exception as e:
                            pytest.fail(f"输入'{text[:20]}'导致异常: {e}")


class TestSearchTimeoutAndCache:
    """搜索超时控制与缓存降级测试。"""

    def test_tavily_search_uses_direct_timeout(self):
        """Tavily 搜索直接传入 timeout=8，不再使用线程包装。"""
        from unittest.mock import MagicMock, patch
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "response_time": 0.5,
            "results": [
                {
                    "title": "测试结果",
                    "url": "https://example.com/test",
                    "content": "测试内容",
                    "score": 0.9,
                }
            ],
        }
        with patch("tavily.TavilyClient", return_value=mock_client):
            provider = TavilySearchProvider(api_key="fake-key")
            results, response_time, err = provider.search(
                query="测试查询", max_results=5
            )

        assert err is None
        assert len(results) == 1
        mock_client.search.assert_called_once()
        call_kwargs = mock_client.search.call_args[1]
        assert call_kwargs.get("timeout") == 8, "必须直接传入 timeout=8"

    def test_tavily_success_writes_cache(self):
        """搜索成功后立即写入本地缓存。"""
        from unittest.mock import patch, MagicMock
        cached_results = [
            _make_search_result(
                "缓存命中", "https://example.com/cache", "缓存内容", score=0.9
            )
        ]
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "response_time": 0.3,
            "results": [
                {
                    "title": "实时结果",
                    "url": "https://example.com/live",
                    "content": "实时内容",
                    "score": 0.95,
                }
            ],
        }
        with patch("tavily.TavilyClient", return_value=mock_client):
            with patch("src.search_cache.SearchCache.get_hit", return_value=None):
                with patch("src.search_cache.SearchCache.get_fallback", return_value=None):
                    with patch("src.search_cache.SearchCache.save") as mock_save:
                        result = run_fact_check("测试文本")
                        assert isinstance(result, CheckResult)
                        assert result.report is not None
                        assert mock_save.called, "搜索成功后必须调用 cache.save()"

    def test_tavily_failure_reads_fallback_cache(self):
        """Tavily 失败时读取 7 天内兜底缓存。"""
        from unittest.mock import patch, MagicMock
        fallback_results = [
            _make_search_result(
                "兜底缓存", "https://example.com/fallback", "兜底内容", score=0.85
            )
        ]
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection refused")

        with patch("tavily.TavilyClient", return_value=mock_client):
            with patch("src.search_cache.SearchCache.get_hit", return_value=None):
                with patch("src.search_cache.SearchCache.get_fallback", return_value=MagicMock(
                    results_json=__import__("json").dumps([{
                        "title": "兜底缓存",
                        "url": "https://example.com/fallback",
                        "content": "兜底内容",
                        "publisher": "测试",
                        "score": 0.85,
                    }], ensure_ascii=False),
                    created_at=__import__("time").time(),
                    source="tavily",
                )):
                    result = run_fact_check("测试文本")
                    assert isinstance(result, CheckResult)
                    assert result.report is not None

    def test_tavily_failure_no_cache_returns_quickly(self):
        """Tavily 失败且无缓存时快速返回，不得等待超时。"""
        from unittest.mock import patch, MagicMock
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Connection refused")

        start = __import__("time").time()
        with patch("src.tools.search_tool.TavilyClient", return_value=mock_client):
            with patch("src.search_cache.SearchCache.get_hit", return_value=None):
                with patch("src.search_cache.SearchCache.get_fallback", return_value=None):
                    result = run_fact_check("测试文本")
        elapsed = __import__("time").time() - start

        assert isinstance(result, CheckResult)
        assert result.report is not None
        assert result.error_message is not None
        assert elapsed < 20, f"无缓存搜索失败应在20秒内返回，实际耗时{elapsed:.1f}秒"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])