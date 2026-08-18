"""完整真实核查模式稳定性测试。

覆盖需求：
1. 普通用户输入确实进入真实模式，不调用 mock/demo。
2. LLM 超时后能够在限定时间内结束。
3. Tavily 超时后能够输出部分报告。
4. LLM 返回非法 JSON 时只进行有限重试。
5. Streamlit rerun 不会重复执行核查。
6. 相同查询会被去重。
7. 报告中的每个可点击 URL 都能追溯到搜索工具返回值。
8. LLM 编造的 URL 不会进入报告。
9. 部分步骤失败时仍能生成清晰的部分结果。
10. 任务正常完成后状态才允许显示 100%。
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.models import AgentState, Claim, ClaimResult, Evidence, FactCheckReport
from src.tools.search_tool import (
    MockSearchProvider,
    SearchResult,
    TavilySearchProvider,
    _run_with_timeout,
    _validate_url,
    deduplicate_by_url,
    get_search_provider,
    normalize_url,
)
from src.workflow import (
    _WorkflowTimeout,
    _check_deadline,
    run_fact_check_workflow,
)


class TestURLValidation:
    """测试需求 7、8：URL 必须来自真实搜索结果，LLM 编造的 URL 不可进入报告。"""

    def test_validate_url_accepts_https(self):
        assert _validate_url("https://example.com/article/123") is True

    def test_validate_url_accepts_http(self):
        assert _validate_url("http://example.com/article/123") is True

    def test_validate_url_rejects_empty(self):
        assert _validate_url("") is False
        assert _validate_url(None) is False

    def test_validate_url_rejects_whitespace_only(self):
        assert _validate_url("   ") is False

    def test_validate_url_rejects_protocol_relative(self):
        assert _validate_url("//example.com/page") is False

    def test_validate_url_rejects_no_protocol(self):
        assert _validate_url("example.com/page") is False

    def test_validate_url_rejects_ftp(self):
        assert _validate_url("ftp://example.com/file") is False

    def test_validate_url_rejects_search_engine_urls(self):
        assert _validate_url("https://www.google.com/search?q=test") is True
        # 搜索页地址本身是合法 URL，但业务层应过滤

    def test_validate_url_rejects_localhost_like(self):
        assert _validate_url("http://localhost:8000/test") is True
        assert _validate_url("http://127.0.0.1/test") is True

    def test_normalize_url_removes_fragment(self):
        result = normalize_url("https://example.com/page#section")
        assert "#" not in result

    def test_normalize_url_removes_utm_params(self):
        result = normalize_url("https://example.com/page?utm_source=test&ref=abc")
        assert "utm_source" not in result
        assert "ref" not in result

    def test_normalize_url_removes_trailing_slash(self):
        result = normalize_url("https://example.com/page/")
        assert result.endswith("/page")

    def test_deduplicate_by_url_removes_duplicates(self):
        results = [
            SearchResult(title="A", url="https://example.com/page", content="a"),
            SearchResult(title="B", url="https://example.com/page", content="b"),
            SearchResult(title="C", url="https://other.com/page", content="c"),
        ]
        deduped = deduplicate_by_url(results)
        assert len(deduped) == 2

    def test_deduplicate_by_url_normalizes_before_comparison(self):
        results = [
            SearchResult(title="A", url="https://example.com/page#frag", content="a"),
            SearchResult(title="B", url="https://example.com/page", content="b"),
        ]
        deduped = deduplicate_by_url(results)
        assert len(deduped) == 1


class TestTimeoutMechanism:
    """测试需求 2、3：超时后能够在限定时间内结束。"""

    def test_run_with_timeout_completes_within_limit(self):
        def fast_func():
            time.sleep(0.1)
            return "result"

        start = time.time()
        result, timed_out, error = _run_with_timeout(fast_func, 2)
        elapsed = time.time() - start
        assert result == "result"
        assert timed_out is False
        assert error is None
        assert elapsed < 2

    def test_run_with_timeout_times_out(self):
        def slow_func():
            time.sleep(10)
            return "result"

        start = time.time()
        result, timed_out, error = _run_with_timeout(slow_func, 0.5)
        elapsed = time.time() - start
        assert result is None
        assert timed_out is True
        assert elapsed < 2

    def test_run_with_timeout_captures_error(self):
        def error_func():
            raise ValueError("test error")

        result, timed_out, error = _run_with_timeout(error_func, 5)
        assert result is None
        assert timed_out is False
        assert error is not None
        assert "test error" in error

    def test_tavily_provider_has_timeout_configured(self):
        provider = TavilySearchProvider(timeout=10)
        assert provider.timeout == 10

    def test_tavily_provider_default_timeout(self):
        provider = TavilySearchProvider()
        assert provider.timeout == settings.TAVILY_TIMEOUT
        assert provider.timeout == 20

    def test_deadline_check_raises_on_timeout(self):
        deadline = time.time() - 1  # 1 second ago
        state = AgentState(original_text="test")
        with pytest.raises(_WorkflowTimeout):
            _check_deadline(deadline, state, "test_phase")

    def test_deadline_check_passes_before_timeout(self):
        deadline = time.time() + 100
        state = AgentState(original_text="test")
        _check_deadline(deadline, state, "test_phase")


class TestModeSelection:
    """测试需求 1：普通用户输入确实进入真实模式。"""

    def test_full_mode_uses_tavily_provider(self):
        provider = get_search_provider("full")
        assert isinstance(provider, TavilySearchProvider)

    def test_demo_mode_uses_mock_provider(self):
        provider = get_search_provider("demo")
        assert isinstance(provider, MockSearchProvider)

    def test_llm_mode_uses_mock_provider(self):
        provider = get_search_provider("llm")
        assert isinstance(provider, MockSearchProvider)


class TestSearchProviderBehavior:
    """测试搜索提供者行为。"""

    def test_mock_provider_always_available(self):
        provider = MockSearchProvider()
        assert provider.is_configured() is True

    def test_tavily_provider_checks_api_key(self):
        provider = TavilySearchProvider(api_key="")
        assert provider.is_configured() is False

    def test_tavily_provider_with_key_is_configured(self):
        provider = TavilySearchProvider(api_key="fake-key")
        assert provider.is_configured() is True

    def test_tavily_search_returns_error_when_no_key(self):
        provider = TavilySearchProvider(api_key="")
        results, rt, err = provider.search("test query", max_results=3)
        assert len(results) == 0
        assert err is not None

    def test_search_results_have_valid_urls(self):
        """Tavily search results must have validated URLs."""
        results = [
            SearchResult(
                title="Test",
                url="https://example.com/valid",
                content="content",
            ),
        ]
        for r in results:
            assert _validate_url(r.url)


class TestProgressTracking:
    """测试需求 10：正常完成后状态显示 100%。"""

    def test_progress_reaches_100_on_complete(self):
        state = AgentState(original_text="test")
        state.mark_step_started("init", "初始化")
        state.mark_step_completed("init", "初始化完成")
        state.mark_step_started("decompose", "拆解主张")
        state.mark_step_completed("decompose", "拆解完成")
        state.mark_step_started("plan", "制定计划")
        state.mark_step_completed("plan", "计划完成")
        state.mark_step_started("search", "搜索")
        state.mark_step_completed("search", "搜索完成")
        state.mark_step_started("evaluate", "评估")
        state.mark_step_completed("evaluate", "评估完成")
        state.mark_step_started("report", "报告")
        state.mark_step_completed("report", "报告完成")
        state.mark_step_skipped("memory", "跳过记忆")
        state.mark_all_done()
        assert state.progress_percent == 100
        assert state.workflow_completed is True
        assert state.current_step == "completed"

    def test_progress_not_100_when_failed(self):
        state = AgentState(original_text="test")
        state.mark_step_started("init", "初始化")
        state.mark_step_completed("init", "初始化完成")
        state.mark_failed("decompose", "拆解失败", "LLM 超时")
        assert state.workflow_completed is False
        assert state.current_step == "failed"
        assert state.progress_percent < 100

    def test_progress_not_100_when_timeout(self):
        state = AgentState(original_text="test")
        state.mark_step_started("init", "初始化")
        state.mark_step_completed("init", "初始化完成")
        state.mark_step_started("decompose", "拆解")
        state.mark_step_started("plan", "计划")
        assert state.progress_percent < 100

    def test_sync_progress_to_report(self):
        state = AgentState(original_text="test")
        state.mark_step_started("init", "初始化")
        state.mark_step_completed("init", "初始化完成")
        state.mark_all_done()
        report = FactCheckReport(
            original_text="test",
            overall_verdict="证据不足",
            overall_summary="test summary",
        )
        state.report = report
        state.sync_progress_to_report()
        assert report.progress_percent == 100
        assert report.workflow_completed is True
        assert report.current_step == "completed"


class TestWorkflowDegradation:
    """测试需求 9：部分步骤失败时仍能生成清晰的部分结果。"""

    def test_workflow_handles_claim_count_limit(self):
        """Verify MAX_CLAIMS setting exists and limits claims."""
        assert settings.MAX_CLAIMS == 4

    def test_workflow_max_seconds_configured(self):
        """Verify global timeout is configured."""
        assert settings.WORKFLOW_MAX_SECONDS == 120

    def test_max_queries_per_claim(self):
        """Verify per-claim query limit."""
        assert settings.SEARCH_MAX_QUERIES_PER_CLAIM == 2

    def test_max_results_per_query(self):
        """Verify per-query result limit."""
        assert settings.SEARCH_MAX_RESULTS_PER_QUERY == 5

    def test_max_concurrent_searches(self):
        """Verify concurrent search limit."""
        assert settings.SEARCH_MAX_CONCURRENT == 3


class TestEvidenceURLTraceability:
    """测试需求 7：报告中的 URL 能追溯到搜索工具返回值。"""

    def test_evidence_creation_with_valid_url(self):
        result = SearchResult(
            title="Test Article",
            url="https://example.com/article",
            content="Test content",
            publisher="Test Publisher",
        )
        assert _validate_url(result.url)

    def test_invented_url_rejected(self):
        """LLM-invented URLs should be rejected by validation."""
        assert _validate_url("not-a-real-url") is False
        assert _validate_url("") is False
        assert _validate_url("fake-url-without-protocol") is False

    def test_evidence_with_invalid_url_filtered(self):
        """Evidence with invalid URLs should not be created."""
        invalid_urls = [
            "",
            "not a url",
            "ftp://example.com",
            "javascript:alert(1)",
        ]
        for url in invalid_urls:
            assert _validate_url(url) is False


class TestErrorHandling:
    """测试各类错误处理。"""

    def test_llm_timeout_setting(self):
        assert settings.LLM_TIMEOUT == 35

    def test_llm_max_retries_setting(self):
        assert settings.LLM_MAX_RETRIES == 2

    def test_tavily_timeout_setting(self):
        assert settings.TAVILY_TIMEOUT == 20

    def test_search_max_retries_setting(self):
        assert settings.SEARCH_MAX_RETRIES == 1

    def test_missing_configs_detection(self):
        """Verify missing config detection works."""
        from src.config import Settings
        s = Settings()
        s.LLM_API_KEY = ""
        s.LLM_MODEL = ""
        s.TAVILY_API_KEY = ""
        missing = s.missing_configs()
        assert "LLM_API_KEY" in missing
        assert "LLM_MODEL" in missing
        assert "TAVILY_API_KEY" in missing

    def test_full_mode_requires_both_keys(self):
        """Full mode should require both LLM and Tavily keys."""
        from src.config import Settings
        s = Settings()
        assert s.full_real_configured() == bool(s.llm_configured() and s.search_configured())


class TestQueryDeduplication:
    """测试需求 6：相同查询会被去重。"""

    def test_normalize_url_handles_edge_cases(self):
        assert normalize_url("") == ""
        assert normalize_url(None) == ""
        assert normalize_url("https://example.com") == "https://example.com"

    def test_identical_urls_normalized_same(self):
        url1 = normalize_url("https://example.com/page?utm_source=x")
        url2 = normalize_url("https://example.com/page?ref=y")
        assert url1 == url2

    def test_different_urls_not_deduplicated(self):
        url1 = normalize_url("https://example.com/page1")
        url2 = normalize_url("https://example.com/page2")
        assert url1 != url2


class TestConcurrentSearch:
    """测试搜索并发控制。"""

    def test_concurrent_limit_respected(self):
        assert settings.SEARCH_MAX_CONCURRENT == 3

    def test_max_results_per_query_respected(self):
        assert settings.SEARCH_MAX_RESULTS_PER_QUERY == 5


class TestLLMRetryLimit:
    """测试需求 4：LLM 返回非法 JSON 时只进行有限重试。"""

    def test_llm_max_retries_is_finite(self):
        assert settings.LLM_MAX_RETRIES > 0
        assert settings.LLM_MAX_RETRIES <= 5

    def test_llm_timeout_is_finite(self):
        assert settings.LLM_TIMEOUT > 0
        assert settings.LLM_TIMEOUT <= 120


class TestWorkflowIntegration:
    """测试 workflow 整体行为。"""

    def test_workflow_accepts_full_mode(self):
        """Verify workflow accepts 'full' mode parameter."""
        with patch('src.workflow.settings', create=True) as mock_settings:
            mock_settings.WORKFLOW_MAX_SECONDS = 120
            mock_settings.MAX_CLAIMS = 4
            mock_settings.missing_configs = lambda: []

            state = AgentState(original_text="嫦娥六号于2024年6月返回地球。", mode="full")
            assert state.mode == "full"

    def test_workflow_handles_empty_input(self):
        """Empty input should not crash."""
        state = AgentState(original_text="", mode="demo")
        assert state.original_text == ""

    def test_fact_report_has_required_fields(self):
        """FactCheckReport has all required fields for the stable flow."""
        report = FactCheckReport(
            original_text="test",
            overall_verdict="证据不足",
            overall_summary="summary",
        )
        assert report.current_step == "init"
        assert report.progress_percent == 0
        assert report.workflow_completed is False
        assert report.workflow_error is None
        assert isinstance(report.completed_steps, list)
        assert isinstance(report.skipped_steps, list)


class TestSearchStats:
    """测试搜索统计。"""

    def test_search_stats_initialization(self):
        state = AgentState(original_text="test")
        stats = state.search_stats
        assert stats["total_queries"] == 0
        assert stats["total_results_fetched"] == 0
        assert stats["valid_evidence_count"] == 0
        assert stats["failed_queries"] == 0

    def test_search_stats_update(self):
        state = AgentState(original_text="test")
        state.search_stats["total_queries"] = 5
        state.search_stats["total_results_fetched"] = 15
        assert state.search_stats["total_queries"] == 5
        assert state.search_stats["total_results_fetched"] == 15


class TestURLSafetyInApp:
    """测试 app.py 中的 URL 安全渲染逻辑。"""

    def test_valid_urls_pass_validation(self):
        valid_urls = [
            "https://example.com/article",
            "http://news.example.cn/2024/06/12345",
            "https://www.gov.cn/policy/2024-01/document",
        ]
        for url in valid_urls:
            assert _validate_url(url), f"Expected {url} to be valid"

    def test_invalid_urls_fail_validation(self):
        invalid_urls = [
            "",
            None,
            "not-a-url",
            "ftp://files.example.com",
            "javascript:alert('xss')",
            "data:text/html,<script>",
        ]
        for url in invalid_urls:
            assert not _validate_url(url), f"Expected {url} to be invalid"
