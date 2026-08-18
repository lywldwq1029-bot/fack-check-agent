"""搜索稳定性测试 - 覆盖重试逻辑和缓存场景。

测试场景：
1. 第一次 SSL 失败、第二次成功
2. 三次均失败但命中缓存
3. 三次均失败且无缓存
4. 401 不重试
5. 缓存 URL 保持原始值
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.search_cache import SearchCache, CacheEntry, HIT_TTL_SECONDS, FALLBACK_TTL_SECONDS
from src.tools.search_tool import TavilySearchProvider, SearchResult
from src.models import build_no_evidence_report, FactCheckReport


class TestSearchCache(unittest.TestCase):
    """测试 SQLite 搜索缓存。"""

    def setUp(self):
        """每个测试使用独立的临时数据库。"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_cache.sqlite")
        self.cache = SearchCache(db_path=self.db_path)

    def tearDown(self):
        """清理临时数据库。"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except (OSError, IOError):
            pass

    def _make_results(self, query: str) -> list[SearchResult]:
        """创建测试用搜索结果。"""
        return [
            SearchResult(
                title=f"Test Result for {query}",
                url=f"https://example.com/{query.replace(' ', '-')}",
                content=f"This is test content for query: {query}",
                publisher="Example Publisher",
                score=0.9,
            )
        ]

    def test_save_and_hit(self):
        """测试保存后立即命中。"""
        query = "test query"
        results = self._make_results(query)
        self.cache.save(query, results)

        entry = self.cache.get_hit(query)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.normalized_query, "test query")

        restored = SearchCache.entry_to_results(entry)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].url, "https://example.com/test-query")
        self.assertEqual(restored[0].title, "Test Result for test query")

    def test_hit_ttl_expired(self):
        """测试 24 小时后直接命中过期。"""
        query = "expired query"
        results = self._make_results(query)
        self.cache.save(query, results)

        # 手动修改创建时间为 25 小时前
        with __import__("sqlite3").connect(self.db_path) as conn:
            conn.execute(
                "UPDATE search_cache SET created_at = ? WHERE normalized_query = ?",
                (time.time() - 25 * 3600, "expired query")
            )
            conn.commit()

        # 25 小时后，直接命中应该过期
        entry = self.cache.get_hit(query)
        self.assertIsNone(entry)

        # 但 72 小时内仍可作为兜底
        entry = self.cache.get_fallback(query)
        self.assertIsNotNone(entry)

    def test_fallback_ttl_expired(self):
        """测试 72 小时后兜底也过期。"""
        query = "old query"
        results = self._make_results(query)
        self.cache.save(query, results)

        # 手动修改创建时间为 73 小时前
        with __import__("sqlite3").connect(self.db_path) as conn:
            conn.execute(
                "UPDATE search_cache SET created_at = ? WHERE normalized_query = ?",
                (time.time() - 73 * 3600, "old query")
            )
            conn.commit()

        entry = self.cache.get_fallback(query)
        self.assertIsNone(entry)

    def test_url_preserved(self):
        """测试缓存 URL 保持原始值。"""
        query = "url test"
        original_url = "https://example.com/original-url?param=1&key=value"
        results = [
            SearchResult(
                title="URL Test",
                url=original_url,
                content="Testing URL preservation",
                publisher="Test",
                score=0.5,
            )
        ]
        self.cache.save(query, results)

        entry = self.cache.get_hit(query)
        restored = SearchCache.entry_to_results(entry)
        self.assertEqual(restored[0].url, original_url)

    def test_normalize_query(self):
        """测试查询标准化。"""
        key1 = SearchCache._normalize_query("Hello World")
        key2 = SearchCache._normalize_query("hello   world")
        self.assertEqual(key1, key2)

        key3 = SearchCache._normalize_query("你好  世界")
        key4 = SearchCache._normalize_query("你好 世界")
        self.assertEqual(key3, key4)


class TestRetryLogic(unittest.TestCase):
    """测试 Tavily 重试逻辑。"""

    def test_is_retryable_error_ssl(self):
        """测试 SSL 错误可重试。"""
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "SSLEOFError: SSL connection has been closed unexpectedly"
        ))
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "ssl.SSLError: certificate verify failed"
        ))

    def test_is_retryable_error_connection(self):
        """测试连接错误可重试。"""
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "ConnectionError: Max retries exceeded"
        ))
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "ConnectionResetError: Connection reset by peer"
        ))

    def test_is_retryable_error_5xx(self):
        """测试 5xx 错误可重试。"""
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "HTTPError: 500 Internal Server Error"
        ))
        self.assertTrue(TavilySearchProvider._is_retryable_error(
            "HTTPError: 503 Service Unavailable"
        ))

    def test_is_not_retryable_401(self):
        """测试 401 不重试。"""
        self.assertFalse(TavilySearchProvider._is_retryable_error(
            "HTTPError: 401 Unauthorized"
        ))

    def test_is_not_retryable_403(self):
        """测试 403 不重试。"""
        self.assertFalse(TavilySearchProvider._is_retryable_error(
            "HTTPError: 403 Forbidden"
        ))

    def test_is_not_retryable_429(self):
        """测试 429 不重试。"""
        self.assertFalse(TavilySearchProvider._is_retryable_error(
            "HTTPError: 429 Too Many Requests"
        ))

    def test_classify_error_401(self):
        """测试 401 错误分类。"""
        msg = TavilySearchProvider._classify_error("HTTPError: 401 Unauthorized")
        self.assertIn("密钥无效", msg)

    def test_classify_error_429(self):
        """测试 429 错误分类。"""
        msg = TavilySearchProvider._classify_error("HTTPError: 429 Too Many Requests")
        self.assertIn("额度", msg)

    def test_classify_error_ssl(self):
        """测试 SSL 错误分类。"""
        msg = TavilySearchProvider._classify_error("SSLEOFError: SSL connection failed")
        self.assertIn("连接临时中断", msg)


class TestNoEvidenceReport(unittest.TestCase):
    """测试优雅降级报告。"""

    def test_build_no_evidence_report(self):
        """测试构建无证据报告。"""
        report = build_no_evidence_report("测试文本")
        self.assertIsInstance(report, FactCheckReport)
        self.assertEqual(report.overall_verdict, "暂无法核查")
        self.assertEqual(report.current_step, "completed")
        self.assertTrue(report.workflow_completed)
        self.assertIsNone(report.workflow_error)
        self.assertEqual(report.progress_percent, 50)
        self.assertIn("receive", report.completed_steps)
        self.assertIn("search", report.completed_steps)
        self.assertIn("analyze", report.skipped_steps)
        self.assertIn("output", report.skipped_steps)
        self.assertEqual(report.claim_results, [])


class TestQuickWorkflowCache(unittest.TestCase):
    """测试 quick_workflow 缓存集成。"""

    def setUp(self):
        """设置测试环境。"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_cache.sqlite")

    def tearDown(self):
        """清理临时数据库。"""
        try:
            os.unlink(self.db_path)
            os.rmdir(self.temp_dir)
        except (OSError, IOError):
            pass

    @patch("src.quick_workflow.TavilySearchProvider")
    def test_first_ssl_fail_second_success(self, MockProvider):
        """场景1: 第一次 SSL 失败、第二次成功。"""
        mock_instance = MockProvider.return_value
        # 第一次调用返回 SSL 错误
        mock_instance.search.side_effect = [
            ([], 0.0, "SSLEOFError: SSL connection failed"),
            # 第二次调用成功
            (
                [SearchResult(
                    title="Test",
                    url="https://example.com/test",
                    content="Test content",
                    publisher="Test",
                    score=0.9,
                )],
                0.5,
                None,
            ),
        ]

        from src.quick_workflow import _search_once, _get_cache
        # 使用临时缓存
        import src.quick_workflow as qw
        original_cache = qw._cache
        qw._cache = SearchCache(db_path=self.db_path)

        try:
            results, err, cache_info = _search_once("test query")
            self.assertIsNone(err)
            self.assertEqual(cache_info, "live")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["url"], "https://example.com/test")
        finally:
            qw._cache = original_cache

    @patch("src.quick_workflow.TavilySearchProvider")
    def test_all_fail_but_hit_cache(self, MockProvider):
        """场景2: 三次均失败但命中缓存。"""
        mock_instance = MockProvider.return_value
        mock_instance.search.return_value = ([], 0.0, "SSLEOFError: SSL connection failed")

        from src.quick_workflow import _search_once
        cache = SearchCache(db_path=self.db_path)

        # 先写入缓存
        cache.save("test query", [
            SearchResult(
                title="Cached Result",
                url="https://example.com/cached",
                content="Cached content",
                publisher="Cache",
                score=0.8,
            )
        ])

        import src.quick_workflow as qw
        original_cache = qw._cache
        qw._cache = cache

        try:
            results, err, cache_info = _search_once("test query")
            # 应该命中 24h 缓存
            self.assertIsNone(err)
            self.assertIn(cache_info, ("hit_24h", "fallback_72h"))
            self.assertEqual(len(results), 1)
        finally:
            qw._cache = original_cache

    @patch("src.quick_workflow.TavilySearchProvider")
    def test_all_fail_no_cache(self, MockProvider):
        """场景3: 三次均失败且无缓存。"""
        mock_instance = MockProvider.return_value
        mock_instance.search.return_value = ([], 0.0, "SSLEOFError: SSL connection failed")

        from src.quick_workflow import _search_once
        cache = SearchCache(db_path=self.db_path)

        import src.quick_workflow as qw
        original_cache = qw._cache
        qw._cache = cache

        try:
            results, err, cache_info = _search_once("no cache query")
            self.assertIsNotNone(err)
            self.assertEqual(cache_info, "none")
            self.assertEqual(results, [])
        finally:
            qw._cache = original_cache

    @patch("src.quick_workflow.TavilySearchProvider")
    def test_401_no_retry(self, MockProvider):
        """场景4: 401 不重试。"""
        mock_instance = MockProvider.return_value
        mock_instance.search.return_value = ([], 0.0, "HTTPError: 401 Unauthorized")

        from src.quick_workflow import _search_once
        cache = SearchCache(db_path=self.db_path)

        import src.quick_workflow as qw
        original_cache = qw._cache
        qw._cache = cache

        try:
            results, err, cache_info = _search_once("401 query")
            # 应该直接返回 401 错误，不重试
            self.assertIsNotNone(err)
            self.assertIn("密钥无效", err)
            self.assertEqual(cache_info, "none")
            # 只应该调用一次（401 不重试）
            self.assertEqual(MockProvider.return_value.search.call_count, 1)
        finally:
            qw._cache = original_cache

    def test_cache_url_preserved(self):
        """场景5: 缓存 URL 保持原始值。"""
        cache = SearchCache(db_path=self.db_path)
        original_url = "https://example.com/original?data=test&key=value"

        results = [SearchResult(
            title="URL Test",
            url=original_url,
            content="Testing",
            publisher="Test",
            score=0.5,
        )]

        cache.save("url query", results)
        entry = cache.get_hit("url query")
        restored = SearchCache.entry_to_results(entry)

        self.assertEqual(restored[0].url, original_url)


if __name__ == "__main__":
    # 运行测试
    unittest.main(verbosity=2)
