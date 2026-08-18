"""搜索工具单元测试。

测试 Tavily 结果解析、空结果、超时、401/429 错误、URL 去重、同源转载识别。
所有测试 mock Tavily 和 LLM，不真实消耗 API 额度。
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.tools.search_tool import (
    MockSearchProvider,
    SearchResult,
    TavilySearchProvider,
    choose_topic_for_claim,
    deduplicate_by_url,
    get_search_provider,
    normalize_url,
    search_web,
)


# ============ Tavily 结果解析 ============


def test_tavily_parses_response_correctly():
    """测试 Tavily 正常响应被正确解析为 SearchResult。"""
    fake_response = {
        "response_time": 1.23,
        "results": [
            {
                "title": "国家航天局：嫦娥六号返回舱成功着陆",
                "url": "https://www.cnsa.gov.cn/article/2024/06/change6.html",
                "content": "2024年6月25日，嫦娥六号返回舱在内蒙古四子王旗着陆场成功着陆。",
                "score": 0.95,
                "published_date": "2024-06-25T10:00:00",
            },
            {
                "title": "新华社：嫦娥六号完成采样返回",
                "url": "https://www.xinhuanet.com/science/2024-06/change6.html",
                "content": "嫦娥六号任务实现了人类首次月球背面采样返回。",
                "score": 0.88,
            },
        ],
    }

    provider = TavilySearchProvider(api_key="fake-key-for-testing")
    with patch("src.tools.search_tool.TavilyClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.search.return_value = fake_response
        mock_client_cls.return_value = mock_client

        results, response_time, err = provider.search(
            query="嫦娥六号 返回 地球", max_results=5, topic="news"
        )

    assert err is None
    assert response_time == 1.23
    assert len(results) == 2
    assert "嫦娥六号" in results[0].title
    assert "cnsa.gov.cn" in results[0].domain
    assert results[0].published_at is not None
    assert results[1].published_at is None  # 没有该字段


def test_tavily_empty_results():
    """测试 Tavily 返回空结果。"""
    fake_response = {"response_time": 0.5, "results": []}
    provider = TavilySearchProvider(api_key="fake-key-for-testing")
    with patch("src.tools.search_tool.TavilyClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.search.return_value = fake_response
        mock_client_cls.return_value = mock_client

        results, response_time, err = provider.search(query="测试", max_results=5)

    assert err is None
    assert len(results) == 0
    assert response_time == 0.5


def test_tavily_timeout_error():
    """测试 Tavily 超时错误被正确识别。"""
    provider = TavilySearchProvider(api_key="fake-key-for-testing")
    with patch("src.tools.search_tool.TavilyClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Request timed out after 30 seconds")
        mock_client_cls.return_value = mock_client

        results, response_time, err = provider.search(query="测试", max_results=5)

    assert err is not None
    assert "超时" in err
    assert len(results) == 0


def test_tavily_401_unauthorized():
    """测试 Tavily 401 无效密钥错误。"""
    provider = TavilySearchProvider(api_key="invalid-key")
    with patch("src.tools.search_tool.TavilyClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("401 Unauthorized: Invalid API key")
        mock_client_cls.return_value = mock_client

        results, _, err = provider.search(query="测试", max_results=5)

    assert err is not None
    assert "401" in err
    assert len(results) == 0


def test_tavily_429_rate_limit():
    """测试 Tavily 429 额度限制错误。"""
    provider = TavilySearchProvider(api_key="fake-key-for-testing")
    with patch("src.tools.search_tool.TavilyClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("429 Rate limit exceeded")
        mock_client_cls.return_value = mock_client

        results, _, err = provider.search(query="测试", max_results=5)

    assert err is not None
    assert "429" in err
    assert len(results) == 0


def test_tavily_not_configured():
    """测试未配置 TAVILY_API_KEY 时返回错误。"""
    provider = TavilySearchProvider(api_key="")
    results, _, err = provider.search(query="测试", max_results=5)

    assert err is not None
    assert "TAVILY_API_KEY" in err
    assert len(results) == 0


# ============ URL 规范化与去重 ============


def test_normalize_url_strips_fragment():
    """测试 URL 规范化去除 fragment。"""
    assert normalize_url("https://example.com/page#section") == "https://example.com/page"
    assert normalize_url("https://example.com/page/") == "https://example.com/page"
    assert normalize_url("HTTPS://EXAMPLE.COM/Page") == "https://example.com/page"


def test_normalize_url_strips_tracking_params():
    """测试 URL 规范化去除跟踪参数。"""
    assert (
        normalize_url("https://example.com/page?utm_source=abc&content=xyz")
        == "https://example.com/page?content=xyz"
    )
    assert normalize_url("https://example.com/page?ref=abc") == "https://example.com/page"


def test_deduplicate_by_url():
    """测试按 URL 去重。"""
    results = [
        SearchResult(title="A", url="https://example.com/page1", content="content1"),
        SearchResult(title="A-dup", url="https://example.com/page1#section", content="content1"),
        SearchResult(title="B", url="https://example.com/page2", content="content2"),
    ]
    unique = deduplicate_by_url(results)
    assert len(unique) == 2
    assert unique[0].title == "A"
    assert unique[1].title == "B"


# ============ topic 选择 ============


def test_choose_topic_news_for_disaster():
    """测试灾害新闻选择 topic=news。"""
    assert choose_topic_for_claim("某市因暴雨导致地铁停运", "事件陈述") == "news"
    assert choose_topic_for_claim("地震造成人员伤亡", "事件陈述") == "news"


def test_choose_topic_general_for_other():
    """测试非新闻选择 topic=general。"""
    assert choose_topic_for_claim("某公司发布财报", "数据声明") == "news"  # 有"发布"关键词
    assert choose_topic_for_claim("某明星代言某品牌", "其他") == "general"


# ============ 工厂方法 ============


def test_get_search_provider_demo():
    """测试 demo 模式返回 MockSearchProvider。"""
    provider = get_search_provider("demo")
    assert isinstance(provider, MockSearchProvider)


def test_get_search_provider_full():
    """测试 full 模式返回 TavilySearchProvider。"""
    provider = get_search_provider("full")
    assert isinstance(provider, TavilySearchProvider)


# ============ MockSearchProvider 行为 ============


def test_mock_search_returns_demo_data():
    """测试 Mock 搜索返回演示数据。"""
    provider = MockSearchProvider()
    results, _, err = provider.search(query="地铁 暴雨 停运", max_results=3)
    assert err is None
    assert len(results) > 0
    assert "轨道交通集团" in results[0].publisher


def test_mock_search_generic_fallback():
    """测试 Mock 搜索无匹配关键词时返回通用演示数据。"""
    provider = MockSearchProvider()
    results, _, _ = provider.search(query="测试无匹配", max_results=3)
    assert len(results) == 1
    assert "模拟" in results[0].title


# ============ 旧版 search_web 兼容 ============


def test_search_web_returns_evidences():
    """测试旧版 search_web 接口仍可工作。"""
    evidences = search_web(query="地铁 暴雨", claim_id="c1", max_results=2)
    assert len(evidences) > 0
    assert evidences[0].claim_id == "c1"
    assert evidences[0].source_grade in ("A", "B", "C", "D", "E")


# ============ 同源转载识别（在 search.py 节点中测试） ============


def test_independence_grouping_by_domain():
    """测试同域名归为同一独立分组（通过 _count_independent_sources）。"""
    from src.nodes.search import _count_independent_sources
    from src.models import Evidence

    evidences = {
        "c1": [
            Evidence(
                evidence_id="c1-e1", claim_id="c1",
                source_title="报道A", source_url="https://news1.com/a",
                publisher="news1", evidence_summary="...",
                source_type="媒体", source_grade="B",
                supports_or_refutes="supports",
                reliability_reason="测试",
                source_domain="news1.com",
                extraction_status="success",
                evidence_stance="supports",
            ),
            Evidence(
                evidence_id="c1-e2", claim_id="c1",
                source_title="报道B（同域名转载）", source_url="https://news1.com/b",
                publisher="news1", evidence_summary="...",
                source_type="媒体", source_grade="C",
                supports_or_refutes="supports",
                reliability_reason="测试",
                source_domain="news1.com",
                extraction_status="success",
                evidence_stance="supports",
            ),
            Evidence(
                evidence_id="c1-e3", claim_id="c1",
                source_title="报道C", source_url="https://news2.com/c",
                publisher="news2", evidence_summary="...",
                source_type="媒体", source_grade="B",
                supports_or_refutes="refutes",
                reliability_reason="测试",
                source_domain="news2.com",
                extraction_status="success",
                evidence_stance="refutes",
            ),
        ]
    }
    count = _count_independent_sources(evidences)
    assert count == 2  # news1.com 和 news2.com 两个独立来源
