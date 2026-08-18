"""搜索工具接口。

提供统一的 SearchProvider 抽象接口：
- TavilySearchProvider：真实 Tavily 搜索（直接 timeout、自动重试、≤18s）
- MockSearchProvider：演示模式模拟数据

通过 get_search_provider 工厂方法根据模式获取对应实现。
保持旧版 search_web 函数以兼容现有 search.py 节点。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
import time

from src.config import settings
from src.models import Evidence


class SearchResult:
    """搜索返回的单条原始网页结果（尚未提取证据）。"""

    def __init__(
        self,
        title: str,
        url: str,
        content: str,
        publisher: str = "",
        published_at: Optional[datetime] = None,
        score: float = 0.0,
        raw_response: Optional[dict] = None,
    ) -> None:
        self.title = title
        self.url = url
        self.content = content
        self.publisher = publisher
        self.published_at = published_at
        self.score = score
        self.raw_response = raw_response or {}

    @property
    def domain(self) -> str:
        try:
            parsed = urlparse(self.url)
            return parsed.netloc.lower()
        except Exception:
            return ""


class SearchProvider(ABC):
    """搜索服务提供方抽象接口。"""

    @abstractmethod
    def search(
        self,
        query: str,
        max_results: int,
        topic: str = "general",
    ) -> tuple[list[SearchResult], float, Optional[str]]:
        ...

    def extract(self, urls: list[str]) -> dict[str, str]:
        return {}

    @abstractmethod
    def is_configured(self) -> bool:
        ...


def _validate_url(url: str) -> bool:
    """验证 URL 是否为合法的 http/https 链接。"""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        parsed = urlparse(url)
        return bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


class TavilySearchProvider(SearchProvider):
    """Tavily 搜索真实实现。

    使用 Tavily SDK 原生 timeout 参数（而非线程包装），
    确保网络请求在 TAVILY_TIMEOUT（默认 25s）内被服务端/客户端正确终止。
    最多重试 2 次（总耗时 ≤ timeout + 0.5s + timeout）。
    """

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[int] = None) -> None:
        self.api_key = api_key or settings.TAVILY_API_KEY
        self.timeout = timeout or settings.TAVILY_TIMEOUT

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _client(self):
        from tavily import TavilyClient
        try:
            return TavilyClient(api_key=self.api_key, timeout=self.timeout)
        except TypeError:
            return TavilyClient(api_key=self.api_key)

    def extract(self, urls: list[str]) -> dict[str, str]:
        if not self.is_configured() or not urls:
            return {}
        valid_urls = [u for u in urls if _validate_url(u)]
        if not valid_urls:
            return {}

        try:
            client = self._client()
        except Exception:
            return {}

        out: dict[str, str] = {}
        try:
            resp = client.extract(urls=list(valid_urls), include_images=False)
            items: list[dict] = []
            if isinstance(resp, list):
                items = list(resp)
            elif isinstance(resp, dict):
                items = list(resp.get("results") or resp.get("documents") or [])
            for it in items:
                if not isinstance(it, dict):
                    continue
                u = str(it.get("url", "")).strip()
                c = str(it.get("raw_content") or it.get("content") or "").strip()
                if u and c and _validate_url(u):
                    out[u] = c
        except Exception:
            return {}
        return out

    def search(
        self,
        query: str,
        max_results: int,
        topic: str = "general",
    ) -> tuple[list[SearchResult], float, Optional[str]]:
        if not self.is_configured():
            return [], 0.0, "未配置 TAVILY_API_KEY"

        try:
            from tavily import TavilyClient
        except ImportError:
            return [], 0.0, "未安装 tavily-python，请运行 pip install tavily-python"

        try:
            client = self._client()
        except Exception as e:
            return [], 0.0, f"Tavily 客户端初始化失败：{e}"

        # 最多尝试2次：第一次8秒，等待0.5秒，第二次8秒
        # 总耗时 ≤ 8 + 0.5 + 8 = 16.5 秒
        last_error = None
        result = None

        for attempt in range(2):
            if attempt > 0:
                time.sleep(0.5)

            try:
                result = client.search(
                    query=query,
                    search_depth="basic",
                    include_answer=False,
                    include_raw_content="markdown",
                    max_results=max_results,
                    topic=topic,
                    timeout=self.timeout,
                )
                last_error = None
                break
            except Exception as e:
                error_msg = str(e)
                last_error = error_msg
                if attempt < 1 and self._is_retryable_error(error_msg):
                    continue
                break

        if last_error and not result:
            return [], 0.0, self._classify_error(last_error)

        if result is None:
            return [], 0.0, "Tavily 搜索返回空结果"

        response = result
        response_time = 0.0
        if isinstance(response, dict):
            response_time = float(response.get("response_time", 0.0))
        raw_results = response.get("results", []) if isinstance(response, dict) else []

        search_results: list[SearchResult] = []
        for item in raw_results:
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()

            if not title or not url:
                continue
            if not _validate_url(url):
                continue

            content = str(item.get("content", "") or item.get("raw_content", "") or "").strip()
            score = float(item.get("score", 0.0))
            published_at = None
            pub_str = item.get("published_date")
            if pub_str:
                try:
                    published_at = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00"))
                except Exception:
                    published_at = None

            search_results.append(
                SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    publisher=item.get("publisher", "") or urlparse(url).netloc,
                    published_at=published_at,
                    score=score,
                    raw_response=item,
                )
            )

        return search_results, response_time, None

    @staticmethod
    def _is_retryable_error(error: str) -> bool:
        """判断错误是否为可重试的临时性错误。"""
        error_lower = error.lower()
        # 只对 Timeout、ConnectionError、SSLError、5xx 重试
        retryable_patterns = [
            "ssl", "ssleoferror", "ssl_eof", "sslerror",
            "connectionerror", "connectionreset", "connectionrefused",
            "connectiontimedout", "connecttimeout",
            "timeout", "timed out",
            "500", "502", "503", "504",
            "temporarily", "temporary",
        ]
        # 明确不对 401、403、429 重试
        non_retryable_patterns = ["401", "403", "429", "unauthorized", "forbidden", "rate limit", "quota"]
        for p in non_retryable_patterns:
            if p in error_lower:
                return False
        return any(p in error_lower for p in retryable_patterns)

    @staticmethod
    def _classify_error(err_msg: str) -> str:
        """将错误消息分类为用户友好的提示。"""
        err_lower = err_msg.lower()

        if "401" in err_lower or "403" in err_lower or "unauthorized" in err_lower or "forbidden" in err_lower:
            return "搜索服务密钥无效：请检查配置"

        if "429" in err_lower or "rate limit" in err_lower or "quota" in err_lower:
            return "搜索额度或请求频率受限：请稍后重试"

        if "ssl" in err_lower or "ssleoferror" in err_lower or "ssl_eof" in err_lower:
            return "搜索服务连接临时中断，请重试"

        if "connection" in err_lower:
            return "搜索服务连接临时中断，请重试"

        if "超时" in err_msg or "timeout" in err_lower or "timed out" in err_lower:
            return "搜索服务暂时不可用"

        if "15秒" in err_msg or "总时间" in err_msg:
            return "搜索服务暂时不可用"

        if "空结果" in err_msg or "empty" in err_lower or "no results" in err_lower:
            return "未找到相关搜索结果"

        return "搜索服务暂时不可用"


class MockSearchProvider(SearchProvider):
    """演示模式搜索实现。"""

    def is_configured(self) -> bool:
        return True

    def _is_round2(self, query: str) -> bool:
        r2_markers = [
            "官方 个人介绍", "权威媒体 人物资料", "原始公告 官方名单",
            "原始报道 主流媒体 核实",
        ]
        return any(m in query for m in r2_markers)

    def search(
        self,
        query: str,
        max_results: int,
        topic: str = "general",
    ) -> tuple[list[SearchResult], float, Optional[str]]:
        now = datetime.now()
        is_round2 = self._is_round2(query)

        if "地铁" in query or "停运" in query:
            results = [
                SearchResult(
                    title="市轨道交通集团：暴雨期间部分线路临时停运",
                    url="https://example.com/demo/metro-partial",
                    content="暴雨导致3条线路临时停运，其余线路限速运行，并非全线停运。",
                    publisher="市轨道交通集团",
                    published_at=now,
                    score=0.95,
                ),
                SearchResult(
                    title="本地日报：暴雨影响早高峰交通",
                    url="https://example.com/demo/daily-rain",
                    content="报道提到部分地铁站点积水，未提及全线停运。",
                    publisher="本地日报",
                    published_at=now,
                    score=0.80,
                ),
            ]
        elif "失联" in query or "人员" in query:
            results = [
                SearchResult(
                    title="市应急管理局：暂未接到人员失联报告",
                    url="https://example.com/demo/emergency",
                    content="截至发布时，未接到因暴雨导致的人员失联报告。",
                    publisher="市应急管理局",
                    published_at=now,
                    score=0.92,
                ),
            ]
        elif "停课" in query or "学校" in query:
            results = [
                SearchResult(
                    title="市教育局：暴雨红色预警期间可灵活调整上课时间",
                    url="https://example.com/demo/education",
                    content="教育局发布指引，建议学校根据预警灵活安排，未发布全市停课三天的通知。",
                    publisher="市教育局",
                    published_at=now,
                    score=0.93,
                ),
            ]
        elif "嫦娥" in query:
            results = [
                SearchResult(
                    title="新华网：嫦娥六号成功着陆月球背面并完成采样",
                    url="https://example.com/demo/change6-xinhua",
                    content="嫦娥六号于2024年6月成功着陆月球背面南极-艾特肯盆地，完成人类首次月球背面采样返回任务。",
                    publisher="新华网",
                    published_at=now,
                    score=0.98,
                ),
                SearchResult(
                    title="国家航天局：嫦娥六号任务圆满成功",
                    url="https://example.com/demo/change6-cnsa",
                    content="国家航天局宣布嫦娥六号任务取得圆满成功，飞船携带月球背面样品顺利返回地球。",
                    publisher="国家航天局",
                    published_at=now,
                    score=0.96,
                ),
            ]
        elif "左航" in query and ("TF家族" in query or "TF 家族" in query):
            if not is_round2:
                results = [
                    SearchResult(
                        title="粉丝号搬运：TF家族三代成员猜测名单（非官方）",
                        url="https://example.com/demo/tf-fan-c1-fandom",
                        content="有粉丝号整理了TF家族三代成员名单，其中提到左航等练习生。该名单未注明来源。",
                        publisher="娱乐搬运号",
                        published_at=now,
                        score=0.55,
                    ),
                ]
            else:
                results = [
                    SearchResult(
                        title="北京时代峰峻官方网站：TF家族三代练习生公开名单",
                        url="https://example.com/demo/tf-official-3rdgen",
                        content="北京时代峰峻公开TF家族三代练习生名单：朱志鑫、左航、童禹坤、邓佳鑫等。",
                        publisher="北京时代峰峻（TF家族所属公司官网）",
                        published_at=now,
                        score=0.98,
                    ),
                ]
        elif "邓佳鑫" in query and ("TF家族" in query or "TF 家族" in query):
            if not is_round2:
                results = [
                    SearchResult(
                        title="粉丝号搬运：TF家族三代成员猜测名单",
                        url="https://example.com/demo/tf-fan-c2-fandom",
                        content="转载自粉丝号的TF家族三代成员名单，提到邓佳鑫等练习生。",
                        publisher="娱乐搬运号（转载）",
                        published_at=now,
                        score=0.52,
                    ),
                ]
            else:
                results = [
                    SearchResult(
                        title="北京时代峰峻官方网站：TF家族三代练习生公开名单",
                        url="https://example.com/demo/tf-official-3rdgen",
                        content="北京时代峰峻公开TF家族三代练习生名单：邓佳鑫为其中一员。",
                        publisher="北京时代峰峻（TF家族所属公司官网）",
                        published_at=now,
                        score=0.98,
                    ),
                ]
        elif "情侣" in query or "恋爱" in query or ("左航" in query and "邓佳鑫" in query):
            results = [
                SearchResult(
                    title="匿名爆料号：左航与邓佳鑫恋爱传闻整理",
                    url="https://example.com/demo/tf-ship-original-A",
                    content="匿名用户在粉丝社区发帖称左航与邓佳鑫曾在2020年恋爱。该帖未得到双方本人或经纪公司佐证。",
                    publisher="匿名粉丝爆料号",
                    published_at=now,
                    score=0.40,
                ),
                SearchResult(
                    title="搬运号：左航邓佳鑫情侣传闻",
                    url="https://example.com/demo/tf-ship-repost-A1",
                    content="转载并整理匿名爆料号的传闻帖。未添加任何新信息。",
                    publisher="娱乐搬运号（转载）",
                    published_at=now,
                    score=0.35,
                ),
            ]
        else:
            results = [
                SearchResult(
                    title="模拟搜索结果：相关背景信息",
                    url="https://example.com/demo/generic",
                    content=f"与查询「{query}」相关的演示证据摘要。本阶段为模拟数据，不具备真实参考价值。",
                    publisher="模拟来源",
                    published_at=now,
                    score=0.50,
                )
            ]

        return results[:max_results], 0.1, None

    def extract(self, urls: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for u in urls or []:
            if "tf-official-3rdgen" in u:
                out[u] = (
                    "# TF家族三代练习生公开名单\n"
                    "北京时代峰峻公开TF家族三代练习生名单："
                    "朱志鑫、左航、童禹坤、邓佳鑫、余宇涵、苏新皓、张极、张泽禹、张峻豪等。"
                )
            elif "metro-partial" in u:
                out[u] = "暴雨导致3条线路临时停运，其余线路限速运行，并非全线停运。"
            elif "change6" in u:
                out[u] = "嫦娥六号于2024年6月成功着陆月球背面，完成人类首次月球背面采样返回。"
            else:
                out[u] = f"[模拟抽取] URL={u} 的正文内容。"
        return out


def get_search_provider(mode: str = "demo") -> SearchProvider:
    if mode == "full":
        return TavilySearchProvider()
    return MockSearchProvider()


def choose_topic_for_claim(claim_text: str, claim_type: str = "") -> str:
    news_keywords = ["暴雨", "地震", "事故", "停运", "停课", "失联", "伤亡", "政策", "通知", "公布", "发布", "嫦娥", "月球", "采样"]
    if any(kw in claim_text for kw in news_keywords):
        return "news"
    if claim_type in ("事件陈述", "政策通知", "数据声明"):
        return "news"
    return "general"


def normalize_url(url: str) -> str:
    if not url:
        return ""
    url = url.strip()
    if "#" in url:
        url = url.split("#", 1)[0]
    url = url.rstrip("/")
    if "?" in url:
        base, query = url.split("?", 1)
        params = query.split("&")
        filtered = [
            p for p in params
            if not p.lower().startswith(("utm_", "ref=", "source=", "from=", "spm="))
        ]
        url = base + ("?" + "&".join(filtered) if filtered else "")
    return url.lower()


def deduplicate_by_url(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        norm = normalize_url(r.url)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(r)
    return unique


def search_web(query: str, claim_id: str, max_results: int = 3) -> list[Evidence]:
    provider = MockSearchProvider()
    results, _, err = provider.search(query=query, max_results=max_results)
    if err or not results:
        return []

    now = datetime.now()
    evidences: list[Evidence] = []
    for idx, r in enumerate(results, start=1):
        if "轨道交通集团" in r.publisher or "应急管理局" in r.publisher or "教育局" in r.publisher:
            grade, stance, is_primary = "A", "refutes", True
        elif "日报" in r.publisher or "新闻" in r.publisher:
            grade, stance, is_primary = "B", "partial", False
        elif "社交" in r.publisher or "网络用户" in r.publisher:
            grade, stance, is_primary = "E", "unclear", False
        else:
            grade, stance, is_primary = "D", "unclear", False

        evidences.append(
            Evidence(
                evidence_id=f"{claim_id}-e{idx}",
                claim_id=claim_id,
                source_title=r.title,
                source_url=r.url,
                publisher=r.publisher,
                published_at=r.published_at or now,
                retrieved_at=now,
                evidence_summary=r.content[:200],
                source_type="演示数据" if "example.com" in r.url else "其他",
                source_grade=grade,
                supports_or_refutes=stance,
                is_primary_source=is_primary,
                reliability_reason="本阶段模拟数据，不具备真实参考价值",
            )
        )
    return evidences
