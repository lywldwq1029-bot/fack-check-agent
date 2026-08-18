"""会话级短期记忆缓存。

用于在同一次会话内缓存已检索的证据，避免重复调用搜索工具。
与长期记忆 (MemoryStore) 的区别：
- 短期：仅在当前会话有效，会话结束后清除
- 长期：持久化存储，跨会话可用

功能：
1. 缓存已搜索的查询和结果
2. 缓存已验证的主张和结论
3. 识别已确认不可靠的来源，不再采信
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Optional

from src.models import Evidence


class SessionCache:
    """会话级短期记忆缓存。"""

    def __init__(self, max_age_seconds: int = 3600):
        """初始化会话缓存。

        Args:
            max_age_seconds: 缓存最大存活时间（秒），默认 1 小时
        """
        self.max_age = max_age_seconds
        self._search_cache: dict[str, dict] = {}  # query -> {results, timestamp}
        self._evidence_cache: dict[str, list[Evidence]] = {}  # claim -> evidences
        self._verdict_cache: dict[str, dict] = {}  # claim_hash -> verdict info
        self._unreliable_sources: set[str] = set()  # 已确认不可靠的来源 URL
        self._source_credibility: dict[str, float] = defaultdict(float)  # 来源可信度缓存
        self._created_at = time.monotonic()

    def is_expired(self) -> bool:
        """检查缓存是否过期。"""
        return (time.monotonic() - self._created_at) > self.max_age

    def get_search_results(self, query: str) -> Optional[dict]:
        """获取缓存的搜索结果。

        Args:
            query: 搜索关键词

        Returns:
            缓存的搜索结果，如果不存在或已过期返回 None
        """
        # 规范化查询
        normalized_query = self._normalize_query(query)

        if normalized_query in self._search_cache:
            cached = self._search_cache[normalized_query]
            # 检查是否过期
            if (time.monotonic() - cached["timestamp"]) < self.max_age:
                return cached["results"]
            # 过期则删除
            del self._search_cache[normalized_query]

        return None

    def set_search_results(self, query: str, results: list[dict]) -> None:
        """缓存搜索结果。

        Args:
            query: 搜索关键词
            results: 搜索结果列表
        """
        normalized_query = self._normalize_query(query)
        self._search_cache[normalized_query] = {
            "results": results,
            "timestamp": time.monotonic(),
        }

    def get_cached_evidences(self, claim: str) -> list[Evidence]:
        """获取缓存的证据列表。

        Args:
            claim: 主张文本

        Returns:
            缓存的证据列表
        """
        normalized_claim = self._normalize_query(claim)
        return self._evidence_cache.get(normalized_claim, [])

    def set_cached_evidences(self, claim: str, evidences: list[Evidence]) -> None:
        """缓存证据列表。

        Args:
            claim: 主张文本
            evidences: 证据列表
        """
        normalized_claim = self._normalize_query(claim)
        self._evidence_cache[normalized_claim] = evidences

        # 更新来源可信度
        for ev in evidences:
            self._source_credibility[ev.source_url] = self._grade_to_score(ev.source_grade)

    def get_cached_verdict(self, claim: str) -> Optional[dict]:
        """获取缓存的判断结果。

        Args:
            claim: 主张文本

        Returns:
            缓存的判断结果，如果不存在返回 None
        """
        claim_hash = self._hash_claim(claim)
        return self._verdict_cache.get(claim_hash)

    def set_cached_verdict(self, claim: str, verdict_info: dict) -> None:
        """缓存判断结果。

        Args:
            claim: 主张文本
            verdict_info: 判断信息
        """
        claim_hash = self._hash_claim(claim)
        self._verdict_cache[claim_hash] = verdict_info

    def mark_source_unreliable(self, url: str) -> None:
        """标记来源为不可靠。

        Args:
            url: 来源 URL
        """
        self._unreliable_sources.add(url)

    def is_source_unreliable(self, url: str) -> bool:
        """检查来源是否被标记为不可靠。

        Args:
            url: 来源 URL

        Returns:
            是否不可靠
        """
        return url in self._unreliable_sources

    def get_source_credibility(self, url: str) -> float:
        """获取来源可信度分数。

        Args:
            url: 来源 URL

        Returns:
            可信度分数 (0.0-1.0)
        """
        return self._source_credibility.get(url, 0.0)

    def clear(self) -> None:
        """清除所有缓存。"""
        self._search_cache.clear()
        self._evidence_cache.clear()
        self._verdict_cache.clear()
        self._unreliable_sources.clear()
        self._source_credibility.clear()
        self._created_at = time.monotonic()

    def get_stats(self) -> dict:
        """获取缓存统计信息。"""
        return {
            "search_queries_cached": len(self._search_cache),
            "claims_with_evidence": len(self._evidence_cache),
            "verdicts_cached": len(self._verdict_cache),
            "unreliable_sources": len(self._unreliable_sources),
            "sources_rated": len(self._source_credibility),
            "age_seconds": int(time.monotonic() - self._created_at),
            "expires_in": max(0, self.max_age - int(time.monotonic() - self._created_at)),
        }

    @staticmethod
    def _normalize_query(text: str) -> str:
        """规范化查询文本。"""
        return ' '.join(text.lower().split())

    @staticmethod
    def _hash_claim(claim: str) -> str:
        """生成主张哈希。"""
        import hashlib
        normalized = ' '.join(claim.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    @staticmethod
    def _grade_to_score(grade: str) -> float:
        """将来源等级转换为可信度分数。"""
        score_map = {
            "A": 1.0,
            "B": 0.8,
            "C": 0.5,
            "D": 0.2,
            "E": 0.1,
        }
        return score_map.get(grade, 0.3)


# 全局会话缓存实例（使用单例模式）
_session_cache: Optional[SessionCache] = None


def get_session_cache() -> SessionCache:
    """获取全局会话缓存实例。"""
    global _session_cache
    if _session_cache is None or _session_cache.is_expired():
        _session_cache = SessionCache()
    return _session_cache


def reset_session_cache() -> None:
    """重置全局会话缓存。"""
    global _session_cache
    _session_cache = None
