"""搜索缓存模块 - SQLite 持久化缓存。

支持：
- 24 小时内相同查询直接命中缓存
- 实时搜索失败时，72 小时内的旧缓存可作为兜底
- 原始 URL 保持不变
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.tools.search_tool import SearchResult


# 24小时内直接命中缓存
HIT_TTL_SECONDS = 24 * 3600
# 72小时内可作为兜底缓存
FALLBACK_TTL_SECONDS = 72 * 3600


@dataclass
class CacheEntry:
    """单条缓存记录。"""
    normalized_query: str
    results_json: str
    created_at: float
    source: str = "tavily"  # 原始来源


class SearchCache:
    """SQLite 搜索缓存管理器。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is None:
            # 默认放在项目根目录
            base = Path(__file__).resolve().parent.parent.parent
            db_path = str(base / ".search_cache.sqlite")
        self._db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS search_cache (
                    normalized_query TEXT PRIMARY KEY,
                    results_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    source TEXT NOT NULL DEFAULT 'tavily'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON search_cache(created_at)
            """)
            conn.commit()

    @staticmethod
    def _normalize_query(query: str) -> str:
        """将查询标准化，便于缓存命中。"""
        # 简单标准化：去除空白、统一小写
        q = " ".join(query.strip().lower().split())
        return q

    @staticmethod
    def _make_key(query: str) -> str:
        """生成缓存键。"""
        normalized = SearchCache._normalize_query(query)
        return normalized

    def get_hit(self, query: str) -> Optional[CacheEntry]:
        """获取 24 小时内的直接命中缓存。"""
        key = self._make_key(query)
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM search_cache WHERE normalized_query = ?",
                (key,)
            ).fetchone()

        if row is None:
            return None

        age = now - row["created_at"]
        if age > HIT_TTL_SECONDS:
            return None

        return CacheEntry(
            normalized_query=row["normalized_query"],
            results_json=row["results_json"],
            created_at=row["created_at"],
            source=row["source"],
        )

    def get_fallback(self, query: str) -> Optional[CacheEntry]:
        """获取 72 小时内的兜底缓存（仅在实时搜索失败时使用）。"""
        key = self._make_key(query)
        now = time.time()
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM search_cache WHERE normalized_query = ?",
                (key,)
            ).fetchone()

        if row is None:
            return None

        age = now - row["created_at"]
        if age > FALLBACK_TTL_SECONDS:
            return None

        return CacheEntry(
            normalized_query=row["normalized_query"],
            results_json=row["results_json"],
            created_at=row["created_at"],
            source=row["source"],
        )

    def save(self, query: str, results: list[SearchResult], source: str = "tavily") -> None:
        """保存搜索结果到缓存。"""
        key = self._make_key(query)
        results_data = []
        for r in results:
            item = {
                "title": r.title,
                "url": r.url,  # 保持原始 URL
                "content": r.content,
                "publisher": r.publisher,
                "score": r.score,
            }
            if r.published_at:
                item["published_at"] = r.published_at.isoformat()
            results_data.append(item)

        results_json = json.dumps(results_data, ensure_ascii=False)
        now = time.time()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO search_cache
                (normalized_query, results_json, created_at, source)
                VALUES (?, ?, ?, ?)
            """, (key, results_json, now, source))
            conn.commit()

    @staticmethod
    def entry_to_results(entry: CacheEntry) -> list[SearchResult]:
        """将缓存条目转换回 SearchResult 列表。"""
        try:
            data = json.loads(entry.results_json)
        except (json.JSONDecodeError, TypeError):
            return []

        results: list[SearchResult] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            published_at = None
            pub_str = item.get("published_at")
            if pub_str:
                try:
                    from datetime import datetime
                    published_at = datetime.fromisoformat(str(pub_str))
                except Exception:
                    pass

            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),  # 使用原始 URL
                content=item.get("content", ""),
                publisher=item.get("publisher", ""),
                published_at=published_at,
                score=float(item.get("score", 0.0)),
            ))
        return results

    def age_hours(self, entry: CacheEntry) -> float:
        """返回缓存条目的年龄（小时）。"""
        return (time.time() - entry.created_at) / 3600.0
