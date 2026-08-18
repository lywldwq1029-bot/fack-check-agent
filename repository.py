"""核查记忆库：使用 SQLite 持久化核查报告与执行日志。

当前阶段仅提供基础建表和保存接口，完整的查询、去重、相似报告检索
等功能可在后续迭代中扩展。
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import settings
from src.models import FactCheckReport


class MemoryRepository:
    """核查记忆库操作类。"""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or settings.MEMORY_DB_PATH
        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fact_check_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_text TEXT NOT NULL,
                    overall_verdict TEXT NOT NULL,
                    overall_summary TEXT NOT NULL,
                    claim_count INTEGER NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def save_report(self, report: FactCheckReport) -> int:
        """保存核查报告到记忆库，返回记录 ID。"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO fact_check_reports
                (original_text, overall_verdict, overall_summary, claim_count, report_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    report.original_text,
                    report.overall_verdict,
                    report.overall_summary,
                    len(report.claim_results),
                    report.model_dump_json(ensure_ascii=False),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def list_reports(self, limit: int = 10) -> list[dict]:
        """列出最近的核查报告摘要。"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, original_text, overall_verdict, claim_count, created_at
                FROM fact_check_reports
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
