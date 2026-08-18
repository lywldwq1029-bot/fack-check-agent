"""SQLite 长期记忆存储。

存储历史核查记录，用于：
- 查询相同或相似历史案例
- 避免重复核查相同主张
- 为敏感信息提供历史参考

注意：历史记录只能作为参考，不能代替当前公开证据。
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import FactCheckReport


# 默认数据库路径
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "fact_check_memory.sqlite"


class MemoryStore:
    """SQLite 长期记忆存储。"""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接。"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """初始化数据库表结构。"""
        conn = self._get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS verification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    normalized_claim TEXT NOT NULL,
                    original_claim TEXT NOT NULL,
                    claim_hash TEXT,
                    verdict TEXT NOT NULL,
                    conclusion_summary TEXT,
                    checked_at TEXT NOT NULL,
                    search_keywords TEXT,
                    supplemental_search INTEGER DEFAULT 0,
                    evidence_count INTEGER DEFAULT 0,
                    source_grades TEXT,
                    credibility_score REAL DEFAULT 0.0,
                    risk_level TEXT DEFAULT '不确定',
                    recommendation TEXT,
                    is_sensitive INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evidence_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    history_id INTEGER NOT NULL,
                    evidence_id TEXT,
                    title TEXT,
                    url TEXT,
                    grade TEXT,
                    summary TEXT,
                    directly_supports INTEGER DEFAULT 0,
                    is_independent INTEGER DEFAULT 1,
                    FOREIGN KEY (history_id) REFERENCES verification_history(id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_normalized_claim 
                ON verification_history(normalized_claim)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_checked_at 
                ON verification_history(checked_at)
            """)
            conn.commit()
        finally:
            conn.close()

    def save_report(self, report: FactCheckReport) -> int:
        """保存核查报告到记忆。

        Returns:
            保存记录的 ID
        """
        conn = self._get_conn()
        try:
            checked_at = datetime.now().isoformat()

            # 计算标准化主张（简单处理：取前50字符）
            normalized = report.original_text[:50].strip()

            # 提取来源等级
            source_grades = set()
            evidence_count = 0
            for result in report.claim_results:
                for ev in result.evidence:
                    evidence_count += 1
                    source_grades.add(ev.source_grade)

            # 检查是否敏感信息
            sensitive_keywords = ["怀孕", "恋情", "违法", "健康", "私人"]
            is_sensitive = any(kw in report.original_text for kw in sensitive_keywords)

            cursor = conn.execute("""
                INSERT INTO verification_history
                (normalized_claim, original_claim, claim_hash, verdict, conclusion_summary,
                 checked_at, search_keywords, supplemental_search, evidence_count, source_grades,
                 credibility_score, risk_level, recommendation, is_sensitive)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                normalized,
                report.original_text[:200],
                normalized,
                report.overall_verdict,
                report.overall_summary[:200] if report.overall_summary else "",
                checked_at,
                report.original_text[:100],  # 搜索关键词
                1 if report.did_supplemental_search else 0,
                evidence_count,
                ",".join(sorted(source_grades)),
                report.credibility_score,
                report.risk_level or "不确定",
                report.recommendation or "",
                1 if is_sensitive else 0,
            ))
            history_id = cursor.lastrowid

            # 保存证据记录
            for result in report.claim_results:
                for ev in result.evidence:
                    conn.execute("""
                        INSERT INTO evidence_records
                        (history_id, evidence_id, title, url, grade, summary, directly_supports, is_independent)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        history_id,
                        ev.evidence_id,
                        ev.source_title[:200],
                        ev.source_url[:500],
                        ev.source_grade,
                        (ev.summary or ev.evidence_summary)[:200],
                        1 if ev.directly_supports else 0,
                        1 if ev.is_independent else 0,
                    ))

            conn.commit()
            print(f"[MEMORY] 核查报告已保存到历史记忆，ID={history_id}")
            return history_id
        except Exception as e:
            print(f"[MEMORY] 保存失败: {e}")
            conn.rollback()
            return 0
        finally:
            conn.close()

    def search_similar(self, claim: str, limit: int = 3) -> list[dict]:
        """搜索相似的历史核查记录。

        Args:
            claim: 待核查的主张
            limit: 最多返回的历史记录数

        Returns:
            历史记录列表，每项包含 verdict, checked_at, 等字段
        """
        conn = self._get_conn()
        try:
            normalized = claim[:50].strip()
            results = []

            # 精确匹配标准化主张
            rows = conn.execute("""
                SELECT * FROM verification_history
                WHERE normalized_claim LIKE ?
                ORDER BY checked_at DESC
                LIMIT ?
            """, (f"%{normalized[:20]}%", limit * 2)).fetchall()

            for row in rows:
                # 检查是否为敏感信息且超过72小时
                is_sensitive = row["is_sensitive"]
                checked_at = row["checked_at"]

                # 对于敏感信息，超过72小时的历史记录降低参考价值
                if is_sensitive:
                    try:
                        check_time = datetime.fromisoformat(checked_at)
                        age_hours = (datetime.now() - check_time).total_seconds() / 3600
                        if age_hours > 72:
                            continue  # 超过72小时的敏感信息不返回
                    except (ValueError, TypeError):
                        pass

                result = {
                    "id": row["id"],
                    "normalized_claim": row["normalized_claim"],
                    "verdict": row["verdict"],
                    "conclusion_summary": row["conclusion_summary"],
                    "checked_at": row["checked_at"],
                    "search_keywords": row["search_keywords"],
                    "supplemental_search": bool(row["supplemental_search"]),
                    "evidence_count": row["evidence_count"],
                    "source_grades": row["source_grades"],
                    "credibility_score": row["credibility_score"],
                    "risk_level": row["risk_level"],
                    "recommendation": row["recommendation"],
                    "is_sensitive": bool(row["is_sensitive"]),
                }

                # 获取关联的证据
                evidence_rows = conn.execute("""
                    SELECT * FROM evidence_records
                    WHERE history_id = ?
                    LIMIT 5
                """, (row["id"],)).fetchall()

                result["evidences"] = [
                    {
                        "evidence_id": er["evidence_id"],
                        "title": er["title"],
                        "url": er["url"],
                        "grade": er["grade"],
                        "summary": er["summary"],
                        "directly_supports": bool(er["directly_supports"]),
                    }
                    for er in evidence_rows
                ]

                results.append(result)
                if len(results) >= limit:
                    break

            return results
        except Exception as e:
            print(f"[MEMORY] 搜索历史失败: {e}")
            return []
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """获取记忆统计信息。"""
        conn = self._get_conn()
        try:
            total = conn.execute("SELECT COUNT(*) as cnt FROM verification_history").fetchone()["cnt"]
            today = conn.execute("""
                SELECT COUNT(*) as cnt FROM verification_history
                WHERE DATE(checked_at) = DATE('now')
            """).fetchone()["cnt"]

            verdicts = conn.execute("""
                SELECT verdict, COUNT(*) as cnt
                FROM verification_history
                GROUP BY verdict
            """).fetchall()

            return {
                "total": total,
                "today": today,
                "by_verdict": {v["verdict"]: v["cnt"] for v in verdicts},
            }
        except Exception:
            return {"total": 0, "today": 0, "by_verdict": {}}
        finally:
            conn.close()

    def clean_old_records(self, max_days: int = 30) -> int:
        """清理超过指定天数的旧记录。"""
        conn = self._get_conn()
        try:
            conn.execute("""
                DELETE FROM verification_history
                WHERE julianday('now') - julianday(checked_at) > ?
            """, (max_days,))
            deleted = conn.total_changes
            conn.commit()
            return deleted
        except Exception as e:
            print(f"[MEMORY] 清理失败: {e}")
            return 0
        finally:
            conn.close()
