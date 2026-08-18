"""报告导出器包：支持 Word / PDF 等格式的导出。"""

from src.exporters.docx_exporter import build_fact_check_docx

__all__ = ["build_fact_check_docx"]
