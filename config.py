"""项目配置。

所有密钥均从 .env 读取，禁止将密钥硬编码到源码中。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# 加载 .env 文件（如果存在）
load_dotenv(BASE_DIR / ".env")


def _get_int(env_key: str, default: int) -> int:
    """安全读取整数环境变量。"""
    try:
        return int(os.getenv(env_key, str(default)))
    except (ValueError, TypeError):
        return default


class Settings:
    """应用配置类。"""

    # 大模型配置（兼容 OpenAI API 格式）
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "")
    LLM_TIMEOUT: int = _get_int("LLM_TIMEOUT", 35)
    LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 2)

    # Tavily 搜索配置
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    TAVILY_TIMEOUT: int = _get_int("TAVILY_TIMEOUT", 25)
    SEARCH_MAX_QUERIES_PER_CLAIM: int = _get_int("SEARCH_MAX_QUERIES_PER_CLAIM", 2)
    SEARCH_MAX_RESULTS_PER_QUERY: int = _get_int("SEARCH_MAX_RESULTS_PER_QUERY", 5)
    SEARCH_MAX_TOTAL_RESULTS: int = _get_int("SEARCH_MAX_TOTAL_RESULTS", 20)
    SEARCH_MAX_CONCURRENT: int = _get_int("SEARCH_MAX_CONCURRENT", 3)
    SEARCH_MAX_RETRIES: int = _get_int("SEARCH_MAX_RETRIES", 1)

    # 全局工作流配置
    WORKFLOW_MAX_SECONDS: int = _get_int("WORKFLOW_MAX_SECONDS", 120)
    MAX_CLAIMS: int = _get_int("MAX_CLAIMS", 4)

    # 记忆库配置
    MEMORY_DB_PATH: str = os.getenv("MEMORY_DB_PATH", str(DATA_DIR / "memory.db"))

    # 模拟模式开关
    USE_MOCK_SEARCH: bool = os.getenv("USE_MOCK_SEARCH", "true").lower() in ("true", "1", "yes")
    USE_MOCK: bool = USE_MOCK_SEARCH

    def llm_configured(self) -> bool:
        return bool(self.LLM_API_KEY and self.LLM_MODEL)

    def search_configured(self) -> bool:
        return bool(self.TAVILY_API_KEY)

    def full_real_configured(self) -> bool:
        return self.llm_configured() and self.search_configured()

    def missing_configs(self) -> list[str]:
        missing: list[str] = []
        if not self.LLM_API_KEY:
            missing.append("LLM_API_KEY")
        if not self.LLM_MODEL:
            missing.append("LLM_MODEL")
        if not self.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        return missing

    def masked_model_name(self) -> str:
        return self.LLM_MODEL if self.LLM_MODEL else "未配置"


settings = Settings()
