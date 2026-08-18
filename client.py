"""统一的大模型客户端封装。

特性：
- 兼容 OpenAI API 格式（含 DeepSeek、通义千问、Moonshot 等兼容服务）
- 超时处理与最多 2 次重试（使用 tenacity）
- 清晰的中文错误信息
- JSON 结构化输出解析与 Pydantic 验证
- 自动去除模型返回的 Markdown 代码围栏
- 调用失败时抛出异常，绝不伪装成真实结果

安全约定：
- 密钥仅从环境变量读取，不打印、不落库、不写入日志
"""

import json
import re
from typing import Optional, Type, TypeVar

from openai import OpenAI, APIError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings


T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """大模型调用过程中的统一异常，携带中文错误信息。"""


def _strip_code_fence(text: str) -> str:
    """去除模型返回内容首尾可能出现的 Markdown 代码围栏。

    处理形如：
        ```json
        { ... }
        ```
        ```
        { ... }
        ```
    """
    stripped = text.strip()
    # 匹配开头的 ```json 或 ``` 及结尾的 ```
    fence_pattern = re.compile(r"^```(?:json)?\s*\n(.*?)\n```$", re.DOTALL)
    match = fence_pattern.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


class LLMClient:
    """大模型客户端。

    通过 OpenAI SDK 兼容格式调用，支持自定义 base_url。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """初始化客户端。

        参数留空时从 settings 读取。不做密钥校验——由调用方在使用前
        通过 settings.llm_configured() 判断。
        """
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model if model is not None else settings.LLM_MODEL
        self.timeout = timeout if timeout is not None else settings.LLM_TIMEOUT

    def _ensure_configured(self) -> None:
        """检查是否已配置 api_key 和 model，未配置时直接抛出 LLMError。

        这样可以避免在未配置时仍然发起真实网络请求，导致长时间超时。
        """
        if not self.api_key:
            raise LLMError("未配置 LLM_API_KEY，请在 .env 中填写后再使用真实 LLM 模式。")
        if not self.model:
            raise LLMError("未配置 LLM_MODEL，请在 .env 中填写模型名称后再使用真实 LLM 模式。")

    def _build_client(self) -> OpenAI:
        """构建 OpenAI SDK 客户端实例。"""
        kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    @retry(
        retry=retry_if_exception_type((APITimeoutError, APIConnectionError)),
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES + 1),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        reraise=True,
    )
    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """调用聊天接口，返回模型文本输出。仅在超时/连接错误时重试，最多 2 次。"""
        client = self._build_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
        except APITimeoutError:
            raise
        except APIConnectionError:
            raise
        except APIError as e:
            raise LLMError(f"大模型接口返回错误：{e.message if hasattr(e, 'message') else e}")
        except Exception as e:
            raise LLMError(f"大模型调用失败：{e}")

        if not response.choices:
            raise LLMError("大模型返回空响应，没有可用的 choices。")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise LLMError("大模型返回空内容。")
        return content.strip()

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        """公开的聊天接口，带重试。"""
        try:
            return self._chat(system_prompt, user_prompt, temperature)
        except APITimeoutError:
            raise LLMError(f"大模型请求超时（{self.timeout}秒，已重试 {settings.LLM_MAX_RETRIES} 次），请检查网络或增大 LLM_TIMEOUT。")
        except APIConnectionError:
            raise LLMError("无法连接大模型服务，请检查 LLM_BASE_URL 或网络。")

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        output_model: Type[T],
        temperature: float = 0.2,
        repair_attempt: bool = True,
    ) -> T:
        """调用大模型并返回经 Pydantic 验证的结构化对象。

        Args:
            system_prompt: 系统提示词
            user_prompt: 用户提示词
            output_model: 期望的 Pydantic 模型类
            temperature: 温度参数
            repair_attempt: 首次解析失败时是否尝试一次修复调用

        Raises:
            LLMError: 调用或解析失败时抛出，绝不返回伪造结果。
        """
        raw = self.chat(system_prompt, user_prompt, temperature)
        cleaned = _strip_code_fence(raw)

        # 第一次尝试解析
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as first_err:
            if not repair_attempt:
                raise LLMError(f"大模型返回的内容不是合法 JSON：{first_err}\n原始内容片段：{cleaned[:200]}")

            # 尝试一次修复调用
            repair_prompt = (
                f"你上一次返回的内容无法被解析为合法 JSON，错误信息：{first_err}\n"
                f"请只输出严格合法的 JSON，不要包含任何解释、前言或 Markdown 代码围栏。\n"
                f"上次返回的内容（供参考）：\n{raw[:1000]}"
            )
            raw = self.chat(system_prompt, repair_prompt, temperature=0.0)
            cleaned = _strip_code_fence(raw)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError as second_err:
                raise LLMError(
                    f"大模型返回的内容经过一次修复仍不是合法 JSON：{second_err}\n原始内容片段：{cleaned[:200]}"
                )

        # Pydantic 验证
        try:
            return output_model.model_validate(data)
        except ValidationError as ve:
            raise LLMError(f"大模型返回的 JSON 结构不符合预期：{ve}\n原始 JSON 片段：{cleaned[:300]}")
