"""大模型客户端单元测试。

使用 mock 客户端测试，不真实消耗 API 额度。
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from src.llm.client import LLMClient, LLMError, _strip_code_fence


class _DummyModel(BaseModel):
    """用于测试 chat_json 的简单模型。"""

    name: str
    age: int


# ===== _strip_code_fence 测试 =====


def test_strip_code_fence_plain_json():
    """纯 JSON 不受影响。"""
    text = '{"name": "张三", "age": 25}'
    assert _strip_code_fence(text) == text


def test_strip_code_fence_json_block():
    """带 ```json 围栏的 JSON 被正确清理。"""
    text = '```json\n{"name": "张三", "age": 25}\n```'
    assert _strip_code_fence(text) == '{"name": "张三", "age": 25}'


def test_strip_code_fence_plain_block():
    """带 ``` 围栏（无 json 标记）的 JSON 被正确清理。"""
    text = '```\n{"name": "张三", "age": 25}\n```'
    assert _strip_code_fence(text) == '{"name": "张三", "age": 25}'


def test_strip_code_fence_with_extra_spaces():
    """带额外空格的围栏被正确清理。"""
    text = '```json\n\n  {"name": "张三", "age": 25}  \n```'
    assert _strip_code_fence(text) == '{"name": "张三", "age": 25}'


# ===== chat_json 正常情况测试 =====


def _mock_openai_response(content: str):
    """构造一个模拟的 OpenAI 响应对象。"""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def test_chat_json_normal_json():
    """测试正常 JSON 返回能被正确解析。"""
    mock_response = _mock_openai_response('{"name": "张三", "age": 25}')

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        result = client.chat_json(
            system_prompt="test",
            user_prompt="test",
            output_model=_DummyModel,
        )

    assert result.name == "张三"
    assert result.age == 25


def test_chat_json_markdown_fenced_json():
    """测试带 Markdown 围栏的 JSON 能被正确解析。"""
    mock_response = _mock_openai_response('```json\n{"name": "李四", "age": 30}\n```')

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        result = client.chat_json(
            system_prompt="test",
            user_prompt="test",
            output_model=_DummyModel,
        )

    assert result.name == "李四"
    assert result.age == 30


def test_chat_json_with_explanation_prefix():
    """模型返回了前言 + JSON，修复调用后应成功。"""
    bad_response = _mock_openai_response("好的，以下是结果：\n{invalid}")
    good_response = _mock_openai_response('{"name": "王五", "age": 40}')

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [bad_response, good_response]
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        result = client.chat_json(
            system_prompt="test",
            user_prompt="test",
            output_model=_DummyModel,
        )

    assert result.name == "王五"


# ===== chat_json 异常情况测试 =====


def test_chat_json_invalid_json_no_repair():
    """非法 JSON 且不允许修复时抛出 LLMError。"""
    mock_response = _mock_openai_response("这不是 JSON")

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(LLMError, match="不是合法 JSON"):
            client.chat_json(
                system_prompt="test",
                user_prompt="test",
                output_model=_DummyModel,
                repair_attempt=False,
            )


def test_chat_json_repair_still_fails():
    """修复调用后仍非法 JSON 时抛出 LLMError。"""
    bad_response1 = _mock_openai_response("{invalid")
    bad_response2 = _mock_openai_response("还是不对")

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [bad_response1, bad_response2]
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(LLMError, match="经过一次修复仍不是合法 JSON"):
            client.chat_json(
                system_prompt="test",
                user_prompt="test",
                output_model=_DummyModel,
            )


def test_chat_json_missing_fields():
    """JSON 字段缺失时抛出 LLMError（Pydantic 验证失败）。"""
    mock_response = _mock_openai_response('{"name": "张三"}')

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(LLMError, match="不符合预期"):
            client.chat_json(
                system_prompt="test",
                user_prompt="test",
                output_model=_DummyModel,
                repair_attempt=False,
            )


def test_chat_empty_response():
    """大模型返回空内容时抛出 LLMError。"""
    mock_response = MagicMock()
    mock_response.choices = []

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(LLMError, match="空响应"):
            client.chat("test", "test")


def test_chat_empty_content():
    """大模型返回空字符串内容时抛出 LLMError。"""
    mock_response = _mock_openai_response("")

    with patch.object(LLMClient, "_build_client") as mock_build:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_build.return_value = mock_client

        client = LLMClient(api_key="fake", model="fake")
        with pytest.raises(LLMError, match="空内容"):
            client.chat("test", "test")
