"""主张拆解节点单元测试。

使用 mock 客户端测试 LLM 拆解，不真实消耗 API 额度。
"""

from unittest.mock import MagicMock, patch

import pytest

from src.config import settings
from src.models import AgentState
from src.nodes.decompose import decompose_claims, decompose_claims_llm, decompose_claims_mock


def _mock_llm_response(content: str):
    """构造模拟的 OpenAI 响应。"""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_settings_configured():
    """临时伪造 LLM 配置，让 settings.llm_configured() 返回 True。

    使用上下文退出时恢复原值。返回 (原key, 原model)。
    """
    original_key = settings.LLM_API_KEY
    original_model = settings.LLM_MODEL
    settings.LLM_API_KEY = "fake-key-for-testing"
    settings.LLM_MODEL = "fake-model-for-testing"
    return original_key, original_model


def _restore_settings(key: str, model: str) -> None:
    settings.LLM_API_KEY = key
    settings.LLM_MODEL = model


# ===== 演示模式测试 =====


def test_decompose_mock_with_rain_news():
    """演示模式拆解暴雨新闻。"""
    state = AgentState(
        original_text="网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。",
        mode="demo",
    )
    state = decompose_claims(state)

    assert len(state.claims) >= 3
    assert any("地铁" in c.text for c in state.claims)
    assert any("失联" in c.text for c in state.claims)
    assert any("停课" in c.text for c in state.claims)


def test_decompose_mock_high_risk_flag():
    """演示模式下失联主张应标记为 high 风险。"""
    state = AgentState(
        original_text="网传某市因暴雨导致地铁全线停运，目前已有多人失联。",
        mode="demo",
    )
    state = decompose_claims(state)

    high_risk = [c for c in state.claims if c.risk_level == "high"]
    assert len(high_risk) > 0
    assert any("失联" in c.text for c in high_risk)


# ===== 真实 LLM 模式测试（mock） =====


def test_decompose_llm_case1_luhan_split():
    """案例1：鹿晗和关晓彤分手应拆成至少两个主张。"""
    llm_output = """
    {
      "claims": [
        {
          "text": "鹿晗和关晓彤已经分手",
          "claim_type": "事件陈述",
          "entities": ["鹿晗", "关晓彤"],
          "time_reference": "近期",
          "location": null,
          "verification_priority": 1,
          "verification_question": "鹿晗和关晓彤是否已正式宣布分手？",
          "search_keywords": ["鹿晗 关晓彤 分手", "鹿晗 关晓彤 官宣"],
          "preferred_source_types": ["当事人社交媒体", "权威娱乐媒体"],
          "risk_level": "medium",
          "sensitive_reason": "涉及个人感情生活",
          "is_opinion": false,
          "is_checkable": true
        },
        {
          "text": "分手原因是鹿晗男女关系混乱",
          "claim_type": "归因判断",
          "entities": ["鹿晗"],
          "time_reference": "近期",
          "location": null,
          "verification_priority": 1,
          "verification_question": "分手是否因鹿晗男女关系混乱？",
          "search_keywords": ["鹿晗 男女关系", "鹿晗 分手原因"],
          "preferred_source_types": ["权威媒体", "当事人声明"],
          "risk_level": "high",
          "sensitive_reason": "涉及个人名誉的负面指控",
          "is_opinion": false,
          "is_checkable": true
        }
      ],
      "summary": "已将事件与原因拆分为两个独立主张"
    }
    """
    mock_response = _mock_llm_response(llm_output)

    state = AgentState(
        original_text="鹿晗和关晓彤已经分手，原因是鹿晗男女关系混乱。",
        mode="llm",
    )

    # 绕过配置检查（测试不真实消耗额度）
    orig_key, orig_model = _mock_settings_configured()
    try:
        with patch("src.llm.client.LLMClient._build_client") as mock_build:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_build.return_value = mock_client

            state = decompose_claims_llm(state)
    finally:
        _restore_settings(orig_key, orig_model)

    assert len(state.claims) >= 2
    # 第二条应为 high 风险
    high_risk = [c for c in state.claims if c.risk_level == "high"]
    assert len(high_risk) >= 1
    assert any("男女关系" in c.text for c in high_risk)
    # claim_id 由程序生成
    for idx, c in enumerate(state.claims, start=1):
        assert c.claim_id == f"c{idx}"


def test_decompose_llm_case2_rain_news():
    """案例2：暴雨新闻应拆成地铁、失联、停课等主张。"""
    llm_output = """
    {
      "claims": [
        {
          "text": "某市因暴雨导致地铁全线停运",
          "claim_type": "事件陈述",
          "entities": ["某市", "地铁", "暴雨"],
          "time_reference": "当前",
          "location": "某市",
          "verification_priority": 1,
          "verification_question": "某市地铁是否因暴雨全线停运？",
          "search_keywords": ["某市 地铁 停运"],
          "preferred_source_types": ["官方通报"],
          "risk_level": "high",
          "sensitive_reason": "涉及公共安全",
          "is_opinion": false,
          "is_checkable": true
        },
        {
          "text": "目前已有多人失联",
          "claim_type": "数据声明",
          "entities": ["失联人员"],
          "time_reference": "当前",
          "location": "某市",
          "verification_priority": 1,
          "verification_question": "暴雨是否已导致多人失联？",
          "search_keywords": ["某市 暴雨 失联"],
          "preferred_source_types": ["应急部门"],
          "risk_level": "high",
          "sensitive_reason": "涉及人员伤亡",
          "is_opinion": false,
          "is_checkable": true
        },
        {
          "text": "教育部门通知全市学校停课三天",
          "claim_type": "政策通知",
          "entities": ["教育部门", "学校"],
          "time_reference": "未来三天",
          "location": "某市",
          "verification_priority": 2,
          "verification_question": "教育部门是否通知全市学校停课三天？",
          "search_keywords": ["某市 学校 停课"],
          "preferred_source_types": ["教育部门"],
          "risk_level": "medium",
          "sensitive_reason": null,
          "is_opinion": false,
          "is_checkable": true
        }
      ],
      "summary": "拆分为3个主张"
    }
    """
    mock_response = _mock_llm_response(llm_output)

    state = AgentState(
        original_text="网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。",
        mode="llm",
    )

    orig_key, orig_model = _mock_settings_configured()
    try:
        with patch("src.llm.client.LLMClient._build_client") as mock_build:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_build.return_value = mock_client

            state = decompose_claims_llm(state)
    finally:
        _restore_settings(orig_key, orig_model)

    assert len(state.claims) >= 3


def test_decompose_llm_case3_opinion():
    """案例3：纯观点文本应返回空主张。"""
    llm_output = """
    {
      "claims": [],
      "summary": "文本主要为个人观点表达，无可核查的事实主张"
    }
    """
    mock_response = _mock_llm_response(llm_output)

    state = AgentState(
        original_text="我觉得这家媒体越来越不负责任了。",
        mode="llm",
    )

    orig_key, orig_model = _mock_settings_configured()
    try:
        with patch("src.llm.client.LLMClient._build_client") as mock_build:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_build.return_value = mock_client

            state = decompose_claims_llm(state)
    finally:
        _restore_settings(orig_key, orig_model)

    assert len(state.claims) == 0
    assert state.errors == []


def test_decompose_llm_empty_input():
    """空输入应阻止执行。"""
    state = AgentState(original_text="", mode="llm")
    state = decompose_claims_llm(state)

    assert len(state.claims) == 0
    assert len(state.errors) > 0
    assert "为空" in state.errors[0]


def test_decompose_llm_call_failure_no_silent_fallback():
    """LLM 调用失败时不允许静默切换到模拟数据。"""
    from src.llm.client import LLMError

    state = AgentState(
        original_text="网传某市因暴雨导致地铁全线停运。",
        mode="llm",
    )

    orig_key, orig_model = _mock_settings_configured()
    try:
        with patch("src.llm.client.LLMClient.chat_json", side_effect=LLMError("模拟调用失败")):
            state = decompose_claims_llm(state)
    finally:
        _restore_settings(orig_key, orig_model)

    # 不应有任何主张被生成（不能静默回退到模拟数据）
    assert len(state.claims) == 0
    assert len(state.errors) > 0
    assert "失败" in state.errors[0]


def test_decompose_llm_markdown_fenced():
    """模型返回带 Markdown 围栏的 JSON 应能正常解析。"""
    llm_output = '```json\n{"claims": [{"text": "测试主张", "claim_type": "事件陈述", "entities": [], "verification_priority": 1, "risk_level": "low", "is_opinion": false, "is_checkable": true}], "summary": "测试"}\n```'
    mock_response = _mock_llm_response(llm_output)

    state = AgentState(original_text="测试输入", mode="llm")

    orig_key, orig_model = _mock_settings_configured()
    try:
        with patch("src.llm.client.LLMClient._build_client") as mock_build:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_build.return_value = mock_client

            state = decompose_claims_llm(state)
    finally:
        _restore_settings(orig_key, orig_model)

    assert len(state.claims) == 1
    assert state.claims[0].text == "测试主张"
