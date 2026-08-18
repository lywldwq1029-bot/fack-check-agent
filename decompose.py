"""主张拆解节点。

提供两种实现：
- decompose_claims_mock：原有基于关键词的模拟拆解（演示模式）
- decompose_claims_llm：调用真实大模型拆解（真实 LLM 模式）

两者均通过 decompose_claims 入口分发，保持工作流编排不变。
"""

from pydantic import BaseModel, Field

from src.models import AgentState, Claim
from src.prompts.system_prompts import DECOMPOSE_SYSTEM_PROMPT


class _DecomposeOutput(BaseModel):
    """大模型拆解结果的内部校验模型。"""

    claims: list[dict] = Field(default_factory=list)
    summary: str = Field(default="")


def decompose_claims(state: AgentState) -> AgentState:
    """主张拆解入口：根据 state.mode 分发到模拟或真实实现。"""
    if state.mode in ("llm", "full"):
        return decompose_claims_llm(state)
    return decompose_claims_mock(state)


def decompose_claims_mock(state: AgentState) -> AgentState:
    """演示模式：基于关键词的模拟拆解。"""
    text = state.original_text
    state.log(
        step="decompose",
        action="正在拆解新闻文本为可核查主张（演示模式）",
        status="running",
        details={"text_length": len(text)},
    )

    claims: list[Claim] = []

    # TF家族 + 恋爱传闻 案例（本阶段验收案例）
    if "TF家族" in text and ("情侣" in text or "恋爱" in text):
        # c1: 左航身份前提（context）
        claims.append(Claim(
            claim_id="c1",
            text="左航是TF家族三代成员",
            claim_type="身份声明",
            claim_role="context",
            entities=["左航", "TF家族", "TF家族三代"],
            verification_priority=1,
            verification_question="左航是否为TF家族三代成员？",
            search_keywords=[
                "左航 TF家族 三代 成员",
                "左航 官方 个人介绍 TF家族",
                "TF家族三代 成员名单 左航",
            ],
            preferred_source_types=["经纪公司官方介绍", "权威媒体", "官方百科"],
            risk_level="low",
            needs_background_verification=True,
            entity_aliases={
                "左航": ["TF家族", "三代", "成员"],
                "TF家族": ["时代峰峻", "练习生"],
            },
        ))
        # c2: 邓佳鑫身份前提（context）
        claims.append(Claim(
            claim_id="c2",
            text="邓佳鑫是TF家族三代成员",
            claim_type="身份声明",
            claim_role="context",
            entities=["邓佳鑫", "TF家族", "TF家族三代"],
            verification_priority=1,
            verification_question="邓佳鑫是否为TF家族三代成员？",
            search_keywords=[
                "邓佳鑫 TF家族 三代 成员",
                "邓佳鑫 官方 个人介绍 TF家族",
                "TF家族三代 成员名单 邓佳鑫",
            ],
            preferred_source_types=["经纪公司官方介绍", "权威媒体", "官方百科"],
            risk_level="low",
            needs_background_verification=True,
            entity_aliases={
                "邓佳鑫": ["TF家族", "三代", "成员"],
                "TF家族": ["时代峰峻", "练习生"],
            },
        ))
        # c3: 核心传闻（core，高风险私生活指控）
        claims.append(Claim(
            claim_id="c3",
            text="左航和邓佳鑫曾经是情侣",
            claim_type="事件陈述",
            claim_role="core",
            entities=["左航", "邓佳鑫"],
            verification_priority=1,
            verification_question="左航和邓佳鑫是否曾经为情侣关系？",
            search_keywords=[
                "左航 邓佳鑫 情侣 传闻",
                "左航 邓佳鑫 TF家族 恋爱 声明",
                "左航 邓佳鑫 当事人 回应",
            ],
            preferred_source_types=["当事人公开声明", "经纪公司公告", "署名主流媒体采访"],
            risk_level="high",
            sensitive_reason="涉及自然人私生活与名誉的高风险传闻",
            depends_on_claim_ids=["c1", "c2"],
            entity_aliases={
                "左航": ["TF家族", "三代", "成员"],
                "邓佳鑫": ["TF家族", "三代", "成员"],
            },
        ))
        # c4: 2020年时间细节（causal_or_detail）
        if "2020年" in text or "2020" in text:
            claims.append(Claim(
                claim_id="c4",
                text="左航和邓佳鑫在2020年处于恋爱关系",
                claim_type="时间声明",
                claim_role="causal_or_detail",
                entities=["左航", "邓佳鑫", "2020年"],
                time_reference="2020年",
                verification_priority=2,
                verification_question="左航和邓佳鑫在2020年是否处于恋爱关系？",
                search_keywords=[
                    "左航 邓佳鑫 2020 恋爱",
                    "左航 邓佳鑫 2020年 情侣 传闻",
                    "TF家族三代 2020 左航 邓佳鑫 关系",
                ],
                preferred_source_types=["当事人声明", "同期权威报道", "经纪公司公告"],
                risk_level="high",
                sensitive_reason="涉及自然人私生活与时间精准指控，高风险",
                depends_on_claim_ids=["c1", "c2", "c3"],
                entity_aliases={
                    "左航": ["TF家族", "三代", "成员"],
                    "邓佳鑫": ["TF家族", "三代", "成员"],
                },
            ))

    if "地铁" in text and "停运" in text:
        claims.append(
            Claim(
                claim_id="c1" if not claims else f"c{len(claims) + 1}",
                text="某市因暴雨导致地铁全线停运",
                claim_type="事件陈述",
                claim_role="core",
                entities=["某市", "地铁", "暴雨"],
                time_reference="当前",
                location="某市",
                verification_priority=1,
                verification_question="某市地铁是否因暴雨全线停运？",
                search_keywords=["某市 地铁 暴雨 停运", "轨道交通集团 暴雨 通知"],
                preferred_source_types=["官方通报", "权威媒体"],
                risk_level="medium",
                sensitive_reason="涉及公共安全信息",
            )
        )

    if "失联" in text or "多人" in text:
        claims.append(
            Claim(
                claim_id=f"c{len(claims) + 1}",
                text="暴雨已导致多人失联",
                claim_type="数据声明",
                claim_role="core",
                entities=["暴雨", "失联人员"],
                time_reference="当前",
                location="某市",
                verification_priority=1,
                verification_question="暴雨是否已导致多人失联？",
                search_keywords=["某市 暴雨 失联", "应急管理局 人员失踪"],
                preferred_source_types=["官方通报", "应急部门"],
                risk_level="high",
                sensitive_reason="涉及人员伤亡与公共安全",
            )
        )

    if "停课" in text and "学校" in text:
        claims.append(
            Claim(
                claim_id=f"c{len(claims) + 1}",
                text="教育部门通知全市学校停课三天",
                claim_type="政策通知",
                claim_role="core",
                entities=["教育部门", "全市学校"],
                time_reference="未来三天",
                location="某市",
                verification_priority=2,
                verification_question="教育部门是否发布了全市学校停课三天的通知？",
                search_keywords=["某市 学校 停课 三天", "教育局 暴雨 停课通知"],
                preferred_source_types=["教育部门", "学校官网"],
                risk_level="medium",
                sensitive_reason="涉及公共教育安排",
            )
        )

    if not claims:
        claims.append(
            Claim(
                claim_id="c1",
                text=text,
                claim_type="待分类",
                entities=[],
                time_reference="未知",
                location="未知",
                verification_priority=3,
                is_checkable=False,
                is_opinion=True,
            )
        )

    state.claims = claims
    high_risk_ids = [c.claim_id for c in claims if c.risk_level == "high"]
    state.log(
        step="decompose",
        action=f"已拆解出 {len(claims)} 个可核查主张（演示模式）",
        status="completed",
        details={
            "claim_ids": [c.claim_id for c in claims],
            "high_risk_claims": high_risk_ids,
        },
    )
    return state


def decompose_claims_llm(state: AgentState) -> AgentState:
    """真实 LLM 模式：调用大模型拆解新闻主张。"""
    from src.config import settings
    from src.llm.client import LLMClient, LLMError

    text = state.original_text.strip()
    if not text:
        state.log(
            step="decompose",
            action="输入文本为空，阻止执行拆解",
            status="error",
        )
        state.errors.append("输入文本为空，无法执行主张拆解。")
        return state

    # 未配置 API 密钥时直接失败，避免发起真实网络请求导致长时间超时
    if not settings.llm_configured():
        state.log(
            step="decompose",
            action="未配置 LLM_API_KEY 或 LLM_MODEL，阻止执行真实拆解",
            status="error",
        )
        state.errors.append(
            "未配置大模型，请在 .env 中填写 LLM_API_KEY 和 LLM_MODEL 后再使用真实 LLM 模式。"
        )
        return state

    state.log(
        step="decompose",
        action="正在调用大模型拆解新闻主张（真实 LLM 模式）",
        status="running",
        details={"text_length": len(text)},
    )

    try:
        client = LLMClient()
        user_prompt = f"请将以下新闻文本拆解为可核查主张：\n\n{text}"
        result: _DecomposeOutput = client.chat_json(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=_DecomposeOutput,
        )
    except LLMError as e:
        # 真实模式调用失败时不允许静默切换到模拟数据
        state.log(
            step="decompose",
            action="大模型拆解失败",
            status="error",
            details={"error": str(e)},
        )
        state.errors.append(f"主张拆解失败：{e}")
        return state

    # claim_id 由程序统一生成，不依赖模型
    claims: list[Claim] = []
    for idx, raw_claim in enumerate(result.claims, start=1):
        try:
            claim = Claim(
                claim_id=f"c{idx}",
                **{k: v for k, v in raw_claim.items() if k != "claim_id"},
            )
            claims.append(claim)
        except Exception as e:
            state.log(
                step="decompose",
                action=f"主张 c{idx} 字段校验失败，已跳过",
                status="running",
                details={"error": str(e), "raw_text": str(raw_claim.get("text", ""))[:100]},
            )

    if not claims:
        state.log(
            step="decompose",
            action=result.summary or "文本主要为观点或信息不足，无可核查主张",
            status="completed",
            details={"summary": result.summary},
        )
        state.claims = []
        return state

    # 超过 8 条时截断
    if len(claims) > 8:
        state.log(
            step="decompose",
            action=f"模型返回 {len(claims)} 条主张，已截断为 8 条",
            status="running",
        )
        claims = claims[:8]

    state.claims = claims
    high_risk_ids = [c.claim_id for c in claims if c.risk_level == "high"]
    state.log(
        step="decompose",
        action=f"大模型拆解完成，共识别 {len(claims)} 个可核查主张",
        status="completed",
        details={
            "claim_ids": [c.claim_id for c in claims],
            "high_risk_claims": high_risk_ids,
            "summary": result.summary,
        },
    )
    return state
