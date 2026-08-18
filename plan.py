"""核查计划节点。

提供两种实现：
- plan_verification_mock：原有基于规则的模拟计划（演示模式）
- plan_verification_llm：调用真实大模型生成计划（真实 LLM 模式）

两者均通过 plan_verification 入口分发，保持工作流编排不变。
"""

from pydantic import BaseModel, Field

from src.models import AgentState, VerificationPlan
from src.prompts.system_prompts import PLAN_SYSTEM_PROMPT


class _PlanOutput(BaseModel):
    """大模型计划结果的内部校验模型。"""

    plans: list[dict] = Field(default_factory=list)


def plan_verification(state: AgentState) -> AgentState:
    """核查计划入口：根据 state.mode 分发到模拟或真实实现。"""
    if state.mode in ("llm", "full"):
        return plan_verification_llm(state)
    return plan_verification_mock(state)


def plan_verification_mock(state: AgentState) -> AgentState:
    """演示模式：基于规则生成核查计划。"""
    state.log(
        step="plan",
        action="正在为每个主张制定核查计划（演示模式）",
        status="running",
        details={"claim_count": len(state.claims)},
    )

    plans: list[VerificationPlan] = []
    for claim in state.claims:
        if "地铁" in claim.text:
            plan = VerificationPlan(
                claim_id=claim.claim_id,
                verification_steps=["查找轨道交通集团官方通报", "交叉验证媒体报道"],
                search_queries=["某市 地铁 暴雨 停运", "轨道交通集团 暴雨 通知"],
                preferred_sources=["官方通报", "权威媒体"],
                required_evidence_level="权威一手来源",
                priority=1,
                priority_reason="涉及公共安全，需权威一手来源",
            )
        elif "失联" in claim.text:
            plan = VerificationPlan(
                claim_id=claim.claim_id,
                verification_steps=["查找应急管理局通报", "核实社媒视频来源"],
                search_queries=["某市 暴雨 失联", "应急管理局 人员失踪"],
                preferred_sources=["官方通报", "应急部门"],
                required_evidence_level="多源交叉验证",
                priority=1,
                priority_reason="涉及人员伤亡，需多源交叉验证",
            )
        elif "停课" in claim.text:
            plan = VerificationPlan(
                claim_id=claim.claim_id,
                verification_steps=["查找教育局官方通知", "核实学校原始通知"],
                search_queries=["某市 学校 停课 三天", "教育局 暴雨 停课通知"],
                preferred_sources=["教育部门", "学校官网"],
                required_evidence_level="权威一手来源",
                priority=2,
                priority_reason="涉及公共教育安排，需官方来源",
            )
        else:
            plan = VerificationPlan(
                claim_id=claim.claim_id,
                verification_steps=["查找权威媒体报道", "交叉验证信息"],
                search_queries=[claim.text[:20]],
                preferred_sources=["权威媒体", "官方通报"],
                required_evidence_level="一般证据",
                priority=3,
                priority_reason="常规核查",
            )
        plans.append(plan)

    state.verification_plan = plans
    state.log(
        step="plan",
        action=f"核查计划已生成（演示模式），共 {len(plans)} 条计划",
        status="completed",
        details={
            "plan_count": len(plans),
            "search_queries_per_claim": {p.claim_id: len(p.search_queries) for p in plans},
        },
    )
    return state


def plan_verification_llm(state: AgentState) -> AgentState:
    """真实 LLM 模式：调用大模型生成核查计划。"""
    from src.config import settings
    from src.llm.client import LLMClient, LLMError

    if not state.claims:
        state.log(
            step="plan",
            action="无可核查主张，跳过计划生成",
            status="completed",
        )
        state.verification_plan = []
        return state

    # 未配置 API 密钥时直接失败，避免发起真实网络请求
    if not settings.llm_configured():
        state.log(
            step="plan",
            action="未配置 LLM_API_KEY 或 LLM_MODEL，阻止执行真实计划生成",
            status="error",
        )
        state.errors.append(
            "未配置大模型，请在 .env 中填写 LLM_API_KEY 和 LLM_MODEL 后再使用真实 LLM 模式。"
        )
        state.verification_plan = []
        return state

    state.log(
        step="plan",
        action="正在调用大模型生成核查计划（真实 LLM 模式）",
        status="running",
        details={"claim_count": len(state.claims)},
    )

    # 构造输入：将主张列表序列化为 JSON 供模型参考
    import json

    claims_for_model = [
        {
            "claim_id": c.claim_id,
            "text": c.text,
            "claim_type": c.claim_type,
            "entities": c.entities,
            "time_reference": c.time_reference,
            "location": c.location,
            "risk_level": c.risk_level,
            "verification_question": c.verification_question,
        }
        for c in state.claims
    ]
    user_prompt = (
        f"请为以下主张列表分别制定核查计划：\n\n"
        f"{json.dumps(claims_for_model, ensure_ascii=False, indent=2)}"
    )

    try:
        client = LLMClient()
        result: _PlanOutput = client.chat_json(
            system_prompt=PLAN_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=_PlanOutput,
        )
    except LLMError as e:
        state.log(
            step="plan",
            action="大模型计划生成失败",
            status="error",
            details={"error": str(e)},
        )
        state.errors.append(f"核查计划生成失败：{e}")
        state.verification_plan = []
        return state

    # 将模型返回的计划映射到已有 claim_id
    claim_id_set = {c.claim_id for c in state.claims}
    plans: list[VerificationPlan] = []
    for raw_plan in result.plans:
        claim_id = raw_plan.get("claim_id", "")
        if claim_id not in claim_id_set:
            # 跳过模型臆造的 claim_id
            continue
        try:
            plan = VerificationPlan(
                **{k: v for k, v in raw_plan.items() if k in VerificationPlan.model_fields.keys()}
            )
            plans.append(plan)
        except Exception as e:
            state.log(
                step="plan",
                action=f"计划 {claim_id} 字段校验失败，已跳过",
                status="running",
                details={"error": str(e)},
            )

    state.verification_plan = plans
    state.log(
        step="plan",
        action=f"核查计划已生成（真实 LLM 模式），共 {len(plans)} 条计划",
        status="completed",
        details={
            "plan_count": len(plans),
            "search_queries_per_claim": {p.claim_id: len(p.search_queries) for p in plans},
        },
    )
    return state
