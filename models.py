"""溯真核心数据模型。

使用 Pydantic 定义结构化数据，确保工作流各节点之间的数据契约清晰、可验证。
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


VERDICT_TYPES = Literal[
    "基本属实",
    "部分属实",
    "存在错误",
    "已证伪",
    "证据不足",
    "暂无法核查",
]
"""核查结论类型（专业判定标准）。

严格规则：
- 基本属实：有权威来源支持，核心内容正确
- 部分属实：部分内容有证据，部分未证实
- 存在错误：有可靠来源指出具体错误
- 已证伪：权威来源明确否认，或可靠事实直接矛盾
- 证据不足：没有足够证据支持或反驳（不等于已证伪）
- 暂无法核查：搜索失败且无缓存可用

注意：
- "没有证据支持"只能判为"证据不足"，不能判为"已证伪"
- "已证伪"必须有权威来源明确否认
- 私人关系、怀孕、违法、健康等敏感信息，没有A级来源时默认"证据不足"
"""

SOURCE_GRADES = Literal["A", "B", "C", "D"]
"""证据来源可信度等级（专业分级）。

A级：官方机构、当事人正式声明、原始文件
B级：正规媒体原创报道、专业机构
C级：百科、聚合媒体、普通转载
D级：论坛、自媒体、匿名爆料
"""

SOURCE_GRADE_DESCRIPTIONS = {
    "A": "官方来源/原始文件",
    "B": "权威媒体/专业机构",
    "C": "一般媒体/转载",
    "D": "自媒体/论坛",
}

# 主张类型
CLAIM_TYPES = Literal["事实", "观点", "预测", "私人传闻"]

# 敏感度等级
SENSITIVITY_LEVELS = Literal["低", "中", "高"]

# Agent 决策类型
AGENT_ACTIONS = Literal["STOP", "SEARCH_AGAIN"]

SUPPORT_TYPES = Literal["supports", "refutes", "partial", "unclear"]

RISK_LEVELS = Literal["低", "中", "高", "不确定"]

EVIDENCE_LEVELS = Literal["一般证据", "多源交叉验证", "权威一手来源", "极高强度证据"]

# 证据立场：与 SUPPORT_TYPES 区分，新增 context 和 irrelevant
EVIDENCE_STANCE = Literal["supports", "refutes", "context", "irrelevant"]

# 证据直接性
EVIDENCE_DIRECTNESS = Literal["direct", "indirect", "hearsay", "unclear"]

# 证据提取状态
EXTRACTION_STATUS = Literal[
    "success",
    "insufficient_content",
    "invalid_excerpt",
    "failed",
]

# 主张角色：区分背景前提、核心事件、原因/时间/数量等附加细节
CLAIM_ROLE = Literal["context", "core", "causal_or_detail"]


class AgentDecision(BaseModel):
    """Agent 决策结构：结构化的证据充分性评估。

    用于指导 LLM 在首次搜索后进行反思，决定是否需要补搜。
    """
    normalized_claim: str = Field(..., description="标准化主张")
    claim_type: CLAIM_TYPES = Field(..., description="主张类型：事实/观点/预测/私人传闻")
    sensitivity: SENSITIVITY_LEVELS = Field(..., description="敏感度等级")
    evidence_requirement: str = Field(..., description="形成结论需要什么证据")
    evidence_sufficient: bool = Field(..., description="当前证据是否充分")
    missing_evidence: list[str] = Field(default_factory=list, description="还缺少什么")
    action: AGENT_ACTIONS = Field(..., description="STOP 或 SEARCH_AGAIN")
    supplemental_query: Optional[str] = Field(None, description="补搜关键词（仅当 action=SEARCH_AGAIN）")
    action_reason: str = Field(..., description="一句话解释为什么停止或补搜")


class KeyEvidenceCard(BaseModel):
    """关键证据卡片：用于 UI 上突出展示的重要证据摘要。"""
    card_id: str = Field(..., description="卡片唯一标识")
    title: str = Field(..., description="证据标题")
    source_url: str = Field(..., description="来源链接")
    source_grade: str = Field(..., description="来源等级 A/B/C/D/E")
    summary: str = Field(..., description="50字摘要")
    directly_supports: bool = Field(default=True, description="是否直接支持/反驳主张")


class Claim(BaseModel):
    """可核查主张：从新闻中拆解出的最小核查单元。"""

    claim_id: str = Field(..., description="主张唯一标识")
    text: str = Field(..., description="主张文本")
    claim_type: str = Field(..., description="主张类型，如事件陈述、数据声明、归因判断等")
    entities: list[str] = Field(default_factory=list, description="提到的关键实体")
    time_reference: Optional[str] = Field(None, description="时间参照，如当前、过去某时刻")
    location: Optional[str] = Field(None, description="地点信息")
    verification_priority: int = Field(default=1, ge=1, le=5, description="核查优先级 1-5，数字越小越优先")

    # 以下为本阶段新增字段（均有默认值，保持与旧代码兼容）
    verification_question: Optional[str] = Field(None, description="该主张需要回答的核查问题")
    search_keywords: list[str] = Field(default_factory=list, description="建议搜索关键词")
    preferred_source_types: list[str] = Field(default_factory=list, description="优先寻找的来源类型")
    risk_level: RISK_LEVELS = Field(default="low", description="风险等级")
    sensitive_reason: Optional[str] = Field(None, description="涉及名誉、隐私、公共安全等风险时的说明")
    is_opinion: bool = Field(default=False, description="是否属于观点而非可验证事实")
    is_checkable: bool = Field(default=True, description="当前是否具备可核查性")

    # ===== 本阶段新增：身份前提/角色/依赖 =====
    claim_role: CLAIM_ROLE = Field(
        default="core",
        description=(
            "主张角色："
            "context=背景或身份前提（人物身份、机构身份、地点时间）；"
            "core=核心事件主张；"
            "causal_or_detail=原因、时间、数量等附加细节主张"
        ),
    )
    depends_on_claim_ids: list[str] = Field(
        default_factory=list,
        description="该主张依赖的背景/前提主张 ID 列表（如恋爱传闻依赖两人的身份前提）",
    )
    entity_aliases: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "实体及其常见别名/消歧上下文。"
            "键：实体名；值：用于消歧的上下文字段列表，例如 {'左航': ['TF家族', '三代', '成员']}"
        ),
    )
    needs_background_verification: bool = Field(
        default=False,
        description="是否需要独立核查其背景事实（人物身份/机构/地点/时间前提等通常为 True）",
    )


class VerificationPlan(BaseModel):
    """单条主张的核查计划。"""

    claim_id: str = Field(..., description="关联的主张 ID")
    verification_steps: list[str] = Field(default_factory=list, description="核查步骤")
    search_queries: list[str] = Field(default_factory=list, description="建议使用的搜索语句（2-4 条）")
    preferred_sources: list[str] = Field(default_factory=list, description="优先访问的来源类型")
    required_evidence_level: EVIDENCE_LEVELS = Field(default="一般证据", description="所需证据强度")
    priority: int = Field(default=3, ge=1, le=5, description="优先级 1-5，数字越小越优先")
    priority_reason: Optional[str] = Field(None, description="该优先级的理由")


class Evidence(BaseModel):
    """证据：支持或反驳某一主张的信息单元。

    专业评价标准：
    - source_grade: A/B/C/D 等级
    - directly_supports: 是否直接支持主张
    - is_primary_source: 是否为一手来源
    - is_independent: 是否独立于其他来源（非转载）
    """

    evidence_id: str = Field(..., description="证据唯一标识")
    claim_id: str = Field(..., description="关联的主张 ID")
    source_title: str = Field(..., description="来源标题")
    source_url: str = Field(..., description="来源链接")
    publisher: str = Field(..., description="发布者")
    published_at: Optional[datetime] = Field(None, description="来源发布时间")
    retrieved_at: datetime = Field(default_factory=datetime.now, description="证据被检索时间")
    evidence_summary: str = Field(..., description="证据内容摘要（Tavily 原文，后台保留）")
    summary: str = Field(default="", description="一句话概括（LLM 生成，用于页面展示，50字以内）")
    source_type: str = Field(..., description="来源类型，如官方通报、权威媒体、社交媒体")
    source_grade: SOURCE_GRADES = Field(..., description="来源可信度等级（A/B/C/D）")
    supports_or_refutes: SUPPORT_TYPES = Field(..., description="证据对主张的态度（旧字段，兼容保留）")
    is_primary_source: bool = Field(default=False, description="是否为一手来源")
    reliability_reason: str = Field(..., description="可信度评级理由")

    # 专业评价字段
    directly_supports: bool = Field(default=False, description="是否直接支持主张")
    is_independent: bool = Field(default=True, description="是否独立于其他来源（非转载）")
    evidence_key_points: list[str] = Field(default_factory=list, description="关键证据点列表")

    # 以下为本阶段新增字段（均有默认值，保持与旧代码兼容）
    search_query: Optional[str] = Field(None, description="检索到该证据的搜索语句")
    source_domain: Optional[str] = Field(None, description="来源域名")
    source_content: Optional[str] = Field(None, description="网页抓取的原始内容（markdown）")
    relevant_excerpt: Optional[str] = Field(None, description="与主张最相关的原文片段")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="相关性评分 0-1")
    evidence_stance: EVIDENCE_STANCE = Field(default="context", description="证据立场")
    directness: EVIDENCE_DIRECTNESS = Field(default="unclear", description="证据直接性")
    independence_group: Optional[str] = Field(None, description="独立来源分组（转载同源时归为同组）")
    extraction_status: EXTRACTION_STATUS = Field(default="success", description="证据提取状态")


class ClaimResult(BaseModel):
    """单条主张的核查结果。

    字段兼容两种来源：
    - 运行期构建（Pydantic 模型实例）：claim 字段为 Claim 对象，verdict 存于 verdict。
    - docx_exporter 旧版读取（dict 字段）：保留 text/rationale/missing_info 别名兼容。
    """

    claim: Claim = Field(..., description="被核查的主张")
    verdict: VERDICT_TYPES = Field(..., description="核查结论")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")
    reasoning: str = Field(..., description="结论推理过程")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑该结论的证据列表")
    missing_information: Optional[str] = Field(None, description="仍缺少的关键信息")

    # ===== 兼容字段（docx_exporter 旧写法 / 旧 workflow 节点可能使用）=====
    @property
    def claim_id(self) -> str:
        return self.claim.claim_id

    @property
    def rationale(self) -> str:
        """docx_exporter / 旧测试可能读取：result.rationale。"""
        return self.reasoning

    @property
    def justification(self) -> str:
        """docx_exporter 可能读取：result.justification（旧字段别名）。"""
        return self.reasoning

    @property
    def missing_info(self) -> str:
        """docx_exporter 可能读取：result.missing_info。"""
        return self.missing_information or ""

    @property
    def used_evidence_ids(self) -> list[str]:
        """docx_exporter 可能读取：result.used_evidence_ids（旧字段别名）。"""
        return [e.evidence_id for e in self.evidence]

    @property
    def supporting_evidence_ids(self) -> list[str]:
        return [e.evidence_id for e in self.evidence if e.evidence_stance in {"supports", "context"}]

    @property
    def opposing_evidence_ids(self) -> list[str]:
        return [e.evidence_id for e in self.evidence if e.evidence_stance == "refutes"]

    @property
    def unresolved_questions(self) -> list[str]:
        if self.missing_information:
            return [self.missing_information]
        return []


# ===== ClaimResult 工厂函数 =====

def create_claim_result(
    claim: "Claim",
    verdict: str,
    confidence: float,
    reasoning: str,
    evidence: list["Evidence"] | None = None,
    missing_information: str | None = None,
) -> "ClaimResult":
    """统一构建 ClaimResult。

    所有 ClaimResult 创建必须通过此函数，确保字段契约一致。

    参数:
        claim: 被核查的主张
        verdict: 核查结论（VERDICT_TYPES 之一）
        confidence: 置信度 0.0-1.0（无法核查时允许传 0.0）
        reasoning: 结论推理过程
        evidence: 支撑证据列表
        missing_information: 缺少的关键信息
    """
    return ClaimResult(
        claim=claim,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        evidence=evidence or [],
        missing_information=missing_information,
    )


class TimelineEvent(BaseModel):
    """时间线事件：用于还原新闻发展过程。"""

    event_time: Optional[datetime] = Field(None, description="事件发生时间")
    description: str = Field(..., description="事件描述")
    source_url: Optional[str] = Field(None, description="来源链接")


# 工作流阶段的规范内部标识（用于 structured 状态渲染，不依赖中文日志文本）
WORKFLOW_PHASES: list[str] = [
    "init",          # 初始化
    "decompose",     # 主张拆解
    "plan",          # 核查计划
    "search",        # 第一轮证据检索
    "sufficiency",   # 证据充分性评估 + 自动第二轮补充检索（如需要）
    "evaluate",      # 来源分级与交叉验证
    "report",        # 生成报告
    "memory",        # 保存核查记忆
]
WORKFLOW_PHASE_LABELS: dict[str, str] = {
    "init": "初始化",
    "decompose": "正在拆解主张",
    "plan": "正在制定计划",
    "search": "正在搜索网页证据",
    "sufficiency": "正在评估证据充分性与补充检索",
    "evaluate": "正在交叉验证与独立判定",
    "report": "正在生成报告",
    "memory": "保存核查记忆",
}


class FactCheckReport(BaseModel):
    """统一核查报告模型。

    版本历史：
    - v1: 初始版本
    - v2: 增加 schema_version、credibility_score 改为 Optional[int]
    """

    # 数据契约版本号（session_state 中用于检测旧报告）
    schema_version: int = Field(
        default=2,
        description="数据契约版本号，模型字段变化时递增",
    )

    original_text: str = Field(..., description="原始新闻文本")
    overall_verdict: VERDICT_TYPES = Field(..., description="总体结论")
    overall_summary: str = Field(..., description="总体摘要（一句话核心理由）")
    claim_results: list[ClaimResult] = Field(default_factory=list, description="各主张核查结果")
    timeline: list[TimelineEvent] = Field(default_factory=list, description="事件时间线")
    propagation_risk: str = Field(..., description="传播风险等级与说明")
    risk_level: str = Field(default="不确定", description="传播风险等级：高/中/低/不确定")
    risk_reason: str = Field(default="", description="传播风险原因")
    risk_factors: list[str] = Field(default_factory=list, description="传播风险因素列表")
    unresolved_questions: list[str] = Field(default_factory=list, description="仍待核实的问题")
    execution_log: list[dict] = Field(
        default_factory=list,
        description="Agent 执行日志（从 AgentState.execution_log 显式传入，包含 step/action/status/details）",
    )

    # ========== Agent 决策轨迹（折叠展示）==========
    decision_trace: list[dict] = Field(
        default_factory=list,
        description="Agent 决策轨迹：包含查询历史、搜索关键词、证据评估、补搜决策等",
    )
    agent_decision: Optional[AgentDecision] = Field(
        default=None,
        description="Agent 最后的结构化决策",
    )
    did_supplemental_search: bool = Field(
        default=False,
        description="是否执行了补搜",
    )
    tool_calls_count: int = Field(
        default=0,
        description="总共调用工具次数（Tavily + LLM）",
    )

    # ========== 历史核查记忆 ==========
    historical_matches: list[dict] = Field(
        default_factory=list,
        description="命中的历史核查记录（仅作参考，不代替当前证据）",
    )

    # ========== 关键证据卡片 ==========
    key_evidence_cards: list[dict] = Field(
        default_factory=list,
        description="关键证据卡片列表，包含标题、来源、等级、50字摘要",
    )

    # ========== 可信度评估 ==========
    credibility_score: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="整体可信度评分 0-100，失败时为 None",
    )
    recommendation: str = Field(
        default="",
        description="传播建议：如'不建议继续传播'、'可谨慎参考'等",
    )

    # ========== 结构化进度状态（前端渲染使用，不依赖 execution_log 中文匹配）==========
    current_step: str = Field(
        default="init",
        description="当前执行阶段：init/decompose/plan/search/evaluate/report/memory/completed/failed",
    )
    completed_steps: list[str] = Field(
        default_factory=list,
        description="已成功完成的阶段（使用 WORKFLOW_PHASES 中定义的标识）",
    )
    skipped_steps: list[str] = Field(
        default_factory=list,
        description="明确跳过的阶段（例如记忆功能关闭时 memory 会出现在此）",
    )
    progress_percent: int = Field(
        default=0, ge=0, le=100,
        description="工作流整体进度百分比，结束时必须为 100",
    )
    workflow_completed: bool = Field(
        default=False,
        description="工作流是否已成功结束（完成/失败均可能，失败时为 False + workflow_error 有值）",
    )
    workflow_error: Optional[str] = Field(
        default=None,
        description="若发生异常，记录失败所在阶段标识；未失败时为 None",
    )

    generated_at: datetime = Field(default_factory=datetime.now, description="报告生成时间")


# 数据契约版本号 - 模型字段变化时递增，session_state 中旧报告将被自动清除
REPORT_SCHEMA_VERSION = 2

# 核查结果状态枚举
CHECK_STATUS_SUCCESS = "success"       # 正常完成
CHECK_STATUS_PARTIAL = "partial"       # 部分完成（搜索失败但有缓存等）
CHECK_STATUS_UNAVAILABLE = "unavailable"  # 服务不可用（完全失败）


class CheckResult(BaseModel):
    """统一核查结果，供上层调用方使用。

    无论成功、部分成功还是完全失败，都通过此结构返回。
    Python 异常不得直接传递到页面层。
    """
    status: str = Field(
        default=CHECK_STATUS_UNAVAILABLE,
        description="核查状态: success/partial/unavailable",
    )
    report: Optional[FactCheckReport] = Field(
        default=None,
        description="核查报告，所有状态都有报告",
    )
    error_message: str = Field(
        default="",
        description="中文友好错误消息，不包含 Python traceback",
    )

    @property
    def is_ok(self) -> bool:
        return self.status == CHECK_STATUS_SUCCESS

    @property
    def is_partial(self) -> bool:
        return self.status == CHECK_STATUS_PARTIAL


def _base_report_kwargs(original_text: str) -> dict:
    """构建报告的基础字段，确保所有工厂函数都有一致的默认值。"""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "original_text": original_text,
        "claim_results": [],
        "timeline": [],
        "risk_level": "不确定",
        "risk_reason": "",
        "risk_factors": [],
        "unresolved_questions": [],
        "execution_log": [],
        "decision_trace": [],
        "agent_decision": None,
        "did_supplemental_search": False,
        "tool_calls_count": 0,
        "historical_matches": [],
        "key_evidence_cards": [],
        "credibility_score": None,
        "recommendation": "",
        "current_step": "init",
        "completed_steps": [],
        "skipped_steps": [],
        "progress_percent": 0,
        "workflow_completed": False,
        "workflow_error": None,
        "generated_at": datetime.now(),
    }


def build_failure_report(
    original_text: str,
    error_msg: str,
    error_stage: str = "init",
) -> "FactCheckReport":
    """统一构建失败报告。所有失败态必须通过此函数创建，避免遗漏必填字段。

    注意：不再使用"核查失败"verdict，统一使用"暂无法核查"。
    propagation_risk 固定为"待判断"，绝不留空。
    """
    error_text = str(error_msg) if not isinstance(error_msg, str) else error_msg
    kwargs = _base_report_kwargs(original_text)
    kwargs.update({
        "overall_verdict": "暂无法核查",
        "overall_summary": error_text,
        "propagation_risk": "待判断",
        "unresolved_questions": [original_text[:100]] if original_text else [],
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "step": error_stage,
            "action": error_text,
            "status": "error",
            "details": {"error": error_text},
        }],
        "current_step": "failed",
        "completed_steps": ["receive"],
        "progress_percent": 0,
        "workflow_completed": False,
        "workflow_error": error_stage,
    })
    return FactCheckReport(**kwargs)


def build_no_evidence_report(
    original_text: str,
    reason: str = "当前未取得公开证据，请稍后重试",
) -> "FactCheckReport":
    """构建搜索失败且无缓存可用时的优雅降级报告。

    工作流正常结束，进度显示"部分完成"（receive + search 完成，analyze/output 跳过）。
    不调用 LLM，不编造判断。
    """
    kwargs = _base_report_kwargs(original_text)
    kwargs.update({
        "overall_verdict": "暂无法核查",
        "overall_summary": reason,
        "propagation_risk": "待判断",
        "unresolved_questions": [original_text[:100]] if original_text else [],
        "execution_log": [{
            "timestamp": datetime.now().isoformat(),
            "step": "search",
            "action": reason,
            "status": "completed",
            "details": {"skipped": True, "reason": reason},
        }],
        "current_step": "completed",
        "completed_steps": ["receive", "search"],
        "skipped_steps": ["analyze", "output"],
        "progress_percent": 50,
        "workflow_completed": True,
        "workflow_error": None,
        "credibility_score": None,
        "recommendation": "请稍后重试或提供更多信息",
    })
    return FactCheckReport(**kwargs)


class AgentState(BaseModel):
    """Agent 工作流运行状态，记录全流程中间结果与日志。"""

    original_text: str = Field(..., description="用户输入的原始文本")
    claims: list[Claim] = Field(default_factory=list, description="拆解出的主张")
    verification_plan: list[VerificationPlan] = Field(default_factory=list, description="核查计划列表")
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict, description="每个主张对应的证据")
    claim_results: list[ClaimResult] = Field(default_factory=list, description="主张核查结果")
    report: Optional[FactCheckReport] = Field(None, description="最终报告")
    current_step: str = Field(default="init", description="当前执行阶段标识")
    execution_log: list[dict] = Field(default_factory=list, description="执行日志")
    errors: list[str] = Field(default_factory=list, description="运行中产生的错误")
    mode: str = Field(default="demo", description="运行模式：demo / llm / full")

    # ========== 结构化进度状态（与 FactCheckReport 同名字段含义一致）==========
    completed_steps: list[str] = Field(default_factory=list, description="已成功完成的阶段")
    skipped_steps: list[str] = Field(default_factory=list, description="明确跳过的阶段（如 save_to_memory=False）")
    progress_percent: int = Field(default=0, ge=0, le=100, description="工作流整体进度（0-100）")
    workflow_completed: bool = Field(default=False, description="是否已成功结束")
    workflow_error: Optional[str] = Field(default=None, description="失败所在阶段标识，未失败为 None")

    # 搜索统计（完整真实模式下记录）
    search_stats: dict = Field(
        default_factory=lambda: {
            "total_queries": 0,
            "total_results_fetched": 0,
            "valid_evidence_count": 0,
            "independent_sources_count": 0,
            "failed_queries": 0,
            "total_response_time_ms": 0,
        },
        description="搜索与证据提取统计",
    )

    # ========== 本阶段新增：通用元数据容器（证据充分性评估、第二轮检索等节点使用）==========
    # 每个 AgentState 实例独立：Pydantic default_factory=dict 在实例化时会生成新对象，不共享。
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "节点间临时共享的通用元数据容器。"
            "约定使用 setdefault/get 避免 KeyError，常用键："
            "_search_rounds: dict[str,int] 每条主张已执行检索轮次；"
            "follow_up: dict 第二轮补充检索计划；"
            "sufficiency: dict 每条主张充分性评估结果"
        ),
    )

    # ========== 本阶段新增：扁平化证据池（search 节点合并，sufficiency/evaluate 直接遍历）==========
    evidence_pool: list = Field(
        default_factory=list,
        description="扁平化证据池（所有主张证据的并集，便于 sufficiency 遍历与独立来源统计）。",
    )

    # ---- 内部总阶段数（用于按阶段线性计算进度）----
    _PHASES_TOTAL: int = 7  # init + decompose + plan + search + evaluate + report + memory

    def log(self, step: str, action: str, status: str = "running", details: Optional[dict] = None) -> None:
        """记录一条执行日志。"""
        self.execution_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "action": action,
                "status": status,
                "details": details or {},
            }
        )
        self.current_step = step

    # =========================================================================
    # 结构化进度辅助：统一更新阶段状态，不再依赖 execution_log 中文匹配
    # =========================================================================

    def _compute_progress(self) -> int:
        """按已完成 + 已跳过的阶段数，线性计算 0-100 的进度。"""
        done = len(set(self.completed_steps) | set(self.skipped_steps))
        total = max(self._PHASES_TOTAL, 1)
        percent = int(round(done / total * 100))
        if percent > 100:
            percent = 100
        return percent

    def mark_step_started(self, step: str, action: str, details: Optional[dict] = None) -> None:
        """统一：阶段开始时写入 running 日志并更新 current_step。"""
        self.current_step = step
        self.log(step=step, action=action, status="running", details=details)

    def mark_step_completed(self, step: str, action: str, details: Optional[dict] = None) -> None:
        """统一：阶段成功完成时写入 completed 日志、加入 completed_steps、刷新进度。"""
        if step not in self.completed_steps:
            self.completed_steps.append(step)
        self.progress_percent = self._compute_progress()
        self.current_step = step
        self.log(step=step, action=action, status="completed", details=details)

    def mark_step_skipped(self, step: str, action: str, details: Optional[dict] = None) -> None:
        """统一：阶段明确跳过时写入 skipped 日志、加入 skipped_steps、刷新进度。"""
        if step not in self.skipped_steps:
            self.skipped_steps.append(step)
        self.progress_percent = self._compute_progress()
        self.log(step=step, action=action, status="skipped", details=details)

    def mark_failed(self, step: str, action: str, error_msg: str, details: Optional[dict] = None) -> None:
        """统一：阶段失败时写入 error 日志、设置 workflow_error、current_step=failed。"""
        detail = dict(details or {})
        detail["error"] = error_msg
        self.workflow_error = step
        self.current_step = "failed"
        self.workflow_completed = False
        self.log(step=step, action=action, status="error", details=detail)
        if error_msg and error_msg not in self.errors:
            self.errors.append(error_msg)

    def mark_all_done(self) -> None:
        """整个流程成功结束时调用：current_step=completed、progress=100、workflow_completed=True。"""
        self.current_step = "completed"
        self.progress_percent = 100
        self.workflow_completed = True
        self.log(
            step="completed",
            action="Agent 工作流执行完毕",
            status="completed",
            details={
                "completed_steps": list(self.completed_steps),
                "skipped_steps": list(self.skipped_steps),
            },
        )

    def sync_progress_to_report(self) -> None:
        """把 AgentState 的结构化进度字段同步到已创建的 state.report（如存在）。

        用于修正 generate_report 返回后、memory 写入、workflow 结束这些时刻的状态快照，
        让前端拿到的 FactCheckReport 自带最新结构化进度（而不是生成时刻的老快照）。
        """
        if self.report is None:
            return
        self.report.current_step = self.current_step
        self.report.completed_steps = list(self.completed_steps)
        self.report.skipped_steps = list(self.skipped_steps)
        self.report.progress_percent = self.progress_percent
        self.report.workflow_completed = self.workflow_completed
        self.report.workflow_error = self.workflow_error
        # 同步 execution_log 增量（生成报告之后追加的 memory/completed 日志等）
        if len(self.execution_log) > len(self.report.execution_log):
            self.report.execution_log = list(self.execution_log)
