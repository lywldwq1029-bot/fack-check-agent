"""数据模型基础测试。"""

from datetime import datetime

from src.models import Claim, Evidence, ClaimResult, TimelineEvent, FactCheckReport, AgentState, VerificationPlan


def test_claim_creation():
    claim = Claim(
        claim_id="c1",
        text="某市因暴雨导致地铁全线停运",
        claim_type="事件陈述",
        entities=["某市", "地铁"],
        time_reference="当前",
        location="某市",
        verification_priority=1,
    )
    assert claim.claim_id == "c1"
    assert "地铁" in claim.text


def test_evidence_creation():
    evidence = Evidence(
        evidence_id="e1",
        claim_id="c1",
        source_title="市气象局暴雨橙色预警",
        source_url="https://example.com/demo",
        publisher="市气象局",
        published_at=datetime.now(),
        retrieved_at=datetime.now(),
        evidence_summary="部分线路因积水临时停运",
        source_type="官方通报",
        source_grade="A",
        supports_or_refutes="partial",
        is_primary_source=True,
        reliability_reason="官方渠道发布",
    )
    assert evidence.source_grade == "A"


def test_claim_result_verdict_constraint():
    result = ClaimResult(
        claim=Claim(
            claim_id="c1",
            text="某市因暴雨导致地铁全线停运",
            claim_type="事件陈述",
            entities=["某市", "地铁"],
            time_reference="当前",
            location="某市",
            verification_priority=1,
        ),
        verdict="部分属实",
        confidence=0.6,
        reasoning="仅部分线路停运，非全线",
        evidence=[],
        missing_information="具体停运线路清单",
    )
    assert result.verdict == "部分属实"


def test_fact_check_report_creation():
    report = FactCheckReport(
        original_text="测试新闻文本",
        overall_verdict="部分属实",
        overall_summary="演示报告",
        claim_results=[],
        timeline=[],
        propagation_risk="中",
        unresolved_questions=["后续进展"],
        generated_at=datetime.now(),
    )
    assert report.overall_verdict == "部分属实"


def test_agent_state_creation():
    state = AgentState(
        original_text="测试输入",
        claims=[],
        verification_plan=[],
        evidence={},
        claim_results=[],
        report=None,
        current_step="init",
        execution_log=[],
        errors=[],
    )
    assert state.current_step == "init"
    assert state.errors == []


def test_claim_new_fields():
    """测试本阶段新增的 Claim 字段。"""
    claim = Claim(
        claim_id="c1",
        text="鹿晗和关晓彤已经分手",
        claim_type="事件陈述",
        entities=["鹿晗", "关晓彤"],
        time_reference="近期",
        location=None,
        verification_priority=1,
        verification_question="鹿晗和关晓彤是否已正式宣布分手？",
        search_keywords=["鹿晗 关晓彤 分手", "鹿晗 关晓彤 官宣"],
        preferred_source_types=["当事人社交媒体", "权威娱乐媒体"],
        risk_level="medium",
        sensitive_reason="涉及个人感情生活",
        is_opinion=False,
        is_checkable=True,
    )
    assert claim.risk_level == "medium"
    assert claim.is_checkable is True
    assert len(claim.search_keywords) == 2


def test_verification_plan_creation():
    """测试 VerificationPlan 模型。"""
    plan = VerificationPlan(
        claim_id="c1",
        verification_steps=["查找当事人社交媒体声明", "查找权威媒体报道"],
        search_queries=["鹿晗 关晓彤 分手 官宣", "鹿晗 关晓彤 感情状态"],
        preferred_sources=["当事人社交媒体", "权威娱乐媒体"],
        required_evidence_level="极高强度证据",
        priority=1,
        priority_reason="涉及个人名誉，需极高强度证据",
    )
    assert plan.required_evidence_level == "极高强度证据"
    assert plan.priority == 1


def test_claim_high_risk_field():
    """测试高风险主张字段。"""
    claim = Claim(
        claim_id="c2",
        text="分手原因是鹿晗男女关系混乱",
        claim_type="归因判断",
        entities=["鹿晗"],
        verification_priority=1,
        risk_level="high",
        sensitive_reason="涉及个人名誉的负面指控",
    )
    assert claim.risk_level == "high"


# ============ FactCheckReport.execution_log 字段测试 ============


def test_fact_check_report_execution_log_default():
    """FactCheckReport 缺少 execution_log 参数时应使用默认空列表。"""
    report = FactCheckReport(
        original_text="测试新闻文本",
        overall_verdict="部分属实",
        overall_summary="演示报告",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
    )
    assert report.execution_log == []
    assert isinstance(report.execution_log, list)


def test_fact_check_report_execution_log_serialization_roundtrip():
    """execution_log 应能正常通过 Pydantic 序列化和反序列化。"""
    log_entries = [
        {
            "timestamp": "2026-08-02T10:00:00",
            "step": "decompose",
            "action": "拆解主张完成",
            "status": "completed",
            "details": {"claim_count": 2},
        },
        {
            "timestamp": "2026-08-02T10:00:05",
            "step": "search",
            "action": "搜索证据完成",
            "status": "completed",
            "details": {"total_results": 8},
        },
    ]
    report = FactCheckReport(
        original_text="测试新闻文本",
        overall_verdict="证据不足",
        overall_summary="测试摘要",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
        execution_log=list(log_entries),
    )

    # JSON 序列化与反序列化往返
    json_str = report.model_dump_json()
    restored = FactCheckReport.model_validate_json(json_str)

    assert len(restored.execution_log) == 2
    assert restored.execution_log[0]["step"] == "decompose"
    assert restored.execution_log[1]["action"] == "搜索证据完成"
    assert restored.execution_log[1]["details"]["total_results"] == 8


def test_fact_check_report_execution_log_model_dump():
    """execution_log 应出现在 model_dump() 输出中。"""
    log_entry = {
        "timestamp": "2026-08-02T10:00:00",
        "step": "plan",
        "action": "生成计划",
        "status": "completed",
        "details": {},
    }
    report = FactCheckReport(
        original_text="测试",
        overall_verdict="证据不足",
        overall_summary="测试",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
        execution_log=[log_entry],
    )
    data = report.model_dump()
    assert "execution_log" in data
    assert data["execution_log"][0]["step"] == "plan"


def test_fact_check_report_execution_log_not_shared_between_instances():
    """不同 FactCheckReport 实例的 execution_log 应独立，不共享默认列表。"""
    report_a = FactCheckReport(
        original_text="报告 A",
        overall_verdict="证据不足",
        overall_summary="摘要 A",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
    )
    report_b = FactCheckReport(
        original_text="报告 B",
        overall_verdict="证据不足",
        overall_summary="摘要 B",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
    )

    # 向 A 的日志中追加
    report_a.execution_log.append({"step": "decompose", "status": "completed"})

    # B 的日志应仍为空
    assert len(report_a.execution_log) == 1
    assert report_b.execution_log == []
    assert report_a.execution_log is not report_b.execution_log


def test_report_execution_log_isolates_from_state():
    """generate_report 传入 state.execution_log 时应复制，后续修改 state 不影响已创建的 report。"""
    from src.models import AgentState
    from src.nodes.report import generate_report

    state = AgentState(original_text="测试新闻", mode="demo")
    state.log(step="decompose", action="拆解", status="completed")
    state = generate_report(state)

    # 向 state 日志追加新条目
    state.log(step="memory", action="保存", status="completed")

    assert state.report is not None
    # report 保存的日志应不包含后追加的 memory 步骤
    steps_in_report = [e["step"] for e in state.report.execution_log]
    assert "decompose" in steps_in_report
    # report 是在 generate_report 内部创建的，最后的 memory 步骤写入的是 state，不应出现在 report 中
    assert "memory" not in steps_in_report


def test_workflow_mock_full_mode_no_execution_log_error():
    """完整真实模式（mock 工作流）不应再触发 'FactCheckReport has no field execution_log' 错误。

    本测试不调用真实 LLM 和 Tavily，使用演示模式验证工作流输出的 report 中 execution_log 存在且可访问。
    """
    from src.workflow import run_fact_check_workflow

    text = "网传某市因暴雨导致地铁全线停运。"
    report = run_fact_check_workflow(text, mode="demo")

    # Pydantic 字段应可访问，赋值不应报错
    assert hasattr(report, "execution_log")
    assert isinstance(report.execution_log, list)
    assert len(report.execution_log) >= 2  # 至少包含拆解和报告生成步骤

    # 执行状态卡片所需的 step 字段应存在
    steps = {entry.get("step", "") for entry in report.execution_log}
    assert "decompose" in steps
    assert "report" in steps


def test_old_report_without_execution_log_field_still_readable():
    """旧版报告（JSON 中无 execution_log 字段）通过 model_validate 构造时应成功读取，
    并使用 default_factory 返回默认空列表。
    """
    old_report_dict = {
        "original_text": "旧版测试新闻",
        "overall_verdict": "基本属实",
        "overall_summary": "旧版摘要",
        "claim_results": [],
        "timeline": [],
        "propagation_risk": "低",
        "unresolved_questions": [],
        # 故意不包含 execution_log 字段
        "generated_at": "2026-07-01T12:00:00",
    }
    report = FactCheckReport.model_validate(old_report_dict)
    assert report.execution_log == []
    assert report.original_text == "旧版测试新闻"


def test_no_direct_assignment_on_missing_fields():
    """确认 report 对象不再依赖 report.xxx = [] 式的即时字段注入。"""
    import json

    report = FactCheckReport(
        original_text="测试新闻文本",
        overall_verdict="部分属实",
        overall_summary="演示报告",
        claim_results=[],
        timeline=[],
        propagation_risk="中",
        unresolved_questions=["后续进展"],
    )

    # 读取所有 FactCheckReport 显式声明的字段均不应报错
    fields = {
        "original_text": report.original_text,
        "overall_verdict": report.overall_verdict,
        "overall_summary": report.overall_summary,
        "claim_results": report.claim_results,
        "timeline": report.timeline,
        "propagation_risk": report.propagation_risk,
        "unresolved_questions": report.unresolved_questions,
        "execution_log": report.execution_log,
        "generated_at": report.generated_at,
    }
    # 全部字段均可被访问并被序列化为 JSON
    assert json.dumps({k: str(v) for k, v in fields.items()}, ensure_ascii=False)
