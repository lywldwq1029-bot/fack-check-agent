"""报告生成节点。

当前基于规则汇总结果；后续可替换为调用 LLM + REPORT_SYSTEM_PROMPT。
"""

from datetime import datetime

from src.models import AgentState, FactCheckReport, TimelineEvent


def generate_report(state: AgentState) -> AgentState:
    """根据各主张核查结果生成完整报告。

    语义校正：
    - 若 state.workflow_error 存在（LLM、搜索、代码或数据模型异常），
      overall_verdict 强制写为 "核查失败"，不能伪装成 "证据不足"。
    - 单条 ClaimResult.verdict 不允许使用 "核查失败"；ClaimResult 仅表示业务结论。
    """
    state.log(
        step="report",
        action="正在汇总核查结果并生成报告",
        status="running",
        details={"result_count": len(state.claim_results)},
    )

    failed_phase: str | None = getattr(state, "workflow_error", None)
    verdicts = [r.verdict for r in state.claim_results]

    if failed_phase:
        overall = "核查失败"
        summary = _generate_failure_summary(state, failed_phase)
    else:
        overall = _compute_overall_verdict(verdicts)
        summary = _generate_summary(state, overall)

    timeline = _build_timeline(state)
    risk = _assess_propagation_risk(state)
    unresolved = _list_unresolved_questions(state)

    report = FactCheckReport(
        original_text=state.original_text,
        overall_verdict=overall,  # type: ignore[arg-type]
        overall_summary=summary,
        claim_results=list(state.claim_results),
        timeline=timeline,
        propagation_risk=risk,
        unresolved_questions=unresolved,
        execution_log=list(state.execution_log),
        generated_at=datetime.now(),
    )
    # 显式同步 AgentState 结构化进度到 FactCheckReport（失败时尤其重要）
    report.current_step = getattr(state, "current_step", report.current_step)
    report.completed_steps = list(getattr(state, "completed_steps", report.completed_steps))
    report.skipped_steps = list(getattr(state, "skipped_steps", report.skipped_steps))
    report.progress_percent = int(getattr(state, "progress_percent", report.progress_percent) or 0)
    report.workflow_completed = bool(getattr(state, "workflow_completed", report.workflow_completed))
    report.workflow_error = failed_phase
    # 失败态：workflow_completed 不得被 True 伪装
    if failed_phase:
        report.workflow_completed = False

    state.report = report
    state.log(
        step="report",
        action="核查报告已生成",
        status="completed" if not failed_phase else "error",
        details={
            "overall_verdict": overall,
            "claim_count": len(state.claim_results),
            "workflow_error": failed_phase,
        },
    )
    return state


def _compute_overall_verdict(verdicts: list[str]) -> str:
    """根据各主张结论综合判断总体结论。异常态不得进入此函数。"""
    if not verdicts:
        return "证据不足"

    # 优先级：存在误导 > 已证伪 > 仍在发展 > 证据不足 > 部分属实 > 基本属实 > 已证实
    priority = ["存在误导", "已证伪", "仍在发展", "证据不足", "部分属实", "基本属实", "已证实"]
    for candidate in priority:
        if candidate in verdicts:
            return candidate
    return "证据不足"


def _generate_failure_summary(state: AgentState, failed_phase: str) -> str:
    """失败态摘要：明确写"核查失败"，不混淆"证据不足"。"""
    from src.workflow import (  # 局部导入防循环
        _PHASE_DECOMPOSE,
        _PHASE_EVALUATE,
        _PHASE_PLAN,
        _PHASE_REPORT,
        _PHASE_SEARCH,
        _PHASE_SUFFICIENCY,
    )

    phase_labels = {
        "init": "初始化",
        _PHASE_DECOMPOSE: "主张拆解",
        _PHASE_PLAN: "核查计划",
        _PHASE_SEARCH: "网页搜索与证据提取",
        _PHASE_SUFFICIENCY: "证据充分性评估与补充检索",
        _PHASE_EVALUATE: "来源分级与交叉验证",
        _PHASE_REPORT: "汇总生成报告",
        "memory": "保存核查记忆",
    }
    phase_cn = phase_labels.get(failed_phase, failed_phase)
    errors = list(getattr(state, "errors", None) or [])
    err_tail = ""
    if errors:
        # 只取最后一条简洁错误，避免把整段栈追踪塞进报告摘要
        err_tail = f"最近一条错误：{str(errors[-1])[:160]}"
    return (
        f"事实核查未完成：Agent 在阶段「{phase_cn}」（{failed_phase}）执行异常，"
        "已提前终止后续步骤。此状态为「系统执行失败」而非「核查成功但证据不足」，"
        "请检查日志或稍后重试。" + (f" {err_tail}" if err_tail else "")
    )


def _generate_summary(state: AgentState, overall: str) -> str:
    """生成总体摘要（非失败态）。"""
    total = len(state.claim_results)
    if total == 0:
        if state.errors:
            return f"主张拆解或核查过程中出现错误，未能生成可核查主张。错误信息：{'；'.join(state.errors)}"
        return "文本主要为观点或信息不足，未能拆解出可核查主张。"

    verified = sum(1 for r in state.claim_results if r.verdict in ("已证实", "基本属实"))
    partial = sum(1 for r in state.claim_results if r.verdict == "部分属实")
    misleading = sum(1 for r in state.claim_results if r.verdict in ("存在误导", "已证伪"))
    insufficient = sum(1 for r in state.claim_results if r.verdict in ("证据不足", "仍在发展"))

    return (
        f"本次核查共拆解 {total} 个主张："
        f"{verified} 个基本属实，{partial} 个部分属实，"
        f"{misleading} 个存在误导或已证伪，{insufficient} 个证据不足或仍在发展。"
        f"总体判断为「{overall}」。"
    )


def _build_timeline(state: AgentState) -> list[TimelineEvent]:
    """根据证据和主张构建事件时间线。"""
    events: list[TimelineEvent] = []
    now = datetime.now()

    # 模拟时间线：基于原始文本主题生成关键节点
    if any("暴雨" in r.claim.text for r in state.claim_results):
        events.append(
            TimelineEvent(
                event_time=now,
                description="暴雨天气影响某市，社交媒体开始流传停运、失联、停课等信息",
                source_url="https://example.com/demo/social",
            )
        )
        events.append(
            TimelineEvent(
                event_time=now,
                description="市轨道交通集团通报部分线路临时停运，并非全线停运",
                source_url="https://example.com/demo/metro-partial",
            )
        )
        events.append(
            TimelineEvent(
                event_time=now,
                description="市教育局澄清未发布全市停课三天通知",
                source_url="https://example.com/demo/education",
            )
        )

    return events


def _assess_propagation_risk(state: AgentState) -> str:
    """评估传播风险。"""
    misleading_count = sum(1 for r in state.claim_results if r.verdict in ("存在误导", "已证伪"))
    insufficient_count = sum(1 for r in state.claim_results if r.verdict in ("证据不足", "仍在发展"))

    if misleading_count >= 2 or (misleading_count >= 1 and insufficient_count >= 1):
        return "高：存在明显夸大或失实信息，且部分关键信息尚未核实，易引发恐慌传播"
    if misleading_count == 1 or insufficient_count >= 2:
        return "中：部分内容存在误导或证据不足，需及时澄清"
    return "低：主要信息已有较可靠来源支撑，传播风险可控"


def _list_unresolved_questions(state: AgentState) -> list[str]:
    """列出仍待核实的问题。"""
    questions: list[str] = []
    for result in state.claim_results:
        if result.missing_information:
            questions.append(f"[{result.claim.text}] {result.missing_information}")
    if not questions:
        questions.append("暂无待核实问题")
    return questions
