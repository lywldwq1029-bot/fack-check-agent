"""溯真 Agent 工作流编排。

流程：新闻输入 → 主张拆解 → 生成核查计划 → 搜索证据 → 评价来源 → 交叉验证 →
     生成结论 → 保存核查记忆 → 输出报告

支持三种运行模式：
- demo：全流程模拟，不需要 API 密钥
- llm：主张拆解与计划生成调用真实大模型，证据检索仍为模拟
- full：完整真实核查，Tavily 搜索 + LLM 证据提取与交叉验证

全局超时控制：WORKFLOW_MAX_SECONDS（默认120s），超时后进入降级逻辑，
保留已有证据生成"部分完成"报告。
"""

from __future__ import annotations
import time
from datetime import datetime
from typing import Optional

from src.config import settings

# ===== 全局活跃核查截止时间（跨模块共享，用于 LLM/Tavily 内部超时检查）=====
_ACTIVE_DEADLINE: Optional[float] = None


def set_active_deadline(deadline: float) -> None:
    """设置当前活跃核查的截止时间。"""
    global _ACTIVE_DEADLINE
    _ACTIVE_DEADLINE = deadline


def clear_active_deadline() -> None:
    """清除当前活跃核查的截止时间。"""
    global _ACTIVE_DEADLINE
    _ACTIVE_DEADLINE = None


def check_active_deadline() -> None:
    """检查当前活跃核查是否已超时。超时则抛出 _WorkflowTimeout。

    可在任何模块（LLM client、Tavily search）中调用，确保 API 调用
    不会在工作流已超时后继续阻塞。
    """
    if _ACTIVE_DEADLINE is not None and time.time() > _ACTIVE_DEADLINE:
        raise _WorkflowTimeout(
            f"工作流超时：已超过全局 {settings.WORKFLOW_MAX_SECONDS}s 限制，停止后续请求"
        )


from src.memory.repository import MemoryRepository
from src.models import AgentState, FactCheckReport
from src.nodes.decompose import decompose_claims
from src.nodes.evaluate import evaluate_claims
from src.nodes.plan import plan_verification
from src.nodes.report import generate_report
from src.nodes.search import search_evidence
from src.nodes.sufficiency import assess_sufficiency, MAX_ROUNDS_PER_CLAIM


_PHASE_DECOMPOSE = "decompose"
_PHASE_PLAN = "plan"
_PHASE_SEARCH = "search"
_PHASE_SUFFICIENCY = "sufficiency"
_PHASE_EVALUATE = "evaluate"
_PHASE_REPORT = "report"
_PHASE_MEMORY = "memory"


class _WorkflowTimeout(Exception):
    """工作流整体超时异常。"""
    pass


def _check_deadline(deadline: float, state: AgentState, phase: str) -> None:
    """检查是否超过全局截止时间。超过则抛出 _WorkflowTimeout。"""
    if time.time() > deadline:
        elapsed = time.time() - (deadline - settings.WORKFLOW_MAX_SECONDS)
        raise _WorkflowTimeout(
            f"工作流超时：已运行 {elapsed:.0f}s，超过 {settings.WORKFLOW_MAX_SECONDS}s 限制"
        )


def _safe_execute_phase(
    state: AgentState,
    phase_name: str,
    phase_label: str,
    deadline: float,
    func,
    *args,
    **kwargs,
) -> AgentState:
    """安全执行单个阶段，带超时检查和异常捕获。"""
    _check_deadline(deadline, state, phase_name)

    state.mark_step_started(phase_name, phase_label)
    phase_start = time.time()

    try:
        state = func(state, *args, **kwargs)
    except _WorkflowTimeout:
        phase_elapsed = time.time() - phase_start
        state.errors.append(
            f"工作流超时：阶段「{phase_label}」运行 {phase_elapsed:.1f}s 后整体超时"
        )
        raise
    except Exception as e:
        phase_elapsed = time.time() - phase_start
        err_msg = str(e)
        is_timeout = "超时" in err_msg or "timeout" in err_msg.lower()

        state.mark_failed(
            phase_name,
            f"{phase_label}发生异常",
            error_msg=f"{phase_label}异常：{err_msg[:80]}",
            details={"phase_elapsed_s": round(phase_elapsed, 1)},
        )
        if is_timeout:
            state.errors.append(f"{phase_label}超时：已使用现有证据生成部分报告")
            state.workflow_error = "timeout"
        else:
            state.errors.append(f"{phase_label}异常：{err_msg[:100]}")
            state.workflow_error = state.workflow_error or "error"
        state = generate_report(state)
        state.metadata["_early_return"] = True
        return state

    phase_elapsed = time.time() - phase_start
    if state.workflow_error is None:
        state.mark_step_completed(
            phase_name,
            f"{phase_label}完成",
            details={"phase_elapsed_s": round(phase_elapsed, 1)},
        )

    _check_deadline(deadline, state, phase_name)
    return state


def _early_return(state: AgentState, timeout_occurred: bool = False) -> FactCheckReport:
    """早期出口公共收尾。"""
    clear_active_deadline()

    if _PHASE_REPORT not in state.completed_steps:
        state.mark_step_completed(_PHASE_REPORT, "报告生成完成")

    _save_memory(state, save_to_memory=True)

    if timeout_occurred:
        state.workflow_completed = False
        state.current_step = "failed"
        if not state.workflow_error:
            state.workflow_error = "timeout"
        # 保留实际进度，不标记100%
        state.sync_progress_to_report()
    else:
        state.mark_all_done()
        state.sync_progress_to_report()

    if state.report is None:
        raise RuntimeError("工作流未生成报告")
    return state.report


def run_fact_check_workflow(
    original_text: str,
    save_to_memory: bool = True,
    mode: str = "demo",
) -> FactCheckReport:
    """运行完整的事实核查工作流。"""
    workflow_start = time.time()
    deadline = workflow_start + settings.WORKFLOW_MAX_SECONDS
    set_active_deadline(deadline)

    # 完整真实模式预检
    if mode == "full":
        missing = settings.missing_configs()
        if missing:
            state = AgentState(original_text=original_text, mode=mode)
            state.mark_step_started("init", "开始事实核查工作流")
            state.mark_step_completed("init", "初始化完成")
            state.mark_step_started(_PHASE_DECOMPOSE, "准备拆解主张")
            state.mark_failed(
                _PHASE_DECOMPOSE,
                "完整真实核查模式缺少配置",
                error_msg=f"完整真实核查模式缺少以下配置：{', '.join(missing)}。",
            )
            state = generate_report(state)
            return _early_return(state)

    state = AgentState(original_text=original_text, mode=mode)
    state.mark_step_started("init", f"开始事实核查工作流（{mode}模式）")
    state.mark_step_completed(
        "init",
        "初始化完成",
        details={"text_length": len(original_text), "mode": mode},
    )

    # 限制主张数量
    max_claims = settings.MAX_CLAIMS
    state.metadata["max_claims"] = max_claims
    state.metadata["_workflow_deadline"] = deadline
    state.metadata["_workflow_start"] = workflow_start

    try:
        # 1. 主张拆解
        state = _safe_execute_phase(
            state, _PHASE_DECOMPOSE, "正在拆解新闻文本为可核查主张",
            deadline, decompose_claims,
        )
        if state.metadata.get("_early_return"):
            return _early_return(state, timeout_occurred=(state.workflow_error == "timeout"))

        if state.errors and not state.claims:
            state = generate_report(state)
            return _early_return(state)

        # 限制主张数量
        if len(state.claims) > max_claims:
            state.claims = state.claims[:max_claims]
            state.log(
                step="decompose",
                action=f"主张数量已限制为 {max_claims} 条",
                status="running",
                details={"original_count": len(state.claims) + (len(state.claims) - max_claims)},
            )

        # 2. 生成核查计划
        state = _safe_execute_phase(
            state, _PHASE_PLAN, "正在为每个主张制定核查计划",
            deadline, plan_verification,
        )
        if state.metadata.get("_early_return"):
            return _early_return(state, timeout_occurred=(state.workflow_error == "timeout"))

        # 3. 第一轮证据搜索
        state = _safe_execute_phase(
            state, _PHASE_SEARCH, "正在进行第1轮网页搜索与证据提取",
            deadline, search_evidence, round_label=1,
        )
        if state.metadata.get("_early_return"):
            return _early_return(state, timeout_occurred=(state.workflow_error == "timeout"))

        # 如果搜索已超时，跳过后续阶段，直接生成部分报告
        if state.metadata.get("_search_timed_out"):
            state.errors.append("搜索阶段超时：已使用现有证据生成部分报告")
            state.workflow_error = "timeout"
            state.mark_failed(
                _PHASE_SEARCH,
                "搜索超时",
                error_msg="搜索请求超时，已使用现有证据生成部分报告",
            )
            state = generate_report(state)
            return _early_return(state, timeout_occurred=True)

        # 4. 证据充分性评估（含第二轮补充检索）
        state.mark_step_started(_PHASE_SUFFICIENCY, "正在评估证据充分性")
        suf_start = time.time()
        try:
            state = assess_sufficiency(state)
            follow_up = state.metadata.get("follow_up") or {}
            if follow_up:
                _check_deadline(deadline, state, _PHASE_SUFFICIENCY)
                override = {cid: list(info.get("queries") or []) for cid, info in follow_up.items()}
                state = _safe_execute_phase(
                    state, _PHASE_SEARCH, "正在进行第2轮补充检索",
                    deadline, search_evidence,
                    claim_queries_override=override, round_label=2,
                )
                if state.metadata.get("_early_return"):
                    return _early_return(state, timeout_occurred=(state.workflow_error == "timeout"))

                rounds = state.metadata.get("_search_rounds") or {}
                for cid in override:
                    rounds[cid] = MAX_ROUNDS_PER_CLAIM
                state.metadata["_search_rounds"] = rounds
                state = assess_sufficiency(state)

            suf_elapsed = time.time() - suf_start
            if state.workflow_error is None:
                state.mark_step_completed(
                    _PHASE_SUFFICIENCY,
                    "证据充分性评估完成",
                    details={"phase_elapsed_s": round(suf_elapsed, 1)},
                )
        except _WorkflowTimeout:
            raise
        except Exception as e:
            state.errors.append(f"sufficiency异常：{e}")
            suf_elapsed = time.time() - suf_start
            state.mark_step_completed(
                _PHASE_SUFFICIENCY,
                "证据充分性评估完成（有警告）",
                details={"phase_elapsed_s": round(suf_elapsed, 1), "errors": [str(e)[:60]]},
            )

        # 5. 评价来源与交叉验证
        state = _safe_execute_phase(
            state, _PHASE_EVALUATE, "正在进行来源分级与交叉验证",
            deadline, evaluate_claims,
        )
        if state.metadata.get("_early_return"):
            return _early_return(state, timeout_occurred=(state.workflow_error == "timeout"))

        # 6. 生成报告
        state = _safe_execute_phase(
            state, _PHASE_REPORT, "正在汇总核查结果并生成报告",
            deadline, generate_report,
        )

        # 7. 保存核查记忆
        _save_memory(state, save_to_memory)

        # 8. 收尾
        state.mark_all_done()
        state.sync_progress_to_report()

    except _WorkflowTimeout:
        state.errors.append("核查未完全完成：整体超时，以下为部分结果")
        state.log(
            step="timeout",
            action="工作流超时，降级生成部分报告",
            status="error",
            details={
                "elapsed_s": round(time.time() - workflow_start, 1),
                "max_seconds": settings.WORKFLOW_MAX_SECONDS,
            },
        )
        try:
            state = generate_report(state)
            report = state.report
            if report:
                report.workflow_completed = False
                report.workflow_error = report.workflow_error or "timeout"
                report.current_step = "failed"
                # 在摘要中说明超时
                if report.overall_summary:
                    report.overall_summary = (
                        f"核查未完全完成：部分步骤因超时未执行。以下为已取得的证据，不代表最终结论。\n\n"
                        f"（已运行 {time.time() - workflow_start:.0f}s，上限 {settings.WORKFLOW_MAX_SECONDS}s）\n\n"
                        f"原始摘要：{report.overall_summary}"
                    )
                else:
                    report.overall_summary = (
                        f"核查未完全完成：工作流在 {time.time() - workflow_start:.0f}s 时超时"
                        f"（上限 {settings.WORKFLOW_MAX_SECONDS}s）。以下为当前已取得的部分证据。"
                    )
        except Exception:
            pass
        return _early_return(state, timeout_occurred=True)

    if state.report is None:
        raise RuntimeError("工作流未生成报告")
    return state.report


def _save_memory(state: AgentState, save_to_memory: bool) -> None:
    """保存报告到记忆库。"""
    if not save_to_memory:
        state.mark_step_skipped(
            _PHASE_MEMORY,
            "记忆功能未启用",
            details={"reason": "save_to_memory=False"},
        )
        return

    if not state.report:
        state.mark_step_skipped(
            _PHASE_MEMORY,
            "无报告可保存",
        )
        return

    try:
        state.mark_step_started(_PHASE_MEMORY, "正在保存核查记忆")
        repo = MemoryRepository()
        report_id = repo.save_report(state.report)
        state.mark_step_completed(
            _PHASE_MEMORY,
            "核查记忆保存完成",
            details={"report_id": report_id},
        )
    except Exception as e:
        state.mark_step_skipped(
            _PHASE_MEMORY,
            "保存核查记忆失败，已跳过",
            details={"error": str(e)},
        )
        state.errors.append(f"保存记忆库失败: {e}")
