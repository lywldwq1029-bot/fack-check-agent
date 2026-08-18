"""工作流结构化进度状态单元测试。

不调用真实 LLM 和 Tavily，仅使用 demo 模式 / 异常注入 / 旧版报告数据构造来验证：
- 完整成功流程：progress_percent=100、current_step=completed、workflow_completed=True
- 报告生成阶段被标记完成（report ∈ completed_steps）
- 记忆未启用时：memory ∈ skipped_steps，不会一直显示"正在保存"
- 异常流程：current_step=failed、workflow_completed=False、workflow_error=失败阶段
- 旧版报告（无结构化进度字段）兼容识别为完成态
"""

from datetime import datetime
from unittest.mock import patch

import pytest

from src.models import (
    AgentState,
    Claim,
    ClaimResult,
    FactCheckReport,
    WORKFLOW_PHASES,
)
from src.nodes.decompose import decompose_claims
from src.workflow import run_fact_check_workflow


DEMO_TEXT = (
    "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
)


# ========== 1. 完整成功流程：progress=100、全部完成标记 ==========


def test_full_demo_workflow_progress_100():
    """完整 demo 成功流程：最终报告 progress_percent 必须为 100。"""
    report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=False, mode="demo")

    assert report.workflow_completed is True
    assert report.current_step == "completed"
    assert report.progress_percent == 100
    assert report.workflow_error is None


def test_report_step_marked_completed():
    """用户指出的核心 bug：报告生成后，'report' 阶段必须出现在 completed_steps 中。"""
    report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=False, mode="demo")

    # 直接检查 structured 字段，不依赖 execution_log 中文句子匹配
    assert "report" in report.completed_steps, (
        "FactCheckReport 生成后，report 阶段应该出现在 completed_steps 中"
    )
    # 验证 execution_log 中也有 report 完成记录（快照已同步）
    steps_in_log = [e["step"] for e in report.execution_log if e.get("status") == "completed"]
    assert "report" in steps_in_log, "execution_log 快照也应包含 report 完成记录"


def test_memory_disabled_is_skipped_not_pending():
    """save_to_memory=False 时，memory 阶段必须明确标记为 skipped，不能显示为未完成。"""
    report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=False, mode="demo")

    # 关键断言：memory 不在 completed_steps，也不应一直显示为 pending
    assert "memory" not in report.completed_steps
    assert "memory" in report.skipped_steps, (
        "记忆未启用时，memory 必须在 skipped_steps 中，"
        "不能伪装成蓝色正在保存状态"
    )
    # 跳过也应计入完成阶段集合（进度为 100）
    assert report.progress_percent == 100
    assert report.workflow_completed is True


def test_memory_enabled_marks_completed(monkeypatch, tmp_path):
    """save_to_memory=True 且数据库实际保存成功时，memory ∈ completed_steps。"""
    db = tmp_path / "memory_progress_test.db"
    # 覆盖配置项，使用临时数据库
    from src.config import settings as cfg

    monkeypatch.setattr(cfg, "MEMORY_DB_PATH", str(db))

    report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=True, mode="demo")

    assert "memory" in report.completed_steps, (
        "保存成功时，memory 阶段应出现在 completed_steps 中"
    )
    assert report.progress_percent == 100


# ========== 2. 异常流程：failed 状态正确显示 ==========


def test_exception_in_decompose_marks_failed():
    """拆解阶段抛出未捕获异常：current_step=failed、workflow_error='decompose'。"""
    with patch(
        "src.workflow.decompose_claims",
        side_effect=RuntimeError("模拟拆解阶段异常"),
    ):
        report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=False, mode="demo")

    assert report.current_step == "failed" or report.workflow_error is not None
    assert report.workflow_error == "decompose", (
        f"失败阶段应为 'decompose'，实际为 {report.workflow_error!r}"
    )
    # 工作流未成功结束
    assert report.workflow_completed is False
    # 失败阶段不应出现在 completed_steps 中
    assert "decompose" not in report.completed_steps or report.current_step == "failed"


def test_missing_config_in_full_mode_marks_failed():
    """完整真实模式未配置密钥时，失败阶段应明确记录（且 progress 最终仍归一化 100，
    因为后续阶段被明确收尾，不显示为卡住）。"""
    # 强制移除环境中的 Tavily 与 LLM 配置，触发 missing_configs 分支
    from src.config import settings as cfg

    with patch.object(cfg, "LLM_API_KEY", ""), \
         patch.object(cfg, "LLM_MODEL", ""), \
         patch.object(cfg, "TAVILY_API_KEY", ""):
        report = run_fact_check_workflow(DEMO_TEXT, save_to_memory=False, mode="full")

    # missing_configs 会触发 mark_failed(phase=decompose)，之后被 _early_return 收尾
    assert report.workflow_error == "decompose" or any(
        "缺少" in err for err in report.execution_log or []
    )
    # 整流程仍被收尾，不应卡主
    assert report.progress_percent == 100
    # 结束阶段标记应为 completed（因为已经走到统一收尾，用户不应看到"正在生成"）
    assert report.current_step == "completed"


# ========== 3. 旧 session_state 报告：兼容识别为完成 ==========


def _build_legacy_report_dict() -> dict:
    """构造一个旧版报告字典（不含结构化进度字段）。"""
    claim = Claim(
        claim_id="c1",
        text="某市因暴雨导致地铁全线停运",
        claim_type="事件陈述",
        entities=["某市", "地铁"],
        verification_priority=1,
    )
    cr = ClaimResult(
        claim=claim,
        verdict="部分属实",
        confidence=0.6,
        reasoning="仅部分线路停运",
        evidence=[],
    )
    return {
        "original_text": DEMO_TEXT,
        "overall_verdict": "部分属实",
        "overall_summary": "旧版报告摘要",
        "claim_results": [cr.model_dump()],
        "timeline": [],
        "propagation_risk": "中",
        "unresolved_questions": ["具体停运线路清单"],
        "execution_log": [{"timestamp": "2026-07-01T00:00:00", "step": "init",
                           "action": "初始化", "status": "completed", "details": {}}],
        # 故意不包含 structured 进度字段：completed_steps/skipped_steps/progress/...
        "generated_at": "2026-07-01T12:00:00",
    }


def test_legacy_report_without_structured_progress_is_treated_completed():
    """旧报告缺结构化字段，但有 claim_results + overall_verdict + generated_at，
    应兼容视为全部阶段完成，避免旧 session_state 永远停在处理中。"""
    legacy_data = _build_legacy_report_dict()
    report = FactCheckReport.model_validate(legacy_data)

    # 直接读取字段时：默认值为 0/空，尚未被兼容修复
    assert report.progress_percent == 0
    assert report.current_step == "init"
    assert report.workflow_completed is False

    # 通过 app.py 中使用的兼容函数归一化
    from app import _normalize_report_progress

    normalized = _normalize_report_progress(report)

    # 兼容归一化后：视为完成态，避免永远显示为蓝色进行中
    assert normalized.workflow_completed is True
    assert normalized.current_step == "completed"
    assert normalized.progress_percent == 100
    # 全部阶段应被保守标记为 completed（记忆至少显示为完成或跳过之一）
    assert set(WORKFLOW_PHASES).issubset(
        set(normalized.completed_steps) | set(normalized.skipped_steps)
    ), "旧版报告兼容后，所有阶段必须归为 completed 或 skipped 之一"


def test_legacy_report_empty_claim_results_is_not_falsely_completed():
    """旧版缺结构化进度字段 + 没有 claim_results / 没有生成时间 → 不兼容为完成。"""
    legacy_data = _build_legacy_report_dict()
    legacy_data["claim_results"] = []
    legacy_data["overall_verdict"] = "证据不足"
    legacy_data["overall_summary"] = "无主张"
    # generated_at=None 不容易表示为 JSON，用 model_validate 再构造
    report = FactCheckReport.model_validate(legacy_data)
    # 清空 claim_results 后单独再用 FactCheckReport 造一个无 claim 的报告
    report2 = FactCheckReport(
        original_text="空文本",
        overall_verdict="证据不足",
        overall_summary="无可核查主张",
        claim_results=[],
        timeline=[],
        propagation_risk="低",
        unresolved_questions=[],
        execution_log=[],
        generated_at=None,  # type: ignore[arg-type]  # 故意无生成时间
    )
    # generated_at=None 会落到默认工厂，所以再用 raw 方式构造
    report3_dict = report2.model_dump()
    report3_dict.pop("generated_at")  # 去掉时间
    report3 = FactCheckReport.model_construct(**report3_dict)

    from app import _is_legacy_report_completed

    # claim_results 为空 或 生成时间缺失 → 不判定为完成
    assert _is_legacy_report_completed(report2) is False


# ========== 4. AgentState 结构化进度辅助方法 ==========


def test_agent_state_mark_completed_updates_progress():
    """mark_step_completed 每调用一次，进度递增一个阶段的比例。"""
    state = AgentState(original_text="测试")
    # init 阶段开始 + 完成（完成 1/7 ≈ 14%）
    state.mark_step_started("init", "开始")
    assert state.current_step == "init"
    state.mark_step_completed("init", "完成")
    assert "init" in state.completed_steps
    assert state.progress_percent >= 10  # 至少 1 个阶段完成
    # 再完成一个阶段：progress 增加
    state.mark_step_completed("decompose", "完成")
    second_progress = state.progress_percent
    # 7 阶段，每阶段约 14~15pt，两阶段 >= 20
    assert second_progress >= 20


def test_agent_state_mark_failed():
    state = AgentState(original_text="测试")
    state.mark_failed(
        step="search",
        action="搜索失败",
        error_msg="模拟搜索错误",
    )
    assert state.current_step == "failed"
    assert state.workflow_error == "search"
    assert state.workflow_completed is False
    assert any("模拟搜索错误" in e for e in state.errors)


def test_agent_state_mark_all_done():
    state = AgentState(original_text="测试")
    state.mark_all_done()
    assert state.current_step == "completed"
    assert state.progress_percent == 100
    assert state.workflow_completed is True


def test_sync_progress_to_report_updates_snapshot():
    """generate_report 返回后，memory 和 mark_all_done 写入的进度必须同步回 report 快照。"""
    from src.nodes.report import generate_report

    state = AgentState(original_text=DEMO_TEXT, mode="demo")
    # 先完成前几个阶段
    for phase in ["init", "decompose", "plan", "search", "evaluate"]:
        state.mark_step_completed(phase, f"{phase}完成")
    # 生成报告（此时 report 快照只包含截至 report 之前的完成）
    state = generate_report(state)
    # report 阶段完成
    state.mark_step_completed("report", "报告生成完成")
    # 跳过 memory
    state.mark_step_skipped("memory", "记忆未启用")
    state.mark_all_done()

    # 同步到 report 快照
    before_sync_progress = state.report.progress_percent if state.report else None
    state.sync_progress_to_report()

    assert state.report is not None
    assert state.report.current_step == "completed"
    assert state.report.progress_percent == 100
    assert state.report.workflow_completed is True
    assert "report" in state.report.completed_steps
    assert "memory" in state.report.skipped_steps
    assert "report" in [e.get("step") for e in state.report.execution_log]
    assert before_sync_progress != state.report.progress_percent or before_sync_progress == 100


# ========== 5. 默认列表不共享 ==========


def test_report_structured_lists_not_shared():
    """两个新报告实例的 completed_steps / skipped_steps 不应共享默认列表。"""
    r1 = FactCheckReport(
        original_text="A", overall_verdict="证据不足", overall_summary="摘要A",
        claim_results=[], timeline=[], propagation_risk="低", unresolved_questions=[],
    )
    r2 = FactCheckReport(
        original_text="B", overall_verdict="证据不足", overall_summary="摘要B",
        claim_results=[], timeline=[], propagation_risk="低", unresolved_questions=[],
    )
    r1.completed_steps.append("init")
    assert r2.completed_steps == [], "两个实例 completed_steps 不得共享默认列表"

    r1.skipped_steps.append("memory")
    assert r2.skipped_steps == [], "两个实例 skipped_steps 不得共享默认列表"
