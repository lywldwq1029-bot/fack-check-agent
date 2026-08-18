"""工作流基础测试。"""

from src.workflow import run_fact_check_workflow


def test_workflow_runs_with_demo_text():
    text = "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
    report = run_fact_check_workflow(text)
    assert report is not None
    assert report.original_text == text
    assert len(report.claim_results) >= 3
    assert report.overall_verdict in {
        "已证实",
        "基本属实",
        "部分属实",
        "证据不足",
        "存在误导",
        "已证伪",
        "仍在发展",
    }


def test_workflow_includes_timeline():
    text = "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
    report = run_fact_check_workflow(text)
    assert len(report.timeline) > 0
