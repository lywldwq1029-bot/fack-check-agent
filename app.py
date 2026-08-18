"""溯真 Streamlit 网页界面。

状态渲染约定：不再根据 execution_log 的中文/位置信息猜测阶段完成情况，
而是完全依赖 FactCheckReport 的结构化字段：
- completed_steps: list[str]     已完成的阶段
- skipped_steps: list[str]       明确跳过的阶段（如记忆未启用）
- current_step: str              当前阶段 / "completed" / "failed"
- workflow_completed: bool       整流程是否已成功结束
- workflow_error: str | None     失败所在阶段
- progress_percent: int          0-100 整体进度

对于旧版本报告（缺少上述字段，例如升级前缓存于 session_state 中的旧报告），
使用兼容判断：若 report.claim_results 非空、overall_verdict 有值、generated_at 有值，
则视为"旧报告完成态"，将阶段全部标记为 completed。
"""

from __future__ import annotations

from datetime import datetime
import time as _time

import streamlit as st

from src.config import settings
from src.models import FactCheckReport, build_failure_report, REPORT_SCHEMA_VERSION, CheckResult
from src.quick_workflow import run_fact_check


DEMO_TEXT = (
    "网传某市因暴雨导致地铁全线停运，目前已有多人失联，教育部门通知全市学校停课三天。"
)

# ===== 七阶段（demo / llm 模式使用）=====
USER_PHASES = [
    "decompose",
    "plan",
    "search",
    "sufficiency",
    "evaluate",
    "conclusion",
    "report",
]
USER_PHASE_LABELS = {
    "init": "初始化",
    "decompose": "拆解待核查主张",
    "plan": "制定核查计划",
    "search": "搜索公开网页",
    "sufficiency": "整理来源证据",
    "evaluate": "交叉验证证据",
    "conclusion": "形成核查结论",
    "report": "生成核查报告",
    "timeout": "执行超时",
}

# ===== 六阶段（专业核查模式使用）=====
QUICK_PHASES = [
    "receive",
    "memory",
    "search",
    "plan",
    "analyze",
    "output",
]
QUICK_PHASE_LABELS = {
    "init": "初始化",
    "receive": "接收主张",
    "memory": "查询历史记忆",
    "search": "搜索公开来源",
    "plan": "Agent 评估证据",
    "analyze": "最终判断",
    "output": "输出结论",
    "timeout": "执行超时",
}

USER_PHASE_FINAL = "核查完成"

# 状态标记
_PHASE_WAITING = "等待"
_PHASE_RUNNING = "正在执行"
_PHASE_DONE = "已完成"
_PHASE_PARTIAL = "部分完成"
_PHASE_FAILED = "失败"
_PHASE_TIMEOUT = "超时"

RISK_LABELS = {"low": "低", "medium": "中", "high": "高"}

MODE_LABELS = {
    "demo": "演示模式（零网络）",
    "llm": "真实LLM拆解",
    "full": "真实核查（推荐）",
}
MODE_DESCRIPTIONS = {
    "demo": "完全离线演示：无需 Tavily/LLM，使用内置模拟数据。适合快速体验整套 UI。\n耗时：< 5 秒",
    "llm": "只用 LLM 完成主张拆解与判断，搜索走 Mock 数据。\n需要 LLM 配置，无需 Tavily。",
    "full": "真实核查：单次 Tavily 搜索 + 单次 LLM 判断。\n需要完整配置，60 秒内完成。",
}


def _has_structured_progress(report: FactCheckReport) -> bool:
    """判断报告是否携带结构化进度字段。"""
    phases = QUICK_PHASES
    known = {"completed", "failed", *phases}
    if report.progress_percent != 0:
        return True
    if report.current_step in known and report.current_step not in ("init",):
        return True
    if report.completed_steps or report.skipped_steps:
        return True
    if report.workflow_completed is True or report.workflow_error is not None:
        return True
    return False


def _is_legacy_report_completed(report: FactCheckReport) -> bool:
    """旧报告兼容性判断：报告已生成且包含主张结论 + 生成时间，视为已完成。"""
    if (
        report.claim_results
        and report.overall_verdict
        and report.generated_at is not None
    ):
        return True
    return False


def _normalize_report_progress(report: FactCheckReport) -> FactCheckReport:
    """就地归一化阶段状态。返回同一个 report 对象。"""
    if _has_structured_progress(report):
        return report

    if _is_legacy_report_completed(report):
        report.current_step = "completed"
        report.completed_steps = list(QUICK_PHASES)
        report.skipped_steps = []
        report.progress_percent = 100
        report.workflow_completed = True
        report.workflow_error = None
    else:
        report.current_step = "failed"
        report.completed_steps = []
        report.skipped_steps = []
        report.progress_percent = 0
        report.workflow_completed = False
        report.workflow_error = "init"
    return report


def render_header() -> None:
    """渲染页面标题与说明（复古报纸+现代明亮风格）。"""
    st.set_page_config(
        page_title="溯真｜新闻溯源核查 Agent",
        page_icon="📰",
        layout="wide",
    )

    # ====== 自定义 CSS：复古报纸 + 现代明亮风格 ======
    st.markdown(
        """
        <style>
        /* ===== 全局背景：淡纸张纹理 ===== */
        .stApp {
            background-color: #FFFBF2;
            background-image:
                radial-gradient(rgba(222,184,135,0.06) 1px, transparent 1px),
                radial-gradient(rgba(222,184,135,0.04) 1px, transparent 1px);
            background-size: 20px 20px, 10px 10px;
            background-position: 0 0, 5px 5px;
        }
        /* ===== 顶部标题栏：报纸风格 ===== */
        .newspaper-header {
            background: linear-gradient(180deg, #FFFBF2 0%, #FDF4E3 100%);
            border-bottom: 2px solid #1E40AF;
            border-top: 1px solid rgba(30,64,175,0.2);
            padding: 20px 32px 16px;
            margin: -24px -24px 24px;
            position: relative;
        }
        .newspaper-header::before {
            content: "";
            position: absolute;
            left: 32px; right: 32px; top: 12px;
            border-top: 1px solid rgba(30,64,175,0.3);
        }
        .newspaper-title {
            font-family: Georgia, 'Times New Roman', 'Noto Serif SC', serif;
            font-size: 38px;
            font-weight: 700;
            color: #1E3A8A;
            letter-spacing: 2px;
            margin: 0;
            text-align: center;
        }
        .newspaper-subtitle {
            text-align: center;
            color: #64748B;
            font-size: 14px;
            margin-top: 6px;
            letter-spacing: 1px;
        }
        .newspaper-date {
            text-align: center;
            color: #94A3B8;
            font-size: 12px;
            margin-top: 4px;
            font-style: italic;
        }
        /* ===== 卡片：纸张质感 ===== */
        .paper-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFFCF6 100%);
            border: 1px solid rgba(180,140,100,0.2);
            border-radius: 6px;
            padding: 20px;
            box-shadow:
                0 1px 3px rgba(0,0,0,0.04),
                0 4px 12px rgba(180,140,100,0.06),
                inset 0 0 60px rgba(222,184,135,0.03);
            position: relative;
            margin-bottom: 16px;
        }
        .card-title {
            font-family: Georgia, 'Noto Serif SC', serif;
            color: #1E3A8A;
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 12px;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(30,64,175,0.15);
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        /* ===== 输入区大卡片 ===== */
        .input-card {
            background: linear-gradient(180deg, #FFFFFF 0%, #FFFCF6 100%);
            border: 1px solid rgba(30,64,175,0.15);
            border-radius: 8px;
            padding: 24px;
            box-shadow:
                0 2px 8px rgba(30,64,175,0.06),
                0 8px 24px rgba(180,140,100,0.08);
        }
        .input-card-title {
            font-family: Georgia, 'Noto Serif SC', serif;
            font-size: 22px;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 4px;
        }
        .input-card-desc {
            color: #64748B;
            font-size: 13px;
            margin-bottom: 16px;
        }
        /* ===== 按钮：现代扁平+纸张浮雕 ===== */
        .stButton > button {
            background: linear-gradient(180deg, #2563EB 0%, #1E40AF 100%);
            color: #FFFFFF !important;
            border: none;
            border-radius: 6px;
            padding: 10px 24px;
            font-weight: 600;
            letter-spacing: 0.5px;
            box-shadow: 0 2px 6px rgba(30,64,175,0.15), inset 0 1px 0 rgba(255,255,255,0.15);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%);
            box-shadow: 0 4px 12px rgba(30,64,175,0.25), inset 0 1px 0 rgba(255,255,255,0.15);
            transform: translateY(-1px);
        }
        .stDownloadButton > button {
            background: #FFFFFF;
            color: #1E3A8A !important;
            border: 1px solid rgba(30,64,175,0.25);
            border-radius: 6px;
            font-weight: 500;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.8);
            transition: all 0.2s ease;
        }
        .stDownloadButton > button:hover {
            background: #EFF6FF;
            border-color: rgba(30,64,175,0.4);
            box-shadow: 0 2px 6px rgba(30,64,175,0.1);
        }
        /* ===== 结论标签 ===== */
        .verdict-badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
            letter-spacing: 0.5px;
        }
        .verdict-true { background: #ECFDF5; color: #0D9488; border: 1px solid #6EE7B7; }
        .verdict-false { background: #FEF2F2; color: #B91C1C; border: 1px solid #FCA5A5; }
        .verdict-partial { background: #FFF7ED; color: #C2410C; border: 1px solid #FDBA74; }
        .verdict-doubt { background: #EFF6FF; color: #1E40AF; border: 1px solid #93C5FD; }
        /* ===== 风险标记 ===== */
        .risk-high { background: #FEF2F2; color: #991B1B; border-left: 4px solid #DC2626; padding: 12px 16px; border-radius: 4px; }
        .risk-medium { background: #FFFBEB; color: #92400E; border-left: 4px solid #D97706; padding: 12px 16px; border-radius: 4px; }
        .risk-low { background: #ECFDF5; color: #065F46; border-left: 4px solid #059669; padding: 12px 16px; border-radius: 4px; }
        /* ===== 证据标签 ===== */
        .evidence-tag {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
            margin-right: 6px;
        }
        .tag-grade-a { background: #DCFCE7; color: #166534; }
        .tag-grade-b { background: #DBEAFE; color: #1E40AF; }
        .tag-grade-c { background: #FEF3C7; color: #92400E; }
        .tag-grade-d { background: #FEE2E2; color: #991B1B; }
        .tag-support { background: #D1FAE5; color: #065F46; }
        .tag-refute { background: #FEE2E2; color: #991B1B; }
        .tag-partial { background: #FEF9C3; color: #854D0E; }
        /* ===== 时间线 ===== */
        .timeline-node { display: flex; align-items: flex-start; margin-bottom: 16px; position: relative; }
        .timeline-node::before {
            content: ""; position: absolute; left: 11px; top: 24px; bottom: -16px;
            width: 1px; background: rgba(30,64,175,0.2);
        }
        .timeline-node:last-child::before { display: none; }
        .timeline-dot {
            width: 24px; height: 24px; border-radius: 50%;
            background: #2563EB; border: 3px solid #DBEAFE;
            flex-shrink: 0; margin-right: 12px; margin-top: 2px;
        }
        .timeline-content { flex: 1; }
        .timeline-date { font-size: 12px; color: #64748B; font-weight: 500; }
        .timeline-title { font-size: 14px; font-weight: 600; color: #1E293B; margin: 2px 0; }
        .timeline-source { font-size: 12px; color: #94A3B8; }
        /* ===== 可信度进度条自定义 ===== */
        .credibility-bar {
            height: 14px;
            background: #F1F5F9;
            border-radius: 10px;
            overflow: hidden;
            position: relative;
            border: 1px solid rgba(30,64,175,0.1);
        }
        .credibility-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.6s ease;
        }
        .cred-high { background: linear-gradient(90deg, #059669, #10B981); }
        .cred-mid  { background: linear-gradient(90deg, #D97706, #F59E0B); }
        .cred-low  { background: linear-gradient(90deg, #DC2626, #EF4444); }
        .cred-label {
            display: flex; justify-content: space-between;
            font-size: 12px; color: #64748B; margin-bottom: 4px;
        }
        /* ===== Streamlit 组件覆盖 ===== */
        .stTextArea > div > div > textarea {
            background: #FFFFFF !important;
            border: 1px solid rgba(210,180,140,0.3) !important;
            border-radius: 6px !important;
            font-size: 14px; line-height: 1.7; padding: 14px 16px !important;
        }
        .stTextArea > div > div > textarea:focus {
            border-color: #2563EB !important;
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: transparent;
            border-bottom: 1px solid rgba(30,64,175,0.15); gap: 0;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px; font-size: 14px; font-weight: 500; color: #64748B;
            border-bottom: 2px solid transparent; border-radius: 0; background: transparent !important;
        }
        .stTabs [data-baseweb="tab"]:hover { color: #1E40AF; }
        .stTabs [aria-selected="true"] { color: #1E40AF !important; border-bottom-color: #2563EB !important; }
        .stMetric {
            background: #FFFFFF; padding: 12px 16px; border-radius: 6px;
            border: 1px solid rgba(210,180,140,0.2);
        }
        .stSidebar {
            background: linear-gradient(180deg, #FFFBF2 0%, #FDF4E3 100%) !important;
            border-right: 1px solid rgba(210,180,140,0.2);
        }
        hr {
            border: none !important; height: 1px !important;
            background: linear-gradient(90deg, transparent 0%, rgba(30,64,175,0.25) 20%, rgba(30,64,175,0.4) 50%, rgba(30,64,175,0.25) 80%, transparent 100%) !important;
            margin: 20px 0 !important;
        }
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #2563EB, #3B82F6) !important;
            border-radius: 10px !important;
        }
        .stExpander {
            border: 1px solid rgba(210,180,140,0.2) !important;
            border-radius: 6px !important;
            background: #FFFCF6 !important;
        }
        /* ===== 证据跳转标签 ===== */
        .evidence-link {
            color: #1E40AF;
            text-decoration: none;
            font-size: 13px;
            border-bottom: 1px dashed rgba(30,64,175,0.3);
            transition: all 0.15s ease;
        }
        .evidence-link:hover {
            color: #2563EB;
            border-bottom-color: #2563EB;
            border-bottom-style: solid;
        }
        /* ===== 复古分栏线 ===== */
        .column-divider {
            border-left: 1px solid rgba(30,64,175,0.12);
            height: 100%;
            margin: 0 12px;
        }
        .section-subtitle {
            font-family: Georgia, 'Noto Serif SC', serif;
            font-size: 15px;
            color: #1E3A8A;
            font-weight: 600;
            margin: 12px 0 8px;
            padding-bottom: 4px;
            border-bottom: 1px dotted rgba(30,64,175,0.15);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 渲染报纸风格标题栏
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y年%m月%d日")
    weekday_map = {"Monday": "星期一", "Tuesday": "星期二", "Wednesday": "星期三",
                   "Thursday": "星期四", "Friday": "星期五", "Saturday": "星期六", "Sunday": "星期日"}
    wd = weekday_map.get(_dt.now().strftime("%A"), "")
    st.markdown(
        f"""
        <div class="newspaper-header">
            <h1 class="newspaper-title">📰 溯 真 · 新闻溯源核查 Agent</h1>
            <div class="newspaper-subtitle">NEWS VERIFICATION CHRONICLE · 以事实追溯真相</div>
            <div class="newspaper-date">No.2026 · {today} {wd} · 第 {_dt.now().strftime('%H:%M')} 版</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_sidebar() -> str:
    """渲染侧边栏，支持三种模式切换，默认选中「演示模式」。返回所选模式标识。"""
    with st.sidebar:
        st.header("运行配置")

        # 模式选择器（默认演示模式，无需联网即可体验）
        mode_options = list(MODE_LABELS.keys())
        mode_labels = [MODE_LABELS[k] for k in mode_options]
        if "fact_check_mode" not in st.session_state:
            st.session_state.fact_check_mode = "demo"
        selected_label = st.selectbox(
            "核查模式",
            mode_labels,
            index=mode_options.index(st.session_state.fact_check_mode),
            help="演示模式无需任何配置即可立即体验 UI 全貌",
        )
        selected_mode = mode_options[mode_labels.index(selected_label)]
        st.session_state.fact_check_mode = selected_mode
        st.info(f"**当前模式：{MODE_LABELS[selected_mode]}**\n\n{MODE_DESCRIPTIONS[selected_mode]}")

        st.divider()
        st.subheader("模型信息")
        st.markdown(f"**当前模型**：`{settings.masked_model_name()}`")
        llm_ok = settings.llm_configured()
        st.markdown(f"**LLM 配置状态**：{'✅ 已配置' if llm_ok else '❌ 未配置'}")
        search_ok = settings.search_configured()
        st.markdown(f"**Tavily 配置状态**：{'✅ 已配置' if search_ok else '❌ 未配置'}")

        # 模式预检
        missing: list[str] = []
        if selected_mode in ("llm", "full") and not llm_ok:
            missing += ["LLM_API_KEY", "LLM_MODEL"]
        if selected_mode == "full" and not search_ok:
            missing.append("TAVILY_API_KEY")
        if missing:
            st.warning(
                f"{MODE_LABELS[selected_mode]} 缺少以下配置：{', '.join(missing)}\n\n"
                "请在项目根目录的 `.env` 中补全，或切换至「演示模式」。"
            )

        # 测试按钮（根据当前模式禁用不可用的）
        col_a, col_b = st.columns(2)
        with col_a:
            if llm_ok and st.button("测试模型连接", use_container_width=True):
                _test_llm_connection()
        with col_b:
            if search_ok and st.button("测试搜索连接", use_container_width=True):
                _test_search_connection()

        st.divider()
        if selected_mode == "demo":
            st.success(
                "✅ 演示模式无需任何网络连接：\n"
                "- 使用内置模拟数据生成完整报告\n"
                "- 可用于验证 UI、导出 Markdown 报告\n"
                "- 建议先用此模式熟悉流程"
            )
        elif selected_mode == "llm":
            st.info(
                "真实LLM拆解模式：\n"
                "- LLM 完成主张拆解与判断\n"
                f"- LLM 超时：{settings.LLM_TIMEOUT}s\n"
                "- 搜索走本地 Mock，不需要 Tavily"
            )
        else:
            st.info(
                "真实核查模式（快速链路）：\n"
                f"- 单次 Tavily 搜索（{settings.TAVILY_TIMEOUT}s 超时，自动重试 1 次）\n"
                "- 保留最相关 3 条来源\n"
                f"- 单次 LLM 判断（{settings.LLM_TIMEOUT}s 超时）\n"
                "- 整体 60s 内必须结束"
            )

    return selected_mode


def _test_llm_connection() -> None:
    """测试大模型连接。"""
    from src.llm.client import LLMClient, LLMError

    try:
        client = LLMClient()
        reply = client.chat(
            system_prompt="你是一个测试助手。",
            user_prompt="请回复：连接成功",
            temperature=0.0,
        )
        st.success(f"✅ 连接成功，模型回复：{reply[:50]}")
    except LLMError as e:
        st.error(f"❌ 连接失败：{e}")
    except Exception as e:
        st.error(f"❌ 未知错误：{e}")


def _test_search_connection() -> None:
    """测试搜索连接（只显示成功、失败和耗时）。"""
    import time
    from src.tools.search_tool import TavilySearchProvider

    try:
        provider = TavilySearchProvider()
        start = time.time()
        results, response_time, err = provider.search(
            query="测试", max_results=1, topic="general"
        )
        elapsed = time.time() - start
        if err:
            st.error(f"❌ 搜索失败：{err}")
        else:
            st.success(f"✅ 搜索成功，返回 {len(results)} 条结果，耗时 {elapsed:.2f} 秒")
    except Exception as e:
        st.error(f"❌ 搜索失败：{e}")


def render_input() -> str | None:
    """渲染输入区域并返回用户输入文本（纸张大卡片风格）。"""
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="input-card-title">📝 输入待核查内容</h2>', unsafe_allow_html=True)
    st.markdown('<div class="input-card-desc">粘贴新闻文本、微博、朋友圈截图转文字或文章链接，Agent 将自动溯源核查其真实性与信息来源。</div>', unsafe_allow_html=True)

    with st.form("fact_check_form", clear_on_submit=False):
        text = st.text_area(
            label="新闻文本",
            value=DEMO_TEXT,
            height=170,
            label_visibility="collapsed",
            placeholder="请粘贴新闻、微博、朋友圈截图转文字、文章链接等内容...",
        )

        col1, col2, col3 = st.columns([1.2, 3.8, 1])
        with col1:
            missing = settings.missing_configs()
            can_start = not missing

            submitted = st.form_submit_button(
                "🔍 开始溯源核查",
                type="primary",
                use_container_width=True,
                disabled=not can_start,
            )
        with col2:
            if not can_start:
                st.caption(f"⚠️ 请先在 .env 中配置：{', '.join(missing)}")
            else:
                st.caption("提交后，Agent 将在 60 秒内完成检索-验证-报告生成全流程。")
        with col3:
            st.empty()  # 占位，右侧留白

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        return text
    return None


def _friendly_error_message(err_phase: str, summary: str = "") -> str:
    """将技术错误码转换为面向普通用户的中文提示。"""
    err_text = (summary or "") + " " + (err_phase or "")
    if "timeout" in err_text.lower() or "超时" in err_text:
        return "本次核查超过时间上限，请稍后重试。"
    if "model" in err_text.lower() or "llm" in err_text.lower() or "api_key" in err_text.lower():
        return "模型连接失败：请检查模型名称、API 地址、密钥或额度。"
    if "密钥无效" in err_text or "401" in err_text or "403" in err_text:
        return "搜索服务密钥无效：请检查 TAVILY_API_KEY 是否正确。"
    if "额度或请求频率受限" in err_text or "429" in err_text or "rate limit" in err_text.lower():
        return "搜索额度或请求频率受限：请稍后重试或升级 Tavily 套餐。"
    if "连接临时中断" in err_text or "ssl" in err_text.lower():
        return "搜索服务连接临时中断，请重试。"
    if "tavily" in err_text.lower() or "搜索服务" in err_text or "搜索" in err_text:
        return "搜索服务连接失败：请检查 Tavily 密钥或额度。"
    if "json" in err_text.lower() or "解析" in err_text or "格式" in err_text:
        return "模型返回内容格式异常：请重新核查。"
    if "无搜索结果" in err_text or "无结果" in err_text or "没有结果" in err_text:
        return "未检索到有效来源：当前不能形成可靠结论。"
    if "不可用" in err_text or "所有来源" in err_text:
        return "所有来源均不可用：当前无法完成核查。"
    if "报告生成" in err_text:
        return "报告生成失败：请稍后重试。"
    return f"核查在「{err_phase}」阶段发生异常，请稍后重试。"


def _phase_status(
    phase: str,
    completed: set,
    in_progress: str,
    failed: bool,
    failed_phase: str | None,
    report_done: bool,
    has_error: bool,
) -> tuple[str, str]:
    """返回 (emoji+status_text, color_style) 用于展示单个阶段的状态。

    规则：
    - 已完成的阶段 → 绿色
    - 当前失败阶段 → 红色
    - 当前进行中阶段 → 蓝色
    - 其他等待阶段 → 灰色
    """
    if phase in completed:
        return f"✅ {_PHASE_DONE}", "green"

    if failed and failed_phase == phase:
        return f"❌ {_PHASE_FAILED}", "red"

    # 映射：evaluate 进行中时，conclusion 也显示为进行中
    is_active = (in_progress == phase)
    if phase == "conclusion" and in_progress == "evaluate":
        is_active = True

    if not failed and is_active:
        return f"🔵 {_PHASE_RUNNING}", "blue"

    return f"⚪ {_PHASE_WAITING}", "gray"


def _is_valid_http_url(url: str) -> bool:
    """验证 URL 是否为合法的 http/https 链接。"""
    if not url or not isinstance(url, str):
        return False
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return bool(parsed.netloc) and "." in parsed.netloc
    except Exception:
        return False


def render_execution_state(report: FactCheckReport, elapsed_seconds: float = 0.0) -> None:
    """渲染执行状态区域。"""
    st.subheader("核查执行状态")

    phases = QUICK_PHASES
    phase_labels = QUICK_PHASE_LABELS

    report = _normalize_report_progress(report)

    completed = set(report.completed_steps)

    in_progress = report.current_step
    failed = report.current_step == "failed"
    failed_phase = report.workflow_error
    has_error = bool(report.workflow_error)
    report_done = report.workflow_completed or report.current_step == "completed"

    completed_user = [p for p in phases if p in completed]
    progress = max(0, min(100, int(len(completed_user) / len(phases) * 100)))
    st.progress(progress / 100.0, text=f"工作流进度：{progress}%")

    elapsed_text = f"已用时 {elapsed_seconds:.1f} 秒" if elapsed_seconds > 0 else "计时中..."
    st.caption(f"⏱️ {elapsed_text}")

    cols = st.columns(len(phases))
    for i, phase in enumerate(phases):
        with cols[i]:
            label = phase_labels.get(phase, phase)
            status_text, color = _phase_status(
                phase, completed, in_progress, failed, failed_phase, report_done, has_error
            )
            st.markdown(f"<span style='color:{color};'>{status_text}</span>", unsafe_allow_html=True)
            st.caption(f"{i + 1}. {label}")

    if report_done and not failed:
        final_label = USER_PHASE_FINAL
        last_col = cols[-1]
        with last_col:
            st.markdown(f"<span style='color:green;'>✅ {_PHASE_DONE}</span>", unsafe_allow_html=True)
            st.caption(f"{len(phases)}. {final_label}")

    if failed and report.workflow_error:
        err = report.workflow_error
        friendly = _friendly_error_message(err, report.overall_summary)
        st.error(f"❌ {friendly}")
    elif report.workflow_error == "timeout" or (report.overall_summary and "超时" in report.overall_summary):
        st.warning(
            "⚠️ **核查未完全完成**：工作流因超时提前结束。"
            "以下为已取得的证据，不代表最终结论。建议稍后重试或简化输入文本。"
        )


def render_decompose_results(report: FactCheckReport) -> None:
    """渲染主张拆解结果。"""
    st.subheader("主张拆解结果")

    if not report.claim_results:
        st.info("本次未拆解出可核查主张（文本可能主要为观点或信息不足）。")
        return

    for idx, result in enumerate(report.claim_results, start=1):
        claim = result.claim
        risk_label = RISK_LABELS.get(claim.risk_level, claim.risk_level)
        with st.expander(f"{idx}. {claim.text}　[{risk_label}风险]"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**主张类型**：{claim.claim_type}")
                st.markdown(f"**风险等级**：{risk_label}")
                st.markdown(f"**是否可核查**：{'是' if claim.is_checkable else '否'}")
            with col_b:
                st.markdown(f"**是否观点**：{'是' if claim.is_opinion else '否'}")
                if claim.time_reference:
                    st.markdown(f"**时间参照**：{claim.time_reference}")
                if claim.location:
                    st.markdown(f"**地点**：{claim.location}")

            if claim.verification_question:
                st.markdown(f"**核查问题**：{claim.verification_question}")
            if claim.sensitive_reason:
                st.markdown(f"**敏感性说明**：{claim.sensitive_reason}")
            if claim.search_keywords:
                st.markdown(f"**建议搜索关键词**：{', '.join(claim.search_keywords)}")
            if claim.preferred_source_types:
                st.markdown(f"**优先来源类型**：{', '.join(claim.preferred_source_types)}")


def render_verification_plan(report: FactCheckReport) -> None:
    """渲染 Agent 核查计划。"""
    st.subheader("Agent 核查计划")

    if not report.claim_results:
        st.info("无可核查主张，未生成核查计划。")
        return

    for idx, result in enumerate(report.claim_results, start=1):
        claim = result.claim
        with st.expander(f"计划 {idx}：{claim.text}"):
            if claim.verification_question:
                st.markdown(f"**核查方向**：{claim.verification_question}")
            if claim.search_keywords:
                st.markdown("**搜索语句**：")
                for kw in claim.search_keywords:
                    st.markdown(f"- {kw}")
            if claim.preferred_source_types:
                st.markdown(f"**优先来源**：{', '.join(claim.preferred_source_types)}")
            st.markdown(f"**核查优先级**：{claim.verification_priority}")


def render_overall_verdict(report: FactCheckReport) -> None:
    """渲染总体结论（复古报纸卡片 + 自定义可信度进度条）。"""
    verdict = getattr(report, "overall_verdict", "暂无法核查")
    summary = getattr(report, "overall_summary", "") or "判断失败"

    # 专业判定标准颜色映射
    verdict_colors = {
        "基本属实": "#2E7D32",
        "部分属实": "#FF8F00",
        "存在错误": "#E53935",
        "已证伪": "#B71C1C",
        "证据不足": "#757575",
        "暂无法核查": "#FF6F00",
    }
    color = verdict_colors.get(verdict, "gray")
    verdict_class_map = {
        "基本属实": "verdict-true",
        "部分属实": "verdict-partial",
        "存在错误": "verdict-false",
        "已证伪": "verdict-false",
        "证据不足": "verdict-doubt",
        "暂无法核查": "verdict-doubt",
    }
    vcls = verdict_class_map.get(verdict, "verdict-doubt")

    # 安全读取可选字段
    credibility = getattr(report, "credibility_score", None)
    recommendation = getattr(report, "recommendation", "") or ""

    # ==== 复古报纸卡片 ====
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🎯 核查结论 · Overall Verdict</h2>', unsafe_allow_html=True)

    # 结论卡片
    st.markdown(
        f"""
        <div style="padding: 1.2rem 1.4rem; border-left: 5px solid {color}; background: linear-gradient(180deg, #FFFCF6 0%, #FFFFFF 100%); border-radius: 6px; border-top: 1px solid rgba(210,180,140,0.15); border-right: 1px solid rgba(210,180,140,0.15); border-bottom: 1px solid rgba(210,180,140,0.15);">
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
            <span class="verdict-badge {vcls}" style="font-size: 18px; padding: 6px 20px;">{verdict}</span>
          </div>
          <p style="margin: 0; font-size: 1.05rem; line-height: 1.7; color: #1E293B;">{summary}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 可信度进度条（自定义 HTML）
    cred_label = "可信度评估"
    if credibility is None:
        cred_pct = 0
        cred_class = "cred-low"
        cred_label += "：暂无法评估"
    elif credibility >= 70:
        cred_pct = credibility
        cred_class = "cred-high"
        cred_label += f"：高 ({credibility}%)"
    elif credibility >= 40:
        cred_pct = credibility
        cred_class = "cred-mid"
        cred_label += f"：中 ({credibility}%)"
    else:
        cred_pct = credibility
        cred_class = "cred-low"
        cred_label += f"：低 ({credibility}%)"

    st.markdown(
        f"""
        <div style="padding: 12px 16px; background: rgba(37,99,235,0.03); border-radius: 6px; border: 1px solid rgba(37,99,235,0.08);">
          <div class="cred-label"><span style="font-weight: 600; color: #1E3A8A;">📊 {cred_label}</span><span>0%  100%</span></div>
          <div class="credibility-bar">
            <div class="credibility-fill {cred_class}" style="width: {cred_pct}%;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 建议
    if recommendation:
        if "不建议" in recommendation:
            st.warning(f"⚠️ {recommendation}")
        else:
            st.info(f"💡 {recommendation}")

    # 工具调用统计
    tool_calls = getattr(report, "tool_calls_count", 0) or 0
    if tool_calls > 0:
        tool_info = f"本次核查调用工具 {tool_calls} 次"
        did_supp = getattr(report, "did_supplemental_search", False)
        if did_supp:
            tool_info += "（含 1 次补充搜索）"
        st.caption(tool_info)

    st.markdown('</div>', unsafe_allow_html=True)

    # 提示信息
    if verdict == "暂无法核查":
        st.info(
            "💡 搜索服务暂不可用，您可以点击「🔄 重新搜索」按钮重新尝试，或稍后再试。"
        )
    elif verdict == "证据不足":
        st.info(
            "💡 现有证据不足以做出明确判断，建议谨慎参考或进一步核实。"
        )
    elif verdict == "已证伪":
        st.warning(
            "⚠️ 此结论已被权威来源证伪，请谨慎对待相关信息。"
        )


def render_claim_results(report: FactCheckReport) -> None:
    """渲染主张核查表。"""
    st.subheader("主张核查表")

    if not report.claim_results:
        st.info("无可核查主张。")
        return

    for idx, result in enumerate(report.claim_results, start=1):
        with st.expander(f"{idx}. {result.claim.text} — {result.verdict}"):
            st.markdown(f"**结论**：{result.verdict}")
            st.markdown(f"**置信度**：{result.confidence:.0%}")
            st.markdown(f"**推理**：{result.reasoning}")
            if result.missing_information:
                st.markdown(f"**缺失信息**：{result.missing_information}")

def render_timeline(report: FactCheckReport) -> None:
    """渲染事件时间线（复古报纸风格自定义时间线节点）。"""
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">⏰ 事件时间线溯源 · Timeline</h2>', unsafe_allow_html=True)

    if not report.timeline or len(report.timeline) == 0:
        st.info("本次核查未整理出明确事件时间线。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    timeline = list(report.timeline)
    def _tl_sort_key(e):
        t = getattr(e, "event_time", None) or getattr(e, "date", None) or ""
        desc = getattr(e, "description", None) or ""
        return (str(t), desc)
    try:
        timeline = sorted(timeline, key=_tl_sort_key)
    except Exception:
        timeline = list(report.timeline)

    timeline_html_parts = ['<div class="timeline">']
    for idx, event in enumerate(timeline):
        date = getattr(event, "date", None) or ""
        if not date:
            et = getattr(event, "event_time", None)
            if et:
                try:
                    date = et.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date = str(et)
        desc = getattr(event, "description", None) or str(event)
        sources = getattr(event, "sources", None) or []
        src_url = getattr(event, "source_url", None) or ""
        if not sources and src_url:
            sources = [{"source_url": src_url, "source_domain": ""}]

        if idx == 0:
            node_cls = "timeline-node first"
        elif idx == len(timeline) - 1:
            node_cls = "timeline-node last"
        else:
            node_cls = "timeline-node"

        src_html = ""
        if sources:
            src_parts = []
            for src in sources[:3]:
                domain = ""; url = ""
                if isinstance(src, dict):
                    domain = src.get("source_domain") or src.get("domain") or ""
                    url = src.get("source_url") or src.get("url") or ""
                else:
                    domain = getattr(src, "source_domain", None) or getattr(src, "domain", None) or ""
                    url = getattr(src, "source_url", None) or getattr(src, "url", None) or ""
                if url:
                    src_parts.append(f'<a href="{url}" target="_blank" class="evidence-link" rel="noopener">{domain or "证据链接"}</a>')
                elif domain:
                    src_parts.append(f'<span class="tag tag-secondary">{domain}</span>')
            if src_parts:
                src_html = '<div style="margin-top: 6px;">📎 证据：' + ' ｜ '.join(src_parts) + '</div>'
        elif src_url and _is_valid_http_url(src_url):
            src_html = f'<div style="margin-top: 6px;">📎 证据：<a href="{src_url}" target="_blank" class="evidence-link" rel="noopener">查看来源</a></div>'

        date_display = f"📅 {date}" if date else "📅 时间不详"
        timeline_html_parts.append(f'<div class="timeline-item"><div class="{node_cls}"></div><div class="timeline-content"><div class="timeline-date">{date_display}</div><div class="timeline-desc">{desc}</div>{src_html}</div></div>')
    timeline_html_parts.append('</div>')
    st.markdown('\n'.join(timeline_html_parts), unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_risk_and_questions(report: FactCheckReport) -> None:
    """渲染传播风险和待核实问题（复古报纸卡片 + 专业风险类名）。"""
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">⚠️ 风险标记与待核实问题 · Risk Assessment</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h3 class="subsection-title">📌 传播风险</h3>', unsafe_allow_html=True)
        risks = getattr(report, "propagation_risks", None) or []
        if isinstance(risks, str):
            risks = [{"level": "medium", "description": risks, "suggestion": None}]
        elif not risks:
            old_level = getattr(report, "risk_level", None)
            old_reason = getattr(report, "risk_reason", None)
            old_factors = getattr(report, "risk_factors", None) or []
            if old_level or old_reason:
                risks = [{"level": old_level or "medium", "description": (old_reason or "未评估"), "suggestion": None}]
                for f in old_factors[:3]:
                    risks.append({"level": old_level or "medium", "description": f"风险因素：{f}", "suggestion": None})

        if risks:
            for r in risks:
                level = r.get("level") if isinstance(r, dict) else (getattr(r, "level", None) if hasattr(r, "level") else "medium")
                desc = r.get("description") if isinstance(r, dict) else (getattr(r, "description", None) or str(r))
                suggestion = r.get("suggestion") if isinstance(r, dict) else getattr(r, "suggestion", None)
                lvl_map = {"high":"risk-high","HIGH":"risk-high","高":"risk-high","medium":"risk-medium","MID":"risk-medium","中":"risk-medium","low":"risk-low","LOW":"risk-low","低":"risk-low"}
                lvl_label_map = {"high":"🔴 高风险","HIGH":"🔴 高风险","高":"🔴 高风险","medium":"🟠 中风险","MID":"🟠 中风险","中":"🟠 中风险","low":"🟢 低风险","LOW":"🟢 低风险","低":"🟢 低风险"}
                cls = lvl_map.get(str(level), "risk-medium")
                lbl = lvl_label_map.get(str(level), "🟠 中风险")
                sugg_html = f" <br />💡 建议：{suggestion}" if suggestion else ""
                st.markdown(f'<div class="{cls}" style="margin-bottom: 8px;"><strong>{lbl}：</strong>{desc}{sugg_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="risk-low">🟢 低风险：本次未发现明显传播风险。</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="subsection-title">❓ 待核实问题</h3>', unsafe_allow_html=True)
        questions = getattr(report, "open_questions", None) or getattr(report, "unresolved_questions", None) or []
        if questions:
            for i, q in enumerate(questions, start=1):
                if isinstance(q, dict):
                    q_text = q.get("question") or q.get("text") or str(q)
                else:
                    q_text = getattr(q, "question", None) or str(q)
                st.markdown(f'<div style="padding: 8px 12px; background: rgba(255, 152, 0, 0.04); border: 1px dashed rgba(255, 152, 0, 0.25); border-radius: 6px; margin-bottom: 6px; color: #4527A0;"><span style="color: #FF9800; font-weight: bold; margin-right: 6px;">Q{i}.</span>{q_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="padding: 8px 12px; background: rgba(46, 125, 50, 0.04); border: 1px dashed rgba(46, 125, 50, 0.2); border-radius: 6px; color: #1B5E20;">✅ 关键信息已核实，无待进一步核实的开放式问题。</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_key_evidence_cards(report: FactCheckReport) -> None:
    """渲染关键证据卡片（复古报纸风格）。"""
    key_cards = getattr(report, "key_evidence_cards", []) or []
    if not key_cards:
        return

    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">🔑 关键证据 · Key Evidence</h2>', unsafe_allow_html=True)

    grade_colors = {"A": "#2e7d32", "B": "#66bb6a", "C": "#ef6c00", "D": "#757575", "E": "#c62828"}
    for card in key_cards[:3]:
        grade = card.get("grade", "C")
        grade_color = grade_colors.get(grade, "#757575")
        title = card.get("title", "未知标题")
        url = card.get("url") or card.get("source_url") or ""
        summary = card.get("summary", "") or ""
        grade_desc = card.get("grade_desc", "")
        valid_url = _is_valid_http_url(url)
        tpl_title = '<a href="%s" target="_blank" rel="noopener" class="evidence-link"><strong>%s</strong></a>' % (url, title) if valid_url else "<strong style='color:#1E3A8A;'>%s</strong>" % title
        stance_html = ""
        if card.get("directly_supports") is True:
            stance_html = '<span class="tag tag-success" style="margin-left: 8px;">✅ 支持主张</span>'
        elif card.get("directly_supports") is False:
            stance_html = '<span class="tag tag-danger" style="margin-left: 8px;">❌ 反驳主张</span>'
        card_html = f'''<div style="padding: 14px 16px; margin-bottom: 10px; border-radius: 8px; background: linear-gradient(180deg, #FFFFFF 0%, #FFFBF4 100%); border: 1px solid rgba(210,180,140,0.2); box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
              <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; gap: 10px;">
                <div style="display:flex; align-items:center; gap: 8px; flex-wrap:wrap;">
                  <span style="background-color:{grade_color}; color:white; padding: 3px 12px; border-radius: 4px; font-size: 12px; font-weight: 600;">{grade}级 {grade_desc}</span>
                  {tpl_title}
                  {stance_html}
                </div>
              </div>
              <div style="color: #334155; font-size: 14px; line-height: 1.65;">{summary[:160]}</div>
            </div>'''
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_verification_table(report: FactCheckReport) -> None:
    """渲染结构化核查表格：符合课程要求的 7 列表格 + CSV 导出。"""
    st.markdown('<h3 class="subsection-title">📋 结构化核查表</h3>', unsafe_allow_html=True)
    if not report.claim_results:
        st.info("暂无核查结果")
        return
    import pandas as pd
    table_data = []
    for result in report.claim_results:
        claim_text = result.claim.text
        verdict = result.verdict
        reasoning = result.reasoning or result.rationale or ""
        if not result.evidence:
            table_data.append({"事实主张": claim_text, "来源链接": "—", "来源标题": "未检索到证据", "证据摘要": "—", "来源类型": "—", "判断": verdict, "理由": reasoning})
            continue
        for ev in result.evidence:
            summary = ev.summary if ev.summary else (ev.evidence_summary or "")
            summary = " ".join(summary.split())[:150]
            row = {
                "事实主张": claim_text[:80] if len(claim_text) > 80 else claim_text,
                "来源链接": ev.source_url,
                "来源标题": ev.source_title[:60] if len(ev.source_title) > 60 else ev.source_title,
                "证据摘要": summary,
                "来源类型": "%s级 · %s" % (ev.source_grade, ev.publisher),
                "判断": verdict,
                "理由": reasoning[:100] if len(reasoning) > 100 else reasoning,
            }
            table_data.append(row)
    if not table_data:
        st.info("暂无数据")
        return
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, column_config={
        "来源链接": st.column_config.LinkColumn("来源链接", display_text="🔗 打开"),
        "事实主张": st.column_config.TextColumn("事实主张", width="large"),
        "证据摘要": st.column_config.TextColumn("证据摘要", width="large"),
    }, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下载核查结果 (CSV)", data=csv, file_name="verification_result.csv", mime="text/csv", use_container_width=True, key="download_verification_table")

def render_final_summary_table(report: FactCheckReport) -> None:
    """渲染最终总结表格：符合课程要求的完整 7+1 列表格。"""
    st.markdown('<h3 class="subsection-title">📋 最终核查总结表</h3>', unsafe_allow_html=True)
    if not report.claim_results:
        st.info("暂无核查结果")
        return
    import pandas as pd
    table_data = []
    for result in report.claim_results:
        claim_text = result.claim.text
        verdict = result.verdict
        reasoning = result.reasoning or result.rationale or ""
        if not result.evidence:
            r1 = reasoning[:120] if reasoning and len(reasoning) > 120 else (reasoning or "—")
            ct = claim_text[:100] if len(claim_text) > 100 else claim_text
            table_data.append({"事实主张": ct, "来源链接": "", "来源标题": "未检索到证据", "证据摘要": "—", "来源类型": "—", "判断": verdict, "理由": r1, "支持/反驳": "—"})
            continue
        for ev in result.evidence:
            summary = ev.summary if ev.summary else (ev.evidence_summary or "")
            summary = " ".join(summary.split())[:120]
            if ev.supports_or_refutes == "supports" or getattr(ev, "directly_supports", None) is True:
                support_label = "✅ 支持"
            elif ev.supports_or_refutes == "refutes" or getattr(ev, "directly_supports", None) is False:
                support_label = "❌ 反驳"
            elif ev.supports_or_refutes == "partial":
                support_label = "⚖️ 部分"
            else:
                support_label = "❓ 未知"
            stype = "%s级 · %s" % (ev.source_grade, (ev.publisher or "")[:20])
            reason_out = reasoning[:120] if reasoning and len(reasoning) > 120 else (reasoning or "—")
            row = {
                "事实主张": claim_text[:100] if len(claim_text) > 100 else claim_text,
                "来源链接": ev.source_url,
                "来源标题": ev.source_title[:80] if len(ev.source_title) > 80 else ev.source_title,
                "证据摘要": summary,
                "来源类型": stype,
                "判断": verdict,
                "理由": reason_out,
                "支持/反驳": support_label,
            }
            table_data.append(row)
    if not table_data:
        st.info("暂无数据")
        return
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, column_config={
        "来源链接": st.column_config.LinkColumn("来源链接", display_text="🔗 打开", width="small"),
        "事实主张": st.column_config.TextColumn("事实主张", width="large"),
        "证据摘要": st.column_config.TextColumn("证据摘要", width="large"),
        "支持/反驳": st.column_config.TextColumn("支持/反驳", width="small"),
    }, hide_index=True)
    csv2 = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 下载核查结果 (CSV)", data=csv2, file_name="verification_final_summary.csv", mime="text/csv", use_container_width=True, key="download_final_summary")

def render_agent_decision_trace(report: FactCheckReport) -> None:
    """渲染 Agent 决策轨迹（折叠展示）。"""
    decision_trace = getattr(report, "decision_trace", []) or []
    agent_decision = getattr(report, "agent_decision", None)
    tool_calls = getattr(report, "tool_calls_count", 0) or 0
    if not decision_trace and not agent_decision and tool_calls <= 0:
        return
    with st.expander("🤖 Agent 决策轨迹", expanded=False):
        if decision_trace:
            st.markdown("### 决策过程")
            for i, step in enumerate(decision_trace, 1):
                step_name = step.get("step", "")
                st.markdown("**%s. %s**" % (i, step_name))
                if "query" in step:
                    st.markdown("🔍 搜索关键词：`%s`" % (str(step["query"])[:50],))
                if "result" in step:
                    st.markdown("📋 结果：%s" % (step["result"],))
                if "action" in step:
                    st.markdown("⚡ 行动：%s" % (step["action"],))
                if "reason" in step:
                    st.markdown("💭 理由：%s" % (step["reason"],))
                if "cache" in step:
                    cache_map = {"hit_24h":"24小时缓存命中","fallback_72h":"72小时兜底缓存","live":"实时搜索","none":"无缓存"}
                    cache_label = cache_map.get(step["cache"], step["cache"])
                    st.markdown("💾 缓存：%s" % (cache_label,))
                st.divider()
        if agent_decision:
            st.markdown("### 最终决策")
            d = agent_decision
            st.markdown("- **主张类型**：%s" % (getattr(d, "claim_type", "未说明"),))
            st.markdown("- **敏感度**：%s" % (getattr(d, "sensitivity", "未说明"),))
            st.markdown("- **证据要求**：%s" % (getattr(d, "evidence_requirement", "未说明"),))
            es = getattr(d, "evidence_sufficient", False)
            st.markdown("- **证据是否充分**：%s" % ("✅ 是" if es else "❌ 否",))
            me = getattr(d, "missing_evidence", [])
            if me:
                st.markdown("- **缺少的证据**：%s" % (", ".join([str(x) for x in me]),))
            st.markdown("- **最终行动**：%s" % (getattr(d, "action", "未说明"),))
            st.markdown("- **理由**：%s" % (getattr(d, "action_reason", ""),))
        if tool_calls > 0:
            st.markdown("### 工具调用统计")
            st.markdown("- 总调用次数：**%s**" % (tool_calls,))
            did_supp = getattr(report, "did_supplemental_search", False)
            st.markdown("- 是否补充搜索：**%s**" % ("是" if did_supp else "否",))


def render_evidence(report: FactCheckReport) -> None:
    """渲染证据及来源等级（复古报纸卡片风格 + 证据跳转链接）。精简展示：只显示一句话概括，不显示长篇原文。"""
    st.markdown('<div class="paper-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="card-title">📚 证据及来源引用 · Evidence Citations</h2>', unsafe_allow_html=True)

    total_evidence = 0
    valid_evidence = 0
    independent_sources = set()
    for result in report.claim_results:
        for ev in result.evidence:
            total_evidence += 1
            if ev.extraction_status == "success" and ev.evidence_stance != "irrelevant":
                valid_evidence += 1
                group = ev.independence_group or ev.source_domain or ev.source_url
                independent_sources.add(group)

    if total_evidence > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📄 证据总数", value=total_evidence)
        with col2:
            pct = int(valid_evidence / total_evidence * 100) if total_evidence > 0 else 0
            st.metric(label="✅ 有效证据", value=f"{valid_evidence} ({pct}%)")
        with col3:
            st.metric(label="🔗 独立信源", value=len(independent_sources))

    # 来源等级彩色标签
    source_level_counts = {}
    for result in report.claim_results:
        for ev in result.evidence:
            if ev.extraction_status != "success": continue
            lvl = getattr(ev, "source_level", None) or "standard"
            source_level_counts[lvl] = source_level_counts.get(lvl, 0) + 1
    if source_level_counts:
        lvl_label = {"official":"🏛️ 官媒/政府","mainstream":"📰 权威主流","specialized":"🔬 专业机构","standard":"📑 普通来源","user_generated":"👤 用户生成","unreliable":"⚠️ 低可信度"}
        lvl_color = {"official":"#0044AA","mainstream":"#008833","specialized":"#882266","standard":"#555555","user_generated":"#886600","unreliable":"#BB2222"}
        tags_html = []
        for lvl, cnt in sorted(source_level_counts.items(), key=lambda x: x[1], reverse=True):
            lbl = lvl_label.get(lvl, lvl); color = lvl_color.get(lvl, "gray")
            tags_html.append(f'<span class="tag tag-secondary" style="border-color:{color};color:{color};border-width:1.5px;margin-right:6px;margin-bottom:6px;display:inline-block;">{lbl} × {cnt}</span>')
        st.markdown('<div style="margin-bottom: 16px;">' + '\n'.join(tags_html) + '</div>', unsafe_allow_html=True)

    # 收集有效证据
    valid_evidences = []
    for result in report.claim_results:
        for ev in result.evidence:
            if ev.extraction_status == "success" and ev.evidence_stance != "irrelevant":
                valid_evidences.append(ev)

    if not valid_evidences:
        st.info("尚无有效证据（可能是搜索失败或证据提取异常）。")
        st.markdown('</div>', unsafe_allow_html=True)
        return

    # 按来源等级分组展示
    grouped = {}
    for ev in valid_evidences:
        lvl = getattr(ev, "source_level", None) or "standard"
        grouped.setdefault(lvl, []).append(ev)
    lvl_order = ["official","mainstream","specialized","standard","user_generated","unreliable"]
    lvl_titles = {"official":"🏛️ 官方/政府来源 (Official)","mainstream":"📰 权威主流来源 (Mainstream)","specialized":"🔬 专业机构来源 (Specialized)","standard":"📑 普通来源 (Standard)","user_generated":"👤 用户生成来源 (User Generated)","unreliable":"⚠️ 低可信度来源 (Low Credibility)"}

    MAX_DISPLAY = 30
    display_count = 0

    for lvl in lvl_order:
        if lvl not in grouped: continue
        evs = grouped[lvl]
        st.markdown(f'<h3 class="subsection-title">{lvl_titles.get(lvl,lvl)}（{len(evs)} 条）</h3>', unsafe_allow_html=True)
        for ev in evs:
            if display_count >= MAX_DISPLAY: break
            display_count += 1

            grade_colors = {"A":"#2e7d32","B":"#66bb6a","C":"#ef6c00","D":"#757575","E":"#c62828"}
            g = ev.source_grade
            tag_color = grade_colors.get(g, "#757575")
            status_label = f"⚠️ [{ev.extraction_status}]" if ev.extraction_status != "success" else ""
            valid_url = _is_valid_http_url(ev.source_url)
            if valid_url:
                title_html = f'<a href="{ev.source_url}" target="_blank" rel="noopener" class="evidence-link">{ev.source_title}</a>'
            else:
                title_html = f"<span>{ev.source_title}</span>"
            summary_text = ev.summary if ev.summary else (ev.evidence_summary or "")[:120]
            summary_text = ' '.join((summary_text or "").split())

            # 支持/反驳标签
            if getattr(ev, "supports_or_refutes", None) == "supports" or getattr(ev, "directly_supports", None) is True:
                stance_html = '<span class="tag tag-success" style="margin-left:6px;">✅ 支持主张</span>'
            elif getattr(ev, "supports_or_refutes", None) == "refutes" or getattr(ev, "directly_supports", None) is False:
                stance_html = '<span class="tag tag-danger" style="margin-left:6px;">❌ 反驳主张</span>'
            elif getattr(ev, "supports_or_refutes", None) == "partial":
                stance_html = '<span class="tag tag-warning" style="margin-left:6px;">⚖️ 部分支持</span>'
            else:
                stance_html = ""

            # 来源等级标记
            lvl_label_map = {"official":"🏛️","mainstream":"📰","specialized":"🔬","standard":"📑","user_generated":"👤","unreliable":"⚠️"}
            lvl_icon = lvl_label_map.get(lvl, "")
            publisher = ev.publisher or ""
            source_type = ev.source_type or ""

            st.markdown(
                f"""
                <div style="padding: 12px 14px; margin-bottom: 8px; border-radius: 6px; background: #FFFFFF; border: 1px solid rgba(210,180,140,0.18); box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
                  <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px; flex-wrap: wrap;">
                    <span style="background-color: {tag_color}; color: white; padding: 2px 10px; border-radius: 10px; font-size: 12px; font-weight: 600;">{g}级</span>
                    {title_html}
                    {stance_html}
                  </div>
                  <div style="color: #666; font-size: 13px; margin-bottom: 4px;">
                    {lvl_icon} <strong>{publisher}</strong> · {source_type}{status_label}
                  </div>
                  <div style="color: #222; font-size: 13.5px; line-height: 1.65;">
                    {summary_text}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if display_count >= MAX_DISPLAY:
            break

    if total_evidence > MAX_DISPLAY:
        remaining = total_evidence - MAX_DISPLAY
        st.info(f"📌 为保证页面性能，仅展示前 {MAX_DISPLAY} 条证据，另有 {remaining} 条未展示。可在上方「📜 详细证据」页中逐主张查看。")

    st.markdown('</div>', unsafe_allow_html=True)


def render_historical_memory(report: FactCheckReport) -> None:
    """渲染历史核查记忆。"""
    matches = getattr(report, "historical_matches", []) or []
    if not matches:
        return

    with st.expander("📚 历史核查记忆", expanded=False):
        st.markdown("### 相关历史案例")
        for i, match in enumerate(matches, 1):
            st.markdown(f"**{i}. {match.get('verdict', '未知结论')}**")
            st.markdown(f"- 结论：{match.get('verdict', '')}")
            st.markdown(f"- 时间：{match.get('checked_at', '')}")
            st.caption("⚠️ 历史记录仅作参考，不代替当前公开证据")
            if i < len(matches):
                st.divider()


def render_source_assessment(report: FactCheckReport) -> None:
    """渲染来源评估。"""
    st.subheader("📊 来源评估")

    # 统计各级别来源
    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    total_evidences = 0

    for result in report.claim_results:
        for ev in result.evidence:
            total_evidences += 1
            grade = ev.source_grade
            if grade in grade_counts:
                grade_counts[grade] += 1

    if total_evidences > 0:
        # 显示来源等级分布
        cols = st.columns(4)
        grade_desc = {"A": "官方", "B": "权威", "C": "一般", "D": "自媒体"}
        for i, (grade, count) in enumerate(grade_counts.items()):
            with cols[i]:
                st.metric(f"{grade}级({grade_desc[grade]})", count)

        # 显示独立来源分析
        independent_count = sum(
            1 for result in report.claim_results
            for ev in result.evidence
            if ev.is_independent
        )
        st.caption(f"共 {total_evidences} 条证据，其中 {independent_count} 条为独立来源")


def render_mode_warning() -> None:
    """渲染模式相关的警告。"""
    st.success(
        "✅ 当前为真实核查模式（快速链路）：\n"
        "接收主张 → 单次 Tavily 搜索 → 保留 Top3 证据 → 单次 LLM 判断。"
    )


def render_download_section(report: FactCheckReport) -> None:
    """Word 导出已停用，此函数保留为空操作以避免调用出错。"""
    pass


def render_debug_panel() -> None:
    """调试面板：仅当 DEBUG=true 时显示。"""
    import os
    if os.environ.get("DEBUG", "").lower() not in ("1", "true", "yes"):
        return

    with st.sidebar:
        st.divider()
        st.subheader("🛠️ 调试面板（DEBUG）")
        if st.button("🔍 检查模型字段", use_container_width=True):
            from src.models import FactCheckReport
            annotation = FactCheckReport.model_fields["unresolved_questions"].annotation
            st.code(f"unresolved_questions annotation:\n{annotation}", language="python")

        if st.button("🧪 测试 build_failure_report", use_container_width=True):
            from src.models import build_failure_report
            try:
                report = build_failure_report("测试", "测试错误", "init")
                st.success(f"✅ 成功！类型: {type(report).__name__}")
                st.code(
                    f"unresolved_questions: {report.unresolved_questions!r}\n"
                    f"type: {type(report.unresolved_questions).__name__}\n"
                    f"overall_verdict: {report.overall_verdict!r}\n"
                    f"current_step: {report.current_step!r}\n"
                    f"credibility_score: {report.credibility_score!r}",
                    language="python",
                )
            except Exception as e:
                import traceback
                st.error(f"❌ 失败: {e}")
                st.code(traceback.format_exc(), language="python")

        if st.button("🎨 注入测试报告（正常）", use_container_width=True):
            _inject_test_report(normal=True)

        if st.button("🧯 注入测试报告（搜索失败）", use_container_width=True):
            _inject_test_report(normal=False)


def _inject_test_report(normal: bool = True) -> None:
    """注入测试报告以验证渲染逻辑。"""
    from src.models import (
        FactCheckReport, Claim, ClaimResult, Evidence, TimelineEvent,
        AgentDecision, KeyEvidenceCard,
    )
    now = datetime.now()

    if normal:
        test_claim = Claim(
            claim_id="main",
            text="网传某市因暴雨导致地铁全线停运",
            claim_type="事实",
            entities=[],
            verification_priority=1,
        )
        test_evidences = [
            Evidence(
                evidence_id="E1",
                claim_id="main",
                source_title="广州暴雨：部分地铁线路临时停运",
                source_url="https://example.com/news1",
                publisher="南方都市报",
                published_at=now,
                retrieved_at=now,
                evidence_summary="广州暴雨导致部分地铁线路临时停运，官方发布了通知。",
                summary="广州暴雨致部分线路停运，官方已发布通知。",
                source_type="权威媒体",
                source_grade="B",
                supports_or_refutes="supports",
                is_primary_source=False,
                reliability_reason="权威媒体报道",
            ),
            Evidence(
                evidence_id="E2",
                claim_id="main",
                source_title="教育局：个别学校停课一天",
                source_url="https://example.com/news2",
                publisher="广州教育局",
                published_at=now,
                retrieved_at=now,
                evidence_summary="教育局通知个别学校因暴雨停课一天，并非全市三天。",
                summary="教育局通知个别学校停课一天，范围非全市。",
                source_type="官方通报",
                source_grade="A",
                supports_or_refutes="refutes",
                is_primary_source=True,
                reliability_reason="官方发布",
            ),
        ]
        test_timeline = [
            TimelineEvent(
                event_time=datetime(2024, 6, 20, 8, 30),
                description="广州发布暴雨红色预警",
                source_url="https://example.com/news1",
            ),
            TimelineEvent(
                event_time=datetime(2024, 6, 20, 10, 0),
                description="部分地铁线路临时停运",
                source_url="https://example.com/news1",
            ),
            TimelineEvent(
                event_time=datetime(2024, 6, 20, 12, 0),
                description="教育局通知个别学校停课一天",
                source_url="https://example.com/news2",
            ),
        ]
        test_claim_result = ClaimResult(
            claim=test_claim,
            verdict="部分属实",
            confidence=0.75,
            reasoning="证据显示广州暴雨致部分区停课一日、地铁停运，但未提及全市停课三天及多人失联。",
            evidence=test_evidences,
            missing_information=None,
        )
        test_cards = [
            KeyEvidenceCard(
                card_id="K1",
                title="官方通报：教育局通知个别学校停课",
                source_url="https://example.com/news2",
                source_grade="A",
                summary="教育局仅通知个别学校停课一天，并非全市范围。",
                directly_supports=True,
            ),
        ]
        test_decision = AgentDecision(
            normalized_claim="教育局通知个别学校停课一天",
            claim_type="事实",
            sensitivity="中",
            evidence_requirement="至少 2 个独立来源",
            evidence_sufficient=True,
            missing_evidence=[],
            action="STOP",
            action_reason="已获取足够权威来源证据",
        )
        test_report = FactCheckReport(
            schema_version=REPORT_SCHEMA_VERSION,
            original_text="网传某市因暴雨导致地铁全线停运。",
            overall_verdict="部分属实",
            overall_summary="部分属实：有真实暴雨和停运，但细节不符。",
            claim_results=[test_claim_result],
            timeline=test_timeline,
            propagation_risk="中风险：涉及公共安全但来源权威",
            risk_level="中",
            risk_reason="涉及公共安全信息，需警惕夸大传播",
            risk_factors=["涉及公共安全", "细节与事实有出入"],
            unresolved_questions=["各线路恢复运营的精确时间"],
            execution_log=[],
            current_step="completed",
            completed_steps=["receive", "search", "analyze", "output"],
            skipped_steps=[],
            progress_percent=100,
            workflow_completed=True,
            workflow_error=None,
            credibility_score=80,
            recommendation="可谨慎参考",
            decision_trace=[],
            agent_decision=test_decision,
            did_supplemental_search=False,
            tool_calls_count=1,
            historical_matches=[],
            key_evidence_cards=test_cards,
            generated_at=now,
        )
        st.success("✅ 正常报告已注入！")
    else:
        test_report = build_failure_report(
            "网传某市因暴雨导致地铁全线停运。",
            "搜索服务暂时不可用",
            "search",
        )
        st.warning("🧯 搜索失败报告已注入！")

    st.session_state.report = test_report
    st.rerun()


def render_clear_button() -> None:
    """渲染清空按钮。"""
    if st.button("🗑️ 清空本次结果"):
        st.session_state.pop("report", None)
        st.session_state.pop("input_text", None)
        st.session_state.pop("running", None)
        st.session_state.pop("run_start_time", None)
        st.session_state.pop("retry_search", None)
        st.rerun()


def _build_mock_report_for_text(original_text: str) -> FactCheckReport:
    """根据用户输入文本，构造一个演示模式下的完整 FactCheckReport（零网络依赖）。"""
    from src.models import (
        FactCheckReport, Claim, ClaimResult, Evidence, TimelineEvent,
        AgentDecision, KeyEvidenceCard,
    )
    now = datetime.now()

    text_snippet = (original_text or "").strip() or "网传某市因暴雨导致地铁全线停运。"
    # 基本主张：基于输入生成
    main_claim = Claim(
        claim_id="main",
        text=text_snippet[:120],
        claim_type="事实",
        entities=[],
        verification_priority=1,
    )

    # 两条假证据
    evidences = [
        Evidence(
            evidence_id="E1",
            claim_id="main",
            source_title="演示来源 A：主流媒体报道",
            source_url="https://example.com/demo/source-a",
            publisher="演示 · 南方都市报级",
            published_at=now,
            retrieved_at=now,
            evidence_summary="演示证据 A：检索到相关报道，提及暴雨导致部分设施受影响，整体情况基本属实。",
            summary="演示证据 A：整体情况基本属实。",
            source_type="权威媒体",
            source_grade="B",
            supports_or_refutes="supports",
            is_primary_source=False,
            reliability_reason="权威媒体（演示）",
        ),
        Evidence(
            evidence_id="E2",
            claim_id="main",
            source_title="演示来源 B：官方通报",
            source_url="https://example.com/demo/source-b",
            publisher="演示 · 教育局级",
            published_at=now,
            retrieved_at=now,
            evidence_summary="演示证据 B：官方通报显示影响范围有限，与网络流传的部分细节存在出入。",
            summary="演示证据 B：部分细节被夸大。",
            source_type="官方通报",
            source_grade="A",
            supports_or_refutes="refutes",
            is_primary_source=True,
            reliability_reason="官方发布（演示）",
        ),
    ]

    timeline = [
        TimelineEvent(
            event_time=datetime(2024, 6, 20, 8, 30),
            description="发布气象预警",
            source_url="https://example.com/demo/source-a",
        ),
        TimelineEvent(
            event_time=datetime(2024, 6, 20, 10, 0),
            description="部分公共设施临时调整",
            source_url="https://example.com/demo/source-a",
        ),
        TimelineEvent(
            event_time=datetime(2024, 6, 20, 12, 0),
            description="官方回应：影响范围有限",
            source_url="https://example.com/demo/source-b",
        ),
    ]

    claim_result = ClaimResult(
        claim=main_claim,
        verdict="部分属实",
        confidence=0.72,
        reasoning="【演示数据】证据显示基础事实存在，但网络流传版本在细节和范围上有夸大，属部分属实。",
        evidence=evidences,
        missing_information=None,
    )

    cards = [
        KeyEvidenceCard(
            card_id="K1",
            title="演示关键证据 A：主流媒体报道",
            source_url="https://example.com/demo/source-a",
            source_grade="B",
            summary="演示：媒体报道确认基础事实存在。",
            grade_desc="权威主流",
            directly_supports=True,
        ),
        KeyEvidenceCard(
            card_id="K2",
            title="演示关键证据 B：官方通报",
            source_url="https://example.com/demo/source-b",
            source_grade="A",
            summary="演示：官方通报指出细节与流传版本存在出入。",
            grade_desc="官方/政府",
            directly_supports=False,
        ),
    ]

    decision = AgentDecision(
        normalized_claim=main_claim.text,
        claim_type="事实",
        sensitivity="中",
        evidence_requirement="至少 2 个独立来源",
        evidence_sufficient=True,
        missing_evidence=[],
        action="STOP",
        action_reason="已获得两条独立证据（A/B 级），证据充足",
    )

    return FactCheckReport(
        schema_version=REPORT_SCHEMA_VERSION,
        original_text=text_snippet,
        overall_verdict="部分属实",
        overall_summary="【演示报告】基础事实存在，但流传版本在范围/细节上被夸大，需谨慎参考。",
        claim_results=[claim_result],
        timeline=timeline,
        propagation_risks=[
            {"level": "medium", "description": "涉及公共安全类话题，易引发恐慌性转发", "suggestion": "建议关注官方通报的精确时间与范围"},
            {"level": "low", "description": "已有权威来源辟谣", "suggestion": None},
        ],
        risk_level="中",
        risk_reason="涉及公共安全信息，传播风险中等",
        risk_factors=["涉及公共安全", "细节与事实有出入"],
        unresolved_questions=["精确恢复时间", "具体受影响范围"],
        execution_log=[],
        current_step="completed",
        completed_steps=list(QUICK_PHASES) if 'QUICK_PHASES' in globals() else ["receive", "decompose", "search", "analyze", "output"],
        skipped_steps=[],
        progress_percent=100,
        workflow_completed=True,
        workflow_error=None,
        credibility_score=74,
        recommendation="可谨慎参考，建议以官方通报为准",
        decision_trace=[
            {"step": "主张拆解", "query": text_snippet[:30], "result": "识别 1 条主要事实主张", "action": "进入搜索阶段", "cache": "none"},
            {"step": "证据检索", "query": "演示 A/B 级来源", "result": "获得 2 条独立证据", "action": "进入判断阶段", "cache": "fallback_72h"},
            {"step": "最终判断", "result": "证据充分，停止", "reason": "已覆盖主要事实", "action": "输出报告", "cache": "none"},
        ],
        agent_decision=decision,
        did_supplemental_search=False,
        tool_calls_count=3,
        historical_matches=[],
        key_evidence_cards=cards,
        generated_at=now,
    )


def run_fact_check_async(original_text: str, progress_placeholder=None, mode: str | None = None) -> FactCheckReport:
    """核查流程的统一入口（命名为 async 只是历史习惯，实际以同步方式运行）。

    根据 mode 分发：
    - demo：_build_mock_report_for_text（零网络，< 5s）
    - llm：调用 quick_workflow.run_fact_check，但搜索走 Mock（由 USE_MOCK=true 或节点本身选择）
    - full：调用 quick_workflow.run_fact_check，走真实 Tavily + LLM
    """
    use_mode = mode or st.session_state.get("fact_check_mode", "demo")

    if progress_placeholder is not None:
        with progress_placeholder.container():
            st.caption(f"模式：{MODE_LABELS.get(use_mode, use_mode)} · 正在准备数据...")

    # === 演示模式（零网络）===
    if use_mode == "demo":
        _time.sleep(1.0)
        if progress_placeholder is not None:
            with progress_placeholder.container():
                st.caption("✅ 演示模式：生成结构化报告...")
        report = _build_mock_report_for_text(original_text)
        if progress_placeholder is not None:
            with progress_placeholder.container():
                st.caption("✅ 演示报告生成完成！")
        return report

    # === 真实模式：调用 quick_workflow.run_fact_check ===
    # 对于 use_mode = 'llm'，可以通过环境变量 USE_MOCK_SEARCH=true 强制走 Mock 搜索
    import os
    prev_mock = os.environ.get("USE_MOCK_SEARCH")
    if use_mode == "llm":
        os.environ["USE_MOCK_SEARCH"] = "true"
    try:
        check_result: CheckResult = run_fact_check(original_text)
        report = getattr(check_result, "report", None)
        if report is None:
            # 兼容旧接口返回 FactCheckReport 的情况
            if isinstance(check_result, FactCheckReport):
                report = check_result
            else:
                report = build_failure_report(
                    original_text,
                    "核查结果格式异常，未能提取报告",
                    "output",
                )
        return report
    except Exception as ex:
        import traceback
        _tb = traceback.format_exc()
        err_msg = str(ex) or "核查流程抛出异常"
        st.session_state["_last_traceback"] = _tb
        return build_failure_report(original_text, err_msg, "workflow")
    finally:
        if prev_mock is None:
            os.environ.pop("USE_MOCK_SEARCH", None)
        else:
            os.environ["USE_MOCK_SEARCH"] = prev_mock


def main() -> None:
    """Streamlit 应用主入口（左右分栏布局：左输入 / 右结果面板）。"""
    render_header()
    # 读取侧边栏选择的模式（默认 demo，无需联网即可体验 UI）
    selected_mode = render_sidebar()

    # ===== Schema 版本检查：旧版本报告自动清除 =====
    stored_version = st.session_state.get("_schema_version", 0)
    if stored_version != REPORT_SCHEMA_VERSION:
        if st.session_state.get("report") is not None:
            st.session_state.report = None
        if st.session_state.get("error_message"):
            st.session_state.error_message = None
        st.session_state["_schema_version"] = REPORT_SCHEMA_VERSION

    # 模式切换时，如果之前是 full 失败了，切到 demo 后要能重跑
    mode_changed = st.session_state.get("_last_mode") != selected_mode
    if mode_changed:
        st.session_state["_last_mode"] = selected_mode
        # 保留 input_text，但重置 report 和 running，让用户点一次按钮即可跑
        st.session_state.report = None
        st.session_state.running = False

    defaults = {
        "report": None,
        "input_text": "",
        "running": False,
        "run_start_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    render_debug_panel()

    # ===== 左右两栏布局：左输入 / 右结果 =====
    col_left, col_right = st.columns([4.5, 7.5], gap="large")

    # ============== 左栏：输入区 + 执行状态 ==============
    with col_left:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)

        text = render_input()

        # 检查是否为重新搜索（搜索失败后重试）
        is_retry = st.session_state.get("retry_search", False)

        # 新提交或重新搜索：开始一次核查
        should_run = (text and text != st.session_state.input_text) or is_retry

        if should_run:
            if text and text != st.session_state.input_text:
                st.session_state.input_text = text
            st.session_state.report = None
            st.session_state.running = True
            st.session_state.run_start_time = _time.time()
            st.session_state.error_message = None
            if is_retry:
                st.session_state.retry_search = False

            # 立即显示空占位（触发 UI 刷新）
            with st.spinner("Agent 正在溯源核查，大约需要 60 秒..."):
                try:
                    input_for_run = st.session_state.input_text or text or ""
                    progress_placeholder = st.container()
                    report = run_fact_check_async(input_for_run, progress_placeholder, mode=selected_mode)
                    st.session_state.report = report
                except Exception as ex:
                    import traceback
                    tb_str = traceback.format_exc()
                    st.error(f"[核查执行异常]\n{tb_str}")
                    st.session_state.report = build_failure_report(
                        st.session_state.input_text or text,
                        "核查服务暂时不可用，请稍后重试",
                        "init",
                    )
                    st.session_state.error_message = "核查服务暂时不可用，请稍后重试"
                finally:
                    st.session_state.running = False

        # 正在运行中不渲染报告（等待本次运行完成）
        if st.session_state.running:
            st.info("⏳ 核查正在进行中，请稍候...（普通页面交互不会重复发起核查）")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        report = st.session_state.report
        if report is not None:
            elapsed = 0.0
            if st.session_state.run_start_time:
                elapsed = _time.time() - st.session_state.run_start_time
            render_execution_state(report, elapsed_seconds=elapsed)
            render_mode_warning()

        st.markdown('</div>', unsafe_allow_html=True)

    # ============== 右栏：溯源结果面板 ==============
    with col_right:
        report = st.session_state.report
        if report is None:
            # 无报告时：占位提示
            st.markdown('<div class="paper-card result-panel">', unsafe_allow_html=True)
            st.markdown('<h2 class="card-title">📰 溯源核查结果 · Verification Report</h2>', unsafe_allow_html=True)
            st.markdown(
                """
                <div style="padding: 40px 20px; text-align: center; color: #64748B;">
                  <div style="font-size: 72px; margin-bottom: 16px;">🗞️</div>
                  <h3 style="color: #334155; margin-bottom: 8px;">还没有核查报告</h3>
                  <p style="font-size: 15px; line-height: 1.7; color: #64748B;">
                    在左侧输入框中粘贴新闻文本、微博、文章链接等内容，
                    <br />
                    点击「🔍 开始溯源核查」按钮，Agent 将在 60 秒内为您生成完整的溯源报告。
                  </p>
                  <div style="margin-top: 24px; display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; text-align: left;">
                    <div style="padding: 14px; background: rgba(37,99,235,0.04); border-radius: 6px; border: 1px solid rgba(37,99,235,0.08);">
                      <strong style="color: #1E3A8A; display:block; margin-bottom:4px;">🔎 智能拆解</strong>
                      <span style="font-size: 13px; color: #475569;">自动识别可核查主张与实体</span>
                    </div>
                    <div style="padding: 14px; background: rgba(16,185,129,0.04); border-radius: 6px; border: 1px solid rgba(16,185,129,0.08);">
                      <strong style="color: #065F46; display:block; margin-bottom:4px;">📚 多源检索</strong>
                      <span style="font-size: 13px; color: #475569;">Tavily 检索官媒/权威/专业来源</span>
                    </div>
                    <div style="padding: 14px; background: rgba(255,152,0,0.04); border-radius: 6px; border: 1px solid rgba(255,152,0,0.08);">
                      <strong style="color: #7C2D12; display:block; margin-bottom:4px;">⏰ 时间线还原</strong>
                      <span style="font-size: 13px; color: #475569;">串联事件先后并标注证据引用</span>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            # 有报告时：渲染分模块结果
            st.markdown('<div class="result-panel">', unsafe_allow_html=True)

            # 顶部：报告标题栏 + 导出按钮
            col_title, col_btn1, col_btn2 = st.columns([5, 1.3, 1.3])
            with col_title:
                st.markdown('<h2 class="report-panel-title">📰 溯源核查报告 · Verification Result</h2>', unsafe_allow_html=True)
                st.caption(
                    f"生成时间：{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} ｜ "
                    f"模式：专业核查 ｜ "
                    f"输入：{(st.session_state.input_text or '')[:40]}..."
                )
            with col_btn1:
                # 导出 Markdown 报告按钮
                try:
                    from io import StringIO
                    md_buf = StringIO()
                    md_buf.write("# 溯真 · 新闻溯源核查报告\n\n")
                    md_buf.write(f"**生成时间**：{report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    md_buf.write(f"**总体结论**：{getattr(report, 'overall_verdict', '未判定')}\n\n")
                    md_buf.write(f"**结论说明**：{getattr(report, 'overall_summary', '')}\n\n")
                    cred = getattr(report, 'credibility_score', None)
                    if cred is not None:
                        md_buf.write(f"**可信度评分**：{cred}%\n\n")
                    rec = getattr(report, 'recommendation', '')
                    if rec:
                        md_buf.write(f"**建议**：{rec}\n\n")
                    md_buf.write("---\n\n## 1. 主张拆解\n\n")
                    for i, cr in enumerate((report.claim_results or []), start=1):
                        md_buf.write(f"### 主张 {i}：{cr.claim.text}\n\n")
                        md_buf.write(f"- 判断：{cr.verdict}\n")
                        md_buf.write(f"- 理由：{cr.reasoning or cr.rationale or ''}\n\n")
                        for j, ev in enumerate(cr.evidence, start=1):
                            ev_summ = ev.summary or (ev.evidence_summary[:80] if ev.evidence_summary else "")
                            md_buf.write(f"  - 证据{j}：[{ev.source_title}]({ev.source_url}) 「{ev_summ}」\n")
                        md_buf.write("\n")
                    md_buf.write("## 2. 时间线\n\n")
                    for ev in (report.timeline or []):
                        d = getattr(ev, 'date', None) or getattr(ev, 'event_time', '')
                        desc = getattr(ev, 'description', None) or str(ev)
                        md_buf.write(f"- {d}：{desc}\n")
                    md_report_bytes = md_buf.getvalue().encode('utf-8')
                    st.download_button(
                        "📄 导出 Markdown",
                        data=md_report_bytes,
                        file_name=f"溯源核查报告_{report.generated_at.strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        key="export_md_btn",
                    )
                except Exception:
                    st.button("📄 导出报告", use_container_width=True, disabled=True)
            with col_btn2:
                # 重置按钮
                if st.button("🗑️ 清空结果", use_container_width=True, key="col_clear_btn"):
                    st.session_state.report = None
                    st.session_state.input_text = ""
                    st.session_state.error_message = None
                    st.rerun()

            st.markdown('<div class="section-hr"></div>', unsafe_allow_html=True)

            # 判断状态
            failed = bool(report.workflow_error) or report.current_step == "failed"
            no_evidence = report.overall_verdict == "暂无法核查"

            # 1. 总体结论（最高优先级展示）
            render_overall_verdict(report)

            if no_evidence:
                st.warning(
                    "⚠️ **当前未取得公开证据，请稍后重试**。"
                    "搜索服务暂不可用，您可以点击下方按钮重新搜索。"
                )
                if st.button("🔄 重新搜索", use_container_width=True):
                    st.session_state.retry_search = True
                    st.rerun()

            elif failed and report.workflow_error == "search":
                st.warning(
                    "⚠️ **搜索失败**：搜索服务连接临时中断。您可以尝试重新搜索。"
                )
                if st.button("🔄 重新搜索", use_container_width=True):
                    st.session_state.retry_search = True
                    st.rerun()

            if not failed:
                # 2. 关键证据卡片
                render_key_evidence_cards(report)

                # 3. 使用标签页展示详细内容
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📋 结构化核查表",
                    "📜 详细证据",
                    "⏰ 事件时间线",
                    "📊 来源评估",
                    "🤖 Agent 决策轨迹",
                    "📚 历史核查"
                ])

                with tab1:
                    render_verification_table(report)

                with tab2:
                    render_claim_results(report)
                    render_evidence(report)

                with tab3:
                    render_timeline(report)

                with tab4:
                    render_source_assessment(report)

                with tab5:
                    render_agent_decision_trace(report)

                with tab6:
                    render_historical_memory(report)

                # 4. 风险和待核实问题
                render_risk_and_questions(report)

                # 5. 最终总结表格（符合老师要求的 7+1 列表格）
                st.markdown('<div class="section-hr"></div>', unsafe_allow_html=True)
                render_final_summary_table(report)

            render_clear_button()

            st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
