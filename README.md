# 📰 溯真 · 新闻溯源核查 Agent

<p align="center">
  <i>现代明亮 UI × 复古报纸纸质感 · AI 驱动的新闻事实核查系统</i>
  <br>
  <a href="#特性"><b>特性</b></a> ·
  <a href="#快速开始"><b>快速开始</b></a> ·
  <a href="#核查模式"><b>核查模式</b></a> ·
  <a href="#配置说明"><b>配置</b></a> ·
  <a href="#目录结构"><b>目录</b></a> ·
  <a href="#贡献指南"><b>贡献</b></a> ·
  <a href="#许可证"><b>许可证</b></a>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Pydantic" src="https://img.shields.io/badge/Model-Pydantic-E92063?logo=pydantic&logoColor=white">
  <img alt="License" src="https://img.shields.io/github/license/lywldwq1029-bot/fact-check-agent?color=blue">
</p>

---

## 为什么要造「溯真」？

面对社交媒体上泛滥的突发传闻、半真半假的拼接新闻、未经证实的截图转发，**人工核查太慢、只看单一来源容易被带偏**。

「溯真」把新闻事实核查的专业流程做成了一个开箱即用的 Web Agent：
输入新闻文本或链接，它会**自主拆解主张 → 规划检索 → 调用搜索引擎做多源取证 → 交叉验证正反证据 → 输出带时间线和可信度评分的结构化报告**。

整套界面延续"老式报纸"的纸质感与分栏排版，兼顾信息密度与阅读舒适度；
为了让新同学零门槛体验，还内置了**演示模式（零网络、无需 API 密钥）**。

---

## 特性

### Agent 三大核心能力

| 能力 | 实现方式 | 说明 |
| :-- | :-- | :-- |
| 🧭 **规划** | `quick_workflow.py` + 节点编排 | 拆解主张 → 规划检索 → 评估充分性 → 决定补充搜索/停止 → 输出报告 |
| 🧠 **记忆** | SQLite 记忆库 + 会话级缓存 + 搜索结果 72h 缓存 | 相同/相似主张命中历史结果直接复用，避免重复消耗 API 额度 |
| 🛠️ **工具调用** | `search_tool.py`（Tavily）/ MockSearchProvider | 动态生成搜索 query、指定话题、超时与重试、结果聚合与去重 |

### UI 亮点

- 🪄 **三模式切换**（侧边栏下拉）：演示模式 / 真实 LLM 拆解 / 完整真实核查
- 📜 **复古报纸风格**：暖米黄纸张肌理 + 印刷细线分割 + 衬线标题 × 无衬线正文
- 🎯 **左右分栏 Dashboard**：左（输入卡片 + 核查按钮）| 右（溯源结果面板）
- 📊 **结果模块齐全**：核查结论徽章、可信度进度条、主张拆解表格、时间线溯源链条、
  关键证据卡片（带来源等级 A/B/C/D）、证据引用（支持/反驳）、风险标记、最终核查总结表
- 📄 **一键导出**：Markdown 报告 / CSV 核查表

### 工程亮点

- 🔌 **OpenAI 兼容接口**：支持 DeepSeek、通义千问、Moonshot 等所有兼容格式
- 🛟 **故障安全**：LLM/Tavily 超时或异常时自动降级为"暂无法核查"，不会挂死页面
- 🧪 **单元测试齐全**：`tests/` 下覆盖模型序列化、主张拆解、证据评估、工作流、导出器、多模式运行等
- 🔒 **零密钥硬编码**：所有敏感信息从 `.env` 读取；`.gitignore` 已完整屏蔽数据库、缓存与临时脚本

---

## 快速开始

> ✅ **推荐新手直接用「演示模式」**，不用申请任何 Key，5 秒内看到完整 UI 效果。

### 0. 环境要求

- **Python**：3.10 / 3.11 / 3.12
- **OS**：Windows / macOS / Linux
- **浏览器**：Chrome / Edge（用于访问 Streamlit 本地服务）

### 1. 克隆与安装

```bash
git clone https://github.com/lywldwq1029-bot/fact-check-agent.git
cd fact-check-agent

# （推荐）创建虚拟环境
python -m venv .venv
# Windows:   .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. 配置环境变量（真实模式才需要）

```bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env
```

编辑 `.env`，至少填这几项：

```env
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

TAVILY_API_KEY=tvly-xxxx            # 完整真实核查模式必填
```

> 不想填？没关系，下一步启动后在侧边栏选 **「演示模式（零网络）」**。

### 3. 启动 Web 应用

```bash
streamlit run app.py --server.port 8501
```

浏览器会自动打开 http://localhost:8501 — 完成，开始核查吧 🎉

---

## 核查模式

侧边栏下拉切换，三种模式覆盖从"零配置体验"到"真实生产级"的全部场景：

| 模式 | LLM 密钥 | Tavily 密钥 | 典型耗时 | 用途 |
| :-- | :--: | :--: | :--: | :-- |
| **演示模式（零网络）** | ❌ | ❌ | < 5 s | 体验 UI、验证导出、给别人演示 |
| **真实 LLM 拆解** | ✅ | ❌ | 10-20 s | 验证 LLM 主张拆解与判断逻辑（搜索走 Mock） |
| **完整真实核查（推荐）** | ✅ | ✅ | 30-60 s | 对真实新闻做多源取证与事实判断 |

---

## 配置说明

`.env.example` 里列出了全部可用配置，常用项速查：

| 环境变量 | 默认值 | 说明 |
| :-- | :--: | :-- |
| `LLM_API_KEY` | — | LLM 提供商给你的 API Key |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | 兼容接口地址（国产模型需要改） |
| `LLM_MODEL` | — | 模型名，如 `deepseek-chat` / `gpt-4o-mini` / `qwen-plus` |
| `LLM_TIMEOUT` | `60` | 单次 LLM 请求超时秒数 |
| `TAVILY_API_KEY` | — | [Tavily](https://tavily.com) 控制台获取 |
| `TAVILY_TIMEOUT` | `25` | 单次搜索超时秒数（国内网络建议 ≥ 20） |
| `SEARCH_MAX_QUERIES_PER_CLAIM` | `2` | 每条主张最多生成几条搜索语句 |
| `SEARCH_MAX_RESULTS_PER_QUERY` | `5` | 每条语句最多取多少条网页 |
| `SEARCH_MAX_TOTAL_RESULTS` | `20` | 单份报告最多处理的网页数上限 |
| `MEMORY_DB_PATH` | `data/memory.db` | SQLite 历史记忆库路径 |

---

## 目录结构

```text
fact-check-agent/
├─ app.py                     # Streamlit UI 入口（左右分栏、三种模式、结果可视化）
├─ requirements.txt           # Python 依赖清单
├─ LICENSE                    # MIT 协议
├─ README.md                  # 本文档
├─ .env.example               # 环境变量模板（复制为 .env 后填密钥）
├─ .gitignore                 # Git 忽略规则
├─ data/                      # SQLite 记忆库目录（.gitkeep 占位，.db 被忽略）
├─ docs/design.md             # 设计文档
├─ src/
│  ├─ config.py               # 配置中心（dotenv + dataclass + 预检工具）
│  ├─ models.py               # Pydantic 数据模型（FactCheckReport 等全套结构）
│  ├─ quick_workflow.py       # 快速核查工作流（ReAct 规划+60秒总时长）
│  ├─ workflow.py             # 标准工作流（节点编排版）
│  ├─ claim_decomposer.py     # 长新闻 → 独立事实主张
│  ├─ session_cache.py        # 会话级短期缓存
│  ├─ search_cache.py         # 搜索结果 72h 磁盘缓存
│  ├─ memory_store.py         # SQLite 历史记忆库
│  ├─ llm/client.py           # 兼容 OpenAI SDK 的 LLM 客户端（流式/重试/超时）
│  ├─ tools/search_tool.py    # TavilySearchProvider + MockSearchProvider
│  ├─ tools/extractor.py      # URL 文本抽取工具
│  ├─ nodes/                  # 标准工作流各节点：decompose/search/evaluate/...
│  ├─ prompts/system_prompts.py
│  └─ exporters/docx_exporter.py
└─ tests/                     # pytest 单元测试集（≥ 12 个套件）
```

---

## 常见问题

<details>
<summary><b>Q1：国内访问 Tavily 经常"搜索服务连接临时中断"？</b></summary>

1. 把 `TAVILY_TIMEOUT=30`（或更大）写入 `.env`
2. 使用代理后在代理开启的终端里启动 Streamlit
3. 先用「演示模式」和「真实 LLM 拆解」模式把整套 UI 跑通
</details>

<details>
<summary><b>Q2：切到完整真实核查后报错"暂无法核查"怎么办？</b></summary>

右侧会显示失败原因。常见：
- **LLM 请求超时**：把 `LLM_TIMEOUT` 改大或换一个更低延迟的模型
- **密钥无效**：在侧边栏点「测试模型连接 / 测试搜索连接」排查
- **搜索额度耗尽**：登录 Tavily 控制台查剩余额度

</details>

<details>
<summary><b>Q3：支持本地开源大模型吗？</b></summary>

支持。只要提供 `OpenAI 兼容` 接口即可（如 Ollama 的 `http://localhost:11434/v1`、LM Studio、vLLM 等），
填入 `LLM_BASE_URL` 与 `LLM_MODEL` 即可。

</details>

<details>
<summary><b>Q4：跑 pytest 失败？</b></summary>

部分测试用例会尝试连接 LLM / Tavily，若未配置 `.env` 会自动跳过 Mock 之外的用例。
若要跑全量，请先配置 `.env` 再执行：`pytest -q`。

</details>

---

## 贡献指南

欢迎 PR！无论是功能、文档、测试、UI 细节都非常感谢。

1. Fork 本仓库，创建你的特性分支 `git checkout -b feature/awesome-xxx`
2. 完成改动后 `pytest -q` 跑一轮测试
3. 提交前确认 `app.py` 能正常启动（`streamlit run app.py` 无 SyntaxError 即可）
4. 提交 PR，并附一张截图或简要说明改动点

> ⚠️ 提交前**不要**把你的 `.env`、`data/*.db`、`__pycache__`、临时调试脚本（`_*.py`/`_*.log`/`*.bat`）加进 commit —
> 这些都已在 `.gitignore` 中屏蔽，但手动 `git add -f` 仍会绕过，请注意。

---

## 许可证

[MIT © 2026 lywldwq1029-bot](./LICENSE)

**免责声明**：本项目提供的核查结论和可信度评分基于公开搜索结果与 LLM 推断自动生成，**仅供参考，不代表官方观点或任何法律层面的事实认定**。请以权威官方渠道发布的信息为准，谨慎二次传播。
