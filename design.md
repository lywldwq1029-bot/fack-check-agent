# 溯真设计文档

## 一、产品定位

“溯真”是一款面向突发新闻的动态事实核查 Agent。它的核心目标不是对整篇新闻给出一个简单的“真/假”标签，而是：

- 把新闻拆解成多个可独立核查的主张（claim）；
- 为每个主张搜索证据、评价来源、比较冲突信息；
- 还原事件时间线，标注信息缺口；
- 最终输出一份结构化、带证据和不确定性的核查报告。

本阶段为课程作品的 MVP（最小可行产品），使用模拟数据跑通全流程，并为后续接入真实大模型和搜索引擎预留清晰接口。

## 二、与普通真假判断工具的区别

| 维度 | 普通真假判断工具 | 溯真 |
|------|------------------|------|
| 输出粒度 | 整篇新闻：真 / 假 | 每个主张独立结论 |
| 结论类型 | 二元或三元 | 七种精细结论（已证实、基本属实、部分属实、证据不足、存在误导、已证伪、仍在发展） |
| 证据展示 | 通常不展示 | 每条结论均附证据与来源等级 |
| 时间线 | 通常无 | 还原关键事件节点 |
| 不确定性 | 忽略 | 显式标注缺失信息和待核实问题 |
| 记忆能力 | 通常无 | 保存到 SQLite，支持后续检索 |

## 三、工作流设计

```
新闻输入
  │
  ▼
主张拆解（decompose）
  │
  ▼
生成核查计划（plan）
  │
  ▼
搜索证据（search）
  │
  ▼
评价来源 / 交叉验证（evaluate）
  │
  ▼
生成结论与报告（report）
  │
  ▼
保存核查记忆（memory）
  │
  ▼
输出报告
```

### 各节点职责

1. **decompose**：将原始文本切分为结构化主张，标注实体、时间、地点、优先级。
2. **plan**：为每个主张生成检索关键词、优先来源类型和难度评估。
3. **search**：调用搜索工具，按关键词为每个主张收集证据。
4. **evaluate**：对证据进行可信度评级，比较多源信息，生成单条主张结论。
5. **report**：汇总所有主张结论，生成总体结论、时间线、传播风险和待核实问题。
6. **memory**：将报告持久化到 SQLite。

## 四、规划如何体现

规划能力主要体现在 `plan` 节点：

- 为每个主张输出 `keywords`、`preferred_sources`、`difficulty`；
- 这些计划被 `search` 节点直接使用，决定检索方向；
- 后续接入 LLM 时，可用 `PLAN_SYSTEM_PROMPT` 让模型动态生成更精细的计划。

## 五、记忆如何体现

记忆能力通过 `src/memory/repository.py` 实现：

- 使用 SQLite 持久化 `fact_check_reports` 表；
- 保存字段包括原始文本、总体结论、摘要、主张数量、完整报告 JSON、创建时间；
- 当前提供 `save_report` 和 `list_reports` 两个基础接口；
- 后续可扩展：按文本相似度检索历史报告、证据去重、用户反馈记录。

## 六、工具调用如何体现

工具调用通过 `src/tools/search_tool.py` 封装：

- `search_web(query, claim_id, max_results)` 是统一接口；
- 当前为模拟实现，根据关键词返回演示证据；
- 真实接入时，在 `search_web` 中判断 `settings.USE_MOCK`，调用 `_real_search` 实现；
- 可接入 Serper、Tavily、Bing 等搜索引擎 API，返回结果统一封装为 `Evidence` 列表。

## 七、后续接入真实大模型的位置

| 节点 | 当前实现 | 替换方式 |
|------|----------|----------|
| `src/nodes/decompose.py` | 关键词规则 | 调用 LLM + `DECOMPOSE_SYSTEM_PROMPT`，解析为 `list[Claim]` |
| `src/nodes/plan.py` | 关键词规则 | 调用 LLM + `PLAN_SYSTEM_PROMPT`，生成动态计划 |
| `src/nodes/evaluate.py` | 关键词规则 | 调用 LLM + `EVALUATE_SYSTEM_PROMPT`，输入证据和主张，输出 `ClaimResult` |
| `src/nodes/report.py` | 规则汇总 | 调用 LLM + `REPORT_SYSTEM_PROMPT`，输入所有 `ClaimResult`，输出报告文本与时间线 |

配置入口位于 `src/config.py`，已从 `.env` 读取 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`LLM_MODEL`。

## 八、后续接入真实搜索工具的位置

- 配置文件：`src/config.py` 已预留 `SERPER_API_KEY`、`TAVILY_API_KEY`。
- 实现位置：`src/tools/search_tool.py` 中的 `_real_search` 函数。
- 切换方式：将 `settings.USE_MOCK` 设为 `false`，并在 `search_web` 中调用 `_real_search`。

## 九、数据模型说明

核心模型定义在 `src/models.py`：

- `Claim`：可核查主张
- `Evidence`：证据
- `ClaimResult`：单条主张核查结果
- `TimelineEvent`：时间线事件
- `FactCheckReport`：完整报告
- `AgentState`：工作流状态与执行日志

所有 verdict 字段均使用 `Literal` 限定为七种标准结论，确保数据一致性。

## 十、本阶段限制

- 不调用真实大模型和真实搜索服务；
- 证据和结论为演示数据，不具备真实新闻核查效力；
- 主张拆解依赖关键词匹配，无法处理复杂句式；
- 记忆库仅支持保存和列表查询，暂不支持语义检索。
