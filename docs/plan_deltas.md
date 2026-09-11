# 计划与实现偏差登记

本文件记录《通用电路自动设计系统重构实施计划》（`计划/通用电路自动设计系统重构实施计划.md`）
与 `circuit_ai/` 实际实现之间的偏差，用于后续逐项对齐。

约定：

- `计划:N` 指计划文档行号；代码位置一律为 `文件:行号`。
- 每消除一项，在该条后追加 `已解决（日期 / 变更说明）`，不删除原始记录。
- 本文件只登记偏差与事实，不代替架构决策。是否把契约块改成实现现状、是否补齐缺失字段，
  属于架构决策，需用户拍板。
- 建档时间：2026-09-11（基于该日的只读侦察；当时 `pint` 缺失，47 个测试模块无法收集）。

---

## 1. 已勾选但存在残留（R1–R4）

| ID | 计划声明 | 实现事实 | 影响判定 |
|---|---|---|---|
| **R1** | 阶段 1 `计划:1552`“主流程通过 resolver 获取执行计划”；退出标准 `计划:1557` | 电源链走 resolver（`circuit_ai/pipeline.py:139-160` 三次 `resolver.resolve`），但 AC 链在 `circuit_ai/synthesis.py:987` 直接 `ConstraintEvaluator().evaluate(...)`，未解析 `constraints.declarative.v1`（该能力注册于 `circuit_ai/capabilities/builtins.py:160`） | AC 侧旁路削弱“主流程统一 resolver”与 §19.1“新增 metric：ConstraintEvaluator 代码变更数必须为 0”（`计划:1865`） |
| **R2** | 阶段 3 `计划:1598` ngspice 从事后 verifier 改造成可调用 backend | `verification.ngspice.ac` 注册于 `circuit_ai/capabilities/builtins.py:289`，但全仓 grep `VERIFICATION_BACKEND` 仅命中该注册与枚举定义（`circuit_ai/capabilities/contracts.py:15,25`），**无任何 resolver 调用点**；CLI 仍直接 `NgspiceVerifier()`（`circuit_ai/cli.py:13,131`） | 冗余注册；§19.1“新增 backend：orchestrator 变更数为 0”（`计划:1864`）未获验证 |
| **R3** | §3.4 `计划:172-184`“能力分派不以拓扑族为唯一键”；§19.1 `计划:1866`“family 名称变化不得影响后端能力解析” | `CapabilityRequest.family`（`circuit_ai/capabilities/contracts.py:150`）参与过滤（`circuit_ai/capabilities/registry.py:125-126`）；`circuit_ai/pipeline.py:404` 传入 `candidate.family`；`_POWER_FAMILIES` 定义于 `circuit_ai/capabilities/builtins.py:24` 并写入各 target（:117,133,166,192,210） | 与 §3.4/§19.1 直接冲突，family 仍是实际执行身份键 |
| **R4** | §5.7 `计划:429` 把 `evidence_hash` 列为 `SimulationResult` 字段 | 实现为 property：`circuit_ai/simulation/contracts.py:555`（序列化时写出，见 :582；反序列化校验见 :609-611） | 等价实现，非缺陷；仅登记契约表达差异 |

---

## 2. 文档/实现不一致（D1–D6）

| ID | 文档声明 | 实现事实 | 影响判定 |
|---|---|---|---|
| **D1** | §5.4 `计划:354-377`：`CircuitGraph` 字段为 `schema_version, graph_id, ports, nets, components, variables, relations, layout_hints, provenance` | `circuit_ai/graph/model.py:422-442` 实为 `graph_id, name, domains, nets, components, ports, variables, description, family, preferred_solver_ids, provenance, extensions, schema, schema_version`；**无 `relations`，无 `layout_hints`**（grep `layout_hint` 在 `circuit_ai/graph/` 无命中） | 阶段 6 的 `计划:1685`（LayoutHint）没有落脚字段；§11.2 的布局提示无处存放 |
| **D2** | §5.2 `计划:310-317`：`ModelRef` 含 `model_version`、`source: Literal["builtin","spice","vendor","user"]`、`source_uri`、`checksum` | `circuit_ai/graph/model.py:75-80` 实为 `model_id, kind, terminals, version, source`；**无 `source_uri`、无 `checksum`** | 阶段 8 的 `计划:1728`（模型 manifest/checksum/许可证）与 §12.2 来源证据要求在数据结构上无处存放 |
| **D3** | §5.9 `计划:485-492` `GraphAction.action_kind` 12 值词表：`add_module, add_component, connect, disconnect, create_net, merge_net, bind_port, set_model, expose_variable, terminate_port, replace_subgraph, finish_graph` | `circuit_ai/graph/actions.py:36-110` 实为 9 个动作类型：`OpenTerminal, AddComponent, AddFragment, ConnectTerminal, SplitNet, AddFeedback, TerminatePort, ReplaceModel, RemoveComponent`（联合类型 :116，执行器 `apply_graph_action` :128）；缺失 `add_module, create_net, merge_net, disconnect, bind_port, set_model, expose_variable, replace_subgraph, finish_graph` | 阶段 7 的 `计划:1704`（定义统一图构造动作）与 §10.4 词表未对齐；计划声称“共享同一动作词汇”，但词汇本身与文档不同 |
| **D4** | §5.9 `计划:504-511` 类名 `PartialCircuitState`（字段含 `open_requirements`、`legal_action_mask`、`depth`） | 全仓 grep `PartialCircuitState`/`partial_state` **0 命中**；等价物为 `circuit_ai/graph/search.py:80` `GraphSearchState`，合法动作掩码由生成器动态计算（`circuit_ai/graph/search.py:275`、:339 `actions()`），非字段 | §15 目标结构 `graph/partial_state.py` 不存在；阶段 7 的 `计划:1705` 语义等价但命名与字段不符 |
| **D5** | §8.2 `计划:690-701` 第一批 MetricProvider 共 9 类，含 `impedance_metrics`、`stability_metrics`、`noise_metrics`、`thermal_metrics` | `circuit_ai/metrics/providers.py` 实为 7 个：`simulation.scalar`(:78)、`dc_port_metrics`(:110)、`frequency_response`(:164)、`transient_ripple`(:171)、`device_stress`(:179)、`graph.resource`(:201)、`graph.parameter`(:284)；**阻抗/稳定性/噪声/热 4 类缺失** | 阶段 4 的 `计划:1620` 只声明 5 类（与实现一致），故不构成虚报；但 §8.2 作为架构承诺未兑现，与 `测试集/清单.json` 中 39 个 `unsupported` 期望（含 noise/thermal）一致 |
| **D6** | §21 `计划:1946-1953` “需要 ADR 决定”8 项未决 | 其中 5 项已在代码中事实上确定但无 ADR：① 单位库 = `pint`（`circuit_ai/units.py:12`，`端口描述语言/functions.py:16` 亦直接 import）；② 图实现 = 纯 dataclass，grep `networkx` 在 `circuit_ai/` 无命中；③ schema 校验 = 手写（`circuit_ai/graph/model.py:422-447` + `circuit_ai/graph/validation.py`），Pydantic 仅用于 Web 层（`circuit_ai/webapp.py`）；④ replay 存储 = JSONL（`circuit_ai/replay.py:408` `merge_jsonl`）；⑤ 外部主后端 = ngspice（Xyce 无代码） | 违反 `计划:19`“若实现需要违反第 3 章架构原则，必须先新增 ADR”；① 与 ② 已由 ADR 006/007 追认 |

---

## 3. 文档自相矛盾（C1–C7）

| ID | 矛盾 | 位置 |
|---|---|---|
| **C1** | §5.4 把 `relations`、`layout_hints` 列为 `CircuitGraph` 必备字段，§11.2/阶段 6 又把 LayoutHint 当作未实现待办 | `计划:354-377`（必备）vs `计划:953-961`、`计划:1685`（待办）；实现 `circuit_ai/graph/model.py:422-442` 两字段均无 |
| **C2** | §5.7 要求 `status` 区分 `failed`（数值失败/不收敛）与其它状态，但 `SimulationStatus` 无 `nonconverged`，阶段 0 又把它列为待确定的独立错误类 | `计划:432-441` vs `计划:1531`；实现 `circuit_ai/simulation/contracts.py:26-31` 仅五态，`circuit_ai/simulation/ngspice_backend.py:549` 以裸字符串返回 `nonconverged` |
| **C3** | §16.2 声明阶段 7A/7B/7C 原型“必须复用阶段 2、3、4 的最小版本和迁移器，不能建立第二套事实模型”，但阶段 2 的 `计划:1576` 已勾选“核心语义从 metadata 提升为正式字段”，主流程却仍以 metadata 做能力分派 | `计划:1495-1503`、`计划:1576` vs `circuit_ai/pipeline.py:404`（`candidate.metadata.get("solver")`） |
| **C4** | 章节编号断裂：`## 网页端全流程验收与错误报告` 未纳入 §1–§23 编号体系，但正文又把它当作强制验收要求 | `计划:2007`（无编号新章，紧接 §23 结尾 `计划:2006`） |
| **C5** | §19.1“family 名称变化不得影响后端能力解析”与实现的 family 过滤冲突 | `计划:1866` vs `circuit_ai/capabilities/contracts.py:150`、`circuit_ai/capabilities/registry.py:125-126`、`circuit_ai/capabilities/builtins.py:24` |
| **C6** | §7.3 规则 4 要求后端差异超门槛时标记 `model_disagreement` 并进入诊断队列；实现中只有 surrogate 集成分歧 | `计划:652-653` vs `grep -rn "model_disagreement" circuit_ai/` 无命中，仅 `circuit_ai/surrogate.py:314` 的 ensemble disagreement |
| **C7** | 文档声明路径为旧仓库名，与实际目录不符 | `计划:4` `适用仓库：E:/Pioneer/ai电路` vs 实际 `E:/Pioneer/自动电路设计研究` |

---

## 4. 进度事实被低估的区域（已回填）

§18.2 的 15 项测试复选框原文为 0/15，但经核查 10 项已有镜像测试，已在计划文档
`计划:1830-1859` 回填为 `[x]` 并附 `文件:行号` 证据。仍为 `[ ]` 的 5 项
（适配前后 golden 等价、surrogate 越域回退、隔离域非法短接拒绝、多保真 disagreement、
replay 按 spec 的训练/验证隔离）经核查确实缺少对应测试，不是漏勾。

计划头部 `计划:5` 的“状态：待实施”与阶段 0–4 已实施的事实不符，已更新为
“阶段 0 部分完成（2/7）、阶段 1–4 已完成、阶段 5 进行中、阶段 6–10 未开始”。

---

## 5. 待用户拍板事项

1. **D1 的字段补齐**：是否把 `relations` 与 `layout_hints` 补进 `CircuitGraph`。
   涉及 canonical hash，属破坏性 schema 变更，需要迁移器（`计划:280`）。
2. **D2 的来源证据字段**：是否把 `source_uri`/`checksum` 补进 `ModelRef`，以支撑
   阶段 8 的模型 manifest 与许可证记录（`计划:1728`、§12.2）。
3. **D3/D4 的契约命名**：是否把 §5.9 的动作词表与 `PartialCircuitState` 命名对齐到
   实现（`GraphAction` 9 类、`GraphSearchState`），或反向补实现。
4. **R3 的 family 键**：是否解除 `CapabilityRequest.family` 的分派地位，以满足
   §3.4 与 §19.1。
5. **版本控制**：本仓库当前不是 git 仓库（`git status` → `fatal: not a git repository`），
   `.git` 无历史，导致 `计划:15` 的“记录关联变更”、阶段退出门槛与回退能力无法执行。
   是否执行 `git init` 并建立首个提交基线，需用户决定。
   **已解决（2026-09-11）**：用户确认执行 `git init` 并按现有 `.gitignore` 提交源码/文档/测试集。
   首个基线提交为 `7574633`（`chore: baseline commit (374 passed / 1 skipped)`），
   提交时工作树干净、`git status` 无输出。详见本文件 §7。
6. **D5 的 4 类缺失 provider**（阻抗/稳定性/噪声/热）：是补实现，还是收窄 §8.2 承诺。

---

## 6. 建档期间观察到的并发变更（非本次改动）

本文件建档于 2026-09-11。建档期间观察到仓库在被其他工作流并发修改，记录如下，
以免与本文件的判定混淆：

| 观察 | 证据 | 对本文档的影响 |
|---|---|---|
| `circuit_ai/mna.py` 的批量线性求解已被修复 | 建档前该行为 `np.linalg.solve(matrices, rhs)`（原 `circuit_ai/mna.py:209`，rhs 形状 `(n_freq, size)`）；建档后文件 mtime 为 2026-09-11 13:10:38，现值位于 `circuit_ai/mna.py:213`：`np.linalg.solve(matrices, rhs[..., None])[..., 0]`，并附注释说明 NumPy >= 2 把 `b` 视为核心 `(m, n)` 矩阵栈 | 计划文档阶段 0 新增的“恢复可运行基线”待办中，`mna.py` stacked solve 一项可能已由并发工作流完成，该待办实际剩余部分为安装 `pint`/`hypothesis` 与建立回归基线；措辞是否收窄待用户确认 |

本文件本身不修改任何代码或测试；上述变更不属于本次文档任务。

---

## 7. 版本控制基线（2026-09-11 建立）

本节记录 `git init` 与首个基线提交的事实，供后续变更对比与回退使用。

| 项 | 值 |
|---|---|
| 基线提交 | `7574633`（`7574633027c5781d5526c692cc064ef74ad13655`） |
| 提交信息 | `chore: baseline commit (374 passed / 1 skipped)` |
| 提交时工作树 | 干净（`git status --porcelain` 无输出） |
| 纳管条目 | 330（329 个常规文件 + 1 个 gitlink） |
| 提交时全量测试 | `374 passed, 1 skipped, 0 failed`（55.04s） |
| 提交者身份 | `w3743 <wangjiayu0403@qq.com>`（仅写入本仓库 `.git/config`） |

范围与例外（依据用户确认的“按现有 `.gitignore` 提交源码+文档+测试集”）：

- `outputs/`、`__pycache__/`、`.pytest_cache/`、`.pytest_tmp/`、`.hypothesis/`、
  `*.egg-info/`、`models/`、`data/*.jsonl` 按现有 `.gitignore` 排除。
  **后果：报告引用的证据产物（`outputs/_baseline_report.md`、
  `outputs/_handover_inventory.txt`、`outputs/_pip_freeze_*.txt`、
  各 `episode.jsonl` 冒烟产物）不在版本控制内**，其可追溯性依赖文件系统本身。
- `third_party/digikey-partner-kicad-library` 记录为 **gitlink**
  （`160000 b0bdcd1e0d2b817815c2bbf6fb656673b258bbe7`），未做 vendor 复制。
  该目录本身是干净的上游克隆（remote `github.com/Digi-Key/digikey-partner-kicad-library.git`），
  固定提交即可复现；但**未创建 `.gitmodules`**，故其他克隆方不会自动获取其内容。
- `_paper_2608.25512/` 已随基线提交纳入（5 个文件，约 2.7 MB）。该目录是
  arXiv 2608.25512《A Programming Paradigm for Spatiotemporal Composability》
  的抓取产物，**与本项目源码无关**，其来源归属仍待用户确认（见 §8）。

已知环境瑕疵（不影响基线有效性）：

- `.pytest_cache/` 目录当前**拒绝访问**（`Access to the path ... is denied`），
  属性为普通 `Directory`。pytest 因此每轮都报 `PytestCacheWarning: could not create
  cache path`，`.pytest_cache` 的 `--lf` / `--ff` 失效。该现象在 `git init` 之前即存在
  （见基线测试输出），非本次引入。

---

## 8. 移交报告事实核对（2026-09-11，接手方独立复核）

《移交报告》§8 把“填充 episode 的 `budget` / `random_seed`（缺陷 #1）”列为 **P0**，
理由为“`pipeline.py` 已有同名辅助函数，疑似未接线”。接手方独立复核后认为**该归因不成立**，
登记如下以免被当作已知缺陷继续传递：

| 报告原话 | 复核事实 | 证据 |
|---|---|---|
| “疑似未接线” | **已接线**，两处调用均在活动代码路径中 | `circuit_ai/pipeline.py:349` `budget=_search_budget(ir)`、`:350` `random_seed=_configured_seed(ir)`；辅助函数定义于 `:624` 与 `:636` |
| “`random_seed=None`” | **AC 路径已填充**；仅电源路径为 `None` | `outputs/episode_smoke_ac/stage_1_voltage_transfer/episode.jsonl` 三行均 `seed=7`；`outputs/episode_smoke_ac/stage_2_output_impedance/episode.jsonl` 一行 `seed=7` |
| “两行均如此” | 与 AC 实测矛盾，疑来自电源路径产物 | 同上 |

复核得到的**真实剩余问题**（性质与报告不同，属契约表达力而非接线缺陷）：

- `budget` 取值为 `{}` 是**当前输入的正确取值**：`_search_budget` 只读取
  `ir.optimization["graph_search"]` 的边界键（`max_expansions` / `beam_width` /
  `max_candidates` / `max_components` / `max_nodes` / `max_depth`，`pipeline.py:629`），
  而所用 PBDL 文件未配置这些边界，故“无预算约束”的忠实表达就是 `{}`。
- `random_seed=None`（电源 CLI 路径）同样是忠实取值：`_configured_seed` 读
  `ir.optimization["seed"]`（`pipeline.py:637`），PBDL 输入未提供种子。
- 因此**不应把 `{}` / `None` 直接当成“未采集”**。真正的缺口是：
  episode 契约无法区分“**未配置预算/种子**”与“**未采集该证据**”两种语义。
  若要闭合该缺口，应扩充契约（例如显式来源标记或 `null` 与缺省的区分），
  属架构决策，需用户拍板后再动。

证据时效性提醒：上表 AC 侧的 `seed=7` 取自 `outputs/episode_smoke_ac/`
（2026-09-11 13:29 产物，早于报告定稿）。**接手后重新生成**的产物中，
`python pbdl_synth.py 测试集/pbdl/baseline/5v_to_10v_dc_boost.json --out outputs/_h_overtake_dc`
得到 `budget={} seed=None`（两行），与上述解释一致。读取 `outputs/` 下旧产物作为证据前，
应先确认其是否由当前代码生成。

尚未复核、**不得据报告直接采信**的项（报告 §4.2 自述未完成，与本表无关）：
`SearchActionRecord.parent_hash` 链的真实搜索驱动、`orchestrator`/`pipeline` 写同一
`episode.jsonl` 的重复写入风险、`isolation_short` 自造反例直跑、holdout 组完整性、
`mna.py:213` 定向复数/多 RHS 实验、episode 与两个 replay 文件的交叉一致性。

遗留卫生项：

- `_paper_2608.25512/` 归属仍待用户确认（报告 §2.4 与 §6 缺陷 #7）。
  复核确认其内容为 arXiv 2608.25512 论文抓取物（`meta.py` 为提取脚本，
  标题《A Programming Paradigm for Spatiotemporal Composability》，
  作者含 DeepSeek-AI），**与本项目电路设计主题无关**。

