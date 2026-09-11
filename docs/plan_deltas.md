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
| **C6** | §7.3 规则 4 要求后端差异超门槛时标记 `model_disagreement` 并进入诊断队列；实现中只有 surrogate 集成分歧 | `计划:652-653` vs `grep -rn "model_disagreement" circuit_ai/` 无命中，仅 `circuit_ai/surrogate.py:314` 的 ensemble disagreement。**已解决（2026-09-11）**：契约实现于 `circuit_ai/optimization/fidelity.py`，接入 AC 真值门与电源路径，见 §9 |
| **C7** | 文档声明路径为旧仓库名，与实际目录不符 | `计划:4` `适用仓库：E:/Pioneer/ai电路` vs 实际 `E:/Pioneer/自动电路设计研究`（项目英文名已定为 **Lattice**，远端 `github.com/w3743/Lattice`；导入包名仍为 `circuit_ai`） |

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

---

## 9. `model_disagreement` 落地记录（2026-09-11，任务 H）

§7.3 规则 4 从"缺分支"变为已实现。登记本节的目的是说明**实现范围与未验证边界**，
避免后续把本机测不出来的部分当成已验证。

### 9.1 交付物

| 项 | 位置 |
|---|---|
| 偏差与分歧契约 | `circuit_ai/optimization/fidelity.py`：`ObservableDeviation`、`FidelityDisagreement`、`compare_observable`、`compare_fidelity_results`、`attach_disagreement_diagnostics`、`disagreement_from_options`、`graphs_are_comparable` |
| 稳定诊断 code | `MODEL_DISAGREEMENT_CODE = "model_disagreement"`（severity `warning`） |
| 策略键 | `optimization.fidelity_disagreement {enabled, threshold}`，**缺省关闭**；`circuit_ai/spec.py` `OptimizationSpec` 新增同名字段 |
| AC 接入 | `circuit_ai/synthesis.py` `_attach_simulation_evidence`：请求真值后端时**同时**跑内部模型，两侧都 PASSED 才比对 |
| 电源接入 | `circuit_ai/pipeline.py` `_pair_truth_with_planning` + 候选循环；比较记录写入 `report.json` 的 `fidelity_comparisons` |
| 结果字段 | `SynthesisResult.fidelity_comparison`、`PowerCandidateEvaluation.fidelity_comparisons`、`PowerDesignResult.fidelity_comparisons` / `model_disagreements` |
| 测试 | `tests/test_fidelity_disagreement.py`（12 项）、`tests/test_power_pipeline.py` 新增电源路径端到端 1 项 |
| 计划回填 | `计划:1856` 多保真 disagreement 由 `[ ]` 改为 `[x]`；§18.2 现为 15/15 |

### 9.2 过程中发现并修复的两个真实缺陷

1. **`_as_float_array` 强制 complex dtype**，导致 `_is_complex_like` 恒为真：
   所有观测量（含实值 DC 量）都走 dB 分支，文档承诺的"实值按较大幅值取比例"分支
   **是死代码**。已改为保留 dtype（仅 `c`/`O` 才转 complex），两条分支均有用例覆盖。
2. **`_pair_truth_with_planning` 按元组位置配对**：原实现用 `zip(truth_requests, truth_results)`
   取请求 id，若两个元组次序不一致就会静默配错——正是"比较不同对象却报告为同一候选"的错误。
   已改为按 `SimulationResult.request_id` 建索引配对，并以乱序用例锁定。

附带登记：标量比较原先在 `compare_observable` 内被硬编码为 `kind="waveform"`，
由调用方覆盖；现已改为显式 `kind` 参数。此为非行为性清理。

### 9.3 未验证边界（不得声称已验证）

- **本机无 `ngspice`**，因此"真实外部 SPICE 后端与内部模型产生分歧并被标记"的端到端路径
  **未在真实后端上跑过**。AC 与电源两侧的集成测试均以**委托后端**（把内部模型结果按固定
  dB/比例移动）替代真值后端，测试文档字符串已显式声明这一点。
- 因此当前可声称的是：**分歧判定、留痕、诊断投递与两处配对逻辑已实现并有测试**；
  **不能**声称"已完成真实 SPICE 与内部模型的一致性复核"。
- AC 路径的比较仅在 `spice_verification.enabled=true` 时发生（此时才存在两个真后端）。
  未开启时 AC 仍只有单一后端结果，不产生比较记录——这是设计选择，不是遗漏。

### 9.4 关键设计选择（需用户知悉，未擅自变更既有契约）

- **severity 取 `warning` 而非 `error`**：`SimulationResult` 的既有校验禁止 PASSED 结果携带
  error 诊断（`circuit_ai/simulation/contracts.py:539-542`），且分歧是"低保证据不足以单独签发"，
  不是"候选错误"。故选 warning，使不确定性可见而非把结果变成失败。
- **分歧不改变 PASSED 状态**：结果仍为 PASSED 并附带诊断，避免把"证据质量"与"仿真成败"混为一谈。
- **未收敛 `synthesis.py` 的第二套 `FidelitySchedule`**（报告 §8 的 P2 项）：
  本次只新增比较能力，未改动 `ac.staged.v1` 的选阶逻辑，以免把两件事耦合成一次变更。
  该项仍待办。
- **共用阈值常量**：新增 `DEFAULT_DISAGREEMENT_THRESHOLD = 1.0` 作为默认值单一来源，
  但**未**改动 `benchmark.py` / `replay.py` 中既有的 `accept_rmse_db: float = 1.0`
  字面量——它们属"验收指标阈值"，与"后端分歧阈值"语义不同，合并会改变既有判定含义。

### 9.5 顺带发现的既有测试隔离缺陷（未修，仅登记）

`SimulationExecutor` 的默认缓存是**进程级全局单例**（`circuit_ai/simulation/execution.py:77,92`），
且测试套件中无任何清理点。后果：先运行的用例会把同 spec 的仿真结果留给后运行的用例，
使"依赖真实调用次数"的用例**随执行顺序失败**。本次在新增的电源用例内显式
`SimulationExecutor().cache.clear()` 规避，但根因未动（清理点属于测试基建决策，
可能影响既有缓存收益断言）。建议后续单列任务处理。

---

## 10. 经典测试集的"可规划 ≠ 可执行"缺口（2026-09-11 实测）

### 10.1 实测方法与结果

`测试集/清单.json` 把 111 个用例分为 `expected_planning` 与 `execution_tier` 两个维度，
其中 4 个为 `execution`（正式冒烟验证），其余 107 个为 `schema`，`validation_scope` 为
`planner`。**即清单本身只承诺"规划器能正确分类"，从未承诺这些用例能跑通。**
此前没有任何数据说明这 67 个 planner-supported 用例能否执行，因此本次实测补上。

实测方式：对全部 67 个 `expected_planning=supported` 且 `execution_tier=schema` 的用例
逐个调用 `circuit_ai.pbdl_runner.run_pbdl_file(..., top_k=1)`。

| 结果 | 数量 |
|---|---|
| 执行成功 | **57** |
| 执行失败 | **10** |

### 10.2 失败用例与根因（均已复现）

| 用例 | 根因 |
|---|---|
| `ac_notch_1khz`、`ac_notch_50hz`、`ac_notch_60hz` | 综合层不支持 `bandstop` 行为：`spec failed feasibility checks: unsupported behavior kind: 'bandstop'` |
| `ac_allpass_phase_1khz`、`ac_allpass_phase_10khz` | 同上，不支持 `allpass` |
| `ac_tia_10kohm`、`ac_tia_100kohm`、`ac_tia_1mohm`、`ac_tia_10mohm` | `stage validation failed`；其 analysis 声明为 `voltage_transfer` 但 source 是 `current_input`，target 为 `target_kind=amplifier`，当前阶段校验无法处理该组合 |
| `bandgap_2v5` | 清单路径记为 `pbdl/classic/bandgap_2v5.json`，**实际位于 `pbdl/baseline/`** → `FileNotFoundError`；属清单路径错误，非能力缺失。即使路径修正，该 spec 选中的电源拓扑也要求库外元件（`['C','L','ideal_diode','ideal_switch']`） |

### 10.3 判定

- **`bandstop` / `allpass` 属规划器与执行器的能力不一致**：`端口描述语言/targets.py:101`
  已把两者列入 `FilterKind` 并有 `_bandstop_complex` 实现，`circuit_ai/pbdl_boundary.py:138`
  也接受这两种 kind，但 AC 综合的可行性检查不认。规划器因此把它们判为 supported。
  这既是能力缺口，也是**规划器过于乐观**的缺陷——两者应取其一。
- **TIA 4 例**属"声明了 analysis 但未声明可执行行为"的边界，需补 transimpedance 综合路径
  或把规划器判为 unsupported，同样属于规划-执行一致性决策。
- **`bandgap_2v5` 的清单路径错误**是纯数据缺陷，可直接修（但会导致该用例从
  "planner-supported 但执行失败"变成"规划即不支持"，需用户确认期望）。

以上 10 项**未修**，因为每一项都涉及"补能力"还是"收窄规划器承诺"的产品决策。
建议按 §7 待拍板事项一并处理。

### 10.4 本次能力结论（可对外表述的口径）

- 能**规划**（分类正确）的需求：111 个用例中 71 个 supported / 39 个 unsupported / 1 个 mixed。
- 能**执行**并有正式验证的：4 个（`execution_tier=smoke`，实测 4/4 通过）。
- 能**执行**但仅有规划级承诺的：67 个中的 **57** 个（本次实测）。
- 全新需求（不在仓库任何测试集中、`cutoff_hz=2200 Hz` 的一阶 Butterworth）：
  用标准 PBDL 格式综合成功，`rmse = 3.4e-11 dB`。
- **不得表述为**"71 个需求全部可综合"。准确口径是"71 个可规划，其中 61 个经实测可执行
  （4 smoke + 57 抽查），10 个规划通过但执行失败"。

---

## 11. 泛化能力补强记录（2026-09-11 起，进行中）

目标：让系统覆盖更多样的设计需求，且新增需求类型时主流程代码变更最小。

### 11.1 已交付

| 项 | 位置 | 解决了什么 |
|---|---|---|
| 损耗模型 | `circuit_ai/power_loss.py` | 效率原先恒为 1.0（理想平均模型不计任何损耗），"效率最大化"作为目标是恒等成立的。现在按开关/导通/二极管/开关电容/ESR 分项计算，效率随功率、频率、开关电压应力变化。族只贡献 3 个数据点（开关占空、整流占空、开关阻断电压），其余共享。 |
| DC 方程数据化 | `circuit_ai/power.py` `_ideal_stage_dc` | 四个近乎重复的 solver 合并为一，族只提供电压比与纹波因子。**新增一个电源族从"再写一个 solver"变成一次注册。** 各族的算式按原始写法保留（`vin/(1-d)` 与 `vin*(1/(1-d))` 差最后一位，而 DE 是随机优化器，这个差异会改变被选中的设计），golden 特征化锁可用于验证。 |
| 工作包络 | `circuit_ai/operating_envelope.py` | 需求从"单点"变为"范围"：`operating_envelope {min/max_input_voltage_v, min/max_load_fraction}`。`enumerate_corners` 展开出标称 + 各极值角；退化包络只出 1 个角，因此单点行为逐字节不变。 |
| 最坏情况分析 | 同上 `worst_case_summary` | 对**选定设计**在全部角上重算，给出最低效率/最大损耗/最大输入电流/占空范围/最大纹波/输出轨范围，并**逐项标注是哪个角造成的**。角求值失败会抛出而非静默丢弃。 |
| 稳压验收 | 同上 | 给出目标电压与容差后，`regulates_output` 判定每个角是否守住目标；未给目标时为 `None`，**不把"未检查"报成"通过"**。 |

三项均为**可选启用**：未声明 `operating_envelope` 的 spec 产物逐字节不变，
`report.json` 的 `operating_envelope` 与 `worst_case` 为 `null`。

### 11.2 由此暴露的真实缺口（已量化，未修）

在真实需求"隔离 36 V→5 V（2 A），输入 24–48 V，负载 10–100%"上实测：

| 角 | Vin | 负载 | 效率 | 输出 |
|---|---|---|---|---|
| nominal | 36 V | 100% | 0.9358 | **5.000 V** |
| input_min__nominal | 24 V | 100% | 0.9096 | **3.333 V** |
| input_max__nominal | 48 V | 100% | 0.9495 | **6.667 V** |

**结论：该设计只在标称点输出 5 V。** 原因是占空比被固定为标称变比
（`solve_ideal_flyback_dc` 用一次解出的 duty），而模型**没有反馈控制环**，
所以占空比不随输入调整。这不是实现缺陷，而是方法缺口：

- 要真正满足"24–48 V 输入仍输出 5 V"，需要**闭环控制/反馈补偿**设计。
  本项目的计划中尚无此项，`docs/plan_deltas.md` §9.4 与交接报告 §6 也已把
  "环路稳定性"列为缺失分析维度。
- 当前系统能做的是**如实报告**而非掩盖：输出轨范围与逐角违规已进入
  `worst_case.output_voltage.{spread,regulates,violations}`。

**这是"泛化到各种设计需求"当前最大的单点缺口**：绝大多数真实电源需求都写
"某输入范围 → 某输出"，而固定占空比设计无法满足其中的稳压部分。

### 11.3 新增需求类型的改动成本（实测，不再只是声称）

目标里"新增需求类型时主流程代码变更最小"是一条**可度量**的断言。本节用一个真实的新族
来量它，而不是声称它。

**方法**：在 `tests/test_new_power_family.py` 内**完全从包外**定义一个正激变换器
（isolated forward，`vout = n·d·vin`），经真实的 capability 注册表与 pipeline 驱动它。

**结果**：

| 度量 | 值 |
|---|---|
| 新族的物理定义 | 1 个函数，3 个事实（电压比、纹波因子、开关阻断电压） |
| 注册 | 2 条 `CapabilityRegistration`（optimizer + simulation backend） |
| **`circuit_ai/` 内的改动** | **0 个文件**（`git status --porcelain circuit_ai/` 为空） |
| 复用的共享机制 | 损耗模型、包络分析、最差情况、优化器驱动、能力解析——全部无需改动 |

正向族方程已逐点验证：`vin=48/n=0.5/d=0.2083 → Vout=5.0000`（理想 eff=1.0，
有损 eff=0.9092），三个不同工作点全部精确命中目标。

**过程中发现的两件事**（都属族定义自身要负责的部分，不是架构问题）：

1. 正激需要 `n > vout/rail`：在等号处所需占空比恰为 1.0，**没有任何余量**。
   族定义必须自己保留占空余量（本例取 5%）。
2. 正激**没有占空自由度**：占空比由匝比与所调度的电压轨共同决定。所以启用损耗模型后，
   占空比与匝比同时被定死，可优化的是 L/C/频率——这与 boost 派生族正好相反。

**已知边界（未修，如实记录）**：正激族被注册且能正确求解，但**不会被自动选中**，
因为 `topology_grammar` 生成的是 flyback/SEPIC 变体，不含正激。这是**拓扑生成**的限制，
不是架构限制；测试在两个方向上都加了断言，避免"注册了但没接线"被误判为通过。

### 11.4 拓扑生成的可达性检查与数据化扩展（2026-09-11，第二轮）

§11.3 的边界暴露出：新族能被**注册**并正确求解，但**不会被生成**。本轮处理这一层。

#### 11.4.1 变比可达性（修掉一个真实缺陷）

`PowerTopologyGrammar` 原先只用知识库的粗粒度 `voltage_relation`（`step_up`/`step_down`/`any`）
过滤，**完全不检查目标变比是否落在某族可实现的范围内**。后果：一个 40 倍升压请求
（5 V → 200 V）会通过 `step_up` 过滤、被构造成候选、然后在优化阶段才失败，既浪费搜索
又给出令人困惑的错误。

现在 `PowerStageModel` 增加了 `conversion_ratio`（族自己的理想变比函数）、`duty_limits`
与 `turns_ratio_limits`，并由它们**推导**出 `reachable_ratio_range()` 与
`can_reach_ratio()`。`_rejection_reason` 用它在构造后立即拒绝不可达候选，理由写明可达带：

```
dc_boost cannot produce a conversion ratio of 40 inside its usable duty band (reachable 1.053..20)
dc_buck  cannot produce a conversion ratio of 0.02 inside its usable duty band (reachable 0.05..0.95)
```

实测可达带（占空比取 0.05–0.95，匝比取 0.05–4）：

| 族 | 可达变比 |
|---|---|
| `dc_buck` | 0.05 – 0.95（只能降压） |
| `dc_boost` | 1.053 – 20（只能升压） |
| `dc_sepic` | 0.0526 – 19（跨单位增益） |
| `isolated_flyback` | 0.00263 – 76（匝比使其覆盖最广） |

**关键设计选择**：可达带由族的变比函数**推导**，不是手写表。手写表会与它所描述的 solver
脱节——这正是上一轮踩过的坑（代数等价但位级不同的算式）。未声明变比的族不做任何可达性
断言，因此不会因为"没声明"而被拒绝。

#### 11.4.2 生成层是数据化的（已实测）

新增一个**可被生成**的拓扑不需要改代码，只需在知识库 YAML 里加一条 production：
它用**已有的功能模块**（`two_terminal_inductor`、`pwm_switch`、`rectifier`、
`isolated_transformer`、`external_two_terminal_capacitor`、`external_resistive_load`）
组合出新拓扑，并声明 `family`、`solver`、`applicability` 与 `metadata.isolated`。

实测：把正激变换器的 production 加入临时知识库后，`PowerTopologyGrammar` **立即生成**了
`grammar_ideal_forward`，且与既有候选图不重复。

**过程中确认的两件事**：

- `metadata.isolated: true` 是**知识声明**而非从模块列表推导出来的——隔离契约必须显式声明，
  否则候选会以 `isolation contract does not match candidate` 被拒。
- 已声明的 production **不豁免**于可达性检查：它的 solver 若未注册为 `PowerStageModel`，
  就不做可达性断言（无声明即无断言），但一旦注册就同样受约束。

#### 11.4.3 与 §11.3 的关系

两轮合起来，把一个族的完整落地路径拆清了：

| 要做的事 | 成本 |
|---|---|
| 让族能**求解**（物理方程） | 1 个函数、3 个事实；`circuit_ai/` 零改动（§11.3） |
| 让族能**被生成**（拓扑组合） | 1 条知识库 production，用已有模块组合；代码零改动（§11.4.2） |
| 让族受**可达性**约束 | 注册为 `PowerStageModel` 并声明 `conversion_ratio` |

三件事都不需要改主流程。

### 11.5 AC 频带验收（2026-09-11，第三轮）

前两轮把 DC 电源侧的"需求=范围 + 最坏情况"补齐了，AC 侧仍是空的：AC 设计只用
`rmse_db` / `max_abs_db` 对照解析目标打分，回答的是"**平均来看形状有多接近**"，
从不回答"**通带是否守住纹波、阻带是否达到衰减**"——而滤波器需求恰恰是用这两个数写的。

`circuit_ai/frequency_mask.py` 补上这一层：

| 项 | 说明 |
|---|---|
| `BandSpec` | 一个频带的增益上下限（dB），必须至少给一个限值 |
| `FilterMask` | 有序频带集合；**过渡带由相邻频带推导，不重复声明** |
| `evaluate_filter_mask` | 逐带给出裕量、**决定该带的最坏频率**、越限的是哪个限值、带内实测极值 |
| `FilterMaskReport` | 汇总判定；`passed` 要求每个带都守住 |

**关键设计选择**：

- **频带判定与形状评分分离**。一个设计可以在平均意义上紧贴目标（`rmse_db < 1 dB`）
  却仍然击穿阻带底线——测试 `test_band_verdict_is_independent_of_the_shape_score`
  正是锁定这一点。
- **空频带默认不通过**。"没有采到样本"不是"满足要求"的证据；除非显式声明
  `allow_empty: true`。这与 DC 侧 `regulates = None` 而非 `true` 是同一条原则。
- **过渡带不设要求**，因此也不参与判定；但它被显式列出并给出宽度比。

#### 11.5.1 顺带修掉的一个真实缺陷

自检时发现 `filter_mask` 写进 spec 后**到不了执行层**。追溯发现
`circuit_ai/pbdl_boundary.py` 的 `translate_pbdl_dict` 把 legacy 两端口 spec 经
PBDL 往返转换，而 PBDL 翻译器只输出固定键集（`name/ports/behavior/library/optimization`），
于是 `behavior` 里**不在该键集内的一切都被静默丢弃**——包括新的 `filter_mask`，
也包括既有的 `gain` 与 `order`。

修法：`translate_pbdl_dict` 在转换后把需求级键**重新挂回 `behavior`**
（`_reattach_requirement_keys`）。挂回 `behavior` 而非顶层，是因为
`SynthesisSpec.from_dict` 只暴露 `behavior`，顶层键仍会被它丢掉。

这是一个**静默数据丢失**类缺陷，此前没有任何测试覆盖到它。

#### 11.5.2 实测效果

在 1 kHz 一阶 RC 低通 + 掩模「通带 ≤500 Hz 需 ±1 dB、阻带 ≥30 kHz 需 ≤−20 dB」上，
CLI 产物 `report.json` 现在带 `filter_mask` 字段，给出 `passed`、`failed_bands`、
`worst_band`、`worst_margin_db` 与逐带明细。

#### 11.5.2 掩模参与优化（不止判定）

第一版只**判定**最终响应。现在掩模也**引导搜索**：`_score_response` 增加
`band_mask_penalty` 项——各带越限量的 dB 之和（满足时为 0），乘以
`weights.band_mask`（默认 1.0）与各权重之和。

三条防作弊规则：

- **未声明掩模时罚项恒为 0**，未用掩模的设计其 score 与改动前**逐位相同**。
- **响应无法求值**（非有限或形状不符）时按固定大罚处理，不允许"算不出来所以没违规"。
- **空频带额外计罚**，防止优化器把截止频率挪出分析范围来"逃出"某个带。

实测验证了罚项确实在起作用：把阻带底线设成不可达的 −80 dB 后，优化器**放弃了通带**去
追赶阻带——即它真的在按掩模张力重新取舍，而不是只被事后判一次。

### 11.6 占空比调度：把"可行"与"已稳压"分开（2026-09-11，第四轮）

§11.2 记录了一个真实缺口：固定占空比的设计只在标称轨输出目标电压。本轮把它拆成
**两个不同的断言**，分别如实报告。

#### 11.6.1 先确认物理事实

对隔离 24–48 V → 5 V，按族的变比反解各轨所需占空比（匝比 0.30，即实测优化选中值）：

| 输入 | 所需占空比 |
|---|---|
| 24 V | 0.4098 |
| 36 V | 0.3165 |
| 48 V | 0.2577 |

可用占空比区间是 [0.05, 0.95]，**所以每个轨所需占空比都是物理可实现的**。
结论：拓扑没问题，问题只在于**固定占空比无法同时满足它们**。

#### 11.6.2 两个断言，分开报告

| 字段 | 含义 |
|---|---|
| `regulates_output` | **这台的实物**做到了什么。固定占空比 → 跨轨不稳压 → `false` |
| `duty_schedule.all_reachable` | **这个拓扑**能否做到。所有轨所需占空比都落在可用带内 → `true` |

`duty_schedule` 逐角给出所需占空比、是否在带内、到上下限的余量，并汇总
`control_effort`（占空比需要覆盖的范围）、`spread` 与 `limiting_corner`（余量最小的角）。

实测（24–48 V、10–100% 负载、6 角）：

```
regulates_output = False        输出轨极差 3.3333 V
duty_schedule.all_reachable = True
control_effort = (0.6757, 0.8065)   spread = 0.1308
limiting_corner = input_min__load_min   （到上限余量 0.144）
```

即：**同一套硬件，只要控制器能在 13% 的占空范围内按输入调节，就能守住 5 V。**
缺的是控制器，不是可行性。

#### 11.6.3 明确不声称的事

`DutySchedule.justifies_stability_claim` **恒为 False**，并且写进序列化产物。
该字段存在的目的就是让报告**无法被读成"环路已补偿"**：占空比可达性只说明转换可行，
不说明环路能否稳定——增益/相位裕度、穿越频率、瞬态恢复都需要本项目**没有**的分析能力。
要改这个字段必须显式改代码，不会因为别的改动而悄悄变成 True。

### 11.7 负载阶跃瞬态（2026-09-11，第五轮）

此前唯一的瞬态模型是**启动**：输出向直流目标爬升。而"负载从 10% 跳到 100% 时
输出不得跌落超过 X mV"是实际电源需求中最常见的一条，此前**无法表达、无法评价、
无法报告**。

`circuit_ai/load_step.py` 补上这一分析类型：

| 项 | 说明 |
|---|---|
| `LoadStepRequirement` | 阶跃起点/终点（额定电流的倍数）+ 跌落/过冲/恢复时间预算 + 环路响应时间 |
| `LoadStepResult` | 实测落差、恢复时间、是否结算稳定、**达成预算所需电容** |
| `evaluate_load_step` | 电荷平衡模型的逐步积分 |

**模型是什么**：负载突变的瞬间，输出电容补上缺口，直到环路把变换器输出电流拉到新需求。
主导项是缺失电荷 `Q = dI·τ/2`，因此落差为 `dI·τ/(2C)`、恢复时间为回转时间本身——
这是标准的电容定量关系，也是让结论**可行动**的原因。

实测（36 V→5 V/2 A，0.1→1.0 阶跃，τ=50 µs）：

| 输出电容 | 落差 | 判定 | 达成 50 mV 所需 |
|---|---|---|---|
| 100 µF | 449.3 mV | 不通过 | 900 µF |
| 470 µF | 95.6 mV | 不通过 | 900 µF |
| 1000 µF | 44.9 mV | 通过 | 900 µF |

数值解与闭式解相差 0.15%（Euler 网格误差），两者都记录在产物里。

#### 11.7.1 预算参与优化

与发展 §11.5.2 的掩模同理：第一版只事后判定。现在越限量的平方被计入级联目标，
**未声明预算时罚项恒为 0**。实测效果：

```
无预算   -> 选中 100.0 µF
50mV 预算 -> 选中 898.6 µF   （计算出的最优即 900 µF）
```

**预算确实改变了被选中的设计**，而不只是多打一个标记。

#### 11.7.2 明确不声称的事

`LoadStepResult.justifies_stability_claim` **恒为 False** 并写入产物。
`loop_response_s` 是**假设的**闭环回转时间，不是设计出来的：本分析回答
"这个预算能否达成、需要多少电容"，**从不回答"环路是否稳定"**。增益/相位裕度、
穿越频率、条件稳定性需要补偿器设计与小信号环路模型，本项目没有。

#### 11.7.3 过程中修掉的一个静默缺陷

`load_step_from_mapping` 原本只在**容器**里查 `load_step` 键，而调用方直接传**块本身**，
于是查找永远落空、静默返回 `None`——表现成"用户没声明需求"而不是"解析出错"。
已改为同时接受容器与裸块，并加测试锁定两种形式。这类"静默返回空"的错误比抛异常危险得多，
因为它看起来像正常的缺省行为。

### 11.8 验收域注册表（2026-09-11，第六轮）

到本轮为止已经有三个验收域：**工作包络**（§11.2）、**频带掩模**（§11.5）、**负载阶跃**（§11.7）。
它们在结构上是同一件事：声明一个可接受域 → 对选定设计逐点判定 → 归属最坏点 →
拒绝声称超出证据的结论。

**问题（实测）**：每加一个域都要在 `power.py` 里手改 **4 个近乎相同的构造点**，
因为四个 power 结果类是逐字段的副本。以 `load_step` 那一次提交为证：

| 度量 | 值 |
|---|---|
| 那次提交规模 | 7 个文件、753 行新增 |
| `power.py` 中 `load_step` 的机械改动 | 4 个类字段 + 4 个构造点 + 2 处罚项 = **11 处** |

**改法**：新增 `circuit_ai/acceptance.py`，把"验收域"变成注册项。

```python
register_domain(AcceptanceDomain(name="load_step", evaluate=_load_step_domain))
```

四个构造点各自收缩为**一行**：

```python
**_acceptance(ir, params, vin, vout, iout, design_rail, solver, loss, ResultType),
```

新增一个验收域现在是：**一个新模块 + 一次注册**，不再需要改四个地方。

#### 11.8.1 防静默丢弃

注册名**必须**是结果记录上已存在的字段，否则 `register_domain` 直接抛错：

```
acceptance domain 'not_a_field' has no field on the power result records;
known fields are ['envelope_analysis', 'load_step']
```

这条守卫是必要的，因为结果类是 frozen dataclass，**无法在运行时长出字段**——
一个注册了但落不到字段上的域会被静默丢弃，那比报错危险得多。

#### 11.8.2 行为等价性证明（不是声称）

重构后**产物是否逐字节相同**是可验证的，所以验了：对两个 spec（带包络+负载阶跃的隔离电源、
无任何验收域的升压）分别用重构前（`HEAD` 版本）与重构后跑一遍，剔除**时钟派生字段**
（`runtime_s`、`duration_s`、`evidence_hash`——后者是对含 `runtime_s` 的载荷取摘要，因此必然随之变化），
其余全部相同：

```
enveloped.json: identical outside wall-clock fields
plain.json:     identical outside wall-clock fields
```

#### 11.8.3 一次失败的重构尝试（如实记录，已回退）

我尝试进一步把**四个结果类本身**合并为一个 `PowerStageResult` + 四个别名——
它们逐字段相同，合并后每个域只需改 1 处而非 4 处。评估显示风险很低：全仓只有
**类型注解**与 `__init__` 导出引用这些名字，**没有任何 `isinstance` 判断**，
别名可保证调用方不受影响。

但基于文本块的替换脚本两次切错边界：第一次吞掉了 `FlybackParameters`（`end` 扫描起点
用了最后一个结果类之后，而不是第一个之后），修正后仍在别的区域出错。我**回退了这次尝试**，
没有留下半成品，并把剩余的类重复登记为已知项。

**判断**：那次清理的收益（去掉 12 行重复定义）远小于风险与已耗时间，且不影响泛化能力本身；
注册表已经把**每域改动成本**从 4 处降到 1 处，这才是目标关心的量。

### 11.9 目标容差从未被编译（2026-09-11，第七轮，真实缺陷）

**发现过程**：做覆盖矩阵审计时（§11.10）探测"AC 滤波器阶数"维度，结果异常：

| 需求 | 选中器件 | RMSE | `constraint_report.passed` |
|---|---|---|---|
| 1 阶 Butterworth | rc_lowpass (2 元件) | 2.3e-8 dB | True |
| **2 阶** Butterworth | rc_lowpass (2 元件) | **9.6 dB** | **True** |
| **4 阶** Butterworth | rc_lowpass (2 元件) | **32.1 dB** | **True** |

一个 4 阶 Butterworth 需求被一个一阶 RC 满足，误差 32 dB，而系统报告
`verified_feasible`。

**根因**：`circuit_ai/constraints/compiler.py` 的 `_compile_frequency_target`
对滤波器目标**只生成一条软目标**（`target.N.minimize_rmse_db`，severity=objective），
**从不读取 `targets[N].tolerance`**。因此验收门实际上对滤波器需求不设任何硬限：
`constraint_report` 里除了结构性硬约束外，与频响有关的只有 `status=observed` 的目标项，
`passed` 恒为 true。

`计划:186` §3.5「无证据不宣称」在此处被违反——不是靠伪造，而是靠**没有生成约束**。

**修法**：新增 `_target_tolerance_limit()`，把目标容差编译成一条 **HARD MAXIMUM** 约束
（`target.N.rmse_db_within_tolerance`），与软目标**共用同一个 metric**，因此两者不会漂移。

| 容差写法 | 转成的 dB 限值 |
|---|---|
| `absolute: 0.5` | 0.5 dB |
| `relative: 0.05` | `20·log10(1.05)` ≈ 0.424 dB |

同时给出时的优先级：**relative 优先**，因为相对带随需求缩放，而绝对误差是用户需要猜的数字。
转换按**对称带**处理，使限值不依赖滤波器自身增益——这正是写相对容差的意义。
`bool` 被显式排除（Python 里 `True` 是 `int`，否则会被读成 1 dB 容差）。

#### 11.9.1 影响（实测，非推算）

| 度量 | 改动前 | 改动后 |
|---|---|---|
| 全量测试 | 527 passed | **546 passed / 0 failed**（无回归） |
| 经典用例（planner-supported 且 schema 级）执行成功 | 57 / 67 | **55 / 67** |

**为什么少了 2 个**：`ac_active_highpass_5khz` 与 `ac_active_highpass_20khz`
此前**一直在违反它们自己声明的容差**——实测误差 9.578 dB，声明 `maximum = 1 dB`。
它们过去"通过"只是因为限值从未被求值。

**这不是能力退化，而是把假通过改成了真报告。** §10 里"57 个可执行"的口径应更新为
"55 个可执行，另 2 个经容差求值后确认超限"。

#### 11.9.2 遗留

- 这 2 个用例现在失败是正确的（设计确实不满足需求），但**是否要修设计**（例如允许更高阶
  或有源模板）还是**收窄它们的声明容差**，属产品决策。
- 覆盖矩阵同时确认：`bandstop` / `allpass` 仍不支持（§10 已登记），
  4 个 TIA 用例的 observable 名不匹配，`bandgap_2v5` 是清单路径缺陷。

### 11.10 覆盖矩阵审计（2026-09-11）

对"能覆盖更多样的设计需求"做一次实测审计，逐轴探测，失败项如实列出：

| 轴 | 结果 |
|---|---|
| AC 滤波类型 | 2/5（lowpass、highpass 通过；bandpass 在我的 R/C 库里无模板，换 R/C/L 后 7.8e-8 dB 通过；bandstop、allpass 不支持） |
| AC 阶数 | 4/4（1–4 阶都能构造，但 2 阶以上**不再假通过**，见 §11.9） |
| AC 阻抗类 | 3/3 |
| 非隔离电源族 | 2/2（buck、boost） |
| DC 验收域组合 | 4/4（包络、包络+损耗、包络+损耗+定标轨、包络+负载阶跃） |
| AC 验收域 | 1/1（频带掩模） |
| 显式拒绝 | 2/2（noise、s_parameter 均以 `CapabilityPlanError` 拒绝而非静默通过） |

**结论**：18/21 通过；3 个失败中 1 个是我的探测库限制（bandpass 需要 R/C/L），
2 个是已登记的 bandstop/allpass 不支持。

### 11.11 效率约束读的是"理想 1.0"而非设计的真实效率（2026-09-11，第八轮，真实缺陷）

**发现过程**：§11.10 的审计确认 55 个用例执行且 0 个超限后，继续追问"DC 用例声明了
`efficiency: 0.8`，这个数字真的被求值吗"。答案是：**被求值了，但求的是错的东西**。

**现象**（实测）：

| 需求 | `operating_point.efficiency` | 效率约束的 `actual` | 判定 |
|---|---|---|---|
| 无效率要求，无损耗模型 | 1.0000 | 无约束 | passed |
| 无效率要求，有损耗模型 | **0.9215** | 无约束 | passed |
| `efficiency: 0.8`，有损耗模型 | 0.9215 | **1.0** | passed |
| **`efficiency: 0.99`**，有损耗模型 | 0.9215 | **1.0** | **passed（错）** |

**根因**：解析电源后端注册时绑定的是族的**无损耗** solver
（`_stage_dc_solver(adapter, solver)` 只传两个参数）。于是：

- 优化器自己算出的工作点是**有损的**（0.9215）；
- 仿真**证据**（`SimulationResult.scalars['efficiency']`）是**无损的**（1.0）；
- 而效率约束的指标读的是**后端证据**。

所以一个 0.99 的效率要求在 0.9215 的设计上**通过**了。这是 §3.5「每个最终结论必须能追溯到
后端、模型版本…原始结果摘要」被违反：结论追溯到了一个**不是该设计**的数字。

#### 11.11.1 修法

损耗系数改为**随请求传递**，而不是由后端去读 spec——因为"结果必须能从它自己的请求复现"，
且仿真缓存键包含请求条件。

| 位置 | 改动 |
|---|---|
| `circuit_ai/power_request.py`（新建，叶子模块） | 定义 `POWER_LOSS_CONDITIONS_KEY` 与 `loss_parameters_from_conditions()`；放在叶子模块是为了让请求构造方与后端都能引用而不产生循环导入 |
| `simulation_tasks.py` | 仅当 spec 启用损耗模型时，把 `optimization.loss_model` 写入任务条件 |
| `power.py` `_stage_dc_solver` | 适配器接受可选 `loss_parameters` 关键字 |
| `simulation/backends.py` | 从请求条件解析损耗系数；**仅在存在时**才向 solver 传该关键字 |

**"仅当存在时"是刻意的**：未启用损耗模型的 spec，其请求哈希、缓存键与产物**逐字节不变**。

#### 11.11.2 修复效果（实测）

| 需求 | 修复前判定 | 修复后判定 |
|---|---|---|
| `efficiency: 0.8`，有损 | passed | passed（0.9215 ≥ 0.8，正确） |
| **`efficiency: 0.99`，有损** | **passed（错）** | **failed（对）** |
| 无损耗模型的一切路径 | 不变 | 不变 |

全量：546 → **557 passed / 0 failed**（0 回归）。
经典 67 个 planner-supported schema 用例：**55 执行、0 超限**，与修复前一致
（这些用例的 `efficiency: 0.8` 在 0.9215 下本就满足）。

### 11.12 外部工具其实可用——"未验证"结论建立在错误的搜索范围上（2026-09-11，第九轮）

**这是本项目至今最重要的一次更正。**

自移交报告以来，全部文档都写着：

> `ngspice`、`kicad-cli`、`kicad` **均不在 PATH**，无 KiCad 安装目录
> → 所有"真实 SPICE / KiCad 可打开"类验证**未做**，后续也不得声称已做。

这条结论**是错的**，而且错了 9 轮。实测：

| 工具 | 实际位置 | 状态 |
|---|---|---|
| **ngspice** | `C:\Users\wangj\AppData\Local\Autodesk\webdeploy\production\<hash>\Applications\Electron\LibEagle\ngspice\bin\ngspice.exe` | **可用**，批处理模式 0.17 s 跑完一次 AC 扫描 |
| **KiCad 10.0.4** | `E:\kicad\bin\kicad-cli.exe`（含 `eeschema.exe`） | **可用** |
| EasyEDA 模拟器 | `C:\Program Files\lceda-pro-sim\ngspice.dll` + 器件模型库（BSIMBULK/HICUM/PSP/VBIC） | 仅 DLL，无独立 exe |

**错误来源**：`NgspiceSimulatorBackend.available()` 与 `NgspiceVerifier.available()`
只查 `PATH` 和调用者给定的路径。ngspice 是**作为另一个应用的组件**安装的——
这是打包副本的常见形态，而这类副本**用于批处理完全正常**。

**后果**：一个错误的可用性判断，被当成"无法验证"的正当理由沿用了 9 轮，
进而让每一轮我都拒绝声称任何外部验证。**结论正确与否，取决于搜索范围是否诚实**，
而这一条从未被复查过。

#### 11.12.1 修法

新增 `circuit_ai/spice_discovery.py`，搜索顺序**显式**：显式路径 → 环境变量
（`LATTICE_NGSPICE` / `CIRCUIT_AI_NGSPICE`）→ `PATH` → 已知安装/打包位置。
只做 `is_file()` 判断，**不执行**——GUI-capable 程序拿到不认识的参数会**弹窗**
而不是报错（本轮我实际触发过一次，打扰到了用户）。

两个细节值得记录：

- **`definitely_missing*` 惯例必须保留**：多个测试与 spec 用这个前缀名**要求"无外部后端"**。
  若发现机制覆盖它，一台装有 ngspice 的机器会把**有意的可用性测试变成假通过**。
- **Windows 需补扩展名**：`Path(dir) / "ngspice"` 找不到 `ngspice.exe`，
  而 `shutil.which` 自己会补。

#### 11.12.2 顺带修掉一个真实缺陷：ngspice 会挂起而不是报错

`NgspiceVerifier` 传入**相对** `work_dir` 加**相对** netlist 路径。ngspice 打不开
netlist 参数时**退回交互模式**，于是调用一直阻塞到超时（实测 20 s），
而**不是**报告文件缺失。**所有手工测试我都用了绝对路径**，所以这个 bug 一直没暴露。

修法：`work_dir` 与 netlist 路径一律 `resolve()` 成绝对路径，`cwd=str(...)`。
`NgspiceSimulatorBackend` 同样处理（它的 `TemporaryDirectory` 本就绝对，加 `resolve()` 防退化）。

另修：`NgspiceVerifier` 原先**只在 `available()` 里做发现**，`self.executable` 仍是裸名，
于是"检查的路径"与"运行的路径"不一致——可用性报 true，实际运行却找不到程序。
现在构造时就解析一次，检查与运行用同一个路径。

### 11.13 首次真实外部验证结果（2026-09-11）

**这是本项目第一次用独立仿真器验证内部模型。** 此前所有 "verified" 都只意味着
"内部模型自洽"。

| 电路 | 内部模型最大误差 | ngspice 最大误差 | 偏差 |
|---|---|---|---|
| RC 低通 1 kHz | 8.75e-08 dB | 1.83e-05 dB | ~1.8e-05 dB |
| RC 高通 1 kHz | 1.15e-08 dB | 4.40e-05 dB | ~4.4e-05 dB |
| RC 低通 10 kHz | 1.75e-03 dB | 1.75e-03 dB | 一致 |
| RLC 带通 1 kHz | 2.47e-07 dB | 2.30e-05 dB | ~2.3e-05 dB |

**结论**：内部线性 MNA 与 ngspice 在 R/C/L 网络上一致到 **4.4e-5 dB** 以内。
测试 `tests/test_external_spice.py` 以 **0.1 dB** 为界锁定（刻意宽松：这是验证两个模型
描述同一个电路，不是验证它们逐位相同），并在无外部 SPICE 的机器上自动 skip。

**仍不得声称**：KiCad 可打开性尚未验证（`kicad-cli` 已确认可用，但导出文件尚未真正打开过）；
非线性/有源/s 参数仍未实现，因此**那些**结论依然是"未验证"。

全量：572 → **584 passed / 0 failed**。

### 11.15 非线性有源器件层（2026-09-12）

**这是本项目第一次有能力求解含非线性器件的电路。** 在此之前元件词表只有
`C, L, R, current_source, vcvs, voltage_source` 六种，即**零个有源器件**——这
意味着任何一个真正做事的电路都表达不出来。现在加入了二极管（`circuit_ai/devices.py`
的 `Diode`）与牛顿法直流工作点 + 小信号交流分析（`circuit_ai/nonlinear.py`）。

架构要点：

- **矩阵装配被抽出为 `mna.assemble_system`**，线性与非线性两条路径共用同一份
  装配代码与同一个盖印顺序。原因是既有功率族的黄金表征按位锁定，任何顺序变动都会
  被发现，所以共享而非重写是唯一不会漂移的做法。
- **直流与交流用两套计划**。直流路径把器件内部节点**消去**（`solve_junction_voltage`
  解单调标量方程）；交流路径**携带**该内部节点（`with_internal_nodes`）。
  这不是风格问题：把内部节点带进直流牛顿迭代会让结电压一步跳到斜率小几个数量级的
  位置，内部节点随即悬空，迭代会**以满容差收敛到"二极管不导通"的退化解**。
  交流是单次线性求解，没有这个盆地问题，而内部节点在交流里是必需的（见下）。
- **串联电阻在交流里不能折合**。折合成等效端电导在直流是精确的
  （`gd/(1+gd*RS)`），但折合形式没有位移电流绕过电阻的路径：实测在 1 GHz 差
  0.9%、在 10 GHz 差 **68%**。携带内部节点后 10 GHz 吻合到 1.2e-6。

**与 ngspice 的实测一致性**（全部参考值由批处理模式采集，`reltol=1e-10
abstol=1e-16 vntol=1e-12 numdgt=14`，非抄写公式）：

| 情形 | 最大偏差 |
|---|---|
| 默认模型正向直流扫描 0.3–0.8 V | 1.5e-5 V |
| 温度扫描 −40…150 °C | 1.8e-5 V |
| 串联电阻 RS=10 直流 | 1.6e-5 V |
| 默认模型小信号（1 Hz–1 GHz，平坦） | 7.4e-5 V |
| 结电容+渡越时间交流 1 kHz–10 GHz | 8.0e-5 V |
| 击穿 BV=5.5 | 9.8e-4 V |

`tests/test_nonlinear_devices.py`，44 项，锁定以上全部。

**过程中被实测否定的三个"看起来对"的公式**（这是本次最有价值的部分，三条都是
单点核对无法发现的）：

1. **饱和电流温度律**。SPICE3 公开式、ngspice 手册式、ngspice 源码式三者
   **在标称温度完全一致**，在 85 °C 分别差 1.76× 与 1.35×。三句中只有源码那句是
   实际运行的。已按源码实现并逐温度核对。测试里专门有一项断言"被否定的两式必须
   不通过"，以防将来被"简化"回去。
2. **击穿指数有自己的理想因子 NBV，默认 1，与正向 N 无关**。把两者当同一个，
   在 N=1.8 的器件上击穿电流差 **243 倍**。而两者在膝点处相同，所以只有在击穿区
   深处取样才能发现。
3. **结电压限幅器的方向**。参考实现只限制"继续导通"方向；本项目最初把关断方向也
   钳到一个下限，结果本该关断的结被永久正向偏置，迭代落入极限环、永不收敛。

**如实记录未做到的部分**：ngspice 会把膝点电压外移一点使反向区与击穿区电流连续，
其迭代式本项目的**直译版本在测试参数集上不收敛**，实测膝点也无法用它复现。因此
本实现采用数据手册定义（`IBV` 恰在 `BV` 处），并不声称复现了参考的膝点调整。
代价已量化：在膝点偏移可忽略的击穿电流下两者差 5e-4 相对；在较大的击穿电流下参考
的膝点低约 0.08 V、电流高约 2.4 倍。**击穿的斜率（即击穿特性的用途所在）已跨三个
数量级与实测核对。**

### 11.16 s 参数（2026-09-12）

2.4 GHz 目标所需的 RF 词汇里，s 参数是**唯一不需要新数值内核**的一块：小信号求解
已经给出线性网络，s 参数只是它端口导纳矩阵的一次基变换。新增
`circuit_ai/sparameters.py`。

**端口测量方式（以及为什么不是另外几种）**：每个端口轮流用**诺顿源**驱动
（`1/Z0` 电流源并联该端口自身的参考阻抗，等价于 1 V 戴维南源串 Z0），其余端口
接参考阻抗。两种半成品都试过并都被实测否定：

- **不接端口自身参考阻抗**：测到的是网络在另一种配置下的行为，不是反射系数所定义
  的那种。在串联 10 Ω 的二端口上给出 0.818，正确值 0.0909。
- **端口接参考阻抗但不加驱动**：没有可测的量。
- **只注入电流、不接参考阻抗**：需要把测得的电压反解成端口阻抗，而这个反解关系
  对网络拓扑敏感，先后三种写法各错一次（`v`、`v/(1-v)`、`v/(2-v)`），每种都在
  某一个特定阻抗上恰好正确——所以匹配负载的检查看不出任何一种。

最终形式不需要反解：入射波恒为 1/2，所以 `S11 = 2V1 - 1`、`Si1 = 2Vi`；四个极限
（短路 −1、匹配 0、开路 +1、传输因子 2）各自锁住一个常数。

**验证**：

| 参考 | 一致性 |
|---|---|
| 串联元件闭式解 `Z/(Z+2Z0)`、`2Z0/(Z+2Z0)` | 4 个阻值全部 ≤1e-9 |
| 并联元件闭式解 `−Z0/(2R+Z0)`、`2R/(2R+Z0)` | 4 个阻值全部 ≤1e-6 |
| 短路/开路/匹配极限 | −1 / +1 / 0 |
| 无源性（奇异值 ≤ 1） | 通过 |
| **ngspice 实测**（同一网络，1 V 戴维南源 + 两端 50 Ω，1 MHz–1 GHz） | S11 与 S21 均 **6.7e-16** |

最后一行是本次最强的外部验证：两条完全独立的路径（ngspice 内部求解 + 标准行波
提取，与本实现）在双精度舍入极限上一致。

**过程记录**：这一节花的时间远超预期，原因值得记下来——我在端口阻抗的反解上连续
错了三次，每次都是"看起来对、在匹配点上恰好也对"。**闭式解是唯一把每次错误都立刻
暴露出来的参考**；只对 ngspice 对比时，我自己写错参考提取式也会得到"一致"。测试
文件里因此同时保留闭式解与实测两套，并在 docstring 里说明了各自能抓什么错。

`tests/test_sparameters.py`，31 项。全量 588 → **619 passed / 0 failed**。

### 11.17 下一步（按价值排序）

1. **有源器件词表的其余部分**：BJT / MOSFET / 运放 / 电流镜。牛顿框架、限幅器、
   小信号线性化都已就位，新增一个器件现在是写 `NonlinearDevice` 的一件事。
2. **噪声分析**：模拟分析四类里唯一还缺的一类。
3. **s 参数接入 PBDL / 报告 / KiCad 流程**：目前 `sparameters` 是可直接调用的库，
   尚未接进 `pipeline.py` 的 report.json，也还没有 CLI 开关。
4. **闭环/稳压**（§11.2，最高价值）：完整做法是反馈补偿设计；当前只做到
   "选择在哪个轨上定标"并如实报告剩余偏差，以及报告所需占空比调度（§11.6）。
5. **10 个规划-执行不一致用例**（§10）：需用户拍板补能力还是收窄承诺。
6. **把正激 production 正式并入 `power_topologies.yaml`**：是否纳入需用户确认
   （会改变默认候选集）。
7. **合并四个重复的结果类**（§11.8.3 的失败尝试）：收益有限、风险已知，
   建议在下次顺手改动 `power.py` 结构时一并做。
8. **验收域的可视化/汇总**：三个域的记录目前分散在 `report.json` 的不同键下，
   尚无统一的"需求达成总览"。是否需要一个统一汇总视图属产品决策。

