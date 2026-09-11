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

### 11.4 下一步（按价值排序）

1. **闭环/稳压**（§11.2，最高价值）：完整做法是反馈补偿设计；当前只做到
   "选择在哪个轨上定标"并如实报告剩余偏差。
2. **拓扑生成可扩展性**：让新族能被 `topology_grammar` 生成而不只是被注册，
   这是 §11.3 边界所暴露的下一个瓶颈。
3. **多分析类型**：目前 AC 与 DC 电源是两条路径，尚未统一到同一包络/角框架下。
4. **10 个规划-执行不一致用例**（§10）：需用户拍板补能力还是收窄承诺。

