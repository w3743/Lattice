# ADR 007: 图内部实现采用纯 dataclass，不引入 NetworkX

## 状态

已接受（实现已锁定）。由实现事实追认，待用户确认。

计划 §21（`计划:1947`）把“图内部实现：纯 dataclass 还是引入 NetworkX 作为算法层”
列为“需要 ADR 决定”。实现已选定纯 dataclass，本 ADR 按 `计划:19` 追认该决策，
并给出未来引入 NetworkX 的准入条件。

## 背景

ADR 002 已确立 `CircuitGraph` schema v1 为唯一完整候选结构，并要求三种哈希
（`document_hash`、`graph_hash`、`topology_hash`）与双向兼容适配器。此后
阶段 2 与阶段 7 的实现进一步扩展了算法层：

- `circuit_ai/graph/model.py`：全部图实体为 `@dataclass(frozen=True)`
  （`ModelRef:75`、`ComponentInstance:180`、`Net:221`、`GraphDomain:255`、
  `GraphPort:315`、`ParameterVariable:349`、`CircuitGraph:422` 等）。
- `circuit_ai/graph/canonical.py`：`:154-172` 是自实现的 Weisfeiler-Lehman 风格
  迭代着色（`colors` 迭代 + `edge_colors` 排序），`:177` `exact_isomorphism_key`
  为显式同构键。
- `circuit_ai/graph/actions.py`：动作类型为不可变 dataclass（:36-110），
  执行器 `apply_graph_action:128` 返回新状态而非原地改图。
- `circuit_ai/graph/search.py`：`GraphSearchState:80`、`CircuitGraphBeamSearcher:454`、
  `ParetoBeamSearch:557` 全部基于 dataclass 状态与元组实现。
- `circuit_ai/graph/{adapters,migration,validation,primitives}.py` 提供转换、
  版本迁移、不变量校验与基础类型。
- 全仓 `grep -rn "networkx"` 在代码与依赖声明中 **0 命中**；该词仅作为文档参考
  出现在 `docs/adr/002-versioned-circuit-graph.md:127`（WL 哈希算法的出处）。
  即 `pyproject.toml` 的依赖列表中没有 `networkx`。

## 决策驱动因素

- 契约优先：`CircuitGraph` 必须可稳定 JSON 序列化、可版本迁移、可在报告与 replay 中
  长期保存，不能绑定算法库对象图。
- 不可变边界：搜索状态必须能安全比较、哈希与回溯，原地修改会破坏 beam/Pareto 层。
- 依赖最小化：计划 §1.3（`计划:90`）声明“不在早期自行重写完整工业 SPICE 内核”，
  体现出对外部依赖的克制；图算法同理。
- 哈希自控：结构哈希进入候选身份与 replay 分组键，必须完全可控、跨版本可复现，
  不能依赖第三方库的实现细节或版本行为。
- 现有闭环：AC 模板与 DC 电源候选必须无损转换（`计划:1580-1585`），引入邻接表
  视图会增加两套事实模型的风险。

## 采用的方案

`circuit_ai/graph/` 内以纯 dataclass + 元组作为唯一表示，算法层自实现：

1. 图实体与动作全部为 `frozen=True` dataclass；状态演化产生新对象。
2. 邻接关系在需要时由 `components[].terminals[].net_id` 现算（如
   `circuit_ai/graph/canonical.py:154-172` 的邻接表），不维护并行索引结构。
3. 结构指纹用自实现的 WL 风格迭代着色；严格同构用
   `exact_isomorphism_key`（`canonical.py:177`），并在 ADR 002 中明确
   “哈希不是同构证明”。
4. `networkx` 不进入 `pyproject.toml` 依赖。

## 未采用的方案

### 引入 NetworkX 作为算法层

可立即获得 shortest path、isomorphism（VF2）、WL hash、连通分量等成熟实现，减少
自研算法量。放弃理由（当前阶段）：

- 会引入第二套图事实模型：`CircuitGraph` 与 `nx.Graph` 之间需要持续同步，
  违反 §3.1 单一事实来源与 §16.2“不能建立第二套事实模型”。
- 序列化与版本迁移边界变模糊：NetworkX 节点/边属性不是版本化契约，
  不利于 `/report.json`、`replay.jsonl` 与训练数据快照。
- 哈希进入候选身份与数据分组键，依赖第三方实现会使历史 replay 的可复现性受
  库版本影响（`计划:1068-1075` 的数据准入与快照要求）。
- 现有需求（同构去重、连通性、beam 搜索）规模有限（元件数、节点数在
  `计划:895` 已设硬上界），自实现成本可控。

### 用邻接矩阵/邻接表作为主表示

放弃理由：丢失模型端子语义、域与端口身份，无法满足 §5.4 的图不变量与 §5.9
的类型化动作合法性判定。

## 未来引入 NetworkX 的准入条件

只有在**同时满足**以下条件时，才允许新增 ADR 修订本决策：

1. 出现自实现无法在可接受复杂度内解决的具体算法缺口，并给出基准证据
   （例如同构检查在真实候选规模下的耗时或正确性问题，而非假设）。
2. NetworkX 仅作为**算法层**使用：`CircuitGraph` 保持唯一事实来源，转换发生在
   一次性只读调用内，不产生长期并行的第二套图对象。
3. 结构哈希与候选身份**不切换**到 NetworkX 实现，或提供明确的哈希版本迁移器，
   使既有 replay/报告仍可复现（`计划:280` 的迁移器要求）。
4. 新增依赖对 §15.1 包依赖方向无破坏：`graph` 包仍不得依赖求解器、优化器、
   family 专家或 UI。
5. 有测试证明接入前后的候选集合、去重结果与哈希身份一致，或差异被显式迁移。

## 后果

正面结果：

- 图契约可稳定序列化与版本迁移，报告、replay 与训练数据共享同一结构。
- 不可变状态使 beam/Pareto 搜索的比较与回溯语义清晰。
- 无额外运行时依赖，包依赖方向与 §15.1 保持一致。

代价与限制：

- 图算法需自维护：WL 哈希为自实现（`canonical.py:154-172`），
  `exact_isomorphism_key` 有内部 net 数上限（`canonical.py:177`
  默认 `max_internal_nets=8`），超出规模时的严格同构能力有限。
- 若未来需要大规模图算法（谱方法、复杂子图挖掘、motif 发现，见 §10.7
  `计划:903-921`），可能需要重新评估本决策。

## 验证

- `grep -rn "networkx" circuit_ai/ pyproject.toml` 无命中，`networkx` 不在依赖列表中。
- `tests/test_circuit_graph.py:258,282,307,326` 验证结构哈希对顺序与内部标识不变、
  参数变化只影响 graph hash；`tests/test_graph_actions_search.py:72` 验证
  `exact_key` 忽略内部与实例标签。
- `tests/test_graph_actions_search.py:126,140` 验证 beam 搜索证书确定性与重复状态计数。

## 参考

- [NetworkX Weisfeiler-Lehman graph hash](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html)
- [VF2 同构图匹配算法（NetworkX）](https://networkx.org/documentation/stable/reference/algorithms/isomorphism.vf2.html)
- `docs/adr/002-versioned-circuit-graph.md`（同一结构的契约、哈希与迁移决策）
