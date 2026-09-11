# ADR 002: 版本化 CircuitGraph 作为唯一候选结构

## 状态

已接受并进入 AC、DC 候选与报告边界。

## 背景

项目曾同时使用 `LinearCircuit`、`GraphCircuitTemplate`、`TopologyCandidate`、
YAML 元件列表和渲染器专用结构。它们分别满足局部流程，但不能共同表达多端口、
隔离域、模型身份、参数变量、额定值与来源。搜索、仿真、验证、replay 和导出因而
可能读取不同结构，无法证明最终原理图就是被验证的候选。

## 决策驱动因素

- AC 模板和 DC 功率候选必须无损进入同一结构契约。
- 元件实例、连接关系和模型定义必须分离，物理方程不能复制到每个候选中。
- 内部节点名和元件排列不能制造重复候选。
- 参数改变必须改变实例图身份，但不能改变拓扑身份。
- 已保存的报告、replay 和训练数据必须有显式版本迁移路径。
- 未识别数据不能在读取和重写时静默丢失。
- 现有闭环必须通过适配器渐进迁移，而不是一次性重写。

## 采用的方案

采用 `CircuitGraph` schema v1 作为唯一完整候选结构。核心实体为：

- `ModelRef`：模型 id、类型、版本、来源和端子契约。
- `ComponentInstance`：实例身份、端子到 net 的连接、参数绑定、额定值和属性。
- `GraphDomain`：物理域、参考网和对称隔离关系。
- `GraphPort`：外部稳定身份、方向、域和多端子定义。
- `Net`：域内连接、参考属性和用途。
- `ParameterVariable`：类型、单位、范围、初值、离散选择和尺度。
- `ProvenanceRecord`：知识、适配器或生成动作的来源证据。

图只保存模型引用，不保存 MNA stamp、Jacobian、噪声方程或导出实现。模型库和
能力注册表负责把 `model_id` 解析为具体物理行为。

### 三种哈希

- `document_hash`：规范排序后的完整文档哈希，保留展示身份与 provenance。
- `graph_hash`：忽略内部标识符和排列，但包含参数及额定值的结构指纹。
- `topology_hash`：进一步忽略参数值，用于候选拓扑去重。

结构指纹使用 Weisfeiler-Lehman 风格的迭代着色。它适合快速去重，但不是图同构
证明；发生哈希相同时，要求严格证明的流程仍需运行显式同构检查。

外部端口名称、方向和端子语义属于设计契约，因此进入拓扑哈希。内部 net 名、
`instance_id`、元件顺序和 net 顺序不进入结构身份。

### 版本和迁移

序列化文档必须包含固定 schema id 与整数 `schema_version`。读取路径先运行
`migrate_circuit_graph_payload()`，只允许通过注册的逐版本迁移升级；未来版本和
缺失迁移会明确失败。

v0 到 v1 的迁移把单一 `preferred_solver_id` 升级为有序候选列表，并补齐端口、
变量、来源和扩展容器。未知顶层字段被移入
`extensions.unknown_top_level_fields`，而不是丢弃。迁移不修改调用者输入。

新增字段优先采用有默认值的兼容变化。删除或改变语义的字段必须增加迁移器和
旧版 fixture；不能仅修改 `from_dict()` 使旧数据碰巧可读。

### 兼容适配器

- `LinearCircuit` 与 `CircuitGraph` 双向转换，覆盖 R、C、L、独立电压/电流源和 VCVS。
- MNA 地别名 `0`、`gnd`、`GND` 在导入时规范为域参考网。
- `TopologyCandidate` 与 `CircuitGraph` 双向转换，正式保存 solver id、参数变量、
  端口方向、隔离域和 PBDL 来源。
- `CircuitTemplate.to_graph()` 是 AC 模板统一出口。
- YAML functional module 在加载时直接成为通过验证的 CircuitGraph fragment。

旧容器暂时保留为 compatibility adapter，不能继续承载新的核心语义。

## 未采用的方案

### 让每个后端维护自己的图

局部实现简单，但搜索图、仿真图和原理图会持续漂移，无法建立端到端证据链。

### 把方程嵌入每个候选

可自包含，但模型修复和版本升级会复制到海量候选，且同一器件可能出现不同方程。

### 只使用序列化 JSON 作为内部对象

灵活但缺少不可变边界、类型约束和结构化验证，错误会推迟到求解阶段。

### 把结构哈希当作最小性或同构证明

哈希只能加速候选去重；它不能证明两个一般图同构，也不能证明搜索空间已穷尽。

## 结果

正面结果：

- AC 和 DC 候选共享同一可版本化图结构。
- 报告可保存验证所用图及文档、实例、拓扑三类身份。
- 专家知识、参数优化器、仿真器和输出器可以围绕稳定公共 API 插件化。
- 内部重命名和排列不再污染候选集合。
- replay 与训练数据具备明确升级入口。

代价与限制：

- 当前非线性模型、行为模型层次和多物理端子仍需后续模型库扩展。
- 未知字段保留目前覆盖图文档顶层；嵌套破坏性变化必须由版本迁移显式处理。
- DC 专用公式和拓扑专用渲染器尚未全部改为通用 CircuitGraph backend。
- 结构哈希碰撞需要后续同构检查器处理。

## 验证

`tests/test_circuit_graph.py` 覆盖：

- schema v1 JSON round-trip 与三种哈希稳定性。
- v0 到 v1 迁移、未来版本拒绝、未知字段保留和输入不变性。
- Hypothesis 生成的数值、复杂源、嵌套扩展、排列和内部重命名。
- 参数值改变不影响 topology hash。
- 所有现有 AC 模板及支持的 Linear MNA 原语双向无损转换。
- Buck、Boost、SEPIC、Flyback 候选和隔离域转换。
- 地别名规范化前后的 MNA 响应等价性。

## 参考

- [CIRCT HW dialect](https://circt.llvm.org/docs/Dialects/HW/)
- [CIRCT HW rationale](https://circt.llvm.org/docs/Dialects/HW/RationaleHW/)
- [SKiDL Circuit API](https://devbisme.github.io/skidl/api/html/rst_output/skidl.circuit.html)
- [NetworkX Weisfeiler-Lehman graph hash](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash.html)
- [Protocol Buffers: updating a message type and unknown fields](https://protobuf.dev/programming-guides/proto3/)
- [KiCad schematic file format](https://dev-docs.kicad.org/en/file-formats/sexpr-schematic/)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/en/latest/)
