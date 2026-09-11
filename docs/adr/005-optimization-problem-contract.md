# ADR 005: 统一优化问题、预算与真值证据

## 状态

已接受。首个实现覆盖 AC 模板和理想 Boost、Buck、SEPIC、Flyback 参数搜索。

## 背景

项目过去有两套参数搜索路径：AC 模板直接调用 SciPy 并返回模板专用结果；四种理想电源
又各自构造边界、调用 SciPy 并返回 `BoostParameters` 或 `FlybackParameters`。它们没有
共同的变量类型、单位/缩放边界、预算语义、缓存统计或最终复算记录。结果是优化器无法
替换，报告也无法回答一个候选实际消耗了多少求解调用、变量在什么物理范围内搜索，以及
最终接受的是搜索模型还是独立真值。

## 决策

新增 `circuit_ai.optimization`，作为所有参数优化器的稳定契约层。

### 问题与变量

`OptimizationProblem` 包含稳定 problem id、版本、目标 id、约束 id、元数据和一组
`OptimizationVariable`。变量显式区分：

- `continuous`：连续物理参数；
- `integer`：整数物理参数；
- `categorical`：离散型号或策略；
- `derived`：由其他变量推导、不可由优化器直接改变的量。

每个变量声明单位、物理上下界、可选初始值、来源路径和缩放方式。线性与 `log10` 缩放
只属于求解器坐标，`decode()` 产生的始终是带有物理语义的值。整数和类别坐标使用求解器
原生整数约束，不能通过字符串或变量名猜测类型。

### 统一 DE 驱动

`DifferentialEvolutionDriver` 是当前默认连续优化器。它只消费 `OptimizationProblem`、
目标函数和 `DifferentialEvolutionConfig`，不依赖拓扑族。配置使用确定性 `rng`，保留
SciPy 的原生边界、整数性和局部 polish；配置错误在 PBDL 参数边界处明确报错。

电源族仍拥有物理公式和可行初始点的知识，但只负责：编译变量范围、提供目标函数、把
decode 后的值转换为其兼容结果对象。它们不再直接拥有 SciPy 调用循环。解析可行点作为
`OptimizationProblem.initial` 进入通用驱动，这保证有限预算下首先评估已知可行候选，
而不是绕开统一优化器。

### 预算、缓存和真值

`EvaluationBudget` 目前支持最大目标调用次数和最长时间。驱动在每次未命中缓存的调用前
检查预算；耗尽后返回当前最佳已评估坐标及 `budget_exhausted` 状态。结果对象记录：

- 求解器与版本；
- decode 后的物理变量和最终目标；
- 搜索目标与最终 truth 目标；
- 真实评估次数、缓存命中、真值复算次数、耗时和预算；
- 完成状态和可审计消息。

搜索模型和接受模型必须在接口上分开。当前理想电源的 truth evaluator 仍是同一理想
公式，随后主流程会通过统一 `SimulationResult` 和 `ConstraintReport` 复核；这不能被
解释为高保真独立验证。AC 路径同样保存优化记录，并在既有 MNA 终检后裁决。

`optimization_run` 写入电源 `report.json`、候选记录和 `replay.jsonl`，使训练数据可以
区分搜索证据与最终裁决。

### 保真度调度

`FidelitySchedule`、`FidelityStage` 与 `FidelityScheduleRun` 将“哪些已排序候选进入
下一种物理模型”从编排代码中的切片操作变为版本化证据。阶段只有两种角色：
`screen` 用于低成本筛选，`truth` 用于接受前的权威复核；最后一个阶段必须为 `truth`。

AC 默认 schedule 是：所有候选完成 `optimization_model` 参数优化，然后仅将 `top_k`
候选提升到 `linear_frequency_domain` 的 Linear MNA 复核。每个最终 AC 结果都保存完整
schedule、每阶段输入/选中数和候选 ID。调度器只消费既有排序，不决定电路优劣，也不替代
`ConstraintReport` 的裁决。

这一步固化了已有的 Top-K MNA 行为并使其可审计，但尚未减少拓扑枚举或参数优化本身的
成本。功率链路目前只有 `ideal_averaged` 后端，不能把它标记为高保真 SPICE；待开关级
模型或对应 SPICE 后端可用后，才可配置为真实的第二阶段。

## 后果

正面结果：

- 新增黑盒优化器可复用变量、预算、结果和报告协议，不需要修改 orchestrator。
- 参数的单位和 log 坐标不会混入物理模型。
- 有限预算、早停与缓存成为可见的工程约束，不再是隐式实现细节。
- 专家公式保留为先验和快速模型，而不是优化框架的控制流。

当前限制：

- 驱动暂时只执行单目标 DE；Pareto archive 仍由候选选择层处理。
- 通用多保真 backend 解析、surrogate 成本模型和电源/SPICE Top-K 队列尚未接入本契约。
- AC 的 Top-K Linear MNA 已接入版本化 schedule；通用 backend 解析、多保真预算和
  电源/SPICE 阶段仍未接入。
- Torch 局部优化还未实现为 `OptimizationProblem` adapter。
- `BoostParameters` 等兼容结果类型仍在主流程中，后续需要收敛为通用参数结果加领域视图。

## 验证

- 覆盖连续、整数、类别、派生、单位、线性/log10 encode/decode 的单元测试。
- 覆盖 DE 的整数/类别变量、缓存、严格评估预算和独立 truth evaluator。
- Boost 和隔离 Flyback 端到端测试断言 `optimization_run` 的 problem id 与真值复算证据。
- 关键 AC、电源、能力注册和优化测试共同回归。
