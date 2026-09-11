# ADR 004: 声明式指标、单位与约束裁决

## 状态

已接受并进入 AC、Boost、Buck、SEPIC 与 Flyback 主流程。

## 背景

项目过去同时存在频响评分、`SimulationTask` 比较和 Boost 命名的专用验证器。它们对
缺失数据、单位换算、容差、硬约束和优化目标的解释并不完全一致。候选排序还使用固定
大惩罚把不可行设计塞回一个标量分数，既不能表达“未知”和“执行失败”，也会让权重
改变可行性语义。

统一仿真合同已经明确：`SimulationResult` 是证据，不直接宣判设计是否满足需求。因此
需要在 PBDL/IR 与候选排序之间建立独立、可序列化、与拓扑无关的指标和约束层。

## 决策驱动因素

- 每个端口、物理量和分析上下文必须独立编译与评价。
- 数值比较前必须先完成显式单位换算，物理值与优化缩放不能混为一谈。
- 硬约束、软约束和优化目标具有不同语义，不能只靠权重大小区分。
- 缺失、后端不支持、执行错误和已测得但不满足必须保留不同状态。
- 候选先按可行性分层，再在同层内做多目标取舍。
- 报告必须能追溯到 PBDL 来源、指标提供器和仿真/图证据。

## 采用的方案

### 单位合同

`circuit_ai.units` 使用受控 Pint registry 解析和换算单位。项目增加 `CNY` 成本维度及
`mm2` 等输入别名。对数单位采用严格策略：`dB` 只允许与相同对数单位比较，不把它
隐式当作普通无量纲数。容差在约束单位中计算，比较值先从指标单位转换到约束单位。

### 指标合同

`MetricSpec` 声明指标 ID、提供器、证据源、物理量、单位、分析类型、归约方式和选择器。
`MetricValue` 保存可用状态、标量值、单位、诊断与证据引用；`MetricSet` 按稳定 ID 汇总。
`MetricEngine` 只负责调用注册提供器并把提供器异常转换为结构化错误证据。

内置提供器覆盖：

- 仿真标量与 DC 端口电压、电流、功率、效率；
- AC 幅相误差、指定频点值和带宽；
- 瞬态稳态窗口、纹波、RMS 与极值；
- 器件应力；
- CircuitGraph 元件数量、成本、面积、库合法性和参数值。

### 约束编译与裁决

`compile_constraint_program()` 将 PBDL targets、每端口 variable constraints、结构/资源
限制、参数范围、额定值和显式 objectives 编译为唯一的 `ConstraintProgram`。

`ConstraintSpec` 支持 `equal`、`minimum`、`maximum`、`interval`、`minimize` 和
`maximize`，并显式区分：

- `hard`：决定可行性；
- `soft`：产生归一化加权违反量，但不改变硬可行性；
- `objective`：记录有方向的目标值，供 Pareto 排序使用。

等值和边界比较支持绝对容差与相对容差之和。`ConstraintEvaluator` 输出每条约束的
需求、实测值、单位、有效容差、归一化违反量、来源路径、提供器、证据等级和诊断。

硬约束裁决为：

- 全部有证据且通过：`verified_feasible`；
- 有证据且至少一条失败：`known_infeasible`；
- 指标缺失或不支持：`indeterminate`；
- 指标提供或执行错误：`execution_failed`。

### 候选排序

固定百万惩罚被删除。候选按以下稳定顺序排序：

1. 可行性层级；
2. 不可用硬约束数量；
3. 硬约束归一化违反总量；
4. 同层 Pareto rank 与 crowding distance；
5. 软约束惩罚；
6. 不承载可行性语义的确定性 tie-breaker。

旧 `circuit_ai.validation.validate_boost()` 门面已删除。功率能力注册使用
`DeclarativeConstraintValidator`；AC 最终 MNA 复核也生成同一 `ConstraintReport`。
`SimulationTask` 只保留兼容视图，指标和比较实现复用上述合同。

## 未采用的方案

### 为每类拓扑保留独立验证器

短期直接，但会把端口和资源语义复制到每个拓扑族，新增拓扑仍需修改裁决代码。

### 把不可行候选加一个极大常数

实现简单，但常数会污染目标尺度，无法区分未知、失败和已知不可行，也不能给出边界内
可解释排序。

### 自动把所有无量纲值互相转换

会错误混淆比例、百分比、弧度和 dB。当前选择在输入边界显式声明单位并拒绝含糊换算。

## 结果

正面结果：

- RC、Boost 与 Flyback 使用同一指标、约束和报告模型。
- 四端口约束保持独立 metric ID，不会因变量同名而互相覆盖。
- 仿真缺失和失败不会被静默跳过或伪装成 NaN。
- 成本、面积、元件数量和性能目标可共同进入 Pareto 评价。
- GUI、CLI、replay 和未来学习排序器可消费同一机器可读报告。

代价与限制：

- 当前器件应力指标只覆盖图额定值和已有仿真证据，尚未包含完整热模型和半导体 SOA。
- AC 内环仍保留旧标量 score 作为优化器目标；最终候选已由统一报告复核。
- `SimulationTask` 兼容视图仍存在，后续统一 orchestrator 时再删除旧序列化表面。

## 验证

- 单位兼容、换算、绝对/相对容差和非法单位测试。
- hard/soft/objective 及 missing/unsupported/error 状态测试。
- 约束顺序不变性和四端口独立性测试。
- RC 最终 AC 误差与旧评分逐值交叉验证。
- Boost 与 Flyback 端到端使用同一声明式验证能力。
- 可行性优先、Pareto rank、crowding 和 tie-breaker 排序测试。

## 参考

- [Pint: Defining Quantities](https://pint.readthedocs.io/en/stable/user/defining-quantities.html)
- [Pint: Non-multiplicative Units](https://pint.readthedocs.io/en/stable/user/nonmult.html)
- [OpenMDAO: Adding Constraints](https://openmdao.org/newdocs/versions/latest/features/core_features/adding_desvars_cons_objs/adding_constraint.html)
- [OpenMDAO: Scaling Variables](https://openmdao.org/newdocs/versions/latest/features/core_features/working_with_components/scaling.html)
- [FMI 3.0 specification](https://fmi-standard.org/docs/main/)
