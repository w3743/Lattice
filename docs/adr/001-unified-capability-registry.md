# ADR 001: 统一能力注册与可解释解析

## 状态

已接受并进入主流程。

## 背景

功率设计主流程曾直接导入 Buck、Boost、SEPIC 和 Flyback 四个优化器，
并通过 `_optimizer_for()` 按 `candidate.family` 分派。验证器、SVG 渲染器和
KiCad 导出器也由 `pipeline.py` 直接调用。这使新增后端必须修改编排层，
候选元数据中的 `solver` 字段无法被校验，缺少能力时只能得到拼接的错误文本。

项目还存在 Linear MNA、ngspice、解析功率模型等不同保真度实现。它们需要共享
同一发现和描述机制，但执行协议仍应按职责拆分。

## 决策驱动因素

- PBDL 和 UnifiedIR 必须保持唯一公共输入链路。
- 主流程不能知道具体拓扑优化器、验证函数或导出函数。
- 后端选择必须同时考虑分析、模型和求解器，而不能只看拓扑族。
- 缺失能力必须可机器读取，并保留每个候选实现的拒绝理由。
- 内置能力必须确定性加载；第三方代码不能在导入核心包时隐式执行。
- 当前专用公式需要保留为迁移适配器，不能为了抽象而重写已验证数值行为。

## 采用的方案

采用一个 `CapabilityRegistry` 保存所有职责的注册项，而不是分别建立互相可能失配的
Solver、Validator 和 Renderer 注册表。

每个 `CapabilityRegistration` 由两部分组成：

1. 不可变的 `CapabilityTarget`，描述 id、版本、职责、支持的分析、模型、求解器、
   输出格式、保真度、优先级和来源。
2. 可执行 implementation，实现与职责对应的公开 Protocol。

`CapabilityResolver` 接收 `CapabilityRequest`，按以下顺序解析：

1. 匹配职责。
2. 匹配迁移期拓扑族约束。
3. 匹配候选声明的 solver id。
4. 检查分析集合与元件模型集合是否被覆盖。
5. 检查输出格式。
6. 仅在静态条件全部匹配后探测运行时可用性。
7. 按优先级、capability id 和版本确定性排序。

解析成功时同时返回选中能力和备选能力。解析失败时返回
`CapabilityGap`；无法组装完整候选执行计划时抛出包含多个 gap 的
`CapabilityPlanError`。错误 JSON 保留请求和逐注册项拒绝理由。

内置能力通过 `build_default_registry()` 显式注册。外部插件使用
`circuit_ai.capabilities` entry-point group，但只有调用
`discover_entry_points()` 或设置 `include_external_plugins=True` 时才加载。

## 当前适配范围

- 四个理想功率参数优化器。
- 理想功率约束验证适配器。
- 功率 SVG 渲染和 KiCad 项目导出适配器。
- Linear MNA AC 分析适配器。
- ngspice AC 验证适配器及运行时可用性检查。

`family` 目前仍用于防止专用公式误用于不兼容候选。它是迁移期保护条件，
不是长期后端身份。长期匹配应以 `CircuitGraph` 中的模型、分析和功能要求为主。

## 未采用的方案

### 保留主流程字典分派

实现最少，但每增加一种拓扑都必须修改编排代码，无法形成插件边界，也无法解释
solver 元数据是否真实可执行。

### 建立多个独立注册表

每个职责看似更简单，但 optimizer、simulator、validator 和 exporter 可能分别匹配
成功却无法组成兼容计划。单一注册中心可共享同一能力描述和诊断模型。

### 导入时自动发现所有插件

使用方便，但核心包导入会执行未知第三方代码，测试结果和启动环境不再确定。
因此插件发现保持显式。

## 结果

正面结果：

- `pipeline.py` 不再直接导入四个电源优化器、验证函数或专用输出函数。
- 新优化器可通过注册和优先级注入，不修改主流程。
- 报告、replay 和 capability manifest 能追溯实际执行实现。
- 缺失分析、模型、solver 或运行时依赖时有结构化证据。
- 内置能力和外部插件遵守同一注册契约。

代价与限制：

- 当前功率适配器内部仍调用专用公式和专用渲染器。
- 一个注册表不等于一个统一物理结果模型；后续仍需 `SimulationRequest`、
  `SimulationResult`、`CircuitGraph` 和通用约束引擎。
- capability 版本目前用于身份和审计，尚未实现依赖版本范围求解。
- 注册机制本身不会增加可搜索拓扑空间。

## 验证

`tests/test_capabilities.py` 覆盖：

- 重复 id 和职责方法契约。
- 优先级选择及备选项记录。
- family、solver、analysis 和 model 的结构化拒绝理由。
- 静态不匹配时不触发外部可用性探测。
- Linear MNA 与四个功率优化器的能力覆盖。
- entry point 只在显式请求时加载。
- 注入高优先级优化器后主流程实际使用该实现。
- 主流程不存在 `_optimizer_for()` 或具体 Boost 优化器引用。

## 参考

- [Python Packaging User Guide: Creating and discovering plugins](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- [Python documentation: importlib.metadata](https://docs.python.org/3/library/importlib.metadata.html)
- [IBM Quantum: BackendV2 and Target](https://quantum.cloud.ibm.com/docs/en/guides/custom-backend)
- [MADR: Markdown Architectural Decision Records](https://github.com/adr/madr)
- [OpenFASOC documentation](https://openfasoc.readthedocs.io/en/latest/)
