# ADR 003: 统一仿真请求、结果与后端执行

## 状态

已接受并进入 AC、DC 合成主流程。

## 背景

项目曾有三套互不兼容的仿真接口：`AnalysisRequest/ACAnalysisResult`、功率优化器返回的
`BoostOperatingPoint/TransientTrace`，以及 `NgspiceVerifier/SpiceVerificationResult`。
主流程直接读取拓扑专用工作点，失败主要表现为异常、NaN 或 verifier 文本。后端无法
共享能力解析、缓存、状态分类和证据报告。

## 决策驱动因素

- 规格任务、单次后端执行和约束裁决必须分层。
- DC、AC、瞬态、PSS、噪声和多端口张量需要共用结果容器。
- 后端必须区分不支持、不可用、数值失败和超时。
- 结果必须绑定请求、图、后端、模型版本、保真度和运行成本。
- 参数优化可以使用快速模型，但最终候选需要独立物理后端复核。
- CircuitGraph 到矩阵或网表的编译必须可缓存。
- 可选依赖缺失不能使核心包导入失败。

## 采用的方案

### 三层语义

- `SimulationTask`：规格层描述要检查的分析和指标，可编译为一个或多个 request。
- `SimulationRequest`：一次后端执行的完整输入，包括分析、sweep、激励、条件、参数、
  observable、保真度、随机种子和预算。
- `SimulationResult`：后端产生的原始证据，不直接宣判设计约束。

`voltage_transfer`、`transimpedance` 和阻抗等规格分析在执行边界规范为
`small_signal_ac`。固定输入的 `dc_transfer` 保留固定工作点语义；只有显式
`dc_sweep` 才产生扫描。DC 请求忽略不相干的正弦 waveform，优先采用目标输入电压。

### 数据契约

请求和结果使用独立 schema id、v1 版本与确定性 SHA-256 哈希。读取时验证调用者提供
的 `request_hash` 和 `evidence_hash`，防止参数或结果被修改后仍沿用旧证据身份。

结果数据分为：

- `Quantity`：带单位、可选不确定度的标量。
- `Waveform`：带一个或多个 `ResultAxis`、显式 shape 和扁平值的实数或复数张量。
- `Diagnostic`：code、severity、message、path 和结构化 details。

Waveform 的轴长度、shape 与值数量必须严格一致。复数通过 CircuitGraph 共用的 JSON
编码往返，不使用 NaN 表示状态。

### 状态模型

`SimulationStatus` 包含：

- `passed`：后端执行成功且证据契约完整。
- `failed`：已执行但编译、矩阵、收敛、进程或解析失败。
- `unsupported`：分析、observable 或模型不在后端能力内。
- `unavailable`：可执行文件或可选运行库不可用。
- `timeout`：执行预算耗尽或结果完成时已超预算。

`nonconverged`、`singular`、`parse_failed` 等作为稳定 diagnostic code 保存。任何非
passed 状态都不能进入约束通过结论。

### 后端协议与执行器

Simulator Protocol 采用：

```python
compiled = backend.compile(graph)
result = backend.simulate(compiled, request)
```

每个后端按 `graph_hash` 缓存编译结果。`SimulationExecutor` 统一处理 availability、
compile 异常、unsupported model、未捕获 backend exception、请求关联校验和预算超限。
外部 ngspice 使用 subprocess timeout 真正终止超时进程；内存内后端完成后若超预算，
结果改标 timeout，不能作为有效证据。

### 当前后端

- `LinearMNASimulatorBackend`：R/C/L、独立源、VCVS 的 small-signal AC。
- `TorchMNASimulatorBackend`：批量矩阵 solve 的可微 AC；补齐电流源 stamp 与跨阻。
- 四个 `AnalyticPowerSimulatorBackend`：注册时注入参数类型和 DC 方程，不在后端内部按
  family 分派；支持固定工作点与输入电压 sweep。
- `NgspiceSimulatorBackend`：线性 CircuitGraph 的 AC、DC OP/DC sweep、transient；
  解析 ASCII rawfile，并区分 unavailable、timeout、singular 和 nonconverged。

当前机器检测到嘉立创仿真的 `ngspice.dll`，但没有独立 `ngspice.exe`。共享库存在不等于
subprocess backend 可执行，因此 capability 正确报告 unavailable。未来可增加独立的
shared-library backend，不能伪装成命令行后端。

### 多保真迁移

AC 参数优化内环继续使用模板快速响应。Top-K 结果必须通过注册的 Linear MNA 再执行一次，
并把请求、结果与 capability resolution 写入报告。功率优化器可继续内部计算旧工作点，
但 orchestration、验证、任务评价、排序和 replay 只消费统一 SimulationResult 标量。
旧 `dc_solution.json` 由统一结果派生，作为兼容输出保留。

## 未采用的方案

### 给每种分析定义独立 Result 类

类型直观，但 AC、噪声、S 参数、Monte Carlo 和多物理结果会迅速形成无法组合的类层次。

### 用 NaN 和异常表示所有失败

实现简单，但无法区分模型不支持、工具未安装、数值不收敛和预算耗尽，也无法训练失败
预测器或生成可信 capability gap。

### 让优化器返回值直接作为最终验证

速度最快，但优化模型与裁决模型完全相同，系统性模型误差不会被发现。

### 发现 DLL 即宣称 ngspice 可用

DLL 与 CLI 有不同 ABI、初始化和回调协议。没有对应 shared-library adapter 时不能执行。

## 结果

正面结果：

- AC 与 DC 主流程保存同一版本化仿真证据。
- orchestration 不再 import 或读取 `BoostOperatingPoint`。
- 新后端只需注册 compile/simulate 能力，不需修改候选流程。
- replay 可学习失败类别、后端差异、运行成本和模型版本。
- DC UI 中的无关频率设置不会进入 DC request。

代价与限制：

- ngspice 线性适配器尚未支持真实半导体 model binding、噪声和 S 参数。
- Torch backend 当前统一执行只覆盖电压增益与跨阻，阻抗仍回退 Numeric MNA。
- 内存内 timeout 是完成后的预算裁决，不能中断底层 BLAS/Torch kernel。
- 约束裁决已由 [ADR 004](004-declarative-metrics-and-constraints.md) 的声明式引擎接管；
  `SimulationResult` 继续只承担原始证据职责。

## 验证

- Request/Result JSON round-trip、哈希防篡改、复数和多轴 shape 属性测试。
- Linear MNA 与旧分析器逐频点一致并验证 compile cache。
- Torch MNA 与 Numeric MNA 逐频点一致，现有可微测试保持通过。
- 四个功率解析 backend 可被能力解析，Boost 结果与旧 golden 方程一致。
- 假 ngspice 进程覆盖 AC rawfile、unavailable、timeout、nonconverged、DC 和 transient 命令。
- 同一 RC 的解析传函、Numeric MNA 和 ngspice raw 误差小于测试门槛。
- 故障注入覆盖 availability、unsupported model、compile failure、backend exception、
  result/request mismatch 和内部超预算。

## 参考

- [ngspice User's Manual](https://ngspice.sourceforge.io/docs/ngspice-html-manual/manual.xhtml)
- [Xyce Users' Guide](https://xyce.sandia.gov/files/xyce/Xyce_Users_Guide_7.8.pdf)
- [FMI 3.0 specification](https://fmi-standard.org/docs/3.0/)
- [OpenMDAO Case recording](https://openmdao.org/newdocs/versions/latest/basic_user_guide/reading_recording/basic_recording_example.html)
