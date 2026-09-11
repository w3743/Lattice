# 端口行为描述语言

PBDL 用来描述“我要什么电路”，而不是“怎么实现电路”。核心结构是：

```text
Ports + Analyses + Targets + Constraints + OperatingPoint
```

## 严格翻译

`to_circuit_ai_spec(spec)` 只接受单个可执行合成问题。它适合低通、高通、带通、放大器、输入阻抗、跨阻放大器等当前 core 已支持的单阶段任务。

如果 spec 包含多个 analysis/target，或包含 DC、输出阻抗、晶体管等尚未实现能力，它会返回明确错误。

## 多阶段规划

真实电路往往同时包含多个目标，例如：

```text
光电二极管 TIA = AC 跨阻带宽 + DC 偏置点 + 噪声/功耗约束
传感器接口 = 低通抗混叠 + 增益 + 输出阻抗匹配
```

这类规格用 `to_circuit_ai_plan(spec)` 拆解：

```python
from 端口描述语言 import load_spec, to_circuit_ai_plan

spec = load_spec("examples/photodiode_tia.json")
plan = to_circuit_ai_plan(spec)

for stage in plan.stages:
    print(stage.name, stage.status, stage.analysis_kind, stage.target_kind)
```

每个 stage 会是：

- `supported`: 已翻译成 `circuit_ai.SynthesisSpec` dict，可直接合成
- `unsupported`: 当前 core 尚不支持，但保留 `missing_capability` 和 `suggestion`

例如 `photodiode_tia.json` 当前会拆成：

```text
stage 1: transimpedance + amplifier -> supported
stage 2: dc_transfer + dc           -> unsupported, waiting for DC operating-point solver
```

`sensor_interface.json` 当前会拆成：

```text
stage 1: voltage_transfer + filter -> supported
stage 2: output impedance target   -> supported
```

这让上层模型可以先合成已支持的子电路，同时明确知道还缺哪些专家。

## 命令行执行

仓库根目录提供 `pbdl_synth.py`：

```powershell
python pbdl_synth.py 端口描述语言/examples/sensor_interface.json --out outputs/sensor_interface_pbdl
python pbdl_synth.py 端口描述语言/examples/photodiode_tia.json --out outputs/photodiode_tia_pbdl
```

输出目录包含：

- `plan.json`: PBDL 分阶段翻译结果
- `pbdl_report.json`: 每个阶段的执行状态和最佳候选摘要
- `stage_N_*`: 每个可执行阶段的 `report.json`、`best.spice`、`best.svg` 和候选原理图

只想看规划、不执行合成时：

```powershell
python pbdl_synth.py 端口描述语言/examples/sensor_interface.json --plan-only --out outputs/sensor_interface_plan
```
## Canonical Boundary

PBDL is the only public design specification in this repository. The
canonical object is:

```text
CircuitSpec = ports + relations + analyses + targets + constraints
              + operating_point + optimization
```

The browser workbench writes this structure directly. The old workbench JSON
shape is accepted only by `circuit_ai.pbdl_boundary` as a compatibility input
and is normalized before topology search or optimization. `circuit_ai`'s
`SynthesisSpec` is an internal execution IR, not a user-facing specification.
