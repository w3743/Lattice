# ADR 006: 单位库采用 Pint

## 状态

已接受（实现已锁定）。由实现事实追认，待用户确认。

本 ADR 记录并确认一个既成事实：主流程的单位解析与量纲转换已经依赖 `pint`，
而计划文档 §21（`计划:1946`）仍把“采用 `pint` 还是项目内最小 Quantity 实现”
列为“需要 ADR 决定”。此处分歧按 `计划:19` 的要求补记决策。

## 背景

计划要求所有核心契约“字段有单位和方向定义”，且约束比较“必须使用单位转换后的
标准量，不能直接比较裸浮点数”（§5.8，`计划:473-474`）。这要求一个能同时处理
量纲兼容性、单位别名、对数单位、偏移单位（°C）和换算系数的单位系统。

实现已经选定 Pint 并把它隔离在单一边界内：

- `circuit_ai/units.py:12` `import pint`；:19 `UNIT_REGISTRY = pint.UnitRegistry(autoconvert_offset_to_baseunit=False)`；
  :45,73,77 分别处理 `UndefinedUnitError`、`DimensionalityError`、`OffsetUnitCalculusError`。
- `circuit_ai/units.py:1-6` 明确该模块的定位：“Serialized contracts keep units as strings.
  Pint is deliberately isolated in this module so schemas do not expose library-specific
  objects and every layer uses the same aliases and failure semantics.”
- `端口描述语言/functions.py:16` 同样直接 `import pint`，即 PBDL 侧也在使用同一库。
- `pyproject.toml:12` 声明依赖 `Pint>=0.25.3,<0.26`。

单位安全比较、容差合并与报告的单位转换均建立在该模块之上，例如
`tests/test_units_and_constraints.py:41,51,71,102,182` 覆盖别名、对数单位严格匹配、
等值约束换算与容差合并、边界算子与电压换算往返。

## 决策驱动因素

- 量纲安全：不同量纲必须报错而不是静默比较（`计划:1920` 风险项“单位/方向错误导致假通过”）。
- 别名统一：`ohm`/`ohms`/`Ω`、`mm2`/`cm2` 等必须在所有层使用同一套语义。
- 对数与偏移单位：`dB` 需严格匹配，`°C` 需按偏移单位规则处理。
- 序列化边界：schema 只保存单位字符串，不暴露库对象，避免契约与库版本绑定。
- 迁移成本：更换单位库会同时影响 `circuit_ai/units.py`、`端口描述语言/functions.py`
  以及全部单位相关测试与已保存报告。

## 采用的方案

采用 `pint` 作为唯一单位库，并强制以下边界约束：

1. Pint 只允许出现在 `circuit_ai/units.py`（与 PBDL 的 `端口描述语言/functions.py`）
   内部；其他模块只调用本项目的单位 API。
2. 所有序列化契约（`MetricSpec`、`ConstraintSpec`、`SimulationResult.scalars` 等）
   以字符串保存单位，反序列化时经注册表规范化。
3. 项目自定义单位与别名（如 `CNY` 货币、`Ω`/`ohms` 等）在 `units.py` 内集中定义，
   不散落到各调用点。
4. 单位错误统一抛出项目异常 `UnitContractError`（`circuit_ai/units.py:15`），
   使上层无需捕获 Pint 异常类型。

## 未采用的方案

### 项目内最小 Quantity 实现

自研一个只覆盖 R/C/L/V/A/Hz/s 等少数单位的最小实现，可去掉第三方依赖并完全控制
语义。放弃理由：

- 需要自行实现量纲代数、别名表、对数单位、偏移单位与浮点容差，属于重复造轮子且
  正确性风险高（正是 §20 风险登记中“单位/方向错误导致假通过”的高影响路径）。
- 计划在 §8.2/§12.3 明确要求扩展噪声谱、结温、面积、价格、封装等新单位；最小实现
  会在每个新领域重复扩展。
- 现有 9 项单位与约束测试（`tests/test_units_and_constraints.py`）已围绕 Pint 语义
  建立，替换会造成大面积回归。

### 不使用单位库、仅靠命名约定

放弃理由：无法在运行时阻止 `mV` 与 `V` 直接比较，直接违反 §5.8 的单位安全要求。

## 后果

正面结果：

- 量纲兼容性、别名和对数/偏移单位由成熟库保证，约束引擎可专注于比较语义。
- 单位字符串契约与库对象解耦，报告与 replay 的 schema 不随库版本变化。

代价与限制：

- 引入一个第三方运行时依赖，属于计划 §21 未决项中影响面最大的一项。
- `pint` 版本区间被锁定为 `>=0.25.3,<0.26`，跨大版本升级需要回归单位测试。
- **现状阻塞**：本机当前未安装 `pint`，导致 `import circuit_ai` 失败、`python -m pytest`
  共 47 个测试模块无法收集（`circuit_ai/units.py:12` → `ModuleNotFoundError: No module named 'pint'`）。
  项目又无虚拟环境目录，需显式安装依赖。该项已登记为阶段 0 待恢复任务
  （计划文档阶段 0 任务区）与 `docs/plan_deltas.md` 的待拍板事项。

## 验证

- `circuit_ai/units.py` 是仓库中唯一出现 `import pint` 的业务模块（`端口描述语言/functions.py`
  为 PBDL 侧对等边界）。
- `tests/test_units_and_constraints.py:41,51,57,71,102,115,155,171,182` 覆盖别名、
  对数单位、序列化 round-trip、容差合并、hard/soft/objective 语义、缺失指标、
  顺序不变性与换算往返。
- 安装 `pint` 后 `python -m pytest -q tests/test_units_and_constraints.py` 必须全通过，
  这是本 ADR 的验收前提。

## 参考

- [Pint documentation](https://pint.readthedocs.io/)
- [Pint: offset units and `autoconvert_offset_to_baseunit`](https://pint.readthedocs.io/en/stable/user/nonmult.html)
