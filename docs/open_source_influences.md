# Open-source architecture review

Review date: 2026-07-14

This project uses the following repositories as architectural references. No
source code was copied or vendored in this change; the implementation in
`circuit_ai/knowledge.py` and `circuit_ai/simulation_tasks.py` is original.

## CircuitGenome

- Repository: https://github.com/nayang-lab/CircuitGenome
- License: MIT
- Reviewed revision: `64fd17a05438e40e0e969400176d6dc9644228fb`
- Inspected areas: synthesizer models and loader, op-amp module/topology YAML,
  SPICE deck generation, and measurement handling.
- Adopted concept: represent reusable functional modules with canonical ports,
  then assemble topologies through named slots and explicit bindings.
- Local result: `circuit_ai/knowledge/power_topologies.yaml` and the validated
  loader in `circuit_ai/knowledge.py`.

## AnalogGym

- Repository: https://github.com/xiayouran/AnalogGym
- License: BSD-3-Clause
- Reviewed revision: `0a9d1390ade361e2b4a2d33181e22367edbb8afc`
- Inspected areas: amplifier and LDO circuit definitions, design variables,
  testbenches, and metric extraction scripts.
- Adopted concept: keep circuit structure, simulation conditions, measured
  metrics, and optimization variables as separate contracts.
- Local result: solver-independent `SimulationTask` and `MetricSpec` objects in
  `circuit_ai/simulation_tasks.py`.

## AnalogGenie

- Repository: https://github.com/xz-group/AnalogGenie
- License: MIT at review time.
- Review scope: public repository structure and graph-to-sequence approach.
- Current decision: retain as a future reference for learned graph codecs. It
  is not integrated because the current replay corpus does not yet justify
  replacing the existing proposer.

## Projects intentionally not reused

- AutoCkt was reviewed conceptually, but no code or data was reused because a
  clear repository-level license was not confirmed during this audit.
- ALIGN and OpenFASOC remain useful future references for hierarchy and
  layout-aware generation, but they are outside this ideal-schematic stage.

## Design boundary

The YAML knowledge base may describe new module compositions, but a production
is executable only when its `family` has a registered optimizer and its
`solver` names a supported physics model. This prevents a structural pattern
from being reported as electrically verified merely because it can be drawn.
