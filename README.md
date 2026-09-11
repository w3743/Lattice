# Lattice

**From a circuit requirement to a verifiable schematic.**

Lattice takes a port/behaviour requirement plus an element library and produces
a candidate circuit, an optimised parameter set, and an auditable evidence chain
that records how each conclusion was reached:

```text
port behavior + element library -> candidate circuits -> optimized SPICE netlist
```

Two names are in play during the rename: **Lattice** is the project, while the
import package remains `circuit_ai` and serialised schema ids remain
`circuit_ai.*`. Those are versioned wire contracts, so they are only renamed
together with a migration.

PBDL is the single public specification boundary. The browser workbench and
the command-line synthesizer normalize input into the PBDL CircuitSpec
(ports, relations, analyses, targets, constraints, operating_point, and
optimization) before invoking the internal execution IR. Older two-port JSON
examples remain readable only through that compatibility adapter.

The current MVP supports bounded linear frequency-domain graph search and ideal
two-port DC power-stage search.  Standard circuits are retained as expert
priors, but PBDL synthesis can also construct typed topology candidates and
verify them with family-matched physics models.

### What it can and cannot claim

Being explicit here matters more than sounding capable:

- **Supported now:** linear AC filters (R/C/L/VCVS) solved by internal MNA;
  four ideal averaged DC power families (buck, boost, SEPIC, isolated flyback)
  with an optional parameterised loss model; KiCad, SVG and SPICE export; a
  complete evidence chain (`report.json`, `replay.jsonl`, `episode.jsonl`).
- **Explicitly refused, not silently wrong:** noise, S-parameter, thermal,
  impedance and stability requirements currently return a capability gap.
- **Not verified by external tools:** `ngspice` and KiCad are not installed in
  this environment, so every "verified" verdict is internal-model consistency,
  not an external physics check. Exports are text-validated but have never been
  opened by KiCad. Do not read them as tape-out ready.
- **Not modelled:** magnetic core loss, semiconductor thermal behaviour,
  electromagnetic interference, and control-loop stability. The loss model
  reports `core_loss_modelled: false` rather than hiding the omission.


## Run

Install the project and all optional development/runtime capabilities into an
isolated Python environment:

```powershell
python -m pip install -e ".[all]"
```

```powershell
python synth.py examples/lowpass_1khz.json --top 3 --out outputs/lowpass
python synth.py examples/highpass_5khz_gain.json --top 3 --out outputs/highpass
python synth.py examples/bandpass_10khz.json --top 2 --out outputs/bandpass
python synth.py examples/lowpass_1khz_second_order_active.json --top 3 --out outputs/lowpass_second_order_active
python synth.py examples/transimpedance_1k_100khz.json --out outputs/tia_1k_100khz
python synth.py examples/output_impedance_50.json --out outputs/output_impedance_50
python pbdl_synth.py 端口描述语言/examples/sensor_interface.json --out outputs/sensor_interface_pbdl
python pbdl_synth.py 端口描述语言/examples/photodiode_tia.json --out outputs/photodiode_tia_pbdl
python synth.py examples/lowpass_1khz_robust.json --out outputs/lowpass_robust
python synth.py examples/lowpass_1khz.json --spice-verify --out outputs/lowpass_spice
python synth.py examples/lowpass_1khz_spice.json --out outputs/lowpass_spice_from_spec
python synth.py examples/lowpass_1khz_diff_graph.json --out outputs/lowpass_diff_graph
python synth.py examples/unrealizable_unstable_zpk.json --feasibility-only
python self_improve.py examples/lowpass_1khz.json examples/bandpass_10khz.json --replay data/replay_graphs.jsonl --model-out models/graph_proposer_replay.joblib --overwrite-replay
python run_benchmark.py --generated --count-per-kind 2 --proposer graph-search --out outputs/benchmark/graph_search.json
```

## Browser workbench

Run a local graphical workbench for defining a port behavior, selecting an
element library, comparing candidates, viewing Bode responses, and downloading
the generated KiCad schematic:

```powershell
python workbench.py
```

Then open `http://127.0.0.1:8000`.  The workbench stores each run under
`outputs/workbench/` and uses the same synthesis engine as the command line.

The output directory contains:

- `report.json`: ranked candidates, parameters, Pareto objectives, metrics, and generated netlist
- `best.spice`: best candidate as a SPICE-style AC netlist
- `best.svg`: best candidate as a rendered schematic from the unified circuit IR
- `candidate_*.svg` and `schematics.html`: visual schematics for ranked candidates

## Current capabilities

- Built-in target behaviors: low-pass, high-pass, band-pass, ZPK, sampled
  magnitude/phase data
- Element-library filtering: R, C, L, ideal op-amp/VCVS-style blocks
- Candidate topology templates: passive RC, passive RLC, buffered cascade RC,
  Sallen-Key active RC, transimpedance amplifier, and gain plus RC
- Curated standard topology catalog with behavior/analysis compatibility,
  family metadata, tags, priority, and topology risk for expert-style proposal
- Continuous parameter optimization with SciPy differential evolution
- Versioned `OptimizationProblem` contracts for continuous, integer,
  categorical, and derived variables, with unit-safe physical values kept
  separate from solver coordinates, explicit budgets, cache statistics, and
  final truth-evaluation evidence
- Auditable AC fidelity scheduling: all candidates are optimized, then only
  the selected Top-K are promoted to Linear MNA and declarative constraint
  verification; ideal power stages remain explicitly labeled `ideal_averaged`
- Multi-objective Pareto report: magnitude error, phase error, component count,
  estimated cost, estimated area, topology risk, optimizer failure, and
  robustness objectives
- AI proposer interface: a learned graph/Transformer/RL proposer can replace
  the heuristic proposer without changing the optimizer or output layer
- Frequency-domain MNA simulator for linear R/C/L circuit graphs
- Current-source AC excitation and transimpedance analysis for TIA-style current-input circuits
- Output-impedance analysis and a minimum output-termination topology for matching stages
- PBDL multi-stage planner that splits complex port specs into supported synthesis stages and explicit unsupported stages
- PBDL runner CLI that writes `plan.json`, `pbdl_report.json`, and per-stage schematic/netlist/report outputs
- Synthetic data generation for training or nearest-neighbor topology proposal
- Generated R/C/L graph search for small topologies, evaluated through MNA
- Monte Carlo tolerance analysis with robustness-aware scoring
- Optional ngspice backend for external SPICE verification
- SVG schematic rendering for optimized candidates, including R/C/L,
  independent voltage sources, ideal VCVS/op-amp blocks, node labels, and values
- Differentiable PyTorch MNA refinement for linear `LinearCircuit` templates, including generated graphs and VCVS blocks
- Validated-data response surrogate: per-topology ensemble prediction, uncertainty/domain gating, and truth-evaluated fallback
- Repeatable benchmark suite with success rate, error, graph rate, and timing
- Feasibility checks for stability, causality/properness, numeric samples, and passive-library warnings
- Structural graph checks, internal-node canonicalization, and degenerate-branch filtering
- Data-driven DC power knowledge base for Buck, Boost, SEPIC, and isolated Flyback families
- Bounded topology-search certificates recording productions, library limits,
  accepted candidates, and rejection reasons
- Solver-independent simulation tasks derived from PBDL analyses, targets, and
  per-port variable constraints
- Unit-safe declarative metrics and constraints for DC ports, AC response,
  transient ripple, graph resources, parameter bounds, and device stress
- Feasibility-first candidate selection followed by Pareto rank, crowding,
  soft penalties, and deterministic tie-breaking
- Multi-candidate power optimization with verified replay rows and KiCad/SVG output

## Data-driven power topology search

PBDL DC conversion no longer commits to one hard-coded topology before
optimization. Functional modules and production slots are loaded from
`circuit_ai/knowledge/power_topologies.yaml`; adding a structurally compatible
production does not require adding another Python connection function. The
power grammar constructs every registered family compatible
with voltage direction, isolation, allowed elements, required elements, and the
component-count bound.  Each production is executable only when a matching
ideal solver and parameter optimizer are registered.

Power runs add these artifacts:

- `topology_search.json`: explicit search bounds, grammar version, and rejected candidates
- `simulation_tasks.json`: normalized measurements and pass/fail observations derived from PBDL
- `report.json`: ranked candidate evaluations and the selected topology
- `optimization_run`: the selected run's solver version, decoded variables,
  search/truth objectives, evaluation count, cache hits, duration, and budget
- `fidelity_schedule`: the ordered screening and truth-verification stages,
  including the candidate IDs promoted at each stage
- `replay.jsonl`: verified positive and negative synthesis rows for future learned proposal
- `episode.jsonl`: one `circuit_ai.design_episode` per candidate, recording the
  graph actions that produced it, each step's before/after state hash, and the
  rejected, reverted, timeout, and unsupported paths with their failure
  category; a whole-search episode keeps the interleaved search log
- `best.svg`, `best.kicad_sch`, and `best.kicad_pro`: visible and editable output

The architecture was informed by selected open-source analog-generation
projects. See `docs/open_source_influences.md` for the audited revisions,
licenses, adopted concepts, and reuse boundary.

## Cost and area objectives

Element libraries may include simple resource estimates:

```json
{
  "library": {
    "allowed": ["R", "C", "opamp"],
    "unit_costs": {"R": 0.02, "C": 0.04, "opamp": 1.5},
    "unit_areas_mm2": {"R": 1.2, "C": 1.8, "opamp": 12.0}
  }
}
```

Each candidate report includes `component_inventory`, `estimated_cost`,
`estimated_area_mm2`, and `topology_risk`. These are also Pareto objectives, so
the ranking can expose engineering trade-offs instead of optimizing response
error alone. Add optional score weights such as `weights.cost` or
`weights.area_mm2` when a spec should scalarize those trade-offs.

## Feasibility checks

```powershell
python synth.py examples/unrealizable_unstable_zpk.json --feasibility-only
```

Feasibility runs before synthesis.  Strict errors such as unstable ZPK poles or
improper noncausal rational targets stop the search by default; warnings are
included in each candidate report.

## External SPICE verification

```powershell
python synth.py examples/lowpass_1khz.json --spice-verify --out outputs/lowpass_spice
```

If `ngspice` is available on `PATH`, the CLI writes verification netlists,
runs AC simulation, parses ASCII raw output, and adds `spice_verification` to
`report.json`.  If ngspice is unavailable, the report marks the verification as
`unavailable` rather than pretending the external gate passed.

The same gate can be enabled from JSON via
`optimization.spice_verification.enabled`.

## Differentiable circuit refinement

```powershell
python synth.py examples/lowpass_1khz_diff_graph.json --out outputs/lowpass_diff_graph
python synth.py examples/highpass_5khz_gain_diff.json --out outputs/highpass_gain_diff
```

When `optimization.differentiable.enabled` is true, linear templates exported as
`LinearCircuit` can be refined through a PyTorch-autograd MNA solver.  Generated
R/C/L graphs and hand-written templates with ideal VCVS gain blocks share this
differentiable physics layer for local parameter improvement and future
learned/RL training.

## Self-improvement replay loop

```powershell
python self_improve.py examples/lowpass_1khz.json examples/bandpass_10khz.json --replay data/replay_graphs.jsonl --model-out models/graph_proposer_replay.joblib --overwrite-replay
python synth.py examples/lowpass_1khz.json --graph-model-proposer models/graph_proposer_replay.joblib --graph-model-only
```

The replay loop runs synthesis tasks, exports generated graph candidates with
performance labels, resamples mixed-spec responses to a common log-frequency
grid, and trains a new graph proposer from accepted replay rows.  Training
bundles now include class counts, training accuracy, holdout validation
accuracy, and top-k accuracy; replay-trained bundles also record accepted,
rejected, and unlabeled row counts so failed candidates are not silently hidden.
Rejected rows train a graph quality/risk reranker by default; they are not used
as ordinary topology-class examples unless the legacy `--train-with-rejected`
flag is explicitly set.  Replay rows also store the target response and
candidate-vs-target error features, enabling a pairwise ranker that learns which
candidate is better within the same synthesis task.

## Benchmark suite

```powershell
python run_benchmark.py --generated --count-per-kind 2 --proposer heuristic --out outputs/benchmark/heuristic.json
python run_benchmark.py --generated --count-per-kind 2 --proposer graph-search --graph-candidates 24 --out outputs/benchmark/graph_search.json
python run_benchmark.py --generated --count-per-kind 2 --proposer graph-model --graph-model models/graph_proposer_replay.joblib --graph-model-only --out outputs/benchmark/replay_model.json
```

The benchmark report includes total success rate, per-filter-kind success,
mean/median error, graph-candidate rate, component count, and runtime.

## Robust synthesis report

```powershell
python synth.py examples/lowpass_1khz_robust.json --out outputs/lowpass_robust
```

When `optimization.robustness.enabled` is true, `report.json` includes yield,
finite-response fraction, P95 error, and worst-case error under randomized
component tolerance perturbations.

## Loss model, operating envelope, and worst-case analysis

A requirement is rarely a single point. Two optional spec blocks let a design be
sized against a range and judged on real efficiency instead of an idealised 1.0:

```json
{
  "operating_envelope": {
    "min_input_voltage_v": 24, "max_input_voltage_v": 48,
    "min_load_fraction": 0.1, "max_load_fraction": 1.0
  },
  "optimization": {
    "loss_model": { "enabled": true },
    "weights": { "efficiency": 3.0 },
    "design_corner": { "rail": "input_min" },
    "worst_case": { "output_tolerance_fraction": 0.05 }
  }
}
```

- **`loss_model`** switches the ideal averaged stages from `efficiency == 1.0`
  to a parameterised estimate: switch and inductor conduction, rectifier drop,
  switching transition, switch capacitance, and output-capacitor ESR. Without it
  "maximise efficiency" is a no-op, because nothing is lost to trade against.
  Core loss is **not** modelled and the record says so
  (`core_loss_modelled: false`).
- **`operating_envelope`** declares the range to survive. `report.json` then
  carries `worst_case`: minimum efficiency, maximum loss, maximum input current,
  duty range, maximum ripple and the delivered rail's full spread, each naming
  the corner responsible, plus `regulates` and per-corner violations against a
  target and tolerance.
- **`design_corner.rail`** chooses where the duty cycle is scheduled:
  `nominal` (default, historical behaviour), `input_min`, `input_max` or
  `geometric_mean`. This is a real trade-off, not a preference: with no feedback
  loop no fixed ratio holds a target across a wide range. On a 2:1 input range,
  `nominal` droops below target at low line, `input_min` makes the target exact
  at low line and overshoots everywhere above it, and `geometric_mean` makes the
  worst deviation symmetric in ratio terms.
- **`regulates` is `null` when no target is given**, never `true`. "Not checked"
  is never reported as "passed".

Omitting both blocks reproduces the previous behaviour byte for byte:
`operating_envelope` and `worst_case` are `null`.

A design that cannot hold its output across its envelope needs a control loop,
which this project does not model. The envelope report makes that visible rather
than letting a nominal-only figure stand in for the result.

## Bounded minimum-component search

```powershell
python prove_minimal.py examples/lowpass_1khz.json --max-components 4 --out outputs/lowpass_minimal_proof.json
```

This proves minimum component count only inside the declared bounded search
space: generated simple R/C/L graphs, selected internal node count, parameter
ranges, optimizer budget, and acceptance thresholds.

Generated graph candidates also carry a `structure` report with canonical keys
and structural issues, so topology search can reject dangling or disconnected
graphs before simulation.

## Generated topology search

```powershell
python synth.py examples/lowpass_1khz.json --graph-search --graph-candidates 24 --out outputs/lowpass_graph
```

This enumerates small legal R/C/L connection graphs, optimizes values for each
candidate, and ranks them beside hand-written or data-driven candidates.

## Data-driven proposal baseline

```powershell
python generate_dataset.py --out data/synthetic_linear.jsonl --samples-per-template 32
python synth.py examples/lowpass_1khz.json --dataset-proposer data/synthetic_linear.jsonl
```

This is not the final neural model.  It is a useful baseline and a data contract
for training a stronger learned proposer.

## Learned template proposer

```powershell
python generate_dataset.py --out data/synthetic_linear.jsonl --samples-per-template 64
python train_template_proposer.py data/synthetic_linear.jsonl --out models/template_proposer.joblib
python synth.py examples/lowpass_1khz.json --model-proposer models/template_proposer.joblib
```

The learned proposer ranks topology classes from the target response.  The
physics loop still optimizes and verifies the resulting candidates.

## Learned typed-graph proposer

```powershell
python generate_graph_dataset.py --out data/synthetic_graphs.jsonl --samples-per-graph 16 --max-components 3 --graph-limit 64
python train_graph_proposer.py data/synthetic_graphs.jsonl --out models/graph_proposer.joblib
python synth.py examples/lowpass_1khz.json --graph-model-proposer models/graph_proposer.joblib --top 3 --out outputs/lowpass_graph_ai
python synth.py examples/lowpass_1khz.json --graph-model-proposer models/graph_proposer.joblib --graph-model-only --top 3 --out outputs/lowpass_graph_ai_only
```

This model predicts generated R/C/L graph topologies directly.  Its guesses are
still optimized and verified by the same physical loop.

## Near-term path to advanced level

1. Add a general modified nodal analysis simulator for arbitrary generated
   typed circuit graphs.
2. Add ngspice/Xyce verification as a final backend.
3. Generate synthetic training data by sampling circuits and simulating their
   port behavior.
4. Train a graph neural network or Transformer proposer:
   `target response + library -> Top-K typed circuit graphs`.
5. Add a differentiable simulator for parameter refinement and gradient-based
   inverse design.
6. Add formal feasibility checks: passivity, causality, positive-real
   conditions, stability, and parameter realizability.
