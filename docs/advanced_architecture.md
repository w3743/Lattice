# Advanced Circuit AI Architecture

Goal:

```text
Given port behavior and an element library, generate a verified circuit.
```

An internationally competitive system should not rely on a single neural model.
It should combine neural proposal, mathematical feasibility checks, differentiable
optimization, and high-fidelity simulation.

## 1. Formal problem

Input:

```text
S = target port behavior
L = allowed element library
C = hidden or explicit engineering constraints
```

Output:

```text
G = typed circuit graph
theta = element parameters
```

Objective:

```text
minimize error(simulate(G, theta), S)
       + component_count_penalty
       + cost/power/sensitivity penalties

subject to:
  element legality
  parameter ranges
  stability
  passivity/causality when required
  manufacturability constraints
```

## 2. System layers

```text
Specification layer
  natural language / functions / samples / S-parameters
  -> normalized target object

Feasibility layer
  passivity, causality, stability, positive-real checks

Topology proposal layer
  templates
  constraint search
  GNN/Transformer/RL proposer

Parameter optimization layer
  global search
  local gradient optimization
  differentiable simulation

Verification layer
  analytic simulator for fast loops
  ngspice/Xyce for final verification
  corner and tolerance analysis

Artifact layer
  SPICE netlist
  schematic
  ranking report
  failure explanation
```

## 3. Learning strategy

Training data can be generated automatically:

```text
sample legal circuit graph
sample legal parameters
simulate port behavior
store (behavior, library, graph, parameters, score)
```

The learned proposer is trained on the reverse task:

```text
behavior + library -> graph candidates
```

The system should output Top-K candidates, not a single circuit.  The verifier
then turns uncertain neural guesses into reliable engineering artifacts.

The current repository already includes the first version of this data flywheel:

```text
generate_dataset.py
  -> synthetic JSONL rows of template, parameters, magnitude, phase

DataDrivenProposer
  -> nearest-neighbor topology proposal baseline
```

It also now has a first typed-graph synthesis path:

```text
generate_linear_graph_templates()
  -> small legal R/C/L graph candidates

GraphCircuitTemplate
  -> MNA-evaluated candidate with optimized element values

GraphSearchProposer
  -> topology search plugged into the same proposer interface as AI models
```

And a first learned proposer path:

```text
train_template_proposer.py
  -> trains an MLP classifier from port response features to topology class

SklearnTemplateProposer
  -> ranks topology templates before physical optimization and verification
```

The next graph-level learning path is also implemented:

```text
generate_graph_dataset.py
  -> samples typed R/C/L graph topologies and simulates port responses

train_graph_proposer.py
  -> trains response-to-graph-topology classifier

SklearnGraphProposer
  -> reconstructs GraphCircuitTemplate candidates from predicted graph records
```

The verification layer now includes a first Monte Carlo tolerance analyzer:

```text
optimization.robustness
  -> component perturbation settings and acceptance thresholds

RobustnessMetrics
  -> yield, finite fraction, P95 error, worst-case error

robustness_penalty()
  -> optional score penalty so fragile candidates rank lower
```

Candidate selection now consumes the declarative constraint report before its
Pareto layer:

```text
PBDL + CircuitGraph + SimulationResult
  -> MetricEngine produces unit-bearing evidence
  -> ConstraintEvaluator separates hard, soft, and objective semantics
  -> feasibility tier, missing evidence count, and hard violation
  -> non-dominated sorting inside each feasibility tier
  -> Pareto rank and crowding distance
  -> soft penalty and deterministic scalar tie-breaker
```

This removes the fixed invalid-candidate penalty. A better weighted score can no
longer outrank a verified hard-constraint failure, while feasible candidates can
still expose cost, area, component count, response error, and robustness trade-offs.

For minimum-component claims, the system must be explicit about the search
boundary.  The prototype includes `prove_minimal.py`, which searches exact
component counts and reports a bounded proof:

```text
for k in 2..max_components:
  enumerate generated graphs with exactly k elements
  optimize each candidate
  accept the first k whose best circuit passes thresholds
```

External simulator verification is now represented by `NgspiceVerifier`:

```text
generated netlist
  -> validation netlist with requested AC sweep
  -> ngspice batch run
  -> ASCII raw parser
  -> target-response error metrics in report.json
```

This keeps fast internal MNA useful for search while preserving a separate
high-fidelity gate for final candidate evidence.

The differentiable physics layer is represented by `TorchLinearCircuitMNA`:

```text
CircuitTemplate.to_circuit(torch log-parameters)
  -> differentiable R/C/L and VCVS MNA matrix stamps
  -> torch.linalg.solve AC response
  -> differentiable response loss
  -> Adam local refinement
```

`TorchGraphMNA` remains as a compatibility name, but the implementation now
uses the unified `LinearCircuit` representation. This is the bridge from
black-box search toward trainable inverse design: graph proposal can remain
discrete while both generated graphs and hand-written linear active templates
refine values through gradients.

The active-learning layer is represented by replay data:

```text
synthesis result
  -> graph replay row with topology, optimized parameters, response, target, labels
  -> candidate-vs-target error feature extraction
  -> common frequency-grid resampling across mixed specs
  -> accepted rows train the topology proposer
  -> accepted/rejected labels train the graph quality reranker
  -> same-spec candidate pairs train the replay pairwise ranker
  -> replay-trained graph proposer bundle
```

This closes the learning loop: the system can reuse its own verified successes
as future proposal training data while also learning which graph structures have
failed for a behavior family and which candidates outrank others for the same
synthesis objective.

The benchmarking layer is represented by `run_benchmark.py`:

```text
generated or explicit specs
  -> selected proposer
  -> synthesis run
  -> per-case result rows
  -> aggregate success/error/component/runtime metrics
```

This gives the project a measurable learning target instead of relying on
single demonstrations.

The feasibility layer is represented by `analyze_feasibility()`:

```text
target behavior + library + optimization settings
  -> stability / causality / numeric consistency checks
  -> error or warning report
  -> optional strict gate before search
```

This prevents the search stack from silently spending time on specifications
that are already mathematically or physically contradictory.

The structural legality layer is represented by `analyze_slots()`:

```text
typed graph slots
  -> graph connectivity and source-shunt checks
  -> dangling internal-node rejection
  -> internal-node-renaming canonical key
  -> optional warnings for compressible series/parallel patterns
```

This reduces invalid topology search before numerical optimization begins.

## 4. Research-grade extensions

- Typed graph grammar for valid circuit construction.
- CP-SAT or SMT search for proving minimum component count at small sizes.
- Differentiable modified nodal analysis for gradient-based parameter fitting.
- Neural surrogate simulator for fast pre-screening.
- Reinforcement learning for sequential component placement.
- Self-improving replay buffer from successful and failed synthesis attempts.
- Multi-fidelity verification: analytic -> surrogate -> SPICE -> corners.

## 5. First production target

The first product-grade target should be:

```text
linear two-port analog synthesis
R/C/L/ideal-opamp libraries
frequency-domain target behavior
SPICE netlist output
```

After that is reliable, expand to transistor-level analog blocks and layout-aware
flows using generator frameworks such as OpenFASOC-style parameterized blocks.
