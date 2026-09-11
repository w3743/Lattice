from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import sample_parameters
from .formatting import db20
from .graph_templates import ElementSlot, GraphCircuitTemplate
from .io import write_jsonl
from .learned import load_jsonl_rows, load_model, save_model, target_feature, train_template_classifier
from .proposers import TopologyProposer
from .quality import score_graph_quality
from .spec import LibrarySpec, SynthesisSpec
from .templates import CircuitTemplate
from .topology import generate_linear_graph_templates


def graph_to_record(template: GraphCircuitTemplate) -> dict[str, Any]:
    return {
        "name": template.name,
        "output_node": template.output_node,
        "source_name": template.source_name,
        "slots": [
            {"name": slot.name, "kind": slot.kind, "n1": slot.n1, "n2": slot.n2}
            for slot in template.slots
        ],
    }


def graph_from_record(record: dict[str, Any]) -> GraphCircuitTemplate:
    slots = tuple(
        ElementSlot(
            name=str(slot["name"]),
            kind=str(slot["kind"]),
            n1=str(slot["n1"]),
            n2=str(slot["n2"]),
        )
        for slot in record["slots"]
    )
    return GraphCircuitTemplate(
        slots,
        output_node=str(record.get("output_node", "out")),
        source_name=str(record.get("source_name", "Vin")),
    )


def generate_graph_rows(
    library: LibrarySpec,
    samples_per_graph: int = 24,
    f_min_hz: float = 10.0,
    f_max_hz: float = 1_000_000.0,
    points: int = 96,
    seed: int = 41,
    max_components: int = 3,
    internal_nodes: int = 1,
    graph_limit: int = 64,
    behavior_kind: str = "*",
) -> list[dict[str, Any]]:
    from .targets import frequency_grid

    rng = np.random.default_rng(seed)
    freqs = frequency_grid(f_min_hz, f_max_hz, points)
    templates = _build_graph_catalog(
        library,
        max_components=max_components,
        internal_nodes=internal_nodes,
        graph_limit=graph_limit,
        behavior_kind=behavior_kind,
    )
    rows: list[dict[str, Any]] = []
    for template in templates:
        for _ in range(samples_per_graph):
            params = sample_parameters(template, library, rng)
            response = template.response(params, freqs)
            if not np.all(np.isfinite(response)):
                continue
            rows.append(
                {
                    "template": template.name,
                    "graph": graph_to_record(template),
                    "component_count": template.component_count,
                    "parameters": params,
                    "frequency_hz": freqs.tolist(),
                    "magnitude_db": db20(response).tolist(),
                    "phase_deg": np.rad2deg(np.unwrap(np.angle(response))).tolist(),
                }
            )
    return rows


def _build_graph_catalog(
    library: LibrarySpec,
    max_components: int,
    internal_nodes: int,
    graph_limit: int,
    behavior_kind: str,
) -> list[GraphCircuitTemplate]:
    kinds = (
        ("lowpass", "highpass", "bandpass", "*")
        if behavior_kind.strip().lower() in {"*", "all", "auto"}
        else (behavior_kind,)
    )
    templates: list[GraphCircuitTemplate] = []
    seen: set[str] = set()
    for kind in kinds:
        for template in generate_linear_graph_templates(
            library,
            max_components=max_components,
            internal_nodes=internal_nodes,
            limit=graph_limit,
            behavior_kind=kind,
        ):
            if template.name not in seen:
                templates.append(template)
                seen.add(template.name)
    return templates[: graph_limit * len(kinds)]


def train_graph_classifier(
    rows: list[dict[str, Any]],
    hidden_layer_sizes: tuple[int, ...] = (96, 48),
    max_iter: int = 1000,
    seed: int = 43,
    use_phase: bool = True,
    validation_fraction: float = 0.2,
    top_k: int = 3,
) -> dict[str, Any]:
    bundle = train_template_classifier(
        rows,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        seed=seed,
        use_phase=use_phase,
        validation_fraction=validation_fraction,
        top_k=top_k,
    )
    catalog: dict[str, dict[str, Any]] = {}
    for row in rows:
        catalog[row["template"]] = row["graph"]
    bundle.update(
        {
            "model_kind": "graph_topology_proposer",
            "graph_catalog": catalog,
        }
    )
    return bundle


@dataclass
class SklearnGraphProposer:
    """Learned response-to-typed-graph proposer.

    The classifier predicts graph topology names.  Each predicted class is
    reconstructed into a GraphCircuitTemplate, then the normal optimizer,
    robustness analyzer, and SPICE verifier provide the engineering gate.
    """

    model_path: str | Path
    fallback: TopologyProposer | None = None
    top_n: int = 12

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        bundle = load_model(self.model_path)
        model = bundle["model"]
        catalog = bundle.get("graph_catalog", {})
        freqs = np.asarray(bundle["frequency_hz"], dtype=float)
        feature = target_feature(spec, freqs, use_phase=bool(bundle.get("use_phase", True)))

        probabilities = model.predict_proba(feature.reshape(1, -1))[0]
        classes = list(model.classes_)
        quality_model = bundle.get("quality_model")
        quality_weight = float(bundle.get("quality_rerank_weight", 0.35))
        behavior_kind = str(spec.behavior["kind"]).strip().lower()
        ranked_names = []
        for probability, name in zip(probabilities, classes):
            record = catalog.get(name)
            quality = score_graph_quality(quality_model, record, behavior_kind) if record else 1.0
            combined = float(probability) * (max(quality, 1e-6) ** quality_weight)
            ranked_names.append((combined, float(probability), quality, name))
        ranked_names = [
            name
            for _, _, _, name in sorted(
                ranked_names,
                key=lambda item: (item[0], item[1], item[2]),
                reverse=True,
            )
        ]

        proposed: list[CircuitTemplate] = []
        seen: set[str] = set()
        for name in ranked_names[: self.top_n]:
            record = catalog.get(name)
            if record is None or name in seen:
                continue
            template = graph_from_record(record)
            if template.is_compatible(spec.library, spec.optimization.max_components, behavior_kind):
                proposed.append(template)
                seen.add(template.name)

        if self.fallback is not None:
            for template in self.fallback.propose(spec):
                if template.name not in seen:
                    proposed.append(template)
                    seen.add(template.name)
        return proposed


def save_graph_model(bundle: dict[str, Any], path: str | Path) -> None:
    save_model(bundle, path)


def load_graph_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = load_jsonl_rows(path)
    for row in rows:
        if "graph" not in row:
            raise ValueError("graph proposer training rows must include graph records")
    return rows
