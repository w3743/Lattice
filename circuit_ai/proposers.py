from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from .catalog import CatalogProposer, TopologyRecord
from .formatting import db20
from .spec import SynthesisSpec
from .targets import frequency_grid, target_from_behavior
from .templates import CircuitTemplate, default_templates


class TopologyProposer(Protocol):
    """Interface for heuristic, learned, or search-based topology proposal."""

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        ...


@dataclass
class HeuristicProposer:
    """Strong deterministic baseline.

    This is the first "AI hook": it turns the target kind and element library
    into a ranked topology shortlist.  A GNN/Transformer/RL proposer can later
    implement the same protocol and return candidates learned from data.
    """

    templates: list[CircuitTemplate] | None = None
    catalog: list[TopologyRecord] | None = None

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        if self.templates is None:
            return CatalogProposer(self.catalog).propose(spec)

        behavior_kind = str(spec.behavior["kind"]).strip().lower()
        compatible = [
            template
            for template in self.templates
            if template.is_compatible(
                spec.library,
                spec.optimization.max_components,
                behavior_kind,
            )
        ]

        def rank(template: CircuitTemplate) -> tuple[int, int, str]:
            exact_kind = 0 if behavior_kind in template.supported_kinds else 1
            return (exact_kind, template.component_count, template.name)

        return sorted(compatible, key=rank)


class LearnedProposerPlaceholder:
    """Documented extension point for a learned topology generator.

    Expected future contract:
      input  = encoded target response + encoded element library
      output = Top-K CircuitTemplate or generated typed circuit graphs

    The rest of the synthesis stack does not need to know whether a candidate
    came from hand-coded templates, a graph neural network, or reinforcement
    learning.
    """

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        raise NotImplementedError("train or load a learned proposer, then implement propose()")


@dataclass
class DataDrivenProposer:
    """Nearest-neighbor baseline for learned topology proposal.

    This is deliberately simple and transparent.  It consumes synthetic rows of
    (template, parameters, frequency response), compares the requested behavior
    to stored responses, and proposes templates that appeared near the target.
    A neural proposer can later replace this with the same `propose()` contract.
    """

    dataset_path: str | Path
    fallback: TopologyProposer | None = None
    neighbors: int = 24

    def propose(self, spec: SynthesisSpec) -> list[CircuitTemplate]:
        rows = self._load_rows()
        if not rows:
            return (self.fallback or HeuristicProposer()).propose(spec)

        template_by_name = {template.name: template for template in default_templates()}
        freqs = np.asarray(rows[0]["frequency_hz"], dtype=float)
        target = target_from_behavior(spec.behavior, freqs, spec.analysis)
        target_db = db20(target.values)

        ranked: list[tuple[float, str]] = []
        for row in rows:
            if len(row.get("frequency_hz", [])) != len(freqs):
                continue
            mag_db = np.asarray(row["magnitude_db"], dtype=float)
            distance = float(np.sqrt(np.mean((mag_db - target_db) ** 2)))
            ranked.append((distance, row["template"]))

        behavior_kind = str(spec.behavior["kind"]).strip().lower()
        proposed: list[CircuitTemplate] = []
        seen: set[str] = set()
        for _, name in sorted(ranked)[: self.neighbors]:
            template = template_by_name.get(name)
            if template is None or name in seen:
                continue
            if template.is_compatible(spec.library, spec.optimization.max_components, behavior_kind):
                proposed.append(template)
                seen.add(name)

        if not proposed:
            return (self.fallback or HeuristicProposer()).propose(spec)

        heuristic = (self.fallback or HeuristicProposer()).propose(spec)
        for template in heuristic:
            if template.name not in seen:
                proposed.append(template)
                seen.add(template.name)
        return proposed

    def _load_rows(self) -> list[dict]:
        path = Path(self.dataset_path)
        rows = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
