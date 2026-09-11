"""Versioned DesignEpisode replay contract for complete design experiences.

One episode is one design experience: the ordered graph actions that were
attempted, the state hash before and after each attempt, and the simulation,
constraint, and cost evidence attached to it.  Success, rejection, rollback,
timeout, and unsupported paths are first-class steps because a learned model
must see *what a modification did to the candidate*, not only the final
accepted label.

The module owns three things:

* the :class:`DesignStep` / :class:`DesignEpisode` wire contracts and their
  stable JSON codecs,
* a lossless :class:`~circuit_ai.graph.actions.GraphAction` codec (the search
  certificate's ``action_payload`` shape is accepted as a legacy input),
* conversions from the existing search certificate and evaluated candidates
  into episodes, so callers never re-implement the mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, get_args

from ..constraints.contracts import ConstraintReport, FeasibilityStatus
from ..graph.actions import (
    AddComponent,
    AddFeedback,
    AddFragment,
    ConnectTerminal,
    GraphAction,
    OpenTerminal,
    RemoveComponent,
    ReplaceModel,
    SplitNet,
    TerminatePort,
)
from ..graph.model import (
    CircuitGraph,
    ModelRef,
    ParameterBinding,
    Rating,
    decode_graph_value,
    encode_graph_value,
)
from ..graph.primitives import primitive_model_ref
from ..simulation.contracts import SimulationResult, SimulationStatus

EPISODE_SCHEMA = "circuit_ai.design_episode"
EPISODE_SCHEMA_VERSION = 1

#: Every concrete member of the typed graph-action union, so the codec coverage
#: check fails loudly when a new action type is added without a codec.
_ACTION_UNION = get_args(GraphAction)

#: ``structural_status`` values.  They describe the graph-structure verdict of
#: a step, independently of whether physics later accepted the candidate.
STRUCTURAL_VALID = "valid_graph"
STRUCTURAL_NOT_APPLIED = "not_applied"
STRUCTURAL_REVERTED = "reverted"
STRUCTURAL_COMPLETE = "complete_graph"


class EpisodeContractError(ValueError):
    """Raised when a serialized episode, step, or action violates its contract."""


class EpisodeChainError(EpisodeContractError):
    """Raised when a step sequence does not form one continuous state chain."""

    def __init__(
        self,
        step_index: int,
        message: str,
        *,
        expected: Any = None,
        actual: Any = None,
    ):
        self.step_index = step_index
        self.expected = expected
        self.actual = actual
        super().__init__(f"step {step_index}: {message}")


# --------------------------------------------------------------------------
# Stable hashing helpers
# --------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_payload_hash(payload: Any) -> str:
    """Return the stable fingerprint used for spec and manifest identities."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def empty_state_hash(spec_hash: str) -> str:
    """Identify the empty starting state of a spec's design space.

    Runs that retrieve whole candidate graphs instead of growing one graph
    action by action have no graph skeleton; their initial state is the empty
    design space of the spec, which this identity names explicitly.
    """

    return stable_payload_hash({"spec_hash": str(spec_hash), "state": "empty"})


def manifest_hash(manifest: Any) -> str:
    """Hash a capability manifest as the episode's environment identity."""

    if isinstance(manifest, Mapping):
        payload: Any = dict(manifest)
    elif hasattr(manifest, "as_dict"):
        payload = manifest.as_dict()
    else:
        payload = {"manifest": str(manifest)}
    return stable_payload_hash(payload)


def spec_hash_for(source: Mapping[str, Any]) -> str:
    """Canonical spec identity for a PBDL/IR source document."""

    from ..pbdl_boundary import canonicalize_pbdl_dict

    return stable_payload_hash(canonicalize_pbdl_dict(dict(source)))


def episode_hash(episode: "DesignEpisode") -> str:
    """Return the stable episode fingerprint used for replay verification.

    The fingerprint covers the contract payload, so a replayed action
    sequence, state chain, cost, or budget change is detected.  Fields that a
    producer added on top of this schema version are preserved but excluded:
    they are by definition not part of the versioned contract.
    """

    return stable_payload_hash(_fingerprint_payload(episode.as_dict()))


#: Preservation buckets for fields outside the current schema version.
UNKNOWN_EPISODE_FIELDS = "unknown_top_level_fields"
UNKNOWN_STEP_FIELDS = "unknown_step_fields"


def _fingerprint_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The contract payload of an episode, without unpinned producer fields."""

    core = {key: value for key, value in payload.items() if key != "episode_hash"}
    extensions = {key: value for key, value in dict(core.get("extensions") or {}).items()}
    extensions.pop(UNKNOWN_EPISODE_FIELDS, None)
    core["extensions"] = extensions
    steps: list[dict[str, Any]] = []
    for step in core.get("steps") or ():
        entry = dict(step)
        step_extensions = {key: value for key, value in dict(entry.get("extensions") or {}).items()}
        step_extensions.pop(UNKNOWN_STEP_FIELDS, None)
        entry["extensions"] = step_extensions
        steps.append(entry)
    core["steps"] = steps
    return core


# --------------------------------------------------------------------------
# Graph action codec
# --------------------------------------------------------------------------


def _action_type_name(action_type: type) -> str:
    return str(action_type.__dataclass_fields__["action_type"].default)


def _encode_model(model: Any) -> dict[str, Any]:
    if isinstance(model, ModelRef):
        return {"kind": "model_ref", "value": model.as_dict()}
    if hasattr(model, "model_id") and hasattr(model, "as_dict"):
        return {"kind": "model_ref", "value": model.as_dict()}
    return {"kind": "model_id", "value": str(model)}


def _decode_model(payload: Any) -> Any:
    if isinstance(payload, str):
        return primitive_model_ref(payload)
    if not isinstance(payload, Mapping):
        raise EpisodeContractError(f"unsupported action model encoding {payload!r}")
    kind = str(payload.get("kind", ""))
    value = payload.get("value")
    if kind == "model_ref":
        if not isinstance(value, Mapping):
            raise EpisodeContractError(f"model_ref encoding needs a mapping, got {value!r}")
        return ModelRef.from_dict(value)
    if kind == "model_id":
        return primitive_model_ref(str(value))
    if "model_id" in payload:
        # Legacy ``action_payload`` shape: the model ref dictionary itself.
        return ModelRef.from_dict(payload)
    raise EpisodeContractError(f"unknown action model encoding kind {kind!r}")


def _encode_endpoint(value: Any) -> dict[str, Any]:
    if isinstance(value, OpenTerminal):
        return {"kind": "open_terminal", **value.as_dict()}
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return {
            "kind": "terminal_pair",
            "component_id": encode_graph_value(value[0]),
            "terminal_name": encode_graph_value(value[1]),
        }
    # ``AddFeedback`` carries semantic endpoints (for example a component
    # reference and a port name), so any remaining value is encoded literally.
    return {"kind": "literal", "value": encode_graph_value(value)}


def _decode_endpoint(payload: Any) -> Any:
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return (decode_graph_value(payload[0]), decode_graph_value(payload[1]))
    if not isinstance(payload, Mapping):
        return decode_graph_value(payload)
    kind = str(payload.get("kind", ""))
    if kind == "open_terminal":
        return OpenTerminal(
            str(payload.get("component_id", "")),
            str(payload.get("terminal_name", "")),
            str(payload.get("net_id", "")),
        )
    if kind == "terminal_pair":
        return (
            decode_graph_value(payload.get("component_id")),
            decode_graph_value(payload.get("terminal_name")),
        )
    if "component_id" in payload and "terminal_name" in payload:
        # Legacy ``action_payload`` shape: ``OpenTerminal.as_dict()``.
        return OpenTerminal(
            str(payload.get("component_id", "")),
            str(payload.get("terminal_name", "")),
            str(payload.get("net_id", "")),
        )
    if kind == "literal" or "value" in payload:
        return decode_graph_value(payload.get("value"))
    raise EpisodeContractError(f"unknown action terminal encoding kind {kind!r}")


def _encode_endpoints(values: Iterable[Any]) -> list[dict[str, Any]]:
    return [_encode_endpoint(item) for item in values]


def _encode_parameters(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {"kind": "mapping", "value": encode_graph_value(value)}
    items = []
    for item in value:
        if isinstance(item, ParameterBinding):
            items.append({"kind": "binding", "value": item.as_dict()})
        else:
            items.append({"kind": "raw", "value": encode_graph_value(item)})
    return {"kind": "bindings", "value": items}


def _decode_binding_item(item: Any) -> Any:
    if not isinstance(item, Mapping):
        raise EpisodeContractError(f"unsupported parameter binding entry {item!r}")
    kind = str(item.get("kind", ""))
    if kind == "binding":
        return ParameterBinding.from_dict(item.get("value") or {})
    if kind == "raw":
        return decode_graph_value(item.get("value"))
    if "name" in item:
        # Legacy ``action_payload`` shape: ``ParameterBinding.as_dict()``.
        return ParameterBinding.from_dict(item)
    raise EpisodeContractError(f"unknown parameter binding encoding kind {kind!r}")


def _decode_parameters(payload: Any) -> Any:
    if isinstance(payload, Mapping) and "kind" in payload:
        kind = str(payload.get("kind"))
        value = payload.get("value")
        if kind == "mapping":
            return dict(decode_graph_value(value or {}))
        if kind == "bindings":
            return tuple(_decode_binding_item(item) for item in value or ())
        raise EpisodeContractError(f"unknown parameter encoding kind {kind!r}")
    if isinstance(payload, Mapping):
        # Legacy ``action_payload`` emits the mapping directly.
        return dict(decode_graph_value(payload))
    if payload is None:
        return ()
    if isinstance(payload, (list, tuple)):
        return tuple(_decode_binding_item(item) for item in payload)
    raise EpisodeContractError(f"unsupported action parameters encoding {payload!r}")


def _encode_ratings(values: Iterable[Any]) -> list[dict[str, Any]]:
    items = []
    for item in values:
        if isinstance(item, Rating):
            items.append({"kind": "rating", "value": item.as_dict()})
        else:
            items.append({"kind": "value", "value": encode_graph_value(item)})
    return items


def _decode_ratings(payload: Any) -> tuple[Any, ...]:
    items = []
    for item in payload or ():
        if not isinstance(item, Mapping):
            raise EpisodeContractError(f"unsupported rating encoding {item!r}")
        kind = str(item.get("kind", ""))
        if kind == "rating":
            items.append(Rating.from_dict(item.get("value") or {}))
        elif kind == "value":
            items.append(decode_graph_value(item.get("value")))
        elif "quantity" in item:
            items.append(Rating.from_dict(item))
        else:
            raise EpisodeContractError(f"unknown rating encoding kind {kind!r}")
    return tuple(items)


def _encode_add_component(action: AddComponent) -> dict[str, Any]:
    return {
        "model": _encode_model(action.model),
        "instance_id": action.instance_id,
        "connections": encode_graph_value(action.connections),
        "parameters": _encode_parameters(action.parameters),
        "ratings": _encode_ratings(action.ratings),
        "attributes": encode_graph_value(action.attributes),
    }


def _decode_add_component(payload: Mapping[str, Any]) -> AddComponent:
    if "model" not in payload:
        raise EpisodeContractError("AddComponent payload is missing its model")
    return AddComponent(
        model=_decode_model(payload.get("model")),
        instance_id=(
            str(payload["instance_id"]) if payload.get("instance_id") is not None else None
        ),
        connections=dict(decode_graph_value(payload.get("connections") or {})),
        parameters=_decode_parameters(payload.get("parameters")),
        ratings=_decode_ratings(payload.get("ratings")),
        attributes=dict(decode_graph_value(payload.get("attributes") or {})),
    )


def _encode_add_fragment(action: AddFragment) -> dict[str, Any]:
    return {
        "graph_id": action.fragment.graph_id,
        "fragment": action.fragment.as_dict(),
        "prefix": action.prefix,
    }


def _decode_add_fragment(payload: Mapping[str, Any]) -> AddFragment:
    fragment = payload.get("fragment")
    if not isinstance(fragment, Mapping):
        raise EpisodeContractError(
            "AddFragment payload has no fragment graph; the lossy certificate "
            "action_payload form cannot be replayed"
        )
    return AddFragment(
        fragment=CircuitGraph.from_dict(fragment),
        prefix=str(payload.get("prefix", "")),
    )


def _encode_connect_terminal(action: ConnectTerminal) -> dict[str, Any]:
    return {"terminal": _encode_endpoint(action.terminal), "net": action.net}


def _decode_connect_terminal(payload: Mapping[str, Any]) -> ConnectTerminal:
    if "terminal" not in payload or "net" not in payload:
        raise EpisodeContractError("ConnectTerminal payload is missing terminal or net")
    return ConnectTerminal(_decode_endpoint(payload["terminal"]), str(payload["net"]))


def _encode_split_net(action: SplitNet) -> dict[str, Any]:
    return {
        "net": action.net,
        "new_net": action.new_net,
        "terminals": _encode_endpoints(action.terminals),
    }


def _decode_split_net(payload: Mapping[str, Any]) -> SplitNet:
    if "net" not in payload or "new_net" not in payload:
        raise EpisodeContractError("SplitNet payload is missing net or new_net")
    return SplitNet(
        str(payload["net"]),
        str(payload["new_net"]),
        tuple(_decode_endpoint(item) for item in payload.get("terminals") or ()),
    )


def _encode_add_feedback(action: AddFeedback) -> dict[str, Any]:
    return {
        "source": _encode_endpoint(action.source),
        "destination": _encode_endpoint(action.destination),
    }


def _decode_add_feedback(payload: Mapping[str, Any]) -> AddFeedback:
    if "source" not in payload or "destination" not in payload:
        raise EpisodeContractError("AddFeedback payload is missing source or destination")
    return AddFeedback(_decode_endpoint(payload["source"]), _decode_endpoint(payload["destination"]))


def _encode_terminate_port(action: TerminatePort) -> dict[str, Any]:
    return {"port": action.port}


def _decode_terminate_port(payload: Mapping[str, Any]) -> TerminatePort:
    return TerminatePort(str(payload.get("port", "")))


def _encode_replace_model(action: ReplaceModel) -> dict[str, Any]:
    return {"instance": action.instance, "model": _encode_model(action.model)}


def _decode_replace_model(payload: Mapping[str, Any]) -> ReplaceModel:
    if "instance" not in payload or "model" not in payload:
        raise EpisodeContractError("ReplaceModel payload is missing instance or model")
    return ReplaceModel(str(payload["instance"]), _decode_model(payload["model"]))


def _encode_remove_component(action: RemoveComponent) -> dict[str, Any]:
    return {"instance": action.instance}


def _decode_remove_component(payload: Mapping[str, Any]) -> RemoveComponent:
    return RemoveComponent(str(payload.get("instance", "")))


_ACTION_ENCODERS: Mapping[str, tuple[type, Callable[[Any], dict[str, Any]]]] = MappingProxyType(
    {
        "AddComponent": (AddComponent, _encode_add_component),
        "AddFragment": (AddFragment, _encode_add_fragment),
        "ConnectTerminal": (ConnectTerminal, _encode_connect_terminal),
        "SplitNet": (SplitNet, _encode_split_net),
        "AddFeedback": (AddFeedback, _encode_add_feedback),
        "TerminatePort": (TerminatePort, _encode_terminate_port),
        "ReplaceModel": (ReplaceModel, _encode_replace_model),
        "RemoveComponent": (RemoveComponent, _encode_remove_component),
    }
)

_ACTION_DECODERS: Mapping[str, Callable[[Mapping[str, Any]], GraphAction]] = MappingProxyType(
    {
        "AddComponent": _decode_add_component,
        "AddFragment": _decode_add_fragment,
        "ConnectTerminal": _decode_connect_terminal,
        "SplitNet": _decode_split_net,
        "AddFeedback": _decode_add_feedback,
        "TerminatePort": _decode_terminate_port,
        "ReplaceModel": _decode_replace_model,
        "RemoveComponent": _decode_remove_component,
    }
)


def _verify_action_codec() -> None:
    """Fail import when the action union grows an action without a codec."""

    declared = {_action_type_name(item) for item in _ACTION_UNION}
    missing_decode = sorted(declared - set(_ACTION_DECODERS))
    missing_encode = sorted(declared - set(_ACTION_ENCODERS))
    extra = sorted((set(_ACTION_ENCODERS) | set(_ACTION_DECODERS)) - declared)
    if missing_decode or missing_encode or extra:
        raise RuntimeError(
            "DesignEpisode action codec does not match GraphAction: "
            f"missing_decode={missing_decode} missing_encode={missing_encode} extra={extra}"
        )


_verify_action_codec()


def encode_action(action: GraphAction) -> dict[str, Any]:
    """Serialize one typed graph action losslessly, including its full graph."""

    name = _action_type_name(type(action))
    entry = _ACTION_ENCODERS.get(name)
    if entry is None or not isinstance(action, entry[0]):
        raise EpisodeContractError(f"unsupported graph action {type(action).__name__}")
    payload: dict[str, Any] = {"type": name}
    payload.update(entry[1](action))
    return payload


def decode_action(payload: Mapping[str, Any]) -> GraphAction:
    """Rebuild a typed graph action from :func:`encode_action` output.

    Unknown action types are rejected explicitly; unknown extra keys are
    ignored, matching the graph migration policy for unknown fields.
    """

    if not isinstance(payload, Mapping):
        raise EpisodeContractError(f"action payload must be a mapping, got {payload!r}")
    action_type = str(payload.get("type", ""))
    decoder = _ACTION_DECODERS.get(action_type)
    if decoder is None:
        raise EpisodeContractError(f"unknown graph action type {action_type!r}")
    try:
        return decoder(payload)
    except EpisodeContractError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EpisodeContractError(
            f"invalid {action_type} payload: {type(exc).__name__}: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# Step and episode contracts
# --------------------------------------------------------------------------


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _finite_cost(value: Mapping[str, Any] | None) -> Mapping[str, float]:
    cost: dict[str, float] = {}
    for key, item in (value or {}).items():
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        numeric = float(item)
        if math.isfinite(numeric):
            cost[str(key)] = numeric
    return MappingProxyType(cost)


def _string_tuple(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(str(item) for item in (values or ()))


@dataclass(frozen=True)
class DesignStep:
    """One attempted graph modification and everything observed about it."""

    step_index: int
    state_before_hash: str
    action: GraphAction
    state_after_hash: str | None
    structural_status: str
    simulation_request_ids: tuple[str, ...] = ()
    simulation_result_ids: tuple[str, ...] = ()
    constraint_evaluation_ids: tuple[str, ...] = ()
    failure_category: str | None = None
    cost: Mapping[str, float] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise EpisodeContractError("step_index must not be negative")
        if not isinstance(self.action, _ACTION_UNION):
            raise EpisodeContractError(f"step action must be a GraphAction, got {self.action!r}")
        object.__setattr__(self, "state_before_hash", str(self.state_before_hash))
        object.__setattr__(
            self,
            "state_after_hash",
            None if self.state_after_hash is None else str(self.state_after_hash),
        )
        object.__setattr__(self, "structural_status", str(self.structural_status))
        object.__setattr__(self, "simulation_request_ids", _string_tuple(self.simulation_request_ids))
        object.__setattr__(self, "simulation_result_ids", _string_tuple(self.simulation_result_ids))
        object.__setattr__(
            self,
            "constraint_evaluation_ids",
            _string_tuple(self.constraint_evaluation_ids),
        )
        object.__setattr__(
            self,
            "failure_category",
            None if self.failure_category is None else str(self.failure_category),
        )
        object.__setattr__(self, "cost", _finite_cost(self.cost))
        object.__setattr__(
            self,
            "extensions",
            _freeze(decode_graph_value(dict(self.extensions or {}))),
        )

    @property
    def applied(self) -> bool:
        """True when the action advanced the candidate state."""

        return self.state_after_hash is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "state_before_hash": self.state_before_hash,
            "action": encode_action(self.action),
            "state_after_hash": self.state_after_hash,
            "structural_status": self.structural_status,
            "simulation_request_ids": list(self.simulation_request_ids),
            "simulation_result_ids": list(self.simulation_result_ids),
            "constraint_evaluation_ids": list(self.constraint_evaluation_ids),
            "failure_category": self.failure_category,
            "cost": {key: float(value) for key, value in self.cost.items()},
            "extensions": encode_graph_value(dict(self.extensions)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignStep":
        if not isinstance(data, Mapping):
            raise EpisodeContractError(f"design step must be a mapping, got {data!r}")
        known = {
            "step_index",
            "state_before_hash",
            "action",
            "state_after_hash",
            "structural_status",
            "simulation_request_ids",
            "simulation_result_ids",
            "constraint_evaluation_ids",
            "failure_category",
            "cost",
            "extensions",
        }
        extensions = {
            **dict(decode_graph_value(data.get("extensions") or {})),
        }
        unknown = {key: decode_graph_value(value) for key, value in data.items() if key not in known}
        if unknown:
            preserved = dict(extensions.get(UNKNOWN_STEP_FIELDS) or {})
            for key, value in unknown.items():
                preserved.setdefault(key, value)
            extensions[UNKNOWN_STEP_FIELDS] = preserved
        if "action" not in data:
            raise EpisodeContractError("design step is missing its action")
        return cls(
            step_index=int(data.get("step_index", 0)),
            state_before_hash=str(data.get("state_before_hash", "")),
            action=decode_action(data["action"]),
            state_after_hash=(
                None if data.get("state_after_hash") is None else str(data["state_after_hash"])
            ),
            structural_status=str(data.get("structural_status", "unknown")),
            simulation_request_ids=_string_tuple(data.get("simulation_request_ids")),
            simulation_result_ids=_string_tuple(data.get("simulation_result_ids")),
            constraint_evaluation_ids=_string_tuple(data.get("constraint_evaluation_ids")),
            failure_category=(
                None
                if data.get("failure_category") is None
                else str(data["failure_category"])
            ),
            cost=dict(data.get("cost") or {}),
            extensions=extensions,
        )


@dataclass(frozen=True)
class DesignEpisode:
    """One complete design experience, replayable and comparable by hash."""

    episode_id: str
    spec_hash: str
    environment_manifest_hash: str
    initial_state_hash: str
    steps: tuple[DesignStep, ...] = ()
    final_graph_ids: tuple[str, ...] = ()
    termination_reason: str = "unknown"
    budget: Mapping[str, float] = field(default_factory=dict)
    random_seed: int | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema: str = EPISODE_SCHEMA
    schema_version: int = EPISODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", str(self.episode_id))
        object.__setattr__(self, "spec_hash", str(self.spec_hash))
        object.__setattr__(self, "environment_manifest_hash", str(self.environment_manifest_hash))
        object.__setattr__(self, "initial_state_hash", str(self.initial_state_hash))
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "final_graph_ids", _string_tuple(self.final_graph_ids))
        object.__setattr__(self, "termination_reason", str(self.termination_reason))
        object.__setattr__(self, "budget", _finite_cost(self.budget))
        object.__setattr__(
            self,
            "random_seed",
            None if self.random_seed is None else int(self.random_seed),
        )
        object.__setattr__(
            self,
            "extensions",
            _freeze(decode_graph_value(dict(self.extensions or {}))),
        )

    @property
    def episode_hash(self) -> str:
        return episode_hash(self)

    @property
    def applied_steps(self) -> tuple[DesignStep, ...]:
        return tuple(step for step in self.steps if step.applied)

    @property
    def failure_categories(self) -> tuple[str, ...]:
        return tuple(
            sorted({step.failure_category for step in self.steps if step.failure_category})
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "episode_id": self.episode_id,
            "spec_hash": self.spec_hash,
            "environment_manifest_hash": self.environment_manifest_hash,
            "initial_state_hash": self.initial_state_hash,
            "steps": [step.as_dict() for step in self.steps],
            "final_graph_ids": list(self.final_graph_ids),
            "termination_reason": self.termination_reason,
            "budget": {key: float(value) for key, value in self.budget.items()},
            "random_seed": self.random_seed,
            "extensions": encode_graph_value(dict(self.extensions)),
        }
        payload["episode_hash"] = stable_payload_hash(_fingerprint_payload(payload))
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignEpisode":
        """Rebuild an episode, preserving unknown fields instead of dropping them."""

        if not isinstance(data, Mapping):
            raise EpisodeContractError(f"design episode must be a mapping, got {data!r}")
        raw_version = data.get("schema_version", EPISODE_SCHEMA_VERSION)
        try:
            version = int(raw_version)
        except (TypeError, ValueError) as exc:
            raise EpisodeContractError(
                f"invalid design episode schema_version={raw_version!r}"
            ) from exc
        schema = data.get("schema")
        if version > 0 and schema != EPISODE_SCHEMA:
            raise EpisodeContractError(f"unsupported design episode schema {schema!r}")
        if version > EPISODE_SCHEMA_VERSION:
            raise EpisodeContractError(
                f"unsupported future design episode schema_version={version}"
            )

        known = {
            "schema",
            "schema_version",
            "episode_id",
            "spec_hash",
            "environment_manifest_hash",
            "initial_state_hash",
            "steps",
            "final_graph_ids",
            "termination_reason",
            "budget",
            "random_seed",
            "extensions",
            "episode_hash",
        }
        extensions = {**dict(decode_graph_value(data.get("extensions") or {}))}
        unknown = {key: decode_graph_value(value) for key, value in data.items() if key not in known}
        if unknown:
            preserved = dict(extensions.get(UNKNOWN_EPISODE_FIELDS) or {})
            for key, value in unknown.items():
                preserved.setdefault(key, value)
            extensions[UNKNOWN_EPISODE_FIELDS] = preserved

        steps = tuple(
            DesignStep.from_dict(item)
            for item in data.get("steps") or ()
        )
        random_seed = data.get("random_seed")
        episode = cls(
            episode_id=str(data.get("episode_id", "")),
            spec_hash=str(data.get("spec_hash", "")),
            environment_manifest_hash=str(data.get("environment_manifest_hash", "")),
            initial_state_hash=str(data.get("initial_state_hash", "")),
            steps=steps,
            final_graph_ids=_string_tuple(data.get("final_graph_ids")),
            termination_reason=str(data.get("termination_reason", "unknown")),
            budget=dict(data.get("budget") or {}),
            random_seed=None if random_seed is None else int(random_seed),
            extensions=extensions,
            schema=EPISODE_SCHEMA,
            schema_version=version,
        )
        supplied_hash = data.get("episode_hash")
        if supplied_hash is not None and str(supplied_hash) != episode.episode_hash:
            raise EpisodeContractError(
                "DesignEpisode episode_hash mismatch: the replayed episode is not "
                "the one the hash was computed for"
            )
        return episode

    # -- conversions -------------------------------------------------------

    @classmethod
    def from_search_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        *,
        spec_hash: str,
        environment_manifest_hash: str,
        initial_state_hash: str | None = None,
        episode_id: str | None = None,
        final_graph_ids: Iterable[str] = (),
        termination_reason: str = "frontier_exhausted",
        budget: Mapping[str, float] | None = None,
        random_seed: int | None = None,
        cost: Mapping[str, float] | None = None,
        structural_status: str = STRUCTURAL_VALID,
        validate_chain: bool = True,
    ) -> "DesignEpisode":
        """Convert ``SearchActionRecord`` mappings of one trajectory into steps.

        ``records`` must describe a single lineage (a ``GraphSearchState``
        derivation), which is what makes ``state_before_hash`` exact: accepted
        records move the state to their ``graph_hash``, rejected records leave
        it untouched and carry the attempted state in ``failure_category``.
        """

        attempted = tuple(records)
        steps: list[DesignStep] = []
        events: list[dict[str, Any]] = []
        expected = initial_state_hash
        for index, record in enumerate(attempted):
            payload = record.get("action") if isinstance(record, Mapping) else None
            accepted = bool(record.get("accepted", False)) if isinstance(record, Mapping) else False
            reason = str(record.get("reason", "") or "") if isinstance(record, Mapping) else ""
            graph_hash = (
                str(record["graph_hash"])
                if isinstance(record, Mapping) and record.get("graph_hash") is not None
                else None
            )
            parent_hash = (
                str(record["parent_hash"])
                if isinstance(record, Mapping) and record.get("parent_hash") is not None
                else None
            )
            if not payload:
                # A search-lifecycle event (for example max-depth pruning) is
                # not an action; keep it as evidence instead of inventing one.
                events.append(
                    {
                        "index": index,
                        "reason": reason,
                        "state_hash": parent_hash or graph_hash,
                    }
                )
                continue
            try:
                action = decode_action(payload)
            except EpisodeContractError as exc:
                raise EpisodeContractError(
                    f"search record {index} cannot be replayed: {exc}"
                ) from exc
            if expected is None:
                expected = parent_hash or empty_state_hash(spec_hash)
            before_hash = parent_hash or expected
            if accepted:
                after_hash: str | None = graph_hash or before_hash
                failure_category: str | None = None
                status = structural_status
                extensions: dict[str, Any] = {}
            else:
                after_hash = None
                failure_category = reason or "rejected"
                status = STRUCTURAL_NOT_APPLIED
                extensions = (
                    {"rejected_state_hash": graph_hash}
                    if graph_hash is not None and graph_hash != before_hash
                    else {}
                )
            steps.append(
                cls._step(
                    index=len(steps),
                    before_hash=before_hash,
                    action=action,
                    after_hash=after_hash,
                    structural_status=status,
                    failure_category=failure_category,
                    cost=cost,
                    extensions=extensions,
                )
            )
            expected = after_hash if after_hash is not None else before_hash

        extensions: dict[str, Any] = {}
        if events:
            extensions["search_events"] = events
        episode = cls(
            episode_id=episode_id or _derived_episode_id(spec_hash, expected or "", random_seed, len(steps)),
            spec_hash=spec_hash,
            environment_manifest_hash=environment_manifest_hash,
            initial_state_hash=(
                initial_state_hash
                if initial_state_hash is not None
                else (steps[0].state_before_hash if steps else empty_state_hash(spec_hash))
            ),
            steps=tuple(steps),
            final_graph_ids=tuple(final_graph_ids),
            termination_reason=termination_reason,
            budget=budget,
            random_seed=random_seed,
            extensions=extensions,
        )
        if validate_chain:
            validate_state_chain(episode)
        return episode

    @classmethod
    def from_search_certificate(
        cls,
        certificate: Any,
        *,
        spec_hash: str,
        environment_manifest_hash: str,
        initial_state_hash: str | None = None,
        episode_id: str | None = None,
        final_graph_ids: Iterable[str] = (),
        termination_reason: str | None = None,
        budget: Mapping[str, float] | None = None,
        random_seed: int | None = None,
    ) -> "DesignEpisode":
        """Convert a search certificate into a whole-search episode.

        Accepts a ``GraphSearchCertificate`` or its serialized mapping.  A
        certificate log interleaves lineages, so its steps are *not* a single
        state chain and the chain invariant is not asserted here; each step
        still carries the exact ``parent_hash`` recorded by the runner.
        """

        if isinstance(certificate, Mapping):
            payload: Any = certificate
        elif hasattr(certificate, "as_dict"):
            payload = certificate.as_dict()
        else:
            raise EpisodeContractError(
                f"unsupported search certificate {type(certificate).__name__}"
            )
        if not isinstance(payload, Mapping):
            raise EpisodeContractError("search certificate must serialize to a mapping")
        records = tuple(payload.get("records") or ())
        stop_reason = str(payload.get("stop_reason", "") or "")
        return cls.from_search_records(
            records,
            spec_hash=spec_hash,
            environment_manifest_hash=environment_manifest_hash,
            initial_state_hash=initial_state_hash,
            episode_id=episode_id,
            final_graph_ids=final_graph_ids,
            termination_reason=termination_reason or search_termination_reason(stop_reason),
            budget=budget,
            random_seed=random_seed,
            validate_chain=False,
        )

    @classmethod
    def from_candidate_graph(
        cls,
        graph: CircuitGraph,
        *,
        spec_hash: str,
        environment_manifest_hash: str,
        accepted: bool,
        episode_id: str | None = None,
        failure_category: str | None = None,
        structural_status: str | None = None,
        termination_reason: str | None = None,
        cost: Mapping[str, float] | None = None,
        budget: Mapping[str, float] | None = None,
        random_seed: int | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> "DesignEpisode":
        """Convert one retrieved candidate graph into a single-step episode.

        Runs that retrieve whole graphs (template catalog, knowledge-base
        production) make no incremental action; the retrieved graph is the
        one action that entered the design space, so it is recorded as an
        ``AddFragment`` of the complete graph.
        """

        graph.require_valid()
        initial = empty_state_hash(spec_hash)
        step = cls._step(
            index=0,
            before_hash=initial,
            action=AddFragment(fragment=graph, prefix=""),
            after_hash=graph.topology_hash if accepted else None,
            structural_status=(
                structural_status
                if structural_status is not None
                else STRUCTURAL_COMPLETE
                if accepted
                else STRUCTURAL_NOT_APPLIED
            ),
            failure_category=None if accepted else (failure_category or "rejected"),
            cost=cost,
            extensions={"candidate_provenance": dict(provenance)} if provenance else None,
        )
        episode = cls(
            episode_id=episode_id or _derived_episode_id(spec_hash, graph.topology_hash, random_seed, 1),
            spec_hash=spec_hash,
            environment_manifest_hash=environment_manifest_hash,
            initial_state_hash=initial,
            steps=(step,),
            final_graph_ids=(graph.graph_id,) if accepted else (),
            termination_reason=(
                termination_reason
                if termination_reason is not None
                else "target_reached"
                if accepted
                else "candidate_rejected"
            ),
            budget=budget,
            random_seed=random_seed,
        )
        validate_state_chain(episode)
        return episode

    @staticmethod
    def _step(
        *,
        index: int,
        before_hash: str,
        action: GraphAction,
        after_hash: str | None,
        structural_status: str,
        failure_category: str | None,
        cost: Mapping[str, float] | None,
        extensions: Mapping[str, Any] | None = None,
    ) -> DesignStep:
        return DesignStep(
            step_index=index,
            state_before_hash=before_hash,
            action=action,
            state_after_hash=after_hash,
            structural_status=structural_status,
            failure_category=failure_category,
            cost=cost,
            extensions=extensions or {},
        )


# --------------------------------------------------------------------------
# Candidate-level conversion and chain validation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeCandidate:
    """One evaluated candidate presented to the episode builder.

    ``records`` carries the candidate's ``GraphSearchState`` derivation when a
    graph search actually built it, so callers never rebuild action semantics
    from metadata themselves.
    """

    candidate_id: str
    graph: CircuitGraph
    accepted: bool
    records: tuple[Mapping[str, Any], ...] = ()
    failure_category: str | None = None
    cost: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)


def build_candidate_episodes(
    candidates: Iterable[EpisodeCandidate],
    *,
    spec_hash: str,
    environment_manifest_hash: str,
    budget: Mapping[str, float] | None = None,
    random_seed: int | None = None,
) -> tuple[DesignEpisode, ...]:
    """Build one episode per evaluated candidate, richest evidence first."""

    episodes: list[DesignEpisode] = []
    for candidate in candidates:
        if candidate.records:
            initial = next(
                (
                    str(record["parent_hash"])
                    for record in candidate.records
                    if isinstance(record, Mapping) and record.get("parent_hash")
                ),
                empty_state_hash(spec_hash),
            )
            episode = DesignEpisode.from_search_records(
                candidate.records,
                spec_hash=spec_hash,
                environment_manifest_hash=environment_manifest_hash,
                initial_state_hash=initial,
                episode_id=candidate.candidate_id,
                final_graph_ids=(candidate.graph.graph_id,),
                termination_reason=(
                    "target_reached" if candidate.accepted else "candidate_rejected"
                ),
                budget=budget,
                random_seed=random_seed,
            )
            episodes.append(_note_candidate_graph(episode, candidate.graph))
            continue
        episodes.append(
            DesignEpisode.from_candidate_graph(
                candidate.graph,
                spec_hash=spec_hash,
                environment_manifest_hash=environment_manifest_hash,
                accepted=candidate.accepted,
                episode_id=candidate.candidate_id,
                failure_category=candidate.failure_category,
                cost=candidate.cost,
                budget=budget,
                random_seed=random_seed,
                provenance=candidate.provenance,
            )
        )
    return tuple(episodes)


def _note_candidate_graph(episode: "DesignEpisode", graph: CircuitGraph) -> "DesignEpisode":
    """Record where candidate normalization diverged from the search lineage.

    Power candidates are rebuilt from their normalized topology record, which
    renames references and drops search extensions, so the candidate's
    topology hash can differ from the hash the search actually reached.  The
    episode keeps the search hash and names the candidate hash next to it
    instead of silently rewriting the chain.
    """

    applied = episode.applied_steps
    reached = applied[-1].state_after_hash if applied else None
    if reached is not None and reached == graph.topology_hash:
        return episode
    return replace(
        episode,
        extensions={
            **dict(episode.extensions),
            "candidate_graph": {
                "graph_id": graph.graph_id,
                "topology_hash": graph.topology_hash,
                "search_state_hash": reached,
            },
        },
    )


def candidate_accepted(
    validation: ConstraintReport | None,
    *,
    task_evaluations: Iterable[Any] = (),
    simulation_results: Iterable[SimulationResult] = (),
) -> bool:
    """The single acceptance rule shared by the replay label and the episode."""

    if validation is None:
        return False
    return (
        bool(validation.passed)
        and all(bool(getattr(task, "passed", True)) for task in task_evaluations)
        and all(result.status is SimulationStatus.PASSED for result in simulation_results)
    )


def episode_candidate(
    candidate_id: str,
    graph: CircuitGraph,
    *,
    validation: ConstraintReport | None = None,
    task_evaluations: Iterable[Any] = (),
    simulation_results: Iterable[SimulationResult] = (),
    cost: Mapping[str, float] | None = None,
    provenance: Mapping[str, Any] | None = None,
    records: Iterable[Mapping[str, Any]] = (),
) -> EpisodeCandidate:
    """Build one candidate's episode input, deriving acceptance and failure.

    Callers pass evidence, never a hand-computed label: the acceptance rule and
    the failure vocabulary stay identical to the rest of the evidence stream.
    """

    tasks = tuple(task_evaluations)
    results = tuple(simulation_results)
    accepted = candidate_accepted(
        validation,
        task_evaluations=tasks,
        simulation_results=results,
    )
    failure = None
    if not accepted:
        failure = candidate_failure_category(
            validation,
            task_evaluations=tasks,
            simulation_results=results,
        ) or "rejected"
    return EpisodeCandidate(
        candidate_id=str(candidate_id),
        graph=graph,
        accepted=accepted,
        # Only real ``SearchActionRecord`` mappings form a replayable lineage;
        # other derivation evidence stays in ``provenance`` untouched.
        records=tuple(
            item
            for item in records
            if isinstance(item, Mapping) and item.get("action")
        ),
        failure_category=failure,
        cost=dict(cost or {}),
        provenance=dict(provenance or {}),
    )


def search_episode(
    certificate: Any,
    *,
    spec_hash: str,
    environment_manifest_hash: str,
    episode_id: str = "search",
    budget: Mapping[str, float] | None = None,
    random_seed: int | None = None,
    final_graph_ids: Iterable[str] = (),
) -> "DesignEpisode | None":
    """Whole-search episode for a certificate, or ``None`` when it logged none.

    This is the episode that keeps rejected, duplicate, and pruned attempts;
    the per-candidate episodes of :func:`build_candidate_episodes` carry the
    physics evidence of one lineage.
    """

    records = (
        tuple(certificate.get("records") or ())
        if isinstance(certificate, Mapping)
        else tuple(getattr(certificate, "records", ()) or ())
    )
    if not records:
        return None
    return DesignEpisode.from_search_certificate(
        certificate,
        spec_hash=spec_hash,
        environment_manifest_hash=environment_manifest_hash,
        episode_id=episode_id,
        final_graph_ids=final_graph_ids,
        budget=budget,
        random_seed=random_seed,
    )


def write_episode_artifact(
    path: str | Path,
    candidates: Iterable[EpisodeCandidate],
    *,
    spec_hash: str,
    environment_manifest_hash: str,
    certificate: Any | None = None,
    budget: Mapping[str, float] | None = None,
    random_seed: int | None = None,
) -> int:
    """Write ``episode.jsonl`` beside a replay stream and return the row count."""

    episodes = list(
        build_candidate_episodes(
            candidates,
            spec_hash=spec_hash,
            environment_manifest_hash=environment_manifest_hash,
            budget=budget,
            random_seed=random_seed,
        )
    )
    if certificate is not None:
        whole = search_episode(
            certificate,
            spec_hash=spec_hash,
            environment_manifest_hash=environment_manifest_hash,
            budget=budget,
            random_seed=random_seed,
        )
        if whole is not None:
            episodes.append(whole)
    if not episodes:
        return 0
    return write_episodes(path, episodes)


def candidate_failure_category(
    validation: ConstraintReport | None,
    *,
    task_evaluations: Iterable[Any] = (),
    simulation_results: Iterable[SimulationResult] = (),
) -> str | None:
    """Classify why a candidate was not accepted; ``None`` means it was.

    Simulation statuses win over constraint verdicts because timeout,
    unsupported, and unavailable paths are the ones a model must be able to
    tell apart from a physics verdict.
    """

    for result in simulation_results:
        status = result.status
        if status is SimulationStatus.TIMEOUT:
            return "timeout"
        if status is SimulationStatus.UNSUPPORTED:
            return "unsupported"
        if status is SimulationStatus.UNAVAILABLE:
            return "unavailable"
        if status is SimulationStatus.FAILED:
            return "simulation_failed"
    if not all(bool(getattr(task, "passed", True)) for task in task_evaluations):
        return "task_failed"
    if validation is None:
        return None
    if validation.passed:
        return None
    feasibility = validation.feasibility
    if feasibility is FeasibilityStatus.KNOWN_INFEASIBLE:
        return "known_infeasible"
    if feasibility is FeasibilityStatus.INDETERMINATE:
        return "indeterminate"
    if feasibility is FeasibilityStatus.EXECUTION_FAILED:
        return "execution_failed"
    return "infeasible"


_SEARCH_TERMINATION_REASONS = {
    "target_count": "target_reached",
    "expansion_budget": "budget_exhausted",
    "simulation_budget": "budget_exhausted",
    "frontier_empty": "frontier_exhausted",
    "max_depth": "depth_limit",
}


def search_termination_reason(stop_reason: str) -> str:
    """Map a search stop reason onto the episode termination vocabulary."""

    return _SEARCH_TERMINATION_REASONS.get(str(stop_reason), str(stop_reason) or "unknown")


def validate_state_chain(episode: DesignEpisode) -> None:
    """Assert that the steps form one continuous, indexed state chain.

    A step whose ``state_after_hash`` is ``None`` leaves the state untouched
    (rejected, reverted, or timed-out action), so the next step must observe
    the same ``state_before_hash``.
    """

    expected = episode.initial_state_hash
    for position, step in enumerate(episode.steps):
        if step.step_index != position:
            raise EpisodeChainError(
                step.step_index,
                f"expected step_index {position} at position {position}",
                expected=position,
                actual=step.step_index,
            )
        if step.state_before_hash != expected:
            raise EpisodeChainError(
                step.step_index,
                "state_before_hash does not continue the previous step",
                expected=expected,
                actual=step.state_before_hash,
            )
        expected = step.state_after_hash if step.state_after_hash is not None else expected


def _derived_episode_id(
    spec_hash: str,
    state_hash: str,
    random_seed: int | None,
    step_count: int,
) -> str:
    digest = stable_payload_hash(
        {
            "spec_hash": str(spec_hash),
            "state_hash": str(state_hash),
            "random_seed": None if random_seed is None else int(random_seed),
            "step_count": int(step_count),
        }
    )
    return f"episode:{digest[:16]}"


def write_episodes(path: str | Path, episodes: Sequence[DesignEpisode]) -> int:
    """Write one episode dict per line and return the number of rows."""

    target = Path(path)
    rows = [episode.as_dict() for episode in episodes]
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return len(rows)


def load_episodes(path: str | Path) -> tuple[DesignEpisode, ...]:
    """Read an ``episode.jsonl`` stream back into episodes."""

    source = Path(path)
    return tuple(
        DesignEpisode.from_dict(json.loads(line))
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
