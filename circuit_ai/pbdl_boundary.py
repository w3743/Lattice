"""The single specification boundary for the workbench.

PBDL is the public contract.  The legacy workbench JSON is accepted only at
this boundary and is immediately normalized into a ``CircuitSpec`` before any
planner, optimizer, or simulator sees it.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

PBDL_MODULE = "\u7aef\u53e3\u63cf\u8ff0\u8bed\u8a00"


def _pbdl():
    return importlib.import_module(PBDL_MODULE)


def load_pbdl_dict(data: dict[str, Any]):
    """Load canonical PBDL or normalize one older workbench payload."""
    module = _pbdl()
    if isinstance(data.get("ports"), list) and (
        "functions" in data or ("analyses" in data and "targets" in data)
    ):
        return module.CircuitSpec.from_dict(data)
    return module.CircuitSpec.from_dict(_legacy_to_pbdl(data))


def canonicalize_pbdl_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return the only schema that may cross the application boundary."""
    return load_pbdl_dict(data).as_dict()


def translate_pbdl_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Translate canonical PBDL into the internal execution IR.

    The PBDL translation layer emits a fixed key set derived from a
    ``CircuitSpec``, so anything a legacy two-port spec keeps *outside* that set
    is dropped on the way through -- notably the band-acceptance mask, but also
    ``gain`` and ``order``.  Those keys are requirement-level information, not
    noise, so they are re-attached here rather than silently lost.
    """

    module = _pbdl()
    translated = module.to_circuit_ai_spec(load_pbdl_dict(data))
    return _reattach_requirement_keys(translated, data)


#: Keys a two-port spec may declare that the PBDL round trip does not represent.
_CARRIED_REQUIREMENT_KEYS = ("filter_mask",)


def _reattach_requirement_keys(
    translated: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Carry requirement keys the PBDL translation cannot represent.

    They are re-attached **inside** ``behavior`` because that is the only
    container the frequency-domain synthesis spec exposes; a top-level key would
    survive this function and then be dropped by ``SynthesisSpec.from_dict``.
    A key may be declared at the spec's top level or already inside
    ``behavior``; the behaviour form is the natural place for a filter
    requirement and the top-level form is the fallback for specs that do not use
    ``behavior`` at all.
    """

    behavior = dict(translated.get("behavior") or {})
    original = source.get("behavior")
    original = original if isinstance(original, dict) else {}
    for key in _CARRIED_REQUIREMENT_KEYS:
        if key in behavior:
            continue
        value = original.get(key, source.get(key))
        if value is not None:
            behavior[key] = value
    if behavior:
        translated["behavior"] = behavior
    return translated


def _legacy_to_pbdl(data: dict[str, Any]) -> dict[str, Any]:
    interface = dict(data.get("interface", {}))
    raw_ports = interface.get("ports", [])
    if not raw_ports:
        raw_ports = [
            {"name": "input", "terminals": {"positive": "in", "negative": "0"}},
            {"name": "output", "terminals": {"positive": "out", "negative": "0"}},
        ][: max(2, int(data.get("ports", 2)))]
    ports = [_legacy_port(item, index) for index, item in enumerate(raw_ports)]
    relations = list(interface.get("relations", interface.get("measurements", [])))

    behavior = dict(data.get("behavior", {}))
    analysis = dict(data.get("analysis", {}))
    if not analysis:
        analysis = {"kind": "voltage_transfer"}
    if behavior.get("kind") in {"dc_conversion", "isolated_dc_conversion"}:
        analysis = {
            "kind": "dc_transfer",
            "source_port": behavior.get("input_port", "input"),
            "output_port": behavior.get("output_port", "output"),
        }

    target = _legacy_target(behavior)
    library = dict(data.get("library", {}))
    constraints = {
        "element_types": library.get("allowed", library.get("element_types", [])),
        "required_elements": library.get("required", library.get("required_elements", [])),
        "parameter_ranges": library.get("parameter_ranges", {}),
        "real_components": library.get("real_components", []),
        "model_bindings": library.get("model_bindings", []),
        "unit_costs": library.get("unit_costs", {}),
        "unit_areas_mm2": library.get("unit_areas_mm2", {}),
        "max_component_count": data.get("optimization", {}).get("max_components", 8),
    }
    return {
        "name": data.get("name", "unnamed"),
        "description": data.get("description", data.get("_description", "")),
        "ports": ports,
        "relations": relations,
        "functions": list(data.get("functions", [])),
        "analyses": [analysis],
        "targets": [target],
        "constraints": constraints,
        "operating_point": data.get("operating_point", {}),
        "optimization": data.get("optimization", {}),
    }


def _legacy_port(item: dict[str, Any], index: int) -> dict[str, Any]:
    terminals = item.get("terminals", {})
    if isinstance(terminals, dict):
        terminals = [
            {"name": terminals.get("positive", f"p{index + 1}"), "quantity": "voltage"},
            {"name": terminals.get("negative", "0"), "quantity": "ground"},
        ]
    variables = []
    for variable in item.get("variables", []):
        variables.append(
            {
                "name": variable.get("name", variable.get("symbol", "v")),
                "quantity": variable.get("quantity", "unspecified"),
                "role": variable.get("role", "potential"),
                "unit": variable.get("unit", ""),
                "expression": variable.get("expression", variable.get("definition")),
            }
        )
    constraints = []
    for constraint in item.get("variable_constraints", []):
        value = dict(constraint)
        analysis = value.get("analysis", {"kind": "dc_operating_point"})
        value["analysis"] = {"kind": analysis} if isinstance(analysis, str) else analysis
        constraints.append(value)
    return {
        "name": item.get("name", f"port_{index + 1}"),
        "terminals": terminals,
        "description": item.get("description", ""),
        "variables": variables,
        "variable_constraints": constraints,
        "role": item.get("role", "bidirectional"),
        "domain": item.get("domain", "unspecified"),
        "excitation": item.get("excitation"),
    }


def _legacy_target(behavior: dict[str, Any]) -> dict[str, Any]:
    kind = str(behavior.get("kind", "lowpass"))
    if kind in {"dc_conversion", "isolated_dc_conversion"}:
        return {
            "target_kind": "dc",
            "input_voltage_v": behavior.get("input_voltage_v"),
            "output_voltage_v": behavior.get("output_voltage_v"),
            "output_current_a": behavior.get("output_current_a"),
            "efficiency": behavior.get("efficiency"),
            "ripple_mv": behavior.get("ripple_mv"),
        }
    if kind in {"lowpass", "highpass", "bandpass", "bandstop", "allpass"}:
        target = {
            "target_kind": "filter",
            "kind": kind,
            "cutoff_hz": behavior.get("cutoff_hz", behavior.get("center_hz", 1000.0)),
            "order": behavior.get("order", 1),
            "q": behavior.get("q"),
        }
        if behavior.get("gain") is not None:
            target["gain_db"] = 20.0 * math.log10(max(float(behavior["gain"]), 1e-18))
        return target
    if kind == "transimpedance":
        gain = float(behavior.get("transimpedance_ohm", 1.0))
        return {
            "target_kind": "amplifier",
            "gain_db": 20.0 * math.log10(max(gain, 1e-18)),
            "bandwidth_hz": [
                float(behavior.get("frequency_range_hz", [10.0, behavior.get("cutoff_hz", 100_000.0)])[0]),
                float(behavior.get("cutoff_hz", 100_000.0)),
            ],
        }
    if kind in {"impedance", "output_impedance", "constant_impedance", "constant_output_impedance"}:
        return {"target_kind": "impedance", "ohms": behavior.get("ohms", behavior.get("resistance_ohm", 50.0))}
    return {"target_kind": "filter", "kind": "lowpass", "cutoff_hz": 1000.0, "order": 1}
