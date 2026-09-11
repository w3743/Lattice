"""Built-in primitive model identities and terminal contracts."""

from __future__ import annotations

from .model import ModelRef, ModelTerminal


# The galvanic isolation device is the only element allowed to connect terminals that
# live in mutually isolated graph domains; every other component must stay inside one
# domain (see ``circuit_ai/knowledge/power_topologies.yaml`` "isolated_transformer").
ISOLATION_BARRIER_KIND = "ideal_transformer"


def primitive_model_ref(kind: str, terminal_count: int | None = None) -> ModelRef:
    normalized = kind.strip()
    key = normalized.casefold()
    definitions = {
        "r": ("builtin.linear.resistor", "R", (_passive("p"), _passive("n"))),
        "c": ("builtin.linear.capacitor", "C", (_passive("p"), _passive("n"))),
        "l": ("builtin.linear.inductor", "L", (_passive("p"), _passive("n"))),
        "voltage_source": (
            "builtin.source.voltage",
            "voltage_source",
            (_terminal("p", "output", "voltage"), _terminal("n", "output", "voltage")),
        ),
        "current_source": (
            "builtin.source.current",
            "current_source",
            (_terminal("p", "output", "current"), _terminal("n", "output", "current")),
        ),
        "vcvs": (
            "builtin.controlled_source.vcvs",
            "vcvs",
            (
                _terminal("p", "output", "voltage"),
                _terminal("n", "output", "voltage"),
                _terminal("control_p", "input", "voltage"),
                _terminal("control_n", "input", "voltage"),
            ),
        ),
        "ideal_switch": (
            "builtin.power.ideal_switch",
            "ideal_switch",
            (_passive("a"), _passive("b")),
        ),
        "ideal_diode": (
            "builtin.power.ideal_diode",
            "ideal_diode",
            (_passive("anode"), _passive("cathode")),
        ),
        "ideal_transformer": (
            "builtin.power.ideal_transformer",
            "ideal_transformer",
            (
                _passive("primary_p"),
                _passive("primary_n"),
                _passive("secondary_p"),
                _passive("secondary_n"),
            ),
        ),
        "opamp": (
            "builtin.active.ideal_opamp",
            "opamp",
            (
                _terminal("non_inverting", "input", "voltage"),
                _terminal("inverting", "input", "voltage"),
                _terminal("output", "output", "voltage"),
            ),
        ),
    }
    definition = definitions.get(key)
    if definition is not None:
        model_id, canonical_kind, terminals = definition
        if terminal_count is not None and terminal_count != len(terminals):
            raise ValueError(
                f"model {canonical_kind!r} expects {len(terminals)} terminals, got {terminal_count}"
            )
        return ModelRef(model_id, canonical_kind, terminals)
    count = int(terminal_count or 0)
    if count < 1:
        raise ValueError(f"unknown model kind {kind!r} requires terminal_count")
    return ModelRef(
        model_id=f"unresolved.{normalized}",
        kind=normalized,
        terminals=tuple(_passive(f"t{index + 1}") for index in range(count)),
        source="unresolved",
    )


def parameter_unit(kind: str, parameter: str) -> str:
    key = (kind.casefold(), parameter.casefold())
    return {
        ("r", "value"): "ohm",
        ("c", "value"): "F",
        ("l", "value"): "H",
        ("voltage_source", "value"): "V",
        ("current_source", "value"): "A",
        ("vcvs", "gain"): "1",
        ("ideal_switch", "duty_cycle"): "1",
        ("ideal_switch", "frequency_hz"): "Hz",
        ("ideal_transformer", "turns_ratio"): "1",
    }.get(key, "")


def expected_parameters(kind: str) -> tuple[str, ...]:
    return {
        "r": ("value",),
        "c": ("value",),
        "l": ("value",),
        "voltage_source": ("value",),
        "current_source": ("value",),
        "vcvs": ("gain",),
        "ideal_switch": ("duty_cycle", "frequency_hz"),
        "ideal_transformer": ("turns_ratio",),
    }.get(kind.casefold(), ())


def _passive(name: str) -> ModelTerminal:
    return _terminal(name, "passive", "voltage")


def _terminal(name: str, direction: str, quantity: str) -> ModelTerminal:
    return ModelTerminal(name=name, direction=direction, domain="electrical", quantity=quantity)
