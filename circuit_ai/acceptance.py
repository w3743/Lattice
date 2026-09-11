"""Acceptance domains: the plug-in points where a requirement is judged.

Three of these exist now -- the operating envelope, the frequency-band mask and
the load-step budget -- and they are structurally the same thing: each declares
an acceptable region, evaluates one chosen design pointwise across it, attributes
the worst case, and refuses to imply more than it proves.

Before this module each one had to be wired by hand at four near-identical result
sites, because the four power result classes are copies of one another.  Adding a
fourth domain meant four more edits that could silently disagree.  This module
turns that into one registration:

    register_domain(AcceptanceDomain(name=..., evaluate=...))

What a domain is given
---------------------
:class:`DesignContext` carries everything any domain could need: the spec IR, the
chosen parameter values, the nominal target, and the family's DC solver and loss
parameters.  A domain takes what it needs and ignores the rest, so the call site
does not grow when a domain starts needing something new.

What a domain must return
-------------------------
``None`` when the spec states no requirement for it, which is how "not checked"
stays distinct from "checked and failed".  Otherwise the domain's own record,
which must serialise through ``as_dict``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, replace
from typing import Any

__all__ = [
    "ACCEPTANCE_DOMAIN_NAMES",
    "AcceptanceDomain",
    "DesignContext",
    "acceptance_domains",
    "acceptance_fields",
    "apply_acceptance_domains",
    "domain_names",
    "domain_results",
    "register_domain",
]


@dataclass(frozen=True)
class DesignContext:
    """Everything an acceptance domain could need about one chosen design."""

    ir: Any
    design: Any
    nominal_input_voltage_v: float
    nominal_output_voltage_v: float
    nominal_output_current_a: float
    design_rail_v: float
    dc_solver: Callable[..., Any] | None = None
    loss_parameters: Any = None

    @property
    def nominal_output_power_w(self) -> float:
        return self.nominal_output_voltage_v * self.nominal_output_current_a


@dataclass(frozen=True)
class AcceptanceDomain:
    """One judgement applied to a finished design.

    ``name`` is both the registration key and the result attribute the record is
    written to, so a domain cannot be registered under one name and land on
    another.
    """

    name: str
    evaluate: Callable[[DesignContext], Any]
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("acceptance domain needs a name")
        if not callable(self.evaluate):
            raise TypeError(f"acceptance domain {self.name!r} needs a callable evaluate")


_REGISTRY: dict[str, AcceptanceDomain] = {}

#: Attribute names that already exist on the power result records.  A domain may
#: only register under one of these, because the results are frozen dataclasses
#: and cannot grow a field at run time.  Widening this set is a deliberate act,
#: which is the point: a silently dropped domain is worse than a loud error.
ACCEPTANCE_DOMAIN_NAMES: tuple[str, ...] = (
    "envelope_analysis",
    "load_step",
)


def register_domain(domain: AcceptanceDomain) -> AcceptanceDomain:
    """Register one acceptance domain, replacing any earlier one by that name."""

    if domain.name not in ACCEPTANCE_DOMAIN_NAMES:
        raise ValueError(
            f"acceptance domain {domain.name!r} has no field on the power result "
            f"records; known fields are {sorted(ACCEPTANCE_DOMAIN_NAMES)}"
        )
    _REGISTRY[domain.name] = domain
    return domain


def acceptance_domains() -> tuple[AcceptanceDomain, ...]:
    """Registered domains, in registration order."""

    return tuple(_REGISTRY.values())


def domain_names() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def domain_results(context: DesignContext) -> dict[str, Any]:
    """Evaluate every registered domain against one design.

    Only names that produce a record are returned, so an undeclared requirement
    leaves its field at its default rather than being written as an explicit
    ``None`` by this layer.
    """

    produced: dict[str, Any] = {}
    for domain in acceptance_domains():
        record = domain.evaluate(context)
        if record is not None:
            produced[domain.name] = record
    return produced


def acceptance_fields(context: DesignContext, *, result_type: Any) -> dict[str, Any]:
    """Domain records ready to splat into a frozen result constructor.

    This exists alongside :func:`apply_acceptance_domains` because the power
    optimizers build their result in one expression.  Returning the fields keeps
    the call site a single line and keeps the field names checked here rather
    than at every site.
    """

    produced = domain_results(context)
    if not produced:
        return {}
    known = {item.name for item in fields(result_type)}
    unknown = set(produced) - known
    if unknown:
        raise ValueError(
            f"acceptance domains produced fields the result does not have: {sorted(unknown)}"
        )
    return produced


def apply_acceptance_domains(result: Any, context: DesignContext) -> Any:
    """Return *result* with every declared domain's record attached.

    Uses ``dataclasses.replace``, so the result classes keep their explicit
    fields -- callers and serialisation are unchanged -- while the call sites no
    longer have to name each domain.
    """

    produced = acceptance_fields(context, result_type=type(result))
    if not produced:
        return result
    return replace(result, **produced)


def _clear_registry_for_tests() -> None:
    """Test hook: drop every registration."""

    _REGISTRY.clear()


#: Re-exported so a caller can inspect what a record looks like without
#: importing the domain module itself.
def as_dict_of(records: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: (record.as_dict() if hasattr(record, "as_dict") else record)
        for name, record in records.items()
    }
