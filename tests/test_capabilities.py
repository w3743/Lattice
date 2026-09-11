from __future__ import annotations

import inspect

import pytest

import circuit_ai.capabilities.registry as registry_module
import circuit_ai.pipeline as pipeline_module
from circuit_ai.capabilities import (
    CapabilityRegistration,
    CapabilityRegistrationError,
    CapabilityRegistry,
    CapabilityPlanError,
    CapabilityRequest,
    CapabilityResolutionError,
    CapabilityResolver,
    CapabilityRole,
    CapabilityTarget,
    build_default_registry,
)
from circuit_ai.pipeline import design_from_pbdl
from circuit_ai.power import BoostParameterOptimizer


class _Optimizer:
    def optimize(self, ir, candidate):
        return None


def _optimizer_registration(
    capability_id: str,
    *,
    priority: int = 0,
    family: str = "dc_boost",
    solver_id: str = "ideal_boost_averaged",
    supported_models: frozenset[str] = frozenset(),
    implementation=None,
) -> CapabilityRegistration:
    return CapabilityRegistration(
        CapabilityTarget(
            capability_id=capability_id,
            version="1.0.0",
            role=CapabilityRole.PARAMETER_OPTIMIZER,
            description="test optimizer",
            families=frozenset({family}),
            solver_ids=frozenset({solver_id}),
            analysis_kinds=frozenset({"dc_transfer"}),
            model_kinds=supported_models,
            priority=priority,
            source="test",
        ),
        implementation or _Optimizer(),
    )


def _request(**overrides) -> CapabilityRequest:
    values = {
        "role": CapabilityRole.PARAMETER_OPTIMIZER,
        "family": "dc_boost",
        "solver_id": "ideal_boost_averaged",
        "analysis_kinds": frozenset({"dc_transfer"}),
    }
    values.update(overrides)
    return CapabilityRequest(**values)


def _boost_spec() -> dict:
    return {
        "name": "capability_injection",
        "ports": [
            {
                "name": "input",
                "role": "input",
                "terminals": [
                    {"name": "in", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
            {
                "name": "output",
                "role": "output",
                "terminals": [
                    {"name": "out", "quantity": "voltage"},
                    {"name": "0", "quantity": "ground"},
                ],
            },
        ],
        "analyses": [
            {"kind": "dc_transfer", "source_port": "input", "output_port": "output"}
        ],
        "targets": [
            {
                "target_kind": "dc",
                "input_voltage_v": 5.0,
                "output_voltage_v": 10.0,
                "output_current_a": 1.0,
            }
        ],
        "constraints": {
            "element_types": ["R", "C", "L", "ideal_switch", "ideal_diode"],
            "max_component_count": 5,
        },
        "optimization": {"max_iterations": 2, "popsize": 4, "seed": 7},
    }


def test_registration_rejects_duplicate_global_id() -> None:
    registration = _optimizer_registration("test.optimizer")
    registry = CapabilityRegistry((registration,))

    with pytest.raises(CapabilityRegistrationError, match="duplicate capability id"):
        registry.register(registration)


def test_registration_checks_role_method_contract() -> None:
    with pytest.raises(TypeError, match="requires a callable optimize"):
        _optimizer_registration("test.invalid", implementation=object())


def test_resolver_selects_highest_priority_and_records_alternatives() -> None:
    registry = CapabilityRegistry(
        (
            _optimizer_registration("test.low", priority=10),
            _optimizer_registration("test.high", priority=20),
        )
    )

    resolution = CapabilityResolver(registry).resolve(_request())

    assert resolution.selected.target.capability_id == "test.high"
    assert [item.target.capability_id for item in resolution.alternatives] == ["test.low"]


def test_resolver_reports_structured_mismatch_reasons() -> None:
    registry = CapabilityRegistry(
        (
            _optimizer_registration(
                "test.buck",
                family="dc_buck",
                solver_id="ideal_buck_averaged",
                supported_models=frozenset({"R", "C", "L"}),
            ),
        )
    )

    with pytest.raises(CapabilityResolutionError) as caught:
        CapabilityResolver(registry).resolve(
            _request(
                analysis_kinds=frozenset({"dc_transfer", "transient"}),
                model_kinds=frozenset({"ideal_transformer"}),
            )
        )

    gap = caught.value.gap
    assert gap.request.family == "dc_boost"
    assert gap.rejections[0].capability_id == "test.buck"
    assert any("family 'dc_boost'" in reason for reason in gap.rejections[0].reasons)
    assert any("solver 'ideal_boost_averaged'" in reason for reason in gap.rejections[0].reasons)
    assert any("analysis kinds are missing" in reason for reason in gap.rejections[0].reasons)
    assert any("model kinds are missing" in reason for reason in gap.rejections[0].reasons)
    assert gap.as_dict()["request"]["role"] == "parameter_optimizer"


def test_resolver_does_not_probe_availability_for_static_mismatch() -> None:
    probes = []
    registration = CapabilityRegistration(
        CapabilityTarget(
            capability_id="test.external_verifier",
            version="1.0.0",
            role=CapabilityRole.VERIFICATION_BACKEND,
            description="external verifier",
        ),
        type("Verifier", (), {"verify": lambda self, *args: None})(),
        availability=lambda: probes.append(True) or False,
    )

    with pytest.raises(CapabilityResolutionError):
        CapabilityResolver(CapabilityRegistry((registration,))).resolve(_request())

    assert probes == []


def test_builtin_registry_covers_power_and_linear_backends() -> None:
    registry = build_default_registry()
    resolver = CapabilityResolver(registry)
    expected = {
        ("dc_buck", "ideal_buck_averaged"),
        ("dc_boost", "ideal_boost_averaged"),
        ("dc_sepic", "ideal_sepic_averaged"),
        ("isolated_flyback", "ideal_flyback_averaged"),
    }

    for family, solver_id in expected:
        resolution = resolver.resolve(_request(family=family, solver_id=solver_id))
        assert resolution.selected.target.families == frozenset({family})
        simulator = resolver.resolve(
            CapabilityRequest(
                role=CapabilityRole.SIMULATION_BACKEND,
                family=family,
                solver_id=solver_id,
                analysis_kinds=frozenset({"dc_transfer"}),
                model_kinds=frozenset({"R", "C", "L", "ideal_switch", "ideal_diode"}),
            )
        )
        assert simulator.selected.target.capability_id == f"simulation.power.{solver_id}"

    mna = resolver.resolve(
        CapabilityRequest(
            role=CapabilityRole.SIMULATION_BACKEND,
            solver_id="linear_mna",
            analysis_kinds=frozenset({"small_signal_ac"}),
            model_kinds=frozenset({"R", "C", "L"}),
        )
    )
    assert mna.selected.target.capability_id == "simulation.linear_mna.ac"
    torch_mna = resolver.resolve(
        CapabilityRequest(
            role=CapabilityRole.SIMULATION_BACKEND,
            solver_id="torch_mna",
            analysis_kinds=frozenset({"small_signal_ac"}),
            model_kinds=frozenset({"R", "C", "L"}),
        )
    )
    assert torch_mna.selected.target.capability_id == "simulation.torch_mna.ac"
    manifest = registry.manifest()
    assert all(item["version"] for item in manifest["capabilities"])
    capability_ids = {
        item["capability_id"] for item in manifest["capabilities"]
    }
    assert "simulation.ngspice" in capability_ids
    assert "constraints.declarative.v1" in capability_ids
    assert {
        "metrics.dc_port_metrics",
        "metrics.frequency_response",
        "metrics.graph.resource",
    } <= capability_ids


def test_external_plugins_are_loaded_only_when_explicitly_requested(monkeypatch) -> None:
    calls = []

    def unexpected_entry_points(**kwargs):
        calls.append(kwargs)
        return []

    monkeypatch.setattr(registry_module.metadata, "entry_points", unexpected_entry_points)

    build_default_registry()
    assert calls == []

    build_default_registry(include_external_plugins=True)
    assert calls == [{"group": "circuit_ai.capabilities"}]


def test_entry_point_contract_registers_loaded_capability(monkeypatch) -> None:
    registration = _optimizer_registration("plugin.optimizer")

    class EntryPoint:
        name = "test-plugin"
        value = "test_plugin:registration"

        @staticmethod
        def load():
            return registration

    monkeypatch.setattr(
        registry_module.metadata,
        "entry_points",
        lambda **kwargs: [EntryPoint()],
    )
    registry = CapabilityRegistry()

    loaded = registry.discover_entry_points()

    assert loaded == (registration,)
    assert registry.get("plugin.optimizer") is registration


def test_pipeline_uses_injected_registration_instead_of_family_mapping() -> None:
    class TrackingOptimizer:
        def __init__(self) -> None:
            self.calls = 0
            self.delegate = BoostParameterOptimizer()

        def optimize(self, ir, candidate):
            self.calls += 1
            return self.delegate.optimize(ir, candidate)

    tracker = TrackingOptimizer()
    registry = build_default_registry()
    registry.register(
        _optimizer_registration(
            "test.tracking_boost",
            priority=1000,
            implementation=tracker,
        )
    )

    result = design_from_pbdl(_boost_spec(), capability_registry=registry)

    assert result.succeeded
    assert tracker.calls == 1
    assert result.capability_resolutions[0].selected.target.capability_id == "test.tracking_boost"


def test_pipeline_preserves_structured_gaps_when_no_plan_exists(tmp_path) -> None:
    with pytest.raises(CapabilityPlanError) as caught:
        design_from_pbdl(
            _boost_spec(),
            tmp_path,
            capability_registry=CapabilityRegistry(),
        )

    assert caught.value.gaps
    assert caught.value.as_dict()["error_type"] == "capability_plan"
    gap_file = tmp_path / "capability_gaps.json"
    assert gap_file.exists()
    assert "parameter_optimizer" in gap_file.read_text(encoding="utf-8")


def test_pipeline_source_has_no_power_optimizer_dispatch_table() -> None:
    source = inspect.getsource(pipeline_module)

    assert "_optimizer_for" not in source
    assert "BoostParameterOptimizer" not in source
    assert "validate_boost" not in source
