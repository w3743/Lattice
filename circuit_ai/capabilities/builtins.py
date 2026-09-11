"""Built-in adapters registered through the same public capability contract."""

from __future__ import annotations

from pathlib import Path

from ..constraints import DeclarativeConstraintValidator
from ..metrics import builtin_metric_providers
from ..power import POWER_STAGE_MODELS
from ..power_kicad import write_boost_kicad_project
from ..power_svg import render_power_svg
from ..spice import NgspiceVerifier
from ..simulation import (
    AnalyticPowerSimulatorBackend,
    LinearMNASimulatorBackend,
    NgspiceSimulatorBackend,
    TorchMNASimulatorBackend,
)
from .contracts import CapabilityRegistration, CapabilityRole, CapabilityTarget
from .registry import CapabilityRegistry


_POWER_ANALYSES = frozenset({"dc_transfer", "dc_operating_point"})
_POWER_MODELS = frozenset(
    {"R", "C", "L", "ideal_switch", "ideal_diode", "ideal_transformer"}
)
# Derived from the stage registry so the capability layer has one data source
# for the ideal power stages instead of a second hard-coded family table.
_POWER_FAMILIES = frozenset(stage.family for stage in POWER_STAGE_MODELS)
_POWER_SOLVERS = frozenset(stage.solver_id for stage in POWER_STAGE_MODELS)


class _PowerSvgRenderer:
    def render(self, result) -> str:
        return render_power_svg(result)


class _PowerKicadExporter:
    def export(self, result, output_dir, *, project_name: str = "best"):
        return write_boost_kicad_project(result, Path(output_dir), project_name=project_name)


class _NgspiceVerificationBackend:
    def __init__(self, verifier: NgspiceVerifier | None = None) -> None:
        self.verifier = verifier or NgspiceVerifier()

    def verify(self, result, spec, output_dir=None):
        return self.verifier.verify(result, spec, output_dir)

    def available(self) -> tuple[bool, str]:
        available = self.verifier.available()
        return available, "ngspice executable is available" if available else "ngspice executable is unavailable"


def register_builtin_capabilities(registry: CapabilityRegistry) -> CapabilityRegistry:
    """Populate a registry without importing or executing external plugins."""

    for stage in POWER_STAGE_MODELS:
        registry.register(
            CapabilityRegistration(
                CapabilityTarget(
                    capability_id=stage.optimizer_capability_id,
                    version="1.0.0",
                    role=CapabilityRole.PARAMETER_OPTIMIZER,
                    description=(
                        f"Parameter optimizer for the {stage.solver_id} ideal power model"
                    ),
                    families=frozenset({stage.family}),
                    solver_ids=frozenset({stage.solver_id}),
                    analysis_kinds=_POWER_ANALYSES,
                    model_kinds=_POWER_MODELS,
                    fidelity="ideal_averaged",
                    priority=100,
                ),
                stage.optimizer_type(),
            )
        )
        registry.register(
            CapabilityRegistration(
                CapabilityTarget(
                    capability_id=stage.simulation_capability_id,
                    version="1.0.0",
                    role=CapabilityRole.SIMULATION_BACKEND,
                    description=(
                        f"Unified simulation adapter for the {stage.solver_id} equations"
                    ),
                    families=frozenset({stage.family}),
                    solver_ids=frozenset({stage.solver_id}),
                    analysis_kinds=_POWER_ANALYSES,
                    model_kinds=_POWER_MODELS,
                    fidelity="ideal_averaged",
                    priority=100,
                ),
                AnalyticPowerSimulatorBackend(
                    backend_id=stage.solver_id,
                    solver_id=stage.solver_id,
                    parameter_type=stage.parameter_type,
                    dc_solver=stage.dc_solver,
                    model_kinds=_POWER_MODELS,
                ),
            )
        )

    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="constraints.declarative.v1",
                version="1.0.0",
                role=CapabilityRole.CONSTRAINT_VALIDATOR,
                description="Topology-neutral metric and declarative constraint evaluation",
                families=_POWER_FAMILIES,
                solver_ids=_POWER_SOLVERS,
                analysis_kinds=_POWER_ANALYSES,
                model_kinds=_POWER_MODELS,
                fidelity="ideal_averaged",
                priority=100,
            ),
            DeclarativeConstraintValidator(),
        )
    )
    for provider in builtin_metric_providers():
        registry.register(
            CapabilityRegistration(
                CapabilityTarget(
                    capability_id=f"metrics.{provider.provider_id}",
                    version="1.0.0",
                    role=CapabilityRole.METRIC_PROVIDER,
                    description=f"Built-in {provider.provider_id} metric provider",
                    priority=100,
                ),
                provider,
            )
        )
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="power.renderer.svg",
                version="1.0.0",
                role=CapabilityRole.SCHEMATIC_RENDERER,
                description="Browser SVG renderer for registered ideal power stages",
                families=_POWER_FAMILIES,
                solver_ids=_POWER_SOLVERS,
                model_kinds=_POWER_MODELS,
                export_formats=frozenset({"svg"}),
                priority=100,
            ),
            _PowerSvgRenderer(),
        )
    )
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="power.exporter.kicad",
                version="1.0.0",
                role=CapabilityRole.PROJECT_EXPORTER,
                description="KiCad schematic and project exporter for ideal power stages",
                families=_POWER_FAMILIES,
                solver_ids=_POWER_SOLVERS,
                model_kinds=_POWER_MODELS,
                export_formats=frozenset({"kicad"}),
                priority=100,
            ),
            _PowerKicadExporter(),
        )
    )
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="simulation.linear_mna.ac",
                version="1.0.0",
                role=CapabilityRole.SIMULATION_BACKEND,
                description="Internal linear modified nodal analysis backend",
                solver_ids=frozenset({"linear_mna"}),
                analysis_kinds=frozenset(
                    {
                        "small_signal_ac",
                        "voltage_transfer",
                        "transimpedance",
                        "input_impedance",
                        "output_impedance",
                    }
                ),
                model_kinds=frozenset({"R", "C", "L", "voltage_source", "current_source", "vcvs"}),
                fidelity="linear_frequency_domain",
                priority=100,
            ),
            LinearMNASimulatorBackend(),
        )
    )
    torch_mna = TorchMNASimulatorBackend()
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="simulation.torch_mna.ac",
                version="1.0.0",
                role=CapabilityRole.SIMULATION_BACKEND,
                description="Batched differentiable linear MNA backend",
                solver_ids=frozenset({"torch_mna"}),
                analysis_kinds=frozenset({"small_signal_ac"}),
                model_kinds=frozenset(
                    {"R", "C", "L", "voltage_source", "current_source", "vcvs"}
                ),
                fidelity="differentiable",
                priority=90,
            ),
            torch_mna,
            availability=torch_mna.available,
        )
    )
    ngspice_simulator = NgspiceSimulatorBackend()
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="simulation.ngspice",
                version="1.0.0",
                role=CapabilityRole.SIMULATION_BACKEND,
                description="External ngspice AC, DC, and transient backend",
                solver_ids=frozenset({"ngspice"}),
                analysis_kinds=frozenset(
                    {"small_signal_ac", "dc_operating_point", "dc_transfer", "transient"}
                ),
                model_kinds=frozenset(
                    {"R", "C", "L", "voltage_source", "current_source", "vcvs"}
                ),
                fidelity="spice",
                priority=80,
            ),
            ngspice_simulator,
            availability=ngspice_simulator.available,
        )
    )
    ngspice = _NgspiceVerificationBackend()
    registry.register(
        CapabilityRegistration(
            CapabilityTarget(
                capability_id="verification.ngspice.ac",
                version="1.0.0",
                role=CapabilityRole.VERIFICATION_BACKEND,
                description="External ngspice AC voltage-transfer verification backend",
                solver_ids=frozenset({"ngspice"}),
                analysis_kinds=frozenset({"voltage_transfer"}),
                fidelity="spice",
                priority=100,
            ),
            ngspice,
            availability=ngspice.available,
        )
    )
    return registry


def build_default_registry(*, include_external_plugins: bool = False) -> CapabilityRegistry:
    registry = register_builtin_capabilities(CapabilityRegistry())
    if include_external_plugins:
        registry.discover_entry_points()
    return registry
