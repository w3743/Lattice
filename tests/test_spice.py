from __future__ import annotations

from dataclasses import replace
import math
import sys

from circuit_ai import AnalysisRequest, CircuitSynthesizer, SynthesisSpec
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.mna import LinearCircuit, LinearElement, VoltageSource
from circuit_ai.simulation import (
    AnalysisSpec,
    Excitation,
    LinearMNASimulatorBackend,
    NgspiceSimulatorBackend,
    ObservableSpec,
    SimulationRequest,
    SimulationExecutor,
    SimulationStatus,
    SweepSpec,
    build_ngspice_netlist,
    legacy_ac_simulation_request,
)
from circuit_ai.spice import NgspiceVerifier, apply_model_bindings, build_validation_netlist, parse_ngspice_ascii_raw
from circuit_ai.surrogate import export_verified_surrogate_rows, load_surrogate_rows


def _fake_raw() -> str:
    points = [10.0, 1000.0, 100000.0]
    lines = [
        "Title: fake ngspice",
        "Date: now",
        "Plotname: AC Analysis",
        "Flags: complex",
        "No. Variables: 2",
        f"No. Points: {len(points)}",
        "Variables:",
        "  0 frequency frequency",
        "  1 v(out) voltage",
        "Values:",
    ]
    for idx, freq in enumerate(points):
        h = 1.0 / (1.0 + 1j * freq / 1000.0)
        lines.append(f"{idx} {freq:.12g},0")
        lines.append(f"  {h.real:.12g},{h.imag:.12g}")
    return "\n".join(lines)


def _backend_fixture():
    graph = linear_circuit_to_graph(
        LinearCircuit(
            elements=(
                LinearElement("R1", "R", "in", "out", 1000.0),
                LinearElement("C1", "C", "out", "0", 159.154943e-9),
            ),
            voltage_sources=(VoltageSource("Vin", "in", "0", 1.0),),
        ),
        name="ngspice_backend_graph",
    )
    request = legacy_ac_simulation_request(
        graph,
        AnalysisRequest.voltage_transfer(),
        (10.0, 1000.0, 100000.0),
        request_id="ngspice_ac",
    )
    return graph, request


def test_parse_ngspice_ascii_raw_complex_values() -> None:
    raw = parse_ngspice_ascii_raw(_fake_raw())

    assert list(raw) == ["frequency", "v(out)"]
    assert len(raw["frequency"]) == 3
    assert math.isclose(raw["v(out)"][1].real, 0.5, rel_tol=1e-9)
    assert math.isclose(raw["v(out)"][1].imag, -0.5, rel_tol=1e-9)


def test_build_validation_netlist_replaces_analysis_block() -> None:
    netlist = "\n".join(["Vin in 0 AC 1", "R1 in out 1k", "C1 out 0 1n", ".ac dec 5 1 1k", ".end"])
    validation = build_validation_netlist(netlist, "run.raw", 10, 100000, 80)

    assert ".ac dec 5" not in validation
    assert "ac dec 80 10 100000" in validation
    assert "write run.raw all" in validation
    assert validation.strip().endswith(".end")


def test_apply_model_bindings_reorders_subcircuit_nodes_and_adds_include() -> None:
    netlist = "\n".join(["Vin in 0 AC 1", "Eop out 0 in 0 1e6", ".end"])
    bound = apply_model_bindings(
        netlist,
        [
            {
                "component": "Eop",
                "library": "models/opamp.lib",
                "name": "OPAMP_TEST",
                "pins": "1=OUT 2=0 3=INP 4=INM",
                "model_pins": ["INP", "INM", "OUT", "0"],
                "device": "X",
            }
        ],
    )

    assert '.include "models/opamp.lib"' in bound
    assert "XEop in 0 out 0 OPAMP_TEST" in bound
    assert "Eop out 0 in 0 1e6" not in bound


def test_build_validation_netlist_accepts_bound_model_netlist() -> None:
    validation = build_validation_netlist(
        "Vin in 0 AC 1\nR1 in out 1k\n.end\n",
        "run.raw",
        10,
        100000,
        80,
        model_bindings=[
            {
                "component": "R1",
                "library": "models/r.lib",
                "name": "R_REAL",
                "model_pins": ["1", "2"],
                "pins": "1=1 2=2",
                "device": "X",
            }
        ],
    )

    assert 'include "models/r.lib"' in validation
    assert "XR1 in out R_REAL" in validation


def test_ngspice_verifier_with_fake_executable(tmp_path) -> None:
    fake = tmp_path / "fake_ngspice.py"
    raw_text = _fake_raw()
    fake.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "netlist = Path(sys.argv[-1])",
                "raw = None",
                "for line in netlist.read_text().splitlines():",
                "    if line.strip().startswith('write '):",
                "        raw = line.split()[1]",
                "if raw is None:",
                "    raise SystemExit(2)",
                f"Path(raw).write_text({raw_text!r})",
            ]
        ),
        encoding="utf-8",
    )
    spec = SynthesisSpec.from_dict(
        {
            "name": "fake_spice_lowpass",
            "ports": 2,
            "behavior": {
                "kind": "lowpass",
                "cutoff_hz": 1000,
                "gain": 1.0,
                "order": 1,
                "frequency_range_hz": [10, 100000],
            },
            "library": {"allowed": ["R", "C"], "parameter_ranges": {"R": [100, 1000000], "C": [1e-10, 1e-4]}},
            "optimization": {"points": 48, "max_iterations": 20, "top_k": 1, "seed": 19},
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]
    verifier = NgspiceVerifier(executable=sys.executable, extra_args=(str(fake),), timeout_s=5)

    verification = verifier.verify(result, spec, output_dir=tmp_path / "spice")

    assert verification.status == "passed"
    assert verification.points == 3
    assert verification.rmse_db is not None
    assert verification.rmse_db < 1e-8
    assert len(verification.trace_frequency_hz) == 3
    assert len(verification.trace_magnitude_db) == 3

    dataset_path = tmp_path / "validated_rows.jsonl"
    summary = export_verified_surrogate_rows([result], [verification], dataset_path, append=False)
    assert summary["validated_rows"] == 1
    row = load_surrogate_rows(dataset_path)[0]
    assert row["provenance"]["validated"] is True
    assert row["frequency_hz"] == list(verification.trace_frequency_hz)


def test_unified_ngspice_backend_parses_ac_evidence_from_fake_process(tmp_path) -> None:
    fake = tmp_path / "fake_ngspice_backend.py"
    fake.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "netlist = Path(sys.argv[-1])",
                "raw = next(line.split()[1] for line in netlist.read_text().splitlines() if line.strip().startswith('write '))",
                f"Path(raw).write_text({_fake_raw()!r})",
            ]
        ),
        encoding="utf-8",
    )
    graph, request = _backend_fixture()
    backend = NgspiceSimulatorBackend(
        executable=sys.executable,
        extra_args=(str(fake),),
        timeout_s=5.0,
        backend_version="fake-1",
    )

    compiled = backend.compile(graph)
    assert backend.compile(graph) is compiled
    result = backend.simulate(compiled, request)

    expected = parse_ngspice_ascii_raw(_fake_raw())["v(out)"]
    numeric = SimulationExecutor().execute(
        LinearMNASimulatorBackend(),
        graph,
        (request,),
    )[0]
    numeric_values = numeric.waveforms["voltage_transfer"].values
    assert result.status is SimulationStatus.PASSED
    assert result.backend_version == "fake-1"
    assert result.waveforms["voltage_transfer"].axes[0].values == (
        10.0,
        1000.0,
        100000.0,
    )
    assert result.waveforms["voltage_transfer"].values == tuple(expected)
    assert max(
        abs(external - internal)
        for external, internal in zip(
            result.waveforms["voltage_transfer"].values,
            numeric_values,
        )
    ) < 1e-8
    assert result.evidence_hash


def test_unified_ngspice_backend_reports_unavailable_timeout_and_nonconvergence(tmp_path) -> None:
    graph, request = _backend_fixture()
    unavailable_backend = NgspiceSimulatorBackend(
        executable=str(tmp_path / "missing-ngspice.exe")
    )
    unavailable = unavailable_backend.simulate(
        unavailable_backend.compile(graph),
        request,
    )

    sleeper = tmp_path / "sleep_ngspice.py"
    sleeper.write_text("import time\ntime.sleep(1.0)\n", encoding="utf-8")
    timeout_backend = NgspiceSimulatorBackend(
        executable=sys.executable,
        extra_args=(str(sleeper),),
        timeout_s=5.0,
    )
    timeout = timeout_backend.simulate(
        timeout_backend.compile(graph),
        replace(request, timeout_s=0.05),
    )

    failing = tmp_path / "failing_ngspice.py"
    failing.write_text(
        "import sys\nprint('timestep too small; failed to converge', file=sys.stderr)\nraise SystemExit(1)\n",
        encoding="utf-8",
    )
    failing_backend = NgspiceSimulatorBackend(
        executable=sys.executable,
        extra_args=(str(failing),),
        timeout_s=5.0,
    )
    nonconverged = failing_backend.simulate(failing_backend.compile(graph), request)

    assert unavailable.status is SimulationStatus.UNAVAILABLE
    assert unavailable.diagnostics[0].code == "backend_unavailable"
    assert timeout.status is SimulationStatus.TIMEOUT
    assert timeout.diagnostics[0].code == "timeout"
    assert nonconverged.status is SimulationStatus.FAILED
    assert nonconverged.diagnostics[0].code == "nonconverged"


def test_unified_ngspice_backend_builds_dc_and_transient_commands() -> None:
    graph, _ = _backend_fixture()
    backend = NgspiceSimulatorBackend()
    compiled = backend.compile(graph)
    dc_request = SimulationRequest(
        request_id="ngspice_dc",
        analysis=AnalysisSpec("dc_operating_point"),
        graph_id=graph.graph_id,
        excitations=(Excitation("vin_dc", "dc_voltage", "component", "Vin", {"value": 5.0}),),
        requested_observables=(
            ObservableSpec("vout", "voltage", "V", "port", "output"),
        ),
        fidelity="spice",
    )
    transient_request = SimulationRequest(
        request_id="ngspice_transient",
        analysis=AnalysisSpec(
            "transient",
            sweeps=(SweepSpec("time", "s", start=0.0, stop=1e-3, points=101),),
        ),
        graph_id=graph.graph_id,
        excitations=(
            Excitation(
                "vin_sine",
                "sine",
                "component",
                "Vin",
                {"offset_v": 0.0, "amplitude_v": 1.0, "frequency_hz": 1000.0},
            ),
        ),
        requested_observables=(
            ObservableSpec("vout", "voltage", "V", "port", "output", representation="waveform"),
        ),
        fidelity="spice",
    )

    dc_netlist = build_ngspice_netlist(compiled, dc_request, "dc.raw")
    transient_netlist = build_ngspice_netlist(
        compiled,
        transient_request,
        "transient.raw",
    )

    assert "\nop\n" in dc_netlist
    assert " DC 5" in dc_netlist
    assert "\ntran 1e-05 0.001 0\n" in transient_netlist
    assert "SIN(0 1 1000 0 0 0)" in transient_netlist
