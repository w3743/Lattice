"""Finding an external SPICE, and the bug that made it look unavailable.

Two things are pinned here.

First, discovery: this project reported ngspice as unavailable for a long time
because it only looked at ``PATH``. ngspice was installed all along, as a
component of another application, and a bundled copy works fine for batch use.
The wrong conclusion was not harmless -- it justified "no external verification is
possible" as a standing caveat.

Second, a hang: the verifier passed a *relative* working directory together with a
relative netlist path. ngspice is GUI-capable, so when it cannot open the netlist
argument it falls back to an interactive session instead of exiting, and the call
blocks until the timeout rather than reporting a missing file. Every manual test
had used an absolute path, which is why the bug survived.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from circuit_ai import spice_discovery
from circuit_ai.simulation import NgspiceSimulatorBackend
from circuit_ai.spice import NgspiceVerifier

# ---------------------------------------------------------------------------
# The absence convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "definitely_missing_ngspice_for_test",
        "definitely_missing_ngspice_for_power_test",
        "DEFINITELY_MISSING_anything",
    ],
)
def test_a_deliberately_missing_name_is_never_resolved(name) -> None:
    """A caller naming an executable this way is asking for *absence*.

    Several tests and specs rely on that, so discovery must not override it: on a
    machine that happens to have ngspice installed, resolving it would turn a
    deliberate availability test into a false pass.
    """

    assert spice_discovery.is_deliberately_missing(name)
    resolved, reason = spice_discovery.resolve(name)
    assert resolved == name
    assert "convention" in reason
    backend = NgspiceSimulatorBackend(name)
    assert backend.executable == name
    assert backend.available()[0] is False


def test_an_ordinary_name_is_not_treated_as_missing() -> None:
    assert not spice_discovery.is_deliberately_missing("ngspice")
    assert not spice_discovery.is_deliberately_missing("/usr/bin/ngspice")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_an_explicit_existing_path_is_used_verbatim(tmp_path) -> None:
    fake = tmp_path / "my_ngspice"
    fake.write_text("", encoding="utf-8")
    resolved, reason = spice_discovery.resolve("ngspice", explicit=fake)
    assert Path(resolved) == fake
    assert "explicit path" in reason


def test_an_explicit_missing_path_is_reported_rather_than_replaced(tmp_path) -> None:
    """An explicit path is an instruction, not a hint."""

    missing = tmp_path / "not_here"
    resolved, reason = spice_discovery.resolve("ngspice", explicit=missing)
    assert "does not exist" in reason
    assert Path(resolved) != missing or not Path(resolved).exists()


def test_discovery_can_be_disabled() -> None:
    backend = NgspiceSimulatorBackend("ngspice", discover=False)
    assert backend.discovery_reason == "discovery disabled"


def test_the_search_is_reported_even_when_it_fails() -> None:
    """An availability message has to say where it looked."""

    description = spice_discovery.describe_search("a_program_that_does_not_exist")
    assert "searched" in description
    candidates = spice_discovery.candidate_paths("a_program_that_does_not_exist")
    assert len(candidates) > 5


def test_environment_override_is_consulted_first(monkeypatch, tmp_path) -> None:
    fake = tmp_path / "ngspice"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv(spice_discovery.ENVIRONMENT_VARIABLES[0], str(fake))
    assert spice_discovery.find_executable("ngspice") == fake


def test_windows_looks_for_the_executable_extension() -> None:
    """A bare ``Path(dir) / "ngspice"`` misses ``ngspice.exe`` on Windows."""

    names = spice_discovery._names_for("ngspice")
    if names == ("ngspice",):
        pytest.skip("not Windows")
    assert "ngspice.exe" in names


# ---------------------------------------------------------------------------
# The hang
# ---------------------------------------------------------------------------


def test_the_verifier_uses_an_absolute_working_directory(monkeypatch, tmp_path) -> None:
    """The relative-cwd defect: it made ngspice hang instead of failing.

    The verifier is driven with a *relative* output directory, which is the shape
    that used to hang, and the recorded subprocess call is checked instead of
    waiting for a timeout.
    """

    import subprocess as subprocess_module

    from circuit_ai.spec import SynthesisSpec
    from circuit_ai.synthesis import CircuitSynthesizer

    recorded: list[dict] = []
    real_run = subprocess_module.run

    def recorder(argv, *args, **kwargs):
        recorded.append({"argv": list(argv), "cwd": kwargs.get("cwd")})
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess_module, "run", recorder)

    spec = SynthesisSpec.from_dict(
        {
            "name": "abs_cwd",
            "ports": 2,
            "behavior": {
                "kind": "lowpass", "cutoff_hz": 1000, "order": 1,
                "response": "butterworth", "gain": 1.0, "frequency_range_hz": [10, 100000],
            },
            "library": {"allowed": ["R", "C"], "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4]}},
            "optimization": {"points": 16, "max_iterations": 3, "top_k": 1, "seed": 3},
        }
    )
    result = CircuitSynthesizer().synthesize(spec)[0]

    monkeypatch.chdir(tmp_path)
    relative = Path("relative_out")
    verifier = NgspiceVerifier(timeout_s=20.0, points_per_decade=6)
    if not verifier.available():
        pytest.skip("no external ngspice on this machine")
    verifier.verify(result, spec, output_dir=relative)

    assert recorded, "the verifier must have invoked the external program"
    call = recorded[0]
    assert Path(call["cwd"]).is_absolute(), (
        "a relative cwd plus a relative netlist path makes ngspice fall back to "
        "an interactive session and hang instead of reporting the missing file"
    )
    assert Path(call["argv"][-1]).is_absolute(), call["argv"][-1]


# ---------------------------------------------------------------------------
# The real thing, when it is present
# ---------------------------------------------------------------------------


def _external_available() -> bool:
    return NgspiceSimulatorBackend().available()[0]


@pytest.mark.skipif(not _external_available(), reason="no external ngspice on this machine")
def test_internal_mna_agrees_with_ngspice(tmp_path) -> None:
    """The point of all of the above: an independent check of the internal model.

    Every "verified" verdict in this project previously meant "the internal model
    agrees with itself". This compares it with an external simulator, and the
    tolerance is deliberately tight so a real disagreement would show.
    """

    from circuit_ai.spec import SynthesisSpec
    from circuit_ai.synthesis import CircuitSynthesizer

    cases = {
        "lowpass": {"kind": "lowpass", "cutoff_hz": 1000, "order": 1},
        "highpass": {"kind": "highpass", "cutoff_hz": 1000, "order": 1},
        "bandpass": {"kind": "bandpass", "center_hz": 1000, "q": 3.0, "order": 2},
    }
    backend = NgspiceSimulatorBackend()
    for label, behavior in cases.items():
        spec = SynthesisSpec.from_dict(
            {
                "name": label,
                "ports": 2,
                "behavior": {
                    "response": "butterworth", "gain": 1.0,
                    "frequency_range_hz": [10, 100000], **behavior,
                },
                "library": {
                    "allowed": ["R", "C", "L"],
                    "parameter_ranges": {"R": [100, 1e6], "C": [1e-10, 1e-4], "L": [1e-6, 1.0]},
                },
                "optimization": {"points": 24, "max_iterations": 8, "top_k": 1, "seed": 3},
            }
        )
        result = CircuitSynthesizer().synthesize(spec)[0]
        verifier = NgspiceVerifier(
            executable=str(backend.executable), timeout_s=30.0, points_per_decade=10
        )
        verification = verifier.verify(result, spec, output_dir=tmp_path / label)

        assert verification.status == "passed", verification.message
        assert verification.max_abs_db is not None
        # A tenth of a dB is a loose bound on purpose: this is checking that the
        # two models describe the same circuit, not that they round identically.
        assert verification.max_abs_db < 0.1, (
            f"{label}: internal and external models disagree by "
            f"{verification.max_abs_db:.4g} dB"
        )
