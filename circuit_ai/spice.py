from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import spice_discovery
from .formatting import db20
from .targets import target_from_behavior


@dataclass(frozen=True)
class SpiceVerificationResult:
    backend: str
    status: str
    executable: str
    points: int
    rmse_db: float | None
    max_abs_db: float | None
    phase_rmse_deg: float | None
    raw_path: str | None
    message: str
    trace_frequency_hz: tuple[float, ...] = ()
    trace_magnitude_db: tuple[float, ...] = ()
    trace_phase_deg: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "status": self.status,
            "executable": self.executable,
            "points": self.points,
            "rmse_db": self.rmse_db,
            "max_abs_db": self.max_abs_db,
            "phase_rmse_deg": self.phase_rmse_deg,
            "raw_path": self.raw_path,
            "message": self.message,
            "trace_points": len(self.trace_frequency_hz),
        }


def apply_model_bindings(
    netlist: str,
    bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> str:
    """Rewrite ideal element cards into bound SPICE model instances."""
    if not bindings:
        return netlist
    by_component = {
        str(binding.get("component", binding.get("source_name", ""))).casefold(): dict(binding)
        for binding in bindings
    }
    include_paths: list[str] = []
    include_keys: set[str] = set()
    rewritten: list[str] = []
    used: set[str] = set()

    for line in netlist.splitlines():
        tokens = line.split()
        binding = by_component.get(tokens[0].casefold()) if tokens else None
        if binding is None or not tokens or tokens[0].startswith(".") or tokens[0].startswith("*"):
            rewritten.append(line)
            continue

        component = tokens[0]
        model_name = str(binding.get("name", "")).strip()
        library = str(binding.get("library", "")).strip()
        device = str(binding.get("device", "X")).strip().upper() or "X"
        mapping = _parse_sim_pins(str(binding.get("pins", "")))
        if not model_name or not library:
            raise ValueError(f"model binding for {component!r} requires library and name")
        if not mapping:
            raise ValueError(f"model binding for {component!r} requires a Sim.Pins mapping")

        symbol_pins = tuple(str(index) for index in range(1, len(mapping) + 1))
        if set(mapping) != set(symbol_pins):
            raise ValueError(f"model binding for {component!r} must map symbol pins 1..{len(mapping)}")
        actual_nodes = tokens[1 : 1 + len(symbol_pins)]
        if len(actual_nodes) != len(symbol_pins):
            raise ValueError(f"netlist card {component!r} has too few nodes for its model binding")

        if device == "X":
            model_pins = _model_pin_order(binding, mapping)
            node_by_model_pin = {
                mapping[symbol_pin]: actual_nodes[int(symbol_pin) - 1]
                for symbol_pin in symbol_pins
            }
            try:
                model_nodes = [node_by_model_pin[model_pin] for model_pin in model_pins]
            except KeyError as exc:
                raise ValueError(f"model pin {exc.args[0]!r} is missing from binding for {component!r}") from exc
            rewritten.append(f"X{component} {' '.join(model_nodes)} {model_name}")
        elif device == component[0].upper():
            value_index = 1 + len(symbol_pins)
            if len(tokens) <= value_index:
                raise ValueError(f"netlist card {component!r} has no value for model binding")
            rewritten.append(" ".join(tokens[: value_index + 1] + [model_name] + tokens[value_index + 1 :]))
        else:
            raise ValueError(f"model device {device!r} is incompatible with netlist card {component!r}")

        used.add(tokens[0].casefold())
        key = library.casefold()
        if key not in include_keys:
            include_keys.add(key)
            include_paths.append(library)

    unused = sorted(set(by_component) - used)
    if unused:
        raise ValueError(f"model binding target(s) not found in netlist: {', '.join(unused)}")
    includes = [f".include {_spice_quote(path)}" for path in include_paths]
    return "\n".join(includes + rewritten) + "\n"


def _parse_sim_pins(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for token in text.split():
        symbol_pin, separator, model_pin = token.partition("=")
        if not separator or not symbol_pin.strip() or not model_pin.strip():
            raise ValueError(f"invalid Sim.Pins mapping token: {token!r}")
        symbol_pin = symbol_pin.strip()
        if symbol_pin in mapping:
            raise ValueError(f"duplicate symbol pin in Sim.Pins mapping: {symbol_pin!r}")
        mapping[symbol_pin] = model_pin.strip()
    return mapping


def _model_pin_order(binding: dict[str, Any], mapping: dict[str, str]) -> tuple[str, ...]:
    raw = binding.get("model_pins", ())
    if isinstance(raw, str):
        pins = tuple(item for item in raw.split() if item)
    else:
        pins = tuple(str(item).strip() for item in raw if str(item).strip())
    if not pins:
        pins = tuple(mapping[str(index)] for index in range(1, len(mapping) + 1))
    if len(pins) != len(mapping) or set(pins) != set(mapping.values()):
        raise ValueError("model_pins must contain each Sim.Pins model node exactly once")
    return pins


def _spice_quote(path: str) -> str:
    return '"' + path.replace('"', '\\"') + '"'


class NgspiceVerifier:
    """Optional external ngspice verification backend."""

    def __init__(
        self,
        executable: str = "ngspice",
        extra_args: tuple[str, ...] = (),
        timeout_s: float = 30.0,
        points_per_decade: int = 80,
    ):
        # Resolve once, here, so the path that is *checked* is the path that is
        # *run*. Resolving only inside available() reported success while the
        # invocation still used the bare name, which fails unless the program is
        # on PATH -- the very case discovery exists to handle.
        resolved, reason = spice_discovery.resolve(executable)
        self.executable = resolved
        self.discovery_reason = reason
        self.extra_args = extra_args
        self.timeout_s = timeout_s
        self.points_per_decade = points_per_decade

    def available(self) -> bool:
        if Path(self.executable).is_file():
            return True
        return shutil.which(self.executable) is not None

    def verify(
        self,
        result,
        spec,
        output_dir: str | Path | None = None,
    ) -> SpiceVerificationResult:
        if spec.analysis.kind != "voltage_transfer":
            return SpiceVerificationResult(
                backend="ngspice",
                status="skipped",
                executable=self.executable,
                points=0,
                rmse_db=None,
                max_abs_db=None,
                phase_rmse_deg=None,
                raw_path=None,
                message="ngspice verifier currently supports voltage_transfer analysis only",
            )
        if not self.available():
            return SpiceVerificationResult(
                backend="ngspice",
                status="unavailable",
                executable=self.executable,
                points=0,
                rmse_db=None,
                max_abs_db=None,
                phase_rmse_deg=None,
                raw_path=None,
                message=f"ngspice executable not found: {self.executable}",
            )

        base_dir = Path(output_dir) if output_dir else None
        if base_dir:
            base_dir.mkdir(parents=True, exist_ok=True)
            work_ctx = None
            # Resolve to an absolute path, and pass absolute paths to the
            # process below. ngspice is a GUI-capable program: given a netlist
            # argument it cannot open -- which a relative cwd plus a relative
            # argument makes easy to produce -- it falls back to an interactive
            # session instead of exiting, and the call then hangs until the
            # timeout instead of reporting a missing file.
            work_dir = base_dir.resolve()
        else:
            work_ctx = tempfile.TemporaryDirectory(prefix="circuit_ai_ngspice_")
            work_dir = Path(work_ctx.name).resolve()

        try:
            raw_path = work_dir / f"{result.template.name}_ngspice.raw"
            netlist_path = work_dir / f"{result.template.name}_verify.cir"
            model_bindings = _prepare_model_bindings(result.model_bindings, work_dir)
            netlist_path.write_text(
                build_validation_netlist(
                    result.netlist("ngspice_candidate"),
                    raw_filename=raw_path.name,
                    f_min_hz=spec.optimization.f_min_hz,
                    f_max_hz=spec.optimization.f_max_hz,
                    points_per_decade=self.points_per_decade,
                    model_bindings=model_bindings,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [self.executable, *self.extra_args, "-b", str(netlist_path)],
                cwd=str(work_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_s,
                check=False,
            )
            if completed.returncode != 0:
                return SpiceVerificationResult(
                    backend="ngspice",
                    status="error",
                    executable=self.executable,
                    points=0,
                    rmse_db=None,
                    max_abs_db=None,
                    phase_rmse_deg=None,
                    raw_path=str(raw_path),
                    message=(completed.stderr or completed.stdout).strip()[:1000],
                )
            if not raw_path.exists():
                return SpiceVerificationResult(
                    backend="ngspice",
                    status="error",
                    executable=self.executable,
                    points=0,
                    rmse_db=None,
                    max_abs_db=None,
                    phase_rmse_deg=None,
                    raw_path=str(raw_path),
                    message="ngspice completed but did not produce raw output",
                )

            raw = parse_ngspice_ascii_raw(raw_path.read_text(encoding="utf-8", errors="replace"))
            frequencies = np.real(raw["frequency"])
            response = _pick_output_response(raw)
            metrics = _compare_spice_response(spec.behavior, spec.analysis, frequencies, response)
            status = "passed" if metrics["rmse_db"] < 1.0 and metrics["max_abs_db"] < 6.0 else "failed"
            return SpiceVerificationResult(
                backend="ngspice",
                status=status,
                executable=self.executable,
                points=len(frequencies),
                rmse_db=metrics["rmse_db"],
                max_abs_db=metrics["max_abs_db"],
                phase_rmse_deg=metrics["phase_rmse_deg"],
                raw_path=str(raw_path),
                message="external SPICE verification completed",
                trace_frequency_hz=tuple(float(value) for value in frequencies),
                trace_magnitude_db=tuple(float(value) for value in db20(response)),
                trace_phase_deg=tuple(
                    float(value) for value in np.rad2deg(np.unwrap(np.angle(response)))
                ),
            )
        except subprocess.TimeoutExpired:
            return SpiceVerificationResult(
                backend="ngspice",
                status="error",
                executable=self.executable,
                points=0,
                rmse_db=None,
                max_abs_db=None,
                phase_rmse_deg=None,
                raw_path=None,
                message=f"ngspice timed out after {self.timeout_s:g}s",
            )
        except (KeyError, ValueError, np.linalg.LinAlgError) as exc:
            return SpiceVerificationResult(
                backend="ngspice",
                status="error",
                executable=self.executable,
                points=0,
                rmse_db=None,
                max_abs_db=None,
                phase_rmse_deg=None,
                raw_path=None,
                message=f"failed to parse or score ngspice output: {exc}",
            )
        finally:
            if work_ctx is not None:
                work_ctx.cleanup()


def build_validation_netlist(
    netlist: str,
    raw_filename: str,
    f_min_hz: float,
    f_max_hz: float,
    points_per_decade: int,
    model_bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
) -> str:
    netlist = apply_model_bindings(netlist, model_bindings)
    body = []
    skip_control = False
    for line in netlist.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower == ".control":
            skip_control = True
            continue
        if lower == ".endc":
            skip_control = False
            continue
        if skip_control:
            continue
        if lower.startswith(".ac") or lower == ".end":
            continue
        body.append(line)

    body.extend(
        [
            ".option filetype=ascii",
            ".control",
            "set filetype=ascii",
            f"ac dec {int(points_per_decade)} {f_min_hz:.12g} {f_max_hz:.12g}",
            f"write {raw_filename} all",
            "quit",
            ".endc",
            ".end",
        ]
    )
    return "\n".join(body) + "\n"


def _prepare_model_bindings(
    bindings: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    work_dir: Path,
) -> tuple[dict[str, Any], ...]:
    prepared: list[dict[str, Any]] = []
    destination_dir = work_dir / "models"
    for binding in bindings:
        item = dict(binding)
        source = Path(str(item.get("library", "")))
        candidates = [source]
        if not source.is_absolute():
            candidates.append(work_dir.parent / source)
        resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
        if resolved is not None:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination = destination_dir / resolved.name
            shutil.copy2(resolved, destination)
            item["library"] = str(destination)
        prepared.append(item)
    return tuple(prepared)


def parse_ngspice_ascii_raw(text: str) -> dict[str, np.ndarray]:
    lines = text.splitlines()
    var_count = _parse_header_int(lines, "No. Variables")
    point_count = _parse_header_int(lines, "No. Points")
    variables_start = _find_line(lines, "Variables:") + 1
    values_start = _find_line(lines, "Values:") + 1

    variables: list[str] = []
    for line in lines[variables_start:values_start - 1]:
        parts = line.split()
        if len(parts) >= 2:
            variables.append(parts[1].lower())
    if len(variables) != var_count:
        raise ValueError(f"expected {var_count} variables, found {len(variables)}")

    columns = [[] for _ in range(var_count)]
    current: list[complex] = []
    for line in lines[values_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        value_text = _strip_optional_point_index(stripped, expect_index=len(current) == 0)
        current.append(_parse_raw_value(value_text))
        if len(current) == var_count:
            for idx, value in enumerate(current):
                columns[idx].append(value)
            current = []

    if current:
        raise ValueError("raw Values section ended in the middle of a point")
    if any(len(column) != point_count for column in columns):
        found = [len(column) for column in columns]
        raise ValueError(f"expected {point_count} points, found {found}")
    return {name: np.asarray(values, dtype=np.complex128) for name, values in zip(variables, columns)}


def _parse_header_int(lines: list[str], key: str) -> int:
    prefix = key.lower() + ":"
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(prefix):
            return int(stripped.split(":", 1)[1].strip())
    raise ValueError(f"missing raw header {key!r}")


def _find_line(lines: list[str], marker: str) -> int:
    marker = marker.lower()
    for idx, line in enumerate(lines):
        if line.strip().lower() == marker:
            return idx
    raise ValueError(f"missing raw section {marker!r}")


def _strip_optional_point_index(text: str, expect_index: bool) -> str:
    if not expect_index:
        return text
    match = re.match(r"^\d+\s+(.+)$", text)
    return match.group(1) if match else text


def _parse_raw_value(text: str) -> complex:
    cleaned = text.strip().replace("(", "").replace(")", "")
    if "," in cleaned:
        real, imag = cleaned.split(",", 1)
        return complex(float(real), float(imag))
    parts = cleaned.split()
    if len(parts) >= 2:
        return complex(float(parts[0]), float(parts[1]))
    return complex(float(parts[0]), 0.0)


def _pick_output_response(raw: dict[str, np.ndarray]) -> np.ndarray:
    for key in ("v(out)", "out"):
        if key in raw:
            return raw[key]
    voltage_keys = [key for key in raw if key.startswith("v(") and key != "v(in)"]
    if voltage_keys:
        return raw[voltage_keys[0]]
    raise KeyError("raw output has no v(out) vector")


def _compare_spice_response(
    behavior: dict[str, Any],
    analysis,
    frequencies_hz: np.ndarray,
    response: np.ndarray,
) -> dict[str, float]:
    target = target_from_behavior(behavior, frequencies_hz, analysis)
    db_error = db20(response) - db20(target.values)
    phase_error = np.angle(response / np.maximum(np.abs(response), 1e-18)) - np.angle(
        target.values / np.maximum(np.abs(target.values), 1e-18)
    )
    phase_error = np.angle(np.exp(1j * phase_error))
    return {
        "rmse_db": float(np.sqrt(np.mean(db_error**2))),
        "max_abs_db": float(np.max(np.abs(db_error))),
        "phase_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(phase_error**2)))),
    }
