import json
from pathlib import Path
import subprocess

import pytest

from circuit_ai.lceda_export import LcedaClient, validate_export_plan
from circuit_ai.graph import CircuitGraph


def plan():
    return json.loads((Path(__file__).parents[1] / "examples/lceda_rc.json").read_text(encoding="utf-8"))


def test_explicit_plan_matches_four_physical_pins():
    graph, pins, expected = validate_export_plan(plan())
    assert {c.reference for c in graph.components} == {"R1", "C1"}
    assert expected == {("R1", "1"): "IN", ("R1", "2"): "OUT",
                        ("C1", "1"): "OUT", ("C1", "2"): "GND"}


def test_stale_device_selection_is_rejected_before_editor_mutation():
    data = plan()
    data["graph"]["components"][0]["parameters"][0]["value"] *= 2
    with pytest.raises(ValueError, match="different graph hash"):
        validate_export_plan(data)


def test_missing_device_and_conflicting_pin_mappings_are_rejected():
    data = plan()
    del data["devices"]["C1"]
    with pytest.raises(ValueError, match="missing device binding"):
        validate_export_plan(data)
    data = plan()
    data["devices"]["C1"]["terminal_pins"]["n"] = ["1"]
    with pytest.raises(ValueError, match="different nets"):
        validate_export_plan(data)


def test_wrong_nominal_library_value_is_rejected_even_with_current_hash():
    data = plan()
    data["graph"]["components"][0]["parameters"][0]["value"] *= 10
    data["graph_hash"] = CircuitGraph.from_dict(data["graph"]).graph_hash
    with pytest.raises(ValueError, match="nominal value differs"):
        validate_export_plan(data)


@pytest.mark.parametrize("stdout,code", [
    ('{"ok":false,"error":{"code":"IMPORT_FAILED"}}', 0),
    ('{"ok":true,"value":{}}', 1),
    ('editor startup text', 0),
])
def test_transport_rejects_process_and_envelope_failures(monkeypatch, stdout, code):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], code, stdout, ""))
    with pytest.raises(RuntimeError):
        LcedaClient("client").command("doctor")


def test_transport_closes_only_its_session_after_failure(monkeypatch, tmp_path):
    client = LcedaClient("client")
    calls = []

    def command(*args, **kwargs):
        calls.append(args)
        if args[0] == "open":
            return {"sessionId": "owned"}
        if args[0] == "invoke":
            return True
        return {"status": "closed"}

    monkeypatch.setattr(client, "command", command)
    with pytest.raises(ValueError, match="work failed"):
        with client.open(tmp_path / "sample.eprj2"):
            raise ValueError("work failed")
    assert calls[-1] == ("session", "close", "--session", "owned", "--destroy")


def test_success_envelope_without_closed_state_is_not_cleanup_success(monkeypatch, tmp_path):
    client = LcedaClient("client")
    monkeypatch.setattr(client, "command", lambda *args, **kwargs:
                        {"sessionId": "owned"} if args[0] == "open" else
                        True if args[0] == "invoke" else None)
    with pytest.raises(RuntimeError, match="not confirmed closed"):
        with client.open(tmp_path / "sample.eprj2"):
            pass
