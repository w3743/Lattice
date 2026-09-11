from circuit_ai.catalog import compatible_topology_records
from circuit_ai.spec import LibrarySpec, SynthesisSpec


def _spec(real_components):
    return SynthesisSpec.from_dict({
        "name": "real_component_filter",
        "ports": 2,
        "analysis": {"kind": "voltage_transfer"},
        "behavior": {"kind": "lowpass", "cutoff_hz": 1000},
        "library": {"allowed": ["R", "C"], "real_components": real_components},
        "optimization": {"points": 8, "max_iterations": 2},
    })


def test_selected_real_parts_must_cover_template_families():
    assert any(record.name == "rc_lowpass" for record in compatible_topology_records(_spec(["Device:R", "Device:C"])))
    assert not any(record.name == "rc_lowpass" for record in compatible_topology_records(_spec(["Device:R"])))


def test_empty_real_part_selection_preserves_abstract_library_behavior():
    library = LibrarySpec.from_dict({"allowed": ["R", "C"]})
    assert library.supports_real_component_family("R")
    assert library.supports_real_component_family("C")
