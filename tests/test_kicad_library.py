from pathlib import Path

from circuit_ai.kicad_library import LibraryPart, SpiceModel, build_index, parse_spice_file, parse_symbol_file, search_index, validate_model_binding


def test_indexes_real_kicad_symbol_and_footprint_files(tmp_path: Path):
    root = tmp_path / "kicad"
    symbols = root / "symbols"
    footprints = root / "footprints" / "Package_SO.pretty"
    symbols.mkdir(parents=True)
    footprints.mkdir(parents=True)
    symbol_file = symbols / "Amplifier_Operational.kicad_sym"
    symbol_file.write_text(
        '''(kicad_symbol_lib (version 20231120) (generator kicad_symbol_editor)
  (symbol "LM358"
    (property "Reference" "U")
    (property "Value" "LM358")
    (property "Footprint" "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    (property "Description" "Dual operational amplifier")
    (property "ki_keywords" "dual opamp")
    (property "Sim.Device" "X")
    (pin input line (at 0 0 0) (length 2.54) (name "IN" (effects (font (size 1 1)))) (number "1" (effects (font (size 1 1)))))
    (pin output line (at 5 0 0) (length 2.54) (name "OUT" (effects (font (size 1 1)))) (number "2" (effects (font (size 1 1)))))))''',
        encoding="utf-8",
    )
    (footprints / "SOIC-8_3.9x4.9mm_P1.27mm.kicad_mod").write_text("(footprint \"SOIC-8\")", encoding="utf-8")
    (root / "models.lib").write_text(".SUBCKT LM358_TEST 1 2 3 4 5\n.ENDS LM358_TEST", encoding="utf-8")

    parsed = parse_symbol_file(symbol_file)
    assert parsed[0].name == "LM358"
    assert parsed[0].footprint.startswith("Package_SO:")
    assert parsed[0].pin_count == 2
    assert parsed[0].simulation_device == "X"

    index_path = tmp_path / "index.json"
    status = build_index(index_path, [root])
    assert status["symbol_count"] == 1
    assert status["footprint_count"] == 1
    assert status["model_count"] == 1
    result = search_index(index_path, "dual opamp", kind="symbol")
    assert result[0]["name"] == "LM358"


def test_missing_library_produces_empty_but_valid_index(tmp_path: Path):
    index_path = tmp_path / "index.json"
    status = build_index(index_path, [tmp_path / "missing"])
    assert status["symbol_count"] == 0
    assert search_index(index_path, "anything") == []


def test_parses_subcircuits_and_models_with_pin_order(tmp_path: Path):
    spice = tmp_path / "models.lib"
    spice.write_text(".SUBCKT TL_TEST IN OUT VCC VEE\nR1 IN OUT 1k\n.ENDS TL_TEST\n.MODEL D_TEST D(Is=1n)", encoding="utf-8")
    models = parse_spice_file(spice)
    assert models[0].name == "TL_TEST"
    assert models[0].kind == "subcircuit"
    assert models[0].pins == ("IN", "OUT", "VCC", "VEE")
    assert models[1].model_type == "D"


def test_model_binding_checks_inferred_passives_and_pin_mismatch():
    passive = LibraryPart("symbol", "R", "Device", "R.kicad_sym", reference="R", pin_count=2)
    assert validate_model_binding(passive, []) ["status"] == "inferred_ideal"
    active = LibraryPart("symbol", "U", "Test", "U.kicad_sym", reference="U", pin_count=5, simulation_library="models.lib", simulation_name="U_TEST")
    model = SpiceModel("U_TEST", "subcircuit", "models.lib", ("1", "2", "3"))
    assert validate_model_binding(active, [model])["status"] == "pin_mismatch"
