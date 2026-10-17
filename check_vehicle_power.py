"""Read vehicle sources; check selected passive circuits with Lattice tools.

Writes one report in this workspace, never edits the inspected design or opens EDA.
These extracted ideal subcircuits do not establish whole-board feasibility.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys

import numpy as np

from circuit_ai.constraints import ConstraintEvaluator, ConstraintOperator, ConstraintSpec
from circuit_ai.graph import linear_circuit_to_graph
from circuit_ai.lceda import verify_epro2_library_references
from circuit_ai.metrics import MetricSet, MetricSpec, MetricStatus, MetricValue
from circuit_ai.mna import CurrentSource, LinearCircuit, LinearElement, VoltageSource
from circuit_ai.simulation import (AnalysisSpec, Excitation, LinearMNASimulatorBackend,
    NgspiceSimulatorBackend, ObservableSpec, SimulationRequest, SweepSpec)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(part, kind):
    """Only explicit leading R/C engineering values; never infer from MPN."""
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)([munpkR]?)(?:F|H|R)?(?=\s|$)", part['value'])
    if not match:
        raise ValueError(f"unparsed value: {part['ref']} {part['value']!r}")
    multiplier = {'': 1, 'R': 1, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'k': 1e3}
    if kind == 'C' and match[2] in ('', 'R', 'k'):
        raise ValueError('capacitor needs an explicit capacitance suffix')
    return float(match[1]) * multiplier[match[2]]


def maximum_check(name, measured, limit, unit, source):
    spec = MetricSpec(name, 'vehicle_adapter', source, name, unit)
    metric = MetricValue(name, MetricStatus.AVAILABLE, measured, unit, spec.provider_id, source)
    return ConstraintEvaluator().evaluate(
        [ConstraintSpec(name, spec, ConstraintOperator.MAXIMUM, maximum=limit)],
        MetricSet({name: metric})).as_dict()


def check_ac(name, parts, kinds, ground, input_net, output_net, output_ref,
             expected, *, current=False, scales=None):
    normalize = lambda net: '0' if net == ground else net
    elements = tuple(LinearElement(p['ref'], kind, normalize(p['pins']['1']),
        normalize(p['pins']['2']), value(p, kind)*(scales or {}).get(kind, 1))
        for p, kind in zip(parts, kinds))
    source_name = 'Iprobe' if current else 'Vin'
    circuit = LinearCircuit(elements, () if current else
        (VoltageSource(source_name, input_net, '0', 1),),
        (CurrentSource(source_name, '0', input_net, 1),) if current else ())
    graph = linear_circuit_to_graph(circuit, name=name, input_node=input_net,
        output_node=output_net, source_name=source_name)
    # The BMS filter output is differential: SRP_F minus SRN_F, not SRP_F to ground.
    ports = tuple(replace(port, terminals=tuple(replace(t, net_id=output_ref)
        if t.role == 'reference' else t for t in port.terminals))
        if port.port_id == 'output' else port for port in graph.ports)
    graph = replace(graph, ports=ports)
    graph.require_valid()
    request = SimulationRequest(name, AnalysisSpec('small_signal_ac',
        (SweepSpec('frequency', 'Hz', start=.001, stop=1e6, points=181, scale='log'),),
        options={'allow_native_sweep_approximation': True}),
        graph.graph_id, excitations=(Excitation('test_source',
        'ac_current' if current else 'ac_voltage', 'component', source_name, {'magnitude': 1}),),
        requested_observables=(ObservableSpec('response',
        'transimpedance' if current else 'voltage_gain', 'ohm' if current else '1',
        'port', 'output', reference_id=None if current else 'input', representation='complex'),))
    results, checks = {}, {}
    for backend in (LinearMNASimulatorBackend(), NgspiceSimulatorBackend(timeout_s=20)):
        result = backend.simulate(backend.compile(graph), request)
        results[backend.capabilities.backend_id] = result.as_dict()
        if result.status.value != 'passed':
            continue
        wave = result.waveforms['response']
        frequencies = np.array(wave.axes[0].values)
        actual = np.array(wave.values, dtype=complex)
        reference = expected(frequencies, elements)
        relative = float(np.max(np.abs(actual-reference)/np.maximum(np.abs(reference), 1e-20)))
        checks[backend.capabilities.backend_id] = maximum_check(
            'relative_response_error', relative, 1e-5, '1', 'closed_form_of_extracted_ideal_network')
    return {'graph': graph.as_dict(), 'request': request.as_dict(), 'results': results,
            'comparison_constraints': checks, 'source_parts': parts, 'ground_boundary': ground}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('--out', type=Path, default=Path('outputs/vehicle_power_check/report.json'))
    args = parser.parse_args()
    root = args.target.resolve()
    sources = [root/'硬件/FOC驱动与储能/设计数据.py', root/'硬件/FOC驱动与储能/储能模块.py',
               root/'硬件/充电与BMS/设计数据.py', root/'硬件/充电与BMS/充电模块.py']
    before = {str(p): sha(p) for p in sources}
    drive = {p['ref']: p for p in runpy.run_path(str(sources[0]))['PARTS']}
    bms = {p['ref']: p for p in runpy.run_path(str(sources[2]))['PARTS']}
    runs = {}
    for tag, scale in [('nominal', {'R': 1, 'C': 1}),
                       ('minimum_tau', {'R': .99, 'C': .9}),
                       ('maximum_tau', {'R': 1.01, 'C': 1.1})]:
        parts = [bms[r] for r in ['R_SRP', 'R_SRN', 'C_SR']]
        expected = lambda f, e: 1/(1+2j*np.pi*f*(e[0].value+e[1].value)*e[2].value)
        result = check_ac('bms_filter_'+tag, parts, ['R', 'R', 'C'],
            'KELVIN_P', 'KELVIN_B', 'SRP_F', 'SRN_F', expected, scales=scale)
        tau = sum(value(p, 'R') for p in parts[:2])*scale['R']*value(parts[2], 'C')*scale['C']
        result['derived'] = {'tau_s': tau, 'cutoff_Hz': 1/(2*math.pi*tau)}
        runs['bms_filter_'+tag] = result
    bus_parts = [drive['R_BLEED']] + [drive['C_BUS'+str(i)] for i in range(1, 9)]
    impedance = lambda f, e: 1/(1/e[0].value+2j*np.pi*f*sum(x.value for x in e[1:]))
    runs['bus_bleed'] = check_ac('bus_bleed', bus_parts, ['R']+['C']*8,
        'GND', 'VBUS', 'VBUS', '0', impedance, current=True)
    capacitance = sum(value(p, 'C') for p in bus_parts[1:])
    bleed = value(bus_parts[0], 'R')
    runs['bus_bleed']['derived'] = {'capacitance_F': capacitance, 'tau_s': bleed*capacitance,
        '42V_to_5V_s_no_injection_nominal': bleed*capacitance*math.log(42/5),
        '48V_to_5V_s_Rplus5pct_Cplus20pct': bleed*1.05*capacitance*1.2*math.log(48/5),
        'condition': 'Isolation of all power sources, no backfeed; tolerance case is an assumption, not full temperature bound.'}
    pre_parts = [drive['R_PRE'], drive['R_BAT']] + bus_parts
    def pre_response(f, e):
        series = e[0].value+e[1].value
        z = 1/(1/e[2].value+2j*np.pi*f*sum(x.value for x in e[3:]))
        return z/(series+z)
    runs['precharge'] = check_ac('precharge', pre_parts, ['R']*3+['C']*8,
        'GND', drive['R_PRE']['pins']['1'], 'VBUS', '0', pre_response)
    series = value(drive['R_PRE'], 'R')+value(drive['R_BAT'], 'R')
    runs['precharge']['derived'] = {'initial_current_A_at42V': 42/series,
        'capacitor_energy_J_at42V': .5*capacitance*42**2,
        'small_signal_tau_s': series*bleed/(series+bleed)*capacitance,
        'condition': 'Ideal precharge contacts/fuse closed, K2 open, no motor/converter load; no switch timing or resistor pulse qualification.'}
    # Correct per-side allocation, not the whole vehicle auxiliary current.
    fuse_limit = min(float(re.match(r'([\d.]+)A', drive[r]['value'])[1])
                     for r in ['F_AUXBAT', 'F_AUXBUS'])
    modules = [drive[r] for r in ['PSU15_EXT', 'PSU12_MAIN_EXT', 'PSU5_EXT']]
    for p, ref in zip(modules, ['F_MOD1', 'F_MAIN1', 'F_MOD2']):
        assert drive[ref]['pins']['1'] == 'AUX_IN'
        assert drive[ref]['pins']['2'] == p['pins']['2']  # actual module +VIN pin
    module_W = sum(float(re.search(r'\s([\d.]+)W(?:\s|$)', p['value'])[1]) for p in modules)
    allocation = maximum_check('per_side_full_module_allocation_current', module_W/.7/23,
        fuse_limit, 'A', 'selected_three_module_ratings_at_assumed_23V_eta70pct')
    archives = {}
    # Inspect only the four delivered archives; never select historical cache copies.
    for relative in ['硬件/FOC驱动与储能/导出/电机驱动原理图.epro2',
                     '硬件/充电与BMS/BMS板/BMS板原理图候选.epro2',
                     '硬件/充电与BMS/外充板/外充板原理图候选.epro2',
                     '硬件/配电与水泵/导出/系统辅助板候选.epro2']:
        path = root / relative
        digest = sha(path)
        try:
            outcome = {'status': 'passed', 'counts': verify_epro2_library_references(path.read_bytes())}
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            outcome = {'status': 'not_verified', 'reason': str(exc)}
        outcome.update(sha256=digest, unchanged_during_check=sha(path)==digest)
        archives[str(path)] = outcome
    report = {'scope': 'EXTRACTED_PASSIVE_SUBCIRCUITS_ONLY', 'whole_vehicle_released': False,
        'source_sha256': before, 'checker_sha256': sha(Path(__file__)),
        'source_unchanged_during_check': all(sha(Path(p))==h for p,h in before.items()),
        'tools': ['CircuitGraph.require_valid', 'LinearMNASimulatorBackend', 'NgspiceSimulatorBackend',
                  'ConstraintEvaluator', 'verify_epro2_library_references'],
        'subcircuits': runs, 'auxiliary_full_allocation_constraint': allocation,
        'auxiliary_assumptions': {'module_output_W_each_side': module_W, 'input_V': 23, 'efficiency': .7,
            'actual_loads_verified': False, 'diode_drop_included': False, 'other_loads_included': False},
        'native_archive_library_closure': archives,
        'unsupported': ['FOC PWM/switching and motor', 'vendor IC models and logic/state machine',
            'BMS fault timing and protection coordination', 'thermal/SOA/EMI and PCB',
            'native full pin-net verification without an official netlist'],
        'limitations': ['AC and derived RC step estimates are not measured transients.',
            'No vendor loading, parasitics, temperature drift or fault injection in extracted passive models.',
            'Archive reference closure does not establish complete footprints, correct pins or ERC.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(args.out.resolve()),
        'source_unchanged': report['source_unchanged_during_check'],
        'subcircuits': {k:{b:r['status'] for b,r in v['results'].items()} for k,v in runs.items()},
        'auxiliary_constraint': allocation['feasibility'], 'archives': archives}, ensure_ascii=False))


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    sys.stdout.reconfigure(encoding='utf-8')
    main()
