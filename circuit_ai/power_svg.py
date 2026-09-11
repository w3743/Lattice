"""Browser-visible schematics for ideal power-stage expert results."""

from __future__ import annotations

from html import escape


def render_power_svg(result) -> str:
    if result.candidate.family == "isolated_flyback":
        return _render_flyback(result)
    if result.candidate.family == "dc_buck":
        return _render_buck(result)
    if result.candidate.family == "dc_sepic":
        return _render_sepic(result)
    return _render_boost(result)


def _header(title: str, subtitle: str) -> list[str]:
    return [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 520" role="img">',
        "<style>text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#172033}.title{font-size:25px;font-weight:700}.sub{font-size:14px;fill:#667085}.wire{stroke:#263238;stroke-width:3;fill:none}.part{stroke:#176b61;stroke-width:3;fill:white}.node{fill:#263238}.label{font-size:14px;font-weight:700}.value{font-size:12px;fill:#52606d}.ground{stroke:#263238;stroke-width:2}.isolation{stroke:#d18b28;stroke-width:2;stroke-dasharray:8 7}</style>",
        '<rect width="1000" height="520" fill="#fff"/>',
        f'<text class="title" x="40" y="42">{escape(title)}</text>',
        f'<text class="sub" x="40" y="68">{escape(subtitle)}</text>',
    ]


def _render_boost(result) -> str:
    p = result.parameters
    op = result.operating_point
    lines = _header(
        "Ideal non-isolated boost converter",
        f"Vin={op.input_voltage_v:.4g} V, Vout={op.output_voltage_v:.4g} V, Iout={op.output_current_a:.4g} A",
    )
    lines.extend([
        '<line class="wire" x1="100" y1="180" x2="185" y2="180"/>',
        '<circle class="part" cx="100" cy="260" r="34"/>',
        '<text class="label" x="82" y="255">Vin</text><text class="value" x="74" y="278">DC</text>',
        '<line class="wire" x1="100" y1="226" x2="100" y2="180"/><line class="wire" x1="100" y1="294" x2="100" y2="390"/>',
        '<path class="part" d="M185 180 c12 -26 24 -26 36 0 c12 -26 24 -26 36 0 c12 -26 24 -26 36 0"/>',
        f'<text class="label" x="215" y="145">L1</text><text class="value" x="190" y="214">{p.inductance_h:.4g} H</text>',
        '<line class="wire" x1="293" y1="180" x2="385" y2="180"/>',
        '<line class="part" x1="385" y1="160" x2="430" y2="180"/><line class="part" x1="430" y1="150" x2="430" y2="210"/>',
        '<line class="wire" x1="430" y1="180" x2="560" y2="180"/>',
        '<text class="label" x="397" y="135">D1</text>',
        '<line class="wire" x1="340" y1="180" x2="340" y2="255"/><line class="part" x1="315" y1="280" x2="340" y2="255"/><line class="part" x1="365" y1="280" x2="340" y2="280"/><line class="wire" x1="340" y1="280" x2="340" y2="390"/>',
        f'<text class="label" x="290" y="316">Q1 PWM</text><text class="value" x="282" y="336">D={p.duty_cycle:.4f}, f={p.switching_frequency_hz:.4g} Hz</text>',
        '<line class="wire" x1="560" y1="180" x2="840" y2="180"/>',
        '<line class="wire" x1="620" y1="180" x2="620" y2="245"/><line class="part" x1="590" y1="245" x2="650" y2="245"/><line class="part" x1="590" y1="267" x2="650" y2="267"/><line class="wire" x1="620" y1="267" x2="620" y2="390"/>',
        f'<text class="label" x="660" y="245">C1</text><text class="value" x="660" y="266">{p.capacitance_f:.4g} F</text>',
        '<line class="wire" x1="790" y1="180" x2="790" y2="235"/><polyline class="part" points="790,235 770,250 810,270 770,290 810,310 790,325"/><line class="wire" x1="790" y1="325" x2="790" y2="390"/>',
        f'<text class="label" x="825" y="270">Load</text><text class="value" x="825" y="290">{p.load_ohm:.4g} ohm</text>',
        '<line class="wire" x1="100" y1="390" x2="840" y2="390"/>',
        '<circle class="node" cx="340" cy="180" r="5"/><circle class="node" cx="620" cy="180" r="5"/>',
        '<text class="label" x="70" y="155">INPUT</text><text class="label" x="820" y="155">OUTPUT</text>',
        '</svg>',
    ])
    return "\n".join(lines)


def _render_flyback(result) -> str:
    p = result.parameters
    op = result.operating_point
    lines = _header(
        "Ideal isolated flyback converter",
        f"Vin={op.input_voltage_v:.4g} V, Vout={op.output_voltage_v:.4g} V, Iout={op.output_current_a:.4g} A, N={p.turns_ratio:.4g}",
    )
    lines.extend([
        '<line class="isolation" x1="490" y1="105" x2="490" y2="435"/>',
        '<text class="value" x="458" y="455">isolation barrier</text>',
        '<circle class="part" cx="105" cy="260" r="34"/><text class="label" x="86" y="255">Vin</text><text class="value" x="78" y="278">DC</text>',
        '<line class="wire" x1="105" y1="226" x2="105" y2="170"/><line class="wire" x1="105" y1="170" x2="300" y2="170"/>',
        '<line class="wire" x1="105" y1="294" x2="105" y2="390"/><line class="wire" x1="105" y1="390" x2="365" y2="390"/>',
        '<path class="part" d="M300 170 c-24 15 -24 35 0 50 c-24 15 -24 35 0 50 c-24 15 -24 35 0 50"/>',
        '<path class="part" d="M420 170 c24 15 24 35 0 50 c24 15 24 35 0 50 c24 15 24 35 0 50"/>',
        '<line class="part" x1="350" y1="155" x2="350" y2="335"/><line class="part" x1="370" y1="155" x2="370" y2="335"/>',
        f'<text class="label" x="333" y="128">T1</text><text class="value" x="315" y="350">Ns/Np={p.turns_ratio:.4g}</text>',
        '<line class="wire" x1="300" y1="320" x2="365" y2="320"/><line class="wire" x1="365" y1="320" x2="365" y2="390"/>',
        '<line class="part" x1="340" y1="340" x2="365" y2="320"/><line class="part" x1="390" y1="340" x2="365" y2="340"/>',
        f'<text class="label" x="300" y="375">Q1 PWM</text><text class="value" x="260" y="410">D={p.duty_cycle:.4f}, f={p.switching_frequency_hz:.4g} Hz</text>',
        '<line class="wire" x1="420" y1="170" x2="555" y2="170"/>',
        '<line class="part" x1="555" y1="150" x2="600" y2="170"/><line class="part" x1="600" y1="140" x2="600" y2="200"/><line class="wire" x1="600" y1="170" x2="880" y2="170"/>',
        '<text class="label" x="565" y="125">D1</text>',
        '<line class="wire" x1="420" y1="320" x2="420" y2="390"/><line class="wire" x1="420" y1="390" x2="880" y2="390"/>',
        '<line class="wire" x1="680" y1="170" x2="680" y2="245"/><line class="part" x1="650" y1="245" x2="710" y2="245"/><line class="part" x1="650" y1="267" x2="710" y2="267"/><line class="wire" x1="680" y1="267" x2="680" y2="390"/>',
        f'<text class="label" x="720" y="245">C1</text><text class="value" x="720" y="266">{p.capacitance_f:.4g} F</text>',
        '<line class="wire" x1="830" y1="170" x2="830" y2="230"/><polyline class="part" points="830,230 810,248 850,268 810,288 850,308 830,326"/><line class="wire" x1="830" y1="326" x2="830" y2="390"/>',
        f'<text class="label" x="860" y="265">Load</text><text class="value" x="860" y="286">{p.load_ohm:.4g} ohm</text>',
        '<text class="label" x="70" y="145">INPUT</text><text class="label" x="815" y="145">ISOLATED OUTPUT</text>',
        '</svg>',
    ])
    return "\n".join(lines)


def _render_buck(result) -> str:
    p = result.parameters
    op = result.operating_point
    lines = _header(
        "Grammar-generated ideal buck converter",
        f"Vin={op.input_voltage_v:.4g} V, Vout={op.output_voltage_v:.4g} V, Iout={op.output_current_a:.4g} A",
    )
    lines.extend([
        '<circle class="part" cx="100" cy="260" r="34"/><text class="label" x="82" y="255">Vin</text>',
        '<line class="wire" x1="100" y1="226" x2="100" y2="180"/><line class="wire" x1="100" y1="180" x2="210" y2="180"/>',
        '<rect class="part" x="210" y="155" width="90" height="50"/><text class="label" x="236" y="185">Q1</text>',
        '<line class="wire" x1="300" y1="180" x2="390" y2="180"/>',
        '<path class="part" d="M390 180 c12 -26 24 -26 36 0 c12 -26 24 -26 36 0 c12 -26 24 -26 36 0"/>',
        f'<text class="label" x="420" y="145">L1</text><text class="value" x="395" y="215">{p.inductance_h:.4g} H</text>',
        '<line class="wire" x1="498" y1="180" x2="850" y2="180"/>',
        '<line class="wire" x1="345" y1="180" x2="345" y2="250"/><rect class="part" x="320" y="250" width="50" height="55"/><text class="label" x="334" y="283">D1</text><line class="wire" x1="345" y1="305" x2="345" y2="390"/>',
        '<line class="wire" x1="620" y1="180" x2="620" y2="245"/><line class="part" x1="590" y1="245" x2="650" y2="245"/><line class="part" x1="590" y1="267" x2="650" y2="267"/><line class="wire" x1="620" y1="267" x2="620" y2="390"/>',
        f'<text class="label" x="660" y="245">C1</text><text class="value" x="660" y="266">{p.capacitance_f:.4g} F</text>',
        '<line class="wire" x1="790" y1="180" x2="790" y2="235"/><polyline class="part" points="790,235 770,250 810,270 770,290 810,310 790,325"/><line class="wire" x1="790" y1="325" x2="790" y2="390"/>',
        '<line class="wire" x1="100" y1="294" x2="100" y2="390"/><line class="wire" x1="100" y1="390" x2="850" y2="390"/>',
        f'<text class="value" x="205" y="225">D={p.duty_cycle:.4f}, f={p.switching_frequency_hz:.4g} Hz</text>',
        '<circle class="node" cx="345" cy="180" r="5"/><circle class="node" cx="620" cy="180" r="5"/>',
        '<text class="label" x="70" y="145">INPUT</text><text class="label" x="815" y="145">OUTPUT</text>',
        '</svg>',
    ])
    return "\n".join(lines)


def _render_sepic(result) -> str:
    p = result.parameters
    op = result.operating_point
    lines = _header(
        "Grammar-generated ideal SEPIC converter",
        f"Vin={op.input_voltage_v:.4g} V, Vout={op.output_voltage_v:.4g} V, Iout={op.output_current_a:.4g} A",
    )
    lines.extend([
        '<circle class="part" cx="75" cy="260" r="30"/><text class="label" x="58" y="255">Vin</text>',
        '<line class="wire" x1="75" y1="230" x2="75" y2="175"/><line class="wire" x1="75" y1="175" x2="145" y2="175"/>',
        '<rect class="part" x="145" y="150" width="85" height="50"/><text class="label" x="177" y="180">L1</text>',
        '<line class="wire" x1="230" y1="175" x2="300" y2="175"/><rect class="part" x="300" y="150" width="85" height="50"/><text class="label" x="332" y="180">C1</text>',
        '<line class="wire" x1="385" y1="175" x2="455" y2="175"/><rect class="part" x="455" y="150" width="85" height="50"/><text class="label" x="487" y="180">D1</text>',
        '<line class="wire" x1="540" y1="175" x2="900" y2="175"/>',
        '<line class="wire" x1="265" y1="175" x2="265" y2="245"/><rect class="part" x="225" y="245" width="80" height="55"/><text class="label" x="252" y="278">Q1</text><line class="wire" x1="265" y1="300" x2="265" y2="395"/>',
        '<line class="wire" x1="420" y1="175" x2="420" y2="245"/><rect class="part" x="380" y="245" width="80" height="55"/><text class="label" x="408" y="278">L2</text><line class="wire" x1="420" y1="300" x2="420" y2="395"/>',
        '<line class="wire" x1="650" y1="175" x2="650" y2="245"/><line class="part" x1="620" y1="245" x2="680" y2="245"/><line class="part" x1="620" y1="267" x2="680" y2="267"/><line class="wire" x1="650" y1="267" x2="650" y2="395"/><text class="label" x="690" y="260">C2</text>',
        '<line class="wire" x1="810" y1="175" x2="810" y2="235"/><polyline class="part" points="810,235 790,250 830,270 790,290 830,310 810,325"/><line class="wire" x1="810" y1="325" x2="810" y2="395"/><text class="label" x="845" y="275">Load</text>',
        '<line class="wire" x1="75" y1="290" x2="75" y2="395"/><line class="wire" x1="75" y1="395" x2="900" y2="395"/>',
        f'<text class="value" x="210" y="335">D={p.duty_cycle:.4f}, f={p.switching_frequency_hz:.4g} Hz</text>',
        f'<text class="value" x="145" y="222">L={p.inductance_h:.4g} H, C={p.capacitance_f:.4g} F</text>',
        '<circle class="node" cx="265" cy="175" r="5"/><circle class="node" cx="420" cy="175" r="5"/><circle class="node" cx="650" cy="175" r="5"/>',
        '<text class="label" x="45" y="140">INPUT</text><text class="label" x="840" y="140">OUTPUT</text>',
        '</svg>',
    ])
    return "\n".join(lines)
