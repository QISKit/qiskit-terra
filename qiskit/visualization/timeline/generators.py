# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
A collection of functions that generate drawings from formatted input data.
See :py:mod:`~qiskit.visualization.timeline.types` for more info on the required data.

An end-user can write arbitrary functions that generate custom drawings.
Generators in this module are called with the `formatter` kwarg. This data provides
the stylesheet configuration.


There are 4 types of generators in this module.

1. generator.gates

In this stylesheet entry the input data is `types.ScheduledGate` and generates gate objects
such as time buckets and gate name annotations.

The function signature of the generator is restricted to:

    ```python

    def my_object_generator(
            gate: types.ScheduledGate,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # your code here: create and return drawings related to the gate object.
    ```

If a generator object has the attribute ``accepts_program`` set to ``True``, then the generator will
be called with an additional keyword argument ``program: QuantumCircuit``.

2. generator.bits

In this stylesheet entry the input data is `types.Bits` and generates timeline objects
such as zero line and name of bit associated with the timeline.

The function signature of the generator is restricted to:

    ```python

    def my_object_generator(
            bit: types.Bits,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # your code here: create and return drawings related to the bit object.
    ```

If a generator object has the attribute ``accepts_program`` set to ``True``, then the generator will
be called with an additional keyword argument ``program: QuantumCircuit``.

3. generator.barriers

In this stylesheet entry the input data is `types.Barrier` and generates barrier objects
such as barrier lines.

The function signature of the generator is restricted to:

    ```python

    def my_object_generator(
            barrier: types.Barrier,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # your code here: create and return drawings related to the barrier object.
    ```

If a generator object has the attribute ``accepts_program`` set to ``True``, then the generator will
be called with an additional keyword argument ``program: QuantumCircuit``.

4. generator.gate_links

In this stylesheet entry the input data is `types.GateLink` and generates barrier objects
such as barrier lines.

The function signature of the generator is restricted to:

    ```python

    def my_object_generator(
            link: types.GateLink,
            formatter: Dict[str, Any]) -> List[ElementaryData]:

        # your code here: create and return drawings related to the link object.
    ```

If a generator object has the attribute ``accepts_program`` set to ``True``, then the generator will
be called with an additional keyword argument ``program: QuantumCircuit``.

Arbitrary generator function satisfying the above format can be accepted.
Returned `ElementaryData` can be arbitrary subclasses that are implemented in
the plotter API.
"""

import warnings
from typing import Any

from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.visualization.timeline import types, drawings


def gen_sched_gate(
    gate: types.ScheduledGate,
    formatter: dict[str, Any],
) -> list[drawings.TextData | drawings.BoxData]:
    """Generate time bucket or symbol of scheduled gate.

    If gate duration is zero or frame change a symbol is generated instead of time box.
    The face color of gates depends on the operand type.

    Stylesheet:
        - The `gate` style is applied for finite duration gate.
        - The `frame_change` style is applied for zero duration gate.
        - The `gate_face_color` style is applied for face color.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` or `BoxData` drawings.
    """
    try:
        label = gate.operand.label or "n/a"
    except AttributeError:
        label = "n/a"

    meta = {
        "name": gate.operand.name,
        "label": label,
        "bits": gate.bits,
        "t0": gate.t0,
        "duration": gate.duration,
        "parameters": ", ".join(map(str, gate.operand.params)),
    }

    # find color
    color = formatter["color.gates"].get(gate.operand.name, formatter["color.default_gate"])

    if gate.duration > 0:
        # gate with finite duration pulse
        styles = {
            "zorder": formatter["layer.gate"],
            "facecolor": color,
            "alpha": formatter["alpha.gate"],
            "linewidth": formatter["line_width.gate"],
        }

        # assign special name to delay for filtering
        if gate.operand.name == "delay":
            data_type = types.BoxType.DELAY
        else:
            data_type = types.BoxType.SCHED_GATE

        drawing = drawings.BoxData(
            data_type=data_type,
            xvals=[gate.t0, gate.t0 + gate.duration],
            yvals=[-0.5 * formatter["box_height.gate"], 0.5 * formatter["box_height.gate"]],
            bit=gate.bits[gate.bit_position],
            meta=meta,
            styles=styles,
        )
    else:
        # frame change
        styles = {
            "zorder": formatter["layer.frame_change"],
            "color": color,
            "size": formatter["text_size.frame_change"],
            "va": "center",
            "ha": "center",
        }
        unicode_symbol = formatter["unicode_symbol.frame_change"]
        latex_symbol = formatter["latex_symbol.frame_change"]

        drawing = drawings.TextData(
            data_type=types.SymbolType.FRAME,
            bit=gate.bits[gate.bit_position],
            xval=gate.t0,
            yval=0,
            text=unicode_symbol,
            latex=latex_symbol,
            styles=styles,
        )

    return [drawing]


def gen_full_gate_name(
    gate: types.ScheduledGate, formatter: dict[str, Any], program: QuantumCircuit | None = None
) -> list[drawings.TextData]:
    """Generate gate name.

    Parameters and associated bits are also shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.
        program: Optional program that the bits are a part of.

    Returns:
        List of `TextData` drawings.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = "center"
        v_pos = 0
    else:
        # frame change
        v_align = "bottom"
        v_pos = formatter["label_offset.frame_change"]

    styles = {
        "zorder": formatter["layer.gate_name"],
        "color": formatter["color.gate_name"],
        "size": formatter["text_size.gate_name"],
        "va": v_align,
        "ha": "center",
    }
    # find latex representation
    default_name = rf"{{\rm {gate.operand.name}}}"
    latex_name = formatter["latex_symbol.gates"].get(gate.operand.name, default_name)

    label_plain = f"{gate.operand.name}"
    label_latex = rf"{latex_name}"

    # bit index
    if len(gate.bits) > 1:
        if program is None:
            # This is horribly hacky and mostly meaningless, but there's no other distinguisher
            # available to us if all we have is a `Bit` instance.
            bits_str = ", ".join(str(id(bit))[-3:] for bit in gate.bits)
        else:
            bits_str = ", ".join(f"{program.find_bit(bit).index}" for bit in gate.bits)
        label_plain += f"[{bits_str}]"
        label_latex += f"[{bits_str}]"

    # parameter list
    params = []
    for val in gate.operand.params:
        try:
            params.append(f"{float(val):.2f}")
        except ValueError:
            params.append(f"{val}")
    params_str = ", ".join(params)

    if params_str and gate.operand.name != "delay":
        label_plain += f"({params_str})"
        label_latex += f"({params_str})"

    # duration
    if gate.duration > 0:
        label_plain += f"[{gate.duration}]"
        label_latex += f"[{gate.duration}]"

    # assign special name to delay for filtering
    if gate.operand.name == "delay":
        data_type = types.LabelType.DELAY
    else:
        data_type = types.LabelType.GATE_NAME

    drawing = drawings.TextData(
        data_type=data_type,
        xval=gate.t0 + 0.5 * gate.duration,
        yval=v_pos,
        bit=gate.bits[gate.bit_position],
        text=label_plain,
        latex=label_latex,
        styles=styles,
    )

    return [drawing]


gen_full_gate_name.accepts_program = True


def gen_short_gate_name(
    gate: types.ScheduledGate, formatter: dict[str, Any]
) -> list[drawings.TextData]:
    """Generate gate name.

    Only operand name is shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawings.
    """
    if gate.duration > 0:
        # gate with finite duration pulse
        v_align = "center"
        v_pos = 0
    else:
        # frame change
        v_align = "bottom"
        v_pos = formatter["label_offset.frame_change"]

    styles = {
        "zorder": formatter["layer.gate_name"],
        "color": formatter["color.gate_name"],
        "size": formatter["text_size.gate_name"],
        "va": v_align,
        "ha": "center",
    }
    # find latex representation
    default_name = rf"{{\rm {gate.operand.name}}}"
    latex_name = formatter["latex_symbol.gates"].get(gate.operand.name, default_name)

    label_plain = f"{gate.operand.name}"
    label_latex = f"{latex_name}"

    # assign special name for delay to filtering
    if gate.operand.name == "delay":
        data_type = types.LabelType.DELAY
    else:
        data_type = types.LabelType.GATE_NAME

    drawing = drawings.TextData(
        data_type=data_type,
        xval=gate.t0 + 0.5 * gate.duration,
        yval=v_pos,
        bit=gate.bits[gate.bit_position],
        text=label_plain,
        latex=label_latex,
        styles=styles,
    )

    return [drawing]


def gen_timeslot(bit: types.Bits, formatter: dict[str, Any]) -> list[drawings.BoxData]:
    """Generate time slot of associated bit.

    Stylesheet:
        - `timeslot` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `BoxData` drawings.
    """
    styles = {
        "zorder": formatter["layer.timeslot"],
        "alpha": formatter["alpha.timeslot"],
        "linewidth": formatter["line_width.timeslot"],
        "facecolor": formatter["color.timeslot"],
    }

    drawing = drawings.BoxData(
        data_type=types.BoxType.TIMELINE,
        xvals=[types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT],
        yvals=[-0.5 * formatter["box_height.timeslot"], 0.5 * formatter["box_height.timeslot"]],
        bit=bit,
        styles=styles,
    )

    return [drawing]


def gen_bit_name(
    bit: types.Bits,
    formatter: dict[str, Any],
    program: QuantumCircuit | None = None,
) -> list[drawings.TextData]:
    """Generate bit label.

    Stylesheet:
        - `bit_name` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.
        program: Optional program that the bits are a part of.

    Returns:
        List of `TextData` drawings.
    """
    styles = {
        "zorder": formatter["layer.bit_name"],
        "color": formatter["color.bit_name"],
        "size": formatter["text_size.bit_name"],
        "va": "center",
        "ha": "right",
    }

    if program is None:
        warnings.warn("bits cannot be accurately named without passing a 'program'", stacklevel=2)
        label_plain = "q" if isinstance(bit, Qubit) else "c"
        label_latex = rf"{{\rm {label_plain}}}"
    else:
        loc = program.find_bit(bit)
        if loc.registers:
            label_plain = loc.registers[-1][0].name
            label_latex = rf"{{\rm {loc.registers[-1][0].prefix}}}_{{{loc.registers[-1][1]}}}"
        else:
            label_plain = "q" if isinstance(bit, Qubit) else "c"
            label_latex = rf"{{\rm {label_plain}}}_{{{loc.index}}}"

    drawing = drawings.TextData(
        data_type=types.LabelType.BIT_NAME,
        xval=types.AbstractCoordinate.LEFT,
        yval=0,
        bit=bit,
        text=label_plain,
        latex=label_latex,
        styles=styles,
    )

    return [drawing]


gen_bit_name.accepts_program = True


def gen_barrier(barrier: types.Barrier, formatter: dict[str, Any]) -> list[drawings.LineData]:
    """Generate barrier line.

    Stylesheet:
        - `barrier` style is applied.

    Args:
        barrier: Barrier instruction.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `LineData` drawings.
    """
    styles = {
        "alpha": formatter["alpha.barrier"],
        "zorder": formatter["layer.barrier"],
        "linewidth": formatter["line_width.barrier"],
        "linestyle": formatter["line_style.barrier"],
        "color": formatter["color.barrier"],
    }

    drawing = drawings.LineData(
        data_type=types.LineType.BARRIER,
        xvals=[barrier.t0, barrier.t0],
        yvals=[-0.5, 0.5],
        bit=barrier.bits[barrier.bit_position],
        styles=styles,
    )

    return [drawing]


def gen_gate_link(link: types.GateLink, formatter: dict[str, Any]) -> list[drawings.GateLinkData]:
    """Generate gate link line.

    Line color depends on the operand type.

    Stylesheet:
        - `gate_link` style is applied.
        - The `gate_face_color` style is applied for line color.

    Args:
        link: Gate link object.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `GateLinkData` drawings.
    """

    # find line color
    color = formatter["color.gates"].get(link.opname, formatter["color.default_gate"])

    styles = {
        "alpha": formatter["alpha.gate_link"],
        "zorder": formatter["layer.gate_link"],
        "linewidth": formatter["line_width.gate_link"],
        "linestyle": formatter["line_style.gate_link"],
        "color": color,
    }

    drawing = drawings.GateLinkData(bits=link.bits, xval=link.t0, styles=styles)

    return [drawing]
