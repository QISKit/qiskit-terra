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
Stylesheet for timeline drawer.

# TODO merge this docstring with pulse drawer.

The stylesheet `QiskitTimelineStyle` is initialized with the hard-corded default values in
`default_style`.

The `QiskitTimelineStyle` is a wrapper class of python dictionary with
the nested keys written such as `<type>.<group>.<item>` to represent a specific item
from many configuration options. This key representation is imitative of
`rcParams` of `matplotlib`.  However, the `QiskitTimelineStyle` does not need to be compatible
with the `rcParams` because the timeline stylesheet is heavily specialized to the context of
the scheduled circuit visualization.

Type of stylesheet is broadly separated into `formatter`, `generator` and `layout`.
The formatter is a nested dictionary of drawing parameters to control the appearance of
each visualization element. This data structure is similar to the `rcParams` of `matplotlib`.

The generator is a list of callback functions that generates drawings from
a given data source and the formatter. Each item can take multiple functions so that
several drawing data, for example, box, text, etc..., are generated from the single data source.
The layout is a callback function that determines the appearance of the output image.
Because a single stylesheet doesn't generate multiple images with different appearance,
only one layout function can be chosen for each stylesheet.
"""

from typing import Any
from collections.abc import Mapping

from qiskit.visualization.timeline import generators, layouts


class QiskitTimelineStyle(dict):
    """Stylesheet for pulse drawer."""

    def __init__(self):
        super().__init__()
        # to inform which stylesheet is applied. some plotter may not support specific style.
        self.stylesheet = None
        self.update(default_style())

    def update(self, __m: Mapping[str, Any], **kwargs) -> None:
        super().update(__m, **kwargs)
        for key, value in __m.items():
            self.__setitem__(key, value)
        self.stylesheet = __m.__class__.__name__

    @property
    def formatter(self):
        """Return formatter field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "formatter":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict

    @property
    def generator(self):
        """Return generator field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "generator":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict

    @property
    def layout(self):
        """Return layout field of style dictionary."""
        sub_dict = {}
        for key, value in self.items():
            sub_keys = key.split(".")
            if sub_keys[0] == "layout":
                sub_dict[".".join(sub_keys[1:])] = value
        return sub_dict


class IQXStandard(dict):
    """Standard timeline stylesheet.

    - Show time buckets.
    - Show only operand name.
    - Show bit name.
    - Show barriers.
    - Show idle timeline.
    - Show gate link.
    - Remove classical bits.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.control.show_idle": True,
            "formatter.control.show_clbits": False,
            "formatter.control.show_barriers": True,
            "formatter.control.show_delays": False,
            "generator.gates": [generators.gen_sched_gate, generators.gen_short_gate_name],
            "generator.bits": [generators.gen_bit_name, generators.gen_timeslot],
            "generator.barriers": [generators.gen_barrier],
            "generator.gate_links": [generators.gen_gate_link],
            "layout.bit_arrange": layouts.qreg_creg_ascending,
            "layout.time_axis_map": layouts.time_map_in_dt,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Standard timeline style sheet."


class IQXSimple(dict):
    """Simple timeline stylesheet.

    - Show time buckets.
    - Show bit name.
    - Show gate link.
    - Remove idle timeline.
    - Remove classical bits.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.control.show_idle": False,
            "formatter.control.show_clbits": False,
            "formatter.control.show_barriers": False,
            "formatter.control.show_delays": False,
            "generator.gates": [generators.gen_sched_gate],
            "generator.bits": [generators.gen_bit_name, generators.gen_timeslot],
            "generator.barriers": [generators.gen_barrier],
            "generator.gate_links": [generators.gen_gate_link],
            "layout.bit_arrange": layouts.qreg_creg_ascending,
            "layout.time_axis_map": layouts.time_map_in_dt,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Simplified timeline style sheet."


class IQXDebugging(dict):
    """Timeline stylesheet for programmers. Show details of instructions.

    - Show time buckets.
    - Show operand name, qubits, and parameters.
    - Show barriers.
    - Show delays.
    - Show idle timeline.
    - Show bit name.
    - Show gate link.
    """

    def __init__(self, **kwargs):
        super().__init__()
        style = {
            "formatter.control.show_idle": True,
            "formatter.control.show_clbits": True,
            "formatter.control.show_barriers": True,
            "formatter.control.show_delays": True,
            "generator.gates": [generators.gen_sched_gate, generators.gen_full_gate_name],
            "generator.bits": [generators.gen_bit_name, generators.gen_timeslot],
            "generator.barriers": [generators.gen_barrier],
            "generator.gate_links": [generators.gen_gate_link],
            "layout.bit_arrange": layouts.qreg_creg_ascending,
            "layout.time_axis_map": layouts.time_map_in_dt,
        }
        style.update(**kwargs)
        self.update(style)

    def __repr__(self):
        return "Timeline style sheet for timeline programmers."


def default_style() -> dict[str, Any]:
    """Define default values of the timeline stylesheet."""
    return {
        "formatter.general.fig_width": 14,
        "formatter.general.fig_unit_height": 0.8,
        "formatter.general.dpi": 150,
        "formatter.margin.top": 0.5,
        "formatter.margin.bottom": 0.5,
        "formatter.margin.left_percent": 0.02,
        "formatter.margin.right_percent": 0.02,
        "formatter.margin.link_interval_percent": 0.01,
        "formatter.margin.minimum_duration": 50,
        "formatter.time_bucket.edge_dt": 10,
        "formatter.color.background": "#FFFFFF",
        "formatter.color.timeslot": "#DDDDDD",
        "formatter.color.gate_name": "#000000",
        "formatter.color.bit_name": "#000000",
        "formatter.color.barrier": "#222222",
        "formatter.color.gates": {
            "u0": "#FA74A6",
            "u1": "#000000",
            "u2": "#FA74A6",
            "u3": "#FA74A6",
            "id": "#05BAB6",
            "sx": "#FA74A6",
            "sxdg": "#FA74A6",
            "x": "#05BAB6",
            "y": "#05BAB6",
            "z": "#05BAB6",
            "h": "#6FA4FF",
            "cx": "#6FA4FF",
            "cy": "#6FA4FF",
            "cz": "#6FA4FF",
            "swap": "#6FA4FF",
            "s": "#6FA4FF",
            "sdg": "#6FA4FF",
            "dcx": "#6FA4FF",
            "iswap": "#6FA4FF",
            "t": "#BB8BFF",
            "tdg": "#BB8BFF",
            "r": "#BB8BFF",
            "rx": "#BB8BFF",
            "ry": "#BB8BFF",
            "rz": "#000000",
            "reset": "#808080",
            "measure": "#808080",
        },
        "formatter.color.default_gate": "#BB8BFF",
        "formatter.latex_symbol.gates": {
            "u0": r"{\rm U}_0",
            "u1": r"{\rm U}_1",
            "u2": r"{\rm U}_2",
            "u3": r"{\rm U}_3",
            "id": r"{\rm Id}",
            "x": r"{\rm X}",
            "y": r"{\rm Y}",
            "z": r"{\rm Z}",
            "h": r"{\rm H}",
            "cx": r"{\rm CX}",
            "cy": r"{\rm CY}",
            "cz": r"{\rm CZ}",
            "swap": r"{\rm SWAP}",
            "s": r"{\rm S}",
            "sdg": r"{\rm S}^\dagger",
            "sx": r"{\rm √X}",
            "sxdg": r"{\rm √X}^\dagger",
            "dcx": r"{\rm DCX}",
            "iswap": r"{\rm iSWAP}",
            "t": r"{\rm T}",
            "tdg": r"{\rm T}^\dagger",
            "r": r"{\rm R}",
            "rx": r"{\rm R}_x",
            "ry": r"{\rm R}_y",
            "rz": r"{\rm R}_z",
            "reset": r"|0\rangle",
            "measure": r"{\rm Measure}",
        },
        "formatter.latex_symbol.frame_change": r"\circlearrowleft",
        "formatter.unicode_symbol.frame_change": "\u21ba",
        "formatter.box_height.gate": 0.5,
        "formatter.box_height.timeslot": 0.6,
        "formatter.layer.gate": 3,
        "formatter.layer.timeslot": 0,
        "formatter.layer.gate_name": 5,
        "formatter.layer.bit_name": 5,
        "formatter.layer.frame_change": 4,
        "formatter.layer.barrier": 1,
        "formatter.layer.gate_link": 2,
        "formatter.alpha.gate": 1.0,
        "formatter.alpha.timeslot": 0.7,
        "formatter.alpha.barrier": 0.5,
        "formatter.alpha.gate_link": 0.8,
        "formatter.line_width.gate": 0,
        "formatter.line_width.timeslot": 0,
        "formatter.line_width.barrier": 3,
        "formatter.line_width.gate_link": 3,
        "formatter.line_style.barrier": "-",
        "formatter.line_style.gate_link": "-",
        "formatter.text_size.gate_name": 12,
        "formatter.text_size.bit_name": 15,
        "formatter.text_size.frame_change": 18,
        "formatter.text_size.axis_label": 13,
        "formatter.label_offset.frame_change": 0.25,
        "formatter.control.show_idle": True,
        "formatter.control.show_clbits": True,
        "formatter.control.show_barriers": True,
        "formatter.control.show_delays": True,
        "generator.gates": [],
        "generator.bits": [],
        "generator.barriers": [],
        "generator.gate_links": [],
        "layout.bit_arrange": None,
        "layout.time_axis_map": None,
    }
