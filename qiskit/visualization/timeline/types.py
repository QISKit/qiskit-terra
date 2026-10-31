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
Special data types.
"""

from enum import Enum, StrEnum
from typing import NamedTuple, NewType

from qiskit import circuit


class ScheduledGate(NamedTuple):
    t0: int
    operand: circuit.Gate
    duration: int
    bits: list[circuit.Qubit | circuit.Clbit]
    bit_position: int


ScheduledGate.__doc__ = "A gate instruction with embedded time."
ScheduledGate.t0.__doc__ = "Time when the instruction is issued."
ScheduledGate.operand.__doc__ = "Gate object associated with the gate."
ScheduledGate.duration.__doc__ = "Time duration of the instruction."
ScheduledGate.bits.__doc__ = "List of bit associated with the gate."
ScheduledGate.bit_position.__doc__ = "Position of bit associated with this drawing source."


class GateLink(NamedTuple):
    t0: int
    opname: str
    bits: list[circuit.Qubit | circuit.Clbit]


GateLink.__doc__ = "Dedicated object to represent a relationship between instructions."
GateLink.t0.__doc__ = "A position where the link is placed."
GateLink.opname.__doc__ = "Name of gate associated with this link."
GateLink.bits.__doc__ = "List of bit associated with the instruction."


class Barrier(NamedTuple):
    t0: int
    bits: list[circuit.Qubit | circuit.Clbit]
    bit_position: int


Barrier.__doc__ = "Dedicated object to represent a barrier instruction."
Barrier.t0.__doc__ = "A position where the barrier is placed."
Barrier.bits.__doc__ = "List of bit associated with the instruction."
Barrier.bit_position.__doc__ = "Position of bit associated with this drawing source."


class HorizontalAxis(NamedTuple):
    window: tuple[int, int]
    axis_map: dict[int, int]
    label: str


HorizontalAxis.__doc__ = "Data to represent configuration of horizontal axis."
HorizontalAxis.window.__doc__ = "Left and right edge of graph."
HorizontalAxis.axis_map.__doc__ = "Mapping of apparent coordinate system and actual location."
HorizontalAxis.label.__doc__ = "Label of horizontal axis."


class BoxType(StrEnum):
    """Box type.

    SCHED_GATE: Box that represents occupation time by gate.
    DELAY: Box associated with delay.
    TIMELINE: Box that represents time slot of a bit.
    """

    SCHED_GATE = "Box.ScheduledGate"
    DELAY = "Box.Delay"
    TIMELINE = "Box.Timeline"


class LineType(StrEnum):
    """Line type.

    BARRIER: Line that represents barrier instruction.
    GATE_LINK: Line that represents a link among gates.
    """

    BARRIER = "Line.Barrier"
    GATE_LINK = "Line.GateLink"


class SymbolType(StrEnum):
    """Symbol type.

    FRAME: Symbol that represents zero time frame change (Rz) instruction.
    """

    FRAME = "Symbol.Frame"


class LabelType(StrEnum):
    """Label type.

    GATE_NAME: Label that represents name of gate.
    DELAY: Label associated with delay.
    GATE_PARAM: Label that represents parameter of gate.
    BIT_NAME: Label that represents name of bit.
    """

    GATE_NAME = "Label.Gate.Name"
    DELAY = "Label.Delay"
    GATE_PARAM = "Label.Gate.Param"
    BIT_NAME = "Label.Bit.Name"


class AbstractCoordinate(Enum):
    """Abstract coordinate that the exact value depends on the user preference.

    RIGHT: The horizontal coordinate at t0 shifted by the left margin.
    LEFT: The horizontal coordinate at tf shifted by the right margin.
    TOP: The vertical coordinate at the top of the canvas.
    BOTTOM: The vertical coordinate at the bottom of the canvas.
    """

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"


class Plotter(StrEnum):
    """Name of timeline plotter APIs.

    MPL: Matplotlib plotter interface. Show timeline in 2D canvas.
    """

    MPL = "mpl"


# convenient type to represent union of drawing data
DataTypes = NewType("DataTypes", BoxType | LabelType | LineType | SymbolType)

# convenient type to represent union of values to represent a coordinate
Coordinate = NewType("Coordinate", float | AbstractCoordinate)

# Valid bit objects
Bits = NewType("Bits", circuit.Qubit | circuit.Clbit)
