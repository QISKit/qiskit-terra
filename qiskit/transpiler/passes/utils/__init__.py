# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility passes and functions used for other main passes."""

from .check_map import CheckMap
from .check_gate_direction import CheckGateDirection
from .gate_direction import GateDirection
from .barrier_before_final_measurements import BarrierBeforeFinalMeasurements
from .remove_final_measurements import RemoveFinalMeasurements
from .merge_adjacent_barriers import MergeAdjacentBarriers
from .dag_fixed_point import DAGFixedPoint
from .fixed_point import FixedPoint
from .error import Error
from .remove_barriers import RemoveBarriers
from .contains_instruction import ContainsInstruction
from .gates_basis import GatesInBasis
from .unroll_forloops import UnrollForLoops
from .minimum_point import MinimumPoint
from .filter_op_nodes import FilterOpNodes
from .wrap_angles import WrapAngles

# Utility functions
from . import control_flow

__all__ = [
    "BarrierBeforeFinalMeasurements",
    "CheckGateDirection",
    "CheckMap",
    "ContainsInstruction",
    "DAGFixedPoint",
    "Error",
    "FilterOpNodes",
    "FixedPoint",
    "GateDirection",
    "GatesInBasis",
    "MergeAdjacentBarriers",
    "MinimumPoint",
    "RemoveBarriers",
    "RemoveFinalMeasurements",
    "UnrollForLoops",
    "WrapAngles",
    "control_flow",
]
