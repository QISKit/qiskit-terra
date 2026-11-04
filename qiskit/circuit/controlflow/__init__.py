# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Instruction sub-classes for dynamic circuits."""


from ._builder_utils import condition_resources, node_resources, LegacyResources
from .control_flow import ControlFlowOp
from .continue_loop import ContinueLoopOp
from .break_loop import BreakLoopOp

from .box import BoxOp
from .if_else import IfElseOp
from .while_loop import WhileLoopOp
from .for_loop import ForLoopOp
from .switch_case import SwitchCaseOp, CASE_DEFAULT

__all__ = [
    "CASE_DEFAULT",
    "CONTROL_FLOW_OP_NAMES",
    "BoxOp",
    "BreakLoopOp",
    "ContinueLoopOp",
    "ControlFlowOp",
    "ForLoopOp",
    "IfElseOp",
    "LegacyResources",
    "SwitchCaseOp",
    "WhileLoopOp",
    "condition_resources",
    "get_control_flow_name_mapping",
    "node_resources",
]

CONTROL_FLOW_OP_NAMES = frozenset(("for_loop", "while_loop", "if_else", "switch_case", "box"))
"""Set of the instruction names of Qiskit's known control-flow operations."""


def get_control_flow_name_mapping():
    """Return a dictionary mapping the names of control-flow operations
    to their corresponding classes."

    Examples:

        .. code-block:: python

            from qiskit.circuit import get_control_flow_name_mapping

            ctrl_flow_name_map = get_control_flow_name_mapping()
            if_else_object = ctrl_flow_name_map["if_else"]

            print(if_else_object)

        .. code-block:: text

            <class 'qiskit.circuit.controlflow.if_else.IfElseOp'>
    """

    name_mapping = {
        "if_else": IfElseOp,
        "while_loop": WhileLoopOp,
        "for_loop": ForLoopOp,
        "switch_case": SwitchCaseOp,
        "box": BoxOp,
    }
    return name_mapping
