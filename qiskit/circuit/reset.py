# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qubit reset to computational zero.
"""

from qiskit.circuit.singleton import SingletonInstruction, stdlib_singleton_key
from qiskit._accelerate.circuit import StandardInstructionType


class Reset(SingletonInstruction):
    r"""Incoherently reset a qubit to the :math:`\lvert0\rangle` state."""

    _standard_instruction_type = StandardInstructionType.Reset

    def __init__(self, label=None):
        """
        Args:
            label: optional string label of this instruction.
        """
        super().__init__("reset", 1, 0, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def broadcast_arguments(self, qargs, cargs):
        for qarg in qargs[0]:
            yield [qarg], []
