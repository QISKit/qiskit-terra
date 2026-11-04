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
Instruction collection.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import TYPE_CHECKING
from collections.abc import Callable

from qiskit.circuit.exceptions import CircuitError
from .operation import Operation
from .quantumcircuitdata import CircuitInstruction

if TYPE_CHECKING:
    from qiskit.circuit import Clbit, ClassicalRegister


class InstructionSet:
    """Instruction collection, and their contexts."""

    __slots__ = ("_instructions", "_requester")

    def __init__(
        self,
        *,
        resource_requester: Callable[..., ClassicalRegister | Clbit] | None = None,
    ):
        """New collection of instructions.

        The context (``qargs`` and ``cargs`` that each instruction is attached to) is also stored
        separately for each instruction.

        Args:
            resource_requester: A callable that takes in the classical resource used in the
                condition, verifies that it is present in the attached circuit, resolves any indices
                into concrete :obj:`.Clbit` instances, and returns the concrete resource.  If this
                is not given, specifying a condition with an index is forbidden, and all concrete
                :obj:`.Clbit` and :obj:`.ClassicalRegister` resources will be assumed to be valid.

                .. note::

                    The callback ``resource_requester`` assumes that a call implies that the
                    resource will now be used.  It may throw an error if the resource is not valid
                    for usage.

        """
        self._instructions: list[
            CircuitInstruction | (MutableSequence[CircuitInstruction], int)
        ] = []
        self._requester = resource_requester

    def __len__(self):
        """Return number of instructions in set"""
        return len(self._instructions)

    def __getitem__(self, i):
        """Return instruction at index"""
        inst = self._instructions[i]
        if isinstance(inst, CircuitInstruction):
            return inst
        data, idx = inst
        return data[idx]

    def add(self, instruction, qargs=None, cargs=None):
        """Add an instruction and its context (where it is attached)."""
        if not isinstance(instruction, CircuitInstruction):
            if not isinstance(instruction, Operation):
                raise CircuitError("attempt to add non-Operation to InstructionSet")
            if qargs is None or cargs is None:
                raise CircuitError("missing qargs or cargs in old-style InstructionSet.add")
            instruction = CircuitInstruction(instruction, tuple(qargs), tuple(cargs))
        self._instructions.append(instruction)

    def _add_ref(self, data: MutableSequence[CircuitInstruction], pos: int):
        """Add a reference to an instruction and its context within a mutable sequence.
        Updates to the instruction set will modify the specified sequence in place."""
        self._instructions.append((data, pos))

    def inverse(self, annotated: bool = False):
        """Invert all instructions.

        .. note::
            It is preferable to take the inverse *before* appending the gate(s) to the circuit.
        """
        for i, instruction in enumerate(self._instructions):
            if isinstance(instruction, CircuitInstruction):
                self._instructions[i] = instruction.replace(
                    operation=instruction.operation.inverse(annotated=annotated)
                )
            else:
                data, idx = instruction
                instruction = data[idx]
                data[idx] = instruction.replace(
                    operation=instruction.operation.inverse(annotated=annotated)
                )
        return self

    # Legacy support for properties.  Added in Terra 0.21 to support the internal switch in
    # `QuantumCircuit.data` from the 3-tuple to `CircuitInstruction`.

    def _instructions_iter(self):
        return (i if isinstance(i, CircuitInstruction) else i[0][i[1]] for i in self._instructions)

    @property
    def instructions(self):
        """Legacy getter for the instruction components of an instruction set.  This does not
        support mutation."""
        return [instruction.operation for instruction in self._instructions_iter()]

    @property
    def qargs(self):
        """Legacy getter for the qargs components of an instruction set.  This does not support
        mutation."""
        return [list(instruction.qubits) for instruction in self._instructions_iter()]

    @property
    def cargs(self):
        """Legacy getter for the cargs components of an instruction set.  This does not support
        mutation."""
        return [list(instruction.clbits) for instruction in self._instructions_iter()]
