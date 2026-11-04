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

"Container to encapsulate all control flow operations."

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.exceptions import CircuitError

if typing.TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.classical import expr


class ControlFlowOp(Instruction, ABC):
    """Abstract class to encapsulate all control flow operations.

    All subclasses of :class:`ControlFlowOp` have an internal attribute,
    :attr:`~ControlFlowOp.blocks`, which exposes the inner subcircuits used in the different blocks
    of the control flow.
    """

    @property
    @abstractmethod
    def blocks(self) -> tuple[QuantumCircuit, ...]:
        """Tuple of :class:`.QuantumCircuit`\\ s which may be executed as part of the
        execution of this :class:`ControlFlowOp`."""

    @abstractmethod
    def replace_blocks(self, blocks: typing.Iterable[QuantumCircuit]) -> ControlFlowOp:
        """Return a new version of this control-flow operations with the :attr:`blocks` mapped to
        the given new ones.

        Typically this is used in a workflow such as::

            existing_op = ...

            def map_block(block: QuantumCircuit) -> QuantumCircuit:
                new_block = block.copy_empty_like()
                # ... do something to `new_block` ...
                return new_block

            new_op = existing_op.replace_blocks(
                map_block(block) for block in existing_op.blocks
            )

        It is the caller's responsibility to ensure that the mapped blocks are defined over a
        unified set of circuit resources, much like constructing a :class:`ControlFlowOp` using its
        default constructor.

        Args:
            blocks: the new subcircuit blocks to use.

        Returns:
            New :class:`ControlFlowOp` with replaced blocks.
        """

    def iter_captured_vars(self) -> typing.Iterable[expr.Var]:
        """Get an iterator over the unique captured variables in all blocks of this construct."""
        seen = set()
        for block in self.blocks:
            for var in block.iter_captured_vars():
                if var not in seen:
                    seen.add(var)
                    yield var

    def iter_captured_stretches(self) -> typing.Iterable[expr.Stretch]:
        """Get an iterator over the unique captured stretch variables in all blocks of this
        construct."""
        seen = set()
        for block in self.blocks:
            for stretch in block.iter_captured_stretches():
                if stretch not in seen:
                    seen.add(stretch)
                    yield stretch

    @staticmethod
    def _unexpected_input_var_error() -> Exception:
        """Create a suitable exception to raise when a block has unexpected input variables."""
        return CircuitError("Only for-loop blocks can contain input variables")
