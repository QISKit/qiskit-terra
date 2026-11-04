# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate made of commuting two-qubit gates."""
from __future__ import annotations

from collections.abc import Iterable

from qiskit.exceptions import QiskitError
from qiskit.circuit import Clbit, Gate, Parameter, ParameterExpression, Qubit
from qiskit.dagcircuit import DAGOpNode


class Commuting2qBlock(Gate):
    """A gate made of commuting two-qubit gates.

    This gate is intended for use with commuting swap strategies to make it convenient
    for the swap strategy router to identify which blocks of operations commute.

    Note that :attr:`params` of this gate reports the free parameters the nodes contain.
    There is no guaranteed order they are stored in.
    """

    def __init__(self, node_block: Iterable[DAGOpNode]) -> None:
        """
        Args:
            node_block: A block of nodes that commute.

        Raises:
            QiskitError: If the nodes in the node block do not apply to two-qubits.
        """
        qubits: set[Qubit] = set()
        cbits: set[Clbit] = set()
        params_set: set[Parameter] = set()

        for node in node_block:
            if len(node.qargs) != 2:
                raise QiskitError(f"Node {node.name} does not apply to two-qubits.")

            qubits.update(node.qargs)
            cbits.update(node.cargs)

            for param in node.op.params:
                if isinstance(param, ParameterExpression):
                    params_set.update(param.parameters)
        params_list = list(params_set)

        if cbits:
            raise QiskitError(
                f"{self.__class__.__name__} does not accept nodes with classical bits."
            )

        super().__init__(
            "commuting_2q_block",
            num_qubits=len(qubits),
            params=params_list,
            label="Commuting 2q gates",
        )
        self.node_block = node_block
        self.qubits = qubits

    def __iter__(self):
        """Iterate through the nodes in the block."""
        return iter(self.node_block)
