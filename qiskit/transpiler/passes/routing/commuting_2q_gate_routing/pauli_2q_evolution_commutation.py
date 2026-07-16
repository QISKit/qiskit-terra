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

"""An analysis pass to find evolution gates in which the Paulis commute."""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass
from qiskit.quantum_info import SparsePauliOp, Pauli
from .commuting_2q_block import Commuting2qBlock


class FindCommutingPauliEvolutions(TransformationPass):
    """Finds :class:`.PauliEvolutionGate` objects where the operators, that are evolved, all commute."""

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Check for :class:`.PauliEvolutionGate` objects where the summands all commute.

        Args:
            dag: The DAG circuit in which to look for the commuting evolutions.

        Returns:
            The dag in which :class:`.PauliEvolutionGate` objects made of commuting two-qubit Paulis
            have been replaced with :class:`.Commuting2qBlocks`` gate instructions. These gates
            contain nodes of two-qubit :class:`.PauliEvolutionGate` objects.
        """

        for node in dag.op_nodes():
            if isinstance(node.op, PauliEvolutionGate):
                operator = node.op.operator
                if self.single_qubit_terms_only(operator):
                    continue

                if self.summands_commute(node.op.operator):
                    sub_dag = self._decompose_to_2q(dag, node.op)

                    block_op = Commuting2qBlock(set(sub_dag.op_nodes()))
                    wire_order = {
                        wire: idx
                        for idx, wire in enumerate(sub_dag.qubits)
                        if wire not in sub_dag.idle_wires()
                    }
                    dag.replace_block_with_op([node], block_op, wire_order)

        return dag

    @staticmethod
    def single_qubit_terms_only(operator: SparsePauliOp) -> bool:
        """Determine if the Paulis are made of single qubit terms only.

        Args:
            operator: The operator to check if it consists only of single qubit terms.

        Returns:
            True if the operator consists of only single qubit terms (like ``IIX + IZI``),
            and False otherwise.
        """

        for pauli in operator.paulis:
            if sum(np.logical_or(pauli.x, pauli.z)) > 1:
                return False

        return True

    @staticmethod
    def summands_commute(operator: SparsePauliOp) -> bool:
        """Check if all summands in the evolved operator commute.

        Args:
            operator: The operator to check if all its summands commute.

        Returns:
            True if all summands commute, False otherwise.
        """
        # get a list of summands that commute
        commuting_subparts = operator.paulis.group_qubit_wise_commuting()

        # if all commute we only have one summand!
        return len(commuting_subparts) == 1

    @staticmethod
    def _pauli_to_edge(pauli: Pauli) -> tuple[int, ...]:
        """Convert a pauli to an edge.

        Args:
            pauli: A pauli that is converted to a string to find out where non-identity
                Paulis are.

        Returns:
            A tuple representing where the Paulis are. For example, the Pauli "IZIZ" will
            return (0, 2) since virtual qubits 0 and 2 interact.

        Raises:
            QiskitError: If the pauli does not exactly have two non-identity terms.
        """
        edge = tuple(np.logical_or(pauli.x, pauli.z).nonzero()[0])

        if len(edge) != 2:
            raise QiskitError(f"{pauli} does not have length two.")

        return edge

    def _decompose_to_2q(self, dag: DAGCircuit, op: PauliEvolutionGate) -> DAGCircuit:
        """Decompose the SparsePauliOp into two-qubit.

        Args:
            dag: The dag needed to get access to qubits.
            op: The operator with all the Pauli terms we need to apply.

        Returns:
            A dag made of two-qubit :class:`.PauliEvolutionGate`.
        """
        sub_dag = dag.copy_empty_like()

        required_paulis = {
            self._pauli_to_edge(pauli): (pauli, coeff)
            for pauli, coeff in zip(op.operator.paulis, op.operator.coeffs)
        }

        for edge, (pauli, coeff) in required_paulis.items():

            qubits = [dag.qubits[edge[0]], dag.qubits[edge[1]]]

            simple_pauli = Pauli(pauli.to_label().replace("I", ""))

            pauli_2q = PauliEvolutionGate(simple_pauli, op.time * np.real(coeff))
            sub_dag.apply_operation_back(pauli_2q, qubits)

        return sub_dag
