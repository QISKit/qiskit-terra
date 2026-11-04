# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Splits each two-qubit gate in the `dag` into two single-qubit gates, if possible without error."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit._accelerate.split_2q_unitaries import split_2q_unitaries


class Split2QUnitaries(TransformationPass):
    """Attempt to split two-qubit unitaries in a :class:`.DAGCircuit` into two single-qubit gates.

    This pass will analyze all :class:`.UnitaryGate` instances and determine whether the
    matrix is actually a product of 2 single qubit gates. In these cases the 2q gate can be
    simplified into two single qubit gates and this pass will perform this optimization and will
    replace the two qubit gate with two single qubit :class:`.UnitaryGate`.

    If some of the gates can be viewed as a swap joined by the product of 2 single qubit gates,
    the pass will recreate the DAG, permuting the swapped qubits similar
    to how it's done in :class:`ElidePermutations`.
    """

    def __init__(self, fidelity: float = 1.0 - 1e-16, split_swap: bool = False):
        """
        Args:
            fidelity: Allowed tolerance for splitting two-qubit unitaries and gate decompositions.
            split_swap: Whether to attempt to split swap gates, resulting in a permutation of the qubits.
        """
        super().__init__()
        self.requested_fidelity = fidelity
        self.split_swap = split_swap

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Split2QUnitaries pass on `dag`."""
        result = split_2q_unitaries(dag, self.requested_fidelity, self.split_swap)
        if result is None:
            return dag

        (new_dag, qubit_mapping) = result
        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout, dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout
        return new_dag
