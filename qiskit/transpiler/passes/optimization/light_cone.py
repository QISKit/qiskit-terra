# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Cancel the redundant (self-adjoint) gates through commutation relations."""

from __future__ import annotations
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import PauliGate, ZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils.remove_final_measurements import calc_final_ops

translation_table = str.maketrans({"+": "X", "-": "X", "l": "Y", "r": "Y", "0": "Z", "1": "Z"})


class LightCone(TransformationPass):
    """Remove the gates that do not affect the outcome of a measurement on a circuit.

    Pass for computing the light-cone of an observable or measurement. The Pass can handle
    either an observable one would like to measure or a measurement on a set of qubits.
    """

    def __init__(self, bit_terms: str | None = None, indices: list[int] | None = None) -> None:
        """
        Args:
            bit_terms: If ``None`` the light-cone will be computed for the set of measurements
                in the circuit. If a string is specified, the light-cone will correspond to the
                reduced circuit with the same expectation value for the observable.
            indices: list of non-trivial indices corresponding to the observable in ``bit_terms``.
        """
        super().__init__()
        valid_characters = {"X", "Y", "Z", "+", "-", "l", "r", "0", "1"}
        self.bit_terms = None
        if bit_terms is not None:
            if not indices:
                raise ValueError("`indices` must be non-empty when providing `bit_terms`.")
            if not set(bit_terms).issubset(valid_characters):
                raise ValueError(
                    f"`bit_terms` should contain only characters in {valid_characters}."
                )
            if len(bit_terms) != len(indices):
                raise ValueError("`bit_terms` must be the same length as `indices`.")
            self.bit_terms = bit_terms.translate(translation_table)
        self.indices = indices

    @staticmethod
    def _find_measurement_qubits(dag: DAGCircuit) -> set[Qubit]:
        final_nodes = calc_final_ops(dag, {"measure"})
        qubits_measured = set()
        for node in final_nodes:
            qubits_measured |= set(node.qargs)
        return qubits_measured

    def _get_initial_lightcone(
        self, dag: DAGCircuit
    ) -> tuple[set[Qubit], list[tuple[Gate, list[Qubit]]]]:
        """Returns the initial light-cone.
        If observable is `None`, the light-cone is the set of measured qubits.
        If a `bit_terms` is provided, the qubits corresponding to the
        non-trivial Paulis define the light-cone.
        """
        lightcone_qubits = self._find_measurement_qubits(dag)
        if self.bit_terms is None:
            lightcone_operations = [(ZGate(), [qubit_index]) for qubit_index in lightcone_qubits]
        else:
            # Having both measurements and an observable is not allowed
            if len(dag.qubits) < max(self.indices) + 1:
                raise ValueError("`indices` contains values outside the qubit range.")
            if lightcone_qubits:
                raise ValueError(
                    "The circuit contains measurements and an observable has been given: "
                    "remove the observable or the measurements."
                )
            lightcone_qubits = [dag.qubits[i] for i in self.indices]
            # `lightcone_operations` is a list of tuples, each containing (operation, list_of_qubits)
            lightcone_operations = [(PauliGate(self.bit_terms), lightcone_qubits)]

        return set(lightcone_qubits), lightcone_operations

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the LightCone pass on `dag`.

        Args:
            dag: The DAG to reduce.

        Returns:
            The DAG reduced to the light-cone of the observable.
        """

        # Get the initial light-cone and operations
        lightcone_qubits, lightcone_operations = self._get_initial_lightcone(dag)

        #  Initialize a new, empty DAG
        new_dag = dag.copy_empty_like()

        # Iterate over the nodes in reverse topological order
        for node in dag.topological_op_nodes(reverse=True):
            # Check if the node belongs to the light-cone
            if lightcone_qubits.intersection(node.qargs):
                # Check commutation with all previous operations
                commutes_bool = True
                for op in lightcone_operations:
                    commute_bool = scc.commute(op[0], op[1], [], node.op, node.qargs, [])
                    if not commute_bool:
                        # If the current node does not commute, update the light-cone
                        lightcone_qubits.update(node.qargs)
                        lightcone_operations.append((node.op, node.qargs))
                        commutes_bool = False
                        break

                # If the node is in the light-cone and commutes with previous `ops`,
                # add it to the new DAG at the front
                if not commutes_bool:
                    new_dag.apply_operation_front(node.op, node.qargs, node.cargs)
        return new_dag
