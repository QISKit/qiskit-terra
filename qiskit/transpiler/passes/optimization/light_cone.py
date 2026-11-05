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
from qiskit.circuit.library import PauliGate, ZGate, PauliProductMeasurement
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info import Pauli

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
    def _measurement_commutation_ops(node) -> list[tuple]:
        """Return the list of (measurement_gate, qubits) for a measurement node.
        For a standard measure, we use one ZGate per qubit.
        For a PauliProductMeasurement, we use a PauliGate.
        """
        if node.op.name == "measure":
            return [(ZGate(), [qubit]) for qubit in node.qargs]
        elif isinstance(node.op, PauliProductMeasurement):
            pauli_label = Pauli((node.op._pauli_z, node.op._pauli_x, 0)).to_label()
            return [(PauliGate(pauli_label), list(node.qargs))]

        raise ValueError("_measurement_commutation_ops called with unexpected operation")

    def _get_initial_lightcone(
        self, dag: DAGCircuit
    ) -> tuple[set[Qubit], list[tuple[Gate, list[Qubit]]]]:
        """Returns the initial light-cone.
        If observable is `None`, the light-cone is the set of measured qubits.
        If a `bit_terms` is provided, the qubits corresponding to the
        non-trivial Paulis define the light-cone.
        """
        if self.bit_terms is None:
            # will be discovered dynamically during the reverse traversal in run()
            return set(), []
        else:
            # Having both measurements and an observable is not allowed
            all_measured_qubits = set()
            for node in dag.topological_op_nodes():
                if node.op.name == "measure" or isinstance(node.op, PauliProductMeasurement):
                    all_measured_qubits.update(node.qargs)
            if len(dag.qubits) < max(self.indices) + 1:
                raise ValueError("`indices` contains values outside the qubit range.")
            if all_measured_qubits:
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
            # Each time we encounter a measurement, extend the cone with its
            # qubits and add the appropriate measurement gate.
            if node.op.name in ("measure", "pauli_product_measurement"):
                new_qubits = set(node.qargs) - lightcone_qubits
                if new_qubits:
                    lightcone_qubits.update(new_qubits)
                    lightcone_operations.extend(self._measurement_commutation_ops(node))
                new_dag.apply_operation_front(node.op, node.qargs, node.cargs)
                continue
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
