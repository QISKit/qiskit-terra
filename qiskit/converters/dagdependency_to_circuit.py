# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper function for converting a dag dependency to a circuit"""
from qiskit.circuit import QuantumCircuit, CircuitInstruction


def dagdependency_to_circuit(dagdependency):
    """Build a ``QuantumCircuit`` object from a ``DAGDependency``.

    Args:
        dagdependency (DAGDependency): the input dag.

    Return:
        QuantumCircuit: the circuit representing the input dag dependency.
    """

    name = dagdependency.name or None
    circuit = QuantumCircuit(
        dagdependency.qubits,
        dagdependency.clbits,
        *dagdependency.qregs.values(),
        *dagdependency.cregs.values(),
        name=name,
    )
    circuit.metadata = dagdependency.metadata
    circuit.global_phase = dagdependency.global_phase

    for node in dagdependency.topological_nodes():
        circuit._append(CircuitInstruction(node.op.copy(), node.qargs, node.cargs))

    return circuit
