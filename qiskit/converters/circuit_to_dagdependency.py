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

"""Helper function for converting a circuit to a dag dependency"""

from qiskit.dagcircuit.dagdependency import DAGDependency


def circuit_to_dagdependency(circuit, create_preds_and_succs=True):
    """Build a ``DAGDependency`` object from a :class:`~.QuantumCircuit`.

    Args:
        circuit (QuantumCircuit): the input circuit.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = DAGDependency()
    dagdependency.name = circuit.name
    dagdependency.metadata = circuit.metadata

    dagdependency.add_qubits(circuit.qubits)
    dagdependency.add_clbits(circuit.clbits)

    for register in circuit.qregs:
        dagdependency.add_qreg(register)

    for register in circuit.cregs:
        dagdependency.add_creg(register)

    for instruction in circuit.data:
        dagdependency.add_op_node(instruction.operation, instruction.qubits, instruction.clbits)

    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()

    # copy global phase
    dagdependency.global_phase = circuit.global_phase

    return dagdependency
