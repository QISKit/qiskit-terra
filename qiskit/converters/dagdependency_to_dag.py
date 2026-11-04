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

"""Helper function for converting a dag dependency to a dag circuit"""
from qiskit.dagcircuit.dagcircuit import DAGCircuit


def dagdependency_to_dag(dagdependency):
    """Build a ``DAGCircuit`` object from a ``DAGDependency``.

    Args:
        dagdependency (DAGDependency): the input dag.

    Return:
        DAGCircuit: the DAG representing the input circuit.
    """

    dagcircuit = DAGCircuit()
    dagcircuit.name = dagdependency.name
    dagcircuit.metadata = dagdependency.metadata

    dagcircuit.add_qubits(dagdependency.qubits)
    dagcircuit.add_clbits(dagdependency.clbits)

    for register in dagdependency.qregs.values():
        dagcircuit.add_qreg(register)

    for register in dagdependency.cregs.values():
        dagcircuit.add_creg(register)

    for node in dagdependency.topological_nodes():
        # Get arguments for classical control (if any)
        inst = node.op.copy()
        dagcircuit.apply_operation_back(inst, node.qargs, node.cargs)

    # copy metadata
    dagcircuit.global_phase = dagdependency.global_phase

    return dagcircuit
