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

"""Helper function for converting a dag circuit to a dag dependency"""
from qiskit.dagcircuit.dagdependency_v2 import _DAGDependencyV2


def _dag_to_dagdependency_v2(dag):
    """Build a ``_DAGDependencyV2`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.

    Return:
        _DAGDependencyV2: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = _DAGDependencyV2()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata
    dagdependency.global_phase = dag.global_phase

    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)

    for register in dag.qregs.values():
        dagdependency.add_qreg(register)

    for register in dag.cregs.values():
        dagdependency.add_creg(register)

    for node in dag.topological_op_nodes():
        dagdependency.apply_operation_back(node.op.copy(), node.qargs, node.cargs)

    return dagdependency
