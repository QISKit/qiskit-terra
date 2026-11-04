# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Replace resets after measure with a conditional XGate."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import IfElseOp
from qiskit.circuit.reset import Reset
from qiskit.circuit.measure import Measure
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagnode import DAGOpNode


class ResetAfterMeasureSimplification(TransformationPass):
    """This pass replaces reset after measure with a conditional X gate.

    This optimization is suitable for use on IBM Quantum systems where the
    reset operation is performed by a measurement followed by a conditional
    x-gate. It might not be desirable on other backends if reset is implemented
    differently.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the pass on a dag."""
        for node in dag.op_nodes(Measure):
            succ = next(dag.quantum_successors(node))
            if isinstance(succ, DAGOpNode) and isinstance(succ.op, Reset):
                x_body = QuantumCircuit(1)
                x_body.x(0)
                new_x = IfElseOp((node.cargs[0], 1), x_body)
                new_dag = DAGCircuit()
                new_dag.add_qubits(node.qargs)
                new_dag.add_clbits(node.cargs)
                new_dag.apply_operation_back(node.op, node.qargs, node.cargs)
                new_dag.apply_operation_back(new_x, node.qargs)
                dag.remove_op_node(succ)
                dag.substitute_node_with_dag(node, new_dag)
        return dag
