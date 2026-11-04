# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""UnrollForLoops transpilation pass"""

from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag


class UnrollForLoops(TransformationPass):
    """``UnrollForLoops`` transpilation pass unrolls for-loops when possible."""

    def __init__(self, max_target_depth=-1):
        """Things like ``for x in {0, 3, 4} {rx(x) qr[1];}`` will turn into
        ``rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];``.

        .. note::
            The ``UnrollForLoops`` unrolls only one level of block depth. No inner loop will
            be considered by ``max_target_depth``.

        Args:
            max_target_depth (int): Optional. Checks if the unrolled block is over a particular
                subcircuit depth. To disable the check, use ``-1`` (Default).
        """
        super().__init__()
        self.max_target_depth = max_target_depth

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the UnrollForLoops pass on ``dag``.

        Args:
            dag (DAGCircuit): the directed acyclic graph to run on.

        Returns:
            DAGCircuit: Transformed DAG.
        """
        for forloop_op in dag.op_nodes(ForLoopOp):
            (indexset, loop_param, body) = forloop_op.op.params

            # skip unrolling if it results in bigger than max_target_depth
            if 0 < self.max_target_depth < len(indexset) * body.depth():
                continue

            # skip unroll when break_loop or continue_loop inside body
            if _body_contains_continue_or_break(body):
                continue

            unrolled_dag = circuit_to_dag(body).copy_empty_like()
            unrolled_dag.global_phase = 0
            for index_value in indexset:
                bound_body = (
                    body.assign_parameters({loop_param: index_value}) if loop_param else body
                )
                unrolled_dag.compose(circuit_to_dag(bound_body), inplace=True)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag)

        return dag


def _body_contains_continue_or_break(circuit):
    """Checks if a circuit contains ``continue``s or ``break``s. Conditional bodies are inspected."""
    for inst in circuit.data:
        operation = inst.operation
        if isinstance(operation, (ContinueLoopOp, BreakLoopOp)):
            return True
        if isinstance(operation, IfElseOp):
            for block in operation.params:
                if _body_contains_continue_or_break(block):
                    return True
    return False
