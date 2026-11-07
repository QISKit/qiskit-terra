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

from qiskit.circuit import ForLoopOp, ContinueLoopOp, BreakLoopOp, IfElseOp, Parameter
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.utils import control_flow
from qiskit.converters import circuit_to_dag
from qiskit.circuit.classical import expr as _expr
from qiskit.circuit.classical.expr import Range, Var, Value, Cast
from qiskit.circuit.classical.types import Uint


def _materialize_expr_range(indexset):
    """Return a Python range only when every bound has an unambiguous integer value."""

    def bound_value(node):
        if isinstance(node, Value) and isinstance(node.type, Uint):
            value = node.value
        elif isinstance(node, Cast) and isinstance(node.type, Uint):
            value = bound_value(node.operand)
            # Narrowing is safe only when the value still fits the target Uint.
            if value is None or value >= 1 << node.type.width:
                return None
        else:
            return None
        return value if value >= 0 else None

    if not indexset.const:
        return None
    start, stop, step = (
        bound_value(bound) for bound in (indexset.start, indexset.stop, indexset.step)
    )
    if start is None or stop is None or step in (None, 0):
        return None
    materialized = range(start, stop + 1, step)
    # The pass calls len(indexset), which requires the iteration count to fit Py_ssize_t.
    try:
        len(materialized)
    except OverflowError:
        return None
    return materialized


class UnrollForLoops(TransformationPass):
    """``UnrollForLoops`` transpilation pass unrolls for-loops when possible.

    This pass unrolls :class:`~.ForLoopOp` instructions whose ``indexset`` is a
    Python :class:`range`, a tuple of integers, or a :class:`~.expr.Range` whose
    bounds are unsigned literals or value-preserving casts of literals. Other
    :class:`~.expr.Range` indexsets are skipped by default; set ``strict=True``
    to raise instead.

    For constant :class:`~.expr.Range` indexsets paired with an :class:`~.expr.Var`
    loop variable, each iteration substitutes the :class:`~.expr.Var` with the
    iteration value (of type :class:`~.expr.Value`) in every classical expression in
    the body via :meth:`~.QuantumCircuit.substitute_vars`.
    """

    def __init__(self, max_target_depth=-1, *, strict=False):
        """Things like ``for x in {0, 3, 4} {rx(x) qr[1];}`` will turn into
        ``rx(0) qr[1]; rx(3) qr[1]; rx(4) qr[1];``.

        Supported :class:`~.expr.Range` indexsets are materialized to a Python
        :class:`range` and unrolled the same way. Other :class:`~.expr.Range`
        indexsets are left unchanged unless ``strict`` is set.

        .. note::
            The ``UnrollForLoops`` unrolls only one level of block depth. No inner loop will
            be considered by ``max_target_depth``.

        Args:
            max_target_depth (int): Optional. Checks if the unrolled block is over a particular
                subcircuit depth. To disable the check, use ``-1`` (Default).
            strict (bool): If ``True``, raise :class:`~.TranspilerError` when a
                :class:`~.ForLoopOp` uses a :class:`~.expr.Range` whose bounds
                cannot be safely materialized. If ``False`` (default), skip such loops.
        """
        super().__init__()
        self.max_target_depth = max_target_depth
        self.strict = strict

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

            if isinstance(indexset, Range):
                materialized = _materialize_expr_range(indexset)
                if materialized is None:
                    if self.strict:
                        raise TranspilerError(
                            "Cannot unroll for_loop: expr.Range bounds cannot be safely "
                            "materialized as Python integers."
                        )
                    continue
                indexset = materialized

            # skip unrolling if it results in bigger than max_target_depth
            if 0 < self.max_target_depth < len(indexset) * body.depth():
                continue

            # skip unroll when break_loop or continue_loop inside body
            if _body_contains_continue_or_break(body):
                continue

            # Start the unrolled DAG empty of variables, then add back only the body's
            # captured variables.  The loop variable is the body's ``input`` variable, which we
            # substitute away on every iteration (below), so it must not be carried into the
            # accumulator.  ``vars_mode="captures"`` would re-export the body's input/declared
            # variables, so we drop all vars and re-add only the captures explicitly.
            body_dag = circuit_to_dag(body)
            unrolled_dag = body_dag.copy_empty_like(vars_mode="drop")
            for captured in body_dag.iter_captured_vars():
                unrolled_dag.add_captured_var(captured)
            unrolled_dag.global_phase = 0
            var_type = loop_param.type if isinstance(loop_param, Var) else None
            for index_value in indexset:
                if isinstance(loop_param, Var):
                    replacement = _expr.lift(index_value, type=var_type)
                    bound_body = body.substitute_vars({loop_param: replacement})
                elif isinstance(loop_param, Parameter):
                    bound_body = body.assign_parameters({loop_param: index_value})
                else:  # loop param is None
                    bound_body = body
                # ``inline_captures=True``: each iteration's body shares the body's captured
                # variables with the outer (now-unrolled) DAG, so composing repeated copies
                # doesn't try to re-declare the same captures.
                unrolled_dag.compose(circuit_to_dag(bound_body), inplace=True, inline_captures=True)
            # ``substitute_node_with_dag`` infers the wire mapping from ``input_dag.wires`` by
            # default, which includes any classical :class:`~.expr.Var` wires. The wire-count
            # check then mismatches the for-loop op (which carries no Var in its qargs/cargs).
            # Pass the qubit+clbit wires of the body explicitly so the check sees only those;
            # any vars in the unrolled DAG are matched by identity to the outer DAG's vars.
            explicit_wires = list(body.qubits) + list(body.clbits)
            dag.substitute_node_with_dag(forloop_op, unrolled_dag, wires=explicit_wires)

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
