# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Internal utilities for working with control-flow operations."""

import functools
from collections.abc import Callable

from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit


def map_blocks(dag_mapping: Callable[[DAGCircuit], DAGCircuit], op: ControlFlowOp) -> ControlFlowOp:
    """Use the ``dag_mapping`` function to replace the blocks of a :class:`.ControlFlowOp` with new
    ones.  Each block will be automatically converted to a :class:`.DAGCircuit` and then returned
    to a :class:`.QuantumCircuit`."""
    return op.replace_blocks(
        [
            dag_to_circuit(dag_mapping(circuit_to_dag(block)), copy_operations=False)
            for block in op.blocks
        ]
    )


def trivial_recurse(method):
    """Decorator that causes :class:`.BasePass.run` to iterate over all control-flow nodes,
    replacing their operations with a new :class:`.ControlFlowOp` whose blocks have all had
    :class`.BasePass.run` called on them.

    This is only suitable for simple run calls that store no state between calls, do not need
    circuit-specific information feeding into them (such as via a :class:`.PropertySet`), and will
    safely do nothing to control-flow operations that are in the DAG.

    If slightly finer control is needed on when the control-flow operations are modified, one can
    use :func:`map_blocks` as::

        if isinstance(node.op, ControlFlowOp):
            dag.substitute_node(node, map_blocks(self.run, node.op))

    from with :meth:`.BasePass.run`."""

    @functools.wraps(method)
    def out(self, dag):
        def bound_wrapped_method(dag):
            return out(self, dag)

        for node in dag.control_flow_op_nodes():
            dag.substitute_node(node, map_blocks(bound_wrapped_method, node.op))
        return method(self, dag)

    return out
