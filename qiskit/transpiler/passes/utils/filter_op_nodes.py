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

"""Filter ops from a circuit"""

from collections.abc import Callable

from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

from qiskit._accelerate.filter_op_nodes import filter_op_nodes


class FilterOpNodes(TransformationPass):
    """Remove all operations that match a filter function

    This transformation pass is used to remove any operations that matches a
    the provided filter function.

    Args:
       predicate: A given callable that will be passed the :class:`.DAGOpNode`
           for each node in the :class:`.DAGCircuit`. If the callable returns
           ``True`` the :class:`.DAGOpNode` is retained in the circuit and if it
           returns ``False`` it is removed from the circuit.

    Example:

        Filter out operations that are labeled ``"foo"``

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import FilterOpNodes

            circuit = QuantumCircuit(1)
            circuit.x(0, label='foo')
            circuit.barrier()
            circuit.h(0)

            circuit = FilterOpNodes(
                lambda node: getattr(node.op, "label") != "foo"
            )(circuit)
            circuit.draw('mpl')
    """

    def __init__(self, predicate: Callable[[DAGOpNode], bool]):
        super().__init__()
        self.predicate = predicate

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveBarriers pass on `dag`."""
        filter_op_nodes(dag, self.predicate)
        return dag
