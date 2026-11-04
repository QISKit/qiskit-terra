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

"""Remove all barriers in a circuit"""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow


class RemoveBarriers(TransformationPass):
    """Return a circuit with any barrier removed.

    This transformation is not semantics preserving.

    Example:

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.transpiler.passes import RemoveBarriers

            circuit = QuantumCircuit(1)
            circuit.x(0)
            circuit.barrier()
            circuit.h(0)

            circuit = RemoveBarriers()(circuit)
            circuit.draw('mpl')

    """

    @control_flow.trivial_recurse
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveBarriers pass on `dag`."""

        dag.remove_all_ops_named("barrier")

        return dag
