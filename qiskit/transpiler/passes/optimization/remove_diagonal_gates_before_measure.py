# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Remove diagonal gates (including diagonal 2Q gates) before a measurement."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow

from qiskit._accelerate.remove_diagonal_gates_before_measure import (
    remove_diagonal_gates_before_measure,
)


class RemoveDiagonalGatesBeforeMeasure(TransformationPass):
    """Remove diagonal gates (including diagonal 2Q gates) before a measurement.

    Transpiler pass to remove diagonal gates (like RZ, T, Z, etc) before
    a measurement. Including diagonal 2Q gates.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the RemoveDiagonalGatesBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        remove_diagonal_gates_before_measure(dag)
        return dag
