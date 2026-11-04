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


"""Remove the swaps followed by measurement (and adapt the measurement)."""

from qiskit.circuit import Measure
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.dagcircuit import DAGCircuit, DAGOpNode, DAGOutNode


class OptimizeSwapBeforeMeasure(TransformationPass):
    """Remove the swaps followed by measurement (and adapt the measurement).

    Transpiler pass to remove swaps in front of measurements by re-targeting
    the classical bit of the measure instruction.
    """

    @control_flow.trivial_recurse
    def run(self, dag):
        """Run the OptimizeSwapBeforeMeasure pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        swaps = dag.op_nodes(SwapGate)
        for swap in swaps[::-1]:
            if getattr(swap.op, "_condition", None) is not None:
                continue
            final_successor = []
            for successor in dag.descendants(swap):
                final_successor.append(
                    isinstance(successor, DAGOutNode)
                    or (isinstance(successor, DAGOpNode) and isinstance(successor.op, Measure))
                )
            if all(final_successor):
                # the node swap needs to be removed and, if a measure follows, needs to be adapted
                swap_qargs = swap.qargs
                measure_layer = DAGCircuit()
                measure_layer.add_qubits(dag.qubits)
                measure_layer.add_clbits(dag.clbits)
                for qreg in dag.qregs.values():
                    measure_layer.add_qreg(qreg)
                for creg in dag.cregs.values():
                    measure_layer.add_creg(creg)

                for successor in list(dag.descendants(swap)):
                    if isinstance(successor, DAGOpNode) and isinstance(successor.op, Measure):
                        # replace measure node with a new one, where qargs is set with the "other"
                        # swap qarg.
                        dag.remove_op_node(successor)
                        old_measure_qarg = successor.qargs[0]
                        new_measure_qarg = swap_qargs[swap_qargs.index(old_measure_qarg) - 1]
                        measure_layer.apply_operation_back(
                            Measure(), (new_measure_qarg,), (successor.cargs[0],), check=False
                        )
                dag.compose(measure_layer)
                dag.remove_op_node(swap)
        return dag
