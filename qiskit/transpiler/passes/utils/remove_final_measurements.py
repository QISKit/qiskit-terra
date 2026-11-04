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

"""Remove final measurements and barriers at the end of a circuit."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def calc_final_ops(dag: DAGCircuit, final_op_names: set[str]) -> list[DAGOpNode]:
    """Find the final operations of a circuit of a given type.
    Args:
        dag: the DAG circuit
        final_op_names: names of the operations to find at the end of the circuit.

    Returns:
    List of nodes corresponding to the relevant operations at the end of the circuit.
    """
    final_ops = []

    to_visit = [next(dag.predecessors(dag.output_map[qubit])) for qubit in dag.qubits]
    barrier_encounters_remaining = {}

    while to_visit:
        node = to_visit.pop()
        if not isinstance(node, DAGOpNode):
            continue

        if node.name == "barrier":
            # Barrier is final if all children are final, so we track
            # how many times we still need to encounter each barrier
            # via a child node.
            if node not in barrier_encounters_remaining:
                barrier_encounters_remaining[node] = sum(1 for _ in dag.quantum_successors(node))
            if barrier_encounters_remaining[node] - 1 > 0:
                # We've encountered the barrier, but not (yet) via all children.
                # Record the encounter, and bail!
                barrier_encounters_remaining[node] -= 1
                continue
        if node.name in final_op_names:
            # Current node is either a measure, or a barrier with all final op children.
            final_ops.append(node)
            to_visit.extend(dag.quantum_predecessors(node))

    return final_ops


class RemoveFinalMeasurements(TransformationPass):
    """Remove final measurements and barriers at the end of a circuit.

    This pass removes final barriers and final measurements, as well as all
    unused classical registers and bits they are connected to.
    Measurements and barriers are considered final if they are
    followed by no other operations (aside from other measurements or barriers.)

    Classical registers are removed iff they reference at least one bit
    that has become unused by the circuit as a result of the operation, and all
    of their other bits are also unused. Separately, classical bits are removed
    iff they have become unused by the circuit as a result of the operation,
    or they appear in a removed classical register, but do not appear
    in a classical register that will remain.
    """

    def run(self, dag):
        """Run the RemoveFinalMeasurements pass on `dag`.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        final_ops = calc_final_ops(dag, {"measure", "barrier"})
        if not final_ops:
            return dag

        # remove final measure and barrier operations from DAG, keeping track
        # of their clbits
        clbits_with_final_measures = set()
        for node in final_ops:
            for carg in node.cargs:
                clbits_with_final_measures.add(carg)
            dag.remove_op_node(node)

        # ignore any non-idle clbits now that all final op nodes are removed
        idle_wires = set(dag.idle_wires())
        clbits_with_final_measures &= idle_wires

        if not clbits_with_final_measures:
            # no idle wires to remove
            return dag

        # determine bits of all registers where register is now idle
        # as a result of the removal.
        idle_register_bits = set()
        busy_register_bits = set()
        for creg in dag.cregs.values():
            clbits = set(creg)
            if not clbits.isdisjoint(clbits_with_final_measures) and clbits.issubset(idle_wires):
                # register contains a newly idle bit, and all other bits are idle.
                idle_register_bits |= clbits
            else:
                # register does not contain a newly idle bit, or contains other busy bits
                # and thus should not be removed.
                busy_register_bits |= clbits

        # note: `clbits_with_final_measure` is needed here to account for loose
        # bits not in any register.
        bits_to_remove = (clbits_with_final_measures | idle_register_bits) - busy_register_bits

        dag.remove_clbits(*bits_to_remove)
        return dag
