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

"""Reduce 1Q gate complexity by commuting through 2Q gates and resynthesizing."""

from copy import copy
import logging
from collections import deque

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, RZXGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.optimization.optimize_1q_decomposition import (
    Optimize1qGatesDecomposition,
)

logger = logging.getLogger(__name__)


commutation_table = {
    RZXGate: (["rz", "p"], ["x", "sx", "rx"]),
    CXGate: (["rz", "p"], ["x", "sx", "rx"]),
}
"""
Simple commutation rules: G belongs to commutation_table[barrier_type][qubit_preindex] when G
commutes with the indicated barrier on that qubit wire.

NOTE: Does not cover identities like
          (X (x) I) .   CX = CX .   (X (x) X) ,  (duplication)
          (U (x) I) . SWAP = SWAP . (I (x) U) .  (permutation)

NOTE: These rules are _symmetric_, so that they may be applied in reverse.
"""


class Optimize1qGatesSimpleCommutation(TransformationPass):
    """
    Optimizes 1Q gate strings interrupted by 2Q gates by commuting the components and
    resynthesizing the results.  The commutation rules are stored in ``commutation_table``.

    NOTE: In addition to those mentioned in ``commutation_table``, this pass has some limitations:
          + Does not handle multiple commutations in a row without intermediate progress.
          + Can only commute into positions where there are pre-existing runs.
          + Does not exhaustively test all the different ways commuting gates can be assigned to
            either side of a barrier to try to find low-depth configurations.  (This is particularly
            evident if all the gates in a run commute with both the predecessor and the successor
            barriers.)
    """

    # NOTE: A run from dag.collect_1q_runs is always nonempty, so we sometimes use an empty list
    #       to signify the absence of a run.

    def __init__(self, basis=None, run_to_completion=False, target=None):
        """
        Args:
            basis (List[str]): See also `Optimize1qGatesDecomposition`.
            run_to_completion (bool): If `True`, this pass retries until it is unable to do any more
                work.  If `False`, it finds and performs one optimization, and for full optimization
                the user is obligated to re-call the pass until the output stabilizes.
            target (Target): The :class:`~.Target` representing the target backend, if both
                ``basis`` and this are specified then this argument will take
                precedence and ``basis`` will be ignored.
        """
        super().__init__()

        self._optimize1q = Optimize1qGatesDecomposition(basis=basis, target=target)
        self._run_to_completion = run_to_completion

    @staticmethod
    def _find_adjoining_run(dag, runs, run, front=True):
        """
        Finds the run which abuts `run` from the front (or the rear if `front == False`), separated
        by a blocking node.

        Returns a pair of the abutting multiqubit gate and the run which it separates from this
        one. The next run can be the empty list `[]` if it is absent.
        """
        edge_node = run[0] if front else run[-1]
        blocker = next(dag.predecessors(edge_node) if front else dag.successors(edge_node))
        possibilities = dag.predecessors(blocker) if front else dag.successors(blocker)

        adjoining_run = []
        for possibility in possibilities:
            if isinstance(possibility, DAGOpNode) and possibility.qargs == edge_node.qargs:
                adjoining_run = []
                for single_run in runs:
                    if (
                        len(single_run) != 0 and single_run[0].qargs == possibility.qargs
                    ):  # allows us to only check the run on a particular qubit
                        if possibility in single_run:
                            adjoining_run = single_run
                            break
                break
        return (blocker, adjoining_run)

    @staticmethod
    def _commute_through(blocker, run, front=True):
        """
        Pulls `DAGOpNode`s from the front of `run` (or the back, if `front == False`) until it
        encounters a gate which does not commute with `blocker`.

        Returns a pair of lists whose concatenation is `run`.
        """

        if run == []:
            return [], []
        # use deque to have modification
        # operations which are constant
        # time
        run_clone = deque(run)

        commuted = deque()
        preindex, commutation_rule = None, None
        if isinstance(blocker, DAGOpNode):
            preindex = None
            for i, q in enumerate(blocker.qargs):
                if q == run[0].qargs[0]:
                    preindex = i

            commutation_rule = None
            if (
                preindex is not None
                and isinstance(blocker, DAGOpNode)
                and blocker.op.base_class in commutation_table
            ):
                commutation_rule = commutation_table[blocker.op.base_class][preindex]

        if commutation_rule is not None:
            while run_clone:
                next_gate = run_clone[0] if front else run_clone[-1]
                if next_gate.name not in commutation_rule:
                    break
                if front:
                    run_clone.popleft()
                    commuted.append(next_gate)
                else:
                    run_clone.pop()
                    commuted.appendleft(next_gate)
        if front:
            return list(commuted), list(run_clone)
        else:
            return list(run_clone), list(commuted)

    def _resynthesize(self, run, qubit):
        """
        Synthesizes an efficient circuit from a sequence `run` of `DAGOpNode`s.

        NOTE: Returns None when resynthesis is not possible.
        """
        if len(run) == 0:
            dag = DAGCircuit()
            dag.add_qreg(QuantumRegister(1))
            return dag
        operator = run[0].op.to_matrix()
        for gate in run[1:]:
            operator = gate.op.to_matrix().dot(operator)
        return self._optimize1q._gate_sequence_to_dag(
            self._optimize1q._resynthesize_run(operator, qubit)
        )

    @staticmethod
    def _replace_subdag(dag, old_run, new_dag):
        """
        Replaces a nonempty sequence `old_run` of `DAGNode`s, assumed to be a complete chain in
        `dag`, with the circuit `new_circ`.
        """

        node_map = dag.substitute_node_with_dag(old_run[0], new_dag)

        for node in old_run[1:]:
            dag.remove_op_node(node)

        spliced_run = [node_map[node._node_id] for node in new_dag.topological_op_nodes()]
        mov_list(old_run, spliced_run)

    def _step(self, dag):
        """
        Performs one full pass of optimization work.

        Returns True if `dag` changed, False if no work on `dag` was possible.
        """

        runs = dag.collect_1q_runs()
        did_work = False

        for run in runs:
            # identify the preceding blocking gates
            run_clone = copy(run)
            if run == []:
                continue

            # try to modify preceding_run
            preceding_blocker, preceding_run = self._find_adjoining_run(dag, runs, run)
            commuted_preceding = []
            if preceding_run != []:
                commuted_preceding, run_clone = self._commute_through(preceding_blocker, run_clone)

            # try to modify succeeding run
            succeeding_blocker, succeeding_run = self._find_adjoining_run(
                dag, runs, run, front=False
            )
            commuted_succeeding = []
            if succeeding_run != []:
                run_clone, commuted_succeeding = self._commute_through(
                    succeeding_blocker, run_clone, front=False
                )

            # re-synthesize
            qubit = dag.find_bit(run[0].qargs[0]).index
            new_preceding_run = self._resynthesize(preceding_run + commuted_preceding, qubit)
            new_succeeding_run = self._resynthesize(commuted_succeeding + succeeding_run, qubit)
            new_run = self._resynthesize(run_clone, qubit)

            # perform the replacement if it was indeed a good idea
            if self._optimize1q._substitution_checks(
                (preceding_run or []) + run + (succeeding_run or []),
                new_preceding_run.op_nodes() + new_run.op_nodes() + new_succeeding_run.op_nodes(),
                self._optimize1q._basis_gates,
                dag.find_bit(run[0].qargs[0]).index,
            ):
                if preceding_run and new_preceding_run is not None:
                    self._replace_subdag(dag, preceding_run, new_preceding_run)
                if succeeding_run and new_succeeding_run is not None:
                    self._replace_subdag(dag, succeeding_run, new_succeeding_run)
                if new_run is not None:
                    self._replace_subdag(dag, run, new_run)
                did_work = True

        return did_work

    def run(self, dag):
        """
        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """

        # python doesn't support tail calls
        while True:
            did_work = self._step(dag)
            if not self._run_to_completion or not did_work:
                break

        return dag


def mov_list(destination, source):
    """
    Replace `destination` in-place with `source`.
    """

    while destination:
        del destination[0]
    destination += source
