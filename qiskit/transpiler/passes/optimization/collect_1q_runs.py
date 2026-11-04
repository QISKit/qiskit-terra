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

"""Collect sequences of uninterrupted gates acting on 1 qubit."""

from __future__ import annotations

import typing

from collections.abc import Callable

from qiskit.transpiler.basepasses import AnalysisPass

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit, DAGOpNode


class Collect1qRuns(AnalysisPass):
    """Collect one-qubit subcircuits."""

    def __init__(self, filter_fn: Callable[[DAGCircuit, list[DAGOpNode]], bool] | None = None):
        """
        Args:
            filter_fn: An optional function that filters collected one-qubit runs.
        """
        self.filter_fn = filter_fn
        super().__init__()

    def run(self, dag):
        """Run the Collect1qBlocks pass on `dag`.

        The blocks contain "op" nodes in topological order such that all gates
        in a block act on the same qubits, are adjacent in the circuit, and
        satisfy the filtering condition (when specified).

        After the execution, ``property_set['run_list']`` is set to a list of
        tuples of "op" node.
        """
        run_list = dag.collect_1q_runs()
        if self.filter_fn is not None:
            run_list = [run for run in run_list if self.filter_fn(dag, run)]
        self.property_set["run_list"] = run_list
        return dag
