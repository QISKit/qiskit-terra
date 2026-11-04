# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""VF2PostLayout pass to find a layout after transpile using subgraph isomorphism"""

from enum import Enum

from qiskit.transpiler.layout import Layout
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit._accelerate.vf2_layout import (
    vf2_layout_pass_average,
    vf2_layout_pass_exact,
    MultiQEncountered,
    VF2PassConfiguration,
)


class VF2PostLayoutStopReason(Enum):
    """Stop reasons for VF2PostLayout pass."""

    SOLUTION_FOUND = "solution found"
    NO_BETTER_SOLUTION_FOUND = "no better solution found"
    NO_SOLUTION_FOUND = "nonexistent solution"
    MORE_THAN_2Q = ">2q gates in basis"


class VF2PostLayout(AnalysisPass):
    """A pass for improving an existing Layout after transpilation of a circuit onto a
    Coupling graph, as a subgraph isomorphism problem, solved by VF2++.

    Unlike the :class:`~.VF2Layout` transpiler pass which is designed to find an
    initial layout for a circuit early in the transpilation pipeline this transpiler
    pass is designed to try and find a better layout after transpilation is complete.
    The initial layout phase of the transpiler doesn't have as much information available
    as we do after transpilation. This pass is designed to be paired in a similar pipeline
    as the layout passes. This pass will strip any idle wires from the circuit, use VF2
    to find a subgraph in the coupling graph for the circuit to run on with better fidelity
    and then update the circuit layout to use the new qubits. The algorithm used in this
    pass is described in `arXiv:2209.15512 <https://arxiv.org/abs/2209.15512>`__.

    If a solution is found that means there is a lower error layout available for the
    circuit. If a solution is found the layout will be set in the property set as
    ``property_set['post_layout']``. However, if no solution or no better solution is found, no
    ``property_set['post_layout']`` is set. The stopping reason is
    set in ``property_set['VF2PostLayout_stop_reason']`` in all the cases and will be
    one of the values enumerated in ``VF2PostLayoutStopReason`` which has the
    following values:

        * ``"solution found"``: If a solution was found.
        * ``"no better solution found"``: If the initial layout of the circuit is the best solution.
        * ``"nonexistent solution"``: If no solution was found.
        * ``">2q gates in basis"``: If VF2PostLayout can't work with the basis of the circuit.

    By default, this pass will construct a heuristic scoring map based on
    the error rates in the provided ``target``. However, analysis passes can be run prior to this pass
    and set ``vf2_avg_error_map`` in the property set with a :class:`~.ErrorMap`
    instance. If a value is ``NaN`` that is treated as an ideal edge
    For example if an error map is created as::

        from qiskit.transpiler.passes.layout.vf2_utils import ErrorMap

        error_map = ErrorMap(3)
        error_map.add_error((0, 0), 0.0024)
        error_map.add_error((0, 1), 0.01)
        error_map.add_error((1, 1), 0.0032)

    that represents the error map for a 2 qubit target, where the avg 1q error
    rate is ``0.0024`` on qubit 0 and ``0.0032`` on qubit 1. Then the avg 2q
    error rate for gates that operate on (0, 1) is 0.01 and (1, 0) is not
    supported by the target. This will be used for scoring if it's set as the
    ``vf2_avg_error_map`` key in the property set when :class:`~.VF2PostLayout`
    is run.
    """

    def __init__(
        self,
        target=None,
        seed=None,
        call_limit=None,
        time_limit=None,
        strict_direction=True,
        max_trials=0,
    ):
        """Initialize a ``VF2PostLayout`` pass instance

        Args:
            target (Target): A target representing the backend device to run ``VF2PostLayout`` on.
            seed (int): Sets the seed of the PRNG. -1 Means no node shuffling.
            call_limit (int): The number of state visits to attempt in each execution of
                VF2.
            time_limit (float): The total time limit in seconds to run ``VF2PostLayout``
            strict_direction (bool): Whether the pass is configured to follow
                the strict direction in the coupling graph. If this is set to
                false, the pass will treat any edge in the coupling graph as
                a weak edge and the interaction graph will be undirected. For
                the purposes of evaluating layouts the avg error rate for
                each qubit and 2q link will be used. This enables the pass to be
                run prior to basis translation and work with any 1q and 2q operations.
                However, if ``strict_direction=True`` the pass expects the input
                :class:`~.DAGCircuit` object to :meth:`~.VF2PostLayout.run` to be in
                the target set of instructions.
            max_trials (int): The maximum number of trials to run VF2 to find
                a layout. A value of ``0`` (the default) means 'unlimited'.

        Raises:
            TypeError: At runtime, if ``target`` isn't provided.
        """
        super().__init__()
        self.target = target
        self.call_limit = call_limit
        self.time_limit = time_limit
        self.max_trials = max_trials
        self.seed = seed
        self.strict_direction = strict_direction
        self.avg_error_map = None

    def run(self, dag):
        """run the layout method"""
        if self.target is None:
            raise TranspilerError("A target must be specified")
        self.avg_error_map = self.property_set["vf2_avg_error_map"]
        config = VF2PassConfiguration.from_legacy_api(
            call_limit=self.call_limit,
            time_limit=self.time_limit,
            max_trials=self.max_trials,
            shuffle_seed=self.seed,
            score_initial_layout=True,
        )
        try:
            if self.strict_direction:
                output = vf2_layout_pass_exact(dag, self.target, config=config)
            else:
                output = vf2_layout_pass_average(
                    dag,
                    self.target,
                    strict_direction=False,
                    avg_error_map=self.avg_error_map,
                    config=config,
                )
        except MultiQEncountered:
            self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.MORE_THAN_2Q
            return
        if not output.has_solution:
            self.property_set["VF2PostLayout_stop_reason"] = (
                VF2PostLayoutStopReason.NO_SOLUTION_FOUND
            )
            return
        if dag.is_empty() or (layout := output.new_mapping()) is None:
            self.property_set["VF2PostLayout_stop_reason"] = (
                VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND
            )
            return
        self.property_set["VF2PostLayout_stop_reason"] = VF2PostLayoutStopReason.SOLUTION_FOUND
        layout = Layout({dag.qubits[virt]: phys for virt, phys in layout.items()})
        for reg in dag.qregs.values():
            layout.add_register(reg)
        self.property_set["post_layout"] = layout
