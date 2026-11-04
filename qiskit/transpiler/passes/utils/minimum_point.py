# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if the DAG has reached a relative semi-stable point over previous runs."""

from copy import deepcopy
from dataclasses import dataclass
import math

from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class MinimumPoint(TransformationPass):
    """Check if the DAG has reached a relative semi-stable point over previous runs

    This pass is similar to the :class:`~.FixedPoint` transpiler pass and is intended
    primarily to be used to set a loop break condition in the property set.
    However, unlike the :class:`~.FixedPoint` class which only sets the
    condition if 2 consecutive runs have the same value property set value
    this pass is designed to find a local minimum and use that instead. This
    pass is designed for an optimization loop where a fixed point may never
    get reached (for example if synthesis is used and there are multiple
    equivalent outputs for some cases).

    This pass will track the state of fields in the property set over its past
    executions and set a boolean field when either a fixed point is reached
    over the backtracking depth or selecting the minimum value found if the
    backtracking depth is reached. To do this it stores a deep copy of the
    current minimum DAG in the property set and when ``backtrack_depth`` number
    of executions is reached since the last minimum the output dag is set to
    that copy of the earlier minimum.

    Fields used by this pass in the property set are (all relative to the ``prefix``
    argument):

    * ``{prefix}_minimum_point_state`` - Used to track the state of the minimum point search
    * ``{prefix}_minimum_point`` - This value gets set to ``True`` when either a fixed point
        is reached over the ``backtrack_depth`` executions, or ``backtrack_depth`` was exceeded
        and an earlier minimum is restored.
    """

    def __init__(self, property_set_list, prefix, backtrack_depth=5):
        """Initialize an instance of this pass

        Args:
            property_set_list (list): A list of property set keys that will
                be used to evaluate the local minimum. The values of these
                property set keys will be used as a tuple for comparison
            prefix (str): The prefix to use for the property set key that is used
                for tracking previous evaluations
            backtrack_depth (int): The maximum number of entries to store. If
                this number is reached and the next iteration doesn't have
                a decrease in the number of values the minimum of the previous
                n will be set as the output dag and ``minimum_point`` will be set to
                ``True`` in the property set
        """
        super().__init__()
        self.property_set_list = property_set_list

        self.backtrack_name = f"{prefix}_minimum_point_state"
        self.minimum_reached = f"{prefix}_minimum_point"
        self.backtrack_depth = backtrack_depth

    def run(self, dag):
        """Run the MinimumPoint pass on `dag`."""
        score = tuple(self.property_set[x] for x in self.property_set_list)
        state = self.property_set[self.backtrack_name]

        # The pass starts at None and the first iteration doesn't set a real
        # score so the overall loop is treated as a do-while to ensure we have
        # at least 2 iterations.
        if state is None:
            self.property_set[self.backtrack_name] = _MinimumPointState(
                dag=None, score=(math.inf,) * len(self.property_set_list), since=0
            )
        # If the score of this execution is worse than the previous execution
        # increment 'since' since we have not found a new minimum point
        elif score > state.score:
            state.since += 1
            if state.since == self.backtrack_depth:
                self.property_set[self.minimum_reached] = True
                return self.property_set[self.backtrack_name].dag
        # If the score has decreased (gotten better) then this iteration is
        # better performing and this iteration should be the new minimum state.
        # So update the state to be this iteration and reset counter
        elif score < state.score:
            state.since = 1
            state.score = score
            state.dag = deepcopy(dag)
        # If the current execution is equal to the previous minimum value then
        # we've reached an equivalent fixed point and we should use this iteration's
        # dag as the output and set the property set flag that we've found a minimum
        # point.
        elif score == state.score:
            self.property_set[self.minimum_reached] = True
            return dag

        return dag


@dataclass
class _MinimumPointState:
    __slots__ = ("dag", "score", "since")

    dag: DAGCircuit
    score: tuple[float, ...]
    since: int
