# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Add a barrier before final measurements."""

from qiskit._accelerate.barrier_before_final_measurement import barrier_before_final_measurements
from qiskit.transpiler.basepasses import TransformationPass
from .merge_adjacent_barriers import MergeAdjacentBarriers


class BarrierBeforeFinalMeasurements(TransformationPass):
    """Add a barrier before final measurements.

    This pass adds a barrier before the set of final measurements. Measurements
    are considered final if they are followed by no other operations (aside from
    other measurements or barriers.)
    """

    def __init__(self, label=None):
        super().__init__()
        self.label = label

    def run(self, dag):
        """Run the BarrierBeforeFinalMeasurements pass on `dag`."""
        barrier_before_final_measurements(dag, self.label)
        if self.label is None:
            # Merge the new barrier into any other barriers
            adjacent_pass = MergeAdjacentBarriers()
            return adjacent_pass.run(dag)
        else:
            return dag
