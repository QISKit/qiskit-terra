# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ConstrainedReschedule pass."""

import unittest

from qiskit import QuantumCircuit
from qiskit.transpiler import InstructionDurations
from qiskit.transpiler.passes import (
    ASAPScheduleAnalysis,
    ConstrainedReschedule,
    ALAPScheduleAnalysis,
)
from qiskit.transpiler.passmanager import PassManager

from test import QiskitTestCase


class TestConstrainedReschedule(QiskitTestCase):
    """Tests for the :class:`.ConstrainedReschedule` pass."""

    def test_alignment_only_construction(self):
        """Regression test of #16245."""
        # Test contributed with assistance from Claude Opus 4.7 (Claude Code).
        durations = InstructionDurations(
            [("x", 0, 160, "dt"), ("measure", 0, 1000, "dt")], dt=2.22e-10
        )

        # ``x`` starts at t=0 (aligned to any pulse_alignment), ``measure``
        # starts at t=160 which is a multiple of acquire_alignment=16, so the
        # pass should not need to shift anything; it just has to run.
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)

        pm = PassManager(
            [
                ASAPScheduleAnalysis(durations),
                ConstrainedReschedule(acquire_alignment=16, pulse_alignment=1),
            ]
        )
        pm.run(qc)

        node_start_time = pm.property_set["node_start_time"]
        starts_by_name = {node.op.name: time for node, time in node_start_time.items()}
        self.assertEqual(starts_by_name, {"x": 0, "measure": 160})

    def test_no_panic_underflow(self):
        """Regression test of #16231."""
        durations = InstructionDurations(
            [("x", 0, 160, "dt"), ("measure", 0, 1000, "dt")], dt=2.22e-10
        )
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qc.delay(200, 0, unit="dt")
        qc.measure(0, 0)

        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                ConstrainedReschedule(acquire_alignment=16, pulse_alignment=16),
            ]
        )
        _ = pm.run(qc)
        for node, start_time in pm.property_set["node_start_time"].items():
            if node.op.name == "measure":
                self.assertEqual(0, start_time % 16)


if __name__ == "__main__":
    unittest.main()
