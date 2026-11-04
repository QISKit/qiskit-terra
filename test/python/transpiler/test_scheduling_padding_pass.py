# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Scheduling/PadDelay passes"""

import unittest

from ddt import ddt, data
from qiskit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit.library import CXGate, HGate
from qiskit.transpiler import PropertySet
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import (
    ASAPScheduleAnalysis,
    ALAPScheduleAnalysis,
    ConstrainedReschedule,
    PadDelay,
    TimeUnitConversion,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase


@ddt
class TestSchedulingAndPaddingPass(QiskitTestCase):
    """Tests the Scheduling passes"""

    def test_alap_agree_with_reverse_asap_reverse(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        durations = InstructionDurations(
            [("h", 0, 200), ("cx", [0, 1], 700), ("measure", None, 1000)], dt=1e-7
        )

        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        alap_qc = pm.run(qc)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    def test_alap_agree_with_reverse_asap_with_target(self):
        """Test if ALAP schedule agrees with doubly-reversed ASAP schedule."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)
        qc.measure_all()

        target = Target(num_qubits=2, dt=3.5555555555555554)
        target.add_instruction(HGate(), {(0,): InstructionProperties(duration=200)})
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=700)})
        target.add_instruction(
            Measure(),
            {
                (0,): InstructionProperties(duration=1000),
                (1,): InstructionProperties(duration=1000),
            },
        )

        pm = PassManager([ALAPScheduleAnalysis(target=target), PadDelay(target=target)])
        alap_qc = pm.run(qc)

        pm = PassManager([ASAPScheduleAnalysis(target=target), PadDelay(target=target)])
        new_qc = pm.run(qc.reverse_ops())
        new_qc = new_qc.reverse_ops()
        new_qc.name = new_qc.name

        self.assertEqual(alap_qc, new_qc)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with measure after measure with a common clbit.
        See: https://github.com/Qiskit/qiskit-terra/issues/7654

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             └───┘└╥┘┌─┐
        q_1: ──────╫─┤M├
                   ║ └╥┘
        c: 1/══════╩══╩═
                   0  0

        (scheduled)
                    ┌───┐       ┌─┐┌─────────────────┐
        q_0: ───────┤ X ├───────┤M├┤ Delay(1000[dt]) ├
             ┌──────┴───┴──────┐└╥┘└───────┬─┬───────┘
        q_1: ┤ Delay(1200[dt]) ├─╫─────────┤M├────────
             └─────────────────┘ ║         └╥┘
        c: 1/════════════════════╩══════════╩═════════
                                 0          0
        """
        qc = QuantumCircuit(2, 1)
        qc.x(0)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("x", None, 200), ("measure", None, 1000)], dt=1e-7)
        pm = PassManager([schedule_pass(durations), PadDelay(durations=durations)])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.x(0)
        expected.measure(0, 0)
        expected.delay(1200, 1)
        expected.measure(1, 0)
        expected.delay(1000, 0)

        self.assertEqual(expected, scheduled)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_empty_circuit(self, schedule_pass):
        """An empty circuit is trivially scheduled, so we should succeed without error."""
        target = Target(num_qubits=4)
        target.add_instruction(
            CXGate(), {(i, i + 1): InstructionProperties(duration=1e-3) for i in range(3)}
        )
        target.add_instruction(
            Measure(), {(i,): InstructionProperties(duration=1e-3) for i in range(4)}
        )
        qc = QuantumCircuit(4, 4)
        property_set = PropertySet()
        pass_ = schedule_pass(target=target)
        pass_(qc, property_set=property_set)
        self.assertEqual(property_set["node_start_time"], {})

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_shorter_measure_after_measure(self, schedule_pass):
        """Test if ALAP/ASAP schedules circuits with shorter measure after measure with a common clbit.

        (input)
             ┌─┐
        q_0: ┤M├───
             └╥┘┌─┐
        q_1: ─╫─┤M├
              ║ └╥┘
        c: 1/═╩══╩═
              0  0

        (scheduled)
                                ┌─┐┌────────────────┐
        q_0: ───────────────────┤M├┤ Delay(700[dt]) ├
             ┌─────────────────┐└╥┘└──────┬─┬───────┘
        q_1: ┤ Delay(1000[dt]) ├─╫────────┤M├────────
             └─────────────────┘ ║        └╥┘
        c: 1/════════════════════╩═════════╩═════════
                                 0         0
        """
        qc = QuantumCircuit(2, 1)
        qc.measure(0, 0)
        qc.measure(1, 0)

        durations = InstructionDurations([("measure", [0], 1000), ("measure", [1], 700)], dt=1e-7)
        pm = PassManager([schedule_pass(durations), PadDelay(durations=durations)])
        scheduled = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.measure(0, 0)
        expected.delay(1000, 1)
        expected.measure(1, 0)
        expected.delay(700, 0)

        self.assertEqual(expected, scheduled)

    def test_parallel_gate_different_length(self):
        """Test circuit having two parallel instruction with different length.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├┤M├
             └─────┬───┬──────┘└┬─┬┘└╥┘
        q_1: ──────┤ X ├────────┤M├──╫─
                   └───┘        └╥┘  ║
        c: 2/════════════════════╩═══╩═
                                 1   0

        (expected, ASAP)
             ┌───┐┌─┐┌────────────────┐
        q_0: ┤ X ├┤M├┤ Delay(200[dt]) ├
             ├───┤└╥┘└──────┬─┬───────┘
        q_1: ┤ X ├─╫────────┤M├────────
             └───┘ ║        └╥┘
        c: 2/══════╩═════════╩═════════
                   0         1

        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)], dt=1e-7
        )
        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.x(1)
        asap_expected.measure(0, 0)  # immediately start after X gate
        asap_expected.measure(1, 1)
        asap_expected.delay(200, 0)

        self.assertEqual(qc_asap, asap_expected)

    def test_parallel_gate_different_length_with_barrier(self):
        """Test circuit having two parallel instruction with different length with barrier.

        (input)
             ┌───┐┌─┐
        q_0: ┤ X ├┤M├───
             ├───┤└╥┘┌─┐
        q_1: ┤ X ├─╫─┤M├
             └───┘ ║ └╥┘
        c: 2/══════╩══╩═
                   0  1

        (expected, ALAP)
             ┌────────────────┐┌───┐ ░ ┌─┐
        q_0: ┤ Delay(200[dt]) ├┤ X ├─░─┤M├───
             └─────┬───┬──────┘└───┘ ░ └╥┘┌─┐
        q_1: ──────┤ X ├─────────────░──╫─┤M├
                   └───┘             ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1

        (expected, ASAP)
             ┌───┐┌────────────────┐ ░ ┌─┐
        q_0: ┤ X ├┤ Delay(200[dt]) ├─░─┤M├───
             ├───┤└────────────────┘ ░ └╥┘┌─┐
        q_1: ┤ X ├───────────────────░──╫─┤M├
             └───┘                   ░  ║ └╥┘
        c: 2/═══════════════════════════╩══╩═
                                        0  1
        """
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.x(1)
        qc.barrier()
        qc.measure(0, 0)
        qc.measure(1, 1)

        durations = InstructionDurations(
            [("x", [0], 200), ("x", [1], 400), ("measure", None, 1000)], dt=1e-7
        )
        pm = PassManager([ALAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        qc_alap = pm.run(qc)

        alap_expected = QuantumCircuit(2, 2)
        alap_expected.delay(200, 0)
        alap_expected.x(0)
        alap_expected.x(1)
        alap_expected.barrier()
        alap_expected.measure(0, 0)
        alap_expected.measure(1, 1)

        self.assertEqual(qc_alap, alap_expected)

        pm = PassManager([ASAPScheduleAnalysis(durations), PadDelay(durations=durations)])
        qc_asap = pm.run(qc)

        asap_expected = QuantumCircuit(2, 2)
        asap_expected.x(0)
        asap_expected.delay(200, 0)
        asap_expected.x(1)
        asap_expected.barrier()
        asap_expected.measure(0, 0)
        asap_expected.measure(1, 1)

        self.assertEqual(qc_asap, asap_expected)

    def test_constrained_reschedule_accepts_barrier(self):
        """Test that ConstrainedReschedule allows compiler directives."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.barrier(0)

        target = Target(num_qubits=2, dt=1, pulse_alignment=1, acquire_alignment=1)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(duration=100)})

        pm = PassManager(
            [
                TimeUnitConversion(target.durations()),
                ALAPScheduleAnalysis(target=target),
                ConstrainedReschedule(target=target),
            ]
        )

        self.assertEqual(qc, pm.run(qc))

    def test_padding_not_working_without_scheduling(self):
        """Test padding fails when un-scheduled DAG is input."""
        qc = QuantumCircuit(1, 1)
        qc.delay(100, 0)
        qc.x(0)
        qc.measure(0, 0)

        with self.assertRaises(TranspilerError):
            PassManager(PadDelay()).run(qc)

    def test_no_pad_very_end_of_circuit(self):
        """Test padding option that inserts no delay at the very end of circuit.

        This circuit will be unchanged after ASAP-schedule/padding.

             ┌────────────────┐┌─┐
        q_0: ┤ Delay(100[dt]) ├┤M├
             └─────┬───┬──────┘└╥┘
        q_1: ──────┤ X ├────────╫─
                   └───┘        ║
        c: 1/═══════════════════╩═
                                0
        """
        qc = QuantumCircuit(2, 1)
        qc.delay(100, 0)
        qc.x(1)
        qc.measure(0, 0)

        durations = InstructionDurations([("x", None, 160), ("measure", None, 1000)], dt=1e-77)

        scheduled = PassManager(
            [
                ASAPScheduleAnalysis(durations),
                PadDelay(fill_very_end=False, durations=durations),
            ]
        ).run(qc)

        self.assertEqual(scheduled, qc)

    @data(ALAPScheduleAnalysis, ASAPScheduleAnalysis)
    def test_respect_target_instruction_constraints(self, schedule_pass):
        """Test if DD pass does not pad delays for qubits that do not support delay instructions.
        See: https://github.com/Qiskit/qiskit-terra/issues/9993
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)

        target = Target(dt=1)
        target.add_instruction(CXGate(), {(1, 2): InstructionProperties(duration=1000)})
        # delays are not supported

        pm = PassManager([schedule_pass(target=target), PadDelay(target=target)])
        scheduled = pm.run(qc)

        self.assertEqual(qc, scheduled)


if __name__ == "__main__":
    unittest.main()
