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

"""Test dynamical decoupling insertion pass."""

import unittest
import numpy as np
from numpy import pi
from ddt import ddt

from qiskit.circuit import QuantumCircuit, Delay, Measure, Reset, Parameter
from qiskit.circuit.library import XGate, YGate, RXGate, UGate, CXGate, HGate
from qiskit.quantum_info import Operator
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes import (
    ASAPScheduleAnalysis,
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
)
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase


@ddt
class TestPadDynamicalDecoupling(QiskitTestCase):
    """Tests PadDynamicalDecoupling pass."""

    def setUp(self):
        """Circuits to test DD on.

             ┌───┐
        q_0: ┤ H ├──■────────────
             └───┘┌─┴─┐
        q_1: ─────┤ X ├──■───────
                  └───┘┌─┴─┐
        q_2: ──────────┤ X ├──■──
                       └───┘┌─┴─┐
        q_3: ───────────────┤ X ├
                            └───┘

                  ┌──────────┐
        q_0: ──■──┤ U(π,0,π) ├──────────■──
             ┌─┴─┐└──────────┘        ┌─┴─┐
        q_1: ┤ X ├─────■───────────■──┤ X ├
             └───┘   ┌─┴─┐    ┌─┐┌─┴─┐└───┘
        q_2: ────────┤ X ├────┤M├┤ X ├─────
                     └───┘    └╥┘└───┘
        c: 1/══════════════════╩═══════════
                               0
        """
        super().setUp()

        self.ghz4 = QuantumCircuit(4)
        self.ghz4.h(0)
        self.ghz4.cx(0, 1)
        self.ghz4.cx(1, 2)
        self.ghz4.cx(2, 3)

        self.midmeas = QuantumCircuit(3, 1)
        self.midmeas.cx(0, 1)
        self.midmeas.cx(1, 2)
        self.midmeas.u(pi, 0, pi, 0)
        self.midmeas.measure(2, 0)
        self.midmeas.cx(1, 2)
        self.midmeas.cx(0, 1)

        self.durations = InstructionDurations(
            [
                ("h", 0, 50),
                ("cx", [0, 1], 700),
                ("cx", [1, 2], 200),
                ("cx", [2, 3], 300),
                ("x", None, 50),
                ("y", None, 50),
                ("u", None, 100),
                ("rx", None, 100),
                ("measure", None, 1000),
                ("reset", None, 1500),
            ],
            dt=1e-9,
        )

    def test_insert_dd_ghz(self):
        """Test DD gates are inserted in correct spots.

                   ┌───┐            ┌────────────────┐      ┌───┐      »
        q_0: ──────┤ H ├─────────■──┤ Delay(100[dt]) ├──────┤ X ├──────»
             ┌─────┴───┴─────┐ ┌─┴─┐└────────────────┘┌─────┴───┴─────┐»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├────────■─────────┤ Delay(50[dt]) ├»
             ├───────────────┴┐└───┘      ┌─┴─┐       └───────────────┘»
        q_2: ┤ Delay(750[dt]) ├───────────┤ X ├───────────────■────────»
             ├────────────────┤           └───┘             ┌─┴─┐      »
        q_3: ┤ Delay(950[dt]) ├─────────────────────────────┤ X ├──────»
             └────────────────┘                             └───┘      »
        «     ┌────────────────┐      ┌───┐       ┌────────────────┐
        «q_0: ┤ Delay(200[dt]) ├──────┤ X ├───────┤ Delay(100[dt]) ├─────────────────
        «     └─────┬───┬──────┘┌─────┴───┴──────┐└─────┬───┬──────┘┌───────────────┐
        «q_1: ──────┤ X ├───────┤ Delay(100[dt]) ├──────┤ X ├───────┤ Delay(50[dt]) ├
        «           └───┘       └────────────────┘      └───┘       └───────────────┘
        «q_2: ───────────────────────────────────────────────────────────────────────
        «
        «q_3: ───────────────────────────────────────────────────────────────────────
        «
        """
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(50), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(100), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(50), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_with_target(self):
        """Test DD gates are inserted in correct spots.

                   ┌───┐            ┌────────────────┐      ┌───┐      »
        q_0: ──────┤ H ├─────────■──┤ Delay(100[dt]) ├──────┤ X ├──────»
             ┌─────┴───┴─────┐ ┌─┴─┐└────────────────┘┌─────┴───┴─────┐»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├────────■─────────┤ Delay(50[dt]) ├»
             ├───────────────┴┐└───┘      ┌─┴─┐       └───────────────┘»
        q_2: ┤ Delay(750[dt]) ├───────────┤ X ├───────────────■────────»
             ├────────────────┤           └───┘             ┌─┴─┐      »
        q_3: ┤ Delay(950[dt]) ├─────────────────────────────┤ X ├──────»
             └────────────────┘                             └───┘      »
        «     ┌────────────────┐      ┌───┐       ┌────────────────┐
        «q_0: ┤ Delay(200[dt]) ├──────┤ X ├───────┤ Delay(100[dt]) ├─────────────────
        «     └─────┬───┬──────┘┌─────┴───┴──────┐└─────┬───┬──────┘┌───────────────┐
        «q_1: ──────┤ X ├───────┤ Delay(100[dt]) ├──────┤ X ├───────┤ Delay(50[dt]) ├
        «           └───┘       └────────────────┘      └───┘       └───────────────┘
        «q_2: ───────────────────────────────────────────────────────────────────────
        «
        «q_3: ───────────────────────────────────────────────────────────────────────
        «
        """
        target = Target(num_qubits=4, dt=1)
        target.add_instruction(HGate(), {(0,): InstructionProperties(duration=50)})
        target.add_instruction(
            CXGate(),
            {
                (0, 1): InstructionProperties(duration=700),
                (1, 2): InstructionProperties(duration=200),
                (2, 3): InstructionProperties(duration=300),
            },
        )
        target.add_instruction(
            XGate(), {(x,): InstructionProperties(duration=50) for x in range(4)}
        )
        target.add_instruction(
            YGate(), {(x,): InstructionProperties(duration=50) for x in range(4)}
        )
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),
            {(x,): InstructionProperties(duration=100) for x in range(4)},
        )
        target.add_instruction(
            RXGate(Parameter("theta")),
            {(x,): InstructionProperties(duration=100) for x in range(4)},
        )
        target.add_instruction(
            Measure(), {(x,): InstructionProperties(duration=1000) for x in range(4)}
        )
        target.add_instruction(
            Reset(), {(x,): InstructionProperties(duration=1500) for x in range(4)}
        )
        target.add_instruction(Delay(Parameter("t")), {(x,): None for x in range(4)})
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(target=target),
                PadDynamicalDecoupling(target=target, dd_sequence=dd_sequence),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(50), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(100), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(50), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_one_qubit(self):
        """Test DD gates are inserted on only one qubit.

                      ┌───┐            ┌────────────────┐      ┌───┐       »
           q_0: ──────┤ H ├─────────■──┤ Delay(100[dt]) ├──────┤ X ├───────»
                ┌─────┴───┴─────┐ ┌─┴─┐└────────────────┘┌─────┴───┴──────┐»
           q_1: ┤ Delay(50[dt]) ├─┤ X ├────────■─────────┤ Delay(300[dt]) ├»
                ├───────────────┴┐└───┘      ┌─┴─┐       └────────────────┘»
           q_2: ┤ Delay(750[dt]) ├───────────┤ X ├───────────────■─────────»
                ├────────────────┤           └───┘             ┌─┴─┐       »
           q_3: ┤ Delay(950[dt]) ├─────────────────────────────┤ X ├───────»
                └────────────────┘                             └───┘       »
        meas: 4/═══════════════════════════════════════════════════════════»
                                                                           »
        «        ┌────────────────┐┌───┐┌────────────────┐ ░ ┌─┐
        «   q_0: ┤ Delay(200[dt]) ├┤ X ├┤ Delay(100[dt]) ├─░─┤M├─────────
        «        └────────────────┘└───┘└────────────────┘ ░ └╥┘┌─┐
        «   q_1: ──────────────────────────────────────────░──╫─┤M├──────
        «                                                  ░  ║ └╥┘┌─┐
        «   q_2: ──────────────────────────────────────────░──╫──╫─┤M├───
        «                                                  ░  ║  ║ └╥┘┌─┐
        «   q_3: ──────────────────────────────────────────░──╫──╫──╫─┤M├
        «                                                  ░  ║  ║  ║ └╥┘
        «meas: 4/═════════════════════════════════════════════╩══╩══╩══╩═
        «                                                     0  1  2  3
        """
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, qubits=[0]),
            ]
        )

        ghz4_dd = pm.run(self.ghz4.measure_all(inplace=False))

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(300), [1])

        expected.measure_all()

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_everywhere(self):
        """Test DD gates even on initial idle spots.

                   ┌───┐            ┌────────────────┐┌───┐┌────────────────┐┌───┐»
        q_0: ──────┤ H ├─────────■──┤ Delay(100[dt]) ├┤ Y ├┤ Delay(200[dt]) ├┤ Y ├»
             ┌─────┴───┴─────┐ ┌─┴─┐└────────────────┘└───┘└────────────────┘└───┘»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├───────────────────────────────────────────■──»
             ├───────────────┴┐├───┤┌────────────────┐┌───┐┌────────────────┐┌─┴─┐»
        q_2: ┤ Delay(162[dt]) ├┤ Y ├┤ Delay(326[dt]) ├┤ Y ├┤ Delay(162[dt]) ├┤ X ├»
             ├────────────────┤├───┤├────────────────┤├───┤├────────────────┤└───┘»
        q_3: ┤ Delay(212[dt]) ├┤ Y ├┤ Delay(426[dt]) ├┤ Y ├┤ Delay(212[dt]) ├─────»
             └────────────────┘└───┘└────────────────┘└───┘└────────────────┘     »
        «     ┌────────────────┐
        «q_0: ┤ Delay(100[dt]) ├─────────────────────────────────────────────
        «     ├───────────────┬┘┌───┐┌────────────────┐┌───┐┌───────────────┐
        «q_1: ┤ Delay(50[dt]) ├─┤ Y ├┤ Delay(100[dt]) ├┤ Y ├┤ Delay(50[dt]) ├
        «     └───────────────┘ └───┘└────────────────┘└───┘└───────────────┘
        «q_2: ────────■──────────────────────────────────────────────────────
        «           ┌─┴─┐
        «q_3: ──────┤ X ├────────────────────────────────────────────────────
        «           └───┘
        """
        dd_sequence = [YGate(), YGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, skip_reset_qubits=False),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)

        expected = expected.compose(Delay(162), [2], front=True)
        expected = expected.compose(YGate(), [2], front=True)
        expected = expected.compose(Delay(326), [2], front=True)
        expected = expected.compose(YGate(), [2], front=True)
        expected = expected.compose(Delay(162), [2], front=True)

        expected = expected.compose(Delay(212), [3], front=True)
        expected = expected.compose(YGate(), [3], front=True)
        expected = expected.compose(Delay(426), [3], front=True)
        expected = expected.compose(YGate(), [3], front=True)
        expected = expected.compose(Delay(212), [3], front=True)

        expected = expected.compose(Delay(100), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(200), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(100), [0])

        expected = expected.compose(Delay(50), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(100), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(50), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_insert_dd_ghz_xy4(self):
        """Test XY4 sequence of DD gates.

                   ┌───┐            ┌───────────────┐      ┌───┐      ┌───────────────┐»
        q_0: ──────┤ H ├─────────■──┤ Delay(37[dt]) ├──────┤ X ├──────┤ Delay(75[dt]) ├»
             ┌─────┴───┴─────┐ ┌─┴─┐└───────────────┘┌─────┴───┴─────┐└─────┬───┬─────┘»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├────────■────────┤ Delay(12[dt]) ├──────┤ X ├──────»
             ├───────────────┴┐└───┘      ┌─┴─┐      └───────────────┘      └───┘      »
        q_2: ┤ Delay(750[dt]) ├───────────┤ X ├──────────────■─────────────────────────»
             ├────────────────┤           └───┘            ┌─┴─┐                       »
        q_3: ┤ Delay(950[dt]) ├────────────────────────────┤ X ├───────────────────────»
             └────────────────┘                            └───┘                       »
        «           ┌───┐      ┌───────────────┐      ┌───┐      ┌───────────────┐»
        «q_0: ──────┤ Y ├──────┤ Delay(76[dt]) ├──────┤ X ├──────┤ Delay(75[dt]) ├»
        «     ┌─────┴───┴─────┐└─────┬───┬─────┘┌─────┴───┴─────┐└─────┬───┬─────┘»
        «q_1: ┤ Delay(25[dt]) ├──────┤ Y ├──────┤ Delay(26[dt]) ├──────┤ X ├──────»
        «     └───────────────┘      └───┘      └───────────────┘      └───┘      »
        «q_2: ────────────────────────────────────────────────────────────────────»
        «                                                                         »
        «q_3: ────────────────────────────────────────────────────────────────────»
        «                                                                         »
        «           ┌───┐      ┌───────────────┐
        «q_0: ──────┤ Y ├──────┤ Delay(37[dt]) ├─────────────────
        «     ┌─────┴───┴─────┐└─────┬───┬─────┘┌───────────────┐
        «q_1: ┤ Delay(25[dt]) ├──────┤ Y ├──────┤ Delay(12[dt]) ├
        «     └───────────────┘      └───┘      └───────────────┘
        «q_2: ───────────────────────────────────────────────────
        «
        «q_3: ───────────────────────────────────────────────────
        """
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(37), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(75), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(76), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(75), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(37), [0])

        expected = expected.compose(Delay(12), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(25), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(26), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(25), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(12), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_insert_midmeas_hahn_alap(self):
        """Test a single X gate as Hahn echo can absorb in the downstream circuit.

        global phase: 3π/2
                               ┌────────────────┐       ┌───┐       ┌────────────────┐»
        q_0: ────────■─────────┤ Delay(625[dt]) ├───────┤ X ├───────┤ Delay(625[dt]) ├»
                   ┌─┴─┐       └────────────────┘┌──────┴───┴──────┐└────────────────┘»
        q_1: ──────┤ X ├───────────────■─────────┤ Delay(1000[dt]) ├────────■─────────»
             ┌─────┴───┴──────┐      ┌─┴─┐       └───────┬─┬───────┘      ┌─┴─┐       »
        q_2: ┤ Delay(700[dt]) ├──────┤ X ├───────────────┤M├──────────────┤ X ├───────»
             └────────────────┘      └───┘               └╥┘              └───┘       »
        c: 1/═════════════════════════════════════════════╩═══════════════════════════»
                                                          0                           »
        «     ┌───────────────┐
        «q_0: ┤ U(0,π/2,-π/2) ├───■──
        «     └───────────────┘ ┌─┴─┐
        «q_1: ──────────────────┤ X ├
        «     ┌────────────────┐└───┘
        «q_2: ┤ Delay(700[dt]) ├─────
        «     └────────────────┘
        «c: 1/═══════════════════════
        """
        dd_sequence = [XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        midmeas_dd = pm.run(self.midmeas)

        combined_u = UGate(0, 0, 0)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.delay(625, 0)
        expected.x(0)
        expected.delay(625, 0)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)
        expected.global_phase = pi

        self.assertEqual(midmeas_dd, expected)
        # check the absorption into U was done correctly
        self.assertEqual(Operator(combined_u), Operator(XGate()) & Operator(XGate()))

    def test_insert_midmeas_hahn_asap(self):
        """Test a single X gate as Hahn echo can absorb in the upstream circuit.

                               ┌──────────────────┐ ┌────────────────┐┌─────────┐»
        q_0: ────────■─────────┤ U(3π/4,-π/2,π/2) ├─┤ Delay(600[dt]) ├┤ Rx(π/4) ├»
                   ┌─┴─┐       └──────────────────┘┌┴────────────────┤└─────────┘»
        q_1: ──────┤ X ├────────────────■──────────┤ Delay(1000[dt]) ├─────■─────»
             ┌─────┴───┴──────┐       ┌─┴─┐        └───────┬─┬───────┘   ┌─┴─┐   »
        q_2: ┤ Delay(700[dt]) ├───────┤ X ├────────────────┤M├───────────┤ X ├───»
             └────────────────┘       └───┘                └╥┘           └───┘   »
        c: 1/═══════════════════════════════════════════════╩════════════════════»
                                                            0                    »
        «     ┌────────────────┐
        «q_0: ┤ Delay(600[dt]) ├──■──
        «     └────────────────┘┌─┴─┐
        «q_1: ──────────────────┤ X ├
        «     ┌────────────────┐└───┘
        «q_2: ┤ Delay(700[dt]) ├─────
        «     └────────────────┘
        «c: 1/═══════════════════════
        «
        """
        dd_sequence = [RXGate(pi / 4)]
        pm = PassManager(
            [
                ASAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        midmeas_dd = pm.run(self.midmeas)

        combined_u = UGate(3 * pi / 4, -pi / 2, pi / 2)

        expected = QuantumCircuit(3, 1)
        expected.cx(0, 1)
        expected.compose(combined_u, [0], inplace=True)
        expected.delay(600, 0)
        expected.rx(pi / 4, 0)
        expected.delay(600, 0)
        expected.delay(700, 2)
        expected.cx(1, 2)
        expected.delay(1000, 1)
        expected.measure(2, 0)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.delay(700, 2)

        self.assertEqual(midmeas_dd, expected)
        # check the absorption into U was done correctly
        self.assertTrue(
            Operator(XGate()).equiv(
                Operator(UGate(3 * pi / 4, -pi / 2, pi / 2)) & Operator(RXGate(pi / 4))
            )
        )

    def test_insert_ghz_uhrig(self):
        """Test custom spacing (following Uhrig DD [1]).

        [1] Uhrig, G. "Keeping a quantum bit alive by optimized π-pulse sequences."
        Physical Review Letters 98.10 (2007): 100504.

                   ┌───┐            ┌──────────────┐      ┌───┐       ┌──────────────┐┌───┐»
        q_0: ──────┤ H ├─────────■──┤ Delay(3[dt]) ├──────┤ X ├───────┤ Delay(8[dt]) ├┤ X ├»
             ┌─────┴───┴─────┐ ┌─┴─┐└──────────────┘┌─────┴───┴──────┐└──────────────┘└───┘»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├───────■────────┤ Delay(300[dt]) ├─────────────────────»
             ├───────────────┴┐└───┘     ┌─┴─┐      └────────────────┘                     »
        q_2: ┤ Delay(750[dt]) ├──────────┤ X ├──────────────■──────────────────────────────»
             ├────────────────┤          └───┘            ┌─┴─┐                            »
        q_3: ┤ Delay(950[dt]) ├───────────────────────────┤ X ├────────────────────────────»
             └────────────────┘                           └───┘                            »
        «     ┌───────────────┐┌───┐┌───────────────┐┌───┐┌───────────────┐┌───┐┌───────────────┐»
        «q_0: ┤ Delay(13[dt]) ├┤ X ├┤ Delay(16[dt]) ├┤ X ├┤ Delay(20[dt]) ├┤ X ├┤ Delay(16[dt]) ├»
        «     └───────────────┘└───┘└───────────────┘└───┘└───────────────┘└───┘└───────────────┘»
        «q_1: ───────────────────────────────────────────────────────────────────────────────────»
        «                                                                                        »
        «q_2: ───────────────────────────────────────────────────────────────────────────────────»
        «                                                                                        »
        «q_3: ───────────────────────────────────────────────────────────────────────────────────»
        «                                                                                        »
        «     ┌───┐┌───────────────┐┌───┐┌──────────────┐┌───┐┌──────────────┐
        «q_0: ┤ X ├┤ Delay(13[dt]) ├┤ X ├┤ Delay(8[dt]) ├┤ X ├┤ Delay(3[dt]) ├
        «     └───┘└───────────────┘└───┘└──────────────┘└───┘└──────────────┘
        «q_1: ────────────────────────────────────────────────────────────────
        «
        «q_2: ────────────────────────────────────────────────────────────────
        «
        «q_3: ────────────────────────────────────────────────────────────────
        «
        """
        n = 8
        dd_sequence = [XGate()] * n

        # uhrig specifies the location of the k'th pulse
        def uhrig(k):
            return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2

        # convert that to spacing between pulses (whatever finite duration pulses have)
        spacing = []
        for k in range(n):
            spacing.append(uhrig(k) - sum(spacing))
        spacing.append(1 - sum(spacing))

        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, qubits=[0], spacing=spacing),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(3), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(8), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(13), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(16), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(20), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(16), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(13), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(8), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(3), [0])

        expected = expected.compose(Delay(300), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_asymmetric_xy4_in_t2(self):
        """Test insertion of XY4 sequence with unbalanced spacing.

        global phase: π
             ┌───┐┌───┐┌────────────────┐┌───┐┌────────────────┐┌───┐┌────────────────┐»
        q_0: ┤ H ├┤ X ├┤ Delay(450[dt]) ├┤ Y ├┤ Delay(450[dt]) ├┤ X ├┤ Delay(450[dt]) ├»
             └───┘└───┘└────────────────┘└───┘└────────────────┘└───┘└────────────────┘»
        «     ┌───┐┌────────────────┐┌───┐
        «q_0: ┤ Y ├┤ Delay(450[dt]) ├┤ H ├
        «     └───┘└────────────────┘└───┘
        """
        dd_sequence = [XGate(), YGate()] * 2
        spacing = [0] + [1 / 4] * 4
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, spacing=spacing),
            ]
        )

        t2 = QuantumCircuit(1)
        t2.h(0)
        t2.delay(2000, 0)
        t2.h(0)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.x(0)
        expected.delay(450, 0)
        expected.y(0)
        expected.delay(450, 0)
        expected.x(0)
        expected.delay(450, 0)
        expected.y(0)
        expected.delay(450, 0)
        expected.h(0)
        expected.global_phase = pi

        t2_dd = pm.run(t2)

        self.assertEqual(t2_dd, expected)
        # check global phase is correct
        self.assertEqual(Operator(t2), Operator(expected))

    def test_dd_after_reset(self):
        """Test skip_reset_qubits option works.

                  ┌─────────────────┐┌───┐┌────────────────┐┌───┐┌─────────────────┐»
        q_0: ─|0>─┤ Delay(1000[dt]) ├┤ H ├┤ Delay(190[dt]) ├┤ X ├┤ Delay(1710[dt]) ├»
                  └─────────────────┘└───┘└────────────────┘└───┘└─────────────────┘»
        «     ┌───┐┌───┐
        «q_0: ┤ X ├┤ H ├
        «     └───┘└───┘
        """
        dd_sequence = [XGate(), XGate()]
        spacing = [0.1, 0.9]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(
                    self.durations, dd_sequence, spacing=spacing, skip_reset_qubits=True
                ),
            ]
        )

        t2 = QuantumCircuit(1)
        t2.reset(0)
        t2.delay(1000)
        t2.h(0)
        t2.delay(2000, 0)
        t2.h(0)

        expected = QuantumCircuit(1)
        expected.reset(0)
        expected.delay(1000)
        expected.h(0)
        expected.delay(190, 0)
        expected.x(0)
        expected.delay(1710, 0)
        expected.x(0)
        expected.h(0)

        t2_dd = pm.run(t2)

        self.assertEqual(t2_dd, expected)

    def test_insert_dd_bad_sequence(self):
        """Test DD raises when non-identity sequence is inserted."""
        dd_sequence = [XGate(), YGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        with self.assertRaises(TranspilerError):
            pm.run(self.ghz4)

    def test_insert_dd_ghz_xy4_with_alignment(self):
        """Test DD with pulse alignment constraints.

                   ┌───┐            ┌───────────────┐      ┌───┐      ┌───────────────┐»
        q_0: ──────┤ H ├─────────■──┤ Delay(40[dt]) ├──────┤ X ├──────┤ Delay(70[dt]) ├»
             ┌─────┴───┴─────┐ ┌─┴─┐└───────────────┘┌─────┴───┴─────┐└─────┬───┬─────┘»
        q_1: ┤ Delay(50[dt]) ├─┤ X ├────────■────────┤ Delay(20[dt]) ├──────┤ X ├──────»
             ├───────────────┴┐└───┘      ┌─┴─┐      └───────────────┘      └───┘      »
        q_2: ┤ Delay(750[dt]) ├───────────┤ X ├──────────────■─────────────────────────»
             ├────────────────┤           └───┘            ┌─┴─┐                       »
        q_3: ┤ Delay(950[dt]) ├────────────────────────────┤ X ├───────────────────────»
             └────────────────┘                            └───┘                       »
        «           ┌───┐      ┌───────────────┐      ┌───┐      ┌───────────────┐»
        «q_0: ──────┤ Y ├──────┤ Delay(70[dt]) ├──────┤ X ├──────┤ Delay(70[dt]) ├»
        «     ┌─────┴───┴─────┐└─────┬───┬─────┘┌─────┴───┴─────┐└─────┬───┬─────┘»
        «q_1: ┤ Delay(20[dt]) ├──────┤ Y ├──────┤ Delay(20[dt]) ├──────┤ X ├──────»
        «     └───────────────┘      └───┘      └───────────────┘      └───┘      »
        «q_2: ────────────────────────────────────────────────────────────────────»
        «                                                                         »
        «q_3: ────────────────────────────────────────────────────────────────────»
        «                                                                         »
        «           ┌───┐      ┌───────────────┐
        «q_0: ──────┤ Y ├──────┤ Delay(50[dt]) ├─────────────────
        «     ┌─────┴───┴─────┐└─────┬───┬─────┘┌───────────────┐
        «q_1: ┤ Delay(20[dt]) ├──────┤ Y ├──────┤ Delay(20[dt]) ├
        «     └───────────────┘      └───┘      └───────────────┘
        «q_2: ───────────────────────────────────────────────────
        «
        «q_3: ───────────────────────────────────────────────────
        «
        """
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(
                    self.durations,
                    dd_sequence,
                    pulse_alignment=10,
                    extra_slack_distribution="edges",
                ),
            ]
        )

        ghz4_dd = pm.run(self.ghz4)

        expected = self.ghz4.copy()
        expected = expected.compose(Delay(50), [1], front=True)
        expected = expected.compose(Delay(750), [2], front=True)
        expected = expected.compose(Delay(950), [3], front=True)

        expected = expected.compose(Delay(40), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(70), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(70), [0])
        expected = expected.compose(XGate(), [0])
        expected = expected.compose(Delay(70), [0])
        expected = expected.compose(YGate(), [0])
        expected = expected.compose(Delay(50), [0])

        expected = expected.compose(Delay(20), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(20), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(20), [1])
        expected = expected.compose(XGate(), [1])
        expected = expected.compose(Delay(20), [1])
        expected = expected.compose(YGate(), [1])
        expected = expected.compose(Delay(20), [1])

        self.assertEqual(ghz4_dd, expected)

    def test_dd_can_sequentially_called(self):
        """Test if sequentially called DD pass can output the same circuit.

        This test verifies:
        - if global phase is properly propagated from the previous padding node.
        - if node_start_time property is properly updated for new dag circuit.
        """
        dd_sequence = [XGate(), YGate(), XGate(), YGate()]

        pm1 = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, qubits=[0]),
                PadDynamicalDecoupling(self.durations, dd_sequence, qubits=[1]),
            ]
        )
        circ1 = pm1.run(self.ghz4)

        pm2 = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, qubits=[0, 1]),
            ]
        )
        circ2 = pm2.run(self.ghz4)

        self.assertEqual(circ1, circ2)

    def test_respect_target_instruction_constraints(self):
        """Test if DD pass does not pad delays for qubits that do not support delay instructions
        and does not insert DD gates for qubits that do not support necessary gates.
        See: https://github.com/Qiskit/qiskit-terra/issues/9993
        """
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        target = Target(dt=1)
        # Y is partially supported (not supported on qubit 2)
        target.add_instruction(
            XGate(), {(q,): InstructionProperties(duration=100) for q in range(2)}
        )
        target.add_instruction(
            CXGate(),
            {
                (0, 1): InstructionProperties(duration=1000),
                (1, 2): InstructionProperties(duration=1000),
            },
        )
        # delays are not supported

        # No DD instructions nor delays are padded due to no delay support in the target
        pm_xx = PassManager(
            [
                ALAPScheduleAnalysis(target=target),
                PadDynamicalDecoupling(dd_sequence=[XGate(), XGate()], target=target),
            ]
        )
        scheduled = pm_xx.run(qc)
        self.assertEqual(qc, scheduled)

        # Fails since Y is not supported in the target
        with self.assertRaises(TranspilerError):
            PassManager(
                [
                    ALAPScheduleAnalysis(target=target),
                    PadDynamicalDecoupling(
                        dd_sequence=[XGate(), YGate(), XGate(), YGate()], target=target
                    ),
                ]
            )

        # Add delay support to the target
        target.add_instruction(Delay(Parameter("t")), {(q,): None for q in range(3)})
        # No error but no DD on qubit 2 (just delay is padded) since X is not supported on it
        scheduled = pm_xx.run(qc)

        expected = QuantumCircuit(3)
        expected.delay(1000, [2])
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.delay(200, [0])
        expected.x([0])
        expected.delay(400, [0])
        expected.x([0])
        expected.delay(200, [0])
        self.assertEqual(expected, scheduled)

    def test_parameterized_global_phase(self):
        """Test parameterized global phase in DD circuit.
        See:https://github.com/Qiskit/qiskit-terra/issues/10569
        """
        dd_sequence = [XGate(), YGate()] * 2
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.delay(1700, 0)
        qc.y(0)
        qc.global_phase = Parameter("a")
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence),
            ]
        )

        self.assertEqual(qc.global_phase + np.pi, pm.run(qc).global_phase)

    def test_misalignment_at_boundaries(self):
        """Test the correct error message is raised for misalignments at In/Out nodes."""
        # a circuit where the previous node is DAGInNode, and the next DAGOutNode
        circuit = QuantumCircuit(1)
        circuit.delay(101)

        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(self.durations),
                PadDynamicalDecoupling(self.durations, dd_sequence, pulse_alignment=2),
            ]
        )

        with self.assertRaises(TranspilerError):
            _ = pm.run(circuit)


if __name__ == "__main__":
    unittest.main()
