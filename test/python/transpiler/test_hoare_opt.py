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

"""Test the HoareOptimizer pass"""

import unittest
from numpy import pi

from qiskit.utils import optionals
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import HoareOptimizer
from qiskit.converters import circuit_to_dag
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, RZGate, CSwapGate, SwapGate
from qiskit.dagcircuit import DAGOpNode
from qiskit.quantum_info import Statevector
from qiskit.qasm3 import loads as qasm3_loads
from test import QiskitTestCase


@unittest.skipUnless(optionals.HAS_Z3, "z3-solver needs to be installed to run these tests")
class TestHoareOptimizer(QiskitTestCase):
    """Test the HoareOptimizer pass"""

    def test_phasegate_removal(self):
        """Should remove the phase on a classical state,
        but not on a superposition state.
        """

        #      ┌───┐
        # q_0: ┤ Z ├──────
        #      ├───┤┌───┐
        # q_1:─┤ H ├┤ Z ├─
        #      └───┘└───┘
        circuit = QuantumCircuit(3)
        circuit.z(0)
        circuit.h(1)
        circuit.z(1)

        # q_0: ───────────
        #      ┌───┐┌───┐
        # q_1:─┤ H ├┤ Z ├─
        #      └───┘└───┘
        expected = QuantumCircuit(3)
        expected.h(1)
        expected.z(1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_cswap_removal(self):
        """Should remove Fredkin gates because the optimizer
        can deduce the targets are in the same state
        """

        #      ┌───┐┌───┐     ┌───┐     ┌───┐          ┌───┐
        # q_0: ┤ X ├┤ X ├──■──┤ X ├──■──┤ X ├──■────■──┤ X ├─────────────────────────────────
        #      └───┘└─┬─┘┌─┴─┐└─┬─┘  │  └─┬─┘┌─┴─┐  │  └─┬─┘
        # q_1: ───────┼──┤ X ├──■────┼────┼──┤ X ├──┼────■───■──■──■──■─────■─────■──────────
        #             │  └─┬─┘     ┌─┴─┐  │  └─┬─┘┌─┴─┐  │   │  │  │  │     │     │
        # q_2: ───────┼────┼───────┤ X ├──■────┼──┤ X ├──■───┼──┼──┼──┼──■──┼──■──┼──■──■──■─
        #      ┌───┐  │    │       └─┬─┘       │  └─┬─┘      │  │  │  │  │  │  │  │  │  │  │
        # q_3: ┤ H ├──■────┼─────────┼─────────┼────┼────────┼──┼──X──X──┼──┼──X──┼──┼──X──┼─
        #      ├───┤       │         │         │    │        │  │  │  │  │  │  │  │  │  │  │
        # q_4: ┤ H ├───────■─────────┼─────────┼────┼────────┼──┼──┼──X──┼──X──┼──┼──X──┼──X─
        #      ├───┤                 │         │    │        │  │  │     │  │  │  │  │  │  │
        # q_5: ┤ H ├─────────────────■─────────┼────┼────────┼──┼──┼─────┼──X──┼──X──┼──X──┼─
        #      ├───┤                           │    │        │  │  │     │     │  │  │     │
        # q_6: ┤ H ├───────────────────────────■────■────────┼──┼──┼─────┼─────┼──X──┼─────X─
        #      └───┘                                         │  │  │     │     │     │
        # q_7: ──────────────────────────────────────────────X──┼──┼─────X─────┼─────┼───────
        #                                                    │  │  │     │     │     │
        # q_8: ──────────────────────────────────────────────X──X──┼─────┼─────X─────┼───────
        #                                                       │  │     │           │
        # q_9: ─────────────────────────────────────────────────X──X─────X───────────X───────
        circuit = QuantumCircuit(10)
        # prep
        circuit.x(0)
        circuit.h(3)
        circuit.h(4)
        circuit.h(5)
        circuit.h(6)
        # find first non-zero bit of reg(3-6), store position in reg(1-2)
        circuit.cx(3, 0)
        circuit.ccx(0, 4, 1)
        circuit.cx(1, 0)
        circuit.ccx(0, 5, 2)
        circuit.cx(2, 0)
        circuit.ccx(0, 6, 1)
        circuit.ccx(0, 6, 2)
        circuit.ccx(1, 2, 0)
        # shift circuit
        circuit.cswap(1, 7, 8)
        circuit.cswap(1, 8, 9)
        circuit.cswap(1, 9, 3)
        circuit.cswap(1, 3, 4)
        circuit.cswap(1, 4, 5)
        circuit.cswap(1, 5, 6)
        circuit.cswap(2, 7, 9)
        circuit.cswap(2, 8, 3)
        circuit.cswap(2, 9, 4)
        circuit.cswap(2, 3, 5)
        circuit.cswap(2, 4, 6)

        #      ┌───┐┌───┐     ┌───┐     ┌───┐          ┌───┐
        # q_0: ┤ X ├┤ X ├──■──┤ X ├──■──┤ X ├──■────■──┤ X ├───────────────
        #      └───┘└─┬─┘┌─┴─┐└─┬─┘  │  └─┬─┘┌─┴─┐  │  └─┬─┘
        # q_1: ───────┼──┤ X ├──■────┼────┼──┤ X ├──┼────■───■──■──■───────
        #             │  └─┬─┘     ┌─┴─┐  │  └─┬─┘┌─┴─┐  │   │  │  │
        # q_2: ───────┼────┼───────┤ X ├──■────┼──┤ X ├──■───┼──┼──┼──■──■─
        #      ┌───┐  │    │       └─┬─┘       │  └─┬─┘      │  │  │  │  │
        # q_3: ┤ H ├──■────┼─────────┼─────────┼────┼────────X──┼──┼──X──┼─
        #      ├───┤       │         │         │    │        │  │  │  │  │
        # q_4: ┤ H ├───────■─────────┼─────────┼────┼────────X──X──┼──┼──X─
        #      ├───┤                 │         │    │           │  │  │  │
        # q_5: ┤ H ├─────────────────■─────────┼────┼───────────X──X──X──┼─
        #      ├───┤                           │    │              │     │
        # q_6: ┤ H ├───────────────────────────■────■──────────────X─────X─
        #      └───┘
        # q_7: ────────────────────────────────────────────────────────────
        #
        # q_8: ────────────────────────────────────────────────────────────
        #
        # q_9: ────────────────────────────────────────────────────────────
        expected = QuantumCircuit(10)
        # prep
        expected.x(0)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        expected.h(6)
        # find first non-zero bit of reg(3-6), store position in reg(1-2)
        expected.cx(3, 0)
        expected.ccx(0, 4, 1)
        expected.cx(1, 0)
        expected.ccx(0, 5, 2)
        expected.cx(2, 0)
        expected.ccx(0, 6, 1)
        expected.ccx(0, 6, 2)
        expected.ccx(1, 2, 0)
        # optimized shift circuit
        expected.cswap(1, 3, 4)
        expected.cswap(1, 4, 5)
        expected.cswap(1, 5, 6)
        expected.cswap(2, 3, 5)
        expected.cswap(2, 4, 6)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_lnn_cnot_removal(self):
        """Should remove some cnots from swaps introduced
        because of linear nearest architecture. Only uses
        single-gate optimization techniques.
        """

        #      ┌───┐     ┌───┐                                                       »
        # q_0: ┤ H ├──■──┤ X ├──■────────────────────────────────────────────────────»
        #      └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌───┐                                        »
        # q_1: ─────┤ X ├──■──┤ X ├──■──┤ X ├──■──────────────────────────────────■──»
        #           └───┘     └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌───┐               ┌───┐┌─┴─┐»
        # q_2: ────────────────────┤ X ├──■──┤ X ├──■──┤ X ├──■─────────■──┤ X ├┤ X ├»
        #                          └───┘     └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌─┴─┐└─┬─┘└───┘»
        # q_3: ───────────────────────────────────┤ X ├──■──┤ X ├──■──┤ X ├──■───────»
        #                                         └───┘     └───┘┌─┴─┐└───┘          »
        # q_4: ──────────────────────────────────────────────────┤ X ├───────────────»
        #                                                        └───┘               »
        # «               ┌───┐
        # «q_0: ───────■──┤ X ├
        # «     ┌───┐┌─┴─┐└─┬─┘
        # «q_1: ┤ X ├┤ X ├──■──
        # «     └─┬─┘└───┘
        # «q_2: ──■────────────
        # «
        # «q_3: ───────────────
        # «
        # «q_4: ───────────────
        circuit = QuantumCircuit(5)
        circuit.h(0)
        for i in range(3):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
            circuit.cx(i, i + 1)
        circuit.cx(3, 4)
        for i in range(3, 0, -1):
            circuit.cx(i - 1, i)
            circuit.cx(i, i - 1)

        #      ┌───┐     ┌───┐                                   ┌───┐
        # q_0: ┤ H ├──■──┤ X ├───────────────────────────────────┤ X ├
        #      └───┘┌─┴─┐└─┬─┘     ┌───┐                    ┌───┐└─┬─┘
        # q_1: ─────┤ X ├──■────■──┤ X ├────────────────────┤ X ├──■──
        #           └───┘     ┌─┴─┐└─┬─┘     ┌───┐     ┌───┐└─┬─┘
        # q_2: ───────────────┤ X ├──■────■──┤ X ├─────┤ X ├──■───────
        #                     └───┘     ┌─┴─┐└─┬─┘     └─┬─┘
        # q_3: ─────────────────────────┤ X ├──■────■────■────────────
        #                               └───┘     ┌─┴─┐
        # q_4: ───────────────────────────────────┤ X ├───────────────
        #                                         └───┘
        expected = QuantumCircuit(5)
        expected.h(0)
        for i in range(3):
            expected.cx(i, i + 1)
            expected.cx(i + 1, i)
        expected.cx(3, 4)
        for i in range(3, 0, -1):
            expected.cx(i, i - 1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=0)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_lnncnot_advanced_removal(self):
        """Should remove all cnots from swaps introduced
        because of linear nearest architecture. This time
        using multi-gate optimization techniques.
        """

        #      ┌───┐     ┌───┐                                                       »
        # q_0: ┤ H ├──■──┤ X ├──■────────────────────────────────────────────────────»
        #      └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌───┐                                        »
        # q_1: ─────┤ X ├──■──┤ X ├──■──┤ X ├──■──────────────────────────────────■──»
        #           └───┘     └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌───┐               ┌───┐┌─┴─┐»
        # q_2: ────────────────────┤ X ├──■──┤ X ├──■──┤ X ├──■─────────■──┤ X ├┤ X ├»
        #                          └───┘     └───┘┌─┴─┐└─┬─┘┌─┴─┐     ┌─┴─┐└─┬─┘└───┘»
        # q_3: ───────────────────────────────────┤ X ├──■──┤ X ├──■──┤ X ├──■───────»
        #                                         └───┘     └───┘┌─┴─┐└───┘          »
        # q_4: ──────────────────────────────────────────────────┤ X ├───────────────»
        #                                                        └───┘               »
        # «               ┌───┐
        # «q_0: ───────■──┤ X ├
        # «     ┌───┐┌─┴─┐└─┬─┘
        # «q_1: ┤ X ├┤ X ├──■──
        # «     └─┬─┘└───┘
        # «q_2: ──■────────────
        # «
        # «q_3: ───────────────
        # «
        # «q_4: ───────────────
        circuit = QuantumCircuit(5)
        circuit.h(0)
        for i in range(3):
            circuit.cx(i, i + 1)
            circuit.cx(i + 1, i)
            circuit.cx(i, i + 1)
        circuit.cx(3, 4)
        for i in range(3, 0, -1):
            circuit.cx(i - 1, i)
            circuit.cx(i, i - 1)

        #      ┌───┐
        # q_0: ┤ H ├──■─────────────────
        #      └───┘┌─┴─┐
        # q_1: ─────┤ X ├──■────────────
        #           └───┘┌─┴─┐
        # q_2: ──────────┤ X ├──■───────
        #                └───┘┌─┴─┐
        # q_3: ───────────────┤ X ├──■──
        #                     └───┘┌─┴─┐
        # q_4: ────────────────────┤ X ├
        #                          └───┘
        expected = QuantumCircuit(5)
        expected.h(0)
        for i in range(4):
            expected.cx(i, i + 1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=6)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_successive_identity_removal(self):
        """Should remove a successive pair of H gates applying
        on the same qubit.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.h(0)
        circuit.h(0)

        expected = QuantumCircuit(1)
        expected.h(0)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=4)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    @unittest.skipUnless(optionals.HAS_QASM3_IMPORT, "qiskit-qasm3-import is required")
    def test_issue_14969_regression(self):
        """Cache eviction must not crash or change the state in issue 14969."""

        bug_qasm = """OPENQASM 3.0;
include \"stdgates.inc\";
gate rzx(p0) _gate_q_0, _gate_q_1 {
  h _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  rz(p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  h _gate_q_1;
}
gate cu1(p0) _gate_q_0, _gate_q_1 {
  p(0.5*p0) _gate_q_0;
  cx _gate_q_0, _gate_q_1;
  p((-0.5)*p0) _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  p(0.5*p0) _gate_q_1;
}
gate cu1_o0(p0) _gate_q_0, _gate_q_1 {
  x _gate_q_0;
  cu1(p0) _gate_q_0, _gate_q_1;
  x _gate_q_0;
}
gate cz_o0 _gate_q_0, _gate_q_1 {
  x _gate_q_0;
  cz _gate_q_0, _gate_q_1;
  x _gate_q_0;
}
bit[1] c;
qubit[4] q;
cu1_o0(6) q[3], q[2];
cy q[2], q[1];
rzx(2) q[1], q[3];
cx q[1], q[0];
rx(2) q[0];
cx q[1], q[0];
cy q[0], q[3];
sdg q[0];
x q[0];
cz_o0 q[3], q[0];
sdg q[0];
sdg q[0];
cz_o0 q[0], q[1];
cz q[3], q[2];
s q[2];
swap q[3], q[2];
x q[2];
rz(5) q[2];
id q[2];
y q[2];
id q[2];
c[0] = measure q[2];
"""

        circuit = qasm3_loads(bug_qasm)
        expected = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        for size in (2, 3, 5, 10, 20):
            with self.subTest(size=size):
                result = PassManager(HoareOptimizer(size=size)).run(circuit)
                self.assertEqual(result.data[-1], circuit.data[-1])
                self.assertTrue(
                    expected.equiv(
                        Statevector.from_instruction(
                            result.remove_final_measurements(inplace=False)
                        )
                    )
                )

    def test_issue_14969_shared_cache_entries(self):
        """Shared cache entries in the October 2026 reproducer can be evicted safely."""
        # Reduced HHL example reported by karineek:
        # https://github.com/Qiskit/qiskit/issues/14969#issuecomment-6081522377
        for reverse in (False, True):
            system = QuantumRegister(1, "sys")
            phase = QuantumRegister(10, "phase_0")
            ancilla = QuantumRegister(1, "ancilla")
            classical = ClassicalRegister(1, "c")
            circuit = QuantumCircuit(system, phase, ancilla, classical)
            circuit.x(system[0])
            for i in sorted(range(1, 10), reverse=reverse):
                circuit.h(phase[i])
            for i in range(1, 10):
                circuit.cp(0.0, phase[i], system[0])
            circuit.append(QuantumCircuit(10, name="func1").to_gate(), phase)
            for i in range(10):
                circuit.cry(pi if i < 9 else 1.2, phase[i], ancilla[0])
            circuit.append(QuantumCircuit(10, name="func2").to_gate(), phase)
            circuit.cp(0.0, phase[0], system[0])
            circuit.measure(system[0], classical[0])

            expected = Statevector.from_instruction(
                circuit.remove_final_measurements(inplace=False)
            )
            for size in (2, 3, 5, 10, 20):
                with self.subTest(size=size, reverse=reverse):
                    result = PassManager(HoareOptimizer(size=size)).run(circuit)
                    self.assertEqual(result.data[-1], circuit.data[-1])
                    self.assertTrue(
                        expected.equiv(
                            Statevector.from_instruction(
                                result.remove_final_measurements(inplace=False)
                            )
                        )
                    )

    def test_targetsuccessive_identity_removal(self):
        """Should remove pair of controlled target successive
        which are the inverse of each other, if they can be
        identified to be executed as a unit (either both or none).
        """

        #      ┌───┐     ┌───┐┌───┐
        # q_0: ┤ H ├──■──┤ X ├┤ X ├──■──
        #      ├───┤  │  └─┬─┘└───┘  │
        # q_1: ┤ H ├──■────■─────────■──
        #      ├───┤┌─┴─┐          ┌─┴─┐
        # q_2: ┤ H ├┤ X ├──────────┤ X ├
        #      └───┘└───┘          └───┘
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.ccx(0, 1, 2)
        circuit.cx(1, 0)
        circuit.x(0)
        circuit.ccx(0, 1, 2)

        #      ┌───┐┌───┐┌───┐
        # q_0: ┤ H ├┤ X ├┤ X ├
        #      ├───┤└─┬─┘└───┘
        # q_1: ┤ H ├──■───────
        #      ├───┤
        # q_2: ┤ H ├──────────
        #      └───┘
        expected = QuantumCircuit(3)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.cx(1, 0)
        expected.x(0)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=4)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_targetsuccessive_identity_advanced_removal(self):
        """Should remove target successive identity gates
        with DIFFERENT sets of control qubits.
        In this case CCCX(4,5,6,7) & CCX(5,6,7).
        """

        #      ┌───┐┌───┐                                                            »
        # q_0: ┤ H ├┤ X ├───────■─────────────────────────────■───────────────────■──»
        #      ├───┤└─┬─┘       │                             │                   │  »
        # q_1: ┤ H ├──■─────────■─────────────────────────────■───────────────────■──»
        #      ├───┤┌───┐       │  ┌───┐                      │                   │  »
        # q_2: ┤ H ├┤ X ├───────┼──┤ X ├──■──────────────■────┼───────────────────┼──»
        #      ├───┤└─┬─┘     ┌─┴─┐└─┬─┘  │              │  ┌─┴─┐               ┌─┴─┐»
        # q_3: ┤ H ├──■────■──┤ X ├──■────■──────────────■──┤ X ├──■─────────■──┤ X ├»
        #      ├───┤┌───┐  │  └───┘       │  ┌───┐       │  └───┘  │         │  └───┘»
        # q_4: ┤ H ├┤ X ├──┼──────────────┼──┤ X ├──■────┼─────────┼─────────┼───────»
        #      ├───┤└─┬─┘┌─┴─┐          ┌─┴─┐└─┬─┘  │  ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     »
        # q_5: ┤ H ├──■──┤ X ├──────────┤ X ├──■────■──┤ X ├─────┤ X ├──■──┤ X ├─────»
        #      └───┘     └───┘          └───┘     ┌─┴─┐└───┘     └───┘┌─┴─┐├───┤     »
        # q_6: ───────────────────────────────────┤ X ├───────────────┤ X ├┤ X ├─────»
        #                                         └───┘               └───┘└───┘     »
        # q_7: ──────────────────────────────────────────────────────────────────────»
        #                                                                            »
        # «                              ┌───┐┌───┐                                   »
        # «q_0: ──────────────────────■──┤ X ├┤ X ├──■─────────────────────────────■──»
        # «                           │  └─┬─┘└─┬─┘  │                             │  »
        # «q_1: ──────────────────────■────■────■────■─────────────────────────────■──»
        # «                    ┌───┐  │         │    │  ┌───┐                      │  »
        # «q_2: ──■─────────■──┤ X ├──┼─────────┼────┼──┤ X ├──■──────────────■────┼──»
        # «       │         │  └─┬─┘┌─┴─┐       │  ┌─┴─┐└─┬─┘  │              │  ┌─┴─┐»
        # «q_3: ──■─────────■────■──┤ X ├───────┼──┤ X ├──■────■──────────────■──┤ X ├»
        # «       │  ┌───┐  │       └───┘       │  └───┘  │    │  ┌───┐       │  └───┘»
        # «q_4: ──┼──┤ X ├──┼───────────────────┼─────────┼────┼──┤ X ├──■────┼───────»
        # «     ┌─┴─┐└─┬─┘┌─┴─┐                 │         │  ┌─┴─┐└─┬─┘  │  ┌─┴─┐     »
        # «q_5: ┤ X ├──■──┤ X ├─────────────────┼─────────┼──┤ X ├──■────■──┤ X ├─────»
        # «     └───┘     └───┘                 │         │  └───┘  │    │  └───┘     »
        # «q_6: ────────────────────────────────■─────────■─────────■────■────────────»
        # «                                                            ┌─┴─┐          »
        # «q_7: ───────────────────────────────────────────────────────┤ X ├──────────»
        # «                                                            └───┘          »
        # «
        # «q_0: ───────────────
        # «
        # «q_1: ───────────────
        # «          ┌───┐
        # «q_2: ─────┤ X ├─────
        # «          └─┬─┘
        # «q_3: ──■────■───────
        # «       │  ┌───┐
        # «q_4: ──┼──┤ X ├─────
        # «     ┌─┴─┐└─┬─┘
        # «q_5: ┤ X ├──■────■──
        # «     └───┘       │
        # «q_6: ────────────■──
        # «               ┌─┴─┐
        # «q_7: ──────────┤ X ├
        # «               └───┘
        circuit = QuantumCircuit(8)
        circuit.h(0)
        circuit.h(1)
        circuit.h(2)
        circuit.h(3)
        circuit.h(4)
        circuit.h(5)
        for i in range(3):
            circuit.cx(i * 2 + 1, i * 2)
        circuit.cx(3, 5)
        for i in range(2):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
            circuit.cx(i * 2 + 3, i * 2 + 2)
        circuit.ccx(4, 5, 6)
        for i in range(1, -1, -1):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(3, 5)
        circuit.cx(5, 6)
        circuit.cx(3, 5)
        circuit.x(6)
        for i in range(2):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        for i in range(1, -1, -1):
            circuit.cx(i * 2 + 3, i * 2 + 2)
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(1, 0)
        circuit.ccx(6, 1, 0)
        circuit.ccx(0, 1, 3)
        circuit.ccx(6, 3, 2)
        circuit.ccx(2, 3, 5)
        circuit.ccx(6, 5, 4)
        circuit.append(XGate().control(3, annotated=False), [4, 5, 6, 7], [])
        for i in range(1, -1, -1):
            circuit.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        circuit.cx(3, 5)
        for i in range(1, 3):
            circuit.cx(i * 2 + 1, i * 2)
        circuit.ccx(5, 6, 7)

        #      ┌───┐┌───┐                                                            »
        # q_0: ┤ H ├┤ X ├───────■─────────────────────────────■───────────────────■──»
        #      ├───┤└─┬─┘       │                             │                   │  »
        # q_1: ┤ H ├──■─────────■─────────────────────────────■───────────────────■──»
        #      ├───┤┌───┐       │  ┌───┐                      │                   │  »
        # q_2: ┤ H ├┤ X ├───────┼──┤ X ├──■──────────────■────┼───────────────────┼──»
        #      ├───┤└─┬─┘     ┌─┴─┐└─┬─┘  │              │  ┌─┴─┐               ┌─┴─┐»
        # q_3: ┤ H ├──■────■──┤ X ├──■────■──────────────■──┤ X ├──■─────────■──┤ X ├»
        #      ├───┤┌───┐  │  └───┘       │  ┌───┐       │  └───┘  │         │  └───┘»
        # q_4: ┤ H ├┤ X ├──┼──────────────┼──┤ X ├──■────┼─────────┼─────────┼───────»
        #      ├───┤└─┬─┘┌─┴─┐          ┌─┴─┐└─┬─┘  │  ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     »
        # q_5: ┤ H ├──■──┤ X ├──────────┤ X ├──■────■──┤ X ├─────┤ X ├──■──┤ X ├─────»
        #      └───┘     └───┘          └───┘     ┌─┴─┐└───┘     └───┘┌─┴─┐├───┤     »
        # q_6: ───────────────────────────────────┤ X ├───────────────┤ X ├┤ X ├─────»
        #                                         └───┘               └───┘└───┘     »
        # q_7: ──────────────────────────────────────────────────────────────────────»
        #                                                                            »
        # «                              ┌───┐┌───┐                                   »
        # «q_0: ──────────────────────■──┤ X ├┤ X ├──■────────────────────────■───────»
        # «                           │  └─┬─┘└─┬─┘  │                        │       »
        # «q_1: ──────────────────────■────■────■────■────────────────────────■───────»
        # «                    ┌───┐  │         │    │  ┌───┐                 │       »
        # «q_2: ──■─────────■──┤ X ├──┼─────────┼────┼──┤ X ├──■─────────■────┼───────»
        # «       │         │  └─┬─┘┌─┴─┐       │  ┌─┴─┐└─┬─┘  │         │  ┌─┴─┐     »
        # «q_3: ──■─────────■────■──┤ X ├───────┼──┤ X ├──■────■─────────■──┤ X ├──■──»
        # «       │  ┌───┐  │       └───┘       │  └───┘  │    │  ┌───┐  │  └───┘  │  »
        # «q_4: ──┼──┤ X ├──┼───────────────────┼─────────┼────┼──┤ X ├──┼─────────┼──»
        # «     ┌─┴─┐└─┬─┘┌─┴─┐                 │         │  ┌─┴─┐└─┬─┘┌─┴─┐     ┌─┴─┐»
        # «q_5: ┤ X ├──■──┤ X ├─────────────────┼─────────┼──┤ X ├──■──┤ X ├─────┤ X ├»
        # «     └───┘     └───┘                 │         │  └───┘  │  └───┘     └───┘»
        # «q_6: ────────────────────────────────■─────────■─────────■─────────────────»
        # «                                                                           »
        # «q_7: ──────────────────────────────────────────────────────────────────────»
        # «                                                                           »
        # «
        # «q_0: ─────
        # «
        # «q_1: ─────
        # «     ┌───┐
        # «q_2: ┤ X ├
        # «     └─┬─┘
        # «q_3: ──■──
        # «     ┌───┐
        # «q_4: ┤ X ├
        # «     └─┬─┘
        # «q_5: ──■──
        # «
        # «q_6: ─────
        # «
        # «q_7: ─────
        # «
        expected = QuantumCircuit(8)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        for i in range(3):
            expected.cx(i * 2 + 1, i * 2)
        expected.cx(3, 5)
        for i in range(2):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
            expected.cx(i * 2 + 3, i * 2 + 2)
        expected.ccx(4, 5, 6)
        for i in range(1, -1, -1):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(3, 5)
        expected.cx(5, 6)
        expected.cx(3, 5)
        expected.x(6)
        for i in range(2):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        for i in range(1, -1, -1):
            expected.cx(i * 2 + 3, i * 2 + 2)
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(1, 0)
        expected.ccx(6, 1, 0)
        expected.ccx(0, 1, 3)
        expected.ccx(6, 3, 2)
        expected.ccx(2, 3, 5)
        expected.ccx(6, 5, 4)
        for i in range(1, -1, -1):
            expected.ccx(i * 2, i * 2 + 1, i * 2 + 3)
        expected.cx(3, 5)
        for i in range(1, 3):
            expected.cx(i * 2 + 1, i * 2)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_control_removal(self):
        """Should replace CX by X."""

        #      ┌───┐
        # q_0: ┤ X ├──■──
        #      └───┘┌─┴─┐
        # q_1: ─────┤ X ├
        #           └───┘
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.cx(0, 1)

        #      ┌───┐
        # q_0: ┤ X ├
        #      ├───┤
        # q_1: ┤ X ├
        #      └───┘
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.x(1)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

        # Should replace CZ by Z
        #
        #      ┌───┐   ┌───┐
        # q_0: ┤ H ├─■─┤ H ├
        #      ├───┤ │ └───┘
        # q_1: ┤ X ├─■──────
        #      └───┘
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.x(1)
        circuit.cz(0, 1)
        circuit.h(0)

        #      ┌───┐┌───┐┌───┐
        # q_0: ┤ H ├┤ Z ├┤ H ├
        #      ├───┤└───┘└───┘
        # q_1: ┤ X ├──────────
        #      └───┘
        expected = QuantumCircuit(2)
        expected.h(0)
        expected.x(1)
        expected.z(0)
        expected.h(0)

        stv = Statevector.from_label("0" * circuit.num_qubits)
        self.assertEqual(stv & circuit, stv & expected)

        pass_ = HoareOptimizer(size=5)
        result = pass_.run(circuit_to_dag(circuit))

        self.assertEqual(result, circuit_to_dag(expected))

    def test_is_identity(self):
        """The is_identity function determines whether a pair of gates
        forms the identity, when ignoring control qubits.
        """
        seq = [DAGOpNode(op=XGate().control()), DAGOpNode(op=XGate().control(2, annotated=False))]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

        seq = [
            DAGOpNode(op=RZGate(-pi / 2).control(annotated=False)),
            DAGOpNode(op=RZGate(pi / 2).control(2, annotated=False)),
        ]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

        seq = [DAGOpNode(op=CSwapGate()), DAGOpNode(op=SwapGate())]
        self.assertTrue(HoareOptimizer()._is_identity(seq))

    def test_multiple_pass(self):
        """Verify that multiple pass can be run
        with the same Hoare instance.
        """

        #      ┌───┐┌───┐
        # q_0:─┤ H ├┤ Z ├─
        #      ├───┤└───┘
        # q_1: ┤ Z ├──────
        #      └───┘
        circuit1 = QuantumCircuit(2)
        circuit1.z(0)
        circuit1.h(1)
        circuit1.z(1)

        circuit2 = QuantumCircuit(2)
        circuit2.z(1)
        circuit2.h(0)
        circuit2.z(0)

        #      ┌───┐┌───┐
        # q_0:─┤ H ├┤ Z ├─
        #      └───┘└───┘
        # q_1: ───────────
        expected = QuantumCircuit(2)
        expected.h(0)
        expected.z(0)

        pass_ = HoareOptimizer()

        pass_.run(circuit_to_dag(circuit1))
        result2 = pass_.run(circuit_to_dag(circuit2))

        self.assertEqual(result2, circuit_to_dag(expected))

    def test_remove_control_then_identity(self):
        """This first simplifies a gate by removing its control, then removes the
        simplified gate by canceling it with another gate.
        See: https://github.com/Qiskit/qiskit-terra/issues/13079
        """
        #      ┌───┐┌───┐┌───┐
        # q_0: ┤ X ├┤ X ├┤ X ├
        #      └─┬─┘└───┘└─┬─┘
        # q_1: ──■─────────┼──
        #      ┌───┐       │
        # q_2: ┤ X ├───────■──
        #      └───┘
        circuit = QuantumCircuit(3)
        circuit.cx(1, 0)
        circuit.x(2)
        circuit.x(0)
        circuit.cx(2, 0)

        simplified = HoareOptimizer()(circuit)

        # The CX(1, 0) gate is removed as the control qubit q_1 is initially 0.
        # The CX(2, 0) gate is first replaced by X(0) gate as the control qubit q_2 is at 1,
        # then the two X(0) gates are removed.
        #
        # q_0: ─────
        #
        # q_1: ─────
        #      ┌───┐
        # q_2: ┤ X ├
        #      └───┘
        expected = QuantumCircuit(3)
        expected.x(2)

        self.assertEqual(simplified, expected)


if __name__ == "__main__":
    unittest.main()
