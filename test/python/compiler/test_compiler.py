# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compiler Test."""

import os
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager, CouplingMap
from qiskit.circuit.library import U1Gate, U2Gate
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.qasm2 import dumps
from test import QiskitTestCase

from ..legacy_cmaps import TOKYO_CMAP


class TestCompiler(QiskitTestCase):
    """Qiskit Compiler Tests."""

    def setUp(self):
        super().setUp()
        self.seed_simulator = 42
        self.backend = BasicSimulator()

    def test_example_multiple_compile(self):
        """Test a toy example compiling multiple circuits.

        Pass if the results are correct.
        """
        backend = BasicSimulator()
        coupling_map = [[0, 1], [0, 2], [1, 2], [3, 2], [3, 4], [4, 2]]

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        bell = QuantumCircuit(qr, cr)
        ghz = QuantumCircuit(qr, cr)
        # Create a GHZ state
        ghz.h(qr[0])
        for i in range(4):
            ghz.cx(qr[i], qr[i + 1])
        # Insert a barrier before measurement
        ghz.barrier()
        # Measure all of the qubits in the standard basis
        for i in range(5):
            ghz.measure(qr[i], cr[i])
        # Create a Bell state
        bell.h(qr[0])
        bell.cx(qr[0], qr[1])
        bell.barrier()
        bell.measure(qr[0], cr[0])
        bell.measure(qr[1], cr[1])
        shots = 2048
        bell_qcirc = transpile(bell, backend=backend)
        ghz_qcirc = transpile(ghz, coupling_map=coupling_map)
        bell_result = backend.run(bell_qcirc, shots=shots, seed_simulator=10).result()
        ghz_result = backend.run(ghz_qcirc, shots=shots, seed_simulator=10).result()

        threshold = 0.05 * shots
        counts_bell = bell_result.get_counts()
        target_bell = {"00000": shots / 2, "00011": shots / 2}
        self.assertDictAlmostEqual(counts_bell, target_bell, threshold)

        counts_ghz = ghz_result.get_counts()
        target_ghz = {"00000": shots / 2, "11111": shots / 2}
        self.assertDictAlmostEqual(counts_ghz, target_ghz, threshold)

    def test_compile_coupling_map(self):
        """Test compile_coupling_map.
        If all correct should return data with the same stats. The circuit may
        be different.
        """
        backend = BasicSimulator()

        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr, cr, name="qccccccc")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        shots = 2048
        coupling_map = [[0, 1], [1, 2]]
        initial_layout = [0, 1, 2]
        qc_b = transpile(qc, coupling_map=coupling_map, initial_layout=initial_layout)
        job = backend.run(qc_b, shots=shots, seed_simulator=88)
        result = job.result()
        qasm_to_check = dumps(qc)
        self.assertEqual(len(qasm_to_check), 172)

        counts = result.get_counts(qc)
        target = {"000": shots / 2, "111": shots / 2}
        threshold = 0.05 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_example_swap_bits(self):
        """Test a toy example swapping a set bit around.

        Uses the mapper. Pass if results are correct.
        """
        backend = BasicSimulator()
        coupling_map = [
            [0, 1],
            [0, 8],
            [1, 2],
            [1, 9],
            [2, 3],
            [2, 10],
            [3, 4],
            [3, 11],
            [4, 5],
            [4, 12],
            [5, 6],
            [5, 13],
            [6, 7],
            [6, 14],
            [7, 15],
            [8, 9],
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            [14, 15],
        ]

        #       ┌───┐             ░ ┌─┐
        # q0_0: ┤ X ├─X───────────░─┤M├───────────────
        #       └───┘ │           ░ └╥┘   ┌─┐
        # q0_1: ──────┼─────X──X──░──╫────┤M├─────────
        #             │     │  │  ░  ║    └╥┘   ┌─┐
        # q0_2: ──────X──X──┼──┼──░──╫─────╫────┤M├───
        #                │  │  │  ░  ║ ┌─┐ ║    └╥┘
        # q1_0: ─────────┼──┼──┼──░──╫─┤M├─╫─────╫────
        #                │  │  │  ░  ║ └╥┘ ║ ┌─┐ ║
        # q1_1: ─────────┼──┼──X──░──╫──╫──╫─┤M├─╫────
        #                │  │     ░  ║  ║  ║ └╥┘ ║ ┌─┐
        # q1_2: ─────────X──X─────░──╫──╫──╫──╫──╫─┤M├
        #                         ░  ║  ║  ║  ║  ║ └╥┘
        # c0: 6/═════════════════════╩══╩══╩══╩══╩══╩═
        #                            0  3  1  4  2  5
        n = 3  # make this at least 3
        qr0 = QuantumRegister(n)
        qr1 = QuantumRegister(n)
        ans = ClassicalRegister(2 * n)
        qc = QuantumCircuit(qr0, qr1, ans)
        # Set the first bit of qr0
        qc.x(qr0[0])
        # Swap the set bit
        qc.swap(qr0[0], qr0[n - 1])
        qc.swap(qr0[n - 1], qr1[n - 1])
        qc.swap(qr1[n - 1], qr0[1])
        qc.swap(qr0[1], qr1[1])
        # Insert a barrier before measurement
        qc.barrier()
        # Measure all of the qubits in the standard basis
        for j in range(n):
            qc.measure(qr0[j], ans[j])
            qc.measure(qr1[j], ans[j + n])
        # First version: no mapping
        result = backend.run(
            transpile(qc, coupling_map=None), shots=1024, seed_simulator=14
        ).result()
        self.assertEqual(result.get_counts(qc), {"010000": 1024})
        # Second version: map to coupling graph
        result = backend.run(
            transpile(qc, coupling_map=coupling_map),
            shots=1024,
            seed_simulator=14,
        ).result()
        self.assertEqual(result.get_counts(qc), {"010000": 1024})

    def test_parallel_compile(self):
        """Trigger parallel routines in compile."""
        qr = QuantumRegister(16)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 15):
            qc.cx(qr[0], qr[k])
        qc.measure(qr[5], cr[0])
        qlist = [qc for k in range(10)]
        backend = GenericBackendV2(num_qubits=20, coupling_map=TOKYO_CMAP, seed=0)
        out = transpile(qlist, backend=backend)
        self.assertEqual(len(out), 10)

    def test_no_conflict_backend_passmanager(self):
        """See: https://github.com/Qiskit/qiskit-terra/issues/5037"""
        backend = BasicSimulator()
        qc = QuantumCircuit(2)
        qc.append(U1Gate(0), [0])
        qc.measure_all()
        job = backend.run(PassManager().run(qc))
        result = job.result().get_counts()
        self.assertEqual(result, {"00": 1024})

    def test_compile_single_qubit(self):
        """Compile a single-qubit circuit in a non-trivial layout"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        layout = {qr[0]: 12}
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        circuit2 = transpile(
            circuit, backend=None, coupling_map=cmap, basis_gates=["u2"], initial_layout=layout
        )

        compiled_instruction = circuit2.data[0]

        self.assertEqual(compiled_instruction.name, "u2")
        self.assertEqual(circuit2.find_bit(compiled_instruction.qubits[0]).index, 12)
        self.assertEqual(compiled_instruction.params, [0, 3.141592653589793])

    def test_compile_pass_manager(self):
        """Test compile with and without an empty pass manager."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.append(U1Gate(3.14), [qr[0]])
        qc.append(U2Gate(3.14, 1.57), [qr[0]])
        qc.barrier(qr)
        qc.measure(qr, cr)
        backend = BasicSimulator()
        qrtrue = transpile(qc, backend, seed_transpiler=8)
        rtrue = backend.run(qrtrue, seed_simulator=42).result()
        qrfalse = PassManager().run(qc)
        rfalse = backend.run(qrfalse, seed_simulator=42).result()
        self.assertEqual(rtrue.get_counts(), rfalse.get_counts())

    def test_mapper_overoptimization(self):
        """Check mapper overoptimization.

        The mapper should not change the semantics of the input.
        An overoptimization introduced issue #81:
        https://github.com/Qiskit/qiskit-terra/issues/81
        """
        #       ┌───┐          ┌─┐
        # q0_0: ┤ X ├──■───────┤M├───────────
        #       ├───┤┌─┴─┐┌───┐└╥┘     ┌─┐
        # q0_1: ┤ Y ├┤ X ├┤ S ├─╫───■──┤M├───
        #       ├───┤└───┘├───┤ ║ ┌─┴─┐└╥┘┌─┐
        # q0_2: ┤ Z ├──■──┤ T ├─╫─┤ X ├─╫─┤M├
        #       └───┘┌─┴─┐├───┤ ║ └┬─┬┘ ║ └╥┘
        # q0_3: ─────┤ X ├┤ H ├─╫──┤M├──╫──╫─
        #            └───┘└───┘ ║  └╥┘  ║  ║
        # c0: 4/════════════════╩═══╩═══╩══╩═
        #                       0   3   1  2
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.x(qr[0])
        circ.y(qr[1])
        circ.z(qr[2])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[2], qr[3])
        circ.s(qr[1])
        circ.t(qr[2])
        circ.h(qr[3])
        circ.cx(qr[1], qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        coupling_map = [[0, 2], [1, 2], [2, 3]]
        shots = 1000

        result1 = self.backend.run(
            transpile(circ, coupling_map=coupling_map, seed_transpiler=8),
            seed_simulator=self.seed_simulator,
            shots=shots,
        )
        count1 = result1.result().get_counts()
        result2 = self.backend.run(
            transpile(
                circ,
                coupling_map=None,
                seed_transpiler=8,
            ),
            seed_simulator=self.seed_simulator,
            shots=shots,
        )
        count2 = result2.result().get_counts()
        self.assertDictAlmostEqual(count1, count2, shots * 0.02)

    def test_grovers_circuit(self):
        """Testing a circuit originated in the Grover algorithm"""
        shots = 1000
        coupling_map = None

        # 6-qubit grovers
        #
        #       ┌───┐┌───┐     ┌───┐┌───┐     ┌───┐     ┌───┐┌───┐      ░ ┌─┐
        # q0_0: ┤ H ├┤ X ├──■──┤ X ├┤ X ├──■──┤ X ├──■──┤ X ├┤ H ├──────░─┤M├───
        #       ├───┤└───┘  │  └───┘└───┘  │  ├───┤  │  ├───┤├───┤      ░ └╥┘┌─┐
        # q0_1: ┤ H ├──■────┼─────────■────┼──┤ X ├──■──┤ X ├┤ H ├──────░──╫─┤M├
        #       ├───┤  │  ┌─┴─┐       │  ┌─┴─┐└───┘  │  └───┘└───┘      ░  ║ └╥┘
        # q0_2: ┤ X ├──┼──┤ X ├──■────┼──┤ X ├───────┼──────────────────░──╫──╫─
        #       ├───┤┌─┴─┐└───┘  │  ┌─┴─┐└───┘       │                  ░  ║  ║
        # q0_3: ┤ X ├┤ X ├───────■──┤ X ├────────────┼──────────────────░──╫──╫─
        #       └───┘└───┘     ┌─┴─┐├───┤┌───┐     ┌─┴─┐┌───┐┌───┐┌───┐ ░  ║  ║
        # q0_4: ───────────────┤ X ├┤ X ├┤ H ├─────┤ X ├┤ H ├┤ X ├┤ H ├─░──╫──╫─
        #                      └───┘└───┘└───┘     └───┘└───┘└───┘└───┘ ░  ║  ║
        # q0_5: ────────────────────────────────────────────────────────░──╫──╫─
        #                                                               ░  ║  ║
        # c0: 2/═══════════════════════════════════════════════════════════╩══╩═
        #                                                                  0  1
        qr = QuantumRegister(6)
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr, name="grovers")

        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.x(qr[2])
        circuit.x(qr[3])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.x(qr[0])
        circuit.cx(qr[1], qr[3])
        circuit.ccx(qr[2], qr[3], qr[4])
        circuit.cx(qr[1], qr[3])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.x(qr[4])
        circuit.h(qr[4])
        circuit.ccx(qr[0], qr[1], qr[4])
        circuit.h(qr[4])
        circuit.x(qr[0])
        circuit.x(qr[1])
        circuit.x(qr[4])
        circuit.h(qr[0])
        circuit.h(qr[1])
        circuit.h(qr[4])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])

        result = self.backend.run(
            transpile(
                circuit,
                self.backend,
                coupling_map=coupling_map,
            ),
            seed_simulator=self.seed_simulator,
            shots=shots,
        )
        counts = result.result().get_counts()

        expected_probs = {"00": 0.64, "01": 0.117, "10": 0.113, "11": 0.13}

        target = {key: shots * val for key, val in expected_probs.items()}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_math_domain_error(self):
        """Check for floating point errors.

        The math library operates over floats and introduces floating point
        errors that should be avoided.
        See: https://github.com/Qiskit/qiskit-terra/issues/111
        """
        #       ┌───┐┌───┐     ┌─┐
        # q0_0: ┤ Y ├┤ X ├─────┤M├─────────────────────
        #       └───┘└─┬─┘     └╥┘               ┌─┐
        # q0_1: ───────■────────╫─────────────■──┤M├───
        #       ┌───┐┌───┐┌───┐ ║ ┌───┐┌───┐┌─┴─┐└╥┘┌─┐
        # q0_2: ┤ Z ├┤ H ├┤ Y ├─╫─┤ T ├┤ Z ├┤ X ├─╫─┤M├
        #       └┬─┬┘└───┘└───┘ ║ └───┘└───┘└───┘ ║ └╥┘
        # q0_3: ─┤M├────────────╫─────────────────╫──╫─
        #        └╥┘            ║                 ║  ║
        # c0: 4/══╩═════════════╩═════════════════╩══╩═
        #         3             0                 1  2
        qr = QuantumRegister(4)
        cr = ClassicalRegister(4)
        circ = QuantumCircuit(qr, cr)
        circ.y(qr[0])
        circ.z(qr[2])
        circ.h(qr[2])
        circ.cx(qr[1], qr[0])
        circ.y(qr[2])
        circ.t(qr[2])
        circ.z(qr[2])
        circ.cx(qr[1], qr[2])
        circ.measure(qr[0], cr[0])
        circ.measure(qr[1], cr[1])
        circ.measure(qr[2], cr[2])
        circ.measure(qr[3], cr[3])

        coupling_map = [[0, 2], [1, 2], [2, 3]]
        shots = 2000
        job = self.backend.run(
            transpile(
                circ,
                coupling_map=coupling_map,
            ),
            seed_simulator=self.seed_simulator,
            shots=shots,
        )
        counts = job.result().get_counts()
        target = {"0001": shots / 2, "0101": shots / 2}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_random_parameter_circuit(self):
        """Run a circuit with randomly generated parameters."""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "random_n5_d5.qasm"))
        coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])
        coupling_map.make_symmetric()
        shots = 1024
        job = self.backend.run(
            transpile(circ, coupling_map=coupling_map, seed_transpiler=42),
            shots=shots,
            seed_simulator=self.seed_simulator,
        )
        counts = job.result().get_counts()
        expected_probs = {
            "00000": 0.079239867254200971,
            "00001": 0.032859032998526903,
            "00010": 0.10752610993531816,
            "00011": 0.018818532050952699,
            "00100": 0.054830807251011054,
            "00101": 0.0034141983951965164,
            "00110": 0.041649309748902276,
            "00111": 0.039967731207338125,
            "01000": 0.10516937819949743,
            "01001": 0.026635620063700002,
            "01010": 0.0053475143548793866,
            "01011": 0.01940513314416064,
            "01100": 0.0044028405481225047,
            "01101": 0.057524760052126644,
            "01110": 0.010795354134597078,
            "01111": 0.026491296821535528,
            "10000": 0.094827455395274859,
            "10001": 0.0008373965072688836,
            "10010": 0.029082297894094441,
            "10011": 0.012386622870598416,
            "10100": 0.018739140061148799,
            "10101": 0.01367656456536896,
            "10110": 0.039184170706009248,
            "10111": 0.062339335178438288,
            "11000": 0.00293674365989009,
            "11001": 0.012848433960739968,
            "11010": 0.018472497159499782,
            "11011": 0.0088903691234912003,
            "11100": 0.031305389080034329,
            "11101": 0.0004788556283690458,
            "11110": 0.002232419390471667,
            "11111": 0.017684822659235985,
        }
        target = {key: shots * val for key, val in expected_probs.items()}
        threshold = 0.04 * shots
        self.assertDictAlmostEqual(counts, target, threshold)


if __name__ == "__main__":
    unittest.main(verbosity=2)
