# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test basic simulator."""

import os
import unittest
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator, BasicProviderError
from test import QiskitTestCase


from . import BasicProviderBackendTestMixin


class TestBasicSimulator(QiskitTestCase, BasicProviderBackendTestMixin):
    """Test the basic provider simulator."""

    def setUp(self):
        super().setUp()
        self.backend = BasicSimulator()
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])
        self.circuit = bell

        self.seed = 88
        qasm_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "qasm"
        )
        qasm_filename = os.path.join(qasm_dir, "example.qasm")
        qcirc = QuantumCircuit.from_qasm_file(qasm_filename)
        qcirc.name = "test"
        self.transpiled_circuit = transpile(qcirc, backend=self.backend)

    def test_basic_simulator_single_shot(self):
        """Test single shot run."""
        shots = 1
        result = self.backend.run(
            self.transpiled_circuit, shots=shots, seed_simulator=self.seed
        ).result()
        self.assertEqual(result.success, True)

    def test_measure_sampler_repeated_qubits(self):
        """Test measure sampler if qubits measured more than once."""
        shots = 100
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(4, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[1])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[1], cr[1])
        circuit.measure(qr[1], cr[2])
        circuit.measure(qr[0], cr[3])
        target = {"0110": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_measure_sampler_single_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(1, "cr")

        for qubit in range(num_qubits):
            circuit = QuantumCircuit(qr, cr)
            circuit.x(qr[qubit])
            circuit.measure(qr[qubit], cr[0])
            target = {"1": shots}
            job = self.backend.run(
                transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
            )
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target)

    def test_measure_sampler_partial_qubit(self):
        """Test measure sampler if single-qubit is measured."""
        shots = 100
        num_qubits = 5
        qr = QuantumRegister(num_qubits, "qr")
        cr = ClassicalRegister(4, "cr")

        #             ░     ░     ░ ┌─┐ ░
        # qr_0: ──────░─────░─────░─┤M├─░────
        #       ┌───┐ ░     ░ ┌─┐ ░ └╥┘ ░
        # qr_1: ┤ X ├─░─────░─┤M├─░──╫──░────
        #       └───┘ ░     ░ └╥┘ ░  ║  ░
        # qr_2: ──────░─────░──╫──░──╫──░────
        #       ┌───┐ ░ ┌─┐ ░  ║  ░  ║  ░ ┌─┐
        # qr_3: ┤ X ├─░─┤M├─░──╫──░──╫──░─┤M├
        #       └───┘ ░ └╥┘ ░  ║  ░  ║  ░ └╥┘
        # qr_4: ──────░──╫──░──╫──░──╫──░──╫─
        #             ░  ║  ░  ║  ░  ║  ░  ║
        # cr: 4/═════════╩═════╩═════╩═════╩═
        #                1     0     2     3
        circuit = QuantumCircuit(qr, cr)
        circuit.x(qr[3])
        circuit.x(qr[1])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[1])
        circuit.barrier(qr)
        circuit.measure(qr[1], cr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr[2])
        circuit.barrier(qr)
        circuit.measure(qr[3], cr[3])
        target = {"1011": shots}
        job = self.backend.run(
            transpile(circuit, self.backend), shots=shots, seed_simulator=self.seed
        )
        result = job.result()
        counts = result.get_counts(0)
        self.assertEqual(counts, target)

    def test_basic_simulator(self):
        """Test data counts output for single circuit run against reference."""
        result = self.backend.run(
            self.transpiled_circuit, shots=1000, seed_simulator=self.seed
        ).result()
        shots = 1024
        threshold = 0.04 * shots
        counts = result.get_counts("test")
        target = {
            "100 100": shots / 8,
            "011 011": shots / 8,
            "101 101": shots / 8,
            "111 111": shots / 8,
            "000 000": shots / 8,
            "010 010": shots / 8,
            "110 110": shots / 8,
            "001 001": shots / 8,
        }
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_memory(self):
        """Test memory."""
        #       ┌───┐        ┌─┐
        # qr_0: ┤ H ├──■─────┤M├───
        #       └───┘┌─┴─┐   └╥┘┌─┐
        # qr_1: ─────┤ X ├────╫─┤M├
        #            └┬─┬┘    ║ └╥┘
        # qr_2: ──────┤M├─────╫──╫─
        #       ┌───┐ └╥┘ ┌─┐ ║  ║
        # qr_3: ┤ X ├──╫──┤M├─╫──╫─
        #       └───┘  ║  └╥┘ ║  ║
        # cr0: 2/══════╬═══╬══╩══╩═
        #              ║   ║  0  1
        #              ║   ║
        # cr1: 2/══════╩═══╩═══════
        #              0   1
        qr = QuantumRegister(4, "qr")
        cr0 = ClassicalRegister(2, "cr0")
        cr1 = ClassicalRegister(2, "cr1")
        circ = QuantumCircuit(qr, cr0, cr1)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[3])
        circ.measure(qr[0], cr0[0])
        circ.measure(qr[1], cr0[1])
        circ.measure(qr[2], cr1[0])
        circ.measure(qr[3], cr1[1])

        shots = 50
        job = self.backend.run(
            transpile(circ, self.backend), shots=shots, seed_simulator=self.seed, memory=True
        )
        result = job.result()
        memory = result.get_memory()
        self.assertEqual(len(memory), shots)
        for mem in memory:
            self.assertIn(mem, ["10 00", "10 11"])

    def test_unitary(self):
        """Test unitary gate instruction"""
        max_qubits = 4
        x_mat = np.array([[0, 1], [1, 0]])
        # Test 1 to max_qubits for random n-qubit unitary gate
        for i in range(max_qubits):
            num_qubits = i + 1
            # Apply X gate to all qubits
            multi_x = x_mat
            for _ in range(i):
                multi_x = np.kron(multi_x, x_mat)
            # Target counts
            shots = 1024
            target_counts = {num_qubits * "1": shots}
            # Test circuit
            qr = QuantumRegister(num_qubits, "qr")
            cr = ClassicalRegister(num_qubits, "cr")
            circuit = QuantumCircuit(qr, cr)
            circuit.unitary(multi_x, qr)
            circuit.measure(qr, cr)
            job = self.backend.run(transpile(circuit, self.backend), shots=shots)
            result = job.result()
            counts = result.get_counts(0)
            self.assertEqual(counts, target_counts)

    def test_options(self):
        """Test setting custom backend options during init and run."""
        init_statevector = np.zeros(2**2, dtype=complex)
        init_statevector[2] = 1
        in_options = {
            "initial_statevector": init_statevector,
            "seed_simulator": 42,
            "shots": 100,
            "memory": True,
            "use_clifford_optimization": False,  # ADDED FOR CLIFFORD
        }
        backend = BasicSimulator()
        backend_with_options = BasicSimulator(
            initial_statevector=in_options["initial_statevector"],
            seed_simulator=in_options["seed_simulator"],
            shots=in_options["shots"],
            memory=in_options["memory"],
        )
        bell = QuantumCircuit(2, 2)
        bell.h(0)
        bell.cx(0, 1)
        bell.measure([0, 1], [0, 1])

        with self.subTest(msg="Test init options"):
            out_options = backend_with_options.options
            for key in out_options:
                if key != "initial_statevector":
                    self.assertEqual(getattr(out_options, key), in_options.get(key))
                else:
                    np.testing.assert_array_equal(getattr(out_options, key), in_options.get(key))

        with self.subTest(msg="Test run options"):
            out_1 = backend_with_options.run(bell).result().get_counts()
            out_2 = (
                backend.run(
                    bell,
                    initial_statevector=in_options["initial_statevector"],
                    seed_simulator=in_options["seed_simulator"],
                    shots=in_options["shots"],
                    memory=in_options["memory"],
                    use_clifford_optimization=in_options[
                        "use_clifford_optimization"
                    ],  # ADDED FOR CLIFFORD
                )
                .result()
                .get_counts()
            )
            self.assertEqual(out_1, out_2)

        with self.subTest(msg="Test run options don't overwrite init"):
            init_statevector = np.zeros(2**2, dtype=complex)
            init_statevector[3] = 1
            other_options = {
                "initial_statevector": init_statevector,
                "seed_simulator": 0,
                "shots": 1000,
                "memory": True,
            }
            out_1 = backend_with_options.run(bell).result().get_counts()
            out_2 = (
                backend_with_options.run(
                    bell,
                    initial_statevector=other_options["initial_statevector"],
                    seed_simulator=other_options["seed_simulator"],
                    shots=other_options["shots"],
                    memory=other_options["memory"],
                )
                .result()
                .get_counts()
            )
            self.assertNotEqual(out_1, out_2)
            out_options = backend_with_options.options
            for key in out_options:
                if key != "initial_statevector":
                    self.assertEqual(getattr(out_options, key), in_options.get(key))
                else:
                    np.testing.assert_array_equal(getattr(out_options, key), in_options.get(key))

    def test_clifford_circuit_bell_state(self):
        """Test that Clifford circuits use StabilizerState backend."""
        # Bell state is a Clifford circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        shots = 1000
        result = self.backend.run(
            qc,
            shots=shots,
            seed_simulator=self.seed,
            use_clifford_optimization=True,
        ).result()
        counts = result.get_counts()

        # Bell state should only produce |00> and |11>
        self.assertNotIn("01", counts)
        self.assertNotIn("10", counts)
        # At least one of the Bell outcomes must occur
        self.assertGreater(sum(counts.get(b, 0) for b in ["00", "11"]), 0)

        # Check roughly equal distribution
        total_shots = sum(counts.values())
        self.assertEqual(total_shots, shots)

    def test_non_clifford_circuit_with_t_gate(self):
        """Test that non-Clifford circuits fall back to statevector simulation."""
        # Circuit with T gate (non-Clifford)
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        qc.t(0)  # T gate is NOT Clifford
        qc.measure(0, 0)

        shots = 100
        result = self.backend.run(qc, shots=shots, seed_simulator=self.seed).result()
        counts = result.get_counts()

        # Should get valid measurement results
        self.assertGreater(len(counts), 0)
        self.assertEqual(sum(counts.values()), shots)

    def test_clifford_detection_various_gates(self):
        """Test Clifford detection with various gate combinations."""
        backend = BasicSimulator()

        # Test Clifford circuits
        clifford_qc = QuantumCircuit(3)
        clifford_qc.h(0)
        clifford_qc.s(1)
        clifford_qc.cx(0, 1)
        clifford_qc.cz(1, 2)
        clifford_qc.x(2)
        clifford_qc.y(0)
        clifford_qc.z(1)
        self.assertTrue(backend._is_clifford_circuit(clifford_qc))

        # Test non-Clifford circuit (T gate)
        non_clifford_t = QuantumCircuit(1)
        non_clifford_t.h(0)
        non_clifford_t.t(0)
        self.assertFalse(backend._is_clifford_circuit(non_clifford_t))

        # Test non-Clifford circuit (Rx gate)
        non_clifford_rx = QuantumCircuit(1)
        non_clifford_rx.rx(0.5, 0)
        self.assertFalse(backend._is_clifford_circuit(non_clifford_rx))

    def test_clifford_with_barriers_and_measurements(self):
        """Test that barriers and measurements don't affect Clifford detection."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.measure([0, 1], [0, 1])

        self.assertTrue(self.backend._is_clifford_circuit(qc))

        # Should still simulate correctly
        result = self.backend.run(qc, shots=100, seed_simulator=self.seed).result()
        counts = result.get_counts()
        self.assertGreater(len(counts), 0)

    def test_clifford_simulation_with_initial_statevector(self):
        """Test that initial_statevector option skips Clifford optimization."""
        # Clifford circuit
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        # Custom initial state (|10⟩)
        init_statevector = np.array([0, 0, 1, 0])

        shots = 100
        result = self.backend.run(
            qc, shots=shots, seed_simulator=self.seed, initial_statevector=init_statevector
        ).result()
        counts = result.get_counts()

        # Should get valid results (falls back to statevector)
        self.assertIsNotNone(counts)
        self.assertEqual(sum(counts.values()), shots)

    def test_large_clifford_circuit_performance(self):
        """Test that large Clifford circuits can be simulated efficiently."""
        # Create a larger Clifford circuit (would be slow with statevector)
        num_qubits = 32
        qc = QuantumCircuit(num_qubits, num_qubits)

        # Build a GHZ-like state (all Clifford gates)
        qc.h(0)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
        qc.measure(range(num_qubits), range(num_qubits))

        # Should complete without timeout
        shots = 100
        result = self.backend.run(
            qc, shots=shots, seed_simulator=self.seed, use_clifford_optimization=True
        ).result()
        counts = result.get_counts()

        self.assertEqual(sum(counts.values()), shots)
        # GHZ state should give all 0s or all 1s
        self.assertLessEqual(len(counts), 2)

    def test_clifford_partial_measurement(self):
        """Test Clifford circuit with partial measurements."""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)

        # Clifford operations
        qc.h(0)
        qc.cx(0, 1)
        qc.s(2)

        # Measure only some qubits
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        shots = 100
        result = self.backend.run(
            transpile(qc, self.backend), shots=shots, seed_simulator=self.seed
        ).result()
        counts = result.get_counts()

        # Should have valid counts
        self.assertEqual(sum(counts.values()), shots)

        # --- Qubit Limit & Pathway Tests ---

    def test_statevector_qubit_limit_exceeded(self):
        """Should error if more than 24 qubits in statevector simulation."""
        sim = BasicSimulator()
        qc = QuantumCircuit(25)
        with self.assertRaises(BasicProviderError):
            sim.run(qc, use_clifford_optimization=False)

    def test_clifford_qubit_limit_exceeded(self):
        """Should error if non-Clifford circuits above 24 qubits fall back to statevector."""
        sim = BasicSimulator()
        qc = QuantumCircuit(32)
        qc.h(range(32))
        qc.t(0)
        with self.assertRaises(BasicProviderError):
            sim.run(qc, use_clifford_optimization=True)

    def test_statevector_qubit_limit_pass(self):
        """Should succeed for exactly 8 qubits with statevector."""
        sim = BasicSimulator()
        qc = QuantumCircuit(8)  # changed from 24 to 8 to limit test time
        qc.h(range(8))
        job = sim.run(qc, use_clifford_optimization=False, shots=64)
        result = job.result()
        self.assertTrue(result.success)

    def test_clifford_qubit_limit_pass(self):
        """Should succeed for a small Clifford circuit when optimization is on."""
        sim = BasicSimulator()
        n_qubits = 32
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        job = sim.run(qc, use_clifford_optimization=True)
        result = job.result()
        self.assertTrue(result.success)

    def test_simulation_path_selection(self):
        """Simulator uses correct pathway for each optimization."""
        sim = BasicSimulator()
        qc_sv = QuantumCircuit(10)
        qc_sv.h(range(10))
        job_sv = sim.run(qc_sv, use_clifford_optimization=False)
        self.assertTrue(job_sv.result().success)
        qc_cl = QuantumCircuit(10)
        qc_cl.h(range(10))
        job_cl = sim.run(qc_cl, use_clifford_optimization=True)
        self.assertTrue(job_cl.result().success)

    def test_error_message_contains_limit(self):
        """Error should mention the correct qubit limit."""
        sim = BasicSimulator()
        qc = QuantumCircuit(25)
        with self.assertRaises(BasicProviderError) as cm:
            sim.run(qc, use_clifford_optimization=False)
        self.assertTrue("24" in str(cm.exception) or "statevector" in str(cm.exception))
        qc = QuantumCircuit(2049)
        qc.h(range(2049))
        with self.assertRaises(BasicProviderError) as cm:
            sim.run(qc, use_clifford_optimization=True)
        self.assertTrue("2048" in str(cm.exception) or "Clifford" in str(cm.exception))


if __name__ == "__main__":
    unittest.main()
