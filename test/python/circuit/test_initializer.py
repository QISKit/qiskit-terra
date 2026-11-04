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

"""
Initialize test.
"""

import math
import unittest
import numpy as np
from ddt import ddt, data

from qiskit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    transpile,
)
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import state_fidelity, Statevector, Operator
from qiskit.exceptions import QiskitError
from qiskit.circuit.library import Initialize
from test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError


@ddt
class TestInitialize(QiskitTestCase):
    """Qiskit Initialize tests."""

    _desired_fidelity = 0.99

    def test_disentangled(self):
        """test real-valued disentangled state initialization"""
        state1 = np.random.rand(8)
        state1 = state1 / np.linalg.norm(state1)
        state2 = np.random.rand(8)
        state2 = state2 / np.linalg.norm(state2)
        state3 = np.random.rand(8)
        state3 = state3 / np.linalg.norm(state3)

        qc1 = QuantumCircuit(9)
        qc1.initialize(state1, [0, 2, 3])
        qc1.initialize(state2, [1, 8, 5])
        qc1.initialize(state3, [7, 6, 4])

        statevector = Statevector(qc1)

        qc2 = QuantumCircuit(9)
        qc2.initialize(statevector)

        qc1 = transpile(qc1, basis_gates=["u", "cx"])
        qc2 = transpile(qc2, basis_gates=["u", "cx"])

        statevector1 = Statevector(qc1)
        statevector2 = Statevector(qc2)

        counts1 = qc1.count_ops()["cx"]
        counts2 = qc2.count_ops()["cx"]

        self.assertTrue(counts2 == counts1)
        self.assertTrue(np.allclose(statevector1, statevector2))

    def test_uniform_superposition(self):
        """Initialize a uniform superposition on 2 qubits."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_deterministic_state(self):
        """Initialize a computational-basis state |01> on 2 qubits."""
        desired_vector = [0, 1, 0, 0]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qc = transpile(qc, basis_gates=["u", "cx"])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_statevector(self):
        """Initialize gates from a statevector."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/5134 (footnote)
        desired_vector = [0, 0, 0, 1]
        qc = QuantumCircuit(2)
        statevector = Statevector.from_label("11")
        qc.initialize(statevector, [0, 1])
        self.assertEqual(qc.data[0].operation.params, desired_vector)

    def test_bell_state(self):
        """Initialize a Bell state on 2 qubits."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_ghz_state(self):
        """Initialize a GHZ state on 3 qubits."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_initialize_register(self):
        """Initialize one register out of two."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qr2 = QuantumRegister(2, "qr2")
        qc = QuantumCircuit(qr, qr2)
        qc.initialize(desired_vector, qr)
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, np.kron([1, 0, 0, 0], desired_vector))
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_initialize_one_by_one(self):
        """Initializing qubits individually into product state same as initializing the pair."""
        qubit_0_state = [1, 0]
        qubit_1_state = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        qr = QuantumRegister(2, "qr")
        qc_a = QuantumCircuit(qr)
        qc_a.initialize(np.kron(qubit_1_state, qubit_0_state), qr)

        qc_b = QuantumCircuit(qr)
        qc_b.initialize(qubit_0_state, [qr[0]])
        qc_b.initialize(qubit_1_state, [qr[1]])

        statevector_a = Statevector(qc_a)
        statevector_b = Statevector(qc_b)
        fidelity = state_fidelity(statevector_a, statevector_b)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_single_qubit(self):
        """Initialize a single qubit to a weighted superposition state."""
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3)]
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_random_3qubit(self):
        """Initialize to a non-trivial 3-qubit state."""
        desired_vector = [
            1 / math.sqrt(16) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(16) * complex(1, 1),
            0,
            0,
            1 / math.sqrt(8) * complex(1, 2),
            1 / math.sqrt(16) * complex(1, 0),
            0,
        ]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_random_4qubit(self):
        """Initialize to a non-trivial 4-qubit state."""
        desired_vector = [
            1 / math.sqrt(4) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            0,
            0,
            0,
            0,
            0,
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            0,
            0,
            0,
            1 / math.sqrt(4) * complex(1, 0),
            1 / math.sqrt(8) * complex(1, 0),
        ]
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_malformed_amplitudes(self):
        """Initializing to a vector with 3 amplitudes fails."""
        desired_vector = [1 / math.sqrt(3), math.sqrt(2) / math.sqrt(3), 0]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(QiskitError, qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_non_unit_probability(self):
        """Initializing to a vector with probabilities not summing to 1 fails."""
        desired_vector = [1, 1]
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        self.assertRaises(QiskitError, qc.initialize, desired_vector, [qr[0], qr[1]])

    def test_normalize(self):
        """Test initializing with a non-normalized vector is normalized, if specified."""
        desired_vector = [1, 1]
        normalized = np.asarray(desired_vector) / np.linalg.norm(desired_vector)

        qc = QuantumCircuit(1)
        qc.initialize(desired_vector, [0], normalize=True)
        op = qc.data[0].operation
        self.assertAlmostEqual(np.linalg.norm(op.params), 1)
        self.assertEqual(Statevector(qc), Statevector(normalized))

    def test_wrong_vector_size(self):
        """Initializing to a vector with a size different to the qubit parameter length.
        See https://github.com/Qiskit/qiskit-terra/issues/2372"""
        qr = QuantumRegister(2)

        random_state = [
            1 / math.sqrt(4) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            1 / math.sqrt(4) * complex(1, 0),
            1 / math.sqrt(8) * complex(1, 0),
        ]

        qc = QuantumCircuit(qr)

        self.assertRaises(QiskitError, qc.initialize, random_state, qr[0:2])

    def test_initialize_middle_circuit(self):
        """Reset + initialize gives the correct statevector."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.reset(qr[0])
        qc.reset(qr[1])
        qc.initialize(desired_vector, [qr[0], qr[1]])
        qc.measure(qr, cr)
        # statevector simulator does not support reset
        shots = 2000
        threshold = 0.05 * shots
        backend = BasicSimulator()
        job = backend.run(transpile(qc, backend), shots=shots, seed_simulator=42)
        result = job.result()
        counts = result.get_counts()
        target = {"00": shots / 4, "01": shots / 4, "10": shots / 4, "11": shots / 4}
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_math_amplitudes(self):
        """Initialize to amplitudes given by math expressions"""
        desired_vector = [
            0,
            math.cos(math.pi / 3) * complex(0, 1) / math.sqrt(4),
            math.sin(math.pi / 3) / math.sqrt(4),
            0,
            0,
            0,
            0,
            0,
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(8) * complex(0, 1),
            0,
            0,
            0,
            0,
            1 / math.sqrt(4),
            1 / math.sqrt(4) * complex(0, 1),
        ]
        qr = QuantumRegister(4, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2], qr[3]])
        statevector = Statevector(qc)
        fidelity = state_fidelity(statevector, desired_vector)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_combiner(self):
        """Combining two circuits containing initialize."""
        desired_vector_1 = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        desired_vector_2 = [1.0 / math.sqrt(2), -1.0 / math.sqrt(2)]
        qr = QuantumRegister(1, "qr")
        cr = ClassicalRegister(1, "cr")
        qc1 = QuantumCircuit(qr, cr)
        qc1.initialize(desired_vector_1, [qr[0]])

        qc2 = QuantumCircuit(qr, cr)
        qc2.initialize(desired_vector_2, [qr[0]])

        statevector = Statevector(qc1.compose(qc2))
        fidelity = state_fidelity(statevector, desired_vector_2)
        self.assertGreater(
            fidelity,
            self._desired_fidelity,
            f"Initializer has low fidelity {fidelity:.2g}.",
        )

    def test_equivalence(self):
        """Test two similar initialize instructions evaluate to equal."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        qr = QuantumRegister(2, "qr")

        qc1 = QuantumCircuit(qr, name="circuit")
        qc1.initialize(desired_vector, [qr[0], qr[1]])

        qc2 = QuantumCircuit(qr, name="circuit")
        qc2.initialize(desired_vector, [qr[0], qr[1]])

        self.assertEqual(qc1, qc2)

    def test_max_number_cnots(self):
        """
        Check if the number of cnots <= 2^(n+1) - 2n (arXiv:quant-ph/0406176)
        """
        num_qubits = 4
        _optimization_level = 0

        vector = np.array(
            [
                0.1314346 + 0.0j,
                0.32078572 - 0.01542775j,
                0.13146466 + 0.0945312j,
                0.21090852 + 0.07935982j,
                0.1700122 - 0.07905648j,
                0.15570757 - 0.12309154j,
                0.18039667 + 0.04904504j,
                0.22227187 - 0.05055569j,
                0.23573255 - 0.09894111j,
                0.27307292 - 0.10372994j,
                0.24162792 + 0.1090791j,
                0.3115577 + 0.1211683j,
                0.1851788 + 0.08679141j,
                0.36226463 - 0.09940202j,
                0.13863395 + 0.10558225j,
                0.30767986 + 0.02073838j,
            ]
        )

        vector = vector / np.linalg.norm(vector)

        qr = QuantumRegister(num_qubits, "qr")
        circuit = QuantumCircuit(qr)
        circuit.initialize(vector, qr)

        b = transpile(
            circuit,
            basis_gates=["u1", "u2", "u3", "cx"],
            optimization_level=_optimization_level,
            seed_transpiler=42,
        )

        number_cnots = b.count_ops()["cx"]
        max_cnots = 2 ** (num_qubits + 1) - 2 * num_qubits

        self.assertLessEqual(number_cnots, max_cnots)

    def test_from_labels(self):
        """Initialize from labels."""
        desired_sv = Statevector.from_label("01+-lr")
        qc = QuantumCircuit(6)
        qc.initialize("01+-lr", range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)

    def test_from_int(self):
        """Initialize from int."""
        desired_sv = Statevector.from_label("110101")
        qc = QuantumCircuit(6)
        qc.initialize(53, range(6))
        actual_sv = Statevector.from_instruction(qc)
        self.assertTrue(desired_sv == actual_sv)

    def _remove_resets(self, circ):
        circ.data = [instr for instr in circ.data if instr.operation.name != "reset"]

    def test_global_phase_random(self):
        """Test global phase preservation with random state vectors"""
        from qiskit.quantum_info import random_statevector

        repeats = 5
        for n_qubits in [1, 2, 4]:
            for irep in range(repeats):
                with self.subTest(i=f"{n_qubits}_{irep}"):
                    dim = 2**n_qubits
                    qr = QuantumRegister(n_qubits)
                    initializer = QuantumCircuit(qr)
                    target = random_statevector(dim)
                    initializer.initialize(target, qr)
                    uninit = initializer.data[0].operation.definition
                    self._remove_resets(uninit)
                    evolve = Statevector(uninit)
                    self.assertEqual(target, evolve)

    def test_global_phase_1q(self):
        """Test global phase preservation with some simple 1q statevectors"""
        target_list = [
            Statevector([1j, 0]),
            Statevector([0, 1j]),
            Statevector([1j / np.sqrt(2), 1j / np.sqrt(2)]),
        ]
        n_qubits = 1
        dim = 2**n_qubits
        qr = QuantumRegister(n_qubits)
        for target in target_list:
            with self.subTest(i=target):
                initializer = QuantumCircuit(qr)
                initializer.initialize(target, qr)
                # need to get rid of the resets in order to use the Operator class
                disentangler = Operator(initializer.data[0].operation.definition.data[1].operation)
                zero = Statevector.from_int(0, dim)
                actual = zero & disentangler
                self.assertEqual(target, actual)

    @data(2, "11", [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)])
    def test_decompose_contains_stateprep(self, state):
        """Test initialize decomposes to a StatePreparation and reset"""
        qc = QuantumCircuit(2)
        qc.initialize(state)
        decom_circ = qc.decompose()

        self.assertEqual(decom_circ.data[0].operation.name, "reset")
        self.assertEqual(decom_circ.data[1].operation.name, "reset")
        self.assertEqual(decom_circ.data[2].operation.name, "state_preparation")

    def test_mutating_params(self):
        """Test mutating Initialize params correctly updates StatePreparation params"""
        init = Initialize("11")
        init.params = "00"
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.append(init, qr)
        decom_circ = qc.decompose()

        self.assertEqual(decom_circ.data[2].operation.name, "state_preparation")
        self.assertEqual(decom_circ.data[2].operation.params, ["0", "0"])

    def test_gates_to_uncompute(self):
        """Test the gates_to_uncompute() method."""
        desired_vector = [0.5, 0.5, 0.5, 0.5]
        initialize = Initialize(desired_vector)
        qc = initialize.gates_to_uncompute().inverse()
        vec = Statevector(qc)
        self.assertTrue(vec == Statevector(desired_vector))

    def test_repeat(self):
        """Test the repeat() method."""
        desired_vector = np.array([0.5, 0.5, 0.5, 0.5])
        initialize = Initialize(desired_vector)
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.append(initialize.repeat(2), qr)
        statevector = Statevector(qc)
        self.assertTrue(np.allclose(statevector, desired_vector))

    def test_irreversible_initialize(self):
        """Test failure of the initialize.inverse() method"""
        circuit = QuantumCircuit(4, 0)
        circuit.initialize(0, circuit.qubits)
        with self.assertRaises(CircuitError):
            _ = circuit.inverse()

    def test_gates_to_uncompute_integer_input(self):
        """Test that gates_to_uncompute works correctly for integer-bitmap inputs.
        Regression test for https://github.com/Qiskit/qiskit/issues/16413
        """
        # Integer-bitmap input: Initialize(1, num_qubits=2) prepares |01>
        int_init = Initialize(1, num_qubits=2)
        uncompute_int = int_init.gates_to_uncompute()
        self.assertEqual(uncompute_int.num_qubits, 2)

        # Verify integer-bitmap uncompute actually maps |01> back to |00...0>
        result = Statevector.from_label("01").evolve(uncompute_int)
        zero_state = Statevector.from_label("00")
        self.assertTrue(result.equiv(zero_state))


class TestInstructionParam(QiskitTestCase):
    """Test conversion of numpy type parameters."""

    def test_init(self):
        """Verify initialize gate converts numpy.complex to complex."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4151
        qc = QuantumCircuit(1)
        vec = np.array([0, 0 + 1j])
        qc.initialize(vec, 0)

        params = qc.data[0].operation.params
        self.assertTrue(
            all(isinstance(p, complex) and not isinstance(p, np.number) for p in params)
        )


if __name__ == "__main__":
    unittest.main()
