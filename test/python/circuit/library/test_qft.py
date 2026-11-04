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

"""Test library of QFT circuits."""

import io

import unittest
import warnings
import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, QFTGate
from qiskit.quantum_info import Operator
from qiskit.qpy import dump, load
from qiskit.qasm2 import dumps
from test import QiskitTestCase


@ddt
class TestQFT(QiskitTestCase):
    """Test the QFT."""

    def assertQFTIsCorrect(self, qft, num_qubits=None, inverse=False, add_swaps_at_end=False):
        """Assert that the QFT circuit produces the correct matrix.

        Can be provided with an explicit number of qubits, if None is provided the number
        of qubits is set to ``qft.num_qubits``.
        """
        if add_swaps_at_end:
            circuit = QuantumCircuit(*qft.qregs)
            for i in range(circuit.num_qubits // 2):
                circuit.swap(i, circuit.num_qubits - i - 1)

            qft.compose(circuit, inplace=True)

        simulated = Operator(qft)

        num_qubits = num_qubits or qft.num_qubits
        expected = np.empty((2**num_qubits, 2**num_qubits), dtype=complex)
        for i in range(2**num_qubits):
            i_index = int(bin(i)[2:].zfill(num_qubits), 2)
            for j in range(i, 2**num_qubits):
                entry = np.exp(2 * np.pi * 1j * i * j / 2**num_qubits) / 2 ** (num_qubits / 2)
                j_index = int(bin(j)[2:].zfill(num_qubits), 2)
                expected[i_index, j_index] = entry
                if i != j:
                    expected[j_index, i_index] = entry

        if inverse:
            expected = np.conj(expected)

        expected = Operator(expected)

        self.assertTrue(expected.equiv(simulated))

    @data(True, False)
    def test_qft_matrix(self, inverse):
        """Test the matrix representation of the QFT."""
        num_qubits = 5
        qft = QuantumCircuit(num_qubits)
        qft.append(QFTGate(num_qubits), range(num_qubits))
        if inverse:
            qft = qft.inverse()
        self.assertQFTIsCorrect(qft, inverse=inverse)

    def test_qft_is_inverse(self):
        """Test the is_inverse() method."""
        with self.assertWarns(DeprecationWarning):
            qft = QFT(2)

        with self.subTest(msg="initial object is not inverse"):
            self.assertFalse(qft.is_inverse())

        qft = qft.inverse()
        with self.subTest(msg="inverted"):
            self.assertTrue(qft.is_inverse())

        qft = qft.inverse()
        with self.subTest(msg="re-inverted"):
            self.assertFalse(qft.is_inverse())

    def test_qft_mutability(self):
        """Test the mutability of the QFT circuit."""
        with self.assertWarns(DeprecationWarning):
            qft = QFT()

        with self.subTest(msg="empty initialization"):
            self.assertEqual(qft.num_qubits, 0)
            self.assertEqual(qft.data, [])

        with self.subTest(msg="changing number of qubits"):
            qft.num_qubits = 3
            self.assertQFTIsCorrect(qft, num_qubits=3)

        with self.subTest(msg="test diminishing the number of qubits"):
            qft.num_qubits = 1
            self.assertQFTIsCorrect(qft, num_qubits=1)

        with self.subTest(msg="test with swaps"):
            qft.num_qubits = 4
            qft.do_swaps = False
            self.assertQFTIsCorrect(qft, add_swaps_at_end=True)

        with self.subTest(msg="inverse"):
            qft = qft.inverse()
            qft.do_swaps = True
            self.assertQFTIsCorrect(qft, inverse=True)

        with self.subTest(msg="double inverse"):
            qft = qft.inverse()
            self.assertQFTIsCorrect(qft)

        with self.subTest(msg="set approximation"):
            qft.approximation_degree = 2
            qft.do_swaps = True
            with self.assertRaises(AssertionError):
                self.assertQFTIsCorrect(qft)

    @data(
        (4, 0, False),
        (3, 0, True),
        (6, 2, False),
        (4, 5, True),
    )
    @unpack
    def test_qft_num_gates(self, num_qubits, approximation_degree, insert_barriers):
        """Test the number of gates in the QFT and the approximated QFT."""
        basis_gates = ["h", "swap", "cu1"]
        with self.assertWarns(DeprecationWarning):
            qft = QFT(
                num_qubits,
                approximation_degree=approximation_degree,
                insert_barriers=insert_barriers,
            )
        ops = transpile(qft, basis_gates=basis_gates, optimization_level=1).count_ops()

        with self.subTest(msg="assert H count"):
            self.assertEqual(ops["h"], num_qubits)

        with self.subTest(msg="assert swap count"):
            self.assertEqual(ops["swap"], num_qubits // 2)

        with self.subTest(msg="assert CU1 count"):
            expected = sum(
                max(0, min(num_qubits - 1 - k, num_qubits - 1 - approximation_degree))
                for k in range(num_qubits)
            )
            self.assertEqual(ops.get("cu1", 0), expected)

        with self.subTest(msg="assert barrier count"):
            expected = qft.num_qubits if insert_barriers else 0
            self.assertEqual(ops.get("barrier", 0), expected)

    def test_name_after_inverting(self):
        """Test the name after inverting the QFT is IQFT and not QFT_dg."""
        with self.assertWarns(DeprecationWarning):
            iqft = QFT(1).inverse()
        i2qft = iqft.inverse()

        with self.subTest(msg="inverted once"):
            self.assertEqual(iqft.name, "IQFT")

        with self.subTest(msg="inverted twice"):
            self.assertEqual(i2qft.name, "QFT")

        with self.subTest(msg="inverse as kwarg"):
            with self.assertWarns(DeprecationWarning):
                qft = QFT(1, inverse=True)
            self.assertEqual(qft.name, "IQFT")

    def test_warns_if_too_large(self):
        """Test that a warning is issued if the user tries to make a circuit that would need to
        represent angles smaller than the smallest normal double-precision floating-point number.
        It's too slow to actually let QFT construct a 1050+ qubit circuit for such a simple test, so
        we temporarily prevent QuantumCircuits from being created in order to short-circuit the QFT
        builder."""

        class SentinelException(Exception):
            """Dummy exception that raises itself as soon as it is created."""

            def __init__(self, *_args, **_kwargs):
                super().__init__()
                raise self

        # We don't want to issue a warning on mutation until we know that the values are
        # finalized; this is because a user might want to mutate the number of qubits and the
        # approximation degree.  In these cases, wait until we try to build the circuit.
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.filterwarnings(
                "always",
                category=RuntimeWarning,
                module=r"qiskit\..*",
                message=r".*precision loss in QFT.*",
            )
            with self.assertWarns(DeprecationWarning):
                qft = QFT()
            # Even with the approximation this will trigger the warning.
            qft.num_qubits = 1080
            qft.approximation_degree = 20

        self.assertFalse(caught_warnings)

        # Short-circuit the build method so it exits after input validation, but without actually
        # spinning the CPU to build a huge, useless object.
        with unittest.mock.patch("qiskit.circuit.QuantumCircuit.__init__", SentinelException):
            with self.assertWarnsRegex(RuntimeWarning, "precision loss in QFT"):
                with self.assertRaises(SentinelException):
                    qft._build()

    def test_name_after_inverse_rebuild(self):
        """Test the inverse QFT is correctly labeled, even after triggering rebuilds.

        Regression test of #14758.
        """
        with self.assertWarns(DeprecationWarning):
            qft = QFT(2)

        iqft = qft.inverse()  # name is IQFT
        iqft.num_qubits = 1  # name should still be IQFT, and not display IQFT_dg

        expect = "\n".join(
            [
                "   ┌──────┐",
                "q: ┤ IQFT ├",
                "   └──────┘",
            ]
        )
        out = str(iqft.draw())
        self.assertEqual(expect, out)


@ddt
class TestQFTGate(QiskitTestCase):
    """Test the QFT Gate."""

    @data(2, 3, 4, 5, 6)
    def test_array_equivalent_to_decomposition(self, num_qubits):
        """Test that the provided __array__ method and that the provided basic
        definition are equivalent.
        """
        qft_gate = QFTGate(num_qubits=num_qubits)
        qft_gate_decomposition = qft_gate.definition
        self.assertEqual(Operator(qft_gate), Operator(qft_gate_decomposition))

    @data(2, 3, 4, 5, 6)
    def test_gate_equivalent_to_original(self, num_qubits):
        """Test that the Operator can be constructed out of a QFT gate, and is
        equivalent to the Operator constructed out of a QFT circuit.
        """
        qft_gate = QFTGate(num_qubits=num_qubits)
        with self.assertWarns(DeprecationWarning):
            qft_circuit = QFT(num_qubits=num_qubits)
        self.assertEqual(Operator(qft_gate), Operator(qft_circuit))

    def test_append_to_circuit(self):
        """Test adding a QFTGate to a quantum circuit."""
        qc = QuantumCircuit(5)
        qc.append(QFTGate(4), [1, 2, 0, 4])
        self.assertIsInstance(qc.data[0].operation, QFTGate)

    @data(2, 3, 4, 5, 6)
    def test_circuit_with_gate_equivalent_to_original(self, num_qubits):
        """Test that the Operator can be constructed out of a circuit containing a QFT gate, and is
        equivalent to the Operator constructed out of a QFT circuit.
        """
        qft_gate = QFTGate(num_qubits=num_qubits)
        circuit_with_qft_gate = QuantumCircuit(num_qubits)
        circuit_with_qft_gate.append(qft_gate, range(num_qubits))
        with self.assertWarns(DeprecationWarning):
            qft_circuit = QFT(num_qubits=num_qubits)
        self.assertEqual(Operator(circuit_with_qft_gate), Operator(qft_circuit))

    def test_inverse(self):
        """Test that inverse can be constructed for a circuit with a QFTGate."""
        qc = QuantumCircuit(5)
        qc.append(QFTGate(4), [1, 2, 0, 4])
        qci = qc.inverse()
        self.assertEqual(Operator(qci), Operator(qc).adjoint())

    def test_reverse_ops(self):
        """Test reverse_ops works for a circuit with a QFTGate."""
        qc = QuantumCircuit(5)
        qc.cx(1, 3)
        qc.append(QFTGate(4), [1, 2, 0, 4])
        qc.h(0)
        qcr = qc.reverse_ops()
        expected = QuantumCircuit(5)
        expected.h(0)
        expected.append(QFTGate(4), [1, 2, 0, 4])
        expected.cx(1, 3)
        self.assertEqual(qcr, expected)

    def test_qasm(self):
        """Test qasm for circuits with QFTGates."""
        qr = QuantumRegister(5, "q0")
        qc = QuantumCircuit(qr)
        qc.append(QFTGate(num_qubits=4), [1, 2, 0, 4])
        qc.append(QFTGate(num_qubits=3), [0, 1, 2])
        qc.h(qr[0])
        qc_qasm = dumps(qc)
        reconstructed = QuantumCircuit.from_qasm_str(qc_qasm)
        self.assertEqual(Operator(qc), Operator(reconstructed))

    def test_qpy(self):
        """Test qpy for circuits with QFTGates."""
        qc = QuantumCircuit(6, 1)
        qc.append(QFTGate(num_qubits=4), [1, 2, 0, 4])
        qc.append(QFTGate(num_qubits=3), [0, 1, 2])
        qc.h(0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_gate_equality(self):
        """Test checking equality of QFTGates."""
        self.assertEqual(QFTGate(num_qubits=3), QFTGate(num_qubits=3))
        self.assertNotEqual(QFTGate(num_qubits=3), QFTGate(num_qubits=4))

    def test_circuit_with_gate_equality(self):
        """Test checking equality of circuits with QFTGates."""
        qc1 = QuantumCircuit(5)
        qc1.append(QFTGate(num_qubits=3), [1, 2, 0])

        qc2 = QuantumCircuit(5)
        qc2.append(QFTGate(num_qubits=3), [1, 2, 0])

        qc3 = QuantumCircuit(5)
        qc3.append(QFTGate(num_qubits=4), [1, 2, 0, 4])

        self.assertEqual(qc1, qc2)
        self.assertNotEqual(qc1, qc3)


if __name__ == "__main__":
    unittest.main()
