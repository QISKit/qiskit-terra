# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for MCGupDiag — multi-controlled gate up to diagonal synthesis."""

import unittest

import numpy as np
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal import MCGupDiag
from qiskit.quantum_info import Operator
from test import QiskitTestCase


def _random_su2(seed=None):
    """Return a random 2x2 unitary matrix."""
    rng = np.random.default_rng(seed)
    # Random SU(2): use scipy-style parametrisation via QR decomposition
    z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / np.sqrt(2)
    q, _ = np.linalg.qr(z)
    # Make determinant 1
    q = q * (np.linalg.det(q) ** -0.5)
    return q


# Standard 2x2 unitaries used as test inputs
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.diag([1, np.exp(1j * np.pi / 4)])
_I = np.eye(2, dtype=complex)


@ddt
class TestMCGupDiag(QiskitTestCase):
    """Test MCGupDiag gate construction and synthesis."""

    # ------------------------------------------------------------------
    # Construction / validation
    # ------------------------------------------------------------------

    def test_identity_gate_no_controls(self):
        """MCGupDiag with 0 controls and identity gate creates a 1-qubit gate."""
        gate = MCGupDiag(_I, num_controls=0, num_ancillas_zero=0, num_ancillas_dirty=0)
        self.assertEqual(gate.num_qubits, 1)

    def test_x_gate_one_control(self):
        """MCGupDiag with 1 control creates a 2-qubit gate."""
        gate = MCGupDiag(_X, num_controls=1, num_ancillas_zero=0, num_ancillas_dirty=0)
        self.assertEqual(gate.num_qubits, 2)

    def test_num_qubits_formula(self):
        """num_qubits == 1 + num_controls + num_ancillas_zero + num_ancillas_dirty."""
        for nc, naz, nad in [(0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 0, 2), (2, 1, 1)]:
            with self.subTest(nc=nc, naz=naz, nad=nad):
                gate = MCGupDiag(_H, num_controls=nc, num_ancillas_zero=naz, num_ancillas_dirty=nad)
                self.assertEqual(gate.num_qubits, 1 + nc + naz + nad)

    def test_non_unitary_raises(self):
        """Non-unitary matrix raises QiskitError."""
        from qiskit.exceptions import QiskitError

        bad = np.array([[2, 0], [0, 1]], dtype=complex)
        with self.assertRaises(QiskitError):
            MCGupDiag(bad, num_controls=1, num_ancillas_zero=0, num_ancillas_dirty=0)

    def test_wrong_shape_raises(self):
        """Non-2x2 matrix raises QiskitError."""
        from qiskit.exceptions import QiskitError

        bad = np.eye(4, dtype=complex)
        with self.assertRaises(QiskitError):
            MCGupDiag(bad, num_controls=1, num_ancillas_zero=0, num_ancillas_dirty=0)

    # ------------------------------------------------------------------
    # Synthesis correctness: U = D * U'
    # ------------------------------------------------------------------

    @data(
        (_I, 0, "identity, 0 controls"),
        (_X, 0, "X, 0 controls"),
        (_H, 0, "H, 0 controls"),
        (_X, 1, "X, 1 control"),
        (_H, 1, "H, 1 control"),
        (_T, 1, "T, 1 control"),
        (_X, 2, "X, 2 controls"),
        (_H, 2, "H, 2 controls"),
    )
    @unpack
    def test_synthesis_up_to_diagonal(self, mat, num_ctrls, label):
        """Synthesised circuit times diagonal equals the original MCG unitary."""
        gate = MCGupDiag(mat, num_controls=num_ctrls, num_ancillas_zero=0, num_ancillas_dirty=0)

        # Full MCG unitary: only the last column block is `mat`, rest is identity
        num_qubits = 1 + num_ctrls
        dim = 2**num_qubits
        expected = np.eye(dim, dtype=complex)
        expected[-2:, -2:] = mat

        # Decompose definition down to basis gates and get its unitary
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))
        basis_circ = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        actual = Operator(basis_circ).data

        # actual = D @ expected  =>  D = actual @ expected†  =>  D must be diagonal
        d_mat = actual @ expected.conj().T
        # Check D is diagonal (off-diagonal entries are zero)
        off_diag = d_mat - np.diag(np.diag(d_mat))
        self.assertTrue(
            np.allclose(off_diag, 0, atol=1e-10),
            msg=f"[{label}] synthesised circuit is not equal to MCG up to a diagonal",
        )
        # Check diagonal entries have magnitude 1
        self.assertTrue(
            np.allclose(np.abs(np.diag(d_mat)), 1, atol=1e-10),
            msg=f"[{label}] diagonal entries do not have unit magnitude",
        )

    def test_random_su2_one_control(self):
        """Random SU(2) with 1 control synthesises correctly up to diagonal."""
        mat = _random_su2(seed=42)
        gate = MCGupDiag(mat, num_controls=1, num_ancillas_zero=0, num_ancillas_dirty=0)

        expected = np.eye(4, dtype=complex)
        expected[-2:, -2:] = mat

        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, range(gate.num_qubits))
        basis_circ = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        actual = Operator(basis_circ).data

        d_mat = actual @ expected.conj().T
        off_diag = d_mat - np.diag(np.diag(d_mat))
        self.assertTrue(np.allclose(off_diag, 0, atol=1e-10))
        self.assertTrue(np.allclose(np.abs(np.diag(d_mat)), 1, atol=1e-10))

    # ------------------------------------------------------------------
    # Synthesis correctness with ancilla qubits
    # ------------------------------------------------------------------

    @data(
        (_X, 1, 1, 0, "X, 1 control, 1 zero-ancilla"),
        (_X, 1, 0, 1, "X, 1 control, 1 dirty-ancilla"),
        (_H, 2, 1, 1, "H, 2 controls, 1 zero + 1 dirty ancilla"),
        (_T, 0, 2, 0, "T, 0 controls, 2 zero-ancillas"),
    )
    @unpack
    def test_synthesis_with_ancillas(
        self, mat, num_ctrls, num_ancillas_zero, num_ancillas_dirty, label
    ):
        """Synthesised circuit times diagonal equals the MCG unitary tensored with
        identity on the ancilla qubits.

        Regression test: MCGupDiag._define() used to append the Rust-synthesised
        sub-circuit onto the full ancilla-padded register with the wrong qubit
        slice, raising CircuitError whenever any ancillas were present.
        """
        gate = MCGupDiag(
            mat,
            num_controls=num_ctrls,
            num_ancillas_zero=num_ancillas_zero,
            num_ancillas_dirty=num_ancillas_dirty,
        )

        num_active = 1 + num_ctrls
        dim_active = 2**num_active
        dim_ancillas = 2 ** (num_ancillas_zero + num_ancillas_dirty)

        expected_active = np.eye(dim_active, dtype=complex)
        expected_active[-2:, -2:] = mat
        expected = np.kron(np.eye(dim_ancillas), expected_active)

        definition = gate.definition
        self.assertEqual(definition.num_qubits, gate.num_qubits)

        actual = Operator(definition).data
        diag = np.asarray(gate._get_diagonal())
        d_full = np.kron(np.eye(dim_ancillas), np.diag(diag))
        reconstructed = d_full @ actual

        self.assertTrue(
            np.allclose(reconstructed, expected, atol=1e-10),
            msg=f"[{label}] synthesised circuit with ancillas is not equal to MCG (x) I up to a diagonal",
        )

    @data(
        (0, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (2, 1, 1),
        (3, 0, 2),
    )
    @unpack
    def test_definition_active_qubit_count(self, num_ctrls, num_ancillas_zero, num_ancillas_dirty):
        """The synthesised sub-instruction inside `definition` must act on exactly
        `num_controls + 1` qubits (target + controls), never on the ancillas."""
        gate = MCGupDiag(
            _X,
            num_controls=num_ctrls,
            num_ancillas_zero=num_ancillas_zero,
            num_ancillas_dirty=num_ancillas_dirty,
        )
        definition = gate.definition
        self.assertEqual(definition.num_qubits, gate.num_qubits)
        self.assertEqual(len(definition.data), 1)
        instruction = definition.data[0]
        self.assertEqual(len(instruction.qubits), num_ctrls + 1)


if __name__ == "__main__":
    unittest.main()
