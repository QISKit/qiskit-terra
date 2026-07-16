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

"""Tests for QS decomposition synthesis methods."""

import unittest
import itertools
from test import combine
from ddt import ddt, data
import numpy as np
import scipy
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.unitary import qsd
from qiskit.circuit.library import XGate, ZGate, PhaseGate, UGate, UCGate, UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.predicates import matrix_equal
from test import QiskitTestCase


@ddt
class TestQuantumShannonDecomposer(QiskitTestCase):
    """
    Test Quantum Shannon Decomposition.
    """

    def setUp(self):
        super().setUp()
        np.random.seed(657)  # this seed should work for calls to scipy.stats.<method>.rvs()
        self.qsd = qsd.qs_decomposition

    def _get_lower_cx_bound(self, n):
        return 1 / 4 * (4**n - 3 * n - 1)

    def _qsd_l2_cx_count(self, n):
        """expected unoptimized cnot count for down to 2q"""
        return 9 / 16 * 4**n - 3 / 2 * 2**n

    def _qsd_l2_a1_mod(self, n):
        """expected optimized cnot count with opt_a1=True for down to 2q"""
        return (2 * 4 ** (n - 2) - 3) // 3

    def _qsd_l2_a2_mod(self, n):
        """expected optimized cnot count with opt_a2=True for down to 2q"""
        return 4 ** (n - 1) - 1

    def _qsd_l2_a1a2_mod(self, n):
        """expected optimized cnot count with opt_a1=True and opt_a2=True for down to 2q"""
        return (22 / 48) * 4**n - (3 / 2) * 2**n + 5 / 3

    def _qsd_ucrz(self, n):
        """expected cnot count of ucry/ucrz for down to 2q"""
        return 2 ** (n - 1)

    @combine(nqubits=[1, 2, 3, 4], opt_a1=[False, None], opt_a2=[False, None])
    def test_random_decomposition_l2_no_opt(self, nqubits, opt_a1, opt_a2):
        """test decomposition of random SU(n) down to 2 qubits without optimizations."""
        dim = 2**nqubits
        mat = scipy.stats.unitary_group.rvs(dim, random_state=1559)
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(np.allclose(mat, Operator(ccirc).data))
        if nqubits > 1:
            self.assertLessEqual(ccirc.count_ops().get("cx"), self._qsd_l2_cx_count(nqubits))
        else:
            self.assertEqual(sum(ccirc.count_ops().values()), 1)

    @combine(nqubits=[1, 2, 3, 4], opt_a1=[True, None], opt_a2=[False, None])
    def test_random_decomposition_l2_a1_opt(self, nqubits, opt_a1, opt_a2):
        """test decomposition of random SU(n) down to 2 qubits with 'a1' optimization."""
        dim = 2**nqubits
        mat = scipy.stats.unitary_group.rvs(dim, random_state=789)
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(np.allclose(mat, Operator(ccirc).data))
        if nqubits > 1:
            expected_cx = self._qsd_l2_cx_count(nqubits) - self._qsd_l2_a1_mod(nqubits)
            self.assertLessEqual(ccirc.count_ops().get("cx"), expected_cx)

    @combine(opt_a1=[True, None], opt_a2=[False, None])
    def test_SO3_decomposition_l2_a1_opt(self, opt_a1, opt_a2):
        """test decomposition of random So(3) down to 2 qubits with 'a1' optimization."""
        nqubits = 3
        dim = 2**nqubits
        mat = scipy.stats.ortho_group.rvs(dim)
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(np.allclose(mat, Operator(ccirc).data))
        expected_cx = self._qsd_l2_cx_count(nqubits) - self._qsd_l2_a1_mod(nqubits)
        self.assertLessEqual(ccirc.count_ops().get("cx"), expected_cx)

    @combine(opt_a1=[True, False, None], opt_a2=[True, False, None])
    def test_identity_decomposition(self, opt_a1, opt_a2):
        """Test decomposition on identity matrix"""
        nqubits = 3
        dim = 2**nqubits
        mat = np.identity(dim)
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        self.assertTrue(np.allclose(mat, Operator(circ).data))
        self.assertEqual(sum(circ.count_ops().values()), 0)

    @combine(nqubits=[1, 2, 3, 4], opt_a1=[True, None], opt_a2=[False, None])
    def test_diagonal(self, nqubits, opt_a1, opt_a2):
        """Test decomposition on diagonal -- qsd is not optimal"""
        dim = 2**nqubits
        mat = np.diag(np.exp(1j * np.random.normal(size=dim)))
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(np.allclose(mat, Operator(ccirc).data))
        if nqubits > 1:
            expected_cx = self._qsd_l2_cx_count(nqubits) - self._qsd_l2_a1_mod(nqubits)
            self.assertLessEqual(ccirc.count_ops().get("cx"), expected_cx)

    @combine(nqubits=[2, 3, 4], opt_a1=[True, None], opt_a2=[False, None])
    def test_hermitian(self, nqubits, opt_a1, opt_a2):
        """Test decomposition on hermitian -- qsd is not optimal"""
        # better might be (arXiv:1405.6741)
        dim = 2**nqubits
        umat = scipy.stats.unitary_group.rvs(dim, random_state=750)
        dmat = np.diag(np.exp(1j * np.random.normal(size=dim)))
        mat = umat.T.conjugate() @ dmat @ umat
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(mat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(np.allclose(mat, Operator(ccirc).data))
        if nqubits > 1:
            expected_cx = self._qsd_l2_cx_count(nqubits) - self._qsd_l2_a1_mod(nqubits)
            self.assertLessEqual(ccirc.count_ops().get("cx"), expected_cx)

    @combine(nqubits=[1, 2, 3, 4, 5], opt_a1=[True, None], opt_a2=[True, None])
    def test_opt_a1a2(self, nqubits, opt_a1, opt_a2):
        """Test decomposition with both optimization a1 and a2"""
        dim = 2**nqubits
        umat = scipy.stats.unitary_group.rvs(dim, random_state=1224)
        with self.assertWarns(DeprecationWarning):
            circ = self.qsd(umat, opt_a1=opt_a1, opt_a2=opt_a2)
        ccirc = transpile(circ, basis_gates=["u", "cx"], optimization_level=0)
        self.assertTrue(Operator(umat) == Operator(ccirc))
        if nqubits > 2:  # if nqubits = 3 this bound is 19
            self.assertLessEqual(
                ccirc.count_ops().get("cx"),
                self._qsd_l2_a1a2_mod(nqubits),
            )
        elif nqubits == 1:
            self.assertEqual(ccirc.count_ops().get("cx", 0), 0)
        elif nqubits == 2:
            self.assertLessEqual(ccirc.count_ops().get("cx", 0), 3)

    def test_a2_opt_single_2q(self):
        """
        Test a2_opt when a unitary causes a single final 2-qubit unitary for which this optimization
        won't help. This came up in issue 10787.
        """
        # this somewhat unique signed permutation matrix seems to cause the issue
        mat = np.array(
            [
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0 + 0.0j,
                ],
                [
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
                [
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    -1.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                    0.0 + 0.0j,
                ],
            ]
        )

        gate = UnitaryGate(mat)
        qc = QuantumCircuit(3)
        qc.append(gate, range(3))
        try:
            qc.to_gate().control(1, annotated=False)
        except UnboundLocalError as uerr:
            self.fail(str(uerr))

    def _create_random_multiplexed_gate(self, num_qubits):
        num_blocks = 2
        blocks = [scipy.stats.unitary_group.rvs(2 ** (num_qubits - 1)) for _ in range(num_blocks)]
        mat = scipy.linalg.block_diag(*blocks)  # control on "top" qubit
        multiplexed_gate = UnitaryGate(mat)
        return multiplexed_gate, mat

    def test_tensor_block_uc_2q(self):
        """Create 2q gate with multiplexed controls"""
        num_qubits = 2
        gate, _ = self._create_random_multiplexed_gate(num_qubits)
        for layout in itertools.permutations(range(num_qubits)):
            # create gate with "control" on different qubits
            qc = QuantumCircuit(num_qubits)
            qc.append(gate, layout)
            hidden_mat = Operator(qc).data
            num_mult = 0
            for j in range(num_qubits):
                um00, um11, um01, um10 = qsd._extract_multiplex_blocks(hidden_mat, k=j)
                # Check the off-diagonal
                if qsd._off_diagonals_are_zero(um01, um10):
                    num_mult += 1
                    qc_uc = QuantumCircuit(num_qubits)
                    ucgate = UCGate([um00, um11])
                    qc_uc.append(ucgate, layout)
                    uc_op = Operator(qc_uc)
                    self.assertEqual(Operator(qc), uc_op)
            self.assertTrue(num_mult)

    def _get_multiplex_matrix(self, um00, um11, k):
        """form matrix multiplexed wrt qubit k"""
        halfdim = um00.shape[0]
        dim = 2 * halfdim
        ndim = halfdim.bit_length()
        ure4 = np.zeros((2, halfdim, 2, halfdim), dtype=complex)
        ure4[0, :, 0, :] = um00
        ure4[1, :, 1, :] = um11
        urend = ure4.reshape((2,) * ndim + (2,) * ndim)
        urend = np.moveaxis(urend, ndim, k + ndim)
        urend = np.moveaxis(urend, 0, k)
        ure = urend.reshape(dim, dim)
        return ure

    def test_tensor_block_3q(self):
        """Create 3q gate with multiplexed controls"""
        num_qubits = 3
        gate, _ = self._create_random_multiplexed_gate(num_qubits)
        for layout in itertools.permutations(range(num_qubits)):
            # create gate with "control" on different qubits
            qc = QuantumCircuit(num_qubits)
            qc.append(gate, layout)
            hidden_mat = Operator(qc).data
            num_mult = 0
            for j in range(num_qubits):
                um00, um11, um01, um10 = qsd._extract_multiplex_blocks(hidden_mat, k=j)
                # Check the off-diagonal
                if qsd._off_diagonals_are_zero(um01, um10):
                    num_mult += 1
                    qc_uc = QuantumCircuit(num_qubits)
                    uc_mat = self._get_multiplex_matrix(um00, um11, j)
                    uc_gate = UnitaryGate(uc_mat)
                    qc_uc.append(uc_gate, range(num_qubits))
                    uc_op = Operator(qc_uc)
                    self.assertEqual(Operator(qc), uc_op)
            self.assertTrue(num_mult)

    @data(3, 4, 5, 6)
    def test_block_diag_opt(self, num_qubits):
        """Create a random multiplexed gate on num_qubits"""
        gate, _ = self._create_random_multiplexed_gate(num_qubits)
        layout = tuple(np.random.permutation(range(num_qubits)))
        # create gate with "control" on different qubits
        qc = QuantumCircuit(num_qubits)
        qc.append(gate, layout)
        hidden_op = Operator(qc)
        hidden_mat = hidden_op.data

        qc2 = qsd.qs_decomposition(hidden_mat)
        cqc2 = transpile(qc2, basis_gates=["u", "cx"], optimization_level=0)
        op2 = Operator(qc2)
        self.assertEqual(hidden_op, op2)
        self.assertLessEqual(
            cqc2.count_ops().get("cx", 0),
            2 * self._qsd_l2_cx_count(num_qubits - 1) + self._qsd_ucrz(num_qubits),
        )

    @combine(
        num_qubits=[3, 4, 5],
        base_gate=[XGate(), ZGate(), PhaseGate(0.321), UGate(0.21, 0.43, 0.65)],
    )
    def test_mc_1qubit_opt(self, num_qubits, base_gate):
        """Create a multi-controlled Z, P or U gate on num_qubits.
        This is less efficient than synthesizing MCX directly."""

        layout = tuple(np.random.permutation(range(num_qubits)))
        # create gate with "control" on different qubits
        qc = QuantumCircuit(num_qubits)
        gate = base_gate.control(num_qubits - 1, annotated=False)
        qc.append(gate, layout)

        hidden_op = Operator(qc)
        hidden_mat = hidden_op.data

        qc2 = qsd.qs_decomposition(hidden_mat)
        cqc2 = transpile(qc2, basis_gates=["u", "cx"], optimization_level=0)
        op2 = Operator(qc2)
        self.assertTrue(matrix_equal(hidden_op.to_matrix(), op2.to_matrix(), atol=1e-8))
        self.assertLessEqual(
            cqc2.count_ops().get("cx", 0),
            2 * self._qsd_l2_cx_count(num_qubits - 1) + self._qsd_ucrz(num_qubits),
        )

    @data(3, 4, 5, 6)
    def test_mc_2qubit_opt(self, num_qubits):
        """Create a multi-controlled 2-qubit unitary gate on num_qubits."""

        layout = tuple(np.random.permutation(range(num_qubits)))
        # create gate with "control" on different qubits
        base_gate = UnitaryGate(random_unitary(4, seed=1234))
        qc = QuantumCircuit(num_qubits)
        gate = base_gate.control(num_qubits - 2, annotated=False)
        qc.append(gate, layout)

        hidden_op = Operator(qc)
        hidden_mat = hidden_op.data

        qc2 = qsd.qs_decomposition(hidden_mat)
        cqc2 = transpile(qc2, basis_gates=["u", "cx"], optimization_level=0)
        op2 = Operator(qc2)
        self.assertTrue(matrix_equal(hidden_op.to_matrix(), op2.to_matrix(), atol=1e-8))
        self.assertLessEqual(
            cqc2.count_ops().get("cx", 0),
            2 * self._qsd_l2_cx_count(num_qubits - 1) + self._qsd_ucrz(num_qubits),
        )


if __name__ == "__main__":
    unittest.main()
