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

"""Tests for SparsePauliOp class."""

import itertools as it
import unittest
from test import QiskitTestCase, combine

import numpy as np
import rustworkx as rx
import scipy.sparse
import ddt

from qiskit import QiskitError
from qiskit.circuit import Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.library import efficient_su2
from qiskit.circuit.parametertable import ParameterView
from qiskit.compiler.transpiler import transpile
from qiskit.primitives import BackendEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparseObservable
from qiskit.quantum_info.operators import (
    Operator,
    Pauli,
    PauliList,
    SparsePauliOp,
)
from qiskit.utils import optionals


def pauli_mat(label):
    """Return Pauli matrix from a Pauli label."""
    mat = np.eye(1, dtype=complex)
    for i in label:
        if i == "I":
            mat = np.kron(mat, np.eye(2, dtype=complex))
        elif i == "X":
            mat = np.kron(mat, np.array([[0, 1], [1, 0]], dtype=complex))
        elif i == "Y":
            mat = np.kron(mat, np.array([[0, -1j], [1j, 0]], dtype=complex))
        elif i == "Z":
            mat = np.kron(mat, np.array([[1, 0], [0, -1]], dtype=complex))
        else:
            raise QiskitError(f"Invalid Pauli string {i}")
    return mat


class TestSparsePauliOpInit(QiskitTestCase):
    """Tests for SparsePauliOp initialization."""

    def test_str_init(self):
        """Test str initialization."""
        for label in ["IZ", "XI", "YX", "ZZ"]:
            pauli_list = PauliList(label)
            spp_op = SparsePauliOp(label)
            self.assertEqual(spp_op.paulis, pauli_list)
            np.testing.assert_array_equal(spp_op.coeffs, [1])

    def test_pauli_list_init(self):
        """Test PauliList initialization."""
        labels = ["I", "X", "Y", "-Z", "iZ", "-iX"]
        paulis = PauliList(labels)
        with self.subTest(msg="no coeffs"):
            spp_op = SparsePauliOp(paulis)
            np.testing.assert_array_equal(spp_op.coeffs, [1, 1, 1, -1, 1j, -1j])
            paulis.phase = 0
            self.assertEqual(spp_op.paulis, paulis)
        paulis = PauliList(labels)
        with self.subTest(msg="with coeffs"):
            coeffs = [1, 2, 3, 4, 5, 6]
            spp_op = SparsePauliOp(paulis, coeffs)
            np.testing.assert_array_equal(spp_op.coeffs, [1, 2, 3, -4, 5j, -6j])
            paulis.phase = 0
            self.assertEqual(spp_op.paulis, paulis)
        paulis = PauliList(labels)
        with self.subTest(msg="with Parameterized coeffs"):
            params = ParameterVector("params", 6)
            coeffs = np.array(params)
            spp_op = SparsePauliOp(paulis, coeffs)
            target = coeffs.copy()
            target[3] *= -1
            target[4] *= 1j
            target[5] *= -1j
            np.testing.assert_array_equal(spp_op.coeffs, target)
            paulis.phase = 0
            self.assertEqual(spp_op.paulis, paulis)

    def test_sparse_pauli_op_init(self):
        """Test SparsePauliOp initialization."""
        labels = ["I", "X", "Y", "-Z", "iZ", "-iX"]
        with self.subTest(msg="make SparsePauliOp from SparsePauliOp"):
            op = SparsePauliOp(labels)
            ref_op = op.copy()
            spp_op = SparsePauliOp(op)
            self.assertEqual(spp_op, ref_op)
            np.testing.assert_array_equal(ref_op.paulis.phase, np.zeros(ref_op.size))
            np.testing.assert_array_equal(spp_op.paulis.phase, np.zeros(spp_op.size))
            # make sure the changes of `op` do not propagate through to `spp_op`
            op.paulis.z[:] = False
            op.coeffs *= 2
            self.assertNotEqual(spp_op, op)
            self.assertEqual(spp_op, ref_op)
        with self.subTest(msg="make SparsePauliOp from SparsePauliOp and ndarray"):
            op = SparsePauliOp(labels)
            coeffs = np.array([1, 2, 3, 4, 5, 6])
            spp_op = SparsePauliOp(op, coeffs)
            ref_op = SparsePauliOp(op.paulis.copy(), coeffs.copy())
            self.assertEqual(spp_op, ref_op)
            np.testing.assert_array_equal(ref_op.paulis.phase, np.zeros(ref_op.size))
            np.testing.assert_array_equal(spp_op.paulis.phase, np.zeros(spp_op.size))
            # make sure the changes of `op` and `coeffs` do not propagate through to `spp_op`
            op.paulis.z[:] = False
            coeffs *= 2
            self.assertNotEqual(spp_op, op)
            self.assertEqual(spp_op, ref_op)
        with self.subTest(msg="make SparsePauliOp from PauliList"):
            paulis = PauliList(labels)
            spp_op = SparsePauliOp(paulis)
            ref_op = SparsePauliOp(labels)
            self.assertEqual(spp_op, ref_op)
            np.testing.assert_array_equal(ref_op.paulis.phase, np.zeros(ref_op.size))
            np.testing.assert_array_equal(spp_op.paulis.phase, np.zeros(spp_op.size))
            # make sure the change of `paulis` does not propagate through to `spp_op`
            paulis.z[:] = False
            self.assertEqual(spp_op, ref_op)
        with self.subTest(msg="make SparsePauliOp from PauliList and ndarray"):
            paulis = PauliList(labels)
            coeffs = np.array([1, 2, 3, 4, 5, 6])
            spp_op = SparsePauliOp(paulis, coeffs)
            ref_op = SparsePauliOp(labels, coeffs.copy())
            self.assertEqual(spp_op, ref_op)
            np.testing.assert_array_equal(ref_op.paulis.phase, np.zeros(ref_op.size))
            np.testing.assert_array_equal(spp_op.paulis.phase, np.zeros(spp_op.size))
            # make sure the changes of `paulis` and `coeffs` do not propagate through to `spp_op`
            paulis.z[:] = False
            coeffs[:] = 0
            self.assertEqual(spp_op, ref_op)

    def test_sparse_pauli_op_init_long_ys(self):
        """Test heavy-weight SparsePauliOp initialization."""
        y = SparsePauliOp("Y" * 1000)
        self.assertEqual(1, y.coeffs[0])


@ddt.ddt
class TestSparsePauliOpConversions(QiskitTestCase):
    """Tests SparsePauliOp representation conversions."""

    @ddt.data(1, 2, 4)
    def test_from_operator_single(self, num_qubits):
        """Test from_operator methods."""
        for tup in it.product(["I", "X", "Y", "Z"], repeat=num_qubits):
            label = "".join(tup)
            with self.subTest(msg=label):
                spp_op = SparsePauliOp.from_operator(Operator(pauli_mat(label)))
                np.testing.assert_array_equal(spp_op.coeffs, [1])
                self.assertEqual(spp_op.paulis, PauliList(label))

    @ddt.data(
        SparsePauliOp.from_sparse_list([("", (), 1.0), ("X", (0,), -2.0j)], num_qubits=1),
        SparsePauliOp.from_sparse_list([("", (), 1.0), ("Y", (0,), -2.0j)], num_qubits=1),
        SparsePauliOp.from_sparse_list([("Y", (0,), 1.0), ("Z", (0,), -2.0j)], num_qubits=1),
        SparsePauliOp.from_sparse_list(
            [("Y", (0,), 1.0), ("YY", (1, 0), -0.5), ("YYY", (2, 1, 0), 1j)], num_qubits=3
        ),
        SparsePauliOp.from_sparse_list(
            [("XZ", (2, 0), 1.0), ("YZ", (1, 0), -0.5), ("ZZ", (2, 1), 1j)], num_qubits=3
        ),
    )
    def test_from_operator_roundtrip(self, op):
        """Test `SparsePauliOp.from_operator` roundtrips things correctly."""
        # Ensure canonical order of the input. Part of this test is ensuring that the output is
        # given in canonical order too.  The coefficients in the inputs are chosen to be simple
        # multiples of powers of two, so there are no floating-point rounding or associativity
        # concerns.
        op = op.simplify().sort()
        roundtrip = SparsePauliOp.from_operator(op.to_matrix())
        self.assertEqual(roundtrip, op)

    def test_from_operator_tolerance(self):
        """Test that terms whose coefficient falls below the tolerance are removed."""
        operator = SparsePauliOp.from_list(
            [("IIXI", 0.25), ("IIZI", -0.25j), ("IXYI", 0.5j)]
        ).to_matrix()
        expected = SparsePauliOp.from_list([("IXYI", 0.5j)])
        self.assertEqual(SparsePauliOp.from_operator(operator, 0.26), expected)

    def test_from_list(self):
        """Test from_list method."""
        labels = ["XXZ", "IXI", "YZZ", "III"]
        coeffs = [3.0, 5.5, -1j, 23.3333]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        np.testing.assert_array_equal(spp_op.coeffs, coeffs)
        self.assertEqual(spp_op.paulis, PauliList(labels))
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    def test_from_list_parameters(self):
        """Test from_list method with parameters."""
        labels = ["XXZ", "IXI", "YZZ", "III"]
        coeffs = ParameterVector("a", 4)
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs), dtype=object)
        np.testing.assert_array_equal(spp_op.coeffs, coeffs)
        self.assertEqual(spp_op.paulis, PauliList(labels))
        self.assertEqual(spp_op.coeffs.dtype, object)

    def test_from_index_list(self):
        """Test from_sparse_list method specifying the Paulis via indices."""
        expected_labels = ["XXZ", "IXI", "YIZ", "III"]
        paulis = ["XXZ", "X", "YZ", ""]
        indices = [[2, 1, 0], [1], [2, 0], []]
        coeffs = [3.0, 5.5, -1j, 23.3333]
        spp_op = SparsePauliOp.from_sparse_list(zip(paulis, indices, coeffs), num_qubits=3)
        np.testing.assert_array_equal(spp_op.coeffs, coeffs)
        self.assertEqual(spp_op.paulis, PauliList(expected_labels))
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    def test_from_index_list_parameters(self):
        """Test from_sparse_list method specifying the Paulis via indices with parameters."""
        expected_labels = ["XXZ", "IXI", "YIZ", "III"]
        paulis = ["XXZ", "X", "YZ", ""]
        indices = [[2, 1, 0], [1], [2, 0], []]
        coeffs = ParameterVector("a", 4)
        spp_op = SparsePauliOp.from_sparse_list(
            zip(paulis, indices, coeffs), num_qubits=3, dtype=object
        )
        np.testing.assert_array_equal(spp_op.coeffs, coeffs)
        self.assertEqual(spp_op.paulis, PauliList(expected_labels))
        self.assertEqual(spp_op.coeffs.dtype, object)

    def test_from_sparse_list_dtype_inference(self):
        """Test from_sparse_list auto-infers dtype correctly."""
        # infer complex128 dtype.
        spp_op = SparsePauliOp.from_sparse_list([("X", [0], 1.0), ("Z", [1], 2.0j)], num_qubits=2)
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

        # infer object dtype due to ParameterExpression.
        param = Parameter("a")
        spp_op = SparsePauliOp.from_sparse_list([("X", [0], param)], num_qubits=1)
        self.assertEqual(spp_op.coeffs.dtype, object)

    def test_from_list_dtype_inference(self):
        """Test from_list auto-infers dtype correctly."""
        # infer complex128 dtype.
        spp_op = SparsePauliOp.from_list([("XI", 1.0), ("IZ", 2.0j)])
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

        # infer object dtype due to ParameterExpression.
        param = Parameter("a")
        spp_op = SparsePauliOp.from_list([("X", param)])
        self.assertEqual(spp_op.coeffs.dtype, object)

    def test_from_sparse_list_automatically_determine_coeff_type_regression(self):
        """Regression test for issue where from_sparse_list failed to infer dtype."""
        param = Parameter("a")

        op1 = SparsePauliOp.from_sparse_list([("IX", [0, 1], 2), ("ZI", [0, 1], 3)], num_qubits=2)
        op1 *= param

        op2 = SparsePauliOp.from_sparse_list(op1.to_sparse_list(), num_qubits=2)

        self.assertEqual(op2.paulis, op1.paulis)
        self.assertEqual(op2.coeffs.dtype, object)

    def test_from_list_automatically_determine_coeff_type_regression(self):
        """Regression test for issue where from_list failed to infer dtype."""
        param = Parameter("a")

        op1 = SparsePauliOp.from_list([("IX", 2), ("ZI", 3)])
        op1 *= param

        op2 = SparsePauliOp.from_list(op1.to_list())

        self.assertEqual(op2.paulis, op1.paulis)
        self.assertEqual(op2.coeffs.dtype, object)

    def test_from_index_list_endianness(self):
        """Test the construction from index list has the right endianness."""
        spp_op = SparsePauliOp.from_sparse_list([("ZX", [1, 4], 1)], num_qubits=5)
        expected = Pauli("XIIZI")
        self.assertEqual(spp_op.paulis[0], expected)

    def test_from_index_list_raises(self):
        """Test from_sparse_list via Pauli + indices raises correctly, if number of qubits invalid."""
        with self.assertRaises(QiskitError):
            _ = SparsePauliOp.from_sparse_list([("Z", [2], 1)], 1)

    def test_from_index_list_same_index(self):
        """Test from_sparse_list via Pauli + number of qubits raises correctly, if indices duplicate."""
        with self.assertRaises(QiskitError):
            _ = SparsePauliOp.from_sparse_list([("ZZ", [0, 0], 1)], 2)
        with self.assertRaises(QiskitError):
            _ = SparsePauliOp.from_sparse_list([("ZI", [0, 0], 1)], 2)
        with self.assertRaises(QiskitError):
            _ = SparsePauliOp.from_sparse_list([("IZ", [0, 0], 1)], 2)

    def test_from_zip(self):
        """Test from_list method for zipped input."""
        labels = ["XXZ", "IXI", "YZZ", "III"]
        coeffs = [3.0, 5.5, -1j, 23.3333]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        np.testing.assert_array_equal(spp_op.coeffs, coeffs)
        self.assertEqual(spp_op.paulis, PauliList(labels))
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    @combine(iterable=[[], (), zip()], num_qubits=[1, 2, 3])
    def test_from_empty_iterable(self, iterable, num_qubits):
        """Test from_list method for empty iterable input."""
        with self.assertRaises(QiskitError):
            _ = SparsePauliOp.from_list(iterable)
        spp_op = SparsePauliOp.from_list(iterable, num_qubits=num_qubits)
        self.assertEqual(spp_op.paulis, PauliList("I" * num_qubits))
        np.testing.assert_array_equal(spp_op.coeffs, [0])
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    @combine(iterable=[[], (), zip()], num_qubits=[1, 2, 3])
    def test_from_sparse_empty_iterable(self, iterable, num_qubits):
        """Test from_sparse_list method for empty iterable input."""
        spp_op = SparsePauliOp.from_sparse_list(iterable, num_qubits)
        self.assertEqual(spp_op.paulis, PauliList("I" * num_qubits))
        np.testing.assert_array_equal(spp_op.coeffs, [0])
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    def test_to_matrix(self):
        """Test to_matrix method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        spp_op = SparsePauliOp(labels, coeffs)
        target = np.zeros((4, 4), dtype=complex)
        for coeff, label in zip(coeffs, labels):
            target += coeff * pauli_mat(label)
        np.testing.assert_array_equal(spp_op.to_matrix(), target)
        np.testing.assert_array_equal(spp_op.to_matrix(sparse=True).toarray(), target)

    def test_to_matrix_large(self):
        """Test to_matrix method with a large number of qubits."""
        reps = 5
        labels = ["XI" * reps, "YZ" * reps, "YY" * reps, "ZZ" * reps]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        spp_op = SparsePauliOp(labels, coeffs)
        size = 1 << 2 * reps
        target = np.zeros((size, size), dtype=complex)
        for coeff, label in zip(coeffs, labels):
            target += coeff * pauli_mat(label)
        np.testing.assert_array_equal(spp_op.to_matrix(), target)
        np.testing.assert_array_equal(spp_op.to_matrix(sparse=True).toarray(), target)

    def test_to_matrix_zero(self):
        """Test `to_matrix` with a zero operator."""
        num_qubits = 4
        zero_numpy = np.zeros((2**num_qubits, 2**num_qubits), dtype=np.complex128)
        zero = SparsePauliOp.from_list([], num_qubits=num_qubits)

        zero_dense = zero.to_matrix(sparse=False)
        np.testing.assert_array_equal(zero_dense, zero_numpy)

        zero_sparse = zero.to_matrix(sparse=True)
        self.assertIsInstance(zero_sparse, scipy.sparse.csr_matrix)
        np.testing.assert_array_equal(zero_sparse.todense(), zero_numpy)

    def test_to_matrix_parallel_vs_serial(self):
        """Parallel execution should produce the same results as serial execution up to
        floating-point associativity effects."""
        # Using powers-of-two coefficients to make floating-point arithmetic associative so we can
        # do bit-for-bit assertions.  Choose labels that have at least few overlapping locations.
        labels = ["XZIXYX", "YIIYXY", "ZZZIIZ", "IIIIII"]
        coeffs = [0.25, 0.125j, 0.5 - 0.25j, -0.125 + 0.5j]
        op = SparsePauliOp(labels, coeffs)
        np.testing.assert_array_equal(
            op.to_matrix(sparse=True, force_serial=False).toarray(),
            op.to_matrix(sparse=True, force_serial=True).toarray(),
        )
        np.testing.assert_array_equal(
            op.to_matrix(sparse=False, force_serial=False),
            op.to_matrix(sparse=False, force_serial=True),
        )

    def test_to_matrix_parameters(self):
        """Test to_matrix method for parameterized SparsePauliOp."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = np.array(ParameterVector("a", 4))
        spp_op = SparsePauliOp(labels, coeffs)
        target = np.zeros((4, 4), dtype=object)
        for coeff, label in zip(coeffs, labels):
            target += coeff * pauli_mat(label)
        np.testing.assert_array_equal(spp_op.to_matrix(), target)

    def test_to_operator(self):
        """Test to_operator method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        spp_op = SparsePauliOp(labels, coeffs)
        target = Operator(np.zeros((4, 4), dtype=complex))
        for coeff, label in zip(coeffs, labels):
            target = target + Operator(coeff * pauli_mat(label))
        self.assertEqual(spp_op.to_operator(), target)

    def test_to_list(self):
        """Test to_operator method."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = [-3, 4.4j, 0.2 - 0.1j, 66.12]
        op = SparsePauliOp(labels, coeffs)
        target = list(zip(labels, coeffs))
        self.assertEqual(op.to_list(), target)

    def test_to_list_parameters(self):
        """Test to_operator method with parameters."""
        labels = ["XI", "YZ", "YY", "ZZ"]
        coeffs = np.array(ParameterVector("a", 4))
        op = SparsePauliOp(labels, coeffs)
        target = list(zip(labels, coeffs))
        self.assertEqual(op.to_list(), target)

    def test_from_sparse_observable(self):
        """Test from a SparseObservable."""
        with self.subTest("zero(0)"):
            obs = SparseObservable.zero(0)
            expected = SparsePauliOp([""], coeffs=[0])
            self.assertEqual(expected, SparsePauliOp.from_sparse_observable(obs))

        with self.subTest("identity(0)"):
            obs = SparseObservable.identity(0)
            expected = SparsePauliOp([""], coeffs=[1])
            self.assertEqual(expected, SparsePauliOp.from_sparse_observable(obs))

        with self.subTest("zero(10)"):
            obs = SparseObservable.zero(10)
            expected = SparsePauliOp(["I" * 10], coeffs=[0])
            self.assertEqual(expected, SparsePauliOp.from_sparse_observable(obs))

        with self.subTest("identity(10)"):
            obs = SparseObservable.identity(10)
            expected = SparsePauliOp(["I" * 10], coeffs=[1])
            self.assertEqual(expected, SparsePauliOp.from_sparse_observable(obs))

        with self.subTest("XrZ"):
            obs = SparseObservable("XrZ")
            spo = SparsePauliOp.from_sparse_observable(obs)
            expected = SparsePauliOp(["XIZ", "XYZ"], coeffs=[0.5, 0.5])

            # we don't guarantee the order of Paulis, so check equality by comparing
            # the matrix representation and that all Pauli strings are present
            self.assertEqual(Operator(expected), Operator(spo))
            self.assertTrue(set(spo.paulis.to_labels()) == set(expected.paulis.to_labels()))

    def test_sparse_observable_roundtrip(self):
        """Test SPO -> OBS -> SPO."""
        with self.subTest(msg="empty"):
            op = SparsePauliOp([""], coeffs=[1])
            obs = SparseObservable.from_sparse_pauli_op(op)
            roundtrip = SparsePauliOp.from_sparse_observable(obs)
            self.assertEqual(op, roundtrip)

        with self.subTest(msg="zero"):
            op = SparsePauliOp(["I"], coeffs=[0])
            obs = SparseObservable.from_sparse_pauli_op(op)
            roundtrip = SparsePauliOp.from_sparse_observable(obs)
            self.assertEqual(op, roundtrip)

        with self.subTest(msg="identity"):
            op = SparsePauliOp(["I" * 25])
            obs = SparseObservable.from_sparse_pauli_op(op)
            roundtrip = SparsePauliOp.from_sparse_observable(obs)
            self.assertEqual(op, roundtrip)

        with self.subTest(msg="ising like"):
            op = SparsePauliOp(["ZZI", "IZZ", "IIX", "IXI", "YII"])
            obs = SparseObservable.from_sparse_pauli_op(op)
            roundtrip = SparsePauliOp.from_sparse_observable(obs)

            self.assertEqual(Operator(op), Operator(roundtrip))
            self.assertTrue(set(op.paulis.to_labels()) == set(roundtrip.paulis.to_labels()))


class TestSparsePauliOpIteration(QiskitTestCase):
    """Tests for SparsePauliOp iterators class."""

    def test_enumerate(self):
        """Test enumerate with SparsePauliOp."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_enumerate_parameters(self):
        """Test enumerate with SparsePauliOp with parameters."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array(ParameterVector("a", 6))
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_iter(self):
        """Test iter with SparsePauliOp."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(iter(op)):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_iter_parameters(self):
        """Test iter with SparsePauliOp with parameters."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array(ParameterVector("a", 6))
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(iter(op)):
            self.assertEqual(i, SparsePauliOp(labels[idx], coeffs[[idx]]))

    def test_label_iter(self):
        """Test SparsePauliOp label_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op.label_iter()):
            self.assertEqual(i, (labels[idx], coeffs[idx]))

    def test_label_iter_parameters(self):
        """Test SparsePauliOp label_iter method with parameters."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array(ParameterVector("a", 6))
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op.label_iter()):
            self.assertEqual(i, (labels[idx], coeffs[idx]))

    def test_matrix_iter(self):
        """Test SparsePauliOp dense matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op.matrix_iter()):
            np.testing.assert_array_equal(i, coeffs[idx] * pauli_mat(labels[idx]))

    def test_matrix_iter_parameters(self):
        """Test SparsePauliOp dense matrix_iter method. with parameters"""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array(ParameterVector("a", 6))
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op.matrix_iter()):
            np.testing.assert_array_equal(i, coeffs[idx] * pauli_mat(labels[idx]))

    def test_matrix_iter_sparse(self):
        """Test SparsePauliOp sparse matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        coeffs = np.array([1, 2, 3, 4, 5, 6])
        op = SparsePauliOp(labels, coeffs)
        for idx, i in enumerate(op.matrix_iter(sparse=True)):
            np.testing.assert_array_equal(i.toarray(), coeffs[idx] * pauli_mat(labels[idx]))


def bind_parameters_to_one(array):
    """Bind parameters to one. The purpose of using this method is to bind some value and
    use ``assert_allclose``, since it is impossible to verify equivalence in the case of
    numerical errors with parameters existing.
    """

    def bind_one(a):
        parameters = a.parameters
        return complex(a.bind(dict(zip(parameters, [1] * len(parameters)))))

    return np.vectorize(bind_one, otypes=[complex])(array)


@ddt.ddt
class TestSparsePauliOpMethods(QiskitTestCase):
    """Tests for SparsePauliOp operator methods."""

    RNG = np.random.default_rng(1994)

    def setUp(self):
        super().setUp()

        self.parameter_names = (f"param_{x}" for x in it.count())

    def random_spp_op(self, num_qubits, num_terms, use_parameters=False):
        """Generate a pseudo-random SparsePauliOp."""
        if use_parameters:
            coeffs = np.array(ParameterVector(next(self.parameter_names), num_terms))
        else:
            coeffs = self.RNG.uniform(-1, 1, size=num_terms) + 1j * self.RNG.uniform(
                -1, 1, size=num_terms
            )
        labels = [
            "".join(self.RNG.choice(["I", "X", "Y", "Z"], size=num_qubits))
            for _ in range(num_terms)
        ]
        return SparsePauliOp(labels, coeffs)

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_conjugate(self, num_qubits, use_parameters):
        """Test conjugate method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op.to_matrix().conjugate()
        op = spp_op.conjugate()
        value = op.to_matrix()
        np.testing.assert_array_equal(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_transpose(self, num_qubits, use_parameters):
        """Test transpose method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op.to_matrix().transpose()
        op = spp_op.transpose()
        value = op.to_matrix()
        np.testing.assert_array_equal(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_adjoint(self, num_qubits, use_parameters):
        """Test adjoint method for {num_qubits}-qubits."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op.to_matrix().transpose().conjugate()
        op = spp_op.adjoint()
        value = op.to_matrix()
        np.testing.assert_array_equal(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_compose(self, num_qubits, use_parameters):
        """Test {num_qubits}-qubit compose methods."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op2.to_matrix() @ spp_op1.to_matrix()

        op = spp_op1.compose(spp_op2)
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

        op = spp_op1 & spp_op2
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_dot(self, num_qubits, use_parameters):
        """Test {num_qubits}-qubit dot methods."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op1.to_matrix() @ spp_op2.to_matrix()

        op = spp_op1.dot(spp_op2)
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

        op = spp_op1 @ spp_op2
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3])
    def test_qargs_compose(self, num_qubits):
        """Test 3-qubit compose method with {num_qubits}-qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2**3)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1).compose(Operator(spp_op2), qargs=qargs)

        op = spp_op1.compose(spp_op2, qargs=qargs)
        value = op.to_operator()
        self.assertEqual(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

        op = spp_op1 & spp_op2(qargs)
        value = op.to_operator()
        self.assertEqual(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3])
    def test_qargs_dot(self, num_qubits):
        """Test 3-qubit dot method with {num_qubits}-qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2**3)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1).dot(Operator(spp_op2), qargs=qargs)

        op = spp_op1.dot(spp_op2, qargs=qargs)
        value = op.to_operator()
        self.assertEqual(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits1=[1, 2, 3], num_qubits2=[1, 2, 3], use_parameters=[True, False])
    def test_tensor(self, num_qubits1, num_qubits2, use_parameters):
        """Test tensor method for {num_qubits1} and {num_qubits2} qubits."""
        spp_op1 = self.random_spp_op(num_qubits1, 2**num_qubits1, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits2, 2**num_qubits2, use_parameters)
        target = np.kron(spp_op1.to_matrix(), spp_op2.to_matrix())
        op = spp_op1.tensor(spp_op2)
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits1=[1, 2, 3], num_qubits2=[1, 2, 3], use_parameters=[True, False])
    def test_expand(self, num_qubits1, num_qubits2, use_parameters):
        """Test expand method for {num_qubits1} and {num_qubits2} qubits."""
        spp_op1 = self.random_spp_op(num_qubits1, 2**num_qubits1, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits2, 2**num_qubits2, use_parameters)
        target = np.kron(spp_op2.to_matrix(), spp_op1.to_matrix())
        op = spp_op1.expand(spp_op2)
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_add(self, num_qubits, use_parameters):
        """Test + method for {num_qubits} qubits."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op1.to_matrix() + spp_op2.to_matrix()
        op = spp_op1 + spp_op2
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_sub(self, num_qubits, use_parameters):
        """Test + method for {num_qubits} qubits."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        target = spp_op1.to_matrix() - spp_op2.to_matrix()
        op = spp_op1 - spp_op2
        value = op.to_matrix()
        if use_parameters:
            value = bind_parameters_to_one(value)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3])
    def test_add_qargs(self, num_qubits):
        """Test + method for 3 qubits with {num_qubits} qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2**3)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1) + Operator(spp_op2)(qargs)
        op = spp_op1 + spp_op2(qargs)
        value = op.to_operator()
        self.assertEqual(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3])
    def test_sub_qargs(self, num_qubits):
        """Test - method for 3 qubits with {num_qubits} qubit qargs."""
        spp_op1 = self.random_spp_op(3, 2**3)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits)
        qargs = self.RNG.choice(3, size=num_qubits, replace=False).tolist()
        target = Operator(spp_op1) - Operator(spp_op2)(qargs)
        op = spp_op1 - spp_op2(qargs)
        value = op.to_operator()
        self.assertEqual(value, target)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(
        num_qubits=[1, 2, 3],
        value=[
            0,
            1,
            1j,
            -3 + 4.4j,
            np.int64(2),
            Parameter("x"),
            0 * Parameter("x"),
            (-2 + 1.7j) * Parameter("x"),
        ],
        param=[None, "a"],
    )
    def test_mul(self, num_qubits, value, param):
        """Test * method for {num_qubits} qubits and value {value}."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits, param)
        target = value * spp_op.to_matrix()
        op = value * spp_op
        value_mat = op.to_matrix()
        has_parameters = isinstance(value, ParameterExpression) or param is not None
        if value != 0 and has_parameters:
            value_mat = bind_parameters_to_one(value_mat)
            target = bind_parameters_to_one(target)
        if value == 0:
            np.testing.assert_array_equal(value_mat, target.astype(complex))
        else:
            np.testing.assert_allclose(value_mat, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))
        target = spp_op.to_matrix() * value
        op = spp_op * value
        value_mat = op.to_matrix()
        if value != 0 and has_parameters:
            value_mat = bind_parameters_to_one(value_mat)
            target = bind_parameters_to_one(target)
        if value == 0:
            np.testing.assert_array_equal(value_mat, target.astype(complex))
        else:
            np.testing.assert_allclose(value_mat, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    @combine(num_qubits=[1, 2, 3], value=[1, 1j, -3 + 4.4j], param=[None, "a"])
    def test_div(self, num_qubits, value, param):
        """Test / method for {num_qubits} qubits and value {value}."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits, param)
        target = spp_op.to_matrix() / value
        op = spp_op / value
        value_mat = op.to_matrix()
        if param is not None:
            value_mat = bind_parameters_to_one(value_mat)
            target = bind_parameters_to_one(target)
        np.testing.assert_allclose(value_mat, target, atol=1e-8)
        np.testing.assert_array_equal(op.paulis.phase, np.zeros(op.size))

    def test_simplify(self):
        """Test simplify method"""
        coeffs = [3 + 1j, -3 - 1j, 0, 4, -5, 2.2, -1.1j]
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        spp_op = SparsePauliOp.from_list(zip(labels, coeffs))
        simplified_op = spp_op.simplify()
        target_coeffs = [-1, 2.2 - 1.1j]
        target_labels = ["III", "XXX"]
        target_op = SparsePauliOp.from_list(zip(target_labels, target_coeffs))
        self.assertEqual(simplified_op, target_op)
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))
        self.assertEqual(spp_op.coeffs.dtype, np.complex128)

    @combine(num_qubits=[1, 2, 3, 4], num_adds=[0, 1, 2, 3])
    def test_simplify2(self, num_qubits, num_adds):
        """Test simplify method for {num_qubits} qubits with {num_adds} `add` calls."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits)
        for _ in range(num_adds):
            spp_op += spp_op
        simplified_op = spp_op.simplify()
        value = Operator(simplified_op)
        target = Operator(spp_op)
        self.assertEqual(value, target)
        np.testing.assert_array_equal(spp_op.paulis.phase, np.zeros(spp_op.size))
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))

    @combine(num_qubits=[1, 2, 3, 4])
    def test_simplify_zero(self, num_qubits):
        """Test simplify method for {num_qubits} qubits with zero operators."""
        spp_op = self.random_spp_op(num_qubits, 2**num_qubits)
        zero_op = spp_op - spp_op
        simplified_op = zero_op.simplify()
        value = Operator(simplified_op)
        target = Operator(zero_op)
        self.assertEqual(value, target)
        np.testing.assert_array_equal(simplified_op.coeffs, [0])
        np.testing.assert_array_equal(zero_op.paulis.phase, np.zeros(zero_op.size))
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))

    def test_simplify_parameters(self):
        """Test simplify methods for parameterized SparsePauliOp."""
        a = Parameter("a")
        coeffs = np.array([a, -a, 0, a, a, a, 2 * a])
        labels = ["IXI", "IXI", "ZZZ", "III", "III", "XXX", "XXX"]
        spp_op = SparsePauliOp(labels, coeffs)
        simplified_op = spp_op.simplify()
        target_coeffs = np.array([2 * a, 3 * a])
        target_labels = ["III", "XXX"]
        target_op = SparsePauliOp(target_labels, target_coeffs)
        self.assertEqual(simplified_op, target_op)
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))

    def test_simplify_complex_parameters(self):
        """Test calling simplify when a parameter has a complex coefficient."""
        a = Parameter("a")
        b = Parameter("b")
        coeffs = np.array([a, 1j * a, 1j * b, -1j * b])
        labels = ["X", "X", "Z", "Z"]
        spp_op = SparsePauliOp(labels, coeffs)
        simplified_op = spp_op.simplify()
        target_coeffs = np.array([(1 + 1j) * a])
        target_labels = ["X"]
        target_op = SparsePauliOp(target_labels, target_coeffs)
        self.assertEqual(simplified_op, target_op)
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))

    def test_simplify_multiparameter_cancellation(self):
        """Multi-parameter coefficients that cancel to zero are dropped without sympy."""
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        # The four X terms sum to a + b - a - b == 0 and must be removed;
        # the Z term must survive.
        coeffs = np.array([a, b, -a, -b, c])
        labels = ["X", "X", "X", "X", "Z"]
        spp_op = SparsePauliOp(labels, coeffs)
        simplified_op = spp_op.simplify()
        target_op = SparsePauliOp(["Z"], np.array([c]))
        self.assertEqual(simplified_op, target_op)
        np.testing.assert_array_equal(simplified_op.paulis.phase, np.zeros(simplified_op.size))

    def test_sort(self):
        """Test sort method."""
        with self.assertRaises(QiskitError):
            target = SparsePauliOp([], [])

        with self.subTest(msg="1 qubit real number"):
            target = SparsePauliOp(
                ["I", "I", "I", "I"], [-3.0 + 0.0j, 1.0 + 0.0j, 2.0 + 0.0j, 4.0 + 0.0j]
            )
            value = SparsePauliOp(["I", "I", "I", "I"], [1, 2, -3, 4]).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit complex"):
            target = SparsePauliOp(
                ["I", "I", "I", "I"], [-1.0 + 0.0j, 0.0 - 1.0j, 0.0 + 1.0j, 1.0 + 0.0j]
            )
            value = SparsePauliOp(
                ["I", "I", "I", "I"], [1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j, -1.0 + 0.0j]
            ).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit Pauli I, X, Y, Z"):
            target = SparsePauliOp(
                ["I", "X", "Y", "Z"], [-1.0 + 2.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 - 4.0j]
            )
            value = SparsePauliOp(
                ["Y", "X", "Z", "I"], [2.0 + 0.0j, 1.0 + 0.0j, 3.0 - 4.0j, -1.0 + 2.0j]
            ).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit weight order"):
            target = SparsePauliOp(
                ["I", "X", "Y", "Z"], [-1.0 + 2.0j, 1.0 + 0.0j, 2.0 + 0.0j, 3.0 - 4.0j]
            )
            value = SparsePauliOp(
                ["Y", "X", "Z", "I"], [2.0 + 0.0j, 1.0 + 0.0j, 3.0 - 4.0j, -1.0 + 2.0j]
            ).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit multi Pauli"):
            target = SparsePauliOp(
                ["I", "I", "I", "I", "X", "X", "Y", "Z"],
                [
                    -1.0 + 2.0j,
                    1.0 + 0.0j,
                    2.0 + 0.0j,
                    3.0 - 4.0j,
                    -1.0 + 4.0j,
                    -1.0 + 5.0j,
                    -1.0 + 3.0j,
                    -1.0 + 2.0j,
                ],
            )
            value = SparsePauliOp(
                ["I", "I", "I", "I", "X", "Z", "Y", "X"],
                [
                    2.0 + 0.0j,
                    1.0 + 0.0j,
                    3.0 - 4.0j,
                    -1.0 + 2.0j,
                    -1.0 + 5.0j,
                    -1.0 + 2.0j,
                    -1.0 + 3.0j,
                    -1.0 + 4.0j,
                ],
            ).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit standard order"):
            target = SparsePauliOp(
                ["II", "XI", "XX", "XX", "XX", "XY", "XZ", "YI"],
                [
                    4.0 + 0.0j,
                    7.0 + 0.0j,
                    2.0 + 1.0j,
                    2.0 + 2.0j,
                    3.0 + 0.0j,
                    6.0 + 0.0j,
                    5.0 + 0.0j,
                    3.0 + 0.0j,
                ],
            )
            value = SparsePauliOp(
                ["XX", "XX", "XX", "YI", "II", "XZ", "XY", "XI"],
                [
                    2.0 + 1.0j,
                    2.0 + 2.0j,
                    3.0 + 0.0j,
                    3.0 + 0.0j,
                    4.0 + 0.0j,
                    5.0 + 0.0j,
                    6.0 + 0.0j,
                    7.0 + 0.0j,
                ],
            ).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit weight order"):
            target = SparsePauliOp(
                ["II", "XI", "YI", "XX", "XX", "XX", "XY", "XZ"],
                [
                    4.0 + 0.0j,
                    7.0 + 0.0j,
                    3.0 + 0.0j,
                    2.0 + 1.0j,
                    2.0 + 2.0j,
                    3.0 + 0.0j,
                    6.0 + 0.0j,
                    5.0 + 0.0j,
                ],
            )
            value = SparsePauliOp(
                ["XX", "XX", "XX", "YI", "II", "XZ", "XY", "XI"],
                [
                    2.0 + 1.0j,
                    2.0 + 2.0j,
                    3.0 + 0.0j,
                    3.0 + 0.0j,
                    4.0 + 0.0j,
                    5.0 + 0.0j,
                    6.0 + 0.0j,
                    7.0 + 0.0j,
                ],
            ).sort(weight=True)
            self.assertEqual(target, value)

    def test_chop(self):
        """Test chop, which individually truncates real and imaginary parts of the coeffs."""
        eps = 1e-10
        op = SparsePauliOp(
            ["XYZ", "ZII", "ZII", "YZY"], coeffs=[eps + 1j * eps, 1 + 1j * eps, eps + 1j, 1 + 1j]
        )
        simplified = op.chop(tol=eps)
        expected_coeffs = [1, 1j, 1 + 1j]
        expected_paulis = ["ZII", "ZII", "YZY"]
        self.assertListEqual(simplified.coeffs.tolist(), expected_coeffs)
        self.assertListEqual(simplified.paulis.to_labels(), expected_paulis)

    def test_chop_all(self):
        """Test that chop returns an identity operator with coeff 0 if all coeffs are chopped."""
        eps = 1e-10
        op = SparsePauliOp(["X", "Z"], coeffs=[eps, eps])
        simplified = op.chop(tol=eps)
        expected = SparsePauliOp(["I"], coeffs=[0.0])
        self.assertEqual(simplified, expected)

    @combine(num_qubits=[1, 2, 3, 4], num_ops=[1, 2, 3, 4], param=[None, "a"])
    def test_sum(self, num_qubits, num_ops, param):
        """Test sum method for {num_qubits} qubits with {num_ops} operators."""
        ops = [
            self.random_spp_op(
                num_qubits, 2**num_qubits, param if param is None else f"{param}_{i}"
            )
            for i in range(num_ops)
        ]
        sum_op = SparsePauliOp.sum(ops)
        value = sum_op.to_matrix()
        target_operator = sum((op.to_matrix() for op in ops[1:]), ops[0].to_matrix())
        if param is not None:
            value = bind_parameters_to_one(value)
            target_operator = bind_parameters_to_one(target_operator)
        np.testing.assert_allclose(value, target_operator, atol=1e-8)
        target_spp_op = sum((op for op in ops[1:]), ops[0])
        self.assertEqual(sum_op, target_spp_op)
        np.testing.assert_array_equal(sum_op.paulis.phase, np.zeros(sum_op.size))

    def test_sum_error(self):
        """Test sum method with invalid cases."""
        with self.assertRaises(QiskitError):
            SparsePauliOp.sum([])
        with self.assertRaises(QiskitError):
            ops = [self.random_spp_op(num_qubits, 2**num_qubits) for num_qubits in [1, 2]]
            SparsePauliOp.sum(ops)
        with self.assertRaises(QiskitError):
            SparsePauliOp.sum([1, 2])

    @combine(num_qubits=[1, 2, 3, 4], use_parameters=[True, False])
    def test_eq(self, num_qubits, use_parameters):
        """Test __eq__ method for {num_qubits} qubits."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        spp_op3 = self.random_spp_op(num_qubits, 2**num_qubits, use_parameters)
        zero = spp_op3 - spp_op3
        self.assertEqual(spp_op1, spp_op1)
        self.assertEqual(spp_op2, spp_op2)
        self.assertNotEqual(spp_op1, spp_op1 + zero)
        self.assertNotEqual(spp_op2, spp_op2 + zero)
        if spp_op1 != spp_op2:
            self.assertNotEqual(spp_op1 + spp_op2, spp_op2 + spp_op1)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_equiv(self, num_qubits):
        """Test equiv method for {num_qubits} qubits."""
        spp_op1 = self.random_spp_op(num_qubits, 2**num_qubits)
        spp_op2 = self.random_spp_op(num_qubits, 2**num_qubits)
        spp_op3 = self.random_spp_op(num_qubits, 2**num_qubits)
        spp_op4 = self.random_spp_op(num_qubits, 2**num_qubits)
        zero = spp_op3 - spp_op3
        zero2 = spp_op4 - spp_op4
        self.assertTrue(spp_op1.equiv(spp_op1))
        self.assertTrue(spp_op1.equiv(spp_op1 + zero))
        self.assertTrue(spp_op2.equiv(spp_op2))
        self.assertTrue(spp_op2.equiv(spp_op2 + zero))
        self.assertTrue(zero.equiv(zero2))
        self.assertTrue((zero + zero2).equiv(zero2 + zero))
        self.assertTrue((zero2 + zero).equiv(zero + zero2))
        self.assertTrue((spp_op1 + spp_op2).equiv(spp_op2 + spp_op1))
        self.assertTrue((spp_op2 + spp_op1).equiv(spp_op1 + spp_op2))
        self.assertTrue((spp_op1 - spp_op1).equiv(spp_op2 - spp_op2))
        self.assertTrue((2 * spp_op1).equiv(spp_op1 + spp_op1))
        self.assertTrue((2 * spp_op2).equiv(spp_op2 + spp_op2))
        if not spp_op1.equiv(zero):
            self.assertFalse(spp_op1.equiv(spp_op1 + spp_op1))
        if not spp_op2.equiv(zero):
            self.assertFalse(spp_op2.equiv(spp_op2 + spp_op2))

    def test_equiv_atol(self):
        """Test equiv method with atol."""
        op1 = SparsePauliOp.from_list([("X", 1), ("Y", 2)])
        op2 = op1 + 1e-7 * SparsePauliOp.from_list([("I", 1)])
        self.assertFalse(op1.equiv(op2))
        self.assertTrue(op1.equiv(op2, atol=1e-7))
        self.assertEqual(op1.coeffs.dtype, np.complex128)

    def test_eq_equiv(self):
        """Test __eq__ and equiv methods with some specific cases."""
        with self.subTest("shuffled"):
            spp_op1 = SparsePauliOp.from_list([("X", 1), ("Y", 2)])
            spp_op2 = SparsePauliOp.from_list([("Y", 2), ("X", 1)])
            self.assertNotEqual(spp_op1, spp_op2)
            self.assertTrue(spp_op1.equiv(spp_op2))

        with self.subTest("w/ zero"):
            spp_op1 = SparsePauliOp.from_list([("X", 1), ("Y", 1)])
            spp_op2 = SparsePauliOp.from_list([("X", 1), ("Y", 1), ("Z", 0)])
            self.assertNotEqual(spp_op1, spp_op2)
            self.assertTrue(spp_op1.equiv(spp_op2))

    @combine(parameterized=[True, False], qubit_wise=[True, False])
    def test_noncommutation_graph(self, parameterized, qubit_wise):
        """Test noncommutation graph."""

        def commutes(left: Pauli, right: Pauli, qubit_wise: bool) -> bool:
            if len(left) != len(right):
                return False
            if not qubit_wise:
                return left.commutes(right)
            else:
                # qubit-wise commuting check
                vec_l = left.z + 2 * left.x
                vec_r = right.z + 2 * right.x
                qubit_wise_comparison = (vec_l * vec_r) * (vec_l - vec_r)
                return np.all(qubit_wise_comparison == 0)

        input_labels = ["IX", "IY", "IZ", "XX", "YY", "ZZ", "XY", "iYX", "ZX", "-iZY", "XZ", "YZ"]
        np.random.shuffle(input_labels)
        if parameterized:
            coeffs = np.array(ParameterVector("a", len(input_labels)))
        else:
            coeffs = np.random.random(len(input_labels)) + np.random.random(len(input_labels)) * 1j
        sparse_pauli_list = SparsePauliOp(input_labels, coeffs)
        graph = sparse_pauli_list.noncommutation_graph(qubit_wise)

        expected = rx.PyGraph()
        expected.add_nodes_from(range(len(input_labels)))
        edges = [
            (ia, ib)
            for (ia, a), (ib, b) in it.combinations(enumerate(input_labels), 2)
            if not commutes(Pauli(a), Pauli(b), qubit_wise)
        ]
        expected.add_edges_from_no_data(edges)

        self.assertTrue(rx.is_isomorphic(graph, expected))

    @combine(parameterized=[True, False], qubit_wise=[True, False])
    def test_group_commuting(self, parameterized, qubit_wise):
        """Test general grouping of commuting operators."""

        def commutes(left: Pauli, right: Pauli, qubit_wise: bool) -> bool:
            if len(left) != len(right):
                return False
            if not qubit_wise:
                return left.commutes(right)
            else:
                # qubit-wise commuting check
                vec_l = left.z + 2 * left.x
                vec_r = right.z + 2 * right.x
                qubit_wise_comparison = (vec_l * vec_r) * (vec_l - vec_r)
                return np.all(qubit_wise_comparison == 0)

        input_labels = ["IX", "IY", "IZ", "XX", "YY", "ZZ", "XY", "YX", "ZX", "ZY", "XZ", "YZ"]
        np.random.shuffle(input_labels)
        if parameterized:
            coeffs = np.array(ParameterVector("a", len(input_labels)))
        else:
            coeffs = np.random.random(len(input_labels)) + np.random.random(len(input_labels)) * 1j
        sparse_pauli_list = SparsePauliOp(input_labels, coeffs)
        groups = sparse_pauli_list.group_commuting(qubit_wise)
        # checking that every input Pauli in sparse_pauli_list is in a group in the output
        output_labels = [pauli.to_label() for group in groups for pauli in group.paulis]
        self.assertListEqual(sorted(output_labels), sorted(input_labels))
        # checking that every coeffs are grouped according to sparse_pauli_list group
        paulis_coeff_dict = dict(
            sum(  # noqa: RUF017
                [list(zip(group.paulis.to_labels(), group.coeffs)) for group in groups], []
            )
        )
        self.assertDictEqual(dict(zip(input_labels, coeffs)), paulis_coeff_dict)

        # Within each group, every operator commutes with every other operator.
        for group in groups:
            self.assertTrue(
                all(
                    commutes(pauli1, pauli2, qubit_wise)
                    for pauli1, pauli2 in it.combinations(group.paulis, 2)
                )
            )
        # For every pair of groups, at least one element from one group does not commute with
        # at least one element of the other.
        for group1, group2 in it.combinations(groups, 2):
            self.assertFalse(
                all(
                    commutes(group1_pauli, group2_pauli, qubit_wise)
                    for group1_pauli, group2_pauli in it.product(group1.paulis, group2.paulis)
                )
            )

    def test_dot_real(self):
        """Test dot for real coefficients."""
        x = SparsePauliOp("X", np.array([1]))
        y = SparsePauliOp("Y", np.array([1]))
        iz = SparsePauliOp("Z", 1j)
        self.assertEqual(x.dot(y), iz)

    def test_get_parameters(self):
        """Test getting the parameters."""
        x, y = Parameter("x"), Parameter("y")
        op = SparsePauliOp(["X", "Y", "Z"], coeffs=[1, x, x * y])

        with self.subTest(msg="all parameters"):
            self.assertEqual(ParameterView([x, y]), op.parameters)

        op.assign_parameters({y: 2}, inplace=True)
        with self.subTest(msg="after partial binding"):
            self.assertEqual(ParameterView([x]), op.parameters)

    def test_assign_parameters(self):
        """Test assign parameters."""
        x, y = Parameter("x"), Parameter("y")
        op = SparsePauliOp(["X", "Y", "Z"], coeffs=[1, x, x * y])

        # partial binding inplace
        op.assign_parameters({y: 2}, inplace=True)
        with self.subTest(msg="partial binding"):
            self.assertListEqual(op.coeffs.tolist(), [1, x, 2 * x])

        # bind via array
        bound = op.assign_parameters([3])
        with self.subTest(msg="fully bound"):
            self.assertEqual(bound.coeffs.dtype, np.complex128)
            self.assertTrue(np.allclose(bound.coeffs, [1, 3, 6]))

    def test_paulis_setter_rejects_bad_inputs(self):
        """Test that the setter for `paulis` rejects different-sized inputs."""
        op = SparsePauliOp(["XY", "ZX"], coeffs=[1, 1j])
        with self.assertRaisesRegex(ValueError, "incorrect number of qubits"):
            op.paulis = PauliList([Pauli("X"), Pauli("Y")])
        with self.assertRaisesRegex(ValueError, "incorrect number of operators"):
            op.paulis = PauliList([Pauli("XY"), Pauli("ZX"), Pauli("YZ")])

    def test_paulis_setter_absorbs_phase(self):
        """Test that the setter for `paulis` absorbs `paulis.phase` to `self.coeffs`."""
        coeffs_init = np.array([1, 1j])
        op = SparsePauliOp(["XY", "ZX"], coeffs=coeffs_init)
        paulis_new = PauliList(["-1jXY", "1jZX"])
        op.paulis = paulis_new
        # Paulis attribute should have no phase:
        self.assertEqual(op.paulis, PauliList(["XY", "ZX"]))
        # Coeffs attribute should now include that phase:
        self.assertTrue(np.allclose(op.coeffs, coeffs_init * np.array([-1j, 1j])))
        # The phase of the input array is now zero:
        self.assertTrue(np.allclose(paulis_new.phase, np.array([0, 0])))

    def test_paulis_setter_absorbs_phase_2(self):
        """Test that `paulis` setter followed by `simplify()` handle phase OK."""
        spo = SparsePauliOp(["X", "X"])
        spo.paulis = ["X", "-X"]
        self.assertEqual(spo.simplify(), SparsePauliOp(["I"], coeffs=[0.0 + 0.0j]))

    def test_apply_layout_with_transpile(self):
        """Test the apply_layout method with a transpiler layout."""
        psi = efficient_su2(4, reps=4, entanglement="circular")
        op = SparsePauliOp.from_list([("IIII", 1), ("IZZZ", 2), ("XXXI", 3)])
        backend = GenericBackendV2(num_qubits=7)
        transpiled_psi = transpile(psi, backend, optimization_level=3, seed_transpiler=12345)
        permuted_op = op.apply_layout(transpiled_psi.layout)
        identity_op = SparsePauliOp("I" * 7)
        initial_layout = transpiled_psi.layout.initial_index_layout(filter_ancillas=True)
        final_layout = transpiled_psi.layout.routing_permutation()
        qargs = [final_layout[x] for x in initial_layout]
        expected_op = identity_op.compose(op, qargs=qargs)
        self.assertNotEqual(op, permuted_op)
        self.assertEqual(permuted_op, expected_op)

    def test_permute_sparse_pauli_op_estimator_example(self):
        """Test using the apply_layout method with an estimator workflow."""
        psi = efficient_su2(4, reps=4, entanglement="circular")
        op = SparsePauliOp.from_list([("IIII", 1), ("IZZZ", 2), ("XXXI", 3)])
        backend = GenericBackendV2(num_qubits=7, seed=0)
        backend.set_options(seed_simulator=123)
        estimator = BackendEstimatorV2(backend=backend)
        thetas = list(range(len(psi.parameters)))
        transpiled_psi = transpile(psi, backend, optimization_level=3)
        permuted_op = op.apply_layout(transpiled_psi.layout)
        job = estimator.run([(transpiled_psi, permuted_op, thetas)])
        res = job.result()[0].data.evs
        if optionals.HAS_AER:
            np.testing.assert_allclose(res, [1.419922], rtol=0.5, atol=0.2)
        else:
            np.testing.assert_allclose(res, [1.660156], rtol=0.5, atol=0.2)

    def test_apply_layout_invalid_qubits_list(self):
        """Test that apply_layout with an invalid qubit count raises."""
        op = SparsePauliOp.from_list([("YI", 2), ("XI", 1)])
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 1], 1)

    def test_apply_layout_invalid_layout_list(self):
        """Test that apply_layout with an invalid layout list raises."""
        op = SparsePauliOp.from_list([("YI", 2), ("IX", 1)])
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 3], 2)

    def test_apply_layout_invalid_layout_list_no_num_qubits(self):
        """Test that apply_layout with an invalid layout list raises."""
        op = SparsePauliOp.from_list([("YI", 2), ("XI", 1)])
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 2])

    def test_apply_layout_layout_list_no_num_qubits(self):
        """Test apply_layout with a layout list and no qubit count"""
        op = SparsePauliOp.from_list([("YI", 2), ("XI", 1)])
        res = op.apply_layout([1, 0])
        self.assertEqual(SparsePauliOp.from_list([("IY", 2), ("IX", 1)]), res)

    def test_apply_layout_layout_list_and_num_qubits(self):
        """Test apply_layout with a layout list and qubit count"""
        op = SparsePauliOp.from_list([("YI", 2), ("XI", 1)])
        res = op.apply_layout([4, 0], 5)
        self.assertEqual(SparsePauliOp.from_list([("IIIIY", 2), ("IIIIX", 1)]), res)

    def test_apply_layout_null_layout_no_num_qubits(self):
        """Test apply_layout with a null layout"""
        op = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        res = op.apply_layout(layout=None)
        self.assertEqual(op, res)

    def test_apply_layout_null_layout_and_num_qubits(self):
        """Test apply_layout with a null layout and num_qubits provided."""
        op = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        res = op.apply_layout(layout=None, num_qubits=5)
        # this should expand the operator
        self.assertEqual(SparsePauliOp.from_list([("IIIII", 1), ("IIIIZ", 2), ("IIIXI", 3)]), res)

    def test_apply_layout_null_layout_invalid_num_qubits(self):
        """Test apply_layout with a null layout and num_qubits smaller than capable"""
        op = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=None, num_qubits=1)

    def test_apply_layout_negative_indices(self):
        """Test apply_layout with negative indices"""
        op = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=[-1, 0], num_qubits=3)

    def test_apply_layout_duplicate_indices(self):
        """Test apply_layout with duplicate indices"""
        op = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=[0, 0], num_qubits=3)

    @combine(layout=[None, []])
    def test_apply_layout_zero_qubit(self, layout):
        """Test apply_layout with a zero-qubit operator"""
        with self.subTest("default"):
            op = SparsePauliOp("")
            res = op.apply_layout(layout=layout, num_qubits=5)
            self.assertEqual(SparsePauliOp("IIIII"), res)
        with self.subTest("coeff"):
            op = SparsePauliOp("", 2)
            res = op.apply_layout(layout=layout, num_qubits=5)
            self.assertEqual(SparsePauliOp("IIIII", 2), res)
        with self.subTest("multiple ops"):
            op = SparsePauliOp.from_list([("", 1), ("", 2)])
            res = op.apply_layout(layout=layout, num_qubits=5)
            self.assertEqual(SparsePauliOp.from_list([("IIIII", 1), ("IIIII", 2)]), res)

    def test_simplify_sum_above_tolerance(self):
        """Test that simplify sums duplicates before applying atol threshold."""
        # Each coeff < atol, but sum > atol
        op = SparsePauliOp(["XX"] * 10, [1e-9] * 10)
        res = op.simplify(atol=2e-9)
        self.assertEqual(SparsePauliOp.from_list([("XX", 1.0e-08 + 0.0j)]), res)

    def test_simplify_sum_below_tolerance(self):
        """Test that simplify sums duplicates before applying atol threshold."""
        # Each coeff > atol, but sum < atol
        op = SparsePauliOp(["YY", "YY"], [1e-6, -1e-6])
        res = op.simplify(atol=1e-7)
        self.assertEqual(SparsePauliOp(["II"], [0j]), res)

    def test_is_unitary_tolerance(self):
        """Test that is_unitary respects the input tolerance values.

        This test verifies that the tolerance parameters are properly passed through
        to the internal operations. See issue #14107.
        """
        # Create a matrix that's approximately unitary but not exactly
        a = np.array(
            [
                [-9.9801135e-01 + 6.3036762e-02j, 5.6710692e-06 + 8.1099635e-05j],
                [5.6710610e-06 + 8.1099643e-05j, -9.9707150e-01 + 7.6473624e-02j],
            ]
        )

        # Verify the matrix is approximately unitary
        identity = a @ a.conj().T
        np.testing.assert_allclose(identity, np.eye(2), atol=1e-5, rtol=1e-3)

        # Create SparsePauliOp from the matrix
        op = SparsePauliOp.from_operator(a)

        # Test with tolerance that should pass
        self.assertTrue(
            op.is_unitary(atol=1e-5, rtol=1e-3),
            "Operator should be considered unitary with given tolerance",
        )

        # Test with very small tolerance that should fail
        self.assertFalse(
            op.is_unitary(atol=1e-10, rtol=1e-10),
            "Operator should not be considered unitary with very small tolerance",
        )


if __name__ == "__main__":
    unittest.main()
