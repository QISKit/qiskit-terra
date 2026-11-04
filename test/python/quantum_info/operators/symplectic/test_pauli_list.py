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

"""Tests for PauliList class."""

import itertools
import unittest

import numpy as np
import rustworkx as rx
from ddt import ddt
from scipy.sparse import csr_matrix

from qiskit import QiskitError
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    HGate,
    IGate,
    ECRGate,
    SdgGate,
    SGate,
    SwapGate,
    iSwapGate,
    XGate,
    YGate,
    ZGate,
    SXGate,
    SXdgGate,
    RXGate,
    RYGate,
    RZGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.quantum_info.operators import (
    Clifford,
    Operator,
    Pauli,
    PauliList,
)
from qiskit.quantum_info import random_clifford, random_pauli_list
from test import combine
from test import QiskitTestCase

from .test_pauli import pauli_group_labels


def pauli_mat(label):
    """Return Pauli matrix from a Pauli label"""
    mat = np.eye(1, dtype=complex)
    if label[0:2] == "-i":
        mat *= -1j
        label = label[2:]
    elif label[0] == "-":
        mat *= -1
        label = label[1:]
    elif label[0] == "i":
        mat *= 1j
        label = label[1:]
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


class TestPauliListInit(QiskitTestCase):
    """Tests for PauliList initialization."""

    def test_array_init(self):
        """Test array initialization."""
        # Matrix array initialization

        with self.subTest(msg="Empty array"):
            x = np.array([], dtype=bool).reshape((1, 0))
            z = np.array([], dtype=bool).reshape((1, 0))
            pauli_list = PauliList.from_symplectic(x, z)
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg="bool array"):
            z = np.array([[False], [True]])
            x = np.array([[False], [True]])
            pauli_list = PauliList.from_symplectic(z, x)
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg="bool array no copy"):
            z = np.array([[False], [True]])
            x = np.array([[True], [True]])
            pauli_list = PauliList.from_symplectic(z, x)
            z[0, 0] = not z[0, 0]
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

    def test_string_init(self):
        """Test string initialization."""
        # String initialization
        with self.subTest(msg='str init "I"'):
            pauli_list = PauliList("I")
            z = np.array([[False]])
            x = np.array([[False]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "X"'):
            pauli_list = PauliList("X")
            z = np.array([[False]])
            x = np.array([[True]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "Y"'):
            pauli_list = PauliList("Y")
            z = np.array([[True]])
            x = np.array([[True]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "Z"'):
            pauli_list = PauliList("Z")
            z = np.array([[True]])
            x = np.array([[False]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "iZ"'):
            pauli_list = PauliList("iZ")
            z = np.array([[True]])
            x = np.array([[False]])
            phase = np.array([3])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)
            np.testing.assert_equal(pauli_list.phase, phase)

        with self.subTest(msg='str init "-Z"'):
            pauli_list = PauliList("-Z")
            z = np.array([[True]])
            x = np.array([[False]])
            phase = np.array([2])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)
            np.testing.assert_equal(pauli_list.phase, phase)

        with self.subTest(msg='str init "-iZ"'):
            pauli_list = PauliList("-iZ")
            z = np.array([[True]])
            x = np.array([[False]])
            phase = np.array([1])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)
            np.testing.assert_equal(pauli_list.phase, phase)

        with self.subTest(msg='str init "IX"'):
            pauli_list = PauliList("IX")
            z = np.array([[False, False]])
            x = np.array([[True, False]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "XI"'):
            pauli_list = PauliList("XI")
            z = np.array([[False, False]])
            x = np.array([[False, True]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "YZ"'):
            pauli_list = PauliList("YZ")
            z = np.array([[True, True]])
            x = np.array([[False, True]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg='str init "iZY"'):
            pauli_list = PauliList("iZY")
            z = np.array([[True, True]])
            x = np.array([[True, False]])
            phase = np.array([3])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)
            np.testing.assert_equal(pauli_list.phase, phase)

        with self.subTest(msg='str init "XIZ"'):
            pauli_list = PauliList("XIZ")
            z = np.array([[True, False, False]])
            x = np.array([[False, False, True]])
            np.testing.assert_equal(pauli_list.z, z)
            np.testing.assert_equal(pauli_list.x, x)

        with self.subTest(msg="str init prevent broadcasting"):
            with self.assertRaises(ValueError):
                PauliList(["XYZ", "I"])

    def test_list_init(self):
        """Test list initialization."""
        with self.subTest(msg="PauliList"):
            target = PauliList(["iXI", "IX", "IZ"])
            value = PauliList(target)
            self.assertEqual(value, target)

        with self.subTest(msg="PauliList no copy"):
            target = PauliList(["iXI", "IX", "IZ"])
            value = PauliList(target)
            value[0] = "-iII"
            self.assertEqual(value, target)

    def test_init_from_settings(self):
        """Test initializing from the settings dictionary."""
        pauli_list = PauliList(["IX", "-iYZ", "YY"])
        from_settings = PauliList(**pauli_list.settings)
        self.assertEqual(pauli_list, from_settings)

    def test_from_symplectic_phase_check(self):
        """Test the from_symplectic method of PauliList for phase dimension check."""
        z = np.array([[0], [0]])
        x = np.array([[0], [0]])
        phase = np.array([[0], [0]])  # 2D phase
        with self.assertRaisesRegex(ValueError, "phase should be at most 1D but has 2 dimensions."):
            PauliList.from_symplectic(z, x, phase)


@ddt
class TestPauliListProperties(QiskitTestCase):
    """Tests for PauliList properties."""

    def test_x_property(self):
        """Test X property"""
        with self.subTest(msg="X"):
            pauli = PauliList(["XI", "IZ", "YY"])
            array = np.array([[False, True], [False, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(pauli.x == array))

        with self.subTest(msg="set X"):
            pauli = PauliList(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            pauli.x = val
            self.assertEqual(pauli, PauliList(["II", "iXY"]))

        with self.subTest(msg="set X raises"):
            with self.assertRaises(Exception):
                pauli = PauliList(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                pauli.x = val

    def test_z_property(self):
        """Test Z property"""
        with self.subTest(msg="Z"):
            pauli = PauliList(["XI", "IZ", "YY"])
            array = np.array([[False, False], [True, False], [True, True]], dtype=bool)
            self.assertTrue(np.all(pauli.z == array))

        with self.subTest(msg="set Z"):
            pauli = PauliList(["XI", "IZ"])
            val = np.array([[False, False], [True, True]], dtype=bool)
            pauli.z = val
            self.assertEqual(pauli, PauliList(["XI", "ZZ"]))

        with self.subTest(msg="set Z raises"):
            with self.assertRaises(Exception):
                pauli = PauliList(["XI", "IZ"])
                val = np.array([[False, False, False], [True, True, True]], dtype=bool)
                pauli.z = val

    def test_phase_property(self):
        """Test phase property"""
        with self.subTest(msg="phase"):
            pauli = PauliList(["XI", "IZ", "YY", "YI"])
            array = np.array([0, 0, 0, 0], dtype=int)
            np.testing.assert_equal(pauli.phase, array)

        with self.subTest(msg="set phase"):
            pauli = PauliList(["XI", "IZ"])
            val = np.array([2, 3], dtype=int)
            pauli.phase = val
            self.assertEqual(pauli, PauliList(["-XI", "iIZ"]))

        with self.subTest(msg="set Z raises"):
            with self.assertRaises(Exception):
                pauli = PauliList(["XI", "IZ"])
                val = np.array([1, 2, 3], dtype=int)
                pauli.phase = val

    def test_shape_property(self):
        """Test shape property"""
        shape = (3, 4)
        pauli = PauliList.from_symplectic(np.zeros(shape), np.zeros(shape))
        self.assertEqual(pauli.shape, shape)

    @combine(j=range(1, 10))
    def test_size_property(self, j):
        """Test size property"""
        shape = (j, 4)
        pauli = PauliList.from_symplectic(np.zeros(shape), np.zeros(shape))
        self.assertEqual(len(pauli), j)

    @combine(j=range(1, 10))
    def test_n_qubit_property(self, j):
        """Test n_qubit property"""
        shape = (5, j)
        pauli = PauliList.from_symplectic(np.zeros(shape), np.zeros(shape))
        self.assertEqual(pauli.num_qubits, j)

    def test_eq(self):
        """Test __eq__ method."""
        pauli1 = PauliList(["II", "XI"])
        pauli2 = PauliList(["XI", "II"])
        self.assertEqual(pauli1, pauli1)
        self.assertNotEqual(pauli1, pauli2)

    def test_len_methods(self):
        """Test __len__ method."""
        for j in range(1, 10):
            labels = j * ["XX"]
            pauli = PauliList(labels)
            self.assertEqual(len(pauli), j)

    def test_add_methods(self):
        """Test __add__ method."""
        labels1 = ["XXI", "IXX"]
        labels2 = ["XXI", "ZZI", "ZYZ"]
        pauli1 = PauliList(labels1)
        pauli2 = PauliList(labels2)
        target = PauliList(labels1 + labels2)
        self.assertEqual(target, pauli1 + pauli2)

    def test_add_qargs(self):
        """Test add method with qargs."""
        pauli1 = PauliList(["IIII", "YYYY"])
        pauli2 = PauliList(["XY", "YZ"])
        pauli3 = PauliList(["X", "Y", "Z"])

        with self.subTest(msg="qargs=[0, 1]"):
            target = PauliList(["IIII", "YYYY", "IIXY", "IIYZ"])
            self.assertEqual(pauli1 + pauli2([0, 1]), target)

        with self.subTest(msg="qargs=[0, 3]"):
            target = PauliList(["IIII", "YYYY", "XIIY", "YIIZ"])
            self.assertEqual(pauli1 + pauli2([0, 3]), target)

        with self.subTest(msg="qargs=[2, 1]"):
            target = PauliList(["IIII", "YYYY", "IYXI", "IZYI"])
            self.assertEqual(pauli1 + pauli2([2, 1]), target)

        with self.subTest(msg="qargs=[3, 1]"):
            target = PauliList(["IIII", "YYYY", "YIXI", "ZIYI"])
            self.assertEqual(pauli1 + pauli2([3, 1]), target)

        with self.subTest(msg="qargs=[0]"):
            target = PauliList(["IIII", "YYYY", "IIIX", "IIIY", "IIIZ"])
            self.assertEqual(pauli1 + pauli3([0]), target)

        with self.subTest(msg="qargs=[1]"):
            target = PauliList(["IIII", "YYYY", "IIXI", "IIYI", "IIZI"])
            self.assertEqual(pauli1 + pauli3([1]), target)

        with self.subTest(msg="qargs=[2]"):
            target = PauliList(["IIII", "YYYY", "IXII", "IYII", "IZII"])
            self.assertEqual(pauli1 + pauli3([2]), target)

        with self.subTest(msg="qargs=[3]"):
            target = PauliList(["IIII", "YYYY", "XIII", "YIII", "ZIII"])
            self.assertEqual(pauli1 + pauli3([3]), target)

    def test_getitem_methods(self):
        """Test __getitem__ method."""
        with self.subTest(msg="__getitem__ single"):
            labels = ["XI", "IY"]
            pauli = PauliList(labels)
            self.assertEqual(pauli[0], PauliList(labels[0]))
            self.assertEqual(pauli[1], PauliList(labels[1]))

        with self.subTest(msg="__getitem__ array"):
            labels = np.array(["XI", "IY", "IZ", "XY", "ZX"])
            pauli = PauliList(labels)
            inds = [0, 3]
            self.assertEqual(pauli[inds], PauliList(labels[inds]))
            inds = np.array([4, 1])
            self.assertEqual(pauli[inds], PauliList(labels[inds]))

        with self.subTest(msg="__getitem__ slice"):
            labels = np.array(["XI", "IY", "IZ", "XY", "ZX"])
            pauli = PauliList(labels)
            self.assertEqual(pauli[:], pauli)
            self.assertEqual(pauli[1:3], PauliList(labels[1:3]))

    def test_setitem_methods(self):
        """Test __setitem__ method."""
        with self.subTest(msg="__setitem__ single"):
            labels = ["XI", "IY"]
            pauli = PauliList(["XI", "IY"])
            pauli[0] = "II"
            self.assertEqual(pauli[0], PauliList("II"))
            pauli[1] = "-iXX"
            self.assertEqual(pauli[1], PauliList("-iXX"))

            with self.assertRaises(Exception):
                # Wrong size Pauli
                pauli[0] = "XXX"

        with self.subTest(msg="__setitem__ array"):
            labels = np.array(["XI", "IY", "IZ"])
            pauli = PauliList(labels)
            target = PauliList(["II", "ZZ"])
            inds = [2, 0]
            pauli[inds] = target
            self.assertEqual(pauli[inds], target)

            with self.assertRaises(Exception):
                pauli[inds] = PauliList(["YY", "ZZ", "XX"])

        with self.subTest(msg="__setitem__ slice"):
            labels = np.array(5 * ["III"])
            pauli = PauliList(labels)
            target = PauliList(5 * ["XXX"])
            pauli[:] = target
            self.assertEqual(pauli[:], target)
            target = PauliList(2 * ["ZZZ"])
            pauli[1:3] = target
            self.assertEqual(pauli[1:3], target)


class TestPauliListLabels(QiskitTestCase):
    """Tests PauliList label representation conversions."""

    def test_from_labels_1q(self):
        """Test 1-qubit from_labels method."""
        labels = ["I", "Z", "Z", "X", "Y"]
        target = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
        )
        value = PauliList(labels)
        self.assertEqual(target, value)

    def test_from_labels_1q_with_phase(self):
        """Test 1-qubit from_labels method with phase."""
        labels = ["-I", "iZ", "iZ", "X", "-iY"]
        target = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
            np.array([2, 3, 3, 0, 1]),
        )
        value = PauliList(labels)
        self.assertEqual(target, value)

    def test_from_labels_2q(self):
        """Test 2-qubit from_labels method."""
        labels = ["II", "YY", "XZ"]
        target = PauliList.from_symplectic(
            np.array([[False, False], [True, True], [True, False]]),
            np.array([[False, False], [True, True], [False, True]]),
        )
        value = PauliList(labels)
        self.assertEqual(target, value)

    def test_from_labels_2q_with_phase(self):
        """Test 2-qubit from_labels method."""
        labels = ["iII", "iYY", "-iXZ"]
        target = PauliList.from_symplectic(
            np.array([[False, False], [True, True], [True, False]]),
            np.array([[False, False], [True, True], [False, True]]),
            np.array([3, 3, 1]),
        )
        value = PauliList(labels)
        self.assertEqual(target, value)

    def test_from_labels_5q(self):
        """Test 5-qubit from_labels method."""
        labels = [5 * "I", 5 * "X", 5 * "Y", 5 * "Z"]
        target = PauliList.from_symplectic(
            np.array([[False] * 5, [False] * 5, [True] * 5, [True] * 5]),
            np.array([[False] * 5, [True] * 5, [True] * 5, [False] * 5]),
        )
        value = PauliList(labels)
        self.assertEqual(target, value)

    def test_to_labels_1q(self):
        """Test 1-qubit to_labels method."""
        pauli = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
        )
        target = ["I", "Z", "Z", "X", "Y"]
        value = pauli.to_labels()
        self.assertEqual(value, target)

    def test_to_labels_1q_with_phase(self):
        """Test 1-qubit to_labels method with phase."""
        pauli = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
            np.array([1, 3, 2, 3, 1]),
        )
        target = ["-iI", "iZ", "-Z", "iX", "-iY"]
        value = pauli.to_labels()
        self.assertEqual(value, target)

    def test_to_labels_1q_array(self):
        """Test 1-qubit to_labels method w/ array=True."""
        pauli = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
        )
        target = np.array(["I", "Z", "Z", "X", "Y"])
        value = pauli.to_labels(array=True)
        self.assertTrue(np.all(value == target))

    def test_to_labels_1q_array_with_phase(self):
        """Test 1-qubit to_labels method w/ array=True."""
        pauli = PauliList.from_symplectic(
            np.array([[False], [True], [True], [False], [True]]),
            np.array([[False], [False], [False], [True], [True]]),
            np.array([2, 3, 0, 1, 0]),
        )
        target = np.array(["-I", "iZ", "Z", "-iX", "Y"])
        value = pauli.to_labels(array=True)
        self.assertTrue(np.all(value == target))

    def test_labels_round_trip(self):
        """Test from_labels and to_labels round trip."""
        target = ["III", "IXZ", "XYI", "ZZZ", "-iZIX", "-IYX"]
        value = PauliList(target).to_labels()
        self.assertEqual(value, target)

    def test_labels_round_trip_array(self):
        """Test from_labels and to_labels round trip w/ array=True."""
        labels = ["III", "IXZ", "XYI", "ZZZ", "-iZIX", "-IYX"]
        target = np.array(labels)
        value = PauliList(labels).to_labels(array=True)
        self.assertTrue(np.all(value == target))


class TestPauliListMatrix(QiskitTestCase):
    """Tests PauliList matrix representation conversions."""

    def test_to_matrix_1q(self):
        """Test 1-qubit to_matrix method."""
        labels = ["X", "I", "Z", "Y"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_array(self):
        """Test 1-qubit to_matrix method w/ array=True."""
        labels = ["Z", "I", "Y", "X"]
        target = np.array([pauli_mat(i) for i in labels])
        value = PauliList(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_1q_sparse(self):
        """Test 1-qubit to_matrix method w/ sparse=True."""
        labels = ["X", "I", "Z", "Y"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_2q(self):
        """Test 2-qubit to_matrix method."""
        labels = ["IX", "YI", "II", "ZZ"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_array(self):
        """Test 2-qubit to_matrix method w/ array=True."""
        labels = ["ZZ", "XY", "YX", "IZ"]
        target = np.array([pauli_mat(i) for i in labels])
        value = PauliList(labels).to_matrix(array=True)
        self.assertTrue(isinstance(value, np.ndarray))
        self.assertTrue(np.all(value == target))

    def test_to_matrix_2q_sparse(self):
        """Test 2-qubit to_matrix method w/ sparse=True."""
        labels = ["IX", "II", "ZY", "YZ"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_5q(self):
        """Test 5-qubit to_matrix method."""
        labels = ["IXIXI", "YZIXI", "IIXYZ"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_5q_sparse(self):
        """Test 5-qubit to_matrix method w/ sparse=True."""
        labels = ["XXXYY", "IXIZY", "ZYXIX"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))

    def test_to_matrix_5q_with_phase(self):
        """Test 5-qubit to_matrix method with phase."""
        labels = ["iIXIXI", "-YZIXI", "-iIIXYZ"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix()
        self.assertTrue(isinstance(values, list))
        for target, value in zip(targets, values):
            self.assertTrue(np.all(value == target))

    def test_to_matrix_5q_sparse_with_phase(self):
        """Test 5-qubit to_matrix method w/ sparse=True with phase."""
        labels = ["iXXXYY", "-IXIZY", "-iZYXIX"]
        targets = [pauli_mat(i) for i in labels]
        values = PauliList(labels).to_matrix(sparse=True)
        for mat, targ in zip(values, targets):
            self.assertTrue(isinstance(mat, csr_matrix))
            self.assertTrue(np.all(targ == mat.toarray()))


class TestPauliListIteration(QiskitTestCase):
    """Tests for PauliList iterators class."""

    def test_enumerate(self):
        """Test enumerate with PauliList."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for idx, i in enumerate(pauli):
            self.assertEqual(i, PauliList(labels[idx]))

    def test_iter(self):
        """Test iter with PauliList."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for idx, i in enumerate(iter(pauli)):
            self.assertEqual(i, PauliList(labels[idx]))

    def test_zip(self):
        """Test zip with PauliList."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for label, i in zip(labels, pauli):
            self.assertEqual(i, PauliList(label))

    def test_label_iter(self):
        """Test PauliList label_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for idx, i in enumerate(pauli.label_iter()):
            self.assertEqual(i, labels[idx])

    def test_matrix_iter(self):
        """Test PauliList dense matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for idx, i in enumerate(pauli.matrix_iter()):
            self.assertTrue(np.all(i == pauli_mat(labels[idx])))

    def test_matrix_iter_sparse(self):
        """Test PauliList sparse matrix_iter method."""
        labels = ["III", "IXI", "IYY", "YIZ", "XYZ", "III"]
        pauli = PauliList(labels)
        for idx, i in enumerate(pauli.matrix_iter(sparse=True)):
            self.assertTrue(isinstance(i, csr_matrix))
            self.assertTrue(np.all(i.toarray() == pauli_mat(labels[idx])))


@ddt
class TestPauliListOperator(QiskitTestCase):
    """Tests for PauliList base operator methods."""

    @combine(j=range(1, 10))
    def test_tensor(self, j):
        """Test tensor method j={j}."""
        labels1 = ["XX", "YY"]
        labels2 = [j * "I", j * "Z"]
        pauli1 = PauliList(labels1)
        pauli2 = PauliList(labels2)

        value = pauli1.tensor(pauli2)
        target = PauliList([l1 + l2 for l1 in labels1 for l2 in labels2])
        self.assertEqual(value, target)

    @combine(j=range(1, 10))
    def test_tensor_with_phase(self, j):
        """Test tensor method j={j} with phase."""
        labels1 = ["XX", "iYY"]
        labels2 = [j * "I", "i" + j * "Z"]
        pauli1 = PauliList(labels1)
        pauli2 = PauliList(labels2)

        value = pauli1.tensor(pauli2)
        target = PauliList(["XX" + "I" * j, "iXX" + "Z" * j, "iYY" + "I" * j, "-YY" + "Z" * j])
        self.assertEqual(value, target)

    @combine(j=range(1, 10))
    def test_expand(self, j):
        """Test expand method j={j}."""
        labels1 = ["XX", "YY"]
        labels2 = [j * "I", j * "Z"]
        pauli1 = PauliList(labels1)
        pauli2 = PauliList(labels2)

        value = pauli1.expand(pauli2)
        target = PauliList([j + i for j in labels2 for i in labels1])
        self.assertEqual(value, target)

    @combine(j=range(1, 10))
    def test_expand_with_phase(self, j):
        """Test expand method j={j}."""
        labels1 = ["-XX", "iYY"]
        labels2 = ["i" + j * "I", "-i" + j * "Z"]
        pauli1 = PauliList(labels1)
        pauli2 = PauliList(labels2)

        value = pauli1.expand(pauli2)
        target = PauliList(
            ["-i" + "I" * j + "XX", "-" + "I" * j + "YY", "i" + "Z" * j + "XX", "Z" * j + "YY"]
        )
        self.assertEqual(value, target)

    def test_compose_1q(self):
        """Test 1-qubit compose methods."""
        # Test single qubit Pauli dot products
        pauli = PauliList(["I", "X", "Y", "Z"])

        with self.subTest(msg="compose single I"):
            target = PauliList(["I", "X", "Y", "Z"])
            value = pauli.compose("I")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single X"):
            target = PauliList(["X", "I", "iZ", "-iY"])
            value = pauli.compose("X")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single Y"):
            target = PauliList(["Y", "-iZ", "I", "iX"])
            value = pauli.compose("Y")
            self.assertEqual(target, value)

        with self.subTest(msg="compose single Z"):
            target = PauliList(["Z", "iY", "-iX", "I"])
            value = pauli.compose("Z")
            self.assertEqual(target, value)

    def test_dot_1q(self):
        """Test 1-qubit dot method."""
        # Test single qubit Pauli dot products
        pauli = PauliList(["I", "X", "Y", "Z"])

        with self.subTest(msg="dot single I"):
            target = PauliList(["I", "X", "Y", "Z"])
            value = pauli.dot("I")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single X"):
            target = PauliList(["X", "I", "-iZ", "iY"])
            value = pauli.dot("X")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single Y"):
            target = PauliList(["Y", "iZ", "I", "-iX"])
            value = pauli.dot("Y")
            self.assertEqual(target, value)

        with self.subTest(msg="dot single Z"):
            target = PauliList(["Z", "-iY", "iX", "I"])
            value = pauli.dot("Z")
            self.assertEqual(target, value)

    def test_qargs_compose_1q(self):
        """Test 1-qubit compose method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("Z")

        with self.subTest(msg="compose 1-qubit qargs=[0]"):
            target = PauliList(["IIZ", "iXXY"])
            value = pauli1.compose(pauli2, qargs=[0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[1]"):
            target = PauliList(["IZI", "iXYX"])
            value = pauli1.compose(pauli2, qargs=[1])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 1-qubit qargs=[2]"):
            target = PauliList(["ZII", "iYXX"])
            value = pauli1.compose(pauli2, qargs=[2])
            self.assertEqual(value, target)

    def test_qargs_dot_1q(self):
        """Test 1-qubit dot method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("Z")

        with self.subTest(msg="dot 1-qubit qargs=[0]"):
            target = PauliList(["IIZ", "-iXXY"])
            value = pauli1.dot(pauli2, qargs=[0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 1-qubit qargs=[1]"):
            target = PauliList(["IZI", "-iXYX"])
            value = pauli1.dot(pauli2, qargs=[1])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 1-qubit qargs=[2]"):
            target = PauliList(["ZII", "-iYXX"])
            value = pauli1.dot(pauli2, qargs=[2])
            self.assertEqual(value, target)

    def test_qargs_compose_2q(self):
        """Test 2-qubit compose method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("ZY")

        with self.subTest(msg="compose 2-qubit qargs=[0, 1]"):
            target = PauliList(["IZY", "XYZ"])
            value = pauli1.compose(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[1, 0]"):
            target = PauliList(["IYZ", "XZY"])
            value = pauli1.compose(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[0, 2]"):
            target = PauliList(["ZIY", "YXZ"])
            value = pauli1.compose(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 2-qubit qargs=[2, 0]"):
            target = PauliList(["YIZ", "ZXY"])
            value = pauli1.compose(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

    def test_qargs_dot_2q(self):
        """Test 2-qubit dot method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("ZY")

        with self.subTest(msg="dot 2-qubit qargs=[0, 1]"):
            target = PauliList(["IZY", "XYZ"])
            value = pauli1.dot(pauli2, qargs=[0, 1])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[1, 0]"):
            target = PauliList(["IYZ", "XZY"])
            value = pauli1.dot(pauli2, qargs=[1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[0, 2]"):
            target = PauliList(["ZIY", "YXZ"])
            value = pauli1.dot(pauli2, qargs=[0, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 2-qubit qargs=[2, 0]"):
            target = PauliList(["YIZ", "ZXY"])
            value = pauli1.dot(pauli2, qargs=[2, 0])
            self.assertEqual(value, target)

    def test_qargs_compose_3q(self):
        """Test 3-qubit compose method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("XYZ")

        with self.subTest(msg="compose 3-qubit qargs=None"):
            target = PauliList(["XYZ", "IZY"])
            value = pauli1.compose(pauli2)
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[0, 1, 2]"):
            target = PauliList(["XYZ", "IZY"])
            value = pauli1.compose(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[2, 1, 0]"):
            target = PauliList(["ZYX", "YZI"])
            value = pauli1.compose(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="compose 3-qubit qargs=[1, 0, 2]"):
            target = PauliList(["XZY", "IYZ"])
            value = pauli1.compose(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)

    def test_qargs_dot_3q(self):
        """Test 3-qubit dot method with qargs."""

        pauli1 = PauliList(["III", "XXX"])
        pauli2 = PauliList("XYZ")

        with self.subTest(msg="dot 3-qubit qargs=None"):
            target = PauliList(["XYZ", "IZY"])
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[0, 1, 2]"):
            target = PauliList(["XYZ", "IZY"])
            value = pauli1.dot(pauli2, qargs=[0, 1, 2])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[2, 1, 0]"):
            target = PauliList(["ZYX", "YZI"])
            value = pauli1.dot(pauli2, qargs=[2, 1, 0])
            self.assertEqual(value, target)

        with self.subTest(msg="dot 3-qubit qargs=[1, 0, 2]"):
            target = PauliList(["XZY", "IYZ"])
            value = pauli1.dot(pauli2, qargs=[1, 0, 2])
            self.assertEqual(value, target)


@ddt
class TestPauliListMethods(QiskitTestCase):
    """Tests for PauliList utility methods class."""

    def test_sort(self):
        """Test sort method."""
        with self.subTest(msg="1 qubit standard order"):
            unsrt = ["X", "Z", "I", "Y", "-iI", "X", "Z", "iI", "-I", "-iY"]
            srt = ["I", "-iI", "-I", "iI", "X", "X", "Y", "-iY", "Z", "Z"]
            target = PauliList(srt)
            value = PauliList(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit weight order"):
            unsrt = ["X", "Z", "I", "Y", "-iI", "X", "Z", "iI", "-I", "-iY"]
            srt = ["I", "-iI", "-I", "iI", "X", "X", "Y", "-iY", "Z", "Z"]
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit phase order"):
            unsrt = ["X", "Z", "I", "Y", "-iI", "X", "Z", "iI", "-I", "-iY"]
            srt = ["I", "X", "X", "Y", "Z", "Z", "-iI", "-iY", "-I", "iI"]
            target = PauliList(srt)
            value = PauliList(unsrt).sort(phase=True)
            self.assertEqual(target, value)

        with self.subTest(msg="1 qubit weight & phase order"):
            unsrt = ["X", "Z", "I", "Y", "-iI", "X", "Z", "iI", "-I", "-iY"]
            srt = ["I", "X", "X", "Y", "Z", "Z", "-iI", "-iY", "-I", "iI"]
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True, phase=True)
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit standard order"):
            srt = [
                "II",
                "IX",
                "IX",
                "IY",
                "IZ",
                "iIZ",
                "XI",
                "XX",
                "XX",
                "iXX",
                "XY",
                "XZ",
                "iXZ",
                "YI",
                "YI",
                "-YI",
                "YX",
                "-iYX",
                "YY",
                "-iYY",
                "-YY",
                "iYY",
                "YZ",
                "ZI",
                "ZX",
                "ZX",
                "ZY",
                "ZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit weight order"):
            srt = [
                "II",
                "IX",
                "IX",
                "IY",
                "IZ",
                "iIZ",
                "XI",
                "YI",
                "YI",
                "-YI",
                "ZI",
                "XX",
                "XX",
                "iXX",
                "XY",
                "XZ",
                "iXZ",
                "YX",
                "-iYX",
                "YY",
                "YY",
                "-YY",
                "YZ",
                "ZX",
                "ZX",
                "ZY",
                "ZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit phase order"):
            srt = [
                "II",
                "IX",
                "IX",
                "IY",
                "IZ",
                "XI",
                "XX",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YI",
                "YX",
                "YY",
                "YY",
                "YZ",
                "ZI",
                "ZX",
                "ZX",
                "ZY",
                "ZZ",
                "-iYX",
                "-YI",
                "-YY",
                "iIZ",
                "iXX",
                "iXZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(phase=True)
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit weight & phase order"):
            srt = [
                "II",
                "IX",
                "IX",
                "IY",
                "IZ",
                "XI",
                "YI",
                "YI",
                "ZI",
                "XX",
                "XX",
                "XY",
                "XZ",
                "YX",
                "YY",
                "YY",
                "YZ",
                "ZX",
                "ZX",
                "ZY",
                "ZZ",
                "-iYX",
                "-YI",
                "-YY",
                "iIZ",
                "iXX",
                "iXZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True, phase=True)
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit standard order"):
            srt = [
                "III",
                "III",
                "IIX",
                "IIY",
                "-IIY",
                "IIZ",
                "IXI",
                "IXX",
                "IXY",
                "iIXY",
                "IXZ",
                "IYI",
                "IYX",
                "IYY",
                "IYZ",
                "IZI",
                "IZX",
                "IZY",
                "IZY",
                "IZZ",
                "XII",
                "XII",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XXX",
                "-iXXX",
                "XXY",
                "XXZ",
                "XYI",
                "XYX",
                "iXYX",
                "XYY",
                "XYZ",
                "XYZ",
                "XZI",
                "XZX",
                "XZY",
                "XZZ",
                "YII",
                "YIX",
                "YIY",
                "YIZ",
                "YXI",
                "YXX",
                "YXY",
                "YXZ",
                "YXZ",
                "YYI",
                "YYX",
                "YYX",
                "YYY",
                "YYZ",
                "YZI",
                "YZX",
                "YZY",
                "YZZ",
                "ZII",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZXX",
                "iZXX",
                "ZXY",
                "ZXZ",
                "ZYI",
                "ZYI",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZI",
                "ZZX",
                "ZZY",
                "ZZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort()
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit weight order"):
            srt = [
                "III",
                "III",
                "IIX",
                "IIY",
                "-IIY",
                "IIZ",
                "IXI",
                "IYI",
                "IZI",
                "XII",
                "XII",
                "YII",
                "ZII",
                "IXX",
                "IXY",
                "iIXY",
                "IXZ",
                "IYX",
                "IYY",
                "IYZ",
                "IZX",
                "IZY",
                "IZY",
                "IZZ",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XYI",
                "XZI",
                "YIX",
                "YIY",
                "YIZ",
                "YXI",
                "YYI",
                "YZI",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZYI",
                "ZYI",
                "ZZI",
                "XXX",
                "-iXXX",
                "XXY",
                "XXZ",
                "XYX",
                "iXYX",
                "XYY",
                "XYZ",
                "XYZ",
                "XZX",
                "XZY",
                "XZZ",
                "YXX",
                "YXY",
                "YXZ",
                "YXZ",
                "YYX",
                "YYX",
                "YYY",
                "YYZ",
                "YZX",
                "YZY",
                "YZZ",
                "ZXX",
                "iZXX",
                "ZXY",
                "ZXZ",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZX",
                "ZZY",
                "ZZZ",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True)
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit phase order"):
            srt = [
                "III",
                "III",
                "IIX",
                "IIY",
                "IIZ",
                "IXI",
                "IXX",
                "IXY",
                "IXZ",
                "IYI",
                "IYX",
                "IYY",
                "IYZ",
                "IZI",
                "IZX",
                "IZY",
                "IZY",
                "IZZ",
                "XII",
                "XII",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XXX",
                "XXY",
                "XXZ",
                "XYI",
                "XYX",
                "XYY",
                "XYZ",
                "XYZ",
                "XZI",
                "XZX",
                "XZY",
                "XZZ",
                "YII",
                "YIX",
                "YIY",
                "YIZ",
                "YXI",
                "YXX",
                "YXY",
                "YXZ",
                "YXZ",
                "YYI",
                "YYX",
                "YYX",
                "YYY",
                "YYZ",
                "YZI",
                "YZX",
                "YZY",
                "YZZ",
                "ZII",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZXX",
                "ZXY",
                "ZXZ",
                "ZYI",
                "ZYI",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZI",
                "ZZX",
                "ZZY",
                "ZZZ",
                "-iXXX",
                "-IIY",
                "iIXY",
                "iXYX",
                "iZXX",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(phase=True)
            self.assertEqual(target, value)

        with self.subTest(msg="3 qubit weight & phase order"):
            srt = [
                "III",
                "III",
                "IIX",
                "IIY",
                "IYI",
                "IZI",
                "XII",
                "XII",
                "YII",
                "ZII",
                "IXX",
                "IXY",
                "IXZ",
                "IYX",
                "IYY",
                "IYZ",
                "IZX",
                "IZY",
                "IZY",
                "IZZ",
                "XIX",
                "XIY",
                "XIZ",
                "XXI",
                "XYI",
                "XZI",
                "YIX",
                "YIY",
                "YIZ",
                "YYI",
                "YZI",
                "ZIX",
                "ZIY",
                "ZIZ",
                "ZXI",
                "ZYI",
                "ZYI",
                "ZZI",
                "XXX",
                "XXY",
                "XXZ",
                "XYX",
                "XYY",
                "XYZ",
                "XZX",
                "XZY",
                "XZZ",
                "YXX",
                "YXY",
                "YXZ",
                "YXZ",
                "YYX",
                "YYX",
                "YYY",
                "YYZ",
                "YZX",
                "YZY",
                "YZZ",
                "ZXX",
                "ZXY",
                "ZXZ",
                "ZYX",
                "ZYY",
                "ZYZ",
                "ZZX",
                "ZZY",
                "ZZZ",
                "-iZIZ",
                "-iXXX",
                "-IIY",
                "iIXI",
                "iIXY",
                "iYXI",
                "iXYX",
                "iXYZ",
                "iZXX",
            ]
            unsrt = srt.copy()
            np.random.shuffle(unsrt)
            target = PauliList(srt)
            value = PauliList(unsrt).sort(weight=True, phase=True)
            self.assertEqual(target, value)

    def test_unique(self):
        """Test unique method."""
        with self.subTest(msg="1 qubit"):
            labels = ["X", "Z", "X", "X", "I", "Y", "I", "X", "Z", "Z", "X", "I"]
            unique = ["X", "Z", "I", "Y"]
            target = PauliList(unique)
            value = PauliList(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg="2 qubit"):
            labels = ["XX", "IX", "XX", "II", "IZ", "ZI", "YX", "YX", "ZZ", "IX", "XI"]
            unique = ["XX", "IX", "II", "IZ", "ZI", "YX", "ZZ", "XI"]
            target = PauliList(unique)
            value = PauliList(labels).unique()
            self.assertEqual(target, value)

        with self.subTest(msg="10 qubit"):
            labels = [10 * "X", 10 * "I", 10 * "X"]
            unique = [10 * "X", 10 * "I"]
            target = PauliList(unique)
            value = PauliList(labels).unique()
            self.assertEqual(target, value)

    def test_delete(self):
        """Test delete method."""
        with self.subTest(msg="no rows"):
            pauli = PauliList(["XX", "ZZ"])
            self.assertEqual(pauli.delete([]), pauli)

        with self.subTest(msg="single row"):
            for j in range(1, 6):
                pauli = PauliList([j * "X", j * "Y"])
                self.assertEqual(pauli.delete(0), PauliList(j * "Y"))
                self.assertEqual(pauli.delete(1), PauliList(j * "X"))

        with self.subTest(msg="multiple rows"):
            for j in range(1, 6):
                pauli = PauliList([j * "X", "-i" + j * "Y", j * "Z"])
                self.assertEqual(pauli.delete([0, 2]), PauliList("-i" + j * "Y"))
                self.assertEqual(pauli.delete([1, 2]), PauliList(j * "X"))
                self.assertEqual(pauli.delete([0, 1]), PauliList(j * "Z"))

        with self.subTest(msg="no qubits"):
            pauli = PauliList(["XX", "ZZ"])
            self.assertEqual(pauli.delete([], qubit=True), pauli)

        with self.subTest(msg="single qubit"):
            pauli = PauliList(["IIX", "iIYI", "ZII"])
            value = pauli.delete(0, qubit=True)
            target = PauliList(["II", "iIY", "ZI"])
            self.assertEqual(value, target)
            value = pauli.delete(1, qubit=True)
            target = PauliList(["IX", "iII", "ZI"])
            self.assertEqual(value, target)
            value = pauli.delete(2, qubit=True)
            target = PauliList(["IX", "iYI", "II"])
            self.assertEqual(value, target)

        with self.subTest(msg="multiple qubits"):
            pauli = PauliList(["IIX", "IYI", "-ZII"])
            value = pauli.delete([0, 1], qubit=True)
            target = PauliList(["I", "I", "-Z"])
            self.assertEqual(value, target)
            value = pauli.delete([1, 2], qubit=True)
            target = PauliList(["X", "I", "-I"])
            self.assertEqual(value, target)
            value = pauli.delete([0, 2], qubit=True)
            target = PauliList(["I", "Y", "-I"])
            self.assertEqual(value, target)

    def test_insert(self):
        """Test insert method."""
        # Insert single row
        for j in range(1, 10):
            pauli = PauliList(j * "X")
            target0 = PauliList([j * "I", j * "X"])
            target1 = PauliList([j * "X", j * "I"])

            with self.subTest(msg=f"single row from str ({j})"):
                value0 = pauli.insert(0, j * "I")
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, j * "I")
                self.assertEqual(value1, target1)

            with self.subTest(msg=f"single row from PauliList ({j})"):
                value0 = pauli.insert(0, PauliList(j * "I"))
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, PauliList(j * "I"))
                self.assertEqual(value1, target1)

            target0 = PauliList(["i" + j * "I", j * "X"])
            target1 = PauliList([j * "X", "i" + j * "I"])

            with self.subTest(msg=f"single row with phase from str ({j})"):
                value0 = pauli.insert(0, "i" + j * "I")
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, "i" + j * "I")
                self.assertEqual(value1, target1)

            with self.subTest(msg=f"single row with phase from PauliList ({j})"):
                value0 = pauli.insert(0, PauliList("i" + j * "I"))
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, PauliList("i" + j * "I"))
                self.assertEqual(value1, target1)

        # Insert multiple rows
        for j in range(1, 10):
            pauli = PauliList("i" + j * "X")
            insert = PauliList([j * "I", j * "Y", j * "Z", "-i" + j * "X"])
            target0 = insert + pauli
            target1 = pauli + insert

            with self.subTest(msg=f"multiple-rows from PauliList ({j})"):
                value0 = pauli.insert(0, insert)
                self.assertEqual(value0, target0)
                value1 = pauli.insert(1, insert)
                self.assertEqual(value1, target1)

        # Insert single column to length-1 PauliList:
        with self.subTest(msg="length-1, single-column, single-val"):
            pauli = PauliList(["X"])
            insert = PauliList(["Y"])
            target0 = PauliList(["YX"])
            value0 = pauli.insert(1, insert, qubit=True)
            self.assertEqual(value0, target0)
            self.assertEqual(value0.phase.shape, (1,))

        # Insert single column
        pauli = PauliList(["X", "Y", "Z", "-iI"])
        for i in ["I", "X", "Y", "Z", "iY"]:
            phase = "" if len(i) == 1 else i[0]
            p = i if len(i) == 1 else i[1]
            target0 = PauliList(
                [
                    phase + "X" + p,
                    phase + "Y" + p,
                    phase + "Z" + p,
                    ("" if phase else "-i") + "I" + p,
                ]
            )
            target1 = PauliList(
                [
                    i + "X",
                    i + "Y",
                    i + "Z",
                    ("" if phase else "-i") + p + "I",
                ]
            )

            with self.subTest(msg="single-column single-val from str"):
                value = pauli.insert(0, i, qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, i, qubit=True)
                self.assertEqual(value, target1)

            with self.subTest(msg="single-column single-val from PauliList"):
                value = pauli.insert(0, PauliList(i), qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliList(i), qubit=True)
                self.assertEqual(value, target1)

        # Insert single column with multiple values
        pauli = PauliList(["X", "Y", "iZ"])
        for i in [["I", "X", "Y"], ["X", "iY", "Z"], ["Y", "Z", "I"]]:
            target0 = PauliList(
                ["X" + i[0], "Y" + i[1] if len(i[1]) == 1 else i[1][0] + "Y" + i[1][1], "iZ" + i[2]]
            )
            target1 = PauliList([i[0] + "X", i[1] + "Y", "i" + i[2] + "Z"])

            with self.subTest(msg="single-column multiple-vals from PauliList"):
                value = pauli.insert(0, PauliList(i), qubit=True)
                self.assertEqual(value, target0)
                value = pauli.insert(1, PauliList(i), qubit=True)
                self.assertEqual(value, target1)

        # Insert multiple columns from single
        pauli = PauliList(["X", "iY", "Z"])
        for j in range(1, 5):
            for i in [j * "I", j * "X", j * "Y", "i" + j * "Z"]:
                phase = "" if len(i) == j else i[0]
                p = i if len(i) == j else i[1:]
                target0 = PauliList(
                    [
                        phase + "X" + p,
                        ("-" if phase else "i") + "Y" + p,
                        phase + "Z" + p,
                    ]
                )
                target1 = PauliList([i + "X", ("-" if phase else "i") + p + "Y", i + "Z"])

                with self.subTest(msg="multiple-columns single-val from str"):
                    value = pauli.insert(0, i, qubit=True)
                    self.assertEqual(value, target0)
                    value = pauli.insert(1, i, qubit=True)
                    self.assertEqual(value, target1)

                with self.subTest(msg="multiple-columns single-val from PauliList"):
                    value = pauli.insert(0, PauliList(i), qubit=True)
                    self.assertEqual(value, target0)
                    value = pauli.insert(1, PauliList(i), qubit=True)
                    self.assertEqual(value, target1)

        # Insert multiple columns multiple row values
        pauli = PauliList(["X", "Y", "-iZ"])
        for j in range(1, 5):
            for i in [
                [j * "I", j * "X", j * "Y"],
                [j * "X", j * "Z", "i" + j * "Y"],
                [j * "Y", j * "Z", j * "I"],
            ]:
                target0 = PauliList(
                    [
                        "X" + i[0],
                        "Y" + i[1],
                        ("-i" if len(i[2]) == j else "") + "Z" + i[2][-j:],
                    ]
                )
                target1 = PauliList(
                    [
                        i[0] + "X",
                        i[1] + "Y",
                        ("-i" if len(i[2]) == j else "") + i[2][-j:] + "Z",
                    ]
                )

                with self.subTest(msg="multiple-column multiple-vals from PauliList"):
                    value = pauli.insert(0, PauliList(i), qubit=True)
                    self.assertEqual(value, target0)
                    value = pauli.insert(1, PauliList(i), qubit=True)
                    self.assertEqual(value, target1)

    def test_commutes(self):
        """Test commutes method."""
        # Single qubit Pauli
        pauli = PauliList(["I", "X", "Y", "Z", "-iY"])
        with self.subTest(msg="commutes single-Pauli I"):
            value = list(pauli.commutes("I"))
            target = [True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli X"):
            value = list(pauli.commutes("X"))
            target = [True, True, False, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli Y"):
            value = list(pauli.commutes("Y"))
            target = [True, False, True, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli Z"):
            value = list(pauli.commutes("Z"))
            target = [True, False, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli iZ"):
            value = list(pauli.commutes("iZ"))
            target = [True, False, False, True, False]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliList(["II", "IX", "YI", "XY", "ZZ", "-iYY"])
        with self.subTest(msg="commutes single-Pauli II"):
            value = list(pauli.commutes("II"))
            target = [True, True, True, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli IX"):
            value = list(pauli.commutes("IX"))
            target = [True, True, True, False, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli XI"):
            value = list(pauli.commutes("XI"))
            target = [True, True, False, True, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli YI"):
            value = list(pauli.commutes("YI"))
            target = [True, True, True, False, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli IY"):
            value = list(pauli.commutes("IY"))
            target = [True, False, True, True, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli XY"):
            value = list(pauli.commutes("XY"))
            target = [True, False, False, True, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli YX"):
            value = list(pauli.commutes("YX"))
            target = [True, True, True, True, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli ZZ"):
            value = list(pauli.commutes("ZZ"))
            target = [True, False, False, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes single-Pauli iYX"):
            value = list(pauli.commutes("iYX"))
            target = [True, True, True, True, True, False]
            self.assertEqual(value, target)

    def test_anticommutes(self):
        """Test anticommutes method."""
        # Single qubit Pauli
        pauli = PauliList(["I", "X", "Y", "Z", "-iY"])
        with self.subTest(msg="anticommutes single-Pauli I"):
            value = list(pauli.anticommutes("I"))
            target = [False, False, False, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli X"):
            value = list(pauli.anticommutes("X"))
            target = [False, False, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli Y"):
            value = list(pauli.anticommutes("Y"))
            target = [False, True, False, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli Z"):
            value = list(pauli.anticommutes("Z"))
            target = [False, True, True, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli iZ"):
            value = list(pauli.anticommutes("iZ"))
            target = [False, True, True, False, True]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliList(["II", "IX", "YI", "XY", "ZZ", "iZX"])
        with self.subTest(msg="anticommutes single-Pauli II"):
            value = list(pauli.anticommutes("II"))
            target = [False, False, False, False, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli IX"):
            value = list(pauli.anticommutes("IX"))
            target = [False, False, False, True, True, False]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli XI"):
            value = list(pauli.anticommutes("XI"))
            target = [False, False, True, False, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli YI"):
            value = list(pauli.anticommutes("YI"))
            target = [False, False, False, True, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli IY"):
            value = list(pauli.anticommutes("IY"))
            target = [False, True, False, False, True, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli XY"):
            value = list(pauli.anticommutes("XY"))
            target = [False, True, True, False, False, False]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli YX"):
            value = list(pauli.anticommutes("YX"))
            target = [False, False, False, False, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli ZZ"):
            value = list(pauli.anticommutes("ZZ"))
            target = [False, True, True, False, False, True]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes single-Pauli iXY"):
            value = list(pauli.anticommutes("iXY"))
            target = [False, True, True, False, False, False]
            self.assertEqual(value, target)

    def test_commutes_with_all(self):
        """Test commutes_with_all method."""
        # 1-qubit
        pauli = PauliList(["I", "X", "Y", "Z", "-iY"])
        with self.subTest(msg="commutes_with_all [I]"):
            value = list(pauli.commutes_with_all("I"))
            target = [0, 1, 2, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [X]"):
            value = list(pauli.commutes_with_all("X"))
            target = [0, 1]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [Y]"):
            value = list(pauli.commutes_with_all("Y"))
            target = [0, 2, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [Z]"):
            value = list(pauli.commutes_with_all("Z"))
            target = [0, 3]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [iY]"):
            value = list(pauli.commutes_with_all("iY"))
            target = [0, 2, 4]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliList(["II", "IX", "YI", "XY", "ZZ", "iXY"])

        with self.subTest(msg="commutes_with_all [IX, YI]"):
            other = PauliList(["IX", "YI"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, ZZ]"):
            other = PauliList(["XY", "ZZ"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4, 5]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [YX, ZZ]"):
            other = PauliList(["YX", "ZZ"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4, 5]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, YX]"):
            other = PauliList(["XY", "YX"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4, 5]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [XY, IX]"):
            other = PauliList(["XY", "IX"])
            value = list(pauli.commutes_with_all(other))
            target = [0]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [YX, IX]"):
            other = PauliList(["YX", "IX"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="commutes_with_all [-iYX, iZZ]"):
            other = PauliList(["-iYX", "iZZ"])
            value = list(pauli.commutes_with_all(other))
            target = [0, 3, 4, 5]
            self.assertEqual(value, target)

    def test_anticommutes_with_all(self):
        """Test anticommutes_with_all method."""
        # 1-qubit
        pauli = PauliList(["I", "X", "Y", "Z", "-iY"])
        with self.subTest(msg="anticommutes_with_all [I]"):
            value = list(pauli.anticommutes_with_all("I"))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="antianticommutes_with_all [X]"):
            value = list(pauli.anticommutes_with_all("X"))
            target = [2, 3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [Y]"):
            value = list(pauli.anticommutes_with_all("Y"))
            target = [1, 3]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [Z]"):
            value = list(pauli.anticommutes_with_all("Z"))
            target = [1, 2, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [iY]"):
            value = list(pauli.anticommutes_with_all("iY"))
            target = [1, 3]
            self.assertEqual(value, target)

        # 2-qubit Pauli
        pauli = PauliList(["II", "IX", "YI", "XY", "ZZ", "iZX"])

        with self.subTest(msg="anticommutes_with_all [IX, YI]"):
            other = PauliList(["IX", "YI"])
            value = list(pauli.anticommutes_with_all(other))
            target = [3, 4]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, ZZ]"):
            other = PauliList(["XY", "ZZ"])
            value = list(pauli.anticommutes_with_all(other))
            target = [1, 2]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [YX, ZZ]"):
            other = PauliList(["YX", "ZZ"])
            value = list(pauli.anticommutes_with_all(other))
            target = [5]
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, YX]"):
            other = PauliList(["XY", "YX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [XY, IX]"):
            other = PauliList(["XY", "IX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

        with self.subTest(msg="anticommutes_with_all [YX, IX]"):
            other = PauliList(["YX", "IX"])
            value = list(pauli.anticommutes_with_all(other))
            target = []
            self.assertEqual(value, target)

    @combine(
        gate=(
            IGate(),
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(theta=np.pi / 2),
            RYGate(theta=np.pi / 2),
            RZGate(phi=np.pi / 2),
            Clifford(IGate()),
            Clifford(XGate()),
            Clifford(YGate()),
            Clifford(ZGate()),
            Clifford(HGate()),
            Clifford(SGate()),
            Clifford(SdgGate()),
        )
    )
    def test_evolve_clifford1(self, gate):
        """Test evolve method for 1-qubit Clifford gates."""
        op = Operator(gate)
        pauli_list = PauliList(pauli_group_labels(1, True))
        value = [Operator(pauli) for pauli in pauli_list.evolve(gate)]
        value_h = [Operator(pauli) for pauli in pauli_list.evolve(gate, frame="h")]
        value_s = [Operator(pauli) for pauli in pauli_list.evolve(gate, frame="s")]
        if isinstance(gate, Clifford):
            value_inv = [Operator(pauli) for pauli in pauli_list.evolve(gate.adjoint())]
        else:
            value_inv = [Operator(pauli) for pauli in pauli_list.evolve(gate.inverse())]
        target = [op.adjoint().dot(pauli).dot(op) for pauli in pauli_list]
        self.assertListEqual(value, target)
        self.assertListEqual(value, value_h)
        self.assertListEqual(value_inv, value_s)

    @combine(
        gate=(
            CXGate(),
            CYGate(),
            CZGate(),
            SwapGate(),
            iSwapGate(),
            CPhaseGate(theta=np.pi),
            CRXGate(theta=np.pi),
            CRYGate(theta=np.pi),
            CRZGate(theta=np.pi),
            RXXGate(theta=np.pi / 2),
            RYYGate(theta=np.pi / 2),
            RZZGate(theta=np.pi / 2),
            RZXGate(theta=np.pi / 2),
            XXMinusYYGate(theta=np.pi),
            XXPlusYYGate(theta=-np.pi),
            ECRGate(),
            Clifford(CXGate()),
            Clifford(CYGate()),
            Clifford(CZGate()),
            Clifford(SwapGate()),
            Clifford(ECRGate()),
        )
    )
    def test_evolve_clifford2(self, gate):
        """Test evolve method for 2-qubit Clifford gates."""
        op = Operator(gate)
        pauli_list = PauliList(pauli_group_labels(2, True))
        value = [Operator(pauli) for pauli in pauli_list.evolve(gate)]
        value_h = [Operator(pauli) for pauli in pauli_list.evolve(gate, frame="h")]
        value_s = [Operator(pauli) for pauli in pauli_list.evolve(gate, frame="s")]
        if isinstance(gate, Clifford):
            value_inv = [Operator(pauli) for pauli in pauli_list.evolve(gate.adjoint())]
        else:
            value_inv = [Operator(pauli) for pauli in pauli_list.evolve(gate.inverse())]
        target = [op.adjoint().dot(pauli).dot(op) for pauli in pauli_list]
        self.assertListEqual(value, target)
        self.assertListEqual(value, value_h)
        self.assertListEqual(value_inv, value_s)

    def test_phase_dtype_evolve_clifford(self):
        """Test phase dtype during evolve method for Clifford gates."""
        gates = (
            IGate(),
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            CXGate(),
            CYGate(),
            CZGate(),
            SwapGate(),
            ECRGate(),
        )
        dtypes = [
            int,
            np.int8,
            np.uint8,
            np.int16,
            np.uint16,
            np.int32,
            np.uint32,
            np.int64,
            np.uint64,
        ]
        for gate, dtype in itertools.product(gates, dtypes):
            z = np.ones(gate.num_qubits, dtype=bool)
            x = np.ones(gate.num_qubits, dtype=bool)
            phase = (np.sum(z & x) % 4).astype(dtype)
            paulis = Pauli((z, x, phase))
            evo = paulis.evolve(gate)
            self.assertEqual(evo.phase.dtype, dtype)

    @combine(phase=(True, False))
    def test_evolve_clifford_qargs(self, phase):
        """Test evolve method for random Clifford"""
        cliff = random_clifford(3, seed=10)
        op = Operator(cliff)
        pauli_list = random_pauli_list(5, 3, seed=10, phase=phase)
        qargs = [3, 0, 1]
        value = [Operator(pauli) for pauli in pauli_list.evolve(cliff, qargs=qargs)]
        value_inv = [Operator(pauli) for pauli in pauli_list.evolve(cliff.adjoint(), qargs=qargs)]
        value_h = [Operator(pauli) for pauli in pauli_list.evolve(cliff, qargs=qargs, frame="h")]
        value_s = [Operator(pauli) for pauli in pauli_list.evolve(cliff, qargs=qargs, frame="s")]
        target = [
            Operator(pauli).compose(op.adjoint(), qargs=qargs).dot(op, qargs=qargs)
            for pauli in pauli_list
        ]
        self.assertListEqual(value, target)
        self.assertListEqual(value, value_h)
        self.assertListEqual(value_inv, value_s)

    @combine(qubit_wise=[True, False])
    def test_noncommutation_graph(self, qubit_wise):
        """Test noncommutation graph"""

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

        input_labels = ["IY", "ZX", "XZ", "-YI", "YX", "YY", "-iYZ", "ZI", "ZX", "ZY", "iZZ", "II"]
        np.random.shuffle(input_labels)
        pauli_list = PauliList(input_labels)
        graph = pauli_list.noncommutation_graph(qubit_wise=qubit_wise)

        expected = rx.PyGraph()
        expected.add_nodes_from(range(len(input_labels)))
        edges = [
            (ia, ib)
            for (ia, a), (ib, b) in itertools.combinations(enumerate(input_labels), 2)
            if not commutes(Pauli(a), Pauli(b), qubit_wise)
        ]
        expected.add_edges_from_no_data(edges)

        self.assertTrue(rx.is_isomorphic(graph, expected))

    def test_group_qubit_wise_commuting(self):
        """Test grouping qubit-wise commuting operators"""

        def qubitwise_commutes(left: Pauli, right: Pauli) -> bool:
            return len(left) == len(right) and all(a.commutes(b) for a, b in zip(left, right))

        input_labels = ["IY", "ZX", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "iZZ", "II"]
        np.random.shuffle(input_labels)
        pauli_list = PauliList(input_labels)
        groups = pauli_list.group_qubit_wise_commuting()

        # checking that every input Pauli in pauli_list is in a group in the output
        output_labels = [pauli.to_label() for group in groups for pauli in group]
        self.assertListEqual(sorted(output_labels), sorted(input_labels))

        # Within each group, every operator qubit-wise commutes with every other operator.
        for group in groups:
            self.assertTrue(
                all(
                    qubitwise_commutes(pauli1, pauli2)
                    for pauli1, pauli2 in itertools.combinations(group, 2)
                )
            )
        # For every pair of groups, at least one element from one does not qubit-wise commute with
        # at least one element of the other.
        for group1, group2 in itertools.combinations(groups, 2):
            self.assertFalse(
                all(
                    qubitwise_commutes(group1_pauli, group2_pauli)
                    for group1_pauli, group2_pauli in itertools.product(group1, group2)
                )
            )

    def test_group_commuting(self):
        """Test general grouping commuting operators"""

        def commutes(left: Pauli, right: Pauli) -> bool:
            return len(left) == len(right) and left.commutes(right)

        input_labels = ["IY", "ZX", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "iZZ", "II"]
        np.random.shuffle(input_labels)
        pauli_list = PauliList(input_labels)
        #  if qubit_wise=True, equivalent to test_group_qubit_wise_commuting
        groups = pauli_list.group_commuting(qubit_wise=False)

        # checking that every input Pauli in pauli_list is in a group in the output
        output_labels = [pauli.to_label() for group in groups for pauli in group]
        self.assertListEqual(sorted(output_labels), sorted(input_labels))
        # Within each group, every operator commutes with every other operator.
        for group in groups:
            self.assertTrue(
                all(commutes(pauli1, pauli2) for pauli1, pauli2 in itertools.combinations(group, 2))
            )
        # For every pair of groups, at least one element from one group does not commute with
        # at least one element of the other.
        for group1, group2 in itertools.combinations(groups, 2):
            self.assertFalse(
                all(
                    commutes(group1_pauli, group2_pauli)
                    for group1_pauli, group2_pauli in itertools.product(group1, group2)
                )
            )


if __name__ == "__main__":
    unittest.main()
