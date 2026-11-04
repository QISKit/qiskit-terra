# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests quaternion conversion"""

import math
import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg as la

from qiskit.quantum_info.quaternion import Quaternion
from test import QiskitTestCase


class TestQuaternions(QiskitTestCase):
    """Tests qiskit.quantum_info.operators.quaternion"""

    def setUp(self):
        super().setUp()
        self.rnd_array = np.array([0.5, 0.8, 0.9, -0.3])
        self.quat_unnormalized = Quaternion(self.rnd_array)
        axes = ["x", "y", "z"]
        rnd = np.array([-0.92545003, -2.19985357, 6.01761209])
        idx = np.array([0, 2, 1])
        self.mat1 = rotation_matrix(rnd[0], axes[idx[0]]).dot(
            rotation_matrix(rnd[1], axes[idx[1]]).dot(rotation_matrix(rnd[2], axes[idx[2]]))
        )
        axes_str = "".join(axes[i] for i in idx)
        quat = Quaternion.from_euler(rnd, axes_str)
        self.mat2 = quat.to_matrix()

    def test_str(self):
        """Quaternion should have a correct string representation."""
        self.assertEqual(self.quat_unnormalized.__str__(), self.rnd_array.__str__())

    def test_repr(self):
        """Quaternion should have a correct string representation."""
        self.assertEqual(self.quat_unnormalized.__repr__(), self.rnd_array.__str__())

    def test_norm(self):
        """Quaternions should give correct norm."""
        norm = la.norm(self.rnd_array)
        self.assertEqual(norm, self.quat_unnormalized.norm())

    def test_normalize(self):
        """Quaternions should be normalizable"""
        self.assertAlmostEqual(self.quat_unnormalized.normalize().norm(), 1, places=5)

    def test_random_euler(self):
        """Quaternion from Euler rotations."""
        assert_allclose(self.mat1, self.mat2)

    def test_orthogonality(self):
        """Quaternion rotation matrix orthogonality"""
        assert_allclose(self.mat2.dot(self.mat2.T), np.identity(3, dtype=float), atol=1e-8)

    def test_det(self):
        """Quaternion det = 1"""
        assert_allclose(la.det(self.mat2), 1)

    def test_equiv_quaternions(self):
        """Different Euler rotations give same quaternion, up to sign."""
        # Check if Euler angles from to_zyz return same quaternion
        # up to a sign (2pi rotation)
        rot = ["xyz", "xyx", "xzy", "xzx", "yzx", "yzy", "yxz", "yxy", "zxy", "zxz", "zyx", "zyz"]
        for value in rot:
            rnd = np.array([-1.57657536, 5.66384302, 2.91532185])
            quat1 = Quaternion.from_euler(rnd, value)
            euler = quat1.to_zyz()
            quat2 = Quaternion.from_euler(euler, "zyz")
            assert_allclose(abs(quat1.data.dot(quat2.data)), 1)

    def test_mul_by_quat(self):
        """Quaternions should multiply correctly."""
        # multiplication of quaternions is equivalent to the
        # multiplication of corresponding rotation matrices.
        other_quat = Quaternion(np.array([0.4, 0.2, -0.7, 0.8]))
        other_mat = other_quat.to_matrix()
        product_quat = self.quat_unnormalized * other_quat
        product_mat = (self.quat_unnormalized.to_matrix()).dot(other_mat)
        assert_allclose(product_quat.to_matrix(), product_mat)

    def test_mul_by_array(self):
        """Quaternions cannot be multiplied with an array."""
        other_array = np.array([0.1, 0.2, 0.3, 0.4])
        with self.assertRaises(TypeError):
            _ = self.quat_unnormalized * other_array

    def test_mul_by_scalar(self):
        """Quaternions cannot be multiplied with a scalar."""
        other_scalar = 0.123456789
        with self.assertRaises(TypeError):
            _ = self.quat_unnormalized * other_scalar

    def test_rotation(self):
        """Multiplication by -1 should give the same rotation."""
        neg_quat = Quaternion(self.quat_unnormalized.data * -1)
        assert_allclose(neg_quat.to_matrix(), self.quat_unnormalized.to_matrix())

    def test_one_euler_angle(self):
        """Quaternion should return a correct sequence of zyz representation
        in the case of rotations when there is only one non-zero Euler angle."""
        rand_rot_angle = 0.123456789
        some_quat = Quaternion.from_axis_rotation(rand_rot_angle, "z")
        assert_allclose(some_quat.to_zyz(), np.array([rand_rot_angle, 0, 0]))

    def test_two_euler_angle_0123456789(self):
        """Quaternion should return a correct sequence of zyz representation
        in the case of rotations when there are only two non-zero Euler angle.
        angle = 0.123456789"""
        rand_rot_angle = 0.123456789
        some_quat = Quaternion.from_axis_rotation(
            rand_rot_angle, "z"
        ) * Quaternion.from_axis_rotation(np.pi, "y")
        assert_allclose(some_quat.to_zyz(), np.array([rand_rot_angle, np.pi, 0]))

    def test_two_euler_angle_0987654321(self):
        """Quaternion should return a correct sequence of zyz representation
        in the case of rotations when there are only two non-zero Euler angle.
        angle = 0.987654321"""
        rand_rot_angle = 0.987654321
        some_quat = Quaternion.from_axis_rotation(
            rand_rot_angle, "z"
        ) * Quaternion.from_axis_rotation(np.pi, "y")
        assert_allclose(some_quat.to_zyz(), np.array([rand_rot_angle, np.pi, 0]))

    def test_quaternion_from_rotation_invalid_axis(self):
        """Cannot generate quaternion from rotations around invalid axis."""
        rand_axis = "a"
        rand_angle = 0.123456789
        self.assertRaises(ValueError, Quaternion.from_axis_rotation, rand_angle, rand_axis)


def rotation_matrix(angle, axis):
    """Generates a rotation matrix for a given angle and axis.

    Args:
        angle (float): Rotation angle in radians.
        axis (str): Axis for rotation: 'x', 'y', 'z'

    Returns:
        ndarray: Rotation matrix.

    Raises:
        ValueError: Invalid input axis.
    """
    direction = np.zeros(3, dtype=float)
    if axis == "x":
        direction[0] = 1
    elif axis == "y":
        direction[1] = 1
    elif axis == "z":
        direction[2] = 1
    else:
        raise ValueError("Invalid axis.")
    direction = np.asarray(direction, dtype=float)
    sin_angle = math.sin(angle)
    cos_angle = math.cos(angle)
    rot = np.diag([cos_angle, cos_angle, cos_angle])
    rot += np.outer(direction, direction) * (1.0 - cos_angle)
    direction *= sin_angle
    rot += np.array(
        [
            [0, -direction[2], direction[1]],
            [direction[2], 0, -direction[0]],
            [-direction[1], direction[0], 0],
        ]
    )
    return rot
