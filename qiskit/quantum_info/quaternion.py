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

"""
A module for using quaternions.
"""
from __future__ import annotations
import math
import numpy as np


class Quaternion:
    """A class representing a Quaternion."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)

    def __call__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return np.array_str(self.data)

    def __str__(self):
        return np.array_str(self.data)

    def __mul__(self, r):
        if isinstance(r, Quaternion):
            q = self
            out_data = np.zeros(4, dtype=float)
            out_data[0] = r(0) * q(0) - r(1) * q(1) - r(2) * q(2) - r(3) * q(3)
            out_data[1] = r(0) * q(1) + r(1) * q(0) - r(2) * q(3) + r(3) * q(2)
            out_data[2] = r(0) * q(2) + r(1) * q(3) + r(2) * q(0) - r(3) * q(1)
            out_data[3] = r(0) * q(3) - r(1) * q(2) + r(2) * q(1) + r(3) * q(0)
            return Quaternion(out_data)
        else:
            return NotImplemented

    def norm(self):
        """Norm of quaternion."""
        import scipy.linalg as la

        return la.norm(self.data)

    def normalize(self, inplace: bool = False) -> Quaternion:
        """Normalizes a Quaternion to unit length
        so that it represents a valid rotation.

        Args:
            inplace (bool): Do an inplace normalization.

        Returns:
            Quaternion: Normalized quaternion.
        """
        if inplace:
            nrm = self.norm()
            self.data /= nrm
            return None
        nrm = self.norm()
        data_copy = np.array(self.data, copy=True)
        data_copy /= nrm
        return Quaternion(data_copy)

    def to_matrix(self) -> np.ndarray:
        """Converts a unit-length quaternion to a rotation matrix.

        Returns:
            ndarray: Rotation matrix.
        """
        w, x, y, z = self.normalize().data
        mat = np.array(
            [
                [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
                [2 * x * y + 2 * z * w, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * x * w],
                [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x**2 - 2 * y**2],
            ],
            dtype=float,
        )
        return mat

    def to_zyz(self) -> np.ndarray:
        """Converts a unit-length quaternion to a sequence
        of ZYZ Euler angles.

        Returns:
            ndarray: Array of Euler angles.
        """
        mat = self.to_matrix()
        euler = np.zeros(3, dtype=float)
        if mat[2, 2] < 1:
            if mat[2, 2] > -1:
                euler[0] = math.atan2(mat[1, 2], mat[0, 2])
                euler[1] = math.acos(mat[2, 2])
                euler[2] = math.atan2(mat[2, 1], -mat[2, 0])
            else:
                euler[0] = -math.atan2(mat[1, 0], mat[1, 1])
                euler[1] = np.pi
        else:
            euler[0] = math.atan2(mat[1, 0], mat[1, 1])
        return euler

    @classmethod
    def from_axis_rotation(cls, angle: float, axis: str) -> Quaternion:
        """Return quaternion for rotation about given axis.

        Args:
            angle (float): Angle in radians.
            axis (str): Axis for rotation

        Returns:
            Quaternion: Quaternion for axis rotation.

        Raises:
            ValueError: Invalid input axis.
        """
        out = np.zeros(4, dtype=float)
        if axis == "x":
            out[1] = 1
        elif axis == "y":
            out[2] = 1
        elif axis == "z":
            out[3] = 1
        else:
            raise ValueError("Invalid axis input.")
        out *= math.sin(angle / 2.0)
        out[0] = math.cos(angle / 2.0)
        return cls(out)

    @classmethod
    def from_euler(cls, angles: list | np.ndarray, order: str = "yzy") -> Quaternion:
        """Generate a quaternion from a set of Euler angles.

        Args:
            angles (array_like): Array of Euler angles.
            order (str): Order of Euler rotations.  'yzy' is default.

        Returns:
            Quaternion: Quaternion representation of Euler rotation.
        """
        angles = np.asarray(angles, dtype=float)
        quat = (
            cls.from_axis_rotation(angles[0], order[0])
            * cls.from_axis_rotation(angles[1], order[1])
            * cls.from_axis_rotation(angles[2], order[2])
        )
        quat.normalize(inplace=True)
        return quat
