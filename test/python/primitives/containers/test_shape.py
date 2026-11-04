# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test shape.py module"""


import numpy as np

from qiskit.primitives.containers.shape import Shaped, ShapedMixin, array_coerce, shape_tuple
from test import QiskitTestCase


class DummyShaped(ShapedMixin):
    """Dummy ShapedMixin child for testing."""

    def __init__(self, arr):
        super().__init__()
        self._shape = arr.shape
        self._arr = arr

    def __getitem__(self, arg):
        return self._arr[arg]


class ShapedTestCase(QiskitTestCase):
    """Test the Shaped protocol class"""

    def test_ndarray_is_shaped(self):
        """Test that ndarrays are shaped"""
        self.assertTrue(isinstance(np.empty((1, 2, 3)), Shaped))

    def test_mixin_is_shaped(self):
        """Test that ShapedMixin is shaped"""
        self.assertTrue(isinstance(DummyShaped(np.empty((1, 2, 3))), Shaped))


class ShapedMixinTestCase(QiskitTestCase):
    """Test the ShapedMixin class"""

    def test_shape(self):
        """Test the shape attribute."""
        self.assertEqual(DummyShaped(np.empty((1, 2, 3))).shape, (1, 2, 3))
        self.assertEqual(DummyShaped(np.empty(())).shape, ())

    def test_ndim(self):
        """Test the ndim attribute."""
        self.assertEqual(DummyShaped(np.empty(())).ndim, 0)
        self.assertEqual(DummyShaped(np.empty((1, 2, 3))).ndim, 3)

    def test_size(self):
        """Test the size attribute."""
        self.assertEqual(DummyShaped(np.empty(())).size, 1)
        self.assertEqual(DummyShaped(np.empty((0, 1))).size, 0)
        self.assertEqual(DummyShaped(np.empty((1, 2, 3))).size, 6)

    def test_getitem(self):
        """Missing docstring."""
        arr = np.arange(100).reshape(2, 5, 10)
        np.testing.assert_allclose(DummyShaped(arr)[:, 0, :2], arr[:, 0, :2])


class ArrayCoerceTestCase(QiskitTestCase):
    """Test array_coerce() function."""

    def test_shaped(self):
        """Test that array_coerce() works with ShapedMixin objects."""
        sh = DummyShaped(np.empty((1, 2, 3)))
        self.assertIs(sh, array_coerce(sh))

    def test_ndarray(self):
        """Test that array_coerce() works with ndarray objects."""
        sh = np.arange(100).reshape(5, 2, 2, 5)
        np.testing.assert_allclose(sh, array_coerce(sh))


class ShapeTupleTestCase(QiskitTestCase):
    """Test shape_tuple() function."""

    def test_int(self):
        """Test shape_tuple() with int inputs."""
        self.assertEqual(shape_tuple(), ())
        self.assertEqual(shape_tuple(5), (5,))
        self.assertEqual(shape_tuple(5, 10), (5, 10))
        self.assertEqual(shape_tuple(1e2), (100,))

    def test_nested(self):
        """Test shape_tuple() with nested inputs."""
        self.assertEqual(shape_tuple(0, (), (1, (2, (3,)), (4, 5))), (0, 1, 2, 3, 4, 5))

    def test_exceptions(self):
        """Test shape_tuple() raises correctly."""
        with self.assertRaisesRegex(ValueError, "iterable or an integer"):
            shape_tuple(None)

        with self.assertRaisesRegex(ValueError, "iterable or an integer"):
            shape_tuple(1.5)
