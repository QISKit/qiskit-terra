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


"""Unit tests for DataBin."""


import numpy as np

from qiskit.primitives.containers.data_bin import DataBin
from test import QiskitTestCase


class DataBinTestCase(QiskitTestCase):
    """Test the DataBin class."""

    def test_make_databin_no_fields(self):
        """Test DataBin when no fields are given."""
        data_bin = DataBin()
        self.assertEqual(len(data_bin), 0)
        self.assertEqual(data_bin.shape, ())

    def test_data_bin_basic(self):
        """Test DataBin function basic access."""
        alpha = np.empty((10, 20), dtype=np.uint16)
        beta = np.empty((10, 20), dtype=int)
        my_bin = DataBin(alpha=alpha, beta=beta)

        self.assertEqual(len(my_bin), 2)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))
        self.assertTrue("alpha=" in str(my_bin))
        self.assertTrue(str(my_bin).startswith("DataBin"))
        self.assertEqual(my_bin._FIELDS, ("alpha", "beta"))
        self.assertEqual(my_bin._FIELD_TYPES, (np.ndarray, np.ndarray))

        my_bin = DataBin(beta=beta, alpha=alpha)
        self.assertTrue(np.all(my_bin.alpha == alpha))
        self.assertTrue(np.all(my_bin.beta == beta))

    def test_constructor_failures(self):
        """Test that the constructor fails when expected."""

        with self.assertRaisesRegex(ValueError, "Cannot assign with these field names"):
            DataBin(values=6)

        with self.assertRaisesRegex(ValueError, "does not lead with the shape"):
            DataBin(x=np.empty((5,)), shape=(1,))

        with self.assertRaisesRegex(ValueError, "does not lead with the shape"):
            DataBin(x=np.empty((5, 2, 3)), shape=(5, 2, 3, 4))

    def test_shape(self):
        """Test shape setting and attributes."""
        databin = DataBin(x=6, y=np.empty((2, 3)))
        self.assertEqual(databin.shape, ())

        databin = DataBin(x=np.empty((5, 2)), y=np.empty((5, 2, 6)), shape=(5, 2))
        self.assertEqual(databin.shape, (5, 2))

    def test_make_databin_mapping(self):
        """Test DataBin with mapping features."""
        data_bin = DataBin(alpha=10, beta={1: 2})
        self.assertEqual(len(data_bin), 2)

        with self.subTest("iterator"):
            iterator = iter(data_bin)
            key = next(iterator)
            self.assertEqual(key, "alpha")
            key = next(iterator)
            self.assertEqual(key, "beta")
            with self.assertRaises(StopIteration):
                _ = next(iterator)

        with self.subTest("keys"):
            lst = list(data_bin.keys())
            key = lst[0]
            self.assertEqual(key, "alpha")
            key = lst[1]
            self.assertEqual(key, "beta")

        with self.subTest("values"):
            lst = list(data_bin.values())
            val = lst[0]
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            val = lst[1]
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("items"):
            lst = list(data_bin.items())
            key, val = lst[0]
            self.assertEqual(key, "alpha")
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            key, val = lst[1]
            self.assertEqual(key, "beta")
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("contains"):
            self.assertIn("alpha", data_bin)
            self.assertIn("beta", data_bin)
            self.assertNotIn("gamma", data_bin)

        with self.subTest("getitem"):
            val = data_bin["alpha"]
            self.assertIsInstance(val, int)
            self.assertEqual(val, 10)
            val = data_bin["beta"]
            self.assertIsInstance(val, dict)
            self.assertEqual(val, {1: 2})

        with self.subTest("error"):
            with self.assertRaises(KeyError):
                _ = data_bin["gamma"]
