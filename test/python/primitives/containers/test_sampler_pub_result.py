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


"""Unit tests for SamplerPubResult."""

from test import QiskitTestCase

import numpy as np

from qiskit.primitives.containers import BitArray, DataBin, SamplerPubResult


class SamplerPubResultCase(QiskitTestCase):
    """Test the SamplerPubResult class."""

    def test_construction(self):
        """Test that the constructor works."""
        ba = BitArray.from_samples(["00", "11"], 2)
        counts = {"00": 1, "11": 1}
        data_bin = DataBin(a=ba, b=ba)
        pub_result = SamplerPubResult(data_bin)
        self.assertEqual(pub_result.data.a.get_counts(), counts)
        self.assertEqual(pub_result.data.b.get_counts(), counts)
        self.assertEqual(pub_result.metadata, {})

        pub_result = SamplerPubResult(data_bin, {"x": 1})
        self.assertEqual(pub_result.data.a.get_counts(), counts)
        self.assertEqual(pub_result.data.b.get_counts(), counts)
        self.assertEqual(pub_result.metadata, {"x": 1})

    def test_repr(self):
        """Test that the repr doesn't fail"""
        # we are primarily interested in making sure some future change doesn't cause the repr to
        # raise an error. it is more sensible for humans to detect a deficiency in the formatting
        # itself, should one be uncovered
        ba = BitArray.from_samples(["00", "11"], 2)
        data_bin = DataBin(a=ba, b=ba)
        self.assertTrue(repr(SamplerPubResult(data_bin)).startswith("SamplerPubResult"))
        self.assertTrue(repr(SamplerPubResult(data_bin, {"x": 1})).startswith("SamplerPubResult"))

    def test_join_data_failures(self):
        """Test the join_data() failure mechanisms work."""

        result = SamplerPubResult(DataBin())
        with self.assertRaisesRegex(ValueError, "No entry exists in the data bin"):
            result.join_data()

        alpha = BitArray.from_samples(["00", "11"], 2)
        beta = BitArray.from_samples(["010", "101"], 3)
        result = SamplerPubResult(DataBin(alpha=alpha, beta=beta))
        with self.assertRaisesRegex(ValueError, "An empty name list is given"):
            result.join_data([])

        alpha = BitArray.from_samples(["00", "11"], 2)
        beta = BitArray.from_samples(["010", "101"], 3)
        result = SamplerPubResult(DataBin(alpha=alpha, beta=beta))
        with self.assertRaisesRegex(ValueError, "Name 'foo' does not exist"):
            result.join_data(["alpha", "foo"])

        alpha = BitArray.from_samples(["00", "11"], 2)
        beta = np.empty((2,))
        result = SamplerPubResult(DataBin(alpha=alpha, beta=beta))
        with self.assertRaisesRegex(TypeError, "Data comes from incompatible types"):
            result.join_data()

        alpha = np.empty((2,))
        beta = BitArray.from_samples(["00", "11"], 2)
        result = SamplerPubResult(DataBin(alpha=alpha, beta=beta))
        with self.assertRaisesRegex(TypeError, "Data comes from incompatible types"):
            result.join_data()

        result = SamplerPubResult(DataBin(alpha=1, beta={}))
        with self.assertRaisesRegex(TypeError, "Data comes from incompatible types"):
            result.join_data()

    def test_join_data_bit_array_default(self):
        """Test the join_data() method with no arguments and bit arrays."""
        alpha = BitArray.from_samples(["00", "11"], 2)
        beta = BitArray.from_samples(["010", "101"], 3)
        data_bin = DataBin(alpha=alpha, beta=beta)
        result = SamplerPubResult(data_bin)

        gamma = result.join_data()
        self.assertEqual(list(gamma.get_bitstrings()), ["01000", "10111"])

    def test_join_data_ndarray_default(self):
        """Test the join_data() method with no arguments and ndarrays."""
        alpha = np.linspace(0, 1, 30).reshape((2, 3, 5))
        beta = np.linspace(0, 1, 12).reshape((2, 3, 2))
        data_bin = DataBin(alpha=alpha, beta=beta, shape=(2, 3))
        result = SamplerPubResult(data_bin)

        gamma = result.join_data()
        np.testing.assert_allclose(gamma, np.concatenate([alpha, beta], axis=2))
