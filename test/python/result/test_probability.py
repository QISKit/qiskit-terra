# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test conversion to probability distribution"""

from qiskit.result import ProbDistribution
from test import QiskitTestCase


class TestProbDistribution(QiskitTestCase):
    """Tests for probsdistributions."""

    def test_hex_probs(self):
        """Test hexadecimal input."""
        in_probs = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {0: 2 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 2 / 7}
        self.assertEqual(expected, probs)

    def test_bin_probs(self):
        """Test binary input."""
        in_probs = {"0b0": 2 / 7, "0b1": 1 / 7, "0b10": 1 / 7, "0b11": 1 / 7, "0b100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {0: 2 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 2 / 7}
        self.assertEqual(expected, probs)

    def test_bin_probs_no_0b(self):
        """Test binary input without 0b in front."""
        in_probs = {"000": 2 / 7, "001": 1 / 7, "010": 1 / 7, "011": 1 / 7, "100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {0: 2 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 2 / 7}
        self.assertEqual(expected, probs)

    def test_bin_probs2(self):
        """Test binary input."""
        in_probs = {"000": 2 / 7, "001": 1 / 7, "010": 1 / 7, "011": 1 / 7, "100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {0: 2 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 2 / 7}
        self.assertEqual(expected, probs)

    def test_bin_no_prefix_probs(self):
        """Test binary input without 0b prefix."""
        in_probs = {"0": 2 / 7, "1": 1 / 7, "10": 1 / 7, "11": 1 / 7, "100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {0: 2 / 7, 1: 1 / 7, 2: 1 / 7, 3: 1 / 7, 4: 2 / 7}
        self.assertEqual(expected, probs)

    def test_hex_probs_hex_out(self):
        """Test hexadecimal input and hexadecimal output."""
        in_probs = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        probs = ProbDistribution(in_probs)
        self.assertEqual(in_probs, probs.hex_probabilities())

    def test_bin_probs_hex_out(self):
        """Test binary input and hexadecimal output."""
        in_probs = {"0b0": 2 / 7, "0b1": 1 / 7, "0b10": 1 / 7, "0b11": 1 / 7, "0b100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        self.assertEqual(expected, probs.hex_probabilities())

    def test_bin_no_prefix_probs_hex_out(self):
        """Test binary input without a 0b prefix and hexadecimal output."""
        in_probs = {"0": 2 / 7, "1": 1 / 7, "10": 1 / 7, "11": 1 / 7, "100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        self.assertEqual(expected, probs.hex_probabilities())

    def test_hex_probs_bin_out(self):
        """Test hexadecimal input and binary output."""
        in_probs = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {"000": 2 / 7, "001": 1 / 7, "010": 1 / 7, "011": 1 / 7, "100": 2 / 7}
        self.assertEqual(expected, probs.binary_probabilities())

    def test_bin_probs_bin_out(self):
        """Test binary input and binary output."""
        in_probs = {"0b0": 2 / 7, "0b1": 1 / 7, "0b10": 1 / 7, "0b11": 1 / 7, "0b100": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {"000": 2 / 7, "001": 1 / 7, "010": 1 / 7, "011": 1 / 7, "100": 2 / 7}
        self.assertEqual(expected, probs.binary_probabilities())

    def test_bin_no_prefix_probs_bin_out(self):
        """Test binary input without a 0b prefix and binary output."""
        in_probs = {"000": 2 / 7, "001": 1 / 7, "010": 1 / 7, "011": 1 / 7, "100": 2 / 7}
        probs = ProbDistribution(in_probs)
        self.assertEqual(in_probs, probs.binary_probabilities())

    def test_bin_no_prefix_w_heading_zeros_probs_bin_out(self):
        """Test binary input without a 0b prefix with heading 0 and binary output."""
        in_probs = {"00000": 2 / 7, "00001": 1 / 7, "00010": 1 / 7, "00011": 1 / 7, "00100": 2 / 7}
        probs = ProbDistribution(in_probs)
        self.assertEqual(in_probs, probs.binary_probabilities())

    def test_bin_no_prefix_w_diff_heading_zero_probs_bin_out(self):
        """Test binary input without a 0b prefix with heading 0 of different sizes and binary output."""
        in_probs = {
            "0": 3 / 5,
            "01": 1 / 2,
            "10": 7 / 20,
            "011": 1 / 10,
            "00100": -11 / 20,
        }
        probs = ProbDistribution(in_probs)
        expected = {
            "00000": 3 / 5,
            "00001": 1 / 2,
            "00010": 7 / 20,
            "00011": 1 / 10,
            "00100": -11 / 20,
        }
        self.assertEqual(expected, probs.binary_probabilities())

    def test_bin_no_prefix_w_diff_heading_zero_probs_bin_out_padded(self):
        """Test binary input without a 0b prefix with heading 0 of different sizes and binary output,
        padded with zeros."""
        in_probs = {
            "0": 3 / 5,
            "01": 1 / 2,
            "10": 7 / 20,
            "011": 1 / 10,
            "00100": -11 / 20,
        }
        probs = ProbDistribution(in_probs)
        expected = {
            "0000000": 3 / 5,
            "0000001": 1 / 2,
            "0000010": 7 / 20,
            "0000011": 1 / 10,
            "0000100": -11 / 20,
        }
        self.assertEqual(expected, probs.binary_probabilities(7))

    def test_bin_no_prefix_out_padded(self):
        """Test binary input without a 0b prefix, padded with zeros."""
        n = 5
        in_probs = {"0": 1}
        probs = ProbDistribution(in_probs)
        expected = {"0" * n: 1}
        self.assertEqual(expected, probs.binary_probabilities(num_bits=n))

    def test_hex_probs_bin_out_padded(self):
        """Test hexadecimal input and binary output, padded with zeros."""
        in_probs = {"0x0": 2 / 7, "0x1": 1 / 7, "0x2": 1 / 7, "0x3": 1 / 7, "0x4": 2 / 7}
        probs = ProbDistribution(in_probs)
        expected = {"0000": 2 / 7, "0001": 1 / 7, "0010": 1 / 7, "0011": 1 / 7, "0100": 2 / 7}
        self.assertEqual(expected, probs.binary_probabilities(num_bits=4))

    def test_empty(self):
        """Test empty input."""
        probs = ProbDistribution({})
        self.assertEqual(probs, {})

    def test_empty_hex_out(self):
        """Test empty input with hexadecimal output."""
        probs = ProbDistribution({})
        self.assertEqual(probs.hex_probabilities(), {})

    def test_empty_bin_out(self):
        """Test empty input with binary output."""
        probs = ProbDistribution({})
        self.assertEqual(probs.binary_probabilities(), {})

    def test_empty_bin_out_padding(self):
        """Test empty input with binary output and padding."""
        probs = ProbDistribution({})
        self.assertEqual(probs.binary_probabilities(5), {})

    def test_invalid_keys(self):
        """Test invalid key type raises."""
        with self.assertRaises(TypeError):
            ProbDistribution({1 + 2j: 3 / 5})

    def test_invalid_key_string(self):
        """Test invalid key string format raises."""
        with self.assertRaises(ValueError):
            ProbDistribution({"1a2b": 3 / 5})
