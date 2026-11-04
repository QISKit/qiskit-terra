# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Counts class."""

import unittest

import numpy as np

from qiskit.result import counts
from qiskit import exceptions
from qiskit.result import utils


class TestCounts(unittest.TestCase):
    def test_just_counts(self):
        raw_counts = {"0x0": 21, "0x2": 12}
        expected = {"0": 21, "10": 12}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_counts_with_exta_formatting_data(self):
        raw_counts = {"0x0": 4, "0x2": 10}
        expected = {"0 0 00": 4, "0 0 10": 10}
        result = counts.Counts(
            raw_counts, "test_counts", creg_sizes=[["c0", 2], ["c0", 1], ["c1", 1]], memory_slots=4
        )
        self.assertEqual(result, expected)

    def test_marginal_counts(self):
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_counts(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution(self):
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_numpy_indices(self):
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        expected = {"00": 4, "01": 27, "10": 23}
        indices = np.asarray([0, 1])
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, indices)
        self.assertEqual(expected, result)

    def test_int_outcomes(self):
        raw_counts = {"0x0": 21, "0x2": 12, "0x3": 5, "0x2E": 265}
        expected = {0: 21, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(expected, result)

    def test_most_frequent(self):
        raw_counts = {"0x0": 21, "0x2": 12, "0x3": 5, "0x2E": 265}
        expected = "101110"
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.most_frequent()
        self.assertEqual(expected, result)

    def test_most_frequent_duplicate(self):
        raw_counts = {"0x0": 265, "0x2": 12, "0x3": 5, "0x2E": 265}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.most_frequent)

    def test_hex_outcomes(self):
        raw_counts = {"0x0": 21, "0x2": 12, "0x3": 5, "0x2E": 265}
        expected = {"0x0": 21, "0x2": 12, "0x3": 5, "0x2e": 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.hex_outcomes()
        self.assertEqual(expected, result)

    def test_just_int_counts(self):
        raw_counts = {0: 21, 2: 12}
        expected = {"0": 21, "10": 12}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_int_counts_with_exta_formatting_data(self):
        raw_counts = {0: 4, 2: 10}
        expected = {"0 0 00": 4, "0 0 10": 10}
        result = counts.Counts(
            raw_counts, "test_counts", creg_sizes=[["c0", 2], ["c0", 1], ["c1", 1]], memory_slots=4
        )
        self.assertEqual(result, expected)

    def test_marginal_int_counts(self):
        raw_counts = {0: 4, 1: 7, 2: 10, 6: 5, 9: 11, 13: 9, 14: 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_counts(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_int_counts(self):
        raw_counts = {0: 4, 1: 7, 2: 10, 6: 5, 9: 11, 13: 9, 14: 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_int_counts_numpy_64_bit(self):
        raw_counts = {
            0: np.int64(4),
            1: np.int64(7),
            2: np.int64(10),
            6: np.int64(5),
            9: np.int64(11),
            13: np.int64(9),
            14: np.int64(8),
        }
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_int_counts_numpy_8_bit(self):
        raw_counts = {
            0: np.int8(4),
            1: np.int8(7),
            2: np.int8(10),
            6: np.int8(5),
            9: np.int8(11),
            13: np.int8(9),
            14: np.int8(8),
        }
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_int_outcomes_with_int_counts(self):
        raw_counts = {0: 21, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(raw_counts, result)

    def test_most_frequent_int_counts(self):
        raw_counts = {0: 21, 2: 12, 3: 5, 46: 265}
        expected = "101110"
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.most_frequent()
        self.assertEqual(expected, result)

    def test_most_frequent_duplicate_int_counts(self):
        raw_counts = {0: 265, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.most_frequent)

    def test_hex_outcomes_int_counts(self):
        raw_counts = {0: 265, 2: 12, 3: 5, 46: 265}
        expected = {"0x0": 265, "0x2": 12, "0x3": 5, "0x2e": 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.hex_outcomes()
        self.assertEqual(expected, result)

    def test_invalid_input_type(self):
        self.assertRaises(TypeError, counts.Counts, {2.4: 1024})

    def test_mixed_input_types(self):
        self.assertRaises(TypeError, counts.Counts, {"0101": 1, 3: 1})

    def test_just_bitstring_counts(self):
        raw_counts = {"0": 21, "10": 12}
        expected = {"0": 21, "10": 12}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_bistring_counts_with_exta_formatting_data(self):
        raw_counts = {"0": 4, "10": 10}
        expected = {"0 0 00": 4, "0 0 10": 10}
        result = counts.Counts(
            raw_counts, "test_counts", creg_sizes=[["c0", 2], ["c0", 1], ["c1", 1]], memory_slots=4
        )
        self.assertEqual(result, expected)

    def test_marginal_bitstring_counts(self):
        raw_counts = {"0": 4, "1": 7, "10": 10, "110": 5, "1001": 11, "1101": 9, "1110": 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_counts(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_bitstring_counts(self):
        raw_counts = {"0": 4, "1": 7, "10": 10, "110": 5, "1001": 11, "1101": 9, "1110": 8}
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_int_outcomes_with_bitstring_counts(self):
        raw_counts = {"0": 21, "10": 12, "11": 5, "101110": 265}
        expected = {0: 21, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(expected, result)

    def test_most_frequent_bitstring_counts(self):
        raw_counts = {"0": 21, "10": 12, "11": 5, "101110": 265}
        expected = "101110"
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.most_frequent()
        self.assertEqual(expected, result)

    def test_most_frequent_duplicate_bitstring_counts(self):
        raw_counts = {"0": 265, "10": 12, "11": 5, "101110": 265}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.most_frequent)

    def test_hex_outcomes_bitstring_counts(self):
        raw_counts = {"0": 265, "10": 12, "11": 5, "101110": 265}
        expected = {"0x0": 265, "0x2": 12, "0x3": 5, "0x2e": 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.hex_outcomes()
        self.assertEqual(expected, result)

    def test_qudit_counts(self):
        raw_counts = {
            "00": 121,
            "01": 109,
            "02": 114,
            "10": 113,
            "11": 106,
            "12": 114,
            "20": 117,
            "21": 104,
            "22": 102,
        }
        result = counts.Counts(raw_counts)
        self.assertEqual(raw_counts, result)

    def test_qudit_counts_raises_with_format(self):
        raw_counts = {
            "00": 121,
            "01": 109,
            "02": 114,
            "10": 113,
            "11": 106,
            "12": 114,
            "20": 117,
            "21": 104,
            "22": 102,
        }
        self.assertRaises(exceptions.QiskitError, counts.Counts, raw_counts, creg_sizes=[["c0", 4]])

    def test_qudit_counts_hex_outcome(self):
        raw_counts = {
            "00": 121,
            "01": 109,
            "02": 114,
            "10": 113,
            "11": 106,
            "12": 114,
            "20": 117,
            "21": 104,
            "22": 102,
        }
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.hex_outcomes)

    def test_qudit_counts_int_outcome(self):
        raw_counts = {
            "00": 121,
            "01": 109,
            "02": 114,
            "10": 113,
            "11": 106,
            "12": 114,
            "20": 117,
            "21": 104,
            "22": 102,
        }
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.int_outcomes)

    def test_qudit_counts_most_frequent(self):
        raw_counts = {
            "00": 121,
            "01": 109,
            "02": 114,
            "10": 113,
            "11": 106,
            "12": 114,
            "20": 117,
            "21": 104,
            "22": 102,
        }
        counts_obj = counts.Counts(raw_counts)
        self.assertEqual("00", counts_obj.most_frequent())

    def test_just_0b_bitstring_counts(self):
        raw_counts = {"0b0": 21, "0b10": 12}
        expected = {"0": 21, "10": 12}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_0b_bistring_counts_with_exta_formatting_data(self):
        raw_counts = {"0b0": 4, "0b10": 10}
        expected = {"0 0 00": 4, "0 0 10": 10}
        result = counts.Counts(
            raw_counts, "test_counts", creg_sizes=[["c0", 2], ["c0", 1], ["c1", 1]], memory_slots=4
        )
        self.assertEqual(result, expected)

    def test_marginal_0b_string_counts(self):
        raw_counts = {
            "0b0": 4,
            "0b1": 7,
            "0b10": 10,
            "0b110": 5,
            "0b1001": 11,
            "0b1101": 9,
            "0b1110": 8,
        }
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_counts(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_marginal_distribution_0b_string_counts(self):
        raw_counts = {
            "0b0": 4,
            "0b1": 7,
            "0b10": 10,
            "0b110": 5,
            "0b1001": 11,
            "0b1101": 9,
            "0b1110": 8,
        }
        expected = {"00": 4, "01": 27, "10": 23}
        counts_obj = counts.Counts(raw_counts, creg_sizes=[["c0", 4]], memory_slots=4)
        result = utils.marginal_distribution(counts_obj, [0, 1])
        self.assertEqual(expected, result)

    def test_int_outcomes_with_0b_bitstring_counts(self):
        raw_counts = {"0b0": 21, "0b10": 12, "0b11": 5, "0b101110": 265}
        expected = {0: 21, 2: 12, 3: 5, 46: 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(expected, result)

    def test_most_frequent_0b_bitstring_counts(self):
        raw_counts = {"0b0": 21, "0b10": 12, "0b11": 5, "0b101110": 265}
        expected = "101110"
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.most_frequent()
        self.assertEqual(expected, result)

    def test_most_frequent_duplicate_0b_bitstring_counts(self):
        raw_counts = {"0b0": 265, "0b10": 12, "0b11": 5, "0b101110": 265}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.most_frequent)

    def test_hex_outcomes_0b_bitstring_counts(self):
        raw_counts = {"0b0": 265, "0b10": 12, "0b11": 5, "0b101110": 265}
        expected = {"0x0": 265, "0x2": 12, "0x3": 5, "0x2e": 265}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.hex_outcomes()
        self.assertEqual(expected, result)

    def test_empty_bitstring_counts(self):
        raw_counts = {}
        expected = {}
        result = counts.Counts(raw_counts)
        self.assertEqual(expected, result)

    def test_empty_bistring_counts_with_exta_formatting_data(self):
        raw_counts = {}
        expected = {}
        result = counts.Counts(
            raw_counts, "test_counts", creg_sizes=[["c0", 2], ["c0", 1], ["c1", 1]], memory_slots=4
        )
        self.assertEqual(result, expected)

    def test_int_outcomes_with_empty_counts(self):
        raw_counts = {}
        expected = {}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.int_outcomes()
        self.assertEqual(expected, result)

    def test_most_frequent_empty_bitstring_counts(self):
        raw_counts = {}
        counts_obj = counts.Counts(raw_counts)
        self.assertRaises(exceptions.QiskitError, counts_obj.most_frequent)

    def test_hex_outcomes_empty_bitstring_counts(self):
        raw_counts = {}
        expected = {}
        counts_obj = counts.Counts(raw_counts)
        result = counts_obj.hex_outcomes()
        self.assertEqual(expected, result)
