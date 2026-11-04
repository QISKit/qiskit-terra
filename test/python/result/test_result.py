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

"""Test Qiskit's Result class."""

import numpy as np

from qiskit.result import models
from qiskit.result import marginal_counts
from qiskit.result import marginal_distribution
from qiskit.result import Result
from qiskit.exceptions import QiskitError
from test import QiskitTestCase


class TestResultOperations(QiskitTestCase):
    """Result operations methods."""

    def setUp(self):
        self.base_result_args = {
            "backend_name": "test_backend",
            "backend_version": "1.0.0",
            "job_id": "job-123",
            "success": True,
        }

        super().setUp()

    def generate_qiskit_result(self):
        """Generate standard Result for testing"""
        memory = [hex(ii) for ii in range(8)]
        counts = dict.fromkeys(memory, 1)
        data_1 = models.ExperimentResultData(counts=counts, memory=memory)
        exp_result_header_1 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_1 = models.ExperimentResult(
            shots=8, success=True, data=data_1, header=exp_result_header_1
        )

        result = Result(results=[exp_result_1], **self.base_result_args)
        return result

    def test_counts_no_header(self):
        """Test that counts are extracted properly without header."""
        raw_counts = {"0x0": 4, "0x2": 10}
        no_header_processed_counts = {
            bin(int(bs[2:], 16))[2:]: counts for (bs, counts) in raw_counts.items()
        }
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), no_header_processed_counts)

    def test_counts_header(self):
        """Test that counts are extracted properly with header."""
        raw_counts = {"0x0": 4, "0x2": 10}
        processed_counts = {"0 0 00": 4, "0 0 10": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts(0), processed_counts)

    def test_counts_by_name(self):
        """Test that counts are extracted properly by name."""
        raw_counts = {"0x0": 4, "0x2": 10}
        processed_counts = {"0 0 00": 4, "0 0 10": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {
            "creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]],
            "memory_slots": 4,
            "name": "a_name",
        }
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_counts("a_name"), processed_counts)

    def test_counts_duplicate_name(self):
        """Test results containing multiple entries of a single name will warn."""
        data = models.ExperimentResultData(counts={})
        exp_result_header = {"name": "foo"}
        exp_result = models.ExperimentResult(
            shots=14, success=True, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result] * 2, **self.base_result_args)

        with self.assertWarnsRegex(UserWarning, r"multiple.*foo"):
            result.get_counts("foo")

    def test_result_repr(self):
        """Test that repr is constructed correctly for a results object."""
        raw_counts = {"0x0": 4, "0x2": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)
        expected = (
            "Result(backend_name='test_backend', backend_version='1.0.0', "
            "job_id='job-123', success=True, results=[ExperimentResult("
            "shots=14, success=True, meas_level=2, data=ExperimentResultData("
            "counts={'0x0': 4, '0x2': 10}), header={'creg_sizes': [['c0', 2], "
            "['c0', 1], ['c1', 1]], 'memory_slots': 4})], date=None, "
            "status=None, header=None)"
        )
        self.assertEqual(expected, repr(result))

    def test_result_to_dict(self):
        """Test that dictionary is constructed correctly for a results object."""
        raw_counts = {"0x0": 4, "0x2": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data, header=exp_result_header
        )
        header = {"header-property": 1}
        result = Result(results=[exp_result], header=header, **self.base_result_args)

        expected = {
            "backend_name": "test_backend",
            "backend_version": "1.0.0",
            "header": {"header-property": 1},
            "date": None,
            "job_id": "job-123",
            "status": None,
            "success": True,
            "results": [
                {
                    "shots": 14,
                    "success": True,
                    "data": {"counts": {"0x0": 4, "0x2": 10}},
                    "meas_level": 2,
                    "header": {"creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]], "memory_slots": 4},
                }
            ],
        }

        self.assertEqual(expected, result.to_dict())

    def test_multiple_circuits_counts(self):
        """Test that counts are returned either as a list or a single item.

        Counts are returned as a list when multiple experiments are executed
        and get_counts() is called with no arguments. In all the other cases
        get_counts() returns a single item containing the counts for a
        single experiment.
        """
        raw_counts_1 = {"0x0": 5, "0x3": 12, "0x5": 9, "0xD": 6, "0xE": 2}
        processed_counts_1 = {"0000": 5, "0011": 12, "0101": 9, "1101": 6, "1110": 2}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_1 = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data_1, header=exp_result_header_1
        )

        raw_counts_2 = {"0x1": 0, "0x4": 3, "0x6": 6, "0xA": 1, "0xB": 2}
        processed_counts_2 = {"0001": 0, "0100": 3, "0110": 6, "1010": 1, "1011": 2}
        data_2 = models.ExperimentResultData(counts=raw_counts_2)
        exp_result_header_2 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_2 = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data_2, header=exp_result_header_2
        )

        raw_counts_3 = {"0xC": 27, "0xF": 20}
        processed_counts_3 = {"1100": 27, "1111": 20}
        data_3 = models.ExperimentResultData(counts=raw_counts_3)
        exp_result_header_3 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_3 = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data_3, header=exp_result_header_3
        )

        mult_result = Result(
            results=[exp_result_1, exp_result_2, exp_result_3], **self.base_result_args
        )
        sing_result = Result(results=[exp_result_1], **self.base_result_args)

        self.assertEqual(
            mult_result.get_counts(), [processed_counts_1, processed_counts_2, processed_counts_3]
        )
        self.assertEqual(sing_result.get_counts(), processed_counts_1)

    def test_marginal_counts(self):
        """Test that counts are marginalized correctly."""
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=54, success=True, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)
        expected_marginal_counts = {"00": 4, "01": 27, "10": 23}

        self.assertEqual(marginal_counts(result.get_counts(), [0, 1]), expected_marginal_counts)
        self.assertEqual(marginal_counts(result.get_counts(), [1, 0]), expected_marginal_counts)

    def test_marginal_distribution(self):
        """Test that counts are marginalized correctly."""
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=54, success=True, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)
        expected_marginal_counts = {"00": 4, "01": 27, "10": 23}
        expected_reverse = {"00": 4, "10": 27, "01": 23}

        self.assertEqual(
            marginal_distribution(result.get_counts(), [0, 1]), expected_marginal_counts
        )
        self.assertEqual(marginal_distribution(result.get_counts(), [1, 0]), expected_reverse)
        # test with register spacing, bitstrings are in form of "00 00" for register split
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 2], ["c1", 2]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=54, success=True, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)
        self.assertEqual(
            marginal_distribution(result.get_counts(), [0, 1]), expected_marginal_counts
        )
        self.assertEqual(marginal_distribution(result.get_counts(), [1, 0]), expected_reverse)

    def test_marginal_counts_result(self):
        """Test that a Result object containing counts marginalizes correctly."""
        raw_counts_1 = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_1 = models.ExperimentResult(
            shots=54, success=True, data=data_1, header=exp_result_header_1
        )

        raw_counts_2 = {"0x2": 5, "0x3": 8}
        data_2 = models.ExperimentResultData(counts=raw_counts_2)
        exp_result_header_2 = {"creg_sizes": [["c0", 2]], "memory_slots": 2}
        exp_result_2 = models.ExperimentResult(
            shots=13, success=True, data=data_2, header=exp_result_header_2
        )

        result = Result(results=[exp_result_1, exp_result_2], **self.base_result_args)

        expected_marginal_counts_1 = {"00": 4, "01": 27, "10": 23}
        expected_marginal_counts_2 = {"0": 5, "1": 8}
        expected_marginal_counts_none = {
            "0000": 4,
            "0001": 7,
            "0010": 10,
            "0110": 5,
            "1001": 11,
            "1101": 9,
            "1110": 8,
        }

        self.assertEqual(marginal_counts(result, [0, 1]).get_counts(0), expected_marginal_counts_1)
        self.assertEqual(marginal_counts(result, [0]).get_counts(1), expected_marginal_counts_2)
        self.assertEqual(marginal_counts(result, None).get_counts(0), expected_marginal_counts_none)

    def test_marginal_counts_result_memory(self):
        """Test that a Result object containing memory marginalizes correctly."""
        result = self.generate_qiskit_result()
        marginal_result = marginal_counts(result, indices=[0])
        marginal_memory = marginal_result.results[0].data.memory
        self.assertEqual(marginal_memory, [hex(ii % 2) for ii in range(8)])

    def test_marginal_counts_result_memory_nonzero_indices(self):
        """Test that a Result object containing memory marginalizes correctly."""
        result = self.generate_qiskit_result()
        index = 2
        marginal_result = marginal_counts(result, indices=[index])
        marginal_memory = marginal_result.results[0].data.memory
        mask = 1 << index
        expected = [hex((ii & mask) >> index) for ii in range(8)]
        self.assertEqual(marginal_memory, expected)

    def test_marginal_counts_result_memory_indices_None(self):
        """Test that a Result object containing memory marginalizes correctly."""
        result = self.generate_qiskit_result()
        memory = "should not be touched"
        result.results[0].data.memory = memory
        marginal_result = marginal_counts(result, indices=None)
        marginal_memory = marginal_result.results[0].data.memory
        self.assertEqual(marginal_memory, memory)

    def test_marginal_counts_result_invalid_indices(self):
        """Test that a Result object containing memory marginalizes correctly inplace."""

        result = self.generate_qiskit_result()
        with self.assertRaises(QiskitError):
            _ = marginal_counts(result, indices=[0, 1, 100], inplace=True)

    def test_marginal_counts_result_marginalize_memory(self):
        """Test that a Result object containing memory marginalizes correctly inplace."""

        result = self.generate_qiskit_result()
        marginal_result = marginal_counts(
            result, indices=[0], inplace=True, marginalize_memory=False
        )
        self.assertFalse(hasattr(marginal_result.results[0].data, "memory"))
        result = self.generate_qiskit_result()
        marginal_result = marginal_counts(
            result, indices=[0], inplace=True, marginalize_memory=None
        )
        self.assertTrue(hasattr(marginal_result.results[0].data, "memory"))
        result = self.generate_qiskit_result()
        marginal_result = marginal_counts(
            result, indices=[0], inplace=True, marginalize_memory=True
        )
        self.assertTrue(hasattr(marginal_result.results[0].data, "memory"))

        result = self.generate_qiskit_result()
        marginal_result = marginal_counts(
            result, indices=[0], inplace=False, marginalize_memory=False
        )
        self.assertFalse(hasattr(marginal_result.results[0].data, "memory"))
        marginal_result = marginal_counts(
            result, indices=[0], inplace=False, marginalize_memory=None
        )
        self.assertTrue(hasattr(marginal_result.results[0].data, "memory"))
        marginal_result = marginal_counts(
            result, indices=[0], inplace=False, marginalize_memory=True
        )
        self.assertTrue(hasattr(marginal_result.results[0].data, "memory"))

    def test_marginal_counts_result_inplace(self):
        """Test that a Result object containing memory marginalizes correctly inplace."""
        result = self.generate_qiskit_result()

        marginal_result = marginal_counts(result, indices=[0], inplace=True)
        self.assertEqual(id(result), id(marginal_result))
        marginal_memory = marginal_result.results[0].data.memory
        self.assertEqual(marginal_memory, [hex(ii % 2) for ii in range(8)])

    def test_marginal_counts_result_creg_sizes(self):
        """Test that marginal_counts with Result input properly changes creg_sizes."""
        raw_counts = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {"creg_sizes": [["c0", 1], ["c1", 3]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=54, success=True, data=data, header=exp_result_header
        )

        result = Result(results=[exp_result], **self.base_result_args)

        expected_marginal_counts = {"0 0": 14, "0 1": 18, "1 0": 13, "1 1": 9}
        expected_creg_sizes = [["c0", 1], ["c1", 1]]
        expected_memory_slots = 2
        marginal_counts_result = marginal_counts(result, [0, 2])
        self.assertEqual(
            marginal_counts_result.results[0].header["creg_sizes"], expected_creg_sizes
        )
        self.assertEqual(
            marginal_counts_result.results[0].header["memory_slots"], expected_memory_slots
        )
        self.assertEqual(marginal_counts_result.get_counts(0), expected_marginal_counts)

    def test_marginal_counts_result_format(self):
        """Test that marginal_counts with format_marginal true properly formats output."""
        raw_counts_1 = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0x12": 8}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"creg_sizes": [["c0", 2], ["c1", 3]], "memory_slots": 5}
        exp_result_1 = models.ExperimentResult(
            shots=54, success=True, data=data_1, header=exp_result_header_1
        )

        result = Result(results=[exp_result_1], **self.base_result_args)

        expected_marginal_counts_1 = {
            "0_0 _0": 14,
            "0_0 _1": 18,
            "0_1 _0": 5,
            "0_1 _1": 9,
            "1_0 _0": 8,
        }
        marginal_counts_result = marginal_counts(
            result.get_counts(), [0, 2, 4], format_marginal=True
        )
        self.assertEqual(marginal_counts_result, expected_marginal_counts_1)

    def test_marginal_counts_inplace_true(self):
        """Test marginal_counts(Result, inplace = True)"""
        raw_counts_1 = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_1 = models.ExperimentResult(
            shots=54, success=True, data=data_1, header=exp_result_header_1
        )

        raw_counts_2 = {"0x2": 5, "0x3": 8}
        data_2 = models.ExperimentResultData(counts=raw_counts_2)
        exp_result_header_2 = {"creg_sizes": [["c0", 2]], "memory_slots": 2}
        exp_result_2 = models.ExperimentResult(
            shots=13, success=True, data=data_2, header=exp_result_header_2
        )

        result = Result(results=[exp_result_1, exp_result_2], **self.base_result_args)

        expected_marginal_counts = {"0": 27, "1": 27}

        self.assertEqual(
            marginal_counts(result, [0], inplace=True).get_counts(0), expected_marginal_counts
        )
        self.assertEqual(result.get_counts(0), expected_marginal_counts)

    def test_marginal_counts_inplace_false(self):
        """Test marginal_counts(Result, inplace=False)"""
        raw_counts_1 = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0xE": 8}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"creg_sizes": [["c0", 4]], "memory_slots": 4}
        exp_result_1 = models.ExperimentResult(
            shots=54, success=True, data=data_1, header=exp_result_header_1
        )

        raw_counts_2 = {"0x2": 5, "0x3": 8}
        data_2 = models.ExperimentResultData(counts=raw_counts_2)
        exp_result_header_2 = {"creg_sizes": [["c0", 2]], "memory_slots": 2}
        exp_result_2 = models.ExperimentResult(
            shots=13, success=True, data=data_2, header=exp_result_header_2
        )

        result = Result(results=[exp_result_1, exp_result_2], **self.base_result_args)

        expected_marginal_counts = {"0": 27, "1": 27}

        self.assertEqual(
            marginal_counts(result, [0], inplace=False).get_counts(0), expected_marginal_counts
        )
        self.assertNotEqual(result.get_counts(0), expected_marginal_counts)

    def test_marginal_counts_with_dict(self):
        """Test the marginal_counts method with dictionary instead of Result object."""
        dict_counts_1 = {
            "0000": 4,
            "0001": 7,
            "0010": 10,
            "0110": 5,
            "1001": 11,
            "1101": 9,
            "1110": 8,
        }
        dict_counts_2 = {"10": 5, "11": 8}

        expected_marginal_counts_1 = {"00": 4, "01": 27, "10": 23}
        expected_marginal_counts_2 = {"0": 5, "1": 8}

        self.assertEqual(marginal_counts(dict_counts_1, [0, 1]), expected_marginal_counts_1)
        self.assertEqual(
            marginal_counts(dict_counts_2, [0], inplace=True), expected_marginal_counts_2
        )
        self.assertNotEqual(dict_counts_2, expected_marginal_counts_2)
        self.assertRaises(
            AttributeError, lambda: marginal_counts(dict_counts_1, [0, 1]).get_counts(0)
        )

    def test_memory_counts_no_header(self):
        """Test that memory bitstrings are extracted properly without header."""
        raw_memory = ["0x0", "0x0", "0x2", "0x2", "0x2", "0x2", "0x2"]
        no_header_processed_memory = [bin(int(bs[2:], 16))[2:] for bs in raw_memory]
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, memory=True, data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_memory(0), no_header_processed_memory)

    def test_memory_counts_header(self):
        """Test that memory bitstrings are extracted properly with header."""
        raw_memory = ["0x0", "0x0", "0x2", "0x2", "0x2", "0x2", "0x2"]
        no_header_processed_memory = [
            "0 0 00",
            "0 0 00",
            "0 0 10",
            "0 0 10",
            "0 0 10",
            "0 0 10",
            "0 0 10",
        ]
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result_header = {"creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]], "memory_slots": 4}
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, memory=True, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)

        self.assertEqual(result.get_memory(0), no_header_processed_memory)

    def test_meas_level_1_avg(self):
        """Test measurement level 1 average result."""
        # 3 qubits
        raw_memory = [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]
        processed_memory = np.array([1.0j, 1.0, 0.5 + 0.5j], dtype=np.complex128)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=2, success=True, meas_level=1, meas_return="avg", data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (3,))
        self.assertEqual(memory.dtype, np.complex128)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_1_single(self):
        """Test measurement level 1 single result."""
        # 3 qubits
        raw_memory = [[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]]]
        processed_memory = np.array(
            [[1.0j, 1.0, 0.5 + 0.5j], [0.5 + 0.5j, 1.0, 1.0j]], dtype=np.complex128
        )
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=2, success=True, meas_level=1, meas_return="single", data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 3))
        self.assertEqual(memory.dtype, np.complex128)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_0_avg(self):
        """Test measurement level 0 average result."""
        # 3 qubits
        raw_memory = [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]
        processed_memory = np.array([[1.0j, 1.0j, 1.0j], [1.0, 1.0, 1.0]], dtype=np.complex128)
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=2, success=True, meas_level=0, meas_return="avg", data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 3))
        self.assertEqual(memory.dtype, np.complex128)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_meas_level_0_single(self):
        """Test measurement level 0 single result."""
        # 3 qubits
        raw_memory = [
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        ]
        processed_memory = np.array(
            [[[1.0j, 1.0j, 1.0j], [1.0, 1.0, 1.0]], [[1.0j, 1.0j, 1.0j], [1.0, 1.0, 1.0]]],
            dtype=np.complex128,
        )
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=2, success=True, meas_level=0, meas_return="single", data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)
        memory = result.get_memory(0)

        self.assertEqual(memory.shape, (2, 2, 3))
        self.assertEqual(memory.dtype, np.complex128)
        np.testing.assert_almost_equal(memory, processed_memory)

    def test_circuit_statevector_repr_without_decimal(self):
        """Test postprocessing of statevector without giving any decimals arg."""
        raw_statevector = np.array(
            [
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
            ],
            dtype=np.complex128,
        )
        processed_sv = np.array(
            [
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
            ],
            dtype=np.complex128,
        )
        data = models.ExperimentResultData(statevector=raw_statevector)
        exp_result = models.ExperimentResult(shots=1, success=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        statevector = result.get_statevector()
        self.assertEqual(statevector.shape, (8,))
        self.assertEqual(statevector.dtype, np.complex128)
        np.testing.assert_almost_equal(statevector, processed_sv)

    def test_circuit_statevector_repr_decimal(self):
        """Test postprocessing of statevector giving decimals arg."""
        raw_statevector = np.array(
            [
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
            ],
            dtype=np.complex128,
        )
        processed_sv = np.array(
            [
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
                0.354 + 0.0j,
            ],
            dtype=np.complex128,
        )
        data = models.ExperimentResultData(statevector=raw_statevector)
        exp_result = models.ExperimentResult(shots=1, success=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        statevector = result.get_statevector(decimals=3)
        self.assertEqual(statevector.shape, (8,))
        self.assertEqual(statevector.dtype, np.complex128)
        np.testing.assert_almost_equal(statevector, processed_sv)

    def test_circuit_unitary_repr_without_decimal(self):
        """Test postprocessing of unitary without giving any decimals arg."""
        raw_unitary = np.array(
            [
                [0.70710678 + 0.00000000e00j, 0.70710678 - 8.65956056e-17j],
                [0.70710678 + 0.00000000e00j, -0.70710678 + 8.65956056e-17j],
            ],
            dtype=np.complex128,
        )
        processed_unitary = np.array(
            [
                [0.70710678 + 0.00000000e00j, 0.70710678 - 8.65956056e-17j],
                [0.70710678 + 0.00000000e00j, -0.70710678 + 8.65956056e-17j],
            ],
            dtype=np.complex128,
        )
        data = models.ExperimentResultData(unitary=raw_unitary)
        exp_result = models.ExperimentResult(shots=1, success=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        unitary = result.get_unitary()
        self.assertEqual(unitary.shape, (2, 2))
        self.assertEqual(unitary.dtype, np.complex128)
        np.testing.assert_almost_equal(unitary, processed_unitary)

    def test_circuit_unitary_repr_decimal(self):
        """Test postprocessing of unitary giving decimals arg."""
        raw_unitary = np.array(
            [
                [0.70710678 + 0.00000000e00j, 0.70710678 - 8.65956056e-17j],
                [0.70710678 + 0.00000000e00j, -0.70710678 + 8.65956056e-17j],
            ],
            dtype=np.complex128,
        )
        processed_unitary = np.array(
            [[0.707 + 0.0j, 0.707 - 0.0j], [0.707 + 0.0j, -0.707 + 0.0j]], dtype=np.complex128
        )
        data = models.ExperimentResultData(unitary=raw_unitary)
        exp_result = models.ExperimentResult(shots=1, success=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        unitary = result.get_unitary(decimals=3)
        self.assertEqual(unitary.shape, (2, 2))
        self.assertEqual(unitary.dtype, np.complex128)
        np.testing.assert_almost_equal(unitary, processed_unitary)

    def test_additional_result_data(self):
        """Test construction of ExperimentResult with additional data"""
        target_probs = {"0x0": 0.5, "0x1": 0.5}
        data = models.ExperimentResultData(probabilities=target_probs)
        exp_result = models.ExperimentResult(shots=1, success=True, data=data)
        result = Result(results=[exp_result], **self.base_result_args)
        result_probs = result.data(0)["probabilities"]
        self.assertEqual(result_probs, target_probs)


class TestResultOperationsFailed(QiskitTestCase):
    """Result operations methods."""

    def setUp(self):
        self.base_result_args = {
            "backend_name": "test_backend",
            "backend_version": "1.0.0",
            "job_id": "job-123",
            "success": True,
        }
        super().setUp()

    def test_counts_int_out(self):
        """Test that fails when get_count is called with a nonexistent int."""
        raw_counts = {"0x0": 4, "0x2": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result = models.ExperimentResult(shots=14, success=True, meas_level=2, data=data)
        result = Result(results=[exp_result], **self.base_result_args)

        with self.assertRaises(Exception) as context:
            result.get_counts(99)
        self.assertEqual(
            'Result for experiment "99" could not be found.', context.exception.message
        )

    def test_counts_name_out(self):
        """Test that fails when get_count is called with a nonexistent name."""
        raw_counts = {"0x0": 4, "0x2": 10}
        data = models.ExperimentResultData(counts=raw_counts)
        exp_result_header = {
            "creg_sizes": [["c0", 2], ["c0", 1], ["c1", 1]],
            "memory_slots": 4,
            "name": "a_name",
        }
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, data=data, header=exp_result_header
        )
        result = Result(results=[exp_result], **self.base_result_args)

        with self.assertRaises(Exception) as context:
            result.get_counts("another_name")
        self.assertEqual(
            'Data for experiment "another_name" could not be found.', context.exception.message
        )

    def test_memory_int_out(self):
        """Test that memory bitstrings are extracted properly without header."""
        raw_memory = ["0x0", "0x0", "0x2", "0x2", "0x2", "0x2", "0x2"]
        data = models.ExperimentResultData(memory=raw_memory)
        exp_result = models.ExperimentResult(
            shots=14, success=True, meas_level=2, memory=True, data=data
        )
        result = Result(results=[exp_result], **self.base_result_args)

        with self.assertRaises(Exception) as context:
            result.get_memory(99)
        self.assertEqual(
            'Result for experiment "99" could not be found.', context.exception.message
        )

    def test_marginal_counts_no_cregs(self):
        """Test that marginal_counts without cregs See qiskit-terra/6430."""
        raw_counts_1 = {"0x0": 4, "0x1": 7, "0x2": 10, "0x6": 5, "0x9": 11, "0xD": 9, "0x12": 8}
        data_1 = models.ExperimentResultData(counts=raw_counts_1)
        exp_result_header_1 = {"memory_slots": 5}
        exp_result_1 = models.ExperimentResult(
            shots=54, success=True, data=data_1, header=exp_result_header_1
        )

        result = Result(results=[exp_result_1], **self.base_result_args)

        _ = marginal_counts(result, indices=[0])
        marginal_counts_result = marginal_counts(result, indices=[0])
        self.assertEqual(marginal_counts_result.get_counts(), {"0": 27, "1": 27})
