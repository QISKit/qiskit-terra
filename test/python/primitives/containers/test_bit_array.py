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

"""Unit tests for BitArray."""

from itertools import product
from test import QiskitTestCase

import ddt
import numpy as np

from qiskit.primitives.containers import BitArray
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.result import Counts


def u_8(arr):
    """Convenience function to instantiate a ``numpy.uint8`` array with few characters."""
    return np.array(arr, dtype=np.uint8)


@ddt.ddt
class BitArrayTestCase(QiskitTestCase):
    """Test the DataBin class."""

    @ddt.idata(product([(), (3, 4, 5)], [1, 6], [8, 13, 26]))
    @ddt.unpack
    def test_container(self, shape, num_shots, num_bits):
        """Test the constructor and basic attributes."""
        num_bytes = num_bits // 8 + (num_bits % 8 > 0)
        size = np.prod(shape).astype(int) * num_shots * num_bytes
        arr = np.arange(size, dtype=np.uint8).reshape(shape + (num_shots, num_bytes))

        bit_array = BitArray(arr, num_bits)
        self.assertEqual(bit_array.shape, shape)
        self.assertEqual(bit_array.size, np.prod(shape).astype(int))
        self.assertEqual(bit_array.ndim, len(shape))
        self.assertEqual(bit_array.num_shots, num_shots)
        self.assertEqual(bit_array.num_bits, num_bits)
        self.assertTrue(np.all(bit_array.array == arr))
        self.assertEqual(bit_array.array.shape[-1], num_bytes)

    def test_constructor_exceptions(self):
        """Test the constructor raises exceptions properly."""
        with self.assertRaisesRegex(TypeError, "must be a numpy.ndarray"):
            BitArray([], 1)

        with self.assertRaisesRegex(TypeError, "must have dtype uint8"):
            BitArray(np.zeros(1, dtype=float), 1)

        with self.assertRaisesRegex(ValueError, "at least two axes"):
            BitArray(np.zeros((1,), dtype=np.uint8), 1)

        with self.assertRaisesRegex(ValueError, "3 bytes per shot"):
            BitArray(np.empty((2, 3, 4, 5), dtype=np.uint8), 23)

    def test_get_counts(self):
        """Test conversion to counts."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_counts(), {bs1: 2, bs2: 1})

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_counts(0), {bs1: 2, bs2: 1})
        bs3 = "0000000" + "00000001"  # 0, 1
        bs4 = "0000001" + "00000000"  # 1, 0
        self.assertEqual(bit_array.get_counts(1), {bs3: 1, bs4: 2})

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_counts(), {bs1: 2, bs2: 1, bs3: 1, bs4: 2})

    def test_get_int_counts(self):
        """Test conversion to int counts."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        val1 = (3 << 8) + 5
        val2 = ((234 & 127) << 8) + 100
        self.assertEqual(bit_array.get_int_counts(), {val1: 2, val2: 1})

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        val1 = (3 << 8) + 5
        val2 = ((234 & 127) << 8) + 100
        self.assertEqual(bit_array.get_int_counts(0), {val1: 2, val2: 1})
        val3 = 1
        val4 = 1 << 8
        self.assertEqual(bit_array.get_int_counts(1), {val3: 1, val4: 2})

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_int_counts(), {val1: 2, val2: 1, val3: 1, val4: 2})

    def test_get_bitstrings(self):
        """Test conversion to bitstrings."""
        # note that [234, 100] requires 16 bits, not 15; we are testing that get_counts ignores the
        # junk columns
        bit_array = BitArray(u_8([[3, 5], [3, 5], [234, 100]]), num_bits=15)
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_bitstrings(), [bs1, bs1, bs2])

        bit_array = BitArray(
            u_8([[[3, 5], [3, 5], [234, 100]], [[0, 1], [1, 0], [1, 0]]]),
            num_bits=15,
        )
        bs1 = "0000011" + "00000101"  # 3, 5
        bs2 = "1101010" + "01100100"  # 234, 100
        self.assertEqual(bit_array.get_bitstrings(0), [bs1, bs1, bs2])
        bs3 = "0000000" + "00000001"  # 0, 1
        bs4 = "0000001" + "00000000"  # 1, 0
        self.assertEqual(bit_array.get_bitstrings(1), [bs3, bs4, bs4])

        # test that providing no location takes the union over all shots
        self.assertEqual(bit_array.get_bitstrings(), [bs1, bs1, bs2, bs3, bs4, bs4])

    def test_equality(self):
        """Test the equality operator"""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba3 = BitArray.from_bool_array([[1, 1, 0], [1, 1, 0]])
        ba4 = BitArray.from_bool_array([[[1, 0, 0], [1, 1, 0]]])
        self.assertEqual(ba1, ba1)
        self.assertEqual(ba1, ba2)
        self.assertNotEqual(ba1, ba3)
        self.assertNotEqual(ba1, ba4)

        ba5 = BitArray(u_8([[4, 200], [255, 10]]), num_bits=13)
        ba6 = BitArray(u_8([[4, 200], [255, 10]]), num_bits=12)
        ba7 = BitArray(u_8([[4, 200], [31, 10]]), num_bits=13)
        self.assertNotEqual(ba5, ba6)
        self.assertEqual(ba5, ba7)  # test masking

    def test_logical_and(self):
        """Test the logical AND operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 1], [0, 1, 1]])
        self.assertEqual(ba1 & ba2, BitArray.from_bool_array([[1, 0, 0], [0, 1, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 & ba2, BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]]))

    def test_logical_or(self):
        """Test the logical OR operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 1], [0, 1, 0]])
        self.assertEqual(ba1 | ba2, BitArray.from_bool_array([[1, 0, 1], [1, 1, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 | ba2, BitArray.from_bool_array([[1, 1, 0], [1, 1, 0]]))

    def test_logical_not(self):
        """Test the logical OR operator."""
        ba = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        self.assertEqual(~ba, BitArray.from_bool_array([[0, 1, 1], [0, 0, 1]]))

    @ddt.data(1, 7, 9, 13)
    def test_bitcount_ignores_padding_bits(self, num_bits):
        """Test bitcount only counts logical bits."""
        ba = BitArray.from_samples([0], num_bits)
        self.assertEqual(ba.bitcount().tolist(), [0])

        ba = ~BitArray.from_samples([0], num_bits)
        np.testing.assert_array_equal(ba.bitcount(), ba.to_bool_array().sum(axis=-1))
        self.assertEqual(ba.bitcount().tolist(), [num_bits])

    def test_logical_xor(self):
        """Test the logical XOR operator."""
        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 0, 1], [0, 1, 0]])
        self.assertEqual(ba1 ^ ba2, BitArray.from_bool_array([[0, 0, 1], [1, 0, 0]]))

        ba1 = BitArray.from_bool_array([[1, 0, 0], [1, 1, 0]])
        ba2 = BitArray.from_bool_array([[1, 1, 0]])
        self.assertEqual(ba1 ^ ba2, BitArray.from_bool_array([[0, 1, 0], [0, 0, 0]]))

    def test_from_bool_array(self):
        """Test the from_bool_array static_constructor."""

        bit_array = BitArray.from_bool_array(
            [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]]
        )
        self.assertEqual(bit_array, BitArray(u_8([[[10], [3]], [[8], [1]]]), 4))

        bit_array = BitArray.from_bool_array(
            [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]], order="little"
        )
        self.assertEqual(bit_array, BitArray(u_8([[[5], [12]], [[1], [8]]]), 4))

        bit_array = BitArray.from_bool_array(
            [[0, 0, 1, 1, 1] + [0, 0, 0, 0, 0, 0, 1, 1] + [0, 0, 0, 0, 0, 0, 0, 1]]
        )
        self.assertEqual(bit_array, BitArray(u_8([[7, 3, 1]]), 21))

        bit_array = BitArray.from_bool_array(
            [[1, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0]], order="little"
        )
        self.assertEqual(bit_array, BitArray(u_8([[7, 3, 1]]), 21))

        with self.assertRaisesRegex(ValueError, "unknown value for order"):
            BitArray.from_bool_array(
                [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]], order="litle"
            )

        with self.assertRaisesRegex(ValueError, "unknown value for order"):
            BitArray.from_bool_array(
                [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]], order="bg"
            )

    def test_to_bool_array(self):
        """Test the to_bool_array method."""

        bit_array = BitArray(u_8([[[10], [3]], [[8], [1]]]), 4)
        expected_array = np.array(
            [[[1, 0, 1, 0], [0, 0, 1, 1]], [[1, 0, 0, 0], [0, 0, 0, 1]]], dtype=np.bool_
        )
        self.assertTrue(np.array_equal(bit_array.to_bool_array(), expected_array))

        bit_array = BitArray(u_8([[[10], [3]], [[8], [1]]]), 4)
        expected_array = np.array(
            [[[0, 1, 0, 1], [1, 1, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]]], dtype=np.bool_
        )
        self.assertTrue(np.array_equal(bit_array.to_bool_array(order="little"), expected_array))

        bit_array = BitArray(u_8([[7, 3, 1]]), 21)
        expected_array = np.array(
            [[0, 0, 1, 1, 1] + [0, 0, 0, 0, 0, 0, 1, 1] + [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.bool_
        )
        self.assertTrue(np.array_equal(bit_array.to_bool_array(), expected_array))

        bit_array = BitArray(u_8([[7, 3, 1]]), 21)
        expected_array = np.array(
            [[1, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 0, 0]], dtype=np.bool_
        )
        self.assertTrue(np.array_equal(bit_array.to_bool_array(order="little"), expected_array))

        with self.assertRaisesRegex(ValueError, "Invalid value for order"):
            bit_array.to_bool_array(order="invalid")

    @ddt.data("counts", "int", "hex", "bit")
    def test_from_counts(self, counts_type):
        """Test the from_counts static constructor."""

        def convert(counts: Counts):
            if counts_type == "int":
                return counts.int_outcomes()
            if counts_type == "hex":
                return counts.hex_outcomes()
            if counts_type == "bit":
                return {bin(val): count for val, count in counts.int_outcomes().items()}
            return counts

        counts1 = convert(Counts({"0b101010": 2, "0b1": 3, "0x010203": 4}))
        counts2 = convert(Counts({1: 3, 2: 6}))
        counts3 = convert(Counts({0: 2}))

        bit_array = BitArray.from_counts(counts1)
        expected = BitArray(u_8([[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4), 17)
        self.assertEqual(bit_array, expected)

        bit_array = BitArray.from_counts(iter([counts1]))
        expected = BitArray(u_8([[[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4]), 17)
        self.assertEqual(bit_array, expected)

        bit_array = BitArray.from_counts(iter([counts1, counts2]))
        expected = [
            [[0, 0, 42]] * 2 + [[0, 0, 1]] * 3 + [[1, 2, 3]] * 4,
            [[0, 0, 1]] * 3 + [[0, 0, 2]] * 6,
        ]
        self.assertEqual(bit_array, BitArray(u_8(expected), 17))

        bit_array = BitArray.from_counts(counts3)
        expected = BitArray(u_8([[0], [0]]), 1)
        self.assertEqual(bit_array, expected)

    def test_from_samples_bitstring(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples(["110", "1", "1111111111"])
        self.assertEqual(bit_array, BitArray(u_8([[0, 6], [0, 1], [3, 255]]), 10))

        bit_array = BitArray.from_samples(["110", "1", "1111111111"], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 6], [0, 0, 1], [0, 3, 255]]), 20))

        bit_array = BitArray.from_samples(["000", "0"])
        self.assertEqual(bit_array, BitArray(u_8([[0], [0]]), 1))

    def test_from_samples_hex(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples(["0x01", "0x0a12", "0x0105"])
        self.assertEqual(bit_array, BitArray(u_8([[0, 1], [10, 18], [1, 5]]), 12))

        bit_array = BitArray.from_samples(["0x01", "0x0a12", "0x0105"], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 1], [0, 10, 18], [0, 1, 5]]), 20))

        bit_array = BitArray.from_samples(["0x0", "0x0"])
        self.assertEqual(bit_array, BitArray(u_8([[0], [0]]), 1))

    def test_from_samples_int(self):
        """Test the from_samples static constructor."""
        bit_array = BitArray.from_samples([1, 2578, 261])
        self.assertEqual(bit_array, BitArray(u_8([[0, 1], [10, 18], [1, 5]]), 12))

        bit_array = BitArray.from_samples([1, 2578, 261], 20)
        self.assertEqual(bit_array, BitArray(u_8([[0, 0, 1], [0, 10, 18], [0, 1, 5]]), 20))

        bit_array = BitArray.from_samples([0, 0, 0])
        self.assertEqual(bit_array, BitArray(u_8([[0], [0], [0]]), 1))

    def test_reshape(self):
        """Test the reshape method."""
        # this creates incrementing bitstrings from 0 to 360 * 32 - 1
        data = np.frombuffer(np.arange(360 * 32, dtype=np.uint64).tobytes(), dtype=np.uint8)
        data = data.reshape(-1, 32, 8)[..., 1::-1]
        ba = BitArray(data, 15)

        self.assertEqual(ba.reshape(120, 3).shape, (120, 3))
        self.assertEqual(ba.reshape(120, 3).num_shots, 32)
        self.assertEqual(ba.reshape(120, 3).num_bits, 15)
        self.assertTrue(
            np.array_equal(ba.reshape(60, 6).array[2, 3], data.reshape(60, 6, 32, 2)[2, 3])
        )

        self.assertEqual(ba.reshape(360 * 32).shape, ())
        self.assertEqual(ba.reshape(360 * 32).num_shots, 360 * 32)
        self.assertEqual(ba.reshape(360 * 32).num_bits, 15)

        self.assertEqual(ba.reshape(360 * 2, 16).shape, (720,))
        self.assertEqual(ba.reshape(360 * 2, 16).num_shots, 16)
        self.assertEqual(ba.reshape(360 * 2, 16).num_bits, 15)

    def test_transpose(self):
        """Test the transpose method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        # Since the input dtype is uint16, bit array requires at least two u8.
        # Thus, 9 is the minimum number of qubits, i.e., 8 + 1.
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("default arg"):
            ba2 = ba.transpose()
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("tuple 1"):
            ba2 = ba.transpose((2, 1, 0))
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("tuple 2"):
            ba2 = ba.transpose((0, 1, 2))
            self.assertEqual(ba2.shape, (1, 2, 3))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("tuple 3"):
            ba2 = ba.transpose((0, 2, 1))
            self.assertEqual(ba2.shape, (1, 3, 2))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, k, j)))

        with self.subTest("tuple, negative indices"):
            ba2 = ba.transpose((0, -1, -2))
            self.assertEqual(ba2.shape, (1, 3, 2))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, k, j)))

        with self.subTest("ints"):
            ba2 = ba.transpose(2, 1, 0)
            self.assertEqual(ba2.shape, (3, 2, 1))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((k, j, i)))

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "axes don't match bit array"):
                _ = ba.transpose((0, 1))
            with self.assertRaisesRegex(ValueError, "axes don't match bit array"):
                _ = ba.transpose((0, 1, 2, 3))
            with self.assertRaisesRegex(ValueError, "axis [0-9]+ is out of bounds for bit array"):
                _ = ba.transpose((0, 1, 4))
            with self.assertRaisesRegex(ValueError, "axis -[0-9]+ is out of bounds for bit array"):
                _ = ba.transpose((0, 1, -4))
            with self.assertRaisesRegex(ValueError, "repeated axis in transpose"):
                _ = ba.transpose((0, 1, 1))

    def test_concatenate(self):
        """Test the concatenate function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate = BitArray.concatenate

        with self.subTest("2 arrays, default"):
            ba2 = concatenate([ba, ba])
            self.assertEqual(ba2.shape, (2, 2, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((1, j, k)))

        with self.subTest("2 arrays, axis"):
            ba2 = concatenate([ba, ba], axis=1)
            self.assertEqual(ba2.shape, (1, 4, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((0, j + 2, k)))

        with self.subTest("3 arrays"):
            ba2 = concatenate([ba, ba, ba])
            self.assertEqual(ba2.shape, (3, 2, 3))
            for j, k in product(range(2), range(3)):
                self.assertEqual(ba2.get_counts((0, j, k)), ba2.get_counts((1, j, k)))
                self.assertEqual(ba2.get_counts((1, j, k)), ba2.get_counts((2, j, k)))

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to concatenate"):
                _ = concatenate([])
            with self.assertRaisesRegex(ValueError, "axis -1 is out of bounds"):
                _ = concatenate([ba, ba], -1)
            with self.assertRaisesRegex(ValueError, "axis 100 is out of bounds"):
                _ = concatenate([ba, ba], 100)

            ba2 = BitArray(data, 10)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same number of bits"):
                _ = concatenate([ba, ba2])

            data2 = np.frombuffer(np.arange(30, dtype=np.uint16).tobytes(), dtype=np.uint8)
            data2 = data2.reshape(1, 2, 3, 5, 2)[..., ::-1]
            ba2 = BitArray(data2, 9)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of shots"
            ):
                _ = concatenate([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of dimensions"
            ):
                _ = concatenate([ba, ba2])

    def test_concatenate_shots(self):
        """Test the concatenate_shots function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate_shots = BitArray.concatenate_shots

        with self.subTest("2 arrays"):
            ba2 = concatenate_shots([ba, ba])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 9)
            self.assertEqual(ba2.num_shots, 2 * ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                expected = {key: val * 2 for key, val in ba.get_counts((i, j, k)).items()}
                counts2 = ba2.get_counts((i, j, k))
                self.assertEqual(counts2, expected)

        with self.subTest("3 arrays"):
            ba2 = concatenate_shots([ba, ba, ba])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 9)
            self.assertEqual(ba2.num_shots, 3 * ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                expected = {key: val * 3 for key, val in ba.get_counts((i, j, k)).items()}
                counts2 = ba2.get_counts((i, j, k))
                self.assertEqual(counts2, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to stack"):
                _ = concatenate_shots([])

            ba2 = BitArray(data, 10)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same number of bits"):
                _ = concatenate_shots([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same shape"):
                _ = concatenate_shots([ba, ba2])

    def test_concatenate_bits(self):
        """Test the concatenate_bits function."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        concatenate_bits = BitArray.concatenate_bits

        with self.subTest("2 arrays"):
            ba_01 = ba.slice_bits([0, 1])
            ba2 = concatenate_bits([ba, ba_01])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 11)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                bs = ba.get_bitstrings((i, j, k))
                bs_01 = ba_01.get_bitstrings((i, j, k))
                expected = [s1 + s2 for s1, s2 in zip(bs_01, bs)]
                bs2 = ba2.get_bitstrings((i, j, k))
                self.assertEqual(bs2, expected)

        with self.subTest("3 arrays"):
            ba_01 = ba.slice_bits([0, 1])
            ba2 = concatenate_bits([ba, ba_01, ba_01])
            self.assertEqual(ba2.shape, (1, 2, 3))
            self.assertEqual(ba2.num_bits, 13)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                bs = ba.get_bitstrings((i, j, k))
                bs_01 = ba_01.get_bitstrings((i, j, k))
                expected = [s1 + s1 + s2 for s1, s2 in zip(bs_01, bs)]
                bs2 = ba2.get_bitstrings((i, j, k))
                self.assertEqual(bs2, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "Need at least one bit array to stack"):
                _ = concatenate_bits([])

            data2 = np.frombuffer(np.arange(30, dtype=np.uint16).tobytes(), dtype=np.uint8)
            data2 = data2.reshape(1, 2, 3, 5, 2)[..., ::-1]
            ba2 = BitArray(data2, 9)
            with self.assertRaisesRegex(
                ValueError, "All bit arrays must have same number of shots"
            ):
                _ = concatenate_bits([ba, ba2])

            ba2 = ba.reshape(2, 3)
            with self.assertRaisesRegex(ValueError, "All bit arrays must have same shape"):
                _ = concatenate_bits([ba, ba2])

    def test_getitem(self):
        """Test the __getitem__ method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba[:]
            self.assertEqual(ba2.shape, (1, 2, 3))
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("no slice"):
            ba2 = ba[0, 1, 2]
            self.assertEqual(ba2.shape, ())
            self.assertEqual(ba.get_counts((0, 1, 2)), ba2.get_counts())

        with self.subTest("slice"):
            ba2 = ba[0, :, 2]
            self.assertEqual(ba2.shape, (2,))
            for j in range(2):
                self.assertEqual(ba.get_counts((0, j, 2)), ba2.get_counts(j))

        with self.subTest("errors"):
            with self.assertRaisesRegex(IndexError, "index 2 is out of bounds"):
                _ = ba[0, 2, 2]
            with self.assertRaisesRegex(IndexError, "index -3 is out of bounds"):
                _ = ba[0, -3, 2]
            with self.assertRaisesRegex(
                IndexError, "BitArray cannot be sliced along the shots axis"
            ):
                _ = ba[0, 1, 2, 3]
            with self.assertRaisesRegex(
                IndexError, "BitArray cannot be sliced along the bits axis"
            ):
                _ = ba[0, 1, 2, 3, 4]

    def test_slice_bits(self):
        """Test the slice_bits method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba.slice_bits(range(ba.num_bits))
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("1 bit, int"):
            ba2 = ba.slice_bits(0)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_counts((i, j, k)), {"0": 5, "1": 5})

        with self.subTest("1 bit, list"):
            ba2 = ba.slice_bits([0])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_counts((i, j, k)), {"0": 5, "1": 5})

        with self.subTest("2 bits"):
            ba2 = ba.slice_bits([0, 1])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            self.assertEqual(ba2.num_bits, 2)
            even = {"00": 3, "01": 3, "10": 2, "11": 2}
            odd = {"10": 3, "11": 3, "00": 2, "01": 2}
            for count, (i, j, k) in enumerate(product(range(1), range(2), range(3))):
                expect = even if count % 2 == 0 else odd
                self.assertEqual(ba2.get_counts((i, j, k)), expect)

        with self.subTest("errors"):
            with self.assertRaisesRegex(IndexError, "index -1 is out of bounds"):
                _ = ba.slice_bits(-1)
            with self.assertRaisesRegex(IndexError, "index 9 is out of bounds"):
                _ = ba.slice_bits(9)

    def test_slice_shots(self):
        """Test the slice_shots method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))

        with self.subTest("all"):
            ba2 = ba.slice_shots(range(ba.num_shots))
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, ba.num_shots)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba.get_counts((i, j, k)), ba2.get_counts((i, j, k)))

        with self.subTest("1 shot, int"):
            ba2 = ba.slice_shots(0)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_bitstrings((i, j, k)), [ba.get_bitstrings((i, j, k))[0]])

        with self.subTest("1 shot, list"):
            ba2 = ba.slice_shots([0])
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, 1)
            for i, j, k in product(range(1), range(2), range(3)):
                self.assertEqual(ba2.get_bitstrings((i, j, k)), [ba.get_bitstrings((i, j, k))[0]])

        with self.subTest("multiple shots"):
            indices = [1, 2, 3, 5, 8]
            ba2 = ba.slice_shots(indices)
            self.assertEqual(ba2.shape, ba.shape)
            self.assertEqual(ba2.num_bits, ba.num_bits)
            self.assertEqual(ba2.num_shots, len(indices))
            for i, j, k in product(range(1), range(2), range(3)):
                expected = [
                    bs for ind, bs in enumerate(ba.get_bitstrings((i, j, k))) if ind in indices
                ]
                self.assertEqual(ba2.get_bitstrings((i, j, k)), expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(IndexError, "index -1 is out of bounds"):
                _ = ba.slice_shots(-1)
            with self.assertRaisesRegex(IndexError, "index 10 is out of bounds"):
                _ = ba.slice_shots(10)

    def test_expectation_values(self):
        """Test the expectation_values method."""
        # this creates incrementing bitstrings from 0 to 59
        data = np.frombuffer(np.arange(60, dtype=np.uint16).tobytes(), dtype=np.uint8)
        data = data.reshape(1, 2, 3, 10, 2)[..., ::-1]
        ba = BitArray(data, 9)
        self.assertEqual(ba.shape, (1, 2, 3))
        op = "I" * 8 + "Z"
        op2 = "I" * 8 + "0"
        op3 = "I" * 8 + "1"
        pauli = Pauli(op)
        sp_op = SparsePauliOp(op)
        sp_op2 = SparsePauliOp.from_sparse_list([("Z", [6], 1)], num_qubits=9)

        with self.subTest("str"):
            expval = ba.expectation_values(op)
            # both 0 and 1 appear 5 times
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros(ba.shape))

            expval = ba.expectation_values(op2)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.full((ba.shape), 0.5))

            expval = ba.expectation_values(op3)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.full((ba.shape), 0.5))

            ba2 = ba.slice_bits(6)
            # 6th bit are all 0
            expval = ba2.expectation_values("Z")
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.ones(ba.shape))

            ba3 = ba.slice_bits(5)
            # 5th bit distributes as follows.
            # (0, 0, 0) {'0': 10}
            # (0, 0, 1) {'0': 10}
            # (0, 0, 2) {'0': 10}
            # (0, 1, 0) {'0': 2, '1': 8}
            # (0, 1, 1) {'1': 10}
            # (0, 1, 2) {'1': 10}
            expval = ba3.expectation_values("0")
            expected = np.array([[[1, 1, 1], [0.2, 0, 0]]])
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, expected)

        with self.subTest("Pauli"):
            expval = ba.expectation_values(pauli)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros(ba.shape))

        with self.subTest("SparsePauliOp"):
            expval = ba.expectation_values(sp_op)
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.zeros(ba.shape))

            expval = ba.expectation_values(sp_op2)
            # 6th bit are all 0
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, np.ones(ba.shape))

        with self.subTest("ObservableArray"):
            obs = ["Z", "0", "1"]
            ba2 = ba.slice_bits(5)
            expval = ba2.expectation_values(obs)
            expected = np.array([[[1, 1, 0], [-0.6, 0, 1]]])
            self.assertEqual(expval.shape, ba.shape)
            np.testing.assert_allclose(expval, expected)

            ba4 = BitArray.from_counts([{0: 1}, {1: 1}]).reshape(2, 1)
            expval = ba4.expectation_values(obs)
            expected = np.array([[1, 1, 0], [-1, 0, 1]])
            self.assertEqual(expval.shape, (2, 3))
            np.testing.assert_allclose(expval, expected)

        with self.subTest("errors"):
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                _ = ba.expectation_values([op, op2])
            with self.assertRaisesRegex(ValueError, "One or more operators not same length"):
                _ = ba.expectation_values("Z")
            with self.assertRaisesRegex(ValueError, "is not diagonal"):
                _ = ba.expectation_values("X" * ba.num_bits)

    def test_postselection(self):
        """Test the postselection method."""

        flat_data = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=bool,
        )

        shaped_data = np.array(
            [
                [
                    [
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    ],
                    [
                        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ],
                ]
            ],
            dtype=bool,
        )

        for dataname, bool_array in zip(["flat", "shaped"], [flat_data, shaped_data]):

            bit_array = BitArray.from_bool_array(bool_array, order="little")
            # indices value of i <-> creg[i] <-> bool_array[..., i]

            num_bits = bool_array.shape[-1]
            bool_array = bool_array.reshape(-1, num_bits)

            test_cases = [
                ("basic", [0, 1], [0, 0]),
                ("multibyte", [0, 9], [0, 1]),
                ("repeated", [5, 5, 5], [0, 0, 0]),
                ("contradict", [5, 5, 5], [1, 0, 0]),
                ("unsorted", [5, 0, 9, 3], [1, 0, 1, 0]),
                ("negative", [-5, 1, -2, -10], [1, 0, 1, 0]),
                ("negcontradict", [4, -6], [1, 0]),
                ("trivial", [], []),
                ("bareindex", 6, 0),
            ]

            for name, indices, selection in test_cases:
                with self.subTest("_".join([dataname, name])):
                    postselected_bools = np.unpackbits(
                        bit_array.postselect(indices, selection).array[:, ::-1],
                        count=num_bits,
                        axis=-1,
                        bitorder="little",
                    ).astype(bool)
                    if isinstance(indices, int):
                        indices = (indices,)
                    if isinstance(selection, bool):
                        selection = (selection,)
                    answer = bool_array[np.all(bool_array[:, indices] == selection, axis=-1)]
                    if name in ["contradict", "negcontradict"]:
                        self.assertEqual(len(answer), 0)
                    else:
                        self.assertGreater(len(answer), 0)
                    np.testing.assert_equal(postselected_bools, answer)

            error_cases = [
                ("aboverange", [0, 6, 10], [True, True, False], IndexError),
                ("belowrange", [0, 6, -11], [True, True, False], IndexError),
                ("mismatch", [0, 1, 2], [False, False], ValueError),
            ]
            for name, indices, selection, error in error_cases:
                with self.subTest(dataname + "_" + name):
                    with self.assertRaises(error):
                        bit_array.postselect(indices, selection)
