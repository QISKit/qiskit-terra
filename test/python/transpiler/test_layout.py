# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests the layout object"""

import copy
import pickle
import unittest
import numpy

from qiskit.circuit import QuantumRegister, Qubit
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.exceptions import LayoutError
from qiskit._accelerate.nlayout import NLayout
from test import QiskitTestCase


class LayoutTest(QiskitTestCase):
    """Test the methods in the layout object."""

    def setUp(self):
        super().setUp()
        self.qr = QuantumRegister(3, "qr")

    def test_default_layout(self):
        """Static method generate_trivial_layout creates a Layout"""
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(2, "qr1")
        layout = Layout.generate_trivial_layout(qr0, qr1)

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr0[2]], 2)
        self.assertEqual(layout[qr1[0]], 3)
        self.assertEqual(layout[qr1[1]], 4)

    def test_layout_from_dict(self):
        """Constructor from a dict"""
        layout = Layout({self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2})

        self.assertEqual(layout[self.qr[0]], 0)
        self.assertEqual(layout[self.qr[1]], 1)
        self.assertEqual(layout[self.qr[2]], 2)
        self.assertEqual(layout[0], self.qr[0])
        self.assertEqual(layout[1], self.qr[1])
        self.assertEqual(layout[2], self.qr[2])

    def test_layout_from_dict_hole(self):
        """Constructor from a dict with a hole"""
        qr0 = QuantumRegister(2)
        qr1 = QuantumRegister(2)

        layout = Layout({qr0[0]: 0, qr1[0]: 1, qr1[1]: 3, qr0[1]: 4})

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr1[0]], 1)
        with self.assertRaises(KeyError):
            _ = layout[None]
        self.assertEqual(layout[qr1[1]], 3)
        self.assertEqual(layout[qr0[1]], 4)

        self.assertEqual(layout[0], qr0[0])
        self.assertEqual(layout[1], qr1[0])
        self.assertEqual(layout[3], qr1[1])
        self.assertEqual(layout[4], qr0[1])

    def test_layout_set(self):
        """Setter"""
        layout = Layout()
        layout[self.qr[0]] = 0
        self.assertEqual(layout[self.qr[0]], 0)
        self.assertEqual(layout[0], self.qr[0])

    def test_layout_del(self):
        """Deleter"""
        layout = Layout()
        layout[self.qr[0]] = 0
        del layout[self.qr[0]]
        self.assertTrue(self.qr[0] not in layout)

    def test_layout_avoid_dangling_physical(self):
        """No dangling pointers for physical qubits."""
        layout = Layout({self.qr[0]: 0})
        self.assertEqual(layout[0], self.qr[0])
        layout[self.qr[0]] = 1
        with self.assertRaises(KeyError):
            _ = layout[0]

    def test_layout_avoid_dangling_virtual(self):
        """No dangling pointers for virtual qubits."""
        layout = Layout({self.qr[0]: 0})
        self.assertEqual(layout[0], self.qr[0])
        layout[0] = self.qr[1]
        with self.assertRaises(KeyError):
            _ = layout[self.qr[0]]

    def test_layout_len(self):
        """Length of the layout is the amount of physical bits"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add(self.qr[2])
        self.assertEqual(len(layout), 1)
        layout.add(self.qr[1], 3)
        self.assertEqual(len(layout), 2)

    def test_layout_len_with_idle(self):
        """Length of the layout is the amount of physical bits"""
        layout = Layout()
        self.assertEqual(len(layout), 0)
        layout.add(self.qr[2])
        self.assertEqual(len(layout), 1)
        layout.add(self.qr[1], 3)
        self.assertEqual(len(layout), 2)

    def test_layout_get_bits(self):
        """Get the map from the (qu)bits view"""
        layout_dict = {self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2}
        layout = Layout(layout_dict)
        self.assertDictEqual(layout_dict, layout.get_virtual_bits())

    def test_layout_get_physical_bits(self):
        """Get the map from the physical bits view"""
        layout = Layout({self.qr[0]: 0, self.qr[1]: 1, self.qr[2]: 2})
        self.assertDictEqual(
            layout.get_physical_bits(), {0: self.qr[0], 1: self.qr[1], 2: self.qr[2]}
        )

    def test_layout_add(self):
        """add() method"""
        layout = Layout()
        layout[self.qr[0]] = 0
        layout.add(self.qr[1])

        self.assertEqual(layout[self.qr[1]], 1)

    def test_layout_add_register(self):
        """add_register() method"""
        layout = Layout()
        qr0 = QuantumRegister(2, "q0")
        qr1 = QuantumRegister(1, "qr1")
        layout.add_register(qr0)
        layout.add_register(qr1)

        self.assertEqual(layout[qr0[0]], 0)
        self.assertEqual(layout[qr0[1]], 1)
        self.assertEqual(layout[qr1[0]], 2)
        self.assertIn(qr0, layout.get_registers())
        self.assertIn(qr1, layout.get_registers())

    def test_physical_keyerror(self):
        """When asking for an unexistant physical qubit, KeyError"""
        layout = Layout()
        layout[self.qr[0]] = 1

        with self.assertRaises(KeyError):
            _ = layout[0]

    def test_virtual_keyerror(self):
        """When asking for an unexistant virtual qubit, KeyError"""
        layout = Layout()
        layout[self.qr[0]] = 1

        with self.assertRaises(KeyError):
            _ = layout[self.qr[1]]

    def test_layout_swap(self):
        """swap() method"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        layout.swap(0, 1)
        self.assertDictEqual(layout.get_virtual_bits(), {self.qr[0]: 1, self.qr[1]: 0})

    def test_layout_swap_error(self):
        """swap() method error"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        with self.assertRaises(LayoutError):
            layout.swap(0, self.qr[0])

    def test_layout_combine(self):
        """combine_into_edge_map() method"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {self.qr[0]: self.qr[1], self.qr[1]: self.qr[0]})

    def test_layout_combine_bigger(self):
        """combine_into_edge_map() method with another_layout is bigger"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])
        another_layout.add(self.qr[2])

        edge_map = layout.combine_into_edge_map(another_layout)
        self.assertDictEqual(edge_map, {self.qr[0]: self.qr[1], self.qr[1]: self.qr[0]})

    def test_set_virtual_without_physical(self):
        """When adding a virtual without care in which physical is going"""
        layout = Layout()
        layout.add(self.qr[1], 2)
        layout.add(self.qr[0])

        self.assertDictEqual(layout.get_virtual_bits(), {self.qr[0]: 0, self.qr[1]: 2})

    def test_layout_combine_smaller(self):
        """combine_into_edge_map() method with another_layout is smaller and raises an Error"""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])
        layout.add(self.qr[2])
        another_layout = Layout()
        another_layout.add(self.qr[1])
        another_layout.add(self.qr[0])

        with self.assertRaises(LayoutError):
            _ = layout.combine_into_edge_map(another_layout)

    def test_copy(self):
        """Test copy methods return equivalent layouts."""
        layout = Layout()
        layout.add(self.qr[0])
        layout.add(self.qr[1])

        layout_dict_copy = layout.copy()
        self.assertTrue(isinstance(layout_dict_copy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

        layout_copy_copy = copy.copy(layout)
        self.assertTrue(isinstance(layout_copy_copy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

        layout_copy_deepcopy = copy.deepcopy(layout)
        self.assertTrue(isinstance(layout_copy_deepcopy, Layout))
        self.assertDictEqual(layout.get_physical_bits(), layout_dict_copy.get_physical_bits())
        self.assertDictEqual(layout.get_virtual_bits(), layout_dict_copy.get_virtual_bits())

    def test_layout_error_str_key(self):
        """Layout does not work with strings"""
        layout = Layout()

        with self.assertRaises(LayoutError):
            layout["a_string"] = 3

        with self.assertRaises(LayoutError):
            layout[2] = "a_string"

    def test_layout_error_when_same_type(self):
        """Layout does not work when key and value are the same type"""
        layout = Layout()

        with self.assertRaises(LayoutError):
            layout[self.qr[0]] = [self.qr[1]]

        with self.assertRaises(LayoutError):
            layout[0] = 1

    def test_layout_from_intlist(self):
        """Create a layout from a list of integers.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         q3_0  ->  8
         q3_1  ->  9
         q3_2  ->  10
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = [4, 5, 6, 8, 9, 10]
        layout = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

        expected = Layout({4: qr1[0], 5: qr2[0], 6: qr2[1], 8: qr3[0], 9: qr3[1], 10: qr3[2]})
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_from_intlist_numpy(self):
        """Create a layout from a list of numpy integers. See #3097"""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = numpy.array([0, 1, 2, 3, 4, 5])
        layout = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

        expected = Layout.generate_trivial_layout(qr1, qr2, qr3)
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_from_intlist_short(self):
        """Raise if the intlist is longer that your quantum register.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         None  ->  8
         None  ->  9
         None  ->  10
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")

        intlist_layout = [4, 5, 6, 8, 9, 10]
        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2)

    def test_layout_from_intlist_long(self):
        """If the intlist is shorter that your quantum register, fail.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  5
         q2_1  ->  6
         q3_0  ->  8
         q3_1  ->  ?
         q3_2  ->  ?
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        intlist_layout = [4, 5, 6, 8]

        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2, qr3)

    def test_layout_from_intlist_duplicated(self):
        """If the intlist contains duplicated ints, fail.
        virtual  physical
         q1_0  ->  4
         q2_0  ->  6 -- This is
         q2_1  ->  6 -- not allowed
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        intlist_layout = [4, 6, 6]

        with self.assertRaises(LayoutError):
            _ = Layout.from_intlist(intlist_layout, qr1, qr2)

    def test_layout_from_tuplelist(self):
        """Create a Layout from list of tuples
        virtual  physical
         q1_0  ->  3
         q2_0  ->  5
         q2_1  ->  7
        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        tuplelist_layout = [None, None, None, qr1[0], None, qr2[0], None, qr2[1]]

        layout = Layout.from_qubit_list(tuplelist_layout)

        expected = Layout(
            {
                3: qr1[0],
                5: qr2[0],
                7: qr2[1],
            }
        )
        self.assertDictEqual(layout._p2v, expected._p2v)
        self.assertDictEqual(layout._v2p, expected._v2p)

    def test_layout_contains(self):
        """Verify Layouts support __contains__."""
        qr = QuantumRegister(2, "qr")
        layout = Layout()
        layout.add(qr[0], 0)

        self.assertIn(qr[0], layout)
        self.assertIn(0, layout)
        self.assertNotIn(qr[1], layout)
        self.assertNotIn(1, layout)

    def test_compose(self):
        """Test the compose method."""
        qubits = [Qubit() for _ in range(4)]
        first = [0, 3, 1, 2]
        second = [2, 3, 1, 0]
        first_layout = Layout(dict(zip(qubits, first)))
        second_layout = Layout(dict(zip(qubits, second)))
        expected = Layout({qubits[0]: 1, qubits[1]: 2, qubits[2]: 3, qubits[3]: 0})
        self.assertEqual(second_layout.compose(first_layout, qubits), expected)

    def test_compose_different_qubit_lists(self):
        """Test the compose method when the lists of source qubits for the
        two layouts are different.
        """
        qr1 = QuantumRegister(4, "qr1")
        qr2 = QuantumRegister(4, "qr2")

        qubits1 = [Qubit(qr1, 0), Qubit(qr1, 1), Qubit(qr1, 2), Qubit(qr1, 3)]
        qubits2 = [Qubit(qr2, 0), Qubit(qr2, 1), Qubit(qr2, 2), Qubit(qr2, 3)]
        positions1 = [1, 2, 3, 0]
        positions2 = [0, 2, 1, 3]
        layout1 = Layout(dict(zip(qubits1, positions1)))
        layout2 = Layout(dict(zip(qubits2, positions2)))
        # qr1[0] -> 1 <--> qr2[1] -> 2
        # qr1[1] -> 2 <--> qr2[2] -> 1
        # qr1[2] -> 3 <--> qr2[3] -> 3
        # qr1[3] -> 0 <--> qr2[0] -> 0
        expected = Layout({qubits1[0]: 2, qubits1[1]: 1, qubits1[2]: 3, qubits1[3]: 0})
        composed = layout1.compose(layout2, qubits2)
        self.assertEqual(composed, expected)

    def test_compose_no_permutation_original(self):
        """Test compose where the first doesn't permute anything."""
        qubits = [Qubit() for _ in range(4)]
        first = [0, 1, 2, 3]
        second = [2, 3, 1, 0]
        first_layout = Layout(dict(zip(qubits, first)))
        second_layout = Layout(dict(zip(qubits, second)))
        self.assertEqual(second_layout.compose(first_layout, qubits), second_layout)

    def test_compose_no_permutation_second(self):
        """Test compose where the second doesn't permute anything."""
        qubits = [Qubit() for _ in range(4)]
        second = [0, 1, 2, 3]
        first = [2, 3, 1, 0]
        first_layout = Layout(dict(zip(qubits, first)))
        second_layout = Layout(dict(zip(qubits, second)))
        self.assertEqual(second_layout.compose(first_layout, qubits), first_layout)

    def test_inverse(self):
        """Test inverse method."""
        qubits = [Qubit() for _ in range(4)]
        layout = Layout({qubits[0]: 1, qubits[1]: 2, qubits[2]: 3, qubits[3]: 0})
        inverse = layout.inverse(qubits, qubits)
        expected = Layout({qubits[1]: 0, qubits[2]: 1, qubits[3]: 2, qubits[0]: 3})
        self.assertEqual(inverse, expected)

    def test_inverse_different_qubit_lists(self):
        """Test inverse method when the source and target qubits are different."""
        qr1 = QuantumRegister(4, "qr1")
        qr2 = QuantumRegister(4, "qr2")

        qubits1 = [Qubit(qr1, 0), Qubit(qr1, 1), Qubit(qr1, 2), Qubit(qr1, 3)]
        qubits2 = [Qubit(qr2, 0), Qubit(qr2, 1), Qubit(qr2, 2), Qubit(qr2, 3)]

        layout = Layout({qubits1[0]: 1, qubits1[1]: 2, qubits1[2]: 3, qubits1[3]: 0})
        inverse = layout.inverse(qubits1, qubits2)

        expected = Layout({qubits2[1]: 0, qubits2[2]: 1, qubits2[3]: 2, qubits2[0]: 3})
        self.assertEqual(inverse, expected)

        not_expected = Layout({qubits1[1]: 0, qubits1[2]: 1, qubits1[3]: 2, qubits1[0]: 3})
        self.assertNotEqual(inverse, not_expected)

    def test_to_permutation(self):
        """Test to_permutation method."""
        qubits = [Qubit() for _ in range(3)]
        positions = [2, 0, 1]
        layout = Layout(dict(zip(qubits, positions)))
        permutation = layout.to_permutation(qubits)
        self.assertEqual(permutation, [1, 2, 0])


class TestNLayout(QiskitTestCase):
    """This is a private class, so mostly doesn't need direct tests."""

    def test_pickle(self):
        """Test that the layout roundtrips through pickle."""
        v2p = [3, 5, 1, 2, 0, 4]
        size = len(v2p)
        layout = NLayout.from_virtual_to_physical(v2p)
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        roundtripped = pickle.loads(pickle.dumps(layout))
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], v2p)

        # No changes to `layout`.
        roundtripped.swap_virtual(0, 1)
        expected = [5, 3, 1, 2, 0, 4]
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], expected)

    def test_copy(self):
        """Test that the layout roundtrips through copy."""
        v2p = [3, 5, 1, 2, 0, 4]
        size = len(v2p)
        layout = NLayout.from_virtual_to_physical(v2p)
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        roundtripped = copy.copy(layout)
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], v2p)

        # No changes to `layout`.
        roundtripped.swap_virtual(0, 1)
        expected = [5, 3, 1, 2, 0, 4]
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], expected)

    def test_deepcopy(self):
        """Test that the layout roundtrips through deepcopy."""
        v2p = [3, 5, 1, 2, 0, 4]
        size = len(v2p)
        layout = NLayout.from_virtual_to_physical(v2p)
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        roundtripped = copy.deepcopy(layout)
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], v2p)

        # No changes to `layout`.
        roundtripped.swap_virtual(0, 1)
        expected = [5, 3, 1, 2, 0, 4]
        self.assertEqual([layout.virtual_to_physical(x) for x in range(size)], v2p)
        self.assertEqual([roundtripped.virtual_to_physical(x) for x in range(size)], expected)


if __name__ == "__main__":
    unittest.main()
