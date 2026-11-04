# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit.circuit.classical import types
from test import QiskitTestCase


class TestTypesOrdering(QiskitTestCase):
    def test_order(self):
        self.assertIs(types.order(types.Uint(8), types.Uint(16)), types.Ordering.LESS)
        self.assertIs(types.order(types.Uint(16), types.Uint(8)), types.Ordering.GREATER)
        self.assertIs(types.order(types.Uint(8), types.Uint(8)), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Bool(), types.Bool()), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Float(), types.Float()), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Duration(), types.Duration()), types.Ordering.EQUAL)

        self.assertIs(types.order(types.Bool(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Bool(), types.Float()), types.Ordering.NONE)
        self.assertIs(types.order(types.Bool(), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Float()), types.Ordering.NONE)
        self.assertIs(types.order(types.Uint(8), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Float(), types.Duration()), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Bool()), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Uint(8)), types.Ordering.NONE)
        self.assertIs(types.order(types.Duration(), types.Float()), types.Ordering.NONE)

    def test_is_subtype(self):
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(16)))
        self.assertFalse(types.is_subtype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_subtype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Uint(8), strict=True))

        self.assertTrue(types.is_subtype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Bool(), strict=True))

        self.assertTrue(types.is_subtype(types.Float(), types.Float()))
        self.assertFalse(types.is_subtype(types.Float(), types.Float(), strict=True))

        self.assertTrue(types.is_subtype(types.Duration(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Duration(), strict=True))

        self.assertFalse(types.is_subtype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Bool(), types.Float()))
        self.assertFalse(types.is_subtype(types.Bool(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Bool()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Float()))
        self.assertFalse(types.is_subtype(types.Uint(8), types.Duration()))
        self.assertFalse(types.is_subtype(types.Float(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Float(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Float(), types.Duration()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Bool()))
        self.assertFalse(types.is_subtype(types.Duration(), types.Uint(8)))
        self.assertFalse(types.is_subtype(types.Duration(), types.Float()))

    def test_is_supertype(self):
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(16)))
        self.assertTrue(types.is_supertype(types.Uint(16), types.Uint(8)))
        self.assertTrue(types.is_supertype(types.Uint(8), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Uint(8), strict=True))

        self.assertTrue(types.is_supertype(types.Bool(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Bool(), strict=True))

        self.assertTrue(types.is_supertype(types.Float(), types.Float()))
        self.assertFalse(types.is_supertype(types.Float(), types.Float(), strict=True))

        self.assertTrue(types.is_supertype(types.Duration(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Duration(), strict=True))

        self.assertFalse(types.is_supertype(types.Bool(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Bool(), types.Float()))
        self.assertFalse(types.is_supertype(types.Bool(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Bool()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Float()))
        self.assertFalse(types.is_supertype(types.Uint(8), types.Duration()))
        self.assertFalse(types.is_supertype(types.Float(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Float(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Float(), types.Duration()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Bool()))
        self.assertFalse(types.is_supertype(types.Duration(), types.Uint(8)))
        self.assertFalse(types.is_supertype(types.Duration(), types.Float()))

    def test_greater(self):
        self.assertEqual(types.greater(types.Uint(16), types.Uint(8)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(16)), types.Uint(16))
        self.assertEqual(types.greater(types.Uint(8), types.Uint(8)), types.Uint(8))
        self.assertEqual(types.greater(types.Bool(), types.Bool()), types.Bool())
        self.assertEqual(types.greater(types.Float(), types.Float()), types.Float())
        self.assertEqual(types.greater(types.Duration(), types.Duration()), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Uint(8))
            types.greater(types.Bool(), types.Float())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Bool(), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Uint(8), types.Duration())
        with self.assertRaisesRegex(TypeError, "no ordering"):
            types.greater(types.Float(), types.Duration())


class TestTypesCastKind(QiskitTestCase):
    def test_basic_examples(self):
        """This is used extensively throughout the expression construction functions, but since it
        is public API, it should have some direct unit tests as well."""
        # Bool -> Bool
        self.assertIs(types.cast_kind(types.Bool(), types.Bool()), types.CastKind.EQUAL)

        # Float -> Float
        self.assertIs(types.cast_kind(types.Float(), types.Float()), types.CastKind.EQUAL)

        # Uint -> Bool
        self.assertIs(types.cast_kind(types.Uint(8), types.Bool()), types.CastKind.IMPLICIT)

        # Float -> Bool
        self.assertIs(types.cast_kind(types.Float(), types.Bool()), types.CastKind.DANGEROUS)

        # Bool -> Uint
        self.assertIs(types.cast_kind(types.Bool(), types.Uint(8)), types.CastKind.LOSSLESS)

        # Uint(16) -> Uint(8)
        self.assertIs(types.cast_kind(types.Uint(16), types.Uint(8)), types.CastKind.DANGEROUS)

        # Uint widening
        self.assertIs(types.cast_kind(types.Uint(8), types.Uint(16)), types.CastKind.LOSSLESS)

        # Uint -> Float
        self.assertIs(types.cast_kind(types.Uint(16), types.Float()), types.CastKind.DANGEROUS)

        # Float -> Uint(8)
        self.assertIs(types.cast_kind(types.Float(), types.Uint(8)), types.CastKind.DANGEROUS)

        # Float -> Uint(16)
        self.assertIs(types.cast_kind(types.Float(), types.Uint(16)), types.CastKind.DANGEROUS)

        # Bool -> Float
        self.assertIs(types.cast_kind(types.Bool(), types.Float()), types.CastKind.LOSSLESS)

        # Duration -> Duration
        self.assertIs(types.cast_kind(types.Duration(), types.Duration()), types.CastKind.EQUAL)

        # Duration -> Other types (not allowed)
        self.assertIs(types.cast_kind(types.Duration(), types.Bool()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Duration(), types.Uint(8)), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Duration(), types.Uint(16)), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Duration(), types.Float()), types.CastKind.NONE)

        # Other types -> Duration (not allowed)
        self.assertIs(types.cast_kind(types.Bool(), types.Duration()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Uint(8), types.Duration()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Uint(16), types.Duration()), types.CastKind.NONE)
        self.assertIs(types.cast_kind(types.Float(), types.Duration()), types.CastKind.NONE)
