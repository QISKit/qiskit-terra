# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import copy
import pickle
import itertools
import random

import ddt
import numpy as np

from qiskit import transpile
from qiskit.circuit import Measure, Parameter, library, QuantumCircuit
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList, Pauli, PauliList
from qiskit.transpiler import Target

from test import QiskitTestCase


def single_cases():
    return [
        QubitSparsePauli(""),
        QubitSparsePauli("I" * 10),
        QubitSparsePauli.from_label("IIXIZI"),
        QubitSparsePauli.from_label("ZZYYXX"),
    ]


def single_cases_list():
    return [
        QubitSparsePauliList.empty(0),
        QubitSparsePauliList.empty(10),
        QubitSparsePauliList.from_label("IIXIZI"),
        QubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        # Includes a duplicate entry.
        QubitSparsePauliList.from_list(["IXZ", "ZZI", "IXZ"]),
    ]


@ddt.ddt
class TestQubitSparsePauli(QiskitTestCase):

    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(QubitSparsePauli(data), QubitSparsePauli.from_pauli(data))
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=data.num_qubits), QubitSparsePauli.from_pauli(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauli(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(QubitSparsePauli(with_phase), QubitSparsePauli.from_pauli(with_phase))
        self.assertEqual(
            QubitSparsePauli(with_phase, num_qubits=data.num_qubits),
            QubitSparsePauli.from_pauli(with_phase),
        )

        self.assertEqual(QubitSparsePauli(Pauli("")), QubitSparsePauli.from_pauli(Pauli("")))

    def test_default_constructor_label(self):
        data = "IXIIZ"
        self.assertEqual(QubitSparsePauli(data), QubitSparsePauli.from_label(data))
        self.assertEqual(QubitSparsePauli(data, num_qubits=5), QubitSparsePauli.from_label(data))
        with self.assertRaisesRegex(ValueError, "does not match label"):
            QubitSparsePauli(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "does not match label"):
            QubitSparsePauli(data, num_qubits=6)

    def test_default_constructor_sparse_label(self):
        data = ("ZX", (0, 3))
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=5),
            QubitSparsePauli.from_sparse_label(data, num_qubits=5),
        )
        self.assertEqual(
            QubitSparsePauli(data, num_qubits=10),
            QubitSparsePauli.from_sparse_label(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            QubitSparsePauli(data)
        self.assertEqual(
            QubitSparsePauli(("", []), num_qubits=5),
            QubitSparsePauli.from_sparse_label(("", []), num_qubits=5),
        )

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            QubitSparsePauli.from_label("IXXIIZZIYYIXYZ"),
            QubitSparsePauli.from_raw_parts(
                14,
                [
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
            ),
        )

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauli.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauli.from_label("🐍")

    def test_from_sparse_label(self):
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("XY", (0, 1)), num_qubits=5),
            QubitSparsePauli.from_label("IIIYX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("XX", (1, 3)), num_qubits=5),
            QubitSparsePauli.from_label("IXIXI"),
        )
        self.assertEqual(
            QubitSparsePauli.from_sparse_label(("YYZ", (0, 2, 4)), num_qubits=5),
            QubitSparsePauli.from_label("ZIYIY"),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = QubitSparsePauli.from_sparse_label(("XYZ", (2, 0, 1)), num_qubits=3)
        self.assertEqual(from_unsorted, QubitSparsePauli.from_label("XZY"))
        np.testing.assert_equal(from_unsorted.indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = QubitSparsePauli.from_sparse_label(("ZXI", (0, 1, 2)), num_qubits=10)
        self.assertEqual(
            explicit_identity,
            QubitSparsePauli.from_sparse_label(("XZ", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

        explicit_identity = QubitSparsePauli.from_sparse_label(
            ("XYIII", (0, 1, 2, 3, 8)), num_qubits=10
        )
        self.assertEqual(
            explicit_identity,
            QubitSparsePauli.from_sparse_label(("YX", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauli.from_sparse_label(("+$", (2, 1)), num_qubits=8)
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            QubitSparsePauli.from_sparse_label(("\xff", (1,)), num_qubits=5)
        with self.assertRaises(ValueError):
            QubitSparsePauli.from_sparse_label(("\xff", (1, 2)), num_qubits=5)

        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 1"
        ):
            QubitSparsePauli.from_sparse_label(("XZ", (0,)), num_qubits=5)
        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 3"
        ):
            QubitSparsePauli.from_sparse_label(("XZ", (0, 1, 2)), num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            QubitSparsePauli.from_sparse_label(("XZY", (0, 1, 3)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            QubitSparsePauli.from_sparse_label(("XZY", (0, 1, 4)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            QubitSparsePauli.from_sparse_label(("XXI", (0, 1, 3)), num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauli.from_sparse_label(("XZ", (3, 3)), num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauli.from_sparse_label(("XYZXZ", (3, 0, 1, 2, 3)), num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("IIXZI")), QubitSparsePauli.from_label("IIXZI")
        )

        # `Pauli` accepts a phase in its label, which gets dropped
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("iIXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("-iIXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )
        self.assertEqual(
            QubitSparsePauli.from_pauli(Pauli("-IXZIX")),
            QubitSparsePauli.from_label("IXZIX"),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (QubitSparsePauli.from_pauli(pauli), QubitSparsePauli.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (QubitSparsePauli.from_pauli(pauli), QubitSparsePauli.from_label(label))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            QubitSparsePauli(5, num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(QubitSparsePauli("").num_qubits, 0)
        self.assertEqual(QubitSparsePauli("I" * 10).num_qubits, 10)

    def test_pauli_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(QubitSparsePauli.Pauli),
            {
                QubitSparsePauli.Pauli.X,
                QubitSparsePauli.Pauli.Y,
                QubitSparsePauli.Pauli.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(QubitSparsePauli.Pauli.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual({name: getattr(QubitSparsePauli.Pauli, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": QubitSparsePauli.Pauli.X,
            "Y": QubitSparsePauli.Pauli.Y,
            "Z": QubitSparsePauli.Pauli.Z,
        }
        self.assertEqual({label: QubitSparsePauli.Pauli[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual({pauli.label: pauli for pauli in QubitSparsePauli.Pauli}, labels)

    @ddt.idata(single_cases())
    def test_pickle(self, qubit_sparse_pauli):
        self.assertEqual(qubit_sparse_pauli, copy.copy(qubit_sparse_pauli))
        self.assertIsNot(qubit_sparse_pauli, copy.copy(qubit_sparse_pauli))
        self.assertEqual(qubit_sparse_pauli, copy.deepcopy(qubit_sparse_pauli))
        self.assertEqual(qubit_sparse_pauli, pickle.loads(pickle.dumps(qubit_sparse_pauli)))

    @ddt.data(
        QubitSparsePauli.from_label("IIXIZI"),
        QubitSparsePauli.from_label("X"),
        QubitSparsePauli.from_label("III"),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("QubitSparsePauli", repr(data))

    @ddt.idata(single_cases())
    def test_copy(self, qubit_sparse_pauli):
        self.assertEqual(qubit_sparse_pauli, qubit_sparse_pauli.copy())
        self.assertIsNot(qubit_sparse_pauli, qubit_sparse_pauli.copy())

    def test_equality(self):
        sparse_data = ("XYY", (3, 1, 0))
        pauli = QubitSparsePauli.from_sparse_label(sparse_data, num_qubits=5)
        self.assertEqual(pauli, pauli.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli, pauli)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli, QubitSparsePauli.from_sparse_label(sparse_data, num_qubits=pauli.num_qubits + 1)
        )

        # Difference in bit terms.
        self.assertNotEqual(
            QubitSparsePauli.from_label("IIXZI"),
            QubitSparsePauli.from_label("IIYZI"),
        )
        self.assertNotEqual(
            QubitSparsePauli.from_label("XXYYZ"),
            QubitSparsePauli.from_label("XXYYY"),
        )

        # Difference in indices.
        self.assertNotEqual(
            QubitSparsePauli.from_label("IIXZI"),
            QubitSparsePauli.from_label("IXIZI"),
        )
        self.assertNotEqual(
            QubitSparsePauli.from_label("XIYYZ"),
            QubitSparsePauli.from_label("IXYYZ"),
        )

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        pauli = QubitSparsePauli.from_label("XZY")
        self.assertEqual(len(pauli.indices), 3)
        self.assertEqual(len(pauli.paulis), 3)

        # Iteration
        self.assertEqual(tuple(pauli.indices), (0, 1, 2))
        # multiple iteration through same object
        paulis = pauli.paulis
        self.assertEqual(set(paulis), {QubitSparsePauli.Pauli[x] for x in "XZY"})

        # Implicit iteration methods.
        self.assertIn(QubitSparsePauli.Pauli.Y, pauli.paulis)
        self.assertNotIn(4, pauli.indices)

        # Index by scalar
        self.assertEqual(pauli.indices[-1], 2)
        self.assertEqual(pauli.paulis[0], QubitSparsePauli.Pauli.Y)

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(pauli.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            pauli.indices[::-1],
            np.array([2, 1, 0], dtype=np.uint32),
            strict=True,
        )
        self.assertIsInstance(pauli.paulis[0:2], np.ndarray)
        np.testing.assert_array_equal(
            pauli.paulis[0:2],
            np.array([QubitSparsePauli.Pauli.Y, QubitSparsePauli.Pauli.Z], dtype=np.uint8),
            strict=True,
        )

    def test_attributes_to_array(self):
        pauli = QubitSparsePauli.from_label("XZY")

        # Natural dtypes.
        np.testing.assert_array_equal(
            pauli.indices, np.array([0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            pauli.paulis,
            np.array([QubitSparsePauli.Pauli[x] for x in "YZX"], dtype=np.uint8),
            strict=True,
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(pauli.indices, dtype=np.uint8),
            np.array([0, 1, 2], dtype=np.uint8),
            strict=True,
        )

    def test_identity(self):
        identity_5 = QubitSparsePauli.identity(5)
        self.assertEqual(identity_5.num_qubits, 5)
        self.assertEqual(len(identity_5.paulis), 0)
        self.assertEqual(len(identity_5.indices), 0)

        identity_0 = QubitSparsePauli.identity(0)
        self.assertEqual(identity_0.num_qubits, 0)
        self.assertEqual(len(identity_0.paulis), 0)
        self.assertEqual(len(identity_0.indices), 0)

    def test_compose(self):
        p0 = QubitSparsePauli.from_label("XZY")
        p1 = QubitSparsePauli.from_label("ZIY")

        self.assertEqual(p0.compose(p1), QubitSparsePauli.from_label("YZI"))
        self.assertEqual(p1.compose(p0), QubitSparsePauli.from_label("YZI"))

        p0 = QubitSparsePauli.from_label("III")
        p1 = QubitSparsePauli.from_label("ZIY")

        self.assertEqual(p0 @ p1, QubitSparsePauli.from_label("ZIY"))
        self.assertEqual(p1 @ p0, QubitSparsePauli.from_label("ZIY"))

        p0 = QubitSparsePauli.from_label("IIIXXY")
        p1 = QubitSparsePauli.from_label("ZIYIII")

        self.assertEqual(p0 @ p1, QubitSparsePauli.from_label("ZIYXXY"))
        self.assertEqual(p1 @ p0, QubitSparsePauli.from_label("ZIYXXY"))

        p0 = QubitSparsePauli.from_label("IIIXXYZIXIZIZ")
        p1 = QubitSparsePauli.from_label("ZIYIIIXYZIYIX")

        self.assertEqual(p0 @ p1, QubitSparsePauli.from_label("ZIYXXYYYYIXIY"))
        self.assertEqual(p1 @ p0, QubitSparsePauli.from_label("ZIYXXYYYYIXIY"))

        self.assertEqual(p0 @ p0, QubitSparsePauli.from_label("I" * 13))
        self.assertEqual(p1 @ p1, QubitSparsePauli.from_label("I" * 13))

    def test_compose_errors(self):
        p0 = QubitSparsePauli.from_label("XZYI")
        p1 = QubitSparsePauli.from_label("ZIY")
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 4, 3"):
            p0.compose(p1)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 3, 4"):
            p1.compose(p0)

    def test_commutes(self):
        p0 = QubitSparsePauli("XIY")
        p1 = QubitSparsePauli("IZI")
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = QubitSparsePauli("XXY")
        p1 = QubitSparsePauli("IZI")
        self.assertFalse(p0.commutes(p1))
        self.assertFalse(p1.commutes(p0))

        p0 = QubitSparsePauli("XXY")
        p1 = QubitSparsePauli("IZX")
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = QubitSparsePauli("XXYY")
        p1 = QubitSparsePauli("IZXY")
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = QubitSparsePauli("XXYYZ")
        p1 = QubitSparsePauli("IZXYX")
        self.assertFalse(p0.commutes(p1))
        self.assertFalse(p1.commutes(p0))

    def test_commutes_errors(self):
        p0 = QubitSparsePauli.from_label("XZYI")
        p1 = QubitSparsePauli.from_label("ZIY")
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 4, 3"):
            p0.commutes(p1)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 3, 4"):
            p1.commutes(p0)

    def test_to_pauli(self):
        pauli = Pauli("XIZIY")
        self.assertEqual(pauli, QubitSparsePauli(pauli).to_pauli())

        # leading identities
        pauli = Pauli("IIZIY")
        self.assertEqual(pauli, QubitSparsePauli(pauli).to_pauli())

        # trailing identities
        pauli = Pauli("XIZIYII")
        self.assertEqual(pauli, QubitSparsePauli(pauli).to_pauli())

        # both
        pauli = Pauli("IIXIZIYII")
        self.assertEqual(pauli, QubitSparsePauli(pauli).to_pauli())


@ddt.ddt
class TestQubitSparsePauliList(QiskitTestCase):
    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_pauli(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=data.num_qubits),
            QubitSparsePauliList.from_pauli(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(
            QubitSparsePauliList(with_phase), QubitSparsePauliList.from_pauli(with_phase)
        )
        self.assertEqual(
            QubitSparsePauliList(with_phase, num_qubits=data.num_qubits),
            QubitSparsePauliList.from_pauli(with_phase),
        )

        self.assertEqual(
            QubitSparsePauliList(Pauli("")), QubitSparsePauliList.from_pauli(Pauli(""))
        )

    def test_default_constructor_list(self):
        data = ["IXIIZ", "XIXII", "IIXYI"]
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_list(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=5), QubitSparsePauliList.from_list(data)
        )
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            QubitSparsePauliList(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            QubitSparsePauliList(data, num_qubits=6)
        self.assertEqual(
            QubitSparsePauliList([], num_qubits=5), QubitSparsePauliList.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3)), ("XY", (2, 4)), ("ZY", (2, 1))]
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=5),
            QubitSparsePauliList.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=10),
            QubitSparsePauliList.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            QubitSparsePauliList(data)
        self.assertEqual(
            QubitSparsePauliList([], num_qubits=5),
            QubitSparsePauliList.from_sparse_list([], num_qubits=5),
        )

    def test_default_constructor_label(self):
        data = "IIXIXXIZZYYIYZ"
        self.assertEqual(QubitSparsePauliList(data), QubitSparsePauliList.from_label(data))
        self.assertEqual(
            QubitSparsePauliList(data, num_qubits=len(data)), QubitSparsePauliList.from_label(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(data, num_qubits=len(data) + 1)

    def test_default_constructor_copy(self):
        base = QubitSparsePauliList.from_list(["IXIZIY", "XYZIII"])
        copied = QubitSparsePauliList(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            QubitSparsePauliList(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = QubitSparsePauliList.from_list(["IIZXII"])
        self.assertEqual(QubitSparsePauliList(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = QubitSparsePauliList.from_list(["IIZXII", "IIIIII"])
        terms = [expected[0], expected[1]]
        self.assertEqual(QubitSparsePauliList(list(terms)), expected)
        self.assertEqual(QubitSparsePauliList(tuple(terms)), expected)
        self.assertEqual(QubitSparsePauliList(term for term in terms), expected)

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        qs_list = QubitSparsePauliList.from_label("IXXIIZZIYYIXYZ")
        self.assertEqual(len(qs_list), 1)

        self.assertEqual(qs_list[0], QubitSparsePauli.from_label("IXXIIZZIYYIXYZ"))

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauliList.from_label("🐍")

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            QubitSparsePauliList.from_list([label]), QubitSparsePauliList.from_label(label)
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label], num_qubits=len(label)),
            QubitSparsePauliList.from_label(label),
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label])[0],
            QubitSparsePauli.from_raw_parts(
                len(label),
                [
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
            ),
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([label], num_qubits=len(label))[0],
            QubitSparsePauli.from_raw_parts(
                len(label),
                [
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.Y,
                    QubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
            ),
        )

        self.assertEqual(
            QubitSparsePauliList.from_list(["IIIXZI", "XXIIII"])[0],
            QubitSparsePauli.from_raw_parts(
                6,
                [
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.X,
                ],
                [1, 2],
            ),
        )

        self.assertEqual(
            QubitSparsePauliList.from_list(["IIIXZI", "XXIIII"])[1],
            QubitSparsePauli.from_raw_parts(
                6,
                [
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.X,
                ],
                [4, 5],
            ),
        )

        self.assertEqual(
            QubitSparsePauliList.from_list([], num_qubits=5), QubitSparsePauliList.empty(5)
        )
        self.assertEqual(
            QubitSparsePauliList.from_list([], num_qubits=0), QubitSparsePauliList.empty(0)
        )

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_list(["XZIIZY", "I+-$%I"])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            QubitSparsePauliList.from_list(["🐍"])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IIXI"])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "II"])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            QubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            QubitSparsePauliList.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(
                [
                    ("XY", (0, 1)),
                    ("XX", (1, 3)),
                    ("YYZ", (0, 2, 4)),
                ],
                num_qubits=5,
            ),
            QubitSparsePauliList.from_list(["IIIYX", "IXIXI", "ZIYIY"]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = QubitSparsePauliList.from_sparse_list(
            [
                ("XYZ", (2, 1, 0)),
                ("XYZ", (2, 0, 1)),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, QubitSparsePauliList.from_list(["XYZ", "XZY"]))
        np.testing.assert_equal(from_unsorted[0].indices, np.array([0, 1, 2], dtype=np.uint32))
        np.testing.assert_equal(from_unsorted[1].indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = QubitSparsePauliList.from_sparse_list(
            [
                ("ZXI", (0, 1, 2)),
                ("XYIII", (0, 1, 2, 3, 8)),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            QubitSparsePauliList.from_sparse_list([("XZ", (1, 0)), ("YX", (1, 0))], num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity[0].indices, np.array([0, 1], dtype=np.uint32))
        np.testing.assert_equal(explicit_identity[1].indices, np.array([0, 1], dtype=np.uint32))

        self.assertEqual(
            QubitSparsePauliList.from_sparse_list([], num_qubits=1_000_000),
            QubitSparsePauliList.empty(1_000_000),
        )
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list([], num_qubits=0),
            QubitSparsePauliList.empty(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            QubitSparsePauliList.from_sparse_list(
                [("XZZY", (5, 3, 1, 0)), ("+$", (2, 1))], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            QubitSparsePauliList.from_sparse_list([("\xff", (1,))], num_qubits=5)
        with self.assertRaises(ValueError):
            QubitSparsePauliList.from_sparse_list([("\xff", (1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            QubitSparsePauliList.from_sparse_list([("XZ", (0,))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            QubitSparsePauliList.from_sparse_list([("XZ", (0, 1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            QubitSparsePauliList.from_sparse_list([("XZY", (0, 1, 3))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            QubitSparsePauliList.from_sparse_list([("XZY", (0, 1, 4))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            QubitSparsePauliList.from_sparse_list([("XXI", (0, 1, 3))], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauliList.from_sparse_list([("XZ", (3, 3))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            QubitSparsePauliList.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3))], num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("IIXZI")),
            QubitSparsePauliList.from_label("IIXZI"),
        )

        # `Pauli` accepts a phase in its label, which gets dropped
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("iIXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("-iIXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )
        self.assertEqual(
            QubitSparsePauliList.from_pauli(Pauli("-IXZIX")),
            QubitSparsePauliList.from_list(["IXZIX"]),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (QubitSparsePauliList.from_pauli(pauli), QubitSparsePauliList.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (QubitSparsePauliList.from_pauli(pauli), QubitSparsePauliList.from_list([label]))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

    def test_from_qubit_sparse_paulis(self):
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis([], num_qubits=5),
            QubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis((), num_qubits=0),
            QubitSparsePauliList.empty(0),
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis((None for _ in []), num_qubits=3),
            QubitSparsePauliList.empty(3),
        )

        expected = QubitSparsePauliList.from_sparse_list(
            [
                ("XYZ", (4, 2, 1)),
                ("XXYY", (8, 5, 3, 2)),
                ("ZZ", (5, 0)),
            ],
            num_qubits=10,
        )
        self.assertEqual(QubitSparsePauliList.from_qubit_sparse_paulis(list(expected)), expected)
        self.assertEqual(QubitSparsePauliList.from_qubit_sparse_paulis(tuple(expected)), expected)
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis(term for term in expected), expected
        )
        self.assertEqual(
            QubitSparsePauliList.from_qubit_sparse_paulis(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_qubit_sparse_paulis_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            QubitSparsePauliList.from_qubit_sparse_paulis([])

        left, right = (
            QubitSparsePauliList(["IIXYI"])[0],
            QubitSparsePauliList(["IIIIIIIIX"])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_qubit_sparse_paulis([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            QubitSparsePauliList.from_qubit_sparse_paulis([left], num_qubits=100)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            QubitSparsePauliList(["IIXIZ", ("IZ", (2, 3))], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(QubitSparsePauliList.empty(0).num_qubits, 0)
        self.assertEqual(QubitSparsePauliList.empty(10).num_qubits, 10)

    def test_num_terms(self):
        self.assertEqual(QubitSparsePauliList.empty(0).num_terms, 0)
        self.assertEqual(QubitSparsePauliList.empty(10).num_terms, 0)
        self.assertEqual(QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"]).num_terms, 2)

    def test_empty(self):
        empty_5 = QubitSparsePauliList.empty(5)
        self.assertEqual(empty_5.num_qubits, 5)
        self.assertEqual(len(empty_5), 0)
        self.assertEqual(empty_5.to_sparse_list(), [])

        empty_0 = QubitSparsePauliList.empty(0)
        self.assertEqual(empty_0.num_qubits, 0)
        self.assertEqual(len(empty_0), 0)
        self.assertEqual(empty_0.to_sparse_list(), [])

    def test_len(self):
        self.assertEqual(len(QubitSparsePauliList.empty(0)), 0)
        self.assertEqual(len(QubitSparsePauliList.empty(10)), 0)
        self.assertEqual(len(QubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"])), 2)

    @ddt.idata(single_cases_list())
    def test_pickle(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertIsNot(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertEqual(qubit_sparse_pauli_list, copy.deepcopy(qubit_sparse_pauli_list))
        self.assertEqual(
            qubit_sparse_pauli_list, pickle.loads(pickle.dumps(qubit_sparse_pauli_list))
        )

    @ddt.data(
        QubitSparsePauliList.empty(0),
        QubitSparsePauliList.empty(1),
        QubitSparsePauliList.empty(10),
        QubitSparsePauliList.from_label("IIXIZI"),
        QubitSparsePauliList.from_label("X"),
        QubitSparsePauliList.from_list(["YIXZII"]),
        QubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        QubitSparsePauliList.from_list(["IIIIII", "ZZYYXX"]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("QubitSparsePauliList", repr(data))

    @ddt.idata(single_cases_list())
    def test_copy(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, qubit_sparse_pauli_list.copy())
        self.assertIsNot(qubit_sparse_pauli_list, qubit_sparse_pauli_list.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0)), ("XYY", (3, 1, 0))]
        pauli_list = QubitSparsePauliList.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(pauli_list, pauli_list.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli_list, pauli_list)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli_list, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli_list,
            QubitSparsePauliList.from_sparse_list(
                sparse_data, num_qubits=pauli_list.num_qubits + 1
            ),
        )
        self.assertNotEqual(QubitSparsePauliList.empty(2), QubitSparsePauliList.empty(3))

        # Difference in bit terms.
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IIYZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IIXZI", "XXYYY"]),
        )

        # Difference in indices.
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            QubitSparsePauliList.from_list(["IXIZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            QubitSparsePauliList.from_list(["IIXZI", "XIYYZ"]),
            QubitSparsePauliList.from_list(["IIXZI", "IXYYZ"]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            QubitSparsePauliList.from_sparse_list([("XZ", (0, 1)), ("XX", (2, 3))], num_qubits=5),
            QubitSparsePauliList.from_sparse_list([("XZX", (0, 1, 2)), ("X", (3,))], num_qubits=5),
        )

    @ddt.idata(single_cases_list())
    def test_clear(self, pauli_list):
        num_qubits = pauli_list.num_qubits
        pauli_list.clear()
        self.assertEqual(pauli_list, QubitSparsePauliList.empty(num_qubits))

    def test_apply_layout_list(self):
        self.assertEqual(
            QubitSparsePauliList.empty(5).apply_layout([4, 3, 2, 1, 0]),
            QubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            QubitSparsePauliList.empty(3).apply_layout([0, 2, 1], 8), QubitSparsePauliList.empty(8)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(2).apply_layout([1, 0]), QubitSparsePauliList.empty(2)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(3).apply_layout([100, 10_000, 3], 100_000_000),
            QubitSparsePauliList.empty(100_000_000),
        )

        terms = [
            ("ZYX", (4, 2, 1)),
            ("", ()),
            ("XXYYZZ", (10, 8, 6, 4, 2, 0)),
        ]

        def map_indices(terms, layout):
            return [(terms, tuple(layout[bit] for bit in bits)) for terms, bits in terms]

        identity = list(range(12))
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(identity),
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12),
        )
        # We've already tested elsewhere that `QubitSparsePauliList.from_sparse_list` produces termwise
        # sorted indices, so these tests also ensure `apply_layout` is maintaining that invariant.
        backwards = list(range(12))[::-1]
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(backwards),
            QubitSparsePauliList.from_sparse_list(map_indices(terms, backwards), num_qubits=12),
        )
        shuffled = [4, 7, 1, 10, 0, 11, 3, 2, 8, 5, 6, 9]
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled),
            QubitSparsePauliList.from_sparse_list(map_indices(terms, shuffled), num_qubits=12),
        )
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled, 100),
            QubitSparsePauliList.from_sparse_list(map_indices(terms, shuffled), num_qubits=100),
        )
        expanded = [78, 69, 82, 68, 32, 97, 108, 101, 114, 116, 33]
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=11).apply_layout(expanded, 120),
            QubitSparsePauliList.from_sparse_list(map_indices(terms, expanded), num_qubits=120),
        )

    def test_apply_layout_transpiled(self):
        base = QubitSparsePauliList.from_sparse_list(
            [
                ("ZYX", (4, 2, 1)),
                ("", ()),
                ("XXY", (3, 2, 0)),
            ],
            num_qubits=5,
        )

        qc = QuantumCircuit(5)
        initial_list = [3, 4, 0, 2, 1]
        no_routing = transpile(
            qc, target=lnn_target(5), initial_layout=initial_list, seed_transpiler=2024_10_25_0
        ).layout
        # It's easiest here to test against the `list` form, which we verify separately and
        # explicitly.
        self.assertEqual(base.apply_layout(no_routing), base.apply_layout(initial_list))

        expanded = transpile(
            qc, target=lnn_target(100), initial_layout=initial_list, seed_transpiler=2024_10_25_1
        ).layout
        self.assertEqual(
            base.apply_layout(expanded), base.apply_layout(initial_list, num_qubits=100)
        )

        qc = QuantumCircuit(5)
        qargs = list(itertools.permutations(range(5), 2))
        random.Random(2024_10_25_2).shuffle(qargs)
        for pair in qargs:
            qc.cx(*pair)

        routed = transpile(qc, target=lnn_target(5), seed_transpiler=2024_10_25_3).layout
        self.assertEqual(
            base.apply_layout(routed),
            base.apply_layout(routed.final_index_layout(filter_ancillas=True)),
        )

        routed_expanded = transpile(qc, target=lnn_target(20), seed_transpiler=2024_10_25_3).layout
        self.assertEqual(
            base.apply_layout(routed_expanded),
            base.apply_layout(
                routed_expanded.final_index_layout(filter_ancillas=True), num_qubits=20
            ),
        )

    def test_apply_layout_none(self):
        self.assertEqual(
            QubitSparsePauliList.empty(0).apply_layout(None), QubitSparsePauliList.empty(0)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(0).apply_layout(None, 3), QubitSparsePauliList.empty(3)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(5).apply_layout(None), QubitSparsePauliList.empty(5)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(3).apply_layout(None, 8), QubitSparsePauliList.empty(8)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(0).apply_layout(None), QubitSparsePauliList.empty(0)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(0).apply_layout(None, 8), QubitSparsePauliList.empty(8)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(2).apply_layout(None), QubitSparsePauliList.empty(2)
        )
        self.assertEqual(
            QubitSparsePauliList.empty(3).apply_layout(None, 100_000_000),
            QubitSparsePauliList.empty(100_000_000),
        )

        terms = [
            ("ZYX", (2, 1, 0)),
            ("", ()),
            ("XXYYZZ", (10, 8, 6, 4, 2, 0)),
        ]
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(None),
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12),
        )
        self.assertEqual(
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                None, num_qubits=200
            ),
            QubitSparsePauliList.from_sparse_list(terms, num_qubits=200),
        )

    def test_apply_layout_failures(self):
        obs = QubitSparsePauliList.from_list(["IIYI", "IIIX"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            obs.apply_layout([0, 0, 1, 2])
        with self.assertRaisesRegex(ValueError, "does not account for all contained qubits"):
            obs.apply_layout([0, 1])
        with self.assertRaisesRegex(ValueError, "less than the number of qubits"):
            obs.apply_layout([0, 2, 4, 6])
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            obs.apply_layout([0, 1], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            obs.apply_layout(None, num_qubits=2)

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        layout = transpile(qc, target=lnn_target(3), seed_transpiler=2024_10_25).layout
        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            obs.apply_layout(layout, num_qubits=2)

    def test_iteration(self):
        self.assertEqual(list(QubitSparsePauliList.empty(5)), [])
        self.assertEqual(tuple(QubitSparsePauliList.empty(0)), ())

        pauli_list = QubitSparsePauliList.from_sparse_list(
            [
                ("XYY", (4, 2, 1)),
                ("", ()),
                ("ZZ", (3, 0)),
                ("XX", (2, 1)),
                ("YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = QubitSparsePauli.Pauli
        expected = [
            QubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            QubitSparsePauli.from_raw_parts(5, [], []),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3]),
            QubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2]),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(list(pauli_list), expected)

    def test_indexing(self):
        pauli_list = QubitSparsePauliList.from_sparse_list(
            [
                ("XYY", (4, 2, 1)),
                ("", ()),
                ("ZZ", (3, 0)),
                ("XX", (2, 1)),
                ("YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = QubitSparsePauli.Pauli
        expected = [
            QubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            QubitSparsePauli.from_raw_parts(5, [], []),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3]),
            QubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2]),
            QubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(pauli_list[0], expected[0])
        self.assertEqual(pauli_list[-2], expected[-2])
        self.assertEqual(pauli_list[2:4], QubitSparsePauliList(expected[2:4]))
        self.assertEqual(pauli_list[1::2], QubitSparsePauliList(expected[1::2]))
        self.assertEqual(pauli_list[:], QubitSparsePauliList(expected))
        self.assertEqual(pauli_list[-1:-4:-1], QubitSparsePauliList(expected[-1:-4:-1]))

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="empty"):
            pauli_list = QubitSparsePauliList.empty(100)
            expected = []
            self.assertEqual(expected, pauli_list.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            pauli_list = QubitSparsePauliList(["IXYZ"])
            expected = [("ZYX", [0, 1, 2])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )

        with self.subTest(msg="multiple"):
            pauli_list = QubitSparsePauliList.from_list(["XXIZ", "YYIZ"])
            expected = [("XXZ", [3, 2, 0]), ("ZYY", [0, 2, 3])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )

    def test_to_pauli_list(self):
        pauli_strings = ["XIZIY", "IIZIY", "ZIYII", "IIZII"]
        pauli_list = PauliList(pauli_strings)
        self.assertEqual(pauli_list, QubitSparsePauliList.from_list(pauli_strings).to_pauli_list())

        # single element
        pauli_strings = ["XIZIY"]
        pauli_list = PauliList(pauli_strings)
        self.assertEqual(pauli_list, QubitSparsePauliList.from_list(pauli_strings).to_pauli_list())

    def test_to_dense_array(self):
        """Test converting to a dense array."""
        with self.subTest(msg="empty"):
            pauli_list = QubitSparsePauliList.empty(100)
            expected = np.zeros((0, 100), dtype=np.uint8)
            np.testing.assert_array_equal(pauli_list.to_dense_array(), expected, strict=True)

        with self.subTest(msg="single"):
            pauli_list = QubitSparsePauliList.from_sparse_list([("XZ", (0, 2))], num_qubits=5)
            expected = np.array([[2, 0, 1, 0, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(pauli_list.to_dense_array(), expected, strict=True)

        with self.subTest(msg="multiple"):
            pauli_list = QubitSparsePauliList.from_list(["IXI", "YII", "IIZ"])
            expected = np.array([[0, 2, 0], [0, 0, 3], [1, 0, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(pauli_list.to_dense_array(), expected, strict=True)

    def test_commutes(self):
        p0 = QubitSparsePauliList.from_list(["XIY", "IXI", "IZI"])
        p1 = QubitSparsePauliList.from_list(["IZI", "ZII"])
        expected = np.array([[True, False], [False, True], [True, True]], dtype=np.bool_)
        np.testing.assert_array_equal(p0.commutes(p1), expected, strict=True)
        np.testing.assert_array_equal(p1.commutes(p0), expected.T, strict=True)

    def test_commutes_empty(self):
        p0 = QubitSparsePauliList.from_sparse_list([], num_qubits=3)
        p1 = QubitSparsePauliList.from_list(["ZZZ"])
        expected = np.empty((0, 1), dtype=np.bool_)
        np.testing.assert_array_equal(p0.commutes(p1), expected, strict=True)
        np.testing.assert_array_equal(p1.commutes(p0), expected.T, strict=True)

    def test_commutes_errors(self):
        p0 = QubitSparsePauliList.from_label("XZYI")
        p1 = QubitSparsePauliList.from_label("ZIY")
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 4, 3"):
            p0.commutes(p1)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 3, 4"):
            p1.commutes(p0)


def canonicalize_term(pauli, indices):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (sorted_paulis, np.asarray(indices)[idcs].tolist())


def canonicalize_sparse_list(sparse_list):
    # sort a sparse list representation by canonicalizing the terms and then applying
    # Python's built-in sort
    canonicalized_terms = [canonicalize_term(*term) for term in sparse_list]
    return sorted(canonicalized_terms)


def lnn_target(num_qubits):
    """Create a simple `Target` object with an arbitrary basis-gate set, and open-path
    connectivity."""
    out = Target()
    out.add_instruction(library.RZGate(Parameter("a")), {(q,): None for q in range(num_qubits)})
    out.add_instruction(library.SXGate(), {(q,): None for q in range(num_qubits)})
    out.add_instruction(Measure(), {(q,): None for q in range(num_qubits)})
    out.add_instruction(
        library.CXGate(),
        {
            pair: None
            for lower in range(num_qubits - 1)
            for pair in [(lower, lower + 1), (lower + 1, lower)]
        },
    )
    return out
