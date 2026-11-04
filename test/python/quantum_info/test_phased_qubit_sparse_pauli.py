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
from qiskit.quantum_info import PhasedQubitSparsePauli, PhasedQubitSparsePauliList, Pauli, PauliList
from qiskit.transpiler import Target

from test import QiskitTestCase


def single_cases():
    return [
        PhasedQubitSparsePauli(""),
        PhasedQubitSparsePauli("I" * 10),
        PhasedQubitSparsePauli.from_label("IIXIZI"),
        PhasedQubitSparsePauli(Pauli("iZZYYXX")),
    ]


def single_cases_list():
    return [
        PhasedQubitSparsePauliList.empty(0),
        PhasedQubitSparsePauliList.empty(10),
        PhasedQubitSparsePauliList.from_label("IIXIZI"),
        PhasedQubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        # Includes a duplicate entry.
        PhasedQubitSparsePauliList.from_list(["IXZ", "ZZI", "IXZ"]),
        PhasedQubitSparsePauli(Pauli("iZZYYXX")).to_phased_qubit_sparse_pauli_list(),
    ]


@ddt.ddt
class TestPhasedQubitSparsePauli(QiskitTestCase):

    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(PhasedQubitSparsePauli(data), PhasedQubitSparsePauli.from_pauli(data))
        self.assertEqual(
            PhasedQubitSparsePauli(data, num_qubits=data.num_qubits),
            PhasedQubitSparsePauli.from_pauli(data),
        )
        self.assertEqual(PhasedQubitSparsePauli(data).phase, 0)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PhasedQubitSparsePauli(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(
            PhasedQubitSparsePauli(with_phase), PhasedQubitSparsePauli.from_pauli(with_phase)
        )
        self.assertEqual(
            PhasedQubitSparsePauli(with_phase, num_qubits=data.num_qubits),
            PhasedQubitSparsePauli.from_pauli(with_phase),
        )
        self.assertEqual(PhasedQubitSparsePauli(with_phase).phase, 1)

        self.assertEqual(
            PhasedQubitSparsePauli(Pauli("")), PhasedQubitSparsePauli.from_pauli(Pauli(""))
        )

    def test_default_constructor_label(self):
        data = "IXIIZ"
        self.assertEqual(PhasedQubitSparsePauli(data), PhasedQubitSparsePauli.from_label(data))
        self.assertEqual(
            PhasedQubitSparsePauli(data, num_qubits=5), PhasedQubitSparsePauli.from_label(data)
        )
        with self.assertRaisesRegex(ValueError, "does not match label"):
            PhasedQubitSparsePauli(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "does not match label"):
            PhasedQubitSparsePauli(data, num_qubits=6)

    def test_default_constructor_sparse_label(self):
        data = (1, "ZX", (0, 3))
        self.assertEqual(
            PhasedQubitSparsePauli(data, num_qubits=5),
            PhasedQubitSparsePauli.from_sparse_label(data, num_qubits=5),
        )
        self.assertEqual(
            PhasedQubitSparsePauli(data, num_qubits=10),
            PhasedQubitSparsePauli.from_sparse_label(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            PhasedQubitSparsePauli(data)
        self.assertEqual(
            PhasedQubitSparsePauli((0, "", []), num_qubits=5),
            PhasedQubitSparsePauli.from_sparse_label((0, "", []), num_qubits=5),
        )

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            PhasedQubitSparsePauli.from_label("IXXIIZZIYYIXYZ"),
            PhasedQubitSparsePauli.from_raw_parts(
                14,
                [
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.X,
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.X,
                    PhasedQubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
                0,
            ),
        )

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PhasedQubitSparsePauli.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PhasedQubitSparsePauli.from_label("🐍")

    def test_from_sparse_label(self):
        self.assertEqual(
            PhasedQubitSparsePauli.from_sparse_label((1, "XY", (0, 1)), num_qubits=5),
            PhasedQubitSparsePauli(Pauli("-iIIIYX")),
        )
        self.assertEqual(
            PhasedQubitSparsePauli.from_sparse_label((0, "XX", (1, 3)), num_qubits=5),
            PhasedQubitSparsePauli.from_label("IXIXI"),
        )
        self.assertEqual(
            PhasedQubitSparsePauli.from_sparse_label((3, "YYZ", (0, 2, 4)), num_qubits=5),
            PhasedQubitSparsePauli(Pauli("iZIYIY")),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = PhasedQubitSparsePauli.from_sparse_label(
            (0, "XYZ", (2, 0, 1)), num_qubits=3
        )
        self.assertEqual(from_unsorted, PhasedQubitSparsePauli.from_label("XZY"))
        np.testing.assert_equal(from_unsorted.indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = PhasedQubitSparsePauli.from_sparse_label(
            (0, "ZXI", (0, 1, 2)), num_qubits=10
        )
        self.assertEqual(
            explicit_identity,
            PhasedQubitSparsePauli.from_sparse_label((0, "XZ", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

        explicit_identity = PhasedQubitSparsePauli.from_sparse_label(
            (0, "XYIII", (0, 1, 2, 3, 8)), num_qubits=10
        )
        self.assertEqual(
            explicit_identity,
            PhasedQubitSparsePauli.from_sparse_label((0, "YX", (1, 0)), num_qubits=10),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1], dtype=np.uint32))

    def test_from_sparse_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PhasedQubitSparsePauli.from_sparse_label((0, "+$", (2, 1)), num_qubits=8)
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            PhasedQubitSparsePauli.from_sparse_label((0, "\xff", (1,)), num_qubits=5)
        with self.assertRaises(ValueError):
            PhasedQubitSparsePauli.from_sparse_label((0, "\xff", (1, 2)), num_qubits=5)

        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 1"
        ):
            PhasedQubitSparsePauli.from_sparse_label((0, "XZ", (0,)), num_qubits=5)
        with self.assertRaisesRegex(
            ValueError, "label with length 2 does not match indices of length 3"
        ):
            PhasedQubitSparsePauli.from_sparse_label((0, "XZ", (0, 1, 2)), num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            PhasedQubitSparsePauli.from_sparse_label((2, "XZY", (0, 1, 3)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            PhasedQubitSparsePauli.from_sparse_label((1, "XZY", (0, 1, 4)), num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            PhasedQubitSparsePauli.from_sparse_label((4, "XXI", (0, 1, 3)), num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PhasedQubitSparsePauli.from_sparse_label((0, "XZ", (3, 3)), num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PhasedQubitSparsePauli.from_sparse_label((0, "XYZXZ", (3, 0, 1, 2, 3)), num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            PhasedQubitSparsePauli.from_pauli(Pauli("IIXZI")),
            PhasedQubitSparsePauli.from_label("IIXZI"),
        )

        self.assertEqual(
            PhasedQubitSparsePauli.from_pauli(Pauli("iIXZIX")).phase,
            3,
        )
        self.assertEqual(
            PhasedQubitSparsePauli.from_pauli(Pauli("-iIXZIX")).phase,
            1,
        )
        self.assertEqual(
            PhasedQubitSparsePauli.from_pauli(Pauli("-IXZIX")).phase,
            2,
        )

        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (PhasedQubitSparsePauli.from_pauli(pauli), PhasedQubitSparsePauli.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            PhasedQubitSparsePauli(5, num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(PhasedQubitSparsePauli("").num_qubits, 0)
        self.assertEqual(PhasedQubitSparsePauli("I" * 10).num_qubits, 10)

    def test_pauli_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(PhasedQubitSparsePauli.Pauli),
            {
                PhasedQubitSparsePauli.Pauli.X,
                PhasedQubitSparsePauli.Pauli.Y,
                PhasedQubitSparsePauli.Pauli.Z,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(PhasedQubitSparsePauli.Pauli.X, int)
        values = {
            "X": 0b10,
            "Y": 0b11,
            "Z": 0b01,
        }
        self.assertEqual(
            {name: getattr(PhasedQubitSparsePauli.Pauli, name) for name in values}, values
        )

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": PhasedQubitSparsePauli.Pauli.X,
            "Y": PhasedQubitSparsePauli.Pauli.Y,
            "Z": PhasedQubitSparsePauli.Pauli.Z,
        }
        self.assertEqual({label: PhasedQubitSparsePauli.Pauli[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual({pauli.label: pauli for pauli in PhasedQubitSparsePauli.Pauli}, labels)

    @ddt.idata(single_cases())
    def test_pickle(self, phased_qubit_sparse_pauli):
        self.assertEqual(phased_qubit_sparse_pauli, copy.copy(phased_qubit_sparse_pauli))
        self.assertIsNot(phased_qubit_sparse_pauli, copy.copy(phased_qubit_sparse_pauli))
        self.assertEqual(phased_qubit_sparse_pauli, copy.deepcopy(phased_qubit_sparse_pauli))
        self.assertEqual(
            phased_qubit_sparse_pauli, pickle.loads(pickle.dumps(phased_qubit_sparse_pauli))
        )

    @ddt.data(
        PhasedQubitSparsePauli.from_label("IIXIZI"),
        PhasedQubitSparsePauli.from_label("X"),
        PhasedQubitSparsePauli(Pauli("iIII")),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("PhasedQubitSparsePauli", repr(data))

    @ddt.idata(single_cases())
    def test_copy(self, phased_qubit_sparse_pauli):
        self.assertEqual(phased_qubit_sparse_pauli, phased_qubit_sparse_pauli.copy())
        self.assertIsNot(phased_qubit_sparse_pauli, phased_qubit_sparse_pauli.copy())

    def test_equality(self):
        sparse_data = (1, "XYY", (3, 1, 0))
        pauli = PhasedQubitSparsePauli.from_sparse_label(sparse_data, num_qubits=5)
        self.assertEqual(pauli, pauli.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli, pauli)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli,
            PhasedQubitSparsePauli.from_sparse_label(sparse_data, num_qubits=pauli.num_qubits + 1),
        )

        # Difference in bit terms.
        self.assertNotEqual(
            PhasedQubitSparsePauli.from_label("IIXZI"),
            PhasedQubitSparsePauli.from_label("IIYZI"),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauli.from_label("XXYYZ"),
            PhasedQubitSparsePauli.from_label("XXYYY"),
        )

        # Difference in indices.
        self.assertNotEqual(
            PhasedQubitSparsePauli.from_label("IIXZI"),
            PhasedQubitSparsePauli.from_label("IXIZI"),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauli.from_label("XIYYZ"),
            PhasedQubitSparsePauli.from_label("IXYYZ"),
        )

        # Difference in phase.
        self.assertNotEqual(
            PhasedQubitSparsePauli(Pauli("IIXZI")),
            PhasedQubitSparsePauli(Pauli("-IIXZI")),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauli(Pauli("iXIYYZ")),
            PhasedQubitSparsePauli(Pauli("-iXIYYZ")),
        )

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        pauli = PhasedQubitSparsePauli.from_label("XZY")
        self.assertEqual(len(pauli.indices), 3)
        self.assertEqual(len(pauli.paulis), 3)

        # Iteration
        self.assertEqual(tuple(pauli.indices), (0, 1, 2))
        # multiple iteration through same object
        paulis = pauli.paulis
        self.assertEqual(set(paulis), {PhasedQubitSparsePauli.Pauli[x] for x in "XZY"})

        # Implicit iteration methods.
        self.assertIn(PhasedQubitSparsePauli.Pauli.Y, pauli.paulis)
        self.assertNotIn(4, pauli.indices)

        # Index by scalar
        self.assertEqual(pauli.indices[-1], 2)
        self.assertEqual(pauli.paulis[0], PhasedQubitSparsePauli.Pauli.Y)

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
            np.array(
                [PhasedQubitSparsePauli.Pauli.Y, PhasedQubitSparsePauli.Pauli.Z], dtype=np.uint8
            ),
            strict=True,
        )

    def test_attributes_to_array(self):
        pauli = PhasedQubitSparsePauli.from_label("XZY")

        # Natural dtypes.
        np.testing.assert_array_equal(
            pauli.indices, np.array([0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            pauli.paulis,
            np.array([PhasedQubitSparsePauli.Pauli[x] for x in "YZX"], dtype=np.uint8),
            strict=True,
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(pauli.indices, dtype=np.uint8),
            np.array([0, 1, 2], dtype=np.uint8),
            strict=True,
        )

    def test_identity(self):
        identity_5 = PhasedQubitSparsePauli.identity(5)
        self.assertEqual(identity_5.num_qubits, 5)
        self.assertEqual(len(identity_5.paulis), 0)
        self.assertEqual(len(identity_5.indices), 0)

        identity_0 = PhasedQubitSparsePauli.identity(0)
        self.assertEqual(identity_0.num_qubits, 0)
        self.assertEqual(len(identity_0.paulis), 0)
        self.assertEqual(len(identity_0.indices), 0)

    def test_compose(self):
        p0 = PhasedQubitSparsePauli("XZY")
        p1 = PhasedQubitSparsePauli("ZIY")

        self.assertEqual(p0.compose(p1), PhasedQubitSparsePauli(Pauli("-iYZI")))
        self.assertEqual(p1.compose(p0), PhasedQubitSparsePauli(Pauli("iYZI")))

        p0 = PhasedQubitSparsePauli.from_label("III")
        p1 = PhasedQubitSparsePauli.from_label("ZIY")

        self.assertEqual(p0 @ p1, PhasedQubitSparsePauli.from_label("ZIY"))
        self.assertEqual(p1 @ p0, PhasedQubitSparsePauli.from_label("ZIY"))

        p0 = PhasedQubitSparsePauli.from_label("IIIXXY")
        p1 = PhasedQubitSparsePauli.from_label("ZIYIII")

        self.assertEqual(p0 @ p1, PhasedQubitSparsePauli.from_label("ZIYXXY"))
        self.assertEqual(p1 @ p0, PhasedQubitSparsePauli.from_label("ZIYXXY"))

        p0 = PhasedQubitSparsePauli(Pauli("IIIXXYZIXIZIZ"))
        p1 = PhasedQubitSparsePauli(Pauli("ZIYIIIXYZIYIX"))

        self.assertEqual(
            p0 @ p1, PhasedQubitSparsePauli(Pauli("IIIXXYZIXIZIZ") @ Pauli("ZIYIIIXYZIYIX"))
        )
        self.assertEqual(
            p1 @ p0, PhasedQubitSparsePauli(Pauli("ZIYIIIXYZIYIX") @ Pauli("IIIXXYZIXIZIZ"))
        )

        self.assertEqual(p0 @ p0, PhasedQubitSparsePauli.from_label("I" * 13))
        self.assertEqual(p1 @ p1, PhasedQubitSparsePauli.from_label("I" * 13))

    def test_compose_errors(self):
        p0 = PhasedQubitSparsePauli.from_label("XZYI")
        p1 = PhasedQubitSparsePauli.from_label("ZIY")
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 4, 3"):
            p0.compose(p1)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 3, 4"):
            p1.compose(p0)

    def test_commutes(self):
        p0 = PhasedQubitSparsePauli("XIY")
        p1 = PhasedQubitSparsePauli("IZI")
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = PhasedQubitSparsePauli(Pauli("-XXY"))
        p1 = PhasedQubitSparsePauli("IZI")
        self.assertFalse(p0.commutes(p1))
        self.assertFalse(p1.commutes(p0))

        p0 = PhasedQubitSparsePauli("XXY")
        p1 = PhasedQubitSparsePauli(Pauli("-iIZX"))
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = PhasedQubitSparsePauli(Pauli("-jXXYY"))
        p1 = PhasedQubitSparsePauli(Pauli("jIZXY"))
        self.assertTrue(p0.commutes(p1))
        self.assertTrue(p1.commutes(p0))

        p0 = PhasedQubitSparsePauli("XXYYZ")
        p1 = PhasedQubitSparsePauli("IZXYX")
        self.assertFalse(p0.commutes(p1))
        self.assertFalse(p1.commutes(p0))

    def test_commutes_errors(self):
        p0 = PhasedQubitSparsePauli.from_label("XZYI")
        p1 = PhasedQubitSparsePauli(Pauli("jZIY"))
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 4, 3"):
            p0.commutes(p1)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits: 3, 4"):
            p1.commutes(p0)

    def test_to_pauli(self):
        pauli = Pauli("XIZIY")
        self.assertEqual(pauli, PhasedQubitSparsePauli(pauli).to_pauli())

        # leading identities
        pauli = Pauli("iIIZIY")
        self.assertEqual(pauli, PhasedQubitSparsePauli(pauli).to_pauli())

        # trailing identities
        pauli = Pauli("-iXIZIYII")
        self.assertEqual(pauli, PhasedQubitSparsePauli(pauli).to_pauli())

        # both
        pauli = Pauli("-IIXIZIYII")
        self.assertEqual(pauli, PhasedQubitSparsePauli(pauli).to_pauli())


@ddt.ddt
class TestPhasedQubitSparsePauliList(QiskitTestCase):

    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(
            PhasedQubitSparsePauliList(data), PhasedQubitSparsePauliList.from_pauli(data)
        )
        self.assertEqual(
            PhasedQubitSparsePauliList(data, num_qubits=data.num_qubits),
            PhasedQubitSparsePauliList.from_pauli(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PhasedQubitSparsePauliList(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(
            PhasedQubitSparsePauliList(with_phase),
            PhasedQubitSparsePauliList.from_pauli(with_phase),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList(with_phase, num_qubits=data.num_qubits),
            PhasedQubitSparsePauliList.from_pauli(with_phase),
        )

        self.assertEqual(
            PhasedQubitSparsePauliList(Pauli("")), PhasedQubitSparsePauliList.from_pauli(Pauli(""))
        )

    def test_default_constructor_list(self):
        data = ["IXIIZ", "XIXII", "IIXYI"]
        self.assertEqual(
            PhasedQubitSparsePauliList(data), PhasedQubitSparsePauliList.from_list(data)
        )
        self.assertEqual(
            PhasedQubitSparsePauliList(data, num_qubits=5),
            PhasedQubitSparsePauliList.from_list(data),
        )
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PhasedQubitSparsePauliList(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PhasedQubitSparsePauliList(data, num_qubits=6)
        self.assertEqual(
            PhasedQubitSparsePauliList([], num_qubits=5),
            PhasedQubitSparsePauliList.from_list([], num_qubits=5),
        )

    def test_default_constructor_sparse_list(self):
        data = [(0, "ZX", (0, 3)), (1, "XY", (2, 4)), (2, "ZY", (2, 1))]
        self.assertEqual(
            PhasedQubitSparsePauliList(data, num_qubits=5),
            PhasedQubitSparsePauliList.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList(data, num_qubits=10),
            PhasedQubitSparsePauliList.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            PhasedQubitSparsePauliList(data)
        self.assertEqual(
            PhasedQubitSparsePauliList([], num_qubits=5),
            PhasedQubitSparsePauliList.from_sparse_list([], num_qubits=5),
        )

    def test_default_constructor_label(self):
        data = "IIXIXXIZZYYIYZ"
        self.assertEqual(
            PhasedQubitSparsePauliList(data), PhasedQubitSparsePauliList.from_label(data)
        )
        self.assertEqual(
            PhasedQubitSparsePauliList(data, num_qubits=len(data)),
            PhasedQubitSparsePauliList.from_label(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PhasedQubitSparsePauliList(data, num_qubits=len(data) + 1)

    def test_default_constructor_copy(self):
        base = PhasedQubitSparsePauliList.from_list(["IXIZIY", "XYZIII"])
        copied = PhasedQubitSparsePauliList(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PhasedQubitSparsePauliList(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = PhasedQubitSparsePauliList.from_list(["IIZXII"])
        self.assertEqual(PhasedQubitSparsePauliList(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = PhasedQubitSparsePauliList.from_list(["IIZXII", "IIIIII"])
        terms = [expected[0], expected[1]]
        self.assertEqual(PhasedQubitSparsePauliList(list(terms)), expected)
        self.assertEqual(PhasedQubitSparsePauliList(tuple(terms)), expected)
        self.assertEqual(PhasedQubitSparsePauliList(term for term in terms), expected)

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        qs_list = PhasedQubitSparsePauliList.from_label("IXXIIZZIYYIXYZ")
        self.assertEqual(len(qs_list), 1)

        self.assertEqual(qs_list[0], PhasedQubitSparsePauli.from_label("IXXIIZZIYYIXYZ"))

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PhasedQubitSparsePauliList.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PhasedQubitSparsePauliList.from_label("🐍")

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([label]),
            PhasedQubitSparsePauliList.from_label(label),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([label], num_qubits=len(label)),
            PhasedQubitSparsePauliList.from_label(label),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([label])[0],
            PhasedQubitSparsePauli.from_raw_parts(
                len(label),
                [
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
            ),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([label], num_qubits=len(label))[0],
            PhasedQubitSparsePauli.from_raw_parts(
                len(label),
                [
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.Y,
                    PhasedQubitSparsePauli.Pauli.X,
                ],
                [0, 1, 2, 4, 5],
            ),
        )

        self.assertEqual(
            PhasedQubitSparsePauliList.from_list(["IIIXZI", "XXIIII"])[0],
            PhasedQubitSparsePauli.from_raw_parts(
                6,
                [
                    PhasedQubitSparsePauli.Pauli.Z,
                    PhasedQubitSparsePauli.Pauli.X,
                ],
                [1, 2],
            ),
        )

        self.assertEqual(
            PhasedQubitSparsePauliList.from_list(["IIIXZI", "XXIIII"])[1],
            PhasedQubitSparsePauli.from_raw_parts(
                6,
                [
                    PhasedQubitSparsePauli.Pauli.X,
                    PhasedQubitSparsePauli.Pauli.X,
                ],
                [4, 5],
            ),
        )

        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([], num_qubits=5),
            PhasedQubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_list([], num_qubits=0),
            PhasedQubitSparsePauliList.empty(0),
        )

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PhasedQubitSparsePauliList.from_list(["XZIIZY", "I+-$%I"])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PhasedQubitSparsePauliList.from_list(["🐍"])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            PhasedQubitSparsePauliList.from_list(["IIZ", "IIXI"])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            PhasedQubitSparsePauliList.from_list(["IIZ", "II"])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PhasedQubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PhasedQubitSparsePauliList.from_list(["IIZ", "IXI"], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PhasedQubitSparsePauliList.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(
                [
                    (0, "XY", (0, 1)),
                    (0, "XX", (1, 3)),
                    (0, "YYZ", (0, 2, 4)),
                ],
                num_qubits=5,
            ),
            PhasedQubitSparsePauliList.from_list(["IIIYX", "IXIXI", "ZIYIY"]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "XYZ", (2, 1, 0)),
                (0, "XYZ", (2, 0, 1)),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, PhasedQubitSparsePauliList.from_list(["XYZ", "XZY"]))
        np.testing.assert_equal(from_unsorted[0].indices, np.array([0, 1, 2], dtype=np.uint32))
        np.testing.assert_equal(from_unsorted[1].indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "ZXI", (0, 1, 2)),
                (0, "XYIII", (0, 1, 2, 3, 8)),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZ", (1, 0)), (0, "YX", (1, 0))], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity[0].indices, np.array([0, 1], dtype=np.uint32))
        np.testing.assert_equal(explicit_identity[1].indices, np.array([0, 1], dtype=np.uint32))

        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list([], num_qubits=1_000_000),
            PhasedQubitSparsePauliList.empty(1_000_000),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list([], num_qubits=0),
            PhasedQubitSparsePauliList.empty(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZZY", (5, 3, 1, 0)), (0, "+$", (2, 1))], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "\xff", (1,))], num_qubits=5)
        with self.assertRaises(ValueError):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "\xff", (1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XZ", (0,))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XZ", (0, 1, 2))], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XZY", (0, 1, 3))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XZY", (0, 1, 4))], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XXI", (0, 1, 3))], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PhasedQubitSparsePauliList.from_sparse_list([(0, "XZ", (3, 3))], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XYZXZ", (3, 0, 1, 2, 3))], num_qubits=5
            )

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            PhasedQubitSparsePauliList.from_pauli(Pauli("IIXZI")),
            PhasedQubitSparsePauliList.from_label("IIXZI"),
        )

        self.assertEqual(
            PhasedQubitSparsePauliList.from_pauli(Pauli("iIXZIX")),
            PhasedQubitSparsePauli((3, "XZX", (0, 2, 3)), 5).to_phased_qubit_sparse_pauli_list(),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_pauli(Pauli("-iIXZIX")),
            PhasedQubitSparsePauli((1, "XZX", (0, 2, 3)), 5).to_phased_qubit_sparse_pauli_list(),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_pauli(Pauli("-IXZIX")),
            PhasedQubitSparsePauli((2, "XZX", (0, 2, 3)), 5).to_phased_qubit_sparse_pauli_list(),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (
                    PhasedQubitSparsePauliList.from_pauli(pauli),
                    PhasedQubitSparsePauliList.from_label(label),
                )
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

    def test_from_phased_qubit_sparse_paulis(self):
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis([], num_qubits=5),
            PhasedQubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis((), num_qubits=0),
            PhasedQubitSparsePauliList.empty(0),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(
                (None for _ in []), num_qubits=3
            ),
            PhasedQubitSparsePauliList.empty(3),
        )

        expected = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "XYZ", (4, 2, 1)),
                (0, "XXYY", (8, 5, 3, 2)),
                (0, "ZZ", (5, 0)),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(list(expected)), expected
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(tuple(expected)), expected
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(term for term in expected),
            expected,
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_qubit_sparse_paulis_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis([])

        left, right = (
            PhasedQubitSparsePauliList(["IIXYI"])[0],
            PhasedQubitSparsePauliList(["IIIIIIIIX"])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis([left], num_qubits=100)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            PhasedQubitSparsePauliList(["IIXIZ", (0, "IZ", (2, 3))], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(PhasedQubitSparsePauliList.empty(0).num_qubits, 0)
        self.assertEqual(PhasedQubitSparsePauliList.empty(10).num_qubits, 10)

    def test_num_terms(self):
        self.assertEqual(PhasedQubitSparsePauliList.empty(0).num_terms, 0)
        self.assertEqual(PhasedQubitSparsePauliList.empty(10).num_terms, 0)
        self.assertEqual(PhasedQubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"]).num_terms, 2)

    def test_empty(self):
        empty_5 = PhasedQubitSparsePauliList.empty(5)
        self.assertEqual(empty_5.num_qubits, 5)
        self.assertEqual(len(empty_5), 0)
        self.assertEqual(empty_5.to_sparse_list(), [])

        empty_0 = PhasedQubitSparsePauliList.empty(0)
        self.assertEqual(empty_0.num_qubits, 0)
        self.assertEqual(len(empty_0), 0)
        self.assertEqual(empty_0.to_sparse_list(), [])

    def test_len(self):
        self.assertEqual(len(PhasedQubitSparsePauliList.empty(0)), 0)
        self.assertEqual(len(PhasedQubitSparsePauliList.empty(10)), 0)
        self.assertEqual(len(PhasedQubitSparsePauliList.from_list(["IIIXIZ", "YYXXII"])), 2)

    @ddt.idata(single_cases_list())
    def test_pickle(self, qubit_sparse_pauli_list):
        self.assertEqual(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertIsNot(qubit_sparse_pauli_list, copy.copy(qubit_sparse_pauli_list))
        self.assertEqual(qubit_sparse_pauli_list, copy.deepcopy(qubit_sparse_pauli_list))
        self.assertEqual(
            qubit_sparse_pauli_list, pickle.loads(pickle.dumps(qubit_sparse_pauli_list))
        )

    @ddt.data(
        PhasedQubitSparsePauliList.empty(0),
        PhasedQubitSparsePauliList.empty(1),
        PhasedQubitSparsePauliList.empty(10),
        PhasedQubitSparsePauliList.from_label("IIXIZI"),
        PhasedQubitSparsePauliList.from_label("X"),
        PhasedQubitSparsePauliList(Pauli("iXY")),
        PhasedQubitSparsePauliList.from_list(["YIXZII"]),
        PhasedQubitSparsePauliList.from_list(["YIXZII", "ZZYYXX"]),
        PhasedQubitSparsePauliList.from_list(["IIIIII", "ZZYYXX"]),
        PhasedQubitSparsePauliList.from_list(["IIIIII", "ZZYYXX"]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("PhasedQubitSparsePauliList", repr(data))

    @ddt.idata(single_cases_list())
    def test_copy(self, phased_qubit_sparse_pauli_list):
        self.assertEqual(phased_qubit_sparse_pauli_list, phased_qubit_sparse_pauli_list.copy())
        self.assertIsNot(phased_qubit_sparse_pauli_list, phased_qubit_sparse_pauli_list.copy())

    def test_equality(self):
        sparse_data = [(0, "XZ", (1, 0)), (1, "XYY", (3, 1, 0))]
        pauli_list = PhasedQubitSparsePauliList.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(pauli_list, pauli_list.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(pauli_list, pauli_list)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(pauli_list, None)

        # Difference in qubit count.
        self.assertNotEqual(
            pauli_list,
            PhasedQubitSparsePauliList.from_sparse_list(
                sparse_data, num_qubits=pauli_list.num_qubits + 1
            ),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauliList.empty(2), PhasedQubitSparsePauliList.empty(3)
        )

        # Difference in bit terms.
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            PhasedQubitSparsePauliList.from_list(["IIYZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            PhasedQubitSparsePauliList.from_list(["IIXZI", "XXYYY"]),
        )

        # Difference in indices.
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_list(["IIXZI", "XXYYZ"]),
            PhasedQubitSparsePauliList.from_list(["IXIZI", "XXYYZ"]),
        )
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_list(["IIXZI", "XIYYZ"]),
            PhasedQubitSparsePauliList.from_list(["IIXZI", "IXYYZ"]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZ", (0, 1)), (0, "XX", (2, 3))], num_qubits=5
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZX", (0, 1, 2)), (0, "X", (3,))], num_qubits=5
            ),
        )

        # Difference in phase.
        self.assertNotEqual(
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZ", (0, 1)), (0, "XX", (2, 3))], num_qubits=5
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZ", (0, 1)), (2, "XX", (2, 3))], num_qubits=5
            ),
        )

        # Same phase mod 4.
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(
                [(0, "XZ", (0, 1)), (2, "XX", (2, 3))], num_qubits=5
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                [(4, "XZ", (0, 1)), (10, "XX", (2, 3))], num_qubits=5
            ),
        )

    @ddt.idata(single_cases_list())
    def test_clear(self, pauli_list):
        num_qubits = pauli_list.num_qubits
        pauli_list.clear()
        self.assertEqual(pauli_list, PhasedQubitSparsePauliList.empty(num_qubits))

    def test_apply_layout_list(self):
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(5).apply_layout([4, 3, 2, 1, 0]),
            PhasedQubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(3).apply_layout([0, 2, 1], 8),
            PhasedQubitSparsePauliList.empty(8),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(2).apply_layout([1, 0]),
            PhasedQubitSparsePauliList.empty(2),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(3).apply_layout([100, 10_000, 3], 100_000_000),
            PhasedQubitSparsePauliList.empty(100_000_000),
        )

        terms = [
            (0, "ZYX", (4, 2, 1)),
            (1, "", ()),
            (2, "XXYYZZ", (10, 8, 6, 4, 2, 0)),
        ]

        def map_indices(terms, layout):
            return [
                (phase, terms, tuple(layout[bit] for bit in bits)) for phase, terms, bits in terms
            ]

        identity = list(range(12))
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                identity
            ),
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12),
        )
        # We've already tested elsewhere that `PhasedQubitSparsePauliList.from_sparse_list` produces
        # termwise sorted indices, so these tests also ensure `apply_layout` is maintaining that
        # invariant.
        backwards = list(range(12))[::-1]
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                backwards
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                map_indices(terms, backwards), num_qubits=12
            ),
        )
        shuffled = [4, 7, 1, 10, 0, 11, 3, 2, 8, 5, 6, 9]
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                shuffled
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                map_indices(terms, shuffled), num_qubits=12
            ),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                shuffled, 100
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                map_indices(terms, shuffled), num_qubits=100
            ),
        )
        expanded = [78, 69, 82, 68, 32, 97, 108, 101, 114, 116, 33]
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=11).apply_layout(
                expanded, 120
            ),
            PhasedQubitSparsePauliList.from_sparse_list(
                map_indices(terms, expanded), num_qubits=120
            ),
        )

    def test_apply_layout_transpiled(self):
        base = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "ZYX", (4, 2, 1)),
                (1, "", ()),
                (2, "XXY", (3, 2, 0)),
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
            PhasedQubitSparsePauliList.empty(0).apply_layout(None),
            PhasedQubitSparsePauliList.empty(0),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(0).apply_layout(None, 3),
            PhasedQubitSparsePauliList.empty(3),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(5).apply_layout(None),
            PhasedQubitSparsePauliList.empty(5),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(3).apply_layout(None, 8),
            PhasedQubitSparsePauliList.empty(8),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(0).apply_layout(None),
            PhasedQubitSparsePauliList.empty(0),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(0).apply_layout(None, 8),
            PhasedQubitSparsePauliList.empty(8),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(2).apply_layout(None),
            PhasedQubitSparsePauliList.empty(2),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.empty(3).apply_layout(None, 100_000_000),
            PhasedQubitSparsePauliList.empty(100_000_000),
        )

        terms = [
            (0, "ZYX", (2, 1, 0)),
            (1, "", ()),
            (3, "XXYYZZ", (10, 8, 6, 4, 2, 0)),
        ]
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(None),
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12),
        )
        self.assertEqual(
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=12).apply_layout(
                None, num_qubits=200
            ),
            PhasedQubitSparsePauliList.from_sparse_list(terms, num_qubits=200),
        )

    def test_apply_layout_failures(self):
        obs = PhasedQubitSparsePauliList.from_list(["IIYI", "IIIX"])
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
        self.assertEqual(list(PhasedQubitSparsePauliList.empty(5)), [])
        self.assertEqual(tuple(PhasedQubitSparsePauliList.empty(0)), ())

        pauli_list = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "XYY", (4, 2, 1)),
                (2, "", ()),
                (0, "ZZ", (3, 0)),
                (3, "XX", (2, 1)),
                (0, "YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = PhasedQubitSparsePauli.Pauli
        expected = [
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            PhasedQubitSparsePauli.from_raw_parts(5, [], [], 2),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3]),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2], 3),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(list(pauli_list), expected)

    def test_indexing(self):
        pauli_list = PhasedQubitSparsePauliList.from_sparse_list(
            [
                (0, "XYY", (4, 2, 1)),
                (1, "", ()),
                (1, "ZZ", (3, 0)),
                (2, "XX", (2, 1)),
                (0, "YZ", (4, 1)),
            ],
            num_qubits=5,
        )
        pauli = PhasedQubitSparsePauli.Pauli
        expected = [
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Y, pauli.Y, pauli.X], [1, 2, 4]),
            PhasedQubitSparsePauli.from_raw_parts(5, [], [], 1),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Z], [0, 3], 1),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.X, pauli.X], [1, 2], 2),
            PhasedQubitSparsePauli.from_raw_parts(5, [pauli.Z, pauli.Y], [1, 4]),
        ]
        self.assertEqual(pauli_list[0], expected[0])
        self.assertEqual(pauli_list[-2], expected[-2])
        self.assertEqual(pauli_list[2:4], PhasedQubitSparsePauliList(expected[2:4]))
        self.assertEqual(pauli_list[1::2], PhasedQubitSparsePauliList(expected[1::2]))
        self.assertEqual(pauli_list[:], PhasedQubitSparsePauliList(expected))
        self.assertEqual(pauli_list[-1:-4:-1], PhasedQubitSparsePauliList(expected[-1:-4:-1]))

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="empty"):
            pauli_list = PhasedQubitSparsePauliList.empty(100)
            expected = []
            self.assertEqual(expected, pauli_list.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            pauli_list = PhasedQubitSparsePauliList(Pauli("-iIXYZ"))
            expected = [(1, "ZYX", [0, 1, 2])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )

        with self.subTest(msg="multiple"):
            pauli_list = PhasedQubitSparsePauliList.from_list(["XXIZ", "YYIZ"])
            expected = [(0, "XXZ", [3, 2, 0]), (0, "ZYY", [0, 2, 3])]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_list.to_sparse_list()),
            )

    def test_to_pauli_list(self):
        pauli_strings = ["XIZIY", "iIIZIY", "-iZIYII", "-IIZII"]
        pauli_list = PauliList(pauli_strings)
        phased_qubit_sparse_pauli_list = PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(
            [PhasedQubitSparsePauli(x) for x in pauli_list]
        )
        self.assertEqual(pauli_list, phased_qubit_sparse_pauli_list.to_pauli_list())

        # single element
        pauli_strings = ["XIZIY"]
        pauli_list = PauliList(pauli_strings)
        phased_qubit_sparse_pauli_list = PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(
            [PhasedQubitSparsePauli(x) for x in pauli_list]
        )
        self.assertEqual(pauli_list, phased_qubit_sparse_pauli_list.to_pauli_list())

        # single element with phase
        pauli_strings = ["iXIZIY"]
        pauli_list = PauliList(pauli_strings)
        phased_qubit_sparse_pauli_list = PhasedQubitSparsePauliList.from_phased_qubit_sparse_paulis(
            [PhasedQubitSparsePauli(x) for x in pauli_list]
        )
        self.assertEqual(pauli_list, phased_qubit_sparse_pauli_list.to_pauli_list())


def canonicalize_term(phase, pauli, indices):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (phase, sorted_paulis, np.asarray(indices)[idcs].tolist())


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
