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
from qiskit.quantum_info import QubitSparsePauli, QubitSparsePauliList, PauliLindbladMap
from qiskit.transpiler import Target

from test import QiskitTestCase


def single_cases():
    return [
        PauliLindbladMap.identity(0),
        PauliLindbladMap.identity(10),
        PauliLindbladMap.from_list([("YIXZII", -0.25), ("ZZYYXX", 0.25)]),
        # Includes a duplicate entry.
        PauliLindbladMap.from_list([("IXZ", -0.25), ("ZZI", 0.25), ("IXZ", 0.75)]),
    ]


@ddt.ddt
class TestPauliLindbladMap(QiskitTestCase):

    def test_default_constructor_list(self):
        data = [("IXIIZ", 0.5), ("XIXII", 1.0), ("IIXYI", -0.75)]
        self.assertEqual(PauliLindbladMap(data), PauliLindbladMap.from_list(data))
        self.assertEqual(PauliLindbladMap(data, num_qubits=5), PauliLindbladMap.from_list(data))
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PauliLindbladMap(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            PauliLindbladMap(data, num_qubits=6)
        self.assertEqual(
            PauliLindbladMap([], num_qubits=5), PauliLindbladMap.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3), 0.5), ("XY", (2, 4), 1.0), ("ZY", (2, 1), -0.75)]
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=5),
            PauliLindbladMap.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            PauliLindbladMap(data, num_qubits=10),
            PauliLindbladMap.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            PauliLindbladMap(data)
        self.assertEqual(
            PauliLindbladMap([], num_qubits=5), PauliLindbladMap.from_sparse_list([], num_qubits=5)
        )

    def test_default_constructor_copy(self):
        base = PauliLindbladMap.from_list([("IXIZIY", 1.0), ("XYZIII", -1.0)])
        copied = PauliLindbladMap(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            PauliLindbladMap(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = PauliLindbladMap.from_list([("IIZXII", 2)])
        self.assertEqual(PauliLindbladMap(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = PauliLindbladMap.from_list([("IIZXII", 2), ("IIIIII", 0.5)])
        terms = [expected[0], expected[1]]
        self.assertEqual(PauliLindbladMap(list(terms)), expected)
        self.assertEqual(PauliLindbladMap(tuple(terms)), expected)
        self.assertEqual(PauliLindbladMap(term for term in terms), expected)

    def test_from_list(self):
        label = "IXYIZZY"
        self.assertEqual(
            PauliLindbladMap.from_list([(label, 1.0)]),
            PauliLindbladMap.from_components([1.0], QubitSparsePauliList.from_label(label)),
        )
        self.assertEqual(
            PauliLindbladMap.from_list([(label, 1.0)], num_qubits=len(label)),
            PauliLindbladMap.from_components([1.0], QubitSparsePauliList.from_label(label)),
        )

        self.assertEqual(
            PauliLindbladMap.from_list([("IIIXZI", 1.0), ("XXIIII", -0.5)]),
            PauliLindbladMap.from_components(
                [1.0, -0.5], QubitSparsePauliList.from_list(["IIIXZI", "XXIIII"])
            ),
        )

        self.assertEqual(PauliLindbladMap.from_list([], num_qubits=5), PauliLindbladMap.identity(5))
        self.assertEqual(PauliLindbladMap.from_list([], num_qubits=0), PauliLindbladMap.identity(0))

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PauliLindbladMap.from_list([("XZIIZY", 0.5), ("I+-$%I", 1.0)])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            PauliLindbladMap.from_list([("🐍", 0.5)])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IIXI", 1.0)])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("II", 1.0)])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IXI", 1.0)], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            PauliLindbladMap.from_list([("IIZ", 0.5), ("IXI", 1.0)], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PauliLindbladMap.from_list([])

    def test_from_components(self):
        self.assertEqual(
            PauliLindbladMap.from_components(
                [0.5, -0.25, 1.0],
                QubitSparsePauliList.from_sparse_list(
                    [
                        (
                            "XY",
                            (0, 1),
                        ),
                        (
                            "XX",
                            (1, 3),
                        ),
                        ("YYZ", (0, 2, 4)),
                    ],
                    num_qubits=5,
                ),
            ),
            PauliLindbladMap.from_list([("IIIYX", 0.5), ("IXIXI", -0.25), ("ZIYIY", 1.0)]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = PauliLindbladMap.from_components(
            [1.5, -0.5],
            QubitSparsePauliList.from_sparse_list(
                [
                    ("XYZ", (2, 1, 0)),
                    ("XYY", (2, 0, 1)),
                ],
                num_qubits=3,
            ),
        )
        self.assertEqual(from_unsorted, PauliLindbladMap.from_list([("XYZ", 1.5), ("XYY", -0.5)]))
        np.testing.assert_equal(from_unsorted[0].indices, np.array([0, 1, 2], dtype=np.uint32))
        np.testing.assert_equal(from_unsorted[1].indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = PauliLindbladMap.from_components(
            [1.0, -0.5],
            QubitSparsePauliList.from_sparse_list(
                [
                    ("ZXI", (0, 1, 2)),
                    ("XYIII", (0, 1, 2, 3, 8)),
                ],
                num_qubits=10,
            ),
        )
        self.assertEqual(
            explicit_identity,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (1, 0), 1.0), ("YX", (1, 0), -0.5)], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity[0].indices, np.array([0, 1], dtype=np.uint32))
        np.testing.assert_equal(explicit_identity[1].indices, np.array([0, 1], dtype=np.uint32))

        self.assertEqual(
            PauliLindbladMap.from_components([], QubitSparsePauliList.empty(1_000_000)),
            PauliLindbladMap.identity(1_000_000),
        )
        self.assertEqual(
            PauliLindbladMap.from_components([], QubitSparsePauliList.empty(0)),
            PauliLindbladMap.identity(0),
        )

    def test_from_components_failures(self):
        with self.assertRaisesRegex(
            ValueError, r"`rates` \(1\) must be the same length as `qubit_sparse_pauli_list` \(2\)"
        ):
            PauliLindbladMap.from_components([1.0], QubitSparsePauliList(["II", "XX"]))

    def test_from_sparse_list(self):
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(
                [
                    ("XY", (0, 1), 0.5),
                    ("XX", (1, 3), -0.25),
                    ("YYZ", (0, 2, 4), 1.0),
                ],
                num_qubits=5,
            ),
            PauliLindbladMap.from_list([("IIIYX", 0.5), ("IXIXI", -0.25), ("ZIYIY", 1.0)]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = PauliLindbladMap.from_sparse_list(
            [
                ("XYZ", (2, 1, 0), 1.5),
                ("XYY", (2, 0, 1), -0.5),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, PauliLindbladMap.from_list([("XYZ", 1.5), ("XYY", -0.5)]))
        np.testing.assert_equal(from_unsorted[0].indices, np.array([0, 1, 2], dtype=np.uint32))
        np.testing.assert_equal(from_unsorted[1].indices, np.array([0, 1, 2], dtype=np.uint32))

        # Explicit identities should still work, just be skipped over.
        explicit_identity = PauliLindbladMap.from_sparse_list(
            [
                ("ZXI", (0, 1, 2), 1.0),
                ("XYIII", (0, 1, 2, 3, 8), -0.5),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            PauliLindbladMap.from_sparse_list(
                [("XZ", (1, 0), 1.0), ("YX", (1, 0), -0.5)], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity[0].indices, np.array([0, 1], dtype=np.uint32))
        np.testing.assert_equal(explicit_identity[1].indices, np.array([0, 1], dtype=np.uint32))

        self.assertEqual(
            PauliLindbladMap.from_sparse_list([], num_qubits=1_000_000),
            PauliLindbladMap.identity(1_000_000),
        )
        self.assertEqual(
            PauliLindbladMap.from_sparse_list([], num_qubits=0),
            PauliLindbladMap.identity(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            PauliLindbladMap.from_sparse_list(
                [("XZZY", (5, 3, 1, 0), 0.5), ("+$", (2, 1), 1.0)], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            PauliLindbladMap.from_sparse_list([("\xff", (1,), 0.5)], num_qubits=5)
        with self.assertRaises(ValueError):
            PauliLindbladMap.from_sparse_list([("\xff", (1, 2), 0.5)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PauliLindbladMap.from_sparse_list([("XZ", (0,), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            PauliLindbladMap.from_sparse_list([("XZ", (0, 1, 2), 1.0)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            PauliLindbladMap.from_sparse_list([("XZY", (0, 1, 3), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            PauliLindbladMap.from_sparse_list([("XZY", (0, 1, 4), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            PauliLindbladMap.from_sparse_list([("XXI", (0, 1, 3), 1.0)], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PauliLindbladMap.from_sparse_list([("XZ", (3, 3), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            PauliLindbladMap.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3), 1.0)], num_qubits=5)

    def test_from_terms(self):
        self.assertEqual(
            PauliLindbladMap.from_terms([], num_qubits=5), PauliLindbladMap.identity(5)
        )
        self.assertEqual(
            PauliLindbladMap.from_terms((), num_qubits=0), PauliLindbladMap.identity(0)
        )
        self.assertEqual(
            PauliLindbladMap.from_terms((None for _ in []), num_qubits=3),
            PauliLindbladMap.identity(3),
        )

        expected = PauliLindbladMap.from_sparse_list(
            [
                ("XYZ", (4, 2, 1), 1),
                ("XXYY", (8, 5, 3, 2), 0.5),
                ("ZZ", (5, 0), 2.0),
            ],
            num_qubits=10,
        )
        self.assertEqual(PauliLindbladMap.from_terms(list(expected)), expected)
        self.assertEqual(PauliLindbladMap.from_terms(tuple(expected)), expected)
        self.assertEqual(PauliLindbladMap.from_terms(term for term in expected), expected)
        self.assertEqual(
            PauliLindbladMap.from_terms(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_terms_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            PauliLindbladMap.from_terms([])

        left, right = (
            PauliLindbladMap([("IIXYI", 1.0)])[0],
            PauliLindbladMap([("IIIIIIIIX", 1.0)])[0],
        )
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PauliLindbladMap.from_terms([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            PauliLindbladMap.from_terms([left], num_qubits=100)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            PauliLindbladMap([("IIXIZ", 1.0), ("IZ", (2, 3), -1.0)], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(PauliLindbladMap.identity(0).num_qubits, 0)
        self.assertEqual(PauliLindbladMap.identity(10).num_qubits, 10)

    def test_num_terms(self):
        self.assertEqual(PauliLindbladMap.identity(0).num_terms, 0)
        self.assertEqual(PauliLindbladMap.identity(10).num_terms, 0)
        self.assertEqual(
            PauliLindbladMap.from_list([("IIIXIZ", 1.0), ("YYXXII", 0.5)]).num_terms, 2
        )

    def test_identity(self):
        identity_5 = PauliLindbladMap.identity(5)
        self.assertEqual(identity_5.num_qubits, 5)
        self.assertEqual(identity_5.gamma(), 1.0)
        self.assertEqual(identity_5.num_terms, 0.0)
        np.testing.assert_equal(identity_5.rates, np.array([], dtype=float))
        np.testing.assert_equal(identity_5.probabilities(), np.array([], dtype=float))

        identity_0 = PauliLindbladMap.identity(0)
        self.assertEqual(identity_0.num_qubits, 0)
        self.assertEqual(identity_0.gamma(), 1.0)
        self.assertEqual(identity_0.num_terms, 0.0)
        np.testing.assert_equal(identity_0.rates, np.array([], dtype=float))
        np.testing.assert_equal(identity_0.probabilities(), np.array([], dtype=float))

    def test_len(self):
        self.assertEqual(len(PauliLindbladMap.identity(0)), 0)
        self.assertEqual(len(PauliLindbladMap.identity(10)), 0)
        self.assertEqual(len(PauliLindbladMap.from_list([("IIIXIZ", 1.0), ("YYXXII", 0.5)])), 2)

    @ddt.idata(single_cases())
    def test_pickle(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map, copy.copy(pauli_lindblad_map))
        self.assertIsNot(pauli_lindblad_map, copy.copy(pauli_lindblad_map))
        self.assertEqual(pauli_lindblad_map, copy.deepcopy(pauli_lindblad_map))
        self.assertEqual(pauli_lindblad_map, pickle.loads(pickle.dumps(pauli_lindblad_map)))

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        PauliLindbladMap.identity(0),
        PauliLindbladMap.identity(1),
        PauliLindbladMap.identity(10),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
        PauliLindbladMap.from_list([("YIXZII", -0.25), ("ZZYYXX", 0.25)]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("PauliLindbladMap", repr(data))

    @ddt.idata(single_cases())
    def test_copy(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map, pauli_lindblad_map.copy())
        self.assertIsNot(pauli_lindblad_map, pauli_lindblad_map.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0), 0.5), ("XYY", (3, 1, 0), -0.25)]
        op = PauliLindbladMap.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(op, op.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(op, op)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(op, None)

        # No costly automatic simplification (mathematically, these operators _are_ the same).
        self.assertNotEqual(
            PauliLindbladMap.from_list([("X", 2.0), ("X", -1.0)]), PauliLindbladMap([("X", 1.0)])
        )

        # Difference in qubit count.
        self.assertNotEqual(
            op, PauliLindbladMap.from_sparse_list(sparse_data, num_qubits=op.num_qubits + 1)
        )
        self.assertNotEqual(PauliLindbladMap.identity(2), PauliLindbladMap.identity(3))

        # Difference in rates.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", 0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", -1.0), ("XXYYZ", -0.5)]),
        )

        # Difference in bit terms.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIYZI", 1.0), ("XXYYZ", -0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYY", -0.5)]),
        )

        # Difference in indices.
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XXYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IXIZI", 1.0), ("XXYYZ", -0.5)]),
        )
        self.assertNotEqual(
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("XIYYZ", -0.5)]),
            PauliLindbladMap.from_list([("IIXZI", 1.0), ("IXYYZ", -0.5)]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            PauliLindbladMap.from_sparse_list(
                [("XZ", (0, 1), 1.5), ("XX", (2, 3), -0.5)], num_qubits=5
            ),
            PauliLindbladMap.from_sparse_list(
                [("XZX", (0, 1, 2), 1.5), ("X", (3,), -0.5)], num_qubits=5
            ),
        )

    def test_attributes_immutable(self):
        pauli_lindblad_map = PauliLindbladMap.from_list([("XZY", 1.5), ("XXY", -0.5)])
        with self.assertRaisesRegex(AttributeError, "attribute 'rates'"):
            pauli_lindblad_map.rates = 1.0
        with self.assertRaisesRegex(ValueError, "assignment destination is read-only"):
            pauli_lindblad_map.rates[0] = 1.0

    def test_with_history(self):
        """Test that generators() method returns the same result as
        get_qubit_sparse_pauli_list_copy()."""
        pauli_lindblad_map = PauliLindbladMap.from_list([("IIXIZ", 2), ("IIZIX", 3)])
        generators = pauli_lindblad_map.generators()
        expected = pauli_lindblad_map.get_qubit_sparse_pauli_list_copy()
        self.assertEqual(generators, expected)
        self.assertIsInstance(generators, QubitSparsePauliList)

    @ddt.idata(single_cases())
    def test_clear(self, pauli_lindblad_map):
        num_qubits = pauli_lindblad_map.num_qubits
        pauli_lindblad_map.clear()
        self.assertEqual(pauli_lindblad_map, PauliLindbladMap.identity(num_qubits))

    def test_iteration(self):
        self.assertEqual(list(PauliLindbladMap.identity(5)), [])
        self.assertEqual(tuple(PauliLindbladMap.identity(0)), ())

        pauli_lindblad_map = PauliLindbladMap.from_sparse_list(
            [
                ("XYY", (4, 2, 1), 2),
                ("", (), 0.5),
                ("ZZ", (3, 0), -0.25),
                ("XX", (2, 1), 1.0),
                ("YZ", (4, 1), 1),
            ],
            num_qubits=5,
        )
        expected = [
            PauliLindbladMap.GeneratorTerm(2, QubitSparsePauli(("YYX", [1, 2, 4]), 5)),
            PauliLindbladMap.GeneratorTerm(0.5, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(-0.25, QubitSparsePauli(("ZZ", [0, 3]), 5)),
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("XX", [1, 2]), 5)),
            PauliLindbladMap.GeneratorTerm(1, QubitSparsePauli(("ZY", [1, 4]), 5)),
        ]
        self.assertEqual(list(pauli_lindblad_map), expected)

    def test_apply_layout_list(self):
        self.assertEqual(
            PauliLindbladMap.identity(5).apply_layout([4, 3, 2, 1, 0]), PauliLindbladMap.identity(5)
        )
        self.assertEqual(
            PauliLindbladMap.identity(3).apply_layout([0, 2, 1], 8), PauliLindbladMap.identity(8)
        )
        self.assertEqual(
            PauliLindbladMap.identity(2).apply_layout([1, 0]), PauliLindbladMap.identity(2)
        )
        self.assertEqual(
            PauliLindbladMap.identity(3).apply_layout([100, 10_000, 3], 100_000_000),
            PauliLindbladMap.identity(100_000_000),
        )

        terms = [
            ("ZYX", (4, 2, 1), 1),
            ("", (), -0.5),
            ("XXYYZZ", (10, 8, 6, 4, 2, 0), 2.0),
        ]

        def map_indices(terms, layout):
            return [
                (terms, tuple(layout[bit] for bit in bits), coeff) for terms, bits, coeff in terms
            ]

        identity = list(range(12))
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(identity),
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12),
        )
        # We've already tested elsewhere that `PauliLindbladMap.from_sparse_list` produces termwise
        # sorted indices, so these tests also ensure `apply_layout` is maintaining that invariant.
        backwards = list(range(12))[::-1]
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(backwards),
            PauliLindbladMap.from_sparse_list(map_indices(terms, backwards), num_qubits=12),
        )
        shuffled = [4, 7, 1, 10, 0, 11, 3, 2, 8, 5, 6, 9]
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled),
            PauliLindbladMap.from_sparse_list(map_indices(terms, shuffled), num_qubits=12),
        )
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled, 100),
            PauliLindbladMap.from_sparse_list(map_indices(terms, shuffled), num_qubits=100),
        )
        expanded = [78, 69, 82, 68, 32, 97, 108, 101, 114, 116, 33]
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=11).apply_layout(expanded, 120),
            PauliLindbladMap.from_sparse_list(map_indices(terms, expanded), num_qubits=120),
        )

    def test_apply_layout_transpiled(self):
        base = PauliLindbladMap.from_sparse_list(
            [
                ("ZYX", (4, 2, 1), 1),
                ("", (), -0.5),
                ("XXY", (3, 2, 0), 2.0),
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
            PauliLindbladMap.identity(0).apply_layout(None), PauliLindbladMap.identity(0)
        )
        self.assertEqual(
            PauliLindbladMap.identity(0).apply_layout(None, 3), PauliLindbladMap.identity(3)
        )
        self.assertEqual(
            PauliLindbladMap.identity(5).apply_layout(None), PauliLindbladMap.identity(5)
        )
        self.assertEqual(
            PauliLindbladMap.identity(3).apply_layout(None, 8), PauliLindbladMap.identity(8)
        )
        self.assertEqual(
            PauliLindbladMap.identity(0).apply_layout(None), PauliLindbladMap.identity(0)
        )
        self.assertEqual(
            PauliLindbladMap.identity(0).apply_layout(None, 8), PauliLindbladMap.identity(8)
        )
        self.assertEqual(
            PauliLindbladMap.identity(2).apply_layout(None), PauliLindbladMap.identity(2)
        )
        self.assertEqual(
            PauliLindbladMap.identity(3).apply_layout(None, 100_000_000),
            PauliLindbladMap.identity(100_000_000),
        )

        terms = [
            ("ZYX", (2, 1, 0), 1),
            ("", (), -0.5),
            ("XXYYZZ", (10, 8, 6, 4, 2, 0), 2.0),
        ]
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(None),
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12),
        )
        self.assertEqual(
            PauliLindbladMap.from_sparse_list(terms, num_qubits=12).apply_layout(
                None, num_qubits=200
            ),
            PauliLindbladMap.from_sparse_list(terms, num_qubits=200),
        )

    def test_apply_layout_failures(self):
        obs = PauliLindbladMap.from_list([("IIYI", 2.0), ("IIIX", -1)])
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

    def test_indexing(self):
        pauli_lindblad_map = PauliLindbladMap.from_sparse_list(
            [
                ("XYY", (4, 2, 1), 2),
                ("", (), 0.5),
                ("ZZ", (3, 0), -0.25),
                ("XX", (2, 1), 1.0),
                ("YZ", (4, 1), 1),
            ],
            num_qubits=5,
        )
        expected = [
            PauliLindbladMap.GeneratorTerm(2, QubitSparsePauli(("YYX", [1, 2, 4]), 5)),
            PauliLindbladMap.GeneratorTerm(0.5, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(-0.25, QubitSparsePauli(("ZZ", [0, 3]), 5)),
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("XX", [1, 2]), 5)),
            PauliLindbladMap.GeneratorTerm(1, QubitSparsePauli(("ZY", [1, 4]), 5)),
        ]
        self.assertEqual(pauli_lindblad_map[0], expected[0])
        self.assertEqual(pauli_lindblad_map[-2], expected[-2])
        self.assertEqual(pauli_lindblad_map[2:4], PauliLindbladMap(expected[2:4]))
        self.assertEqual(pauli_lindblad_map[1::2], PauliLindbladMap(expected[1::2]))
        self.assertEqual(pauli_lindblad_map[:], PauliLindbladMap(expected))
        self.assertEqual(pauli_lindblad_map[-1:-4:-1], PauliLindbladMap(expected[-1:-4:-1]))

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_repr(self, pauli_lindblad_map):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        term = pauli_lindblad_map[0]
        self.assertIsInstance(repr(term), str)
        self.assertIn("PauliLindbladMap.GeneratorTerm", repr(term))

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_to_pauli_lindblad_map(self, pauli_lindblad_map):
        self.assertEqual(pauli_lindblad_map[0].to_pauli_lindblad_map(), pauli_lindblad_map)
        self.assertIsNot(pauli_lindblad_map[0].to_pauli_lindblad_map(), pauli_lindblad_map)

    def test_term_equality(self):
        self.assertEqual(
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 5)),
        )
        self.assertNotEqual(
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 8)),
        )
        self.assertNotEqual(
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(2.0, QubitSparsePauli(("", []), 5)),
        )
        self.assertNotEqual(
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("", []), 5)),
            PauliLindbladMap.GeneratorTerm(-1, QubitSparsePauli(("", []), 8)),
        )

        pauli_lindblad_map = PauliLindbladMap.from_list(
            [
                ("IIXIZ", 2),
                ("IIZIX", 2),
                ("XXIII", -1.5),
                ("XYIII", -1.5),
                ("IYIYI", 0.5),
                ("IIYIY", 0.5),
            ]
        )
        self.assertEqual(pauli_lindblad_map[0], pauli_lindblad_map[0])
        self.assertEqual(pauli_lindblad_map[1], pauli_lindblad_map[1])
        self.assertNotEqual(pauli_lindblad_map[0], pauli_lindblad_map[1])
        self.assertEqual(pauli_lindblad_map[2], pauli_lindblad_map[2])
        self.assertEqual(pauli_lindblad_map[3], pauli_lindblad_map[3])
        self.assertNotEqual(pauli_lindblad_map[2], pauli_lindblad_map[3])
        self.assertEqual(pauli_lindblad_map[4], pauli_lindblad_map[4])
        self.assertEqual(pauli_lindblad_map[5], pauli_lindblad_map[5])
        self.assertNotEqual(pauli_lindblad_map[4], pauli_lindblad_map[5])

    @ddt.data(
        PauliLindbladMap.from_sparse_list([("YXZ", [2, 3, 5], -0.25)], num_qubits=6),
        PauliLindbladMap.from_list([("YIXZII", -0.25)]),
    )
    def test_term_pickle(self, pauli_lindblad_map):
        term = pauli_lindblad_map[0]
        self.assertEqual(pickle.loads(pickle.dumps(term)), term)
        self.assertEqual(copy.copy(term), term)
        self.assertEqual(copy.deepcopy(term), term)

    def test_term_attributes(self):
        term = PauliLindbladMap([("IIXIIXZ", 5.0)])[0]
        self.assertEqual(term.num_qubits, 7)
        self.assertEqual(term.rate, 5.0)
        np.testing.assert_equal(
            term.paulis,
            np.array(
                [
                    QubitSparsePauli.Pauli.Z,
                    QubitSparsePauli.Pauli.X,
                    QubitSparsePauli.Pauli.X,
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(term.indices, np.array([0, 1, 4], dtype=np.uintp))

        term = PauliLindbladMap.from_list([("IIXYZ", 0.5)])[0]
        self.assertEqual(term.num_qubits, 5)
        self.assertEqual(term.rate, 0.5)
        self.assertEqual(
            list(term.paulis),
            [
                QubitSparsePauli.Pauli.Z,
                QubitSparsePauli.Pauli.Y,
                QubitSparsePauli.Pauli.X,
            ],
        )
        self.assertEqual(list(term.indices), [0, 1, 2])

        self.assertEqual(term.qubit_sparse_pauli, QubitSparsePauli.from_label("IIXYZ"))

    def test_term_new(self):
        expected = PauliLindbladMap([("IIIXXZIII", 1.0)])[0]

        self.assertEqual(
            PauliLindbladMap.GeneratorTerm(1.0, QubitSparsePauli(("ZXX", [3, 4, 5]), 9)),
            expected,
        )

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="identity"):
            pauli_lindblad_map = PauliLindbladMap.identity(100)
            expected = []
            self.assertEqual(expected, pauli_lindblad_map.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            pauli_lindblad_map = PauliLindbladMap([("IXYZ", 1.0)])
            expected = [("ZYX", [0, 1, 2], 1)]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_lindblad_map.to_sparse_list()),
            )

        with self.subTest(msg="multiple"):
            pauli_lindblad_map = PauliLindbladMap.from_list([("XXIZ", 0.5), ("YYIZ", -1)])
            expected = [("XXZ", [3, 2, 0], 0.5), ("ZYY", [0, 2, 3], -1)]
            self.assertEqual(
                canonicalize_sparse_list(expected),
                canonicalize_sparse_list(pauli_lindblad_map.to_sparse_list()),
            )

    def test_sparse_term_pauli_labels(self):
        """Test getting the bit labels of a SparseTerm."""

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", 1.0)])
        term = pauli_lindblad_map[0]
        indices = term.indices
        labels = term.pauli_labels()

        label_dict = dict(zip(indices, labels))
        expected = dict(enumerate("ZYXZYXZYX"))

        for i, label in expected.items():
            self.assertEqual(label, label_dict[i])

        reconstructed = PauliLindbladMap.from_sparse_list(
            [(labels, indices, 1)], pauli_lindblad_map.num_qubits
        )
        self.assertEqual(pauli_lindblad_map, reconstructed)

    def test_attributes(self):
        pauli_lindblad_map = PauliLindbladMap.from_components(
            [1.0, 2.0], QubitSparsePauliList(["II", "XX"])
        )
        self.assertEqual(
            pauli_lindblad_map.get_qubit_sparse_pauli_list_copy(),
            QubitSparsePauliList.from_list(["II", "XX"]),
        )

    def test_derived_properties(self):
        """Test whether gamma and probabilities are correctly calculated."""

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", 1.0)])
        w = 0.5 * (1 + np.exp(-2 * 1.0))
        self.assertTrue(np.allclose(w, pauli_lindblad_map.probabilities()[0]))
        self.assertTrue(np.allclose(1.0, pauli_lindblad_map.gamma()))

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", -1.0)])
        w = 0.5 * (1 + np.exp(-2 * -1.0))
        gamma = w + np.abs(1 - w)
        prob = w / gamma
        self.assertTrue(np.allclose(prob, pauli_lindblad_map.probabilities()[0]))
        self.assertTrue(np.allclose(gamma, pauli_lindblad_map.gamma()))

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", -0.5)])
        w = 0.5 * (1 + np.exp(-2 * -0.5))
        gamma = w + np.abs(1 - w)
        prob = w / gamma
        self.assertTrue(np.allclose(prob, pauli_lindblad_map.probabilities()[0]))
        self.assertTrue(np.allclose(gamma, pauli_lindblad_map.gamma()))

        pauli_lindblad_map = PauliLindbladMap(
            [("IXYZXYZXYZ", -1.0), ("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", -0.5)]
        )
        rates = np.array([-1.0, 1.0, -0.5])
        w = 0.5 * (1 + np.exp(-2 * rates))
        gammas = w + np.abs(1 - w)
        probs = w / gammas
        gamma = np.prod(gammas)
        self.assertTrue(np.allclose(probs, pauli_lindblad_map.probabilities()))
        self.assertTrue(np.allclose(gamma, pauli_lindblad_map.gamma()))

    def test_drop_paulis(self):
        """Test the `drop_paulis` method."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )
        self.assertEqual(pauli_map_in, pauli_map_in.drop_paulis([]))

        pauli_map_out = pauli_map_in.drop_paulis([0])
        expected = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYI", 0.5), ("ZIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.drop_paulis([0, 3])
        expected = PauliLindbladMap.from_list([("XIIZI", 2.0), ("IIIYI", 0.5), ("ZIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.drop_paulis([0, 4])
        expected = PauliLindbladMap.from_list([("IXIZI", 2.0), ("IIIYI", 0.5), ("IIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

    def test_drop_paulis_raises(self):
        """Test that `drop_paulis` raises."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )

        with self.assertRaisesRegex(
            ValueError, "cannot drop Paulis for index 5 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.drop_paulis([0, 5])

        with self.assertRaisesRegex(
            ValueError, "cannot drop Paulis for index 8 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.drop_paulis([0, 8])

    def test_keep_paulis(self):
        """Test the `keep_paulis` method."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )
        self.assertEqual(pauli_map_in, pauli_map_in.keep_paulis(range(5)))

        pauli_map_out = pauli_map_in.keep_paulis(range(1, 5))
        expected = PauliLindbladMap.from_list([("XXIZI", 2.0), ("IIIYI", 0.5), ("ZIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.keep_paulis([4, 1, 2])
        expected = PauliLindbladMap.from_list([("XIIZI", 2.0), ("IIIYI", 0.5), ("ZIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.keep_paulis([1, 2, 3])
        expected = PauliLindbladMap.from_list([("IXIZI", 2.0), ("IIIYI", 0.5), ("IIIXI", -0.25)])
        self.assertEqual(pauli_map_out, expected)

    def test_keep_paulis_raises(self):
        """Test that `keep_paulis` raises."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )

        with self.assertRaisesRegex(
            ValueError, "cannot keep Paulis for index 5 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.keep_paulis([5])

        with self.assertRaisesRegex(
            ValueError, "cannot keep Paulis for index 8 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.keep_paulis([0, 8])

    def test_drop_qubits(self):
        """Test the `drop_qubits` method."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )
        self.assertEqual(pauli_map_in, pauli_map_in.drop_qubits([]))

        pauli_map_out = pauli_map_in.drop_qubits([0])
        expected = PauliLindbladMap.from_list([("XXIZ", 2.0), ("IIIY", 0.5), ("ZIIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.drop_qubits([0, 3])
        expected = PauliLindbladMap.from_list([("XIZ", 2.0), ("IIY", 0.5), ("ZIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.drop_qubits([0, 4])
        expected = PauliLindbladMap.from_list([("XIZ", 2.0), ("IIY", 0.5), ("IIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.drop_qubits([0, 1, 3, 4])
        expected = PauliLindbladMap.from_list([("I", 2.0), ("I", 0.5), ("I", -0.25)])
        self.assertEqual(pauli_map_out, expected)

    def test_drop_qubits_raises(self):
        """Test that `drop_qubits` raises."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )

        with self.assertRaisesRegex(
            ValueError, "cannot drop qubits for index 5 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.drop_qubits([0, 5])

        with self.assertRaisesRegex(
            ValueError, "cannot drop qubits for index 8 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.drop_qubits([0, 8])

        with self.assertRaisesRegex(
            ValueError, "cannot drop every qubit in the given PauliLindbladMap"
        ):
            pauli_map_in.drop_qubits([0, 1, 2, 3, 4])

    def test_keep_qubits(self):
        """Test the `keep_qubits` method."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )
        self.assertEqual(pauli_map_in, pauli_map_in.keep_qubits(range(5)))

        pauli_map_out = pauli_map_in.keep_qubits(range(1, 5))
        expected = PauliLindbladMap.from_list([("XXIZ", 2.0), ("IIIY", 0.5), ("ZIIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.keep_qubits([4, 1, 2])
        expected = PauliLindbladMap.from_list([("XIZ", 2.0), ("IIY", 0.5), ("ZIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

        pauli_map_out = pauli_map_in.keep_qubits([1, 2, 3])
        expected = PauliLindbladMap.from_list([("XIZ", 2.0), ("IIY", 0.5), ("IIX", -0.25)])
        self.assertEqual(pauli_map_out, expected)

    def test_keep_qubits_raises(self):
        """Test that `keep_qubits` raises."""
        pauli_map_in = PauliLindbladMap.from_list(
            [("XXIZI", 2.0), ("IIIYZ", 0.5), ("ZIIXY", -0.25)]
        )

        with self.assertRaisesRegex(
            ValueError, "cannot keep qubits for index 5 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.keep_qubits([5])

        with self.assertRaisesRegex(
            ValueError, "cannot keep qubits for index 8 in a 5-qubit PauliLindbladMap"
        ):
            pauli_map_in.keep_qubits([0, 8])

        with self.assertRaisesRegex(
            ValueError, "cannot drop every qubit in the given PauliLindbladMap"
        ):
            pauli_map_in.keep_qubits([])

    def test_simplify(self):
        duplicate = PauliLindbladMap([("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", 2.0)])
        self.assertEqual(duplicate.simplify(), PauliLindbladMap([("IXYZXYZXYZ", 3.0)]))

        cancel = PauliLindbladMap([("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", -1.0)])
        self.assertEqual(cancel.simplify(), PauliLindbladMap.identity(10))

        drop_identities = PauliLindbladMap([("IXY", 1.0), ("III", -1.0)])
        self.assertEqual(drop_identities.simplify(), PauliLindbladMap([("IXY", 1.0)]))

        # note: some calls to simplify() are to enforce canonical ordering
        threshold = PauliLindbladMap([("X", 1e-9), ("Z", -2), ("Y", 1.0)]).simplify(1e-10)
        self.assertEqual(threshold.simplify(), PauliLindbladMap([("Y", 1.0), ("Z", -2)]).simplify())
        self.assertEqual(threshold.simplify(1e-10), threshold)
        self.assertEqual(threshold.simplify(1.1), PauliLindbladMap([("Z", -2)]))

    def test_scale_rates(self):
        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", 1.0)])
        self.assertEqual(
            pauli_lindblad_map.scale_rates(12.32), PauliLindbladMap([("IXYZXYZXYZ", 12.32)])
        )

        pauli_lindblad_map = PauliLindbladMap(
            [("IXYZXYZXYZ", -1.0), ("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", -0.5)]
        )
        self.assertEqual(
            pauli_lindblad_map.scale_rates(3.2),
            PauliLindbladMap([("IXYZXYZXYZ", -3.2), ("IXYZXYZXYZ", 3.2), ("IXYZXYZXYZ", -1.6)]),
        )

        self.assertEqual(
            PauliLindbladMap.identity(5).scale_rates(5.0), PauliLindbladMap.identity(5)
        )

    def test_inverse(self):

        pauli_lindblad_map = PauliLindbladMap([("IXYZXYZXYZ", 1.0)])
        self.assertEqual(pauli_lindblad_map.inverse(), PauliLindbladMap([("IXYZXYZXYZ", -1.0)]))

        pauli_lindblad_map = PauliLindbladMap(
            [("IXYZXYZXYZ", -1.0), ("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", -0.5)]
        )
        self.assertEqual(
            pauli_lindblad_map.inverse(),
            PauliLindbladMap([("IXYZXYZXYZ", 1.0), ("IXYZXYZXYZ", -1.0), ("IXYZXYZXYZ", 0.5)]),
        )

        self.assertEqual(PauliLindbladMap.identity(5).inverse(), PauliLindbladMap.identity(5))

    def test_compose(self):
        """Test compose method."""
        p0 = PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1)], 4)
        expected = PauliLindbladMap.from_sparse_list(
            [("XYZ", [3, 2, 1], 2.1), ("XYZ", [3, 2, 1], 2.1)], 4
        )
        self.assertEqual(p0.compose(p0), expected)

        # validate original object unchanged
        self.assertEqual(p0, PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1)], 4))

        p0 = PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1), ("Y", [0], 0.1)], 4)
        p1 = PauliLindbladMap.from_sparse_list([("X", [3], 0.2), ("Z", [1], 0.1)], 4)
        expected = PauliLindbladMap.from_sparse_list(
            [("XYZ", [3, 2, 1], 2.1), ("Y", [0], 0.1), ("X", [3], 0.2), ("Z", [1], 0.1)], 4
        )
        self.assertEqual(p0 @ p1, expected)

        # validate original objects unchanged
        self.assertEqual(
            p0, PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1), ("Y", [0], 0.1)], 4)
        )
        self.assertEqual(
            p1, PauliLindbladMap.from_sparse_list([("X", [3], 0.2), ("Z", [1], 0.1)], 4)
        )

        # test composition with identity map
        p0 = PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1)], 4)
        p1 = PauliLindbladMap.identity(4)
        self.assertEqual(p0 @ p1, p0)
        self.assertEqual(p1 @ p0, p0)

        p0 = PauliLindbladMap.identity(20)
        self.assertEqual(p0 @ p0, p0)

    def test_compose_errors(self):

        p0 = PauliLindbladMap.from_sparse_list([("XYZ", [3, 2, 1], 2.1)], 4)
        p1 = PauliLindbladMap.identity(3)

        with self.assertRaisesRegex(ValueError, r"mismatched numbers of qubits: 4, 3"):
            p0.compose(p1)

        with self.assertRaisesRegex(TypeError, r"unknown type for compose"):
            p0.compose(1.0)

    def test_pauli_fidelity(self):

        pauli_lindblad_map = PauliLindbladMap(
            [("XY", [0, 1], 1.23), ("Z", [1], -0.23), ("X", [2], 0.3)], num_qubits=4
        )
        self.assertEqual(pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("X", [0]), 4)), 1.0)
        np.testing.assert_array_max_ulp(
            pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("Y", [0]), 4)),
            np.exp(-2 * 1.23),
            maxulp=3,
        )
        np.testing.assert_array_max_ulp(
            pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("X", [1]), 4)),
            np.exp(-2 * 1.0),
            maxulp=3,
        )
        self.assertEqual(pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("Z", [3]), 4)), 1.0)
        np.testing.assert_array_max_ulp(
            pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("ZXY", [0, 1, 2]), 4)),
            np.exp(-2 * 0.07),
            maxulp=3,
        )

        self.assertEqual(
            PauliLindbladMap.identity(5).pauli_fidelity(QubitSparsePauli("IXXYZ")), 1.0
        )

    def test_pauli_fidelity_errors(self):

        pauli_lindblad_map = PauliLindbladMap(
            [("XY", [0, 1], 1.23), ("Z", [1], -0.23), ("X", [2], 0.3)], num_qubits=4
        )

        with self.assertRaisesRegex(ValueError, r"mismatched numbers of qubits: 5, 4"):
            pauli_lindblad_map.pauli_fidelity(QubitSparsePauli(("X", [0]), 5))

    def test_signed_sample(self):

        # test all negative rates
        pauli_lindblad_map = PauliLindbladMap([("X", -1.0), ("Y", -0.5)])
        probs = pauli_lindblad_map.probabilities()
        probs_dict = {
            "I": probs[0] * probs[1],
            "X": (1 - probs[0]) * probs[1],
            "Y": probs[0] * (1 - probs[1]),
            "Z": (1 - probs[0]) * (1 - probs[1]),
        }
        expected_signs = {"I": True, "X": False, "Y": False, "Z": True}

        num_samples = 10000
        signs, qubit_sparse_pauli_list = pauli_lindblad_map.signed_sample(num_samples, 12312)

        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        for sign, q in zip(signs, qubit_sparse_pauli_list):
            for symbol in counts:
                if q == QubitSparsePauli(symbol):
                    counts[symbol] += 1
                    self.assertEqual(expected_signs[symbol], sign)

        for symbol, count in counts.items():
            self.assertTrue(np.abs(count / num_samples - probs_dict[symbol]) < 1e-2)

        # test all positive rates
        pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", 0.5)])
        probs = pauli_lindblad_map.probabilities()
        probs_dict = {
            "I": probs[0] * probs[1],
            "X": (1 - probs[0]) * probs[1],
            "Y": probs[0] * (1 - probs[1]),
            "Z": (1 - probs[0]) * (1 - probs[1]),
        }
        expected_signs = {"I": True, "X": True, "Y": True, "Z": True}

        num_samples = 10000
        signs, qubit_sparse_pauli_list = pauli_lindblad_map.signed_sample(num_samples, 12312)

        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        for sign, q in zip(signs, qubit_sparse_pauli_list):
            for symbol in counts:
                if q == QubitSparsePauli(symbol):
                    counts[symbol] += 1
                    self.assertEqual(expected_signs[symbol], sign)
        for symbol, count in counts.items():
            self.assertTrue(np.abs(count / num_samples - probs_dict[symbol]) < 1e-2)

        # test mix of positive and negative rates
        pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -0.5)])
        probs = pauli_lindblad_map.probabilities()
        probs_dict = {
            "I": probs[0] * probs[1],
            "X": (1 - probs[0]) * probs[1],
            "Y": probs[0] * (1 - probs[1]),
            "Z": (1 - probs[0]) * (1 - probs[1]),
        }
        expected_signs = {"I": True, "X": True, "Y": False, "Z": False}

        num_samples = 10000
        signs, qubit_sparse_pauli_list = pauli_lindblad_map.signed_sample(num_samples, 12312)

        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        for sign, q in zip(signs, qubit_sparse_pauli_list):
            for symbol in counts:
                if q == QubitSparsePauli(symbol):
                    counts[symbol] += 1
                    self.assertEqual(expected_signs[symbol], sign)

        for symbol, count in counts.items():
            self.assertTrue(np.abs(count / num_samples - probs_dict[symbol]) < 1e-2)

        # test callable without seed
        signs, qubit_sparse_pauli_list = pauli_lindblad_map.signed_sample(5)
        self.assertTrue(isinstance(qubit_sparse_pauli_list, QubitSparsePauliList))
        self.assertEqual(len(qubit_sparse_pauli_list), 5)
        self.assertEqual(len(signs), 5)

    def _assert_parity_sample_results(
        self,
        plm_to_sample,
        expected_signs,
        num_samples=10000,
        seed=12312,
        scale=None,
        local_scale=None,
        plm_reference=None,
    ):
        if plm_reference is None:
            plm_reference = plm_to_sample

        generators = plm_reference.generators()
        probs = plm_reference.probabilities()
        probs_dict = {
            "I": probs[0] * probs[1],
            "X": (1 - probs[0]) * probs[1],
            "Y": probs[0] * (1 - probs[1]),
            "Z": (1 - probs[0]) * (1 - probs[1]),
        }

        signs, qubit_sparse_pauli_list, sampled_with_history, sampled_signs = (
            plm_to_sample.parity_sample_with_history(
                num_samples, seed, scale=scale, local_scale=local_scale
            )
        )
        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        for sign, q, _gens, _signs in zip(
            signs, qubit_sparse_pauli_list, sampled_with_history, sampled_signs
        ):
            for symbol in counts:
                if q == QubitSparsePauli(symbol):
                    counts[symbol] += 1
                    self.assertEqual(expected_signs[symbol], sign)

            sampled_sign = False
            sampled_pauli = QubitSparsePauli.from_label("I")
            for _i, (_g, _s) in enumerate(zip(_gens, _signs)):
                if _g:
                    sampled_sign = sampled_sign == _s
                    sampled_pauli = sampled_pauli.compose(generators[_i])

            self.assertEqual(sign, sampled_sign)
            self.assertEqual(q, sampled_pauli)

        for symbol, count in counts.items():
            self.assertTrue(np.abs(count / num_samples - probs_dict[symbol]) < 1e-2)

    def test_parity_sample(self):

        with self.subTest("all negative rates"):
            pauli_lindblad_map = PauliLindbladMap([("X", -1.0), ("Y", -0.5)])
            expected_signs = {"I": False, "X": True, "Y": True, "Z": False}

            self._assert_parity_sample_results(pauli_lindblad_map, expected_signs)

        with self.subTest("all positive rates"):
            pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", 0.5)])
            expected_signs = {"I": False, "X": False, "Y": False, "Z": False}

            self._assert_parity_sample_results(pauli_lindblad_map, expected_signs)

        with self.subTest("mix of positive and negative rates"):
            pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -0.5)])
            expected_signs = {"I": False, "X": False, "Y": True, "Z": True}

            self._assert_parity_sample_results(pauli_lindblad_map, expected_signs)

        with self.subTest("scale with mix of positive and negative rates"):
            pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -0.5)])
            pauli_lindblad_map_downscaled = PauliLindbladMap([("X", 0.5), ("Y", -0.25)])
            expected_signs = {"I": False, "X": False, "Y": True, "Z": True}

            self._assert_parity_sample_results(
                pauli_lindblad_map_downscaled,
                expected_signs,
                scale=2.0,
                plm_reference=pauli_lindblad_map,
            )

        with self.subTest("local_scale with mix of positive and negative rates"):
            pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -0.5)])
            pauli_lindblad_map_downscaled = PauliLindbladMap([("X", 1.0), ("Y", -0.25)])

            self._assert_parity_sample_results(
                pauli_lindblad_map_downscaled,
                expected_signs,
                local_scale=[1.0, 2.0],
                plm_reference=pauli_lindblad_map,
            )

    def test_parity_sample_without_seed(self):
        pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -0.5)])
        signs, qubit_sparse_pauli_list = pauli_lindblad_map.parity_sample(5)
        self.assertTrue(isinstance(qubit_sparse_pauli_list, QubitSparsePauliList))
        self.assertEqual(len(qubit_sparse_pauli_list), 5)
        self.assertEqual(len(signs), 5)

    def test_sample(self):
        pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", 0.5)])
        probs = pauli_lindblad_map.probabilities()
        probs_dict = {
            "I": probs[0] * probs[1],
            "X": (1 - probs[0]) * probs[1],
            "Y": probs[0] * (1 - probs[1]),
            "Z": (1 - probs[0]) * (1 - probs[1]),
        }

        num_samples = 10000
        qubit_sparse_pauli_list = pauli_lindblad_map.sample(num_samples, 12312)

        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        for q in qubit_sparse_pauli_list:
            for symbol in counts:
                if q == QubitSparsePauli(symbol):
                    counts[symbol] += 1
        for symbol, count in counts.items():
            self.assertTrue(np.abs(count / num_samples - probs_dict[symbol]) < 1e-2)

        # test callable without seed
        qubit_sparse_pauli_list = pauli_lindblad_map.sample(5)
        self.assertTrue(isinstance(qubit_sparse_pauli_list, QubitSparsePauliList))
        self.assertEqual(len(qubit_sparse_pauli_list), 5)

    def test_sample_errors(self):
        pauli_lindblad_map = PauliLindbladMap([("X", 1.0), ("Y", -1.0)])

        with self.assertRaisesRegex(
            ValueError, "PauliLindbladMap.sample called for a map with negative rates"
        ):
            pauli_lindblad_map.sample(1)


def canonicalize_term(pauli, indices, rate):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (sorted_paulis, np.asarray(indices)[idcs].tolist(), float(rate))


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
