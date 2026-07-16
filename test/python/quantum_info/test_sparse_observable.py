# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import copy
import itertools
import pickle
import random

import ddt
import numpy as np

from qiskit import transpile
from qiskit.circuit import Measure, Parameter, library, QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import SparseObservable, SparsePauliOp, Pauli, PauliList
from qiskit.transpiler import Target

from test import QiskitTestCase, combine


def single_cases():
    return [
        SparseObservable.zero(0),
        SparseObservable.zero(10),
        SparseObservable.identity(0),
        SparseObservable.identity(1_000),
        SparseObservable.from_label("IIXIZI"),
        SparseObservable.from_list([("YIXZII", -0.25), ("01rl+-", 0.25 + 0.5j)]),
        # Includes a duplicate entry.
        SparseObservable.from_list([("IXZ", -0.25), ("01I", 0.25 + 0.5j), ("IXZ", 0.75)]),
    ]


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


class AllowRightArithmetic:
    """Some type that implements only the right-hand-sided arithmetic operations, and allows
    `SparseObservable` to pass through them.

    The purpose of this is to detect that `SparseObservable` is correctly delegating binary
    operators to the other type if given an object it cannot coerce because of its type."""

    SENTINEL = object()

    __radd__ = __rsub__ = __rmul__ = __rtruediv__ = __rxor__ = lambda self, other: self.SENTINEL


@ddt.ddt
class TestSparseObservable(QiskitTestCase):
    def test_default_constructor_pauli(self):
        data = Pauli("IXYIZ")
        self.assertEqual(SparseObservable(data), SparseObservable.from_pauli(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=data.num_qubits), SparseObservable.from_pauli(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=data.num_qubits + 1)

        with_phase = Pauli("-jIYYXY")
        self.assertEqual(SparseObservable(with_phase), SparseObservable.from_pauli(with_phase))
        self.assertEqual(
            SparseObservable(with_phase, num_qubits=data.num_qubits),
            SparseObservable.from_pauli(with_phase),
        )

        self.assertEqual(SparseObservable(Pauli("")), SparseObservable.from_pauli(Pauli("")))

    def test_default_constructor_sparse_pauli_op(self):
        data = SparsePauliOp.from_list([("IIXIY", 1.0), ("XYYZI", -0.25), ("XYIYY", -0.25 + 0.75j)])
        self.assertEqual(SparseObservable(data), SparseObservable.from_sparse_pauli_op(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=data.num_qubits),
            SparseObservable.from_sparse_pauli_op(data),
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=data.num_qubits + 1)
        with self.assertRaisesRegex(TypeError, "complex-typed coefficients"):
            SparseObservable(SparsePauliOp(["XX"], [Parameter("x")]))

    def test_default_constructor_label(self):
        data = "IIXI+-I01rlIYZ"
        self.assertEqual(SparseObservable(data), SparseObservable.from_label(data))
        self.assertEqual(
            SparseObservable(data, num_qubits=len(data)), SparseObservable.from_label(data)
        )
        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(data, num_qubits=len(data) + 1)

    def test_default_constructor_list(self):
        data = [("IXIIZ", 0.5), ("+I-II", 1.0 - 0.25j), ("IIrlI", -0.75)]
        self.assertEqual(SparseObservable(data), SparseObservable.from_list(data))
        self.assertEqual(SparseObservable(data, num_qubits=5), SparseObservable.from_list(data))
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            SparseObservable(data, num_qubits=4)
        with self.assertRaisesRegex(ValueError, "label with length 5 cannot be added"):
            SparseObservable(data, num_qubits=6)
        self.assertEqual(
            SparseObservable([], num_qubits=5), SparseObservable.from_list([], num_qubits=5)
        )

    def test_default_constructor_sparse_list(self):
        data = [("ZX", (0, 3), 0.5), ("-+", (2, 4), 1.0 - 0.25j), ("rl", (2, 1), -0.75)]
        self.assertEqual(
            SparseObservable(data, num_qubits=5),
            SparseObservable.from_sparse_list(data, num_qubits=5),
        )
        self.assertEqual(
            SparseObservable(data, num_qubits=10),
            SparseObservable.from_sparse_list(data, num_qubits=10),
        )
        with self.assertRaisesRegex(ValueError, "'num_qubits' must be provided"):
            SparseObservable(data)
        self.assertEqual(
            SparseObservable([], num_qubits=5), SparseObservable.from_sparse_list([], num_qubits=5)
        )

    def test_default_constructor_copy(self):
        base = SparseObservable.from_list([("IXIZIY", 1.0), ("+-rl01", -1.0j)])
        copied = SparseObservable(base)
        self.assertEqual(base, copied)
        self.assertIsNot(base, copied)

        # Modifications to `copied` don't propagate back.
        copied.coeffs[1] = -0.5j
        self.assertNotEqual(base, copied)

        with self.assertRaisesRegex(ValueError, "explicitly given 'num_qubits'"):
            SparseObservable(base, num_qubits=base.num_qubits + 1)

    def test_default_constructor_term(self):
        expected = SparseObservable.from_list([("IIZXII+-", 2j)])
        self.assertEqual(SparseObservable(expected[0]), expected)

    def test_default_constructor_term_iterable(self):
        expected = SparseObservable.from_list([("IIZXII+-", 2j), ("rlIIIIII", 0.5)])
        terms = [expected[0], expected[1]]
        self.assertEqual(SparseObservable(list(terms)), expected)
        self.assertEqual(SparseObservable(tuple(terms)), expected)
        self.assertEqual(SparseObservable(term for term in terms), expected)

    def test_default_constructor_failed_inference(self):
        with self.assertRaises(TypeError):
            # Mixed dense/sparse list.
            SparseObservable([("IIXIZ", 1.0), ("+-", (2, 3), -1.0)], num_qubits=5)

    def test_num_qubits(self):
        self.assertEqual(SparseObservable.zero(0).num_qubits, 0)
        self.assertEqual(SparseObservable.zero(10).num_qubits, 10)

        self.assertEqual(SparseObservable.identity(0).num_qubits, 0)
        self.assertEqual(SparseObservable.identity(1_000_000).num_qubits, 1_000_000)

    def test_num_terms(self):
        self.assertEqual(SparseObservable.zero(0).num_terms, 0)
        self.assertEqual(SparseObservable.zero(10).num_terms, 0)
        self.assertEqual(SparseObservable.identity(0).num_terms, 1)
        self.assertEqual(SparseObservable.identity(1_000_000).num_terms, 1)
        self.assertEqual(
            SparseObservable.from_list([("IIIXIZ", 1.0), ("YY+-II", 0.5j)]).num_terms, 2
        )

    def test_len(self):
        self.assertEqual(len(SparseObservable.zero(0)), 0)
        self.assertEqual(len(SparseObservable.zero(10)), 0)
        self.assertEqual(len(SparseObservable.identity(0)), 1)
        self.assertEqual(len(SparseObservable.identity(1_000_000)), 1)
        self.assertEqual(len(SparseObservable.from_list([("IIIXIZ", 1.0), ("YY+-II", 0.5j)])), 2)

    def test_bit_term_enum(self):
        # These are very explicit tests that effectively just duplicate magic numbers, but the point
        # is that those magic numbers are required to be constant as their values are part of the
        # public interface.

        self.assertEqual(
            set(SparseObservable.BitTerm),
            {
                SparseObservable.BitTerm.X,
                SparseObservable.BitTerm.Y,
                SparseObservable.BitTerm.Z,
                SparseObservable.BitTerm.PLUS,
                SparseObservable.BitTerm.MINUS,
                SparseObservable.BitTerm.RIGHT,
                SparseObservable.BitTerm.LEFT,
                SparseObservable.BitTerm.ZERO,
                SparseObservable.BitTerm.ONE,
            },
        )
        # All the enumeration items should also be integers.
        self.assertIsInstance(SparseObservable.BitTerm.X, int)
        values = {
            "X": 0b00_10,
            "Y": 0b00_11,
            "Z": 0b00_01,
            "PLUS": 0b10_10,
            "MINUS": 0b01_10,
            "RIGHT": 0b10_11,
            "LEFT": 0b01_11,
            "ZERO": 0b10_01,
            "ONE": 0b01_01,
        }
        self.assertEqual({name: getattr(SparseObservable.BitTerm, name) for name in values}, values)

        # The single-character label aliases can be accessed with index notation.
        labels = {
            "X": SparseObservable.BitTerm.X,
            "Y": SparseObservable.BitTerm.Y,
            "Z": SparseObservable.BitTerm.Z,
            "+": SparseObservable.BitTerm.PLUS,
            "-": SparseObservable.BitTerm.MINUS,
            "r": SparseObservable.BitTerm.RIGHT,
            "l": SparseObservable.BitTerm.LEFT,
            "0": SparseObservable.BitTerm.ZERO,
            "1": SparseObservable.BitTerm.ONE,
        }
        self.assertEqual({label: SparseObservable.BitTerm[label] for label in labels}, labels)
        # The `label` property returns known values.
        self.assertEqual(
            {bit_term.label: bit_term for bit_term in SparseObservable.BitTerm}, labels
        )

    @ddt.idata(single_cases())
    def test_pickle(self, observable):
        self.assertEqual(observable, copy.copy(observable))
        self.assertIsNot(observable, copy.copy(observable))
        self.assertEqual(observable, copy.deepcopy(observable))
        self.assertEqual(observable, pickle.loads(pickle.dumps(observable)))

    @ddt.data(
        # This is every combination of (0, 1, many) for (terms, qubits, non-identites per term).
        SparseObservable.zero(0),
        SparseObservable.zero(1),
        SparseObservable.zero(10),
        SparseObservable.identity(0),
        SparseObservable.identity(1),
        SparseObservable.identity(1_000),
        SparseObservable.from_label("IIXIZI"),
        SparseObservable.from_label("X"),
        SparseObservable.from_list([("YIXZII", -0.25), ("01rl+-", 0.25 + 0.5j)]),
    )
    def test_repr(self, data):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        self.assertIsInstance(repr(data), str)
        self.assertIn("SparseObservable", repr(data))

    def test_from_raw_parts(self):
        # Happiest path: exactly typed inputs.
        num_qubits = 100
        terms = np.full((num_qubits,), SparseObservable.BitTerm.Z, dtype=np.uint8)
        indices = np.arange(num_qubits, dtype=np.uint32)
        coeffs = np.ones((num_qubits,), dtype=complex)
        boundaries = np.arange(num_qubits + 1, dtype=np.uintp)
        observable = SparseObservable.from_raw_parts(num_qubits, coeffs, terms, indices, boundaries)
        self.assertEqual(observable.num_qubits, num_qubits)
        np.testing.assert_equal(observable.bit_terms, terms)
        np.testing.assert_equal(observable.indices, indices)
        np.testing.assert_equal(observable.coeffs, coeffs)
        np.testing.assert_equal(observable.boundaries, boundaries)

        self.assertEqual(
            observable,
            SparseObservable.from_raw_parts(
                num_qubits, coeffs, terms, indices, boundaries, check=False
            ),
        )

        # At least the initial implementation of `SparseObservable` requires `from_raw_parts` to be
        # a copy constructor in order to allow it to be resized by Rust space.  This is checking for
        # that, but if the implementation changes, it could potentially be relaxed.
        self.assertFalse(np.may_share_memory(observable.coeffs, coeffs))

        # Conversion from array-likes, including mis-typed but compatible arrays.
        observable = SparseObservable.from_raw_parts(
            num_qubits, list(coeffs), tuple(terms), observable.indices, boundaries.astype(np.int16)
        )
        self.assertEqual(observable.num_qubits, num_qubits)
        np.testing.assert_equal(observable.bit_terms, terms)
        np.testing.assert_equal(observable.indices, indices)
        np.testing.assert_equal(observable.coeffs, coeffs)
        np.testing.assert_equal(observable.boundaries, boundaries)

        # Construction of zero operator.
        self.assertEqual(
            SparseObservable.from_raw_parts(10, [], [], [], [0]), SparseObservable.zero(10)
        )

        # Construction of an operator with an intermediate identity term.  For the initial
        # constructor tests, it's hard to check anything more than the construction succeeded.
        self.assertEqual(
            SparseObservable.from_raw_parts(
                10, [1.0j, 0.5, 2.0], [1, 3, 2], [0, 1, 2], [0, 1, 1, 3]
            ).num_terms,
            # The three are [(1.0j)*(Z_1), 0.5, 2.0*(X_2 Y_1)]
            3,
        )

    def test_from_raw_parts_checks_coherence(self):
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            SparseObservable.from_raw_parts(2, [1.0j], [ord("$")], [0], [0, 1])
        with self.assertRaisesRegex(ValueError, r"boundaries.*must be one element longer"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [0], [0])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(1\) and `indices` \(0\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [], [0, 1])
        with self.assertRaisesRegex(ValueError, r"`bit_terms` \(0\) and `indices` \(1\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [], [1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"the first item of `boundaries` \(1\) must be 0"):
            SparseObservable.from_raw_parts(2, [1.0j], [SparseObservable.BitTerm.Z], [0], [1, 1])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(2\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [1], [0], [0, 2])
        with self.assertRaisesRegex(ValueError, r"the last item of `boundaries` \(1\)"):
            SparseObservable.from_raw_parts(2, [1.0j], [1, 2], [0, 1], [0, 1])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            SparseObservable.from_raw_parts(4, [1.0j], [1, 2], [0, 4], [0, 2])
        with self.assertRaisesRegex(ValueError, r"all qubit indices must be less than the number"):
            SparseObservable.from_raw_parts(4, [1.0j, -0.5j], [1, 2], [0, 4], [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "the values in `boundaries` include backwards"):
            SparseObservable.from_raw_parts(
                5, [1.0j, -0.5j, 2.0], [1, 2, 3, 2], [0, 1, 2, 3], [0, 2, 1, 4]
            )
        with self.assertRaisesRegex(
            ValueError, "the values in `indices` are not term-wise increasing"
        ):
            SparseObservable.from_raw_parts(4, [1.0j], [1, 2], [1, 0], [0, 2])

        # There's no test of attempting to pass incoherent data and `check=False` because that
        # permits undefined behaviour in Rust (it's unsafe), so all bets would be off.

    def test_from_label(self):
        # The label is interpreted like a bitstring, with the right-most item associated with qubit
        # 0, and increasing as we move to the left (like `Pauli`, and other bitstring conventions).
        self.assertEqual(
            # Ruler for counting terms:  dcba9876543210
            SparseObservable.from_label("I+-II01IrlIXYZ"),
            SparseObservable.from_raw_parts(
                14,
                [1.0],
                [
                    SparseObservable.BitTerm.Z,
                    SparseObservable.BitTerm.Y,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.LEFT,
                    SparseObservable.BitTerm.RIGHT,
                    SparseObservable.BitTerm.ONE,
                    SparseObservable.BitTerm.ZERO,
                    SparseObservable.BitTerm.MINUS,
                    SparseObservable.BitTerm.PLUS,
                ],
                [0, 1, 2, 4, 5, 7, 8, 11, 12],
                [0, 9],
            ),
        )

        self.assertEqual(SparseObservable.from_label("I" * 10), SparseObservable.identity(10))

        # The empty label case is a 0-qubit identity, since `from_label` always sets a coefficient
        # of 1.
        self.assertEqual(SparseObservable.from_label(""), SparseObservable.identity(0))

    def test_from_label_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_label("I+-$%I")
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            SparseObservable.from_label("🐍")

    def test_from_list(self):
        label = "IXYI+-0lr1ZZY"
        self.assertEqual(
            SparseObservable.from_list([(label, 1.0)]), SparseObservable.from_label(label)
        )
        self.assertEqual(
            SparseObservable.from_list([(label, 1.0)], num_qubits=len(label)),
            SparseObservable.from_label(label),
        )

        self.assertEqual(
            SparseObservable.from_list([("IIIXZI", 1.0j), ("+-IIII", -0.5)]),
            SparseObservable.from_raw_parts(
                6,
                [1.0j, -0.5],
                [
                    SparseObservable.BitTerm.Z,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.MINUS,
                    SparseObservable.BitTerm.PLUS,
                ],
                [1, 2, 4, 5],
                [0, 2, 4],
            ),
        )

        self.assertEqual(SparseObservable.from_list([], num_qubits=5), SparseObservable.zero(5))
        self.assertEqual(SparseObservable.from_list([], num_qubits=0), SparseObservable.zero(0))

    def test_from_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_list([("XZIIZY", 0.5), ("I+-$%I", 1.0j)])
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Unicode shenangigans.
            SparseObservable.from_list([("🐍", 0.5)])
        with self.assertRaisesRegex(ValueError, "label with length 4 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IIXI", 1.0j)])
        with self.assertRaisesRegex(ValueError, "label with length 2 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("II", 1.0j)])
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IXI", 1.0j)], num_qubits=2)
        with self.assertRaisesRegex(ValueError, "label with length 3 cannot be added"):
            SparseObservable.from_list([("IIZ", 0.5), ("IXI", 1.0j)], num_qubits=4)
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            SparseObservable.from_list([])

    def test_from_sparse_list(self):
        self.assertEqual(
            SparseObservable.from_sparse_list(
                [
                    ("XY", (0, 1), 0.5),
                    ("+-", (1, 3), -0.25j),
                    ("rl0", (0, 2, 4), 1.0j),
                ],
                num_qubits=5,
            ),
            SparseObservable.from_list([("IIIYX", 0.5), ("I-I+I", -0.25j), ("0IlIr", 1.0j)]),
        )

        # The indices should be allowed to be given in unsorted order, but they should be term-wise
        # sorted in the output.
        from_unsorted = SparseObservable.from_sparse_list(
            [
                ("XYZ", (2, 1, 0), 1.5j),
                ("+rl", (2, 0, 1), -0.5),
            ],
            num_qubits=3,
        )
        self.assertEqual(from_unsorted, SparseObservable.from_list([("XYZ", 1.5j), ("+lr", -0.5)]))
        np.testing.assert_equal(
            from_unsorted.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32)
        )

        # Explicit identities should still work, just be skipped over.
        explicit_identity = SparseObservable.from_sparse_list(
            [
                ("ZXI", (0, 1, 2), 1.0j),
                ("XYIII", (0, 1, 2, 3, 8), -0.5j),
            ],
            num_qubits=10,
        )
        self.assertEqual(
            explicit_identity,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), 1.0j), ("YX", (1, 0), -0.5j)], num_qubits=10
            ),
        )
        np.testing.assert_equal(explicit_identity.indices, np.array([0, 1, 0, 1], dtype=np.uint32))

        self.assertEqual(
            SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=5),
            SparseObservable.identity(5),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list([("", (), 1.0)], num_qubits=0),
            SparseObservable.identity(0),
        )

        self.assertEqual(
            SparseObservable.from_sparse_list([], num_qubits=1_000_000),
            SparseObservable.zero(1_000_000),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list([], num_qubits=0),
            SparseObservable.zero(0),
        )

    def test_from_sparse_list_failures(self):
        with self.assertRaisesRegex(ValueError, "labels must only contain letters from"):
            # Bad letters that are still ASCII.
            SparseObservable.from_sparse_list(
                [("XZZY", (5, 3, 1, 0), 0.5), ("+$", (2, 1), 1.0j)], num_qubits=8
            )
        # Unicode shenangigans.  These two should fail with a `ValueError`, but the exact message
        # isn't important.  "\xff" is "ÿ", which is two bytes in UTF-8 (so has a length of 2 in
        # Rust), but has a length of 1 in Python, so try with both a length-1 and length-2 index
        # sequence, and both should still raise `ValueError`.
        with self.assertRaises(ValueError):
            SparseObservable.from_sparse_list([("\xff", (1,), 0.5)], num_qubits=5)
        with self.assertRaises(ValueError):
            SparseObservable.from_sparse_list([("\xff", (1, 2), 0.5)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            SparseObservable.from_sparse_list([("XZ", (0,), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "label with length 2 does not match indices"):
            SparseObservable.from_sparse_list([("XZ", (0, 1, 2), 1.0)], num_qubits=5)

        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            SparseObservable.from_sparse_list([("XZY", (0, 1, 3), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 4 is out of range for a 3-qubit operator"):
            SparseObservable.from_sparse_list([("XZY", (0, 1, 4), 1.0)], num_qubits=3)
        with self.assertRaisesRegex(ValueError, "index 3 is out of range for a 3-qubit operator"):
            # ... even if it's for an explicit identity.
            SparseObservable.from_sparse_list([("+-I", (0, 1, 3), 1.0)], num_qubits=3)

        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            SparseObservable.from_sparse_list([("XZ", (3, 3), 1.0)], num_qubits=5)
        with self.assertRaisesRegex(ValueError, "index 3 is duplicated"):
            SparseObservable.from_sparse_list([("XYZXZ", (3, 0, 1, 2, 3), 1.0)], num_qubits=5)

    def test_from_pauli(self):
        # This function should be infallible provided `Pauli` doesn't change its interface and the
        # user doesn't violate the typing.

        # Simple check that the labels are interpreted in the same order.
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("IIXZI")), SparseObservable.from_label("IIXZI")
        )

        # `Pauli` accepts a phase in its label, which we can't (because of clashes with the other
        # alphabet letters), and we should get that right.
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("iIXZIX")),
            SparseObservable.from_list([("IXZIX", 1.0j)]),
        )
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("-iIXZIX")),
            SparseObservable.from_list([("IXZIX", -1.0j)]),
        )
        self.assertEqual(
            SparseObservable.from_pauli(Pauli("-IXZIX")),
            SparseObservable.from_list([("IXZIX", -1.0)]),
        )

        # `Pauli` has its internal phase convention for how it stores `Y`; we should get this right
        # regardless of how many Ys are in the label, or if there's a phase.
        paulis = {"IXYZ" * n: Pauli("IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_labels = zip(
            *(
                (SparseObservable.from_pauli(pauli), SparseObservable.from_label(label))
                for label, pauli in paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_labels)

        phased_paulis = {"IXYZ" * n: Pauli("j" + "IXYZ" * n) for n in range(1, 5)}
        from_paulis, from_lists = zip(
            *(
                (SparseObservable.from_pauli(pauli), SparseObservable.from_list([(label, 1.0j)]))
                for label, pauli in phased_paulis.items()
            )
        )
        self.assertEqual(from_paulis, from_lists)

        self.assertEqual(SparseObservable.from_pauli(Pauli("III")), SparseObservable.identity(3))
        self.assertEqual(SparseObservable.from_pauli(Pauli("")), SparseObservable.identity(0))

    def test_from_sparse_pauli_op(self):
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([("IIIII", 1.0)])),
            SparseObservable.identity(5),
        )

        data = [("ZXZXZ", 0.25), ("IYXZI", 1.0j), ("IYYZX", 0.5), ("YYYXI", -0.5), ("IYYYY", 2j)]
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list(data)),
            SparseObservable.from_list(data),
        )

        # These two _should_ produce the same structure as `SparseObservable.zero(num_qubits)`, but
        # because `SparsePauliOp` doesn't represent the zero operator "natively" - with an empty sum
        # - they actually come out looking like `0.0` times the identity, which is less efficient
        # but acceptable.
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([], num_qubits=1)),
            SparseObservable.from_list([("I", 0.0)]),
        )
        self.assertEqual(
            SparseObservable.from_sparse_pauli_op(SparsePauliOp.from_list([], num_qubits=0)),
            SparseObservable.from_list([("", 0.0)]),
        )

    def test_from_sparse_pauli_op_failures(self):
        parametric = SparsePauliOp.from_list([("IIXZ", Parameter("x"))], dtype=object)
        with self.assertRaisesRegex(TypeError, "complex-typed coefficients"):
            SparseObservable.from_sparse_pauli_op(parametric)

    def test_from_terms(self):
        self.assertEqual(SparseObservable.from_terms([], num_qubits=5), SparseObservable.zero(5))
        self.assertEqual(SparseObservable.from_terms((), num_qubits=0), SparseObservable.zero(0))
        self.assertEqual(
            SparseObservable.from_terms((None for _ in []), num_qubits=3), SparseObservable.zero(3)
        )

        expected = SparseObservable.from_sparse_list(
            [
                ("XYZ", (4, 2, 1), 1j),
                ("+-rl", (8, 5, 3, 2), 0.5),
                ("01", (5, 0), 2.0),
            ],
            num_qubits=10,
        )
        self.assertEqual(SparseObservable.from_terms(list(expected)), expected)
        self.assertEqual(SparseObservable.from_terms(tuple(expected)), expected)
        self.assertEqual(SparseObservable.from_terms(term for term in expected), expected)
        self.assertEqual(
            SparseObservable.from_terms(
                (term for term in expected), num_qubits=expected.num_qubits
            ),
            expected,
        )

    def test_from_terms_failures(self):
        with self.assertRaisesRegex(ValueError, "cannot construct.*without knowing `num_qubits`"):
            SparseObservable.from_terms([])

        left, right = SparseObservable("IIXYI")[0], SparseObservable("IIIIIIIIX")[0]
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            SparseObservable.from_terms([left, right])
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            SparseObservable.from_terms([left], num_qubits=100)

    def test_zero(self):
        zero_5 = SparseObservable.zero(5)
        self.assertEqual(zero_5.num_qubits, 5)
        np.testing.assert_equal(zero_5.coeffs, np.array([], dtype=complex))
        np.testing.assert_equal(zero_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(zero_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(zero_5.boundaries, np.array([0], dtype=np.uintp))

        zero_0 = SparseObservable.zero(0)
        self.assertEqual(zero_0.num_qubits, 0)
        np.testing.assert_equal(zero_0.coeffs, np.array([], dtype=complex))
        np.testing.assert_equal(zero_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(zero_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(zero_0.boundaries, np.array([0], dtype=np.uintp))

    def test_identity(self):
        id_5 = SparseObservable.identity(5)
        self.assertEqual(id_5.num_qubits, 5)
        np.testing.assert_equal(id_5.coeffs, np.array([1], dtype=complex))
        np.testing.assert_equal(id_5.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(id_5.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(id_5.boundaries, np.array([0, 0], dtype=np.uintp))

        id_0 = SparseObservable.identity(0)
        self.assertEqual(id_0.num_qubits, 0)
        np.testing.assert_equal(id_0.coeffs, np.array([1], dtype=complex))
        np.testing.assert_equal(id_0.bit_terms, np.array([], dtype=np.uint8))
        np.testing.assert_equal(id_0.indices, np.array([], dtype=np.uint32))
        np.testing.assert_equal(id_0.boundaries, np.array([0, 0], dtype=np.uintp))

    @ddt.idata(single_cases())
    def test_copy(self, obs):
        self.assertEqual(obs, obs.copy())
        self.assertIsNot(obs, obs.copy())

    def test_equality(self):
        sparse_data = [("XZ", (1, 0), 0.5j), ("+lr", (3, 1, 0), -0.25)]
        op = SparseObservable.from_sparse_list(sparse_data, num_qubits=5)
        self.assertEqual(op, op.copy())
        # Take care that Rust space allows multiple views onto the same object.
        self.assertEqual(op, op)

        # Comparison to some other object shouldn't fail.
        self.assertNotEqual(op, None)

        # No costly automatic simplification (mathematically, these operators _are_ the same).
        self.assertNotEqual(
            SparseObservable.from_list([("+", 1.0), ("-", 1.0)]), SparseObservable.from_label("X")
        )

        # Difference in qubit count.
        self.assertNotEqual(
            op, SparseObservable.from_sparse_list(sparse_data, num_qubits=op.num_qubits + 1)
        )
        self.assertNotEqual(SparseObservable.zero(2), SparseObservable.zero(3))
        self.assertNotEqual(SparseObservable.identity(2), SparseObservable.identity(3))

        # Difference in coeffs.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", 0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0j), ("+-rl0", -0.5j)]),
        )

        # Difference in bit terms.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIYZI", 1.0), ("+-rl0", -0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("+-rl1", -0.5j)]),
        )

        # Difference in indices.
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+Irl0", -0.5j)]),
            SparseObservable.from_list([("IXIZI", 1.0), ("+Irl0", -0.5j)]),
        )
        self.assertNotEqual(
            SparseObservable.from_list([("IIXZI", 1.0), ("+Irl0", -0.5j)]),
            SparseObservable.from_list([("IIXZI", 1.0), ("I+rl0", -0.5j)]),
        )

        # Difference in boundaries.
        self.assertNotEqual(
            SparseObservable.from_sparse_list(
                [("XZ", (0, 1), 1.5), ("+-", (2, 3), -0.5j)], num_qubits=5
            ),
            SparseObservable.from_sparse_list(
                [("XZ+", (0, 1, 2), 1.5), ("-", (3,), -0.5j)], num_qubits=5
            ),
        )

    def test_write_into_attributes_scalar(self):
        coeffs = SparseObservable.from_sparse_list(
            [("XZ", (1, 0), 1.5j), ("+-", (3, 2), -1.5j)], num_qubits=8
        )
        coeffs.coeffs[0] = -2.0
        self.assertEqual(
            coeffs,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("+-", (3, 2), -1.5j)], num_qubits=8
            ),
        )
        coeffs.coeffs[1] = 1.5 + 0.25j
        self.assertEqual(
            coeffs,
            SparseObservable.from_sparse_list(
                [("XZ", (1, 0), -2.0), ("+-", (3, 2), 1.5 + 0.25j)], num_qubits=8
            ),
        )

        bit_terms = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        bit_terms.bit_terms[0] = SparseObservable.BitTerm.Y
        bit_terms.bit_terms[3] = SparseObservable.BitTerm.LEFT
        self.assertEqual(
            bit_terms,
            SparseObservable.from_sparse_list(
                [("YZ", (0, 1), 1.5j), ("+l", (2, 3), -1.5j)], num_qubits=8
            ),
        )

        indices = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        # These two sets keep the observable in term-wise increasing order.  We don't test what
        # happens if somebody violates the Rust-space requirement to be term-wise increasing.
        indices.indices[1] = 4
        indices.indices[3] = 7
        self.assertEqual(
            indices,
            SparseObservable.from_sparse_list(
                [("XZ", (0, 4), 1.5j), ("+-", (2, 7), -1.5j)], num_qubits=8
            ),
        )

        boundaries = SparseObservable.from_sparse_list(
            [("XZ", (0, 1), 1.5j), ("+-", (2, 3), -1.5j)], num_qubits=8
        )
        # Move a single-qubit term from the second summand into the first (the particular indices
        # ensure we remain term-wise sorted).
        boundaries.boundaries[1] += 1
        self.assertEqual(
            boundaries,
            SparseObservable.from_sparse_list(
                [("XZ+", (0, 1, 2), 1.5j), ("-", (3,), -1.5j)], num_qubits=8
            ),
        )

    def test_write_into_attributes_broadcast(self):
        coeffs = SparseObservable.from_list([("XIIZI", 1.5j), ("IIIl0", -0.25), ("1IIIY", 0.5)])
        coeffs.coeffs[:] = 1.5j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, 1.5j, 1.5j])
        coeffs.coeffs[1:] = 1.0j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, 1.0j, 1.0j])
        coeffs.coeffs[:2] = -0.5
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5, -0.5, 1.0j])
        coeffs.coeffs[::2] = 1.5j
        np.testing.assert_array_equal(coeffs.coeffs, [1.5j, -0.5, 1.5j])
        coeffs.coeffs[::-1] = -0.5j
        np.testing.assert_array_equal(coeffs.coeffs, [-0.5j, -0.5j, -0.5j])

        # It's hard to broadcast into `indices` without breaking data coherence; the broadcasting is
        # more meant for fast modifications to `coeffs` and `bit_terms`.
        indices = SparseObservable.from_list([("XIIZI", 1.5j), ("IIlI0", -0.25), ("1IIIY", 0.5)])
        indices.indices[::2] = 1
        self.assertEqual(
            indices, SparseObservable.from_list([("XIIZI", 1.5j), ("IIl0I", -0.25), ("1IIYI", 0.5)])
        )

        bit_terms = SparseObservable.from_list([("XIIZI", 1.5j), ("IIlI0", -0.25), ("1IIIY", 0.5)])
        bit_terms.bit_terms[::2] = SparseObservable.BitTerm.Z
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("IIlIZ", -0.25), ("1IIIZ", 0.5)]),
        )
        bit_terms.bit_terms[3:1:-1] = SparseObservable.BitTerm.PLUS
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("II+I+", -0.25), ("1IIIZ", 0.5)]),
        )
        bit_terms.bit_terms[bit_terms.boundaries[2] : bit_terms.boundaries[3]] = (
            SparseObservable.BitTerm.MINUS
        )
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("XIIZI", 1.5j), ("II+I+", -0.25), ("-III-", 0.5)]),
        )

        boundaries = SparseObservable.from_list([("IIIIZX", 1j), ("II+-II", -0.5), ("rlIIII", 0.5)])
        boundaries.boundaries[1:3] = 1
        self.assertEqual(
            boundaries,
            SparseObservable.from_list([("IIIIIX", 1j), ("IIIIII", -0.5), ("rl+-ZI", 0.5)]),
        )

    def test_write_into_attributes_slice(self):
        coeffs = SparseObservable.from_list([("XIIZI", 1.5j), ("IIIl0", -0.25), ("1IIIY", 0.5)])
        coeffs.coeffs[:] = [2.0, 0.5, -0.25]
        self.assertEqual(
            coeffs, SparseObservable.from_list([("XIIZI", 2.0), ("IIIl0", 0.5), ("1IIIY", -0.25)])
        )
        # This should assign the coefficients in reverse order - we more usually spell it
        # `coeffs[:] = coeffs{::-1]`, but the idea is to check the set-item slicing order.
        coeffs.coeffs[::-1] = coeffs.coeffs[:]
        self.assertEqual(
            coeffs, SparseObservable.from_list([("XIIZI", -0.25), ("IIIl0", 0.5), ("1IIIY", 2.0)])
        )

        indices = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        indices.indices[:4] = [4, 5, 1, 2]
        self.assertEqual(
            indices, SparseObservable.from_list([("ZXIIII", 0.25), ("III+-I", 1j), ("rlIIII", 0.5)])
        )

        bit_terms = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        bit_terms.bit_terms[::2] = [
            SparseObservable.BitTerm.Y,
            SparseObservable.BitTerm.RIGHT,
            SparseObservable.BitTerm.ZERO,
        ]
        self.assertEqual(
            bit_terms,
            SparseObservable.from_list([("IIIIZY", 0.25), ("II+rII", 1j), ("r0IIII", 0.5)]),
        )

        operators = SparseObservable.from_list([("XZY", 1.5j), ("+1r", -0.5)])
        # Reduce all single-qubit terms to the relevant Pauli operator, if they are a projector.
        operators.bit_terms[:] = operators.bit_terms[:] & 0b00_11
        self.assertEqual(operators, SparseObservable.from_list([("XZY", 1.5j), ("XZY", -0.5)]))

        boundaries = SparseObservable.from_list([("IIIIZX", 0.25), ("II+-II", 1j), ("rlIIII", 0.5)])
        boundaries.boundaries[1:-1] = [1, 5]
        self.assertEqual(
            boundaries,
            SparseObservable.from_list([("IIIIIX", 0.25), ("Il+-ZI", 1j), ("rIIIII", 0.5)]),
        )

    def test_attributes_reject_bad_writes(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        with self.assertRaises(TypeError):
            obs.coeffs[0] = [0.25j, 0.5j]
        with self.assertRaises(TypeError):
            obs.bit_terms[0] = [SparseObservable.BitTerm.PLUS] * 4
        with self.assertRaises(TypeError):
            obs.indices[0] = [0, 1]
        with self.assertRaises(TypeError):
            obs.boundaries[0] = (0, 1)
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            obs.bit_terms[0] = 0
        with self.assertRaisesRegex(ValueError, "not a valid letter"):
            obs.bit_terms[:] = 0
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 2 with a sequence of length 1"
        ):
            obs.coeffs[:] = [1.0j]
        with self.assertRaisesRegex(
            ValueError, "tried to set a slice of length 6 with a sequence of length 8"
        ):
            obs.bit_terms[:] = [SparseObservable.BitTerm.Z] * 8

    def test_attributes_sequence(self):
        """Test attributes of the `Sequence` protocol."""
        # Length
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        self.assertEqual(len(obs.coeffs), 2)
        self.assertEqual(len(obs.indices), 6)
        self.assertEqual(len(obs.bit_terms), 6)
        self.assertEqual(len(obs.boundaries), 3)

        # Iteration
        self.assertEqual(list(obs.coeffs), [1.5j, -0.5])
        self.assertEqual(tuple(obs.indices), (0, 1, 2, 0, 1, 2))
        self.assertEqual(next(iter(obs.boundaries)), 0)
        # multiple iteration through same object
        bit_terms = obs.bit_terms
        self.assertEqual(set(bit_terms), {SparseObservable.BitTerm[x] for x in "XYZ+-r"})
        self.assertEqual(set(bit_terms), {SparseObservable.BitTerm[x] for x in "XYZ+-r"})

        # Implicit iteration methods.
        self.assertIn(SparseObservable.BitTerm.PLUS, obs.bit_terms)
        self.assertNotIn(4, obs.indices)
        self.assertEqual(list(reversed(obs.coeffs)), [-0.5, 1.5j])

        # Index by scalar
        self.assertEqual(obs.coeffs[1], -0.5)
        self.assertEqual(obs.indices[-1], 2)
        self.assertEqual(obs.bit_terms[0], SparseObservable.BitTerm.Y)
        # Make sure that Rust-space actually returns the enum value, not just an `int` (which could
        # have compared equal).
        self.assertIsInstance(obs.bit_terms[0], SparseObservable.BitTerm)
        self.assertEqual(obs.boundaries[-2], 3)
        with self.assertRaises(IndexError):
            _ = obs.coeffs[10]
        with self.assertRaises(IndexError):
            _ = obs.boundaries[-4]

        # Index by slice.  This is API guaranteed to be a Numpy array to make it easier to
        # manipulate subslices with mathematic operations.
        self.assertIsInstance(obs.coeffs[:], np.ndarray)
        np.testing.assert_array_equal(
            obs.coeffs[:], np.array([1.5j, -0.5], dtype=np.complex128), strict=True
        )
        self.assertIsInstance(obs.indices[::-1], np.ndarray)
        np.testing.assert_array_equal(
            obs.indices[::-1], np.array([2, 1, 0, 2, 1, 0], dtype=np.uint32), strict=True
        )
        self.assertIsInstance(obs.bit_terms[2:4], np.ndarray)
        np.testing.assert_array_equal(
            obs.bit_terms[2:4],
            np.array([SparseObservable.BitTerm.X, SparseObservable.BitTerm.RIGHT], dtype=np.uint8),
            strict=True,
        )
        self.assertIsInstance(obs.boundaries[-2:-3:-1], np.ndarray)
        np.testing.assert_array_equal(
            obs.boundaries[-2:-3:-1], np.array([3], dtype=np.uintp), strict=True
        )

    def test_attributes_to_array(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])

        # Natural dtypes.
        np.testing.assert_array_equal(
            obs.coeffs, np.array([1.5j, -0.5], dtype=np.complex128), strict=True
        )
        np.testing.assert_array_equal(
            obs.indices, np.array([0, 1, 2, 0, 1, 2], dtype=np.uint32), strict=True
        )
        np.testing.assert_array_equal(
            obs.bit_terms,
            np.array([SparseObservable.BitTerm[x] for x in "YZXr-+"], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            obs.boundaries, np.array([0, 3, 6], dtype=np.uintp), strict=True
        )

        # Cast dtypes.
        np.testing.assert_array_equal(
            np.array(obs.indices, dtype=np.uint8),
            np.array([0, 1, 2, 0, 1, 2], dtype=np.uint8),
            strict=True,
        )
        np.testing.assert_array_equal(
            np.array(obs.boundaries, dtype=np.int64),
            np.array([0, 3, 6], dtype=np.int64),
            strict=True,
        )

    def test_attributes_reject_no_copy_array(self):
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.coeffs, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.indices, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.bit_terms, copy=False)
        with self.assertRaisesRegex(ValueError, "cannot produce a safe view"):
            np.asarray(obs.boundaries, copy=False)

    def test_attributes_repr(self):
        # We're not testing much about the outputs here, just that they don't crash.
        obs = SparseObservable.from_list([("XZY", 1.5j), ("+-r", -0.5)])
        self.assertIn("coeffs", repr(obs.coeffs))
        self.assertIn("bit_terms", repr(obs.bit_terms))
        self.assertIn("indices", repr(obs.indices))
        self.assertIn("boundaries", repr(obs.boundaries))

    @combine(
        obs=single_cases(),
        # This includes some elements that aren't native `complex`, but still should be cast.
        coeff=[0.5, 3j, 2, 0.25 - 0.75j],
    )
    def test_multiply(self, obs, coeff):
        obs = obs.copy()
        initial = obs.copy()
        expected = obs.copy()
        expected.coeffs[:] = np.asarray(expected.coeffs) * complex(coeff)
        self.assertEqual(obs * coeff, expected)
        self.assertEqual(coeff * obs, expected)
        # Check that nothing applied in-place.
        self.assertEqual(obs, initial)
        obs *= coeff
        self.assertEqual(obs, expected)
        self.assertIs(obs * AllowRightArithmetic(), AllowRightArithmetic.SENTINEL)

    @ddt.idata(single_cases())
    def test_multiply_zero(self, obs):
        initial = obs.copy()
        self.assertEqual(obs * 0.0, SparseObservable.zero(initial.num_qubits))
        self.assertEqual(0.0 * obs, SparseObservable.zero(initial.num_qubits))
        self.assertEqual(obs, initial)

        obs *= 0.0
        self.assertEqual(obs, SparseObservable.zero(initial.num_qubits))

    @combine(
        obs=single_cases(),
        # This includes some elements that aren't native `complex`, but still should be cast.  Be
        # careful that the floating-point operation should not involve rounding.
        coeff=[0.5, 4j, 2, -0.25],
    )
    def test_divide(self, obs, coeff):
        obs = obs.copy()
        initial = obs.copy()
        expected = obs.copy()
        expected.coeffs[:] = np.asarray(expected.coeffs) / complex(coeff)
        self.assertEqual(obs / coeff, expected)
        # Check that nothing applied in-place.
        self.assertEqual(obs, initial)
        obs /= coeff
        self.assertEqual(obs, expected)
        self.assertIs(obs / AllowRightArithmetic(), AllowRightArithmetic.SENTINEL)

    @ddt.idata(single_cases())
    def test_divide_zero_raises(self, obs):
        with self.assertRaises(ZeroDivisionError):
            _ = obs / 0.0j
        with self.assertRaises(ZeroDivisionError):
            obs /= 0.0j

    def test_add_simple(self):
        num_qubits = 12
        terms = [
            ("ZXY", (5, 2, 1), 1.5j),
            ("+r", (8, 0), -0.25),
            ("-0l1", (10, 9, 4, 3), 0.5 + 1j),
            ("XZ", (7, 5), 0.75j),
            ("rl01", (5, 3, 1, 0), 0.25j),
        ]
        expected = SparseObservable.from_sparse_list(terms, num_qubits=num_qubits)
        for pivot in range(1, len(terms) - 1):
            left = SparseObservable.from_sparse_list(terms[:pivot], num_qubits=num_qubits)
            left_initial = left.copy()
            right = SparseObservable.from_sparse_list(terms[pivot:], num_qubits=num_qubits)
            right_initial = right.copy()
            # Addition is documented to be term-stacking, so structural equality without `simplify`
            # should hold.
            self.assertEqual(left + right, expected)
            # This is a different order, so check the simplification and canonicalisation works.
            self.assertEqual((right + left).simplify(), expected.simplify())
            # Neither was modified in place.
            self.assertEqual(left, left_initial)
            self.assertEqual(right, right_initial)

            left += right
            self.assertEqual(left, expected)
            self.assertEqual(right, right_initial)

    @ddt.idata(single_cases())
    def test_add_self(self, obs):
        """Test that addition to `self` works fine, including in-place mutation.  This is a case
        where we might fall afoul of Rust's borrowing rules."""
        initial = obs.copy()
        expected = (2.0 * obs).simplify()
        self.assertEqual((obs + obs).simplify(), expected)
        self.assertEqual(obs, initial)

        obs += obs
        self.assertEqual(obs.simplify(), expected)

    @ddt.idata(single_cases())
    def test_add_zero(self, obs):
        expected = obs.copy()
        zero = SparseObservable.zero(obs.num_qubits)
        self.assertEqual(obs + zero, expected)
        self.assertEqual(zero + obs, expected)

        obs += zero
        self.assertEqual(obs, expected)
        zero += obs
        self.assertEqual(zero, expected)

    def test_add_coercion(self):
        """Other quantum-info operators coerce with the ``+`` operator, so we do too."""
        base = SparseObservable.zero(9)

        pauli_label = "IIIXYZIII"
        expected = SparseObservable.from_label(pauli_label)
        self.assertEqual(base + pauli_label, expected)
        self.assertEqual(pauli_label + base, expected)

        pauli = Pauli(pauli_label)
        self.assertEqual(base + pauli, expected)
        self.assertEqual(pauli + base, expected)

        spo = SparsePauliOp(pauli_label)
        self.assertEqual(base + spo, expected)
        with self.assertRaisesRegex(QiskitError, "Invalid input data for Pauli"):
            # This doesn't work because `SparsePauliOp` is badly behaved in its coercion (it gets
            # first dibs at `__add__`, not our `__radd__`), and will not return `NotImplemented` for
            # bad types.  This _shouldn't_ raise, and this test here is to remind us to flip it to a
            # proper assertion of correctness if `Pauli` starts playing nicely.
            _ = spo + base

        obs_label = "10+-rlXYZ"
        expected = SparseObservable.from_label(obs_label)
        self.assertEqual(base + obs_label, expected)
        self.assertEqual(obs_label + base, expected)

        expected = 3j * SparseObservable.from_label("IXYrlII0I")
        self.assertEqual(base + expected[0], expected)
        self.assertEqual(expected[0] + base, expected)

        with self.assertRaises(TypeError):
            _ = base + {}
        with self.assertRaises(TypeError):
            _ = {} + base
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = base + "$$$"
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = "$$$" + base

        self.assertIs(base + AllowRightArithmetic(), AllowRightArithmetic.SENTINEL)
        with self.assertRaisesRegex(TypeError, "invalid object for in-place addition"):
            # This actually _shouldn't_ be a `TypeError` - `__iadd_` should defer to
            # `AllowRightArithmetic.__radd__` in the same way that `__add__` does, but a limitation
            # in PyO3 (see PyO3/pyo3#4605) prevents this.
            base += AllowRightArithmetic()

    def test_add_failures(self):
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            _ = SparseObservable.zero(4) + SparseObservable.zero(6)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            _ = SparseObservable.zero(6) + SparseObservable.zero(4)

    def test_sub_simple(self):
        num_qubits = 12
        terms = [
            ("ZXY", (5, 2, 1), 1.5j),
            ("+r", (8, 0), -0.25),
            ("-0l1", (10, 9, 4, 3), 0.5 + 1j),
            ("XZ", (7, 5), 0.75j),
            ("rl01", (5, 3, 1, 0), 0.25j),
        ]
        for pivot in range(1, len(terms) - 1):
            expected = SparseObservable.from_sparse_list(
                [
                    (label, indices, coeff if i < pivot else -coeff)
                    for i, (label, indices, coeff) in enumerate(terms)
                ],
                num_qubits=num_qubits,
            )
            left = SparseObservable.from_sparse_list(terms[:pivot], num_qubits=num_qubits)
            left_initial = left.copy()
            right = SparseObservable.from_sparse_list(terms[pivot:], num_qubits=num_qubits)
            right_initial = right.copy()
            # Addition is documented to be term-stacking, so structural equality without `simplify`
            # should hold.
            self.assertEqual(left - right, expected)
            # This is a different order, so check the simplification and canonicalisation works.
            self.assertEqual((right - left).simplify(), -expected.simplify())
            # Neither was modified in place.
            self.assertEqual(left, left_initial)
            self.assertEqual(right, right_initial)

            left -= right
            self.assertEqual(left, expected)
            self.assertEqual(right, right_initial)

    @ddt.idata(single_cases())
    def test_sub_self(self, obs):
        """Test that subtraction of `self` works fine, including in-place mutation.  This is a case
        where we might fall afoul of Rust's borrowing rules."""
        initial = obs.copy()
        expected = SparseObservable.zero(obs.num_qubits)
        self.assertEqual((obs - obs).simplify(), expected)
        self.assertEqual(obs, initial)

        obs -= obs
        self.assertEqual(obs.simplify(), expected)

    @ddt.idata(single_cases())
    def test_sub_zero(self, obs):
        expected = obs.copy()
        zero = SparseObservable.zero(obs.num_qubits)
        self.assertEqual(obs - zero, expected)
        self.assertEqual(zero - obs, -expected)

        obs -= zero
        self.assertEqual(obs, expected)
        zero -= obs
        self.assertEqual(zero, -expected)

    def test_sub_coercion(self):
        """Other quantum-info operators coerce with the ``-`` operator, so we do too."""
        base = SparseObservable.zero(9)

        pauli_label = "IIIXYZIII"
        expected = SparseObservable.from_label(pauli_label)
        self.assertEqual(base - pauli_label, -expected)
        self.assertEqual(pauli_label - base, expected)

        pauli = Pauli(pauli_label)
        self.assertEqual(base - pauli, -expected)
        self.assertEqual(pauli - base, expected)

        spo = SparsePauliOp(pauli_label)
        self.assertEqual(base - spo, -expected)
        with self.assertRaisesRegex(QiskitError, "Invalid input data for Pauli"):
            # This doesn't work because `SparsePauliOp` is badly behaved in its coercion (it gets
            # first dibs at `__add__`, not our `__radd__`), and will not return `NotImplemented` for
            # bad types.  This _shouldn't_ raise, and this test here is to remind us to flip it to a
            # proper assertion of correctness if `Pauli` starts playing nicely.
            _ = spo + base

        obs_label = "10+-rlXYZ"
        expected = SparseObservable.from_label(obs_label)
        self.assertEqual(base - obs_label, -expected)
        self.assertEqual(obs_label - base, expected)

        expected = 3j * SparseObservable.from_label("IXYrlII0I")
        self.assertEqual(base - expected[0], -expected)
        self.assertEqual(expected[0] - base, expected)

        with self.assertRaises(TypeError):
            _ = base - {}
        with self.assertRaises(TypeError):
            _ = {} - base
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = base - "$$$"
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = "$$$" - base

        self.assertIs(base + AllowRightArithmetic(), AllowRightArithmetic.SENTINEL)
        with self.assertRaisesRegex(TypeError, "invalid object for in-place subtraction"):
            # This actually _shouldn't_ be a `TypeError` - `__isub_` should defer to
            # `AllowRightArithmetic.__rsub__` in the same way that `__sub__` does, but a limitation
            # in PyO3 (see PyO3/pyo3#4605) prevents this.
            base -= AllowRightArithmetic()

    def test_sub_failures(self):
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            _ = SparseObservable.zero(4) - SparseObservable.zero(6)
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            _ = SparseObservable.zero(6) - SparseObservable.zero(4)

    @ddt.idata(single_cases())
    def test_neg(self, obs):
        initial = obs.copy()
        expected = obs.copy()
        expected.coeffs[:] = -np.asarray(expected.coeffs)
        self.assertEqual(-obs, expected)
        # Test that there's no in-place modification.
        self.assertEqual(obs, initial)

    @ddt.idata(single_cases())
    def test_pos(self, obs):
        initial = obs.copy()
        self.assertEqual(+obs, initial)
        self.assertIsNot(+obs, obs)

    @combine(left=single_cases(), right=single_cases())
    def test_tensor(self, left, right):

        def expected(left, right):
            coeffs = []
            bit_terms = []
            indices = []
            boundaries = [0]
            for left_ptr in range(left.num_terms):
                left_start, left_end = left.boundaries[left_ptr], left.boundaries[left_ptr + 1]
                for right_ptr in range(right.num_terms):
                    right_start = right.boundaries[right_ptr]
                    right_end = right.boundaries[right_ptr + 1]
                    coeffs.append(left.coeffs[left_ptr] * right.coeffs[right_ptr])
                    bit_terms.extend(right.bit_terms[right_start:right_end])
                    bit_terms.extend(left.bit_terms[left_start:left_end])
                    indices.extend(right.indices[right_start:right_end])
                    indices.extend(i + right.num_qubits for i in left.indices[left_start:left_end])
                    boundaries.append(len(indices))
            return SparseObservable.from_raw_parts(
                left.num_qubits + right.num_qubits, coeffs, bit_terms, indices, boundaries
            )

        # We deliberately have the arguments flipped when appropriate, here.

        left_initial = left.copy()
        right_initial = right.copy()
        self.assertEqual(left.tensor(right), expected(left, right))
        self.assertEqual(left, left_initial)
        self.assertEqual(right, right_initial)
        self.assertEqual(right.tensor(left), expected(right, left))

        self.assertEqual(left.expand(right), expected(right, left))
        self.assertEqual(left, left_initial)
        self.assertEqual(right, right_initial)
        self.assertEqual(right.expand(left), expected(left, right))

        self.assertEqual(left.tensor(right), right.expand(left))
        self.assertEqual(left.expand(right), right.tensor(left))

    @combine(
        obs=single_cases(), identity=[SparseObservable.identity(0), SparseObservable.identity(5)]
    )
    def test_tensor_identity(self, obs, identity):
        initial = obs.copy()
        expected_left = SparseObservable.from_raw_parts(
            obs.num_qubits + identity.num_qubits,
            obs.coeffs,
            obs.bit_terms,
            [x + identity.num_qubits for x in obs.indices],
            obs.boundaries,
        )
        expected_right = SparseObservable.from_raw_parts(
            obs.num_qubits + identity.num_qubits,
            obs.coeffs,
            obs.bit_terms,
            obs.indices,
            obs.boundaries,
        )
        self.assertEqual(obs.tensor(identity), expected_left)
        self.assertEqual(identity.tensor(obs), expected_right)
        self.assertEqual(obs.expand(identity), expected_right)
        self.assertEqual(identity.expand(obs), expected_left)
        self.assertEqual(obs ^ identity, expected_left)
        self.assertEqual(identity ^ obs, expected_right)
        self.assertEqual(obs, initial)
        obs ^= identity
        self.assertEqual(obs, expected_left)

    @combine(obs=single_cases(), zero=[SparseObservable.zero(0), SparseObservable.zero(5)])
    def test_tensor_zero(self, obs, zero):
        initial = obs.copy()
        expected = SparseObservable.zero(obs.num_qubits + zero.num_qubits)
        self.assertEqual(obs.tensor(zero), expected)
        self.assertEqual(zero.tensor(obs), expected)
        self.assertEqual(obs.expand(zero), expected)
        self.assertEqual(zero.expand(obs), expected)
        self.assertEqual(obs ^ zero, expected)
        self.assertEqual(zero ^ obs, expected)
        self.assertEqual(obs, initial)
        obs ^= zero
        self.assertEqual(obs, expected)

    def test_tensor_coercion(self):
        """Other quantum-info operators coerce with the ``tensor`` method and operator, so we do
        too."""
        base = SparseObservable.identity(0)

        pauli_label = "IIXYZII"
        expected = SparseObservable.from_label(pauli_label)
        self.assertEqual(base.tensor(pauli_label), expected)
        self.assertEqual(base.expand(pauli_label), expected)
        self.assertEqual(base ^ pauli_label, expected)
        self.assertEqual(pauli_label ^ base, expected)

        pauli = Pauli(pauli_label)
        self.assertEqual(base.tensor(pauli), expected)
        self.assertEqual(base.expand(pauli), expected)
        self.assertEqual(base ^ pauli, expected)
        with self.assertRaisesRegex(QiskitError, "Invalid input data for Pauli"):
            # This doesn't work because `Pauli` is badly behaved in its coercion (it gets first dibs
            # at `__xor__`, not our `__rxor__`), and will not return `NotImplemented` for bad types.
            # This _shouldn't_ raise, and this test here is to remind us to flip it to a proper
            # assertion of correctness if `Pauli` starts playing nicely.
            _ = pauli ^ base

        spo = SparsePauliOp(pauli_label)
        self.assertEqual(base.tensor(spo), expected)
        self.assertEqual(base.expand(spo), expected)
        self.assertEqual(base ^ spo, expected)
        with self.assertRaisesRegex(QiskitError, "Invalid input data for Pauli"):
            # This doesn't work because `SparsePauliOp` is badly behaved in its coercion (it gets
            # first dibs at `__xor__`, not our `__rxor__`), and will not return `NotImplemented` for
            # bad types.  This _shouldn't_ raise, and this test here is to remind us to flip it to a
            # proper assertion of correctness if `Pauli` starts playing nicely.
            _ = spo ^ base

        obs_label = "10+-rlXYZ"
        expected = SparseObservable.from_label(obs_label)
        self.assertEqual(base.tensor(obs_label), expected)
        self.assertEqual(base.expand(obs_label), expected)
        self.assertEqual(base ^ obs_label, expected)
        self.assertEqual(obs_label ^ base, expected)

        with self.assertRaises(TypeError):
            _ = base ^ {}
        with self.assertRaises(TypeError):
            _ = {} ^ base
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = base ^ "$$$"
        with self.assertRaisesRegex(ValueError, "only contain letters from the alphabet"):
            _ = "$$$" ^ base

        self.assertIs(base ^ AllowRightArithmetic(), AllowRightArithmetic.SENTINEL)

    @ddt.idata(single_cases())
    def test_adjoint(self, obs):
        initial = obs.copy()
        expected = obs.copy()
        expected.coeffs[:] = np.conjugate(expected.coeffs)
        self.assertEqual(obs.adjoint(), expected)
        self.assertEqual(obs, initial)
        self.assertEqual(obs.adjoint().adjoint(), initial)
        self.assertEqual(obs.adjoint(), obs.conjugate().transpose())
        self.assertEqual(obs.adjoint(), obs.transpose().conjugate())

    @ddt.idata(single_cases())
    def test_conjugate(self, obs):
        initial = obs.copy()

        term_map = {term: (term, 1.0) for term in SparseObservable.BitTerm}
        term_map[SparseObservable.BitTerm.Y] = (SparseObservable.BitTerm.Y, -1.0)
        term_map[SparseObservable.BitTerm.RIGHT] = (SparseObservable.BitTerm.LEFT, 1.0)
        term_map[SparseObservable.BitTerm.LEFT] = (SparseObservable.BitTerm.RIGHT, 1.0)

        expected = obs.copy()
        for i in range(expected.num_terms):
            start, end = expected.boundaries[i], expected.boundaries[i + 1]
            coeff = expected.coeffs[i]
            for offset, bit_term in enumerate(expected.bit_terms[start:end]):
                new_term, multiplier = term_map[bit_term]
                coeff *= multiplier
                expected.bit_terms[start + offset] = new_term
            expected.coeffs[i] = coeff.conjugate()

        self.assertEqual(obs.conjugate(), expected)
        self.assertEqual(obs, initial)
        self.assertEqual(obs.conjugate().conjugate(), initial)
        self.assertEqual(obs.conjugate(), obs.transpose().adjoint())
        self.assertEqual(obs.conjugate(), obs.adjoint().transpose())

    def test_conjugate_explicit(self):
        # The description of conjugation on the operator is not 100% trivial to see is correct, so
        # here's an explicit case to verify.
        obs = SparseObservable.from_sparse_list(
            [
                ("Y", (1,), 2.0),
                ("X+-", (5, 4, 3), 1.5),
                ("Z01", (5, 4, 3), 1.5j),
                ("YY", (2, 0), 0.25),
                ("YY", (3, 1), 0.25j),
                ("YYY", (3, 2, 1), 0.75),
                ("rlrl", (4, 3, 2, 1), 1.0),
                ("lrlr", (4, 3, 2, 1), 1.0j),
                ("", (), 1.5j),
            ],
            num_qubits=6,
        )
        expected = SparseObservable.from_sparse_list(
            [
                ("Y", (1,), -2.0),
                ("X+-", (5, 4, 3), 1.5),
                ("Z01", (5, 4, 3), -1.5j),
                ("YY", (2, 0), 0.25),
                ("YY", (3, 1), -0.25j),
                ("YYY", (3, 2, 1), -0.75),
                ("lrlr", (4, 3, 2, 1), 1.0),
                ("rlrl", (4, 3, 2, 1), -1.0j),
                ("", (), -1.5j),
            ],
            num_qubits=6,
        )
        self.assertEqual(obs.conjugate(), expected)
        self.assertEqual(obs.conjugate().conjugate(), obs)

    @ddt.idata(single_cases())
    def test_transpose(self, obs):
        initial = obs.copy()

        term_map = {term: (term, 1.0) for term in SparseObservable.BitTerm}
        term_map[SparseObservable.BitTerm.Y] = (SparseObservable.BitTerm.Y, -1.0)
        term_map[SparseObservable.BitTerm.RIGHT] = (SparseObservable.BitTerm.LEFT, 1.0)
        term_map[SparseObservable.BitTerm.LEFT] = (SparseObservable.BitTerm.RIGHT, 1.0)

        expected = obs.copy()
        for i in range(expected.num_terms):
            start, end = expected.boundaries[i], expected.boundaries[i + 1]
            coeff = expected.coeffs[i]
            for offset, bit_term in enumerate(expected.bit_terms[start:end]):
                new_term, multiplier = term_map[bit_term]
                coeff *= multiplier
                expected.bit_terms[start + offset] = new_term
            expected.coeffs[i] = coeff

        self.assertEqual(obs.transpose(), expected)
        self.assertEqual(obs, initial)
        self.assertEqual(obs.transpose().transpose(), initial)
        self.assertEqual(obs.transpose(), obs.conjugate().adjoint())
        self.assertEqual(obs.transpose(), obs.adjoint().conjugate())

    def test_transpose_explicit(self):
        # The description of transposition on the operator is not 100% trivial to see is correct, so
        # here's a few explicit cases to verify.
        obs = SparseObservable.from_sparse_list(
            [
                ("Y", (1,), 2.0),
                ("X+-", (5, 4, 3), 1.5),
                ("Z01", (5, 4, 3), 1.5j),
                ("YY", (2, 0), 0.25),
                ("YY", (3, 1), 0.25j),
                ("YYY", (3, 2, 1), 0.75),
                ("rlrl", (4, 3, 2, 1), 1.0),
                ("lrlr", (4, 3, 2, 1), 1.0j),
                ("", (), 1.5j),
            ],
            num_qubits=6,
        )
        expected = SparseObservable.from_sparse_list(
            [
                ("Y", (1,), -2.0),
                ("X+-", (5, 4, 3), 1.5),
                ("Z01", (5, 4, 3), 1.5j),
                ("YY", (2, 0), 0.25),
                ("YY", (3, 1), 0.25j),
                ("YYY", (3, 2, 1), -0.75),
                ("lrlr", (4, 3, 2, 1), 1.0),
                ("rlrl", (4, 3, 2, 1), 1.0j),
                ("", (), 1.5j),
            ],
            num_qubits=6,
        )
        self.assertEqual(obs.transpose(), expected)
        self.assertEqual(obs.transpose().transpose(), obs)

    def test_simplify(self):
        self.assertEqual((1e-10 * SparseObservable("XX")).simplify(1e-8), SparseObservable.zero(2))
        self.assertEqual((1e-10j * SparseObservable("XX")).simplify(1e-8), SparseObservable.zero(2))
        self.assertEqual(
            (1e-7 * SparseObservable("XX")).simplify(1e-8), 1e-7 * SparseObservable("XX")
        )

        exact_coeff = 2.0**-10
        self.assertEqual(
            (exact_coeff * SparseObservable("XX")).simplify(exact_coeff), SparseObservable.zero(2)
        )
        self.assertEqual(
            (exact_coeff * 1j * SparseObservable("XX")).simplify(exact_coeff),
            SparseObservable.zero(2),
        )
        coeff = 3e-5 + 4e-5j
        self.assertEqual(
            (coeff * SparseObservable("ZZ")).simplify(abs(coeff)), SparseObservable.zero(2)
        )

        sum_alike = SparseObservable.from_list(
            [
                ("XX", 1.0),
                ("YY", 1j),
                ("XX", -1.0),
            ]
        )
        self.assertEqual(sum_alike.simplify(), 1j * SparseObservable("YY"))

        terms = [
            ("XYIZI", 1.5),
            ("+-IYI", 2.0),
            ("XYIZI", 2j),
            ("+-IYI", -2.0),
            ("rlIZI", -2.0),
        ]
        canonical_forwards = SparseObservable.from_list(terms)
        canonical_backwards = SparseObservable.from_list(list(reversed(terms)))
        self.assertNotEqual(canonical_forwards.simplify(), canonical_forwards)
        self.assertNotEqual(canonical_forwards, canonical_backwards)
        self.assertEqual(canonical_forwards.simplify(), canonical_backwards.simplify())
        self.assertEqual(canonical_forwards.simplify(), canonical_forwards.simplify().simplify())

    @ddt.idata(single_cases())
    def test_clear(self, obs):
        num_qubits = obs.num_qubits
        obs.clear()
        self.assertEqual(obs, SparseObservable.zero(num_qubits))

    def test_apply_layout_list(self):
        self.assertEqual(
            SparseObservable.zero(5).apply_layout([4, 3, 2, 1, 0]), SparseObservable.zero(5)
        )
        self.assertEqual(
            SparseObservable.zero(3).apply_layout([0, 2, 1], 8), SparseObservable.zero(8)
        )
        self.assertEqual(
            SparseObservable.identity(2).apply_layout([1, 0]), SparseObservable.identity(2)
        )
        self.assertEqual(
            SparseObservable.identity(3).apply_layout([100, 10_000, 3], 100_000_000),
            SparseObservable.identity(100_000_000),
        )

        terms = [
            ("ZYX", (4, 2, 1), 1j),
            ("", (), -0.5),
            ("+-rl01", (10, 8, 6, 4, 2, 0), 2.0),
        ]

        def map_indices(terms, layout):
            return [
                (terms, tuple(layout[bit] for bit in bits), coeff) for terms, bits, coeff in terms
            ]

        identity = list(range(12))
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(identity),
            SparseObservable.from_sparse_list(terms, num_qubits=12),
        )
        # We've already tested elsewhere that `SparseObservable.from_sparse_list` produces termwise
        # sorted indices, so these tests also ensure `apply_layout` is maintaining that invariant.
        backwards = list(range(12))[::-1]
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(backwards),
            SparseObservable.from_sparse_list(map_indices(terms, backwards), num_qubits=12),
        )
        shuffled = [4, 7, 1, 10, 0, 11, 3, 2, 8, 5, 6, 9]
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled),
            SparseObservable.from_sparse_list(map_indices(terms, shuffled), num_qubits=12),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(shuffled, 100),
            SparseObservable.from_sparse_list(map_indices(terms, shuffled), num_qubits=100),
        )
        expanded = [78, 69, 82, 68, 32, 97, 108, 101, 114, 116, 33]
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=11).apply_layout(expanded, 120),
            SparseObservable.from_sparse_list(map_indices(terms, expanded), num_qubits=120),
        )

    def test_apply_layout_transpiled(self):
        base = SparseObservable.from_sparse_list(
            [
                ("ZYX", (4, 2, 1), 1j),
                ("", (), -0.5),
                ("+-r", (3, 2, 0), 2.0),
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
        self.assertEqual(SparseObservable.zero(0).apply_layout(None), SparseObservable.zero(0))
        self.assertEqual(SparseObservable.zero(0).apply_layout(None, 3), SparseObservable.zero(3))
        self.assertEqual(SparseObservable.zero(5).apply_layout(None), SparseObservable.zero(5))
        self.assertEqual(SparseObservable.zero(3).apply_layout(None, 8), SparseObservable.zero(8))
        self.assertEqual(
            SparseObservable.identity(0).apply_layout(None), SparseObservable.identity(0)
        )
        self.assertEqual(
            SparseObservable.identity(0).apply_layout(None, 8), SparseObservable.identity(8)
        )
        self.assertEqual(
            SparseObservable.identity(2).apply_layout(None), SparseObservable.identity(2)
        )
        self.assertEqual(
            SparseObservable.identity(3).apply_layout(None, 100_000_000),
            SparseObservable.identity(100_000_000),
        )

        terms = [
            ("ZYX", (2, 1, 0), 1j),
            ("", (), -0.5),
            ("+-rl01", (10, 8, 6, 4, 2, 0), 2.0),
        ]
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(None),
            SparseObservable.from_sparse_list(terms, num_qubits=12),
        )
        self.assertEqual(
            SparseObservable.from_sparse_list(terms, num_qubits=12).apply_layout(
                None, num_qubits=200
            ),
            SparseObservable.from_sparse_list(terms, num_qubits=200),
        )

    def test_apply_layout_failures(self):
        obs = SparseObservable.from_list([("IIYI", 2.0), ("IIIX", -1j)])
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

    def test_pauli_bases(self):
        obs = SparseObservable.from_list(
            [
                ("IIIII", 1.0),
                ("IXYZI", 2.0),
                ("+-II+", 1j),
                ("rlrlr", -0.5),
                ("01010", -0.25),
                ("rlYII", 1.0),
            ]
        )
        expected = PauliList(
            [
                Pauli("IIIII"),
                Pauli("IXYZI"),
                Pauli("XXIIX"),
                Pauli("YYYYY"),
                Pauli("ZZZZZ"),
                Pauli("YYYII"),
            ]
        )
        self.assertEqual(obs.pauli_bases(), expected)

    def test_iteration(self):
        self.assertEqual(list(SparseObservable.zero(5)), [])
        self.assertEqual(tuple(SparseObservable.zero(0)), ())

        obs = SparseObservable.from_sparse_list(
            [
                ("Xrl", (4, 2, 1), 2j),
                ("", (), 0.5),
                ("01", (3, 0), -0.25),
                ("+-", (2, 1), 1.0),
                ("YZ", (4, 1), 1j),
            ],
            num_qubits=5,
        )
        bit_term = SparseObservable.BitTerm
        expected = [
            SparseObservable.Term(5, 2j, [bit_term.LEFT, bit_term.RIGHT, bit_term.X], [1, 2, 4]),
            SparseObservable.Term(5, 0.5, [], []),
            SparseObservable.Term(5, -0.25, [bit_term.ONE, bit_term.ZERO], [0, 3]),
            SparseObservable.Term(5, 1.0, [bit_term.MINUS, bit_term.PLUS], [1, 2]),
            SparseObservable.Term(5, 1j, [bit_term.Z, bit_term.Y], [1, 4]),
        ]
        self.assertEqual(list(obs), expected)

    def test_indexing(self):
        obs = SparseObservable.from_sparse_list(
            [
                ("Xrl", (4, 2, 1), 2j),
                ("", (), 0.5),
                ("01", (3, 0), -0.25),
                ("+-", (2, 1), 1.0),
                ("YZ", (4, 1), 1j),
            ],
            num_qubits=5,
        )
        bit_term = SparseObservable.BitTerm
        expected = [
            SparseObservable.Term(5, 2j, [bit_term.LEFT, bit_term.RIGHT, bit_term.X], [1, 2, 4]),
            SparseObservable.Term(5, 0.5, [], []),
            SparseObservable.Term(5, -0.25, [bit_term.ZERO, bit_term.ONE], [3, 0]),
            SparseObservable.Term(5, 1.0, [bit_term.MINUS, bit_term.PLUS], [1, 2]),
            SparseObservable.Term(5, 1j, [bit_term.Y, bit_term.Z], [4, 1]),
        ]
        self.assertEqual(obs[0], expected[0])
        self.assertEqual(obs[-2], expected[-2])
        self.assertEqual(obs[2:4], SparseObservable(expected[2:4]))
        self.assertEqual(obs[1::2], SparseObservable(expected[1::2]))
        self.assertEqual(obs[:], SparseObservable(expected))
        self.assertEqual(obs[-1:-4:-1], SparseObservable(expected[-1:-4:-1]))

    @ddt.data(
        SparseObservable.identity(0),
        SparseObservable.identity(1_000),
        SparseObservable.from_label("IIXIZI"),
        SparseObservable.from_label("X"),
        SparseObservable.from_list([("YIXZII", -0.25)]),
        SparseObservable.from_list([("01rl+-", 0.25 + 0.5j)]),
    )
    def test_term_repr(self, obs):
        # The purpose of this is just to test that the `repr` doesn't crash, rather than asserting
        # that it has any particular form.
        term = obs[0]
        self.assertIsInstance(repr(term), str)
        self.assertIn("SparseObservable.Term", repr(term))

    @ddt.data(
        SparseObservable.identity(0),
        2j * SparseObservable.identity(1),
        SparseObservable.identity(100),
        SparseObservable.from_label("IIX+-rlYZ01IIIII"),
    )
    def test_term_to_observable(self, obs):
        self.assertEqual(obs[0].to_observable(), obs)
        self.assertIsNot(obs[0].to_observable(), obs)

    def test_term_equality(self):
        self.assertEqual(
            SparseObservable.Term(5, 1.0, [], []), SparseObservable.Term(5, 1.0, [], [])
        )
        self.assertNotEqual(
            SparseObservable.Term(5, 1.0, [], []), SparseObservable.Term(8, 1.0, [], [])
        )
        self.assertNotEqual(
            SparseObservable.Term(5, 1.0, [], []), SparseObservable.Term(5, 1j, [], [])
        )
        self.assertNotEqual(
            SparseObservable.Term(5, 1.0, [], []), SparseObservable.Term(8, -1, [], [])
        )

        obs = SparseObservable.from_list(
            [
                ("IIXIZ", 2j),
                ("IIZIX", 2j),
                ("++III", -1.5),
                ("--III", -1.5),
                ("IrIlI", 0.5),
                ("IIrIl", 0.5),
            ]
        )
        self.assertEqual(obs[0], obs[0])
        self.assertEqual(obs[1], obs[1])
        self.assertNotEqual(obs[0], obs[1])
        self.assertEqual(obs[2], obs[2])
        self.assertEqual(obs[3], obs[3])
        self.assertNotEqual(obs[2], obs[3])
        self.assertEqual(obs[4], obs[4])
        self.assertEqual(obs[5], obs[5])
        self.assertNotEqual(obs[4], obs[5])

    @ddt.data(
        SparseObservable.identity(0),
        2j * SparseObservable.identity(1),
        SparseObservable.identity(100),
        SparseObservable.from_label("IIX+-rlYZ01IIIII"),
    )
    def test_term_pickle(self, obs):
        term = obs[0]
        self.assertEqual(pickle.loads(pickle.dumps(term)), term)
        self.assertEqual(copy.copy(term), term)
        self.assertEqual(copy.deepcopy(term), term)

    def test_term_attributes(self):
        term = SparseObservable.from_label("II+IIX0")[0]
        self.assertEqual(term.num_qubits, 7)
        self.assertEqual(term.coeff, 1.0)
        np.testing.assert_equal(
            term.bit_terms,
            np.array(
                [
                    SparseObservable.BitTerm.ZERO,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.PLUS,
                ],
                dtype=np.uint8,
            ),
        )
        np.testing.assert_equal(term.indices, np.array([0, 1, 4], dtype=np.uintp))

        term = SparseObservable.identity(10)[0]
        self.assertEqual(term.num_qubits, 10)
        self.assertEqual(term.coeff, 1.0)
        self.assertEqual(list(term.bit_terms), [])
        self.assertEqual(list(term.indices), [])

        term = SparseObservable.from_list([("IIrlZ", 0.5j)])[0]
        self.assertEqual(term.num_qubits, 5)
        self.assertEqual(term.coeff, 0.5j)
        self.assertEqual(
            list(term.bit_terms),
            [
                SparseObservable.BitTerm.Z,
                SparseObservable.BitTerm.LEFT,
                SparseObservable.BitTerm.RIGHT,
            ],
        )
        self.assertEqual(list(term.indices), [0, 1, 2])

    def test_term_new(self):
        expected = SparseObservable.from_label("IIIX+1III")[0]

        self.assertEqual(
            SparseObservable.Term(
                9,
                1.0,
                [
                    SparseObservable.BitTerm.ONE,
                    SparseObservable.BitTerm.PLUS,
                    SparseObservable.BitTerm.X,
                ],
                [3, 4, 5],
            ),
            expected,
        )

        # Constructor should allow being given unsorted inputs, and but them in the right order.
        self.assertEqual(
            SparseObservable.Term(
                9,
                1.0,
                [
                    SparseObservable.BitTerm.PLUS,
                    SparseObservable.BitTerm.X,
                    SparseObservable.BitTerm.ONE,
                ],
                [4, 5, 3],
            ),
            expected,
        )
        self.assertEqual(list(expected.indices), [3, 4, 5])

        with self.assertRaisesRegex(ValueError, "not term-wise increasing"):
            SparseObservable.Term(2, 2j, [SparseObservable.BitTerm.RIGHT] * 2, [0, 0])

    def test_term_pauli_base(self):
        obs = SparseObservable.from_list(
            [
                ("IIIII", 1.0),
                ("IXYZI", 2.0),
                ("+-II+", 1j),
                ("rlrlr", -0.5),
                ("01010", -0.25),
                ("rlYII", 1.0),
            ]
        )
        expected = [
            Pauli("IIIII"),
            Pauli("IXYZI"),
            Pauli("XXIIX"),
            Pauli("YYYYY"),
            Pauli("ZZZZZ"),
            Pauli("YYYII"),
        ]
        self.assertEqual([term.pauli_base() for term in obs], expected)

    def test_to_sparse_list(self):
        """Test converting to a sparse list."""
        with self.subTest(msg="zero"):
            obs = SparseObservable.zero(100)
            expected = []
            self.assertEqual(expected, obs.to_sparse_list())

        with self.subTest(msg="identity"):
            obs = SparseObservable.identity(100)
            expected = [("", [], 1)]
            self.assertEqual(expected, obs.to_sparse_list())

        with self.subTest(msg="IXYZ"):
            obs = SparseObservable("IXYZ")
            expected = [("ZYX", [0, 1, 2], 1)]
            self.assertEqual(
                canonicalize_sparse_list(expected), canonicalize_sparse_list(obs.to_sparse_list())
            )

        with self.subTest(msg="multiple"):
            obs = SparseObservable.from_list([("lrI0", 0.5), ("YYIZ", -1j)])
            expected = [("lr0", [3, 2, 0], 0.5), ("ZYY", [0, 2, 3], -1j)]
            self.assertEqual(
                canonicalize_sparse_list(expected), canonicalize_sparse_list(obs.to_sparse_list())
            )

    def test_as_paulis(self):
        """Test converting to Paulis."""
        # test on zero operator
        with self.subTest(msg="zero"):
            obs = SparseObservable.zero(10)
            obs_paulis = obs.as_paulis()
            self.assertEqual(obs, obs_paulis)

        # test on identity operator
        with self.subTest(msg="identity"):
            obs = SparseObservable.identity(10)
            obs_paulis = obs.as_paulis()
            self.assertEqual(obs, obs_paulis)

        # test it does nothing on Paulis
        with self.subTest(msg="paulis"):
            obs = SparseObservable.from_list([("IIX", 1), ("ZZY", -1)])
            obs_paulis = obs.as_paulis()
            self.assertEqual(obs, obs_paulis)

        # test multiple +1 projectors
        with self.subTest(msg="00"):
            obs = SparseObservable("00")
            obs_paulis = obs.as_paulis()
            expected = SparseObservable.from_sparse_list(
                [
                    ("", [], 1 / 4),
                    ("Z", [0], 1 / 4),
                    ("Z", [1], 1 / 4),
                    ("ZZ", [0, 1], 1 / 4),
                ],
                2,
            )
            self.assertEqual(expected.simplify(), obs_paulis.simplify())

        # test multiple -1 projectors
        with self.subTest(msg="11"):
            obs = SparseObservable("11")
            obs_paulis = obs.as_paulis()
            expected = SparseObservable.from_sparse_list(
                [
                    ("", [], 1 / 4),
                    ("Z", [0], -1 / 4),
                    ("Z", [1], -1 / 4),
                    ("ZZ", [0, 1], 1 / 4),
                ],
                2,
            )
            self.assertEqual(expected.simplify(), obs_paulis.simplify())

        # test +1 -1 projector
        with self.subTest(msg="01"):
            obs = SparseObservable("01")
            obs_paulis = obs.as_paulis()
            expected = SparseObservable.from_sparse_list(
                [
                    ("", [], 1 / 4),
                    ("Z", [0], -1 / 4),
                    ("Z", [1], 1 / 4),
                    ("ZZ", [0, 1], -1 / 4),
                ],
                2,
            )
            self.assertEqual(expected.simplify(), obs_paulis.simplify())

        # test multiple negative projectors with a positive
        with self.subTest(msg="011"):
            obs = SparseObservable("011")
            obs_paulis = obs.as_paulis()
            expected = SparseObservable.from_sparse_list(
                [
                    ("", [], 1 / 8),
                    ("Z", [0], -1 / 8),
                    ("Z", [1], -1 / 8),
                    ("Z", [2], 1 / 8),
                    ("ZZ", [0, 1], 1 / 8),
                    ("ZZ", [0, 2], -1 / 8),
                    ("ZZ", [1, 2], -1 / 8),
                    ("ZZZ", [0, 1, 2], 1 / 8),
                ],
                3,
            )
            self.assertEqual(expected.simplify(), obs_paulis.simplify())

        # test explicitly on written-out projector
        with self.subTest(msg="lrI0"):
            obs = SparseObservable("lrI0")
            obs_paulis = obs.as_paulis()
            expected = SparseObservable.from_sparse_list(
                [
                    ("", [], 1 / 8),
                    ("Y", [2], 1 / 8),
                    ("YY", [3, 2], -1 / 8),
                    ("Z", [0], 1 / 8),
                    ("YZ", [2, 0], 1 / 8),
                    ("YYZ", [3, 2, 0], -1 / 8),
                    ("Y", [3], -1 / 8),
                    ("YZ", [3, 0], -1 / 8),
                ],
                4,
            )
            self.assertEqual(expected.simplify(), obs_paulis.simplify())

        # test multiple terms
        with self.subTest(msg="+X + lY - ZI"):
            obs = SparseObservable.from_list([("+X", 1), ("lY", 1), ("ZI", -1)])
            obs_paulis = obs.as_paulis()

            expected = SparseObservable.from_list(
                [("IX", 0.5), ("XX", 0.5), ("IY", 0.5), ("YY", -0.5), ("ZI", -1)]
            )

            self.assertEqual(expected.simplify(), obs_paulis.simplify())

    def test_sparse_list_roundtrip(self):
        """Test dumping into a sparse list and constructing from one."""
        obs = SparseObservable.from_list(
            [
                ("IIXIZ", 2j),
                ("IIZIX", 2j),
                ("++III", -1.5),
                ("--III", -1.5),
                ("IrIlI", 0.5),
                ("IIrIl", 0.5),
            ]
        )

        reconstructed = SparseObservable.from_sparse_list(obs.to_sparse_list(), obs.num_qubits)
        self.assertEqual(obs.simplify(), reconstructed.simplify())

    def test_sparse_term_bit_labels(self):
        """Test getting the bit labels of a SparseTerm."""

        obs = SparseObservable("IXYZ+r0-l1")
        term = obs[0]
        indices = term.indices
        labels = term.bit_labels()

        label_dict = dict(zip(indices, labels))
        expected = dict(enumerate("1l-0r+ZYX"))

        for i, label in expected.items():
            self.assertEqual(label, label_dict[i])

        reconstructed = SparseObservable.from_sparse_list([(labels, indices, 1)], obs.num_qubits)
        self.assertEqual(obs, reconstructed)

    @combine(first=SparseObservable.BitTerm, second=SparseObservable.BitTerm)
    def test_compose_single_qubit(self, first, second):
        first = SparseObservable.from_label(first.label)
        second = SparseObservable.from_label(second.label)
        self.assertEqual(
            SparsePauliOp.from_sparse_observable(first.compose(second)).simplify().sort(),
            SparsePauliOp.from_sparse_observable(first)
            .compose(SparsePauliOp.from_sparse_observable(second))
            .simplify()
            .sort(),
        )

    @ddt.idata(SparseObservable.BitTerm)
    def test_compose_single_qubit_identity(self, term):
        ident = SparseObservable.identity(1)
        single_term = SparseObservable(term.label)
        self.assertEqual(ident.compose(single_term), single_term)
        self.assertEqual(single_term.compose(ident), single_term)

        num_qubits = 5
        big_ident = SparseObservable.identity(num_qubits)
        for i in range(num_qubits):
            big_term = SparseObservable.from_sparse_list(
                [(term.label, (i,), 2.0)], num_qubits=num_qubits
            )
            self.assertEqual(big_ident.compose(big_term), big_term)
            self.assertEqual(big_term.compose(big_ident), big_term)

    def test_compose_with_zero_operator(self):
        zero_10 = SparseObservable.zero(10)
        self.assertEqual(zero_10.compose(SparseObservable("IIX+rIIZ01")), zero_10)
        self.assertEqual(SparseObservable("ZXYII+-01l").compose(zero_10), zero_10)

        zero_0 = SparseObservable.zero(0)
        self.assertEqual(zero_0.compose(zero_0), zero_0)
        self.assertEqual(zero_0.compose(SparseObservable.identity(0)), zero_0)
        self.assertEqual(SparseObservable.identity(0).compose(zero_0), zero_0)

    def test_compose_multiplies_identity_coefficients(self):
        a, b = 1.5j, -1.25 + 3j
        ident = SparseObservable.identity(5)
        self.assertEqual((a * ident).compose(b * ident), a * b * ident)

    def test_compose_term_expansion(self):
        left = SparseObservable("+-rl01")
        right = SparseObservable("YZZXXY")
        self.assertEqual(
            SparsePauliOp.from_sparse_observable(left.compose(right)).simplify().sort(),
            SparsePauliOp.from_sparse_observable(left)
            .compose(SparsePauliOp.from_sparse_observable(right))
            .simplify()
            .sort(),
        )

    def test_compose_expands_sums(self):
        one = SparseObservable.from_list([("IIXIZ", 1.5), ("IY+II", 2j)])
        two = SparseObservable.from_list([("I+ZIr", -0.25), ("01rl+", 0.75j)])
        op_one = SparsePauliOp.from_sparse_observable(one)
        op_two = SparsePauliOp.from_sparse_observable(two)
        self.assertEqual(
            SparsePauliOp.from_sparse_observable(one.compose(two)).simplify().sort(),
            op_one.compose(op_two).simplify().sort(),
        )
        self.assertEqual(
            SparsePauliOp.from_sparse_observable(two.compose(one)).simplify().sort(),
            op_two.compose(op_one).simplify().sort(),
        )

    def test_compose_coerces(self):
        base = SparseObservable("IXYZ")
        label = "rl+-"
        self.assertEqual(base.compose(label), base.compose(SparseObservable(label)))
        self.assertEqual(base.compose(label, front=True), SparseObservable(label).compose(base))

    def test_compose_scalar_observable(self):
        """Edge-case test where one operand is 0q."""
        a, b = 1.5, -0.25j
        scalar = SparseObservable.identity(0)
        self.assertEqual((a * scalar).compose(b * scalar), a * b * scalar)
        self.assertEqual((a * scalar).compose(b * scalar, front=True), a * b * scalar)
        self.assertEqual((a * scalar).compose(b * scalar, qargs=[]), a * b * scalar)
        self.assertEqual((a * scalar).compose(b * scalar, qargs=[], front=True), a * b * scalar)

    def test_compose_scalar_observable_onto_nonscalar(self):
        a = -1.5
        scalar = a * SparseObservable.identity(0)
        base = SparseObservable.from_list([("IXY+l", 0.75), ("ZXYII", -0.5j)])
        self.assertEqual(base.compose(scalar, qargs=[]), a * base)
        self.assertEqual(base.compose(scalar, front=True, qargs=[]), a * base)

    def test_compose_front(self):
        one = SparseObservable.from_list([("IIXIZ", 1.5), ("IY+II", 2j)])
        two = SparseObservable.from_list([("I+ZIr", -0.25), ("01rl+", 0.75j)])
        self.assertEqual(one.compose(two, front=True), two.compose(one))

    def test_compose_front_qargs(self):
        qargs = [4, 3, 8]
        one = SparseObservable.from_sparse_list(
            [("XYZ", (4, 2, 1), 2.5), ("+-rl01", (9, 7, 5, 3, 2, 1), 1.25j)], num_qubits=10
        )
        two = SparseObservable.from_list([("X+Z", -0.5), ("01l", 0.25j)])
        self.assertEqual(
            one.compose(two, front=True, qargs=qargs),
            two.apply_layout(qargs, num_qubits=10).compose(one),
        )

    def test_compose_qargs(self):
        qargs = [4, 3, 8]
        one = SparseObservable.from_sparse_list(
            [("XYZ", (4, 2, 1), 2.5), ("+-rl01", (9, 7, 5, 3, 2, 1), 1.25j)], num_qubits=10
        )
        two = SparseObservable.from_list([("X+Z", -0.5), ("01l", 0.25j)])
        self.assertEqual(
            one.compose(two, qargs=qargs),
            one.compose(two.apply_layout(qargs, num_qubits=10)),
        )

    def test_compose_coercion_failure(self):
        with self.assertRaisesRegex(TypeError, "unknown type for compose"):
            SparseObservable.identity(5).compose(None)
        with self.assertRaisesRegex(TypeError, "unknown type for compose"):
            SparseObservable.identity(5).compose(...)

    def test_compose_failure_qubit_mismatch(self):
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            SparseObservable.identity(5).compose(SparseObservable.identity(8))
        with self.assertRaisesRegex(ValueError, "mismatched numbers of qubits"):
            SparseObservable.identity(7).compose(SparseObservable.identity(2))

    def test_compose_qargs_failure_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "input qargs has length"):
            SparseObservable.identity(5).compose(SparseObservable.identity(2), qargs=[0, 1, 2])
        with self.assertRaisesRegex(ValueError, "input qargs has length"):
            SparseObservable.identity(5).compose(SparseObservable.identity(2), qargs=[0])

    def test_compose_qargs_failure_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "qargs contains out-of-range qubits"):
            SparseObservable.identity(5).compose("XYZ", qargs=[0, 1, 5])
        with self.assertRaisesRegex(ValueError, "qargs contains out-of-range qubits"):
            SparseObservable.identity(5).compose("XYZ", qargs=[9, 1, 0])

    def test_compose_qargs_failure_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate indices in qargs"):
            SparseObservable.identity(5).compose("XX", qargs=[0, 0])
        with self.assertRaisesRegex(ValueError, "duplicate indices in qargs"):
            SparseObservable.identity(5).compose("XYZX", qargs=[0, 1, 2, 0])
        with self.assertRaisesRegex(ValueError, "duplicate indices in qargs"):
            SparseObservable.identity(5).compose("XYZX", qargs=[0, 1, 1, 0])

    @ddt.data("X", "Y", "Z", "I")
    def test_pauli_evolve_single_qubit(self, pauli_label):
        """
        Test that double conjugation of Q with P == Q evolved by P for
        single-qubit Pauli operators P and all Q.
        """
        p = Pauli(pauli_label)

        for q_label in ["+", "-", "r", "l", "0", "1", "I", "X", "Y", "Z"]:
            q = SparseObservable.from_label(q_label)

            # Convert Pauli to SparseObservable for composition
            p_obs = SparseObservable.from_label(pauli_label)
            result = p_obs.adjoint().compose(q).compose(p_obs).simplify()
            expected = q.evolve(p)

            # These could and should be replaced with :meth:`SparseObservable.matrix`
            # once relevant PR #15022 is merged.
            ma = SparsePauliOp.from_sparse_observable(result).to_matrix()
            mb = SparsePauliOp.from_sparse_observable(expected).to_matrix()

            np.testing.assert_allclose(ma, mb, atol=1e-10, rtol=0)

    def test_evolve_multi_qubit_pauli(self):
        """Test multi-qubit evolution with mixed observable and all Pauli operators."""
        obs = SparseObservable.from_list([("XY0I", 1.0), ("Z+rI", 0.5)])
        pauli = Pauli("IXYZ")

        # Expected result computed manually:
        # "XY0I" evolved by "IXYZ":
        #   I (qubit 0) by Z -> I
        #   0 (qubit 1) by Y -> 1
        #   Y (qubit 2) by X -> -Y
        #   X (qubit 3) by I -> X
        # Result: "XY1" with coeff 1.0
        #
        # "Z+rI" evolved by "IXYZ":
        #   I (qubit 0) by Z -> I
        #   r (qubit 1) by Y -> r
        #   + (qubit 2) by X -> +
        #   Z (qubit 3) by I -> Z
        # Result: "Z+r" with coeff 0.5

        expected = SparseObservable.from_list([("XY1I", -1.0), ("Z+rI", 0.5)])
        result = obs.evolve(pauli)
        self.assertEqual(result, expected)

    def test_evolve_qargs_subset(self):
        """Test evolving a subset of qubits."""
        obs = SparseObservable("XYZI")
        pauli = Pauli("YY")
        result = obs.evolve(pauli, qargs=[0, 1])

        # X evolved by Y is -X, Y evolved by Y is Y
        expected = SparseObservable.from_list([("XYZI", -1.0)])
        self.assertEqual(result, expected)

    def test_evolve_qargs_non_contiguous(self):
        """Test evolving non-contiguous qubits."""
        obs = SparseObservable.from_sparse_list([("XYZ", (0, 2, 4), 1.0)], num_qubits=5)
        pauli = Pauli("YZ")
        result = obs.evolve(pauli, qargs=[0, 4])

        # Z@0 by X is -X, Y@4 by Z is -Z
        expected = SparseObservable.from_sparse_list([("XYZ", (0, 2, 4), 1.0)], num_qubits=5)
        self.assertEqual(result, expected)

    def test_evolve_qargs_pauli_with_coeff(self):
        """Test evolution with Pauli operator with coefficient."""
        obs = SparseObservable("XYZ")
        pauli = -1j * Pauli("X")
        result = obs.evolve(pauli, qargs=[1])
        # With coefficient -1j: (1j) * (-XYZ) * (-1j) = -XYZ
        expected = -obs
        self.assertEqual(result, expected)

    def test_evolve_identity_with_coeff(self):
        """Test evolution with identity Pauli with coefficient (scalar)."""
        # There won't really be any change as Pauli Type objects only support
        # normalized coeffs. This test is just to check that the conditional
        # branch when encountered with a 0 qubit Scalar works.
        obs = SparseObservable.from_list([("XYZ", 1.0), ("ZXY", 0.5)])
        # Identity Pauli (0-qubit) with coefficient 1j
        pauli = 1j * Pauli("")
        result = obs.evolve(pauli, qargs=[])
        # With coefficient 1j: (-1j) * O * (1j) = 1 × O = O
        expected = obs
        self.assertEqual(result, expected)

    def test_evolve_qargs_wrong_length(self):
        """Test qargs length must match operator size."""
        obs = SparseObservable("XYZI")
        pauli = Pauli("XX")

        with self.assertRaisesRegex(
            ValueError, r"qargs has length 3, but operator has 2 qubit\(s\)"
        ):
            obs.evolve(pauli, qargs=[0, 1, 2])

        with self.assertRaisesRegex(
            ValueError, r"qargs has length 1, but operator has 2 qubit\(s\)"
        ):
            obs.evolve(pauli, qargs=[0])

    def test_evolve_qargs_out_of_range(self):
        """Test qargs must be valid qubit indices."""
        obs = SparseObservable("XYZ")
        pauli = Pauli("X")

        with self.assertRaisesRegex(ValueError, "qargs contains out-of-range qubits"):
            obs.evolve(pauli, qargs=[3])

    def test_evolve_qargs_duplicates(self):
        """Test qargs cannot have duplicates."""
        obs = SparseObservable("XYZX")
        pauli = Pauli("XX")

        with self.assertRaisesRegex(ValueError, "duplicate indices in qargs"):
            obs.evolve(pauli, qargs=[0, 0])

    def test_evolve_type_error(self):
        """Test that non-Pauli types raise TypeError."""
        obs = SparseObservable("XYZ")

        # Test with SparseObservable
        with self.assertRaisesRegex(TypeError, "evolve only accepts Pauli instances"):
            obs.evolve(SparseObservable("XYZ"))

        # Test with SparsePauliOp
        with self.assertRaisesRegex(TypeError, "evolve only accepts Pauli instances"):
            obs.evolve(SparsePauliOp.from_list([("X", 1.0)]))

        # Test with string
        with self.assertRaisesRegex(TypeError, "evolve only accepts Pauli instances"):
            obs.evolve("XYZ")

    def test_evolve_qubit_mismatch_no_qargs(self):
        """Test mismatched qubits fails without qargs."""
        obs = SparseObservable.identity(3)
        pauli = Pauli("IIIII")
        with self.assertRaisesRegex(ValueError, r"mismatched numbers of qubits: 3, 5"):
            obs.evolve(pauli)

    def test_evolve_qargs_with_sums(self):
        """Test qargs evolution with sum observables."""
        obs = SparseObservable.from_list([("XYZ", 1.0), ("ZXY", 0.5)])
        pauli = Pauli("Z")

        result = obs.evolve(pauli, qargs=[1])

        # Each term evolved independently
        term1 = SparseObservable.from_list([("XYZ", -1.0)])
        term2 = SparseObservable.from_list([("ZXY", -0.5)])
        expected = term1 + term2

        self.assertEqual(result, expected.simplify())

    @ddt.data(
        ("Y", "Y", True),
        ("Y", "Z", False),
        ("XX", "ZZ", True),
        ("0XX", "0ZZ", True),
        ("1XX", "0ZZ", True),
        ("+XX", "0ZZ", False),
    )
    @ddt.unpack
    def test_commute_single(self, p1, p2, expected):
        """Test commutation"""
        op1 = SparseObservable(p1)
        op2 = SparseObservable(p2)
        self.assertEqual(op1.commutes(op2), expected)

    def test_commute_sums(self):
        """Test commutation for sums of Paulis."""
        xz = SparseObservable.from_sparse_list([("X", [0], 1), ("Z", [0], 1)], num_qubits=3)
        zx = SparseObservable.from_sparse_list([("Z", [0], 1), ("X", [0], 1)], num_qubits=3)
        self.assertTrue(xz.commutes(zx))

        x = SparseObservable.from_sparse_list([("X", [0], 1)], num_qubits=3)
        self.assertFalse(xz.commutes(x))

        y = SparseObservable.from_sparse_list([("Y", [2], 1)], num_qubits=3)
        self.assertTrue(xz.commutes(y))

        n = 10
        x_pairs = SparseObservable.from_sparse_list(
            [("XX", [i, i + 1], 1) for i in range(n - 1)], num_qubits=n
        )
        z_pairs = SparseObservable.from_sparse_list(
            [("ZZ", [i, i + 1], 1) for i in range(n - 1)], num_qubits=n
        )
        self.assertTrue(x_pairs.commutes(x_pairs))
        self.assertFalse(z_pairs.commutes(x_pairs))

        xxyy = SparseObservable.from_sparse_list(
            [("XX", [0, 1], 1), ("YY", [0, 1], 1)], num_qubits=2
        )
        zz = SparseObservable.from_sparse_list([("ZZ", [0, 1], 1)], num_qubits=2)
        self.assertTrue(zz.commutes(xxyy))


def canonicalize_term(pauli, indices, coeff):
    # canonicalize a sparse list term by sorting by indices (which is unique as
    # indices cannot be repeated)
    idcs = np.argsort(indices)
    sorted_paulis = "".join(pauli[i] for i in idcs)
    return (sorted_paulis, np.asarray(indices)[idcs].tolist(), complex(coeff))


def canonicalize_sparse_list(sparse_list):
    # sort a sparse list representation by canonicalizing the terms and then applying
    # Python's built-in sort
    canonicalized_terms = [canonicalize_term(*term) for term in sparse_list]
    return sorted(canonicalized_terms)
