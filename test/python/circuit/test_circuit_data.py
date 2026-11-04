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

"""Test operations on circuit.data."""
import warnings
import pickle
import ddt

from qiskit._accelerate.circuit import CircuitData
from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    Parameter,
    CircuitInstruction,
    Operation,
    Qubit,
    Clbit,
)
from qiskit.circuit.classical import types, expr
from qiskit.circuit.library import HGate, XGate, CXGate, RXGate, Measure
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit import transpile


@ddt.ddt
class TestQuantumCircuitData(QiskitTestCase):
    """CircuitData (Rust) operation tests."""

    def test_add_qubit(self):
        """Test adding new and duplicate qubits."""
        qr = QuantumRegister(2)
        data = CircuitData(qubits=[qr[0]])
        data.add_qubit(qr[1])
        self.assertEqual(data.qubits, list(qr))

        # Test re-adding is disallowed by default.
        with self.assertRaisesRegex(ValueError, "cannot add object"):
            data.add_qubit(qr[0])

        # Make sure re-adding is allowed in non-strict mode
        # and does not change order.
        data.add_qubit(qr[0], strict=False)
        self.assertEqual(data.qubits, list(qr))

    def test_add_qubit_new_style(self):
        """Test adding new and duplicate new-style qubits."""
        qubits = [Qubit(), Qubit()]
        data = CircuitData(qubits=[qubits[0]])
        data.add_qubit(qubits[1])
        self.assertEqual(data.qubits, qubits)

        # Test re-adding is disallowed by default.
        with self.assertRaisesRegex(ValueError, "cannot add object"):
            data.add_qubit(qubits[0])

        # Make sure re-adding is allowed in non-strict mode
        # and does not change order.
        data.add_qubit(qubits[0], strict=False)
        self.assertEqual(data.qubits, qubits)

    def test_add_qubit_raises_warning_with_layout(self):
        """Test warning is raised when qubit is added to a circuit with layout"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0, 1)
        backend = GenericBackendV2(num_qubits=7)
        isa_qc = transpile(qc, backend)
        with self.assertWarnsRegex(UserWarning, "Trying to add bits .* layout"):
            isa_qc.add_bits([Qubit() for _ in range(1)])

    def test_add_quantum_register_raises_warning_with_layout(self):
        """Test warning is raised when quantum register is added to a circuit with layout"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0, 1)
        backend = GenericBackendV2(num_qubits=7)
        isa_qc = transpile(qc, backend)
        qr = QuantumRegister(1)
        with self.assertWarnsRegex(UserWarning, "Trying to add QuantumRegister .* layout"):
            isa_qc.add_register(qr)

    def test_add_clbit_not_raises_warning_with_layout(self):
        """Test warning is not raised when clbit is added to a circuit with layout"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.rx(0, 1)
        backend = GenericBackendV2(num_qubits=7)
        isa_qc = transpile(qc, backend)
        with warnings.catch_warnings(record=True) as warnings_log:
            isa_qc.add_bits([Clbit() for _ in range(1)])

        self.assertEqual(warnings_log, [])

    def test_add_clbit(self):
        """Test adding new and duplicate clbits."""
        cr = ClassicalRegister(2)
        data = CircuitData(clbits=[cr[0]])
        data.add_clbit(cr[1])
        self.assertEqual(data.clbits, list(cr))

        # Test re-adding is disallowed by default.
        with self.assertRaisesRegex(ValueError, "cannot add object"):
            data.add_clbit(cr[0])

        # Make sure re-adding is allowed in non-strict mode
        # and does not change order.
        data.add_clbit(cr[0], strict=False)
        self.assertEqual(data.clbits, list(cr))

    def test_add_clbit_new_style(self):
        """Test adding new and duplicate new-style clbits."""
        clbits = [Clbit(), Clbit()]
        data = CircuitData(clbits=[clbits[0]])
        data.add_clbit(clbits[1])
        self.assertEqual(data.clbits, clbits)

        # Test re-adding is disallowed by default.
        with self.assertRaisesRegex(ValueError, "cannot add object"):
            data.add_clbit(clbits[0])

        # Make sure re-adding is allowed in non-strict mode
        # and does not change order.
        data.add_clbit(clbits[0], strict=False)
        self.assertEqual(data.clbits, clbits)

    def test_copy(self):
        """Test shallow copy behavior."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        data = CircuitData(
            qubits=qr,
            clbits=cr,
            data=[
                CircuitInstruction(XGate(), [qr[0]], []),
                CircuitInstruction(XGate(), [qr[1]], []),
                CircuitInstruction(Measure(), [qr[0]], [cr[1]]),
                CircuitInstruction(Measure(), [qr[1]], [cr[0]]),
            ],
            global_phase=1,
        )
        qubits = data.qubits
        clbits = data.clbits
        data_copy = data.copy()

        with self.subTest("list contents are equal"):
            self.assertEqual(list(data_copy), list(data))

        with self.subTest("qubits are equal but held in a new list"):
            self.assertIsNot(data_copy.qubits, qubits)
            self.assertEqual(data_copy.qubits, qubits)

        with self.subTest("clbits are equal but held in a new list"):
            self.assertIsNot(data_copy.clbits, clbits)
            self.assertEqual(data_copy.clbits, clbits)

        with self.subTest("global_phase is equal"):
            self.assertEqual(data.global_phase, data_copy.global_phase)

    def test_pickle_roundtrip(self):
        """Test pickle roundtrip coverage"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        data = CircuitData(
            qubits=qr,
            clbits=cr,
            data=[
                CircuitInstruction(XGate(), [qr[0]], []),
                CircuitInstruction(Measure(), [qr[0]], [cr[0]]),
            ],
            global_phase=1,
        )

        data.add_input_var(expr.Var.new("in1", types.Bool()))
        data.add_declared_var(expr.Var.new("v1", types.Float()))
        data.add_declared_stretch(expr.Stretch.new("s1"))

        self.assertEqual(data, pickle.loads(pickle.dumps(data)))

        # Now with captured variables
        data = CircuitData()
        data.add_captured_var(expr.Var.new("v1", types.Bool()))
        data.add_declared_var(expr.Var.new("v2", types.Float()))
        data.add_declared_stretch(expr.Stretch.new("s1"))
        data.add_captured_stretch(expr.Stretch.new("s2"))

        self.assertEqual(data, pickle.loads(pickle.dumps(data)))

    @ddt.data(
        (QuantumRegister(5), ClassicalRegister(5)),
        ([Qubit() for _ in range(5)], [Clbit() for _ in range(5)]),
    )
    @ddt.unpack
    def test_active_bits(self, qr, cr):
        """Test only active bits are returned."""
        data = CircuitData(
            qubits=qr,
            clbits=cr,
            data=[
                CircuitInstruction(XGate(), [qr[0]], []),
                CircuitInstruction(XGate(), [qr[3]], []),
                CircuitInstruction(Measure(), [qr[0]], [cr[1]]),
                CircuitInstruction(Measure(), [qr[3]], [cr[4]]),
            ],
        )

        expected_qubits = {qr[0], qr[3]}
        expected_clbits = {cr[1], cr[4]}
        actual_qubits, actual_clbits = data.active_bits()
        self.assertEqual(actual_qubits, expected_qubits)
        self.assertEqual(actual_clbits, expected_clbits)

    def test_foreach_op(self):
        """Test all operations are visited."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)

        visited_ops = []
        data.foreach_op(visited_ops.append)
        self.assertEqual(len(visited_ops), len(data_list))
        self.assertTrue(all(op is inst.operation for op, inst in zip(visited_ops, data_list)))

    def test_foreach_op_indexed(self):
        """Test all operations are visited."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)

        visited_ops = []
        data.foreach_op_indexed(visited_ops.insert)
        self.assertEqual(len(visited_ops), len(data_list))
        self.assertTrue(all(op is inst.operation for op, inst in zip(visited_ops, data_list)))

    def test_map_nonstandard_ops(self):
        """Test all operations are replaced."""
        qr = QuantumRegister(5)

        # Use a custom gate to ensure we get a gate class returned and not
        # a standard gate.
        class CustomXGate(XGate):
            """A custom X gate that doesn't have rust native representation."""

            _standard_gate = None

        data_list = [
            CircuitInstruction(CustomXGate(), [qr[0]], []),
            CircuitInstruction(CustomXGate(), [qr[1]], []),
            CircuitInstruction(CustomXGate(), [qr[2]], []),
            CircuitInstruction(CustomXGate(), [qr[3]], []),
            CircuitInstruction(CustomXGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)
        data.map_nonstandard_ops(lambda op: op.to_mutable())
        self.assertTrue(all(inst.operation.mutable for inst in data))

    def test_replace_bits(self):
        """Test replacing qubits and clbits with sequence reversed."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        data = CircuitData(
            qubits=qr,
            clbits=cr,
            data=[
                CircuitInstruction(XGate(), [qr[0]], []),
                CircuitInstruction(XGate(), [qr[1]], []),
                CircuitInstruction(XGate(), [qr[2]], []),
                CircuitInstruction(Measure(), [qr[0]], [cr[0]]),
                CircuitInstruction(Measure(), [qr[1]], [cr[1]]),
                CircuitInstruction(Measure(), [qr[2]], [cr[2]]),
            ],
        )

        data.replace_bits(qubits=list(reversed(qr)), clbits=list(reversed(cr)))
        self.assertEqual(
            data,
            [
                CircuitInstruction(XGate(), [qr[2]], []),
                CircuitInstruction(XGate(), [qr[1]], []),
                CircuitInstruction(XGate(), [qr[0]], []),
                CircuitInstruction(Measure(), [qr[2]], [cr[2]]),
                CircuitInstruction(Measure(), [qr[1]], [cr[1]]),
                CircuitInstruction(Measure(), [qr[0]], [cr[0]]),
            ],
        )

    def test_replace_bits_negative(self):
        """Test replacing with smaller bit sequence is rejected."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        data = CircuitData(qr, cr)
        with self.assertRaisesRegex(ValueError, "must contain at least"):
            data.replace_bits(qubits=qr[1:])
        with self.assertRaisesRegex(ValueError, "must contain at least"):
            data.replace_bits(clbits=cr[1:])

    @ddt.data(
        slice(0, 5, 1),  # Get everything.
        slice(-1, -6, -1),  # Get everything, reversed.
        slice(0, 4, 1),  # Get subslice.
        slice(0, 5, 2),  # Get every other.
        slice(-1, -6, -2),  # Get every other, reversed.
        slice(2, 2, 1),  # Get nothing.
        slice(2, 3, 1),  # Get at index 2.
        slice(4, 10, 1),  # Get index 4 to end, using excessive upper bound.
        slice(5, 0, -2),  # Get every other, reversed, excluding index 0.
        slice(-10, -5, 1),  # Get nothing.
        slice(0, 10, 1),  # Get everything.
    )
    def test_getitem_slice(self, sli):
        """Test that __getitem__ with slice is equivalent to that of list."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)
        self.assertEqual(data[sli], data_list[sli])

    @ddt.data(
        slice(0, 5, 1),  # Delete everything.
        slice(-1, -6, -1),  # Delete everything, reversed.
        slice(0, 4, 1),  # Delete subslice.
        slice(0, 5, 2),  # Delete every other.
        slice(-1, -6, -2),  # Delete every other, reversed.
        slice(2, 2, 1),  # Delete nothing.
        slice(2, 3, 1),  # Delete at index 2.
        slice(4, 10, 1),  # Delete index 4 to end, excessive upper bound.
        slice(5, 0, -2),  # Delete every other, reversed, excluding index 0.
        slice(-10, -5, 1),  # Delete nothing.
        slice(0, 10, 1),  # Delete everything, excessive upper bound.
    )
    def test_delitem_slice(self, sli):
        """Test that __delitem__ with slice is equivalent to that of list."""
        qr = QuantumRegister(5)
        data_list = [
            CircuitInstruction(XGate(), [qr[0]], []),
            CircuitInstruction(XGate(), [qr[1]], []),
            CircuitInstruction(XGate(), [qr[2]], []),
            CircuitInstruction(XGate(), [qr[3]], []),
            CircuitInstruction(XGate(), [qr[4]], []),
        ]
        data = CircuitData(qubits=list(qr), data=data_list)

        del data_list[sli]
        del data[sli]
        self.assertEqual(data[sli], data_list[sli])

    @ddt.data(
        (slice(0, 5, 1), 5),  # Replace entire slice.
        (slice(-1, -6, -1), 5),  # Replace entire slice, reversed.
        (slice(0, 4, 1), 4),  # Replace subslice.
        (slice(0, 4, 1), 10),  # Replace subslice with bigger sequence.
        (slice(0, 5, 2), 3),  # Replace every other.
        (slice(-1, -6, -2), 3),  # Replace every other, reversed.
        (slice(2, 2, 1), 1),  # Insert at index 2.
        (slice(2, 3, 1), 1),  # Replace at index 2.
        (slice(2, 3, 1), 10),  # Replace at index 2 with bigger sequence.
        (slice(4, 10, 1), 2),  # Replace index 4 with bigger sequence, excessive upper bound.
        (slice(5, 10, 1), 10),  # Append sequence.
        (slice(4, 0, -1), 4),  # Replace subslice at end, reversed.
    )
    @ddt.unpack
    def test_setitem_slice(self, sli, value_length):
        """Test that __setitem__ with slice is equivalent to that of list."""
        reg_size = 20
        self.assertLessEqual(value_length, reg_size)
        qr = QuantumRegister(reg_size)
        default_bit = Qubit()
        data_list = [
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
        ]
        data = CircuitData(qubits=list(qr) + [default_bit], data=data_list)

        value = [CircuitInstruction(XGate(), [qr[i]]) for i in range(value_length)]
        data_list[sli] = value
        data[sli] = value
        self.assertEqual(data, data_list)

    @ddt.data(
        (slice(0, 5, 2), 2),  # Replace smaller, with gaps.
        (slice(0, 5, 2), 4),  # Replace larger, with gaps.
        (slice(4, 0, -1), 10),  # Replace larger, reversed.
        (slice(-1, -6, -1), 6),  # Replace larger, reversed, negative notation.
        (slice(4, 3, -1), 10),  # Replace at index 4 with bigger sequence, reversed.
    )
    @ddt.unpack
    def test_setitem_slice_negative(self, sli, value_length):
        """Test that __setitem__ with slice is equivalent to that of list."""
        reg_size = 20
        self.assertLessEqual(value_length, reg_size)
        qr = QuantumRegister(reg_size)
        default_bit = Qubit()
        data_list = [
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
            CircuitInstruction(XGate(), [default_bit], []),
        ]
        data = CircuitData(qubits=list(qr) + [default_bit], data=data_list)

        value = [CircuitInstruction(XGate(), [qr[i]]) for i in range(value_length)]
        with self.assertRaises(ValueError):
            data_list[sli] = value
        with self.assertRaises(ValueError):
            data[sli] = value
        self.assertEqual(data, data_list)

    def test_unregistered_bit_error_new(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        with self.assertRaisesRegex(KeyError, "is not present"):
            CircuitData(qr, cr, [CircuitInstruction(XGate(), [Qubit()], [])])

    def test_unregistered_bit_error_append(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        data = CircuitData(qr, cr)
        with self.assertRaisesRegex(KeyError, "is not present"):
            qr_foreign = QuantumRegister(1)
            data.append(CircuitInstruction(XGate(), [qr_foreign[0]], []))

    def test_unregistered_bit_error_set(self):
        """Test using foreign bits is not allowed."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        data = CircuitData(qr, cr, [CircuitInstruction(XGate(), [qr[0]], [])])
        with self.assertRaisesRegex(KeyError, "is not present"):
            qr_foreign = QuantumRegister(1)
            data[0] = CircuitInstruction(XGate(), [qr_foreign[0]], [])


@ddt.ddt
class TestQuantumCircuitInstructionData(QiskitTestCase):
    """QuantumCircuit.data operation tests."""

    # N.B. Most of the cases here are not expected use cases of circuit.data
    # but are included as tests to maintain compatibility with the previous
    # list interface of circuit.data.

    def test_iteration_of_data_entry(self):
        """Verify that the base types of the legacy tuple iteration are correct, since they're
        different to attribute access."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure([0, 1, 2], [0, 1, 2])

        def to_legacy(instruction):
            return (instruction.operation, list(instruction.qubits), list(instruction.clbits))

        expected = [to_legacy(instruction) for instruction in qc.data]

        with self.assertWarnsRegex(
            DeprecationWarning, "Treating CircuitInstruction as an iterable is deprecated"
        ):
            actual = [tuple(instruction) for instruction in qc.data]
        self.assertEqual(actual, expected)

    def test_getitem_by_insertion_order(self):
        """Verify one can get circuit.data items in insertion order."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        data = qc.data

        self.assertEqual(data[0], CircuitInstruction(HGate(), [qr[0]], []))
        self.assertEqual(data[1], CircuitInstruction(CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(data[2], CircuitInstruction(HGate(), [qr[1]], []))

    def test_count_gates(self):
        """Verify circuit.data can count inst/qarg/carg tuples."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.x(0)
        qc.h(1)
        qc.h(0)

        data = qc.data

        self.assertEqual(data.count(CircuitInstruction(HGate(), [qr[0]], [])), 2)

    def test_len(self):
        """Verify finding the length of circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        self.assertEqual(len(qc.data), 0)
        qc.h(0)
        self.assertEqual(len(qc.data), 1)
        qc.cx(0, 1)
        self.assertEqual(len(qc.data), 2)

    def test_contains(self):
        """Verify checking if a inst/qarg/carg tuple is in circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)

        self.assertTrue(CircuitInstruction(HGate(), [qr[0]], []) in qc.data)
        self.assertFalse(CircuitInstruction(HGate(), [qr[1]], []) in qc.data)
        self.assertFalse(CircuitInstruction(XGate(), [qr[0]], []) in qc.data)

    def test_index_gates(self):
        """Verify finding the index of a inst/qarg/carg tuple in circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        self.assertEqual(qc.data.index(CircuitInstruction(HGate(), [qr[0]], [])), 0)
        self.assertEqual(qc.data.index(CircuitInstruction(CXGate(), [qr[0], qr[1]], [])), 1)
        self.assertEqual(qc.data.index(CircuitInstruction(HGate(), [qr[1]], [])), 2)

    def test_iter(self):
        """Verify circuit.data can behave as an iterator."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        iter_ = iter(qc.data)
        self.assertEqual(next(iter_), CircuitInstruction(HGate(), [qr[0]], []))
        self.assertEqual(next(iter_), CircuitInstruction(CXGate(), [qr[0], qr[1]], []))
        self.assertEqual(next(iter_), CircuitInstruction(HGate(), [qr[1]], []))
        self.assertRaises(StopIteration, next, iter_)

    def test_slice(self):
        """Verify circuit.data can be sliced."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.cx(1, 0)
        qc.h(1)
        qc.cx(0, 1)
        qc.h(0)

        h_slice = qc.data[::2]
        cx_slice = qc.data[1:-1:2]

        self.assertEqual(
            h_slice,
            [
                CircuitInstruction(HGate(), [qr[0]], []),
                CircuitInstruction(HGate(), [qr[1]], []),
                CircuitInstruction(HGate(), [qr[1]], []),
                CircuitInstruction(HGate(), [qr[0]], []),
            ],
        )
        self.assertEqual(
            cx_slice,
            [
                CircuitInstruction(CXGate(), [qr[0], qr[1]], []),
                CircuitInstruction(CXGate(), [qr[1], qr[0]], []),
                CircuitInstruction(CXGate(), [qr[0], qr[1]], []),
            ],
        )

    def test_copy(self):
        """Verify one can create a shallow copy circuit.data."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        data_copy = qc.data.copy()

        self.assertEqual(data_copy, qc.data)

    def test_copy_empty_like(self):
        """Test copy_empty_like with variable handling"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        data = CircuitData(
            qubits=qr,
            clbits=cr,
            global_phase=1,
        )
        data.add_input_var(expr.Var.new("in1", types.Bool()))
        data.add_declared_var(expr.Var.new("v1", types.Float()))
        data.add_declared_stretch(expr.Stretch.new("s1"))

        with self.subTest("vars_mode validation"):
            with self.assertRaises(ValueError):
                data.copy_empty_like(vars_mode="invalid-mode")

        with self.subTest("variables copied alike"):
            data_copy = data.copy_empty_like(vars_mode="alike")
            self.assertEqual(data_copy, data)

        with self.subTest("variables copied as captured"):
            data_copy = data.copy_empty_like(vars_mode="captures")
            self.assertNotEqual(data_copy, data)
            self.assertEqual(data_copy.num_captured_vars, 2)
            self.assertEqual(data_copy.num_captured_stretches, 1)

        with self.subTest("no variables are copied"):
            data_copy = data.copy_empty_like(vars_mode="drop")
            self.assertNotEqual(data_copy, data)
            self.assertEqual(
                data_copy.num_input_vars
                + data_copy.num_declared_vars
                + data_copy.num_captured_vars
                + data_copy.num_declared_stretches
                + data_copy.num_captured_stretches,
                0,
            )

    def test_repr(self):
        """Verify circuit.data repr."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(repr(qc.data), repr([g1[0], g2[0], g3[0]]))

    def test_str(self):
        """Verify circuit.data string representation."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        g1 = qc.h(0)
        g2 = qc.cx(0, 1)
        g3 = qc.h(1)

        self.assertEqual(str(qc.data), str([g1[0], g2[0], g3[0]]))

    def test_remove_gate(self):
        """Verify removing a gate via circuit.data.remove."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        qc.data.remove(CircuitInstruction(HGate(), [qr[0]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_del(self):
        """Verify removing a gate via circuit.data.delattr."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)
        qc.h(0)

        del qc.data[0]

        expected_qc = QuantumCircuit(qr)

        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_pop_gate(self):
        """Verify removing a gate via circuit.data.pop."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        last_h = qc.data.pop()

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)

        self.assertEqual(qc, expected_qc)
        self.assertEqual(last_h, CircuitInstruction(HGate(), [qr[1]], []))

    def test_clear_gates(self):
        """Verify emptying a circuit via circuit.data.clear."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data.clear()

        self.assertEqual(qc.data, [])

    def test_reverse_gates(self):
        """Verify reversing a circuit via circuit.data.reverse."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data.reverse()

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(1)
        expected_qc.cx(0, 1)
        expected_qc.h(0)

        self.assertEqual(qc, expected_qc)

    def test_repeating_a_circuit_via_mul(self):
        """Verify repeating a circuit via circuit.data.__mul__."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.h(0)
        qc.cx(0, 1)
        qc.h(1)

        qc.data *= 2

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

    def test_add_radd(self):
        """Verify adding lists of gates via circuit.data.__add__."""
        qr = QuantumRegister(2)
        qc1 = QuantumCircuit(qr)

        qc1.h(0)
        qc1.cx(0, 1)
        qc1.h(1)

        qc2 = QuantumCircuit(qr)
        qc2.cz(0, 1)

        qc1.data += qc2.data

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)
        expected_qc.cz(0, 1)

        self.assertEqual(qc1, expected_qc)

    def test_append_is_validated(self):
        """Verify appended gates via circuit.data are broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.append(CircuitInstruction(HGate(), [qr[0]], []))
        qc.data.append(CircuitInstruction(CXGate(), [0, 1], []))
        qc.data.append(CircuitInstruction(HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.append, CircuitInstruction(HGate(), [qr[0], qr[1]], [])
        )
        self.assertRaises(CircuitError, qc.data.append, CircuitInstruction(HGate(), [], [qr[0]]))

    def test_insert_is_validated(self):
        """Verify inserting gates via circuit.data are broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.insert(0, CircuitInstruction(HGate(), [qr[0]], []))
        qc.data.insert(1, CircuitInstruction(CXGate(), [0, 1], []))
        qc.data.insert(2, CircuitInstruction(HGate(), [qr[1]], []))

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.insert, 0, CircuitInstruction(HGate(), [qr[0], qr[1]], [])
        )
        self.assertRaises(CircuitError, qc.data.insert, 0, CircuitInstruction(HGate(), [], [qr[0]]))

    def test_extend_is_validated(self):
        """Verify extending circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data.extend(
            [
                CircuitInstruction(HGate(), [qr[0]], []),
                CircuitInstruction(CXGate(), [0, 1], []),
                CircuitInstruction(HGate(), [qr[1]], []),
            ]
        )

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        self.assertRaises(
            CircuitError, qc.data.extend, [CircuitInstruction(HGate(), [qr[0], qr[1]], [])]
        )
        self.assertRaises(CircuitError, qc.data.extend, [CircuitInstruction(HGate(), [], [qr[0]])])

    def test_setting_data_is_validated(self):
        """Verify setting circuit.data is broadcast and validated."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        qc.data = [
            CircuitInstruction(HGate(), [qr[0]], []),
            CircuitInstruction(CXGate(), [0, 1], []),
            CircuitInstruction(HGate(), [qr[1]], []),
        ]

        expected_qc = QuantumCircuit(qr)

        expected_qc.h(0)
        expected_qc.cx(0, 1)
        expected_qc.h(1)

        self.assertEqual(qc, expected_qc)

        with self.assertRaises(CircuitError):
            qc.data = [CircuitInstruction(HGate(), [qr[0], qr[1]], [])]
        with self.assertRaises(CircuitError):
            qc.data = [CircuitInstruction(HGate(), [], [qr[0]])]

    def test_setting_data_coerces_to_instruction(self):
        """Verify that the `to_instruction` coercion also happens when setting data using the legacy
        3-tuple format."""
        qc = QuantumCircuit(2)
        qc.cz(0, 1)

        class NotAnInstruction:
            def to_instruction(self):
                return CXGate()

        qc.data[0] = (NotAnInstruction(), qc.qubits, [])

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(qc, expected)

    def test_setting_data_allows_operation(self):
        """Test that using the legacy 3-tuple setter to the data allows arbitrary `Operation`
        classes to be used, not just `Instruction`."""

        class MyOp(Operation):

            @property
            def name(self):
                return "myop"

            @property
            def num_qubits(self):
                return 2

            @property
            def num_clbits(self):
                return 0

            def __eq__(self, other):
                return isinstance(other, MyOp)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.data[0] = (MyOp(), qc.qubits, [])

        expected = QuantumCircuit(2)
        expected.append(MyOp(), [0, 1], [])
        self.assertEqual(qc, expected)

    def test_param_gate_instance(self):
        """Verify that the same Parameter gate instance is not being used in
        multiple circuits."""
        a, b = Parameter("a"), Parameter("b")
        rx = RXGate(a)
        qc0, qc1 = QuantumCircuit(1), QuantumCircuit(1)
        qc0.append(rx, [0])
        qc1.append(rx, [0])
        qc0.assign_parameters({a: b}, inplace=True)
        # A fancy way of doing qc0_instance = qc0.data[0] and qc1_instance = qc1.data[0]
        # but this at least verifies the parameter table is point from the parameter to
        # the correct instruction (which is the only one)
        qc0_instance = qc0._data[next(iter(qc0._data._raw_parameter_table_entry(b)))[0]]
        qc1_instance = qc1._data[next(iter(qc1._data._raw_parameter_table_entry(a)))[0]]
        self.assertNotEqual(qc0_instance, qc1_instance)

    def test_variables(self):
        """Test input variables handling"""
        data = CircuitData()
        d1 = expr.Var.new("d1", types.Bool())
        data.add_declared_var(d1)
        with self.assertRaisesRegex(CircuitError, "already present in the circuit"):
            data.add_declared_var(d1)
        data.add_input_var(expr.Var.new("in1", types.Bool()))
        in2 = expr.Var.new("in2", types.Uint(4))
        data.add_input_var(in2)

        self.assertTrue(data.has_var("in1"))
        self.assertTrue(data.has_var(in2))
        self.assertFalse(data.has_var("vv"))
        self.assertEqual(data.num_input_vars, 2)
        self.assertEqual(data.get_var("in2"), in2)
        self.assertIsNone(data.get_var("v22"))
        self.assertTrue(data.has_input_var("in1"))
        self.assertFalse(data.has_input_var("in22"))
        self.assertTrue(in2 in data.get_input_vars())
        self.assertTrue(d1 in data.get_declared_vars())
        self.assertEqual(len(data.get_declared_vars()), 1)
        self.assertEqual(len(data.get_input_vars()), 2)

        with self.assertRaisesRegex(
            CircuitError, "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances"
        ):
            data.add_input_var(expr.Var(ClassicalRegister(1, "c"), types.Bool()))
        with self.assertRaisesRegex(
            CircuitError, "cannot add variables that wrap `Clbit` or `ClassicalRegister` instances"
        ):
            data.add_input_var(expr.Var(Clbit(), types.Bool()))

    def test_captured_variables(self):
        """Test captured variables handling"""
        data = CircuitData()
        c1 = expr.Var.new("c1", types.Bool())
        data.add_declared_var(expr.Var.new("d1", types.Bool()))
        data.add_captured_var(c1)

        self.assertEqual(data.get_var("c1"), c1)
        self.assertEqual(data.num_input_vars, 0)
        self.assertEqual(data.num_captured_vars, 1)
        self.assertEqual(data.num_declared_vars, 1)
        self.assertIn(c1, data.get_captured_vars())

    def test_local_stretches(self):
        """Test local stretch variables handling"""
        data = CircuitData()
        s1 = expr.Stretch.new("s1")
        data.add_declared_stretch(s1)
        with self.assertRaisesRegex(CircuitError, "already present in the circuit"):
            data.add_declared_stretch(s1)

        data.add_declared_stretch(expr.Stretch.new("s2"))
        self.assertTrue(data.has_stretch("s1"))
        self.assertTrue(data.has_declared_stretch("s1"))
        self.assertTrue(data.has_stretch("s2"))
        self.assertFalse(data.has_stretch("s3"))
        self.assertTrue(data.has_stretch(s1))
        self.assertEqual(data.get_stretch("s1"), s1)
        self.assertEqual(data.num_declared_stretches, 2)
        self.assertEqual(len(data.get_declared_stretches()), 2)

    def test_captured_stretches(self):
        """Test captured stretch variables handling"""
        data = CircuitData()
        s1 = expr.Stretch.new("s1")
        data.add_captured_stretch(s1)
        data.add_declared_stretch(expr.Stretch.new("s2"))
        self.assertTrue(data.has_stretch("s1"))
        self.assertEqual(data.get_stretch("s1"), s1)
        self.assertEqual(len([data.get_captured_stretches()]), 1)
        self.assertEqual(data.num_captured_stretches, 1)
        self.assertTrue(s1 in data.get_captured_stretches())

    def test_variable_declaration_rules(self):
        """Verify that variable and stretch declaration rules
        are enforced when adding new variables, including checks for name collisions"""
        with self.subTest("circuit with input variables"):
            data = CircuitData()
            data.add_input_var(expr.Var.new("in1", types.Bool()))
            with self.assertRaisesRegex(
                CircuitError, "cannot add var as its name shadows an existing identifier"
            ):
                data.add_input_var(expr.Var.new("in1", types.Bool()))
            with self.assertRaisesRegex(
                CircuitError,
                "circuits with input variables cannot be enclosed, so they cannot be closures",
            ):
                data.add_captured_var(expr.Var.new("v1", types.Bool()))

            data.add_declared_stretch(expr.Stretch.new("s1"))
            with self.assertRaisesRegex(
                CircuitError, "cannot add stretch as its name shadows an existing identifier"
            ):
                data.add_declared_stretch(expr.Stretch.new("s1"))
            with self.assertRaisesRegex(
                CircuitError,
                "circuits with input variables cannot be enclosed, so they cannot be closures",
            ):
                data.add_captured_stretch(expr.Stretch.new("s2"))

        with self.subTest("circuits with captured variables"):
            data = CircuitData()
            data.add_captured_var(expr.Var.new("v1", types.Bool()))
            data.add_declared_stretch(expr.Stretch.new("s1"))
            with self.assertRaisesRegex(
                CircuitError, "circuits to be enclosed with captures cannot have input variables"
            ):
                data.add_input_var(expr.Var.new("in1", types.Bool()))

            data = CircuitData()
            data.add_captured_stretch(expr.Stretch.new("s1"))
            with self.assertRaisesRegex(
                CircuitError, "circuits to be enclosed with captures cannot have input variables"
            ):
                data.add_input_var(expr.Var.new("in1", types.Bool()))

    def test_equality_vars(self):
        """Test equality checking over variables and stretches"""
        data = CircuitData()
        v1 = expr.Var.new("v1", types.Float())
        v2 = expr.Var.new("v2", types.Bool())
        data.add_input_var(v1)
        data.add_input_var(v2)
        s1 = expr.Stretch.new("s1")
        s2 = expr.Stretch.new("s2")

        with self.subTest("same number of vars, different names"):
            data_rhs = CircuitData()
            data_rhs.add_input_var(expr.Var.new("vv1", types.Float()))
            data_rhs.add_input_var(expr.Var.new("vv2", types.Bool()))

            self.assertFalse(data == data_rhs)

        with self.subTest("same names, different categories"):
            data_rhs = CircuitData()
            data_rhs.add_declared_stretch(s1)
            data_rhs.add_declared_stretch(s2)

            self.assertFalse(data == data_rhs)

        with self.subTest("same vars, different declaration scopes"):
            data_rhs = CircuitData()
            data_rhs.add_declared_var(v1)
            data_rhs.add_declared_var(v2)

            self.assertFalse(data == data_rhs)

        data.add_declared_stretch(s1)
        data.add_declared_stretch(s2)

        with self.subTest("same vars, different declared stretch order"):
            data_rhs = CircuitData()
            data_rhs.add_input_var(v1)
            data_rhs.add_input_var(v2)
            data_rhs.add_declared_stretch(s2)  # These two should be the same order
            data_rhs.add_declared_stretch(s1)

            self.assertFalse(data == data_rhs)

        d1 = expr.Var.new("d1", types.Bool())
        d2 = expr.Var.new("d2", types.Bool())
        data.add_declared_var(d1)
        data.add_declared_var(d2)

        with self.subTest("same vars, different variable order"):
            data_rhs = CircuitData()
            data_rhs.add_input_var(v2)
            data_rhs.add_declared_var(d2)
            data_rhs.add_input_var(v1)
            # Declared stretches should follow the same declaration order
            # (but not necessarily relative to all other variables )
            data_rhs.add_declared_stretch(s1)
            data_rhs.add_declared_var(d1)
            data_rhs.add_declared_stretch(s2)

            self.assertTrue(data == data_rhs)

        with self.subTest("captured vars, in different order"):
            data = CircuitData()
            data.add_captured_var(d1)
            data.add_captured_stretch(s1)
            data.add_captured_var(d2)
            data.add_captured_stretch(s2)

            data_rhs = CircuitData()
            data_rhs.add_captured_stretch(s2)
            data_rhs.add_captured_var(d2)
            data_rhs.add_captured_var(d1)
            data_rhs.add_captured_stretch(s1)

            self.assertTrue(data == data_rhs)

    @ddt.data(
        ("add_input_var", expr.Var.new("v", types.Bool()), expr.Var.new("v", types.Bool())),
        ("add_declared_var", expr.Var.new("v", types.Bool()), expr.Var.new("v", types.Bool())),
        ("add_captured_var", expr.Var.new("v", types.Bool()), expr.Var.new("v", types.Bool())),
        ("add_declared_stretch", expr.Stretch.new("s"), expr.Stretch.new("s")),
        ("add_captured_stretch", expr.Stretch.new("s"), expr.Stretch.new("s")),
    )
    @ddt.unpack
    def test_equality_vars_not_shared(self, func, lhs_v, rhs_v):
        """Verify that equality check fails when var instances are different"""
        data_lhs = CircuitData()
        getattr(data_lhs, func)(lhs_v)

        data_rhs = CircuitData()
        getattr(data_rhs, func)(rhs_v)
        self.assertFalse(data_rhs == data_lhs)
