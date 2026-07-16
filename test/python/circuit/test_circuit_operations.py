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


"""Test Qiskit's QuantumCircuit class."""
import copy
import pickle
import warnings
from itertools import combinations

import numpy as np
from ddt import data, ddt

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate, Instruction, Measure, Parameter, Barrier, AnnotatedOperation
from qiskit.circuit.classical import expr, types
from qiskit.circuit import Clbit
from qiskit.circuit.controlflow.box import BoxOp
from qiskit.circuit.controlflow.for_loop import ForLoopOp
from qiskit.circuit.controlflow.switch_case import SwitchCaseOp
from qiskit.circuit.controlflow.while_loop import WhileLoopOp
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import CXGate, HGate, XGate, YGate, ZGate, SXGate
from qiskit.circuit.library.standard_gates import SGate
from qiskit.circuit.quantumcircuit import BitLocations
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit import AncillaQubit, AncillaRegister, Qubit
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info import Operator
from qiskit.transpiler import Layout, CouplingMap, passes, Target, InstructionProperties
from qiskit.providers.fake_provider import GenericBackendV2
from test import QiskitTestCase


@ddt
class TestCircuitOperations(QiskitTestCase):
    """QuantumCircuit Operations tests."""

    @data(0, 1, -1, -2)
    def test_append_resolves_integers(self, index):
        """Test that integer arguments to append are correctly resolved."""
        # We need to assume that appending ``Bit`` instances will always work, so we have something
        # to test against.
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(qubits, clbits)
        expected.append(Measure(), [qubits[index]], [clbits[index]])
        self.assertEqual(test, expected)

    @data(np.int32(0), np.int8(-1), np.uint64(1))
    def test_append_resolves_numpy_integers(self, index):
        """Test that Numpy's integers can be used to reference qubits and clbits."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(qubits, clbits)
        expected.append(Measure(), [qubits[int(index)]], [clbits[int(index)]])
        self.assertEqual(test, expected)

    @data(
        slice(0, 2),
        slice(None, 1),
        slice(1, None),
        slice(None, None),
        slice(0, 2, 2),
        slice(2, -1, -1),
        slice(1000, 1003),
    )
    def test_append_resolves_slices(self, index):
        """Test that slices can be used to reference qubits and clbits with the same semantics that
        they have on lists."""
        qregs = [QuantumRegister(2), QuantumRegister(1)]
        cregs = [ClassicalRegister(1), ClassicalRegister(2)]
        test = QuantumCircuit(*qregs, *cregs)
        test.append(Measure(), [index], [index])
        expected = QuantumCircuit(*qregs, *cregs)
        for qubit, clbit in zip(expected.qubits[index], expected.clbits[index]):
            expected.append(Measure(), [qubit], [clbit])
        self.assertEqual(test, expected)

    def test_append_resolves_scalar_numpy_array(self):
        """Test that size-1 Numpy arrays can be used to index arguments.  These arrays can be passed
        to ``int``, which means they sometimes might be involved in spurious casts."""
        test = QuantumCircuit(1, 1)
        test.append(Measure(), [np.array([0])], [np.array([0])])

        expected = QuantumCircuit(1, 1)
        expected.measure(0, 0)

        self.assertEqual(test, expected)

    @data([3], [-3], [0, 1, 3])
    def test_append_rejects_out_of_range_input(self, specifier):
        """Test that append rejects an integer that's out of range."""
        test = QuantumCircuit(2, 2)
        with self.subTest("qubit"), self.assertRaisesRegex(CircuitError, "out of range"):
            opaque = Instruction("opaque", len(specifier), 1, [])
            test.append(opaque, specifier, [0])
        with self.subTest("clbit"), self.assertRaisesRegex(CircuitError, "out of range"):
            opaque = Instruction("opaque", 1, len(specifier), [])
            test.append(opaque, [0], specifier)

    def test_append_rejects_duplicates(self):
        test = QuantumCircuit(3, 3)

        qargs = Instruction("qargs", 2, 0, [])
        with self.subTest("qubit-int"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(qargs, [0, 0], [])
        with self.subTest("qubit-bit"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(qargs, [test.qubits[0]] * 2, [])
        with self.subTest("qubit-mixed"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(qargs, [0, test.qubits[0]], [])
        with self.subTest("qubit-many"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(Instruction("many", 3, 0, []), [0, 1, 0], [])

        cargs = Instruction("cargs", 0, 2, [])
        with self.subTest("clbit-int"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(cargs, [], [0, 0])
        with self.subTest("clbit-bit"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(cargs, [], [test.clbits[0]] * 2)
        with self.subTest("clbit-mixed"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(cargs, [], [0, test.clbits[0]])
        with self.subTest("clbit-many"), self.assertRaisesRegex(CircuitError, "duplicate"):
            test.append(Instruction("many", 0, 3, []), [], [0, 1, -2])

    def test_append_rejects_bits_not_in_circuit(self):
        """Test that append rejects bits that are not in the circuit."""
        test = QuantumCircuit(2, 2)
        with self.subTest("qubit"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [Qubit()], [test.clbits[0]])
        with self.subTest("clbit"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [test.qubits[0]], [Clbit()])
        with self.subTest("qubit list"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [[test.qubits[0], Qubit()]], [test.clbits])
        with self.subTest("clbit list"), self.assertRaisesRegex(CircuitError, "not in the circuit"):
            test.append(Measure(), [test.qubits], [[test.clbits[0], Clbit()]])

    def test_append_rejects_bit_of_wrong_type(self):
        """Test that append rejects bits of the wrong type in an argument list."""
        qubits = [Qubit(), Qubit()]
        clbits = [Clbit(), Clbit()]
        test = QuantumCircuit(qubits, clbits)
        with self.subTest("c to q"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [clbits[0]], [clbits[1]])
        with self.subTest("q to c"), self.assertRaisesRegex(CircuitError, "Incorrect bit type"):
            test.append(Measure(), [qubits[0]], [qubits[1]])

    @data(0.0, 1.0, 1.0 + 0.0j, "0")
    def test_append_rejects_wrong_types(self, specifier):
        """Test that various bad inputs are rejected, both given loose or in sublists."""
        test = QuantumCircuit(2, 2)
        # Use a default Instruction to be sure that there's not overridden broadcasting.
        opaque = Instruction("opaque", 1, 1, [])
        with self.subTest("q"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [specifier], [0])
        with self.subTest("c"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [0], [specifier])
        with self.subTest("q list"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [[specifier]], [[0]])
        with self.subTest("c list"), self.assertRaisesRegex(CircuitError, "Invalid bit index"):
            test.append(opaque, [[0]], [[specifier]])

    @data([], [0], [0, 1, 2])
    def test_append_rejects_bad_arguments_opaque(self, bad_arg):
        """Test that a suitable exception is raised when there is an argument mismatch."""
        inst = QuantumCircuit(2, 2).to_instruction()
        qc = QuantumCircuit(3, 3)
        with self.assertRaisesRegex(CircuitError, "The amount of qubit arguments"):
            qc.append(inst, bad_arg, [0, 1])
        with self.assertRaisesRegex(CircuitError, "The amount of clbit arguments"):
            qc.append(inst, [0, 1], bad_arg)
        with self.assertRaisesRegex(CircuitError, "The amount of qubit arguments"):
            qc.append(Barrier(4), bad_arg)

    def test_anding_self(self):
        """Test that qc &= qc finishes, which can be prone to infinite while-loops.

        This can occur e.g. when a user tries
        >>> other_qc = qc
        >>> other_qc &= qc  # or qc2.compose(qc)
        """
        qc = QuantumCircuit(1)
        qc.x(0)  # must contain at least one operation to end up in a infinite while-loop

        # attempt addition, times out if qc is added via reference
        qc &= qc

        # finally, qc should contain two X gates
        self.assertEqual(["x", "x"], [x.operation.name for x in qc.data])

    def test_compose_circuit(self):
        """Test composing two circuits"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc3 = qc1.compose(qc2)
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc3, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_and(self):
        """Test composing two circuits using & operator"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc3 = qc1 & qc2
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc3, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_iand(self):
        """Test composing circuits using &= operator (in place)"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc1 = QuantumCircuit(qr, cr)
        qc2 = QuantumCircuit(qr, cr)
        qc1.h(qr[0])
        qc1.measure(qr[0], cr[0])
        qc2.measure(qr[1], cr[1])

        qc1 &= qc2
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc1, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_compose_circuit_fail_circ_size(self):
        """Test composing circuit fails when number of wires in circuit is not enough"""
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(4)

        # Creating our circuits
        qc1 = QuantumCircuit(qr1)
        qc1.x(0)
        qc1.h(1)

        qc2 = QuantumCircuit(qr2)
        qc2.h([1, 2])
        qc2.cx(2, 3)

        # Composing will fail because qc2 requires 4 wires
        self.assertRaises(CircuitError, qc1.compose, qc2)

    def test_compose_circuit_fail_arg_size(self):
        """Test composing circuit fails when arg size does not match number of wires"""
        qr1 = QuantumRegister(2)
        qr2 = QuantumRegister(2)

        qc1 = QuantumCircuit(qr1)
        qc1.h(0)

        qc2 = QuantumCircuit(qr2)
        qc2.cx(0, 1)

        self.assertRaises(CircuitError, qc1.compose, qc2, qubits=[0])

    def test_tensor_circuit(self):
        """Test tensoring two circuits"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc3 = qc1.tensor(qc2)
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc3, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc1.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_tensor_circuit_xor(self):
        """Test tensoring two circuits using ^ operator"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc3 = qc1 ^ qc2
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc3, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc3.count_ops(), {"h": 1, "measure": 2})
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictEqual(qc1.count_ops(), {"measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_tensor_circuit_ixor(self):
        """Test tensoring two circuits using ^= operator"""
        qc1 = QuantumCircuit(1, 1)
        qc2 = QuantumCircuit(1, 1)

        qc2.h(0)
        qc2.measure(0, 0)
        qc1.measure(0, 0)

        qc1 ^= qc2
        backend = BasicSimulator()
        shots = 1024
        result = backend.run(qc1, shots=shots, seed_simulator=78).result()
        counts = result.get_counts()
        target = {"00": shots / 2, "01": shots / 2}
        threshold = 0.04 * shots
        self.assertDictEqual(qc1.count_ops(), {"h": 1, "measure": 2})  # changes "in-place"
        self.assertDictEqual(qc2.count_ops(), {"h": 1, "measure": 1})  # no changes "in-place"
        self.assertDictAlmostEqual(counts, target, threshold)

    def test_measure_args_type_cohesion(self):
        """Test for proper args types for measure function."""
        quantum_reg = QuantumRegister(3)
        classical_reg_0 = ClassicalRegister(1)
        classical_reg_1 = ClassicalRegister(2)
        quantum_circuit = QuantumCircuit(quantum_reg, classical_reg_0, classical_reg_1)
        quantum_circuit.h(quantum_reg)

        with self.assertRaises(CircuitError) as ctx:
            quantum_circuit.measure(quantum_reg, classical_reg_1)
        self.assertEqual(ctx.exception.message, "register size error")

    def test_copy_circuit(self):
        """Test copy method makes a copy"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        self.assertEqual(qc, qc.copy())

    def test_copy_copies_registers(self):
        """Test copy copies the registers not via reference."""
        qc = QuantumCircuit(1, 1)
        copied = qc.copy()

        copied.add_register(QuantumRegister(1, "additional_q"))
        copied.add_register(ClassicalRegister(1, "additional_c"))

        self.assertEqual(len(qc.qregs), 1)
        self.assertEqual(len(copied.qregs), 2)

        self.assertEqual(len(qc.cregs), 1)
        self.assertEqual(len(copied.cregs), 2)

    def test_copy_handles_global_phase(self):
        """Test that the global phase is included in the copy, including parameters."""
        a, b = Parameter("a"), Parameter("b")

        nonparametric = QuantumCircuit(global_phase=1.0).copy()
        self.assertEqual(nonparametric.global_phase, 1.0)
        self.assertEqual(set(nonparametric.parameters), set())

        parameter_phase = QuantumCircuit(global_phase=a).copy()
        self.assertEqual(parameter_phase.global_phase, a)
        self.assertEqual(set(parameter_phase.parameters), {a})
        # The `assign_parameters` is an indirect test that the `ParameterTable` is fully valid.
        self.assertEqual(parameter_phase.assign_parameters({a: 1.0}).global_phase, 1.0)

        expression_phase = QuantumCircuit(global_phase=a - b).copy()
        self.assertEqual(expression_phase.global_phase, a - b)
        self.assertEqual(set(expression_phase.parameters), {a, b})
        self.assertEqual(expression_phase.assign_parameters({a: 3, b: 2}).global_phase, 1.0)

    def test_copy_empty_like_circuit(self):
        """Test copy_empty_like method makes a clear copy."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr, global_phase=1.0, name="qc", metadata={"key": "value"})
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])

        copied = qc.copy_empty_like()
        qc.clear()

        self.assertEqual(qc, copied)
        self.assertEqual(qc.global_phase, copied.global_phase)
        self.assertEqual(qc.name, copied.name)
        self.assertEqual(qc.metadata, copied.metadata)

        copied = qc.copy_empty_like("copy")
        self.assertEqual(copied.name, "copy")

    def test_copy_variables(self):
        """Test that a full copy of circuits including variables copies them across."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))
        e = expr.Stretch.new("e")
        f = expr.Stretch.new("f")

        qc = QuantumCircuit(inputs=[a], declarations=[(c, expr.lift(False))])
        qc.add_stretch(e)
        copied = qc.copy()
        self.assertEqual({a}, set(copied.iter_input_vars()))
        self.assertEqual({c}, set(copied.iter_declared_vars()))
        self.assertEqual({e}, set(copied.iter_declared_stretches()))
        self.assertEqual(
            [instruction.operation for instruction in qc],
            [instruction.operation for instruction in copied.data],
        )

        # Check that the original circuit is not mutated.
        copied.add_input(b)
        copied.add_var(d, 0xFF)
        copied.add_stretch(f)
        self.assertEqual({a, b}, set(copied.iter_input_vars()))
        self.assertEqual({c, d}, set(copied.iter_declared_vars()))
        self.assertEqual({e, f}, set(copied.iter_declared_stretches()))
        self.assertEqual({a}, set(qc.iter_input_vars()))
        self.assertEqual({c}, set(qc.iter_declared_vars()))
        self.assertEqual({e}, set(qc.iter_declared_stretches()))

        qc = QuantumCircuit(captures=[b], declarations=[(a, expr.lift(False)), (c, a)])
        copied = qc.copy()
        self.assertEqual({b}, set(copied.iter_captured_vars()))
        self.assertEqual({a, c}, set(copied.iter_declared_vars()))
        self.assertEqual(
            [instruction.operation for instruction in qc],
            [instruction.operation for instruction in copied.data],
        )

        # Check that the original circuit is not mutated.
        copied.add_capture(d)
        copied.add_stretch(f)
        self.assertEqual({b, d}, set(copied.iter_captured_vars()))
        self.assertEqual({a, c}, set(copied.iter_declared_vars()))
        self.assertEqual({f}, set(copied.iter_declared_stretches()))
        self.assertEqual({b}, set(qc.iter_captured_vars()))
        self.assertEqual({a, c}, set(qc.iter_declared_vars()))
        self.assertEqual(set(), set(qc.iter_declared_stretches()))

    def test_copy_empty_variables(self):
        """Test that an empty copy of circuits including variables copies them across, but does not
        initialise them."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))

        qc = QuantumCircuit(inputs=[a], declarations=[(c, expr.lift(False))])
        copied = qc.copy_empty_like()
        self.assertEqual({a}, set(copied.iter_input_vars()))
        self.assertEqual({c}, set(copied.iter_declared_vars()))
        self.assertEqual([], list(copied.data))

        # Check that the original circuit is not mutated.
        copied.add_input(b)
        copied.add_var(d, 0xFF)
        e = copied.add_stretch("e")
        self.assertEqual({a, b}, set(copied.iter_input_vars()))
        self.assertEqual({c, d}, set(copied.iter_declared_vars()))
        self.assertEqual({e}, set(copied.iter_declared_stretches()))
        self.assertEqual({a}, set(qc.iter_input_vars()))
        self.assertEqual({c}, set(qc.iter_declared_vars()))
        self.assertEqual(set(), set(qc.iter_declared_stretches()))

        qc = QuantumCircuit(captures=[b], declarations=[(a, expr.lift(False)), (c, a)])
        copied = qc.copy_empty_like()
        self.assertEqual({b}, set(copied.iter_captured_vars()))
        self.assertEqual({a, c}, set(copied.iter_declared_vars()))
        self.assertEqual([], list(copied.data))

        # Check that the original circuit is not mutated.
        copied.add_capture(d)
        copied.add_capture(e)
        self.assertEqual({b, d}, set(copied.iter_captured_vars()))
        self.assertEqual({e}, set(copied.iter_captured_stretches()))
        self.assertEqual({b}, set(qc.iter_captured_vars()))
        self.assertEqual(set(), set(qc.iter_captured_stretches()))

    def test_copy_empty_variables_alike(self):
        """Test that an empty copy of circuits including variables copies them across, but does not
        initialise them.  This is the same as the default, just spelled explicitly."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))
        e = expr.Stretch.new("e")

        qc = QuantumCircuit(inputs=[a], declarations=[(c, expr.lift(False))])
        copied = qc.copy_empty_like(vars_mode="alike")
        self.assertEqual({a}, set(copied.iter_input_vars()))
        self.assertEqual({c}, set(copied.iter_declared_vars()))
        self.assertEqual([], list(copied.data))

        # Check that the original circuit is not mutated.
        copied.add_input(b)
        copied.add_var(d, 0xFF)
        copied.add_stretch(e)
        self.assertEqual({a, b}, set(copied.iter_input_vars()))
        self.assertEqual({c, d}, set(copied.iter_declared_vars()))
        self.assertEqual({e}, set(copied.iter_declared_stretches()))
        self.assertEqual({a}, set(qc.iter_input_vars()))
        self.assertEqual({c}, set(qc.iter_declared_vars()))
        self.assertEqual(set(), set(qc.iter_declared_stretches()))

        qc = QuantumCircuit(captures=[b], declarations=[(a, expr.lift(False)), (c, a)])
        copied = qc.copy_empty_like(vars_mode="alike")
        self.assertEqual({b}, set(copied.iter_captured_vars()))
        self.assertEqual({a, c}, set(copied.iter_declared_vars()))
        self.assertEqual([], list(copied.data))

        # Check that the original circuit is not mutated.
        copied.add_capture(d)
        copied.add_capture(e)
        self.assertEqual({b, d}, set(copied.iter_captured_vars()))
        self.assertEqual({e}, set(copied.iter_captured_stretches()))
        self.assertEqual({b}, set(qc.iter_captured_vars()))

    def test_copy_empty_variables_to_captures(self):
        """``vars_mode="captures"`` should convert all variables to captures."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Var.new("d", types.Uint(8))

        qc = QuantumCircuit(inputs=[a, b], declarations=[(c, expr.lift(False))])
        e = qc.add_stretch("e")
        copied = qc.copy_empty_like(vars_mode="captures")
        self.assertEqual({a, b, c}, set(copied.iter_captured_vars()))
        self.assertEqual({e}, set(copied.iter_captured_stretches()))
        self.assertEqual({a, b, c}, set(copied.iter_vars()))
        self.assertEqual({e}, set(copied.iter_stretches()))
        self.assertEqual([], list(copied.data))

        qc = QuantumCircuit(captures=[c, d, e])
        copied = qc.copy_empty_like(vars_mode="captures")
        self.assertEqual({c, d}, set(copied.iter_captured_vars()))
        self.assertEqual({e}, set(copied.iter_captured_stretches()))
        self.assertEqual({c, d}, set(copied.iter_vars()))
        self.assertEqual({e}, set(copied.iter_stretches()))
        self.assertEqual([], list(copied.data))

    def test_copy_empty_variables_drop(self):
        """``vars_mode="drop"`` should not have variables in the output."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Bool())
        d = expr.Stretch.new("s")

        qc = QuantumCircuit(captures=[a, b, d], declarations=[(c, expr.lift(False))])
        copied = qc.copy_empty_like(vars_mode="drop")
        self.assertEqual(set(), set(copied.iter_vars()))
        self.assertEqual(set(), set(copied.iter_stretches()))
        self.assertEqual([], list(copied.data))

    def test_copy_empty_like_parametric_phase(self):
        """Test that the parameter table of an empty circuit remains valid after copying a circuit
        with a parametric global phase."""
        a, b = Parameter("a"), Parameter("b")

        single = QuantumCircuit(global_phase=a).copy_empty_like()
        self.assertEqual(single.global_phase, a)
        self.assertEqual(set(single.parameters), {a})
        # The `assign_parameters` is an indirect test that the `ParameterTable` is fully valid.
        self.assertEqual(single.assign_parameters({a: 1.0}).global_phase, 1.0)

        stripped_instructions = QuantumCircuit(1, global_phase=a - b)
        stripped_instructions.rz(a, 0)
        stripped_instructions = stripped_instructions.copy_empty_like()
        self.assertEqual(stripped_instructions.global_phase, a - b)
        self.assertEqual(set(stripped_instructions.parameters), {a, b})
        self.assertEqual(stripped_instructions.assign_parameters({a: 3, b: 2}).global_phase, 1.0)

    def test_circuit_copy_rejects_invalid_types(self):
        """Test copy method rejects argument with type other than 'string' and 'None' type."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)

        with self.assertRaises(TypeError):
            qc.copy([1, "2", 3])

    def test_circuit_copy_empty_like_rejects_invalid_types(self):
        """Test copy_empty_like method rejects argument with type other than 'string' and 'None' type."""
        qc = QuantumCircuit(1, 1)
        qc.h(0)

        with self.assertRaises(TypeError):
            qc.copy_empty_like(123)

    def test_clear_circuit(self):
        """Test clear method deletes instructions in circuit."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.clear()

        self.assertEqual(len(qc.data), 0)
        self.assertEqual(qc._data.num_parameters(), 0)

    def test_barrier(self):
        """Test multiple argument forms of barrier."""
        qr1, qr2 = QuantumRegister(3, "qr1"), QuantumRegister(4, "qr2")
        qc = QuantumCircuit(qr1, qr2)
        qc.barrier()  # All qubits.
        qc.barrier(0, 1)
        qc.barrier([4, 2])
        qc.barrier(qr1)
        qc.barrier(slice(3, 5))
        qc.barrier({1, 4, 2}, range(5, 7))

        expected = QuantumCircuit(qr1, qr2)
        expected.append(Barrier(expected.num_qubits), expected.qubits.copy(), [])
        expected.append(Barrier(2), [expected.qubits[0], expected.qubits[1]], [])
        expected.append(Barrier(2), [expected.qubits[2], expected.qubits[4]], [])
        expected.append(Barrier(3), expected.qubits[0:3], [])
        expected.append(Barrier(2), [expected.qubits[3], expected.qubits[4]], [])
        expected.append(Barrier(5), [expected.qubits[x] for x in [1, 2, 4, 5, 6]], [])

        self.assertEqual(qc, expected)

    def test_barrier_in_context(self):
        """Test barrier statement in context, see gh-11345"""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        with qc.if_test((qc.clbits[0], False)):
            qc.h(0)
            qc.barrier()

        operation_names = [c.operation.name for c in qc]
        self.assertNotIn("barrier", operation_names)

    def test_measure_active(self):
        """Test measure_active
        Applies measurements only to non-idle qubits. Creates a ClassicalRegister of size equal to
        the amount of non-idle qubits to store the measured values.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[2])
        circuit.measure_active()

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[2])
        expected.add_register(cr)
        expected.barrier()
        expected.measure([qr[0], qr[2]], [cr[0], cr[1]])

        self.assertEqual(expected, circuit)

    def test_measure_active_copy(self):
        """Test measure_active copy
        Applies measurements only to non-idle qubits. Creates a ClassicalRegister of size equal to
        the amount of non-idle qubits to store the measured values.
        """
        qr = QuantumRegister(4)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[2])
        new_circuit = circuit.measure_active(inplace=False)

        expected = QuantumCircuit(qr)
        expected.h(qr[0])
        expected.h(qr[2])
        expected.add_register(cr)
        expected.barrier()
        expected.measure([qr[0], qr[2]], [cr[0], cr[1]])

        self.assertEqual(expected, new_circuit)
        self.assertFalse("measure" in circuit.count_ops().keys())

    def test_measure_active_repetition(self):
        """Test measure_active in a circuit with a 'measure' creg.
        measure_active should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "measure")

        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.measure_active()

        self.assertEqual(len(circuit.cregs), 2)  # Two cregs
        self.assertEqual(len(circuit.cregs[0]), 2)  # Both length 2
        self.assertEqual(len(circuit.cregs[1]), 2)

    def test_measure_all(self):
        """Test measure_all applies measurements to all qubits.
        Creates a ClassicalRegister of size equal to the total amount of qubits to
        store those measured values.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        circuit.measure_all()

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, circuit)

    def test_measure_all_after_copy(self):
        """
        Test measure_all on a circuit that has been copied.
        """
        qc = QuantumCircuit(2)
        qc.h(1)

        qc2 = qc.copy()

        qc.measure_all()
        qc2.measure_all()

        expected_cregs = [ClassicalRegister(2, "meas")]
        self.assertEqual(qc.cregs, expected_cregs)
        self.assertEqual(qc2.cregs, expected_cregs)
        self.assertEqual(qc, qc2)

    def test_measure_all_after_deepcopy(self):
        """
        Test measure_all on a circuit that has been deep-copied.
        """
        qc = QuantumCircuit(2)
        qc.h(1)

        qc2 = copy.deepcopy(qc)

        qc.measure_all()
        qc2.measure_all()

        expected_cregs = [ClassicalRegister(2, "meas")]
        self.assertEqual(qc.cregs, expected_cregs)
        self.assertEqual(qc2.cregs, expected_cregs)
        self.assertEqual(qc, qc2)

    def test_measure_all_after_pickle(self):
        """
        Test measure_all on a circuit that has been pickled.
        """
        qc = QuantumCircuit(2)
        qc.h(1)

        qc2 = pickle.loads(pickle.dumps(qc))

        qc.measure_all()
        qc2.measure_all()

        expected_cregs = [ClassicalRegister(2, "meas")]
        self.assertEqual(qc.cregs, expected_cregs)
        self.assertEqual(qc2.cregs, expected_cregs)
        self.assertEqual(qc, qc2)

    def test_measure_all_not_add_bits_equal(self):
        """Test measure_all applies measurements to all qubits.
        Does not create a new ClassicalRegister if the existing one is big enough.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all(add_bits=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, circuit)

    def test_measure_all_not_add_bits_bigger(self):
        """Test measure_all applies measurements to all qubits.
        Does not create a new ClassicalRegister if the existing one is big enough.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(3, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all(add_bits=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr[0:2])

        self.assertEqual(expected, circuit)

    def test_measure_all_not_add_bits_smaller(self):
        """Test measure_all applies measurements to all qubits.
        Raises an error if there are not enough classical bits to store the measurements.
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)

        with self.assertRaisesRegex(CircuitError, "The number of classical bits"):
            circuit.measure_all(add_bits=False)

    def test_measure_all_copy(self):
        """Test measure_all with inplace=False"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr)
        new_circuit = circuit.measure_all(inplace=False)

        expected = QuantumCircuit(qr, cr)
        expected.barrier()
        expected.measure(qr, cr)

        self.assertEqual(expected, new_circuit)
        self.assertFalse("measure" in circuit.count_ops().keys())

    def test_measure_all_repetition(self):
        """Test measure_all in a circuit with a 'measure' creg.
        measure_all should be aware that the creg 'measure' might exists.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "measure")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure_all()

        self.assertEqual(len(circuit.cregs), 2)  # Two cregs
        self.assertEqual(len(circuit.cregs[0]), 2)  # Both length 2
        self.assertEqual(len(circuit.cregs[1]), 2)

    def test_measure_all_with_multiple_regs_creation(self):
        """Test measure_all in a circuit where the method is called
        multiple times consecutively and checks that a register of
        a different name is created on each call."""

        circuit = QuantumCircuit(1)

        # First call should create a new register
        circuit.measure_all()
        self.assertEqual(len(circuit.cregs), 1)  # One creg
        self.assertEqual(len(circuit.cregs[0]), 1)  # Of length 1

        # Second call should also create a new register
        circuit.measure_all()
        self.assertEqual(len(circuit.cregs), 2)  # Now two cregs
        self.assertTrue(all(len(reg) == 1 for reg in circuit.cregs))  # All of length 1
        # Check that no name is the same
        self.assertEqual(len({reg.name for reg in circuit.cregs}), 2)

        # Third call should also create a new register
        circuit.measure_all()
        self.assertEqual(len(circuit.cregs), 3)  # Now three cregs
        self.assertTrue(all(len(reg) == 1 for reg in circuit.cregs))  # All of length 1
        # Check that no name is the same
        self.assertEqual(len({reg.name for reg in circuit.cregs}), 3)

    def test_remove_final_measurements(self):
        """Test remove_final_measurements
        Removes all measurements at end of circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()

        expected = QuantumCircuit(qr)

        self.assertEqual(expected, circuit)

    def test_remove_final_measurements_copy(self):
        """Test remove_final_measurements on copy
        Removes all measurements at end of circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        new_circuit = circuit.remove_final_measurements(inplace=False)

        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)
        self.assertTrue("measure" in circuit.count_ops().keys())

    def test_remove_final_measurements_copy_with_parameters(self):
        """Test remove_final_measurements doesn't corrupt ParameterTable

        See https://github.com/Qiskit/qiskit-terra/issues/6108 for more details
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2, "meas")
        theta = Parameter("theta")

        circuit = QuantumCircuit(qr, cr)
        circuit.rz(theta, qr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()
        copied = circuit.copy()

        self.assertEqual(copied, circuit)

    def test_remove_final_measurements_multiple_measures(self):
        """Test remove_final_measurements only removes measurements at the end of the circuit
        remove_final_measurements should not remove measurements in the beginning or middle of the
        circuit.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr[0], cr)
        circuit.h(0)
        circuit.measure(qr[0], cr)
        circuit.h(0)
        circuit.measure(qr[0], cr)
        circuit.remove_final_measurements()

        expected = QuantumCircuit(qr, cr)
        expected.measure(qr[0], cr)
        expected.h(0)
        expected.measure(qr[0], cr)
        expected.h(0)

        self.assertEqual(expected, circuit)

    def test_remove_final_measurements_5802(self):
        """Test remove_final_measurements removes classical bits
        https://github.com/Qiskit/qiskit-terra/issues/5802.
        """
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        circuit = QuantumCircuit(qr, cr)
        circuit.measure(qr, cr)
        circuit.remove_final_measurements()

        self.assertEqual(circuit.cregs, [])
        self.assertEqual(circuit.clbits, [])

    def test_remove_final_measurements_7089(self):
        """Test remove_final_measurements removes resulting unused registers
        even if not all bits were measured into.
        https://github.com/Qiskit/qiskit-terra/issues/7089.
        """
        circuit = QuantumCircuit(2, 5)
        circuit.measure(0, 0)
        circuit.measure(1, 1)
        circuit.remove_final_measurements(inplace=True)

        self.assertEqual(circuit.cregs, [])
        self.assertEqual(circuit.clbits, [])

    def test_remove_final_measurements_bit_locations(self):
        """Test remove_final_measurements properly recalculates clbit indices
        and preserves order of remaining cregs and clbits.
        """
        c0 = ClassicalRegister(1)
        c1_0 = Clbit()
        c2 = ClassicalRegister(1)
        c3 = ClassicalRegister(1)

        # add an individual bit that's not in any register of this circuit
        circuit = QuantumCircuit(QuantumRegister(1), c0, [c1_0], c2, c3)

        circuit.measure(0, c1_0)
        circuit.measure(0, c2[0])

        # assert cregs and clbits before measure removal
        self.assertEqual(circuit.cregs, [c0, c2, c3])
        self.assertEqual(circuit.clbits, [c0[0], c1_0, c2[0], c3[0]])

        # assert clbit indices prior to measure removal
        self.assertEqual(circuit.find_bit(c0[0]), BitLocations(0, [(c0, 0)]))
        self.assertEqual(circuit.find_bit(c1_0), BitLocations(1, []))
        self.assertEqual(circuit.find_bit(c2[0]), BitLocations(2, [(c2, 0)]))
        self.assertEqual(circuit.find_bit(c3[0]), BitLocations(3, [(c3, 0)]))

        circuit.remove_final_measurements()

        # after measure removal, creg c2 should be gone, as should lone bit c1_0
        # and c0 should still come before c3
        self.assertEqual(circuit.cregs, [c0, c3])
        self.assertEqual(circuit.clbits, [c0[0], c3[0]])

        # there should be no gaps in clbit indices
        # e.g. c3[0] is now the second clbit
        self.assertEqual(circuit.find_bit(c0[0]), BitLocations(0, [(c0, 0)]))
        self.assertEqual(circuit.find_bit(c3[0]), BitLocations(1, [(c3, 0)]))

    def test_remove_final_measurements_parametric_global_phase(self):
        """Test that a parametric global phase is respected in the table afterwards."""
        a = Parameter("a")
        qc = QuantumCircuit(2, 2, global_phase=a)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])

        expected = QuantumCircuit(2, global_phase=1)
        expected.h(0)
        expected.cx(0, 1)

        self.assertEqual(
            qc.remove_final_measurements(inplace=False).assign_parameters({a: 1}), expected
        )
        qc.remove_final_measurements(inplace=True)
        self.assertEqual(qc.assign_parameters({a: 1}), expected)

    def test_remove_final_measurement_with_layout(self):
        def apply_layout(qc):
            layout = Layout(dict(zip(qc.qubits, [1, 0])))
            return passes.ApplyLayout()(qc, property_set={"layout": layout})

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc = apply_layout(qc)

        expected = QuantumCircuit(2)
        expected.h(0)
        expected.cx(0, 1)
        expected = apply_layout(expected)

        with warnings.catch_warnings():
            warnings.filterwarnings("error", module=r"(qiskit|test)")
            self.assertEqual(qc.remove_final_measurements(inplace=False), expected)
            qc.remove_final_measurements(inplace=True)
            self.assertEqual(qc, expected)

    def test_reverse(self):
        """Test reverse method reverses but does not invert."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.x(0)
        qc.y(1)

        expected = QuantumCircuit(2, 2)
        expected.y(1)
        expected.x(0)
        expected.measure([0, 1], [0, 1])
        expected.cx(0, 1)
        expected.s(1)
        expected.h(0)

        self.assertEqual(qc.reverse_ops(), expected)

    def test_reverse_with_standlone_vars(self):
        """Test that instruction-reversing works in the presence of stand-alone variables."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Uint(8))

        qc = QuantumCircuit(2, inputs=[a])
        qc.add_var(b, 12)
        qc.h(0)
        qc.cx(0, 1)
        with qc.if_test(a):
            # We don't really comment on what should happen within control-flow operations in this
            # method - it's not really defined in a non-linear CFG.  This deliberately uses a body
            # of length 1 (a single `Store`), so there's only one possibility.
            qc.add_var(c, 12)

        expected = qc.copy_empty_like()
        with expected.if_test(a):
            expected.add_var(c, 12)
        expected.cx(0, 1)
        expected.h(0)
        expected.store(b, 12)

        self.assertEqual(qc.reverse_ops(), expected)

    def test_repeat(self):
        """Test repeating the circuit works."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1)

        with self.subTest("repeat 0 times"):
            rep = qc.repeat(0)
            self.assertEqual(rep, QuantumCircuit(qr, cr))

        with self.subTest("repeat 3 times"):
            inst = qc.to_instruction()
            ref = QuantumCircuit(qr, cr)
            for _ in range(3):
                ref.append(inst, ref.qubits, ref.clbits)
            rep = qc.repeat(3)
            self.assertEqual(rep, ref)

    @data(0, 1, 4)
    def test_repeat_global_phase(self, num):
        """Test the global phase is properly handled upon repeat."""
        phase = 0.123
        qc = QuantumCircuit(1, global_phase=phase)
        expected = np.exp(1j * phase * num) * np.identity(2)
        np.testing.assert_array_almost_equal(Operator(qc.repeat(num)).data, expected)

    def test_bind_global_phase(self):
        """Test binding global phase."""
        x = Parameter("x")
        circuit = QuantumCircuit(1, global_phase=x)
        self.assertEqual(circuit.parameters, {x})

        bound = circuit.assign_parameters({x: 2})
        self.assertEqual(bound.global_phase, 2)
        self.assertEqual(bound.parameters, set())

    def test_bind_parameter_in_phase_and_gate(self):
        """Test binding a parameter present in the global phase and the gates."""
        x = Parameter("x")
        circuit = QuantumCircuit(1, global_phase=x)
        circuit.rx(x, 0)
        self.assertEqual(circuit.parameters, {x})

        ref = QuantumCircuit(1, global_phase=2)
        ref.rx(2, 0)

        bound = circuit.assign_parameters({x: 2})
        self.assertEqual(bound, ref)
        self.assertEqual(bound.parameters, set())

    def test_power(self):
        """Test taking the circuit to a power works."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(0.2, 1)

        gate = qc.to_gate()

        with self.subTest("power(int >= 0) equals repeat"):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest("explicit matrix power"):
            self.assertEqual(qc.power(4, matrix_power=True).data[0].operation, gate.power(4))

        with self.subTest("float power"):
            self.assertEqual(qc.power(1.23).data[0].operation, gate.power(1.23))

        with self.subTest("negative power"):
            self.assertEqual(qc.power(-2).data[0].operation, gate.power(-2))

        with self.subTest("integer circuit power via annotation"):
            power_qc = qc.power(4, annotated=True)
            self.assertIsInstance(power_qc[0].operation, AnnotatedOperation)
            self.assertEqual(Operator(power_qc), Operator(qc).power(4))

        with self.subTest("float circuit power via annotation"):
            power_qc = qc.power(1.5, annotated=True)
            self.assertIsInstance(power_qc[0].operation, AnnotatedOperation)
            self.assertEqual(Operator(power_qc), Operator(qc).power(1.5))

        with self.subTest("negative circuit power via annotation"):
            power_qc = qc.power(-2, annotated=True)
            self.assertIsInstance(power_qc[0].operation, AnnotatedOperation)
            self.assertEqual(Operator(power_qc), Operator(qc).power(-2))

    def test_power_parameterized_circuit(self):
        """Test taking a parameterized circuit to a power."""
        theta = Parameter("th")
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.rx(theta, 1)

        with self.subTest("power(int >= 0) equals repeat"):
            self.assertEqual(qc.power(4), qc.repeat(4))

        with self.subTest("cannot to matrix power if parameterized"):
            with self.assertRaises(CircuitError):
                _ = qc.power(0.5)

    def test_control(self):
        """Test controlling the circuit."""
        qc = QuantumCircuit(2, name="my_qc")
        qc.cry(0.2, 0, 1)

        c_qc = qc.control(annotated=False)
        with self.subTest("return type is circuit"):
            self.assertIsInstance(c_qc, QuantumCircuit)

        with self.subTest("test name"):
            self.assertEqual(c_qc.name, "c_my_qc")

        with self.subTest("repeated control"):
            cc_qc = c_qc.control(annotated=False)
            self.assertEqual(cc_qc.num_qubits, c_qc.num_qubits + 1)

        with self.subTest("controlled circuit has same parameter"):
            param = Parameter("p")
            qc.rx(param, 0)
            c_qc = qc.control(annotated=False)
            self.assertEqual(qc.parameters, c_qc.parameters)

        with self.subTest("non-unitary operation raises"):
            qc.reset(0)
            with self.assertRaises(CircuitError):
                _ = qc.control(annotated=False)

    def test_control_implementation(self):
        """Run a test case for controlling the circuit, which should use ``Gate.control``."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cry(0.2, 0, 1)
        qc.t(0)
        qc.append(SGate().control(2, annotated=False), [0, 1, 2])
        qc.iswap(2, 0)

        c_qc = qc.control(2, ctrl_state="10", annotated=False)

        cgate = qc.to_gate().control(2, ctrl_state="10", annotated=False)
        ref = QuantumCircuit(*c_qc.qregs)
        ref.append(cgate, ref.qubits)

        self.assertEqual(ref, c_qc)

    @data("gate", "instruction")
    def test_repeat_appended_type(self, subtype):
        """Test repeat appends Gate if circuit contains only gates and Instructions otherwise."""
        sub = QuantumCircuit(2)
        sub.x(0)

        if subtype == "gate":
            sub = sub.to_gate()
        else:
            sub = sub.to_instruction()

        qc = QuantumCircuit(2)
        qc.append(sub, [0, 1])
        rep = qc.repeat(3)

        if subtype == "gate":
            self.assertTrue(all(isinstance(op.operation, Gate) for op in rep.data))
        else:
            self.assertTrue(all(isinstance(op.operation, Instruction) for op in rep.data))

    def test_reverse_bits(self):
        """Test reversing order of bits."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.measure(0, 1)
        qc.x(0)
        qc.y(1)
        qc.global_phase = -1

        expected = QuantumCircuit(3, 2)
        expected.h(2)
        expected.s(1)
        expected.cx(2, 1)
        expected.measure(2, 0)
        expected.x(2)
        expected.y(1)
        expected.global_phase = -1

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_boxed(self):
        """Test reversing order of bits in a hierarchical circuit."""
        wide_cx = QuantumCircuit(3)
        wide_cx.cx(0, 1)
        wide_cx.cx(1, 2)

        wide_cxg = wide_cx.to_gate()
        cx_box = QuantumCircuit(3)
        cx_box.append(wide_cxg, [0, 1, 2])

        expected = QuantumCircuit(3)
        expected.cx(2, 1)
        expected.cx(1, 0)

        self.assertEqual(cx_box.reverse_bits().decompose(), expected)
        self.assertEqual(cx_box.decompose().reverse_bits(), expected)

        # box one more layer to be safe.
        cx_box_g = cx_box.to_gate()
        cx_box_box = QuantumCircuit(4)
        cx_box_box.append(cx_box_g, [0, 1, 2])
        cx_box_box.cx(0, 3)

        expected2 = QuantumCircuit(4)
        expected2.cx(3, 2)
        expected2.cx(2, 1)
        expected2.cx(3, 0)

        self.assertEqual(cx_box_box.reverse_bits().decompose().decompose(), expected2)

    def test_reverse_bits_with_registers(self):
        """Test reversing order of bits when registers are present."""
        qr1 = QuantumRegister(3, "a")
        qr2 = QuantumRegister(2, "b")
        qc = QuantumCircuit(qr1, qr2)
        qc.h(qr1[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        qc.cx(qr2[0], qr2[1])

        expected = QuantumCircuit(qr2, qr1)
        expected.h(qr1[2])
        expected.cx(qr1[2], qr1[1])
        expected.cx(qr1[1], qr1[0])
        expected.cx(qr1[0], qr2[1])
        expected.cx(qr2[1], qr2[0])

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_with_overlapped_registers(self):
        """Test reversing order of bits when registers are overlapped."""
        qr1 = QuantumRegister(2, "a")
        qr2 = QuantumRegister(bits=[qr1[0], qr1[1], Qubit()], name="b")
        qc = QuantumCircuit(qr1, qr2)
        qc.h(qr1[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr2[2])

        qr2 = QuantumRegister(bits=[Qubit(), qr1[0], qr1[1]], name="b")
        expected = QuantumCircuit(qr2, qr1)
        expected.h(qr1[1])
        expected.cx(qr1[1], qr1[0])
        expected.cx(qr1[0], qr2[0])

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_with_registerless_bits(self):
        """Test reversing order of registerless bits."""
        q0 = Qubit()
        q1 = Qubit()
        c0 = Clbit()
        c1 = Clbit()
        qc = QuantumCircuit([q0, q1], [c0, c1])
        qc.h(0)
        qc.cx(0, 1)
        qc.x(0)
        qc.measure(0, 0)

        expected = QuantumCircuit([c1, c0], [q1, q0])
        expected.h(1)
        expected.cx(1, 0)
        expected.x(1)
        expected.measure(1, 1)

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_with_registers_and_bits(self):
        """Test reversing order of bits with registers and registerless bits."""
        qr = QuantumRegister(2, "a")
        q = Qubit()
        qc = QuantumCircuit(qr, [q])
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], q)

        expected = QuantumCircuit([q], qr)
        expected.h(qr[1])
        expected.cx(qr[1], qr[0])
        expected.cx(qr[0], q)

        self.assertEqual(qc.reverse_bits(), expected)

    def test_reverse_bits_with_mixed_overlapped_registers(self):
        """Test reversing order of bits with overlapped registers and registerless bits."""
        q = Qubit()
        qr1 = QuantumRegister(bits=[q, Qubit()], name="qr1")
        qr2 = QuantumRegister(bits=[qr1[1], Qubit()], name="qr2")
        qc = QuantumCircuit(qr1, qr2, [Qubit()])
        qc.h(q)
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr2[1])
        qc.cx(2, 3)

        qr2 = QuantumRegister(2, "qr2")
        qr1 = QuantumRegister(bits=[qr2[1], q], name="qr1")
        expected = QuantumCircuit([Qubit()], qr2, qr1)
        expected.h(qr1[1])
        expected.cx(qr1[1], qr1[0])
        expected.cx(qr1[0], qr2[0])
        expected.cx(1, 0)

        self.assertEqual(qc.reverse_bits(), expected)

    def test_cnot_alias(self):
        """Test that the cnot method alias adds a cx gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        expected = QuantumCircuit(2)
        expected.cx(0, 1)
        self.assertEqual(qc, expected)

    def test_inverse(self):
        """Test inverse circuit."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr, global_phase=0.5)
        qc.h(0)
        qc.barrier(qr)
        qc.t(1)

        expected = QuantumCircuit(qr)
        expected.tdg(1)
        expected.barrier(qr)
        expected.h(0)
        expected.global_phase = -0.5
        self.assertEqual(qc.inverse(), expected)

    def test_compare_two_equal_circuits(self):
        """Test to compare that 2 circuits are equal."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = QuantumCircuit(2, 2)
        qc2.h(0)

        self.assertTrue(qc1 == qc2)

    def test_compare_two_different_circuits(self):
        """Test to compare that 2 circuits are different."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = QuantumCircuit(2, 2)
        qc2.x(0)

        self.assertFalse(qc1 == qc2)

    def test_compare_a_circuit_with_none(self):
        """Test to compare that a circuit is different to None."""
        qc1 = QuantumCircuit(2, 2)
        qc1.h(0)

        qc2 = None

        self.assertFalse(qc1 == qc2)

    def test_overlapped_add_bits_and_add_register(self):
        """Test add registers whose bits have already been added by add_bits."""
        qc = QuantumCircuit()
        for bit_type, reg_type in (
            [Qubit, QuantumRegister],
            [Clbit, ClassicalRegister],
            [AncillaQubit, AncillaRegister],
        ):
            bits = [bit_type() for _ in range(10)]
            reg = reg_type(bits=bits)
            qc.add_bits(bits)
            qc.add_register(reg)

        self.assertEqual(qc.num_qubits, 20)
        self.assertEqual(qc.num_clbits, 10)
        self.assertEqual(qc.num_ancillas, 10)

    def test_overlapped_add_register_and_add_register(self):
        """Test add registers whose bits have already been added by add_register."""
        qc = QuantumCircuit()
        for bit_type, reg_type in (
            [Qubit, QuantumRegister],
            [Clbit, ClassicalRegister],
            [AncillaQubit, AncillaRegister],
        ):
            bits = [bit_type() for _ in range(10)]
            reg1 = reg_type(bits=bits)
            reg2 = reg_type(bits=bits)
            qc.add_register(reg1)
            qc.add_register(reg2)

        self.assertEqual(qc.num_qubits, 20)
        self.assertEqual(qc.num_clbits, 10)
        self.assertEqual(qc.num_ancillas, 10)

    def test_from_instructions(self):
        """Test from_instructions method."""

        qreg = QuantumRegister(4)
        creg = ClassicalRegister(3)

        a, b, c, d = qreg
        x, y, z = creg

        circuit_1 = QuantumCircuit(2, 1)
        circuit_1.x(0)
        circuit_2 = QuantumCircuit(2, 1)
        circuit_2.y(0)

        def instructions():
            yield CircuitInstruction(HGate(), [a], [])
            yield CircuitInstruction(CXGate(), [a, b], [])
            yield CircuitInstruction(Measure(), [a], [x])
            yield CircuitInstruction(Measure(), [b], [y])
            yield CircuitInstruction(IfElseOp((z, 1), circuit_1, circuit_2), [c, d], [z])

        def instruction_tuples():
            yield HGate(), [a], []
            yield CXGate(), [a, b], []
            yield CircuitInstruction(Measure(), [a], [x])
            yield Measure(), [b], [y]
            yield IfElseOp((z, 1), circuit_1, circuit_2), [c, d], [z]

        def instruction_tuples_partial():
            yield HGate(), [a]
            yield CXGate(), [a, b], []
            yield CircuitInstruction(Measure(), [a], [x])
            yield Measure(), [b], [y]
            yield IfElseOp((z, 1), circuit_1, circuit_2), [c, d], [z]

        circuit = QuantumCircuit.from_instructions(instructions())
        circuit_tuples = QuantumCircuit.from_instructions(instruction_tuples())
        circuit_tuples_partial = QuantumCircuit.from_instructions(instruction_tuples_partial())

        expected = QuantumCircuit([a, b, c, d], [x, y, z])
        for instruction in instructions():
            expected.append(instruction.operation, instruction.qubits, instruction.clbits)

        self.assertEqual(circuit, expected)
        self.assertEqual(circuit_tuples, expected)
        self.assertEqual(circuit_tuples_partial, expected)

    def test_from_instructions_bit_order(self):
        """Test from_instructions method bit order."""
        qreg = QuantumRegister(2)
        creg = ClassicalRegister(2)
        a, b = qreg
        c, d = creg

        def instructions():
            yield CircuitInstruction(HGate(), [b], [])
            yield CircuitInstruction(CXGate(), [a, b], [])
            yield CircuitInstruction(Measure(), [b], [d])
            yield CircuitInstruction(Measure(), [a], [c])

        circuit = QuantumCircuit.from_instructions(instructions())
        self.assertEqual(circuit.qubits, [b, a])
        self.assertEqual(circuit.clbits, [d, c])

        circuit = QuantumCircuit.from_instructions(instructions(), qubits=qreg)
        self.assertEqual(circuit.qubits, [a, b])
        self.assertEqual(circuit.clbits, [d, c])

        circuit = QuantumCircuit.from_instructions(instructions(), clbits=creg)
        self.assertEqual(circuit.qubits, [b, a])
        self.assertEqual(circuit.clbits, [c, d])

        circuit = QuantumCircuit.from_instructions(
            instructions(), qubits=iter([a, b]), clbits=[c, d]
        )
        self.assertEqual(circuit.qubits, [a, b])
        self.assertEqual(circuit.clbits, [c, d])

    def test_from_instructions_metadata(self):
        """Test from_instructions method passes metadata."""
        qreg = QuantumRegister(2)
        a, b = qreg

        def instructions():
            yield CircuitInstruction(HGate(), [a], [])
            yield CircuitInstruction(CXGate(), [a, b], [])

        circuit = QuantumCircuit.from_instructions(instructions(), name="test", global_phase=0.1)

        expected = QuantumCircuit([a, b], global_phase=0.1)
        for instruction in instructions():
            expected.append(instruction.operation, instruction.qubits, instruction.clbits)

        self.assertEqual(circuit, expected)
        self.assertEqual(circuit.name, "test")

    def test_circuit_has_control_flow_op(self):
        """Test `has_control_flow_op` method"""
        circuit_1 = QuantumCircuit(2, 1)
        circuit_1.x(0)
        circuit_2 = QuantumCircuit(2, 1)
        circuit_2.y(0)

        # Build a circuit
        circ = QuantumCircuit(2, 2)
        circ.h(0)
        circ.cx(0, 1)
        circ.measure(0, 0)
        circ.measure(1, 1)

        # Check if circuit has any control flow operations
        self.assertFalse(circ.has_control_flow_op())

        # Create examples of all control flow operations
        control_flow_ops = [
            IfElseOp((circ.clbits[1], 1), circuit_1, circuit_2),
            WhileLoopOp((circ.clbits[1], 1), circuit_1),
            ForLoopOp((0, 1), None, body=circuit_1),
            SwitchCaseOp(circ.clbits[1], [(0, circuit_1), (1, circuit_2)]),
            BoxOp(circuit_1),
        ]

        # Create combinations of all control flow operations for the
        # circuit.
        op_combinations = [
            comb_list
            for comb_n in range(5)
            for comb_list in combinations(control_flow_ops, comb_n + 1)
        ]

        # Use combinatorics to test all combinations of control flow operations
        # to see if we can detect all of them.
        for op_combination in op_combinations:
            # Build a circuit
            circ = QuantumCircuit(2, 2)
            circ.h(0)
            circ.cx(0, 1)
            circ.measure(0, 0)
            circ.measure(1, 1)
            self.assertFalse(circ.has_control_flow_op())
            for op in op_combination:
                circ.append(op, [0, 1], [1])
            # Check if circuit has any control flow operation
            self.assertTrue(circ.has_control_flow_op())

    @data(True, False)
    def test_ensure_physical_same_size(self, apply_layout):
        """`num_qubits` need not be set."""
        qr = QuantumRegister(10, "virtuals")
        loose = [Qubit() for _ in range(5)]
        num_qubits = len(qr) + len(loose)
        qc = QuantumCircuit(qr, loose)
        qc.h(0)
        qc.cx(0, 1)
        self.assertTrue(qc.ensure_physical(apply_layout=apply_layout))

        physical_qreg = QuantumRegister(num_qubits, "q")
        self.assertEqual(qc.qubits, list(physical_qreg))
        self.assertEqual(qc.qregs, [physical_qreg])
        if apply_layout:
            self.assertEqual(
                qc.layout.initial_virtual_layout(),
                Layout(dict(enumerate(list(qr) + loose))),
            )
            self.assertEqual(qc.layout.final_index_layout(), list(range(num_qubits)))
            self.assertEqual(qc.layout.routing_permutation(), list(range(num_qubits)))
            # After all this, `ensure_physical` should be a no-op.
            self.assertFalse(qc.ensure_physical())
        else:
            self.assertIsNone(qc.layout)

    def test_ensure_physical_can_expand_circuit(self):
        """`ensure_physical` can add ancillas."""
        qr = QuantumRegister(10, "virtuals")
        loose = [Qubit() for _ in range(5)]
        num_virtual_qubits = len(qr) + len(loose)
        num_physical_qubits = 20
        qc = QuantumCircuit(qr, loose)
        qc.h(0)
        qc.cx(0, 1)
        self.assertTrue(qc.ensure_physical(num_physical_qubits))

        ancilla_qreg = QuantumRegister(num_physical_qubits - num_virtual_qubits, "ancilla")
        physical_qreg = QuantumRegister(num_physical_qubits, "q")
        self.assertEqual(qc.qubits, list(physical_qreg))
        self.assertEqual(qc.qregs, [physical_qreg])
        self.assertEqual(
            qc.layout.initial_virtual_layout(filter_ancillas=True),
            Layout(dict(enumerate(list(qr) + loose))),
        )
        self.assertEqual(
            qc.layout.initial_virtual_layout(filter_ancillas=False),
            Layout(dict(enumerate(list(qr) + loose + list(ancilla_qreg)))),
        )
        self.assertEqual(
            qc.layout.final_index_layout(filter_ancillas=True), list(range(num_virtual_qubits))
        )
        self.assertEqual(qc.layout.routing_permutation(), list(range(num_physical_qubits)))

        # After all this, `ensure_physical` should be a no-op.
        self.assertFalse(qc.ensure_physical())

    def test_ensure_physical_equivalence_to_trivial_layout(self):
        """`ensure_physical` is equivalent to the `trivial` layout method in metadata."""
        qr = QuantumRegister(5, "virtuals")
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        num_physical_qubits = 10
        expected = transpile(
            qc,
            backend=GenericBackendV2(
                num_qubits=num_physical_qubits,
                coupling_map=CouplingMap.from_line(10),
                basis_gates=["h", "rz", "sx", "cx"],
                seed=2025_07_22,
            ),
            layout_method="trivial",
            optimization_level=0,
            seed_transpiler=2025_07_22,
        )
        self.assertTrue(qc.ensure_physical(num_physical_qubits))
        self.assertEqual(qc.layout, expected.layout)
        self.assertEqual(qc, expected)

    def test_ensure_physical_no_op_if_physical(self):
        """If the circuit is already physical, nothing should happen."""
        qr = QuantumRegister(5, "virtuals")
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        initial = [4, 5, 6, 7, 8]
        qc = transpile(
            qc,
            backend=GenericBackendV2(
                num_qubits=10,
                coupling_map=CouplingMap.from_line(10),
                basis_gates=["h", "rz", "sx", "cx"],
                seed=2025_07_22,
            ),
            initial_layout=initial,
            optimization_level=0,
            seed_transpiler=2025_07_22,
        )
        self.assertFalse(qc.ensure_physical())
        # The layout shouldn't have changed.
        self.assertEqual(qc.layout.initial_index_layout(filter_ancillas=True), initial)

    def test_ensure_physical_cannot_shrink_circuit(self):
        """It's an error to try and make a circuit narrower."""
        qc = QuantumCircuit(10)
        with self.assertRaisesRegex(ValueError, "cannot have fewer physical qubits"):
            qc.ensure_physical(5)

    def test_estimate_fidelity(self):
        target = Target(num_qubits=1)
        target.add_instruction(XGate(), {(0,): InstructionProperties(error=0.5)})
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        qc.x(0)
        expected = 0.5**3
        fidelity = qc.estimate_fidelity(target)
        self.assertEqual(fidelity, expected)

    def test_estimate_fidelity_with_barrier(self):
        target = Target(num_qubits=1)
        target.add_instruction(XGate(), {(0,): InstructionProperties(error=0.5)})
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier(0)
        qc.x(0)
        qc.barrier(0)
        qc.x(0)
        qc.barrier(0)
        fidelity = qc.estimate_fidelity(target)
        expected = 0.5**3
        self.assertEqual(fidelity, expected)

    def test_estimate_fidelity_non_physical(self):
        target = Target(num_qubits=1)
        target.add_instruction(YGate(), {(0,): InstructionProperties(error=0.5)})
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.x(0)
        qc.x(0)
        fidelity = qc.estimate_fidelity(target)
        self.assertIsNone(fidelity)

    def test_estimate_fidelity_errors_on_control_flow(self):
        target = Target(num_qubits=1)
        target.add_instruction(YGate(), {(0,): InstructionProperties(error=0.5)})
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        with qc.if_test((0, 1)):
            qc.x(0)
            qc.x(0)
        with self.assertRaises(CircuitError):
            qc.estimate_fidelity(target)

    def test_estimate_fidelity_with_ideal_gates(self):
        target = Target(num_qubits=1)
        target.add_instruction(XGate(), {(0,): InstructionProperties(error=0.25)})
        target.add_instruction(YGate(), {(0,): InstructionProperties(error=None)})
        target.add_instruction(ZGate())
        target.add_instruction(SXGate(), {None: InstructionProperties(error=0.1)})
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.y(0)
        qc.z(0)
        qc.sx(0)
        expected = 0.75 * 0.9
        fidelity = qc.estimate_fidelity(target)
        self.assertEqual(fidelity, expected)


class TestCircuitPrivateOperations(QiskitTestCase):
    """Direct tests of some of the private methods of QuantumCircuit.  These do not represent
    functionality that we want to expose to users, but there are some cases where private methods
    are used internally (similar to "protected" access in .NET or "friend" access in C++), and we
    want to make sure they work in those cases."""

    def test_previous_instruction_in_scope_failures(self):
        """Test the failure paths of the peek and pop methods for retrieving the most recent
        instruction in a scope."""
        test = QuantumCircuit(1, 1)
        with self.assertRaisesRegex(CircuitError, r"This circuit contains no instructions\."):
            test._peek_previous_instruction_in_scope()
        with self.assertRaisesRegex(CircuitError, r"This circuit contains no instructions\."):
            test._pop_previous_instruction_in_scope()
        with test.for_loop(range(2)):
            with self.assertRaisesRegex(CircuitError, r"This scope contains no instructions\."):
                test._peek_previous_instruction_in_scope()
            with self.assertRaisesRegex(CircuitError, r"This scope contains no instructions\."):
                test._pop_previous_instruction_in_scope()

    def test_pop_previous_instruction_removes_parameters(self):
        """Test that the private "pop instruction" method removes parameters from the parameter
        table if that instruction is the only instance."""
        x, y = Parameter("x"), Parameter("y")
        test = QuantumCircuit(1, 1)
        test.rx(y, 0)
        last_instructions = list(test.u(x, y, 0, 0))
        self.assertEqual({x, y}, set(test.parameters))

        instruction = test._pop_previous_instruction_in_scope()
        self.assertEqual(last_instructions, [instruction])
        self.assertEqual({y}, set(test.parameters))

    def test_decompose_gate_type(self):
        """Test decompose specifying gate type."""
        circuit = QuantumCircuit(1)
        circuit.append(SGate(label="s_gate"), [0])
        decomposed = circuit.decompose(gates_to_decompose=SGate)
        self.assertNotIn("s", decomposed.count_ops())
