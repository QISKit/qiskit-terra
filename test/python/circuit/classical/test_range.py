# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Range expression class."""

import ddt
from qiskit.circuit.classical import expr, types
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt.ddt
class TestRange(QiskitTestCase):
    """Test the Range expression class."""

    def test_range_with_uint(self):
        """Test creating a Range with Uint values."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))
        step = expr.lift(2, types.Uint(8))

        range_expr = expr.Range(start, stop, step)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, step)
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertTrue(range_expr.const)

    def test_range_with_different_uint_sizes(self):
        """Test creating a Range with different Uint sizes."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        range_expr = expr.Range(start, stop, step)

        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertTrue(range_expr.const)

        self.assertIsInstance(range_expr.start, expr.Cast)
        self.assertEqual(range_expr.start.type, types.Uint(32))
        self.assertEqual(range_expr.start.operand, start)
        self.assertTrue(range_expr.start.implicit)

        self.assertIsInstance(range_expr.stop, expr.Cast)
        self.assertEqual(range_expr.stop.type, types.Uint(32))
        self.assertEqual(range_expr.stop.operand, stop)
        self.assertTrue(range_expr.stop.implicit)

        self.assertEqual(range_expr.step, step)
        self.assertNotIsInstance(range_expr.step, expr.Cast)

    def test_range_without_step(self):
        """Test creating a Range without a step value."""
        start = expr.lift(0, types.Uint(8))
        stop = expr.lift(5, types.Uint(8))

        range_expr = expr.Range(start, stop)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, expr.lift(1, types.Uint(8)))
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertTrue(range_expr.const)

    def test_len_constant_matches_inclusive_openqasm_range(self):
        """Constant Range len() matches OpenQASM 3 inclusive-stop semantics."""
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(5, types.Uint(8)))
        # Inclusive [0:1:5] → 0..5 inclusive (Python exclusive equivalent: range(6)).
        self.assertEqual(len(range_expr), len(range(6)))
        self.assertEqual(len(range_expr), 6)

        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(10, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        self.assertEqual(len(range_expr), len(range(0, 11, 2)))
        self.assertEqual(len(range_expr), 6)

        singleton = expr.Range(expr.lift(5, types.Uint(8)), expr.lift(5, types.Uint(8)))
        self.assertEqual(len(singleton), 1)

        empty = expr.Range(expr.lift(6, types.Uint(8)), expr.lift(5, types.Uint(8)))
        self.assertEqual(len(empty), 0)

    def test_len_non_constant_placeholder(self):
        """Non-constant Range len() returns a conservative placeholder."""
        cr = ClassicalRegister(8, "c")
        range_expr = expr.Range(expr.lift(cr), expr.lift(10, types.Uint(8)))
        self.assertFalse(range_expr.const)
        self.assertEqual(len(range_expr), 1)

    @ddt.data(
        # Wrong start, correct stop, no step
        {
            "start": expr.lift(5.0, types.Float()),
            "stop": expr.lift(10, types.Uint(8)),
            "step": None,
        },
        # Correct start, wrong stop, no step
        {
            "start": expr.lift(5, types.Uint(8)),
            "stop": expr.lift(10.0, types.Float()),
            "step": None,
        },
        # Correct start/stop, wrong step
        {
            "start": expr.lift(0, types.Uint(8)),
            "stop": expr.lift(10, types.Uint(8)),
            "step": expr.lift(1.5, types.Float()),
        },
        # Wrong start and stop, no step
        {
            "start": expr.lift(5.0, types.Float()),
            "stop": expr.lift(10.0, types.Float()),
            "step": None,
        },
    )
    @ddt.unpack
    def test_range_with_wrong_expr_types(self, start, stop, step):
        """Test that creating a Range with wrong expr types raises an error.

        Each dataset isolates the wrong variable to avoid shadowing, with a case
        where both bounds are wrong included for completeness.
        """
        with self.assertRaisesRegex(TypeError, "Range values must be of unsigned integer type"):
            if step is None:
                expr.Range(start, stop)
            else:
                expr.Range(start, stop, step)

    def test_range_with_non_constant_values(self):
        """Test creating a Range with non-constant values."""

        cr = ClassicalRegister(8, "c")

        start = expr.lift(cr)
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop)

        self.assertEqual(range_expr.start, start)
        self.assertEqual(range_expr.stop, stop)
        self.assertEqual(range_expr.step, expr.lift(1, types.Uint(8)))
        self.assertEqual(range_expr.type, types.Uint(8))
        self.assertFalse(range_expr.const)

    def test_range_with_invalid_type_specification(self):
        """Test that specifying a non-unsigned integer type raises an error."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        with self.assertRaisesRegex(TypeError, "Range type must be an unsigned integer type"):
            expr.Range(start, stop, ty=types.Float())

    def test_range_with_valid_uint_type_specification(self):
        """Test that specifying a valid Uint type works correctly."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop, ty=types.Uint(16))
        self.assertEqual(range_expr.type, types.Uint(16))
        self.assertEqual(range_expr.start.type, types.Uint(16))
        self.assertEqual(range_expr.stop.type, types.Uint(16))

    def test_range_with_different_input_types(self):
        """Test that values are lifted to the specified type."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        range_expr = expr.Range(start, stop, step, ty=types.Uint(64))
        self.assertEqual(range_expr.type, types.Uint(64))
        self.assertEqual(range_expr.start.type, types.Uint(64))
        self.assertEqual(range_expr.stop.type, types.Uint(64))
        self.assertEqual(range_expr.step.type, types.Uint(64))

    def test_range_with_step_lifted_to_type(self):
        """Test that step value is lifted to the specified type."""
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))

        range_expr = expr.Range(start, stop, step=expr.lift(2, types.Uint(16)), ty=types.Uint(32))
        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertEqual(range_expr.start.type, types.Uint(32))
        self.assertEqual(range_expr.stop.type, types.Uint(32))
        self.assertEqual(range_expr.step.type, types.Uint(32))

    def test_range_with_valid_type_casting(self):
        """Test that values are properly cast to the specified type."""
        # Create expressions with different types
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        # Cast all to Uint(64)
        range_expr = expr.Range(start, stop, step, ty=types.Uint(64))

        # Verify all expressions were cast to Uint(64)
        self.assertEqual(range_expr.type, types.Uint(64))
        self.assertEqual(range_expr.start.type, types.Uint(64))
        self.assertEqual(range_expr.stop.type, types.Uint(64))
        self.assertEqual(range_expr.step.type, types.Uint(64))

        # Verify the expressions are Cast nodes
        self.assertIsInstance(range_expr.start, expr.Cast)
        self.assertIsInstance(range_expr.stop, expr.Cast)
        self.assertIsInstance(range_expr.step, expr.Cast)

        # Verify the casts are explicit (not implicit)
        self.assertFalse(range_expr.start.implicit)
        self.assertFalse(range_expr.stop.implicit)
        self.assertFalse(range_expr.step.implicit)

    def test_range_with_invalid_type_casting(self):
        """Test that invalid type casts raise appropriate errors."""
        # Create expressions with different types
        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(16))
        step = expr.lift(2, types.Uint(32))

        # Try to cast to a non-uint type (should fail)
        with self.assertRaisesRegex(TypeError, "Range type must be an unsigned integer type"):
            expr.Range(start, stop, step, ty=types.Float())

    def test_range_with_mixed_type_casting(self):
        """Test casting with mixed constant and non-constant expressions."""

        cr = ClassicalRegister(8, "c")
        start = expr.lift(cr)  # Non-constant
        stop = expr.lift(10, types.Uint(8))  # Constant
        step = expr.lift(2, types.Uint(16))  # Constant

        # Cast all to Uint(32)
        range_expr = expr.Range(start, stop, step, ty=types.Uint(32))

        # Verify all expressions were cast to Uint(32)
        self.assertEqual(range_expr.type, types.Uint(32))
        self.assertEqual(range_expr.start.type, types.Uint(32))
        self.assertEqual(range_expr.stop.type, types.Uint(32))
        self.assertEqual(range_expr.step.type, types.Uint(32))

        # Verify constant flag is preserved
        self.assertFalse(range_expr.start.const)  # Non-constant
        self.assertTrue(range_expr.stop.const)  # Constant
        self.assertTrue(range_expr.step.const)  # Constant

    def test_range_in_forloop_basic(self):
        """Test that Range can be used in ForLoop with basic constant values."""
        qc = QuantumCircuit(1, 1)

        # Create a Range with constant values
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(5, types.Uint(8)))

        with qc.for_loop(range_expr):
            qc.h(0)
            qc.measure(0, 0)

        # Check that the ForLoop instruction exists
        instruction = qc.data[0]
        self.assertEqual(instruction.operation.name, "for_loop")
        # Unused auto-generated loop variable is dropped.
        self.assertIsNone(instruction.operation.params[1])

        # Check that the Range parameters are correctly stored
        self.assertIsInstance(range_expr, expr.Range)
        self.assertEqual(range_expr.start, expr.lift(0, types.Uint(8)))
        self.assertEqual(range_expr.stop, expr.lift(5, types.Uint(8)))
        self.assertEqual(range_expr.step, expr.lift(1, types.Uint(8)))

    def test_range_in_forloop_auto_generates_var_when_used(self):
        """Using the loop variable in the body keeps an auto-generated expr.Var.

        The loop variable is the body's ``input`` variable; using it to index a register (rather
        than closing over an outer ``expr.Var``) keeps the body free of captures, which a body
        with an input variable is not allowed to have.
        """
        cr = ClassicalRegister(8, "cr")
        qc = QuantumCircuit(1)
        qc.add_register(cr)
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(5, types.Uint(8)))

        with qc.for_loop(range_expr) as loop_var:
            qc.store(expr.index(cr, loop_var), True)

        for_loop = next(inst.operation for inst in qc.data if inst.operation.name == "for_loop")
        self.assertIsInstance(for_loop.params[1], expr.Var)
        self.assertEqual(for_loop.params[1].type, range_expr.type)
        self.assertIn(for_loop.params[1], list(for_loop.params[2].iter_input_vars()))

    def test_range_in_forloop_with_step(self):
        """Test that Range with step can be used in ForLoop."""
        qc = QuantumCircuit(1, 1)

        # Create a Range with step
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(10, types.Uint(8)), expr.lift(2, types.Uint(8))
        )

        with qc.for_loop(range_expr):
            qc.h(0)
            qc.measure(0, 0)

        # Check the Range parameters
        instruction = qc.data[0]
        range_param = instruction.operation.params[0]
        self.assertEqual(range_param.start, expr.lift(0, types.Uint(8)))
        self.assertEqual(range_param.stop, expr.lift(10, types.Uint(8)))
        self.assertEqual(range_param.step, expr.lift(2, types.Uint(8)))

    def test_range_in_forloop_with_variables(self):
        """Test that Range with variables captures them in circuit variables."""
        qc = QuantumCircuit(1, 1)

        # Create variables for the Range
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        step_var = qc.add_var("step", expr.lift(2, types.Uint(6)))

        # Create a Range with variables
        range_expr = expr.Range(start_var, stop_var, step_var)

        with qc.for_loop(range_expr):
            qc.h(0)
            qc.measure(0, 0)

        # Verify the Range contains the expected variables
        self.assertEqual(range_expr.start, expr.Cast(start_var, types.Uint(10), implicit=True))
        self.assertEqual(range_expr.stop, stop_var)
        self.assertEqual(range_expr.step, expr.Cast(step_var, types.Uint(10), implicit=True))

        # Check that all variables in the Range are captured in circuit variables
        circuit_vars = qc.iter_vars()
        self.assertIn(start_var, circuit_vars)
        self.assertIn(stop_var, circuit_vars)
        self.assertIn(step_var, circuit_vars)

    def test_range_with_no_explicit_step(self):
        """Test that Range with no explicit step is inferred correctly."""
        qc = QuantumCircuit(1, 1)
        start = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop = qc.add_var("stop", expr.lift(10, types.Uint(10)))
        range_expr = expr.Range(start, stop)
        with qc.for_loop(range_expr):
            qc.h(0)
            qc.measure(0, 0)
        self.assertEqual(range_expr.step, expr.lift(1, types.Uint(10)))
        self.assertEqual(range_expr.start, expr.Cast(start, types.Uint(10), implicit=True))
        self.assertEqual(range_expr.stop, stop)

    def test_range_deepcopy(self):
        """Test that Range objects can be deep copied."""
        import copy

        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))
        step = expr.lift(2, types.Uint(8))

        range_expr = expr.Range(start, stop, step)

        # Test shallow copy
        range_copy = copy.copy(range_expr)
        self.assertIs(range_copy, range_expr)  # Should be the same object (immutable)

        # Test deep copy
        range_deepcopy = copy.deepcopy(range_expr)
        self.assertIs(range_deepcopy, range_expr)  # Should be the same object (immutable)

        # Verify the copied object has the same properties
        self.assertEqual(range_deepcopy.start, range_expr.start)
        self.assertEqual(range_deepcopy.stop, range_expr.stop)
        self.assertEqual(range_deepcopy.step, range_expr.step)
        self.assertEqual(range_deepcopy.type, range_expr.type)
        self.assertEqual(range_deepcopy.const, range_expr.const)

    def test_range_pickle(self):
        """Test that Range objects can be pickled and unpickled."""
        import pickle

        start = expr.lift(5, types.Uint(8))
        stop = expr.lift(10, types.Uint(8))
        step = expr.lift(2, types.Uint(8))

        range_expr = expr.Range(start, stop, step)

        # Pickle the Range object
        pickled = pickle.dumps(range_expr)

        # Unpickle the Range object
        unpickled = pickle.loads(pickled)

        # Verify the unpickled object has the same properties
        self.assertEqual(unpickled.start, range_expr.start)
        self.assertEqual(unpickled.stop, range_expr.stop)
        self.assertEqual(unpickled.step, range_expr.step)
        self.assertEqual(unpickled.type, range_expr.type)
        self.assertEqual(unpickled.const, range_expr.const)

        # Test with non-constant values
        cr = ClassicalRegister(8, "c")
        start_var = expr.lift(cr)
        stop_var = expr.lift(10, types.Uint(8))

        range_expr_var = expr.Range(start_var, stop_var)

        # Pickle the Range object with variables
        pickled_var = pickle.dumps(range_expr_var)

        # Unpickle the Range object with variables
        unpickled_var = pickle.loads(pickled_var)

        # Verify the unpickled object has the same properties
        self.assertEqual(unpickled_var.start, range_expr_var.start)
        self.assertEqual(unpickled_var.stop, range_expr_var.stop)
        self.assertEqual(unpickled_var.step, range_expr_var.step)
        self.assertEqual(unpickled_var.type, range_expr_var.type)
        self.assertEqual(unpickled_var.const, range_expr_var.const)
