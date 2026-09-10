# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for ProgramStepper and Request, which a backend drives a program with."""

from collections import namedtuple

import numpy as np

from qiskit._accelerate.quantum_program import FunctionBuilder
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_program import (
    DataTree,
    ProgramStepper,
    Request,
    TensorType,
    bit,
    build,
    f64,
    qp_input,
    shot_loop,
)
from qiskit.quantum_program.ops import Add
from test import QiskitTestCase  # pylint: disable=wrong-import-order


#: The two inputs `sum_program` declares, and the output they give.
ARRAYS = {"x": np.array([1.0, 2.0]), "y": np.array([10.0, 20.0])}
EXPECTED = np.array([11.0, 22.0])


def sum_program():
    """A program `f(x, y) = x + y` over `F64[2]` inputs, with one output named `sum`."""
    x = qp_input("x", f64[2])
    y = qp_input("y", f64[2])
    return build({"sum": x + y})


def sampling_program():
    """A partitioned program whose only unit is a shot loop over one 3-qubit circuit.

    Returns the program, and the position of the one function that is the caller's to perform.
    """
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.measure_all()
    outcomes = shot_loop([circuit], shots=8)
    program = build({"excited": outcomes[0]["meas"].mean(axis=0)})
    program, resource = program.partition([["qiskit.shot_loop"]])
    return program, [index for index, unit in enumerate(resource) if unit == 0]


class TestInputs(QiskitTestCase):
    """Tests for how a stepper reads the inputs it is given."""

    def evaluate(self, *args):
        """The one output of a finished stepper of `sum_program` over the given inputs."""
        stepper = ProgramStepper(sum_program(), [], *args)
        stepper.step()
        return stepper.outputs["sum"]

    def test_a_mapping_names_each_input(self):
        """Test that a dict of inputs is read by name."""
        np.testing.assert_array_equal(self.evaluate(ARRAYS), EXPECTED)

    def test_a_namedtuple_names_each_input(self):
        """Test that any value a data tree can be built from is accepted."""
        named = namedtuple("named", ["x", "y"])
        np.testing.assert_array_equal(self.evaluate(named(**ARRAYS)), EXPECTED)

    def test_a_sequence_fills_a_positional_structure(self):
        """Test that a list of inputs fills a structure that names nothing."""
        builder = FunctionBuilder()
        left = builder.add_parameter(f64[2])
        right = builder.add_parameter(f64[2])
        builder.add_result(builder.add_op(Add(), [left, right])[0])
        program = builder.seal(DataTree([None, None]), DataTree([None]))

        stepper = ProgramStepper(program, [], list(ARRAYS.values()))
        stepper.step()
        np.testing.assert_array_equal(stepper.outputs[0], EXPECTED)

    def test_a_data_tree_is_taken_as_it_is(self):
        """Test that a DataTree of inputs is used without being parsed again."""
        np.testing.assert_array_equal(self.evaluate(DataTree(ARRAYS)), EXPECTED)

    def test_a_nested_sequence_is_structure(self):
        """Test that a list among the inputs is structure rather than one array."""
        with self.assertRaisesRegex(ValueError, r"structured \[x: \[_, _\], y: \[_, _\]\]"):
            self.evaluate({"x": [1.0, 2.0], "y": [10.0, 20.0]})
        # `leaf_of` is what holds a sequence as one value.
        leaves = {name: DataTree.leaf_of(list(array)) for name, array in ARRAYS.items()}
        np.testing.assert_array_equal(self.evaluate(DataTree(leaves)), EXPECTED)

    def test_a_list_cannot_fill_named_inputs(self):
        """Test that a list cannot fill a structure that names its inputs."""
        with self.assertRaisesRegex(ValueError, r"structured \[_, _\].*declares \[x: _, y: _\]"):
            self.evaluate(list(ARRAYS.values()))

    def test_a_wrong_input_type_names_the_input(self):
        """Test that an input that does not match its declared type is refused."""
        with self.assertRaisesRegex(ValueError, r"input 'y': expected F64\[2\], got I64\[2\]"):
            self.evaluate({"x": ARRAYS["x"], "y": np.array([1, 2])})

    def test_inputs_are_not_taken_by_keyword(self):
        """Test that an input cannot be named as a keyword argument."""
        with self.assertRaises(TypeError):
            ProgramStepper(sum_program(), [], **ARRAYS)

    def test_no_inputs_when_none_are_declared(self):
        """Test that a program taking no inputs needs no inputs argument."""
        program, external = sampling_program()
        self.assertIsNone(ProgramStepper(program, external).outputs)


class TestExternal(QiskitTestCase):
    """Tests for which functions an stepper hands over."""

    def test_external_is_required(self):
        """Test that the functions to hand over are not optional."""
        with self.assertRaises(TypeError):
            ProgramStepper(sum_program())

    def test_a_negative_position_counts_from_the_end(self):
        """Test that a function is addressed from the end when its position is negative."""
        program, _ = sampling_program()
        # @0 is the unit and @1 is the entry point, so -2 is the unit.
        stepper = ProgramStepper(program, [-2])
        stepper.step()
        self.assertEqual(len(stepper.outstanding()), 1)

    def test_an_out_of_range_position_refused(self):
        """Test that a position outside the program is refused."""
        with self.assertRaisesRegex(IndexError, "function 4 is out of range"):
            ProgramStepper(sum_program(), [4], ARRAYS)

    def test_the_entry_point_cannot_be_handed_over(self):
        """Test that declaring the entry point external is refused."""
        with self.assertRaisesRegex(ValueError, "@0 is the entry point"):
            ProgramStepper(sum_program(), [0], ARRAYS)

    def test_work_qiskit_cannot_do_is_refused(self):
        """Test that a shot loop Qiskit is left to run is refused before anything runs."""
        program, _ = sampling_program()
        with self.assertRaisesRegex(ValueError, "qiskit.shot_loop.*no built-in implementation"):
            ProgramStepper(program, [])


class TestStepping(QiskitTestCase):
    """Tests for driving an stepper to its outputs."""

    def test_a_local_program_finishes_in_one_step(self):
        """Test that a program with nothing external is finished by one step."""
        stepper = ProgramStepper(sum_program(), [], ARRAYS)
        self.assertIsNone(stepper.outputs)
        stepper.step()
        self.assertIsInstance(stepper.outputs, DataTree)
        np.testing.assert_array_equal(stepper.outputs["sum"], [11.0, 22.0])

    def test_the_full_loop_over_a_partitioned_program(self):
        """Test that a shot loop handed over is answered and the rest evaluated."""
        program, external = sampling_program()
        stepper = ProgramStepper(program, external)

        answered = 0
        while (outputs := stepper.outputs) is None:
            stepper.step()
            for request in stepper.outstanding():
                samples = [np.ones(ty.shape, dtype=bool) for ty in request.output_types]
                stepper.fulfil(request.id, samples)
                answered += 1

        self.assertEqual(answered, 1)
        np.testing.assert_array_equal(outputs["excited"], [1.0, 1.0, 1.0])

    def test_stepping_past_the_end_is_a_no_op(self):
        """Test that a step past the end leaves the outputs alone."""
        stepper = ProgramStepper(sum_program(), [], ARRAYS)
        stepper.step()
        stepper.step()
        np.testing.assert_array_equal(stepper.outputs["sum"], [11.0, 22.0])

    def test_repr_reports_what_is_outstanding(self):
        """Test that the repr says how much is outstanding and whether the program finished."""
        program, external = sampling_program()
        stepper = ProgramStepper(program, external)
        stepper.step()
        self.assertEqual(repr(stepper), "ProgramStepper(1 outstanding, finished=False)")


class TestRequest(QiskitTestCase):
    """Tests for the work an stepper hands over."""

    def setUp(self):
        super().setUp()
        program, external = sampling_program()
        self.program = program
        self.stepper = ProgramStepper(program, external)
        self.stepper.step()
        self.request = self.stepper.outstanding()[0]

    def test_a_request_names_the_function_to_perform(self):
        """Test that a request reports its function both as a reader and by position."""
        self.assertEqual(self.request.function_id, 0)
        self.assertEqual(self.request.function.index, 0)
        self.assertFalse(self.request.function.is_entry)
        self.assertEqual(
            [instruction.full_name for instruction in self.request.function],
            ["qiskit.parameter", "qiskit.shot_loop", "qiskit.result"],
        )

    def test_a_request_holds_its_inputs(self):
        """Test that a request hands over one read-only array per parameter of its function."""
        self.assertEqual(len(self.request.inputs), 1)
        array = self.request.inputs[0]
        self.assertEqual(array.shape, (0,))
        self.assertFalse(array.flags.writeable)

    def test_a_request_declares_what_it_must_produce(self):
        """Test that a request reports one tensor type per result of its function."""
        self.assertEqual(self.request.output_types, [TensorType(bit, (8, 3))])

    def test_a_request_reports_its_id(self):
        """Test that the id is what identifies a request and what answers it."""
        self.assertIsInstance(self.request, Request)
        self.assertEqual(self.request.id, 0)
        self.assertEqual(repr(self.request), "Request(#0, @0)")

    def test_a_call_names_the_same_function(self):
        """Test that a call instruction reports its callee by position as well as by reader."""
        call = next(
            instruction
            for instruction in self.program.entry
            if instruction.full_name == "qiskit.call"
        )
        self.assertEqual(call.callee_id, self.request.function_id)
        self.assertEqual(call.callee.index, self.request.function_id)

    def test_a_non_call_has_no_callee(self):
        """Test that only a call instruction names a function."""
        for instruction in self.request.function:
            self.assertIsNone(instruction.callee_id)
            self.assertIsNone(instruction.callee)


class TestFulfil(QiskitTestCase):
    """Tests for answering a request."""

    def setUp(self):
        super().setUp()
        program, external = sampling_program()
        self.stepper = ProgramStepper(program, external)
        self.stepper.step()
        self.samples = np.ones((8, 3), dtype=bool)

    def test_a_request_is_answered_by_its_id(self):
        """Test that answering a request takes it out of the outstanding ones."""
        self.stepper.fulfil(0, [self.samples])
        self.assertEqual(self.stepper.outstanding(), [])

    def test_answering_evaluates_nothing_of_its_own(self):
        """Test that the outputs arrive only on the next step."""
        self.stepper.fulfil(0, [self.samples])
        self.assertIsNone(self.stepper.outputs)
        self.stepper.step()
        self.assertIsNotNone(self.stepper.outputs)

    def test_the_wrong_number_of_outputs_is_refused(self):
        """Test that an answer of the wrong length is refused."""
        with self.assertRaisesRegex(ValueError, r"request #0 takes 1 output\(s\), got 2"):
            self.stepper.fulfil(0, [self.samples, self.samples])

    def test_an_output_of_the_wrong_type_is_refused(self):
        """Test that an answer whose type the call does not declare is refused."""
        with self.assertRaisesRegex(ValueError, r"expected Bit\[8, 3\], got F64\[8, 3\]"):
            self.stepper.fulfil(0, [np.ones((8, 3))])

    def test_a_refused_answer_can_be_retried(self):
        """Test that a request can still be answered after a rejection."""
        with self.assertRaises(ValueError):
            self.stepper.fulfil(0, [])
        self.stepper.fulfil(0, [self.samples])
        self.assertEqual(self.stepper.outstanding(), [])

    def test_a_request_is_answered_only_once(self):
        """Test that answering a request twice is refused."""
        self.stepper.fulfil(0, [self.samples])
        with self.assertRaisesRegex(ValueError, "request #0 is not outstanding"):
            self.stepper.fulfil(0, [self.samples])

    def test_an_unknown_id_is_refused(self):
        """Test that an id no request has is refused."""
        with self.assertRaisesRegex(ValueError, "request #3 is not outstanding"):
            self.stepper.fulfil(3, [self.samples])
