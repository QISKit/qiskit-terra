# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test circuits with variable parameters."""

import unittest
import cmath
import math
import pickle
from operator import add, mul, sub, truediv
import numpy
from ddt import data, ddt, named_data

import qiskit
import qiskit.circuit.library as circlib
from qiskit.circuit.library import RZGate, PauliEvolutionGate
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Instruction, Parameter, ParameterExpression, ParameterVector
from qiskit.circuit.parametertable import ParameterView
from qiskit.circuit.exceptions import CircuitError
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator, Pauli
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.utils import parallel_map
from test import QiskitTestCase, combine
from ..legacy_cmaps import BOGOTA_CMAP


def raise_if_parameter_table_invalid(circuit):
    """Validates the internal consistency of a ParameterTable and its
    containing QuantumCircuit. Intended for use in testing.

    Raises:
       CircuitError: if QuantumCircuit and ParameterTable are inconsistent.
    """

    # Assert parameters present in circuit match those in table.
    circuit_parameters = {
        parameter
        for instruction in circuit._data
        for param in instruction.operation.params
        for parameter in param.parameters
        if isinstance(param, ParameterExpression)
    }
    if isinstance(circuit.global_phase, ParameterExpression):
        circuit_parameters |= circuit.global_phase.parameters
    table_parameters = set(circuit._data.unsorted_parameters())

    if circuit_parameters != table_parameters:
        raise CircuitError(
            "Circuit/ParameterTable Parameter mismatch. "
            f"Circuit parameters: {circuit_parameters}. "
            f"Table parameters: {table_parameters}."
        )

    # Assert parameter locations in table are present in circuit.
    circuit_instructions = [instr.operation for instr in circuit._data]

    for parameter in table_parameters:
        instr_list = circuit._data._raw_parameter_table_entry(parameter)
        for instr_index, param_index in instr_list:
            if instr_index is None:
                # Global phase.
                expression = circuit.global_phase
                instr = "<global phase>"
            else:
                instr = circuit.data[instr_index].operation
                if instr not in circuit_instructions:
                    raise CircuitError(
                        f"ParameterTable instruction not present in circuit: {instr}."
                    )
                expression = instr.params[param_index]

            if not isinstance(expression, ParameterExpression):
                raise CircuitError(
                    "ParameterTable instruction does not have a "
                    f"ParameterExpression at param_index {param_index}: {instr}."
                )

            if parameter not in expression.parameters:
                raise CircuitError(
                    "ParameterTable instruction parameters does not match ParameterTable key."
                    f"\nInstruction parameters: {expression.parameters}"
                    f"\nParameterTable key: {parameter}."
                )

    # Assert circuit has no other parameter locations other than those in table.
    for instr_index, instruction in enumerate(circuit._data):
        for param_index, param in enumerate(instruction.operation.params):
            if isinstance(param, ParameterExpression):
                parameters = param.parameters

                for parameter in parameters:
                    if (instr_index, param_index) not in circuit._data._raw_parameter_table_entry(
                        parameter
                    ):
                        raise CircuitError(
                            "Found parameterized instruction not "
                            f"present in table. Instruction: {instruction.operation} "
                            f"param_index: {param_index}"
                        )


@ddt
class TestParameters(QiskitTestCase):
    """Test Parameters."""

    def test_equality(self):
        """Test Parameter equality"""
        param = Parameter("a")
        param_copy = Parameter(param.name, uuid=param.uuid)
        param_different = Parameter("a")
        param_same_uuid_diff_name = Parameter("b", uuid=param.uuid)

        self.assertEqual(param, param, "Parameter does not equal itself")
        self.assertEqual(param, param_copy, "Parameters with same data are not equal")
        self.assertNotEqual(param, param_different, "Different Parameters are treated as equal")
        self.assertNotEqual(
            param, param_same_uuid_diff_name, "Parameters with different names are treated as equal"
        )

    def test_gate(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter("θ")
        theta_gate = Gate("test", 1, params=[theta])
        self.assertEqual(theta_gate.name, "test")
        self.assertIsInstance(theta_gate.params[0], Parameter)

    def test_compile_quantum_circuit(self):
        """Test instantiating gate with variable parameters"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicSimulator()
        qc_aer = transpile(qc, backend)
        self.assertIn(theta, qc_aer.parameters)

    def test_duplicate_name_on_append(self):
        """Test adding a second parameter object with the same name fails."""
        param_a = Parameter("a")
        param_a_again = Parameter("a")
        qc = QuantumCircuit(1)
        qc.rx(param_a, 0)
        self.assertRaises(CircuitError, qc.rx, param_a_again, 0)

    def test_append_copies_parametric(self):
        """Test that `QuantumCircuit.append` copies instructions when they contain compile
        parameters and expressions."""
        param = Parameter("a")
        expr = param * 2
        gate_param = RZGate(param)
        gate_expr = RZGate(expr)

        qc = QuantumCircuit(1)
        qc.append(gate_param, [0], copy=True)
        self.assertIsNot(qc.data[-1].operation, gate_param)
        self.assertEqual(qc.data[-1].operation, gate_param)

        # Standard gates are not stored as Python objects so a fresh object
        # is always instantiated on accessing `CircuitInstruction.operation`
        qc.append(gate_param, [0], copy=False)
        self.assertEqual(qc.data[-1].operation, gate_param)

        qc.append(gate_expr, [0], copy=True)
        self.assertIsNot(qc.data[-1].operation, gate_expr)
        self.assertEqual(qc.data[-1].operation, gate_expr)

        # Standard gates are not stored as Python objects so a fresh object
        # is always instantiated on accessing `CircuitInstruction.operation`
        qc.append(gate_expr, [0], copy=False)
        self.assertEqual(qc.data[-1].operation, gate_expr)

    def test_parameters_property(self):
        """Test instantiating gate with variable parameters"""
        from qiskit.circuit.library.standard_gates.rx import RXGate

        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        rxg = RXGate(theta)
        qc.append(rxg, [qr[0]], [])
        self.assertEqual(qc._data.num_parameters(), 1)
        self.assertEqual(theta, next(iter(qc._data.unsorted_parameters())))
        ((instruction_index, _),) = list(qc._data._raw_parameter_table_entry(theta))
        self.assertEqual(rxg, qc.data[instruction_index].operation)

    def test_parameters_property_by_index(self):
        """Test getting parameters by index"""
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        qc = QuantumCircuit(1)
        qc.rx(x, 0)
        qc.rz(z, 0)
        qc.ry(y, 0)
        qc.u(*v, 0)
        self.assertEqual(x, qc.parameters[3])
        self.assertEqual(y, qc.parameters[4])
        self.assertEqual(z, qc.parameters[5])
        for i, vi in enumerate(v):
            self.assertEqual(vi, qc.parameters[i])

    def test_parameters_property_independent_after_copy(self):
        """Test that any `parameters` property caching is invalidated after a copy operation."""
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        qc1 = QuantumCircuit(1)
        qc1.rz(a, 0)
        self.assertEqual(set(qc1.parameters), {a})

        qc2 = qc1.copy_empty_like()
        self.assertEqual(set(qc2.parameters), set())

        qc3 = qc1.copy()
        self.assertEqual(set(qc3.parameters), {a})
        qc3.rz(b, 0)
        self.assertEqual(set(qc3.parameters), {a, b})
        self.assertEqual(set(qc1.parameters), {a})

        qc1.rz(c, 0)
        self.assertEqual(set(qc1.parameters), {a, c})
        self.assertEqual(set(qc3.parameters), {a, b})

    def test_get_parameter(self):
        """Test the `get_parameter` method."""
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)

        qc = QuantumCircuit(1)
        qc.rx(x + y + z + sum(v), 0)

        self.assertEqual(qc.get_parameter("x"), x)
        self.assertEqual(qc.get_parameter("y"), y)
        self.assertEqual(qc.get_parameter("z"), z)
        self.assertEqual(qc.get_parameter(v[1].name), v[1])

        self.assertIsNone(qc.get_parameter("abc", None))
        self.assertEqual(qc.get_parameter("jfkdla", "not present"), "not present")

        with self.assertRaisesRegex(KeyError, "no parameter named"):
            qc.get_parameter("jfklda")

    def test_get_parameter_global_phase(self):
        """Test that `get_parameter` works on parameters that only appear in the global phase."""
        x = Parameter("x")
        qc = QuantumCircuit(0, global_phase=x)

        self.assertEqual(qc.get_parameter("x"), x)
        self.assertIsNone(qc.get_parameter("y", None), None)

    def test_setting_global_phase_invalidates_cache(self):
        """Test that setting the global phase to a non-parametric value invalidates the `parameters`
        cache of the circuit."""
        x = Parameter("x")
        qc = QuantumCircuit(0, global_phase=x)
        self.assertEqual(qc.global_phase, x)
        self.assertEqual(set(qc.parameters), {x})
        qc.global_phase = 0
        self.assertEqual(qc.global_phase, 0)
        self.assertEqual(set(qc.parameters), set())

        qc = QuantumCircuit(0, global_phase=0)
        self.assertEqual(qc.global_phase, 0)
        self.assertEqual(set(qc.parameters), set())
        qc.global_phase = x
        self.assertEqual(qc.global_phase, x)
        self.assertEqual(set(qc.parameters), {x})

    def test_has_parameter(self):
        """Test the `has_parameter` method."""
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)

        qc = QuantumCircuit(1)
        qc.rx(x + y + z + sum(v), 0)

        self.assertTrue(qc.has_parameter("x"))
        self.assertTrue(qc.has_parameter("y"))
        self.assertTrue(qc.has_parameter("z"))
        self.assertTrue(qc.has_parameter(v[1].name))

        self.assertFalse(qc.has_parameter("abc"))
        self.assertFalse(qc.has_parameter("jfkdla"))

        self.assertTrue(qc.has_parameter(x))
        self.assertTrue(qc.has_parameter(y))

        # This `z` should compare unequal to the first one, so it should appear absent.
        self.assertFalse(qc.has_parameter(Parameter("z")))

    def test_bind_parameters_anonymously(self):
        """Test setting parameters by insertion order anonymously"""
        phase = Parameter("phase")
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        qc = QuantumCircuit(1, global_phase=phase)
        qc.rx(x, 0)
        qc.rz(z, 0)
        qc.ry(y, 0)
        qc.u(*v, 0)
        params = [0.1 * i for i in range(len(qc.parameters))]

        order = [phase] + v[:] + [x, y, z]
        param_dict = dict(zip(order, params))
        bqc_anonymous = qc.assign_parameters(params)
        bqc_list = qc.assign_parameters(param_dict)
        self.assertEqual(bqc_anonymous, bqc_list)

    def test_bind_parameters_allow_unknown(self):
        """Test binding parameters allowing unknown parameters."""
        a = Parameter("a")
        b = Parameter("b")
        c = a.bind({a: 1, b: 1}, allow_unknown_parameters=True)
        self.assertEqual(c, a.bind({a: 1}))

    def test_bind_all(self):
        """Fast-path for binding numeric results."""
        a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
        bindings = {a: -1.0, b: 2.5, c: 7}
        self.assertEqual((a + b).bind_all(bindings), 1.5)
        self.assertEqual(a.bind_all(bindings), -1.0)

    def test_bind_all_failures(self):
        """`bind_all` does not explicitly check failures for performance, but we want failure to not
        pass completely silently."""
        a, b = Parameter("a"), Parameter("b")
        # Feel free to change the exact exceptions; it's only important that it's not silent.
        expr = a + b
        with self.assertRaises(KeyError):
            expr.bind_all({a: -1.0})
        with self.assertRaises(KeyError):
            a.bind_all({b: 2.5})

    def test_assign_parameters_by_name(self):
        """Test that parameters can be assigned by name as well as value."""
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        qc = QuantumCircuit(2, global_phase=a * 2)
        qc.rx(b + 0.125 * c, 0)

        self.assertEqual(
            qc.assign_parameters({a: 1, b: 2, c: 3}), qc.assign_parameters({"a": 1, "b": 2, "c": 3})
        )

    def test_assign_parameters_by_iterable(self):
        """Assignment works with weird iterables."""
        a, b, c = Parameter("a"), Parameter("b"), Parameter("c")
        qc = QuantumCircuit(1)
        qc.rz(a, 0)
        qc.rz(b + c, 0)

        binds = [1.25, 2.5, 0.125]
        expected = qc.assign_parameters(dict(zip(qc.parameters, binds)))
        self.assertEqual(qc.assign_parameters(iter(binds)), expected)
        self.assertEqual(qc.assign_parameters(dict.fromkeys(binds).keys()), expected)
        self.assertEqual(qc.assign_parameters(dict(zip(qc.parameters, binds)).values()), expected)
        self.assertEqual(qc.assign_parameters(bind for bind in binds), expected)

    def test_assign_parameters_with_cache(self):
        """Test assigning parameters on a circuit with already triggered cache."""
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.append(RZGate(x), [0])  # add via ``append`` to create a CircuitInstruction

        _ = qc.data[0].operation.definition  # trigger building the cache

        binds = [1.2]
        qc.assign_parameters(binds, inplace=True)

        self.assertAlmostEqual(binds[0], qc.data[0].operation.params[0])

    def test_assign_parameters_with_string_values_and_strict_equals_false(self):
        """Test that a string parameter with strict=False does not return an error"""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        qc.rz(a, 0)
        bound = qc.assign_parameters({"a": 1.0, "b": 2.0}, strict=False)
        expected = qc.assign_parameters({a: 1.0})
        self.assertEqual(bound, expected)

    def test_bind_parameters_custom_definition_global_phase(self):
        """Test that a custom gate with a parametrized `global_phase` is assigned correctly."""
        x = Parameter("x")
        custom = QuantumCircuit(1, global_phase=x).to_gate()
        base = QuantumCircuit(1)
        base.append(custom, [0], [])

        test = Operator(base.assign_parameters({x: math.pi}))
        expected = Operator(numpy.array([[-1, 0], [0, -1]]))
        self.assertEqual(test, expected)

    def test_bind_half_single_precision(self):
        """Test binding with 16bit and 32bit floats."""
        phase = Parameter("phase")
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        v = ParameterVector("v", 3)
        for i in (numpy.float16, numpy.float32):
            with self.subTest(float_type=i):
                expr = (v[0] * (x + y + z) + phase) - (v[2] * v[1])
                params = numpy.array([0.1 * j for j in range(8)], dtype=i)
                order = [phase] + v[:] + [x, y, z]
                param_dict = dict(zip(order, params))
                bound_value = expr.bind(param_dict)
                self.assertAlmostEqual(float(bound_value), 0.09, delta=1e-4)

    def test_parameter_order(self):
        """Test the parameters are sorted by name but parameter vector order takes precedence.

        This means that the following set of parameters

            {a, z, x[0], x[1], x[2], x[3], x[10], x[11]}

        will be sorted as

            [a, x[0], x[1], x[2], x[3], x[10], x[11], z]

        """
        a, b, some_name, z = (Parameter(name) for name in ["a", "b", "some_name", "z"])
        x = ParameterVector("x", 12)
        a_vector = ParameterVector("a_vector", 15)

        qc = QuantumCircuit(2)
        qc.p(z, 0)
        for i, x_i in enumerate(reversed(x)):
            qc.rx(x_i, i % 2)
        qc.cry(a, 0, 1)
        qc.crz(some_name, 1, 0)
        for v_i in a_vector[::2]:
            qc.p(v_i, 0)
        for v_i in a_vector[1::2]:
            qc.p(v_i, 1)
        qc.p(b, 0)

        expected_order = [a] + a_vector[:] + [b, some_name] + x[:] + [z]
        actual_order = qc.parameters

        self.assertListEqual(expected_order, list(actual_order))

    @data(True, False)
    def test_parameter_order_compose(self, front):
        """Test the parameter order is correctly maintained upon composing circuits."""
        x = Parameter("x")
        y = Parameter("y")
        qc1 = QuantumCircuit(1)
        qc1.p(x, 0)
        qc2 = QuantumCircuit(1)
        qc2.rz(y, 0)

        order = [x, y]

        composed = qc1.compose(qc2, front=front)

        self.assertListEqual(list(composed.parameters), order)

    def test_parameter_order_append(self):
        """Test the parameter order is correctly maintained upon appending circuits."""
        x = Parameter("x")
        y = Parameter("y")
        qc1 = QuantumCircuit(1)
        qc1.p(x, 0)
        qc2 = QuantumCircuit(1)
        qc2.rz(y, 0)

        qc1.append(qc2, [0])

        self.assertListEqual(list(qc1.parameters), [x, y])

    def test_parameter_order_composing_nested_circuit(self):
        """Test the parameter order after nesting circuits and instructions."""
        x = ParameterVector("x", 5)
        inner = QuantumCircuit(1)
        inner.rx(x[0], [0])

        mid = QuantumCircuit(2)
        mid.p(x[1], 1)
        mid.append(inner, [0])
        mid.p(x[2], 0)
        mid.append(inner, [0])

        outer = QuantumCircuit(2)
        outer.compose(mid, inplace=True)
        outer.ryy(x[3], 0, 1)
        outer.compose(inner, inplace=True)
        outer.rz(x[4], 0)

        order = [x[0], x[1], x[2], x[3], x[4]]

        self.assertListEqual(list(outer.parameters), order)

    def test_is_parameterized(self):
        """Test checking if a gate is parameterized (bound/unbound)"""
        from qiskit.circuit.library.standard_gates.h import HGate
        from qiskit.circuit.library.standard_gates.rx import RXGate

        theta = Parameter("θ")
        rxg = RXGate(theta)
        self.assertTrue(rxg.is_parameterized())
        theta_bound = theta.bind({theta: 3.14})
        rxg = RXGate(theta_bound)
        self.assertFalse(rxg.is_parameterized())
        h_gate = HGate()
        self.assertFalse(h_gate.is_parameterized())

    def test_fix_variable(self):
        """Test setting a variable to a constant value"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, 0, qr)

        bqc = qc.assign_parameters({theta: 0.5})
        self.assertEqual(float(bqc.data[0].operation.params[0]), 0.5)
        self.assertEqual(float(bqc.data[1].operation.params[1]), 0.5)
        bqc = qc.assign_parameters({theta: 0.6})
        self.assertEqual(float(bqc.data[0].operation.params[0]), 0.6)
        self.assertEqual(float(bqc.data[1].operation.params[1]), 0.6)

    def test_multiple_parameters(self):
        """Test setting multiple parameters"""
        theta = Parameter("θ")
        x = Parameter("x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)
        self.assertEqual(qc.parameters, {theta, x})

    def test_multiple_named_parameters(self):
        """Test setting multiple named/keyword argument based parameters"""
        theta = Parameter(name="θ")
        x = Parameter(name="x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)
        self.assertEqual(theta.name, "θ")
        self.assertEqual(qc.parameters, {theta, x})

    @named_data(
        ["int", 2, int],
        ["float", 2.5, float],
        ["float16", numpy.float16(2.5), float],
        ["float32", numpy.float32(2.5), float],
        ["float64", numpy.float64(2.5), float],
    )
    def test_circuit_assignment_to_numeric(self, value, type_):
        """Test binding a numeric value to a circuit instruction"""
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.append(Instruction("inst", 1, 0, [x]), (0,))
        qc.assign_parameters({x: value}, inplace=True)
        bound = qc.data[0].operation.params[0]
        self.assertIsInstance(bound, type_)
        self.assertEqual(bound, value)

    def test_partial_binding(self):
        """Test that binding a subset of circuit parameters returns a new parameterized circuit."""
        theta = Parameter("θ")
        x = Parameter("x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)

        pqc = qc.assign_parameters({theta: 2})

        self.assertEqual(pqc.parameters, {x})

        self.assertEqual(float(pqc.data[0].operation.params[0]), 2)
        self.assertEqual(float(pqc.data[1].operation.params[1]), 2)

    @data(True, False)
    def test_mixed_binding(self, inplace):
        """Test we can bind a mixed dict with Parameter objects and floats."""
        theta = Parameter("θ")
        x, new_x = Parameter("x"), Parameter("new_x")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        qc.u(0, theta, x, qr)

        pqc = qc.assign_parameters({theta: 2, x: new_x}, inplace=inplace)
        if inplace:
            self.assertEqual(qc.parameters, {new_x})
        else:
            self.assertEqual(pqc.parameters, {new_x})

    def test_expression_partial_binding(self):
        """Test that binding a subset of expression parameters returns a new
        parameterized circuit."""
        theta = Parameter("θ")
        phi = Parameter("phi")

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta + phi, qr)

        pqc = qc.assign_parameters({theta: 2})

        self.assertEqual(pqc.parameters, {phi})

        self.assertTrue(isinstance(pqc.data[0].operation.params[0], ParameterExpression))
        self.assertEqual(str(pqc.data[0].operation.params[0]), "2 + phi")

        fbqc = pqc.assign_parameters({phi: 1.0})

        self.assertEqual(fbqc.parameters, set())
        self.assertIsInstance(fbqc.data[0].operation.params[0], float)
        self.assertEqual(float(fbqc.data[0].operation.params[0]), 3)

    def test_two_parameter_expression_binding(self):
        """Verify that for a circuit with parameters theta and phi that
        we can correctly assign theta to -phi.
        """
        theta = Parameter("theta")
        phi = Parameter("phi")

        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        qc.ry(phi, 0)

        self.assertEqual(len(qc._data._raw_parameter_table_entry(theta)), 1)
        self.assertEqual(len(qc._data._raw_parameter_table_entry(phi)), 1)

        qc.assign_parameters({theta: -phi}, inplace=True)

        self.assertEqual(len(qc._data._raw_parameter_table_entry(phi)), 2)

    def test_expression_partial_binding_zero(self):
        """Verify that binding remains possible even if a previous partial bind
        would reduce the expression to zero.
        """
        theta = Parameter("theta")
        phi = Parameter("phi")

        qc = QuantumCircuit(1)
        qc.p(theta * phi, 0)

        pqc = qc.assign_parameters({theta: 0})

        self.assertEqual(pqc.parameters, {phi})

        self.assertTrue(isinstance(pqc.data[0].operation.params[0], ParameterExpression))
        self.assertEqual(str(pqc.data[0].operation.params[0]), "0")

        fbqc = pqc.assign_parameters({phi: 1})

        self.assertEqual(fbqc.parameters, set())
        self.assertEqual(float(fbqc.data[0].operation.params[0]), 0)

    def test_assignment_to_annotated_operation(self):
        """Test that assignments to an ``AnnotatedOperation`` are propagated all the way down."""

        class MyGate(Gate):
            """Arbitrary non-standard gate."""

            def __init__(self, param):
                super().__init__("my_gate", 1, [param])
                # Eagerly create our definition.
                _ = self.definition

            def _define(self):
                self._definition = QuantumCircuit(1, name="my_gate_inner")
                self._definition.ry(self.params[0], 0)

        theta = Parameter("theta")

        # Sanity check for the test; it won't catch errors if this fails.
        self.assertEqual(MyGate(theta), MyGate(theta))
        self.assertNotEqual(MyGate(theta), MyGate(1.23))

        parametric_gate = MyGate(theta)
        qc = QuantumCircuit(2)
        qc.append(parametric_gate.control(1, annotated=True), [0, 1], copy=True)
        assigned = qc.assign_parameters([1.23])

        expected = QuantumCircuit(2)
        expected.append(MyGate(1.23).control(1, annotated=True), [0, 1])
        self.assertEqual(assigned, expected)
        self.assertEqual(
            assigned.data[0].operation.base_op.definition,
            expected.data[0].operation.base_op.definition,
        )

        qc.assign_parameters([1.23], inplace=True)
        self.assertEqual(qc, expected)

        # Test that the underlying gate was not modified.
        self.assertEqual(parametric_gate.params, [theta])
        self.assertEqual(set(parametric_gate.definition.parameters), {theta})

    def test_raise_if_assigning_params_not_in_circuit(self):
        """Verify binding parameters which are not present in the circuit raises an error."""
        x = Parameter("x")
        y = Parameter("y")
        z = ParameterVector("z", 3)
        qr = QuantumRegister(1)

        qc = QuantumCircuit(qr)
        qc.p(0.1, qr[0])
        with self.assertRaises(CircuitError):
            qc.assign_parameters({x: 1})
        qc.p(x, qr[0])
        with self.assertRaises(CircuitError):
            qc.assign_parameters({x: 1, y: 2})
        qc.p(z[1], qr[0])
        with self.assertRaises(CircuitError):
            qc.assign_parameters({z: [3, 4, 5]})
        with self.assertRaises(CircuitError):
            qc.assign_parameters({6: 6})
        with self.assertRaises(CircuitError):
            qc.assign_parameters({None: 7})

    def test_gate_multiplicity_binding(self):
        """Test binding when circuit contains multiple references to same gate"""

        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        gate = RZGate(theta)
        qc.append(gate, [0], [])
        qc.append(gate, [0], [])
        qc2 = qc.assign_parameters({theta: 1.0})
        self.assertEqual(qc2._data.num_parameters(), 0)
        for instruction in qc2.data:
            self.assertEqual(float(instruction.operation.params[0]), 1.0)

    def test_circuit_generation(self):
        """Test creating a series of circuits parametrically"""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(theta, qr)
        backend = BasicSimulator()
        qc_aer = transpile(qc, backend)

        # generate list of circuits
        circs = []
        theta_list = numpy.linspace(0, numpy.pi, 20)
        for theta_i in theta_list:
            circs.append(qc_aer.assign_parameters({theta: theta_i}))

        for index, theta_i in enumerate(theta_list):
            res = float(circs[index].data[0].params[0])
            self.assertTrue(math.isclose(res, theta_i), f"{res} != {theta_i}")

    def test_circuit_composition(self):
        """Test preservation of parameters when combining circuits."""
        theta = Parameter("θ")
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc1 = QuantumCircuit(qr, cr)
        qc1.rx(theta, qr)

        phi = Parameter("phi")
        qc2 = QuantumCircuit(qr, cr)
        qc2.ry(phi, qr)
        qc2.h(qr)
        qc2.measure(qr, cr)

        qc3 = qc1.compose(qc2)
        self.assertEqual(qc3.parameters, {theta, phi})

    def test_composite_instruction(self):
        """Test preservation of parameters via parameterized instructions."""
        theta = Parameter("θ")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)
        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi / 2, qr1)
        qc1.ry(theta, qr1)
        gate = qc1.to_instruction()
        self.assertEqual(gate.params, [theta])

        phi = Parameter("phi")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(phi, qr2[0])
        qc2.h(qr2)
        qc2.append(gate, qargs=[qr2[1]])
        self.assertEqual(qc2.parameters, {theta, phi})

    def test_parameter_name_conflicts_raises(self):
        """Verify attempting to add different parameters with matching names raises an error."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        qc.p(theta1, 0)

        self.assertRaises(CircuitError, qc.p, theta2, 0)

    def test_bind_ryrz_vector(self):
        """Test binding a list of floats to a ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector("θ", length=len(qc.qubits) * depth * 2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i + 1])
            qc.barrier()
        theta_vals = numpy.linspace(0, 1, len(theta)) * numpy.pi
        self.assertEqual(set(qc.parameters), set(theta.params))
        bqc = qc.assign_parameters({theta: theta_vals})
        for instruction in bqc.data:
            if hasattr(instruction.operation, "params") and instruction.operation.params:
                self.assertIn(float(instruction.operation.params[0]), theta_vals)

    def test_compile_vector(self):
        """Test compiling a circuit with an unbound ParameterVector"""
        qc = QuantumCircuit(4)
        depth = 4
        theta = ParameterVector("θ", length=len(qc.qubits) * depth * 2)
        theta_iter = iter(theta)
        for _ in range(depth):
            for q in qc.qubits:
                qc.ry(next(theta_iter), q)
                qc.rz(next(theta_iter), q)
            for i, q in enumerate(qc.qubits[:-1]):
                qc.cx(qc.qubits[i], qc.qubits[i + 1])
            qc.barrier()
        backend = BasicSimulator()
        qc_aer = transpile(qc, backend)
        for param in theta:
            self.assertIn(param, qc_aer.parameters)

    def test_instruction_ryrz_vector(self):
        """Test constructing a circuit from instructions with remapped ParameterVectors"""
        qubits = 5
        depth = 4
        ryrz = QuantumCircuit(qubits, name="ryrz")
        theta = ParameterVector("θ0", length=len(ryrz.qubits) * 2)
        theta_iter = iter(theta)
        for q in ryrz.qubits:
            ryrz.ry(next(theta_iter), q)
            ryrz.rz(next(theta_iter), q)

        cxs = QuantumCircuit(qubits - 1, name="cxs")
        for i, _ in enumerate(cxs.qubits[:-1:2]):
            cxs.cx(cxs.qubits[2 * i], cxs.qubits[2 * i + 1])

        paramvecs = []
        qc = QuantumCircuit(qubits)
        for i in range(depth):
            theta_l = ParameterVector(f"θ{i + 1}", length=len(ryrz.qubits) * 2)
            ryrz_inst = ryrz.to_instruction(parameter_map={theta: theta_l})
            paramvecs += [theta_l]
            qc.append(ryrz_inst, qargs=qc.qubits)
            qc.append(cxs, qargs=qc.qubits[1:])
            qc.append(cxs, qargs=qc.qubits[:-1])
            qc.barrier()

        backend = BasicSimulator()
        qc_aer = transpile(qc, backend)
        for vec in paramvecs:
            for param in vec:
                self.assertIn(param, qc_aer.parameters)

    def test_parameter_equality_through_serialization(self):
        """Verify parameters maintain their equality after serialization."""

        x1 = Parameter("x")
        x2 = Parameter("x")

        x1_p = pickle.loads(pickle.dumps(x1))
        x2_p = pickle.loads(pickle.dumps(x2))

        self.assertEqual(x1, x1_p)
        self.assertEqual(x2, x2_p)

        self.assertNotEqual(x1, x2_p)
        self.assertNotEqual(x2, x1_p)

    def test_parameter_vector_equality_through_serialization(self):
        """Verify elements of parameter vectors maintain their equality after serialization."""

        x1 = ParameterVector("x", 2)
        x2 = ParameterVector("x", 2)

        x1_p = pickle.loads(pickle.dumps(x1))
        x2_p = pickle.loads(pickle.dumps(x2))

        self.assertEqual(x1[0], x1_p[0])
        self.assertEqual(x2[0], x2_p[0])

        self.assertNotEqual(x1[0], x2_p[0])
        self.assertNotEqual(x2[0], x1_p[0])

        self.assertEqual(x1_p[0].vector, x1_p)
        self.assertEqual(x2_p[0].vector, x2_p)
        self.assertEqual([p.index for p in x1_p], list(range(len(x1_p))))
        self.assertEqual([p.index for p in x2_p], list(range(len(x2_p))))

    @data("single", "vector")
    def test_parameter_equality_to_expression(self, ptype):
        """Verify that parameters compare equal to `ParameterExpression`s that represent the same
        thing."""

        if ptype == "single":
            x1 = Parameter("x")
            x2 = Parameter("x")
        else:
            x1 = ParameterVector("x", 2)[0]
            x2 = ParameterVector("x", 2)[0]

        x1_expr = x1 + 0
        # Smoke test: the test isn't valid if that above expression remains a `Parameter`; we need
        # it to have upcast to `ParameterExpression`.
        self.assertNotIsInstance(x1_expr, Parameter)
        x2_expr = x2 + 0
        self.assertNotIsInstance(x2_expr, Parameter)

        self.assertEqual(x1, x1_expr)
        self.assertEqual(x2, x2_expr)

        self.assertNotEqual(x1, x2_expr)
        self.assertNotEqual(x2, x1_expr)

        # Since these two pairs of objects compared equal, they must have the same hash as well.
        self.assertEqual(hash(x1), hash(x1_expr))
        self.assertEqual(hash(x2), hash(x2_expr))

    def test_parameter_vector_elements_can_be_recreated(self):
        left = ParameterVector("a", 5)
        right = ParameterVector("a", 5, uuid=left.uuid)
        # We can't compare the two vectors directly because `ParameterVector` only implements
        # identity-based equality, since it's internally mutable.
        self.assertEqual(list(left), list(right))

        # UUIDs don't match.
        other = ParameterVector("a", 5)
        self.assertNotEqual(list(left), list(other))

    def test_parameter_vector_backrefs_equal(self):
        vector = ParameterVector("a", 10)
        # With `ParameterVector` implemented in Rust and the actual Python object not cached,
        # there's no guarantee that `pv[0].vector is pv` (as the pure-Python implementation used to
        # preserve), but we need to maintain `pv[0].vector == pv`.
        self.assertEqual([x.vector for x in vector], [vector] * len(vector))

    def test_binding_parameterized_circuits_built_in_multiproc(self):
        """Verify subcircuits built in a subprocess can still be bound."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2429

        num_processes = 4

        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)

        circuit = QuantumCircuit(qr, cr)
        parameters = [Parameter(f"x{i}") for i in range(num_processes)]

        results = parallel_map(
            _construct_circuit, parameters, task_args=(qr,), num_processes=num_processes
        )

        for qc in results:
            circuit.compose(qc, inplace=True)

        parameter_values = dict.fromkeys(parameters, 1.0)

        bind_circuit = circuit.assign_parameters(parameter_values)

        self.assertEqual(len(bind_circuit.data), 4)
        self.assertTrue(
            all(
                len(inst.operation.params) == 1
                and isinstance(inst.operation.params[0], float)
                and float(inst.operation.params[0]) == 1
                for inst in bind_circuit.data
            )
        )

    def test_transpiling_multiple_parameterized_circuits(self):
        """Verify several parameterized circuits can be transpiled at once."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2864

        qr = QuantumRegister(1)
        qc1 = QuantumCircuit(qr)
        qc2 = QuantumCircuit(qr)

        theta = Parameter("theta")

        qc1.u(theta, 0, 0, qr[0])
        qc2.u(theta, 3.14, 0, qr[0])

        qc1 = qc1.assign_parameters({theta: 1})
        qc2 = qc2.assign_parameters({theta: 1})

        circuits = [qc1, qc2]

        backend = BasicSimulator()
        job = backend.run(transpile(circuits, backend), shots=512)

        self.assertTrue(len(job.result().results), 2)

    @data(0, 1, 2, 3)
    def test_transpile_across_optimization_levels(self, opt_level):
        """Verify parameterized circuits can be transpiled with all default pass managers."""

        qc = QuantumCircuit(5, 5)

        theta = Parameter("theta")
        phi = Parameter("phi")

        qc.rx(theta, 0)
        qc.x(0)
        for i in range(5 - 1):
            qc.rxx(phi, i, i + 1)

        qc.measure(range(5 - 1), range(5 - 1))

        transpile(
            qc,
            GenericBackendV2(
                num_qubits=5,
                coupling_map=BOGOTA_CMAP,
                seed=42,
            ),
            optimization_level=opt_level,
        )

    def test_repeated_gates_to_dag_and_back(self):
        """Verify circuits with repeated parameterized gates can be converted
        to DAG and back, maintaining consistency of circuit._parameter_table."""

        from qiskit.converters import circuit_to_dag, dag_to_circuit

        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta = Parameter("theta")

        qc.p(theta, qr[0])

        double_qc = qc.compose(qc)
        test_qc = dag_to_circuit(circuit_to_dag(double_qc))

        bound_test_qc = test_qc.assign_parameters({theta: 1})
        self.assertEqual(len(bound_test_qc.parameters), 0)

    def test_global_phase_to_dag_and_back(self):
        """Test global phase parameters are correctly handled to dag and back."""
        from qiskit.converters import circuit_to_dag, dag_to_circuit

        theta = Parameter("theta")
        qc = QuantumCircuit(global_phase=theta)

        test_qc = dag_to_circuit(circuit_to_dag(qc))

        bound_test_qc = test_qc.assign_parameters({theta: 1})
        self.assertEqual(bound_test_qc.global_phase, 1)

    def test_rebinding_instruction_copy(self):
        """Test rebinding a copied instruction does not modify the original."""

        theta = Parameter("th")

        qc = QuantumCircuit(1)
        qc.rx(theta, 0)
        instr = qc.to_instruction()

        qc1 = QuantumCircuit(1)
        qc1.append(instr, [0])

        output1 = qc1.assign_parameters({theta: 0.1}).decompose()
        output2 = qc1.assign_parameters({theta: 0.2}).decompose()

        expected1 = QuantumCircuit(1)
        expected1.rx(0.1, 0)

        expected2 = QuantumCircuit(1)
        expected2.rx(0.2, 0)

        self.assertEqual(expected1, output1)
        self.assertEqual(expected2, output2)

    def test_sign_of_parameter(self):
        """Test returning the sign of the value of the parameter"""

        b = Parameter("phi")
        sign_of_parameter = b.sign()
        self.assertEqual(sign_of_parameter.assign(b, -3), -1)
        self.assertEqual(sign_of_parameter.assign(b, 2), 1)
        self.assertEqual(sign_of_parameter.assign(b, 0), 0)

    @combine(target_type=["gate", "instruction"], parameter_type=["numbers", "parameters"])
    def test_decompose_propagates_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482
        theta = Parameter("th")
        qc = QuantumCircuit(1)
        qc.rx(theta, 0)

        if target_type == "gate":
            inst = qc.to_gate()
        else:  # target_type == "instruction":
            inst = qc.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if parameter_type == "numbers":
            bound_qc2 = qc2.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(0.5, 0)
        else:
            phi = Parameter("ph")
            bound_qc2 = qc2.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc2 = QuantumCircuit(1)
            expected_qc2.rx(phi, 0)

        decomposed_qc2 = bound_qc2.decompose()

        with self.subTest(msg="testing parameters of initial circuit"):
            self.assertEqual(qc2.parameters, {theta})

        with self.subTest(msg="testing parameters of bound circuit"):
            self.assertEqual(bound_qc2.parameters, expected_parameters)

        with self.subTest(msg="testing parameters of deep decomposed bound circuit"):
            self.assertEqual(decomposed_qc2.parameters, expected_parameters)

        with self.subTest(msg="testing deep decomposed circuit"):
            self.assertEqual(decomposed_qc2, expected_qc2)

    @combine(target_type=["gate", "instruction"], parameter_type=["numbers", "parameters"])
    def test_decompose_propagates_deeply_bound_parameters(self, target_type, parameter_type):
        """Verify bind-before-decompose preserves deeply bound values."""
        theta = Parameter("th")
        qc1 = QuantumCircuit(1)
        qc1.rx(theta, 0)

        if target_type == "gate":
            inst = qc1.to_gate()
        else:  # target_type == "instruction":
            inst = qc1.to_instruction()

        qc2 = QuantumCircuit(1)
        qc2.append(inst, [0])

        if target_type == "gate":
            inst = qc2.to_gate()
        elif target_type == "instruction":
            inst = qc2.to_instruction()

        qc3 = QuantumCircuit(1)
        qc3.append(inst, [0])

        if parameter_type == "numbers":
            bound_qc3 = qc3.assign_parameters({theta: 0.5})
            expected_parameters = set()
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(0.5, 0)
        else:
            phi = Parameter("ph")
            bound_qc3 = qc3.assign_parameters({theta: phi})
            expected_parameters = {phi}
            expected_qc3 = QuantumCircuit(1)
            expected_qc3.rx(phi, 0)

        deep_decomposed_qc3 = bound_qc3.decompose().decompose()

        with self.subTest(msg="testing parameters of initial circuit"):
            self.assertEqual(qc3.parameters, {theta})

        with self.subTest(msg="testing parameters of bound circuit"):
            self.assertEqual(bound_qc3.parameters, expected_parameters)

        with self.subTest(msg="testing parameters of deep decomposed bound circuit"):
            self.assertEqual(deep_decomposed_qc3.parameters, expected_parameters)

        with self.subTest(msg="testing deep decomposed circuit"):
            self.assertEqual(deep_decomposed_qc3, expected_qc3)

    @data("gate", "instruction")
    def test_executing_parameterized_instruction_bound_early(self, target_type):
        """Verify bind-before-execute preserves bound values."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2482

        theta = Parameter("theta")

        sub_qc = QuantumCircuit(2)
        sub_qc.h(0)
        sub_qc.cx(0, 1)
        sub_qc.rz(theta, [0, 1])
        sub_qc.cx(0, 1)
        sub_qc.h(0)

        if target_type == "gate":
            sub_inst = sub_qc.to_gate()
        else:  # target_type == "instruction":
            sub_inst = sub_qc.to_instruction()

        unbound_qc = QuantumCircuit(2, 1)
        unbound_qc.append(sub_inst, [0, 1], [])
        unbound_qc.measure(0, 0)

        bound_qc = unbound_qc.assign_parameters({theta: numpy.pi / 2})

        shots = 1024
        backend = BasicSimulator()
        job = backend.run(transpile(bound_qc, backend), shots=shots)
        self.assertDictAlmostEqual(job.result().get_counts(), {"1": shots}, 0.05 * shots)

    def test_num_parameters(self):
        """Test the num_parameters property."""
        with self.subTest(msg="standard case"):
            theta = Parameter("θ")
            x = Parameter("x")
            qc = QuantumCircuit(1)
            qc.rx(theta, 0)
            qc.u(0, theta, x, 0)
            self.assertEqual(qc.num_parameters, 2)

        with self.subTest(msg="parameter vector"):
            params = ParameterVector("x", length=3)
            qc = QuantumCircuit(4)
            qc.rx(params[0], 2)
            qc.ry(params[1], 1)
            qc.rz(params[2], 3)
            self.assertEqual(qc.num_parameters, 3)

        with self.subTest(msg="no params"):
            qc = QuantumCircuit(1)
            qc.x(0)
            self.assertEqual(qc.num_parameters, 0)

    def test_to_instruction_after_inverse(self):
        """Verify converting an inverse generates a valid ParameterTable"""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/4235
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        inv_instr = qc.inverse().to_instruction()
        self.assertIsInstance(inv_instr, Instruction)

    def test_repeated_circuit(self):
        """Test repeating a circuit maintains the parameters."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)
        rep = qc.repeat(3)

        self.assertEqual(rep.parameters, {theta})

    def test_copy_after_inverse(self):
        """Verify circuit.inverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        inverse = qc.inverse()
        self.assertIn(theta, inverse.parameters)
        raise_if_parameter_table_invalid(inverse)

    def test_copy_after_reverse(self):
        """Verify circuit.reverse generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        reverse = qc.reverse_ops()
        self.assertIn(theta, reverse.parameters)
        raise_if_parameter_table_invalid(reverse)

    def test_copy_after_dot_data_setter(self):
        """Verify setting circuit.data generates a valid ParameterTable."""
        qc = QuantumCircuit(1)
        theta = Parameter("theta")
        qc.rz(theta, 0)

        qc.data = []
        self.assertEqual(qc.parameters, set())
        raise_if_parameter_table_invalid(qc)

    def test_nonfinal_insert_maintains_valid_table(self):
        """Inserts other than appends should still maintain valid tracking, for as long as we
        continue to allow non-final inserts."""
        a, b, c = [Parameter(x) for x in "abc"]
        qc = QuantumCircuit(1)
        qc.global_phase = a / 2
        qc.rz(a, 0)
        qc.rz(b + c, 0)
        raise_if_parameter_table_invalid(qc)
        qc.data.insert(0, qc.data.pop())
        raise_if_parameter_table_invalid(qc)

    def test_circuit_with_ufunc(self):
        """Test construction of circuit and binding of parameters
        after we apply universal functions."""
        from math import pi

        phi = Parameter(name="phi")
        theta = Parameter(name="theta")

        qc = QuantumCircuit(2)
        qc.p(numpy.abs(-phi), 0)
        qc.p(numpy.cos(phi), 0)
        qc.p(numpy.sin(phi), 0)
        qc.p(numpy.tan(phi), 0)
        qc.rz(numpy.arccos(theta), 1)
        qc.rz(numpy.arctan(theta), 1)
        qc.rz(numpy.arcsin(theta), 1)

        qc.assign_parameters({phi: pi, theta: 1}, inplace=True)

        qc_ref = QuantumCircuit(2)
        qc_ref.p(pi, 0)
        qc_ref.p(-1, 0)
        qc_ref.p(0, 0)
        qc_ref.p(0, 0)
        qc_ref.rz(0, 1)
        qc_ref.rz(pi / 4, 1)
        qc_ref.rz(pi / 2, 1)

        self.assertEqual(qc, qc_ref)

    def test_compile_with_ufunc(self):
        """Test compiling of circuit with unbound parameters
        after we apply universal functions."""
        from math import pi

        theta = ParameterVector("theta", length=7)

        qc = QuantumCircuit(7)
        qc.rx(numpy.abs(theta[0]), 0)
        qc.rx(numpy.cos(theta[1]), 1)
        qc.rx(numpy.sin(theta[2]), 2)
        qc.rx(numpy.tan(theta[3]), 3)
        qc.rx(numpy.arccos(theta[4]), 4)
        qc.rx(numpy.arctan(theta[5]), 5)
        qc.rx(numpy.arcsin(theta[6]), 6)

        # transpile to different basis
        transpiled = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)

        for x in theta:
            self.assertIn(x, transpiled.parameters)

        bound = transpiled.assign_parameters({theta: [-1, pi, pi, pi, 1, 1, 1]})

        expected = QuantumCircuit(7)
        expected.rx(1.0, 0)
        expected.rx(-1.0, 1)
        expected.rx(0.0, 2)
        expected.rx(0.0, 3)
        expected.rx(0.0, 4)
        expected.rx(pi / 4, 5)
        expected.rx(pi / 2, 6)
        expected = transpile(expected, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)

        self.assertEqual(expected, bound)

    def test_parametervector_resize(self):
        """Test the resize method of the parameter vector."""

        vec = ParameterVector("x", 2)
        element = vec[1]  # store an entry for instancecheck later on

        with self.subTest("shorten"):
            vec.resize(1)
            self.assertEqual(len(vec), 1)
            self.assertListEqual([param.name for param in vec], _paramvec_names("x", 1))

        with self.subTest("enlargen"):
            vec.resize(3)
            self.assertEqual(len(vec), 3)
            # ensure we still have an element with the same uuid
            self.assertEqual(element, vec[1])
            self.assertListEqual([param.name for param in vec], _paramvec_names("x", 3))

    def test_parametervector_repr(self):
        """Test the __repr__ method of the parameter vector."""
        vec = ParameterVector("x", 2)
        self.assertEqual(repr(vec), "ParameterVector(name='x', length=2)")

    def test_parametervector_str(self):
        """Test the __str__ method of the parameter vector."""
        vec = ParameterVector("x", 2)
        self.assertEqual(str(vec), "x, ['x[0]', 'x[1]']")

    def test_parametervector_index(self):
        """Test the index method of the parameter vector."""
        vec = ParameterVector("x", 2)
        self.assertEqual(vec.index(vec[1]), 1)

    def test_raise_if_sub_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")

        with self.assertRaisesRegex(CircuitError, "not present"):
            x.subs({y: z})

    def test_sub_allow_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")

        subbed = x.subs({y: z}, allow_unknown_parameters=True)
        self.assertEqual(subbed, x)

    def test_decompose_with_global_phase(self):
        """Test decomposing a circuit which introduces a global phase is correctly bound.

        Regression test of #13534.
        """
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.rz(x, 0)

        bound = qc.decompose().assign_parameters([1])

        expect = QuantumCircuit(1)
        expect.rz(1, 0)

        self.assertEqual(expect.decompose(), bound)


def _construct_circuit(param, qr):
    qc = QuantumCircuit(qr)
    qc.ry(param, qr[0])
    return qc


def _paramvec_names(prefix, length):
    return [f"{prefix}[{i}]" for i in range(length)]


@ddt
class TestParameterExpressions(QiskitTestCase):
    """Test expressions of Parameters."""

    # supported operations dictionary operation : accuracy (0=exact match)
    supported_operations = {
        add: 0,
        sub: 0,
        mul: 0,
        truediv: 0,
        pow: 1e-12,
    }

    def test_compare_to_value_when_bound(self):
        """Verify expression can be compared to a fixed value
        when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(bound_expr, 2.3)
        self.assertEqual(hash(bound_expr), hash(2.3))

    def test_abs_function_when_bound(self):
        """Verify expression can be used with
        abs functions when bound."""

        x = Parameter("x")
        xb_1 = x.bind({x: 2.0})
        xb_2 = x.bind({x: 3.0 + 4.0j})

        self.assertEqual(abs(xb_1), 2.0)
        self.assertEqual(abs(-xb_1), 2.0)
        self.assertEqual(abs(xb_2), 5.0)

    def test_abs_function_when_not_bound(self):
        """Verify expression can be used with
        abs functions when not bound."""

        x = Parameter("x")
        y = Parameter("y")

        self.assertEqual(abs(x), abs(-x))
        self.assertEqual(abs(x) * abs(y), abs(x * y))
        self.assertEqual(abs(x) / abs(y), abs(x / y))

    def test_cast_to_complex_when_bound(self):
        """Verify that the cast to complex works for bound objects."""
        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 1.0, y: 1j})
        self.assertEqual(complex(bound_expr), 1 + 1j)

    def test_raise_if_cast_to_complex_when_not_fully_bound(self):
        """Verify raises if casting to complex and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 1j})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            complex(bound_expr)

    def test_cast_to_float_when_bound(self):
        """Verify expression can be cast to a float when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(float(bound_expr), 2.3)

    def test_cast_to_float_when_underlying_expression_bound(self):
        """Verify expression can be cast to a float when it still contains unbound parameters, but
        the underlying symbolic expression has a knowable value."""
        x = Parameter("x")
        expr = x - x + 2.3
        self.assertEqual(float(expr), 2.3)

    def test_cast_to_float_intermediate_complex_value(self):
        """Verify expression can be cast to a float when it is fully bound, but an intermediate part
        of the expression evaluation involved complex types.
        """
        x = Parameter("x")
        bound_expr = (x + 1.0 + 1.0j).bind({x: -1.0j})
        self.assertEqual(float(bound_expr), 1.0)

    def test_cast_to_float_of_complex_fails(self):
        """Test that an attempt to produce a float from a complex value fails if there is an
        imaginary part, with a sensible error message."""
        x = Parameter("x")
        bound_expr = (x + 1.0j).bind({x: 1.0})
        with self.assertRaisesRegex(
            TypeError, "Could not cast complex parameter expression to float"
        ):
            float(bound_expr)

    def test_raise_if_cast_to_float_when_not_fully_bound(self):
        """Verify raises if casting to float and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            float(bound_expr)

    def test_cast_to_int_when_bound(self):
        """Verify expression can be cast to an int when fully bound."""

        x = Parameter("x")
        bound_expr = x.bind({x: 2.3})
        self.assertEqual(int(bound_expr), 2)

    def test_cast_to_int_when_bound_truncates_after_evaluation(self):
        """Verify expression can be cast to an int when fully bound, but
        truncated only after evaluation."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3, y: 0.8})
        self.assertEqual(int(bound_expr), 3)

    def test_cast_to_int_when_underlying_expression_bound(self):
        """Verify expression can be cast to a int when it still contains unbound parameters, but the
        underlying symbolic expression has a knowable value."""
        x = Parameter("x")
        expr = x - x + 2.3
        self.assertEqual(int(expr), 2)

    def test_raise_if_cast_to_int_when_not_fully_bound(self):
        """Verify raises if casting to int and not fully bound."""

        x = Parameter("x")
        y = Parameter("y")
        bound_expr = (x + y).bind({x: 2.3})
        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            int(bound_expr)

    def test_raise_if_sub_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")
        expr = x + 2

        y = Parameter("y")
        z = Parameter("z")

        with self.assertRaisesRegex(CircuitError, "not present"):
            expr.subs({y: z})

    def test_sub_allow_unknown_parameters(self):
        """Verify we raise if asked to sub a parameter not in self."""
        x = Parameter("x")
        expr = x + 2

        y = Parameter("y")
        z = Parameter("z")

        subbed = expr.subs({y: z}, allow_unknown_parameters=True)
        self.assertEqual(subbed, expr)

    def test_raise_if_subbing_in_parameter_name_conflict(self):
        """Verify we raise if substituting in conflicting parameter names."""
        x = Parameter("x")
        y_first = Parameter("y")

        expr = x + y_first

        y_second = Parameter("y")

        # Replacing an existing name is okay.
        expr.subs({y_first: y_second})

        with self.assertRaisesRegex(CircuitError, "Name conflict"):
            expr.subs({x: y_second})

    @data(2, 1.3, 0, -1, -1.0, numpy.pi, 1j)
    def test_expressions_of_parameter_with_constant(self, const):
        """Verify operating on a Parameter with a constant."""

        x = Parameter("x")

        for op, rel_tol in self.supported_operations.items():
            expr = op(const, x)
            bound_expr = expr.bind({x: 2.3})

            res = complex(bound_expr)
            expected = op(const, 2.3)
            if rel_tol > 0:
                self.assertTrue(
                    cmath.isclose(res, expected, rel_tol=rel_tol), f"{res} != {expected}"
                )
            else:
                self.assertEqual(res, expected)

            # Division by zero will raise. Tested elsewhere.
            if const == 0 and op == truediv:
                continue

            # Repeat above, swapping position of Parameter and constant.
            expr = op(x, const)
            bound_expr = expr.bind({x: 2.3})

            res = complex(bound_expr)
            expected = op(2.3, const)
            self.assertTrue(cmath.isclose(res, expected), f"{res} != {expected}")

    def test_complex_parameter_bound_to_real(self):
        """Test a complex parameter expression can be real if bound correctly."""

        x, y = Parameter("x"), Parameter("y")

        with self.subTest("simple 1j * x"):
            qc = QuantumCircuit(1)
            qc.rx(1j * x, 0)
            bound = qc.assign_parameters({x: 1j})
            ref = QuantumCircuit(1)
            ref.rx(-1, 0)
            self.assertEqual(bound, ref)

        with self.subTest("more complex expression"):
            qc = QuantumCircuit(1)
            qc.rx(0.5j * x - y * y + 2 * y, 0)
            bound = qc.assign_parameters({x: -4, y: 1j})
            ref = QuantumCircuit(1)
            ref.rx(1, 0)
            self.assertEqual(bound, ref)

    def test_complex_angle_raises_when_not_supported(self):
        """Test parameters are validated when fully bound and errors are raised accordingly."""
        x = Parameter("x")
        qc = QuantumCircuit(1)
        qc.r(x, 1j * x, 0)

        with self.subTest("binding x to 0 yields real parameters"):
            bound = qc.assign_parameters({x: 0})
            ref = QuantumCircuit(1)
            ref.r(0, 0, 0)
            self.assertEqual(bound, ref)

        with self.subTest("binding x to 1 yields complex parameters"):
            # RGate does not support complex parameters
            with self.assertRaises(CircuitError):
                bound = qc.assign_parameters({x: 1})

    def test_operating_on_a_parameter_with_a_non_float_will_raise(self):
        """Verify operations between a Parameter and a non-float will raise."""

        bad_constants = ["1", numpy.inf, numpy.nan, None, {}, []]

        x = Parameter("x")

        for op in self.supported_operations:
            for const in bad_constants:
                with self.subTest(op=op, const=const):
                    with self.assertRaises(TypeError):
                        _ = op(const, x)

                    with self.assertRaises(TypeError):
                        _ = op(x, const)

    def test_expressions_division_by_zero(self):
        """Verify dividing a Parameter by 0, or binding 0 as a denominator raises."""

        x = Parameter("x")

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0

        with self.assertRaises(ZeroDivisionError):
            _ = x / 0.0

        expr = 2 / x

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0})

        with self.assertRaises(ZeroDivisionError):
            _ = expr.bind({x: 0.0})

    def test_expressions_of_parameter_with_parameter(self):
        """Verify operating on two Parameters."""

        x = Parameter("x")
        y = Parameter("y")

        for op in self.supported_operations:
            expr = op(x, y)

            partially_bound_expr = expr.bind({x: 2.3})

            self.assertEqual(partially_bound_expr.parameters, {y})

            fully_bound_expr = partially_bound_expr.bind({y: -numpy.pi})

            self.assertEqual(fully_bound_expr.parameters, set())
            self.assertEqual(float(fully_bound_expr), op(2.3, -numpy.pi))

            bound_expr = expr.bind({x: 2.3, y: -numpy.pi})

            self.assertEqual(bound_expr.parameters, set())
            self.assertEqual(float(bound_expr), op(2.3, -numpy.pi))

    def test_expressions_operation_order(self):
        """Verify ParameterExpressions respect order of operations."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        # Parenthesis before multiplication/division
        expr = (x + y) * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 9)

        expr = x * (y + z)
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

        # Multiplication/division before addition/subtraction
        expr = x + y * z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 7)

        expr = x * y + z
        bound_expr = expr.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr), 5)

    def test_nested_expressions(self):
        """Verify ParameterExpressions can also be the target of operations."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        expr1 = x * y
        expr2 = expr1 + z
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 5)

    def test_negated_expression(self):
        """Verify ParameterExpressions can be negated."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        expr1 = -x + y
        expr2 = -expr1 * (-z)
        bound_expr2 = expr2.bind({x: 1, y: 2, z: 3})

        self.assertEqual(float(bound_expr2), 3)

    def test_positive_expression(self):
        """This tests parameter unary plus."""
        x = Parameter("x")
        y = +x
        self.assertEqual(float(y.bind({x: 1})), 1.0)
        self.assertIsInstance(+x, type(-x))

    def test_standard_cu3(self):
        """This tests parameter negation in standard extension gate cu3."""
        from qiskit.circuit.library import CU3Gate

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        qc = qiskit.QuantumCircuit(2)
        qc.append(CU3Gate(x, y, z), [0, 1])
        try:
            qc.decompose()
        except TypeError:
            self.fail("failed to decompose cu3 gate with negated parameter expression")

    def test_name_collision(self):
        """Verify Expressions of distinct Parameters of shared name raises."""
        x = Parameter("p")
        y = Parameter("p")

        # Expression of the same Parameter are valid.
        _ = x + x
        _ = x - x
        _ = x * x
        _ = x / x

        with self.assertRaises(CircuitError):
            _ = x + y
        with self.assertRaises(CircuitError):
            _ = x - y
        with self.assertRaises(CircuitError):
            _ = x * y
        with self.assertRaises(CircuitError):
            _ = x / y

    @combine(target_type=["gate", "instruction"], order=["bind-decompose", "decompose-bind"])
    def test_to_instruction_with_expression(self, target_type, order):
        """Test preservation of expressions via parameterized instructions.

                  ┌───────┐┌──────────┐┌───────────┐
        qr1_0: |0>┤ Rx(θ) ├┤ Rz(pi/2) ├┤ Ry(phi*θ) ├
                  └───────┘└──────────┘└───────────┘

                     ┌───────────┐
        qr2_0: |0>───┤ Ry(delta) ├───
                  ┌──┴───────────┴──┐
        qr2_1: |0>┤ Circuit0(phi,θ) ├
                  └─────────────────┘
        qr2_2: |0>───────────────────
        """

        theta = Parameter("θ")
        phi = Parameter("phi")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi / 2, qr1)
        qc1.ry(theta * phi, qr1)

        if target_type == "gate":
            gate = qc1.to_gate()
        elif target_type == "instruction":
            gate = qc1.to_instruction()

        self.assertEqual(gate.params, [phi, theta])

        delta = Parameter("delta")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertEqual(qc2.parameters, {delta, theta, phi})

        binds = {delta: 1, theta: 2, phi: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(numpy.pi / 2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, numpy.pi / 2, 0)

        if order == "bind-decompose":
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == "decompose-bind":
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    @combine(target_type=["gate", "instruction"], order=["bind-decompose", "decompose-bind"])
    def test_to_instruction_expression_parameter_map(self, target_type, order):
        """Test preservation of expressions via instruction parameter_map."""

        theta = Parameter("θ")
        phi = Parameter("phi")
        qr1 = QuantumRegister(1, name="qr1")
        qc1 = QuantumCircuit(qr1)

        qc1.rx(theta, qr1)
        qc1.rz(numpy.pi / 2, qr1)
        qc1.ry(theta * phi, qr1)

        theta_p = Parameter("theta")
        phi_p = Parameter("phi")

        if target_type == "gate":
            gate = qc1.to_gate(parameter_map={theta: theta_p, phi: phi_p})
        elif target_type == "instruction":
            gate = qc1.to_instruction(parameter_map={theta: theta_p, phi: phi_p})

        self.assertListEqual(gate.params, [theta_p, phi_p])

        delta = Parameter("delta")
        qr2 = QuantumRegister(3, name="qr2")
        qc2 = QuantumCircuit(qr2)
        qc2.ry(delta, qr2[0])
        qc2.append(gate, qargs=[qr2[1]])

        self.assertListEqual(list(qc2.parameters), [delta, phi_p, theta_p])

        binds = {delta: 1, theta_p: 2, phi_p: 3}
        expected_qc = QuantumCircuit(qr2)
        expected_qc.rx(2, 1)
        expected_qc.rz(numpy.pi / 2, 1)
        expected_qc.ry(3 * 2, 1)
        expected_qc.r(1, numpy.pi / 2, 0)

        if order == "bind-decompose":
            decomp_bound_qc = qc2.assign_parameters(binds).decompose()
        elif order == "decompose-bind":
            decomp_bound_qc = qc2.decompose().assign_parameters(binds)

        self.assertEqual(decomp_bound_qc.parameters, set())
        self.assertEqual(decomp_bound_qc, expected_qc)

    def test_binding_across_broadcast_instruction(self):
        """Bind a parameter which was included via a broadcast instruction."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3008

        theta = Parameter("θ")
        n = 5

        qc = QuantumCircuit(n, 1)

        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(n))
        qc.barrier()

        for i in reversed(range(n - 1)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)

        theta_range = numpy.linspace(0, 2 * numpy.pi, 128)
        circuits = [qc.assign_parameters({theta: theta_val}) for theta_val in theta_range]

        self.assertEqual(len(circuits), len(theta_range))
        for theta_val, bound_circ in zip(theta_range, circuits):
            rz_gates = [
                inst.operation for inst in bound_circ.data if isinstance(inst.operation, RZGate)
            ]

            self.assertEqual(len(rz_gates), n)
            self.assertTrue(all(float(gate.params[0]) == theta_val for gate in rz_gates))

    def test_substituting_parameter_with_simple_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter("x")

        y = Parameter("y")
        sub_ = y / 2

        updated_expr = x.subs({x: sub_})

        expected = y / 2

        self.assertEqual(updated_expr, expected)

    def test_substituting_parameter_with_compound_expression(self):
        """Substitute a simple parameter expression for a parameter."""
        x = Parameter("x")

        y = Parameter("y")
        z = Parameter("z")
        sub_ = y * z

        updated_expr = x.subs({x: sub_})

        expected = y * z

        self.assertEqual(updated_expr, expected)

    def test_substituting_simple_with_simple_expression(self):
        """Substitute a simple parameter expression in a parameter expression."""
        x = Parameter("x")
        expr = x * x

        y = Parameter("y")
        sub_ = y / 2

        updated_expr = expr.subs({x: sub_})

        expected = y * y / 4

        self.assertEqual(updated_expr, expected)

    def test_substituting_compound_expression(self):
        """Substitute a compound parameter expression in a parameter expression."""
        x = Parameter("x")
        expr = x * x

        y = Parameter("y")
        z = Parameter("z")
        sub_ = y + z

        updated_expr = expr.subs({x: sub_})

        expected = (y + z) * (y + z)

        self.assertEqual(updated_expr, expected)

    def test_conjugate(self):
        """Test calling conjugate on a ParameterExpression."""
        x = Parameter("x")
        self.assertEqual((x.conjugate() + 1j), (x - 1j).conjugate())

    @data(
        circlib.RGate,
        circlib.RXGate,
        circlib.RYGate,
        circlib.RZGate,
        circlib.RXXGate,
        circlib.RYYGate,
        circlib.RZXGate,
        circlib.RZZGate,
        circlib.CRXGate,
        circlib.CRYGate,
        circlib.CRZGate,
        circlib.XXPlusYYGate,
    )
    def test_bound_gate_to_matrix(self, gate_class):
        """Test to_matrix works if previously free parameters are bound.

        The conversion might fail, if trigonometric functions such as cos are called on the
        parameters and the parameters are still of type ParameterExpression.
        """
        num_parameters = 2 if gate_class == circlib.RGate else 1
        params = list(range(1, 1 + num_parameters))
        free_params = ParameterVector("th", num_parameters)
        gate = gate_class(*params)
        num_qubits = gate.num_qubits

        circuit = QuantumCircuit(num_qubits)
        circuit.append(gate_class(*free_params), list(range(num_qubits)))
        bound_circuit = circuit.assign_parameters({free_params: params})

        numpy.testing.assert_array_almost_equal(Operator(bound_circuit).data, gate.to_matrix())

    def test_parameter_expression_grad(self):
        """Verify correctness of ParameterExpression gradients."""

        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")

        with self.subTest(msg="first order gradient"):
            expr = (x + y) * z
            self.assertEqual(expr.gradient(x), z)
            self.assertEqual(expr.gradient(z), (x + y))

        with self.subTest(msg="second order gradient"):
            expr = x * x
            self.assertEqual(expr.gradient(x), 2 * x)
            self.assertEqual(expr.gradient(x).gradient(x), 2)

    def test_parameter_expression_exp_log_vs_pow(self):
        """Test exp, log, pow for ParameterExpressions by asserting x**y = exp(y log(x))."""

        x = Parameter("x")
        y = Parameter("y")
        pow1 = x**y
        pow2 = (y * x.log()).exp()
        for x_val in [2, 1.3, numpy.pi]:
            for y_val in [2, 1.3, 0, -1, -1.0, numpy.pi, 1j]:
                with self.subTest(msg="with x={x_val}, y={y_val}"):
                    vals = {x: x_val, y: y_val}
                    pow1_val = pow1.bind(vals)
                    pow2_val = pow2.bind(vals)
                    self.assertTrue(cmath.isclose(pow1_val, pow2_val), f"{pow1_val} != {pow2_val}")

    def test_bound_expression_is_real(self):
        """Test is_real on bound parameters."""
        x = Parameter("x")
        self.assertEqual(x.is_real(), None)
        self.assertEqual((1j * x).is_real(), None)

        expr = 1j * x
        bound = expr.bind({x: 2})
        self.assertEqual(bound.is_real(), False)

        bound = x.bind({x: 0 + 0j})
        self.assertEqual(bound.is_real(), True)

        bound = x.bind({x: 0 + 1j})
        self.assertEqual(bound.is_real(), False)

        bound = x.bind({x: 1 + 0j})
        self.assertEqual(bound.is_real(), True)

        bound = x.bind({x: 1 + 1j})
        self.assertEqual(bound.is_real(), False)

    def test_numeric(self):
        """Tests of the 'numeric' method."""
        a, b = Parameter("a"), Parameter("b")
        one_int = (1 + a).assign(a, 0)
        self.assertIsInstance(one_int.numeric(), int)
        self.assertEqual(one_int.numeric(), 1)
        one_float = (1.0 + a).assign(a, 0.0)
        self.assertIsInstance(one_float.numeric(), float)
        self.assertEqual(one_float.numeric(), 1.0)
        one_imaginary = (1j + a).assign(a, 0.0)
        self.assertIsInstance(one_imaginary.numeric(), complex)
        self.assertEqual(one_imaginary.numeric(), 1j)

        # This is one particular case where symbolic libraries struggled (e.g. symengine 0.9.2) when
        # evaluating in the complex domain, but got the right answer if forced to the real domain.
        problem = (0.5 * a * b).assign(b, 0.5).assign(a, 0.5)
        self.assertIsInstance(problem.numeric(), float)
        self.assertEqual(problem.numeric(), 0.125)

        with self.assertRaisesRegex(TypeError, "unbound parameters"):
            (a + b).numeric()

    def test_repeat_with_parameterized_gates(self):
        """Tests that repeating a circuit with parameterized gates
        handles parameters correctly.

        See https://github.com/Qiskit/qiskit/issues/15645.
        """

        t_param = Parameter("t")

        # Create a quantum circuit with 2 parameterized gates.
        circuit = QuantumCircuit(5)
        circuit.append(PauliEvolutionGate(Pauli("XZXZX"), time=t_param), [0, 1, 2, 3, 4])
        circuit.append(PauliEvolutionGate(Pauli("IIIXX"), time=t_param / 2), [0, 1, 2, 3, 4])

        # Repeat the circuit 10 times.
        repeated_circuit = circuit.repeat(10)

        # Divide each parameter by num_steps.
        num_steps = 5
        repeated_circuit.assign_parameters({t_param: t_param / num_steps}, inplace=True)

        # Decompose the repeated circuit. The decomposed circuit has the two operations
        # from the circuit repeated 10 times.
        decomposed_circuit = repeated_circuit.decompose()

        for idx in [0, 1]:
            expected_param = circuit[idx].operation.params[0] / num_steps
            actual_param = decomposed_circuit[idx].operation.params[0]
            self.assertEqual(expected_param, actual_param)


class TestParameterEquality(QiskitTestCase):
    """Test equality of Parameters and ParameterExpressions."""

    def test_parameter_equal_self(self):
        """Verify a parameter is equal to it self."""
        theta = Parameter("theta")
        self.assertEqual(theta, theta)

    def test_parameter_not_equal_param_of_same_name(self):
        """Verify a parameter is not equal to a Parameter of the same name."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")
        self.assertNotEqual(theta1, theta2)

    def test_parameter_expression_equal_to_self(self):
        """Verify an expression is equal to itself."""
        theta = Parameter("theta")
        expr = 2 * theta

        self.assertEqual(expr, expr)

    def test_parameter_expression_equal_to_identical(self):
        """Verify an expression is equal an identical expression."""
        theta = Parameter("theta")
        expr1 = 2 * theta
        expr2 = 2 * theta

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_equal_floats_to_ints(self):
        """Verify an expression with float and int is identical."""
        theta = Parameter("theta")
        expr1 = 2.0 * theta
        expr2 = 2 * theta

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_equal_floats_to_divide_by_int(self):
        """Verify an expression with float and division by int is identical."""
        theta = Parameter("theta")
        expr1 = 0.25 * theta
        expr2 = theta / 4

        self.assertEqual(expr1, expr2)

    def test_parameter_expression_not_equal_if_params_differ(self):
        """Verify expressions not equal if parameters are different."""
        theta1 = Parameter("theta")
        theta2 = Parameter("theta")
        expr1 = 2 * theta1
        expr2 = 2 * theta2

        self.assertNotEqual(expr1, expr2)

    def test_parameter_equal_to_identical_expression(self):
        """Verify parameters and ParameterExpressions can be equal if identical."""
        theta = Parameter("theta")
        phi = Parameter("phi")

        expr = (theta + phi).bind({phi: 0})

        self.assertEqual(expr, theta)
        self.assertEqual(theta, expr)


class TestParameterView(QiskitTestCase):
    """Test the ParameterView object."""

    def setUp(self):
        super().setUp()
        x, y, z = Parameter("x"), Parameter("y"), Parameter("z")
        self.params = [x, y, z]
        self.view1 = ParameterView([x, y])
        self.view2 = ParameterView([y, z])
        self.view3 = ParameterView([x])

    def test_and(self):
        """Test __and__."""
        self.assertEqual(self.view1 & self.view2, {self.params[1]})

    def test_or(self):
        """Test __or__."""
        self.assertEqual(self.view1 | self.view2, set(self.params))

    def test_xor(self):
        """Test __xor__."""
        self.assertEqual(self.view1 ^ self.view2, {self.params[0], self.params[2]})

    def test_len(self):
        """Test __len__."""
        self.assertEqual(len(self.view1), 2)

    def test_le(self):
        """Test __le__."""
        self.assertTrue(self.view1 <= self.view1)
        self.assertFalse(self.view1 <= self.view3)

    def test_lt(self):
        """Test __lt__."""
        self.assertTrue(self.view3 < self.view1)

    def test_ge(self):
        """Test __ge__."""
        self.assertTrue(self.view1 >= self.view1)
        self.assertFalse(self.view3 >= self.view1)

    def test_gt(self):
        """Test __lt__."""
        self.assertTrue(self.view1 > self.view3)

    def test_eq(self):
        """Test __eq__."""
        self.assertTrue(self.view1 == self.view1)
        self.assertFalse(self.view3 == self.view1)

    def test_ne(self):
        """Test __eq__."""
        self.assertTrue(self.view1 != self.view2)
        self.assertFalse(self.view3 != self.view3)


if __name__ == "__main__":
    unittest.main()
