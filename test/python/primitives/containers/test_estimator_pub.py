# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test EstimatorPub class"""

import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.observables_array import ObservablesArray
from qiskit.primitives.containers.bindings_array import BindingsArray
from test import QiskitTestCase


@ddt.ddt
class EstimatorPubTestCase(QiskitTestCase):
    """Test the EstimatorPub class."""

    def test_properties(self):
        """Test EstimatorPub properties."""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        parameter_values = BindingsArray(data={params: np.ones((10, 2))})
        observables = ObservablesArray([{"XX": 0.1}])
        precision = 0.05

        pub = EstimatorPub(
            circuit=circuit,
            observables=observables,
            parameter_values=parameter_values,
            precision=precision,
        )
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(
            pub.observables,
            observables,
            msg="incorrect value for `observables` property",
        )
        self.assertEqual(
            pub.parameter_values,
            parameter_values,
            msg="incorrect value for `parameter_values` property",
        )
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")

    def test_invalidate_circuit(self):
        """Test validation of circuit argument"""
        # Invalid circuit, it is an instruction
        circuit = QuantumCircuit(3).to_instruction()
        obs = ObservablesArray([{"XYZ": 1}])
        with self.assertRaisesRegex(TypeError, "must be QuantumCircuit"):
            EstimatorPub(circuit, obs)

    @ddt.data("a", (1.0,))
    def test_invalidate_precision_type(self, precision):
        """Test validation of precision argument type"""
        obs = ObservablesArray([{"XYZ": 1}])
        with self.assertRaisesRegex(TypeError, "must be a real number"):
            EstimatorPub(QuantumCircuit(3), obs, precision=precision)

    def test_invalidate_precision_value(self):
        """Test invalid precision argument value"""
        obs = ObservablesArray([{"XYZ": 1}])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            EstimatorPub(QuantumCircuit(3), obs, precision=-1)

    @ddt.idata(range(5))
    def test_validate_no_parameters(self, num_params):
        """Test unparameterized circuit raises for parameter values"""
        circuit = QuantumCircuit(2)
        obs = ObservablesArray([{"XY": 1}])
        parameter_values = BindingsArray(
            {(f"a{idx}" for idx in range(num_params)): np.zeros((2, num_params))}, shape=2
        )
        if num_params == 0:
            EstimatorPub(circuit, obs, parameter_values=parameter_values)
            return

        with self.assertRaisesRegex(ValueError, rf"number.+\({num_params}\).+not match.+\(0\)"):
            EstimatorPub(circuit, obs, parameter_values=parameter_values)

    def test_validate_num_qubits(self):
        """Test unparameterized circuit raises for parameter values"""
        circuit = QuantumCircuit(2)
        EstimatorPub(circuit, ObservablesArray([{"XY": 1}]))

        with self.assertRaisesRegex(ValueError, r"qubits .+ \(2\) does not match .+ \(3\)"):
            EstimatorPub(circuit, ObservablesArray([{"XYZ": 1}]))

    @ddt.idata(range(5))
    def test_validate_num_parameters(self, num_params):
        """Test unparameterized circuit raises for parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)

        obs = ObservablesArray([{"XY": 1}])
        parameter_values = BindingsArray(
            {(f"a{idx}" for idx in range(num_params)): np.zeros((2, num_params))}, shape=2
        )

        if num_params == len(params):
            EstimatorPub(circuit, obs, parameter_values=parameter_values)
            return

        with self.assertRaisesRegex(ValueError, "does not match"):
            EstimatorPub(circuit, obs, parameter_values=parameter_values)

    @ddt.data((), (3,), (2, 3))
    def test_shaped_zero_parameter_values(self, shape):
        """Test Passing in a shaped array with no parameters works"""
        circuit = QuantumCircuit(2)
        obs = ObservablesArray({"XZ": 1})
        parameter_values = BindingsArray({(): np.zeros((*shape, 0))}, shape=shape)
        pub = EstimatorPub(circuit, obs, parameter_values=parameter_values)
        self.assertEqual(pub.shape, shape)

    def test_coerce_circuit(self):
        """Test coercing an unparameterized circuit"""
        circuit = QuantumCircuit(10)

        obs = ObservablesArray({"XYZXYZXYZX": 1})

        pub = EstimatorPub.coerce((circuit, obs))
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.observables, obs, msg="incorrect value for `observables` property")
        self.assertEqual(pub.precision, None, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape, (), msg="incorrect shape for `parameter_values` property"
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            0,
            msg="incorrect num parameters for `parameter_values` property",
        )

    def test_invalid_coerce_circuit(self):
        """Test coercing parameterized circuit raises"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(10)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)

        obs = ObservablesArray({"XYZXYZXYZX": 1})

        with self.assertRaises(ValueError):
            EstimatorPub.coerce((circuit, obs))

    @ddt.data(0.01, 0.02)
    def test_coerce_pub_with_precision(self, precision):
        """Test coercing an EstimatorPub"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        pub1 = EstimatorPub(
            circuit,
            obs,
            parameter_values=BindingsArray(data={params: np.ones((10, 2))}),
            precision=0.01,
        )
        pub2 = EstimatorPub.coerce(pub1, precision=precision)
        self.assertEqual(pub1, pub2)

    def test_coerce_pub_with_exact_types(self):
        """Test coercing an EstimatorPub"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        params = BindingsArray(data={params: np.ones((10, 2))})
        pub = EstimatorPub.coerce((circuit, obs, params))
        self.assertIs(pub.circuit, circuit)
        self.assertIs(pub.observables, obs)
        self.assertIs(pub.parameter_values, params)

    @ddt.data(0.01, 0.02)
    def test_coerce_pub_without_shots(self, precision):
        """Test coercing an EstimatorPub"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        pub1 = EstimatorPub(
            circuit,
            obs,
            parameter_values=BindingsArray(data={params: np.ones((10, 2))}),
            precision=None,
        )
        pub2 = EstimatorPub.coerce(pub1, precision=precision)
        self.assertEqual(pub1.circuit, pub2.circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub1.observables, pub2.observables)
        self.assertEqual(
            pub1.parameter_values,
            pub2.parameter_values,
            msg="incorrect value for `parameter_values` property",
        )
        self.assertEqual(pub2.precision, precision, msg="incorrect value for `precision` property")

    @ddt.data(None, 0.08)
    def test_coerce_tuple_1(self, precision):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        obs = ObservablesArray({"XY": 1})
        pub = EstimatorPub.coerce((circuit, obs), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.observables, obs, msg="incorrect value for `observables` property")
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape, (), msg="incorrect shape for `parameter_values` property"
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            0,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(None, 1, 100)
    def test_coerce_tuple_2(self, precision):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        parameter_values = np.zeros((4, 3, 2))
        pub = EstimatorPub.coerce((circuit, obs, parameter_values), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.observables, obs, msg="incorrect value for `observables` property")
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape,
            (4, 3),
            msg="incorrect shape for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            2,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(None, 1, 100)
    def test_coerce_tuple_2_trivial_params(self, precision):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        obs = ObservablesArray({"ZZ": 1})
        pub = EstimatorPub.coerce((circuit, obs, None), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.observables, obs, msg="incorrect value for `observables` property")
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape, (), msg="incorrect shape for `parameter_values` property"
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            0,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(None, 0.08)
    def test_coerce_tuple_3(self, precision):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        parameter_values = np.zeros((4, 3, 2))
        pub = EstimatorPub.coerce((circuit, obs, parameter_values, 0.08), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.precision, 0.08, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape,
            (4, 3),
            msg="incorrect shape for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            2,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(None, 0.07)
    def test_coerce_tuple_3_trivial_shots(self, precision):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        obs = ObservablesArray({"XY": 1})
        parameter_values = np.zeros((4, 3, 2))
        pub = EstimatorPub.coerce((circuit, obs, parameter_values, None), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape,
            (4, 3),
            msg="incorrect shape for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            2,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(None, 1, 100)
    def test_coerce_tuple_3_trivial_params_shots(self, precision):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        obs = ObservablesArray({"XY": 1})
        pub = EstimatorPub.coerce((circuit, obs, None, None), precision=precision)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.precision, precision, msg="incorrect value for `precision` property")
        # Check bindings array, this is more cumbersome since the class doesn't have an eq method
        self.assertIsInstance(
            pub.parameter_values,
            BindingsArray,
            msg="incorrect type for `parameter_values` property",
        )
        self.assertEqual(
            pub.parameter_values.shape, (), msg="incorrect shape for `parameter_values` property"
        )
        self.assertEqual(
            pub.parameter_values.num_parameters,
            0,
            msg="incorrect num parameters for `parameter_values` property",
        )

    @ddt.data(
        [(), (), ()],
        [(5,), (5,), (5,)],
        [(1,), (5,), (5,)],
        [(5,), (1,), (5,)],
        [(), (5,), (5,)],
        [(5,), (), (5,)],
        [(3, 4, 5), (3, 4, 5), (3, 4, 5)],
        [(2, 1, 10), (4, 1), (2, 4, 10)],
    )
    @ddt.unpack
    def test_broadcasting(self, obs_shape, params_shape, pub_shape):
        """Test that we end up with the correct broadcasted shape."""
        # sanity check that we agree with the NumPy convention
        self.assertEqual(np.broadcast_shapes(obs_shape, params_shape), pub_shape)

        params = list(map(Parameter, "abcdef"))
        circuit = QuantumCircuit(2)
        for idx in range(3):
            circuit.rz(params[2 * idx], 0)
            circuit.rz(params[2 * idx + 1], 1)

        obs = ObservablesArray([{"XX": 1}] * np.prod(obs_shape, dtype=int)).reshape(obs_shape)
        params = BindingsArray({tuple(params): np.empty(params_shape + (6,))})

        pub = EstimatorPub(circuit, obs, params)
        self.assertEqual(obs.shape, obs_shape)
        self.assertEqual(params.shape, params_shape)
        self.assertEqual(pub.shape, pub_shape)

    @ddt.data(
        [(5,), (6,)],
        [(3,), (5,)],
        [(3, 8, 5), (3, 4, 5)],
        [(1, 1, 10), (4, 11)],
    )
    @ddt.unpack
    def test_broadcasting_fails(self, obs_shape, params_shape):
        """Test that we get the right error if the entries are not broadcastable."""
        # sanity check that we agree with the NumPy convention
        with self.assertRaises(ValueError):
            np.broadcast_shapes(obs_shape, params_shape)

        params = list(map(Parameter, "abcdef"))
        circuit = QuantumCircuit(2)
        for idx in range(3):
            circuit.rz(params[2 * idx], 0)
            circuit.rz(params[2 * idx + 1], 1)

        obs = ObservablesArray([{"XX": 1}] * np.prod(obs_shape, dtype=int)).reshape(obs_shape)
        params = BindingsArray({tuple(params): np.empty(params_shape + (6,))})
        self.assertEqual(obs.shape, obs_shape)
        self.assertEqual(params.shape, params_shape)

        msg = rf"observables shape \({obs_shape}\) .+ values shape \({params_shape}\) are not"
        with self.assertRaisesRegex(ValueError, msg):
            EstimatorPub(circuit, obs, params)
