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

"""Test SamplerPub class"""

import ddt
import numpy as np

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.primitives.containers.bindings_array import BindingsArray
from test import QiskitTestCase


@ddt.ddt
class SamplerPubTestCase(QiskitTestCase):
    """Test the SamplerPub class."""

    def test_properties(self):
        """Test SamplerPub properties."""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        parameter_values = BindingsArray(data={params: np.ones((10, 2))})
        shots = 1000

        pub = SamplerPub(
            circuit=circuit,
            parameter_values=parameter_values,
            shots=shots,
        )
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(
            pub.parameter_values,
            parameter_values,
            msg="incorrect value for `parameter_values` property",
        )
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")

    def test_invalidate_circuit(self):
        """Test validation of circuit argument"""
        # Invalid circuit, it is an instruction
        circuit = QuantumCircuit(3).to_instruction()
        with self.assertRaises(TypeError):
            SamplerPub(circuit)

    @ddt.data(100.0, True, False, 100j, 1e5)
    def test_invalidate_shots_type(self, shots):
        """Test validation of shots argument type"""
        with self.assertRaises(TypeError, msg=f"shots type {type(shots)} should raise TypeError"):
            SamplerPub(QuantumCircuit(), shots=shots)

    @ddt.data(-1, 0)
    def test_invalidate_shots_value(self, shots):
        """Test invalid shots argument value"""
        with self.assertRaises(ValueError, msg="non-positive shots should raise ValueError"):
            SamplerPub(QuantumCircuit(), shots=shots)

    @ddt.idata(range(5))
    def test_validate_no_parameters(self, num_params):
        """Test unparameterized circuit raises for parameter values"""
        circuit = QuantumCircuit(2)
        parameter_values = BindingsArray(
            {(f"a{idx}" for idx in range(num_params)): np.zeros((2, num_params))}, shape=2
        )
        if num_params == 0:
            SamplerPub(circuit, parameter_values=parameter_values)
            return

        with self.assertRaises(ValueError):
            SamplerPub(circuit, parameter_values=parameter_values)

    @ddt.idata(range(5))
    def test_validate_num_parameters(self, num_params):
        """Test unparameterized circuit raises for parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        parameter_values = BindingsArray(
            {(f"a{idx}" for idx in range(num_params)): np.zeros((2, num_params))}, shape=2
        )
        if num_params == len(params):
            SamplerPub(circuit, parameter_values=parameter_values)
            return

        with self.assertRaises(ValueError):
            SamplerPub(circuit, parameter_values=parameter_values)

    @ddt.data((), (3,), (2, 3))
    def test_shaped_zero_parameter_values(self, shape):
        """Test Passing in a shaped array with no parameters works"""
        circuit = QuantumCircuit(2)
        parameter_values = BindingsArray({(): np.zeros((*shape, 0))}, shape=shape)
        pub = SamplerPub(circuit, parameter_values=parameter_values)
        self.assertEqual(pub.shape, shape)

    def test_coerce_circuit(self):
        """Test coercing an unparameterized circuit"""
        circuit = QuantumCircuit(10)
        circuit.measure_all()

        pub = SamplerPub.coerce(circuit)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, None, msg="incorrect value for `shots` property")
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
        circuit.measure_all()
        with self.assertRaises(ValueError):
            SamplerPub.coerce(circuit)

    @ddt.data(1, 10, 100, 1000)
    def test_coerce_pub_with_shots(self, shots):
        """Test coercing a SamplerPub"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        pub1 = SamplerPub(
            circuit=circuit,
            parameter_values=BindingsArray(data={params: np.ones((10, 2))}),
            shots=1000,
        )
        pub2 = SamplerPub.coerce(pub1, shots=shots)
        self.assertEqual(pub1, pub2)

    @ddt.data(1, 10, 100, 1000)
    def test_coerce_pub_without_shots(self, shots):
        """Test coercing a SamplerPub"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        pub1 = SamplerPub(
            circuit=circuit,
            parameter_values=BindingsArray(data={params: np.ones((10, 2))}),
            shots=None,
        )
        pub2 = SamplerPub.coerce(pub1, shots=shots)
        self.assertEqual(pub1.circuit, pub2.circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(
            pub1.parameter_values,
            pub2.parameter_values,
            msg="incorrect value for `parameter_values` property",
        )
        self.assertEqual(pub2.shots, shots, msg="incorrect value for `shots` property")

    @ddt.data(None, 1, 100)
    def test_coerce_tuple_1(self, shots):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        circuit.measure_all()
        pub = SamplerPub.coerce((circuit,), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")
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
    def test_coerce_tuple_2(self, shots):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        parameter_values = np.zeros((4, 3, 2))
        pub = SamplerPub.coerce((circuit, parameter_values), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")
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
    def test_coerce_tuple_2_trivial_params(self, shots):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        circuit.measure_all()
        pub = SamplerPub.coerce((circuit, None), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")
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
    def test_coerce_tuple_3(self, shots):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        parameter_values = np.zeros((4, 3, 2))
        pub = SamplerPub.coerce((circuit, parameter_values, 1000), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, 1000, msg="incorrect value for `shots` property")
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
    def test_coerce_tuple_3_trivial_shots(self, shots):
        """Test coercing circuit and parameter values"""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)
        circuit.measure_all()
        parameter_values = np.zeros((4, 3, 2))
        pub = SamplerPub.coerce((circuit, parameter_values, None), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")
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
    def test_coerce_tuple_3_trivial_params_shots(self, shots):
        """Test coercing circuit and parameter values"""
        circuit = QuantumCircuit(2)
        circuit.measure_all()
        pub = SamplerPub.coerce((circuit, None, None), shots=shots)
        self.assertEqual(pub.circuit, circuit, msg="incorrect value for `circuit` property")
        self.assertEqual(pub.shots, shots, msg="incorrect value for `shots` property")
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

    def test_coerce_pub_with_exact_types(self):
        """Test coercing a SamplerPub with exact types."""
        params = (Parameter("a"), Parameter("b"))
        circuit = QuantumCircuit(2)
        circuit.rx(params[0], 0)
        circuit.ry(params[1], 1)

        params = BindingsArray(data={params: np.ones((10, 2))})
        pub = SamplerPub.coerce((circuit, params))
        self.assertIs(pub.circuit, circuit)
        self.assertIs(pub.parameter_values, params)
