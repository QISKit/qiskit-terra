# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for validation functions used in BaseSamplerV1 and BaseEstimatorV1."""

from ddt import data, ddt, unpack
from numpy import array, float32, float64, int32, int64

from qiskit.circuit.random import random_circuit
from qiskit.primitives.base import validation_v1
from test import QiskitTestCase


@ddt
class TestCircuitValidation(QiskitTestCase):
    """Test circuits validation logic."""

    @data(
        (random_circuit(2, 2, seed=0), (random_circuit(2, 2, seed=0),)),
        (
            [random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)],
            (random_circuit(2, 2, seed=0), random_circuit(2, 2, seed=1)),
        ),
    )
    @unpack
    def test_validate_circuits(self, circuits, expected):
        """Test circuits standardization."""
        self.assertEqual(validation_v1._validate_circuits(circuits), expected)

    @data(None, "ERROR", True, 0, 1.0, 1j, [0.0])
    def test_type_error(self, circuits):
        """Test type error if invalid input."""
        with self.assertRaises(TypeError):
            validation_v1._validate_circuits(circuits)

    @data((), [], "")
    def test_value_error(self, circuits):
        """Test value error if no circuits are provided."""
        with self.assertRaises(ValueError):
            validation_v1._validate_circuits(circuits)


@ddt
class TestParameterValuesValidation(QiskitTestCase):
    """Test parameter_values validation logic."""

    @data(
        ((), ((),)),
        ([], ((),)),
        (0, ((0,),)),
        (1.2, ((1.2,),)),
        ((0,), ((0,),)),
        ([0], ((0,),)),
        ([1.2], ((1.2,),)),
        ((0, 1), ((0, 1),)),
        ([0, 1], ((0, 1),)),
        ([0, 1.2], ((0, 1.2),)),
        ([0.3, 1.2], ((0.3, 1.2),)),
        (((0, 1)), ((0, 1),)),
        (([0, 1]), ((0, 1),)),
        ([(0, 1)], ((0, 1),)),
        ([[0, 1]], ((0, 1),)),
        ([[0, 1.2]], ((0, 1.2),)),
        ([[0.3, 1.2]], ((0.3, 1.2),)),
        # Test for numpy dtypes
        (int32(5), ((float(int32(5)),),)),
        (int64(6), ((float(int64(6)),),)),
        (float32(3.2), ((float(float32(3.2)),),)),
        (float64(6.4), ((float(float64(6.4)),),)),
        ([int32(5), float32(3.2)], ((float(int32(5)), float(float32(3.2))),)),
    )
    @unpack
    def test_validate_parameter_values(self, _parameter_values, expected):
        """Test parameter_values standardization."""
        for parameter_values in [_parameter_values, array(_parameter_values)]:  # Numpy
            self.assertEqual(validation_v1._validate_parameter_values(parameter_values), expected)
            self.assertEqual(
                validation_v1._validate_parameter_values(None, default=parameter_values), expected
            )

    @data(
        "ERROR",
        ("E", "R", "R", "O", "R"),
        (["E", "R", "R"], ["O", "R"]),
        1j,
        (1j,),
        ((1j,),),
        True,
        False,
        float("inf"),
        float("-inf"),
        float("nan"),
    )
    def test_type_error(self, parameter_values):
        """Test type error if invalid input."""
        with self.assertRaises(TypeError):
            validation_v1._validate_parameter_values(parameter_values)

    def test_value_error(self):
        """Test value error if no parameter_values or default are provided."""
        with self.assertRaises(ValueError):
            validation_v1._validate_parameter_values(None)
