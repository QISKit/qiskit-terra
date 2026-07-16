# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A test for circuit tools"""
import unittest

from test import combine
from ddt import ddt
from numpy import pi

from qiskit.circuit.tools.pi_check import pi_check
from qiskit.circuit import Parameter
from test import QiskitTestCase


@ddt
class TestPiCheck(QiskitTestCase):
    """qiskit/visualization/tools/pi_check.py"""

    @combine(
        case=[
            (3.14, "3.14"),
            (3.141592653589793, "π"),
            (6.283185307179586, "2π"),
            (2.99, "2.99"),
            (2.9999999999999999, "3.0"),
            (0.99, "0.99"),
            (0.99999999999999999, "1.0"),
            (pi, "π"),
            (-pi, "-π"),
            (3 * pi, "3π"),
            (-3 * pi, "-3π"),
            (pi / 35, "π/35"),
            (-pi / 35, "-π/35"),
            (3 * pi / 35, "3π/35"),
            (-3 * pi / 35, "-3π/35"),
            (17 * pi, "17π"),
            (-17 * pi, "-17π"),
            (pi**2, "π**2"),
            (-(pi**2), "-π**2"),
            (1e9, "1000000000.0"),
            (-1e9, "-1000000000.0"),
            (1e-9, "1e-09"),
            (-1e-9, "-1e-09"),
            (6 * pi / 11, "6π/11"),
            (-6 * pi / 11, "-6π/11"),
            (6 * pi / 1, "6π"),
            (-6 * pi / 1, "-6π"),
            (6 * pi / 2, "3π"),
            (-6 * pi / 2, "-3π"),
            (1j * 3 / (7 * pi), "3/7πj"),
            (-1j * 3 / (7 * pi), "-3/7πj"),
            (6 * pi / 5 + 1j * 3 * pi / 7, "6π/5+3π/7j"),
            (-6 * pi / 5 + 1j * 3 * pi / 7, "-6π/5+3π/7j"),
            (6 * pi / 5 - 1j * 3 * pi / 7, "6π/5-3π/7j"),
            (-6 * pi / 5 - 1j * 3 * pi / 7, "-6π/5-3π/7j"),
            (1 / pi, "1/π"),
            (-1 / pi, "-1/π"),
            (6 / (5 * pi), "6/5π"),
            (-6 / (5 * pi), "-6/5π"),
            (-382578.0 + 0.0234567j, "-382578.0+0.0234567j"),
            (-382578.0 - 0.0234567j, "-382578.0-0.0234567j"),
        ]
    )
    def test_default(self, case):
        """Default pi_check({case[0]})='{case[1]}'"""
        input_number = case[0]
        expected_string = case[1]
        result = pi_check(input_number)
        self.assertEqual(result, expected_string)

    def test_params(self):
        """Test pi_check with Parameters"""
        x = Parameter("x")
        y = Parameter("y")
        z = Parameter("z")
        pcase = (pi**3 * x + 3 / (4 * pi) * y - 13 * pi / 2 * z, "π**3*x + 3/4π*y - 13π/2*z")
        input_number = pcase[0]
        expected_string = pcase[1]
        result = pi_check(input_number)
        self.assertEqual(result, expected_string)

    def test_params_str(self):
        """Test pi/2 displays properly in ParameterExpression - #6758"""
        x = Parameter("x")
        input_number = x + pi / 2
        expected_string = "π/2 + x"
        result = pi_check(input_number)
        self.assertEqual(result, expected_string)

    def test_near_pi_over_k_no_redundant_coefficient(self):
        """Floats near π/k format like π/k, not 1π/k — #16170"""
        a = 1.5707963276708181
        self.assertEqual(pi_check(a), "π/2")
        self.assertEqual(pi_check(a, output="latex"), "\\frac{\\pi}{2}")
        self.assertEqual(pi_check(a, output="mpl"), "$\\pi$/2")
        self.assertEqual(pi_check(a, output="qasm"), "pi/2")
        self.assertEqual(pi_check(-a), "-π/2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
