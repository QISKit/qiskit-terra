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

"""FixedPoint pass testing"""

import unittest

from qiskit.transpiler.passes import FixedPoint
from test import QiskitTestCase


class TestFixedPointPass(QiskitTestCase):
    """Tests for FixedPoint pass."""

    def setUp(self):
        super().setUp()
        self.pass_ = FixedPoint("property")
        self.pset = self.pass_.property_set
        self.dag = None  # The pass do not read the DAG.

    def test_fixed_point_setting_to_none(self):
        """Setting a property to None twice does not create a fixed-point."""
        self.pass_.property_set["property"] = None
        self.pass_.run(self.dag)
        self.pass_.property_set["property"] = None
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])

    def test_fixed_point_reached(self):
        """Setting a property to the same value twice creates a fixed-point."""
        self.pset["property"] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])
        self.pset["property"] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset["property_fixed_point"])

    def test_fixed_point_not_reached(self):
        """Setting a property with different values does not create a fixed-point."""
        self.pset["property"] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])
        self.pset["property"] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])

    def test_fixed_point_left(self):
        """A fixed-point is not permanent."""
        self.pset["property"] = 1
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])
        self.pset["property"] = 1
        self.pass_.run(self.dag)
        self.assertTrue(self.pset["property_fixed_point"])
        self.pset["property"] = 2
        self.pass_.run(self.dag)
        self.assertFalse(self.pset["property_fixed_point"])

    def test_fixed_point_tuple_getter(self):
        """Test correctness of the FixedPoint pass when getter returns a tuple
        of values from the property set.
        """
        pass_ = FixedPoint("xy", lambda pset: (pset["x"], pset["y"]))
        pset = pass_.property_set

        pset["x"] = 1
        pset["y"] = 2
        pass_.run(dag=None)
        self.assertFalse(pset["xy_fixed_point"])

        pset["x"] = 1
        pset["y"] = 1
        pass_.run(dag=None)
        self.assertFalse(pset["xy_fixed_point"])

        pset["x"] = 1
        pset["y"] = 1
        pass_.run(dag=None)
        self.assertTrue(pset["xy_fixed_point"])

    def test_fixed_point_sum_getter(self):
        """Test correctness of the FixedPoint pass when getter returns a sum
        of two values from the property set.
        """
        pass_ = FixedPoint("xy", lambda pset: pset["x"] + pset["y"])
        pset = pass_.property_set

        pset["x"] = 1
        pset["y"] = 4
        pass_.run(dag=None)
        self.assertFalse(pset["xy_fixed_point"])

        pset["x"] = 2
        pset["y"] = 2
        pass_.run(dag=None)
        self.assertFalse(pset["xy_fixed_point"])

        pset["x"] = 1
        pset["y"] = 3
        pass_.run(dag=None)
        self.assertTrue(pset["xy_fixed_point"])


if __name__ == "__main__":
    unittest.main()
