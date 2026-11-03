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

"""Test __repr__ method for PassManager class."""

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, Depth
from test import QiskitTestCase


class TestPassManagerRepr(QiskitTestCase):
    """Tests for PassManager.__repr__ method."""

    def test_passmanager_repr_empty(self):
        """Test empty PassManager repr."""
        pm = PassManager()
        result = repr(pm)
        expected = f"<PassManager with 0 sets, 0 passes, {pm.max_iteration} max iterations, and 0 properties>"
        self.assertEqual(result, expected)

    def test_passmanager_repr_single_pass(self):
        """Test PassManager repr with single pass."""
        pm = PassManager(Optimize1qGates())
        result = repr(pm)
        expected = f"<PassManager with 1 sets, 1 passes, {pm.max_iteration} max iterations, and 0 properties>"
        self.assertEqual(result, expected)

    def test_passmanager_repr_multiple_passes(self):
        """Test PassManager repr with multiple passes."""
        pm = PassManager([Optimize1qGates(), Depth()])
        result = repr(pm)
        expected = f"<PassManager with 1 sets, 2 passes, {pm.max_iteration} max iterations, and 0 properties>"
        self.assertEqual(result, expected)

    def test_passmanager_repr_with_max_iteration(self):
        """Test PassManager repr with custom max_iteration."""
        pm = PassManager(Optimize1qGates(), max_iteration=5)
        result = repr(pm)
        expected = "<PassManager with 1 sets, 1 passes, 5 max iterations, and 0 properties>"
        self.assertEqual(result, expected)

    def test_passmanager_repr_with_properties(self):
        """Test PassManager repr with properties."""
        pm = PassManager(Optimize1qGates())
        pm.property_set["test_prop1"] = "value1"
        pm.property_set["test_prop2"] = "value2"
        pm.property_set["test_prop3"] = "value3"
        result = repr(pm)
        expected = f"<PassManager with 1 sets, 1 passes, {pm.max_iteration} max iterations, and 3 properties ('test_prop1', 'test_prop2', 'test_prop3')>"
        self.assertEqual(result, expected)

    def test_passmanager_repr_with_many_properties(self):
        """Test PassManager repr with more than 3 properties (should truncate)."""
        pm = PassManager(Optimize1qGates())
        pm.property_set["prop1"] = "value1"
        pm.property_set["prop2"] = "value2"
        pm.property_set["prop3"] = "value3"
        pm.property_set["prop4"] = "value4"
        pm.property_set["prop5"] = "value5"
        result = repr(pm)
        # Verify full string - should show first 3 properties and "..."
        expected = f"<PassManager with 1 sets, 1 passes, {pm.max_iteration} max iterations, and 5 properties ('prop1', 'prop2', 'prop3', ...)>"
        self.assertEqual(result, expected)

