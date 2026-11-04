# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MinimumPoint pass testing"""

import math

from qiskit.transpiler.passes import MinimumPoint
from qiskit.dagcircuit import DAGCircuit
from test import QiskitTestCase


class TestMinimumPointtPass(QiskitTestCase):
    """Tests for MinimumPoint pass."""

    def test_minimum_point_reached_fixed_point_single_field(self):
        """Test a fixed point is reached with a single field."""

        min_pass = MinimumPoint(["depth"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["depth"] = 42
        min_pass.run(dag)
        # After first iteration state is only initialized but not populated
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf,), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Second iteration
        min_pass.run(dag)
        # After second iteration the state is initialized
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (42,))
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Third iteration
        out_dag = min_pass.run(dag)
        # After 3rd iteration we've reached a fixed point equal to our minimum
        # ooint so we return the minimum dag
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual((42,), state.score)
        self.assertTrue(min_pass.property_set["test_minimum_point"])
        # In case of fixed point we don't return copy but the state of the dag
        # after the fixed point so only assert equality
        self.assertEqual(out_dag, state.dag)

    def test_minimum_point_reached_fixed_point_multiple_fields(self):
        """Test a fixed point is reached with a multiple fields."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        # After first iteration state is only initialized but not populated
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Two
        min_pass.run(dag)
        # After second iteration the state is initialized
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (0.875, 15, 20))
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Three
        out_dag = min_pass.run(dag)
        # After 3rd iteration we've reached a fixed point equal to our minimum
        # ooint so we return the minimum dag
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual(state.score, (0.875, 15, 20))
        self.assertTrue(min_pass.property_set["test_minimum_point"])
        # In case of fixed point we don't return copy but the state of the dag
        # after the fixed point so only assert equality
        self.assertEqual(out_dag, state.dag)

    def test_min_over_backtrack_range(self):
        """Test minimum returned over backtrack depth."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        # After first iteration state is only initialized but not populated
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Two
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # After second iteration we've set a current minimum state
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration three
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # After third iteration score is worse than minimum point just bump since
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration four
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        # After fourth iteration score is worse than minimum point just bump since
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration five
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        # After fifth iteration score is worse than minimum point just bump since
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration six
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        out_dag = min_pass.run(dag)
        # After sixth iteration score is worse, but we've reached backtrack depth and the
        # dag copy is returned
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 5)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertTrue(min_pass.property_set["test_minimum_point"])
        self.assertIs(out_dag, state.dag)

    def test_min_reset_backtrack_range(self):
        """Test minimum resets backtrack depth."""
        min_pass = MinimumPoint(["fidelity", "depth", "size"], prefix="test")
        dag = DAGCircuit()
        min_pass.property_set["fidelity"] = 0.875
        min_pass.property_set["depth"] = 15
        min_pass.property_set["size"] = 20
        min_pass.run(dag)
        # After first iteration state is only initialized but not populated
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 0)
        self.assertEqual((math.inf, math.inf, math.inf), state.score)
        self.assertIsNone(state.dag)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration two:
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # After second iteration we've set a current minimum state
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration three:
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # Third iteration the score is worse (because of depth increasing) so do
        # not set new minimum point
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration four:
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        # Fourth iteration the score is also worse than minimum although depth
        # is better than iteration three it's still higher than the minimum point
        # Also size has increased:. Do not update minimum point and since is increased
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration five
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        # Fifth iteration the score is also worse than minimum although the same
        # with previous iteration. This means do not update minimum point and bump since
        # value
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 25, 35), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Six
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 10
        min_pass.property_set["size"] = 10
        min_pass.run(dag)
        # Sixth iteration the score is lower (fidelity is the same but depth and size decreased)
        # set new minimum point
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 1)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration seven
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 25
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # Iteration seven the score is worse than the minimum point.  Do not update minimum point
        # and since is bumped
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 2)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Eight
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 45
        min_pass.property_set["size"] = 35
        min_pass.run(dag)
        # Iteration eight the score is worse than the minimum point. Do not update minimum point
        # and since is bumped
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 3)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration Nine
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        min_pass.run(dag)
        # Iteration nine the score is worse than the minimum point. Do not update minimum point
        # and since is bumped
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 4)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertIsNone(min_pass.property_set["test_minimum_point"])

        # Iteration 10
        min_pass.property_set["fidelity"] = 0.775
        min_pass.property_set["depth"] = 36
        min_pass.property_set["size"] = 40
        out_dag = min_pass.run(dag)
        # Iteration 10 score is worse, but we've reached the set backtrack
        # depth of 5 iterations since the last minimum so we exit here
        state = min_pass.property_set["test_minimum_point_state"]
        self.assertEqual(state.since, 5)
        self.assertEqual((0.775, 10, 10), state.score)
        self.assertTrue(min_pass.property_set["test_minimum_point"])
        self.assertIs(out_dag, state.dag)
