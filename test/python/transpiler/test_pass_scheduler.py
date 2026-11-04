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

"""Transpiler testing"""

import io
from logging import StreamHandler, getLogger
import unittest.mock
import sys

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, TranspilerError
from qiskit.passmanager import DoWhileController, ConditionalController
from test import QiskitTestCase
from ._dummy_passes import (
    PassA_TP_NR_NP,
    PassB_TP_RA_PA,
    PassC_TP_RA_PA,
    PassD_TP_NR_NP,
    PassE_AP_NR_NP,
    PassF_reduce_dag_property,
    PassJ_Bad_NoReturn,
    PassK_check_fixed_point_property,
    PassM_AP_NR_NP,
)


class SchedulerTestCase(QiskitTestCase):
    """Asserts for the scheduler."""

    def assertScheduler(self, circuit, passmanager, expected):
        """
        Run `transpile(circuit, passmanager)` and check
        if the passes run as expected.

        Args:
            circuit (QuantumCircuit): Circuit to transform via transpilation.
            passmanager (PassManager): pass manager instance for the transpilation process
            expected (list): List of things the passes are logging
        """
        logger = "LocalLogger"
        with self.assertLogs(logger, level="INFO") as cm:
            out = passmanager.run(circuit)
        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual([record.message for record in cm.records], expected)

    def assertSchedulerRaises(self, circuit, passmanager, expected, exception_type):
        """
        Run `transpile(circuit, passmanager)` and check
        if the passes run as expected until exception_type is raised.

        Args:
            circuit (QuantumCircuit): Circuit to transform via transpilation
            passmanager (PassManager): pass manager instance for the transpilation process
            expected (list): List of things the passes are logging
            exception_type (Exception): Exception that is expected to be raised.
        """
        logger = "LocalLogger"
        with self.assertLogs(logger, level="INFO") as cm:
            self.assertRaises(exception_type, passmanager.run, circuit)
        self.assertEqual([record.message for record in cm.records], expected)


class TestPassManagerInit(SchedulerTestCase):
    """The pass manager sets things at init time."""

    def test_passes(self):
        """A single chain of passes, with Requests and Preserves, at __init__ time"""
        circuit = QuantumCircuit(QuantumRegister(1))
        passmanager = PassManager(
            passes=[
                PassC_TP_RA_PA(),  # Request: PassA / Preserves: PassA
                PassB_TP_RA_PA(),  # Request: PassA / Preserves: PassA
                PassD_TP_NR_NP(argument1=[1, 2]),  # Requires: {}/ Preserves: {}
                PassB_TP_RA_PA(),
            ]
        )
        self.assertScheduler(
            circuit,
            passmanager,
            [
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassC_TP_RA_PA",
                "run transformation pass PassB_TP_RA_PA",
                "run transformation pass PassD_TP_NR_NP",
                "argument [1, 2]",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassB_TP_RA_PA",
            ],
        )


class TestUseCases(SchedulerTestCase):
    """Combine passes in different ways and checks that passes are run
    in the right order."""

    def setUp(self):
        super().setUp()
        self.circuit = QuantumCircuit(QuantumRegister(1))
        self.passmanager = PassManager()

    def test_chain(self):
        """A single chain of passes, with Requires and Preserves."""
        self.passmanager.append(PassC_TP_RA_PA())  # Requires: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Requires: PassA / Preserves: PassA
        self.passmanager.append(PassD_TP_NR_NP(argument1=[1, 2]))  # Requires: {}/ Preserves: {}
        self.passmanager.append(PassB_TP_RA_PA())
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassC_TP_RA_PA",
                "run transformation pass PassB_TP_RA_PA",
                "run transformation pass PassD_TP_NR_NP",
                "argument [1, 2]",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassB_TP_RA_PA",
            ],
        )

    def test_conditional_passes_true(self):
        """A pass set with a conditional parameter. The callable is True."""
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(
            ConditionalController(
                PassA_TP_NR_NP(), condition=lambda property_set: property_set["property"]
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassE_AP_NR_NP",
                "set property as True",
                "run transformation pass PassA_TP_NR_NP",
            ],
        )

    def test_conditional_passes_true_fc(self):
        """A pass set with a conditional parameter (with FlowController). The callable is True."""
        self.passmanager.append(PassE_AP_NR_NP(True))

        self.passmanager.append(
            ConditionalController(
                [PassA_TP_NR_NP()], condition=lambda property_set: property_set["property"]
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassE_AP_NR_NP",
                "set property as True",
                "run transformation pass PassA_TP_NR_NP",
            ],
        )

    def test_conditional_passes_false(self):
        """A pass set with a conditional parameter. The callable is False."""
        self.passmanager.append(PassE_AP_NR_NP(False))
        self.passmanager.append(
            ConditionalController(
                PassA_TP_NR_NP(), condition=lambda property_set: property_set["property"]
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            ["run analysis pass PassE_AP_NR_NP", "set property as False"],
        )

    def test_conditional_and_loop(self):
        """Run a conditional first, then a loop."""
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(
            ConditionalController(
                DoWhileController(
                    [
                        PassK_check_fixed_point_property(),
                        PassA_TP_NR_NP(),
                        PassF_reduce_dag_property(),
                    ],
                    do_while=lambda property_set: not property_set["property_fixed_point"],
                ),
                condition=lambda property_set: property_set["property"],
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassE_AP_NR_NP",
                "set property as True",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 8 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 6",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 6 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 5",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 5 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 4",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 4 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 3",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 3 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
            ],
        )

    def test_loop_and_conditional(self):
        """Run a loop first, then a conditional."""
        self.passmanager.append(PassK_check_fixed_point_property())
        self.passmanager.append(
            ConditionalController(
                DoWhileController(
                    [
                        PassK_check_fixed_point_property(),
                        PassA_TP_NR_NP(),
                        PassF_reduce_dag_property(),
                    ],
                    do_while=lambda property_set: not property_set["property_fixed_point"],
                ),
                condition=lambda property_set: not property_set["property_fixed_point"],
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassG_calculates_dag_property",
                "set property as 8 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 6",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 6 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 5",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 5 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 4",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 4 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 3",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 3 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
            ],
        )

    def test_do_not_repeat_based_on_preservation(self):
        """When a pass is still a valid pass (because the following passes
        preserved it), it should not run again."""
        self.passmanager.append([PassB_TP_RA_PA(), PassA_TP_NR_NP(), PassB_TP_RA_PA()])
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            ["run transformation pass PassA_TP_NR_NP", "run transformation pass PassB_TP_RA_PA"],
        )

    def test_do_not_repeat_based_on_idempotence(self):
        """Repetition can be optimized to a single execution when
        the pass is idempotent."""
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append([PassA_TP_NR_NP(), PassA_TP_NR_NP()])
        self.passmanager.append(PassA_TP_NR_NP())
        self.assertScheduler(
            self.circuit, self.passmanager, ["run transformation pass PassA_TP_NR_NP"]
        )

    def test_non_idempotent_pass(self):
        """Two or more runs of a non-idempotent pass cannot be optimized."""
        self.passmanager.append(PassF_reduce_dag_property())
        self.passmanager.append([PassF_reduce_dag_property(), PassF_reduce_dag_property()])
        self.passmanager.append(PassF_reduce_dag_property())
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 6",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 5",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 4",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 3",
            ],
        )

    def test_analysis_pass_is_idempotent(self):
        """Analysis passes are idempotent."""
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(
            self.circuit, passmanager, ["run analysis pass PassE_AP_NR_NP", "set property as 1"]
        )

    def test_ap_before_and_after_a_tp(self):
        """A default transformation does not preserves anything
        and analysis passes need to be re-run"""
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        passmanager.append(PassA_TP_NR_NP())
        passmanager.append(PassE_AP_NR_NP(argument1=1))
        self.assertScheduler(
            self.circuit,
            passmanager,
            [
                "run analysis pass PassE_AP_NR_NP",
                "set property as 1",
                "run transformation pass PassA_TP_NR_NP",
                "run analysis pass PassE_AP_NR_NP",
                "set property as 1",
            ],
        )

    def test_pass_no_return(self):
        """Transformation passes that don't return a DAG raise error."""
        self.passmanager.append(PassJ_Bad_NoReturn())
        self.assertSchedulerRaises(
            self.circuit,
            self.passmanager,
            ["run transformation pass PassJ_Bad_NoReturn"],
            TranspilerError,
        )

    def test_fixed_point_pass(self):
        """A pass set with a do_while parameter that checks for a fixed point."""
        self.passmanager.append(
            DoWhileController(
                [
                    PassK_check_fixed_point_property(),
                    PassA_TP_NR_NP(),
                    PassF_reduce_dag_property(),
                ],
                do_while=lambda property_set: not property_set["property_fixed_point"],
            )
        )
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassG_calculates_dag_property",
                "set property as 8 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 6",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 6 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 5",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 5 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 4",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 4 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 3",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 3 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 2 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 2",
            ],
        )

    def test_fixed_point_fc(self):
        """A fixed point scheduler with flow control."""
        self.passmanager.append(
            DoWhileController(
                [
                    PassK_check_fixed_point_property(),
                    PassA_TP_NR_NP(),
                    PassF_reduce_dag_property(),
                ],
                do_while=lambda property_set: not property_set["property_fixed_point"],
            )
        )

        expected = [
            "run analysis pass PassG_calculates_dag_property",
            "set property as 8 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 6",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 6 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 5",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 5 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 4",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 4 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 3",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 3 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
        ]

        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_fixed_point_pass_max_iteration(self):
        """A pass set with a do_while parameter that checks that
        the max_iteration is raised."""
        self.passmanager.append(
            DoWhileController(
                [PassK_check_fixed_point_property(), PassA_TP_NR_NP(), PassF_reduce_dag_property()],
                do_while=lambda property_set: not property_set["property_fixed_point"],
                options={"max_iteration": 2},
            ),
        )
        self.assertSchedulerRaises(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassG_calculates_dag_property",
                "set property as 8 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 6",
                "run analysis pass PassG_calculates_dag_property",
                "set property as 6 (from dag.property)",
                "run analysis pass PassK_check_fixed_point_property",
                "run transformation pass PassA_TP_NR_NP",
                "run transformation pass PassF_reduce_dag_property",
                "dag property = 5",
            ],
            TranspilerError,
        )

    def test_fresh_initial_state(self):
        """New construction gives fresh instance."""
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassM_AP_NR_NP(argument1=1))
        self.assertScheduler(
            self.circuit,
            self.passmanager,
            [
                "run analysis pass PassM_AP_NR_NP",
                "self.argument1 = 2",
                "run transformation pass PassA_TP_NR_NP",
                "run analysis pass PassM_AP_NR_NP",
                "self.argument1 = 2",
            ],
        )

    def test_nested_conditional_in_loop(self):
        """Run a loop with a nested conditional."""
        nested_conditional = ConditionalController(
            PassA_TP_NR_NP(), condition=lambda property_set: property_set["property"] >= 5
        )
        self.passmanager.append(
            DoWhileController(
                [
                    PassK_check_fixed_point_property(),
                    nested_conditional,
                    PassF_reduce_dag_property(),
                ],
                do_while=lambda property_set: not property_set["property_fixed_point"],
            )
        )
        expected = [
            "run analysis pass PassG_calculates_dag_property",
            "set property as 8 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 6",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 6 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 5",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 5 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 4",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 4 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 3",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 3 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)


class TestControlFlowPlugin(SchedulerTestCase):
    """Testing the control flow plugin system."""

    def setUp(self):
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))


class TestDumpPasses(SchedulerTestCase):
    """Testing the passes method."""

    def test_passes(self):
        """Dump passes in different FlowControllerLinear"""
        passmanager = PassManager()
        passmanager.append(PassC_TP_RA_PA())
        passmanager.append(PassB_TP_RA_PA())
        expected = PassC_TP_RA_PA(), PassB_TP_RA_PA()
        self.assertEqual(expected, passmanager.to_flow_controller().tasks)

    def test_passes_in_linear(self):
        """Dump passes in the same FlowControllerLinear"""
        passmanager = PassManager(
            passes=[
                PassC_TP_RA_PA(),
                PassB_TP_RA_PA(),
                PassD_TP_NR_NP(argument1=[1, 2]),
                PassB_TP_RA_PA(),
            ]
        )
        expected = (
            PassC_TP_RA_PA(),
            PassB_TP_RA_PA(),
            PassD_TP_NR_NP(argument1=[1, 2]),
            PassB_TP_RA_PA(),
        )
        self.assertEqual(expected, passmanager.to_flow_controller().tasks)

    def test_conditional_and_loop(self):
        """Dump passes with a conditional and a loop."""
        nested_controller = ConditionalController(
            DoWhileController(
                [
                    PassK_check_fixed_point_property(),
                    PassA_TP_NR_NP(),
                    PassF_reduce_dag_property(),
                ],
                do_while=lambda property_set: not property_set["property_fixed_point"],
            ),
            condition=lambda property_set: property_set["property_fixed_point"],
        )

        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(True))
        passmanager.append(nested_controller)

        expected = (PassE_AP_NR_NP(True), nested_controller)
        self.assertEqual(expected, passmanager.to_flow_controller().tasks)


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogPasses(QiskitTestCase):
    """Testing the log_passes option."""

    def setUp(self):
        super().setUp()
        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel("DEBUG")
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertPassLog(self, passmanager, list_of_passes):
        """Runs the passmanager and checks that the elements in
        passmanager.property_set['pass_log'] match list_of_passes (the names)."""
        passmanager.run(self.circuit)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        pass_log_lines = [x for x in output_lines if x.startswith("Pass:")]
        for index, pass_name in enumerate(list_of_passes):
            self.assertTrue(pass_log_lines[index].startswith(f"Pass: {pass_name} -"))

    def test_passes(self):
        """Dump passes in different FlowControllerLinear"""
        passmanager = PassManager()
        passmanager.append(PassC_TP_RA_PA())
        passmanager.append(PassB_TP_RA_PA())

        self.assertPassLog(passmanager, ["PassA_TP_NR_NP", "PassC_TP_RA_PA", "PassB_TP_RA_PA"])

    def test_passes_in_linear(self):
        """Dump passes in the same FlowControllerLinear"""
        passmanager = PassManager(
            passes=[
                PassC_TP_RA_PA(),
                PassB_TP_RA_PA(),
                PassD_TP_NR_NP(argument1=[1, 2]),
                PassB_TP_RA_PA(),
            ]
        )

        self.assertPassLog(
            passmanager,
            [
                "PassA_TP_NR_NP",
                "PassC_TP_RA_PA",
                "PassB_TP_RA_PA",
                "PassD_TP_NR_NP",
                "PassA_TP_NR_NP",
                "PassB_TP_RA_PA",
            ],
        )

    def test_conditional_and_loop(self):
        """Dump passes with a conditional and a loop"""
        passmanager = PassManager()
        passmanager.append(PassE_AP_NR_NP(True))
        passmanager.append(
            ConditionalController(
                DoWhileController(
                    [
                        PassK_check_fixed_point_property(),
                        PassA_TP_NR_NP(),
                        PassF_reduce_dag_property(),
                    ],
                    do_while=lambda property_set: not property_set["property_fixed_point"],
                ),
                condition=lambda property_set: property_set["property_fixed_point"],
            )
        )
        self.assertPassLog(passmanager, ["PassE_AP_NR_NP"])


class TestPassManagerReuse(SchedulerTestCase):
    """The PassManager instance should be reusable."""

    def setUp(self):
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_chain_twice(self):
        """Run a chain twice."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
            "run transformation pass PassB_TP_RA_PA",
        ]

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_conditional_twice(self):
        """Run a conditional twice."""
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(
            ConditionalController(
                PassA_TP_NR_NP(), condition=lambda property_set: property_set["property"]
            )
        )

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as True",
            "run transformation pass PassA_TP_NR_NP",
        ]

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_fixed_point_twice(self):
        """A fixed point scheduler, twice."""
        self.passmanager.append(
            DoWhileController(
                [
                    PassK_check_fixed_point_property(),
                    PassA_TP_NR_NP(),
                    PassF_reduce_dag_property(),
                ],
                do_while=lambda property_set: not property_set["property_fixed_point"],
            )
        )

        expected = [
            "run analysis pass PassG_calculates_dag_property",
            "set property as 8 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 6",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 6 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 5",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 5 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 4",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 4 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 3",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 3 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
        ]

        self.assertScheduler(self.circuit, self.passmanager, expected)
        self.assertScheduler(self.circuit, self.passmanager, expected)


class TestPassManagerChanges(SchedulerTestCase):
    """Test PassManager manipulation with changes"""

    def setUp(self):
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_replace0(self):
        """Test passmanager.replace(0, ...)."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager.replace(0, PassB_TP_RA_PA())

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace1(self):
        """Test passmanager.replace(1, ...)."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager.replace(1, PassC_TP_RA_PA())

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove0(self):
        """Test passmanager.remove(0)."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager.remove(0)

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove1(self):
        """Test passmanager.remove(1)."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager.remove(1)

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_remove_minus_1(self):
        """Test passmanager.remove(-1)."""
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager.remove(-1)

        expected = ["run transformation pass PassA_TP_NR_NP"]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_setitem(self):
        """Test passmanager[1] = ..."""
        self.passmanager.append(PassC_TP_RA_PA())  # Request: PassA / Preserves: PassA
        self.passmanager.append(PassB_TP_RA_PA())  # Request: PassA / Preserves: PassA

        self.passmanager[1] = PassC_TP_RA_PA()

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace_with_conditional(self):
        """Replace a pass with a conditional pass."""
        self.passmanager.append(PassE_AP_NR_NP(False))
        self.passmanager.append(PassB_TP_RA_PA())

        self.passmanager.replace(
            1,
            ConditionalController(
                PassA_TP_NR_NP(), condition=lambda property_set: property_set["property"]
            ),
        )

        expected = ["run analysis pass PassE_AP_NR_NP", "set property as False"]
        self.assertScheduler(self.circuit, self.passmanager, expected)

    def test_replace_error(self):
        """Replace a non-existing index."""
        self.passmanager.append(PassB_TP_RA_PA())

        with self.assertRaises(TranspilerError):
            self.passmanager.replace(99, PassA_TP_NR_NP())


class TestPassManagerSlicing(SchedulerTestCase):
    """test PassManager slicing."""

    def setUp(self):
        super().setUp()
        self.passmanager = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_empty_passmanager_length(self):
        """test len(PassManager) when PassManager is empty"""
        length = len(self.passmanager)

        expected_length = 0
        self.assertEqual(length, expected_length)

    def test_passmanager_length(self):
        """test len(PassManager) when PassManager is not empty"""
        self.passmanager.append(PassA_TP_NR_NP())
        self.passmanager.append(PassA_TP_NR_NP())

        length = len(self.passmanager)

        expected_length = 2
        self.assertEqual(length, expected_length)

    def test_accessing_passmanager_by_index(self):
        """test accessing PassManager's passes by index"""
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassC_TP_RA_PA())

        new_passmanager = self.passmanager[1]

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_index_with_condition(self):
        """test accessing PassManager's conditioned passes by index"""
        self.passmanager.append(PassF_reduce_dag_property())
        self.passmanager.append(
            ConditionalController(
                DoWhileController(
                    [
                        PassK_check_fixed_point_property(),
                        PassA_TP_NR_NP(),
                        PassF_reduce_dag_property(),
                    ],
                    do_while=lambda property_set: not property_set["property_fixed_point"],
                ),
                condition=lambda property_set: True,
            )
        )

        new_passmanager = self.passmanager[1]

        expected = [
            "run analysis pass PassG_calculates_dag_property",
            "set property as 8 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 6",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 6 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 5",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 5 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 4",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 4 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 3",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 3 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
            "run analysis pass PassG_calculates_dag_property",
            "set property as 2 (from dag.property)",
            "run analysis pass PassK_check_fixed_point_property",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassF_reduce_dag_property",
            "dag property = 2",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_range(self):
        """test accessing PassManager's passes by range"""
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassC_TP_RA_PA())
        self.passmanager.append(PassD_TP_NR_NP())

        new_passmanager = self.passmanager[1:3]

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_by_range_with_condition(self):
        """test accessing PassManager's passes by range with condition"""
        self.passmanager.append(PassB_TP_RA_PA())
        self.passmanager.append(PassE_AP_NR_NP(True))
        self.passmanager.append(
            ConditionalController(
                PassA_TP_NR_NP(),
                condition=lambda property_set: property_set["property"],
            )
        )
        self.passmanager.append(PassB_TP_RA_PA())

        new_passmanager = self.passmanager[1:3]

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as True",
            "run transformation pass PassA_TP_NR_NP",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_accessing_passmanager_error(self):
        """testing accessing a pass item not in list"""
        self.passmanager.append(PassB_TP_RA_PA())

        with self.assertRaises(IndexError):
            self.passmanager = self.passmanager[99]


class TestPassManagerConcatenation(SchedulerTestCase):
    """test PassManager concatenation by + operator."""

    def setUp(self):
        super().setUp()
        self.passmanager1 = PassManager()
        self.passmanager2 = PassManager()
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def test_concatenating_passmanagers(self):
        """test adding two PassManagers together"""
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager2.append(PassC_TP_RA_PA())

        new_passmanager = self.passmanager1 + self.passmanager2

        expected = [
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_concatenating_passmanagers_with_condition(self):
        """test adding two pass managers with condition"""
        self.passmanager1.append(PassE_AP_NR_NP(True))
        self.passmanager1.append(PassB_TP_RA_PA())
        self.passmanager2.append(
            ConditionalController(
                PassC_TP_RA_PA(), condition=lambda property_set: property_set["property"]
            )
        )
        self.passmanager2.append(PassB_TP_RA_PA())

        new_passmanager = self.passmanager1 + self.passmanager2

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as True",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
            "run transformation pass PassC_TP_RA_PA",
            "run transformation pass PassB_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, new_passmanager, expected)

    def test_adding_pass_to_passmanager(self):
        """test adding a pass to PassManager"""
        self.passmanager1.append(PassE_AP_NR_NP(argument1=1))
        self.passmanager1.append(PassB_TP_RA_PA())

        self.passmanager1 += PassC_TP_RA_PA()

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as 1",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_list_of_passes_to_passmanager(self):
        """test adding a list of passes to PassManager"""
        self.passmanager1.append(PassE_AP_NR_NP(argument1=1))
        self.passmanager1.append(PassB_TP_RA_PA())

        self.passmanager1 += [PassC_TP_RA_PA(), PassB_TP_RA_PA()]

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as 1",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassB_TP_RA_PA",
            "run transformation pass PassC_TP_RA_PA",
            "run transformation pass PassB_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_list_of_passes_to_passmanager_with_condition(self):
        """test adding a list of passes to a PassManager that have conditions"""
        self.passmanager1.append(PassE_AP_NR_NP(False))
        self.passmanager1.append(
            ConditionalController(
                PassB_TP_RA_PA(), condition=lambda property_set: property_set["property"]
            )
        )

        self.passmanager1 += PassC_TP_RA_PA()

        expected = [
            "run analysis pass PassE_AP_NR_NP",
            "set property as False",
            "run transformation pass PassA_TP_NR_NP",
            "run transformation pass PassC_TP_RA_PA",
        ]
        self.assertScheduler(self.circuit, self.passmanager1, expected)

    def test_adding_pass_to_passmanager_error(self):
        """testing adding a non-pass item to PassManager"""
        with self.assertRaises(TypeError):
            self.passmanager1 += "not a pass"

    def test_adding_list_to_passmanager_error(self):
        """testing adding a list having a non-pass item to PassManager"""
        with self.assertRaises(TypeError):
            self.passmanager1 += [PassB_TP_RA_PA(), "not a pass"]


if __name__ == "__main__":
    unittest.main()
