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


"""Pass manager test cases."""

from test.python.passmanager import BasePassManagerTestCase

from logging import getLogger

from qiskit.passmanager import GenericPass
from qiskit.passmanager import PassManagerState, WorkflowStatus, PropertySet
from qiskit.passmanager.compilation_status import RunState


class TestGenericPass(BasePassManagerTestCase):
    """Tests for the GenericPass subclass."""

    def setUp(self):
        super().setUp()

        self.state = PassManagerState(
            workflow_status=WorkflowStatus(),
            property_set=PropertySet(),
        )

    def test_run_task(self):
        """Test case: Simple successful task execution."""

        class Task(GenericPass):
            def run(self, passmanager_ir):
                return passmanager_ir

        task = Task()
        data = "test_data"
        expected = [r"Pass: Task - (\d*\.)?\d+ \(ms\)"]

        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 1)
        self.assertIn(task, self.state.workflow_status.completed_passes)
        self.assertEqual(self.state.workflow_status.previous_run, RunState.SUCCESS)

    def test_failure_task(self):
        """Test case: Log is created regardless of success."""

        class TestError(Exception):
            pass

        class RaiseError(GenericPass):
            def run(self, passmanager_ir):
                raise TestError

        task = RaiseError()
        data = "test_data"
        expected = [r"Pass: RaiseError - (\d*\.)?\d+ \(ms\)"]

        with self.assertLogContains(expected):
            with self.assertRaises(TestError):
                task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 0)
        self.assertNotIn(task, self.state.workflow_status.completed_passes)
        self.assertEqual(self.state.workflow_status.previous_run, RunState.FAIL)

    def test_requires(self):
        """Test case: Dependency tasks are run in advance to user provided task."""

        class TaskA(GenericPass):
            def run(self, passmanager_ir):
                return passmanager_ir

        class TaskB(GenericPass):
            def __init__(self):
                super().__init__()
                self.requires = [TaskA()]

            def run(self, passmanager_ir):
                return passmanager_ir

        task = TaskB()
        data = "test_data"
        expected = [
            r"Pass: TaskA - (\d*\.)?\d+ \(ms\)",
            r"Pass: TaskB - (\d*\.)?\d+ \(ms\)",
        ]
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 2)

    def test_requires_in_list(self):
        """Test case: Dependency tasks are not executed multiple times."""

        class TaskA(GenericPass):
            def run(self, passmanager_ir):
                return passmanager_ir

        class TaskB(GenericPass):
            def __init__(self):
                super().__init__()
                self.requires = [TaskA()]

            def run(self, passmanager_ir):
                return passmanager_ir

        task = TaskB()
        data = "test_data"
        expected = [
            r"Pass: TaskB - (\d*\.)?\d+ \(ms\)",
        ]
        self.state.workflow_status.completed_passes.add(task.requires[0])  # already done
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state)
        self.assertEqual(self.state.workflow_status.count, 1)

    def test_run_with_callable(self):
        """Test case: Callable is called after generic pass is run."""

        def test_callable(task, passmanager_ir, property_set, running_time, count):
            logger = getLogger()
            logger.info("%s is running on %s", task.name(), passmanager_ir)

        class Task(GenericPass):
            def run(self, passmanager_ir):
                return passmanager_ir

        task = Task()
        data = "test_data"
        expected = [
            r"Pass: Task - (\d*\.)?\d+ \(ms\)",
            r"Task is running on test_data",
        ]
        with self.assertLogContains(expected):
            task.execute(passmanager_ir=data, state=self.state, callback=test_callable)
