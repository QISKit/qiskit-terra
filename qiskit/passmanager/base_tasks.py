# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base classes for the Qiskit passmanager optimization tasks."""

from __future__ import annotations

import logging
import time
from abc import abstractmethod, ABC
from collections.abc import Iterable, Callable, Generator
from typing import Any, TypeVar, Generic, TypeAlias

from .compilation_status import RunState, PassManagerState, PropertySet

logger = logging.getLogger(__name__)

IR = TypeVar("IR")
IR_OUT = TypeVar("IR_OUT")


class Task(ABC, Generic[IR, IR_OUT]):
    """An interface of the pass manager task.

    The task takes a Qiskit IR, and outputs new Qiskit IR after some operation on it.
    A task can rely on the :class:`.PropertySet` to communicate intermediate data among tasks.
    """

    @abstractmethod
    def execute(
        self,
        passmanager_ir: IR,
        state: PassManagerState,
        callback: Callable[[Task, IR_OUT, PropertySet, float, int], None] | None = None,
    ) -> tuple[IR_OUT, PassManagerState]:
        """Execute optimization task for input Qiskit IR.

        Args:
            passmanager_ir: Qiskit IR to optimize.
            state: State associated with workflow execution by the pass manager itself.
            callback: A callback function which is called per execution of optimization task.

        Returns:
            Optimized Qiskit IR and state of the workflow.
        """


# We can only define this type alias after Task has been defined.
Callback: TypeAlias = Callable[[Task, IR_OUT, PropertySet, float, int], None]


class GenericPass(Task[IR, IR_OUT], ABC):
    """Base class of a single pass manager task.

    A pass instance can read and write to the provided :class:`.PropertySet`,
    and may modify the input pass manager IR.
    """

    def __init__(self):
        self.property_set = PropertySet()
        self.requires: Iterable[Task] = []

    def name(self) -> str:
        """Name of the pass."""
        return self.__class__.__name__

    def execute(
        self,
        passmanager_ir: IR,
        state: PassManagerState,
        callback: Callback[IR_OUT] | None = None,
    ) -> tuple[IR_OUT, PassManagerState]:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.
        self.property_set = state.property_set

        if self.requires:

            from .flow_controllers import FlowControllerLinear

            passmanager_ir, state = FlowControllerLinear(self.requires).execute(
                passmanager_ir=passmanager_ir,
                state=state,
                callback=callback,
            )

        run_state = None
        ret = None
        start_time = time.time()
        try:
            if self not in state.workflow_status.completed_passes:
                ret = self.run(passmanager_ir)
                run_state = RunState.SUCCESS
            else:
                run_state = RunState.SKIP
        except Exception:
            run_state = RunState.FAIL
            raise
        finally:
            ret = passmanager_ir if ret is None else ret
            if run_state != RunState.SKIP:
                running_time = time.time() - start_time
                logger.info("Pass: %s - %.5f (ms)", self.name(), running_time * 1000)
                if callback is not None:
                    callback(
                        task=self,
                        passmanager_ir=ret,
                        property_set=state.property_set,
                        running_time=running_time,
                        count=state.workflow_status.count,
                    )
        return ret, self.update_status(state, run_state)

    def update_status(
        self,
        state: PassManagerState,
        run_state: RunState,
    ) -> PassManagerState:
        """Update workflow status.

        Args:
            state: Pass manager state to update.
            run_state: Completion status of current task.

        Returns:
            Updated pass manager state.
        """
        state.workflow_status.previous_run = run_state
        if run_state == RunState.SUCCESS:
            state.workflow_status.count += 1
            state.workflow_status.completed_passes.add(self)
        return state

    @abstractmethod
    def run(
        self,
        passmanager_ir: IR,
    ) -> IR_OUT:
        """Run optimization task.

        Args:
            passmanager_ir: Qiskit IR to optimize.

        Returns:
            Optimized Qiskit IR.
        """


class BaseController(Task[IR, IR_OUT], ABC):
    """Base class of controller.

    A controller is built with a collection of pass manager tasks,
    and a subclass provides a custom logic to choose next task to run.
    Note a controller can be nested into another controller,
    and a controller itself doesn't provide any subroutine to modify the input IR.
    """

    def __init__(
        self,
        options: dict[str, Any] | None = None,
    ):
        """Create new flow controller.

        Args:
            options: Option for this flow controller.
        """
        self._options = options or {}

    @abstractmethod
    def iter_tasks(
        self,
        state: PassManagerState,
    ) -> Generator[Task[Any, Any], PassManagerState, None]:
        """A custom logic to choose a next task to run.

        Controller subclass can consume the state to build a proper task pipeline.  The updated
        state after a task execution will be fed back in as the "return" value of any ``yield``
        statements.  This indicates the order of task execution is only determined at running time.
        This method is not allowed to mutate the given state object.

        Args:
            state: The state of the passmanager workflow at the beginning of this flow controller's
                execution.

        Receives:
            state: the state of pass manager after the execution of the last task that was yielded.
                The generator does not need to inspect this if it is irrelevant to its logic, nor
                update it.

        Yields:
            Task: Next task to run.
        """

    def execute(
        self,
        passmanager_ir: IR,
        state: PassManagerState,
        callback: Callable | None = None,
    ) -> tuple[IR_OUT, PassManagerState]:
        # Overriding this method is not safe.
        # Pass subclass must keep current implementation.
        # Especially, task execution may break when method signature is modified.

        task_generator = self.iter_tasks(state)
        try:
            next_task = task_generator.send(None)
        except StopIteration:
            return passmanager_ir, state
        while True:
            passmanager_ir, state = next_task.execute(
                passmanager_ir=passmanager_ir,
                state=state,
                callback=callback,
            )
            try:
                # Sending the object through the generator implies the custom controllers
                # can always rely on the latest data to choose the next task to run.
                next_task = task_generator.send(state)
            except StopIteration:
                break

        return passmanager_ir, state
