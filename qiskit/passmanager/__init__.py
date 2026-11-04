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

"""
=========================================
Pass Managers (:mod:`qiskit.passmanager`)
=========================================

.. currentmodule:: qiskit.passmanager

.. note::
    As of Qiskit v2.6, there are two separate non-interacting pass-manager systems defined in
    Qiskit, and Qiskit is beginning to transition away from the pure-Python system.  The original,
    earlier pass-manager infrastructure is discussed in :ref:`passmanager-legacy`.  The new,
    Rust-native system, which supports multiple IRs and natively handles interoperation with C and
    Rust code is discussed in :ref:`passmanager-lowering`.

.. _passmanager-legacy:

-----------------------------------
Original Python-native pass manager
-----------------------------------

Overview
========

The Qiskit pass manager is inspired by the `LLVM compiler <https://llvm.org/>`_.
The compiler infrastructure separates responsibilities into three main components:
tasks, flow controllers, and pass managers.

A compilation pipeline executes a sequence of :class:`Task` objects, each of which
takes an intermediate representation (IR) as input, performs work, and returns a, possibly
different, IR as output. Where :class:`Task` defines the interface, an atomic task is a *pass*,
which subclasses :class:`GenericPass` and implements its abstract :meth:`~GenericPass.run` method.
This is the class that should be used as base class when implementing a custom compiler pass.

Flow controllers provide execution models for a set of tasks.
The simplest flow controller is a :class:`FlowControllerLinear`, which simply executes
a set of tasks in a linear sequence.
More advanced flow controllers include loops or conditional execution.
These are, for example, used in Qiskit's preset transpiler pipelines for higher optimization
levels where optimizations are run until a convergence criterion is met.

Pass managers are responsible for managing the tasks, including scheduling required analyses and
enabling modification of the task sequence by the user.
Qiskit provides two IR-generic pass managers in this module, and a pass manager specialized
to :class:`.DAGCircuit` as IR in :mod:`qiskit.transpiler`. The IR-generic ones are:

* :class:`BasePassManager`: a pass manager with fixed IR. This pass manager allows modifying the
  set of tasks to be run and supports parallel execution of multiple inputs by means
  of :func:`.parallel_map`. This class has support for additional conversion of an input program
  representation to the internal IR, and a conversion to an output program format.

  The :class:`BasePassManager` is the base class for Qiskit's preset pass managers
  for :class:`.DAGCircuit` transpilation, such as returned by :func:`.generate_preset_pass_manager`.
  There, implicit conversions to and from :class:`.QuantumCircuit` as input and output program
  format are used.

* :class:`MultiStagePassManager`: a staged pass manager where each stage can preserve or lower the IR.
  A stage is defined by a :class:`~.passmanager.Task` or an iterable thereof, which can
  also be grouped inside a :class:`BasePassManager`.
  The stages must be set up such that the output IR of the current stage matches the input IR
  of the next stage, there are (currently) no automatic translations.

Pass managers also provide infrastructure to pass a :class:`PropertySet` with context-information
through every task and a callback function for introspection.
The :class:`PropertySet` is a free-form dictionary, which can be populated and read by a pass during
execution, or read by a flow-controller to control pass execution.
The property set is portable and handed over from pass to pass at execution.
In addition to the property set, tasks also receive a :class:`WorkflowStatus` data structure.
This object is initialized when the pass manager is run and handed over to underlying tasks.
The status is updated after every pass is run, and contains information about the pipeline state
(number of passes run, failure state, and so on) as opposed to the :class:`PropertySet`, which
contains information about the IR being optimized.

The callback is called by :class:`GenericPass` instances expecting the following signature:

.. code-block:: python

    def callback(
        *,
        task: Task[IR_IN, IR_OUT],
        passmanager_ir: IR_OUT,
        property_set: PropertySet,
        running_time: float,
        count: int
    ) -> None:
        ...

Note that this signature differs slightly for passes and pass managers defined in the
:mod:`qiskit.transpiler` module.


Examples
========

We look into a toy optimization task, namely, preparing a row of numbers
and removing a digit if the number is five.
Such a task might be easily done by converting the input numbers into string.
We use the pass manager framework here, putting the efficiency aside for
a moment to learn how to build a custom Qiskit compiler.

.. plot::
   :include-source:
   :nofigs:
   :context:

    from qiskit.passmanager import BasePassManager, GenericPass, ConditionalController

    class ToyPassManager(BasePassManager):

        def _passmanager_frontend(self, input_program: int, **kwargs) -> str:
            return str(input_program)

        def _passmanager_backend(self, passmanager_ir: str, in_program: int, **kwargs) -> int:
            return int(passmanager_ir)

This pass manager inputs and outputs an integer number, while
performing the optimization tasks on a string data.
Hence, input, IR, output type are integer, string, integer, respectively.
The :meth:`~.BasePassManager._passmanager_frontend` method defines the conversion from the
input data to IR, and :meth:`~.BasePassManager._passmanager_backend` defines
the conversion from the IR to output data.
The pass manager backend is also given an :code:`in_program` parameter that contains the original
``input_program`` to the front end, for referencing any original metadata of the input program for
the final conversion.

Next, we implement a pass that removes a digit when the number is five.

.. plot::
   :include-source:
   :nofigs:
   :context:

    class RemoveFive(GenericPass):

        def run(self, passmanager_ir: str):
            return passmanager_ir.replace("5", "")

    task = RemoveFive()

Finally, we instantiate a pass manager and schedule the task with it.
Running the pass manager with a random row of numbers returns
new numbers that don't contain five.

.. plot::
   :include-source:
   :nofigs:
   :context:

    pm = ToyPassManager()
    pm.append(task)

    pm.run([123456789, 45654, 36785554])

Output:

.. code-block:: text

    [12346789, 464, 36784]

Now we consider the case of conditional execution.
We avoid execution of the "remove five" task when the input number is
six digits or less. Such control can be implemented by a flow controller.
We start from an analysis pass that provides the flow controller
with information about the number of digits.

.. plot::
   :include-source:
   :nofigs:
   :context:

    class CountDigits(GenericPass):

        def run(self, passmanager_ir: str):
            self.property_set["ndigits"] = len(passmanager_ir)

    analysis_task = CountDigits()

Then, we wrap the remove five task with the :class:`.ConditionalController`
that runs the stored tasks only when the condition is met.

.. plot::
   :include-source:
   :nofigs:
   :context:

    def digit_condition(property_set):
        # Return True when condition is met.
        return property_set["ndigits"] > 6

    conditional_task = ConditionalController(
        tasks=[RemoveFive()],
        condition=digit_condition,
    )

As before, we schedule these passes with the pass manager and run.

.. plot::
   :include-source:
   :nofigs:
   :context:

    pm = ToyPassManager()
    pm.append(analysis_task)
    pm.append(conditional_task)

    pm.run([123456789, 45654, 36785554])

Output:

.. code-block:: text

    [12346789, 45654, 36784]

The "remove five" task is triggered only for the first and third input
values, which have more than six digits.

With the pass manager framework, a developer can flexibly customize
the optimization task by combining multiple passes and flow controllers.
See details in the following class API documentation.


Interface
=========

Passes
------

.. autosummary::
   :toctree: ../stubs/

   GenericPass
   Task

Pass managers
-------------

.. autosummary::
   :toctree: ../stubs/

   BasePassManager
   MultiStagePassManager

Flow controllers
----------------

.. autosummary::
   :toctree: ../stubs/

   BaseController
   FlowControllerLinear
   ConditionalController
   DoWhileController

Compilation state
-----------------

.. autosummary::
   :toctree: ../stubs/

   PropertySet
   WorkflowStatus
   PassManagerState

Exceptions
----------

.. autoexception:: PassManagerError

.. _passmanager-lowering:

------------------------------
Multi-IR lowering pass manager
------------------------------

The "new" Qiskit pass manager is :class:`LoweringPassManager`.  This is newly designed, with the
intention to permit native interoperation between compiled code and Python, and to permit a
pipeline to lower through a series of intermediate representations (IRs).

From a user perspective, you will create a :class:`LoweringPassManager`, populate it with a series
of :class:`Pass` instances, each of which defines which :class:`IR` it accepts and returns, and then
simply :meth:`~LoweringPassManager.run` the pipeline.  Alternatively, you may receive a
:class:`LoweringPassManager` from a constructor function, which builds a preconfigured pipeline for
you.

.. autoclass:: LoweringPassManager

Developers will need to implement passes, and potentially introduce new IRs.  Both of these base
objects define their subclass interfaces.

.. autoclass:: IR
    :members:
    :private-members:
.. autoclass:: Pass
    :members:
    :private-members:

Implementers of :class:`Pass` get a "handle" object, which can be used to interact with the pipeline
execution context.

.. autoclass:: PassContextHandle

There are two exceptions associated with the lowering pass-manager infrastructure:

.. autoexception:: PassError
.. autoexception:: LoweringPassManagerError
"""

from .base_passmanager import BasePassManager
from .multistage_passmanager import MultiStagePassManager
from .flow_controllers import (
    FlowControllerLinear,
    ConditionalController,
    DoWhileController,
)
from .base_tasks import GenericPass, BaseController, Task
from .compilation_status import PropertySet, WorkflowStatus, PassManagerState
from .exceptions import PassManagerError
from .lowering_passmanager import *
from . import lowering_passmanager

__all__ = [
    "BaseController",
    "BasePassManager",
    "ConditionalController",
    "DoWhileController",
    "FlowControllerLinear",
    "GenericPass",
    "MultiStagePassManager",
    "PassManagerError",
    "PassManagerState",
    "PropertySet",
    "Task",
    "WorkflowStatus",
]
__all__ += lowering_passmanager.__all__
