# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
================================================================================
BasicProvider: Python-based Simulators (:mod:`qiskit.providers.basic_provider`)
================================================================================

.. currentmodule:: qiskit.providers.basic_provider

A module of Python-based quantum simulators. Simulators can be accessed
via the `BasicProvider` provider, e.g.:

.. plot::
   :include-source:
   :nofigs:

   from qiskit.providers.basic_provider import BasicProvider

   backend = BasicProvider().get_backend('basic_simulator')


Running a quantum program
=========================

A :class:`~qiskit.quantum_program.QuantumProgram` describes a shot loop together with the arithmetic
around it. :class:`BasicProgramSimulator` runs one: each shot loop is simulated on a state vector,
and Qiskit evaluates the rest itself.

.. plot::
   :include-source:
   :nofigs:

   from qiskit.circuit import QuantumCircuit
   from qiskit.providers.basic_provider import BasicProgramSimulator
   from qiskit.quantum_program import build, shot_loop

   circuit = QuantumCircuit(2)
   circuit.h(0)
   circuit.measure_all()

   outcomes = shot_loop([circuit], shots=1024)
   program = build({"excited": outcomes[0]["meas"].mean(axis=0)})
   BasicProgramSimulator(seed_simulator=42).run(program).result()

:class:`BasicProgramRuntime` does the same in stages, holding each run under a job id so that it can
be carried on by anything that has the id. That is the shape a backend submitting work elsewhere
takes.


Classes
=======

.. autosummary::
   :toctree: ../stubs/

   BasicSimulator
   BasicProvider
   BasicProviderJob
   BasicProviderError
   BasicProgramSimulator
   BasicProgramJob
   BasicProgramRuntime
   BasicRuntimeJob

Functions
=========

.. autosummary::
   :toctree: ../stubs/

   replay
"""

from .basic_program_runtime import BasicProgramRuntime, BasicRuntimeJob, replay
from .basic_program_simulator import BasicProgramJob, BasicProgramSimulator
from .basic_provider import BasicProvider
from .basic_provider_job import BasicProviderJob
from .basic_simulator import BasicSimulator
from .exceptions import BasicProviderError

__all__ = [
    "BasicProgramJob",
    "BasicProgramRuntime",
    "BasicProgramSimulator",
    "BasicProvider",
    "BasicProviderError",
    "BasicProviderJob",
    "BasicRuntimeJob",
    "BasicSimulator",
    "replay",
]
