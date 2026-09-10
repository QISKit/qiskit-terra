# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A quantum program run in process, its shot loops simulated on a state vector."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from qiskit.providers.jobstatus import JobStatus
from qiskit.quantum_program import ProgramStepper

from .basic_program_sampling import answer_request
from .basic_simulator import BasicSimulator

if TYPE_CHECKING:
    from qiskit.quantum_program import DataTree, QuantumProgram, Tracer

#: The op a shot loop applies, which is the work this simulator performs.
_SHOT_LOOP = "qiskit.shot_loop"


class BasicProgramJob:
    """One finished run of a quantum program.

    The program is evaluated before the job is made, so its outputs are already in hand and
    :meth:`result` returns immediately.

    ``BasicProgramJob(outputs)`` constructs one.

    Args:
        outputs: The program's outputs, arranged as its output structure.
    """

    def __init__(self, outputs: DataTree) -> None:
        self._outputs = outputs
        self._job_id = str(uuid.uuid4())

    @property
    def job_id(self) -> str:
        """This job's identity."""
        return self._job_id

    def result(self) -> DataTree:
        """Return the program's outputs.

        Returns:
            A data tree of arrays, arranged as the program's output structure.
        """
        return self._outputs

    def status(self) -> JobStatus:
        """Return how far along the run is.

        Returns:
            :attr:`~qiskit.providers.JobStatus.DONE`, since a job exists only once its program
            has finished.
        """
        return JobStatus.DONE

    def __repr__(self) -> str:
        return f"BasicProgramJob({self._job_id}, {self.status().name})"


class BasicProgramSimulator:
    """Run a quantum program in process, simulating each shot loop on a state vector.

    A program's shot loops are the work Qiskit cannot perform as ordinary arithmetic, so each is put
    into a function of its own and answered by :class:`.BasicSimulator`: one simulation per
    parameter set of each circuit. Everything else the program describes, such as averaging over the
    shots axis, Qiskit evaluates itself::

        from qiskit.circuit import QuantumCircuit
        from qiskit.providers.basic_provider import BasicProgramSimulator
        from qiskit.quantum_program import build, shot_loop

        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.measure_all()

        outcomes = shot_loop([circuit], shots=1024)
        program = build({"excited": outcomes[0]["meas"].mean(axis=0)})
        BasicProgramSimulator(seed_simulator=42).run(program).result()

    ``BasicProgramSimulator(seed_simulator=None)`` constructs one. A simulator is reusable, and each
    run is seeded afresh, so two runs of one program with the same seed agree.

    Args:
        seed_simulator: The seed every simulation of a run is derived from. Runs are random when it
            is ``None``.
    """

    def __init__(self, *, seed_simulator: int | None = None) -> None:
        self._seed_simulator = seed_simulator
        self._simulator = BasicSimulator()

    @property
    def seed_simulator(self) -> int | None:
        """The seed every simulation of a run is derived from."""
        return self._seed_simulator

    def run(
        self, program: QuantumProgram | Tracer, inputs: Any | None = None, /
    ) -> BasicProgramJob:
        """Evaluate ``program`` on ``inputs``, simulating the shot loops it holds.

        Args:
            program: The program to run. Its entry point must call no function of its own.
            inputs: Arguments to the program, anything coercible into an array-valued
                :class:`~qiskit.quantum_program.DataTree`.

        Returns:
            A finished :class:`BasicProgramJob`, whose :meth:`~BasicProgramJob.result` is a data
            tree of arrays.

        Raises:
            ValueError: If the program's entry point already calls a function, if it holds an
                operation that is neither a shot loop nor one Qiskit can perform, or if the inputs
                are arranged differently than it declares.
            BasicProviderError: If a circuit cannot be simulated.
        """
        from qiskit.quantum_program import Tracer, build

        if isinstance(program, Tracer):
            program = build(program)

        # One execution resource, so every function the table describes is a shot loop to perform.
        partitioned, resources = program.partition([[_SHOT_LOOP]])
        stepper = ProgramStepper(partitioned, range(len(resources)), inputs)
        generator = np.random.default_rng(self._seed_simulator)

        while (outputs := stepper.outputs) is None:
            stepper.step()
            for request in stepper.outstanding():
                answer = answer_request(request, self._simulator, generator)
                stepper.fulfil(request.id, answer)
        return BasicProgramJob(outputs)

    def __repr__(self) -> str:
        return f"BasicProgramSimulator(seed_simulator={self._seed_simulator})"
