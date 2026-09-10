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

"""A quantum program submitted for later, and resumed from its job id."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

from qiskit.providers.jobstatus import JobStatus
from qiskit.quantum_program import ProgramStepper

from .basic_program_sampling import answer_request
from .basic_simulator import BasicSimulator
from .exceptions import BasicProviderError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.quantum_program import DataTree, QuantumProgram

#: The op a shot loop applies, which is the work this runtime performs.
_SHOT_LOOP = "qiskit.shot_loop"

#: What has been answered so far, by the id of the request it answered.
Answers = dict


def replay(
    program: QuantumProgram,
    external: Sequence[int],
    inputs: Any | None,
    answers: Answers,
) -> ProgramStepper:
    """Rebuild the stepper a run had reached, from what was submitted and what was answered.

    A stepper is not itself part of a run's stored state. It follows from the program, the inputs
    and the answers given so far, because requests are raised in a deterministic order and so an
    answer recorded against an id still finds its request. Replaying is therefore all that resuming
    needs, and a service keeps only values it has to keep anyway.

    Args:
        program: The partitioned program that was submitted.
        external: Which of its functions are the caller's to perform, by index.
        inputs: The arguments it was submitted with.
        answers: What has been answered so far, by request id.

    Returns:
        A stepper advanced as far as ``answers`` allows.
    """
    stepper = ProgramStepper(program, external, inputs)
    stepper.step()
    while stepper.outputs is None:
        known = [request.id for request in stepper.outstanding() if request.id in answers]
        if not known:
            break
        for request_id in known:
            stepper.fulfil(request_id, answers[request_id])
        stepper.step()
    return stepper


class BasicRuntimeJob:
    """One run of a quantum program, part-way through, holding everything it needs to carry on.

    A job holds the program being run and a stepper standing where the run has reached. The stepper
    is rebuilt by :func:`replay` when the run is fetched, so it is the client's rather than the
    service's, and a job needs nothing of whatever produced it in order to finish.

    Each answer a job produces is recorded against the run, so a run fetched again resumes from
    what has already been performed rather than performing it twice.

    ``BasicRuntimeJob(job_id, program, stepper, answers, simulator, generator)`` constructs one,
    though :meth:`BasicProgramRuntime.run` and :meth:`BasicProgramRuntime.job` are how one is
    obtained.

    Args:
        job_id: The id this run was submitted under.
        program: The partitioned program being run.
        stepper: Where the run has reached.
        answers: The run's record of what has been answered, which this adds to.
        simulator: The simulator to answer each request on.
        generator: The source of each simulation's seed.
    """

    def __init__(
        self,
        job_id: str,
        program: QuantumProgram,
        stepper: ProgramStepper,
        answers: Answers,
        simulator: BasicSimulator,
        generator: np.random.Generator,
    ) -> None:
        self._job_id = job_id
        self._program = program
        self._stepper = stepper
        self._answers = answers
        self._simulator = simulator
        self._generator = generator

    @property
    def job_id(self) -> str:
        """The id this run was submitted under."""
        return self._job_id

    def status(self) -> JobStatus:
        """Return how far along the run is.

        Returns:
            ``JobStatus.DONE`` once the program has finished, and ``JobStatus.RUNNING`` while work
            is still outstanding.
        """
        return JobStatus.DONE if self._stepper.outputs is not None else JobStatus.RUNNING

    def result(self) -> DataTree:
        """Return the result, blocking until completion.

        Returns:
            A data tree of arrays, arranged as the program's output structure.

        Raises:
            BasicProviderError: If a circuit cannot be simulated.
        """
        # The outstanding work is read back out of the stepper, so nothing had to be kept beside
        # it. A program of several stages comes round here once per stage.
        while (outputs := self._stepper.outputs) is None:
            for request in self._stepper.outstanding():
                answer = answer_request(request, self._simulator, self._generator)
                self._answers[request.id] = answer
                self._stepper.fulfil(request.id, answer)
            self._stepper.step()
        return outputs

    def __repr__(self) -> str:
        return f"BasicRuntimeJob({self._job_id}, {self.status().name})"


class BasicProgramRuntime:
    """Stand in for a service that runs quantum programs, holding each one under a job id.

    This is :class:`BasicProgramSimulator` split across a boundary. A run is submitted and kept, and
    fetching it by id gives back a job that can be carried to completion by itself::

        runtime = BasicProgramRuntime(seed_simulator=42)
        job_id = runtime.run(program).job_id

        # Anything holding the id can finish the run, and needs nothing else once it has the job.
        runtime.job(job_id).result()

    What is kept per run is the program, which of its functions the caller performs, the inputs, and
    what has been answered so far. A stepper is not among them: it is rebuilt by :func:`replay` on
    every fetch, which is what keeps the interpreter's own bookkeeping out of anything a real
    service would have to write down. Those four values are all a wire format would need, and the
    first three of them are the submission itself.

    ``BasicProgramRuntime(seed_simulator=None)`` constructs one. Runs are held for as long as the
    runtime is.

    Args:
        seed_simulator: The seed every simulation of a run is derived from. Runs are random when it
            is ``None``.
    """

    def __init__(self, *, seed_simulator: int | None = None) -> None:
        self._seed_simulator = seed_simulator
        self._simulator = BasicSimulator()
        # One entry per run: what was submitted, and what has been answered against it.
        self._submitted: dict[str, tuple[QuantumProgram, list[int], Any, Answers]] = {}

    @property
    def seed_simulator(self) -> int | None:
        """The seed every simulation of a run is derived from."""
        return self._seed_simulator

    def run(
        self,
        program: QuantumProgram,
        inputs: Any | None = None,
        /,
    ) -> BasicRuntimeJob:
        """Submit ``program``, and take a job over the run.

        Args:
            program: The program to run. Its entry point must call no function of its own.
            inputs: Arguments to the program, anything coercible into an array-valued
                :class:`~qiskit.quantum_program.DataTree`.

        Returns:
            A job over the submitted run.

        Raises:
            ValueError: If the program's entry point already calls a function, if it holds an
                operation that is neither a shot loop nor one Qiskit can perform, or if the inputs
                are arranged differently than it declares.
        """
        partitioned, resources = program.partition([[_SHOT_LOOP]])
        job_id = str(uuid.uuid4())
        self._submitted[job_id] = (partitioned, list(range(len(resources))), inputs, {})
        return self.job(job_id)

    def job(self, job_id: str, /) -> BasicRuntimeJob:
        """Fetch the run submitted under ``job_id``.

        Args:
            job_id: Which run to fetch.

        Returns:
            A job over that run, holding a stepper replayed to where it had reached.

        Raises:
            BasicProviderError: If no run was submitted under that id.
            ValueError: If the inputs no longer match the program, which a caller mutating them
                after submitting would cause.
        """
        if job_id not in self._submitted:
            raise BasicProviderError(f"no run was submitted under job id {job_id!r}")
        program, external, inputs, answers = self._submitted[job_id]
        return BasicRuntimeJob(
            job_id,
            program,
            replay(program, external, inputs, answers),
            answers,
            self._simulator,
            np.random.default_rng(self._seed_simulator),
        )

    def __repr__(self) -> str:
        return (
            f"BasicProgramRuntime(seed_simulator={self._seed_simulator}, "
            f"{len(self._submitted)} run(s) submitted)"
        )
