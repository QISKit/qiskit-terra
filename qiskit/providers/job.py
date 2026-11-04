# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Job abstract interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from collections.abc import Callable

from qiskit.exceptions import QiskitError
from qiskit.providers.backend import Backend
from qiskit.providers.exceptions import JobTimeoutError
from qiskit.providers.jobstatus import JOB_FINAL_STATES, JobStatus

if TYPE_CHECKING:
    from qiskit.result import Result


class Job:
    """Base common type for all versioned Job abstract classes.

    Note this class should not be inherited from directly, it is intended
    to be used for type checking. When implementing a provider you should use
    the versioned abstract classes as the parent class and not this class
    directly.
    """

    version = 0


class JobV1(Job, ABC):
    """Class to handle jobs

    This first version of the Job abstract class is written to be mostly
    backwards compatible with the legacy providers interface. This was done to ease
    the transition for users and provider maintainers to the new versioned providers. Expect,
    future versions of this abstract class to change the data model and
    interface.
    """

    version = 1
    _async = True

    def __init__(self, backend: Backend | None, job_id: str, **kwargs) -> None:
        """Initializes the asynchronous job.

        Args:
            backend: the backend used to run the job.
            job_id: a unique id in the context of the backend used to run
                the job.
            kwargs: Any key value metadata to associate with this job.
        """
        self._job_id = job_id
        self._backend = backend
        self.metadata = kwargs

    def job_id(self) -> str:
        """Return a unique id identifying the job."""
        return self._job_id

    def backend(self) -> Backend:
        """Return the backend where this job was executed."""
        if self._backend is None:
            raise QiskitError("The job does not have any backend.")
        return self._backend

    def done(self) -> bool:
        """Return whether the job has successfully run."""
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        """Return whether the job is actively running."""
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        """Return whether the job has been cancelled."""
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        """Return whether the job is in a final job state such as ``DONE`` or ``ERROR``."""
        return self.status() in JOB_FINAL_STATES

    def wait_for_final_state(
        self, timeout: float | None = None, wait: float = 5, callback: Callable | None = None
    ) -> None:
        """Poll the job status until it progresses to a final state such as ``DONE`` or ``ERROR``.

        Args:
            timeout: Seconds to wait for the job. If ``None``, wait indefinitely.
            wait: Seconds between queries.
            callback: Callback function invoked after each query.
                The following positional arguments are provided to the callback function:

                * job_id: Job ID
                * job_status: Status of the job from the last query
                * job: This JobV1 instance

                Note: different subclass might provide different arguments to
                the callback function.

        Raises:
            JobTimeoutError: If the job does not reach a final state before the
                specified timeout.
        """
        if not self._async:
            return
        start_time = time.time()
        status = self.status()
        while status not in JOB_FINAL_STATES:
            elapsed_time = time.time() - start_time
            if timeout is not None and elapsed_time >= timeout:
                raise JobTimeoutError(f"Timeout while waiting for job {self.job_id()}.")
            if callback:
                callback(self.job_id(), status, self)
            time.sleep(wait)
            status = self.status()
        return

    @abstractmethod
    def submit(self):
        """Submit the job to the backend for execution."""

    @abstractmethod
    def result(self) -> Result:
        """Return the results of the job."""

    def cancel(self):
        """Attempt to cancel the job."""
        raise NotImplementedError

    @abstractmethod
    def status(self) -> JobStatus:
        """Return the status of the job, among the values of ``JobStatus``."""
