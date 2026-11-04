# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Job for the reference implementations of Primitives V1 and V2.
"""

import uuid
from concurrent.futures import ThreadPoolExecutor

from qiskit.providers import JobError, JobStatus
from qiskit.providers.jobstatus import JOB_FINAL_STATES

from .base.base_primitive_job import BasePrimitiveJob, ResultT


class PrimitiveJob(BasePrimitiveJob[ResultT, JobStatus]):
    """Handle to a job from the reference implementations of the primitives in Qiskit.

    This is a concrete implementation of the :class:`.BasePrimitiveJob` interface.  See the
    documentation of that class for a discussion of the interface.

    Primitives implementers looking to create their own job classes should not subclass this, but
    instead subclass the interface definition :class:`.BasePrimitiveJob`.
    """

    def __init__(self, function, *args, **kwargs):
        """
        Args:
            function: A callable function to execute the job.
            args: any additional positional arguments
            kwargs: any additional keyword arguments
        """
        super().__init__(str(uuid.uuid4()))
        self._future = None
        self._function = function
        self._result = None
        self._status = None
        self._args = args
        self._kwargs = kwargs

    def _submit(self):
        if self._future is not None:
            raise JobError("Primitive job has been submitted already.")

        executor = ThreadPoolExecutor(max_workers=1)
        self._future = executor.submit(self._function, *self._args, **self._kwargs)
        executor.shutdown(wait=False)

    def __getstate__(self):
        _ = self.result()
        _ = self.status()
        state = self.__dict__.copy()
        state["_future"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._future = None

    def result(self) -> ResultT:
        if self._result is None:
            self._check_submitted()
            self._result = self._future.result()
        return self._result

    def status(self) -> JobStatus:
        if self._status is None:
            self._check_submitted()
            if self._future.running():
                # we should not store status running because it is not completed
                return JobStatus.RUNNING
            elif self._future.cancelled():
                self._status = JobStatus.CANCELLED
            elif self._future.done() and self._future.exception() is None:
                self._status = JobStatus.DONE
            else:
                self._status = JobStatus.ERROR
        return self._status

    def _check_submitted(self):
        if self._future is None:
            raise JobError("Primitive Job has not been submitted yet.")

    def cancel(self):
        self._check_submitted()
        return self._future.cancel()

    def done(self) -> bool:
        return self.status() == JobStatus.DONE

    def running(self) -> bool:
        return self.status() == JobStatus.RUNNING

    def cancelled(self) -> bool:
        return self.status() == JobStatus.CANCELLED

    def in_final_state(self) -> bool:
        return self.status() in JOB_FINAL_STATES
