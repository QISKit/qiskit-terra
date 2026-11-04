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

"""This module defines an enumerated type for the state of backend jobs"""

import enum


class JobStatus(enum.Enum):
    """Class for job status enumerated type."""

    INITIALIZING = "job is being initialized"
    QUEUED = "job is queued"
    VALIDATING = "job is being validated"
    RUNNING = "job is actively running"
    CANCELLED = "job has been cancelled"
    DONE = "job has successfully run"
    ERROR = "job incurred error"


JOB_FINAL_STATES = (JobStatus.DONE, JobStatus.CANCELLED, JobStatus.ERROR)
