# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Validation and optimization for hardware instruction alignment constraints.

This is a control electronics aware analysis pass group.

In many quantum computing architectures gates (instructions) are implemented with
shaped analog stimulus signals. These signals are digitally stored in the
waveform memory of the control electronics and converted into analog voltage signals
by electronic components called digital to analog converters (DAC).

In a typical hardware implementation of superconducting quantum processors,
a single qubit instruction is implemented by a
microwave signal with the duration of around several tens of ns with a per-sample
time resolution of ~0.1-10ns, as reported by ``backend.configuration().dt``.
In such systems requiring higher DAC bandwidth, control electronics often
defines a `pulse granularity`, in other words a data chunk, to allow the DAC to
perform the signal conversion in parallel to gain the bandwidth.

A control electronics, i.e. micro-architecture, of the real quantum backend may
impose some constraints on the start time of microinstructions.
In Qiskit SDK, the duration of :class:`qiskit.circuit.Delay` can take arbitrary
value in units of dt, thus circuits involving delays may violate the constraints,
which may result in failure in the circuit execution on the backend.

There are two alignment constraint values reported by your quantum backend.
In addition, if you want to define a custom instruction as a pulse gate, i.e. calibration,
the underlying pulse instruction should satisfy other two waveform constraints.

Pulse alignment constraint

    This value is reported by ``timing_constraints["pulse_alignment"]`` in the backend
    configuration in units of dt. The start time of all pulse instructions should be
    multiple of this value. Violation of this constraint may result in the
    backend execution failure.

    In most of the scenarios, the scheduled start time of ``DAGOpNode`` corresponds to the
    start time of the underlying pulse instruction composing the node operation.
    However, this assumption can be intentionally broken by defining a pulse gate,
    i.e. calibration, with the schedule involving pre-buffer, i.e. some random pulse delay
    followed by a pulse instruction. Because this pass is not aware of such edge case,
    the user must take special care of pulse gates if any.

Acquire alignment constraint

    This value is reported by ``timing_constraints["acquire_alignment"]`` in the backend
    configuration in units of dt. The start time of the :class:`~qiskit.circuit.Measure`
    instruction should be multiple of this value.

Granularity constraint

    This value is reported by ``timing_constraints["granularity"]`` in the backend
    configuration in units of dt. This is the constraint for a single pulse :class:`Play`
    instruction that may constitute your pulse gate.
    The length of waveform samples should be multiple of this constraint value.
    Violation of this constraint may result in failure in backend execution.

Minimum pulse length constraint

    This value is reported by ``timing_constraints["min_length"]`` in the backend
    configuration in units of dt. This is the constraint for a single pulse :class:`Play`
    instruction that may constitute your pulse gate.
    The length of waveform samples should be greater than this constraint value.
    Violation of this constraint may result in failure in backend execution.

"""

from .check_durations import InstructionDurationCheck
from .reschedule import ConstrainedReschedule

__all__ = ["ConstrainedReschedule", "InstructionDurationCheck"]
