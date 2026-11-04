# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities for handling duration of a circuit instruction.
"""
import warnings

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.utils.units import apply_prefix


def duration_in_dt(duration_in_sec: float, dt_in_sec: float) -> int:
    """
    Return duration in dt.

    Args:
        duration_in_sec: duration [s] to be converted.
        dt_in_sec: duration of dt in seconds used for conversion.

    Returns:
        Duration in dt.
    """
    res = round(duration_in_sec / dt_in_sec)
    rounding_error = abs(duration_in_sec - res * dt_in_sec)
    if rounding_error > 1e-15:
        warnings.warn(
            f"Duration is rounded to {res:d} [dt] = {res * dt_in_sec:e} [s] "
            f"from {duration_in_sec:e} [s]",
            UserWarning,
        )
    return res


def convert_durations_to_dt(qc: QuantumCircuit, dt_in_sec: float, inplace=True):
    """Convert all the durations in SI (seconds) into those in dt.

    Returns a new circuit if `inplace=False`.

    Args:
        qc (QuantumCircuit): Duration of dt in seconds used for conversion.
        dt_in_sec (float): Duration of dt in seconds used for conversion.
        inplace (bool): All durations are converted inplace or return new circuit.

    Returns:
        QuantumCircuit: Converted circuit if `inplace = False`, otherwise None.

    Raises:
        CircuitError: if fail to convert durations.
    """
    if inplace:
        circ = qc
    else:
        circ = qc.copy()

    if circ.duration is not None and circ.unit != "dt":
        if not circ.unit.endswith("s"):
            raise CircuitError(f"Invalid time unit: '{circ.unit}'")

        duration = circ.duration
        if circ.unit != "s":
            duration = apply_prefix(duration, circ.unit)

        circ.duration = duration_in_dt(duration, dt_in_sec)
        circ.unit = "dt"

    if not inplace:
        return circ
    else:
        return None
