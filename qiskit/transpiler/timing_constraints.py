# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Timing Constraints class."""

from qiskit.transpiler.exceptions import TranspilerError


class TimingConstraints:
    """Hardware Instruction Timing Constraints."""

    def __init__(
        self,
        granularity: int = 1,
        min_length: int = 1,
        pulse_alignment: int = 1,
        acquire_alignment: int = 1,
    ):
        """Initialize a TimingConstraints object

        Args:
            granularity: An integer value representing minimum pulse gate
                resolution in units of ``dt``. A user-defined pulse gate should have
                duration of a multiple of this granularity value.
            min_length: An integer value representing minimum pulse gate
                length in units of ``dt``. A user-defined pulse gate should be longer
                than this length.
            pulse_alignment: An integer value representing a time resolution of gate
                instruction starting time. Gate instruction should start at time which
                is a multiple of the alignment value.
            acquire_alignment: An integer value representing a time resolution of measure
                instruction starting time. Measure instruction should start at time which
                is a multiple of the alignment value.

        Notes:
            This information will be provided by the backend configuration.

        Raises:
            TranspilerError: When any invalid constraint value is passed.
        """
        self.granularity = granularity
        self.min_length = min_length
        self.pulse_alignment = pulse_alignment
        self.acquire_alignment = acquire_alignment

        for key, value in self.__dict__.items():
            if not isinstance(value, int) or value < 1:
                raise TranspilerError(
                    f"Timing constraint {key} should be nonzero integer. Not {value}."
                )
