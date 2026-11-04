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

"""Exception definitions for the OQ2 module."""

from qiskit.exceptions import QiskitError


class QASM2Error(QiskitError):
    """A general error raised by the OpenQASM 2 interoperation layer."""


class QASM2ParseError(QASM2Error):
    """An error raised because of a failure to parse an OpenQASM 2 file."""


class QASM2ExportError(QASM2Error):
    """An error raised because of a failure to convert a Qiskit object to an OpenQASM 2 form."""
