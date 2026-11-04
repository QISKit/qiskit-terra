# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Exception for errors raised by the transpiler.
"""
from qiskit.exceptions import QiskitError
from qiskit.passmanager.exceptions import PassManagerError


class TranspilerAccessError(PassManagerError):
    """DEPRECATED: Exception of access error in the transpiler passes."""


class TranspilerError(TranspilerAccessError):
    """Exceptions raised during transpilation."""


class CouplingError(QiskitError):
    """Base class for errors raised by the coupling graph object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = " ".join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)


class LayoutError(QiskitError):
    """Errors raised by the layout object."""

    def __init__(self, *msg):
        """Set the error message."""
        super().__init__(*msg)
        self.msg = " ".join(msg)

    def __str__(self):
        """Return the message."""
        return repr(self.msg)


class CircuitTooWideForTarget(TranspilerError):
    """Error raised if the circuit is too wide for the target."""


class InvalidLayoutError(TranspilerError):
    """Error raised when a user provided layout is invalid."""
