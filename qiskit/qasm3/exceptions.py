# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exceptions that may be raised during processing OpenQASM 3."""

from qiskit.exceptions import QiskitError


class QASM3Error(QiskitError):
    """An error raised while working with OpenQASM 3 representations of circuits."""


class QASM3ExporterError(QASM3Error):
    """An error raised during running the OpenQASM 3 exporter."""


class QASM3ImporterError(QASM3Error):
    """An error raised during the OpenQASM 3 importer."""
