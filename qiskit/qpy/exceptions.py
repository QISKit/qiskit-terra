# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Exception for errors raised by the QPY module."""

from qiskit.exceptions import QiskitError, QiskitWarning


class QpyError(QiskitError):
    """Errors raised by the qpy module."""

    def __init__(self, *message):
        """Set the error message."""
        super().__init__(*message)
        self.message = " ".join(message)

    def __str__(self):
        """Return the message."""
        return repr(self.message)


class UnsupportedFeatureForVersion(QpyError):
    """QPY error raised when the target dump version is too low for a feature that is present in the
    object to be serialized."""

    def __init__(self, feature: str, required: int, target: int):
        """
        Args:
            feature: a description of the problematic feature.
            required: the minimum version of QPY that would be required to represent this
                feature.
            target: the version of QPY that is being used in the serialization.
        """
        self.feature = feature
        self.required = required
        self.target = target
        super().__init__(
            f"Dumping QPY version {target}, but version {required} is required for: {feature}."
        )


class QPYLoadingDeprecatedFeatureWarning(QiskitWarning):
    """Visible deprecation warning for QPY loading functions without
    a stable point in the call stack."""
