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

"""Primitive V1 abstract base class."""

from __future__ import annotations

from abc import ABC

from qiskit.providers import Options


class BasePrimitiveV1(ABC):
    """Primitive V1 abstract base class."""

    def __init__(self, options: dict | None = None):
        self._run_options = Options()
        if options is not None:
            self._run_options.update_options(**options)

    @property
    def options(self) -> Options:
        """Return options values for the estimator.

        Returns:
            options
        """
        return self._run_options

    def set_options(self, **fields):
        """Set options values for the estimator.

        Args:
            **fields: The fields to update the options
        """
        self._run_options.update_options(**fields)
