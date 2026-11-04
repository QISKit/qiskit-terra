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
"""
Estimator V1 result class
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .base_result_v1 import _BasePrimitiveResultV1

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class EstimatorResult(_BasePrimitiveResultV1):
    """Result of Estimator V1.

    .. code-block:: python

        result = estimator.run(circuits, observables, params).result()

    where the i-th elements of ``result`` correspond to the circuit and observable given by
    ``circuits[i]``, ``observables[i]``, and the parameter values bounds by ``params[i]``.
    For example, ``results.values[i]`` gives the expectation value, and ``result.metadata[i]``
    is a metadata dictionary for this circuit and parameters.

    Args:
        values (np.ndarray): The array of the expectation values.
        metadata (list[dict]): List of the metadata.
    """

    values: np.ndarray[Any, np.dtype[np.float64]]
    metadata: list[dict[str, Any]]
