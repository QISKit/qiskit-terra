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
Abstract base classes for primitives module.
"""

from .base_sampler import BaseSamplerV1, BaseSamplerV2
from .base_estimator import BaseEstimatorV1, BaseEstimatorV2
from .estimator_result_v1 import EstimatorResult
from .sampler_result_v1 import SamplerResult

__all__ = [
    "BaseEstimatorV1",
    "BaseEstimatorV2",
    "BaseSamplerV1",
    "BaseSamplerV2",
    "EstimatorResult",
    "SamplerResult",
]
