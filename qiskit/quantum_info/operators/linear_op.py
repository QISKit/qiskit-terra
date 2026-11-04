# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract LinearOperator class.
"""

from __future__ import annotations
from abc import ABC

from .base_operator import BaseOperator
from .mixins import LinearMixin, AdjointMixin, TolerancesMixin


class LinearOp(BaseOperator, AdjointMixin, LinearMixin, TolerancesMixin, ABC):
    """Abstract linear operator base class."""
