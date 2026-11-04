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
=========================================
Experiment Results (:mod:`qiskit.result`)
=========================================

.. currentmodule:: qiskit.result

Core classes
============

.. autosummary::
   :toctree: ../stubs/

   Result
   ResultError
   Counts

Marginalization
===============

.. autofunction:: marginal_counts
.. autofunction:: marginal_distribution
.. autofunction:: marginal_memory

Distributions
=============

.. autosummary::
   :toctree: ../stubs/

   ProbDistribution
   QuasiDistribution

Expectation values
==================

.. autofunction:: sampled_expectation_value

"""

from . import distributions

from .counts import Counts
from .distributions import *
from .exceptions import ResultError
from .models import MeasLevel, MeasReturnType
from .result import Result
from .sampled_expval import sampled_expectation_value
from .utils import marginal_counts
from .utils import marginal_distribution
from .utils import marginal_memory

__all__ = [
    "Counts",
    "MeasLevel",
    "MeasReturnType",
    "Result",
    "ResultError",
    "marginal_counts",
    "marginal_distribution",
    "marginal_memory",
    "sampled_expectation_value",
]
__all__ += distributions.__all__
