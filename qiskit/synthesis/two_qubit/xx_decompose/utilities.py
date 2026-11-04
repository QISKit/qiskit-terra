# This code is part of Qiskit.
#
# (C) Copyright IBM 2021
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Depository for generic utility snippets.
"""

from __future__ import annotations
import warnings

import numpy as np


EPSILON = 1e-6  # Fraction(1, 1_000_000)


# TODO: THIS IS A STOPGAP!!!
def safe_arccos(numerator, denominator):
    """
    Computes arccos(n/d) with different (better?) numerical stability.
    """
    threshold = 0.005

    if abs(numerator) > abs(denominator) and abs(numerator - denominator) < threshold:
        return 0.0
    elif abs(numerator) > abs(denominator) and abs(numerator + denominator) < threshold:
        return np.pi
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return np.arccos(numerator / denominator)
