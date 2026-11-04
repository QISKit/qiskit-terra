# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Enumeration of optimization metrics.
"""

import enum


class OptimizationMetric(enum.Enum):
    """Optimization metric considered during transpilation.

    The metric :data:`COUNT_2Q` targets optimizing the two-qubit gate count of
    the output circuit.  This is generally the preferred choice for
    near-term execution.

    The metric :data:`COUNT_T` targets optimizing the T-count of the output circuit
    when the circuit is transpiled into the Clifford+T basis set.
    """

    COUNT_2Q = enum.auto()
    """The transpilation is optimized towards minimizing the 2q-count."""
    COUNT_T = enum.auto()
    """The transpilation is optimized towards the T-count."""
