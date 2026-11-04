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

"""Module containing circuit analysis passes."""

from .resource_estimation import ResourceEstimation
from .depth import Depth
from .width import Width
from .size import Size
from .count_ops import CountOps
from .count_ops_longest_path import CountOpsLongestPath
from .num_tensor_factors import NumTensorFactors
from .num_qubits import NumQubits
from .dag_longest_path import DAGLongestPath

__all__ = [
    "CountOps",
    "CountOpsLongestPath",
    "DAGLongestPath",
    "Depth",
    "NumQubits",
    "NumTensorFactors",
    "ResourceEstimation",
    "Size",
    "Width",
]
