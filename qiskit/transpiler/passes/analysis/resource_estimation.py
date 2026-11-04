# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Automatically require analysis passes for resource estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.passes.analysis.depth import Depth
from qiskit.transpiler.passes.analysis.width import Width
from qiskit.transpiler.passes.analysis.size import Size
from qiskit.transpiler.passes.analysis.count_ops import CountOps
from qiskit.transpiler.passes.analysis.num_tensor_factors import NumTensorFactors
from qiskit.transpiler.passes.analysis.num_qubits import NumQubits


class ResourceEstimation(AnalysisPass):
    """Automatically require analysis passes for resource estimation.

    An analysis pass for automatically running:

    * Depth()
    * Width()
    * Size()
    * CountOps()
    * NumTensorFactors()
    * NumQubits()
    """

    def __init__(self) -> None:
        super().__init__()
        self.requires += [Depth(), Width(), Size(), CountOps(), NumTensorFactors(), NumQubits()]

    def run(self, dag: DAGCircuit) -> None:
        """Run the ResourceEstimation pass on ``dag``."""
