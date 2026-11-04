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

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Type definitions used within the permutation package."""

from typing import TypeVar, NamedTuple

from qiskit.circuit import Qubit
from qiskit.dagcircuit import DAGCircuit

PermuteElement = TypeVar("PermuteElement")
Permutation = dict[PermuteElement, PermuteElement]
Swap = tuple[PermuteElement, PermuteElement]


# Represents a circuit for permuting to a mapping.
class PermutationCircuit(NamedTuple):
    circuit: DAGCircuit
    inputmap: dict[int | Qubit, Qubit]  # A mapping from architecture nodes to circuit registers.
