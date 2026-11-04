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

"""Utility functions shared between permutation functionality."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar
from collections.abc import MutableMapping

from qiskit.circuit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import SwapGate

from .types import Swap, PermutationCircuit

_K = TypeVar("_K")
_V = TypeVar("_V")


def swap_permutation(
    swaps: Iterable[Iterable[Swap[_K]]],
    mapping: MutableMapping[_K, _V],
    allow_missing_keys: bool = False,
) -> None:
    """Given a circuit of swaps, apply them to the permutation (in-place).

    Args:
      swaps: param mapping: A mapping of Keys to Values, where the Keys are being swapped.
      mapping: The permutation to have swaps applied to.
      allow_missing_keys: Whether to allow swaps of missing keys in mapping.
    """
    for swap_step in swaps:
        for sw1, sw2 in swap_step:
            # Take into account non-existent keys.
            val1: _V | None = None
            val2: _V | None = None
            if allow_missing_keys:
                val1 = mapping.pop(sw1, None)
                val2 = mapping.pop(sw2, None)
            else:
                # Asserts that both keys exist
                val1, val2 = mapping.pop(sw1), mapping.pop(sw2)

            if val1 is not None:
                mapping[sw2] = val1
            if val2 is not None:
                mapping[sw1] = val2


def permutation_circuit(swaps: Iterable[list[Swap[_V]]]) -> PermutationCircuit:
    """Produce a circuit description of a list of swaps.
        With a given permutation and permuter you can compute the swaps using the permuter function
        then feed it into this circuit function to obtain a circuit description.
    Args:
      swaps: An iterable of swaps to perform.
    Returns:
      A MappingCircuit with the circuit and a mapping of node to qubit in the circuit.
    """
    # Construct a circuit with each unique node id becoming a quantum register of size 1.
    dag = DAGCircuit()
    swap_list = list(swaps)

    # Set of unique nodes used in the swaps.
    nodes = {
        swap_node for swap_step in swap_list for swap_nodes in swap_step for swap_node in swap_nodes
    }

    node_qargs = {node: QuantumRegister(1) for node in nodes}
    for qubit in node_qargs.values():
        dag.add_qreg(qubit)

    inputmap = {node: q[0] for node, q in node_qargs.items()}

    # Apply swaps to the circuit.
    for swap_step in swap_list:
        for swap0, swap1 in swap_step:
            dag.apply_operation_back(SwapGate(), (inputmap[swap0], inputmap[swap1]), check=False)

    return PermutationCircuit(dag, inputmap)
