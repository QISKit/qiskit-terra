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

"""Utility functions for routing"""

from qiskit.transpiler.exceptions import TranspilerError
from .algorithms import ApproximateTokenSwapper


def get_swap_map_dag(dag, coupling_map, from_layout, to_layout, seed, trials=4):
    """Get the circuit of swaps to go from from_layout to to_layout, and the physical qubits
    (integers) that the swap circuit should be applied on."""
    if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
        raise TranspilerError("layout transformation runs on physical circuits only")
    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError("The layout does not match the amount of qubits in the DAG")
    token_swapper = ApproximateTokenSwapper(coupling_map.graph.to_undirected(), seed)
    # Find the permutation between the initial physical qubits and final physical qubits.
    permutation = {
        pqubit: to_layout[vqubit] for vqubit, pqubit in from_layout.get_virtual_bits().items()
    }
    # The mapping produced here maps physical qubit indices of the outer dag to the bits used to
    # represent them in the inner map.  For later composing, we actually want the opposite map.
    swap_circuit, phys_to_circuit_qubits = token_swapper.permutation_circuit(permutation, trials)
    circuit_to_phys = {inner: outer for outer, inner in phys_to_circuit_qubits.items()}
    return swap_circuit, [circuit_to_phys[bit] for bit in swap_circuit.qubits]
