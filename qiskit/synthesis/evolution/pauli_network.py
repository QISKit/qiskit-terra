# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Circuit synthesis for pauli evolution gates."""

from __future__ import annotations

from qiskit.circuit import QuantumCircuit

from qiskit._accelerate.synthesis.evolution import (
    pauli_network_synthesis as pauli_network_synthesis_inner,
    pauli_network_mcts as pauli_network_mcts_inner,
)


def synth_pauli_network_rustiq(
    num_qubits: int,
    pauli_network: list,
    optimize_count: bool = True,
    preserve_order: bool = True,
    upto_clifford: bool = False,
    upto_phase: bool = False,
    resynth_clifford_method: int = 0,
) -> QuantumCircuit:
    """
    Calls Rustiq's Pauli network synthesis algorithm.

    The algorithm is described in [1]. The source code (in Rust) is available at
    https://github.com/qiskit-community/rustiq-core.

    Args:
        num_qubits: the number of qubits over which the Pauli network is defined.
        pauli_network: a list of Pauli rotations, represented in sparse format: a list of
            triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
        optimize_count: if `True` the synthesis algorithm will try to optimize the 2-qubit
            gate count; and if `False` then the 2-qubit depth.
        preserve_order: whether the order of Paulis should be preserved, up to
            commutativity. If the order is not preserved, the returned circuit will
            generally not be equivalent to the given Pauli network.
        upto_clifford: if `True`, the final Clifford operator is not synthesized
            and the returned circuit will generally not be equivalent to the given
            Pauli network. In addition, the argument `upto_phase` would be ignored.
        upto_phase: if `True`, the global phase of the returned circuit may differ
             from the global phase of the given Pauli network. The argument is ignored
             when `upto_clifford` is `True`.
        resynth_clifford_method: describes the strategy to synthesize the final Clifford
            operator. If `0` a naive approach is used, which doubles the number of gates
            but preserves the global phase of the circuit. If `1`, the Clifford is
            resynthesized using Qiskit's greedy Clifford synthesis algorithm. If `2`, it
            is resynthesized by Rustiq itself. If `upto_phase` is `False`, the naive
            approach is used, as neither synthesis method preserves the global phase.

    If `preserve_order` is `true` and both `upto_clifford` and `upto_phase` are `false`,
    the returned circuit is equivalent to the given pauli network.

    Returns:
        A circuit implementation of the Pauli network.

    References:
        1. Timothée Goubault de Brugière and Simon Martiel,
           *Faster and shorter synthesis of Hamiltonian simulation circuits*,
           `arXiv:2404.03280 [quant-ph] <https://arxiv.org/abs/2404.03280>`_

    """
    out = pauli_network_synthesis_inner(
        num_qubits=num_qubits,
        pauli_network=pauli_network,
        optimize_count=optimize_count,
        preserve_order=preserve_order,
        upto_clifford=upto_clifford,
        upto_phase=upto_phase,
        resynth_clifford_method=resynth_clifford_method,
    )
    circuit = QuantumCircuit._from_circuit_data(out, legacy_qubits=True)
    return circuit


def synth_pauli_network_mcts(
    num_qubits: int,
    pauli_network: list,
    preserve_order: bool = True,
    upto_clifford: bool = False,
    upto_phase: bool = False,
    num_simulations: int = 1,
    max_parallel_simulations: int | None = None,
) -> QuantumCircuit:
    """
    Synthesize a Pauli network using Monte Carlo Tree Search (MCTS).

    The algorithm is described in [1].

    Args:
        num_qubits: the number of qubits over which the Pauli network is defined.
        pauli_network: a list of Pauli rotations, represented in sparse format: a list of
            triples such as `[("XX", [0, 3], theta), ("ZZ", [0, 1], 0.1)]`.
        preserve_order: whether the order of Paulis should be preserved, up to
            commutativity. If the order is not preserved, the returned circuit will
            generally not be equivalent to the given Pauli network.
        upto_clifford: if `True`, the final Clifford operator is not synthesized
            and the returned circuit will generally not be equivalent to the given
            Pauli network. In addition, the argument `upto_phase` would be ignored.
        upto_phase: if `True`, the global phase of the returned circuit may differ
             from the global phase of the given Pauli network. The argument is ignored
             when `upto_clifford` is `True`.
        num_simulations : Number of Monte Carlo simulations to perform. This value
            must be at least 1.
        max_parallel_simulations: Maximum number of simulations that can be
            performed in parallel. If integer, this value must be at least 1. The value
            of `None` means "unlimited".

    Returns:
        A circuit implementation of the Pauli network.

    References:
        1. Mulundano Machiya, Matt Menickelly, Paul Hovland, Ji Liu,
           *MonteQ: A Monte Carlo Tree Search Based Quantum Circuit Synthesis Framework*,
           `arXiv:2604.19029 <https://arxiv.org/abs/2604.19029>`_

    """
    out = pauli_network_mcts_inner(
        num_qubits=num_qubits,
        pauli_network=pauli_network,
        preserve_order=preserve_order,
        upto_clifford=upto_clifford,
        upto_phase=upto_phase,
        num_simulations=num_simulations,
        max_parallel_simulations=max_parallel_simulations,
    )
    circuit = QuantumCircuit._from_circuit_data(out, legacy_qubits=True)
    return circuit
