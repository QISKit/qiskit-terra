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

"""Compute the sum of two qubit registers without any ancillary qubits."""

from __future__ import annotations
from math import ceil

from qiskit.circuit import QuantumRegister, QuantumCircuit


def _mcx_ladder(n_mcx: int, alpha: int):
    r"""Implements a log-depth MCX ladder circuit as outlined in Algorithm 2 of [1]. The circuit
    relies on log-depth decomposition of MCX gate that uses conditionally clean ancillae of [2].
    Selecting :math:`\alpha=1` creates a CX ladder as shown in Fig. 2 of [1] and selecting
    :math:`\alpha=2` creates a Toffoli ladder as shown in Fig. 3 of [1].

    Args:
        n_mcx: Number of MCX gates in the ladder.
        alpha: Number of controls per MCX gate.

    Returns:
        QuantumCircuit: The MCX ladder circuit.

    References:
        1. Remaud and Vandaele, Ancilla-free Quantum Adder with Sublinear Depth, 2025.
        `arXiv:2501.16802 <https://arxiv.org/abs/2501.16802>`__

        2. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def helper(qubit_indices, alphas) -> list[list[int]]:
        k = len(alphas) + 1
        if k == 1:
            return []
        if k == 2:
            return [qubit_indices[: alphas[0] + 1]]

        x_bar = [qubit_indices[alphas[0]]]
        alpha_bar = []
        right_pairs = [qubit_indices[0 : alphas[0] + 1]]
        left_pairs = [qubit_indices[alphas[k - 3] : alphas[-1] + 1]]

        for i in range(1, ceil(k / 2) - 1):
            left_pairs += [qubit_indices[alphas[2 * i - 2] : alphas[2 * i - 1] + 1]]
            right_pairs += [qubit_indices[alphas[2 * i - 1] : alphas[2 * i] + 1]]
            x_bar += (
                qubit_indices[alphas[2 * i - 2] + 1 : alphas[2 * i - 1]]
                + qubit_indices[alphas[2 * i - 1] + 1 : alphas[2 * i] + 1]
            )
            alpha_bar.append(alphas[2 * i] - alphas[0] - i)

        if k % 2 == 0:
            x_bar += qubit_indices[alphas[k - 4] + 1 : alphas[k - 3] + 1]
            alpha_bar.append(alphas[k - 3] - alphas[0] - int(k / 2) + 2)

        return left_pairs + helper(x_bar, alpha_bar) + right_pairs

    n = n_mcx * alpha + 1
    qc = QuantumCircuit(n)
    qubit_indices, alphas = list(range(n)), list(range(alpha, n, alpha))
    mcxs = helper(qubit_indices, alphas)
    for mcx in mcxs:
        qc.mcx(mcx[:-1], mcx[-1])

    return qc


def adder_ripple_r25(num_qubits: int) -> QuantumCircuit:
    r"""The RV ripple carry adder [1].
    Construct an ancilla-free quantum adder circuit with sublinear depth based on the RV ripple-carry
    adder shown in [1]. The implementation has a depth of :math:`O(\log^2 n)` and uses
    math:`O(n \log n)` gates.

    As an example, a ripple-carry adder circuit that performs addition on two 4-qubit sized
    registers is as follows:

    .. parsed-literal::

                                       ┌───────────┐                    ┌────────┐
         a_0: ─────────────────────────┤0          ├────────────────────┤0       ├───────────────■─────────────────
                             ┌────────┐│           │                    │        │┌───────────┐  │
         a_1: ──■────────────┤0       ├┤2          ├──■─────────────────┤2       ├┤0          ├──┼────■────────────
                │            │        ││           │  │                 │        ││           │  │    │
         a_2: ──┼────■───────┤1       ├┤4          ├──┼────■────────────┤4       ├┤1 LAD_1_dg ├──┼────┼────■───────
                │    │       │        ││           │  │    │            │        ││           │  │    │    │
         a_3: ──┼────┼────■──┤2       ├┤6          ├──┼────┼────■───────┤6 LAD_2 ├┤2          ├──┼────┼────┼────■──
                │    │    │  │        ││           │  │    │    │       │        │└───────────┘┌─┴─┐  │    │    │
         b_0: ──┼────┼────┼──┤        ├┤1 LAD_2_dg ├──┼────┼────┼───────┤1       ├─────────────┤ X ├──┼────┼────┼──
              ┌─┴─┐  │    │  │  LAD_1 ││           │┌─┴─┐  │    │  ┌───┐│        │    ┌───┐    └───┘┌─┴─┐  │    │
         b_1: ┤ X ├──┼────┼──┤        ├┤3          ├┤ X ├──┼────┼──┤ X ├┤3       ├────┤ X ├─────────┤ X ├──┼────┼──
              └───┘┌─┴─┐  │  │        ││           │└───┘┌─┴─┐  │  ├───┤│        │    ├───┤         └───┘┌─┴─┐  │
         b_2: ─────┤ X ├──┼──┤        ├┤5          ├─────┤ X ├──┼──┤ X ├┤5       ├────┤ X ├──────────────┤ X ├──┼──
                   └───┘┌─┴─┐│        ││           │     └───┘┌─┴─┐└───┘└────────┘    └───┘              └───┘┌─┴─┐
         b_3: ──────────┤ X ├┤        ├┤7          ├──────────┤ X ├───────────────────────────────────────────┤ X ├
                        └───┘│        ││           │          └───┘                                           └───┘
        cout: ───────────────┤3       ├┤8          ├───────────────────────────────────────────────────────────────
                             └────────┘└───────────┘

    Here *LAD_1* and *LAD_2* are the CX and CCX ladders respectively introduced in [1]. Note that
    in this implementation the input register qubits are ordered as all qubits from
    the first input register, followed by all qubits from the second input register.

    Args:
        num_qubits: The size of the register.

    Returns:
        The quantum circuit implementing the RV ripple carry adder.

    Raises:
        ValueError: If ``num_qubits`` is lower than 1.

    References:

    1. Remaud and Vandaele, Ancilla-free Quantum Adder with Sublinear Depth, 2025.
    `arXiv:2501.16802 <https://arxiv.org/abs/2501.16802>`__

    """

    if num_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits, "b")
    qr_z = QuantumRegister(1, "cout")
    qc = QuantumCircuit(qr_a, qr_b, qr_z)

    if num_qubits == 1:
        qc.ccx(qr_a[0], qr_b[0], qr_z[0])
        qc.cx(qr_a[0], qr_b[0])
        return qc

    ab_interleaved = [q for pair in zip(qr_a, qr_b) for q in pair]

    qc.cx(qr_a[1:], qr_b[1:])
    qc.compose(_mcx_ladder(num_qubits - 1, 1), qr_a[1:] + qr_z[:], inplace=True)
    qc.compose(_mcx_ladder(num_qubits, 2).inverse(), ab_interleaved + qr_z[:], inplace=True)
    qc.cx(qr_a[1:], qr_b[1:])
    qc.x(qr_b[1 : max(2, num_qubits - 1)])
    qc.compose(_mcx_ladder(num_qubits - 1, 2), ab_interleaved[:-1], inplace=True)
    qc.x(qr_b[1 : max(2, num_qubits - 1)])
    qc.compose(_mcx_ladder(num_qubits - 2, 1).inverse(), qr_a[1:], inplace=True)
    qc.cx(qr_a, qr_b)

    return qc
