# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""QDrift Class"""

from __future__ import annotations

import math
import typing
from itertools import chain
from collections.abc import Callable
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.exceptions import QiskitError
import qiskit.quantum_info

from .product_formula import ProductFormula, reorder_paulis

if typing.TYPE_CHECKING:
    from qiskit.circuit.library import PauliEvolutionGate


class QDrift(ProductFormula):
    r"""The QDrift Trotterization method, which selects each term in the
    Trotterization randomly, with a probability proportional to its weight. Based on the work
    of Earl Campbell in Ref. [1].

    References:
        [1]: E. Campbell, "A random compiler for fast Hamiltonian simulation" (2018).
        `arXiv:quant-ph/1811.08017 <https://arxiv.org/abs/1811.08017>`_
    """

    def __init__(
        self,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[QuantumCircuit, qiskit.quantum_info.Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        seed: int | None = None,
        wrap: bool = False,
        preserve_order: bool = True,
        *,
        atomic_evolution_sparse_observable: bool = False,
    ) -> None:
        r"""
        Args:
            reps: The number of times to repeat the Trotterization circuit.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``, where all
                qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single
                :class:`~.quantum_info.Pauli`, or :class:`.SparsePauliOp` of only commuting terms,
                to a circuit. The function takes in three arguments: the circuit to append the
                evolution to, the Pauli operator to evolve, and the evolution time. By default, a
                single Pauli evolution is decomposed into a chain of ``CX`` gates and a single
                ``RZ`` gate.
            seed: An optional seed for reproducibility of the random sampling process.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. This only takes
                effect when ``atomic_evolution is None``.
            preserve_order: If ``False``, allows reordering the terms of the operator to
                potentially yield a shallower evolution circuit. Not relevant
                when synthesizing operator with a single term.
            atomic_evolution_sparse_observable: If a custom ``atomic_evolution`` is passed,
                which does not yet support :class:`.SparseObservable`\ s as input, set this
                argument to ``False`` to automatically apply a conversion to :class:`.SparsePauliOp`.
                This argument is supported until Qiskit 2.2, at which point all atomic evolutions
                are required to support :class:`.SparseObservable`\ s as input.
        """
        super().__init__(
            1,
            reps,
            insert_barriers,
            cx_structure,
            atomic_evolution,
            wrap,
            preserve_order,
            atomic_evolution_sparse_observable=atomic_evolution_sparse_observable,
        )
        self.sampled_ops = None
        self.rng = np.random.default_rng(seed)

    def expand(self, evolution: PauliEvolutionGate) -> list[tuple[str, tuple[int], float]]:
        operators = evolution.operator
        time = evolution.time  # used to determine the number of gates

        # QDrift is based on first-order Lie-Trotter, hence we can just concatenate all
        # Pauli terms and ignore commutations
        if isinstance(operators, list):
            paulis = list(chain.from_iterable([op.to_sparse_list() for op in operators]))
        else:
            paulis = operators.to_sparse_list()

        try:
            coeffs = [float(np.real_if_close(coeff)) for _, _, coeff in paulis]
        except TypeError as exc:
            raise QiskitError("QDrift requires bound, real coefficients.") from exc

        # We artificially make the weights positive
        weights = np.abs(coeffs)
        lambd = np.sum(weights)

        num_gates = math.ceil(2 * (lambd**2) * (time**2) * self.reps)

        # The protocol calls for the removal of the individual coefficients,
        # and multiplication by a constant evolution time.
        sampled = self.rng.choice(
            np.array(paulis, dtype=object), size=(num_gates,), p=weights / lambd
        )

        rescaled_time = 2 * lambd / num_gates * time
        sampled_paulis = [
            (pauli[0], pauli[1], np.real(np.sign(pauli[2])) * rescaled_time) for pauli in sampled
        ]

        if not self.preserve_order:
            sampled_paulis = reorder_paulis(sampled_paulis)

        return sampled_paulis
