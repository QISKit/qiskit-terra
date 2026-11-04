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

"""The Lie-Trotter product formula."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import qiskit.quantum_info

from .suzuki_trotter import SuzukiTrotter


class LieTrotter(SuzukiTrotter):
    r"""The Lie-Trotter product formula.

    The Lie-Trotter formula approximates the exponential of two non-commuting operators
    with products of their exponentials up to a second order error:

    .. math::

        e^{A + B} \approx e^{A}e^{B}.

    In this implementation, the operators are provided as sum terms of a Pauli operator.
    For example, we approximate

    .. math::

        e^{-it(XI + ZZ)} = e^{-it XI}e^{-it ZZ} + \mathcal{O}(t^2).

    References:

        [1]: D. Berry, G. Ahokas, R. Cleve and B. Sanders,
        "Efficient quantum algorithms for simulating sparse Hamiltonians" (2006).
        `arXiv:quant-ph/0508139 <https://arxiv.org/abs/quant-ph/0508139>`_
        [2]: N. Hatano and M. Suzuki,
        "Finding Exponential Product Formulas of Higher Orders" (2005).
        `arXiv:math-ph/0506007 <https://arxiv.org/pdf/math-ph/0506007.pdf>`_
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
        wrap: bool = False,
        preserve_order: bool = True,
        *,
        atomic_evolution_sparse_observable: bool = False,
    ) -> None:
        r"""
        Args:
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single
                :class:`~.quantum_info.Pauli`, or :class:`.SparsePauliOp` of only commuting terms,
                to a circuit. The function takes in three arguments: the circuit to append the
                evolution to, the Pauli operator to evolve, and the evolution time. By default, a
                single Pauli evolution is decomposed into a chain of ``CX`` gates and a single
                ``RZ`` gate.
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
            preserve_order=preserve_order,
            atomic_evolution_sparse_observable=atomic_evolution_sparse_observable,
        )

    @property
    def settings(self) -> dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.
        """
        if self._atomic_evolution is not None:
            raise NotImplementedError(
                "Cannot serialize a product formula with a custom atomic evolution."
            )

        return {
            "reps": self.reps,
            "insert_barriers": self.insert_barriers,
            "cx_structure": self._cx_structure,
            "wrap": self._wrap,
        }
