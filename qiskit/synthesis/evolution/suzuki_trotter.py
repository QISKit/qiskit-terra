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

"""The Suzuki-Trotter product formula."""

from __future__ import annotations

import typing
from collections.abc import Callable
from itertools import chain
import numpy as np

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import qiskit.quantum_info

from .product_formula import ProductFormula, reorder_paulis

if typing.TYPE_CHECKING:
    from qiskit.circuit.quantumcircuit import ParameterValueType
    from qiskit.circuit.library import PauliEvolutionGate


class SuzukiTrotter(ProductFormula):
    r"""The (higher order) Suzuki-Trotter product formula.

    The Suzuki-Trotter formulas improve the error of the Lie-Trotter approximation.
    For example, the second order decomposition is

    .. math::

        e^{A + B} \approx e^{B/2} e^{A} e^{B/2}.

    Higher order decompositions are based on recursions, see Ref. [1] for more details.

    In this implementation, the operators are provided as sum terms of a Pauli operator.
    For example, in the second order Suzuki-Trotter decomposition we approximate

    .. math::

        e^{-it(XI + ZZ)} = e^{-it/2 XI}e^{-it ZZ}e^{-it/2 XI} + \mathcal{O}(t^3).

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
        order: int = 2,
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
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be ``"chain"``,
                where next neighbor connections are used, or ``"fountain"``, where all qubits are
                connected to one. This only takes effect when ``atomic_evolution is None``.
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

        Raises:
            ValueError: If order is not even
        """
        if order > 1 and order % 2 == 1:
            raise ValueError(
                "Suzuki product formulae are symmetric and therefore only defined "
                f"for when the order is 1 or even, not {order}."
            )

        super().__init__(
            order,
            reps,
            insert_barriers,
            cx_structure,
            atomic_evolution,
            wrap,
            preserve_order=preserve_order,
            atomic_evolution_sparse_observable=atomic_evolution_sparse_observable,
        )

    def expand(
        self, evolution: PauliEvolutionGate
    ) -> list[tuple[str, list[int], ParameterValueType]]:
        """Expand the Hamiltonian into a Suzuki-Trotter sequence of sparse gates.

        For example, the Hamiltonian ``H = IX + ZZ`` for an evolution time ``t`` and
        1 repetition for an order 2 formula would get decomposed into a list of 3-tuples
        containing ``(pauli, indices, rz_rotation_angle)``, that is:

        .. code-block:: text

            ("X", [0], t), ("ZZ", [0, 1], 2t), ("X", [0], t)

        Note that the rotation angle contains a factor of 2, such that the evolution
        of a Pauli :math:`P` over time :math:`t`, which is :math:`e^{itP}`, is represented
        by ``(P, indices, 2 * t)``.

        For ``N`` repetitions, this sequence would be repeated ``N`` times and the coefficients
        divided by ``N``.

        Args:
            evolution: The evolution gate to expand.

        Returns:
            The Pauli network implementing the Trotter expansion.
        """
        operators = evolution.operator
        time = evolution.time

        def to_sparse_list(operator):
            sparse_list = operator.to_sparse_list()
            paulis = [
                (pauli, indices, real_or_fail(coeff) * time * 2 / self.reps)
                for pauli, indices, coeff in sparse_list
            ]
            if not self.preserve_order:
                return reorder_paulis(paulis)

            return paulis

        # construct the evolution circuit
        if isinstance(operators, list):  # already sorted into commuting bits
            non_commuting = [to_sparse_list(operator) for operator in operators]
        else:
            # Assume no commutativity here. If we were to group commuting Paulis,
            # here would be the location to do so.
            non_commuting = [[op] for op in to_sparse_list(operators)]

        # we're already done here since Lie Trotter does not do any operator repetition
        product_formula = self._recurse(self.order, non_commuting)
        flattened = self.reps * list(chain.from_iterable(product_formula))

        return flattened

    @staticmethod
    def _recurse(order, grouped_paulis):
        if order == 1:
            return grouped_paulis

        elif order == 2:
            halves = [
                [(label, qubits, coeff / 2) for label, qubits, coeff in paulis]
                for paulis in grouped_paulis[:-1]
            ]
            full = [grouped_paulis[-1]]
            return halves + full + list(reversed(halves))

        else:
            reduction = 1 / (4 - 4 ** (1 / (order - 1)))
            outer = 2 * SuzukiTrotter._recurse(
                order - 2,
                [
                    [(label, qubits, coeff * reduction) for label, qubits, coeff in paulis]
                    for paulis in grouped_paulis
                ],
            )
            inner = SuzukiTrotter._recurse(
                order - 2,
                [
                    [
                        (label, qubits, coeff * (1 - 4 * reduction))
                        for label, qubits, coeff in paulis
                    ]
                    for paulis in grouped_paulis
                ],
            )
            return outer + inner + outer


def real_or_fail(value, tol=100):
    """Return real if close, otherwise fail. Unbound parameters are left unchanged.

    Based on NumPy's ``real_if_close``, i.e. ``tol`` is in terms of machine precision for float.
    """
    if isinstance(value, ParameterExpression):
        return value

    abstol = tol * np.finfo(float).eps
    if abs(np.imag(value)) < abstol:
        return np.real(value)

    raise ValueError(f"Encountered complex value {value}, but expected real.")
