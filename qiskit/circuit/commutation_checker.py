# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Code from commutative_analysis pass that checks commutation relations between DAG nodes."""

from __future__ import annotations

from collections.abc import Sequence
import typing

from qiskit.circuit.operation import Operation
from qiskit.circuit import Qubit
from qiskit._accelerate.commutation_checker import CommutationChecker as RustChecker
from qiskit.utils import deprecate_arg, deprecate_func

if typing.TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode


class CommutationChecker:
    r"""Check commutations of two operations.

    Two unitaries :math:`A` and :math:`B` on :math:`n` qubits commute if

    .. math::

        \frac{2^n F_{\text{process}}(AB, BA) + 1}{2^n + 1} > 1 - \varepsilon,

    where

    .. math::

        F_{\text{process}}(U_1, U_2) = \left|\frac{\mathrm{Tr}(U_1 U_2^\dagger)}{2^n} \right|^2,

    and we set :math:`\varepsilon` to :math:`10^{-12}` to account for round-off errors on
    few-qubit systems. This metric is chosen for consistency with other closeness checks in
    Qiskit.

    When possible, commutation relations are queried from a lookup table. This is the case
    for standard gates without parameters (such as :class:`.XGate` or :class:`.HGate`) or
    gates with free parameters (such as :class:`.RXGate` with a :class:`.ParameterExpression` as
    angle). Otherwise, a matrix-based check is performed, where two operations are said to
    commute, if the average gate fidelity of performing the commutation is above a certain threshold
    (see ``approximation_degree``).
    """

    @deprecate_arg("cache_max_entries", since="2.5.0", removal_timeline="in Qiskit 3.0")
    def __init__(
        self,
        standard_gate_commutations: dict | None = None,
        cache_max_entries: int = 10**6,
        *,
        gates: set[str] | None = None,
    ):
        self.cc = RustChecker(standard_gate_commutations, gates)

    def commute_nodes(
        self,
        op1: DAGOpNode,
        op2: DAGOpNode,
        max_num_qubits: int = 3,
        approximation_degree: float = 1.0,
    ) -> bool:
        """Checks if two :class:`.DAGOpNode` objects commute.

        This is equivalent to :meth:`commute` but with the operation, qubits, and clbits
        bundled in the :class:`.DAGOpNode` object. See :meth:`commute` for more details.
        """
        return self.cc.commute_nodes(op1, op2, max_num_qubits, approximation_degree, max_num_qubits)

    def commute(
        self,
        op1: Operation,
        qargs1: Sequence[Qubit | int],
        cargs1: Sequence[Qubit | int],
        op2: Operation,
        qargs2: Sequence[Qubit | int],
        cargs2: Sequence[Qubit | int],
        max_num_qubits: int | None = None,
        approximation_degree: float = 1.0,
        matrix_max_num_qubits: int = 3,
        check_inputs: bool = True,
    ) -> bool:
        """
        Checks if two Operations commute. The return value of ``True`` means that the operations
        truly commute, and the return value of ``False`` means that either the operations do not
        commute or that the commutation check was skipped (for example, when the operations
        have conditions or have too many qubits).

        Args:
            op1: first operation.
            qargs1: first operation's qubits.
            cargs1: first operation's clbits.
            op2: second operation.
            qargs2: second operation's qubits.
            cargs2: second operation's clbits.
            max_num_qubits: the maximum number of qubits to consider, the check may be skipped if
                the number of qubits for either operation exceeds this amount. Defaults to ``None``,
                which means no limit. See also ``matrix_max_num_qubits`` to limit the dimension
                of matrices computed.
            approximation_degree: If the average gate fidelity in between the two operations
                is above this number (up to ``1e-12``) they are assumed to commute.
            matrix_max_num_qubits: the maximum number of qubits for which it is allowed to compute
                the matrix representation. This is needed if there is no efficient check readily
                available, e.g. for custom gates.
            check_inputs: If ``True`` (default), reject duplicate qubit arguments with
                :exc:`.QiskitError` at the public boundary. Pass ``False`` to skip that
                validation when the caller already guarantees distinct qubits.

        Returns:
            Whether two operations commute.
        """
        return self.cc.commute(
            op1,
            tuple(qargs1),
            tuple(cargs1),
            op2,
            tuple(qargs2),
            tuple(cargs2),
            max_num_qubits,
            approximation_degree,
            matrix_max_num_qubits,
            check_inputs,
        )

    @deprecate_func(since="2.5", removal_timeline="in Qiskit 3.0")
    def num_cached_entries(self):
        """Returns number of cached entries

        This method will always return 0 because there is no longer an
        internal cache.
        """
        return 0

    @deprecate_func(since="2.5", removal_timeline="in Qiskit 3.0")
    def clear_cached_commutations(self):
        """Clears the dictionary holding cached commutations

        This method is a no-op as there is no longer an internal cache
        """

    def check_commutation_entries(
        self,
        first_op: Operation,
        first_qargs: list,
        second_op: Operation,
        second_qargs: list,
    ) -> bool | None:
        """Returns stored commutation relation if any

        Args:
            first_op: first operation.
            first_qargs: first operation's qubits.
            second_op: second operation.
            second_qargs: second operation's qubits.

        Return:
            bool: True if the gates commute and false if it is not the case.
        """
        return self.cc.library.check_commutation_entries(
            first_op, first_qargs, second_op, second_qargs
        )
