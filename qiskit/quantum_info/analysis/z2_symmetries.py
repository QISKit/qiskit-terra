# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Z2Symmetries for SparsePauliOp."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from copy import deepcopy
import math
from typing import cast

import numpy as np

from qiskit.exceptions import QiskitError
from ..operators import Pauli, SparsePauliOp


class Z2Symmetries:
    r"""
    The $Z_2$ symmetry converter identifies symmetries from the problem hamiltonian and uses them to
    provide a tapered - more efficient - representation of operators as Paulis for this problem. For each
    identified symmetry, one qubit can be eliminated in the Pauli representation at the cost of having to
    test two symmetry sectors (for the two possible eigenvalues - tapering values - of the symmetry).
    In certain problems such as the finding of the main operator's ground state, one can a priori
    identify the symmetry sector of the solution and thus effectively reduce the computational overhead.

    The following attributes can be read and updated once the ``Z2Symmetries`` object has been
    constructed.

    Attributes:
        tapering_values (list[int] or None): Values determining the sector.
        tol (float): The tolerance threshold for ignoring real and complex parts of a coefficient.

    References:
        [1]: Bravyi, S., et al, "Tapering off qubits to simulate fermionic Hamiltonians"
            `arXiv:1701.08213 <https://arxiv.org/abs/1701.08213>`__
    """

    def __init__(
        self,
        symmetries: Iterable[Pauli],
        sq_paulis: Iterable[Pauli],
        sq_list: Iterable[int],
        tapering_values: Iterable[int] | None = None,
        *,
        tol: float = 1e-14,
    ):
        r"""
        Args:
            symmetries: Object representing the list of $Z_2$ symmetries. These correspond to
                the generators of the symmetry group $\langle \tau_1, \tau_2\dots \rangle>$.
            sq_paulis: Object representing the list of single-qubit Pauli $\sigma^x_{q(i)}$
                anti-commuting with the symmetry $\tau_i$ and commuting with all the other symmetries
                $\tau_{j\neq i}$. These operators are used to construct the unitary Clifford operators.
            sq_list: The list of indices $q(i)$ of the single-qubit Pauli operators used to build the
                Clifford operators.
            tapering_values: List of eigenvalues determining the symmetry sector for each symmetry.
            tol: Tolerance threshold for ignoring real and complex parts of a coefficient.

        Raises:
            QiskitError: Invalid paulis. The lists of symmetries, single-qubit paulis support paulis
                and tapering values must be of equal length. This length is the number of applied
                symmetries and translates directly to the number of eliminated qubits.
        """
        symmetries = list(symmetries)
        sq_paulis = list(sq_paulis)
        sq_list = list(sq_list)
        tapering_values = None if tapering_values is None else list(tapering_values)

        if len(symmetries) != len(sq_paulis):
            raise QiskitError(
                f"The number of Z2 symmetries, {len(symmetries)}, has to match the number \
                of single-qubit pauli operators, {len(sq_paulis)}."
            )

        if len(sq_paulis) != len(sq_list):
            raise QiskitError(
                f"The number of single-qubit pauli operators, {len(sq_paulis)}, has to match the length \
                of the of single-qubit list, {len(sq_list)}."
            )

        if tapering_values is not None:
            if len(sq_list) != len(tapering_values):
                raise QiskitError(
                    f"The length of the single-qubit list, {len(sq_list)}, must match the length of the \
                    tapering values, {len(tapering_values)} ."
                )

        self._symmetries = symmetries
        self._sq_paulis = sq_paulis
        self._sq_list = sq_list
        self.tapering_values = tapering_values
        self.tol = tol

    @property
    def symmetries(self) -> list[Pauli]:
        """Return symmetries."""
        return self._symmetries

    @property
    def sq_paulis(self) -> list[Pauli]:
        """Return sq paulis."""
        return self._sq_paulis

    @property
    def cliffords(self) -> list[SparsePauliOp]:
        """
        Get clifford operators, built based on symmetries and single-qubit X.

        Returns:
            A list of unitaries used to diagonalize the Hamiltonian.
        """
        cliffords = [
            (SparsePauliOp(pauli_symm) + SparsePauliOp(sq_pauli)) / math.sqrt(2)
            for pauli_symm, sq_pauli in zip(self._symmetries, self._sq_paulis)
        ]
        return cliffords

    @property
    def sq_list(self) -> list[int]:
        """Return sq list."""
        return self._sq_list

    @property
    def settings(self) -> dict:
        """Return operator settings."""
        return {
            "symmetries": self._symmetries,
            "sq_paulis": self._sq_paulis,
            "sq_list": self._sq_list,
            "tapering_values": self.tapering_values,
        }

    def __str__(self):
        ret = ["Z2 symmetries:"]
        ret.append("Symmetries:")
        for symmetry in self._symmetries:
            ret.append(symmetry.to_label())
        ret.append("Single-Qubit Pauli X:")
        for x in self._sq_paulis:
            ret.append(x.to_label())
        ret.append("Cliffords:")
        for c in self.cliffords:
            ret.append(str(c))
        ret.append("Qubit index:")
        ret.append(str(self._sq_list))
        ret.append("Tapering values:")
        if self.tapering_values is None:
            possible_values = [
                str(list(coeff)) for coeff in itertools.product([1, -1], repeat=len(self._sq_list))
            ]
            possible_values = ", ".join(x for x in possible_values)
            ret.append("  - Possible values: " + possible_values)
        else:
            ret.append(str(self.tapering_values))

        ret = "\n".join(ret)
        return ret

    def is_empty(self) -> bool:
        """
        Check the z2_symmetries is empty or not.

        Returns:
            Empty or not.
        """
        return len(self._symmetries) == 0 or len(self._sq_paulis) == 0 or len(self._sq_list) == 0

    @classmethod
    def find_z2_symmetries(cls, operator: SparsePauliOp) -> Z2Symmetries:
        """
        Finds Z2 Pauli-type symmetries of a :class:`.SparsePauliOp`.

        Returns:
            A ``Z2Symmetries`` instance.
        """
        pauli_symmetries = []
        sq_paulis = []
        sq_list = []

        stacked_paulis = []

        test_idx = {
            "X_or_I": [(0, 0), (1, 0)],
            "Y_or_I": [(0, 0), (1, 1)],
            "Z_or_I": [(0, 0), (0, 1)],
        }
        test_row = {
            "Z_or_I": [(1, 0), (1, 1)],
            "X_or_I": [(0, 1), (1, 1)],
            "Y_or_I": [(0, 1), (1, 0)],
        }

        pauli_bool = {
            "Z_or_I": [False, True],
            "X_or_I": [True, False],
            "Y_or_I": [True, True],
        }

        if _sparse_pauli_op_is_zero(operator):
            return cls([], [], [], None)

        for pauli in iter(operator):
            stacked_paulis.append(
                np.concatenate((pauli.paulis.x[0], pauli.paulis.z[0]), axis=0).astype(int)
            )

        stacked_matrix = np.stack(stacked_paulis)
        symmetries = _kernel_f2(stacked_matrix)

        if not symmetries:
            return cls([], [], [], None)

        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape
        half_symm_shape = symm_shape[1] // 2
        stacked_symm_del = [
            np.delete(stacked_symmetries, row, axis=0) for row in range(symm_shape[0])
        ]

        def _test_symmetry_row_col(row: int, col: int, idx_test: list, row_test: list) -> bool:
            """
            Utility method that determines how to build the list of single-qubit Pauli X operators and
            the list of corresponding qubit indices from the stacked symmetries.
            This method is successively applied to Z type, X type and Y type symmetries (in this order)
            to build the letter at position (col) of the Pauli word corresponding to the symmetry at
            position (row).

            Args:
                row (int): Index of the symmetry for which the single-qubit Pauli X operator is being
                    built.
                col (int): Index of the letter in the Pauli word corresponding to the single-qubit Pauli
                    X operator.
                idx_test (list): List of possibilities for the stacked symmetries at all other rows
                    than row.
                row_test (list): List of possibilities for the stacked symmetries at row.

            Returns:
                Whether or not this symmetry type should be used to build this letter of this
                single-qubit Pauli X operator.
            """
            stacked_symm_idx_tests = np.array(
                [
                    (
                        stacked_symm_del[row][symm_idx, col],
                        stacked_symm_del[row][symm_idx, col + half_symm_shape],
                    )
                    in idx_test
                    for symm_idx in range(symm_shape[0] - 1)
                ]
            )

            stacked_symm_row_test = (
                stacked_symmetries[row, col],
                stacked_symmetries[row, col + half_symm_shape],
            ) in row_test

            return bool(np.all(stacked_symm_idx_tests)) and stacked_symm_row_test

        for row in range(symm_shape[0]):
            pauli_symmetries.append(
                Pauli(
                    (
                        stacked_symmetries[row, :half_symm_shape],
                        stacked_symmetries[row, half_symm_shape:],
                    )
                )
            )
            # Try all cases for the symmetries other than row: Z or I, Y or I, X or I on col qubit.
            # One test will return true.
            # Build the single-qubit Pauli accordingly.
            # Build the index list accordingly.
            for col in range(half_symm_shape):
                for key in ("Z_or_I", "X_or_I", "Y_or_I"):
                    current_test_result = _test_symmetry_row_col(
                        row, col, test_idx[key], test_row[key]
                    )
                    if current_test_result:
                        sq_paulis.append(
                            Pauli((np.zeros(half_symm_shape), np.zeros(half_symm_shape)))
                        )
                        sq_paulis[row].z[col] = pauli_bool[key][0]
                        sq_paulis[row].x[col] = pauli_bool[key][1]
                        sq_list.append(col)
                        break
                if current_test_result:
                    # We break out of the loop over columns only when one valid test is identified.
                    break

        return cls(pauli_symmetries, sq_paulis, sq_list, None)

    def convert_clifford(self, operator: SparsePauliOp) -> SparsePauliOp:
        """This method operates the first part of the tapering.
        It converts the operator by composing it with the clifford unitaries defined in the current
        symmetry.

        Args:
            operator: The to-be-tapered operator.

        Returns:
            ``SparsePauliOp`` corresponding to the converted operator.

        """

        if not self.is_empty() and not _sparse_pauli_op_is_zero(operator):
            # If the operator is zero then we can skip the following.
            for clifford in self.cliffords:
                operator = cast(SparsePauliOp, clifford @ operator @ clifford)
                operator = operator.simplify(atol=0.0)

        return operator

    def taper_clifford(self, operator: SparsePauliOp) -> SparsePauliOp | list[SparsePauliOp]:
        """Operate the second part of the tapering.
        This function assumes that the input operators have already been transformed using
        :meth:`convert_clifford`. The redundant qubits due to the symmetries are dropped and
        replaced by their two possible eigenvalues.

        Args:
            operator: Partially tapered operator resulting from a call to :meth:`convert_clifford`.

        Returns:
            If tapering_values is None: [:class:`SparsePauliOp`]; otherwise, :class:`SparsePauliOp`.

        """

        tapered_ops: SparsePauliOp | list[SparsePauliOp]
        if self.is_empty():
            tapered_ops = operator
        # If the operator is zero we still need to taper the operator to reduce its size i.e. the
        # number of qubits so for example 0*"IIII" could taper to 0*"II" when symmetries remove
        # two qubits.
        elif self.tapering_values is None:
            tapered_ops = [
                self._taper(operator, list(coeff))
                for coeff in itertools.product([1, -1], repeat=len(self._sq_list))
            ]
        else:
            tapered_ops = self._taper(operator, self.tapering_values)

        return tapered_ops

    def taper(self, operator: SparsePauliOp) -> SparsePauliOp | list[SparsePauliOp]:
        """
        Taper an operator based on the z2_symmetries info and sector defined by `tapering_values`.
        Returns operator if the symmetry object is empty.

        The tapering is a two-step algorithm which first converts the operator into a
        :class:`SparsePauliOp` with same eigenvalues but where some qubits are only acted upon
        with the Pauli operators I or X.
        The number M of these redundant qubits is equal to the number M of identified symmetries.

        The second step of the reduction consists in replacing these qubits with the possible
        eigenvalues of the corresponding Pauli X, giving 2^M new operators with M less qubits.
        If an eigenvalue sector was previously identified for the solution, then this reduces to
        1 new operator with M less qubits.

        Args:
            operator: The to-be-tapered operator.

        Returns:
            If tapering_values is None: [:class:`SparsePauliOp`]; otherwise, :class:`SparsePauliOp`.

        """

        converted_ops = self.convert_clifford(operator)
        tapered_ops = self.taper_clifford(converted_ops)

        return tapered_ops

    def _taper(self, op: SparsePauliOp, curr_tapering_values: list[int]) -> SparsePauliOp:
        pauli_list = []
        for pauli_term in iter(op):
            coeff_out = pauli_term.coeffs[0]
            for idx, qubit_idx in enumerate(self._sq_list):
                if pauli_term.paulis.z[0, qubit_idx] or pauli_term.paulis.x[0, qubit_idx]:
                    coeff_out = curr_tapering_values[idx] * coeff_out
            z_temp = np.delete(pauli_term.paulis.z[0].copy(), np.asarray(self._sq_list))
            x_temp = np.delete(pauli_term.paulis.x[0].copy(), np.asarray(self._sq_list))
            pauli_list.append((Pauli((z_temp, x_temp)).to_label(), coeff_out))

        spo = SparsePauliOp.from_list(pauli_list).simplify(atol=0.0)
        spo = spo.chop(self.tol)
        return spo

    def __eq__(self, other: Z2Symmetries) -> bool:
        """
        Overload `==` operation to evaluate equality between Z2Symmetries.

        Args:
            other: The `Z2Symmetries` to compare to self.

        Returns:
            A bool equal to the equality of self and other.
        """
        if not isinstance(other, Z2Symmetries):
            return False

        return (
            self.symmetries == other.symmetries
            and self.sq_paulis == other.sq_paulis
            and self.sq_list == other.sq_list
            and self.tapering_values == other.tapering_values
        )


def _kernel_f2(matrix_in):
    """
    Compute the kernel of a binary matrix on the binary finite field.

    Args:
        matrix_in (numpy.ndarray): Binary matrix.

    Returns:
        The list of kernel vectors.
    """
    size = matrix_in.shape
    kernel = []
    matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
    matrix_in_id_ech = (_row_echelon_f2(matrix_in_id.transpose())).transpose()

    for col in range(size[1]):
        if np.array_equal(
            matrix_in_id_ech[0 : size[0], col], np.zeros(size[0])
        ) and not np.array_equal(matrix_in_id_ech[size[0] :, col], np.zeros(size[1])):
            kernel.append(matrix_in_id_ech[size[0] :, col])

    return kernel


def _row_echelon_f2(matrix_in):
    """
    Compute the row Echelon form of a binary matrix on the binary finite field.

    Args:
        matrix_in (numpy.ndarray): Binary matrix.

    Returns:
        Matrix_in in Echelon row form.
    """
    size = matrix_in.shape

    for i in range(size[0]):
        pivot_index = 0
        for j in range(size[1]):
            if matrix_in[i, j] == 1:
                pivot_index = j
                break
        for k in range(size[0]):
            if k != i and matrix_in[k, pivot_index] == 1:
                matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)

    matrix_out_temp = deepcopy(matrix_in)
    indices = []
    matrix_out = np.zeros(size)

    for i in range(size[0] - 1):
        if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
            indices.append(i)
    for row in np.sort(indices)[::-1]:
        matrix_out_temp = np.delete(matrix_out_temp, (row), axis=0)

    matrix_out[0 : size[0] - len(indices), :] = matrix_out_temp
    matrix_out = matrix_out.astype(int)

    return matrix_out


def _sparse_pauli_op_is_zero(op: SparsePauliOp) -> bool:
    """Returns whether or not this operator represents a zero operation."""
    op = op.simplify()
    return len(op.coeffs) == 1 and op.coeffs[0] == 0
