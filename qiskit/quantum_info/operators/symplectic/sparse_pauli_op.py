# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
N-Qubit Sparse Pauli Operator class.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from collections.abc import Mapping, Sequence, Iterable
from numbers import Number
from copy import deepcopy

import numpy as np
import rustworkx as rx

from qiskit._accelerate.sparse_pauli_op import (
    ZXPaulis,
    decompose_dense,
    to_matrix_dense,
    to_matrix_sparse,
    unordered_unique,
)
from qiskit._accelerate.sparse_observable import SparseObservable
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametertable import ParameterView
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.symplectic.pauli import Pauli

if TYPE_CHECKING:
    from qiskit.transpiler.layout import TranspileLayout


class SparsePauliOp(LinearOp):
    """Sparse N-qubit operator in a Pauli basis representation.

    This is a sparse representation of an N-qubit matrix
    :class:`~qiskit.quantum_info.Operator` in terms of N-qubit
    :class:`~qiskit.quantum_info.PauliList` and complex coefficients.

    It can be used for performing operator arithmetic for hundreds of qubits
    if the number of non-zero Pauli basis terms is sufficiently small.

    The Pauli basis components are stored as a
    :class:`~qiskit.quantum_info.PauliList` object and can be accessed
    using the :attr:`~SparsePauliOp.paulis` attribute. The coefficients
    are stored as a complex Numpy array vector and can be accessed using
    the :attr:`~SparsePauliOp.coeffs` attribute.

    .. rubric:: Data type of coefficients

    The default ``dtype`` of the internal ``coeffs`` Numpy array is ``complex128``.  Users can
    configure this by passing ``np.ndarray`` with a different dtype.  For example, a parameterized
    :class:`SparsePauliOp` can be made as follows:

    .. plot::
       :include-source:
       :nofigs:

        >>> import numpy as np
        >>> from qiskit.circuit import ParameterVector
        >>> from qiskit.quantum_info import SparsePauliOp

        >>> SparsePauliOp(["II", "XZ"], np.array(ParameterVector("a", 2)))
        SparsePauliOp(['II', 'XZ'],
              coeffs=[ParameterExpression(1.0*a[0]), ParameterExpression(1.0*a[1])])

    .. note::

      Parameterized :class:`SparsePauliOp` does not support the following methods:

      - ``to_matrix(sparse=True)`` since ``scipy.sparse`` cannot have objects as elements.
      - ``to_operator()`` since :class:`~.quantum_info.Operator` does not support objects.
      - ``sort``, ``argsort`` since :class:`.ParameterExpression` does not support comparison.
      - ``equiv`` since :class:`.ParameterExpression` cannot be converted into complex.
      - ``chop`` since :class:`.ParameterExpression` does not support absolute value.
    """

    def __init__(
        self,
        data: PauliList | SparsePauliOp | Pauli | list | str,
        coeffs: np.ndarray | None = None,
        *,
        ignore_pauli_phase: bool = False,
        copy: bool = True,
    ):
        """Initialize an operator object.

        Args:
            data (PauliList or SparsePauliOp or Pauli or list or str): Pauli list of
                terms.  A list of Pauli strings or a Pauli string is also allowed.
            coeffs (np.ndarray): complex coefficients for Pauli terms.

                .. note::

                    If ``data`` is a :obj:`~SparsePauliOp` and ``coeffs`` is not ``None``, the value
                    of the ``SparsePauliOp.coeffs`` will be ignored, and only the passed keyword
                    argument ``coeffs`` will be used.

            ignore_pauli_phase (bool): if true, any ``phase`` component of a given :obj:`~PauliList`
                will be assumed to be zero.  This is more efficient in cases where a
                :obj:`~PauliList` has been constructed purely for this object, and it is already
                known that the phases in the ZX-convention are zero.  It only makes sense to pass
                this option when giving :obj:`~PauliList` data.  (Default: False)
            copy (bool): copy the input data if True, otherwise assign it directly, if possible.
                (Default: True)

        Raises:
            QiskitError: If the input data or coeffs are invalid.
        """
        if ignore_pauli_phase and not isinstance(data, PauliList):
            raise QiskitError("ignore_pauli_phase=True is only valid with PauliList data")

        if isinstance(data, SparsePauliOp):
            if coeffs is None:
                coeffs = data.coeffs
            data = data._pauli_list
            # `SparsePauliOp._pauli_list` is already compatible with the internal ZX-phase
            # convention.  See `BasePauli._from_array` for the internal ZX-phase convention.
            ignore_pauli_phase = True

        pauli_list = PauliList(data.copy() if copy and hasattr(data, "copy") else data)

        if coeffs is None:
            coeffs = np.ones(pauli_list.size, dtype=complex)
        else:
            if isinstance(coeffs, np.ndarray):
                dtype = object if coeffs.dtype == object else complex
            else:
                if not isinstance(coeffs, Sequence):
                    coeffs = [coeffs]
                if any(isinstance(coeff, ParameterExpression) for coeff in coeffs):
                    dtype = object
                else:
                    dtype = complex

            coeffs_asarray = np.asarray(coeffs, dtype=dtype)
            coeffs = (
                coeffs_asarray.copy()
                if copy and np.may_share_memory(coeffs, coeffs_asarray)
                else coeffs_asarray
            )

        if ignore_pauli_phase:
            # Fast path used in copy operations, where the phase of the PauliList is already known
            # to be zero.  This path only works if the input data is compatible with the internal
            # ZX-phase convention.
            self._pauli_list = pauli_list
            self._coeffs = coeffs
        else:
            # move the phase of `pauli_list` to `self._coeffs`
            phase = pauli_list._phase
            count_y = pauli_list._count_y()

            # Compute exponentiation via integer arithmetic and lookup table to avoid
            # floating point errors
            exponent = (phase - count_y) % 4
            lookup = np.array([1 + 0j, -1j, -1 + 0j, 1j], dtype=coeffs.dtype)

            vals = lookup[exponent]
            self._coeffs = vals * coeffs

            # Update pauli_list phase
            pauli_list._phase = np.mod(count_y, 4)
            self._pauli_list = pauli_list

        if self._coeffs.shape != (self._pauli_list.size,):
            raise QiskitError(
                "coeff vector is incorrect shape for number"
                f" of Paulis {self._coeffs.shape} != {self._pauli_list.size}"
            )
        # Initialize LinearOp
        super().__init__(num_qubits=self._pauli_list.num_qubits)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        arr = self.to_matrix()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def __repr__(self):
        prefix = "SparsePauliOp("
        pad = len(prefix) * " "
        return (
            f"{prefix}{self.paulis.to_labels()},\n{pad}"
            f"coeffs={np.array2string(self.coeffs, separator=', ')})"
        )

    def __eq__(self, other):
        """Entrywise comparison of two SparsePauliOp operators"""
        return (
            super().__eq__(other)
            and self.coeffs.dtype == other.coeffs.dtype
            and self.coeffs.shape == other.coeffs.shape
            and self.paulis == other.paulis
            and (
                np.allclose(self.coeffs, other.coeffs)
                if self.coeffs.dtype != object
                else (self.coeffs == other.coeffs).all()
            )
        )

    def equiv(self, other: SparsePauliOp, atol: float | None = None) -> bool:
        """Check if two SparsePauliOp operators are equivalent.

        Args:
            other (SparsePauliOp): an operator object.
            atol: Absolute numerical tolerance for checking equivalence.

        Returns:
            bool: True if the operator is equivalent to ``self``.
        """
        if not super().__eq__(other):
            return False
        if atol is None:
            atol = self.atol
        return np.allclose((self - other).simplify().coeffs, 0.0, atol=atol)

    @property
    def settings(self) -> dict:
        """Return settings."""
        return {"data": self._pauli_list, "coeffs": self._coeffs}

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def size(self):
        """The number of Pauli of Pauli terms in the operator."""
        return self._pauli_list.size

    def __len__(self):
        """Return the size."""
        return self.size

    @property
    def paulis(self):
        """Return the PauliList."""
        return self._pauli_list

    @paulis.setter
    def paulis(self, value):
        if not isinstance(value, PauliList):
            value = PauliList(value)
        if value.num_qubits != self.num_qubits:
            raise ValueError(
                f"incorrect number of qubits: expected {self.num_qubits}, got {value.num_qubits}"
            )
        if len(value) != len(self.paulis):
            raise ValueError(
                f"incorrect number of operators: expected {len(self.paulis)}, got {len(value)}"
            )
        self.coeffs *= (-1j) ** value.phase
        value.phase = 0
        self._pauli_list = value

    @property
    def coeffs(self):
        """Return the Pauli coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        """Set Pauli coefficients."""
        self._coeffs[:] = value

    def __getitem__(self, key):
        """Return a view of the SparsePauliOp."""
        # Returns a view of specified rows of the PauliList
        # This supports all slicing operations the underlying array supports.
        if isinstance(key, (int, np.integer)):
            key = [key]
        return SparsePauliOp(self.paulis[key], self.coeffs[key])

    def __setitem__(self, key, value):
        """Update SparsePauliOp."""
        # Modify specified rows of the PauliList
        if not isinstance(value, SparsePauliOp):
            value = SparsePauliOp(value)
        self.paulis[key] = value.paulis
        self.coeffs[key] = value.coeffs

    # ---------------------------------------------------------------------
    # LinearOp Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        # Conjugation conjugates phases and also Y.conj() = -Y
        # Hence we need to multiply conjugated coeffs by -1
        # for rows with an odd number of Y terms.
        # Find rows with Ys
        ret = self.transpose()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def transpose(self):
        # The only effect transposition has is Y.T = -Y
        # Hence we need to multiply coeffs by -1 for rows with an
        # odd number of Y terms.
        ret = self.copy()
        minus = (-1) ** ret.paulis._count_y()
        ret._coeffs *= minus
        return ret

    def adjoint(self):
        # Pauli's are self adjoint, so we only need to
        # conjugate the phases
        ret = self.copy()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def compose(
        self, other: SparsePauliOp, qargs: list | None = None, front: bool = False
    ) -> SparsePauliOp:
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        # Validate composition dimensions and qargs match
        self._op_shape.compose(other._op_shape, qargs, front)

        if qargs is not None:
            x1, z1 = self.paulis.x[:, qargs], self.paulis.z[:, qargs]
        else:
            x1, z1 = self.paulis.x, self.paulis.z
        x2, z2 = other.paulis.x, other.paulis.z
        num_qubits = other.num_qubits

        # This method is the outer version of `BasePauli.compose`.
        # `x1` and `z1` have shape `(self.size, num_qubits)`.
        # `x2` and `z2` have shape `(other.size, num_qubits)`.
        # `x1[:, no.newaxis]` results in shape `(self.size, 1, num_qubits)`.
        # `ar = ufunc(x1[:, np.newaxis], x2)` will be in shape `(self.size, other.size, num_qubits)`.
        # So, `ar.reshape((-1, num_qubits))` will be in shape `(self.size * other.size, num_qubits)`.
        # Ref: https://numpy.org/doc/stable/user/theory.broadcasting.html

        phase = np.add.outer(self.paulis._phase, other.paulis._phase).reshape(-1)
        if front:
            q = np.logical_and(x1[:, np.newaxis], z2).reshape((-1, num_qubits))
        else:
            q = np.logical_and(z1[:, np.newaxis], x2).reshape((-1, num_qubits))
        # `np.mod` will be applied to `phase` in `SparsePauliOp.__init__`
        phase = phase + 2 * q.sum(axis=1, dtype=np.uint8)

        x3 = np.logical_xor(x1[:, np.newaxis], x2).reshape((-1, num_qubits))
        z3 = np.logical_xor(z1[:, np.newaxis], z2).reshape((-1, num_qubits))

        if qargs is None:
            pauli_list = PauliList(BasePauli(z3, x3, phase))
        else:
            x4 = np.repeat(self.paulis.x, other.size, axis=0)
            z4 = np.repeat(self.paulis.z, other.size, axis=0)
            x4[:, qargs] = x3
            z4[:, qargs] = z3
            pauli_list = PauliList(BasePauli(z4, x4, phase))

        # note: the following is a faster code equivalent to
        # `coeffs = np.kron(self.coeffs, other.coeffs)`
        # since `self.coeffs` and `other.coeffs` are both 1d arrays.
        coeffs = np.multiply.outer(self.coeffs, other.coeffs).ravel()
        return SparsePauliOp(pauli_list, coeffs, copy=False)

    def tensor(self, other: SparsePauliOp) -> SparsePauliOp:
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        return self._tensor(self, other)

    def expand(self, other: SparsePauliOp) -> SparsePauliOp:
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        paulis = a.paulis.tensor(b.paulis)
        coeffs = np.kron(a.coeffs, b.coeffs)
        return SparsePauliOp(paulis, coeffs, ignore_pauli_phase=True, copy=False)

    def _add(self, other, qargs=None):
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        paulis = self.paulis._add(other.paulis, qargs=qargs)
        coeffs = np.hstack((self.coeffs, other.coeffs))
        return SparsePauliOp(paulis, coeffs, ignore_pauli_phase=True, copy=False)

    def _multiply(self, other):
        if not isinstance(other, (Number, ParameterExpression)):
            raise QiskitError("other is neither a Number nor a Parameter/ParameterExpression")
        if other == 0:
            # Check edge case that we deleted all Paulis
            # In this case we return an identity Pauli with a zero coefficient
            paulis = PauliList.from_symplectic(
                np.zeros((1, self.num_qubits), dtype=bool),
                np.zeros((1, self.num_qubits), dtype=bool),
            )
            coeffs = np.array([0j])
            return SparsePauliOp(paulis, coeffs, ignore_pauli_phase=True, copy=False)
        # Otherwise we just update the phases
        return SparsePauliOp(
            self.paulis.copy(), other * self.coeffs, ignore_pauli_phase=True, copy=False
        )

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def is_unitary(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Return True if operator is a unitary matrix.

        This method checks whether the operator composed with its adjoint equals
        the identity, up to the provided tolerance. The tolerance is used when
        simplifying the composed operator and checking if the result is the identity.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optional. Relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            bool: True if the operator is unitary, False otherwise.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        # Compose with adjoint
        val = self.compose(self.adjoint()).simplify(atol=atol, rtol=rtol)
        # See if the result is an identity
        return (
            val.size == 1
            and np.isclose(val.coeffs[0], 1.0, atol=atol, rtol=rtol)
            and not np.any(val.paulis.x)
            and not np.any(val.paulis.z)
        )

    def simplify(self, atol: float | None = None, rtol: float | None = None) -> SparsePauliOp:
        """Simplify PauliList by combining duplicates and removing zeros.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optional. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            SparsePauliOp: the simplified SparsePauliOp operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        paulis_x = self.paulis.x
        paulis_z = self.paulis.z
        nz_coeffs = self.coeffs

        array = np.packbits(paulis_x, axis=1).astype(np.uint16) * 256 + np.packbits(
            paulis_z, axis=1
        )
        indexes, inverses = unordered_unique(array)

        coeffs = np.zeros(indexes.shape[0], dtype=self.coeffs.dtype)
        np.add.at(coeffs, inverses, nz_coeffs)

        # Filter non-zero coefficients
        if self.coeffs.dtype == object:

            def to_complex(coeff):
                if not hasattr(coeff, "numeric"):
                    return coeff
                # simplify() collapses cancellations like a + b - a - b to 0,
                # so numeric() can evaluate them without needing sympy.
                try:
                    return complex(coeff.simplify().numeric(strict=False))
                except TypeError:
                    return np.nan

            non_zero = np.logical_not(
                np.isclose([to_complex(x) for x in self.coeffs], 0, atol=atol, rtol=rtol)
            )
        else:
            non_zero = np.logical_not(np.isclose(self.coeffs, 0, atol=atol, rtol=rtol))

        # Delete zero coefficient rows
        if self.coeffs.dtype == object:
            is_zero = np.array(
                [np.isclose(to_complex(coeff), 0, atol=atol, rtol=rtol) for coeff in coeffs]
            )
        else:
            is_zero = np.isclose(coeffs, 0, atol=atol, rtol=rtol)
        # Check edge case that we deleted all Paulis
        # In this case we return an identity Pauli with a zero coefficient
        if np.all(is_zero):
            x = np.zeros((1, self.num_qubits), dtype=bool)
            z = np.zeros((1, self.num_qubits), dtype=bool)
            coeffs = np.array([0j], dtype=self.coeffs.dtype)
        else:
            non_zero = np.logical_not(is_zero)
            non_zero_indexes = indexes[non_zero]
            x = paulis_x[non_zero_indexes]
            z = paulis_z[non_zero_indexes]
            coeffs = coeffs[non_zero]

        return SparsePauliOp(
            PauliList.from_symplectic(z, x), coeffs, ignore_pauli_phase=True, copy=False
        )

    def argsort(self, weight: bool = False):
        """Return indices for sorting the rows of the table.

        Returns the composition of permutations in the order of sorting
        by coefficient and sorting by Pauli.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        **Example**

        Here is an example of how to use SparsePauliOp argsort.

        .. plot::
           :include-source:
           :nofigs:

            import numpy as np
            from qiskit.quantum_info import SparsePauliOp

            # 2-qubit labels
            labels = ["XX", "XX", "XX", "YI", "II", "XZ", "XY", "XI"]
            # coeffs
            coeffs = [2.+1.j, 2.+2.j, 3.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j]

            # init
            spo = SparsePauliOp(labels, coeffs)
            print('Initial Ordering')
            print(spo)

            # Lexicographic Ordering
            srt = spo.argsort()
            print('Lexicographically sorted')
            print(srt)

            # Lexicographic Ordering
            srt = spo.argsort(weight=False)
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = spo.argsort(weight=True)
            print('Weight sorted')
            print(srt)

        .. code-block:: text

            Initial Ordering
            SparsePauliOp(['XX', 'XX', 'XX', 'YI', 'II', 'XZ', 'XY', 'XI'],
                          coeffs=[2.+1.j, 2.+2.j, 3.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j])
            Lexicographically sorted
            [4 7 0 1 2 6 5 3]
            Lexicographically sorted
            [4 7 0 1 2 6 5 3]
            Weight sorted
            [4 7 3 0 1 2 6 5]

        Args:
            weight (bool): optionally sort by weight if True (Default: False).
            By using the weight kwarg the output can additionally be sorted
            by the number of non-identity terms in the Pauli.

        Returns:
            array: the indices for sorting the table.
        """
        sort_coeffs_inds = np.argsort(self._coeffs, kind="stable")
        pauli_list = self._pauli_list[sort_coeffs_inds]
        sort_pauli_inds = pauli_list.argsort(weight=weight, phase=False)
        return sort_coeffs_inds[sort_pauli_inds]

    def sort(self, weight: bool = False):
        """Sort the rows of the table.

        After sorting the coefficients using numpy's argsort, sort by Pauli.
        Pauli sort takes precedence.
        If Pauli is the same, it will be sorted by coefficient.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        **Example**

        Here is an example of how to use SparsePauliOp sort.

        .. plot::
           :include-source:
           :nofigs:

            import numpy as np
            from qiskit.quantum_info import SparsePauliOp

            # 2-qubit labels
            labels = ["XX", "XX", "XX", "YI", "II", "XZ", "XY", "XI"]
            # coeffs
            coeffs = [2.+1.j, 2.+2.j, 3.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j]

            # init
            spo = SparsePauliOp(labels, coeffs)
            print('Initial Ordering')
            print(spo)

            # Lexicographic Ordering
            srt = spo.sort()
            print('Lexicographically sorted')
            print(srt)

            # Lexicographic Ordering
            srt = spo.sort(weight=False)
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = spo.sort(weight=True)
            print('Weight sorted')
            print(srt)

        .. code-block:: text

            Initial Ordering
            SparsePauliOp(['XX', 'XX', 'XX', 'YI', 'II', 'XZ', 'XY', 'XI'],
                          coeffs=[2.+1.j, 2.+2.j, 3.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j])
            Lexicographically sorted
            SparsePauliOp(['II', 'XI', 'XX', 'XX', 'XX', 'XY', 'XZ', 'YI'],
                          coeffs=[4.+0.j, 7.+0.j, 2.+1.j, 2.+2.j, 3.+0.j, 6.+0.j, 5.+0.j, 3.+0.j])
            Lexicographically sorted
            SparsePauliOp(['II', 'XI', 'XX', 'XX', 'XX', 'XY', 'XZ', 'YI'],
                          coeffs=[4.+0.j, 7.+0.j, 2.+1.j, 2.+2.j, 3.+0.j, 6.+0.j, 5.+0.j, 3.+0.j])
            Weight sorted
            SparsePauliOp(['II', 'XI', 'YI', 'XX', 'XX', 'XX', 'XY', 'XZ'],
                          coeffs=[4.+0.j, 7.+0.j, 3.+0.j, 2.+1.j, 2.+2.j, 3.+0.j, 6.+0.j, 5.+0.j])

        Args:
            weight (bool): optionally sort by weight if True (Default: False).
            By using the weight kwarg the output can additionally be sorted
            by the number of non-identity terms in the Pauli.

        Returns:
            SparsePauliOp: a sorted copy of the original table.
        """
        indices = self.argsort(weight=weight)
        return SparsePauliOp(self._pauli_list[indices], self._coeffs[indices])

    def chop(self, tol: float = 1e-14) -> SparsePauliOp:
        """Set real and imaginary parts of the coefficients to 0 if ``< tol`` in magnitude.

        For example, the operator representing ``1+1e-17j X + 1e-17 Y`` with a tolerance larger
        than ``1e-17`` will be reduced to ``1 X`` whereas :meth:`.SparsePauliOp.simplify` would
        return ``1+1e-17j X``.

        If both the real and imaginary part of a coefficient is 0 after chopping, the
        corresponding Pauli is removed from the operator.

        Args:
            tol (float): The absolute tolerance to check whether a real or imaginary part should
                be set to 0.

        Returns:
            SparsePauliOp: This operator with chopped coefficients.
        """
        realpart_nonzero = np.abs(self.coeffs.real) > tol
        imagpart_nonzero = np.abs(self.coeffs.imag) > tol

        remaining_indices = np.logical_or(realpart_nonzero, imagpart_nonzero)
        remaining_real = realpart_nonzero[remaining_indices]
        remaining_imag = imagpart_nonzero[remaining_indices]

        if len(remaining_real) == 0:  # if no Paulis are left
            x = np.zeros((1, self.num_qubits), dtype=bool)
            z = np.zeros((1, self.num_qubits), dtype=bool)
            coeffs = np.array([0j], dtype=complex)
        else:
            coeffs = np.zeros(np.count_nonzero(remaining_indices), dtype=complex)
            coeffs.real[remaining_real] = self.coeffs.real[realpart_nonzero]
            coeffs.imag[remaining_imag] = self.coeffs.imag[imagpart_nonzero]

            x = self.paulis.x[remaining_indices]
            z = self.paulis.z[remaining_indices]

        return SparsePauliOp(
            PauliList.from_symplectic(z, x), coeffs, ignore_pauli_phase=True, copy=False
        )

    @staticmethod
    def sum(ops: list[SparsePauliOp]) -> SparsePauliOp:
        """Sum of SparsePauliOps.

        This is a specialized version of the builtin ``sum`` function for SparsePauliOp
        with smaller overhead.

        Args:
            ops (list[SparsePauliOp]): a list of SparsePauliOps.

        Returns:
            SparsePauliOp: the SparsePauliOp representing the sum of the input list.

        Raises:
            QiskitError: if the input list is empty.
            QiskitError: if the input list includes an object that is not SparsePauliOp.
            QiskitError: if the numbers of qubits of the objects in the input list do not match.
        """
        if len(ops) == 0:
            raise QiskitError("Input list is empty")
        if not all(isinstance(op, SparsePauliOp) for op in ops):
            raise QiskitError("Input list includes an object that is not SparsePauliOp")
        for other in ops[1:]:
            ops[0]._op_shape._validate_add(other._op_shape)

        z = np.vstack([op.paulis.z for op in ops])
        x = np.vstack([op.paulis.x for op in ops])
        phase = np.hstack([op.paulis._phase for op in ops])
        pauli_list = PauliList(BasePauli(z, x, phase))
        coeffs = np.hstack([op.coeffs for op in ops])
        return SparsePauliOp(pauli_list, coeffs, ignore_pauli_phase=True, copy=False)

    # ---------------------------------------------------------------------
    # Additional conversions
    # ---------------------------------------------------------------------

    @staticmethod
    def from_operator(
        obj: Operator, atol: float | None = None, rtol: float | None = None
    ) -> SparsePauliOp:
        """Construct from an Operator object.

        Note that the cost of this construction is exponential in general because the number of
        possible Pauli terms in the decomposition is exponential in the number of qubits.

        Internally this uses an implementation of the "tensorized Pauli decomposition" presented in
        `Hantzko, Binkowski and Gupta (2023) <https://arxiv.org/abs/2310.13421>`__.

        Args:
            obj (Operator): an N-qubit operator.
            atol (float): Optional. Absolute tolerance for checking if coefficients are zero
                (Default: 1e-8).  Since the comparison is to zero, in effect the tolerance used is
                the maximum of ``atol`` and ``rtol``.
            rtol (float): Optional. relative tolerance for checking if coefficients are zero
                (Default: 1e-5).  Since the comparison is to zero, in effect the tolerance used is
                the maximum of ``atol`` and ``rtol``.

        Returns:
            SparsePauliOp: the SparsePauliOp representation of the operator.

        Raises:
            QiskitError: if the input operator is not an N-qubit operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = SparsePauliOp.atol
        if rtol is None:
            rtol = SparsePauliOp.rtol
        tol = max((atol, rtol))

        if not isinstance(obj, Operator):
            obj = Operator(obj)

        # Check dimension is N-qubit operator
        num_qubits = obj.num_qubits
        if num_qubits is None:
            raise QiskitError("Input Operator is not an N-qubit operator.")
        zx_paulis = decompose_dense(obj.data, tol)
        # Indirection through `BasePauli` is because we're already supplying the phase in the ZX
        # convention.
        pauli_list = PauliList(BasePauli(zx_paulis.z, zx_paulis.x, zx_paulis.phases))
        return SparsePauliOp(pauli_list, zx_paulis.coeffs, ignore_pauli_phase=True, copy=False)

    @staticmethod
    def from_list(
        obj: Iterable[tuple[str, complex]],
        dtype: type | None = None,
        *,
        num_qubits: int | None = None,
    ) -> SparsePauliOp:
        """Construct from a list of Pauli strings and coefficients.

        For example, the 5-qubit Hamiltonian

        .. math::

            H = Z_1 X_4 + 2 Y_0 Y_3

        can be constructed as

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.quantum_info import SparsePauliOp

            # via tuples and the full Pauli string
            op = SparsePauliOp.from_list([("XIIZI", 1), ("IYIIY", 2)])

        Args:
            obj (Iterable[Tuple[str, complex]]): The list of 2-tuples specifying the Pauli terms.
            dtype (type | None): Data type for the coefficients. If ``None`` (default), the dtype is
                automatically inferred.
            num_qubits (int): The number of qubits of the operator (Default: None).

        Returns:
            SparsePauliOp: The SparsePauliOp representation of the Pauli terms.

        Raises:
            QiskitError: If an empty list is passed and num_qubits is None.
            QiskitError: If num_qubits and the objects in the input list do not match.
        """
        obj = list(obj)  # To convert zip or other iterable
        size = len(obj)

        if size == 0 and num_qubits is None:
            raise QiskitError(
                "Could not determine the number of qubits from an empty list. Try passing num_qubits."
            )
        if size > 0 and num_qubits is not None:
            if len(obj[0][0]) != num_qubits:
                raise QiskitError(
                    f"num_qubits ({num_qubits}) and the objects in the input list do not match."
                )
        if num_qubits is None:
            num_qubits = len(obj[0][0])
        if size == 0:
            obj = [("I" * num_qubits, 0)]
            size = len(obj)

        if dtype is None:
            if any(isinstance(coeff, ParameterExpression) for (_, coeff) in obj):
                dtype = object
            else:
                dtype = complex

        coeffs = np.zeros(size, dtype=dtype)
        labels = np.zeros(size, dtype=f"<U{num_qubits}")
        for i, item in enumerate(obj):
            labels[i] = item[0]
            coeffs[i] = item[1]

        paulis = PauliList(labels)
        return SparsePauliOp(paulis, coeffs, copy=False)

    @staticmethod
    def from_sparse_list(
        obj: Iterable[tuple[str, list[int], complex]],
        num_qubits: int,
        do_checks: bool = True,
        dtype: type | None = None,
    ) -> SparsePauliOp:
        """Construct from a list of local Pauli strings and coefficients.

        Each list element is a 3-tuple of a local Pauli string, indices where to apply it,
        and a coefficient.

        For example, the 5-qubit Hamiltonian

        .. math::

            H = Z_1 X_4 + 2 Y_0 Y_3

        can be constructed as

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.quantum_info import SparsePauliOp

            # via triples and local Paulis with indices
            op = SparsePauliOp.from_sparse_list([("ZX", [1, 4], 1), ("YY", [0, 3], 2)], num_qubits=5)

            # equals the following construction from "dense" Paulis
            op = SparsePauliOp.from_list([("XIIZI", 1), ("IYIIY", 2)])

        Args:
            obj (Iterable[tuple[str, list[int], complex]]): The list 3-tuples specifying the Paulis.
            num_qubits (int): The number of qubits of the operator.
            do_checks (bool): Whether to perform validity checks on the input indices.
            dtype (type | None): Data type for the coefficients. If ``None`` (default), the dtype is
                automatically inferred.


        Returns:
            SparsePauliOp: The SparsePauliOp representation of the Pauli terms.

        Raises:
            QiskitError: If the number of qubits is incompatible with the indices of the Pauli terms.
            QiskitError: If the designated qubit is already assigned.
        """
        obj = list(obj)  # To convert zip or other iterable
        size = len(obj)

        if size == 0:
            obj = [("I" * num_qubits, range(num_qubits), 0)]
            size = len(obj)

        if dtype is None:
            if any(isinstance(coeff, ParameterExpression) for (_, _, coeff) in obj):
                dtype = object
            else:
                dtype = complex

        coeffs = np.zeros(size, dtype=dtype)
        labels = np.zeros(size, dtype=f"<U{num_qubits}")

        for i, (paulis, indices, coeff) in enumerate(obj):
            if do_checks and len(indices) != len(set(indices)):
                raise QiskitError("Input indices are duplicated.")
            # construct the full label based off the non-trivial Paulis and indices
            label = ["I"] * num_qubits
            for pauli, index in zip(paulis, indices):
                if index >= num_qubits:
                    raise QiskitError(
                        f"The number of qubits ({num_qubits}) is smaller than a required index {index}."
                    )
                label[~index] = pauli

            labels[i] = "".join(label)
            coeffs[i] = coeff

        paulis = PauliList(labels)
        return SparsePauliOp(paulis, coeffs, copy=False)

    @staticmethod
    def from_sparse_observable(obs: SparseObservable) -> SparsePauliOp:
        r"""Initialize from a :class:`.SparseObservable`.

        .. warning::

            A :class:`.SparseObservable` can efficiently represent eigenstate projectors
            (such as :math:`|0\langle\rangle 0|`), but a :class:`.SparsePauliOp` **cannot**.
            If the input ``obs`` has :math:`n` single-qubit projectors, the resulting
            :class:`.SparsePauliOp` will use :math:`2^n` terms, which is an exponentially
            expensive representation that can quickly run out of memory.

        Args:
            obs: The :class:`.SparseObservable` to convert.

        Returns:
            A :class:`.SparsePauliOp` version of the observable.
        """
        as_sparse_list = obs.as_paulis().to_sparse_list()
        return SparsePauliOp.from_sparse_list(as_sparse_list, obs.num_qubits)

    def to_list(self, array: bool = False):
        """Convert to a list Pauli string labels and coefficients.

        For operators with a lot of terms converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: List of pairs (label, coeff) for rows of the PauliList.
        """
        # Dtype for a structured array with string labels and complex coeffs
        pauli_labels = self.paulis.to_labels(array=True)
        labels = np.zeros(
            self.size, dtype=[("labels", pauli_labels.dtype), ("coeffs", self.coeffs.dtype)]
        )
        labels["labels"] = pauli_labels
        labels["coeffs"] = self.coeffs
        if array:
            return labels
        return labels.tolist()

    def to_sparse_list(self):
        """Convert to a sparse Pauli list format with elements (pauli, qubits, coefficient)."""
        pauli_labels = self.paulis.to_labels()
        sparse_list = [
            (*sparsify_label(label), coeff) for label, coeff in zip(pauli_labels, self.coeffs)
        ]
        return sparse_list

    def to_matrix(self, sparse: bool = False, force_serial: bool = False) -> np.ndarray:
        """Convert to a dense or sparse matrix.

        Args:
            sparse: if ``True`` return a sparse CSR matrix, otherwise return dense Numpy
                array (the default).
            force_serial: if ``True``, use an unthreaded implementation, regardless of the state of
                the `Qiskit threading-control environment variables
                <https://quantum.cloud.ibm.com/docs/guides/configure-qiskit-local#environment-variables>`__.
                By default, this will use threaded parallelism over the available CPUs.

        Returns:
            array: A dense matrix if `sparse=False`.
            csr_matrix: A sparse matrix in CSR format if `sparse=True`.
        """
        if self.coeffs.dtype == object:
            # Fallback to slow Python-space method.
            return sum(self.matrix_iter(sparse=sparse))

        pauli_list = self.paulis
        zx = ZXPaulis(
            pauli_list.x.astype(np.bool_),
            pauli_list.z.astype(np.bool_),
            pauli_list.phase.astype(np.uint8),
            self.coeffs.astype(np.complex128),
        )
        if sparse:
            from scipy.sparse import csr_matrix

            data, indices, indptr = to_matrix_sparse(zx, force_serial=force_serial)
            side = 1 << self.num_qubits
            return csr_matrix((data, indices, indptr), shape=(side, side))
        return to_matrix_dense(zx, force_serial=force_serial)

    def to_operator(self) -> Operator:
        """Convert to a matrix Operator object"""
        return Operator(self.to_matrix())

    # ---------------------------------------------------------------------
    # Custom Iterators
    # ---------------------------------------------------------------------

    def label_iter(self):
        """Return a label representation iterator.

        This is a lazy iterator that converts each term in the SparsePauliOp
        into a tuple (label, coeff). To convert the entire table to labels
        use the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the SparsePauliOp.
        """

        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""

            def __repr__(self):
                return f"<SparsePauliOp_label_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                pauli = self.obj.paulis.label_iter()[key]
                return (pauli, coeff)

        return LabelIterator(self)

    def matrix_iter(self, sparse: bool = False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each term in the SparsePauliOp
        into a matrix as it is used. To convert to a single matrix use the
        :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if True,
                           otherwise return Numpy array matrices
                           (Default: False)

        Returns:
            MatrixIterator: matrix iterator object for the PauliList.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return f"<SparsePauliOp_matrix_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                mat = self.obj.paulis[key].to_matrix(sparse)
                return coeff * mat

        return MatrixIterator(self)

    def noncommutation_graph(self, qubit_wise: bool) -> rx.PyGraph:
        """Create the non-commutation graph of this SparsePauliOp.

        This transforms the measurement operator grouping problem into graph coloring problem. The
        constructed graph contains one node for each Pauli. The nodes will be connecting for any two
        Pauli terms that do _not_ commute.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.

        Returns:
            rustworkx.PyGraph: the non-commutation graph with nodes for each Pauli and edges
                indicating a non-commutation relation. Each node will hold the index of the Pauli
                term it corresponds to in its data. The edges of the graph hold no data.
        """
        return self.paulis.noncommutation_graph(qubit_wise)

    def group_commuting(self, qubit_wise: bool = False) -> list[SparsePauliOp]:
        """Partition a SparsePauliOp into sets of commuting Pauli strings.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.  For example:

                .. plot::
                   :include-source:
                   :nofigs:

                    >>> from qiskit.quantum_info import SparsePauliOp
                    >>> op = SparsePauliOp.from_list([("XX", 2), ("YY", 1), ("IZ",2j), ("ZZ",1j)])
                    >>> op.group_commuting()
                    [SparsePauliOp(["IZ", "ZZ"], coeffs=[0.+2.j, 0.+1j]),
                     SparsePauliOp(["XX", "YY"], coeffs=[2.+0.j, 1.+0.j])]
                    >>> op.group_commuting(qubit_wise=True)
                    [SparsePauliOp(['XX'], coeffs=[2.+0.j]),
                     SparsePauliOp(['YY'], coeffs=[1.+0.j]),
                     SparsePauliOp(['IZ', 'ZZ'], coeffs=[0.+2.j, 0.+1.j])]

        Returns:
            list[SparsePauliOp]: List of SparsePauliOp where each SparsePauliOp contains
                commuting Pauli operators.
        """
        groups = self.paulis._commuting_groups(qubit_wise)
        return [self[group] for group in groups.values()]

    @property
    def parameters(self) -> ParameterView:
        r"""Return the free ``Parameter``\s in the coefficients."""
        ret = set()
        for coeff in self.coeffs:
            if isinstance(coeff, ParameterExpression):
                ret |= coeff.parameters
        return ParameterView(ret)

    def assign_parameters(
        self,
        parameters: (
            Mapping[Parameter, complex | ParameterExpression]
            | Sequence[complex | ParameterExpression]
        ),
        inplace: bool = False,
    ) -> SparsePauliOp | None:
        r"""Bind the free ``Parameter``\s in the coefficients to provided values.

        .. note::
            If all the parameters in the circuit are bound to numeric values, the coefficients array
            will be returned with a :class:`complex` dtype.

        Args:
            parameters: The values to bind the parameters to.
            inplace: If ``False``, a copy of the operator with the bound parameters is returned.
                If ``True`` the operator itself is modified.

        Returns:
            A copy of the operator with bound parameters, if ``inplace`` is ``False``, otherwise
            ``None``.
        """
        if inplace:
            bound = self
        else:
            bound = deepcopy(self)

        # turn the parameters to a dictionary
        if isinstance(parameters, Sequence):
            free_parameters = bound.parameters
            if len(parameters) != len(free_parameters):
                raise ValueError(
                    f"Mismatching number of values ({len(parameters)}) and parameters "
                    f"({len(free_parameters)}). For partial binding please pass a dictionary of "
                    "{parameter: value} pairs."
                )
            parameters = dict(zip(free_parameters, parameters))

        for i, coeff in enumerate(bound.coeffs):
            if isinstance(coeff, ParameterExpression):
                for key in coeff.parameters & parameters.keys():
                    coeff = coeff.assign(key, parameters[key])
                if len(coeff.parameters) == 0:
                    coeff = complex(coeff)
                bound.coeffs[i] = coeff

        if bound.coeffs.dtype == object and not any(
            isinstance(c, ParameterExpression) for c in bound.coeffs
        ):
            bound._coeffs = bound.coeffs.astype(complex)

        return None if inplace else bound

    def apply_layout(
        self, layout: TranspileLayout | list[int] | None, num_qubits: int | None = None
    ) -> SparsePauliOp:
        """Apply a transpiler layout to this :class:`~.SparsePauliOp`

        Args:
            layout: Either a :class:`~.TranspileLayout`, a list of integers or None.
                    If both layout and num_qubits are none, a copy of the operator is
                    returned.
            num_qubits: The number of qubits to expand the operator to. If not
                provided then if ``layout`` is a :class:`~.TranspileLayout` the
                number of the transpiler output circuit qubits will be used by
                default. If ``layout`` is a list of integers the permutation
                specified will be applied without any expansion. If layout is
                None, the operator will be expanded to the given number of qubits.

        Returns:
            A new :class:`.SparsePauliOp` with the provided layout applied
        """
        from qiskit.transpiler.layout import TranspileLayout

        if layout is None and num_qubits is None:
            return self.copy()

        n_qubits = self.num_qubits
        if isinstance(layout, TranspileLayout):
            n_qubits = len(layout._output_qubit_list)
            layout = layout.final_index_layout()
        if num_qubits is not None:
            if num_qubits < n_qubits:
                raise QiskitError(
                    f"The input num_qubits is too small, a {num_qubits} qubit layout cannot be "
                    f"applied to a {n_qubits} qubit operator"
                )
            n_qubits = num_qubits
        if layout is None:
            layout = list(range(self.num_qubits))
        else:
            if any(x < 0 or x >= n_qubits for x in layout):
                raise QiskitError("Provided layout contains indices outside the number of qubits.")
            if len(set(layout)) != len(layout):
                raise QiskitError("Provided layout contains duplicate indices.")
        if self.num_qubits == 0:
            return type(self)(["I" * n_qubits] * self.size, self.coeffs)
        new_op = type(self)("I" * n_qubits)
        return new_op.compose(self, qargs=layout)


def sparsify_label(pauli_string):
    """Return a sparse format of a Pauli string, e.g. "XIIIZ" -> ("XZ", [0, 4])."""
    qubits = [i for i, label in enumerate(reversed(pauli_string)) if label != "I"]
    sparse_label = "".join(pauli_string[~i] for i in qubits)

    return sparse_label, qubits


# Update docstrings for API docs
generate_apidocs(SparsePauliOp)
