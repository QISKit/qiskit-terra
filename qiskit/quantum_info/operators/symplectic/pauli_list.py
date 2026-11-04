# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Optimized list of Pauli operators
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

import numpy as np
import rustworkx as rx

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import GroupMixin, LinearMixin
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.clifford import Clifford
from qiskit.quantum_info.operators.symplectic.pauli import Pauli


class PauliList(BasePauli, LinearMixin, GroupMixin):
    r"""List of N-qubit Pauli operators.

    This class is an efficient representation of a list of
    :class:`Pauli` operators. It supports 1D numpy array indexing
    returning a :class:`Pauli` for integer indexes or a
    :class:`PauliList` for slice or list indices.

    **Initialization**

    A PauliList object can be initialized in several ways.

        ``PauliList(list[str])``
            where strings are same representation with :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList(Pauli) and PauliList(list[Pauli])``
            where Pauli is :class:`~qiskit.quantum_info.Pauli`.

        ``PauliList.from_symplectic(z, x, phase)``
            where ``z`` and ``x`` are 2 dimensional boolean ``numpy.ndarrays`` and ``phase`` is
            an integer in ``[0, 1, 2, 3]``.

    For example,

    .. plot::
       :include-source:
       :nofigs:
       :context: reset

        import numpy as np

        from qiskit.quantum_info import Pauli, PauliList

        # 1. init from list[str]
        pauli_list = PauliList(["II", "+ZI", "-iYY"])
        print("1. ", pauli_list)

        pauli1 = Pauli("iXI")
        pauli2 = Pauli("iZZ")

        # 2. init from Pauli
        print("2. ", PauliList(pauli1))

        # 3. init from list[Pauli]
        print("3. ", PauliList([pauli1, pauli2]))

        # 4. init from np.ndarray
        z = np.array([[True, True], [False, False]])
        x = np.array([[False, True], [True, False]])
        phase = np.array([0, 1])
        pauli_list = PauliList.from_symplectic(z, x, phase)
        print("4. ", pauli_list)

    .. code-block:: text

        1.  ['II', 'ZI', '-iYY']
        2.  ['iXI']
        3.  ['iXI', 'iZZ']
        4.  ['YZ', '-iIX']

    **Data Access**

    The individual Paulis can be accessed and updated using the ``[]``
    operator which accepts integer, lists, or slices for selecting subsets
    of PauliList. If integer is given, it returns Pauli not PauliList.

    .. plot::
       :include-source:
       :nofigs:
       :context:

        pauli_list = PauliList(["XX", "ZZ", "IZ"])
        print("Integer: ", repr(pauli_list[1]))
        print("List: ", repr(pauli_list[[0, 2]]))
        print("Slice: ", repr(pauli_list[0:2]))

    .. code-block:: text

        Integer:  Pauli('ZZ')
        List:  PauliList(['XX', 'IZ'])
        Slice:  PauliList(['XX', 'ZZ'])

    **Iteration**

    Rows in the Pauli table can be iterated over like a list. Iteration can
    also be done using the label or matrix representation of each row using the
    :meth:`label_iter` and :meth:`matrix_iter` methods.
    """

    # Set the max number of qubits * paulis before string truncation
    __truncate__ = 2000

    def __init__(self, data: Pauli | list):
        """Initialize the PauliList.

        Args:
            data (Pauli or list): input data for Paulis. If input is a list each item in the list
                                  must be a Pauli object or Pauli str.

        Raises:
            QiskitError: if input array is invalid shape.

        Additional Information:
            The input array is not copied so multiple Pauli tables
            can share the same underlying array.
        """
        if isinstance(data, BasePauli):
            base_z, base_x, base_phase = data._z, data._x, data._phase
        else:
            # Conversion as iterable of Paulis
            base_z, base_x, base_phase = self._from_paulis(data)

        # Initialize BasePauli
        super().__init__(base_z, base_x, base_phase)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    @property
    def settings(self):
        """Return settings."""
        return {"data": self.to_labels()}

    def __array__(self, dtype=None, copy=None):
        """Convert to numpy array"""
        if copy is False:
            raise ValueError("cannot provide a matrix without calculation")
        shape = (len(self),) + 2 * (2**self.num_qubits,)
        ret = np.zeros(shape, dtype=complex)
        for i, mat in enumerate(self.matrix_iter()):
            ret[i] = mat
        return ret if dtype is None else ret.astype(dtype, copy=False)

    @staticmethod
    def _from_paulis(data):
        """Construct a PauliList from a list of Pauli data.

        Args:
            data (iterable): list of Pauli data.

        Returns:
            PauliList: the constructed PauliList.

        Raises:
            QiskitError: If the input list is empty or contains invalid
            Pauli strings.
        """
        if not isinstance(data, (list, tuple, set, np.ndarray)):
            data = [data]
        num_paulis = len(data)
        if num_paulis == 0:
            raise QiskitError("Input Pauli list is empty.")
        paulis = []
        for i in data:
            if not isinstance(i, Pauli):
                paulis.append(Pauli(i))
            else:
                paulis.append(i)
        num_qubits = paulis[0].num_qubits
        base_z = np.zeros((num_paulis, num_qubits), dtype=bool)
        base_x = np.zeros((num_paulis, num_qubits), dtype=bool)
        base_phase = np.zeros(num_paulis, dtype=int)
        for i, pauli in enumerate(paulis):
            if pauli.num_qubits != num_qubits:
                raise ValueError(
                    f"The {i}th Pauli is defined over {pauli.num_qubits} qubits, "
                    f"but num_qubits == {num_qubits} was expected."
                )
            base_z[i] = pauli._z
            base_x[i] = pauli._x
            base_phase[i] = pauli._phase.item()
        return base_z, base_x, base_phase

    def __repr__(self):
        """Display representation."""
        return self._truncated_str(True)

    def __str__(self):
        """Print representation."""
        return self._truncated_str(False)

    def _truncated_str(self, show_class):
        stop = self._num_paulis
        if self.__truncate__ and self.num_qubits > 0:
            max_paulis = self.__truncate__ // self.num_qubits
            if self._num_paulis > max_paulis:
                stop = max_paulis
        labels = [str(self[i]) for i in range(stop)]
        prefix = "PauliList(" if show_class else ""
        tail = ")" if show_class else ""
        if stop != self._num_paulis:
            suffix = ", ...]" + tail
        else:
            suffix = "]" + tail
        list_str = np.array2string(
            np.array(labels), threshold=stop + 1, separator=", ", prefix=prefix, suffix=suffix
        )
        return prefix + list_str[:-1] + suffix

    def __eq__(self, other):
        """Entrywise comparison of Pauli equality."""
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if not isinstance(other, BasePauli):
            return False
        return self._eq(other)

    def equiv(self, other: PauliList | Pauli) -> np.ndarray:
        """Entrywise comparison of Pauli equivalence up to global phase.

        Args:
            other (PauliList or Pauli): a comparison object.

        Returns:
            np.ndarray: An array of ``True`` or ``False`` for entrywise equivalence
                        of the current table.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        return np.all(self.z == other.z, axis=1) & np.all(self.x == other.x, axis=1)

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------
    @property
    def phase(self):
        """Return the phase exponent of the PauliList."""
        # Convert internal ZX-phase convention to group phase convention
        return np.mod(self._phase - self._count_y(dtype=self._phase.dtype), 4)

    @phase.setter
    def phase(self, value):
        # Convert group phase convention to internal ZX-phase convention
        self._phase[:] = np.mod(value + self._count_y(dtype=self._phase.dtype), 4)

    @property
    def x(self):
        """The x array for the symplectic representation."""
        return self._x

    @x.setter
    def x(self, val):
        self._x[:] = val

    @property
    def z(self):
        """The z array for the symplectic representation."""
        return self._z

    @z.setter
    def z(self, val):
        self._z[:] = val

    # ---------------------------------------------------------------------
    # Size Properties
    # ---------------------------------------------------------------------

    @property
    def shape(self):
        """The full shape of the :meth:`array`"""
        return self._num_paulis, self.num_qubits

    @property
    def size(self):
        """The number of Pauli rows in the table."""
        return self._num_paulis

    def __len__(self):
        """Return the number of Pauli rows in the table."""
        return self._num_paulis

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __getitem__(self, index):
        """Return a view of the PauliList."""
        # Returns a view of specified rows of the PauliList
        # This supports all slicing operations the underlying array supports.
        if isinstance(index, tuple):
            if len(index) == 1:
                index = index[0]
            elif len(index) > 2:
                raise IndexError(f"Invalid PauliList index {index}")

        # Row-only indexing
        if isinstance(index, (int, np.integer)):
            # Single Pauli
            return Pauli(
                BasePauli(
                    self._z[np.newaxis, index],
                    self._x[np.newaxis, index],
                    self._phase[np.newaxis, index],
                )
            )
        elif isinstance(index, (slice, list, np.ndarray)):
            # Sub-Table view
            return PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))

        # Row and Qubit indexing
        return PauliList((self._z[index], self._x[index], 0))

    def __setitem__(self, index, value):
        """Update PauliList."""
        if isinstance(index, tuple):
            if len(index) == 1:
                row, qubit = index[0], None
            elif len(index) > 2:
                raise IndexError(f"Invalid PauliList index {index}")
            else:
                row, qubit = index
        else:
            row, qubit = index, None

        # Modify specified rows of the PauliList
        if not isinstance(value, PauliList):
            value = PauliList(value)

        # It's not valid to set a single item with a sequence, even if the sequence is length 1.
        phase = value._phase.item() if isinstance(row, (int, np.integer)) else value._phase

        if qubit is None:
            self._z[row] = value._z
            self._x[row] = value._x
            self._phase[row] = phase
        else:
            self._z[row, qubit] = value._z
            self._x[row, qubit] = value._x
            self._phase[row] += phase
            self._phase %= 4

    def delete(self, ind: int | list, qubit: bool = False) -> PauliList:
        """Return a copy with Pauli rows deleted from table.

        When deleting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int or list): index(es) to delete.
            qubit (bool): if ``True`` delete qubit columns, otherwise delete
                          Pauli rows (Default: ``False``).

        Returns:
            PauliList: the resulting table with the entries removed.

        Raises:
            QiskitError: if ``ind`` is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(ind, int):
            ind = [ind]
        if len(ind) == 0:
            return PauliList.from_symplectic(self._z, self._x, self.phase)
        # Row deletion
        if not qubit:
            if max(ind) >= len(self):
                raise QiskitError(
                    f"Indices {ind} are not all less than the size"
                    f" of the PauliList ({len(self)})"
                )
            z = np.delete(self._z, ind, axis=0)
            x = np.delete(self._x, ind, axis=0)
            phase = np.delete(self._phase, ind)

            return PauliList(BasePauli(z, x, phase))

        # Column (qubit) deletion
        if max(ind) >= self.num_qubits:
            raise QiskitError(
                f"Indices {ind} are not all less than the number of"
                f" qubits in the PauliList ({self.num_qubits})"
            )
        z = np.delete(self._z, ind, axis=1)
        x = np.delete(self._x, ind, axis=1)
        # Use self.phase, not self._phase as deleting qubits can change the
        # ZX phase convention
        return PauliList.from_symplectic(z, x, self.phase)

    def insert(self, ind: int, value: PauliList, qubit: bool = False) -> PauliList:
        """Insert Paulis into the table.

        When inserting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int): index to insert at.
            value (PauliList): values to insert.
            qubit (bool): if ``True`` insert qubit columns, otherwise insert
                          Pauli rows (Default: ``False``).

        Returns:
            PauliList: the resulting table with the entries inserted.

        Raises:
            QiskitError: if the insertion index is invalid.
        """
        if not isinstance(ind, int):
            raise QiskitError("Insert index must be an integer.")

        if not isinstance(value, PauliList):
            value = PauliList(value)

        # Row insertion
        size = self._num_paulis
        if not qubit:
            if ind > size:
                raise QiskitError(
                    f"Index {ind} is larger than the number of rows in the" f" PauliList ({size})."
                )
            base_z = np.insert(self._z, ind, value._z, axis=0)
            base_x = np.insert(self._x, ind, value._x, axis=0)
            base_phase = np.insert(self._phase, ind, value._phase)
            return PauliList(BasePauli(base_z, base_x, base_phase))

        # Column insertion
        if ind > self.num_qubits:
            raise QiskitError(
                f"Index {ind} is greater than number of qubits"
                f" in the PauliList ({self.num_qubits})"
            )
        if len(value) == size:
            #  Blocks are already correct size
            value_x = value.x
            value_z = value.z
        elif len(value) == 1:
            # Pad blocks to correct size
            value_x = np.vstack(size * [value.x])
            value_z = np.vstack(size * [value.z])
        else:
            # Blocks are incorrect size
            raise QiskitError(
                "Input PauliList must have a single row, or"
                " the same number of rows as the Pauli Table"
                f" ({size})."
            )
        # Build new array by blocks
        z = np.hstack([self.z[:, :ind], value_z, self.z[:, ind:]])
        x = np.hstack([self.x[:, :ind], value_x, self.x[:, ind:]])
        phase = self.phase + value.phase

        return PauliList.from_symplectic(z, x, phase)

    def argsort(self, weight: bool = False, phase: bool = False) -> np.ndarray:
        """Return indices for sorting the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Paulis of a given weight are still ordered lexicographically.

        Args:
            weight (bool): Optionally sort by weight if ``True`` (Default: ``False``).
            phase (bool): Optionally sort by phase before weight or order
                          (Default: ``False``).

        Returns:
            array: the indices for sorting the table.
        """
        # Get order of each Pauli using
        # I => 0, X => 1, Y => 2, Z => 3
        x = self.x
        z = self.z
        order = 1 * (x & ~z) + 2 * (x & z) + 3 * (~x & z)
        phases = self.phase
        # Optionally get the weight of Pauli
        # This is the number of non identity terms
        if weight:
            weights = np.sum(x | z, axis=1)

        # To preserve ordering between successive sorts we
        # are use the 'stable' sort method
        indices = np.arange(self._num_paulis)

        # Initial sort by phases
        sort_inds = phases.argsort(kind="stable")
        indices = indices[sort_inds]
        order = order[sort_inds]
        if phase:
            phases = phases[sort_inds]
        if weight:
            weights = weights[sort_inds]

        # Sort by order
        for i in range(self.num_qubits):
            sort_inds = order[:, i].argsort(kind="stable")
            order = order[sort_inds]
            indices = indices[sort_inds]
            if weight:
                weights = weights[sort_inds]
            if phase:
                phases = phases[sort_inds]

        # If using weights we implement a sort by total number
        # of non-identity Paulis
        if weight:
            sort_inds = weights.argsort(kind="stable")
            indices = indices[sort_inds]
            phases = phases[sort_inds]

        # If sorting by phase we perform a final sort by the phase value
        # of each pauli
        if phase:
            indices = indices[phases.argsort(kind="stable")]
        return indices

    def sort(self, weight: bool = False, phase: bool = False) -> PauliList:
        """Sort the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Paulis of a given weight are still ordered lexicographically.

        **Example**

        Consider sorting all a random ordering of all 2-qubit Paulis

        .. plot::
           :include-source:
           :nofigs:

            from numpy.random import shuffle
            from qiskit.quantum_info.operators import PauliList

            # 2-qubit labels
            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                      'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
            # Shuffle Labels
            shuffle(labels)
            pt = PauliList(labels)
            print('Initial Ordering')
            print(pt)

            # Lexicographic Ordering
            srt = pt.sort()
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = pt.sort(weight=True)
            print('Weight sorted')
            print(srt)

        .. code-block:: text

            Initial Ordering
            ['YX', 'ZZ', 'XZ', 'YI', 'YZ', 'II', 'XX', 'XI', 'XY', 'YY', 'IX', 'IZ',
             'ZY', 'ZI', 'ZX', 'IY']
            Lexicographically sorted
            ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ', 'YI', 'YX', 'YY', 'YZ',
             'ZI', 'ZX', 'ZY', 'ZZ']
            Weight sorted
            ['II', 'IX', 'IY', 'IZ', 'XI', 'YI', 'ZI', 'XX', 'XY', 'XZ', 'YX', 'YY',
             'YZ', 'ZX', 'ZY', 'ZZ']

        Args:
            weight (bool): optionally sort by weight if ``True`` (Default: ``False``).
            phase (bool): Optionally sort by phase before weight or order
                          (Default: ``False``).

        Returns:
            PauliList: a sorted copy of the original table.
        """
        return self[self.argsort(weight=weight, phase=phase)]

    def unique(self, return_index: bool = False, return_counts: bool = False) -> PauliList:
        """Return unique Paulis from the table.

        **Example**

        .. plot::
           :include-source:
           :nofigs:

            from qiskit.quantum_info.operators import PauliList

            pt = PauliList(['X', 'Y', '-X', 'I', 'I', 'Z', 'X', 'iZ'])
            unique = pt.unique()
            print(unique)

        .. code-block:: text

            ['X', 'Y', '-X', 'I', 'Z', 'iZ']

        Args:
            return_index (bool): If ``True``, also return the indices that
                                 result in the unique array.
                                 (Default: ``False``)
            return_counts (bool): If ``True``, also return the number of times
                                  each unique item appears in the table.

        Returns:
            PauliList: unique
                the table of the unique rows.

            unique_indices: np.ndarray, optional
                The indices of the first occurrences of the unique values in
                the original array. Only provided if ``return_index`` is ``True``.

            unique_counts: np.array, optional
                The number of times each of the unique values comes up in the
                original array. Only provided if ``return_counts`` is ``True``.
        """
        # Check if we need to stack the phase array
        if np.any(self._phase != self._phase[0]):
            # Create a single array of Pauli's and phases for calling np.unique on
            # so that we treat different phased Pauli's as unique
            array = np.hstack([self._z, self._x, self.phase.reshape((self.phase.shape[0], 1))])
        else:
            # All Pauli's have the same phase so we only need to sort the array
            array = np.hstack([self._z, self._x])

        # Get indexes of unique entries
        if return_counts:
            _, index, counts = np.unique(array, return_index=True, return_counts=True, axis=0)
        else:
            _, index = np.unique(array, return_index=True, axis=0)

        # Sort the index so we return unique rows in the original array order
        sort_inds = index.argsort()
        index = index[sort_inds]
        unique = PauliList(BasePauli(self._z[index], self._x[index], self._phase[index]))

        # Concatenate return tuples
        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_counts:
            ret += (counts[sort_inds],)
        if len(ret) == 1:
            return ret[0]
        return ret

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other: PauliList) -> PauliList:
        """Return the tensor product with each Pauli in the list.

        Args:
            other (PauliList): another PauliList.

        Returns:
            PauliList: the list of tensor product Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        return PauliList(super().tensor(other))

    def expand(self, other: PauliList) -> PauliList:
        """Return the expand product of each Pauli in the list.

        Args:
            other (PauliList): another PauliList.

        Returns:
            PauliList: the list of tensor product Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError(
                "Incompatible PauliLists. Other list must "
                "have either 1 or the same number of Paulis."
            )
        return PauliList(super().expand(other))

    def compose(
        self,
        other: PauliList,
        qargs: None | list = None,
        front: bool = False,
        inplace: bool = False,
    ) -> PauliList:
        """Return the composition self∘other for each Pauli in the list.

        Args:
            other (PauliList): another PauliList.
            qargs (None or list): qubits to apply dot product on (Default: ``None``).
            front (bool): If True use `dot` composition method [default: ``False``].
            inplace (bool): If ``True`` update in-place (default: ``False``).

        Returns:
            PauliList: the list of composed Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list, or has the wrong number of qubits
                         for the specified ``qargs``.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, PauliList):
            other = PauliList(other)
        if len(other) not in [1, len(self)]:
            raise QiskitError(
                "Incompatible PauliLists. Other list must "
                "have either 1 or the same number of Paulis."
            )
        return PauliList(super().compose(other, qargs=qargs, front=front, inplace=inplace))

    def dot(self, other: PauliList, qargs: None | list = None, inplace: bool = False) -> PauliList:
        """Return the composition other∘self for each Pauli in the list.

        Args:
            other (PauliList): another PauliList.
            qargs (None or list): qubits to apply dot product on (Default: ``None``).
            inplace (bool): If True update in-place (default: ``False``).

        Returns:
            PauliList: the list of composed Paulis.

        Raises:
            QiskitError: if other cannot be converted to a PauliList, does
                         not have either 1 or the same number of Paulis as
                         the current list, or has the wrong number of qubits
                         for the specified ``qargs``.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def _add(self, other, qargs=None):
        """Append two PauliLists.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (PauliList): another table.
            qargs (None or list): optional subsystems to add on
                                  (Default: ``None``)

        Returns:
            PauliList: the concatenated list ``self`` + ``other``.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, PauliList):
            other = PauliList(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        base_phase = np.hstack((self._phase, other._phase))

        if qargs is None or (sorted(qargs) == qargs and len(qargs) == self.num_qubits):
            base_z = np.vstack([self._z, other._z])
            base_x = np.vstack([self._x, other._x])
        else:
            # Pad other with identity and then add
            padded = BasePauli(
                np.zeros((other.size, self.num_qubits), dtype=bool),
                np.zeros((other.size, self.num_qubits), dtype=bool),
                np.zeros(other.size, dtype=int),
            )
            padded = padded.compose(other, qargs=qargs, inplace=True)
            base_z = np.vstack([self._z, padded._z])
            base_x = np.vstack([self._x, padded._x])

        return PauliList(BasePauli(base_z, base_x, base_phase))

    def _multiply(self, other):
        """Multiply each Pauli in the list by a phase.

        Args:
            other (complex or array): a complex number in [1, -1j, -1, 1j]

        Returns:
            PauliList: the list of Paulis other * self.

        Raises:
            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].
        """
        return PauliList(super()._multiply(other))

    def conjugate(self):
        """Return the complex conjugate of each Pauli in the list with respect to the Z basis."""
        return PauliList(super().conjugate())

    def transpose(self):
        """Return the transpose of each Pauli in the list."""
        return PauliList(super().transpose())

    def adjoint(self):
        """Return the adjoint of each Pauli in the list."""
        return PauliList(super().adjoint())

    def inverse(self):
        """Return the inverse of each Pauli in the list."""
        return PauliList(super().adjoint())

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, other: BasePauli, qargs: list | None = None) -> bool:
        """Return True for each Pauli that commutes with other.

        Args:
            other (PauliList): another PauliList operator.
            qargs (list): qubits to apply dot product on (default: ``None``).

        Returns:
            bool: ``True`` if Paulis commute, ``False`` if they anti-commute.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, BasePauli):
            other = PauliList(other)
        return super().commutes(other, qargs=qargs)

    def anticommutes(self, other: BasePauli, qargs: list | None = None) -> bool:
        """Return ``True`` if the other Pauli anticommutes with this one.

        Args:
            other (PauliList): another PauliList operator.
            qargs (list): qubits to apply dot product on (default: ``None``).

        Returns:
            bool: ``True`` if Paulis anticommute, ``False`` if they commute.
        """
        return np.logical_not(self.commutes(other, qargs=qargs))

    def commutes_with_all(self, other: PauliList) -> np.ndarray:
        """Return indexes of rows that commute ``other``.

        If ``other`` is a multi-row Pauli list the returned vector indexes rows
        of the current PauliList that commute with *all* Paulis in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliList): a single Pauli or multi-row PauliList.

        Returns:
            array: index array of the commuting rows.
        """
        return self._commutes_with_all(other)

    def anticommutes_with_all(self, other: PauliList) -> np.ndarray:
        """Return indexes of rows that anticommute with the other Pauli list.

        If ``other`` is a multi-row Pauli list the returned vector indexes rows
        of the current PauliList that anti-commute with *all* Paulis in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliList): a single Pauli or multi-row PauliList.

        Returns:
            array: index array of the anti-commuting rows.
        """
        return self._commutes_with_all(other, anti=True)

    def _commutes_with_all(self, other, anti=False):
        """Return row indexes that commute with all rows in another PauliList.

        Args:
            other (PauliList): a PauliList.
            anti (bool): if ``True`` return rows that anti-commute, otherwise
                         return rows that commute (Default: ``False``).

        Returns:
            array: index array of commuting or anti-commuting row.
        """
        if not isinstance(other, PauliList):
            other = PauliList(other)
        comms = self.commutes(other[0])
        (inds,) = np.where(comms == int(not anti))
        for pauli in other[1:]:
            comms = self[inds].commutes(pauli)
            (new_inds,) = np.where(comms == int(not anti))
            if new_inds.size == 0:
                # No commuting rows
                return new_inds
            inds = inds[new_inds]
        return inds

    def evolve(
        self,
        other: Pauli | Clifford | QuantumCircuit,
        qargs: list | None = None,
        frame: Literal["h", "s"] = "h",
    ) -> Pauli:
        r"""Performs either Heisenberg (default) or Schrödinger picture
        evolution of the Pauli by a Clifford and returns the evolved Pauli.

        Schrödinger picture evolution can be chosen by passing parameter ``frame='s'``.
        This option yields a faster calculation.

        Heisenberg picture evolves the Pauli as :math:`P^\prime = C^\dagger.P.C`.

        Schrödinger picture evolves the Pauli as :math:`P^\prime = C.P.C^\dagger`.

        Args:
            other (Pauli or Clifford or QuantumCircuit): The Clifford operator to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.
            frame (string): ``'h'`` for Heisenberg (default) or ``'s'`` for Schrödinger framework.

        Returns:
            PauliList: the Pauli :math:`C^\dagger.P.C` (Heisenberg picture)
            or the Pauli :math:`C.P.C^\dagger` (Schrödinger picture).

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        from qiskit.circuit import Instruction

        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, (BasePauli, Instruction, QuantumCircuit, Clifford)):
            # Convert to a PauliList
            other = PauliList(other)

        return PauliList(super().evolve(other, qargs=qargs, frame=frame))

    def to_labels(self, array: bool = False):
        r"""Convert a PauliList to a list Pauli string labels.

        For large PauliLists converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            array (bool): return a Numpy array if ``True``, otherwise
                          return a list (Default: ``False``).

        Returns:
            list or array: The rows of the PauliList in label form.
        """
        if (self.phase == 1).any():
            prefix_len = 2
        elif (self.phase > 0).any():
            prefix_len = 1
        else:
            prefix_len = 0
        str_len = self.num_qubits + prefix_len
        ret = np.zeros(self.size, dtype=f"<U{str_len}")
        iterator = self.label_iter()
        for i in range(self.size):
            ret[i] = next(iterator)
        if array:
            return ret
        return ret.tolist()

    def to_matrix(self, sparse: bool = False, array: bool = False) -> list:
        r"""Convert to a list or array of Pauli matrices.

        For large PauliLists converting using the ``array=True``
        kwarg will be more efficient since it allocates memory a full
        rank-3 Numpy array of matrices in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            sparse (bool): if ``True`` return sparse CSR matrices, otherwise
                           return dense Numpy arrays (Default: ``False``).
            array (bool): return as rank-3 numpy array if ``True``, otherwise
                          return a list of Numpy arrays (Default: ``False``).

        Returns:
            list: A list of dense Pauli matrices if ``array=False`` and ``sparse=False``.
            list: A list of sparse Pauli matrices if ``array=False`` and ``sparse=True``.
            array: A dense rank-3 array of Pauli matrices if ``array=True``.
        """
        if not array:
            # We return a list of Numpy array matrices
            return list(self.matrix_iter(sparse=sparse))
        # For efficiency we also allow returning a single rank-3
        # array where first index is the Pauli row, and second two
        # indices are the matrix indices
        dim = 2**self.num_qubits
        ret = np.zeros((self.size, dim, dim), dtype=complex)
        iterator = self.matrix_iter(sparse=sparse)
        for i in range(self.size):
            ret[i] = next(iterator)
        return ret

    # ---------------------------------------------------------------------
    # Custom Iterators
    # ---------------------------------------------------------------------

    def label_iter(self):
        """Return a label representation iterator.

        This is a lazy iterator that converts each row into the string
        label only as it is used. To convert the entire table to labels use
        the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the PauliList.
        """

        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""

            def __repr__(self):
                return f"<PauliList_label_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                return self.obj._to_label(self.obj._z[key], self.obj._x[key], self.obj._phase[key])

        return LabelIterator(self)

    def matrix_iter(self, sparse: bool = False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each row into the Pauli matrix
        representation only as it is used. To convert the entire table to
        matrices use the :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if ``True``,
                           otherwise return Numpy array matrices
                           (Default: ``False``)

        Returns:
            MatrixIterator: matrix iterator object for the PauliList.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return f"<PauliList_matrix_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                return self.obj._to_matrix(
                    self.obj._z[key], self.obj._x[key], self.obj._phase[key], sparse=sparse
                )

        return MatrixIterator(self)

    # ---------------------------------------------------------------------
    # Class methods
    # ---------------------------------------------------------------------

    @classmethod
    def from_symplectic(
        cls, z: np.ndarray, x: np.ndarray, phase: np.ndarray | None = 0
    ) -> PauliList:
        """Construct a PauliList from a symplectic data.

        Args:
            z (np.ndarray): 2D boolean Numpy array.
            x (np.ndarray): 2D boolean Numpy array.
            phase (np.ndarray or None):  1D integer array from Z_4.

        Returns:
            PauliList: the constructed PauliList.
        """
        if isinstance(phase, np.ndarray) and np.ndim(phase) > 1:
            raise ValueError(f"phase should be at most 1D but has {np.ndim(phase)} dimensions.")
        base_z, base_x, base_phase = cls._from_array(z, x, phase)
        return cls(BasePauli(base_z, base_x, base_phase))

    def _noncommutation_graph(self, qubit_wise):
        """Create an edge list representing the non-commutation graph (Pauli Graph).

        An edge (i, j) is present if i and j are not commutable.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.

        Returns:
            list[tuple[int,int]]: A list of pairs of indices of the PauliList that are not commutable.
        """
        # convert a Pauli operator into int vector where {I: 0, X: 2, Y: 3, Z: 1}
        mat1 = np.array(
            [op.z + 2 * op.x for op in self],
            dtype=np.int8,
        )
        mat2 = mat1[:, None]
        # This is 0 (false-y) iff one of the operators is the identity and/or both operators are the
        # same.  In other cases, it is non-zero (truth-y).
        qubit_anticommutation_mat = (mat1 * mat2) * (mat1 - mat2)
        # 'adjacency_mat[i, j]' is True iff Paulis 'i' and 'j' do not commute in the given strategy.
        if qubit_wise:
            adjacency_mat = np.logical_or.reduce(qubit_anticommutation_mat, axis=2)
        else:
            # Don't commute if there's an odd number of element-wise anti-commutations.
            adjacency_mat = np.logical_xor.reduce(qubit_anticommutation_mat, axis=2)
        # Convert into list where tuple elements are non-commuting operators.  We only want to
        # results from one triangle to avoid symmetric duplications.
        return list(zip(*np.where(np.triu(adjacency_mat, k=1))))

    def noncommutation_graph(self, qubit_wise: bool) -> rx.PyGraph:
        """Create the non-commutation graph of this PauliList.

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
        edges = self._noncommutation_graph(qubit_wise)
        graph = rx.PyGraph()
        graph.add_nodes_from(range(self.size))
        graph.add_edges_from_no_data(edges)
        return graph

    def _commuting_groups(self, qubit_wise: bool) -> dict[int, list[int]]:
        """Partition a PauliList into sets of commuting Pauli strings.

        This is the internal logic of the public ``PauliList.group_commuting`` method which returns
        a mapping of colors to Pauli indices. The same logic is re-used by
        ``SparsePauliOp.group_commuting``.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.

        Returns:
            dict[int, list[int]]: Dictionary of color indices mapping to a list of Pauli indices.
        """
        graph = self.noncommutation_graph(qubit_wise)
        # Keys in coloring_dict are nodes, values are colors
        coloring_dict = rx.graph_greedy_color(graph)
        groups = defaultdict(list)
        for idx, color in coloring_dict.items():
            groups[color].append(idx)
        return groups

    def group_qubit_wise_commuting(self) -> list[PauliList]:
        """Partition a PauliList into sets of mutually qubit-wise commuting Pauli strings.

        Returns:
            list[PauliList]: List of PauliLists where each PauliList contains commutable Pauli operators.
        """
        return self.group_commuting(qubit_wise=True)

    def group_commuting(self, qubit_wise: bool = False) -> list[PauliList]:
        """Partition a PauliList into sets of commuting Pauli strings.

        Args:
            qubit_wise (bool): whether the commutation rule is applied to the whole operator,
                or on a per-qubit basis.  For example:

                .. plot::
                   :include-source:
                   :nofigs:

                    >>> from qiskit.quantum_info import PauliList
                    >>> op = PauliList(["XX", "YY", "IZ", "ZZ"])
                    >>> op.group_commuting()
                    [PauliList(['XX', 'YY']), PauliList(['IZ', 'ZZ'])]
                    >>> op.group_commuting(qubit_wise=True)
                    [PauliList(['XX']), PauliList(['YY']), PauliList(['IZ', 'ZZ'])]

        Returns:
            list[PauliList]: List of PauliLists where each PauliList contains commuting Pauli operators.
        """
        groups = self._commuting_groups(qubit_wise)
        return [self[group] for group in groups.values()]
