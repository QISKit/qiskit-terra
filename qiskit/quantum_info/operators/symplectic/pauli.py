# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
N-qubit Pauli Operator Class
"""

from __future__ import annotations

import re
from typing import Literal, TYPE_CHECKING

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli, _count_y

if TYPE_CHECKING:
    from qiskit.quantum_info.operators.symplectic.clifford import Clifford
    from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
    from qiskit.transpiler.layout import TranspileLayout


class Pauli(BasePauli):
    r"""N-qubit Pauli operator.

    This class represents an operator :math:`P` from the full :math:`n`-qubit
    *Pauli* group

    .. math::

        P = (-i)^{q} P_{n-1} \otimes ... \otimes P_{0}

    where :math:`q\in \mathbb{Z}_4` and :math:`P_i \in \{I, X, Y, Z\}`
    are single-qubit Pauli matrices:

    .. math::

        I = \begin{pmatrix} 1 & 0  \\ 0 & 1  \end{pmatrix},
        X = \begin{pmatrix} 0 & 1  \\ 1 & 0  \end{pmatrix},
        Y = \begin{pmatrix} 0 & -i \\ i & 0  \end{pmatrix},
        Z = \begin{pmatrix} 1 & 0  \\ 0 & -1 \end{pmatrix}.

    **Initialization**

    A Pauli object can be initialized in several ways:

        ``Pauli(obj)``
            where ``obj`` is a Pauli string, ``Pauli`` or
            :class:`~qiskit.quantum_info.ScalarOp` operator, or a Pauli
            gate or :class:`~qiskit.QuantumCircuit` containing only
            Pauli gates.

        ``Pauli((z, x, phase))``
            where ``z`` and ``x`` are boolean ``numpy.ndarrays`` and ``phase`` is
            an integer in ``[0, 1, 2, 3]``.

        ``Pauli((z, x))``
            equivalent to ``Pauli((z, x, 0))`` with trivial phase.

    **String representation**

    An :math:`n`-qubit Pauli may be represented by a string consisting of
    :math:`n` characters from ``['I', 'X', 'Y', 'Z']``, and optionally phase
    coefficient in ``['', '-i', '-', 'i']``. For example: ``'XYZ'`` or
    ``'-iZIZ'``.

    In the string representation qubit-0 corresponds to the right-most
    Pauli character, and qubit-:math:`(n-1)` to the left-most Pauli
    character. For example ``'XYZ'`` represents
    :math:`X\otimes Y \otimes Z` with ``'Z'`` on qubit-0,
    ``'Y'`` on qubit-1, and ``'X'`` on qubit-2.

    The string representation can be converted to a ``Pauli`` using the
    class initialization (``Pauli('-iXYZ')``). A ``Pauli`` object can be
    converted back to the string representation using the
    :meth:`to_label` method or ``str(pauli)``.

    .. note::

        Using ``str`` to convert a ``Pauli`` to a string will truncate the
        returned string for large numbers of qubits while :meth:`to_label`
        will return the full string with no truncation. The default
        truncation length is 50 characters. The default value can be
        changed by setting the class ``__truncate__`` attribute to an integer
        value. If set to ``0`` no truncation will be performed.

    **Array Representation**

    The internal data structure of an :math:`n`-qubit Pauli is two
    length-:math:`n` boolean vectors :math:`z \in \mathbb{Z}_2^N`,
    :math:`x \in \mathbb{Z}_2^N`, and an integer :math:`q \in \mathbb{Z}_4`
    defining the Pauli operator

    .. math::

        P = (-i)^{q + z\cdot x} Z^z \cdot X^x.

    The :math:`k`-th qubit corresponds to the :math:`k`-th entry in the
    :math:`z` and :math:`x` arrays

    .. math::

        \begin{aligned}
        P &= P_{n-1} \otimes ... \otimes P_{0} \\
        P_k &= (-i)^{z[k] * x[k]} Z^{z[k]}\cdot X^{x[k]}
        \end{aligned}

    where ``z[k] = P.z[k]``, ``x[k] = P.x[k]`` respectively.

    The :math:`z` and :math:`x` arrays can be accessed and updated using
    the :attr:`z` and :attr:`x` properties respectively. The phase integer
    :math:`q` can be accessed and updated using the :attr:`phase` property.

    **Matrix Operator Representation**

    Pauli's can be converted to :math:`(2^n, 2^n)`
    :class:`~qiskit.quantum_info.Operator` using the :meth:`to_operator` method,
    or to a dense or sparse complex matrix using the :meth:`to_matrix` method.

    **Data Access**

    The individual qubit Paulis can be accessed and updated using the ``[]``
    operator which accepts integer, lists, or slices for selecting subsets
    of Paulis. Note that selecting subsets of Pauli's will discard the
    phase of the current Pauli.

    For example

    .. plot::
       :include-source:
       :nofigs:

        from qiskit.quantum_info import Pauli

        P = Pauli('-iXYZ')

        print('P[0] =', repr(P[0]))
        print('P[1] =', repr(P[1]))
        print('P[2] =', repr(P[2]))
        print('P[:] =', repr(P[:]))
        print('P[::-1] =', repr(P[::-1]))
    """

    # Set the max Pauli string size before truncation
    __truncate__ = 50

    _VALID_LABEL_PATTERN = re.compile(r"(?P<coeff>[+-]?1?[ij]?)(?P<pauli>[IXYZ]*)")
    _CANONICAL_PHASE_LABEL = {"": 0, "-i": 1, "-": 2, "i": 3}

    def __init__(self, data: str | tuple | Pauli | ScalarOp | QuantumCircuit | None = None):
        r"""Initialize the Pauli.

        When using the symplectic array input data both z and x arguments must
        be provided, however the first (z) argument can be used alone for string
        label, Pauli operator, or :class:`.ScalarOp` input data.

        Args:
            data (str or tuple or Pauli or ScalarOp): input data for Pauli. If input is
                a tuple it must be of the form ``(z, x)`` or ``(z, x, phase)`` where
                ``z`` and ``x`` are boolean Numpy arrays, and phase is an integer from
                :math:`\mathbb{Z}_4`.
                If input is a string, it must be a concatenation of a phase and a Pauli string
                (e.g. ``'XYZ', '-iZIZ'``) where a phase string is a combination of at most three
                characters from ``['+', '-', '']``, ``['1', '']``, and ``['i', 'j', '']`` in this order,
                e.g. ``''``, ``'-1j'`` while a Pauli string is 1 or more
                characters of ``'I'``, ``'X'``, ``'Y'``, or ``'Z'``,
                e.g. ``'Z'``, ``'XIYY'``.

        Raises:
            QiskitError: if input array is invalid shape.
        """
        if isinstance(data, BasePauli):
            base_z, base_x, base_phase = data._z, data._x, data._phase
        elif isinstance(data, tuple):
            if len(data) not in [2, 3]:
                raise QiskitError(
                    "Invalid input tuple for Pauli, input tuple must be `(z, x, phase)` or `(z, x)`"
                )
            base_z, base_x, base_phase = self._from_array(*data)
        elif isinstance(data, str):
            base_z, base_x, base_phase = self._from_label(data)
        elif isinstance(data, ScalarOp):
            base_z, base_x, base_phase = self._from_scalar_op(data)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            base_z, base_x, base_phase = self._from_circuit(data)
        else:
            raise QiskitError("Invalid input data for Pauli.")

        # Initialize BasePauli
        if base_z.shape[0] != 1:
            raise QiskitError("Input is not a single Pauli")
        super().__init__(base_z, base_x, base_phase)

    @property
    def name(self):
        """Unique string identifier for operation type."""
        return "pauli"

    @property
    def num_clbits(self):
        """Number of classical bits."""
        return 0

    def __repr__(self):
        """Display representation."""
        return f"Pauli('{self.__str__()}')"

    def __str__(self):
        """Print representation."""
        if self.__truncate__ and self.num_qubits > self.__truncate__:
            front = self[-self.__truncate__ :].to_label()
            return front + "..."
        return self.to_label()

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        arr = self.to_matrix()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    @classmethod
    def set_truncation(cls, val: int):
        """Set the max number of Pauli characters to display before truncation.

        Args:
            val (int): the number of characters.

        .. note::

            Truncation will be disabled if the truncation value is set to 0.
        """
        cls.__truncate__ = int(val)

    def __eq__(self, other):
        """Test if two Paulis are equal."""
        if not isinstance(other, BasePauli):
            return False
        return self._eq(other)

    def equiv(self, other: Pauli) -> bool:
        """Return True if Pauli's are equivalent up to group phase.

        Args:
            other (Pauli): an operator object.

        Returns:
            bool: True if the Pauli's are equivalent up to group phase.
        """
        if not isinstance(other, Pauli):
            try:
                other = Pauli(other)
            except QiskitError:
                return False
        return np.all(self._z == other._z) and np.all(self._x == other._x)

    @property
    def settings(self) -> dict:
        """Return settings."""
        return {"data": self.to_label()}

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------
    @property
    def phase(self):
        """Return the group phase exponent for the Pauli."""
        # Convert internal ZX-phase convention of BasePauli to group phase
        return np.mod(self._phase - self._count_y(dtype=self._phase.dtype), 4)[0]

    @phase.setter
    def phase(self, value):
        # Convert group phase convention to internal ZX-phase convention
        self._phase[:] = np.mod(value + self._count_y(dtype=self._phase.dtype), 4)

    @property
    def x(self):
        """The x vector for the Pauli."""
        return self._x[0]

    @x.setter
    def x(self, val):
        self._x[0, :] = val

    @property
    def z(self):
        """The z vector for the Pauli."""
        return self._z[0]

    @z.setter
    def z(self, val):
        self._z[0, :] = val

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __len__(self):
        """Return the number of qubits in the Pauli."""
        return self.num_qubits

    def __getitem__(self, qubits):
        """Return the unsigned Pauli group Pauli for subset of qubits."""
        # Set group phase to 0 so returned Pauli is always +1 coeff
        if isinstance(qubits, (int, np.integer)):
            qubits = [qubits]
        return Pauli((self.z[qubits], self.x[qubits]))

    def __setitem__(self, qubits, value):
        """Update the Pauli for a subset of qubits."""
        if not isinstance(value, Pauli):
            value = Pauli(value)
        self._z[0, qubits] = value.z
        self._x[0, qubits] = value.x
        # Add extra phase from new Pauli to current
        self._phase = self._phase + value._phase

    def delete(self, qubits: int | list) -> Pauli:
        """Return a Pauli with qubits deleted.

        Args:
            qubits (int or list): qubits to delete from Pauli.

        Returns:
            Pauli: the resulting Pauli with the specified qubits removed.

        Raises:
            QiskitError: if ind is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(qubits, (int, np.integer)):
            qubits = [qubits]
        if len(qubits) == 0:
            return Pauli((self._z, self._x, self.phase))
        if max(qubits) > self.num_qubits - 1:
            raise QiskitError(
                "Qubit index is larger than the number of qubits "
                f"({max(qubits)}>{self.num_qubits - 1})."
            )
        if len(qubits) == self.num_qubits:
            raise QiskitError("Cannot delete all qubits of Pauli")
        z = np.delete(self._z, qubits, axis=1)
        x = np.delete(self._x, qubits, axis=1)
        return Pauli((z, x, self.phase))

    def insert(self, qubits: int | list, value: Pauli) -> Pauli:
        """Insert a Pauli at specific qubit value.

        Args:
            qubits (int or list): qubits index to insert at.
            value (Pauli): value to insert.

        Returns:
            Pauli: the resulting Pauli with the entries inserted.

        Raises:
            QiskitError: if the insertion qubits are invalid.
        """
        if not isinstance(value, Pauli):
            value = Pauli(value)

        # Initialize empty operator
        ret_qubits = self.num_qubits + value.num_qubits
        ret = Pauli((np.zeros(ret_qubits, dtype=bool), np.zeros(ret_qubits, dtype=bool)))
        if isinstance(qubits, (int, np.integer)):
            if value.num_qubits == 1:
                qubits = [qubits]
            else:
                qubits = list(range(qubits, qubits + value.num_qubits))
        if len(qubits) != value.num_qubits:
            raise QiskitError(
                "Number of indices does not match number of qubits for "
                f"the inserted Pauli ({len(qubits)}!={value.num_qubits})"
            )
        if max(qubits) > ret.num_qubits - 1:
            raise QiskitError(
                "Index is too larger for combined Pauli number of qubits "
                f"({max(qubits)}>{ret.num_qubits - 1})."
            )
        # Qubit positions for original op
        self_qubits = [i for i in range(ret.num_qubits) if i not in qubits]
        ret[self_qubits] = self
        ret[qubits] = value
        return ret

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def __hash__(self):
        """Make hashable based on string representation."""
        return hash(self.to_label())

    def to_label(self) -> str:
        """Convert a Pauli to a string label.

        .. note::

            The difference between `to_label` and :meth:`__str__` is that
            the latter will truncate the output for large numbers of qubits.

        Returns:
            str: the Pauli string label.
        """
        return self._to_label(self.z, self.x, self._phase[0])

    def to_matrix(self, sparse: bool = False) -> np.ndarray:
        r"""Convert to a Numpy array or sparse CSR matrix.

        Args:
            sparse (bool): if True return sparse CSR matrices, otherwise
                           return dense Numpy arrays (default: False).

        Returns:
            array: The Pauli matrix.
        """
        return self._to_matrix(self.z, self.x, self._phase[0], sparse=sparse)

    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from math import pi

        pauli, phase = self._to_label(
            self.z, self.x, self._phase[0], full_group=False, return_phase=True
        )
        if len(pauli) == 1:
            gate = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}[pauli]
        else:
            gate = PauliGate(pauli)
        if not phase:
            return gate
        # Add global phase
        circuit = QuantumCircuit(self.num_qubits, name=str(self))
        circuit.global_phase = -phase * pi / 2
        circuit.append(gate, range(self.num_qubits))
        return circuit.to_instruction()

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def compose(
        self, other: Pauli, qargs: list | None = None, front: bool = False, inplace: bool = False
    ) -> Pauli:
        """Return the operator composition with another Pauli.

        Args:
            other (Pauli): a Pauli object.
            qargs (list or None):  qubits to apply dot product
                                  on (default: None).
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: The composed Pauli.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
                         incompatible dimensions for specified subsystems.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for
            matrix operators, while :meth:`dot` is defined as `right` matrix
            multiplication. That is that ``A & B == A.compose(B)`` is equivalent to
            ``B.dot(A)`` when ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` kwarg changes this to `right` matrix
            multiplication and is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().compose(other, qargs=qargs, front=front, inplace=inplace))

    def dot(self, other: Pauli, qargs: list | None = None, inplace: bool = False) -> Pauli:
        """Return the right multiplied operator self * other.

        Args:
            other (Pauli): an operator object.
            qargs (list or None):  qubits to apply dot product
                                  on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: The operator self * other.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def tensor(self, other: Pauli) -> Pauli:
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().tensor(other))

    def expand(self, other: Pauli) -> Pauli:
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().expand(other))

    def _multiply(self, other):
        return Pauli(super()._multiply(other))

    def conjugate(self):
        return Pauli(super().conjugate())

    def transpose(self):
        return Pauli(super().transpose())

    def adjoint(self):
        return Pauli(super().adjoint())

    def inverse(self):
        """Return the inverse of the Pauli."""
        return Pauli(super().adjoint())

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, other: Pauli | PauliList, qargs: list | None = None) -> bool:
        """Return True if the Pauli commutes with other.

        Args:
            other (Pauli or PauliList): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's commute, False if they anti-commute.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, BasePauli):
            other = Pauli(other)
        ret = super().commutes(other, qargs=qargs)
        if len(ret) == 1:
            return ret[0]
        return ret

    def anticommutes(self, other: Pauli, qargs: list | None = None) -> bool:
        """Return True if other Pauli anticommutes with self.

        Args:
            other (Pauli): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's anticommute, False if they commute.
        """
        return np.logical_not(self.commutes(other, qargs=qargs))

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
            frame (string): ``'h'`` for Heisenberg (default) or ``'s'`` for
                Schrödinger framework.

        Returns:
            Pauli: the Pauli :math:`C^\dagger.P.C` (Heisenberg picture)
            or the Pauli :math:`C.P.C^\dagger` (Schrödinger picture).

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        if not isinstance(other, (Pauli, Instruction, QuantumCircuit, Clifford)):
            # Convert to a Pauli
            other = Pauli(other)

        return Pauli(super().evolve(other, qargs=qargs, frame=frame))

    # ---------------------------------------------------------------------
    # Initialization helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _from_label(label):
        """Return the symplectic representation of Pauli string.

        Args:
            label (str): the Pauli string label.

        Returns:
            BasePauli: the BasePauli corresponding to the label.

        Raises:
            QiskitError: if Pauli string is not valid.
        """
        match_ = Pauli._VALID_LABEL_PATTERN.fullmatch(label)
        if match_ is None:
            raise QiskitError(f'Pauli string label "{label}" is not valid.')
        phase = Pauli._CANONICAL_PHASE_LABEL[
            (match_["coeff"] or "").replace("1", "").replace("+", "").replace("j", "i")
        ]

        # Convert to Symplectic representation
        pauli_bytes = np.frombuffer(match_["pauli"].encode("ascii"), dtype=np.uint8)[::-1]
        ys = pauli_bytes == ord("Y")
        base_x = np.logical_or(pauli_bytes == ord("X"), ys).reshape(1, -1)
        base_z = np.logical_or(pauli_bytes == ord("Z"), ys).reshape(1, -1)
        base_phase = np.array([(phase + np.count_nonzero(ys)) % 4], dtype=int)
        return base_z, base_x, base_phase

    @classmethod
    def _from_scalar_op(cls, op):
        """Convert a ScalarOp to BasePauli data."""
        if op.num_qubits is None:
            raise QiskitError(f"{op} is not an N-qubit identity")
        base_z = np.zeros((1, op.num_qubits), dtype=bool)
        base_x = np.zeros((1, op.num_qubits), dtype=bool)
        base_phase = np.mod(
            cls._phase_from_complex(op.coeff) + _count_y(base_x, base_z), 4, dtype=int
        )
        return base_z, base_x, base_phase

    @classmethod
    def _from_pauli_instruction(cls, instr):
        """Convert a Pauli instruction to BasePauli data."""
        if isinstance(instr, PauliGate):
            return cls._from_label(instr.params[0])
        if isinstance(instr, IGate):
            return np.array([[False]]), np.array([[False]]), np.array([0])
        if isinstance(instr, XGate):
            return np.array([[False]]), np.array([[True]]), np.array([0])
        if isinstance(instr, YGate):
            return np.array([[True]]), np.array([[True]]), np.array([1])
        if isinstance(instr, ZGate):
            return np.array([[True]]), np.array([[False]]), np.array([0])
        raise QiskitError("Invalid Pauli instruction.")

    @classmethod
    def _from_circuit(cls, instr):
        """Convert a Pauli circuit to BasePauli data."""
        # Try and convert single instruction
        if isinstance(instr, (PauliGate, IGate, XGate, YGate, ZGate)):
            return cls._from_pauli_instruction(instr)

        if isinstance(instr, Instruction):
            # Convert other instructions to circuit definition
            if instr.definition is None:
                raise QiskitError(f"Cannot apply Instruction: {instr.name}")
            # Convert to circuit
            instr = instr.definition

        # Initialize identity Pauli
        ret = Pauli(
            BasePauli(
                np.zeros((1, instr.num_qubits), dtype=bool),
                np.zeros((1, instr.num_qubits), dtype=bool),
                np.zeros(1, dtype=int),
            )
        )

        # Add circuit global phase if specified
        if instr.global_phase:
            ret.phase = cls._phase_from_complex(np.exp(1j * float(instr.global_phase)))

        # Recursively apply instructions
        for inner in instr.data:
            if inner.clbits:
                raise QiskitError(
                    f"Cannot apply instruction with classical bits: {inner.operation.name}"
                )
            if not isinstance(inner.operation, (Barrier, Delay)):
                next_instr = BasePauli(*cls._from_circuit(inner.operation))
                if next_instr is not None:
                    qargs = [instr.find_bit(tup).index for tup in inner.qubits]
                    ret = ret.compose(next_instr, qargs=qargs)
        return ret._z, ret._x, ret._phase

    def apply_layout(
        self, layout: TranspileLayout | list[int] | None, num_qubits: int | None = None
    ) -> Pauli:
        """Apply a transpiler layout to this :class:`~.quantum_info.Pauli`

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
            A new :class:`~.quantum_info.Pauli` with the provided layout applied
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
        new_op = type(self)("I" * n_qubits)
        return new_op.compose(self, qargs=layout)


# Update docstrings for API docs
generate_apidocs(Pauli)
