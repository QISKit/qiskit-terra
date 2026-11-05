# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023
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
import copy
from typing import Literal, TYPE_CHECKING

import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, MultiplyMixin
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _n_half_pis

if TYPE_CHECKING:
    from qiskit.quantum_info.operators.symplectic.clifford import Clifford


# utility for _to_matrix
_PARITY = np.array([-1 if bin(i).count("1") % 2 else 1 for i in range(256)], dtype=complex)
# Utility for `_to_label`
_TO_LABEL_CHARS = np.array([ord("I"), ord("X"), ord("Z"), ord("Y")], dtype=np.uint8)


class BasePauli(BaseOperator, AdjointMixin, MultiplyMixin):
    r"""Symplectic representation of a list of N-qubit Paulis.

    Base class for Pauli and PauliList.
    """

    def __init__(self, z: np.ndarray, x: np.ndarray, phase: np.ndarray):
        """Initialize the BasePauli.

        This is an array of M N-qubit Paulis defined as
        P = (-i)^phase Z^z X^x.

        Args:
            z (np.ndarray): input z matrix.
            x (np.ndarray): input x matrix.
            phase (np.ndarray): input phase vector.
        """
        self._z = z
        self._x = x
        self._phase = phase
        self._num_paulis, num_qubits = self._z.shape
        super().__init__(num_qubits=num_qubits)

    def copy(self):
        """Make a deep copy of current operator."""
        # Deepcopy has terrible performance on objects with Numpy arrays
        # attributes so we make a shallow copy and then manually copy the
        # Numpy arrays to efficiently mimic a deepcopy
        ret = copy.copy(self)
        ret._z = self._z.copy()
        ret._x = self._x.copy()
        ret._phase = self._phase.copy()
        return ret

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        return self._tensor(self, other)

    def expand(self, other):
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        x1 = cls._stack(a._x, b._num_paulis, False)
        x2 = cls._stack(b._x, a._num_paulis)
        z1 = cls._stack(a._z, b._num_paulis, False)
        z2 = cls._stack(b._z, a._num_paulis)
        phase1 = (
            np.vstack(b._num_paulis * [a._phase])
            .transpose(1, 0)
            .reshape(a._num_paulis * b._num_paulis)
        )
        phase2 = cls._stack(b._phase, a._num_paulis)
        z = np.hstack([z2, z1])
        x = np.hstack([x2, x1])
        phase = np.mod(phase1 + phase2, 4)
        return BasePauli(z, x, phase)

    def compose(self, other, qargs: list | None = None, front: bool = False, inplace=False):
        """Return the composition of Paulis.

        Args:
            a ({cls}): an operator object.
            b ({cls}): an operator object.
            qargs (list or None):  qubits to apply dot product
                                  on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            {cls}: The operator a.compose(b)

        Raises:
            QiskitError: if number of qubits of other does not match qargs.
        """.format(
            cls=type(self).__name__
        )
        # Validation
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError(f"other {type(self).__name__} must be on the same number of qubits.")

        if qargs and other.num_qubits != len(qargs):
            raise QiskitError(
                f"Number of qubits of the other {type(self).__name__} does not match qargs."
            )

        if other._num_paulis not in [1, self._num_paulis]:
            raise QiskitError(
                "Incompatible BasePaulis. Second list must "
                "either have 1 or the same number of Paulis."
            )

        # Compute phase shift
        if qargs is not None:
            x1, z1 = self._x[:, qargs], self._z[:, qargs]
        else:
            x1, z1 = self._x, self._z
        x2, z2 = other._x, other._z

        # Get phase shift
        phase = self._phase + other._phase
        if front:
            phase += 2 * _count_y(x1, z2, dtype=phase.dtype)
        else:
            phase += 2 * _count_y(x2, z1, dtype=phase.dtype)

        # Update Pauli
        x = np.logical_xor(x1, x2)
        z = np.logical_xor(z1, z2)

        if qargs is None:
            if not inplace:
                return BasePauli(z, x, phase)
            # Inplace update
            self._x = x
            self._z = z
            self._phase = phase
            return self

        # Qargs update
        ret = self if inplace else self.copy()
        ret._x[:, qargs] = x
        ret._z[:, qargs] = z
        ret._phase = np.mod(phase, 4)
        return ret

    def _multiply(self, other):
        """Return the {cls} other * self.

        Args:
            other (complex): a complex number in ``[1, -1j, -1, 1j]``.

        Returns:
            {cls}: the {cls} other * self.

        Raises:
            QiskitError: if the phase is not in the set ``[1, -1j, -1, 1j]``.
        """.format(
            cls=type(self).__name__
        )
        if isinstance(other, (np.ndarray, list, tuple)):
            phase = np.array([self._phase_from_complex(phase) for phase in other])
        else:
            phase = self._phase_from_complex(other)
        return BasePauli(self._z, self._x, np.mod(self._phase + phase, 4))

    def conjugate(self):
        """Return the complex conjugate of the Pauli with respect to the Z basis."""
        complex_phase = np.mod(self._phase, 2)
        if np.all(complex_phase == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * complex_phase, 4))

    def transpose(self):
        """Return the transpose of each Pauli in the list."""
        # Transpose sets Y -> -Y. This has effect on changing the phase
        parity_y = self._count_y(dtype=self._phase.dtype) % 2
        if np.all(parity_y == 0):
            return self
        return BasePauli(self._z, self._x, np.mod(self._phase + 2 * parity_y, 4))

    def commutes(self, other: BasePauli, qargs: list | None = None) -> np.ndarray:
        """Return ``True`` if Pauli commutes with ``other``.

        Args:
            other (BasePauli): another BasePauli operator.
            qargs (list): qubits to apply dot product on (default: ``None``).

        Returns:
            np.array: Boolean array of ``True`` if Paulis commute, ``False`` if
                      they anti-commute.

        Raises:
            QiskitError: if number of qubits of ``other`` does not match ``qargs``.
        """
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError(
                "Number of qubits of other Pauli does not match number of "
                f"qargs ({other.num_qubits} != {len(qargs)})."
            )
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                "Number of qubits of other Pauli does not match the current "
                f"Pauli ({other.num_qubits} != {self.num_qubits})."
            )
        if qargs is not None:
            inds = list(qargs)
            x1, z1 = self._x[:, inds], self._z[:, inds]
        else:
            x1, z1 = self._x, self._z
        a_dot_b = np.mod(_count_y(x1, other._z), 2)
        b_dot_a = np.mod(_count_y(other._x, z1), 2)
        return a_dot_b == b_dot_a

    def evolve(
        self,
        other: BasePauli | QuantumCircuit | Clifford,
        qargs: list | None = None,
        frame: Literal["h", "s"] = "h",
    ) -> BasePauli:
        r"""Performs either Heisenberg (default) or Schrödinger picture
        evolution of the Pauli by a Clifford and returns the evolved Pauli.

        Schrödinger picture evolution can be chosen by passing parameter ``frame='s'``.
        This option yields a faster calculation.

        Heisenberg picture evolves the Pauli as :math:`P^\prime = C^\dagger.P.C`.

        Schrödinger picture evolves the Pauli as :math:`P^\prime = C.P.C^\dagger`.

        Args:
            other (BasePauli or QuantumCircuit): The Clifford circuit to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.
            frame (string): ``'h'`` for Heisenberg or ``'s'`` for Schrödinger framework.

        Returns:
            BasePauli: the Pauli :math:`C^\dagger.P.C` (Heisenberg picture)
            or the Pauli :math:`C.P.C^\dagger` (Schrödinger picture).

        Raises:
            QiskitError: if the Clifford number of qubits and ``qargs`` don't match.
        """
        # Check dimension
        if qargs is not None and len(qargs) != other.num_qubits:
            raise QiskitError(
                f"Incorrect number of qubits for Clifford circuit ({other.num_qubits} != {len(qargs)})."
            )
        if qargs is None and self.num_qubits != other.num_qubits:
            raise QiskitError(
                f"Incorrect number of qubits for Clifford circuit "
                f"({other.num_qubits} != {self.num_qubits})."
            )

        # Evolve via Pauli
        if isinstance(other, BasePauli):
            if frame == "s":
                ret = self.compose(other, qargs=qargs)
                ret = ret.compose(other.adjoint(), front=True, qargs=qargs)
            else:
                ret = self.compose(other.adjoint(), qargs=qargs)
                ret = ret.compose(other, front=True, qargs=qargs)
            return ret

        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        # Convert Clifford to quantum circuits
        if isinstance(other, Clifford):
            return self._evolve_clifford(other, qargs=qargs, frame=frame)

        # Otherwise evolve by the inverse circuit to compute C^dg.P.C
        if frame == "s":
            return self.copy()._append_circuit(other, qargs=qargs)
        return self.copy()._append_circuit(other.inverse(), qargs=qargs)

    def _evolve_clifford(self, other, qargs=None, frame="h"):
        """Evolve a Pauli by a Clifford (default is Heisenberg frame)."""

        if frame == "h":
            # Heisenberg evolution C^dg.P.C.

            # Pinning signs (phases) is expensive. We can pin the 2N signs of `C^dg`'s rows, or
            # just the L signs of the result `(C^dg.P.C)`; pin whichever set is smaller. This
            # simple threshold (2N vs L) was near-optimal for benchmarks up to 250 qubits.
            if 2 * other.num_qubits <= self._x.shape[0]:
                # Few qubits, many Paulis: Compute C^dg (pins signs) and evolve by it
                other = other.adjoint()  # O(N^3)
                return self._evolve_clifford(other, qargs=qargs, frame="s")

            # Many qubits, few Paulis: Get result's signs by enforcing round-trip signs are +1,
            # never paying for C^dg's signs. Build `inv` with ZX tableau of C^dg but wrong signs
            # via the cheap part of `Clifford._conjugate_transpose` (at time of writing):
            inv = other.copy()
            tmp = inv.destab_x.copy()
            inv.destab_x = inv.stab_z.T
            inv.destab_z = inv.destab_z.T
            inv.stab_x = inv.stab_x.T
            inv.stab_z = tmp.T
            # Evolving by `inv` gives the correct ZX content of C^dg.P.C but wrong signs:
            ret = self._evolve_clifford(inv, qargs=qargs, frame="s")
            # Recover the signs by evolving back: C.(C^dg.P.C).C^dg = P. Since `ret`
            # already has the right ZX content, evolving it forward reproduces P's ZX content
            # exactly; only the signs can differ.
            fwd = ret._evolve_clifford(other, qargs=qargs, frame="s")
            # Evolution merely adds to the phase (is linear), so `ret`'s phase error passes
            # unchanged into `fwd`; read it off as `fwd.phase - self.phase` and subtract it:
            ret.phase -= fwd.phase - self.phase
            return ret

        # Schrodinger evolution C.P.C^dg.
        if qargs is None:
            qargs_ = slice(None)
        else:
            qargs_ = list(qargs)

        from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList

        num_paulis = self._x.shape[0]

        ret = self.copy()
        ret._x[:, qargs_] = False
        ret._z[:, qargs_] = False

        idx = np.concatenate((self._x[:, qargs_], self._z[:, qargs_]), axis=1)
        # Only iterate rows of `other` selected by at least one Pauli in `self`:
        keep = np.nonzero(idx.any(axis=0))[0]
        for idx_, row in zip(
            idx[:, keep].T,
            PauliList.from_symplectic(
                z=other.z[keep], x=other.x[keep], phase=2 * other.phase[keep]
            ),
            strict=True,
        ):
            # most of the logic below is to properly index if self is a PauliList (2D),
            # while not trying to index if the object is just a Pauli (1D).
            if np.sum(idx_) == num_paulis:
                ret.compose(row, qargs=qargs, inplace=True)
            else:
                ret[idx_] = ret[idx_].compose(row, qargs=qargs)

        return ret

    def _eq(self, other):
        """Entrywise comparison of Pauli equality."""
        return (
            self.num_qubits == other.num_qubits
            and np.all(np.mod(self._phase, 4) == np.mod(other._phase, 4))
            and np.all(self._z == other._z)
            and np.all(self._x == other._x)
        )

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

    def __imul__(self, other):
        return self.compose(other, front=True, inplace=True)

    def __neg__(self):
        ret = copy.copy(self)
        ret._phase = np.mod(self._phase + 2, 4)
        return ret

    def _count_y(self, dtype=None):
        """Count the number of Y Paulis"""
        return _count_y(self._x, self._z, dtype=dtype)

    @staticmethod
    def _stack(array, size, vertical=True):
        """Stack array."""
        if size == 1:
            return array
        if vertical:
            return np.vstack(size * [array]).reshape((size * len(array),) + array.shape[1:])
        return np.hstack(size * [array]).reshape((size * len(array),) + array.shape[1:])

    @staticmethod
    def _phase_from_complex(coeff):
        """Return the phase from a label"""
        if np.isclose(coeff, 1):
            return 0
        if np.isclose(coeff, -1j):
            return 1
        if np.isclose(coeff, -1):
            return 2
        if np.isclose(coeff, 1j):
            return 3
        raise QiskitError("Pauli can only be multiplied by 1, -1j, -1, 1j.")

    @staticmethod
    def _from_array(z, x, phase=0):
        """Convert array data to BasePauli data."""
        if isinstance(z, np.ndarray) and z.dtype == bool:
            base_z = z
        else:
            base_z = np.asarray(z, dtype=bool)
        if base_z.ndim == 1:
            base_z = base_z.reshape((1, base_z.size))
        elif base_z.ndim != 2:
            raise QiskitError("Invalid Pauli z vector shape.")

        if isinstance(x, np.ndarray) and x.dtype == bool:
            base_x = x
        else:
            base_x = np.asarray(x, dtype=bool)
        if base_x.ndim == 1:
            base_x = base_x.reshape((1, base_x.size))
        elif base_x.ndim != 2:
            raise QiskitError("Invalid Pauli x vector shape.")

        if base_z.shape != base_x.shape:
            raise QiskitError("z and x vectors are different size.")

        # Convert group phase convention to internal ZX-phase conversion.
        dtype = getattr(phase, "dtype", None)
        base_phase = np.mod(_count_y(base_x, base_z, dtype=dtype) + phase, 4)
        return base_z, base_x, base_phase

    @staticmethod
    def _to_matrix(z, x, phase=0, group_phase=False, sparse=False):
        """Return the matrix from symplectic representation.

        The Pauli is defined as :math:`P = (-i)^{phase + z.x} * Z^z.x^x`
        where ``array = [x, z]``.

        Args:
            z (array): The symplectic representation z vector.
            x (array): The symplectic representation x vector.
            phase (int): Pauli phase.
            group_phase (bool): Optional. If ``True`` use group-phase convention
                                instead of BasePauli ZX-phase convention.
                                (default: ``False``).
            sparse (bool): Optional. If ``True`` return a sparse CSR matrix,
                           otherwise return a dense Numpy array
                           (default: ``False``).

        Returns:
            array: if ``sparse=False``.
            csr_matrix: if ``sparse=True``.
        """
        num_qubits = z.size

        # Convert to zx_phase
        if group_phase:
            phase += np.sum(x & z)
            phase %= 4

        dim = 2**num_qubits
        twos_array = 1 << np.arange(num_qubits, dtype=np.uint)
        x_indices = np.asarray(x).dot(twos_array)
        z_indices = np.asarray(z).dot(twos_array)

        indptr = np.arange(dim + 1, dtype=np.uint)
        indices = indptr ^ x_indices
        if phase:
            coeff = (-1j) ** phase
        else:
            coeff = 1

        # Compute parities of `z_indices & indptr`, i.e.,
        # np.array([(-1) ** bin(i).count("1") for i in z_indices & indptr])
        vec_u64 = z_indices & indptr
        mat_u8 = np.zeros((vec_u64.size, 8), dtype=np.uint8)
        for i in range(8):
            mat_u8[:, i] = vec_u64 & 255
            vec_u64 >>= 8
            if np.all(vec_u64 == 0):
                break
        parity = _PARITY[np.bitwise_xor.reduce(mat_u8, axis=1)]

        data = coeff * parity
        if sparse:
            # Return sparse matrix
            from scipy.sparse import csr_matrix

            return csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=complex)

        # Build dense matrix using csr format
        mat = np.zeros((dim, dim), dtype=complex)
        mat[range(dim), indices[:dim]] = data[:dim]
        return mat

    @staticmethod
    def _to_label(z, x, phase, group_phase=False, full_group=True, return_phase=False):
        """Return the label string for a Pauli.

        Args:
            z (array): The symplectic representation z vector.
            x (array): The symplectic representation x vector.
            phase (int): Pauli phase.
            group_phase (bool): Optional. If ``True`` use group-phase convention
                                instead of BasePauli ZX-phase convention.
                                (default: ``False``).
            full_group (bool): If True return the Pauli label from the full Pauli group
                including complex coefficient from [1, -1, 1j, -1j]. If
                ``False`` return the unsigned Pauli label with coefficient 1
                (default: ``True``).
            return_phase (bool): If ``True`` return the adjusted phase for the coefficient
                of the returned Pauli label. This can be used even if
                ``full_group=False``.

        Returns:
            str: the Pauli label from the full Pauli group (if ``full_group=True``) or
                from the unsigned Pauli group (if ``full_group=False``).
            tuple[str, int]: if ``return_phase=True`` returns a tuple of the Pauli
                            label (from either the full or unsigned Pauli group) and
                            the phase ``q`` for the coefficient :math:`(-i)^(q + x.z)`
                            for the label from the full Pauli group.
        """
        # Map each qubit to the {I: 0, X: 1, Z: 2, Y: 3} integer form, then use Numpy advanced
        # indexing to get a new data buffer which is compatible with an ASCII string label.
        index = z << 1
        index += x
        ascii_label = _TO_LABEL_CHARS[index[::-1]].data.tobytes()
        phase = (int(phase) if group_phase else int(phase) - ascii_label.count(b"Y")) % 4
        label = ascii_label.decode("ascii")
        if phase and full_group:
            label = ("", "-i", "-", "i")[phase] + label
        if return_phase:
            return label, phase
        return label

    def _append_circuit(self, circuit, qargs=None):
        """Update BasePauli inplace by applying a Clifford circuit.

        Args:
            circuit (QuantumCircuit or Instruction): the gate or composite gate to apply.
            qargs (list or None): The qubits to apply gate to.

        Returns:
            BasePauli: the updated Pauli.

        Raises:
            QiskitError: if input gate cannot be decomposed into Clifford gates.
        """
        if isinstance(circuit, (Barrier, Delay)):
            return self

        if qargs is None:
            qargs = list(range(self.num_qubits))

        if not isinstance(circuit, QuantumCircuit):
            gate = circuit

            if isinstance(gate, str):
                # Check if gate is a valid Clifford basis gate string
                if gate not in _basis_1q and gate not in _basis_2q:
                    raise QiskitError(f"Invalid Clifford gate name string {gate}")
                name = gate
            else:
                # Assume gate is an Instruction
                name = gate.name

            # Apply gate if it is a Clifford basis gate
            if name in _non_clifford:
                raise QiskitError(f"Cannot update Pauli with non-Clifford gate {name}")
            if name in _basis_1q:
                if len(qargs) != 1:
                    raise QiskitError("Invalid qubits for 1-qubit gate.")
                return _basis_1q[name](self, qargs[0])
            if name in _basis_2q:
                if len(qargs) != 2:
                    raise QiskitError("Invalid qubits for 2-qubit gate.")
                return _basis_2q[name](self, qargs[0], qargs[1])

            # If u gate, check if it is a Clifford, and if so, apply it
            if isinstance(gate, Gate) and name == "u" and len(qargs) == 1:
                try:
                    theta, phi, lam = tuple(_n_half_pis(par) for par in gate.params)
                    base_pauli = self
                except ValueError as err:
                    raise QiskitError(
                        "U gate angles must be multiples of pi/2 to be a Clifford"
                    ) from err
                if theta == 0:
                    base_pauli = _evolve_rz(base_pauli, qargs[0], lam + phi)
                elif theta == 1:
                    base_pauli = _evolve_rz(base_pauli, qargs[0], lam - 2)
                    base_pauli = _evolve_h(base_pauli, qargs[0])
                    base_pauli = _evolve_rz(base_pauli, qargs[0], phi)
                elif theta == 2:
                    base_pauli = _evolve_rz(base_pauli, qargs[0], lam - 1)
                    base_pauli = _evolve_x(base_pauli, qargs[0])
                    base_pauli = _evolve_rz(base_pauli, qargs[0], phi + 1)
                elif theta == 3:
                    base_pauli = _evolve_rz(base_pauli, qargs[0], lam)
                    base_pauli = _evolve_h(base_pauli, qargs[0])
                    base_pauli = _evolve_rz(base_pauli, qargs[0], phi + 2)
                return base_pauli

            # If not a Clifford basis gate we try to unroll the gate and
            # raise an exception if unrolling reaches a non-Clifford gate.
            if gate.definition is None:
                raise QiskitError(f"Cannot apply Instruction: {gate.name}")
            if not isinstance(gate.definition, QuantumCircuit):
                raise QiskitError(
                    f"{gate.name} instruction definition is {type(gate.definition)};"
                    f" expected QuantumCircuit"
                )

            circuit = gate.definition

        for instruction in circuit:
            if instruction.clbits:
                raise QiskitError(
                    f"Cannot apply Instruction with classical bits: {instruction.operation.name}"
                )
            # Get the integer position of the flat register
            new_qubits = [qargs[circuit.find_bit(qb)[0]] for qb in instruction.qubits]
            self._append_circuit(instruction.operation, new_qubits)

        # Since the individual gate evolution functions don't take mod
        # of phase we update it at the end
        self._phase %= 4
        return self


# ---------------------------------------------------------------------
# Evolution by Clifford gates
# ---------------------------------------------------------------------


def _evolve_h(base_pauli, qubit):
    """Update P -> H.P.H"""
    x = base_pauli._x[:, qubit].copy()
    z = base_pauli._z[:, qubit].copy()
    base_pauli._x[:, qubit] = z
    base_pauli._z[:, qubit] = x
    base_pauli._phase += 2 * np.logical_and(x, z).T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_s(base_pauli, qubit):
    """Update P -> S.P.Sdg"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase += x.T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_sdg(base_pauli, qubit):
    """Update P -> Sdg.P.S"""
    x = base_pauli._x[:, qubit]
    base_pauli._z[:, qubit] ^= x
    base_pauli._phase -= x.T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_sx(base_pauli, qubit):
    """Update P -> SX.P.SXdg"""
    z = base_pauli._z[:, qubit]
    base_pauli._x[:, qubit] ^= z
    base_pauli._phase -= z.T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_sxdg(base_pauli, qubit):
    """Update P -> SXdg.P.SX"""
    z = base_pauli._z[:, qubit]
    base_pauli._x[:, qubit] ^= z
    base_pauli._phase += z.T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_i(base_pauli, qubit):
    """Update P -> P"""
    return base_pauli


def _evolve_x(base_pauli, qubit):
    """Update P -> X.P.X"""
    base_pauli._phase += 2 * base_pauli._z[:, qubit].T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_y(base_pauli, qubit):
    """Update P -> Y.P.Y"""
    xp = base_pauli._x[:, qubit].T.astype(base_pauli._phase.dtype)
    zp = base_pauli._z[:, qubit].T.astype(base_pauli._phase.dtype)
    base_pauli._phase += 2 * (xp + zp)
    return base_pauli


def _evolve_z(base_pauli, qubit):
    """Update P -> Z.P.Z"""
    base_pauli._phase += 2 * base_pauli._x[:, qubit].T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_cx(base_pauli, qctrl, qtrgt):
    """Update P -> CX.P.CX"""
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    return base_pauli


def _evolve_cz(base_pauli, q1, q2):
    """Update P -> CZ.P.CZ"""
    x1 = base_pauli._x[:, q1].copy()
    x2 = base_pauli._x[:, q2].copy()
    base_pauli._z[:, q1] ^= x2
    base_pauli._z[:, q2] ^= x1
    base_pauli._phase += 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_cy(base_pauli, qctrl, qtrgt):
    """Update P -> CY.P.CY"""
    x1 = base_pauli._x[:, qctrl].copy()
    x2 = base_pauli._x[:, qtrgt].copy()
    z2 = base_pauli._z[:, qtrgt].copy()
    base_pauli._x[:, qtrgt] ^= x1
    base_pauli._z[:, qtrgt] ^= x1
    base_pauli._z[:, qctrl] ^= np.logical_xor(x2, z2)
    base_pauli._phase += x1 + 2 * np.logical_and(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_dcx(base_pauli, qctrl, qtrgt):
    """Update P -> DCX.P.DCXdg"""
    base_pauli._x[:, qtrgt] ^= base_pauli._x[:, qctrl]
    base_pauli._z[:, qctrl] ^= base_pauli._z[:, qtrgt]
    base_pauli._x[:, qctrl] ^= base_pauli._x[:, qtrgt]
    base_pauli._z[:, qtrgt] ^= base_pauli._z[:, qctrl]
    return base_pauli


def _evolve_swap(base_pauli, q1, q2):
    """Update P -> SWAP.P.SWAP"""
    x1 = base_pauli._x[:, q1].copy()
    z1 = base_pauli._z[:, q1].copy()
    base_pauli._x[:, q1] = base_pauli._x[:, q2]
    base_pauli._z[:, q1] = base_pauli._z[:, q2]
    base_pauli._x[:, q2] = x1
    base_pauli._z[:, q2] = z1
    return base_pauli


def _evolve_iswap(base_pauli, q1, q2):
    """Update P -> iSWAP.P.iSWAP"""
    x1 = base_pauli._x[:, q1].copy()
    z1 = base_pauli._z[:, q1].copy()
    x2 = base_pauli._x[:, q2].copy()
    z2 = base_pauli._z[:, q2].copy()

    base_pauli._x[:, q1] = x2
    base_pauli._x[:, q2] = x1
    base_pauli._z[:, q1] = x1 ^ x2 ^ z2
    base_pauli._z[:, q2] = x1 ^ x2 ^ z1

    base_pauli._phase += np.logical_xor(x1, x2).T.astype(base_pauli._phase.dtype)
    return base_pauli


def _evolve_ecr(base_pauli, q1, q2):
    """Update P -> ECR.P.ECR"""
    base_pauli = _evolve_s(base_pauli, q1)
    base_pauli = _evolve_h(base_pauli, q2)
    base_pauli = _evolve_s(base_pauli, q2)
    base_pauli = _evolve_h(base_pauli, q2)
    base_pauli = _evolve_cx(base_pauli, q1, q2)
    base_pauli = _evolve_x(base_pauli, q1)
    return base_pauli


def _evolve_rz(base_pauli, qubit, multiple):
    """Update P -> RZ.P.RZ, with an angle equals to a multiple of pi/2"""
    if multiple % 4 == 1:
        return _evolve_s(base_pauli, qubit)
    if multiple % 4 == 2:
        return _evolve_z(base_pauli, qubit)
    if multiple % 4 == 3:
        return _evolve_sdg(base_pauli, qubit)
    return base_pauli


def _count_y(x, z, dtype=None):
    """Count the number of Y Paulis"""
    return (x & z).sum(axis=1, dtype=dtype)


# Basis Clifford Gates
_basis_1q = {
    "i": _evolve_i,
    "id": _evolve_i,
    "iden": _evolve_i,
    "x": _evolve_x,
    "y": _evolve_y,
    "z": _evolve_z,
    "h": _evolve_h,
    "s": _evolve_s,
    "sdg": _evolve_sdg,
    "sinv": _evolve_sdg,
    "sx": _evolve_sx,
    "sxdg": _evolve_sxdg,
}
_basis_2q = {
    "cx": _evolve_cx,
    "cz": _evolve_cz,
    "cy": _evolve_cy,
    "swap": _evolve_swap,
    "iswap": _evolve_iswap,
    "ecr": _evolve_ecr,
    "dcx": _evolve_dcx,
}

# Non-Clifford gates
_non_clifford = ["t", "tdg", "ccx", "ccz"]
