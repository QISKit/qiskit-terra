# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract base class for Quantum Channels.
"""

from __future__ import annotations
import copy
import math
from abc import abstractmethod
from numbers import Number, Integral

import numpy as np

from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.transformations import _transform_rep
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.channel.transformations import _to_operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp

from typing import Self


class QuantumChannel(LinearOp):
    """Quantum channel representation base class."""

    def __init__(
        self,
        data: list | np.ndarray,
        num_qubits: int | None = None,
        op_shape: OpShape | None = None,
    ):
        """Initialize a quantum channel Superoperator operator.

        Args:
            data: quantum channel data array.
            op_shape: the operator shape of the channel.
            num_qubits: the number of qubits if the channel is N-qubit.

        Raises:
            QiskitError: if arguments are invalid.
        """
        self._data = data
        super().__init__(num_qubits=num_qubits, op_shape=op_shape)

    def __repr__(self):
        prefix = f"{self._channel_rep}("
        pad = len(prefix) * " "
        return (
            f"{prefix}{np.array2string(np.asarray(self.data), separator=', ', prefix=prefix)}"
            f",\n{pad}input_dims={self.input_dims()}, output_dims={self.output_dims()})"
        )

    def __eq__(self, other: Self):
        """Test if two QuantumChannels are equal."""
        if not super().__eq__(other):
            return False
        return np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)

    @property
    def data(self):
        """Return data."""
        return self._data

    @property
    def _channel_rep(self):
        """Return channel representation string"""
        return type(self).__name__

    @property
    def settings(self):
        """Return settings."""
        return {
            "data": self.data,
            "input_dims": self.input_dims(),
            "output_dims": self.output_dims(),
        }

    # ---------------------------------------------------------------------
    # LinearOp methods
    # ---------------------------------------------------------------------

    @abstractmethod
    def conjugate(self):
        r"""Return the conjugate quantum channel.

        .. note::
            This is equivalent to the matrix complex conjugate in the
            :class:`~qiskit.quantum_info.SuperOp` representation
            ie. for a channel :math:`\mathcal{E}`, the SuperOp of
            the conjugate channel :math:`\overline{{\mathcal{{E}}}}` is
            :math:`S_{\overline{\mathcal{E}^\dagger}} = \overline{S_{\mathcal{E}}}`.
        """

    @abstractmethod
    def transpose(self) -> Self:
        r"""Return the transpose quantum channel.

        .. note::
            This is equivalent to the matrix transpose in the
            :class:`~qiskit.quantum_info.SuperOp` representation,
            ie. for a channel :math:`\mathcal{E}`, the SuperOp of
            the transpose channel :math:`\mathcal{{E}}^T` is
            :math:`S_{\mathcal{E}^T} = S_{\mathcal{E}}^T`.
        """

    def adjoint(self) -> Self:
        r"""Return the adjoint quantum channel.

        .. note::
            This is equivalent to the matrix Hermitian conjugate in the
            :class:`~qiskit.quantum_info.SuperOp` representation
            ie. for a channel :math:`\mathcal{E}`, the SuperOp of
            the adjoint channel :math:`\mathcal{{E}}^\dagger` is
            :math:`S_{\mathcal{E}^\dagger} = S_{\mathcal{E}}^\dagger`.
        """
        return self.conjugate().transpose()

    def power(self, n: float) -> Self:
        r"""Return the power of the quantum channel.

        Args:
            n (float): the power exponent.

        Returns:
            CLASS: the channel :math:`\mathcal{{E}} ^n`.

        Raises:
            QiskitError: if the input and output dimensions of the
                         CLASS are not equal.

        .. note::
            For non-positive or non-integer exponents the power is
            defined as the matrix power of the
            :class:`~qiskit.quantum_info.SuperOp` representation
            ie. for a channel :math:`\mathcal{{E}}`, the SuperOp of
            the powered channel :math:`\mathcal{{E}}^\n` is
            :math:`S_{{\mathcal{{E}}^n}} = S_{{\mathcal{{E}}}}^n`.
        """
        if n > 0 and isinstance(n, Integral):
            return super().power(n)

        # Conversion to superoperator
        if self._input_dim != self._output_dim:
            raise QiskitError("Can only take power with input_dim = output_dim.")
        rep = self._channel_rep
        input_dim, output_dim = self.dim
        superop = _transform_rep(rep, "SuperOp", self._data, input_dim, output_dim)
        superop = np.linalg.matrix_power(superop, n)

        # Convert back to original representation
        ret = copy.copy(self)
        ret._data = _transform_rep("SuperOp", rep, superop, input_dim, output_dim)
        return ret

    def __sub__(self, other) -> Self:
        qargs = getattr(other, "qargs", None)
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return self._add(-other, qargs=qargs)

    def _add(self, other, qargs=None):
        # NOTE: this method must be overridden for subclasses
        # that don't have a linear matrix representation
        # ie Kraus and Stinespring
        if not isinstance(other, type(self)):
            other = type(self)(other)
        self._op_shape._validate_add(other._op_shape, qargs)
        other = ScalarOp._pad_with_identity(self, other, qargs)
        ret = copy.copy(self)
        ret._data = self._data + other._data
        return ret

    def _multiply(self, other):
        # NOTE: this method must be overridden for subclasses
        # that don't have a linear matrix representation
        # ie Kraus and Stinespring
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = copy.copy(self)
        ret._data = other * self._data
        return ret

    # ---------------------------------------------------------------------
    # Additional methods
    # ---------------------------------------------------------------------

    def is_cptp(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Return True if completely-positive trace-preserving (CPTP)."""
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol) and self._is_tp_helper(choi, atol, rtol)

    def is_tp(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Test if a channel is trace-preserving (TP)"""
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_tp_helper(choi, atol, rtol)

    def is_cp(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Test if Choi-matrix is completely-positive (CP)"""
        choi = _to_choi(self._channel_rep, self._data, *self.dim)
        return self._is_cp_helper(choi, atol, rtol)

    def is_unitary(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Return True if QuantumChannel is a unitary channel."""
        try:
            op = self.to_operator()
            return op.is_unitary(atol=atol, rtol=rtol)
        except QiskitError:
            return False

    def to_operator(self) -> Operator:
        """Try to convert channel to a unitary representation Operator."""
        mat = _to_operator(self._channel_rep, self._data, *self.dim)
        return Operator(mat, self.input_dims(), self.output_dims())

    def to_instruction(self) -> Instruction:
        """Convert to a Kraus or UnitaryGate circuit instruction.

        If the channel is unitary it will be added as a unitary gate,
        otherwise it will be added as a kraus simulator instruction.

        Returns:
            qiskit.circuit.Instruction: A kraus instruction for the channel.

        Raises:
            QiskitError: if input data is not an N-qubit CPTP quantum channel.
        """

        # Check if input is an N-qubit CPTP channel.
        num_qubits = int(math.log2(self._input_dim))
        if self._input_dim != self._output_dim or 2**num_qubits != self._input_dim:
            raise QiskitError(
                "Cannot convert QuantumChannel to Instruction: channel is not an N-qubit channel."
            )
        if not self.is_cptp():
            raise QiskitError("Cannot convert QuantumChannel to Instruction: channel is not CPTP.")
        # Next we convert to the Kraus representation. Since channel is CPTP we know
        # that there is only a single set of Kraus operators
        kraus, _ = _to_kraus(self._channel_rep, self._data, *self.dim)
        # If we only have a single Kraus operator then the channel is
        # a unitary channel so can be converted to a UnitaryGate. We do this by
        # converting to an Operator and using its to_instruction method
        if len(kraus) == 1:
            return Operator(kraus[0]).to_instruction()
        return Instruction("kraus", num_qubits, 0, kraus)

    def _is_cp_helper(self, choi, atol, rtol):
        """Test if a channel is completely-positive (CP)"""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return is_positive_semidefinite_matrix(choi, rtol=rtol, atol=atol)

    def _is_tp_helper(self, choi, atol, rtol):
        """Test if Choi-matrix is trace-preserving (TP)"""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        # Check if the partial trace is the identity matrix
        d_in, d_out = self.dim
        mat = np.trace(np.reshape(choi, (d_in, d_out, d_in, d_out)), axis1=1, axis2=3)
        tp_cond = np.linalg.eigvalsh(mat - np.eye(len(mat)))
        zero = np.isclose(tp_cond, 0, atol=atol, rtol=rtol)
        return np.all(zero)

    def _format_state(self, state, density_matrix=False):
        """Format input state so it is statevector or density matrix"""
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError("Input state is not a vector or matrix.")
        # Flatten column-vector to vector
        if ndim == 2:
            if shape[1] != 1 and shape[1] != shape[0]:
                raise QiskitError("Input state is not a vector or matrix.")
            if shape[1] == 1:
                # flatten column-vector to vector
                state = np.reshape(state, shape[0])
        # Convert statevector to density matrix if required
        if density_matrix and ndim == 1:
            state = np.outer(state, np.transpose(np.conj(state)))
        return state

    @abstractmethod
    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the quantum channel.

        Args:
            state (DensityMatrix or Statevector): The input state.
            qargs (list): a list of quantum state subsystem positions to apply
                           the quantum channel on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the quantum channel dimension does not match the
                         specified quantum state subsystem dimensions.
        """

    @classmethod
    def _init_transformer(cls, data):
        """Convert input into a QuantumChannel subclass object or Operator object"""
        # This handles common conversion for all QuantumChannel subclasses.
        # If the input is already a QuantumChannel subclass it will return
        # the original object
        if isinstance(data, QuantumChannel):
            return data
        if hasattr(data, "to_quantumchannel"):
            # If the data object is not a QuantumChannel it will give
            # preference to a 'to_quantumchannel' attribute that allows
            # an arbitrary object to define its own conversion to any
            # quantum channel subclass.
            return data.to_quantumchannel()
        if hasattr(data, "to_channel"):
            # TODO: this 'to_channel' method is the same case as the above
            # but is used by current version of Aer. It should be removed
            # once Aer is updated to use `to_quantumchannel`
            # instead of `to_channel`,
            return data.to_channel()
        # Finally if the input is not a QuantumChannel and doesn't have a
        # 'to_quantumchannel' conversion method we try and initialize it as a
        # regular matrix Operator which can be converted into a QuantumChannel.
        return Operator(data)
