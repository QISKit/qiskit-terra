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
Kraus representation of a Quantum Channel.
"""

from __future__ import annotations
import copy
import math
from numbers import Number
import numpy as np

from qiskit import circuit
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_kraus
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator


class Kraus(QuantumChannel):
    r"""Kraus representation of a quantum channel.

    For a quantum channel :math:`\mathcal{E}`, the Kraus representation is
    given by a set of matrices :math:`[A_0,...,A_{K-1}]` such that the
    evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` is given by

    .. math::

        \mathcal{E}(\rho) = \sum_{i=0}^{K-1} A_i \rho A_i^\dagger

    A general operator map :math:`\mathcal{G}` can also be written using the
    generalized Kraus representation which is given by two sets of matrices
    :math:`[A_0,...,A_{K-1}]`, :math:`[B_0,...,B_{K-1}]` such that

    .. math::

        \mathcal{G}(\rho) = \sum_{i=0}^{K-1} A_i \rho B_i^\dagger

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(
        self,
        data: QuantumCircuit | circuit.instruction.Instruction | BaseOperator | np.ndarray,
        input_dims: tuple | None = None,
        output_dims: tuple | None = None,
    ):
        """Initialize a quantum channel Kraus operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data cannot be initialized as a list of Kraus matrices.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a list of Numpy arrays of shape :math:`(2^N,\\,2^N)` qubit systems will be
            used. If the input does not correspond to an N-qubit channel, it
            will assign a single subsystem with dimension specified by the
            shape of the input.
        """
        # If the input is a list or tuple we assume it is a list of Kraus
        # matrices, if it is a numpy array we assume that it is a single Kraus
        # operator
        # TODO properly handle array construction from ragged data (like tuple(np.ndarray, None))
        # and document these accepted input cases. See also Qiskit/qiskit-terra#9307.
        if isinstance(data, (list, tuple, np.ndarray)):
            # Check if it is a single unitary matrix A for channel:
            # E(rho) = A * rho * A^\dagger
            if _is_matrix(data):
                # Convert single Kraus op to general Kraus pair
                kraus = ([np.asarray(data, dtype=complex)], None)
                shape = kraus[0][0].shape

            # Check if single Kraus set [A_i] for channel:
            # E(rho) = sum_i A_i * rho * A_i^dagger
            elif isinstance(data, list) and len(data) > 0:
                # Get dimensions from first Kraus op
                kraus = [np.asarray(data[0], dtype=complex)]
                shape = kraus[0].shape
                # Iterate over remaining ops and check they are same shape
                for i in data[1:]:
                    op = np.asarray(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError("Kraus operators are different dimensions.")
                    kraus.append(op)
                # Convert single Kraus set to general Kraus pair
                kraus = (kraus, None)

            # Check if generalized Kraus set ([A_i], [B_i]) for channel:
            # E(rho) = sum_i A_i * rho * B_i^dagger
            elif isinstance(data, tuple) and len(data) == 2 and len(data[0]) > 0:
                kraus_left = [np.asarray(data[0][0], dtype=complex)]
                shape = kraus_left[0].shape
                for i in data[0][1:]:
                    op = np.asarray(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError("Kraus operators are different dimensions.")
                    kraus_left.append(op)
                if data[1] is None:
                    kraus = (kraus_left, None)
                else:
                    kraus_right = []
                    for i in data[1]:
                        op = np.asarray(i, dtype=complex)
                        if op.shape != shape:
                            raise QiskitError("Kraus operators are different dimensions.")
                        kraus_right.append(op)
                    kraus = (kraus_left, kraus_right)
            else:
                raise QiskitError("Invalid input for Kraus channel.")
            op_shape = OpShape.auto(dims_l=output_dims, dims_r=input_dims, shape=kraus[0][0].shape)
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # convert it to a SuperOp
                data = SuperOp._init_instruction(data)
            else:
                # We use the QuantumChannel init transform to initialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            op_shape = data._op_shape
            output_dim, input_dim = op_shape.shape
            # Now that the input is an operator we convert it to a Kraus
            rep = getattr(data, "_channel_rep", "Operator")
            kraus = _to_kraus(rep, data._data, input_dim, output_dim)

        # Initialize either single or general Kraus
        if kraus[1] is None or np.allclose(kraus[0], kraus[1]):
            # Standard Kraus map
            data = (kraus[0], None)
        else:
            # General (non-CPTP) Kraus map
            data = kraus
        super().__init__(data, op_shape=op_shape)

    @property
    def data(self):
        """Return list of Kraus matrices for channel."""
        if self._data[1] is None:
            # If only a single Kraus set, don't return the tuple
            # Just the fist set
            return self._data[0]
        else:
            # Otherwise return the tuple of both kraus sets
            return self._data

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving."""
        if self._data[1] is not None:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        accum = 0j
        for op in self._data[0]:
            accum += np.dot(np.transpose(np.conj(op)), op)
        return is_identity_matrix(accum, rtol=rtol, atol=atol)

    def _evolve(self, state, qargs=None):
        return SuperOp(self)._evolve(state, qargs)

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        ret = copy.copy(self)
        kraus_l, kraus_r = self._data
        kraus_l = [np.conj(k) for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.conj() for k in kraus_r]
        ret._data = (kraus_l, kraus_r)
        return ret

    def transpose(self):
        ret = copy.copy(self)
        ret._op_shape = self._op_shape.transpose()
        kraus_l, kraus_r = self._data
        kraus_l = [np.transpose(k) for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [np.transpose(k) for k in kraus_r]
        ret._data = (kraus_l, kraus_r)
        return ret

    def adjoint(self):
        ret = copy.copy(self)
        ret._op_shape = self._op_shape.transpose()
        kraus_l, kraus_r = self._data
        kraus_l = [np.conj(np.transpose(k)) for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [np.conj(np.transpose(k)) for k in kraus_r]
        ret._data = (kraus_l, kraus_r)
        return ret

    def compose(self, other: Kraus, qargs: list | None = None, front: bool = False) -> Kraus:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if qargs is not None:
            return Kraus(SuperOp(self).compose(other, qargs=qargs, front=front))

        if not isinstance(other, Kraus):
            other = Kraus(other)
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()

        if front:
            ka_l, ka_r = self._data
            kb_l, kb_r = other._data
        else:
            ka_l, ka_r = other._data
            kb_l, kb_r = self._data

        kab_l = [np.dot(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        elif ka_r is None:
            kab_r = [np.dot(a, b) for a in ka_l for b in kb_r]
        elif kb_r is None:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_l]
        else:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_r]
        ret = Kraus((kab_l, kab_r), input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def tensor(self, other: Kraus) -> Kraus:
        if not isinstance(other, Kraus):
            other = Kraus(other)
        return self._tensor(self, other)

    def expand(self, other: Kraus) -> Kraus:
        if not isinstance(other, Kraus):
            other = Kraus(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)

        # Get tensor matrix
        ka_l, ka_r = a._data
        kb_l, kb_r = b._data
        kab_l = [np.kron(ka, kb) for ka in ka_l for kb in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        else:
            if ka_r is None:
                ka_r = ka_l
            if kb_r is None:
                kb_r = kb_l
            kab_r = [np.kron(a, b) for a in ka_r for b in kb_r]
        ret._data = (kab_l, kab_r)
        return ret

    def __add__(self, other):
        qargs = getattr(other, "qargs", None)
        if not isinstance(other, QuantumChannel):
            other = Choi(other)
        return self._add(other, qargs=qargs)

    def __sub__(self, other):
        qargs = getattr(other, "qargs", None)
        if not isinstance(other, QuantumChannel):
            other = Choi(other)
        return self._add(-other, qargs=qargs)

    def _add(self, other, qargs=None):
        # Since we cannot directly add two channels in the Kraus
        # representation we try and use the other channels method
        # or convert to the Choi representation
        return Kraus(Choi(self)._add(other, qargs=qargs))

    def _multiply(self, other):
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")

        ret = copy.copy(self)
        # If the number is complex we need to convert to general
        # kraus channel so we multiply via Choi representation
        if isinstance(other, complex) or other < 0:
            # Convert to Choi-matrix
            ret._data = Kraus(Choi(self)._multiply(other))._data
            return ret
        # If the number is real we can update the Kraus operators
        # directly
        val = math.sqrt(other)
        kraus_r = None
        kraus_l = [val * k for k in self._data[0]]
        if self._data[1] is not None:
            kraus_r = [val * k for k in self._data[1]]
        ret._data = (kraus_l, kraus_r)
        return ret


def _is_matrix(data):
    # return True if data is a 2-d array/tuple/list
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=object)

    return data.ndim == 2


# Update docstrings for API docs
generate_apidocs(Kraus)
