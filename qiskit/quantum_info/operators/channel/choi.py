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
Choi-matrix representation of a Quantum Channel.
"""

from __future__ import annotations
import copy as _copy
import math
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator

if TYPE_CHECKING:
    from qiskit import circuit


class Choi(QuantumChannel):
    r"""Choi-matrix representation of a Quantum Channel.

    The Choi-matrix representation of a quantum channel :math:`\mathcal{E}`
    is a matrix

    .. math::

        \Lambda = \sum_{i,j} |i\rangle\!\langle j|\otimes
                    \mathcal{E}\left(|i\rangle\!\langle j|\right)

    Evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` with respect to the Choi-matrix is given by

    .. math::

        \mathcal{E}(\rho) = \mbox{Tr}_{1}\left[\Lambda
                            (\rho^T \otimes \mathbb{I})\right]

    where :math:`\mbox{Tr}_1` is the :func:`partial_trace` over subsystem 1.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(
        self,
        data: QuantumCircuit | circuit.instruction.Instruction | BaseOperator | np.ndarray,
        input_dims: int | tuple | None = None,
        output_dims: int | tuple | None = None,
    ):
        """Initialize a quantum channel Choi matrix operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data cannot be initialized as a
                         Choi matrix.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (4**N, 4**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Choi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Initialize from raw numpy or list matrix.
            choi_mat = np.asarray(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = choi_mat.shape
            if dim_l != dim_r:
                raise QiskitError("Invalid Choi-matrix input.")
            if input_dims:
                input_dim = np.prod(input_dims)
            if output_dims:
                output_dim = np.prod(output_dims)
            if output_dims is None and input_dims is None:
                output_dim = int(math.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dims is None:
                input_dim = dim_l // output_dim
            elif output_dims is None:
                output_dim = dim_l // input_dim
            # Check dimensions
            if input_dim * output_dim != dim_l:
                raise QiskitError("Invalid shape for input Choi-matrix.")
            op_shape = OpShape.auto(
                dims_l=output_dims, dims_r=input_dims, shape=(output_dim, input_dim)
            )
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
            # Now that the input is an operator we convert it to a Choi object
            rep = getattr(data, "_channel_rep", "Operator")
            choi_mat = _to_choi(rep, data._data, input_dim, output_dim)
        super().__init__(choi_mat, op_shape=op_shape)

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim, self._input_dim, self._output_dim)

    def _evolve(self, state, qargs=None):
        return SuperOp(self)._evolve(state, qargs)

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        ret = _copy.copy(self)
        ret._data = np.conj(self._data)
        return ret

    def transpose(self):
        ret = _copy.copy(self)
        ret._op_shape = self._op_shape.transpose()
        # Make bipartite matrix
        d_in, d_out = self.dim
        data = np.reshape(self._data, (d_in, d_out, d_in, d_out))
        # Swap input and output indices on bipartite matrix
        data = np.transpose(data, (1, 0, 3, 2))
        ret._data = np.reshape(data, (d_in * d_out, d_in * d_out))
        return ret

    def compose(self, other: Choi, qargs: list | None = None, front: bool = False) -> Choi:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if qargs is not None:
            return Choi(SuperOp(self).compose(other, qargs=qargs, front=front))

        if not isinstance(other, Choi):
            other = Choi(other)
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        output_dim, input_dim = new_shape.shape

        if front:
            first = np.reshape(other._data, other._bipartite_shape)
            second = np.reshape(self._data, self._bipartite_shape)
        else:
            first = np.reshape(self._data, self._bipartite_shape)
            second = np.reshape(other._data, other._bipartite_shape)

        # Contract Choi matrices for composition
        data = np.reshape(
            np.einsum("iAjB,AkBl->ikjl", first, second),
            (input_dim * output_dim, input_dim * output_dim),
        )
        ret = Choi(data)
        ret._op_shape = new_shape
        return ret

    def tensor(self, other: Choi) -> Choi:
        if not isinstance(other, Choi):
            other = Choi(other)
        return self._tensor(self, other)

    def expand(self, other: Choi) -> Choi:
        if not isinstance(other, Choi):
            other = Choi(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = _copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = _bipartite_tensor(
            a._data, b.data, shape1=a._bipartite_shape, shape2=b._bipartite_shape
        )
        return ret


# Update docstrings for API docs
generate_apidocs(Choi)
