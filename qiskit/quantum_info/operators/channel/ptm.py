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
Pauli Transfer Matrix (PTM) representation of a Quantum Channel.
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
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_ptm
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator

if TYPE_CHECKING:
    from qiskit import circuit


class PTM(QuantumChannel):
    r"""Pauli Transfer Matrix (PTM) representation of a Quantum Channel.

    The PTM representation of an :math:`n`-qubit quantum channel
    :math:`\mathcal{E}` is an :math:`n`-qubit :class:`SuperOp` :math:`R`
    defined with respect to vectorization in the Pauli basis instead of
    column-vectorization. The elements of the PTM :math:`R` are
    given by

    .. math::

        R_{i,j} = \frac{1}{2^n} \mbox{Tr}\left[P_i \mathcal{E}(P_j) \right]

    where :math:`[P_0, P_1, ..., P_{4^{n}-1}]` is the :math:`n`-qubit Pauli basis in
    lexicographic order.

    Evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` with respect to the PTM is given by

    .. math::

        |\mathcal{E}(\rho)\rangle\!\rangle_P = S_P |\rho\rangle\!\rangle_P

    where :math:`|A\rangle\!\rangle_P` denotes vectorization in the Pauli basis
    :math:`\langle i | A\rangle\!\rangle_P = \sqrt{\frac{1}{2^n}} \mbox{Tr}[P_i A]`.

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
        """Initialize a PTM quantum channel operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data is not an N-qubit channel or
                         cannot be initialized as a PTM.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. The PTM
            representation is only valid for N-qubit channels.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Chi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Should we force this to be real?
            ptm = np.asarray(data, dtype=complex)
            # Determine input and output dimensions
            dout, din = ptm.shape
            if input_dims:
                input_dim = np.prod(input_dims)
            else:
                input_dim = int(math.sqrt(din))
            if output_dims:
                output_dim = np.prod(input_dims)
            else:
                output_dim = int(math.sqrt(dout))
            if output_dim**2 != dout or input_dim**2 != din or input_dim != output_dim:
                raise QiskitError("Invalid shape for PTM matrix.")
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
            input_dim, output_dim = data.dim
            # Now that the input is an operator we convert it to a PTM object
            rep = getattr(data, "_channel_rep", "Operator")
            ptm = _to_ptm(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check input is N-qubit channel
        num_qubits = int(math.log2(input_dim))
        if 2**num_qubits != input_dim or input_dim != output_dim:
            raise QiskitError("Input is not an n-qubit Pauli transfer matrix.")
        super().__init__(ptm, num_qubits=num_qubits)

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim, self._input_dim)

    def _evolve(self, state, qargs=None):
        return SuperOp(self)._evolve(state, qargs)

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        # Since conjugation is basis dependent we transform
        # to the SuperOp representation to compute the
        # conjugate channel
        return PTM(SuperOp(self).conjugate())

    def transpose(self):
        return PTM(SuperOp(self).transpose())

    def adjoint(self):
        return PTM(SuperOp(self).adjoint())

    def compose(self, other: PTM, qargs: list | None = None, front: bool = False) -> PTM:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if qargs is not None:
            return PTM(SuperOp(self).compose(other, qargs=qargs, front=front))

        # Convert other to PTM
        if not isinstance(other, PTM):
            other = PTM(other)
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()
        if front:
            data = np.dot(self._data, other.data)
        else:
            data = np.dot(other.data, self._data)
        ret = PTM(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def tensor(self, other: PTM) -> PTM:
        if not isinstance(other, PTM):
            other = PTM(other)
        return self._tensor(self, other)

    def expand(self, other: PTM) -> PTM:
        if not isinstance(other, PTM):
            other = PTM(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = _copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = np.kron(a._data, b.data)
        return ret


# Update docstrings for API docs
generate_apidocs(PTM)
