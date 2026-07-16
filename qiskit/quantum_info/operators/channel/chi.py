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
Chi-matrix representation of a Quantum Channel.
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
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_chi
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator

if TYPE_CHECKING:
    from qiskit import circuit


class Chi(QuantumChannel):
    r"""Pauli basis Chi-matrix representation of a quantum channel.

    The Chi-matrix representation of an :math:`n`-qubit quantum channel
    :math:`\mathcal{E}` is a matrix :math:`\chi` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        \mathcal{E}(ρ) = \frac{1}{2^n} \sum_{i, j} \chi_{i,j} P_i ρ P_j

    where :math:`[P_0, P_1, ..., P_{4^{n}-1}]` is the :math:`n`-qubit Pauli basis in
    lexicographic order. It is related to the :class:`Choi` representation by a change
    of basis of the Choi-matrix into the Pauli basis. The :math:`\frac{1}{2^n}`
    in the definition above is a normalization factor that arises from scaling the
    Pauli basis to make it orthonormal.

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
        """Initialize a quantum channel Chi-matrix operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data is not an N-qubit channel or
                         cannot be initialized as a Chi-matrix.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. The Chi matrix
            representation is only valid for N-qubit channels.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Chi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Initialize from raw numpy or list matrix.
            chi_mat = np.asarray(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = chi_mat.shape
            if dim_l != dim_r:
                raise QiskitError("Invalid Chi-matrix input.")
            if input_dims:
                input_dim = np.prod(input_dims)
            if output_dims:
                output_dim = np.prod(input_dims)
            if output_dims is None and input_dims is None:
                output_dim = int(math.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dims is None:
                input_dim = dim_l // output_dim
            elif output_dims is None:
                output_dim = dim_l // input_dim
            # Check dimensions
            if input_dim * output_dim != dim_l:
                raise QiskitError("Invalid shape for Chi-matrix input.")
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
            # Now that the input is an operator we convert it to a Chi object
            rep = getattr(data, "_channel_rep", "Operator")
            chi_mat = _to_chi(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check input is N-qubit channel
        num_qubits = int(math.log2(input_dim))
        if 2**num_qubits != input_dim or input_dim != output_dim:
            raise QiskitError("Input is not an n-qubit Chi matrix.")
        super().__init__(chi_mat, num_qubits=num_qubits)

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype
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
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        return Chi(Choi(self).conjugate())

    def transpose(self):
        return Chi(Choi(self).transpose())

    def adjoint(self):
        return Chi(Choi(self).adjoint())

    def compose(self, other: Chi, qargs: list | None = None, front: bool = False) -> Chi:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if qargs is not None:
            return Chi(SuperOp(self).compose(other, qargs=qargs, front=front))
        # If no qargs we compose via Choi representation to avoid an additional
        # representation conversion to SuperOp and then convert back to Chi
        return Chi(Choi(self).compose(other, front=front))

    def tensor(self, other: Chi) -> Chi:
        if not isinstance(other, Chi):
            other = Chi(other)
        return self._tensor(self, other)

    def expand(self, other: Chi) -> Chi:
        if not isinstance(other, Chi):
            other = Chi(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = _copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = np.kron(a._data, b.data)
        return ret


# Update docstrings for API docs
generate_apidocs(Chi)
