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
Stinespring representation of a Quantum Channel.
"""

from __future__ import annotations
import copy
import math
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_stinespring
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.base_operator import BaseOperator

if TYPE_CHECKING:
    from qiskit import circuit


class Stinespring(QuantumChannel):
    r"""Stinespring representation of a quantum channel.

    The Stinespring representation of a quantum channel :math:`\mathcal{E}`
    is a rectangular matrix :math:`A` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        \mathcal{E}(ρ) = \mbox{Tr}_2\left[A ρ A^\dagger\right]

    where :math:`\mbox{Tr}_2` is the :func:`partial_trace` over subsystem 2.

    A general operator map :math:`\mathcal{G}` can also be written using the
    generalized Stinespring representation which is given by two matrices
    :math:`A`, :math:`B` such that

    .. math::

        \mathcal{G}(ρ) = \mbox{Tr}_2\left[A ρ B^\dagger\right]

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
        """Initialize a quantum channel Stinespring operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data cannot be initialized as
                         a list of Kraus matrices.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. This can fail for the
            Stinespring operator if the output dimension cannot be automatically
            determined.
        """
        # If the input is a list or tuple we assume it is a pair of general
        # Stinespring matrices. If it is a numpy array we assume that it is
        # a single Stinespring matrix.
        if isinstance(data, (list, tuple, np.ndarray)):
            if not isinstance(data, tuple):
                # Convert single Stinespring set to length 1 tuple
                stine = (np.asarray(data, dtype=complex), None)
            if isinstance(data, tuple) and len(data) == 2:
                if data[1] is None:
                    stine = (np.asarray(data[0], dtype=complex), None)
                else:
                    stine = (np.asarray(data[0], dtype=complex), np.asarray(data[1], dtype=complex))

            dim_left, dim_right = stine[0].shape
            # If two Stinespring matrices check they are same shape
            if stine[1] is not None:
                if stine[1].shape != (dim_left, dim_right):
                    raise QiskitError("Invalid Stinespring input.")
            input_dim = dim_right
            if output_dims:
                output_dim = np.prod(output_dims)
            else:
                output_dim = input_dim
            if dim_left % output_dim != 0:
                raise QiskitError("Invalid output_dim")
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
            # Now that the input is an operator we convert it to a
            # Stinespring operator
            rep = getattr(data, "_channel_rep", "Operator")
            stine = _to_stinespring(rep, data._data, input_dim, output_dim)

        # Initialize either single or general Stinespring
        if stine[1] is None or (stine[1] == stine[0]).all():
            # Standard Stinespring map
            data = (stine[0], None)
        else:
            # General (non-CPTP) Stinespring map
            data = stine
        super().__init__(data, op_shape=op_shape)

    @property
    def data(self):
        # Override to deal with data being either tuple or not
        if self._data[1] is None:
            return self._data[0]
        else:
            return self._data

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        if self._data[1] is not None:
            return False
        check = np.dot(np.transpose(np.conj(self._data[0])), self._data[0])
        return is_identity_matrix(check, rtol=rtol, atol=atol)

    def _evolve(self, state, qargs=None):
        return SuperOp(self)._evolve(state, qargs)

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        ret = copy.copy(self)
        stine_l = np.conjugate(self._data[0])
        stine_r = None
        if self._data[1] is not None:
            stine_r = np.conjugate(self._data[1])
        ret._data = (stine_l, stine_r)
        return ret

    def transpose(self):
        ret = copy.copy(self)
        ret._op_shape = self._op_shape.transpose()
        din, dout = self.dim
        dtr = self._data[0].shape[0] // dout
        stine = [None, None]
        for i, mat in enumerate(self._data):
            if mat is not None:
                stine[i] = np.reshape(
                    np.transpose(np.reshape(mat, (dout, dtr, din)), (2, 1, 0)), (din * dtr, dout)
                )
        ret._data = (stine[0], stine[1])
        return ret

    def compose(
        self, other: Stinespring, qargs: list | None = None, front: bool = False
    ) -> Stinespring:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if qargs is not None:
            return Stinespring(SuperOp(self).compose(other, qargs=qargs, front=front))
        # Otherwise we convert via Kraus representation rather than
        # superoperator to avoid unnecessary representation conversions
        return Stinespring(Kraus(self).compose(other, front=front))

    def tensor(self, other: Stinespring) -> Stinespring:
        if not isinstance(other, Stinespring):
            other = Stinespring(other)
        return self._tensor(self, other)

    def expand(self, other: Stinespring) -> Stinespring:
        if not isinstance(other, Stinespring):
            other = Stinespring(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        # Tensor Stinespring ops
        sa_l, sa_r = a._data
        sb_l, sb_r = b._data

        # Reshuffle tensor dimensions
        din_a, dout_a = a.dim
        din_b, dout_b = b.dim
        dtr_a = sa_l.shape[0] // dout_a
        dtr_b = sb_l.shape[0] // dout_b

        shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
        shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)
        sab_l = np.kron(sa_l, sb_l)
        # Reravel indices
        sab_l = np.reshape(np.transpose(np.reshape(sab_l, shape_in), (0, 2, 1, 3, 4)), shape_out)

        # Compute right Stinespring op
        if sa_r is None and sb_r is None:
            sab_r = None
        else:
            if sa_r is None:
                sa_r = sa_l
            elif sb_r is None:
                sb_r = sb_l
            sab_r = np.kron(sa_r, sb_r)
            # Reravel indices
            sab_r = np.reshape(
                np.transpose(np.reshape(sab_r, shape_in), (0, 2, 1, 3, 4)), shape_out
            )
        ret = copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = (sab_l, sab_r)
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
        # Since we cannot directly add two channels in the Stinespring
        # representation we convert to the Choi representation
        return Stinespring(Choi(self)._add(other, qargs=qargs))

    def _multiply(self, other):
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")

        ret = copy.copy(self)
        # If the number is complex or negative we need to convert to
        # general Stinespring representation so we first convert to
        # the Choi representation
        if isinstance(other, complex) or other < 1:
            # Convert to Choi-matrix
            ret._data = Stinespring(Choi(self)._multiply(other))._data
            return ret
        # If the number is real we can update the Kraus operators
        # directly
        num = math.sqrt(other)
        stine_l, stine_r = self._data
        stine_l = num * self._data[0]
        stine_r = None
        if self._data[1] is not None:
            stine_r = num * self._data[1]
        ret._data = (stine_l, stine_r)
        return ret


# Update docstrings for API docs
generate_apidocs(Stinespring)
