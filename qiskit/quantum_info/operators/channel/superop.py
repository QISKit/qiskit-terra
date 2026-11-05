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
Superoperator representation of a Quantum Channel."""

from __future__ import annotations

import copy as _copy
import math
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor, _to_superop
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator

if TYPE_CHECKING:
    from qiskit import circuit


class SuperOp(QuantumChannel):
    r"""Superoperator representation of a quantum channel.

    The Superoperator representation of a quantum channel :math:`\mathcal{E}`
    is a matrix :math:`S` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        |\mathcal{E}(\rho)\rangle\!\rangle = S |\rho\rangle\!\rangle

    where the double-ket notation :math:`|A\rangle\!\rangle` denotes a vector
    formed by stacking the columns of the matrix :math:`A`
    *(column-vectorization)*.

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
        """Initialize a quantum channel Superoperator operator.

        Args:
            data: data to initialize superoperator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data cannot be initialized as a
                         superoperator.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (4**N, 4**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a superoperator.
        if isinstance(data, (list, np.ndarray)):
            # We initialize directly from superoperator matrix
            super_mat = np.asarray(data, dtype=complex)
            # Determine total input and output dimensions
            dout, din = super_mat.shape
            input_dim = int(math.sqrt(din))
            output_dim = int(math.sqrt(dout))
            if output_dim**2 != dout or input_dim**2 != din:
                raise QiskitError("Invalid shape for SuperOp matrix.")
            op_shape = OpShape.auto(
                dims_l=output_dims, dims_r=input_dims, shape=(output_dim, input_dim)
            )
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # perform a simulation to construct the circuit superoperator.
                # This will only work if the circuit or instruction can be
                # defined in terms of instructions which have no classical
                # register components. The instructions can be gates, reset,
                # or Kraus instructions. Any conditional gates or measure
                # will cause an exception to be raised.
                data = self._init_instruction(data)
            else:
                # We use the QuantumChannel init transform to initialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            # Now that the input is an operator we convert it to a
            # SuperOp object
            op_shape = data._op_shape
            input_dim, output_dim = data.dim
            rep = getattr(data, "_channel_rep", "Operator")
            super_mat = _to_superop(rep, data._data, input_dim, output_dim)
        # Initialize QuantumChannel
        super().__init__(super_mat, op_shape=op_shape)

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    @property
    def _tensor_shape(self):
        """Return the tensor shape of the superoperator matrix"""
        return 2 * tuple(reversed(self._op_shape.dims_l())) + 2 * tuple(
            reversed(self._op_shape.dims_r())
        )

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim, self._input_dim)

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        ret = _copy.copy(self)
        ret._data = np.conj(self._data)
        return ret

    def transpose(self):
        ret = _copy.copy(self)
        ret._data = np.transpose(self._data)
        ret._op_shape = self._op_shape.transpose()
        return ret

    def adjoint(self):
        ret = _copy.copy(self)
        ret._data = np.conj(np.transpose(self._data))
        ret._op_shape = self._op_shape.transpose()
        return ret

    def tensor(self, other: SuperOp) -> SuperOp:
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        return self._tensor(self, other)

    def expand(self, other: SuperOp) -> SuperOp:
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = _copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = _bipartite_tensor(
            a._data, b.data, shape1=a._bipartite_shape, shape2=b._bipartite_shape
        )
        return ret

    def compose(self, other: SuperOp, qargs: list | None = None, front: bool = False) -> SuperOp:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        # Validate dimensions are compatible and return the composed
        # operator dimensions
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()

        # Full composition of superoperators
        if qargs is None:
            if front:
                data = np.dot(self._data, other.data)
            else:
                data = np.dot(other.data, self._data)
            ret = SuperOp(data, input_dims, output_dims)
            ret._op_shape = new_shape
            return ret

        # Compute tensor contraction indices from qargs
        num_qargs_l, num_qargs_r = self._op_shape.num_qargs
        if front:
            num_indices = num_qargs_r
            shift = 2 * num_qargs_l
            right_mul = True
        else:
            num_indices = num_qargs_l
            shift = 0
            right_mul = False

        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._tensor_shape)
        mat = np.reshape(other.data, other._tensor_shape)
        # Add first set of indices
        indices = [2 * num_indices - 1 - qubit for qubit in qargs] + [
            num_indices - 1 - qubit for qubit in qargs
        ]
        final_shape = [np.prod(output_dims) ** 2, np.prod(input_dims) ** 2]
        data = np.reshape(
            Operator._einsum_matmul(tensor, mat, indices, shift, right_mul), final_shape
        )
        ret = SuperOp(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    # ---------------------------------------------------------------------
    # Additional methods
    # ---------------------------------------------------------------------

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
        # Prevent cyclic imports by importing DensityMatrix here

        from qiskit.quantum_info.states.densitymatrix import DensityMatrix

        if not isinstance(state, DensityMatrix):
            state = DensityMatrix(state)

        if qargs is None:
            # Evolution on full matrix
            if state._op_shape.shape[0] != self._op_shape.shape[1]:
                raise QiskitError(
                    "Operator input dimension is not equal to density matrix dimension."
                )
            # We reshape in column-major vectorization (Fortran order in Numpy)
            # since that is how the SuperOp is defined
            vec = np.ravel(state.data, order="F")
            mat = np.reshape(
                np.dot(self.data, vec), (self._output_dim, self._output_dim), order="F"
            )
            return DensityMatrix(mat, dims=self.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if state.dims(qargs) != self.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(state.data, state._op_shape.tensor_shape)
        mat = np.reshape(self.data, self._tensor_shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(state.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs] + [
            2 * num_indices - 1 - qubit for qubit in qargs
        ]
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        # Replace evolved dimensions
        new_dims = list(state.dims())
        output_dims = self.output_dims()
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = output_dims[i]
        new_dim = np.prod(new_dims)
        # reshape tensor to density matrix
        tensor = np.reshape(tensor, (new_dim, new_dim))
        return DensityMatrix(tensor, dims=new_dims)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to a SuperOp."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an identity superoperator of the correct size
        # of the circuit
        op = SuperOp(np.eye(4**instruction.num_qubits))
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_superop(cls, obj):
        """Return superop for instruction if defined or None otherwise."""
        if not isinstance(obj, Instruction):
            raise QiskitError("Input is not an instruction.")
        chan = None
        if obj.name == "reset":
            # For superoperator evolution we can simulate a reset as
            # a non-unitary superoperator matrix
            chan = SuperOp(np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]))
        if obj.name == "kraus":
            kraus = obj.params
            dim = len(kraus[0])
            chan = SuperOp(_to_superop("Kraus", (kraus, None), dim, dim))
        elif hasattr(obj, "to_matrix"):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                kraus = [obj.to_matrix()]
                dim = len(kraus[0])
                chan = SuperOp(_to_superop("Kraus", (kraus, None), dim, dim))
            except QiskitError:
                pass
        return chan

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        from qiskit.circuit.barrier import Barrier

        chan = self._instruction_to_superop(obj)
        if chan is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(chan, qargs=qargs)
            self._data = op.data
        elif isinstance(obj, Barrier):
            return
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError(f"Cannot apply Instruction: {obj.name}")
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError(
                    f"{obj.name} instruction definition is {type(obj.definition)}; "
                    "expected QuantumCircuit"
                )
            qubit_indices = {bit: idx for idx, bit in enumerate(obj.definition.qubits)}
            for instruction in obj.definition.data:
                if instruction.clbits:
                    raise QiskitError(
                        "Cannot apply instruction with classical bits:"
                        f" {instruction.operation.name}"
                    )
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [qubit_indices[tup] for tup in instruction.qubits]
                else:
                    new_qargs = [qargs[qubit_indices[tup]] for tup in instruction.qubits]
                self._append_instruction(instruction.operation, qargs=new_qargs)


# Update docstrings for API docs
generate_apidocs(SuperOp)
