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
Abstract BaseOperator class.
"""

from __future__ import annotations
import copy
from abc import ABC

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape

from .mixins import GroupMixin


class BaseOperator(GroupMixin, ABC):
    """Abstract operator base class."""

    def __init__(
        self,
        input_dims: tuple | int | None = None,
        output_dims: tuple | int | None = None,
        num_qubits: int | None = None,
        shape: tuple | None = None,
        op_shape: OpShape | None = None,
    ):
        """Initialize a BaseOperator shape

        Args:
            input_dims (tuple or int or None):  input dimensions.
            output_dims (tuple or int or None):  output dimensions.
            num_qubits (int):  the number of qubits of the operator.
            shape (tuple):  matrix shape for automatically determining
                           qubit dimensions.
            op_shape (OpShape):  an OpShape object for operator dimensions.

        .. note::

            If ``op_shape`` is specified it will take precedence over other
            kwargs.
        """
        self._qargs = None
        if op_shape:
            self._op_shape = op_shape
        else:
            self._op_shape = OpShape.auto(
                shape=shape, dims_l=output_dims, dims_r=input_dims, num_qubits=num_qubits
            )

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def __call__(self, *qargs):
        """Return a shallow copy with qargs attribute set"""
        if len(qargs) == 1 and isinstance(qargs[0], (tuple, list)):
            qargs = qargs[0]
        n_qargs = len(qargs)
        if n_qargs not in self._op_shape.num_qargs:
            raise QiskitError(
                "qargs does not match the number of operator qargs "
                f"({n_qargs} not in {self._op_shape.num_qargs})"
            )
        ret = copy.copy(self)
        ret._qargs = tuple(qargs)
        return ret

    def __eq__(self, other):
        return isinstance(other, type(self)) and self._op_shape == other._op_shape

    @property
    def qargs(self):
        """Return the qargs for the operator."""
        return self._qargs

    @property
    def dim(self):
        """Return tuple (input_shape, output_shape)."""
        return self._op_shape._dim_r, self._op_shape._dim_l

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit operator or None otherwise."""
        return self._op_shape.num_qubits

    @property
    def _input_dim(self):
        """Return the total input dimension."""
        return self._op_shape._dim_r

    @property
    def _output_dim(self):
        """Return the total output dimension."""
        return self._op_shape._dim_l

    def reshape(
        self,
        input_dims: None | tuple | int = None,
        output_dims: None | tuple | int = None,
        num_qubits: None | int = None,
    ) -> BaseOperator:
        """Return a shallow copy with reshaped input and output subsystem dimensions.

        Args:
            input_dims (None or tuple): new subsystem input dimensions.
                If None the original input dims will be preserved [Default: None].
            output_dims (None or tuple): new subsystem output dimensions.
                If None the original output dims will be preserved [Default: None].
            num_qubits (None or int): reshape to an N-qubit operator [Default: None].

        Returns:
            BaseOperator: returns self with reshaped input and output dimensions.

        Raises:
            QiskitError: if combined size of all subsystem input dimension or
                         subsystem output dimensions is not constant.
        """
        new_shape = OpShape.auto(
            dims_l=output_dims, dims_r=input_dims, num_qubits=num_qubits, shape=self._op_shape.shape
        )
        ret = copy.copy(self)
        ret._op_shape = new_shape
        return ret

    def input_dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        return self._op_shape.dims_r(qargs)

    def output_dims(self, qargs=None):
        """Return tuple of output dimension for specified subsystems."""
        return self._op_shape.dims_l(qargs)

    def copy(self):
        """Make a deep copy of current operator."""
        return copy.deepcopy(self)
