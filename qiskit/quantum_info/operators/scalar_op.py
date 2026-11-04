# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ScalarOp class
"""

from __future__ import annotations
import copy as _copy
from numbers import Number
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.mixins import generate_apidocs


class ScalarOp(LinearOp):
    """Scalar identity operator class.

    This is a symbolic representation of a scalar identity operator on
    multiple subsystems. It may be used to initialize a symbolic scalar
    multiplication of an identity and then be implicitly converted to other
    kinds of operator subclasses by using the :meth:`compose`, :meth:`dot`,
    :meth:`tensor`, :meth:`expand` methods.
    """

    def __init__(self, dims: int | tuple | None = None, coeff: Number = 1):
        """Initialize an operator object.

        Args:
            dims (int or tuple): subsystem dimensions.
            coeff (Number): scalar coefficient for the identity
                            operator (Default: 1).

        Raises:
            QiskitError: If the optional coefficient is invalid.
        """
        if not isinstance(coeff, Number):
            raise QiskitError(f"coeff {coeff} must be a number.")
        self._coeff = coeff
        super().__init__(input_dims=dims, output_dims=dims)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("could not produce matrix without calculation")
        arr = self.to_matrix()
        return arr if dtype is None else arr.astype(dtype, copy=False)

    def __repr__(self):
        return f"ScalarOp({self.input_dims()}, coeff={self.coeff})"

    @property
    def coeff(self):
        """Return the coefficient"""
        return self._coeff

    def conjugate(self):
        ret = self.copy()
        ret._coeff = np.conjugate(self.coeff)
        return ret

    def transpose(self):
        return self.copy()

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return np.isclose(np.abs(self.coeff), 1, atol=atol, rtol=rtol)

    def to_matrix(self):
        """Convert to a Numpy matrix."""
        dim, _ = self.dim
        iden = np.eye(dim, dtype=complex)
        return self.coeff * iden

    def to_operator(self) -> Operator:
        """Convert to an Operator object."""
        return Operator(
            self.to_matrix(), input_dims=self.input_dims(), output_dims=self.output_dims()
        )

    def compose(self, other: ScalarOp, qargs: list | None = None, front: bool = False) -> ScalarOp:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, BaseOperator):
            other = Operator(other)

        new_shape = self._op_shape.compose(other._op_shape, qargs, front)

        # If other is also an ScalarOp we only need to
        # update the coefficient and dimensions
        if isinstance(other, ScalarOp):
            ret = _copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = new_shape
            return ret

        # If we are composing on the full system we return the
        # other operator with reshaped dimensions
        if qargs is None:
            ret = _copy.copy(other)
            ret._op_shape = new_shape
            # Other operator might not support scalar multiplication
            # so we treat the identity as a special case to avoid a
            # possible error
            if self.coeff == 1:
                return ret
            return self.coeff * ret

        # For qargs composition we initialize the scalar operator
        # as an instance of the other BaseOperators subclass. We then
        # perform subsystem qargs composition using the BaseOperator
        # subclasses compose method.
        # Note that this will raise an error if the other operator does
        # not support initialization from a ScalarOp or the ScalarOps
        # `to_operator` method).
        return other.__class__(self).compose(other, qargs=qargs, front=front)

    def power(self, n: float) -> ScalarOp:
        """Return the power of the ScalarOp.

        Args:
            n (float): the exponent for the scalar op.

        Returns:
            ScalarOp: the ``coeff ** n`` ScalarOp.
        """
        ret = self.copy()
        ret._coeff = self.coeff**n
        return ret

    def tensor(self, other: ScalarOp) -> ScalarOp:
        if not isinstance(other, BaseOperator):
            other = Operator(other)

        if isinstance(other, ScalarOp):
            ret = _copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = self._op_shape.tensor(other._op_shape)
            return ret

        return other.expand(self)

    def expand(self, other: ScalarOp) -> ScalarOp:
        if not isinstance(other, BaseOperator):
            other = Operator(other)

        if isinstance(other, ScalarOp):
            ret = _copy.copy(self)
            ret._coeff = self.coeff * other.coeff
            ret._op_shape = self._op_shape.expand(other._op_shape)
            return ret

        return other.tensor(self)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (BaseOperator): an operator object.
            qargs (None or list): optional subsystems to subtract on
                                  (Default: None)

        Returns:
            ScalarOp: if other is an ScalarOp.
            BaseOperator: if other is not an ScalarOp.

        Raises:
            QiskitError: if other has incompatible dimensions.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, BaseOperator):
            other = Operator(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        # Next if we are adding two ScalarOps we return a ScalarOp
        if isinstance(other, ScalarOp):
            return ScalarOp(self.input_dims(), coeff=self.coeff + other.coeff)

        # If qargs are specified we have to pad the other BaseOperator
        # with identities on remaining subsystems. We do this by
        # composing it with an identity ScalarOp.
        other = ScalarOp._pad_with_identity(self, other, qargs)

        # First we check the special case where coeff=0. In this case
        # we simply return the other operator reshaped so that its
        # subsystem dimensions are equal to the current operator for the
        # case where total dimensions agree but subsystem dimensions differ.
        if self.coeff == 0:
            return other.reshape(self.input_dims(), self.output_dims())

        # Finally if we are adding another BaseOperator subclass
        # we use that subclasses `_add` method and reshape the
        # final dimensions.
        return other.reshape(self.input_dims(), self.output_dims())._add(self)

    def _multiply(self, other):
        """Return the ScalarOp other * self.

        Args:
            other (Number): a complex number.

        Returns:
            ScalarOp: the scaled identity operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError(f"other ({other}) is not a number")
        ret = self.copy()
        ret._coeff = other * self.coeff
        return ret

    @staticmethod
    def _pad_with_identity(current, other, qargs=None):
        """Pad another operator with identities.

        Args:
            current (BaseOperator): current operator.
            other (BaseOperator): other operator.
            qargs (None or list): qargs

        Returns:
            BaseOperator: the padded operator.
        """
        if qargs is None:
            return other
        return ScalarOp(current.input_dims()).compose(other, qargs=qargs)


# Update docstrings for API docs
generate_apidocs(ScalarOp)
