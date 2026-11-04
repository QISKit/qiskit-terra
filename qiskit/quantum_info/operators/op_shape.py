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
Multi-partite matrix and vector shape class
"""

from __future__ import annotations
import copy
from functools import reduce
from operator import mul
from math import log2
from numbers import Integral

from qiskit.exceptions import QiskitError


class OpShape:
    """Multipartite matrix and vector shape class."""

    def __init__(self, dims_l=None, dims_r=None, num_qargs_l=None, num_qargs_r=None):
        """Initialize an operator object."""
        # The number of left and right qargs
        self._num_qargs_l = 0  # the number of left (output) subsystems
        self._num_qargs_r = 0  # the number of right (input) subsystems

        # Subsystem dimensions
        # This is a tuple of dimensions for each subsystem
        # If None each subsystem is assumed to be a dim=2 (qubit)
        self._dims_l = None  # Tuple of left (output) dimensions
        self._dims_r = None  # tuple of right (input) dimensions

        # Set attributes
        if num_qargs_r:
            self._num_qargs_r = int(num_qargs_r)
        if dims_r:
            self._dims_r = tuple(dims_r)
            self._num_qargs_r = len(self._dims_r)
        if num_qargs_l:
            self._num_qargs_l = int(num_qargs_l)
        if dims_l:
            self._dims_l = tuple(dims_l)
            self._num_qargs_l = len(self._dims_l)

    @property
    def settings(self):
        """Return the settings of the ``OpShape`` as dictionary."""
        return {
            "dims_l": self._dims_l,
            "dims_r": self._dims_r,
            "num_qargs_l": self._num_qargs_l,
            "num_qargs_r": self._num_qargs_r,
        }

    def __repr__(self):
        if self._dims_l:
            left = f"dims_l={self._dims_l}"
        elif self._num_qargs_l:
            left = f"num_qargs_l={self._num_qargs_l}"
        else:
            left = ""
        if self._dims_r:
            right = f"dims_r={self._dims_r}"
        elif self._num_qargs_r:
            right = f"num_qargs_r={self._num_qargs_r}"
        else:
            right = ""
        if left and right:
            inner = f"{left}, {right}"
        elif left:
            inner = left
        else:
            inner = right
        return f"OpShape({inner})"

    def __eq__(self, other):
        """Check types and subsystem dimensions are equal"""
        if not isinstance(other, OpShape):
            return False
        return (
            self._num_qargs_r == other._num_qargs_r
            and self._num_qargs_l == other._num_qargs_l
            and self._dims_r == other._dims_r
            and self._dims_l == other._dims_l
        )

    def copy(self):
        """Make a deep copy of current operator."""
        return copy.copy(self)

    @property
    def size(self):
        """Return the combined dimensions of the object"""
        return self._dim_l * self._dim_r

    @property
    def num_qubits(self):
        """Return number of qubits if shape is N-qubit.

        If Shape is not N-qubit return None
        """
        if self._dims_l or self._dims_r:
            return None
        if self._num_qargs_l:
            if self._num_qargs_r and self._num_qargs_l != self._num_qargs_r:
                return None
            return self._num_qargs_l
        return self._num_qargs_r

    @property
    def num_qargs(self):
        """Return a tuple of the number of left and right wires"""
        return self._num_qargs_l, self._num_qargs_r

    @property
    def shape(self):
        """Return a tuple of the matrix shape"""
        if self._num_qargs_l == self._num_qargs_r == 0:
            # Scalar shape is op-like
            return (1, 1)
        if not self._num_qargs_r:
            # Vector shape
            return (self._dim_l,)
        # Matrix shape
        return self._dim_l, self._dim_r

    @property
    def tensor_shape(self):
        """Return a tuple of the tensor shape"""
        return tuple(reversed(self.dims_l())) + tuple(reversed(self.dims_r()))

    @property
    def is_square(self):
        """Return True if the left and right dimensions are equal."""
        return self._num_qargs_l == self._num_qargs_r and self._dims_l == self._dims_r

    def dims_r(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        if self._dims_r:
            if qargs:
                return tuple(self._dims_r[i] for i in qargs)
            return self._dims_r
        num = self._num_qargs_r if qargs is None else len(qargs)
        return num * (2,)

    def dims_l(self, qargs=None):
        """Return tuple of output dimension for specified subsystems."""
        if self._dims_l:
            if qargs:
                return tuple(self._dims_l[i] for i in qargs)
            return self._dims_l
        num = self._num_qargs_l if qargs is None else len(qargs)
        return num * (2,)

    @property
    def _dim_r(self):
        """Return the total input dimension."""
        if self._dims_r:
            return reduce(mul, self._dims_r)
        return 2**self._num_qargs_r

    @property
    def _dim_l(self):
        """Return the total output dimension."""
        if self._dims_l:
            return reduce(mul, self._dims_l)
        return 2**self._num_qargs_l

    def validate_shape(self, shape):
        """Raise an exception if shape is not valid for the OpShape"""
        return self._validate(shape, raise_exception=True)

    def _validate(self, shape, raise_exception=False):
        """Validate OpShape against a matrix or vector shape."""

        ndim = len(shape)
        if ndim > 2:
            if raise_exception:
                raise QiskitError(f"Input shape is not 1 or 2-dimensional (shape = {shape})")
            return False

        if self._dims_l:
            if reduce(mul, self._dims_l) != shape[0]:
                if raise_exception:
                    raise QiskitError(
                        "Output dimensions do not match matrix shape "
                        f"({reduce(mul, self._dims_l)} != {shape[0]})"
                    )
                return False
        elif shape[0] != 2**self._num_qargs_l:
            if raise_exception:
                raise QiskitError("Number of left qubits does not match matrix shape")
            return False

        if ndim == 2:
            if self._dims_r:
                if reduce(mul, self._dims_r) != shape[1]:
                    if raise_exception:
                        raise QiskitError(
                            "Input dimensions do not match matrix shape "
                            f"({reduce(mul, self._dims_r)} != {shape[1]})"
                        )
                    return False
            elif shape[1] != 2**self._num_qargs_r:
                if raise_exception:
                    raise QiskitError("Number of right qubits does not match matrix shape")
                return False
        elif self._dims_r or self._num_qargs_r:
            if raise_exception:
                raise QiskitError("Input dimension should be empty for vector shape.")
            return False

        return True

    @classmethod
    def auto(
        cls,
        shape=None,
        dims_l=None,
        dims_r=None,
        dims=None,
        num_qubits_l=None,
        num_qubits_r=None,
        num_qubits=None,
    ):
        """Construct TensorShape with automatic checking of qubit dimensions"""
        if dims and (dims_l or dims_r):
            raise QiskitError("`dims` kwarg cannot be used with `dims_l` or `dims_r`")
        if num_qubits and (num_qubits_l or num_qubits_r):
            raise QiskitError(
                "`num_qubits` kwarg cannot be used with `num_qubits_l` or `num_qubits_r`"
            )

        if num_qubits:
            num_qubits_l = num_qubits
            num_qubits_r = num_qubits
        if dims:
            dims_l = dims
            dims_r = dims

        if num_qubits_r and num_qubits_l:
            matrix_shape = cls(num_qargs_l=num_qubits_r, num_qargs_r=num_qubits_l)
        else:
            ndim = len(shape) if shape else 0
            if dims_r is None and num_qubits_r is None and ndim > 1:
                dims_r = shape[1]

            if dims_l is None and num_qubits_l is None and ndim > 0:
                dims_l = shape[0]

            if num_qubits_r is None:
                if isinstance(dims_r, Integral):
                    if dims_r != 0 and (dims_r & (dims_r - 1) == 0):
                        num_qubits_r = int(log2(dims_r))
                        dims_r = None
                    else:
                        dims_r = (dims_r,)
                elif dims_r is not None:
                    if set(dims_r) == {2}:
                        num_qubits_r = len(dims_r)
                        dims_r = None
                    else:
                        dims_r = tuple(dims_r)

            if num_qubits_l is None:
                if isinstance(dims_l, Integral):
                    if dims_l != 0 and (dims_l & (dims_l - 1) == 0):
                        num_qubits_l = int(log2(dims_l))
                        dims_l = None
                    else:
                        dims_l = (dims_l,)
                elif dims_l is not None:
                    if set(dims_l) == {2}:
                        num_qubits_l = len(dims_l)
                        dims_l = None
                    else:
                        dims_l = tuple(dims_l)

            matrix_shape = cls(
                dims_l=dims_l, dims_r=dims_r, num_qargs_l=num_qubits_l, num_qargs_r=num_qubits_r
            )
        # Validate shape
        if shape:
            matrix_shape.validate_shape(shape)
        return matrix_shape

    def subset(self, qargs=None, qargs_l=None, qargs_r=None):
        """Return the reduced OpShape of the specified qargs"""
        if qargs:
            # Convert qargs to left and right qargs
            if qargs_l or qargs_r:
                raise QiskitError("qargs cannot be specified with qargs_l or qargs_r")
            if self._num_qargs_l:
                qargs_l = qargs
            if self._num_qargs_r:
                qargs_r = qargs

        # Format integer qargs
        if isinstance(qargs_l, Integral):
            qargs_l = (qargs_l,)
        if isinstance(qargs_r, Integral):
            qargs_r = (qargs_r,)

        # Validate qargs
        if qargs_l and max(qargs_l) >= self._num_qargs_l:
            raise QiskitError("Max qargs_l is larger than number of left qargs")

        if qargs_r and max(qargs_r) >= self._num_qargs_r:
            raise QiskitError("Max qargs_r is larger than number of right qargs")

        num_qargs_l = 0
        dims_l = None
        if qargs_l:
            num_qargs_l = len(qargs_l)
            if self._dims_l:
                dims_l = self.dims_l(qargs)

        num_qargs_r = 0
        dims_r = None
        if qargs_r:
            num_qargs_r = len(qargs_r)
            if self._dims_r:
                dims_l = self.dims_r(qargs)

        return OpShape(
            dims_l=dims_l, dims_r=dims_r, num_qargs_l=num_qargs_l, num_qargs_r=num_qargs_r
        )

    def remove(self, qargs=None, qargs_l=None, qargs_r=None):
        """Return a new :class:`OpShape` with the specified qargs removed"""
        if qargs:
            # Convert qargs to left and right qargs
            if qargs_l or qargs_r:
                raise QiskitError("qargs cannot be specified with qargs_l or qargs_r")
            if self._num_qargs_l:
                qargs_l = qargs
            if self._num_qargs_r:
                qargs_r = qargs
        if qargs_l is None and qargs_r is None:
            return self.copy()

        # Format integer qargs
        if isinstance(qargs_l, Integral):
            qargs_l = (qargs_l,)
        if isinstance(qargs_r, Integral):
            qargs_r = (qargs_r,)

        # Validate qargs
        if qargs_l and max(qargs_l) >= self._num_qargs_l:
            raise QiskitError("Max qargs_l is larger than number of left qargs")

        if qargs_r and max(qargs_r) >= self._num_qargs_r:
            raise QiskitError("Max qargs_r is larger than number of right qargs")

        num_qargs_l = 0
        dims_l = None
        if qargs_l:
            num_qargs_l = self._num_qargs_l - len(qargs_l)
            if self._dims_l:
                dims_l = self.dims_l(tuple(i for i in range(self._num_qargs_l) if i not in qargs_l))

        num_qargs_r = 0
        dims_r = None
        if qargs_r:
            num_qargs_r = self._num_qargs_r - len(qargs_r)
            if self._dims_r:
                dims_l = self.dims_r(tuple(i for i in range(self._num_qargs_r) if i not in qargs_r))

        return OpShape(
            dims_l=dims_l, dims_r=dims_r, num_qargs_l=num_qargs_l, num_qargs_r=num_qargs_r
        )

    def reverse(self):
        """Reverse order of left and right qargs"""
        ret = copy.copy(self)
        if self._dims_r:
            ret._dims_r = tuple(reversed(self._dims_r))
        if self._dims_l:
            ret._dims_l = tuple(reversed(self._dims_l))
        return ret

    def transpose(self):
        """Return the transposed OpShape."""
        ret = copy.copy(self)
        ret._dims_l = self._dims_r
        ret._dims_r = self._dims_l
        ret._num_qargs_l = self._num_qargs_r
        ret._num_qargs_r = self._num_qargs_l
        return ret

    def tensor(self, other):
        """Return the tensor product OpShape"""
        return self._tensor(self, other)

    def expand(self, other):
        """Return the expand product OpShape"""
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        """Return the tensor product OpShape"""
        if a._dims_l or b._dims_l:
            dims_l = b.dims_l() + a.dims_l()
            num_qargs_l = None
        else:
            dims_l = None
            num_qargs_l = b._num_qargs_l + a._num_qargs_l
        if a._dims_r or b._dims_r:
            dims_r = b.dims_r() + a.dims_r()
            num_qargs_r = None
        else:
            dims_r = None
            num_qargs_r = b._num_qargs_r + a._num_qargs_r
        return cls(dims_l=dims_l, dims_r=dims_r, num_qargs_l=num_qargs_l, num_qargs_r=num_qargs_r)

    def compose(self, other, qargs=None, front=False):
        """Return composed OpShape."""
        ret = OpShape()
        if qargs is None:
            if front:
                if self._num_qargs_r != other._num_qargs_l or self._dims_r != other._dims_l:
                    raise QiskitError(
                        "Left and right compose dimensions don't match "
                        f"({self.dims_r()} != {other.dims_l()})"
                    )
                ret._dims_l = self._dims_l
                ret._dims_r = other._dims_r
                ret._num_qargs_l = self._num_qargs_l
                ret._num_qargs_r = other._num_qargs_r
            else:
                if self._num_qargs_l != other._num_qargs_r or self._dims_l != other._dims_r:
                    raise QiskitError(
                        "Left and right compose dimensions don't match "
                        f"({self.dims_l()} != {other.dims_r()})"
                    )
                ret._dims_l = other._dims_l
                ret._dims_r = self._dims_r
                ret._num_qargs_l = other._num_qargs_l
                ret._num_qargs_r = self._num_qargs_r
            return ret

        if front:
            ret._dims_l = self._dims_l
            ret._num_qargs_l = self._num_qargs_l
            if len(qargs) != other._num_qargs_l:
                raise QiskitError(
                    f"Number of qargs does not match ({len(qargs)} != {other._num_qargs_l})"
                )
            if self._dims_r or other._dims_r:
                if self.dims_r(qargs) != other.dims_l():
                    raise QiskitError(
                        "Subsystem dimension do not match on specified qargs "
                        f"{self.dims_r(qargs)} != {other.dims_l()}"
                    )
                dims_r = list(self.dims_r())
                for i, dim in zip(qargs, other.dims_r()):
                    dims_r[i] = dim
                ret._dims_r = tuple(dims_r)
                ret._num_qargs_r = len(ret._dims_r)
            else:
                ret._num_qargs_r = self._num_qargs_r
        else:
            ret._dims_r = self._dims_r
            ret._num_qargs_r = self._num_qargs_r
            if len(qargs) != other._num_qargs_r:
                raise QiskitError(
                    f"Number of qargs does not match ({len(qargs)} != {other._num_qargs_r})"
                )
            if self._dims_l or other._dims_l:
                if self.dims_l(qargs) != other.dims_r():
                    raise QiskitError(
                        "Subsystem dimension do not match on specified qargs "
                        f"{self.dims_l(qargs)} != {other.dims_r()}"
                    )
                dims_l = list(self.dims_l())
                for i, dim in zip(qargs, other.dims_l()):
                    dims_l[i] = dim
                ret._dims_l = tuple(dims_l)
                ret._num_qargs_l = len(ret._dims_l)
            else:
                ret._num_qargs_l = self._num_qargs_l
        return ret

    def dot(self, other, qargs=None):
        """Return the dot product operator OpShape"""
        return self.compose(other, qargs, front=True)

    def _validate_add(self, other, qargs=None):
        # Validate shapes can be added
        if qargs:
            if self._num_qargs_l != self._num_qargs_r:
                raise QiskitError(
                    "Cannot add using qargs if number of left and right qargs are not equal."
                )
            if self.dims_l(qargs) != other.dims_l():
                raise QiskitError(
                    "Cannot add shapes with different left "
                    f"dimension on specified qargs {self.dims_l(qargs)} != {other.dims_l()}"
                )
            if self.dims_r(qargs) != other.dims_r():
                raise QiskitError(
                    "Cannot add shapes with different total right "
                    f"dimension on specified qargs{self.dims_r(qargs)} != {other.dims_r()}"
                )
        elif self != other:
            if self._dim_l != other._dim_l:
                raise QiskitError(
                    "Cannot add shapes with different total left "
                    f"dimension {self._dim_l} != {other._dim_l}"
                )
            if self._dim_r != other._dim_r:
                raise QiskitError(
                    "Cannot add shapes with different total right "
                    f"dimension {self._dim_r} != {other._dim_r}"
                )
        return self
