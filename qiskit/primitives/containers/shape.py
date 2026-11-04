# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Array shape related classes and functions
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

ShapeInput = Union[int, "Iterable[ShapeInput]"]
"""An input that is coercible into a shape tuple."""


@runtime_checkable
class Shaped(Protocol):
    """Protocol that defines what it means to be a shaped object.

    Note that static type checkers will classify ``numpy.ndarray`` as being :class:`Shaped`.
    Moreover, since this protocol is runtime-checkable, we will even have
    ``isinstance(<numpy.ndarray instance>, Shaped) == True``.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """The array shape of this object."""
        raise NotImplementedError("A `Shaped` protocol must implement the `shape` property")

    @property
    def ndim(self) -> int:
        """The number of array dimensions of this object."""
        raise NotImplementedError("A `Shaped` protocol must implement the `ndim` property")

    @property
    def size(self) -> int:
        """The total dimension of this object, i.e. the product of the entries of :attr:`~shape`."""
        raise NotImplementedError("A `Shaped` protocol must implement the `size` property")


class ShapedMixin(Shaped):
    """Mixin class to create :class:`~Shaped` types by only providing :attr:`_shape` attribute."""

    _shape: tuple[int, ...]

    def __repr__(self):
        return f"{type(self).__name__}(<{self.shape}>)"

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return int(np.prod(self._shape, dtype=int))


def array_coerce(arr: ArrayLike | Shaped) -> NDArray | Shaped:
    """Coerce the input into an object with a shape attribute.

    Copies are avoided.

    Args:
        arr: The object to coerce.

    Returns:
        Something that is :class:`~Shaped`, and always ``numpy.ndarray`` if the input is not
        already :class:`~Shaped`.
    """
    if isinstance(arr, Shaped):
        return arr
    return np.asarray(arr)


def _flatten_to_ints(arg: ShapeInput) -> Iterable[int]:
    """
    Yield one integer at a time.

    Args:
        arg: Integers or iterables of integers, possibly nested, to be yielded.

    Yields:
        The provided integers in depth-first recursive order.

    Raises:
        ValueError: If an input is not an iterable or an integer.
    """
    for item in arg:
        try:
            if isinstance(item, Iterable):
                yield from _flatten_to_ints(item)
            elif int(item) == item:
                yield int(item)
            else:
                raise ValueError(f"Expected {item} to be iterable or an integer.")
        except (TypeError, RecursionError) as ex:
            raise ValueError(f"Expected {item} to be iterable or an integer.") from ex


def shape_tuple(*shapes: ShapeInput) -> tuple[int, ...]:
    """
    Flatten the input into a single tuple of integers, preserving order.

    Args:
        shapes: Integers or iterables of integers, possibly nested.

    Returns:
        A tuple of integers.

    Raises:
        ValueError: If some member of ``shapes`` is not an integer or iterable.
    """
    return tuple(_flatten_to_ints(shapes))
