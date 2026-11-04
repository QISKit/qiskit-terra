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
Object ND-array initialization function.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike


def object_array(
    arr: ArrayLike,
    order: str | None = None,
    copy: bool = True,
    list_types: Sequence[type] | None = (),
) -> np.ndarray:
    """Convert an array-like of objects into an object array.

    .. note::

        If the objects in the array like input define ``__array__`` methods
        this avoids calling them and will instead set the returned array values
        to the Python objects themselves.

    Args:
        arr: An array-like input.
        order:  the order of the returned array (C, F, A, K). If None
               the default NumPy ordering of C is used.
        copy: If True make a copy of the input if it is already an array.
        list_types:  a sequence of types to treat as lists of array
            element objects when inferring the array shape from the input.

    Returns:
        A NumPy ND-array with ``dtype=object``.

    Raises:
        ValueError: If the input cannot be coerced into an object array.
    """
    if isinstance(arr, np.ndarray):
        if arr.dtype != object or order is not None or copy is True:
            arr = arr.astype(object, order=order, copy=copy)
        return arr

    shape = _infer_shape(arr, list_types=tuple(list_types))
    obj_arr = np.empty(shape, dtype=object, order=order)
    if not shape:
        # We call fill here instead of [()] to avoid invoking the
        # objects `__array__` method if it has one (eg for Pauli's).
        obj_arr.fill(arr)
    else:
        # For other arrays we need to do some tricks to avoid invoking the
        # objects __array__ method by flattening the input and initializing
        # using `np.fromiter` which does not invoke `__array__` for object
        # dtypes.
        def _flatten(nested, k):
            if k == 1:
                return nested
            else:
                return [item for sublist in nested for item in _flatten(sublist, k - 1)]

        flattened = _flatten(arr, len(shape))
        if len(flattened) != obj_arr.size:
            raise ValueError(
                "Input object size does not match the inferred array shape."
                " This most likely occurs when the input is a ragged array."
            )
        obj_arr.flat = np.fromiter(flattened, dtype=object, count=len(flattened))

    return obj_arr


def _infer_shape(obj: ArrayLike, list_types: tuple[type, ...] = ()) -> tuple[int, ...]:
    """Infer the shape of an array-like object without casting"""
    if isinstance(obj, np.ndarray):
        return obj.shape
    if not isinstance(obj, (list, *list_types)):
        return ()
    size = len(obj)
    if size == 0:
        return (size,)
    return (size, *_infer_shape(obj[0], list_types=list_types))
