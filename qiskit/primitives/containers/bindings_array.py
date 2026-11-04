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
Bindings array class
"""
from __future__ import annotations


from collections.abc import Mapping
from collections.abc import Iterable, Mapping as _Mapping
from itertools import chain, islice

import numpy as np
from numpy.typing import ArrayLike

from qiskit.circuit import Parameter, QuantumCircuit

from .shape import ShapedMixin, ShapeInput, shape_tuple

# Public API classes
__all__ = ["BindingsArrayLike", "ParameterLike"]


ParameterLike = Parameter | str
"""A parameter or parameter name."""

BindingsArrayLike = Mapping[ParameterLike | tuple[ParameterLike, ...], ArrayLike]
"""A mapping of numeric bindings for circuit parameters.

This allows array values for single or multi-dimensional sweeps over parameter values.

The mapping keys can be single or tuple of Parameter or parameter name strings.
The values can be a float for a single parameter an N-dimensional array like where
the array's last dimension indexes the circuit parameters, with its leading dimensions
representing the array shape of its parameter bindings.

Examples:
- 0-d array (single binding): ``{"a": 4, ("b", "c"): [5, 6]}``
- Single array (last index for parameters): ``{ParameterVector("a", 100): np.ones((10, 10, 100)))``
- Multiple arrays (last index for parameters, flexible dimensions):
  ``{("c", "a"): np.ones((10, 10, 2)), "b": np.zeros((10, 10))}``
"""


class BindingsArray(ShapedMixin):
    r"""Stores parameter binding value sets for a :class:`qiskit.QuantumCircuit`.

    A single parameter binding set provides numeric values to bind to a circuit with free
    :class:`qiskit.circuit.Parameter`\s. An instance of this class stores an array-valued collection
    of such sets. The simplest example is a 0-d array consisting of a single parameter binding set,
    whereas an n-d array of parameter binding sets represents an n-d sweep over values.

    The storage format is a dictionary of arrays attached to parameters,
    ``{params_0: values_0,...}``. A convention is used where the last dimension of each array
    indexes (a subset of) circuit parameters. For example, if the last dimension of ``values_0`` is
    25, then it represents an array of possible binding values for the 25 distinct parameters
    ``params_0``, where its leading shape is the array :attr:`~.shape` of its binding array. This
    allows flexibility about whether values for different parameters are stored in one big array, or
    across several smaller arrays.

    .. plot::
       :include-source:
       :nofigs:

       import numpy as np
       from qiskit.primitives import BindingsArray

       # 0-d array (i.e. only one binding)
       BindingsArray({"a": 4, ("b", "c"): [5, 6]})

       # single array, last index is parameters
       parameters = tuple(f"a{idx}" for idx in range(100))
       BindingsArray({parameters: np.ones((10, 10, 100))})

       # multiple arrays, where each last index is parameters. notice that it's smart enough to
       # figure out that a missing last dimension corresponds to a single parameter.
       BindingsArray(
           {("c", "a"): np.zeros((10, 10, 2)), "b": np.ones((10, 10))}
       )
    """

    def __init__(
        self,
        data: BindingsArrayLike | None = None,
        shape: ShapeInput | None = None,
    ):
        r"""
        Initialize a :class:`~.BindingsArray`.

        The ``shape`` argument does not need to be provided whenever it can unambiguously
        be inferred from the provided arrays. Ambiguity arises whenever the key of an entry of
        ``data`` contains only one parameter and the corresponding array's shape ends in a one.
        In this case, it can't be decided whether that one is an index over parameters, or whether
        it should be incorporated in :attr:`~shape`.

        Since :class:`~.Parameter` objects are only allowed to represent float values, this
        class casts all given values to float. If an incompatible dtype is given, such as complex
        numbers, a ``TypeError`` will be raised.

        Args:
            data: A mapping from one or more parameters to arrays of values to bind
                them to, where the last axis is over parameters.
            shape: The leading shape of every array in these bindings.

        Raises:
            ValueError: If all inputs are ``None``.
            ValueError: If the shape cannot be automatically inferred from the arrays, or if there
                is some inconsistency in the shape of the given arrays.
            TypeError: If some of the values can't be cast to a float type.
        """
        super().__init__()

        if data is None:
            self._data = {}
        else:
            self._data = {
                (
                    _format_key((key,)) if isinstance(key, (Parameter, str)) else _format_key(key)
                ): np.asarray(val, dtype=float)
                for key, val in data.items()
            }

        self._shape = _infer_shape(self._data) if shape is None else shape_tuple(shape)
        self._num_parameters = None

        self.validate()

    def __getitem__(self, args) -> BindingsArray:
        # because the parameters live on the last axis, we don't need to do anything special to
        # accommodate them because there will always be an implicit slice(None, None, None)
        # on all unspecified trailing dimensions
        # separately, we choose to not disallow args which touch the last dimension, even though it
        # would not be a particularly friendly way to chop parameters
        data = {params: val[args] for params, val in self._data.items()}
        try:
            shape = next(iter(data.values())).shape[:-1]
        except StopIteration:
            shape = ()
        return BindingsArray(data, shape)

    def __repr__(self):
        descriptions = [f"shape={self.shape}", f"num_parameters={self.num_parameters}"]
        if self.num_parameters:
            names = list(islice(map(repr, chain.from_iterable(map(_format_key, self._data))), 5))
            if len(names) < self.num_parameters:
                names.append("...")
            descriptions.append(f"parameters=[{', '.join(names)}]")
        return f"{type(self).__name__}(<{', '.join(descriptions)}>)"

    @property
    def data(self) -> dict[tuple[str, ...], np.ndarray]:
        """The keyword values of this array."""
        return self._data

    @property
    def num_parameters(self) -> int:
        """The total number of parameters."""
        if self._num_parameters is None:
            self._num_parameters = sum(val.shape[-1] for val in self._data.values())
        return self._num_parameters

    def as_array(self, parameters: Iterable[ParameterLike] | None = None) -> np.ndarray:
        """Return the contents of this bindings array as a single NumPy array.

        The parameters are indexed along the last dimension of the returned array.

        Args:
            parameters: Optional parameters that determine the order of the output.

        Returns:
            This bindings array as a single NumPy array.

        Raises:
            ValueError: If ``parameters`` are provided, but do not match those found in ``data``.
        """
        position = 0
        ret = np.empty(shape_tuple(self.shape, self.num_parameters))

        if parameters is None:
            # preserve the order of both the dict and the parameters in the keys
            for arr in self.data.values():
                size = arr.shape[-1]
                ret[..., position : position + size] = arr
                position += size
        else:
            # use the order of the provided parameters
            parameters = list(parameters)
            if len(parameters) != self.num_parameters:
                raise ValueError(
                    f"Expected {self.num_parameters} parameters but {len(parameters)} received."
                )

            # If we make it through the following loop without a KeyError, we will know that the
            # data parameters are a subset of the given parameters. However, the above check
            # ensures there are at least as many of them as parameters. Thus we will know that
            # set(parameters) == set(chain(*data.values())).
            idx_lookup = {_param_name(parameter): idx for idx, parameter in enumerate(parameters)}
            for arr_params, arr in self.data.items():
                try:
                    idxs = [idx_lookup[_param_name(param)] + position for param in arr_params]
                except KeyError as ex:
                    missing = next(p for p in map(_param_name, arr_params) if p not in idx_lookup)
                    raise ValueError(f"Could not find placement for parameter '{missing}'.") from ex
                ret[..., idxs] = arr

        return ret

    def bind(self, circuit: QuantumCircuit, loc: tuple[int, ...]) -> QuantumCircuit:
        """Return a new circuit bound to the values at the provided index.

        Args:
            circuit: The circuit to bind.
            loc: A tuple of indices, one for each dimension of this array.

        Returns:
            The bound circuit.

        Raises:
            ValueError: If the index doesn't have the right number of values.
        """
        if len(loc) != self.ndim:
            raise ValueError(f"Expected {loc} to index all dimensions of {self.shape}.")

        parameters = {
            param: val
            for params, vals in self._data.items()
            for param, val in zip(params, vals[loc])
        }
        return circuit.assign_parameters(parameters)

    def bind_all(self, circuit: QuantumCircuit) -> np.ndarray:
        """Return an object array of bound circuits with the same shape.

        Args:
            circuit: The circuit to bind.

        Returns:
            An object array of the same shape containing all bound circuits.
        """
        arr = np.empty(self.shape, dtype=object)
        for idx in np.ndindex(self.shape):
            arr[idx] = self.bind(circuit, idx)
        return arr

    def ravel(self) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with one dimension.

        The returned bindings array has a :attr:`shape` given by ``(size, )``, where the size is the
        :attr:`~size` of this bindings array.

        Returns:
            A new bindings array.
        """
        return self.reshape(self.size)

    def reshape(self, *shape: int | Iterable[int]) -> BindingsArray:
        """Return a new :class:`~BindingsArray` with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned bindings array.

        Returns:
            A new bindings array.

        Raises:
            ValueError: If the provided shape has a different product than the current size.
        """
        shape = shape_tuple(shape)
        if any(dim < 0 for dim in shape):
            # to reliably catch the ValueError, we need to manually deal with negative values
            positive_size = np.prod([dim for dim in shape if dim >= 0], dtype=int)
            missing_dim = self.size // positive_size
            shape = tuple(dim if dim >= 0 else missing_dim for dim in shape)

        if np.prod(shape, dtype=int) != self.size:
            raise ValueError("Reshaping cannot change the total number of elements.")

        data = {ps: val.reshape(shape + val.shape[-1:]) for ps, val in self._data.items()}
        return BindingsArray(data, shape=shape)

    @classmethod
    def coerce(cls, bindings_array: BindingsArrayLike) -> BindingsArray:
        """Coerce an input that is :class:`~BindingsArrayLike` into a new :class:`~BindingsArray`.

        Args:
            bindings_array: An object to be bindings array.

        Returns:
            A new bindings array.
        """
        if bindings_array is None:
            bindings_array = cls()
        elif isinstance(bindings_array, _Mapping):
            bindings_array = cls(data=bindings_array)
        elif isinstance(bindings_array, BindingsArray):
            return bindings_array
        else:
            raise TypeError(f"Unsupported type {type(bindings_array)} is given.")
        return bindings_array

    def validate(self):
        """Validate the consistency in bindings_array."""
        for parameters, val in self._data.items():
            val = self._data[parameters] = _standardize_shape(val, self._shape)
            if len(parameters) != val.shape[-1]:
                raise ValueError(
                    f"Length of {parameters} inconsistent with last dimension of {val}"
                )


def _standardize_shape(val: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return ``val`` or ``val[..., None]``.

    Args:
        val: The array whose shape to standardize.
        shape: The shape to standardize to.

    Returns:
        An array with one more dimension than ``len(shape)``, and whose leading dimensions match
        ``shape``.

    Raises:
        ValueError: If the leading shape of ``val`` does not match the ``shape``.
    """
    if val.shape == shape:
        val = val[..., None]
    elif val.ndim - 1 != len(shape) or val.shape[:-1] != shape:
        raise ValueError(f"Array with shape {val.shape} inconsistent with {shape}")
    return val


def _infer_shape(data: dict[tuple[Parameter, ...], np.ndarray]) -> tuple[int, ...]:
    """Return a shape tuple that consistently defines the leading dimensions of all arrays.

    Args:
        data: A mapping from tuples to arrays, where the length of each tuple should match the
            last dimension of the corresponding array.

    Returns:
        A shape tuple that matches the leading dimension of every array.

    Raises:
        ValueError: If this cannot be done unambiguously.
    """
    only_possible_shapes = None

    def examine_array(*possible_shapes):
        nonlocal only_possible_shapes
        if only_possible_shapes is None:
            only_possible_shapes = set(possible_shapes)
        else:
            only_possible_shapes.intersection_update(possible_shapes)

    for parameters, val in data.items():
        if len(parameters) != 1:
            # the last dimension _has_ to be over parameters
            examine_array(val.shape[:-1])
        elif val.shape and val.shape[-1] == 1:
            # this case is a convention, and separated from the previous case for clarity:
            # if the last axis is 1-d, make an assumption that it is for our 1 parameter
            examine_array(val.shape[:-1])
        else:
            # otherwise, the user has left off the last axis and we'll be nice to them
            examine_array(val.shape)

    if only_possible_shapes is None:
        return ()
    if len(only_possible_shapes) == 1:
        return next(iter(only_possible_shapes))
    elif len(only_possible_shapes) == 0:
        raise ValueError("Could not find any consistent shape.")
    raise ValueError(
        "Could not unambiguously determine the intended shape, all shapes in "
        f"{only_possible_shapes} are consistent with the input; specify shape manually."
    )


def _format_key(key: tuple[Parameter | str, ...]):
    return tuple(map(_param_name, key))


def _param_name(param: Parameter | str) -> str:
    return param.name if isinstance(param, Parameter) else param
