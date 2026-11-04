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
Abstract QuantumState class.
"""

from __future__ import annotations
import copy
from abc import abstractmethod

import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.result.counts import Counts


class QuantumState:
    """Abstract quantum state base class"""

    def __init__(self, op_shape: OpShape | None = None):
        """Initialize a QuantumState object.

        Args:
            op_shape (OpShape):  an OpShape object for state dimensions.

        .. note::

            If ``op_shape`` is specified it will take precedence over other
            kwargs.
        """
        self._op_shape = op_shape
        # RNG for measure functions
        self._rng_generator = None

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.dims() == other.dims()

    @property
    def dim(self):
        """Return total state dimension."""
        return self._op_shape.shape[0]

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit state or None otherwise."""
        return self._op_shape.num_qubits

    @property
    def _rng(self):
        if self._rng_generator is None:
            return np.random.default_rng()
        return self._rng_generator

    def dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        return self._op_shape.dims_l(qargs)

    def copy(self):
        """Make a copy of current operator."""
        return copy.deepcopy(self)

    def seed(self, value=None):
        """Set the seed for the quantum state RNG."""
        if value is None:
            self._rng_generator = None
        elif isinstance(value, np.random.Generator):
            self._rng_generator = value
        else:
            self._rng_generator = np.random.default_rng(value)

    @abstractmethod
    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid quantum state."""

    @abstractmethod
    def to_operator(self):
        """Convert state to matrix operator class"""

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the operator."""

    @abstractmethod
    def trace(self):
        """Return the trace of the quantum state as a density matrix."""

    @abstractmethod
    def purity(self):
        """Return the purity of the quantum state."""

    @abstractmethod
    def tensor(self, other: QuantumState) -> QuantumState:
        """Return the tensor product state self ⊗ other.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """

    @abstractmethod
    def expand(self, other: QuantumState) -> QuantumState:
        """Return the tensor product state other ⊗ self.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (QuantumState): a state object.

        Returns:
            QuantumState: the linear combination self + other.

        Raises:
            NotImplementedError: if subclass does not support addition.
        """
        raise NotImplementedError(f"{type(self)} does not support addition")

    def _multiply(self, other):
        """Return the scalar multiplied state other * self.

        Args:
            other (complex): a complex number.

        Returns:
            QuantumState: the scalar multiplied state other * self.

        Raises:
            NotImplementedError: if subclass does not support scalar
                                 multiplication.
        """
        raise NotImplementedError(f"{type(self)} does not support scalar multiplication")

    @abstractmethod
    def evolve(self, other: Operator | QuantumChannel, qargs: list | None = None) -> QuantumState:
        """Evolve a quantum state by the operator.

        Args:
            other (Operator or QuantumChannel): The operator to evolve by.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified QuantumState subsystem dimensions.
        """

    @abstractmethod
    def expectation_value(self, oper: BaseOperator, qargs: None | list = None) -> complex:
        """Compute the expectation value of an operator.

        Args:
            oper (BaseOperator): an operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value.
        """

    @abstractmethod
    def probabilities(self, qargs: None | list = None, decimals: None | int = None) -> np.ndarray:
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """

    def probabilities_dict(self, qargs: None | list = None, decimals: None | int = None) -> dict:
        """Return the subsystem measurement probability dictionary.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            dict: The measurement probabilities in dict (ket) form.
        """
        return self._vector_to_dict(
            self.probabilities(qargs=qargs, decimals=decimals),
            self.dims(qargs),
            string_labels=True,
        )

    def sample_memory(self, shots: int, qargs: None | list = None) -> np.ndarray:
        """Sample a list of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            np.array: list of sampled counts in the order sampled.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the state's :meth:`seed` method.
        """
        # Get measurement probabilities for measured qubits
        probs = self.probabilities(qargs)

        # Generate list of possible outcome string labels
        labels = self._index_to_ket_array(
            np.arange(len(probs)), self.dims(qargs), string_labels=True
        )
        return self._rng.choice(labels, p=probs, size=shots)

    def sample_counts(self, shots: int, qargs: None | list = None) -> Counts:
        """Sample a dict of qubit measurement outcomes in the computational basis.

        Args:
            shots (int): number of samples to generate.
            qargs (None or list): subsystems to sample measurements for,
                                if None sample measurement of all
                                subsystems (Default: None).

        Returns:
            Counts: sampled counts dictionary.

        Additional Information:

            This function *samples* measurement outcomes using the measure
            :meth:`probabilities` for the current state and `qargs`. It does
            not actually implement the measurement so the current state is
            not modified.

            The seed for random number generator used for sampling can be
            set to a fixed value by using the state's :meth:`seed` method.
        """
        # Sample list of outcomes
        samples = self.sample_memory(shots, qargs=qargs)

        # Combine all samples into a counts dictionary
        inds, counts = np.unique(samples, return_counts=True)
        return Counts(zip(inds, counts))

    def measure(self, qargs: list | None = None) -> tuple:
        """Measure subsystems and return outcome and post-measure state.

        Note that this function uses the QuantumStates internal random
        number generator for sampling the measurement outcome. The RNG
        seed can be set using the :meth:`seed` method.

        Args:
            qargs (list or None): subsystems to sample measurements for,
                                  if None sample measurement of all
                                  subsystems (Default: None).

        Returns:
            tuple: the pair ``(outcome, state)`` where ``outcome`` is the
                   measurement outcome string label, and ``state`` is the
                   collapsed post-measurement state for the corresponding
                   outcome.
        """
        # Sample a single measurement outcome from probabilities
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)

        # Format outcome
        outcome = self._index_to_ket_array(sample, self.dims(qargs), string_labels=True)[0]

        # Convert to projector for state update
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])

        # Update state object
        # TODO: implement a more efficient state update method for
        # diagonal matrix multiplication
        ret = self.evolve(Operator(np.diag(proj), input_dims=dims, output_dims=dims), qargs=qargs)

        return outcome, ret

    @staticmethod
    def _index_to_ket_array(
        inds: np.ndarray, dims: tuple, string_labels: bool = False
    ) -> np.ndarray:
        """Convert an index array into a ket array.

        Args:
            inds (np.array): an integer index array.
            dims (tuple): a list of subsystem dimensions.
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            np.array: an array of ket strings if string_label=True, otherwise
                      an array of ket lists.
        """
        shifts = [1]
        for dim in dims[:-1]:
            shifts.append(shifts[-1] * dim)
        kets = np.array([(inds // shift) % dim for dim, shift in zip(dims, shifts)])

        if string_labels:
            max_dim = max(dims)
            char_kets = np.asarray(kets, dtype=np.str_)
            str_kets = char_kets[0]
            for row in char_kets[1:]:
                if max_dim > 10:
                    str_kets = np.char.add(",", str_kets)
                str_kets = np.char.add(row, str_kets)
            return str_kets.T

        return kets.T

    @staticmethod
    def _vector_to_dict(vec, dims, decimals=None, string_labels=False):
        """Convert a vector to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            vec (array): a Numpy vector array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the vector in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = vec if decimals is None else vec.round(decimals=decimals)
        (inds,) = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        kets = QuantumState._index_to_ket_array(inds, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return dict(zip(kets, vals[inds]))

        return {tuple(ket): val for ket, val in zip(kets, vals[inds])}

    @staticmethod
    def _matrix_to_dict(mat, dims, decimals=None, string_labels=False):
        """Convert a matrix to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            mat (array): a Numpy matrix array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the matrix in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = mat if decimals is None else mat.round(decimals=decimals)
        (
            inds_row,
            inds_col,
        ) = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        bras = QuantumState._index_to_ket_array(inds_row, dims, string_labels=string_labels)
        kets = QuantumState._index_to_ket_array(inds_col, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return {
                f"{ket}|{bra}": val for ket, bra, val in zip(kets, bras, vals[inds_row, inds_col])
            }

        return {
            (tuple(ket), tuple(bra)): val
            for ket, bra, val in zip(kets, bras, vals[inds_row, inds_col])
        }

    @staticmethod
    def _subsystem_probabilities(
        probs: np.ndarray, dims: tuple, qargs: None | list = None
    ) -> np.ndarray:
        """Marginalize a probability vector according to subsystems.

        Args:
            probs (np.array): a probability vector Numpy array.
            dims (tuple): subsystem dimensions.
            qargs (None or list): a list of subsystems to return
                marginalized probabilities for. If None return all
                probabilities (Default: None).

        Returns:
            np.array: the marginalized probability vector flattened
                      for the specified qargs.
        """
        if qargs is None:
            return probs
        # Convert qargs to tensor axes
        probs_tens = np.reshape(probs, list(reversed(dims)))
        ndim = probs_tens.ndim
        qargs_axes = [ndim - 1 - i for i in reversed(qargs)]
        # Get sum axis for marginalized subsystems
        sum_axis = tuple(i for i in range(ndim) if i not in qargs_axes)
        if sum_axis:
            probs_tens = np.sum(probs_tens, axis=sum_axis)
            qargs_axes = np.argsort(np.argsort(qargs_axes))
        # Permute probability vector for desired qargs order
        probs_tens = np.transpose(probs_tens, axes=qargs_axes)
        new_probs = np.reshape(probs_tens, (probs_tens.size,))
        return new_probs

    # Overloads
    def __and__(self, other):
        return self.evolve(other)

    def __xor__(self, other):
        return self.tensor(other)

    def __mul__(self, other):
        return self._multiply(other)

    def __truediv__(self, other):
        return self._multiply(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        return self._add(-other)

    def __neg__(self):
        return self._multiply(-1)
