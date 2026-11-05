# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Base classes for an approximate circuit definition."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import SupportsFloat
import numpy as np

from qiskit.circuit import QuantumCircuit


class ApproximateCircuit(QuantumCircuit, ABC):
    """A base class that represents an approximate circuit."""

    def __init__(self, num_qubits: int, name: str | None = None) -> None:
        """
        Args:
            num_qubits: number of qubits this circuit will span.
            name: a name of the circuit.
        """
        super().__init__(num_qubits, name=name)

    @property
    @abstractmethod
    def thetas(self) -> np.ndarray:
        """
        The property is not implemented and raises a ``NotImplementedError`` exception.

        Returns:
            a vector of parameters of this circuit.
        """
        raise NotImplementedError

    @abstractmethod
    def build(self, thetas: np.ndarray) -> None:
        """
        Constructs this circuit out of the parameters(thetas). Parameter values must be set before
            constructing the circuit.

        Args:
            thetas: a vector of parameters to be set in this circuit.
        """
        raise NotImplementedError


class ApproximatingObjective(ABC):
    """
    A base class for an optimization problem definition. An implementing class must provide at least
    an implementation of the ``objective`` method. In such case only gradient free optimizers can
    be used. Both methods, ``objective`` and ``gradient``, are preferable to have in an implementation.
    """

    def __init__(self) -> None:
        # must be set before optimization
        self._target_matrix: np.ndarray | None = None

    @abstractmethod
    def objective(self, param_values: np.ndarray) -> SupportsFloat:
        """
        Computes a value of the objective function given a vector of parameter values.

        Args:
            param_values: a vector of parameter values for the optimization problem.

        Returns:
            a float value of the objective function.
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        """
        Computes a gradient with respect to parameters given a vector of parameter values.

        Args:
            param_values: a vector of parameter values for the optimization problem.

        Returns:
            an array of gradient values.
        """
        raise NotImplementedError

    @property
    def target_matrix(self) -> np.ndarray:
        """
        Returns:
            a matrix being approximated
        """
        return self._target_matrix

    @target_matrix.setter
    def target_matrix(self, target_matrix: np.ndarray) -> None:
        """
        Args:
            target_matrix: a matrix to approximate in the optimization procedure.
        """
        self._target_matrix = target_matrix

    @property
    @abstractmethod
    def num_thetas(self) -> int:
        """

        Returns:
            the number of parameters in this optimization problem.
        """
        raise NotImplementedError
