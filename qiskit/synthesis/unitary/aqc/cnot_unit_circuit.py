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
"""
This is the Parametric Circuit class: anything that you need for a circuit
to be parametrized and used for approximate compiling optimization.
"""
from __future__ import annotations


import numpy as np

from .approximate import ApproximateCircuit


class CNOTUnitCircuit(ApproximateCircuit):
    """A class that represents an approximate circuit based on CNOT unit blocks."""

    def __init__(
        self,
        num_qubits: int,
        cnots: np.ndarray,
        tol: float | None = 0.0,
        name: str | None = None,
    ) -> None:
        """
        Args:
            num_qubits: the number of qubits in this circuit.
            cnots: an array of dimensions ``(2, L)`` indicating where the CNOT units will be placed.
            tol: angle parameter less or equal this (small) value is considered equal zero and
                corresponding gate is not inserted into the output circuit (because it becomes
                identity one in this case).
            name: name of this circuit

        Raises:
            ValueError: if an unsupported parameter is passed.
        """
        super().__init__(num_qubits=num_qubits, name=name)

        if cnots.ndim != 2 or cnots.shape[0] != 2:
            raise ValueError("CNOT structure must be defined as an array of the size (2, N)")

        self._cnots = cnots
        self._num_cnots = cnots.shape[1]
        self._tol = tol

        # Thetas to be optimized by the AQC algorithm
        self._thetas: np.ndarray | None = None

    @property
    def thetas(self) -> np.ndarray:
        """
        Returns a vector of rotation angles used by CNOT units in this circuit.

        Returns:
            Parameters of the rotation gates in this circuit.
        """
        return self._thetas

    def build(self, thetas: np.ndarray) -> None:
        """
        Constructs a Qiskit quantum circuit out of the parameters (angles) of this circuit. If a
            parameter value is less in absolute value than the specified tolerance then the
            corresponding rotation gate will be skipped in the circuit.
        """
        n = self.num_qubits
        self._thetas = thetas
        cnots = self._cnots

        for k in range(n):
            # add initial three rotation gates for each qubit
            p = 4 * self._num_cnots + 3 * k
            k = n - k - 1
            if np.abs(thetas[2 + p]) > self._tol:
                self.rz(thetas[2 + p], k)
            if np.abs(thetas[1 + p]) > self._tol:
                self.ry(thetas[1 + p], k)
            if np.abs(thetas[0 + p]) > self._tol:
                self.rz(thetas[0 + p], k)

        for c in range(self._num_cnots):
            p = 4 * c
            # Extract where the CNOT goes
            q1 = n - 1 - int(cnots[0, c])
            q2 = n - 1 - int(cnots[1, c])
            # Construct a CNOT unit
            self.cx(q1, q2)
            if np.abs(thetas[0 + p]) > self._tol:
                self.ry(thetas[0 + p], q1)
            if np.abs(thetas[1 + p]) > self._tol:
                self.rz(thetas[1 + p], q1)
            if np.abs(thetas[2 + p]) > self._tol:
                self.ry(thetas[2 + p], q2)
            if np.abs(thetas[3 + p]) > self._tol:
                self.rx(thetas[3 + p], q2)
