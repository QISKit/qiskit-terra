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
A definition of the approximate circuit compilation optimization problem based on CNOT unit
definition.
"""
from __future__ import annotations
import typing
from abc import ABC

import numpy as np
from numpy import linalg as la

from .approximate import ApproximatingObjective
from .elementary_operations import ry_matrix, rz_matrix, place_unitary, place_cnot, rx_matrix


class CNOTUnitObjective(ApproximatingObjective, ABC):
    """
    A base class for a problem definition based on CNOT unit. This class may have different
    subclasses for objective and gradient computations.
    """

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        """
        Args:
            num_qubits: number of qubits.
            cnots: a CNOT structure to be used in the optimization procedure.
        """
        super().__init__()
        self._num_qubits = num_qubits
        self._cnots = cnots
        self._num_cnots = cnots.shape[1]

    @property
    def num_cnots(self):
        """
        Returns:
            A number of CNOT units to be used by the approximate circuit.
        """
        return self._num_cnots

    @property
    def num_thetas(self):
        """
        Returns:
            Number of parameters (angles) of rotation gates in this circuit.
        """
        return 3 * self._num_qubits + 4 * self._num_cnots


class DefaultCNOTUnitObjective(CNOTUnitObjective):
    """A naive implementation of the objective function based on CNOT units."""

    def __init__(self, num_qubits: int, cnots: np.ndarray) -> None:
        """
        Args:
            num_qubits: number of qubits.
            cnots: a CNOT structure to be used in the optimization procedure.
        """
        super().__init__(num_qubits, cnots)

        # last objective computations to be re-used by gradient
        self._last_thetas: np.ndarray | None = None
        self._cnot_right_collection: np.ndarray | None = None
        self._cnot_left_collection: np.ndarray | None = None
        self._rotation_matrix: int | np.ndarray | None = None
        self._cnot_matrix: np.ndarray | None = None

    def objective(self, param_values: np.ndarray) -> typing.SupportsFloat:
        # rename parameters just to make shorter and make use of our dictionary
        thetas = param_values
        n = self._num_qubits
        d = int(2**n)
        cnots = self._cnots
        num_cnots = self.num_cnots

        # to save intermediate computations we define the following matrices
        # this is the collection of cnot unit matrices ordered from left to
        # right as in the circuit, not matrix product
        cnot_unit_collection = np.zeros((d, d * num_cnots), dtype=complex)
        # this is the collection of matrix products of the cnot units up
        # to the given position from the right of the circuit
        cnot_right_collection = np.zeros((d, d * num_cnots), dtype=complex)
        # this is the collection of matrix products of the cnot units up
        # to the given position from the left of the circuit
        cnot_left_collection = np.zeros((d, d * num_cnots), dtype=complex)
        # first, we construct each cnot unit matrix
        for cnot_index in range(num_cnots):
            theta_index = 4 * cnot_index

            # cnot qubit indices for the cnot unit identified by cnot_index
            q1 = int(cnots[0, cnot_index])
            q2 = int(cnots[1, cnot_index])

            # rotations that are applied on the q1 qubit
            ry1 = ry_matrix(thetas[0 + theta_index])
            rz1 = rz_matrix(thetas[1 + theta_index])

            # rotations that are applied on the q2 qubit
            ry2 = ry_matrix(thetas[2 + theta_index])
            rx2 = rx_matrix(thetas[3 + theta_index])

            # combine the rotations on qubits q1 and q2
            single_q1 = np.dot(rz1, ry1)
            single_q2 = np.dot(rx2, ry2)

            # we place single qubit matrices at the corresponding locations in the (2^n, 2^n) matrix
            full_q1 = place_unitary(single_q1, n, q1)
            full_q2 = place_unitary(single_q2, n, q2)

            # we place a cnot matrix at the qubits q1 and q2 in the full matrix
            cnot_q1q2 = place_cnot(n, q1, q2)

            # compute the cnot unit matrix and store in cnot_unit_collection
            cnot_unit_collection[:, d * cnot_index : d * (cnot_index + 1)] = la.multi_dot(
                [full_q2, full_q1, cnot_q1q2]
            )

        # this is the matrix corresponding to the intermediate matrix products
        # it will end up being the matrix product of all the cnot unit matrices
        # first we multiply from the right-hand side of the circuit
        cnot_matrix = np.eye(d)
        for cnot_index in range(num_cnots - 1, -1, -1):
            cnot_matrix = np.dot(
                cnot_matrix, cnot_unit_collection[:, d * cnot_index : d * (cnot_index + 1)]
            )
            cnot_right_collection[:, d * cnot_index : d * (cnot_index + 1)] = cnot_matrix
        # now we multiply from the left-hand side of the circuit
        cnot_matrix = np.eye(d)
        for cnot_index in range(num_cnots):
            cnot_matrix = np.dot(
                cnot_unit_collection[:, d * cnot_index : d * (cnot_index + 1)], cnot_matrix
            )
            cnot_left_collection[:, d * cnot_index : d * (cnot_index + 1)] = cnot_matrix

        # this is the matrix corresponding to the initial rotations
        # we start with 1 and kronecker product each qubit's rotations
        rotation_matrix: int | np.ndarray = 1
        for q in range(n):
            theta_index = 4 * num_cnots + 3 * q
            rz0 = rz_matrix(thetas[0 + theta_index])
            ry1 = ry_matrix(thetas[1 + theta_index])
            rz2 = rz_matrix(thetas[2 + theta_index])
            rotation_matrix = np.kron(rotation_matrix, la.multi_dot([rz0, ry1, rz2]))

        # the matrix corresponding to the full circuit is the cnot part and
        # rotation part multiplied together
        circuit_matrix = np.dot(cnot_matrix, rotation_matrix)

        # compute error
        error = 0.5 * (la.norm(circuit_matrix - self._target_matrix, "fro") ** 2)

        # cache computations for gradient
        self._last_thetas = thetas
        self._cnot_left_collection = cnot_left_collection
        self._cnot_right_collection = cnot_right_collection
        self._rotation_matrix = rotation_matrix
        self._cnot_matrix = cnot_matrix

        return error

    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        # just to make shorter
        thetas = param_values
        # if given thetas are the same as used at the previous objective computations, then
        # we re-use computations, otherwise we have to re-compute objective
        if not np.all(np.isclose(thetas, self._last_thetas)):
            self.objective(thetas)

        # the partial derivative of the circuit with respect to an angle
        # is the same circuit with the corresponding pauli gate, multiplied
        # by a global phase of -1j / 2, next to the rotation gate (it commutes)
        pauli_x = np.multiply(-1j / 2, np.asarray([[0, 1], [1, 0]]))
        pauli_y = np.multiply(-1j / 2, np.asarray([[0, -1j], [1j, 0]]))
        pauli_z = np.multiply(-1j / 2, np.asarray([[1, 0], [0, -1]]))

        n = self._num_qubits
        d = int(2**n)
        cnots = self._cnots
        num_cnots = self.num_cnots

        # the partial derivative of the cost function is -Re<V',U>
        # where V' is the partial derivative of the circuit
        # first we compute the partial derivatives in the cnot part
        der = np.zeros(4 * num_cnots + 3 * n)
        for cnot_index in range(num_cnots):
            theta_index = 4 * cnot_index

            # cnot qubit indices for the cnot unit identified by cnot_index
            q1 = int(cnots[0, cnot_index])
            q2 = int(cnots[1, cnot_index])

            # rotations that are applied on the q1 qubit
            ry1 = ry_matrix(thetas[0 + theta_index])
            rz1 = rz_matrix(thetas[1 + theta_index])

            # rotations that are applied on the q2 qubit
            ry2 = ry_matrix(thetas[2 + theta_index])
            rx2 = rx_matrix(thetas[3 + theta_index])

            # combine the rotations on qubits q1 and q2
            # note we have to insert an extra pauli gate to take the derivative
            # of the appropriate rotation gate
            for i in range(4):
                if i == 0:
                    single_q1 = la.multi_dot([rz1, pauli_y, ry1])
                    single_q2 = np.dot(rx2, ry2)
                elif i == 1:
                    single_q1 = la.multi_dot([pauli_z, rz1, ry1])
                    single_q2 = np.dot(rx2, ry2)
                elif i == 2:
                    single_q1 = np.dot(rz1, ry1)
                    single_q2 = la.multi_dot([rx2, pauli_y, ry2])
                else:
                    single_q1 = np.dot(rz1, ry1)
                    single_q2 = la.multi_dot([pauli_x, rx2, ry2])

                # we place single qubit matrices at the corresponding locations in
                # the (2^n, 2^n) matrix
                full_q1 = place_unitary(single_q1, n, q1)
                full_q2 = place_unitary(single_q2, n, q2)

                # we place a cnot matrix at the qubits q1 and q2 in the full matrix
                cnot_q1q2 = place_cnot(n, q1, q2)

                # partial derivative of that particular cnot unit, size of (2^n, 2^n)
                der_cnot_unit = la.multi_dot([full_q2, full_q1, cnot_q1q2])
                # der_cnot_unit is multiplied by the matrix product of cnot units to the left
                # of it (if there are any) and to the right of it (if there are any)
                if cnot_index == 0:
                    der_cnot_matrix = np.dot(
                        self._cnot_right_collection[:, d : 2 * d],
                        der_cnot_unit,
                    )
                elif num_cnots - 1 == cnot_index:
                    der_cnot_matrix = np.dot(
                        der_cnot_unit,
                        self._cnot_left_collection[:, d * (num_cnots - 2) : d * (num_cnots - 1)],
                    )
                else:
                    der_cnot_matrix = la.multi_dot(
                        [
                            self._cnot_right_collection[
                                :, d * (cnot_index + 1) : d * (cnot_index + 2)
                            ],
                            der_cnot_unit,
                            self._cnot_left_collection[:, d * (cnot_index - 1) : d * cnot_index],
                        ]
                    )

                # the matrix corresponding to the full circuit partial derivative
                # is the partial derivative of the cnot part multiplied by the usual
                # rotation part
                der_circuit_matrix = np.dot(der_cnot_matrix, self._rotation_matrix)
                # we compute the partial derivative of the cost function
                der[i + theta_index] = -np.real(
                    np.trace(np.dot(der_circuit_matrix.conj().T, self._target_matrix))
                )

        # now we compute the partial derivatives in the rotation part
        # we start with 1 and kronecker product each qubit's rotations
        for i in range(3 * n):
            der_rotation_matrix: int | np.ndarray = 1
            for q in range(n):
                theta_index = 4 * num_cnots + 3 * q
                rz0 = rz_matrix(thetas[0 + theta_index])
                ry1 = ry_matrix(thetas[1 + theta_index])
                rz2 = rz_matrix(thetas[2 + theta_index])
                # for the appropriate rotation gate that we are taking
                # the partial derivative of, we have to insert the
                # corresponding pauli matrix
                if i - 3 * q == 0:
                    rz0 = np.dot(pauli_z, rz0)
                elif i - 3 * q == 1:
                    ry1 = np.dot(pauli_y, ry1)
                elif i - 3 * q == 2:
                    rz2 = np.dot(pauli_z, rz2)
                der_rotation_matrix = np.kron(der_rotation_matrix, la.multi_dot([rz0, ry1, rz2]))

            # the matrix corresponding to the full circuit partial derivative
            # is the usual cnot part multiplied by the partial derivative of
            # the rotation part
            der_circuit_matrix = np.dot(self._cnot_matrix, der_rotation_matrix)
            # we compute the partial derivative of the cost function
            der[4 * num_cnots + i] = -np.real(
                np.trace(np.dot(der_circuit_matrix.conj().T, self._target_matrix))
            )

        return der
