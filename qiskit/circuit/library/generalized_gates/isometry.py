# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Generic isometries from m to n qubits.
"""

from __future__ import annotations

import math
import numpy as np
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_isometry
from qiskit._accelerate import isometry as isometry_rs

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class Isometry(Instruction):
    r"""Decomposition of arbitrary isometries from :math:`m` to :math:`n` qubits.

    In particular, this allows to decompose unitaries (m=n) and to do state preparation (:math:`m=0`).

    The decomposition is based on [1].

    References:

    [1] Iten et al., Quantum circuits for isometries (2016).
    `Phys. Rev. A 93, 032318
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318>`__.

    """

    # Notation: In the following decomposition we label the qubit by
    # 0 -> most significant one
    # ...
    # n -> least significant one
    # finally, we convert the labels back to the qubit numbering used in Qiskit
    # (using: _get_qubits_by_label)

    def __init__(
        self,
        isometry: np.ndarray,
        num_ancillas_zero: int,
        num_ancillas_dirty: int,
        epsilon: float = _EPS,
    ) -> None:
        r"""
        Args:
            isometry: An isometry from :math:`m` to :math`n` qubits, i.e., a complex
                ``np.ndarray`` of dimension :math:`2^n \times 2^m` with orthonormal columns (given
                in the computational basis specified by the order of the ancillas
                and the input qubits, where the ancillas are considered to be more
                significant than the input qubits).
            num_ancillas_zero: Number of additional ancillas that start in the state :math:`|0\rangle`
                (the :math:`n-m` ancillas required for providing the output of the isometry are
                not accounted for here).
            num_ancillas_dirty: Number of additional ancillas that start in an arbitrary state.
            epsilon: Error tolerance of calculations.
        """
        # Convert to numpy array in case not already an array
        isometry = np.array(isometry, dtype=complex)

        # change a row vector to a column vector (in the case of state preparation)
        if len(isometry.shape) == 1:
            isometry = isometry.reshape(isometry.shape[0], 1)

        self.iso_data = isometry

        self.num_ancillas_zero = num_ancillas_zero
        self.num_ancillas_dirty = num_ancillas_dirty
        self._inverse = None
        self._epsilon = epsilon

        # Check if the isometry has the right dimension and if the columns are orthonormal
        n = math.log2(isometry.shape[0])
        m = math.log2(isometry.shape[1])
        if not n.is_integer() or n < 0:
            raise QiskitError(
                "The number of rows of the isometry is not a non negative power of 2."
            )
        if not m.is_integer() or m < 0:
            raise QiskitError(
                "The number of columns of the isometry is not a non negative power of 2."
            )
        if m > n:
            raise QiskitError(
                "The input matrix has more columns than rows and hence it can't be an isometry."
            )
        if not is_isometry(isometry, self._epsilon):
            raise QiskitError(
                "The input matrix has non orthonormal columns and hence it is not an isometry."
            )

        num_qubits = int(n) + num_ancillas_zero + num_ancillas_dirty

        super().__init__("isometry", num_qubits, 0, [isometry])

    def _define(self):
        # TODO The inverse().inverse() is because there is code to uncompute (_gates_to_uncompute)
        #  an isometry, but not for generating its decomposition. It would be cheaper to do the
        #  later here instead.
        gate = self.inv_gate()
        gate = gate.inverse()
        q = QuantumRegister(self.num_qubits, "q")
        iso_circuit = QuantumCircuit(q)
        iso_circuit.append(gate, q[:])
        self.definition = iso_circuit

    def inverse(self, annotated: bool = False):
        self.params = []
        inv = super().inverse(annotated=annotated)
        self.params = [self.iso_data]
        return inv

    def _gates_to_uncompute(self):
        return isometry_rs.synth_isometry(
            self.iso_data, self.num_ancillas_zero, self.num_ancillas_dirty, self._epsilon
        )

    def validate_parameter(self, parameter):
        """Isometry parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        if isinstance(parameter, (list, int)):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} for gate {self.name}")

    def inv_gate(self):
        """Return the adjoint of the unitary."""
        if self._inverse is None:
            # call to generate the circuit that takes the isometry to the first 2^m columns
            # of the 2^n identity matrix
            iso_circuit = self._gates_to_uncompute()
            self._inverse = iso_circuit.to_instruction()

        return self._inverse
