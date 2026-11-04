# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# The structure of the code is based on Emanuel Malvetti's semester thesis at
# ETH in 2018, which was supervised by Raban Iten and Prof. Renato Renner.


"""Uniformly controlled gates (also called multiplexed gates)."""

from __future__ import annotations

import math

import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit._accelerate.synthesis import uc_gate

_EPS = 1e-10  # global variable used to chop very small numbers to zero


class UCGate(Gate):
    r"""Uniformly controlled gate (also called multiplexed gate).

    These gates can have several control qubits and a single target qubit.
    If the k control qubits are in the state :math:`|i\rangle` (in the computational basis),
    a single-qubit unitary :math:`U_i` is applied to the target qubit.

    This gate is represented by a block-diagonal matrix, where each block is a
    :math:`2\times 2` unitary, that is

    .. math::

        \begin{pmatrix}
            U_0 & 0 & \cdots & 0 \\
            0 & U_1 & \cdots & 0 \\
            \vdots  &     & \ddots & \vdots \\
            0 & 0   &  \cdots & U_{2^{k-1}}
        \end{pmatrix}.

    The decomposition is based on Ref. [1].

    Unnecessary controls and repeated operators can be removed as described in Ref [2].

    References:

    [1] Bergholm et al., Quantum circuits with uniformly controlled one-qubit gates (2005).
    `Phys. Rev. A 71, 052330 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330>`__.

    [2] de Carvalho et al., Quantum multiplexer simplification for state preparation (2024).
    `arXiv:2409.05618 <https://arxiv.org/abs/2409.05618>`__.

    """

    def __init__(
        self, gate_list: list[np.ndarray], up_to_diagonal: bool = False, mux_simp: bool = True
    ):
        r"""
        Args:
            gate_list: List of two qubit unitaries :math:`[U_0, ..., U_{2^{k-1}}]`, where each
                single-qubit unitary :math:`U_i` is given as a :math:`2 \times 2` numpy array.
            up_to_diagonal: Determines if the gate is implemented up to a diagonal.
                or if it is decomposed completely (default: False).
                If the ``UCGate`` :math:`U` is decomposed up to a diagonal :math:`D`, this means
                that the circuit implements a unitary :math:`U'` such that :math:`D U' = U`.
            mux_simp: Determines whether the search for repetitions is conducted (default: True).
                The intention is to perform a possible simplification in the number of controls
                and operators.

        Raises:
            QiskitError: in case of bad input to the constructor
        """
        # check input format
        if not isinstance(gate_list, list):
            raise QiskitError("The single-qubit unitaries are not provided in a list.")
        for gate in gate_list:
            if not gate.shape == (2, 2):
                raise QiskitError("The dimension of a controlled gate is not equal to (2,2).")
        if not gate_list:
            raise QiskitError("The gate list cannot be empty.")

        # Check if number of gates in gate_list is a positive power of two
        num_contr = math.log2(len(gate_list))
        if num_contr < 0 or not num_contr.is_integer():
            raise QiskitError(
                "The number of controlled single-qubit gates is not a non-negative power of 2."
            )

        # Check if the single-qubit gates are unitaries
        for gate in gate_list:
            if not is_unitary_matrix(gate, _EPS):
                raise QiskitError("A controlled gate is not unitary.")

        # Create new gate.
        super().__init__("multiplexer", int(num_contr) + 1, gate_list)
        self.up_to_diagonal = up_to_diagonal
        self.mux_simp = mux_simp

    def inverse(self, annotated: bool = False) -> Gate:
        """Return the inverse.

        This does not re-compute the decomposition for the multiplexer with the inverse of the
        gates but simply inverts the existing decomposition.
        """
        if not annotated:
            inverse_gate = Gate(
                name=self.name + "_dg", num_qubits=self.num_qubits, params=[]
            )  # removing the params because arrays are deprecated

            definition = QuantumCircuit(list(self.definition.qubits))
            for inst in reversed(self._definition):
                definition._append(
                    inst.replace(operation=inst.operation.inverse(annotated=annotated))
                )

            definition.global_phase = -self.definition.global_phase

            inverse_gate.definition = definition
        else:
            inverse_gate = super().inverse(annotated=annotated)
        return inverse_gate

    def _get_diagonal(self):
        # Important: for a control list q_controls = [q[0],...,q_[k-1]] the
        # diagonal gate is provided in the computational basis of the qubits
        # q[k-1],...,q[0],q_target, decreasingly ordered with respect to the
        # significance of the qubit in the computational basis
        _, diag = self._dec_ucg()
        return diag

    def _define(self):
        ucg_circuit, _ = self._dec_ucg()
        self.definition = ucg_circuit

    def _dec_ucg(self):
        """
        This method finds the single qubit gate arising in the decomposition of UCGates given in
        https://arxiv.org/pdf/quant-ph/0410066.pdf.
        """
        single_qubit_gates = [np.asarray(gate, dtype=complex, order="f") for gate in self.params]
        return uc_gate.dec_ucg(
            single_qubit_gates, self.num_qubits, self.up_to_diagonal, self.mux_simp
        )

    def validate_parameter(self, parameter):
        """Uniformly controlled gate parameter has to be an ndarray."""
        if isinstance(parameter, np.ndarray):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")
