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

from .diagonal import DiagonalGate
from .uc import UCGate
from .mcg_up_to_diagonal import MCGupDiag

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
        """
        Call to create a circuit with gates that take the desired isometry to the first 2^m columns
         of the 2^n*2^n identity matrix (see https://arxiv.org/abs/1501.06911)
        """
        q = QuantumRegister(self.num_qubits, "q")
        circuit = QuantumCircuit(q, name="isometry_to_uncompute")
        (
            q_input,
            _q_ancillas_for_output,
            _q_ancillas_zero,
            _q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        # Copy the isometry (this is computationally expensive for large isometries but guarantees
        # to keep a copyof the input isometry)
        remaining_isometry = self.iso_data.astype(complex)  # note: "astype" does copy the isometry
        diag = []
        m = int(math.log2(self.iso_data.shape[1]))
        # Decompose the column with index column_index and attach the gate to the circuit object.
        # Return the isometry that is left to decompose, where the columns up to index column_index
        # correspond to the firstfew columns of the identity matrix up to diag, and hence we only
        # have to save a list containing them.
        for column_index in range(2**m):
            remaining_isometry, diag = self._decompose_column(
                circuit, q, diag, remaining_isometry, column_index
            )
            # extract phase of the state that was sent to the basis state ket(column_index)
            diag.append(remaining_isometry[column_index, 0])
            # remove first column (which is now stored in diag)
            remaining_isometry = remaining_isometry[:, 1:]
        if len(diag) > 1 and not isometry_rs.diag_is_identity_up_to_global_phase(
            diag, self._epsilon
        ):
            diagonal = DiagonalGate(np.conj(diag))
            circuit.append(diagonal, q_input)
        return circuit

    def _decompose_column(self, circuit, q, diag, remaining_isometry, column_index):
        """
        Decomposes the column with index column_index.
        """
        n = int(math.log2(self.iso_data.shape[0]))
        for s in range(n):
            remaining_isometry, diag = self._disentangle(
                circuit, q, diag, remaining_isometry, column_index, s
            )
        return remaining_isometry, diag

    def _disentangle(self, circuit, q, diag, remaining_isometry, column_index, s):
        """
        Disentangle the s-th significant qubit (starting with s = 0) into the zero or the one state
        (dependent on column_index)
        """
        # To shorten the notation, we introduce:
        k = column_index
        # k_prime is the index of the column with index column_index in the remaining isometry
        # (note that we remove columns of the isometry during the procedure for efficiency)
        k_prime = 0
        v = remaining_isometry
        n = int(math.log2(self.iso_data.shape[0]))

        # MCG to set one entry to zero (preparation for disentangling with UCGate):
        index1 = 2 * isometry_rs.a(k, s + 1) * 2**s + isometry_rs.b(k, s + 1)
        index2 = (2 * isometry_rs.a(k, s + 1) + 1) * 2**s + isometry_rs.b(k, s + 1)
        target_label = n - s - 1
        # Check if a MCG is required
        if (
            isometry_rs.k_s(k, s) == 0
            and isometry_rs.b(k, s + 1) != 0
            and np.abs(v[index2, k_prime]) > self._epsilon
        ):
            # Find the MCG, decompose it and apply it to the remaining isometry
            gate = isometry_rs.reverse_qubit_state(
                [v[index1, k_prime], v[index2, k_prime]], 0, self._epsilon
            )
            control_labels = [
                i
                for i, x in enumerate(_get_binary_rep_as_list(k, n))
                if x == 1 and i != target_label
            ]
            diagonal_mcg = self._append_mcg_up_to_diagonal(
                circuit, q, gate, control_labels, target_label
            )
            # apply the MCG to the remaining isometry
            v = isometry_rs.apply_multi_controlled_gate(v, control_labels, target_label, gate)
            # correct for the implementation "up to diagonal"
            diag_mcg_inverse = np.conj(diagonal_mcg).astype(complex, copy=False)
            v = isometry_rs.apply_diagonal_gate(
                v, control_labels + [target_label], diag_mcg_inverse
            )
            # update the diag according to the applied diagonal gate
            diag = isometry_rs.apply_diagonal_gate_to_diag(
                diag, control_labels + [target_label], diag_mcg_inverse, n
            )

        # UCGate to disentangle a qubit:
        # Find the UCGate, decompose it and apply it to the remaining isometry
        single_qubit_gates = self._find_squs_for_disentangling(v, k, s)
        if not isometry_rs.ucg_is_identity_up_to_global_phase(single_qubit_gates, self._epsilon):
            control_labels = list(range(target_label))
            diagonal_ucg = self._append_ucg_up_to_diagonal(
                circuit, q, single_qubit_gates, control_labels, target_label
            )
            # merge the diagonal into the UCGate for efficient application of both together
            diagonal_ucg_inverse = np.conj(diagonal_ucg).astype(complex, copy=False)
            single_qubit_gates = isometry_rs.merge_ucgate_and_diag(
                single_qubit_gates, diagonal_ucg_inverse
            )
            # apply the UCGate (with the merged diagonal gate) to the remaining isometry
            v = isometry_rs.apply_ucg(v, len(control_labels), single_qubit_gates)
            # update the diag according to the applied diagonal gate
            diag = isometry_rs.apply_diagonal_gate_to_diag(
                diag, control_labels + [target_label], diagonal_ucg_inverse, n
            )
            # # correct for the implementation "up to diagonal"
            # diag_inv = np.conj(diag).tolist()
            # _apply_diagonal_gate(v, control_labels + [target_label], diag_inv)
        return v, diag

    # This method finds the single-qubit gates for a UCGate to disentangle a qubit:
    # we consider the n-qubit state v[:,0] starting with k zeros (in the computational basis).
    # The qubit with label n-s-1 is disentangled into the basis state k_s(k,s).
    def _find_squs_for_disentangling(self, v, k, s):
        res = isometry_rs.find_squs_for_disentangling(
            v, k, s, self._epsilon, n=int(math.log2(self.iso_data.shape[0]))
        )
        return res

    # Append a UCGate up to diagonal to the circuit circ.
    def _append_ucg_up_to_diagonal(self, circ, q, single_qubit_gates, control_labels, target_label):
        (
            q_input,
            q_ancillas_for_output,
            _q_ancillas_zero,
            _q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        n = int(math.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        # Note that we have to reverse the control labels, since controls are provided by
        # increasing qubit number toa UCGate by convention
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        ucg = UCGate(single_qubit_gates, up_to_diagonal=True)
        circ.append(ucg, [target_qubit] + control_qubits)
        return ucg._get_diagonal()

    # Append a MCG up to diagonal to the circuit circ. The diagonal should only act on the control
    # and target qubits and not on the ancillas. In principle, it would be allowed to act on the
    # dirty ancillas on which we perform the isometry (i.e., on the qubits listed in "qubits"
    # below). But for simplicity, the current code version ignores this future optimization
    # possibility.
    def _append_mcg_up_to_diagonal(self, circ, q, gate, control_labels, target_label):
        (
            q_input,
            q_ancillas_for_output,
            q_ancillas_zero,
            q_ancillas_dirty,
        ) = self._define_qubit_role(q)
        n = int(math.log2(self.iso_data.shape[0]))
        qubits = q_input + q_ancillas_for_output
        control_qubits = _reverse_qubit_oder(_get_qubits_by_label(control_labels, qubits, n))
        target_qubit = _get_qubits_by_label([target_label], qubits, n)[0]
        # The qubits on which we neither act nor control on with the MCG, can be used
        # as dirty ancillas
        ancilla_dirty_labels = [i for i in range(n) if i not in control_labels + [target_label]]
        ancillas_dirty = (
            _reverse_qubit_oder(_get_qubits_by_label(ancilla_dirty_labels, qubits, n))
            + q_ancillas_dirty
        )
        mcg_up_to_diag = MCGupDiag(
            gate, len(control_qubits), len(q_ancillas_zero), len(ancillas_dirty)
        )
        circ.append(
            mcg_up_to_diag, [target_qubit] + control_qubits + q_ancillas_zero + ancillas_dirty
        )
        return mcg_up_to_diag._get_diagonal()

    def _define_qubit_role(self, q):

        n = int(math.log2(self.iso_data.shape[0]))
        m = int(math.log2(self.iso_data.shape[1]))

        # Define the role of the qubits
        q_input = q[:m]
        q_ancillas_for_output = q[m:n]
        q_ancillas_zero = q[n : n + self.num_ancillas_zero]
        q_ancillas_dirty = q[n + self.num_ancillas_zero :]
        return q_input, q_ancillas_for_output, q_ancillas_zero, q_ancillas_dirty

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
            # invert the circuit to create the circuit implementing the isometry
            self._inverse = iso_circuit.to_instruction()
        return self._inverse


# Get the qubits in the list qubits corresponding to the labels listed in labels. The total number
# of qubits is given by num_qubits (and determines the convention for the qubit labeling)

# Remark: We labeled the qubits with decreasing significance. So we have to transform the labels to
# be compatible with the standard convention of Qiskit.


def _get_qubits_by_label(labels, qubits, num_qubits):
    return [qubits[num_qubits - label - 1] for label in labels]


def _reverse_qubit_oder(qubits):
    return list(reversed(qubits))


# Convert list of binary digits to integer


def _get_binary_rep_as_list(n, num_digits):
    binary_string = np.binary_repr(n).zfill(num_digits)
    binary = []
    for line in binary_string:
        for c in line:
            binary.append(int(c))
    return binary[-num_digits:]
