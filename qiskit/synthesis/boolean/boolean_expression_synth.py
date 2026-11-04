# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Circuit synthesizers and related classes for boolean expressions"""

import itertools
from math import pi

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import ZGate, XGate


class EsopGenerator:
    """Generates an ESOP (Exclusive-sum-of-products) representation
    for a boolean function given by its truth table"""

    def __init__(self, truth_table):
        self.truth_table = truth_table
        self.cofactor_table = {}
        self.esop = self.generate_esop(assignment=())

    def clause_diff(self, clause1, clause2):
        """The indices of variables where the clauses differ"""
        return [i for i in range(len(clause1)) if clause1[i] != clause2[i]]

    def combine_and_optimize(self, clauses_1, clauses_2):
        """Combining clauses of distance 1 until no more combinations can be performed"""
        something_changed = True
        combined_clauses = clauses_1 + clauses_2
        while something_changed:
            something_changed = False
            for clause_1, clause_2 in itertools.combinations(combined_clauses, 2):
                diff = self.clause_diff(clause_1, clause_2)
                if len(diff) == 1:
                    new_clause = clause_1[: diff[0]] + "-" + clause_1[diff[0] + 1 :]
                    if new_clause not in combined_clauses:
                        combined_clauses.remove(clause_1)
                        combined_clauses.remove(clause_2)
                        combined_clauses.append(new_clause)
                        something_changed = True
                        break
        return combined_clauses

    def generate_esop(self, assignment):
        """Recursively generates an ESOP for a partially determined boolean expression
        'assignment' is a partial assignment to the expression's variables
        """
        # this base case is non-optimized;
        # we may be able to terminate even before having a full assignment
        if len(assignment) == self.truth_table.num_bits:
            if self.truth_table[assignment]:
                return ["-" * self.truth_table.num_bits]  # True; a don't care clause
            else:
                return []  # False
        i = len(assignment)  # current variable
        negative_esop = self.generate_esop(assignment + (False,))
        positive_esop = self.generate_esop(assignment + (True,))
        # Shannon's expansion
        negative_esop_bool_expanded = [
            clause[:i] + "0" + clause[i + 1 :] for clause in negative_esop
        ]
        positive_esop_bool_expanded = [
            clause[:i] + "1" + clause[i + 1 :] for clause in positive_esop
        ]
        self.cofactor_table[assignment] = self.combine_and_optimize(
            negative_esop_bool_expanded, positive_esop_bool_expanded
        )
        return self.cofactor_table[assignment]


def synth_phase_oracle_from_esop(esop, num_qubits):
    """
    Generates a phase oracle for the boolean function f given in ESOP (Exclusive sum of products) form
    esop is of the form ('01-1', '11-0', ...) etc
    where 1 is the variable, 0 is negated variable and - is don't care
    """
    qc = QuantumCircuit(num_qubits)
    for clause in esop:
        clause_data = [(index, bit) for index, bit in enumerate(clause) if bit != "-"]
        if len(clause_data) == 0:
            qc.global_phase += pi
            continue
        qubit_indices, control_data = zip(*clause_data)
        control_state = "".join(control_data)
        if len(control_state) == 1:  # single qubit; either Z or XZX
            if control_state == "0":
                qc.x(qubit_indices[0])
            qc.z(qubit_indices[0])
            if control_state == "0":
                qc.x(qubit_indices[0])
        else:  # use custom controlled-Z gate
            # we use the last qubit as the target, flipping it if the control is 0 for that qubit
            gate = ZGate().control(
                len(qubit_indices) - 1, ctrl_state=control_state[:-1][::-1], annotated=False
            )
            if control_state[-1] == "0":
                qc.x(qubit_indices[-1])
            qc.append(gate, qubit_indices)
            if control_state[-1] == "0":
                qc.x(qubit_indices[-1])
    return qc


def synth_bit_oracle_from_esop(esop, num_qubits):
    """
    Generates a bit-flip oracle for the boolean function f given in ESOP (Exclusive sum of products) form
    esop is of the form ('01-1', '11-0', ...) etc
    where 1 is the variable, 0 is negated variable and - is don't care
    """
    output_index = num_qubits - 1
    qc = QuantumCircuit(num_qubits)
    for clause in esop:
        clause_data = [(index, bit) for index, bit in enumerate(clause) if bit != "-"]
        if len(clause_data) == 0:
            qc.x(output_index)
            continue
        qubit_indices, control_data = zip(*clause_data)
        control_state = "".join(control_data)
        # use custom controlled-X gate
        gate = XGate().control(len(qubit_indices), ctrl_state=control_state[::-1], annotated=False)
        qc.append(gate, qubit_indices + (output_index,))
    return qc
