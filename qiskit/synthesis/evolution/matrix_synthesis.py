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

"""Exact synthesis of operator evolution via (exponentially expensive) matrix exponentiation."""

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HamiltonianGate

from .evolution_synthesis import EvolutionSynthesis


class MatrixExponential(EvolutionSynthesis):
    r"""Exact operator evolution via matrix exponentiation and unitary synthesis.

    This class synthesizes the exponential of operators by calculating their exponentially-sized
    matrix representation and using exact matrix exponentiation followed by unitary synthesis
    to obtain a circuit. This process is not scalable and serves as comparison or benchmark
    for small systems.
    """

    def synthesize(self, evolution):

        # get operators and time to evolve
        operators = evolution.operator
        time = evolution.time

        if not isinstance(operators, list):
            matrix = operators.to_matrix()
        else:
            matrix = sum(op.to_matrix() for op in operators)

        # construct the evolution circuit
        evolution_circuit = QuantumCircuit(operators[0].num_qubits)
        gate = HamiltonianGate(matrix, time)
        evolution_circuit.append(gate, evolution_circuit.qubits)

        return evolution_circuit
