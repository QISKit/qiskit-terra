# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import random

import numpy as np

from qiskit.quantum_info import (
    random_clifford,
    Clifford,
    random_pauli,
    SparsePauliOp,
)
from qiskit.synthesis import synth_clifford_full
from qiskit.quantum_info.operators.symplectic.random import random_pauli_list
from qiskit.quantum_info import random_cnotdihedral, CNOTDihedral


class RandomCliffordBench:
    params = ["1,3000", "2,2500", "3,2000", "4,1500", "5,1000", "6,700"]
    param_names = ["nqubits,length"]

    def time_random_clifford(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        for _ in range(length):
            random_clifford(nqubits)


class CliffordComposeBench:
    params = ["1,7000", "2,5000", "3,5000", "4,2500", "5,2000"]
    param_names = ["nqubits,length"]

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        self.random_clifford = [random_clifford(nqubits) for _ in range(length)]

    def time_compose(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        clifford = Clifford(np.eye(2 * nqubits))
        for i in range(length):
            clifford.compose(self.random_clifford[i])


class CliffordDecomposeBench:
    params = ["1,1000", "2,500", "3,100", "4,50", "5,10"]
    param_names = ["nqubits,length"]

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        self.random_clifford = [random_clifford(nqubits) for _ in range(length)]

    def time_decompose(self, nqubits_length):
        length = int(nqubits_length.split(",")[1])
        for i in range(length):
            synth_clifford_full(self.random_clifford[i])


class RandomCnotDihedralBench:
    params = ["1,2000", "2,1500", "3,1200", "4,1000", "5,800", "6,700"]
    param_names = ["nqubits,length"]

    def time_random_cnotdihedral(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        for _ in range(length):
            random_cnotdihedral(nqubits)


class CnotDihedralComposeBench:
    params = ["1,1500", "2,400", "3,100", "4,40", "5,10"]
    param_names = ["nqubits,length"]

    def setup(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        self.random_cnotdihedral = [random_cnotdihedral(nqubits) for _ in range(length)]

    def time_compose(self, nqubits_length):
        (nqubits, length) = map(int, nqubits_length.split(","))
        cxdihedral = CNOTDihedral(num_qubits=nqubits)
        for i in range(length):
            cxdihedral.compose(self.random_cnotdihedral[i])


class PauliBench:
    params = [100, 200, 300, 400, 500]
    param_names = ["num_qubits"]

    def setup(self, num_qubits):
        self.p1 = random_pauli(num_qubits, True)
        self.p2 = random_pauli(num_qubits, True)

    def time_compose(self, _):
        self.p1.compose(self.p2)

    def time_evolve(self, _):
        self.p1.evolve(self.p2)  # by another Pauli, so by composition

    def time_commutes(self, _):
        self.p1.commutes(self.p2)

    def time_to_instruction(self, _):
        self.p1.to_instruction()

    def time_to_label(self, _):
        self.p1.to_label()

    def time_evolve_by_clifford(self, num_qubits):
        c1 = random_clifford(num_qubits)

        self.p1.evolve(c1)

    time_evolve_by_clifford.params = [10]


class PauliListBench:
    params = [[100, 200, 300, 400, 500], [500]]
    param_names = ["num_qubits", "length"]

    def setup(self, num_qubits, length):
        self.pl1 = random_pauli_list(num_qubits=num_qubits, size=length, phase=True)
        self.pl2 = random_pauli_list(num_qubits=num_qubits, size=length, phase=True)

    def time_commutes(self, _, __):
        self.pl1.commutes(self.pl2)

    def time_commutes_with_all(self, _, __):
        self.pl1.commutes_with_all(self.pl2)

    def time_argsort(self, _, __):
        self.pl1.argsort()

    def time_compose(self, _, __):
        self.pl1.compose(self.pl2)

    def time_group_qubit_wise_commuting(self, _, __):
        self.pl1.group_qubit_wise_commuting()  # exercise retworkx-based code

    def time_evolve_by_clifford(self, num_qubits, __):
        c1 = random_clifford(num_qubits)
        self.pl1.evolve(c1)

    time_evolve_by_clifford.params = [[20], [100]]


class PauliListQargsBench:
    params = [[100, 200, 300, 400, 500], [500]]
    param_names = ["num_qubits", "length"]

    def setup(self, num_qubits, length):
        half_qubits = int(num_qubits / 2)

        self.pl1 = random_pauli_list(num_qubits=num_qubits, size=length, phase=True)
        self.pl2 = random_pauli_list(num_qubits=half_qubits, size=length, phase=True)
        self.qargs = [random.randint(0, num_qubits - 1) for _ in range(half_qubits)]

    def time_commutes_with_qargs(self, _, __):
        self.pl1.commutes(self.pl2, self.qargs)

    def time_compose_with_qargs(self, _, __):
        self.pl1.compose(self.pl2, self.qargs)


class SparsePauliOpBench:
    params = [[50, 100, 150, 200], [100]]
    param_names = ["num_qubits", "length"]

    def setup(self, num_qubits, length):
        self.p1 = SparsePauliOp(random_pauli_list(num_qubits=num_qubits, size=length, phase=True))
        self.p2 = SparsePauliOp(random_pauli_list(num_qubits=num_qubits, size=length, phase=True))

    def time_compose(self, _, __):
        self.p1.compose(self.p2)

    def time_simplify(self, _, __):
        self.p1.simplify()

    def time_tensor(self, _, __):
        self.p1.tensor(self.p2)

    def time_add(self, _, __):
        _ = self.p1 + self.p2

    time_add.params = [[50, 100, 150, 200], [10000]]

    def time_to_list(self, _, __):
        self.p1.to_list()

    time_to_list.params = [[2, 4, 6, 8, 10], [50]]

    def time_to_operator(self, _, __):
        self.p1.to_operator()

    time_to_operator.params = [[2, 4, 6, 8, 10], [50]]

    def time_to_matrix(self, _, __):
        self.p1.to_matrix()

    time_to_matrix.params = [[2, 4, 6, 8, 10], [50]]
