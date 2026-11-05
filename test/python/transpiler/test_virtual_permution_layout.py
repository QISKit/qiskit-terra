# This code is part of Qiskit.
#
# (C) Copyright IBM 2026
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for setting and updating virtual permutation layout"""

import ddt

import numpy as np
from itertools import combinations, permutations

from qiskit.quantum_info import Operator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, PermutationGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ElidePermutations, StarPreRouting, Split2QUnitaries

from test import QiskitTestCase
from test import combine


@ddt.ddt
class TestVirtualPermutationLayouts(QiskitTestCase):
    """
    Tests that virtual layouts are set and updated correctly.
    """

    @combine(
        swap_list=[
            [(0, 1), (1, 0)],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(0, 3), (5, 6), (2, 3), (1, 5), (2, 1)],
            [(0, 5), (2, 1), (1, 2), (6, 0), (3, 2), (4, 2), (0, 1), (4, 2), (1, 3), (6, 4)],
        ],
        elide_before=[True, False],
        seed=[42, 13, 55],
    )
    def test_split2q_and_elide(self, swap_list, elide_before, seed):
        """Tests that 2q-swap with the elide permutations pass"""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.global_phase += 1.2345
        np.random.seed(seed)
        num_qubits = 7
        qc_split = QuantumCircuit(num_qubits)
        unitary_swap_list = np.random.choice([True, False], size=len(swap_list))
        expected_layout = list(range(num_qubits))
        for swap, unitary_swap in zip(swap_list, unitary_swap_list):
            expected_layout[swap[0]], expected_layout[swap[1]] = (
                expected_layout[swap[1]],
                expected_layout[swap[0]],
            )
            if unitary_swap:
                qc_split.append(UnitaryGate(Operator(qc)), swap)
            else:
                qc_split.swap(*swap)

        pm = PassManager()
        if elide_before:
            pm.append(ElidePermutations())
            pm.append(Split2QUnitaries(split_swap=True))
        else:
            pm.append(Split2QUnitaries(split_swap=True))
            pm.append(ElidePermutations())

        res = pm.run(qc_split)
        self.assertEqual(expected_layout, res.layout.final_index_layout())

        res_op = Operator.from_circuit(res)
        expected_op = Operator(qc_split)
        self.assertEqual(expected_op, res_op)

    def test_all_combinations_of_split2q_swap_and_elide(self):
        """Test all possible combinations of passes that set or update virtual_permutation_layout."""
        ucirc = QuantumCircuit(2)
        ucirc.swap(0, 1)
        ucirc.global_phase += 1.2345
        ugate = UnitaryGate(Operator(ucirc))

        # A quantum circuit for which either of 3 passes applies nontrivially
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.append(PermutationGate([0, 2, 1]), [0, 2, 3])
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.append(ugate, [1, 2])
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        for pass_list in ordered_subsets(
            [ElidePermutations(), StarPreRouting(), Split2QUnitaries(split_swap=True)]
        ):
            with self.subTest(pass_list):
                pm = PassManager(pass_list)
                qct = pm.run(qc)
                self.assertEqual(Operator.from_circuit(qct), Operator.from_circuit(qc))


def ordered_subsets(items: list):
    """Yields all ordered subsets of elements in items."""
    for r in range(len(items) + 1):
        for comb in combinations(items, r):
            for perm in permutations(comb):
                yield list(perm)
