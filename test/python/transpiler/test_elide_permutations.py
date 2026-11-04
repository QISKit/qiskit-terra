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

"""Test ElidePermutations pass"""

import itertools
import unittest

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.generalized_gates import PermutationGate
from qiskit.transpiler.passes.optimization.elide_permutations import ElidePermutations
from qiskit.transpiler.passes.routing import StarPreRouting
from qiskit.circuit.controlflow import IfElseOp
from qiskit.quantum_info import Operator
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase


class TestElidePermutations(QiskitTestCase):
    """Test elide permutations logical optimization pass."""

    def setUp(self):
        super().setUp()
        self.swap_pass = ElidePermutations()

    def test_no_swap(self):
        """Test no swap means no transform."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_swap_in_middle(self):
        """Test swap in middle of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_beginning(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(1)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_end(self):
        """Test swap at the end of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.swap(0, 1)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_multiple_swaps(self):
        """Test quantum circuit with multiple swaps."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.h(1)

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.cx(2, 1)
        expected.h(2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_before_measure(self):
        """Test swap before measure is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.swap(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_if_else_block(self):
        """Test swap elision only happens outside control flow."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        with qc.if_test((0, 0)):
            qc.swap(0, 1)
        qc.cx(0, 1)
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_swap_if_else_block_with_outside_swap(self):
        """Test swap elision only happens outside control flow."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(2, 0)
        body = QuantumCircuit(2)
        body.swap(0, 1)
        if_else_op = IfElseOp((qc.clbits[0], 0), body)

        qc.append(if_else_op, [0, 1])
        qc.cx(0, 1)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.append(IfElseOp((expected.clbits[0], 0), body), [2, 1])
        expected.cx(2, 1)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_in_middle(self):
        """Test permutation in middle of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_partial_permutation_in_middle(self):
        """Test with a permutation gate in the middle of a circuit,
        with the permutation gate defined only on a subset of qubits.
        """
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        qc.append(PermutationGate([1, 2, 0]), [0, 2, 4])
        qc.cx(2, 3)

        # The permutation corresponding to the permutation gate maps
        #   2 -> 0, 4 -> 2, 0 -> 4, and 1 -> 1, 3 -> 3.
        # Instead of the permutation gate, we can relabel the qubits
        #   0 -> 2, 1 -> 1, 2 -> 4, 3 -> 3, 4 -> 2.
        # Hence cx(2, 3) becomes cx(4, 3).
        expected = QuantumCircuit(5)
        expected.cx(0, 1)
        expected.cx(4, 3)

        pass_ = ElidePermutations()
        res = pass_(qc)

        # Make sure that the pass removes the permutation gate and remaps
        # the following gate
        self.assertEqual(res, expected)

        # Make sure that the transpiled circuit *with* the final permutation
        # is equivalent to the original circuit
        res.append(PermutationGate(res.layout.routing_permutation()), [0, 1, 2, 3, 4])
        self.assertEqual(Operator(res), Operator(qc))

    def test_permutation_at_beginning(self):
        """Test permutation in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(2)
        expected.cx(1, 0)
        expected.barrier(0, 1, 2)
        expected.measure(2, 0)
        expected.measure(1, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_at_end(self):
        """Test permutation at end of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_and_permutation(self):
        """Test a combination of swap and permutation gates."""
        qc = QuantumCircuit(3, 3)
        qc.append(PermutationGate([2, 1, 0]), [0, 1, 2])
        qc.swap(0, 2)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_permutation_before_measure(self):
        """Test permutation before measure is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.append(PermutationGate([1, 2, 0]), [0, 1, 2])
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(2, 1)
        expected.measure(0, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_single_qubit_permutation(self):
        """Test permutation defined over a single qubit."""
        qc = QuantumCircuit(3)
        qc.append(PermutationGate([0]), [1])
        qc.h(1)

        expected = QuantumCircuit(3)
        expected.h(1)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)


class TestElidePermutationsInTranspileFlow(QiskitTestCase):
    """
    Test elide permutations in the full transpile pipeline, especially that
    "layout" and "final_layout" attributes are updated correctly
    as to preserve unitary equivalence.
    """

    def test_not_run_after_layout(self):
        """Test ElidePermutations doesn't do anything after layout."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.h(1)

        spm = generate_preset_pass_manager(
            optimization_level=1, initial_layout=list(range(2, -1, -1)), seed_transpiler=42
        )
        spm.layout += ElidePermutations()
        res = spm.run(qc)
        self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))
        self.assertIn("swap", res.count_ops())
        self.assertTrue(res.layout.final_index_layout(), [0, 1, 2])

    def test_unitary_equivalence(self):
        """Test unitary equivalence of the original and transpiled circuits."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.h(1)

        with self.subTest("no coupling map"):
            spm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

        with self.subTest("with coupling map"):
            spm = generate_preset_pass_manager(
                optimization_level=3, seed_transpiler=42, coupling_map=CouplingMap.from_line(3)
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

    def test_unitary_equivalence_permutation_gates(self):
        """Test unitary equivalence of the original and transpiled circuits."""

        for perm in itertools.permutations([0, 1, 2]):
            qc = QuantumCircuit(5)
            qc.h(1)
            qc.swap(1, 2)
            qc.swap(4, 3)
            qc.append(PermutationGate(perm), [1, 2, 3])

            pm = PassManager([ElidePermutations()])
            res = pm.run(qc)
            self.assertEqual(Operator.from_circuit(res), (Operator(qc)))

    def test_unitary_equivalence_routing_and_basis_translation(self):
        """Test on a larger example that includes routing and basis translation."""

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        with self.subTest("no coupling map"):
            spm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

        with self.subTest("with coupling map"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=1234,
                coupling_map=CouplingMap.from_line(5),
                basis_gates=["u", "cz"],
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

        with self.subTest("no coupling map but with initial layout"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=1234,
                initial_layout=[4, 2, 1, 3, 0],
                basis_gates=["u", "cz"],
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

        with self.subTest("coupling map and initial layout"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=1234,
                initial_layout=[4, 2, 1, 3, 0],
                basis_gates=["u", "cz"],
                coupling_map=CouplingMap.from_line(5),
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))

        with self.subTest("larger coupling map"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=42,
                coupling_map=CouplingMap.from_line(8),
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)

            qc_with_ancillas = QuantumCircuit(8)
            qc_with_ancillas.append(qc, [0, 1, 2, 3, 4])
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc_with_ancillas)))

        with self.subTest("larger coupling map and initial layout"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=42,
                initial_layout=[4, 2, 7, 3, 6],
                coupling_map=CouplingMap.from_line(8),
            )
            spm.init += ElidePermutations()
            res = spm.run(qc)

            qc_with_ancillas = QuantumCircuit(8)
            qc_with_ancillas.append(qc, [0, 1, 2, 3, 4])
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc_with_ancillas)))

    def test_unitary_equivalence_virtual_permutation_layout_composition(self):
        """Test on a larger example that includes routing and basis translation."""

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.swap(0, 2)
        qc.cx(0, 1)
        qc.swap(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.append(PermutationGate([0, 2, 1]), [0, 1, 2])
        qc.h(1)

        with self.subTest("with coupling map"):
            spm = generate_preset_pass_manager(
                optimization_level=3,
                seed_transpiler=1234,
                coupling_map=CouplingMap.from_line(5),
                basis_gates=["u", "cz"],
            )
            spm.init += ElidePermutations()
            spm.init += StarPreRouting()
            res = spm.run(qc)
            self.assertTrue(Operator.from_circuit(res).equiv(Operator(qc)))


if __name__ == "__main__":
    unittest.main()
