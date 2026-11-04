# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the evolved operator ansatz."""

import unittest
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator, Pauli, SparseObservable

from qiskit.circuit.library import HamiltonianGate
from qiskit.circuit.library.n_local import (
    EvolvedOperatorAnsatz,
    evolved_operator_ansatz,
    hamiltonian_variational_ansatz,
)
from qiskit.synthesis.evolution import MatrixExponential
from qiskit.utils import optionals
from test import QiskitTestCase


@ddt
class TestEvolvedOperatorAnsatz(QiskitTestCase):
    """Test the evolved operator ansatz."""

    @data(True, False)
    def test_evolved_op_ansatz(self, use_function):
        """Test the default evolution."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]

        if use_function:
            evo = evolved_operator_ansatz(ops, 2)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(ops, 2)

        parameters = evo.parameters

        reference = QuantumCircuit(num_qubits)
        strings = ["z" * num_qubits, "y" * num_qubits, "x" * num_qubits] * 2
        for string, time in zip(strings, parameters):
            reference.compose(evolve(string, time), inplace=True)

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    @data(True, False)
    def test_custom_evolution(self, use_function):
        """Test using another evolution than the default (e.g. matrix evolution)."""
        op = SparsePauliOp(["ZIX"])
        matrix = np.array(op)
        evolution = MatrixExponential()

        if use_function:
            evo = evolved_operator_ansatz(op, evolution=evolution)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(op, evolution=evolution)

        parameters = evo.parameters

        reference = QuantumCircuit(3)
        reference.append(HamiltonianGate(matrix, parameters[0]), [0, 1, 2])

        if not use_function:
            evo = evo.decompose().decompose()

        self.assertEqual(evo, reference)

    def test_changing_operators(self):
        """Test rebuilding after the operators changed."""
        ops = [Pauli("X"), Pauli("Y"), Pauli("Z")]
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(ops)

        evo.operators = [Pauli("X"), Pauli("Y")]
        parameters = evo.parameters

        reference = QuantumCircuit(1)
        reference.rx(2 * parameters[0], 0)
        reference.ry(2 * parameters[1], 0)

        self.assertEqual(evo.decompose(reps=2), reference)

    @data(True, False)
    def test_invalid_reps(self, use_function):
        """Test setting an invalid number of reps."""
        with self.assertRaises(ValueError):
            if use_function:
                _ = evolved_operator_ansatz(Pauli("X"), reps=-1)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = EvolvedOperatorAnsatz(Pauli("X"), reps=-1)

    def test_insert_barriers_circuit(self):
        """Test using insert_barriers."""
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for parameter in evo.parameters:
            ref.rz(2.0 * parameter, 0)
            ref.barrier()

        self.assertEqual(evo.decompose(reps=2), ref)

    def test_insert_barriers(self):
        """Test using insert_barriers."""
        evo = evolved_operator_ansatz(Pauli("Z"), reps=4, insert_barriers=True)
        ref = QuantumCircuit(1)
        for i, parameter in enumerate(evo.parameters):
            ref.rz(2.0 * parameter, 0)
            if i < evo.num_parameters - 1:
                ref.barrier()

        self.assertEqual(evo, ref)

    def test_empty_build_fails(self):
        """Test setting no operators to evolve raises the appropriate error."""
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz()
        with self.assertRaises(ValueError):
            _ = evo.draw()

    @data(True, False)
    def test_empty_operator_list(self, use_function):
        """Test setting an empty list of operators to be equal to an empty circuit."""
        if use_function:
            evo = evolved_operator_ansatz([])
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz([])

        self.assertEqual(evo, QuantumCircuit())

    @data(True, False)
    def test_matrix_operator(self, use_function):
        """Test passing a quantum_info.Operator uses the HamiltonianGate."""
        unitary = Operator([[0, 1], [1, 0]])

        if use_function:
            evo = evolved_operator_ansatz(unitary, reps=3)
        else:
            with self.assertWarns(DeprecationWarning):
                evo = EvolvedOperatorAnsatz(unitary, reps=3).decompose()

        self.assertEqual(evo.count_ops()["hamiltonian"], 3)

    def test_flattened(self):
        """Test flatten option is actually flattened."""
        num_qubits = 3
        ops = [Pauli("Z" * num_qubits), Pauli("Y" * num_qubits), Pauli("X" * num_qubits)]
        with self.assertWarns(DeprecationWarning):
            evo = EvolvedOperatorAnsatz(ops, reps=3, flatten=True)
        self.assertNotIn("hamiltonian", evo.count_ops())
        self.assertNotIn("EvolvedOps", evo.count_ops())
        self.assertNotIn("PauliEvolution", evo.count_ops())

    def test_flattening(self):
        """Test ``flatten`` on the function."""
        operators = [Operator.from_label("X"), SparsePauliOp(["Z"])]

        with self.subTest(flatten=None):
            evo = evolved_operator_ansatz(operators, flatten=None)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=True):
            # check we get a warning when trying to flatten a HamiltonianGate,
            # which has an unbound param and cannot be flattened
            with self.assertWarnsRegex(UserWarning, "Cannot flatten"):
                evo = evolved_operator_ansatz(operators, flatten=True)

            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertNotIn("PauliEvolution", ops)

        with self.subTest(flatten=False):
            evo = evolved_operator_ansatz(operators, flatten=False)
            ops = evo.count_ops()
            self.assertIn("hamiltonian", ops)
            self.assertIn("PauliEvolution", ops)

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_sympify_is_real(self):
        """Test converting the parameters to sympy is real."""
        evo = evolved_operator_ansatz(SparsePauliOp(["Z"], coeffs=[1 + 0j]))
        param = evo.parameters[0]  # get the gamma parameter

        angle = evo.data[0].operation.params[0]
        expected = (2.0 * param).sympify()
        self.assertEqual(expected, angle.sympify())

    def test_all_identities_not_empty_circuit(self):
        """Test that all identities still creates a circuit with correct num_qubits."""
        # Create all identity operators using SparsePauliOp
        op1 = SparsePauliOp(["III"])
        op2 = SparsePauliOp(["III"])

        # With remove_identities=True, all should be removed but circuit should still be created
        ansatz = evolved_operator_ansatz([op1, op2], reps=1, remove_identities=True)

        # Should have 0 parameters (all identities removed)
        self.assertEqual(ansatz.num_parameters, 0)
        # But should still have correct number of qubits (not empty circuit)
        self.assertEqual(ansatz.num_qubits, 3)
        # Circuit should not be completely empty (has qubits registered)
        self.assertGreater(len(ansatz.qubits), 0)

    def test_string_prefix_with_identity_removal(self):
        """Test that string prefix is preserved when identities are removed."""
        op1 = SparsePauliOp(["III"])
        op2 = SparsePauliOp(["XII"])
        op3 = SparsePauliOp(["ZII"])

        # Use string prefix
        ansatz = evolved_operator_ansatz(
            [op1, op2, op3], reps=2, remove_identities=True, parameter_prefix="theta"
        )

        # Should have 2 operators * 2 reps = 4 parameters
        self.assertEqual(ansatz.num_parameters, 4)
        # All parameters should have "theta" prefix
        param_names = [str(p) for p in ansatz.parameters]
        self.assertTrue(all("theta" in name for name in param_names))

    def test_sparse_observable_basic(self):
        """Test that SparseObservable can be used with evolved_operator_ansatz."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=1)
        ansatz = evolved_operator_ansatz(obs, reps=1)

        # Should create a valid circuit
        self.assertIsInstance(ansatz, QuantumCircuit)
        self.assertEqual(ansatz.num_qubits, 1)
        self.assertEqual(ansatz.num_parameters, 1)

    def test_sparse_observable_remove_identities(self):
        """Test that SparseObservable identity operators are removed."""
        obs1 = SparseObservable.identity(2)  # Identity
        obs2 = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=2)

        # With remove_identities=True, identity should be removed
        ansatz = evolved_operator_ansatz([obs1, obs2], reps=1, remove_identities=True)
        self.assertEqual(ansatz.num_parameters, 1)
        self.assertEqual(ansatz.num_qubits, 2)

        # Without remove_identities, both should be included
        ansatz2 = evolved_operator_ansatz([obs1, obs2], reps=1, remove_identities=False)
        self.assertEqual(ansatz2.num_parameters, 2)

    def test_all_identities_sparse_observable_not_empty(self):
        """Test that all SparseObservable identities still creates a circuit with correct num_qubits."""
        obs1 = SparseObservable.identity(3)
        obs2 = SparseObservable.identity(3)

        # With remove_identities=True, all should be removed but circuit should still be created
        ansatz = evolved_operator_ansatz([obs1, obs2], reps=1, remove_identities=True)

        # Should have 0 parameters (all identities removed)
        self.assertEqual(ansatz.num_parameters, 0)
        # But should still have correct number of qubits (not empty circuit)
        self.assertEqual(ansatz.num_qubits, 3)
        # Circuit should not be completely empty (has qubits registered)
        self.assertGreater(len(ansatz.qubits), 0)

    def test_sparse_observable_fast_rust_path(self):
        """Test that SparseObservable uses fast Rust path when conditions are met."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0), ("Z", [1], 1.0)], num_qubits=2)

        # Should use fast path (flatten=True, evolution=None, SparseObservable)
        ansatz = evolved_operator_ansatz(obs, reps=2, flatten=True)

        # Verify fast path was used (no PauliEvolutionGate)
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)
        # Should have correct number of parameters
        self.assertEqual(ansatz.num_parameters, 2)
        # Should have correct number of qubits
        self.assertEqual(ansatz.num_qubits, 2)

    def test_sparse_observable_string_prefix_with_identity_removal(self):
        """Test that string prefix is preserved when SparseObservable identities are removed."""
        obs1 = SparseObservable.identity(2)
        obs2 = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=2)
        obs3 = SparseObservable.from_sparse_list([("Z", [1], 1.0)], num_qubits=2)

        # Use string prefix
        ansatz = evolved_operator_ansatz(
            [obs1, obs2, obs3], reps=2, remove_identities=True, parameter_prefix="theta"
        )

        # Should have 2 operators * 2 reps = 4 parameters
        self.assertEqual(ansatz.num_parameters, 4)
        # All parameters should have "theta" prefix
        param_names = [str(p) for p in ansatz.parameters]
        self.assertTrue(all("theta" in name for name in param_names))

    def test_sparse_observable_mixed_with_sparse_pauli_op(self):
        """Test mixing SparseObservable and SparsePauliOp uses Rust path."""
        obs = SparseObservable.from_sparse_list([("X", [0], 1.0)], num_qubits=1)
        pauli_op = SparsePauliOp(["Z"])

        # Mixed types should use Rust path (both support to_sparse_list)
        ansatz = evolved_operator_ansatz([obs, pauli_op], reps=1, flatten=True)

        # Should still work and use Rust path
        self.assertGreater(ansatz.num_parameters, 0)
        # Should not have PauliEvolutionGate (Rust path)
        ops = ansatz.count_ops()
        self.assertNotIn("PauliEvolution", ops)


class TestHamiltonianVariationalAnsatz(QiskitTestCase):
    """Test the hamiltonian_variational_ansatz function.

    This is essentially already tested by the evolved_operator_ansatz, we just need
    to test the additional commuting functionality.
    """

    def test_detect_commutation(self):
        """Test the operator is split into commuting terms."""
        hamiltonian = SparsePauliOp(["XII", "ZZI", "IXI", "IZZ", "IIX"])
        circuit = hamiltonian_variational_ansatz(hamiltonian)

        # this Hamiltonian should be split into 2 commuting groups, hence we get 2 parameters
        self.assertEqual(2, circuit.num_parameters)

    def test_evolution_with_identity(self):
        """Test a Hamiltonian containing an identity term.

        Regression test of #13644.
        """
        hamiltonian = SparsePauliOp(["III", "IZZ", "IXI"])
        ansatz = hamiltonian_variational_ansatz(hamiltonian, reps=1)
        bound = ansatz.assign_parameters([1, 1])  # we have two non-commuting groups, hence 2 params

        expected = QuantumCircuit(3, global_phase=-1)
        expected.rzz(2, 0, 1)
        expected.rx(2, 1)

        self.assertEqual(expected, bound)


def evolve(pauli_string, time):
    """Get the reference evolution circuit for a single Pauli string."""

    num_qubits = len(pauli_string)
    forward = QuantumCircuit(num_qubits)
    for i, pauli in enumerate(pauli_string):
        if pauli == "x":
            forward.h(i)
        elif pauli == "y":
            forward.sx(i)

    for i in range(1, num_qubits):
        forward.cx(num_qubits - i, num_qubits - i - 1)

    circuit = QuantumCircuit(num_qubits)
    circuit.compose(forward, inplace=True)
    circuit.rz(2 * time, 0)
    circuit.compose(forward.inverse(), inplace=True)

    return circuit
