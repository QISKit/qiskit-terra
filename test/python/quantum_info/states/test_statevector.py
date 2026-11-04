# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for Statevector quantum state class."""

import unittest
import logging
from itertools import permutations
from ddt import ddt, data
import numpy as np
from numpy.testing import assert_allclose

from qiskit import QiskitError
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import transpile
from qiskit.circuit.library import HGate, QFTGate, GlobalPhaseGate, DiagonalGate, CXGate, XGate
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.utils import optionals
from qiskit.quantum_info.random import random_unitary, random_statevector, random_pauli
from qiskit.quantum_info.states import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.visualization.state_visualization import state_to_latex
from test import QiskitTestCase

logger = logging.getLogger(__name__)


@ddt
class TestStatevector(QiskitTestCase):
    """Tests for Statevector class."""

    def test_from_circuit_bell_state(self):
        """Test from_circuit creates Bell state correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sv = Statevector.from_circuit(qc)
        expected_data = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])
        expected = Statevector(expected_data)

        self.assertTrue(sv.equiv(expected))

    def test_from_circuit_transpilation_consistency(self):
        """Statevector.from_circuit gives consistent results under transpilation."""

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.swap(1, 2)
        qc.z(0)

        backend = BasicSimulator()
        transpiled_opt0 = transpile(qc, backend, optimization_level=0)
        transpiled_opt2 = transpile(qc, backend, optimization_level=2)

        sv1 = Statevector.from_circuit(transpiled_opt0)
        sv2 = Statevector.from_circuit(transpiled_opt2)

        self.assertTrue(sv1.equiv(sv2))

        sv2_nolayout = Statevector.from_circuit(transpiled_opt2, ignore_set_layout=True)
        self.assertFalse(sv1.equiv(sv2_nolayout))

    def test_from_circuit_vs_manual_method(self):
        """Test from_circuit matches manual operator evolution."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        sv1 = Statevector.from_circuit(qc)

        initial_state = Statevector.from_label("00")
        op = Operator.from_circuit(qc)
        sv2 = initial_state.evolve(op)

        self.assertTrue(sv1.equiv(sv2))

    def test_from_circuit_ignore_layout(self):
        """Test from_circuit with ignore_set_layout parameter."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Without layout, both should give same result
        sv1 = Statevector.from_circuit(qc, ignore_set_layout=False)
        sv2 = Statevector.from_circuit(qc, ignore_set_layout=True)

        self.assertTrue(sv1.equiv(sv2))

    @classmethod
    def rand_vec(cls, n, normalize=False):
        """Return complex vector or statevector"""
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        logger.debug("rand_vec default_rng seeded with seed=%s", seed)
        rng = np.random.default_rng(seed)

        vec = rng.random(n) + 1j * rng.random(n)
        if normalize:
            vec /= np.sqrt(np.dot(vec, np.conj(vec)))
        return vec

    def test_init_array_qubit(self):
        """Test subsystem initialization from N-qubit array."""
        # Test automatic inference of qubit subsystems
        vec = self.rand_vec(8)
        for dims in [None, 8]:
            state = Statevector(vec, dims=dims)
            assert_allclose(state.data, vec)
            self.assertEqual(state.dim, 8)
            self.assertEqual(state.dims(), (2, 2, 2))
            self.assertEqual(state.num_qubits, 3)

    def test_init_array(self):
        """Test initialization from array."""
        vec = self.rand_vec(3)
        state = Statevector(vec)
        assert_allclose(state.data, vec)
        self.assertEqual(state.dim, 3)
        self.assertEqual(state.dims(), (3,))
        self.assertIsNone(state.num_qubits)

        vec = self.rand_vec(2 * 3 * 4)
        state = Statevector(vec, dims=[2, 3, 4])
        assert_allclose(state.data, vec)
        self.assertEqual(state.dim, 2 * 3 * 4)
        self.assertEqual(state.dims(), (2, 3, 4))
        self.assertIsNone(state.num_qubits)

    def test_init_circuit(self):
        """Test initialization from circuit."""
        circuit = QuantumCircuit(3)
        circuit.x(0)
        state = Statevector(circuit)

        self.assertEqual(state.dim, 8)
        self.assertEqual(state.dims(), (2, 2, 2))
        self.assertTrue(all(state.data == np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=complex)))
        self.assertEqual(state.num_qubits, 3)

    def test_init_array_except(self):
        """Test initialization exception from array."""
        vec = self.rand_vec(4)
        self.assertRaises(QiskitError, Statevector, vec, dims=[4, 2])
        self.assertRaises(QiskitError, Statevector, vec, dims=[2, 4])
        self.assertRaises(QiskitError, Statevector, vec, dims=5)

    def test_init_statevector(self):
        """Test initialization from Statevector."""
        vec1 = Statevector(self.rand_vec(4))
        vec2 = Statevector(vec1)
        self.assertEqual(vec1, vec2)

    def test_from_circuit(self):
        """Test initialization from a circuit."""
        # random unitaries
        u0 = random_unitary(2).data
        u1 = random_unitary(2).data
        # add to circuit
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.unitary(u0, [qr[0]])
        circ.unitary(u1, [qr[1]])
        target = Statevector(np.kron(u1, u0).dot([1, 0, 0, 0]))
        vec = Statevector.from_instruction(circ)
        self.assertEqual(vec, target)

        # Test tensor product of 1-qubit gates
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(1)
        circuit.ry(np.pi / 2, 2)
        target = Statevector.from_label("000").evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of Controlled-Phase gate
        lam = np.pi / 4
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(1)
        circuit.cp(lam, 0, 1)
        target = Statevector.from_label("00").evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test decomposition of controlled-H gate
        circuit = QuantumCircuit(2)
        circ.x(0)
        circuit.ch(0, 1)
        target = Statevector.from_label("00").evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test custom controlled gate
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(1)
        gate = qc.to_gate()
        gate_ctrl = gate.control(annotated=False)

        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.append(gate_ctrl, range(3))
        target = Statevector.from_label("000").evolve(Operator(circuit))
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test initialize instruction
        target = Statevector([1, 0, 0, 1j]) / np.sqrt(2)
        circuit = QuantumCircuit(2)
        circuit.initialize(target.data, [0, 1])
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        target = Statevector([1, 0, 1, 0]) / np.sqrt(2)
        circuit = QuantumCircuit(2)
        circuit.initialize("+", [1])
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        target = Statevector([1, 0, 0, 0])
        circuit = QuantumCircuit(2)
        circuit.initialize(0, [0, 1])  # initialize from int
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test reset instruction
        target = Statevector([1, 0])
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.reset(0)
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

        # Test 0q instruction
        target = Statevector([1j, 0])
        circuit = QuantumCircuit(1)
        circuit.append(GlobalPhaseGate(np.pi / 2), [], [])
        psi = Statevector.from_instruction(circuit)
        self.assertEqual(psi, target)

    def test_from_instruction(self):
        """Test initialization from an instruction."""
        target = np.dot(HGate().to_matrix(), [1, 0])
        vec = Statevector.from_instruction(HGate()).data
        global_phase_equivalent = matrix_equal(vec, target, ignore_phase=True)
        self.assertTrue(global_phase_equivalent)

    def test_from_label(self):
        """Test initialization from a label"""
        x_p = Statevector(np.array([1, 1]) / np.sqrt(2))
        x_m = Statevector(np.array([1, -1]) / np.sqrt(2))
        y_p = Statevector(np.array([1, 1j]) / np.sqrt(2))
        y_m = Statevector(np.array([1, -1j]) / np.sqrt(2))
        z_p = Statevector(np.array([1, 0]))
        z_m = Statevector(np.array([0, 1]))

        label = "01"
        target = z_p.tensor(z_m)
        self.assertEqual(target, Statevector.from_label(label))

        label = "+-"
        target = x_p.tensor(x_m)
        self.assertEqual(target, Statevector.from_label(label))

        label = "rl"
        target = y_p.tensor(y_m)
        self.assertEqual(target, Statevector.from_label(label))

    def test_equal(self):
        """Test __eq__ method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            self.assertEqual(Statevector(vec), Statevector(vec.tolist()))

    def test_getitem(self):
        """Test __getitem__ method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = Statevector(vec)
            for i in range(4):
                self.assertEqual(state[i], vec[i])
                self.assertEqual(state[format(i, "b")], vec[i])

    def test_getitem_except(self):
        """Test __getitem__ method raises exceptions."""
        for i in range(1, 4):
            state = Statevector(self.rand_vec(2**i))
            self.assertRaises(QiskitError, state.__getitem__, 2**i)
            self.assertRaises(QiskitError, state.__getitem__, -1)

    def test_copy(self):
        """Test Statevector copy method"""
        for _ in range(5):
            vec = self.rand_vec(4)
            orig = Statevector(vec)
            cpy = orig.copy()
            cpy._data[0] += 1.0
            self.assertFalse(cpy == orig)

    def test_is_valid(self):
        """Test is_valid method."""
        state = Statevector([1, 1])
        self.assertFalse(state.is_valid())
        for _ in range(10):
            state = Statevector(self.rand_vec(4, normalize=True))
            self.assertTrue(state.is_valid())

    def test_to_operator(self):
        """Test to_operator method for returning projector."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = Operator(np.outer(vec, np.conj(vec)))
            op = Statevector(vec).to_operator()
            self.assertEqual(op, target)

    def test_evolve(self):
        """Test _evolve method."""
        for _ in range(10):
            op = random_unitary(4)
            vec = self.rand_vec(4)
            target = Statevector(np.dot(op.data, vec))
            evolved = Statevector(vec).evolve(op)
            self.assertEqual(target, evolved)

    def test_evolve_no_modification_with_xgate(self):
        """Test evolve by XGate doesn't modify original vector"""

        # |00> is sensitive to IX
        original_vec = Statevector([1.0, 0.0, 0.0, 0.0])

        # Copy so we can compare to original_vec
        new_vec = original_vec.copy()
        evolved_vec = new_vec.evolve(XGate(), [0])

        # Check that the evolved state is different to the original state. If they aren't, then
        # either our initial state is an eigenstate of the operator or evolve() with the given
        # operator is broken.
        self.assertNotEqual(
            evolved_vec,
            original_vec,
            "The evolved state is unchanged. Either the state is an eigenvector or evolve() is broken.",
        )
        self.assertEqual(original_vec, new_vec, "Evolving by XGate modified original Statevector")

    def test_evolve_no_modification_with_cxgate(self):
        """Test evolve by CXGate doesn't modify original vector"""

        # |01> is sensitive to CX
        original_vec = Statevector([0.0, 1.0, 0.0, 0.0])

        # Copy so we can compare to original_vec
        new_vec = original_vec.copy()
        evolved_vec = new_vec.evolve(CXGate(), [0, 1])

        # Check that the evolved state is different to the original state. If they aren't, then
        # either our initial state is an eigenstate of the operator or evolve() with the given
        # operator is broken.
        self.assertNotEqual(
            evolved_vec,
            original_vec,
            "The evolved state is unchanged. Either the state is an eigenvector or evolve() is broken.",
        )
        self.assertEqual(original_vec, new_vec, "Evolving by CXGate modified original Statevector")

    def test_evolve_no_modification_with_diagonalgate(self):
        """Test evolve by DiagonalGate doesn't modify original vector"""

        # All superposition is sensitive to the given DiagonalGate
        original_vec = Statevector(np.ones(4) / np.sqrt(4))

        # Copy so we can compare to original_vec
        new_vec = original_vec.copy()
        evolved_vec = new_vec.evolve(DiagonalGate([1.0, -1.0, -1.0, 1.0]), [0, 1])

        # Check that the evolved state is different to the original state. If they aren't, then
        # either our initial state is an eigenstate of the operator or evolve() with the given
        # operator is broken.
        self.assertNotEqual(
            evolved_vec,
            original_vec,
            "The evolved state is unchanged. Either the state is an eigenvector or evolve() is broken.",
        )
        self.assertEqual(
            original_vec, new_vec, "Evolving by DiagonalGate modified original Statevector"
        )

    def test_evolve_no_modification_with_operator(self):
        """Test evolve by Operator doesn't modify original vector"""

        # |0> is sensitive to X
        original_vec = Statevector([1.0, 0.0, 0.0, 0.0])
        # Copy so we can compare to original_vec
        new_vec = original_vec.copy()
        evolved_vec = new_vec.evolve(Operator(XGate()), [0])

        # Check that the evolved state is different to the original state. If they aren't, then
        # either our initial state is an eigenstate of the operator or evolve() with the given
        # operator is broken.
        self.assertNotEqual(
            evolved_vec,
            original_vec,
            "The evolved state is unchanged. Either the state is an eigenvector or evolve() is broken.",
        )
        self.assertEqual(
            original_vec, new_vec, "Evolving by Operator modified original Statevector"
        )

    def test_evolve_no_modification_with_quantumcircuit(self):
        """Test by QuantumCircuit method doesn't modify original vector"""

        # |00> is sensitive to a Bell-state preparation circuit
        original_vec = Statevector([1.0, 0.0, 0.0, 0.0])

        # Copy so we can compare to original_vec
        new_vec = original_vec.copy()
        circ = QuantumCircuit(2)
        circ.h(0)
        circ.cx(0, 1)
        evolved_vec = new_vec.evolve(circ, [0, 1])

        # Check that the evolved state is different to the original state. If they aren't, then
        # either our initial state is an eigenstate of the operator or evolve() with the given
        # operator is broken.

        self.assertNotEqual(
            evolved_vec,
            original_vec,
            "The evolved state is unchanged. Either the state is an eigenvector or evolve() is broken.",
        )
        self.assertEqual(
            original_vec, new_vec, "Evolving by QuantumCircuit modified original Statevector"
        )

    def test_evolve_operator_overload_dimensions(self):
        """Test that the @ operator returns a Statevector of correct dimension, type and value."""
        op = random_unitary(4)  # 4x4 unitary
        vec = Statevector(self.rand_vec(4))  # 4-dim state vector
        result = op @ vec
        target = op.data @ vec.data
        self.assertIsInstance(result, Statevector)
        self.assertEqual(result.data.shape, (4,))
        self.assertTrue(np.array_equal(result.data, target))

    def test_evolve_subsystem(self):
        """Test subsystem _evolve method."""
        # Test evolving single-qubit of 3-qubit system
        for _ in range(5):
            vec = self.rand_vec(8)
            state = Statevector(vec)
            op0 = random_unitary(2)
            op1 = random_unitary(2)
            op2 = random_unitary(2)

            # Test evolve on 1-qubit
            op = op0
            op_full = Operator(np.eye(4)).tensor(op)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0]), target)

            # Evolve on qubit 1
            op_full = Operator(np.eye(2)).tensor(op).tensor(np.eye(2))
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[1]), target)

            # Evolve on qubit 2
            op_full = op.tensor(np.eye(4))
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2]), target)

            # Test evolve on 2-qubits
            op = op1.tensor(op0)

            # Evolve on qubits [0, 2]
            op_full = op1.tensor(np.eye(2)).tensor(op0)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 2]), target)

            # Evolve on qubits [2, 0]
            op_full = op0.tensor(np.eye(2)).tensor(op1)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 0]), target)

            # Test evolve on 3-qubits
            op = op2.tensor(op1).tensor(op0)

            # Evolve on qubits [0, 1, 2]
            op_full = op
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[0, 1, 2]), target)

            # Evolve on qubits [2, 1, 0]
            op_full = op0.tensor(op1).tensor(op2)
            target = Statevector(np.dot(op_full.data, vec))
            self.assertEqual(state.evolve(op, qargs=[2, 1, 0]), target)

    def test_evolve_qudit_subsystems(self):
        """Test nested evolve calls on qudit subsystems."""
        dims = (3, 4, 5)
        init = self.rand_vec(np.prod(dims))
        ops = [random_unitary((dim,)) for dim in dims]
        state = Statevector(init, dims)
        for i, op in enumerate(ops):
            state = state.evolve(op, [i])
        target_op = np.eye(1)
        for op in ops:
            target_op = np.kron(op.data, target_op)
        target = Statevector(np.dot(target_op, init), dims)
        self.assertEqual(state, target)

    def test_evolve_global_phase(self):
        """Test evolve circuit with global phase."""
        state_i = Statevector([1, 0])
        qr = QuantumRegister(2)
        phase = np.pi / 4
        circ = QuantumCircuit(qr, global_phase=phase)
        circ.x(0)
        state_f = state_i.evolve(circ, qargs=[0])
        target = Statevector([0, 1]) * np.exp(1j * phase)
        self.assertEqual(state_f, target)

    def test_conjugate(self):
        """Test conjugate method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            target = Statevector(np.conj(vec))
            state = Statevector(vec).conjugate()
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(3)
            target = np.kron(vec1, vec0)
            state = Statevector(vec0).expand(Statevector(vec1))
            self.assertEqual(state.dim, 6)
            self.assertEqual(state.dims(), (2, 3))
            assert_allclose(state.data, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(10):
            vec0 = self.rand_vec(2)
            vec1 = self.rand_vec(3)
            target = np.kron(vec0, vec1)
            state = Statevector(vec0).tensor(Statevector(vec1))
            self.assertEqual(state.dim, 6)
            self.assertEqual(state.dims(), (3, 2))
            assert_allclose(state.data, target)

    def test_inner(self):
        """Test inner method."""
        for _ in range(10):
            vec0 = Statevector(self.rand_vec(4))
            vec1 = Statevector(self.rand_vec(4))
            target = np.vdot(vec0.data, vec1.data)
            result = vec0.inner(vec1)
            self.assertAlmostEqual(result, target)
            vec0 = Statevector(self.rand_vec(6), dims=(2, 3))
            vec1 = Statevector(self.rand_vec(6), dims=(2, 3))
            target = np.vdot(vec0.data, vec1.data)
            result = vec0.inner(vec1)
            self.assertAlmostEqual(result, target)

    def test_inner_except(self):
        """Test inner method raises exceptions."""
        vec0 = Statevector(self.rand_vec(4))
        vec1 = Statevector(self.rand_vec(3))
        self.assertRaises(QiskitError, vec0.inner, vec1)
        vec0 = Statevector(self.rand_vec(6), dims=(2, 3))
        vec1 = Statevector(self.rand_vec(6), dims=(3, 2))
        self.assertRaises(QiskitError, vec0.inner, vec1)

    def test_add(self):
        """Test add method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = Statevector(vec0)
            state1 = Statevector(vec1)
            self.assertEqual(state0 + state1, Statevector(vec0 + vec1))

    def test_add_except(self):
        """Test add method raises exceptions."""
        state1 = Statevector(self.rand_vec(2))
        state2 = Statevector(self.rand_vec(3))
        self.assertRaises(QiskitError, state1.__add__, state2)

    def test_subtract(self):
        """Test subtract method."""
        for _ in range(10):
            vec0 = self.rand_vec(4)
            vec1 = self.rand_vec(4)
            state0 = Statevector(vec0)
            state1 = Statevector(vec1)
            self.assertEqual(state0 - state1, Statevector(vec0 - vec1))

    def test_multiply(self):
        """Test multiply method."""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = Statevector(vec)
            val = np.random.rand() + 1j * np.random.rand()
            self.assertEqual(val * state, Statevector(val * state))

    def test_negate(self):
        """Test negate method"""
        for _ in range(10):
            vec = self.rand_vec(4)
            state = Statevector(vec)
            self.assertEqual(-state, Statevector(-1 * vec))

    def test_equiv(self):
        """Test equiv method"""
        vec = np.array([1, 0, 0, -1j]) / np.sqrt(2)
        phase = np.exp(-1j * np.pi / 4)
        statevec = Statevector(vec)
        self.assertTrue(statevec.equiv(phase * vec))
        self.assertTrue(statevec.equiv(Statevector(phase * vec)))
        self.assertFalse(statevec.equiv(2 * vec))

    def test_equiv_on_circuit(self):
        """Test the equiv method on different types of input."""
        statevec = Statevector([1, 0])

        qc = QuantumCircuit(1)
        self.assertTrue(statevec.equiv(qc))
        qc.x(0)
        self.assertFalse(statevec.equiv(qc))

    def test_to_dict(self):
        """Test to_dict method"""

        with self.subTest(msg="dims = (2, 3)"):
            vec = Statevector(np.arange(1, 7), dims=(2, 3))
            target = {"00": 1, "01": 2, "10": 3, "11": 4, "20": 5, "21": 6}
            self.assertDictAlmostEqual(target, vec.to_dict())

        with self.subTest(msg="dims = (11, )"):
            vec = Statevector(np.arange(1, 12), dims=(11,))
            target = {str(i): i + 1 for i in range(11)}
            self.assertDictAlmostEqual(target, vec.to_dict())

        with self.subTest(msg="dims = (2, 11)"):
            vec = Statevector(np.arange(1, 23), dims=(2, 11))
            target = {}
            for i in range(11):
                for j in range(2):
                    key = f"{i},{j}"
                    target[key] = 2 * i + j + 1
            self.assertDictAlmostEqual(target, vec.to_dict())

    def test_to_dict_decimals(self):
        """Test to_dict method with decimals argument."""
        decimal = 3
        sv = np.array([1 / np.sqrt(2), 0, 0, -1 / np.sqrt(2)], dtype=np.complex128)
        vec_rounded = Statevector(sv).to_dict(decimals=decimal)
        expected = {"00": 0.707, "11": -0.707}
        self.assertEqual(vec_rounded, expected)

    def test_probabilities_product(self):
        """Test probabilities method for product state"""

        state = Statevector.from_label("+0")

        # 2-qubit qargs
        with self.subTest(msg="P(None)"):
            probs = state.probabilities()
            target = np.array([0.5, 0, 0.5, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([0, 1])"):
            probs = state.probabilities([0, 1])
            target = np.array([0.5, 0, 0.5, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([1, 0]"):
            probs = state.probabilities([1, 0])
            target = np.array([0.5, 0.5, 0, 0])
            self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        with self.subTest(msg="P([0])"):
            probs = state.probabilities([0])
            target = np.array([1, 0])
            self.assertTrue(np.allclose(probs, target))

        with self.subTest(msg="P([1])"):
            probs = state.probabilities([1])
            target = np.array([0.5, 0.5])
            self.assertTrue(np.allclose(probs, target))

    def test_probabilities_ghz(self):
        """Test probabilities method for GHZ state"""

        state = (Statevector.from_label("000") + Statevector.from_label("111")) / np.sqrt(2)

        # 3-qubit qargs
        target = np.array([0.5, 0, 0, 0, 0, 0, 0, 0.5])
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 2-qubit qargs
        target = np.array([0.5, 0, 0, 0.5])
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        target = np.array([0.5, 0.5])
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

    def test_probabilities_w(self):
        """Test probabilities method with W state"""

        state = (
            Statevector.from_label("001")
            + Statevector.from_label("010")
            + Statevector.from_label("100")
        ) / np.sqrt(3)

        # 3-qubit qargs
        target = np.array([0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0])
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 2-qubit qargs
        target = np.array([1 / 3, 1 / 3, 1 / 3, 0])
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

        # 1-qubit qargs
        target = np.array([2 / 3, 1 / 3])
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities(qargs)
                self.assertTrue(np.allclose(probs, target))

    def test_probabilities_dict_product(self):
        """Test probabilities_dict method for product state"""

        state = Statevector.from_label("+0")

        # 2-qubit qargs
        with self.subTest(msg="P(None)"):
            probs = state.probabilities_dict()
            target = {"00": 0.5, "10": 0.5}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([0, 1])"):
            probs = state.probabilities_dict([0, 1])
            target = {"00": 0.5, "10": 0.5}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([1, 0]"):
            probs = state.probabilities_dict([1, 0])
            target = {"00": 0.5, "01": 0.5}
            self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        with self.subTest(msg="P([0])"):
            probs = state.probabilities_dict([0])
            target = {"0": 1}
            self.assertDictAlmostEqual(probs, target)

        with self.subTest(msg="P([1])"):
            probs = state.probabilities_dict([1])
            target = {"0": 0.5, "1": 0.5}
            self.assertDictAlmostEqual(probs, target)

    def test_probabilities_dict_ghz(self):
        """Test probabilities_dict method for GHZ state"""

        state = (Statevector.from_label("000") + Statevector.from_label("111")) / np.sqrt(2)

        # 3-qubit qargs
        target = {"000": 0.5, "111": 0.5}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 2-qubit qargs
        target = {"00": 0.5, "11": 0.5}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        target = {"0": 0.5, "1": 0.5}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

    def test_probabilities_dict_w(self):
        """Test probabilities_dict method with W state"""

        state = (
            Statevector.from_label("001")
            + Statevector.from_label("010")
            + Statevector.from_label("100")
        ) / np.sqrt(3)

        # 3-qubit qargs
        target = np.array([0, 1 / 3, 1 / 3, 0, 1 / 3, 0, 0, 0])
        target = {"001": 1 / 3, "010": 1 / 3, "100": 1 / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 2-qubit qargs
        target = {"00": 1 / 3, "01": 1 / 3, "10": 1 / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

        # 1-qubit qargs
        target = {"0": 2 / 3, "1": 1 / 3}
        for qargs in [[0], [1], [2]]:
            with self.subTest(msg=f"P({qargs})"):
                probs = state.probabilities_dict(qargs)
                self.assertDictAlmostEqual(probs, target)

    def test_sample_counts_ghz(self):
        """Test sample_counts method for GHZ state"""

        shots = 2000
        threshold = 0.02 * shots
        state = (Statevector.from_label("000") + Statevector.from_label("111")) / np.sqrt(2)
        state.seed(100)

        # 3-qubit qargs
        target = {"000": shots / 2, "111": shots / 2}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:

            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 2-qubit qargs
        target = {"00": shots / 2, "11": shots / 2}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:

            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 1-qubit qargs
        target = {"0": shots / 2, "1": shots / 2}
        for qargs in [[0], [1], [2]]:

            with self.subTest(msg=f"counts (qargs={qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

    def test_sample_counts_w(self):
        """Test sample_counts method for W state"""
        shots = 3000
        threshold = 0.02 * shots
        state = (
            Statevector.from_label("001")
            + Statevector.from_label("010")
            + Statevector.from_label("100")
        ) / np.sqrt(3)
        state.seed(100)

        target = {"001": shots / 3, "010": shots / 3, "100": shots / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:

            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 2-qubit qargs
        target = {"00": shots / 3, "01": shots / 3, "10": shots / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:

            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

        # 1-qubit qargs
        target = {"0": 2 * shots / 3, "1": shots / 3}
        for qargs in [[0], [1], [2]]:

            with self.subTest(msg=f"P({qargs})"):
                counts = state.sample_counts(shots, qargs=qargs)
                self.assertDictAlmostEqual(counts, target, threshold)

    def test_probabilities_dict_unequal_dims(self):
        """Test probabilities_dict for a state with unequal subsystem dimensions."""

        vec = np.zeros(60, dtype=float)
        vec[15:20] = np.ones(5)
        vec[40:46] = np.ones(6)
        state = Statevector(vec / np.sqrt(11.0), dims=[3, 4, 5])

        p = 1.0 / 11.0

        self.assertDictEqual(
            state.probabilities_dict(),
            dict.fromkeys(
                ["110", "111", "112", "120", "121", "311", "312", "320", "321", "322", "330"], p
            ),
        )

        # differences due to rounding
        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[0]), {"0": 4 * p, "1": 4 * p, "2": 3 * p}, delta=1e-10
        )

        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[1]), {"1": 5 * p, "2": 5 * p, "3": p}, delta=1e-10
        )

        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[2]), {"1": 5 * p, "3": 6 * p}, delta=1e-10
        )

        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[0, 1]),
            {"10": p, "11": 2 * p, "12": 2 * p, "20": 2 * p, "21": 2 * p, "22": p, "30": p},
            delta=1e-10,
        )

        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[1, 0]),
            {"01": p, "11": 2 * p, "21": 2 * p, "02": 2 * p, "12": 2 * p, "22": p, "03": p},
            delta=1e-10,
        )

        self.assertDictAlmostEqual(
            state.probabilities_dict(qargs=[0, 2]),
            {"10": 2 * p, "11": 2 * p, "12": p, "31": 2 * p, "32": 2 * p, "30": 2 * p},
            delta=1e-10,
        )

    def test_sample_counts_qutrit(self):
        """Test sample_counts method for qutrit state"""
        p = 0.3
        shots = 1000
        threshold = 0.03 * shots
        state = Statevector([np.sqrt(p), 0, np.sqrt(1 - p)])
        state.seed(100)

        with self.subTest(msg="counts"):
            target = {"0": shots * p, "2": shots * (1 - p)}
            counts = state.sample_counts(shots=shots)
            self.assertDictAlmostEqual(counts, target, threshold)

    def test_sample_memory_ghz(self):
        """Test sample_memory method for GHZ state"""

        shots = 2000
        state = (Statevector.from_label("000") + Statevector.from_label("111")) / np.sqrt(2)
        state.seed(100)

        # 3-qubit qargs
        target = {"000": shots / 2, "111": shots / 2}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {"00": shots / 2, "11": shots / 2}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": shots / 2, "1": shots / 2}
        for qargs in [[0], [1], [2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

    def test_sample_memory_w(self):
        """Test sample_memory method for W state"""
        shots = 3000
        state = (
            Statevector.from_label("001")
            + Statevector.from_label("010")
            + Statevector.from_label("100")
        ) / np.sqrt(3)
        state.seed(100)

        target = {"001": shots / 3, "010": shots / 3, "100": shots / 3}
        for qargs in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [1, 0, 2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 2-qubit qargs
        target = {"00": shots / 3, "01": shots / 3, "10": shots / 3}
        for qargs in [[0, 1], [2, 1], [1, 2], [1, 2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

        # 1-qubit qargs
        target = {"0": 2 * shots / 3, "1": shots / 3}
        for qargs in [[0], [1], [2]]:

            with self.subTest(msg=f"memory (qargs={qargs})"):
                memory = state.sample_memory(shots, qargs=qargs)
                self.assertEqual(len(memory), shots)
                self.assertEqual(set(memory), set(target))

    def test_sample_memory_qutrit(self):
        """Test sample_memory method for qutrit state"""
        p = 0.3
        shots = 1000
        state = Statevector([np.sqrt(p), 0, np.sqrt(1 - p)])
        state.seed(100)

        with self.subTest(msg="memory"):
            memory = state.sample_memory(shots)
            self.assertEqual(len(memory), shots)
            self.assertEqual(set(memory), {"0", "2"})

    def test_reset_2qubit(self):
        """Test reset method for 2-qubit state"""

        state = Statevector(np.array([1, 0, 0, 1]) / np.sqrt(2))
        state.seed(100)

        with self.subTest(msg="reset"):
            psi = state.copy()
            value = psi.reset()
            target = Statevector(np.array([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset"):
            psi = state.copy()
            value = psi.reset([0, 1])
            target = Statevector(np.array([1, 0, 0, 0]))
            self.assertEqual(value, target)

        with self.subTest(msg="reset [0]"):
            psi = state.copy()
            value = psi.reset([0])
            targets = [Statevector(np.array([1, 0, 0, 0])), Statevector(np.array([0, 0, 1, 0]))]
            self.assertIn(value, targets)

        with self.subTest(msg="reset [0]"):
            psi = state.copy()
            value = psi.reset([1])
            targets = [Statevector(np.array([1, 0, 0, 0])), Statevector(np.array([0, 1, 0, 0]))]
            self.assertIn(value, targets)

    def test_reset_qutrit(self):
        """Test reset method for qutrit"""

        state = Statevector(np.array([1, 1, 1]) / np.sqrt(3))
        state.seed(200)
        value = state.reset()
        target = Statevector(np.array([1, 0, 0]))
        self.assertEqual(value, target)

    def test_measure_2qubit(self):
        """Test measure method for 2-qubit state"""

        state = Statevector.from_label("+0")
        seed = 200
        shots = 100

        with self.subTest(msg="measure"):
            for i in range(shots):
                psi = state.copy()
                psi.seed(seed + i)
                outcome, value = psi.measure()
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = Statevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = Statevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [0, 1]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([0, 1])
                self.assertIn(outcome, ["00", "10"])
                if outcome == "00":
                    target = Statevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = Statevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [1, 0]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([1, 0])
                self.assertIn(outcome, ["00", "01"])
                if outcome == "00":
                    target = Statevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = Statevector.from_label("10")
                    self.assertEqual(value, target)

        with self.subTest(msg="measure [0]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([0])
                self.assertEqual(outcome, "0")
                target = Statevector(np.array([1, 0, 1, 0]) / np.sqrt(2))
                self.assertEqual(value, target)

        with self.subTest(msg="measure [1]"):
            for i in range(shots):
                psi = state.copy()
                outcome, value = psi.measure([1])
                self.assertIn(outcome, ["0", "1"])
                if outcome == "0":
                    target = Statevector.from_label("00")
                    self.assertEqual(value, target)
                else:
                    target = Statevector.from_label("10")
                    self.assertEqual(value, target)

    def test_measure_qutrit(self):
        """Test measure method for qutrit"""

        state = Statevector(np.array([1, 1, 1]) / np.sqrt(3))
        seed = 200
        shots = 100

        for i in range(shots):
            psi = state.copy()
            psi.seed(seed + i)
            outcome, value = psi.measure()
            self.assertIn(outcome, ["0", "1", "2"])
            if outcome == "0":
                target = Statevector([1, 0, 0])
                self.assertEqual(value, target)
            elif outcome == "1":
                target = Statevector([0, 1, 0])
                self.assertEqual(value, target)
            else:
                target = Statevector([0, 0, 1])
                self.assertEqual(value, target)

    def test_from_int(self):
        """Test from_int method"""

        with self.subTest(msg="from_int(0, 4)"):
            target = Statevector([1, 0, 0, 0])
            value = Statevector.from_int(0, 4)
            self.assertEqual(target, value)

        with self.subTest(msg="from_int(3, 4)"):
            target = Statevector([0, 0, 0, 1])
            value = Statevector.from_int(3, 4)
            self.assertEqual(target, value)

        with self.subTest(msg="from_int(8, (3, 3))"):
            target = Statevector([0, 0, 0, 0, 0, 0, 0, 0, 1], dims=(3, 3))
            value = Statevector.from_int(8, (3, 3))
            self.assertEqual(target, value)

    def test_expval(self):
        """Test expectation_value method"""

        psi = Statevector([1, 0, 0, 1]) / np.sqrt(2)
        for label, target in [
            ("II", 1),
            ("XX", 1),
            ("YY", -1),
            ("ZZ", 1),
            ("IX", 0),
            ("YZ", 0),
            ("ZX", 0),
            ("YI", 0),
        ]:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = psi.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        psi = Statevector([np.sqrt(2), 0, 0, 0, 0, 0, 0, 1 + 1j]) / 2
        for label, target in [
            ("XXX", np.sqrt(2) / 2),
            ("YYY", -np.sqrt(2) / 2),
            ("ZZZ", 0),
            ("XYZ", 0),
            ("YIY", 0),
        ]:
            with self.subTest(msg=f"<{label}>"):
                op = Pauli(label)
                expval = psi.expectation_value(op)
                self.assertAlmostEqual(expval, target)

        labels = ["XXX", "IXI", "YYY", "III"]
        coeffs = [3.0, 5.5, -1j, 23]
        spp_op = SparsePauliOp.from_list(list(zip(labels, coeffs)))
        expval = psi.expectation_value(spp_op)
        target = 25.121320343559642 + 0.7071067811865476j
        self.assertAlmostEqual(expval, target)

    @data(
        "II",
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
        "-II",
        "-IX",
        "-IY",
        "-IZ",
        "-XI",
        "-XX",
        "-XY",
        "-XZ",
        "-YI",
        "-YX",
        "-YY",
        "-YZ",
        "-ZI",
        "-ZX",
        "-ZY",
        "-ZZ",
        "iII",
        "iIX",
        "iIY",
        "iIZ",
        "iXI",
        "iXX",
        "iXY",
        "iXZ",
        "iYI",
        "iYX",
        "iYY",
        "iYZ",
        "iZI",
        "iZX",
        "iZY",
        "iZZ",
        "-iII",
        "-iIX",
        "-iIY",
        "-iIZ",
        "-iXI",
        "-iXX",
        "-iXY",
        "-iXZ",
        "-iYI",
        "-iYX",
        "-iYY",
        "-iYZ",
        "-iZI",
        "-iZX",
        "-iZY",
        "-iZZ",
    )
    def test_expval_pauli(self, pauli):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = Pauli(pauli)
        state = random_statevector(2**op.num_qubits, seed=seed)
        target = state.expectation_value(op.to_matrix())
        expval = state.expectation_value(op)
        self.assertAlmostEqual(expval, target)

    @data([0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1])
    def test_expval_pauli_qargs(self, qubits):
        """Test expectation_value method for Pauli op"""
        seed = 1020
        op = random_pauli(2, seed=seed)
        state = random_statevector(2**3, seed=seed)
        target = state.expectation_value(op.to_matrix(), qubits)
        expval = state.expectation_value(op, qubits)
        self.assertAlmostEqual(expval, target)

    def test_expval_identity(self):
        """Test whether the calculation for identity operator has been fixed"""

        # 1 qubit case test
        state_1 = Statevector.from_label("0")
        state_1_n1 = 2 * state_1  # test the same state with different norms
        state_1_n2 = (1 + 2j) * state_1
        identity_op_1 = SparsePauliOp.from_list([("I", 1)])
        expval_state_1 = state_1.expectation_value(identity_op_1)
        expval_state_1_n1 = state_1_n1.expectation_value(identity_op_1)
        expval_state_1_n2 = state_1_n2.expectation_value(identity_op_1)
        self.assertAlmostEqual(expval_state_1, 1.0 + 0j)
        self.assertAlmostEqual(expval_state_1_n1, 4 + 0j)
        self.assertAlmostEqual(expval_state_1_n2, 5 + 0j)

        for prefix, factor in [("", 1), ("-", -1), ("i", 1j), ("-i", -1j)]:
            pauli_label = f"{prefix}I"
            identity_pauli_1 = Pauli(pauli_label)
            with self.subTest(pauli=pauli_label):
                self.assertAlmostEqual(state_1.expectation_value(identity_pauli_1), factor + 0j)
                self.assertAlmostEqual(
                    state_1_n1.expectation_value(identity_pauli_1), factor * 4 + 0j
                )
                self.assertAlmostEqual(
                    state_1_n2.expectation_value(identity_pauli_1), factor * 5 + 0j
                )

        # Let's try a multi-qubit case
        n_qubits = 3
        state_coeff = 3 - 4j
        op_coeff = 2 - 2j
        state_test = state_coeff * Statevector.from_label("0" * n_qubits)
        op_test = SparsePauliOp.from_list([("I" * n_qubits, op_coeff)])
        expval = state_test.expectation_value(op_test)
        target = op_coeff * np.abs(state_coeff) ** 2
        self.assertAlmostEqual(expval, target)

        coeffs = np.array([1 + 2j, -3j, 4])
        op_multi = SparsePauliOp.from_list([("I" * n_qubits, coeff) for coeff in coeffs])
        expval_multi = state_test.expectation_value(op_multi)
        target_multi = np.sum(coeffs) * np.abs(state_coeff) ** 2
        self.assertAlmostEqual(expval_multi, target_multi)

        for prefix, factor in [("", 1), ("-", -1), ("i", 1j), ("-i", -1j)]:
            pauli_label = f"{prefix}{'I' * n_qubits}"
            identity_pauli_n = Pauli(pauli_label)
            with self.subTest(pauli=pauli_label):
                expval_pauli = state_test.expectation_value(identity_pauli_n)
                target_pauli = factor * np.abs(state_coeff) ** 2
                self.assertAlmostEqual(expval_pauli, target_pauli)

    @data(*(qargs for i in range(4) for qargs in permutations(range(4), r=i + 1)))
    def test_probabilities_qargs(self, qargs):
        """Test probabilities method with qargs"""
        # Get initial state
        nq = 4
        nc = len(qargs)
        state_circ = QuantumCircuit(nq, nc)
        for i in range(nq):
            state_circ.ry((i + 1) * np.pi / (nq + 1), i)

        # Get probabilities
        state = Statevector(state_circ)
        probs = state.probabilities(qargs)

        # Estimate target probs from simulator measurement
        sim = BasicSimulator()
        shots = 5000
        seed = 100
        circ = transpile(state_circ, sim)
        circ.measure(qargs, range(nc))
        result = sim.run(circ, shots=shots, seed_simulator=seed).result()
        target = np.zeros(2**nc, dtype=float)
        for i, p in result.get_counts(0).int_outcomes().items():
            target[i] = p / shots
        # Compare
        delta = np.linalg.norm(probs - target)
        self.assertLess(delta, 0.05)

    def test_global_phase(self):
        """Test global phase is handled correctly when evolving statevector."""

        qc = QuantumCircuit(1)
        qc.rz(0.5, 0)
        qc2 = transpile(qc, basis_gates=["p"])
        sv = Statevector.from_instruction(qc2)
        expected = np.array([0.96891242 - 0.24740396j, 0])
        self.assertEqual(float(qc2.global_phase), 2 * np.pi - 0.25)
        self.assertEqual(sv, Statevector(expected))

    def test_reverse_qargs(self):
        """Test reverse_qargs method"""
        circ1 = QFTGate(5).definition
        circ2 = circ1.reverse_bits()

        state1 = Statevector.from_instruction(circ1)
        state2 = Statevector.from_instruction(circ2)
        self.assertEqual(state1.reverse_qargs(), state2)

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "requires matplotlib")
    @unittest.skipUnless(optionals.HAS_PYLATEX, "requires pylatexenc")
    def test_drawings(self):
        """Test draw method"""
        qc1 = QFTGate(5).definition
        sv = Statevector.from_instruction(qc1)
        with self.subTest(msg="str(statevector)"):
            str(sv)
        for drawtype in ["repr", "text", "latex", "latex_source", "qsphere", "hinton", "bloch"]:
            with self.subTest(msg=f"draw('{drawtype}')"):
                sv.draw(drawtype)
        with self.subTest(msg=" draw('latex', convention='vector')"):
            sv.draw("latex", convention="vector")

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_for_none(self):
        """
        Test for `\rangleNone` output in latex representation
        See https://github.com/Qiskit/qiskit-terra/issues/8169
        """
        sv = Statevector(
            [
                7.07106781e-01 - 8.65956056e-17j,
                -5.55111512e-17 - 8.65956056e-17j,
                7.85046229e-17 + 8.65956056e-17j,
                -7.07106781e-01 + 8.65956056e-17j,
                0.00000000e00 + 0.00000000e00j,
                -0.00000000e00 + 0.00000000e00j,
                -0.00000000e00 + 0.00000000e00j,
                0.00000000e00 - 0.00000000e00j,
            ],
            dims=(2, 2, 2),
        )
        latex_representation = state_to_latex(sv)
        self.assertEqual(
            latex_representation,
            "\\frac{\\sqrt{2}}{2} |000\\rangle- \\frac{\\sqrt{2}}{2} |011\\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_for_large_statevector(self):
        """Test conversion of large dense state vector"""
        sv = Statevector(np.ones((2**15, 1)))
        latex_representation = state_to_latex(sv)
        self.assertEqual(
            latex_representation,
            " |000000000000000\\rangle+ |000000000000001\\rangle+ |000000000000010\\rangle+"
            " |000000000000011\\rangle+ |000000000000100\\rangle+ |000000000000101\\rangle +"
            " \\ldots + |111111111111011\\rangle+ |111111111111100\\rangle+"
            " |111111111111101\\rangle+ |111111111111110\\rangle+ |111111111111111\\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_with_prefix(self):
        """Test adding prefix to state vector latex output"""
        psi = Statevector(np.array([np.sqrt(1 / 2), 0, 0, np.sqrt(1 / 2)]))
        prefix = "|\\psi_{AB}\\rangle = "
        latex_sv = state_to_latex(psi)
        latex_expected = prefix + latex_sv
        latex_representation = state_to_latex(psi, prefix=prefix)
        self.assertEqual(latex_representation, latex_expected)

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_for_large_sparse_statevector(self):
        """Test conversion of large sparse state vector"""
        sv = Statevector(np.eye(2**15, 1))
        latex_representation = state_to_latex(sv)
        self.assertEqual(latex_representation, " |000000000000000\\rangle")

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_with_max_size_limit(self):
        """Test limit the maximum number of non-zero terms in the expression"""
        sv = Statevector(
            [
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 - 0.35355339j,
                0.0 + 0.35355339j,
                0.0 + 0.35355339j,
                0.0 - 0.35355339j,
            ],
            dims=(2, 2, 2, 2),
        )
        latex_representation = state_to_latex(sv, max_size=5)
        self.assertEqual(
            latex_representation,
            "\\frac{\\sqrt{2}}{4} |0000\\rangle+"
            "\\frac{\\sqrt{2}}{4} |0001\\rangle + "
            "\\ldots +"
            "\\frac{\\sqrt{2} i}{4} |1110\\rangle- "
            "\\frac{\\sqrt{2} i}{4} |1111\\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_state_to_latex_with_decimals_round(self):
        """Test rounding of decimal places in the expression"""
        sv = Statevector(
            [
                0.35355339 + 0.0j,
                0.35355339 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 - 0.35355339j,
                0.0 + 0.35355339j,
            ],
            dims=(2, 2, 2),
        )
        latex_representation = state_to_latex(sv, decimals=3)
        self.assertEqual(
            latex_representation,
            "0.354 |000\\rangle+0.354 |001\\rangle- 0.354 i |110\\rangle+0.354 i |111\\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_statevector_draw_latex_regression(self):
        """Test numerical rounding errors are not printed"""
        sv = Statevector(np.array([1 - 8e-17, 8.32667268e-17j]))
        latex_string = sv.draw(output="latex_source")
        self.assertTrue(latex_string.startswith(" |0\\rangle"))
        self.assertNotIn("|1\\rangle", latex_string)

    def test_statevctor_iter(self):
        """Test iteration over a state vector"""
        empty_vector = []
        dummy_vector = [1, 2, 3]
        empty_sv = Statevector([])
        sv = Statevector(dummy_vector)

        # Assert that successive iterations behave as expected, i.e., the
        # iterator is reset upon each exhaustion of the corresponding
        # collection of elements.
        for _ in range(2):
            self.assertEqual(empty_vector, list(empty_sv))
            self.assertEqual(dummy_vector, list(sv))

    def test_statevector_len(self):
        """Test state vector length"""
        empty_vector = []
        dummy_vector = [1, 2, 3]
        empty_sv = Statevector([])
        sv = Statevector(dummy_vector)
        self.assertEqual(len(empty_vector), len(empty_sv))
        self.assertEqual(len(dummy_vector), len(sv))

    def test_clip_probabilities(self):
        """Test probabilities are clipped to [0, 1]."""
        sv = Statevector([1.1, 0])

        self.assertEqual(list(sv.probabilities()), [1.0, 0.0])
        # The "1" key should be zero and therefore omitted.
        self.assertEqual(sv.probabilities_dict(), {"0": 1.0})

    def test_round_probabilities(self):
        """Test probabilities are correctly rounded.

        This is good to test to ensure clipping, renormalizing and rounding work together.
        """
        p = np.sqrt(1 / 3)
        sv = Statevector([p, p, p, 0])
        expected = [0.33, 0.33, 0.33, 0]
        self.assertEqual(list(sv.probabilities(decimals=2)), expected)


if __name__ == "__main__":
    unittest.main()
