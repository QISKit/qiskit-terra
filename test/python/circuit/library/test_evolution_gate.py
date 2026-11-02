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

"""Test the evolution gate."""

from itertools import permutations

import unittest
import numpy as np
import scipy
from ddt import ddt, data, unpack

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate, PhaseGate, RZGate
from qiskit.circuit.library.pauli_evolution import _contains_projectors, _merge_two_pauli_evolutions
from qiskit.synthesis import LieTrotter, SuzukiTrotter, MatrixExponential, QDrift
from qiskit.synthesis.evolution.product_formula import reorder_paulis
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Operator, SparsePauliOp, Pauli, Statevector, SparseObservable
from qiskit.transpiler.passes import HLSConfig, HighLevelSynthesis
from qiskit.utils import optionals
from qiskit.circuit._utils import _compute_control_matrix
from test import QiskitTestCase, combine

X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")
I = SparsePauliOp("I")  # noqa: E741


@ddt
class TestEvolutionGate(QiskitTestCase):
    """Test the evolution gate."""

    def setUp(self):
        super().setUp()
        # fix random seed for reproducibility (used in QDrift)
        self.seed = 2

    def assertSuzukiTrotterIsCorrect(self, gate):
        """Assert the Suzuki Trotter evolution is correct."""
        op = gate.operator
        time = gate.time
        synthesis = gate.synthesis

        exact_suzuki = SuzukiTrotter(
            reps=synthesis.reps, order=synthesis.order, atomic_evolution=exact_atomic_evolution
        )
        exact_gate = PauliEvolutionGate(op, time, synthesis=exact_suzuki)

        self.assertTrue(Operator(gate).equiv(exact_gate))

    def test_matrix_decomposition(self):
        """Test the matrix decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123

        matrix = op.to_matrix()
        evolved = scipy.linalg.expm(-1j * time * matrix)

        with self.subTest(msg="test MatrixExponential"):
            evo_gate = PauliEvolutionGate(op, time, synthesis=MatrixExponential())
            self.assertTrue(Operator(evo_gate).equiv(evolved))

        with self.subTest(msg="test to_matrix"):
            evo_gate = PauliEvolutionGate(op, time)
            self.assertTrue(np.allclose(evolved, evo_gate.to_matrix()))

    def test_to_matrix_commuting_blocks(self):
        """Test to_matrix if the evolution is specified with commuting blocks."""
        block_1q = (X ^ I) + (I ^ X)
        block_2q = (X ^ X) + (Y ^ Y) + (Z ^ Z)

        evo = PauliEvolutionGate([block_1q, block_2q])
        matrix = evo.to_matrix()

        expected = PauliEvolutionGate(block_1q + block_2q).to_matrix()
        self.assertTrue(np.allclose(expected, matrix))

    def test_to_matrix_invalid_time(self):
        """Test calling to_matrix with a non-numeric time fails."""
        evo = PauliEvolutionGate(Z, time=Parameter("time"))
        with self.assertRaises(ValueError):
            _ = evo.to_matrix()

    def test_reorder_paulis_invariant(self):
        """
        Tests that reorder_paulis is deterministic and does not depend on the
        order of the terms of the input operator.
        """
        terms = [
            (I ^ I ^ X ^ X),
            (I ^ I ^ Z ^ Z),
            (I ^ Y ^ Y ^ I),
            (X ^ I ^ I ^ I),
            (X ^ X ^ I ^ I),
            (Y ^ I ^ I ^ Y),
        ]
        results = []
        for seed, tms in enumerate(permutations(terms)):
            np.random.seed(seed)
            op = reorder_paulis(SparsePauliOp(sum(tms)).to_sparse_list())
            results.append([(t[0], t[1]) for t in op])
            np.random.seed(seed + 42)
            op = reorder_paulis(SparsePauliOp(sum(tms)).to_sparse_list())
            results.append([(t[0], t[1]) for t in op])

        for lst in results[1:]:
            self.assertListEqual(lst, results[0])

    def test_lie_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4
        evo_gate = PauliEvolutionGate(op, time, synthesis=LieTrotter(reps=reps))
        decomposed = evo_gate.definition.decompose()

        self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_basis_change(self):
        """Test the basis change is correctly implemented."""
        op = I ^ Y  # use a string for which we do not have a basis gate
        time = 0.321
        evo_gate = PauliEvolutionGate(op, time)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_rzx_order(self):
        """Test ZX and XZ is mapped onto the correct qubits."""

        for op, indices in zip([X ^ Z, Z ^ X], [(0, 1), (1, 0)]):
            with self.subTest(op=op, indices=indices):
                evo_gate = PauliEvolutionGate(op)
                decomposed = evo_gate.definition.decompose()

                #           ┌───┐┌───────┐┌───┐
                # q_0: ─────┤ X ├┤ Rz(2) ├┤ X ├─────
                #      ┌───┐└─┬─┘└───────┘└─┬─┘┌───┐
                # q_1: ┤ H ├──■─────────────■──┤ H ├
                #      └───┘                   └───┘
                ref = QuantumCircuit(2)
                ref.h(indices[1])
                ref.cx(indices[1], indices[0])
                ref.rz(2.0, indices[0])
                ref.cx(indices[1], indices[0])
                ref.h(indices[1])

                # don't use circuit equality since RZX here decomposes with RZ on the bottom
                self.assertTrue(Operator(decomposed).equiv(ref))
                # check the circuit indeed contains a single RZX gate
                self.assertTrue(evo_gate.definition.count_ops().get("rzx", 0), 1)

    def test_suzuki_trotter(self):
        """Test constructing the circuit with Lie Trotter decomposition."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4
        for order in [2, 4, 6]:
            if order == 2:
                expected_cx = reps * 5 * 4
            elif order % 2 == 0:
                # recurse (order - 2) / 2 times, base case has 5 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 2) / 2) * 5 * 4
            else:
                # recurse (order - 1) / 2 times, base case has 3 blocks with 4 CX each
                expected_cx = reps * 5 ** ((order - 1) / 2) * 3 * 4

            with self.subTest(order=order):
                evo_gate = PauliEvolutionGate(
                    op, time, synthesis=SuzukiTrotter(order=order, reps=reps)
                )
                decomposed = evo_gate.definition.decompose()
                self.assertEqual(decomposed.count_ops()["cx"], expected_cx)
                self.assertSuzukiTrotterIsCorrect(evo_gate)

    def test_suzuki_trotter_manual_no_reorder(self):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = X + Y
        time = 0.1
        reps = 1
        evo_gate = PauliEvolutionGate(
            op, time, synthesis=SuzukiTrotter(order=4, reps=reps, preserve_order=True)
        )

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        p_4 = 1 / (4 - 4 ** (1 / 3))  # coefficient for reduced time from Suzuki paper
        for _ in range(2):
            # factor of 1/2 already included with factor 1/2 in rotation gate
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        expected.rx((1 - 4 * p_4) * time, 0)
        expected.ry(2 * (1 - 4 * p_4) * time, 0)
        expected.rx((1 - 4 * p_4) * time, 0)

        for _ in range(2):
            expected.rx(p_4 * time, 0)
            expected.ry(2 * p_4 * time, 0)
            expected.rx(p_4 * time, 0)

        self.assertEqual(evo_gate.definition, expected)
        self.assertSuzukiTrotterIsCorrect(evo_gate)

    @data(True, False)
    def test_suzuki_trotter_manual(self, use_plugin):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        op = (X ^ X ^ I ^ I) + (I ^ Y ^ Y ^ I) + (I ^ I ^ Z ^ Z)
        time, reps = 0.1, 1

        synthesis = SuzukiTrotter(order=2, reps=reps)
        if use_plugin:
            hls_config = HLSConfig(PauliEvolution=[("basic", {"preserve_order": False})])
        else:
            synthesis.preserve_order = False
            hls_config = None

        evo_gate = PauliEvolutionGate(op, time, synthesis=synthesis)
        circuit = QuantumCircuit(op.num_qubits)
        circuit.append(evo_gate, circuit.qubits)

        if use_plugin:
            decomposed = HighLevelSynthesis(hls_config=hls_config)(circuit)
        else:
            decomposed = circuit.decompose()

        expected = QuantumCircuit(4)
        expected.rzz(time, 0, 1)
        expected.rxx(time, 2, 3)
        expected.ryy(2 * time, 1, 2)
        expected.rxx(time, 2, 3)
        expected.rzz(time, 0, 1)
        self.assertEqual(decomposed, expected)

    def test_suzuki_trotter_plugin(self):
        """Test setting options via the plugin."""

    @data(
        (X + Y, 0.5, 1, [(Pauli("X"), 0.5), (Pauli("X"), 0.5)]),
        (X, 0.238, 2, [(Pauli("X"), 0.238)]),
    )
    @unpack
    def test_qdrift_manual(self, op, time, reps, sampled_ops):
        """Test the evolution circuit of Suzuki Trotter against a manually constructed circuit."""
        qdrift = QDrift(reps=reps, seed=self.seed)
        evo_gate = PauliEvolutionGate(op, time, synthesis=qdrift)
        evo_gate.definition.decompose()

        # manually construct expected evolution
        expected = QuantumCircuit(1)
        for pauli in sampled_ops:
            if pauli[0].to_label() == "X":
                expected.rx(2 * pauli[1], 0)
            elif pauli[0].to_label() == "Y":
                expected.ry(2 * pauli[1], 0)

        self.assertTrue(Operator(evo_gate.definition).equiv(expected))

    def test_qdrift_evolution(self):
        """Test QDrift on an example."""
        op = 0.1 * (Z ^ Z) - 3.2 * (X ^ I) - 1.0 * (I ^ X) + 0.2 * (X ^ X)
        reps = 20
        time = 0.12
        num_samples = 300
        qdrift_energy = []

        def energy(evo):
            return Statevector(evo).expectation_value(op.to_matrix())

        for i in range(num_samples):
            qdrift = PauliEvolutionGate(
                op, time=time, synthesis=QDrift(reps=reps, seed=self.seed + i)
            ).definition

            qdrift_energy.append(energy(qdrift))

        exact = scipy.linalg.expm(-1j * time * op.to_matrix()).dot(np.eye(4)[0, :])

        self.assertAlmostEqual(energy(exact), np.average(qdrift_energy), places=2)

    def test_qdrift_zero_time(self):
        gate = PauliEvolutionGate(SparsePauliOp("X", coeffs=[1]), time=0)
        expanded = QDrift(seed=123).expand(gate)
        self.assertEqual(expanded, [])

    def test_qdrift_zero_hermitian(self):
        gate = PauliEvolutionGate(SparsePauliOp("X", coeffs=[0]), time=1)
        expanded = QDrift(seed=123).expand(gate)
        self.assertEqual(expanded, [])

    @data(True, False)
    def test_passing_grouped_paulis(self, wrap):
        """Test passing a list of already grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter(wrap=wrap))
        if wrap:
            decomposed = evo_gate.definition.decompose()
        else:
            decomposed = evo_gate.definition

        self.assertEqual(decomposed.count_ops()["rz"], 4)
        self.assertEqual(decomposed.count_ops()["rzz"], 1)
        self.assertEqual(decomposed.count_ops()["rxx"], 1)

    def test_list_from_grouped_paulis(self):
        """Test getting a string representation from grouped Paulis."""
        grouped_ops = [(X ^ Y) + (Y ^ X), (Z ^ I) + (Z ^ Z) + (I ^ Z), (X ^ X)]
        evo_gate = PauliEvolutionGate(grouped_ops, time=0.12, synthesis=LieTrotter())

        pauli_strings = []
        for op in evo_gate.operator:
            if isinstance(op, SparsePauliOp):
                pauli_strings.append(op.to_list())
            else:
                pauli_strings.append([(str(op), 1 + 0j)])

        expected = [
            [("XY", 1 + 0j), ("YX", 1 + 0j)],
            [("ZI", 1 + 0j), ("ZZ", 1 + 0j), ("IZ", 1 + 0j)],
            [("XX", 1 + 0j)],
        ]
        self.assertListEqual(pauli_strings, expected)

    def test_dag_conversion(self):
        """Test constructing a circuit with evolutions yields a DAG with evolution blocks."""
        time = Parameter("t")
        evo = PauliEvolutionGate((Z ^ Z) + (X ^ X), time=time)

        circuit = QuantumCircuit(2)
        circuit.h(circuit.qubits)
        circuit.append(evo, circuit.qubits)
        circuit.cx(0, 1)

        dag = circuit_to_dag(circuit)

        expected_ops = {"HGate", "CXGate", "PauliEvolutionGate"}
        ops = {node.op.base_class.__name__ for node in dag.op_nodes()}

        self.assertEqual(ops, expected_ops)

    @combine(option=["chain", "fountain"], use_sparse_observable=[True, False])
    def test_cnot_chain_options(self, option, use_sparse_observable):
        """Test selecting different kinds of CNOT chains."""
        op = SparseObservable("ZZZ") if use_sparse_observable else SparsePauliOp(["ZZZ"])
        synthesis = LieTrotter(reps=1, cx_structure=option)
        evo = PauliEvolutionGate(op, synthesis=synthesis)

        expected = QuantumCircuit(3)
        if option == "chain":
            expected.cx(2, 1)
            expected.cx(1, 0)
        else:
            expected.cx(1, 0)
            expected.cx(2, 0)

        expected.rz(2, 0)

        if option == "chain":
            expected.cx(1, 0)
            expected.cx(2, 1)
        else:
            expected.cx(2, 0)
            expected.cx(1, 0)

        self.assertEqual(expected, evo.definition)
        self.assertSuzukiTrotterIsCorrect(evo)

    @data(
        Pauli("XI"),
        SparsePauliOp(Pauli("XI")),
    )
    def test_different_input_types(self, op):
        """Test all different supported input types and that they yield the same."""
        expected = QuantumCircuit(2)
        expected.rx(4, 1)

        with self.subTest(msg="plain"):
            evo = PauliEvolutionGate(op, time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

        with self.subTest(msg="wrapped in list"):
            evo = PauliEvolutionGate([op], time=2, synthesis=LieTrotter())
            self.assertEqual(evo.definition, expected)

    def test_pauliop_coefficients_respected(self):
        """Test that global ``PauliOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (Z ^ I), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angle = circuit.data[0].operation.params[0]
        self.assertEqual(rz_angle, 10)
        self.assertSuzukiTrotterIsCorrect(evo)

    def test_sparse_pauli_coefficients_respected(self):
        """Test that global ``SparsePauliOp`` coefficients are being taken care of."""
        evo = PauliEvolutionGate(5 * (2 * X + 3 * Y - Z), time=1, synthesis=LieTrotter())
        circuit = evo.definition.decompose()
        rz_angles = [
            circuit.data[0].operation.params[0],  # X
            circuit.data[1].operation.params[0],  # Y
            circuit.data[2].operation.params[0],  # Z
        ]
        self.assertListEqual(rz_angles, [20, 30, -10])
        self.assertSuzukiTrotterIsCorrect(evo)

    def test_lie_trotter_two_qubit_correct_order(self):
        """Test that evolutions on two qubit operators are in the right order.

        Regression test of Qiskit/qiskit-terra#7544.
        """
        operator = I ^ Z ^ Z
        time = 0.5
        lie_trotter = PauliEvolutionGate(operator, time, synthesis=LieTrotter())

        self.assertSuzukiTrotterIsCorrect(lie_trotter)

    def test_lie_trotter_reordered_manual(self):
        """Test the evolution circuit of Lie Trotter against a manually constructed circuit."""
        op = (X ^ I ^ I ^ I) + (X ^ X ^ I ^ I) + (I ^ Y ^ Y ^ I) + (I ^ I ^ Z ^ Z)
        time, reps = 0.1, 1
        evo_gate = PauliEvolutionGate(
            op,
            time,
            synthesis=LieTrotter(reps=reps, preserve_order=False),
        )
        # manually construct expected evolution
        expected = QuantumCircuit(4)
        expected.rxx(2 * time, 2, 3)
        expected.rzz(2 * time, 0, 1)
        expected.rx(2 * time, 3)
        expected.ryy(2 * time, 1, 2)
        self.assertEqual(evo_gate.definition, expected)

    def test_complex_op_raises(self):
        """Test an operator with complex coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(Pauli("iZ"))

    def test_paramtrized_op_raises(self):
        """Test an operator with parametrized coefficient raises an error."""
        with self.assertRaises(ValueError):
            _ = PauliEvolutionGate(SparsePauliOp("Z", np.array(Parameter("t"))))

    @data(LieTrotter, MatrixExponential)
    def test_inverse(self, synth_cls):
        """Test calculating the inverse is correct."""
        evo = PauliEvolutionGate(X + Y, time=0.12, synthesis=synth_cls())

        circuit = QuantumCircuit(1)
        circuit.append(evo, circuit.qubits)
        circuit.append(evo.inverse(), circuit.qubits)

        self.assertTrue(Operator(circuit).equiv(np.identity(2**circuit.num_qubits)))

    def test_inverse_to_matrix(self):
        """Test inverse().to_matrix()."""
        op = X + Y
        time = 0.512
        exact = scipy.linalg.expm(-1j * time * op.to_matrix())
        exact_inv = np.transpose(np.conjugate(exact))

        evo = PauliEvolutionGate(op, time)
        evo_inv = evo.inverse()

        self.assertTrue(np.allclose(exact, evo.to_matrix()))
        self.assertTrue(np.allclose(exact_inv, evo_inv.to_matrix()))

    def test_power(self):
        """Test calling the power method."""
        power = 31
        dt = 0.1
        op = (X ^ X) + (Z ^ Z)  # pick commuting operator to check matrices

        evo = PauliEvolutionGate(op, time=dt)
        evo_pow = evo.power(power)

        expect = PauliEvolutionGate(op, time=power * dt)

        with self.subTest("check efficient power"):
            self.assertEqual(expect, evo_pow)

        matrix = scipy.linalg.expm(-1j * dt * power * op.to_matrix())
        with self.subTest("check unitary"):
            self.assertTrue(np.allclose(matrix, Operator(evo_pow).data))

    def test_control_grouped(self):
        """Test the control method on grouped operators."""
        block_1q = (I ^ X) + (X ^ I)
        block_2q = (Z ^ Z) + (Y ^ Y) + (X ^ X)

        evo = PauliEvolutionGate([block_1q, block_2q], time=1, synthesis=LieTrotter())
        controlled = evo.control(2, ctrl_state="01", annotated=False)

        summed = PauliEvolutionGate(block_1q + block_2q, time=1, synthesis=LieTrotter())
        reference = summed.control(2, ctrl_state="01", annotated=False)

        self.assertEqual(reference, controlled)

    def test_labels_and_name(self):
        """Test the name and labels are correct."""
        operators = [
            SparsePauliOp("XY"),
            (X + Y),
            ((I ^ Z) + (Z ^ I) - 0.2 * (X ^ X)),
            SparseObservable("01Z+-XlrY"),
        ]

        expected_labels = ["XY", "(X + Y)", "(IZ + ZI + XX)", "Y0 r1 l2 X3 -4 +5 Z6 17 08"]
        for op, label in zip(operators, expected_labels):
            with self.subTest(op=op, label=label):
                evo = PauliEvolutionGate(op)
                self.assertEqual(evo.name, "PauliEvolution")
                if isinstance(op, SparseObservable):
                    self.assertIn(f"exp(-it {label})", evo.label)
                else:
                    self.assertEqual(evo.label, f"exp(-it {label})")

    def test_atomic_evolution(self):
        """Test a custom atomic_evolution."""
        op = (X ^ X ^ X) + (Y ^ Y ^ Y) + (Z ^ Z ^ Z)
        time = 0.123
        reps = 4

        with self.subTest(msg="default"):
            with self.assertWarns(PendingDeprecationWarning):
                evo_gate = PauliEvolutionGate(
                    op,
                    time,
                    synthesis=LieTrotter(reps=reps, atomic_evolution=custom_atomic_evolution),
                )
            decomposed = evo_gate.definition.decompose()
            self.assertEqual(decomposed.count_ops()["cx"], reps * 3 * 4)

        with self.subTest(msg="supporting sparse obs"):
            # no pending deprecation warning here
            evo_gate = PauliEvolutionGate(
                op,
                time,
                synthesis=LieTrotter(
                    reps=reps,
                    atomic_evolution=observable_supporting_evolution,
                    atomic_evolution_sparse_observable=True,
                ),
            )

    def test_invalid_atomic_evolution(self):
        """Test a lying about the support of SparseObservable."""
        op = SparseObservable("IYXZ")
        time = 0.123
        reps = 4

        evo_gate = PauliEvolutionGate(
            op,
            time,
            synthesis=LieTrotter(
                reps=reps,
                atomic_evolution=custom_atomic_evolution,
                atomic_evolution_sparse_observable=True,
            ),
        )

        with self.assertRaises(AttributeError):
            _ = evo_gate.definition

    def test_all_identity(self):
        """Test circuit with all identity Paulis works correctly."""
        ops = [I ^ I, Pauli("II"), SparseObservable.identity(2)]

        for op in ops:
            with self.subTest(op=op):
                evo = PauliEvolutionGate(op, time=1).definition
                expected = QuantumCircuit(2, global_phase=-1)
                self.assertEqual(expected, evo)

    def test_global_phase(self):
        """Test a circuit with parameterized global phase terms.

        Regression test of #13625.
        """
        pauli = (X ^ X) + (I ^ I) + (I ^ X)
        time = Parameter("t")
        evo = PauliEvolutionGate(pauli, time=time)

        expected = QuantumCircuit(2, global_phase=-time)
        expected.rxx(2 * time, 0, 1)
        expected.rx(2 * time, 0)

        with self.subTest(msg="check circuit"):
            self.assertEqual(expected, evo.definition)

        # since all terms in the Pauli operator commute, we can compare to an
        # exact matrix exponential
        time_value = 1.76123
        bound = evo.definition.assign_parameters([time_value])
        exact = scipy.linalg.expm(-1j * time_value * pauli.to_matrix())
        with self.subTest(msg="check correctness"):
            self.assertEqual(Operator(exact), Operator(bound))

    @unittest.skipUnless(optionals.HAS_SYMPY, "sympy required")
    def test_sympify_is_real(self):
        """Test converting the parameters to sympy is real.

        Regression test of #13642, where the parameters in the Pauli evolution had a spurious
        zero complex part. Even though this is not noticeable upon binding or printing the parameter,
        it does affect the output of Parameter.sympify.
        """
        time = Parameter("t")
        evo = PauliEvolutionGate(Z, time=time)

        angle = evo.definition.data[0].operation.params[0]
        expected = (2 * time).sympify()
        self.assertEqual(expected, angle.sympify())

    def test_zero(self):
        """Test the SparseObservable zero operator."""
        op = SparseObservable.zero(2)
        evo = PauliEvolutionGate(op, time=1).definition
        expected = QuantumCircuit(2, global_phase=0)
        self.assertEqual(expected, evo)

    @data(
        "0",
        "1",
        "+",
        "-",
        "r",
        "l",
        "XIYZ",
        "001",
        "100",
        "Z00",
        "10Z",
        "+-lr01",
        "+-rl01",
    )
    def test_projectors(self, projector):
        """Test a SparseObservable with projectors."""
        op = SparseObservable(projector)
        evo = PauliEvolutionGate(op, time=1).definition
        pauli = SparsePauliOp.from_sparse_observable(op)
        ref = PauliEvolutionGate(pauli, time=1).definition
        self.assertEqual(Operator(ref), Operator(evo))

    @data(("00+1", [2, 3, 0, 1], "001+"), ("Z0+Y", [1, 3, 0, 2], "0YZ+"))
    @unpack
    def test_projector_custom_index(self, initial, order, final):
        """Test with some random index order."""
        op = SparseObservable.from_sparse_list([(initial, order, 1)], 4)
        evo = PauliEvolutionGate(op, time=1).definition

        direct = SparseObservable(final)
        ref = PauliEvolutionGate(direct, time=1).definition
        self.assertEqual(Operator(ref), Operator(evo))

    def test_projector_circuit(self):
        """Test a SparseObservable with projectors.

        This evolves ``SparseObservable("-+rl10")`` which will translate to the following
        circuit (modulo basis-changing Cliffords)

            q5 q4 q3 q2 q1 q0
            -----------------
             -  +  r  l  1  0  // bit term
            -1 +1 +1 -1 -1 +1  // eigenvalue
             1  0  0  1  1  0  // ctrl state
            -----------------
            --> [X P(-1) X].ctrl_state(11001) applied on qubits [5, 4, 3, 2, 1, 0]

        """
        op = SparseObservable("-+rl10")
        evo = PauliEvolutionGate(op, time=1).definition

        reference = QuantumCircuit(*evo.qregs)
        reference.sx([2, 3])
        reference.h([4, 5])

        reference.x(0)
        reference.append(
            PhaseGate(-1.0).control(5, ctrl_state="11001", annotated=False), reference.qubits[::-1]
        )
        reference.x(0)

        reference.sxdg([2, 3])
        reference.h([4, 5])

        self.assertEqual(reference, evo)

    def test_projector_and_pauli(self):
        """Test a mix of Paulis and projectors."""
        op = SparseObservable.from_list([("01", 1), ("X+", -1), ("YY", 1)])
        evo = PauliEvolutionGate(op, time=1).definition

        pauli = SparsePauliOp.from_sparse_observable(op)
        ref = PauliEvolutionGate(pauli, time=1).definition

        self.assertEqual(Operator(ref), Operator(evo))

    def test_sparse_observable_atomic_evo(self):
        """Test a SparseObservable input with a legacy atomic evolution."""
        op = SparseObservable("IZ01X+-Ylr")
        with self.assertWarns(PendingDeprecationWarning):
            synth = LieTrotter(atomic_evolution=custom_atomic_evolution)

        evo = PauliEvolutionGate(op, time=2, synthesis=synth).definition

        pauli = SparsePauliOp.from_sparse_observable(op)
        ref = PauliEvolutionGate(pauli, time=2, synthesis=synth).definition

        self.assertEqual(ref, evo)

    def test_single_qubit_evolutions(self):
        """Test all single qubit evolutions."""
        op = SparseObservable.from_sparse_list(
            [(pauli, [i], 1) for i, pauli in enumerate("Z01X+-Yrl")], 9
        )
        evo = PauliEvolutionGate(op, time=1).definition

        ref = QuantumCircuit(9)

        # Z
        ref.rz(2, 0)

        # |0><0|
        ref.x(1)
        ref.p(-1, 1)
        ref.x(1)

        # |1><1|
        ref.p(-1, 2)

        # X
        ref.rx(2, 3)

        # |+><+|
        ref.h(4)
        ref.x(4)
        ref.p(-1, 4)
        ref.x(4)
        ref.h(4)

        # |-><-|
        ref.h(5)
        ref.p(-1, 5)
        ref.h(5)

        # Y
        ref.ry(2, 6)

        # |r><r|
        ref.sx(7)
        ref.x(7)
        ref.p(-1, 7)
        ref.x(7)
        ref.sxdg(7)

        # |l><l|
        ref.sx(8)
        ref.p(-1, 8)
        ref.sxdg(8)

        self.assertEqual(ref, evo)

    @data(
        "ZZ+Z+Z++",
        "+ZZ+++++",
        "++++++++",
    )
    def test_gate_count(self, term):
        """Test the gate count upper bound.

        The goal here is to check that the projectors don't use an exponential number of gates,
        not to check the precise numbers.

        Note that this test requires 4 or more projectors to work, otherwise the MCRZ bound
        does not hold.
        """
        num_paulis = sum(bit in "XYZ" for bit in term)
        num_projectors = len(term) - num_paulis

        if num_paulis > 0:
            num_cx = 2 * (num_paulis - 1)  # number of CX for the Pauli part
            num_cx += 16 * num_projectors - 40  # MCRZ gate count
        else:
            num_cx = (num_projectors - 1) * (16 * num_projectors - 40)

        # if we did use an exponential number of CX, this is what we'd get
        exponential_paulis = num_paulis + 2**num_projectors
        exponential_cx = 2 * (exponential_paulis - 1)

        evo = PauliEvolutionGate(SparseObservable(term))
        tqc = transpile(evo.definition, basis_gates=["u", "cx"], optimization_level=0)
        cx_count = tqc.count_ops().get("cx", 0)

        # we should for sure be less than this
        self.assertLess(cx_count, exponential_cx)
        # we should also be less (or equal) to this
        self.assertLessEqual(cx_count, num_cx)

    @data("110", 6)
    def test_ctrl_state(self, ctrl_state):
        """Test controlled evolution gate with a control state."""
        obs = SparseObservable("ZZ")
        evo = PauliEvolutionGate(obs)
        controlled = evo.control(num_ctrl_qubits=3, ctrl_state=ctrl_state, annotated=False)
        qc = controlled.definition

        reference = QuantumCircuit(*qc.qregs)
        reference.cx(4, 3)
        reference.append(RZGate(2).control(3, ctrl_state="011", annotated=False), [2, 1, 0, 3])
        reference.cx(4, 3)
        with self.subTest("check decomp"):
            self.assertEqual(reference, qc)

        zz_mat = np.diag([1, -1, -1, 1])
        rzz_mat = scipy.linalg.expm(-1j * zz_mat)
        ctrl_mat = _compute_control_matrix(rzz_mat, 3, ctrl_state)
        with self.subTest("check correctness"):
            self.assertTrue(np.allclose(ctrl_mat, Operator(qc).data))

    def test_raises_on_empty_list(self):
        """Test that an error gets raised when a Pauli evolution gate is created from an empty list."""
        with self.assertRaises(ValueError):
            PauliEvolutionGate([], time=1)

    def test_raises_on_list_with_different_num_qubits(self):
        """Test that an error gets raised when a Pauli evolution gate is created from a list,
        where not all of the operators have the same number of qubits.
        """

        with self.assertRaises(ValueError):
            pauli = Pauli("XYZ")  # 3 qubits
            op = SparsePauliOp(["XYIZ"], [1])  # 4 qubits
            PauliEvolutionGate([pauli, op], time=1)

    @data(True, False)
    def test_merge_two_pauli_evolutions(self, use_sparse_observable):
        """Test merging two Pauli evolution gates."""
        obs_cls = SparseObservable if use_sparse_observable else SparsePauliOp
        with self.subTest("identical"):
            gate1 = PauliEvolutionGate(obs_cls("XY"))
            gate2 = PauliEvolutionGate(obs_cls("XY"))
            merged = _merge_two_pauli_evolutions(gate1, gate2)
            self.assertIsNotNone(merged)
        with self.subTest("different"):
            gate1 = PauliEvolutionGate(obs_cls("XY"))
            gate2 = PauliEvolutionGate(obs_cls("YX"))
            merged = _merge_two_pauli_evolutions(gate1, gate2)
            self.assertIsNone(merged)
        with self.subTest("identical after canonicalization"):
            gate1 = PauliEvolutionGate(obs_cls("X") + obs_cls("Z"))
            gate2 = PauliEvolutionGate(obs_cls("Z") + obs_cls("X"))
            merged = _merge_two_pauli_evolutions(gate1, gate2)
            self.assertIsNotNone(merged)

    def test_contains_projectors(self):
        """Test correctness of the `_contains_projector` helper function."""
        with self.subTest("SparsePauliOp"):
            gate = PauliEvolutionGate(SparsePauliOp("XY"))
            self.assertFalse(_contains_projectors(gate))
        with self.subTest("SparseObservable without projectors"):
            gate = PauliEvolutionGate(SparseObservable("XY"))
            self.assertFalse(_contains_projectors(gate))
        with self.subTest("SparseObservable with projectors"):
            gate = PauliEvolutionGate(SparseObservable("X+"))
            self.assertTrue(_contains_projectors(gate))
        with self.subTest("List of SparseObservables without projectors"):
            gate = PauliEvolutionGate([SparseObservable("XX"), SparseObservable("YY")])
            self.assertFalse(_contains_projectors(gate))
        with self.subTest("List of SparseObservables with projectors"):
            gate = PauliEvolutionGate([SparseObservable("XX"), SparseObservable("rY")])
            self.assertTrue(_contains_projectors(gate))


def exact_atomic_evolution(circuit, pauli, time):
    """An exact atomic evolution for Suzuki-Trotter.

    Note that the Pauli has a x2 coefficient already, hence we evolve for time/2.
    """
    # workaround until SparseObservable does not natively implement to_matrix
    if isinstance(pauli, SparseObservable):
        pauli = SparsePauliOp.from_sparse_observable(pauli)

    circuit.append(HamiltonianGate(pauli.to_matrix(), time / 2), circuit.qubits)


def diagonalizing_clifford(pauli: Pauli) -> QuantumCircuit:
    """Get the clifford circuit to diagonalize the Pauli operator."""
    cliff = QuantumCircuit(pauli.num_qubits)
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i == "Y":
            cliff.sx(i)
        elif pauli_i == "X":
            cliff.h(i)

    return cliff


def cnot_chain(pauli: Pauli) -> QuantumCircuit:
    """CX chain.

    For example, for the Pauli with the label 'XYZIX'.

    .. parsed-literal::

                       ┌───┐
        q_0: ──────────┤ X ├
                       └─┬─┘
        q_1: ────────────┼──
                  ┌───┐  │
        q_2: ─────┤ X ├──■──
             ┌───┐└─┬─┘
        q_3: ┤ X ├──■───────
             └─┬─┘
        q_4: ──■────────────

    """

    chain = QuantumCircuit(pauli.num_qubits)
    control, target = None, None

    # iterate over the Pauli's and add CNOTs
    for i, pauli_i in enumerate(pauli.to_label()):
        i = pauli.num_qubits - i - 1
        if pauli_i != "I":
            if control is None:
                control = i
            else:
                target = i

        if control is not None and target is not None:
            chain.cx(control, target)
            control = i
            target = None

    return chain


def custom_atomic_evolution(circuit, pauli, time):
    """A custom atomic evolution not supporting SparseObservable."""
    if isinstance(pauli, SparsePauliOp):
        for single_term, coeff in zip(pauli.paulis, pauli.coeffs):
            custom_atomic_evolution(circuit, single_term, np.real(coeff) * time)

        return circuit

    cliff = diagonalizing_clifford(pauli)
    chain = cnot_chain(pauli)

    target = None
    for i, pauli_i in enumerate(reversed(pauli.to_label())):
        if pauli_i != "I":
            target = i
            break

    circuit.compose(cliff, inplace=True)
    circuit.compose(chain, inplace=True)
    circuit.rz(2 * time, target)
    circuit.compose(chain.inverse(), inplace=True)
    circuit.compose(cliff.inverse(), inplace=True)

    return circuit


def observable_supporting_evolution(circuit, pauli, time):
    """A custom atomic evolution supporting SparseObservable."""
    if isinstance(pauli, SparseObservable):
        pauli = SparsePauliOp.from_sparse_observable(pauli)

    custom_atomic_evolution(circuit, pauli, time)


@ddt
class TestPauliEvolutionGateLabels(QiskitTestCase):
    """Tests for verifying PauliEvolutionGate label correctness."""

    def test_single_term_label(self):
        """Test label generation for a single-term SparseObservable."""
        evo = PauliEvolutionGate(SparseObservable.from_list([("XXII", 1)]), time=1)
        expected_label = "exp(-it X2 X3)"
        self.assertEqual(expected_label, evo.label)

    def test_multiple_term_label(self):
        """Test label generation for a multi-term SparseObservable."""
        evo = PauliEvolutionGate(
            SparseObservable.from_list([("IIXX", 1), ("IYYI", 2), ("ZZII", 3)]), time=1
        )
        expected_label = "exp(-it (X0 X1 + Y1 Y2 + Z2 Z3))"
        self.assertEqual(expected_label, evo.label)

    @data(True, False)
    def test_list_of_observables_label(self, use_sparse_observable):
        """Test label generation for a list of SparseObservable operators."""
        obs_cls = SparseObservable if use_sparse_observable else SparsePauliOp
        evo = PauliEvolutionGate(
            [
                obs_cls.from_list([("IIXX", 1), ("IYYI", 2), ("ZZII", 3)]),
                obs_cls.from_list([("XXII", 4)]),
            ],
            time=1,
        )

        if use_sparse_observable:
            expected_label = "exp(-it [(X0 X1 + Y1 Y2 + Z2 Z3), X2 X3])"
        else:
            expected_label = "exp(-it [(IIXX + IYYI + ZZII), XXII])"

        self.assertEqual(expected_label, evo.label)

    def test_circuit_display_labels(self):
        """Test that the labels are correctly displayed in a circuit context."""
        evo = PauliEvolutionGate(SparseObservable.from_list([("XXII", 1)]), time=1)
        qc = QuantumCircuit(4)
        qc.append(evo, [0, 1, 2, 3])
        text = str(qc.draw(output="text"))
        self.assertIn("X2 X3", text)


if __name__ == "__main__":
    unittest.main()
