# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the Solovay Kitaev transpilation pass."""

import os
import unittest
import tempfile
import numpy as np
import scipy

from ddt import ddt, data

from qiskit import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit import ClassicalRegister
from qiskit.circuit.library import (
    TGate,
    TdgGate,
    HGate,
    CXGate,
    SGate,
    SdgGate,
    QFTGate,
    RXGate,
    RYGate,
    RZGate,
    PauliProductRotationGate,
)
from qiskit.circuit import QuantumRegister
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator, Pauli
from qiskit.synthesis.discrete_basis.generate_basis_approximations import (
    generate_basic_approximations,
)
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnitarySynthesis, Collect1qRuns, ConsolidateBlocks
from qiskit.transpiler.passes.synthesis import SolovayKitaev, SolovayKitaevSynthesis
from qiskit.synthesis.discrete_basis import SolovayKitaevDecomposition
from test import QiskitTestCase, combine


def _trace_distance(circuit1, circuit2):
    """Return the trace distance of the two input circuits."""
    op1, op2 = Operator(circuit1), Operator(circuit2)
    return 0.5 * np.trace(scipy.linalg.sqrtm(np.conj(op1 - op2).T.dot(op1 - op2))).real


def is_so3_matrix(array: np.ndarray) -> bool:
    """Check if the input array is a SO(3) matrix."""
    if array.shape != (3, 3):
        return False

    if abs(np.linalg.det(array) - 1.0) > 1e-10:
        return False

    if False in np.isreal(array):
        return False

    return True


@ddt
class TestSolovayKitaev(QiskitTestCase):
    """Test the Solovay Kitaev algorithm and transformation pass."""

    def setUp(self):
        super().setUp()

        with self.assertWarnsRegex(
            DeprecationWarning,
            r".* is deprecated as of Qiskit 2\.3\. .* Use the SolovayKitaevDecomposition class",
        ):
            self.basic_approx = generate_basic_approximations([HGate(), TGate(), TdgGate()], 3)
        self.default_sk = SolovayKitaev()

    @data("default", "sk")
    def test_unitary_synthesis(self, method):
        """Test the unitary synthesis transpiler pass with Clifford+T basis set."""
        circuit = QuantumCircuit(2)
        circuit.rx(0.8, 0)
        circuit.cx(0, 1)
        circuit.x(1)

        _1q = Collect1qRuns()
        _cons = ConsolidateBlocks()
        _synth = UnitarySynthesis(["h", "t", "tdg"], method=method)
        passes = PassManager([_1q, _cons, _synth])
        compiled = passes.run(circuit)

        diff = _trace_distance(circuit, compiled)
        self.assertLess(diff, 1e-4)
        self.assertEqual(set(compiled.count_ops().keys()), {"h", "t", "tdg", "cx"})

    def test_plugin(self):
        """Test calling the plugin directly."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        unitary = Operator(circuit).data

        plugin = SolovayKitaevSynthesis()
        out = plugin.run(unitary, basis_gates=["h", "s"], depth=10, recursion_degree=3)

        self.assertTrue(set(out.count_ops().keys()).issubset({"h", "s", "sdg", "cx"}))

    def test_multiple_plugins(self):
        """Test calling the plugins directly but with different instances of basis set."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)
        unitary = Operator(circuit).data
        plugin = SolovayKitaevSynthesis()

        # First, use the ["h", "t", "tdg"] set
        out = plugin.run(unitary, basis_gates=["h", "t", "tdg"])
        self.assertTrue(set(out.count_ops().keys()).issubset({"h", "t", "tdg"}))

        # Second, use the ["h", "s"] set
        out = plugin.run(unitary, basis_gates=["h", "s", "sdg"])
        self.assertTrue(set(out.count_ops().keys()).issubset({"h", "s", "sdg"}))

        # Third, use the ["h", "t", "tdg"] set again
        out = plugin.run(unitary, basis_gates=["h", "t", "tdg"])
        self.assertTrue(set(out.count_ops().keys()).issubset({"h", "t", "tdg"}))

    def test_i_returns_empty_circuit(self):
        """Test that ``SolovayKitaev`` returns an empty circuit when
        it approximates the I-gate."""
        circuit = QuantumCircuit(1)
        circuit.id(0)

        decomposed_circuit = self.default_sk(circuit)
        self.assertEqual(QuantumCircuit(1), decomposed_circuit)

    def test_exact_decomposition_acts_trivially(self):
        """Test that the a circuit that can be represented exactly is represented exactly."""
        circuit = QuantumCircuit(1)
        circuit.t(0)
        circuit.h(0)
        circuit.tdg(0)

        sk = SolovayKitaev(3, basis_gates=["h", "t", "tdg"], depth=12)

        dag = circuit_to_dag(circuit)
        decomposed_dag = sk.run(dag)
        decomposed_circuit = dag_to_circuit(decomposed_dag)
        self.assertEqual(circuit, decomposed_circuit)

    def test_str_basis_gates(self):
        """Test specifying the basis gates by string works."""
        circuit = QuantumCircuit(1)
        circuit.rx(0.8, 0)

        with self.assertWarnsRegex(
            DeprecationWarning,
            r".* is deprecated as of Qiskit 2\.3\. .* Use the SolovayKitaevDecomposition class",
        ):
            basic_approx = generate_basic_approximations(["h", "t", "s"], 3)

        with self.assertWarnsRegex(
            DeprecationWarning,
            r"basic_approximations in the binary format produced by .* deprecated since Qiskit 2.3 ",
        ):
            sk = SolovayKitaev(3, basic_approx)

        dag = circuit_to_dag(circuit)
        discretized = dag_to_circuit(sk.run(dag))

        reference = QuantumCircuit(1, global_phase=15 * np.pi / 8)
        reference.h(0)
        reference.t(0)
        reference.h(0)

        # Make sure that the discretized circuit gives a valid approximation
        diff = _trace_distance(circuit, discretized)
        self.assertLess(diff, 0.01)
        self.assertEqual(discretized, reference)

    def test_approximation_on_qft(self):
        """Test the Solovay-Kitaev decomposition on the QFT circuit."""
        qft = QuantumCircuit(3)
        qft.append(QFTGate(3), [0, 1, 2], [])
        transpiled = transpile(
            qft, basis_gates=["u1", "u2", "u3", "u", "cx", "id"], optimization_level=1
        )

        skd = SolovayKitaev(5)

        with self.subTest("1 recursion"):
            discretized = skd(transpiled)
            self.assertLess(_trace_distance(transpiled, discretized), 15)

        skd.recursion_degree = 5
        with self.subTest("2 recursions"):
            discretized = skd(transpiled)
            self.assertLess(_trace_distance(transpiled, discretized), 7)

    def test_u_gates_work(self):
        """Test SK works on Qiskit's UGate.

        Regression test of Qiskit/qiskit-terra#9437.
        """
        circuit = QuantumCircuit(1)
        circuit.u(np.pi / 2, -np.pi, -np.pi, 0)
        circuit.u(np.pi / 2, np.pi / 2, -np.pi, 0)
        circuit.u(-np.pi / 4, 0, -np.pi / 2, 0)
        circuit.u(np.pi / 4, -np.pi / 16, 0, 0)
        circuit.u(0, 0, np.pi / 16, 0)
        circuit.u(0, np.pi / 4, np.pi / 4, 0)
        circuit.u(np.pi / 2, 0, -15 * np.pi / 16, 0)
        circuit.p(-np.pi / 4, 0)
        circuit.p(np.pi / 4, 0)
        circuit.u(np.pi / 2, 0, -3 * np.pi / 4, 0)
        circuit.u(0, 0, -np.pi / 16, 0)
        circuit.u(np.pi / 2, 0, 15 * np.pi / 16, 0)

        depth = 4
        basis_gates = ["h", "t", "tdg", "s", "sdg", "z"]
        with self.assertWarnsRegex(
            DeprecationWarning,
            r".* is deprecated as of Qiskit 2\.3\. .* Use the SolovayKitaevDecomposition class",
        ):
            gate_approx_library = generate_basic_approximations(
                basis_gates=basis_gates, depth=depth
            )

        with self.assertWarnsRegex(
            DeprecationWarning,
            r"basic_approximations in the binary format produced by .* deprecated since Qiskit 2.3 ",
        ):
            skd = SolovayKitaev(recursion_degree=2, basic_approximations=gate_approx_library)
        discretized = skd(circuit)

        included_gates = set(discretized.count_ops().keys())
        self.assertTrue(included_gates.issubset(basis_gates))

    def test_load_from_file(self):
        """Test loading basic approximations from a file works.

        Regression test of Qiskit/qiskit#12576.
        """
        filename = "approximations.bin"

        with tempfile.TemporaryDirectory() as tmp_dir:
            fullpath = os.path.join(tmp_dir, filename)

            # dump approximations to file
            with self.assertWarnsRegex(
                DeprecationWarning,
                r".* is deprecated as of Qiskit 2\.3\. .* Use the SolovayKitaevDecomposition class",
            ):
                gate_approx_library = generate_basic_approximations(
                    basis_gates=["h", "s", "sdg"], depth=3, filename=fullpath
                )

            # circuit to decompose and reference decomp
            circuit = QuantumCircuit(1)
            circuit.rx(0.8, 0)

            # Run SK pass using gate_approx_library
            with self.assertWarnsRegex(
                DeprecationWarning,
                r"basic_approximations in the binary format produced by .* deprecated since Qiskit 2.3 ",
            ):
                reference = SolovayKitaev(basic_approximations=gate_approx_library)(circuit)

            # Run SK pass using stored basis_approximations
            discretized = SolovayKitaev(basic_approximations=fullpath)(circuit)

        # Check that both flows produce the same result
        self.assertEqual(discretized, reference)

    def test_load_legacy_format(self):
        """Test loading basic approximations from a legacy npy format."""
        filename = "approximations.npy"

        # build a minimal set of basic approximations, which is stored in the format
        # {gate_labels: (matrix_so3, phase)}
        approximations = {}
        gates = {gate.name: gate.to_matrix() for gate in [HGate(), TGate()]}
        for label1, matrix1 in gates.items():
            # single gates
            su2, phase = _convert_u2_to_su2(matrix1)
            so3 = _convert_su2_to_so3(su2)
            approximations[label1] = (so3, phase)

            for label2, matrix2 in gates.items():
                # gate products
                su2, phase = _convert_u2_to_su2(matrix2.dot(matrix1))
                so3 = _convert_su2_to_so3(su2)
                approximations[f"{label1}  {label2}"] = so3

        with tempfile.TemporaryDirectory() as tmp_dir:
            fullpath = os.path.join(tmp_dir, filename)

            # store in the legacy NPY format
            np.save(fullpath, approximations)

            # circuit to decompose and reference decomp
            circuit = QuantumCircuit(1)
            circuit.rx(0.8, 0)

            # Run SK pass using gate_approx_library
            with self.assertWarnsRegex(
                DeprecationWarning,
                r"basic_approximations in the binary format produced by .* deprecated since Qiskit 2.3 ",
            ):
                reference = SolovayKitaev(basic_approximations=approximations)(circuit)

            # Run SK pass using stored basis_approximations
            with self.assertWarnsRegex(
                DeprecationWarning,
                r"basic_approximations in the binary format produced by .* deprecated since Qiskit 2.3 ",
            ):
                discretized = SolovayKitaev(basic_approximations=fullpath)(circuit)

        # Check that both flows produce the same result
        self.assertEqual(discretized, reference)

    def test_measure(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with measure operators."""
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        transpiled = self.default_sk(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "measure"})

    def test_barrier(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with barriers."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier(0)
        transpiled = self.default_sk(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "barrier"})

    def test_parameterized_gates(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with parameterized gates."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.rz(Parameter("t"), 0)
        transpiled = self.default_sk(qc)
        self.assertEqual(set(transpiled.count_ops()), {"h", "t", "rz"})

    def test_control_flow_if(self):
        """Test the Solovay-Kitaev transpiler pass on circuits with control flow ops"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)

        with qc.if_test((cr[0], 0)) as else_:
            qc.y(0)
        with else_:
            qc.z(0)
        transpiled = self.default_sk(qc)

        # check that we still have an if-else block and all the operations within
        # have been recursively synthesized
        self.assertEqual(transpiled[0].name, "if_else")
        for block in transpiled[0].operation.blocks:
            self.assertTrue(set(block.count_ops()).issubset({"h", "t", "tdg"}))

    def test_no_to_matrix(self):
        """Test the Solovay-Kitaev transpiler pass ignores gates without to_matrix."""
        qc = QuantumCircuit(1)
        qc.initialize("0")

        transpiled = self.default_sk(qc)
        self.assertEqual(set(transpiled.count_ops()), {"initialize"})

    def test_y_gate(self):
        """Test the Solovay-Kitaev decomposition on the circuit with a Y-gate (see issue #9552)."""
        circuit = QuantumCircuit(1)
        circuit.y(0)

        transpiled = self.default_sk(circuit)
        diff = _trace_distance(circuit, transpiled)
        self.assertLess(diff, 1e-6)

    def test_ppr(self):
        """Test a circuit with a Pauli-product rotation."""
        ppr = PauliProductRotationGate(Pauli("X"), 0.2)
        circuit = QuantumCircuit(1)
        circuit.append(ppr, [0])

        transpiled = self.default_sk(circuit)
        with self.subTest(msg="test gate set"):
            self.assertEqual(set(transpiled.count_ops()), {"h", "t", "tdg"})

        with self.subTest(msg="test approximation"):
            diff = _trace_distance(circuit, transpiled)
            self.assertLess(diff, 1e-5)

    @data(["unitary"], ["rz"])
    def test_sk_synth_gates_to_basis(self, synth_gates):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        unitary = QuantumCircuit(1)
        unitary.h(0)
        unitary_op = Operator(unitary)

        qc = QuantumCircuit(1)
        qc.unitary(unitary_op, 0)
        qc.rz(0.1, 0)
        dag = circuit_to_dag(qc)

        basis_gates = ["h", "t", "tdg"]
        out = UnitarySynthesis(basis_gates=basis_gates, synth_gates=synth_gates, method="sk").run(
            dag
        )
        self.assertTrue(set(out.count_ops()).isdisjoint(synth_gates))

    def test_generate_basis_approximation_gates(self):
        """Test the basis approximation generation works for all supported gates.

        Regression test of Qiskit/qiskit-terra#9585.
        """
        basis = ["i", "x", "y", "z", "h", "t", "tdg", "s", "sdg", "sx", "sxdg"]
        with self.assertWarnsRegex(
            DeprecationWarning,
            r".* is deprecated as of Qiskit 2\.3\. .* Use the SolovayKitaevDecomposition class",
        ):
            approx = generate_basic_approximations(basis, depth=2)

        # This mainly checks that there are no errors in the generation (like
        # in computing the inverse as described in #9585), so a simple check is enough.
        self.assertGreater(len(approx), len(basis))


@ddt
class TestSolovayKitaevDecomposition(QiskitTestCase):
    """Tests for the underlying SK decomposition class."""

    def setUp(self):
        super().setUp()
        self.default_sk = SolovayKitaevDecomposition()

    @combine(use_matrix=[True, False], gate_cls=[RXGate, RYGate, RZGate])
    def test_approximation(self, use_matrix, gate_cls):
        """Test the approximation works."""
        recursion = 4
        for angle in np.pi / np.arange(1, 10, 2):
            with self.subTest(angle=angle):
                gate = gate_cls(angle)
                if use_matrix:
                    gate = gate.to_matrix()
                synth = self.default_sk.run(gate, recursion)

                error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                self.assertLess(error, 2e-3)

    def test_query_basic_approximation(self):
        """Test finding the basic approximations for some gates."""
        # exactly representable for depth of 12
        for gate_cls in [RXGate, RYGate, RZGate]:
            for angle in [np.pi, np.pi / 2, np.pi / 4]:
                with self.subTest(gate_cls=gate_cls, angle=angle):
                    gate = gate_cls(angle)
                    synth = self.default_sk.query_basic_approximation(gate)
                    error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                    self.assertLess(error, 1e-10)

        # approximately representable
        for gate_cls in [RXGate, RYGate, RZGate]:
            for angle in [np.pi / 8, np.pi / 10, np.pi / 20]:
                with self.subTest(gate_cls=gate_cls, angle=angle):
                    gate = gate_cls(angle)
                    synth = self.default_sk.query_basic_approximation(gate)
                    error = np.linalg.norm(Operator(gate).data - Operator(synth).data)
                    self.assertLess(error, 0.5)

    @data(True, False)
    def test_basis_gates(self, use_str):
        """Test specifying the basis gates by string or gate works."""
        if use_str:
            basis_gates = ["h", "s", "sdg"]
        else:
            basis_gates = [HGate(), SGate(), SdgGate()]

        sk = SolovayKitaevDecomposition(basis_gates=basis_gates, depth=3)

        synth = sk.run(RXGate(np.pi / 2), 3)

        expected = QuantumCircuit(1, global_phase=7 * np.pi / 4)
        expected.h(0)
        expected.s(0)
        expected.h(0)

        self.assertEqual(synth, expected)

    @data("cx", CXGate())
    def test_rejects_multi_qubit_basis_gate(self, gate):
        """Test that multi-qubit basis gates are rejected before calling into Rust."""
        with self.assertRaisesRegex(
            ValueError,
            "Solovay-Kitaev synthesis only supports single-qubit basis gates, "
            "but 'cx' acts on 2 qubits",
        ):
            SolovayKitaevDecomposition(basis_gates=["h", "t", "tdg", gate], depth=1)


def _convert_u2_to_su2(u2_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Convert a U(2) matrix to SU(2) by adding a global phase."""
    z = 1 / np.sqrt(np.linalg.det(u2_matrix))
    su2_matrix = z * u2_matrix
    phase = np.arctan2(np.imag(z), np.real(z))

    return su2_matrix, phase


def _convert_su2_to_so3(matrix: np.ndarray) -> np.ndarray:
    """Computes SO(3)-matrix from input SU(2)-matrix.

    Args:
        matrix: The SU(2)-matrix for which a corresponding SO(3)-matrix needs to be computed.

    Returns:
        The SO(3)-matrix corresponding to ``matrix``.

    Raises:
        ValueError: if ``matrix`` is not an SU(2)-matrix.
    """
    matrix = matrix.astype(complex)
    a = np.real(matrix[0][0])
    b = np.imag(matrix[0][0])
    c = -np.real(matrix[0][1])
    d = -np.imag(matrix[0][1])
    rotation = np.array(
        [
            [a**2 - b**2 - c**2 + d**2, 2 * a * b + 2 * c * d, -2 * a * c + 2 * b * d],
            [-2 * a * b + 2 * c * d, a**2 - b**2 + c**2 - d**2, 2 * a * d + 2 * b * c],
            [2 * a * c + 2 * b * d, 2 * b * c - 2 * a * d, a**2 + b**2 - c**2 - d**2],
        ],
        dtype=float,
    )
    return rotation


if __name__ == "__main__":
    unittest.main()
