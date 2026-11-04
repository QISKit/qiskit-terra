# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A Test for the Pauli product rotation gate."""

import io
import numpy as np
import scipy as sc

from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, Parameter, CircuitError, CommutationChecker
from qiskit.circuit.library import (
    RXGate,
    PauliProductRotationGate,
    PauliEvolutionGate,
    PauliProductMeasurement,
)
from qiskit.quantum_info import Pauli, Operator, SparsePauliOp
from qiskit.qpy import dump, load
from qiskit.compiler import transpile
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@ddt
class TestPauliProductRotationGate(QiskitTestCase):
    """Tests for the PauliProductRotationGate."""

    def test_simple(self):
        """Test a simple rotation gate."""
        pauli = Pauli("XIZZY")
        angle = 0.23
        rotation = PauliProductRotationGate(pauli, angle)

        expected = QuantumCircuit(pauli.num_qubits)
        expected.h(4)
        expected.sx(0)
        expected.cx(4, 2)
        expected.cx(2, 1)
        expected.cx(1, 0)
        expected.rz(angle, 0)
        expected.cx(1, 0)
        expected.cx(2, 1)
        expected.cx(4, 2)
        expected.sxdg(0)
        expected.h(4)

        with self.subTest(msg="decomposition"):
            self.assertEqual(expected, rotation.definition)

        with self.subTest(msg="matrix"):
            np.testing.assert_allclose(rotation.to_matrix(), Operator(rotation.definition).data)

    @data(1.2, np.pi / 4, 1, -1, 0, Parameter("a"), Parameter("a") + Parameter("b"))
    def test_append_to_circuit(self, angle):
        """Test that appending a Pauli product rotation to a circuit does not panic."""
        pauli = Pauli("XIZZY")
        rotation = PauliProductRotationGate(pauli, angle)
        qc = QuantumCircuit(5)
        qc.append(rotation, [0, 1, 2, 3, 4])

    def test_equality(self):
        """Test some equalities."""
        x = Parameter("x")
        self.assertEqual(
            PauliProductRotationGate(Pauli("X"), 0.1), PauliProductRotationGate(Pauli("X"), 0.1)
        )
        self.assertEqual(
            PauliProductRotationGate(Pauli("X"), x), PauliProductRotationGate(Pauli("X"), x)
        )
        self.assertEqual(
            PauliProductRotationGate(Pauli("X"), -x), PauliProductRotationGate(Pauli("-X"), x)
        )

        self.assertNotEqual(
            PauliProductRotationGate(Pauli("X"), 0.1), PauliProductRotationGate(Pauli("X"), 0.2)
        )
        self.assertNotEqual(
            PauliProductRotationGate(Pauli("Y"), 0.1), PauliProductRotationGate(Pauli("X"), 0.1)
        )

        qc1 = QuantumCircuit(2)
        qc1.append(PauliProductRotationGate(Pauli("XX"), angle=1.2), [0, 1])

        qc2 = QuantumCircuit(2)
        qc2.append(PauliProductRotationGate(Pauli("XZ"), angle=1.2), [0, 1])
        self.assertNotEqual(qc1, qc2)

        qc3 = QuantumCircuit(2)
        qc3.append(PauliProductRotationGate(Pauli("XZ"), angle=1.2), [1, 0])
        self.assertNotEqual(qc2, qc3)

    @data("iX", "-iX")
    def test_invalid_phase(self, pauli):
        """Test invalid Pauli phases raises an error."""
        with self.assertRaises(CircuitError):
            _ = PauliProductRotationGate(Pauli(pauli), 1.0)

    def test_commutation_checks(self):
        """Test commutative optimization handles the rotation gate."""
        cc = CommutationChecker()
        xx = PauliProductRotationGate(Pauli("XX"), 0.1)
        yy = PauliProductRotationGate(Pauli("YY"), -0.5)
        self.assertTrue(cc.commute(xx, [0, 1], [], yy, [0, 1], []))
        self.assertTrue(cc.commute(xx, [1, 0], [], yy, [0, 1], []))
        self.assertFalse(cc.commute(xx, [2, 0], [], yy, [0, 1], []))

        zzm = PauliProductMeasurement(Pauli("ZZ"))
        self.assertTrue(cc.commute(zzm, [0, 1], [0], xx, [1, 0], []))
        self.assertFalse(cc.commute(zzm, [0, 2], [0], xx, [1, 0], []))

        summed = SparsePauliOp(["XXI", "IXI", "IIZ"])
        evo = PauliEvolutionGate(summed, time=42.2)
        self.assertTrue(cc.commute(evo, [0, 1, 2], [], xx, [1, 2], []))
        self.assertFalse(cc.commute(evo, [0, 1, 2], [], xx, [0, 1], []))
        self.assertFalse(cc.commute(evo, [0, 1, 2], [], yy, [0, 1], []))

    def test_pauli_evolution_equivalence(self):
        """Test consistency with the PauliEvolutionGate."""
        pauli = Pauli("XIYIZ")
        angle = Parameter("theta")
        evo = PauliEvolutionGate(pauli, angle / 2)
        rotation = PauliProductRotationGate(pauli, angle)

        self.assertEqual(evo.definition, rotation.definition)

    def test_python_rust_equivalence(self):
        """Test that the Python definition and Rust HLS definition match."""
        pauli = Pauli("IXYZ")
        angle = Parameter("theta")
        circuit = QuantumCircuit(pauli.num_qubits)
        circuit.append(PauliProductRotationGate(pauli, angle), circuit.qubits)

        # trigger Rust-path, HLS decomposition
        basis_gates = ["h", "sx", "sxdg", "rz", "cx"]
        rust_path = transpile(circuit, basis_gates=basis_gates, optimization_level=0)

        # compare to Python-side decomposition
        python_path = circuit.decompose()

        self.assertEqual(rust_path, python_path)

    def test_qpy(self):
        """Test qpy for circuits with PauliProductRotationGates."""
        qc = QuantumCircuit(6, 2)
        x = Parameter("x")
        qc.append(PauliProductRotationGate(Pauli("XZ"), 0.2), [4, 1])
        qc.append(PauliProductRotationGate(Pauli("Z"), -12, label="wohooo rotation"), [2])
        qc.append(PauliProductRotationGate(Pauli("ZZ"), x), [3, 2])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_pauli_accessor(self):
        """Check that ``pauli()`` returns the original Pauli."""
        pauli = Pauli("XIYZ")
        rotation = PauliProductRotationGate(pauli, 0.0)
        self.assertEqual(rotation.pauli(), pauli)

    def test_draw(self):
        """Test drawing a Pauli rotation gate circuit."""
        qc = QuantumCircuit(3)
        qc.append(PauliProductRotationGate(Pauli("X"), 0.1), [0])
        qc.append(PauliProductRotationGate(Pauli("YY"), np.pi / 2), [1, 2])
        qc.append(PauliProductRotationGate(Pauli("ZZZ"), -np.pi / 4), [0, 1, 2])
        out = str(qc.draw("text"))
        expected = "\n".join(
            [
                "      ┌──────────┐ ┌──────────────┐",
                "q_0: ─┤ R_X(0.1) ├─┤0             ├",
                "     ┌┴──────────┴┐│              │",
                "q_1: ┤0           ├┤1 R_ZZZ(-π/4) ├",
                "     │  R_YY(π/2) ││              │",
                "q_2: ┤1           ├┤2             ├",
                "     └────────────┘└──────────────┘",
            ]
        )
        self.assertEqual(expected, out)

    def test_inverse(self):
        """Test the inverse is correct."""
        ppr = PauliProductRotationGate(Pauli("XIYZ"), 1.2345)
        ppr_dg = ppr.inverse()

        with self.subTest(msg="inverse type"):
            self.assertIsInstance(ppr_dg, PauliProductRotationGate)

        with self.subTest(msg="inverse"):
            iden = Operator(ppr).data.dot(Operator(ppr_dg))
            self.assertTrue(np.allclose(iden, np.eye(16)))

    @data(True, False)
    def test_to_matrix(self, neg):
        """Check conversion to matrix."""
        angle = -2.13
        pauli = Pauli("-X") if neg else Pauli("XIYZ")
        ppr = PauliProductRotationGate(pauli, angle)
        expected = sc.linalg.expm(-0.5j * angle * pauli.to_matrix())
        self.assertTrue(np.allclose(expected, ppr.to_matrix()))

    def test_matrix_conventions(self):
        """Check the matrix conventions."""
        angle = -2.13
        pauli = Pauli("X")
        ppr = PauliProductRotationGate(pauli, angle)

        with self.subTest(msg="PPR and standard gates"):
            rx = RXGate(angle)
            self.assertTrue(np.allclose(rx.to_matrix(), ppr.to_matrix()))

        with self.subTest(msg="PPR and Pauli evolution"):
            evo = PauliEvolutionGate(pauli, time=angle / 2)
            self.assertTrue(np.allclose(evo.to_matrix(), ppr.to_matrix()))

    @data(0, 1, 2)
    def test_control(self, num_ctrl_qubits):
        """Check calling the control method."""
        angle = 5.4321
        pauli = Pauli("XYZ")
        ppr = PauliProductRotationGate(pauli, angle)

        ctrl = ppr.control(num_ctrl_qubits)

        if num_ctrl_qubits == 0:
            expected = ppr.copy()
            self.assertEqual(expected, ctrl)

        else:
            expected = PauliEvolutionGate(pauli, time=angle / 2).control(num_ctrl_qubits)
            # We don't want to hardcode a check that PPR.control returns a PauliEvolutionGate,
            # we want to ensure that the output is an efficient controlled version.
            self.assertTrue(np.allclose(Operator(expected).data, Operator(ctrl).data))

            counts = {"sx": 1, "sxdg": 1, "h": 2, "cx": 4, num_ctrl_qubits * "c" + "rz": 1}
            self.assertDictEqual(counts, ctrl.definition.count_ops())

    def test_parameter_assignment(self):
        """Test parameter assignment."""
        angle = Parameter("x")
        pauli = Pauli("XYZ")
        ppr = PauliProductRotationGate(pauli, angle)

        value = 5.4321
        circuit = QuantumCircuit(pauli.num_qubits)
        circuit.append(ppr, circuit.qubits)
        self.assertEqual(circuit.num_parameters, 1)

        bound = circuit.assign_parameters({angle: value})
        self.assertEqual(bound.num_parameters, 0)

        params = bound.data[0].operation.params
        self.assertEqual(len(params), 1)
        self.assertAlmostEqual(value, params[0])

        expected = QuantumCircuit(pauli.num_qubits)
        expected.append(PauliProductRotationGate(pauli, value), expected.qubits)
        self.assertEqual(expected, bound)
