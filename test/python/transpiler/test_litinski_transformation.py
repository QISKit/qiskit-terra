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

"""Test Litinski transformation pass"""

import numpy as np

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    HGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    SwapGate,
    iSwapGate,
    ECRGate,
    QFTGate,
    PauliEvolutionGate,
    PauliProductMeasurement,
    PauliProductRotationGate,
    U1Gate,
)
from qiskit.circuit.random import random_clifford_circuit
from qiskit.compiler import transpile
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import LitinskiTransformation
from qiskit.quantum_info import Operator, Pauli, random_pauli
from test import QiskitTestCase, combine


@ddt
class TestLitinskiTransformation(QiskitTestCase):
    """Test the Litinski Transformation pass."""

    def test_default(self):
        """Test the default behavior for backward compat."""
        angle = 0.1
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.rz(angle, 0)

        default = LitinskiTransformation()
        qct = default(qc)

        expected = QuantumCircuit(1)
        expected.append(PauliEvolutionGate(Pauli("X"), angle / 2), [0])
        expected.h(0)

        self.assertEqual(expected, qct)

    @data(True, False)
    def test_t_tdg_gates(self, use_ppr):
        """Test circuit with T/Tdg gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.t(0)
        qc.cx(0, 2)
        qc.t(1)
        qc.tdg(0)
        qc.s(2)
        qc.t(2)

        qct = LitinskiTransformation(use_ppr=use_ppr)(qc)
        ppr_name = "pauli_product_rotation" if use_ppr else "PauliEvolution"

        self.assertEqual(qct.count_ops(), {ppr_name: 4, "cx": 2, "h": 1, "s": 1})
        self.assertEqual(Operator(qct), Operator(qc))

    @data(True, False)
    def test_rz_gates(self, use_ppr):
        """Test circuit with RZ-rotation gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cx(0, 2)
        qc.rz(-0.4, 1)
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation(use_ppr=use_ppr)(qc)
        ppr_name = "pauli_product_rotation" if use_ppr else "PauliEvolution"

        self.assertEqual(qct.count_ops(), {ppr_name: 3, "cx": 2, "h": 1, "s": 1})
        self.assertEqual(Operator(qct), Operator(qc))

    @data(True, False)
    def test_omit_clifford_gates(self, use_ppr):
        """Test fix_clifford."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cx(0, 2)
        qc.rz(-0.4, 1)
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation(fix_clifford=False, use_ppr=use_ppr)(qc)
        ppr_name = "pauli_product_rotation" if use_ppr else "PauliEvolution"

        self.assertEqual(qct.count_ops(), {ppr_name: 3})

    @data(True, False)
    def test_parametric_rz_rx_ry_gates(self, use_ppr):
        """Test circuit with parameterized RZ-rotation gates."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")
        gamma = Parameter("gamma")

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(alpha, 0)
        qc.cx(0, 2)
        qc.rx(beta, 1)
        qc.s(2)
        qc.rz(0.1, 1)
        qc.ry(gamma, 1)

        qct = LitinskiTransformation(use_ppr=use_ppr)(qc)
        ppr_name = "pauli_product_rotation" if use_ppr else "PauliEvolution"
        self.assertEqual(qct.count_ops(), {ppr_name: 4, "cx": 2, "h": 1, "s": 1})

        qc_bound = qc.assign_parameters([0.123, -1.234, 0.567])
        qct_bound = qct.assign_parameters([0.123, -1.234, 0.567])
        self.assertEqual(Operator(qct_bound), Operator(qc_bound))

    def test_parametric_ppr_gates(self):
        """Test circuit with parameterized PPR rotation gates."""
        alpha = Parameter("alpha")
        beta = Parameter("beta")

        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(PauliProductRotationGate(Pauli("ZXY"), alpha), [0, 1, 3])
        qc.cx(0, 2)
        qc.append(PauliProductRotationGate(Pauli("YIX"), beta), [0, 2, 3])
        qc.s(2)
        qc.rz(0.1, 1)

        qct = LitinskiTransformation(use_ppr=True)(qc)
        self.assertEqual(qct.count_ops(), {"pauli_product_rotation": 3, "cx": 2, "h": 1, "s": 1})

        qc_bound = qc.assign_parameters([0.123, -1.567])
        qct_bound = qct.assign_parameters([0.123, -1.567])
        self.assertEqual(Operator(qct_bound), Operator(qc_bound))

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_qft_circuits(self, num_qubits):
        """Test more complex circuits produced by transpiling QFT gates into [cx, sx, rz] basis."""
        qc = QuantumCircuit(num_qubits)
        qc.append(QFTGate(num_qubits), range(num_qubits))

        # transpile the circuit into ["cx", "rz", "sx"] so that Litinski's transform can be applied
        qc = transpile(qc, basis_gates=["cx", "rz", "sx"])

        # apply Litinski's transform
        qc_litinski = LitinskiTransformation(use_ppr=True)(qc)
        qc_litinski_no_fix = LitinskiTransformation(fix_clifford=False, use_ppr=True)(qc)

        # make sure the transform was applied
        self.assertEqual(list(qc_litinski_no_fix.count_ops()), ["pauli_product_rotation"])
        self.assertEqual(
            qc_litinski_no_fix.count_ops()["pauli_product_rotation"],
            qc_litinski.count_ops()["pauli_product_rotation"],
        )

        # make sure the result is correct
        self.assertEqual(Operator(qc_litinski), Operator(qc))

    def test_all_supported_clifford_gates(self):
        """Test circuit with all of the supported clifford gates."""

        qc = QuantumCircuit(4)

        # Put all possible Clifford gates at the front of the circuit,
        # so that the algorithm will need to combine these into a Clifford,
        # and commute through the rotation gates.
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.s(0)
        qc.sdg(1)
        qc.cz(0, 2)
        qc.cz(1, 3)
        qc.z(0)
        qc.h(1)
        qc.x(2)
        qc.y(3)
        qc.swap(0, 1)
        qc.sx(0)
        qc.sxdg(1)
        qc.cy(1, 2)
        qc.id(3)
        qc.ecr(0, 3)
        qc.iswap(1, 2)
        qc.dcx(1, 3)

        # Rotations
        qc.t(0)
        qc.rz(0.1, 1)
        qc.tdg(2)
        qc.rz(-0.2, 3)
        qc.p(0.3, 0)
        qc.rx(0.4, 2)
        qc.ry(-0.5, 1)
        qc.append(U1Gate(-0.5), [0])

        qc_litinski = LitinskiTransformation()(qc)
        ops_litinski = qc_litinski.count_ops()

        # make sure the transform was applied
        for z_rot in ["t", "tdg", "rz", "rx", "ry", "p", "u1"]:
            self.assertNotIn(z_rot, ops_litinski)

        # make sure the result is correct
        self.assertEqual(Operator(qc_litinski), Operator(qc))

    def test_rotation_clifford_gates(self):
        """Test circuit with rotation gates with multiple pi/2 angles which are clifford gates."""

        qc = QuantumCircuit(4)

        # Cliffords
        qc.h(0)
        qc.cx(0, 1)
        qc.sx(2)
        qc.cx(1, 2)
        qc.sxdg(1)
        qc.cz(2, 3)
        qc.s(0)
        qc.sdg(1)

        # Rotations with pi/2 angles
        qc.rz(np.pi / 2, 1)
        qc.rz(-np.pi, 3)
        qc.p(3 * np.pi / 2, 0)
        qc.rx(-5 * np.pi / 2, 2)
        qc.ry(7 * np.pi / 2, 1)
        qc.append(U1Gate(2 * np.pi), [0])

        # circuit is not changed by the pass (since all gates are cliffords)
        qc_litinski = LitinskiTransformation()(qc)

        # make sure the result is correct
        self.assertEqual(qc_litinski, qc)

    def test_random_circuits(self):
        """Test on random Clifford+T circuits."""

        for trial in range(10):
            start_seed = 1234 + 10 * trial

            # create a circuit with multiple layers of Clifford and T/Tdg gates
            qc = QuantumCircuit(5)
            for layer in range(5):
                clifford_circuit = random_clifford_circuit(
                    num_qubits=5, num_gates=20, gates="all", seed=start_seed + layer
                )
                qc.compose(clifford_circuit, inplace=True)
                qc.t(0)
                qc.tdg(1)

            # apply the transform
            qc_litinski = LitinskiTransformation()(qc)
            ops_litinski = qc_litinski.count_ops()

            # make sure the transform was applied
            self.assertNotIn("t", ops_litinski)
            self.assertNotIn("tdg", ops_litinski)

            # make sure the result is correct
            self.assertEqual(Operator(qc_litinski), Operator(qc))

    def test_raises_on_unsupported_gates(self):
        """Test that the pass returns an error when it runs on unsupported gates."""
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(0.1, 0)
        qc.cp(0.1, 0, 1)  # unsupported
        qc.cx(0, 2)

        with self.assertRaises(TranspilerError):
            _ = LitinskiTransformation()(qc)

    def test_t(self):
        """Test the transform on a circuit with a T-gate."""
        qc = QuantumCircuit(2)
        qc.t(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])

        self.assertEqual(qct, expected)

    def test_tdg(self):
        """Test the transform on a circuit with a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.tdg(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("Z"), -np.pi / 4), [0])

        self.assertEqual(qct, expected)

    def test_p(self):
        """Test the phase gate, ensuring we got the global phase right."""
        angle = 0.231
        qc = QuantumCircuit(1)
        qc.p(angle, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)
        self.assertTrue(np.allclose(Operator(qc).data, Operator(qct).data))

        expected = QuantumCircuit(1, global_phase=angle / 2)
        expected.append(PauliProductRotationGate(Pauli("Z"), angle), [0])
        self.assertEqual(qct, expected)

    def test_h_t(self):
        """Test the transform on a circuit with an H-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.t(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("X"), np.pi / 4), [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_h_tdg(self):
        """Test the transform on a circuit with an H-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.tdg(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("X"), -np.pi / 4), [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_sx_t(self):
        """Test the transform on a circuit with an SX-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.t(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("Y"), np.pi / 4), [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_sx_tdg(self):
        """Test the transform on a circuit with an SX-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.sx(0)
        qc.tdg(0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("Y"), -np.pi / 4), [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_cx_t(self):
        """Test the transform on a circuit with a CX-gate and a T-gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.t(1)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("ZZ"), np.pi / 4), [0, 1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_cx_tdg(self):
        """Test the transform on a circuit with a CX-gate and a Tdg-gate."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.tdg(1)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, global_phase=-np.pi / 8)
        expected.append(PauliProductRotationGate(Pauli("ZZ"), -np.pi / 4), [0, 1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_measure(self):
        """Test the transform on a circuit with a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("Z")), [0], [0])

        self.assertEqual(qct, expected)

    def test_h_measure(self):
        """Test the transform on a circuit with an H-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("X")), [0], [0])
        expected.h(0)

        self.assertEqual(qct, expected)

    def test_sx_measure(self):
        """Test the transform on a circuit with an SX-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.sx(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("Y")), [0], [0])
        expected.sx(0)

        self.assertEqual(qct, expected)

    def test_cx_measure(self):
        """Test the transform on a circuit with an CX-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure(1, 1)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("ZZ")), [0, 1], [1])
        expected.cx(0, 1)

        self.assertEqual(qct, expected)

    def test_x_measure(self):
        """Test the transform on a circuit with a X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-Z")), [0], [0])
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_h_x_measure(self):
        """Test the transform on a circuit with an H-gate, an X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-X")), [0], [0])
        expected.h(0)
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_sx_x_measure(self):
        """Test the transform on a circuit with an H-gate, an X-gate and a Z-measurement."""
        qc = QuantumCircuit(2, 2)
        qc.sx(0)
        qc.x(0)
        qc.measure(0, 0)

        qct = LitinskiTransformation(use_ppr=True)(qc)

        expected = QuantumCircuit(2, 2)
        expected.append(PauliProductMeasurement(Pauli("-Y")), [0], [0])
        expected.sx(0)
        expected.x(0)

        self.assertEqual(qct, expected)

    def test_on_circuits_with_measures(self):
        """Test the Litinski transformation pass on a more complex with Clifford gates,
        T gates and Z-measures.
        """
        # This is the example from Figure 4 in the paper "A Game of Surface Codes" by Litinski.

        # The original circuit (as shown at the top-left of the figure).
        qc = QuantumCircuit(4, 4)
        qc.t(0)
        qc.cx(2, 1)
        qc.sxdg(3)
        qc.cx(1, 0)
        qc.sx(2)
        qc.t(3)
        qc.cx(3, 0)
        qc.t(0)
        qc.s(1)
        qc.t(2)
        qc.s(3)
        qc.sxdg(0)
        qc.sx(1)
        qc.sx(2)
        qc.sx(3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.measure(3, 3)

        # Apply the Litinski transform with fix_cliffords=False (ignoring the Clifford gates
        # at the end of the transformed circuit, and clearing the global phase).
        qct = LitinskiTransformation(fix_clifford=False, use_ppr=True)(qc)
        qct.global_phase = 0

        # The transformed circuit (as shown at the bottom-right of the figure).
        expected = QuantumCircuit(4, 4)
        expected.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        expected.append(PauliProductRotationGate(Pauli("YX"), np.pi / 4), [1, 2])
        expected.append(PauliProductRotationGate(Pauli("Y"), -np.pi / 4), [3])
        expected.append(PauliProductRotationGate(Pauli("YZZZ"), -np.pi / 4), [0, 1, 2, 3])
        expected.append(PauliProductMeasurement(Pauli("YZZY")), [0, 1, 2, 3], [0])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 1], [1])
        expected.append(PauliProductMeasurement(Pauli("-Z")), [2], [2])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 3], [3])

        self.assertEqual(qct, expected)

    def test_on_circuits_with_rotations_measures(self):
        """Test the Litinski transformation pass on a more complex with Clifford gates,
        T gates and Z-measures.
        """
        # This is the example from Figure 4 in the paper "A Game of Surface Codes" by Litinski.
        # with rotation gates

        # The original circuit (as shown at the top-left of the figure).
        qc = QuantumCircuit(4, 4)
        qc.rz(np.pi / 4, 0)
        qc.cx(2, 1)
        qc.rx(-np.pi / 2, 3)
        qc.cx(1, 0)
        qc.rx(np.pi / 2, 2)
        qc.rz(np.pi / 4, 3)
        qc.cx(3, 0)
        qc.rz(np.pi / 4, 0)
        qc.rz(np.pi / 2, 1)
        qc.rz(np.pi / 4, 2)
        qc.rz(np.pi / 2, 3)
        qc.rx(-np.pi / 2, 0)
        qc.rx(np.pi / 2, 1)
        qc.rx(np.pi / 2, 2)
        qc.rx(np.pi / 2, 3)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.measure(3, 3)

        # Apply the Litinski transform with fix_cliffords=False (ignoring the Clifford gates
        # at the end of the transformed circuit).
        qct = LitinskiTransformation(fix_clifford=False, use_ppr=True)(qc)

        # The transformed circuit (as shown at the bottom-right of the figure).
        expected = QuantumCircuit(4, 4)
        expected.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        expected.append(PauliProductRotationGate(Pauli("YX"), np.pi / 4), [1, 2])
        expected.append(PauliProductRotationGate(Pauli("Y"), -np.pi / 4), [3])
        expected.append(PauliProductRotationGate(Pauli("YZZZ"), -np.pi / 4), [0, 1, 2, 3])
        expected.append(PauliProductMeasurement(Pauli("YZZY")), [0, 1, 2, 3], [0])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 1], [1])
        expected.append(PauliProductMeasurement(Pauli("-Z")), [2], [2])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 3], [3])

        self.assertEqual(qct, expected)

    def test_on_circuits_with_ppr_ppm(self):
        """Test the Litinski transformation pass on a more complex with Clifford gates,
        T gates and Z-measures.
        """
        # This is the example from Figure 4 in the paper "A Game of Surface Codes" by Litinski.
        # with PPR and PPM gates

        # The original circuit (as shown at the top-left of the figure).
        qc = QuantumCircuit(4, 4)
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        qc.cx(2, 1)
        qc.append(PauliProductRotationGate(Pauli("X"), -np.pi / 2), [3])
        qc.cx(1, 0)
        qc.append(PauliProductRotationGate(Pauli("X"), np.pi / 2), [2])
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [3])
        qc.cx(3, 0)
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 2), [1])
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [2])
        qc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 2), [3])
        qc.append(PauliProductRotationGate(Pauli("X"), -np.pi / 2), [0])
        qc.append(PauliProductRotationGate(Pauli("X"), np.pi / 2), [1])
        qc.append(PauliProductRotationGate(Pauli("X"), np.pi / 2), [2])
        qc.append(PauliProductRotationGate(Pauli("X"), np.pi / 2), [3])
        qc.append(PauliProductMeasurement(Pauli("Z")), [0], [0])
        qc.append(PauliProductMeasurement(Pauli("Z")), [1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [2])
        qc.append(PauliProductMeasurement(Pauli("Z")), [3], [3])

        # Apply the Litinski transform with fix_cliffords=False (ignoring the Clifford gates
        # at the end of the transformed circuit).
        qct = LitinskiTransformation(fix_clifford=False, use_ppr=True)(qc)

        # The transformed circuit (as shown at the bottom-right of the figure).
        expected = QuantumCircuit(4, 4)
        expected.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        expected.append(PauliProductRotationGate(Pauli("YX"), np.pi / 4), [1, 2])
        expected.append(PauliProductRotationGate(Pauli("Y"), -np.pi / 4), [3])
        expected.append(PauliProductRotationGate(Pauli("YZZZ"), -np.pi / 4), [0, 1, 2, 3])
        expected.append(PauliProductMeasurement(Pauli("YZZY")), [0, 1, 2, 3], [0])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 1], [1])
        expected.append(PauliProductMeasurement(Pauli("-Z")), [2], [2])
        expected.append(PauliProductMeasurement(Pauli("XX")), [0, 3], [3])

        self.assertEqual(qct, expected)

    @data(True, False)
    def test_barrier(self, fix_clifford):
        """Test adding a barrier."""
        circuit = QuantumCircuit(1, 1)
        circuit.h(0)
        circuit.rz(1.0, 0)
        circuit.measure(0, 0)

        transform = LitinskiTransformation(
            fix_clifford=fix_clifford, insert_barrier=True, use_ppr=True
        )
        out = transform(circuit)

        if fix_clifford:
            expected_ops = ["pauli_product_rotation", "pauli_product_measurement", "barrier", "h"]
        else:
            expected_ops = ["pauli_product_rotation", "pauli_product_measurement"]

        ops = [op.name for op in out.data]
        self.assertListEqual(expected_ops, ops)

    @combine(
        p=["X", "Y", "Z"],
        cliff_cls=[
            IGate,
            XGate,
            YGate,
            ZGate,
            SGate,
            SdgGate,
            HGate,
            SXGate,
            SXdgGate,
        ],
    )
    def test_single_qubit_evolve_clifford_gate(self, p, cliff_cls):
        """Test that LitinskiTransformation is correct for one qubit rotations RX/RY/RZ
        and all single qubit Clifford gates"""
        cliff = cliff_cls()
        circuit = QuantumCircuit(1)
        circuit.append(cliff, [0])
        circuit_in = circuit.copy()
        pauli = Pauli(p)
        pauli_ev = pauli.evolve(circuit)
        if p == "X":
            circuit_in.rx(1.0, 0)
        elif p == "Z":
            circuit_in.rz(1.0, 0)
        elif p == "Y":
            circuit_in.ry(1.0, 0)
        transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
        circuit_out = transform(circuit_in)
        ppr = PauliProductRotationGate(pauli_ev, 1.0)
        circuit_target = QuantumCircuit(1)
        circuit_target.append(ppr, [0])
        circuit_target.compose(circuit, [0], inplace=True)
        self.assertEqual(circuit_out, circuit_target)

    @data("X", "Y", "Z")
    def test_single_qubit_evolve_random_clifford(self, p):
        """Test that LitinskiTransformation is correct for one qubit rotations RX/RY/RZ
        and random single qubit Clifford circuits"""
        for seed in range(10):
            circuit = random_clifford_circuit(1, num_gates=10, seed=seed)
            circuit_in = circuit.copy()
            pauli = Pauli(p)
            pauli_ev = pauli.evolve(circuit)
            if p == "X":
                circuit_in.rx(1.0, 0)
            elif p == "Z":
                circuit_in.rz(1.0, 0)
            elif p == "Y":
                circuit_in.ry(1.0, 0)
            transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
            circuit_out = transform(circuit_in)
            ppr = PauliProductRotationGate(pauli_ev, 1.0)
            circuit_target = QuantumCircuit(1)
            circuit_target.append(ppr, [0])
            circuit_target.compose(circuit, [0], inplace=True)
            self.assertEqual(circuit_out, circuit_target)

    @combine(
        p=["X", "Y", "Z"],
        cliff_cls=[
            CXGate,
            CYGate,
            CZGate,
            DCXGate,
            SwapGate,
            iSwapGate,
            ECRGate,
        ],
        qbit=[0, 1],
    )
    def test_two_qubit_evolve_clifford_gate(self, p, cliff_cls, qbit):
        """Test that LitinskiTransformation is correct for one qubit rotations RX/RY/RZ
        and all two qubit Clifford gates"""
        cliff = cliff_cls()
        circuit = QuantumCircuit(2)
        circuit.append(cliff, [0, 1])
        circuit_in = circuit.copy()
        if qbit == 0:
            pauli = Pauli("I" + p)
        else:
            pauli = Pauli(p + "I")
        pauli_ev = pauli.evolve(circuit)
        if p == "X":
            circuit_in.rx(1.0, qbit)
        elif p == "Z":
            circuit_in.rz(1.0, qbit)
        elif p == "Y":
            circuit_in.ry(1.0, qbit)
        transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
        circuit_out = transform(circuit_in)

        # Remove "I" terms to get the same PPR
        # since PPR('IZ') is not the same gate as PPR(Z)=RZ on qubit 0
        if pauli_ev.to_label()[-2] == "I":
            pauli_z_x = Pauli(pauli_ev.to_label()[-1])
            phase = pauli_ev.phase
            pauli_ev = Pauli((pauli_z_x.z, pauli_z_x.x, phase))
            qubits = [0]
        elif pauli_ev.to_label()[-1] == "I":
            phase = pauli_ev.phase
            pauli_z_x = Pauli(pauli_ev.to_label()[-2])
            pauli_ev = Pauli((pauli_z_x.z, pauli_z_x.x, phase))
            qubits = [1]
        else:  # no "I" terms
            qubits = [0, 1]

        ppr = PauliProductRotationGate(pauli_ev, 1.0)
        circuit_target = QuantumCircuit(2)
        circuit_target.append(ppr, qubits)
        circuit_target.compose(circuit, [0, 1], inplace=True)
        self.assertEqual(circuit_out, circuit_target)

    @data("ppm", "ppr")
    def test_litinski_with_ppr_ppm_input(self, pp_type):
        """Test that LitinskiTransformation is correct for PPR/PPM as input"""
        num_qubits = 5
        qarg_paulis = [1, 2, 4]
        cliff = random_clifford_circuit(num_qubits, num_gates=20, seed=1234)
        pauli = random_pauli(len(qarg_paulis), seed=5678)

        # pad the original pauli
        p = pauli.to_label()
        p_pad = Pauli(p[0] + "I" + p[1] + p[2] + "I")
        pauli_ev = p_pad.evolve(cliff)
        # unpad the evolved pauli
        q = pauli_ev.to_label()
        phase = 0
        if q[0] == "-":
            q = q[1:]
            phase = 2
        out_str = ""
        out_ind = []
        for i in range(num_qubits):
            if q[i] != "I":
                out_str += q[i]
                out_ind.append(num_qubits - i - 1)
        out_ev = Pauli(out_str)
        out_ev.phase = phase

        if pp_type == "ppr":
            circuit = QuantumCircuit(num_qubits)
            circuit.compose(cliff, range(num_qubits), inplace=True)
            circuit.append(PauliProductRotationGate(pauli, angle=0.123), qarg_paulis)
        else:  # pp_type == "ppm"
            circuit = QuantumCircuit(num_qubits, 1)
            circuit.compose(cliff, range(num_qubits), inplace=True)
            circuit.append(PauliProductMeasurement(pauli), qarg_paulis, [0])

        transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
        circuit_out = transform(circuit)

        if pp_type == "ppr":
            circuit_target = QuantumCircuit(num_qubits)
            circuit_target.append(PauliProductRotationGate(out_ev, angle=0.123), out_ind)
            circuit_target.compose(cliff, range(num_qubits), inplace=True)
        else:  # pp_type == "ppm"
            circuit_target = QuantumCircuit(num_qubits, 1)
            circuit_target.append(PauliProductMeasurement(out_ev), out_ind, [0])
            circuit_target.compose(cliff, range(num_qubits), inplace=True)

        self.assertEqual(circuit_out, circuit_target)

    def test_litinski_with_clifford_ppr_pi2_angle(self):
        """Test that LitinskiTransformation is does not change the gates for Clifford and PPR input with pi/2 angle"""
        num_qubits = 5
        cliff = random_clifford_circuit(num_qubits, num_gates=20, seed=1234)

        circuit = QuantumCircuit(num_qubits)
        circuit.append(PauliProductRotationGate(Pauli("XZIY"), -np.pi / 2), [0, 1, 2, 4])
        circuit.compose(cliff, range(num_qubits), inplace=True)
        circuit.append(PauliProductRotationGate(Pauli("XYZ"), np.pi / 2), [1, 2, 4])

        transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
        circuit_out = transform(circuit)
        self.assertEqual(circuit_out, circuit)

    def test_litinski_with_ppr(self):
        """Test that LitinskiTransformation is correct for Clifford gates and PPR gates
        with and without pi/2 angles"""
        num_qubits = 5
        cliff1 = random_clifford_circuit(num_qubits, num_gates=10, seed=1234)
        cliff2 = random_clifford_circuit(num_qubits, num_gates=10, seed=2345)
        cliff3 = random_clifford_circuit(num_qubits, num_gates=10, seed=3456)

        circuit = QuantumCircuit(num_qubits)
        circuit.append(PauliProductRotationGate(Pauli("ZXIY"), -np.pi / 2), [0, 1, 2, 4])
        circuit.compose(cliff1, range(num_qubits), inplace=True)
        circuit.append(PauliProductRotationGate(Pauli("IYZ"), 3 * np.pi / 2), [2, 3, 4])
        circuit.compose(cliff2, range(num_qubits), inplace=True)
        circuit.append(PauliProductRotationGate(Pauli("ZYY"), 0.456), [0, 1, 2])

        circuit.append(PauliProductRotationGate(Pauli("XXYI"), 5 * np.pi / 2), [1, 2, 3, 4])
        circuit.compose(cliff3, range(num_qubits), inplace=True)
        circuit.append(PauliProductRotationGate(Pauli("XYZ"), 0.123), [1, 2, 4])

        transform = LitinskiTransformation(fix_clifford=True, use_ppr=True)
        circuit_out = transform(circuit)
        self.assertEqual(Operator(circuit_out), Operator(circuit))

    def test_simple_pbc_circuit(self):
        """Test a simple PBC circuit: H T H"""
        pbc = QuantumCircuit(1)
        pbc.append(PauliProductRotationGate(Pauli("Y"), np.pi / 2), [0])
        pbc.append(PauliProductRotationGate(Pauli("X"), np.pi), [0])
        pbc.append(PauliProductRotationGate(Pauli("Z"), np.pi / 4), [0])
        pbc.append(PauliProductRotationGate(Pauli("Y"), np.pi / 2), [0])
        pbc.append(PauliProductRotationGate(Pauli("X"), np.pi), [0])

        lit = LitinskiTransformation(fix_clifford=False, use_ppr=True)
        pbc = lit(pbc)

        out = QuantumCircuit(1)
        out.append(PauliProductRotationGate(Pauli("X"), np.pi / 4), [0])
        self.assertEqual(pbc, out)
