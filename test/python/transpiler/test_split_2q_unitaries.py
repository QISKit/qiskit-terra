# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for the Split2QUnitaries transpiler pass.
"""

import io

from math import pi
from test import QiskitTestCase
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate, XGate, ZGate, HGate, CPhaseGate
from qiskit.circuit import Parameter, CircuitInstruction, Gate
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.transpiler.passes.optimization.split_2q_unitaries import Split2QUnitaries
from qiskit import qpy


class TestSplit2QUnitaries(QiskitTestCase):
    """
    Tests to verify that splitting two-qubit unitaries into two single-qubit unitaries works correctly.
    """

    def test_splits(self):
        """Test that the kronecker product of matrices is correctly identified by the pass and that the
        global phase is set correctly."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )

    def test_2q_identity(self):
        """Test that a 2q unitary matching the identity is correctly processed."""
        qc = QuantumCircuit(2)
        qc.id(0)
        qc.id(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_1q_identity(self):
        """Test that a Kronecker product with one identity gate on top is correctly processed."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.id(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_1q_identity2(self):
        """Test that a Kronecker product with one identity gate on bottom is correctly processed."""
        qc = QuantumCircuit(2)
        qc.id(0)
        qc.x(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_2_1q(self):
        """Test that a Kronecker product of two X gates is correctly processed."""
        x_mat = np.array([[0, 1], [1, 0]])
        multi_x = np.kron(x_mat, x_mat)
        qr = QuantumRegister(2, "qr")
        backend = GenericBackendV2(2)
        qc = QuantumCircuit(qr)
        qc.unitary(multi_x, qr)
        qct = transpile(qc, backend, optimization_level=2)
        self.assertTrue(Operator(qc).equiv(qct))
        self.assertTrue(matrix_equal(Operator(qc).data, Operator(qct).data, ignore_phase=False))
        self.assertEqual(qct.size(), 2)

    def test_no_split(self):
        """Test that the pass does not split a non-local two-qubit unitary."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.global_phase += 1.2345

        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        # either not a unitary gate, or the unitary has been consolidated to a 2q-unitary by another pass
        self.assertTrue(
            all(
                op.name != "unitary" or (op.name == "unitary" and len(op.qubits) > 1)
                for op in qc_split.data
            )
        )

    def test_almost_identity(self):
        """Test that the pass handles QFT correctly."""
        qc = QuantumCircuit(2)
        qc.unitary(CPhaseGate(pi * 2 ** -(26)).to_matrix(), [0, 1])
        pm = PassManager()
        pm.append(Split2QUnitaries(fidelity=1.0 - 1e-9))
        qc_split = pm.run(qc)
        pm = PassManager()
        pm.append(Split2QUnitaries())
        qc_split2 = pm.run(qc)
        self.assertEqual(qc_split.num_nonlocal_gates(), 0)
        self.assertEqual(qc_split2.num_nonlocal_gates(), 1)

    def test_almost_identity_param(self):
        """Test that the pass handles parameterized gates correctly."""
        qc = QuantumCircuit(2)
        param = Parameter("p*2**-26")
        qc.cp(param, 0, 1)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(fidelity=1.0 - 1e-9))
        qc_split = pm.run(qc)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split2 = pm.run(qc)
        self.assertEqual(qc_split.num_nonlocal_gates(), 1)
        self.assertEqual(qc_split2.num_nonlocal_gates(), 1)

    def test_single_q_gates(self):
        """Test that the pass handles circuits with single-qubit gates correctly."""
        qr = QuantumRegister(5)
        qc = QuantumCircuit(qr)
        qc.x(0)
        qc.z(1)
        qc.h(2)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(fidelity=1.0 - 1e-9))
        qc_split = pm.run(qc)
        self.assertEqual(qc_split.num_nonlocal_gates(), 0)
        self.assertEqual(qc_split.size(), 3)

        self.assertTrue(CircuitInstruction(XGate(), qubits=[qr[0]], clbits=[]) in qc.data)
        self.assertTrue(CircuitInstruction(ZGate(), qubits=[qr[1]], clbits=[]) in qc.data)
        self.assertTrue(CircuitInstruction(HGate(), qubits=[qr[2]], clbits=[]) in qc.data)

    def test_gate_no_array(self):
        """
        Test that the pass doesn't fail when the circuit contains a custom gate
        with no ``__array__`` implementation.

        Reproduce from: https://github.com/Qiskit/qiskit/issues/12970
        """

        class MyGate(Gate):
            """Custom gate"""

            def __init__(self):
                super().__init__("mygate", 2, [])

            def to_matrix(self):
                return np.eye(4, dtype=complex)
                # return np.eye(4, dtype=float)

        def mygate(self, qubit1, qubit2):
            return self.append(MyGate(), [qubit1, qubit2], [])

        QuantumCircuit.mygate = mygate

        qc = QuantumCircuit(2)
        qc.mygate(0, 1)

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )

    def test_nosplit_custom(self):
        """Test a single custom gate is not split, even if it is a product.

        That is because we cannot guarantee the split gate is valid in a given basis.
        Regression test for #12970.
        """

        class MyGate(Gate):
            """A custom gate that could be split."""

            def __init__(self):
                super().__init__("mygate", 2, [])

            def to_matrix(self):
                return np.eye(4, dtype=complex)

        qc = QuantumCircuit(2)
        qc.append(MyGate(), [0, 1])

        no_split = Split2QUnitaries()(qc)

        self.assertDictEqual({"mygate": 1}, no_split.count_ops())

    def test_2q_swap(self):
        """Test that a 2q unitary matching a swap gate is correctly processed."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(split_swap=True))
        res = pm.run(qc_split)
        res_op = Operator.from_circuit(res)
        expected_op = Operator(qc_split)
        self.assertNotIn("swap", res.count_ops())
        self.assertEqual(
            res.count_ops()["unitary"], 2
        )  # the original 2-qubit unitary should be split into 2 1-qubit unitaries.
        self.assertTrue(expected_op.equiv(res_op))
        self.assertTrue(matrix_equal(expected_op.data, res_op.data, ignore_phase=False))

    def test_2q_swap_with_1_qubit_gates(self):
        """Test that a 2q unitary matching a swap gate with 1-qubit
        gates before and after is correctly processed."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)
        qc.sx(0)
        qc.sdg(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(split_swap=True))
        res = pm.run(qc_split)
        res_op = Operator.from_circuit(res)
        expected_op = Operator(qc_split)

        self.assertNotIn("swap", res.count_ops())
        self.assertEqual(
            res.count_ops()["unitary"], 2
        )  # the original 2-qubit unitary should be split into 2 1-qubit unitaries.
        self.assertTrue(expected_op.equiv(res_op))
        self.assertTrue(matrix_equal(expected_op.data, res_op.data, ignore_phase=False))

    def test_2q_swap_with_non_unitary_swaps(self):
        """Test a 2q unitary matching a swap gate in a
        circuit containing explicit swap gates."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)

        qc_split = QuantumCircuit(3)
        qc_split.swap(0, 2)
        qc_split.swap(0, 1)

        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(split_swap=True))
        res = pm.run(qc_split)
        res_op = Operator.from_circuit(res)
        expected_op = Operator(qc_split)

        self.assertEqual(
            res.count_ops()["unitary"], 2
        )  # the original 2-qubit unitary should be split into 2 1-qubit unitaries.
        self.assertTrue(expected_op.equiv(res_op))
        self.assertTrue(matrix_equal(expected_op.data, res_op.data, ignore_phase=False))

    def test_2q_swap_with_large_circuit(self):
        """Test that a 2q unitary matching a swap gate with 1-qubit gates is
        correctly processed as part of a full transpilation pass on a large circuit."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        qc.swap(0, 1)
        qc.sx(0)
        qc.sdg(1)
        qc.global_phase += 1.2345

        qc_split = QuantumCircuit(5)
        qc_split.h(0)
        qc_split.swap(0, 2)
        qc_split.cx(0, 1)
        qc_split.swap(1, 0)
        qc_split.cx(0, 1)
        qc_split.append(UnitaryGate(Operator(qc)), [3, 1])
        qc_split.cx(0, 2)
        qc_split.cx(0, 3)
        qc_split.cx(0, 4)
        qc_split.h(1)

        pm = PassManager()
        pm.append(Split2QUnitaries(split_swap=True))
        res = pm.run(qc_split)
        res_op = Operator.from_circuit(res)
        expected_op = Operator(qc_split)

        self.assertEqual(
            res.count_ops()["unitary"], 2
        )  # the original 2-qubit unitary should be split into 2 1-qubit unitaries.
        self.assertTrue(expected_op.equiv(res_op))
        self.assertTrue(matrix_equal(expected_op.data, res_op.data, ignore_phase=False))

    def test_overloaded_unitary_name_from_qasm(self):
        """Test that an otherwise invalid custom gate named unitary created via valid Qiskit
        API calls doesn't crash the pass

        See: https://github.com/Qiskit/qiskit/issues/14103
        """

        qasm_str = """OPENQASM 2.0;
        include "qelib1.inc";
        gate cx_o0 q0,q1 { x q0; cx q0,q1; x q0; }
        gate unitary q0,q1 { u(pi/2,0.6763483147328913,0) q0; u(1.6719020266110614,-pi/2,0) q1; cx q0,q1; u(pi,-0.9111063207475532,3.1249343449042435) q0; u(1.6719020266110616,0,-pi/2) q1; }
        qreg v__0__0_[0];
        qreg l___0__0___1_[2];
        qreg l___0__0___2_[2];
        qreg v__0__1_[0];
        qreg l___0__1___1_[2];
        qreg v__1__0_[0];
        qreg l___1__0___1_[2];
        qreg l___1__0___2_[2];
        qreg v__1__1_[0];
        qreg l___1__1___1_[2];
        creg meas[12];
        unitary l___0__0___2_[0],l___0__1___1_[0];
        """
        # Parse qasm string to get custom unitary gate that's not a UnitaryGate (but has matrix defined)
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        # Roundtrip QPY to lose the custom matrix definition from qasm2
        # unitary
        with io.BytesIO() as buf:
            qpy.dump(qc, buf)
            # Rewind back to the beginning of the "file".
            buf.seek(0)
            qc = qpy.load(buf)[0]
        # Run split unitaries pass with custom gate named unitary that has
        # no matrix.
        res = Split2QUnitaries()(qc)
        self.assertEqual(res, qc)
