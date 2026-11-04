# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""UnitaryGate tests"""

import numpy
from numpy.testing import assert_allclose

from qiskit.circuit.library import UnitaryGate, CXGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators import Operator
from qiskit.transpiler.passes import InverseCancellation
from qiskit.qasm2 import dumps
from test import QiskitTestCase


class TestUnitaryGate(QiskitTestCase):
    """Tests for the Unitary class."""

    def test_set_matrix(self):
        """Test instantiation"""
        try:
            UnitaryGate([[0, 1], [1, 0]])

        except Exception as err:
            self.fail(f"unexpected exception in init of Unitary: {err}")

    def test_set_matrix_raises(self):
        """test non-unitary"""
        try:
            UnitaryGate([[1, 1], [1, 0]])

        except Exception:
            pass
        else:
            self.fail("setting Unitary with non-unitary did not raise")

    def test_set_init_with_unitary(self):
        """test instantiation of new unitary with another one (copy)"""
        uni1 = UnitaryGate([[0, 1], [1, 0]])
        uni2 = UnitaryGate(uni1)
        self.assertEqual(uni1, uni2)
        self.assertFalse(uni1 is uni2)

    def test_conjugate(self):
        """test conjugate"""
        ymat = numpy.array([[0, -1j], [1j, 0]])
        uni = UnitaryGate([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.conjugate().to_matrix(), ymat))

    def test_adjoint(self):
        """test adjoint operation"""
        uni = UnitaryGate([[0, 1j], [-1j, 0]])
        self.assertTrue(numpy.array_equal(uni.adjoint().to_matrix(), uni.to_matrix()))

    def test_repeat(self):
        """test repeat operation"""
        uni = UnitaryGate([[1, 0], [0, 1j]])
        self.assertTrue(numpy.array_equal(Operator(uni.repeat(2)), Operator(uni) @ Operator(uni)))


class TestUnitaryCircuit(QiskitTestCase):
    """Matrix gate circuit tests."""

    def test_1q_unitary(self):
        """test 1 qubit unitary matrix"""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        qc.x(qr[0])
        qc.append(UnitaryGate(matrix), [qr[0]])
        # test of qasm output
        self.log.info(dumps(qc))
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        dag_nodes = dag.named_nodes("unitary")
        self.assertTrue(len(dag_nodes) == 1)
        dnode = dag_nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        self.assertEqual(dnode.qargs, (qr[0],))
        assert_allclose(dnode.op.to_matrix(), matrix)

    def test_2q_unitary(self):
        """test 2 qubit unitary matrix"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmax, sigmay)
        qc.x(qr[0])
        uni2q = UnitaryGate(matrix)
        qc.append(uni2q, [qr[0], qr[1]])
        passman = PassManager()
        passman.append(InverseCancellation([CXGate()]))
        qc2 = passman.run(qc)
        # test of qasm output
        self.log.info(dumps(qc2))
        # test of text drawer
        self.log.info(qc2)
        dag = circuit_to_dag(qc)
        nodes = dag.two_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        self.assertEqual(dnode.qargs, (qr[0], qr[1]))
        assert_allclose(dnode.op.to_matrix(), matrix)
        qc3 = dag_to_circuit(dag)
        self.assertEqual(qc2, qc3)

    def test_3q_unitary(self):
        """test 3 qubit unitary matrix on non-consecutive bits"""
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr)
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmay = numpy.array([[0, -1j], [1j, 0]])
        matrix = numpy.kron(sigmay, numpy.kron(sigmax, sigmay))
        qc.x(qr[0])
        uni3q = UnitaryGate(matrix)
        qc.append(uni3q, [qr[0], qr[1], qr[3]])
        qc.cx(qr[3], qr[2])
        # test of text drawer
        self.log.info(qc)
        dag = circuit_to_dag(qc)
        nodes = dag.multi_qubit_ops()
        self.assertEqual(len(nodes), 1)
        dnode = nodes[0]
        self.assertIsInstance(dnode.op, UnitaryGate)
        self.assertEqual(dnode.qargs, (qr[0], qr[1], qr[3]))
        assert_allclose(dnode.op.to_matrix(), matrix)

    def test_1q_unitary_int_qargs(self):
        """test single qubit unitary matrix with 'int' and 'list of ints' qubits argument"""
        sigmax = numpy.array([[0, 1], [1, 0]])
        sigmaz = numpy.array([[1, 0], [0, -1]])
        # new syntax
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.unitary(sigmax, 0)
        qc.unitary(sigmax, qr[1])
        qc.unitary(sigmaz, [0, 1])
        # expected circuit
        qc_target = QuantumCircuit(qr)
        qc_target.append(UnitaryGate(sigmax), [0])
        qc_target.append(UnitaryGate(sigmax), [qr[1]])
        qc_target.append(UnitaryGate(sigmaz), [[0, 1]])
        self.assertEqual(qc, qc_target)

    def test_qasm_unitary_only_one_def(self):
        """test that a custom unitary can be converted to qasm and the
        definition is only written once"""
        qr = QuantumRegister(2, "q0")
        cr = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        unitary_gate = UnitaryGate(matrix)

        qc.x(qr[0])
        qc.append(unitary_gate, [qr[0]])
        qc.append(unitary_gate, [qr[1]])

        expected_qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "gate unitary q0 { u(0,0,0) q0; }\n"
            "qreg q0[2];\ncreg c0[1];\n"
            "x q0[0];\n"
            "unitary q0[0];\n"
            "unitary q0[1];"
        )
        self.assertEqual(expected_qasm, dumps(qc))

    def test_qasm_unitary_twice(self):
        """test that a custom unitary can be converted to qasm and that if
        the qasm is called twice it is the same every time"""
        qr = QuantumRegister(2, "q0")
        cr = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.array([[1, 0], [0, 1]])
        unitary_gate = UnitaryGate(matrix)

        qc.x(qr[0])
        qc.append(unitary_gate, [qr[0]])
        qc.append(unitary_gate, [qr[1]])

        expected_qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "gate unitary q0 { u(0,0,0) q0; }\n"
            "qreg q0[2];\ncreg c0[1];\n"
            "x q0[0];\n"
            "unitary q0[0];\n"
            "unitary q0[1];"
        )
        self.assertEqual(expected_qasm, dumps(qc))
        self.assertEqual(expected_qasm, dumps(qc))

    def test_qasm_2q_unitary(self):
        """test that a 2 qubit custom unitary can be converted to qasm"""
        qr = QuantumRegister(2, "q0")
        cr = ClassicalRegister(1, "c0")
        qc = QuantumCircuit(qr, cr)
        matrix = numpy.asarray([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
        unitary_gate = UnitaryGate(matrix)

        qc.x(qr[0])
        qc.append(unitary_gate, [qr[0], qr[1]])
        qc.append(unitary_gate, [qr[1], qr[0]])

        expected_qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "gate unitary q0,q1 { u(pi,-pi/2,pi/2) q0; u(pi,pi/2,-pi/2) q1; }\n"
            "qreg q0[2];\n"
            "creg c0[1];\n"
            "x q0[0];\n"
            "unitary q0[0],q0[1];\n"
            "unitary q0[1],q0[0];"
        )
        self.assertEqual(expected_qasm, dumps(qc))

    def test_qasm_unitary_noop(self):
        """Test that an identity unitary can be converted to OpenQASM 2"""
        qc = QuantumCircuit(QuantumRegister(3, "q0"))
        qc.unitary(numpy.eye(8), qc.qubits)
        expected_qasm = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "gate unitary q0,q1,q2 {  }\n"
            "qreg q0[3];\n"
            "unitary q0[0],q0[1],q0[2];"
        )
        self.assertEqual(expected_qasm, dumps(qc))

    def test_unitary_decomposition(self):
        """Test decomposition for unitary gates over 2 qubits."""
        qc = QuantumCircuit(3)
        qc.unitary(random_unitary(8, seed=42), [0, 1, 2])
        self.assertTrue(Operator(qc).equiv(Operator(qc.decompose())))

    def test_unitary_decomposition_via_definition(self):
        """Test decomposition for 1Q unitary via definition."""
        mat = numpy.array([[0, 1], [1, 0]])
        self.assertTrue(numpy.allclose(Operator(UnitaryGate(mat).definition).data, mat))

    def test_unitary_decomposition_via_definition_2q(self):
        """Test decomposition for 2Q unitary via definition."""
        mat = numpy.array([[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, -1, 0, 0]])
        self.assertTrue(numpy.allclose(Operator(UnitaryGate(mat).definition).data, mat))

    def test_unitary_control(self):
        """Test parameters of controlled - unitary."""
        mat = numpy.array([[0, 1], [1, 0]])
        gate = UnitaryGate(mat).control(annotated=False)
        self.assertTrue(numpy.allclose(gate.params, mat))
        self.assertTrue(numpy.allclose(gate.base_gate.params, mat))
