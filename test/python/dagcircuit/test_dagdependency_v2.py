# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for the _DAGDependencyV2 object"""

import unittest

from qiskit.dagcircuit.dagdependency_v2 import _DAGDependencyV2
from qiskit.dagcircuit import DAGOpNode
from qiskit.circuit import QuantumRegister, ClassicalRegister, QuantumCircuit, Qubit, Clbit
from qiskit.circuit import Measure
from qiskit.circuit import Instruction
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.converters.circuit_to_dagdependency_v2 import _circuit_to_dagdependency_v2
from test import QiskitTestCase

try:
    import rustworkx as rx
except ImportError:
    pass


def raise_if_dagdependency_invalid(dag):
    """Validates the internal consistency of a _DAGDependencyV2._multi_graph.
    Intended for use in testing.

    Raises:
       DAGDependencyError: if _DAGDependencyV2._multi_graph is inconsistent.
    """

    multi_graph = dag._multi_graph

    if not rx.is_directed_acyclic_graph(multi_graph):
        raise DAGDependencyError("multi_graph is not a DAG.")

    # Every node should be of type op.
    for node in dag.op_nodes():
        if not isinstance(node, DAGOpNode):
            raise DAGDependencyError(f"Found node of unexpected type: {node.type}")


class TestDagRegisters(QiskitTestCase):
    """Test qreg and creg inside the dag"""

    def test_add_qreg_creg(self):
        """add_qreg() and  add_creg() methods"""
        dag = _DAGDependencyV2()
        dag.add_qreg(QuantumRegister(2, "qr"))
        dag.add_creg(ClassicalRegister(1, "cr"))
        self.assertDictEqual(dag.qregs, {"qr": QuantumRegister(2, "qr")})
        self.assertDictEqual(dag.cregs, {"cr": ClassicalRegister(1, "cr")})

    def test_dag_get_qubits(self):
        """get_qubits() method"""
        dag = _DAGDependencyV2()
        dag.add_qreg(QuantumRegister(1, "qr1"))
        dag.add_qreg(QuantumRegister(1, "qr10"))
        dag.add_qreg(QuantumRegister(1, "qr0"))
        dag.add_qreg(QuantumRegister(1, "qr3"))
        dag.add_qreg(QuantumRegister(1, "qr4"))
        dag.add_qreg(QuantumRegister(1, "qr6"))
        self.assertListEqual(
            dag.qubits,
            [
                QuantumRegister(1, "qr1")[0],
                QuantumRegister(1, "qr10")[0],
                QuantumRegister(1, "qr0")[0],
                QuantumRegister(1, "qr3")[0],
                QuantumRegister(1, "qr4")[0],
                QuantumRegister(1, "qr6")[0],
            ],
        )

    def test_add_reg_duplicate(self):
        """add_qreg with the same register twice is not allowed."""
        dag = _DAGDependencyV2()
        qr = QuantumRegister(2)
        dag.add_qreg(qr)
        self.assertRaises(DAGDependencyError, dag.add_qreg, qr)

    def test_add_reg_duplicate_name(self):
        """Adding quantum registers with the same name is not allowed."""
        dag = _DAGDependencyV2()
        qr1 = QuantumRegister(3, "qr")
        dag.add_qreg(qr1)
        qr2 = QuantumRegister(2, "qr")
        self.assertRaises(DAGDependencyError, dag.add_qreg, qr2)

    def test_add_reg_bad_type(self):
        """add_qreg with a classical register is not allowed."""
        dag = _DAGDependencyV2()
        cr = ClassicalRegister(2)
        self.assertRaises(DAGDependencyError, dag.add_qreg, cr)

    def test_add_registerless_bits(self):
        """Verify we can add are retrieve bits without an associated register."""
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(3)]

        dag = _DAGDependencyV2()
        dag.add_qubits(qubits)
        dag.add_clbits(clbits)

        self.assertEqual(dag.qubits, qubits)
        self.assertEqual(dag.clbits, clbits)

    def test_add_duplicate_registerless_bits(self):
        """Verify we raise when adding a bit already present in the circuit."""
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(3)]

        dag = _DAGDependencyV2()
        dag.add_qubits(qubits)
        dag.add_clbits(clbits)

        with self.assertRaisesRegex(DAGDependencyError, r"duplicate qubits"):
            dag.add_qubits(qubits[:1])
        with self.assertRaisesRegex(DAGDependencyError, r"duplicate clbits"):
            dag.add_clbits(clbits[:1])


class TestDagNodeEdge(QiskitTestCase):
    """Test adding nodes and edges to a dag and related functions."""

    def setUp(self):
        super().setUp()
        self.dag = _DAGDependencyV2()
        self.qreg = QuantumRegister(2, "qr")
        self.creg = ClassicalRegister(2, "cr")

        self.dag.add_qreg(self.qreg)
        self.dag.add_creg(self.creg)

    def test_node(self):
        """Test the methods apply_operation_back(), _get_node() and op_nodes()"""
        circuit = QuantumCircuit(self.qreg, self.creg)

        circuit.h(self.qreg[0])
        self.dag.apply_operation_back(
            circuit.data[0].operation, circuit.data[0].qubits, circuit.data[0].clbits
        )
        self.assertIsInstance(self.dag._get_node(0).op, HGate)

        circuit.measure(self.qreg[0], self.creg[0])
        self.dag.apply_operation_back(
            circuit.data[1].operation, circuit.data[1].qubits, circuit.data[1].clbits
        )
        self.assertIsInstance(self.dag._get_node(1).op, Measure)

        nodes = list(self.dag.op_nodes())
        self.assertEqual(len(list(nodes)), 2)

        for node in nodes:
            self.assertIsInstance(node.op, Instruction)

        node_1 = nodes.pop()
        node_2 = nodes.pop()

        self.assertIsInstance(node_1.op, Measure)
        self.assertIsInstance(node_2.op, HGate)


class TestDagNodeSelection(QiskitTestCase):
    """Test methods that select successors and predecessors"""

    def setUp(self):
        super().setUp()
        self.dag = _DAGDependencyV2()
        self.qreg = QuantumRegister(2, "qr")
        self.creg = ClassicalRegister(2, "cr")
        self.dag.add_qreg(self.qreg)
        self.dag.add_creg(self.creg)

    def test_successors_predecessors(self):
        """Test the method get_successors."""

        circuit = QuantumCircuit(self.qreg, self.creg)
        circuit.h(self.qreg[0])
        circuit.x(self.qreg[0])
        circuit.h(self.qreg[0])
        circuit.x(self.qreg[1])
        circuit.h(self.qreg[0])
        circuit.measure(self.qreg[0], self.creg[0])

        self.dag = _circuit_to_dagdependency_v2(circuit)

        test_successors = [2, 4]
        successors = sorted([node._node_id for node in self.dag.successors(self.dag._get_node(1))])
        self.assertEqual(successors, test_successors)

        test_successors = []
        successors = sorted([node._node_id for node in self.dag.successors(self.dag._get_node(3))])
        self.assertEqual(successors, test_successors)

        test_descendants = [2, 4, 5]
        descendants = sorted(
            [node._node_id for node in self.dag.descendants(self.dag._get_node(1))]
        )
        self.assertEqual(descendants, test_descendants)

        test_descendants = []
        descendants = sorted(
            [node._node_id for node in self.dag.descendants(self.dag._get_node(3))]
        )
        self.assertEqual(descendants, test_descendants)

        test_predecessors = [2, 4]
        predecessors = sorted(
            [node._node_id for node in self.dag.predecessors(self.dag._get_node(5))]
        )
        self.assertEqual(predecessors, test_predecessors)

        test_predecessors = []
        predecessors = sorted(
            [node._node_id for node in self.dag.predecessors(self.dag._get_node(3))]
        )
        self.assertEqual(predecessors, test_predecessors)

        test_ancestors = [0, 1, 2, 4]
        ancestors = sorted([node._node_id for node in self.dag.ancestors(self.dag._get_node(5))])
        self.assertEqual(ancestors, test_ancestors)

        test_ancestors = []
        ancestors = sorted([node._node_id for node in self.dag.ancestors(self.dag._get_node(3))])
        self.assertEqual(ancestors, test_ancestors)


class TestDagProperties(QiskitTestCase):
    """Test the DAG properties."""

    def setUp(self):
        #       ┌───┐                ┌───┐
        # q0_0: ┤ H ├────────────────┤ X ├──────────
        #       └───┘                └─┬─┘     ┌───┐
        # q0_1: ───────────────────────┼───────┤ H ├
        #                 ┌───┐        │  ┌───┐└─┬─┘
        # q0_2: ──■───────┤ H ├────────┼──┤ T ├──■──
        #       ┌─┴─┐┌────┴───┴─────┐  │  └───┘
        # q0_3: ┤ X ├┤ U(0,0.1,0.2) ├──┼────────────
        #       └───┘└──────────────┘  │
        # q1_0: ───────────────────────■────────────
        #                              │
        # q1_1: ───────────────────────■────────────
        super().setUp()
        qr1 = QuantumRegister(4)
        qr2 = QuantumRegister(2)
        circ = QuantumCircuit(qr1, qr2)
        circ.h(qr1[0])
        circ.cx(qr1[2], qr1[3])
        circ.h(qr1[2])
        circ.t(qr1[2])
        circ.ch(qr1[2], qr1[1])
        circ.u(0.0, 0.1, 0.2, qr1[3])
        circ.ccx(qr2[0], qr2[1], qr1[0])

        self.dag = _circuit_to_dagdependency_v2(circ)

    def test_size(self):
        """Test total number of operations in dag."""
        self.assertEqual(self.dag.size(), 7)

    def test_dag_depth(self):
        """Test dag depth."""
        self.assertEqual(self.dag.depth(), 2)

    def test_dag_depth_empty(self):
        """Empty circuit DAG is zero depth"""
        q = QuantumRegister(5, "q")
        qc = QuantumCircuit(q)
        dag = _circuit_to_dagdependency_v2(qc)
        self.assertEqual(dag.depth(), 0)

    def test_default_metadata_value(self):
        """Test that the default _DAGDependencyV2 metadata is valid QuantumCircuit metadata."""
        qc = QuantumCircuit(1)
        qc.metadata = self.dag.metadata
        self.assertEqual(qc.metadata, {})


class TestCopy(QiskitTestCase):
    """Test removal of registers and idle wires."""

    def setUp(self):
        super().setUp()
        self.dag = _DAGDependencyV2()
        self.dag.name = "Name"
        self.dag.metadata = "Metadata"
        qreg = QuantumRegister(3, "qr")
        creg0 = ClassicalRegister(2, "c0")
        creg1 = ClassicalRegister(2, "c1")
        creg2 = ClassicalRegister(name="c2", bits=list(creg1))
        clbit = Clbit()
        self.dag.add_qreg(qreg)
        self.dag.add_creg(creg0)
        self.dag.add_creg(creg1)
        self.dag.add_creg(creg2)
        self.dag.add_clbits([clbit])

    def test_copy_empty_like(self):
        """Copy dag circuit metadata with copy_empty_like."""
        result_dag = self.dag.copy_empty_like()
        self.assertEqual(self.dag.name, result_dag.name)
        self.assertEqual(self.dag.metadata, result_dag.metadata)
        self.assertEqual(self.dag.clbits, result_dag.clbits)
        self.assertEqual(self.dag.qubits, result_dag.qubits)
        self.assertEqual(self.dag.cregs, result_dag.cregs)
        self.assertEqual(self.dag.qregs, result_dag.qregs)
        self.assertEqual(self.dag.duration, result_dag.duration)
        self.assertEqual(self.dag.unit, result_dag.unit)
        self.assertEqual(self.dag.comm_checker, result_dag.comm_checker)


if __name__ == "__main__":
    unittest.main()
