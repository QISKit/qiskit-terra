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

"""Test the FullAncillaAllocation pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout, Target
from qiskit.circuit.library import CXGate
from qiskit.transpiler.passes import FullAncillaAllocation
from qiskit.transpiler.exceptions import TranspilerError
from test import QiskitTestCase


class TestFullAncillaAllocation(QiskitTestCase):
    """Tests the ExtendLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap5 = CouplingMap([[1, 0], [2, 0], [2, 1], [3, 2], [3, 4], [4, 2]])

    def test_3q_circuit_5q_coupling(self):
        """Allocates 2 ancillas for a 3q circuit in a 5q coupling map

                    0 -> q0
        q0 -> 0     1 -> q1
        q1 -> 1  => 2 -> q2
        q2 -> 2     3 -> ancilla0
                    4 -> ancilla1
        """
        qr = QuantumRegister(3, "q")
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[1] = qr[1]
        initial_layout[2] = qr[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set["layout"] = initial_layout

        pass_.run(dag)
        after_layout = pass_.property_set["layout"]

        ancilla = QuantumRegister(2, "ancilla")

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], qr[1])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], ancilla[0])
        self.assertEqual(after_layout[4], ancilla[1])

    def test_3q_circuit_5q_target(self):
        """Allocates 2 ancillas for a 3q circuit in a 5q coupling map

                    0 -> q0
        q0 -> 0     1 -> q1
        q1 -> 1  => 2 -> q2
        q2 -> 2     3 -> ancilla0
                    4 -> ancilla1
        """
        target = Target(num_qubits=5)
        target.add_instruction(CXGate(), dict.fromkeys(self.cmap5.get_edges()))

        qr = QuantumRegister(3, "q")
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[1] = qr[1]
        initial_layout[2] = qr[2]

        pass_ = FullAncillaAllocation(target)
        pass_.property_set["layout"] = initial_layout

        pass_.run(dag)
        after_layout = pass_.property_set["layout"]

        ancilla = QuantumRegister(2, "ancilla")

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], qr[1])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], ancilla[0])
        self.assertEqual(after_layout[4], ancilla[1])

    def test_3q_with_holes_5q_coupling(self):
        """Allocates 3 ancillas for a 2q circuit on a 5q coupling, with holes

                       0 -> q0
        q0 -> 0        1 -> ancilla0
        q1 -> 2    =>  2 ->  q2
                       3 -> ancilla1
                       4 -> ancilla2
        """
        qr = QuantumRegister(2, "q")
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[2] = qr[1]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set["layout"] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set["layout"]

        ancilla = QuantumRegister(3, "ancilla")

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], ancilla[0])
        self.assertEqual(after_layout[2], qr[1])
        self.assertEqual(after_layout[3], ancilla[1])
        self.assertEqual(after_layout[4], ancilla[2])

    def test_3q_out_of_order_5q_coupling(self):
        """Allocates 2 ancillas a 3q circuit on a 5q coupling map, out of order

                       0 <- q0
        q0 -> 0        1 <- ancilla0
        q1 -> 3   =>   2 <- q2
        q2 -> 2        3 <- q1
                       4 <- ancilla1
        """
        qr = QuantumRegister(3, "q")
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = qr[0]
        initial_layout[3] = qr[1]
        initial_layout[2] = qr[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set["layout"] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set["layout"]

        ancilla = QuantumRegister(2, "ancilla")

        self.assertEqual(after_layout[0], qr[0])
        self.assertEqual(after_layout[1], ancilla[0])
        self.assertEqual(after_layout[2], qr[2])
        self.assertEqual(after_layout[3], qr[1])
        self.assertEqual(after_layout[4], ancilla[1])

    def test_name_collision(self):
        """Name collision during ancilla allocation."""
        qr_ancilla = QuantumRegister(3, "ancilla")
        circuit = QuantumCircuit(qr_ancilla)
        circuit.h(qr_ancilla)
        dag = circuit_to_dag(circuit)

        initial_layout = Layout()
        initial_layout[0] = qr_ancilla[0]
        initial_layout[1] = qr_ancilla[1]
        initial_layout[2] = qr_ancilla[2]
        initial_layout.add_register(qr_ancilla)

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set["layout"] = initial_layout
        pass_.run(dag)
        after_layout = pass_.property_set["layout"]

        layout_qregs = after_layout.get_registers()
        self.assertEqual(len(layout_qregs), 2)
        self.assertIn(qr_ancilla, layout_qregs)

        layout_qregs.remove(qr_ancilla)
        after_ancilla_register = layout_qregs.pop()

        self.assertEqual(len(after_ancilla_register), 2)
        self.assertRegex(after_ancilla_register.name, r"^ancilla\d+$")

        self.assertTrue(
            all(
                qubit in qr_ancilla or qubit in after_ancilla_register
                for qubit in after_layout.get_virtual_bits()
            )
        )

    def test_bad_layout(self):
        """Layout refers to a register that do not exist in the circuit"""
        qr = QuantumRegister(3, "q")
        circ = QuantumCircuit(qr)
        dag = circuit_to_dag(circ)

        initial_layout = Layout()
        initial_layout[0] = QuantumRegister(4, "q")[0]
        initial_layout[1] = QuantumRegister(4, "q")[1]
        initial_layout[2] = QuantumRegister(4, "q")[2]

        pass_ = FullAncillaAllocation(self.cmap5)
        pass_.property_set["layout"] = initial_layout

        with self.assertRaises(TranspilerError) as cm:
            pass_.run(dag)
        self.assertEqual(
            "FullAncillaAllocation: The layout refers to a qubit that does not exist in circuit.",
            cm.exception.message,
        )

    def test_target_without_cmap(self):
        """Test that FullAncillaAllocation works when the target does not have a coupling map.

        This situation occurs at the early stages of backend bring-up.
        """
        target_data = {"basis_gates": ["h"], "num_qubits": 3}
        target = Target.from_configuration(**target_data)

        circ = QuantumCircuit(1)
        circ.h(0)

        pass_ = FullAncillaAllocation(target)
        pass_.property_set["layout"] = Layout.from_intlist([0], *circ.qregs)

        # Pre pass check
        self.assertEqual(len(pass_.property_set["layout"]), 1)

        pass_.run(circuit_to_dag(circ))

        # Post pass check
        self.assertEqual(len(pass_.property_set["layout"]), target.num_qubits)


if __name__ == "__main__":
    unittest.main()
