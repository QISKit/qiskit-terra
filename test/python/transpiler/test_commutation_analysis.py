# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Commutation analysis and transformation pass testing"""

import unittest

from qiskit.circuit import Gate, QuantumRegister, QuantumCircuit, Qubit
from qiskit.transpiler import PropertySet
from qiskit.transpiler.passes import CommutationAnalysis
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestCommutationAnalysis(QiskitTestCase):
    """Test the Commutation pass."""

    def setUp(self):
        super().setUp()
        self.pass_ = CommutationAnalysis()
        self.pset = self.pass_.property_set = PropertySet()

    def assertCommutationSet(self, result, expected):
        """Compares the result of propertyset["commutation_set"] with a dictionary of the form
        {'q[0]': [ [node_id, ...], [node_id, ...] ]}
        """
        result_to_compare = {}
        for qbit, sets in result.items():
            if not isinstance(qbit, Qubit):
                continue
            result_to_compare[qbit] = []
            for commutation_set in sets:
                result_to_compare[qbit].append(sorted(node._node_id for node in commutation_set))

        for qbit, sets in expected.items():
            for commutation_set in sets:
                commutation_set.sort()

        self.assertDictEqual(result_to_compare, expected)

    def test_commutation_set_property_is_created(self):
        """Test property is created"""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        self.assertIsNone(self.pset["commutation_set"])
        self.pass_.run(dag)
        self.assertIsNotNone(self.pset["commutation_set"])

    def test_all_gates(self):
        """Test all gates on 1 and 2 qubits

        qr0:----[H]---[x]---[y]---[t]---[s]---[rz]---[p]---[u]---[u]---.---.---.--
                                                                       |   |   |
        qr1:----------------------------------------------------------(+)-(Y)--.--
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.t(qr[0])
        circuit.s(qr[0])
        circuit.rz(0.5, qr[0])
        circuit.p(0.5, qr[0])
        circuit.u(1.57, 0.5, 0.6, qr[0])
        circuit.u(0.5, 0.6, 0.7, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [4], [5], [6], [7, 8, 9, 10], [11], [12], [13], [14], [15], [1]],
            qr[1]: [[2], [13], [14], [15], [3]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_non_commutative_circuit(self):
        """A simple circuit where no gates commute

        qr0:---[H]---

        qr1:---[H]---

        qr2:---[H]---
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {qr[0]: [[0], [6], [1]], qr[1]: [[2], [7], [3]], qr[2]: [[4], [8], [5]]}
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_non_commutative_circuit_2(self):
        """A simple circuit where no gates commute

        qr0:----.-------------
                |
        qr1:---(+)------.-----
                        |
        qr2:---[H]-----(+)----
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [6], [1]],
            qr[1]: [[2], [6], [8], [3]],
            qr[2]: [[4], [7], [8], [5]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_commutative_circuit(self):
        """A simple circuit where two CNOTs commute

        qr0:----.------------
                |
        qr1:---(+)-----(+)---
                        |
        qr2:---[H]------.----
        """

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {qr[0]: [[0], [6], [1]], qr[1]: [[2], [6, 8], [3]], qr[2]: [[4], [7], [8], [5]]}
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_commutative_circuit_2(self):
        """A simple circuit where a CNOT and a Z gate commute,
        and a CNOT and a CNOT commute

        qr0:----.-----[Z]-----
                |
        qr1:---(+)----(+)----
                       |
        qr2:---[H]-----.----
        """

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[0])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [6, 7], [1]],
            qr[1]: [[2], [6, 9], [3]],
            qr[2]: [[4], [8], [9], [5]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_commutative_circuit_3(self):
        """A simple circuit where multiple gates commute

        qr0:----.-----[Z]-----.----[z]-----
                |             |
        qr1:---(+)----(+)----(+)----.------
                       |            |
        qr2:---[H]-----.-----[x]---(+)-----
        """
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.z(qr[0])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[2])
        circuit.z(qr[0])
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [6, 8, 10, 12], [1]],
            qr[1]: [[2], [6, 9, 10], [13], [3]],
            qr[2]: [[4], [7], [9], [11, 13], [5]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_jordan_wigner_type_circuit(self):
        """A Jordan-Wigner type circuit where consecutive CNOTs commute

        qr0:----.-------------------------------------------------------------.----
                |                                                             |
        qr1:---(+)----.-------------------------------------------------.----(+)---
                      |                                                 |
        qr2:---------(+)----.-------------------------------------.----(+)---------
                            |                                     |
        qr3:---------------(+)----.-------------------------.----(+)---------------
                                  |                         |
        qr4:---------------------(+)----.-------------.----(+)---------------------
                                        |             |
        qr5:---------------------------(+)----[z]----(+)---------------------------
        """
        qr = QuantumRegister(6, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.z(qr[5])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [12, 22], [1]],
            qr[1]: [[2], [12], [13, 21], [22], [3]],
            qr[2]: [[4], [13], [14, 20], [21], [5]],
            qr[3]: [[6], [14], [15, 19], [20], [7]],
            qr[4]: [[8], [15], [16, 18], [19], [9]],
            qr[5]: [[10], [16], [17], [18], [11]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_all_commute_circuit(self):
        """Test circuit with that all commute"""

        #            ┌───┐
        # qr_0: ──■──┤ Z ├──■────────────
        #       ┌─┴─┐├───┤┌─┴─┐┌───┐
        # qr_1: ┤ X ├┤ X ├┤ X ├┤ X ├─────
        #       └───┘└─┬─┘└───┘└─┬─┘
        # qr_2: ───────■────■────■────■──
        #       ┌───┐     ┌─┴─┐┌───┐┌─┴─┐
        # qr_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├
        #       └─┬─┘┌───┐└───┘└─┬─┘└───┘
        # qr_4: ──■──┤ Z ├───────■───────
        #            └───┘
        qr = QuantumRegister(5, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[4], qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.z(qr[0])
        circuit.z(qr[4])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[4], qr[3])
        circuit.cx(qr[2], qr[3])
        dag = circuit_to_dag(circuit)

        self.pass_.run(dag)

        expected = {
            qr[0]: [[0], [10, 14, 16], [1]],
            qr[1]: [[2], [10, 11, 16, 17], [3]],
            qr[2]: [[4], [11, 13, 17, 19], [5]],
            qr[3]: [[6], [12, 13, 18, 19], [7]],
            qr[4]: [[8], [12, 15, 18], [9]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)

    def test_custom_gates_sharing_wire(self):
        """Custom gates sharing a wire do not commute"""
        circuit = QuantumCircuit(2)
        circuit.append(Gate("foo", 1, []), [1])
        circuit.append(Gate("bar", 2, []), [0, 1])

        self.pass_.run(circuit_to_dag(circuit))

        expected = {
            circuit.qubits[0]: [[0], [5], [1]],
            circuit.qubits[1]: [[2], [4], [5], [3]],
        }
        self.assertCommutationSet(self.pset["commutation_set"], expected)


if __name__ == "__main__":
    unittest.main()
