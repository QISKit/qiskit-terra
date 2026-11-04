# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from qiskit.circuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import ContainsInstruction
from test import QiskitTestCase


class TestContainsInstructionPass(QiskitTestCase):
    def test_empty_dag(self):
        pass_ = ContainsInstruction("x")
        pass_.run(DAGCircuit())
        self.assertFalse(pass_.property_set["contains_x"])

        gates = ["x", "z"]
        pass_ = ContainsInstruction(gates)
        pass_.run(DAGCircuit())
        for gate in gates:
            self.assertFalse(pass_.property_set[f"contains_{gate}"])

    def test_simple_dag(self):
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(0)
        qc.cx(0, 1)

        pass_ = ContainsInstruction("cx")
        pass_(qc)
        self.assertTrue(pass_.property_set["contains_cx"])

        pass_ = ContainsInstruction("measure")
        pass_(qc)
        self.assertFalse(pass_.property_set["contains_measure"])

        pass_ = ContainsInstruction(["cx", "measure"])
        pass_(qc)
        self.assertTrue(pass_.property_set["contains_cx"])
        self.assertFalse(pass_.property_set["contains_measure"])

    def test_control_flow_dag(self):
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
            qc.cx(2, 3)
        with else_:
            qc.x(1)
            with qc.for_loop(range(3)):
                qc.z(2)
                with qc.for_loop((4, 0, 1)):
                    qc.y(2)
        with qc.while_loop((qc.clbits[0], True)):
            qc.h(0)
            qc.measure(0, 0)

        in_base = ["h", "measure", "if_else", "while_loop"]
        in_nested = ["x", "cx", "for_loop", "z", "y"]
        not_in = ["reset", "delay"]
        all_ = in_base + in_nested + not_in

        pass_ = ContainsInstruction(all_, recurse=False)
        pass_(qc)
        for present in in_base:
            self.assertTrue(pass_.property_set[f"contains_{present}"])
        for not_present in in_nested + not_in:
            self.assertFalse(pass_.property_set[f"contains_{not_present}"])

        pass_ = ContainsInstruction(all_, recurse=True)
        pass_(qc)
        for present in in_base + in_nested:
            self.assertTrue(pass_.property_set[f"contains_{present}"])
        for not_present in not_in:
            self.assertFalse(pass_.property_set[f"contains_{not_present}"])
