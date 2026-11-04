# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SetLayout pass"""

import unittest

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import SetLayout, ApplyLayout, FullAncillaAllocation
from qiskit.transpiler import PassManager, TranspilerError, InvalidLayoutError
from test import QiskitTestCase


class TestSetLayout(QiskitTestCase):
    """Tests the SetLayout pass"""

    def assertEqualToReference(self, result_to_compare):
        """Compare result_to_compare to a reference

                       ┌───┐ ░ ┌─┐
              q_0 -> 0 ┤ H ├─░─┤M├───────────────
                       ├───┤ ░ └╥┘┌─┐
              q_1 -> 1 ┤ H ├─░──╫─┤M├────────────
                       ├───┤ ░  ║ └╥┘      ┌─┐
              q_4 -> 2 ┤ H ├─░──╫──╫───────┤M├───
                       ├───┤ ░  ║  ║ ┌─┐   └╥┘
              q_2 -> 3 ┤ H ├─░──╫──╫─┤M├────╫────
                       └───┘ ░  ║  ║ └╥┘    ║
        ancilla_0 -> 4 ─────────╫──╫──╫─────╫────
                       ┌───┐ ░  ║  ║  ║ ┌─┐ ║
              q_3 -> 5 ┤ H ├─░──╫──╫──╫─┤M├─╫────
                       ├───┤ ░  ║  ║  ║ └╥┘ ║ ┌─┐
              q_5 -> 6 ┤ H ├─░──╫──╫──╫──╫──╫─┤M├
                       └───┘ ░  ║  ║  ║  ║  ║ └╥┘
               meas: 6/═════════╩══╩══╩══╩══╩══╩═
                                0  1  2  3  4  5
        """
        qr = QuantumRegister(6, "q")
        ancilla = QuantumRegister(1, "ancilla")
        cl = ClassicalRegister(6, "meas")
        reference = QuantumCircuit(qr, ancilla, cl)
        reference.h(qr)
        reference.barrier(qr)
        reference.measure(qr, cl)
        pass_manager = PassManager()
        pass_manager.append(
            SetLayout(
                Layout({qr[0]: 0, qr[1]: 1, qr[4]: 2, qr[2]: 3, ancilla[0]: 4, qr[3]: 5, qr[5]: 6})
            )
        )
        pass_manager.append(ApplyLayout())
        self.assertEqual(result_to_compare, pass_manager.run(reference))

    def test_setlayout_as_Layout(self):
        """Construct SetLayout with a Layout."""

        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.measure_all()

        pass_manager = PassManager()
        pass_manager.append(
            SetLayout(Layout.from_intlist([0, 1, 3, 5, 2, 6], QuantumRegister(6, "q")))
        )
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())
        result = pass_manager.run(circuit)

        self.assertEqualToReference(result)

    def test_setlayout_as_list(self):
        """Construct SetLayout with a list."""

        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.h(qr)
        circuit.measure_all()

        pass_manager = PassManager()
        pass_manager.append(SetLayout([0, 1, 3, 5, 2, 6]))
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())
        result = pass_manager.run(circuit)

        self.assertEqualToReference(result)

    def test_raise_when_layout_len_does_not_match(self):
        """Test error is raised if layout defined as list does not match the circuit size."""

        qr = QuantumRegister(42, "q")
        circuit = QuantumCircuit(qr)

        pass_manager = PassManager()
        pass_manager.append(SetLayout([0, 1, 3, 5, 2, 6]))
        pass_manager.append(FullAncillaAllocation(CouplingMap.from_line(7)))
        pass_manager.append(ApplyLayout())

        with self.assertRaises(TranspilerError):
            pass_manager.run(circuit)

    def test_raise_if_int_list_layout_contains_duplicates(self):
        """Test the error is raised if the specified intlist contains duplicates"""
        circuit = QuantumCircuit(4)
        layout_pass = SetLayout([0, 1, 1, 2])
        with self.assertRaises(InvalidLayoutError):
            layout_pass(circuit)


if __name__ == "__main__":
    unittest.main()
