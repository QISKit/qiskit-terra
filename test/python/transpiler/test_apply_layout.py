# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the ApplyLayout pass"""

import unittest

from qiskit.circuit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit.classical import expr, types
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes import ApplyLayout, SetLayout
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers import common
from qiskit.transpiler import PassManager, CouplingMap
from test import QiskitTestCase

from ..legacy_cmaps import YORKTOWN_CMAP


class TestApplyLayout(QiskitTestCase):
    """Tests the ApplyLayout pass."""

    def test_trivial(self):
        """Test if the bell circuit with virtual qubits is transformed into
        the circuit with physical qubits under trivial layout.
        """
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        q = QuantumRegister(2, "q")
        expected = QuantumCircuit(q)
        expected.h(q[0])
        expected.cx(q[0], q[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 0, v[1]: 1})
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_empty_layout(self):
        """If the layout and the backend are empty, the pass should still be well behaved."""
        qc = QuantumCircuit()
        out = ApplyLayout()(qc, property_set={"layout": Layout()})
        self.assertEqual(out.layout.initial_virtual_layout(), Layout())

    def test_empty_post_layout(self):
        """If the layout and the backend are empty, the pass should still be well behaved."""
        qc = QuantumCircuit()
        out = ApplyLayout()(qc, property_set={"layout": Layout(), "post_layout": Layout()})
        self.assertEqual(out.layout.initial_virtual_layout(), Layout())

    def test_raise_when_no_layout_is_supplied(self):
        """Test error is raised if no layout is found in property_set."""
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_raise_when_no_full_layout_is_given(self):
        """Test error is raised if no full layout is given."""
        v = QuantumRegister(2, "v")
        circuit = QuantumCircuit(v)
        circuit.h(v[0])
        circuit.cx(v[0], v[1])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 2, v[1]: 1})
        with self.assertRaises(TranspilerError):
            pass_.run(dag)

    def test_circuit_with_swap_gate(self):
        """Test if a virtual circuit with one swap gate is transformed into
        a circuit with physical qubits.

        [Circuit with virtual qubits]
          v0:--X---.---M(v0->c0)
               |   |
          v1:--X---|---M(v1->c1)
                   |
          v2:-----(+)--M(v2->c2)

         Initial layout: {v[0]: 2, v[1]: 1, v[2]: 0}

        [Circuit with physical qubits]
          q2:--X---.---M(q2->c0)
               |   |
          q1:--X---|---M(q1->c1)
                   |
          q0:-----(+)--M(q0->c2)
        """
        v = QuantumRegister(3, "v")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(v, cr)
        circuit.swap(v[0], v[1])
        circuit.cx(v[0], v[2])
        circuit.measure(v[0], cr[0])
        circuit.measure(v[1], cr[1])
        circuit.measure(v[2], cr[2])

        q = QuantumRegister(3, "q")
        expected = QuantumCircuit(q, cr)
        expected.swap(q[2], q[1])
        expected.cx(q[2], q[0])
        expected.measure(q[2], cr[0])
        expected.measure(q[1], cr[1])
        expected.measure(q[0], cr[2])

        dag = circuit_to_dag(circuit)
        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout({v[0]: 2, v[1]: 1, v[2]: 0})
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_final_layout_is_updated(self):
        """Test that if vf2postlayout runs that we've updated the final layout."""
        qubits = 3
        qc = QuantumCircuit(qubits)
        for i in range(5):
            qc.cx(i % qubits, int(i + qubits / 2) % qubits)
        initial_pm = PassManager([SetLayout([1, 3, 4])])
        cmap = CouplingMap(YORKTOWN_CMAP)
        initial_pm += common.generate_embed_passmanager(cmap)
        first_layout_circ = initial_pm.run(qc)
        out_pass = ApplyLayout()
        out_pass.property_set["layout"] = first_layout_circ.layout.initial_layout
        out_pass.property_set["original_qubit_indices"] = (
            first_layout_circ.layout.input_qubit_mapping
        )
        out_pass.property_set["final_layout"] = Layout(
            {
                first_layout_circ.qubits[0]: 0,
                first_layout_circ.qubits[1]: 3,
                first_layout_circ.qubits[2]: 2,
                first_layout_circ.qubits[3]: 4,
                first_layout_circ.qubits[4]: 1,
            }
        )
        # Set a post layout like vf2postlayout would:
        out_pass.property_set["post_layout"] = Layout(
            {
                first_layout_circ.qubits[0]: 0,
                first_layout_circ.qubits[2]: 4,
                first_layout_circ.qubits[1]: 2,
                first_layout_circ.qubits[3]: 1,
                first_layout_circ.qubits[4]: 3,
            }
        )
        out_pass.run(circuit_to_dag(first_layout_circ))
        self.assertEqual(
            out_pass.property_set["final_layout"],
            Layout(
                {
                    first_layout_circ.qubits[0]: 0,
                    first_layout_circ.qubits[1]: 3,
                    first_layout_circ.qubits[2]: 1,
                    first_layout_circ.qubits[3]: 2,
                    first_layout_circ.qubits[4]: 4,
                }
            ),
        )

    def test_works_with_var_nodes(self):
        """Test that standalone var nodes work."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(2, 2, inputs=[a])
        qc.add_var(b, 12)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.store(a, expr.bit_and(a, expr.bit_xor(qc.clbits[0], qc.clbits[1])))

        expected = QuantumCircuit(QuantumRegister(2, "q"), *qc.cregs, inputs=[a])
        expected.add_var(b, 12)
        expected.h(1)
        expected.cx(1, 0)
        expected.measure([1, 0], [0, 1])
        expected.store(a, expr.bit_and(a, expr.bit_xor(qc.clbits[0], qc.clbits[1])))

        pass_ = ApplyLayout()
        pass_.property_set["layout"] = Layout(dict(enumerate(reversed(qc.qubits))))
        after = pass_(qc)

        self.assertEqual(after, expected)


if __name__ == "__main__":
    unittest.main()
