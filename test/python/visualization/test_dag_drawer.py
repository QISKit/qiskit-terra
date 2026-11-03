# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for DAG visualization tool."""

import os
import tempfile
import unittest
import itertools

from qiskit.circuit import QuantumRegister, QuantumCircuit, Qubit, Clbit, Store, ClassicalRegister
from qiskit.visualization import dag_drawer
from qiskit.exceptions import InvalidFileError
from qiskit.visualization.exceptions import VisualizationError
from qiskit.converters import circuit_to_dag, circuit_to_dagdependency
from qiskit.utils import optionals as _optionals
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.classical import expr, types
from .visualization import path_to_diagram_reference, QiskitVisualizationTestCase


class TestDagDrawer(QiskitVisualizationTestCase):
    """Qiskit DAG drawer tests."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        self.dag = circuit_to_dag(circuit)

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_invalid_style(self):
        """Test dag draw with invalid style."""
        with self.assertRaisesRegex(VisualizationError, "Invalid style multicolor"):
            dag_drawer(self.dag, style="multicolor")

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_empty_style(self):
        """
        Test that dag_drawer() with an empty dict returns a plain DAG
        """
        dag_drawer(self.dag, style={})

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_custom_style(self):
        """
        Test dag with various custom styles
        """

        style = {
            "fontsize": 12,
            "bgcolor": "white",
            "dpi": 10,
            "pad": 0,
            "nodecolor": "green",
            "inputnodecolor": "blue",
            "inputnodefontcolor": "white",
            "outputnodecolor": "red",
            "outputnodefontcolor": "white",
            "opnodecolor": "black",
            "opnodefontcolor": "white",
            "edgecolor": "black",
            "qubitedgecolor": "black",
            "clbitedgecolor": "black",
        }

        for r in range(2):
            combinations = itertools.combinations(style, r)
            for c in combinations:
                curr_style = {x: style[x] for x in c}
                dag_drawer(self.dag, style=curr_style)

        dag_drawer(self.dag, style=style)

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_checks_filename_correct_format(self):
        """filename must contain name and extension"""
        with self.assertRaisesRegex(
            InvalidFileError, "Parameter 'filename' must be in format 'name.extension'"
        ):
            dag_drawer(self.dag, filename="aaabc")

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_checks_filename_extension(self):
        """filename must have a valid extension"""
        with self.assertRaisesRegex(
            ValueError,
            "The specified value for the image_type argument, 'abc' is not a "
            "valid choice. It must be one of: .*",
        ):
            dag_drawer(self.dag, filename="aa.abc")

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_no_register(self):
        """Test dag visualization with a circuit with no registers."""
        from PIL import Image

        qubit = Qubit()
        clbit = Clbit()
        qc = QuantumCircuit([qubit, clbit])
        qc.h(0)
        qc.measure(0, 0)
        dag = circuit_to_dag(qc)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dag_drawer(dag, filename=tmp_path)
            image_ref = path_to_diagram_reference("dag_no_reg.png")
            image = Image.open(tmp_path)
            self.assertImagesAreEqual(image, image_ref, 0.1)

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_node_indices(self):
        """Test the `node_indices` keyword argument functions."""
        from PIL import Image  # pylint: disable=import-error

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(qc.qubits, qc.clbits)
        dag = circuit_to_dag(qc)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag.png")
            dag.draw(filename=tmp_path, node_indices=True)
            image_ref = path_to_diagram_reference("dag_node_indices.png")
            image = Image.open(tmp_path)
            self.assertImagesAreEqual(image, image_ref, 0.1)

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_with_dag_dep(self):
        """Test dag dependency visualization."""
        from PIL import Image

        bits = [Qubit(), Clbit()]
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(4, "cr")
        qc = QuantumCircuit(qr, bits, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with qc.if_test((cr[1], 1)):
            qc.x(3)
        qc.h(3)
        qc.x(4)
        qc.barrier(0, 1)
        qc.measure(0, 0)
        dag = circuit_to_dagdependency(qc)
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_path = os.path.join(tmpdirname, "dag_d.png")
            dag_drawer(dag, filename=tmp_path)
            image_ref = path_to_diagram_reference("dag_dep.png")
            image = Image.open(tmp_path)
            self.assertImagesAreEqual(image, image_ref, 0.1)

    @unittest.skipUnless(_optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(_optionals.HAS_PIL, "PIL not installed")
    def test_dag_drawer_with_var_wires(self):
        """Test visualization works with var nodes."""
        a = expr.Var.new("a", types.Bool())
        dag = DAGCircuit()
        dag.add_input_var(a)
        dag.apply_operation_back(Store(a, a), (), ())
        image = dag_drawer(dag)
        self.assertIsNotNone(image)


if __name__ == "__main__":
    unittest.main(verbosity=2)
