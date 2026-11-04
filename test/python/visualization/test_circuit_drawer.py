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


import os
import pathlib
import re
import shutil
import tempfile
import unittest
import warnings
from unittest.mock import patch

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, visualization
from qiskit.utils import optionals
from qiskit.visualization.circuit import styles, text
from qiskit.visualization.exceptions import VisualizationError
from test import QiskitTestCase

if optionals.HAS_MATPLOTLIB:
    from matplotlib import figure
if optionals.HAS_PIL:
    from PIL import Image


_latex_drawer_condition = unittest.skipUnless(
    all(
        (
            optionals.HAS_PYLATEX,
            optionals.HAS_PIL,
            optionals.HAS_PDFLATEX,
            optionals.HAS_PDFTOCAIRO,
        )
    ),
    "Skipped because not all of PIL, pylatex, pdflatex and pdftocairo are available",
)


class TestCircuitDrawer(QiskitTestCase):
    def test_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_mpl_config_with_path(self):

        # It's too easy to get too nested in a test with many context managers.
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        clifford_style = pathlib.Path(styles.__file__).parent / "clifford.json"
        shutil.copyfile(clifford_style, pathlib.Path(tempdir.name) / "my_clifford.json")

        circuit = QuantumCircuit(1)
        circuit.h(0)

        def config(style_name):
            return {
                "circuit_drawer": "mpl",
                "circuit_mpl_style": style_name,
                "circuit_mpl_style_path": [tempdir.name],
            }

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message="Style JSON file.*not found")

            # Test that a non-standard style can be loaded by name.
            with patch("qiskit.user_config.get_config", return_value=config("my_clifford")):
                self.assertIsInstance(visualization.circuit_drawer(circuit), figure.Figure)

            # Test that a non-existent style issues a warning, but still draws something.
            with patch("qiskit.user_config.get_config", return_value=config("NONEXISTENT")):
                with self.assertWarnsRegex(UserWarning, "Style JSON file.*not found"):
                    fig = visualization.circuit_drawer(circuit)
                self.assertIsInstance(fig, figure.Figure)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_user_config_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "mpl"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_output_with_user_config_not_set(self):
        with patch("qiskit.user_config.get_config", return_value={"other_option": True}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, text.TextDrawing)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_kwarg_priority_over_user_config_default_output(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit, output="mpl")
            self.assertIsInstance(out, figure.Figure)

    @unittest.skipUnless(optionals.HAS_MATPLOTLIB, "Skipped because matplotlib is not available")
    def test_default_backend_auto_output_with_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            circuit = QuantumCircuit()
            out = visualization.circuit_drawer(circuit)
            self.assertIsInstance(out, figure.Figure)

    def test_default_backend_auto_output_without_mpl(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "auto"}):
            with optionals.HAS_MATPLOTLIB.disable_locally():
                circuit = QuantumCircuit()
                out = visualization.circuit_drawer(circuit)
                self.assertIsInstance(out, text.TextDrawing)

    @_latex_drawer_condition
    def test_latex_unsupported_image_format_error_message(self):
        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            with self.assertRaises(VisualizationError, msg="Pillow could not write the image file"):
                visualization.circuit_drawer(circuit, filename="file.spooky")

    @_latex_drawer_condition
    def test_latex_output_file_correct_format(self):

        with patch("qiskit.user_config.get_config", return_value={"circuit_drawer": "latex"}):
            circuit = QuantumCircuit()
            filename = "file.gif"
            visualization.circuit_drawer(circuit, filename=filename)
            with Image.open(filename) as im:
                if filename.endswith("jpg"):
                    self.assertIn(im.format.lower(), "jpeg")
                else:
                    self.assertIn(im.format.lower(), filename.rsplit(".", maxsplit=1)[-1])
            os.remove(filename)

    def test_wire_order(self):
        """Test wire_order
        See: https://github.com/Qiskit/qiskit-terra/pull/9893"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "ca")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        with circuit.if_test((cr, 10)):
            circuit.x(3)

        expected = "\n".join(
            [
                "                                  ",
                " q_2: ────────────────────────────",
                "      ┌───┐┌────── ┌───┐ ───────┐ ",
                " q_3: ┤ H ├┤ If-0  ┤ X ├  End-0 ├─",
                "      ├───┤└──╥─── └───┘ ───────┘ ",
                " q_0: ┤ H ├───╫───────────────────",
                "      ├───┤   ║                   ",
                " q_1: ┤ X ├───╫───────────────────",
                "      └───┘┌──╨──┐                ",
                " c: 4/═════╡ 0xa ╞════════════════",
                "           └─────┘                ",
                "ca: 2/════════════════════════════",
                "                                  ",
            ]
        )
        result = visualization.circuit_drawer(circuit, output="text", wire_order=[2, 3, 0, 1])
        self.assertEqual(result.__str__(), expected)

    def test_wire_order_cregbundle(self):
        """Test wire_order with cregbundle=True
        See: https://github.com/Qiskit/qiskit-terra/pull/9893"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "ca")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        with circuit.if_test((cr, 10)):
            circuit.x(3)

        expected = "\n".join(
            [
                "                                  ",
                " q_2: ────────────────────────────",
                "      ┌───┐┌────── ┌───┐ ───────┐ ",
                " q_3: ┤ H ├┤ If-0  ┤ X ├  End-0 ├─",
                "      ├───┤└──╥─── └───┘ ───────┘ ",
                " q_0: ┤ H ├───╫───────────────────",
                "      ├───┤   ║                   ",
                " q_1: ┤ X ├───╫───────────────────",
                "      └───┘┌──╨──┐                ",
                " c: 4/═════╡ 0xa ╞════════════════",
                "           └─────┘                ",
                "ca: 2/════════════════════════════",
                "                                  ",
            ]
        )
        result = visualization.circuit_drawer(
            circuit, output="text", wire_order=[2, 3, 0, 1], cregbundle=True
        )
        self.assertEqual(result.__str__(), expected)

    def test_wire_order_raises(self):
        """Verify we raise if using wire order incorrectly."""

        circuit = QuantumCircuit(3, 3)
        circuit.x(1)

        with self.assertRaisesRegex(VisualizationError, "should not have repeated elements"):
            visualization.circuit_drawer(circuit, wire_order=[2, 1, 0, 3, 1, 5])

        with self.assertRaisesRegex(VisualizationError, "cannot be set when the reverse_bits"):
            visualization.circuit_drawer(circuit, wire_order=[0, 1, 2, 5, 4, 3], reverse_bits=True)

        with self.assertWarnsRegex(RuntimeWarning, "cregbundle set"):
            visualization.circuit_drawer(circuit, cregbundle=True, wire_order=[0, 1, 2, 5, 4, 3])

    def test_reverse_bits(self):
        """Test reverse_bits should not raise warnings when no classical qubits:
        See: https://github.com/Qiskit/qiskit-terra/pull/8689"""
        circuit = QuantumCircuit(3)
        circuit.x(1)
        expected = "\n".join(
            [
                "          ",
                "q_2: ─────",
                "     ┌───┐",
                "q_1: ┤ X ├",
                "     └───┘",
                "q_0: ─────",
                "          ",
            ]
        )
        result = visualization.circuit_drawer(circuit, output="text", reverse_bits=True)
        self.assertEqual(result.__str__(), expected)

    def test_warning_for_bad_justify_argument(self):
        """Test that the correct DeprecationWarning is raised when the justify parameter is badly input,
        for both of the public interfaces."""
        circuit = QuantumCircuit()
        bad_arg = "bad"
        error_message = re.escape(
            f"Setting QuantumCircuit.draw()’s or circuit_drawer()'s justify argument: {bad_arg}, to a "
            "value other than 'left', 'right', 'none' or None (='left'). Default 'left' will be used. "
            "Support for invalid justify arguments is deprecated as of Qiskit 1.2.0. Starting no "
            "earlier than 3 months after the release date, invalid arguments will error.",
        )

        with self.assertWarnsRegex(DeprecationWarning, error_message):
            visualization.circuit_drawer(circuit, justify=bad_arg)

        with self.assertWarnsRegex(DeprecationWarning, error_message):
            circuit.draw(justify=bad_arg)

    @unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc for LaTeX conversion")
    def test_no_explict_cregbundle(self):
        """Test no explicit cregbundle should not raise warnings about being disabled
        See: https://github.com/Qiskit/qiskit-terra/issues/8690"""
        inner = QuantumCircuit(1, 1, name="inner")
        inner.measure(0, 0)
        circuit = QuantumCircuit(2, 2)
        circuit.append(inner, [0], [0])
        expected = "\n".join(
            [
                "     ┌────────┐",
                "q_0: ┤0       ├",
                "     │        │",
                "q_1: ┤  inner ├",
                "     │        │",
                "c_0: ╡0       ╞",
                "     └────────┘",
                "c_1: ══════════",
                "               ",
            ]
        )
        result = circuit.draw("text")
        self.assertEqual(result.__str__(), expected)
        # Extra tests that no cregbundle (or any other) warning is raised with the default settings
        # for the other drawers, if they're available to test.
        circuit.draw("latex_source")
        if optionals.HAS_MATPLOTLIB and optionals.HAS_PYLATEX:
            circuit.draw("mpl")
