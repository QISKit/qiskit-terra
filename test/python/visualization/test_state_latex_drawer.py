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

"""Tests for visualization of latex state and unitary drawers"""

import unittest

from qiskit.quantum_info import Statevector
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.state_visualization import state_drawer, state_to_latex
from qiskit.utils import optionals
from .visualization import QiskitVisualizationTestCase


class TestLatexStateDrawer(QiskitVisualizationTestCase):
    """Qiskit state and unitary latex drawer."""

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state(self):
        """Test latex state vector drawer works with default settings."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source")
        expected_output = (
            r"\frac{1}{4} |0000\rangle- \frac{i}{4} |0001\rangle+\frac{i}{4} |0010\rangle"
            r"+\frac{1}{4} |0011\rangle- \frac{1}{4} |0100\rangle+\frac{i}{4} |0101\rangle"
            r" + \ldots +\frac{1}{4} |1011\rangle- \frac{1}{4} |1100\rangle"
            r"+\frac{i}{4} |1101\rangle- \frac{i}{4} |1110\rangle- \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_max_size(self):
        """Test `max_size` parameter for latex ket notation."""

        sv = Statevector.from_label("+-rl")
        output = state_drawer(sv, "latex_source", max_size=4)
        expected_output = (
            r"\frac{1}{4} |0000\rangle- \frac{i}{4} |0001\rangle"
            r" + \ldots - \frac{1}{4} |1111\rangle"
        )
        self.assertEqual(output, expected_output)

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_z(self):
        """Test state_to_latex with z basis (default)."""
        sv = Statevector.from_label("0")
        latex_default = state_to_latex(sv)
        latex_z = state_to_latex(sv, ket_basis="z")
        # default should be same as explicit z basis
        self.assertEqual(latex_default, latex_z)
        self.assertEqual(latex_z, r" |0\rangle")

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_x(self):
        """Test state_to_latex with x basis relabels the kets to |+>, |->."""
        # |0> in the X basis is (|+> + |->)/sqrt(2)
        latex_x = state_to_latex(Statevector.from_label("0"), ket_basis="x")
        self.assertEqual(
            latex_x,
            r"\frac{\sqrt{2}}{2} |+\rangle+\frac{\sqrt{2}}{2} |-\rangle",
        )
        # An X eigenstate collapses to a single ket in the X basis.
        self.assertEqual(state_to_latex(Statevector.from_label("+"), ket_basis="x"), r" |+\rangle")
        self.assertEqual(state_to_latex(Statevector.from_label("-"), ket_basis="x"), r" |-\rangle")

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_y(self):
        """Test state_to_latex with y basis relabels the kets to |+i>, |-i>."""
        # |0> in the Y basis is (|+i> + |-i>)/sqrt(2)
        latex_y = state_to_latex(Statevector.from_label("0"), ket_basis="y")
        self.assertEqual(
            latex_y,
            r"\frac{\sqrt{2}}{2} |+i\rangle+\frac{\sqrt{2}}{2} |-i\rangle",
        )
        # A Y eigenstate collapses to a single ket in the Y basis
        # ('r' is |+i> = (|0> + i|1>)/sqrt(2), 'l' is |-i>).
        self.assertEqual(state_to_latex(Statevector.from_label("r"), ket_basis="y"), r" |+i\rangle")
        self.assertEqual(state_to_latex(Statevector.from_label("l"), ket_basis="y"), r" |-i\rangle")

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_multiqubit(self):
        """Test state_to_latex relabels each qubit for multi-qubit states."""
        # |01> in the X basis expands to four equally-weighted X-basis kets.
        latex_x = state_to_latex(Statevector.from_label("01"), ket_basis="x")
        self.assertEqual(
            latex_x,
            r"\frac{1}{2} |++\rangle- \frac{1}{2} |+-\rangle"
            r"+\frac{1}{2} |-+\rangle- \frac{1}{2} |--\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_draw(self):
        """Test the ket basis can be set via ``Statevector.draw``."""
        latex_x = Statevector.from_label("+").draw("latex_source", ket_basis="x")
        self.assertEqual(latex_x, r" |+\rangle")

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_y_multiqubit(self):
        """Test Y-basis phases and qubit order for a product of eigenstates."""
        self.assertEqual(
            state_to_latex(Statevector.from_label("rl"), ket_basis="y"),
            r" |+i-i\rangle",
        )
        self.assertEqual(
            state_to_latex(Statevector.from_label("1"), ket_basis="y"),
            r"- \frac{\sqrt{2} i}{2} |+i\rangle+\frac{\sqrt{2} i}{2} |-i\rangle",
        )

    @unittest.skipUnless(optionals.HAS_SYMPY, "needs sympy")
    def test_state_ket_basis_invalid(self):
        """Test state_to_latex with invalid basis raises error."""
        sv = Statevector.from_label("0")
        for basis in ("invalid", "h"):
            with self.subTest(basis=basis), self.assertRaises(VisualizationError):
                state_to_latex(sv, ket_basis=basis)


if __name__ == "__main__":
    unittest.main(verbosity=2)
