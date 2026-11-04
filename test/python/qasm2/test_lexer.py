# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import ddt

import qiskit.qasm2
from test import QiskitTestCase


@ddt.ddt
class TestLexer(QiskitTestCase):
    # Most of the lexer is fully exercised in the parser tests.  These tests here are really mopping
    # up some error messages and whatnot that might otherwise be missed.

    def test_pathological_formatting(self):
        # This is deliberately _terribly_ formatted, included multiple blanks lines in quick
        # succession and comments in places you really wouldn't expect to see comments.
        program = """
            OPENQASM


            // do we really need a comment here?

            2.0//and another comment very squished up
            ;

            include // this line introduces a file import
            "qelib1.inc" // this is the file imported
            ; // this is a semicolon

            gate // we're making a gate
            bell( // void, with loose parenthesis in comment )
            ) a,//
b{h a;cx a //,,,,
,b;}

            qreg // a quantum register
            q
            [ // a square bracket




            2];bell q[0],//
q[1];creg c[2];measure q->c;"""
        parsed = qiskit.qasm2.loads(program)
        expected_unrolled = qiskit.QuantumCircuit(
            qiskit.QuantumRegister(2, "q"), qiskit.ClassicalRegister(2, "c")
        )
        expected_unrolled.h(0)
        expected_unrolled.cx(0, 1)
        expected_unrolled.measure([0, 1], [0, 1])
        self.assertEqual(parsed.decompose(), expected_unrolled)

    @ddt.data("0.25", "00.25", "2.5e-1", "2.5e-01", "0.025E+1", ".25", ".025e1", "25e-2")
    def test_float_lexes(self, number):
        program = f"qreg q[1]; U({number}, 0, 0) q[0];"
        parsed = qiskit.qasm2.loads(program)
        self.assertEqual(list(parsed.data[0].operation.params), [0.25, 0, 0])

    def test_no_decimal_float_rejected_in_strict_mode(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError,
            r"\[strict\] all floats must include a decimal point",
        ):
            qiskit.qasm2.loads("OPENQASM 2.0; qreg q[1]; U(25e-2, 0, 0) q[0];", strict=True)

    @ddt.data("", "qre", "cre", ".")
    def test_non_ascii_bytes_error(self, prefix):
        token = f"{prefix}\xff"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "encountered a non-ASCII byte"):
            qiskit.qasm2.loads(token)

    def test_integers_cannot_start_with_zero(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "integers cannot have leading zeroes"
        ):
            qiskit.qasm2.loads("0123")

    @ddt.data("", "+", "-")
    def test_float_exponents_must_have_a_digit(self, sign):
        token = f"12.34e{sign}"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "needed to see an integer exponent"
        ):
            qiskit.qasm2.loads(token)

    def test_non_builtins_cannot_be_capitalised(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "identifiers cannot start with capital"
        ):
            qiskit.qasm2.loads("Qubit")

    def test_unterminated_filename_is_invalid(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "unexpected end-of-file while lexing string literal"
        ):
            qiskit.qasm2.loads('include "qelib1.inc')

    def test_filename_with_linebreak_is_invalid(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "unexpected line break while lexing string literal"
        ):
            qiskit.qasm2.loads('include "qe\nlib1.inc";')

    def test_strict_single_quoted_path_rejected(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"\[strict\] paths must be in double quotes"
        ):
            qiskit.qasm2.loads("OPENQASM 2.0; include 'qelib1.inc';", strict=True)

    def test_version_must_have_word_boundary_after(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after a version"
        ):
            qiskit.qasm2.loads("OPENQASM 2a;")
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after a version"
        ):
            qiskit.qasm2.loads("OPENQASM 2.0a;")

    def test_no_boundary_float_in_version_position(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after a float"
        ):
            qiskit.qasm2.loads("OPENQASM .5a;")
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after a float"
        ):
            qiskit.qasm2.loads("OPENQASM 0.2e1a;")

    def test_integers_must_have_word_boundaries_after(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after an integer"
        ):
            qiskit.qasm2.loads("OPENQASM 2.0; qreg q[2a];")

    def test_floats_must_have_word_boundaries_after(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a word boundary after a float"
        ):
            qiskit.qasm2.loads("OPENQASM 2.0; qreg q[1]; U(2.0a, 0, 0) q[0];")

    def test_single_equals_is_rejected(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"single equals '=' is never valid"
        ):
            qiskit.qasm2.loads("if (a = 2) U(0, 0, 0) q[0];")

    def test_bare_dot_is_not_valid_float(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"expected a numeric fractional part"
        ):
            qiskit.qasm2.loads("qreg q[0]; U(2 + ., 0, 0) q[0];")

    def test_invalid_token(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"encountered '!', which doesn't match"
        ):
            qiskit.qasm2.loads("!")
