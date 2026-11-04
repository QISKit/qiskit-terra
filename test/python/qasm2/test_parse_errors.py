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


import enum
import math

import ddt

import qiskit.qasm2
from qiskit.circuit import Gate, library as lib
from test import combine
from test import QiskitTestCase


# We need to use this enum a _bunch_ of times, so let's not give it a long name.


class T(enum.Enum):
    # This is a deliberately stripped-down list that doesn't include most of the expression-specific
    # tokens, because we don't want to complicate matters with those in tests of the general parser
    # errors.  We test the expression subparser elsewhere.
    OPENQASM = "OPENQASM"
    BARRIER = "barrier"
    CREG = "creg"
    GATE = "gate"
    IF = "if"
    INCLUDE = "include"
    MEASURE = "measure"
    OPAQUE = "opaque"
    QREG = "qreg"
    RESET = "reset"
    PI = "pi"
    ARROW = "->"
    EQUALS = "=="
    SEMICOLON = ";"
    COMMA = ","
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    LBRACE = "{"
    RBRACE = "}"
    ID = "q"
    REAL = "1.5"
    INTEGER = "1"
    FILENAME = '"qelib1.inc"'


def bad_token_parametrisation():
    """Generate the test cases for the "bad token" tests; this makes a sequence of OpenQASM 2
    statements, then puts various invalid tokens after them to verify that the parser correctly
    throws an error on them."""

    token_set = frozenset(T)

    def without(*tokens):
        return token_set - set(tokens)

    # ddt isn't a particularly great parametriser - it'll only correctly unpack tuples and lists in
    # the way we really want, but if we want to control the test id, we also have to set `__name__`
    # which isn't settable on either of those.  We can't use unpack, then, so we just need a class
    # to pass.
    class BadTokenCase:
        def __init__(self, statement, disallowed, name=None):
            self.statement = statement
            self.disallowed = disallowed
            self.__name__ = name

    for statement, disallowed in [
        # This should only include stopping points where the next token is somewhat fixed; in
        # places where there's a real decision to be made (such as number of qubits in a gate,
        # or the statement type in a gate body), there should be a better error message.
        #
        # There's a large subset of OQ2 that's reducible to a regular language, so we _could_
        # define that, build a DFA for it, and use that to very quickly generate a complete set
        # of tests.  That would be more complex to read and verify for correctness, though.
        (
            "",
            without(
                T.OPENQASM,
                T.ID,
                T.INCLUDE,
                T.OPAQUE,
                T.GATE,
                T.QREG,
                T.CREG,
                T.IF,
                T.RESET,
                T.BARRIER,
                T.MEASURE,
                T.SEMICOLON,
            ),
        ),
        ("OPENQASM", without(T.REAL, T.INTEGER)),
        ("OPENQASM 2.0", without(T.SEMICOLON)),
        ("include", without(T.FILENAME)),
        ('include "qelib1.inc"', without(T.SEMICOLON)),
        ("opaque", without(T.ID)),
        ("opaque bell", without(T.LPAREN, T.ID, T.SEMICOLON)),
        ("opaque bell (", without(T.ID, T.RPAREN)),
        ("opaque bell (a", without(T.COMMA, T.RPAREN)),
        ("opaque bell (a,", without(T.ID, T.RPAREN)),
        ("opaque bell (a, b", without(T.COMMA, T.RPAREN)),
        ("opaque bell (a, b)", without(T.ID, T.SEMICOLON)),
        ("opaque bell (a, b) q1", without(T.COMMA, T.SEMICOLON)),
        ("opaque bell (a, b) q1,", without(T.ID, T.SEMICOLON)),
        ("opaque bell (a, b) q1, q2", without(T.COMMA, T.SEMICOLON)),
        ("gate", without(T.ID)),
        ("gate bell (", without(T.ID, T.RPAREN)),
        ("gate bell (a", without(T.COMMA, T.RPAREN)),
        ("gate bell (a,", without(T.ID, T.RPAREN)),
        ("gate bell (a, b", without(T.COMMA, T.RPAREN)),
        ("gate bell (a, b) q1", without(T.COMMA, T.LBRACE)),
        ("gate bell (a, b) q1,", without(T.ID, T.LBRACE)),
        ("gate bell (a, b) q1, q2", without(T.COMMA, T.LBRACE)),
        ("qreg", without(T.ID)),
        ("qreg reg", without(T.LBRACKET)),
        ("qreg reg[", without(T.INTEGER)),
        ("qreg reg[5", without(T.RBRACKET)),
        ("qreg reg[5]", without(T.SEMICOLON)),
        ("creg", without(T.ID)),
        ("creg reg", without(T.LBRACKET)),
        ("creg reg[", without(T.INTEGER)),
        ("creg reg[5", without(T.RBRACKET)),
        ("creg reg[5]", without(T.SEMICOLON)),
        ("CX", without(T.LPAREN, T.ID, T.SEMICOLON)),
        ("CX(", without(T.PI, T.INTEGER, T.REAL, T.ID, T.LPAREN, T.RPAREN)),
        ("CX()", without(T.ID, T.SEMICOLON)),
        ("CX q", without(T.LBRACKET, T.COMMA, T.SEMICOLON)),
        ("CX q[", without(T.INTEGER)),
        ("CX q[0", without(T.RBRACKET)),
        ("CX q[0]", without(T.COMMA, T.SEMICOLON)),
        ("CX q[0],", without(T.ID, T.SEMICOLON)),
        ("CX q[0], q", without(T.LBRACKET, T.COMMA, T.SEMICOLON)),
        # No need to repeatedly "every" possible number of arguments.
        ("measure", without(T.ID)),
        ("measure q", without(T.LBRACKET, T.ARROW)),
        ("measure q[", without(T.INTEGER)),
        ("measure q[0", without(T.RBRACKET)),
        ("measure q[0]", without(T.ARROW)),
        ("measure q[0] ->", without(T.ID)),
        ("measure q[0] -> c", without(T.LBRACKET, T.SEMICOLON)),
        ("measure q[0] -> c[", without(T.INTEGER)),
        ("measure q[0] -> c[0", without(T.RBRACKET)),
        ("measure q[0] -> c[0]", without(T.SEMICOLON)),
        ("reset", without(T.ID)),
        ("reset q", without(T.LBRACKET, T.SEMICOLON)),
        ("reset q[", without(T.INTEGER)),
        ("reset q[0", without(T.RBRACKET)),
        ("reset q[0]", without(T.SEMICOLON)),
        ("barrier", without(T.ID, T.SEMICOLON)),
        ("barrier q", without(T.LBRACKET, T.COMMA, T.SEMICOLON)),
        ("barrier q[", without(T.INTEGER)),
        ("barrier q[0", without(T.RBRACKET)),
        ("barrier q[0]", without(T.COMMA, T.SEMICOLON)),
        ("if", without(T.LPAREN)),
        ("if (", without(T.ID)),
        ("if (cond", without(T.EQUALS)),
        ("if (cond ==", without(T.INTEGER)),
        ("if (cond == 0", without(T.RPAREN)),
        ("if (cond == 0)", without(T.ID, T.RESET, T.MEASURE)),
    ]:
        for token in disallowed:
            yield BadTokenCase(statement, token.value, name=f"'{statement}'-{token.name.lower()}")


def eof_parametrisation():
    for tokens in [
        ("OPENQASM", "2.0", ";"),
        ("include", '"qelib1.inc"', ";"),
        ("opaque", "bell", "(", "a", ",", "b", ")", "q1", ",", "q2", ";"),
        ("gate", "bell", "(", "a", ",", "b", ")", "q1", ",", "q2", "{", "}"),
        ("qreg", "qr", "[", "5", "]", ";"),
        ("creg", "cr", "[", "5", "]", ";"),
        ("CX", "(", ")", "q", "[", "0", "]", ",", "q", "[", "1", "]", ";"),
        ("measure", "q", "[", "0", "]", "->", "c", "[", "0", "]", ";"),
        ("reset", "q", "[", "0", "]", ";"),
        ("barrier", "q", ";"),
        # No need to test every combination of `if`, really.
        ("if", "(", "cond", "==", "0", ")", "CX q[0], q[1];"),
    ]:
        prefix = ""
        for token in tokens[:-1]:
            prefix = f"{prefix} {token}".strip()
            yield prefix


@ddt.ddt
class TestIncompleteStructure(QiskitTestCase):
    PRELUDE = "OPENQASM 2.0; qreg q[5]; creg c[5]; creg cond[1];"

    @ddt.idata(bad_token_parametrisation())
    def test_bad_token(self, case):
        """Test that the parser raises an error when an incorrect token is given."""
        statement = case.statement
        disallowed = case.disallowed

        prelude = "" if statement.startswith("OPENQASM") else self.PRELUDE
        full = f"{prelude} {statement} {disallowed}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "needed .*, but instead"):
            qiskit.qasm2.loads(full)

    @ddt.idata(eof_parametrisation())
    def test_eof(self, statement):
        """Test that the parser raises an error when the end-of-file is reached instead of a token
        that is required."""
        prelude = "" if statement.startswith("OPENQASM") else self.PRELUDE
        full = f"{prelude} {statement}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "unexpected end-of-file"):
            qiskit.qasm2.loads(full)

    def test_loading_directory(self):
        """Test that the correct error is raised when a file fails to open."""
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "failed to read"):
            qiskit.qasm2.load(".")


class TestVersion(QiskitTestCase):
    def test_invalid_version(self):
        program = "OPENQASM 3.0;"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "can only handle OpenQASM 2.0"):
            qiskit.qasm2.loads(program)

        program = "OPENQASM 2.1;"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "can only handle OpenQASM 2.0"):
            qiskit.qasm2.loads(program)

        program = "OPENQASM 20.e-1;"
        with self.assertRaises(qiskit.qasm2.QASM2ParseError):
            qiskit.qasm2.loads(program)

    def test_openqasm_must_be_first_statement(self):
        program = "qreg q[0]; OPENQASM 2.0;"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "only the first statement"):
            qiskit.qasm2.loads(program)


@ddt.ddt
class TestScoping(QiskitTestCase):
    def test_register_use_before_definition(self):
        program = "CX after[0], after[1]; qreg after[2];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "not defined in this scope"):
            qiskit.qasm2.loads(program)

        program = "qreg q[2]; measure q[0] -> c[0]; creg c[2];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "not defined in this scope"):
            qiskit.qasm2.loads(program)

    @combine(
        definer=["qreg reg[2];", "creg reg[2];", "gate reg a {}", "opaque reg a;"],
        bad_definer=["qreg reg[2];", "creg reg[2];"],
    )
    def test_register_already_defined(self, definer, bad_definer):
        program = f"{definer} {bad_definer}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    def test_qelib1_not_implicit(self):
        program = """
            OPENQASM 2.0;
            qreg q[2];
            cx q[0], q[1];
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'cx' is not defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_access_gates_before_definition(self):
        program = """
            qreg q[2];
            cx q[0], q[1];
            gate cx a, b {
                CX a, b;
            }
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'cx' is not defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_access_gate_recursively(self):
        program = """
            gate cx a, b {
                cx a, b;
            }
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'cx' is not defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_access_qubits_from_previous_gate(self):
        program = """
            gate cx a, b {
                CX a, b;
            }
            gate other c {
                CX a, b;
            }
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'a' is not defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_access_parameters_from_previous_gate(self):
        program = """
            gate first(a, b) q {
                U(a, 0, b) q;
            }
            gate second q {
                U(a, 0, b) q;
            }
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "'a' is not a parameter.*defined"
        ):
            qiskit.qasm2.loads(program)

    def test_cannot_access_quantum_registers_within_gate(self):
        program = """
            qreg q[2];
            gate my_gate a {
                CX a, q;
            }
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'q' is a quantum register"):
            qiskit.qasm2.loads(program)

    def test_parameters_not_defined_outside_gate(self):
        program = """
            gate my_gate(a) q {}
            qreg qr[2];
            U(a, 0, 0) qr;
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "'a' is not a parameter.*defined"
        ):
            qiskit.qasm2.loads(program)

    def test_qubits_not_defined_outside_gate(self):
        program = """
            gate my_gate(a) q {}
            U(0, 0, 0) q;
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'q' is not defined"):
            qiskit.qasm2.loads(program)

    @ddt.data('include "qelib1.inc";', "gate h q { }")
    def test_gates_cannot_redefine(self, definer):
        program = f"{definer} gate h q {{ }}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_use_undeclared_register_conditional(self):
        program = "qreg q[1]; if (c == 0) U(0, 0, 0) q[0];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "not defined"):
            qiskit.qasm2.loads(program)


@ddt.ddt
class TestTyping(QiskitTestCase):
    @ddt.data(
        "CX q[0], U;",
        "measure U -> c[0];",
        "measure q[0] -> U;",
        "reset U;",
        "barrier U;",
        "if (U == 0) CX q[0], q[1];",
        "gate my_gate a { U(0, 0, 0) U; }",
    )
    def test_cannot_use_gates_incorrectly(self, usage):
        program = f"qreg q[2]; creg c[2]; {usage}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'U' is a gate"):
            qiskit.qasm2.loads(program)

    @ddt.data(
        "measure q[0] -> q[1];",
        "if (q == 0) CX q[0], q[1];",
        "q q[0], q[1];",
        "gate my_gate a { U(0, 0, 0) q; }",
    )
    def test_cannot_use_qregs_incorrectly(self, usage):
        program = f"qreg q[2]; creg c[2]; {usage}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'q' is a quantum register"):
            qiskit.qasm2.loads(program)

    @ddt.data(
        "CX q[0], c[1];",
        "measure c[0] -> c[1];",
        "reset c[0];",
        "barrier c[0];",
        "c q[0], q[1];",
        "gate my_gate a { U(0, 0, 0) c; }",
    )
    def test_cannot_use_cregs_incorrectly(self, usage):
        program = f"qreg q[2]; creg c[2]; {usage}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'c' is a classical register"):
            qiskit.qasm2.loads(program)

    def test_cannot_use_parameters_incorrectly(self):
        program = "gate my_gate(p) q { CX p, q; }"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'p' is a parameter"):
            qiskit.qasm2.loads(program)

    def test_cannot_use_qubits_incorrectly(self):
        program = "gate my_gate(p) q { U(q, q, q) q; }"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'q' is a gate qubit"):
            qiskit.qasm2.loads(program)

    @ddt.data(("h", 0), ("h", 2), ("CX", 0), ("CX", 1), ("CX", 3), ("ccx", 2), ("ccx", 4))
    @ddt.unpack
    def test_gates_accept_only_valid_number_qubits(self, gate, bad_count):
        arguments = ", ".join(f"q[{i}]" for i in range(bad_count))
        program = f'include "qelib1.inc"; qreg q[5];\n{gate} {arguments};'
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "takes .* quantum arguments?"):
            qiskit.qasm2.loads(program)

    @ddt.data(("U", 2), ("U", 4), ("rx", 0), ("rx", 2), ("u3", 1))
    @ddt.unpack
    def test_gates_accept_only_valid_number_parameters(self, gate, bad_count):
        arguments = ", ".join("0" for _ in [None] * bad_count)
        program = f'include "qelib1.inc"; qreg q[5];\n{gate}({arguments}) q[0];'
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "takes .* parameters?"):
            qiskit.qasm2.loads(program)


@ddt.ddt
class TestGateDefinition(QiskitTestCase):
    def test_no_zero_qubit(self):
        program = "gate zero {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "gates must act on at least one"):
            qiskit.qasm2.loads(program)

        program = "gate zero(a) {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "gates must act on at least one"):
            qiskit.qasm2.loads(program)

    def test_no_zero_qubit_opaque(self):
        program = "opaque zero;"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "gates must act on at least one"):
            qiskit.qasm2.loads(program)

        program = "opaque zero(a);"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "gates must act on at least one"):
            qiskit.qasm2.loads(program)

    def test_cannot_subscript_qubit(self):
        program = """
            gate my_gate a {
                CX a[0], a[1];
            }
        """
        with self.assertRaises(qiskit.qasm2.QASM2ParseError):
            qiskit.qasm2.loads(program)

    def test_cannot_repeat_parameters(self):
        program = "gate my_gate(a, a) q {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_repeat_qubits(self):
        program = "gate my_gate a, a {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    def test_qubit_cannot_shadow_parameter(self):
        program = "gate my_gate(a) a {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    @ddt.data("measure q -> c;", "reset q", "if (c == 0) U(0, 0, 0) q;", "gate my_x q {}")
    def test_definition_cannot_contain_nonunitary(self, statement):
        program = f"OPENQASM 2.0; creg c[5]; gate my_gate q {{ {statement} }}"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "only gate applications are valid"
        ):
            qiskit.qasm2.loads(program)

    def test_cannot_redefine_u(self):
        program = "gate U(a, b, c) q {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)

    def test_cannot_redefine_cx(self):
        program = "gate CX a, b {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(program)


@ddt.ddt
class TestBitResolution(QiskitTestCase):
    def test_disallow_out_of_range(self):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "out-of-range"):
            qiskit.qasm2.loads("qreg q[2]; U(0, 0, 0) q[2];")

        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "out-of-range"):
            qiskit.qasm2.loads("qreg q[2]; creg c[2]; measure q[2] -> c[0];")

        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "out-of-range"):
            qiskit.qasm2.loads("qreg q[2]; creg c[2]; measure q[0] -> c[2];")

    @combine(
        conditional=[True, False],
        call=[
            "CX q1[0], q1[0];",
            "CX q1, q1[0];",
            "CX q1[0], q1;",
            "CX q1, q1;",
            "ccx q1[0], q1[1], q1[0];",
            "ccx q2, q1, q2[0];",
        ],
    )
    def test_disallow_duplicate_qubits(self, call, conditional):
        program = """
            include "qelib1.inc";
            qreg q1[3];
            qreg q2[3];
            qreg q3[3];
        """
        if conditional:
            program += "creg cond[1]; if (cond == 0) "
        program += call
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "duplicate qubit"):
            qiskit.qasm2.loads(program)

    @ddt.data(
        (("q1[1]", "q2[2]"), "CX q1, q2"),
        (("q1[1]", "q2[2]"), "CX q2, q1"),
        (("q1[3]", "q2[2]"), "CX q1, q2"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q1, q2, q3"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q2, q3, q1"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q3, q1, q2"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q1, q2[0], q3"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q2[0], q3, q1"),
        (("q1[2]", "q2[3]", "q3[3]"), "ccx q3, q1, q2[0]"),
    )
    @ddt.unpack
    def test_incorrect_gate_broadcast_lengths(self, registers, call):
        setup = 'include "qelib1.inc";\n' + "\n".join(f"qreg {reg};" for reg in registers)
        program = f"{setup}\n{call};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "cannot resolve broadcast"):
            qiskit.qasm2.loads(program)

        cond = "creg cond[1];\nif (cond == 0)"
        program = f"{setup}\n{cond} {call};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "cannot resolve broadcast"):
            qiskit.qasm2.loads(program)

    @ddt.data(
        ("qreg q[2]; creg c[2];", "q[0] -> c"),
        ("qreg q[2]; creg c[2];", "q -> c[0]"),
        ("qreg q[1]; creg c[2];", "q -> c[0]"),
        ("qreg q[2]; creg c[1];", "q[0] -> c"),
        ("qreg q[2]; creg c[3];", "q -> c"),
    )
    @ddt.unpack
    def test_incorrect_measure_broadcast_lengths(self, setup, operands):
        program = f"{setup}\nmeasure {operands};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "cannot resolve broadcast"):
            qiskit.qasm2.loads(program)

        program = f"{setup}\ncreg cond[1];\nif (cond == 0) measure {operands};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "cannot resolve broadcast"):
            qiskit.qasm2.loads(program)


@ddt.ddt
class TestCustomInstructions(QiskitTestCase):
    def test_cannot_use_custom_before_definition(self):
        program = "qreg q[2]; my_gate q[0], q[1];"

        class MyGate(Gate):
            def __init__(self):
                super().__init__("my_gate", 2, [])

        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "cannot use .* before definition"
        ):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[qiskit.qasm2.CustomInstruction("my_gate", 0, 2, MyGate)],
            )

    def test_cannot_misdefine_u(self):
        program = "qreg q[1]; U(0.5, 0.25) q[0]"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "custom instruction .* mismatched"
        ):
            qiskit.qasm2.loads(
                program, custom_instructions=[qiskit.qasm2.CustomInstruction("U", 2, 1, lib.U2Gate)]
            )

    def test_cannot_misdefine_cx(self):
        program = "qreg q[1]; CX q[0]"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "custom instruction .* mismatched"
        ):
            qiskit.qasm2.loads(
                program, custom_instructions=[qiskit.qasm2.CustomInstruction("CX", 0, 1, lib.XGate)]
            )

    def test_builtin_is_typechecked(self):
        program = "qreg q[1]; my(0.5) q[0];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'my' takes 2 quantum arguments"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("my", 1, 2, lib.RXXGate, builtin=True)
                ],
            )
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'my' takes 2 parameters"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("my", 2, 1, lib.U2Gate, builtin=True)
                ],
            )

    def test_cannot_define_builtin_twice(self):
        program = "gate builtin q {}; gate builtin q {};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'builtin' is already defined"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("builtin", 0, 1, lambda: Gate("builtin", 1, []))
                ],
            )

    def test_cannot_redefine_custom_u(self):
        program = "gate U(a, b, c) q {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("U", 3, 1, lib.UGate, builtin=True)
                ],
            )

    def test_cannot_redefine_custom_cx(self):
        program = "gate CX a, b {}"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "already defined"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("CX", 0, 2, lib.CXGate, builtin=True)
                ],
            )

    @combine(
        program=["gate my(a) q {}", "opaque my(a) q;"],
        builtin=[True, False],
    )
    def test_custom_definition_must_match_gate(self, program, builtin):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'my' is mismatched"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("my", 1, 2, lib.RXXGate, builtin=builtin)
                ],
            )
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "'my' is mismatched"):
            qiskit.qasm2.loads(
                program,
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("my", 2, 1, lib.U2Gate, builtin=builtin)
                ],
            )

    def test_cannot_have_duplicate_customs(self):
        customs = [
            qiskit.qasm2.CustomInstruction("my", 1, 2, lib.RXXGate),
            qiskit.qasm2.CustomInstruction("x", 0, 1, lib.XGate),
            qiskit.qasm2.CustomInstruction("my", 1, 2, lib.RZZGate),
        ]
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "duplicate custom instruction"):
            qiskit.qasm2.loads("", custom_instructions=customs)

    def test_qiskit_delay_float_input_wraps_exception(self):
        program = "opaque delay(t) q; qreg q[1]; delay(1.5) q[0];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "can only accept an integer"):
            qiskit.qasm2.loads(program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)

    def test_u0_float_input_wraps_exception(self):
        program = "opaque u0(n) q; qreg q[1]; u0(1.1) q[0];"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "must be an integer"):
            qiskit.qasm2.loads(program, custom_instructions=qiskit.qasm2.LEGACY_CUSTOM_INSTRUCTIONS)


@ddt.ddt
class TestCustomClassical(QiskitTestCase):
    @ddt.data("cos", "exp", "sin", "sqrt", "tan", "ln")
    def test_cannot_override_builtin(self, builtin):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"cannot override builtin"):
            qiskit.qasm2.loads(
                "",
                custom_classical=[qiskit.qasm2.CustomClassical(builtin, 1, math.exp)],
            )

    def test_duplicate_names_disallowed(self):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"duplicate custom classical"):
            qiskit.qasm2.loads(
                "",
                custom_classical=[
                    qiskit.qasm2.CustomClassical("f", 1, math.exp),
                    qiskit.qasm2.CustomClassical("f", 1, math.exp),
                ],
            )

    def test_cannot_shadow_custom_instruction(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"custom classical.*naming clash"
        ):
            qiskit.qasm2.loads(
                "",
                custom_instructions=[
                    qiskit.qasm2.CustomInstruction("f", 0, 1, lib.RXGate, builtin=True)
                ],
                custom_classical=[qiskit.qasm2.CustomClassical("f", 1, math.exp)],
            )

    def test_cannot_shadow_builtin_instruction(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"custom classical.*cannot shadow"
        ):
            qiskit.qasm2.loads(
                "",
                custom_classical=[qiskit.qasm2.CustomClassical("U", 1, math.exp)],
            )

    def test_cannot_shadow_with_gate_definition(self):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"'f' is already defined"):
            qiskit.qasm2.loads(
                "gate f q {}",
                custom_classical=[qiskit.qasm2.CustomClassical("f", 1, math.exp)],
            )

    @ddt.data("qreg", "creg")
    def test_cannot_shadow_with_register_definition(self, regtype):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"'f' is already defined"):
            qiskit.qasm2.loads(
                f"{regtype} f[2];",
                custom_classical=[qiskit.qasm2.CustomClassical("f", 1, math.exp)],
            )

    @ddt.data((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    @ddt.unpack
    def test_mismatched_argument_count(self, n_good, n_bad):
        arg_string = ", ".join(["0" for _ in [None] * n_bad])
        program = f"""
            qreg q[1];
            U(f({arg_string}), 0, 0) q[0];
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"custom function argument-count mismatch"
        ):
            qiskit.qasm2.loads(
                program, custom_classical=[qiskit.qasm2.CustomClassical("f", n_good, lambda *_: 0)]
            )

    def test_output_type_error_is_caught(self):
        program = """
            qreg q[1];
            U(f(), 0, 0) q[0];
        """
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"user.*returned non-float"):
            qiskit.qasm2.loads(
                program,
                custom_classical=[qiskit.qasm2.CustomClassical("f", 0, lambda: "not a float")],
            )

    def test_inner_exception_is_wrapped(self):
        inner_exception = Exception("custom exception")

        def raises():
            raise inner_exception

        program = """
            qreg q[1];
            U(raises(), 0, 0) q[0];
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "caught exception when constant folding"
        ) as excinfo:
            qiskit.qasm2.loads(
                program, custom_classical=[qiskit.qasm2.CustomClassical("raises", 0, raises)]
            )
        self.assertIs(excinfo.exception.__cause__, inner_exception)

    def test_cannot_be_used_as_gate(self):
        program = """
            qreg q[1];
            f(0) q[0];
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"'f' is a custom classical function"
        ):
            qiskit.qasm2.loads(
                program, custom_classical=[qiskit.qasm2.CustomClassical("f", 1, lambda x: x)]
            )

    def test_cannot_be_used_as_qarg(self):
        program = """
            U(0, 0, 0) f;
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"'f' is a custom classical function"
        ):
            qiskit.qasm2.loads(
                program, custom_classical=[qiskit.qasm2.CustomClassical("f", 1, lambda x: x)]
            )

    def test_cannot_be_used_as_carg(self):
        program = """
            qreg q[1];
            measure q[0] -> f;
        """
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"'f' is a custom classical function"
        ):
            qiskit.qasm2.loads(
                program, custom_classical=[qiskit.qasm2.CustomClassical("f", 1, lambda x: x)]
            )

    def test_runtime_failure_in_gate_body_raises_qasm2_parse_error(self):
        def always_raises(*_):
            raise RuntimeError("expected failure")

        program = """
            OPENQASM 2.0;
            gate mygate(a) q { U(bad(a), 0, 0) q; }
            qreg q[1];
            mygate(0.5) q[0];
        """
        # loads() succeeds: the gate body is stored as an expression tree, not evaluated yet.
        circuit = qiskit.qasm2.loads(
            program,
            custom_classical=[qiskit.qasm2.CustomClassical("bad", 1, always_raises)],
        )
        # The exception is raised here, when the Rust evaluator calls the custom function.
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, "caught exception when constant folding"
        ):
            _ = circuit.data[0].operation.definition

    def test_runtime_non_float_in_gate_body_raises_qasm2_parse_error(self):
        program = """
            OPENQASM 2.0;
            gate mygate(a) q { U(stringify(a), 0, 0) q; }
            qreg q[1];
            mygate(0.5) q[0];
        """
        circuit = qiskit.qasm2.loads(
            program,
            custom_classical=[qiskit.qasm2.CustomClassical("stringify", 1, str)],
        )
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "non-float"):
            _ = circuit.data[0].operation.definition


@ddt.ddt
class TestStrict(QiskitTestCase):
    @ddt.data(
        "gate my_gate(p0, p1,) q0, q1 {}",
        "gate my_gate(p0, p1) q0, q1, {}",
        "opaque my_gate(p0, p1,) q0, q1;",
        "opaque my_gate(p0, p1) q0, q1,;",
        'include "qelib1.inc"; qreg q[2]; cu3(0.5, 0.25, 0.125,) q[0], q[1];',
        'include "qelib1.inc"; qreg q[2]; cu3(0.5, 0.25, 0.125) q[0], q[1],;',
        "qreg q[2]; barrier q[0], q[1],;",
        'include "qelib1.inc"; qreg q[1]; rx(sin(pi,)) q[0];',
    )
    def test_trailing_comma(self, program):
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"\[strict\] .*trailing comma"):
            qiskit.qasm2.loads("OPENQASM 2.0;\n" + program, strict=True)

    def test_trailing_semicolon_after_gate(self):
        program = "OPENQASM 2.0; gate my_gate q {};"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"\[strict\] .*extra semicolon"):
            qiskit.qasm2.loads(program, strict=True)

    def test_empty_statement(self):
        program = "OPENQASM 2.0; ;"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, r"\[strict\] .*empty statement"):
            qiskit.qasm2.loads(program, strict=True)

    def test_required_version_regular(self):
        program = "qreg q[1];"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"\[strict\] the first statement"
        ):
            qiskit.qasm2.loads(program, strict=True)

    def test_required_version_empty(self):
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"\[strict\] .*needed a version statement"
        ):
            qiskit.qasm2.loads("", strict=True)

    def test_barrier_requires_args(self):
        program = "OPENQASM 2.0; qreg q[2]; barrier;"
        with self.assertRaisesRegex(
            qiskit.qasm2.QASM2ParseError, r"\[strict\] barrier statements must have at least one"
        ):
            qiskit.qasm2.loads(program, strict=True)
