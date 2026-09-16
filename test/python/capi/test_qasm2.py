# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parity tests between the native OpenQASM 2 importer, reached through the C API, and the
Python-space bytecode interpreter behind :func:`qiskit.qasm2.loads`."""

import unittest

import ddt

import qiskit.qasm2
from qiskit.quantum_info import Operator
from test import QiskitTestCase

from . import ffi

HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'

# Programs built only from `qelib1.inc` and the language builtins, which both importers lower to
# the same standard gates, so the circuits must compare exactly equal.
STANDARD_PROGRAMS = [
    "qreg q[1];\nh q[0];\n",
    "qreg q[2];\nh q[0];\ncx q[0], q[1];\n",
    "qreg q[2];\ncreg c[2];\nh q[0];\ncx q[0], q[1];\nmeasure q -> c;\n",
    "qreg q[1];\nrx(0.5) q[0];\nry(-1.25) q[0];\nrz(pi / 4) q[0];\n",
    "qreg q[1];\nu3(0.1, 0.2, 0.3) q[0];\nu2(0.4, 0.5) q[0];\nu1(0.6) q[0];\n",
    "qreg q[3];\nccx q[0], q[1], q[2];\n",
    "qreg q[2];\ncrz(0.5) q[0], q[1];\ncu1(0.25) q[0], q[1];\ncu3(0.1, 0.2, 0.3) q[0], q[1];\n",
    "qreg q[2];\nbarrier q;\nh q[0];\nbarrier q[0], q[1];\n",
    "qreg q[1];\ncreg c[1];\nx q[0];\nmeasure q[0] -> c[0];\nreset q[0];\n",
    "qreg q[1];\nrx(sin(0.5) + cos(0.25) * 2) q[0];\nrz(sqrt(2.0) / ln(3.0)) q[0];\n",
    "qreg q[2];\nid q[0];\ny q[0];\nz q[0];\ns q[0];\nsdg q[0];\nt q[0];\ntdg q[0];\n"
    "cz q[0], q[1];\ncy q[0], q[1];\nch q[0], q[1];\n",
    "qreg q[2];\ncreg c[2];\nif (c == 1) cx q[0], q[1];\n",
    "qreg q[2];\ncreg c[2];\ncreg d[1];\n"
    "if (c == 2) measure q[0] -> d[0];\nif (d == 1) reset q[1];\n",
    "qreg q[2];\ncreg c[1];\nmeasure q[0] -> c[0];\nif (c == 1) x q[1];\n",
    "qreg q[1];\ncreg c[2];\nif (c == 1) measure q[0] -> c[0];\n",
    "qreg q[1];\ncreg c[2];\nif (c == 2) measure q[0] -> c[1];\n",
]

# Programs with `gate` declarations.  The native builder keeps the declared gate as a single Rust
# custom operation, where the Python interpreter builds a `_DefinedGate`.
DEFINED_GATE_PROGRAMS = [
    "gate my_h q { h q; }\nqreg q[1];\nmy_h q[0];\n",
    "gate rot(a) q { rx(a) q; rz(2 * a) q; }\nqreg q[1];\nrot(0.5) q[0];\n",
    "gate flip q0, q1 { cx q0, q1; cx q1, q0; }\nqreg q[2];\nflip q[0], q[1];\n",
]


@ddt.ddt
class TestQasm2CParity(QiskitTestCase):
    @ddt.idata(STANDARD_PROGRAMS)
    def test_matches_python_loader_exactly(self, body):
        """Standard-gate programs must import identically through both paths."""
        program = HEADER + body
        self.assertEqual(qiskit.qasm2.loads(program), ffi.load_qasm2_from_c(program))

    @ddt.idata(DEFINED_GATE_PROGRAMS)
    @unittest.expectedFailure
    def test_defined_gates_agree_on_structure_and_unitary(self, body):
        """`gate` declarations cannot round-trip through the C API yet.

        ``qk_circuit_to_python_full`` raises ``NotImplementedError: Custom operations from Rust
        cannot be exposed to Python``, so a program containing a ``gate`` declaration cannot be
        brought back into Python space at all.  The native importer itself handles these (see the
        Rust tests in ``crates/qasm2/src/build.rs`` and ``test/c/test_qasm2.c``); it is only the
        C-to-Python bridge that cannot carry them.  Even once it can, ``Instruction.__eq__``
        compares ``base_class``, so these will agree on structure and unitary rather than being
        equal to the Python loader's ``_DefinedGate``.
        """
        program = HEADER + body
        from_python = qiskit.qasm2.loads(program)
        from_c = ffi.load_qasm2_from_c(program)

        self.assertEqual(from_python.num_qubits, from_c.num_qubits)
        self.assertEqual(from_python.num_clbits, from_c.num_clbits)
        self.assertEqual(
            [instruction.operation.name for instruction in from_python.data],
            [instruction.operation.name for instruction in from_c.data],
        )
        self.assertEqual(Operator(from_python), Operator(from_c))

    def test_reports_a_parse_error(self):
        """A malformed program raises `QASM2ParseError`, as `qasm2.loads` would."""
        program = "qreg q[1];\nnot_a_gate q[0];\n"
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "not_a_gate"):
            ffi.load_qasm2_from_c(program)
        with self.assertRaisesRegex(qiskit.qasm2.QASM2ParseError, "not_a_gate"):
            qiskit.qasm2.loads(program)

    def test_strict_mode_is_honoured(self):
        """Strict mode demands the version statement; the default tolerates its absence."""
        # `U` is a language builtin, so this needs no `include`; the only thing strict mode can
        # object to here is the missing version statement.
        program = "qreg q[1];\nU(0, 0, 0) q[0];\n"
        self.assertEqual(qiskit.qasm2.loads(program), ffi.load_qasm2_from_c(program))
        with self.assertRaises(qiskit.qasm2.QASM2ParseError):
            ffi.load_qasm2_from_c(program, strict=True)
