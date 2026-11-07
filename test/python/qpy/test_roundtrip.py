# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for python write/rust read flow and vice versa"""

import io
import uuid

from ddt import ddt, idata, unpack

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Duration
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.random import random_circuit
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.quantum_info import SparsePauliOp, SparseObservable
from qiskit.circuit.classical import expr, types
from qiskit.synthesis import LieTrotter
from qiskit.qpy.common import QPY_RUST_READ_MIN_VERSION, QPY_RUST_WRITE_MIN_VERSION, QPY_VERSION
from qiskit.qpy import dump, load
from qiskit.qpy import UnsupportedFeatureForVersion
from test import QiskitTestCase
from unittest.mock import patch


def all_qpy_combinations(min_version):
    def wrapper(func):
        return idata(
            (version, write_with, read_with)
            for version in range(min_version, QPY_VERSION + 1)
            for write_with in (
                ("Python", "Rust") if version >= QPY_RUST_WRITE_MIN_VERSION else ("Python",)
            )
            for read_with in (
                ("Python", "Rust") if version >= QPY_RUST_READ_MIN_VERSION else ("Python",)
            )
            # Python writer/reader are not supported for v >= QPY_RUST_WRITE_MIN_VERSION
            if not (version >= QPY_RUST_WRITE_MIN_VERSION and write_with == "Python")
            if not (version >= QPY_RUST_WRITE_MIN_VERSION and read_with == "Python")
        )(unpack(func))

    return wrapper


@ddt
class TestQPYRoundtrip(QiskitTestCase):
    """QPY circuit testing platform."""

    def assert_roundtrip_equal(
        self,
        circuit,
        version,
        write_with,
        read_with,
        annotation_factories=None,
    ):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        use_rust_for_write = write_with == "Rust"
        use_rust_for_read = read_with == "Rust"
        # To ensure we use python for writing, we patch the QPY_RUST_WRITE_MIN_VERSION to be higher than the version we are testing.
        with patch(
            "qiskit.qpy.common.QPY_RUST_WRITE_MIN_VERSION",
            QPY_RUST_WRITE_MIN_VERSION if use_rust_for_write else QPY_VERSION + 1,
        ):
            dump(circuit, qpy_file, version=version, annotation_factories=annotation_factories)
        qpy_file.seek(0)
        # To ensure we use python for reading, we patch the QPY_RUST_READ_MIN_VERSION to be higher than the version we are testing.
        with patch(
            "qiskit.qpy.common.QPY_RUST_READ_MIN_VERSION",
            QPY_RUST_READ_MIN_VERSION if use_rust_for_read else QPY_VERSION + 1,
        ):
            new_circuit = load(qpy_file, annotation_factories=annotation_factories)[0]
        self.assertEqual(circuit, new_circuit)
        self.assertEqual(circuit.layout, new_circuit.layout)
        self.assertEqual(circuit.parameters, new_circuit.parameters)

    def assert_dump_raises_unsupported(self, circuit, version, write_with):
        """Assert dump() rejects the circuit for the given writer backend."""
        use_rust_for_write = write_with == "Rust"
        with patch(
            "qiskit.qpy.common.QPY_RUST_WRITE_MIN_VERSION",
            QPY_RUST_WRITE_MIN_VERSION if use_rust_for_write else QPY_VERSION + 1,
        ):
            with io.BytesIO() as fptr, self.assertRaises(UnsupportedFeatureForVersion):
                dump(circuit, fptr, version=version)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_simple(self, version, write_with, read_with):
        """Basic roundtrip test"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_ifelse(self, version, write_with, read_with):
        """Check the IfElse control flow gate passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        condition = (qc.cregs[0], 0)
        body = QuantumCircuit([qc.qubits[0]])
        body.x(0)
        false_body = QuantumCircuit([qc.qubits[1]])
        false_body.y(0)
        qc.if_else(condition, body, false_body, [qc.qubits[0]], [])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_box(self, version, write_with, read_with):
        """Check the BoxOp control flow gate passes roundtrip"""
        qc = QuantumCircuit(2)
        with qc.box(duration=13):
            qc.cx(0, 1)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

        # test with Expression duration
        qc = QuantumCircuit(2)
        a = qc.add_stretch("a")
        with qc.box(duration=expr.mul(2, a)):
            qc.cx(0, 1)
        if version < 14:
            self.assert_dump_raises_unsupported(qc, version=version, write_with=write_with)
        else:
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_forloop(self, version, write_with, read_with):
        """Check the ForLoop control flow gate passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

        qc = QuantumCircuit(2, 1)
        with qc.for_loop((1, 4)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_nested_for_loop(self, version, write_with, read_with):
        """Check the nested ForLoop control flow gate passes roundtrip"""
        qc = QuantumCircuit(6, 6)
        with qc.if_test(expr.equal(qc.cregs[0], 1)) as else_:
            qc.cx(0, 1)
            qc.cz(0, 2)
            qc.cz(0, 3)
        with else_:
            qc.cz(1, 4)
            with qc.for_loop((1, 2)):
                qc.cx(1, 5)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_switch(self, version, write_with, read_with):
        """Check the SwitchOp control flow gate passes roundtrip"""
        body = QuantumCircuit(1)
        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy()), (2, body.copy())], [0], [])
        qc.switch(expr.logic_not(qc.clbits[0]), [(False, body.copy())], [0], [])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_for_loop_with_range(self, version, write_with, read_with):
        """Check the ForLoop control flow gate with Range expression passes roundtrip"""
        qc = QuantumCircuit(2, 1)
        start = expr.lift(0, types.Uint(8))
        stop = expr.lift(5, types.Uint(8))
        step = expr.lift(1, types.Uint(8))
        range_expr = expr.Range(start, stop, step)

        with qc.for_loop(range_expr):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        if version < 18:
            self.assert_dump_raises_unsupported(qc, version=version, write_with=write_with)
        else:
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(18)
    def test_for_loop_with_var_loop_parameter(self, version, write_with, read_with):
        """For-loop expr.Var loop parameters round-trip faithfully in QPY v18+."""
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(QuantumRegister(1), cr)
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(6, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        with qc.for_loop(range_expr) as loop_var:
            qc.store(expr.index(cr, loop_var), expr.lift(True))

        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(18)
    def test_for_loop_with_explicit_unused_var_loop_parameter(self, version, write_with, read_with):
        """Explicit loop Vars are preserved even when unused in the body."""
        loop_var = expr.Var.new("i", types.Uint(8))
        qc = QuantumCircuit(1)
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)), expr.lift(1, types.Uint(8))
        )
        with qc.for_loop(range_expr, loop_var):
            qc.h(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(18)
    def test_for_loop_with_dynamic_range_and_var_loop_parameter(
        self, version, write_with, read_with
    ):
        """Dynamic expr.Range for-loops with loop Vars round-trip in QPY v18+."""
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(QuantumRegister(1), cr)
        start_var = qc.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = qc.add_var("stop", expr.lift(6, types.Uint(8)))
        range_expr = expr.Range(start_var, stop_var)
        with qc.for_loop(range_expr) as loop_var:
            qc.store(expr.index(cr, loop_var), expr.lift(True))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(18)
    def test_nested_for_loop_with_expr_range(self, version, write_with, read_with):
        """Nested for-loops with expr.Range indexsets round-trip at QPY v18+."""
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(QuantumRegister(2), cr)
        outer = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(2, types.Uint(8)))
        inner = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(3, types.Uint(8)))
        with qc.for_loop(outer) as outer_var:
            qc.store(expr.index(cr, outer_var), expr.lift(True))
            with qc.for_loop(inner) as inner_var:
                qc.h(0)
                qc.store(expr.index(cr, inner_var), expr.lift(True))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    def test_for_loop_expr_range_unsupported_below_v18(self):
        """expr.Range for-loops cannot be dumped with QPY format versions below 18."""
        from qiskit.circuit.controlflow import ForLoopOp

        cr = ClassicalRegister(16, "cr")
        constant_range = QuantumCircuit(QuantumRegister(1), cr)
        range_expr = expr.Range(
            expr.lift(0, types.Uint(8)), expr.lift(6, types.Uint(8)), expr.lift(2, types.Uint(8))
        )
        with constant_range.for_loop(range_expr) as loop_var:
            constant_range.store(expr.index(cr, loop_var), expr.lift(True))

        dynamic_range = QuantumCircuit(QuantumRegister(1), ClassicalRegister(16, "cr"))
        start_var = dynamic_range.add_var("start", expr.lift(0, types.Uint(8)))
        stop_var = dynamic_range.add_var("stop", expr.lift(6, types.Uint(8)))
        dynamic_range_expr = expr.Range(start_var, stop_var)
        with dynamic_range.for_loop(dynamic_range_expr) as loop_var:
            dynamic_range.store(expr.index(dynamic_range.cregs[0], loop_var), expr.lift(True))

        explicit_loop_var = QuantumCircuit(1)
        loop_var = expr.Var.new("i", types.Uint(8))
        with explicit_loop_var.for_loop(
            expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8))), loop_var
        ):
            explicit_loop_var.h(0)

        none_loop_param = QuantumCircuit(1)
        body = QuantumCircuit(1)
        body.h(0)
        none_loop_param.append(
            ForLoopOp(
                expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8))), None, body
            ),
            [0],
        )

        circuits = (constant_range, dynamic_range, explicit_loop_var, none_loop_param)

        for qc in circuits:
            for version in (16, 17):
                writers = (
                    ("Python", "Rust") if version >= QPY_RUST_WRITE_MIN_VERSION else ("Python",)
                )
                for write_with in writers:
                    self.assert_dump_raises_unsupported(qc, version=version, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_for_loop_with_expr_range_and_none_loop_parameter(self, version, write_with, read_with):
        """ForLoopOp with expr.Range indexset and explicit None loop parameter round-trips.

        When the user opts out of a loop variable (param[1] is None), the ForLoopOp
        still serialises and deserialises correctly at QPY v18+.
        This exercises the NULL param path for ForLoopOp param[1] with an expr.Range
        indexset, distinct from the LOOP_VARIABLE path and the Python-range NULL path.
        """
        qc = QuantumCircuit(1)
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))
        # Explicit None: no loop variable requested.
        from qiskit.circuit.controlflow import ForLoopOp

        body = QuantumCircuit(1)
        body.h(0)
        qc.append(ForLoopOp(range_expr, None, body), [0])
        if version < 18:
            self.assert_dump_raises_unsupported(qc, version=version, write_with=write_with)
        else:
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_evolutiongate(self, version, write_with, read_with):
        """Test loading a circuit with evolution gate works."""
        synthesis = LieTrotter(reps=2)
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=2, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression(self, version, write_with, read_with):
        """Test loading a circuit with parameter expression works"""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_param = theta + phi
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(sum_param, range(3))
        qc.rz(phi, 3)
        qc.rz(theta, 4)
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression_with_vectors(self, version, write_with, read_with):
        """Test loading a circuit with parameter expression works"""
        theta = ParameterVector("θ", 3)
        beta = Parameter("β")
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.rx(beta + theta[1], qr)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_parameter_expression_subs(self, version, write_with, read_with):
        """Test loading a circuit with parameter substitution works"""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        exp = a + b
        exp = exp.subs({b: a})
        qc.ry(exp, 0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_degenerate_parameter_expression(self, version, write_with, read_with):
        """Test a circuit with a parameter expression that simplifies to 0."""
        x = Parameter("x")
        y_vec = ParameterVector("y", 2)
        z = Parameter("z")
        cases = [0 * x, 0 * x + 2, 0 * x + z, x - x, 0 * y_vec[0], 0 * (x + y_vec[1])]
        for case in cases:
            qc = QuantumCircuit(1)
            qc.rz(case, 0)
            self.assert_roundtrip_equal(
                qc, version=version, write_with=write_with, read_with=read_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_random_circuits(self, version, write_with, read_with):
        """Test loading a random circuit works"""
        for i in range(10):
            qc = random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
            # Make sure the circuits round-trip as a sanity check
            self.assert_roundtrip_equal(
                qc, version=version, read_with=read_with, write_with=write_with
            )

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_delay_roundtrip(self, version, write_with, read_with):
        qc = QuantumCircuit(1)
        qc.delay(1, 0, "dt")
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(14)
    def test_delay_expr_roundtrip(self, version, write_with, read_with):
        stretch_expr = QuantumCircuit(1, name="stretch_expr_delay_circuit")
        s = expr.Stretch(uuid.uuid4(), "a")
        stretch = stretch_expr.add_stretch(s)
        stretch_expr.delay(stretch, 0)
        stretch_expr.delay(expr.add(Duration.dt(200), stretch), 0)
        stretch_expr.delay(expr.sub(Duration.ns(3.14159), stretch), 0)
        self.assert_roundtrip_equal(
            stretch_expr, version=version, read_with=read_with, write_with=write_with
        )

    @all_qpy_combinations(14)
    def test_box_expr_roundtrip(self, version, write_with, read_with):
        qc = QuantumCircuit(1, name="box_expr_circuit")
        s = qc.add_stretch("s")
        duration = expr.add(Duration.dt(100), expr.sub(s, Duration.ns(16.25)))
        with qc.box(duration=duration):
            qc.x(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_literal_integers_in_for(self, version, write_with, read_with):
        qc = QuantumCircuit(1)
        with qc.for_loop((2, 5, (1 << 60))) as _:
            qc.x(0)
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

    @all_qpy_combinations(QPY_RUST_READ_MIN_VERSION)
    def test_large_generic_instruction_label(self, version, write_with, read_with):
        """Test round-tripping a library instruction with a large label."""

        label = "x" * 10000000
        gate = PauliEvolutionGate(
            SparsePauliOp.from_list([("Z", 1)]),
            label=label,
        )
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)

        qpy_file = io.BytesIO()
        with self.assertWarns(UserWarning):
            dump(qc, qpy_file, version=version)

    @all_qpy_combinations(17)
    def test_evolutiongate_sparse_observable(self, version, write_with, read_with):
        """Test loading a circuit with an evolution gate over a SparseObservable works.

        ``SparseObservable`` support was added to QPY in version 17.
        """
        op = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])
        evo = PauliEvolutionGate(op, time=2, synthesis=LieTrotter(reps=2))

        qc = QuantumCircuit(op.num_qubits)
        qc.append(evo, range(op.num_qubits))
        self.assert_roundtrip_equal(qc, version=version, read_with=read_with, write_with=write_with)
