# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for circuit qpy loading and saving."""

import io
import struct

from ddt import ddt, data, idata

from qiskit.circuit import QuantumCircuit, QuantumRegister, Qubit, Parameter, Gate, annotation
from qiskit.circuit.random import random_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.exceptions import QiskitError
from qiskit.qpy import dump, load, formats, get_qpy_version, QPY_COMPATIBILITY_VERSION, QpyError
from qiskit.qpy.common import QPY_VERSION
from qiskit.transpiler import TranspileLayout, CouplingMap
from qiskit.compiler import transpile
from qiskit.qpy.formats import FILE_HEADER_V10_PACK, FILE_HEADER_V10, FILE_HEADER_V10_SIZE
from test import QiskitTestCase


class QpyCircuitTestCase(QiskitTestCase):
    """QPY circuit testing platform."""

    def assert_roundtrip_equal(
        self, circuit, version=None, use_symengine=None, annotation_factories=None
    ):
        """QPY roundtrip equal test."""
        qpy_file = io.BytesIO()
        if use_symengine is None:
            dump(circuit, qpy_file, version=version, annotation_factories=annotation_factories)
        else:
            dump(
                circuit,
                qpy_file,
                version=version,
                use_symengine=use_symengine,
                annotation_factories=annotation_factories,
            )
        qpy_file.seek(0)
        new_circuit = load(qpy_file, annotation_factories=annotation_factories)[0]

        self.assertEqual(circuit, new_circuit)
        self.assertEqual(circuit.layout, new_circuit.layout)
        if version is not None:
            qpy_file.seek(0)
            file_version = struct.unpack("!6sB", qpy_file.read(7))[1]
            self.assertEqual(
                version,
                file_version,
                f"Generated QPY file version {file_version} does not match request version {version}",
            )
        if annotation_factories is not None:

            def flat_annotations(qc):
                for inst in qc.data:
                    op = inst.operation
                    if inst.name == "box":
                        yield op.annotations
                    if inst.is_control_flow():
                        for block in op.blocks:
                            yield from flat_annotations(block)

            self.assertEqual(list(flat_annotations(circuit)), list(flat_annotations(new_circuit)))


class TestVersions(QpyCircuitTestCase):
    """Test version handling in qpy."""

    def test_invalid_qpy_version(self):
        """Test a descriptive exception is raised if QPY version is too new."""
        with io.BytesIO() as buf:
            buf.write(
                struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", QPY_VERSION + 4, 42, 42, 1, 2)
            )
            buf.seek(0)
            with self.assertRaisesRegex(QiskitError, str(QPY_VERSION + 4)):
                load(buf)

    def test_get_qpy_version(self):
        """Test the get_qpy_version function."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        for version in range(QPY_COMPATIBILITY_VERSION, QPY_VERSION + 1):
            with io.BytesIO() as qpy_file:
                dump(qc, qpy_file, version=version)
                qpy_file.seek(0)
                file_version = get_qpy_version(qpy_file)
            self.assertEqual(version, file_version)

    def test_get_qpy_version_read(self):
        """Ensure we don't advance the cursor exiting the get_qpy_version() function."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        with io.BytesIO() as qpy_file:
            dump(qc, qpy_file)
            qpy_file.seek(0)
            file_version = get_qpy_version(qpy_file)
            self.assertEqual(file_version, QPY_VERSION)
            res = load(qpy_file)[0]
        self.assertEqual(res, qc)


@ddt
class TestLayout(QpyCircuitTestCase):
    """Test circuit serialization for layout preservation."""

    @data(0, 1, 2, 3)
    def test_transpile_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, coupling_map=CouplingMap.from_line(127), seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_with_routing(self, opt_level):
        """Test full layout with routing is preserved."""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, coupling_map=CouplingMap.from_line(127), seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_transpile_layout_explicit_None_final_layout(self, opt_level):
        """Test layout preserved after transpile."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, coupling_map=CouplingMap.from_line(127), seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        tqc.layout.final_layout = None
        self.assert_roundtrip_equal(tqc)

    def test_empty_layout(self):
        """Test an empty layout is preserved correctly."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qc._layout = TranspileLayout(None, None, None)
        self.assert_roundtrip_equal(qc)

    def test_overlapping_definitions(self):
        """Test serialization of custom gates with overlapping definitions."""

        class MyParamGate(Gate):
            """Custom gate class with a parameter."""

            def __init__(self, phi):
                super().__init__("my_gate", 1, [phi])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.rx(self.params[0], 0)
                self.definition = qc

        theta = Parameter("theta")
        two_theta = 2 * theta

        qc = QuantumCircuit(1)
        qc.append(MyParamGate(1.1), [0])
        qc.append(MyParamGate(1.2), [0])
        qc.append(MyParamGate(3.14159), [0])
        qc.append(MyParamGate(theta), [0])
        qc.append(MyParamGate(two_theta), [0])
        with io.BytesIO() as qpy_file:
            dump(qc, qpy_file)
            qpy_file.seek(0)
            new_circ = load(qpy_file)[0]
        # Custom gate classes are lowered to Gate to avoid arbitrary code
        # execution on deserialization. To compare circuit equality we
        # need to go instruction by instruction and check that they're
        # equivalent instead of doing a circuit equality check
        for new_inst, old_inst in zip(new_circ.data, qc.data):
            new_gate = new_inst.operation
            old_gate = old_inst.operation
            self.assertIsInstance(new_gate, Gate)
            self.assertEqual(new_gate.name, old_gate.name)
            self.assertEqual(new_gate.params, old_gate.params)
            self.assertEqual(new_gate.definition, old_gate.definition)

    @data(0, 1, 2, 3)
    def test_custom_register_name(self, opt_level):
        """Test layout preserved with custom register names."""
        qr = QuantumRegister(5, name="abc123")
        qc = QuantumCircuit(qr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, coupling_map=CouplingMap.from_line(127), seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        self.assert_roundtrip_equal(tqc)

    @data(0, 1, 2, 3)
    def test_no_register(self, opt_level):
        """Test layout preserved with no register."""
        qubits = [Qubit(), Qubit()]
        qc = QuantumCircuit(qubits)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        backend = GenericBackendV2(num_qubits=127, coupling_map=CouplingMap.from_line(127), seed=42)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        # Manually validate to deal with qubit equality needing exact objects
        qpy_file = io.BytesIO()
        dump(tqc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(tqc, new_circuit)
        initial_layout_old = tqc.layout.initial_layout.get_physical_bits()
        initial_layout_new = new_circuit.layout.initial_layout.get_physical_bits()
        for i in initial_layout_old:
            self.assertIsInstance(initial_layout_old[i], Qubit)
            self.assertIsInstance(initial_layout_new[i], Qubit)
            if initial_layout_old[i]._register is not None:
                self.assertEqual(initial_layout_new[i], initial_layout_old[i])
            else:
                self.assertIsNone(initial_layout_new[i]._register)
                self.assertIsNone(initial_layout_old[i]._index)
                self.assertIsNone(initial_layout_new[i]._index)
        self.assertEqual(
            list(tqc.layout.input_qubit_mapping.values()),
            list(new_circuit.layout.input_qubit_mapping.values()),
        )
        self.assertEqual(tqc.layout.final_layout, new_circuit.layout.final_layout)


class TestVersionArg(QpyCircuitTestCase):
    """Test explicitly setting a qpy version in dump()."""

    def test_invalid_version_value(self):
        """Assert we raise an error with an invalid version request."""
        qc = QuantumCircuit(2)
        with self.assertRaises(ValueError):
            dump(qc, io.BytesIO(), version=3)

    def test_compatibility_version_roundtrip(self):
        """Test the version is set correctly when specified."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        self.assert_roundtrip_equal(qc, version=QPY_COMPATIBILITY_VERSION)

    def test_nested_params_subs(self):
        """Test substitution works."""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        expr = a + b
        expr = expr.subs({b: a})
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_all_the_expression_ops(self):
        """Test a circuit with an expression that uses all the ops available."""
        qc = QuantumCircuit(1)
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")
        d = Parameter("d")

        expression = (a + b.sin() / 4) * c**2
        final_expr = (
            (expression.cos() + d.arccos() - d.arcsin() + d.arctan() + d.tan()) / d.exp()
            + expression.gradient(a)
            + expression.log()
            - a.sin()
            - b.conjugate()
        )
        final_expr = final_expr.abs()
        final_expr = final_expr.subs({c: a})

        qc.rx(final_expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rpow(self):
        """Test rpow works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159**a
        expr = expr**b
        expr = 1.2345**expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rsub(self):
        """Test rsub works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159 - a
        expr = expr - b
        expr = 1.2345 - expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)

    def test_rdiv(self):
        """Test rdiv works as expected"""
        qc = QuantumCircuit(1)
        a = Parameter("A")
        b = Parameter("B")
        expr = 3.14159 / a
        expr = expr / b
        expr = 1.2345 / expr
        qc.ry(expr, 0)
        self.assert_roundtrip_equal(qc)


@ddt
class TestUseSymengineFlag(QpyCircuitTestCase):
    """Test that the symengine flag works correctly."""

    @data(True, False)
    def test_use_symengine_with_bool_like(self, use_symengine):
        """Test that the use_symengine flag is set correctly with a bool-like input."""

        class Booly:
            def __init__(self, value):
                self.value = value

            def __bool__(self):
                return self.value

        theta = Parameter("theta")
        two_theta = 2 * theta
        qc = QuantumCircuit(1)
        qc.rx(two_theta, 0)
        qc.measure_all()
        # Assert Roundtrip works
        # `use_symengine` is near-completely ignored with QPY versions 13+; it doesn't actually
        # matter if we _have_ symengine installed or not, because those QPYs don't ever use it
        # (except for setting a single byte in the header, which is promptly ignored).
        self.assert_roundtrip_equal(qc, use_symengine=Booly(use_symengine), version=13)
        # Also check the qpy symbolic expression encoding is correct in the
        # payload
        with io.BytesIO() as file_obj:
            dump(qc, file_obj, use_symengine=Booly(use_symengine), version=13)
            file_obj.seek(0)
            header_data = FILE_HEADER_V10._make(
                struct.unpack(
                    FILE_HEADER_V10_PACK,
                    file_obj.read(FILE_HEADER_V10_SIZE),
                )
            )
            self.assertEqual(header_data.symbolic_encoding, b"e" if use_symengine else b"p")


class TestSymbolExpr(QpyCircuitTestCase):
    """Test QPY with SymbolExpr"""

    def test_back_slash(self):
        """Test Parameter with back slash"""
        qc = QuantumCircuit(2)
        alpha = Parameter(r"\alpha")
        beta = Parameter(r"\beta")
        qc.rz(alpha + beta, 0)
        self.assert_roundtrip_equal(qc)

    def test_math_mode_backslash(self):
        """Test Parameter with mathmode backslah."""
        qc = QuantumCircuit(2)
        alpha = Parameter(r"$\alpha$")
        beta = Parameter(r"$\beta$")
        qc.rz(alpha + beta, 0)
        self.assert_roundtrip_equal(qc)


class TestAnnotations(QpyCircuitTestCase):

    def test_wrapping_openqasm3(self):
        class My(annotation.Annotation):
            def __init__(self, namespace, value):
                self.namespace = namespace
                self.value = value

            def __eq__(self, other):
                return (
                    isinstance(other, My)
                    and self.namespace == other.namespace
                    and self.value == other.value
                )

            def __repr__(self):
                return f"My({self.namespace!r}, {self.value!r})"

        class Serializer(annotation.OpenQASM3Serializer):
            def load(self, namespace, payload):
                return My(namespace, payload)

            def dump(self, annotation):
                return annotation.value

        qc = QuantumCircuit()
        with qc.box([My("my.a", "hello"), My("my.b", "world")]):
            with qc.box([My("my.c", "")]):
                pass
        self.assert_roundtrip_equal(qc, annotation_factories={"my": Serializer().as_qpy()})

    def test_simple_serializer(self):
        outer_self = self

        class Dummy(annotation.Annotation):
            namespace = "dummy"

            def __eq__(self, other):
                return isinstance(other, Dummy)

            def __repr__(self):
                return "Dummy()"

        class Serializer(annotation.QPYSerializer):
            def load_annotation(self, payload):
                outer_self.assertEqual(payload, b"SENTINEL")
                return Dummy()

            def dump_annotation(self, namespace, annotation):
                outer_self.assertEqual(namespace, "dummy")
                return b"SENTINEL"

        qc = QuantumCircuit()
        with qc.box([Dummy()]):
            pass
        self.assert_roundtrip_equal(qc, annotation_factories={"dummy": Serializer})

    def test_stateful_serializer(self):
        outer_self = self

        class My(annotation.Annotation):
            namespace = "my"

            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, My) and self.value == other.value

            def __repr__(self):
                return f"My({self.value!r})"

        class Serializer(annotation.QPYSerializer):
            def __init__(self):
                self.loaded_state = False
                self.dumped_state = False
                self.num_serialized = 0

            def dump_annotation(self, namespace, annotation):
                outer_self.assertFalse(self.dumped_state)
                outer_self.assertFalse(self.loaded_state)
                self.num_serialized += 1
                return annotation.value.encode("utf-8")

            def dump_state(self):
                # This is a check that there are two separate instances when moving into an inner
                # circuit; both the outer and inner `box`es have two annotations.
                outer_self.assertEqual(self.num_serialized, 2)
                outer_self.assertFalse(self.dumped_state)
                outer_self.assertFalse(self.loaded_state)
                self.dumped_state = True
                return b"SENTINEL"

            def load_annotation(self, payload):
                outer_self.assertFalse(self.dumped_state)
                outer_self.assertTrue(self.loaded_state)
                return My(payload.decode("utf-8"))

            def load_state(self, namespace, payload):
                outer_self.assertEqual(namespace, "my")
                outer_self.assertEqual(payload, b"SENTINEL")
                outer_self.assertFalse(self.dumped_state)
                outer_self.assertFalse(self.loaded_state)
                self.loaded_state = True

        qc = QuantumCircuit()
        with qc.box([My("hello"), My("world")]):
            with qc.box([My("inner"), My("number 2")]):
                pass
        self.assert_roundtrip_equal(qc, annotation_factories={"my": Serializer})

    def test_multiple_serializers(self):
        outer_self = self

        class TypeA(annotation.Annotation):
            namespace = "a"

            def __eq__(self, other):
                return isinstance(other, TypeA)

            def __repr__(self):
                return "TypeA()"

        class TypeB(annotation.Annotation):
            namespace = "b"

            def __eq__(self, other):
                return isinstance(other, TypeB)

            def __repr__(self):
                return "TypeB()"

        class SerializerA(annotation.QPYSerializer):
            def dump_annotation(self, namespace, annotation):
                outer_self.assertEqual(namespace, "a")
                outer_self.assertIsInstance(annotation, TypeA)
                return b"A"

            def dump_state(self):
                return b"STATE A"

            def load_annotation(self, payload):
                outer_self.assertEqual(payload, b"A")
                return TypeA()

            def load_state(self, namespace, payload):
                outer_self.assertEqual(namespace, "a")
                outer_self.assertEqual(payload, b"STATE A")

        class SerializerB(annotation.QPYSerializer):
            def dump_annotation(self, namespace, annotation):
                outer_self.assertEqual(namespace, "b")
                outer_self.assertIsInstance(annotation, TypeB)
                return b"B"

            def dump_state(self):
                return b"STATE B"

            def load_annotation(self, payload):
                outer_self.assertEqual(payload, b"B")
                return TypeB()

            def load_state(self, namespace, payload):
                outer_self.assertEqual(namespace, "b")
                outer_self.assertEqual(payload, b"STATE B")

        qc = QuantumCircuit()
        with qc.box([TypeA(), TypeB()]):
            with qc.box([TypeB(), TypeA()]):
                pass
        with qc.box([TypeB(), TypeA()]):
            with qc.box([TypeA()]):
                pass
        self.assert_roundtrip_equal(qc, annotation_factories={"a": SerializerA, "b": SerializerB})

    def test_parent_namespacing(self):
        outer_self = self

        class My(annotation.Annotation):
            def __init__(self, namespace, value):
                self.namespace = namespace
                self.value = value

            def __eq__(self, other):
                return (
                    isinstance(other, My)
                    and self.namespace == other.namespace
                    and self.value == other.value
                )

            def __repr__(self):
                return f"My({self.namespace!r}, {self.value!r})"

        triggered_not_implemented = False

        class Serializer(annotation.QPYSerializer):
            # The idea with this family of serialisers is that we say the "value" we expect to see
            # in the annotations it will handle, then don't actually store _anything_ in the QPY for
            # that.  We only store the _actual_ namespace of the annotation (not the parent we were
            # called with) in the state field.

            def __init__(self, value):
                self.actual_namespace = None
                self.value = value

            def dump_annotation(self, namespace, annotation):
                if annotation.value != self.value:
                    nonlocal triggered_not_implemented
                    triggered_not_implemented = True
                    return NotImplemented
                if self.actual_namespace is None:
                    self.actual_namespace = annotation.namespace
                else:
                    outer_self.assertEqual(annotation.namespace, self.actual_namespace)
                return b""

            def load_annotation(self, payload):
                return My(self.actual_namespace, self.value)

            def dump_state(self):
                outer_self.assertIsNotNone(self.actual_namespace)
                return self.actual_namespace.encode("utf-8") + b"\x00" + self.value.encode("utf-8")

            def load_state(self, namespace, payload):
                namespace, value = payload.split(b"\x00")
                self.actual_namespace = namespace.decode("utf-8")
                self.value = value.decode("utf-8")

        qc = QuantumCircuit()
        with qc.box(
            [My("a.b", "looks at a"), My("a.c", "looks at a.c"), My("a.d", "looks at global")]
        ):
            pass
        self.assert_roundtrip_equal(
            qc,
            annotation_factories={
                "a.c": (lambda: Serializer("looks at a.c")),
                "a": (lambda: Serializer("looks at a")),
                "": (lambda: Serializer("looks at global")),
            },
        )
        self.assertTrue(triggered_not_implemented)

    def test_reject_unknown_namespace(self):
        class Unknown(annotation.Annotation):
            namespace = "unknown"

        qc = QuantumCircuit()
        with qc.box([Unknown()]):
            pass

        with io.BytesIO() as fptr, self.assertRaisesRegex(QpyError, "No configured annotation"):
            dump(qc, fptr)


@ddt
class TestOutputStreamProperties(QpyCircuitTestCase):
    """Test that QPY works with streams based on capability."""

    class UnseekableStream(io.IOBase):
        """A wrapper around a binary stream that is not seekable."""

        def __init__(self, base):
            self._base = base

        def seekable(self) -> bool:  # type: ignore[override]
            return False

        def read(self, size=-1):
            return self._base.read(size)

        def write(self, b):
            return self._base.write(b)

        def readable(self):
            return self._base.readable()

        def writable(self):
            return self._base.writable()

        def close(self):
            return self._base.close()

        def closed(self):
            return self._base.closed

    @idata(range(QPY_COMPATIBILITY_VERSION, QPY_VERSION + 1))
    def test_unseekable_equality(self, version):
        """Test QPY output is equal for seekable and unseekable streams."""
        circuits = []
        for i in range(10):
            circuits.append(
                random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
            )
            # Make sure the circuits round-trip as a sanity check
            self.assert_roundtrip_equal(circuits[i], version=version)

        # Check that an unseekable stream and seekable stream ended up with the same
        # contents.
        with io.BytesIO() as seekable:
            dump(circuits, seekable)
            with io.BytesIO() as internal_buffer:
                unseekable = TestOutputStreamProperties.UnseekableStream(internal_buffer)
                dump(circuits, unseekable)
                self.assertEqual(internal_buffer.getbuffer(), seekable.getbuffer())
