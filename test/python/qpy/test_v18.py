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

"""Tests for QPY v18 format changes compared to v17."""

import io
import struct

from qiskit.circuit import (
    AncillaRegister,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Parameter,
    ParameterVector,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider.generic_backend_v2 import GenericBackendV2
from qiskit.qpy import dump, load
from qiskit.qpy import formats
from qiskit.qpy.exceptions import QpyError
from qiskit.quantum_info import SparseObservable, SparsePauliOp
from qiskit.transpiler.coupling import CouplingMap
from test import QiskitTestCase


def _dump(qc: QuantumCircuit, version: int | None = None) -> bytes:
    buf = io.BytesIO()
    dump(qc, buf, version=version)
    return buf.getvalue()


class TestV17VsV18(QiskitTestCase):
    """Verify the binary-level differences between QPY v17 and v18."""

    def test_v18_smaller_than_v17_by_calibration_header(self):
        """v18 output is exactly 2 bytes smaller than v17 (CalibrationsPack removed)."""
        # Use raw bits because register sizes also shrink in QPY v18
        qubits = [Qubit(), Qubit()]
        qc = QuantumCircuit(qubits)
        qc.h(0)
        qc.cx(0, 1)

        size17 = len(_dump(qc, 17))
        size18 = len(_dump(qc, 18))
        cal_header_size = struct.calcsize(formats.CALIBRATION_PACK)  # 2 bytes
        empty_vector_table_size = struct.calcsize("!H")  # the num_vectors count alone

        self.assertEqual(
            size17 - size18,
            cal_header_size - empty_vector_table_size,
            f"Expected v18 to differ from v17 by "
            f"{cal_header_size - empty_vector_table_size} bytes, "
            f"got v17={size17} v18={size18} diff={size17 - size18}",
        )

    def test_v18_smaller_than_v17_for_anonymous_qbits(self):
        """v18 output is exactly 1 bytes smaller than v17."""
        # anonymous qubit
        qubits = [Qubit()]
        qc = QuantumCircuit(qubits)
        qc.h(0)
        backend = GenericBackendV2(num_qubits=2, coupling_map=CouplingMap.from_line(2), seed=42)
        tqc = transpile(qc, backend, optimization_level=0)

        size17 = len(_dump(tqc, 17))
        size18 = len(_dump(tqc, 18))

        self.assertTrue(
            size17 - size18 > 0,
            f"Expected v18 to be smaller than v17, "
            f"got v17={size17} v18={size18} diff={size17 - size18}",
        )

    def test_float_param_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise float parameters in different byte order."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_float_param_v17_is_little_endian(self):
        """v17 float parameter bytes are little-endian on disk."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertIn(struct.pack("<d", 1.23456789), _dump(qc, 17))

    def test_float_param_v18_is_big_endian(self):
        """v18 float parameter bytes are big-endian on disk."""
        qc = QuantumCircuit(1)
        qc.rz(1.23456789, 0)
        self.assertIn(struct.pack(">d", 1.23456789), _dump(qc, 18))

    def test_for_loop_integers_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise ForLoop integer lists in different byte order."""
        qc = QuantumCircuit(1, 1)
        with qc.for_loop((1, 4, 9)):
            qc.h(0)
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_custom_gate_dump_is_deterministic(self):
        class MyGate(Gate):
            def __init__(self):
                super().__init__("my_gate", 1, [])

            def _define(self):
                qc = QuantumCircuit(1)
                qc.h(0)
                self.definition = qc

        # Start with baseline:
        qc = QuantumCircuit(1)
        qc.h(0)
        self.assertEqual(
            _dump(qc),
            _dump(qc),
            "Dumping same circuit (no custom gate) should produce the same dump",
        )

        # Now with custom gate:
        qc.append(MyGate(), [0])
        self.assertEqual(
            _dump(qc),
            _dump(qc),
            "Dumping same circuit (with custom gate) should produce the same dump",
        )

    def test_switch_case_labels_bytes_differ_v17_vs_v18(self):
        """v17 and v18 serialise SwitchCase integer labels in different byte order."""
        body = QuantumCircuit(1)
        body.h(0)
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy()), (2, body.copy())], [0], [])
        self.assertNotEqual(_dump(qc, 17), _dump(qc, 18))

    def test_ancilla_register_round_trip(self):
        """Ancilla register and qubit types survive serialization."""
        circuit = QuantumCircuit(AncillaRegister(2, "ancilla"))
        loaded = load(io.BytesIO(_dump(circuit, 18)))[0]
        self.assertIsInstance(loaded.qregs[0], AncillaRegister)
        self.assertEqual(loaded, circuit)

        ar = AncillaRegister(2, "ancilla")
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr, ar)
        circuit.cx(qr[0], ar[0])
        loaded = load(io.BytesIO(_dump(circuit, 18)))[0]
        self.assertIsInstance(loaded.qregs[0], QuantumRegister)
        self.assertIsInstance(loaded.qregs[1], AncillaRegister)
        self.assertEqual(loaded, circuit)


class TestV18RegisterParam(QiskitTestCase):
    """The `Register` payload gained a tag byte in v18.

    It identifies either a whole :class:`.ClassicalRegister` or a single :class:`.Clbit`.  Up to v17
    the two shared one untyped string -- a register was its bare name, a clbit was a null byte
    followed by its index in ASCII digits -- so the payload could not be identified from its own
    bytes.
    """

    #: The condition payload follows the instruction's class name, as these circuits set no label.
    GATE_NAME = b"IfElseOp"

    @staticmethod
    def _clbit_condition(index=1, num_clbits=2):
        """A circuit whose only instruction is conditioned on a single clbit."""
        qreg, creg = QuantumRegister(1, "q"), ClassicalRegister(num_clbits, "creg")
        circuit = QuantumCircuit(qreg, creg)
        body = QuantumCircuit(1)
        body.x(0)
        circuit.if_test((creg[index], True), body, [qreg[0]], [])
        return circuit

    @staticmethod
    def _register_condition(name="creg"):
        """A circuit whose only instruction is conditioned on a whole classical register."""
        qreg, creg = QuantumRegister(1, "q"), ClassicalRegister(2, name)
        circuit = QuantumCircuit(qreg, creg)
        body = QuantumCircuit(1)
        body.x(0)
        circuit.if_test((creg, 1), body, [qreg[0]], [])
        return circuit

    def _condition_payload(self, data, length):
        """The ``length`` bytes of condition payload that follow the instruction's class name."""
        start = data.rindex(self.GATE_NAME) + len(self.GATE_NAME)
        return bytes(data[start : start + length])

    def test_clbit_condition_is_tagged_in_v18(self):
        """A clbit is a tag byte plus a uint32 index from v18; ASCII digits up to v17."""
        for index in (0, 1, 3):
            with self.subTest(index=index):
                circuit = self._clbit_condition(index=index, num_clbits=4)
                legacy = b"\x00" + str(index).encode("utf8")
                self.assertEqual(self._condition_payload(_dump(circuit, 17), len(legacy)), legacy)
                tagged = struct.pack(
                    formats.REGISTER_PARAM_CLBIT_PACK, formats.REGISTER_PARAM_TAG_CLBIT, index
                )
                self.assertEqual(self._condition_payload(_dump(circuit, 18), len(tagged)), tagged)

    def test_register_condition_is_tagged_in_v18(self):
        """A register is a tag byte plus its name from v18; the bare name up to v17."""
        circuit = self._register_condition()
        self.assertEqual(self._condition_payload(_dump(circuit, 17), 4), b"creg")
        tagged = (
            struct.pack(formats.REGISTER_PARAM_TAG_PACK, formats.REGISTER_PARAM_TAG_REGISTER)
            + b"creg"
        )
        self.assertEqual(self._condition_payload(_dump(circuit, 18), len(tagged)), tagged)

    def test_null_prefixed_register_name_needs_v18(self):
        """A register name starting with a null byte is only representable from v18.

        Qiskit accepts such a name, but up to v17 it collides with the marker for a single-bit
        condition, so the reader takes the name for a bit index and fails.  The tag removes the
        ambiguity.
        """
        circuit = self._register_condition(name="\x00weird")
        with self.assertRaises((QpyError, ValueError)):
            load(io.BytesIO(_dump(circuit, 17)))
        self.assertEqual(load(io.BytesIO(_dump(circuit, 18)))[0], circuit)

    def test_v18_unknown_tag_is_rejected(self):
        """An unrecognised tag byte must fail rather than be guessed at."""
        data = bytearray(_dump(self._clbit_condition(), 18))
        data[data.rindex(self.GATE_NAME) + len(self.GATE_NAME)] = 7
        with self.assertRaises(QpyError):
            load(io.BytesIO(bytes(data)))

    def test_v18_unknown_register_name_is_rejected(self):
        """A register name that is not in the circuit must fail."""
        data = bytearray(_dump(self._register_condition(), 18))
        start = data.rindex(self.GATE_NAME) + len(self.GATE_NAME)
        data[start + 1 : start + 5] = b"zzzz"  # same length, so no size field to fix up
        with self.assertRaises(QpyError):
            load(io.BytesIO(bytes(data)))


class TestV18SparseObservable(QiskitTestCase):
    """``SPARSE_OBSERVABLE`` payloads, whose bit terms narrowed from ``uint16_t`` to ``uint8_t``.

    QPY gained :class:`.SparseObservable` in v17 and v18 narrowed the stored bit terms, so between
    them these two versions cover both encodings.

    Only those two are covered, for two independent reasons.  The Rust writer is the only one that
    emits QPY >= 17 (``QPY_RUST_WRITE_MIN_VERSION``), and a payload written by one implementation
    cannot currently be read by the other, because the two codecs disagree over whether the
    ``*_data_len`` fields hold a byte length or an element count.  Separately, asking for a version
    below 17 does not raise: the writer emits a v17-shaped element into the older payload, which no
    reader can then parse.
    """

    # TODO - the cross-implementation half of this is bug #16722; once that is fixed these can also
    # be covered by the writer/reader matrix in test_roundtrip.py.
    VERSIONS = (17, 18)

    def _assert_roundtrips(self, circuit):
        """The circuit survives a dump/load at each version that can express it."""
        for version in self.VERSIONS:
            with self.subTest(version=version):
                self.assertEqual(load(io.BytesIO(_dump(circuit, version)))[0], circuit)

    def test_evolutiongate_sparse_observable(self):
        """An evolution gate over a SparseObservable round-trips under both bit-term widths.

        The operator uses every :class:`.SparseObservable.BitTerm` variant, so the full value range
        of that field is exercised.
        """
        op = SparseObservable.from_list(
            [
                ("XIII", 0.1),
                ("YIII", 0.2),
                ("ZIII", 0.3),
                ("+III", 0.4),
                ("-III", 0.5),
                ("rIII", 0.6),
                ("lIII", 0.7),
                ("0III", 0.8),
                ("1III", 0.9),
            ]
        )
        qc = QuantumCircuit(op.num_qubits)
        qc.append(PauliEvolutionGate(op, time=0.3), qc.qubits)
        self._assert_roundtrips(qc)

    def test_evolutiongate_mixed_operators(self):
        """An evolution gate over a list mixing SparseObservable and SparsePauliOp."""
        op1 = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])
        op2 = SparsePauliOp.from_list([("ZZI", 1), ("XIX", -0.1)])
        evo = PauliEvolutionGate([op1, op2], time=0.5)
        qc = QuantumCircuit(evo.num_qubits)
        qc.append(evo, qc.qubits)
        self._assert_roundtrips(qc)


class TestV18ParameterVectorTable(QiskitTestCase):
    """From v18 a ``ParameterVector`` is stored once per payload and its elements point at it.

    Up to v17 every element repeated the vector's name and size, plus a UUID that is the vector's own
    offset by the element index.
    """

    @staticmethod
    def _vector_circuit(name="v", length=3):
        """A circuit applying one rotation per element of a single vector."""
        vector = ParameterVector(name, length)
        circuit = QuantumCircuit(1, name="vector_circuit")
        for parameter in vector:
            circuit.rx(parameter, 0)
        return circuit, vector

    def test_element_shrinks_by_expected_amount(self):
        """An element costs 10 bytes at v18 against 34 plus the vector name at v17.

        Measured by differencing twice: once between v17 and v18 for a given circuit, then between two
        vector lengths.  Everything that does not scale with the number of elements -- the gate that
        carries each one, the table, the calibration header -- cancels out.
        """
        extra = 10
        short, _ = self._vector_circuit(length=1)
        long, _ = self._vector_circuit(length=1 + extra)

        saving = (len(_dump(long, 17)) - len(_dump(long, 18))) - (
            len(_dump(short, 17)) - len(_dump(short, 18))
        )
        self.assertEqual(saving, extra * ((34 + len("v")) - 10))

    def test_roundtrip_preserves_vector_identity(self):
        """Reloaded elements belong to one vector, with the original name, length and UUIDs."""
        circuit, vector = self._vector_circuit(length=4)
        reloaded = load(io.BytesIO(_dump(circuit, 18)))[0]
        self.assertEqual(reloaded, circuit)

        elements = [instruction.operation.params[0] for instruction in reloaded.data]
        vectors = {element.vector for element in elements}
        self.assertEqual(len(vectors), 1)
        reloaded_vector = vectors.pop()
        self.assertEqual(reloaded_vector.name, vector.name)
        self.assertEqual(len(reloaded_vector), len(vector))
        self.assertEqual([element.uuid for element in elements], [p.uuid for p in vector])

    def test_two_vectors_stay_distinct(self):
        """Two vectors get separate table entries and do not collapse into one."""
        first, second = ParameterVector("a", 2), ParameterVector("b", 2)
        circuit = QuantumCircuit(1)
        for parameter in list(first) + list(second):
            circuit.rx(parameter, 0)

        reloaded = load(io.BytesIO(_dump(circuit, 18)))[0]
        self.assertEqual(reloaded, circuit)
        names = {instruction.operation.params[0].vector.name for instruction in reloaded.data}
        self.assertEqual(names, {"a", "b"})

    def test_element_inside_an_expression_roundtrips(self):
        """A vector element reached through a parameter expression uses the table too."""
        vector, theta = ParameterVector("w", 2), Parameter("theta")
        circuit = QuantumCircuit(2)
        circuit.rx(vector[0] + theta, 0)
        circuit.ry(vector[1] * 2, 1)
        self.assertEqual(load(io.BytesIO(_dump(circuit, 18)))[0], circuit)

    def test_element_inside_a_control_flow_block_roundtrips(self):
        """A nested payload carries its own table, so a block's elements resolve independently."""
        vector = ParameterVector("n", 2)
        circuit = QuantumCircuit(1, 1)
        circuit.rx(vector[0], 0)
        with circuit.if_test((circuit.clbits[0], True)):
            circuit.rx(vector[1], 0)
        self.assertEqual(load(io.BytesIO(_dump(circuit, 18)))[0], circuit)
