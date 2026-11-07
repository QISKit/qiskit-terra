# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test cases for qpy serialization."""

import gzip
import io
import json
import random
import tempfile
import unittest

import ddt
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit import CASE_DEFAULT, IfElseOp, WhileLoopOp, SwitchCaseOp
from qiskit.circuit.classical import expr, types
from qiskit.circuit import Clbit, Qubit
from qiskit.transpiler import Target, CouplingMap
from qiskit.circuit.random import random_circuit
from qiskit.circuit.gate import Gate
from qiskit.circuit.library import (
    XGate,
    ZGate,
    CXGate,
    RYGate,
    QFT,
    QFTGate,
    QAOAAnsatz,
    PauliEvolutionGate,
    DCXGate,
    MCU1Gate,
    MCXGate,
    MCXGrayCode,
    MCXRecursive,
    MCXVChain,
    MCMTGate,
    UCRXGate,
    UCRYGate,
    UCRZGate,
    UnitaryGate,
    DiagonalGate,
    PauliFeatureMap,
    ZZFeatureMap,
    RealAmplitudes,
    pauli_feature_map,
    zz_feature_map,
    qaoa_ansatz,
    real_amplitudes,
)
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.qpy import (
    dump,
    load,
    UnsupportedFeatureForVersion,
    QPY_COMPATIBILITY_VERSION,
    QPY_VERSION,
    QpyError,
)
from qiskit.quantum_info import Pauli, SparsePauliOp, Clifford
from qiskit.quantum_info import random_unitary
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.utils import optionals
from test import QiskitTestCase


@ddt.ddt
class TestLoadFromQPY(QiskitTestCase):
    """Test qpy set of methods."""

    def assertDeprecatedBitProperties(self, original, roundtripped):
        """Test that deprecated bit attributes are equal if they are set in the original circuit."""
        owned_qubits = [
            (a, b) for a, b in zip(original.qubits, roundtripped.qubits) if a._register is not None
        ]
        if owned_qubits:
            original_qubits, roundtripped_qubits = zip(*owned_qubits)
            self.assertEqual(original_qubits, roundtripped_qubits)
        owned_clbits = [
            (a, b) for a, b in zip(original.clbits, roundtripped.clbits) if a._register is not None
        ]
        if owned_clbits:
            original_clbits, roundtripped_clbits = zip(*owned_clbits)
            self.assertEqual(original_clbits, roundtripped_clbits)

    def assertMinimalVarEqual(self, left, right):
        """Replacement for asserting `QuantumCircuit` equality for use in `Var` tests, for use while
        the `DAGCircuit` does not yet allow full equality checks.  This should be removed and the
        tests changed to directly call `assertEqual` once possible.

        This filters out instructions that have `QuantumCircuit` parameters in the data comparison
        (such as control-flow ops), which need to be handled separately."""
        self.assertEqual(list(left.iter_input_vars()), list(right.iter_input_vars()))
        self.assertEqual(list(left.iter_declared_vars()), list(right.iter_declared_vars()))
        self.assertEqual(list(left.iter_captured_vars()), list(right.iter_captured_vars()))

        def filter_ops(data):
            return [
                ins
                for ins in data
                if not any(isinstance(x, QuantumCircuit) for x in ins.operation.params)
            ]

        self.assertEqual(filter_ops(left.data), filter_ops(right.data))

    def test_qpy_full_path(self):
        """Test full path qpy serialization for basic circuit."""
        qr_a = QuantumRegister(4, "a")
        qr_b = QuantumRegister(4, "b")
        cr_c = ClassicalRegister(4, "c")
        cr_d = ClassicalRegister(4, "d")
        q_circuit = QuantumCircuit(
            qr_a,
            qr_b,
            cr_c,
            cr_d,
            name="MyCircuit",
            metadata={"test": 1, "a": 2},
            global_phase=3.14159,
        )
        q_circuit.h(qr_a)
        q_circuit.cx(qr_a, qr_b)
        q_circuit.barrier(qr_a)
        q_circuit.barrier(qr_b)
        q_circuit.measure(qr_a, cr_c)
        q_circuit.measure(qr_b, cr_d)
        qpy_file = io.BytesIO()
        dump(q_circuit, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(q_circuit, new_circ)
        self.assertEqual(q_circuit.global_phase, new_circ.global_phase)
        self.assertEqual(q_circuit.metadata, new_circ.metadata)
        self.assertEqual(q_circuit.name, new_circ.name)
        self.assertDeprecatedBitProperties(q_circuit, new_circ)

    def test_int_parameter(self):
        """Test that integer parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(3, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_float_parameter(self):
        """Test that float parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(3.14, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_numpy_float_parameter(self):
        """Test that numpy float parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(np.float32(3.14), 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_numpy_int_parameter(self):
        """Test that numpy integer parameters are correctly serialized."""
        qc = QuantumCircuit(1)
        qc.rx(np.int16(3), 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_unitary_gate(self):
        """Test that numpy array parameters are correctly serialized"""
        qc = QuantumCircuit(1)
        unitary = np.array([[0, 1], [1, 0]])
        qc.unitary(unitary, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_controlled_unitary_gate(self):
        """Test that numpy array parameters are correctly serialized
        in controlled unitary gate."""
        qc = QuantumCircuit(2)
        unitary = np.array([[0, 1], [1, 0]])
        gate = UnitaryGate(unitary)
        qc.append(gate.control(1, annotated=False), [0, 1])

        with io.BytesIO() as qpy_file:
            dump(qc, qpy_file)
            qpy_file.seek(0)
            new_circ = load(qpy_file)[0]

        self.assertEqual(qc.decompose(reps=5), new_circ.decompose(reps=5))
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_opaque_gate(self):
        """Test that custom opaque gate is correctly serialized"""
        custom_gate = Gate("black_box", 1, [])
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_opaque_instruction(self):
        """Test that custom opaque instruction is correctly serialized"""
        custom_gate = Instruction("black_box", 1, 0, [])
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_gate(self):
        """Test that custom  gate is correctly serialized"""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition

        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_instruction(self):
        """Test that custom instruction is correctly serialized"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_parameter(self):
        """Test that a circuit with a parameter is correctly serialized."""
        theta = Parameter("theta")
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(5))
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            qc.assign_parameters({theta: 3.14}), new_circ.assign_parameters({theta: 3.14})
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_bound_parameter(self):
        """Test a circuit with a bound parameter is correctly serialized."""
        theta = Parameter("theta")
        qc = QuantumCircuit(5, 1)
        qc.h(0)
        for i in range(4):
            qc.cx(i, i + 1)

        qc.barrier()
        qc.rz(theta, range(5))
        qc.barrier()
        for i in reversed(range(4)):
            qc.cx(i, i + 1)
        qc.h(0)
        qc.measure(0, 0)
        qc.assign_parameters({theta: 3.14})

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_parameter_expression(self):
        """Test a circuit with a parameter expression."""
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

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_degenerate_parameter_expression(self):
        """Test a circuit with a parameter expression that simplifies to 0."""
        x = Parameter("x")
        y_vec = ParameterVector("y", 2)
        z = Parameter("z")
        cases = [0 * x, 0 * x + 2, 0 * x + z, x - x, 0 * y_vec[0], 0 * (x + y_vec[1])]
        for case in cases:
            qc = QuantumCircuit(1)
            qc.rz(case, 0)
            qpy_file = io.BytesIO()
            dump(qc, qpy_file)
            qpy_file.seek(0)
            new_circuit = load(qpy_file)[0]
            self.assertEqual(qc, new_circuit)
            # should still have the same parameters even if they are not used
            self.assertEqual(qc.parameters, new_circuit.parameters)

    def test_string_parameter(self):
        """Test a PauliGate instruction that has string parameters."""

        circ = QuantumCircuit(3)
        circ.z(0)
        circ.y(1)
        circ.x(2)

        qpy_file = io.BytesIO()
        dump(circ, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(circ, new_circuit)
        self.assertDeprecatedBitProperties(circ, new_circuit)

    def test_multiple_circuits(self):
        """Test multiple circuits can be serialized together."""
        circuits = []
        for i in range(10):
            circuits.append(
                random_circuit(10, 10, measure=True, conditional=True, reset=True, seed=42 + i)
            )
        qpy_file = io.BytesIO()
        dump(circuits, qpy_file)
        qpy_file.seek(0)
        new_circs = load(qpy_file)
        self.assertEqual(circuits, new_circs)
        for old, new in zip(circuits, new_circs):
            self.assertDeprecatedBitProperties(old, new)

    def test_shared_bit_register(self):
        """Test a circuit with shared bit registers."""
        qubits = [Qubit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qr = QuantumRegister(bits=qubits)
        qc.add_register(qr)
        qc.h(qr)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_qc = load(qpy_file)[0]
        self.assertEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

    def test_hybrid_standalone_register(self):
        """Test qpy serialization with registers that mix bit types"""
        qr = QuantumRegister(5, "foo")
        qr = QuantumRegister(name="bar", bits=qr[:3] + [Qubit(), Qubit()])
        cr = ClassicalRegister(5, "foo")
        cr = ClassicalRegister(name="classical_bar", bits=cr[:3] + [Clbit(), Clbit()])
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure(qr, cr)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_mixed_registers(self):
        """Test circuit with mix of standalone and shared registers."""
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qc.add_bits(clbits)
        qr = QuantumRegister(bits=qubits)
        cr = ClassicalRegister(bits=clbits)
        qc.add_register(qr)
        qc.add_register(cr)
        qr_standalone = QuantumRegister(2, "standalone")
        qc.add_register(qr_standalone)
        cr_standalone = ClassicalRegister(2, "classical_standalone")
        qc.add_register(cr_standalone)
        qc.unitary(random_unitary(32, seed=42), qr)
        qc.unitary(random_unitary(4, seed=100), qr_standalone)
        qc.measure(qr, cr)
        qc.measure(qr_standalone, cr_standalone)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_standalone_and_shared_out_of_order(self):
        """Test circuit with register bits inserted out of order."""
        qr_standalone = QuantumRegister(2, "standalone")
        qubits = [Qubit() for _ in range(5)]
        clbits = [Clbit() for _ in range(5)]
        qc = QuantumCircuit()
        qc.add_bits(qubits)
        qc.add_bits(clbits)
        random.shuffle(qubits)
        random.shuffle(clbits)
        qr = QuantumRegister(bits=qubits)
        cr = ClassicalRegister(bits=clbits)
        qc.add_register(qr)
        qc.add_register(cr)
        qr_standalone = QuantumRegister(2, "standalone")
        cr_standalone = ClassicalRegister(2, "classical_standalone")
        qc.add_bits([qr_standalone[1], qr_standalone[0]])
        qc.add_bits([cr_standalone[1], cr_standalone[0]])
        qc.add_register(qr_standalone)
        qc.add_register(cr_standalone)
        qc.unitary(random_unitary(32, seed=42), qr)
        qc.unitary(random_unitary(4, seed=100), qr_standalone)
        qc.measure(qr, cr)
        qc.measure(qr_standalone, cr_standalone)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_unitary_gate_with_label(self):
        """Test that numpy array parameters are correctly serialized with a label"""
        qc = QuantumCircuit(1)
        unitary = np.array([[0, 1], [1, 0]])
        unitary_gate = UnitaryGate(unitary, "My Special unitary")
        qc.append(unitary_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_opaque_gate_with_label(self):
        """Test that custom opaque gate is correctly serialized with a label"""
        custom_gate = Gate("black_box", 1, [])
        custom_gate.label = "My Special Black Box"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_opaque_instruction_with_label(self):
        """Test that custom opaque instruction is correctly serialized with a label"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_gate.label = "My Special Black Box Instruction"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_gate_with_label(self):
        """Test that custom  gate is correctly serialized with a label"""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        custom_gate.label = "My special black box with a definition"

        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_instruction_with_label(self):
        """Test that custom instruction is correctly serialized with a label"""
        custom_gate = Instruction("black_box", 1, 0, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition
        custom_gate.label = "My Special Black Box Instruction with a definition"
        qc = QuantumCircuit(1)
        qc.append(custom_gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_gate_with_noop_definition(self):
        """Test that a custom gate whose definition contains no elements is serialized with a
        proper definition.

        Regression test of gh-7429."""
        empty = QuantumCircuit(1, name="empty").to_gate()
        opaque = Gate("opaque", 1, [])
        qc = QuantumCircuit(2)
        qc.append(empty, [0], [])
        qc.append(opaque, [1], [])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(len(new_circ), 2)
        self.assertIsInstance(new_circ.data[0].operation.definition, QuantumCircuit)
        self.assertIs(new_circ.data[1].operation.definition, None)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_custom_instruction_with_noop_definition(self):
        """Test that a custom instruction whose definition contains no elements is serialized with a
        proper definition.

        Regression test of gh-7429."""
        empty = QuantumCircuit(1, name="empty").to_instruction()
        opaque = Instruction("opaque", 1, 0, [])
        qc = QuantumCircuit(2)
        qc.append(empty, [0], [])
        qc.append(opaque, [1], [])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertEqual(len(new_circ), 2)
        self.assertIsInstance(new_circ.data[0].operation.definition, QuantumCircuit)
        self.assertIs(new_circ.data[1].operation.definition, None)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_standard_gate_with_label(self):
        """Test a standard gate with a label."""
        qc = QuantumCircuit(1)
        gate = XGate(label="My special X gate")
        qc.append(gate, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    @ddt.data(True, False)
    def test_initialize_qft(self, use_qft_gate):
        """Test that initialize with a complex statevector and qft work."""
        k = 5
        state = (1 / np.sqrt(8)) * np.array(
            [
                np.exp(-1j * 2 * np.pi * k * (0) / 8),
                np.exp(-1j * 2 * np.pi * k * (1) / 8),
                np.exp(-1j * 2 * np.pi * k * (2) / 8),
                np.exp(-1j * 2 * np.pi * k * 3 / 8),
                np.exp(-1j * 2 * np.pi * k * 4 / 8),
                np.exp(-1j * 2 * np.pi * k * 5 / 8),
                np.exp(-1j * 2 * np.pi * k * 6 / 8),
                np.exp(-1j * 2 * np.pi * k * 7 / 8),
            ]
        )

        qubits = 3
        qc = QuantumCircuit(qubits, qubits)
        qc.initialize(state)
        if use_qft_gate:
            qft = QFTGate(qubits)
        else:
            with self.assertWarns(DeprecationWarning):
                qft = QFT(qubits)

        qc.append(qft, range(qubits))
        qc.measure(range(qubits), range(qubits))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    @ddt.data(True, False)
    def test_statepreparation(self, use_qft_gate):
        """Test that state preparation with a complex statevector and qft work."""
        k = 5
        state = (1 / np.sqrt(8)) * np.array(
            [
                np.exp(-1j * 2 * np.pi * k * (0) / 8),
                np.exp(-1j * 2 * np.pi * k * (1) / 8),
                np.exp(-1j * 2 * np.pi * k * (2) / 8),
                np.exp(-1j * 2 * np.pi * k * 3 / 8),
                np.exp(-1j * 2 * np.pi * k * 4 / 8),
                np.exp(-1j * 2 * np.pi * k * 5 / 8),
                np.exp(-1j * 2 * np.pi * k * 6 / 8),
                np.exp(-1j * 2 * np.pi * k * 7 / 8),
            ]
        )

        qubits = 3
        qc = QuantumCircuit(qubits, qubits)
        qc.prepare_state(state)
        if use_qft_gate:
            qft = QFTGate(qubits)
        else:
            with self.assertWarns(DeprecationWarning):
                qft = QFT(qubits)

        qc.append(qft, range(qubits))
        qc.measure(range(qubits), range(qubits))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_single_bit_teleportation(self):
        """Test a teleportation circuit with single bit conditions."""
        qr = QuantumRegister(1)
        cr = ClassicalRegister(2, name="name")
        qc = QuantumCircuit(qr, cr, name="Reset Test")
        qc.x(0)
        qc.measure(0, cr[0])
        with qc.if_test((cr[0], 1)):
            qc.x(0)
        qc.measure(0, cr[1])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_qaoa(self):
        """Test loading a QAOA circuit works."""
        cost_operator = Pauli("ZIIZ")
        with self.assertWarns(DeprecationWarning):
            qaoa = QAOAAnsatz(cost_operator, reps=2)

        qpy_file = io.BytesIO()
        dump(qaoa, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qaoa, new_circ)
        self.assertEqual(
            [x.operation.label for x in qaoa.data], [x.operation.label for x in new_circ.data]
        )
        self.assertDeprecatedBitProperties(qaoa, new_circ)

    def test_evolutiongate(self):
        """Test loading a circuit with evolution gate works."""
        synthesis = LieTrotter(reps=2)
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=2, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_evolutiongate_param_time(self):
        """Test loading a circuit with an evolution gate that has a parameter for time."""
        synthesis = LieTrotter(reps=2)
        time = Parameter("t")
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=time, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_evolutiongate_param_expr_time(self):
        """Test loading a circuit with an evolution gate that has a parameter for time."""
        synthesis = LieTrotter(reps=2)
        time = Parameter("t")
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=time * time, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_evolutiongate_param_vec_time(self):
        """Test loading a an evolution gate that has a param vector element for time."""
        synthesis = LieTrotter(reps=2)
        time = ParameterVector("TimeVec", 1)
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=time[0], synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_op_list_evolutiongate(self):
        """Test loading a circuit with evolution gate works."""

        evo = PauliEvolutionGate(
            [SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)])] * 5, time=0.2, synthesis=None
        )
        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_op_evolution_gate_suzuki_trotter(self):
        """Test qpy path with a suzuki trotter synthesis method on an evolution gate."""
        synthesis = SuzukiTrotter()
        evo = PauliEvolutionGate(
            SparsePauliOp.from_list([("ZI", 1), ("IZ", 1)]), time=0.2, synthesis=synthesis
        )

        qc = QuantumCircuit(2)
        qc.append(evo, range(2))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]

        self.assertEqual(qc, new_circ)
        self.assertEqual(
            [x.operation.label for x in qc.data], [x.operation.label for x in new_circ.data]
        )

        new_evo = new_circ.data[0].operation
        self.assertIsInstance(new_evo, PauliEvolutionGate)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_pauli_feature_map_legacy(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/13720."""
        # legacy construction
        with self.assertWarns(DeprecationWarning):
            qc = PauliFeatureMap(feature_dimension=5, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_pauli_feature_map_new(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/13720."""
        # new construction
        qc = pauli_feature_map(feature_dimension=5, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_zz_feature_map_legacy(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14088."""
        # legacy construction
        with self.assertWarns(DeprecationWarning):
            qc = ZZFeatureMap(2, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_zz_feature_map_new(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14088."""
        # new construction
        qc = zz_feature_map(2, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_real_amplitudes_legacy(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14088."""
        # legacy construction
        with self.assertWarns(DeprecationWarning):
            qc = RealAmplitudes(2, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_real_amplitudes_new(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14088."""
        # new construction
        qc = real_amplitudes(2, reps=1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_duplicated_param_name_legacy(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14089."""
        op = SparsePauliOp(["ZIZI", "IZIZ", "ZIIZ"])
        x = ParameterVector("γ", 1)
        # legacy construction
        with self.assertWarns(DeprecationWarning):
            ansatz = QAOAAnsatz(op, reps=1)
        ansatz = ansatz.assign_parameters({ansatz.parameters[1]: x[0]})
        qc = QuantumCircuit(4)
        qc.append(ansatz, range(4))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_duplicated_param_name_new(self):
        """Regression test for
        https://github.com/Qiskit/qiskit/issues/14089."""
        op = SparsePauliOp(["ZIZI", "IZIZ", "ZIIZ"])
        x = ParameterVector("γ", 1)
        # new construction
        ansatz = qaoa_ansatz(op, reps=1)
        ansatz = ansatz.assign_parameters({ansatz.parameters[1]: x[0]})
        qc = QuantumCircuit(4)
        qc.append(ansatz, range(4))
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_parameter_expression_global_phase(self):
        """Test a circuit with a parameter expression global_phase."""
        theta = Parameter("theta")
        phi = Parameter("phi")
        sum_param = theta + phi
        qc = QuantumCircuit(5, 1, global_phase=sum_param)
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

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_parameter_global_phase(self):
        """Test a circuit with a parameter expression global_phase."""
        theta = Parameter("theta")
        qc = QuantumCircuit(2, global_phase=theta)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_parameter_vector(self):
        """Test a circuit with a parameter vector for gate parameters."""
        qc = QuantumCircuit(11)
        input_params = ParameterVector("x_par", 11)
        user_params = ParameterVector("θ_par", 11)
        for i, param in enumerate(user_params):
            qc.ry(param, i)
        for i, param in enumerate(input_params):
            qc.rz(param, i)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        expected_params = [x.name for x in qc.parameters]
        self.assertEqual([x.name for x in new_circuit.parameters], expected_params)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_parameter_vector_equality(self):
        """Test parameter vector equality after serialization."""

        def dump_load_param_vec(qc):
            params = qc.parameters
            vector = qc.parameters[0].vector
            qpy_file = io.BytesIO()
            dump(qc, qpy_file)
            qpy_file.seek(0)
            new_circuit = load(qpy_file)[0]
            new_params = new_circuit.parameters
            new_vector = new_circuit.parameters[0].vector
            return params, new_params, vector, new_vector

        with self.subTest("manual"):
            x = ParameterVector("γ", 2)
            qc = QuantumCircuit(3)
            qc.rzz(x[0], 0, 1)
            params, new_params, vector, new_vector = dump_load_param_vec(qc)

            self.assertTrue(all(p == q for p, q in zip(params, new_params)))
            # vector[0] is part of the circuit
            self.assertTrue(vector[0] == new_vector[0])

        with self.subTest("real_amplitudes"):
            qc = real_amplitudes(2, reps=1)
            params, new_params, vector, new_vector = dump_load_param_vec(qc)
            self.assertTrue(all(p == q for p, q in zip(params, new_params)))
            self.assertTrue(all(p == q for p, q in zip(vector, new_vector)))

        with self.subTest("zz_feature_map"):
            qc = zz_feature_map(2, reps=1)
            params, new_params, vector, new_vector = dump_load_param_vec(qc)
            self.assertTrue(all(p == q for p, q in zip(params, new_params)))
            self.assertTrue(all(p == q for p, q in zip(vector, new_vector)))

    def test_parameter_vector_element_in_expression(self):
        """Test a circuit with a parameter vector used in a parameter expression."""
        qc = QuantumCircuit(7)
        entanglement = [[i, i + 1] for i in range(7 - 1)]
        input_params = ParameterVector("x_par", 14)
        user_params = ParameterVector("\u03b8_par", 1)

        for i in range(qc.num_qubits):
            qc.ry(user_params[0], qc.qubits[i])

        for source, target in entanglement:
            qc.cz(qc.qubits[source], qc.qubits[target])

        for i in range(qc.num_qubits):
            qc.rz(-2 * input_params[2 * i + 1], qc.qubits[i])
            qc.rx(-2 * input_params[2 * i], qc.qubits[i])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        expected_params = [x.name for x in qc.parameters]
        self.assertEqual([x.name for x in new_circuit.parameters], expected_params)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_parameter_vector_global_phase(self):
        """Test that a circuit with a standalone ParameterVectorElement phase works."""
        vec = ParameterVector("phase", 1)
        qc = QuantumCircuit(1, global_phase=vec[0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_custom_metadata_serializer_full_path(self):
        """Test that running with custom metadata serialization works."""

        class CustomObject:
            """Custom string container object."""

            def __init__(self, string):
                self.string = string

            def __eq__(self, other):
                return self.string == other.string

        class CustomSerializer(json.JSONEncoder):
            """Custom json encoder to handle CustomObject."""

            def default(self, o):
                if isinstance(o, CustomObject):
                    return {"__type__": "Custom", "value": o.string}
                return json.JSONEncoder.default(self, o)

        class CustomDeserializer(json.JSONDecoder):
            """Custom json decoder to handle CustomObject."""

            def object_hook(self, o):
                """Hook to override default decoder.

                Normally specified as a kwarg on load() that overloads the
                default decoder. Done here to avoid reimplementing the
                decode method.
                """
                if "__type__" in o:
                    obj_type = o["__type__"]
                    if obj_type == "Custom":
                        return CustomObject(o["value"])
                return o

        theta = Parameter("theta")
        qc = QuantumCircuit(2, global_phase=theta)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        circuits = [qc, qc.copy()]
        circuits[0].metadata = {"key": CustomObject("Circuit 1")}
        circuits[1].metadata = {"key": CustomObject("Circuit 2")}
        qpy_file = io.BytesIO()
        dump(circuits, qpy_file, metadata_serializer=CustomSerializer)
        qpy_file.seek(0)
        new_circuits = load(qpy_file, metadata_deserializer=CustomDeserializer)
        self.assertEqual(qc, new_circuits[0])
        self.assertEqual(circuits[0].metadata["key"], CustomObject("Circuit 1"))
        self.assertEqual(qc, new_circuits[1])
        self.assertEqual(circuits[1].metadata["key"], CustomObject("Circuit 2"))
        self.assertDeprecatedBitProperties(qc, new_circuits[0])
        self.assertDeprecatedBitProperties(qc, new_circuits[1])

    def test_qpy_with_ifelseop(self):
        """Test qpy serialization with an if block."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)):
            qc.x(1)
        qc.measure(1, 1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_ifelseop_with_else(self):
        """Test qpy serialization with an else block."""
        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)) as else_:
            qc.x(1)
        with else_:
            qc.y(1)
        qc.measure(1, 1)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_while_loop(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.while_loop((qc.clbits[0], 0)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_for_loop(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_for_loop_iterator(self):
        """Test qpy serialization with a for loop."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(iter(range(5))):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            with qc.if_test((0, True)):
                qc.break_loop()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_for_loop_negative_indexset(self):
        """Test qpy serialization with a for loop containing negative integers in a list indexset."""
        qc = QuantumCircuit(1)
        # Passing a list with negative integers should not raise OverflowError and should roundtrip
        qc.for_loop([-1, 0, 1], None, QuantumCircuit(1), [0], [])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(tuple(new_circuit.data[0].operation.params[0]), (-1, 0, 1))
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_with_for_loop_var_loop_parameter(self):
        """QPY faithfully round-trips expr.Range for-loops with runtime loop variables."""
        cr = ClassicalRegister(8, "cr")
        qc = QuantumCircuit(QuantumRegister(1), cr)
        range_expr = expr.Range(expr.lift(0, types.Uint(8)), expr.lift(4, types.Uint(8)))
        with qc.for_loop(range_expr) as loop_var:
            qc.store(expr.index(cr, loop_var), True)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        loaded = load(qpy_file)[0]

        self.assertEqual(qc, loaded)
        self.assertDeprecatedBitProperties(qc, loaded)

        from qiskit.qasm3 import dumps

        loaded_for_loop = next(
            inst.operation for inst in loaded.data if inst.operation.name == "for_loop"
        )
        loop_param = loaded_for_loop.params[1]
        self.assertIsInstance(loop_param, expr.Var)
        self.assertIn(f"for uint[8] {loop_param.name} in", dumps(loaded))

    def test_qpy_with_for_loop_var_loop_counter(self):
        """Test qpy serialization with a for loop that uses expr.Var as its counter."""
        qc = QuantumCircuit(1, 1)
        cr = ClassicalRegister(5, "reps")
        qc.add_register(cr)

        with qc.for_loop(range(5), expr.Var.new("a", types.Uint(32))) as v:
            qc.measure(0, 0)
            qc.store(expr.index(cr, v), qc.clbits[0])

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_clbit_switch(self):
        """Test QPY serialization for a switch statement with a Clbit target."""
        case_t = QuantumCircuit(2, 1)
        case_t.x(0)
        case_f = QuantumCircuit(2, 1)
        case_f.z(0)
        qc = QuantumCircuit(2, 1)
        qc.switch(0, [(True, case_t), (False, case_f)], qc.qubits, qc.clbits)

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]

        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_qpy_register_switch(self):
        """Test QPY serialization for a switch statement with a ClassicalRegister target."""
        qreg = QuantumRegister(2, "q")
        creg = ClassicalRegister(3, "c")

        case_0 = QuantumCircuit(qreg, creg)
        case_0.x(0)
        case_1 = QuantumCircuit(qreg, creg)
        case_1.z(1)
        case_2 = QuantumCircuit(qreg, creg)
        case_2.x(1)

        qc = QuantumCircuit(qreg, creg)
        qc.switch(creg, [(0, case_0), ((1, 2), case_1), ((3, 4, CASE_DEFAULT), case_2)], qreg, creg)

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_standalone_register_partial_bit_in_circuit(self):
        """Test qpy with only some bits from standalone register."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit([qr[0]])
        qc.x(0)
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_nested_tuple_param(self):
        """Test qpy with an instruction that contains nested tuples."""
        inst = Instruction("tuple_test", 1, 0, [((((0, 1), (0, 1)), 2, 3), ("A", "B", "C"))])
        qc = QuantumCircuit(1)
        qc.append(inst, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_empty_tuple_param(self):
        """Test qpy with an instruction that contains an empty tuple."""
        inst = Instruction("empty_tuple_test", 1, 0, [()])
        qc = QuantumCircuit(1)
        qc.append(inst, [0])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_ucr_gates(self):
        """Test qpy with UCRX, UCRY, and UCRZ gates."""
        qc = QuantumCircuit(3)
        angles = [0, 0, 0, -np.pi]
        ucrx, ucry, ucrz = UCRXGate(angles), UCRYGate(angles), UCRZGate(angles)
        qc.append(ucrz, [2, 0, 1])
        qc.append(ucry, [1, 0, 2])
        qc.append(ucrx, [0, 2, 1])
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc.decompose().decompose(), new_circuit.decompose().decompose())
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_controlled_gate(self):
        """Test a custom controlled gate."""
        qc = QuantumCircuit(3)
        controlled_gate = DCXGate().control(1, annotated=False)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_controlled_gate_open_controls(self):
        """Test a controlled gate with open controls round-trips exactly."""
        qc = QuantumCircuit(3)
        controlled_gate = DCXGate().control(1, ctrl_state=0, annotated=False)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_nested_controlled_gate(self):
        """Test a custom nested controlled gate."""
        custom_gate = Gate("black_box", 1, [])
        custom_definition = QuantumCircuit(1)
        custom_definition.h(0)
        custom_definition.rz(1.5, 0)
        custom_definition.sdg(0)
        custom_gate.definition = custom_definition

        qc = QuantumCircuit(3)
        qc.append(custom_gate, [0])
        controlled_gate = custom_gate.control(2, annotated=False)
        qc.append(controlled_gate, [0, 1, 2])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_open_controlled_gate(self):
        """Test an open control is preserved across serialization."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1, ctrl_state=0)
        with io.BytesIO() as fd:
            dump(qc, fd)
            fd.seek(0)
            new_circ = load(fd)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.data[0].operation.ctrl_state, new_circ.data[0].operation.ctrl_state)
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_standard_control_gates(self):
        """Test standard library controlled gates."""
        qc = QuantumCircuit(6)
        mcu1_gate = MCU1Gate(np.pi, 2)
        mcx_gate = MCXGate(5)
        with self.assertWarns(DeprecationWarning):
            mcx_gray_gate = MCXGrayCode(5)
        with self.assertWarns(DeprecationWarning):
            mcx_recursive_gate = MCXRecursive(4)
        with self.assertWarns(DeprecationWarning):
            mcx_vchain_gate = MCXVChain(3)
        mcmt_gate = MCMTGate(ZGate(), 2, 1)
        qc.append(mcu1_gate, [0, 2, 1])
        qc.append(mcx_gate, list(range(6)))
        qc.append(mcx_gray_gate, list(range(6)))
        qc.append(mcx_recursive_gate, list(range(5)))
        qc.append(mcx_vchain_gate, list(range(5)))
        qc.append(mcmt_gate, list(range(3)))
        qc.mcp(np.pi, [0, 2], 1)
        qc.mcx([0, 2], 1)
        qc.measure_all()
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        with self.assertWarns(DeprecationWarning):
            new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_controlled_gate_subclass_custom_definition(self):
        """Test controlled gate with overloaded definition.

        Reproduce from: https://github.com/Qiskit/qiskit-terra/issues/8794
        """

        class CustomCXGate(ControlledGate):
            """Custom CX with overloaded _define."""

            def __init__(self, label=None, ctrl_state=None):
                super().__init__(
                    "cx", 2, [], label, num_ctrl_qubits=1, ctrl_state=ctrl_state, base_gate=XGate()
                )

            def _define(self) -> None:
                qc = QuantumCircuit(2, name=self.name)
                qc.cx(0, 1)
                self.definition = qc

        qc = QuantumCircuit(2)
        qc.append(CustomCXGate(), [0, 1])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_multiple_controlled_gates(self):
        """Test multiple controlled gates with same name but different
        parameter values.

        Reproduce from: https://github.com/Qiskit/qiskit-terra/issues/10735
        """

        qc = QuantumCircuit(3)
        for i in range(3):
            c2ry = RYGate(i + 1).control(2, annotated=False)
            qc.append(c2ry, [i % 3, (i + 1) % 3, (i + 2) % 3])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(qc, new_circ)
        self.assertEqual(qc.decompose(), new_circ.decompose())
        self.assertDeprecatedBitProperties(qc, new_circ)

    def test_load_with_loose_bits(self):
        """Test that loading from a circuit with loose bits works."""
        qc = QuantumCircuit([Qubit(), Qubit(), Clbit()])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(tuple(new_circuit.qregs), ())
        self.assertEqual(tuple(new_circuit.cregs), ())
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_load_with_loose_bits_and_registers(self):
        """Test that loading from a circuit with loose bits and registers works."""
        qc = QuantumCircuit(QuantumRegister(3), ClassicalRegister(1), [Clbit()])
        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_registers_after_loose_bits(self):
        """Test that a circuit whose registers appear after some loose bits roundtrips. Regression
        test of gh-9094."""
        qc = QuantumCircuit()
        qc.add_bits([Qubit(), Clbit()])
        qc.add_register(QuantumRegister(2, name="q1"))
        qc.add_register(ClassicalRegister(2, name="c1"))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_roundtrip_empty_register(self):
        """Test that empty registers round-trip correctly."""
        qc = QuantumCircuit(QuantumRegister(0), ClassicalRegister(0))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_roundtrip_several_empty_registers(self):
        """Test that several empty registers round-trip correctly."""
        qc = QuantumCircuit(
            QuantumRegister(0, "a"),
            QuantumRegister(0, "b"),
            ClassicalRegister(0, "c"),
            ClassicalRegister(0, "d"),
        )
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_roundtrip_empty_registers_with_loose_bits(self):
        """Test that empty registers still round-trip correctly in the presence of loose bits."""
        loose = [Qubit(), Clbit()]

        qc = QuantumCircuit(loose, QuantumRegister(0), ClassicalRegister(0))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)

        qc = QuantumCircuit(QuantumRegister(0), ClassicalRegister(0), loose)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_incomplete_owned_bits(self):
        """Test that a circuit that contains only some bits that are owned by a register are
        correctly roundtripped."""
        reg = QuantumRegister(5, "q")
        qc = QuantumCircuit(reg[:3])
        qc.ccx(0, 1, 2)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_diagonal_gate(self):
        """Test that a `DiagonalGate` successfully roundtrips."""
        diag = DiagonalGate([1, -1, -1, 1])

        qc = QuantumCircuit(2)
        qc.append(diag, [0, 1])

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        # DiagonalGate (and some of the qiskit.circuit.library gates) have non-deterministic
        # definitions with regard to internal instruction names, so cannot be directly compared for
        # equality.
        self.assertIs(type(qc.data[0].operation), type(new_circuit.data[0].operation))
        self.assertEqual(qc.data[0].operation.params, new_circuit.data[0].operation.params)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    @ddt.data(QuantumCircuit.if_test, QuantumCircuit.while_loop)
    def test_if_else_while_expr_simple(self, control_flow):
        """Test that `IfElseOp` and `WhileLoopOp` can have an `Expr` node as their `condition`, and
        that this round-trips through QPY."""
        body = QuantumCircuit(1)
        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        control_flow(qc, expr.equal(cr, 3), body.copy(), [0], [])
        control_flow(qc, expr.lift(qc.clbits[0]), body.copy(), [0], [])
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    @ddt.data(QuantumCircuit.if_test, QuantumCircuit.while_loop)
    def test_if_else_while_expr_nested(self, control_flow):
        """Test that `IfElseOp` and `WhileLoopOp` can have an `Expr` node as their `condition`, and
        that this round-trips through QPY."""
        inner = QuantumCircuit(1)
        inner.x(0)
        outer = QuantumCircuit(1, 1)
        control_flow(outer, expr.lift(outer.clbits[0]), inner.copy(), [0], [])

        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        control_flow(qc, expr.equal(cr, 3), outer.copy(), [1], [1])
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_if_else_expr_stress(self):
        """Stress-test the `Expr` handling in the condition of an `IfElseOp`.  This should hit on
        every aspect of the `Expr` tree."""
        inner = QuantumCircuit(1)
        inner.x(0)

        outer = QuantumCircuit(1, 1)
        outer.if_test(expr.cast(outer.clbits[0], types.Bool()), inner.copy(), [0], [])

        # Register whose size is deliberately larger that one byte.
        cr1 = ClassicalRegister(256, "c1")
        cr2 = ClassicalRegister(4, "c2")
        loose = Clbit()
        qc = QuantumCircuit([Qubit(), Qubit(), loose], cr1, cr2)
        qc.rz(1.0, 0)
        qc.if_test(
            expr.logic_and(
                expr.logic_and(
                    expr.logic_or(
                        expr.cast(
                            expr.less(expr.bit_and(cr1, 0x0F), expr.bit_not(cr1)),
                            types.Bool(),
                        ),
                        expr.cast(
                            expr.less_equal(expr.bit_or(cr2, 7), expr.bit_xor(cr2, 7)),
                            types.Bool(),
                        ),
                    ),
                    expr.logic_and(
                        expr.logic_or(expr.equal(cr2, 2), expr.logic_not(expr.not_equal(cr2, 3))),
                        expr.logic_or(
                            expr.greater(cr2, 3),
                            expr.greater_equal(cr2, 3),
                        ),
                    ),
                ),
                expr.greater(expr.negate(expr.cast(loose, types.Float())), 1.5),
            ),
            outer.copy(),
            [1],
            [0],
        )
        qc.rz(1.0, 0)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_switch_expr_simple(self):
        """Test that `SwitchCaseOp` can have an `Expr` node as its `target`, and that this
        round-trips through QPY."""
        body = QuantumCircuit(1)
        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.bit_and(cr, 3), [(1, body.copy())], [0], [])
        qc.switch(expr.logic_not(qc.clbits[0]), [(False, body.copy())], [0], [])
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_switch_expr_nested(self):
        """Test that `SwitchCaseOp` can have an `Expr` node as its `target`, and that this
        round-trips through QPY."""
        inner = QuantumCircuit(1)
        outer = QuantumCircuit(1, 1)
        outer.switch(expr.lift(outer.clbits[0]), [(False, inner.copy())], [0], [])

        qr = QuantumRegister(2, "q1")
        cr = ClassicalRegister(2, "c1")
        qc = QuantumCircuit(qr, cr)
        qc.switch(expr.lift(cr), [(3, outer.copy())], [1], [1])
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_switch_expr_stress(self):
        """Stress-test the `Expr` handling in the target of a `SwitchCaseOp`.  This should hit on
        every aspect of the `Expr` tree."""
        inner = QuantumCircuit(1)
        inner.x(0)

        outer = QuantumCircuit(1, 1)
        outer.switch(expr.cast(outer.clbits[0], types.Bool()), [(True, inner.copy())], [0], [])

        # Register whose size is deliberately larger that one byte.
        cr1 = ClassicalRegister(256, "c1")
        cr2 = ClassicalRegister(4, "c2")
        loose = Clbit()
        qc = QuantumCircuit([Qubit(), Qubit(), loose], cr1, cr2)
        qc.rz(1.0, 0)
        qc.switch(
            expr.logic_and(
                expr.logic_and(
                    expr.logic_or(
                        expr.cast(
                            expr.less(expr.bit_and(cr1, 0x0F), expr.bit_not(cr1)),
                            types.Bool(),
                        ),
                        expr.cast(
                            expr.less_equal(expr.bit_or(cr2, 7), expr.bit_xor(cr2, 7)),
                            types.Bool(),
                        ),
                    ),
                    expr.logic_and(
                        expr.logic_or(expr.equal(cr2, 2), expr.logic_not(expr.not_equal(cr2, 3))),
                        expr.logic_or(
                            expr.greater(cr2, 3),
                            expr.greater_equal(cr2, 3),
                        ),
                    ),
                ),
                expr.greater(expr.negate(expr.cast(loose, types.Float())), 1.5),
            ),
            [(False, outer.copy())],
            [1],
            [0],
        )
        qc.rz(1.0, 0)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertEqual(qc.qregs, new_circuit.qregs)
        self.assertEqual(qc.cregs, new_circuit.cregs)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_box(self):
        """Test that box, including duration, unit and label roundtrips."""
        qc = QuantumCircuit(2)
        with qc.box():  # Instruction 0
            qc.cx(0, 1)
        with qc.box(duration=1, unit="dt", label="hello"):  # Instruction 1
            with qc.box(duration=2.5, unit="s", label="world"):  # Instruction 1-0
                qc.cx(0, 1)

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            out = load(fptr)[0]

        self.assertEqual(qc, out)
        self.assertDeprecatedBitProperties(qc, out)

        # ... and a couple of manual checks, to be extra sure we check things not in `__eq__`.
        box_0 = out.data[0].operation
        self.assertIsNone(box_0.duration)
        self.assertIsNone(box_0.label)

        box_1 = out.data[1].operation
        self.assertEqual(box_1.duration, 1)
        self.assertEqual(box_1.unit, "dt")
        self.assertEqual(box_1.label, "hello")

        box_1_0 = box_1.blocks[0].data[0].operation
        self.assertEqual(box_1_0.duration, 2.5)
        self.assertEqual(box_1_0.unit, "s")
        self.assertEqual(box_1_0.label, "world")

    def test_box_with_stretch(self):
        """Test that box's duration and unit round-trip with stretches."""
        qc = QuantumCircuit(2)
        a = qc.add_stretch("a")
        b = qc.add_stretch("b")
        with qc.box(duration=a):
            with qc.box(duration=expr.mul(2, b)):
                pass

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            out = load(fptr)[0]

        box_outer = out.data[0].operation
        self.assertEqual(box_outer.duration, a)
        self.assertEqual(box_outer.unit, "expr")
        box_inner = box_outer.blocks[0].data[0].operation
        self.assertEqual(box_inner.duration, expr.mul(2, b))
        self.assertEqual(box_inner.unit, "expr")
        self.assertEqual(qc, out)

    def test_multiple_nested_control_custom_definitions(self):
        """Test that circuits with multiple controlled custom gates that in turn depend on custom
        gates can be exported successfully when there are several such gates in the outer circuit.
        See gh-9746"""
        inner_1 = QuantumCircuit(1, name="inner_1")
        inner_1.x(0)
        inner_2 = QuantumCircuit(1, name="inner_2")
        inner_2.y(0)

        outer_1 = QuantumCircuit(1, name="outer_1")
        outer_1.append(inner_1.to_gate(), [0], [])
        outer_2 = QuantumCircuit(1, name="outer_2")
        outer_2.append(inner_2.to_gate(), [0], [])

        qc = QuantumCircuit(2)
        qc.append(outer_1.to_gate().control(1, annotated=False), [0, 1], [])
        qc.append(outer_2.to_gate().control(1, annotated=False), [0, 1], [])

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    @ddt.data(0, "01", [1, 0, 0, 0])
    def test_valid_circuit_with_initialize_instruction(self, param):
        """Tests that circuit that has initialize instruction can be saved and correctly retrieved"""
        qc = QuantumCircuit(2)
        qc.initialize(param, qc.qubits)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(qc, new_circuit)
        self.assertDeprecatedBitProperties(qc, new_circuit)

    def test_clifford(self):
        """Test that circuits with Clifford operations can be saved and retrieved correctly."""
        cliff1 = Clifford.from_dict(
            {
                "stabilizer": ["-IZX", "+ZYZ", "+ZII"],
                "destabilizer": ["+ZIZ", "+ZXZ", "-XIX"],
            }
        )
        cliff2 = Clifford.from_dict(
            {
                "stabilizer": ["+YX", "+ZZ"],
                "destabilizer": ["+IZ", "+YI"],
            }
        )

        circuit = QuantumCircuit(6, 1)
        circuit.cx(0, 1)
        circuit.append(cliff1, [2, 4, 5])
        circuit.h(4)
        circuit.append(cliff2, [3, 0])

        with io.BytesIO() as fptr:
            dump(circuit, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(circuit, new_circuit)

    def test_annotated_operations(self):
        """Test that circuits with annotated operations can be saved and retrieved correctly."""
        op1 = AnnotatedOperation(
            CXGate(), [InverseModifier(), ControlModifier(1), PowerModifier(1.4), InverseModifier()]
        )
        op2 = AnnotatedOperation(XGate(), InverseModifier())

        circuit = QuantumCircuit(6, 1)
        circuit.cx(0, 1)
        circuit.append(op1, [0, 1, 2])
        circuit.h(4)
        circuit.append(op2, [1])

        with io.BytesIO() as fptr:
            dump(circuit, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(circuit, new_circuit)

    def test_annotated_operations_iterative(self):
        """Test that circuits with iterative annotated operations can be saved and
        retrieved correctly.
        """
        op = AnnotatedOperation(AnnotatedOperation(XGate(), InverseModifier()), ControlModifier(1))
        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.append(op, [0, 2])
        circuit.cx(2, 3)
        with io.BytesIO() as fptr:
            dump(circuit, fptr)
            fptr.seek(0)
            new_circuit = load(fptr)[0]
        self.assertEqual(circuit, new_circuit)

    def test_load_empty_vars(self):
        """Test loading empty circuits with variables."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        all_vars = {
            a: expr.lift(False),
            b: expr.lift(3, type=b.type),
            expr.Var.new("θψφ", types.Bool()): expr.logic_not(a),
            expr.Var.new("🐍🐍🐍", types.Uint(8)): expr.bit_and(b, b),
        }

        inputs = QuantumCircuit(inputs=list(all_vars))
        with io.BytesIO() as fptr:
            dump(inputs, fptr)
            fptr.seek(0)
            new_inputs = load(fptr)[0]
        self.assertMinimalVarEqual(inputs, new_inputs)
        self.assertDeprecatedBitProperties(inputs, new_inputs)

        # Reversed order just to check there's no sorting shenanigans.
        captures = QuantumCircuit(captures=list(all_vars)[::-1])
        with io.BytesIO() as fptr:
            dump(captures, fptr)
            fptr.seek(0)
            new_captures = load(fptr)[0]
        self.assertMinimalVarEqual(captures, new_captures)
        self.assertDeprecatedBitProperties(captures, new_captures)

        declares = QuantumCircuit(declarations=all_vars)
        with io.BytesIO() as fptr:
            dump(declares, fptr)
            fptr.seek(0)
            new_declares = load(fptr)[0]
        self.assertMinimalVarEqual(declares, new_declares)
        self.assertDeprecatedBitProperties(declares, new_declares)

    def test_load_empty_vars_if(self):
        """Test loading circuit with vars in if/else closures."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("θψφ", types.Bool())
        c = expr.Var.new("c", types.Uint(8))
        d = expr.Var.new("🐍🐍🐍", types.Uint(8))

        qc = QuantumCircuit(inputs=[a])
        qc.add_var(b, expr.logic_not(a))
        qc.add_var(c, expr.lift(0, c.type))
        with qc.if_test(b) as else_:
            qc.store(c, expr.lift(3, c.type))
        with else_:
            qc.add_var(d, expr.lift(7, d.type))

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
        self.assertMinimalVarEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

        old_if_else = qc.data[-1].operation
        new_if_else = new_qc.data[-1].operation
        # Sanity check for test.
        self.assertIsInstance(old_if_else, IfElseOp)
        self.assertIsInstance(new_if_else, IfElseOp)
        self.assertEqual(len(old_if_else.blocks), len(new_if_else.blocks))

        for old, new in zip(old_if_else.blocks, new_if_else.blocks):
            self.assertMinimalVarEqual(old, new)
            self.assertDeprecatedBitProperties(old, new)

    def test_register_condition(self):
        """Test that using register condition passes as long as the register index is not too large"""
        n_qubits = 2
        cr0 = ClassicalRegister(n_qubits)
        cr1 = ClassicalRegister(n_qubits)
        qr = QuantumRegister(n_qubits)
        qc = QuantumCircuit(qr, cr0, cr1)
        qc.x(qr)
        qc.measure(qr, cr0)
        with qc.if_test((cr0, 0)):
            qc.x(qr)
        qc.measure(qr, cr1)
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
            self.assertEqual(qc, new_qc)

        qc = QuantumCircuit(qr, cr0, cr1)
        qc.x(qr)
        qc.measure(qr, cr0)
        with qc.if_test((cr0, 7342574385754343653453453453533935345)):
            qc.x(qr)
        qc.measure(qr, cr1)
        with self.assertRaisesRegex(QpyError, "exceeds i64::MAX"):
            with io.BytesIO() as fptr:
                dump(qc, fptr)

    def test_load_empty_vars_while(self):
        """Test loading circuit with vars in while closures."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("θψφ", types.Bool())
        c = expr.Var.new("🐍🐍🐍", types.Uint(8))

        qc = QuantumCircuit(inputs=[a])
        qc.add_var(b, expr.logic_not(a))
        with qc.while_loop(b):
            qc.add_var(c, expr.lift(7, c.type))

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
        self.assertMinimalVarEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

        old_while = qc.data[-1].operation
        new_while = new_qc.data[-1].operation
        # Sanity check for test.
        self.assertIsInstance(old_while, WhileLoopOp)
        self.assertIsInstance(new_while, WhileLoopOp)
        self.assertEqual(len(old_while.blocks), len(new_while.blocks))

        for old, new in zip(old_while.blocks, new_while.blocks):
            self.assertMinimalVarEqual(old, new)
            self.assertDeprecatedBitProperties(old, new)

    def test_load_empty_vars_switch(self):
        """Test loading circuit with vars in switch closures."""
        a = expr.Var.new("🐍🐍🐍", types.Uint(8))

        qc = QuantumCircuit(1, 1, inputs=[a])
        qc.measure(0, 0)
        b_outer = qc.add_var("b", False)
        with qc.switch(a) as case:
            with case(0):
                qc.store(b_outer, True)
            with case(1):
                qc.store(qc.clbits[0], False)
            with case(2):
                # Explicit shadowing.
                qc.add_var("b", True)
            with case(3):
                qc.store(a, expr.lift(1, a.type))
            with case(case.DEFAULT):
                pass

        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
        self.assertMinimalVarEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

        old_switch = qc.data[-1].operation
        new_switch = new_qc.data[-1].operation
        # Sanity check for test.
        self.assertIsInstance(old_switch, SwitchCaseOp)
        self.assertIsInstance(new_switch, SwitchCaseOp)
        self.assertEqual(len(old_switch.blocks), len(new_switch.blocks))

        for old, new in zip(old_switch.blocks, new_switch.blocks):
            self.assertMinimalVarEqual(old, new)
            self.assertDeprecatedBitProperties(old, new)

    def test_roundtrip_index_expr(self):
        """Test that the `Index` node round-trips."""
        a = expr.Var.new("a", types.Uint(8))
        cr = ClassicalRegister(4, "cr")
        qc = QuantumCircuit(cr, inputs=[a])
        qc.store(expr.index(cr, 0), expr.index(a, a))
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
        self.assertEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

    def test_roundtrip_bitshift_expr(self):
        """Test that bit-shift expressions can round-trip."""
        a = expr.Var.new("a", types.Uint(8))
        cr = ClassicalRegister(4, "cr")
        qc = QuantumCircuit(cr, inputs=[a])
        with qc.if_test(expr.equal(expr.shift_right(expr.shift_left(a, 1), 1), a)):
            pass
        with io.BytesIO() as fptr:
            dump(qc, fptr)
            fptr.seek(0)
            new_qc = load(fptr)[0]
        self.assertEqual(qc, new_qc)
        self.assertDeprecatedBitProperties(qc, new_qc)

    @ddt.idata(range(QPY_COMPATIBILITY_VERSION, 12))
    def test_pre_v12_rejects_standalone_var(self, version):
        """Test that dumping to older QPY versions rejects standalone vars."""
        a = expr.Var.new("a", types.Bool())
        qc = QuantumCircuit(inputs=[a])
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(
                UnsupportedFeatureForVersion, "version 12 is required.*realtime variables"
            ),
        ):
            dump(qc, fptr, version=version)

    @ddt.idata(range(QPY_COMPATIBILITY_VERSION, 12))
    def test_pre_v12_rejects_index(self, version):
        """Test that dumping to older QPY versions rejects the `Index` node."""
        # Be sure to use a register, since standalone vars would be rejected for other reasons.
        qc = QuantumCircuit(ClassicalRegister(2, "cr"))
        qc.store(expr.index(qc.cregs[0], 0), False)
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(UnsupportedFeatureForVersion, "version 12 is required.*Index"),
        ):
            dump(qc, fptr, version=version)

    @ddt.idata(range(QPY_COMPATIBILITY_VERSION, 14))
    def test_pre_v14_rejects_float_typed_expr(self, version):
        """Test that dumping to older QPY versions rejects float-typed expressions."""
        qc = QuantumCircuit()
        with qc.if_test(expr.less(1.0, 2.0)):
            pass
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(UnsupportedFeatureForVersion, "version 14 is required.*float"),
        ):
            dump(qc, fptr, version=version)

    @ddt.idata(range(QPY_COMPATIBILITY_VERSION, 14))
    def test_pre_v14_rejects_duration_typed_expr(self, version):
        """Test that dumping to older QPY versions rejects duration-typed expressions."""
        from qiskit.circuit import Duration

        qc = QuantumCircuit()
        with qc.if_test(expr.less(Duration.dt(10), Duration.dt(100))):
            pass
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(
                UnsupportedFeatureForVersion, "version 14 is required.*duration"
            ),
        ):
            dump(qc, fptr, version=version)

    @ddt.idata(range(QPY_COMPATIBILITY_VERSION, 14))
    def test_pre_v14_rejects_stretch_expr(self, version):
        """Test that dumping to older QPY versions rejects duration-typed expressions."""
        qc = QuantumCircuit()
        qc.add_stretch("a")
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(UnsupportedFeatureForVersion, "version 14 is required.*stretch"),
        ):
            dump(qc, fptr, version=version)

    @ddt.idata(range(max(QPY_COMPATIBILITY_VERSION, 14), 16))
    def test_pre_v16_rejects_ps_duration(self, version):
        """Test that dumping to older QPY versions rejects Duration.ps."""
        from qiskit.circuit import Duration

        qc = QuantumCircuit()
        with qc.if_test(expr.less(Duration.ns(1000), Duration.ps(1))):
            pass
        with (
            io.BytesIO() as fptr,
            self.assertRaisesRegex(
                UnsupportedFeatureForVersion, "version 16 is required.*Duration.ps"
            ),
        ):
            dump(qc, fptr, version=version)

    def test_roundtrip_from_gzip(self):
        """Test that we can write out and read from a stdlib gzip file.

        Regression test of https://github.com/Qiskit/qiskit/issues/15157."""
        qc = QuantumCircuit(2, 2, name="first")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(qc.qubits, qc.clbits)
        circuits = [qc, qc.copy(name="second")]
        with tempfile.TemporaryFile() as base_fptr:
            with gzip.open(base_fptr, mode="w+b") as fptr:
                dump(circuits, fptr)
            base_fptr.seek(0)
            with gzip.open(base_fptr, mode="r+b") as fptr:
                out_circuits = load(fptr)
        self.assertEqual(circuits, out_circuits)
        self.assertEqual([qc.name for qc in circuits], [qc.name for qc in out_circuits])

    @ddt.idata(range(max(QPY_COMPATIBILITY_VERSION, 13), QPY_VERSION + 1))
    def test_ancilla_register_with_physical(self, version):
        """Test for possible conflicts when naming a register 'ancilla'"""
        qc = QuantumCircuit(QuantumRegister(2, "qr"), QuantumRegister(2, "ancilla"))
        qc.ensure_physical(qc.num_qubits + 1)

        with io.BytesIO() as fptr:
            dump(qc, fptr, version=version)
            fptr.seek(0)
            out_circuit = load(fptr)[0]

        self.assertEqual(qc, out_circuit)

        qc = QuantumCircuit(QuantumRegister(2, "qr"), QuantumRegister(2, "ancilla"))
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 2)
        qc = transpile(
            qc,
            target=Target.from_configuration(
                basis_gates=["sx", "rz", "cz"], coupling_map=CouplingMap.from_line(5)
            ),
            optimization_level=1,
        )

        with io.BytesIO() as fptr:
            dump(qc, fptr, version=17)
            fptr.seek(0)
            out_circuit = load(fptr)[0]
        self.assertEqual(qc, out_circuit)


class TestSymengineLoadFromQPY(QiskitTestCase):
    """Test use of symengine in qpy set of methods."""

    def setUp(self):
        super().setUp()

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

        self.qc = qc

    def assertDeprecatedBitProperties(self, original, roundtripped):
        """Test that deprecated bit attributes are equal if they are set in the original circuit."""
        owned_qubits = [
            (a, b) for a, b in zip(original.qubits, roundtripped.qubits) if a._register is not None
        ]
        if owned_qubits:
            original_qubits, roundtripped_qubits = zip(*owned_qubits)
            self.assertEqual(original_qubits, roundtripped_qubits)
        owned_clbits = [
            (a, b) for a, b in zip(original.clbits, roundtripped.clbits) if a._register is not None
        ]
        if owned_clbits:
            original_clbits, roundtripped_clbits = zip(*owned_clbits)
            self.assertEqual(original_clbits, roundtripped_clbits)

    @unittest.skipIf(not optionals.HAS_SYMENGINE, "Install symengine to run this test.")
    def test_symengine_full_path(self):
        """Test use_symengine option for circuit with parameter expressions."""
        qpy_file = io.BytesIO()
        dump(self.qc, qpy_file, use_symengine=True)
        qpy_file.seek(0)
        new_circ = load(qpy_file)[0]
        self.assertEqual(self.qc, new_circ)
        self.assertDeprecatedBitProperties(self.qc, new_circ)
