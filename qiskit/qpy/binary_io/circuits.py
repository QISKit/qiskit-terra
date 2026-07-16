# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Binary IO for circuit objects."""

from __future__ import annotations

import itertools
from collections import defaultdict
import io
import json
import struct
import uuid
import typing
import warnings

import numpy as np


from qiskit import circuit as circuit_mod
from qiskit.circuit import library, controlflow, CircuitInstruction, ControlFlowOp, IfElseOp
from qiskit.circuit.annotation import iter_namespaces
from qiskit.circuit.classical import expr
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit.gate import Gate
from qiskit.circuit.singleton import SingletonInstruction, SingletonGate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    Modifier,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister, Qubit
from qiskit.qpy import common, formats, type_keys
from qiskit.qpy.exceptions import QpyError, UnsupportedFeatureForVersion
from qiskit.qpy.binary_io import value, schedules
from qiskit.quantum_info.operators import SparsePauliOp, Clifford
from qiskit.quantum_info import SparseObservable
from qiskit.circuit.library import PauliProductMeasurement
from qiskit.synthesis import evolution as evo_synth
from qiskit.transpiler.layout import Layout, TranspileLayout
from qiskit._accelerate import qpy as _qpy

if typing.TYPE_CHECKING:
    from qiskit.circuit.annotation import QPYSerializer, Annotation


class _AnnotationSerializationState:
    def __init__(self, factories: dict[str, typing.Callable[[], QPYSerializer]]):
        self.factories = factories
        self.serializers = {}
        self.potential_serializers = {}

    @property
    def num_serializers(self) -> int:
        """The number of constructed serializers."""
        return len(self.serializers)

    def serialize(self, annotation: Annotation) -> (int, bytes):
        """Serialize an annotation using a known serializer (initializing one, if necessary).

        Returns the index of the serializer used, and the serialized annotation."""
        for namespace in iter_namespaces(annotation.namespace):
            if (existing := self.serializers.get(namespace, None)) is not None:
                index, serializer = existing
                if (out := serializer.dump_annotation(namespace, annotation)) is not NotImplemented:
                    return index, out
            if (serializer := self.potential_serializers.get(namespace, None)) is not None:
                if (out := serializer.dump_annotation(namespace, annotation)) is not NotImplemented:
                    del self.potential_serializers[namespace]
                    index = len(self.serializers)
                    self.serializers[namespace] = (index, serializer)
                    return index, out
            if (factory := self.factories.get(namespace, None)) is not None:
                serializer = factory()
                if (out := serializer.dump_annotation(namespace, annotation)) is NotImplemented:
                    self.potential_serializers[namespace] = serializer
                else:
                    index = len(self.serializers)
                    self.serializers[namespace] = (index, serializer)
                    return index, out
        raise QpyError(f"No configured annotation serializer could handle {annotation}")

    def iter_serializers(self) -> typing.Iterator[tuple[str, QPYSerializer]]:
        """Iterate over the namespaces and serializers that have had at least one successful use, in
        order of first use."""
        return (
            # Python dictionaries are insertion ordered, and we assign indices in insertion order.
            (namespace, serializer)
            for (namespace, (_, serializer)) in self.serializers.items()
        )


class _AnnotationDeserializationState:
    def __init__(self, factories: dict[str, typing.Callable[[], QPYSerializer]]):
        self.factories = factories
        self.deserializers = []

    def initialize(self, namespace: str, payload: bytes):
        """Initialize a suitable deserializer using the given state payload."""
        for parent_namespace in iter_namespaces(namespace):
            if (factory := self.factories.get(parent_namespace, None)) is not None:
                deserializer = factory()
                deserializer.load_state(namespace, payload)
                self.deserializers.append(deserializer)
                return
        raise QpyError(f"No configured annotation deserializer matched namespace '{namespace}'")

    def load(self, index: int, payload: bytes) -> Annotation:
        """Load a payload using the deserializer of a given index."""
        return self.deserializers[index].load_annotation(payload)


def _read_header_v12(file_obj, version, vectors, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER_V12._make(
        struct.unpack(
            formats.CIRCUIT_HEADER_V12_PACK, file_obj.read(formats.CIRCUIT_HEADER_V12_SIZE)
        )
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    global_phase = value.loads_value(
        data.global_phase_type,
        file_obj.read(data.global_phase_size),
        version=version,
        vectors=vectors,
    )
    header = {
        "global_phase": global_phase,
        "num_qubits": data.num_qubits,
        "num_clbits": data.num_clbits,
        "num_registers": data.num_registers,
        "num_instructions": data.num_instructions,
        "num_vars": data.num_vars,
    }
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return header, name, metadata


def _read_header_v2(file_obj, version, vectors, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER_V2._make(
        struct.unpack(
            formats.CIRCUIT_HEADER_V2_PACK,
            file_obj.read(formats.CIRCUIT_HEADER_V2_SIZE),
        )
    )

    name = file_obj.read(data.name_size).decode(common.ENCODE)
    global_phase = value.loads_value(
        data.global_phase_type,
        file_obj.read(data.global_phase_size),
        version=version,
        vectors=vectors,
    )
    header = {
        "global_phase": global_phase,
        "num_qubits": data.num_qubits,
        "num_clbits": data.num_clbits,
        "num_registers": data.num_registers,
        "num_instructions": data.num_instructions,
    }

    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return header, name, metadata


def _read_header(file_obj, metadata_deserializer=None):
    data = formats.CIRCUIT_HEADER._make(
        struct.unpack(formats.CIRCUIT_HEADER_PACK, file_obj.read(formats.CIRCUIT_HEADER_SIZE))
    )
    name = file_obj.read(data.name_size).decode(common.ENCODE)
    header = {
        "global_phase": data.global_phase,
        "num_qubits": data.num_qubits,
        "num_clbits": data.num_clbits,
        "num_registers": data.num_registers,
        "num_instructions": data.num_instructions,
    }
    metadata_raw = file_obj.read(data.metadata_size)
    metadata = json.loads(metadata_raw, cls=metadata_deserializer)
    return header, name, metadata


def _read_registers_v4(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        data = formats.REGISTER_V4._make(
            struct.unpack(
                formats.REGISTER_V4_PACK,
                file_obj.read(formats.REGISTER_V4_SIZE),
            )
        )
        name = file_obj.read(data.name_size).decode("utf8")
        REGISTER_ARRAY_PACK = f"!{data.size}q"
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if data.type.decode("utf8") == "q":
            registers["q"][name] = (data.standalone, bit_indices, data.in_circuit)
        else:
            registers["c"][name] = (data.standalone, bit_indices, data.in_circuit)
    return registers


def _read_registers(file_obj, num_registers):
    registers = {"q": {}, "c": {}}
    for _reg in range(num_registers):
        data = formats.REGISTER._make(
            struct.unpack(
                formats.REGISTER_PACK,
                file_obj.read(formats.REGISTER_SIZE),
            )
        )
        name = file_obj.read(data.name_size).decode("utf8")
        REGISTER_ARRAY_PACK = f"!{data.size}I"
        bit_indices_raw = file_obj.read(struct.calcsize(REGISTER_ARRAY_PACK))
        bit_indices = list(struct.unpack(REGISTER_ARRAY_PACK, bit_indices_raw))
        if data.type.decode("utf8") == "q":
            registers["q"][name] = (data.standalone, bit_indices, True)
        else:
            registers["c"][name] = (data.standalone, bit_indices, True)
    return registers


def _read_annotation_states(file_obj, annotation_factories) -> _AnnotationDeserializationState:
    state = _AnnotationDeserializationState(annotation_factories)
    static_payload = formats.ANNOTATION_HEADER_STATIC._make(
        struct.unpack(
            formats.ANNOTATION_HEADER_STATIC_PACK,
            file_obj.read(formats.ANNOTATION_HEADER_STATIC_SIZE),
        )
    )
    for _ in range(static_payload.num_namespaces):
        payload = formats.ANNOTATION_STATE_HEADER._make(
            struct.unpack(
                formats.ANNOTATION_STATE_HEADER_PACK,
                file_obj.read(formats.ANNOTATION_STATE_HEADER_SIZE),
            )
        )
        state.initialize(
            file_obj.read(payload.namespace_size).decode("utf-8"), file_obj.read(payload.state_size)
        )
    return state


def _read_instruction_annotation(file_obj, annotation_state):
    header = formats.INSTRUCTION_ANNOTATION._make(
        struct.unpack(
            formats.INSTRUCTION_ANNOTATION_PACK,
            file_obj.read(formats.INSTRUCTION_ANNOTATION_SIZE),
        )
    )
    return annotation_state.load(header.namespace_index, file_obj.read(header.payload_size))


def _read_instruction_annotations(file_obj, annotation_state):
    header = formats.INSTRUCTION_ANNOTATIONS_HEADER._make(
        struct.unpack(
            formats.INSTRUCTION_ANNOTATIONS_HEADER_PACK,
            file_obj.read(formats.INSTRUCTION_ANNOTATIONS_HEADER_SIZE),
        )
    )
    return [
        _read_instruction_annotation(file_obj, annotation_state)
        for _ in range(header.num_annotations)
    ]


def _loads_instruction_parameter(
    type_key,
    data_bytes,
    version,
    vectors,
    registers,
    circuit,
    use_symengine,
    standalone_vars,
    annotation_factories,
):
    if type_key == type_keys.Program.CIRCUIT:
        param = common.data_from_binary(
            data_bytes,
            read_circuit,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=False,
        )
    elif type_key == type_keys.Value.MODIFIER:
        param = common.data_from_binary(data_bytes, _read_modifier)
    elif type_key == type_keys.Container.RANGE:
        data = formats.RANGE._make(struct.unpack(formats.RANGE_PACK, data_bytes))
        param = range(data.start, data.stop, data.step)
    elif type_key == type_keys.Container.TUPLE:
        param = tuple(
            common.sequence_from_binary(
                data_bytes,
                _loads_instruction_parameter,
                version=version,
                vectors=vectors,
                registers=registers,
                circuit=circuit,
                use_symengine=use_symengine,
                standalone_vars=standalone_vars,
                annotation_factories=annotation_factories,
            )
        )
    elif type_key == type_keys.Value.INTEGER:
        # TODO This uses little endian. Should be fixed in the next QPY version.
        param = struct.unpack("<q", data_bytes)[0]
    elif type_key == type_keys.Value.FLOAT:
        # TODO This uses little endian. Should be fixed in the next QPY version.
        param = struct.unpack("<d", data_bytes)[0]
    elif type_key == type_keys.Value.REGISTER:
        param = _loads_register_param(data_bytes.decode(common.ENCODE), circuit, registers)
    else:
        clbits = circuit.clbits if circuit is not None else ()
        param = value.loads_value(
            type_key,
            data_bytes,
            version,
            vectors,
            clbits=clbits,
            cregs=registers["c"],
            use_symengine=use_symengine,
            standalone_vars=standalone_vars,
        )

    return param


def _loads_register_param(data_bytes, circuit, registers):
    # If register name prefixed with null character it's a clbit index for single bit condition.
    if data_bytes[0] == "\x00":
        conditional_bit = int(data_bytes[1:])
        return circuit.clbits[conditional_bit]
    return registers["c"][data_bytes]


def _read_instruction(
    file_obj,
    circuit,
    registers,
    custom_operations,
    version,
    vectors,
    use_symengine,
    standalone_vars,
    annotation_state,
):
    if version < 5:
        instruction = formats.CIRCUIT_INSTRUCTION._make(
            struct.unpack(
                formats.CIRCUIT_INSTRUCTION_PACK,
                file_obj.read(formats.CIRCUIT_INSTRUCTION_SIZE),
            )
        )
        conditional_key = (
            type_keys.Condition.TWO_TUPLE if instruction.has_condition else type_keys.Condition.NONE
        )
        has_annotations = False
    else:
        instruction = formats.CIRCUIT_INSTRUCTION_V2._make(
            struct.unpack(
                formats.CIRCUIT_INSTRUCTION_V2_PACK,
                file_obj.read(formats.CIRCUIT_INSTRUCTION_V2_SIZE),
            )
        )
        conditional_key = type_keys.Condition(instruction.extras_key & 0b11)
        has_annotations = bool(
            instruction.extras_key & type_keys.InstructionExtraFlags.HAS_ANNOTATIONS
        )

    gate_name = file_obj.read(instruction.name_size).decode(common.ENCODE)
    label = file_obj.read(instruction.label_size).decode(common.ENCODE)
    condition_register = file_obj.read(instruction.condition_register_size).decode(common.ENCODE)
    qargs = []
    cargs = []
    params = []
    condition = None
    if conditional_key == type_keys.Condition.TWO_TUPLE:
        condition = (
            _loads_register_param(condition_register, circuit, registers),
            instruction.condition_value,
        )
    elif conditional_key == type_keys.Condition.EXPRESSION:
        condition = value.read_value(
            file_obj,
            version,
            vectors,
            clbits=circuit.clbits,
            cregs=registers["c"],
            use_symengine=use_symengine,
            standalone_vars=standalone_vars,
        )

    # Load Arguments
    if circuit is not None:
        for _qarg in range(instruction.num_qargs):
            qarg = formats.CIRCUIT_INSTRUCTION_ARG._make(
                struct.unpack(
                    formats.CIRCUIT_INSTRUCTION_ARG_PACK,
                    file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE),
                )
            )
            if qarg.type.decode(common.ENCODE) == "c":
                raise TypeError("Invalid input carg prior to all qargs")
            qargs.append(circuit.qubits[qarg.size])
        for _carg in range(instruction.num_cargs):
            carg = formats.CIRCUIT_INSTRUCTION_ARG._make(
                struct.unpack(
                    formats.CIRCUIT_INSTRUCTION_ARG_PACK,
                    file_obj.read(formats.CIRCUIT_INSTRUCTION_ARG_SIZE),
                )
            )
            if carg.type.decode(common.ENCODE) == "q":
                raise TypeError("Invalid input qarg after all qargs")
            cargs.append(circuit.clbits[carg.size])

    # Load Parameters
    for _param in range(instruction.num_parameters):
        type_key, data_bytes = common.read_generic_typed_data(file_obj)
        param = _loads_instruction_parameter(
            type_key,
            data_bytes,
            version,
            vectors,
            registers,
            circuit,
            use_symengine,
            standalone_vars,
            annotation_factories=annotation_state.factories,
        )
        params.append(param)

    # Load annotations.
    annotations = (
        _read_instruction_annotations(file_obj, annotation_state) if has_annotations else None
    )

    # Load Gate object
    if gate_name in {"Gate", "Instruction", "ControlledGate"}:
        inst_obj = _parse_custom_operation(
            custom_operations,
            gate_name,
            params,
            version,
            vectors,
            registers,
            use_symengine,
            standalone_vars,
            annotation_state=annotation_state,
        )
        if condition is not None:
            warnings.warn(
                f"The .condition attribute on {gate_name} can not be "
                "represented in this version of Qiskit. It will be "
                "represented as an IfElseOp instead.",
                UserWarning,
                stacklevel=3,
            )

            body = QuantumCircuit(qargs, cargs)
            body.append(inst_obj, qargs, cargs)
            inst_obj = IfElseOp(condition, body)
        if instruction.label_size > 0:
            inst_obj.label = label
        if circuit is None:
            return inst_obj
        circuit._append(inst_obj, qargs, cargs)
        return None
    elif gate_name in custom_operations:
        inst_obj = _parse_custom_operation(
            custom_operations,
            gate_name,
            params,
            version,
            vectors,
            registers,
            use_symengine,
            standalone_vars,
            annotation_state=annotation_state,
        )
        inst_obj.condition = condition
        if instruction.label_size > 0:
            inst_obj.label = label
        if circuit is None:
            return inst_obj
        circuit._append(inst_obj, qargs, cargs)
        return None
    elif hasattr(library, gate_name):
        gate_class = getattr(library, gate_name)
    elif hasattr(circuit_mod, gate_name):
        gate_class = getattr(circuit_mod, gate_name)
    elif hasattr(controlflow, gate_name):
        gate_class = getattr(controlflow, gate_name)
    elif gate_name == "Clifford":
        gate_class = Clifford
    elif gate_name == "pauli_product_measurement":
        gate_class = PauliProductMeasurement
    else:
        raise AttributeError(f"Invalid instruction type: {gate_name}")

    if instruction.label_size <= 0:
        label = None
    if gate_name in ("IfElseOp", "WhileLoopOp"):
        gate = gate_class(condition, *params, label=label)
    elif gate_name == "BoxOp":
        *params, duration, unit = params
        gate = gate_class(
            *params, label=label, duration=duration, unit=unit, annotations=annotations or ()
        )
    elif version >= 5 and issubclass(gate_class, ControlledGate):
        if gate_name in {
            "MCPhaseGate",
            "MCU1Gate",
            "MCXGrayCode",
            "MCXGate",
            "MCXRecursive",
            "MCXVChain",
        }:
            gate = gate_class(*params, instruction.num_ctrl_qubits, label=label)
        else:
            gate = gate_class(*params, label=label)
            if (
                gate.num_ctrl_qubits != instruction.num_ctrl_qubits
                or gate.ctrl_state != instruction.ctrl_state
            ):
                gate = gate.to_mutable()
                gate.num_ctrl_qubits = instruction.num_ctrl_qubits
                gate.ctrl_state = instruction.ctrl_state
        if condition:
            body = QuantumCircuit(qargs, cargs)
            body.append(gate, qargs, cargs)
            gate = IfElseOp(condition, body)
    else:
        if gate_name in {"Initialize", "StatePreparation"}:
            if isinstance(params[0], str):
                # the params are the labels of the initial state
                gate = gate_class("".join(label for label in params))
            elif instruction.num_parameters == 1:
                # the params is the integer indicating which qubits to initialize
                gate = gate_class(int(params[0].real), instruction.num_qargs)
            else:
                # the params represent a list of complex amplitudes
                gate = gate_class(params)
        elif gate_name in {
            "UCRXGate",
            "UCRYGate",
            "UCRZGate",
            "DiagonalGate",
        }:
            gate = gate_class(params)
        elif gate_name == "PauliProductMeasurement":
            gate = gate_class._from_pauli_data(*params, label)
        elif gate_name == "QFTGate":
            gate = gate_class(len(qargs), *params)
        else:
            if gate_name == "Barrier":
                params = [len(qargs)]
            elif gate_name in {"BreakLoopOp", "ContinueLoopOp"}:
                params = [len(qargs), len(cargs)]
            if label is not None:
                if issubclass(gate_class, (SingletonInstruction, SingletonGate)):
                    gate = gate_class(*params, label=label)
                else:
                    gate = gate_class(*params)
                    gate.label = label
            else:
                gate = gate_class(*params)
        if condition:
            if not isinstance(gate, ControlFlowOp):
                warnings.warn(
                    f"The .condition attribute on {gate_name} can not be "
                    "represented in this version of Qiskit. It will be "
                    "represented as an IfElseOp instead.",
                    UserWarning,
                    stacklevel=3,
                )
                body = QuantumCircuit(qargs, cargs)
                body.append(gate, qargs, cargs)
                gate = IfElseOp(condition, body)
            else:
                gate.condition = condition
    if circuit is None:
        return gate
    if not isinstance(gate, Instruction):
        circuit.append(gate, qargs, cargs)
    else:
        circuit._append(CircuitInstruction(gate, qargs, cargs))
    return None


def _parse_custom_operation(
    custom_operations,
    gate_name,
    params,
    version,
    vectors,
    registers,
    use_symengine,
    standalone_vars,
    annotation_state,
):
    if version >= 5:
        (
            type_str,
            num_qubits,
            num_clbits,
            definition,
            num_ctrl_qubits,
            ctrl_state,
            base_gate_raw,
        ) = custom_operations[gate_name]
    else:
        type_str, num_qubits, num_clbits, definition = custom_operations[gate_name]
        base_gate_raw = ctrl_state = num_ctrl_qubits = None
    # Strip the trailing "_{uuid}" from the gate name if the version >=11
    if version >= 11:
        gate_name = "_".join(gate_name.split("_")[:-1])
    type_key = type_keys.CircuitInstruction(type_str)

    if type_key == type_keys.CircuitInstruction.INSTRUCTION:
        inst_obj = Instruction(gate_name, num_qubits, num_clbits, params)
        if definition is not None:
            inst_obj.definition = definition
        return inst_obj

    if type_key == type_keys.CircuitInstruction.GATE:
        inst_obj = Gate(gate_name, num_qubits, params)
        inst_obj.definition = definition
        return inst_obj

    if version >= 5 and type_key == type_keys.CircuitInstruction.CONTROLLED_GATE:
        with io.BytesIO(base_gate_raw) as base_gate_obj:
            base_gate = _read_instruction(
                base_gate_obj,
                None,
                registers,
                custom_operations,
                version,
                vectors,
                use_symengine,
                standalone_vars,
                annotation_state=annotation_state,
            )
        if ctrl_state < 2**num_ctrl_qubits - 1:
            # If open controls, we need to discard the control suffix when setting the name.
            gate_name = gate_name.rsplit("_", 1)[0]
        inst_obj = ControlledGate(
            gate_name,
            num_qubits,
            params,
            num_ctrl_qubits=num_ctrl_qubits,
            ctrl_state=ctrl_state,
            base_gate=base_gate,
        )
        inst_obj.definition = definition
        return inst_obj

    if version >= 11 and type_key == type_keys.CircuitInstruction.ANNOTATED_OPERATION:
        with io.BytesIO(base_gate_raw) as base_gate_obj:
            base_gate = _read_instruction(
                base_gate_obj,
                None,
                registers,
                custom_operations,
                version,
                vectors,
                use_symengine,
                standalone_vars,
                annotation_state=annotation_state,
            )
        inst_obj = AnnotatedOperation(base_op=base_gate, modifiers=params)
        return inst_obj

    if type_key == type_keys.CircuitInstruction.PAULI_EVOL_GATE:
        return definition

    raise ValueError(f"Invalid custom instruction type '{type_str}'")


def _read_pauli_evolution_gate(file_obj, version, vectors):
    pauli_evolution_def = formats.PAULI_EVOLUTION_DEF._make(
        struct.unpack(
            formats.PAULI_EVOLUTION_DEF_PACK, file_obj.read(formats.PAULI_EVOLUTION_DEF_SIZE)
        )
    )

    if pauli_evolution_def.operator_size != 1 and pauli_evolution_def.standalone_op:
        raise ValueError(
            "Can't have a standalone operator with {pauli_evolution_raw[0]} operators in the payload"
        )

    operator_list = []
    for _ in range(pauli_evolution_def.operator_size):
        sparse_operator = False
        if version >= 17:
            sparse_operator = struct.unpack("!?", file_obj.read(1))[0]
            if sparse_operator:
                op_elem = formats.SPARSE_OBSERVABLE._make(
                    struct.unpack(
                        formats.SPARSE_OBSERVABLE_PACK,
                        file_obj.read(formats.SPARSE_OBSERVABLE_SIZE),
                    )
                )
                # setting data lengths
                num_paulis = int(op_elem.coeff_data_len / (2 * struct.calcsize("!d")))
                num_bitterms = int(op_elem.bitterm_data_len / struct.calcsize("!H"))
                num_inds = int(op_elem.inds_data_len / struct.calcsize("!I"))
                num_bounds = int(op_elem.bounds_data_len / struct.calcsize("!Q"))

                # reading coeffs
                coeff_data = struct.unpack(
                    f"!{2*num_paulis}d", file_obj.read(op_elem.coeff_data_len)
                )
                coeff_read = np.empty(num_paulis, dtype=np.complex128)
                for ii in range(num_paulis):
                    coeff_read[ii] = complex(coeff_data[2 * ii], coeff_data[2 * ii + 1])

                # reading bit_terms
                bitterms_read = list(
                    struct.unpack(f"!{num_bitterms}H", file_obj.read(op_elem.bitterm_data_len))
                )

                # reading indices
                inds_read = list(
                    struct.unpack(f"!{num_inds}I", file_obj.read(op_elem.inds_data_len))
                )

                # reading boundaries
                bounds_read = list(
                    struct.unpack(f"!{num_bounds}Q", file_obj.read(op_elem.bounds_data_len))
                )

                operator_list.append(
                    SparseObservable.from_raw_parts(
                        op_elem.num_qubits, coeff_read, bitterms_read, inds_read, bounds_read
                    )
                )

        if version < 17 or not sparse_operator:
            # Read SparsePauliOp type operator
            op_elem = formats.SPARSE_PAULI_OP_LIST_ELEM._make(
                struct.unpack(
                    formats.SPARSE_PAULI_OP_LIST_ELEM_PACK,
                    file_obj.read(formats.SPARSE_PAULI_OP_LIST_ELEM_SIZE),
                )
            )
            op_raw_data = common.data_from_binary(file_obj.read(op_elem.size), np.load)
            operator_list.append(SparsePauliOp.from_list(op_raw_data))

    if pauli_evolution_def.standalone_op:
        pauli_op = operator_list[0]
    else:
        pauli_op = operator_list

    time = value.loads_value(
        pauli_evolution_def.time_type,
        file_obj.read(pauli_evolution_def.time_size),
        version=version,
        vectors=vectors,
    )
    synth_data = json.loads(file_obj.read(pauli_evolution_def.synth_method_size))
    synthesis = getattr(evo_synth, synth_data["class"])(**synth_data["settings"])
    return_gate = library.PauliEvolutionGate(pauli_op, time=time, synthesis=synthesis)
    return return_gate


def _read_modifier(file_obj):
    modifier = formats.MODIFIER_DEF._make(
        struct.unpack(
            formats.MODIFIER_DEF_PACK,
            file_obj.read(formats.MODIFIER_DEF_SIZE),
        )
    )
    if modifier.type == b"i":
        return InverseModifier()
    elif modifier.type == b"c":
        return ControlModifier(
            num_ctrl_qubits=modifier.num_ctrl_qubits, ctrl_state=modifier.ctrl_state
        )
    elif modifier.type == b"p":
        return PowerModifier(power=modifier.power)
    else:
        raise TypeError("Unsupported modifier.")


def _read_custom_operations(file_obj, version, vectors, annotation_state):
    custom_operations = {}
    custom_definition_header = formats.CUSTOM_CIRCUIT_DEF_HEADER._make(
        struct.unpack(
            formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK,
            file_obj.read(formats.CUSTOM_CIRCUIT_DEF_HEADER_SIZE),
        )
    )
    if custom_definition_header.size > 0:
        for _ in range(custom_definition_header.size):
            if version < 5:
                data = formats.CUSTOM_CIRCUIT_INST_DEF._make(
                    struct.unpack(
                        formats.CUSTOM_CIRCUIT_INST_DEF_PACK,
                        file_obj.read(formats.CUSTOM_CIRCUIT_INST_DEF_SIZE),
                    )
                )
            else:
                data = formats.CUSTOM_CIRCUIT_INST_DEF_V2._make(
                    struct.unpack(
                        formats.CUSTOM_CIRCUIT_INST_DEF_V2_PACK,
                        file_obj.read(formats.CUSTOM_CIRCUIT_INST_DEF_V2_SIZE),
                    )
                )

            name = file_obj.read(data.gate_name_size).decode(common.ENCODE)
            type_str = data.type
            definition_circuit = None
            if data.custom_definition:
                def_binary = file_obj.read(data.size)
                if version < 3 or not name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = common.data_from_binary(
                        def_binary,
                        read_circuit,
                        version=version,
                        annotation_factories=annotation_state.factories,
                        use_rust=False,
                    )
                elif name.startswith(r"###PauliEvolutionGate_"):
                    definition_circuit = common.data_from_binary(
                        def_binary, _read_pauli_evolution_gate, version=version, vectors=vectors
                    )
            if version < 5:
                data_payload = (type_str, data.num_qubits, data.num_clbits, definition_circuit)
            else:
                base_gate = file_obj.read(data.base_gate_size)
                data_payload = (
                    type_str,
                    data.num_qubits,
                    data.num_clbits,
                    definition_circuit,
                    data.num_ctrl_qubits,
                    data.ctrl_state,
                    base_gate,
                )
            custom_operations[name] = data_payload
    return custom_operations


def _read_calibrations(file_obj, version, vectors, metadata_deserializer):
    """Consume calibrations data, make the file handle point to the next section"""
    header = formats.CALIBRATION._make(
        struct.unpack(formats.CALIBRATION_PACK, file_obj.read(formats.CALIBRATION_SIZE))
    )
    for _ in range(header.num_cals):
        defheader = formats.CALIBRATION_DEF._make(
            struct.unpack(formats.CALIBRATION_DEF_PACK, file_obj.read(formats.CALIBRATION_DEF_SIZE))
        )
        name = file_obj.read(defheader.name_size).decode(common.ENCODE)
        if name:
            warnings.warn(
                category=UserWarning,
                message="Support for loading pulse gates has been removed in Qiskit 2.0. "
                f"If `{name}` is in the circuit it will be left as an opaque instruction.",
            )

        for _ in range(defheader.num_qubits):  # read qubits info
            file_obj.read(struct.calcsize("!q"))

        for _ in range(defheader.num_params):  # read params info
            value.read_value(file_obj, version, vectors)

        schedules.read_schedule_block(file_obj, version, metadata_deserializer)


def _py_serialize_register_param(register, index_map):
    if isinstance(register, ClassicalRegister):
        return register.name.encode(common.ENCODE)
    # Clbit.
    return b"\x00" + str(index_map["c"][register]).encode(common.ENCODE)


def _dumps_instruction_parameter(
    param, index_map, use_symengine, *, version, standalone_var_indices, annotation_factories
):
    if isinstance(param, QuantumCircuit):
        type_key = type_keys.Program.CIRCUIT
        data_bytes = common.data_to_binary(
            param,
            write_circuit,
            version=version,
            annotation_factories=annotation_factories,
            use_rust=False,
        )
    elif isinstance(param, Modifier):
        type_key = type_keys.Value.MODIFIER
        data_bytes = common.data_to_binary(param, _write_modifier)
    elif isinstance(param, range):
        type_key = type_keys.Container.RANGE
        data_bytes = struct.pack(formats.RANGE_PACK, param.start, param.stop, param.step)
    elif isinstance(param, tuple):
        type_key = type_keys.Container.TUPLE
        data_bytes = common.sequence_to_binary(
            param,
            _dumps_instruction_parameter,
            index_map=index_map,
            use_symengine=use_symengine,
            version=version,
            standalone_var_indices=standalone_var_indices,
            annotation_factories=annotation_factories,
        )
    elif isinstance(param, int):
        # TODO This uses little endian. This should be fixed in next QPY version.
        type_key = type_keys.Value.INTEGER
        data_bytes = struct.pack("<q", param)
    elif isinstance(param, float):
        # TODO This uses little endian. This should be fixed in next QPY version.
        type_key = type_keys.Value.FLOAT
        data_bytes = struct.pack("<d", param)
    elif isinstance(param, (Clbit, ClassicalRegister)):
        type_key = type_keys.Value.REGISTER
        data_bytes = _py_serialize_register_param(param, index_map)
    else:
        type_key, data_bytes = value.dumps_value(
            param,
            index_map=index_map,
            use_symengine=use_symengine,
            standalone_var_indices=standalone_var_indices,
            version=version,
        )

    return type_key, data_bytes


def _write_instruction(
    file_obj,
    instruction,
    custom_operations,
    index_map,
    use_symengine,
    version,
    annotation_state,
    standalone_var_indices=None,
):
    if isinstance(instruction.operation, Instruction):
        gate_class_name = instruction.operation.base_class.__name__
    else:
        gate_class_name = instruction.operation.__class__.__name__

    custom_operations_list = []
    if (
        (
            not hasattr(library, gate_class_name)
            and not hasattr(circuit_mod, gate_class_name)
            and not hasattr(controlflow, gate_class_name)
            and gate_class_name not in ["Clifford", "PauliProductMeasurement"]
        )
        or gate_class_name in {"Gate", "Instruction"}
        or isinstance(instruction.operation, library.BlueprintCircuit)
    ):
        gate_class_name = instruction.operation.name
        # Assign a uuid to each instance of a custom operation
        if instruction.operation.name not in {"ucrx_dg", "ucry_dg", "ucrz_dg"}:
            gate_class_name = f"{gate_class_name}_{uuid.uuid4().hex}"
        else:
            # ucr*_dg gates can have different numbers of parameters,
            # the uuid is appended to avoid storing a single definition
            # in circuits with multiple ucr*_dg gates. For legacy reasons
            # the uuid is stored in a different format as this was done
            # prior to QPY 11.
            gate_class_name = f"{gate_class_name}_{uuid.uuid4()}"

        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)

    elif gate_class_name in {"ControlledGate", "AnnotatedOperation"}:
        # controlled or annotated gates can have the same name but different parameter
        # values, the uuid is appended to avoid storing a single definition
        # in circuits with multiple controlled gates.
        gate_class_name = instruction.operation.name + "_" + str(uuid.uuid4())
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)

    elif isinstance(instruction.operation, library.PauliEvolutionGate):
        gate_class_name = r"###PauliEvolutionGate_" + str(uuid.uuid4())
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)

    elif isinstance(instruction.operation, library.MCMTGate):
        gate_class_name = instruction.operation.name + "_" + str(uuid.uuid4())
        custom_operations[gate_class_name] = instruction.operation
        custom_operations_list.append(gate_class_name)

    extra_type = type_keys.Condition.NONE
    condition_register = b""
    condition_value = 0
    if (op_condition := getattr(instruction.operation, "_condition", None)) is not None:
        if isinstance(op_condition, expr.Expr):
            extra_type = type_keys.Condition.EXPRESSION
        else:
            extra_type = type_keys.Condition.TWO_TUPLE
            condition_register = _py_serialize_register_param(
                instruction.operation._condition[0], index_map
            )
            condition_value = int(instruction.operation._condition[1])

    gate_class_name = gate_class_name.encode(common.ENCODE)
    label = getattr(instruction.operation, "label", None)
    if label:
        label_raw = label.encode(common.ENCODE)
    else:
        label_raw = b""

    annotations = []
    # The instruction params we store are about being able to reconstruct the objects; they don't
    # necessarily need to match one-to-one to the `params` field.
    if isinstance(instruction.operation, controlflow.SwitchCaseOp):
        instruction_params = [
            instruction.operation.target,
            tuple(instruction.operation.cases_specifier()),
        ]
    elif isinstance(instruction.operation, controlflow.BoxOp):
        instruction_params = [
            instruction.operation.blocks[0],
            instruction.operation.duration,
            instruction.operation.unit,
        ]
        annotations = [
            annotation_state.serialize(annotation)
            for annotation in instruction.operation.annotations
        ]
    elif isinstance(instruction.operation, Clifford):
        instruction_params = [instruction.operation.tableau]
    elif isinstance(instruction.operation, AnnotatedOperation):
        instruction_params = instruction.operation.modifiers
    elif isinstance(instruction.operation, PauliProductMeasurement):
        instruction_params = instruction.operation._to_pauli_data()
    else:
        instruction_params = getattr(instruction.operation, "params", [])

    if annotations:
        extra_type |= type_keys.InstructionExtraFlags.HAS_ANNOTATIONS

    num_ctrl_qubits = getattr(instruction.operation, "num_ctrl_qubits", 0)
    ctrl_state = getattr(instruction.operation, "ctrl_state", 0)
    instruction_raw = struct.pack(
        formats.CIRCUIT_INSTRUCTION_V2_PACK,
        len(gate_class_name),
        len(label_raw),
        len(instruction_params),
        instruction.operation.num_qubits,
        instruction.operation.num_clbits,
        int(extra_type),
        len(condition_register),
        condition_value,
        num_ctrl_qubits,
        ctrl_state,
    )
    file_obj.write(instruction_raw)
    file_obj.write(gate_class_name)
    file_obj.write(label_raw)
    condition_type = type_keys.Condition(extra_type & 0b11)
    if condition_type is type_keys.Condition.EXPRESSION:
        value.write_value(
            file_obj,
            op_condition,
            version=version,
            index_map=index_map,
            standalone_var_indices=standalone_var_indices,
        )
    else:
        file_obj.write(condition_register)

    # Encode instruction args
    for qbit in instruction.qubits:
        instruction_arg_raw = struct.pack(
            formats.CIRCUIT_INSTRUCTION_ARG_PACK, b"q", index_map["q"][qbit]
        )
        file_obj.write(instruction_arg_raw)
    for clbit in instruction.clbits:
        instruction_arg_raw = struct.pack(
            formats.CIRCUIT_INSTRUCTION_ARG_PACK, b"c", index_map["c"][clbit]
        )
        file_obj.write(instruction_arg_raw)
    # Encode instruction params
    for param in instruction_params:
        type_key, data_bytes = _dumps_instruction_parameter(
            param,
            index_map,
            use_symengine,
            version=version,
            standalone_var_indices=standalone_var_indices,
            annotation_factories=annotation_state.factories,
        )
        common.write_generic_typed_data(file_obj, type_key, data_bytes)
    if annotations:
        if version < 15:
            raise UnsupportedFeatureForVersion("annotations", 15, version)
        file_obj.write(struct.pack(formats.INSTRUCTION_ANNOTATIONS_HEADER_PACK, len(annotations)))
        for serializer_index, annotation_payload in annotations:
            file_obj.write(
                struct.pack(
                    formats.INSTRUCTION_ANNOTATION_PACK, serializer_index, len(annotation_payload)
                )
            )
            file_obj.write(annotation_payload)
    return custom_operations_list


def _write_pauli_evolution_gate(file_obj, evolution_gate, version):
    operator_list = evolution_gate.operator
    standalone = False

    if not isinstance(operator_list, list):
        operator_list = [operator_list]
        standalone = True
    num_operators = len(operator_list)

    def _write_elem(buffer, op):
        elem_data = common.data_to_binary(op.to_list(array=True), np.save)
        elem_metadata = struct.pack(formats.SPARSE_PAULI_OP_LIST_ELEM_PACK, len(elem_data))
        if version >= 17:
            elem_sparse_operator = struct.pack("!?", False)
            buffer.write(elem_sparse_operator)
        buffer.write(elem_metadata)
        buffer.write(elem_data)

    def _write_elem_sparse(buffer, op):
        bitterms = op.bit_terms
        coeffs = op.coeffs
        bounds = op.boundaries
        inds = op.indices
        num_qubits = op.num_qubits

        # pack elements as [c1.real, c1.imag, c2.real, c2.imag, ...]
        coeff_data = struct.pack(
            f"!{len(coeffs)*2}d", *(val for coeff in coeffs for val in (coeff.real, coeff.imag))
        )
        bitterm_data = struct.pack(f"!{len(bitterms)}H", *bitterms)
        inds_data = struct.pack(f"!{len(inds)}I", *inds)
        bounds_data = struct.pack(f"!{len(bounds)}Q", *bounds)

        sparse_observable_data_length = struct.pack(
            formats.SPARSE_OBSERVABLE_PACK,
            num_qubits,
            len(coeff_data),
            len(bitterm_data),
            len(inds_data),
            len(bounds_data),
        )

        elem_sparse_operator = struct.pack("!?", True)
        buffer.write(elem_sparse_operator)
        buffer.write(sparse_observable_data_length)
        buffer.write(coeff_data)
        buffer.write(bitterm_data)
        buffer.write(inds_data)
        buffer.write(bounds_data)

    pauli_data_buf = io.BytesIO()

    for operator in operator_list:
        if isinstance(operator, SparseObservable):
            data = common.data_to_binary(operator, _write_elem_sparse)
        else:
            data = common.data_to_binary(operator, _write_elem)
        pauli_data_buf.write(data)

    time_type, time_data = value.dumps_value(evolution_gate.time, version=version)
    time_size = len(time_data)
    synth_class = str(type(evolution_gate.synthesis).__name__)
    settings_dict = evolution_gate.synthesis.settings
    synth_data = json.dumps({"class": synth_class, "settings": settings_dict}).encode(common.ENCODE)
    synth_size = len(synth_data)
    pauli_evolution_raw = struct.pack(
        formats.PAULI_EVOLUTION_DEF_PACK,
        num_operators,
        standalone,
        time_type,
        time_size,
        synth_size,
    )
    file_obj.write(pauli_evolution_raw)
    file_obj.write(pauli_data_buf.getvalue())
    pauli_data_buf.close()
    file_obj.write(time_data)
    file_obj.write(synth_data)


def _write_modifier(file_obj, modifier):
    if isinstance(modifier, InverseModifier):
        type_key = b"i"
        num_ctrl_qubits = 0
        ctrl_state = 0
        power = 0.0
    elif isinstance(modifier, ControlModifier):
        type_key = b"c"
        num_ctrl_qubits = modifier.num_ctrl_qubits
        ctrl_state = modifier.ctrl_state
        power = 0.0
    elif isinstance(modifier, PowerModifier):
        type_key = b"p"
        num_ctrl_qubits = 0
        ctrl_state = 0
        power = modifier.power
    else:
        raise TypeError("Unsupported modifier.")

    modifier_data = struct.pack(
        formats.MODIFIER_DEF_PACK, type_key, num_ctrl_qubits, ctrl_state, power
    )
    file_obj.write(modifier_data)


def _write_custom_operation(
    file_obj,
    name,
    operation,
    custom_operations,
    use_symengine,
    version,
    *,
    standalone_var_indices,
    annotation_state,
):
    type_key = type_keys.CircuitInstruction.assign(operation)
    has_definition = False
    size = 0
    data = None
    num_qubits = operation.num_qubits
    num_clbits = operation.num_clbits
    ctrl_state = 0
    num_ctrl_qubits = 0
    base_gate = None
    new_custom_instruction = []

    if type_key == type_keys.CircuitInstruction.PAULI_EVOL_GATE:
        has_definition = True
        data = common.data_to_binary(operation, _write_pauli_evolution_gate, version=version)
        size = len(data)
    elif type_key == type_keys.CircuitInstruction.CONTROLLED_GATE:
        # For ControlledGate, we have to access and store the private `_definition` rather than the
        # public one, because the public one is mutated to include additional logic if the control
        # state is open, and the definition setter (during a subsequent read) uses the "fully
        # excited" control definition only.
        has_definition = True
        # Build internal definition to support overloaded subclasses by
        # calling definition getter on object
        operation.definition
        data = common.data_to_binary(
            operation._definition,
            write_circuit,
            version=version,
            annotation_factories=annotation_state.factories,
            use_rust=False,
        )
        size = len(data)
        num_ctrl_qubits = operation.num_ctrl_qubits
        ctrl_state = operation.ctrl_state
        base_gate = operation.base_gate
    elif type_key == type_keys.CircuitInstruction.ANNOTATED_OPERATION:
        has_definition = False
        base_gate = operation.base_op
    elif operation.definition is not None:
        has_definition = True
        data = common.data_to_binary(
            operation.definition,
            write_circuit,
            version=version,
            annotation_factories=annotation_state.factories,
            use_rust=False,
        )
        size = len(data)
    if base_gate is None:
        base_gate_raw = b""
    else:
        with io.BytesIO() as base_gate_buffer:
            new_custom_instruction = _write_instruction(
                base_gate_buffer,
                CircuitInstruction(base_gate, (), ()),
                custom_operations,
                {},
                use_symengine,
                version,
                standalone_var_indices=standalone_var_indices,
                annotation_state=annotation_state,
            )
            base_gate_raw = base_gate_buffer.getvalue()
    name_raw = name.encode(common.ENCODE)
    custom_operation_raw = struct.pack(
        formats.CUSTOM_CIRCUIT_INST_DEF_V2_PACK,
        len(name_raw),
        type_key,
        num_qubits,
        num_clbits,
        has_definition,
        size,
        num_ctrl_qubits,
        ctrl_state,
        len(base_gate_raw),
    )
    file_obj.write(custom_operation_raw)
    file_obj.write(name_raw)
    if data:
        file_obj.write(data)
    file_obj.write(base_gate_raw)
    return new_custom_instruction


def _write_registers(file_obj, in_circ_regs, full_bits):
    bitmap = {bit: index for index, bit in enumerate(full_bits)}

    out_circ_regs = set()
    for bit in full_bits:
        if bit._register is not None and bit._register not in in_circ_regs:
            out_circ_regs.add(bit._register)

    for regs, is_in_circuit in [(in_circ_regs, True), (out_circ_regs, False)]:
        for reg in regs:
            standalone = all(
                bit._register == reg and bit._index == index for index, bit in enumerate(reg)
            )
            reg_name = reg.name.encode(common.ENCODE)
            reg_type = reg.prefix.encode(common.ENCODE)
            file_obj.write(
                struct.pack(
                    formats.REGISTER_V4_PACK,
                    reg_type,
                    standalone,
                    reg.size,
                    len(reg_name),
                    is_in_circuit,
                )
            )
            file_obj.write(reg_name)
            REGISTER_ARRAY_PACK = f"!{reg.size}q"
            bit_indices = []
            for bit in reg:
                bit_indices.append(bitmap.get(bit, -1))
            file_obj.write(struct.pack(REGISTER_ARRAY_PACK, *bit_indices))

    return len(in_circ_regs) + len(out_circ_regs)


def _write_layout(file_obj, circuit):
    if circuit.layout is None:
        # Write a null header if there is no layout present
        file_obj.write(struct.pack(formats.LAYOUT_V2_PACK, False, -1, -1, -1, 0, 0))
        return
    initial_size = -1
    input_qubit_mapping = {}
    initial_layout_array = []
    extra_registers = defaultdict(list)
    if circuit.layout.initial_layout is not None:
        initial_size = len(circuit.layout.initial_layout)
        layout_mapping = circuit.layout.initial_layout.get_physical_bits()
        for i in range(circuit.num_qubits):
            qubit = layout_mapping[i]
            input_qubit_mapping[qubit] = i
            if qubit._register is not None or qubit._index is not None:
                if qubit._register not in circuit.qregs:
                    extra_registers[qubit._register].append(qubit)
                initial_layout_array.append((qubit._index, qubit._register))
            else:
                initial_layout_array.append((None, None))
    input_qubit_size = -1
    input_qubit_mapping_array = []
    if circuit.layout.input_qubit_mapping is not None:
        input_qubit_size = len(circuit.layout.input_qubit_mapping)
        input_qubit_mapping_array = [None] * input_qubit_size
        layout_mapping = circuit.layout.initial_layout.get_virtual_bits()
        for qubit, index in circuit.layout.input_qubit_mapping.items():
            if (
                getattr(qubit, "_register", None) is not None
                and getattr(qubit, "_index", None) is not None
            ):
                if qubit._register not in circuit.qregs:
                    extra_registers[qubit._register].append(qubit)
                input_qubit_mapping_array[index] = layout_mapping[qubit]
            else:
                input_qubit_mapping_array[index] = layout_mapping[qubit]
    final_layout_size = -1
    final_layout_array = []
    if circuit.layout.final_layout is not None:
        final_layout_size = len(circuit.layout.final_layout)
        final_layout_physical = circuit.layout.final_layout.get_physical_bits()
        for i in range(circuit.num_qubits):
            virtual_bit = final_layout_physical[i]
            final_layout_array.append(circuit.find_bit(virtual_bit).index)

    input_qubit_count = circuit._layout._input_qubit_count
    if input_qubit_count is None:
        input_qubit_count = -1
    file_obj.write(
        struct.pack(
            formats.LAYOUT_V2_PACK,
            True,
            initial_size,
            input_qubit_size,
            final_layout_size,
            len(extra_registers),
            input_qubit_count,
        )
    )
    _write_registers(
        file_obj, list(extra_registers), [x for bits in extra_registers.values() for x in bits]
    )
    for index, register in initial_layout_array:
        reg_name_bytes = None if register is None else register.name.encode(common.ENCODE)
        file_obj.write(
            struct.pack(
                formats.INITIAL_LAYOUT_BIT_PACK,
                -1 if index is None else index,
                -1 if reg_name_bytes is None else len(reg_name_bytes),
            )
        )
        if reg_name_bytes is not None:
            file_obj.write(reg_name_bytes)
    for i in input_qubit_mapping_array:
        file_obj.write(struct.pack("!I", i))
    for i in final_layout_array:
        file_obj.write(struct.pack("!I", i))


def _read_layout(file_obj, circuit):
    header = formats.LAYOUT._make(
        struct.unpack(formats.LAYOUT_PACK, file_obj.read(formats.LAYOUT_SIZE))
    )
    if not header.exists:
        return
    _read_common_layout(file_obj, header, circuit)


def _read_common_layout(file_obj, header, circuit):
    registers = {
        name: QuantumRegister(len(v[1]), name)
        for name, v in _read_registers_v4(file_obj, header.extra_registers)["q"].items()
    }
    initial_layout = None
    initial_layout_virtual_bits = []
    for _ in range(header.initial_layout_size):
        virtual_bit = formats.INITIAL_LAYOUT_BIT._make(
            struct.unpack(
                formats.INITIAL_LAYOUT_BIT_PACK,
                file_obj.read(formats.INITIAL_LAYOUT_BIT_SIZE),
            )
        )
        if virtual_bit.index == -1 and virtual_bit.register_size == -1:
            qubit = Qubit()
        else:
            register_name = file_obj.read(virtual_bit.register_size).decode(common.ENCODE)
            if register_name in registers:
                qubit = registers[register_name][virtual_bit.index]
            else:
                register = next(filter(lambda x, name=register_name: x.name == name, circuit.qregs))
                qubit = register[virtual_bit.index]
        initial_layout_virtual_bits.append(qubit)
    if initial_layout_virtual_bits:
        initial_layout = Layout.from_qubit_list(initial_layout_virtual_bits)
    input_qubit_mapping = None
    input_qubit_mapping_array = []
    for _ in range(header.input_mapping_size):
        input_qubit_mapping_array.append(
            struct.unpack("!I", file_obj.read(struct.calcsize("!I")))[0]
        )
    if input_qubit_mapping_array:
        input_qubit_mapping = {}
        physical_bits = initial_layout.get_physical_bits()
        for index, bit in enumerate(input_qubit_mapping_array):
            input_qubit_mapping[physical_bits[bit]] = index
    final_layout = None
    final_layout_array = []
    for _ in range(header.final_layout_size):
        final_layout_array.append(struct.unpack("!I", file_obj.read(struct.calcsize("!I")))[0])

    if final_layout_array:
        layout_dict = {circuit.qubits[bit]: index for index, bit in enumerate(final_layout_array)}
        final_layout = Layout(layout_dict)

    circuit._layout = TranspileLayout(initial_layout, input_qubit_mapping, final_layout)


def _read_layout_v2(file_obj, circuit):
    header = formats.LAYOUT_V2._make(
        struct.unpack(formats.LAYOUT_V2_PACK, file_obj.read(formats.LAYOUT_V2_SIZE))
    )
    if not header.exists:
        return
    _read_common_layout(file_obj, header, circuit)
    if header.input_qubit_count >= 0:
        circuit._layout._input_qubit_count = header.input_qubit_count
        circuit._layout._output_qubit_list = circuit.qubits


def write_circuit(
    file_obj,
    circuit,
    metadata_serializer=None,
    use_symengine=False,
    version=common.QPY_VERSION,
    annotation_factories=None,
    use_rust=True,
):
    """Write a single QuantumCircuit object in the file like object.

    Args:
        file_obj (FILE): The file like object to write the circuit data in.
        circuit (QuantumCircuit): The circuit data to write.
        metadata_serializer (JSONEncoder): An optional JSONEncoder class that
            will be passed the :attr:`.QuantumCircuit.metadata` dictionary for
            ``circuit`` and will be used as the ``cls`` kwarg
            on the ``json.dump()`` call to JSON serialize that dictionary.
        use_symengine (bool): If True, symbolic objects will be serialized using symengine's
            native mechanism. This is a faster serialization alternative, but not supported in all
            platforms. Please check that your target platform is supported by the symengine library
            before setting this option, as it will be required by qpy to deserialize the payload.
        version (int): The QPY format version to use for serializing this circuit
        annotation_factories (dict): a mapping of namespaces to zero-argument factory functions that
            produce instances of :class:`.annotation.QPYSerializer`.
        use_rust (bool): whether to use the rust based serialization engine. On by default.
    """
    if use_rust:
        if annotation_factories is None:
            annotation_factories = {}
        _qpy.write_circuit(
            file_obj,
            circuit,
            metadata_serializer,
            use_symengine,
            version,
            annotation_factories=annotation_factories,
        )
        return
    annotation_state = _AnnotationSerializationState(annotation_factories or {})
    metadata_raw = json.dumps(
        circuit.metadata, separators=(",", ":"), cls=metadata_serializer
    ).encode(common.ENCODE)
    metadata_size = len(metadata_raw)
    num_instructions = len(circuit)
    circuit_name = circuit.name.encode(common.ENCODE)
    global_phase_type, global_phase_data = value.dumps_value(circuit.global_phase, version=version)

    with io.BytesIO() as reg_buf:
        num_qregs = _write_registers(reg_buf, circuit.qregs, circuit.qubits)
        num_cregs = _write_registers(reg_buf, circuit.cregs, circuit.clbits)
        registers_raw = reg_buf.getvalue()
    num_registers = num_qregs + num_cregs

    # Write circuit header
    header_raw = formats.CIRCUIT_HEADER_V12(
        name_size=len(circuit_name),
        global_phase_type=global_phase_type,
        global_phase_size=len(global_phase_data),
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        metadata_size=metadata_size,
        num_registers=num_registers,
        num_instructions=num_instructions,
        num_vars=circuit.num_identifiers,
    )
    header = struct.pack(formats.CIRCUIT_HEADER_V12_PACK, *header_raw)
    file_obj.write(header)
    file_obj.write(circuit_name)
    file_obj.write(global_phase_data)
    file_obj.write(metadata_raw)
    # Write header payload
    file_obj.write(registers_raw)
    standalone_var_indices = value.write_standalone_vars(file_obj, circuit, version)

    instruction_buffer = io.BytesIO()
    custom_operations = {}
    index_map = {}
    index_map["q"] = {bit: index for index, bit in enumerate(circuit.qubits)}
    index_map["c"] = {bit: index for index, bit in enumerate(circuit.clbits)}
    for instruction in circuit.data:
        _write_instruction(
            instruction_buffer,
            instruction,
            custom_operations,
            index_map,
            use_symengine,
            version,
            standalone_var_indices=standalone_var_indices,
            annotation_state=annotation_state,
        )

    with io.BytesIO() as custom_operations_buffer:
        new_custom_operations = list(custom_operations.keys())
        while new_custom_operations:
            operations_to_serialize = new_custom_operations.copy()
            new_custom_operations = []
            for name in operations_to_serialize:
                operation = custom_operations[name]
                new_custom_operations.extend(
                    _write_custom_operation(
                        custom_operations_buffer,
                        name,
                        operation,
                        custom_operations,
                        use_symengine,
                        version,
                        standalone_var_indices=standalone_var_indices,
                        annotation_state=annotation_state,
                    )
                )
        # We only write this out after we've done the annotations.
        custom_operations_payload = custom_operations_buffer.getvalue()

    if version >= 15:
        file_obj.write(
            struct.pack(formats.ANNOTATION_HEADER_STATIC_PACK, annotation_state.num_serializers)
        )
        for namespace, serializer in annotation_state.iter_serializers():
            namespace_bytes = namespace.encode("utf-8")
            serializer_state = serializer.dump_state()
            file_obj.write(
                struct.pack(
                    formats.ANNOTATION_STATE_HEADER_PACK,
                    len(namespace_bytes),
                    len(serializer_state),
                )
            )
            file_obj.write(namespace_bytes)
            file_obj.write(serializer_state)
    elif annotation_state.num_serializers:
        raise UnsupportedFeatureForVersion(annotations, 15, version)

    file_obj.write(struct.pack(formats.CUSTOM_CIRCUIT_DEF_HEADER_PACK, len(custom_operations)))
    file_obj.write(custom_operations_payload)
    file_obj.write(instruction_buffer.getvalue())
    instruction_buffer.close()

    # Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    # we need to write an empty calibrations header since read_circuit expects it
    header = struct.pack(formats.CALIBRATION_PACK, 0)
    file_obj.write(header)

    _write_layout(file_obj, circuit)


def read_circuit(
    file_obj,
    version,
    metadata_deserializer=None,
    use_symengine=False,
    annotation_factories=None,
    use_rust=True,
):
    """Read a single QuantumCircuit object from the file like object.

    Args:
        file_obj (FILE): The file like object to read the circuit data from.
        version (int): QPY version.
        metadata_deserializer (JSONDecoder): An optional JSONDecoder class
            that will be used for the ``cls`` kwarg on the internal
            ``json.load`` call used to deserialize the JSON payload used for
            the :attr:`.QuantumCircuit.metadata` attribute for a circuit
            in the file-like object. If this is not specified the circuit metadata will
            be parsed as JSON with the stdlib ``json.load()`` function using
            the default ``JSONDecoder`` class.
        use_symengine (bool): If True, symbolic objects will be de-serialized using
            symengine's native mechanism. This is a faster serialization alternative, but not
            supported in all platforms. Please check that your target platform is supported by
            the symengine library before setting this option, as it will be required by qpy to
            deserialize the payload.
        annotation_factories (dict): mapping of namespaces to factory functions for custom
            annotation deserializer objects.
        use_rust (bool): whether to use the rust based deserialization engine. On by default.
    Returns:
        QuantumCircuit: The circuit object from the file.

    Raises:
        QpyError: Invalid register.
    """

    if use_rust:
        if annotation_factories is None:
            annotation_factories = {}
        return _qpy.read_circuit(
            file_obj, version, metadata_deserializer, use_symengine, annotation_factories
        )

    vectors = {}
    if version < 2:
        header, name, metadata = _read_header(file_obj, metadata_deserializer=metadata_deserializer)
    elif version < 12:
        header, name, metadata = _read_header_v2(
            file_obj, version, vectors, metadata_deserializer=metadata_deserializer
        )
    else:
        header, name, metadata = _read_header_v12(
            file_obj, version, vectors, metadata_deserializer=metadata_deserializer
        )

    global_phase = header["global_phase"]
    num_qubits = header["num_qubits"]
    num_clbits = header["num_clbits"]
    num_registers = header["num_registers"]
    num_instructions = header["num_instructions"]
    num_identifiers = header.get("num_vars", 0)
    # `out_registers` is two "name: register" maps segregated by type for the rest of QPY, and
    # `all_registers` is the complete ordered list used to construct the `QuantumCircuit`.
    out_registers = {"q": {}, "c": {}}
    all_registers = []
    out_bits = {"q": [None] * num_qubits, "c": [None] * num_clbits}
    if num_registers > 0:
        if version < 4:
            registers = _read_registers(file_obj, num_registers)
        else:
            registers = _read_registers_v4(file_obj, num_registers)
        for bit_type_label, bit_type, reg_type in [
            ("q", Qubit, QuantumRegister),
            ("c", Clbit, ClassicalRegister),
        ]:
            # This does two passes through the registers. In the first, we're actually just
            # constructing the `Bit` instances: any register that is `standalone` "owns" all its
            # bits in the old Qiskit data model, so we have to construct those by creating the
            # register and taking the bits from them.  That's the case even if that register isn't
            # actually in the circuit, which is why we stored them (with `in_circuit=False`) in QPY.
            #
            # Since there's no guarantees in QPY about the ordering of registers, we have to pass
            # through all registers to create the bits first, because we can't reliably know if a
            # non-standalone register contains bits from a standalone one until we've seen all
            # standalone registers.
            typed_bits = out_bits[bit_type_label]
            typed_registers = registers[bit_type_label]
            for register_name, (standalone, indices, _incircuit) in typed_registers.items():
                if not standalone:
                    continue
                register = reg_type(len(indices), register_name)
                out_registers[bit_type_label][register_name] = register
                for owned, index in zip(register, indices):
                    # Negative indices are for bits that aren't in the circuit.
                    if index >= 0:
                        typed_bits[index] = owned
            # Any remaining unset bits aren't owned, so we can construct them in the standard way.
            typed_bits = [bit if bit is not None else bit_type() for bit in typed_bits]
            # Finally _properly_ construct all the registers.  Bits can be in more than one
            # register, including bits that are old-style "owned" by a register.
            for register_name, (standalone, indices, in_circuit) in typed_registers.items():
                if standalone:
                    register = out_registers[bit_type_label][register_name]
                else:
                    register = reg_type(
                        name=register_name,
                        bits=[typed_bits[x] if x >= 0 else bit_type() for x in indices],
                    )
                    out_registers[bit_type_label][register_name] = register
                if in_circuit:
                    all_registers.append(register)
            out_bits[bit_type_label] = typed_bits
    else:
        out_bits = {
            "q": [Qubit() for _ in out_bits["q"]],
            "c": [Clbit() for _ in out_bits["c"]],
        }
    var_segments, standalone_var_indices = value.read_standalone_vars(file_obj, num_identifiers)
    circ = QuantumCircuit(
        out_bits["q"],
        out_bits["c"],
        *all_registers,
        name=name,
        global_phase=global_phase,
        metadata=metadata,
        inputs=var_segments[type_keys.ExprVarDeclaration.INPUT],
        captures=itertools.chain(
            var_segments[type_keys.ExprVarDeclaration.CAPTURE],
            var_segments[type_keys.ExprVarDeclaration.STRETCH_CAPTURE],
        ),
    )
    for declaration in var_segments[type_keys.ExprVarDeclaration.LOCAL]:
        circ.add_uninitialized_var(declaration)
    for stretch in var_segments[type_keys.ExprVarDeclaration.STRETCH_LOCAL]:
        circ.add_stretch(stretch)
    if version >= 15:
        annotation_state = _read_annotation_states(file_obj, annotation_factories or {})
    else:
        annotation_state = _AnnotationDeserializationState(annotation_factories or {})
    custom_operations = _read_custom_operations(file_obj, version, vectors, annotation_state)
    for _instruction in range(num_instructions):
        _read_instruction(
            file_obj,
            circ,
            out_registers,
            custom_operations,
            version,
            vectors,
            use_symengine,
            standalone_var_indices,
            annotation_state=annotation_state,
        )

    # Consume calibrations, but don't use them since pulse gates are not supported as of Qiskit 2.0
    if version >= 5:
        _read_calibrations(file_obj, version, vectors, metadata_deserializer)

    if version >= 8:
        if version >= 10:
            _read_layout_v2(file_obj, circ)
        else:
            _read_layout(file_obj, circ)
    return circ
