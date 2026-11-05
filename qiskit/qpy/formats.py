# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Binary format definition."""

import struct
from collections import namedtuple


# FILE_HEADER_V10
FILE_HEADER_V10 = namedtuple(
    "FILE_HEADER",
    [
        "preface",
        "qpy_version",
        "major_version",
        "minor_version",
        "patch_version",
        "num_programs",
        "symbolic_encoding",
    ],
)
FILE_HEADER_V10_PACK = "!6sBBBBQc"
FILE_HEADER_V10_SIZE = struct.calcsize(FILE_HEADER_V10_PACK)

# FILE_HEADER
FILE_HEADER = namedtuple(
    "FILE_HEADER",
    ["preface", "qpy_version", "major_version", "minor_version", "patch_version", "num_programs"],
)
FILE_HEADER_PACK = "!6sBBBBQ"
FILE_HEADER_SIZE = struct.calcsize(FILE_HEADER_PACK)

TYPE_KEY = namedtuple("TYPE_KEY", ["key"])
TYPE_KEY_PACK = "!1c"
TYPE_KEY_SIZE = struct.calcsize(TYPE_KEY_PACK)

CIRCUIT_HEADER_V12 = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase_type",
        "global_phase_size",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
        "num_vars",
    ],
)
CIRCUIT_HEADER_V12_PACK = "!H1cHIIQIQI"
CIRCUIT_HEADER_V12_SIZE = struct.calcsize(CIRCUIT_HEADER_V12_PACK)

# CIRCUIT_HEADER_V2
CIRCUIT_HEADER_V2 = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase_type",
        "global_phase_size",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
CIRCUIT_HEADER_V2_PACK = "!H1cHIIQIQ"
CIRCUIT_HEADER_V2_SIZE = struct.calcsize(CIRCUIT_HEADER_V2_PACK)

# CIRCUIT_HEADER
CIRCUIT_HEADER = namedtuple(
    "HEADER",
    [
        "name_size",
        "global_phase",
        "num_qubits",
        "num_clbits",
        "metadata_size",
        "num_registers",
        "num_instructions",
    ],
)
CIRCUIT_HEADER_PACK = "!HdIIQIQ"
CIRCUIT_HEADER_SIZE = struct.calcsize(CIRCUIT_HEADER_PACK)

# CIRCUIT_TABLE_ENTRY
CIRCUIT_TABLE_ENTRY = namedtuple("CIRCUIT_TABLE_ENTRY", ["offset"])
CIRCUIT_TABLE_ENTRY_PACK = "!Q"
CIRCUIT_TABLE_ENTRY_SIZE = struct.calcsize(CIRCUIT_TABLE_ENTRY_PACK)

# REGISTER
REGISTER_V4 = namedtuple("REGISTER", ["type", "standalone", "size", "name_size", "in_circuit"])
REGISTER_V4_PACK = "!1c?IH?"
REGISTER_V4_SIZE = struct.calcsize(REGISTER_V4_PACK)

REGISTER = namedtuple("REGISTER", ["type", "standalone", "size", "name_size"])
REGISTER_PACK = "!1c?IH"
REGISTER_SIZE = struct.calcsize(REGISTER_PACK)

# CIRCUIT_INSTRUCTION
CIRCUIT_INSTRUCTION = namedtuple(
    "CIRCUIT_INSTRUCTION",
    [
        "name_size",
        "label_size",
        "num_parameters",
        "num_qargs",
        "num_cargs",
        "has_condition",
        "condition_register_size",
        "condition_value",
    ],
)
CIRCUIT_INSTRUCTION_PACK = "!HHHII?Hq"
CIRCUIT_INSTRUCTION_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_PACK)

# CIRCUIT_INSTRUCTION_V2
CIRCUIT_INSTRUCTION_V2 = namedtuple(
    "CIRCUIT_INSTRUCTION",
    [
        "name_size",
        "label_size",
        "num_parameters",
        "num_qargs",
        "num_cargs",
        "extras_key",
        "condition_register_size",
        "condition_value",
        "num_ctrl_qubits",
        "ctrl_state",
    ],
)
CIRCUIT_INSTRUCTION_V2_PACK = "!HHHIIBHqII"
CIRCUIT_INSTRUCTION_V2_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_V2_PACK)


# CIRCUIT_INSTRUCTION_ARG
CIRCUIT_INSTRUCTION_ARG = namedtuple("CIRCUIT_INSTRUCTION_ARG", ["type", "size"])
CIRCUIT_INSTRUCTION_ARG_PACK = "!1cI"
CIRCUIT_INSTRUCTION_ARG_SIZE = struct.calcsize(CIRCUIT_INSTRUCTION_ARG_PACK)

ANNOTATION_HEADER_STATIC = namedtuple("ANNOTATION_HEADER_STATIC", ["num_namespaces"])
ANNOTATION_HEADER_STATIC_PACK = "!I"
ANNOTATION_HEADER_STATIC_SIZE = struct.calcsize(ANNOTATION_HEADER_STATIC_PACK)

ANNOTATION_STATE_HEADER = namedtuple("ANNOTATION_STATE_HEADER", ["namespace_size", "state_size"])
ANNOTATION_STATE_HEADER_PACK = "!IQ"
ANNOTATION_STATE_HEADER_SIZE = struct.calcsize(ANNOTATION_STATE_HEADER_PACK)

INSTRUCTION_ANNOTATIONS_HEADER = namedtuple("INSTRUCTION_ANNOTATIONS_HEADER", ["num_annotations"])
INSTRUCTION_ANNOTATIONS_HEADER_PACK = "!I"
INSTRUCTION_ANNOTATIONS_HEADER_SIZE = struct.calcsize(INSTRUCTION_ANNOTATIONS_HEADER_PACK)

INSTRUCTION_ANNOTATION = namedtuple("INSTRUCTION_ANNOTATION", ["namespace_index", "payload_size"])
INSTRUCTION_ANNOTATION_PACK = "!IQ"
INSTRUCTION_ANNOTATION_SIZE = struct.calcsize(INSTRUCTION_ANNOTATION_PACK)

# SparsePauliOp List
SPARSE_PAULI_OP_LIST_ELEM = namedtuple("SPARSE_PAULI_OP_LIST_ELEMENT", ["size"])
SPARSE_PAULI_OP_LIST_ELEM_PACK = "!Q"
SPARSE_PAULI_OP_LIST_ELEM_SIZE = struct.calcsize(SPARSE_PAULI_OP_LIST_ELEM_PACK)

# SparseObservable List - Version 17+
SPARSE_OBSERVABLE = namedtuple(
    "SPARSE_OBSERVABLE",
    ["num_qubits", "coeff_data_len", "bitterm_data_len", "inds_data_len", "bounds_data_len"],
)
SPARSE_OBSERVABLE_PACK = "!IQQQQ"
SPARSE_OBSERVABLE_SIZE = struct.calcsize(SPARSE_OBSERVABLE_PACK)

# Register instruction parameter - Version >=18
#
# A Register param payload holds either a whole classical register or a single clbit, told apart by a
# leading tag byte.  A register's name follows the tag and runs to the end of the payload, whose
# length the enclosing field already gives; a clbit is stored as its index in the circuit.
#
# Up to version 17 both cases shared one untyped string: a register was its bare name, and a clbit
# was a b"\x00" byte followed by its index written out as ASCII digits.
REGISTER_PARAM_TAG_REGISTER = 1
REGISTER_PARAM_TAG_CLBIT = 0
REGISTER_PARAM_TAG_PACK = "!B"
REGISTER_PARAM_TAG_SIZE = struct.calcsize(REGISTER_PARAM_TAG_PACK)
REGISTER_PARAM_CLBIT = namedtuple("REGISTER_PARAM_CLBIT", ["tag", "index"])
REGISTER_PARAM_CLBIT_PACK = "!BI"
REGISTER_PARAM_CLBIT_SIZE = struct.calcsize(REGISTER_PARAM_CLBIT_PACK)

# Pauli Evolution Gate
PAULI_EVOLUTION_DEF = namedtuple(
    "PAULI_EVOLUTION_DEF",
    ["operator_size", "standalone_op", "time_type", "time_size", "synth_method_size"],
)
PAULI_EVOLUTION_DEF_PACK = "!Q?1cQQ"
PAULI_EVOLUTION_DEF_SIZE = struct.calcsize(PAULI_EVOLUTION_DEF_PACK)

# Modifier
MODIFIER_DEF = namedtuple("MODIFIER_DEF", ["type", "num_ctrl_qubits", "ctrl_state", "power"])
MODIFIER_DEF_PACK = "!1cIId"
MODIFIER_DEF_SIZE = struct.calcsize(MODIFIER_DEF_PACK)

# CUSTOM_CIRCUIT_DEF_HEADER
CUSTOM_CIRCUIT_DEF_HEADER = namedtuple("CUSTOM_CIRCUIT_DEF_HEADER", ["size"])
CUSTOM_CIRCUIT_DEF_HEADER_PACK = "!Q"
CUSTOM_CIRCUIT_DEF_HEADER_SIZE = struct.calcsize(CUSTOM_CIRCUIT_DEF_HEADER_PACK)

# CUSTOM_CIRCUIT_INST_DEF_V2
CUSTOM_CIRCUIT_INST_DEF_V2 = namedtuple(
    "CUSTOM_CIRCUIT_INST_DEF",
    [
        "gate_name_size",
        "type",
        "num_qubits",
        "num_clbits",
        "custom_definition",
        "size",
        "num_ctrl_qubits",
        "ctrl_state",
        "base_gate_size",
    ],
)
CUSTOM_CIRCUIT_INST_DEF_V2_PACK = "!H1cII?QIIQ"
CUSTOM_CIRCUIT_INST_DEF_V2_SIZE = struct.calcsize(CUSTOM_CIRCUIT_INST_DEF_V2_PACK)

# CUSTOM_CIRCUIT_INST_DEF
CUSTOM_CIRCUIT_INST_DEF = namedtuple(
    "CUSTOM_CIRCUIT_INST_DEF",
    ["gate_name_size", "type", "num_qubits", "num_clbits", "custom_definition", "size"],
)
CUSTOM_CIRCUIT_INST_DEF_PACK = "!H1cII?Q"
CUSTOM_CIRCUIT_INST_DEF_SIZE = struct.calcsize(CUSTOM_CIRCUIT_INST_DEF_PACK)

# CALIBRATION
CALIBRATION = namedtuple("CALIBRATION", ["num_cals"])
CALIBRATION_PACK = "!H"
CALIBRATION_SIZE = struct.calcsize(CALIBRATION_PACK)

# CALIBRATION_DEF
CALIBRATION_DEF = namedtuple("CALIBRATION_DEF", ["name_size", "num_qubits", "num_params", "type"])
CALIBRATION_DEF_PACK = "!HHH1c"
CALIBRATION_DEF_SIZE = struct.calcsize(CALIBRATION_DEF_PACK)

# SCHEDULE_BLOCK binary format
SCHEDULE_BLOCK_HEADER = namedtuple(
    "SCHEDULE_BLOCK",
    [
        "name_size",
        "metadata_size",
        "num_elements",
    ],
)
SCHEDULE_BLOCK_HEADER_PACK = "!HQH"
SCHEDULE_BLOCK_HEADER_SIZE = struct.calcsize(SCHEDULE_BLOCK_HEADER_PACK)

# WAVEFORM binary format
WAVEFORM = namedtuple("WAVEFORM", ["epsilon", "data_size", "amp_limited"])
WAVEFORM_PACK = "!fI?"
WAVEFORM_SIZE = struct.calcsize(WAVEFORM_PACK)

# SYMBOLIC_PULSE
SYMBOLIC_PULSE = namedtuple(
    "SYMBOLIC_PULSE",
    [
        "type_size",
        "envelope_size",
        "constraints_size",
        "valid_amp_conditions_size",
        "amp_limited",
    ],
)
SYMBOLIC_PULSE_PACK = "!HHHH?"
SYMBOLIC_PULSE_SIZE = struct.calcsize(SYMBOLIC_PULSE_PACK)

# SYMBOLIC_PULSE_V2
SYMBOLIC_PULSE_V2 = namedtuple(
    "SYMBOLIC_PULSE",
    [
        "class_name_size",
        "type_size",
        "envelope_size",
        "constraints_size",
        "valid_amp_conditions_size",
        "amp_limited",
    ],
)
SYMBOLIC_PULSE_PACK_V2 = "!HHHHH?"
SYMBOLIC_PULSE_SIZE_V2 = struct.calcsize(SYMBOLIC_PULSE_PACK_V2)

# INSTRUCTION_PARAM
INSTRUCTION_PARAM = namedtuple("INSTRUCTION_PARAM", ["type", "size"])
INSTRUCTION_PARAM_PACK = "!1cQ"
INSTRUCTION_PARAM_SIZE = struct.calcsize(INSTRUCTION_PARAM_PACK)

# PARAMETER
PARAMETER = namedtuple("PARAMETER", ["name_size", "uuid"])
PARAMETER_PACK = "!H16s"
PARAMETER_SIZE = struct.calcsize(PARAMETER_PACK)

# PARAMETEREXPRESSION_ENTRY
PARAM_EXPR_ELEM_V13 = namedtuple(
    "PARAM_EXPR_ELEM_V13", ["OP_CODE", "LHS_TYPE", "LHS", "RHS_TYPE", "RHS"]
)
PARAM_EXPR_ELEM_V13_PACK = "!Bc16sc16s"
PARAM_EXPR_ELEM_V13_SIZE = struct.calcsize(PARAM_EXPR_ELEM_V13_PACK)

# COMPLEX
COMPLEX = namedtuple("COMPLEX", ["real", "imag"])
COMPLEX_PACK = "!dd"
COMPLEX_SIZE = struct.calcsize(COMPLEX_PACK)

# PARAMETER_VECTOR_ELEMENT
PARAMETER_VECTOR_ELEMENT = namedtuple(
    "PARAMETER_VECTOR_ELEMENT", ["vector_name_size", "vector_size", "uuid", "index"]
)
PARAMETER_VECTOR_ELEMENT_PACK = "!HQ16sQ"
PARAMETER_VECTOR_ELEMENT_SIZE = struct.calcsize(PARAMETER_VECTOR_ELEMENT_PACK)

# PARAMETER_EXPR
PARAMETER_EXPR = namedtuple("PARAMETER_EXPR", ["map_elements", "expr_size"])
PARAMETER_EXPR_PACK = "!QQ"
PARAMETER_EXPR_SIZE = struct.calcsize(PARAMETER_EXPR_PACK)

# PARAMETER_EXPR_MAP_ELEM_V3
PARAM_EXPR_MAP_ELEM_V3 = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["symbol_type", "type", "size"])
PARAM_EXPR_MAP_ELEM_V3_PACK = "!ccQ"
PARAM_EXPR_MAP_ELEM_V3_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_V3_PACK)

# PARAMETER_EXPR_MAP_ELEM
PARAM_EXPR_MAP_ELEM = namedtuple("PARAMETER_EXPR_MAP_ELEM", ["type", "size"])
PARAM_EXPR_MAP_ELEM_PACK = "!cQ"
PARAM_EXPR_MAP_ELEM_SIZE = struct.calcsize(PARAM_EXPR_MAP_ELEM_PACK)

# RANGE
RANGE = namedtuple("RANGE", ["start", "stop", "step"])
RANGE_PACK = "!qqq"
RANGE_SIZE = struct.calcsize(RANGE_PACK)

# SEQUENCE
SEQUENCE = namedtuple("SEQUENCE", ["num_elements"])
SEQUENCE_PACK = "!Q"
SEQUENCE_SIZE = struct.calcsize(SEQUENCE_PACK)

# MAP_ITEM
MAP_ITEM = namedtuple("MAP_ITEM", ["key_size", "type", "size"])
MAP_ITEM_PACK = "!H1cH"
MAP_ITEM_SIZE = struct.calcsize(MAP_ITEM_PACK)

LAYOUT_V2 = namedtuple(
    "LAYOUT",
    [
        "exists",
        "initial_layout_size",
        "input_mapping_size",
        "final_layout_size",
        "extra_registers",
        "input_qubit_count",
    ],
)
LAYOUT_V2_PACK = "!?iiiIi"
LAYOUT_V2_SIZE = struct.calcsize(LAYOUT_V2_PACK)


LAYOUT = namedtuple(
    "LAYOUT",
    ["exists", "initial_layout_size", "input_mapping_size", "final_layout_size", "extra_registers"],
)
LAYOUT_PACK = "!?iiiI"
LAYOUT_SIZE = struct.calcsize(LAYOUT_PACK)

INITIAL_LAYOUT_BIT = namedtuple("INITIAL_LAYOUT_BIT", ["index", "register_size"])
INITIAL_LAYOUT_BIT_PACK = "!ii"
INITIAL_LAYOUT_BIT_SIZE = struct.calcsize(INITIAL_LAYOUT_BIT_PACK)

# EXPR_VAR_DECLARATION

EXPR_VAR_DECLARATION = namedtuple("EXPR_VAR_DECLARATION", ["uuid_bytes", "usage", "name_size"])
EXPR_VAR_DECLARATION_PACK = "!16scH"
EXPR_VAR_DECLARATION_SIZE = struct.calcsize(EXPR_VAR_DECLARATION_PACK)


# EXPRESSION

EXPRESSION_DISCRIMINATOR_SIZE = 1

EXPRESSION_CAST = namedtuple("EXPRESSION_CAST", ["implicit"])
EXPRESSION_CAST_PACK = "!?"
EXPRESSION_CAST_SIZE = struct.calcsize(EXPRESSION_CAST_PACK)

EXPRESSION_UNARY = namedtuple("EXPRESSION_UNARY", ["opcode"])
EXPRESSION_UNARY_PACK = "!B"
EXPRESSION_UNARY_SIZE = struct.calcsize(EXPRESSION_UNARY_PACK)

EXPRESSION_BINARY = namedtuple("EXPRESSION_BINARY", ["opcode"])
EXPRESSION_BINARY_PACK = "!B"
EXPRESSION_BINARY_SIZE = struct.calcsize(EXPRESSION_BINARY_PACK)

EXPRESSION_STRETCH = namedtuple("EXPRESSION_STRETCH", ["var_index"])
EXPRESSION_STRETCH_PACK = "!H"
EXPRESSION_STRETCH_SIZE = struct.calcsize(EXPRESSION_STRETCH_PACK)


# EXPR_TYPE

EXPR_TYPE_DISCRIMINATOR_SIZE = 1

EXPR_TYPE_DURATION = namedtuple("EXPR_TYPE_DURATION", [])
EXPR_TYPE_DURATION_PACK = "!"
EXPR_TYPE_DURATION_SIZE = struct.calcsize(EXPR_TYPE_DURATION_PACK)

EXPR_TYPE_FLOAT = namedtuple("EXPR_TYPE_FLOAT", [])
EXPR_TYPE_FLOAT_PACK = "!"
EXPR_TYPE_FLOAT_SIZE = struct.calcsize(EXPR_TYPE_FLOAT_PACK)

EXPR_TYPE_BOOL = namedtuple("EXPR_TYPE_BOOL", [])
EXPR_TYPE_BOOL_PACK = "!"
EXPR_TYPE_BOOL_SIZE = struct.calcsize(EXPR_TYPE_BOOL_PACK)

EXPR_TYPE_UINT = namedtuple("EXPR_TYPE_UINT", ["width"])
EXPR_TYPE_UINT_PACK = "!L"
EXPR_TYPE_UINT_SIZE = struct.calcsize(EXPR_TYPE_UINT_PACK)


# EXPR_VAR

EXPR_VAR_DISCRIMINATOR_SIZE = 1

EXPR_VAR_CLBIT = namedtuple("EXPR_VAR_CLBIT", ["index"])
EXPR_VAR_CLBIT_PACK = "!L"
EXPR_VAR_CLBIT_SIZE = struct.calcsize(EXPR_VAR_CLBIT_PACK)

EXPR_VAR_REGISTER = namedtuple("EXPR_VAR_REGISTER", ["reg_name_size"])
EXPR_VAR_REGISTER_PACK = "!H"
EXPR_VAR_REGISTER_SIZE = struct.calcsize(EXPR_VAR_REGISTER_PACK)

EXPR_VAR_UUID = namedtuple("EXPR_VAR_UUID", ["var_index"])
EXPR_VAR_UUID_PACK = "!H"
EXPR_VAR_UUID_SIZE = struct.calcsize(EXPR_VAR_UUID_PACK)


# EXPR_VALUE

EXPR_VALUE_DISCRIMINATOR_SIZE = 1

EXPR_VALUE_FLOAT = namedtuple("EXPR_VALUE_FLOAT", ["value"])
EXPR_VALUE_FLOAT_PACK = "!d"
EXPR_VALUE_FLOAT_SIZE = struct.calcsize(EXPR_VALUE_FLOAT_PACK)

EXPR_VALUE_BOOL = namedtuple("EXPR_VALUE_BOOL", ["value"])
EXPR_VALUE_BOOL_PACK = "!?"
EXPR_VALUE_BOOL_SIZE = struct.calcsize(EXPR_VALUE_BOOL_PACK)

EXPR_VALUE_INT = namedtuple("EXPR_VALUE_INT", ["num_bytes"])
EXPR_VALUE_INT_PACK = "!B"
EXPR_VALUE_INT_SIZE = struct.calcsize(EXPR_VALUE_INT_PACK)


# DURATION

DURATION_DISCRIMINATOR_SIZE = 1

DURATION_DT = namedtuple("DURATION_DT", ["value"])
DURATION_DT_PACK = "!Q"
DURATION_DT_SIZE = struct.calcsize(DURATION_DT_PACK)

DURATION_PS = namedtuple("DURATION_PS", ["value"])
DURATION_PS_PACK = "!d"
DURATION_PS_SIZE = struct.calcsize(DURATION_PS_PACK)

DURATION_NS = namedtuple("DURATION_NS", ["value"])
DURATION_NS_PACK = "!d"
DURATION_NS_SIZE = struct.calcsize(DURATION_NS_PACK)

DURATION_US = namedtuple("DURATION_US", ["value"])
DURATION_US_PACK = "!d"
DURATION_US_SIZE = struct.calcsize(DURATION_US_PACK)

DURATION_MS = namedtuple("DURATION_MS", ["value"])
DURATION_MS_PACK = "!d"
DURATION_MS_SIZE = struct.calcsize(DURATION_MS_PACK)

DURATION_S = namedtuple("DURATION_S", ["value"])
DURATION_S_PACK = "!d"
DURATION_S_SIZE = struct.calcsize(DURATION_S_PACK)
