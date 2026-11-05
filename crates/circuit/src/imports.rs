// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// This module contains objects imported from Python that are reused. These are
// typically data model classes that are used to identify an object, or for
// python side casting

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;

use crate::operations::{STANDARD_GATE_SIZE, StandardGate};
use qiskit_util::py::ImportOnceCell;

pub use qiskit_util::py::imports::*;

pub static OPERATION: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.operation", "Operation");
pub static INSTRUCTION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.instruction", "Instruction");
pub static GATE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.gate", "Gate");
pub static CONTROL_FLOW_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.controlflow", "ControlFlowOp");
pub static PARAMETER: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.parameter", "Parameter");
pub static PARAMETER_EXPRESSION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parameterexpression", "ParameterExpression");
pub static PARAMETER_VECTOR: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parametervector", "ParameterVector");
pub static PARAMETER_VECTOR_ELEMENT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parametervector", "ParameterVectorElement");
pub static QUANTUM_CIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.quantumcircuit", "QuantumCircuit");
pub static SINGLETON_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.singleton", "SingletonGate");
pub static SINGLETON_CONTROLLED_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.singleton", "SingletonControlledGate");
pub static VARIABLE_MAPPER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit._classical_resource_map", "VariableMapper");
pub static IF_ELSE_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "IfElseOp");
pub static FOR_LOOP_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "ForLoopOp");
pub static SWITCH_CASE_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "SwitchCaseOp");
pub static SWITCH_CASE_DEFAULT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "CASE_DEFAULT");
pub static WHILE_LOOP_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "WhileLoopOp");
pub static BREAK_LOOP_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "BreakLoopOp");
pub static CONTINUE_LOOP_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "ContinueLoopOp");
pub static STORE_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Store");
pub static BOX_OP: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "BoxOp");
pub static DAG_NODE: ImportOnceCell = ImportOnceCell::new("qiskit.dagcircuit", "DAGNode");
pub static CONTROLLED_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "ControlledGate");
pub static ANNOTATED_OPERATION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "AnnotatedOperation");
pub static MODIFIER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.annotated_operation", "Modifier");
pub static INVERSE_MODIFIER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.annotated_operation", "InverseModifier");
pub static CONTROL_MODIFIER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.annotated_operation", "ControlModifier");
pub static POWER_MODIFIER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.annotated_operation", "PowerModifier");
pub static CASE_DEFAULT: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "CASE_DEFAULT");
pub static CLBIT: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Clbit");
pub static CLASSICAL_REGISTER: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit", "ClassicalRegister");
pub static QI_OPERATOR: ImportOnceCell = ImportOnceCell::new("qiskit.quantum_info", "Operator");
pub static CLIFFORD: ImportOnceCell =
    ImportOnceCell::new("qiskit.quantum_info.operators.symplectic", "Clifford");
pub static SPARSE_PAULI_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.quantum_info.operators", "SparsePauliOp");
pub static CIRCUIT_TO_DAG: ImportOnceCell =
    ImportOnceCell::new("qiskit.converters", "circuit_to_dag");
pub static DAG_TO_CIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.converters", "dag_to_circuit");
pub static BARRIER: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Barrier");
pub static DELAY: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Delay");
pub static MEASURE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Measure");
pub static RESET: ImportOnceCell = ImportOnceCell::new("qiskit.circuit", "Reset");
pub static UNARY_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.classical.expr.expr", "_UnaryOp");
pub static BINARY_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.classical.expr.expr", "_BinaryOp");
pub static UNITARY_GATE: ImportOnceCell = ImportOnceCell::new(
    "qiskit.circuit.library.generalized_gates.unitary",
    "UnitaryGate",
);
pub static PAULI_PRODUCT_ROTATION_GATE: ImportOnceCell = ImportOnceCell::new(
    "qiskit.circuit.library.generalized_gates.pauli_product_rotation",
    "PauliProductRotationGate",
);
pub static PAULI_PRODUCT_MEASUREMENT: ImportOnceCell = ImportOnceCell::new(
    "qiskit.circuit.library.pauli_product_measurement",
    "PauliProductMeasurement",
);
pub static MCPHASE_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library", "MCPhaseGate");
pub static PAULI_GATE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.library", "PauliGate");
pub static PAULI_EVOLUTION_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library", "PauliEvolutionGate");
pub static MCMT_GATE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.library", "MCMTGate");
pub static BLUEPRINT_CIRCUIT: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library", "BlueprintCircuit");
pub static PAULI_ROTATION_TRACE_AND_DIM: ImportOnceCell = ImportOnceCell::new(
    "qiskit.circuit.library.pauli_evolution",
    "_pauli_rotation_trace_and_dim",
);
pub static MERGE_TWO_PAULI_EVOLUTIONS: ImportOnceCell = ImportOnceCell::new(
    "qiskit.circuit.library.pauli_evolution",
    "_merge_two_pauli_evolutions",
);
pub static XX_DECOMPOSER: ImportOnceCell =
    ImportOnceCell::new("qiskit.synthesis.two_qubit.xx_decompose", "XXDecomposer");
pub static XX_EMBODIMENTS: ImportOnceCell =
    ImportOnceCell::new("qiskit.synthesis.two_qubit.xx_decompose", "XXEmbodiments");
pub static HLS_SYNTHESIZE_OP_USING_PLUGINS: ImportOnceCell = ImportOnceCell::new(
    "qiskit.transpiler.passes.synthesis.high_level_synthesis",
    "_synthesize_op_using_plugins",
);
pub static CONTROL_FLOW_CONDITION_RESOURCES: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.controlflow", "condition_resources");
pub static CONTROL_FLOW_NODE_RESOURCES: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.controlflow", "node_resources");

pub static CONTROL_FLOW_BOX_OP: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.controlflow", "BoxOp");
pub static TRANSPILER_LAYOUT: ImportOnceCell =
    ImportOnceCell::new("qiskit.transpiler.layout", "TranspileLayout");
pub static LAYOUT: ImportOnceCell = ImportOnceCell::new("qiskit.transpiler.layout", "Layout");
pub static SYMPIFY_PARAMETER_EXPRESSION: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.parameterexpression", "sympify");
pub static TRANSPILE_LAYOUT: ImportOnceCell =
    ImportOnceCell::new("qiskit.transpiler.layout", "TranspileLayout");
pub static STORE: ImportOnceCell = ImportOnceCell::new("qiskit.circuit.store", "Store");

/// A mapping from the enum variant in crate::operations::StandardGate to the python
/// module path and class name to import it. This is used to populate the conversion table
/// when a gate is added directly via the StandardGate path and there isn't a Python object
/// to poll the _standard_gate attribute for.
///
/// NOTE: the order here is significant, the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
// TODO: replace placeholders with actual implementation
static STDGATE_IMPORT_PATHS: [[&str; 2]; STANDARD_GATE_SIZE] = [
    // GlobalPhaseGate = 0
    [
        "qiskit.circuit.library.standard_gates.global_phase",
        "GlobalPhaseGate",
    ],
    // HGate = 1
    ["qiskit.circuit.library.standard_gates.h", "HGate"],
    // IGate = 2
    ["qiskit.circuit.library.standard_gates.i", "IGate"],
    // XGate = 3
    ["qiskit.circuit.library.standard_gates.x", "XGate"],
    // YGate = 4
    ["qiskit.circuit.library.standard_gates.y", "YGate"],
    // ZGate = 5
    ["qiskit.circuit.library.standard_gates.z", "ZGate"],
    // PhaseGate = 6
    ["qiskit.circuit.library.standard_gates.p", "PhaseGate"],
    // RGate 7
    ["qiskit.circuit.library.standard_gates.r", "RGate"],
    // RXGate = 8
    ["qiskit.circuit.library.standard_gates.rx", "RXGate"],
    // RYGate = 9
    ["qiskit.circuit.library.standard_gates.ry", "RYGate"],
    // RZGate = 10
    ["qiskit.circuit.library.standard_gates.rz", "RZGate"],
    // SGate = 11
    ["qiskit.circuit.library.standard_gates.s", "SGate"],
    // SdgGate = 12
    ["qiskit.circuit.library.standard_gates.s", "SdgGate"],
    // SXGate = 13
    ["qiskit.circuit.library.standard_gates.sx", "SXGate"],
    // SXdgGate = 14
    ["qiskit.circuit.library.standard_gates.sx", "SXdgGate"],
    // TGate = 15
    ["qiskit.circuit.library.standard_gates.t", "TGate"],
    // TdgGate = 16
    ["qiskit.circuit.library.standard_gates.t", "TdgGate"],
    // UGate = 17
    ["qiskit.circuit.library.standard_gates.u", "UGate"],
    // U1Gate = 18
    ["qiskit.circuit.library.standard_gates.u1", "U1Gate"],
    // U2Gate = 19
    ["qiskit.circuit.library.standard_gates.u2", "U2Gate"],
    // U3Gate = 20
    ["qiskit.circuit.library.standard_gates.u3", "U3Gate"],
    // CHGate = 21
    ["qiskit.circuit.library.standard_gates.h", "CHGate"],
    // CXGate = 22
    ["qiskit.circuit.library.standard_gates.x", "CXGate"],
    // CYGate = 23
    ["qiskit.circuit.library.standard_gates.y", "CYGate"],
    // CZGate = 24
    ["qiskit.circuit.library.standard_gates.z", "CZGate"],
    // DCXGate = 25
    ["qiskit.circuit.library.standard_gates.dcx", "DCXGate"],
    // ECRGate = 26
    ["qiskit.circuit.library.standard_gates.ecr", "ECRGate"],
    // SwapGate = 27
    ["qiskit.circuit.library.standard_gates.swap", "SwapGate"],
    // iSWAPGate = 28
    ["qiskit.circuit.library.standard_gates.iswap", "iSwapGate"],
    // CPhaseGate = 29
    ["qiskit.circuit.library.standard_gates.p", "CPhaseGate"],
    // CRXGate = 30
    ["qiskit.circuit.library.standard_gates.rx", "CRXGate"],
    // CRYGate = 31
    ["qiskit.circuit.library.standard_gates.ry", "CRYGate"],
    // CRZGate = 32
    ["qiskit.circuit.library.standard_gates.rz", "CRZGate"],
    // CSGate = 33
    ["qiskit.circuit.library.standard_gates.s", "CSGate"],
    // CSdgGate = 34
    ["qiskit.circuit.library.standard_gates.s", "CSdgGate"],
    // CSXGate = 35
    ["qiskit.circuit.library.standard_gates.sx", "CSXGate"],
    // CUGate = 36
    ["qiskit.circuit.library.standard_gates.u", "CUGate"],
    // CU1Gate = 37
    ["qiskit.circuit.library.standard_gates.u1", "CU1Gate"],
    // CU3Gate = 38
    ["qiskit.circuit.library.standard_gates.u3", "CU3Gate"],
    // RXXGate = 39
    ["qiskit.circuit.library.standard_gates.rxx", "RXXGate"],
    // RYYGate = 40
    ["qiskit.circuit.library.standard_gates.ryy", "RYYGate"],
    // RZZGate = 41
    ["qiskit.circuit.library.standard_gates.rzz", "RZZGate"],
    // RZXGate = 42
    ["qiskit.circuit.library.standard_gates.rzx", "RZXGate"],
    // XXMinusYYGate = 43
    [
        "qiskit.circuit.library.standard_gates.xx_minus_yy",
        "XXMinusYYGate",
    ],
    // XXPlusYYGate = 44
    [
        "qiskit.circuit.library.standard_gates.xx_plus_yy",
        "XXPlusYYGate",
    ],
    // CCXGate = 45
    ["qiskit.circuit.library.standard_gates.x", "CCXGate"],
    // CCZGate = 46
    ["qiskit.circuit.library.standard_gates.z", "CCZGate"],
    // CSwapGate = 47
    ["qiskit.circuit.library.standard_gates.swap", "CSwapGate"],
    // RCCXGate = 48
    ["qiskit.circuit.library.standard_gates.x", "RCCXGate"],
    // C3XGate = 49
    ["qiskit.circuit.library.standard_gates.x", "C3XGate"],
    // C3SXGate = 50
    ["qiskit.circuit.library.standard_gates.x", "C3SXGate"],
    // RC3XGate = 51
    ["qiskit.circuit.library.standard_gates.x", "RC3XGate"],
];

/// A mapping from the enum variant in crate::operations::StandardGate to the python object for the
/// class that matches it. This is typically used when we need to convert from the internal rust
/// representation to a Python object for a python user to interact with.
///
/// NOTE: the order here is significant it must match the StandardGate variant's number must match
/// index of it's entry in this table. This is all done statically for performance
static STDGATE_PYTHON_GATES: [PyOnceLock<Py<PyAny>>; STANDARD_GATE_SIZE] = [
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
    PyOnceLock::new(),
];

#[inline]
pub fn get_std_gate_class_name(rs_gate: &StandardGate) -> String {
    STDGATE_IMPORT_PATHS[*rs_gate as usize][1].to_string()
}

#[inline]
pub fn get_std_gate_class(py: Python, rs_gate: StandardGate) -> PyResult<&'static Py<PyAny>> {
    STDGATE_PYTHON_GATES[rs_gate as usize].get_or_try_init(py, || {
        let [py_mod, py_class] = STDGATE_IMPORT_PATHS[rs_gate as usize];
        Ok(py.import(py_mod)?.getattr(py_class)?.unbind())
    })
}
