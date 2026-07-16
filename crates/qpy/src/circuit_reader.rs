// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Circuit reader module: converts a qpy file into a  QuantumCircuit

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.

use binrw::Endian;
use hashbrown::HashMap;
use num_bigint::BigUint;
use num_complex::Complex64;
use numpy::IntoPyArray;
use pyo3::IntoPyObjectExt;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny, PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use qiskit_circuit::bit::{
    ClassicalRegister, QuantumRegister, Register, ShareableClbit, ShareableQubit,
};
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::instruction::{Parameters, create_py_op};
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{
    ArrayType, BoxDuration, CaseSpecifier, Condition, ControlFlow, ControlFlowInstruction,
    ControlFlowType, LoopParam, Param, PauliBased, PauliProductMeasurement, PauliProductRotation,
    StandardInstruction, StandardInstructionType, SwitchTarget, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter::parameter_expression::{ParameterExpression, PyParameter};
use qiskit_circuit::var_stretch_container::{StretchType, VarType};
use qiskit_circuit::{Block, classical, imports};
use qiskit_circuit::{Clbit, Qubit};
use qiskit_quantum_info::sparse_observable::BitTerm;
use qiskit_quantum_info::sparse_observable::SparseObservable;
use std::str::FromStr;
use std::sync::Arc;

use smallvec::SmallVec;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::consts::standard_gate_from_gate_class_name;
use crate::error::QpyError;
use crate::formats;
use crate::formats::ConditionData;
use crate::formats::QPYCircuit;
use crate::params::generic_value_to_param;
use crate::py_methods::{
    PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME, PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME,
    UNITARY_GATE_CLASS_NAME, get_python_gate_class, py_convert_from_generic_value,
    py_deserialize_numpy_object,
};
use crate::value::ParamRegisterValue;
use crate::value::unpack_for_collection;
use crate::value::{
    BitType, CircuitInstructionType, ExpressionType, ExpressionVarDeclaration, GenericValue,
    QPYReadData, RegisterType, ValueType, deserialize_with_args, load_param_register_value,
    load_value, unpack_duration_value, unpack_generic_value,
};

use ndarray::{Array2, ShapeBuilder};
use npyz::NpyFile;
use std::io::Cursor;

// This is a helper struct, designed to pass data within methods
// It is not meant to be serialized, so it's not in formats.rs
#[derive(Debug)]
struct CustomCircuitInstructionData {
    gate_type: CircuitInstructionType,
    num_qubits: u32,
    num_clbits: u32,
    definition_circuit: Option<Py<PyAny>>,
    base_gate_raw: Bytes,
}

// This is a helper enum to make the code clearer by splitting instruction reading into cases
#[derive(Debug)]
pub enum InstructionType {
    // these instruction types are rust-native
    StandardGate,
    StandardInstruction,
    PauliProductMeasurement,
    PauliProductRotation,
    Unitary,
    ControlFlow,
    // covers instruction types require resorting to python space
    Custom,
    Python,
}

fn deserialize_standard_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
) -> Option<StandardInstruction> {
    match instruction.gate_class_name.as_str() {
        "Barrier" => Some(StandardInstruction::Barrier(instruction.num_qargs)),
        "Measure" => Some(StandardInstruction::Measure),
        "Reset" => Some(StandardInstruction::Reset),
        // TODO it seems currently delays are not treated correctly, and the unit is not stored
        // even in Python QPY. We need to fix the writer to correctly store the delay unit as second parameter
        "Delay" => {
            let unit = if !instruction.params.is_empty() {
                if instruction.params.len() >= 2 {
                    // both duration and unit params; extract the unit param
                    // TODO: for now returning default value; this should be fixed as part of the fix mentioned above
                    qiskit_circuit::operations::DelayUnit::DT
                } else {
                    // only duration; check whether it's an expression
                    if [
                        ValueType::Expression,
                        ValueType::ParameterExpression,
                        ValueType::Parameter,
                        ValueType::ParameterVector,
                    ]
                    .contains(&instruction.params[0].type_key)
                    {
                        qiskit_circuit::operations::DelayUnit::EXPR
                    } else {
                        qiskit_circuit::operations::DelayUnit::DT
                    }
                }
            } else {
                // default case
                qiskit_circuit::operations::DelayUnit::DT
            };
            Some(StandardInstruction::Delay(unit))
        }
        _ => None,
    }
}

fn unpack_condition(
    condition_pack: &formats::ConditionPack,
    qpy_data: &mut QPYReadData,
) -> Result<Option<Condition>, QpyError> {
    match &condition_pack.data {
        ConditionData::None => Ok(None),
        ConditionData::Expression(exp_pack) => {
            let exp_value = unpack_generic_value(exp_pack, qpy_data, Endian::Big)?;
            match exp_value {
                GenericValue::Expression(exp) => Ok(Some(Condition::Expr(exp.clone()))),
                _ => Err(QpyError::InvalidExpression(
                    "could not determine expression in conditional".to_string(),
                )),
            }
        }
        ConditionData::Register(bytes) => match load_param_register_value(bytes, qpy_data)? {
            ParamRegisterValue::ShareableClbit(clbit) => {
                Ok(Some(Condition::Bit(clbit, condition_pack.value != 0)))
            }
            ParamRegisterValue::Register(reg) => Ok(Some(Condition::Register(
                reg,
                BigUint::from(condition_pack.value as u64),
            ))),
        },
    }
}

fn recognize_instruction_type(
    instruction: &formats::CircuitInstructionV2Pack,
    custom_instructions: &HashMap<String, CustomCircuitInstructionData>,
) -> InstructionType {
    let name = instruction.gate_class_name.as_str();
    if name == PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME {
        InstructionType::PauliProductMeasurement
    } else if name == PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME {
        InstructionType::PauliProductRotation
    } else if name == UNITARY_GATE_CLASS_NAME {
        InstructionType::Unitary
    } else if ControlFlowType::from_str(name).is_ok()
        || matches!(
            name,
            "IfElseOp"
                | "WhileLoopOp"
                | "ForLoopOp"
                | "BreakLoopOp"
                | "ContinueLoopOp"
                | "SwitchCaseOp"
        )
    {
        InstructionType::ControlFlow
    }
    // StandardInstructionType names are lowercase, actual class names are capitalized
    else if StandardInstructionType::from_str(name).is_ok()
        || ["Barrier", "Delay", "Measure", "Reset"].contains(&name)
    {
        InstructionType::StandardInstruction
    } else if custom_instructions.get(name).is_some() {
        InstructionType::Custom
    } else {
        // This can either be a standard gate, or something Pythonic.
        // For standard gate, we need both the gate class name to be standard, and the controls should be standard as well
        let has_nonstandard_control = instruction.num_ctrl_qubits > 0
            && (instruction.ctrl_state != (1 << instruction.num_ctrl_qubits) - 1);
        let standard_gate_name =
            standard_gate_from_gate_class_name(instruction.gate_class_name.as_str()).is_some();
        if !has_nonstandard_control && standard_gate_name {
            InstructionType::StandardGate
        } else {
            // it is either a python gate, a python instruction or a python operation; all treated in the same manner
            InstructionType::Python
        }
    }
}

fn get_instruction_bits(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> (Interned<[Qubit]>, Interned<[Clbit]>) {
    let mut qubit_indices = Vec::new();
    let mut clbit_indices = Vec::new();
    for arg in &instruction.bit_data {
        match arg.bit_type {
            BitType::Qubit => qubit_indices.push(Qubit(arg.index)),
            BitType::Clbit => clbit_indices.push(Clbit(arg.index)),
        };
    }
    let qubits = qpy_data.circuit_data.add_qargs(&qubit_indices);
    let clbits = qpy_data.circuit_data.add_cargs(&clbit_indices);
    (qubits, clbits)
}

// Unpacks the instruction's parameters to a list of generic values
// Note that params are stored to enable reconstructing the instruction; they are not
// necessarily the "usual" instruction parameters modeled in rust using Operations::Param
fn get_instruction_values(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
    endian: Endian,
) -> Result<Vec<GenericValue>, QpyError> {
    // note that numbers are not read correctly - they are read in big endian, but for instruction parameters, due to historical reasons,
    // they are stored in little endian
    let inst_params: Vec<GenericValue> = instruction
        .params
        .iter()
        .map(|packed_param: &formats::GenericDataPack| {
            unpack_generic_value(packed_param, qpy_data, endian)
        })
        .collect::<Result<_, QpyError>>()?;
    Ok(inst_params)
}

// converts a list of generic values to the params format expected for a PackedInstruction params
pub fn instruction_values_to_params(
    values: Vec<GenericValue>,
    qpy_data: &mut QPYReadData,
) -> Result<Option<Box<Parameters<Block>>>, QpyError> {
    // currently QPY has no dedicated representation for blocks, only for py circuit objects
    // so we use the following heuristic: if all the NON-NULL elements of `values` are circuits
    // treat `values` as a vector of blocks
    // Null values represent missing else blocks in if-else statements and should be filtered out

    // Filter out Null values and check if remaining values are all circuits
    let non_null_values: Vec<&GenericValue> = values
        .iter()
        .filter(|val| !matches!(val, GenericValue::Null))
        .collect();

    if !non_null_values.is_empty()
        && non_null_values
            .iter()
            .all(|val| matches!(val, GenericValue::Circuit(_) | GenericValue::CircuitData(_)))
    {
        // blocks
        let inst_blocks: Vec<CircuitData> = values
            .iter()
            .filter_map(|value| value.as_circuit_data())
            .collect();
        let params = Parameters::Blocks(inst_blocks);
        Ok(qpy_data
            .circuit_data
            .extract_blocks_from_circuit_parameters(Some(&params)))
    } else {
        // params
        let inst_params: Vec<Param> = values
            .into_iter()
            .map(|value| -> Result<_, QpyError> {
                match value {
                    GenericValue::Float64(float) => Ok(Param::Float(float)),
                    GenericValue::ParameterExpression(exp) => Ok(Param::ParameterExpression(exp)),
                    GenericValue::ParameterExpressionSymbol(symbol)
                    | GenericValue::ParameterExpressionVectorSymbol(symbol) => {
                        Ok(Param::ParameterExpression(Arc::new(
                            ParameterExpression::from_arc_symbol(symbol),
                        )))
                    }
                    _ => Ok(Param::Obj(py_convert_from_generic_value(&value)?)),
                }
            })
            .collect::<Result<_, QpyError>>()?;
        Ok((!inst_params.is_empty()).then(|| {
            Box::new(Parameters::Params(SmallVec::<[Param; 3]>::from_vec(
                inst_params,
            )))
        }))
    }
}

fn unpack_annotations(
    packed_annotations: &Option<formats::InstructionsAnnotationPack>,
    qpy_data: &mut QPYReadData,
) -> Result<Vec<Py<PyAny>>, QpyError> {
    if let Some(annotations_vec) = packed_annotations {
        annotations_vec
            .annotations
            .iter()
            .map(|annotation| {
                qpy_data
                    .annotation_handler
                    .load(annotation.namespace_index, annotation.payload.clone())
            })
            .collect::<Result<_, QpyError>>()
    } else {
        Ok(Vec::new())
    }
}

/// create a new instruction from the packed data
fn unpack_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
    custom_instructions: &HashMap<String, CustomCircuitInstructionData>,
    qpy_data: &mut QPYReadData,
) -> Result<PackedInstruction, QpyError> {
    let label = (!instruction.label.is_empty()).then(|| Box::new(instruction.label.clone()));
    let instruction_type = recognize_instruction_type(instruction, custom_instructions);
    let (op, parameter_values) = match instruction_type {
        InstructionType::StandardGate => unpack_standard_gate(instruction, qpy_data)?,
        InstructionType::StandardInstruction => unpack_standard_instruction(instruction, qpy_data)?,
        InstructionType::PauliProductMeasurement => {
            unpack_pauli_product_measurement(instruction, qpy_data)?
        }
        InstructionType::PauliProductRotation => {
            unpack_pauli_product_rotation(instruction, qpy_data)?
        }
        InstructionType::Unitary => unpack_unitary(instruction, qpy_data)?,
        InstructionType::ControlFlow => unpack_control_flow(instruction, qpy_data)?,
        InstructionType::Custom => {
            unpack_custom_instruction(instruction, label.as_deref(), qpy_data, custom_instructions)?
        }
        InstructionType::Python => unpack_py_instruction(instruction, label.as_deref(), qpy_data)?,
    };
    let (qubits, clbits) = get_instruction_bits(instruction, qpy_data);
    let params = instruction_values_to_params(parameter_values, qpy_data)?;

    // Check if this is a non-control-flow instruction with a condition
    // If so, wrap it in an IfElseOp (for backwards compatibility with old QPY versions)
    let condition = unpack_condition(&instruction.condition, qpy_data)?;
    let (op, params) = match condition {
        Some(cond) if !matches!(instruction_type, InstructionType::ControlFlow) => {
            wrap_conditional_gate(instruction, op, cond, qubits, clbits, params, qpy_data)?
        }
        _ => (op, params),
    };

    Ok(PackedInstruction {
        op,
        qubits,
        clbits,
        params,
        label,
        #[cfg(feature = "cache_pygates")]
        py_op: std::sync::OnceLock::new(),
    })
}

fn unpack_standard_gate(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let op = if let Some(gate) =
        standard_gate_from_gate_class_name(instruction.gate_class_name.as_str())
    {
        PackedOperation::from_standard_gate(gate)
    } else {
        return Err(QpyError::InvalidInstruction(format!(
            "Unrecognized standard gate {}",
            instruction.gate_class_name
        )));
    };
    let param_values = get_instruction_values(instruction, qpy_data, Endian::Little)?;
    Ok((op, param_values))
}

fn unpack_standard_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let op = if let Some(std_instruction) = deserialize_standard_instruction(instruction) {
        // TODO: can we avoid this call? {
        PackedOperation::from_standard_instruction(std_instruction)
    } else {
        return Err(QpyError::InvalidInstruction(format!(
            "Unrecognized standard gate {}",
            instruction.gate_class_name
        )));
    };
    let param_values = get_instruction_values(instruction, qpy_data, Endian::Little)?;
    Ok((op, param_values))
}

fn unpack_pauli_product_measurement(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    if instruction.params.len() != 3 {
        return Err(QpyError::InvalidParameter(
            "Pauli Product Measurement should have exactly 3 parameters".to_string(),
        ));
    }
    let z_values = unpack_generic_value(&instruction.params[0], qpy_data, Endian::Big)?;
    let z: Vec<bool> = z_values.to_boolean_vec().ok_or_else(|| {
        QpyError::InvalidParameter(format!(
            "Pauli product measurement z parameter should be a boolean or integer vector, but got {:?}",
            z_values
        ))
    })?;

    let x_values = unpack_generic_value(&instruction.params[1], qpy_data, Endian::Big)?;
    let x: Vec<bool> = x_values.to_boolean_vec().ok_or_else(|| {
        QpyError::InvalidParameter(format!(
            "Pauli product measurement x parameter should be a boolean or integer vector, but got {:?}",
            x_values
        ))
    })?;
    let neg_value = unpack_generic_value(&instruction.params[2], qpy_data, Endian::Big)?;
    let neg = match neg_value {
        GenericValue::NumpyObject(bytes) => {
            let npy = NpyFile::new(Cursor::new(&bytes.0))?;
            let values: Vec<i64> = npy.into_vec()?;
            let value = *values.first().ok_or_else(|| {
                QpyError::InvalidParameter(format!(
                    "Pauli product measurement phase parameter should be a is invalid, got {:?}",
                    values
                ))
            })?;
            value != 0
        }
        _ => neg_value.as_typed::<bool>().ok_or_else(|| {
                QpyError::InvalidParameter(format!(
                    "Pauli product measurement neg parameter should be a boolean or integer, but got {:?}",
                    neg_value
                ))
            })?
    };
    let ppm = PauliProductMeasurement { z, x, neg };
    let pbc = Box::new(PauliBased::PauliProductMeasurement(ppm));
    let op = PackedOperation::from_pauli_based(pbc);
    let param_values = Vec::new(); // ppm has no "regular" params; the instruction params were used to reconstruct it
    Ok((op, param_values))
}

fn unpack_pauli_product_rotation(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    if instruction.params.len() != 3 {
        return Err(QpyError::InvalidParameter(
            "No angle for pauli product rotation".to_string(),
        ));
    }
    let z_values = unpack_generic_value(&instruction.params[0], qpy_data, Endian::Big)?;
    let z = z_values.to_boolean_vec().ok_or_else(|| {
        QpyError::InvalidParameter(
            "Pauli product rotation z parameter should be a boolean vector".to_string(),
        )
    })?;
    let x_values = unpack_generic_value(&instruction.params[1], qpy_data, Endian::Big)?;
    let x = x_values.to_boolean_vec().ok_or_else(|| {
        QpyError::InvalidParameter(
            "Pauli product rotation x parameter should be a boolean vector".to_string(),
        )
    })?;
    let angle_value = unpack_generic_value(&instruction.params[2], qpy_data, Endian::Little)?;
    let angle = generic_value_to_param(&angle_value)?;
    let rotation = PauliProductRotation { z, x, angle };
    let pbc = Box::new(PauliBased::PauliProductRotation(rotation));
    let op = PackedOperation::from_pauli_based(pbc);
    let param_values = vec![angle_value];
    Ok((op, param_values))
}

fn unpack_unitary(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let GenericValue::NumpyObject(bytes) =
        unpack_generic_value(&instruction.params[0], qpy_data, Endian::Little)?
    else {
        return Err(QpyError::InvalidParameter(
            "No matrix for unitary op".to_string(),
        ));
    };
    let npy = NpyFile::new(Cursor::new(bytes.0))?;
    let shape = npy.shape().to_vec();

    if shape.len() != 2 {
        return Err(QpyError::InvalidParameter(
            "Invalid matrix for unitary op".to_string(),
        ));
    }

    let rows = shape[0] as usize;
    let cols = shape[1] as usize;
    let order = npy.order();

    let data: Vec<Complex64> = npy.into_vec()?;

    let true_shape = (rows, cols).set_f(order == npyz::Order::Fortran);
    let matrix = Array2::from_shape_vec(true_shape, data)
        .map_err(|_| QpyError::InvalidParameter("Invalid matrix for unitary op".to_string()))?;
    let array = ArrayType::NDArray(matrix);
    let unitary = UnitaryGate { array };
    let op = PackedOperation::from_unitary(Box::new(unitary));
    let param_values = Vec::new();
    Ok((op, param_values))
}

fn unpack_control_flow(
    instruction: &formats::CircuitInstructionV2Pack,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let mut param_values: Vec<GenericValue> = Vec::new(); // Params for control structures hold the control flow blocks
    // Convert Python class names (e.g., "IfElseOp") to snake_case (e.g., "if_else")
    // for backwards compatibility with old QPY versions
    let gate_class_name = instruction.gate_class_name.as_str();
    let normalized_name = match gate_class_name {
        "IfElseOp" => "if_else",
        "WhileLoopOp" => "while_loop",
        "ForLoopOp" => "for_loop",
        "SwitchCaseOp" => "switch_case",
        "BreakLoopOp" => "break_loop",
        "ContinueLoopOp" => "continue_loop",
        _ => gate_class_name,
    };

    let control_flow_name = ControlFlowType::from_str(normalized_name)
        .map_err(|_| QpyError::InvalidInstruction("Unable to find control flow".to_string()))?;
    let control_flow = match control_flow_name {
        ControlFlowType::Box => {
            // we need specialized handling for the params here, since the first param is duration
            // which sadly shares the 't' key with tuple, so we can't deserialize using the general `unpack_generic_value`
            param_values = instruction
                .params
                .iter()
                .skip(1)
                .map(|param| unpack_generic_value(param, qpy_data, Endian::Little))
                .collect::<Result<_, QpyError>>()?;
            let duration_value = if let Some(duration_pack) = instruction.params.first() {
                unpack_duration_value(duration_pack, qpy_data)?
            } else {
                return Err(QpyError::MissingData(
                    "Box control flow instruction missing parameters".to_string(),
                ));
            };
            let duration = match duration_value {
                GenericValue::Duration(duration) => Some(BoxDuration::Duration(duration)),
                GenericValue::Expression(exp) => Some(BoxDuration::Expr(exp.clone())),
                _ => None,
            };
            let annotations = unpack_annotations(&instruction.annotations, qpy_data)?;
            ControlFlow::Box {
                duration,
                annotations,
            }
        }
        ControlFlowType::BreakLoop => ControlFlow::BreakLoop,
        ControlFlowType::ContinueLoop => ControlFlow::ContinueLoop,
        ControlFlowType::ForLoop => {
            let mut instruction_values =
                get_instruction_values(instruction, qpy_data, Endian::Big)?;
            param_values = instruction_values.split_off(2);
            let [GenericValue::Circuit(circuit)] = param_values.as_slice() else {
                return Err(QpyError::DeserializationError(
                    "for loops must have a single quantum-circuit body".to_owned(),
                ));
            };
            let mut iter = instruction_values.into_iter();
            let (mut collection_value_pack, loop_param_value_pack) =
                iter.next().zip(iter.next()).ok_or(QpyError::MissingData(
                    "For loop instruction missing some of its parameters".to_string(),
                ))?;
            if gate_class_name == "ForLoopOp" {
                // old style params for loop were stored as little endian
                collection_value_pack = collection_value_pack.as_le();
            }
            let collection = unpack_for_collection(&collection_value_pack)?;
            let loop_param = match loop_param_value_pack {
                GenericValue::ParameterExpressionSymbol(symbol) => {
                    Some(LoopParam::Parameter(Arc::unwrap_or_clone(symbol)))
                }
                GenericValue::Null => Python::attach(|py| -> Result<_, QpyError> {
                    // When writing for loops, we serialise a `Var` loop parameter in the
                    // instruction's parameters field as if it were null, because the `Var` isn't
                    // part of the containing circuit.  Instead, we re-infer its existence from the
                    // `input` variables of the body circuit.
                    let data = circuit
                        .bind(py)
                        .getattr("_data")?
                        .cast_into::<PyCircuitData>()
                        .map_err(PyErr::from)?
                        .borrow();
                    let mut vars = data.vars_stretches_view().iter_vars(VarType::Input);
                    let Some(v) = vars.next() else {
                        // No input vars, so nothing to infer; this is a legacy-type body.
                        return Ok(None);
                    };
                    match vars.next() {
                        Some(_) => Err(QpyError::DeserializationError(
                            "for loop bodies must have at most one input variable".to_owned(),
                        )),
                        None => Ok(Some(LoopParam::Variable(v.clone()))),
                    }
                })?,
                other => {
                    return Err(QpyError::InvalidValueType {
                        expected: "a parameter or none".to_string(),
                        actual: format!("{other:?}"),
                    });
                }
            };
            ControlFlow::ForLoop {
                collection,
                loop_param,
            }
        }
        ControlFlowType::IfElse => {
            let condition = unpack_condition(&instruction.condition, qpy_data)?
                .ok_or_else(|| QpyError::MissingData("if else condition is missing".to_string()))?;
            param_values = get_instruction_values(instruction, qpy_data, Endian::Big)?;
            ControlFlow::IfElse { condition }
        }
        ControlFlowType::WhileLoop => {
            let condition = unpack_condition(&instruction.condition, qpy_data)?
                .ok_or_else(|| QpyError::MissingData("if else condition is missing".to_string()))?;
            param_values = get_instruction_values(instruction, qpy_data, Endian::Big)?;
            ControlFlow::While { condition }
        }
        ControlFlowType::SwitchCase => {
            let mut instruction_values =
                get_instruction_values(instruction, qpy_data, Endian::Big)?;
            let (target_value, case_label_list) = if instruction_values.len() < 3 {
                // we follow the python way of storing switch params
                // the first param is the target, the next param is the cases specifier
                // the cases specifier is a list of pairs (tuples)
                // the second element in each pair is the subcircuit for this case
                // the first element is the list of the case labels, or a single case label
                // or the special default case label
                let [target_value, cases, ..] = &instruction_values[..] else {
                    return Err(QpyError::MissingData(
                        "Switch case requires at least 2 parameters".to_string(),
                    ));
                };
                let mut case_label_list = Vec::new();
                for case in cases.as_slice().ok_or(QpyError::InvalidInstruction(
                    "bad parameters for switch statement".to_string(),
                ))? {
                    let [case_labels, case_circuit, ..] =
                        &case.as_slice().ok_or(QpyError::InvalidInstruction(
                            "bad parameters for switch statement".to_string(),
                        ))?
                    else {
                        return Err(QpyError::InvalidInstruction(
                            "bad parameters for switch statement".to_string(),
                        ));
                    };
                    param_values.push(case_circuit.clone());
                    case_label_list.push(case_labels.clone());
                }
                (target_value, case_label_list)
            } else {
                param_values = instruction_values.split_off(3);
                let [target_value, label_spec_value, ..] = &instruction_values[..] else {
                    return Err(QpyError::MissingData(
                        "Switch case requires at least 3 parameters".to_string(),
                    ));
                };
                (
                    target_value,
                    label_spec_value.to_vec().ok_or_else(|| {
                        QpyError::InvalidInstruction(
                            "bad parameters for switch statement".to_string(),
                        )
                    })?,
                )
            };
            let target = match target_value.clone() {
                GenericValue::Expression(exp) => Ok(SwitchTarget::Expr(exp)),
                GenericValue::Register(ParamRegisterValue::Register(reg)) => {
                    Ok(SwitchTarget::Register(reg))
                }
                GenericValue::Register(ParamRegisterValue::ShareableClbit(clbit)) => {
                    Ok(SwitchTarget::Bit(clbit))
                }
                _ => Err(QpyError::InvalidInstruction(
                    "could not identify switch case target".to_string(),
                )),
            }?;

            // now split the zipped cases: move the circuits to params, keep the labels for further processing
            let mut label_spec = Vec::new();
            for case_labels in case_label_list {
                // label spec handling
                let GenericValue::Tuple(label_spec_element_tuple) = case_labels else {
                    return Err(QpyError::InvalidInstruction(
                        "could not identify switch case label spec".to_string(),
                    ));
                };
                let label_spec_element = label_spec_element_tuple
                    .iter()
                    .map(|label_spec_element| match label_spec_element.as_le() {
                        GenericValue::CaseDefault => Ok(CaseSpecifier::Default),
                        GenericValue::BigInt(value) => Ok(CaseSpecifier::Uint(value.clone())),
                        GenericValue::Int64(value) => {
                            Ok(CaseSpecifier::Uint(BigUint::from(value as u64)))
                        }
                        _ => Err(QpyError::InvalidInstruction(
                            "could not identify switch case label spec".to_string(),
                        )),
                    })
                    .collect::<Result<_, QpyError>>()?;
                label_spec.push(label_spec_element);
            }
            ControlFlow::Switch {
                target,
                label_spec,
                cases: param_values.len() as u32,
            }
        }
    };
    let num_qubits = instruction.num_qargs;
    let num_clbits = instruction.num_cargs;
    let control_flow_instruction = ControlFlowInstruction {
        control_flow,
        num_qubits,
        num_clbits,
    };
    let op = PackedOperation::from_control_flow(Box::new(control_flow_instruction));
    Ok((op, param_values))
}

// This method handles all the non-standard, non-custom gates which have no rust-space implementation
fn unpack_py_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
    label: Option<&String>,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let name = instruction.gate_class_name.clone();
    let mut instruction_values = get_instruction_values(instruction, qpy_data, Endian::Little)?;
    Python::attach(|py| -> Result<_, QpyError> {
        let mut py_params: Vec<Bound<PyAny>> = instruction_values
            .iter()
            .map(|value| -> Result<_, QpyError> {
                generic_value_to_param(value)?
                    .into_pyobject(py)
                    .map_err(QpyError::from)
            })
            .collect::<Result<_, QpyError>>()?;
        let gate_class = get_python_gate_class(py, &instruction.gate_class_name)?;
        // some gates need special treatment for their parameters prior to python-space initialization
        let mut gate_object = match name.as_str() {
            "Initialize" | "StatePreparation" => {
                if py_params[0].is_instance_of::<PyString>() {
                    // the params are the labels of the initial state
                    let label = py_params
                        .iter()
                        .map(|param| param.extract())
                        .collect::<PyResult<Vec<String>>>()?
                        .join("");
                    gate_class.call1((label,))?
                } else if py_params.len() == 1 {
                    // the params is the integer indicating which qubits to initialize
                    let real_param: f64 = py_params[0].getattr("real")?.extract()?;
                    let qubits_to_initialize = real_param as u32;
                    gate_class.call1((qubits_to_initialize, instruction.num_qargs))?
                } else {
                    // the params represent a list of complex amplitudes
                    gate_class.call1((py_params,))?
                }
            }
            "QFTGate" => {
                let mut args: Vec<Py<PyAny>> = vec![instruction.num_qargs.into_py_any(py)?];
                for param in py_params {
                    args.push(param.unbind());
                }
                gate_class.call1(PyTuple::new(py, args)?)?
            }

            "UCRXGate" | "UCRYGate" | "UCRZGate" | "DiagonalGate" => {
                gate_class.call1((py_params,))?
            }
            "MCPhaseGate" | "MCU1Gate" | "MCXGrayCode" | "MCXGate" | "MCXRecursive"
            | "MCXVChain" => {
                let mut args: Vec<Py<PyAny>> = Vec::new();
                for param in py_params {
                    args.push(param.unbind());
                }
                args.push(instruction.num_ctrl_qubits.into_py_any(py)?);
                gate_class.call1(PyTuple::new(py, args)?)?
            }
            "IfElseOp" | "WhileLoopOp" => {
                let condition =
                    unpack_condition(&instruction.condition, qpy_data)?.ok_or_else(|| {
                        QpyError::MissingData(
                            "This control flow gate requires a condition parameter".to_string(),
                        )
                    })?;
                let py_condition = condition.into_py_any(py)?;
                let mut args = vec![py_condition];
                for param in py_params {
                    args.push(param.unbind());
                }
                // in the case if IfElseOp with Null else body, retaining it would confuse the heuristic determining
                // whether parameter are blocks or true params; we can simply dump it.
                instruction_values.retain(|value| !matches!(value, GenericValue::Null));
                gate_class.call1(PyTuple::new(py, args)?)?
            }
            "BoxOp" => {
                if py_params.len() < 2 {
                    return Err(QpyError::InvalidParameter(format!(
                        "BoxOp instruction has only {:?} params; should have at least 2",
                        py_params.len()
                    )));
                }
                let unit = py_params.pop().ok_or_else(|| {
                    QpyError::InvalidParameter("BoxOp missing unit parameter".to_string())
                })?;
                let duration = py_params.pop().ok_or_else(|| {
                    QpyError::InvalidParameter("BoxOp missing duration parameter".to_string())
                })?;
                let annotations = match &instruction.annotations {
                    Some(annotation_pack) => annotation_pack
                        .annotations
                        .iter()
                        .map(|annotation| {
                            qpy_data
                                .annotation_handler
                                .load(annotation.namespace_index, annotation.payload.clone())
                        })
                        .collect::<Result<_, QpyError>>()?,
                    None => Vec::new(),
                }
                .into_pyarray(py)
                .into_any();
                let kwargs = [
                    ("unit", unit),
                    ("duration", duration),
                    ("annotations", annotations),
                ]
                .into_py_dict(py)?;
                let args = PyTuple::new(py, &py_params)?;
                // we used the params to construct the box; they should not be retained as params except the subcircuit
                instruction_values.retain(|value| matches!(value, GenericValue::Circuit(_)));
                gate_class.call(args, Some(&kwargs))?
            }
            "BreakLoopOp" | "ContinueLoopOp" => {
                let mut qubit_count = 0;
                let mut clbit_count = 0;
                for arg in &instruction.bit_data {
                    match arg.bit_type {
                        BitType::Qubit => qubit_count += 1,
                        BitType::Clbit => clbit_count += 1,
                    };
                }
                let args = (qubit_count, clbit_count);
                gate_class.call1(args)?
            }
            _ => {
                let args = PyTuple::new(py, &py_params)?;
                if name.as_str() == "ForLoopOp" {
                    // we used the params to construct the loop; they should not be retained as params except the subcircuit
                    instruction_values.retain(|value| matches!(value, GenericValue::Circuit(_)));
                }
                gate_class.call1(args)?
            }
        };
        if let Some(label_text) = label {
            if !gate_object.hasattr("label")? || gate_object.getattr("label")?.is_none() {
                gate_object.setattr("label", label_text.as_str())?;
            }
        }
        if gate_class
            .cast_into::<PyType>()
            .map_err(|_| QpyError::InvalidPythonType {
                python_type: "PyType".to_string(),
                name: "gate_class".to_string(),
            })?
            .is_subclass(imports::CONTROLLED_GATE.get_bound(py))?
            && (gate_object.getattr("num_ctrl_qubits")?.extract::<u32>()?
                != instruction.num_ctrl_qubits
                || gate_object.getattr("ctrl_state")?.extract::<u32>()? != instruction.ctrl_state)
        {
            gate_object = gate_object.call_method0("to_mutable")?;
            gate_object.setattr("num_ctrl_qubits", instruction.num_ctrl_qubits)?;
            gate_object.setattr("ctrl_state", instruction.ctrl_state)?;
        }

        let op_parts = gate_object.extract::<OperationFromPython<CircuitData>>()?;
        Ok((op_parts.operation, instruction_values))
    })
}

fn unpack_custom_instruction(
    instruction: &formats::CircuitInstructionV2Pack,
    label: Option<&String>,
    qpy_data: &mut QPYReadData,
    custom_instructions_map: &HashMap<String, CustomCircuitInstructionData>,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let name = instruction.gate_class_name.clone();
    let custom_instruction = custom_instructions_map.get(&name).ok_or_else(|| {
        QpyError::MissingData("Custom instruction data not found for {name}".to_string())
    })?;
    let instruction_values = get_instruction_values(instruction, qpy_data, Endian::Little)?;
    Python::attach(|py| -> Result<_, QpyError> {
        let py_params: Vec<Bound<PyAny>> = instruction_values
            .iter()
            .map(|value| -> Result<_, QpyError> {
                generic_value_to_param(value)?
                    .into_pyobject(py)
                    .map_err(QpyError::from)
            })
            .collect::<Result<_, QpyError>>()?;
        // TODO: should have "if version >= 11" check here once we introduce versioning to rust
        let mut gate_class_name = match instruction.gate_class_name.rfind('_') {
            Some(pos) => &instruction.gate_class_name[..pos],
            None => &instruction.gate_class_name,
        };
        let inst_obj = match custom_instruction.gate_type {
            CircuitInstructionType::Gate => {
                let gate_object = imports::GATE.get_bound(py).call1((
                    &gate_class_name,
                    custom_instruction.num_qubits,
                    py_params,
                ))?;
                if let Some(definition) = &custom_instruction.definition_circuit {
                    gate_object.setattr("definition", definition)?;
                }
                if let Some(label_string) = label {
                    gate_object.setattr("label", label_string.as_str())?;
                }
                gate_object.unbind()
            }
            CircuitInstructionType::Instruction => {
                let instruction_object = imports::INSTRUCTION.get_bound(py).call1((
                    &gate_class_name,
                    custom_instruction.num_qubits,
                    custom_instruction.num_clbits,
                    py_params,
                ))?;
                if let Some(definition) = &custom_instruction.definition_circuit {
                    instruction_object.setattr("definition", definition)?;
                }
                if let Some(label_string) = label {
                    instruction_object.setattr("label", label_string.as_str())?;
                }
                instruction_object.unbind()
            }
            CircuitInstructionType::PauliEvolutionGate => {
                if let Some(definition) = &custom_instruction.definition_circuit {
                    let inst = definition.clone();
                    if let Some(label_string) = label {
                        inst.setattr(py, "label", label_string.as_str())?;
                    }
                    inst
                } else {
                    return Err(QpyError::MissingData(
                        "Pauli Evolution Gate missing definition".to_string(),
                    ));
                }
            }
            CircuitInstructionType::ControlledGate => {
                let packed_base_gate =
                    deserialize_with_args::<formats::CircuitInstructionV2Pack, (bool,)>(
                        &custom_instruction.base_gate_raw,
                        (false,),
                    )?
                    .0;
                let base_gate =
                    unpack_instruction(&packed_base_gate, custom_instructions_map, qpy_data)?;
                // If open controls, we need to discard the control suffix when setting the name.
                if instruction.ctrl_state < (1u32 << instruction.num_ctrl_qubits) - 1 {
                    gate_class_name = match gate_class_name.rfind('_') {
                        Some(pos) => &gate_class_name[..pos],
                        None => gate_class_name,
                    };
                }
                let params = qpy_data
                    .circuit_data
                    .unpack_blocks_to_circuit_parameters(base_gate.params.as_deref());
                let py_base_gate = create_py_op(
                    py,
                    base_gate.op.view(),
                    params,
                    base_gate.label.as_deref().map(String::as_str),
                )?;
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "num_ctrl_qubits"), instruction.num_ctrl_qubits)?;
                kwargs.set_item(intern!(py, "ctrl_state"), instruction.ctrl_state)?;
                kwargs.set_item(intern!(py, "base_gate"), py_base_gate)?;

                let controlled_gate_object = imports::CONTROLLED_GATE.get_bound(py).call(
                    (&gate_class_name, custom_instruction.num_qubits, py_params),
                    Some(&kwargs),
                )?;
                if let Some(definition) = &custom_instruction.definition_circuit {
                    controlled_gate_object.setattr("definition", definition)?;
                }
                controlled_gate_object.unbind()
            }
            CircuitInstructionType::AnnotatedOperation => {
                let packed_base_gate =
                    deserialize_with_args::<formats::CircuitInstructionV2Pack, (bool,)>(
                        &custom_instruction.base_gate_raw,
                        (false,),
                    )?
                    .0;
                let base_gate =
                    unpack_instruction(&packed_base_gate, custom_instructions_map, qpy_data)?;
                let params = qpy_data
                    .circuit_data
                    .unpack_blocks_to_circuit_parameters(base_gate.params.as_deref());
                let py_base_gate = create_py_op(
                    py,
                    base_gate.op.view(),
                    params,
                    base_gate.label.as_deref().map(String::as_str),
                )?;
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "base_op"), py_base_gate)?;
                kwargs.set_item(intern!(py, "modifiers"), py_params)?;
                imports::ANNOTATED_OPERATION
                    .get_bound(py)
                    .call((), Some(&kwargs))?
                    .unbind()
            }
        };
        let op = inst_obj
            .extract::<OperationFromPython<CircuitData>>(py)?
            .operation;
        Ok((op, instruction_values))
    })
}

fn deserialize_metadata(
    py: Python,
    metadata_bytes: &Bytes,
    metadata_deserializer: Option<&Bound<PyAny>>,
) -> Result<Py<PyAny>, QpyError> {
    let json = py.import("json")?;
    let kwargs: Bound<'_, PyDict> = PyDict::new(py);
    kwargs.set_item("cls", metadata_deserializer)?;
    let metadata_string = PyString::new(py, metadata_bytes.try_into()?);
    Ok(json
        .call_method("loads", (metadata_string,), Some(&kwargs))?
        .unbind())
}

fn unpack_layout<'py>(
    py: Python<'py>,
    layout: &formats::LayoutV2Pack,
    circuit_data: &PyCircuitData,
) -> Result<Option<Bound<'py, PyAny>>, QpyError> {
    match layout.exists {
        0 => Ok(None),
        _ => Ok(Some(unpack_transpile_layout(py, layout, circuit_data)?)),
    }
}

fn unpack_transpile_layout<'py>(
    py: Python<'py>,
    layout: &formats::LayoutV2Pack,
    circuit_data: &PyCircuitData,
) -> Result<Bound<'py, PyAny>, QpyError> {
    let mut initial_layout = py.None();
    let mut input_qubit_mapping = py.None();
    let mut final_layout = py.None();

    let mut extra_register_map = HashMap::new();
    let mut existing_register_map = HashMap::new();
    for packed_register in &layout.extra_registers {
        if packed_register.register_type == RegisterType::Qreg {
            let register = QuantumRegister::new_owning(
                packed_register.name.clone(),
                packed_register.bit_indices.len() as u32,
            );
            extra_register_map.insert(packed_register.name.as_str(), register);
        }
    }
    // add the registers from the circuit, to streamline the search phase
    for qreg in circuit_data.qregs() {
        existing_register_map.insert(qreg.name(), qreg);
    }
    let initial_layout_virtual_bits = PyList::new(py, Vec::<Py<PyAny>>::new())?;
    for virtual_bit in &layout.initial_layout_items {
        let qubit = if let Some(register) =
            extra_register_map.get(virtual_bit.register_name.as_str())
        {
            if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                qubit
            } else {
                ShareableQubit::new_anonymous()
            }
        } else if let Some(register) = existing_register_map.get(virtual_bit.register_name.as_str())
        {
            if let Some(qubit) = register.get(virtual_bit.index_value as usize) {
                qubit
            } else {
                ShareableQubit::new_anonymous()
            }
        } else {
            ShareableQubit::new_anonymous()
        };
        initial_layout_virtual_bits.append(qubit)?;
    }
    if initial_layout_virtual_bits.len() > 0 {
        initial_layout = imports::LAYOUT
            .get_bound(py)
            .call_method1("from_qubit_list", (initial_layout_virtual_bits,))?
            .unbind();
    }

    if layout.input_mapping_size > 0 {
        let input_qubit_mapping_data = PyDict::new(py);
        let physical_bits_object = initial_layout.call_method0(py, "get_physical_bits")?;
        let physical_bits = physical_bits_object.cast_bound::<PyDict>(py).map_err(|_| {
            QpyError::InvalidPythonType {
                python_type: "PyDict".to_string(),
                name: "physical_bits".to_string(),
            }
        })?;
        for (index, bit) in layout.input_mapping_items.iter().enumerate() {
            let physical_bit = physical_bits.get_item(bit)?.ok_or_else(|| {
                QpyError::InvalidBit(format!("Could not get physical bit for bit {:?}", bit))
            })?;
            input_qubit_mapping_data.set_item(physical_bit, index)?;
        }
        input_qubit_mapping = input_qubit_mapping_data.into_py_any(py)?;
    }

    if layout.final_layout_size > 0 {
        let final_layout_dict = PyDict::new(py);
        let py_qubits = circuit_data.py_qubits(py);
        let qubits = py_qubits.bind(py);
        for (index, bit) in layout.final_layout_items.iter().enumerate() {
            let qubit = qubits.get_item(*bit as usize)?;
            final_layout_dict.set_item(qubit, index)?;
        }
        final_layout = imports::LAYOUT
            .get_bound(py)
            .call1((final_layout_dict,))?
            .unbind();
    }
    let transpiled_layout = imports::TRANSPILER_LAYOUT.get_bound(py).call1((
        initial_layout,
        input_qubit_mapping,
        final_layout,
    ))?;
    // TODO: this is for version >= 10
    if layout.input_qubit_count >= 0 {
        transpiled_layout.setattr("_input_qubit_count", layout.input_qubit_count)?;
        transpiled_layout.setattr("_output_qubit_list", circuit_data.py_qubits(py))?;
    }
    Ok(transpiled_layout)
}

fn deserialize_pauli_evolution_gate(
    py: Python,
    data: &Bytes,
    qpy_data: &mut QPYReadData,
) -> Result<Py<PyAny>, QpyError> {
    let json = py.import("json")?;
    let evo_synth_library = py.import("qiskit.synthesis.evolution")?;
    let (packed_data, _) =
        deserialize_with_args::<formats::PauliEvolutionDefPack, (u32,)>(data, (qpy_data.version,))?;
    // operators as stored as a numpy dump that can be loaded into Python's SparsePauliOp.from_list
    let operators: Vec<Py<PyAny>> = packed_data
        .pauli_data
        .iter()
        .map(|elem| match elem {
            formats::PauliDataPack::V17(formats::PauliDataPackV17::SparseObservable(
                sparse_observable_pack,
            )) => {
                let num_qubits = sparse_observable_pack.num_qubits;
                let coeffs = sparse_observable_pack
                    .coeff_data
                    .chunks_exact(2)
                    .map(|c| Complex64::new(c[0], c[1]))
                    .collect();
                let bit_terms = sparse_observable_pack
                    .bitterm_data
                    .iter()
                    .map(|&bitterm| -> Result<_, QpyError> {
                        let reduced_bitterm = u8::try_from(bitterm)?;
                        BitTerm::try_from(reduced_bitterm).map_err(|_| {
                            QpyError::DeserializationError(
                                "Could not read sparse observable data".to_string(),
                            )
                        })
                    })
                    .collect::<Result<_, QpyError>>()?;
                let indices = sparse_observable_pack.inds_data.clone();
                let boundaries = sparse_observable_pack
                    .bounds_data
                    .iter()
                    .map(|&bounds_value| bounds_value as usize)
                    .collect();
                let sparse_observable =
                    SparseObservable::new(num_qubits, coeffs, bit_terms, indices, boundaries)
                        .map_err(|e| {
                            QpyError::DeserializationError(format!(
                                "Failed to create sparse observable: {}",
                                e
                            ))
                        })?;
                Ok(sparse_observable.into_py_any(py)?)
            }
            formats::PauliDataPack::V17(formats::PauliDataPackV17::SparsePauliOp(
                sparse_pauli_op_pack,
            ))
            | formats::PauliDataPack::V16(formats::PauliDataPackV16::SparsePauliOp(
                sparse_pauli_op_pack,
            )) => {
                // formats::PauliDataPack::SparsePauliOp(sparse_pauli_op_pack) => {
                let data = load_value(
                    ValueType::NumpyObject,
                    &sparse_pauli_op_pack.data,
                    qpy_data,
                    Endian::Big,
                )?;
                if let GenericValue::NumpyObject(op_raw_data) = data {
                    let np_array = py_deserialize_numpy_object(&op_raw_data)?;
                    Ok(imports::SPARSE_PAULI_OP
                        .get_bound(py)
                        .call_method1("from_list", (np_array,))?
                        .unbind())
                } else {
                    Err(QpyError::InvalidParameter(
                        "Pauli Evolution Gate needs data list stored as numpy object".to_string(),
                    ))
                }
            }
        })
        .collect::<Result<_, QpyError>>()?;

    let py_operators = if packed_data.standalone_op != 0 {
        operators[0].clone()
    } else {
        PyList::new(py, operators)?.into_py_any(py)?
    };
    // time is of type ParameterValueType = Union[ParameterExpression, float]
    // we don't have a rust PauliEvolutionGate so we'll convert the time to python
    let time = load_value(
        packed_data.time_type,
        &packed_data.time_data,
        qpy_data,
        Endian::Big,
    )?;
    let py_time: Py<PyAny> = match time {
        GenericValue::Float64(value) => value.into_py_any(py)?,
        GenericValue::ParameterExpression(exp) => exp.as_ref().clone().into_py_any(py)?,
        GenericValue::ParameterExpressionVectorSymbol(symbol)
        | GenericValue::ParameterExpressionSymbol(symbol) => PyParameter(symbol).into_py_any(py)?,
        _ => return Err(QpyError::InvalidParameter(
            "Pauli Evolution Gate 'time' parameter should be either float or parameter expression"
                .to_string(),
        )),
    };
    let synth_data = json.call_method1("loads", (packed_data.synth_data,))?;
    let synth_data = synth_data
        .cast::<PyDict>()
        .map_err(|_| QpyError::InvalidPythonType {
            python_type: "PyDict".to_string(),
            name: "synth_data".to_string(),
        })?;
    let synthesis_class_name = synth_data.get_item("class")?.ok_or_else(|| {
        QpyError::MissingData(
            "Could not find synthesis class name for Pauli Evolution Gate".to_string(),
        )
    })?;
    let synthesis_class_settings = synth_data.get_item("settings")?.ok_or_else(|| {
        QpyError::MissingData(
            "Could not find synthesis class settings for Pauli Evolution Gate".to_string(),
        )
    })?;
    let synthesis_class =
        evo_synth_library.getattr(synthesis_class_name.cast::<PyString>().map_err(|_| {
            QpyError::InvalidPythonType {
                python_type: "PyString".to_string(),
                name: "synthesis_class".to_string(),
            }
        })?)?;
    let synthesis_settings_dict =
        synthesis_class_settings
            .cast::<PyDict>()
            .map_err(|_| QpyError::InvalidPythonType {
                python_type: "PyDict".to_string(),
                name: "synthesis_settings_dict".to_string(),
            })?;
    let synthesis = synthesis_class.call((), Some(synthesis_settings_dict))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "time"), py_time)?;
    kwargs.set_item(intern!(py, "synthesis"), synthesis)?;
    Ok(imports::PAULI_EVOLUTION_GATE
        .get_bound(py)
        .call((py_operators,), Some(&kwargs))?
        .unbind())
}

fn read_custom_instructions(
    py: Python,
    packed_circuit: &formats::QPYCircuit,
    qpy_data: &mut QPYReadData,
) -> Result<HashMap<String, CustomCircuitInstructionData>, QpyError> {
    let mut result = HashMap::new();
    for operation in &packed_circuit.custom_instructions.custom_instructions {
        let definition = if operation.custom_definition != 0 {
            if operation.name.starts_with("###PauliEvolutionGate_") {
                Some(deserialize_pauli_evolution_gate(
                    py,
                    &operation.data,
                    qpy_data,
                )?)
            } else {
                Some(unpack_circuit(
                    py,
                    &deserialize_with_args::<QPYCircuit, (u32,)>(
                        &operation.data,
                        (qpy_data.version,),
                    )?
                    .0,
                    qpy_data.version,
                    None,
                    qpy_data.use_symengine,
                    qpy_data.annotation_handler.annotation_factories,
                )?)
            }
        } else {
            None
        };
        let custom_instruction_data = CustomCircuitInstructionData {
            gate_type: operation.gate_type,
            num_qubits: operation.num_qubits,
            num_clbits: operation.num_clbits,
            definition_circuit: definition,
            base_gate_raw: operation.base_gate_raw.clone(),
        };
        result.insert(operation.name.clone(), custom_instruction_data);
    }
    Ok(result)
}
fn add_standalone_vars(
    packed_circuit: &formats::QPYCircuit,
    qpy_data: &mut QPYReadData,
) -> Result<(), QpyError> {
    let mut index: u16 = 0;
    for packed_var in &packed_circuit.standalone_vars {
        let ty = match packed_var.exp_type {
            ExpressionType::Bool => classical::types::Type::Bool,
            ExpressionType::Duration => classical::types::Type::Duration,
            ExpressionType::Float => classical::types::Type::Float,
            ExpressionType::Uint(val) => classical::types::Type::Uint(val),
        };
        let uuid = u128::from_be_bytes(packed_var.uuid_bytes);
        let name = packed_var.name.clone();
        match packed_var.usage {
            ExpressionVarDeclaration::Local => {
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    VarType::Declare,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            ExpressionVarDeclaration::Input => {
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    VarType::Input,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            ExpressionVarDeclaration::Capture => {
                let var = qpy_data.circuit_data.add_var(
                    classical::expr::Var::Standalone { uuid, name, ty },
                    VarType::Capture,
                )?;
                qpy_data.standalone_vars.insert(index, var);
                index += 1;
            }
            ExpressionVarDeclaration::StretchLocal => {
                let stretch = qpy_data.circuit_data.add_stretch(
                    classical::expr::Stretch { uuid, name },
                    StretchType::Declare,
                )?;
                qpy_data.standalone_stretches.insert(index, stretch);
                index += 1;
            }
            ExpressionVarDeclaration::StretchCapture => {
                let stretch = qpy_data.circuit_data.add_stretch(
                    classical::expr::Stretch { uuid, name },
                    StretchType::Capture,
                )?;
                qpy_data.standalone_stretches.insert(index, stretch);
                index += 1;
            }
        }
    }
    Ok(())
}

fn add_registers_and_bits(
    packed_circuit: &formats::QPYCircuit,
    qpy_data: &mut QPYReadData,
) -> Result<(), QpyError> {
    let num_qubits = packed_circuit.header.num_qubits as usize;
    let num_clbits = packed_circuit.header.num_clbits as usize;
    let mut qubits: Vec<Option<ShareableQubit>> = vec![None; num_qubits];
    let mut clbits: Vec<Option<ShareableClbit>> = vec![None; num_clbits];
    let mut qregs = Vec::new();
    let mut cregs = Vec::new();

    // first, create all owning registers and collect their bits
    let mut non_standalone_registers = Vec::new();
    for packed_register in &packed_circuit.header.registers {
        if packed_register.standalone == 0 {
            non_standalone_registers.push(packed_register);
        } else {
            match packed_register.register_type {
                RegisterType::Qreg => {
                    let qreg = QuantumRegister::new_owning(
                        &packed_register.name,
                        packed_register.bit_indices.len() as u32,
                    );
                    for (qubit, &index) in qreg.bits().zip(packed_register.bit_indices.iter()) {
                        if index >= 0 {
                            // index can be -1, indicating this bit is not in the circuit
                            qubits[index as usize] = Some(qubit);
                        }
                    }
                    if packed_register.in_circuit != 0 {
                        qregs.push(qreg);
                    }
                }
                RegisterType::Creg => {
                    let creg = ClassicalRegister::new_owning(
                        &packed_register.name,
                        packed_register.bit_indices.len() as u32,
                    );
                    for (clbit, &index) in creg.bits().zip(packed_register.bit_indices.iter()) {
                        if index >= 0 {
                            // index can be -1, indicating this bit is not in the circuit
                            clbits[index as usize] = Some(clbit);
                        }
                    }
                    if packed_register.in_circuit != 0 {
                        cregs.push(creg);
                    }
                }
            }
        }
    }
    // First pass is done. Now add Anonymous bits that are not part of any register
    let final_qubit_list: Vec<ShareableQubit> = qubits
        .iter()
        .map(|element| match element {
            Some(qubit) => qubit.clone(),
            None => ShareableQubit::new_anonymous(),
        })
        .collect();
    let final_clbit_list: Vec<ShareableClbit> = clbits
        .iter()
        .map(|element| match element {
            Some(clbit) => clbit.clone(),
            None => ShareableClbit::new_anonymous(),
        })
        .collect();

    // We collected owning registers to qregs, cregs and added all remaining bits and can now deal with the non-standalone registers
    for packed_register in non_standalone_registers {
        match packed_register.register_type {
            RegisterType::Qreg => {
                let bits: Vec<ShareableQubit> = packed_register
                    .bit_indices
                    .iter()
                    .filter_map(|&index| {
                        if index >= 0 {
                            Some(final_qubit_list[index as usize].clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let qreg = QuantumRegister::new_alias(Some(packed_register.name.clone()), bits);
                qregs.push(qreg);
            }
            RegisterType::Creg => {
                let bits: Vec<ShareableClbit> = packed_register
                    .bit_indices
                    .iter()
                    .filter_map(|&index| {
                        if index >= 0 {
                            Some(final_clbit_list[index as usize].clone())
                        } else {
                            None
                        }
                    })
                    .collect();
                let creg = ClassicalRegister::new_alias(Some(packed_register.name.clone()), bits);
                cregs.push(creg);
            }
        }
    }
    // now add the bits to the circuit, and then add the registers
    for qubit in final_qubit_list {
        qpy_data.circuit_data.add_qubit(qubit, true)?;
    }
    for clbit in final_clbit_list {
        qpy_data.circuit_data.add_clbit(clbit, true)?;
    }

    for qreg in qregs {
        qpy_data.circuit_data.add_qreg(qreg, true)?;
    }

    for creg in cregs {
        qpy_data.circuit_data.add_creg(creg, true)?;
    }
    Ok(())
}

pub(crate) fn unpack_circuit(
    py: Python,
    packed_circuit: &QPYCircuit,
    version: u32,
    metadata_deserializer: Option<&Bound<PyAny>>,
    use_symengine: bool,
    annotation_factories: &Bound<PyDict>,
) -> Result<Py<PyAny>, QpyError> {
    let instruction_capacity = packed_circuit.instructions.len();
    // create an empty circuit; we'll fill data as we go along
    let mut circuit_data: PyCircuitData =
        CircuitData::with_capacity(0, 0, instruction_capacity, Param::Float(0.0))?.into();
    let mut qpy_data = QPYReadData {
        circuit_data: &mut circuit_data.inner,
        version,
        use_symengine,
        standalone_vars: HashMap::new(),
        standalone_stretches: HashMap::new(),
        vectors: HashMap::new(),
        annotation_handler: AnnotationHandler::new(annotation_factories),
    };
    if let Some(annotation_headers) = &packed_circuit.annotation_headers {
        let annotation_deserializers_data: Vec<(String, Bytes)> = annotation_headers
            .state_headers
            .iter()
            .map(|data| (data.namespace.clone(), data.state.clone()))
            .collect();

        qpy_data
            .annotation_handler
            .load_deserializers(annotation_deserializers_data)?;
    }
    let global_phase = generic_value_to_param(&load_value(
        packed_circuit.header.global_phase_type,
        &packed_circuit.header.global_phase_data,
        &mut qpy_data,
        Endian::Big,
    )?)?;
    qpy_data.circuit_data.set_global_phase_param(global_phase)?;
    add_standalone_vars(packed_circuit, &mut qpy_data)?;
    add_registers_and_bits(packed_circuit, &mut qpy_data)?;
    let custom_instructions = read_custom_instructions(py, packed_circuit, &mut qpy_data)?;
    for instruction in &packed_circuit.instructions {
        let inst = unpack_instruction(instruction, &custom_instructions, &mut qpy_data)?;
        qpy_data.circuit_data.push(inst)?;
    }
    // since we don't have a rust QuantumCircuit, and the metadata and custom layouts are also in python
    // this pythonic part is unavoidable
    let unpacked_layout = unpack_layout(py, &packed_circuit.layout, &circuit_data)?;
    let metadata =
        deserialize_metadata(py, &packed_circuit.header.metadata, metadata_deserializer)?;
    let circuit = imports::QUANTUM_CIRCUIT
        .get_bound(py)
        .call_method1(intern!(py, "_from_circuit_data"), (circuit_data,))?;
    circuit.setattr("metadata", metadata)?;
    circuit.setattr("name", &packed_circuit.header.circuit_name)?;
    if let Some(layout) = unpacked_layout {
        circuit.setattr("_layout", layout)?;
    }
    Ok(circuit.unbind().as_any().clone())
}

#[pyfunction]
#[pyo3(name = "read_circuit")]
#[pyo3(signature = (file_obj, version, metadata_deserializer, use_symengine, annotation_factories))]
pub(crate) fn py_read_circuit(
    py: Python,
    file_obj: &Bound<PyAny>,
    version: u32,
    metadata_deserializer: &Bound<PyAny>,
    use_symengine: bool,
    annotation_factories: &Bound<PyDict>,
) -> Result<Py<PyAny>, QpyError> {
    let pos = file_obj.call_method0("tell")?.extract::<usize>()?;
    let bytes = file_obj.call_method0("read")?;
    let serialized_circuit: &[u8] = bytes
        .cast::<PyBytes>()
        .map_err(|_| QpyError::InvalidPythonType {
            python_type: "PyBytes".to_string(),
            name: "serialized_circuit".to_string(),
        })?
        .as_bytes();
    let (packed_circuit, bytes_read) =
        deserialize_with_args::<formats::QPYCircuit, (u32,)>(serialized_circuit, (version,))?;
    let unpacked_circuit = unpack_circuit(
        py,
        &packed_circuit,
        version,
        Some(metadata_deserializer),
        use_symengine,
        annotation_factories,
    )?;
    file_obj.call_method1("seek", (pos + bytes_read,))?;
    Ok(unpacked_circuit)
}

// handling for non control flow gates with conditionals, for backwards compatability
pub fn wrap_conditional_gate(
    instruction: &formats::CircuitInstructionV2Pack,
    op: PackedOperation,
    cond: Condition,
    qubits: Interned<[Qubit]>,
    clbits: Interned<[Clbit]>,
    params: Option<Box<Parameters<Block>>>,
    qpy_data: &mut QPYReadData,
) -> PyResult<(PackedOperation, Option<Box<Parameters<Block>>>)> {
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::Param;
    use smallvec::SmallVec;

    // Create an IfElseOp wrapping this instruction
    let control_flow = ControlFlow::IfElse { condition: cond };
    let control_flow_instruction = ControlFlowInstruction {
        control_flow,
        num_qubits: instruction.num_qargs,
        num_clbits: instruction.num_cargs,
    };
    let if_else_op = PackedOperation::from_control_flow(Box::new(control_flow_instruction));

    // Get the actual qubit and clbit indices from the interned values
    let qubit_indices = qpy_data.circuit_data.get_qargs(qubits);
    let clbit_indices = qpy_data.circuit_data.get_cargs(clbits);

    // Convert params from Parameters<Block> to SmallVec<[Param; 3]>
    let param_vec: SmallVec<[Param; 3]> = if let Some(params_box) = params {
        match params_box.as_ref() {
            qiskit_circuit::instruction::Parameters::Params(p) => p.clone(),
            qiskit_circuit::instruction::Parameters::Blocks(_) => {
                // If we have blocks, we can't easily convert them to params
                // This shouldn't happen for simple gates with conditions
                SmallVec::new()
            }
        }
    } else {
        SmallVec::new()
    };

    // Create the body circuit using from_packed_operations
    // This avoids manually creating PackedInstruction and pushing it
    let body_data = CircuitData::from_packed_operations(
        instruction.num_qargs,
        instruction.num_cargs,
        std::iter::once(Ok((
            op,
            param_vec,
            qubit_indices.to_vec(),
            clbit_indices.to_vec(),
        ))),
        Param::Float(0.0),
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{:?}", e)))?;

    // Create GenericValue::CircuitData from the CircuitData
    // The instruction_values_to_params function will handle converting this to a Block
    let if_params = vec![GenericValue::CircuitData(Box::new(body_data))];
    let if_params_converted = instruction_values_to_params(if_params, qpy_data)?;

    Ok((if_else_op, if_params_converted))
}
