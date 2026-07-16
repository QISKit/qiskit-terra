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

// Circuit writer module: converts a QuantumCircuit into a qpy file

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.
use binrw::Endian;
use hashbrown::{HashMap, HashSet};
use num_bigint::BigUint;
use num_traits::ToPrimitive;
use numpy::ToPyArray;
use qiskit_util::IndexSet;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};
use qiskit_circuit::bit::{
    ClassicalRegister, PyClbit, PyQubit, QuantumRegister, Register, ShareableClbit, ShareableQubit,
};
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::{CircuitInstruction, OperationFromPython};
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::duration::Duration;
use qiskit_circuit::imports;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    BoxDuration, CaseSpecifier, Condition, ControlFlow, ControlFlowInstruction, LoopParam,
    Operation, OperationRef, Param, PauliProductMeasurement, PauliProductRotation, PyInstruction,
    PyOpKind, StandardGate, StandardInstruction, SwitchTarget, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::error::QpyError;
use crate::formats;
use crate::params::pack_param_obj;
use crate::py_methods::{
    PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME, PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME,
    UNITARY_GATE_CLASS_NAME, gate_class_name, py_pack_param, py_pack_pauli_evolution_gate,
    recognize_custom_operation, serialize_metadata,
};
use crate::value::{
    BitType, CircuitInstructionType, ExpressionVarDeclaration, GenericValue, ParamRegisterValue,
    QPYWriteData, RegisterType, get_circuit_type_key, pack_for_collection, pack_generic_value,
    pack_standalone_var, pack_stretch, serialize, serialize_param_register_value,
};

use qiskit_circuit::var_stretch_container::{StretchType, VarType};

/// packing the qubits and clbits of a specific instruction into CircuitInstructionArgPack
fn get_packed_bit_list(
    inst: &PackedInstruction,
    circuit_data: &CircuitData,
) -> Vec<formats::CircuitInstructionArgPack> {
    let mut result: Vec<formats::CircuitInstructionArgPack> = Vec::new();
    for qubit in circuit_data.get_qargs(inst.qubits) {
        result.push(formats::CircuitInstructionArgPack {
            bit_type: BitType::Qubit,
            index: (qubit.index() as u32),
        });
    }
    for clbit in circuit_data.get_cargs(inst.clbits) {
        result.push(formats::CircuitInstructionArgPack {
            bit_type: BitType::Clbit,
            index: (clbit.index() as u32),
        });
    }
    result
}

/// pack all the instructions in the circuit, returning both the packed instructions
/// and the dictionary of custom operations generated in the process
fn pack_instructions(
    qpy_data: &mut QPYWriteData,
) -> Result<
    (
        Vec<formats::CircuitInstructionV2Pack>,
        HashMap<String, PackedOperation>,
    ),
    QpyError,
> {
    let mut custom_operations = HashMap::new();
    let mut custom_new_operations = Vec::new();
    let instructions = qpy_data.circuit_data.data().to_vec();
    Ok((
        instructions
            .iter()
            .map(|instruction| {
                pack_instruction(
                    instruction,
                    &mut custom_operations,
                    &mut custom_new_operations,
                    qpy_data,
                )
            })
            .collect::<Result<_, QpyError>>()?,
        custom_operations,
    ))
}

pub(crate) fn pack_annotations(
    annotations: &[Py<PyAny>],
    qpy_data: &mut QPYWriteData,
) -> PyResult<Option<formats::InstructionsAnnotationPack>> {
    let annotations_pack: Vec<formats::InstructionAnnotationPack> = annotations
        .iter()
        .map(|annotation| {
            let (namespace_index, payload) = qpy_data.annotation_handler.serialize(annotation)?;
            Ok(formats::InstructionAnnotationPack {
                namespace_index,
                payload,
            })
        })
        .collect::<Result<_, QpyError>>()?;
    if !annotations_pack.is_empty() {
        Ok(Some(formats::InstructionsAnnotationPack {
            annotations: annotations_pack,
        }))
    } else {
        Ok(None)
    }
}

fn pack_condition(
    condition: Condition,
    qpy_data: &QPYWriteData,
) -> Result<formats::ConditionPack, QpyError> {
    match condition {
        Condition::Expr(exp) => {
            let expression_pack = pack_generic_value(&GenericValue::Expression(exp), qpy_data)?;
            Ok(formats::ConditionPack {
                register_size: 0u16,
                value: 0i64,
                data: formats::ConditionData::Expression(expression_pack),
            })
        }
        Condition::Bit(clbit, target_bool_value) => {
            let bytes = serialize_param_register_value(
                &ParamRegisterValue::ShareableClbit(clbit),
                qpy_data,
            )?;
            let register_size = bytes.len() as u16;
            let data = formats::ConditionData::Register(bytes);
            Ok(formats::ConditionPack {
                register_size,
                value: target_bool_value as i64,
                data,
            })
        }
        Condition::Register(reg, target_value) => {
            let bytes =
                serialize_param_register_value(&ParamRegisterValue::Register(reg), qpy_data)?;
            let register_size = bytes.len() as u16;
            let data = formats::ConditionData::Register(bytes);
            if target_value > BigUint::from(i64::MAX as u64) {
                return Err(QpyError::InvalidInstruction(format!(
                    "Register condition value {} exceeds i64::MAX and cannot be serialized in QPY format",
                    target_value
                )));
            }
            // Handle zero case: iter_u64_digits() returns empty iterator for 0
            let low_digits = target_value.iter_u64_digits().next().unwrap_or(0) as i64;
            Ok(formats::ConditionPack {
                register_size,
                value: low_digits,
                data,
            })
        }
    }
}

// straightforward packing of the instruction parameters for the general cases
// where no additional handling is required
fn pack_instruction_params(
    inst: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<Vec<formats::GenericDataPack>, QpyError> {
    inst.params_view()
        .iter()
        .map(|x| pack_param_obj(x, qpy_data, Endian::Little))
        .collect::<Result<_, QpyError>>()
}

/// for control flow instructions we need to get blocks instead of params
fn pack_instruction_blocks(
    inst: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<Vec<formats::GenericDataPack>, QpyError> {
    let blocks = qpy_data
        .circuit_data
        .unpack_blocks_to_circuit_parameters(inst.params.as_deref())
        .ok_or_else(|| {
            QpyError::ConversionError("Could not extract blocks from instruction".to_string())
        })?;
    match blocks {
        Parameters::Params(_) => Err(QpyError::ConversionError(
            "Instruction has params but expected blocks".to_string(),
        )),
        Parameters::Blocks(blocks) => Python::attach(|py| -> Result<_, QpyError> {
            blocks
                .iter()
                .map(|block: &CircuitData| -> Result<_, QpyError> {
                    // we explicitly name the block "unnamed" because otherwise it will be assigned a serial number name (e.g. "circuit-45")
                    // which would result in inconsistent results, e.g. when packing the same circuit twice on the same run
                    let py_block: PyCircuitData = block.clone().into();
                    let circuit = py_block.into_py_quantum_circuit(py)?;
                    py_pack_param(&circuit, qpy_data, Endian::Little)
                })
                .collect::<Result<_, QpyError>>()
        }),
    }
}

/// extracts instruction blocks from an instruction into a vector of GenericValues
fn extract_instruction_blocks(
    inst: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Vec<GenericValue> {
    inst.blocks_view()
        .iter()
        .filter_map(|&block_id| qpy_data.circuit_data.blocks().get(block_id))
        .map(|block| GenericValue::CircuitData(Box::new(block.clone())))
        .collect()
}

/// packs one specific instruction into CircuitInstructionV2Pack, creating a new custom operation if needed
fn pack_instruction(
    instruction: &PackedInstruction,
    custom_operations: &mut HashMap<String, PackedOperation>,
    new_custom_operations: &mut Vec<String>,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    let mut instruction_pack = match instruction.op.view() {
        OperationRef::StandardGate(gate) => pack_standard_gate(&gate, instruction, qpy_data)?,
        OperationRef::StandardInstruction(inst) => {
            pack_standard_instruction(&inst, instruction, qpy_data)?
        }
        OperationRef::PauliProductMeasurement(ppm) => {
            pack_pauli_product_measurement(ppm, instruction, qpy_data)?
        }
        OperationRef::PauliProductRotation(rotation) => {
            pack_pauli_product_rotation(rotation, instruction, qpy_data)?
        }
        OperationRef::Unitary(unitary_gate) => pack_unitary_gate(unitary_gate, qpy_data)?,
        OperationRef::PyCustom(custom) => pack_py_instruction(custom, instruction, qpy_data)?,
        OperationRef::ControlFlow(control_flow_inst) => {
            pack_control_flow_inst(control_flow_inst, instruction, qpy_data)?
        }
        OperationRef::CustomOperation(_) => {
            return Err(QpyError::SerializationError(
                "compiled custom operations are not yet supported in QPY".to_owned(),
            ));
        }
    };

    // common data extraction for all instruction types
    if let Some(label) = instruction.label.as_deref() {
        instruction_pack.label = label.clone();
    }
    instruction_pack.bit_data = get_packed_bit_list(instruction, qpy_data.circuit_data);
    if let Some(new_name) =
        recognize_custom_operation(&instruction.op, &gate_class_name(&instruction.op)?)?
    {
        instruction_pack.gate_class_name = new_name.clone();
        new_custom_operations.push(new_name.clone());
        custom_operations.insert(new_name.clone(), instruction.op.clone());
    };
    Ok(instruction_pack)
}

fn pack_standard_gate(
    gate: &StandardGate,
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    let params = pack_instruction_params(instruction, qpy_data)?;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: gate.num_qubits(),
        num_cargs: gate.num_clbits(),
        extras_key: 0,
        num_ctrl_qubits: gate.num_ctrl_qubits(),
        ctrl_state: (1 << gate.num_ctrl_qubits()) - 1, // default control state: all 1s
        gate_class_name: imports::get_std_gate_class_name(gate),
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

fn pack_standard_instruction(
    inst: &StandardInstruction,
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    let params = pack_instruction_params(instruction, qpy_data)?;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: inst.num_qubits(),
        num_cargs: inst.num_clbits(),
        extras_key: 0,
        num_ctrl_qubits: 0, // standard instructions have no control qubits
        ctrl_state: 0,
        gate_class_name: standard_instruction_class_name(inst).to_string(),
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

pub fn standard_instruction_class_name(inst: &StandardInstruction) -> &str {
    match inst {
        StandardInstruction::Barrier(_) => "Barrier",
        StandardInstruction::Delay(_) => "Delay",
        StandardInstruction::Measure => "Measure",
        StandardInstruction::Reset => "Reset",
    }
}

fn pack_pauli_product_measurement(
    ppm: &PauliProductMeasurement,
    instruction: &PackedInstruction,
    qpy_data: &QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    // since we won't recreate this gate via python, it's not important to verify the python name is identical to the one we use here
    // so we simply hard-code it instead of going through python
    let gate_class_name = String::from(PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME);
    let params = Python::attach(|py| -> Result<_, QpyError> {
        let z_array = ppm.z.to_pyarray(py);
        let x_array = ppm.x.to_pyarray(py);
        // Pauli phase: 0 means +1, 2 means -1 (i.e. neg)
        let phase: i64 = if ppm.neg { 2 } else { 0 };
        Ok(vec![
            py_pack_param(&z_array, qpy_data, Endian::Big)?,
            py_pack_param(&x_array, qpy_data, Endian::Big)?,
            pack_generic_value(&GenericValue::Int64(phase), qpy_data)?,
        ])
    })?;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: instruction.op.num_qubits(),
        num_cargs: instruction.op.num_clbits(),
        extras_key: 0,
        num_ctrl_qubits: 0, // standard instructions have no control qubits
        ctrl_state: 0,
        gate_class_name,
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

fn pack_pauli_product_rotation(
    rotation: &PauliProductRotation,
    instruction: &PackedInstruction,
    qpy_data: &QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    // since we won't recreate this gate via python, it's not important to verify the python name is identical to the one we use here
    // so we simply hard-code it instead of going through python
    let gate_class_name = String::from(PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME);
    let params = Python::attach(|py| -> Result<_, QpyError> {
        let z_array = rotation.z.to_pyarray(py);
        let x_array = rotation.x.to_pyarray(py);
        Ok(vec![
            py_pack_param(&z_array, qpy_data, Endian::Big)?,
            py_pack_param(&x_array, qpy_data, Endian::Big)?,
            pack_param_obj(&rotation.angle, qpy_data, Endian::Little)?,
        ])
    })?;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: instruction.op.num_qubits(),
        num_cargs: 0,
        extras_key: 0,
        num_ctrl_qubits: 0,
        ctrl_state: 0,
        gate_class_name,
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

/// Get Python class names (e.g., "IfElseOp") for a control flow op.
/// This is for backwards compatibility with old QPY versions, since the
/// Rust name uses snake_case.
fn control_flow_class_name(control_flow: &ControlFlow) -> String {
    String::from(match control_flow {
        ControlFlow::Box { .. } => "BoxOp",
        ControlFlow::BreakLoop => "BreakLoopOp",
        ControlFlow::ContinueLoop => "ContinueLoopOp",
        ControlFlow::IfElse { .. } => "IfElseOp",
        ControlFlow::While { .. } => "WhileLoopOp",
        ControlFlow::ForLoop { .. } => "ForLoopOp",
        ControlFlow::Switch { .. } => "SwitchCaseOp",
    })
}
fn pack_control_flow_inst(
    control_flow_inst: &ControlFlowInstruction,
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    let mut packed_annotations = None;
    let mut packed_condition: formats::ConditionPack = Default::default();
    let params = match control_flow_inst.control_flow.clone() {
        ControlFlow::Box {
            duration,
            annotations,
        } => {
            packed_annotations = pack_annotations(&annotations, qpy_data)?;
            let mut params = Vec::new();
            params.extend(pack_instruction_blocks(instruction, qpy_data)?);
            match duration {
                None => {
                    // we follow the python default
                    let duration_value_pack = pack_generic_value(&GenericValue::Null, qpy_data)?;
                    let duration_unit_string = GenericValue::String("dt".to_string());
                    params.push(duration_value_pack);
                    params.push(pack_generic_value(&duration_unit_string, qpy_data)?);
                }
                Some(box_duration) => match box_duration {
                    BoxDuration::Duration(duration) => {
                        let duration_value = match duration {
                            Duration::dt(v) => GenericValue::Int64(v),
                            Duration::ps(v)
                            | Duration::us(v)
                            | Duration::ns(v)
                            | Duration::ms(v)
                            | Duration::s(v) => GenericValue::Float64(v),
                        };
                        let duration_unit_string =
                            GenericValue::String(duration.unit().to_string());
                        params.push(pack_generic_value(&duration_value.as_le(), qpy_data)?);
                        params.push(pack_generic_value(&duration_unit_string, qpy_data)?);
                    }
                    BoxDuration::Expr(exp) => {
                        let duration_value_pack =
                            pack_generic_value(&GenericValue::Expression(exp), qpy_data)?;
                        let duration_unit_string = GenericValue::String("expr".to_string());
                        params.push(duration_value_pack);
                        params.push(pack_generic_value(&duration_unit_string, qpy_data)?);
                    }
                },
            }
            params
        }
        ControlFlow::BreakLoop | ControlFlow::ContinueLoop => Vec::new(),
        ControlFlow::ForLoop {
            collection,
            loop_param,
        } => {
            let collection_value = pack_for_collection(&collection);
            let loop_param_value = match loop_param {
                None => GenericValue::Null,
                Some(LoopParam::Parameter(symbol)) => {
                    GenericValue::ParameterExpressionSymbol(symbol.into())
                }
                Some(LoopParam::Variable(_)) => GenericValue::Null,
            };
            let mut params = Vec::with_capacity(3);
            params.push(pack_generic_value(&collection_value, qpy_data)?);
            params.push(pack_generic_value(&loop_param_value, qpy_data)?);
            params.extend(pack_instruction_blocks(instruction, qpy_data)?);
            params
        }
        ControlFlow::IfElse { condition } => {
            packed_condition = pack_condition(condition, qpy_data)?;
            pack_instruction_blocks(instruction, qpy_data)?
        }
        ControlFlow::While { condition } => {
            packed_condition = pack_condition(condition, qpy_data)?;
            pack_instruction_blocks(instruction, qpy_data)?
        }
        ControlFlow::Switch {
            target, label_spec, ..
        } => {
            // we follow the python way of storing switch params
            // the first param is the target, the next param is the cases specificer
            // the cases specifier is a list of pairs (tuples)
            // the second element in each pair is the subcircuit for this case
            // the first element is the list of the case labels, or a single case label
            // or the special default case label
            let target_value = match target {
                SwitchTarget::Bit(clbit) => {
                    GenericValue::Register(ParamRegisterValue::ShareableClbit(clbit))
                }
                SwitchTarget::Expr(exp) => GenericValue::Expression(exp),
                SwitchTarget::Register(reg) => {
                    GenericValue::Register(ParamRegisterValue::Register(reg))
                }
            };
            let case_circuits = extract_instruction_blocks(instruction, qpy_data);
            let cases = GenericValue::Tuple(
                label_spec
                    .iter()
                    .map(|label_vec| -> Result<GenericValue, QpyError> {
                        Ok(GenericValue::Tuple(
                            label_vec
                                .iter()
                                .map(|label_element| -> Result<GenericValue, QpyError> {
                                    match label_element {
                                        CaseSpecifier::Default => Ok(GenericValue::CaseDefault),
                                        CaseSpecifier::Uint(val) => Ok(GenericValue::Int64(
                                            val.to_i64().ok_or_else(|| {
                                                QpyError::ConversionError(
                                                    "Case specifier too large".to_string(),
                                                )
                                            })?,
                                        )
                                        .as_le()),
                                    }
                                })
                                .collect::<Result<Vec<GenericValue>, _>>()?,
                        ))
                    })
                    .zip(case_circuits)
                    .map(|(label, circuit)| -> Result<GenericValue, QpyError> {
                        Ok(GenericValue::Tuple(vec![label?, circuit]))
                    })
                    .collect::<Result<Vec<GenericValue>, _>>()?,
            );
            vec![
                pack_generic_value(&target_value, qpy_data)?,
                pack_generic_value(&cases, qpy_data)?,
            ]
        }
    };
    let annotations_key = if packed_annotations.is_some() {
        formats::extras_key_parts::ANNOTATIONS
    } else {
        0
    };
    let condition_key = packed_condition.key() as u8;
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: control_flow_inst.num_qubits,
        num_cargs: control_flow_inst.num_clbits,
        extras_key: condition_key | annotations_key,
        num_ctrl_qubits: 0, // standard instructions have no control qubits
        ctrl_state: 0,
        gate_class_name: control_flow_class_name(&control_flow_inst.control_flow), // need to use the Python class names for backward compatibility
        label: Default::default(),
        condition: packed_condition,
        bit_data: Default::default(),
        params,
        annotations: packed_annotations,
    })
}
fn pack_unitary_gate(
    unitary_gate: &UnitaryGate,
    qpy_data: &QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    // unitary gates are special since they are uniquely determined by a matrix, which is not
    // a "parameter", strictly speaking, but is treated as such when serializing

    let matrix = unitary_gate.matrix().ok_or_else(|| {
        QpyError::InvalidParameter("Could not read matrix for unitary gate".to_string())
    })?;

    // until we change the QPY version or verify we get the exact same result,
    // we translate the matrix to numpy and then serialize it like python does
    let params = Python::attach(|py| -> Result<_, QpyError> {
        let out_array = matrix.to_pyarray(py);
        Ok(vec![py_pack_param(&out_array, qpy_data, Endian::Little)?])
    })?;
    // since we won't recreate this gate via python, it's not important to verify the python name is identical to the one we use here
    // so we simply hard-code it instead of going through python
    let gate_class_name = String::from(UNITARY_GATE_CLASS_NAME);
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: unitary_gate.num_qubits(),
        num_cargs: unitary_gate.num_clbits(),
        extras_key: 0,
        num_ctrl_qubits: 0, // unitary gates have no control qubits
        ctrl_state: 0,
        gate_class_name,
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

fn pack_py_instruction(
    py_inst: &PyInstruction,
    instruction: &PackedInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitInstructionV2Pack, QpyError> {
    let params = match py_inst.kind {
        PyOpKind::Gate | PyOpKind::Instruction => pack_instruction_params(instruction, qpy_data)?,
        PyOpKind::Operation => Python::attach(|py| -> Result<_, QpyError> {
            let py_op_object = py_inst.ob.bind(py);
            if py_op_object.is_instance(imports::CLIFFORD.get_bound(py))? {
                let tableau = py_op_object.getattr("tableau")?;
                Ok(vec![py_pack_param(&tableau, qpy_data, Endian::Little)?])
            } else if py_op_object.is_instance(imports::ANNOTATED_OPERATION.get_bound(py))? {
                let modifiers = py_op_object.getattr("modifiers")?;
                modifiers
                    .try_iter()?
                    .map(|modifier| py_pack_param(&modifier?, qpy_data, Endian::Little))
                    .collect::<Result<_, QpyError>>()
            } else {
                pack_instruction_params(instruction, qpy_data)
            }
        })?,
    };
    Ok(formats::CircuitInstructionV2Pack {
        num_qargs: py_inst.num_qubits(),
        num_cargs: py_inst.num_clbits(),
        extras_key: 0,
        num_ctrl_qubits: py_inst.num_ctrl_qubits().unwrap_or(0),
        ctrl_state: py_inst.ctrl_state().unwrap_or(0),
        gate_class_name: Python::attach(|py| py_inst.class_name(py))?,
        label: Default::default(),
        condition: Default::default(),
        bit_data: Default::default(),
        params,
        annotations: None,
    })
}

// packs the quantum registers in the circuit. we pack:
// 1) registers appearing explicitly in the circuit register list (identified using in_circ_lookup)
// 2) registers appearing implicitly for bits (in python: as their "_register" data field)
fn pack_quantum_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    // let in_circ_lookup: HashSet<QuantumRegister> = circuit_data.qregs().iter().cloned().collect();
    // let mut registers_to_pack: IndexSet<QuantumRegister> =
    //     circuit_data.qregs().iter().cloned().collect();
    let mut in_circ_lookup: HashSet<QuantumRegister> = HashSet::new();
    let mut registers_to_pack: IndexSet<QuantumRegister> = IndexSet::default();
    circuit_data.qregs().iter().for_each(|qreg| {
        in_circ_lookup.insert(qreg.clone());
        registers_to_pack.insert(qreg.clone());
    });

    // add all owning registers for qubits in the circuits (even registers not included in the circuit itself)
    registers_to_pack.extend(
        circuit_data
            .qubits()
            .objects()
            .iter()
            .filter_map(|qubit| qubit.owning_register()),
    );
    // add all registers showing up as qubit indices for some qubit in the circuit
    registers_to_pack.extend(
        circuit_data
            .qubits()
            .objects()
            .iter()
            .filter_map(|qubit| circuit_data.qubit_indices().get(qubit)) // only qubits with register data
            .flat_map(|qreg_locations| {
                qreg_locations
                    .registers()
                    .iter()
                    .map(|(qreg, _)| qreg.clone())
            }),
    );
    registers_to_pack
        .iter()
        .map(|qreg| pack_quantum_register(qreg, circuit_data, in_circ_lookup.contains(qreg)))
        .collect()
}

fn pack_quantum_register(
    qreg: &QuantumRegister,
    circuit_data: &CircuitData,
    in_circuit: bool,
) -> formats::RegisterV4Pack {
    let bit_indices = qreg
        .bits()
        .map(|qubit| {
            circuit_data
                .qubit_index(&qubit)
                .map(|index| index as i64)
                .unwrap_or(-1)
        })
        .collect();
    formats::RegisterV4Pack {
        register_type: RegisterType::Qreg,
        standalone: qreg.is_owning() as u8,
        in_circuit: in_circuit as u8,
        name: qreg.name().to_string(),
        bit_indices,
    }
}

fn pack_classical_registers(circuit_data: &CircuitData) -> Vec<formats::RegisterV4Pack> {
    let in_circ_lookup: HashSet<ClassicalRegister> = circuit_data.cregs().iter().cloned().collect();
    let mut registers_to_pack: IndexSet<ClassicalRegister> =
        circuit_data.cregs().iter().cloned().collect();
    // add all owning registers for clbits in the circuits (even registers not included in the circuit itself)
    registers_to_pack.extend(
        circuit_data
            .clbits()
            .objects()
            .iter()
            .filter_map(|clbit| clbit.owning_register()),
    );
    registers_to_pack.extend(
        circuit_data
            .clbits()
            .objects()
            .iter()
            .filter_map(|clbit| circuit_data.clbit_indices().get(clbit)) // only qubits with register data
            .flat_map(|creg_locations| {
                creg_locations
                    .registers()
                    .iter()
                    .map(|(creg, _)| creg.clone())
            }),
    );
    registers_to_pack
        .iter()
        .map(|creg| {
            let bit_indices = creg
                .bits()
                .map(|clbit| {
                    circuit_data
                        .clbit_index(&clbit)
                        .map(|index| index as i64)
                        .unwrap_or(-1)
                })
                .collect();
            formats::RegisterV4Pack {
                register_type: RegisterType::Creg,
                standalone: creg.is_owning() as u8,
                in_circuit: in_circ_lookup.contains(creg) as u8,
                name: creg.name().to_string(),
                bit_indices,
            }
        })
        .collect()
}

fn pack_circuit_header(
    circuit_name: Option<String>,
    circuit_metadata: Option<Bound<PyAny>>,
    metadata_serializer: Option<&Bound<PyAny>>,
    qpy_data: &QPYWriteData,
) -> Result<formats::CircuitHeaderV12Pack, QpyError> {
    let metadata = serialize_metadata(&circuit_metadata, metadata_serializer)?;
    let global_phase_data = pack_param_obj(
        qpy_data.circuit_data.global_phase(),
        qpy_data,
        binrw::Endian::Big,
    )?;
    let qregs = pack_quantum_registers(qpy_data.circuit_data);
    let cregs = pack_classical_registers(qpy_data.circuit_data);
    let mut registers = qregs;
    registers.extend(cregs);
    let header = formats::CircuitHeaderV12Pack {
        num_qubits: qpy_data.circuit_data.num_qubits() as u32,
        num_clbits: qpy_data.circuit_data.num_clbits() as u32,
        num_instructions: qpy_data.circuit_data.len() as u64,
        num_vars: qpy_data
            .circuit_data
            .vars_stretches_view()
            .num_identifiers() as u32,
        circuit_name: circuit_name.unwrap_or_default(),
        global_phase_data: global_phase_data.data,
        global_phase_type: global_phase_data.type_key,
        metadata,
        registers,
    };

    Ok(header)
}

fn pack_layout(
    transpile_layout: Option<Bound<PyAny>>,
    qpy_data: &QPYWriteData,
) -> Result<formats::LayoutV2Pack, QpyError> {
    let default_layout = formats::LayoutV2Pack {
        exists: 0,
        initial_layout_size: -1,
        input_mapping_size: -1,
        final_layout_size: -1,
        input_qubit_count: 0,
        extra_registers: Vec::new(),
        initial_layout_items: Vec::new(),
        input_mapping_items: Vec::new(),
        final_layout_items: Vec::new(),
    };
    match transpile_layout {
        None => Ok(default_layout),
        Some(transpile_layout) => {
            if transpile_layout.is_none() {
                Ok(default_layout)
            } else {
                pack_transpile_layout(&transpile_layout, qpy_data)
            }
        }
    }
}

fn pack_transpile_layout(
    layout: &Bound<PyAny>,
    qpy_data: &QPYWriteData,
) -> Result<formats::LayoutV2Pack, QpyError> {
    let mut initial_layout_size = -1; // initial_size
    let mut input_qubit_mapping: HashMap<ShareableQubit, usize> = HashMap::new();
    let mut initial_layout_array: Vec<(Option<u32>, Option<QuantumRegister>)> = Vec::new();
    let mut extra_registers: HashSet<QuantumRegister> = HashSet::new();
    let mut extra_registers_qubits: HashSet<ShareableQubit> = HashSet::new();

    let initial_layout = layout.getattr("initial_layout")?;
    if !initial_layout.is_none() {
        initial_layout_size = initial_layout.call_method0("__len__")?.extract::<i32>()?;
        let layout_mapping = initial_layout.call_method0("get_physical_bits")?;
        for i in 0..qpy_data.circuit_data.num_qubits() {
            let qubit = layout_mapping
                .get_item(i)?
                .extract::<ShareableQubit>()
                .map_err(|e| QpyError::from(PyErr::from(e)))?;
            input_qubit_mapping.insert(qubit.clone(), i);
            let register = qubit.owning_register();
            let index = qubit.owning_register_index();
            if let Some(reg) = register.clone() {
                extra_registers.insert(reg);
                extra_registers_qubits.insert(qubit);
            } else if index.is_some() {
                extra_registers_qubits.insert(qubit);
            }
            initial_layout_array.push((index, register));
        }
    }

    let mut input_mapping_size: i32 = -1; //input_qubit_size
    let layout_input_qubit_mapping = layout.getattr("input_qubit_mapping")?;
    let input_mapping_items = if !layout_input_qubit_mapping.is_none() {
        input_mapping_size = layout_input_qubit_mapping
            .call_method0("__len__")?
            .extract()?;
        let mut input_qubit_mapping_array: Vec<u32> = vec![0; input_mapping_size as usize];
        let layout_mapping = initial_layout.call_method0("get_virtual_bits")?;
        for (qubit, index) in layout_input_qubit_mapping
            .cast::<PyDict>()
            .map_err(|e| QpyError::from(PyErr::from(e)))?
        {
            let qubit = qubit
                .extract::<ShareableQubit>()
                .map_err(|e| QpyError::from(PyErr::from(e)))?;
            let register = qubit.owning_register();
            if let Some(reg) = register {
                if qubit.owning_register_index().is_some()
                    && !qpy_data.circuit_data.qregs().contains(&reg)
                {
                    extra_registers.insert(reg);
                }
            };
            let i: usize = index.extract()?;
            input_qubit_mapping_array[i] = layout_mapping.get_item(&qubit)?.extract::<u32>()?;
        }
        input_qubit_mapping_array
    } else {
        Vec::new()
    };

    let mut final_layout_size: i32 = -1; // this is the default value if final_layout is not present
    // let final_layout_array = PyList::empty(py);
    let final_layout = layout.getattr("final_layout")?;
    let final_layout_items: Vec<u32> = if !final_layout.is_none() {
        final_layout_size = final_layout.call_method0("__len__")?.extract::<i32>()?;
        let mut final_layout_items: Vec<u32> = Vec::with_capacity(final_layout_size as usize);
        let final_layout_physical = final_layout.call_method0("get_physical_bits")?;
        for i in 0..qpy_data.circuit_data.num_qubits() {
            let virtual_bit = final_layout_physical.get_item(i)?;
            if virtual_bit.is_instance_of::<PyClbit>() {
                let virtual_clbit = virtual_bit
                    .extract::<ShareableClbit>()
                    .map_err(|e| QpyError::from(PyErr::from(e)))?;
                let index = qpy_data
                    .circuit_data
                    .clbit_index(&virtual_clbit)
                    .ok_or_else(|| {
                        QpyError::ConversionError("Clbit missing an index".to_string())
                    })?;
                final_layout_items.push(index);
            } else if virtual_bit.is_instance_of::<PyQubit>() {
                let virtual_qubit = virtual_bit
                    .extract::<ShareableQubit>()
                    .map_err(|e| QpyError::from(PyErr::from(e)))?;
                let index = qpy_data
                    .circuit_data
                    .qubit_index(&virtual_qubit)
                    .ok_or_else(|| {
                        QpyError::ConversionError("Qubit missing an index".to_string())
                    })?;
                final_layout_items.push(index);
            }
        }
        final_layout_items
    } else {
        Vec::new()
    };

    let input_qubit_count: i32 = if layout.getattr("_input_qubit_count")?.is_none() {
        -1
    } else {
        layout.getattr("_input_qubit_count")?.extract()?
    };

    let extra_registers =
        pack_extra_registers(&extra_registers, &extra_registers_qubits, qpy_data)?;

    let initial_layout_items: Vec<formats::InitialLayoutItemV2Pack> = initial_layout_array
        .iter()
        .map(|(index, qreg)| {
            let index_value = match index {
                None => -1,
                Some(val) => *val as i32,
            };
            let (register_name, register_name_length) = match qreg {
                None => (String::new(), -1),
                Some(qreg_val) => (qreg_val.name().to_string(), qreg_val.name().len() as i32),
            };
            formats::InitialLayoutItemV2Pack {
                index_value,
                register_name_length,
                register_name,
            }
        })
        .collect();

    Ok(formats::LayoutV2Pack {
        exists: true as u8,
        initial_layout_size,
        input_mapping_size,
        final_layout_size,
        input_qubit_count,
        extra_registers,
        initial_layout_items,
        input_mapping_items,
        final_layout_items,
    })
}

fn pack_custom_instructions(
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CustomCircuitInstructionsPack, QpyError> {
    let mut custom_instructions: Vec<formats::CustomCircuitInstructionDefPack> = Vec::new();
    let mut instructions_to_pack: Vec<String> = custom_instructions_hash.keys().cloned().collect();
    Python::attach(|py| -> Result<_, QpyError> {
        while let Some(name) = instructions_to_pack.pop() {
            custom_instructions.push(pack_custom_instruction(
                py,
                &name,
                custom_instructions_hash,
                &mut instructions_to_pack,
                qpy_data,
            )?);
        }
        Ok(())
    })?;
    Ok(formats::CustomCircuitInstructionsPack {
        custom_instructions,
    })
}

fn pack_extra_registers(
    in_circ_regs: &HashSet<QuantumRegister>,
    qubits: &HashSet<ShareableQubit>,
    qpy_data: &QPYWriteData,
) -> Result<Vec<formats::RegisterV4Pack>, QpyError> {
    let mut out_circ_regs: HashSet<QuantumRegister> = HashSet::new();
    for qubit in qubits.iter() {
        if let Some(qreg) = qubit.owning_register() {
            if !in_circ_regs.contains(&qreg) {
                out_circ_regs.insert(qreg);
            }
        }
    }
    let mut result = Vec::new();
    for qreg in in_circ_regs.iter() {
        result.push(pack_quantum_register(qreg, qpy_data.circuit_data, true));
    }
    for qreg in out_circ_regs.iter() {
        result.push(pack_quantum_register(qreg, qpy_data.circuit_data, false));
    }
    Ok(result)
}

fn pack_custom_instruction(
    py: Python,
    name: &String,
    custom_instructions_hash: &mut HashMap<String, PackedOperation>,
    new_instructions_list: &mut Vec<String>,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CustomCircuitInstructionDefPack, QpyError> {
    let operation = custom_instructions_hash.get(name).ok_or_else(|| {
        QpyError::ConversionError(format!("Could not find operation data for {}", name))
    })?;
    let OperationRef::PyCustom(inst) = operation.view() else {
        return Err(QpyError::SerializationError(format!(
            "internal logic error: {} was not a `PyCustom` variant: {:?}",
            name,
            operation.view()
        )));
    };
    let gate_type = get_circuit_type_key(operation)?;
    let mut num_ctrl_qubits = 0;
    let mut ctrl_state = 0;
    let mut base_gate: Bound<PyAny> = py.None().bind(py).clone();
    let mut base_gate_raw: Bytes = Bytes::new();

    let data = match gate_type {
        CircuitInstructionType::PauliEvolutionGate => Some(serialize(
            &py_pack_pauli_evolution_gate(inst.ob.bind(py), qpy_data)?,
        )?),
        CircuitInstructionType::ControlledGate => {
            // For ControlledGate, we have to access and store the private `_definition` rather than
            // the public one, because the public one is mutated to include additional logic if the
            // control state is open, and the definition setter (during a subsequent read) uses the
            // "fully excited" control definition only.
            // Build internal definition to support overloaded subclasses by
            // calling definition getter on object
            let gate = inst.ob.bind(py);
            gate.getattr("definition")?; // this creates the _definition field
            num_ctrl_qubits = gate.getattr("num_ctrl_qubits")?.extract::<u32>()?;
            ctrl_state = gate.getattr("ctrl_state")?.extract::<u32>()?;
            base_gate = gate.getattr("base_gate")?.clone();
            Some(serialize(&pack_circuit(
                &mut gate.getattr("_definition")?.extract()?,
                Some(py.None().bind(py)),
                false,
                qpy_data.version,
                qpy_data.annotation_handler.annotation_factories,
            )?)?)
        }
        CircuitInstructionType::AnnotatedOperation => {
            base_gate = inst.ob.bind(py).getattr("base_op")?.clone();
            None
        }
        CircuitInstructionType::Gate | CircuitInstructionType::Instruction => inst
            .py_definition(py)?
            .map(|mut defn| {
                pack_circuit(
                    &mut defn,
                    Some(py.None().bind(py)),
                    false,
                    qpy_data.version,
                    qpy_data.annotation_handler.annotation_factories,
                )
                .and_then(|fmt| serialize(&fmt))
            })
            .transpose()?,
    };
    let num_qubits = operation.num_qubits();
    let num_clbits = operation.num_clbits();
    if !base_gate.is_none() {
        let op_parts = base_gate.extract::<OperationFromPython<CircuitData>>()?;
        let instruction = CircuitInstruction {
            operation: op_parts.operation,
            qubits: PyTuple::empty(py).into(),
            clbits: PyTuple::empty(py).into(),
            params: op_parts.params,
            label: op_parts.label,
            #[cfg(feature = "cache_pygates")]
            py_op: std::sync::OnceLock::new(),
        };
        // The base gate instruction is not present in the circuit itself and we don't use all it's data (e.g. qubits and clbits)
        // But we still want to serialize it like a regular instruction, so we need to convert it to a PackedInstruction.
        // To avoid changing the original CircuitData we use a hack where it is packed using a dummy circuit data.
        // TODO: Hopefully we'll change all this in a future version of QPY.
        let mut dummy_circuit_data = PyCircuitData {
            inner: CircuitData::new(None, None, Param::Float(0.0))?,
        };
        let packed_instruction = dummy_circuit_data.pack(py, &instruction)?;
        base_gate_raw = serialize(&pack_instruction(
            &packed_instruction,
            custom_instructions_hash,
            new_instructions_list,
            qpy_data,
        )?)?;
    }
    Ok(formats::CustomCircuitInstructionDefPack {
        gate_type,
        num_qubits,
        num_clbits,
        custom_definition: data.is_some() as u8,
        num_ctrl_qubits,
        ctrl_state,
        name: name.to_string(),
        data: data.unwrap_or_default(),
        base_gate_raw,
    })
}

fn pack_standalone_vars(
    qpy_data: &mut QPYWriteData,
) -> Result<Vec<formats::ExpressionVarDeclarationPack>, QpyError> {
    let mut result = Vec::new();
    let mut index: u16 = 0;
    let mut uuid: u128 = 0;
    // input vars
    for var in qpy_data
        .circuit_data
        .vars_stretches_view()
        .iter_vars(VarType::Input)
    {
        let var_pack = pack_standalone_var(
            var,
            ExpressionVarDeclaration::Input,
            qpy_data.version,
            &mut uuid,
        )?;
        result.push(var_pack);
        qpy_data.standalone_var_indices.insert(uuid, index);
        index += 1;
    }

    // captured vars
    for var in qpy_data
        .circuit_data
        .vars_stretches_view()
        .iter_vars(VarType::Capture)
    {
        result.push(pack_standalone_var(
            var,
            ExpressionVarDeclaration::Capture,
            qpy_data.version,
            &mut uuid,
        )?);
        qpy_data.standalone_var_indices.insert(uuid, index);
        index += 1;
    }

    // declared vars
    for var in qpy_data
        .circuit_data
        .vars_stretches_view()
        .iter_vars(VarType::Declare)
    {
        result.push(pack_standalone_var(
            var,
            ExpressionVarDeclaration::Local,
            qpy_data.version,
            &mut uuid,
        )?);
        qpy_data.standalone_var_indices.insert(uuid, index);
        index += 1;
    }
    if qpy_data.version < 14
        && (qpy_data
            .circuit_data
            .vars_stretches_view()
            .num_stretches(StretchType::Capture)
            > 0
            || qpy_data
                .circuit_data
                .vars_stretches_view()
                .num_stretches(StretchType::Declare)
                > 0)
    {
        return Err(QpyError::UnsupportedFeatureForVersion {
            feature: "circuits containing stretch variables".to_string(),
            version: 14,
            min_version: qpy_data.version,
        });
    }
    for stretch in qpy_data
        .circuit_data
        .vars_stretches_view()
        .iter_stretches(StretchType::Capture)
    {
        result.push(pack_stretch(
            stretch,
            ExpressionVarDeclaration::StretchCapture,
        ));
        qpy_data.standalone_var_indices.insert(stretch.uuid, index);
        index += 1;
    }
    for stretch in qpy_data
        .circuit_data
        .vars_stretches_view()
        .iter_stretches(StretchType::Declare)
    {
        result.push(pack_stretch(
            stretch,
            ExpressionVarDeclaration::StretchLocal,
        ));
        qpy_data.standalone_var_indices.insert(stretch.uuid, index);
        index += 1;
    }
    Ok(result)
}

pub(crate) fn pack_circuit(
    circuit: &mut QuantumCircuitData,
    metadata_serializer: Option<&Bound<PyAny>>,
    _use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> Result<formats::QPYCircuit, QpyError> {
    let annotation_handler = AnnotationHandler::new(annotation_factories);
    let mut qpy_data = QPYWriteData {
        circuit_data: &mut circuit.data,
        version,
        standalone_var_indices: HashMap::new(),
        annotation_handler,
    };
    let standalone_vars = pack_standalone_vars(&mut qpy_data)?;
    let header = pack_circuit_header(
        circuit.name.clone(),
        circuit.metadata.clone(),
        metadata_serializer,
        &qpy_data,
    )?;
    // Pulse has been removed in Qiskit 2.0. As long as we keep QPY at version 13,
    // we need to write an empty calibrations header since read_circuit expects it
    let calibrations = formats::CalibrationsPack {
        calibrations: vec![],
    };
    let (instructions, mut custom_instructions_hash) = pack_instructions(&mut qpy_data)?;
    let custom_instructions =
        pack_custom_instructions(&mut custom_instructions_hash, &mut qpy_data)?;
    let layout = pack_layout(circuit.transpile_layout.clone(), &qpy_data)?;
    let state_headers: Vec<formats::AnnotationStateHeaderPack> = qpy_data
        .annotation_handler
        .dump_serializers()?
        .into_iter()
        .map(|(namespace, state)| formats::AnnotationStateHeaderPack { namespace, state })
        .collect();
    let annotation_headers = Some(formats::AnnotationHeaderStaticPack { state_headers });
    Ok(formats::QPYCircuit {
        header,
        standalone_vars,
        annotation_headers,
        custom_instructions,
        instructions,
        calibrations,
        layout,
    })
}

#[pyfunction]
#[pyo3(name = "write_circuit")]
#[pyo3(signature = (file_obj, circuit, metadata_serializer, use_symengine, version, annotation_factories))]
pub(crate) fn py_write_circuit(
    py: Python,
    file_obj: &Bound<PyAny>,
    circuit: &Bound<PyAny>,
    metadata_serializer: &Bound<PyAny>,
    use_symengine: bool,
    version: u32,
    annotation_factories: &Bound<PyDict>,
) -> PyResult<usize> {
    let packed_circuit = pack_circuit(
        &mut circuit.extract()?,
        Some(metadata_serializer),
        use_symengine,
        version,
        annotation_factories,
    )?;
    let serialized_circuit = serialize(&packed_circuit)?;
    file_obj.call_method1(
        "write",
        (pyo3::types::PyBytes::new(py, &serialized_circuit),),
    )?;
    Ok(serialized_circuit.len())
}
