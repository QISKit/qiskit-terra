// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// QPY 19 circuit writer.

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.
#![allow(dead_code)] // The QPY 19 writer is enabled in a later branch.

use hashbrown::{HashMap, HashSet};

use pyo3::prelude::*;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::imports;
use qiskit_circuit::operations::{
    BoxDuration, CaseSpecifier, Condition, ControlFlow, LoopParam, OperationRef, Param,
    PyInstruction, StandardInstruction, StandardInstructionType, SwitchTarget,
};
use qiskit_circuit::packed_instruction::PackedOperation;

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::error::QpyError;
use crate::formats;
use crate::interface::ExtraCircuitData;
use crate::params::{pack_parameter_expression, pack_parameter_vector, pack_symbol};
use crate::py_methods::{
    gate_class_name, py_convert_to_generic_value, py_pack_modifier,
    py_pack_pauli_evolution_operation, recognize_custom_operation,
};
use crate::value::{
    CircuitInstructionType, GenericValue, ParamRegisterValue, QPYWriteData, QpyCaller,
    StringU16Pack, clbit_index, deserialize_with_args, get_circuit_type_key, pack_array_type,
    pack_duration, serialize,
};

use crate::expr::pack_expression;

use crate::circuit_writer::{
    default_layout, pack_annotations, pack_circuit, pack_classical_registers,
    pack_custom_instructions, pack_layout, pack_quantum_registers, pack_standalone_vars,
};
fn generic_value_to_param_data_pack(
    value: &GenericValue,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::ParamDataPack, QpyError> {
    Ok(match value {
        GenericValue::Bool(value) => formats::ParamDataPack::Bool(*value as u8),
        GenericValue::Int64(value) => formats::ParamDataPack::Int64(*value),
        GenericValue::BigInt(value) => formats::ParamDataPack::BigInt(value.clone()),
        GenericValue::Float64(value) => formats::ParamDataPack::Float64(*value),
        GenericValue::Complex64(value) => formats::ParamDataPack::Complex64(*value),
        GenericValue::CaseDefault => formats::ParamDataPack::CaseDefault,
        GenericValue::Range(value) => formats::ParamDataPack::Range(
            i64::try_from(value.start).map_err(|_| {
                QpyError::InvalidParameter("range start does not fit in i64".to_string())
            })?,
            i64::try_from(value.stop).map_err(|_| {
                QpyError::InvalidParameter("range stop does not fit in i64".to_string())
            })?,
            i64::try_from(value.step.get()).map_err(|_| {
                QpyError::InvalidParameter("range step does not fit in i64".to_string())
            })?,
        ),
        GenericValue::NumpyObject(data) => {
            formats::ParamDataPack::NumpyObject { data: data.clone() }
        }
        GenericValue::Tuple(values) => formats::ParamDataPack::Tuple {
            elements: values
                .iter()
                .map(|value| generic_value_to_param_data_pack(value, qpy_data))
                .collect::<Result<_, _>>()?,
        },
        GenericValue::ParameterExpressionSymbol(symbol) => {
            formats::ParamDataPack::Parameter(pack_symbol(symbol))
        }
        GenericValue::ParameterExpressionVectorSymbol(symbol) => {
            formats::ParamDataPack::ParameterVectorElement(pack_parameter_vector(symbol, qpy_data)?)
        }
        GenericValue::ParameterExpression(expression) => {
            formats::ParamDataPack::ParameterExpression(pack_parameter_expression(
                expression, qpy_data,
            )?)
        }
        GenericValue::String(value) => formats::ParamDataPack::String(StringU16Pack {
            value: value.clone(),
        }),
        GenericValue::Null => formats::ParamDataPack::Null,
        GenericValue::Expression(expression) => {
            formats::ParamDataPack::Expression(formats::ExpressionPack {
                expression: pack_expression(expression, qpy_data)?,
            })
        }
        GenericValue::Modifier(modifier) => formats::ParamDataPack::Modifier(
            qpy_data
                .caller
                .attach("pack modifier", |py| py_pack_modifier(py, modifier))?,
        ),
        GenericValue::CircuitData(circuit_data) => {
            circuit_data_to_param_data_pack(circuit_data, qpy_data)?
        }
        GenericValue::Duration(duration) => {
            formats::ParamDataPack::Duration(pack_duration(duration))
        }
        GenericValue::Register(register) => formats::ParamDataPack::Register(match register {
            ParamRegisterValue::Register(register) => {
                formats::ParamDataRegisterPack::Register(StringU16Pack {
                    value: register.name().to_string(),
                })
            }
            ParamRegisterValue::ShareableClbit(clbit) => {
                formats::ParamDataRegisterPack::Clbit(clbit_index(clbit, qpy_data)?)
            }
        }),
    })
}

fn circuit_data_to_param_data_pack(
    circuit_data: &CircuitData,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::ParamDataPack, QpyError> {
    let layout = serialize(&pack_layout(None, circuit_data, qpy_data.version)?)?;
    Ok(formats::ParamDataPack::Circuit(Box::new(pack_circuit(
        circuit_data,
        ExtraCircuitData {
            name: None,
            metadata: "{}".into(),
            layout,
        },
        qpy_data.version,
        qpy_data.annotation_handler.child()?,
        qpy_data.caller,
    )?)))
}

fn pack_param_v19(
    param: &Param,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::ParamDataPack, QpyError> {
    match param {
        Param::Int(value) => Ok(formats::ParamDataPack::Int64(*value)),
        Param::Float(value) => Ok(formats::ParamDataPack::Float64(*value)),
        Param::ParameterExpression(expression) => generic_value_to_param_data_pack(
            &GenericValue::from_parameter_expression(expression),
            qpy_data,
        ),
        Param::Obj(value) => qpy_data.caller.attach("Python parameter", |py| {
            let value = py_convert_to_generic_value(value.bind(py))?;
            generic_value_to_param_data_pack(&value, qpy_data)
        }),
    }
}

fn pack_control_flow_v19(
    control_flow: &ControlFlow,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::ControlFlowPack, QpyError> {
    Ok(match control_flow {
        ControlFlow::Box { duration, .. } => formats::ControlFlowPack::Box(match duration {
            None => formats::BoxDurationPack::None,
            Some(BoxDuration::Duration(duration)) => {
                formats::BoxDurationPack::Duration(pack_duration(duration))
            }
            Some(BoxDuration::Expr(expression)) => {
                formats::BoxDurationPack::Expression(formats::ExpressionPack {
                    expression: pack_expression(expression, qpy_data)?,
                })
            }
        }),
        ControlFlow::BreakLoop => formats::ControlFlowPack::BreakLoop,
        ControlFlow::ContinueLoop => formats::ControlFlowPack::ContinueLoop,
        ControlFlow::ForLoop {
            collection,
            loop_param,
        } => {
            let collection = match collection {
                qiskit_circuit::operations::ForCollection::List(values) => {
                    formats::ForCollectionPack::List {
                        values: values
                            .iter()
                            .map(|value| i64::try_from(*value).map_err(QpyError::from))
                            .collect::<Result<_, _>>()?,
                    }
                }
                qiskit_circuit::operations::ForCollection::PyRange(value) => {
                    formats::ForCollectionPack::Range(
                        i64::try_from(value.start)?,
                        i64::try_from(value.stop)?,
                        i64::try_from(value.step.get())?,
                    )
                }
            };
            let loop_param = match loop_param {
                None => formats::LoopParamPack::None,
                Some(LoopParam::Parameter(symbol)) => {
                    formats::LoopParamPack::Parameter(pack_symbol(symbol))
                }
                Some(LoopParam::Variable(_)) => formats::LoopParamPack::Variable,
            };
            formats::ControlFlowPack::ForLoop(collection, loop_param)
        }
        ControlFlow::IfElse { condition } => {
            formats::ControlFlowPack::IfElse(pack_condition_v19(condition, qpy_data)?)
        }
        ControlFlow::While { condition } => {
            formats::ControlFlowPack::While(pack_condition_v19(condition, qpy_data)?)
        }
        ControlFlow::Switch {
            target, label_spec, ..
        } => {
            let target = match target {
                SwitchTarget::Bit(bit) => {
                    formats::SwitchTargetPack::Bit(clbit_index(bit, qpy_data)?)
                }
                SwitchTarget::Register(register) => {
                    formats::SwitchTargetPack::Register(StringU16Pack {
                        value: register.name().to_string(),
                    })
                }
                SwitchTarget::Expr(expression) => {
                    formats::SwitchTargetPack::Expression(formats::ExpressionPack {
                        expression: pack_expression(expression, qpy_data)?,
                    })
                }
            };
            let labels = label_spec
                .iter()
                .map(|labels| formats::CaseLabelsPack {
                    labels: labels
                        .iter()
                        .map(|label| match label {
                            CaseSpecifier::Default => formats::CaseSpecifierPack::Default,
                            CaseSpecifier::Uint(value) => {
                                formats::CaseSpecifierPack::Uint(value.clone())
                            }
                        })
                        .collect(),
                })
                .collect();
            formats::ControlFlowPack::Switch(target, formats::CaseSpecPack { labels })
        }
    })
}

fn pack_condition_v19(
    condition: &Condition,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::ConditionV19Pack, QpyError> {
    Ok(match condition {
        Condition::Bit(bit, value) => {
            formats::ConditionV19Pack::Bit(clbit_index(bit, qpy_data)?, *value as u8)
        }
        Condition::Register(register, value) => formats::ConditionV19Pack::Register(
            StringU16Pack {
                value: register.name().to_string(),
            },
            value.clone(),
        ),
        Condition::Expr(expression) => {
            formats::ConditionV19Pack::Expression(formats::ExpressionPack {
                expression: pack_expression(expression, qpy_data)?,
            })
        }
    })
}

type PackedInstructionsV19 = (
    Vec<formats::CircuitInstructionV19Pack>,
    HashMap<String, PackedOperation>,
    Vec<String>,
);

fn pack_instructions_v19(qpy_data: &mut QPYWriteData) -> Result<PackedInstructionsV19, QpyError> {
    let mut custom_operations = HashMap::new();
    let mut v19_custom_operations = Vec::new();
    // Copy the circuit-data reference out of `qpy_data` so the instructions can be borrowed while
    // the writer state is mutated below.  Cloning the instructions is both unnecessary and, for
    // circuits borrowed through the C API, may clone cached Python objects while the calling
    // thread is detached from the interpreter.
    let circuit_data = qpy_data.circuit_data;
    let instructions = circuit_data.data();
    let mut packed_instructions = Vec::with_capacity(instructions.len());

    for instruction in instructions {
        let qargs = instruction.qubits.index();
        let cargs = instruction.clbits.index();
        let instruction_type = get_circuit_type_key(&instruction.op, qpy_data.caller)?;
        let operation_view = instruction.op.view();
        let annotations = match operation_view {
            OperationRef::ControlFlow(control_flow) => match &control_flow.control_flow {
                ControlFlow::Box { annotations, .. } => pack_annotations(annotations, qpy_data)?,
                _ => None,
            },
            _ => None,
        };

        let custom_index = if matches!(
            instruction_type,
            CircuitInstructionType::Gate
                | CircuitInstructionType::Instruction
                | CircuitInstructionType::ControlledGate
                | CircuitInstructionType::AnnotatedOperation
        ) && matches!(operation_view, OperationRef::PyCustom(_))
            && matches!(qpy_data.caller, QpyCaller::Python)
        {
            let custom_name = qpy_data.caller.attach(
                "recognize custom operations",
                |py| -> Result<_, QpyError> {
                    recognize_custom_operation(
                        py,
                        &instruction.op,
                        &gate_class_name(py, &instruction.op)?,
                        qpy_data,
                    )
                },
            )?;
            custom_name.map(|name| {
                let index = v19_custom_operations.len();
                v19_custom_operations.push(name.clone());
                custom_operations.insert(name, instruction.op.clone());
                index
            })
        } else {
            None
        };
        let (operation, operation_data) = match instruction.op.view() {
            OperationRef::StandardGate(gate) => (
                formats::CircuitOperationType::StandardGate,
                formats::OperationData::StandardGate(gate as u8),
            ),
            OperationRef::StandardInstruction(inst) => (
                formats::CircuitOperationType::StandardInstruction,
                standard_instruction_operation_data(&inst),
            ),
            OperationRef::Unitary(gate) => (
                formats::CircuitOperationType::UnitaryGate,
                formats::OperationData::UnitaryGate(pack_array_type(&gate.array)?),
            ),
            OperationRef::PyCustom(custom)
                if instruction_type == CircuitInstructionType::PauliEvolutionGate =>
            {
                let data = qpy_data
                    .caller
                    .attach("pack Pauli evolution operation", |py| {
                        py_pack_pauli_evolution_operation(custom.ob.bind(py), qpy_data)
                    })?;
                (
                    formats::CircuitOperationType::PauliEvolution,
                    formats::OperationData::PauliEvolution(data),
                )
            }
            OperationRef::ControlFlow(control_flow) => (
                formats::CircuitOperationType::ControlFlow,
                formats::OperationData::ControlFlow(pack_control_flow_v19(
                    &control_flow.control_flow,
                    qpy_data,
                )?),
            ),
            OperationRef::PauliProductMeasurement(measurement) => (
                formats::CircuitOperationType::PauliProductMeasurement,
                formats::OperationData::PauliProductMeasurement(
                    formats::PauliProductMeasurementPack {
                        z: pack_bool_vector(&measurement.z)?,
                        x: pack_bool_vector(&measurement.x)?,
                        neg: measurement.neg as u8,
                    },
                ),
            ),
            OperationRef::PauliProductRotation(rotation) => (
                formats::CircuitOperationType::PauliProductRotation,
                formats::OperationData::PauliProductRotation(formats::PauliProductRotationPack {
                    z: pack_bool_vector(&rotation.z)?,
                    x: pack_bool_vector(&rotation.x)?,
                }),
            ),
            OperationRef::Store(store) => (
                formats::CircuitOperationType::Store,
                formats::OperationData::Store(
                    pack_expression(store.lvalue(), qpy_data)?,
                    pack_expression(store.rvalue(), qpy_data)?,
                ),
            ),
            OperationRef::PyCustom(_) if custom_index.is_some() => (
                formats::CircuitOperationType::Custom,
                formats::OperationData::Custom(custom_index.ok_or_else(|| {
                    QpyError::SerializationError(
                        "custom operation has no custom-instruction index".to_string(),
                    )
                })? as u64),
            ),
            OperationRef::PyCustom(custom) if custom.num_ctrl_qubits().unwrap_or(0) > 0 => (
                formats::CircuitOperationType::Controlled,
                formats::OperationData::Controlled(formats::ControlledGatePack {
                    from_python: pack_from_python_v19(custom, qpy_data)?,
                    num_ctrl_qubits: custom.num_ctrl_qubits().unwrap_or(0),
                    ctrl_state: custom.ctrl_state().unwrap_or(0),
                }),
            ),
            OperationRef::PyCustom(custom) => (
                formats::CircuitOperationType::FromPython,
                formats::OperationData::FromPython(pack_from_python_v19(custom, qpy_data)?),
            ),
            OperationRef::CustomOperation(_) => unreachable!(
                "pack_instruction rejects compiled custom operations before QPY 19 conversion"
            ),
        };

        let params = if let OperationRef::PyCustom(custom) = instruction.op.view()
            && qpy_data
                .caller
                .attach("identify Clifford operation", |py| {
                    custom
                        .ob
                        .bind(py)
                        .is_instance(imports::CLIFFORD.get_bound(py))
                        .map_err(QpyError::from)
                })? {
            qpy_data.caller.attach("pack Clifford tableau", |py| {
                let tableau = custom.ob.bind(py).getattr("tableau")?;
                Ok::<_, QpyError>(vec![generic_value_to_param_data_pack(
                    &py_convert_to_generic_value(&tableau)?,
                    qpy_data,
                )?])
            })?
        } else if instruction_type == CircuitInstructionType::AnnotatedOperation
            && let OperationRef::PyCustom(custom) = instruction.op.view()
        {
            qpy_data
                .caller
                .attach("pack annotated-operation modifiers", |py| {
                    custom
                        .ob
                        .bind(py)
                        .getattr("modifiers")?
                        .try_iter()?
                        .map(|modifier| {
                            generic_value_to_param_data_pack(
                                &py_convert_to_generic_value(&modifier?)?,
                                qpy_data,
                            )
                        })
                        .collect::<Result<_, QpyError>>()
                })?
        } else if matches!(instruction.op.view(), OperationRef::ControlFlow(_)) {
            instruction
                .blocks_view()
                .iter()
                .filter_map(|&block_id| circuit_data.blocks().get(block_id))
                .map(|block| circuit_data_to_param_data_pack(block, qpy_data))
                .collect::<Result<_, _>>()?
        } else {
            instruction
                .params_view()
                .iter()
                .map(|param| pack_param_v19(param, qpy_data))
                .collect::<Result<_, _>>()?
        };
        let label = instruction
            .label
            .as_deref()
            .filter(|label| !label.is_empty())
            .map(|label| StringU16Pack {
                value: label.clone(),
            });
        packed_instructions.push(formats::CircuitInstructionV19Pack {
            operation,
            qargs,
            cargs,
            operation_data,
            params,
            annotations,
            label,
        });
    }

    Ok((
        packed_instructions,
        custom_operations,
        v19_custom_operations,
    ))
}

fn pack_from_python_v19(
    instruction: &PyInstruction,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::FromPythonPack, QpyError> {
    qpy_data
        .caller
        .attach("pack Python-defined operation", |py| {
            let object = instruction.ob.bind(py);
            let class_name = instruction.class_name(py)?;
            let init_params = pack_init_params_v19(object, &class_name, qpy_data)?;
            Ok(formats::FromPythonPack {
                class_name: StringU16Pack { value: class_name },
                op_name: StringU16Pack {
                    value: instruction.op_name.clone(),
                },
                init_params,
            })
        })
}

fn pack_init_params_v19(
    object: &Bound<PyAny>,
    class_name: &str,
    qpy_data: &mut QPYWriteData,
) -> Result<Vec<formats::NamedParamDataPack>, QpyError> {
    let init_attributes = match class_name {
        "MCXVChain" => vec![
            ("dirty_ancillas", "_dirty_ancillas"),
            ("relative_phase", "_relative_phase"),
            ("action_only", "_action_only"),
        ],
        _ => Vec::new(),
    };
    let init_values = init_attributes
        .into_iter()
        .map(|(name, attribute)| Ok((name, object.getattr(attribute)?)))
        .collect::<Result<Vec<_>, QpyError>>()?;

    init_values
        .into_iter()
        .map(|(name, value)| {
            Ok(formats::NamedParamDataPack {
                name: StringU16Pack {
                    value: name.to_string(),
                },
                value: generic_value_to_param_data_pack(
                    &py_convert_to_generic_value(&value)?,
                    qpy_data,
                )?,
            })
        })
        .collect()
}

fn pack_bool_vector(values: &[bool]) -> Result<formats::BoolVectorPack, QpyError> {
    let num_bits = u32::try_from(values.len()).map_err(|_| {
        QpyError::InvalidParameter("boolean vector is too large for QPY".to_string())
    })?;
    let mut data = vec![0; values.len().div_ceil(8)];
    for (index, value) in values.iter().enumerate() {
        if *value {
            data[index / 8] |= 1 << (index % 8);
        }
    }
    Ok(formats::BoolVectorPack {
        num_bits,
        data: data.into(),
    })
}

fn standard_instruction_operation_data(inst: &StandardInstruction) -> formats::OperationData {
    let (inst_type, delay_unit) = match inst {
        StandardInstruction::Barrier(_) => (StandardInstructionType::Barrier, None),
        StandardInstruction::Delay(unit) => (StandardInstructionType::Delay, Some((*unit).into())),
        StandardInstruction::Measure => (StandardInstructionType::Measure, None),
        StandardInstruction::Reset => (StandardInstructionType::Reset, None),
    };
    formats::OperationData::StandardInstruction(formats::StandardInstructionData {
        discriminant: inst_type as u8,
        delay_unit,
    })
}

fn pack_circuit_header_v19(
    circuit_name: Option<String>,
    metadata: Bytes,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::CircuitHeaderPack, QpyError> {
    let global_phase = pack_global_phase(qpy_data.circuit_data.global_phase(), qpy_data)?;
    let qregs = pack_quantum_registers(qpy_data.circuit_data, qpy_data.version);
    let cregs = pack_classical_registers(qpy_data.circuit_data, qpy_data.version);
    let mut registers = qregs;
    registers.extend(cregs);
    let (qubit_interner, clbit_interner) = pack_interners(qpy_data);
    let header = formats::CircuitHeaderV19Pack {
        circuit_name: StringU16Pack {
            value: circuit_name.unwrap_or_default(),
        },
        global_phase,
        num_qubits: qpy_data.circuit_data.num_qubits() as u32,
        num_clbits: qpy_data.circuit_data.num_clbits() as u32,
        num_instructions: qpy_data.circuit_data.len() as u64,
        num_vars: qpy_data
            .circuit_data
            .vars_stretches_view()
            .num_identifiers() as u32,
        registers,
        qubit_interner,
        clbit_interner,
        metadata,
    };

    Ok(formats::CircuitHeaderPack::V19(header))
}

fn pack_global_phase(
    global_phase: &Param,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::GlobalPhasePack, QpyError> {
    match global_phase {
        Param::Float(val) => Ok(formats::GlobalPhasePack::Float(*val)),
        Param::Int(val) => Ok(formats::GlobalPhasePack::Float(*val as f64)),
        Param::ParameterExpression(exp) => {
            GenericValue::from_parameter_expression(exp).pack_global_phase(qpy_data)
        }
        Param::Obj(py_object) => qpy_data.caller.attach("Python parameter", |py| {
            py_convert_to_generic_value(py_object.bind(py))?.pack_global_phase(qpy_data)
        }),
    }
}

fn pack_interners(
    qpy_data: &mut QPYWriteData,
) -> (Vec<formats::InternerEntry>, Vec<formats::InternerEntry>) {
    fn pack_entry<T>(bits: &[T], num_bits: usize, is_used: bool) -> formats::InternerEntry
    where
        T: Copy + Into<u32>,
    {
        let indices = bits.iter().copied().map(Into::into).collect::<Vec<_>>();
        // Interners can retain entries that are no longer referenced after a circuit is
        // transformed.  In particular, control-flow block compaction can leave entries whose
        // indices refer to the enclosing circuit.  Keep the slot so instruction interner indices
        // remain stable, but do not emit invalid bit references for these stale entries.
        if !is_used && indices.iter().any(|&index| index as usize >= num_bits) {
            return formats::InternerEntry::Unused;
        }
        match indices.as_slice() {
            [a] => formats::InternerEntry::Single(*a),
            [a, b] => formats::InternerEntry::Double(*a, *b),
            [a, b, c] => formats::InternerEntry::Triple(*a, *b, *c),
            bits if bits.len() == num_bits
                && bits
                    .iter()
                    .enumerate()
                    .all(|(index, bit)| *bit as usize == index) =>
            {
                formats::InternerEntry::All
            }
            _ => formats::InternerEntry::VariableSize { bits: indices },
        }
    }

    let used_qargs = qpy_data
        .circuit_data
        .data()
        .iter()
        .map(|instruction| instruction.qubits.index())
        .collect::<HashSet<_>>();
    let used_cargs = qpy_data
        .circuit_data
        .data()
        .iter()
        .map(|instruction| instruction.clbits.index())
        .collect::<HashSet<_>>();
    let qubit_interner = qpy_data
        .circuit_data
        .qargs_interner()
        .values()
        .enumerate()
        .map(|(index, bits)| {
            pack_entry(
                bits,
                qpy_data.circuit_data.num_qubits(),
                used_qargs.contains(&(index as u32)),
            )
        })
        .collect::<Vec<_>>();

    let clbit_interner = qpy_data
        .circuit_data
        .cargs_interner()
        .values()
        .enumerate()
        .map(|(index, bits)| {
            pack_entry(
                bits,
                qpy_data.circuit_data.num_clbits(),
                used_cargs.contains(&(index as u32)),
            )
        })
        .collect::<Vec<_>>();

    (qubit_interner, clbit_interner)
}

pub(crate) fn pack_circuit_v19(
    circuit_data: &CircuitData,
    extra: ExtraCircuitData,
    version: u8,
    annotation_handler: AnnotationHandler,
    caller: QpyCaller,
) -> Result<formats::QPYCircuit, QpyError> {
    let mut qpy_data = QPYWriteData {
        caller,
        circuit_data,
        version,
        standalone_var_indices: HashMap::new(),
        parameter_vectors: Default::default(),
        annotation_handler,
        custom_gate_counter: 0,
    };
    let standalone_vars = pack_standalone_vars(&mut qpy_data)?;
    let header = pack_circuit_header_v19(extra.name, extra.metadata, &mut qpy_data)?;

    let (instructions, mut custom_instructions_hash, custom_instruction_names) =
        pack_instructions_v19(&mut qpy_data)?;
    let instructions: Vec<formats::CircuitInstructionPack> = instructions
        .into_iter()
        .map(formats::CircuitInstructionPack::V19)
        .collect();
    let custom_instructions = pack_custom_instructions(
        &mut custom_instructions_hash,
        Some(custom_instruction_names),
        &mut qpy_data,
    )?;
    let layout = if extra.layout.is_empty() {
        default_layout()
    } else {
        deserialize_with_args::<formats::LayoutV2Pack, (u8,)>(&extra.layout, (version,))?.0
    };
    let state_headers: Vec<formats::AnnotationStateHeaderPack> = qpy_data
        .annotation_handler
        .dump_serializers()?
        .into_iter()
        .map(|(namespace, state)| formats::AnnotationStateHeaderPack { namespace, state })
        .collect();
    let annotation_headers = Some(formats::AnnotationHeaderStaticPack { state_headers });
    let parameter_vectors = Some(qpy_data.parameter_vectors.to_pack());
    Ok(formats::QPYCircuit {
        header,
        standalone_vars,
        annotation_headers,
        parameter_vectors,
        custom_instructions,
        instructions,
        calibrations: None, // calibrations are not present in QPY 19
        layout,
    })
}
