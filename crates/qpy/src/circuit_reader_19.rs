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

// Circuit reader module: converts a qpy file into a  QuantumCircuit

// We use the following terminology:
// 1. "Pack": To create a struct (from formats.rs) from the original data
// 2. "Serialize": To create binary data (Bytes) from the original data
// 3. "Write": To write to a file obj the serialization of the original data
// Ideally, serialization is done by packing in a binrw-enhanced struct and using the
// `write` method into a `Cursor` buffer, but there might be exceptions.

use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::imports;
use qiskit_circuit::instruction::create_py_op;
use qiskit_circuit::interner::Interned;
use qiskit_circuit::operations::{
    BoxDuration, CaseSpecifier, Condition, ControlFlow, ControlFlowInstruction, ForCollection,
    LoopParam, OperationRef, Param, PauliBased, PauliProductMeasurement, PauliProductRotation,
    PyInstruction, PyRange, StandardInstruction, Store, SwitchTarget, UnitaryGate,
};
use qiskit_circuit::packed_instruction::{PackedInstruction, PackedOperation};
use qiskit_circuit::parameter::symbol_expr::SymbolVector;
use qiskit_circuit::var_stretch_container::VarType;
use qiskit_circuit::{Clbit, Qubit};
use std::sync::Arc;
use uuid::Uuid;

use crate::annotations::AnnotationHandler;
use crate::error::QpyError;
use crate::expr::unpack_expression;
use crate::formats;
use crate::formats::QPYCircuit;
use crate::params::{
    generic_value_to_param, unpack_parameter_expression, unpack_parameter_vector, unpack_symbol,
};
use crate::py_methods::{
    deserialize_pauli_evolution_operation, py_convert_from_generic_value, py_unpack_modifier,
};
use crate::value::{
    CircuitInstructionType, GenericValue, ParamRegisterValue, QPYReadData, QpyCaller, clbit_at,
    creg_by_name, deserialize_with_args, unpack_array_type, unpack_duration,
};

use std::num::NonZero;

use crate::circuit_reader::{
    CustomCircuitInstructionData, add_registers_and_bits, add_standalone_vars,
    instruction_values_to_params, read_custom_instructions, unpack_annotations, unpack_circuit,
    unpack_instruction,
};
pub(crate) fn unpack_circuit_v19(
    packed_circuit: &QPYCircuit,
    version: u8,
    use_symengine: bool,
    annotation_handler: AnnotationHandler,
    caller: QpyCaller,
) -> Result<CircuitData, QpyError> {
    let formats::CircuitHeaderPack::V19(header) = &packed_circuit.header else {
        return Err(QpyError::InvalidFormat(
            "QPY >= 19 circuit has a pre-QPY 19 header".to_string(),
        ));
    };
    let mut qpy_data = QPYReadData {
        caller,
        circuit_data: CircuitData::with_capacity(
            0,
            0,
            packed_circuit.instructions.len(),
            Param::Float(0.0),
        )?,
        version,
        use_symengine,
        standalone_vars: HashMap::new(),
        standalone_stretches: HashMap::new(),
        vectors: HashMap::new(),
        parameter_vectors: packed_circuit
            .parameter_vectors
            .as_ref()
            .map(|table| {
                table
                    .vectors
                    .iter()
                    .map(|vector| {
                        Arc::new(SymbolVector {
                            name: vector.name.clone(),
                            uuid: Uuid::from_bytes(vector.uuid),
                            len: (vector.vector_size as usize).into(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default(),
        annotation_handler,
    };
    if let Some(annotation_headers) = &packed_circuit.annotation_headers {
        qpy_data.annotation_handler.load_deserializers(
            annotation_headers
                .state_headers
                .iter()
                .map(|data| (data.namespace.clone(), data.state.clone()))
                .collect(),
        )?;
    }

    let global_phase = header.global_phase.to_param(&mut qpy_data)?;
    qpy_data.circuit_data.set_global_phase_param(global_phase)?;
    add_standalone_vars(packed_circuit, &mut qpy_data)?;
    add_registers_and_bits(packed_circuit, &mut qpy_data)?;

    let qargs = unpack_interner_entries(
        &header.qubit_interner,
        qpy_data.circuit_data.num_qubits(),
        "qubit",
        |indices| qpy_data.circuit_data.add_qargs(indices),
    )?;
    let cargs = unpack_interner_entries(
        &header.clbit_interner,
        qpy_data.circuit_data.num_clbits(),
        "clbit",
        |indices| qpy_data.circuit_data.add_cargs(indices),
    )?;
    let custom_instructions = read_custom_instructions(packed_circuit, &mut qpy_data)?;
    let custom_instruction_names = packed_circuit
        .custom_instructions
        .custom_instructions
        .iter()
        .map(|operation| operation.name.as_str())
        .collect::<Vec<_>>();

    for packed_instruction in &packed_circuit.instructions {
        let formats::CircuitInstructionPack::V19(instruction) = packed_instruction else {
            return Err(QpyError::InvalidFormat(
                "QPY >= 19 circuit has a pre-QPY 19 instruction".to_string(),
            ));
        };
        let instruction = unpack_instruction_v19(
            instruction,
            &qargs,
            &cargs,
            &custom_instructions,
            &custom_instruction_names,
            &mut qpy_data,
        )?;
        qpy_data.circuit_data.push(instruction)?;
    }
    Ok(qpy_data.circuit_data)
}

fn unpack_instruction_v19(
    instruction: &formats::CircuitInstructionV19Pack,
    qargs: &[Interned<[Qubit]>],
    cargs: &[Interned<[Clbit]>],
    custom_instructions: &HashMap<String, CustomCircuitInstructionData>,
    custom_instruction_names: &[&str],
    qpy_data: &mut QPYReadData,
) -> Result<PackedInstruction, QpyError> {
    let qubits = *qargs.get(instruction.qargs as usize).ok_or_else(|| {
        QpyError::InvalidBit(format!(
            "qubit interner index {} out of range",
            instruction.qargs
        ))
    })?;
    let clbits = *cargs.get(instruction.cargs as usize).ok_or_else(|| {
        QpyError::InvalidBit(format!(
            "clbit interner index {} out of range",
            instruction.cargs
        ))
    })?;
    let parameter_values = instruction
        .params
        .iter()
        .map(|param| unpack_param_data_v19(param, qpy_data))
        .collect::<Result<Vec<_>, _>>()?;
    if instruction.annotations.is_some()
        && !matches!(
            instruction.operation_data,
            formats::OperationData::ControlFlow(formats::ControlFlowPack::Box(_))
        )
    {
        return Err(QpyError::DeserializationError(
            "non-Box instructions do not support annotations".to_string(),
        ));
    }
    let op = match &instruction.operation_data {
        formats::OperationData::StandardGate(discriminant) => {
            let gate = bytemuck::checked::try_pod_read_unaligned::<
                qiskit_circuit::operations::StandardGate,
            >(&[*discriminant])
            .map_err(|_| {
                QpyError::InvalidInstruction(format!(
                    "invalid standard-gate discriminant {discriminant}"
                ))
            })?;
            PackedOperation::from_standard_gate(gate)
        }
        formats::OperationData::StandardInstruction(data) => {
            let instruction = match data.discriminant {
                0 => StandardInstruction::Barrier(
                    qpy_data.circuit_data.get_qargs(qubits).len() as u32
                ),
                1 => StandardInstruction::Delay(data.delay_unit.ok_or_else(|| {
                    QpyError::InvalidInstruction("delay instruction has no unit".to_string())
                })?),
                2 => StandardInstruction::Measure,
                3 => StandardInstruction::Reset,
                value => {
                    return Err(QpyError::InvalidInstruction(format!(
                        "invalid standard-instruction discriminant {value}"
                    )));
                }
            };
            PackedOperation::from_standard_instruction(instruction)
        }
        formats::OperationData::UnitaryGate(matrix) => {
            let array = unpack_array_type(matrix.clone())?;
            PackedOperation::from_unitary(Box::new(UnitaryGate { array }))
        }
        formats::OperationData::Custom(index) => {
            let name = custom_instruction_names
                .get(*index as usize)
                .ok_or_else(|| {
                    QpyError::MissingData(format!(
                        "custom instruction index {index} is out of range"
                    ))
                })?;
            let data = custom_instructions.get(*name).ok_or_else(|| {
                QpyError::MissingData(format!("custom instruction data not found for {name}"))
            })?;
            qpy_data.caller.attach("Custom instruction", |py| {
                unpack_custom_instruction_v19(
                    py,
                    name,
                    data,
                    custom_instructions,
                    &parameter_values,
                    instruction.label.as_ref().map(|label| label.value.as_str()),
                    qpy_data,
                )
            })?
        }
        formats::OperationData::PauliEvolution(data) => {
            let [time] = parameter_values.as_slice() else {
                return Err(QpyError::InvalidParameter(format!(
                    "Pauli evolution gate requires exactly one time parameter, got {}",
                    parameter_values.len()
                )));
            };
            qpy_data
                .caller
                .attach("unpack Pauli evolution operation", |py| {
                    let operation = deserialize_pauli_evolution_operation(
                        py,
                        &data.pauli_data,
                        data.standalone_op,
                        &data.synth_data,
                        time.clone(),
                        qpy_data,
                    )?;
                    Ok::<_, QpyError>(
                        operation
                            .extract::<OperationFromPython<CircuitData>>(py)?
                            .operation,
                    )
                })?
        }
        formats::OperationData::ControlFlow(data) => {
            let control_flow = unpack_control_flow_v19(
                data,
                &parameter_values,
                &instruction.annotations,
                qpy_data,
            )?;
            PackedOperation::from_control_flow(Box::new(ControlFlowInstruction {
                control_flow,
                num_qubits: qpy_data.circuit_data.get_qargs(qubits).len() as u32,
                num_clbits: qpy_data.circuit_data.get_cargs(clbits).len() as u32,
            }))
        }
        formats::OperationData::PauliProductMeasurement(data) => {
            if !parameter_values.is_empty() {
                return Err(QpyError::InvalidParameter(format!(
                    "Pauli product measurement requires no parameters, got {}",
                    parameter_values.len()
                )));
            }
            let z = unpack_bool_vector(&data.z)?;
            let x = unpack_bool_vector(&data.x)?;
            if z.len() != x.len() {
                return Err(QpyError::InvalidParameter(
                    "Pauli product measurement z and x vectors have different lengths".to_string(),
                ));
            }
            PackedOperation::from_pauli_based(Box::new(PauliBased::PauliProductMeasurement(
                PauliProductMeasurement {
                    z,
                    x,
                    neg: data.neg != 0,
                },
            )))
        }
        formats::OperationData::PauliProductRotation(data) => {
            let [angle_value] = parameter_values.as_slice() else {
                return Err(QpyError::InvalidParameter(format!(
                    "Pauli product rotation requires exactly one angle parameter, got {}",
                    parameter_values.len()
                )));
            };
            let z = unpack_bool_vector(&data.z)?;
            let x = unpack_bool_vector(&data.x)?;
            if z.len() != x.len() {
                return Err(QpyError::InvalidParameter(
                    "Pauli product rotation z and x vectors have different lengths".to_string(),
                ));
            }
            let angle = generic_value_to_param(angle_value, qpy_data)?;
            PackedOperation::from_pauli_based(Box::new(PauliBased::PauliProductRotation(
                PauliProductRotation { z, x, angle },
            )))
        }
        formats::OperationData::Store(lvalue, rvalue) => {
            if !parameter_values.is_empty() {
                return Err(QpyError::InvalidParameter(format!(
                    "store requires no parameters, got {}",
                    parameter_values.len()
                )));
            }
            PackedOperation::from_store(Box::new(Store::new(
                unpack_expression(lvalue.clone(), qpy_data)?,
                unpack_expression(rvalue.clone(), qpy_data)?,
            )))
        }
        formats::OperationData::FromPython(data) => {
            qpy_data
                .caller
                .attach("unpack Python-defined operation", |py| {
                    unpack_from_python_v19(
                        py,
                        data,
                        &parameter_values,
                        qpy_data.circuit_data.get_qargs(qubits).len(),
                        None,
                        qpy_data,
                    )
                })?
        }
        formats::OperationData::Controlled(data) => {
            qpy_data
                .caller
                .attach("unpack Python-defined controlled gate", |py| {
                    unpack_from_python_v19(
                        py,
                        &data.from_python,
                        &parameter_values,
                        qpy_data.circuit_data.get_qargs(qubits).len(),
                        Some((data.num_ctrl_qubits, data.ctrl_state)),
                        qpy_data,
                    )
                })?
        }
    };
    let params = instruction_values_to_params(parameter_values, qpy_data)?;
    Ok(PackedInstruction {
        op,
        qubits,
        clbits,
        params,
        label: instruction
            .label
            .as_ref()
            .map(|label| Box::new(label.value.clone())),
        #[cfg(feature = "cache_pygates")]
        py_op: std::sync::OnceLock::new(),
    })
}

fn unpack_custom_instruction_v19(
    py: Python<'_>,
    serialized_name: &str,
    data: &CustomCircuitInstructionData,
    custom_instructions: &HashMap<String, CustomCircuitInstructionData>,
    parameter_values: &[GenericValue],
    label: Option<&str>,
    qpy_data: &mut QPYReadData,
) -> Result<PackedOperation, QpyError> {
    let mut name = serialized_name
        .rsplit_once('_')
        .map_or(serialized_name, |(name, _)| name);
    if data.gate_type == CircuitInstructionType::ControlledGate
        && data.ctrl_state < (1u32 << data.num_ctrl_qubits) - 1
    {
        name = name.rsplit_once('_').map_or(name, |(name, _)| name);
    }
    let py_params = parameter_values
        .iter()
        .map(|value| {
            generic_value_to_param(value, qpy_data)?
                .into_pyobject(py)
                .map_err(QpyError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let object = match data.gate_type {
        CircuitInstructionType::Gate => {
            imports::GATE
                .get_bound(py)
                .call1((name, data.num_qubits, py_params))?
        }
        CircuitInstructionType::Instruction => imports::INSTRUCTION.get_bound(py).call1((
            name,
            data.num_qubits,
            data.num_clbits,
            py_params,
        ))?,
        CircuitInstructionType::ControlledGate => {
            let packed_base_gate = deserialize_with_args::<
                formats::CircuitInstructionV2Pack,
                (bool,),
            >(&data.base_gate_raw, (false,))?
            .0;
            let base_gate = unpack_instruction(&packed_base_gate, custom_instructions, qpy_data)?;
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
            kwargs.set_item("num_ctrl_qubits", data.num_ctrl_qubits)?;
            kwargs.set_item("ctrl_state", data.ctrl_state)?;
            kwargs.set_item("base_gate", py_base_gate)?;
            imports::CONTROLLED_GATE
                .get_bound(py)
                .call((name, data.num_qubits, py_params), Some(&kwargs))?
        }
        CircuitInstructionType::AnnotatedOperation => {
            let packed_base_op =
                deserialize_with_args::<formats::CircuitInstructionV2Pack, (bool,)>(
                    &data.base_gate_raw,
                    (false,),
                )?
                .0;
            let base_op = unpack_instruction(&packed_base_op, custom_instructions, qpy_data)?;
            let params = qpy_data
                .circuit_data
                .unpack_blocks_to_circuit_parameters(base_op.params.as_deref());
            let py_base_op = create_py_op(
                py,
                base_op.op.view(),
                params,
                base_op.label.as_deref().map(String::as_str),
            )?;
            imports::ANNOTATED_OPERATION
                .get_bound(py)
                .call1((py_base_op, py_params))?
        }
        other => {
            return Err(QpyError::DeserializationError(format!(
                "QPY 19 custom instruction type {other:?} is not implemented"
            )));
        }
    };
    if let Some(definition) = &data.definition_circuit {
        object.setattr("definition", definition)?;
    }
    if let Some(label) = label {
        object.setattr("label", label)?;
    }
    Ok(object
        .extract::<OperationFromPython<CircuitData>>()?
        .operation)
}

fn unpack_from_python_v19(
    py: Python<'_>,
    data: &formats::FromPythonPack,
    parameter_values: &[GenericValue],
    num_qubits: usize,
    control: Option<(u32, u32)>,
    qpy_data: &mut QPYReadData,
) -> Result<PackedOperation, QpyError> {
    let gate_class = crate::py_methods::get_python_gate_class(py, &data.class_name.value)?;
    let py_params = parameter_values
        .iter()
        .map(|value| py_convert_from_generic_value(py, value))
        .collect::<Result<Vec<_>, _>>()?;
    let kwargs = PyDict::new(py);
    for init_param in &data.init_params {
        let value = unpack_param_data_v19(&init_param.value, qpy_data)?;
        kwargs.set_item(
            &init_param.name.value,
            py_convert_from_generic_value(py, &value)?,
        )?;
    }
    if let Some((num_ctrl_qubits, ctrl_state)) = control {
        // Most standard controlled gates have a fixed number of controls, so their Python
        // constructors do not accept `num_ctrl_qubits` (for example, `CXGate`).  The multi-
        // controlled families below are the exceptions and need the serialized count in order
        // to construct an object of the right size.
        if matches!(
            data.class_name.value.as_str(),
            "MCPhaseGate" | "MCU1Gate" | "MCXGrayCode" | "MCXGate" | "MCXRecursive" | "MCXVChain"
        ) {
            kwargs.set_item("num_ctrl_qubits", num_ctrl_qubits)?;
        }
        kwargs.set_item("ctrl_state", ctrl_state)?;
    }

    let object = if !kwargs.is_empty() {
        gate_class.call(PyTuple::new(py, py_params)?, Some(&kwargs))?
    } else {
        match data.class_name.value.as_str() {
            "Initialize" | "StatePreparation" => {
                if py_params
                    .first()
                    .is_some_and(|param| param.bind(py).is_instance_of::<PyString>())
                {
                    let label = py_params
                        .iter()
                        .map(|param| param.extract(py))
                        .collect::<PyResult<Vec<String>>>()?
                        .join("");
                    gate_class.call1((label,))?
                } else if let [param] = py_params.as_slice() {
                    let value: f64 = param.getattr(py, "real")?.extract(py)?;
                    gate_class.call1((value as u32, num_qubits))?
                } else {
                    gate_class.call1((py_params,))?
                }
            }
            "QFTGate" => gate_class.call1((num_qubits,))?,
            "UCRXGate" | "UCRYGate" | "UCRZGate" | "DiagonalGate" => {
                gate_class.call1((py_params,))?
            }
            _ => gate_class.call1(PyTuple::new(py, py_params)?)?,
        }
    };
    if object.getattr("name")?.extract::<String>()? != data.op_name.value {
        object.setattr("_name", &data.op_name.value)?;
    }
    let operation = object
        .extract::<OperationFromPython<CircuitData>>()?
        .operation;
    if let OperationRef::PyCustom(py_instruction) = operation.view() {
        Ok(PackedOperation::from(PyInstruction {
            op_name: data.op_name.value.clone(),
            ..py_instruction.clone()
        }))
    } else {
        Ok(operation)
    }
}

fn unpack_bool_vector(data: &formats::BoolVectorPack) -> Result<Vec<bool>, QpyError> {
    let num_bits = usize::try_from(data.num_bits).map_err(|_| {
        QpyError::InvalidParameter("boolean vector length does not fit in memory".to_string())
    })?;
    if data.data.len() != num_bits.div_ceil(8) {
        return Err(QpyError::InvalidParameter(
            "invalid bit-packed boolean vector length".to_string(),
        ));
    }
    Ok((0..num_bits)
        .map(|index| data.data[index / 8] & (1 << (index % 8)) != 0)
        .collect())
}

fn unpack_param_data_v19(
    value: &formats::ParamDataPack,
    qpy_data: &mut QPYReadData,
) -> Result<GenericValue, QpyError> {
    Ok(match value {
        formats::ParamDataPack::Bool(value) => GenericValue::Bool(*value != 0),
        formats::ParamDataPack::Int64(value) => GenericValue::Int64(*value),
        formats::ParamDataPack::BigInt(value) => GenericValue::BigInt(value.clone()),
        formats::ParamDataPack::Float64(value) => GenericValue::Float64(*value),
        formats::ParamDataPack::Complex64(value) => GenericValue::Complex64(*value),
        formats::ParamDataPack::CaseDefault => GenericValue::CaseDefault,
        formats::ParamDataPack::Range(start, stop, step) => GenericValue::Range(PyRange {
            start: isize::try_from(*start).map_err(|_| {
                QpyError::InvalidParameter("range start does not fit in isize".to_string())
            })?,
            stop: isize::try_from(*stop).map_err(|_| {
                QpyError::InvalidParameter("range stop does not fit in isize".to_string())
            })?,
            step: NonZero::new(isize::try_from(*step).map_err(|_| {
                QpyError::InvalidParameter("range step does not fit in isize".to_string())
            })?)
            .ok_or_else(|| QpyError::InvalidParameter("range step cannot be zero".to_string()))?,
        }),
        formats::ParamDataPack::NumpyObject { data, .. } => GenericValue::NumpyObject(data.clone()),
        formats::ParamDataPack::Tuple { elements, .. } => GenericValue::Tuple(
            elements
                .iter()
                .map(|element| unpack_param_data_v19(element, qpy_data))
                .collect::<Result<_, _>>()?,
        ),
        formats::ParamDataPack::Parameter(value) => {
            GenericValue::ParameterExpressionSymbol(Arc::new(unpack_symbol(value)))
        }
        formats::ParamDataPack::ParameterVectorElement(value) => {
            GenericValue::ParameterExpressionVectorSymbol(Arc::new(unpack_parameter_vector(
                value, qpy_data,
            )?))
        }
        formats::ParamDataPack::ParameterExpression(value) => GenericValue::ParameterExpression(
            Arc::new(unpack_parameter_expression(value, qpy_data)?),
        ),
        formats::ParamDataPack::String(value) => GenericValue::String(value.value.clone()),
        formats::ParamDataPack::Null => GenericValue::Null,
        formats::ParamDataPack::Expression(value) => GenericValue::Expression(
            crate::expr::unpack_expression(value.expression.clone(), qpy_data)?,
        ),
        formats::ParamDataPack::Modifier(value) => GenericValue::Modifier(
            qpy_data
                .caller
                .attach("unpack modifier", |py| py_unpack_modifier(py, value))?,
        ),
        formats::ParamDataPack::Duration(value) => {
            GenericValue::Duration(unpack_duration(value.clone()))
        }
        formats::ParamDataPack::Register(value) => GenericValue::Register(match value {
            formats::ParamDataRegisterPack::Register(name) => {
                ParamRegisterValue::Register(creg_by_name(&name.value, qpy_data)?)
            }
            formats::ParamDataRegisterPack::Clbit(index) => {
                ParamRegisterValue::ShareableClbit(clbit_at(*index, qpy_data)?)
            }
        }),
        formats::ParamDataPack::Circuit(value) => {
            GenericValue::CircuitData(Box::new(unpack_circuit(
                value,
                qpy_data.version,
                qpy_data.use_symengine,
                qpy_data.annotation_handler.child()?,
                qpy_data.caller,
            )?))
        }
    })
}

fn unpack_control_flow_v19(
    value: &formats::ControlFlowPack,
    blocks: &[GenericValue],
    annotations: &Option<formats::InstructionsAnnotationPack>,
    qpy_data: &mut QPYReadData,
) -> Result<ControlFlow, QpyError> {
    Ok(match value {
        formats::ControlFlowPack::Box(duration) => {
            let duration = match duration {
                formats::BoxDurationPack::None => None,
                formats::BoxDurationPack::Duration(duration) => {
                    Some(BoxDuration::Duration(unpack_duration(duration.clone())))
                }
                formats::BoxDurationPack::Expression(expression) => Some(BoxDuration::Expr(
                    crate::expr::unpack_expression(expression.expression.clone(), qpy_data)?,
                )),
            };
            ControlFlow::Box {
                duration,
                annotations: unpack_annotations(annotations, qpy_data)?,
            }
        }
        formats::ControlFlowPack::BreakLoop => ControlFlow::BreakLoop,
        formats::ControlFlowPack::ContinueLoop => ControlFlow::ContinueLoop,
        formats::ControlFlowPack::ForLoop(collection, loop_param) => {
            let collection = match collection {
                formats::ForCollectionPack::List { values, .. } => ForCollection::List(
                    values
                        .iter()
                        .map(|value| isize::try_from(*value).map_err(QpyError::from))
                        .collect::<Result<_, _>>()?,
                ),
                formats::ForCollectionPack::Range(start, stop, step) => {
                    ForCollection::PyRange(PyRange {
                        start: isize::try_from(*start)?,
                        stop: isize::try_from(*stop)?,
                        step: NonZero::new(isize::try_from(*step)?).ok_or_else(|| {
                            QpyError::InvalidParameter("range step cannot be zero".to_string())
                        })?,
                    })
                }
            };
            let loop_param = match loop_param {
                formats::LoopParamPack::None => None,
                formats::LoopParamPack::Parameter(symbol) => {
                    Some(LoopParam::Parameter(unpack_symbol(symbol)))
                }
                formats::LoopParamPack::Variable => {
                    let [GenericValue::CircuitData(body)] = blocks else {
                        return Err(QpyError::InvalidInstruction(
                            "for loop with a variable parameter requires one body".to_string(),
                        ));
                    };
                    let mut vars = body.vars_stretches_view().iter_vars(VarType::Input);
                    let var = vars.next().ok_or_else(|| {
                        QpyError::MissingData("for-loop input variable is missing".to_string())
                    })?;
                    if vars.next().is_some() {
                        return Err(QpyError::InvalidInstruction(
                            "for-loop body has more than one input variable".to_string(),
                        ));
                    }
                    Some(LoopParam::Variable(var.clone()))
                }
            };
            ControlFlow::ForLoop {
                collection,
                loop_param,
            }
        }
        formats::ControlFlowPack::IfElse(condition) => ControlFlow::IfElse {
            condition: unpack_condition_v19(condition, qpy_data)?,
        },
        formats::ControlFlowPack::While(condition) => ControlFlow::While {
            condition: unpack_condition_v19(condition, qpy_data)?,
        },
        formats::ControlFlowPack::Switch(target, case_spec) => {
            if case_spec.labels.len() != blocks.len() {
                return Err(QpyError::InvalidInstruction(format!(
                    "switch has {} label groups but {} blocks",
                    case_spec.labels.len(),
                    blocks.len()
                )));
            }
            let target = match target {
                formats::SwitchTargetPack::Bit(index) => {
                    SwitchTarget::Bit(clbit_at(*index, qpy_data)?)
                }
                formats::SwitchTargetPack::Register(name) => {
                    SwitchTarget::Register(creg_by_name(&name.value, qpy_data)?)
                }
                formats::SwitchTargetPack::Expression(expression) => SwitchTarget::Expr(
                    crate::expr::unpack_expression(expression.expression.clone(), qpy_data)?,
                ),
            };
            let label_spec = case_spec
                .labels
                .iter()
                .map(|labels| {
                    labels
                        .labels
                        .iter()
                        .map(|label| match label {
                            formats::CaseSpecifierPack::Default => CaseSpecifier::Default,
                            formats::CaseSpecifierPack::Uint(value) => {
                                CaseSpecifier::Uint(value.clone())
                            }
                        })
                        .collect()
                })
                .collect();
            ControlFlow::Switch {
                target,
                label_spec,
                cases: blocks.len() as u32,
            }
        }
    })
}

fn unpack_condition_v19(
    condition: &formats::ConditionV19Pack,
    qpy_data: &mut QPYReadData,
) -> Result<Condition, QpyError> {
    Ok(match condition {
        formats::ConditionV19Pack::Bit(index, value) => {
            Condition::Bit(clbit_at(*index, qpy_data)?, *value != 0)
        }
        formats::ConditionV19Pack::Register(name, value) => {
            Condition::Register(creg_by_name(&name.value, qpy_data)?, value.clone())
        }
        formats::ConditionV19Pack::Expression(expression) => Condition::Expr(
            crate::expr::unpack_expression(expression.expression.clone(), qpy_data)?,
        ),
    })
}

fn unpack_interner_entries<T: From<u32>, I: Copy>(
    entries: &[formats::InternerEntry],
    num_bits: usize,
    bit_name: &str,
    mut intern: impl FnMut(&[T]) -> I,
) -> Result<Vec<I>, QpyError> {
    entries
        .iter()
        .map(|entry| {
            let raw = match entry {
                formats::InternerEntry::Unused => Vec::new(),
                formats::InternerEntry::Single(a) => vec![*a],
                formats::InternerEntry::Double(a, b) => vec![*a, *b],
                formats::InternerEntry::Triple(a, b, c) => vec![*a, *b, *c],
                formats::InternerEntry::VariableSize { bits } => bits.clone(),
                formats::InternerEntry::All => (0..num_bits as u32).collect(),
            };
            if let Some(index) = raw.iter().find(|&&index| index as usize >= num_bits) {
                return Err(QpyError::InvalidBit(format!(
                    "{bit_name} index {index} out of range (circuit has {num_bits} {bit_name}s)"
                )));
            }
            let bits: Vec<T> = raw.into_iter().map(T::from).collect();
            Ok(intern(&bits))
        })
        .collect()
}

// handling for non control flow gates with conditionals, for backwards compatability
