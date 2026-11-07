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

// Methods for QPY serialization working directly with Python-based data
use binrw::Endian;
use hashbrown::HashMap;
use numpy::Complex64;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{
    IntoPyDict, PyAny, PyComplex, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType,
};
use qiskit_circuit::classical::expr::Expr;
use std::num::NonZero;
use std::sync::Arc;

use qiskit_circuit::bit::{ClassicalRegister, ShareableClbit};
use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::classical;
use qiskit_circuit::imports;
use qiskit_circuit::instruction::create_py_op;
use qiskit_circuit::operations::{Operation, OperationRef, PyInstruction, PyOpKind, PyRange};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::{PyParameter, PyParameterExpression};
use qiskit_quantum_info::sparse_observable::{BitTerm, PySparseObservable, SparseObservable};

use crate::bytes::Bytes;
use crate::circuit_reader::{
    CustomCircuitInstructionData, get_instruction_values, unpack_condition, unpack_instruction,
    unpack_layout,
};
use crate::circuit_writer::standard_instruction_class_name;
use crate::error::QpyError;
use crate::formats;
use crate::params::generic_value_to_param;
use crate::value::{
    BitType, CircuitInstructionType, GenericValue, ModifierType, ParamRegisterValue, QPYReadData,
    QPYWriteData, ValueEndian, ValueType, deserialize_with_args, load_value,
    serialize_generic_value,
};

pub const UNITARY_GATE_CLASS_NAME: &str = "UnitaryGate";
pub const STORE_INSTR_CLASS_NAME: &str = "Store";
pub const PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME: &str = "PauliProductMeasurement";
pub const PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME: &str = "PauliProductRotationGate";

fn is_python_gate(
    py: Python,
    op: &PackedOperation,
    python_gate: &Bound<PyAny>,
) -> Result<bool, QpyError> {
    match op.view() {
        OperationRef::PyCustom(PyInstruction {
            kind: PyOpKind::Gate,
            ob,
            ..
        }) => ob.bind(py).is_instance(python_gate).map_err(QpyError::from),
        _ => Ok(false),
    }
}

/// custom gates have unique gate_id attached to their name
/// this method recognizes whether we have such a gate and returns a unique name for it
/// since custom gates are implemented in python, this is a heavy python-space function
pub(crate) fn recognize_custom_operation(
    py: Python,
    op: &PackedOperation,
    name: &String,
    qpy_data: &mut QPYWriteData,
) -> Result<Option<String>, QpyError> {
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let controlflow = py.import("qiskit.circuit.controlflow")?;

    if (!library.hasattr(name)?
        && !circuit_mod.hasattr(name)?
        && !controlflow.hasattr(name)?
        && (name != "Clifford" && name != PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME))
        || name == "Gate"
        || name == "Instruction"
        || is_python_gate(py, op, imports::BLUEPRINT_CIRCUIT.get_bound(py))?
    {
        // Assign a gate_id to each instance of a custom operation
        let new_name = if !["ucrx_dg", "ucry_dg", "ucrz_dg"].contains(&op.name()) {
            format!("{}_{}", op.name(), qpy_data.next_custom_gate_id())
        } else {
            // ucr*_dg gates can have different numbers of parameters,
            // the gate_id is appended to avoid storing a single definition
            // in circuits with multiple ucr*_dg gates. For legacy reasons
            // the gate_id is stored in a different format as this was done
            // prior to QPY 11.
            format!("{}_{}", op.name(), qpy_data.next_custom_gate_id())
        };
        return Ok(Some(new_name));
    }

    if ["ControlledGate", "AnnotatedOperation"].contains(&name.as_str())
        || is_python_gate(py, op, imports::MCMT_GATE.get_bound(py))?
    {
        return Ok(Some(format!(
            "{}_{}",
            op.name(),
            qpy_data.next_custom_gate_id()
        )));
    }

    if is_python_gate(py, op, imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
        return Ok(Some(format!(
            "###PauliEvolutionGate_{}",
            qpy_data.next_custom_gate_id()
        )));
    }

    Ok(None)
}

/// when trying to instantiate nonstandard gates, we turn to the relevant python clas
/// this function obtains the class based on the gate class name
pub(crate) fn get_python_gate_class<'a>(
    py: Python<'a>,
    gate_class_name: &String,
) -> Result<Bound<'a, PyAny>, QpyError> {
    let library = py.import("qiskit.circuit.library")?;
    let circuit_mod = py.import("qiskit.circuit")?;
    let control_flow = py.import("qiskit.circuit.controlflow")?;
    if library.hasattr(gate_class_name)? {
        library.getattr(gate_class_name).map_err(QpyError::from)
    } else if circuit_mod.hasattr(gate_class_name)? {
        circuit_mod.getattr(gate_class_name).map_err(QpyError::from)
    } else if control_flow.hasattr(gate_class_name)? {
        control_flow
            .getattr(gate_class_name)
            .map_err(QpyError::from)
    } else if gate_class_name == "Clifford" {
        Ok(imports::CLIFFORD.get_bound(py).clone())
    } else if gate_class_name == "pauli_product_measurement" {
        Ok(imports::PAULI_PRODUCT_MEASUREMENT.get_bound(py).clone())
    } else {
        Err(QpyError::ConversionError(format!(
            "Gate class not found: {:?}",
            gate_class_name
        )))
    }
}

// serializes python metadata to JSON using a python JSON serializer
pub(crate) fn serialize_metadata(
    metadata_opt: &Option<Bound<PyAny>>,
    metadata_serializer: Option<&Bound<PyAny>>,
) -> Result<Bytes, QpyError> {
    match metadata_opt {
        None => Ok(Bytes::new()),
        Some(metadata) => {
            let py = metadata.py();
            let none = py.None();
            let py_serializer = metadata_serializer.unwrap_or(none.bind(py));
            let json = py.import("json")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("separators", PyTuple::new(py, [",", ":"])?)?;
            kwargs.set_item("cls", py_serializer)?;
            Ok(json
                .call_method("dumps", (metadata,), Some(&kwargs))?
                .extract::<String>()?
                .into())
        }
    }
}

pub(crate) fn py_serialize_numpy_object(py_object: &Bound<PyAny>) -> Result<Bytes, QpyError> {
    let py = py_object.py();
    let np = py.import("numpy")?;
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    np.call_method1("save", (&buffer, py_object))?;
    Ok(buffer.call_method0("getvalue")?.extract::<Bytes>()?)
}

pub(crate) fn py_deserialize_numpy_object(py: Python, data: &Bytes) -> Result<Py<PyAny>, QpyError> {
    let np = py.import("numpy")?;
    let io = py.import("io")?;
    let buffer = io.call_method0("BytesIO")?;
    buffer.call_method1("write", (data.clone(),))?;
    buffer.call_method1("seek", (0,))?;
    Ok(np.call_method1("load", (buffer,))?.unbind())
}

fn pack_sparse_pauli_op(
    operator: &Bound<PyAny>,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::PauliDataPack, QpyError> {
    if operator.is_instance_of::<PySparseObservable>() {
        let py_sparse_observable: PyRef<PySparseObservable> = operator
            .extract()
            .map_err(|e| QpyError::from(PyErr::from(e)))?;
        let sparse_observable = py_sparse_observable.inner.read().map_err(|_| {
            QpyError::ConversionError("Can't extract sparse observable data".to_string())
        })?;
        let num_qubits = sparse_observable.num_qubits();
        let coeff_data = sparse_observable
            .coeffs()
            .iter()
            .flat_map(|coeff| [coeff.re, coeff.im])
            .collect();
        let bitterm_data = sparse_observable
            .bit_terms()
            .iter()
            .map(|&bitterm| bitterm as u8)
            .collect();
        let inds_data = sparse_observable.indices().to_vec();
        let bounds_data = sparse_observable
            .boundaries()
            .iter()
            .map(|&boundary| boundary as u64)
            .collect();
        let sparse_observable_pack = formats::SparsePauliObservableElemPack {
            num_qubits,
            coeff_data,
            bitterm_data,
            inds_data,
            bounds_data,
        };
        Ok(formats::PauliDataPack::V17(
            formats::PauliDataPackV17::SparseObservable(sparse_observable_pack),
        ))
    } else {
        // this is the case of SparsePauliOp, which we convert to a numpy list
        let op_as_np_list = operator.call_method1("to_list", (true,))?;
        let value = py_convert_to_generic_value(&op_as_np_list)?;
        let (_, data) = serialize_generic_value(&value, qpy_data)?;
        let pack = formats::SparsePauliOpListElemPack { data };
        Ok(formats::PauliDataPack::V17(
            formats::PauliDataPackV17::SparsePauliOp(pack),
        ))
    }
}

pub(crate) fn py_pack_pauli_evolution_gate(
    evolution_gate: &Bound<PyAny>,
    qpy_data: &mut QPYWriteData,
) -> Result<formats::PauliEvolutionDefPack, QpyError> {
    let py = evolution_gate.py();
    let operators = evolution_gate.getattr("operator")?;
    let mut standalone = false;
    let operator_list: Bound<PyList> = if !operators.is_instance_of::<PyList>() {
        standalone = true;
        PyList::new(py, [operators])?
    } else {
        operators
            .cast()
            .map_err(|e| QpyError::from(PyErr::from(e)))?
            .clone()
    };
    let pauli_data = operator_list
        .iter()
        .map(|operator| pack_sparse_pauli_op(&operator, qpy_data))
        .collect::<Result<_, QpyError>>()?;

    let time_value = py_convert_to_generic_value(&evolution_gate.getattr("time")?)?;
    let (time_type, time_data) = serialize_generic_value(&time_value, qpy_data)?;
    let synth_class = evolution_gate
        .getattr("synthesis")?
        .get_type()
        .getattr("__name__")?;
    let settings_dict = evolution_gate.getattr("synthesis")?.getattr("settings")?;
    let json = py.import("json")?;
    let args = PyDict::new(py);
    args.set_item("class", synth_class)?;
    args.set_item("settings", settings_dict)?;
    let synth_data: Bytes = json
        .call_method1("dumps", (args,))?
        .extract::<String>()?
        .into();

    let standalone_op = standalone as u8;
    Ok(formats::PauliEvolutionDefPack {
        standalone_op,
        time_type,
        pauli_data,
        time_data,
        synth_data,
    })
}

pub(crate) fn gate_class_name(py: Python, op: &PackedOperation) -> Result<String, QpyError> {
    match op.view() {
        // getting __name__ for standard gates and instructions should
        // eventually be replaced with a Rust-side mapping
        OperationRef::StandardGate(gate) => Ok(imports::get_std_gate_class_name(&gate)),
        OperationRef::StandardInstruction(inst) => {
            Ok(standard_instruction_class_name(&inst).to_string())
        }
        OperationRef::PyCustom(inst) => inst.class_name(py).map_err(QpyError::from),
        OperationRef::Unitary(_) => Ok(UNITARY_GATE_CLASS_NAME.to_string()),
        OperationRef::PauliProductMeasurement(_) => {
            Ok(String::from(PAULI_PRODUCT_MEASUREMENT_GATE_CLASS_NAME))
        }
        OperationRef::PauliProductRotation(_) => {
            Ok(String::from(PAULI_PRODUCT_ROTATION_GATE_CLASS_NAME))
        }
        OperationRef::ControlFlow(inst) => Ok(inst.name().to_string()),
        OperationRef::Store(_store) => Ok(STORE_INSTR_CLASS_NAME.to_string()),
        OperationRef::CustomOperation(_) => {
            Err(PyTypeError::new_err("Custom gates from rust are not classes.").into())
        }
    }
}

pub(crate) fn py_get_type_key(py_object: &Bound<PyAny>) -> Result<ValueType, QpyError> {
    let py: Python<'_> = py_object.py();
    if py_object.is_instance(imports::PARAMETER_VECTOR_ELEMENT.get_bound(py))? {
        return Ok(ValueType::ParameterVector);
    } else if py_object.is_instance(imports::PARAMETER.get_bound(py))? {
        return Ok(ValueType::Parameter);
    } else if py_object.is_instance(imports::PARAMETER_EXPRESSION.get_bound(py))? {
        return Ok(ValueType::ParameterExpression);
    } else if py_object.is_instance(imports::QUANTUM_CIRCUIT.get_bound(py))? {
        return Ok(ValueType::Circuit);
    } else if py_object.is_instance(imports::CLBIT.get_bound(py))?
        || py_object.is_instance(imports::CLASSICAL_REGISTER.get_bound(py))?
    {
        return Ok(ValueType::Register);
    } else if py_object.extract::<classical::expr::Expr>().is_ok() {
        return Ok(ValueType::Expression);
    } else if py_object.is_instance(imports::BUILTIN_RANGE.get_bound(py))? {
        return Ok(ValueType::Range);
    } else if py_object.is_instance(imports::NUMPY_ARRAY.get_bound(py))? {
        return Ok(ValueType::NumpyObject);
    } else if py_object.is_instance(imports::MODIFIER.get_bound(py))? {
        return Ok(ValueType::Modifier);
    } else if py_object.is_instance_of::<PyInt>() {
        return Ok(ValueType::Integer);
    } else if py_object.is_instance_of::<PyFloat>() {
        return Ok(ValueType::Float);
    } else if py_object.is_instance_of::<PyComplex>() {
        return Ok(ValueType::Complex);
    } else if py_object.is_instance_of::<PyString>() {
        return Ok(ValueType::String);
    } else if py_object.is_instance_of::<PyTuple>() || py_object.is_instance_of::<PyList>() {
        return Ok(ValueType::Tuple);
    } else if py_object.is(imports::CASE_DEFAULT.get_bound(py)) {
        return Ok(ValueType::CaseDefault);
    } else if py_object.is_none() {
        return Ok(ValueType::Null);
    }

    Err(QpyError::ConversionError(format!(
        "Unidentified type_key for: {}",
        py_object
    )))
}

pub(crate) fn py_convert_to_generic_value(
    py_object: &Bound<PyAny>,
) -> Result<GenericValue, QpyError> {
    let type_key = py_get_type_key(py_object)?;
    match type_key {
        ValueType::Bool => Ok(GenericValue::Bool(py_object.extract::<bool>()?)),
        ValueType::Integer => Ok(GenericValue::Int64(py_object.extract::<i64>()?)),
        ValueType::Float => Ok(GenericValue::Float64(py_object.extract::<f64>()?)),
        ValueType::Complex => Ok(GenericValue::Complex64(py_object.extract::<Complex64>()?)),
        ValueType::String => Ok(GenericValue::String(py_object.extract::<String>()?)),
        ValueType::Expression => Ok(GenericValue::Expression(py_object.extract::<Expr>()?)),
        ValueType::CaseDefault => Ok(GenericValue::CaseDefault),
        ValueType::Null => Ok(GenericValue::Null),
        ValueType::Parameter => Ok(GenericValue::ParameterExpressionSymbol(
            py_object
                .cast::<PyParameter>()
                .map_err(PyErr::from)?
                .borrow()
                .0
                .clone(),
        )),
        ValueType::ParameterVector => Ok(GenericValue::ParameterExpressionVectorSymbol(
            py_object
                .cast::<PyParameter>()
                .map_err(PyErr::from)?
                .borrow()
                .0
                .clone(),
        )),
        ValueType::ParameterExpression => Ok(GenericValue::ParameterExpression(Arc::new(
            py_object
                .extract::<PyParameterExpression>()
                .map_err(|e| QpyError::from(PyErr::from(e)))?
                .inner,
        ))),
        ValueType::Circuit => {
            py_object.getattr("data")?; // in case _data is lazily generated in python
            let py_circuit_data = py_object.getattr("_data")?;
            let circuit_data = py_circuit_data
                .cast::<PyCircuitData>()
                .map_err(PyErr::from)?
                .borrow()
                .inner
                .clone();
            Ok(GenericValue::CircuitData(Box::new(circuit_data)))
        }
        ValueType::Tuple => {
            let elements: Vec<GenericValue> = py_object
                .try_iter()?
                .map(|data_item| {
                    // let data_item = possible_data_item?;
                    py_convert_to_generic_value(&data_item?)
                })
                .collect::<Result<_, QpyError>>()?;
            Ok(GenericValue::Tuple(elements))
        }
        ValueType::Range => {
            let start = py_object.getattr("start")?.extract::<isize>()?;
            let stop = py_object.getattr("stop")?.extract::<isize>()?;
            let step = py_object.getattr("step")?.extract::<isize>()?;
            let step = NonZero::new(step).ok_or_else(|| {
                QpyError::InvalidParameter("range step cannot be zero".to_string())
            })?;
            let range = PyRange { start, stop, step };
            Ok(GenericValue::Range(range))
        }
        // the python-managed data types
        ValueType::NumpyObject => Ok(GenericValue::NumpyObject(py_serialize_numpy_object(
            py_object,
        )?)),
        ValueType::Modifier => Ok(GenericValue::Modifier(py_object.clone().unbind())),
        ValueType::Register => {
            if let Ok(clbit) = py_object.extract::<ShareableClbit>() {
                Ok(GenericValue::Register(ParamRegisterValue::ShareableClbit(
                    clbit,
                )))
            } else if let Ok(reg) = py_object.extract::<ClassicalRegister>() {
                Ok(GenericValue::Register(ParamRegisterValue::Register(reg)))
            } else {
                Err(QpyError::InvalidRegister(
                    "Could not read python register".to_string(),
                ))
            }
        }
    }
}

pub(crate) fn py_convert_from_generic_value(
    py: Python,
    value: &GenericValue,
) -> Result<Py<PyAny>, QpyError> {
    match value {
        GenericValue::Bool(value) => Ok(value.into_py_any(py)?),
        GenericValue::Int64(value) => Ok(value.into_py_any(py)?),
        GenericValue::Float64(value) => Ok(value.into_py_any(py)?),
        GenericValue::Complex64(value) => Ok(value.into_py_any(py)?),
        GenericValue::String(value) => Ok(value.into_py_any(py)?),
        GenericValue::Expression(exp) => Ok(exp.clone().into_py_any(py)?),
        GenericValue::CaseDefault => Ok(imports::CASE_DEFAULT.get(py).clone()),
        GenericValue::Null => Ok(py.None()),
        GenericValue::ParameterExpressionSymbol(symbol)
        | GenericValue::ParameterExpressionVectorSymbol(symbol) => {
            Ok(PyParameter(symbol.clone()).into_py_any(py)?)
        }
        GenericValue::ParameterExpression(exp) => Ok(exp.as_ref().clone().into_py_any(py)?),
        GenericValue::CircuitData(circuit_data) => {
            Ok(circuit_data.clone().into_py_quantum_circuit(py)?.unbind())
        }
        GenericValue::Modifier(py_object) => Ok(py_object.clone()),
        GenericValue::Range(py_range) => Ok(py_range.into_py_any(py)?),
        GenericValue::NumpyObject(bytes) => py_deserialize_numpy_object(py, bytes),
        GenericValue::Tuple(values) => {
            let elements: Vec<Py<PyAny>> = values
                .iter()
                .map(|v| py_convert_from_generic_value(py, v))
                .collect::<Result<_, QpyError>>()?;
            Ok(PyTuple::new(py, &elements)?.into_py_any(py)?)
        }
        GenericValue::Register(reg_value) => match reg_value {
            ParamRegisterValue::Register(reg) => Ok(reg.clone().into_py_any(py)?),
            ParamRegisterValue::ShareableClbit(clbit) => Ok(clbit.clone().into_py_any(py)?),
        },
        GenericValue::BigInt(bigint) => Ok(bigint.clone().into_py_any(py)?),
        GenericValue::Duration(duration) => Ok((*duration).into_py_any(py)?),
        GenericValue::RangeExpr(range_expr) => Ok(range_expr.clone().into_py_any(py)?),
    }
}

// This functions packs an instruction parameter, which can be an arbitrary piece of data
// Not to be confused with Parameter, which is an atom of ParameterExpression
pub(crate) fn py_pack_param(
    py_object: &Bound<PyAny>,
    qpy_data: &mut QPYWriteData,
    endian: ValueEndian,
) -> Result<formats::GenericDataPack, QpyError> {
    let value = py_convert_to_generic_value(py_object)?;
    let (type_key, data) = if endian.resolve(qpy_data.version) == Endian::Little {
        serialize_generic_value(&value.as_le(), qpy_data)?
    } else {
        serialize_generic_value(&value, qpy_data)?
    };
    Ok(formats::GenericDataPack { type_key, data })
}

pub(crate) fn py_pack_modifier(
    py: Python,
    modifier: &Py<PyAny>,
) -> Result<formats::ModifierPack, QpyError> {
    let modifier = modifier.bind(py);
    let module = py.import("qiskit.circuit.annotated_operation")?;
    if modifier.is_instance(&module.getattr("InverseModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: ModifierType::Inverse,
            num_ctrl_qubits: 0,
            ctrl_state: 0,
            power: 0.0,
        })
    } else if modifier.is_instance(&module.getattr("ControlModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: ModifierType::Control,
            num_ctrl_qubits: modifier.getattr("num_ctrl_qubits")?.extract::<u32>()?,
            ctrl_state: modifier.getattr("ctrl_state")?.extract::<u32>()?,
            power: 0.0,
        })
    } else if modifier.is_instance(&module.getattr("PowerModifier")?)? {
        Ok(formats::ModifierPack {
            modifier_type: ModifierType::Power,
            num_ctrl_qubits: 0,
            ctrl_state: 0,
            power: modifier.getattr("power")?.extract::<f64>()?,
        })
    } else {
        Err(QpyError::ConversionError(
            "Unsupported modifier".to_string(),
        ))
    }
}

pub(crate) fn py_unpack_modifier(
    py: Python,
    packed_modifier: &formats::ModifierPack,
) -> Result<Py<PyAny>, QpyError> {
    match packed_modifier.modifier_type {
        ModifierType::Inverse => Ok(imports::INVERSE_MODIFIER.get_bound(py).call0()?.unbind()),
        ModifierType::Control => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(
                intern!(py, "num_ctrl_qubits"),
                packed_modifier.num_ctrl_qubits,
            )?;
            kwargs.set_item(intern!(py, "ctrl_state"), packed_modifier.ctrl_state)?;
            Ok(imports::CONTROL_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
        ModifierType::Power => {
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "power"), packed_modifier.power)?;
            Ok(imports::POWER_MODIFIER
                .get_bound(py)
                .call((), Some(&kwargs))?
                .unbind())
        }
    }
}

fn deserialize_metadata(
    py: Python,
    metadata_bytes: &Bytes,
    metadata_deserializer: Option<&Py<PyAny>>,
) -> Result<Py<PyAny>, QpyError> {
    let json = py.import("json")?;
    let kwargs: Bound<'_, PyDict> = PyDict::new(py);
    kwargs.set_item("cls", metadata_deserializer)?;
    let metadata_string = PyString::new(py, metadata_bytes.try_into()?);
    Ok(json
        .call_method("loads", (metadata_string,), Some(&kwargs))?
        .unbind())
}

// This function finalizes the creation of QuantumCircuit from CircuitData by performing the Python-only
// required operations: handling layouts and metadata, and creating the Python QuantumCircuit object.
pub(crate) fn py_circuit_data_to_quantum_circuit<'py>(
    py: Python<'py>,
    circuit_data: CircuitData,
    packed_circuit: &formats::QPYCircuit,
    metadata_deserializer: Option<&Py<PyAny>>,
) -> Result<Bound<'py, PyAny>, QpyError> {
    let py_circuit_data: PyCircuitData = circuit_data.into();
    let unpacked_layout = unpack_layout(py, &packed_circuit.layout, &py_circuit_data)?;
    let metadata =
        deserialize_metadata(py, &packed_circuit.header.metadata, metadata_deserializer)?;
    let circuit = imports::QUANTUM_CIRCUIT
        .get_bound(py)
        .call_method1(intern!(py, "_from_circuit_data"), (py_circuit_data,))?;
    circuit.setattr("metadata", metadata)?;
    circuit.setattr("name", &packed_circuit.header.circuit_name)?;
    if let Some(layout) = unpacked_layout {
        circuit.setattr("_layout", layout)?;
    }
    Ok(circuit)
}

// This method handles all the non-standard, non-custom gates which have no rust-space implementation
pub fn unpack_py_instruction(
    py: Python,
    instruction: &formats::CircuitInstructionV2Pack,
    label: Option<&String>,
    qpy_data: &mut QPYReadData,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let name = instruction.gate_class_name.clone();
    let mut instruction_values =
        get_instruction_values(instruction, qpy_data, ValueEndian::LittleForV17AndBelow)?;
    let mut py_params: Vec<Bound<PyAny>> = instruction_values
        .iter()
        .map(|value| -> Result<_, QpyError> {
            generic_value_to_param(value, qpy_data)?
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

        "UCRXGate" | "UCRYGate" | "UCRZGate" | "DiagonalGate" => gate_class.call1((py_params,))?,
        "MCPhaseGate" | "MCU1Gate" | "MCXGrayCode" | "MCXGate" | "MCXRecursive" | "MCXVChain" => {
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
                Some(annotation_pack) => PyList::new(
                    py,
                    annotation_pack
                        .annotations
                        .iter()
                        .map(|annotation| {
                            qpy_data.annotation_handler.load_py(
                                py,
                                annotation.namespace_index,
                                annotation.payload.clone(),
                            )
                        })
                        .collect::<Result<Vec<_>, QpyError>>()?,
                )?,
                None => PyList::empty(py),
            }
            .into_any();
            let kwargs = [
                ("unit", unit),
                ("duration", duration),
                ("annotations", annotations),
            ]
            .into_py_dict(py)?;
            let args = PyTuple::new(py, &py_params)?;
            // we used the params to construct the box; they should not be retained as params except the subcircuit
            instruction_values.retain(|value| matches!(value, GenericValue::CircuitData(_)));
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
                instruction_values.retain(|value| matches!(value, GenericValue::CircuitData(_)));
            }
            gate_class.call1(args)?
        }
    };
    if let Some(label_text) = label
        && (!gate_object.hasattr("label")? || gate_object.getattr("label")?.is_none())
    {
        gate_object.setattr("label", label_text.as_str())?;
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
}

// This method handles all the custom gates written in Python (no support for Rust custom gates yet)
pub fn unpack_custom_instruction(
    py: Python,
    instruction: &formats::CircuitInstructionV2Pack,
    label: Option<&String>,
    qpy_data: &mut QPYReadData,
    custom_instructions_map: &HashMap<String, CustomCircuitInstructionData>,
) -> Result<(PackedOperation, Vec<GenericValue>), QpyError> {
    let name = instruction.gate_class_name.clone();
    let custom_instruction = custom_instructions_map.get(&name).ok_or_else(|| {
        QpyError::MissingData("Custom instruction data not found for {name}".to_string())
    })?;
    let instruction_values =
        get_instruction_values(instruction, qpy_data, ValueEndian::LittleForV17AndBelow)?;
    let py_params: Vec<Bound<PyAny>> = instruction_values
        .iter()
        .map(|value| -> Result<_, QpyError> {
            generic_value_to_param(value, qpy_data)?
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
            let packed_base_gate = deserialize_with_args::<
                formats::CircuitInstructionV2Pack,
                (bool,),
            >(&custom_instruction.base_gate_raw, (false,))?
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
            let packed_base_gate = deserialize_with_args::<
                formats::CircuitInstructionV2Pack,
                (bool,),
            >(&custom_instruction.base_gate_raw, (false,))?
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
}

pub fn deserialize_pauli_evolution_gate(
    py: Python,
    data: &Bytes,
    qpy_data: &mut QPYReadData,
) -> Result<Py<PyAny>, QpyError> {
    let json = py.import("json")?;
    let evo_synth_library = py.import("qiskit.synthesis.evolution")?;
    let (packed_data, _) =
        deserialize_with_args::<formats::PauliEvolutionDefPack, (u8,)>(data, (qpy_data.version,))?;
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
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| Complex64::new(c[0], c[1]))
                    .collect();
                let bit_terms = sparse_observable_pack
                    .bitterm_data
                    .iter()
                    .map(|&bitterm| -> Result<_, QpyError> {
                        BitTerm::try_from(bitterm).map_err(|_| {
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
                    ValueEndian::Big,
                )?;
                if let GenericValue::NumpyObject(op_raw_data) = data {
                    qpy_data.caller.attach(
                        "deserialize numpy object",
                        |py| -> Result<_, QpyError> {
                            let np_array = py_deserialize_numpy_object(py, &op_raw_data)?;
                            Ok(imports::SPARSE_PAULI_OP
                                .get_bound(py)
                                .call_method1("from_list", (np_array,))?
                                .unbind())
                        },
                    )
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
        ValueEndian::Big,
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
