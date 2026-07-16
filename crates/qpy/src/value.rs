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

use std::num::NonZero;
use std::sync::Arc;

use binrw::meta::{ReadEndian, WriteEndian};
use binrw::{BinRead, BinWrite, Endian, binrw};
use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use qiskit_circuit::bit::{ClassicalRegister, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::classical::expr::{Expr, Stretch, Var};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::converters::QuantumCircuitData;
use qiskit_circuit::duration::Duration;
use qiskit_circuit::operations::{ForCollection, OperationRef, PyInstruction, PyOpKind, PyRange};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::{Symbol, SymbolVector};
use qiskit_circuit::{Clbit, imports};

use crate::annotations::AnnotationHandler;
use crate::bytes::Bytes;
use crate::circuit_reader::unpack_circuit;
use crate::circuit_writer::pack_circuit;
use crate::error::QpyError;
use crate::error::from_binrw_error;
use crate::formats::{self, BigIntPack, DurationPack, GenericDataPack, GenericDataSequencePack};
use crate::params::{
    pack_parameter_expression, pack_parameter_vector, pack_symbol, unpack_parameter_expression,
    unpack_parameter_vector, unpack_symbol,
};
use crate::py_methods::{py_pack_modifier, py_unpack_modifier};

use npyz::NpyFile;
use num_bigint::BigUint;
use num_complex::Complex64;
use std::fmt::Debug;
use std::io::Cursor;
use uuid::Uuid;

// Standard char representation of register types: 'q' qreg, 'c' for creg
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq)]
pub enum RegisterType {
    Qreg = b'q',
    Creg = b'c',
}

impl TryFrom<u8> for RegisterType {
    type Error = QpyError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'q' => Ok(Self::Qreg),
            b'c' => Ok(Self::Creg),
            _ => Err(QpyError::InvalidValueType {
                expected: "b'q', b'c'".to_string(),
                actual: value.to_string(),
            }),
        }
    }
}

// Representation for qubit/clbit
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug)]
pub enum BitType {
    Qubit = b'q',
    Clbit = b'c',
}

impl TryFrom<u8> for BitType {
    type Error = QpyError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'q' => Ok(Self::Qubit),
            b'c' => Ok(Self::Clbit),
            _ => Err(QpyError::InvalidValueType {
                expected: "b'q', b'c'".to_string(),
                actual: value.to_string(),
            }),
        }
    }
}

pub(crate) fn pack_biguint(bigint: &BigUint) -> BigIntPack {
    let bytes = Bytes(bigint.to_bytes_be());
    BigIntPack { bytes }
}

pub(crate) fn unpack_biguint(big_int_pack: BigIntPack) -> BigUint {
    BigUint::from_bytes_be(&big_int_pack.bytes)
}

// Data that is needed globally while writing the circuit
#[derive(Debug)]
pub struct QPYWriteData<'a> {
    pub circuit_data: &'a CircuitData,
    pub version: u32,
    pub standalone_var_indices: HashMap<u128, u16>, // mapping from the variable's UUID to its index in the standalone variables list
    pub annotation_handler: AnnotationHandler<'a>,
}

// Data that is needed globally while reading the circuit
#[derive(Debug)]
pub struct QPYReadData<'a> {
    pub circuit_data: &'a mut CircuitData,
    pub version: u32,
    pub use_symengine: bool,
    pub standalone_vars: HashMap<u16, qiskit_circuit::Var>,
    pub standalone_stretches: HashMap<u16, qiskit_circuit::Stretch>,
    pub vectors: HashMap<Uuid, Arc<SymbolVector>>,
    pub annotation_handler: AnnotationHandler<'a>,
}

// this is how tags for various value types are encoded in a QPY file
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ValueType {
    Bool = b'b',
    Integer = b'i', // this is also used by "BigInt" which may arise in more specialized contexts
    Float = b'f',
    Complex = b'c',
    CaseDefault = b'd',
    Register = b'R',
    Range = b'r',
    Tuple = b't', // this is also used by "Duration" which may arise in more specialized contexts
    NumpyObject = b'n',
    Parameter = b'p',
    ParameterVector = b'v',
    ParameterExpression = b'e',
    String = b's',
    Null = b'z',
    Expression = b'x',
    Modifier = b'm',
    Circuit = b'q',
}

pub(crate) fn type_name(type_key: &ValueType) -> String {
    String::from(match type_key {
        ValueType::Bool => "boolean",
        ValueType::Integer => "integer",
        ValueType::Float => "float",
        ValueType::Complex => "complex",
        ValueType::CaseDefault => "case default",
        ValueType::Register => "register",
        ValueType::Range => "range",
        ValueType::Tuple => "tuple",
        ValueType::NumpyObject => "numpy object",
        ValueType::Parameter => "parameter",
        ValueType::ParameterVector => "parameter vector",
        ValueType::ParameterExpression => "parameter expression",
        ValueType::String => "string",
        ValueType::Null => "null",
        ValueType::Expression => "expression",
        ValueType::Modifier => "modifier",
        ValueType::Circuit => "circuit",
    })
}

impl std::fmt::Display for ValueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", type_name(self),)
    }
}

#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug)]
pub enum ModifierType {
    Inverse = b'i',
    Control = b'c',
    Power = b'p',
}

// The types of nodes inside Expressions (not to be confused with ParameterExpressions)
#[binrw]
#[derive(Debug)]
pub enum ExpressionType {
    #[brw(magic = b'b')]
    Bool,
    #[brw(magic = b'u')]
    Uint(u32),
    #[brw(magic = b'f')]
    Float,
    #[brw(magic = b'd')]
    Duration,
}

// The scope of nodes inside Expressions (not to be confused with ParameterExpressions)
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug)]
pub enum ExpressionVarDeclaration {
    Input = b'I',
    Capture = b'C',
    Local = b'L',
    StretchCapture = b'A',
    StretchLocal = b'O',
}

// The various types of circuit instructions
// This is used to differentiate special cases that need separate handling
// Like pauli evolution and controlled gates, from the standard cases ("instruction/gate")
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum CircuitInstructionType {
    Instruction = b'i',
    Gate = b'g',
    PauliEvolutionGate = b'p',
    ControlledGate = b'c',
    AnnotatedOperation = b'a',
}

pub(crate) fn serialize<T>(value: &T) -> Result<Bytes, QpyError>
where
    T: BinWrite + WriteEndian + Debug,
    for<'a> <T as BinWrite>::Args<'a>: Default,
{
    let mut buffer = Cursor::new(Vec::new());
    value.write(&mut buffer)?;
    Ok(buffer.into())
}
pub(crate) fn serialize_with_args<T, A>(value: &T, args: A) -> Result<Bytes, QpyError>
where
    T: BinWrite<Args<'static> = A> + WriteEndian + Debug,
    A: Clone + Debug,
{
    let mut buffer = Cursor::new(Vec::new());
    value
        .write_args(&mut buffer, args)
        .map_err(from_binrw_error)?;
    Ok(buffer.into())
}

pub(crate) fn deserialize<T>(bytes: &[u8]) -> Result<(T, usize), QpyError>
where
    T: BinRead<Args<'static> = ()> + Debug,
{
    let mut cursor = Cursor::new(bytes);
    let value = T::read_options(&mut cursor, Endian::Big, ())?;
    let bytes_read = cursor.position() as usize;
    Ok((value, bytes_read))
}

pub(crate) fn deserialize_with_args<'a, T, A>(bytes: &[u8], args: A) -> Result<(T, usize), QpyError>
where
    T: BinRead<Args<'a> = A> + ReadEndian + Debug,
{
    let mut cursor = Cursor::new(bytes);
    let value = T::read_args(&mut cursor, args)?;
    let bytes_read = cursor.position() as usize;
    Ok((value, bytes_read))
}

pub(crate) fn deserialize_vec<T>(mut bytes: &[u8]) -> Result<Vec<T>, QpyError>
where
    T: BinRead<Args<'static> = ()> + Debug,
{
    let mut result = Vec::new();
    while !bytes.is_empty() {
        let (item, pos) = deserialize::<T>(bytes)?;
        result.push(item);
        bytes = &bytes[pos..];
    }
    Ok(result)
}

// An enum for all data that might be found in a serialized ("Bytes") form inside the qpy file
// Under ideal conditions, we won't need such an enum since every rust based value will be present explicitly
// In the formats.rs file. However, due to the legacy of how Python-based value were stored such explicit
// representation is not always possible or difficult to implement
// This is the pure rust alternative to `DumpedPyValue` which is used to serialize data given via python objects
#[derive(Clone, Debug)]
pub enum GenericValue {
    Bool(bool),
    Int64(i64),
    BigInt(BigUint),
    Float64(f64),
    Complex64(Complex64),
    CaseDefault,
    Register(ParamRegisterValue), // This is not the full register data; rather, it's the name stored inside instructions, or a clbit address
    Range(PyRange),
    Tuple(Vec<GenericValue>),
    NumpyObject(Bytes),
    ParameterExpressionSymbol(Arc<Symbol>),
    ParameterExpressionVectorSymbol(Arc<Symbol>),
    ParameterExpression(Arc<ParameterExpression>),
    String(String),
    Duration(Duration),
    Null,
    Expression(Expr),
    Modifier(Py<PyAny>),
    Circuit(Py<PyAny>), // currently we have no rust class corresponding to a circuit, only to the inner CircuitData
    CircuitData(Box<CircuitData>),
}

// we want to be able to extract the value relatively painlessly;
// e.g. let my_bool = value.as_typed::<bool>().unwrap()
pub trait FromGenericValue: Sized {
    fn from_generic(value: &GenericValue) -> Option<Self>;
}

impl GenericValue {
    pub(crate) fn as_typed<T: FromGenericValue>(&self) -> Option<T> {
        T::from_generic(self)
    }
    // reintreprets int64 and float64 as if they were given in little endian, since this is needed when encoding instruction parameters
    pub(crate) fn as_le(&self) -> Self {
        match self {
            GenericValue::Int64(value) => {
                GenericValue::Int64(i64::from_le_bytes(value.to_be_bytes()))
            }
            GenericValue::Float64(value) => {
                GenericValue::Float64(f64::from_le_bytes(value.to_be_bytes()))
            }
            GenericValue::Tuple(elements) => {
                GenericValue::Tuple(elements.iter().map(GenericValue::as_le).collect())
            }
            _ => self.clone(),
        }
    }
    pub(crate) fn as_circuit_data(&self) -> Option<CircuitData> {
        match self {
            GenericValue::Circuit(py_circuit) => {
                Python::attach(|py| -> Result<CircuitData, QpyError> {
                    Ok(py_circuit.extract::<QuantumCircuitData>(py)?.data)
                })
                .ok()
            }
            GenericValue::CircuitData(circuit_data) => Some(circuit_data.as_ref().clone()),
            _ => None,
        }
    }
    // return the inner GenericValue slice when the GenericData is a Tuple
    pub(crate) fn as_slice(&self) -> Option<&[GenericValue]> {
        match self {
            GenericValue::Tuple(elements) => Some(elements),
            _ => None,
        }
    }
    pub(crate) fn to_vec(&self) -> Option<Vec<GenericValue>> {
        match self {
            GenericValue::Tuple(elements) => Some(elements.clone()),
            _ => None,
        }
    }
    // boolean vectors are tricky since there are several ways to encode them
    pub(crate) fn to_boolean_vec(&self) -> Option<Vec<bool>> {
        match self {
            GenericValue::Tuple(elements) => elements
                .iter()
                .map(|val| val.as_typed::<bool>())
                .collect::<Option<Vec<bool>>>(),
            GenericValue::NumpyObject(bytes) => {
                let npy = NpyFile::new(Cursor::new(&bytes.0)).ok()?;
                if npy.shape().len() != 1 {
                    return None;
                }
                npy.into_vec().ok()
            }
            _ => None,
        }
    }
}

macro_rules! impl_from_generic {
    ($t:ty, $variant:ident) => {
        impl FromGenericValue for $t {
            fn from_generic(value: &GenericValue) -> Option<Self> {
                match value {
                    GenericValue::$variant(v) => Some(v.clone()),
                    _ => None,
                }
            }
        }
    };
}

impl_from_generic!(i64, Int64);
impl_from_generic!(f64, Float64);
impl_from_generic!(BigUint, BigInt);
impl_from_generic!(String, String);
impl_from_generic!(Complex64, Complex64);
// we do not implement Symbol extraction, since it is ambiguous - a symbol can be a Parameter or a ParameterVector

// booleans are stored as i64 in current QPY, we should be able to convert from them
impl FromGenericValue for bool {
    fn from_generic(value: &GenericValue) -> Option<Self> {
        match value {
            GenericValue::Bool(val) => Some(*val),
            GenericValue::Int64(val) => Some(*val != 0),
            _ => None,
        }
    }
}

// Extracting tuples is a little more trick; we'll use macro for the easy case of Vec<T> for a specific T
impl<T: FromGenericValue> FromGenericValue for Vec<T> {
    fn from_generic(value: &GenericValue) -> Option<Self> {
        match value {
            GenericValue::Tuple(vec) => {
                let mut out = Vec::with_capacity(vec.len());
                for item in vec {
                    out.push(T::from_generic(item)?);
                }
                Some(out)
            }
            _ => None,
        }
    }
}

/// Load a bytes array value of a specified type
///
/// # Args
///
/// * `type_key` - The type of the data
/// * `bytes` - The raw data as a u8 array
/// * `qpy_data` - QPY reader metadata
/// * `endian` - The endianess of the data used only for reading
///   `ValueType::Integer` and `ValueType::Float` primitive types
///   (this applies recursively for `ValueType::Tuple` too). All other
///   data is big endian per the QPY format documentation. The use of
///   little endian data for floats and integers in instruction parameter
///   contexts only is an oversight/mistake in the QPY implementation for
///   format versions <=17. All data is supposed to be network byte order.
pub(crate) fn load_value(
    type_key: ValueType,
    bytes: &Bytes,
    qpy_data: &mut QPYReadData,
    endian: Endian,
) -> Result<GenericValue, QpyError> {
    match type_key {
        ValueType::Bool => {
            let value: bool = bytes.try_into()?;
            Ok(GenericValue::Bool(value))
        }
        ValueType::Integer => {
            // a little tricky since this can be either i64 or biguint
            if bytes.len() <= 8 {
                let mut bytes_array: [u8; 8] = [0; 8];
                for (idx, byte) in bytes.iter().enumerate() {
                    bytes_array[idx] = *byte;
                }
                match endian {
                    Endian::Little => Ok(GenericValue::Int64(i64::from_le_bytes(bytes_array))),
                    Endian::Big => Ok(GenericValue::Int64(i64::from_be_bytes(bytes_array))),
                }
            } else {
                load_biguint_value(bytes)
            }
        }
        ValueType::Float => {
            let value: f64 = bytes.try_to_f64(endian)?;
            Ok(GenericValue::Float64(value))
        }
        ValueType::Complex => {
            let value: Complex64 = bytes.try_into()?;
            Ok(GenericValue::Complex64(value))
        }
        ValueType::String => {
            let value: String = bytes.clone().try_into()?;
            Ok(GenericValue::String(value))
        }
        ValueType::Range => {
            let range_pack = deserialize::<formats::RangePack>(bytes)?.0;
            let start = range_pack.start as isize;
            let stop = range_pack.stop as isize;
            let step = NonZero::new(range_pack.step as isize).ok_or_else(|| {
                QpyError::DeserializationError("range stpe cannot be zero".into())
            })?;
            let py_range = PyRange { start, stop, step };
            Ok(GenericValue::Range(py_range))
        }
        ValueType::Parameter => {
            let (parameter_pack, _) = deserialize::<formats::ParameterSymbolPack>(bytes)?;
            let symbol = unpack_symbol(&parameter_pack);
            Ok(GenericValue::ParameterExpressionSymbol(symbol.into()))
        }
        ValueType::ParameterVector => {
            let (parameter_vector_element_pack, _) =
                deserialize::<formats::ParameterVectorElementPack>(bytes)?;
            let symbol = unpack_parameter_vector(&parameter_vector_element_pack, qpy_data)?;
            Ok(GenericValue::ParameterExpressionVectorSymbol(symbol.into()))
        }
        ValueType::ParameterExpression => {
            let (parameter_expression_pack, _) =
                deserialize::<formats::ParameterExpressionPack>(bytes)?;
            let exp = unpack_parameter_expression(&parameter_expression_pack, qpy_data)?;
            Ok(GenericValue::ParameterExpression(Arc::new(exp)))
        }
        ValueType::Tuple => {
            let (elements_pack, _) = deserialize::<GenericDataSequencePack>(bytes)?;
            let values = unpack_generic_value_sequence(elements_pack, qpy_data, endian)?;
            Ok(GenericValue::Tuple(values))
        }
        ValueType::NumpyObject => Ok(GenericValue::NumpyObject(bytes.clone())),
        ValueType::Modifier => {
            let (modifier_pack, _) = deserialize::<formats::ModifierPack>(bytes)?;
            let values = py_unpack_modifier(&modifier_pack)?;
            Ok(GenericValue::Modifier(values))
        }
        ValueType::Expression => {
            let expression = deserialize_expression(bytes, qpy_data)?;
            Ok(GenericValue::Expression(expression))
        }
        ValueType::Null => Ok(GenericValue::Null),
        ValueType::CaseDefault => Ok(GenericValue::CaseDefault),
        ValueType::Register => {
            let register_value = load_param_register_value(bytes, qpy_data)?;
            Ok(GenericValue::Register(register_value))
        }
        ValueType::Circuit => {
            let (packed_circuit, _) =
                deserialize_with_args::<formats::QPYCircuit, (u32,)>(bytes, (qpy_data.version,))?;
            Python::attach(|py| {
                let circuit = unpack_circuit(
                    py,
                    &packed_circuit,
                    qpy_data.version,
                    None,
                    qpy_data.use_symengine,
                    qpy_data.annotation_handler.annotation_factories,
                )?;
                Ok(GenericValue::Circuit(circuit))
            })
        }
    }
}

// a specialized method used for biguints (marked by 'i' like Int64)
// since the general load method will attempt to load a Int64 instead
pub(crate) fn load_biguint_value(bytes: &Bytes) -> Result<GenericValue, QpyError> {
    let (bigint_pack, _) = deserialize::<BigIntPack>(bytes)?;
    let bigint = unpack_biguint(bigint_pack);
    Ok(GenericValue::BigInt(bigint))
}

/// serializes the generic value into bytes and also returns the identifying tag
pub(crate) fn serialize_generic_value(
    value: &GenericValue,
    qpy_data: &QPYWriteData,
) -> Result<(ValueType, Bytes), QpyError> {
    Ok(match value {
        GenericValue::Bool(value) => (ValueType::Bool, value.into()),
        GenericValue::Int64(value) => (ValueType::Integer, value.into()),
        GenericValue::BigInt(bigint) => (ValueType::Integer, serialize(&pack_biguint(bigint))?),
        GenericValue::Float64(value) => (ValueType::Float, value.into()),
        GenericValue::Complex64(value) => (ValueType::Complex, value.into()),
        GenericValue::String(value) => (ValueType::String, value.into()),
        GenericValue::CaseDefault => (ValueType::CaseDefault, Bytes::new()),
        GenericValue::ParameterExpressionSymbol(symbol) => {
            (ValueType::Parameter, serialize(&pack_symbol(symbol))?)
        }
        GenericValue::ParameterExpressionVectorSymbol(symbol) => (
            ValueType::ParameterVector,
            serialize(&pack_parameter_vector(symbol)?)?,
        ),
        GenericValue::ParameterExpression(exp) => (
            ValueType::ParameterExpression,
            serialize(&pack_parameter_expression(exp)?)?,
        ),
        GenericValue::Tuple(values) => (
            ValueType::Tuple,
            serialize(&pack_generic_value_sequence(values, qpy_data)?)?,
        ),
        GenericValue::Duration(duration) => {
            if qpy_data.version < 16 && matches!(duration, Duration::ps(_)) {
                return Err(QpyError::UnsupportedFeatureForVersion {
                    feature: String::from("Duration variant 'Duration.ps'"),
                    version: 16,
                    min_version: qpy_data.version,
                });
            }
            (
                ValueType::Tuple, // due to historical reasons, 't' is shared between these data types
                serialize(&pack_duration(duration))?,
            )
        }
        GenericValue::Expression(exp) => {
            (ValueType::Expression, serialize_expression(exp, qpy_data)?)
        }
        GenericValue::Null => (ValueType::Null, Bytes::new()),
        GenericValue::Circuit(circuit) => Python::attach(|py| -> Result<_, QpyError> {
            let packed_circuit = pack_circuit(
                &mut circuit.extract(py)?, // TODO: can we avoid cloning here?
                None,
                false,
                qpy_data.version,
                qpy_data.annotation_handler.annotation_factories,
            )?;
            let serialized_circuit = serialize(&packed_circuit)?;
            Ok((ValueType::Circuit, serialized_circuit))
        })?,
        GenericValue::CircuitData(circuit_data) => Python::attach(|py| -> PyResult<_> {
            let mut quantum_circuit_data = circuit_data
                .clone()
                .into_py_quantum_circuit(py)?
                .extract()?;
            let packed_circuit = pack_circuit(
                &mut quantum_circuit_data,
                None,
                false,
                qpy_data.version,
                qpy_data.annotation_handler.annotation_factories,
            )?;
            let serialized_circuit = serialize(&packed_circuit)?;
            Ok((ValueType::Circuit, serialized_circuit))
        })?,
        GenericValue::NumpyObject(bytes) => (ValueType::NumpyObject, bytes.clone()),
        GenericValue::Range(py_range) => {
            let start = py_range.start as i64;
            let stop = py_range.stop as i64;
            let step = py_range.step.get() as i64;
            let range_pack = formats::RangePack { start, stop, step };
            (ValueType::Range, serialize(&range_pack)?)
        }
        GenericValue::Modifier(py_object) => (
            ValueType::Modifier,
            serialize(&py_pack_modifier(py_object)?)?,
        ),
        GenericValue::Register(param_register_value) => (
            ValueType::Register,
            serialize_param_register_value(param_register_value, qpy_data)?,
        ),
    })
}

// packing to GenericDataPack is somewhat wasteful in many cases, since given the type_key
// we usually know the byte length of the data and don't need to store it directly,
// but since that's the format currently in place in QPY we don't try to optimize
pub(crate) fn pack_generic_value(
    value: &GenericValue,
    qpy_data: &QPYWriteData,
) -> Result<GenericDataPack, QpyError> {
    let (type_key, data) = serialize_generic_value(value, qpy_data)?;
    Ok(GenericDataPack { type_key, data })
}

pub(crate) fn unpack_generic_value(
    value_pack: &GenericDataPack,
    qpy_data: &mut QPYReadData,
    endian: Endian,
) -> Result<GenericValue, QpyError> {
    let result = load_value(value_pack.type_key, &value_pack.data, qpy_data, endian)?;
    Ok(result)
}

/// dedicated method for handling the special case where the data pack encodes a `Duration` value
/// this cannot be determined automatically since in the current QPY format, both duration and tuple
/// share the 't' key.
pub(crate) fn unpack_duration_value(
    value_pack: &GenericDataPack,
    qpy_data: &mut QPYReadData,
) -> Result<GenericValue, QpyError> {
    match value_pack.type_key {
        ValueType::Tuple => {
            let duration = unpack_duration(deserialize::<DurationPack>(&value_pack.data)?.0);
            Ok(GenericValue::Duration(duration))
        }
        _ => unpack_generic_value(value_pack, qpy_data, Endian::Little), // fallback (duration can also be expression)
    }
}

pub(crate) fn pack_for_collection(value: &ForCollection) -> GenericValue {
    match value {
        ForCollection::List(vec) => GenericValue::Tuple(
            vec.iter()
                .map(|&val| GenericValue::Int64(val as i64).as_le())
                .collect(),
        ),
        ForCollection::PyRange(py_range) => GenericValue::Range(*py_range),
    }
}

pub(crate) fn unpack_for_collection(value: &GenericValue) -> Result<ForCollection, QpyError> {
    match value {
        GenericValue::Range(py_range) => Ok(ForCollection::PyRange(*py_range)),
        GenericValue::Tuple(vec) => {
            let value_list = vec
                .iter()
                .map(|val| -> Result<_, QpyError> {
                    if let GenericValue::Int64(int_val) = val {
                        Ok(*int_val as isize)
                    } else {
                        Err(QpyError::ConversionError(
                            "Could not unpack ForCollection: expected Int64 in tuple".to_string(),
                        ))
                    }
                })
                .collect::<Result<_, QpyError>>()?;
            Ok(ForCollection::List(value_list))
        }
        _ => Err(QpyError::ConversionError(format!(
            "Could not unpack ForCollection: expected Range or Tuple, got {:?}",
            value
        ))),
    }
}

pub(crate) fn pack_generic_value_sequence(
    values: &[GenericValue],
    qpy_data: &QPYWriteData,
) -> Result<GenericDataSequencePack, QpyError> {
    let elements = values
        .iter()
        .map(|value| pack_generic_value(value, qpy_data))
        .collect::<Result<_, QpyError>>()?;
    Ok(GenericDataSequencePack { elements })
}

pub(crate) fn unpack_generic_value_sequence(
    value_seqeunce_pack: GenericDataSequencePack,
    qpy_data: &mut QPYReadData,
    endian: Endian,
) -> Result<Vec<GenericValue>, QpyError> {
    value_seqeunce_pack
        .elements
        .iter()
        .map(|data_pack| unpack_generic_value(data_pack, qpy_data, endian))
        .collect()
}

/// Each instruction type has a char representation in qpy
pub(crate) fn get_circuit_type_key(
    op: &PackedOperation,
) -> Result<CircuitInstructionType, QpyError> {
    match op.view() {
        OperationRef::StandardGate(_)
        | OperationRef::PauliProductRotation(_)
        | OperationRef::Unitary(_) => Ok(CircuitInstructionType::Gate),
        OperationRef::StandardInstruction(_)
        | OperationRef::ControlFlow(_)
        | OperationRef::PauliProductMeasurement(_) => Ok(CircuitInstructionType::Instruction),
        OperationRef::PyCustom(PyInstruction { kind, ob, .. }) => Python::attach(|py| {
            let ob = ob.bind(py);
            match kind {
                PyOpKind::Instruction => Ok(CircuitInstructionType::Instruction),
                PyOpKind::Gate => {
                    if ob.is_instance(imports::PAULI_EVOLUTION_GATE.get_bound(py))? {
                        Ok(CircuitInstructionType::PauliEvolutionGate)
                    } else if ob.is_instance(imports::CONTROLLED_GATE.get_bound(py))? {
                        Ok(CircuitInstructionType::ControlledGate)
                    } else {
                        Ok(CircuitInstructionType::Gate)
                    }
                }
                PyOpKind::Operation => {
                    if ob.is_instance(imports::ANNOTATED_OPERATION.get_bound(py))? {
                        Ok(CircuitInstructionType::AnnotatedOperation)
                    } else {
                        Err(QpyError::InvalidInstruction(format!(
                            "Unable to determine circuit type key for {ob:?}"
                        )))
                    }
                }
            }
        }),
        OperationRef::CustomOperation(custom_gate) => match custom_gate.is_controlled_gate() {
            true => Ok(CircuitInstructionType::ControlledGate),
            false => match custom_gate.is_unitary() {
                true => Ok(CircuitInstructionType::Gate),
                false => Ok(CircuitInstructionType::Instruction),
            },
        },
    }
}

pub(crate) fn serialize_expression(exp: &Expr, qpy_data: &QPYWriteData) -> Result<Bytes, QpyError> {
    let packed_expression = formats::ExpressionPack {
        expression: exp.clone(),
        _phantom: Default::default(),
    };
    serialize_with_args(&packed_expression, (qpy_data,))
}

pub(crate) fn deserialize_expression(
    raw_expression: &Bytes,
    qpy_data: &QPYReadData,
) -> Result<Expr, QpyError> {
    let (exp_pack, _) = deserialize_with_args::<formats::ExpressionPack, (&QPYReadData,)>(
        raw_expression,
        (qpy_data,),
    )?;
    Ok(exp_pack.expression)
}

pub(crate) fn pack_standalone_var(
    var: &Var,
    usage: ExpressionVarDeclaration,
    version: u32,
    uuid_output: &mut u128,
) -> Result<formats::ExpressionVarDeclarationPack, QpyError> {
    match var {
        Var::Standalone { uuid, name, ty } => {
            let exp_type = pack_expression_type(ty, version)?;
            *uuid_output = *uuid;
            let uuid_bytes = uuid.to_be_bytes();
            Ok(formats::ExpressionVarDeclarationPack {
                uuid_bytes,
                usage,
                exp_type,
                name: name.clone(),
            })
        }
        _ => Err(QpyError::InvalidExpression(format!(
            "attempted to pack as standalone var the non-standalone var {:?}",
            var
        ))),
    }
}

pub(crate) fn pack_stretch(
    stretch: &Stretch,
    usage: ExpressionVarDeclaration,
) -> formats::ExpressionVarDeclarationPack {
    formats::ExpressionVarDeclarationPack {
        uuid_bytes: stretch.uuid.to_be_bytes(),
        usage,
        exp_type: ExpressionType::Duration,
        name: stretch.name.clone(),
    }
}

// we convert the type to the serializable struct; this amounts to simple copy unless
// there's a field not supported in the expected version
fn pack_expression_type(exp_type: &Type, version: u32) -> Result<ExpressionType, QpyError> {
    match exp_type {
        Type::Bool => Ok(ExpressionType::Bool),
        Type::Duration => {
            if version >= 14 {
                Ok(ExpressionType::Duration)
            } else {
                Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "duration-typed expressions".to_string(),
                    version: 14,
                    min_version: version,
                })
            }
        }
        Type::Float => {
            if version >= 14 {
                Ok(ExpressionType::Float)
            } else {
                Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "float-typed expressions".to_string(),
                    version: 14,
                    min_version: version,
                })
            }
        }
        Type::Uint(width) => Ok(ExpressionType::Uint(*width)),
    }
}

pub(crate) fn pack_duration(duration: &Duration) -> DurationPack {
    match duration {
        Duration::dt(dt) => DurationPack::DT(*dt as u64),
        Duration::ps(ps) => DurationPack::PS(*ps),
        Duration::ns(ns) => DurationPack::NS(*ns),
        Duration::us(us) => DurationPack::US(*us),
        Duration::ms(ms) => DurationPack::MS(*ms),
        Duration::s(s) => DurationPack::S(*s),
    }
}

pub(crate) fn unpack_duration(duration_pack: DurationPack) -> Duration {
    match duration_pack {
        DurationPack::DT(dt) => Duration::dt(dt as i64),
        DurationPack::PS(ps) => Duration::ps(ps),
        DurationPack::NS(ns) => Duration::ns(ns),
        DurationPack::US(us) => Duration::us(us),
        DurationPack::MS(ms) => Duration::ms(ms),
        DurationPack::S(s) => Duration::s(s),
    }
}

// due to historical reasons, the treatment of instructions params which are registers/clbits is a little strange
// When a register is stored as an instruction param, it is serialized compactly
// For a classical register its name is saved as a string; for a clbit
// its index in the full clbit list is converted into a string, with 0x00 appended at the start
// to differentiate from the register case

#[derive(Debug, PartialEq, Clone)]
pub enum ParamRegisterValue {
    Register(ClassicalRegister),
    ShareableClbit(ShareableClbit),
}

pub(crate) fn serialize_param_register_value(
    value: &ParamRegisterValue,
    qpy_data: &QPYWriteData,
) -> Result<Bytes, QpyError> {
    match value {
        ParamRegisterValue::Register(register) => Ok(register.name().into()),
        ParamRegisterValue::ShareableClbit(clbit) => {
            let name = qpy_data
                .circuit_data
                .clbits()
                .find(clbit)
                .ok_or_else(|| QpyError::InvalidBit("clbit not found".to_string()))?
                .0
                .to_string();
            // this is the part where we get hack-y
            let mut bytes: Bytes = Bytes(Vec::with_capacity(name.len() + 1));
            bytes.push(0u8);
            bytes.extend_from_slice(name.as_bytes());
            Ok(bytes)
        }
    }
}

pub(crate) fn load_param_register_value(
    bytes: &Bytes,
    qpy_data: &mut QPYReadData,
) -> Result<ParamRegisterValue, QpyError> {
    // If register name prefixed with null character it's a clbit index for single bit condition.
    if bytes.is_empty() {
        return Err(QpyError::InvalidRegister(
            "Failed to load register - name missing".to_string(),
        ));
    }
    if bytes[0] == 0u8 {
        let index = Clbit(std::str::from_utf8(&bytes[1..])?.parse().map_err(
            |e: std::num::ParseIntError| {
                QpyError::ConversionError(format!("Failed to parse clbit index: {}", e))
            },
        )?);
        match qpy_data.circuit_data.clbits().get(index) {
            Some(shareable_clbit) => {
                Ok(ParamRegisterValue::ShareableClbit(shareable_clbit.clone()))
            }
            None => Err(QpyError::InvalidBit(format!(
                "Could not find clbit {:?}",
                index
            ))),
        }
    } else {
        // `bytes` has the register name
        let name = std::str::from_utf8(bytes)?;
        for creg in qpy_data.circuit_data.cregs() {
            if creg.name() == name {
                return Ok(ParamRegisterValue::Register(creg.clone()));
            }
        }
        Err(QpyError::InvalidRegister(format!(
            "Could not find classical register {:?}",
            name
        )))
    }
}
