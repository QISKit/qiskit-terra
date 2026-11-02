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
use std::sync::atomic;

use binrw::meta::{ReadEndian, WriteEndian};
use binrw::{BinRead, BinWrite, Endian, binrw};

use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use qiskit_circuit::bit::{ClassicalRegister, ShareableClbit};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::classical::expr::{Expr, Stretch, Var};
use qiskit_circuit::classical::types::Type;
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
use crate::interface::ExtraCircuitData;
use crate::params::{
    pack_parameter_expression, pack_parameter_vector, pack_symbol, unpack_parameter_expression,
    unpack_parameter_vector, unpack_symbol,
};
use crate::py_methods::{py_pack_modifier, py_unpack_modifier};

use ndarray::Array2;
use npyz::{NpyFile, WriterBuilder};
use num_bigint::BigUint;
use num_complex::Complex64;
use std::fmt::Debug;
use std::io::Cursor;
use uuid::Uuid;

// Data that is needed globally while writing the circuit
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QpyCaller {
    Python,
    // This is consumed by the C entry point when it is compiled into the QPY crate.
    #[allow(dead_code)]
    Native,
}

impl QpyCaller {
    pub(crate) fn attach<T, E>(
        self,
        feature: &'static str,
        f: impl for<'py> FnOnce(Python<'py>) -> Result<T, E>,
    ) -> Result<T, E>
    where
        E: From<QpyError>,
    {
        if self != Self::Python {
            return Err(QpyError::PythonOnly(feature).into());
        }
        Python::attach(f)
    }
}

/// Endianness selector for QPY value serialization and deserialization.
///
/// QPY's format spec requires big-endian (network byte order) for all values.
/// However, `INSTRUCTION_PARAM` integers and floats were historically written
/// in little-endian due to an oversight that dates back to the earliest QPY
/// versions. This was acknowledged as a mistake and is corrected in v18.
///
/// Use this enum instead of `binrw::Endian` at every QPY value read/write site.
/// Call `.resolve(version)` to obtain a concrete `binrw::Endian` — all version-dispatch
/// logic lives there, so call sites never need to inspect the version themselves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum ValueEndian {
    /// Always big-endian, regardless of QPY version.
    /// Use for data that was never affected by the little-endian mistake:
    /// numpy arrays, expression values, register data, etc.
    Big,

    /// Always little-endian, regardless of QPY version.
    /// Not currently used, but included for completeness.
    #[allow(dead_code)]
    Little,

    /// Little-endian for QPY v≤17, big-endian for QPY v≥18.
    ///
    /// This is the "legacy compatibility" variant. It encodes the fact that
    /// instruction parameter integers and floats were mistakenly written in
    /// little-endian in QPY v1–17. Version 18 corrects this to big-endian.
    /// The version check is resolved by calling `.resolve(qpy_data.version)`,
    /// so callers never need to inspect the version themselves — just use this
    /// variant for all instruction scalar parameters.
    LittleForV17AndBelow,
}

impl ValueEndian {
    /// Resolve to a concrete `binrw::Endian` given the QPY format version.
    /// This is the single place where `LittleForV17AndBelow` is converted —
    /// every other file calls this instead of repeating the version comparison.
    pub(crate) fn resolve(self, version: u8) -> Endian {
        match self {
            ValueEndian::Big => Endian::Big,
            ValueEndian::Little => Endian::Little,
            ValueEndian::LittleForV17AndBelow => {
                if version >= 18 {
                    Endian::Big
                } else {
                    Endian::Little
                }
            }
        }
    }
}

// Standard char representation of register types: 'q' qreg, 'c' for creg, 'a' for ancilla reg
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq)]
pub enum RegisterType {
    Qreg = b'q',
    Creg = b'c',
    Areg = b'a',
}

impl TryFrom<u8> for RegisterType {
    type Error = QpyError;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            b'q' => Ok(Self::Qreg),
            b'c' => Ok(Self::Creg),
            b'a' => Ok(Self::Areg),
            _ => Err(QpyError::InvalidValueType {
                expected: "b'q', b'c', b'a'".to_string(),
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

// Representation for symbolic encodings (now obsolete)
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug)]
pub enum SymbolicEncoding {
    Sympy = b'p',
    Symengine = b'e',
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
#[derive(Debug, Default)]
pub struct ParameterVectorTableBuilder {
    /// Vector root UUID to its index in `vectors`.
    indices: HashMap<u128, u16>,
    vectors: Vec<Arc<SymbolVector>>,
}

impl ParameterVectorTableBuilder {
    /// The index of `vector` in the table, adding it if this is the first element to reference it.
    pub fn index_of(&mut self, vector: &Arc<SymbolVector>) -> Result<u16, QpyError> {
        if let Some(index) = self.indices.get(&vector.uuid.as_u128()) {
            return Ok(*index);
        }
        if self.vectors.len() >= u16::MAX as usize {
            return Err(QpyError::ConversionError(format!(
                "too many parameter vectors in one circuit: QPY stores at most {}",
                u16::MAX as usize
            )));
        }
        let index = self.vectors.len() as u16;
        self.indices.insert(vector.uuid.as_u128(), index);
        self.vectors.push(Arc::clone(vector));
        Ok(index)
    }

    /// The collected vectors, in index order.
    pub fn to_pack(&self) -> formats::ParameterVectorTablePack {
        formats::ParameterVectorTablePack {
            vectors: self
                .vectors
                .iter()
                .map(|vector| formats::ParameterVectorPack {
                    vector_size: vector.len.load(atomic::Ordering::Relaxed) as u64,
                    uuid: *vector.uuid.as_bytes(),
                    name: vector.name.clone(),
                })
                .collect(),
        }
    }
}

// Data that is needed globally while writing the circuit
#[derive(Debug)]
pub struct QPYWriteData<'a> {
    pub caller: QpyCaller,
    pub circuit_data: &'a CircuitData,
    pub version: u8,
    pub standalone_var_indices: HashMap<u128, u16>, // mapping from the variable's UUID to its index in the standalone variables list
    pub parameter_vectors: ParameterVectorTableBuilder,
    pub annotation_handler: AnnotationHandler,
    custom_gate_counter: u32,
}

impl<'a> QPYWriteData<'a> {
    pub fn next_custom_gate_id(&mut self) -> u32 {
        let id = self.custom_gate_counter;
        self.custom_gate_counter += 1;
        id
    }

    pub fn new(
        caller: QpyCaller,
        circuit_data: &'a CircuitData,
        version: u8,
        standalone_var_indices: HashMap<u128, u16>,
        annotation_handler: AnnotationHandler,
    ) -> Self {
        Self {
            caller,
            circuit_data,
            version,
            standalone_var_indices,
            parameter_vectors: ParameterVectorTableBuilder::default(),
            annotation_handler,
            custom_gate_counter: 0,
        }
    }
}

// Data that is needed globally while reading the circuit
#[derive(Debug)]
pub struct QPYReadData {
    pub caller: QpyCaller,
    pub circuit_data: CircuitData,
    pub version: u8,
    pub use_symengine: bool,
    pub standalone_vars: HashMap<u16, qiskit_circuit::Var>,
    pub standalone_stretches: HashMap<u16, qiskit_circuit::Stretch>,
    pub vectors: HashMap<Uuid, Arc<SymbolVector>>,
    pub parameter_vectors: Vec<Arc<SymbolVector>>,
    pub annotation_handler: AnnotationHandler,
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

// the schedule type is obsolete but needed for backwards compatibility.
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ProgramType {
    Circuit = b'q',
    Schedule = b's',
}

impl std::fmt::Display for ProgramType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            ProgramType::Circuit => "circuit",
            ProgramType::Schedule => "schedule",
        };
        write!(f, "{}", name)
    }
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
    CircuitData(Box<CircuitData>),
}

// we want to be able to extract the value relatively painlessly;
// e.g. let my_bool = value.as_typed::<bool>().unwrap()
pub trait FromGenericValue: Sized {
    fn from_generic(value: &GenericValue) -> Result<Option<Self>, QpyError>;
}

impl GenericValue {
    pub(crate) fn as_typed<T: FromGenericValue>(&self) -> Result<Option<T>, QpyError> {
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
    /// Applies the little-endian encoding workaround for QPY v≤17 instruction parameters.
    /// For v≥18 the value is returned unchanged; for v≤17 it is byte-swapped via `as_le()`.
    /// Mirrors the `ValueEndian::LittleForV17AndBelow` variant — use this at the few remaining
    /// call sites that work directly with `GenericValue` rather than going through `load_value`.
    pub(crate) fn as_little_for_v17_and_below(&self, version: u8) -> Self {
        if version < 18 {
            self.as_le()
        } else {
            self.clone()
        }
    }
    pub(crate) fn as_circuit_data(&self) -> Option<CircuitData> {
        match self {
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
    pub(crate) fn to_boolean_vec(&self) -> Result<Option<Vec<bool>>, QpyError> {
        match self {
            GenericValue::Tuple(elements) => elements
                .iter()
                .map(|val| val.as_typed::<bool>())
                .collect::<Result<Option<Vec<bool>>, _>>(),
            GenericValue::NumpyObject(bytes) => {
                let Some(npy) = NpyFile::new(Cursor::new(&bytes.0)).ok() else {
                    return Ok(None);
                };
                if npy.shape().len() != 1 {
                    return Ok(None);
                }
                Ok(npy.into_vec().ok())
            }
            _ => Ok(None),
        }
    }
    pub(crate) fn numpy_array_from_boolean_vec(values: &[bool]) -> Result<Self, QpyError> {
        let mut bytes = Vec::new();
        {
            let mut writer = npyz::WriteOptions::<bool>::new()
                .default_dtype()
                .shape(&[values.len() as u64])
                .writer(&mut bytes)
                .begin_nd()?;
            writer.extend(values.iter().copied())?;
            writer.finish()?;
        }
        Ok(Self::NumpyObject(bytes.into()))
    }

    pub(crate) fn numpy_array_from_complex_matrix(
        values: &Array2<Complex64>,
    ) -> Result<Self, QpyError> {
        let shape = values.shape();
        let mut bytes = Vec::new();
        {
            let mut writer = npyz::WriteOptions::<Complex64>::new()
                .default_dtype()
                .shape(&[shape[0] as u64, shape[1] as u64])
                .writer(&mut bytes)
                .begin_nd()?;
            writer.extend(values.iter().copied())?;
            writer.finish()?;
        }
        Ok(Self::NumpyObject(bytes.into()))
    }
}

macro_rules! impl_from_generic {
    ($t:ty, $variant:ident) => {
        impl FromGenericValue for $t {
            fn from_generic(value: &GenericValue) -> Result<Option<Self>, QpyError> {
                Ok(match value {
                    GenericValue::$variant(v) => Some(v.clone()),
                    _ => None,
                })
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
    fn from_generic(value: &GenericValue) -> Result<Option<Self>, QpyError> {
        Ok(match value {
            GenericValue::Bool(val) => Some(*val),
            GenericValue::Int64(val) => Some(*val != 0),
            _ => None,
        })
    }
}

// Extracting tuples is a little more trick; we'll use macro for the easy case of Vec<T> for a specific T
impl<T: FromGenericValue> FromGenericValue for Vec<T> {
    fn from_generic(value: &GenericValue) -> Result<Option<Self>, QpyError> {
        match value {
            GenericValue::Tuple(vec) => {
                let mut out = Vec::new();
                out.try_reserve_exact(vec.len())
                    .map_err(QpyError::AllocationError)?;
                for item in vec {
                    let Some(val) = T::from_generic(item)? else {
                        return Ok(None);
                    };
                    out.push(val);
                }
                Ok(Some(out))
            }
            _ => Ok(None),
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
    endian: ValueEndian,
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
                match endian.resolve(qpy_data.version) {
                    Endian::Little => Ok(GenericValue::Int64(i64::from_le_bytes(bytes_array))),
                    Endian::Big => Ok(GenericValue::Int64(i64::from_be_bytes(bytes_array))),
                }
            } else {
                load_biguint_value(bytes)
            }
        }
        ValueType::Float => {
            let value: f64 = bytes.try_to_f64(endian.resolve(qpy_data.version))?;
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
            let (parameter_vector_element_pack, _) = deserialize_with_args::<
                formats::ParameterVectorElementPack,
                _,
            >(bytes, (qpy_data.version,))?;
            let symbol = unpack_parameter_vector(&parameter_vector_element_pack, qpy_data)?;
            Ok(GenericValue::ParameterExpressionVectorSymbol(symbol.into()))
        }
        ValueType::ParameterExpression => {
            let (parameter_expression_pack, _) = deserialize_with_args::<
                formats::ParameterExpressionPack,
                _,
            >(bytes, (qpy_data.version,))?;
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
            let values = qpy_data
                .caller
                .attach("AnnotatedOperation", |py| -> Result<_, QpyError> {
                    py_unpack_modifier(py, &modifier_pack)
                })?;
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
                deserialize_with_args::<formats::QPYCircuit, (u8,)>(bytes, (qpy_data.version,))?;
            let circuit = unpack_circuit(
                &packed_circuit,
                qpy_data.version,
                qpy_data.use_symengine,
                qpy_data.annotation_handler.child()?,
                qpy_data.caller,
            )?;
            Ok(GenericValue::CircuitData(Box::new(circuit)))
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
    qpy_data: &mut QPYWriteData,
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
            serialize_with_args(
                &pack_parameter_vector(symbol, qpy_data)?,
                (qpy_data.version,),
            )?,
        ),
        GenericValue::ParameterExpression(exp) => (
            ValueType::ParameterExpression,
            serialize_with_args(
                &pack_parameter_expression(exp, qpy_data)?,
                (qpy_data.version,),
            )?,
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
        GenericValue::CircuitData(circuit_data) => {
            let layout = serialize(&crate::circuit_writer::pack_layout(
                None,
                circuit_data,
                qpy_data.version,
            )?)?;
            let packed_circuit = pack_circuit(
                circuit_data,
                ExtraCircuitData {
                    name: None,
                    // Circuit metadata is always decoded with `json.loads`, and older Qiskit
                    // releases require the decoded value to be a dictionary.
                    metadata: "{}".into(),
                    layout,
                },
                qpy_data.version,
                qpy_data.annotation_handler.child()?,
                qpy_data.caller,
            )?;
            (ValueType::Circuit, serialize(&packed_circuit)?)
        }
        GenericValue::NumpyObject(bytes) => (ValueType::NumpyObject, bytes.clone()),
        GenericValue::Range(py_range) => {
            let start = py_range.start as i64;
            let stop = py_range.stop as i64;
            let step = py_range.step.get() as i64;
            let range_pack = formats::RangePack { start, stop, step };
            (ValueType::Range, serialize(&range_pack)?)
        }
        GenericValue::Modifier(py_object) => {
            let packed_modifier = qpy_data
                .caller
                .attach("pack modifier", |py| -> Result<_, QpyError> {
                    py_pack_modifier(py, py_object)
                })?;
            (ValueType::Modifier, serialize(&packed_modifier)?)
        }
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
    qpy_data: &mut QPYWriteData,
) -> Result<GenericDataPack, QpyError> {
    let (type_key, data) = serialize_generic_value(value, qpy_data)?;
    Ok(GenericDataPack { type_key, data })
}

pub(crate) fn unpack_generic_value(
    value_pack: &GenericDataPack,
    qpy_data: &mut QPYReadData,
    endian: ValueEndian,
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
        _ => unpack_generic_value(value_pack, qpy_data, ValueEndian::LittleForV17AndBelow), // fallback (duration can also be expression)
    }
}

pub(crate) fn pack_for_collection(value: &ForCollection, version: u8) -> GenericValue {
    match value {
        ForCollection::List(vec) => GenericValue::Tuple(
            vec.iter()
                .map(|&val| GenericValue::Int64(val as i64).as_little_for_v17_and_below(version))
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
    qpy_data: &mut QPYWriteData,
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
    endian: ValueEndian,
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
    caller: QpyCaller,
) -> Result<CircuitInstructionType, QpyError> {
    match op.view() {
        OperationRef::StandardGate(_)
        | OperationRef::PauliProductRotation(_)
        | OperationRef::Unitary(_) => Ok(CircuitInstructionType::Gate),
        OperationRef::StandardInstruction(_)
        | OperationRef::ControlFlow(_)
        | OperationRef::PauliProductMeasurement(_)
        | OperationRef::Store(_) => Ok(CircuitInstructionType::Instruction),
        OperationRef::PyCustom(PyInstruction { kind, ob, .. }) => {
            caller.attach("Python-defined operations", |py| {
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
            })
        }
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
    version: u8,
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
fn pack_expression_type(exp_type: &Type, version: u8) -> Result<ExpressionType, QpyError> {
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
// A `Register` value stored inside an instruction is either a whole classical register or a single
// clbit.  From QPY 18 the two are told apart by a tag byte (`formats::ParamRegisterPack`); up to
// QPY 17 they shared one untyped string, a register being its bare name and a clbit being 0x00
// followed by its index in ASCII digits.

#[derive(Debug, PartialEq, Clone)]
pub enum ParamRegisterValue {
    Register(ClassicalRegister),
    ShareableClbit(ShareableClbit),
}

pub(crate) fn serialize_param_register_value(
    value: &ParamRegisterValue,
    qpy_data: &QPYWriteData,
) -> Result<Bytes, QpyError> {
    if qpy_data.version >= 18 {
        let pack = match value {
            ParamRegisterValue::Register(register) => formats::ParamRegisterPack::Register {
                name: register.name().to_string(),
            },
            ParamRegisterValue::ShareableClbit(clbit) => formats::ParamRegisterPack::Clbit {
                index: clbit_index(clbit, qpy_data)?,
            },
        };
        return serialize(&pack);
    }
    // QPY <= 17: the untyped string form.
    match value {
        ParamRegisterValue::Register(register) => Ok(register.name().into()),
        ParamRegisterValue::ShareableClbit(clbit) => {
            let name = clbit_index(clbit, qpy_data)?.to_string();
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
    if qpy_data.version >= 18 {
        let (pack, _) = deserialize::<formats::ParamRegisterPack>(bytes)?;
        return match pack {
            formats::ParamRegisterPack::Register { name } => {
                Ok(ParamRegisterValue::Register(creg_by_name(&name, qpy_data)?))
            }
            formats::ParamRegisterPack::Clbit { index } => Ok(ParamRegisterValue::ShareableClbit(
                clbit_at(index, qpy_data)?,
            )),
        };
    }
    // QPY <= 17: a leading null character means the rest is a clbit index in ASCII digits.
    if bytes.is_empty() {
        return Err(QpyError::InvalidRegister(
            "Failed to load register - name missing".to_string(),
        ));
    }
    if bytes[0] == 0u8 {
        let index: u32 =
            std::str::from_utf8(&bytes[1..])?
                .parse()
                .map_err(|e: std::num::ParseIntError| {
                    QpyError::ConversionError(format!("Failed to parse clbit index: {e}"))
                })?;
        Ok(ParamRegisterValue::ShareableClbit(clbit_at(
            index, qpy_data,
        )?))
    } else {
        // `bytes` has the register name
        let name = std::str::from_utf8(bytes)?;
        Ok(ParamRegisterValue::Register(creg_by_name(name, qpy_data)?))
    }
}

/// Position of `clbit` in the circuit being written, which is how the format refers to a bit.
///
/// A thin wrapper over [`CircuitData::clbit_index`] that turns the `None` into the QPY error, so
/// every site that needs a bit index reports the same thing.
pub(crate) fn clbit_index(
    clbit: &ShareableClbit,
    qpy_data: &QPYWriteData,
) -> Result<u32, QpyError> {
    qpy_data
        .circuit_data
        .clbit_index(clbit)
        .ok_or_else(|| QpyError::InvalidBit(format!("Could not find clbit {clbit:?} in circuit")))
}

/// The clbit at `index` in the circuit being read.
pub(crate) fn clbit_at(index: u32, qpy_data: &QPYReadData) -> Result<ShareableClbit, QpyError> {
    qpy_data
        .circuit_data
        .clbits()
        .get(Clbit(index))
        .cloned()
        .ok_or_else(|| QpyError::InvalidBit(format!("Could not find clbit {index} in circuit")))
}

/// The classical register called `name` in the circuit being read.
///
/// Uses the name-keyed [`CircuitData::cregs_data`] map rather than scanning `cregs()`.
pub(crate) fn creg_by_name(
    name: &str,
    qpy_data: &QPYReadData,
) -> Result<ClassicalRegister, QpyError> {
    qpy_data
        .circuit_data
        .cregs_data()
        .get(name)
        .cloned()
        .ok_or_else(|| {
            QpyError::InvalidRegister(format!("Could not find classical register {name:?}"))
        })
}

#[cfg(test)]
// Tests are allowed to unwrap; the crate-level deny exists for the deserializer, not for fixtures.
#[allow(clippy::unwrap_used)]
mod qpy_value_tests {
    use super::*;

    #[test]
    fn boolean_vec_numpy_object_roundtrip() {
        for values in [vec![], vec![false], vec![true, false, true]] {
            let value =
                GenericValue::numpy_array_from_boolean_vec(&values).unwrap_or(GenericValue::Null);
            assert_eq!(value.to_boolean_vec().unwrap(), Some(values));
        }
    }

    #[test]
    fn complex_matrix_numpy_object_roundtrip() -> Result<(), Box<dyn std::error::Error>> {
        let matrix = ndarray::array![
            [Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(-1.0, 0.0)],
        ];
        let GenericValue::NumpyObject(bytes) =
            GenericValue::numpy_array_from_complex_matrix(&matrix)?
        else {
            return Err("expected a numpy object".into());
        };
        let npy = NpyFile::new(Cursor::new(bytes.0))?;
        assert_eq!(npy.shape(), &[2, 2]);
        assert_eq!(npy.order(), npyz::Order::C);
        assert_eq!(
            npy.into_vec::<Complex64>()?,
            matrix.iter().copied().collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn native_caller_does_not_attach_python() {
        let result =
            QpyCaller::Native.attach("test feature", |_py| -> Result<(), QpyError> { Ok(()) });
        assert!(matches!(result, Err(QpyError::PythonOnly("test feature"))));
    }
}
