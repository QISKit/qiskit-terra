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
use binrw::Endian;
use num_complex::Complex64;
use pyo3::prelude::*;
use qiskit_circuit::operations::Param;
use qiskit_circuit::parameter::parameter_expression::{
    OPReplay, ParameterExpression, ParameterValueType,
};
use qiskit_circuit::parameter::symbol_expr::{Symbol, SymbolVector};
use std::sync::{Arc, atomic};
use uuid::Uuid;

use crate::bytes::Bytes;
use crate::error::QpyError;
use crate::formats;
use crate::py_methods::{py_convert_from_generic_value, py_pack_param};
use crate::value::{
    GenericValue, QPYReadData, QPYWriteData, ValueType, deserialize, deserialize_vec, load_value,
    pack_generic_value, serialize,
};
use binrw::binrw;
use hashbrown::hash_map::HashMap;

// The various values of values that can exist in a parameter expression node
// This data is stored inside the parent of the node, not in the node itself
// So it has two "dummy" values, LhsExpression and RhsExpression indicating that
// The node has a non-leaf child that the expression reconstruction algorithm should recurse into
// In addition null represents a missing child.
// The nodes can have concrete integer/float/complex values, or be symbols (standalone/part of vector)
#[binrw]
#[brw(repr = u8)]
#[repr(u8)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ParameterType {
    /// The payload is an immediate-value integer.
    Integer = b'i',
    /// The payload is an immediate-value float.
    Float = b'f',
    /// The payload is an immediate-value complex number.
    Complex = b'c',
    /// The payload is an immediate-value UUID corresponding to a `Symbol` in the expression's map.
    Parameter = b'p',
    /// There is no immediate value; take an expression off the stack instead.
    Null = b'n',
    /// Should only occur when the "op code" type is "expression". Carries no data.
    StartExpression = b's',
    /// Should only occur when the "op code" type is "expression". Carries no data.
    EndExpression = b'e',
}

fn parameter_type_name(type_key: &ParameterType) -> String {
    String::from(match type_key {
        ParameterType::Integer => "integer",
        ParameterType::Float => "float",
        ParameterType::Complex => "complex",
        ParameterType::Parameter => "parameter",
        ParameterType::Null => "null",
        ParameterType::StartExpression => "start expression",
        ParameterType::EndExpression => "end expression",
    })
}

impl std::fmt::Display for ParameterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", parameter_type_name(self),)
    }
}

impl TryFrom<ParameterType> for ValueType {
    type Error = QpyError;
    fn try_from(value: ParameterType) -> Result<Self, Self::Error> {
        match value {
            ParameterType::Complex => Ok(ValueType::Complex),
            ParameterType::Float => Ok(ValueType::Float),
            ParameterType::Integer => Ok(ValueType::Integer),
            ParameterType::Null => Ok(ValueType::Null),
            ParameterType::Parameter => Ok(ValueType::Parameter),
            _ => Err(QpyError::ConversionError(format!(
                "Cannot convert to value type {}",
                value
            ))),
        }
    }
}

pub(crate) fn pack_parameter_expression_by_op(
    opcode: u8,
    data: formats::ParameterExpressionStandardOpPack,
) -> Result<formats::ParameterExpressionElementPack, QpyError> {
    match opcode {
        0 => Ok(formats::ParameterExpressionElementPack::Add(data)),
        1 => Ok(formats::ParameterExpressionElementPack::Sub(data)),
        2 => Ok(formats::ParameterExpressionElementPack::Mul(data)),
        3 => Ok(formats::ParameterExpressionElementPack::Div(data)),
        4 => Ok(formats::ParameterExpressionElementPack::Pow(data)),
        5 => Ok(formats::ParameterExpressionElementPack::Sin(data)),
        6 => Ok(formats::ParameterExpressionElementPack::Cos(data)),
        7 => Ok(formats::ParameterExpressionElementPack::Tan(data)),
        8 => Ok(formats::ParameterExpressionElementPack::Asin(data)),
        9 => Ok(formats::ParameterExpressionElementPack::Acos(data)),
        10 => Ok(formats::ParameterExpressionElementPack::Exp(data)),
        11 => Ok(formats::ParameterExpressionElementPack::Log(data)),
        12 => Ok(formats::ParameterExpressionElementPack::Sign(data)),
        13 => Ok(formats::ParameterExpressionElementPack::Grad(data)),
        14 => Ok(formats::ParameterExpressionElementPack::Conj(data)),
        16 => Ok(formats::ParameterExpressionElementPack::Abs(data)),
        17 => Ok(formats::ParameterExpressionElementPack::Atan(data)),
        18 => Ok(formats::ParameterExpressionElementPack::Rsub(data)),
        19 => Ok(formats::ParameterExpressionElementPack::Rdiv(data)),
        20 => Ok(formats::ParameterExpressionElementPack::Rpow(data)),
        255 => Ok(formats::ParameterExpressionElementPack::Expression(data)),
        _ => Err(QpyError::ConversionError(format!(
            "Invalid opcode: {}",
            opcode
        ))),
    }
}

// To store a parameter expression, we keep two pieces of data:
// 1) The **expression data** which is an already serialized Vec<formats::ParameterExpressionElementPack>
// which is an encoding of the replay used to reconstruct the expression, where each element encodes a specific operation
// with the operand data stored in ParameterExpressionStandardOpPack
// 2) The **symbol_table_data** where the symbols appearing anywhere in the expression are stored; only their uuid values
// are referred to in the expression data
// in older QPY versions, parameter expressions could have substitute commands, which made packing more complex
// this is no longer used in the rust-based parameter expressions, so we do not fully utilize the formats
pub(crate) fn pack_parameter_expression(
    exp: &ParameterExpression,
) -> Result<formats::ParameterExpressionPack, QpyError> {
    let packed_expression_data = pack_parameter_expression_elements(exp)?;
    let expression_data = serialize(&packed_expression_data)?;
    let symbol_table_data: Vec<formats::ParameterExpressionSymbolPack> = exp
        .iter_symbols()
        .map(pack_symbol_table_element)
        .collect::<Result<_, QpyError>>()?;
    Ok(formats::ParameterExpressionPack {
        expression_data,
        symbol_table_data,
    })
}

fn pack_symbol_table_element(
    symbol: &Symbol,
) -> Result<formats::ParameterExpressionSymbolPack, QpyError> {
    // The `value_data` part in the QPY format is only used for "substitute" commands in the replay,
    // but the Rust-space writer has no need to output those, so it's always empty for us.
    let value_data = Bytes::new();
    match symbol {
        Symbol::Standalone { .. } => {
            let pack = formats::ParameterExpressionParameterSymbolPack {
                value_key: ValueType::Parameter,
                value_data,
                symbol_data: pack_symbol(symbol),
            };
            Ok(formats::ParameterExpressionSymbolPack::Parameter(pack))
        }
        Symbol::Element { .. } => {
            let pack = formats::ParameterExpressionParameterVectorSymbolPack {
                value_key: ValueType::ParameterVector,
                value_data,
                symbol_data: pack_parameter_vector(symbol)?,
            };
            Ok(formats::ParameterExpressionSymbolPack::ParameterVector(
                pack,
            ))
        }
    }
}

fn pack_parameter_expression_elements(
    exp: &ParameterExpression,
) -> Result<Vec<formats::ParameterExpressionElementPack>, QpyError> {
    exp.qpy_replay()
        .iter()
        .map(pack_parameter_expression_element)
        .collect()
}

fn pack_parameter_expression_element(
    replay_obj: &OPReplay,
) -> Result<formats::ParameterExpressionElementPack, QpyError> {
    let (lhs_type, lhs) = pack_parameter_replay_entry(&replay_obj.lhs)?;
    let (rhs_type, rhs) = pack_parameter_replay_entry(&replay_obj.rhs)?;
    let op_code = replay_obj.op as u8;
    let entry = formats::ParameterExpressionStandardOpPack {
        lhs_type,
        lhs,
        rhs_type,
        rhs,
    };
    pack_parameter_expression_by_op(op_code, entry)
}

// this function identifies the data type of the parameter replay entry
// and returns the u8 for the type, the [u8; 16] encoding for the data (which stores)
// numbers explicitly, not using 8 bytes for f64 or u64, and storing uuid for more complex vals
// subexpressions are packed using the empty [0u8; 16], followed by "extra data" of the expression's encoding
fn pack_parameter_replay_entry(
    inst: &Option<ParameterValueType>,
) -> Result<(ParameterType, [u8; 16]), QpyError> {
    // This is different from `py_dumps_value` since we aim specifically for [u8; 16]
    // This means parameters are not fully stored, only their uuid
    // Also integers and floats are padded with 0
    let value = match inst {
        None => return Ok((ParameterType::Null, [0u8; 16])),
        Some(val) => val,
    };
    Ok(match value {
        ParameterValueType::Int(val) => (
            ParameterType::Integer,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Float(val) => (
            ParameterType::Float,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Complex(val) => (
            ParameterType::Complex,
            Bytes::from(val).try_to_16_byte_slice()?,
        ),
        ParameterValueType::Parameter(parameter) => {
            // We don't distinguish between `Parameter` and `ParameterVectorElement` in the QPY
            // format here, because we only use the UUID as a lookup anyway.
            (ParameterType::Parameter, *parameter.0.uuid().as_bytes())
        }
    })
}

pub(crate) fn unpack_parameter_expression(
    pack: &formats::ParameterExpressionPack,
    qpy_data: &mut QPYReadData,
) -> Result<ParameterExpression, QpyError> {
    let uuid_map = pack.symbol_table_data.iter().try_fold(
        HashMap::new(),
        |mut map, item| -> Result<_, QpyError> {
            let symbol = match item {
                // The QPY format (ever since V1) says that this mapping can have a value type after
                // it.  Actually, anything other than a "value" that's the zero-data case (meaning
                // it's just a `Parameter`/`ParameterVectorElement` definition) is completely
                // meaningless; all versions of Python-space Qiskit QPY loads would immediately
                // violate their own data models by constructing a `ParameterExpression` with an
                // invalid `symbol_map`. No version of Qiskit has ever _written_ a QPY file with
                // such a mapping, which is likely how it went unnoticed for so long.
                //
                // We simply treat a value mapping as an error; the semantics aren't defined.
                formats::ParameterExpressionSymbolPack::Parameter(p) => {
                    if p.value_key != ValueType::Parameter {
                        return Err(QpyError::InvalidValueType {
                            expected: "parameter".to_owned(),
                            actual: p.value_key.to_string(),
                        });
                    }
                    unpack_symbol(&p.symbol_data)
                }
                formats::ParameterExpressionSymbolPack::ParameterVector(v) => {
                    if v.value_key != ValueType::ParameterVector {
                        return Err(QpyError::InvalidValueType {
                            expected: "parameter vector element".to_owned(),
                            actual: v.value_key.to_string(),
                        });
                    }
                    unpack_parameter_vector(&v.symbol_data, qpy_data)?
                }
                // This variant should not exist; see its documentation comment.  We have to
                // silently skip it to handle loading incorrect QPY files from Qiskit 2.0 with
                // substitutions involving expressions.
                formats::ParameterExpressionSymbolPack::ParameterExpression(_) => {
                    return Ok(map);
                }
            };
            map.insert(symbol.uuid(), symbol);
            Ok(map)
        },
    )?;
    let name_map = if qpy_data.version < 15 {
        uuid_map
            .values()
            .map(|sym| (sym.fullname().into_owned(), sym.clone()))
            .collect::<HashMap<_, _>>()
    } else {
        // Not used for QPY >= 15, so no need to bother calculating it.
        Default::default()
    };

    let empty_stack = || {
        QpyError::DeserializationError(
            "malformed expression: stack was empty before expression completed".to_string(),
        )
    };
    let unknown_parameter = |repr: String| {
        QpyError::InvalidParameter(format!(
            "malformed expression: reference to unknown parameter: {repr}"
        ))
    };
    let unexpected_recursion = || {
        QpyError::DeserializationError(
            "malformed expression: encountered recursive marker in unexpected location".to_string(),
        )
    };
    let mut stack = Vec::<ParameterExpression>::new();
    let operand = |stack: &mut Vec<ParameterExpression>, ty, data: [u8; 16]| {
        let upper = |data: [u8; 16]| bytemuck::cast::<[u8; 16], [[u8; 8]; 2]>(data)[1];
        let lower = |data: [u8; 16]| bytemuck::cast::<[u8; 16], [[u8; 8]; 2]>(data)[0];
        match ty {
            ParameterType::Integer => {
                let i = i64::from_be_bytes(upper(data));
                Ok(ParameterValueType::Int(i).into())
            }
            ParameterType::Float => {
                let f = f64::from_be_bytes(upper(data));
                Ok(ParameterValueType::Float(f).into())
            }
            ParameterType::Complex => {
                let re = f64::from_be_bytes(upper(data));
                let im = f64::from_be_bytes(lower(data));
                Ok(ParameterValueType::Complex(Complex64 { re, im }).into())
            }
            ParameterType::Parameter => {
                let key = Uuid::from_bytes(data);
                uuid_map
                    .get(&key)
                    .map(|sym| ParameterExpression::from_symbol(sym.clone()))
                    .ok_or_else(|| unknown_parameter(format!("{key:?}")))
            }
            ParameterType::Null => stack.pop().ok_or_else(empty_stack),
            ParameterType::StartExpression | ParameterType::EndExpression => {
                Err(unexpected_recursion())
            }
        }
    };
    let lhs_rhs = |stack: &mut Vec<ParameterExpression>,
                   pack: &formats::ParameterExpressionStandardOpPack|
     -> Result<[ParameterExpression; 2], QpyError> {
        let rhs = operand(stack, pack.rhs_type, pack.rhs)?;
        let lhs = operand(stack, pack.lhs_type, pack.lhs)?;
        Ok([lhs, rhs])
    };
    let expr_from_value = |value| -> Result<ParameterExpression, QpyError> {
        match value {
            GenericValue::ParameterExpressionSymbol(sym)
            | GenericValue::ParameterExpressionVectorSymbol(sym) => {
                Ok(ParameterExpression::from_arc_symbol(sym))
            }
            GenericValue::ParameterExpression(expr) => Ok((*expr).clone()),
            GenericValue::Int64(_) | GenericValue::Float64(_) | GenericValue::Complex64(_) => {
                // These could/should be handled, but Python-space `ParameterExpression` had a split
                // between `subs`/`bind` (where `bind` was only for numeric values) that
                // Python-space QPY replays have never correctly generated or handled, so for now we
                // skip.  The intended semantics from the QPY specification are clear.
                Err(QpyError::DeserializationError(
                    "internal error: unhandled numeric value in substitution".to_string(),
                ))
            }
            _ => Err(QpyError::InvalidValueType {
                expected: "a parameter expression".to_string(),
                actual: "arbitrary value".to_string(),
            }),
        }
    };
    let elements =
        deserialize_vec::<formats::ParameterExpressionElementPack>(&pack.expression_data)?;
    for element in elements {
        use formats::ParameterExpressionElementPack::*;
        let out = match element {
            Add(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.add(&rhs)?
            }
            Sub(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.sub(&rhs)?
            }
            Mul(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.mul(&rhs)?
            }
            Div(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.div(&rhs)?
            }
            Pow(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.pow(&rhs)?
            }
            Sin(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.sin(),
            Cos(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.cos(),
            Tan(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.tan(),
            Asin(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.asin(),
            Acos(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.acos(),
            Exp(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.exp(),
            Log(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.log(),
            Sign(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.sign(),
            Grad(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                lhs.derivative(&rhs.try_to_symbol()?)?
            }
            Conj(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.conjugate(),
            Abs(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.abs(),
            Atan(vals) => operand(&mut stack, vals.lhs_type, vals.lhs)?.atan(),
            Rsub(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                rhs.sub(&lhs)?
            }
            Rdiv(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                rhs.div(&lhs)?
            }
            Rpow(vals) => {
                let [lhs, rhs] = lhs_rhs(&mut stack, &vals)?;
                rhs.pow(&lhs)?
            }
            Substitute(payload) => {
                let pack = deserialize::<formats::MappingPack>(&payload.mapping_data)?.0;
                let version = qpy_data.version;
                let mapping = pack
                    .items
                    .iter()
                    .map(|item| -> Result<_, QpyError> {
                        let sym = if version >= 15 {
                            let key = Uuid::from_slice(&item.key_bytes).map_err(|_| {
                                QpyError::DeserializationError(
                                    "invalid mapping: uuid incorrect length".to_string(),
                                )
                            })?;
                            uuid_map
                                .get(&key)
                                .ok_or_else(|| unknown_parameter(format!("{key:?}")))?
                                .clone()
                        } else {
                            let key = std::str::from_utf8(&item.key_bytes)?;
                            name_map
                                .get(key)
                                .ok_or_else(|| unknown_parameter(key.to_string()))?
                                .clone()
                        };
                        let replacement = expr_from_value(load_value(
                            item.item_type,
                            &item.item_bytes,
                            qpy_data,
                            Endian::Big,
                        )?)?;
                        Ok((sym, replacement))
                    })
                    .collect::<Result<HashMap<_, _>, QpyError>>()?;
                stack.pop().ok_or_else(empty_stack)?.subs(&mapping, false)?
            }
            // The "expression" payload (marking the start or end of a recursive definition) doesn't
            // actually carry any payload or have any meaning.  If we do nothing in its loop
            // iteration, we still manipulate the stack in the same way we're supposed to.
            Expression(_) => continue,
        };
        stack.push(out);
    }
    if stack.len() > 1 {
        return Err(QpyError::DeserializationError(format!(
            "malformed expression stack: {} remaining items",
            stack.len()
        )));
    }
    stack.pop().ok_or_else(empty_stack)
}

pub(crate) fn pack_symbol(symbol: &Symbol) -> formats::ParameterSymbolPack {
    let uuid = *symbol.uuid().as_bytes();
    let name = symbol.name().to_owned();
    formats::ParameterSymbolPack { uuid, name }
}

pub(crate) fn unpack_symbol(parameter_pack: &formats::ParameterSymbolPack) -> Symbol {
    let name = parameter_pack.name.clone();
    let uuid = Uuid::from_bytes(parameter_pack.uuid);
    Symbol::Standalone { name, uuid }
}

pub(crate) fn pack_parameter_vector(
    symbol: &Symbol,
) -> Result<formats::ParameterVectorElementPack, QpyError> {
    let Symbol::Element { index, base } = symbol else {
        return Err(QpyError::ConversionError(
            "internal logic error: attempted to pack a standalone symbol as a vector element"
                .to_owned(),
        ));
    };
    Ok(formats::ParameterVectorElementPack {
        vector_size: base.len.load(atomic::Ordering::Relaxed) as u64,
        uuid: *symbol.uuid().as_bytes(),
        index: *index as u64,
        name: base.name.clone(),
    })
}

/// Unpack a `ParameterVectorElement` into a `Symbol`.
///
/// Actually, the majority of the work we have to do here is unpacking the underlying _vector_ and
/// ensuring it is consistent with any other definitions of this vector that we've seen.
pub(crate) fn unpack_parameter_vector(
    pack: &formats::ParameterVectorElementPack,
    qpy_data: &mut QPYReadData,
) -> Result<Symbol, QpyError> {
    // Historical versions of Qiskit assigned independent UUIDs to every element of a vector.
    // Modern Qiskit doesn't even permit this representation; elements' UUIDs are offset from the
    // base vector's.  With certain payloads from very old Qiskit versions (pre Terra 0.25), this
    // would cause elements to disagree on the vectors that own them, but that largely shouldn't be
    // observable, and at the time of writing (2026-05-19), we don't handle old enough QPY versions
    // in Rust space for it to trigger.
    let uuid = Uuid::from_u128(u128::from_be_bytes(pack.uuid) - (pack.index as u128));
    let vector = qpy_data.vectors.entry(uuid).or_insert_with(|| {
        Arc::new(SymbolVector {
            name: pack.name.clone(),
            uuid,
            len: (pack.vector_size as usize).into(),
        })
    });
    let vector_len = vector.len.load(atomic::Ordering::Relaxed);
    if vector.name != pack.name || vector_len != pack.vector_size as usize {
        return Err(QpyError::InvalidParameter(format!(
            "'{}[{}]' has a base vector ('{}[{}]') that disagrees with another ('{}[{}]')",
            pack.name, pack.index, pack.name, pack.vector_size, vector.name, vector_len,
        )));
    }
    vector.get(pack.index as usize).ok_or_else(|| {
        QpyError::InvalidParameter(format!(
            "index {} is out of range for vector '{}[{}]'",
            pack.index, vector.name, vector_len
        ))
    })
}

pub(crate) fn pack_param_expression(
    exp: &ParameterExpression,
    qpy_data: &QPYWriteData,
) -> Result<formats::GenericDataPack, QpyError> {
    // if the parameter expression is a single symbol, we should treat it like a parameter
    // or a parameter vector, depending on whether the `vector` field exists
    if let Ok(symbol) = exp.try_to_symbol().map(Arc::new) {
        match &*symbol {
            Symbol::Standalone { .. } => {
                pack_generic_value(&GenericValue::ParameterExpressionSymbol(symbol), qpy_data)
            }
            Symbol::Element { .. } => pack_generic_value(
                &GenericValue::ParameterExpressionVectorSymbol(symbol),
                qpy_data,
            ),
        }
    } else {
        pack_generic_value(
            &GenericValue::ParameterExpression(Arc::new(exp.clone())),
            qpy_data,
        )
    }
}

pub(crate) fn pack_param_obj(
    param: &Param,
    qpy_data: &QPYWriteData,
    endian: Endian,
) -> Result<formats::GenericDataPack, QpyError> {
    Ok(match param {
        Param::Float(val) => match endian {
            Endian::Little => formats::GenericDataPack {
                type_key: ValueType::Float,
                data: val.to_le_bytes().into(),
            },
            Endian::Big => formats::GenericDataPack {
                type_key: ValueType::Float,
                data: val.to_be_bytes().into(),
            },
        },
        Param::ParameterExpression(exp) => pack_param_expression(exp, qpy_data)?,
        Param::Obj(py_object) => {
            Python::attach(|py| py_pack_param(py_object.bind(py), qpy_data, endian))?
        }
    })
}

pub(crate) fn generic_value_to_param(value: &GenericValue) -> Result<Param, QpyError> {
    match value {
        GenericValue::Float64(float_val) => Ok(Param::Float(*float_val)),
        GenericValue::ParameterExpressionSymbol(symbol) => {
            let parameter_expression = ParameterExpression::from_arc_symbol(symbol.clone());
            Ok(Param::ParameterExpression(Arc::new(parameter_expression)))
        }
        GenericValue::ParameterExpressionVectorSymbol(symbol) => {
            let parameter_expression = ParameterExpression::from_arc_symbol(symbol.clone());
            Ok(Param::ParameterExpression(Arc::new(parameter_expression)))
        }
        GenericValue::ParameterExpression(exp) => Ok(Param::ParameterExpression(exp.clone())),
        _ => Ok(Param::Obj(py_convert_from_generic_value(value)?)),
    }
}
