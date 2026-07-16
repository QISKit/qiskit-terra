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

// methods for serialization/deserialization of Expression
use crate::error::{QpyError, to_binrw_error};
use crate::formats::{
    ExpressionElementPack, ExpressionTypePack, ExpressionValueElementPack,
    ExpressionVarElementPack, ExpressionVarRegisterPack,
};
use crate::value::{
    QPYReadData, QPYWriteData, pack_biguint, pack_duration, unpack_biguint, unpack_duration,
};
use binrw::{BinRead, BinResult, BinWrite, Endian, Error};
use num_bigint::BigUint;
use qiskit_circuit::Clbit;
use qiskit_circuit::classical::expr::{
    Binary, BinaryOp, Cast, Expr, Index, Unary, UnaryOp, Value, Var,
};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::duration::Duration;
use std::io::{Read, Seek, Write};

// packed expression types implicitly contain the magic number identifying them in the qpy file
pub(crate) fn pack_expression_type(ty: &Type) -> ExpressionTypePack {
    match ty {
        Type::Bool => ExpressionTypePack::Bool,
        Type::Uint(width) => ExpressionTypePack::Int(*width),
        Type::Duration => ExpressionTypePack::Duration,
        Type::Float => ExpressionTypePack::Float,
    }
}

pub(crate) fn unpack_expression_type(type_pack: ExpressionTypePack) -> Type {
    match type_pack {
        ExpressionTypePack::Bool => Type::Bool,
        ExpressionTypePack::Duration => Type::Duration,
        ExpressionTypePack::Float => Type::Float,
        ExpressionTypePack::Int(width) => Type::Uint(width),
    }
}

pub(crate) fn pack_expression_value(
    value: &Value,
    qpy_data: &QPYWriteData,
) -> Result<ExpressionElementPack, QpyError> {
    let (ty, value_pack) = match value {
        Value::Uint { raw, ty } => {
            match ty {
                Type::Bool => (ty, ExpressionValueElementPack::Bool(raw.to_bytes_le()[0])), // effectively truncating modulo 256
                Type::Uint(_) => (ty, ExpressionValueElementPack::Int(pack_biguint(raw))),
                _ => (ty, ExpressionValueElementPack::Bool(raw.to_bytes_le()[0])), // TODO: should this be different?
            }
        }
        Value::Float { raw, ty } => (ty, ExpressionValueElementPack::Float(*raw)),
        Value::Duration(duration) => {
            if qpy_data.version < 16 && matches!(duration, Duration::ps(_)) {
                return Err(QpyError::UnsupportedFeatureForVersion {
                    feature: "Duration variant 'Duration.ps'".to_string(),
                    version: 16,
                    min_version: qpy_data.version,
                });
            }
            (
                &Type::Duration,
                ExpressionValueElementPack::Duration(pack_duration(duration)),
            )
        }
    };
    Ok(ExpressionElementPack::Value(
        pack_expression_type(ty),
        value_pack,
    ))
}

pub(crate) fn unpack_expression_value(
    value_type_pack: ExpressionTypePack,
    value_element_pack: ExpressionValueElementPack,
) -> Value {
    let ty = unpack_expression_type(value_type_pack);
    match value_element_pack {
        ExpressionValueElementPack::Bool(val) => Value::Uint {
            raw: BigUint::from_bytes_le(&[val]),
            ty,
        },
        ExpressionValueElementPack::Int(val) => Value::Uint {
            raw: unpack_biguint(val),
            ty,
        },
        ExpressionValueElementPack::Duration(duration) => {
            Value::Duration(unpack_duration(duration))
        }
        ExpressionValueElementPack::Float(val) => Value::Float { raw: val, ty },
    }
}

pub(crate) fn pack_expression_var(
    var: &Var,
    qpy_data: &QPYWriteData,
) -> Result<ExpressionElementPack, QpyError> {
    let (ty, value_pack) = match var {
        Var::Bit { bit } => (
            &Type::Bool,
            ExpressionVarElementPack::Clbit(
                qpy_data
                    .circuit_data
                    .clbits()
                    .find(bit)
                    .ok_or_else(|| {
                        QpyError::InvalidBit(format!("Could not find bit {:?} in circuit", bit))
                    })?
                    .0,
            ),
        ),
        Var::Register { register, ty } => (
            ty,
            ExpressionVarElementPack::Register(ExpressionVarRegisterPack {
                name: register.name().to_string(),
            }),
        ),
        Var::Standalone { uuid, name, ty } => (
            ty,
            ExpressionVarElementPack::Uuid(*qpy_data.standalone_var_indices.get(uuid).ok_or_else(
                || {
                    QpyError::InvalidParameter(format!(
                        "Could not find standalone variable {:?} in the qpy data",
                        name
                    ))
                },
            )?),
        ),
    };
    Ok(ExpressionElementPack::Var(
        pack_expression_type(ty),
        value_pack,
    ))
}

pub(crate) fn unpack_expression_var(
    var_type_pack: ExpressionTypePack,
    var_element_pack: ExpressionVarElementPack,
    qpy_data: &QPYReadData,
) -> Result<Var, QpyError> {
    let ty = unpack_expression_type(var_type_pack);
    match var_element_pack {
        ExpressionVarElementPack::Clbit(index) => Ok(Var::Bit {
            bit: qpy_data
                .circuit_data
                .clbits()
                .get(Clbit(index))
                .ok_or_else(|| QpyError::InvalidBit("Clbit not found in circuit data".to_string()))?
                .clone(),
        }),
        ExpressionVarElementPack::Register(packed_register) => Ok(Var::Register {
            register: qpy_data
                .circuit_data
                .cregs_data()
                .get(packed_register.name.as_str())
                .ok_or_else(|| {
                    QpyError::InvalidRegister("Register not found in circuit data".to_string())
                })?
                .clone(),
            ty,
        }), // TODO: can we avoid cloning?
        ExpressionVarElementPack::Uuid(key) => {
            let var = qpy_data.standalone_vars.get(&key).ok_or_else(|| {
                QpyError::InvalidParameter("Standalone var not found in qpy data".to_string())
            })?; // note: this is not an actual expr::Var; merely a key for this var inside the circuit data
            Ok(qpy_data
                .circuit_data
                .vars_stretches_view()
                .vars()
                .get(*var)
                .ok_or_else(|| {
                    QpyError::InvalidParameter(
                        "Standalone var not found in circuit data".to_string(),
                    )
                })?
                .clone()) // TODO: can we avoid cloning?
        }
    }
}

pub(crate) fn write_expression<W: Write + Seek>(
    exp: &Expr,
    writer: &mut W,
    endian: Endian,
    (qpy_data,): (&QPYWriteData,),
) -> binrw::BinResult<()> {
    match exp {
        Expr::Value(val) => {
            pack_expression_value(val, qpy_data)
                .map_err(|e| to_binrw_error(writer, e))?
                .write_options(writer, endian, ())?;
        }
        Expr::Var(var) => {
            pack_expression_var(var, qpy_data)
                .map_err(|e| to_binrw_error(writer, e))?
                .write_options(writer, endian, ())?;
        }
        Expr::Stretch(stretch) => {
            ExpressionElementPack::Stretch(
                ExpressionTypePack::Duration,
                qpy_data.standalone_var_indices[&stretch.uuid],
            )
            .write_options(writer, endian, ())?;
        }
        Expr::Index(index_node) => {
            ExpressionElementPack::Index(pack_expression_type(&index_node.ty)).write_options(
                writer,
                endian,
                (),
            )?;
            write_expression(&index_node.target, writer, endian, (qpy_data,))?;
            write_expression(&index_node.index, writer, endian, (qpy_data,))?;
        }
        Expr::Cast(cast_node) => {
            ExpressionElementPack::Cast(
                pack_expression_type(&cast_node.ty),
                cast_node.implicit as u8,
            )
            .write_options(writer, endian, ())?;
            write_expression(&cast_node.operand, writer, endian, (qpy_data,))?;
        }
        Expr::Unary(unary_node) => {
            ExpressionElementPack::Unary(pack_expression_type(&unary_node.ty), unary_node.op as u8)
                .write_options(writer, endian, ())?;
            write_expression(&unary_node.operand, writer, endian, (qpy_data,))?;
        }
        Expr::Binary(binary_node) => {
            ExpressionElementPack::Binary(
                pack_expression_type(&binary_node.ty),
                binary_node.op as u8,
            )
            .write_options(writer, endian, ())?;
            write_expression(&binary_node.left, writer, endian, (qpy_data,))?;
            write_expression(&binary_node.right, writer, endian, (qpy_data,))?;
        }
    };
    Ok(())
}

pub(crate) fn read_expression<R: Read + Seek>(
    reader: &mut R,
    endian: Endian,
    (qpy_data,): (&QPYReadData,),
) -> BinResult<Expr> {
    let exp_element = ExpressionElementPack::read_options(reader, endian, ())?;
    match exp_element {
        ExpressionElementPack::Value(value_type_pack, value_element_pack) => Ok(Expr::Value(
            unpack_expression_value(value_type_pack, value_element_pack),
        )),
        ExpressionElementPack::Var(var_type_pack, var_element_pack) => Ok(Expr::Var(
            unpack_expression_var(var_type_pack, var_element_pack, qpy_data)
                .map_err(|e| to_binrw_error(reader, e))?,
        )),
        ExpressionElementPack::Stretch(_stretch_type_pack, key) => {
            let stretch = qpy_data.standalone_stretches.get(&key).ok_or_else(|| {
                to_binrw_error(
                    reader,
                    QpyError::InvalidParameter(format!(
                        "Standalone stretch with key {} not found in qpy data",
                        key
                    )),
                )
            })?;
            Ok(Expr::Stretch(
                qpy_data
                    .circuit_data
                    .vars_stretches_view()
                    .stretches()
                    .get(*stretch)
                    .ok_or_else(|| {
                        to_binrw_error(
                            reader,
                            QpyError::InvalidParameter(
                                "Stretch not found in circuit data".to_string(),
                            ),
                        )
                    })?
                    .clone(),
            )) // TODO: can we avoid cloning?
        }
        ExpressionElementPack::Index(index_type_pack) => {
            let target = read_expression(reader, endian, (qpy_data,))?;
            let index = read_expression(reader, endian, (qpy_data,))?;
            let constant = target.is_const() && index.is_const();
            Ok(Expr::Index(Box::new(Index {
                target,
                index,
                ty: unpack_expression_type(index_type_pack),
                constant,
            })))
        }
        ExpressionElementPack::Cast(cast_type_pack, implicit) => {
            let operand = read_expression(reader, endian, (qpy_data,))?;
            let constant = operand.is_const();
            Ok(Expr::Cast(Box::new(Cast {
                operand,
                ty: unpack_expression_type(cast_type_pack),
                constant,
                implicit: implicit != 0,
            })))
        }
        ExpressionElementPack::Unary(unary_type_pack, op) => {
            let operand = read_expression(reader, endian, (qpy_data,))?;
            let constant = operand.is_const();
            Ok(Expr::Unary(Box::new(Unary {
                op: UnaryOp::from_u8(op).map_err(|_| Error::NoVariantMatch { pos: (0) })?,
                operand,
                ty: unpack_expression_type(unary_type_pack),
                constant,
            })))
        }
        ExpressionElementPack::Binary(binary_type_pack, op) => {
            let left = read_expression(reader, endian, (qpy_data,))?;
            let right = read_expression(reader, endian, (qpy_data,))?;
            let constant = left.is_const() && right.is_const();
            Ok(Expr::Binary(Box::new(Binary {
                op: BinaryOp::from_u8(op).map_err(|_| Error::NoVariantMatch { pos: (0) })?,
                left,
                right,
                ty: unpack_expression_type(binary_type_pack),
                constant,
            })))
        }
    }
}
