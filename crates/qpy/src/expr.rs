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
use crate::error::QpyError;
use crate::formats::{
    ExpressionTypePack, ExpressionValueElementPack, ExpressionVarElementPack,
    ExpressionVarRegisterPack, PackedExpression,
};
use crate::value::{
    QPYReadData, QPYWriteData, clbit_at, clbit_index, creg_by_name, pack_biguint, pack_duration,
    unpack_biguint, unpack_duration,
};
use num_bigint::BigUint;
use qiskit_circuit::classical::expr::{
    Binary, BinaryOp, Cast, Expr, Index, Unary, UnaryOp, Value, Var,
};
use qiskit_circuit::classical::types::Type;
use qiskit_circuit::duration::Duration;

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
) -> Result<PackedExpression, QpyError> {
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
    Ok(PackedExpression::Value(
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
) -> Result<PackedExpression, QpyError> {
    let (ty, value_pack) = match var {
        Var::Bit { bit } => (
            &Type::Bool,
            ExpressionVarElementPack::Clbit(clbit_index(bit, qpy_data)?),
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
    Ok(PackedExpression::Var(pack_expression_type(ty), value_pack))
}

pub(crate) fn unpack_expression_var(
    var_type_pack: ExpressionTypePack,
    var_element_pack: ExpressionVarElementPack,
    qpy_data: &QPYReadData,
) -> Result<Var, QpyError> {
    let ty = unpack_expression_type(var_type_pack);
    match var_element_pack {
        ExpressionVarElementPack::Clbit(index) => Ok(Var::Bit {
            bit: clbit_at(index, qpy_data)?,
        }),
        ExpressionVarElementPack::Register(packed_register) => Ok(Var::Register {
            register: creg_by_name(&packed_register.name, qpy_data)?,
            ty,
        }),
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

pub(crate) fn pack_expression(
    exp: &Expr,
    qpy_data: &QPYWriteData,
) -> Result<PackedExpression, QpyError> {
    Ok(match exp {
        Expr::Value(value) => pack_expression_value(value, qpy_data)?,
        Expr::Var(var) => pack_expression_var(var, qpy_data)?,
        Expr::Stretch(stretch) => PackedExpression::Stretch(
            ExpressionTypePack::Duration,
            *qpy_data
                .standalone_var_indices
                .get(&stretch.uuid)
                .ok_or_else(|| {
                    QpyError::InvalidParameter(format!(
                        "Could not find standalone stretch {:?} in the qpy data",
                        stretch.name
                    ))
                })?,
        ),
        Expr::Index(node) => PackedExpression::Index(
            pack_expression_type(&node.ty),
            Box::new(pack_expression(&node.target, qpy_data)?),
            Box::new(pack_expression(&node.index, qpy_data)?),
        ),
        Expr::Cast(node) => PackedExpression::Cast(
            pack_expression_type(&node.ty),
            node.implicit as u8,
            Box::new(pack_expression(&node.operand, qpy_data)?),
        ),
        Expr::Unary(node) => PackedExpression::Unary(
            pack_expression_type(&node.ty),
            node.op as u8,
            Box::new(pack_expression(&node.operand, qpy_data)?),
        ),
        Expr::Binary(node) => PackedExpression::Binary(
            pack_expression_type(&node.ty),
            node.op as u8,
            Box::new(pack_expression(&node.left, qpy_data)?),
            Box::new(pack_expression(&node.right, qpy_data)?),
        ),
    })
}

pub(crate) fn unpack_expression(
    packed: PackedExpression,
    qpy_data: &QPYReadData,
) -> Result<Expr, QpyError> {
    match packed {
        PackedExpression::Value(value_type_pack, value_element_pack) => Ok(Expr::Value(
            unpack_expression_value(value_type_pack, value_element_pack),
        )),
        PackedExpression::Var(var_type_pack, var_element_pack) => Ok(Expr::Var(
            unpack_expression_var(var_type_pack, var_element_pack, qpy_data)?,
        )),
        PackedExpression::Stretch(_stretch_type_pack, key) => {
            let stretch = qpy_data.standalone_stretches.get(&key).ok_or_else(|| {
                QpyError::InvalidParameter(format!(
                    "Standalone stretch with key {} not found in qpy data",
                    key
                ))
            })?;
            Ok(Expr::Stretch(
                qpy_data
                    .circuit_data
                    .vars_stretches_view()
                    .stretches()
                    .get(*stretch)
                    .ok_or_else(|| {
                        QpyError::InvalidParameter("Stretch not found in circuit data".to_string())
                    })?
                    .clone(),
            )) // TODO: can we avoid cloning?
        }
        PackedExpression::Index(index_type_pack, target, index) => {
            let target = unpack_expression(*target, qpy_data)?;
            let index = unpack_expression(*index, qpy_data)?;
            let constant = target.is_const() && index.is_const();
            Ok(Expr::Index(Box::new(Index {
                target,
                index,
                ty: unpack_expression_type(index_type_pack),
                constant,
            })))
        }
        PackedExpression::Cast(cast_type_pack, implicit, operand) => {
            let operand = unpack_expression(*operand, qpy_data)?;
            let constant = operand.is_const();
            Ok(Expr::Cast(Box::new(Cast {
                operand,
                ty: unpack_expression_type(cast_type_pack),
                constant,
                implicit: implicit != 0,
            })))
        }
        PackedExpression::Unary(unary_type_pack, op, operand) => {
            let operand = unpack_expression(*operand, qpy_data)?;
            let constant = operand.is_const();
            Ok(Expr::Unary(Box::new(Unary {
                op: UnaryOp::try_from_u8(op).map_err(|_| QpyError::InvalidValueType {
                    expected: "classical unary operator".to_string(),
                    actual: op.to_string(),
                })?,
                operand,
                ty: unpack_expression_type(unary_type_pack),
                constant,
            })))
        }
        PackedExpression::Binary(binary_type_pack, op, left, right) => {
            let left = unpack_expression(*left, qpy_data)?;
            let right = unpack_expression(*right, qpy_data)?;
            let constant = left.is_const() && right.is_const();
            Ok(Expr::Binary(Box::new(Binary {
                op: BinaryOp::try_from_u8(op).map_err(|_| QpyError::InvalidValueType {
                    expected: "classical binary operator".to_string(),
                    actual: op.to_string(),
                })?,
                left,
                right,
                ty: unpack_expression_type(binary_type_pack),
                constant,
            })))
        }
    }
}
