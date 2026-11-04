// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::IntoPyObjectExt;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use hashbrown::HashMap;

use oq3_semantics::asg;
use oq3_semantics::symbols::{SymbolId, SymbolTable};
use oq3_semantics::types::Type;

use crate::build::PySymbolTable;
use crate::circuit::PyRegister;
use crate::error::QASM3ImporterError;

pub fn eval_gate_param(
    _py: Python,
    _our_symbols: &PySymbolTable,
    _ast_symbols: &SymbolTable,
    param: &asg::TExpr,
) -> PyResult<f64> {
    // Only handling float parameters in this first pass of the importer.
    match param.get_type() {
        Type::Float(_, is_const) => {
            if is_const.clone().into() {
                match param.expression() {
                    asg::Expr::Literal(asg::Literal::Float(lit)) => {
                        lit.value().parse().map_err(|_| {
                            QASM3ImporterError::new_err(format!(
                                "invalid float literal: '{}'",
                                lit.value()
                            ))
                        })
                    }
                    expr => Err(QASM3ImporterError::new_err(format!(
                        "unhandled expression for floating-point constant: {expr:?}"
                    ))),
                }
            } else {
                Err(QASM3ImporterError::new_err(format!(
                    "expected a constant float, but found a runtime value: {param:?}"
                )))
            }
        }
        Type::Angle(_, _) => Err(QASM3ImporterError::new_err(
            "the OpenQASM 3 'angle' type is not yet supported",
        )),
        ty => Err(QASM3ImporterError::new_err(format!(
            "expected an angle-like type, but saw {ty:?}"
        ))),
    }
}

fn eval_const_int(_py: Python, _ast_symbols: &SymbolTable, expr: &asg::TExpr) -> PyResult<isize> {
    match expr.get_type() {
        Type::Int(_, is_const) | Type::UInt(_, is_const) => {
            if is_const.clone().into() {
                match expr.expression() {
                    asg::Expr::Literal(asg::Literal::Int(lit)) => Ok(*lit.value() as isize),
                    expr => Err(QASM3ImporterError::new_err(format!(
                        "unhandled expression type for constant-integer evaluation: {expr:?}"
                    ))),
                }
            } else {
                Err(QASM3ImporterError::new_err(format!(
                    "expected a constant integer, but found a runtime value: {expr:?}"
                )))
            }
        }
        ty => Err(QASM3ImporterError::new_err(format!(
            "expected a constant integer, but found a value of type: {ty:?}"
        ))),
    }
}

fn eval_const_uint(py: Python, ast_symbols: &SymbolTable, expr: &asg::TExpr) -> PyResult<usize> {
    eval_const_int(py, ast_symbols, expr).and_then(|val| {
        val.try_into().map_err(|_| {
            QASM3ImporterError::new_err(format!("expected an unsigned integer but found '{val}'"))
        })
    })
}

pub enum BroadcastItem {
    Bit(Py<PyAny>),
    Register(Vec<Py<PyAny>>),
}

struct BroadcastQubitsIter<'py> {
    py: Python<'py>,
    len: usize,
    offset: usize,
    items: Vec<BroadcastItem>,
}

impl<'py> Iterator for BroadcastQubitsIter<'py> {
    type Item = Bound<'py, PyTuple>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.len {
            return None;
        }
        let offset = self.offset;
        let to_scalar = |item: &BroadcastItem| match item {
            BroadcastItem::Bit(bit) => bit.clone_ref(self.py),
            BroadcastItem::Register(bits) => bits[offset].clone_ref(self.py),
        };
        self.offset += 1;
        Some(PyTuple::new(self.py, self.items.iter().map(to_scalar)).unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.offset, Some(self.len - self.offset))
    }
}
impl ExactSizeIterator for BroadcastQubitsIter<'_> {}

struct BroadcastMeasureIter<'a, 'py> {
    py: Python<'py>,
    len: usize,
    offset: usize,
    qarg: &'a BroadcastItem,
    carg: &'a BroadcastItem,
}

impl<'py> Iterator for BroadcastMeasureIter<'_, 'py> {
    type Item = (Bound<'py, PyTuple>, Bound<'py, PyTuple>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.len {
            return None;
        }
        let offset = self.offset;
        let to_scalar = |item: &BroadcastItem| match item {
            BroadcastItem::Bit(bit) => bit.clone_ref(self.py),
            BroadcastItem::Register(bits) => bits[offset].clone_ref(self.py),
        };
        self.offset += 1;
        Some((
            PyTuple::new(self.py, &[to_scalar(self.qarg)]).unwrap(),
            PyTuple::new(self.py, &[to_scalar(self.carg)]).unwrap(),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len - self.offset, Some(self.len - self.offset))
    }
}
impl ExactSizeIterator for BroadcastMeasureIter<'_, '_> {}

fn broadcast_bits_for_identifier<T: PyRegister>(
    py: Python,
    bits: &HashMap<SymbolId, Py<PyAny>>,
    registers: &HashMap<SymbolId, T>,
    iden_symbol: &SymbolId,
) -> PyResult<BroadcastItem> {
    if let Some(bit) = bits.get(iden_symbol) {
        Ok(BroadcastItem::Bit(bit.clone()))
    } else if let Some(reg) = registers.get(iden_symbol) {
        Ok(BroadcastItem::Register(
            reg.bit_list(py)
                .iter()
                .map(|obj| obj.into_py_any(py).unwrap())
                .collect(),
        ))
    } else {
        Err(QASM3ImporterError::new_err(format!(
            "unknown symbol: {iden_symbol:?}"
        )))
    }
}

fn broadcast_apply_index(
    py: Python,
    ast_symbols: &SymbolTable,
    broadcasted: BroadcastItem,
    index: &asg::IndexOperator,
) -> PyResult<BroadcastItem> {
    let bits = match broadcasted {
        BroadcastItem::Register(bits) => Ok(bits),
        BroadcastItem::Bit(_) => Err(QASM3ImporterError::new_err(
            "cannot index into a scalar value",
        )),
    }?;
    let eval_single_index = |expr: &asg::TExpr| -> PyResult<Py<PyAny>> {
        let index = eval_const_uint(py, ast_symbols, expr)?;
        match bits.get(index) {
            Some(bit) => Ok(bit.clone_ref(py)),
            None => Err(QASM3ImporterError::new_err(format!(
                "index {} out of range for register of length {}",
                index,
                bits.len()
            ))),
        }
    };
    match index {
        asg::IndexOperator::SetExpression(exprs) => exprs
            .expressions()
            .iter()
            .map(eval_single_index)
            .collect::<PyResult<Vec<_>>>()
            .map(BroadcastItem::Register),
        asg::IndexOperator::ExpressionList(exprs) => {
            let expr = match &exprs.expressions[..] {
                [expr] => Ok(expr),
                _ => Err(QASM3ImporterError::new_err(
                    "registers can only be one-dimensional",
                )),
            }?;
            match expr.get_type() {
                Type::UInt(_, _) | Type::Int(_, _) => {
                    Ok(BroadcastItem::Bit(eval_single_index(expr)?))
                }
                ty => Err(QASM3ImporterError::new_err(format!(
                    "unhandled index type: {ty:?}"
                ))),
            }
        }
    }
}

pub fn eval_qarg(
    py: Python,
    our_symbols: &PySymbolTable,
    ast_symbols: &SymbolTable,
    qarg: &asg::GateOperand,
) -> PyResult<BroadcastItem> {
    match qarg {
        asg::GateOperand::Identifier(symbol) => broadcast_bits_for_identifier(
            py,
            &our_symbols.qubits,
            &our_symbols.qregs,
            symbol.as_ref().unwrap(),
        ),
        asg::GateOperand::IndexedIdentifier(indexed) => {
            let iden_symbol = indexed.identifier().as_ref().unwrap();
            indexed.indexes().iter().fold(
                broadcast_bits_for_identifier(
                    py,
                    &our_symbols.qubits,
                    &our_symbols.qregs,
                    iden_symbol,
                ),
                |item, index| {
                    item.and_then(|item| broadcast_apply_index(py, ast_symbols, item, index))
                },
            )
        }
        asg::GateOperand::HardwareQubit(_) => {
            Err(QASM3ImporterError::new_err("cannot handle hardware qubits"))
        }
    }
}

pub fn eval_measure_carg(
    py: Python,
    our_symbols: &PySymbolTable,
    ast_symbols: &SymbolTable,
    carg: &asg::LValue,
) -> PyResult<BroadcastItem> {
    match carg {
        asg::LValue::Identifier(iden) => {
            let symbol_id = iden
                .as_ref()
                .map_err(|err| QASM3ImporterError::new_err(format!("internal error: {err:?}")))?;
            broadcast_bits_for_identifier(py, &our_symbols.clbits, &our_symbols.cregs, symbol_id)
        }
        asg::LValue::IndexedIdentifier(indexed) => {
            let iden_symbol = indexed.identifier().as_ref().unwrap();
            indexed.indexes().iter().fold(
                broadcast_bits_for_identifier(
                    py,
                    &our_symbols.clbits,
                    &our_symbols.cregs,
                    iden_symbol,
                ),
                |item, index| {
                    item.and_then(|item| broadcast_apply_index(py, ast_symbols, item, index))
                },
            )
        }
    }
}

pub fn expect_gate_operand(expr: &asg::TExpr) -> PyResult<&asg::GateOperand> {
    match expr.get_type() {
        Type::Qubit | Type::QubitArray(_) | Type::HardwareQubit => (),
        ty => {
            return Err(QASM3ImporterError::new_err(format!(
                "unhandled gate operand expression type: {ty:?}"
            )));
        }
    }
    match expr.expression() {
        asg::Expr::GateOperand(operand) => Ok(operand),
        expr => Err(QASM3ImporterError::new_err(format!(
            "internal error: not a gate operand {expr:?}"
        ))),
    }
}

pub fn broadcast_qubits<'a, 'py, T>(
    py: Python<'py>,
    our_symbols: &PySymbolTable,
    ast_symbols: &SymbolTable,
    qargs: T,
) -> PyResult<impl Iterator<Item = Bound<'py, PyTuple>> + use<'py, T>>
where
    T: IntoIterator<Item = &'a asg::TExpr> + 'a,
{
    let items = qargs
        .into_iter()
        .map(|item| -> PyResult<BroadcastItem> {
            eval_qarg(py, our_symbols, ast_symbols, expect_gate_operand(item)?)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let mut broadcast_len = None;
    for item in items.iter() {
        match (item, broadcast_len) {
            (BroadcastItem::Bit(_), _) => (),
            (BroadcastItem::Register(reg), Some(len)) => {
                if reg.len() != len {
                    return Err(QASM3ImporterError::new_err("invalid broadcast"));
                }
            }
            (BroadcastItem::Register(reg), None) => {
                broadcast_len = Some(reg.len());
            }
        }
    }
    Ok(BroadcastQubitsIter {
        py,
        len: broadcast_len.unwrap_or(if items.is_empty() { 0 } else { 1 }),
        offset: 0,
        items,
    })
}

pub fn broadcast_measure<'a, 'py>(
    py: Python<'py>,
    qarg: &'a BroadcastItem,
    carg: &'a BroadcastItem,
) -> PyResult<impl Iterator<Item = (Bound<'py, PyTuple>, Bound<'py, PyTuple>)> + 'a>
where
    'py: 'a,
{
    let len = match (qarg, carg) {
        (BroadcastItem::Bit(_), BroadcastItem::Bit(_)) => Ok(1),
        (BroadcastItem::Bit(_), BroadcastItem::Register(_))
        | (BroadcastItem::Register(_), BroadcastItem::Bit(_)) => Err(QASM3ImporterError::new_err(
            "invalid measurement broadcast: cannot broadcast a bit against a register",
        )),
        (BroadcastItem::Register(qreg), BroadcastItem::Register(creg)) => {
            if qreg.len() == creg.len() {
                Ok(qreg.len())
            } else {
                Err(QASM3ImporterError::new_err(format!(
                    "invalid measurement broadcast: qarg has length {}, carg has length {}",
                    qreg.len(),
                    creg.len()
                )))
            }
        }
    }?;
    Ok(BroadcastMeasureIter {
        py,
        len,
        offset: 0,
        qarg,
        carg,
    })
}
