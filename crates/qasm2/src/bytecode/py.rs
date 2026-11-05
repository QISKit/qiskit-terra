// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
pyo3::import_exception!(qiskit.qasm2.exceptions, QASM2ParseError);

use crate::expr::Expr;
use crate::ext::ClassicalEvaluator;

use crate::{CustomClassical, CustomInstruction, lex, parse};

use super::InternalBytecode;

/// The Rust parser produces an iterator of these `Bytecode` instructions, which comprise an opcode
/// integer for operation distinction, and a free-form tuple containing the operands.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen)]
#[derive(Clone)]
pub struct Bytecode {
    #[pyo3(get)]
    pub opcode: OpCode,
    #[pyo3(get)]
    pub operands: Py<PyAny>,
}

/// The operations that are represented by the "bytecode" passed to Python.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq)]
#[derive(Clone, Eq, PartialEq)]
pub enum OpCode {
    // There is only a `Gate` here, not a `GateInBasis`, because in Python space we don't have the
    // same strict typing requirements to satisfy.
    Gate,
    ConditionedGate,
    Measure,
    ConditionedMeasure,
    Reset,
    ConditionedReset,
    Barrier,
    DeclareQreg,
    DeclareCreg,
    DeclareGate,
    EndDeclareGate,
    DeclareOpaque,
    SpecialInclude,
}

/// The custom iterator object that is returned up to Python space for iteration through the
/// bytecode stream.  This is never constructed on the Python side; it is built in Rust space
/// by Python calls to [bytecode_from_string] and [bytecode_from_file].
#[pyclass]
pub struct BytecodeIterator {
    parser_state: parse::State,
    buffer: Vec<Option<InternalBytecode>>,
    buffer_used: usize,
}

impl BytecodeIterator {
    pub fn new(
        tokens: lex::TokenStream,
        include_path: Vec<std::path::PathBuf>,
        custom_instructions: &[CustomInstruction],
        custom_classical: &[CustomClassical],
        strict: bool,
    ) -> PyResult<Self> {
        Ok(BytecodeIterator {
            parser_state: parse::State::new(
                tokens,
                include_path,
                custom_instructions,
                custom_classical,
                strict,
            )?,
            buffer: vec![],
            buffer_used: 0,
        })
    }
}

#[pymethods]
impl BytecodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Bytecode>> {
        if self.buffer_used >= self.buffer.len() {
            self.buffer.clear();
            self.buffer_used = 0;
            self.parser_state
                .parse_next(&mut self.buffer, ClassicalEvaluator::attached(py))?;
        }
        if self.buffer.is_empty() {
            Ok(None)
        } else {
            self.buffer_used += 1;
            Ok(self.buffer[self.buffer_used - 1]
                .take()
                .map(|bytecode| bytecode.into_pyobject(py))
                .transpose()?
                .map(|x| x.get().clone()))
        }
    }
}

#[pyclass(module = "qiskit._accelerate.qasm2", frozen)]
pub struct GateBodyArguments(Vec<Expr>);

impl GateBodyArguments {
    pub(super) fn new(arguments: Vec<Expr>) -> Self {
        Self(arguments)
    }
}

#[pymethods]
impl GateBodyArguments {
    fn evaluate(&self, params: Vec<f64>, py: Python<'_>) -> PyResult<Vec<f64>> {
        self.0
            .iter()
            .map(|expr| {
                crate::expr::evaluate(expr, &params, ClassicalEvaluator::attached(py))
                    .map_err(PyErr::from)
            })
            .collect()
    }
}
