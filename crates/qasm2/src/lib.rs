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

use crate::error::QASM2ParseError;

mod bytecode;
mod error;
mod expr;
mod ext;
mod lex;
mod parse;

pub use self::ext::{
    ClassicalBuiltinExt, ClassicalCallableExt, CustomClassical, CustomInstruction,
};

/// Create a bytecode iterable from a string containing an OpenQASM 2 program.  The iterable will
/// lex and parse the source lazily; evaluating OpenQASM 2 statements as required, without loading
/// the entire token and parse tree into memory at once.
#[pyfunction]
fn bytecode_from_string(
    string: String,
    include_path: Vec<std::path::PathBuf>,
    custom_instructions: Vec<CustomInstruction>,
    custom_classical: Vec<CustomClassical>,
    strict: bool,
) -> PyResult<bytecode::BytecodeIterator> {
    bytecode::BytecodeIterator::new(
        lex::TokenStream::from_string(string, strict),
        include_path,
        &custom_instructions,
        &custom_classical,
        strict,
    )
}

/// Create a bytecode iterable from a path to a file containing an OpenQASM 2 program.  The
/// iterable will lex and parse the source lazily; evaluating OpenQASM 2 statements as required,
/// without loading the entire token and parse tree into memory at once.
#[pyfunction]
fn bytecode_from_file(
    py: Python<'_>,
    path: std::ffi::OsString,
    include_path: Vec<std::path::PathBuf>,
    custom_instructions: Vec<CustomInstruction>,
    custom_classical: Vec<CustomClassical>,
    strict: bool,
) -> PyResult<bytecode::BytecodeIterator> {
    bytecode::BytecodeIterator::new(
        lex::TokenStream::from_path(&path, strict).map_err(|err| {
            let exc = QASM2ParseError::new_err(format!(
                "failed to read a token stream from file '{}'",
                path.to_string_lossy()
            ));
            exc.set_cause(py, Some(err.into()));
            exc
        })?,
        include_path,
        &custom_instructions,
        &custom_classical,
        strict,
    )
}

/// An interface to the Rust components of the parser stack, and the types it uses to represent the
/// output.  The principal entry points for Python are :func:`bytecode_from_string` and
/// :func:`bytecode_from_file`, which produce iterables of :class:`Bytecode` objects.
pub fn qasm2(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_class::<bytecode::OpCode>()?;
    module.add_class::<bytecode::UnaryOpCode>()?;
    module.add_class::<bytecode::BinaryOpCode>()?;
    module.add_class::<bytecode::Bytecode>()?;
    module.add_class::<bytecode::ExprConstant>()?;
    module.add_class::<bytecode::ExprArgument>()?;
    module.add_class::<bytecode::ExprUnary>()?;
    module.add_class::<bytecode::ExprBinary>()?;
    module.add_class::<bytecode::ExprCustom>()?;
    module.add_class::<CustomInstruction>()?;
    module.add_class::<CustomClassical>()?;
    module.add_function(wrap_pyfunction!(bytecode_from_string, module)?)?;
    module.add_function(wrap_pyfunction!(bytecode_from_file, module)?)?;
    Ok(())
}
