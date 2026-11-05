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

#[cfg(feature = "py")]
use pyo3::prelude::*;

use crate::lex::Token;

pub struct Position<'a> {
    filename: &'a std::ffi::OsStr,
    line: usize,
    col: usize,
}

impl<'a> Position<'a> {
    pub fn new(filename: &'a std::ffi::OsStr, line: usize, col: usize) -> Self {
        Self {
            filename,
            line,
            col,
        }
    }
}

impl std::fmt::Display for &Position<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}:{},{}",
            self.filename.to_string_lossy(),
            self.line,
            self.col
        )
    }
}

/// Create an error message that includes span data from the given [token][Token].  The base of the
/// message is `message`, and `filename` is the file the triggering OpenQASM 2 code came from.  For
/// string inputs, this can be a placeholder.
pub fn message_generic(position: Option<&Position>, message: &str) -> String {
    if let Some(position) = position {
        format!("{position}: {message}")
    } else {
        message.to_owned()
    }
}

/// Shorthand form for creating an error message when a particular type of token was required, but
/// something else was `received`.
pub fn message_incorrect_requirement(
    required: &str,
    received: &Token,
    filename: &std::ffi::OsStr,
) -> String {
    message_generic(
        Some(&Position::new(filename, received.line, received.col)),
        &format!(
            "needed {}, but instead saw {}",
            required,
            received.ttype.describe()
        ),
    )
}

/// Shorthand form for creating an error message when a particular type of token was required, but
/// the input ended unexpectedly.
pub fn message_bad_eof(position: Option<&Position>, required: &str) -> String {
    message_generic(
        position,
        &format!("unexpected end-of-file when expecting to see {required}"),
    )
}

/// Pure-Rust error type used as the error channel.
#[derive(Debug)]
pub struct ParseError {
    pub message: String,
    #[cfg(feature = "py")]
    pub source: Option<Box<PyErr>>,
}

impl ParseError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            #[cfg(feature = "py")]
            source: None,
        }
    }

    /// As [`ParseError::new`], but chaining `source` as the original cause of the failure.
    #[cfg(feature = "py")]
    pub fn with_source(message: impl Into<String>, source: PyErr) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
    /// Replace the message (e.g. to prepend position information) while keeping any chained
    /// `source` intact.
    pub fn with_message(self, message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            #[cfg(feature = "py")]
            source: self.source,
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}
impl std::error::Error for ParseError {}

#[cfg(feature = "py")]
pyo3::import_exception!(qiskit.qasm2.exceptions, QASM2ParseError);

/// Convert a `ParseError` from the pyo3-free parsing modules into the `QASM2ParseError`
/// Python exception, at the boundary where results cross back into Python space.
#[cfg(feature = "py")]
impl From<ParseError> for PyErr {
    fn from(e: ParseError) -> PyErr {
        let py_err = QASM2ParseError::new_err(e.message);
        if let Some(source) = e.source {
            Python::attach(|py| py_err.set_cause(py, Some(*source)));
        }
        py_err
    }
}
