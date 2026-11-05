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

mod ast;
mod build;
mod circuit;
mod error;
mod exporter;
mod expr;
mod printer;

use std::ffi::OsString;
use std::ops::Deref;
use std::path::{Path, PathBuf};

use hashbrown::HashMap;

use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use oq3_semantics::syntax_to_semantics::parse_source_string;
use pyo3::pybacked::PyBackedStr;
use qiskit_circuit::circuit_data::PyCircuitData;

use crate::error::QASM3ImporterError;

/// Load an OpenQASM 3 program from a string into a :class:`.QuantumCircuit`.
///
/// .. warning::
///
///     This native version of the OpenQASM 3 importer is currently experimental.  It is typically
///     much faster than :func:`~qiskit.qasm3.loads`, but has a reduced supported feature set,
///     which will expand over time.
///
/// .. warning::
///
///     This experimental importer has no security guarantees.  See :ref:`qasm3-security-import` for
///     more detail.
///
/// Args:
///     source (str): the program source in a Python string.
///     custom_gates (Iterable[CustomGate]): Python constructors to use for particular named gates.
///         If not supplied, Qiskit will use its own standard-library constructors for gates
///         defined in the OpenQASM 3.0 standard-library file ``stdgates.inc``.
///     include_path (Iterable[str]): the path to search when resolving ``include`` statements.
///         If not given, Qiskit will arrange for this to point to a location containing
///         ``stdgates.inc`` only.  Paths are tried in the sequence order.
///
/// Returns:
///     :class:`.QuantumCircuit`: the constructed circuit object.
///
/// Raises:
///     :exc:`.QASM3ImporterError`: if an error occurred during parsing or semantic analysis.
///         In the case of a parsing error, most of the error messages are printed to the terminal
///         and formatted, for better legibility.
#[pyfunction]
#[pyo3(signature = (source, /, *, custom_gates=None, include_path=None))]
pub fn loads(
    py: Python,
    source: String,
    custom_gates: Option<Vec<circuit::PyGate>>,
    include_path: Option<Vec<OsString>>,
) -> PyResult<circuit::PyCircuit> {
    let default_include_path = || -> PyResult<Vec<OsString>> {
        let filename: PyBackedStr = py.import("qiskit")?.filename()?.try_into()?;
        Ok(vec![
            Path::new(filename.deref())
                .parent()
                .unwrap()
                .join(["qasm", "libs", "dummy"].iter().collect::<PathBuf>())
                .into_os_string(),
        ])
    };
    let include_path = include_path.map(Ok).unwrap_or_else(default_include_path)?;
    let result = parse_source_string(source, None, Some(&include_path));
    if result.any_errors() {
        result.print_errors();
        return Err(QASM3ImporterError::new_err(
            "errors during parsing; see printed errors",
        ));
    }
    let gates = match custom_gates {
        Some(gates) => gates
            .into_iter()
            .map(|gate| (gate.name().to_owned(), gate))
            .collect(),
        None => py
            .import("qiskit.qasm3")?
            .getattr("STDGATES_INC_GATES")?
            .try_iter()?
            .map(|obj| {
                let gate = obj?.extract::<circuit::PyGate>()?;
                Ok((gate.name().to_owned(), gate))
            })
            .collect::<PyResult<HashMap<_, _>>>()?,
    };
    crate::build::convert_asg(py, result.program(), result.symbol_table(), gates)
}

/// Load an OpenQASM 3 program from a source file into a :class:`.QuantumCircuit`.
///
/// .. warning::
///
///     This native version of the OpenQASM 3 importer is currently experimental.  It is typically
///     much faster than :func:`~qiskit.qasm3.load`, but has a reduced supported feature set, which
///     will expand over time.
///
/// .. warning::
///
///     This experimental importer has no security guarantees.  See :ref:`qasm3-security-import` for
///     more detail.
///
/// Args:
///     pathlike_or_filelike (str | os.PathLike | io.TextIOBase): the program source.  This can
///         either be given as a filepath, or an open text stream object.  If the stream is already
///         opened it is consumed in Python space, whereas filenames are opened and consumed in
///         Rust space; there might be slightly different performance characteristics, depending on
///         your system and how the streams are buffered by default.
///     custom_gates (Iterable[CustomGate]): Python constructors to use for particular named gates.
///         If not supplied, Qiskit will use its own standard-library constructors for gates
///         defined in the OpenQASM 3.0 standard-library file ``stdgates.inc``.
///     include_path (Iterable[str]): the path to search when resolving ``include`` statements.
///         If not given, Qiskit will arrange for this to point to a location containing
///         ``stdgates.inc`` only.  Paths are tried in the sequence order.
///
/// Returns:
///     :class:`.QuantumCircuit`: the constructed circuit object.
///
/// Raises:
///     :exc:`.QASM3ImporterError`: if an error occurred during parsing or semantic analysis.
///         In the case of a parsing error, most of the error messages are printed to the terminal
///         and formatted, for better legibility.
#[pyfunction]
#[pyo3(
    signature = (pathlike_or_filelike, /, *, custom_gates=None, include_path=None),
)]
pub fn load(
    py: Python,
    pathlike_or_filelike: &Bound<PyAny>,
    custom_gates: Option<Vec<circuit::PyGate>>,
    include_path: Option<Vec<OsString>>,
) -> PyResult<circuit::PyCircuit> {
    let source =
        if pathlike_or_filelike.is_instance(&PyModule::import(py, "io")?.getattr("TextIOBase")?)? {
            pathlike_or_filelike
                .call_method0("read")?
                .extract::<String>()?
        } else {
            let path = PyModule::import(py, "os")?
                .getattr("fspath")?
                .call1((pathlike_or_filelike,))?
                .extract::<OsString>()?;
            ::std::fs::read_to_string(&path).map_err(|err| {
                QASM3ImporterError::new_err(format!("failed to read file '{:?}': {:?}", path, err))
            })?
        };
    loads(py, source, custom_gates, include_path)
}

#[derive(Debug, Clone)]
struct DumpOptions {
    includes: Vec<String>,
    basis_gates: Vec<String>,
    disable_constants: bool,
    allow_aliasing: bool,
    indent: String,
}

impl Default for DumpOptions {
    fn default() -> Self {
        Self {
            includes: vec!["stdgates.inc".to_string()],
            basis_gates: vec![],
            disable_constants: true,
            allow_aliasing: false,
            indent: "  ".to_string(),
        }
    }
}

#[pyfunction]
#[pyo3(signature = (circuit, /, kwargs=None))]
pub fn dumps(
    _py: Python,
    circuit: &Bound<PyAny>,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<String> {
    let mut options = DumpOptions::default();

    if let Some(kw) = kwargs {
        if let Some(val) = kw.get_item("includes")? {
            options.includes = val.extract::<Vec<String>>()?;
        }
        if let Some(val) = kw.get_item("basis_gates")? {
            options.basis_gates = val.extract::<Vec<String>>()?;
        }
        if let Some(val) = kw.get_item("disable_constants")? {
            options.disable_constants = val.extract::<bool>()?;
        }
        if let Some(val) = kw.get_item("allow_aliasing")? {
            options.allow_aliasing = val.extract::<bool>()?;
        }
        if let Some(val) = kw.get_item("indent")? {
            options.indent = val.extract::<String>()?;
        }
    }
    let circuit_data = circuit.getattr("_data")?.cast::<PyCircuitData>()?.borrow();

    let islayout = !circuit.getattr("layout")?.is_none();

    let exporter = exporter::Exporter::new(
        options.includes,
        options.basis_gates,
        options.disable_constants,
        options.allow_aliasing,
        options.indent,
    );

    let stream = exporter.dumps(&circuit_data, islayout).map_err(|err| {
        QASM3ImporterError::new_err(format!(
            "failed to export circuit using qasm3.dumps_experimental: {err:?}"
        ))
    })?;

    Ok(stream)
}

#[pyfunction]
#[pyo3(signature = (circuit,stream, /, kwargs=None))]
pub fn dump(
    _py: Python,
    circuit: &Bound<PyAny>,
    stream: &Bound<PyAny>,
    kwargs: Option<&Bound<PyDict>>,
) -> PyResult<()> {
    let mut options = DumpOptions::default();

    if let Some(kw) = kwargs {
        if let Some(val) = kw.get_item("includes")? {
            options.includes = val.extract::<Vec<String>>()?;
        }
        if let Some(val) = kw.get_item("basis_gates")? {
            options.basis_gates = val.extract::<Vec<String>>()?;
        }
        if let Some(val) = kw.get_item("disable_constants")? {
            options.disable_constants = val.extract::<bool>()?;
        }
        if let Some(val) = kw.get_item("allow_aliasing")? {
            options.allow_aliasing = val.extract::<bool>()?;
        }
        if let Some(val) = kw.get_item("indent")? {
            options.indent = val.extract::<String>()?;
        }
    }
    let circuit_data = circuit.getattr("_data")?.cast::<PyCircuitData>()?.borrow();

    let islayout = !circuit.getattr("layout")?.is_none();

    let exporter = exporter::Exporter::new(
        options.includes,
        options.basis_gates,
        options.disable_constants,
        options.allow_aliasing,
        options.indent,
    );

    let mut output = Vec::new();
    exporter
        .dump(&circuit_data, islayout, &mut output)
        .map_err(|err| {
            QASM3ImporterError::new_err(format!(
                "failed to export circuit using qasm3.dump_experimental: {err:?}"
            ))
        })?;

    let output_str = String::from_utf8(output)
        .map_err(|e| QASM3ImporterError::new_err(format!("invalid utf-8 output: {e:?}")))?;

    stream.call_method1("write", (output_str,))?;

    Ok(())
}

/// Internal module supplying the OpenQASM 3 import capabilities.  The entries in it should largely
/// be re-exposed directly to public Python space.
pub fn qasm3(module: &Bound<PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loads, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    module.add_function(wrap_pyfunction!(dumps, module)?)?;
    module.add_function(wrap_pyfunction!(dump, module)?)?;
    module.add_class::<circuit::PyGate>()?;
    Ok(())
}
