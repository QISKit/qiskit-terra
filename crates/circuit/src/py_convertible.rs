// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::any::TypeId;
use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::types::PyAny;
use smallvec::SmallVec;

use crate::operations::{CustomOperation, Operation, Param};

/// Defines conversion between a [`CustomOperation`] and its Python representation.
pub trait PyConvertible: CustomOperation + Sized {
    /// Create the Python object corresponding to this operation.
    fn create_py_op(
        &self,
        py: Python,
        params: Option<SmallVec<[Param; 3]>>,
        label: Option<&str>,
    ) -> PyResult<Py<PyAny>>;

    /// Try to create this operation from a Python object.
    ///
    /// Returns `Ok(Some(_))` if the object represents this operation, `Ok(None)` if it does not,
    /// and `Err(_)` if the object appears to represent this operation but conversion fails.
    fn extract_from_py(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>>;
}

// CONVERTING FROM RUST TO PYTHON

/// Function used to create a Python object from a custom operation.
pub type CreatePyOp = fn(
    py: Python,
    op: &dyn CustomOperation,
    params: Option<SmallVec<[Param; 3]>>,
    label: Option<&str>,
) -> PyResult<Py<PyAny>>;

/// Entry in the Rust-to-Python conversion table.
pub struct ConversionToPythonEntry {
    /// Returns the `TypeId` of the Rust operation type.
    pub type_id: fn() -> TypeId,

    /// Creates the corresponding Python object.
    pub create: CreatePyOp,
}

/// Rust-to-Python conversion table, initialized once during module setup.
static CONVERSIONS_TO_PYTHON: OnceLock<&'static [ConversionToPythonEntry]> = OnceLock::new();

/// Convert a custom operation to Python after checking its concrete type.
pub fn create_py_op_for<T: PyConvertible>(
    py: Python,
    op: &dyn CustomOperation,
    params: Option<SmallVec<[Param; 3]>>,
    label: Option<&str>,
) -> PyResult<Py<PyAny>> {
    let Some(op) = op.downcast_ref::<T>() else {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "the custom operation '{}' was registered under a mismatched type",
            Operation::name(op)
        )));
    };
    op.create_py_op(py, params, label)
}

/// Find the Python conversion function registered for `op`.
fn find_registered_conversion(op: &dyn CustomOperation) -> Option<CreatePyOp> {
    let type_id = op.type_id();
    let table = CONVERSIONS_TO_PYTHON.get().copied().unwrap_or(&[]);
    table
        .iter()
        .find(|entry| (entry.type_id)() == type_id)
        .map(|entry| entry.create)
}

/// Create the Python object corresponding to `op`.
///
/// # Errors
///
/// Returns an error if `op` has no registered Python conversion.
pub fn create_py_op(
    py: Python,
    op: &dyn CustomOperation,
    params: Option<SmallVec<[Param; 3]>>,
    label: Option<&str>,
) -> PyResult<Py<PyAny>> {
    if let Some(create) = find_registered_conversion(op) {
        return create(py, op, params, label);
    }

    // Future extension point: a C-defined operation registered at runtime would be checked here.

    Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "the custom operation '{}' cannot be exposed to Python",
        Operation::name(op)
    )))
}

/// Register the Rust-to-Python conversion table.
///
/// The first registered table is retained; subsequent registrations are ignored.
pub fn register_conversions_to_python(table: &'static [ConversionToPythonEntry]) {
    let _ = CONVERSIONS_TO_PYTHON.set(table);
}

// CONVERTING FROM PYTHON TO RUST

/// Function used to create a Rust operation from a Python object.
pub type ExtractFromPy =
    fn(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Box<dyn CustomOperation>>>;

/// Converts a Python object to a Rust operation of type `T`.
pub fn extract_from_py_for<T: PyConvertible>(
    ob: Borrowed<'_, '_, PyAny>,
) -> PyResult<Option<Box<dyn CustomOperation>>> {
    Ok(T::extract_from_py(ob)?.map(|op| Box::new(op) as Box<dyn CustomOperation>))
}

/// Entry in the Python-to-Rust conversion table.
///
/// Entries are looked up by the operation's registered Python name. The extractor then determines
/// whether the object represents that operation.
pub struct ConversionFromPythonEntry {
    pub name: &'static str,
    pub extract: ExtractFromPy,
}

/// Python-to-Rust conversion table, initialized once during module setup.
static CONVERSIONS_FROM_PYTHON: OnceLock<&'static [ConversionFromPythonEntry]> = OnceLock::new();

/// Register the Python-to-Rust conversion table.
///
/// The first registered table is retained; subsequent registrations are ignored.
pub fn register_conversions_from_python(table: &'static [ConversionFromPythonEntry]) {
    let _ = CONVERSIONS_FROM_PYTHON.set(table);
}

/// Find the extractor registered for `name`.
pub fn get_extractor(name: &str) -> Option<ExtractFromPy> {
    let table = CONVERSIONS_FROM_PYTHON.get().copied().unwrap_or(&[]);
    table
        .iter()
        .find(|entry| entry.name == name)
        .map(|entry| entry.extract)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::operations::Operation;
    use ndarray::Array2;
    use num_complex::Complex64;
    use pyo3::IntoPyObjectExt;

    /// An operation with a Python class.
    #[derive(Debug, Clone, PartialEq)]
    struct Convertible;

    impl Operation for Convertible {
        fn name(&self) -> &str {
            "test_convertible"
        }
        fn num_qubits(&self) -> u32 {
            1
        }
        fn num_clbits(&self) -> u32 {
            0
        }
        fn num_params(&self) -> u32 {
            0
        }
        fn directive(&self) -> bool {
            false
        }
    }

    impl CustomOperation for Convertible {
        fn is_unitary(&self) -> bool {
            false
        }
    }

    impl PyConvertible for Convertible {
        fn create_py_op(
            &self,
            py: Python,
            _params: Option<SmallVec<[Param; 3]>>,
            _label: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            Ok("converted".into_pyobject(py)?.into_any().unbind())
        }

        fn extract_from_py(_ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>> {
            Ok(Some(Convertible))
        }
    }

    /// An operation whose Python representation carries a label and parameters.
    #[derive(Debug, Clone)]
    struct WithParams {
        params: SmallVec<[Param; 3]>,
        label: Option<String>,
    }

    impl WithParams {
        fn as_float_params(&self) -> Vec<f64> {
            self.params
                .iter()
                .map(|p| match p {
                    Param::Float(f) => *f,
                    _ => panic!("test only uses float params"),
                })
                .collect()
        }
    }

    impl PartialEq for WithParams {
        fn eq(&self, other: &Self) -> bool {
            self.as_float_params() == other.as_float_params() && self.label == other.label
        }
    }

    impl Operation for WithParams {
        fn name(&self) -> &str {
            "test_with_params"
        }
        fn num_qubits(&self) -> u32 {
            1
        }
        fn num_clbits(&self) -> u32 {
            0
        }
        fn num_params(&self) -> u32 {
            self.params.len() as u32
        }
        fn directive(&self) -> bool {
            false
        }
    }

    impl CustomOperation for WithParams {
        fn is_unitary(&self) -> bool {
            false
        }
    }

    impl PyConvertible for WithParams {
        fn create_py_op(
            &self,
            py: Python,
            params: Option<SmallVec<[Param; 3]>>,
            label: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            // Encode as a `(label, params)` tuple so the test can inspect what was passed through.
            let params: Vec<f64> = params
                .unwrap_or_default()
                .iter()
                .map(|p| match p {
                    Param::Float(f) => *f,
                    _ => panic!("test only uses float params"),
                })
                .collect();
            (label, params).into_py_any(py)
        }

        fn extract_from_py(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>> {
            let (label, params): (Option<String>, Vec<f64>) = ob.extract()?;
            Ok(Some(WithParams {
                params: params.into_iter().map(Param::Float).collect(),
                label,
            }))
        }
    }

    /// An operation whose Python representation can decline a lookalike object.
    #[derive(Debug, Clone, PartialEq)]
    struct Picky;

    impl Operation for Picky {
        fn name(&self) -> &str {
            "test_picky"
        }
        fn num_qubits(&self) -> u32 {
            1
        }
        fn num_clbits(&self) -> u32 {
            0
        }
        fn num_params(&self) -> u32 {
            0
        }
        fn directive(&self) -> bool {
            false
        }
    }

    impl CustomOperation for Picky {
        fn is_unitary(&self) -> bool {
            false
        }
    }

    impl PyConvertible for Picky {
        fn create_py_op(
            &self,
            py: Python,
            _params: Option<SmallVec<[Param; 3]>>,
            _label: Option<&str>,
        ) -> PyResult<Py<PyAny>> {
            true.into_py_any(py)
        }

        fn extract_from_py(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>> {
            if ob.get_type().name()? != "bool" {
                return Ok(None);
            }
            Ok(Some(Picky))
        }
    }

    /// An operation with no Python representation.
    #[derive(Debug, Clone, PartialEq)]
    struct RustOnly;

    impl Operation for RustOnly {
        fn name(&self) -> &str {
            "test_rust_only"
        }
        fn num_qubits(&self) -> u32 {
            1
        }
        fn num_clbits(&self) -> u32 {
            0
        }
        fn num_params(&self) -> u32 {
            0
        }
        fn directive(&self) -> bool {
            false
        }
    }

    impl CustomOperation for RustOnly {
        fn is_unitary(&self) -> bool {
            false
        }

        fn matrix(
            &self,
            _params: &[crate::operations::Param],
        ) -> Result<Option<Array2<Complex64>>, Box<dyn std::error::Error>> {
            Ok(None)
        }
    }

    /// The static Rust-to-Python table used by these tests. `CONVERSIONS_TO_PYTHON` can only be
    /// installed once per process, so all tests share this table. `RustOnly` is deliberately
    /// excluded.
    static TEST_CONVERSIONS_TO_PYTHON: &[ConversionToPythonEntry] = &[
        ConversionToPythonEntry {
            type_id: TypeId::of::<Convertible>,
            create: create_py_op_for::<Convertible>,
        },
        ConversionToPythonEntry {
            type_id: TypeId::of::<WithParams>,
            create: create_py_op_for::<WithParams>,
        },
        ConversionToPythonEntry {
            type_id: TypeId::of::<Picky>,
            create: create_py_op_for::<Picky>,
        },
    ];

    /// The static Python-to-Rust table used by these tests.
    static TEST_CONVERSIONS_FROM_PYTHON: &[ConversionFromPythonEntry] = &[
        ConversionFromPythonEntry {
            name: "test_convertible",
            extract: extract_from_py_for::<Convertible>,
        },
        ConversionFromPythonEntry {
            name: "test_with_params",
            extract: extract_from_py_for::<WithParams>,
        },
        ConversionFromPythonEntry {
            name: "test_picky",
            extract: extract_from_py_for::<Picky>,
        },
    ];

    fn ensure_conversions_installed() {
        // Ignore the result: another test in this binary may have installed it already.
        let _ = CONVERSIONS_TO_PYTHON.set(TEST_CONVERSIONS_TO_PYTHON);
        let _ = CONVERSIONS_FROM_PYTHON.set(TEST_CONVERSIONS_FROM_PYTHON);
    }

    #[test]
    fn test_get_extractor() {
        ensure_conversions_installed();

        assert!(get_extractor("test_convertible").is_some());
        assert!(get_extractor("test_absent").is_none());
    }

    #[cfg(not(miri))]
    #[test]
    fn test_create_py_op_uses_registered_conversion() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let op: &dyn CustomOperation = &Convertible;
            let out =
                create_py_op(py, op, None, None).expect("in the static table, so convertible");
            assert_eq!(out.extract::<String>(py).unwrap(), "converted");
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_create_py_op_rejects_mismatched_type() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let op: &dyn CustomOperation = &Convertible;

            let err = create_py_op_for::<WithParams>(py, op, None, None)
                .expect_err("type mismatch should be rejected");

            assert!(err.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py));
            assert!(
                err.value(py)
                    .to_string()
                    .contains("registered under a mismatched type")
            );
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_unregistered_reports_by_name() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let op: &dyn CustomOperation = &RustOnly;
            let err =
                create_py_op(py, op, None, None).expect_err("no Python representation exists");
            assert!(err.is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
            assert!(
                err.value(py).to_string().contains("test_rust_only"),
                "the error should name the operation, got: {err}"
            );
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_create_py_op_threads_params_and_label() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let op = WithParams {
                params: smallvec::smallvec![Param::Float(1.5), Param::Float(-2.0)],
                label: None,
            };
            let out = create_py_op(
                py,
                &op as &dyn CustomOperation,
                Some(op.params.clone()),
                Some("my_label"),
            )
            .expect("in the static table, so convertible");

            let (label, params): (Option<String>, Vec<f64>) = out.extract(py).unwrap();
            assert_eq!(label, Some("my_label".to_string()));
            assert_eq!(params, vec![1.5, -2.0]);
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_extractor_roundtrips_params_and_label() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let ob = (Some("my_label"), vec![1.5_f64, -2.0])
                .into_py_any(py)
                .unwrap();
            let extract = get_extractor("test_with_params").expect("registered by name");
            let extracted = extract(ob.bind(py).as_borrowed())
                .expect("extraction should succeed")
                .expect("object matches this operation");

            let with_params = extracted
                .downcast_ref::<WithParams>()
                .expect("should downcast back to WithParams");
            assert_eq!(
                with_params,
                &WithParams {
                    params: smallvec::smallvec![Param::Float(1.5), Param::Float(-2.0)],
                    label: Some("my_label".to_string()),
                }
            );
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_roundtrip_through_python() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let original = WithParams {
                params: smallvec::smallvec![Param::Float(0.5)],
                label: Some("roundtrip".to_string()),
            };

            let py_op = create_py_op(
                py,
                &original as &dyn CustomOperation,
                Some(original.params.clone()),
                original.label.as_deref(),
            )
            .expect("in the static table, so convertible");

            let extract = get_extractor("test_with_params").expect("registered by name");
            let extracted = extract(py_op.bind(py).as_borrowed())
                .expect("extraction should succeed")
                .expect("object matches this operation");
            let roundtripped = extracted
                .downcast_ref::<WithParams>()
                .expect("should downcast back to WithParams");

            assert_eq!(roundtripped, &original);
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_extract_from_py_can_decline() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let extract = get_extractor("test_picky").expect("registered by name");

            // A real Python `bool` is accepted.
            let matching_ob = true.into_py_any(py).unwrap();
            let accepted =
                extract(matching_ob.bind(py).as_borrowed()).expect("extraction should not error");
            assert!(accepted.is_some());

            // A plain `int` is not: `Picky::extract_from_py` declines it, and the caller sees
            // `Ok(None)` rather than an `Err`.
            let lookalike_ob = 1_i64.into_py_any(py).unwrap();
            let declined =
                extract(lookalike_ob.bind(py).as_borrowed()).expect("declining is not an error");
            assert!(declined.is_none());
        });
    }

    #[cfg(not(miri))]
    #[test]
    fn test_extract_from_py_reports_conversion_error() {
        ensure_conversions_installed();

        Python::initialize();
        Python::attach(|py| {
            let ob = 42_i64.into_py_any(py).unwrap();
            let extract = get_extractor("test_with_params").expect("registered by name");

            assert!(
                extract(ob.bind(py).as_borrowed()).is_err(),
                "malformed Python representation should fail conversion"
            );
        });
    }
}
