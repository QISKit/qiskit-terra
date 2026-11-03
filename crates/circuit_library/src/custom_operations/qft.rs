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

use ndarray::Array2;
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray2, PyArrayDescr, PyArrayDescrMethods, PyUntypedArrayMethods};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use smallvec::SmallVec;
use std::error;
use std::f64::consts::PI;
use std::hash::{DefaultHasher, Hash, Hasher};
use thiserror::Error;

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::operations::{CustomOperation, Operation, Param};
use qiskit_circuit::py_convertible::PyConvertible;
use qiskit_synthesis::qft::qft_decompose_full::synth_qft_full;
use qiskit_util::py::ImportOnceCell;

#[derive(Debug, Error)]
pub enum QftError {
    #[error("matrix for a {0}-qubit QFT is too large to construct")]
    MatrixLimitExceeded(u32),
}

impl From<QftError> for PyErr {
    fn from(error: QftError) -> Self {
        match error {
            QftError::MatrixLimitExceeded(_) => PyValueError::new_err(error.to_string()),
        }
    }
}

/// The Python `QFTGate` class used to wrap a [`PyQftGate`].
static QFT_GATE: ImportOnceCell =
    ImportOnceCell::new("qiskit.circuit.library.basis_change.qft", "QFTGate");

/// Quantum Fourier Transform gate.
///
/// On `n` qubits, the QFT is defined by
///
/// ```text
/// |j> -> 1/sqrt(2^n) * sum_k exp(2 pi i j k / 2^n) |k>
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QftGate {
    num_qubits: u32,
}

impl QftGate {
    pub fn new(num_qubits: u32) -> Self {
        Self { num_qubits }
    }

    /// The number of qubits this gate acts on.
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }
}

impl Operation for QftGate {
    fn name(&self) -> &str {
        "qft"
    }

    fn num_qubits(&self) -> u32 {
        self.num_qubits
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

impl CustomOperation for QftGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn matrix(
        &self,
        _params: &[Param],
    ) -> Result<Option<Array2<Complex64>>, Box<dyn error::Error>> {
        let size = 1usize
            .checked_shl(self.num_qubits)
            .ok_or(QftError::MatrixLimitExceeded(self.num_qubits))?;
        let norm = (size as f64).sqrt().recip();
        Ok(Some(Array2::from_shape_fn((size, size), |(i, j)| {
            let phase = 2.0 * PI * (i * j) as f64 / (size as f64);
            Complex64::from_polar(norm, phase)
        })))
    }

    fn definition(&self, _params: &[Param]) -> Option<CircuitData> {
        synth_qft_full(self.num_qubits as usize, true, 0, false)
            .ok()
            .map(CircuitData::from)
    }
}

/// Python-facing wrapper around [`QftGate`].
#[pyclass(module = "qiskit._accelerate.circuit_library", from_py_object)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyQftGate {
    inner: QftGate,
}

#[pymethods]
impl PyQftGate {
    #[new]
    fn py_new(num_qubits: u32) -> Self {
        Self {
            inner: QftGate::new(num_qubits),
        }
    }

    #[getter]
    #[pyo3(name = "num_qubits")]
    fn py_num_qubits(&self) -> u32 {
        self.inner.num_qubits
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.num_qubits.hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        format!("QftGate({})", self.inner.num_qubits)
    }

    /// Return the same instance for `copy.copy`.
    fn __copy__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Return the same instance for `copy.deepcopy`.
    #[pyo3(signature = (_memo=None))]
    fn __deepcopy__(slf: Py<Self>, _memo: Option<&Bound<PyAny>>) -> Py<Self> {
        slf
    }

    /// Support `pickle`. This class is held by the public `qiskit.circuit.library.QFTGate` in a
    /// private `_inner` attribute rather than pickled directly as part of a circuit, but is
    /// still made picklable in its own right for convenience and consistency with other native
    /// types.
    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (py.get_type::<Self>(), (self.inner.num_qubits,)).into_py_any(py)
    }

    /// Return the QFT unitary matrix as a NumPy array.
    #[pyo3(name = "matrix")]
    fn py_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<Complex64>>> {
        let matrix = self
            .inner
            .matrix(&[])
            .map_err(|e| match e.downcast::<QftError>() {
                Ok(err) => (*err).into(),
                Err(err) => PyRuntimeError::new_err(err.to_string()),
            })?
            .ok_or_else(|| PyRuntimeError::new_err("QftGate has no matrix representation"))?;
        Ok(matrix.into_pyarray(py))
    }

    /// Support the NumPy array protocol, e.g. `np.asarray(gate)`.
    ///
    /// A new array is always created, so `copy=False` is rejected.
    #[pyo3(signature = (dtype=None, copy=None))]
    fn __array__<'py>(
        &self,
        py: Python<'py>,
        dtype: Option<&Bound<'py, PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if copy == Some(false) {
            return Err(PyValueError::new_err(
                "unable to avoid copy while creating an array as requested",
            ));
        }
        let array = self.py_matrix(py)?;
        let base_dtype = array.dtype();
        let dtype = dtype
            .map(|dtype| PyArrayDescr::new(py, dtype))
            .unwrap_or_else(|| Ok(base_dtype.clone()))?;
        if dtype.is_equiv_to(&base_dtype) {
            return Ok(array.into_any());
        }
        PyModule::import(py, intern!(py, "numpy"))?
            .getattr(intern!(py, "array"))?
            .call(
                (array,),
                Some(&[(intern!(py, "dtype"), dtype.as_any())].into_py_dict(py)?),
            )
    }

    fn definition(&self) -> PyResult<PyCircuitData> {
        let defn = self
            .inner
            .definition(&[])
            .expect("QFT should have definition");

        Ok(PyCircuitData { inner: defn })
    }
}

impl PyQftGate {
    pub fn new(inner: QftGate) -> Self {
        Self { inner }
    }

    pub fn inner(&self) -> &QftGate {
        &self.inner
    }

    pub fn into_inner(self) -> QftGate {
        self.inner
    }
}

/// Provides the Python conversion for [`QftGate`].
impl PyConvertible for QftGate {
    /// Wrap `self` in [`PyQftGate`] and construct a Python `QFTGate` around it.
    ///
    /// `QFTGate` has no parameters or label, so `params` and `label` are ignored.
    fn create_py_op(
        &self,
        py: Python,
        _params: Option<SmallVec<[Param; 3]>>,
        _label: Option<&str>,
    ) -> PyResult<Py<PyAny>> {
        let inner = PyQftGate::new(self.clone());
        Ok(QFT_GATE
            .get_bound(py)
            .call_method1(intern!(py, "_from_inner"), (inner,))?
            .unbind())
    }

    /// Extract a [`QftGate`] from an exact Python `QFTGate`.
    ///
    /// Returns `Ok(None)` for other objects, including `QFTGate` subclasses.
    fn extract_from_py(ob: Borrowed<'_, '_, PyAny>) -> PyResult<Option<Self>> {
        if !ob.get_type().is(QFT_GATE.get_bound(ob.py())) {
            return Ok(None);
        }
        let Ok(inner) = ob.getattr(intern!(ob.py(), "_inner")) else {
            return Ok(None);
        };
        Ok(inner.extract::<PyQftGate>().ok().map(PyQftGate::into_inner))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::OperationRef;
    use qiskit_circuit::packed_instruction::PackedOperation;

    // Tests basic QFT gate properties
    #[test]
    /// Tests basic QFT gate properties.
    fn test_qft() {
        let qft3 = QftGate::new(3);
        let another_qft3 = QftGate::new(3);
        assert_eq!(qft3, another_qft3);

        let qft4 = QftGate::new(4);
        assert_ne!(qft3, qft4);

        let mat = CustomOperation::matrix(&qft3, &[]);
        assert!(matches!(mat, Ok(Some(_))));
    }

    /// Tests that requesting the matrix of an unreasonably large QFT reports an error.
    #[test]
    fn test_qft_matrix_too_large() {
        let qft = QftGate::new(64);
        let err = qft.matrix(&[]).expect_err("size should overflow");
        assert!(matches!(
            err.downcast_ref::<QftError>(),
            Some(QftError::MatrixLimitExceeded(_))
        ));
    }

    /// Tests that the gate definition uses the expected QFT synthesis.
    #[test]
    fn test_qft_definition() {
        let qft = QftGate::new(3);
        let definition =
            CustomOperation::definition(&qft, &[]).expect("QftGate should have a definition");
        let expected =
            CircuitData::from(synth_qft_full(3, true, 0, false).expect("synthesis should succeed"));
        assert_eq!(definition.num_qubits(), expected.num_qubits());
        assert_eq!(definition.data().len(), expected.data().len());
    }

    /// Tests that a QFT gate can be stored in and retrieved from a circuit.
    #[test]
    fn test_qft_rountrip() {
        let qft = QftGate::new(4);

        let mut qc = CircuitData::with_capacity(1, 0, 1, 0.0.into())
            .expect("Circuit with small capacity should be built.");
        let qft_op = PackedOperation::from_custom_operation(Box::new(qft.clone()));
        qc.push_packed_operation(qft_op, None, &[Qubit(0)], &[])
            .expect("Instruction should be added to the circuit.");

        let retrieved_op = &qc.data()[0];

        let OperationRef::CustomOperation(dyn_cast_op) = retrieved_op.op.view() else {
            panic!("Gate should be a custom operation");
        };

        let Some(downcast_op) = dyn_cast_op.downcast_ref::<QftGate>() else {
            panic!("Gate should be a custom gate of type QftGate");
        };

        assert!(downcast_op.is_unitary());
        assert_eq!(downcast_op.num_qubits(), 4);
        assert_eq!(downcast_op, &qft);
    }

    /// Tests that the QFT conversion is registered.
    #[test]
    fn test_python_conversion_registered() {
        // Ignore the result: another test in this binary may have registered already.
        let _ = crate::custom_operations::register_custom_operations();

        // Python -> Rust, keyed by the operation name.
        assert_eq!("qft", Operation::name(&QftGate::new(3)));
        assert!(qiskit_circuit::py_convertible::get_extractor("qft").is_some());
    }
}
