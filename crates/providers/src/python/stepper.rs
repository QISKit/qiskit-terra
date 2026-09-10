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

//! The Python binding of the stepper: a program part-way through, and the work it hands over.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::data_tree::{PyDataTree, parse};
use super::program::{PyProgramFunction, PyQuantumProgram, input};
use super::tensor::{tensor, tensor_object, tensor_view};
use super::{position, value_error};
use crate::data_tree::DataTree;
use crate::program::{FunctionId, ProgramStepper, QuantumProgram, RequestId};
use crate::tensor::{Tensor, TensorType};

/// Helps evaluate a :class:`~.QuantumProgram`, requesting an execution for each external
/// function.
///
/// Each step runs everything Qiskit can and hands over one :class:`~.Request` per external
/// call whose operands have arrived, so independent work is dispatched together::
///
///     stepper = ProgramStepper(program, external)
///     while (outputs := stepper.outputs) is None:
///         stepper.step()
///         for request in stepper.outstanding():
///             stepper.fulfil(request.id, sample(request.function, request.inputs))
///
/// Args:
///     program: The program to step through.
///     external: The program functions, by index, that require external evaluation. An index
///         counts from the end when negative.
///     inputs: Arguments to the quantum program, anything coercible into an array-valued
///         :class:`~.DataTree`.
///
/// Raises:
///     ValueError: If ``external`` names the entry point, if the inputs are arranged differently
///         than the program declares or do not match the declared types, or if a function Qiskit
///         is to evaluate holds an instruction it cannot perform.
///     IndexError: If ``external`` addresses no function.
#[pyclass(name = "ProgramStepper", module = "qiskit.quantum_program")]
pub struct PyProgramStepper {
    program: Py<PyQuantumProgram>,
    state: ProgramStepper,
}

impl PyProgramStepper {
    /// Return the program this evaluates.
    fn read(&self) -> &QuantumProgram {
        &self.program.get().0
    }
}

#[pymethods]
impl PyProgramStepper {
    #[new]
    #[pyo3(signature = (program, external, inputs = None, /))]
    fn new(
        py: Python<'_>,
        program: Py<PyQuantumProgram>,
        external: Vec<isize>,
        inputs: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let held = &program.get().0;
        let count = held.functions().len();
        let external = external
            .into_iter()
            .map(|index| Ok(FunctionId::from_index(position(index, count, "function")?)))
            .collect::<PyResult<Vec<_>>>()?;

        // A program taking no inputs is called with none, which is an empty arrangement rather
        // than a value.
        let tree = match inputs {
            Some(inputs) => parse(inputs)?,
            None => DataTree::new(),
        };

        let state = ProgramStepper::new(held, read_inputs(py, held, &tree)?, external)
            .map_err(|error| value_error(&error))?;
        Ok(Self { program, state })
    }

    /// Run everything the answers in hand allow, stopping when only external work is left.
    ///
    /// Every call whose operands have arrived is surfaced by the one step that reaches it.
    /// Stepping again once it has finished does nothing.
    ///
    /// Raises:
    ///     ValueError: If an instruction fails. The stepper is left where it stopped, so the
    ///         same step repeats the same failure.
    fn step(&mut self) -> PyResult<()> {
        let program = &self.program.get().0;
        self.state
            .step(program)
            .map_err(|error| value_error(&error))
    }

    /// Return every request the caller has still to answer.
    ///
    /// Returns:
    ///     One ``Request`` per call handed over, in the order they were raised.
    fn outstanding(&self) -> Vec<PyRequest> {
        self.state
            .outstanding(self.read())
            .map(|request| PyRequest {
                program: self.program.clone(),
                id: request.id(),
                function: request.function(),
                inputs: request.inputs().to_vec(),
                output_types: request.output_types().to_vec(),
            })
            .collect()
    }

    /// Answer the request ``request_id`` with one array per result the call declares.
    ///
    /// Nothing is evaluated here. The next :meth:`step` continues from what has arrived.
    ///
    /// Args:
    ///     request_id: Which request to answer, as :attr:`~.Request.id` gives it.
    ///     outputs: One value per declared result, each read with :func:`numpy.asarray`.
    ///
    /// Raises:
    ///     ValueError: If the request is not outstanding, which includes one already answered, or
    ///         if ``outputs`` is the wrong length or holds a value of the wrong type. The request
    ///         stays outstanding in each case, so one request failing leaves the others untouched.
    #[pyo3(signature = (request_id, outputs, /))]
    fn fulfil(&mut self, request_id: usize, outputs: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        let outputs = outputs
            .iter()
            .map(tensor)
            .collect::<PyResult<Vec<Tensor>>>()?;
        let program = &self.program.get().0;
        self.state
            .fulfil(program, RequestId::from_index(request_id), outputs)
            .map_err(|error| value_error(&error))
    }

    /// Return the program's outputs, arranged as its output structure, or ``None`` if the
    /// evaluation is unfinished.
    #[getter]
    fn outputs(&self, py: Python<'_>) -> Option<PyDataTree> {
        let outputs = self.state.outputs(self.read())?;
        let structure = outputs.structure();
        let arrays: Vec<Py<PyAny>> = outputs
            .into_leaves()
            .map(|value| tensor_object(py, value))
            .collect();
        Some(PyDataTree(structure.unflatten(arrays).expect(
            "a structure has one leaf per leaf of the tree it came from",
        )))
    }

    fn __repr__(&self) -> String {
        let finished = if self.state.outputs(self.read()).is_some() {
            "True"
        } else {
            "False"
        };
        format!(
            "ProgramStepper({} outstanding, finished={finished})",
            self.state.outstanding(self.read()).count(),
        )
    }
}

/// One piece of work a :class:`~.ProgramStepper` has handed over.
///
/// A request names the function to perform, the arrays to perform it on, and the types the answer
/// must satisfy. It is returned to the stepper by id, through :meth:`~.ProgramStepper.fulfil`.
#[pyclass(name = "Request", module = "qiskit.quantum_program", frozen)]
pub struct PyRequest {
    program: Py<PyQuantumProgram>,
    id: RequestId,
    function: FunctionId,
    inputs: Vec<Tensor>,
    output_types: Vec<TensorType>,
}

#[pymethods]
impl PyRequest {
    /// This request's identity within the stepper that raised it. Pass it to
    /// :meth:`~.ProgramStepper.answer`.
    #[getter]
    fn id(&self) -> usize {
        self.id.index()
    }

    /// The function to perform.
    ///
    /// Returns:
    ///     A reader over that function.
    #[getter]
    fn function(&self) -> PyProgramFunction {
        PyProgramFunction {
            program: self.program.clone(),
            id: self.function,
        }
    }

    /// The function to perform, by its position in the program's definition order.
    #[getter]
    fn function_id(&self) -> usize {
        self.function.index()
    }

    /// One array per parameter of that function, in declaration order.
    ///
    /// Each is a read-only array over the request's own buffer. A bit-valued tensor is a copy,
    /// because NumPy spells a bit ``bool``.
    #[getter]
    fn inputs<'py>(slf: &Bound<'py, Self>) -> Vec<Bound<'py, PyAny>> {
        slf.get()
            .inputs
            .iter()
            // SAFETY: this class is frozen and owns the tensor, so no mutable reference to the
            // buffer exists. The array holds a reference to the object, which keeps it alive.
            .map(|value| unsafe { tensor_view(value, slf.clone().into_any()) })
            .collect()
    }

    /// The type each array of the answer must satisfy, one per result of that function.
    #[getter]
    fn output_types(&self) -> Vec<TensorType> {
        self.output_types.clone()
    }

    fn __repr__(&self) -> String {
        format!("Request({}, {})", self.id, self.function)
    }
}

/// Return the inputs `tree` describes, as one tensor per declared input of `program`.
///
/// The arrangement is checked whole before any value is read, and each value is then checked
/// against the type its own input declares.
fn read_inputs(
    py: Python<'_>,
    program: &QuantumProgram,
    tree: &DataTree<Py<PyAny>>,
) -> PyResult<DataTree<Tensor>> {
    let structure = program.input_structure();
    let actual = tree.structure();
    if actual != *structure {
        return Err(PyValueError::new_err(format!(
            "inputs are structured {actual} but the program declares {structure}"
        )));
    }
    let declared = program.input_types();
    let paths = structure.dotted_paths();
    let mut values = Vec::with_capacity(paths.len());
    for ((object, ty), path) in tree.iter_leaves().zip(declared.iter_leaves()).zip(&paths) {
        // A lone input has no path of its own, and is the first one either way.
        let where_ = if path.is_empty() { "0" } else { path.as_str() };
        values.push(input(object.bind(py), ty, where_)?);
    }
    Ok(structure
        .unflatten(values)
        .expect("one value per leaf of the structure they were read against"))
}
