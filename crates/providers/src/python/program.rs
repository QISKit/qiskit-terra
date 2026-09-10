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

//! The Python binding of a program function being built, and of the program it becomes.

use std::collections::BTreeMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList};

use super::data_tree::{ObjectTree, PyDataTree};
use super::ops::{self, PyProgramOp};
use super::tensor::{tensor, tensor_object};
use super::{position, value_error};
use crate::data_tree::DataTree;
use crate::program::{
    FunctionId, InstructionId, InstructionRef, InstructionRole, InstructionView, ProgramFunction,
    QuantumProgram, Value,
};
use crate::tensor::{Tensor, TensorType};
use crate::{partition, render};

/// One tensor value: an output slot of the instruction that produces it.
#[pyclass(
    name = "Value",
    module = "qiskit.quantum_program",
    frozen,
    eq,
    from_py_object,
    hash
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyValue(Value);

#[pymethods]
impl PyValue {
    /// The instruction producing this value, by its position in the function holding it.
    #[getter]
    fn instruction(&self) -> usize {
        self.0.instruction().index()
    }

    /// Which of that instruction's results this value is.
    #[getter]
    fn slot(&self) -> usize {
        self.0.slot()
    }

    fn __repr__(&self) -> String {
        format!("Value({})", self.0)
    }
}

/// A dataflow graph being assembled, instruction by instruction.
///
/// Every instruction is type-checked as it is added, so the function cannot be malformed. Its
/// parameters and results are positional: they are added in the order the structures given to
/// [`seal`](Self::seal) name them.
#[pyclass(
    name = "FunctionBuilder",
    module = "qiskit._accelerate.quantum_program"
)]
pub struct PyFunctionBuilder(ProgramFunction);

#[pymethods]
impl PyFunctionBuilder {
    #[new]
    fn new() -> Self {
        Self(ProgramFunction::new())
    }

    /// Declare a parameter of type `ty`, returning its value.
    #[pyo3(signature = (ty, /))]
    fn add_parameter(&mut self, ty: TensorType) -> PyValue {
        PyValue(self.0.add_parameter(ty))
    }

    /// Apply `op` to `operands`, returning the values it produces.
    #[pyo3(signature = (op, operands, /))]
    fn add_op(&mut self, op: &PyProgramOp, operands: Vec<PyValue>) -> PyResult<Vec<PyValue>> {
        let operands: Vec<Value> = operands.into_iter().map(|value| value.0).collect();
        self.0
            .add_boxed_op(op.op.to_owned(), &operands)
            .map(|values| values.into_iter().map(PyValue).collect())
            .map_err(|error| value_error(&error))
    }

    /// Declare `value` as the next result.
    #[pyo3(signature = (value, /))]
    fn add_result(&mut self, value: PyValue) -> PyResult<()> {
        self.0
            .add_result(value.0)
            .map_err(|error| value_error(&error))
    }

    /// Seal this function into a program whose inputs and outputs are arranged as the given trees.
    ///
    /// Only the shape and the names of the trees are read; their leaves are discarded. The builder
    /// is left empty.
    #[pyo3(signature = (inputs, outputs, /))]
    fn seal(&mut self, inputs: &PyDataTree, outputs: &PyDataTree) -> PyResult<PyQuantumProgram> {
        let function = std::mem::take(&mut self.0);
        QuantumProgram::new(vec![function], inputs.0.structure(), outputs.0.structure())
            .map(PyQuantumProgram)
            .map_err(|error| value_error(&error))
    }
}

/// A hybrid quantum-classical computation, described rather than performed.
///
/// A program declares the type of every input it takes and every output it produces, so both are
/// known before it runs. Calling it supplies one keyword argument per input and gives back a
/// `DataTree` of results arranged as `output_types()` describes.
#[pyclass(name = "QuantumProgram", module = "qiskit.quantum_program", frozen)]
pub struct PyQuantumProgram(pub(super) QuantumProgram);

#[pymethods]
impl PyQuantumProgram {
    /// Return the declared type of every input, arranged as the program's input structure.
    ///
    /// Returns:
    ///     A data tree of tensor types.
    fn input_types(&self, py: Python<'_>) -> PyResult<PyDataTree> {
        let types = self.0.input_types();
        Ok(PyDataTree(object_tree(&types, |ty| {
            Ok(Py::new(py, ty.clone())?.into_any())
        })?))
    }

    /// Return the type of every output, arranged as the program's output structure.
    ///
    /// Type inference ran as the program was built, so this needs no evaluation.
    ///
    /// Returns:
    ///     A data tree of tensor types.
    fn output_types(&self, py: Python<'_>) -> PyResult<PyDataTree> {
        let types = self.0.output_types();
        Ok(PyDataTree(object_tree(&types, |ty| {
            Ok(Py::new(py, ty.clone())?.into_any())
        })?))
    }

    /// Evaluate the program on one keyword argument per declared input.
    ///
    /// Each argument is read with `numpy.asarray` and must then match its declared type: the
    /// program is monomorphic, so nothing is promoted here.
    ///
    /// Args:
    ///     inputs: One value per declared input, by keyword.
    ///
    /// Returns:
    ///     A data tree of arrays, arranged as the program's output structure.
    ///
    /// Raises:
    ///     TypeError: If the keywords are not the declared inputs.
    ///     ValueError: If a value does not match the type its input declares, or if the program
    ///         holds work Qiskit cannot perform in process.
    #[pyo3(signature = (**inputs))]
    fn __call__(&self, py: Python<'_>, inputs: Option<&Bound<'_, PyDict>>) -> PyResult<PyDataTree> {
        let declared = self.0.input_types();
        let expected = keyword_inputs(&declared)?;

        let names: Vec<&str> = expected.iter().map(|&(name, _)| name).collect();
        let mut arguments = Vec::with_capacity(expected.len());
        let mut missing = Vec::new();
        for &(name, _) in &expected {
            match inputs.map(|inputs| inputs.get_item(name)).transpose()? {
                Some(Some(argument)) => arguments.push(argument),
                _ => missing.push(name),
            }
        }
        let mut unexpected = Vec::new();
        for key in inputs.iter().flat_map(|inputs| inputs.keys()) {
            let key = key.extract::<String>()?;
            if !names.contains(&key.as_str()) {
                unexpected.push(key);
            }
        }
        if !missing.is_empty() || !unexpected.is_empty() {
            return Err(PyTypeError::new_err(format!(
                "this program takes inputs {}; missing {}, unexpected {}",
                name_list(&names),
                name_list(&missing),
                name_list(&unexpected),
            )));
        }

        let mut tree = DataTree::with_capacity(expected.len());
        for ((name, ty), argument) in expected.into_iter().zip(arguments) {
            tree.insert_leaf(name, input(&argument, ty, name)?)
                .map_err(|err| value_error(&err))?;
        }

        let outputs = self.0.eval(tree).map_err(|error| value_error(&error))?;
        let structure = outputs.structure();
        let arrays: Vec<Py<PyAny>> = outputs
            .into_leaves()
            .map(|value| tensor_object(py, value))
            .collect();
        Ok(PyDataTree(structure.unflatten(arrays).expect(
            "a structure has one leaf per leaf of the tree it came from",
        )))
    }

    /// Rewrite this program to put each execution resource's work in a function of its own.
    ///
    /// Each such function is called once from the entry point. Operations not declared by any
    /// resource stay in the entry function.
    ///
    /// Args:
    ///     resources: The op names each execution resource handles, one sequence per resource, as
    ///         specified by ``Instruction.full_name``.
    ///
    /// Returns:
    ///     The rewritten program, and the resource each of its functions belongs to.
    ///
    /// Raises:
    ///     ValueError: If this program's entry point already calls a function, if two resources
    ///         declare one op, or if an op no resource declares is one Qiskit cannot evaluate in
    ///         process.
    #[pyo3(signature = (resources, /))]
    fn partition(&self, resources: Vec<Vec<String>>) -> PyResult<(Self, Vec<usize>)> {
        partition(&self.0, resources)
            .map(|(program, table)| (Self(program), table))
            .map_err(|error| value_error(&error))
    }

    /// Return this program as a listing of every instruction it holds, one function per block.
    ///
    /// Returns:
    ///     The listing, which is also what ``str()`` of a program gives.
    fn listing(&self) -> String {
        render::listing(&self.0)
    }

    /// Draw this program's dataflow as a graph, one box per instruction.
    ///
    /// Returns:
    ///     The drawing, as a ``PIL.Image.Image``.
    ///
    /// Raises:
    ///     MissingOptionalLibraryError: If Graphviz or Pillow is missing.
    fn draw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyModule::import(py, "qiskit.quantum_program._render")?
            .call_method1("_image", (render::dot(&self.0),))
    }

    /// How many functions the program holds.
    #[getter]
    fn num_functions(&self) -> usize {
        self.0.functions().len()
    }

    /// The entry point, which is the last function the program defines.
    ///
    /// Returns:
    ///     A reader over the function the program starts at.
    #[getter]
    fn entry(slf: &Bound<'_, Self>) -> PyProgramFunction {
        PyProgramFunction {
            program: slf.clone().unbind(),
            id: slf.get().0.entry(),
        }
    }

    /// Return the function at ``index`` in definition order.
    ///
    /// A call may only name an earlier function, so the order is one every function could run in,
    /// ending at the entry point.
    ///
    /// Args:
    ///     index: Which function to read, counted from the end when negative.
    ///
    /// Returns:
    ///     A reader over that function.
    ///
    /// Raises:
    ///     IndexError: If ``index`` addresses no function.
    #[pyo3(signature = (index, /))]
    fn function(slf: &Bound<'_, Self>, index: isize) -> PyResult<PyProgramFunction> {
        let index = position(index, slf.get().0.functions().len(), "function")?;
        Ok(PyProgramFunction {
            program: slf.clone().unbind(),
            id: FunctionId::from_index(index),
        })
    }

    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        let functions: Vec<PyProgramFunction> = (0..slf.get().0.functions().len())
            .map(|index| PyProgramFunction {
                program: slf.clone().unbind(),
                id: FunctionId::from_index(index),
            })
            .collect();
        PyList::new(slf.py(), functions)?.into_any().try_iter()
    }

    /// Return how many instructions the program holds of each op.
    fn _type_name_counts(&self) -> BTreeMap<String, usize> {
        let mut counts = BTreeMap::new();
        for function in self.0.functions() {
            for instruction in function.iter_instructions() {
                *counts.entry(instruction.full_name()).or_insert(0) += 1;
            }
        }
        counts
    }

    fn __str__(&self) -> String {
        render::listing(&self.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "QuantumProgram(inputs={}, outputs={})",
            self.0.input_structure(),
            self.0.output_structure()
        )
    }
}

/// One function of a program: what it consumes, what it produces, and the instructions between.
///
/// A reader holds the program rather than a copy of the function, and resolves through it on every
/// access. Its instructions are in an order every one of them could run in, so an instruction's
/// operands always come from earlier in the function.
#[pyclass(name = "ProgramFunction", module = "qiskit.quantum_program", frozen)]
pub struct PyProgramFunction {
    pub(super) program: Py<PyQuantumProgram>,
    pub(super) id: FunctionId,
}

impl PyProgramFunction {
    /// Return the function this reads.
    fn read(&self) -> &ProgramFunction {
        self.program
            .get()
            .0
            .function(self.id)
            .expect("a reader is only made for a function its program holds")
    }

    /// Return a reader over the instruction at `id` of this function.
    fn instruction_at(&self, id: InstructionId) -> PyInstruction {
        PyInstruction {
            program: self.program.clone(),
            function: self.id,
            id,
        }
    }
}

#[pymethods]
impl PyProgramFunction {
    /// Where this function sits in the program's definition order.
    #[getter]
    fn index(&self) -> usize {
        self.id.index()
    }

    /// Whether this is the program's entry point.
    #[getter]
    fn is_entry(&self) -> bool {
        self.id == self.program.get().0.entry()
    }

    /// Return the type of each parameter, in the order the function takes them.
    ///
    /// Returns:
    ///     One tensor type per parameter.
    fn input_types(&self) -> Vec<TensorType> {
        self.read().signature().inputs
    }

    /// Return the type of each result, in the order the function declares them.
    ///
    /// Returns:
    ///     One tensor type per result.
    fn output_types(&self) -> Vec<TensorType> {
        self.read().signature().outputs
    }

    /// The instructions declaring this function's parameters, in order.
    #[getter]
    fn parameters(&self) -> Vec<PyInstruction> {
        self.read()
            .parameters()
            .iter()
            .map(|&id| self.instruction_at(id))
            .collect()
    }

    /// The instructions declaring this function's results, in order.
    #[getter]
    fn results(&self) -> Vec<PyInstruction> {
        self.read()
            .results()
            .iter()
            .map(|&id| self.instruction_at(id))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.read().instruction_count()
    }

    fn __getitem__(&self, index: isize) -> PyResult<PyInstruction> {
        let index = position(index, self.read().instruction_count(), "instruction")?;
        Ok(self.instruction_at(InstructionId::from_index(index)))
    }

    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        let reader = slf.get();
        let instructions: Vec<PyInstruction> = reader
            .read()
            .iter_instructions()
            .map(|instruction| reader.instruction_at(instruction.id()))
            .collect();
        PyList::new(slf.py(), instructions)?.into_any().try_iter()
    }

    fn __repr__(&self) -> String {
        format!(
            "ProgramFunction(@{}, {} instructions)",
            self.id.index(),
            self.read().instruction_count()
        )
    }
}

/// One instruction of a program function: what it does, what it reads, and what it produces.
#[pyclass(name = "Instruction", module = "qiskit.quantum_program", frozen)]
pub struct PyInstruction {
    program: Py<PyQuantumProgram>,
    function: FunctionId,
    id: InstructionId,
}

impl PyInstruction {
    /// Return the instruction this reads.
    fn read(&self) -> InstructionRef<'_> {
        self.program
            .get()
            .0
            .function(self.function)
            .expect("a reader is only made for a function its program holds")
            .instruction(self.id)
            .expect("a reader is only made for an instruction its function holds")
    }
}

#[pymethods]
impl PyInstruction {
    /// This instruction's id, which is its position in the function holding it.
    #[getter]
    fn id(&self) -> usize {
        self.id.index()
    }

    /// What part this instruction plays in its function.
    #[getter]
    fn role(&self) -> InstructionRole {
        self.read().role()
    }

    /// The type name of what this instruction does, qualified by its namespace.
    ///
    /// A backend dispatches on this name. A parameter, a call and a result report
    /// ``qiskit.parameter``, ``qiskit.call`` and ``qiskit.result``.
    #[getter]
    fn full_name(&self) -> String {
        self.read().full_name()
    }

    /// A summary of the op's payload, such as ``axis=0``, and ``None`` for anything else.
    #[getter]
    fn describe(&self) -> Option<String> {
        self.read().describe()
    }

    /// The values this instruction consumes, in operand order.
    #[getter]
    fn operands(&self) -> Vec<PyValue> {
        self.read()
            .operands()
            .iter()
            .copied()
            .map(PyValue)
            .collect()
    }

    /// The values this instruction produces, in the order it produces them.
    #[getter]
    fn outputs(&self) -> Vec<PyValue> {
        self.read().outputs().map(PyValue).collect()
    }

    /// Return the type of each operand, read from the instructions producing them.
    ///
    /// Returns:
    ///     One tensor type per operand.
    fn operand_types(&self) -> Vec<TensorType> {
        self.read().operand_types().cloned().collect()
    }

    /// Return the type of each value this instruction produces.
    ///
    /// These are flat: how an operation arranges its results is its own to report, and a built
    /// program keeps only the program's own input and output structures.
    ///
    /// Returns:
    ///     One tensor type per value produced.
    fn output_types(&self) -> Vec<TensorType> {
        self.read().output_types().to_vec()
    }

    /// The operation this instruction applies, and ``None`` for a parameter, a call or a result.
    ///
    /// Returns:
    ///     The op, as the class of its type, or the ``ProgramOp`` base class for an op defined
    ///     outside Qiskit.
    #[getter]
    fn op<'py>(slf: &Bound<'py, Self>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match slf.get().read().view() {
            InstructionView::Op(op) => ops::op_object(slf.py(), op.to_owned()).map(Some),
            _ => Ok(None),
        }
    }

    /// The function this instruction calls, and ``None`` unless it is a call.
    ///
    /// Returns:
    ///     A reader over the callee.
    #[getter]
    fn callee(slf: &Bound<'_, Self>) -> Option<PyProgramFunction> {
        match slf.get().read().view() {
            InstructionView::Call(callee) => Some(PyProgramFunction {
                program: slf.get().program.clone(),
                id: callee,
            }),
            _ => None,
        }
    }

    /// The function this instruction calls, by its position in the program's definition order, and
    /// ``None`` unless it is a call.
    ///
    /// A ``ProgramStepper`` declares a function external by this position, and a ``Request``
    /// reports it.
    #[getter]
    fn callee_id(&self) -> Option<usize> {
        match self.read().view() {
            InstructionView::Call(callee) => Some(callee.index()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Instruction(@{}, {}, {})",
            self.function.index(),
            self.id.index(),
            self.read().full_name()
        )
    }
}

/// Read `object` as the tensor for the input named `where_`, of declared type `ty`.
///
/// A program is monomorphic, so nothing is promoted here: a value that does not match the declared
/// type is refused, naming both types.
pub(super) fn input(object: &Bound<'_, PyAny>, ty: &TensorType, where_: &str) -> PyResult<Tensor> {
    let value = tensor(object)?;
    if !value.matches(ty) {
        return Err(PyValueError::new_err(format!(
            "input '{where_}': expected {ty}, got {}",
            value.tensor_type()
        )));
    }
    Ok(value)
}

/// Format `names` as Python renders a list of strings.
fn name_list(names: &[impl AsRef<str>]) -> String {
    let names: Vec<String> = names
        .iter()
        .map(|name| format!("'{}'", name.as_ref()))
        .collect();
    format!("[{}]", names.join(", "))
}

/// Return the name and declared type of each input, in the order the program takes them.
fn keyword_inputs(declared: &DataTree<TensorType>) -> PyResult<Vec<(&str, &TensorType)>> {
    let refuse = || {
        Err(PyTypeError::new_err(format!(
            "calling a program by keyword needs an input structure of named values, and this one \
             is {}",
            declared.structure()
        )))
    };
    if matches!(declared, DataTree::Leaf(_)) {
        return refuse();
    }
    let mut inputs = Vec::with_capacity(declared.len());
    for (name, child) in declared.iter_children() {
        let (Some(name), DataTree::Leaf(ty)) = (name, child) else {
            return refuse();
        };
        inputs.push((name, ty));
    }
    Ok(inputs)
}

/// Return `tree` with each leaf converted to a Python object by `object`.
fn object_tree<T>(
    tree: &DataTree<T>,
    mut object: impl FnMut(&T) -> PyResult<Py<PyAny>>,
) -> PyResult<ObjectTree> {
    let leaves = tree
        .iter_leaves()
        .map(&mut object)
        .collect::<PyResult<Vec<_>>>()?;
    Ok(tree
        .structure()
        .unflatten(leaves)
        .expect("a structure has one leaf per leaf of the tree it came from"))
}
