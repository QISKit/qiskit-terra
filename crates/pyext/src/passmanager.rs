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

use std::any::Any;
use std::mem;
use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::gc::PyVisit;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyType};
use pyo3::{PyTraverseError, import_exception, intern};

use qiskit_circuit::circuit_data::{CircuitData, PyCircuitData};
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::operations::Param;
use qiskit_passmanager::{IR, PassContext, Task};
use qiskit_util::dyn_types::*;
use qiskit_util::py::ImportOnceCell;

import_exception!(qiskit.passmanager, LoweringPassManagerError);

// TODO: there'll be an abstraction we can make about `PyIrExposer` and the idea of
// "Python-exposable types", but we're not solving everything in a PR series that's running late.
trait PyIrExposer: Send + Sync + 'static {
    fn object_dyn_type_id(&self) -> DynTypeId<'_>;
    /// Move the data in `ob` into a suitable Python object.
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>>;
    /// Extract the object from Python, without cloning.
    ///
    /// This is permitted to assume that it is given the only copy of `ob` from which reading is
    /// permitted.  In particular, implementers may replace the data in `ob` with (valid) nonsense,
    /// to leave the object in a valid state while having extracted all its data.
    ///
    /// Implementers can assume that they should be able to take out a mutable reference to any
    /// inner data, and should return a Python exception if they cannot.
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>>;
    fn gc_traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError>;
}

struct PyDagExposer;
impl PyIrExposer for PyDagExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        DynTypeId::of::<DAGCircuit>()
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let owned = (ob as Box<dyn Any>)
            .downcast::<DAGCircuit>()
            .expect("caller should ensure typing");
        PyDAGCircuit::from(*owned)
            .into_pyobject(py)
            .map(|ob| ob.into_any())
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        let py_dag = ob.cast_into::<PyDAGCircuit>()?;
        let mut py_dag = py_dag.try_borrow_mut()?;
        // Per documentation, we're not required to leave `ob`'s data intact.
        let dag =
            mem::replace(&mut *py_dag, PyDAGCircuit::from(DAGCircuit::default())).into_inner();
        Ok(Box::new(dag))
    }
    fn gc_traverse(&self, _visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

struct PyCircuitExposer;
impl PyIrExposer for PyCircuitExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        DynTypeId::of::<CircuitData>()
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let owned = (ob as Box<dyn Any>)
            .downcast::<CircuitData>()
            .expect("caller should ensure typing");
        PyCircuitData::from(*owned)
            .into_pyobject(py)
            .map(|ob| ob.into_any())
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        let py_circuit = ob.cast_into::<PyCircuitData>()?;
        let mut py_circuit = py_circuit.try_borrow_mut()?;
        // TODO: this should be the infallible implementation of `Default for CircuitData`.
        let empty = CircuitData::with_capacity(0, 0, 0, Param::Float(0.0))?;
        // Per documentation, we're not required to leave `ob`'s data intact.
        let circuit = mem::replace(&mut *py_circuit, PyCircuitData::from(empty)).inner;
        Ok(Box::new(circuit))
    }
    fn gc_traverse(&self, _visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        Ok(())
    }
}

struct PyObjectExposer(Arc<PyIrBase>);
impl PyIrExposer for PyObjectExposer {
    fn object_dyn_type_id(&self) -> DynTypeId<'_> {
        PyIr::dyn_type_for_base(&self.0)
    }
    fn to_py<'py>(&self, py: Python<'py>, ob: Box<dyn IR>) -> PyResult<Bound<'py, PyAny>> {
        let py_ir = (ob as Box<dyn Any>)
            .downcast::<PyIr>()
            .expect("caller should ensure typing");
        Ok(py_ir.ob.into_bound(py))
    }
    fn steal_from_py<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Box<dyn IR>> {
        Ok(Box::new(PyIr {
            ob: ob.unbind(),
            base: Arc::clone(&self.0),
        }))
    }
    fn gc_traverse(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.0.ty)
    }
}

pub struct PyIr {
    ob: Py<PyAny>,
    base: Arc<PyIrBase>,
}
impl PyIr {
    fn dyn_type_for_base(base: &PyIrBase) -> DynTypeId<'_> {
        DynTypeId::of::<Self>().with_dynamic(base.ty.as_ptr().cast(), &base.name)
    }
}
impl DynTyped for PyIr {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        Self::dyn_type_for_base(&self.base)
    }
}
impl IR for PyIr {}

struct PyIrBase {
    name: String,
    ty: Py<PyType>,
}

// TODO: we might want to cache these, rather than recreating them every time.  If we want to store
// them on the Python `type` object, we a) need to do proper integration with the GC so we don't
// make the type immortal via a reference cycle and b) handle the case of static extension types
// which are already immortal but don't support attaching arbitrary Python attributes.  In the
// static-type case, since they're immortal, though, it doesn't matter if we store things in a
// static; the memory use can't be unbounded because static types (naturally) have a bounded number
// in the process.
fn ir_exposer(mut ty: Bound<PyType>) -> PyResult<Box<dyn PyIrExposer>> {
    static PY_IR: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "IR");
    let py = ty.py();
    if ty.hasattr(intern!(py, "_qiskit_ir_builtin_"))? {
        if ty.is_subclass_of::<PyCircuitData>()? {
            return Ok(Box::new(PyCircuitExposer));
        }
        if ty.is_subclass_of::<PyDAGCircuit>()? {
            return Ok(Box::new(PyDagExposer));
        }
        return Err(PyTypeError::new_err(format!(
            "{} declares it is a builtin IR type, but it is not handled",
            ty.repr()?
        )));
    }
    if !ty.is_subclass(PY_IR.get_bound(py))? {
        return Err(PyTypeError::new_err(format!(
            "{} does not implement `IR`",
            ty.repr()?
        )));
    }
    // TODO: if the logic gets more complex, we might want to consider the creation of the
    // `PyIrBase` object into Python via some `pyclass`.
    if let Some(base_ty) = ty
        .getattr(intern!(py, "_qiskit_ir_base_"))?
        .extract::<Option<Bound<PyType>>>()?
    {
        ty = base_ty;
    }
    let base = PyIrBase {
        name: ty.getattr(intern!(py, "_qiskit_ir_name_"))?.extract()?,
        ty: ty.unbind(),
    };
    Ok(Box::new(PyObjectExposer(Arc::new(base))))
}

/// Private Rust-Python boundary object that represents a handle to a pass coming from Python.
#[pyclass(module = "qiskit._accelerate.passmanager")]
pub struct PyPass(Option<PyPassInner>);
impl PyPass {
    /// Take the inner pass out of the handle.
    ///
    /// [`PyPassInner`] owns a boxed exposer, so it cannot be cloned out and has to be moved.
    /// Callers must pass a freshly constructed handle, which makes the failure unreachable.
    fn steal(slf: &Bound<Self>) -> PyResult<PyPassInner> {
        slf.borrow_mut()
            .0
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("pass was already consumed"))
    }
}
#[pymethods]
impl PyPass {
    #[new]
    fn py_new(
        ob: Bound<PyAny>,
        name: String,
        ir_in: Bound<PyType>,
        ir_out: Bound<PyType>,
    ) -> PyResult<Self> {
        let inner = PyPassInner {
            ob: ob.unbind(),
            name,
            ir_in: ir_exposer(ir_in)?,
            ir_out: ir_exposer(ir_out)?,
        };
        Ok(Self(Some(inner)))
    }

    fn __repr__(&self) -> String {
        match self.0.as_ref() {
            None => "<PyPass: consumed>".to_owned(),
            Some(inner) => format!("<PyPass: {}>", inner.name),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(inner) = self.0.as_ref() else {
            return Ok(());
        };
        visit.call(&inner.ob)?;
        inner.ir_in.gc_traverse(visit)?;
        inner.ir_out.gc_traverse(visit)?;
        Ok(())
    }
    // We are not required to do anything in `__clear__` because our Python references are all
    // immutable.
}

static NATIVE_PASS: ImportOnceCell = ImportOnceCell::new(
    "qiskit.passmanager.lowering_passmanager",
    "_native_pass_from_lowering_pass",
);
static PASS: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "Pass");
static TASK_SEQUENCE: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "TaskSequence");
static TASK_STAGES: ImportOnceCell = ImportOnceCell::new("qiskit.passmanager", "TaskStages");

/// The deepest task tree that can be converted to or from Python.
///
/// Python task containers are arbitrarily nestable, including recursively so.
const MAX_TASK_DEPTH: usize = 1000;

/// Build the native task that corresponds to a `qiskit.passmanager` object.
///
/// The set of [`Task`] variants is closed, so the accepted Python types are exactly `Pass`,
/// `TaskSequence` and `TaskStages`.  This is the inverse of [`task_to_py`].
fn task_from_py(ob: &Bound<PyAny>) -> PyResult<Task> {
    task_from_py_inner(ob, 0)
}

fn task_from_py_inner(ob: &Bound<PyAny>, depth: usize) -> PyResult<Task> {
    let py = ob.py();
    if depth >= MAX_TASK_DEPTH {
        return Err(PyValueError::new_err(format!(
            "tasks are nested deeper than the maximum of {MAX_TASK_DEPTH}"
        )));
    }
    if ob.is_instance(PASS.get_bound(py))? {
        // Python owns the interpretation of the `Pass` protocol, including any
        // calling-convention resolution, and hands back a `PyPass` handle.
        let handle = NATIVE_PASS.get_bound(py).call1((ob,))?;
        let pass = PyPass::steal(handle.cast::<PyPass>()?)?;
        return Ok(Task::Transformation(Box::new(pass)));
    }
    if ob.is_instance(TASK_SEQUENCE.get_bound(py))? {
        let tasks = ob
            .getattr(intern!(py, "tasks"))?
            .try_iter()?
            .map(|task| task_from_py_inner(&task?, depth + 1))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Task::Sequence(tasks));
    }
    if ob.is_instance(TASK_STAGES.get_bound(py))? {
        let stages = ob
            .getattr(intern!(py, "stages"))?
            .cast_into::<PyDict>()?
            .iter()
            .map(|(name, task)| {
                Ok((
                    name.extract::<String>()?,
                    task_from_py_inner(&task, depth + 1)?,
                ))
            })
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Task::Stages(stages));
    }
    Err(PyTypeError::new_err(format!(
        "expected a Pass, TaskSequence or TaskStages, got {}: {}",
        ob.get_type().name()?,
        ob.repr()?
    )))
}

/// Expose a native task to Python as the corresponding `qiskit.passmanager` object.
fn task_to_py<'py>(py: Python<'py>, task: &Task) -> PyResult<Bound<'py, PyAny>> {
    match task {
        Task::Transformation(pass) => {
            let Some(inner) = (pass.as_ref() as &dyn Any).downcast_ref::<PyPassInner>() else {
                return Err(PyRuntimeError::new_err(format!(
                    "the pass '{}' did not come from Python",
                    pass.name()
                )));
            };
            Ok(inner.ob.bind(py).clone())
        }
        Task::Sequence(tasks) => {
            let children = tasks
                .iter()
                .map(|task| task_to_py(py, task))
                .collect::<PyResult<Vec<_>>>()?;
            TASK_SEQUENCE
                .get_bound(py)
                .call1((PyList::new(py, children)?,))
        }
        Task::Stages(stages) => {
            let children = stages
                .iter()
                .map(|(name, task)| Ok((name, task_to_py(py, task)?)))
                .collect::<PyResult<Vec<_>>>()?;
            TASK_STAGES
                .get_bound(py)
                .call1((PyList::new(py, children)?,))
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "this task has no Python representation: {task:?}"
        ))),
    }
}

/// The actual logic of Rust wrapping passes that come from Python.  This is what implements `Pass`.
struct PyPassInner {
    name: String,
    ob: Py<PyAny>,
    ir_in: Box<dyn PyIrExposer>,
    ir_out: Box<dyn PyIrExposer>,
}
impl PyPassInner {
    /// The internal logic of the [`Pass::run`](qiskit_passmanager::Pass::run) method, but with an
    /// explicit attachment to an interpreter, and returning `PyErr` so we can centralise the error
    /// handling.
    fn run_py(
        &self,
        py: Python,
        ir: Box<dyn IR>,
        context: PassContextHandle,
    ) -> PyResult<Box<dyn IR>> {
        static CONTEXT_WRAPPER: ImportOnceCell =
            ImportOnceCell::new("qiskit.passmanager", "PassContextHandle");
        let ir = self.ir_in.to_py(py, ir)?;
        let ctx = CONTEXT_WRAPPER.get_bound(py).call1((context,))?;
        let res = self
            .ob
            .bind(py)
            .getattr(intern!(py, "_qiskit_pass_run_"))?
            .call1((ir, ctx))?;
        self.ir_out.steal_from_py(res)
    }
}
impl qiskit_passmanager::Pass for PyPassInner {
    #[inline]
    fn ir_id_in(&self) -> DynTypeId<'_> {
        self.ir_in.object_dyn_type_id()
    }

    #[inline]
    fn ir_id_out(&self) -> DynTypeId<'_> {
        self.ir_out.object_dyn_type_id()
    }

    #[inline]
    fn name(&self) -> &str {
        &self.name
    }

    #[inline]
    fn run(
        &self,
        ir: Box<dyn IR>,
        context: &mut PassContext,
    ) -> Result<Box<dyn IR>, qiskit_passmanager::PassError> {
        // SAFETY: we guarantee that `stolen` will drop (and therefore replace `context`) before
        // control flow leaves this function by leaving ownership of it here.
        let stolen = unsafe { StolenContext::new(context) };

        Python::attach(|py| {
            // SAFETY: per documentation of `PassContextHandle`, no methods allow moving the
            // internal `PassContext` from it, and we ensured that `stolen` drops as this function
            // ends, so the lifetime of the internal object actually ends with `'a`.
            let handle = unsafe { stolen.handle() };
            self.run_py(py, ir, handle)
                .map_err(|e| qiskit_passmanager::PassError::Runtime(e.into()))
        })
    }
}

struct StolenContext<'a, 'b> {
    whence: &'b mut PassContext<'a>,
    shared: Arc<RwLock<Option<PassContext<'static>>>>,
}
impl<'a, 'b> StolenContext<'a, 'b> {
    unsafe fn new(whence: &'b mut PassContext<'a>) -> Self {
        let stolen = mem::replace(whence, PassContext::dummy());
        // SAFETY: per lifetime rules, 'a outlives us. If nothing else moves the `PassContext` out
        // from us, we can put it back in `whence` in our `Drop`, guaranteeing the lifetime.
        let stolen = unsafe { mem::transmute::<PassContext<'a>, PassContext<'static>>(stolen) };
        Self {
            whence,
            shared: Arc::new(RwLock::new(Some(stolen))),
        }
    }

    unsafe fn handle(&self) -> PassContextHandle {
        PassContextHandle(Arc::clone(&self.shared))
    }
}
impl<'a, 'b> Drop for StolenContext<'a, 'b> {
    fn drop(&mut self) {
        let mut lock = self.shared.write().expect("lock should not poison");
        _ = mem::replace(
            self.whence,
            lock.take()
                .expect("no other references should allow taking the inner object"),
        );
    }
}

/// Rust-native wrapper that mediates access to a lifetime-bound `&mut PassContext`.
///
/// # Safety
///
/// This is in general *unsafe*: the [`PassContext`] object is actually backed by a lifetime-bound
/// object which is type-erased to `'static`.  The creator of this object must ensure that the inner
/// `PassContext` is removed before its lifetime actually expires.
///
/// No method on this class may ever move a `PassContext` out of the `Option` (or delete it, etc).
///
/// Methods on this handle object should attempt to hold locks from the [`RwLock`] for as short
/// periods as possible.
#[pyclass(frozen, module = "qiskit._accelerate.passmanager")]
pub struct PassContextHandle(Arc<RwLock<Option<PassContext<'static>>>>);
impl PassContextHandle {
    #[inline]
    fn with_borrow<F, T>(&self, use_fn: F) -> PyResult<T>
    where
        F: FnOnce(&PassContext<'static>) -> T,
    {
        let Ok(inner) = self.0.read() else {
            return Err(PyRuntimeError::new_err("internal lock was poisoned"));
        };
        let ctx = inner.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("attempted to use handle after pass returned")
        })?;
        Ok(use_fn(ctx))
    }

    #[inline]
    fn with_borrow_mut<F, T>(&self, use_fn: F) -> PyResult<T>
    where
        F: FnOnce(&mut PassContext<'static>) -> T,
    {
        let Ok(mut inner) = self.0.write() else {
            return Err(PyRuntimeError::new_err("internal lock was poisoned"));
        };
        let ctx = inner.as_mut().ok_or_else(|| {
            PyRuntimeError::new_err("attempted to use handle after pass returned")
        })?;
        Ok(use_fn(ctx))
    }
}
#[pymethods]
impl PassContextHandle {
    pub fn get_context(&self, py: Python, key: String, default: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.with_borrow(|ctx| {
            let Some(val) = ctx.get(&key) else {
                return Ok(default);
            };
            let Some(ob) = val.downcast_ref::<Py<PyAny>>() else {
                return Err(PyTypeError::new_err(format!(
                    "'{}' did not correspond to a Python object",
                    &key
                )));
            };
            Ok(ob.clone_ref(py))
        })?
    }
    pub fn set_context(&self, key: String, val: Py<PyAny>) -> PyResult<()> {
        let val = Box::new(val);
        // Release the lock before dropping a (maybe) Python object whose destructor might run
        // arbitrary code.
        let _: Option<_> = self.with_borrow_mut(|ctx| ctx.set(key, val))?;
        Ok(())
    }
    pub fn del_context(&self, key: String) -> PyResult<()> {
        // Release the lock before dropping a (maybe) Python object whose destructor might run
        // arbitrary code.  In Python space we raise `KeyError` if the key is not present, even if
        // the actual deletion from the global context is deferred.
        let _: Option<_> = self.with_borrow_mut(|ctx| match ctx.get(&key) {
            Some(_) => Ok(ctx.delete(key)),
            None => Err(PyKeyError::new_err(key)),
        })??;
        Ok(())
    }
    pub fn get_ir_modified(&self) -> PyResult<bool> {
        self.with_borrow(|ctx| ctx.ir_modified)
    }
    pub fn set_ir_modified(&self, val: bool) -> PyResult<()> {
        self.with_borrow_mut(|ctx| ctx.ir_modified = val)
    }
}

/// Wrapper around the Rust-native pass manager; we can take this as owned when we create it from
/// Python or give a Rust-created one to Python.
#[pyclass(name = "PassManager", module = "qiskit._accelerate.passmanager")]
pub struct PyPassManager(qiskit_passmanager::PassManager);
#[pymethods]
impl PyPassManager {
    // TODO: the native `PassManager` contains type-erased objects that can (and _will_, via
    // `PyPassInner`) own Python references.  Badly behaved Python objects can cause reference
    // cycles that fail to be collected because we can't correctly integrate with the GC's
    // `tp_traverse` slot (PyO3's `__traverse__`).  We need to rework the dynamic-typing system to
    // allow arbitrary Rust and C objects to optionally integrate with the traversal logic, without
    // requiring Python in the base interfaces.

    #[new]
    pub fn py_new() -> Self {
        Self(Default::default())
    }

    /// Push a task onto the end of the pass manager.
    ///
    /// The task is validated recursively, and its input type is checked against the output type of
    /// the pipeline so far.  The pass manager is left unmodified if either check fails.
    ///
    /// Raises:
    ///     TypeError: if `task` is not a `Pass`, `TaskSequence` or `TaskStages`, if it is
    ///         ill-typed internally, or if it does not follow on from the pipeline so far.
    pub fn push_task(&mut self, task: &Bound<PyAny>) -> PyResult<()> {
        match self.0.try_push_task(task_from_py(task)?) {
            Ok(()) => Ok(()),
            Err((_task, err)) => Err(PyTypeError::new_err(err.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        match self.0.num_tasks() {
            1 => "<LoweringPassManager: 1 task>".to_owned(),
            n => format!("<LoweringPassManager: {n} tasks>"),
        }
    }

    /// Convert the pipeline into the corresponding `qiskit.passmanager` container objects.
    ///
    /// Raises:
    ///     RuntimeError: if any pass did not come from Python, or if a task has no Python
    ///         representation.
    pub fn to_py_tasks<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        (0..self.0.num_tasks())
            .map(|index| {
                task_to_py(
                    py,
                    self.0
                        .get_task(index)
                        .expect("index is below the task count"),
                )
            })
            .collect()
    }

    pub fn run_simple<'py>(&self, ir: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = ir.py();
        let exposer_in = ir_exposer(ir.get_type())?;
        let Some((ir_id_in, ir_id_out)) = self.0.ir_id_in().zip(self.0.ir_id_out()) else {
            return Ok(ir);
        };
        let actual_in_id = exposer_in.object_dyn_type_id();
        if ir_id_in != actual_in_id {
            return Err(PyTypeError::new_err(format!(
                "incoming IR of type {} does not match expected {}",
                actual_in_id.describe(),
                ir_id_in.describe()
            )));
        }
        match self.0.run_erased(exposer_in.steal_from_py(ir)?) {
            Ok((ir, _ctx)) => {
                // TODO: this is a disgusting hack.  We _should_ have a way to associate a
                // `PyIrExposer` / `CIrExposer` with the actual `DynTypeId` object in some form, but
                // we don't have time to do it properly.  In lieu of that, we just try all the
                // options...
                let base_id = ir.dyn_type_id().to_static();
                if base_id == DynTypeId::of::<CircuitData>() {
                    PyCircuitExposer.to_py(py, ir)
                } else if base_id == DynTypeId::of::<DAGCircuit>() {
                    PyDagExposer.to_py(py, ir)
                } else if base_id == DynTypeId::of::<PyIr>() {
                    Ok((ir as Box<dyn Any>)
                        .downcast::<PyIr>()
                        .expect("`Any` type should match `DynTypeId`")
                        .ob
                        .into_bound(py))
                } else {
                    Err(PyTypeError::new_err(format!(
                        "cannot expose out IR of type {} to Python",
                        ir_id_out.describe()
                    )))
                }
            }
            Err(e @ qiskit_passmanager::PassError::Conversion) => {
                Err(PyTypeError::new_err(e.to_string()))
            }
            Err(qiskit_passmanager::PassError::Runtime(e)) => {
                // TODO: at some point we should add more structure.
                let outer = LoweringPassManagerError::new_err(e.to_string());
                // If there's a Python exception in the chain, we'll attach it as our exception's
                // "cause".  We'll assume that if there are multiple Python exceptions, then it was
                // the propagator's responsibility to chain them, if appropriate.
                if let Some(pyerr) = e.chain().find_map(|e| e.downcast_ref::<PyErr>()) {
                    outer.set_cause(py, Some(pyerr.clone_ref(py)));
                }
                Err(outer)
            }
        }
    }
}

#[pymodule(name = "passmanager")]
pub fn passmanager_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPass>()?;
    m.add_class::<PyPassManager>()?;
    m.add_class::<PassContextHandle>()?;
    Ok(())
}
