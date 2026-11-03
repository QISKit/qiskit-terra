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

mod pass;
mod task;

use anyhow::Context;
use hashbrown::{HashMap, HashSet};
use qiskit_util::dyn_types::*;
use std::{any::Any, sync::LazyLock};

pub use pass::*;
pub use task::*;

/// The pass manager execution environment.
///
/// This contains data managed by the pass manager. A local handle to this is passed into
/// the passes.
#[derive(Default, Debug)]
pub struct PassManagerContext {
    /// The global, catch-all data. The local [PassContext] handles get read-only access to
    /// this data and after pass execution this global state is updated.
    data: HashMap<String, Box<dyn Any + Send + Sync>>,
}
impl PassManagerContext {
    fn new() -> Self {
        Self::default()
    }

    fn update(&mut self, mut updates: ContextUpdates) {
        for (key, value) in updates.insertions.drain() {
            self.data.insert(key, value);
        }
        for key in updates.deletions.iter() {
            self.data.remove(key);
        }
    }

    pub fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        self.data.get(key.as_ref()).map(Box::as_ref)
    }
}

/// A private struct representing the context updates performed. As long as this contains only
/// a HashMap, we could skip this object, but the context is supposed to contain more generic
/// information.
#[derive(Default, Debug)]
struct ContextUpdates {
    /// New values to insert into the global context.
    insertions: HashMap<String, Box<dyn Any + Send + Sync>>,
    /// Keys to delete from the global context.
    deletions: HashSet<String>,
}
impl ContextUpdates {
    fn insert(
        &mut self,
        key: String,
        value: Box<dyn Any + Send + Sync>,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        self.deletions.remove(&key);
        self.insertions.insert(key, value)
    }

    fn delete(&mut self, key: String) -> Option<Box<dyn Any + Send + Sync>> {
        let out = self.insertions.remove(&key);
        self.deletions.insert(key);
        out
    }

    fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        self.insertions.get(key.as_ref()).map(|v| v.as_ref())
    }
}

/// Mutable context information for the pass to interact with the execution pipeline.
///
/// This object is used for two-way information transfer; passes can set flags like `ir_modified` to
/// pass information back to the executing pipeline, or can get/set items in the pipeline's context
/// to pass information between passes.
#[derive(Debug)]
pub struct PassContext<'a> {
    /// A reference to the global execution environment.
    global_context: &'a PassManagerContext,

    /// Whether the pass changed the IR or not.  This defaults to `true`, but passes may set it to
    /// `false` to indicate that they didn't modify the IR.  The pass-manager execution environment
    /// can use that information to optimize caching.
    pub ir_modified: bool,

    /// A local cache of new data.
    updates: ContextUpdates,
}

impl PassContext<'static> {
    /// Get a dummy version of ourselves for use as a default value in situations where we don't
    /// need it to be linked to anything.
    pub fn dummy() -> Self {
        static GLOBAL: LazyLock<PassManagerContext> = LazyLock::new(PassManagerContext::default);
        Self {
            global_context: &GLOBAL,
            ir_modified: true,
            updates: ContextUpdates::default(),
        }
    }
}
impl<'a> PassContext<'a> {
    fn spawn(global_context: &'a PassManagerContext) -> Self {
        Self {
            global_context,
            ir_modified: true,
            updates: ContextUpdates::default(),
        }
    }

    /// Set a new entry in the pass context.
    /// Overwrites the existing value under that key, if it exists.
    ///
    /// Returns the previous local entry, if it existed.
    pub fn set(
        &mut self,
        key: String,
        value: Box<dyn Any + Send + Sync>,
    ) -> Option<Box<dyn Any + Send + Sync>> {
        self.updates.insert(key, value)
    }

    /// Delete the given key.  Returns the corresponding local value, if any.
    pub fn delete(&mut self, key: String) -> Option<Box<dyn Any + Send + Sync>> {
        self.updates.delete(key)
    }

    /// Get an entry, if it exists.
    ///
    /// This first queries from the local context, then the global.
    pub fn get(&self, key: impl AsRef<str>) -> Option<&(dyn Any + Send + Sync)> {
        let key = key.as_ref();

        if self.updates.deletions.contains(key) {
            return None;
        }

        // The local registry takes precedence.
        self.updates.get(key).or_else(|| {
            self.global_context
                .data
                .get(key)
                .map(|value| value.as_ref())
        })
    }
}
/// Types that can be used as an IR by the [`PassManager`].
pub trait IR: DynTyped + Send + Sync + 'static {}

/// Qiskit's pass manager.
///
/// A pass manager is responsible for ensuring that its tasks (including all nested
/// structure) are well-typed at construction time and throughout its lifetime: any
/// two passes that might be executed sequentially must have equal IR types at the
/// boundary.
#[derive(Default, Debug)]
pub struct PassManager {
    tasks: Vec<Task>,
}

impl PassManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run the pass manager on the input IR, and attempt to convert it to a specific output type.
    ///
    /// This is a typed helper wrapper around [`Self::run_erased`].
    pub fn run<IRIn, IROut>(&self, ir: IRIn) -> anyhow::Result<(IROut, PassManagerContext)>
    where
        IRIn: IR,
        IROut: IR,
    {
        let (ir, context) = self.run_erased(Box::new(ir))?;
        (ir as Box<dyn Any>)
            .downcast::<IROut>()
            .map(|ir| (*ir, context))
            .map_err(|_| PassError::Conversion)
            .with_context(|| {
                format!(
                    "trying to convert output to {}",
                    DynTypeId::of::<IROut>().describe()
                )
            })
    }

    /// Run the pass manager
    pub fn run_erased(
        &self,
        mut ir: Box<dyn IR>,
    ) -> Result<(Box<dyn IR>, PassManagerContext), PassError> {
        let mut context = PassManagerContext::new();
        for task in self.tasks.iter() {
            let mut pass_context = PassContext::spawn(&context);
            ir = execute_task(task, ir, &mut pass_context)?;
            let PassContext { updates, .. } = pass_context;
            context.update(updates);
        }
        Ok((ir, context))
    }

    /// The number of first-level tasks in the pass manager.
    ///
    /// Note that this does not count any nested tasks.
    pub fn num_tasks(&self) -> usize {
        self.tasks.len()
    }

    /// Get the type identifier of the input of this pipeline.
    pub fn ir_id_in(&self) -> Option<DynTypeId<'_>> {
        sequence_io_types(self.tasks.iter()).map(|[in_, _]| in_)
    }
    /// Get the type identifier of the output of this pipeline.
    pub fn ir_id_out(&self) -> Option<DynTypeId<'_>> {
        sequence_io_types(self.tasks.iter()).map(|[_, out]| out)
    }

    /// Try to push a [`Task`] onto the end of the task list.
    ///
    /// The task is validated with [`Task::validate`], and its input type is checked against the
    /// output type of the pipeline so far.
    ///
    /// Fails, returning the same task back to the caller along with the reason, if the task is
    /// ill-typed internally or incompatible with the existing pipeline.
    pub fn try_push_task(&mut self, task: Task) -> Result<(), (Task, TaskAppendError)> {
        if let Err(err) = task.validate() {
            return Err((task, TaskAppendError::Task(err)));
        }
        // The existing pipeline is well-typed, and so is the incoming task, so the boundary
        // between the two is the only place a mismatch can arise.
        if let Some((output, input)) = self.ir_id_out().zip(task.io_types().map(|[in_, _]| in_))
            && output != input
        {
            let err = TaskAppendError::boundary(&self.tasks, output, &task, input);
            return Err((task, err));
        }
        self.tasks.push(task);
        Ok(())
    }

    pub fn try_push_static_pass<In: IR, Out: IR>(
        &mut self,
        ob: impl StaticPass<In, Out>,
    ) -> Result<(), (Task, TaskAppendError)> {
        self.try_push_task(Task::Transformation(ob.into_pass()))
    }

    /// Get a reference to a [Task] at a given index.
    pub fn get_task(&self, index: usize) -> Option<&Task> {
        self.tasks.get(index)
    }
}

/// The task runner. This should not be called standalone, passes should be run
/// via the pass manager.
fn execute_task(
    task: &Task,
    mut ir: Box<dyn IR>,
    context: &mut PassContext,
) -> Result<Box<dyn IR>, PassError> {
    // TODO We might be able to only type-check in the pass manager's execution method,
    // since the pipeline is already type-checked upon construction. For now we keep it here,
    // which is the safer choice.
    if let Some([task_in_type, _]) = task.io_types()
        && task_in_type != ir.dyn_type_id()
    {
        return Err(PassError::Conversion);
    }

    match task {
        Task::Transformation(pass) => pass.run(ir, context),
        Task::Sequence(tasks) => {
            for task in tasks.iter() {
                ir = execute_task(task, ir, context)?;
            }
            Ok(ir)
        }
        Task::Stages(stages) => {
            for (_name, task) in stages.iter() {
                ir = execute_task(task, ir, context)?;
            }
            Ok(ir)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::anyhow;
    use std::sync::{Arc, RwLock};

    #[derive(Clone, Debug)]
    struct MyUint(u32);
    static_dyn_typed!(MyUint);
    impl IR for MyUint {}

    #[derive(Debug)]
    struct MyInt(i32);
    static_dyn_typed!(MyInt);
    impl IR for MyInt {}

    struct AddOne;
    impl StaticPass<MyUint> for AddOne {
        fn run(
            &self,
            mut ir: Box<MyUint>,
            _context: &mut PassContext,
        ) -> anyhow::Result<Box<MyUint>> {
            ir.0 += 1;
            Ok(ir)
        }
    }

    struct WriteToContext;
    impl StaticPass<MyUint> for WriteToContext {
        fn run(&self, ir: Box<MyUint>, context: &mut PassContext) -> anyhow::Result<Box<MyUint>> {
            context.set("snapshot".into(), ir.clone());
            context.ir_modified = false;
            Ok(ir)
        }
    }

    struct LeakFromContext<T>(Arc<RwLock<T>>);
    impl<T: Clone + IR + 'static> StaticPass<T> for LeakFromContext<T> {
        fn run(&self, ir: Box<T>, context: &mut PassContext) -> anyhow::Result<Box<T>> {
            let ob = context
                .get("snapshot")
                .ok_or_else(|| anyhow!("object absent"))?;
            let ob = ob
                .downcast_ref::<T>()
                .ok_or_else(|| anyhow!("failed downcast"))?
                .clone();
            *self.0.write().expect("lock shouldn't be poisoned") = ob;
            Ok(ir)
        }
    }

    struct DeleteFromContext;
    impl StaticPass<MyUint> for DeleteFromContext {
        fn run(&self, ir: Box<MyUint>, context: &mut PassContext) -> anyhow::Result<Box<MyUint>> {
            context.delete("snapshot".into());
            context.ir_modified = false;
            Ok(ir)
        }
    }

    struct LowerToInt;
    impl StaticPass<MyUint, MyInt> for LowerToInt {
        fn run(&self, ir: Box<MyUint>, _context: &mut PassContext) -> anyhow::Result<Box<MyInt>> {
            let ir = MyInt(ir.0.try_into().map_err(|_| anyhow!("too big!"))?);
            Ok(Box::new(ir))
        }
    }

    #[test]
    fn test_empty_child_keeps_type_checks() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        assert!(
            pm.try_push_task(Task::Sequence(vec![
                Task::Sequence(vec![]),
                Task::Transformation(AddOne.into_pass()),
            ]))
            .is_err()
        );
    }

    #[test]
    fn test_push_rejects_ill_typed_task() {
        let mut pm = PassManager::new();
        let task = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let (task, err) = pm.try_push_task(task).unwrap_err();
        // The task being appended is itself ill-typed
        let TaskAppendError::Task(TaskTypeError::Sequence { ends }) = &err else {
            panic!("expected a sequence mismatch, got {err}");
        };
        let [output, input] = &**ends;
        assert_eq!(output.task().path(), [TaskPathSegment::Index(0)]);
        assert_eq!(input.task().path(), [TaskPathSegment::Index(1)]);
        // The task is handed back intact, and the pass manager is unchanged.
        assert!(matches!(task, Task::Sequence(tasks) if tasks.len() == 2));
        assert_eq!(pm.num_tasks(), 0);
    }

    #[test]
    fn test_push_rejects_boundary_mismatch() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(LowerToInt).unwrap();
        let (_task, err) = pm
            .try_push_task(Task::Transformation(AddOne.into_pass()))
            .unwrap_err();
        // The existing end is located in the pass manager, the new end in the rejected task.
        let TaskAppendError::Boundary { existing, new } = &err else {
            panic!("expected a boundary mismatch, got {err}");
        };
        assert_eq!(existing.task().path(), [TaskPathSegment::Index(1)]);
        assert_eq!(existing.side(), Side::Output);
        assert_eq!(existing.ty(), "MyInt");
        assert_eq!(new.task().path(), []);
        assert_eq!(new.side(), Side::Input);
        assert_eq!(new.ty(), "MyUint");
        assert!(
            err.to_string().contains("at [1] outputs MyInt")
                && err
                    .to_string()
                    .contains("of the appended task takes MyUint"),
            "{err}"
        );
        assert_eq!(pm.num_tasks(), 2);
    }

    #[test]
    fn test_nested_task_executes() {
        let mut pm = PassManager::new();
        pm.try_push_task(Task::Stages(vec![
            (
                "init".to_string(),
                Task::Sequence(vec![Task::Transformation(AddOne.into_pass())]),
            ),
            ("empty".to_string(), Task::Sequence(vec![])),
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
        ]))
        .unwrap();

        let (MyInt(out), _) = pm.run(MyUint(4)).unwrap();
        assert_eq!(out, 5);
    }

    #[test]
    fn test_pass() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();

        let (out, _) = pm.run::<MyUint, MyUint>(MyUint(4)).unwrap();
        assert_eq!(out.0, 8);
    }

    #[test]
    fn test_incompatible_types() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        assert!(pm.try_push_static_pass(AddOne).is_err());
    }

    #[test]
    fn test_task_retrieval() {
        let make_task = || Task::Transformation(AddOne.into_pass());

        let sequence = Task::Sequence(vec![make_task(), make_task()]);
        let stages = Task::Stages(vec![("one_and_only".to_string(), make_task())]);

        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_task(sequence).unwrap();
        pm.try_push_task(stages).unwrap();

        assert!(matches!(pm.get_task(0), Some(Task::Transformation(_))));

        if let Some(Task::Sequence(tasks)) = pm.get_task(1) {
            assert_eq!(tasks.len(), 2);
        } else {
            panic!("Expected a Task::Sequence");
        }

        if let Some(Task::Stages(stages)) = pm.get_task(2) {
            assert_eq!(stages.len(), 1);
            assert_eq!(stages[0].0, "one_and_only".to_string());
        } else {
            panic!("Expected a Task::Stage");
        }
    }

    #[test]
    fn test_pass_context() {
        let cell = Arc::new(RwLock::new(MyUint(0)));

        let mut pm = PassManager::new();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(LeakFromContext(Arc::clone(&cell)))
            .unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(AddOne).unwrap();
        pm.try_push_static_pass(LowerToInt).unwrap();

        let (MyInt(from_pm), context) = pm.run(MyUint(4)).unwrap();
        assert_eq!(cell.read().unwrap().0, 5);
        let MyUint(from_context) = *context.data["snapshot"].downcast_ref().unwrap();
        assert_eq!(from_context, 6);
        assert_eq!(from_pm, 7);
    }

    #[test]
    fn test_delete_context() {
        let cell = Arc::new(RwLock::new(MyUint(0)));

        let mut pm = PassManager::new();
        pm.try_push_static_pass(WriteToContext).unwrap();
        pm.try_push_static_pass(DeleteFromContext).unwrap();
        pm.try_push_static_pass(LeakFromContext(Arc::clone(&cell)))
            .unwrap();

        let e = pm.run::<_, MyUint>(MyUint(1)).unwrap_err();
        assert!(e.to_string().contains("object absent"), "{:?}", e);
    }

    #[test]
    fn test_pass_error() {
        let mut pm = PassManager::new();
        pm.try_push_static_pass(LowerToInt).unwrap();
        let e = pm.run::<MyUint, MyInt>(MyUint(u32::MAX)).unwrap_err();
        assert!(e.to_string().contains("too big!"), "{:?}", e);
    }

    #[test]
    fn test_pass_name() {
        let add = AddOne.into_pass();
        let lower = LowerToInt.into_pass();
        assert!(add.name().contains("AddOne"));
        assert!(lower.name().contains("LowerToInt"));
    }
}
