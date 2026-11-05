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

use std::{fmt, iter, slice};

use thiserror::Error;

use crate::pass::Pass;
use qiskit_util::dyn_types::DynTypeId;

/// A task in Qiskit's compiler framework.
///
/// This is a single unit of execution flow. Each task variant describes a different mode of
/// flow, ranging from the simple execution of a single pass, to sequences of other tasks,
/// to structured flow control such as loops and conditionals.
///
/// It is allowable and expected for a given task instance to have incorrect IR typing up until
/// the point that it is owned by a [`PassManager`](crate::PassManager), which then assumes the
/// responsibility of validating and maintaining correct IR type flow for its lifetime. This
/// allows a [Task] to represent a partially assembled pipeline.
#[derive(Debug)]
#[non_exhaustive]
pub enum Task {
    // TODO Add Loop and Switch with conditions that can be set from Python/C and
    // proper error handlings that occur during the condition evaluation.
    /// A single pass.
    Transformation(Box<dyn Pass>),
    /// A sequence of tasks.
    Sequence(Vec<Task>),
    /// A sequence of named tasks.
    Stages(Vec<(String, Task)>),
}

impl Task {
    /// Return the input and output IR types, or `None` if not defined.
    pub fn io_types(&self) -> Option<[DynTypeId<'_>; 2]> {
        match self {
            Task::Transformation(pass) => Some([pass.ir_id_in(), pass.ir_id_out()]),
            Task::Sequence(tasks) => sequence_io_types(tasks.iter()),
            Task::Stages(stages) => sequence_io_types(stages.iter().map(|(_name, task)| task)),
        }
    }

    /// Validate (recursively) that the IR typing of this task is self-consistent.
    ///
    /// The paths in a returned error are relative to this task.
    pub fn validate(&self) -> Result<(), TaskTypeError> {
        validate(self, &mut Vec::new())
    }

    /// The kind of this task.
    pub fn kind(&self) -> TaskKind {
        match self {
            Task::Transformation(pass) => TaskKind::Transformation(pass.name().to_owned()),
            Task::Sequence(_) => TaskKind::Sequence,
            Task::Stages(_) => TaskKind::Stages,
        }
    }

    /// The children of this task and their respective locations.
    fn children(&self) -> Children<'_> {
        match self {
            Task::Transformation(_) => Children::None,
            Task::Sequence(tasks) => Children::Sequence(tasks.iter().enumerate()),
            Task::Stages(stages) => Children::Stages(stages.iter()),
        }
    }

    /// The child that the IR type on `side` originates from, or `None` if this task is itself the origin.
    ///
    /// This method is for debugging purposes: use it to try and find out who is responsible for setting
    /// the type of either the input or output type. As currently implemented, a task is the origin either
    /// because it is a single pass, or because its type is the shared responsibility of more than one
    /// child, such as a switch task.
    fn origin_child(&self, side: Side) -> Option<(SegmentRef<'_>, &Task)> {
        match self {
            Task::Transformation(_) => None,
            Task::Sequence(_) | Task::Stages(_) => {
                let mut typed = self
                    .children()
                    .filter(|(_, task)| task.io_types().is_some());
                match side {
                    Side::Input => typed.next(),
                    Side::Output => typed.last(),
                }
            }
        }
    }
}

/// An iterator over the children of a [Task].
enum Children<'a> {
    None,
    Sequence(iter::Enumerate<slice::Iter<'a, Task>>),
    Stages(slice::Iter<'a, (String, Task)>),
}
impl<'a> Iterator for Children<'a> {
    type Item = (SegmentRef<'a>, &'a Task);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::None => None,
            Self::Sequence(tasks) => tasks
                .next()
                .map(|(index, task)| (SegmentRef::Index(index), task)),
            Self::Stages(stages) => stages
                .next()
                .map(|(name, task)| (SegmentRef::Stage(name), task)),
        }
    }
}

/// One side of a task's IR typing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Input,
    Output,
}
impl Side {
    fn verb(self) -> &'static str {
        match self {
            Self::Input => "takes",
            Self::Output => "outputs",
        }
    }
}

/// Return the input and output IR types of a sequence of tasks, or `None` if no task in the
/// sequence has defined types.
pub(crate) fn sequence_io_types<'a>(
    tasks: impl DoubleEndedIterator<Item = &'a Task>,
) -> Option<[DynTypeId<'a>; 2]> {
    let mut typed = tasks.filter_map(|task| task.io_types());
    let first = typed.next()?;
    let last = typed.next_back().unwrap_or(first);
    Some([first[0], last[1]])
}

/// Validate a task whose location in a parent task is `path`.
///
/// `path` lists the outermost segment of the task-tree first. It is restored to its original contents
/// on success, and left in an arbitrary state on failure.
fn validate<'a>(task: &'a Task, path: &mut Vec<SegmentRef<'a>>) -> Result<(), TaskTypeError> {
    match task {
        Task::Transformation(_) => Ok(()),
        Task::Sequence(_) | Task::Stages(_) => validate_sequence(task, path),
    }
}

/// Validate a task whose children run in order.
fn validate_sequence<'a>(
    task: &'a Task,
    path: &mut Vec<SegmentRef<'a>>,
) -> Result<(), TaskTypeError> {
    let mut previous: Option<(SegmentRef<'a>, &'a Task, DynTypeId<'a>)> = None;
    for (segment, child) in task.children() {
        path.push(segment);
        validate(child, path)?;
        path.pop();

        let Some([input, output]) = child.io_types() else {
            continue;
        };
        if let Some((previous_segment, previous_task, previous_output)) = previous
            && previous_output != input
        {
            // Both ends are children of this task, so they share a path but for the last segment.
            // `path` is not restored, since this returns rather than continuing the walk.
            path.push(previous_segment);
            let output = TaskEndpoint::new(previous_task, path, Side::Output, previous_output);
            *path.last_mut().expect("just pushed") = segment;
            let input = TaskEndpoint::new(child, path, Side::Input, input);
            return Err(TaskTypeError::Sequence {
                ends: Box::new([output, input]),
            });
        }
        previous = Some((segment, child, output));
    }
    Ok(())
}

/// One step of the path that locates a [Task] within the tree of tasks that contains it.
///
/// This may be extended if some future [Task] needs a new way to index itself.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskPathSegment {
    /// The position within a [`Task::Sequence`].
    Index(usize),
    /// The name of a stage of a [`Task::Stages`].
    Stage(String),
}
impl fmt::Display for TaskPathSegment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(index) => write!(f, "[{index}]"),
            Self::Stage(name) => write!(f, "['{}']", name.escape_debug()),
        }
    }
}

/// A [TaskPathSegment] that borrows from the task tree being validated.
#[derive(Clone, Copy, Debug)]
enum SegmentRef<'a> {
    Index(usize),
    Stage(&'a str),
}
impl SegmentRef<'_> {
    fn to_owned(self) -> TaskPathSegment {
        match self {
            Self::Index(index) => TaskPathSegment::Index(index),
            Self::Stage(name) => TaskPathSegment::Stage(name.to_owned()),
        }
    }
}

/// The variant of a [Task] as described in error messages.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TaskKind {
    Transformation(String),
    Sequence,
    Stages,
}
impl fmt::Display for TaskKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transformation(name) => write!(f, "the pass '{}'", name.escape_debug()),
            Self::Sequence => f.write_str("the sequence"),
            Self::Stages => f.write_str("the stages"),
        }
    }
}

/// Describes a task within a tree of tasks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskRef {
    /// The path from the root of the tree, outermost segment first.
    path: Vec<TaskPathSegment>,
    kind: TaskKind,
}
impl TaskRef {
    /// The path from the root of the tree the task was found in, outermost segment first.
    pub fn path(&self) -> &[TaskPathSegment] {
        &self.path
    }

    /// The kind of task this refers to.
    pub fn kind(&self) -> &TaskKind {
        &self.kind
    }
}
impl fmt::Display for TaskRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.kind.fmt(f)?;
        if self.path.is_empty() {
            return Ok(());
        }
        f.write_str(" at ")?;
        for segment in self.path.iter() {
            segment.fmt(f)?;
        }
        Ok(())
    }
}

/// Describes the location of an IR within a task tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskEndpoint {
    task: TaskRef,
    side: Side,
    ty: String,
}
impl TaskEndpoint {
    /// The task the type originates from.
    pub fn task(&self) -> &TaskRef {
        &self.task
    }

    /// Which of the task's two types this is.
    pub fn side(&self) -> Side {
        self.side
    }

    /// The IR type, typically [described](qiskit_util::dyn_types::DynTypeId::short_describe).
    pub fn ty(&self) -> &str {
        &self.ty
    }

    /// Create a new endpoint from a [Task] that is at hand.
    fn new(mut task: &Task, path: &[SegmentRef<'_>], side: Side, ty: DynTypeId<'_>) -> Self {
        let mut path: Vec<_> = path.iter().copied().map(SegmentRef::to_owned).collect();
        while let Some((segment, child)) = task.origin_child(side) {
            path.push(segment.to_owned());
            task = child;
        }
        Self {
            task: TaskRef {
                path,
                kind: task.kind(),
            },
            side,
            ty: ty.short_describe().to_owned(),
        }
    }
}
impl fmt::Display for TaskEndpoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} {}", self.task, self.side.verb(), self.ty)
    }
}

/// An IR typing constraint within a [Task] was violated.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TaskTypeError {
    /// Two tasks in a sequence-like disagree.
    ///
    /// These are usually adjacent, but may not be if there are empty sequences between
    /// them, for example. They may point to culprits at different levels of nesting.
    #[error("{}, but {}", ends[0], ends[1])]
    Sequence { ends: Box<[TaskEndpoint; 2]> },
}

/// A [Task] can not be appended to a [`PassManager`](crate::PassManager).
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TaskAppendError {
    /// The task is ill-typed in itself, regardless of what it is appended to.
    #[error("the task being appended is ill-typed: {0}")]
    Task(TaskTypeError),
    /// The task disagrees with the output type of the existing pipeline.
    #[error(
        "cannot append the task: {}, but {} of the appended task {} {}",
        existing, new.task, new.side.verb(), new.ty
    )]
    Boundary {
        existing: Box<TaskEndpoint>,
        new: Box<TaskEndpoint>,
    },
}
impl TaskAppendError {
    /// A mismatch at the boundary between a sequence of `tasks` and an appended `task`.
    pub(crate) fn boundary(
        tasks: &[Task],
        output: DynTypeId<'_>,
        task: &Task,
        input: DynTypeId<'_>,
    ) -> Self {
        let (index, last) = tasks
            .iter()
            .enumerate()
            .rev()
            .find(|(_, task)| task.io_types().is_some())
            .expect("the sequence has an output type, so one of its tasks is typed");
        Self::Boundary {
            existing: Box::new(TaskEndpoint::new(
                last,
                &[SegmentRef::Index(index)],
                Side::Output,
                output,
            )),
            new: Box::new(TaskEndpoint::new(task, &[], Side::Input, input)),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::pass::StaticPass;
    use crate::{IR, PassContext, Task, TaskKind, TaskPathSegment, TaskTypeError};
    use anyhow::anyhow;
    use qiskit_util::{dyn_types::DynTypeId, static_dyn_typed};

    /// The paths and types of the two ends of a [`TaskTypeError::Sequence`].
    fn sequence_ends(err: &TaskTypeError) -> [(&[TaskPathSegment], &str); 2] {
        let TaskTypeError::Sequence { ends } = err;
        ends.each_ref().map(|end| (end.task().path(), end.ty()))
    }

    /// A [`TaskPathSegment::Index`], to keep the path literals in the tests short.
    fn index(index: usize) -> TaskPathSegment {
        TaskPathSegment::Index(index)
    }

    /// A [`TaskPathSegment::Stage`], to keep the path literals in the tests short.
    fn stage(name: &str) -> TaskPathSegment {
        TaskPathSegment::Stage(name.to_owned())
    }

    #[derive(Clone, Debug)]
    struct MyUint(u32);
    static_dyn_typed!(MyUint);
    impl IR for MyUint {}

    #[derive(Debug)]
    struct MyInt(#[allow(dead_code)] i32);
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

    struct LowerToInt;
    impl StaticPass<MyUint, MyInt> for LowerToInt {
        fn run(&self, ir: Box<MyUint>, _context: &mut PassContext) -> anyhow::Result<Box<MyInt>> {
            let ir = MyInt(ir.0.try_into().map_err(|_| anyhow!("too big!"))?);
            Ok(Box::new(ir))
        }
    }

    #[test]
    fn test_io_types() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        assert_eq!(
            Task::Transformation(AddOne.into_pass()).io_types().unwrap(),
            [uint_ty, uint_ty]
        );
        assert_eq!(
            Task::Transformation(LowerToInt.into_pass())
                .io_types()
                .unwrap(),
            [uint_ty, int_ty]
        );

        let seq = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
        ]);
        assert_eq!(seq.io_types().unwrap(), [uint_ty, int_ty]);

        let nested = Task::Stages(vec![
            ("add".to_string(), Task::Transformation(AddOne.into_pass())),
            ("lower".to_string(), Task::Sequence(vec![seq])),
        ]);
        assert_eq!(nested.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_io_types_empty() {
        assert!(Task::Sequence(vec![]).io_types().is_none());
        assert!(Task::Stages(vec![]).io_types().is_none());

        let nested_empty = Task::Sequence(vec![Task::Stages(vec![(
            "empty".to_string(),
            Task::Sequence(vec![]),
        )])]);
        assert!(nested_empty.io_types().is_none());
    }

    #[test]
    fn test_io_types_skips_empty_children() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let int_ty = DynTypeId::of::<MyInt>();

        let empty = || Task::Sequence(vec![]);
        let lower = || Task::Transformation(LowerToInt.into_pass());

        let leading = Task::Sequence(vec![empty(), lower()]);
        assert_eq!(leading.io_types().unwrap(), [uint_ty, int_ty]);

        let trailing = Task::Sequence(vec![lower(), empty()]);
        assert_eq!(trailing.io_types().unwrap(), [uint_ty, int_ty]);

        let surrounded = Task::Stages(vec![
            ("before".to_string(), empty()),
            ("lower".to_string(), lower()),
            ("after".to_string(), empty()),
        ]);
        assert_eq!(surrounded.io_types().unwrap(), [uint_ty, int_ty]);
    }

    #[test]
    fn test_io_types_ignores_internal_mismatch() {
        let uint_ty = DynTypeId::of::<MyUint>();
        let seq = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        assert_eq!(seq.io_types().unwrap(), [uint_ty, uint_ty]);
        assert!(seq.validate().is_err());
    }

    #[test]
    fn test_validate_accepts_valid_chain() {
        assert!(Task::Sequence(vec![]).validate().is_ok());
        assert!(Task::Transformation(AddOne.into_pass()).validate().is_ok());
        assert!(
            Task::Sequence(vec![
                Task::Transformation(AddOne.into_pass()),
                Task::Transformation(LowerToInt.into_pass())
            ])
            .validate()
            .is_ok()
        );
        assert!(
            Task::Stages(vec![
                ("add".to_string(), Task::Transformation(AddOne.into_pass())),
                (
                    "lower".to_string(),
                    Task::Transformation(LowerToInt.into_pass())
                )
            ])
            .validate()
            .is_ok()
        );
    }

    #[test]
    fn test_validate_reports_both_ends() {
        let seq = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let err = seq.validate().unwrap_err();
        assert_eq!(
            sequence_ends(&err),
            [
                ([index(1)].as_slice(), "MyInt"),
                ([index(2)].as_slice(), "MyUint")
            ]
        );
        assert!(
            err.to_string().contains("the pass '")
                && err.to_string().contains("LowerToInt' at [1] outputs MyInt")
                && err.to_string().contains("AddOne' at [2] takes MyUint"),
            "{err}"
        );
    }

    #[test]
    fn test_validate_reports_stage_names() {
        let stages = Task::Stages(vec![
            (
                "lower".to_string(),
                Task::Transformation(LowerToInt.into_pass()),
            ),
            ("add".to_string(), Task::Transformation(AddOne.into_pass())),
        ]);
        let err = stages.validate().unwrap_err();
        assert_eq!(
            sequence_ends(&err),
            [
                ([stage("lower")].as_slice(), "MyInt"),
                ([stage("add")].as_slice(), "MyUint")
            ]
        );
        assert!(err.to_string().contains("at ['lower'] outputs"), "{err}");
        assert!(err.to_string().contains("at ['add'] takes"), "{err}");
    }

    #[test]
    fn test_validate_reports_nested_paths() {
        let inner = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let outer = Task::Sequence(vec![Task::Stages(vec![("layout".to_string(), inner)])]);
        let err = outer.validate().unwrap_err();
        assert_eq!(
            sequence_ends(&err),
            [
                ([index(0), stage("layout"), index(1)].as_slice(), "MyInt"),
                ([index(0), stage("layout"), index(2)].as_slice(), "MyUint"),
            ]
        );
        assert!(err.to_string().contains("at [0]['layout'][1]"), "{err}");
        assert!(err.to_string().contains("at [0]['layout'][2]"), "{err}");
    }

    #[test]
    fn test_validate_descends_to_originating_pass() {
        // Each end of the mismatch is a nested list, so the error must descend through it to the
        // pass that actually produced the type.
        let seq = Task::Sequence(vec![
            Task::Sequence(vec![
                Task::Transformation(AddOne.into_pass()),
                Task::Transformation(LowerToInt.into_pass()),
            ]),
            Task::Stages(vec![(
                "add".to_string(),
                Task::Transformation(AddOne.into_pass()),
            )]),
        ]);
        let err = seq.validate().unwrap_err();
        assert_eq!(
            sequence_ends(&err),
            [
                ([index(0), index(1)].as_slice(), "MyInt"),
                ([index(1), stage("add")].as_slice(), "MyUint"),
            ]
        );
    }

    #[test]
    fn test_validate_names_the_originating_pass() {
        let seq = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let err = seq.validate().unwrap_err();
        let TaskTypeError::Sequence { ends } = &err;
        let [output, input] = &**ends;
        let TaskKind::Transformation(output_name) = output.task().kind() else {
            panic!("expected a pass");
        };
        let TaskKind::Transformation(input_name) = input.task().kind() else {
            panic!("expected a pass");
        };
        assert!(output_name.contains("LowerToInt"), "{output_name}");
        assert!(input_name.contains("AddOne"), "{input_name}");
    }

    #[test]
    fn test_validate_rejects_across_empty() {
        let seq = Task::Sequence(vec![
            Task::Transformation(LowerToInt.into_pass()),
            Task::Sequence(vec![]),
            Task::Transformation(AddOne.into_pass()),
        ]);
        let err = seq.validate().unwrap_err();
        assert_eq!(
            sequence_ends(&err),
            [
                ([index(0)].as_slice(), "MyInt"),
                ([index(2)].as_slice(), "MyUint")
            ]
        );
    }

    #[test]
    fn test_validate_accepts_across_empty() {
        let seq = Task::Sequence(vec![
            Task::Transformation(AddOne.into_pass()),
            Task::Sequence(vec![]),
            Task::Transformation(LowerToInt.into_pass()),
        ]);
        assert!(seq.validate().is_ok());
    }
}
