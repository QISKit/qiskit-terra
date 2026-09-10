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

//! Running a program a step at a time.

use std::fmt;

use hashbrown::HashSet;
use thiserror::Error;

use super::program_function::{
    InstructionId, InstructionRef, InstructionRole, InstructionView, ProgramFunction, Value,
};
use super::quantum_program::{FunctionId, QuantumProgram, arrange};
use crate::data_tree::DataTree;
use crate::ops::BoxedOpError;
use crate::tensor::{Tensor, TensorType};

/// The identity of one [`Request`] within a particular [`ProgramStepper`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(u32);

impl RequestId {
    /// Return the id of the request at `index` in the order the requests were raised.
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a request id fits in a u32"))
    }

    /// Return the underlying index.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for RequestId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Why an [`ProgramStepper`] could not be created, advanced, or fulfilled.
#[derive(Debug, Error)]
pub enum StepperError {
    /// The inputs are arranged differently than the program declares.
    #[error("inputs are structured {actual} but the program declares {expected}")]
    InputStructureMismatch {
        expected: Box<DataTree<()>>,
        actual: Box<DataTree<()>>,
    },

    /// The argument count does not match the number of declared parameters.
    #[error("expected {expected} argument(s), got {actual}")]
    ArgumentArity { expected: usize, actual: usize },

    /// An argument does not satisfy the type this (monomorphic) function declares for that
    /// parameter.
    #[error("argument {parameter}: expected {expected}, got {actual}")]
    ArgumentTypeMismatch {
        parameter: usize,
        expected: TensorType,
        actual: TensorType,
    },

    /// A function declared external is not one the program holds.
    #[error("{function} is not a function of this program")]
    UnknownExternalFunction { function: FunctionId },

    /// The entry point was declared external. Nothing would ever enter the program.
    #[error("{function} is the entry point, which cannot be external")]
    ExternalEntryPoint { function: FunctionId },

    /// A function Qiskit is to evaluate holds an instruction it has no in-process implementation
    /// of.
    #[error("{function} instruction {instruction} ({full_name}) has no built-in implementation")]
    NoBuiltinEval {
        function: FunctionId,
        instruction: InstructionId,
        full_name: String,
    },

    /// An instruction returned an error from its [`ProgramOp::eval`](crate::ops::ProgramOp::eval).
    #[error("evaluating {function} instruction {instruction} ({full_name})")]
    InstructionFailed {
        function: FunctionId,
        instruction: InstructionId,
        full_name: String,
        #[source]
        source: BoxedOpError,
    },

    /// An instruction's `eval` returned a different number of tensors than its type inference
    /// promised when it was added. This is a bug in the op; it cannot be caught statically.
    #[error(
        "{function} instruction {instruction} returned {actual} result(s), expected {expected}"
    )]
    ResultArityMismatch {
        function: FunctionId,
        instruction: InstructionId,
        expected: usize,
        actual: usize,
    },

    /// No request with this id is outstanding, either because none was raised or because it has
    /// already been fulfilled.
    #[error("request {id} is not outstanding")]
    UnknownRequest { id: RequestId },

    /// A request was fulfilled with a different number of tensors than the call declares.
    #[error("request {id} takes {expected} output(s), got {actual}")]
    RequestOutputCount {
        id: RequestId,
        expected: usize,
        actual: usize,
    },

    /// A tensor fulfilling a request does not satisfy the type the call declares for that output.
    #[error("request {id} output {output}: expected {expected}, got {actual}")]
    RequestOutputType {
        id: RequestId,
        output: usize,
        expected: TensorType,
        actual: TensorType,
    },
}

/// A request issued by a [`ProgramStepper`] to evaluate a function.
#[derive(Debug, Clone, Copy)]
pub struct Request<'e> {
    id: RequestId,
    function: FunctionId,
    inputs: &'e [Tensor],
    output_types: &'e [TensorType],
}

impl<'e> Request<'e> {
    pub fn id(&self) -> RequestId {
        self.id
    }

    pub fn function(&self) -> FunctionId {
        self.function
    }

    pub fn inputs(&self) -> &'e [Tensor] {
        self.inputs
    }

    pub fn output_types(&self) -> &'e [TensorType] {
        self.output_types
    }
}

/// An outstanding request marker.
#[derive(Debug, Clone)]
struct Outstanding {
    id: RequestId,
    /// Which frame the request came from.
    frame: usize,
    /// Which call instruction the request represents.
    instruction: InstructionId,
    function: FunctionId,
    inputs: Vec<Tensor>,
}

/// The in-progress evaluation of a single function.
///
/// As function evaluation proceeds, intermediate `values` are populated with the
/// outputs of instructions. Values are cleared when all of their `uses` have been consumed.
#[derive(Debug, Clone)]
struct Frame {
    function: FunctionId,
    /// Every value the function produces, addressed by the offsets of [`value_offsets`]. A
    /// value the function returns is held until the frame finishes.
    values: Vec<Option<Tensor>>,
    /// How many instructions are still to read each value. A value is released at zero.
    uses: Vec<u32>,
    /// The instructions still to run, in storage order, discluding parameter instructions.
    remaining: Vec<InstructionId>,
    /// The frame and the call instruction to return into, `None` for the entry point.
    returns_to: Option<(usize, InstructionId)>,
}

impl Frame {
    /// Write the results of `instruction` into `values`.
    fn store(&mut self, offsets: &[u32], instruction: InstructionId, results: Vec<Tensor>) {
        let base = offsets[instruction.index()] as usize;
        for (slot, tensor) in results.into_iter().enumerate() {
            if self.uses[base + slot] > 0 {
                self.values[base + slot] = Some(tensor);
            }
        }
    }

    /// Record that one instruction has read `operands`, releasing fully used values.
    fn release(&mut self, offsets: &[u32], operands: &[Value]) {
        for &value in operands {
            let index = flat(offsets, value);
            self.uses[index] -= 1;
            if self.uses[index] == 0 {
                self.values[index] = None;
            }
        }
    }

    /// Return whether every operand of `instruction` is available.
    fn ready(&self, offsets: &[u32], instruction: InstructionRef<'_>) -> bool {
        instruction
            .operands()
            .iter()
            .all(|&value| self.values[flat(offsets, value)].is_some())
    }
}

/// Evaluation stepper for a [`QuantumProgram`], requesting an execution for each external
/// function.
///
/// This object is a generic helper tool to evaluate a program. A user of this class declares which
/// functions in the program should be evaluated externally versus internally. Internal function
/// evaluation is possible when all of its instructions have a built-in
/// [`ProgramOp::eval`](crate::ops::ProgramOp::eval) implementation. Even if a function is able to
/// be evaluated internally, it can still be tagged for external evaluation. The entry function
/// cannot be tagged as external. Taking a step causes all unblocked internal evaluation to
/// complete. When a call to a function tagged for external evaluation is encountered, an evaluation
/// request is recorded, so one step surfaces every request whose operands have arrived. The user
/// must eventually fulfil a request before downstream work can proceed via stepping.
///
/// # Example
/// ```rust
/// use qiskit_providers::ops::Add;
/// use qiskit_providers::tensor::{DType, Dim, Tensor, TensorType};
/// use qiskit_providers::{DataTree, ProgramStepper, FunctionId, ProgramFunction, QuantumProgram};
///
/// let ty = TensorType { dtype: DType::F64, shape: vec![Dim::Fixed(1)] };
///
/// // @0 doubles its argument, and the entry point @1 calls it.
/// let mut doubling = ProgramFunction::new();
/// let x = doubling.add_parameter(ty.clone());
/// let doubled = doubling.add_op(Add, &[x, x])?[0];
/// doubling.add_result(doubled)?;
/// let signature = doubling.signature();
///
/// let mut entry = ProgramFunction::new();
/// let x = entry.add_parameter(ty.clone());
/// let called = entry.add_call(FunctionId::from_index(0), &signature, &[x])?[0];
/// entry.add_result(called)?;
///
/// let program =
///     QuantumProgram::new(vec![doubling, entry], DataTree::Leaf(()), DataTree::Leaf(()))?;
///
/// // @0 is declared external, so it is handed over rather than run, even though `Add` has a
/// // built-in evaluation.
/// let inputs = DataTree::Leaf(Tensor::from([1.5_f64]));
/// let mut stepper = ProgramStepper::new(&program, inputs, [FunctionId::from_index(0)])?;
/// let outputs = loop {
///     stepper.step(&program)?;
///     if let Some(outputs) = stepper.outputs(&program) {
///         break outputs;
///     }
///     // Each request holds the arguments the call was reached with. Standing in for @0 means
///     // doubling them.
///     let answers: Vec<_> = stepper
///         .outstanding(&program)
///         .map(|request| {
///             let argument = &request.inputs()[0];
///             (request.id(), argument + argument)
///         })
///         .collect();
///     for (id, answer) in answers {
///         stepper.fulfil(&program, id, vec![answer])?;
///     }
/// };
/// assert_eq!(outputs, DataTree::Leaf(Tensor::from([3.0_f64])));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct ProgramStepper {
    /// One frame per function being evaluated. Finished frames become `None`.
    frames: Vec<Option<Frame>>,
    /// Which functions require external evaluation. All other functions attempt built-in evaluation.
    external: HashSet<FunctionId>,
    /// Outstanding external evaluation requests.
    outstanding: Vec<Outstanding>,
    next_id: u32,
    /// The entry point's results, once it has finished.
    outputs: Option<Vec<Tensor>>,
}

impl ProgramStepper {
    /// Begin evaluating `program` on `inputs`, declaring all `external` functions should become [`Request`]s.
    pub fn new(
        program: &QuantumProgram,
        inputs: DataTree<Tensor>,
        external: impl IntoIterator<Item = FunctionId>,
    ) -> Result<Self, StepperError> {
        let mut declared = HashSet::new();
        for function in external {
            if program.function(function).is_none() {
                return Err(StepperError::UnknownExternalFunction { function });
            }
            if function == program.entry() {
                return Err(StepperError::ExternalEntryPoint { function });
            }
            declared.insert(function);
        }

        let actual = inputs.structure();
        if actual != *program.input_structure() {
            return Err(StepperError::InputStructureMismatch {
                expected: Box::new(program.input_structure().clone()),
                actual: Box::new(actual),
            });
        }

        // Check that all internal functions can be evaluated.
        if let Some((function, instruction)) =
            first_without_builtin_eval(program.functions(), &declared)
        {
            return Err(StepperError::NoBuiltinEval {
                function,
                instruction: instruction.id(),
                full_name: instruction.full_name(),
            });
        }

        // Check input type correctness.
        let arguments: Vec<Tensor> = inputs.into_leaves().collect();
        check_arguments(program.entry_function(), &arguments)?;

        let mut stepper = Self {
            frames: Vec::new(),
            external: declared,
            outstanding: Vec::new(),
            next_id: 0,
            outputs: None,
        };
        stepper.push_frame(program.functions(), program.entry(), arguments, None);
        Ok(stepper)
    }

    /// Advance evaluation until blocked by external work.
    ///
    /// One step finds every call whose operands have become available, so a caller can dispatch
    /// independent work together.
    pub fn step(&mut self, program: &QuantumProgram) -> Result<(), StepperError> {
        self.advance(program.functions())?;
        // Every value a live frame waits on is produced either by an instruction that will run or
        // by a request, so a step that did not finish the program has always left work to answer.
        // A stepper therefore cannot deadlock, and a caller's loop cannot spin.
        debug_assert!(
            self.outputs.is_some() || !self.outstanding.is_empty(),
            "a step left the program unfinished with nothing to answer"
        );
        Ok(())
    }

    /// Return every request that has not been fufilled.
    pub fn outstanding<'e>(
        &'e self,
        program: &'e QuantumProgram,
    ) -> impl Iterator<Item = Request<'e>> {
        self.outstanding.iter().map(move |request| Request {
            id: request.id,
            function: request.function,
            inputs: &request.inputs,
            output_types: self.declared_outputs(program, request.frame, request.instruction),
        })
    }

    /// Fufil the request `id` with one tensor per result the call requires.
    ///
    /// # Errors
    ///
    /// Returns an error if `id` is not outstanding, or if `outputs` is the wrong length
    /// or holds a tensor of the wrong type. The request stays outstanding.
    pub fn fulfil(
        &mut self,
        program: &QuantumProgram,
        id: RequestId,
        outputs: Vec<Tensor>,
    ) -> Result<(), StepperError> {
        let Some(position) = self.outstanding.iter().position(|request| request.id == id) else {
            return Err(StepperError::UnknownRequest { id });
        };
        let (frame, instruction) = {
            let request = &self.outstanding[position];
            (request.frame, request.instruction)
        };

        let expected = self.declared_outputs(program, frame, instruction);
        if outputs.len() != expected.len() {
            return Err(StepperError::RequestOutputCount {
                id,
                expected: expected.len(),
                actual: outputs.len(),
            });
        }
        for (output, (tensor, ty)) in outputs.iter().zip(expected).enumerate() {
            if !tensor.matches(ty) {
                return Err(StepperError::RequestOutputType {
                    id,
                    output,
                    expected: ty.clone(),
                    actual: tensor.tensor_type(),
                });
            }
        }

        self.outstanding.remove(position);
        let frame = self.frames[frame]
            .as_mut()
            .expect("an outstanding request has a live frame");
        let body = &program.functions()[frame.function.index()];
        frame.store(&value_offsets(body), instruction, outputs);
        Ok(())
    }

    /// Return the program's outputs, arranged in its output structure, and `None` until it has
    /// finished.
    pub fn outputs(&self, program: &QuantumProgram) -> Option<DataTree<Tensor>> {
        let outputs = self.outputs.clone()?;
        Some(arrange(program.output_structure(), outputs))
    }

    /// Run each live frame until a whole pass makes no progress.
    fn advance(&mut self, functions: &[ProgramFunction]) -> Result<(), StepperError> {
        loop {
            let mut progressed = false;
            for index in 0..self.frames.len() {
                progressed |= self.run_frame(functions, index)?;
            }
            if !progressed {
                return Ok(());
            }
        }
    }

    /// Run every instruction of the frame at `index` whose operands have arrived, reporting whether
    /// anything happened.
    fn run_frame(
        &mut self,
        functions: &[ProgramFunction],
        index: usize,
    ) -> Result<bool, StepperError> {
        let Some(live) = &self.frames[index] else {
            return Ok(false);
        };
        let id = live.function;
        let body = &functions[id.index()];
        let offsets = value_offsets(body);

        // A child frame mutates the frames the walk is holding, so it is recorded here and pushed
        // once that borrow has ended.
        let mut calls: Vec<(InstructionId, FunctionId, Vec<Tensor>)> = Vec::new();
        let mut progressed = false;
        let mut failure = None;

        let Self {
            frames,
            external,
            outstanding,
            next_id,
            ..
        } = self;
        let frame = frames[index].as_mut().expect("the frame is live");
        let remaining = std::mem::take(&mut frame.remaining);
        let mut blocked = Vec::with_capacity(remaining.len());
        let mut walked = 0;
        while let Some(&instruction_id) = remaining.get(walked) {
            walked += 1;
            let instruction = body
                .instruction(instruction_id)
                .expect("a frame runs the instructions of its own function");
            if !frame.ready(&offsets, instruction) {
                blocked.push(instruction_id);
                continue;
            }
            let operands = instruction.operands();
            let gathered: Vec<Tensor> = operands
                .iter()
                .map(|&value| {
                    frame.values[flat(&offsets, value)]
                        .clone()
                        .expect("a ready instruction has every operand")
                })
                .collect();

            match instruction.view() {
                InstructionView::Parameter => {
                    unreachable!("a parameter is filled when its frame is pushed")
                }
                InstructionView::Op(op) => match op.eval(&gathered) {
                    Err(source) => {
                        failure = Some(StepperError::InstructionFailed {
                            function: id,
                            instruction: instruction_id,
                            full_name: op.full_name(),
                            source,
                        });
                        blocked.push(instruction_id);
                        break;
                    }
                    Ok(results) if results.len() != instruction.output_types().len() => {
                        failure = Some(StepperError::ResultArityMismatch {
                            function: id,
                            instruction: instruction_id,
                            expected: instruction.output_types().len(),
                            actual: results.len(),
                        });
                        blocked.push(instruction_id);
                        break;
                    }
                    Ok(results) => frame.store(&offsets, instruction_id, results),
                },
                InstructionView::Call(callee) if external.contains(&callee) => {
                    outstanding.push(Outstanding {
                        id: RequestId(*next_id),
                        frame: index,
                        instruction: instruction_id,
                        function: callee,
                        inputs: gathered,
                    });
                    *next_id += 1;
                }
                InstructionView::Call(callee) => {
                    calls.push((instruction_id, callee, gathered));
                }
                // We want to end the function with the results in `values`, so do nothing.
                InstructionView::Result => {}
            }
            // A result is not a consumer. Releasing its operand would drop the very value the
            // frame returns, so a returned value's use count never reaches zero.
            if instruction.role() != InstructionRole::Result {
                frame.release(&offsets, operands);
            }
            progressed = true;
        }
        blocked.extend_from_slice(&remaining[walked..]);
        frame.remaining = blocked;

        if let Some(failure) = failure {
            return Err(failure);
        }
        for (instruction, callee, arguments) in calls {
            self.push_frame(functions, callee, arguments, Some((index, instruction)));
        }

        let done = self.frames[index]
            .as_ref()
            .is_some_and(|frame| frame.remaining.is_empty());
        if done && !self.awaiting(index) {
            self.finish_frame(functions, index);
            progressed = true;
        }
        Ok(progressed)
    }

    /// Push a frame of `function` with `arguments` in its parameters' slots, and return its index.
    fn push_frame(
        &mut self,
        functions: &[ProgramFunction],
        function: FunctionId,
        arguments: Vec<Tensor>,
        returns_to: Option<(usize, InstructionId)>,
    ) -> usize {
        let body = &functions[function.index()];
        let offsets = value_offsets(body);
        let total = *offsets.last().expect("offsets always end with the total") as usize;

        let mut uses = vec![0_u32; total];
        for instruction in body.iter_instructions() {
            for &value in instruction.operands() {
                uses[flat(&offsets, value)] += 1;
            }
        }

        // `zip` below would truncate rather than complain, leaving a parameter empty and the frame
        // unable ever to finish, so the invariant it rests on is stated here.
        debug_assert_eq!(
            arguments.len(),
            body.parameters().len(),
            "a frame takes one argument per parameter of the function it runs"
        );
        let mut values = vec![None; total];
        for (&parameter, argument) in body.parameters().iter().zip(arguments) {
            let index = offsets[parameter.index()] as usize;
            // only store if it's going to be used
            if uses[index] > 0 {
                values[index] = Some(argument);
            }
        }

        self.frames.push(Some(Frame {
            function,
            values,
            uses,
            remaining: body
                .iter_instructions()
                .filter(|instruction| instruction.role() != InstructionRole::Parameter)
                .map(|instruction| instruction.id())
                .collect(),
            returns_to,
        }));
        self.frames.len() - 1
    }

    /// Return the results of the finished frame at `index` to its caller, and release the frame.
    fn finish_frame(&mut self, functions: &[ProgramFunction], index: usize) {
        let frame = self.frames[index].take().expect("the frame is live");
        let body = &functions[frame.function.index()];
        let offsets = value_offsets(body);
        let results: Vec<Tensor> = body
            .result_values()
            .map(|value| {
                frame.values[flat(&offsets, value)]
                    .clone()
                    .expect("a finished frame still holds every value it returns")
            })
            .collect();
        let Some((parent, instruction)) = frame.returns_to else {
            self.outputs = Some(results);
            return;
        };
        let caller = self.frames[parent]
            .as_mut()
            .expect("a frame outlives the frames it pushed");
        let body = &functions[caller.function.index()];
        caller.store(&value_offsets(body), instruction, results);
    }

    /// Return whether a child frame or an outstanding request has still to return into the frame at
    /// `index`.
    fn awaiting(&self, index: usize) -> bool {
        self.outstanding
            .iter()
            .any(|request| request.frame == index)
            || self
                .frames
                .iter()
                .flatten()
                .any(|frame| frame.returns_to.is_some_and(|(parent, _)| parent == index))
    }

    /// Return the types the call at `instruction` of the frame at `index` declares for its results.
    fn declared_outputs<'p>(
        &self,
        program: &'p QuantumProgram,
        index: usize,
        instruction: InstructionId,
    ) -> &'p [TensorType] {
        let frame = self.frames[index]
            .as_ref()
            .expect("an outstanding request has a live frame");
        program
            .function(frame.function)
            .expect("a frame names a function of the program being evaluated")
            .instruction(instruction)
            .expect("a frame's requests come from its own instructions")
            .output_types()
    }
}

/// Evaluate a `function` against `args`.
pub(super) fn eval_function(
    function: &ProgramFunction,
    args: &[Tensor],
) -> Result<Vec<Tensor>, StepperError> {
    // The function is monomorphic, so its declared parameter types are the only ones its
    // instructions were built for. Checking them here names the argument the caller supplied.
    check_arguments(function, args)?;
    let entry = FunctionId::from_index(0);
    if let Some(instruction) = function
        .iter_instructions()
        .find(|instruction| !instruction.has_builtin_eval())
    {
        return Err(StepperError::NoBuiltinEval {
            function: entry,
            instruction: instruction.id(),
            full_name: instruction.full_name(),
        });
    }

    let functions = std::slice::from_ref(function);
    let mut stepper = ProgramStepper {
        frames: Vec::new(),
        external: HashSet::new(),
        outstanding: Vec::new(),
        next_id: 0,
        outputs: None,
    };
    stepper.push_frame(functions, entry, args.to_vec(), None);
    stepper.advance(functions)?;
    Ok(stepper
        .outputs
        .expect("a function with nothing external cannot block"))
}

/// Return the first instruction of any function outside `external` that has no built-in evaluation.
pub(super) fn first_without_builtin_eval<'p>(
    functions: &'p [ProgramFunction],
    external: &HashSet<FunctionId>,
) -> Option<(FunctionId, InstructionRef<'p>)> {
    functions.iter().enumerate().find_map(|(index, function)| {
        let id = FunctionId::from_index(index);
        if external.contains(&id) {
            return None;
        }
        function
            .iter_instructions()
            .find(|instruction| {
                instruction.role() != InstructionRole::Call && !instruction.has_builtin_eval()
            })
            .map(|instruction| (id, instruction))
    })
}

/// Validate that every argument satisfies the type `function` declares for its parameter.
fn check_arguments(function: &ProgramFunction, args: &[Tensor]) -> Result<(), StepperError> {
    if args.len() != function.parameters().len() {
        return Err(StepperError::ArgumentArity {
            expected: function.parameters().len(),
            actual: args.len(),
        });
    }
    for (parameter, (arg, value)) in args.iter().zip(function.parameter_values()).enumerate() {
        let expected = function.type_of(value).expect("a parameter always exists");
        if !arg.matches(expected) {
            return Err(StepperError::ArgumentTypeMismatch {
                parameter,
                expected: expected.clone(),
                actual: arg.tensor_type(),
            });
        }
    }
    Ok(())
}

/// Return where each instruction's block of values starts in a dense environment, with the total
/// appended.
///
/// An instruction's values are already grouped by their producer, so this is one prefix sum away
/// and is derived where it is needed rather than stored.
fn value_offsets(function: &ProgramFunction) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(function.instruction_count() + 1);
    let mut total = 0;
    for instruction in function.iter_instructions() {
        offsets.push(total);
        total += instruction.output_types().len() as u32;
    }
    offsets.push(total);
    offsets
}

/// Return where `value` sits in an environment laid out by `offsets`.
fn flat(offsets: &[u32], value: Value) -> usize {
    offsets[value.instruction().index()] as usize + value.slot()
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::super::{Signature, partition};
    use super::*;
    use crate::ops::{Add, ProgramOp};
    use crate::tensor::{DType, Dim};

    /// Return the type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// Return a one-element `F64` tensor.
    fn one(value: f64) -> Tensor {
        Tensor::from([value])
    }

    /// Return a bare leaf, which describes one slot.
    fn leaf() -> DataTree<()> {
        DataTree::Leaf(())
    }

    /// Build `f(x) = x + x` over `F64[len]` tensors: one parameter, one result.
    fn double_function(len: usize) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(len));
        let doubled = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(doubled).unwrap();
        function
    }

    /// Build a program whose `@0` doubles, and whose entry point `@1` calls it and returns what
    /// comes back.
    fn calling_program() -> QuantumProgram {
        let callee = double_function(1);
        let signature = callee.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap()[0];
        entry.add_result(called).unwrap();
        QuantumProgram::new(vec![callee, entry], leaf(), leaf()).unwrap()
    }

    /// Return `@0`, the one function [`calling_program`]'s entry point calls.
    fn callee() -> FunctionId {
        FunctionId::from_index(0)
    }

    /// Return a stepper over `program` and `input`, with `external` handed over.
    fn start(
        program: &QuantumProgram,
        input: f64,
        external: impl IntoIterator<Item = FunctionId>,
    ) -> ProgramStepper {
        ProgramStepper::new(program, DataTree::Leaf(one(input)), external).unwrap()
    }

    // ---------------------------------------------------------------------------
    // Driving a stepper
    // ---------------------------------------------------------------------------

    #[test]
    fn a_local_program_finishes_in_one_step() {
        let program = QuantumProgram::new(vec![double_function(1)], leaf(), leaf()).unwrap();
        let mut stepper = start(&program, 1.5, []);

        assert!(stepper.outputs(&program).is_none(), "nothing has run yet");
        stepper.step(&program).unwrap();
        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::Leaf(one(3.0))),
            "the outputs arrive in the structure the program declares"
        );
    }

    #[test]
    fn an_external_call_is_handed_over_and_fulfilled() {
        let program = calling_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();

        let requests: Vec<Request<'_>> = stepper.outstanding(&program).collect();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].id(), RequestId::from_index(0));
        assert_eq!(requests[0].function(), callee());
        assert_eq!(requests[0].inputs(), &[one(1.5)]);
        assert_eq!(requests[0].output_types(), &[f64_1d(1)]);
        let id = requests[0].id();
        assert!(
            stepper.outputs(&program).is_none(),
            "the program is unfinished while a request is outstanding"
        );

        stepper.fulfil(&program, id, vec![one(10.0)]).unwrap();
        assert_eq!(
            stepper.outstanding(&program).count(),
            0,
            "a fulfilled request is no longer outstanding"
        );
        assert!(
            stepper.outputs(&program).is_none(),
            "fulfilling evaluates nothing of its own"
        );

        stepper.step(&program).unwrap();
        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::Leaf(one(10.0))),
            "the answer is what the call produced, not what @0 would have"
        );
    }

    #[test]
    fn one_step_surfaces_every_ready_call() {
        // The property the stepper exists for: two calls that do not depend on each other are
        // dispatched together rather than one step apart.
        let program = two_external_program();

        let inputs = DataTree::sequence([
            DataTree::Leaf(one(1.5)),
            DataTree::Leaf(Tensor::from([1.0_f64, 2.0])),
        ]);
        let external = [FunctionId::from_index(0), FunctionId::from_index(1)];
        let mut stepper = ProgramStepper::new(&program, inputs, external).unwrap();
        stepper.step(&program).unwrap();

        let requests: Vec<Request<'_>> = stepper.outstanding(&program).collect();
        assert_eq!(requests.len(), 2, "one step surfaced both calls");
        assert_eq!(
            requests
                .iter()
                .map(|request| (request.id(), request.function()))
                .collect::<Vec<_>>(),
            vec![
                (RequestId::from_index(0), FunctionId::from_index(0)),
                (RequestId::from_index(1), FunctionId::from_index(1)),
            ],
            "ids are issued in the order the calls were reached, and each names its own function"
        );
        assert_eq!(
            requests
                .iter()
                .map(|request| request.output_types())
                .collect::<Vec<_>>(),
            vec![&[f64_1d(1)][..], &[f64_1d(2)][..]],
            "each request declares the types its own call produces"
        );

        // Answering them in the reverse of the order they were surfaced makes no difference.
        stepper
            .fulfil(
                &program,
                RequestId::from_index(1),
                vec![Tensor::from([7.0_f64, 8.0])],
            )
            .unwrap();
        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(9.0)])
            .unwrap();
        stepper.step(&program).unwrap();
        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::sequence([
                DataTree::Leaf(one(9.0)),
                DataTree::Leaf(Tensor::from([7.0_f64, 8.0])),
            ]))
        );
    }

    #[test]
    fn a_call_inside_a_call_is_reached_by_one_step() {
        // The frame @1 runs in is pushed by the same step that then reaches the request inside it.
        let program = nested_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();

        let requests: Vec<Request<'_>> = stepper.outstanding(&program).collect();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].function(), callee());
        assert_eq!(requests[0].inputs(), &[one(1.5)]);

        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(10.0)])
            .unwrap();
        stepper.step(&program).unwrap();
        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::Leaf(one(10.0))),
            "the answer travelled back out through both frames"
        );
    }

    /// Build a program whose `@0` doubles an `F64[1]` and whose `@1` doubles an `F64[2]`, and whose
    /// entry point `@2` calls both on two parameters of its own, so one step raises two requests of
    /// different types.
    fn two_external_program() -> QuantumProgram {
        let (narrow, wide) = (double_function(1), double_function(2));
        let (narrow_signature, wide_signature) = (narrow.signature(), wide.signature());
        let mut entry = ProgramFunction::new();
        let a = entry.add_parameter(f64_1d(1));
        let b = entry.add_parameter(f64_1d(2));
        let first = entry
            .add_call(FunctionId::from_index(0), &narrow_signature, &[a])
            .unwrap()[0];
        let second = entry
            .add_call(FunctionId::from_index(1), &wide_signature, &[b])
            .unwrap()[0];
        entry.add_result(first).unwrap();
        entry.add_result(second).unwrap();
        QuantumProgram::new(
            vec![narrow, wide, entry],
            DataTree::sequence([leaf(), leaf()]),
            DataTree::sequence([leaf(), leaf()]),
        )
        .unwrap()
    }

    /// Build a program whose `@0` doubles, whose `@1` calls it, and whose entry point `@2` calls
    /// `@1`, so a request sits inside a frame that another frame pushed.
    fn nested_program() -> QuantumProgram {
        let external = double_function(1);
        let signature = external.signature();
        let middle = calls(FunctionId::from_index(0), &signature);
        let entry = calls(FunctionId::from_index(1), &signature);
        QuantumProgram::new(vec![external, middle, entry], leaf(), leaf()).unwrap()
    }

    /// Return a function that calls `callee` on its one parameter and returns what comes back.
    fn calls(callee: FunctionId, signature: &Signature) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(signature.inputs[0].clone());
        let called = function.add_call(callee, signature, &[x]).unwrap()[0];
        function.add_result(called).unwrap();
        function
    }

    #[test]
    fn work_after_a_blocked_call_runs_in_the_same_step() {
        // A cursor would stop at the call, so the tally standing after it pins that the walk keeps
        // going. The tally's result is declared second and its value arrives first, so this also
        // pins that a result goes to the slot it was declared in whatever order they arrive.
        let runs = Arc::new(AtomicUsize::new(0));
        let external = double_function(1);
        let signature = external.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap()[0];
        let counted = entry.add_op(Tally(Arc::clone(&runs)), &[x]).unwrap()[0];
        entry.add_result(called).unwrap();
        entry.add_result(counted).unwrap();
        let program = QuantumProgram::new(
            vec![external, entry],
            leaf(),
            DataTree::sequence([leaf(), leaf()]),
        )
        .unwrap();

        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "the instruction after the blocked call ran in the same step"
        );
        assert_eq!(stepper.outstanding(&program).count(), 1);

        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(10.0)])
            .unwrap();
        stepper.step(&program).unwrap();
        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::sequence([
                DataTree::Leaf(one(10.0)),
                DataTree::Leaf(one(1.5)),
            ])),
            "each result is in the slot it was declared in, whatever order they landed in"
        );
    }

    #[test]
    fn a_value_read_out_of_order_is_held_until_its_last_reader_has_run() {
        // The add at the end of the function reads `x` and runs first, and the add before it reads
        // `x` and runs a step later. Releasing `x` by position rather than by a count of readers
        // would have dropped it in between, leaving the second add unable to ever run.
        let external = double_function(1);
        let signature = external.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap()[0];
        let blocked = entry.add_op(Add, &[called, x]).unwrap()[0];
        let ready = entry.add_op(Add, &[x, x]).unwrap()[0];
        entry.add_result(blocked).unwrap();
        entry.add_result(ready).unwrap();
        let program = QuantumProgram::new(
            vec![external, entry],
            DataTree::sequence([leaf(), leaf()]),
            DataTree::sequence([leaf(), leaf()]),
        )
        .unwrap();

        // The second parameter is read by nothing, so it is never stored.
        let inputs = DataTree::sequence([DataTree::Leaf(one(1.5)), DataTree::Leaf(one(99.0))]);
        let mut stepper = ProgramStepper::new(&program, inputs, [callee()]).unwrap();
        stepper.step(&program).unwrap();
        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(10.0)])
            .unwrap();
        stepper.step(&program).unwrap();

        assert_eq!(
            stepper.outputs(&program),
            Some(DataTree::sequence([
                DataTree::Leaf(one(11.5)),
                DataTree::Leaf(one(3.0)),
            ]))
        );
    }

    #[test]
    fn a_call_whose_results_nothing_reads_still_has_to_be_answered() {
        // The entry point has nothing left to run once the call is dispatched, so the outstanding
        // request is the only record that the program is unfinished.
        let external = double_function(1);
        let signature = external.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap();
        entry.add_result(x).unwrap();
        let program = QuantumProgram::new(vec![external, entry], leaf(), leaf()).unwrap();

        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        assert_eq!(stepper.outstanding(&program).count(), 1);
        assert!(stepper.outputs(&program).is_none());

        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(10.0)])
            .unwrap();
        stepper.step(&program).unwrap();
        assert_eq!(stepper.outputs(&program), Some(DataTree::Leaf(one(1.5))));
    }

    #[test]
    fn a_partitioned_program_is_evaluated_through_its_units() {
        // What a backend does: cut the work it must perform into functions of its own, hand each
        // one over, and let Qiskit do the rest.
        let mut whole = ProgramFunction::new();
        let x = whole.add_parameter(f64_1d(1));
        let sampled = whole.add_op(Elsewhere { builtin: false }, &[x]).unwrap()[0];
        let doubled = whole.add_op(Add, &[sampled, sampled]).unwrap()[0];
        whole.add_result(doubled).unwrap();
        let whole = QuantumProgram::new(vec![whole], leaf(), leaf()).unwrap();

        let (program, resource) = partition(&whole, [["vendor.elsewhere"]]).unwrap();
        let units: Vec<FunctionId> = resource
            .iter()
            .enumerate()
            .filter(|&(_, &unit)| unit == 0)
            .map(|(index, _)| FunctionId::from_index(index))
            .collect();
        assert_eq!(units.len(), 1);

        let mut stepper = start(&program, 1.5, units);
        let outputs = loop {
            stepper.step(&program).unwrap();
            if let Some(outputs) = stepper.outputs(&program) {
                break outputs;
            }
            let answers: Vec<(RequestId, Tensor)> = stepper
                .outstanding(&program)
                .map(|request| {
                    assert_eq!(request.output_types(), &[f64_1d(1)]);
                    (request.id(), one(4.0))
                })
                .collect();
            for (id, answer) in answers {
                stepper.fulfil(&program, id, vec![answer]).unwrap();
            }
        };
        assert_eq!(outputs, DataTree::Leaf(one(8.0)));
    }

    #[test]
    fn a_clone_continues_independently() {
        let program = calling_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        let mut other = stepper.clone();

        let id = RequestId::from_index(0);
        stepper.fulfil(&program, id, vec![one(10.0)]).unwrap();
        other.fulfil(&program, id, vec![one(20.0)]).unwrap();
        stepper.step(&program).unwrap();
        other.step(&program).unwrap();

        assert_eq!(stepper.outputs(&program), Some(DataTree::Leaf(one(10.0))));
        assert_eq!(other.outputs(&program), Some(DataTree::Leaf(one(20.0))));
    }

    #[test]
    fn a_stepper_can_be_sent_between_threads() {
        // A job drives a stepper and is where all concurrency lives, so it must be able to hold
        // one between steps.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProgramStepper>();
    }

    // ---------------------------------------------------------------------------
    // Suspending and resuming
    // ---------------------------------------------------------------------------

    /// Whatever a backend keeps between processes, keyed by job. A job id and this have to be
    /// enough on their own, since nothing of the process that submitted the job survives.
    type Store = Vec<(u32, QuantumProgram, ProgramStepper)>;

    /// Return every outstanding request, as owned data that outlives the stepper being moved.
    fn described(
        stepper: &ProgramStepper,
        program: &QuantumProgram,
    ) -> Vec<(RequestId, FunctionId, Vec<Tensor>, Vec<TensorType>)> {
        stepper
            .outstanding(program)
            .map(|request| {
                (
                    request.id(),
                    request.function(),
                    request.inputs().to_vec(),
                    request.output_types().to_vec(),
                )
            })
            .collect()
    }

    #[test]
    fn a_job_resumes_from_its_id_and_a_store_alone() {
        // A stepper borrows nothing and holds no callback, so the pair it forms with its program is
        // all a backend has to keep. The request is raised inside a child frame, which is the state
        // most likely to depend on the walk that produced it.
        let program = nested_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        let submitted = described(&stepper, &program);
        assert_eq!(submitted.len(), 1);

        // Submitting moves the pair into the store, and nothing else of this scope survives.
        let mut store: Store = vec![(7, program, stepper)];

        // Resuming is given the store and the id, and nothing else.
        let (_, program, stepper) = store
            .iter_mut()
            .find(|(job, _, _)| *job == 7)
            .expect("the job was stored");
        assert_eq!(
            described(stepper, program),
            submitted,
            "the stored pair reports the same work, so no request had to be kept beside it"
        );
        for (id, ..) in described(stepper, program) {
            stepper.fulfil(program, id, vec![one(10.0)]).unwrap();
        }
        stepper.step(program).unwrap();
        assert_eq!(stepper.outputs(program), Some(DataTree::Leaf(one(10.0))));
    }

    #[test]
    fn a_run_replays_from_its_inputs_and_answers() {
        // A stepper is derived from the program, the inputs and the answers, so resuming needs no
        // format of its own. Ids are minted in the order the walk reaches the calls, which is
        // deterministic, so a recorded answer still finds its request in a fresh stepper.
        let program = two_external_program();
        let inputs = || {
            DataTree::sequence([
                DataTree::Leaf(one(1.5)),
                DataTree::Leaf(Tensor::from([1.0_f64, 2.0])),
            ])
        };
        let external = [FunctionId::from_index(0), FunctionId::from_index(1)];

        // Run it once, recording what was answered and against which id.
        let mut recorded: Vec<(RequestId, Tensor)> = Vec::new();
        let mut first = ProgramStepper::new(&program, inputs(), external).unwrap();
        let expected = loop {
            first.step(&program).unwrap();
            if let Some(outputs) = first.outputs(&program) {
                break outputs;
            }
            let round: Vec<(RequestId, Tensor)> = first
                .outstanding(&program)
                .map(|request| {
                    let argument = &request.inputs()[0];
                    (request.id(), argument + argument)
                })
                .collect();
            for (id, answer) in round {
                first.fulfil(&program, id, vec![answer.clone()]).unwrap();
                recorded.push((id, answer));
            }
        };
        drop(first);

        // Replay from the program, the same inputs, and those answers looked up by id.
        let mut replayed: Vec<RequestId> = Vec::new();
        let mut second = ProgramStepper::new(&program, inputs(), external).unwrap();
        let outputs = loop {
            second.step(&program).unwrap();
            if let Some(outputs) = second.outputs(&program) {
                break outputs;
            }
            let ids: Vec<RequestId> = second.outstanding(&program).map(|r| r.id()).collect();
            for id in ids {
                let answer = recorded
                    .iter()
                    .find(|(against, _)| *against == id)
                    .map(|(_, answer)| answer.clone())
                    .expect("a replayed run raises the requests the recorded one did");
                second.fulfil(&program, id, vec![answer]).unwrap();
                replayed.push(id);
            }
        };

        assert_eq!(
            replayed,
            recorded.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            "the same ids came back in the same order"
        );
        assert_eq!(outputs, expected);
    }

    #[test]
    fn a_suspended_job_finishes_on_another_thread() {
        // Finishing a job somewhere that never saw it built is the closest stand-in for a fresh
        // process that needs no wire format: `spawn` requires `Send` and `'static` of both halves.
        let program = nested_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();

        let outputs = std::thread::spawn(move || {
            let mut stepper = stepper;
            let id = stepper
                .outstanding(&program)
                .next()
                .expect("a request was outstanding when the job was suspended")
                .id();
            stepper.fulfil(&program, id, vec![one(10.0)]).unwrap();
            stepper.step(&program).unwrap();
            stepper.outputs(&program)
        })
        .join()
        .expect("the thread finished the job");

        assert_eq!(outputs, Some(DataTree::Leaf(one(10.0))));
    }

    // ---------------------------------------------------------------------------
    // Rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn declaring_the_wrong_function_external_is_rejected() {
        let program = calling_program();
        let input = || DataTree::Leaf(one(1.5));

        let Err(err) = ProgramStepper::new(&program, input(), [FunctionId::from_index(2)]) else {
            panic!("the program holds two functions")
        };
        assert_eq!(err.to_string(), "@2 is not a function of this program");

        let Err(err) = ProgramStepper::new(&program, input(), [program.entry()]) else {
            panic!("an external entry point could never be entered")
        };
        assert_eq!(
            err.to_string(),
            "@1 is the entry point, which cannot be external"
        );
    }

    #[test]
    fn inputs_arranged_differently_than_declared_are_rejected() {
        let program = calling_program();
        let inputs = DataTree::sequence([DataTree::Leaf(one(1.5))]);

        let Err(err) = ProgramStepper::new(&program, inputs, [callee()]) else {
            panic!("a branch of one leaf cannot stand in for a leaf")
        };
        assert_eq!(
            err.to_string(),
            "inputs are structured [_] but the program declares _"
        );
    }

    #[test]
    fn an_input_that_does_not_match_its_declared_type_is_rejected() {
        let program = calling_program();
        let inputs = DataTree::Leaf(Tensor::from([2_i64]));

        let Err(err) = ProgramStepper::new(&program, inputs, [callee()]) else {
            panic!("a program is monomorphic, so an I64 input is rejected")
        };
        assert_eq!(err.to_string(), "argument 0: expected F64[1], got I64[1]");
    }

    #[test]
    fn an_instruction_needing_a_backend_is_refused_unless_its_function_is_external() {
        // A shot loop is the case this exists for: Qiskit cannot perform one, so a program holding
        // one is evaluable only where the function holding it has been handed over.
        let program = calls_a_vendor_function();
        let input = || DataTree::Leaf(one(1.5));

        let Err(err) = ProgramStepper::new(&program, input(), []) else {
            panic!("@0 holds work Qiskit cannot perform")
        };
        assert_eq!(
            err.to_string(),
            "@0 instruction 1 (vendor.elsewhere) has no built-in implementation",
            "the function as well as the instruction is named"
        );
        assert!(ProgramStepper::new(&program, input(), [callee()]).is_ok());
    }

    #[test]
    fn a_failing_instruction_names_its_function() {
        // @0 is external and the entry point @1 holds the op that fails, so the function named is
        // the one the failure is in rather than the entry point.
        let external = double_function(1);
        let signature = external.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap()[0];
        let failed = entry
            .add_op(Elsewhere { builtin: true }, &[called])
            .unwrap()[0];
        entry.add_result(failed).unwrap();
        let program = QuantumProgram::new(vec![external, entry], leaf(), leaf()).unwrap();

        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        stepper
            .fulfil(&program, RequestId::from_index(0), vec![one(10.0)])
            .unwrap();

        let Err(err) = stepper.step(&program) else {
            panic!("vendor.elsewhere fails as it runs")
        };
        assert!(
            matches!(err, StepperError::InstructionFailed { function, .. }
                if function == program.entry())
        );
        assert_eq!(
            err.to_string(),
            "evaluating @1 instruction 2 (vendor.elsewhere)"
        );

        // The stepper stayed where it stopped, so the same step reports the same failure.
        assert!(stepper.step(&program).is_err());
    }

    #[test]
    fn fulfilling_a_request_wrongly_is_rejected() {
        let program = calling_program();
        let mut stepper = start(&program, 1.5, [callee()]);
        stepper.step(&program).unwrap();
        let id = RequestId::from_index(0);

        let Err(err) = stepper.fulfil(&program, id, vec![one(1.0), one(2.0)]) else {
            panic!("the call declares one result")
        };
        assert_eq!(err.to_string(), "request #0 takes 1 output(s), got 2");

        let Err(err) = stepper.fulfil(&program, id, vec![Tensor::from([1_i64])]) else {
            panic!("the call declares an F64[1]")
        };
        assert_eq!(
            err.to_string(),
            "request #0 output 0: expected F64[1], got I64[1]"
        );

        let Err(err) = stepper.fulfil(&program, RequestId::from_index(7), vec![one(1.0)]) else {
            panic!("no seventh request was raised")
        };
        assert_eq!(err.to_string(), "request #7 is not outstanding");

        // Each rejection left the request outstanding, so it can still be answered, and only once.
        stepper.fulfil(&program, id, vec![one(10.0)]).unwrap();
        let Err(err) = stepper.fulfil(&program, id, vec![one(10.0)]) else {
            panic!("a request is answered once")
        };
        assert_eq!(err.to_string(), "request #0 is not outstanding");
    }

    // ---------------------------------------------------------------------------
    // Test ops
    // ---------------------------------------------------------------------------

    /// An op defined outside the crate whose `eval` always fails, reporting `builtin` for
    /// [`ProgramOp::has_builtin_eval`]. A backend contributes work Qiskit cannot perform with
    /// `builtin` false; with it true, the failure lands in the middle of a walk.
    #[derive(Clone, Debug)]
    struct Elsewhere {
        builtin: bool,
    }

    /// The error [`Elsewhere`] returns when asked to evaluate itself.
    #[derive(Debug)]
    struct NoImplementation;

    impl fmt::Display for NoImplementation {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "vendor.elsewhere has no in-process implementation")
        }
    }

    impl std::error::Error for NoImplementation {}

    impl ProgramOp for Elsewhere {
        type Error = NoImplementation;

        fn name(&self) -> &str {
            "elsewhere"
        }

        fn namespace(&self) -> &str {
            "vendor"
        }

        fn arity(&self) -> usize {
            1
        }

        fn has_builtin_eval(&self) -> bool {
            self.builtin
        }

        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, Self::Error> {
            Ok(vec![inputs[0].clone()])
        }

        fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
            Err(NoImplementation)
        }
    }

    /// Return a program whose entry point @1 calls @0, which holds work Qiskit cannot perform.
    fn calls_a_vendor_function() -> QuantumProgram {
        let mut vendor = ProgramFunction::new();
        let x = vendor.add_parameter(f64_1d(1));
        let out = vendor.add_op(Elsewhere { builtin: false }, &[x]).unwrap()[0];
        vendor.add_result(out).unwrap();
        let signature = vendor.signature();
        QuantumProgram::new(
            vec![vendor, calls(FunctionId::from_index(0), &signature)],
            leaf(),
            leaf(),
        )
        .unwrap()
    }

    /// An op that counts how often it has been evaluated and hands its operand straight back,
    /// which is how a test observes whether one pass reached it.
    #[derive(Clone, Debug)]
    struct Tally(Arc<AtomicUsize>);

    impl ProgramOp for Tally {
        type Error = std::convert::Infallible;

        fn name(&self) -> &str {
            "tally"
        }

        fn namespace(&self) -> &str {
            "vendor"
        }

        fn arity(&self) -> usize {
            1
        }

        fn has_builtin_eval(&self) -> bool {
            true
        }

        fn infer_output_types(
            &self,
            inputs: &[TensorType],
        ) -> Result<Vec<TensorType>, Self::Error> {
            Ok(vec![inputs[0].clone()])
        }

        fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(args.to_vec())
        }
    }
}
