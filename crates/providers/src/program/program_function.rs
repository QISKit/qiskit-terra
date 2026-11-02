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

use std::fmt;

use thiserror::Error;

use super::quantum_program::FunctionId;
use crate::ops::{BoxedOpError, BoxedProgramOp, ErasedProgramOp, ProgramOp, QISKIT};
use crate::tensor::{Tensor, TensorType};

/// A position within one instruction's ordered operands or results.
type Slot = u16;

/// An instruction's position in the function that holds it.
///
/// Ids are dense, never reused, and only meaningful within the function that issued them.
/// They are also an evaluation order: every operand of an instruction is produced by a strictly
/// lower id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstructionId(u32);

impl InstructionId {
    /// Return the underlying index, for use as a dense array subscript.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for InstructionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One tensor value: an output slot of the instruction that produces it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value {
    instruction: InstructionId,
    slot: Slot,
}

impl Value {
    /// Return the instruction that produces this value.
    pub fn instruction(self) -> InstructionId {
        self.instruction
    }

    /// Return which of that instruction's results this value is.
    pub fn slot(self) -> usize {
        self.slot as usize
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.slot == 0 {
            write!(f, "%{}", self.instruction)
        } else {
            write!(f, "%{}.{}", self.instruction, self.slot)
        }
    }
}

/// What part an instruction plays in its function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstructionRole {
    /// A function input, supplied by the caller.
    Parameter,
    /// An operation, with a [`ProgramOp`].
    Op,
    /// A call to another function of the same program.
    Call,
    /// A function output.
    Result,
}

/// View of an instruction.
#[derive(Clone, Copy)]
pub enum InstructionView<'a> {
    /// A function input, supplied by the caller.
    Parameter,
    /// An operation applying this op.
    Op(&'a (dyn ErasedProgramOp + 'static)),
    /// A call to this function of the same program.
    Call(FunctionId),
    /// A function output.
    Result,
}

impl<'a> InstructionView<'a> {
    /// Return what part the instruction plays, without its payload.
    pub fn role(self) -> InstructionRole {
        match self {
            Self::Parameter => InstructionRole::Parameter,
            Self::Op(_) => InstructionRole::Op,
            Self::Call(_) => InstructionRole::Call,
            Self::Result => InstructionRole::Result,
        }
    }

    fn name(self) -> &'a str {
        match self {
            Self::Parameter => "parameter",
            Self::Op(op) => op.name(),
            Self::Call(_) => "call",
            Self::Result => "result",
        }
    }

    fn namespace(self) -> &'a str {
        match self {
            Self::Parameter | Self::Call(_) | Self::Result => QISKIT,
            Self::Op(op) => op.namespace(),
        }
    }

    fn full_name(self) -> String {
        match self {
            Self::Parameter | Self::Call(_) | Self::Result => {
                format!("{}.{}", self.namespace(), self.name())
            }
            Self::Op(op) => op.full_name(),
        }
    }

    /// Return how many operands the instruction takes, or `None` when only the callee settles that.
    /// A call takes one operand per parameter of the function it names.
    fn arity(self) -> Option<usize> {
        match self {
            Self::Parameter => Some(0),
            Self::Op(op) => Some(op.arity()),
            Self::Call(_) => None,
            Self::Result => Some(1),
        }
    }

    fn has_builtin_eval(self) -> bool {
        match self {
            Self::Parameter | Self::Result => true,
            Self::Op(op) => op.has_builtin_eval(),
            // A call is evaluated by the program holding its callee, so a function containing one
            // cannot be evaluated on its own.
            Self::Call(_) => false,
        }
    }
}

/// The types a function consumes and produces, positionally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    /// The type of each parameter, in declaration order.
    pub inputs: Vec<TensorType>,
    /// The type of each declared result, in declaration order.
    pub outputs: Vec<TensorType>,
}

/// The evaluation content of an instruction.
enum InstructionBody {
    Parameter,
    Op(BoxedProgramOp),
    Call(FunctionId),
    Result,
}

impl InstructionBody {
    /// Return what this instruction is, and what it holds.
    fn view(&self) -> InstructionView<'_> {
        match self {
            Self::Parameter => InstructionView::Parameter,
            Self::Op(op) => InstructionView::Op(&**op),
            Self::Call(callee) => InstructionView::Call(*callee),
            Self::Result => InstructionView::Result,
        }
    }
}

/// One instruction of a function: what it is, what it reads, and what it produces.
struct ProgramInstruction {
    body: InstructionBody,
    /// The values this instruction consumes, in operand order.
    operands: Vec<Value>,
    /// The types inference produced for this instruction's results, one per result.
    output_types: Vec<TensorType>,
}

/// A read-only view of one instruction of a [`ProgramFunction`].
#[derive(Clone, Copy)]
pub struct InstructionRef<'a> {
    function: &'a ProgramFunction,
    id: InstructionId,
}

impl<'a> InstructionRef<'a> {
    fn instruction(&self) -> &'a ProgramInstruction {
        &self.function.instructions[self.id.index()]
    }

    /// Return this instruction's position in its function, which is its identity.
    pub fn id(&self) -> InstructionId {
        self.id
    }

    /// Return what part this instruction plays: an operation, or one end of the function's
    /// boundary.
    pub fn role(&self) -> InstructionRole {
        self.view().role()
    }

    /// Return this instruction's type name within its namespace, `add` for instance.
    pub fn name(&self) -> &'a str {
        self.view().name()
    }

    /// Return the namespace this instruction's type belongs to, [`QISKIT`](crate::ops::QISKIT) for
    /// an op Qiskit defines.
    pub fn namespace(&self) -> &'a str {
        self.view().namespace()
    }

    /// Return the name that categorizes this instruction and that a backend dispatches on,
    /// `qiskit.add` for instance.
    pub fn full_name(&self) -> String {
        self.view().full_name()
    }

    /// Return whether this instruction has a built-in implementation.
    pub fn has_builtin_eval(&self) -> bool {
        self.view().has_builtin_eval()
    }

    /// Return what this instruction is, and what it holds.
    pub fn view(&self) -> InstructionView<'a> {
        self.instruction().body.view()
    }

    /// Return the values this instruction consumes, in operand order.
    pub fn operands(&self) -> &'a [Value] {
        &self.instruction().operands
    }

    /// Return the type of each operand, read from the instruction that produces it.
    pub fn operand_types(&self) -> impl Iterator<Item = &'a TensorType> {
        let function = self.function;
        self.instruction()
            .operands
            .iter()
            .map(move |&value| function.type_of(value).expect("an operand always exists"))
    }

    /// Return the values this instruction produces, in result order.
    pub fn outputs(&self) -> impl Iterator<Item = Value> + 'a {
        let instruction = self.id;
        (0..self.instruction().output_types.len() as Slot)
            .map(move |slot| Value { instruction, slot })
    }

    /// Return the type of each value this instruction produces, in result order.
    pub fn output_types(&self) -> &'a [TensorType] {
        &self.instruction().output_types
    }
}

/// Why an instruction or a result could not be added to a [`ProgramFunction`].
#[derive(Debug, Error)]
pub enum FunctionError {
    /// An operand names a value this function has not produced.
    #[error("operand {operand}: {value} was not produced by this function")]
    UnknownOperand { operand: usize, value: Value },

    /// A declared result names a value this function has not produced.
    #[error("{0} was not produced by this function")]
    UnknownValue(Value),

    /// The operand count does not match the instruction's declared [`ProgramOp::arity`].
    #[error("{full_name} takes {expected} operand(s), got {actual}")]
    OperandArity {
        full_name: String,
        expected: usize,
        actual: usize,
    },

    /// The instruction rejected the types of the values wired to it. This is the build-time
    /// counterpart of a run-time dtype or shape error.
    #[error("{full_name} rejected its operand types")]
    TypeError {
        full_name: String,
        #[source]
        source: BoxedOpError,
    },

    /// An operand of a call does not satisfy the parameter type declared for the callee.
    #[error("operand {operand} of the call to {callee}: expected {expected}, got {actual}")]
    CallOperandType {
        operand: usize,
        callee: FunctionId,
        expected: TensorType,
        actual: TensorType,
    },
}

/// Why [`ProgramFunction::eval`] could not produce results.
#[derive(Debug, Error)]
pub enum FunctionEvalError {
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

    /// The function contains an instruction Qiskit has no built-in implementation of.
    #[error("instruction {instruction} ({full_name}) has no built-in implementation")]
    NoBuiltinEval {
        instruction: InstructionId,
        full_name: String,
    },

    /// An instruction returned an error from its [`ProgramOp::eval`].
    #[error("evaluating instruction {instruction} ({full_name})")]
    InstructionFailed {
        instruction: InstructionId,
        full_name: String,
        #[source]
        source: BoxedOpError,
    },

    /// A function reached through a call instruction failed.
    #[error("evaluating the call at instruction {instruction} to {callee}")]
    CallFailed {
        instruction: InstructionId,
        callee: FunctionId,
        #[source]
        source: Box<FunctionEvalError>,
    },

    /// An instruction's `eval` returned a different number of tensors than its type inference expected.
    #[error("instruction {instruction} returned {actual} result(s), expected {expected}")]
    ResultArityMismatch {
        instruction: InstructionId,
        expected: usize,
        actual: usize,
    },
}

/// A tensor dataflow of instructions.
///
/// Each instruction can have one of four roles:
///  * parameter: an input to the function specifying exactly one tensor demanded at call time
///  * result: an output to the function specifying one tensor the function returns
///  * call: a call to another function defined in a [`QuantumProgram`](super::QuantumProgram)
///  * op: some atomic operation represented as a [`ProgramOp`]
///
/// An operand is a [`Value`], which names an existing instruction and one of its output slots. An
/// instruction can be added only once every value it uses exists.
///
/// Operand types are checked against the instruction as it is added, so a `ProgramFunction` cannot
/// be malformed by construction. A call instruction is the exception: its contract is checked against
/// the function it names when a [`QuantumProgram`](super::QuantumProgram) is assembled. Because of SSA
/// discipline, instruction order is a valid execution order.
pub struct ProgramFunction {
    /// Every instruction, indexed by [`InstructionId`].
    instructions: Vec<ProgramInstruction>,
    /// Indices of all parameter instructions in declaration order.
    parameters: Vec<InstructionId>,
    /// Indices of all result instructions in declaration order. Two of them may read one value.
    results: Vec<InstructionId>,
}

impl Default for ProgramFunction {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramFunction {
    /// Construct a new, empty function.
    pub fn new() -> Self {
        Self {
            instructions: Vec::new(),
            parameters: Vec::new(),
            results: Vec::new(),
        }
    }

    /// Declare a parameter of the given type, returning its value.
    pub fn add_parameter(&mut self, ty: TensorType) -> Value {
        let id = self.push(InstructionBody::Parameter, Vec::new(), vec![ty]);
        self.parameters.push(id);
        Value {
            instruction: id,
            slot: 0,
        }
    }

    /// Apply `op` to `operands`, returning the values it produces in result order.
    pub fn add_op<O>(&mut self, op: O, operands: &[Value]) -> Result<Vec<Value>, FunctionError>
    where
        O: ProgramOp + Clone + Send + Sync + 'static,
        O::Error: std::error::Error + Send + Sync + 'static,
    {
        self.add_boxed_op(Box::new(op), operands)
    }

    /// Apply a boxed op, returning the values it produces in result order.
    pub fn add_boxed_op(
        &mut self,
        op: BoxedProgramOp,
        operands: &[Value],
    ) -> Result<Vec<Value>, FunctionError> {
        if operands.len() != op.arity() {
            return Err(FunctionError::OperandArity {
                full_name: op.full_name(),
                expected: op.arity(),
                actual: operands.len(),
            });
        }

        let mut operand_types = Vec::with_capacity(operands.len());
        for (operand, &value) in operands.iter().enumerate() {
            let Some(ty) = self.type_of(value) else {
                return Err(FunctionError::UnknownOperand { operand, value });
            };
            operand_types.push(ty.clone());
        }

        let output_types =
            op.infer_output_types(&operand_types)
                .map_err(|source| FunctionError::TypeError {
                    full_name: op.full_name(),
                    source,
                })?;

        let id = self.push(InstructionBody::Op(op), operands.to_vec(), output_types);
        Ok(self
            .instruction(id)
            .expect("the instruction was just pushed")
            .outputs()
            .collect())
    }

    /// Invoke `callee` on `operands`, returning the values it produces in result order.
    ///
    /// The caller supplies the id `callee` will have within its
    /// [`QuantumProgram`](super::QuantumProgram).
    pub fn add_call(
        &mut self,
        callee: FunctionId,
        signature: &Signature,
        operands: &[Value],
    ) -> Result<Vec<Value>, FunctionError> {
        if operands.len() != signature.inputs.len() {
            return Err(FunctionError::OperandArity {
                full_name: InstructionView::Call(callee).full_name(),
                expected: signature.inputs.len(),
                actual: operands.len(),
            });
        }

        for (operand, (&value, expected)) in operands.iter().zip(&signature.inputs).enumerate() {
            let Some(actual) = self.type_of(value) else {
                return Err(FunctionError::UnknownOperand { operand, value });
            };
            if !expected.admits(actual) {
                return Err(FunctionError::CallOperandType {
                    operand,
                    callee,
                    expected: expected.clone(),
                    actual: actual.clone(),
                });
            }
        }

        let id = self.push(
            InstructionBody::Call(callee),
            operands.to_vec(),
            signature.outputs.clone(),
        );
        Ok(self
            .instruction(id)
            .expect("the instruction was just pushed")
            .outputs()
            .collect())
    }

    /// Declare `value` as the next result of this function.
    pub fn add_result(&mut self, value: Value) -> Result<(), FunctionError> {
        if self.type_of(value).is_none() {
            return Err(FunctionError::UnknownValue(value));
        }
        let id = self.push(InstructionBody::Result, vec![value], Vec::new());
        self.results.push(id);
        Ok(())
    }

    /// Append an instruction, returning the id it was given.
    fn push(
        &mut self,
        body: InstructionBody,
        operands: Vec<Value>,
        output_types: Vec<TensorType>,
    ) -> InstructionId {
        debug_assert!(
            body.view()
                .arity()
                .is_none_or(|arity| operands.len() == arity),
            "an instruction stores exactly as many operands as its arity"
        );
        let id = InstructionId(self.instructions.len() as u32);
        self.instructions.push(ProgramInstruction {
            body,
            operands,
            output_types,
        });
        id
    }

    /// Return the type of `value`, or `None` if it does not belong to this function.
    pub fn type_of(&self, value: Value) -> Option<&TensorType> {
        self.instructions
            .get(value.instruction.index())?
            .output_types
            .get(value.slot())
    }

    /// Return the parameter instructions, in declaration order.
    pub fn parameters(&self) -> &[InstructionId] {
        &self.parameters
    }

    /// Return the result instructions, in declaration order.
    pub fn results(&self) -> &[InstructionId] {
        &self.results
    }

    /// Return the value of each parameter, in declaration order.
    pub fn parameter_values(&self) -> impl Iterator<Item = Value> + '_ {
        self.parameters.iter().map(|&instruction| Value {
            instruction,
            slot: 0,
        })
    }

    /// Return the value each result returns, in declaration order.
    pub fn result_values(&self) -> impl Iterator<Item = Value> + '_ {
        self.results
            .iter()
            .map(|&instruction| self.instructions[instruction.index()].operands[0])
    }

    /// Return this function's type contract: the types of its parameters and of its results.
    pub fn signature(&self) -> Signature {
        Signature {
            inputs: self.value_types(self.parameter_values()),
            outputs: self.value_types(self.result_values()),
        }
    }

    /// Return a view of the instruction with the given id, or `None`.
    pub fn instruction(&self, id: InstructionId) -> Option<InstructionRef<'_>> {
        (id.index() < self.instructions.len()).then_some(InstructionRef { function: self, id })
    }

    /// Return the number of instructions in this function, boundary instructions included.
    pub fn instruction_count(&self) -> usize {
        self.instructions.len()
    }

    /// Iterate over every instruction.
    ///
    /// By construction, this is in topological order.
    pub fn iter_instructions(&self) -> impl Iterator<Item = InstructionRef<'_>> {
        (0..self.instructions.len() as u32).map(|id| InstructionRef {
            function: self,
            id: InstructionId(id),
        })
    }

    /// Return whether every instruction of this function has a built-in implementation.
    pub fn has_builtin_eval(&self) -> bool {
        self.first_without_builtin_eval().is_none()
    }

    /// Evaluate this function against `args`, one per parameter in declaration order.
    ///
    /// Fails when any instruction has no built-in evaluation implementation.
    pub fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, FunctionEvalError> {
        // The function is monomorphic, so its declared parameter types are the only ones its
        // instructions were built for. Checking them here names the argument the caller supplied.
        self.check_argument_types(args)?;
        if let Some(instruction) = self.first_without_builtin_eval() {
            return Err(FunctionEvalError::NoBuiltinEval {
                instruction: instruction.id(),
                full_name: instruction.full_name(),
            });
        }
        self.walk(args, &[])
    }

    /// Evaluate this function against `args`, resolving each call instruction against `functions`.
    ///
    /// The caller establishes that every instruction this may reach has a built-in implementation,
    /// and that `functions` holds the callee of every call instruction reached, since both are
    /// properties of the whole program rather than of this function.
    pub(super) fn eval_in(
        &self,
        args: &[Tensor],
        functions: &[ProgramFunction],
    ) -> Result<Vec<Tensor>, FunctionEvalError> {
        self.check_argument_types(args)?;
        self.walk(args, functions)
    }

    /// Walk the instructions in storage order over a single dense environment, releasing each intermediate
    /// once its last consumer has run.
    fn walk(
        &self,
        args: &[Tensor],
        functions: &[ProgramFunction],
    ) -> Result<Vec<Tensor>, FunctionEvalError> {
        let offsets = self.value_offsets();
        let last_use = self.last_use(&offsets);
        let flat = |value: Value| offsets[value.instruction.index()] as usize + value.slot();

        // Every operand is present: it was produced by an earlier instruction, and it cannot have
        // been released, because this instruction's use of it is at or before its last.
        let gather = |operands: &[Value], env: &[Option<Tensor>]| -> Vec<Tensor> {
            operands
                .iter()
                .map(|&value| {
                    env[flat(value)]
                        .clone()
                        .expect("an operand is produced before its consumer runs")
                })
                .collect()
        };

        let total = *offsets.last().expect("offsets always end with the total");
        let mut env: Vec<Option<Tensor>> = vec![None; total as usize];
        let mut outputs: Vec<Option<Tensor>> = vec![None; self.results.len()];
        let mut next_result = 0;

        for (position, instruction) in self.instructions.iter().enumerate() {
            let id = InstructionId(position as u32);
            match &instruction.body {
                InstructionBody::Parameter => {
                    // Parameters are the sources of the dataflow: their values come from the caller,
                    // in the order the parameters were declared.
                    let parameter = self
                        .parameters
                        .iter()
                        .position(|&declared| declared == id)
                        .expect("a parameter instruction is always in `parameters`");
                    env[offsets[position] as usize] = Some(args[parameter].clone());
                }
                InstructionBody::Op(op) => {
                    let operands = gather(&instruction.operands, &env);
                    let results = op.eval(&operands).map_err(|source| {
                        FunctionEvalError::InstructionFailed {
                            instruction: id,
                            full_name: op.full_name(),
                            source,
                        }
                    })?;
                    if results.len() != instruction.output_types.len() {
                        return Err(FunctionEvalError::ResultArityMismatch {
                            instruction: id,
                            expected: instruction.output_types.len(),
                            actual: results.len(),
                        });
                    }
                    for (slot, tensor) in results.into_iter().enumerate() {
                        env[offsets[position] as usize + slot] = Some(tensor);
                    }
                }
                InstructionBody::Call(callee) => {
                    // Assembling the program checked this call against the function it names, so
                    // the callee is present and its results land one per output slot.
                    let arguments = gather(&instruction.operands, &env);
                    let function = functions
                        .get(callee.index())
                        .expect("a call names a function of the program being evaluated");
                    let results = function.walk(&arguments, functions).map_err(|source| {
                        FunctionEvalError::CallFailed {
                            instruction: id,
                            callee: *callee,
                            source: Box::new(source),
                        }
                    })?;
                    for (slot, tensor) in results.into_iter().enumerate() {
                        env[offsets[position] as usize + slot] = Some(tensor);
                    }
                }
                InstructionBody::Result => {
                    outputs[next_result] = env[flat(instruction.operands[0])].clone();
                    next_result += 1;
                }
            }

            // A result instruction is an ordinary final consumer, so nothing here has to make an
            // exception for a value the function returns.
            for &value in &instruction.operands {
                if last_use[flat(value)] == Some(position) {
                    env[flat(value)] = None;
                }
            }
            for slot in 0..instruction.output_types.len() {
                let index = offsets[position] as usize + slot;
                if last_use[index] == Some(position) {
                    env[index] = None;
                }
            }
        }

        Ok(outputs
            .into_iter()
            .map(|tensor| tensor.expect("every result instruction runs"))
            .collect())
    }

    /// Return the first instruction Qiskit has no in-process implementation of.
    ///
    /// [`Self::eval`] consults this before computing anything, so a function that needs a backend
    /// fails at the top rather than part-way through a walk that has produced intermediates.
    fn first_without_builtin_eval(&self) -> Option<InstructionRef<'_>> {
        self.iter_instructions()
            .find(|instruction| !instruction.has_builtin_eval())
    }

    /// Return the types of `values`, which must all belong to this function.
    fn value_types(&self, values: impl Iterator<Item = Value>) -> Vec<TensorType> {
        values
            .map(|value| {
                self.type_of(value)
                    .expect("a boundary value always exists")
                    .clone()
            })
            .collect()
    }

    /// Validate that every argument satisfies its parameter's declared type.
    fn check_argument_types(&self, args: &[Tensor]) -> Result<(), FunctionEvalError> {
        if args.len() != self.parameters.len() {
            return Err(FunctionEvalError::ArgumentArity {
                expected: self.parameters.len(),
                actual: args.len(),
            });
        }
        for (parameter, (arg, value)) in args.iter().zip(self.parameter_values()).enumerate() {
            let expected = self.type_of(value).expect("a parameter always exists");
            if !arg.matches(expected) {
                return Err(FunctionEvalError::ArgumentTypeMismatch {
                    parameter,
                    expected: expected.clone(),
                    actual: arg.tensor_type(),
                });
            }
        }
        Ok(())
    }

    /// Return where each instruction's block of values starts in a dense environment.
    fn value_offsets(&self) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(self.instructions.len() + 1);
        let mut total = 0;
        for instruction in &self.instructions {
            offsets.push(total);
            total += instruction.output_types.len() as u32;
        }
        offsets.push(total);
        offsets
    }

    /// Return, for each value, the instruction position after which it is not used again.
    fn last_use(&self, offsets: &[u32]) -> Vec<Option<usize>> {
        let total = *offsets.last().expect("offsets always end with the total");
        let mut last = vec![None; total as usize];
        for (position, instruction) in self.instructions.iter().enumerate() {
            for slot in 0..instruction.output_types.len() {
                last[offsets[position] as usize + slot] = Some(position);
            }
            // A consumer overwrites the producer's own position, which is strictly later.
            for &value in &instruction.operands {
                last[offsets[value.instruction.index()] as usize + value.slot()] = Some(position);
            }
        }
        last
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::ops::{Add, BroadcastTo, Cast, Constant, MathOpError, Mean};
    use crate::tensor::{DType, Dim, TensorError};
    use ndarray::arr2;

    /// Return the type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    // ---------------------------------------------------------------------------
    // Building and evaluating
    // ---------------------------------------------------------------------------

    #[test]
    fn a_function_adds_two_parameters_and_evaluates() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(3));
        let sum = function.add_op(Add, &[x, y]).unwrap();
        assert_eq!(sum.len(), 1);
        function.add_result(sum[0]).unwrap();

        let results = function
            .eval(&[
                Tensor::from([1.0_f64, 2.0, 3.0]),
                Tensor::from([10.0_f64, 20.0, 30.0]),
            ])
            .unwrap();

        assert_eq!(results, vec![Tensor::from([11.0_f64, 22.0, 33.0])]);
    }

    #[test]
    fn a_signature_is_available_without_evaluating() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let y = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        let mean = function.add_op(Mean::new(0), &[sum]).unwrap()[0];
        function.add_result(mean).unwrap();

        assert_eq!(
            function.signature(),
            Signature {
                inputs: vec![f64_1d(2), f64_1d(2)],
                outputs: vec![TensorType {
                    dtype: DType::F64,
                    shape: vec![],
                }],
            }
        );
        assert_eq!(function.type_of(sum), Some(&f64_1d(2)));
    }

    #[test]
    fn parameters_and_results_are_positional() {
        // The two parameters have distinct types, so the order in which they reach the instruction
        // is observable in the result.
        let mut function = ProgramFunction::new();
        let short = function.add_parameter(f64_1d(1));
        let long = function.add_parameter(f64_1d(2));
        assert_eq!(
            function.parameter_values().collect::<Vec<_>>(),
            vec![short, long]
        );

        // Both results name the same value, which is allowed, and a parameter may be returned
        // directly.
        function.add_result(long).unwrap();
        function.add_result(long).unwrap();
        function.add_result(short).unwrap();
        assert_eq!(
            function.result_values().collect::<Vec<_>>(),
            vec![long, long, short]
        );

        let results = function
            .eval(&[Tensor::from([1.0_f64]), Tensor::from([2.0_f64, 3.0])])
            .unwrap();
        assert_eq!(
            results,
            vec![
                Tensor::from([2.0_f64, 3.0]),
                Tensor::from([2.0_f64, 3.0]),
                Tensor::from([1.0_f64]),
            ]
        );
    }

    #[test]
    fn a_value_may_be_consumed_more_than_once() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let doubled = function.add_op(Add, &[x, x]).unwrap()[0];
        let quadrupled = function.add_op(Add, &[doubled, doubled]).unwrap()[0];
        function.add_result(quadrupled).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 2.0])]).unwrap(),
            vec![Tensor::from([4.0_f64, 8.0])]
        );
    }

    #[test]
    fn an_instruction_whose_results_nothing_uses_does_not_disturb_the_rest() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        function.add_op(Mean::new(0), &[x]).unwrap();
        let sum = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 2.0])]).unwrap(),
            vec![Tensor::from([2.0_f64, 4.0])]
        );
    }

    #[test]
    fn reducing_a_zero_length_axis_yields_a_non_finite_value() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(0));
        let mean = function.add_op(Mean::new(0), &[x]).unwrap()[0];
        function.add_result(mean).unwrap();

        let results = function
            .eval(&[Tensor::from(&[] as &[f64])])
            .expect("an empty reduction is not an error");
        let [Tensor::F64(mean)] = &results[..] else {
            panic!("expected one F64 result, got {results:?}")
        };
        assert!(mean.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn a_function_can_be_sent_between_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ProgramFunction>();
    }

    #[test]
    fn a_constant_holds_one_tensor() {
        let mut function = ProgramFunction::new();
        let two = function
            .add_op(Constant::new(Tensor::from([2.0_f64, 2.0])), &[])
            .unwrap();
        assert_eq!(two.len(), 1);
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, two[0]]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.eval(&[Tensor::from([1.0_f64, 10.0])]).unwrap(),
            vec![Tensor::from([3.0_f64, 12.0])]
        );
    }

    // ---------------------------------------------------------------------------
    // Instructions, values, and roles
    // ---------------------------------------------------------------------------

    #[test]
    fn an_op_name_carries_its_namespace() {
        assert_eq!(ProgramOp::full_name(&Add), "qiskit.add");
        assert_eq!(ProgramOp::full_name(&Elsewhere), "vendor.elsewhere");
    }

    #[test]
    fn a_function_reports_the_instructions_it_holds() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        function.add_op(Add, &[x, x]).unwrap();
        let second = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(second).unwrap();

        assert_eq!(
            function
                .iter_instructions()
                .filter(|instruction| instruction.full_name() == "qiskit.add")
                .count(),
            2
        );
        // Boundary instructions are instructions, so they are counted too: one parameter, two adds,
        // one result.
        assert_eq!(function.instruction_count(), 4);

        let add = function
            .iter_instructions()
            .find(|instruction| instruction.name() == "add")
            .unwrap();
        assert_eq!(add.role(), InstructionRole::Op);
        assert_eq!(add.operands(), &[x, x]);
        assert_eq!(add.outputs().count(), 1);
        assert!(add.has_builtin_eval());
    }

    #[test]
    fn a_functions_boundary_is_made_of_instructions() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(sum).unwrap();

        let roles: Vec<InstructionRole> = function
            .iter_instructions()
            .map(|instruction| instruction.role())
            .collect();
        assert_eq!(
            roles,
            vec![
                InstructionRole::Parameter,
                InstructionRole::Op,
                InstructionRole::Result
            ]
        );

        // A boundary instruction is named like any other, so a consumer matching on type names sees
        // it.
        let names: Vec<String> = function
            .iter_instructions()
            .map(|instruction| instruction.full_name())
            .collect();
        assert_eq!(
            names,
            vec!["qiskit.parameter", "qiskit.add", "qiskit.result"]
        );

        // A parameter produces its value and consumes nothing; a result is the other way
        // round.
        let parameter = function.instruction(function.parameters()[0]).unwrap();
        assert!(parameter.operands().is_empty());
        assert_eq!(parameter.output_types(), &[f64_1d(2)]);

        let result = function.instruction(function.results()[0]).unwrap();
        assert_eq!(result.operands(), &[sum]);
        assert!(result.output_types().is_empty());
    }

    #[test]
    fn a_value_names_the_instruction_and_slot_that_produced_it() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let sum = function.add_op(Add, &[x, x]).unwrap()[0];

        assert_eq!(x.instruction(), function.parameters()[0]);
        assert_eq!(x.slot(), 0);
        assert_eq!(
            sum.instruction().index(),
            1,
            "the add is the second instruction"
        );
        assert_eq!(sum.to_string(), "%1");

        // An operand's type is read through the instruction that produced it rather than stored
        // twice.
        let add = function.instruction(sum.instruction()).unwrap();
        assert_eq!(
            add.operand_types().collect::<Vec<_>>(),
            vec![&f64_1d(2), &f64_1d(2)]
        );
    }

    // ---------------------------------------------------------------------------
    // Calls
    // ---------------------------------------------------------------------------

    fn double_signature(len: usize) -> Signature {
        Signature {
            inputs: vec![f64_1d(len)],
            outputs: vec![f64_1d(len)],
        }
    }

    #[test]
    fn a_call_names_a_function_by_id_and_one_body_serves_several_sites() {
        let callee = FunctionId::from_index(0);
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        let once = function
            .add_call(callee, &double_signature(2), &[x])
            .unwrap();
        assert_eq!(once.len(), 1, "one value per result the signature declares");
        let twice = function
            .add_call(callee, &double_signature(2), &once)
            .unwrap()[0];
        function.add_result(twice).unwrap();

        let call = function.instruction(once[0].instruction()).unwrap();
        assert_eq!(call.role(), InstructionRole::Call);
        assert_eq!(call.full_name(), "qiskit.call");
        assert!(matches!(call.view(), InstructionView::Call(named) if named == callee));
        assert_eq!(call.operands(), &[x]);
        assert_eq!(
            call.output_types(),
            &[f64_1d(2)],
            "the declared result types are the call instruction's own"
        );

        // Both sites name the same function, and only a call instruction names one at all.
        assert_eq!(
            function
                .iter_instructions()
                .filter_map(named_callee)
                .collect::<Vec<_>>(),
            vec![callee, callee]
        );
        assert_eq!(
            named_callee(function.instruction(x.instruction()).unwrap()),
            None
        );
    }

    /// Return the function `instruction` calls, for an instruction that calls one.
    fn named_callee(instruction: InstructionRef<'_>) -> Option<FunctionId> {
        match instruction.view() {
            InstructionView::Call(callee) => Some(callee),
            _ => None,
        }
    }

    #[test]
    fn a_call_operand_is_checked_against_the_parameter_declared_for_it() {
        let callee = FunctionId::from_index(0);
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));

        let Err(err) = function.add_call(callee, &double_signature(3), &[x]) else {
            panic!("an F64[2] operand cannot fill an F64[3] parameter")
        };
        assert!(
            matches!(
                err,
                FunctionError::CallOperandType {
                    operand: 0,
                    callee,
                    expected,
                    actual,
                } if callee == FunctionId::from_index(0)
                    && expected == f64_1d(3)
                    && actual == f64_1d(2)
            ),
            "the position, the callee, and both types are named"
        );

        let Err(err) = function.add_call(callee, &double_signature(2), &[x, x]) else {
            panic!("a one-parameter signature takes one operand")
        };
        assert!(matches!(
            err,
            FunctionError::OperandArity {
                full_name,
                expected: 1,
                actual: 2,
            } if full_name == "qiskit.call"
        ));

        // A value from another function is unknown here, so long as this one has no instruction at
        // its index.
        let mut other = ProgramFunction::new();
        other.add_parameter(f64_1d(2));
        let stranger = other.add_parameter(f64_1d(2));
        let Err(err) = function.add_call(callee, &double_signature(2), &[stranger]) else {
            panic!("an operand from another function is rejected")
        };
        assert!(matches!(
            err,
            FunctionError::UnknownOperand { operand: 0, .. }
        ));
    }

    #[test]
    fn a_call_operand_may_be_narrower_than_the_parameter_declared_for_it() {
        let bounded = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 4 }],
        };
        let callee = FunctionId::from_index(0);
        let mut function = ProgramFunction::new();
        let fixed = function.add_parameter(f64_1d(3));
        let loose = function.add_parameter(bounded.clone());

        let signature = Signature {
            inputs: vec![bounded],
            outputs: vec![],
        };
        assert!(function.add_call(callee, &signature, &[fixed]).is_ok());

        // Nothing proves the size of a bounded operand is within a true size.
        let signature = Signature {
            inputs: vec![f64_1d(3)],
            outputs: vec![],
        };
        let Err(err) = function.add_call(callee, &signature, &[loose]) else {
            panic!("a bounded operand cannot fill a parameter of a true size")
        };
        assert!(matches!(
            err,
            FunctionError::CallOperandType {
                operand: 0,
                callee,
                expected,
                actual,
            } if callee == FunctionId::from_index(0)
                && expected == f64_1d(3)
                && actual.shape == [Dim::Bounded { max: 4 }]
        ));
    }

    #[test]
    fn a_function_holding_a_call_cannot_be_evaluated_on_its_own() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let called = function
            .add_call(FunctionId::from_index(0), &double_signature(1), &[x])
            .unwrap()[0];
        function.add_result(called).unwrap();

        assert!(!function.has_builtin_eval());
        let Err(err) = function.eval(&[Tensor::from([1.0_f64])]) else {
            panic!("resolving a call needs the program holding its callee")
        };
        assert!(matches!(
            err,
            FunctionEvalError::NoBuiltinEval { full_name, .. } if full_name == "qiskit.call"
        ));
    }

    // ---------------------------------------------------------------------------
    // Build-time rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn the_wrong_operand_count_is_rejected() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));

        let Err(err) = function.add_op(Add, &[x]) else {
            panic!("a one-operand add is rejected")
        };
        assert!(matches!(
            err,
            FunctionError::OperandArity {
                full_name,
                expected: 2,
                actual: 1,
            } if full_name == "qiskit.add"
        ));
    }

    #[test]
    fn an_unknown_operand_is_rejected_naming_its_position() {
        let mut known = ProgramFunction::new();
        let x = known.add_parameter(f64_1d(1));

        let mut other = ProgramFunction::new();
        other.add_parameter(f64_1d(1));
        let stranger = other.add_parameter(f64_1d(1));

        let Err(err) = known.add_op(Add, &[x, stranger]) else {
            panic!("an operand from another function is rejected")
        };
        assert!(matches!(
            err,
            FunctionError::UnknownOperand { operand: 1, value } if value == stranger
        ));
    }

    #[test]
    fn an_unknown_result_is_rejected() {
        let mut known = ProgramFunction::new();
        let mut other = ProgramFunction::new();
        let stranger = other.add_parameter(f64_1d(1));

        let Err(err) = known.add_result(stranger) else {
            panic!("a result from another function is rejected")
        };
        assert!(matches!(err, FunctionError::UnknownValue(value) if value == stranger));
    }

    #[test]
    fn a_rejected_operand_type_is_reported_when_the_instruction_is_added() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(4));

        let Err(err) = function.add_op(Add, &[x, y]) else {
            panic!("shapes that do not broadcast are rejected")
        };
        let FunctionError::TypeError { full_name, source } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert_eq!(full_name, "qiskit.add", "the op is named");
        assert!(
            matches!(
                source.downcast_ref::<MathOpError>(),
                Some(MathOpError::Tensor(TensorError::DimShapeMismatch { lhs, rhs }))
                    if lhs == &[Dim::Fixed(3)] && rhs == &[Dim::Fixed(4)]
            ),
            "both operand shapes are named"
        );
    }

    #[test]
    fn a_dtype_the_instruction_cannot_compute_is_rejected_when_the_instruction_is_added() {
        // An instruction accepts only the dtypes its implementation covers, so a function that
        // type-checks cannot then fail on a dtype as it runs.
        let mut function = ProgramFunction::new();
        let bit = function.add_parameter(TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(2)],
        });

        let Err(err) = function.add_op(Add, &[bit, bit]) else {
            panic!("`add` has no Bit implementation, so a Bit operand is rejected")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert!(matches!(
            source.downcast_ref::<MathOpError>(),
            Some(MathOpError::UnsupportedPromotion {
                lhs: DType::Bit,
                rhs: DType::Bit,
                dtype: DType::Bit,
            })
        ));
    }

    #[test]
    fn a_reduction_out_of_bounds_axis_is_reported_when_the_instruction_is_added() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));

        let Err(err) = function.add_op(Mean::new(1), &[x]) else {
            panic!("an out-of-bounds axis is rejected")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert!(matches!(
            source.downcast_ref::<MathOpError>(),
            Some(MathOpError::InvalidAxis { axis: 1, ndim: 1 })
        ));
    }

    // ---------------------------------------------------------------------------
    // Coercion
    // ---------------------------------------------------------------------------

    #[test]
    fn arithmetic_promotes_its_operand_dtypes() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(TensorType {
            dtype: DType::F32,
            shape: vec![Dim::Fixed(3)],
        });
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(
            function.type_of(sum),
            Some(&f64_1d(3)),
            "F32 and F64 promote to F64"
        );
        assert_eq!(
            function
                .eval(&[
                    Tensor::from([1.0_f64, 2.0, 3.0]),
                    Tensor::from([10.0_f32, 20.0, 30.0]),
                ])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 22.0, 33.0])]
        );
    }

    #[test]
    fn arithmetic_broadcasts_its_operand_shapes() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let y = function.add_parameter(f64_1d(1));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();

        assert_eq!(function.type_of(sum), Some(&f64_1d(3)));
        assert_eq!(
            function
                .eval(&[Tensor::from([1.0_f64, 2.0, 3.0]), Tensor::from([10.0_f64])])
                .unwrap(),
            vec![Tensor::from([11.0_f64, 12.0, 13.0])]
        );
    }

    #[test]
    fn averaging_a_bit_tensor_yields_a_float() {
        let mut function = ProgramFunction::new();
        let shots = function.add_parameter(TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(4)],
        });
        let mean = function.add_op(Mean::new(0), &[shots]).unwrap()[0];
        function.add_result(mean).unwrap();

        assert_eq!(
            function.type_of(mean),
            Some(&TensorType {
                dtype: DType::F64,
                shape: vec![],
            })
        );
        let results = function
            .eval(&[Tensor::from([1_u8, 0, 1, 1]).cast(DType::Bit)])
            .unwrap();
        let [Tensor::F64(mean)] = &results[..] else {
            panic!("expected one F64 result, got {results:?}")
        };
        assert_eq!(mean.iter().copied().collect::<Vec<f64>>(), vec![0.75]);
    }

    #[test]
    fn mean_cast_and_broadcast_combined() {
        // Smoke test with mean, cast, and broadcast combined.
        let mut function = ProgramFunction::new();
        let shots = function.add_parameter(TensorType {
            dtype: DType::Bit,
            shape: vec![Dim::Fixed(4), Dim::Fixed(2)],
        });
        let mean = function.add_op(Mean::new(0), &[shots]).unwrap()[0];
        let whole = function.add_op(Cast::new(DType::I64), &[mean]).unwrap()[0];
        let per_shot = function
            .add_op(
                BroadcastTo::new(vec![Dim::Fixed(4), Dim::Fixed(2)]),
                &[whole],
            )
            .unwrap()[0];
        function.add_result(per_shot).unwrap();

        assert_eq!(
            function.type_of(per_shot),
            Some(&TensorType {
                dtype: DType::I64,
                shape: vec![Dim::Fixed(4), Dim::Fixed(2)],
            })
        );

        // Evaluate
        let bits = arr2(&[[1_u8, 0], [1, 1], [1, 1], [1, 1]]).into_dyn();
        let results = function
            .eval(&[Tensor::from(bits).cast(DType::Bit)])
            .unwrap();
        assert_eq!(
            results,
            vec![Tensor::I64(
                arr2(&[[1_i64, 0], [1, 0], [1, 0], [1, 0]])
                    .into_dyn()
                    .into_shared()
            )]
        );
    }

    #[test]
    fn two_bounded_axes_cannot_be_combined() {
        let bounded = TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }],
        };
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(bounded.clone());
        let y = function.add_parameter(bounded);

        let Err(err) = function.add_op(Add, &[x, y]) else {
            panic!("two bounded axes cannot be paired up")
        };
        let FunctionError::TypeError { source, .. } = &err else {
            panic!("expected a type error, got {err}")
        };
        assert!(matches!(
            source.downcast_ref::<MathOpError>(),
            Some(MathOpError::Tensor(TensorError::DynamicDim { shape }))
                if shape == &[Dim::Bounded { max: 8 }]
        ));
    }

    #[test]
    fn a_bounded_axis_broadcasts_against_a_fixed_one() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 8 }, Dim::Fixed(2)],
        });
        let y = function.add_parameter(f64_1d(2));
        let scaled = function.add_op(Add, &[x, y]).unwrap()[0];

        assert_eq!(
            function.type_of(scaled),
            Some(&TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Bounded { max: 8 }, Dim::Fixed(2)],
            })
        );
    }

    // ---------------------------------------------------------------------------
    // Evaluation-time rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn the_wrong_argument_count_is_rejected() {
        let mut function = ProgramFunction::new();
        function.add_parameter(f64_1d(1));
        function.add_parameter(f64_1d(1));

        let Err(err) = function.eval(&[Tensor::from([1.0_f64])]) else {
            panic!("one argument for two parameters is rejected")
        };
        assert!(matches!(
            err,
            FunctionEvalError::ArgumentArity {
                expected: 2,
                actual: 1
            }
        ));
    }

    #[test]
    fn an_argument_of_the_wrong_type_is_rejected_naming_both_types() {
        let mut function = ProgramFunction::new();
        function.add_parameter(f64_1d(3));

        let Err(err) = function.eval(&[Tensor::from([1.0_f64, 2.0])]) else {
            panic!("an argument of the wrong shape is rejected")
        };
        assert!(
            matches!(
                err,
                FunctionEvalError::ArgumentTypeMismatch {
                    parameter: 0,
                    expected,
                    actual,
                } if expected == f64_1d(3) && actual == f64_1d(2)
            ),
            "both the declared and the supplied type are named"
        );
    }

    #[test]
    fn a_bounded_parameter_admits_any_argument_within_its_bound() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Bounded { max: 4 }],
        });
        let doubled = function.add_op(Elsewhere, &[x]).unwrap()[0];
        function.add_result(doubled).unwrap();

        // `Elsewhere` has no built-in implementation, so evaluation stops at the locality check. That
        // comes after the argument check under test here.
        for len in 0..=4 {
            let arg = Tensor::from(vec![1.0_f64; len].as_slice());
            assert!(
                matches!(
                    function.eval(&[arg]),
                    Err(FunctionEvalError::NoBuiltinEval { .. })
                ),
                "an argument of length {len} is within the bound of 4"
            );
        }

        let Err(err) = function.eval(&[Tensor::from([1.0_f64; 5])]) else {
            panic!("an argument past the bound is rejected")
        };
        assert!(matches!(
            err,
            FunctionEvalError::ArgumentTypeMismatch {
                parameter: 0,
                expected,
                actual,
            } if expected.shape == [Dim::Bounded { max: 4 }] && actual == f64_1d(5)
        ));
    }

    // ---------------------------------------------------------------------------
    // Locality
    // ---------------------------------------------------------------------------

    /// An op defined outside the crate, in its own namespace, with no in-process implementation.
    #[derive(Clone)]
    struct Elsewhere;

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
            false
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

    #[test]
    fn a_function_of_built_in_instructions_is_locally_evaluable() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        assert!(
            function.has_builtin_eval(),
            "an empty function has nothing that cannot run"
        );
        function.add_op(Add, &[x, x]).unwrap();
        assert!(function.has_builtin_eval());
    }

    #[test]
    fn one_instruction_without_a_built_in_implementation_makes_the_function_not_locally_evaluable()
    {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let out = function.add_op(Elsewhere, &[x]).unwrap()[0];
        function.add_result(out).unwrap();

        assert!(!function.has_builtin_eval());

        let Err(err) = function.eval(&[Tensor::from([1.0_f64])]) else {
            panic!("a function with no built-in implementation cannot be evaluated")
        };
        let FunctionEvalError::NoBuiltinEval {
            instruction,
            full_name,
        } = &err
        else {
            panic!("expected a locality error, got {err}")
        };
        assert_eq!(
            *instruction,
            out.instruction(),
            "the offending instruction is named"
        );
        assert_eq!(full_name, "vendor.elsewhere");
    }

    /// An op counting how often it has been evaluated.
    #[derive(Clone)]
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

    #[test]
    fn a_function_that_needs_a_backend_evaluates_nothing_at_all() {
        let evaluations = Arc::new(AtomicUsize::new(0));
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let counted = function
            .add_op(Tally(Arc::clone(&evaluations)), &[x])
            .unwrap()[0];
        let out = function.add_op(Elsewhere, &[counted]).unwrap()[0];
        function.add_result(out).unwrap();

        assert!(matches!(
            function.eval(&[Tensor::from([1.0_f64])]),
            Err(FunctionEvalError::NoBuiltinEval { .. })
        ));
        assert_eq!(
            evaluations.load(Ordering::Relaxed),
            0,
            "the instruction ahead of the one needing a backend never ran"
        );
    }

    #[test]
    fn an_instruction_defined_outside_the_crate_is_type_checked_like_any_other() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(4));
        let out = function.add_op(Elsewhere, &[x]).unwrap()[0];

        assert_eq!(function.type_of(out), Some(&f64_1d(4)));

        let Err(err) = function.add_op(Elsewhere, &[x, x]) else {
            panic!("a two-operand call to a unary instruction is rejected")
        };
        assert!(matches!(
            err,
            FunctionError::OperandArity {
                full_name,
                expected: 1,
                actual: 2,
            } if full_name == "vendor.elsewhere"
        ));
    }
}
