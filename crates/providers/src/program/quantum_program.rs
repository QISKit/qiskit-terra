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

//! A callable collection of functions.

use std::fmt;

use thiserror::Error;

use super::program_function::{
    FunctionEvalError, InstructionId, InstructionRef, InstructionRole, InstructionView,
    ProgramFunction, Signature,
};
use crate::data_tree::DataTree;
use crate::tensor::{Tensor, TensorType};

/// The identity of one function within a [`QuantumProgram`].
///
/// An index is meaningful only against the program that issued it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FunctionId(u32);

impl FunctionId {
    /// Return the id of the function at `index` in definition order.
    pub fn from_index(index: usize) -> Self {
        Self(u32::try_from(index).expect("a function id fits in a u32"))
    }

    /// Return the underlying index, for use as a dense array subscript.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for FunctionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "@{}", self.0)
    }
}

/// Why a set of functions and a pair of structures do not make a well-formed [`QuantumProgram`].
#[derive(Debug, Error)]
pub enum ProgramError {
    /// There is no function to enter at.
    #[error("there are no functions to enter at")]
    NoFunctions,

    /// The input structure describes a different number of values than the entry point takes.
    #[error(
        "the input structure describes {leaves} value(s) but the entry point takes {parameters} \
         parameter(s)"
    )]
    InputArity { leaves: usize, parameters: usize },

    /// The output structure describes a different number of values than the entry point produces.
    #[error(
        "the output structure describes {leaves} value(s) but the entry point produces {results} \
         result(s)"
    )]
    OutputArity { leaves: usize, results: usize },

    /// A call names the function holding it, or one defined after it.
    #[error("{function} instruction {instruction} calls {callee}, which is not defined before it")]
    CallOrder {
        function: FunctionId,
        instruction: InstructionId,
        callee: FunctionId,
    },

    /// A call supplies a different number of operands than its callee takes parameters.
    #[error(
        "{function} instruction {instruction} calls {callee}: it takes {parameters} parameter(s), \
         the call supplies {operands}"
    )]
    CallParameterCount {
        function: FunctionId,
        instruction: InstructionId,
        callee: FunctionId,
        parameters: usize,
        operands: usize,
    },

    /// An operand of a call does not satisfy the corresponding parameter type of its callee.
    #[error(
        "{function} instruction {instruction} calls {callee}: its parameter {slot} is {parameter}, \
         the call supplies {operand}"
    )]
    CallParameterType {
        function: FunctionId,
        instruction: InstructionId,
        callee: FunctionId,
        slot: usize,
        parameter: TensorType,
        operand: TensorType,
    },

    /// A call declares a different number of results than its callee produces.
    #[error(
        "{function} instruction {instruction} calls {callee}: it produces {results} result(s), the \
         call declares {outputs}"
    )]
    CallResultCount {
        function: FunctionId,
        instruction: InstructionId,
        callee: FunctionId,
        results: usize,
        outputs: usize,
    },

    /// A result a call declares does not admit the corresponding result type of its callee.
    #[error(
        "{function} instruction {instruction} calls {callee}: its result {slot} is {result}, the \
         call declares {output}"
    )]
    CallResultType {
        function: FunctionId,
        instruction: InstructionId,
        callee: FunctionId,
        slot: usize,
        result: TensorType,
        output: TensorType,
    },
}

/// Why [`QuantumProgram::eval`] could not produce results.
#[derive(Debug, Error)]
pub enum ProgramEvalError {
    /// The inputs are arranged differently than the program declares.
    #[error("inputs are structured {actual} but the program declares {expected}")]
    InputStructureMismatch {
        expected: Box<DataTree<()>>,
        actual: Box<DataTree<()>>,
    },

    /// A function contains an instruction Qiskit has no in-process implementation of.
    #[error("{function} instruction {instruction} ({full_name}) has no built-in implementation")]
    NoBuiltinEval {
        function: FunctionId,
        instruction: InstructionId,
        full_name: String,
    },

    /// The entry point failed.
    #[error(transparent)]
    Function(#[from] FunctionEvalError),
}

/// A program describing a quantum computation.
///
/// A collection of [`ProgramFunction`]s, each of which is a list of instructions with semantics defined by
/// [`ProgramOp`] implementations.
///
/// Input is specified as a [`DataTree`] of tensor inputs arranged as [`input_types`](Self::input_types),
/// and returns tensors arranged as [`output_types`](Self::output_types).
///
/// The last function is the entry point by convention.
///
/// # Example
/// ```rust
/// use qiskit_providers::ops::Add;
/// use qiskit_providers::tensor::{DType, Dim, Tensor, TensorType};
/// use qiskit_providers::{DataTree, FunctionId, ProgramFunction, QuantumProgram};
///
/// let ty = TensorType { dtype: DType::F64, shape: vec![Dim::Fixed(1)] };
/// let mut function = ProgramFunction::new();
/// let x = function.add_parameter(ty.clone());
/// let y = function.add_parameter(ty.clone());
/// let sum = function.add_op(Add, &[x, y])?[0];
/// function.add_result(sum)?;
///
/// let leaf = || DataTree::Leaf(());
/// let program = QuantumProgram::new(
///     vec![function],
///     DataTree::mapping([("x", leaf()), ("y", leaf())])?,
///     DataTree::mapping([("sum", leaf())])?,
/// )?;
/// assert_eq!(
///     program.output_types(),
///     DataTree::mapping([("sum", DataTree::Leaf(ty))])?,
/// );
///
/// let inputs = DataTree::mapping([
///     ("x", DataTree::Leaf(Tensor::from([1.5_f64]))),
///     ("y", DataTree::Leaf(Tensor::from([2.5_f64]))),
/// ])?;
/// let outputs = program.eval(inputs)?;
/// assert_eq!(
///     outputs.get_by_str_key("sum"),
///     Some(&DataTree::Leaf(Tensor::from([4.0_f64])))
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone)]
pub struct QuantumProgram {
    /// The functions, in definition order, indexed by [`FunctionId`]. The last is the entry point.
    functions: Vec<ProgramFunction>,
    input_structure: DataTree<()>,
    output_structure: DataTree<()>,
}

impl QuantumProgram {
    /// Assemble `functions` into a program whose inputs and outputs are arranged as
    /// `input_structure` and `output_structure`.
    ///
    /// The last of `functions` is the entry point, and the structures describe its slots. A
    /// structure's leaves correspond to them in DFS order.
    ///
    /// Function calls are checked for type compatibility.
    pub fn new(
        functions: Vec<ProgramFunction>,
        input_structure: DataTree<()>,
        output_structure: DataTree<()>,
    ) -> Result<Self, ProgramError> {
        let Some(function) = functions.last() else {
            return Err(ProgramError::NoFunctions);
        };

        let leaves = input_structure.leaf_count();
        let parameters = function.parameters().len();
        if leaves != parameters {
            return Err(ProgramError::InputArity { leaves, parameters });
        }
        let leaves = output_structure.leaf_count();
        let results = function.results().len();
        if leaves != results {
            return Err(ProgramError::OutputArity { leaves, results });
        }

        check_calls(&functions)?;

        Ok(Self {
            functions,
            input_structure,
            output_structure,
        })
    }

    /// Return the functions, in definition order.
    pub fn functions(&self) -> &[ProgramFunction] {
        &self.functions
    }

    /// Return the function `id` names, or `None` if it does not belong to this program.
    pub fn function(&self, id: FunctionId) -> Option<&ProgramFunction> {
        self.functions.get(id.index())
    }

    /// Return the entry point, which is the last function.
    pub fn entry(&self) -> FunctionId {
        FunctionId::from_index(self.functions.len() - 1)
    }

    /// Return the entry point's function, which is the one a caller of this program invokes.
    pub fn entry_function(&self) -> &ProgramFunction {
        self.functions
            .last()
            .expect("a program holds at least one function")
    }

    /// Return how the program's inputs are arranged and named.
    pub fn input_structure(&self) -> &DataTree<()> {
        &self.input_structure
    }

    /// Return how the program's outputs are arranged and named.
    pub fn output_structure(&self) -> &DataTree<()> {
        &self.output_structure
    }

    /// Return the declared type of every input.
    pub fn input_types(&self) -> DataTree<TensorType> {
        arrange(
            &self.input_structure,
            self.entry_function().signature().inputs,
        )
    }

    /// Return the declared type of every output.
    pub fn output_types(&self) -> DataTree<TensorType> {
        arrange(
            &self.output_structure,
            self.entry_function().signature().outputs,
        )
    }

    /// Return whether every instruction in every function has a built-in evaluation; whether
    /// [`eval`](Self::eval) can run.
    pub fn has_builtin_eval(&self) -> bool {
        self.first_without_builtin_eval().is_none()
    }

    /// Evaluate the program on a tree of inputs, returning a tree of outputs.
    ///
    /// `inputs` must be arranged as [`input_structure`](Self::input_structure), checked
    /// before evaluation. The results are arranged as [`output_structure`](Self::output_structure).
    pub fn eval(&self, inputs: DataTree<Tensor>) -> Result<DataTree<Tensor>, ProgramEvalError> {
        let actual = inputs.structure();
        if actual != self.input_structure {
            return Err(ProgramEvalError::InputStructureMismatch {
                expected: Box::new(self.input_structure.clone()),
                actual: Box::new(actual),
            });
        }
        // Every function is checked before anything runs, so a program that needs a backend
        // produces no intermediates.
        if let Some((function, instruction)) = self.first_without_builtin_eval() {
            return Err(ProgramEvalError::NoBuiltinEval {
                function,
                instruction: instruction.id(),
                full_name: instruction.full_name(),
            });
        }
        let arguments: Vec<Tensor> = inputs.into_leaves().collect();
        let results = self.entry_function().eval_in(&arguments, &self.functions)?;
        Ok(arrange(&self.output_structure, results))
    }

    /// Return the first instruction of any function that has no built-in evaluation.
    fn first_without_builtin_eval(&self) -> Option<(FunctionId, InstructionRef<'_>)> {
        self.functions
            .iter()
            .enumerate()
            .find_map(|(index, function)| {
                function
                    .iter_instructions()
                    .find(|instruction| {
                        instruction.role() != InstructionRole::Call
                            && !instruction.has_builtin_eval()
                    })
                    .map(|instruction| (FunctionId::from_index(index), instruction))
            })
    }
}

impl std::fmt::Debug for QuantumProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            return write!(f, "{}", crate::render::listing(self));
        }
        f.debug_struct("QuantumProgram")
            .field("num_functions", &self.functions.len())
            .field("inputs", &self.input_types())
            .field("outputs", &self.output_types())
            .finish_non_exhaustive()
    }
}

/// Arrange one value per slot into `structure`, which describes exactly that many slots.
fn arrange<T>(structure: &DataTree<()>, values: Vec<T>) -> DataTree<T> {
    structure
        .unflatten(values)
        .expect("a structure describes as many slots as the entry point it was checked against")
}

/// Verify every call instruction of every function against the function it names.
///
/// Functions the entry point cannot reach are checked too, so evaluation can resolve every call
/// with no error path of its own.
fn check_calls(functions: &[ProgramFunction]) -> Result<(), ProgramError> {
    for (index, function) in functions.iter().enumerate() {
        let caller = FunctionId::from_index(index);
        for instruction in function.iter_instructions() {
            let InstructionView::Call(callee) = instruction.view() else {
                continue;
            };
            // A call may only name an earlier function. The call graph is acyclic by construction,
            // and the callee is in bounds without a separate check.
            if callee.index() >= index {
                return Err(ProgramError::CallOrder {
                    function: caller,
                    instruction: instruction.id(),
                    callee,
                });
            }
            check_call(
                caller,
                instruction,
                callee,
                &functions[callee.index()].signature(),
            )?;
        }
    }
    Ok(())
}

/// Verify one call instruction's operand and result types against `signature`, the contract of the
/// function it names.
fn check_call(
    function: FunctionId,
    instruction: InstructionRef<'_>,
    callee: FunctionId,
    signature: &Signature,
) -> Result<(), ProgramError> {
    if instruction.operands().len() != signature.inputs.len() {
        return Err(ProgramError::CallParameterCount {
            function,
            instruction: instruction.id(),
            callee,
            parameters: signature.inputs.len(),
            operands: instruction.operands().len(),
        });
    }
    for (slot, (parameter, operand)) in signature
        .inputs
        .iter()
        .zip(instruction.operand_types())
        .enumerate()
    {
        if !parameter.admits(operand) {
            return Err(ProgramError::CallParameterType {
                function,
                instruction: instruction.id(),
                callee,
                slot,
                parameter: parameter.clone(),
                operand: operand.clone(),
            });
        }
    }

    if instruction.output_types().len() != signature.outputs.len() {
        return Err(ProgramError::CallResultCount {
            function,
            instruction: instruction.id(),
            callee,
            results: signature.outputs.len(),
            outputs: instruction.output_types().len(),
        });
    }
    // The instructions reading a result were type-checked against the type the call declares, so
    // that type has to admit what the callee produces.
    for (slot, (result, output)) in signature
        .outputs
        .iter()
        .zip(instruction.output_types())
        .enumerate()
    {
        if !output.admits(result) {
            return Err(ProgramError::CallResultType {
                function,
                instruction: instruction.id(),
                callee,
                slot,
                result: result.clone(),
                output: output.clone(),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::ops::{Add, ProgramOp};
    use crate::program::Signature;
    use crate::tensor::{DType, Dim};

    /// Return the type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// Return a one-element `F64` tensor.
    fn one_element(value: f64) -> Tensor {
        Tensor::from([value])
    }

    /// Build `f(x, y) = x + y` over one-element `F64` tensors: two parameters, one result.
    fn add_function() -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let y = function.add_parameter(f64_1d(1));
        let sum = function.add_op(Add, &[x, y]).unwrap()[0];
        function.add_result(sum).unwrap();
        function
    }

    /// Build `f(x) = x + x` over one-element `F64` tensors: one parameter, one result.
    fn double_function() -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let doubled = function.add_op(Add, &[x, x]).unwrap()[0];
        function.add_result(doubled).unwrap();
        function
    }

    /// Return names for [`add_function`]'s two parameters: `[x: _, y: _]`.
    fn named_inputs() -> DataTree<()> {
        DataTree::mapping([("x", DataTree::Leaf(())), ("y", DataTree::Leaf(()))]).unwrap()
    }

    /// Return the same two inputs with the second one declared nested: `[x: _, y: [_]]`.
    fn nested_inputs() -> DataTree<()> {
        DataTree::mapping([
            ("x", DataTree::Leaf(())),
            ("y", DataTree::sequence([DataTree::Leaf(())])),
        ])
        .unwrap()
    }

    /// Return a name for [`add_function`]'s one result: `[sum: _]`.
    fn named_output() -> DataTree<()> {
        DataTree::mapping([("sum", DataTree::Leaf(()))]).unwrap()
    }

    /// Wrap `function` as a program whose slots are unnamed: one sequence of leaves per side.
    fn positional_program(function: ProgramFunction) -> QuantumProgram {
        let positional = |count| DataTree::sequence(std::iter::repeat_n(DataTree::Leaf(()), count));
        let inputs = positional(function.parameters().len());
        let outputs = positional(function.results().len());
        QuantumProgram::new(vec![function], inputs, outputs).unwrap()
    }

    /// Wrap [`add_function`] as a program, with the structures above.
    fn add_program() -> QuantumProgram {
        QuantumProgram::new(vec![add_function()], named_inputs(), named_output()).unwrap()
    }

    // ---------------------------------------------------------------------------
    // Evaluating through a structure
    // ---------------------------------------------------------------------------

    #[test]
    fn a_program_takes_a_tree_of_inputs_and_returns_a_tree_of_outputs() {
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(one_element(2.5))),
        ])
        .unwrap();

        assert_eq!(
            add_program().eval(inputs).unwrap(),
            DataTree::mapping([("sum", DataTree::Leaf(one_element(4.0)))]).unwrap(),
            "the outputs arrive in the structure the program declares"
        );
    }

    #[test]
    fn a_structures_root_may_be_anything() {
        // A bare leaf on each side describes one parameter and one result.
        let program = QuantumProgram::new(
            vec![double_function()],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        )
        .unwrap();
        assert_eq!(
            program.eval(DataTree::Leaf(one_element(3.0))).unwrap(),
            DataTree::Leaf(one_element(6.0))
        );

        // An empty branch describes no slots at all.
        let program = QuantumProgram::new(
            vec![ProgramFunction::new()],
            DataTree::new(),
            DataTree::new(),
        )
        .unwrap();
        assert_eq!(program.eval(DataTree::new()).unwrap(), DataTree::new());
    }

    #[test]
    fn a_program_whose_structures_are_unnamed_is_positional_on_both_sides() {
        let program = positional_program(add_function());

        assert_eq!(program.input_structure().to_string(), "[_, _]");
        assert_eq!(program.output_structure().to_string(), "[_]");
        assert_eq!(
            program
                .eval(DataTree::sequence([
                    DataTree::Leaf(one_element(1.0)),
                    DataTree::Leaf(one_element(10.0)),
                ]))
                .unwrap(),
            DataTree::sequence([DataTree::Leaf(one_element(11.0))])
        );
    }

    // ---------------------------------------------------------------------------
    // Naming lives in the program's structures
    // ---------------------------------------------------------------------------

    #[test]
    fn a_programs_structures_describe_its_entry_point() {
        // Two functions of different arities. The last is the entry point, and the structures
        // describe that one.
        let program = QuantumProgram::new(
            vec![double_function(), add_function()],
            named_inputs(),
            named_output(),
        )
        .unwrap();

        assert_eq!(program.functions().len(), 2);
        assert_eq!(program.entry(), FunctionId::from_index(1));
        assert_eq!(program.input_structure().to_string(), "[x: _, y: _]");
        assert_eq!(program.output_structure().to_string(), "[sum: _]");
        assert_eq!(
            program.entry_function().signature().inputs.len(),
            2,
            "the entry point is the second function, not the first"
        );
        assert_eq!(
            program
                .function(FunctionId::from_index(0))
                .unwrap()
                .signature(),
            double_function().signature(),
            "a function that is not the entry point is reachable in its own right"
        );

        // The same two structures against the same two functions in the other order describe too
        // many slots, which shows which of them they are checked against.
        let Err(err) = QuantumProgram::new(
            vec![add_function(), double_function()],
            named_inputs(),
            named_output(),
        ) else {
            panic!("two leaves cannot describe the one parameter `double_function` takes")
        };
        assert!(matches!(
            err,
            ProgramError::InputArity {
                leaves: 2,
                parameters: 1
            }
        ));
    }

    #[test]
    fn a_program_can_be_sent_between_threads() {
        // A job holds the program it is driving, and a job is where all concurrency lives.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<QuantumProgram>();
    }

    #[test]
    fn a_cloned_program_evaluates_the_same_way() {
        // Cloning a program clones the boxed op of every instruction.
        let program = add_program();
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(one_element(2.5))),
        ])
        .unwrap();

        assert_eq!(
            program.clone().eval(inputs.clone()).unwrap(),
            program.eval(inputs).unwrap()
        );
    }

    // ---------------------------------------------------------------------------
    // Reading a program back
    // ---------------------------------------------------------------------------

    #[test]
    fn debug_is_one_line_of_summary() {
        let debug = format!("{:?}", add_program());
        assert!(!debug.contains('\n'), "got {debug}");
        assert!(
            debug.starts_with("QuantumProgram { num_functions: 1,"),
            "got {debug}"
        );
    }

    #[test]
    fn the_alternate_debug_form_is_the_listing() {
        let program = add_program();
        assert_eq!(format!("{program:#?}"), crate::render::listing(&program));
    }

    // ---------------------------------------------------------------------------
    // Types and addresses, without evaluating
    // ---------------------------------------------------------------------------

    #[test]
    fn the_type_of_every_input_and_output_is_available_without_evaluating() {
        let program = add_program();

        assert_eq!(
            program.entry_function().signature(),
            Signature {
                inputs: vec![f64_1d(1), f64_1d(1)],
                outputs: vec![f64_1d(1)],
            }
        );
        assert_eq!(
            program.input_types(),
            DataTree::mapping([
                ("x", DataTree::Leaf(f64_1d(1))),
                ("y", DataTree::Leaf(f64_1d(1))),
            ])
            .unwrap(),
            "the declared types arrive arranged in the input structure"
        );
        assert_eq!(
            program.output_types(),
            DataTree::mapping([("sum", DataTree::Leaf(f64_1d(1)))]).unwrap()
        );
    }

    #[test]
    fn an_output_is_addressable_by_a_dotted_path_derived_from_the_structure() {
        // Three results over one parameter: the doubled value, the parameter, and the doubled value
        // again.
        let mut function = double_function();
        let doubled = function.result_values().next().unwrap();
        let x = function.parameter_values().next().unwrap();
        function.add_result(x).unwrap();
        function.add_result(doubled).unwrap();

        // `[doubled: [_, _], original: _]`, so the first two results share a branch.
        let outputs = DataTree::mapping([
            (
                "doubled",
                DataTree::sequence([DataTree::Leaf(()), DataTree::Leaf(())]),
            ),
            ("original", DataTree::Leaf(())),
        ])
        .unwrap();
        let program = QuantumProgram::new(vec![function], DataTree::Leaf(()), outputs).unwrap();

        let paths = program.output_structure().dotted_paths();
        assert_eq!(paths, ["doubled.0", "doubled.1", "original"]);

        // A path addresses a leaf of the structure, and so of any result arranged in it.
        let results = program.eval(DataTree::Leaf(one_element(2.0))).unwrap();
        for (path, expected) in paths.iter().zip([4.0, 2.0, 4.0]) {
            assert_eq!(
                results.get_by_str_key(path),
                Some(&DataTree::Leaf(one_element(expected))),
                "at {path}"
            );
        }
    }

    // ---------------------------------------------------------------------------
    // Several functions, one calling another
    // ---------------------------------------------------------------------------

    /// Build a function that calls `callee` on its one parameter and returns what comes back, given
    /// the signature the callee has.
    fn calling_function(callee: FunctionId, signature: &Signature) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(signature.inputs[0].clone());
        let called = function.add_call(callee, signature, &[x]).unwrap()[0];
        function.add_result(called).unwrap();
        function
    }

    #[test]
    fn a_call_computes_what_the_function_it_names_computes() {
        // @1 is `f(x) = double(x) + double(x)`, over the one body @0.
        let callee = double_function();
        let signature = callee.signature();
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let doubled = entry
            .add_call(FunctionId::from_index(0), &signature, &[x])
            .unwrap()[0];
        let sum = entry.add_op(Add, &[doubled, doubled]).unwrap()[0];
        entry.add_result(sum).unwrap();

        let program =
            QuantumProgram::new(vec![callee, entry], DataTree::Leaf(()), DataTree::Leaf(()))
                .unwrap();
        assert!(program.has_builtin_eval());

        // The same computation with the body written out where it is called.
        let mut inlined = ProgramFunction::new();
        let x = inlined.add_parameter(f64_1d(1));
        let doubled = inlined.add_op(Add, &[x, x]).unwrap()[0];
        let sum = inlined.add_op(Add, &[doubled, doubled]).unwrap()[0];
        inlined.add_result(sum).unwrap();
        let inlined =
            QuantumProgram::new(vec![inlined], DataTree::Leaf(()), DataTree::Leaf(())).unwrap();

        let input = DataTree::Leaf(one_element(3.0));
        assert_eq!(
            program.eval(input.clone()).unwrap(),
            DataTree::Leaf(one_element(12.0))
        );
        assert_eq!(
            program.eval(input.clone()).unwrap(),
            inlined.eval(input).unwrap()
        );
    }

    #[test]
    fn a_call_may_be_reached_through_a_chain_of_calls() {
        // @0 doubles, @1 calls @0, and @2 calls @1, so 1.5 comes back as 3.0.
        let doubling = double_function();
        let signature = doubling.signature();
        let program = QuantumProgram::new(
            vec![
                doubling,
                calling_function(FunctionId::from_index(0), &signature),
                calling_function(FunctionId::from_index(1), &signature),
            ],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        )
        .unwrap();

        assert_eq!(
            program.eval(DataTree::Leaf(one_element(1.5))).unwrap(),
            DataTree::Leaf(one_element(3.0))
        );
    }

    #[test]
    fn a_function_the_entry_point_cannot_reach_is_accepted() {
        // @0 is dead code, and fails if it is ever evaluated, so the program evaluating at all is
        // what shows the entry point @1 never reaches it.
        let mut unreachable = ProgramFunction::new();
        let x = unreachable.add_parameter(f64_1d(1));
        let out = unreachable
            .add_op(Elsewhere { builtin: true }, &[x])
            .unwrap()[0];
        unreachable.add_result(out).unwrap();

        let program = QuantumProgram::new(
            vec![unreachable, add_function()],
            named_inputs(),
            named_output(),
        )
        .unwrap();

        assert!(program.has_builtin_eval());
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.0))),
            ("y", DataTree::Leaf(one_element(10.0))),
        ])
        .unwrap();
        assert_eq!(
            program.eval(inputs).unwrap(),
            DataTree::mapping([("sum", DataTree::Leaf(one_element(11.0)))]).unwrap()
        );
    }

    // ---------------------------------------------------------------------------
    // Assembly rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn an_input_structure_describing_the_wrong_number_of_slots_is_rejected() {
        let Err(err) =
            QuantumProgram::new(vec![add_function()], DataTree::Leaf(()), named_output())
        else {
            panic!("one leaf cannot describe two parameters")
        };

        assert!(
            matches!(
                err,
                ProgramError::InputArity {
                    leaves: 1,
                    parameters: 2
                }
            ),
            "both counts are named"
        );
    }

    #[test]
    fn an_output_structure_describing_the_wrong_number_of_slots_is_rejected() {
        let Err(err) = QuantumProgram::new(vec![add_function()], named_inputs(), named_inputs())
        else {
            panic!("two leaves cannot describe one result")
        };

        assert!(matches!(
            err,
            ProgramError::OutputArity {
                leaves: 2,
                results: 1
            }
        ));
    }

    #[test]
    fn a_program_holding_no_functions_is_rejected() {
        // The entry point is the last function, so there has to be one.
        let Err(err) = QuantumProgram::new(Vec::new(), DataTree::new(), DataTree::new()) else {
            panic!("a program with no functions has nothing to enter at")
        };

        assert!(matches!(err, ProgramError::NoFunctions));
    }

    #[test]
    fn a_call_naming_itself_or_a_later_function_is_rejected() {
        let signature = double_function().signature();

        // @0 calls @0. A call may only name an earlier function, so the call graph cannot have a
        // cycle in it and nothing has to look for one.
        let Err(err) = QuantumProgram::new(
            vec![calling_function(FunctionId::from_index(0), &signature)],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("a function cannot call itself")
        };
        assert!(matches!(
            err,
            ProgramError::CallOrder {
                function,
                callee,
                ..
            } if function == FunctionId::from_index(0) && callee == FunctionId::from_index(0)
        ));

        // @0 calls @1, which is defined after it.
        let Err(err) = QuantumProgram::new(
            vec![
                calling_function(FunctionId::from_index(1), &signature),
                double_function(),
            ],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("a call may not name a later function")
        };
        assert!(matches!(
            err,
            ProgramError::CallOrder {
                function,
                callee,
                ..
            } if function == FunctionId::from_index(0) && callee == FunctionId::from_index(1)
        ));
    }

    #[test]
    fn a_call_whose_operands_disagree_with_its_callee_is_rejected() {
        // The call is built against a signature describing something other than @0, which takes one
        // F64[1]. Nothing is checked against @0 until the program is assembled.
        let two_parameters = Signature {
            inputs: vec![f64_1d(1), f64_1d(1)],
            outputs: vec![f64_1d(1)],
        };
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        let called = entry
            .add_call(FunctionId::from_index(0), &two_parameters, &[x, x])
            .unwrap()[0];
        entry.add_result(called).unwrap();

        let Err(err) = QuantumProgram::new(
            vec![double_function(), entry],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("@0 takes one parameter, not two")
        };
        assert!(matches!(
            err,
            ProgramError::CallParameterCount {
                parameters: 1,
                operands: 2,
                ..
            }
        ));

        // One operand, of a shape @0 does not take.
        let wider = Signature {
            inputs: vec![f64_1d(2)],
            outputs: vec![f64_1d(1)],
        };
        let Err(err) = QuantumProgram::new(
            vec![
                double_function(),
                calling_function(FunctionId::from_index(0), &wider),
            ],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("@0 takes an F64[1], and the call supplies an F64[2]")
        };
        assert!(
            matches!(
                err,
                ProgramError::CallParameterType {
                    function,
                    callee,
                    slot: 0,
                    parameter,
                    operand,
                    ..
                } if function == FunctionId::from_index(1)
                    && callee == FunctionId::from_index(0)
                    && parameter == f64_1d(1)
                    && operand == f64_1d(2)
            ),
            "the calling function, the call instruction, the slot, and both types are named"
        );
    }

    #[test]
    fn a_call_whose_results_disagree_with_its_callee_is_rejected() {
        let no_results = Signature {
            inputs: vec![f64_1d(1)],
            outputs: vec![],
        };
        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(1));
        assert!(
            entry
                .add_call(FunctionId::from_index(0), &no_results, &[x])
                .unwrap()
                .is_empty()
        );
        entry.add_result(x).unwrap();

        let Err(err) = QuantumProgram::new(
            vec![double_function(), entry],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("@0 produces a result the call does not declare")
        };
        assert!(matches!(
            err,
            ProgramError::CallResultCount {
                results: 1,
                outputs: 0,
                ..
            }
        ));

        // One result, of a shape @0 does not produce.
        let wider = Signature {
            inputs: vec![f64_1d(1)],
            outputs: vec![f64_1d(2)],
        };
        let Err(err) = QuantumProgram::new(
            vec![
                double_function(),
                calling_function(FunctionId::from_index(0), &wider),
            ],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        ) else {
            panic!("@0 produces an F64[1], and the call declares an F64[2]")
        };
        assert!(matches!(
            err,
            ProgramError::CallResultType {
                slot: 0,
                result,
                output,
                ..
            } if result == f64_1d(1) && output == f64_1d(2)
        ));
    }

    // ---------------------------------------------------------------------------
    // Evaluation rejections
    // ---------------------------------------------------------------------------

    #[test]
    fn inputs_with_the_declared_leaf_count_but_the_wrong_structure_are_rejected() {
        // Two leaves either way, so only a structural check catches this.
        let inputs = DataTree::sequence([
            DataTree::Leaf(one_element(1.5)),
            DataTree::Leaf(one_element(2.5)),
        ]);

        let Err(err) = add_program().eval(inputs) else {
            panic!("a sequence cannot stand in for a mapping")
        };

        assert!(
            matches!(
                err,
                ProgramEvalError::InputStructureMismatch { expected, actual }
                    if *expected == named_inputs()
                        && *actual == DataTree::sequence([DataTree::Leaf(()), DataTree::Leaf(())])
            ),
            "each structure is reported whole"
        );
    }

    #[test]
    fn a_mismatch_below_the_root_is_reported_as_the_whole_structure() {
        // The second input is declared nested, so what diverges is one child of the root rather than
        // the root itself.
        let program =
            QuantumProgram::new(vec![add_function()], nested_inputs(), named_output()).unwrap();
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(one_element(2.5))),
        ])
        .unwrap();

        let Err(err) = program.eval(inputs) else {
            panic!("a leaf cannot stand in for a branch of one leaf")
        };

        assert!(
            matches!(
                err,
                ProgramEvalError::InputStructureMismatch { expected, actual }
                    if *expected == nested_inputs() && *actual == named_inputs()
            ),
            "the surrounding structure is reported, not the subtree that differs"
        );
    }

    #[test]
    fn an_input_that_does_not_match_its_declared_type_is_rejected_naming_both_types() {
        let inputs = DataTree::mapping([
            ("x", DataTree::Leaf(one_element(1.5))),
            ("y", DataTree::Leaf(Tensor::from([2_i64]))),
        ])
        .unwrap();

        let Err(err) = add_program().eval(inputs) else {
            panic!("a program is monomorphic, so an I64 input is rejected")
        };

        assert!(matches!(
            err,
            ProgramEvalError::Function(FunctionEvalError::ArgumentTypeMismatch {
                parameter: 1,
                expected,
                actual,
            }) if expected.dtype == DType::F64 && actual.dtype == DType::I64
        ));
    }

    /// An op defined outside the crate whose `eval` always fails.
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

    /// Build a one-parameter, one-result function holding a single [`Elsewhere`].
    fn vendor_function(builtin: bool) -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        let out = function.add_op(Elsewhere { builtin }, &[x]).unwrap()[0];
        function.add_result(out).unwrap();
        function
    }

    /// Build a program whose entry point @1 calls @0, which holds a single [`Elsewhere`].
    fn calls_a_vendor_function(builtin: bool) -> QuantumProgram {
        let callee = vendor_function(builtin);
        let signature = callee.signature();
        QuantumProgram::new(
            vec![
                callee,
                calling_function(FunctionId::from_index(0), &signature),
            ],
            DataTree::Leaf(()),
            DataTree::Leaf(()),
        )
        .unwrap()
    }

    #[test]
    fn a_program_whose_call_reaches_an_instruction_needing_a_backend_names_that_instruction() {
        let program = calls_a_vendor_function(false);
        assert!(
            !program.has_builtin_eval(),
            "the entry point reaches @0, which needs a backend"
        );

        let Err(err) = program.eval(DataTree::Leaf(one_element(1.0))) else {
            panic!("a program that needs a backend cannot be evaluated in process")
        };
        assert!(
            matches!(
                err,
                ProgramEvalError::NoBuiltinEval {
                    function,
                    full_name,
                    ..
                } if function == FunctionId::from_index(0) && full_name == "vendor.elsewhere"
            ),
            "the function as well as the instruction is named"
        );
    }

    #[test]
    fn a_function_needing_a_backend_counts_even_where_nothing_calls_it() {
        // The question is asked of every function a program holds, rather than only of those the
        // entry point reaches.
        let program = QuantumProgram::new(
            vec![vendor_function(false), add_function()],
            named_inputs(),
            named_output(),
        )
        .unwrap();

        assert!(!program.has_builtin_eval());
    }

    #[test]
    fn a_call_that_fails_names_the_function_it_reached() {
        let Err(err) = calls_a_vendor_function(true).eval(DataTree::Leaf(one_element(1.0))) else {
            panic!("@0 fails as it runs")
        };

        let ProgramEvalError::Function(FunctionEvalError::CallFailed { callee, source, .. }) = &err
        else {
            panic!("expected a failed call, got {err}")
        };
        assert_eq!(*callee, FunctionId::from_index(0));
        let FunctionEvalError::InstructionFailed {
            full_name, source, ..
        } = source.as_ref()
        else {
            panic!("the call's source is the instruction that failed")
        };
        assert_eq!(full_name, "vendor.elsewhere");
        assert!(
            source.downcast_ref::<NoImplementation>().is_some(),
            "the chain leads from the call to the instruction that failed"
        );
    }
}
