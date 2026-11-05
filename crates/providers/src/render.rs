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

//! Render a quantum program in various formats.

use std::fmt::{self, Write as _};

use crate::program::{
    InstructionId, InstructionRef, InstructionRole, InstructionView, ProgramFunction,
    QuantumProgram, Value,
};

/// Render a program as a listing with one function per block and one line per instruction.
///
/// This produces text like the following:
///
/// ```text
/// @0: // entry point
///   %0: F64[4, 2] = qiskit.parameter angles
///   %1: Bit[4, 100, 2] = qiskit.shot_loop[circuits=1, shots=100](%0)
///   %2: F64[4, 2] = qiskit.mean[axis=1](%1)
///   %4: F64[2] = qiskit.std[axis=0, ddof=0](%2)
///   results:
///     excited = %2
///     spread = %4
/// ```
pub fn listing(program: &QuantumProgram) -> String {
    Listing(program).to_string()
}

/// Render `program` as a Graphviz `dot` source.
pub fn dot(program: &QuantumProgram) -> String {
    Drawing(program).to_string()
}

/// A program as an SSA-style listing.
struct Listing<'a>(&'a QuantumProgram);

impl fmt::Display for Listing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, function) in self.0.functions().iter().enumerate() {
            if index > 0 {
                write!(f, "\n\n")?;
            }
            write!(f, "@{index}:")?;
            if index == self.0.entry().index() {
                // The entry point is always the last function, which is worth saying out loud.
                write!(f, " // entry point")?;
            }

            for (instruction, io_name) in named_io(self.0, index) {
                if instruction.output_types().is_empty() {
                    // instructions without outputs themself represent an output. these are handled
                    // after this loop.
                    continue;
                }

                // write one line for this instruction
                write!(f, "\n  %{}: ", instruction.id())?;
                if let [ty] = instruction.output_types() {
                    write!(f, "{ty}")?;
                } else {
                    // when there are multiple outputs, surround in tuple syntax
                    f.write_char('(')?;
                    write_separated(f, instruction.output_types())?;
                    f.write_char(')')?;
                }
                write!(f, " = {}", instruction.full_name())?;
                match instruction.view() {
                    InstructionView::Parameter | InstructionView::Result => {
                        if let Some(name) = io_name {
                            write!(f, " {name}")?;
                        }
                    }
                    InstructionView::Call(callee) => write!(f, " @{}", callee.index())?,
                    InstructionView::Op(op) => {
                        if let Some(payload) = op.describe() {
                            write!(f, "[{payload}]")?;
                        }
                    }
                }
                if !instruction.operands().is_empty() {
                    f.write_char('(')?;
                    write_separated(
                        f,
                        instruction
                            .operands()
                            .iter()
                            .map(|&value| ValueIn(function, value)),
                    )?;
                    f.write_char(')')?;
                }
            }

            // write the result section of this funciton
            if function.results().is_empty() {
                continue;
            }
            write!(f, "\n  results:")?;
            let names = named_io(self.0, index)
                .filter(|(instruction, _)| instruction.role() == InstructionRole::Result)
                .map(|(_, name)| name);
            for (name, value) in names.zip(function.result_values()) {
                let value = ValueIn(function, value);
                match name {
                    Some(name) => write!(f, "\n    {name} = {value}")?,
                    None => write!(f, "\n    {value}")?,
                }
            }
        }
        Ok(())
    }
}

/// A program as a dataflow Graphviz drawing.
struct Drawing<'a>(&'a QuantumProgram);

impl fmt::Display for Drawing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A lone function needs no frame around it.
        let clustered = self.0.functions().len() > 1;
        let indent = if clustered { "    " } else { "  " };
        write!(f, "digraph program {{")?;
        write!(f, "\n  node [fontname=\"Helvetica\", fontsize=10];")?;
        write!(f, "\n  edge [fontname=\"Helvetica\", fontsize=9];")?;
        for (index, function) in self.0.functions().iter().enumerate() {
            if clustered {
                write!(f, "\n  subgraph cluster_{index} {{")?;
                let label = if index == self.0.entry().index() {
                    format!("@{index} (entry point)")
                } else {
                    format!("@{index}")
                };
                write!(f, "\n    label={};", Quoted(&label))?;
            }
            for (instruction, name) in named_io(self.0, index) {
                let description = match instruction.view() {
                    InstructionView::Parameter | InstructionView::Result => name,
                    InstructionView::Call(callee) => Some(format!("@{}", callee.index())),
                    InstructionView::Op(op) => op.describe(),
                };
                let caption = match description {
                    Some(text) => format!("{}\n{text}", instruction.full_name()),
                    None => instruction.full_name(),
                };
                write!(
                    f,
                    "\n{indent}{} [{}, label={}];",
                    BoxId(index, instruction.id()),
                    attributes(instruction.role()),
                    Quoted(&caption),
                )?;
            }
            for instruction in function.iter_instructions() {
                for (position, (operand, ty)) in instruction
                    .operands()
                    .iter()
                    .zip(instruction.operand_types())
                    .enumerate()
                {
                    write!(
                        f,
                        "\n{indent}{} -> {} [label={}];",
                        BoxId(index, operand.instruction()),
                        BoxId(index, instruction.id()),
                        // The edge itself says which instruction the value came from, so the label
                        // gives only which of that instruction's slots it is, and which operand it
                        // fills.
                        Quoted(&format!(
                            "{PAD}{} \u{2192} {position}{PAD}\n{PAD}{ty}{PAD}",
                            operand.slot()
                        )),
                    )?;
                }
            }
            if clustered {
                write!(f, "\n  }}")?;
            }
        }
        write!(f, "\n}}")
    }
}

/// Return an iterator over names of parameter/result instructions of the indexed function in the
/// program.
///
/// One iterand is always yielded for every instruction in the function. If it's not a
/// parameter/result instruction, then `None` is given instead of the string name of that
/// parameter/result. Also, non-entry functions, or an entry function with a leaf-like output, also
/// don't have names to provide, so those also return `None`.
fn named_io(
    program: &QuantumProgram,
    index: usize,
) -> impl Iterator<Item = (InstructionRef<'_>, Option<String>)> {
    let function = &program.functions()[index];
    debug_assert!(
        function.parameters().is_sorted() && function.results().is_sorted(),
        "a boundary instruction is declared after every instruction already in the function"
    );
    let (inputs, outputs) = if index == program.entry().index() {
        (
            program.input_structure().dotted_paths(),
            program.output_structure().dotted_paths(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let mut inputs = inputs.into_iter();
    let mut outputs = outputs.into_iter();
    function.iter_instructions().map(move |instruction| {
        let path = match instruction.role() {
            InstructionRole::Parameter => inputs.next(),
            InstructionRole::Result => outputs.next(),
            InstructionRole::Op | InstructionRole::Call => None,
        };
        (instruction, path.filter(|path| !path.is_empty()))
    })
}

/// Write `items` in order, separated by commas.
fn write_separated(
    f: &mut fmt::Formatter<'_>,
    items: impl IntoIterator<Item = impl fmt::Display>,
) -> fmt::Result {
    for (position, item) in items.into_iter().enumerate() {
        if position > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{item}")?;
    }
    Ok(())
}

/// One value of `function`.
///
/// The convention is to refer to values whose producer instruction only has one
/// output as `%n`, but if there are multiple instructions, `%n.0`, `%n.1`, etc.
struct ValueIn<'a>(&'a ProgramFunction, Value);

impl fmt::Display for ValueIn<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let producer = self
            .0
            .instruction(self.1.instruction())
            .expect("a value of a function is produced in that function");
        if producer.output_types().len() > 1 {
            write!(f, "%{}.{}", self.1.instruction(), self.1.slot())
        } else {
            write!(f, "%{}", self.1.instruction())
        }
    }
}

/// Padding for graphviz edge labels, otherwise the text overlaps the arrow a bit. These are thin
/// spaces because an ordinary space would be trimmed.
const PAD: &str = "\u{2009}\u{2009}";

/// The graphviz `dot` identifier of one instruction's box. Must be unique across the whole program.
struct BoxId(usize, InstructionId);

impl fmt::Display for BoxId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "f{}n{}", self.0, self.1.index())
    }
}

/// A wrapper that displays a string as an escaped string literal.
///
/// E.g., `Quoted("a\nb")` formats as `"a\nb"`.
struct Quoted<'a>(&'a str);

impl fmt::Display for Quoted<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_char('"')?;
        for character in self.0.chars() {
            match character {
                '"' => f.write_str("\\\"")?,
                '\\' => f.write_str("\\\\")?,
                '\n' => f.write_str("\\n")?,
                _ => f.write_char(character)?,
            }
        }
        f.write_char('"')
    }
}

/// Return how an instruction of `role` is drawn.
fn attributes(role: InstructionRole) -> &'static str {
    match role {
        InstructionRole::Parameter => "shape=ellipse, style=filled, fillcolor=lightblue",
        InstructionRole::Op => "shape=box",
        InstructionRole::Call => "shape=box3d, style=filled, fillcolor=lightyellow",
        InstructionRole::Result => "shape=ellipse, style=filled, fillcolor=lightgrey",
    }
}

#[cfg(test)]
mod test {
    use qiskit_circuit::bit::ClassicalRegister;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::Param;

    use super::*;
    use crate::data_tree::DataTree;
    use crate::ops::{Add, Constant, Mean, ShotLoop};
    use crate::program::{FunctionId, ProgramFunction};
    use crate::tensor::{DType, Dim, Tensor, TensorType};

    /// Return the type of a 1-D `F64` tensor of `len` elements.
    fn f64_1d(len: usize) -> TensorType {
        TensorType {
            dtype: DType::F64,
            shape: vec![Dim::Fixed(len)],
        }
    }

    /// Return a structure of `names`, one named leaf each.
    fn named_branch(names: &[&str]) -> DataTree<()> {
        DataTree::mapping(names.iter().map(|&name| (name, DataTree::Leaf(()))))
            .expect("the test names are valid")
    }

    /// Return a structure of `count` unnamed leaves.
    fn positional_branch(count: usize) -> DataTree<()> {
        DataTree::sequence(std::iter::repeat_n(DataTree::Leaf(()), count))
    }

    /// Return a circuit with no parameters holding `registers` as `(name, width)` pairs.
    fn circuit(registers: &[(&str, u32)]) -> CircuitData {
        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).unwrap();
        for &(name, width) in registers {
            circuit
                .add_creg(ClassicalRegister::new_owning(name, width), true)
                .unwrap();
        }
        circuit
    }

    /// Build `f(x) = (mean(x), mean(x) + 1.0)`, whose mean is shared by both results.
    fn shared_mean() -> ProgramFunction {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(3));
        let mean = function.add_op(Mean::new(0), &[x]).unwrap()[0];
        let one = function
            .add_op(Constant::new(Tensor::from([1.0_f64])), &[])
            .unwrap()[0];
        let shifted = function.add_op(Add, &[mean, one]).unwrap()[0];
        function.add_result(mean).unwrap();
        function.add_result(shifted).unwrap();
        function
    }

    /// Return a shot loop over one circuit of two registers, reading a parameter-free operand.
    fn shot_loop() -> ProgramFunction {
        let loop_instruction = ShotLoop::new(vec![circuit(&[("c", 2), ("d", 3)])], 100).unwrap();
        let mut function = ProgramFunction::new();
        let values = function
            .add_op(Constant::new(Tensor::from([] as [f64; 0])), &[])
            .unwrap();
        let outcomes = function.add_op(loop_instruction, &values).unwrap();
        let mean = function.add_op(Mean::new(0), &[outcomes[1]]).unwrap()[0];
        function.add_result(mean).unwrap();
        function
    }

    /// Build `f(x) = x`, and a caller of it, as the two functions of one program.
    fn call() -> QuantumProgram {
        let mut callee = ProgramFunction::new();
        let x = callee.add_parameter(f64_1d(3));
        callee.add_result(x).unwrap();

        let mut entry = ProgramFunction::new();
        let x = entry.add_parameter(f64_1d(3));
        let called = entry
            .add_call(FunctionId::from_index(0), &callee.signature(), &[x])
            .unwrap()[0];
        entry.add_result(called).unwrap();

        QuantumProgram::new(
            vec![callee, entry],
            named_branch(&["x"]),
            named_branch(&["mean"]),
        )
        .unwrap()
    }

    #[test]
    fn a_listing_writes_each_instruction_once() {
        let program = QuantumProgram::new(
            vec![shared_mean()],
            named_branch(&["x"]),
            named_branch(&["mean", "shifted"]),
        )
        .unwrap();

        assert_eq!(
            listing(&program),
            "\
@0: // entry point
  %0: F64[3] = qiskit.parameter x
  %1: F64[] = qiskit.mean[axis=0](%0)
  %2: F64[1] = qiskit.constant
  %3: F64[1] = qiskit.add(%1, %2)
  results:
    mean = %1
    shifted = %3"
        );
    }

    #[test]
    fn a_listing_addresses_an_unnamed_slot_by_position() {
        let program = QuantumProgram::new(
            vec![shared_mean()],
            positional_branch(1),
            positional_branch(2),
        )
        .unwrap();

        assert_eq!(
            listing(&program),
            "\
@0: // entry point
  %0: F64[3] = qiskit.parameter 0
  %1: F64[] = qiskit.mean[axis=0](%0)
  %2: F64[1] = qiskit.constant
  %3: F64[1] = qiskit.add(%1, %2)
  results:
    0 = %1
    1 = %3"
        );
    }

    #[test]
    fn a_listing_of_a_lone_output_writes_its_value() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(2));
        function.add_result(x).unwrap();
        let program =
            QuantumProgram::new(vec![function], DataTree::Leaf(()), DataTree::Leaf(())).unwrap();

        assert_eq!(
            listing(&program),
            "\
@0: // entry point
  %0: F64[2] = qiskit.parameter
  results:
    %0"
        );
    }

    #[test]
    fn a_listing_names_a_value_of_a_multi_result_instruction() {
        let program = QuantumProgram::new(
            vec![shot_loop()],
            positional_branch(0),
            named_branch(&["mean"]),
        )
        .unwrap();

        assert_eq!(
            listing(&program),
            "\
@0: // entry point
  %0: F64[0] = qiskit.constant
  %1: (Bit[100, 2], Bit[100, 3]) = qiskit.shot_loop[circuits=1, shots=100](%0)
  %2: F64[3] = qiskit.mean[axis=0](%1.1)
  results:
    mean = %2"
        );
    }

    #[test]
    fn a_listing_gives_a_block_per_function_and_names_a_callee() {
        assert_eq!(
            listing(&call()),
            "\
@0:
  %0: F64[3] = qiskit.parameter
  results:
    %0

@1: // entry point
  %0: F64[3] = qiskit.parameter x
  %1: F64[3] = qiskit.call @0(%0)
  results:
    mean = %1"
        );
    }

    #[test]
    fn a_drawing_gives_a_box_per_instruction_and_an_edge_per_operand() {
        let program = QuantumProgram::new(
            vec![shared_mean()],
            named_branch(&["x"]),
            named_branch(&["mean", "shifted"]),
        )
        .unwrap();
        let drawing = dot(&program);

        assert_eq!(
            drawing.matches(" [shape=").count(),
            6,
            "one box per instruction, results included"
        );
        assert!(
            drawing.contains(
                "f0n0 [shape=ellipse, style=filled, fillcolor=lightblue, \
                 label=\"qiskit.parameter\\nx\"];"
            ),
            "a parameter shows its input name and does not look like an operation: {drawing}"
        );
        assert!(
            drawing.contains("f0n1 [shape=box, label=\"qiskit.mean\\naxis=0\"];"),
            "an operation shows the payload it describes itself by: {drawing}"
        );
        assert!(
            drawing.contains("f0n2 [shape=box, label=\"qiskit.constant\"];"),
            "an op describing no payload is captioned by its type name alone: {drawing}"
        );
        assert!(
            drawing.contains(
                "f0n4 [shape=ellipse, style=filled, fillcolor=lightgrey, \
                 label=\"qiskit.result\\nmean\"];"
            ),
            "a result shows its output name: {drawing}"
        );
        assert!(
            drawing.contains(&format!(
                "f0n0 -> f0n1 [label=\"{PAD}0 \u{2192} 0{PAD}\\n{PAD}F64[3]{PAD}\"];"
            )),
            "an edge names its slot, the position it fills and the type: {drawing}"
        );
        assert_eq!(
            drawing.matches("f0n1 -> ").count(),
            2,
            "the shared mean is one box with an edge to each instruction reading it: {drawing}"
        );
    }

    #[test]
    fn a_drawing_names_the_slot_of_a_multi_result_instruction() {
        let program = QuantumProgram::new(
            vec![shot_loop()],
            positional_branch(0),
            positional_branch(1),
        )
        .unwrap();
        let drawing = dot(&program);

        assert!(
            drawing.contains(&format!(
                "f0n1 -> f0n2 [label=\"{PAD}1 \u{2192} 0{PAD}\\n{PAD}Bit[100, 3]{PAD}\"];"
            )),
            "an edge from a multi-result instruction names the slot it came from: {drawing}"
        );
    }

    #[test]
    fn a_drawing_clusters_a_program_of_several_functions() {
        let one = QuantumProgram::new(
            vec![shared_mean()],
            positional_branch(1),
            positional_branch(2),
        )
        .unwrap();
        assert!(
            !dot(&one).contains("cluster"),
            "one function needs no frame around it"
        );

        let drawing = dot(&call());
        assert!(drawing.contains("subgraph cluster_0 {"), "{drawing}");
        assert!(drawing.contains("subgraph cluster_1 {"), "{drawing}");
        assert!(
            drawing.contains("label=\"@1 (entry point)\";"),
            "the frame around the entry point says so: {drawing}"
        );
        assert!(
            drawing.contains(
                "f1n1 [shape=box3d, style=filled, fillcolor=lightyellow, \
                 label=\"qiskit.call\\n@0\"];"
            ),
            "a call names the function it reaches and does not look like an operation: {drawing}"
        );
    }

    #[test]
    fn a_drawing_escapes_a_name_that_would_end_a_label() {
        let mut function = ProgramFunction::new();
        let x = function.add_parameter(f64_1d(1));
        function.add_result(x).unwrap();
        let mut inputs = DataTree::new();
        inputs.insert_leaf("say \"hi\"\\", ()).unwrap();
        let program = QuantumProgram::new(vec![function], inputs, positional_branch(1)).unwrap();

        let drawing = dot(&program);
        assert!(
            drawing.contains("label=\"qiskit.parameter\\nsay \\\"hi\\\"\\\\\""),
            "{drawing}"
        );
    }
}
