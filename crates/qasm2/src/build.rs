// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! Build a native [CircuitData] directly from the parser's bytecode stream, without going
//! through Python.

use std::fmt;
use std::sync::Arc;

use num_bigint::BigUint;

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister, Register};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::instruction::Parameters;
use qiskit_circuit::operations::{
    Condition, ControlFlow, ControlFlowInstruction, CustomOperation, Operation, Param,
    StandardInstruction,
};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::standard_gate::StandardGate;
use qiskit_circuit::{Clbit, Qubit};

use crate::bytecode::InternalBytecode;
use crate::error::ParseError;
use crate::expr::{Expr, evaluate};
use crate::ext::ClassicalEvaluator;
use crate::parse::{ClbitId, CregId, GateId, QELIB1, QubitId};

#[derive(Clone, PartialEq)]
enum GateEntry {
    Standard(StandardGate),
    Defined(Arc<DefinedGateTemplate>),
}

impl fmt::Debug for GateEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Standard(gate) => f.debug_tuple("Standard").field(gate).finish(),
            Self::Defined(template) => f.debug_tuple("Defined").field(&template.name).finish(),
        }
    }
}

impl GateEntry {
    fn num_qubits(&self) -> u32 {
        match self {
            GateEntry::Standard(gate) => gate.num_qubits(),
            GateEntry::Defined(template) => template.num_qubits,
        }
    }

    fn num_params(&self) -> u32 {
        match self {
            GateEntry::Standard(gate) => gate.num_params(),
            GateEntry::Defined(template) => template.num_params,
        }
    }
}

#[derive(Debug, PartialEq)]
enum BodyInstruction {
    Gate {
        entry: GateEntry,
        arguments: Vec<Expr>,
        qubits: Vec<QubitId>,
    },
    Barrier {
        qubits: Vec<QubitId>,
    },
}

/// Declared once per OQ2 `gate`/`opaque` statement and shared by every usage.  `body` is `None`
/// for `opaque`.  Its arguments stay as `Expr` because they can reference this gate's own
/// parameters (`gate rz(theta) q { u1(theta) q; }`), which only a usage can supply.
#[derive(Debug, PartialEq)]
struct DefinedGateTemplate {
    name: String,
    num_qubits: u32,
    num_params: u32,
    body: Option<Vec<BodyInstruction>>,
}

impl DefinedGateTemplate {
    /// The evaluator is detached because [CustomOperation::definition] hands us no interpreter
    /// token, and a C caller may have none to attach.
    ///
    /// [None] covers every failure, since [CustomOperation::definition] has no error channel, so
    /// an unevaluable body is indistinguishable from an `opaque` gate's missing one.  Python's
    /// `_DefinedGate` raises `QASM2ParseError` for the same inputs.
    fn build_definition(&self, params: &[Param]) -> Option<CircuitData> {
        let body = self.body.as_ref()?;
        let float_params: Vec<f64> = params
            .iter()
            .map(|p| match p {
                Param::Float(v) => Some(*v),
                _ => None,
            })
            .collect::<Option<_>>()?;

        let mut circuit = CircuitData::new(None, None, Param::Float(0.0)).ok()?;
        circuit.add_anonymous_qubits(self.num_qubits).ok()?;

        for instruction in body {
            match instruction {
                BodyInstruction::Gate {
                    entry,
                    arguments,
                    qubits,
                } => {
                    let arguments: Vec<f64> = arguments
                        .iter()
                        .map(|expr| {
                            evaluate(expr, &float_params, ClassicalEvaluator::detached()).ok()
                        })
                        .collect::<Option<_>>()?;
                    push_gate(&mut circuit, entry, &arguments, &to_qubits(qubits)).ok()?;
                }
                BodyInstruction::Barrier { qubits } => {
                    push_standard_instruction(
                        &mut circuit,
                        StandardInstruction::Barrier(qubits.len() as u32),
                        &to_qubits(qubits),
                        &[],
                    )
                    .ok()?;
                }
            }
        }

        Some(circuit)
    }
}

/// One usage of an OQ2-defined gate; Python's equivalent is `_gate_builder` in `parse.py`.
///
/// Deliberately uncached: `CircuitData::assign_parameters_inner` rebinds params through
/// `PackedInstruction::params_mut` without touching the operation, so there would be no hook to
/// invalidate a cached definition.
#[derive(Clone, Debug, PartialEq)]
struct DefinedGate {
    template: Arc<DefinedGateTemplate>,
}

impl Operation for DefinedGate {
    fn name(&self) -> &str {
        &self.template.name
    }
    fn num_qubits(&self) -> u32 {
        self.template.num_qubits
    }
    fn num_clbits(&self) -> u32 {
        0
    }
    fn num_params(&self) -> u32 {
        self.template.num_params
    }
    fn directive(&self) -> bool {
        false
    }
}

impl CustomOperation for DefinedGate {
    fn is_unitary(&self) -> bool {
        true
    }

    fn definition(&self, params: &[Param]) -> Option<CircuitData> {
        self.template.build_definition(params)
    }
}

/// Indexed by `GateId`, so the order must match `State::new` in `parse.rs`: `U`=0, `CX`=1, then
/// `qelib1`.  That holds only while no caller passes the parser its own `CustomInstruction`s;
/// supporting those must revisit this, or gate ids will silently resolve to the wrong gate.
struct GateRegistry {
    gates: Vec<GateEntry>,
}

impl GateRegistry {
    fn new() -> Self {
        Self {
            gates: vec![
                GateEntry::Standard(StandardGate::U),
                GateEntry::Standard(StandardGate::CX),
            ],
        }
    }

    fn extend_from_qelib1(&mut self, indices: &[usize]) {
        for &index in indices {
            self.gates.push(GateEntry::Standard(QELIB1[index]));
        }
    }

    fn declare(
        &mut self,
        name: String,
        num_qubits: u32,
        num_params: u32,
        body: Option<Vec<BodyInstruction>>,
    ) {
        self.gates
            .push(GateEntry::Defined(Arc::new(DefinedGateTemplate {
                name,
                num_qubits,
                num_params,
                body,
            })));
    }

    fn get(&self, id: GateId) -> Result<GateEntry, ParseError> {
        self.gates
            .get(id.index())
            .cloned()
            .ok_or_else(|| ParseError::new(format!("gate id {} was not declared", id.index())))
    }
}

pub(crate) fn build_circuit(
    bytecode: impl IntoIterator<Item = Result<InternalBytecode, ParseError>>,
) -> Result<CircuitData, ParseError> {
    let mut circuit = CircuitData::new(None, None, Param::Float(0.0))
        .map_err(|err| ParseError::new(format!("failed to create circuit: {err}")))?;
    let mut registry = GateRegistry::new();
    let mut cregs: Vec<ClassicalRegister> = Vec::new();
    // Set while recording a `gate`/`opaque` body, between `DeclareGate` and `EndDeclareGate`;
    // the fields are the gate's name, qubit count, parameter count and body so far.
    let mut current_body: Option<(String, u32, u32, Vec<BodyInstruction>)> = None;

    for instruction in bytecode {
        match &instruction? {
            InternalBytecode::Gate {
                id,
                arguments,
                qubits,
            } => {
                let entry = registry.get(*id)?;
                push_gate(&mut circuit, &entry, arguments, &to_qubits(qubits))?;
            }
            InternalBytecode::Measure { qubit, clbit } => {
                push_standard_instruction(
                    &mut circuit,
                    StandardInstruction::Measure,
                    &to_qubits(&[*qubit]),
                    &to_clbits(&[*clbit]),
                )?;
            }
            InternalBytecode::Reset { qubit } => {
                push_standard_instruction(
                    &mut circuit,
                    StandardInstruction::Reset,
                    &to_qubits(&[*qubit]),
                    &[],
                )?;
            }
            InternalBytecode::Barrier { qubits } => {
                if let Some((_, _, _, body)) = current_body.as_mut() {
                    body.push(BodyInstruction::Barrier {
                        qubits: qubits.clone(),
                    });
                } else {
                    push_standard_instruction(
                        &mut circuit,
                        StandardInstruction::Barrier(qubits.len() as u32),
                        &to_qubits(qubits),
                        &[],
                    )?;
                }
            }
            InternalBytecode::DeclareQreg { name, size } => {
                let register = QuantumRegister::new_owning(name.clone(), *size as u32);
                circuit.add_qreg(register, true).map_err(|err| {
                    ParseError::new(format!("failed to declare qreg '{name}': {err}"))
                })?;
            }
            InternalBytecode::DeclareCreg { name, size } => {
                let register = ClassicalRegister::new_owning(name.clone(), *size as u32);
                cregs.push(register.clone());
                circuit.add_creg(register, true).map_err(|err| {
                    ParseError::new(format!("failed to declare creg '{name}': {err}"))
                })?;
            }
            InternalBytecode::SpecialInclude { indices } => {
                registry.extend_from_qelib1(indices);
            }
            InternalBytecode::ConditionedGate {
                id,
                arguments,
                qubits,
                creg,
                value,
            } => {
                let entry = registry.get(*id)?;
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    &to_qubits(qubits),
                    &[],
                    |block, _| {
                        let local_qargs: Vec<Qubit> = (0..qubits.len() as u32).map(Qubit).collect();
                        push_gate(block, &entry, arguments, &local_qargs)
                    },
                )?;
            }
            InternalBytecode::ConditionedMeasure {
                qubit,
                clbit,
                creg,
                value,
            } => {
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    &to_qubits(&[*qubit]),
                    &to_clbits(&[*clbit]),
                    |block, local_cargs| {
                        push_standard_instruction(
                            block,
                            StandardInstruction::Measure,
                            &[Qubit(0)],
                            local_cargs,
                        )
                    },
                )?;
            }
            InternalBytecode::ConditionedReset { qubit, creg, value } => {
                push_conditioned(
                    &mut circuit,
                    &cregs,
                    *creg,
                    value,
                    &to_qubits(&[*qubit]),
                    &[],
                    |block, _| {
                        push_standard_instruction(
                            block,
                            StandardInstruction::Reset,
                            &[Qubit(0)],
                            &[],
                        )
                    },
                )?;
            }
            InternalBytecode::DeclareGate {
                name,
                num_qubits,
                num_params,
            } => {
                if current_body.is_some() {
                    return Err(ParseError::new(
                        "nested gate declaration: missing an EndDeclareGate",
                    ));
                }
                current_body = Some((
                    name.clone(),
                    *num_qubits as u32,
                    *num_params as u32,
                    Vec::new(),
                ));
            }
            InternalBytecode::GateInBody {
                id,
                arguments,
                qubits,
            } => {
                let entry = registry.get(*id)?;
                let (_, _, _, body) = current_body.as_mut().ok_or_else(|| {
                    ParseError::new("gate body instruction outside of a gate declaration")
                })?;
                body.push(BodyInstruction::Gate {
                    entry,
                    arguments: arguments.clone(),
                    qubits: qubits.clone(),
                });
            }
            InternalBytecode::EndDeclareGate {} => {
                let (name, num_qubits, num_params, body) =
                    current_body.take().ok_or_else(|| {
                        ParseError::new("EndDeclareGate without a matching DeclareGate")
                    })?;
                registry.declare(name, num_qubits, num_params, Some(body));
            }
            InternalBytecode::DeclareOpaque {
                name,
                num_qubits,
                num_params,
            } => {
                if current_body.is_some() {
                    return Err(ParseError::new(
                        "opaque declaration nested inside another gate declaration",
                    ));
                }
                registry.declare(name.clone(), *num_qubits as u32, *num_params as u32, None);
            }
        }
    }
    if current_body.is_some() {
        return Err(ParseError::new("unterminated gate declaration"));
    }

    Ok(circuit)
}

fn to_qubits(qubits: &[QubitId]) -> Vec<Qubit> {
    qubits.iter().map(|q| Qubit(q.index() as u32)).collect()
}

fn to_clbits(clbits: &[ClbitId]) -> Vec<Clbit> {
    clbits.iter().map(|c| Clbit(c.index() as u32)).collect()
}

fn push_gate(
    circuit: &mut CircuitData,
    entry: &GateEntry,
    arguments: &[f64],
    qargs: &[Qubit],
) -> Result<(), ParseError> {
    if qargs.len() != entry.num_qubits() as usize {
        return Err(ParseError::new(format!(
            "gate registry desync: resolved gate takes {} qubits, but {} were given",
            entry.num_qubits(),
            qargs.len(),
        )));
    }
    if arguments.len() != entry.num_params() as usize {
        return Err(ParseError::new(format!(
            "gate registry desync: resolved gate takes {} parameters, but {} were given",
            entry.num_params(),
            arguments.len(),
        )));
    }
    let params: Vec<Param> = arguments.iter().map(|&v| Param::Float(v)).collect();
    match entry {
        GateEntry::Standard(gate) => circuit
            .push_standard_gate(*gate, &params, qargs)
            .map_err(|err| ParseError::new(format!("failed to apply gate: {err}"))),
        GateEntry::Defined(template) => {
            let defined = DefinedGate {
                template: template.clone(),
            };
            circuit
                .push_packed_operation(
                    PackedOperation::from_custom_operation(Box::new(defined)),
                    Some(Parameters::Params(params.into())),
                    qargs,
                    &[],
                )
                .map_err(|err| ParseError::new(format!("failed to apply gate: {err}")))
        }
    }
}

fn push_standard_instruction(
    circuit: &mut CircuitData,
    instruction: StandardInstruction,
    qubits: &[Qubit],
    clbits: &[Clbit],
) -> Result<(), ParseError> {
    circuit
        .push_packed_operation(PackedOperation::from(instruction), None, qubits, clbits)
        .map_err(|err| ParseError::new(format!("failed to apply instruction: {err}")))
}

fn push_conditioned(
    circuit: &mut CircuitData,
    cregs: &[ClassicalRegister],
    creg: CregId,
    value: &BigUint,
    qargs: &[Qubit],
    cargs: &[Clbit],
    fill_block: impl FnOnce(&mut CircuitData, &[Clbit]) -> Result<(), ParseError>,
) -> Result<(), ParseError> {
    let num_qubits = qargs.len() as u32;
    let register = cregs
        .get(creg.index())
        .ok_or_else(|| ParseError::new(format!("creg id {} was not declared", creg.index())))?;
    let condition_clbits: Vec<Clbit> = register
        .bits()
        .map(|bit| circuit.clbit_index(&bit).map(Clbit))
        .collect::<Option<_>>()
        .ok_or_else(|| {
            ParseError::new(format!(
                "creg '{}' holds a clbit that is not in the circuit",
                register.name()
            ))
        })?;

    let mut block = CircuitData::new(None, None, Param::Float(0.0))
        .map_err(|err| ParseError::new(format!("failed to create circuit: {err}")))?;
    // Reuse the outer circuit's qubits to match the Python loader's register identity.
    for qubit in qargs {
        let bit =
            circuit.qubits().get(*qubit).cloned().ok_or_else(|| {
                ParseError::new(format!("qubit {} is not in the circuit", qubit.0))
            })?;
        block
            .add_qubit(bit, true)
            .map_err(|err| ParseError::new(format!("failed to build conditioned block: {err}")))?;
    }
    // `add_creg` creates the block's first clbits, one per bit of the condition register.
    block
        .add_creg(
            ClassicalRegister::new_owning(
                register.name().to_owned(),
                condition_clbits.len() as u32,
            ),
            true,
        )
        .map_err(|err| ParseError::new(format!("failed to build conditioned block: {err}")))?;

    let mut outer_cargs = condition_clbits;
    let mut local_cargs = Vec::with_capacity(cargs.len());
    for carg in cargs {
        if let Some(local) = outer_cargs.iter().position(|held| held == carg) {
            local_cargs.push(Clbit(local as u32));
            continue;
        }
        let bit =
            circuit.clbits().get(*carg).cloned().ok_or_else(|| {
                ParseError::new(format!("clbit {} is not in the circuit", carg.0))
            })?;
        // Strict, because the block provably lacks this bit; a non-strict add would reset its
        // register locations and so drop the creg added above.
        local_cargs.push(
            block.add_clbit(bit, true).map_err(|err| {
                ParseError::new(format!("failed to build conditioned block: {err}"))
            })?,
        );
        outer_cargs.push(*carg);
    }
    fill_block(&mut block, &local_cargs)?;

    let condition = Condition::Register(register.clone(), value.clone());
    let block_id = circuit.add_block(block);
    let control_flow = ControlFlowInstruction {
        control_flow: ControlFlow::IfElse { condition },
        num_qubits,
        num_clbits: outer_cargs.len() as u32,
    };
    circuit
        .push_packed_operation(
            PackedOperation::from(control_flow),
            Some(Parameters::Blocks(vec![block_id])),
            qargs,
            &outer_cargs,
        )
        .map_err(|err| ParseError::new(format!("failed to apply conditioned instruction: {err}")))
}

#[cfg(test)]
mod tests {
    use crate::circuit_from_string;
    use num_bigint::BigUint;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::operations::{
        Condition, ControlFlow, OperationRef, Param, StandardInstruction,
    };
    use qiskit_circuit::standard_gate::StandardGate;
    use qiskit_circuit::{Clbit, Qubit};

    fn build(program: &str) -> CircuitData {
        circuit_from_string(program.to_owned(), vec![], &[], &[], false)
            .expect("the program is valid OpenQASM 2")
    }

    #[test]
    fn parses_and_builds_a_program_end_to_end() {
        let program = concat!(
            "include \"qelib1.inc\";\n",
            "qreg q[2];\n",
            "creg c[2];\n",
            "gate my_rz(a) qq { u1(2*a) qq; }\n",
            "h q[0];\n",
            "cx q[0], q[1];\n",
            "my_rz(pi/4) q[1];\n",
            "measure q[0] -> c[0];\n",
        );
        let circuit = crate::circuit_from_string(program.to_owned(), vec![], &[], &[], false)
            .expect("the program is valid OpenQASM 2");

        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_clbits(), 2);
        assert_eq!(circuit.data().len(), 4);

        let ops: Vec<_> = circuit.data().iter().map(|inst| inst.op.view()).collect();
        assert!(matches!(
            ops[0],
            OperationRef::StandardGate(StandardGate::H)
        ));
        assert!(matches!(
            ops[1],
            OperationRef::StandardGate(StandardGate::CX)
        ));
        assert!(matches!(
            ops[3],
            OperationRef::StandardInstruction(StandardInstruction::Measure)
        ));
        assert_eq!(
            circuit.get_qargs(circuit.data()[1].qubits),
            &[Qubit(0), Qubit(1)]
        );

        // `my_rz(pi/4)` should expand to `u1(2*pi/4)`, with the body expression folded in Rust.
        let OperationRef::CustomOperation(custom) = ops[2] else {
            panic!("expected the `gate` declaration to build a custom operation");
        };
        assert_eq!(custom.name(), "my_rz");
        let definition = custom
            .definition(&[Param::Float(std::f64::consts::PI / 4.0)])
            .expect("my_rz has a known definition");
        assert_eq!(definition.data().len(), 1);
        assert!(matches!(
            definition.data()[0].op.view(),
            OperationRef::StandardGate(StandardGate::U1)
        ));
        let Param::Float(angle) = definition.data()[0].params_view()[0] else {
            panic!("expected a concrete float parameter");
        };
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-12,
            "got {angle}"
        );
    }

    #[test]
    fn keeps_trailing_resets() {
        // `parse_next` returns as soon as a statement reports that it emitted an instruction, and
        // `circuit_from_string` drains its buffer only then, so a statement that under-reports how
        // much it emitted is silently dropped when nothing follows it in the program.
        for (program, expected) in [
            ("qreg q[1];\nreset q[0];\n", 1),
            ("qreg q[1];\nreset q[0];\nreset q[0];\n", 2),
            (
                "include \"qelib1.inc\";\nqreg q[1];\nx q[0];\nreset q[0];\n",
                2,
            ),
        ] {
            let circuit = crate::circuit_from_string(program.to_owned(), vec![], &[], &[], false)
                .expect("the program is valid OpenQASM 2");
            assert_eq!(
                circuit.data().len(),
                expected,
                "wrong instruction count for {program:?}"
            );
            assert!(matches!(
                circuit.data().last().unwrap().op.view(),
                OperationRef::StandardInstruction(StandardInstruction::Reset)
            ));
        }
    }

    /// Swapping the outer bits for the block-local ones would still build a valid circuit, so
    /// both numberings are asserted.
    #[test]
    fn builds_a_conditioned_gate() {
        let circuit = build(concat!(
            "include \"qelib1.inc\";\n",
            "qreg q[3];\n",
            "creg c[2];\n",
            "if (c == 1) cx q[1], q[2];\n",
        ));
        assert_eq!(circuit.data().len(), 1);
        let instruction = &circuit.data()[0];

        let OperationRef::ControlFlow(control_flow) = instruction.op.view() else {
            panic!("expected a control-flow instruction");
        };
        let ControlFlow::IfElse { condition } = &control_flow.control_flow else {
            panic!("expected an if/else");
        };
        let Condition::Register(register, value) = condition else {
            panic!("expected a register condition");
        };
        assert_eq!(register.name(), "c");
        assert_eq!(*value, BigUint::from(1u32));

        // Outer: the bits the statement actually named.
        assert_eq!(circuit.get_qargs(instruction.qubits), &[Qubit(1), Qubit(2)]);
        assert_eq!(circuit.get_cargs(instruction.clbits), &[Clbit(0), Clbit(1)]);

        // Inner: renumbered from zero, with the condition register declared alongside.
        let block = circuit.blocks()[instruction.blocks_view()[0]].clone();
        assert_eq!(block.num_qubits(), 2);
        assert_eq!(block.num_clbits(), 2);
        assert_eq!(
            block
                .cregs()
                .iter()
                .map(|r| (r.name(), r.len()))
                .collect::<Vec<_>>(),
            vec![("c", 2)],
        );
        assert_eq!(block.data().len(), 1);
        assert!(matches!(
            block.data()[0].op.view(),
            OperationRef::StandardGate(StandardGate::CX)
        ));
        assert_eq!(
            block.get_qargs(block.data()[0].qubits),
            &[Qubit(0), Qubit(1)]
        );
    }

    #[test]
    fn defined_gates_expand_one_level_at_a_time() {
        let circuit = build(concat!(
            "include \"qelib1.inc\";\n",
            "gate my_h q { h q; }\n",
            "gate double_h q { my_h q; my_h q; }\n",
            "qreg q[1];\n",
            "double_h q[0];\n",
        ));
        let OperationRef::CustomOperation(outer) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };
        assert_eq!(outer.name(), "double_h");

        let outer_definition = outer.definition(&[]).expect("double_h has a definition");
        assert_eq!(outer_definition.data().len(), 2);
        for instruction in outer_definition.data() {
            let OperationRef::CustomOperation(inner) = instruction.op.view() else {
                panic!("expected a nested custom operation, not a flattened gate");
            };
            assert_eq!(inner.name(), "my_h");

            let inner_definition = inner.definition(&[]).expect("my_h has a definition");
            assert_eq!(inner_definition.data().len(), 1);
            assert!(matches!(
                inner_definition.data()[0].op.view(),
                OperationRef::StandardGate(StandardGate::H)
            ));
        }
    }

    #[test]
    fn opaque_gates_have_no_definition() {
        let circuit = build("opaque black_box q;\nqreg q[1];\nblack_box q[0];\n");
        let OperationRef::CustomOperation(custom) = circuit.data()[0].op.view() else {
            panic!("expected a custom operation");
        };
        assert_eq!(custom.name(), "black_box");
        assert!(custom.definition(&[]).is_none());
    }

    /// Custom instructions aren't supported yet, so must be refused, not mis-resolved.
    #[test]
    fn rejects_custom_instructions() {
        let custom = [crate::CustomInstruction {
            name: "my_swap".to_owned(),
            num_params: 0,
            num_qubits: 2,
            builtin: false,
        }];
        let result = circuit_from_string(
            "include \"qelib1.inc\";\nqreg q[2];\nx q[0];\n".to_owned(),
            vec![],
            &custom,
            &[],
            false,
        );
        let Err(error) = result else {
            panic!("custom instructions must be rejected until the builder supports them");
        };
        assert!(
            error
                .message
                .contains("custom instructions are not yet supported"),
            "unexpected message: {}",
            error.message
        );
    }
}
