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

//! An op that runs each of a list of circuits for a number of shots, over a sweep of
//! parameterizations.

use qiskit_circuit::circuit_data::CircuitData;
use thiserror::Error;

use super::inference::{float, leading_axes};
use super::{ProgramOp, QISKIT};
use crate::tensor::{DType, Dim, Tensor, TensorType};

/// Run each of a vector of circuits for a fixed number of shots.
///
/// There is one operand per circuit for that circuit's parameter values.
/// For each circuit, one value is returned per classical register.
#[derive(Clone)]
pub struct ShotLoop {
    circuits: Vec<CircuitData>,
    shots: usize,
}

impl ShotLoop {
    /// Construct a shot loop running each of `circuits` for `shots` shots.
    pub fn new(circuits: Vec<CircuitData>, shots: usize) -> Result<Self, ShotLoopError> {
        Ok(Self { circuits, shots })
    }

    /// Return the circuits this op runs.
    pub fn circuits(&self) -> &[CircuitData] {
        &self.circuits
    }

    /// Return a mutable reference to the `i`th circuit, or `None` if there is no such circuit.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut CircuitData> {
        self.circuits.get_mut(i)
    }

    /// Return how many shots each circuit is run for.
    pub fn shots(&self) -> usize {
        self.shots
    }
}

impl std::fmt::Debug for ShotLoop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct Summary<'a>(&'a CircuitData);

        impl std::fmt::Debug for Summary<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("CircuitData")
                    .field("num_qubits", &self.0.num_qubits())
                    .field(
                        "cregs",
                        &self
                            .0
                            .cregs()
                            .iter()
                            .map(|creg| (creg.name(), creg.len()))
                            .collect::<Vec<_>>(),
                    )
                    .finish_non_exhaustive()
            }
        }

        f.debug_struct("ShotLoop")
            .field(
                "circuits",
                &self.circuits.iter().map(|c| Summary(c)).collect::<Vec<_>>(),
            )
            .field("shots", &self.shots)
            .finish()
    }
}

impl ProgramOp for ShotLoop {
    type Error = ShotLoopError;

    fn name(&self) -> &str {
        "shot_loop"
    }
    fn namespace(&self) -> &str {
        QISKIT
    }
    fn describe(&self) -> Option<String> {
        Some(format!(
            "circuits={}, shots={}",
            self.circuits.len(),
            self.shots
        ))
    }
    fn arity(&self) -> usize {
        self.circuits.len()
    }
    fn has_builtin_eval(&self) -> bool {
        false
    }
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error> {
        assert_eq!(
            inputs.len(),
            self.arity(),
            "{} expects one operand per circuit",
            self.full_name()
        );
        let mut outputs = Vec::with_capacity(self.circuits.len());
        for (index, (circuit, operand)) in self.circuits.iter().zip(inputs).enumerate() {
            let parameters = circuit.num_parameters();
            let refuse = || ShotLoopError::ParameterType {
                circuit: index,
                parameters,
                actual: operand.clone(),
            };
            if !float(operand.dtype) {
                return Err(refuse());
            }
            let batch = leading_axes(&operand.shape, parameters).ok_or_else(refuse)?;
            for register in circuit.cregs() {
                let mut shape = batch.to_vec();
                shape.push(Dim::Fixed(self.shots));
                shape.push(Dim::Fixed(register.len()));
                outputs.push(TensorType {
                    dtype: DType::Bit,
                    shape,
                });
            }
        }
        Ok(outputs)
    }
    fn eval(&self, _args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error> {
        Err(ShotLoopError::NoBuiltinEval)
    }
}

/// Errors returned by [`ShotLoop`].
#[derive(Debug, Error)]
pub enum ShotLoopError {
    /// A circuit's parameter values are not a floating-point tensor whose trailing axis is that
    /// circuit's parameter count.
    #[error(
        "circuit {circuit}: expected a floating-point tensor of shape [..., {parameters}], got \
         {actual}"
    )]
    ParameterType {
        circuit: usize,
        parameters: usize,
        actual: TensorType,
    },

    /// Circuits must be executed by a backend.
    #[error("a shot loop has no in-process implementation")]
    NoBuiltinEval,
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::ClassicalRegister;
    use qiskit_circuit::operations::Param;
    use qiskit_circuit::operations::StandardGate;
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;

    use super::*;
    use crate::DataTree;
    use crate::ops::Mean;
    use crate::program::{ProgramFunction, QuantumProgram, StepperError};

    /// Return a circuit taking `parameters` with `registers` as `(name, width)` pairs.
    fn circuit(parameters: usize, registers: &[(&str, u32)]) -> CircuitData {
        let symbol =
            |index| ParameterExpression::from_symbol(Symbol::standalone(format!("p{index}"), None));
        let global_phase = (0..parameters)
            .map(symbol)
            .reduce(|sum, term| sum.add(&term).unwrap())
            .map_or(Param::Float(0.0), |expr| {
                Param::ParameterExpression(Arc::new(expr))
            });

        let mut circuit = CircuitData::new(None, None, global_phase).unwrap();
        for &(name, width) in registers {
            circuit
                .add_creg(ClassicalRegister::new_owning(name, width), true)
                .unwrap();
        }
        circuit
    }

    /// Return a `TensorType` of `dtype` over fixed axes `shape`.
    fn ty(dtype: DType, shape: &[usize]) -> TensorType {
        TensorType {
            dtype,
            shape: shape.iter().copied().map(Dim::Fixed).collect(),
        }
    }

    /// Return the `Bit` type of one register's outcomes.
    fn outcomes(batch: &[Dim], shots: usize, width: usize) -> TensorType {
        let mut shape = batch.to_vec();
        shape.extend([Dim::Fixed(shots), Dim::Fixed(width)]);
        TensorType {
            dtype: DType::Bit,
            shape,
        }
    }

    #[test]
    fn test_shot_loop_full_name_and_arity() {
        let op = ShotLoop::new(vec![circuit(1, &[("c", 2)]), circuit(0, &[])], 100).unwrap();
        assert_eq!(op.full_name(), "qiskit.shot_loop");
        assert_eq!(op.arity(), 2, "one operand per circuit");
        assert_eq!(op.shots(), 100);
        assert_eq!(op.circuits().len(), 2);
        assert!(!op.has_builtin_eval(), "a backend has to run the circuits");
    }

    #[test]
    fn test_get_mut() {
        let mut op = ShotLoop::new(vec![circuit(1, &[("c", 2)])], 100).unwrap();

        let circuit = &op.circuits[0];
        assert_eq!(circuit.len(), 0);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[3, 1])]).unwrap(),
            vec![ty(DType::Bit, &[3, 100, 2])]
        );

        let mut_circuit = op.get_mut(0).unwrap();
        mut_circuit
            .push_standard_gate(StandardGate::H, &[], &[Qubit(0)])
            .unwrap();
        mut_circuit
            .add_creg(ClassicalRegister::new_owning("new_creg", 5), true)
            .unwrap();

        let circuit = &op.circuits[0];
        assert_eq!(circuit.len(), 1);
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[3, 1])]).unwrap(),
            vec![ty(DType::Bit, &[3, 100, 2]), ty(DType::Bit, &[3, 100, 5])]
        );
    }

    #[test]
    fn test_infer_output_types_are_bit_outcomes_per_register() {
        let op = ShotLoop::new(
            vec![
                circuit(2, &[("c", 2), ("meas", 3)]),
                circuit(1, &[("d", 1)]),
            ],
            100,
        )
        .unwrap();
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[2]), ty(DType::F64, &[1])])
                .unwrap(),
            vec![
                outcomes(&[], 100, 2),
                outcomes(&[], 100, 3),
                outcomes(&[], 100, 1),
            ],
            "one result per register, in the order the structure describes"
        );
    }

    #[test]
    fn test_infer_output_types_carries_each_operands_batch_prefix() {
        let op = ShotLoop::new(vec![circuit(2, &[("c", 2)]), circuit(1, &[("d", 1)])], 8).unwrap();
        let bounded = Dim::Bounded { max: 4 };
        assert_eq!(
            op.infer_output_types(&[
                ty(DType::F32, &[5, 3, 2]),
                TensorType {
                    dtype: DType::F64,
                    shape: vec![bounded, Dim::Fixed(1)],
                },
            ])
            .unwrap(),
            vec![
                outcomes(&[Dim::Fixed(5), Dim::Fixed(3)], 8, 2),
                outcomes(&[bounded], 8, 1),
            ]
        );
    }

    #[test]
    fn test_infer_output_types_needs_one_value_per_parameter_even_when_there_are_none() {
        let op = ShotLoop::new(vec![circuit(0, &[("c", 1)])], 8).unwrap();
        assert_eq!(
            op.infer_output_types(&[ty(DType::F64, &[4, 0])]).unwrap(),
            vec![outcomes(&[Dim::Fixed(4)], 8, 1)],
            "the parameter axis of a circuit taking no parameters is empty, not absent"
        );
        let no_axis = ty(DType::F64, &[4]);
        assert!(
            matches!(
                op.infer_output_types(std::slice::from_ref(&no_axis))
                    .unwrap_err(),
                ShotLoopError::ParameterType {
                    circuit: 0,
                    parameters: 0,
                    actual,
                } if actual == no_axis
            ),
            "the trailing axis is the parameter axis, and four values are not none"
        );
    }

    #[test]
    fn test_infer_output_types_rejects_values_that_are_not_this_circuits_parameters() {
        let op = ShotLoop::new(vec![circuit(0, &[]), circuit(2, &[("c", 1)])], 8).unwrap();
        let ok = ty(DType::F64, &[0]);

        // Both the shape a circuit's values must have and the type supplied are named.
        assert!(matches!(
            op.infer_output_types(&[ok.clone(), ty(DType::F64, &[3])])
                .unwrap_err(),
            ShotLoopError::ParameterType {
                circuit: 1,
                parameters: 2,
                actual,
            } if actual == ty(DType::F64, &[3])
        ));

        for operand in [
            ty(DType::I64, &[2]),
            ty(DType::Bit, &[2]),
            ty(DType::F64, &[]),
            TensorType {
                dtype: DType::F64,
                shape: vec![Dim::Bounded { max: 2 }],
            },
        ] {
            assert!(matches!(
                op.infer_output_types(&[ok.clone(), operand.clone()])
                    .unwrap_err(),
                ShotLoopError::ParameterType {
                    circuit: 1,
                    parameters: 2,
                    actual,
                } if actual == operand
            ));
        }
    }

    #[test]
    fn test_eval_has_no_in_process_implementation() {
        let op = ShotLoop::new(vec![circuit(0, &[("c", 1)])], 8).unwrap();
        assert!(matches!(
            op.eval(&[Tensor::from(&[] as &[f64])]).unwrap_err(),
            ShotLoopError::NoBuiltinEval
        ));
    }

    #[test]
    fn test_debug_summarizes_each_circuit() {
        let op = ShotLoop::new(
            vec![circuit(1, &[("c", 2), ("meas", 3)]), circuit(0, &[])],
            100,
        )
        .unwrap();
        let debug = format!("{op:?}");
        assert_eq!(
            debug,
            concat!(
                "ShotLoop { circuits: [",
                "CircuitData { num_qubits: 0, cregs: [(\"c\", 2), (\"meas\", 3)], .. }, ",
                "CircuitData { num_qubits: 0, cregs: [], .. }",
                "], shots: 100 }"
            )
        );
        assert!(
            !debug.contains("interner"),
            "a circuit's storage is summarized rather than dumped"
        );
    }

    #[test]
    fn test_outcomes_are_ordinary_values() {
        let mut function = ProgramFunction::new();
        let values = function.add_parameter(ty(DType::F64, &[2]));
        let op = ShotLoop::new(vec![circuit(2, &[("c", 3)])], 100).unwrap();
        let bits = function.add_op(op, &[values]).unwrap()[0];
        let mean = function.add_op(Mean::new(0), &[bits]).unwrap()[0];

        assert_eq!(function.type_of(bits), Some(&outcomes(&[], 100, 3)));
        assert_eq!(function.type_of(mean), Some(&ty(DType::F64, &[3])));
    }

    #[test]
    fn test_a_program_of_circuits_reports_its_types_and_refuses_to_evaluate() {
        let op = ShotLoop::new(
            vec![circuit(2, &[("c", 2)]), circuit(0, &[("meas", 3)])],
            100,
        )
        .unwrap();

        let mut function = ProgramFunction::new();
        let theta = function.add_parameter(ty(DType::F64, &[2]));
        let none = function.add_parameter(ty(DType::F64, &[0]));
        for outcome in function.add_op(op, &[theta, none]).unwrap() {
            function.add_result(outcome).unwrap();
        }
        let program = QuantumProgram::new(
            vec![function],
            DataTree::mapping([("theta", DataTree::Leaf(())), ("none", DataTree::Leaf(()))])
                .unwrap(),
            DataTree::sequence([
                DataTree::mapping([("c", DataTree::Leaf(()))]).unwrap(),
                DataTree::mapping([("meas", DataTree::Leaf(()))]).unwrap(),
            ]),
        )
        .unwrap();

        assert_eq!(
            program.output_types(),
            DataTree::sequence([
                DataTree::mapping([("c", DataTree::Leaf(outcomes(&[], 100, 2)))]).unwrap(),
                DataTree::mapping([("meas", DataTree::Leaf(outcomes(&[], 100, 3)))]).unwrap(),
            ])
        );
        assert!(!program.has_builtin_eval());

        let inputs = DataTree::mapping([
            ("theta", DataTree::Leaf(Tensor::from([0.5_f64, 1.5]))),
            ("none", DataTree::Leaf(Tensor::from(&[] as &[f64]))),
        ])
        .unwrap();
        let Err(err) = program.eval(inputs) else {
            panic!("a program containing a shot loop cannot be evaluated in process")
        };
        assert!(
            matches!(
                err,
                StepperError::NoBuiltinEval { full_name, .. }
                    if full_name == "qiskit.shot_loop"
            ),
            "the op a backend is needed for is named"
        );
    }
}
