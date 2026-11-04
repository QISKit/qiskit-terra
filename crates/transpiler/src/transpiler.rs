// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use anyhow::Result;

use crate::commutation_checker::CommutationChecker;
use crate::commutation_checker::get_standard_commutation_checker;
use crate::equivalence::EquivalenceLibrary;
use crate::passes::sabre::route::PyRoutingTarget;
use crate::passes::unitary_synthesis;
use crate::passes::*;
use crate::standard_equivalence_library::generate_standard_equivalence_library;
use crate::target::Target;
use crate::transpile_layout::TranspileLayout;
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::{PhysicalQubit, Qubit, VirtualQubit};

#[derive(Copy, Eq, PartialEq, Debug, Clone)]
#[repr(u8)]
pub enum OptimizationLevel {
    Level0 = 0,
    Level1 = 1,
    Level2 = 2,
    Level3 = 3,
}

/// The source of the initial layout found when running [`layout_stage`].
/// This is typically used to control whether `VF2PostLayout` is run during
/// the [`routing_stage`]
pub enum LayoutSource {
    Trivial,
    VF2,
    Sabre,
}

impl From<u8> for OptimizationLevel {
    fn from(value: u8) -> Self {
        match value {
            0 => Self::Level0,
            1 => Self::Level1,
            2 => Self::Level2,
            3 => Self::Level3,
            _ => panic!("Invalid optimization level specified {value}"),
        }
    }
}

fn unroll_3q_or_more(
    dag: &mut DAGCircuit,
    target: &Target,
    synthesis_state: &mut UnitarySynthesisState,
) -> Result<()> {
    let physical_qubits = (0..target.num_qubits.unwrap_or(0))
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    if let Some(out) = run_unitary_synthesis(
        dag,
        &["unitary", "swap"].map(String::from).into_iter().collect(),
        3,
        &physical_qubits,
        synthesis_state,
        target.into(),
        false,
    )? {
        *dag = out;
    }
    run_unroll_3q_or_more(dag, Some(target))?;
    Ok(())
}

#[inline]
pub fn init_stage(
    dag: &mut DAGCircuit,
    target: &Target,
    optimization_level: OptimizationLevel,
    approximation_degree: Option<f64>,
    synthesis_state: &mut UnitarySynthesisState,
    transpile_layout: &mut TranspileLayout,
    commutation_checker: &mut CommutationChecker,
) -> Result<()> {
    // Init stage
    unroll_3q_or_more(dag, target, synthesis_state)?;
    if optimization_level == OptimizationLevel::Level1 {
        run_inverse_cancellation_standard_gates(dag);
    } else if matches!(
        optimization_level,
        OptimizationLevel::Level2 | OptimizationLevel::Level3
    ) {
        if let Some((new_dag, permutation)) = run_elide_permutations(dag)? {
            *dag = new_dag;
            transpile_layout.add_permutation_inside(|q| Qubit::new(permutation[q.index()]));
        };
        run_remove_diagonal_before_measure(dag);
        run_remove_identity_equiv(dag, approximation_degree, Some(target))?;
        run_inverse_cancellation_standard_gates(dag);
        cancel_commutations(dag, commutation_checker, None, 1.0)?;
        run_consolidate_blocks(dag, false, approximation_degree, None)?;
        let result = run_split_2q_unitaries(
            dag,
            approximation_degree
                .unwrap_or(1.0 - f64::EPSILON)
                .min(1.0 - f64::EPSILON),
            true,
        )?;
        if let Some(result) = result {
            *dag = result.0;
            let permutation = result.1;
            transpile_layout.add_permutation_inside(|q| Qubit::new(permutation[q.index()]));
        }
    }
    Ok(())
}

#[inline]
pub fn layout_stage(
    dag: &mut DAGCircuit,
    target: &Target,
    optimization_level: OptimizationLevel,
    seed: Option<u64>,
    sabre_heuristic: &sabre::Heuristic,
    transpile_layout: &mut TranspileLayout,
) -> Result<LayoutSource> {
    let mut layout_source = LayoutSource::Sabre;
    let vf2_config = match optimization_level {
        OptimizationLevel::Level0 => vf2::Vf2PassConfiguration::default_abstract(), // Not used.
        OptimizationLevel::Level1 | OptimizationLevel::Level2 => vf2::Vf2PassConfiguration {
            call_limit: (Some(5_000_000), Some(10_000)),
            time_limit: None,
            max_trials: None,
            shuffle_seed: None,
            score_initial_layout: false,
        },
        OptimizationLevel::Level3 => vf2::Vf2PassConfiguration {
            call_limit: (Some(30_000_000), Some(100_000)),
            time_limit: None,
            max_trials: None,
            shuffle_seed: None,
            score_initial_layout: false,
        },
    };

    if optimization_level == OptimizationLevel::Level0 {
        // Apply a trivial layout
        apply_layout(dag, transpile_layout, target.num_qubits.unwrap(), |x| {
            PhysicalQubit(x.0)
        });
    } else if optimization_level == OptimizationLevel::Level1 {
        // run_check_map returns Some((gate, qargs)) if the circuit violates the connectivity
        // constraints in the target and returns None if the circuit conforms to the undirected
        // connectivity constraints
        if run_check_map(dag, target).is_none() {
            layout_source = LayoutSource::Trivial;
            apply_layout(dag, transpile_layout, target.num_qubits.unwrap(), |x| {
                PhysicalQubit(x.0)
            });
        } else if let vf2::Vf2PassReturn::Solution(layout) =
            vf2_layout_pass_average(dag, target, &vf2_config, false, None)?
        {
            layout_source = LayoutSource::VF2;
            apply_layout(dag, transpile_layout, target.num_qubits.unwrap(), |x| {
                layout[&x]
            });
        } else {
            let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
                dag,
                target,
                sabre_heuristic,
                2,
                20,
                20,
                seed,
                Vec::new(),
                false,
            )?;
            *dag = result;
            *transpile_layout =
                layout_from_sabre_result(dag, initial_layout, &final_layout, transpile_layout);
        }
    } else if optimization_level == OptimizationLevel::Level2 {
        if let vf2::Vf2PassReturn::Solution(layout) =
            vf2_layout_pass_average(dag, target, &vf2_config, false, None)?
        {
            layout_source = LayoutSource::VF2;
            apply_layout(dag, transpile_layout, target.num_qubits.unwrap(), |x| {
                layout[&x]
            });
        } else {
            let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
                dag,
                target,
                sabre_heuristic,
                2,
                20,
                20,
                seed,
                Vec::new(),
                false,
            )?;
            *dag = result;
            *transpile_layout =
                layout_from_sabre_result(dag, initial_layout, &final_layout, transpile_layout);
        }
    } else if let vf2::Vf2PassReturn::Solution(layout) =
        vf2_layout_pass_average(dag, target, &vf2_config, false, None)?
    {
        layout_source = LayoutSource::VF2;
        apply_layout(dag, transpile_layout, target.num_qubits.unwrap(), |x| {
            layout[&x]
        });
    } else {
        let (result, initial_layout, final_layout) = sabre::sabre_layout_and_routing(
            dag,
            target,
            sabre_heuristic,
            4,
            20,
            20,
            seed,
            Vec::new(),
            false,
        )?;
        *dag = result;
        *transpile_layout =
            layout_from_sabre_result(dag, initial_layout, &final_layout, transpile_layout);
    }
    Ok(layout_source)
}

#[inline]
pub fn routing_stage(
    dag: &mut DAGCircuit,
    target: &Target,
    optimization_level: OptimizationLevel,
    seed: Option<u64>,
    sabre_heuristic: &sabre::Heuristic,
    transpile_layout: &mut TranspileLayout,
    layout_source: LayoutSource,
) -> Result<()> {
    if optimization_level == OptimizationLevel::Level0 {
        let routing_target = PyRoutingTarget::from_target(target)?;
        // run_check_map returns Some((gate, qargs)) if the circuit violates the connectivity
        // constraints in the target and returns None if the circuit conforms to the undirected
        // connectivity constraints
        if run_check_map(dag, target).is_some() {
            let initial_layout = transpile_layout
                .initial_layout()
                .expect("a layout pass was already called");
            let (out_dag, final_layout) = sabre::sabre_routing(
                dag,
                &routing_target,
                sabre_heuristic,
                initial_layout,
                5,
                seed,
                Some(true),
            )?;
            *dag = out_dag;
            let routing_permutation =
                TranspileLayout::permutation_from_layouts(initial_layout, &final_layout);
            transpile_layout.add_permutation_inside(|q| routing_permutation[q.index()]);
        }
    } else if !matches!(layout_source, LayoutSource::VF2 | LayoutSource::Trivial) {
        let vf2_config = match optimization_level {
            OptimizationLevel::Level0 => vf2::Vf2PassConfiguration::default_abstract(), // Not used.
            OptimizationLevel::Level1 | OptimizationLevel::Level2 => vf2::Vf2PassConfiguration {
                call_limit: (Some(5_000_000), Some(10_000)),
                time_limit: None,
                max_trials: None,
                shuffle_seed: None,
                score_initial_layout: true,
            },
            OptimizationLevel::Level3 => vf2::Vf2PassConfiguration {
                call_limit: (Some(30_000_000), Some(100_000)),
                time_limit: None,
                max_trials: None,
                shuffle_seed: None,
                score_initial_layout: true,
            },
        };

        if let vf2::Vf2PassReturn::Solution(layout) =
            vf2_layout_pass_average(dag, target, &vf2_config, false, None)?
        {
            update_layout(dag, transpile_layout, |x| {
                Qubit(layout[&VirtualQubit(x.0)].0)
            });
        }
    }
    Ok(())
}

#[inline]
pub fn translation_stage(
    dag: &mut DAGCircuit,
    target: &Target,
    synthesis_state: &mut UnitarySynthesisState,
    equiv_lib: &mut EquivalenceLibrary,
) -> Result<()> {
    let physical_qubits = (0..target.num_qubits.unwrap_or(0))
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    if let Some(out) = run_unitary_synthesis(
        dag,
        &["unitary".to_string()].into_iter().collect(),
        0,
        &physical_qubits,
        synthesis_state,
        target.into(),
        false,
    )? {
        *dag = out;
    }
    if let Some(out_dag) = run_basis_translator(dag, equiv_lib, 0, Some(target), None)? {
        *dag = out_dag;
    }
    if !check_direction_target(dag, target)? {
        fix_direction_target(dag, target)?;
        if gates_missing_from_target(dag, target)?
            && let Some(out_dag) =
                run_basis_translator(dag, equiv_lib, 0, Some(target), None).unwrap()
        {
            *dag = out_dag;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub fn optimization_stage(
    dag: &mut DAGCircuit,
    target: &Target,
    optimization_level: OptimizationLevel,
    approximation_degree: Option<f64>,
    synthesis_state: &mut UnitarySynthesisState,
    commutation_checker: &mut CommutationChecker,
    equivalence_library: &mut EquivalenceLibrary,
    transpile_layout: &mut TranspileLayout,
) -> Result<()> {
    let mut depth: Option<usize> = None;
    let mut size: Option<usize> = None;
    let mut new_depth;
    let mut new_size;
    let physical_qubits = (0..target.num_qubits.unwrap_or(0))
        .map(PhysicalQubit::new)
        .collect::<Vec<_>>();
    if optimization_level == OptimizationLevel::Level1 {
        new_depth = Some(dag.depth(false)?);
        new_size = Some(dag.size(false)?);
        let optimize_1q_state =
            Optimize1qGatesDecompositionState::new(target.num_qubits.unwrap_or(0) as usize);
        while new_depth != depth || new_size != size {
            depth = new_depth;
            size = new_size;
            run_optimize_1q_gates_decomposition(dag, &optimize_1q_state, Some(target), None, None)?;
            run_inverse_cancellation_standard_gates(dag);
            if gates_missing_from_target(dag, target)? {
                translation_stage(dag, target, synthesis_state, equivalence_library)?;
            }
            new_depth = Some(dag.depth(false)?);
            new_size = Some(dag.size(false)?);
        }
    } else if optimization_level == OptimizationLevel::Level2 {
        run_consolidate_blocks(dag, false, approximation_degree, Some(target))?;
        if let Some(out) = run_unitary_synthesis(
            dag,
            &["unitary".to_string()].into_iter().collect(),
            0,
            &physical_qubits,
            synthesis_state,
            target.into(),
            false,
        )? {
            *dag = out;
        }
        new_depth = Some(dag.depth(false)?);
        new_size = Some(dag.size(false)?);
        let optimize_1q_state =
            Optimize1qGatesDecompositionState::new(target.num_qubits.unwrap_or(0) as usize);
        while new_depth != depth || new_size != size {
            depth = new_depth;
            size = new_size;
            run_remove_identity_equiv(dag, approximation_degree, Some(target))?;
            run_optimize_1q_gates_decomposition(dag, &optimize_1q_state, Some(target), None, None)?;
            cancel_commutations(dag, commutation_checker, None, 1.0)?;
            if gates_missing_from_target(dag, target)? {
                translation_stage(dag, target, synthesis_state, equivalence_library)?;
            }
            new_depth = Some(dag.depth(false)?);
            new_size = Some(dag.size(false)?);
        }
    } else if optimization_level == OptimizationLevel::Level3 {
        let mut continue_loop: bool = true;
        let mut min_state = MinPointState::new(dag);
        let optimize_1q_state =
            Optimize1qGatesDecompositionState::new(target.num_qubits.unwrap_or(0) as usize);
        while continue_loop {
            run_consolidate_blocks(dag, false, approximation_degree, Some(target))?;
            if let Some(out) = run_unitary_synthesis(
                dag,
                &["unitary"].into_iter().map(|x| x.to_string()).collect(),
                0,
                &physical_qubits,
                synthesis_state,
                target.into(),
                false,
            )? {
                *dag = out
            }
            run_remove_identity_equiv(dag, approximation_degree, Some(target))?;
            run_optimize_1q_gates_decomposition(dag, &optimize_1q_state, Some(target), None, None)?;
            cancel_commutations(dag, commutation_checker, None, 1.0)?;
            if gates_missing_from_target(dag, target)? {
                translation_stage(dag, target, synthesis_state, equivalence_library)?;
            }
            continue_loop = min_state.update_with(dag);
        }
        *dag = min_state.best_dag;
        let vf2_config = match optimization_level {
            OptimizationLevel::Level0 => vf2::Vf2PassConfiguration::default_concrete(), // Not used.
            OptimizationLevel::Level1 | OptimizationLevel::Level2 => vf2::Vf2PassConfiguration {
                call_limit: (Some(5_000_000), Some(10_000)),
                time_limit: None,
                max_trials: None,
                shuffle_seed: None,
                score_initial_layout: true,
            },
            OptimizationLevel::Level3 => vf2::Vf2PassConfiguration {
                call_limit: (Some(30_000_000), Some(100_000)),
                time_limit: None,
                max_trials: None,
                shuffle_seed: None,
                score_initial_layout: true,
            },
        };

        if let vf2::Vf2PassReturn::Solution(layout) =
            vf2_layout_pass_exact(dag, target, &vf2_config)?
        {
            update_layout(dag, transpile_layout, |x| {
                Qubit(layout[&VirtualQubit(x.0)].0)
            });
        }
    }
    Ok(())
}

#[inline]
pub fn get_sabre_heuristic(target: &Target) -> Result<sabre::Heuristic> {
    Ok(sabre::Heuristic::new(
        None,
        None,
        None,
        target.num_qubits.map(|x| (x * 10) as usize),
        1e-10,
    )
    .with_basic(1.0, sabre::SetScaling::Constant)
    .with_lookahead(
        vec![0.5 / target.num_qubits.unwrap_or(20) as f64],
        sabre::SetScaling::Constant,
    )
    .with_decay(0.001, 5)?)
}

/// A transpilation function for Rust native circuits for use in the C API. This will not cover
/// things that only exist in the Python API such as custom gates or control flow. When those
/// concepts exist in the rust data model this function must be expanded before adding them to the
/// C API.
pub fn transpile(
    circuit: &CircuitData,
    target: &Target,
    optimization_level: OptimizationLevel,
    approximation_degree: Option<f64>,
    seed: Option<u64>,
) -> Result<(CircuitData, TranspileLayout)> {
    let mut dag = DAGCircuit::from_circuit_data(circuit, false, None, None)?;
    let mut commutation_checker = get_standard_commutation_checker();
    let mut equivalence_library = generate_standard_equivalence_library();
    let sabre_heuristic = get_sabre_heuristic(target)?;
    let mut synthesis_state = UnitarySynthesisState::new(UnitarySynthesisConfig {
        approximation: unitary_synthesis::Approximation::from_py_approximation_degree(
            approximation_degree,
        ),
        run_python_decomposers: false,
        ..Default::default()
    });

    let mut transpile_layout: TranspileLayout = TranspileLayout::new(
        None,
        None,
        dag.qubits().objects().to_owned(),
        dag.num_qubits() as u32,
        dag.qregs().to_vec(),
    );

    // Init stage
    init_stage(
        &mut dag,
        target,
        optimization_level,
        approximation_degree,
        &mut synthesis_state,
        &mut transpile_layout,
        &mut commutation_checker,
    )?;
    // layout stage
    let layout_source = layout_stage(
        &mut dag,
        target,
        optimization_level,
        seed,
        &sabre_heuristic,
        &mut transpile_layout,
    )?;
    // Routing stage
    routing_stage(
        &mut dag,
        target,
        optimization_level,
        seed,
        &sabre_heuristic,
        &mut transpile_layout,
        layout_source,
    )?;
    // Translation Stage
    translation_stage(
        &mut dag,
        target,
        &mut synthesis_state,
        &mut equivalence_library,
    )?;
    // optimization stage
    optimization_stage(
        &mut dag,
        target,
        optimization_level,
        approximation_degree,
        &mut synthesis_state,
        &mut commutation_checker,
        &mut equivalence_library,
        &mut transpile_layout,
    )?;
    Ok((CircuitData::from_dag_ref(&dag)?, transpile_layout))
}

struct MinPointState {
    best_depth: Option<usize>,
    best_size: Option<usize>,
    count: usize,
    best_dag: DAGCircuit,
}

impl MinPointState {
    fn new(dag: &DAGCircuit) -> Self {
        MinPointState {
            best_depth: None,
            best_size: None,
            count: 0,
            best_dag: dag.clone(),
        }
    }

    fn update_with(&mut self, dag: &DAGCircuit) -> bool {
        let new_depth = Some(dag.depth(false).unwrap());
        let new_size = Some(dag.size(false).unwrap());
        if self.best_depth.is_none() || self.best_size.is_none() {
            self.best_depth = new_depth;
            self.best_size = new_size;
            self.best_dag = dag.clone();
            true
        } else if (new_depth, new_size) > (self.best_depth, self.best_size) {
            self.count += 1;
            // Limit to 5 iterations without improvement
            // and back track to previous best
            self.count < 5
        } else if (new_depth, new_size) < (self.best_depth, self.best_size) {
            self.count = 1;
            self.best_depth = new_depth;
            self.best_size = new_size;
            self.best_dag = dag.clone();
            true
        } else {
            // When depth and size are unchanged, we reach a fixed point
            // and can stop the optimization loop
            false
        }
    }
}

fn layout_from_sabre_result(
    dag: &DAGCircuit,
    initial_layout: NLayout,
    final_layout: &NLayout,
    old_transpile_layout: &TranspileLayout,
) -> TranspileLayout {
    let mut new_transpile_layout = TranspileLayout::from_layouts(
        initial_layout,
        final_layout,
        dag.qubits().objects().clone(),
        old_transpile_layout.num_input_qubits(),
        dag.qregs().to_vec(),
    );
    if let Some(old_permutation) = old_transpile_layout.output_permutation() {
        new_transpile_layout
            .add_permutation_outside(|q| VirtualQubit(old_permutation[q.index()].0));
    }
    new_transpile_layout
}

#[cfg(all(test, not(miri)))]
mod tests {
    use super::*;
    use crate::target::InstructionProperties;
    use crate::target::Target;
    use qiskit_circuit::circuit_data::CircuitData;
    use qiskit_circuit::instruction::Parameters;
    use qiskit_circuit::operations::{Operation, Param, StandardGate, StandardInstruction};
    use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
    use qiskit_circuit::parameter::symbol_expr::Symbol;
    use qiskit_circuit::{Clbit, PhysicalQubit, Qubit};
    use smallvec::smallvec;
    use std::sync::Arc;

    fn build_universal_star_target() -> Target {
        let mut target = Target::default();
        let u_params = Some(Parameters::Params(smallvec![
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("a".to_owned(), None)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("b".to_owned(), None)
            ))),
            Param::ParameterExpression(Arc::new(ParameterExpression::from_symbol(
                Symbol::standalone("c".to_owned(), None)
            ))),
        ]));

        let props = (0..5)
            .map(|i| {
                (
                    [PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(i as f64 * 1.2e-6),
                        Some(i as f64 * 2.3e-5),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardGate::U.into(), u_params, None, Some(props))
            .unwrap();
        let props = (0..5)
            .map(|i| {
                (
                    [PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(i as f64 * 1.2e-4),
                        Some(i as f64 * 2.3e-3),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardInstruction::Measure.into(), None, None, Some(props))
            .unwrap();
        let props = (1..5)
            .map(|i| {
                (
                    [PhysicalQubit(0), PhysicalQubit(i)].into(),
                    Some(InstructionProperties::new(
                        Some(i as f64 * 2.4e-6),
                        Some(i as f64 * 6.2e-4),
                    )),
                )
            })
            .collect();
        target
            .add_instruction(StandardGate::ECR.into(), None, None, Some(props))
            .unwrap();
        target
    }

    #[test]
    fn test_bell_basic() {
        let target = build_universal_star_target();
        let qc = CircuitData::from_packed_operations(
            2,
            2,
            [
                Ok((StandardGate::H.into(), smallvec![], vec![Qubit(0)], vec![])),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(1)],
                    vec![],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(0)],
                    vec![Clbit(0)],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(1)],
                    vec![Clbit(1)],
                )),
            ],
            Param::Float(0.),
        )
        .unwrap();
        for opt_level in 0..=3 {
            let result = match transpile(&qc, &target, opt_level.into(), Some(1.0), Some(42)) {
                Ok(res) => res,
                Err(e) => panic!("Error: {}", e.backtrace()),
            };
            for inst in result.0.data() {
                if inst.op.num_qubits() == 2 {
                    assert_eq!("ecr", inst.op.name());
                    target.contains_qargs(
                        &result
                            .0
                            .get_qargs(inst.qubits)
                            .iter()
                            .map(|x| PhysicalQubit(x.0))
                            .collect::<Vec<_>>(),
                    );
                } else if inst.op.num_clbits() == 1 {
                    assert_eq!("measure", inst.op.name());
                } else {
                    assert_eq!("u", inst.op.name());
                }
            }
            assert!(result.1.output_permutation().is_none());
        }
    }

    #[test]
    fn test_routing_circuit() {
        let target = build_universal_star_target();
        let qc = CircuitData::from_packed_operations(
            5,
            5,
            [
                Ok((StandardGate::H.into(), smallvec![], vec![Qubit(0)], vec![])),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(1)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(2)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(3)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(0), Qubit(4)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(4), Qubit(1)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(4), Qubit(2)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(4), Qubit(3)],
                    vec![],
                )),
                Ok((
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![Qubit(4), Qubit(0)],
                    vec![],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(0)],
                    vec![Clbit(0)],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(1)],
                    vec![Clbit(1)],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(2)],
                    vec![Clbit(2)],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(3)],
                    vec![Clbit(3)],
                )),
                Ok((
                    StandardInstruction::Measure.into(),
                    smallvec![],
                    vec![Qubit(4)],
                    vec![Clbit(4)],
                )),
            ],
            Param::Float(0.),
        )
        .unwrap();
        for opt_level in 0..=3 {
            let result = match transpile(
                &qc,
                &target,
                opt_level.into(),
                Some(1.0),
                Some(2_025_120_142),
            ) {
                Ok(res) => res,
                Err(e) => panic!("Error: {}", e.backtrace()),
            };
            for inst in result.0.data() {
                if inst.op.num_qubits() == 2 {
                    assert_eq!("ecr", inst.op.name());
                    target.contains_qargs(
                        &result
                            .0
                            .get_qargs(inst.qubits)
                            .iter()
                            .map(|x| PhysicalQubit(x.0))
                            .collect::<Vec<_>>(),
                    );
                } else if inst.op.num_clbits() == 1 {
                    assert_eq!("measure", inst.op.name());
                } else {
                    assert_eq!("u", inst.op.name());
                }
            }
            assert!(result.1.output_permutation().is_some());
        }
    }

    #[test]
    fn test_update_best_dag() {
        let circuit1 = CircuitData::from_packed_operations(
            1,
            1,
            vec![Ok((
                StandardGate::H.into(),
                smallvec![],
                vec![Qubit(0)],
                vec![],
            ))],
            Param::Float(0.),
        )
        .unwrap();

        let dag1 = DAGCircuit::from_circuit_data(&circuit1, false, None, None).unwrap();
        let circuit2 = CircuitData::from_packed_operations(1, 1, vec![], Param::Float(0.)).unwrap();
        let dag2 = DAGCircuit::from_circuit_data(&circuit2, false, None, None).unwrap();

        let mut state = MinPointState::new(&dag1);
        assert!(state.update_with(&dag1));
        assert_eq!(state.count, 0);
        assert!(state.update_with(&dag2));
        assert_eq!(state.count, 1);
        assert_eq!(
            state.best_dag.depth(false).unwrap(),
            dag2.depth(false).unwrap()
        );
        assert_eq!(
            state.best_dag.size(false).unwrap(),
            dag2.size(false).unwrap()
        );
    }

    #[test]
    fn test_backtrack_limit_stops_loop() {
        let circuit1 = CircuitData::from_packed_operations(1, 1, vec![], Param::Float(0.)).unwrap();
        let dag1 = DAGCircuit::from_circuit_data(&circuit1, false, None, None).unwrap();
        let circuit2 = CircuitData::from_packed_operations(
            1,
            1,
            vec![Ok((
                StandardGate::H.into(),
                smallvec![],
                vec![Qubit(0)],
                vec![],
            ))],
            Param::Float(0.),
        )
        .unwrap();

        let dag2 = DAGCircuit::from_circuit_data(&circuit2, false, None, None).unwrap();
        let mut state = MinPointState::new(&dag1);

        state.update_with(&dag1);
        for i in 0..5 {
            let continue_loop = state.update_with(&dag2);
            if i < 4 {
                assert!(continue_loop);
            } else {
                assert!(!continue_loop);
            }
        }
    }

    #[test]
    fn test_backtrack_resets_on_improvement() {
        let circuit1 = CircuitData::from_packed_operations(
            1,
            1,
            vec![
                Ok((StandardGate::H.into(), smallvec![], vec![Qubit(0)], vec![])),
                Ok((StandardGate::H.into(), smallvec![], vec![Qubit(0)], vec![])),
            ],
            Param::Float(0.),
        )
        .unwrap();
        let dag_worst = DAGCircuit::from_circuit_data(&circuit1, false, None, None).unwrap();
        let circuit2 = CircuitData::from_packed_operations(
            1,
            1,
            vec![Ok((
                StandardGate::H.into(),
                smallvec![],
                vec![Qubit(0)],
                vec![],
            ))],
            Param::Float(0.),
        )
        .unwrap();
        let dag_better = DAGCircuit::from_circuit_data(&circuit2, false, None, None).unwrap();
        let circuit3 = CircuitData::from_packed_operations(1, 1, vec![], Param::Float(0.)).unwrap();
        let dag_best = DAGCircuit::from_circuit_data(&circuit3, false, None, None).unwrap();
        let mut state = MinPointState::new(&dag_worst);

        state.update_with(&dag_worst);
        state.update_with(&dag_better);
        for _i in 0..3 {
            state.update_with(&dag_worst);
        }
        state.update_with(&dag_best);
        assert_eq!(state.count, 1);
        // After updating to the dag_best the state tracked dag should
        // be empty
        assert_eq!(state.best_dag.depth(false).unwrap(), 0);
        assert_eq!(state.best_dag.size(false).unwrap(), 0);
    }
}
