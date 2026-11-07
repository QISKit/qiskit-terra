// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;

mod capi;
mod mir;
mod passmanager;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[inline(always)]
#[doc(hidden)]
fn add_submodule<F>(m: &Bound<PyModule>, constructor: F, name: &str) -> PyResult<()>
where
    F: FnOnce(&Bound<PyModule>) -> PyResult<()>,
{
    let new_mod = PyModule::new(m.py(), name)?;
    constructor(&new_mod)?;
    m.add_submodule(&new_mod)
}

// Formatting is off here so every module import can just be a single line in sorted order, to help
// avoid merge conflicts as modules are added.
#[rustfmt::skip]
#[pymodule]
fn _accelerate(m: &Bound<PyModule>) -> PyResult<()> {
    add_submodule(m, capi::capi_mod, "capi")?;
    add_submodule(m, passmanager::passmanager_mod, "passmanager")?;
    add_submodule(m, mir::mir_mod, "mir")?;
    add_submodule(m, ::qiskit_transpiler::passes::alap_schedule_analysis_mod, "alap_schedule_analysis")?;
    add_submodule(m, ::qiskit_transpiler::passes::asap_schedule_analysis_mod, "asap_schedule_analysis")?;
    add_submodule(m, ::qiskit_transpiler::passes::apply_layout_mod, "apply_layout")?;
    add_submodule(m, ::qiskit_transpiler::passes::barrier_before_final_measurements_mod, "barrier_before_final_measurement")?;
    add_submodule(m, ::qiskit_transpiler::passes::basis_translator_mod, "basis_translator")?;
    add_submodule(m, ::qiskit_transpiler::passes::check_map_mod, "check_map")?;
    add_submodule(m, ::qiskit_accelerate::circuit_duration::compute_duration, "circuit_duration")?;
    add_submodule(m, ::qiskit_circuit_library::circuit_library, "circuit_library")?;
    add_submodule(m, ::qiskit_transpiler::passes::commutation_analysis_mod, "commutation_analysis")?;
    add_submodule(m, ::qiskit_transpiler::passes::commutation_cancellation_mod, "commutation_cancellation")?;
    add_submodule(m, ::qiskit_transpiler::commutation_checker::commutation_checker, "commutation_checker")?;
    add_submodule(m, ::qiskit_transpiler::passes::commutative_optimization_mod, "commutative_optimization")?;
    add_submodule(m, ::qiskit_transpiler::passes::consolidate_blocks_mod, "consolidate_blocks")?;
    add_submodule(m, ::qiskit_transpiler::passes::constrained_reschedule_mod, "constrained_reschedule")?;
    add_submodule(m, ::qiskit_synthesis::linalg::cos_sin_decomp::cos_sin_decomp, "cos_sin_decomp")?;
    add_submodule(m, ::qiskit_transpiler::passes::dense_layout_mod, "dense_layout")?;
    add_submodule(m, ::qiskit_transpiler::equivalence::equivalence, "equivalence")?;
    add_submodule(m, ::qiskit_transpiler::passes::error_map_mod, "error_map")?;
    add_submodule(m, ::qiskit_transpiler::passes::elide_permutations_mod, "elide_permutations")?;
    add_submodule(m, ::qiskit_transpiler::passes::litinski_transformation_mod, "litinski_transformation")?;
    add_submodule(m, ::qiskit_synthesis::euler_one_qubit_decomposer::euler_one_qubit_decomposer, "euler_one_qubit_decomposer")?;
    add_submodule(m, ::qiskit_transpiler::passes::disjoint_utils_mod, "disjoint_utils")?;
    add_submodule(m, ::qiskit_transpiler::passes::filter_op_nodes_mod, "filter_op_nodes")?;
    add_submodule(m, ::qiskit_transpiler::passes::gate_direction_mod, "gate_direction")?;
    add_submodule(m, ::qiskit_transpiler::passes::gates_in_basis_mod, "gates_in_basis")?;
    add_submodule(m, ::qiskit_transpiler::passes::instruction_duration_check_mod, "instruction_duration_check")?;
    add_submodule(m, ::qiskit_transpiler::passes::inverse_cancellation_mod, "inverse_cancellation")?;
    add_submodule(m, ::qiskit_accelerate::isometry::isometry, "isometry")?;
    add_submodule(m, ::qiskit_circuit::nlayout::nlayout, "nlayout")?;
    add_submodule(m, ::qiskit_circuit::standard_gate::standard_generators::standard_generators, "standard_generators")?;
    add_submodule(m, ::qiskit_accelerate::optimize_1q_gates::optimize_1q_gates, "optimize_1q_gates")?;
    add_submodule(m, ::qiskit_transpiler::passes::optimize_1q_gates_decomposition_mod, "optimize_1q_gates_decomposition")?;
    add_submodule(m, ::qiskit_accelerate::pauli_exp_val::pauli_expval, "pauli_expval")?;
    add_submodule(m, ::qiskit_quantum_info::pauli_lindblad_map::pauli_lindblad_map, "pauli_lindblad_map")?;
    add_submodule(m, ::qiskit_transpiler::passes::high_level_synthesis_mod, "high_level_synthesis")?;
    add_submodule(m, ::qiskit_transpiler::passes::remove_diagonal_gates_before_measure_mod, "remove_diagonal_gates_before_measure")?;
    add_submodule(m, ::qiskit_transpiler::passes::remove_identity_equiv_mod, "remove_identity_equiv")?;
    add_submodule(
        m,
        ::qiskit_transpiler::passes::two_qubit_peephole_mod,
        "two_qubit_peephole",
    )?;
    add_submodule(m, ::qiskit_accelerate::results::results, "results")?;
    add_submodule(m, ::qiskit_transpiler::passes::sabre::sabre, "sabre")?;
    add_submodule(m, ::qiskit_accelerate::sampled_exp_val::sampled_exp_val, "sampled_exp_val")?;
    add_submodule(m, ::qiskit_quantum_info::sparse_observable::sparse_observable, "sparse_observable")?;
    add_submodule(m, ::qiskit_quantum_info::sparse_pauli_op::sparse_pauli_op, "sparse_pauli_op")?;
    add_submodule(m, ::qiskit_transpiler::passes::scheduling_mod, "scheduling")?;
    add_submodule(m, ::qiskit_synthesis::matrix::sim::unitary_sim, "unitary_sim")?;
    add_submodule(m, ::qiskit_transpiler::passes::split_2q_unitaries_mod, "split_2q_unitaries")?;
    add_submodule(m, ::qiskit_synthesis::synthesis, "synthesis")?;
    add_submodule(m, ::qiskit_transpiler::target::target, "target")?;
    add_submodule(m, ::qiskit_accelerate::twirling::twirling, "twirling")?;
    add_submodule(m, ::qiskit_synthesis::two_qubit_decompose::two_qubit_decompose, "two_qubit_decompose")?;
    add_submodule(m, ::qiskit_synthesis::pauli_products::pauli_products_mod, "pauli_products")?;
    add_submodule(m, ::qiskit_transpiler::passes::unitary_synthesis_mod, "unitary_synthesis")?;
    add_submodule(m, ::qiskit_transpiler::passes::unroll_3q_or_more_mod, "unroll_3q_or_more")?;
    add_submodule(m, ::qiskit_transpiler::passes::vf2_layout_mod, "vf2_layout")?;
    add_submodule(m, ::qiskit_circuit::circuit, "circuit")?;
    add_submodule(m, ::qiskit_circuit::converters::converters, "converters")?;
    add_submodule(m, ::qiskit_qasm2::qasm2, "qasm2")?;
    add_submodule(m, ::qiskit_qasm3::qasm3, "qasm3")?;
    add_submodule(m, ::qiskit_qpy::qpy, "qpy")?;
    add_submodule(m, ::qiskit_synthesis::qsd::qsd_mod, "qsd")?;
    add_submodule(m, ::qiskit_synthesis::ross_selinger::ross_selinger_mod, "ross_selinger")?;
    add_submodule(m, ::qiskit_transpiler::angle_bound_registry::angle_bound_mod, "angle_bound_registry")?;
    add_submodule(m, ::qiskit_transpiler::passes::wrap_angles_mod, "wrap_angles")?;
    add_submodule(m, ::qiskit_transpiler::passes::optimize_clifford_t_mod, "optimize_clifford_t")?;
    add_submodule(m, ::qiskit_transpiler::passes::substitute_pi4_rotations_mod, "substitute_pi4_rotations")?;
    add_submodule(m, ::qiskit_transpiler::passes::synthesize_rz_rotations_mod, "synthesize_rz_rotations")?;

    add_submodule(m, ::qiskit_transpiler::passes::convert_to_pauli_rotations_mod, "convert_to_pauli_rotations")?;
    Ok(())
}
