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

mod impl_;

pub use crate::impl_::{ExportedFunction, ExportedFunctions};

// =================================================================================================
// WARNING
//
// Remember that the exact slot of a function is PUBLIC API and must remain stable between versions.
// =================================================================================================

// We use a (small) number of different tables here so there's more breaks and places to expand.
pub static FUNCTIONS_CIRCUIT: ExportedFunctions =
    ExportedFunctions::leaves(5, || vec![impl_::export_fn!(qiskit_cext::qk_api_version)])
        .add_child(5, &circuit::FUNCTIONS)
        .add_child(105, &dag::FUNCTIONS)
        .add_child(205, &param::FUNCTIONS)
        .add_child(255, &circuit_library::FUNCTIONS)
        .add_child(305, &classical_expr::FUNCTIONS);
pub static FUNCTIONS_QI: ExportedFunctions =
    ExportedFunctions::empty().add_child(0, &sparse_observable::FUNCTIONS);
pub use transpiler::FUNCTIONS as FUNCTIONS_TRANSPILE;

// Below this line is close to a mirror of the actual `cext` structure.  Ideally, all of the
// above exports would be locally within `cext` itself, but that has problems with needing to
// compile `cext` just to run the build script of things like `pyext`, which can end up in compiling
// the entire logic twice and needing to link `libpython` during the build.  This form lets us only
// _optionally_ depend on `cext`, which avoids those problems, at the cost of non-locality of the
// code.
//
// The module structure here should largely match what is in `cext`, but use your common sense - if
// `cext` just has a single file to encapsulate a small number of functions, you can inline it.  The
// idea is to make it easy to find things without too much additional boilerplate.

mod circuit {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::{circuit::*, control_flow::*};

    #[rustfmt::skip]  // Don't wrap long lines so everything stays on one line for counting.
    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(100, || {
        vec![
            export_fn!(qk_circuit_new),
            export_fn!(qk_quantum_register_new),
            export_fn!(qk_quantum_register_free),
            export_fn!(qk_classical_register_free),
            export_fn!(qk_classical_register_new),
            export_fn!(qk_circuit_add_quantum_register),
            export_fn!(qk_circuit_add_classical_register),
            export_fn!(qk_circuit_copy),
            export_fn!(qk_circuit_num_qubits),
            export_fn!(qk_circuit_num_clbits),
            export_fn!(qk_circuit_free),
            export_fn!(qk_circuit_gate),
            export_fn!(qk_gate_num_qubits),
            export_fn!(qk_gate_num_params),
            export_fn!(qk_circuit_measure),
            export_fn!(qk_circuit_reset),
            export_fn!(qk_circuit_barrier),
            export_fn!(qk_circuit_unitary),
            export_fn!(qk_circuit_inst_unitary),
            export_fn!(qk_circuit_pauli_product_rotation),
            export_fn!(qk_circuit_inst_pauli_product_rotation),
            export_fn!(qk_circuit_pauli_product_measurement),
            export_fn!(qk_circuit_inst_pauli_product_measurement),
            export_fn!(qk_circuit_instruction_kind),
            export_fn!(qk_circuit_count_ops),
            export_fn!(qk_circuit_num_instructions),
            export_fn!(qk_circuit_get_instruction),
            export_fn!(qk_circuit_instruction_clear),
            export_fn!(qk_opcounts_clear),
            export_fn!(qk_circuit_delay),
            export_fn!(qk_circuit_to_dag),
            export_fn!(qk_circuit_copy_empty_like),
            export_fn!(qk_circuit_num_param_symbols),
            export_fn!(qk_circuit_parameterized_gate),
            export_fn!(qk_circuit_to_python, feature = "python_binding"),
            export_fn!(qk_circuit_to_python_full, feature = "python_binding"),
            export_fn!(qk_circuit_borrow_from_python, feature = "python_binding"),
            export_fn!(qk_circuit_convert_from_python, feature = "python_binding"),
            export_fn!(qk_quantum_register_to_python, feature = "python_binding"),
            export_fn!(qk_quantum_register_borrow_from_python, feature = "python_binding"),
            export_fn!(qk_quantum_register_convert_from_python, feature = "python_binding"),
            export_fn!(qk_classical_register_to_python, feature = "python_binding"),
            export_fn!(qk_classical_register_borrow_from_python, feature = "python_binding"),
            export_fn!(qk_classical_register_convert_from_python, feature = "python_binding"),
            export_fn!(qk_circuit_draw),
            export_fn!(qk_circuit_global_phase),
            export_fn!(qk_circuit_set_global_phase),
            export_fn!(qk_circuit_estimate_fidelity),
            export_fn!(qk_quantum_register_name),
            export_fn!(qk_quantum_register_num_bits),
            export_fn!(qk_quantum_register_circuit_bits),
            export_fn!(qk_classical_register_name),
            export_fn!(qk_classical_register_num_bits),
            export_fn!(qk_classical_register_circuit_bits),
            export_fn!(qk_circuit_num_quantum_registers),
            export_fn!(qk_circuit_get_quantum_register),
            export_fn!(qk_circuit_num_classical_registers),
            export_fn!(qk_circuit_get_classical_register),
            export_fn!(qk_circuit_get_control_flow_instruction),
            export_fn!(qk_control_flow_instruction_free),
            export_fn!(qk_control_flow_kind),
            export_fn!(qk_control_flow_num_blocks),
            export_fn!(qk_control_flow_block_circuit),
            export_fn!(qk_control_flow_qubit_map),
            export_fn!(qk_control_flow_clbit_map),
            export_fn!(qk_control_flow_condition_type),
            export_fn!(qk_control_flow_condition_bit_info),
            export_fn!(qk_control_flow_condition_reg_cond_bit_width),
            export_fn!(qk_control_flow_condition_reg),
            export_fn!(qk_control_flow_condition_reg_cond_uint),
            export_fn!(qk_control_flow_condition_expr),
            export_fn!(qk_control_flow_box_duration_kind),
            export_fn!(qk_control_flow_box_duration_val_info),
            export_fn!(qk_control_flow_box_duration_expr),
            export_fn!(qk_control_flow_loop_collection_type),
            export_fn!(qk_control_flow_loop_elements),
            export_fn!(qk_control_flow_loop_range),
            export_fn!(qk_control_flow_loop_param_kind),
            export_fn!(qk_control_flow_loop_symbol_info),
            export_fn!(qk_control_flow_loop_variable),
            export_fn!(qk_control_flow_switch_target_type),
            export_fn!(qk_control_flow_switch_target_bit),
            export_fn!(qk_control_flow_switch_target_register),
            export_fn!(qk_control_flow_switch_target_expr),
            export_fn!(qk_control_flow_switch_num_cases),
            export_fn!(qk_control_flow_switch_is_case_default),
            export_fn!(qk_control_flow_switch_case_labels_bit_width),
            export_fn!(qk_control_flow_switch_case_labels_uint),
            export_fn!(qk_control_flow_switch_case_labels_clear),
        ]
    });
}

mod circuit_library {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::circuit_library::*;

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(50, || {
        vec![
            export_fn!(iqp::qk_circuit_library_iqp),
            export_fn!(iqp::qk_circuit_library_random_iqp),
            export_fn!(quantum_volume::qk_circuit_library_quantum_volume),
            export_fn!(suzuki_trotter::qk_circuit_library_suzuki_trotter),
            export_fn!(pbc::qk_pauli_product_rotation_clear),
            export_fn!(pbc::qk_pauli_product_measurement_clear),
        ]
    });
}

mod dag {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::dag::*;

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(100, || {
        vec![
            export_fn!(qk_dag_new),
            export_fn!(qk_dag_add_quantum_register),
            export_fn!(qk_dag_add_classical_register),
            export_fn!(qk_dag_num_qubits),
            export_fn!(qk_dag_num_clbits),
            export_fn!(qk_dag_num_op_nodes),
            export_fn!(qk_dag_node_type),
            export_fn!(qk_dag_qubit_in_node),
            export_fn!(qk_dag_qubit_out_node),
            export_fn!(qk_dag_clbit_in_node),
            export_fn!(qk_dag_clbit_out_node),
            export_fn!(qk_dag_wire_node_value),
            export_fn!(qk_dag_op_node_num_qubits),
            export_fn!(qk_dag_op_node_num_clbits),
            export_fn!(qk_dag_op_node_num_params),
            export_fn!(qk_dag_op_node_qubits),
            export_fn!(qk_dag_op_node_clbits),
            export_fn!(qk_dag_apply_gate),
            export_fn!(qk_dag_apply_measure),
            export_fn!(qk_dag_apply_reset),
            export_fn!(qk_dag_apply_barrier),
            export_fn!(qk_dag_apply_unitary),
            export_fn!(qk_dag_op_node_gate_op),
            export_fn!(qk_dag_op_node_unitary),
            export_fn!(qk_dag_op_node_kind),
            export_fn!(qk_dag_successors),
            export_fn!(qk_dag_predecessors),
            export_fn!(qk_dag_neighbors_clear),
            export_fn!(qk_dag_get_instruction),
            export_fn!(qk_dag_compose),
            export_fn!(qk_dag_free),
            export_fn!(qk_dag_to_circuit),
            export_fn!(qk_dag_topological_op_nodes),
            export_fn!(qk_dag_substitute_node_with_dag),
            export_fn!(qk_dag_copy_empty_like),
            export_fn!(qk_dag_to_python, feature = "python_binding"),
            export_fn!(qk_dag_borrow_from_python, feature = "python_binding"),
            export_fn!(qk_dag_convert_from_python, feature = "python_binding"),
            export_fn!(qk_dag_replace_block_with_unitary),
            export_fn!(qk_dag_substitute_node_with_unitary),
            export_fn!(qk_dag_global_phase),
            export_fn!(qk_dag_set_global_phase),
        ]
    });
}

mod param {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::param::*;

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(50, || {
        vec![
            export_fn!(qk_param_new_symbol),
            export_fn!(qk_param_zero),
            export_fn!(qk_param_free),
            export_fn!(qk_param_from_double),
            export_fn!(qk_param_from_complex),
            export_fn!(qk_param_copy),
            export_fn!(qk_param_str),
            export_fn!(qk_param_add),
            export_fn!(qk_param_sub),
            export_fn!(qk_param_mul),
            export_fn!(qk_param_div),
            export_fn!(qk_param_pow),
            export_fn!(qk_param_sin),
            export_fn!(qk_param_cos),
            export_fn!(qk_param_tan),
            export_fn!(qk_param_asin),
            export_fn!(qk_param_acos),
            export_fn!(qk_param_atan),
            export_fn!(qk_param_log),
            export_fn!(qk_param_exp),
            export_fn!(qk_param_abs),
            export_fn!(qk_param_sign),
            export_fn!(qk_param_neg),
            export_fn!(qk_param_conjugate),
            export_fn!(qk_param_equal),
            export_fn!(qk_param_as_real),
        ]
    });
}

mod sparse_observable {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::sparse_observable::*;

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(50, || {
        vec![
            export_fn!(qk_obs_zero),
            export_fn!(qk_obs_identity),
            export_fn!(qk_obs_new),
            export_fn!(qk_obs_free),
            export_fn!(qk_obs_add_term),
            export_fn!(qk_obs_term),
            export_fn!(qk_obs_num_terms),
            export_fn!(qk_obs_num_qubits),
            export_fn!(qk_obs_len),
            export_fn!(qk_obs_coeffs),
            export_fn!(qk_obs_indices),
            export_fn!(qk_obs_boundaries),
            export_fn!(qk_obs_bit_terms),
            export_fn!(qk_obs_multiply),
            export_fn!(qk_obs_multiply_inplace),
            export_fn!(qk_obs_add),
            export_fn!(qk_obs_add_inplace),
            export_fn!(qk_obs_scaled_add),
            export_fn!(qk_obs_scaled_add_inplace),
            export_fn!(qk_obs_compose),
            export_fn!(qk_obs_compose_map),
            export_fn!(qk_obs_apply_layout),
            export_fn!(qk_obs_canonicalize),
            export_fn!(qk_obs_copy),
            export_fn!(qk_obs_equal),
            export_fn!(qk_obs_str),
            export_fn!(qk_str_free),
            export_fn!(qk_obsterm_str),
            export_fn!(qk_bitterm_label),
            export_fn!(qk_obs_to_python, feature = "python_binding"),
            export_fn!(qk_obs_borrow_from_python, feature = "python_binding"),
            export_fn!(qk_obs_convert_from_python, feature = "python_binding"),
        ]
    });
}

mod transpiler {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::transpiler::{neighbors::*, transpile_function::*, transpile_layout::*};

    pub static TRANSPILE_FUNCTION: ExportedFunctions = ExportedFunctions::leaves(20, || {
        vec![
            export_fn!(qk_transpile),
            export_fn!(qk_transpiler_default_options),
            export_fn!(qk_transpile_stage_init),
            export_fn!(qk_transpile_stage_routing),
            export_fn!(qk_transpile_stage_optimization),
            export_fn!(qk_transpile_stage_translation),
            export_fn!(qk_transpile_stage_layout),
        ]
    });
    pub static NEIGHBORS: ExportedFunctions = ExportedFunctions::leaves(5, || {
        vec![
            export_fn!(qk_neighbors_is_all_to_all),
            export_fn!(qk_neighbors_from_target),
            export_fn!(qk_neighbors_clear),
        ]
    });
    pub static TRANSPILE_LAYOUT: ExportedFunctions = ExportedFunctions::leaves(15, || {
        vec![
            export_fn!(qk_transpile_layout_num_input_qubits),
            export_fn!(qk_transpile_layout_num_output_qubits),
            export_fn!(qk_transpile_layout_initial_layout),
            export_fn!(qk_transpile_layout_output_permutation),
            export_fn!(qk_transpile_layout_final_layout),
            export_fn!(qk_transpile_layout_generate_from_mapping),
            export_fn!(qk_transpile_layout_free),
            export_fn!(qk_transpile_layout_to_python, feature = "python_binding"),
        ]
    });
    pub static TRANSPILE_STATE: ExportedFunctions = ExportedFunctions::leaves(15, || {
        vec![
            export_fn!(qk_transpile_state_new),
            export_fn!(qk_transpile_state_free),
            export_fn!(qk_transpile_state_layout),
            export_fn!(qk_transpile_state_layout_set),
        ]
    });

    mod target {
        use crate::impl_::prelude::*;
        #[cfg(feature = "addr")]
        use qiskit_cext::transpiler::target::*;

        static FUNCTIONS_TARGET: ExportedFunctions = ExportedFunctions::leaves(50, || {
            vec![
                export_fn!(qk_target_new),
                export_fn!(qk_target_num_qubits),
                export_fn!(qk_target_dt),
                export_fn!(qk_target_set_dt),
                export_fn!(qk_target_granularity),
                export_fn!(qk_target_set_granularity),
                export_fn!(qk_target_min_length),
                export_fn!(qk_target_set_min_length),
                export_fn!(qk_target_pulse_alignment),
                export_fn!(qk_target_set_pulse_alignment),
                export_fn!(qk_target_acquire_alignment),
                export_fn!(qk_target_set_acquire_alignment),
                export_fn!(qk_target_copy),
                export_fn!(qk_target_free),
                export_fn!(qk_target_add_instruction),
                export_fn!(qk_target_update_property),
                export_fn!(qk_target_num_instructions),
                export_fn!(qk_target_instruction_supported),
                export_fn!(qk_target_op_index),
                export_fn!(qk_target_op_name),
                export_fn!(qk_target_op_num_properties),
                export_fn!(qk_target_op_qargs_index),
                export_fn!(qk_target_op_qargs),
                export_fn!(qk_target_op_props),
                export_fn!(qk_target_op_get),
                export_fn!(qk_target_op_gate),
                export_fn!(qk_target_op_clear),
                export_fn!(qk_target_borrow_from_python, feature = "python_binding"),
                export_fn!(qk_target_convert_from_python, feature = "python_binding"),
            ]
        });
        static FUNCTIONS_TARGET_ENTRY: ExportedFunctions = ExportedFunctions::leaves(20, || {
            vec![
                export_fn!(qk_target_entry_new),
                export_fn!(qk_target_entry_new_measure),
                export_fn!(qk_target_entry_new_reset),
                export_fn!(qk_target_entry_new_fixed),
                export_fn!(qk_target_entry_num_properties),
                export_fn!(qk_target_entry_free),
                export_fn!(qk_target_entry_add_property),
                export_fn!(qk_target_entry_set_name),
            ]
        });
        pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::empty()
            .add_child(0, &FUNCTIONS_TARGET)
            .add_child(50, &FUNCTIONS_TARGET_ENTRY);
    }

    mod passes {
        use crate::impl_::prelude::*;
        #[cfg(feature = "addr")]
        use qiskit_cext::transpiler::passes::*;

        static FUNCTIONS_PASSES: ExportedFunctions = ExportedFunctions::leaves(50, || {
            vec![
                export_fn!(elide_permutations::qk_transpiler_pass_elide_permutations),
                export_fn!(gate_direction::qk_transpiler_pass_check_gate_direction),
                export_fn!(gate_direction::qk_transpiler_pass_gate_direction),
                export_fn!(optimize_1q_sequences::qk_transpiler_pass_optimize_1q_sequences),
                export_fn!(remove_diagonal_gates_before_measure::qk_transpiler_pass_remove_diagonal_gates_before_measure),
                export_fn!(remove_identity_equiv::qk_transpiler_pass_remove_identity_equivalent),
                export_fn!(split_2q_unitaries::qk_transpiler_pass_split_2q_unitaries),
                export_fn!(two_qubit_peephole::qk_transpiler_pass_2q_peephole_optimization),
            ]
        });
        static FUNCTIONS_STANDALONE: ExportedFunctions = ExportedFunctions::leaves(50, || {
            vec![
                export_fn!(basis_translator::qk_transpiler_pass_standalone_basis_translator),
                export_fn!(commutative_cancellation::qk_transpiler_pass_standalone_commutative_cancellation),
                export_fn!(consolidate_blocks::qk_transpiler_pass_standalone_consolidate_blocks),
                export_fn!(elide_permutations::qk_transpiler_pass_standalone_elide_permutations),
                export_fn!(gate_direction::qk_transpiler_pass_standalone_check_gate_direction),
                export_fn!(gate_direction::qk_transpiler_pass_standalone_gate_direction),
                export_fn!(inverse_cancellation::qk_transpiler_pass_standalone_inverse_cancellation),
                export_fn!(optimize_1q_sequences::qk_transpiler_pass_standalone_optimize_1q_sequences),
                export_fn!(remove_diagonal_gates_before_measure::qk_transpiler_pass_standalone_remove_diagonal_gates_before_measure),
                export_fn!(remove_identity_equiv::qk_transpiler_pass_standalone_remove_identity_equivalent),
                export_fn!(sabre_layout::qk_transpiler_pass_standalone_sabre_layout),
                export_fn!(split_2q_unitaries::qk_transpiler_pass_standalone_split_2q_unitaries),
                export_fn!(unitary_synthesis::qk_transpiler_pass_standalone_unitary_synthesis),
                export_fn!(vf2::qk_transpiler_pass_standalone_vf2_layout_average),
                export_fn!(vf2::qk_transpiler_pass_standalone_vf2_layout_exact),
                export_fn!(convert_to_pauli_rotations::qk_transpiler_pass_standalone_convert_to_pauli_rotations),
                export_fn!(litinski_transformation::qk_transpiler_pass_standalone_litinski_transformation),
                export_fn!(two_qubit_peephole::qk_transpiler_pass_standalone_2q_peephole_optimization),
            ]
        });
        static FUNCTIONS_SABRE: ExportedFunctions = ExportedFunctions::leaves(5, || {
            vec![export_fn!(sabre_layout::qk_sabre_layout_options_default)]
        });
        static FUNCTIONS_VF2: ExportedFunctions = ExportedFunctions::leaves(20, || {
            vec![
                export_fn!(vf2::qk_vf2_layout_result_has_match),
                export_fn!(vf2::qk_vf2_layout_result_has_improvement),
                export_fn!(vf2::qk_vf2_layout_result_map_virtual_qubit),
                export_fn!(vf2::qk_vf2_layout_result_free),
                export_fn!(vf2::qk_vf2_layout_configuration_new),
                export_fn!(vf2::qk_vf2_layout_configuration_free),
                export_fn!(vf2::qk_vf2_layout_configuration_set_call_limit),
                export_fn!(vf2::qk_vf2_layout_configuration_set_time_limit),
                export_fn!(vf2::qk_vf2_layout_configuration_set_max_trials),
                export_fn!(vf2::qk_vf2_layout_configuration_set_shuffle_seed),
                export_fn!(vf2::qk_vf2_layout_configuration_set_score_initial),
            ]
        });

        pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::empty()
            .add_child(0, &FUNCTIONS_PASSES)
            .add_child(100, &FUNCTIONS_STANDALONE)
            .add_child(200, &FUNCTIONS_SABRE)
            .add_child(205, &FUNCTIONS_VF2);
    }

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::empty()
        .add_child(0, &TRANSPILE_FUNCTION)
        .add_child(20, &NEIGHBORS)
        .add_child(35, &TRANSPILE_LAYOUT)
        .add_child(50, &TRANSPILE_STATE)
        .add_child(150, &target::FUNCTIONS)
        .add_child(250, &passes::FUNCTIONS);
}

mod classical_expr {
    use crate::impl_::prelude::*;
    #[cfg(feature = "addr")]
    use qiskit_cext::classical_expr::*;

    pub static FUNCTIONS: ExportedFunctions = ExportedFunctions::leaves(50, || {
        vec![
            export_fn!(qk_expr_kind),
            export_fn!(qk_expr_binary_info),
            export_fn!(qk_expr_unary_info),
            export_fn!(qk_expr_cast_info),
            export_fn!(qk_expr_index_info),
            export_fn!(qk_expr_as_value),
            export_fn!(qk_expr_as_var),
            export_fn!(qk_expr_as_stretch),
            export_fn!(qk_value_type_info),
            export_fn!(qk_value_duration_info),
            export_fn!(qk_value_float),
            export_fn!(qk_value_uint),
            export_fn!(qk_value_bool),
            export_fn!(qk_var_name),
            export_fn!(qk_var_type_info),
            export_fn!(qk_stretch_name),
        ]
    });
}
