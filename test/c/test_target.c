// This code is part of Qiskit.
//
// (C) Copyright IBM 2025.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include "common.h"
#include <complex.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

QkTarget *create_sample_target(bool std_inst) {
    // Build sample target
    QkTarget *target = qk_target_new(0);
    QkTargetEntry *i_entry = qk_target_entry_new(QkGate_I);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(i_entry, qargs, 1, 35.5e-9, 0.);
    }
    qk_target_add_instruction(target, i_entry);

    double rz_params[1] = {3.14};
    QkTargetEntry *rz_entry = qk_target_entry_new_fixed(QkGate_RZ, rz_params, "rz_pi");
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(rz_entry, qargs, 1, 0., 0.);
    }
    qk_target_add_instruction(target, rz_entry);

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(sx_entry, qargs, 1, 35.5e-9, 0.);
    }
    qk_target_add_instruction(target, sx_entry);

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (int i = 0; i < 4; i++) {
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(x_entry, qargs, 1, 35.5e-9, 0.0005);
    }
    qk_target_add_instruction(target, x_entry);

    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    double props[8][2] = {
        {2.7022e-11, 0.00713}, {3.0577e-11, 0.00713}, {4.6222e-11, 0.00929}, {4.9777e-11, 0.00929},
        {2.2755e-11, 0.00659}, {2.6311e-11, 0.00659}, {5.1911e-11, 0.01201}, {5.1911e-11, 0.01201},
    };
    for (int i = 0; i < 8; i++) {
        qk_target_entry_add_property(cx_entry, qarg_samples[i], 2, props[i][0], props[i][1]);
    }
    qk_target_add_instruction(target, cx_entry);

    // Add global Y Gate
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Y));

    // Add glbal phase gate
    QkTargetEntry *gp_entry = qk_target_entry_new(QkGate_GlobalPhase);
    qk_target_entry_add_property(gp_entry, (uint32_t *)4, 0, NAN, NAN);
    qk_target_add_instruction(target, gp_entry);

    if (std_inst) {
        QkTargetEntry *meas = qk_target_entry_new_measure();
        for (uint32_t i = 0; i < 2; i++) {
            uint32_t q[1] = {i};
            qk_target_entry_add_property(meas, q, 1, 1e-6, 1e-4);
        }
        qk_target_add_instruction(target, meas);

        QkTargetEntry *reset = qk_target_entry_new_reset();
        for (uint32_t i = 0; i < 4; i++) {
            uint32_t q[1] = {i};
            qk_target_entry_add_property(reset, q, 1, 1e-6, 1e-4);
        }
        qk_target_add_instruction(target, reset);
    }

    return target;
}

bool compare_qargs(uint32_t *lhs, uint32_t *rhs, size_t length) {
    if (lhs == NULL && rhs != NULL) {
        return false;
    }
    if (rhs == NULL && lhs != NULL) {
        return false;
    }

    return !memcmp(lhs, rhs, length * sizeof(*lhs));
}

void print_qargs(uint32_t *qargs, size_t length) {
    if (qargs == NULL) {
        fprintf(stderr, "Global");
    } else {
        fprintf(stderr, "[");
        for (uint32_t q_idx = 0; q_idx < length; q_idx++) {
            if (q_idx < length - 1) {
                fprintf(stderr, "%u, ", qargs[q_idx]);
            } else {
                fprintf(stderr, "%u", qargs[q_idx]);
            }
        }
        fprintf(stderr, "]");
    }
}

/**
 * Test empty constructor for Target
 */
static int test_empty_target(void) {
    int result = Ok;
    QkTarget *target = qk_target_new(0);
    uint32_t num_qubits = qk_target_num_qubits(target);

    if (num_qubits != 0) {
        fprintf(stderr, "The number of qubits %u is not 0.", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    double retrieved_dt = qk_target_dt(target);
    if (!isnan(retrieved_dt)) {
        fprintf(stderr, "The dt value of this target %f is not %f.", retrieved_dt, NAN);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 1) {
        fprintf(stderr, "The granularity %u is not 1.", retrieved_granularity);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 1) {
        fprintf(stderr, "The min_length %u is not 1.", retrieved_min_length);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 1) {
        fprintf(stderr, "The pulse_alignment values %u is not 1.", pulse_alignment);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 1) {
        fprintf(stderr, "The acquire_alignment values %u is not 1.", acquire_alignment);
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test constructor for Target
 */
static int test_target_construct(void) {
    int result = Ok;
    const uint32_t num_qubits = 2;
    const double dt = 10e-9;
    const uint32_t granularity = 2;
    const uint32_t min_length = 3;
    const uint32_t p_alignment = 4;
    const uint32_t a_alignment = 1;

    QkTarget *target = qk_target_new(num_qubits);

    qk_target_set_dt(target, dt);
    qk_target_set_granularity(target, granularity);
    qk_target_set_min_length(target, min_length);
    qk_target_set_pulse_alignment(target, p_alignment);
    qk_target_set_acquire_alignment(target, a_alignment);

    uint32_t retrieved_num_qubits = qk_target_num_qubits(target);
    if (retrieved_num_qubits != 2) {
        fprintf(stderr, "The number of qubits %u is not 0.", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    double retrieved_dt = qk_target_dt(target);
    if (retrieved_dt != dt) {
        fprintf(stderr, "The dt value of this target %f is not %f.", retrieved_dt, dt);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_granularity = qk_target_granularity(target);
    if (retrieved_granularity != 2) {
        fprintf(stderr, "The granularity %u is not 2.", retrieved_granularity);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t retrieved_min_length = qk_target_min_length(target);
    if (retrieved_min_length != 3) {
        fprintf(stderr, "The min_length %u is not 3.", retrieved_min_length);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t pulse_alignment = qk_target_pulse_alignment(target);
    if (pulse_alignment != 4) {
        fprintf(stderr, "The pulse_alignment values %u is not 4.", pulse_alignment);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t acquire_alignment = qk_target_acquire_alignment(target);
    if (acquire_alignment != 1) {
        fprintf(stderr, "The acquire_alignment values %u is not 1.", acquire_alignment);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

/*
 * Test target construction with a parameterized gate
 */
static int test_target_construction_ibm_like_target(void) {
    int result = Ok;
    QkTarget *target = qk_target_new(5);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t cx_qargs[2] = {0, 1};
    QkExitCode result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 2.2e-4, 6.2e-9);
    if (result_prop != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[0] = 2;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 3.2e-4, 4.2e-9);
    if (result_prop != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[1] = 3;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 1.2e-4, 4.2e-8);
    if (result_prop != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    cx_qargs[0] = 4;
    result_prop = qk_target_entry_add_property(cx_entry, cx_qargs, 2, 1.2e-3, 2.2e-8);
    if (result_prop != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding entry.");
        qk_target_entry_free(cx_entry);
        result = EqualityError;
        goto cleanup;
    }
    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *rz_entry = qk_target_entry_new(QkGate_RZ);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(rz_entry, qargs, 1, 0, 0);
        if (result_prop != QkExitCode_Success) {
            fprintf(stderr, "Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(rz_entry);
            goto cleanup;
        }
    }
    QkExitCode result_rz = qk_target_add_instruction(target, rz_entry);
    if (result_rz != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(sx_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            fprintf(stderr, "Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(sx_entry);
            goto cleanup;
        }
    }
    QkExitCode result_sx = qk_target_add_instruction(target, sx_entry);
    if (result_sx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(x_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            fprintf(stderr, "Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(x_entry);
            goto cleanup;
        }
    }
    QkExitCode result_x = qk_target_add_instruction(target, x_entry);
    if (result_x != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

    QkTargetEntry *measure_entry = qk_target_entry_new_measure();
    for (uint32_t i = 0; i < 5; i++) {
        uint32_t qargs[1] = {i};
        result_prop = qk_target_entry_add_property(measure_entry, qargs, 1, 1.928e-10, 7.9829e-11);
        if (result_prop != QkExitCode_Success) {
            fprintf(stderr, "Unexpected error occurred when adding entry.");
            result = EqualityError;
            qk_target_entry_free(measure_entry);
            goto cleanup;
        }
    }
    QkExitCode result_measure = qk_target_add_instruction(target, measure_entry);
    if (result_measure != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a parameterized RZ gate.");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test construction of a QkTargetEntry
 */
static int test_target_entry_construction(void) {
    int result = Ok;
    QkTargetEntry *property_map = qk_target_entry_new(QkGate_CX);

    // Test length
    const size_t length = qk_target_entry_num_properties(property_map);
    if (length != 0) {
        fprintf(stderr, "The initial length of the provided property map was not zero: %zu",
                length);
        result = EqualityError;
        goto cleanup;
    }

    // Add some qargs and properties
    uint32_t qargs[2] = {0, 1};

    QkExitCode result_prop = qk_target_entry_add_property(property_map, qargs, 2, 0.00018, 0.00002);
    if (result_prop != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding entry.");
    }

    // Test length
    const size_t new_length = qk_target_entry_num_properties(property_map);
    if (new_length != 1) {
        fprintf(stderr, "The initial length of the provided property map was not 1: %zu", length);
        result = EqualityError;
        goto cleanup;
    }

    // Add invalid qargs
    // Add some qargs and properties
    uint32_t invalid_qargs[3] = {0, 1, 2};

    QkExitCode error_result =
        qk_target_entry_add_property(property_map, invalid_qargs, 3, 0.00018, 0.00002);
    if (error_result != QkExitCode_TargetQargMismatch) {
        fprintf(stderr, "The operation did not fail as expected for invalid qargs.");
    }

    // Set name for the entry
    QkExitCode set_name_error = qk_target_entry_set_name(property_map, "cx_gate");
    if (set_name_error != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when setting the name of the entry.\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_entry_free(property_map);
    return result;
}

/**
 * Test adding an instruction to the Target.
 */
static int test_target_add_instruction(void) {
    const uint32_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    // Add an X Gate.
    // This operation is global, no property map is provided
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a global X gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Re-add same gate, check if it fails
    QkExitCode result_x_readded = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x_readded != QkExitCode_TargetInstAlreadyExists) {
        fprintf(stderr, "The addition of a repeated gate did not fail as expected.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should not change.
    uint32_t current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 1) {
        fprintf(stderr, "The number of qubits this target is compatible with is not 1: %u",
                current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    size_t current_size = qk_target_num_instructions(target);
    if (current_size != 1) {
        fprintf(stderr, "The size of this target is not correct: Expected 1, got %zu",
                current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;

    QkExitCode result_cx_props =
        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    if (result_cx_props != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding property to a CX gate entry.");
        result = EqualityError;
        goto cleanup;
    }

    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should change to 2.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 2) {
        fprintf(stderr, "The number of qubits this target is compatible with is not 2: %u",
                current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_num_instructions(target);
    if (current_size != 2) {
        fprintf(stderr, "The size of this target is not correct: Expected 2, got %zu",
                current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a CRX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    double crx_params[1] = {3.14};
    QkTargetEntry *crx_entry = qk_target_entry_new_fixed(QkGate_CRX, crx_params, "crx_fixed");
    uint32_t crx_qargs[2] = {1, 2};
    double crx_inst_error = 0.0129023;
    double crx_inst_duration = 0.92939;
    QkExitCode result_crx_props =
        qk_target_entry_add_property(crx_entry, crx_qargs, 2, crx_inst_duration, crx_inst_error);
    if (result_crx_props != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding property to a CX gate entry.");
        result = EqualityError;
        goto cleanup;
    }

    QkExitCode result_crx = qk_target_add_instruction(target, crx_entry);
    if (result_crx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a CX gate.");
        result = EqualityError;
        goto cleanup;
    }

    // Number of qubits of the target should change to 3.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 3) {
        fprintf(stderr, "The number of qubits this target is compatible with is not 3: %d",
                current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }
    current_size = qk_target_num_instructions(target);
    if (current_size != 3) {
        fprintf(stderr, "The size of this target is not correct: Expected 3, got %zu",
                current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a measurement
    QkTargetEntry *meas = qk_target_entry_new_measure();
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t q[1] = {i};
        qk_target_entry_add_property(meas, q, 1, 1e-6, 1e-4);
    }
    size_t num_meas = qk_target_entry_num_properties(meas);
    if (num_meas != 3) {
        fprintf(stderr, "Expected 3 measurement entries but got: %zu", num_meas);
        result = EqualityError;
        qk_target_entry_free(meas);
        goto cleanup;
    }

    QkExitCode result_meas_props = qk_target_add_instruction(target, meas);
    if (result_meas_props != 0) {
        fprintf(stderr, "Failed adding measurement instruction.");
        result = EqualityError;
        goto cleanup;
    }
    // Number of qubits of the target should remain 3.
    current_num_qubits = qk_target_num_qubits(target);
    if (current_num_qubits != 3) {
        fprintf(stderr, "The number of qubits this target is compatible with is not 3: %d",
                current_num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    current_size = qk_target_num_instructions(target);
    if (current_size != 4) {
        fprintf(stderr, "The size of this target is not correct: Expected 4, got %zu",
                current_size);
        result = EqualityError;
        goto cleanup;
    }

    // Add a reset
    QkTargetEntry *reset = qk_target_entry_new_reset();
    for (uint32_t i = 0; i < 3; i++) {
        uint32_t q[1] = {i};
        qk_target_entry_add_property(reset, q, 1, 2e-6, 2e-4);
    }
    size_t num_reset = qk_target_entry_num_properties(reset);
    if (num_reset != 3) {
        fprintf(stderr, "Expected 3 reset entries but got: %zu", num_reset);
        result = EqualityError;
        qk_target_entry_free(reset);
        goto cleanup;
    }

    qk_target_add_instruction(target, reset);
    current_size = qk_target_num_instructions(target);
    if (current_size != 5) {
        fprintf(stderr, "The size of this target is not correct: Expected 5, got %zu",
                current_size);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test updating an instruction property in the Target using
 * `update_instruction_property`.
 */
static int test_target_update_instruction(void) {
    const uint32_t num_qubits = 1;
    // Let's create a target with one qubit for now
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    // Add a CX Gate.
    // Create prop_map for the instruction
    // Add property for (0, 1)
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    uint32_t qargs[2] = {0, 1};
    double inst_error = 0.0090393;
    double inst_duration = 0.020039;
    qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    // CX Gate is not parametric. Re-use Null
    qk_target_add_instruction(target, cx_entry);

    // Change the instruction property of cx
    double cx_new_inst_error = NAN;
    double cx_new_inst_duration = 0.09457;
    QkExitCode result_1 = qk_target_update_property(target, QkGate_CX, qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_1 != QkExitCode_Success) {
        fprintf(stderr, "An unexpected error occurred while modifying the property.");
        result = RuntimeError;
        goto cleanup;
    }

    // Try to modify wrong instruction
    QkExitCode result_2 = qk_target_update_property(target, QkGate_CH, qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_2 != QkExitCode_TargetInvalidInstKey) {
        fprintf(stderr,
                "The function did not fail as expected when querying the wrong instruction.");
        result = RuntimeError;
        goto cleanup;
    }

    uint32_t new_qargs[2] = {1, 2};
    // Try to modify wrong qargs
    QkExitCode result_3 = qk_target_update_property(target, QkGate_CX, new_qargs, 2,
                                                    cx_new_inst_duration, cx_new_inst_error);
    if (result_3 != QkExitCode_TargetInvalidQargsKey) {
        fprintf(stderr, "The function did not fail as expected when querying with wrong qargs.");
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    return result;
}

static int test_target_iteration(void) {
    QkTarget *target = create_sample_target(true);
    int result = Ok;
    char *names[9] = {"id", "rz_pi", "sx", "x", "cx", "y", "global_phase", "measure", "reset"};
    uint32_t single_qarg[4] = {0, 1, 2, 3};
    uint32_t cx_qargs[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    double cx_props[8][2] = {
        {2.7022e-11, 0.00713}, {3.0577e-11, 0.00713}, {4.6222e-11, 0.00929}, {4.9777e-11, 0.00929},
        {2.2755e-11, 0.00659}, {2.6311e-11, 0.00659}, {5.1911e-11, 0.01201}, {5.1911e-11, 0.01201},
    };

    size_t target_length = qk_target_num_instructions(target);
    for (size_t op_idx = 0; op_idx < target_length; op_idx++) {
        uint32_t *qargs;
        uint32_t qargs_len;
        QkInstructionProperties props;

        char *name = qk_target_op_name(target, op_idx);
        if (strcmp(name, names[(int)op_idx]) != 0) {
            fprintf(stderr, "Unexpected operation name at index %zu, expected: %s, got %s.\n",
                    op_idx, names[(int)op_idx], name);
            result = EqualityError;
            goto cleanup;
        }

        size_t num_props = qk_target_op_num_properties(target, op_idx);
        for (size_t props_idx = 0; props_idx < num_props; props_idx++) {
            qk_target_op_qargs(target, op_idx, props_idx, &qargs, &qargs_len);
            qk_target_op_props(target, op_idx, props_idx, &props);

            // cx
            if (op_idx == 4) {
                if (!compare_qargs(qargs, cx_qargs[props_idx], qargs_len)) {
                    fprintf(
                        stderr,
                        "Unexpected qargs found for operation %s at qarg index: %zu. Expected: '",
                        name, props_idx);
                    print_qargs(cx_qargs[props_idx], qargs_len);
                    fprintf(stderr, "', found '");
                    print_qargs(qargs, qargs_len);
                    fprintf(stderr, "'\n");
                    result = EqualityError;
                    goto cleanup;
                }

                if (props.duration != cx_props[props_idx][0]) {
                    fprintf(stderr,
                            "Unexpected duration value found for operation %s at qarg index: %zu. "
                            "Expected: %f, got: %f.\n",
                            name, props_idx, props.duration, cx_props[props_idx][0]);
                    result = EqualityError;
                    goto cleanup;
                }

                if (props.error != cx_props[props_idx][1]) {
                    fprintf(stderr,
                            "Unexpected error value found for operation %s at qarg index: %zu. "
                            "Expected: %f, got: %f.\n",
                            name, props_idx, props.error, cx_props[props_idx][1]);
                    result = EqualityError;
                    goto cleanup;
                }
            }
            // Global y
            else if (op_idx == 5) {
                if (!compare_qargs(qargs, NULL, 0)) {
                    fprintf(
                        stderr,
                        "Unexpected qargs found for operation %s at qarg index: %zu. Expected: '",
                        name, props_idx);
                    print_qargs(NULL, 0);
                    fprintf(stderr, "', found '");
                    print_qargs(qargs, qargs_len);
                    fprintf(stderr, "'\n");
                    result = EqualityError;
                    goto cleanup;
                }
            }
            // Global phase
            else if (op_idx == 6) {
                if (!compare_qargs(qargs, (uint32_t[]){0}, 0)) {
                    fprintf(
                        stderr,
                        "Unexpected qargs found for operation %s at qarg index: %zu. Expected: '",
                        name, props_idx);
                    print_qargs((uint32_t[]){0}, 0);
                    fprintf(stderr, "', found '");
                    print_qargs(qargs, qargs_len);
                    fprintf(stderr, "'\n");
                    result = EqualityError;
                    goto cleanup;
                }
            }
            // id, rz, sx, x, measure, reset
            else {
                if (!compare_qargs(qargs, &single_qarg[props_idx], qargs_len)) {
                    fprintf(
                        stderr,
                        "Unexpected qargs found for operation %s at qarg index: %zu. Expected: '",
                        name, props_idx);
                    print_qargs(&single_qarg[props_idx], qargs_len);
                    fprintf(stderr, "', found '");
                    print_qargs(qargs, qargs_len);
                    fprintf(stderr, "'\n");
                    result = EqualityError;
                    goto cleanup;
                }

                if (op_idx > 6) {
                    if (props.duration != 1e-6) {
                        fprintf(stderr,
                                "Unexpected duration value found for operation %s at qarg index: "
                                "%zu. Expected: %f, got: %f.\n",
                                name, props_idx, props.duration, 1e-6);
                        result = EqualityError;
                        goto cleanup;
                    }

                    if (props.error != 1e-4) {
                        fprintf(stderr,
                                "Unexpected error value found for operation %s at qarg index: %zu. "
                                "Expected: %f, got: %f.\n",
                                name, props_idx, props.error, 1e-4);
                        result = EqualityError;
                        goto cleanup;
                    }
                }
            }
        }
    cleanup:
        qk_str_free(name);
        if (result != Ok) {
            break;
        }
    }
    qk_target_free(target);
    return result;
}

static int test_target_indexing(void) {
    QkTarget *target = create_sample_target(true);
    int result = Ok;

    // Retrieve the index for CX
    size_t cx_idx = qk_target_op_index(target, "cx");
    if (cx_idx != 4) {
        fprintf(stderr, "Invalid index for cx entry: expected %d, got %zu.\n", 4, cx_idx);
        result = EqualityError;
        goto cleanup;
    }

    // Retrieve the index for cz (should fail).
    size_t cz_idx = qk_target_op_index(target, "cz");
    if (cz_idx != (size_t)-1) {
        fprintf(stderr, "Found index for non-existing cz entry: got %zu.\n", cz_idx);
        result = EqualityError;
        goto cleanup;
    }

    // Check if qargs [0,1] exist in cx
    uint32_t cx_qargs_query[2] = {0, 1};
    if (qk_target_op_qargs_index(target, cx_idx, cx_qargs_query) != 6) {
        fprintf(stderr, "Couldn't find valid qarg entry [0, 1] for cx.\n");
        result = EqualityError;
        goto cleanup;
    }

    // Check if qargs [2,3] exist in cx (should fail)
    uint32_t cx_qargs_bad_query[2] = {2, 3};
    if (qk_target_op_qargs_index(target, cx_idx, cx_qargs_bad_query) != (size_t)-1) {
        fprintf(stderr, "Found valid qarg entry non-existing qargs [2, 3] for cx.\n");
        result = EqualityError;
        goto cleanup;
    }

    // Since index exists, this should return properly
    size_t cx_qargs_idx = qk_target_op_qargs_index(target, cx_idx, cx_qargs_query);
    if (cx_qargs_idx != 6) {
        fprintf(stderr, "Invalid index for cx qargs [0,1]: expected %d, got %zu.\n", 6, cx_idx);
        result = EqualityError;
        goto cleanup;
    }

    // Same test on the invalid query
    size_t cx_qargs_bad_idx = qk_target_op_qargs_index(target, cx_idx, cx_qargs_bad_query);
    if (cx_qargs_bad_idx != (size_t)-1) {
        fprintf(stderr, "Found index for non-existing qargs [2,3].\n");
        result = EqualityError;
        goto cleanup;
    }

    // Retrieving qargs [4,3] at index 1, with properties d=3.0577e-11, e=0.00713
    QkInstructionProperties cx_props;
    uint32_t *cx_qargs;
    uint32_t cx_qargs_len;

    qk_target_op_qargs(target, cx_idx, 1, &cx_qargs, &cx_qargs_len);
    if (!compare_qargs(cx_qargs, (uint32_t[2]){4, 3}, cx_qargs_len)) {
        fprintf(stderr, "Retrieved incorrect qargs, expected [4, 3], got [%u, %u]\n", cx_qargs[0],
                cx_qargs[1]);
        result = EqualityError;
        goto cleanup;
    }

    qk_target_op_props(target, cx_idx, 1, &cx_props);

    if (cx_props.duration != 3.0577e-11) {
        fprintf(stderr, "Retrieved incorrect duration property, expected 3.0577e-11, got %lf.\n",
                cx_props.duration);
        result = EqualityError;
        goto cleanup;
    }
    if (cx_props.error != 0.00713) {
        fprintf(stderr, "Retrieved incorrect error property, expected 0.00713, got %lf.\n",
                cx_props.error);
        result = EqualityError;
        goto cleanup;
    }

    // Try retrieving global index for y gate
    size_t y_idx = qk_target_op_index(target, "y");
    if (y_idx != 5) {
        fprintf(stderr, "Invalid index for y entry: expected %d, got %zu.\n", 5, y_idx);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t *y_qargs;
    uint32_t y_qargs_len;
    qk_target_op_qargs(target, y_idx, 0, &y_qargs, &y_qargs_len);

    if (y_qargs != NULL) {
        fprintf(stderr, "Obtained non-null global qargs at index 0 for 'y'.\n");
        result = EqualityError;
        goto cleanup;
    }

    // Try retrieving [] index for global_phase gate
    size_t gp_idx = qk_target_op_index(target, "global_phase");
    if (gp_idx != 6) {
        fprintf(stderr, "Invalid index for y entry: expected %d, got %zu.\n", 6, gp_idx);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t *gp_qargs;
    uint32_t gp_qargs_len;
    qk_target_op_qargs(target, gp_idx, 0, &gp_qargs, &gp_qargs_len);

    if (gp_qargs == NULL || gp_qargs_len != 0) {
        fprintf(stderr, "Obtained null or invalid qargs at index 0 for 'global_phase'.\n");
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test if our target is compatible with certain instructions.
 */
int test_target_instruction_supported(void) {
    QkTarget *sample_target = create_sample_target(true);
    int result = Ok;

    char *gate_names[7] = {"x", "y", "id", "rz_pi", "sx", "reset"};
    QkParam *rz_params[1] = {
        qk_param_from_double(3.14),
    };
    for (uint32_t qubit = 0; qubit < 5; qubit++) {
        uint32_t qargs[1] = {qubit};
        bool should_be_true = qubit < 4;

        for (int gate = 0; gate < 6; gate++) {
            // If i == 4 condition should be false unless we try with the y gate
            // since y is added as a global gate.
            if (qk_target_instruction_supported(sample_target, gate_names[gate], qargs,
                                                gate != 3 ? NULL : rz_params) !=
                (should_be_true || gate == 1)) {
                fprintf(stderr,
                        "This target did not correctly demonstrate compatibility with %s and qargs "
                        "[%d]",
                        gate_names[gate], qubit);
                result = EqualityError;
                goto cleanup;
            }
        }

        // Try checking with the wrong fixed parameter for RZ
        QkParam *param = qk_param_from_double(1.57);
        if (qk_target_instruction_supported(sample_target, "rz_pi", qargs,
                                            (QkParam *[]){
                                                param,
                                            })) {
            fprintf(stderr,
                    "This target did not correctly demonstrate compatibility with 'rz' and qargs "
                    "[%d]",
                    qubit);
            result = EqualityError;
            qk_param_free(param);
            goto cleanup;
        }
        qk_param_free(param);

        // Test standard instructions reset and measure
        if (!(qk_target_instruction_supported(sample_target, "measure", qargs, NULL) ==
              (qubit < 2))) {
            fprintf(
                stderr,
                "This target did not correctly demonstrate compatibility with 'measure' and qargs "
                "[%d]",
                qubit);
            result = EqualityError;
            goto cleanup;
        }
    }

    // Qarg samples for CX
    uint32_t qarg_samples[8][2] = {
        {3, 4}, {4, 3}, {3, 1}, {1, 3}, {1, 2}, {2, 1}, {0, 1}, {1, 0},
    };
    for (int i = 0; i < 8; i++) {
        if (!qk_target_instruction_supported(sample_target, "cx", qarg_samples[i], NULL)) {
            fprintf(
                stderr,
                "This target did incorrectly demonstrate compatibility with 'cx' and qargs [%d, "
                "%d]",
                qarg_samples[i][0], qarg_samples[i][1]);
            result = EqualityError;
            goto cleanup;
        }
    }

    uint32_t cx_qargs[2] = {3, 2};
    // Instruction should not show compatibility with (3, 2)
    if (qk_target_instruction_supported(sample_target, "cx", cx_qargs, NULL)) {
        fprintf(stderr,
                "This target did incorrectly demonstrate compatibility with 'cx' and qargs [3, 2]");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_param_free(rz_params[0]);
    qk_target_free(sample_target);
    return result;
}

static int test_target_operation(void) {
    QkTarget *target = create_sample_target(true);
    int result = Ok;

    size_t target_length = qk_target_num_instructions(target);
    char *names[9] = {"id", "rz_pi", "sx", "x", "cx", "y", "global_phase", "measure", "reset"};
    QkGate std_gates[7] = {QkGate_I,  QkGate_RZ, QkGate_SX,         QkGate_X,
                           QkGate_CX, QkGate_Y,  QkGate_GlobalPhase};
    // Allocate for the Target operation
    QkTargetOp op;
    for (size_t inst_idx = 0; inst_idx < target_length; inst_idx++) {
        // This method panics if the index is not valid.
        qk_target_op_get(target, inst_idx, &op);

        if (strcmp(op.name, names[inst_idx]) != 0) {
            fprintf(stderr, "The operation names did not match. Expected %s, got %s\n",
                    names[inst_idx], op.name);
            result = EqualityError;
            break;
        }

        char *op_types[6] = {"Gate", "Barrier", "Delay", "Measure", "Reset", "Unitary"};
        if (inst_idx < 7) {
            if (op.op_type != QkOperationKind_Gate) {
                fprintf(stderr, "The operation's type did not match, expected Gate, got %s\n",
                        op_types[(size_t)op.op_type]);
                result = EqualityError;
                break;
            }
            QkGate gate = qk_target_op_gate(target, inst_idx);
            if (gate != std_gates[inst_idx]) {
                fprintf(stderr, "The gate type did not match. Expected %i, got %i\n",
                        std_gates[inst_idx], gate);
                result = EqualityError;
                break;
            }
            if (inst_idx == 1) {
                double param_val = qk_param_as_real(op.params[0]);
                if (param_val != 3.14) {
                    fprintf(
                        stderr,
                        "The param value for instruction '%s' did not match. Expected %f, got %f\n",
                        op.name, 3.14, param_val);
                    result = EqualityError;
                    break;
                }
            } else {
                if (op.params == NULL && op.num_params != 0) {
                    fprintf(
                        stderr,
                        "The param values for instruction '%s' did not match. Got length %u for "
                        "NULL params.\n",
                        op.name, op.num_params);
                    result = EqualityError;
                    break;
                }
            }
        } else if (inst_idx == 7) {
            if (op.op_type != QkOperationKind_Measure) {
                fprintf(stderr, "The operation's type did not match, expected Measure, got %s\n",
                        op_types[(size_t)op.op_type]);
                result = EqualityError;
                break;
            }
        } else {
            if (op.op_type != QkOperationKind_Reset) {
                fprintf(stderr, "The operation's type did not match, expected Reset, got %s\n",
                        op_types[(size_t)op.op_type]);
                result = EqualityError;
                break;
            }
        }
        // Free the operation at the end of each loop iteration.
        qk_target_op_clear(&op);
    }
    // Free the operation once again in case loop ends early.
    qk_target_op_clear(&op);
    qk_target_free(target);
    return result;
}

int test_target(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty_target);
    num_failed += RUN_TEST(test_target_construct);
    num_failed += RUN_TEST(test_target_entry_construction);
    num_failed += RUN_TEST(test_target_add_instruction);
    num_failed += RUN_TEST(test_target_update_instruction);
    num_failed += RUN_TEST(test_target_construction_ibm_like_target);
    num_failed += RUN_TEST(test_target_iteration);
    num_failed += RUN_TEST(test_target_indexing);
    num_failed += RUN_TEST(test_target_instruction_supported);
    num_failed += RUN_TEST(test_target_operation);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
