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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int build_unitary_target(QkTarget *target, uint32_t num_qubits) {
    // Create a target with cx connectivity in a line.
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a global X gate.");
        return RuntimeError;
    }
    QkExitCode result_sx = qk_target_add_instruction(target, qk_target_entry_new(QkGate_SX));
    if (result_sx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a global SX gate.");
        return RuntimeError;
    }

    QkExitCode result_rz = qk_target_add_instruction(target, qk_target_entry_new(QkGate_RZ));
    if (result_rz != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a global RZ gate.");
        return RuntimeError;
    }

    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        QkExitCode result_cx_props =
            qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
        if (result_cx_props != QkExitCode_Success) {
            fprintf(stderr, "Unexpected error occurred when adding property to a CX gate entry.");
            return RuntimeError;
        }
    }
    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a CX gate.");
        return RuntimeError;
    }
    return Ok;
}

/**
 * Test running UnitarySynthesis on a single gate
 */
static int test_unitary_synthesis_identity_matrix(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_unitary_target(target, num_qubits);
    if (result != Ok) {
        goto cleanup;
    }
    // Create a circuit with line connectivity.
    QkCircuit *qc = qk_circuit_new(1, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 c1 = {1., 0.};
    QkComplex64 unitary[4] = {c1, c0,  // row 0
                              c0, c1}; // row 1
    uint32_t qargs[1] = {0};
    qk_circuit_unitary(qc, unitary, qargs, 1, false);
    qk_transpiler_pass_standalone_unitary_synthesis(qc, target, 0, 1.0);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    if (num_instructions != 0) {
        fprintf(stderr, "Identity unitary not removed from the circuit as expected");
        result = EqualityError;
    }
    qk_circuit_free(qc);

cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test running UnitarySynthesis on a single gate
 */
static int test_unitary_synthesis_multiqubit_identity_matrix(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_unitary_target(target, num_qubits);
    if (result != Ok) {
        goto cleanup;
    }
    // Create a circuit with line connectivity.
    QkCircuit *qc = qk_circuit_new(3, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 c1 = {1., 0.};
    QkComplex64 unitary[64] = {c1, c0, c0, c0, c0, c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c0,
                               c0, c0, c1, c0, c0, c0, c0, c0, c0, c0, c0, c1, c0, c0, c0, c0,
                               c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c0, c0, c0, c1, c0, c0,
                               c0, c0, c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c0, c0, c0, c1};
    uint32_t qargs[3] = {0, 1, 2};
    qk_circuit_unitary(qc, unitary, qargs, 3, false);
    qk_transpiler_pass_standalone_unitary_synthesis(qc, target, 0, 1.0);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    if (num_instructions != 0) {
        fprintf(stderr, "Identity unitary not removed from the circuit as expected");
        result = EqualityError;
    }
    qk_circuit_free(qc);

cleanup:
    qk_target_free(target);
    return result;
}

int test_unitary_synthesis(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_unitary_synthesis_identity_matrix);
    num_failed += RUN_TEST(test_unitary_synthesis_multiqubit_identity_matrix);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
