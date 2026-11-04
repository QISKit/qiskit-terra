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

static int test_commutative_cancellation_target(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_Z));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_SX));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_CX));

    int result = Ok;

    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t cx_qargs[2] = {
        0,
        1,
    };
    uint32_t rz_qargs[1] = {
        0,
    };
    double rz_params[1] = {
        3.14159,
    };
    qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);
    qk_circuit_gate(qc, QkGate_RZ, rz_qargs, rz_params);
    qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);

    result = qk_transpiler_pass_standalone_commutative_cancellation(qc, target, 1.0);
    if (result != 0) {
        fprintf(stderr, "Running the pass failed");
        goto cleanup;
    }

    if (qk_circuit_num_instructions(qc) != 1) {
        result = EqualityError;
        fprintf(stderr, "The gates weren't removed by this circuit");
    }
cleanup:
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;
}

static int test_commutative_cancellation_no_target(void) {
    int result = Ok;

    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t cx_qargs[2] = {
        0,
        1,
    };
    uint32_t rz_qargs[1] = {
        0,
    };
    double rz_params[1] = {
        3.14159,
    };
    qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);
    qk_circuit_gate(qc, QkGate_RZ, rz_qargs, rz_params);
    qk_circuit_gate(qc, QkGate_CX, cx_qargs, NULL);

    result = qk_transpiler_pass_standalone_commutative_cancellation(qc, NULL, 1.0);
    if (result != 0) {
        fprintf(stderr, "Running the pass failed");
        goto cleanup;
    }

    if (qk_circuit_num_instructions(qc) != 1) {
        result = EqualityError;
        fprintf(stderr, "The gates weren't removed by this circuit");
    }
cleanup:
    qk_circuit_free(qc);
    return result;
}

int test_commutative_cancellation(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_commutative_cancellation_target);
    num_failed += RUN_TEST(test_commutative_cancellation_no_target);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
