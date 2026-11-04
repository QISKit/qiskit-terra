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

/**
 * Test running sabre layout that requires layout and routing
 */
static int test_sabre_layout_applies_layout(void) {
    int result = Ok;

    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);

    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;
        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    }

    qk_target_add_instruction(target, cx_entry);

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {0, i + 1};
        qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
    }
    QkSabreLayoutOptions options = qk_sabre_layout_options_default();
    options.seed = 2025;
    QkTranspileLayout *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, &options);

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 2) {
        fprintf(stderr, "More than 2 types of gates in circuit, circuit's instructions are:\n");
        print_circuit(qc);
        result = EqualityError;
        goto cleanup;
    }
    for (size_t i = 0; i < op_counts.len; i++) {
        const char *name = op_counts.data[i].name;
        int swap_gate = strcmp(name, "swap");
        int cx_gate = strcmp(name, "cx");
        if (cx_gate != 0 && swap_gate != 0) {
            fprintf(stderr, "Gate type of %s found in the circuit which isn't expected\n", name);
            result = EqualityError;
            goto cleanup;
        }
        size_t count = op_counts.data[i].count;
        if (swap_gate == 0 && count != 2) {
            fprintf(stderr, "Unexpected number of swaps %zu found in the circuit.\n", count);
            result = EqualityError;
            goto cleanup;
        }
    }
    uint32_t expected_initial_layout[5] = {1, 0, 2, 3, 4};
    uint32_t result_initial_layout[5];
    qk_transpile_layout_initial_layout(layout_result, false, result_initial_layout);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_initial_layout[i] != expected_initial_layout[i]) {
            fprintf(stderr, "Initial layout maps qubit %d to %d, expected %d instead\n", i,
                    result_initial_layout[i], expected_initial_layout[i]);
            result = EqualityError;
            goto cleanup;
        }
    }

    uint32_t expected_permutation[5] = {0, 3, 1, 2, 4};
    uint32_t result_permutation[5];
    qk_transpile_layout_output_permutation(layout_result, result_permutation);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_permutation[i] != expected_permutation[i]) {
            fprintf(stderr, "Output permutation maps qubit %d to %d, expected %d instead\n", i,
                    result_permutation[i], expected_permutation[i]);
            result = EqualityError;
            goto cleanup;
        }
    }
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    uint32_t qargs[2] = {1, 0};
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 1;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 1;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 2;
    qargs[1] = 3;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 2;
    qk_circuit_gate(expected_circuit, QkGate_Swap, qargs, NULL);
    qargs[0] = 3;
    qargs[1] = 4;
    qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
    bool compare_result = compare_circuits(qc, expected_circuit);
    if (!compare_result) {
        result = EqualityError;
    }
    qk_circuit_free(expected_circuit);

cleanup:
    qk_opcounts_clear(&op_counts);
    qk_transpile_layout_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;
}

/**
 * Test running sabre layout that performs no transformation.
 */
static int test_sabre_layout_no_swap(void) {
    int result = Ok;

    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        qk_target_entry_add_property(cx_entry, qargs, 2, inst_duration, inst_error);
    }

    qk_target_add_instruction(target, cx_entry);

    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkSabreLayoutOptions options = qk_sabre_layout_options_default();
    options.seed = 2025;
    QkTranspileLayout *layout_result =
        qk_transpiler_pass_standalone_sabre_layout(qc, target, &options);
    QkCircuit *expected_circuit = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(expected_circuit, QkGate_CX, qargs, NULL);
        }
    }
    bool circuit_eq = compare_circuits(qc, expected_circuit);
    if (!circuit_eq) {
        result = EqualityError;
        goto cleanup;
    }
    uint32_t expected_initial_layout[5] = {0, 1, 2, 3, 4};
    uint32_t result_initial_layout[5];
    qk_transpile_layout_initial_layout(layout_result, false, result_initial_layout);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_initial_layout[i] != expected_initial_layout[i]) {
            fprintf(stderr, "Initial layout maps qubit %d to %d, expected %d instead\n", i,
                    result_initial_layout[i], expected_initial_layout[i]);
            result = EqualityError;
            goto cleanup;
        }
    }
    uint32_t expected_permutation[5] = {0, 1, 2, 3, 4};
    uint32_t result_permutation[5];
    qk_transpile_layout_output_permutation(layout_result, result_permutation);
    for (uint32_t i = 0; i < 5; i++) {
        if (result_permutation[i] != expected_permutation[i]) {
            fprintf(stderr, "Output permutation maps qubit %d to %d, expected %d instead\n", i,
                    result_permutation[i], expected_permutation[i]);
            result = EqualityError;
            goto cleanup;
        }
    }

cleanup:
    qk_circuit_free(expected_circuit);
    qk_transpile_layout_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);

    return result;
}

int test_sabre_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_sabre_layout_no_swap);
    num_failed += RUN_TEST(test_sabre_layout_applies_layout);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
