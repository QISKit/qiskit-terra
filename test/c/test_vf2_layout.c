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

static int build_target(QkTarget *target, uint32_t num_qubits) {
    // Create a target with cx connectivity in a line.
    QkExitCode result_x = qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    if (result_x != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a global X gate.\n");
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
            fprintf(stderr, "Unexpected error occurred when adding property to a CX gate entry.\n");
            return RuntimeError;
        }
    }
    QkExitCode result_cx = qk_target_add_instruction(target, cx_entry);
    if (result_cx != QkExitCode_Success) {
        fprintf(stderr, "Unexpected error occurred when adding a CX gate.\n");
        return RuntimeError;
    }
    return Ok;
}

/**
 * Test running VF2Layout on a line connectivity
 */
static int test_vf2_average_line(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_target(target, num_qubits);
    if (result != Ok) {
        goto target_cleanup;
    }
    // Create a circuit with line connectivity.
    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkVF2LayoutConfiguration *layout_config = qk_vf2_layout_configuration_new();
    qk_vf2_layout_configuration_set_call_limit(layout_config, 10000, 10000);
    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_average(qc, target, layout_config, false);
    if (!qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: No layout was found\n", __func__);
        result = EqualityError;
        goto layout_cleanup;
    }
    // Higher indexed physical qubits have lower error rates and
    // higher indexed virtual qubits have more gates. VF2 should
    // chose a trivial layout since it's the lowest error rate
    // in mapping these lines together.
    uint32_t expected[5] = {0, 1, 2, 3, 4};
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t phys = qk_vf2_layout_result_map_virtual_qubit(layout_result, i);
        if (phys != expected[i]) {
            fprintf(stderr, "%s: Unexpected layout result virtual qubit %d mapped to %d\n",
                    __func__, phys, expected[i]);
            result = EqualityError;
            goto layout_cleanup;
        }
    }

layout_cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_vf2_layout_configuration_free(layout_config);
    qk_circuit_free(qc);
target_cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test VF2Layout where a solution isn't possible
 */
static int test_vf2_no_layout_found(void) {
    const uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;
    result = build_target(target, num_qubits);
    if (result != Ok) {
        goto target_cleanup;
    }
    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc); i++) {
        for (uint32_t j = 0; j < qk_circuit_num_qubits(qc); j++) {
            if (i == j) {
                continue;
            }
            uint32_t qargs[2] = {i, j};
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_average(qc, target, NULL, false);
    if (qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: Unexpected layout found when one shouldn't be possible", __func__);
        result = EqualityError;
        goto layout_cleanup;
    }
layout_cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
target_cleanup:
    qk_target_free(target);
    return result;
}

/**
 * Test exact-match VF2 layout when no solution is possible because of available 1q operations.
 */
static int test_vf2_exact_no_layout_1q_semantics(void) {
    int result = RuntimeError;
    uint32_t num_qubits = 5;
    uint32_t line[5] = {0, 1, 2, 3, 4};

    QkTarget *target = qk_target_new(num_qubits);
    result = build_target(target, num_qubits);
    if (result != Ok)
        goto target_failure;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    // The CX line is fine, as are the X gates.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_circuit_gate(qc, QkGate_CX, &line[i], NULL);
        qk_circuit_gate(qc, QkGate_X, &line[i], NULL);
    }
    // This Y gate isn't fine.
    qk_circuit_gate(qc, QkGate_Y, &line[num_qubits - 1], NULL);

    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, NULL);
    if (qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: unexpected match\n", __func__);
        result = EqualityError;
        goto cleanup;
    }
    if (qk_vf2_layout_result_has_improvement(layout_result)) {
        fprintf(stderr, "%s: claimed improvement without a match\n", __func__);
        result = EqualityError;
        goto cleanup;
    }
    result = Ok;

cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;

target_failure:
    fprintf(stderr, "%s: failed to build_target\n", __func__);
    return result;
}

/**
 * Test exact-match VF2 layout when no solution is possible because of available 2q operations, even
 * though the structure of the graph matches.
 */
static int test_vf2_exact_no_layout_2q_semantics(void) {
    int result = RuntimeError;
    uint32_t num_qubits = 5;
    uint32_t line[5] = {0, 1, 2, 3, 4};

    QkTarget *target = qk_target_new(num_qubits);
    result = build_target(target, num_qubits);
    if (result != Ok) {
        goto target_failure;
    }

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    // The CX line is fine, as are the X gates.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_circuit_gate(qc, QkGate_CX, &line[i], NULL);
        qk_circuit_gate(qc, QkGate_X, &line[i], NULL);
    }
    // This CZ is not fine.
    qk_circuit_gate(qc, QkGate_CZ, line, NULL);

    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, NULL);
    if (qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: unexpected match\n", __func__);
        goto cleanup;
    }
    if (qk_vf2_layout_result_has_improvement(layout_result)) {
        fprintf(stderr, "%s: claimed improvement without a match\n", __func__);
        goto cleanup;
    }
    result = Ok;

cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;

target_failure:
    fprintf(stderr, "%s: failed to build_target\n", __func__);
    return result;
}

/**
 * Test exact-match VF2 layout when no solution is possible because the edge structure couldn't
 * match this interaction graph, no matter the nodes.
 */
static int test_vf2_exact_no_layout_2q_structure(void) {
    int result = RuntimeError;
    uint32_t num_qubits = 5;
    uint32_t line[5] = {0, 1, 2, 3, 4};

    QkTarget *target = qk_target_new(num_qubits);
    result = build_target(target, num_qubits);
    if (result != Ok)
        goto target_failure;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    // The CX line is fine, as are the X gates.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_circuit_gate(qc, QkGate_CX, &line[i], NULL);
        qk_circuit_gate(qc, QkGate_X, &line[i], NULL);
    }
    // This CX is backwards and so not fine.
    uint32_t bad_cx[2] = {1, 0};
    qk_circuit_gate(qc, QkGate_CX, bad_cx, NULL);

    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, NULL);
    if (qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: unexpected match\n", __func__);
        goto cleanup;
    }
    if (qk_vf2_layout_result_has_improvement(layout_result)) {
        fprintf(stderr, "%s: claimed improvement without a match\n", __func__);
        goto cleanup;
    }
    result = Ok;

cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;

target_failure:
    result = RuntimeError;
    fprintf(stderr, "%s: failed to build_target\n", __func__);
    qk_target_free(target);
    return result;
}

/**
 * Test exact-match VF2 layout when the existing layout is already as good as it can be.
 */
static int test_vf2_exact_no_improvement(void) {
    int result = RuntimeError;
    uint32_t num_qubits = 5;
    uint32_t line[5] = {0, 1, 2, 3, 4};

    QkTarget *target = qk_target_new(num_qubits);
    if (qk_target_add_instruction(target, qk_target_entry_new(QkGate_X)))
        goto target_failure;
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    // In this target, the errors get stronger on higher-indexed CXs.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        double error = 0.0625 * (i + 1);
        if (qk_target_entry_add_property(cx_entry, line + i, 2, 0.0, error))
            goto target_failure;
    }
    if (qk_target_add_instruction(target, cx_entry))
        goto target_failure;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    // There are more CXes on the early qubits, so we should prefer those to stay on the early
    // indexes, like they already are.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_circuit_gate(qc, QkGate_X, &line[i], NULL);
        for (uint32_t j = num_qubits; j; j--)
            qk_circuit_gate(qc, QkGate_CX, &line[i], NULL);
    }

    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, NULL);
    if (!qk_vf2_layout_result_has_match(layout_result)) {
        fprintf(stderr, "%s: failed to find a layout\n", __func__);
        goto cleanup;
    }
    if (qk_vf2_layout_result_has_improvement(layout_result)) {
        fprintf(stderr, "%s: claimed improvement but original was optimal\n", __func__);
        goto cleanup;
    }
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t mapped = qk_vf2_layout_result_map_virtual_qubit(layout_result, i);
        if (mapped != i) {
            fprintf(stderr, "%s: no-improvement result falsely mapped %u to %u\n", __func__, mapped,
                    i);
            goto cleanup;
        }
    }
    result = Ok;

cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;

target_failure:
    fprintf(stderr, "%s: failed to build_target\n", __func__);
    qk_target_free(target);
    return result;
}

/**
 * Test exact-match VF2 layout when there is a better mapping to find.
 */
static int test_vf2_exact_remap(void) {
    int result = RuntimeError;
    uint32_t num_qubits = 5;
    uint32_t line[5] = {0, 1, 2, 3, 4};

    QkTarget *target = qk_target_new(num_qubits);
    if (qk_target_add_instruction(target, qk_target_entry_new(QkGate_X)))
        goto target_failure;
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    // In this target, the errors get stronger on higher-indexed CXs, and it's bidirectional.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        double error = 0.0625 * (i + 1);
        if (qk_target_entry_add_property(cx_entry, line + i, 2, 0.0, error))
            goto target_failure;
        uint32_t reverse[2] = {line[i + 1], line[i]};
        if (qk_target_entry_add_property(cx_entry, reverse, 2, 0.0, error))
            goto target_failure;
    }
    if (qk_target_add_instruction(target, cx_entry))
        goto target_failure;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    // There are more CXes on the _late_ qubits, so VF2 should completely invert the mapping.
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        for (uint32_t j = i + 1; j; j--)
            qk_circuit_gate(qc, QkGate_CX, &line[i], NULL);
    }

    QkVF2LayoutResult *layout_result =
        qk_transpiler_pass_standalone_vf2_layout_exact(qc, target, NULL);
    if (!qk_vf2_layout_result_has_improvement(layout_result)) {
        fprintf(stderr, "%s: failed to improve non-optimal layout\n", __func__);
        goto cleanup;
    }
    for (uint32_t i = 0; i < num_qubits; i++) {
        // The correct result is {4, 3, 2, 1, 0}.
        uint32_t mapped = qk_vf2_layout_result_map_virtual_qubit(layout_result, i);
        if (mapped != num_qubits - i - 1) {
            fprintf(stderr, "%s: improvement result falsely mapped %u to %u\n", __func__, mapped,
                    i);
            goto cleanup;
        }
    }
    result = Ok;

cleanup:
    qk_vf2_layout_result_free(layout_result);
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;

target_failure:
    fprintf(stderr, "%s: failed to build_target\n", __func__);
    qk_target_free(target);
    return result;
}

int test_vf2_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_vf2_average_line);
    num_failed += RUN_TEST(test_vf2_no_layout_found);

    num_failed += RUN_TEST(test_vf2_exact_no_layout_1q_semantics);
    num_failed += RUN_TEST(test_vf2_exact_no_layout_2q_semantics);
    num_failed += RUN_TEST(test_vf2_exact_no_layout_2q_structure);
    num_failed += RUN_TEST(test_vf2_exact_no_improvement);
    num_failed += RUN_TEST(test_vf2_exact_remap);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
