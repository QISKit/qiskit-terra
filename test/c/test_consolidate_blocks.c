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

static bool args_cmp(uint32_t *args1, uint32_t args1_len, uint32_t *args2, uint32_t args2_len) {
    if (args1_len != args2_len) {
        return false;
    }
    for (size_t i = 0; i < args1_len; i++) {
        if (args1[i] != args2[i]) {
            return false;
        }
    }
    return true;
}

/**
 * Test a small block of gates can be turned into a unitary on same wires
 */
static int test_consolidate_small_block(void) {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(2, 0);

    uint32_t p_qargs[1] = {0};
    uint32_t u_qargs[1] = {1};
    uint32_t cx_qargs[2] = {0, 1};

    double p_params[1] = {0.5};
    double u_params[3] = {1.5708, 0.2, 0.6};
    qk_circuit_gate(circuit, QkGate_Phase, p_qargs, p_params);
    qk_circuit_gate(circuit, QkGate_U, u_qargs, u_params);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs, NULL);

    // Run pass, without Target
    qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circuit);

    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_opcounts_clear(&counts);
    qk_circuit_free(circuit);
    return result;
}

/**
 * Order of qubits and the corresponding unitary is correct
 */
static int test_wire_order(void) {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t cx_qargs[2] = {1, 0};

    qk_circuit_gate(circuit, QkGate_CX, cx_qargs, NULL);

    // Run pass, without Target
    qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circuit);

    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction instruction;
    qk_circuit_get_instruction(circuit, 0, &instruction);

    uint32_t new_cx_qargs[2] = {0, 1};
    if (!args_cmp(new_cx_qargs, 2, instruction.qubits, instruction.num_qubits)) {
        result = EqualityError;
        goto clean_instr;
    }
clean_instr:
    qk_circuit_instruction_clear(&instruction);
cleanup:
    qk_opcounts_clear(&counts);
    qk_circuit_free(circuit);
    return result;
}

/**
 * blocks of more than 2 qubits work.
 */
static int test_3q_blocks(void) {
    int result = Ok;

    // Build circuit
    QkCircuit *circuit = qk_circuit_new(3, 0);

    uint32_t p_qargs[1] = {0};
    uint32_t u_qargs[1] = {1};
    uint32_t cx_qargs_0[2] = {2, 1};
    uint32_t cx_qargs_1[2] = {0, 1};

    double p_params[1] = {0.5};
    double u_params[3] = {1.5708, 0.2, 0.6};
    qk_circuit_gate(circuit, QkGate_Phase, p_qargs, p_params);
    qk_circuit_gate(circuit, QkGate_U, u_qargs, u_params);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs_0, NULL);
    qk_circuit_gate(circuit, QkGate_CX, cx_qargs_1, NULL);

    // Run pass, without Target
    qk_transpiler_pass_standalone_consolidate_blocks(circuit, NULL, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(circuit);
    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_opcounts_clear(&counts);
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test a non-cx kak gate is consolidated correctly with a target.
 */
static int test_non_cx_target(void) {
    int result = Ok;

    // Create circuit
    QkCircuit *qc = qk_circuit_new(2, 0);
    // Add gates
    qk_circuit_gate(qc, QkGate_CZ, (uint32_t[2]){0, 1}, NULL);
    qk_circuit_gate(qc, QkGate_X,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_H,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_Z,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_T,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_H,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_T,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_CZ, (uint32_t[2]){0, 1}, NULL);
    qk_circuit_gate(qc, QkGate_SX,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_SX,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_CZ, (uint32_t[2]){0, 1}, NULL);
    qk_circuit_gate(qc, QkGate_SX,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_SX,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_CZ, (uint32_t[2]){0, 1}, NULL);
    qk_circuit_gate(qc, QkGate_X,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_H,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_Z,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_T,
                    (uint32_t[1]){
                        1,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_H,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_T,
                    (uint32_t[1]){
                        0,
                    },
                    NULL);
    qk_circuit_gate(qc, QkGate_CZ, (uint32_t[2]){0, 1}, NULL);

    // Build target
    QkTarget *target = qk_target_new(2);
    QkGate gates[4] = {QkGate_SX, QkGate_X, QkGate_RZ, QkGate_CZ};
    for (int idx = 0; idx < 4; idx++) {
        QkTargetEntry *entry = qk_target_entry_new(gates[idx]);
        if (gates[idx] == QkGate_CZ) {
            qk_target_entry_add_property(entry, (uint32_t[2]){0, 1}, 2, NAN, NAN);
            qk_target_entry_add_property(entry, (uint32_t[2]){1, 0}, 2, NAN, NAN);
        } else {
            qk_target_entry_add_property(entry, (uint32_t[1]){0}, 1, NAN, NAN);
            qk_target_entry_add_property(entry, (uint32_t[1]){1}, 1, NAN, NAN);
        }
        qk_target_add_instruction(target, entry);
    }

    // Run the pass
    qk_transpiler_pass_standalone_consolidate_blocks(qc, target, 1.0, true);

    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        result = EqualityError;
        fprintf(stderr,
                "The pass run did not result in a circuit with one unitary gate. Expected 1 gate, "
                "got %zu.",
                counts.len);
        goto cleanup;
    }

    QkCircuitInstruction unitary;
    qk_circuit_get_instruction(qc, 0, &unitary);
    if (strcmp(unitary.name, "unitary") != 0) {
        result = EqualityError;
        fprintf(
            stderr,
            "The pass run did not result in a circuit with one unitary gate. Expected 'unitary' "
            "gate, got '%s'.",
            unitary.name);
        goto cleanup_inst;
    }
cleanup_inst:
    qk_circuit_instruction_clear(&unitary);
cleanup:
    qk_opcounts_clear(&counts);
    qk_target_free(target);
    qk_circuit_free(qc);
    return result;
}

int test_consolidate_blocks(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_consolidate_small_block);
    num_failed += RUN_TEST(test_wire_order);
    num_failed += RUN_TEST(test_3q_blocks);
    num_failed += RUN_TEST(test_non_cx_target);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
