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
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test building an IQP circuit from symmetric interactions matrix.
 *
 * For 2 qubits and interactions matrix :
 *
 * [0 1
 * 1 0]
 *
 * we expect :
 * H ⊗ H (2 instructions)
 * 1 two-qubit entangling gate (CPhase-like) from the off-diagonal
 * H ⊗ H (2 instructions)
 *
 * Total of 5 instructions.
 */
static int test_iqp_from_interactions(void) {
    const uint32_t num_qubits = 2;
    int result = Ok;

    // Symmetric 2x2 interactions matrix:
    //[0 1
    // 1 0]
    int64_t interactions[4] = {
        0,
        1,
        1,
        0,
    };

    QkCircuit *iqp = qk_circuit_library_iqp(num_qubits, interactions, true);
    if (iqp == NULL) {
        fprintf(stderr, "iqp_from_interactions returned NULL\n");
        return EqualityError;
    }

    size_t num_instructions = qk_circuit_num_instructions(iqp);
    if (num_instructions != 5) {
        fprintf(stderr, "Unexpected number of instructions: %zu (expected 5)\n", num_instructions);
        result = EqualityError;
        goto cleanup;
    }
    // Optional sanity checks on instruction structure.
    QkCircuitInstruction inst;
    size_t num_two_qubit = 0;
    for (size_t i = 0; i < num_instructions; i++) {
        qk_circuit_get_instruction(iqp, i, &inst);

        // All gates in this IQP construction should be 1 or 2 qubits.
        if (inst.num_qubits != 1 && inst.num_qubits != 2) {
            fprintf(stderr, "Unexpected num_qubits = %u at instruction %zu\n", inst.num_qubits, i);
            result = EqualityError;
            qk_circuit_instruction_clear(&inst);
            goto cleanup;
        }

        if (inst.num_qubits == 2) {
            num_two_qubit += 1;
        }

        qk_circuit_instruction_clear(&inst);
    }

    // With the chosen interactions matrix, we expect exactly one 2-qubit gate.
    if (num_two_qubit != 1) {
        fprintf(stderr, "Unexpected number of 2-qubit gates: %zu (expected 1)\n", num_two_qubit);
        result = EqualityError;
    }

cleanup:
    qk_circuit_free(iqp);
    return result;
}

/**
 * Test that a non-symmetric interactions matrix return NULL.
 */
static int test_iqp_non_symmetric(void) {
    const uint32_t num_qubits = 2;
    int result = Ok;

    // Non-symmetric 2x2 interactions matrix:
    //[0 1
    // 0 1]
    int64_t interactions[4] = {
        0,
        1,
        0,
        1,
    };

    QkCircuit *iqp = qk_circuit_library_iqp(num_qubits, interactions, true);
    if (iqp != NULL) {
        fprintf(stderr, "Expected NULL circuit for non-symmetric matrix, got non-NULL\n");
        qk_circuit_free(iqp);
        result = EqualityError;
    }
    return result;
}

/**
 *  Test that a random IQP circuit can be generated
 */
static int test_random_iqp(void) {
    const uint32_t num_qubits = 4;
    int result = Ok;

    // Fixed seed for deterministic behavior
    QkCircuit *iqp = qk_circuit_library_random_iqp(num_qubits, 1234);
    if (iqp == NULL) {
        fprintf(stderr, "random_iqp returned NULL\n");
        return EqualityError;
    }

    size_t num_instructions = qk_circuit_num_instructions(iqp);
    if (num_instructions == 0) {
        fprintf(stderr, "Random IQP circuit has zero instructions\n");
        result = EqualityError;
        goto cleanup;
    }

    // Basic structural sanity: all gates should be 1- or 2-qubit.
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_instructions; i++) {
        qk_circuit_get_instruction(iqp, i, &inst);
        if (inst.num_qubits != 1 && inst.num_qubits != 2) {
            fprintf(stderr, "Random IQP has instruction with num_qubits = %u at index %zu\n",
                    inst.num_qubits, i);
            result = EqualityError;
            qk_circuit_instruction_clear(&inst);
            goto cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }
cleanup:
    qk_circuit_free(iqp);
    return result;
}

int test_iqp(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_iqp_from_interactions);
    num_failed += RUN_TEST(test_iqp_non_symmetric);
    num_failed += RUN_TEST(test_random_iqp);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (IQP): %i\n", num_failed);

    return num_failed;
}