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

/**
 * Test building a quantum volume circuit
 */
static int test_qv(void) {
    const uint32_t num_qubits = 5;
    int result = Ok;
    QkCircuit *qv = qk_circuit_library_quantum_volume(num_qubits, num_qubits, 42);
    size_t num_instructions = qk_circuit_num_instructions(qv);
    if (num_instructions != 10) {
        fprintf(stderr, "Unexpected number of instructions: %zu\n", num_instructions);
        result = EqualityError;
        goto cleanup;
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_instructions; i++) {
        qk_circuit_get_instruction(qv, i, &inst);
        int cmp_result = strcmp(inst.name, "unitary");
        if (cmp_result != 0) {
            result = EqualityError;
            goto loop_exit;
        }
        if (inst.num_qubits != 2) {
            result = EqualityError;
            goto loop_exit;
        }
    loop_exit:
        qk_circuit_instruction_clear(&inst);
        if (result != 0) {
            break;
        }
    }

cleanup:
    qk_circuit_free(qv);
    return result;
}

int test_quantum_volume(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_qv);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
