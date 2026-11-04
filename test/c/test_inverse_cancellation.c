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
 * Test inverse cancellation.
 */
static int test_inverse_cancellation_removes_gates(void) {
    int result = Ok;

    QkCircuit *qc = qk_circuit_new(2, 2);
    uint32_t qargs[1] = {0};
    qk_circuit_gate(qc, QkGate_X, qargs, NULL);
    qk_circuit_gate(qc, QkGate_H, qargs, NULL);
    qk_circuit_gate(qc, QkGate_H, qargs, NULL);
    qk_circuit_gate(qc, QkGate_Y, qargs, NULL);

    qk_transpiler_pass_standalone_inverse_cancellation(qc);

    qk_circuit_free(qc);
    return result;
}

int test_inverse_cancellation(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_inverse_cancellation_removes_gates);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
