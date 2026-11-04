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

static int test_transpile_layout_generate(void) {
    int result = Ok;
    QkTarget *target = qk_target_new(5);
    QkDag *orig_dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
    qk_dag_add_quantum_register(orig_dag, qr);
    uint32_t layout_mapping[5] = {1, 4, 3, 2, 0};
    QkTranspileLayout *layout =
        qk_transpile_layout_generate_from_mapping(orig_dag, target, layout_mapping);
    if (qk_transpile_layout_num_input_qubits(layout) != 3) {
        fprintf(stderr, "Number of input qubits is %u, expected 3\n",
                qk_transpile_layout_num_input_qubits(layout));
        result = EqualityError;
        goto cleanup;
    }
    if (qk_transpile_layout_num_output_qubits(layout) != 5) {
        fprintf(stderr, "Number of output qubits is %u, expected 5\n",
                qk_transpile_layout_num_input_qubits(layout));
        result = EqualityError;
        goto cleanup;
    }
    bool permutation_set = qk_transpile_layout_output_permutation(layout, NULL);
    if (permutation_set) {
        fprintf(stderr, "Generated unexpectedly has an output permutation");
        goto cleanup;
    }
    uint32_t *output_mapping = malloc(sizeof(uint32_t) * 5);
    bool layout_set = qk_transpile_layout_initial_layout(layout, false, output_mapping);
    if (!layout_set) {
        fprintf(stderr, "Generated layout doesn't have initial layout set\n");
        result = EqualityError;
        goto result_cleanup;
    }
    for (int i = 0; i < 5; i++) {
        if (output_mapping[i] != layout_mapping[i]) {
            fprintf(stderr, "Element %i does not match. Result: %u, Expected: %u\n", i,
                    output_mapping[i], layout_mapping[i]);
            result = EqualityError;
            goto result_cleanup;
        }
    }
result_cleanup:
    free(output_mapping);

cleanup:
    qk_transpile_layout_free(layout);
    qk_quantum_register_free(qr);
    qk_dag_free(orig_dag);
    qk_target_free(target);
    return result;
}

int test_transpile_layout(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_transpile_layout_generate);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
