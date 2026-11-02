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
 * Test running the path with no elision.
 */
static int test_standalone_elide_permutations_no_result(void) {
    const uint32_t num_qubits = 5;

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    int result = Ok;
    QkTranspileLayout *pass_result = qk_transpiler_pass_standalone_elide_permutations(qc);
    if (pass_result != NULL) {
        fprintf(stderr, "A gate was elided when one shouldn't have been\n");
        qk_transpile_layout_free(pass_result);
        result = EqualityError;
    }
    qk_circuit_free(qc);
    return result;
}

/**
 * Test running the path with no elision.
 */
static int test_standalone_elide_permutations_swap_result(void) {
    QkCircuit *qc = qk_circuit_new(5, 0);
    uint32_t swap_qargs[2] = {1, 3};
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
        if (i == 2) {
            qk_circuit_gate(qc, QkGate_Swap, swap_qargs, NULL);
        }
    }
    int result = Ok;
    QkTranspileLayout *pass_result = qk_transpiler_pass_standalone_elide_permutations(qc);
    if (pass_result == NULL) {
        fprintf(stderr, "A gate wasn't elided when one should have been\n");
        result = EqualityError;
        goto cleanup;
    }
    uint32_t permutation[5];
    qk_transpile_layout_output_permutation(pass_result, permutation);
    uint32_t expected_permutation[5] = {0, 3, 2, 1, 4};
    for (int i = 0; i < 5; i++) {
        if (permutation[i] != expected_permutation[i]) {
            fprintf(stderr, "Permutation doesn't match expected\n");
            result = EqualityError;
            goto result_cleanup;
        }
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1) {
        fprintf(stderr, "More than 1 type of gates in circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (size_t i = 0; i < op_counts.len; i++) {
        int swap_gate = strcmp(op_counts.data[i].name, "swap");
        if (swap_gate == 0) {
            fprintf(stderr, "Swap gate in circuit which should have been elided\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }

ops_cleanup:
    qk_opcounts_clear(&op_counts);
result_cleanup:
    qk_transpile_layout_free(pass_result);
cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test running the path with no elision.
 */
static int test_elide_permutations_no_result(void) {
    const uint32_t num_qubits = 5;

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(num_qubits, "q");
    qk_dag_add_quantum_register(dag, qr);

    for (uint32_t i = 0; i < qk_dag_num_qubits(dag) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);
        }
    }
    int result = Ok;
    QkTranspileLayout *pass_result = qk_transpiler_pass_elide_permutations(dag);
    if (pass_result != NULL) {
        fprintf(stderr, "A gate was elided when one shouldn't have been\n");
        qk_transpile_layout_free(pass_result);
        result = EqualityError;
    }
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

/**
 * Test running the path with no elision.
 */
static int test_elide_permutations_swap_result(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(5, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t swap_qargs[2] = {1, 3};
    for (uint32_t i = 0; i < qk_dag_num_qubits(dag) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);
        }
        if (i == 2) {
            qk_dag_apply_gate(dag, QkGate_Swap, swap_qargs, NULL, false);
        }
    }
    int result = Ok;
    QkTranspileLayout *pass_result = qk_transpiler_pass_elide_permutations(dag);
    if (pass_result == NULL) {
        fprintf(stderr, "A gate wasn't elided when one should have been\n");
        result = EqualityError;
        goto cleanup;
    }
    uint32_t permutation[5];
    qk_transpile_layout_output_permutation(pass_result, permutation);
    uint32_t expected_permutation[5] = {0, 3, 2, 1, 4};
    for (int i = 0; i < 5; i++) {
        if (permutation[i] != expected_permutation[i]) {
            fprintf(stderr, "Permutation doesn't match expected\n");
            result = EqualityError;
            goto result_cleanup;
        }
    }
    size_t num_ops = qk_dag_num_op_nodes(dag);
    uint32_t *ops = malloc(num_ops * sizeof(*ops));
    qk_dag_topological_op_nodes(dag, ops);
    QkGate first_gate = qk_dag_op_node_gate_op(dag, ops[0], NULL);
    for (size_t i = 1; i < num_ops; i++) {
        QkGate gate = qk_dag_op_node_gate_op(dag, ops[i], NULL);
        if (gate != first_gate) {
            fprintf(stderr, "More than 1 type of gates in DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        } else if (gate == QkGate_Swap) {
            fprintf(stderr, "Swap gate in DAG which should have been elided\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }

ops_cleanup:
    free(ops);
result_cleanup:
    qk_transpile_layout_free(pass_result);
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

int test_elide_permutations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_standalone_elide_permutations_no_result);
    num_failed += RUN_TEST(test_standalone_elide_permutations_swap_result);
    num_failed += RUN_TEST(test_elide_permutations_no_result);
    num_failed += RUN_TEST(test_elide_permutations_swap_result);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
