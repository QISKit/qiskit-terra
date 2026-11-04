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

static int test_standalone_split_2q_unitaries_no_unitaries(void) {
    QkCircuit *qc = qk_circuit_new(5, 0);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
        }
    }
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        fprintf(stderr, "Permutation returned for a circuit that shouldn't split\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        fprintf(stderr, "More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (size_t i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "cx");
        if (gate != 0) {
            fprintf(stderr, "gates changed when there should be no circuit changes\n");
            result = EqualityError;
            goto cleanup;
        }
    }
ops_cleanup:
    qk_opcounts_clear(&counts);
cleanup:
    qk_circuit_free(qc);
    return result;
}

static int test_standalone_split_2q_unitaries_x_y_unitary(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, c0, neg_im, c0, c0, im, c0, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_circuit_unitary(qc, unitary, qargs, 2, true);
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        fprintf(stderr,
                "Permutation returned for a circuit that shouldn't isn't a swap equivalent\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        fprintf(stderr, "More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (size_t i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            fprintf(stderr, "Gates outside expected set in output circuit\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        size_t count = counts.data[i].count;
        if (count != 2) {
            fprintf(stderr, "Unexpected gate counts found\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (inst.num_qubits != 1) {
            fprintf(stderr, "Gate %zu operates on more than 1 qubit: %u\n", i, inst.num_qubits);
            result = EqualityError;
            goto ops_cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }

ops_cleanup:
    qk_opcounts_clear(&counts);
cleanup:
    qk_circuit_free(qc);
    return result;
}

static int test_standalone_split_2q_unitaries_swap_x_y_unitary(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, neg_im, c0, c0, c0, c0, im, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_circuit_unitary(qc, unitary, qargs, 2, true);
    QkTranspileLayout *split_result =
        qk_transpiler_pass_standalone_split_2q_unitaries(qc, 1 - 1e-16, true);
    int result = Ok;
    if (split_result == NULL) {
        result = EqualityError;
        fprintf(stderr, "Permutation not returned for a circuit that should have one\n");
        goto cleanup;
    }
    uint32_t permutation[2];
    qk_transpile_layout_output_permutation(split_result, permutation);
    uint32_t expected[2] = {1, 0};
    for (int i = 0; i < 2; i++) {
        if (permutation[i] != expected[i]) {
            fprintf(stderr, "Permutation at position %d not as expected, found %u expected %u\n", i,
                    permutation[i], expected[i]);
            goto cleanup;
        }
    }

    QkOpCounts counts = qk_circuit_count_ops(qc);
    if (counts.len != 1) {
        fprintf(stderr, "More than 1 type of gate in the circuit\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    for (size_t i = 0; i < counts.len; i++) {
        int gate = strcmp(counts.data[i].name, "unitary");
        if (gate != 0) {
            fprintf(stderr, "Gates outside expected set in output circuit\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        size_t count = counts.data[i].count;
        if (count != 2) {
            fprintf(stderr, "Unexpected gate counts found\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (inst.num_qubits != 1) {
            fprintf(stderr, "Gate %zu operates on more than 1 qubit: %u\n", i, inst.num_qubits);
            result = EqualityError;
            goto ops_cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }

ops_cleanup:
    qk_opcounts_clear(&counts);
cleanup:
    qk_transpile_layout_free(split_result);
    qk_circuit_free(qc);
    return result;
}

static int test_split_2q_unitaries_no_unitaries(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(5, "qr");
    qk_dag_add_quantum_register(dag, qr);
    for (uint32_t i = 0; i < qk_dag_num_qubits(dag) - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        for (uint32_t j = 0; j < i + 1; j++) {
            qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);
        }
    }
    QkTranspileLayout *split_result = qk_transpiler_pass_split_2q_unitaries(dag, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        fprintf(stderr, "Permutation returned for a circuit that shouldn't split\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    size_t num_ops = qk_dag_num_op_nodes(dag);
    uint32_t *ops = malloc(num_ops * sizeof(*ops));
    qk_dag_topological_op_nodes(dag, ops);
    QkGate first_gate = qk_dag_op_node_gate_op(dag, ops[0], NULL);
    for (size_t i = 1; i < num_ops; i++) {
        QkGate gate = qk_dag_op_node_gate_op(dag, ops[i], NULL);
        if (gate != first_gate) {
            fprintf(stderr, "More than 1 type of gate in DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
    for (size_t i = 0; i < num_ops; i++) {
        QkGate gate = qk_dag_op_node_gate_op(dag, ops[i], NULL);
        if (gate != QkGate_CX) {
            fprintf(stderr, "gates changed when there should be no DAG changes\n");
            result = EqualityError;
            goto ops_cleanup;
        }
    }
ops_cleanup:
    free(ops);
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_split_2q_unitaries_x_y_unitary(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, c0, neg_im, c0, c0, im, c0, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_dag_apply_unitary(dag, unitary, qargs, 2, false);
    QkTranspileLayout *split_result = qk_transpiler_pass_split_2q_unitaries(dag, 1 - 1e-16, true);
    int result = Ok;
    if (split_result != NULL) {
        result = EqualityError;
        fprintf(stderr, "Permutation returned for a circuit that isn't swap-equivalent\n");
        qk_transpile_layout_free(split_result);
        goto cleanup;
    }
    size_t num_ops = qk_dag_num_op_nodes(dag);
    uint32_t *ops = malloc(num_ops * sizeof(*ops));
    qk_dag_topological_op_nodes(dag, ops);
    if (num_ops != 2) {
        fprintf(stderr, "Unexpected gate counts found\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    QkOperationKind first_kind = qk_dag_op_node_kind(dag, ops[0]);
    for (size_t i = 0; i < num_ops; i++) {
        QkOperationKind kind = qk_dag_op_node_kind(dag, ops[i]);
        if (kind != first_kind) {
            fprintf(stderr, "More than 1 type of gate in DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        if (kind != QkOperationKind_Unitary) {
            fprintf(stderr, "Gates outside expected set in output DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        uint32_t nq = qk_dag_op_node_num_qubits(dag, ops[i]);
        if (nq != 1) {
            fprintf(stderr, "Gate %zu operates on more than 1 qubit: %u\n", i, nq);
            result = EqualityError;
            goto ops_cleanup;
        }
    }

ops_cleanup:
    free(ops);
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_split_2q_unitaries_swap_x_y_unitary(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);
    QkComplex64 c0 = {0., 0.};
    QkComplex64 neg_im = {0., -1.};
    QkComplex64 im = {0., 1.};
    QkComplex64 unitary[16] = {
        c0, c0, c0, neg_im, c0, neg_im, c0, c0, c0, c0, im, c0, im, c0, c0, c0,
    };
    uint32_t qargs[2] = {0, 1};
    qk_dag_apply_unitary(dag, unitary, qargs, 2, false);
    QkTranspileLayout *split_result = qk_transpiler_pass_split_2q_unitaries(dag, 1 - 1e-16, true);
    int result = Ok;
    if (split_result == NULL) {
        result = EqualityError;
        fprintf(stderr, "Permutation not returned for a circuit that should have one\n");
        goto cleanup;
    }
    uint32_t permutation[2];
    qk_transpile_layout_output_permutation(split_result, permutation);
    uint32_t expected[2] = {1, 0};
    for (int i = 0; i < 2; i++) {
        if (permutation[i] != expected[i]) {
            fprintf(stderr, "Permutation at position %d not as expected, found %u expected %u\n", i,
                    permutation[i], expected[i]);
            goto cleanup;
        }
    }

    size_t num_ops = qk_dag_num_op_nodes(dag);
    uint32_t *ops = malloc(num_ops * sizeof(*ops));
    qk_dag_topological_op_nodes(dag, ops);
    if (num_ops != 2) {
        fprintf(stderr, "Unexpected gate counts found\n");
        result = EqualityError;
        goto ops_cleanup;
    }
    QkOperationKind first_kind = qk_dag_op_node_kind(dag, ops[0]);
    for (size_t i = 0; i < num_ops; i++) {
        QkOperationKind kind = qk_dag_op_node_kind(dag, ops[i]);
        if (kind != first_kind) {
            fprintf(stderr, "More than 1 type of gate in DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        if (kind != QkOperationKind_Unitary) {
            fprintf(stderr, "Gates outside expected set in output DAG\n");
            result = EqualityError;
            goto ops_cleanup;
        }
        uint32_t nq = qk_dag_op_node_num_qubits(dag, ops[i]);
        if (nq != 1) {
            fprintf(stderr, "Gate %zu operates on more than 1 qubit: %u\n", i, nq);
            result = EqualityError;
            goto ops_cleanup;
        }
    }

ops_cleanup:
    free(ops);
cleanup:
    qk_transpile_layout_free(split_result);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

int test_split_2q_unitaries(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_standalone_split_2q_unitaries_no_unitaries);
    num_failed += RUN_TEST(test_standalone_split_2q_unitaries_x_y_unitary);
    num_failed += RUN_TEST(test_standalone_split_2q_unitaries_swap_x_y_unitary);
    num_failed += RUN_TEST(test_split_2q_unitaries_no_unitaries);
    num_failed += RUN_TEST(test_split_2q_unitaries_x_y_unitary);
    num_failed += RUN_TEST(test_split_2q_unitaries_swap_x_y_unitary);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
