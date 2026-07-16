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

static inline bool instructions_equal(const QkCircuitInstruction *left,
                                      const QkCircuitInstruction *right) {
    if (strcmp(left->name, right->name) || left->num_qubits != right->num_qubits ||
        left->num_clbits != right->num_clbits || left->num_params != right->num_params) {
        return false;
    }
    for (uint32_t i = 0; i < left->num_qubits; i++) {
        if (left->qubits[i] != right->qubits[i]) {
            return false;
        }
    }
    for (uint32_t i = 0; i < left->num_clbits; i++) {
        if (left->clbits[i] != right->clbits[i]) {
            return false;
        }
    }
    for (uint32_t i = 0; i < left->num_params; i++) {
        if (left->params[i] != right->params[i]) {
            return false;
        }
    }
    return true;
}

static int test_empty(void) {
    QkDag *dag = qk_dag_new();
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);

    if (num_qubits != 0) {
        printf("The number of qubits %u is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_quantum_reg(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_register");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    if (num_qubits != 1024) {
        printf("The number of qubits %u is not 1024\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_with_classical_reg(void) {
    QkDag *dag = qk_dag_new();
    QkClassicalRegister *cr = qk_classical_register_new(2048, "my_register");
    qk_dag_add_classical_register(dag, cr);
    uint32_t num_qubits = qk_dag_num_qubits(dag);
    uint32_t num_clbits = qk_dag_num_clbits(dag);
    qk_dag_free(dag);
    qk_classical_register_free(cr);
    if (num_qubits != 0) {
        printf("The number of qubits %u is not 0\n", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 2048) {
        printf("The number of clbits %u is not 2048\n", num_clbits);
        return EqualityError;
    }
    return Ok;
}

static int test_dag_to_circuit(void) {
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "q1");
    qk_dag_add_quantum_register(dag, qr);
    qk_quantum_register_free(qr);
    QkClassicalRegister *cr = qk_classical_register_new(1, "c1");
    qk_dag_add_classical_register(dag, cr);
    qk_classical_register_free(cr);

    qk_dag_apply_gate(dag, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);

    int result = Ok;

    QkCircuit *circuit = qk_dag_to_circuit(dag);
    qk_dag_free(dag);

    if (qk_circuit_num_qubits(circuit) != 2 || qk_circuit_num_clbits(circuit) != 1 ||
        qk_circuit_num_instructions(circuit) != 2) {
        printf("DAG to circuit conversion encountered an issue\n");
        result = EqualityError;
    }

    qk_circuit_free(circuit);
    return result;
}

static int test_dag_apply_gate(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 0) {
        printf("The number of op nodes %zu is not 0\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t h_bits[1] = {0};
    qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 1) {
        printf("The number of op nodes %zu is not 1\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t crx_bits[2] = {0, 1};
    double crx_params[1] = {0.12};
    uint32_t crx_node_idx = qk_dag_apply_gate(dag, QkGate_CRX, crx_bits, crx_params, true);

    num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 2) {
        printf("The number of op nodes %zu is not 2\n", num_ops);
        result = EqualityError;
        goto cleanup;
    }

    // Check the node's kind.
    if (qk_dag_op_node_kind(dag, crx_node_idx) != QkOperationKind_Gate) {
        printf("Expected gate kind\n");
        result = EqualityError;
        goto cleanup;
    }

    // Check the gate has the right number of params.
    uint32_t num_params = qk_dag_op_node_num_params(dag, crx_node_idx);
    if (num_params != 1) {
        printf("Expected num params 1 but got %u\n", num_params);
        result = EqualityError;
        goto cleanup;
    }

    // Make sure we can get the standard gate and params back from the node.
    double actual_crx_params[1];
    QkGate actual_gate = qk_dag_op_node_gate_op(dag, crx_node_idx, actual_crx_params);
    if (actual_gate != QkGate_CRX) {
        printf("Expected gate of type %u but got %u\n", QkGate_CRX, actual_gate);
        result = EqualityError;
        goto cleanup;
    }

    if (actual_crx_params[0] != crx_params[0]) {
        printf("Expected param %f but got %f\n", crx_params[0], actual_crx_params[0]);
        result = EqualityError;
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_op_node_bits_explicit(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t h_bits[1] = {0};
    uint32_t h_node_idx = qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    uint32_t num_qubits = qk_dag_op_node_num_qubits(dag, h_node_idx);
    if (num_qubits != 1) {
        printf("The number of qubits %u is not 1\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    if (qk_dag_op_node_qubits(dag, h_node_idx)[0] != h_bits[0]) {
        printf("Expected a single qubit of value %u but got %u\n", h_bits[0],
               qk_dag_op_node_qubits(dag, h_node_idx)[0]);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t num_clbits = qk_dag_op_node_num_clbits(dag, h_node_idx);
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t cx_bits[2] = {1, 0};
    uint32_t cx_node_idx = qk_dag_apply_gate(dag, QkGate_CX, cx_bits, NULL, true);

    num_qubits = qk_dag_op_node_num_qubits(dag, cx_node_idx);
    if (num_qubits != 2) {
        printf("The number of qubits %u is not 2\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *actual_cx_bits = qk_dag_op_node_qubits(dag, cx_node_idx);
    for (uint32_t i = 0; i < num_qubits; i++) {
        if (actual_cx_bits[i] != cx_bits[i]) {
            printf("Expected a qubit of value %u in position %u but got %u\n", cx_bits[0], i,
                   qk_dag_op_node_qubits(dag, cx_node_idx)[0]);
            result = EqualityError;
            goto cleanup;
        }
    }

    num_clbits = qk_dag_op_node_num_clbits(dag, cx_node_idx);
    if (num_clbits != 0) {
        printf("The number of clbits %u is not 0\n", num_clbits);
        result = EqualityError;
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static int test_dag_node_type(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(1, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);

    // Get the wire node indices for the qubit.
    uint32_t qubit_in_idx = qk_dag_qubit_in_node(dag, 0);
    uint32_t qubit_out_idx = qk_dag_qubit_out_node(dag, 0);

    // Get the wire node indices for the clbit.
    uint32_t clbit_in_idx = qk_dag_clbit_in_node(dag, 0);
    uint32_t clbit_out_idx = qk_dag_clbit_out_node(dag, 0);

    // Add an operation node and save the node index.
    uint32_t h_bits[1] = {0};
    uint32_t h_node_idx = qk_dag_apply_gate(dag, QkGate_H, h_bits, NULL, false);

    QkDagNodeType node_type = qk_dag_node_type(dag, qubit_in_idx);
    if (node_type != QkDagNodeType_QubitIn) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_QubitIn, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, qubit_out_idx);
    if (node_type != QkDagNodeType_QubitOut) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_QubitOut, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, clbit_in_idx);
    if (node_type != QkDagNodeType_ClbitIn) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_ClbitIn, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, clbit_out_idx);
    if (node_type != QkDagNodeType_ClbitOut) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_ClbitOut, node_type);
        result = EqualityError;
        goto cleanup;
    }

    node_type = qk_dag_node_type(dag, h_node_idx);
    if (node_type != QkDagNodeType_Operation) {
        printf("Expected node type %d but got %d\n", QkDagNodeType_Operation, node_type);
        result = EqualityError;
    }

cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);
    return result;
}

static int test_dag_endpoint_node_value(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(2, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);

    // For each bit (we have two qubits and two clbits!), check that the wire values
    // are correct.
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t qubit_in_idx = qk_dag_qubit_in_node(dag, i);
        uint32_t qubit_out_idx = qk_dag_qubit_out_node(dag, i);
        uint32_t clbit_in_idx = qk_dag_clbit_in_node(dag, i);
        uint32_t clbit_out_idx = qk_dag_clbit_out_node(dag, i);

        uint32_t actual = qk_dag_wire_node_value(dag, qubit_in_idx);
        if (actual != i) {
            printf("Expected wire endpoint qubit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, qubit_out_idx);
        if (actual != i) {
            printf("Expected wire endpoint qubit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, clbit_in_idx);
        if (actual != i) {
            printf("Expected wire endpoint clbit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
        actual = qk_dag_wire_node_value(dag, clbit_out_idx);
        if (actual != i) {
            printf("Expected wire endpoint clbit value to be %u but got %u\n", i, actual);
            result = EqualityError;
            goto cleanup;
        }
    }
cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);
    return result;
}

static int test_dag_global_phase(void) {
    int result = Ok;

    QkDag *dag = qk_dag_new();

    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);

    // Default global phase should be 0.0
    QkParam *out_phase = qk_dag_global_phase(dag);
    double out_phase_val = qk_param_as_real(out_phase);
    qk_param_free(out_phase);
    if (out_phase_val != 0.0) {
        printf("Expected 0.0 for global phase, found %f\n", out_phase_val);
        result = EqualityError;
        goto cleanup;
    }

    // Numeric global phase parameter
    QkParam *theta = qk_param_from_double(1.23);
    qk_dag_set_global_phase(dag, theta);
    qk_param_free(theta);

    out_phase = qk_dag_global_phase(dag);
    out_phase_val = qk_param_as_real(out_phase);
    qk_param_free(out_phase);

    if (out_phase_val != 1.23) {
        printf("Expected 1.23 for global phase, found %f\n", out_phase_val);
        result = EqualityError;
        goto cleanup;
    }

    // Symbolic global phase parameter
    theta = qk_param_new_symbol("theta");
    if (qk_dag_set_global_phase(dag, theta) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    qk_param_free(theta);

    QkCircuit *qc = qk_dag_to_circuit(dag);
    size_t num_symbols = qk_circuit_num_param_symbols(qc);
    qk_circuit_free(qc);
    if (num_symbols != 1) {
        printf("Expected 1 symbol, found %zu\n", num_symbols);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_dag_free(dag);
    return result;
}

static int test_dag_get_instruction(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    QkClassicalRegister *cr = qk_classical_register_new(2, "cr");
    qk_dag_add_quantum_register(dag, qr);
    qk_dag_add_classical_register(dag, cr);
    qk_quantum_register_free(qr);
    qk_classical_register_free(cr);

    uint32_t args[2] = {0, 1};
    uint32_t cx_node = qk_dag_apply_gate(dag, QkGate_CX, args, NULL, false);
    uint32_t h_node = qk_dag_apply_gate(dag, QkGate_H, args, NULL, true);
    // If the pointer is null, the number of qubits shouldn't be read.
    uint32_t final_barrier_node = qk_dag_apply_barrier(dag, NULL, 16000, false);
    uint32_t measure_0_node = qk_dag_apply_measure(dag, 0, 0, false);
    uint32_t measure_1_node = qk_dag_apply_measure(dag, 1, 1, false);
    uint32_t front_barrier_node = qk_dag_apply_barrier(dag, args, 2, true);
    uint32_t reset_0_node = qk_dag_apply_reset(dag, 0, true);

    uint32_t indices[] = {reset_0_node,       front_barrier_node, h_node,        cx_node,
                          final_barrier_node, measure_0_node,     measure_1_node};
    const char *desc[] = {"reset 0",       "front barrier", "h",        "cx",
                          "final barrier", "measure 0",     "measure 1"};
    QkCircuitInstruction expected[] = {
        {.name = "reset",
         .qubits = args,
         .num_qubits = 1,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "barrier",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "h",
         .qubits = &args[0],
         .num_qubits = 1,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "cx",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "barrier",
         .qubits = args,
         .num_qubits = 2,
         .clbits = NULL,
         .num_clbits = 0,
         .params = NULL,
         .num_params = 0},
        {.name = "measure",
         .qubits = &args[0],
         .num_qubits = 1,
         .clbits = &args[0],
         .num_clbits = 1,
         .params = NULL,
         .num_params = 0},
        {.name = "measure",
         .qubits = &args[1],
         .num_qubits = 1,
         .clbits = &args[1],
         .num_clbits = 1,
         .params = NULL,
         .num_params = 0},
    };

    QkCircuitInstruction inst;
    for (size_t i = 0; i < sizeof(indices) / sizeof(*indices); i++) {
        qk_dag_get_instruction(dag, indices[i], &inst);
        if (!instructions_equal(&inst, &expected[i])) {
            printf("%s: mismatched instruction: %s\n", __func__, desc[i]);
            result = EqualityError;
            goto cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }
cleanup:
    qk_dag_free(dag);
    return result;
}

static int test_dag_topological_op_nodes(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t qubit[1] = {0};
    uint32_t h_gate_idx = qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
    uint32_t s_gate_idx = qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);
    uint32_t t_gate_idx = qk_dag_apply_gate(dag, QkGate_T, qubit, NULL, true);

    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 3) {
        printf("The number of op nodes %zu shouldn't be 0\n", num_ops);
        result = EqualityError;
        goto early_cleanup;
    }

    uint32_t *out_order = malloc(sizeof(uint32_t) * num_ops);
    qk_dag_topological_op_nodes(dag, out_order);

    if (out_order[0] != t_gate_idx) {
        printf("Expected gate index %u but got %u\n", t_gate_idx, out_order[0]);
        result = EqualityError;
        goto cleanup;
    }

    if (out_order[1] != h_gate_idx) {
        printf("Expected gate index %u but got %u\n", h_gate_idx, out_order[1]);
        result = EqualityError;
        goto cleanup;
    }

    if (out_order[2] != s_gate_idx) {
        printf("Expected gate index %u but got %u\n", s_gate_idx, out_order[2]);
        result = EqualityError;
    }

cleanup:
    free(out_order);

early_cleanup:
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

static inline QkComplex64 complex_mul(QkComplex64 left, QkComplex64 right) {
    return (QkComplex64){left.re * right.re - left.im * right.im,
                         left.re * right.im + left.im * right.re};
}

static void kron(const QkComplex64 *left, size_t left_qubits, const QkComplex64 *right,
                 size_t right_qubits, QkComplex64 *out) {
    size_t left_dim = 1LLU << left_qubits;
    size_t right_dim = 1LLU << right_qubits;
    size_t out_dim = 1LLU << (left_qubits + right_qubits);
    for (size_t left_row = 0; left_row < left_dim; left_row++) {
        for (size_t right_row = 0; right_row < right_dim; right_row++) {
            size_t out_row = right_dim * left_row + right_row;
            for (size_t left_col = 0; left_col < left_dim; left_col++) {
                for (size_t right_col = 0; right_col < right_dim; right_col++) {
                    size_t out_col = right_dim * left_col + right_col;
                    out[out_dim * out_row + out_col] =
                        complex_mul(left[left_dim * left_row + left_col],
                                    right[right_dim * right_row + right_col]);
                }
            }
        }
    }
}

static int check_unitary(const char *func, QkDag *dag, QkComplex64 scalar, const char *pauli,
                         uint32_t *qubits, bool front) {
    int res = Ok;
    static const QkComplex64 mat_i[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    static const QkComplex64 mat_x[4] = {{0, 0}, {1, 0}, {1, 0}, {0, 0}};
    static const QkComplex64 mat_y[4] = {{0, 0}, {0, -1}, {0, 1}, {0, 0}};
    static const QkComplex64 mat_z[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    uint32_t num_qubits = (uint32_t)strlen(pauli);
    size_t dim = 1LLU << num_qubits;
    const QkComplex64 *mat;
    QkComplex64 *tmp;
    QkComplex64 *cur = malloc(sizeof(*cur) * dim * dim);
    QkComplex64 *next = malloc(sizeof(*next) * dim * dim);
    cur[0] = scalar;
    for (uint32_t qubit = 0; qubit < num_qubits; qubit++) {
        switch (pauli[num_qubits - qubit - 1]) {
        case 'I': {
            mat = mat_i;
            break;
        }
        case 'X': {
            mat = mat_x;
            break;
        }
        case 'Y': {
            mat = mat_y;
            break;
        }
        case 'Z': {
            mat = mat_z;
            break;
        }
        default: {
            res = EqualityError;
            printf("%s: bad pauli string '%s'\n", func, pauli);
            goto cleanup;
        }
        }
        kron(mat, 1, cur, qubit, next);
        tmp = cur;
        cur = next;
        next = tmp;
    }
    // Zero out the other array, just to ensure that we've got clear invalid things to test
    // against.
    memset(next, 0, sizeof(*next) * dim * dim);
    uint32_t node = qk_dag_apply_unitary(dag, cur, qubits, num_qubits, front);
    QkOperationKind kind = qk_dag_op_node_kind(dag, node);
    if (kind != QkOperationKind_Unitary) {
        res = EqualityError;
        printf("%s: %s kind incorrect: %d\n", func, pauli, kind);
        goto cleanup;
    }
    qk_dag_op_node_unitary(dag, node, next);
    if (memcmp(cur, next, sizeof(*cur) * dim * dim)) {
        res = EqualityError;
        printf("%s: %s matrices unequal\n", func, pauli);
        goto cleanup;
    }
cleanup:
    free(cur);
    free(next);
    return res;
}

static int test_unitary_gates(void) {
    int res = Ok;
    uint32_t qubits[] = {0, 1, 2};

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(3, "q");
    qk_dag_add_quantum_register(dag, qr);
    qk_quantum_register_free(qr);

    if ((res = check_unitary(__func__, dag, (QkComplex64){1, 0}, "X", qubits, false)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){1, 0}, "Y", qubits, true)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){0, 1}, "XY", qubits, false)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){-1, 0}, "YZ", &qubits[1], true)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){0, -1}, "XYZ", qubits, false)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){1, 0}, "YZZ", qubits, false)))
        goto cleanup;
    if ((res = check_unitary(__func__, dag, (QkComplex64){1, 0}, "", qubits, false)))
        goto cleanup;
    // Check that nothing bad happens if we use a null pointer for no qubits.
    if ((res = check_unitary(__func__, dag, (QkComplex64){0, 1}, "", NULL, false)))
        goto cleanup;

cleanup:
    qk_dag_free(dag);
    return res;
}

static int test_substitute_node_with_dag(void) {
    int res = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t qubit[1] = {0};
    uint32_t node_to_replace = qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
    qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);

    // Build replacement dag for H
    QkDag *replacement = qk_dag_new();
    QkQuantumRegister *replacement_qr = qk_quantum_register_new(1, "other");
    qk_dag_add_quantum_register(replacement, replacement_qr);
    double pi_param[1] = {
        3.14159,
    };
    qk_dag_apply_gate(replacement, QkGate_RZ, qubit, pi_param, false);
    qk_dag_apply_gate(replacement, QkGate_SX, qubit, NULL, false);
    qk_dag_apply_gate(replacement, QkGate_RZ, qubit, pi_param, false);
    qk_dag_substitute_node_with_dag(dag, node_to_replace, replacement);

    if (qk_dag_num_op_nodes(dag) != 4) {
        res = EqualityError;
        printf("Number of instructions is %zu but expected 4\n", qk_dag_num_op_nodes(dag));
        goto cleanup;
    }

cleanup:
    // Free the replacement dag, register, dag, and register
    qk_quantum_register_free(replacement_qr);
    qk_dag_free(replacement);
    qk_quantum_register_free(qr);
    qk_dag_free(dag);

    return res;
}

/*
 * Test qk_dag_successors and qk_dag_predecessors
 */
static int test_dag_node_neighbors(void) {
    int result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
    qk_dag_add_quantum_register(dag, qr);
    qk_quantum_register_free(qr);

    uint32_t node_h = qk_dag_apply_gate(dag, QkGate_H, (uint32_t[]){0}, NULL, false);
    uint32_t node_ccx = qk_dag_apply_gate(dag, QkGate_CCX, (uint32_t[]){0, 1, 2}, NULL, false);
    uint32_t node_cx = qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){1, 2}, NULL, false);

    // H node
    QkDagNeighbors successors = qk_dag_successors(dag, node_h);
    QkDagNeighbors predecessors = qk_dag_predecessors(dag, node_h);
    if (successors.num_neighbors != 1 || successors.neighbors[0] != node_ccx ||
        predecessors.num_neighbors != 1 ||
        qk_dag_node_type(dag, predecessors.neighbors[0]) != QkDagNodeType_QubitIn) {
        printf("Incorrect neighbors information for the H node!\n");
        result = EqualityError;
        goto cleanup;
    }
    qk_dag_neighbors_clear(&successors);
    qk_dag_neighbors_clear(&predecessors);

    if (successors.neighbors != NULL || successors.num_neighbors != 0) {
        printf("qk_dag_neighbors_clear didn't work!\n");
        result = RuntimeError;
        goto cleanup;
    }

    // CCX node
    successors = qk_dag_successors(dag, node_ccx);
    predecessors = qk_dag_predecessors(dag, node_ccx);
    if (successors.num_neighbors != 2 || // CX is counted as a unique successor
        successors.neighbors[0] != node_cx ||
        qk_dag_node_type(dag, successors.neighbors[1]) != QkDagNodeType_QubitOut ||
        predecessors.num_neighbors != 3 ||
        qk_dag_node_type(dag, predecessors.neighbors[0]) != QkDagNodeType_QubitIn ||
        qk_dag_node_type(dag, predecessors.neighbors[1]) != QkDagNodeType_QubitIn ||
        predecessors.neighbors[2] != node_h) {
        printf("Incorrect neighbors information for the CCX node!\n");
        result = EqualityError;
        goto cleanup;
    }
    qk_dag_neighbors_clear(&successors);
    qk_dag_neighbors_clear(&predecessors);

    // CX node
    successors = qk_dag_successors(dag, node_cx);
    predecessors = qk_dag_predecessors(dag, node_cx);
    if (successors.num_neighbors != 2 ||
        qk_dag_node_type(dag, successors.neighbors[0]) != QkDagNodeType_QubitOut ||
        qk_dag_node_type(dag, successors.neighbors[1]) != QkDagNodeType_QubitOut ||
        predecessors.num_neighbors != 1 || // CCX is counted as a unique predecessor
        predecessors.neighbors[0] != node_ccx) {
        printf("Incorrect neighbors information for the CX node!\n");
        result = EqualityError;
    }

cleanup:
    qk_dag_neighbors_clear(&successors);
    qk_dag_neighbors_clear(&predecessors);
    qk_dag_free(dag);
    return result;
}

static int test_dag_copy_empty_like(void) {
    int result = Ok;

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "my_register");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t qubit[1] = {0};
    qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);

    QkDag *copied_dag = qk_dag_copy_empty_like(dag, QkVarsMode_Alike, QkBlocksMode_Drop);

    size_t num_ops_in_dag = qk_dag_num_op_nodes(dag);               // not 0
    size_t num_ops_in_copied_dag = qk_dag_num_op_nodes(copied_dag); // 0

    if (num_ops_in_dag == 0) {
        printf("Expected the original DAG to remain unchanged, but it now empty\n");
        result = EqualityError;
        goto cleanup;
    }

    if (num_ops_in_copied_dag != 0) {
        printf("Expected no operations in the copied-empty-like DAG, but got %zu\n",
               num_ops_in_copied_dag);
        result = EqualityError;
    }

cleanup:
    qk_quantum_register_free(qr);
    qk_dag_free(dag);
    qk_dag_free(copied_dag);

    return result;
}

static int test_dag_compose(void) {
    int result = Ok;
    // Build a dag of the following structure
    //
    //  rqr_0: |0>──■───────
    //              │  ┌───┐
    //  rqr_1: |0>──┼──┤ X ├
    //              │  ├───┤
    //  rqr_2: |0>──┼──┤ Y ├
    //            ┌─┴─┐└───┘
    //  rqr_3: |0>┤ X ├─────
    //            └───┘┌───┐
    //  rqr_4: |0>─────┤ Z ├
    //                 └───┘
    QkDag *dag_right = qk_dag_new();
    QkQuantumRegister *rqr = qk_quantum_register_new(5, "rqr");
    qk_dag_add_quantum_register(dag_right, rqr);
    qk_dag_apply_gate(dag_right, QkGate_CX, (uint32_t[]){0, 3}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_Y, (uint32_t[]){2}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_Z, (uint32_t[]){4}, NULL, false);

    // Build a second dag to compose on
    //             ┌───┐
    // lqr_0: |0>──┤ H ├───
    //             ├───┤
    // lqr_1: |0>──┤ X ├───
    //           ┌─┴───┴──┐
    // lqr_2: |0>┤ P(0.1) ├
    //           └────────┘
    // lqr_3: |0>────■─────
    //             ┌─┴─┐
    // lqr_4: |0>──┤ X ├───
    //             └───┘
    QkDag *dag_left = qk_dag_new();
    QkQuantumRegister *lqr = qk_quantum_register_new(5, "lqr");
    qk_dag_add_quantum_register(dag_left, lqr);
    qk_dag_apply_gate(dag_left, QkGate_CX, (uint32_t[]){3, 4}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_Phase, (uint32_t[]){2}, (double[]){0.1}, false);
    size_t left_op_nodes = qk_dag_num_op_nodes(dag_left);

    // Call compose

    // With resulting circuit
    //             ┌───┐
    // lqr_0: |0>──┤ H ├─────■───────
    //             ├───┤     │  ┌───┐
    // lqr_1: |0>──┤ X ├─────┼──┤ X ├
    //           ┌─┴───┴──┐  │  ├───┤
    // lqr_2: |0>┤ P(0.1) ├──┼──┤ Y ├
    //           └────────┘┌─┴─┐└───┘
    // lqr_3: |0>────■─────┤ X ├─────
    //             ┌─┴─┐   └───┘┌───┐
    // lqr_4: |0>──┤ X ├────────┤ Z ├
    //             └───┘        └───┘
    // lcr_0: 0 ═══════════════════════
    //
    // lcr_1: 0 ═══════════════════════
    QkExitCode res = qk_dag_compose(dag_left, dag_right, NULL, NULL);
    if (res != QkExitCode_Success) {
        result = EqualityError;
        printf("Error during compose. The composible dag possibly exceeded the allowed number of "
               "qubits.\n");
        goto cleanup;
    }

    // We need to wait until we get ``qk_dag_topological_op_nodes`` to check the specific
    // operations. For now, let's compare the number of `op_nodes` as a reference.
    size_t new_op_nodes = qk_dag_num_op_nodes(dag_left);
    size_t right_op_nodes = qk_dag_num_op_nodes(dag_right);

    if (left_op_nodes + right_op_nodes != new_op_nodes) {
        result = EqualityError;
        printf("The operations did not get composed onto the left dag correctly. Expected %zu "
               "operations, got %zu.\n",
               left_op_nodes + right_op_nodes, new_op_nodes);
    }

    // Create a comparison dag
    QkDag *dag_expected = qk_dag_new();
    qk_dag_add_quantum_register(dag_expected, lqr);
    qk_dag_apply_gate(dag_expected, QkGate_CX, (uint32_t[]){3, 4}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Phase, (uint32_t[]){2}, (double[]){0.1}, false);
    qk_dag_apply_gate(dag_expected, QkGate_CX, (uint32_t[]){0, 3}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Y, (uint32_t[]){2}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Z, (uint32_t[]){4}, NULL, false);
    size_t expected_op_nodes = qk_dag_num_op_nodes(dag_left);

    if (new_op_nodes != expected_op_nodes) {
        result = EqualityError;
        printf("The operations did not get composed onto the left dag correctly. Expected %zu "
               "operations, got %zu.\n",
               left_op_nodes + right_op_nodes, new_op_nodes);
        goto expect_cleanup;
    }

    uint32_t res_op_nodes[9];
    uint32_t exp_op_nodes[9];

    qk_dag_topological_op_nodes(dag_left, res_op_nodes);
    qk_dag_topological_op_nodes(dag_expected, exp_op_nodes);

    // Both operations should have the same amount of nodes at this point
    for (int node_idx = 0; node_idx < (int)new_op_nodes; node_idx++) {
        double exp_param[1];
        double left_param[1];
        uint16_t node_idx_exp = exp_op_nodes[node_idx];
        uint16_t node_idx_left = res_op_nodes[node_idx];
        size_t exp_num_param = qk_dag_op_node_num_params(dag_expected, node_idx_exp);
        size_t left_num_param = qk_dag_op_node_num_params(dag_left, node_idx_left);
        QkGate exp_gate = qk_dag_op_node_gate_op(dag_expected, node_idx_exp, exp_param);
        QkGate left_gate = qk_dag_op_node_gate_op(dag_left, node_idx_left, left_param);

        // Check gate instances
        if (exp_gate != left_gate) {
            result = EqualityError;
            printf("Incorrect operation found, expected %d, got %d.\n", exp_gate, left_gate);
            goto loop_cleanup;
        }

        if (exp_num_param != left_num_param) {
            result = EqualityError;
            printf(
                "Correct operation with mismatched number of parameters, expected %zu, got %zu.\n",
                exp_num_param, left_num_param);
            goto loop_cleanup;
        }

        for (int param_idx = 0; param_idx < (int)left_num_param; param_idx++) {
            if (exp_param[param_idx] != left_param[param_idx]) {
                result = EqualityError;
                printf("Correct operation with mismatched parameter, expected %f, got %f.\n",
                       exp_param[param_idx], left_param[param_idx]);
                goto loop_cleanup;
            }
        }

        size_t exp_num_qubits = qk_dag_op_node_num_qubits(dag_expected, node_idx_exp);
        size_t left_num_qubits = qk_dag_op_node_num_qubits(dag_left, node_idx_left);
        if (exp_num_qubits != left_num_qubits) {
            result = EqualityError;
            printf("Correct operation with mismatched number of qubits, expected %zu, got %zu.\n",
                   exp_num_qubits, left_num_qubits);
            goto loop_cleanup;
        }

        const uint32_t *expected_qubits = qk_dag_op_node_qubits(dag_expected, node_idx_exp);
        const uint32_t *left_qubits = qk_dag_op_node_qubits(dag_left, node_idx_left);
        if (memcmp(expected_qubits, left_qubits, exp_num_qubits * sizeof(*expected_qubits))) {
            result = EqualityError;
            printf("Correct operation with mismatched qubits\n");
            goto loop_cleanup;
        }

    loop_cleanup:
        if (result != Ok) {
            break;
        }
    }
expect_cleanup:
    qk_dag_free(dag_expected);
cleanup:
    qk_dag_free(dag_left);
    qk_dag_free(dag_right);
    qk_quantum_register_free(lqr);
    qk_quantum_register_free(rqr);
    return result;
}

static int test_dag_compose_permuted(void) {
    int result = Ok;
    // Build a dag of the following structure
    //
    //  rqr_0: ──■──────────
    //           │  ┌───┐
    //  rqr_1: ──┼──┤ X ├───
    //           │  ├───┤
    //  rqr_2: ──┼──┤ Y ├───
    //         ┌─┴─┐└───┘┌─┐
    //  rqr_3: ┤ X ├─────┤M├
    //         └───┘┌───┐└╥┘
    //  rqr_4: ─────┤ Z ├─╫─
    //              └───┘ ║
    //  rcr: 2/═══════════╩═
    //                    0
    QkDag *dag_right = qk_dag_new();
    QkQuantumRegister *rqr = qk_quantum_register_new(5, "rqr");
    QkClassicalRegister *rcr = qk_classical_register_new(2, "rcr");
    qk_dag_add_quantum_register(dag_right, rqr);
    qk_dag_add_classical_register(dag_right, rcr);
    qk_dag_apply_gate(dag_right, QkGate_CX, (uint32_t[]){0, 3}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_Y, (uint32_t[]){2}, NULL, false);
    qk_dag_apply_gate(dag_right, QkGate_Z, (uint32_t[]){4}, NULL, false);
    qk_dag_apply_measure(dag_right, 3, 0, false);

    // Build a second dag to compose on
    //          ┌───┐
    // lqr_0: ──┤ H ├───
    //          ├───┤
    // lqr_1: ──┤ X ├───
    //        ┌─┴───┴──┐
    // lqr_2: ┤ P(0.1) ├
    //        └────────┘
    // lqr_3: ────■─────
    //          ┌─┴─┐
    // lqr_4: ──┤ X ├───
    //          └───┘
    // lcr: 2/══════════
    QkDag *dag_left = qk_dag_new();
    QkQuantumRegister *lqr = qk_quantum_register_new(5, "lqr");
    QkClassicalRegister *lcr = qk_classical_register_new(2, "lcr");
    qk_dag_add_quantum_register(dag_left, lqr);
    qk_dag_add_classical_register(dag_left, lcr);
    qk_dag_apply_gate(dag_left, QkGate_CX, (uint32_t[]){3, 4}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag_left, QkGate_Phase, (uint32_t[]){2}, (double[]){0.1}, false);
    size_t left_op_nodes = qk_dag_num_op_nodes(dag_left);

    // Call compose

    // With resulting circuit
    //          ┌───┐   ┌───┐
    // lqr_0: ──┤ H ├───┤ Z ├───
    //          ├───┤   ├───┤
    // lqr_1: ──┤ X ├───┤ X ├───
    //        ┌─┴───┴──┐├───┤
    // lqr_2: ┤ P(0.1) ├┤ Y ├───
    //        └────────┘└───┘
    // lqr_3: ────■───────■─────
    //          ┌─┴─┐   ┌─┴─┐┌─┐
    // lqr_4: ──┤ X ├───┤ X ├┤M├
    //          └───┘   └───┘└╥┘
    // lcr: 2/════════════════╩═
    //                        1
    uint32_t qubits[5] = {3, 1, 2, 4, 0};
    uint32_t clbits[2] = {1, 0};
    QkExitCode res = qk_dag_compose(dag_left, dag_right, qubits, clbits);
    if (res != QkExitCode_Success) {
        result = EqualityError;
        printf("Error during compose. The composible dag possibly exceeded the allowed number of "
               "qubits..\n");
        goto cleanup;
    }

    // We need to wait until we get ``qk_dag_topological_op_nodes`` to check the specific
    // operations. For now, let's compare the number of `op_nodes` as a reference.
    size_t new_op_nodes = qk_dag_num_op_nodes(dag_left);
    size_t right_op_nodes = qk_dag_num_op_nodes(dag_right);

    if (left_op_nodes + right_op_nodes != new_op_nodes) {
        result = EqualityError;
        printf("The operations did not get composed onto the left dag correctly. Expected %zu "
               "operations, got %zu.\n",
               left_op_nodes + right_op_nodes, new_op_nodes);
    }

    // Create a comparison dag
    QkDag *dag_expected = qk_dag_new();
    qk_dag_add_quantum_register(dag_expected, lqr);
    qk_dag_add_classical_register(dag_expected, lcr);
    qk_dag_apply_gate(dag_expected, QkGate_CX, (uint32_t[]){3, 4}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Phase, (uint32_t[]){2}, (double[]){0.1}, false);
    qk_dag_apply_gate(dag_expected, QkGate_CX, (uint32_t[]){3, 4}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_X, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Y, (uint32_t[]){2}, NULL, false);
    qk_dag_apply_gate(dag_expected, QkGate_Z, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_measure(dag_expected, 4, 1, false);
    size_t expected_op_nodes = qk_dag_num_op_nodes(dag_left);

    if (new_op_nodes != expected_op_nodes) {
        result = EqualityError;
        printf("The operations did not get composed onto the left dag correctly. Expected %zu "
               "operations, got %zu.\n",
               left_op_nodes + right_op_nodes, new_op_nodes);
        goto expect_cleanup;
    }

    uint32_t res_op_nodes[9];
    uint32_t exp_op_nodes[9];

    qk_dag_topological_op_nodes(dag_left, res_op_nodes);
    qk_dag_topological_op_nodes(dag_expected, exp_op_nodes);

    // Both operations should have the same amount of nodes at this point
    for (int node_idx = 0; node_idx < (int)new_op_nodes; node_idx++) {
        uint16_t node_idx_exp = exp_op_nodes[node_idx];
        uint16_t node_idx_left = res_op_nodes[node_idx];
        QkOperationKind exp_kind = qk_dag_op_node_kind(dag_expected, node_idx_exp);
        QkOperationKind exp_left = qk_dag_op_node_kind(dag_left, node_idx_left);

        if (exp_kind != exp_left) {
            result = EqualityError;
            printf("Incorrect operation type found. Expected: %d got %d.\n", exp_kind, exp_left);
            goto loop_cleanup;
        }
        if (exp_kind == QkOperationKind_Gate) {
            double exp_param[1];
            double left_param[1];
            size_t exp_num_param = qk_dag_op_node_num_params(dag_expected, node_idx_exp);
            size_t left_num_param = qk_dag_op_node_num_params(dag_left, node_idx_left);
            QkGate exp_gate = qk_dag_op_node_gate_op(dag_expected, node_idx_exp, exp_param);
            QkGate left_gate = qk_dag_op_node_gate_op(dag_left, node_idx_left, left_param);

            // Check gate instances
            if (exp_gate != left_gate) {
                result = EqualityError;
                printf("Incorrect operation found, expected %d, got %d.\n", exp_gate, left_gate);
                goto loop_cleanup;
            }

            if (exp_num_param != left_num_param) {
                result = EqualityError;
                printf("Correct operation with mismatched number of parameters, expected %zu, got "
                       "%zu.\n",
                       exp_num_param, left_num_param);
                goto loop_cleanup;
            }

            for (int param_idx = 0; param_idx < (int)left_num_param; param_idx++) {
                if (exp_param[param_idx] != left_param[param_idx]) {
                    result = EqualityError;
                    printf("Correct operation with mismatched parameter, expected %f, got %f.\n",
                           exp_param[param_idx], left_param[param_idx]);
                    goto loop_cleanup;
                }
            }
        }

        size_t exp_num_qubits = qk_dag_op_node_num_qubits(dag_expected, node_idx_exp);
        size_t left_num_qubits = qk_dag_op_node_num_qubits(dag_left, node_idx_left);
        if (exp_num_qubits != left_num_qubits) {
            result = EqualityError;
            printf("Correct operation with mismatched number of qubits, expected %zu, got %zu.\n",
                   exp_num_qubits, left_num_qubits);
            goto loop_cleanup;
        }

        const uint32_t *expected_qubits = qk_dag_op_node_qubits(dag_expected, node_idx_exp);
        const uint32_t *left_qubits = qk_dag_op_node_qubits(dag_left, node_idx_left);
        if (memcmp(expected_qubits, left_qubits, exp_num_qubits * sizeof(*expected_qubits))) {
            result = EqualityError;
            printf("Correct operation with mismatched qubits.\n");
            goto loop_cleanup;
        }

        size_t exp_num_clbits = qk_dag_op_node_num_clbits(dag_expected, node_idx_exp);
        size_t left_num_clbits = qk_dag_op_node_num_clbits(dag_left, node_idx_left);
        if (exp_num_clbits != left_num_clbits) {
            result = EqualityError;
            printf("Correct operation with mismatched number of clbits, expected %zu, got %zu.\n",
                   exp_num_clbits, left_num_clbits);
            goto loop_cleanup;
        }

        const uint32_t *expected_clbits = qk_dag_op_node_clbits(dag_expected, node_idx_exp);
        const uint32_t *left_clbits = qk_dag_op_node_clbits(dag_left, node_idx_left);
        if (memcmp(expected_clbits, left_clbits, exp_num_clbits * sizeof(*expected_clbits))) {
            result = EqualityError;
            printf("Correct operation with mismatched clbits.\n");
            goto loop_cleanup;
        }

    loop_cleanup:
        if (result != Ok) {
            break;
        }
    }
expect_cleanup:
    qk_dag_free(dag_expected);
cleanup:
    qk_dag_free(dag_left);
    qk_dag_free(dag_right);
    qk_quantum_register_free(lqr);
    qk_classical_register_free(lcr);
    qk_quantum_register_free(rqr);
    qk_classical_register_free(rcr);
    return result;
}

static int test_dag_replace_block_with_unitary(void) {
    int result = Ok;

    // Create a DAG with H, T, S, T, H gates on the second qubit
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qubit[1] = {1};
    qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);
    uint32_t idx1 = qk_dag_apply_gate(dag, QkGate_T, qubit, NULL, false);
    uint32_t idx2 = qk_dag_apply_gate(dag, QkGate_S, qubit, NULL, false);
    uint32_t idx3 = qk_dag_apply_gate(dag, QkGate_T, qubit, NULL, false);
    qk_dag_apply_gate(dag, QkGate_H, qubit, NULL, false);

    // Replace the inner T, S, T gates by a unitary gate (representing Z)
    uint32_t replaced_ids[3] = {idx1, idx2, idx3};
    static const QkComplex64 mat_z[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};
    uint32_t new_node_idx =
        qk_dag_replace_block_with_unitary(dag, 3, replaced_ids, mat_z, 1, qubit, false);

    // The resulting DAG should have 3 operations
    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 3) {
        result = EqualityError;
        printf("Number of instructions is %zu but expected 3\n", num_ops);
        goto cleanup;
    }

    // And the new operation must be unitary
    QkOperationKind new_node_kind = qk_dag_op_node_kind(dag, new_node_idx);
    if (new_node_kind != QkOperationKind_Unitary) {
        result = EqualityError;
        printf(
            "The new node with index %u has incorrect operation type: expected: %d but got %d.\n",
            new_node_idx, QkOperationKind_Unitary, new_node_kind);
    }

cleanup:
    qk_quantum_register_free(qr);
    qk_dag_free(dag);

    return result;
}

static int test_dag_replace_qubitless_block_with_unitary(void) {
    int result = Ok;

    // Create a DAG with H, GlobalPhase, CX
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);

    qk_dag_apply_gate(dag, QkGate_H, (uint32_t[]){1}, NULL, false);
    uint32_t idx = qk_dag_apply_gate(dag, QkGate_GlobalPhase, NULL, (double[]){0.0}, false);
    qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);

    // Replace the global phase gate by a 0-qubit unitary gate
    static const QkComplex64 identity_mat[1] = {{1, 0}};
    uint32_t new_node_idx =
        qk_dag_replace_block_with_unitary(dag, 1, (uint32_t[]){idx}, identity_mat, 0, NULL, false);

    // The resulting DAG should have 3 operations
    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 3) {
        result = EqualityError;
        printf("Number of instructions is %zu but expected 3\n", num_ops);
        goto cleanup;
    }

    // And the new operation must be unitary
    QkOperationKind new_node_kind = qk_dag_op_node_kind(dag, new_node_idx);
    if (new_node_kind != QkOperationKind_Unitary) {
        result = EqualityError;
        printf(
            "The new node with index %u has incorrect operation type: expected: %d but got %d.\n",
            new_node_idx, QkOperationKind_Unitary, new_node_kind);
    }

cleanup:
    qk_quantum_register_free(qr);
    qk_dag_free(dag);

    return result;
}

static int test_dag_replace_illegal_block_with_unitary(void) {
    int result = Ok;

    // Create a DAG with CX, H, CX
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);

    uint32_t idx1 = qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);
    qk_dag_apply_gate(dag, QkGate_H, (uint32_t[]){1}, NULL, false);
    uint32_t idx2 = qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);

    // Try to replace the block consisting of the two CX gates while setting cycle_check
    // to true. Note that this would introduce a cycle.
    static const QkComplex64 identity_mat_2[4] = {{1, 0}, {0, 0}, {0, 0}, {1, 0}};
    uint32_t new_node_idx = qk_dag_replace_block_with_unitary(
        dag, 2, (uint32_t[]){idx1, idx2}, identity_mat_2, 2, (uint32_t[]){0, 1}, true);

    // The returned node index should be UINT32_MAX, indicating that
    // replacing the provided node block would introduce a cycle.
    if (new_node_idx != UINT32_MAX) {
        result = EqualityError;
        printf("The new node has index %u but expected %u\n", new_node_idx, UINT32_MAX);
        goto cleanup;
    }

    // The original DAG should be left unchanged.
    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 3) {
        result = EqualityError;
        printf("Number of instructions is %zu but expected 3\n", num_ops);
        goto cleanup;
    }

cleanup:
    qk_quantum_register_free(qr);
    qk_dag_free(dag);

    return result;
}

static int test_dag_substitute_node_with_unitary(void) {
    int result = Ok;

    // Create a DAG with H, CX, Z, S gates
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2, "qr");
    qk_dag_add_quantum_register(dag, qr);

    qk_dag_apply_gate(dag, QkGate_H, (uint32_t[]){0}, NULL, false);
    qk_dag_apply_gate(dag, QkGate_CX, (uint32_t[]){0, 1}, NULL, false);
    uint32_t idx_z = qk_dag_apply_gate(dag, QkGate_Z, (uint32_t[]){1}, NULL, false);
    qk_dag_apply_gate(dag, QkGate_S, (uint32_t[]){1}, NULL, false);

    static const QkComplex64 mat[4] = {{1, 0}, {0, 0}, {0, 0}, {-1, 0}};

    // Replace the Z-gate by a unitary matrix
    qk_dag_substitute_node_with_unitary(dag, idx_z, mat, 1);

    // The resulting DAG should still have 4 operations
    size_t num_ops_z = qk_dag_num_op_nodes(dag);
    if (num_ops_z != 4) {
        result = EqualityError;
        printf("Number of instructions is %zu but expected 4\n", num_ops_z);
        goto cleanup;
    }

    // And the new operation must be unitary
    QkOperationKind new_node_kind_z = qk_dag_op_node_kind(dag, idx_z);
    if (new_node_kind_z != QkOperationKind_Unitary) {
        result = EqualityError;
        printf("The node with index %u has incorrect operation type: expected: %d but got %d.\n",
               idx_z, QkOperationKind_Unitary, new_node_kind_z);
    }

cleanup:
    qk_quantum_register_free(qr);
    qk_dag_free(dag);

    return result;
}

int test_dag(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_dag_with_quantum_reg);
    num_failed += RUN_TEST(test_dag_with_classical_reg);
    num_failed += RUN_TEST(test_dag_to_circuit);
    num_failed += RUN_TEST(test_dag_apply_gate);
    num_failed += RUN_TEST(test_dag_node_type);
    num_failed += RUN_TEST(test_dag_endpoint_node_value);
    num_failed += RUN_TEST(test_dag_global_phase);
    num_failed += RUN_TEST(test_op_node_bits_explicit);
    num_failed += RUN_TEST(test_dag_get_instruction);
    num_failed += RUN_TEST(test_dag_topological_op_nodes);
    num_failed += RUN_TEST(test_unitary_gates);
    num_failed += RUN_TEST(test_substitute_node_with_dag);
    num_failed += RUN_TEST(test_dag_node_neighbors);
    num_failed += RUN_TEST(test_dag_copy_empty_like);
    num_failed += RUN_TEST(test_dag_compose);
    num_failed += RUN_TEST(test_dag_compose_permuted);
    num_failed += RUN_TEST(test_dag_replace_block_with_unitary);
    num_failed += RUN_TEST(test_dag_replace_qubitless_block_with_unitary);
    num_failed += RUN_TEST(test_dag_replace_illegal_block_with_unitary);
    num_failed += RUN_TEST(test_dag_substitute_node_with_unitary);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
