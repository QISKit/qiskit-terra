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
#include <inttypes.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test the zero constructor.
 */
static int test_empty(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);

    QkOpCounts counts = qk_circuit_count_ops(qc);
    size_t opcount = counts.len;

    qk_opcounts_clear(&counts);
    qk_circuit_free(qc);

    if (opcount != 0) {
        printf("The operation count %zu is not 0", opcount);
        return EqualityError;
    }
    if (num_qubits != 0) {
        printf("The number of qubits %d is not 0", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %d is not 0", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_with_quantum_reg(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "my_little_register");
    qk_circuit_add_quantum_register(qc, qr);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);
    qk_quantum_register_free(qr);
    if (num_qubits != 1024) {
        printf("The number of qubits %d is not 1024", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 0) {
        printf("The number of clbits %d is not 0", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    QkCircuit *copy = qk_circuit_copy(qc);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        if (i % 2 == 0) {
            qk_circuit_gate(copy, QkGate_H, qubits, NULL);
        }
    }
    size_t num_instructions = qk_circuit_num_instructions(qc);
    size_t num_copy_instructions = qk_circuit_num_instructions(copy);
    qk_circuit_free(qc);
    qk_circuit_free(copy);
    if (num_instructions == num_copy_instructions) {
        printf("The number of instructions %zu is equal to the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_with_classical_reg(void) {
    QkCircuit *qc = qk_circuit_new(0, 0);
    QkClassicalRegister *cr = qk_classical_register_new(2048, "my_less_little_register");
    qk_circuit_add_classical_register(qc, cr);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);
    qk_classical_register_free(cr);
    if (num_qubits != 0) {
        printf("The number of qubits %d is not 0", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 2048) {
        printf("The number of clbits %d is not 2048", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy_with_instructions(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(qc, QkGate_H, qubits, NULL);
    }
    QkCircuit *copy = qk_circuit_copy(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    size_t num_copy_instructions = qk_circuit_num_instructions(copy);
    if (num_instructions != num_copy_instructions) {
        printf("The number of instructions %zu does not equal the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }

    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(qc, QkGate_Z, qubits, NULL);
    }
    for (int i = 0; i < 15; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {
            i,
        };
        qk_circuit_gate(copy, QkGate_X, qubits, NULL);
    }

    num_instructions = qk_circuit_num_instructions(qc);
    num_copy_instructions = qk_circuit_num_instructions(copy);
    qk_circuit_free(qc);
    qk_circuit_free(copy);
    if (num_instructions == num_copy_instructions) {
        printf("The number of instructions %zu is equal to the copied %zu", num_instructions,
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_circuit_copy_empty_like(void) {
    QkCircuit *qc = qk_circuit_new(10, 10);
    for (int i = 0; i < 10; i++) {
        qk_circuit_measure(qc, i, i);
        uint32_t qubits[1] = {i};
        qk_circuit_gate(qc, QkGate_H, qubits, NULL);
    }
    QkCircuit *copy = qk_circuit_copy_empty_like(qc, QkVarsMode_Alike, QkBlocksMode_Drop);

    size_t num_instructions = qk_circuit_num_instructions(qc);        // not 0
    size_t num_copy_instructions = qk_circuit_num_instructions(copy); // 0

    qk_circuit_free(qc);
    qk_circuit_free(copy);

    if (num_instructions == 0) {
        printf("Expected the original circuit to remain unchanged, but it is now empty\n");
        return EqualityError;
    }

    if (num_copy_instructions != 0) {
        printf("Expected no operations in the copied-empty-like circuit, but got %zu\n",
               num_copy_instructions);
        return EqualityError;
    }
    return Ok;
}

static int test_no_gate_1000_bits(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    uint32_t num_qubits = qk_circuit_num_qubits(qc);
    uint32_t num_clbits = qk_circuit_num_clbits(qc);
    size_t num_instructions = qk_circuit_num_instructions(qc);
    qk_circuit_free(qc);

    if (num_qubits != 1000) {
        printf("The number of qubits %d is not 1000", num_qubits);
        return EqualityError;
    }
    if (num_clbits != 1000) {
        printf("The number of clbits %d is not 1000", num_clbits);
        return EqualityError;
    }
    if (num_instructions != 0) {
        printf("The number of instructions %zu is not 0", num_instructions);
        return EqualityError;
    }

    return Ok;
}

static int test_gate_num_qubits(void) {
    for (uint8_t i = 0; i < 52; i++) {
        if (i == 0) {
            if (qk_gate_num_qubits(i) != 0) {

                return EqualityError;
            }
        } else if (i < 21) {
            if (qk_gate_num_qubits(i) != 1) {
                return EqualityError;
            }
        } else if (i <= 44) {
            if (qk_gate_num_qubits(i) != 2) {
                return EqualityError;
            }
        } else if (i <= 48) {
            if (qk_gate_num_qubits(i) != 3) {
                return EqualityError;
            }
        } else {
            if (qk_gate_num_qubits(i) != 4) {
                return EqualityError;
            }
        }
    }
    return Ok;
}

static bool value_in_array(uint8_t val, uint8_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (arr[i] == val)
            return true;
    }
    return false;
}

static int test_gate_num_params(void) {

    uint8_t zero_param_gates[29] = {1,  2,  3,  4,  5,  11, 12, 13, 14, 15, 16, 21, 22, 23, 24,
                                    25, 26, 27, 28, 33, 34, 35, 45, 46, 47, 48, 49, 50, 51};
    uint8_t one_param_gates[15] = {0, 6, 8, 9, 10, 18, 29, 30, 31, 32, 37, 39, 40, 41, 42};
    uint8_t two_param_gates[4] = {7, 19, 43, 44};

    for (uint8_t i = 0; i < 52; i++) {
        if (value_in_array(i, zero_param_gates, 29)) {
            if (qk_gate_num_params(i) != 0) {
                return EqualityError;
            }
        } else if (value_in_array(i, one_param_gates, 15)) {
            if (qk_gate_num_params(i) != 1) {
                return EqualityError;
            }
        } else if (value_in_array(i, two_param_gates, 4)) {
            if (qk_gate_num_params(i) != 2) {
                return EqualityError;
            }
        } else if (i == 36) {
            // CU takes 4 parameters theta, phi, lambda, gamma
            if (qk_gate_num_params(i) != 4) {
                return EqualityError;
            }
        } else {
            if (qk_gate_num_params(i) != 3) {
                return EqualityError;
            }
        }
    }
    return Ok;
}

/**
 * Test edge cases for getting the op counts.
 */
static int test_get_gate_counts(void) {
    QkCircuit *qc = qk_circuit_new(3, 3);

    // test empty circuit
    int result = Ok;
    QkOpCounts c1 = qk_circuit_count_ops(qc);
    if (c1.len != 0) {
        result = EqualityError;
        qk_opcounts_clear(&c1);
        goto circuit_cleanup;
    }
    qk_opcounts_clear(&c1);

    // add some instructions
    uint32_t qubits[2] = {1, 0};
    double params[1] = {0.12};
    qk_circuit_gate(qc, QkGate_CRX, qubits, params);
    QkOpCounts c2 = qk_circuit_count_ops(qc);

    if (c2.len != 1) {
        result = EqualityError;
        qk_opcounts_clear(&c2);
        goto circuit_cleanup;
    }
    qk_opcounts_clear(&c2);

    // check that after clearing, the object is still valid
    if (c2.len != 0 || c2.data != NULL) {
        result = EqualityError;
    }

circuit_cleanup:
    qk_circuit_free(qc);
    return result;
}

static int test_get_gate_counts_bv_no_measure(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 3) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 4) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "measure");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_barrier_and_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t i;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    uint32_t barrier_qubits[1000];
    for (i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 5) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[4].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[4].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "measure");
    if (result != 0) {
        return result;
    }
    if (op_counts.data[1].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "barrier");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 2) {
        result = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

static int test_get_gate_counts_bv_resets_barrier_and_measures(void) {
    QkCircuit *qc = qk_circuit_new(1000, 1000);
    double *params = NULL;
    uint32_t q1[1] = {999};
    uint32_t q2[2] = {0, 999};

    for (uint32_t i = 0; i < 1000; i++) {
        qk_circuit_reset(qc, i);
    }
    qk_circuit_gate(qc, QkGate_X, q1, params);
    for (uint32_t i = 0; i < 1000; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    uint32_t barrier_qubits[1000];
    for (uint32_t i = 0; i < 1000; i++) {
        barrier_qubits[i] = i;
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (uint32_t i = 0; i < 1000; i += 2) {
        q2[0] = i;
        qk_circuit_gate(qc, QkGate_CX, q2, params);
    }
    qk_circuit_barrier(qc, barrier_qubits, 1000);
    for (uint32_t i = 0; i < 999; i++) {
        q1[0] = i;
        qk_circuit_gate(qc, QkGate_H, q1, params);
    }
    for (uint32_t i = 0; i < 999; i++) {
        qk_circuit_measure(qc, i, i);
    }
    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    int result = Ok;
    if (op_counts.len != 6) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[5].name, "x");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[5].count != 1) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[0].name, "h");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[0].count != 1999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[3].name, "cx");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[3].count != 500) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[2].name, "measure");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[2].count != 999) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[4].name, "barrier");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[4].count != 2) {
        result = EqualityError;
        goto cleanup;
    }
    result = strcmp(op_counts.data[1].name, "reset");
    if (result != 0) {
        goto cleanup;
    }
    if (op_counts.data[1].count != 1000) {
        result = EqualityError;
        goto cleanup;
    }
    size_t num_instructions = qk_circuit_num_instructions(qc);
    if (num_instructions != 1000 + 2 + 999 + 500 + 1999 + 1) {
        result = EqualityError;
        goto cleanup;
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_instructions; i++) {

        qk_circuit_get_instruction(qc, i, &inst);
        if (i < 1000) {
            result = strcmp(inst.name, "reset");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 1000) {
            result = strcmp(inst.name, "x");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != 999 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i < 2001) {
            result = strcmp(inst.name, "h");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 1001 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 2001) {
            result = strcmp(inst.name, "barrier");
            if (result != 0) {
                goto loop_exit;
            }
            for (uint32_t j = 0; j < 1000; j++) {
                if (inst.qubits[j] != j) {
                    result = EqualityError;
                    goto loop_exit;
                }
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1000) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 2501) {
            result = strcmp(inst.name, "cx");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != (i - 2002) * 2) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.qubits[1] != 999 || inst.num_qubits != 2) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i == 2502) {
            result = strcmp(inst.name, "barrier");
            if (result != 0) {
                goto loop_exit;
            }
            for (uint32_t j = 0; j < 1000; j++) {
                if (inst.qubits[j] != j) {
                    result = EqualityError;
                    goto loop_exit;
                }
            }
            if (inst.num_clbits > 0 || inst.num_params > 0 || inst.num_qubits != 1000) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 3501) {
            result = strcmp(inst.name, "h");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 2503 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_clbits > 0 || inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        } else if (i <= 4500) {
            result = strcmp(inst.name, "measure");
            if (result != 0) {
                goto loop_exit;
            }
            if (inst.qubits[0] != i - 3502 || inst.num_qubits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.clbits[0] != i - 3502 || inst.num_clbits != 1) {
                result = EqualityError;
                goto loop_exit;
            }
            if (inst.num_params > 0) {
                result = EqualityError;
                goto loop_exit;
            }
        }
    loop_exit:
        qk_circuit_instruction_clear(&inst);
        if (result != 0) {
            break;
        }
    }
cleanup:
    qk_circuit_free(qc);
    qk_opcounts_clear(&op_counts);
    return result;
}

/**
 * Test appending a 2q unitary gate.
 */
static int test_unitary_gate(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[2] = {0, 1};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[16] = {c1, c0, c0, c0,  // this
                              c0, c1, c0, c0,  // is
                              c0, c0, c1, c0,  // for
                              c0, c0, c0, c1}; // formatting

    int ec = qk_circuit_unitary(qc, matrix, qubits, 2, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 2) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test appending a 1q unitary gate.
 */
static int test_unitary_gate_1q(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[1] = {0};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[4] = {c1, c0,  // this
                             c0, c1}; // is
    QkComplex64 *out = malloc(sizeof(QkComplex64) * 4);

    int ec = qk_circuit_unitary(qc, matrix, qubits, 1, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    // Check the instruction's kind
    QkOperationKind kind = qk_circuit_instruction_kind(qc, num_inst - 1);
    if (kind != QkOperationKind_Unitary) {
        result = EqualityError;
        printf("Expected instruction kind %d but got %d\n", QkOperationKind_Unitary, kind);
        goto cleanup;
    }
    memset(out, 0, sizeof(QkComplex64) * 4);
    qk_circuit_inst_unitary(qc, num_inst - 1, out);
    if (memcmp(out, matrix, sizeof(QkComplex64) * 4) != 0) {
        result = EqualityError;
        printf("Unitary matrix does not match expected\n");
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 1) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    free(out);
    return result;
}

/**
 * Test appending a 3q unitary gate.
 */
static int test_unitary_gate_3q(void) {
    QkCircuit *qc = qk_circuit_new(3, 0);
    uint32_t qubits[3] = {0, 1, 2};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[64] = {c1, c0, c0, c0, c0, c0, c0, c0,  // this
                              c0, c1, c0, c0, c0, c0, c0, c0,  // is
                              c0, c0, c1, c0, c0, c0, c0, c0,  // for
                              c0, c0, c0, c1, c0, c0, c0, c0,  // formatting
                              c0, c0, c0, c0, c1, c0, c0, c0,  // this
                              c0, c0, c0, c0, c0, c1, c0, c0,  // to
                              c0, c0, c0, c0, c0, c0, c1, c0,  // look
                              c0, c0, c0, c0, c0, c0, c0, c1}; // like a matrix

    size_t dim = 1LLU << 3;
    QkComplex64 *out = malloc(sizeof(QkComplex64) * dim * dim);
    int ec = qk_circuit_unitary(qc, matrix, qubits, 3, false);
    if (ec != QkExitCode_Success) {
        qk_circuit_free(qc);
        return ec;
    }

    int result = Ok;

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 1) {
        result = EqualityError;
        goto cleanup;
    }

    // Check the instruction's kind
    QkOperationKind kind = qk_circuit_instruction_kind(qc, num_inst - 1);
    if (kind != QkOperationKind_Unitary) {
        result = EqualityError;
        printf("Expected instruction kind %d but got %d\n", QkOperationKind_Unitary, kind);
        goto cleanup;
    }

    memset(out, 0, sizeof(QkComplex64) * dim * dim);
    qk_circuit_inst_unitary(qc, num_inst - 1, out);
    if (memcmp(out, matrix, sizeof(QkComplex64) * dim * dim) != 0) {
        result = EqualityError;
        printf("Unitary matrix does not match expected\n");
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(qc);
    if (op_counts.len != 1 || strcmp(op_counts.data[0].name, "unitary") != 0 ||
        op_counts.data[0].count != 1) {
        result = EqualityError;
        qk_opcounts_clear(&op_counts);
        goto cleanup;
    }
    qk_opcounts_clear(&op_counts);
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (strcmp(inst.name, "unitary") != 0 || inst.num_clbits != 0 || inst.num_params != 0 ||
        inst.num_qubits != 3) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(qc);
    free(out);
    return result;
}

/**
 * Test passing a non-unitary gate returns the correct exit code.
 */
static int test_not_unitary_gate(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    uint32_t qubits[2] = {0, 1};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    QkComplex64 matrix[16] = {c1, c1, c0, c0,  // this
                              c1, c1, c0, c0,  // is
                              c0, c0, c1, c0,  // for
                              c0, c0, c0, c1}; // formatting

    int exit_code = qk_circuit_unitary(qc, matrix, qubits, 2, true);

    int result = Ok;
    if (exit_code != QkExitCode_ExpectedUnitary) {
        printf("Got exit code %i but expected %i\n", exit_code, QkExitCode_ExpectedUnitary);
        result = EqualityError;
        goto cleanup;
    }

    size_t num_inst = qk_circuit_num_instructions(qc);
    if (num_inst != 0) { // we expect no gate was added
        printf("Found gate when none should be added\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test getting instruction parameters.
 */
static int test_get_instruction_params(void) {
    QkCircuit *qc = qk_circuit_new(1, 0);
    uint32_t q0[1] = {0};
    double angle[1] = {0.23};

    QkParam *theta = qk_param_new_symbol("theta");
    double r_angle = 0.2345;
    QkParam *val = qk_param_from_double(r_angle);
    const QkParam *angles[2] = {theta, val};

    QkComplex64 c0 = {0.0, 0.0};
    QkComplex64 c1 = {1.0, 0.0};
    // clang-format off
    QkComplex64 matrix[16] = {c1, c1, c0, c0,
                              c1, c1, c0, c0,
                              c0, c0, c1, c0,
                              c0, c0, c0, c1};
    // clang-format on

    int result = Ok;

    if (qk_circuit_gate(qc, QkGate_SX, q0, NULL) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_circuit_gate(qc, QkGate_RX, q0, angle) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_circuit_parameterized_gate(qc, QkGate_R, q0, angles) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_circuit_unitary(qc, matrix, q0, 1, false) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    // SX has no parameters
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (inst.num_params != 0) {
        printf("Expected 0 parameters in SX, got %u\n", inst.num_params);
        result = EqualityError;
        goto cleanup_inst;
    }
    qk_circuit_instruction_clear(&inst);

    // RX has one parameter
    qk_circuit_get_instruction(qc, 1, &inst);
    if (inst.num_params != 1) {
        printf("Expected 1 parameter in RX, got %u\n", inst.num_params);
        result = EqualityError;
        goto cleanup_inst;
    }

    double rx_angle = qk_param_as_real(inst.params[0]);
    if (isnan(rx_angle)) {
        printf("Unexpected free symbol in RX gate\n");
        result = EqualityError;
        goto cleanup_inst;
    }
    if (fabs(rx_angle - angle[0]) > 1e-10) {
        printf("Unexpected parameter value in RX gate\n");
        result = EqualityError;
        goto cleanup_inst;
    }

    // R has two parameters, one of which is free
    qk_circuit_instruction_clear(&inst);
    qk_circuit_get_instruction(qc, 2, &inst);
    if (inst.num_params != 2) {
        printf("Expected 2 parameters in R, got %u\n", inst.num_params);
        result = EqualityError;
        goto cleanup_inst;
    }

    double r_fixed = qk_param_as_real(inst.params[1]);
    if (fabs(r_fixed - r_angle) > 1e-10) {
        printf("Unexpected parameter value in R gate\n");
        result = EqualityError;
        goto cleanup_inst;
    }
    const QkParam *r_free = inst.params[0];
    if (!qk_param_equal(r_free, theta)) {
        printf("Unexpected free parameter in R gate\n");
        result = EqualityError;
        goto cleanup_inst;
    }

    // unitary gate does not store the matrix in the params, hence 0 params
    qk_circuit_instruction_clear(&inst);
    qk_circuit_get_instruction(qc, 3, &inst);
    if (inst.num_params != 0) {
        printf("Expected 0 parameters in unitary gate, got %u\n", inst.num_params);
        result = EqualityError;
        goto cleanup_inst;
    }

cleanup_inst:
    qk_circuit_instruction_clear(&inst);
cleanup:
    qk_param_free(theta);
    qk_param_free(val);
    qk_circuit_free(qc);
    return result;
}

/**
 * Test circuit instruction parameter ownership.
 *
 * Even if the original circuit is freed, the parameters should be accessible from
 * the QkCircuitInstruction object, which stores pointers to copies of the parameters.
 */
static int test_instruction_params_ownership(void) {
    QkCircuit *qc = qk_circuit_new(1, 0);
    uint32_t q0[1] = {0};
    QkParam *theta = qk_param_new_symbol("theta");
    const QkParam *angles[1] = {theta};

    // Add a single RX gate with the angle "theta"
    int result = Ok;
    if (qk_circuit_parameterized_gate(qc, QkGate_RX, q0, angles) != QkExitCode_Success) {
        result = RuntimeError;
        qk_circuit_free(qc);
        goto cleanup;
    }

    // Retrieve the circuit instruction, which contains pointers to copies of the parameters
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(qc, 0, &inst);
    if (inst.num_params != 1) {
        printf("Expected 1 parameter in RX, got %u\n", inst.num_params);
        result = EqualityError;

        qk_circuit_instruction_clear(&inst);
        qk_circuit_free(qc);
        goto cleanup;
    }

    // Free the circuit and check the parameter is still correctly accessible
    // from the instruction itself
    qk_circuit_free(qc);

    QkParam *angle = inst.params[0];
    if (!qk_param_equal(angle, theta)) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_param_free(theta);
    return result;
}

/**
 * Test adding delay instruction to a circuit.
 */
static int test_delay_instruction(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    int result = Ok;

    QkExitCode delay_s_code;

    delay_s_code = qk_circuit_delay(qc, 0, 0.001, QkDelayUnit_S);
    if (delay_s_code != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 *  A sanity check to ensure the circuit drawer handles all the supported operations.
 */
static int test_circuit_draw(void) {
    QkCircuit *circuit = qk_circuit_new(10, 2);

    uint32_t qubits[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    double params[4] = {1.41, 2.71, 3.14, 1.5};
    for (uint8_t gate = 0; gate <= QkGate_RC3X; ++gate) {
        qk_circuit_gate(circuit, gate, &qubits[gate % 8], params);
    }

    qk_circuit_barrier(circuit, qubits, 10);
    qk_circuit_delay(circuit, 3, 100.0, QkDelayUnit_NS);
    qk_circuit_measure(circuit, 0, 0);
    qk_circuit_reset(circuit, 0);

    QkComplex64 c0 = {0, 0};
    QkComplex64 c1 = {1, 0};
    QkComplex64 unitary[2 * 2] = {c0, c1, c1, c0};
    qk_circuit_unitary(circuit, unitary, (uint32_t[]){5}, 1, true);

    bool z[4] = {false, false, true, true};
    bool x[4] = {false, true, true, false};
    QkParam *angle = qk_param_from_double(1.0);
    QkPauliProductRotation ppr = {z, x, 4, angle};
    qk_circuit_pauli_product_rotation(circuit, &ppr, qubits);

    QkPauliProductMeasurement ppm = {z, x, 4, true};
    qk_circuit_pauli_product_measurement(circuit, &ppm, qubits, 0);

    QkCircuitDrawerConfig config = {false, true, 80};

    char *circ_str = qk_circuit_draw(circuit, &config);

    qk_str_free(circ_str);
    qk_circuit_free(circuit);

    return Ok;
}

static int test_parameterized_circuit(void) {
    QkCircuit *qc = qk_circuit_new(2, 0);
    QkParam *x = qk_param_new_symbol("x");
    QkParam *y = qk_param_new_symbol("somey longery namey");

    uint32_t q0[1] = {0};
    uint32_t q1[1] = {1};
    uint32_t q01[2] = {0, 1};
    const QkParam *rx_param[1] = {x};
    const QkParam *rzz_param[1] = {y};

    int result = Ok;
    if (qk_circuit_parameterized_gate(qc, QkGate_RX, q0, rx_param) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_circuit_parameterized_gate(qc, QkGate_RX, q1, rx_param) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_circuit_parameterized_gate(qc, QkGate_RZZ, q01, rzz_param) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    // check the number of parameters
    size_t num_symbols = qk_circuit_num_param_symbols(qc);
    if (num_symbols != 2) {
        result = EqualityError;
        printf("Expected 2 symbols, found %zu\n", num_symbols);
        goto cleanup;
    }

    // check number of gates
    size_t num_gates = qk_circuit_num_instructions(qc);
    if (num_gates != 3) {
        result = EqualityError;
        printf("Expected 3 instructions, found %zu\n", num_gates);
        goto cleanup;
    }

cleanup:
    qk_param_free(x);
    qk_param_free(y);
    qk_circuit_free(qc);
    return result;
}

static int test_circuit_global_phase(void) {
    int result = Ok;

    QkCircuit *qc = qk_circuit_new(2, 0);

    // Default global phase should be 0.0
    QkParam *out_phase = qk_circuit_global_phase(qc);
    double out_phase_val = qk_param_as_real(out_phase);
    qk_param_free(out_phase);
    if (out_phase_val != 0.0) {
        printf("Expected 0.0 for global phase, found %f\n", out_phase_val);
        result = EqualityError;
        goto cleanup;
    }

    // Numeric global phase parameter
    QkParam *theta = qk_param_from_double(1.23);
    qk_circuit_set_global_phase(qc, theta);
    qk_param_free(theta);

    out_phase = qk_circuit_global_phase(qc);
    out_phase_val = qk_param_as_real(out_phase);
    qk_param_free(out_phase);

    if (out_phase_val != 1.23) {
        printf("Expected 1.23 for global phase, found %f\n", out_phase_val);
        result = EqualityError;
        goto cleanup;
    }

    // Symbolic global phase parameter
    theta = qk_param_new_symbol("theta");
    if (qk_circuit_set_global_phase(qc, theta) != QkExitCode_Success) {
        result = RuntimeError;
        qk_param_free(theta);
        goto cleanup;
    }
    qk_param_free(theta);

    size_t num_symbols = qk_circuit_num_param_symbols(qc);
    if (num_symbols != 1) {
        printf("Expected 1 symbol, found %zu\n", num_symbols);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(qc);
    return result;
}

/**
 * Test circuit to dag conversion.
 */
static int test_circuit_to_dag(void) {
    QkCircuit *circuit = qk_circuit_new(2, 1);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[]){0, 1}, NULL);

    QkDag *dag = qk_circuit_to_dag(circuit);
    qk_circuit_free(circuit);

    int result = Ok;
    if (qk_dag_num_qubits(dag) != 2 || qk_dag_num_clbits(dag) != 1 ||
        qk_dag_num_op_nodes(dag) != 2) {
        printf("Circuit to DAG conversion encountered an issue\n");
        result = EqualityError;
    }

    qk_dag_free(dag);
    return result;
}

/**
 * Test a circuit with Pauli-based computation instructions.
 */
static int test_pbc_instructions(void) {
    // build a IXYZ Pauli rotation
    bool z[4] = {false, false, true, true};
    bool x[4] = {false, true, true, false};
    QkParam *angle = qk_param_from_double(1.0);
    QkPauliProductRotation rotation = {z, x, 4, angle};

    // .. and some ZY measurement
    bool zm[2] = {true, true};
    bool xm[2] = {false, true};
    QkPauliProductMeasurement measure = {zm, xm, 2, true};

    // append them to a circuit
    QkCircuit *circuit = qk_circuit_new(10, 1);
    uint32_t qubits[4] = {0, 1, 2, 3};
    qk_circuit_pauli_product_rotation(circuit, &rotation, qubits);

    uint32_t measure_qubits[2] = {1, 2};
    uint32_t measure_clbit = 0;
    qk_circuit_pauli_product_measurement(circuit, &measure, measure_qubits, measure_clbit);

    size_t num_inst = qk_circuit_num_instructions(circuit);
    int result = Ok;
    if (num_inst != 2) {
        printf("Expected 2 instructions but found %zu\n", num_inst);
        result = EqualityError;
        goto cleanup;
    }

    QkOperationKind op_kind = qk_circuit_instruction_kind(circuit, 0);
    if (op_kind != QkOperationKind_PauliProductRotation) {
        printf("Operation kind of instruction 0 is not QkOperationKind_PauliProductRotation.\n");
        result = EqualityError;
        goto cleanup;
    }
    op_kind = qk_circuit_instruction_kind(circuit, 1);
    if (op_kind != QkOperationKind_PauliProductMeasurement) {
        printf("Operation kind of instruction 1 is not QkOperationKind_PauliProductMeasurement.\n");
        result = EqualityError;
        goto cleanup;
    }

    // retrieve the instructions and verify their are correct
    QkPauliProductRotation out_rot;
    QkExitCode exit = qk_circuit_inst_pauli_product_rotation(circuit, 0, &out_rot);
    if (exit != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    if (out_rot.len != 4) {
        printf("Pauli length is not 4, but %zu\n", out_rot.len);
        result = EqualityError;
        goto cleanup_out_rot;
    }
    for (size_t i = 0; i < rotation.len; ++i) {
        if (out_rot.x[i] != x[i] || out_rot.z[i] != z[i]) {
            printf("(z, x) term at %zu does not match. Expected (%d, %d), got (%d, %d)\n", i, z[i],
                   x[i], out_rot.z[i], out_rot.x[i]);
            result = EqualityError;
            goto cleanup_out_rot;
        }
    }
    if (!qk_param_equal(out_rot.angle, angle)) {
        char *out_str = qk_param_str(out_rot.angle);
        char *expected_str = qk_param_str(angle);
        printf("Angle (%s) does not match the original angle (%s).\n", out_str, expected_str);
        qk_str_free(out_str);
        qk_str_free(expected_str);
        result = EqualityError;
        goto cleanup_out_rot;
    }

    QkPauliProductMeasurement *out_meas = malloc(sizeof(QkPauliProductMeasurement));
    exit = qk_circuit_inst_pauli_product_measurement(circuit, 1, out_meas);
    if (exit != QkExitCode_Success) {
        result = RuntimeError;
        // free the memory, but there's no content to be cleared, so we do not have to
        // call qk_pauli_product_measurement_clear
        free(out_meas);
        goto cleanup_out_rot;
    }

    if (out_meas->len != 2) {
        printf("Pauli length is not 2, but %zu\n", out_meas->len);
        result = EqualityError;
        goto cleanup_out_meas;
    }
    for (size_t i = 0; i < measure.len; ++i) {
        if (out_meas->x[i] != xm[i] || out_meas->z[i] != zm[i]) {
            printf("(z, x) term at %zu does not match. Expected (%d, %d), got (%d, %d)\n", i, zm[i],
                   xm[i], out_meas->z[i], out_meas->x[i]);
            result = EqualityError;
            goto cleanup_out_meas;
        }
    }
    if (out_meas->flip_outcome != measure.flip_outcome) {
        printf("Flip (%i) does not match the original flip (%i).\n", out_meas->flip_outcome,
               measure.flip_outcome);
        result = EqualityError;
    }

cleanup_out_meas:
    qk_pauli_product_measurement_clear(out_meas);
    free(out_meas);
cleanup_out_rot:
    qk_pauli_product_rotation_clear(&out_rot);
cleanup:
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test estimating a fidelity of a physical circuit
 */
static int test_estimate_fidelity(void) {
    int result = Ok;
    QkCircuit *qc = qk_circuit_new(1, 0);
    qk_circuit_gate(qc, QkGate_X, (uint32_t[]){0}, NULL);
    qk_circuit_gate(qc, QkGate_X, (uint32_t[]){0}, NULL);
    qk_circuit_gate(qc, QkGate_X, (uint32_t[]){0}, NULL);
    QkTarget *target = qk_target_new(1);
    QkTargetEntry *entry = qk_target_entry_new(QkGate_X);
    qk_target_entry_add_property(entry, (uint32_t[]){0}, 1, 0.5, 0.5);
    qk_target_add_instruction(target, entry);
    double fidelity = qk_circuit_estimate_fidelity(qc, target);
    double expected = pow(0.5, 3);
    if (fidelity != expected) {
        printf("Expected %f fidelity got %f instead\n", expected, fidelity);
        result = EqualityError;
    }
    qk_target_free(target);
    qk_circuit_free(qc);
    return result;
}

/**
 * Test estimating a fidelity of a non-physical circuit
 */
static int test_estimate_fidelity_non_physical(void) {
    int result = Ok;
    QkCircuit *qc = qk_circuit_new(1, 0);
    qk_circuit_gate(qc, QkGate_X, (uint32_t[]){0}, NULL);
    QkTarget *target = qk_target_new(1);
    QkTargetEntry *entry = qk_target_entry_new(QkGate_U);
    qk_target_entry_add_property(entry, (uint32_t[]){0}, 1, NAN, 0.5);
    qk_target_add_instruction(target, entry);
    double fidelity = qk_circuit_estimate_fidelity(qc, target);
    if (!isnan(fidelity)) {
        printf("Expected NAN fidelity got %f instead\n", fidelity);
        result = EqualityError;
    }
    qk_target_free(target);
    qk_circuit_free(qc);
    return result;
}

/*
 * Test iteration, name and bit queries
 */
static int test_basic_register_queries(void) {
    int result = Ok;

    QkCircuit *circuit = qk_circuit_new(0, 0);

    QkQuantumRegister *qr1 = qk_quantum_register_new(1, "QR1");
    QkQuantumRegister *qr2 = qk_quantum_register_new(2, "QR2");
    qk_circuit_add_quantum_register(circuit, qr1);
    qk_circuit_add_quantum_register(circuit, qr2);
    qk_quantum_register_free(qr1);
    qk_quantum_register_free(qr2);

    QkClassicalRegister *cr1 = qk_classical_register_new(3, "CR1");
    qk_circuit_add_classical_register(circuit, cr1);
    qk_classical_register_free(cr1);

    size_t num_qregs = qk_circuit_num_quantum_registers(circuit);
    if (num_qregs != 2) {
        printf("Expected 2 quantum registers, got %zu\n", num_qregs);
        result = EqualityError;
        goto cleanup;
    }

    const char *expected_names[2] = {"QR1", "QR2"};
    size_t expected_bits[2] = {1, 2};
    char *name = NULL;
    for (size_t qreg_idx = 0; qreg_idx < num_qregs; qreg_idx++) {
        const QkQuantumRegister *qr_retrieved = qk_circuit_get_quantum_register(circuit, qreg_idx);
        name = qk_quantum_register_name(qr_retrieved);

        if (strcmp(name, expected_names[qreg_idx]) != 0) {
            printf("Expected quantum register name %s, got '%s'\n", expected_names[qreg_idx], name);
            result = EqualityError;
            goto cleanup_name;
        }

        size_t num_bits = qk_quantum_register_num_bits(qr_retrieved);
        if (num_bits != expected_bits[qreg_idx]) {
            printf("Expected quantum register size %zu, got %zu\n", expected_bits[qreg_idx],
                   num_bits);
            result = EqualityError;
            goto cleanup_name;
        }
        qk_str_free(name);
        name = NULL;
    }

    size_t num_cregs = qk_circuit_num_classical_registers(circuit);
    if (num_cregs != 1) {
        printf("Expected 1 classical register, got %zu\n", num_cregs);
        result = EqualityError;
        goto cleanup;
    }

    const QkClassicalRegister *cr1_retrieved = qk_circuit_get_classical_register(circuit, 0);
    name = qk_classical_register_name(cr1_retrieved);
    size_t cr1_size = qk_classical_register_num_bits(cr1_retrieved);

    if (strcmp(name, "CR1") != 0) {
        printf("Expected classical register name 'CR1', got '%s'\n", name);
        result = EqualityError;
        goto cleanup_name;
    }

    if (cr1_size != 3) {
        printf("Expected classical register size 3, got %zu\n", cr1_size);
        result = EqualityError;
        goto cleanup_name;
    }

cleanup_name:
    if (name != NULL) {
        qk_str_free(name);
    }
cleanup:
    qk_circuit_free(circuit);
    return result;
}

/*
 * Test extraction of circuit bit mappings
 */
static int test_register_bits(void) {
    int result = Ok;

    QkCircuit *circuit = qk_circuit_new(1, 2);

    QkQuantumRegister *qr1 = qk_quantum_register_new(3, "QR1");
    qk_circuit_add_quantum_register(circuit, qr1);

    size_t qr1_num_bits = qk_quantum_register_num_bits(qr1);
    if (qr1_num_bits != 3) {
        printf("Expected QR1 to have 3 bits, got %zu\n", qr1_num_bits);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t *bit_indices = malloc(qr1_num_bits * sizeof(uint32_t));
    qk_quantum_register_circuit_bits(qr1, circuit, bit_indices);

    for (size_t bit = 0; bit < qr1_num_bits; bit++) {
        if (bit_indices[bit] == UINT32_MAX) {
            printf("Expected QR1 bit %zu to be in the circuit, but it's not\n", bit);
            result = EqualityError;
            goto cleanup_bit_indices;
        }
        // Positions should be 1, 2, 3, since the circuit has an anonymous qubit
        if (bit_indices[bit] != (uint32_t)bit + 1) {
            printf("Expected QR1 bit %zu to have circuit index %zu, got %" PRIu32 "\n", bit,
                   bit + 1, bit_indices[bit]);
            result = EqualityError;
            goto cleanup_bit_indices;
        }
    }

    QkQuantumRegister *qr2 = qk_quantum_register_new(2, "QR2");
    uint32_t reg_circuit_bits[2];
    qk_quantum_register_circuit_bits(qr2, circuit, reg_circuit_bits);

    for (size_t bit = 0; bit < 2; bit++) {
        if (reg_circuit_bits[bit] != UINT32_MAX) {
            printf("Expected QR2 bit %zu to NOT be in circuit, got bit index %" PRIu32 "\n", bit,
                   reg_circuit_bits[bit]);
            result = EqualityError;
            goto cleanup_qr2;
        }
    }

    QkClassicalRegister *cr1 = qk_classical_register_new(2, "CR1");
    qk_circuit_add_classical_register(circuit, cr1);
    qk_classical_register_circuit_bits(cr1, circuit, reg_circuit_bits);

    // Positions should be 2, 3, since the circuit has two anonymous clbits
    for (size_t bit = 0; bit < 2; bit++) {
        if (reg_circuit_bits[bit] != (uint32_t)bit + 2) {
            printf("Expected CR1 bit %zu to have circuit index %zu, got %" PRIu32 "\n", bit,
                   bit + 2, reg_circuit_bits[bit]);
            result = EqualityError;
            goto cleanup_cr1;
        }
    }

    QkClassicalRegister *cr2 = qk_classical_register_new(1, "cr2");
    qk_classical_register_circuit_bits(cr2, circuit, reg_circuit_bits);

    if (reg_circuit_bits[0] != UINT32_MAX) {
        printf("Expected CR2 bit 0 to NOT be in circuit, got bit index %" PRIu32 "\n",
               reg_circuit_bits[0]);
        result = EqualityError;
    }

    qk_classical_register_free(cr2);
cleanup_cr1:
    qk_classical_register_free(cr1);
cleanup_qr2:
    qk_quantum_register_free(qr2);
cleanup_bit_indices:
    free(bit_indices);
cleanup:
    qk_quantum_register_free(qr1);
    qk_circuit_free(circuit);
    return result;
}

int test_circuit(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_empty);
    num_failed += RUN_TEST(test_circuit_with_quantum_reg);
    num_failed += RUN_TEST(test_circuit_with_classical_reg);
    num_failed += RUN_TEST(test_circuit_copy);
    num_failed += RUN_TEST(test_circuit_copy_with_instructions);
    num_failed += RUN_TEST(test_circuit_copy_empty_like);
    num_failed += RUN_TEST(test_no_gate_1000_bits);
    num_failed += RUN_TEST(test_get_gate_counts);
    num_failed += RUN_TEST(test_get_gate_counts_bv_no_measure);
    num_failed += RUN_TEST(test_get_gate_counts_bv_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_barrier_and_measures);
    num_failed += RUN_TEST(test_get_gate_counts_bv_resets_barrier_and_measures);
    num_failed += RUN_TEST(test_gate_num_qubits);
    num_failed += RUN_TEST(test_gate_num_params);
    num_failed += RUN_TEST(test_delay_instruction);
    num_failed += RUN_TEST(test_unitary_gate);
    num_failed += RUN_TEST(test_not_unitary_gate);
    num_failed += RUN_TEST(test_unitary_gate_1q);
    num_failed += RUN_TEST(test_unitary_gate_3q);
    num_failed += RUN_TEST(test_circuit_draw);
    num_failed += RUN_TEST(test_get_instruction_params);
    num_failed += RUN_TEST(test_instruction_params_ownership);
    num_failed += RUN_TEST(test_parameterized_circuit);
    num_failed += RUN_TEST(test_circuit_global_phase);
    num_failed += RUN_TEST(test_circuit_to_dag);
    num_failed += RUN_TEST(test_pbc_instructions);
    num_failed += RUN_TEST(test_estimate_fidelity);
    num_failed += RUN_TEST(test_estimate_fidelity_non_physical);
    num_failed += RUN_TEST(test_basic_register_queries);
    num_failed += RUN_TEST(test_register_bits);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
