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
#include <math.h>
#include <qiskit.h>
#include <string.h>

/**
 * Build an Ising-like Hamiltonian but with an additional ZYX connections on a chain.
 */
QkObs *build_tri_ising(uint32_t num_qubits) {
    QkComplex64 J = {1., 0.};
    QkComplex64 h = {0.5, 0.};

    QkObs *obs = qk_obs_zero(num_qubits);
    QkBitTerm zyx[3] = {QkBitTerm_Z, QkBitTerm_Y, QkBitTerm_X};
    QkBitTerm zz[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkBitTerm x[1] = {QkBitTerm_X};

    for (uint32_t i = 0; i < num_qubits; i++) {
        if (i < num_qubits - 2) { // ZYX term
            uint32_t qubits[3] = {i, i + 1, i + 2};
            QkObsTerm zyx_term = {J, 3, zyx, qubits, num_qubits};
            if (qk_obs_add_term(obs, &zyx_term) > 0) {
                return NULL; // something went wrong
            }
        }
        if (i < num_qubits - 1) { // ZZ term
            uint32_t qubits[2] = {i, i + 1};
            QkObsTerm zz_term = {J, 2, zz, qubits, num_qubits};
            if (qk_obs_add_term(obs, &zz_term) > 0) {
                return NULL; // something went wrong
            }
        }
        uint32_t qubit[1] = {i};
        QkObsTerm x_term = {h, 1, x, qubit, num_qubits};
        if (qk_obs_add_term(obs, &x_term) > 0) {
            return NULL; // something went wrong
        }
    }
    return obs;
}

static int test_suzuki_trotter_4_order_no_reorder(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    double time = 0.1;
    int reps = 1;
    double p_4 = 1 / (4 - pow(4, 1.0 / 3.0));

    for (size_t i = 0; i < 2; i++) {
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
        qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * p_4 * time});
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
    }

    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){(1 - 4 * p_4) * time});
    qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * (1 - 4 * p_4) * time});
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){(1 - 4 * p_4) * time});

    for (size_t i = 0; i < 2; i++) {
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
        qk_circuit_gate(expected, QkGate_RY, (uint32_t[1]){0}, (double[1]){2 * p_4 * time});
        qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){p_4 * time});
    }

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[1] = {QkBitTerm_Y};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term2);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 4, reps, time, true, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

static int test_suzuki_trotter_2_order_reorder(void) {
    QkCircuit *expected = qk_circuit_new(4, 0);

    qk_circuit_gate(expected, QkGate_RZZ, (uint32_t[2]){0, 1}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RXX, (uint32_t[2]){2, 3}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RYY, (uint32_t[2]){1, 2}, (double[1]){0.2});
    qk_circuit_gate(expected, QkGate_RXX, (uint32_t[2]){2, 3}, (double[1]){0.1});
    qk_circuit_gate(expected, QkGate_RZZ, (uint32_t[2]){0, 1}, (double[1]){0.1});

    int num_qubits = 4;
    QkObs *obs = qk_obs_zero(num_qubits);

    QkBitTerm op1_bits[2] = {QkBitTerm_X, QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 2, op1_bits, (uint32_t[2]){2, 3}, num_qubits};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[2] = {QkBitTerm_Y, QkBitTerm_Y};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 2, op2_bits, (uint32_t[2]){1, 2}, num_qubits};
    qk_obs_add_term(obs, &term2);

    QkBitTerm op3_bits[2] = {QkBitTerm_Z, QkBitTerm_Z};
    QkObsTerm term3 = {(QkComplex64){1.0, 0.0}, 2, op3_bits, (uint32_t[2]){0, 1}, num_qubits};
    qk_obs_add_term(obs, &term3);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 2, 1, 0.1, false, false);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

static int test_suzuki_trotter_with_barriers(void) {
    QkCircuit *expected = qk_circuit_new(1, 0);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){0.2});
    qk_circuit_barrier(expected, (uint32_t[1]){0}, 1);
    qk_circuit_gate(expected, QkGate_RX, (uint32_t[1]){0}, (double[1]){0.2});

    QkObs *obs = qk_obs_zero(1);

    QkBitTerm op1_bits[1] = {QkBitTerm_X};
    QkObsTerm term1 = {(QkComplex64){1.0, 0.0}, 1, op1_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term1);

    QkBitTerm op2_bits[1] = {QkBitTerm_X};
    QkObsTerm term2 = {(QkComplex64){1.0, 0.0}, 1, op2_bits, (uint32_t[1]){0}, 1};
    qk_obs_add_term(obs, &term2);

    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 1, 1, 0.1, true, true);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_obs_free(obs);
    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

/**
 * Test an Ising-like Hamiltonian with some additional terms.
 */
static int test_tri_ising_hamiltonian(void) {
    uint32_t num_qubits = 50;
    QkObs *obs = build_tri_ising(num_qubits);
    if (obs == NULL) {
        return RuntimeError;
    }

    int result = Ok;
    QkCircuit *qc = qk_circuit_library_suzuki_trotter(obs, 1, 1, 0.1, true, false);
    if (qc == NULL) {
        result = RuntimeError;
        goto cleanup_obs;
    }

    QkOpCounts counts = qk_circuit_count_ops(qc);

    // we expect:
    //  - `num_qubits` RX terms for the transversal field
    //  - `num_qubits - 1` RZZ interactions
    //  - `num_qubits - 2` RZYX interactions, which use: 4 CX, 1 RZ, 1 SX, 1 SXdg, 2 H
    const size_t num_types = 7;
    if (counts.len != num_types) {
        result = EqualityError;
        fprintf(stderr, "Expected %zu operations, but found %zu\n", num_types, counts.len);
        // goto cleanup;
    }

    const size_t num_x_terms = num_qubits;
    const size_t num_zz_terms = num_qubits - 1;
    const size_t num_zzz_terms = num_qubits - 2;

    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "rz") == 0 || strcmp(count.name, "sx") == 0 ||
            strcmp(count.name, "sxdg") == 0) {
            if (count.count != num_zzz_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu %s gates, but found %zu\n", num_zzz_terms, count.name,
                        count.count);
                goto cleanup;
            }
        } else if (strcmp(count.name, "h") == 0) {
            if (count.count != 2 * num_zzz_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu h gates, but found %zu\n", 2 * num_zzz_terms,
                        count.count);
                goto cleanup;
            }
        } else if (strcmp(count.name, "cx") == 0) {
            if (count.count != 4 * num_zzz_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu cx gates, but found %zu\n", 4 * num_zzz_terms,
                        count.count);
                goto cleanup;
            }
        } else if (strcmp(count.name, "rzz") == 0) {
            if (count.count != num_zz_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu rzz gates, but found %zu\n", num_zz_terms,
                        count.count);
                goto cleanup;
            }
        } else if (strcmp(count.name, "rzz") == 0) {
            if (count.count != num_zz_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu rzz gates, but found %zu\n", num_zz_terms,
                        count.count);
                goto cleanup;
            }
        } else if (strcmp(count.name, "rx") == 0) {
            if (count.count != num_x_terms) {
                result = EqualityError;
                fprintf(stderr, "Expected %zu rx gates, but found %zu\n", num_x_terms, count.count);
                goto cleanup;
            }
        } else {
            // Invalid name in the dictionary.
            result = EqualityError;
            fprintf(stderr, "Invalid operation: %s\n", count.name);
            goto cleanup;
        }
    }

cleanup:
    qk_opcounts_clear(&counts);
    qk_circuit_free(qc);
cleanup_obs:
    qk_obs_free(obs);

    return result;
}

int test_suzuki_trotter(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_suzuki_trotter_4_order_no_reorder);
    num_failed += RUN_TEST(test_suzuki_trotter_2_order_reorder);
    num_failed += RUN_TEST(test_suzuki_trotter_with_barriers);
    num_failed += RUN_TEST(test_tri_ising_hamiltonian);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (Suzuki-Trotter): %i\n", num_failed);

    return num_failed;
}
