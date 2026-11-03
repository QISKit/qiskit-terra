// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
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

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static QkObs *make_1q_pauli(uint32_t num_qubits, uint32_t q, QkBitTerm bt, QkComplex64 coeff) {
    QkObs *obs = qk_obs_zero(num_qubits);
    if (obs == NULL) {
        return NULL;
    }

    // IMPORTANT: these must NOT be const, because QkObsTerm stores non-const pointers.
    uint32_t idx[1] = {q};
    QkBitTerm bits[1] = {bt};

    QkObsTerm term = (QkObsTerm){
        .coeff = coeff,
        .len = 1,
        .bit_terms = bits,
        .indices = idx,
        .num_qubits = num_qubits,
    };

    if (qk_obs_add_term(obs, &term) != 0) {
        qk_obs_free(obs);
        return NULL;
    }
    return obs;
}

static QkObs *make_2q_pauli(uint32_t num_qubits, uint32_t q0, QkBitTerm bt0, uint32_t q1,
                            QkBitTerm bt1, QkComplex64 coeff) {
    QkObs *obs = qk_obs_zero(num_qubits);
    if (obs == NULL) {
        return NULL;
    }

    // IMPORTANT: these must NOT be const, because QkObsTerm stores non-const pointers.
    uint32_t idx[2] = {q0, q1};
    QkBitTerm bits[2] = {bt0, bt1};

    QkObsTerm term = (QkObsTerm){
        .coeff = coeff,
        .len = 2,
        .bit_terms = bits,
        .indices = idx,
        .num_qubits = num_qubits,
    };

    if (qk_obs_add_term(obs, &term) != 0) {
        qk_obs_free(obs);
        return NULL;
    }
    return obs;
}

// -----------------------------------------------------------------------------
// Tests: qk_circuit_library_qaoa_ansatz (parameterized)
// -----------------------------------------------------------------------------

static int test_qaoa_ansatz_null_cost_returns_null(void) {
    int result = Ok;

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(
        /*cost=*/NULL,
        /*reps=*/1,
        /*insert_barriers=*/false,
        /*mixer=*/NULL);

    if (circ != NULL) {
        printf("Expected NULL circuit for NULL cost, got non-NULL\n");
        qk_circuit_free(circ);
        result = EqualityError;
    }

    return result;
}

static int test_qaoa_ansatz_reps_zero_returns_null(void) {
    int result = Ok;

    QkObs *cost = make_1q_pauli(1, 0, QkBitTerm_Z, (QkComplex64){.re = 1.0, .im = 0.0});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 0, false, NULL);
    if (circ != NULL) {
        printf("Expected NULL circuit for reps=0, got non-NULL\n");
        qk_circuit_free(circ);
        result = EqualityError;
    }

    qk_obs_free(cost);
    return result;
}

static int test_qaoa_ansatz_simple_cost_returns_circuit(void) {
    int result = Ok;

    QkObs *cost = make_1q_pauli(1, 0, QkBitTerm_Z, (QkComplex64){.re = 1.0, .im = 0.0});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 1, false, NULL);
    if (circ == NULL) {
        printf("Expected non-NULL circuit for simple cost, got NULL\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_qubits(circ) != 1) {
        printf("Unexpected num_qubits: %u (expected 1)\n", qk_circuit_num_qubits(circ));
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_instructions(circ) == 0) {
        printf("Circuit has zero instructions\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (circ) {
        qk_circuit_free(circ);
    }
    qk_obs_free(cost);
    return result;
}

static int test_qaoa_ansatz_cost_with_projector_bitterm_returns_null(void) {
    int result = Ok;

    QkObs *cost = make_1q_pauli(1, 0, QkBitTerm_Plus, (QkComplex64){.re = 1.0, .im = 0.0});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 1, false, NULL);
    if (circ != NULL) {
        printf("Expected NULL circuit for projector BitTerm, got non-NULL\n");
        qk_circuit_free(circ);
        result = EqualityError;
    }

    qk_obs_free(cost);
    return result;
}

static int test_qaoa_ansatz_cost_with_imag_coeff_returns_null(void) {
    int result = Ok;

    QkObs *cost = make_1q_pauli(1, 0, QkBitTerm_Z, (QkComplex64){.re = 1.0, .im = 0.25});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 1, false, NULL);
    if (circ != NULL) {
        printf("Expected NULL circuit for imaginary coefficient, got non-NULL\n");
        qk_circuit_free(circ);
        result = EqualityError;
    }

    qk_obs_free(cost);
    return result;
}

static int test_qaoa_ansatz_mixer_num_qubits_mismatch_returns_null(void) {
    int result = Ok;

    QkObs *cost =
        make_2q_pauli(2, 0, QkBitTerm_Z, 1, QkBitTerm_Z, (QkComplex64){.re = 1.0, .im = 0.0});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    // Mixer: 1 qubit (X0) -> mismatch
    QkObs *mixer = make_1q_pauli(1, 0, QkBitTerm_X, (QkComplex64){.re = 1.0, .im = 0.0});
    if (mixer == NULL) {
        printf("Failed to build mixer observable\n");
        qk_obs_free(cost);
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 1, false, mixer);
    if (circ != NULL) {
        printf("Expected NULL circuit for mixer num_qubits mismatch, got non-NULL\n");
        qk_circuit_free(circ);
        result = EqualityError;
    }

    qk_obs_free(cost);
    qk_obs_free(mixer);
    return result;
}

static int test_qaoa_ansatz_insert_barriers_true_returns_circuit(void) {
    int result = Ok;

    QkObs *cost = make_1q_pauli(1, 0, QkBitTerm_Z, (QkComplex64){.re = 1.0, .im = 0.0});
    if (cost == NULL) {
        printf("Failed to build cost observable\n");
        return EqualityError;
    }

    QkCircuit *circ = qk_circuit_library_qaoa_ansatz(cost, 2, /*insert_barriers=*/true, NULL);
    if (circ == NULL) {
        printf("Expected non-NULL circuit when insert_barriers=true, got NULL\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_instructions(circ) == 0) {
        printf("Circuit has zero instructions\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (circ) {
        qk_circuit_free(circ);
    }
    qk_obs_free(cost);
    return result;
}

// -----------------------------------------------------------------------------
// Entry point
// -----------------------------------------------------------------------------

int test_qaoa_ansatz(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_qaoa_ansatz_null_cost_returns_null);
    num_failed += RUN_TEST(test_qaoa_ansatz_reps_zero_returns_null);
    num_failed += RUN_TEST(test_qaoa_ansatz_simple_cost_returns_circuit);
    num_failed += RUN_TEST(test_qaoa_ansatz_cost_with_projector_bitterm_returns_null);
    num_failed += RUN_TEST(test_qaoa_ansatz_cost_with_imag_coeff_returns_null);
    num_failed += RUN_TEST(test_qaoa_ansatz_mixer_num_qubits_mismatch_returns_null);
    num_failed += RUN_TEST(test_qaoa_ansatz_insert_barriers_true_returns_circuit);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (QAOA ansatz): %i\n", num_failed);

    return num_failed;
}
