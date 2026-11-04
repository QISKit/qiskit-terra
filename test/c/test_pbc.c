// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
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
#include <string.h>

#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <math.h> // for M_PI and friends

/**
 * Test gate counts after Litinski transformation.
 */
static int test_counts_litinski(void) {
    QkCircuit *circuit = qk_circuit_new(4, 0);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){0, 1}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){0, 2}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_Tdg, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_S, (uint32_t[1]){2}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){2}, NULL);

    qk_transpiler_pass_standalone_litinski_transformation(circuit, false);
    int result = Ok;
    if (qk_circuit_num_instructions(circuit) != 4) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(circuit);
    for (size_t i = 0; i < op_counts.len; i++) {
        if (strcmp(op_counts.data[i].name, "pauli_product_rotation") != 0) {
            if (op_counts.data[i].count != 4) {
                result = EqualityError;
                break;
            }
        }
    }
    qk_opcounts_clear(&op_counts);
cleanup:
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test gate counts after converting to Pauli rotations.
 */
static int test_counts_convert_to_pauli_rotations(void) {
    QkCircuit *circuit = qk_circuit_new(4, 2);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[1]){0}, NULL);     // 2 PPR gates
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){0, 1}, NULL); // 3 PPR gates
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){1}, NULL);     // 1 PPR gate
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){0, 2}, NULL);
    qk_circuit_measure(circuit, 0, 0);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_Tdg, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_S, (uint32_t[1]){2}, NULL); // 1 PPR gate
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){2}, NULL);
    qk_circuit_measure(circuit, 0, 1);

    qk_transpiler_pass_standalone_convert_to_pauli_rotations(circuit);
    int result = Ok;
    if (qk_circuit_num_instructions(circuit) != 15) {
        result = EqualityError;
        goto cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(circuit);
    const size_t expected_ppr = 13;
    const size_t expected_ppm = 2;
    for (size_t i = 0; i < op_counts.len; i++) {
        if (strcmp(op_counts.data[i].name, "pauli_product_rotation") == 0) {
            if (op_counts.data[i].count != expected_ppr) {
                fprintf(stderr, "Expected %zu PPR but found %zu\n", expected_ppr,
                        op_counts.data[i].count);
                result = EqualityError;
                break;
            }
        } else if (strcmp(op_counts.data[i].name, "pauli_product_measurement") == 0) {
            if (op_counts.data[i].count != 2) {
                fprintf(stderr, "Expected %zu PPM but found %zu\n", expected_ppm,
                        op_counts.data[i].count);
                result = EqualityError;
                break;
            }
        } else {
            // unexpected gate name!
            fprintf(stderr, "Unexpected gate: %s\n", op_counts.data[i].name);
            result = EqualityError;
            break;
        }
    }
    qk_opcounts_clear(&op_counts);

cleanup:
    qk_circuit_free(circuit);
    return result;
}

/**
 * An enum representing non-identity Paulis.
 */
enum Pauli {
    PX,
    PY,
    PZ,
};

/**
 * Helper function checking if an ZX representation matches an expected Pauli product.
 */
int check_paulis(enum Pauli *expected_paulis, bool *x, bool *z, size_t len) {
    int result = Ok;

    bool expected_z;
    bool expected_x;
    for (size_t i = 0; i < len; i++) {
        switch (expected_paulis[i]) {
        case PX:
            expected_x = true;
            expected_z = false;
            break;
        case PY:
            expected_x = true;
            expected_z = true;
            break;
        case PZ:
            expected_x = false;
            expected_z = true;
        }
        if (x[i] != expected_x || z[i] != expected_z) {
            fprintf(stderr, "Expected (%i, %i) but got (%i, %i)\n", expected_x, expected_z, x[i],
                    z[i]);
            result = EqualityError;
            break;
        }
    }

    return result;
}

/**
 * A helper to check a PPR matches the expectation.
 */
int check_pauli_rotation(QkCircuit *circuit, size_t index, enum Pauli *paulis, uint32_t *qubits,
                         const size_t len, double expected_angle) {
    QkPauliProductRotation ppr;
    if (qk_circuit_inst_pauli_product_rotation(circuit, index, &ppr) != QkExitCode_Success) {
        return RuntimeError;
    }

    int result = Ok;

    // (1) check the angle matches
    double angle = qk_param_as_real(ppr.angle);
    if (fabs(angle - expected_angle) > 1e-12) {
        fprintf(stderr, "Expected angle %f but got %f\n", expected_angle, angle);
        result = EqualityError;
        goto cleanup;
    }

    // (2) check the Paulis match
    if (ppr.len != len) {
        fprintf(stderr, "Expected %zu Paulis but got %zu\n", len, ppr.len);
        result = EqualityError;
        goto cleanup;
    }
    result = check_paulis(paulis, ppr.x, ppr.z, len);
    if (result != Ok)
        goto cleanup;

    // (3) check the qubits match
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(circuit, index, &inst);
    for (size_t i = 0; i < len; i++) {
        if (inst.qubits[i] != qubits[i]) {
            result = EqualityError;
            goto cleanup_inst;
        }
    }

cleanup_inst:
    qk_circuit_instruction_clear(&inst);
cleanup:
    qk_pauli_product_rotation_clear(&ppr);
    return result;
}

/**
 * A helper to check a PPM matches the expectation.
 */
int check_pauli_measurement(QkCircuit *circuit, size_t index, enum Pauli *paulis, uint32_t *qubits,
                            const size_t len, uint32_t clbit, bool flip_outcome) {
    QkPauliProductMeasurement ppm;
    if (qk_circuit_inst_pauli_product_measurement(circuit, index, &ppm) != QkExitCode_Success) {
        return RuntimeError;
    }

    int result = Ok;
    // (1) check the sign matches
    if (flip_outcome != ppm.flip_outcome) {
        result = EqualityError;
        goto cleanup;
    }

    // (2) check the Paulis match
    if (ppm.len != len) {
        fprintf(stderr, "Expected %zu Paulis but got %zu\n", len, ppm.len);
        result = EqualityError;
        goto cleanup;
    }
    result = check_paulis(paulis, ppm.x, ppm.z, len);
    if (result != Ok)
        goto cleanup;

    // (3) check the qubits and the clbit match
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(circuit, index, &inst);
    if (clbit != inst.clbits[0]) {
        result = EqualityError;
        goto cleanup_inst;
    }
    for (size_t i = 0; i < len; i++) {
        if (inst.qubits[i] != qubits[i]) {
            result = EqualityError;
            goto cleanup_inst;
        }
    }

cleanup_inst:
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_pauli_product_measurement_clear(&ppm);
    return result;
}

/**
 * Test concrete circuit after running the Litinski transformation.
 */
static int test_concrete_litinski(void) {
    QkCircuit *circuit = qk_circuit_new(4, 4);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){2, 1}, NULL);
    qk_circuit_gate(circuit, QkGate_SXdg, (uint32_t[1]){3}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){1, 0}, NULL);
    qk_circuit_gate(circuit, QkGate_SX, (uint32_t[1]){2}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){3}, NULL);
    qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){3, 0}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_S, (uint32_t[1]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){2}, NULL);
    qk_circuit_gate(circuit, QkGate_S, (uint32_t[1]){3}, NULL);
    qk_circuit_gate(circuit, QkGate_SXdg, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_SX, (uint32_t[1]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_SX, (uint32_t[1]){2}, NULL);
    qk_circuit_gate(circuit, QkGate_SX, (uint32_t[1]){3}, NULL);

    for (uint32_t i = 0; i < 4; i++) {
        qk_circuit_measure(circuit, i, i);
    }

    qk_transpiler_pass_standalone_litinski_transformation(circuit, false);

    int result = Ok;
    result = check_pauli_rotation(circuit, 0, (enum Pauli[1]){PZ}, (uint32_t[1]){0}, 1, M_PI_4);
    if (result != Ok)
        goto cleanup;
    result =
        check_pauli_rotation(circuit, 1, (enum Pauli[2]){PX, PY}, (uint32_t[2]){1, 2}, 2, M_PI_4);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 2, (enum Pauli[1]){PY}, (uint32_t[1]){3}, 1, -M_PI_4);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 3, (enum Pauli[4]){PZ, PZ, PZ, PY},
                                  (uint32_t[4]){0, 1, 2, 3}, 4, -M_PI_4);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_measurement(circuit, 4, (enum Pauli[4]){PY, PZ, PZ, PY},
                                     (uint32_t[4]){0, 1, 2, 3}, 4, 0, false);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_measurement(circuit, 5, (enum Pauli[2]){PX, PX}, (uint32_t[2]){0, 1}, 2, 1,
                                     false);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_measurement(circuit, 6, (enum Pauli[1]){PZ}, (uint32_t[1]){2}, 1, 2, true);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_measurement(circuit, 7, (enum Pauli[2]){PX, PX}, (uint32_t[2]){0, 3}, 2, 3,
                                     false);

cleanup:
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test concrete circuit after running conversion to Pauli rotations.
 */
static int test_concrete_convert_to_pauli_rotations(void) {
    QkCircuit *circuit = qk_circuit_new(4, 1);
    qk_circuit_gate(circuit, QkGate_T, (uint32_t[1]){0}, NULL);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[2]){1}, NULL);
    qk_circuit_gate(circuit, QkGate_RXX, (uint32_t[2]){2, 3}, (double[1]){1.0});
    qk_circuit_gate(circuit, QkGate_SXdg, (uint32_t[1]){3}, NULL);
    qk_circuit_measure(circuit, 0, 0);

    qk_transpiler_pass_standalone_convert_to_pauli_rotations(circuit);

    int result = Ok;
    result = check_pauli_rotation(circuit, 0, (enum Pauli[1]){PZ}, (uint32_t[1]){0}, 1, M_PI_4);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 1, (enum Pauli[1]){PY}, (uint32_t[1]){1}, 1, M_PI_2);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 2, (enum Pauli[1]){PX}, (uint32_t[1]){1}, 1, M_PI);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 3, (enum Pauli[2]){PX, PX}, (uint32_t[2]){2, 3}, 2, 1.0);
    if (result != Ok)
        goto cleanup;
    result = check_pauli_rotation(circuit, 4, (enum Pauli[1]){PX}, (uint32_t[1]){3}, 1, -M_PI_2);
    if (result != Ok)
        goto cleanup;

    result =
        check_pauli_measurement(circuit, 5, (enum Pauli[1]){PZ}, (uint32_t[1]){0}, 1, 0, false);

cleanup:
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test Litinski not doing anything.
 */
static int test_litinski_noop(void) {
    QkCircuit *circuit = qk_circuit_new(1, 0);
    qk_circuit_gate(circuit, QkGate_H, (uint32_t[1]){0}, NULL);

    qk_transpiler_pass_standalone_litinski_transformation(circuit, true);

    int result = Ok;
    if (qk_circuit_num_instructions(circuit) != 1) {
        fprintf(stderr, "Expected 1 instructions, but found %zu",
                qk_circuit_num_instructions(circuit));
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction inst;
    qk_circuit_get_instruction(circuit, 0, &inst);
    if (strcmp(inst.name, "h") != 0) {
        fprintf(stderr, "Instruction at index 0 should be 'h' but is '%s'", inst.name);
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);

cleanup:
    qk_circuit_free(circuit);
    return result;
}

int test_pbc(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_counts_litinski);
    num_failed += RUN_TEST(test_counts_convert_to_pauli_rotations);
    num_failed += RUN_TEST(test_concrete_litinski);
    num_failed += RUN_TEST(test_concrete_convert_to_pauli_rotations);
    num_failed += RUN_TEST(test_litinski_noop);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (PBC transformations): %i\n", num_failed);

    return num_failed;
}
