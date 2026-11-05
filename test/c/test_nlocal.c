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
#include <math.h>
#include <qiskit.h>
#include <string.h>

static void build_circuit_based_on_entanglement_strategy(uint32_t num_qubits, uint32_t reps,
                                                         QkCircuit *circuit,
                                                         QkEntanglementStrategy strategy) {

    uint32_t connections_full[10][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4},
        {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
    };
    uint32_t connections_linear[3][3] = {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}};
    uint32_t connections_reverse_linear[3][3] = {{2, 3, 4}, {1, 2, 3}, {0, 1, 2}};
    uint32_t connections_circular[5][3] = {{3, 4, 0}, {4, 0, 1}, {0, 1, 2}, {1, 2, 3}, {2, 3, 4}};

    for (uint32_t i = 0; i < reps; i++) {
        // Add entanglement layer for CCX Gate
        switch (strategy) {
        case QkEntanglementStrategy_Full:
            for (uint32_t j = 0; j < num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_Y, (uint32_t[1]){j}, NULL);
            for (size_t j = 0; j < 10; j++) {
                qk_circuit_gate(circuit, QkGate_CCX, connections_full[j], NULL);
            }
            break;
        case QkEntanglementStrategy_Linear:
            for (uint32_t j = 0; j < num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_Y, (uint32_t[1]){j}, NULL);
            for (size_t j = 0; j < 3; j++) {
                qk_circuit_gate(circuit, QkGate_CCX, connections_linear[j], NULL);
            }
            break;
        case QkEntanglementStrategy_ReverseLinear:
            for (uint32_t j = 0; j < num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_Y, (uint32_t[1]){j}, NULL);
            for (size_t j = 0; j < 3; j++) {
                qk_circuit_gate(circuit, QkGate_CCX, connections_reverse_linear[j], NULL);
            }
            break;
        case QkEntanglementStrategy_Circular:
            for (uint32_t j = 0; j < num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_Y, (uint32_t[1]){j}, NULL);
            for (size_t j = 0; j < 5; j++) {
                qk_circuit_gate(circuit, QkGate_CCX, connections_circular[j], NULL);
            }
            break;
        case QkEntanglementStrategy_Sca:
            for (uint32_t j = 0; j < num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_Y, (uint32_t[1]){j}, NULL);
            for (size_t k = 0; k < 5; k++) {
                int offset = ((int)k) - ((int)i);
                int ind = offset >= 0 ? offset : 5 + offset;
                uint32_t connection[3] = {0};
                for (size_t m = 0; m < 3; m++) {
                    connection[m] = connections_circular[ind][i % 2 == 1 ? 2 - m : m];
                }
                qk_circuit_gate(circuit, QkGate_CCX, connection, NULL);
            }
            break;
        case QkEntanglementStrategy_Pairwise:
            for (uint32_t j = 0; j < (uint32_t)num_qubits; j++)
                qk_circuit_gate(circuit, QkGate_X, (uint32_t[1]){j}, NULL);

            // Add entanglement layer for CX Gate
            qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){0, 1}, NULL);
            qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){2, 3}, NULL);
            qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){1, 2}, NULL);
            qk_circuit_gate(circuit, QkGate_CX, (uint32_t[2]){3, 4}, NULL);
            break;
        }
    }
}

static int test_nlocal_circuit_creation_with_default_settings(void) {
    uint32_t num_qubits = 5;
    size_t reps = 3;

    uint32_t expected_connections[10][2] = {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {1, 2},
                                            {1, 3}, {1, 4}, {2, 3}, {2, 4}, {3, 4}};

    QkCircuit *expected = qk_circuit_new(num_qubits, 0);

    for (size_t i = 0; i < reps; i++) {
        // Add first rotation layer for X Gate
        for (uint32_t j = 0; j < num_qubits; j++) {
            qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){j}, NULL);
        }

        // Add first entanglement layer for CX Gate
        for (uint32_t j = 0; j < 10; j++) {
            qk_circuit_gate(expected, QkGate_CX, expected_connections[j], NULL);
        }
    }
    // Add final rotation layer
    for (uint32_t j = 0; j < num_qubits; j++) {
        qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){j}, NULL);
    }

    QkGate rotation_blocks[1] = {QkGate_X};
    QkGate entanglement_blocks[1] = {QkGate_CX};

    QkCircuit *qc =
        qk_circuit_library_n_local(num_qubits, rotation_blocks, 1, entanglement_blocks, 1, NULL);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

static int test_parameter_prefix(void) {
    int result = Ok;
    uint32_t num_qubits = 2;

    QkGate rotation_blocks[1] = {QkGate_H};
    QkGate entanglement_blocks[1] = {QkGate_CRX};

    QkNLocalSettings settings = qk_circuit_library_n_local_settings_default();
    settings.reps = 2;
    settings.entanglement_strategy = QkEntanglementStrategy_Linear;
    settings.skip_final_rotation_layer = true;

    settings.parameter_prefix = "myprefix";
    size_t prefix_len = strlen(settings.parameter_prefix);

    QkCircuit *qc = qk_circuit_library_n_local(num_qubits, rotation_blocks, 1, entanglement_blocks,
                                               1, &settings);

    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(qc); i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        if (strstr(inst.name, "crx") != NULL) {
            char *parameter_str = qk_param_str(inst.params[0]);
            if (strncmp(parameter_str, settings.parameter_prefix, prefix_len) != 0) {
                result = EqualityError;
                qk_str_free(parameter_str);
                qk_circuit_instruction_clear(&inst);
                break;
            }
            qk_str_free(parameter_str);
        }
        qk_circuit_instruction_clear(&inst);
    }

    qk_circuit_free(qc);

    return result;
}

static int test_insert_barrier(void) {
    uint32_t num_qubits = 5;

    uint32_t expected_connections[10][3] = {
        {0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4},
        {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4},
    };

    QkCircuit *expected = qk_circuit_new(num_qubits, 0);

    // Add first rotation layer for X Gate
    for (uint32_t j = 0; j < num_qubits; j++) {
        qk_circuit_gate(expected, QkGate_X, (uint32_t[1]){j}, NULL);
    }

    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, num_qubits);

    // Add first entanglement layer for CCX Gate
    for (uint32_t j = 0; j < 10; j++) {
        qk_circuit_gate(expected, QkGate_CCX, expected_connections[j], NULL);
    }

    qk_circuit_barrier(expected, (uint32_t[5]){0, 1, 2, 3, 4}, num_qubits);

    QkGate rotation_blocks[1] = {QkGate_X};
    QkGate entanglement_blocks[1] = {QkGate_CCX};

    QkNLocalSettings settings = qk_circuit_library_n_local_settings_default();
    settings.reps = 1;
    settings.insert_barriers = true;
    settings.skip_final_rotation_layer = true;

    QkCircuit *qc = qk_circuit_library_n_local(num_qubits, rotation_blocks, 1, entanglement_blocks,
                                               1, &settings);

    int result = (compare_circuits(qc, expected)) ? Ok : EqualityError;

    qk_circuit_free(qc);
    qk_circuit_free(expected);

    return result;
}

/**
 * Test setting the entanglement of the layers by entanglement strategy
 */
static int test_entanglement_by_strategy(void) {
    int num_qubits = 5;
    int result = Ok;
    uint32_t reps = 3;

    char *strategies[] = {"Full", "Linear", "ReverseLinear", "Circular", "SCA"};

    // Full entanglement strategy
    QkCircuit *expected_full = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, reps, expected_full,
                                                 QkEntanglementStrategy_Full);

    // Linear entanglement strategy
    QkCircuit *expected_linear = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, reps, expected_linear,
                                                 QkEntanglementStrategy_Linear);

    // ReverseLinear entanglement strategy
    QkCircuit *expected_revlinear = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, reps, expected_revlinear,
                                                 QkEntanglementStrategy_ReverseLinear);

    // Circular entanglement strategy
    QkCircuit *expected_circular = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, reps, expected_circular,
                                                 QkEntanglementStrategy_Circular);

    // SCA entanglement strategy
    QkCircuit *expected_sca = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, reps, expected_sca,
                                                 QkEntanglementStrategy_Sca);

    // Pairwise entanglement strategy
    QkCircuit *expected_pairwise = qk_circuit_new(num_qubits, 0);
    build_circuit_based_on_entanglement_strategy(num_qubits, 1, expected_pairwise,
                                                 QkEntanglementStrategy_Pairwise);

    QkGate p_rotation_blocks[1] = {QkGate_X};
    QkGate p_entanglement_blocks[1] = {QkGate_CX};

    QkGate rotation_blocks[1] = {QkGate_Y};
    QkGate entanglement_blocks[1] = {QkGate_CCX};

    for (QkEntanglementStrategy strategy = 0; strategy < 5; strategy++) {
        bool is_pairwise = strategy == QkEntanglementStrategy_Pairwise;
        QkNLocalSettings settings = qk_circuit_library_n_local_settings_default();
        settings.entanglement_strategy = strategy;
        settings.reps = is_pairwise ? 1 : reps;
        settings.skip_final_rotation_layer = true;

        QkCircuit *qc = qk_circuit_library_n_local(
            num_qubits, is_pairwise ? p_rotation_blocks : rotation_blocks, 1,
            is_pairwise ? p_entanglement_blocks : entanglement_blocks, 1, &settings);

        QkCircuit *expected = NULL;
        switch (strategy) {
        case QkEntanglementStrategy_Full:
            expected = expected_full;
            break;
        case QkEntanglementStrategy_Linear:
            expected = expected_linear;
            break;
        case QkEntanglementStrategy_ReverseLinear:
            expected = expected_revlinear;
            break;
        case QkEntanglementStrategy_Circular:
            expected = expected_circular;
            break;
        case QkEntanglementStrategy_Sca:
            expected = expected_sca;
            break;
        case QkEntanglementStrategy_Pairwise:
            expected = expected_pairwise;
        }
        if (!(compare_circuits(qc, expected))) {
            result = EqualityError;
            fprintf(stderr, "Strategy %s failed", strategies[(size_t)strategy]);
            qk_circuit_free(qc);
            goto cleanup;
        }
        qk_circuit_free(qc);
    }
    goto cleanup;
cleanup:
    qk_circuit_free(expected_full);
    qk_circuit_free(expected_linear);
    qk_circuit_free(expected_revlinear);
    qk_circuit_free(expected_circular);
    qk_circuit_free(expected_pairwise);
    qk_circuit_free(expected_sca);
    return result;
}

static int test_pairwise_with_3_qubits(void) {
    int num_qubits = 3;
    QkGate rotation_blocks[3] = {QkGate_X, QkGate_Z, QkGate_Y};
    QkGate entanglement_blocks[1] = {QkGate_CCX};
    QkNLocalSettings settings = qk_circuit_library_n_local_settings_default();
    settings.entanglement_strategy = QkEntanglementStrategy_Pairwise;
    QkCircuit *qc = qk_circuit_library_n_local(num_qubits, rotation_blocks, 3, entanglement_blocks,
                                               1, &settings);
    if (qc != NULL) {
        fprintf(stderr,
                "Inconsistent result when testing pairwise entanglement with 3 qubits: A null "
                "response was expected, but a circuit was obtained.");
        qk_circuit_free(qc);
        return EqualityError;
    }
    return Ok;
}

/*
 * Test CCX gate on 2 qubits on full entanglement
 */
static int test_full_entanglement_with_3_qubit_gate_for_2_qubits(void) {
    int num_qubits = 2;
    QkGate rotation_blocks[1] = {QkGate_X};
    QkGate entanglement_blocks[1] = {QkGate_CCX};
    QkCircuit *qc =
        qk_circuit_library_n_local(num_qubits, rotation_blocks, 1, entanglement_blocks, 1, NULL);

    if (qc != NULL) {
        fprintf(stderr, "Inconsistent result when testing 2 qubits with a 3 qubit gate: A null "
                        "response was expected, but a circuit was returned.");
        qk_circuit_free(qc);
        return EqualityError;
    }

    return Ok;
}

int test_nlocal(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_nlocal_circuit_creation_with_default_settings);
    num_failed += RUN_TEST(test_parameter_prefix);
    num_failed += RUN_TEST(test_insert_barrier);
    num_failed += RUN_TEST(test_entanglement_by_strategy);
    num_failed += RUN_TEST(test_pairwise_with_3_qubits);
    num_failed += RUN_TEST(test_full_entanglement_with_3_qubit_gate_for_2_qubits);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (n_local): %i\n", num_failed);

    return num_failed;
}
