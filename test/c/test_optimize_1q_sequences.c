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
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef int (*optimize_h_gates_fn)(QkTarget *, char **, uint32_t *, size_t);
typedef int (*optimize_identity_target_fn)(QkTarget *);

/// @brief Generates a typical target where u1 is cheaper than u2 is cheaper than u3.
/// @return The generated target instance.
static QkTarget *get_u1_u2_u3_target(void) {
    QkTarget *target_u1_u2_u3 = qk_target_new(1);

    double u_errors[3] = {0., 1e-4, 1e-4};
    QkGate u_gates[3] = {QkGate_U1, QkGate_U2, QkGate_U3};
    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(u_entry, qargs, 1, NAN, u_errors[idx]);
        qk_target_add_instruction(target_u1_u2_u3, u_entry);
    }
    return target_u1_u2_u3;
}

/// @brief Generates a typical target where continuous rz and rx are available; rz is cheaper.
/// @return The generated target instance.
static QkTarget *get_rz_rx_target(void) {
    QkTarget *target_rz_rx = qk_target_new(1);

    double r_errors[2] = {0., 2.5e-4};
    double r_durations[2] = {0., 5e-9};
    QkGate r_gates[2] = {QkGate_RZ, QkGate_RX};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *r_entry = qk_target_entry_new(r_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(r_entry, qargs, 1, r_durations[idx], r_errors[idx]);
        qk_target_add_instruction(target_rz_rx, r_entry);
    }
    return target_rz_rx;
}

/// @brief Generates a typical target where continuous rz, and discrete sx are available; rz is
/// cheaper.
/// @return The generated target instance.
static QkTarget *get_rz_sx_target(void) {
    QkTarget *target_rz_sx = qk_target_new(1);

    double inst_errors[2] = {0., 2.5e-4};
    double inst_durations[2] = {0., 5e-9};
    QkGate gates[2] = {QkGate_RZ, QkGate_SX};

    for (int idx = 0; idx < 2; idx++) {
        QkTargetEntry *entry;
        entry = qk_target_entry_new(gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(entry, qargs, 1, inst_durations[idx], inst_errors[idx]);
        qk_target_add_instruction(target_rz_sx, entry);
    }
    return target_rz_sx;
}

/// @brief Generates a target with overcomplete basis, rz is cheaper than ry is cheaper than u.
/// @return The generated target instance.
static QkTarget *get_rz_ry_u_target(void) {
    QkTarget *target_rz_ry_u = qk_target_new(1);

    double gate_errors[3] = {1e-4, 2e-4, 5e-4};
    double gate_durations[3] = {1e-9, 5e-9, 9e-9};
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};

    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        uint32_t qargs[1] = {
            0,
        };
        qk_target_entry_add_property(u_entry, qargs, 1, gate_durations[idx], gate_errors[idx]);
        qk_target_add_instruction(target_rz_ry_u, u_entry);
    }
    return target_rz_ry_u;
}

/// @brief Generates a target with rz, ry, and u. Error are not specified so we should prefer
/// shorter decompositions.
/// @return The generated target instance.
static QkTarget *get_rz_ry_u_noerror_target(void) {
    QkTarget *target_rz_ry_u_noerror = qk_target_new(1);
    QkGate u_gates[3] = {QkGate_RZ, QkGate_RY, QkGate_U};

    for (int idx = 0; idx < 3; idx++) {
        QkTargetEntry *u_entry = qk_target_entry_new(u_gates[idx]);
        qk_target_add_instruction(target_rz_ry_u_noerror, u_entry);
    }
    return target_rz_ry_u_noerror;
}

static bool compare_gate_counts(QkOpCounts *counts, char **gates, uint32_t *freq,
                                size_t num_gates) {
    if (counts->len != num_gates) {
        return false;
    }
    for (size_t idx = 0; idx < counts->len; idx++) {
        QkOpCount current = counts->data[idx];
        if (strcmp(current.name, gates[idx]) != 0 || current.count != freq[idx]) {
            return false;
        }
    }
    return true;
}

/**
 * Test running pass on chains of h gates.
 *
 * Transpile: 0:--[H]-[H]-[H]--
 */
static int standalone_inner_optimize_h_gates(QkTarget *target, char **gates, uint32_t *freq,
                                             size_t num_gates) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 3; iter++) {
        qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
    }

    // Run transpiler pass
    qk_transpiler_pass_standalone_optimize_1q_sequences(circuit, target);
    QkOpCounts counts = qk_circuit_count_ops(circuit);
    if (!compare_gate_counts(&counts, gates, freq, num_gates)) {
        result = EqualityError;
    }
    qk_opcounts_clear(&counts);
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

static int run_optimize_h_gates(optimize_h_gates_fn optimize_fn) {
    int num_failed = 0;
    QkTarget *targets[5] = {
        get_u1_u2_u3_target(), get_rz_rx_target(),           get_rz_sx_target(),
        get_rz_ry_u_target(),  get_rz_ry_u_noerror_target(),
    };
    // For the rz_ry_u target, RY(pi/2)-RZ(pi) has a lower total error than the single U gate
    // (~3e-4 vs 5e-4), so the error-aware basis selection picks it; without error rates
    // (rz_ry_u_noerror) the single U gate still wins on gate count.
    char *gates[5][2] = {{
                             "u2",
                         },
                         {"rz", "rx"},
                         {"rz", "sx"},
                         {"ry", "rz"},
                         {"u"}};

    uint32_t freq[5][2] = {
        {
            1,
        },
        {2, 1},
        {2, 1},
        {1, 1},
        {1},
    };

    size_t num_gates[5] = {1, 2, 2, 2, 1};
    char *names[5] = {"u1_u2_u3", "rz_rx", "rz_sx", "rz_ry_u", "rz_ry_u_noerror"};
    printf("Optimize h gates tests.\n");
    for (int idx = 0; idx < 5; idx++) {
        int result = optimize_fn(targets[idx], gates[idx], freq[idx], num_gates[idx]);
        printf("--- Run with %-21s: %s \n", names[idx], (bool)result ? "Fail" : "Ok");
        num_failed += result;
    }
    return num_failed;
}

static int test_standalone_optimize_h_gates(void) {
    return run_optimize_h_gates(standalone_inner_optimize_h_gates);
}

static int standalone_inner_optimize_identity_target(QkTarget *target) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    double params_pos[1] = {3.14 / 7.};
    double params_neg[1] = {-3.14 / 7.};
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_pos);
    qk_circuit_gate(circuit, QkGate_RY, qubits, params_neg);

    // Run transpiler pass
    qk_transpiler_pass_standalone_optimize_1q_sequences(circuit, target);
    if (qk_circuit_num_instructions(circuit) != 0) {
        result = EqualityError;
    }

    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

static int run_optimize_identity_target(optimize_identity_target_fn optimize_fn) {
    int num_failed = 0;
    QkTarget *targets[4] = {
        get_u1_u2_u3_target(),
        get_rz_rx_target(),
        get_rz_sx_target(),
        get_rz_ry_u_target(),
    };
    char *names[4] = {
        "u1_u2_u3",
        "rz_rx",
        "rz_sx",
        "rz_ry_u",
    };
    printf("Optimize identities with target tests.\n");
    for (int idx = 0; idx < 4; idx++) {
        int result = optimize_fn(targets[idx]);
        printf("--- Run with %-21s: %s \n", names[idx], (bool)result ? "Fail" : "Ok");
        num_failed += result;
    }
    return num_failed;
}

/**
 * Transpile: qr:--[RY(θ), RY(-θ)]-- to null.
 */
static int test_standalone_optimize_identity_target(void) {
    return run_optimize_identity_target(standalone_inner_optimize_identity_target);
}

/**
 * Test identity run is removed for no target specified.
 */
static int test_standalone_optimize_identity_no_target(void) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 2; iter++) {
        qk_circuit_gate(circuit, QkGate_H, qubits, NULL);
    }

    // Run transpiler pass
    qk_transpiler_pass_standalone_optimize_1q_sequences(circuit, NULL);

    if (qk_circuit_num_instructions(circuit) != 0) {
        result = EqualityError;
    }

    qk_circuit_free(circuit);
    return result;
}

/**
 * U is shorter than RZ-RY-RZ or RY-RZ-RY so use it when no error given.
 */
static int test_standalone_optimize_error_over_target_3(void) {
    int result = Ok;
    // Build circuit
    QkCircuit *circuit = qk_circuit_new(1, 0);
    uint32_t qubits[1] = {0};
    double params[3] = {3.14 / 7., 3.14 / 4., 3.14 / 3.};
    qk_circuit_gate(circuit, QkGate_U, qubits, params);
    QkTarget *target = get_rz_ry_u_noerror_target();
    // Run transpiler pass
    qk_transpiler_pass_standalone_optimize_1q_sequences(circuit, target);
    QkOpCounts counts = qk_circuit_count_ops(circuit);
    if (counts.len != 1) {
        result = EqualityError;
        goto cleanup;
    }
    if (strcmp(counts.data[0].name, "u") != 0 || counts.data[0].count != 1) {
        result = EqualityError;
    }

cleanup:
    qk_opcounts_clear(&counts);
    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test running pass on chains of h gates.
 *
 * Transpile: 0:--[H]-[H]-[H]--
 */
static int inner_optimize_h_gates(QkTarget *target, char **gates, uint32_t *freq,
                                  size_t num_gates) {
    int result = Ok;
    // Build DAG
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 3; iter++) {
        qk_dag_apply_gate(dag, QkGate_H, qubits, NULL, false);
    }

    // Run transpiler pass
    qk_transpiler_pass_optimize_1q_sequences(dag, target);

    // Compare ops via the DAG interface
    size_t num_ops = qk_dag_num_op_nodes(dag);
    uint32_t *ops = malloc(num_ops * sizeof(uint32_t));
    qk_dag_topological_op_nodes(dag, ops);
    uint32_t *actual_freq = malloc(num_gates * sizeof(uint32_t));
    memset(actual_freq, 0, num_gates * sizeof(uint32_t));
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_ops; i++) {
        qk_dag_get_instruction(dag, ops[i], &inst);
        for (size_t g = 0; g < num_gates; g++) {
            if (strcmp(inst.name, gates[g]) == 0) {
                actual_freq[g]++;
            }
        }
        qk_circuit_instruction_clear(&inst);
    }
    if (memcmp(actual_freq, freq, num_gates * sizeof(uint32_t))) {
        result = EqualityError;
    }

    free(ops);
    free(actual_freq);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_target_free(target);
    return result;
}

static int inner_optimize_identity_target(QkTarget *target) {
    int result = Ok;
    // Build circuit
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qubits[1] = {0};
    double params_pos[1] = {3.14 / 7.};
    double params_neg[1] = {-3.14 / 7.};
    qk_dag_apply_gate(dag, QkGate_RY, qubits, params_pos, false);
    qk_dag_apply_gate(dag, QkGate_RY, qubits, params_neg, false);

    // Run transpiler pass
    qk_transpiler_pass_optimize_1q_sequences(dag, target);
    if (qk_dag_num_op_nodes(dag) != 0) {
        result = EqualityError;
    }

    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_target_free(target);
    return result;
}

static int test_optimize_h_gates(void) { return run_optimize_h_gates(inner_optimize_h_gates); }

/**
 * Transpile: qr:--[RY(θ), RY(-θ)]-- to null.
 */
static int test_optimize_identity_target(void) {
    return run_optimize_identity_target(inner_optimize_identity_target);
}

/**
 * Test identity run is removed for no target specified.
 */
static int test_optimize_identity_no_target(void) {
    int result = Ok;
    // Build DAG
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qubits[1] = {0};
    for (int iter = 0; iter < 2; iter++) {
        qk_dag_apply_gate(dag, QkGate_H, qubits, NULL, false);
    }

    // Run transpiler pass
    qk_transpiler_pass_optimize_1q_sequences(dag, NULL);

    if (qk_dag_num_op_nodes(dag) != 0) {
        result = EqualityError;
    }

    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

/**
 * U is shorter than RZ-RY-RZ or RY-RZ-RY so use it when no error given.
 */
static int test_optimize_error_over_target_3(void) {
    int result = Ok;
    // Build DAG
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qubits[1] = {0};
    double params[3] = {3.14 / 7., 3.14 / 4., 3.14 / 3.};
    qk_dag_apply_gate(dag, QkGate_U, qubits, params, false);
    QkTarget *target = get_rz_ry_u_noerror_target();
    // Run transpiler pass
    qk_transpiler_pass_optimize_1q_sequences(dag, target);

    size_t num_ops = qk_dag_num_op_nodes(dag);
    if (num_ops != 1) {
        result = EqualityError;
        goto cleanup;
    }

    uint32_t *ops = malloc(num_ops * sizeof(uint32_t));
    qk_dag_topological_op_nodes(dag, ops);
    QkCircuitInstruction inst;
    qk_dag_get_instruction(dag, ops[0], &inst);
    if (strcmp(inst.name, "u")) {
        result = EqualityError;
    }
    qk_circuit_instruction_clear(&inst);
    free(ops);

cleanup:
    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

int test_optimize_1q_sequences(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_standalone_optimize_h_gates);
    num_failed += RUN_TEST(test_standalone_optimize_identity_target);
    num_failed += RUN_TEST(test_standalone_optimize_identity_no_target);
    num_failed += RUN_TEST(test_standalone_optimize_error_over_target_3);
    num_failed += RUN_TEST(test_optimize_h_gates);
    num_failed += RUN_TEST(test_optimize_identity_target);
    num_failed += RUN_TEST(test_optimize_identity_no_target);
    num_failed += RUN_TEST(test_optimize_error_over_target_3);
    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
