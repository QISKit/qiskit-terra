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

static int test_transpile_bv(void) {
    const uint32_t num_qubits = 10;
    QkTarget *target = qk_target_new(num_qubits);
    int result = Ok;

    QkTargetEntry *x_entry = qk_target_entry_new(QkGate_X);
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {
            i,
        };
        double error = 0.8e-6 * (i + 1);
        double duration = 1.8e-9 * (i + 1);
        qk_target_entry_add_property(x_entry, qargs, 1, duration, error);
    }
    qk_target_add_instruction(target, x_entry);

    QkTargetEntry *sx_entry = qk_target_entry_new(QkGate_SX);
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {
            i,
        };
        double error = 0.8e-6 * (i + 1);
        double duration = 1.8e-9 * (i + 1);
        qk_target_entry_add_property(sx_entry, qargs, 1, duration, error);
    }
    qk_target_add_instruction(target, sx_entry);

    QkTargetEntry *rz_entry = qk_target_entry_new(QkGate_RZ);
    for (uint32_t i = 0; i < num_qubits; i++) {
        uint32_t qargs[1] = {
            i,
        };
        double error = 0.;
        double duration = 0.;
        qk_target_entry_add_property(rz_entry, qargs, 1, duration, error);
    }
    qk_target_add_instruction(target, rz_entry);

    QkTargetEntry *ecr_entry = qk_target_entry_new(QkGate_ECR);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        uint32_t qargs[2] = {i, i + 1};
        double inst_error = 0.0090393 * (num_qubits - i);
        double inst_duration = 0.020039;

        qk_target_entry_add_property(ecr_entry, qargs, 2, inst_duration, inst_error);
    }
    qk_target_add_instruction(target, ecr_entry);

    QkCircuit *qc = qk_circuit_new(num_qubits, 0);
    uint32_t x_qargs[1] = {
        9,
    };
    qk_circuit_gate(qc, QkGate_X, x_qargs, NULL);
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc); i++) {
        uint32_t qargs[1] = {
            i,
        };
        qk_circuit_gate(qc, QkGate_H, qargs, NULL);
    }
    for (uint32_t i = 0; i < qk_circuit_num_qubits(qc) - 1; i += 2) {
        uint32_t qargs[2] = {i, num_qubits - 1};
        qk_circuit_gate(qc, QkGate_CX, qargs, NULL);
    }
    QkTranspileResult transpile_result = {NULL, NULL};
    char *error = NULL;
    QkTranspileOptions options = qk_transpiler_default_options();
    options.seed = 42;
    int result_code = qk_transpile(qc, target, &options, &transpile_result, &error);
    if (result_code != 0) {
        fprintf(stderr, "Transpilation failed with: %s\n", error);
        result = EqualityError;
        qk_str_free(error);
        goto circuit_cleanup;
    }

    QkOpCounts op_counts = qk_circuit_count_ops(transpile_result.circuit);
    if (op_counts.len != 4) {
        fprintf(stderr, "More than 4 types of gates in circuit, circuit's instructions are:\n");
        print_circuit(transpile_result.circuit);
        result = EqualityError;
        goto transpile_cleanup;
    }
    for (uint32_t i = 0; i < op_counts.len; i++) {
        int sx_gate = strcmp(op_counts.data[i].name, "sx");
        int ecr_gate = strcmp(op_counts.data[i].name, "ecr");
        int x_gate = strcmp(op_counts.data[i].name, "x");
        int rz_gate = strcmp(op_counts.data[i].name, "rz");
        if (sx_gate != 0 && ecr_gate != 0 && x_gate != 0 && rz_gate != 0) {
            fprintf(stderr, "Gate type of %s found in the circuit which isn't expected\n",
                    op_counts.data[i].name);
            result = EqualityError;
            goto transpile_cleanup;
        }
    }
    QkCircuitInstruction inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(transpile_result.circuit); i++) {
        qk_circuit_get_instruction(transpile_result.circuit, i, &inst);
        if (strcmp(inst.name, "ecr") == 0) {
            if (inst.num_qubits != 2) {
                fprintf(stderr, "Unexpected number of qubits for ecr: %d\n", inst.num_qubits);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto transpile_cleanup;
            }
            bool valid = false;
            for (uint32_t qubit = 0; qubit < num_qubits - 1; qubit++) {
                if (inst.qubits[0] == qubit && inst.qubits[1] == qubit + 1) {
                    valid = true;
                    break;
                }
            }
            if (valid == false) {
                fprintf(stderr, "ECR Gate outside target on qubits: {%u, %u}\n", inst.qubits[0],
                        inst.qubits[1]);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto transpile_cleanup;
            }
        }
        qk_circuit_instruction_clear(&inst);
    }

transpile_cleanup:
    qk_circuit_free(transpile_result.circuit);
    qk_transpile_layout_free(transpile_result.layout);
    qk_opcounts_clear(&op_counts);

circuit_cleanup:
    qk_circuit_free(qc);
    qk_target_free(target);
    return result;
}

static int test_transpile_idle_qubits(void) {
    int result = Ok;
    uint32_t num_qubits = 3;
    QkCircuit *circuit = qk_circuit_new(num_qubits, 0);
    uint32_t qargs[4];
    double params[1];
    qargs[0] = 2;
    qargs[1] = 1;
    params[0] = 1.681876;
    qk_circuit_gate(circuit, QkGate_CRZ, qargs, params);
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    for (uint8_t opt_level = 0; opt_level < 4; opt_level++) {
        QkTranspileOptions transpile_options = {opt_level, 1234, 1.0};
        QkTranspileResult transpile_result;
        char *error;
        int result_code =
            qk_transpile(circuit, target, &transpile_options, &transpile_result, &error);
        if (result_code != 0) {
            fprintf(stderr, "Transpilation failed %s\n", error);
            result = EqualityError;
            qk_str_free(error);
            goto cleanup;
        }
        size_t num_instructions = qk_circuit_num_instructions(transpile_result.circuit);
        qk_circuit_free(transpile_result.circuit);
        qk_transpile_layout_free(transpile_result.layout);
        if (opt_level == 0 && num_instructions != 12) {
            fprintf(stderr, "opt_level: %d num_instructions: %zu is not the expected value 12\n",
                    opt_level, num_instructions);
            result = EqualityError;
            goto cleanup;
        }
        if (opt_level == 1 && num_instructions != 7) {
            fprintf(stderr, "opt_level: %d num_instructions: %zu is not the expected value 7\n",
                    opt_level, num_instructions);
            result = EqualityError;
            goto cleanup;
        }
        if ((opt_level == 2 || opt_level == 3) && num_instructions != 7) {
            fprintf(stderr, "opt_level: %d num_instructions: %zu is not the expected value 7\n",
                    opt_level, num_instructions);
            result = EqualityError;
            goto cleanup;
        }
    }

cleanup:
    qk_circuit_free(circuit);
    qk_target_free(target);
    return result;
}

static int test_transpile_options_null(void) {
    const uint32_t n = 10;
    QkTarget *target = qk_target_new(n);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_SX));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_X));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_RZ));

    QkCircuit *circuit = qk_circuit_new(3, 0);
    for (uint32_t i = 0; i < 3; i++) {
        qk_circuit_gate(circuit, QkGate_H, (uint32_t[1]){i}, NULL);
    }

    QkTranspileResult transpile_result = {NULL, NULL};
    QkExitCode exit = qk_transpile(circuit, target, NULL, &transpile_result, NULL);

    int result = Ok;
    if (exit != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    // H gets translated to RZ-SX-RZ on each qubit
    size_t num_inst = qk_circuit_num_instructions(transpile_result.circuit);
    if (num_inst != 9) {
        result = EqualityError;
        fprintf(stderr, "Expected 9 instruction, but got %zu\n", num_inst);
    }

cleanup:
    qk_target_free(target);
    qk_circuit_free(circuit);
    qk_circuit_free(transpile_result.circuit);
    qk_transpile_layout_free(transpile_result.layout);

    return result;
}

static int test_init_stage_empty(void) {
    int result = Ok;
    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "qr");
    qk_dag_add_quantum_register(dag, qr);
    QkTranspilerStageState **state = malloc(sizeof(QkTranspilerStageState *));
    *state = NULL;
    int compile_result = qk_transpile_stage_init(dag, target, NULL, state, NULL);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the init stage failed\n");
        goto cleanup;
    }
    uint32_t num_dag_qubits = qk_dag_num_qubits(dag);
    if (num_dag_qubits != 1024) {
        result = EqualityError;
        fprintf(stderr, "Number of dag qubits %u does not match expected result 1024\n",
                num_dag_qubits);
    }
cleanup:
    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_transpile_state_free(*state);
    free(state);
    return result;
}

static int test_layout_stage_empty(void) {
    int result = Ok;
    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "qr");
    qk_dag_add_quantum_register(dag, qr);
    QkTranspilerStageState **state = malloc(sizeof(QkTranspilerStageState *));
    *state = NULL;
    int compile_result = qk_transpile_stage_layout(dag, target, NULL, state, NULL);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the layout stage failed\n");
        goto cleanup;
    }
    uint32_t num_dag_qubits = qk_dag_num_qubits(dag);
    if (num_dag_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of dag qubits %u does not match expected result 2048\n",
                num_dag_qubits);
    }
    QkTranspileLayout *layout = qk_transpile_state_layout(*state);
    uint32_t num_layout_qubits = qk_transpile_layout_num_output_qubits(layout);
    if (num_layout_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of layout qubits %u does not match expected result 2048\n",
                num_layout_qubits);
    }
cleanup:
    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_transpile_state_free(*state);
    free(state);
    return result;
}

static int test_routing_stage_empty(void) {
    int result = Ok;
    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(1024, "qr");
    qk_dag_add_quantum_register(dag, qr);
    QkTranspilerStageState **state = malloc(sizeof(QkTranspilerStageState *));
    *state = NULL;
    // Run layout stage to populate the QkTranspileLayout
    int compile_result = qk_transpile_stage_layout(dag, target, NULL, state, NULL);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the layout stage failed\n");
        goto cleanup;
    }
    // The layout should already be written to the state object, so we can dereference.
    compile_result = qk_transpile_stage_routing(dag, target, NULL, *state, NULL);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the routing stage failed\n");
        goto cleanup;
    }
    uint32_t num_dag_qubits = qk_dag_num_qubits(dag);
    if (num_dag_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of dag qubits %u does not match expected result 2048\n",
                num_dag_qubits);
    }
    QkTranspileLayout *layout = qk_transpile_state_layout(*state);
    uint32_t num_layout_qubits = qk_transpile_layout_num_output_qubits(layout);
    if (num_layout_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of layout qubits %u does not match expected result 2048\n",
                num_layout_qubits);
    }
cleanup:
    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    qk_transpile_state_free(*state);
    free(state);
    return result;
}

static int test_translation_stage_empty(void) {
    int result = Ok;
    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2048, "qr");
    qk_dag_add_quantum_register(dag, qr);
    int compile_result = qk_transpile_stage_translation(dag, target, NULL, NULL);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the layout stage failed\n");
        goto cleanup;
    }
    uint32_t num_dag_qubits = qk_dag_num_qubits(dag);
    if (num_dag_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of dag qubits %u does not match expected result 2048\n",
                num_dag_qubits);
    }
cleanup:
    qk_target_free(target);
    qk_quantum_register_free(qr);
    qk_dag_free(dag);
    return result;
}

static int test_optimization_stage_empty(void) {
    int result = Ok;
    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);
    for (uint32_t i = 0; i < num_qubits - 1; i++) {
        qk_target_entry_add_property(cx_entry, (uint32_t[]){i, i + 1}, 2, 0.001 * i, 0.002 * i);
    }
    qk_target_add_instruction(target, cx_entry);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_U));

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(2048, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t *layout_mapping = malloc(sizeof(uint32_t) * 2048);
    for (uint32_t i = 0; i < 2048; i++) {
        layout_mapping[2047 - i] = i;
    };
    // Do not free the layout, State will gain ownership of it.
    QkTranspileLayout *layout =
        qk_transpile_layout_generate_from_mapping(dag, target, layout_mapping);
    QkTranspilerStageState **state = malloc(sizeof(QkTranspilerStageState *));
    qk_transpile_state_new(state);
    qk_transpile_state_layout_set(*state, layout);
    int compile_result = qk_transpile_stage_optimization(dag, target, NULL, NULL, *state);
    if (compile_result != 0) {
        result = EqualityError;
        fprintf(stderr, "Running the optimization stage failed\n");
        goto cleanup;
    }
    uint32_t num_dag_qubits = qk_dag_num_qubits(dag);
    if (num_dag_qubits != 2048) {
        result = EqualityError;
        fprintf(stderr, "Number of dag qubits %u does not match expected result 2048\n",
                num_dag_qubits);
    }
cleanup:
    free(layout_mapping);
    qk_transpile_state_free(*state);
    free(state);
    qk_target_free(target);
    qk_quantum_register_free(qr);
    qk_dag_free(dag);
    return result;
}

static int test_transpile_state_layout_unset(void) {
    int result = RuntimeError;

    uint32_t num_qubits = 2048;
    QkTarget *target = qk_target_new(num_qubits);

    QkDag *dag = qk_dag_new();

    QkQuantumRegister *qr = qk_quantum_register_new(num_qubits, "qr");
    qk_dag_add_quantum_register(dag, qr);
    qk_quantum_register_free(qr);

    uint32_t *mapping = malloc(sizeof(uint32_t) * num_qubits);
    for (uint32_t i = 0; i < num_qubits; i++) {
        mapping[num_qubits - i - 1] = i;
    };

    QkTranspileLayout *layout = qk_transpile_layout_generate_from_mapping(dag, target, mapping);

    QkTranspilerStageState *state = NULL;
    qk_transpile_state_new(&state);
    qk_transpile_state_layout_set(state, layout);

    QkTranspileLayout *current = qk_transpile_state_layout(state);
    if (current == NULL) {
        result = EqualityError;
        fprintf(stderr, "state layout not set\n");
        goto cleanup;
    }

    qk_transpile_state_layout_set(state, NULL);
    current = qk_transpile_state_layout(state);

    if (current != NULL) {
        result = EqualityError;
        fprintf(stderr, "state layout exists after unset\n");
    } else {
        result = Ok;
    }

cleanup:
    free(mapping);
    qk_transpile_state_free(state);
    qk_dag_free(dag);
    qk_target_free(target);
    return result;
}

int test_transpiler(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_transpile_bv);
    num_failed += RUN_TEST(test_transpile_idle_qubits);
    num_failed += RUN_TEST(test_transpile_options_null);
    num_failed += RUN_TEST(test_transpile_state_layout_unset);
    num_failed += RUN_TEST(test_init_stage_empty);
    num_failed += RUN_TEST(test_layout_stage_empty);
    num_failed += RUN_TEST(test_routing_stage_empty);
    num_failed += RUN_TEST(test_translation_stage_empty);
    num_failed += RUN_TEST(test_optimization_stage_empty);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
