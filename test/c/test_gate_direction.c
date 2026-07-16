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
#include <stdio.h>

// Creates a target for the the testing functions below
static QkTarget *create_target() {
    QkTarget *target = qk_target_new(3);

    uint32_t qargs[3] = {0, 1, 2};
    QkTargetEntry *cx_entry = qk_target_entry_new(QkGate_CX);

    double rzx_params[1] = {1.5};
    QkTargetEntry *rzx_entry = qk_target_entry_new_fixed(QkGate_RZX, rzx_params, "rzx");

    if (qk_target_entry_add_property(cx_entry, qargs, 2, 0.0, 0.0) != QkExitCode_Success ||
        qk_target_entry_add_property(cx_entry, &qargs[1], 2, 0.0, 0.0) != QkExitCode_Success ||
        qk_target_add_instruction(target, cx_entry) != QkExitCode_Success ||
        qk_target_entry_add_property(rzx_entry, qargs, 2, 0.0, 0.0) != QkExitCode_Success ||
        qk_target_add_instruction(target, rzx_entry) != QkExitCode_Success) {
        printf("Unexpected error encountered in create_target.");
        qk_target_free(target);
        return NULL;
    }

    return target;
}

/**
 * Test running CheckGateDirection on a simple circuit.
 */
static int test_standalone_check_gate_direction(void) {
    QkTarget *target = create_target();
    if (!target)
        return RuntimeError;

    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(3, 0);
    uint32_t qargs[4] = {0, 1, 2, 1};

    if ((result = qk_circuit_gate(circuit, QkGate_CX, qargs, NULL)) != QkExitCode_Success ||
        (result = qk_circuit_gate(circuit, QkGate_CX, &qargs[1], NULL)) != QkExitCode_Success) {
        printf("Unexpected error encountered while adding CX gates in test_check_gate_direction.");
        goto cleanup;
    }

    bool check_pass = qk_transpiler_pass_standalone_check_gate_direction(circuit, target);
    if (!check_pass)
        result = EqualityError;
    else {
        if ((result = qk_circuit_gate(circuit, QkGate_CX, &qargs[2], NULL)) != QkExitCode_Success) {
            printf("Unexpected error encountered while adding a CX gate in "
                   "test_check_gate_direction.");
            goto cleanup;
        }
        check_pass = qk_transpiler_pass_standalone_check_gate_direction(circuit, target);
        if (check_pass)
            result = EqualityError;
    }

cleanup:
    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test running GateDirection on a simple circuit.
 */
static int test_standalone_gate_direction_simple(void) {
    QkTarget *target = create_target();
    if (!target)
        return RuntimeError;

    int result = Ok;

    QkCircuit *circuit = qk_circuit_new(3, 0);
    uint32_t qargs[5] = {0, 1, 2, 1, 0};
    double params[1] = {1.5};

    if ((result = qk_circuit_gate(circuit, QkGate_CX, qargs, NULL)) !=
            QkExitCode_Success || // stays as is
        (result = qk_circuit_gate(circuit, QkGate_CX, &qargs[1], NULL)) !=
            QkExitCode_Success || // stays as is
        (result = qk_circuit_gate(circuit, QkGate_CX, &qargs[2], NULL)) !=
            QkExitCode_Success || // would be replaced by 5 gates
        (result = qk_circuit_gate(circuit, QkGate_RZX, &qargs[3], params)) !=
            QkExitCode_Success) { // would be replaced by 5 gates
        printf("Unexpected error encountered while adding gates in test_gate_direction.");
        goto cleanup;
    }

    qk_transpiler_pass_standalone_gate_direction(circuit, target);

    if (qk_circuit_num_instructions(circuit) != 12) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_circuit_free(circuit);
    return result;
}

/**
 * Test running CheckGateDirection on a simple DAG.
 */
static int test_check_gate_direction(void) {
    QkTarget *target = create_target();
    if (!target)
        return RuntimeError;

    enum TestResult result = Ok;
    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qargs[4] = {0, 1, 2, 1};

    qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);
    qk_dag_apply_gate(dag, QkGate_CX, &qargs[1], NULL, false);

    bool check_pass = qk_transpiler_pass_check_gate_direction(dag, target);
    if (!check_pass)
        result = EqualityError;
    else {
        qk_dag_apply_gate(dag, QkGate_CX, &qargs[2], NULL, false);
        check_pass = qk_transpiler_pass_check_gate_direction(dag, target);
        if (check_pass)
            result = EqualityError;
    }

    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

/**
 * Test running GateDirection on a simple DAG.
 */
static int test_gate_direction_simple(void) {
    QkTarget *target = create_target();
    if (!target)
        return RuntimeError;

    enum TestResult result = Ok;

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qr = qk_quantum_register_new(3, "qr");
    qk_dag_add_quantum_register(dag, qr);
    uint32_t qargs[5] = {0, 1, 2, 1, 0};
    double params[1] = {1.5};

    qk_dag_apply_gate(dag, QkGate_CX, qargs, NULL, false);        // stays as is
    qk_dag_apply_gate(dag, QkGate_CX, &qargs[1], NULL, false);    // stays as is
    qk_dag_apply_gate(dag, QkGate_CX, &qargs[2], NULL, false);    // would be replaced by 5 gates
    qk_dag_apply_gate(dag, QkGate_RZX, &qargs[3], params, false); // would be replaced by 5 gates

    qk_transpiler_pass_gate_direction(dag, target);

    if (qk_dag_num_op_nodes(dag) != 12) {
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_dag_free(dag);
    qk_quantum_register_free(qr);
    return result;
}

int test_gate_direction(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_standalone_check_gate_direction);
    num_failed += RUN_TEST(test_standalone_gate_direction_simple);
    num_failed += RUN_TEST(test_check_gate_direction);
    num_failed += RUN_TEST(test_gate_direction_simple);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
