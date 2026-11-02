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
#include <inttypes.h>
#include <math.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// TODO: remove these forward declarations
// These are used for generating classical expressions for testing. They are non-public C API
// functions which should be removed once we have C API for generating classical expressions.
QkExprNode *inner_test_value(QkExprType, bool, QkDurationInfo, double, uint64_t);
void inner_expr_free(QkExprNode *);

// TODO: remove this forward declaration
// This is used to generate a circuit with control flow instructions. This a non-public C API
// function which should be removed once we have C API for adding control flow operations.
//
// clang-format off
// The circuit contains these instruction in order:
// | inst  | inst kind        |
// +-------+------------------+---------------------------------------------------------------
// |   0   | Box              | Box mapped to qubits [2,0], clbit [1], duration=0.1s
// |   1   | For Loop         | Loop over [1,2], If-Else(clbit: Break, else: Continue), param
// |   2   | Switch           | Target: ClassicalRegister, 2 Cases and Default 
// |   3   | While Loop       | Condition: clbit 
// |   4   | While Loop       | Condition: ClassicalRegister 
// |   5   | While Loop       | Condition: expression 
// |   6   | Switch           | Target: clbit 
// |   7   | Switch           | Target: expression
// |   8   | For Loop         | Loop over Range(1, 10, 3), variable loop param
// |   9   | Switch           | Target: ClassicalRegister, condition width 80 bits
// clang-format on
QkCircuit *inner_test_control_flow_circuit();

// Test a box which is constructed like this in Python
// qc = QuantumCircuit(3,3)
// inner = QuantumCircuit(2,1)
// inner.cx(0,1)
// inner.measure(0,0)
// qc.box(inner, [2,0], [1], duration=0.1, unit='s')
static int test_box_and_bit_mapping(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_Box) {
        fprintf(stderr, "Expected Box, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    size_t num_blocks = qk_control_flow_num_blocks(cf_inst);
    if (num_blocks != 1) {
        fprintf(stderr, "Expected 1 block for Box, got %zu\n", num_blocks);
        result = EqualityError;
        goto cleanup;
    }

    const QkCircuit *block_circuit = qk_control_flow_block_circuit(cf_inst, 0);
    if (block_circuit == NULL) {
        fprintf(stderr, "Failed to get block circuit at index 0\n");
        result = NullptrError;
        goto cleanup;
    }

    uint32_t num_qubits = qk_circuit_num_qubits(block_circuit);
    if (num_qubits != 2) {
        fprintf(stderr, "Expected 2 qubits in block circuit, got %u\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *qubit_mapping = qk_control_flow_qubit_map(cf_inst);
    if (qubit_mapping == NULL) {
        fprintf(stderr, "Failed to get qubit mapping\n");
        result = NullptrError;
        goto cleanup;
    }

    if (qubit_mapping[0] != 2) {
        fprintf(stderr, "Expected qubit_mapping[0] == 2, got %u\n", qubit_mapping[0]);
        result = EqualityError;
        goto cleanup;
    }

    if (qubit_mapping[1] != 0) {
        fprintf(stderr, "Expected qubit_mapping[1] == 0, got %u\n", qubit_mapping[1]);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t num_clbits = qk_circuit_num_clbits(block_circuit);
    if (num_clbits != 1) {
        fprintf(stderr, "Expected 1 clbit in block circuit, got %u\n", num_clbits);
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *clbit_mapping = qk_control_flow_clbit_map(cf_inst);
    if (clbit_mapping == NULL) {
        fprintf(stderr, "Failed to get clbit mapping\n");
        result = NullptrError;
        goto cleanup;
    }

    if (clbit_mapping[0] != 1) {
        fprintf(stderr, "Expected clbit_mapping[0] == 1, got %u\n", clbit_mapping[0]);
        result = EqualityError;
        goto cleanup;
    }

    QkBoxDurationKind duration_kind = qk_control_flow_box_duration_kind(cf_inst);
    if (duration_kind != QkBoxDurationKind_Duration) {
        fprintf(stderr, "Expected QkBoxDurationKind_Duration, got %d\n", duration_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkDurationInfo duration_info = qk_control_flow_box_duration_val_info(cf_inst);
    if (duration_info.ty != QkDurationType_S) {
        fprintf(stderr, "Expected QkDurationType_S, got %d\n", duration_info.ty);
        result = EqualityError;
        goto cleanup;
    }

    if (duration_info.value.time != 0.1) {
        fprintf(stderr, "Expected duration value 0.1, got %f\n", duration_info.value.time);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Build the body shared by the box tests.
static QkCircuit *build_box_body(void) {
    QkCircuit *body = qk_circuit_new(2, 1);
    uint32_t cx_qubits[2] = {0, 1};
    qk_circuit_gate(body, QkGate_CX, cx_qubits, NULL);
    qk_circuit_measure(body, 0, 0);
    return body;
}

// Build a circuit with a single box. Returns NULL if appending the box fails.
static QkCircuit *build_box_circuit_duration_info(const QkDurationInfo *duration) {
    QkCircuit *circuit = qk_circuit_new(3, 3);
    uint32_t box_qubits[2] = {2, 0};
    uint32_t box_clbits[1] = {1};

    // The box takes ownership of the body.
    QkExitCode result = qk_circuit_box_duration_info(circuit, build_box_body(), box_qubits,
                                                     box_clbits, duration, NULL);
    if (result != QkExitCode_Success) {
        fprintf(stderr, "qk_circuit_box_duration_info failed with exit code %d\n", result);
        qk_circuit_free(circuit);
        return NULL;
    }
    return circuit;
}

// Build a circuit with a single box. Returns NULL if appending the box fails.
static QkCircuit *build_box_circuit_expr(const QkExprNode *duration_expr) {
    QkCircuit *circuit = qk_circuit_new(3, 3);
    uint32_t box_qubits[2] = {2, 0};
    uint32_t box_clbits[1] = {1};

    // The box takes ownership of the body.
    QkExitCode result = qk_circuit_box_duration_expr(circuit, build_box_body(), box_qubits,
                                                     box_clbits, duration_expr, NULL);
    if (result != QkExitCode_Success) {
        fprintf(stderr, "qk_circuit_box_duration_expr failed with exit code %d\n", result);
        qk_circuit_free(circuit);
        return NULL;
    }
    return circuit;
}

// Build a circuit with a single box with no duration. Returns NULL if appending the box fails.
static QkCircuit *build_box_circuit_no_duration(void) {
    QkCircuit *circuit = qk_circuit_new(3, 3);
    uint32_t box_qubits[2] = {2, 0};
    uint32_t box_clbits[1] = {1};

    // The box takes ownership of the body.
    QkExitCode result = qk_circuit_box(circuit, build_box_body(), box_qubits, box_clbits, NULL);
    if (result != QkExitCode_Success) {
        fprintf(stderr, "qk_circuit_box failed with exit code %d\n", result);
        qk_circuit_free(circuit);
        return NULL;
    }
    return circuit;
}

// Test that a Box added with `qk_circuit_box` reads back correctly through the control flow
// accessors, on a circuit built entirely from C rather than by the
// `inner_test_control_flow_circuit` fixture.
static int test_box_construction(void) {
    int result = Ok;
    QkDurationInfo duration = {QkDurationType_S, {.time = 0.1}};
    QkCircuit *circuit = build_box_circuit_duration_info(&duration);
    QkControlFlowInstruction *cf_inst = NULL;

    if (circuit == NULL) {
        return RuntimeError;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_Box) {
        fprintf(stderr, "Expected Box, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    size_t num_blocks = qk_control_flow_num_blocks(cf_inst);
    if (num_blocks != 1) {
        fprintf(stderr, "Expected 1 block for Box, got %zu\n", num_blocks);
        result = EqualityError;
        goto cleanup;
    }

    const QkCircuit *block_circuit = qk_control_flow_block_circuit(cf_inst, 0);
    if (block_circuit == NULL) {
        fprintf(stderr, "Failed to get block circuit at index 0\n");
        result = NullptrError;
        goto cleanup;
    }

    uint32_t num_qubits = qk_circuit_num_qubits(block_circuit);
    if (num_qubits != 2) {
        fprintf(stderr, "Expected 2 qubits in block circuit, got %u\n", num_qubits);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t num_clbits = qk_circuit_num_clbits(block_circuit);
    if (num_clbits != 1) {
        fprintf(stderr, "Expected 1 clbit in block circuit, got %u\n", num_clbits);
        result = EqualityError;
        goto cleanup;
    }

    QkCircuitInstruction body_inst;
    qk_circuit_get_instruction(block_circuit, 0, &body_inst);
    bool cx_ok = strcmp(body_inst.name, "cx") == 0 && body_inst.num_qubits == 2 &&
                 body_inst.qubits[0] == 0 && body_inst.qubits[1] == 1;
    qk_circuit_instruction_clear(&body_inst);
    if (!cx_ok) {
        fprintf(stderr, "Expected the block circuit to begin with cx on qubits [0, 1]\n");
        result = EqualityError;
        goto cleanup;
    }

    qk_circuit_get_instruction(block_circuit, 1, &body_inst);
    bool measure_ok = strcmp(body_inst.name, "measure") == 0 && body_inst.num_qubits == 1 &&
                      body_inst.qubits[0] == 0 && body_inst.num_clbits == 1 &&
                      body_inst.clbits[0] == 0;
    qk_circuit_instruction_clear(&body_inst);
    if (!measure_ok) {
        fprintf(stderr, "Expected the block circuit to end with measure of qubit 0 into clbit 0\n");
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *qubit_mapping = qk_control_flow_qubit_map(cf_inst);
    if (qubit_mapping == NULL) {
        fprintf(stderr, "Failed to get qubit mapping\n");
        result = NullptrError;
        goto cleanup;
    }

    if (qubit_mapping[0] != 2 || qubit_mapping[1] != 0) {
        fprintf(stderr, "Expected qubit mapping [2, 0], got [%u, %u]\n", qubit_mapping[0],
                qubit_mapping[1]);
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *clbit_mapping = qk_control_flow_clbit_map(cf_inst);
    if (clbit_mapping == NULL) {
        fprintf(stderr, "Failed to get clbit mapping\n");
        result = NullptrError;
        goto cleanup;
    }

    if (clbit_mapping[0] != 1) {
        fprintf(stderr, "Expected clbit_mapping[0] == 1, got %u\n", clbit_mapping[0]);
        result = EqualityError;
        goto cleanup;
    }

    QkBoxDurationKind duration_kind = qk_control_flow_box_duration_kind(cf_inst);
    if (duration_kind != QkBoxDurationKind_Duration) {
        fprintf(stderr, "Expected QkBoxDurationKind_Duration, got %d\n", duration_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkDurationInfo duration_info = qk_control_flow_box_duration_val_info(cf_inst);
    if (duration_info.ty != QkDurationType_S) {
        fprintf(stderr, "Expected QkDurationType_S, got %d\n", duration_info.ty);
        result = EqualityError;
        goto cleanup;
    }

    if (duration_info.value.time != 0.1) {
        fprintf(stderr, "Expected duration value 0.1, got %f\n", duration_info.value.time);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    return result;
}

// Test that `qk_circuit_box` produces a Box with no duration.
static int test_box_without_duration(void) {
    int result = Ok;
    QkCircuit *circuit = build_box_circuit_no_duration();
    QkControlFlowInstruction *cf_inst = NULL;

    if (circuit == NULL) {
        return RuntimeError;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkBoxDurationKind duration_kind = qk_control_flow_box_duration_kind(cf_inst);
    if (duration_kind != QkBoxDurationKind_NoDuration) {
        fprintf(stderr, "Expected QkBoxDurationKind_NoDuration, got %d\n", duration_kind);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    return result;
}

// Test that a Box with a duration as expression reads back correctly.
static int test_box_with_duration_expr(void) {
    int result = Ok;

    QkDurationInfo duration = {QkDurationType_Ns, {.time = 100}};
    QkExprNode *duration_expr = inner_test_value(QkExprType_Duration, true, duration, 0, 0);

    QkCircuit *circuit = build_box_circuit_expr(duration_expr);
    QkControlFlowInstruction *cf_inst = NULL;

    if (circuit == NULL) {
        inner_expr_free(duration_expr);
        return RuntimeError;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkBoxDurationKind duration_kind = qk_control_flow_box_duration_kind(cf_inst);
    if (duration_kind != QkBoxDurationKind_Expr) {
        fprintf(stderr, "Expected QkBoxDurationKind_Expr, got %d\n", duration_kind);
        result = EqualityError;
        goto cleanup;
    }

    const QkValue *value = qk_expr_as_value(qk_control_flow_box_duration_expr(cf_inst));

    QkDurationInfo duration_info = qk_value_duration_info(value);
    if (duration_info.ty != QkDurationType_Ns || duration_info.value.time != 100.0) {
        fprintf(stderr, "Expected a duration of 100 ns\n");
        result = EqualityError;
    }

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    inner_expr_free(duration_expr);
    return result;
}

// Test that a Box with a duration in ``dt`` units reads back correctly, i.e. the C equivalent of:
// qc.box(body, [2, 0], [1], duration=100, unit="dt")
static int test_box_duration_dt(void) {
    int result = Ok;
    QkDurationInfo duration = {QkDurationType_Dt, {.dt = 100}};
    QkCircuit *circuit = build_box_circuit_duration_info(&duration);
    QkControlFlowInstruction *cf_inst = NULL;

    if (circuit == NULL) {
        return RuntimeError;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkDurationInfo duration_info = qk_control_flow_box_duration_val_info(cf_inst);
    if (duration_info.ty != QkDurationType_Dt || duration_info.value.dt != 100) {
        fprintf(stderr, "Expected a duration of 100 dt\n");
        result = EqualityError;
    }

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    return result;
}

// Test that `qk_circuit_box` accepts NULL clbits and returns the index of the appended box.
static int test_box_index(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t h_qubits[1] = {0};
    qk_circuit_gate(circuit, QkGate_H, h_qubits, NULL);

    QkCircuit *body = qk_circuit_new(1, 0);
    uint32_t box_qubits[1] = {1};
    size_t idx = SIZE_MAX;
    QkExitCode code = qk_circuit_box(circuit, body, box_qubits, NULL, &idx);
    if (code != QkExitCode_Success || idx != 1) {
        fprintf(stderr, "Expected the box at index 1, got exit code %d and index %zu\n", code, idx);
        result = EqualityError;
    }

    qk_circuit_free(circuit);
    return result;
}

// Test the error codes returned when appending a box fails.
static int test_box_errors(void) {
    int result = Ok;
    QkParam *outer_a = qk_param_new_symbol("a");
    QkParam *inner_a = qk_param_new_symbol("a");
    uint32_t qubits[1] = {0};

    // A parameter in the body whose name clashes with a different one in the outer circuit.
    QkCircuit *circuit = qk_circuit_new(1, 0);
    const QkParam *outer_params[1] = {outer_a};
    qk_circuit_parameterized_gate(circuit, QkGate_RX, qubits, outer_params);

    QkCircuit *body = qk_circuit_new(1, 0);
    const QkParam *inner_params[1] = {inner_a};
    qk_circuit_parameterized_gate(body, QkGate_RX, qubits, inner_params);

    QkExitCode code = qk_circuit_box(circuit, body, qubits, NULL, NULL);
    if (code != QkExitCode_ParameterNameConflict) {
        fprintf(stderr, "Expected QkExitCode_ParameterNameConflict, got %d\n", code);
        result = EqualityError;
    }

    qk_circuit_free(circuit);
    qk_param_free(outer_a);
    qk_param_free(inner_a);
    return result;
}

// Test a for loop which is constructed like this in Python:
// qc = QuantumCircuit(2, 1)
// with qc.for_loop([1,2], loop_parameter=Parameter("x")):
//     qc.h(0)
//     qc.cx(0, 1)
//     qc.measure(0, 0)
//     with qc.if_test((0, True)) as else_:
//         qc.break_loop()
//     with else_:
//         qc.continue_loop()
//     qc.h(0)
static int test_for_nested_break_continue(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;
    QkControlFlowInstruction *if_inst = NULL;
    QkControlFlowInstruction *break_inst = NULL;
    QkControlFlowInstruction *cont_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 1, NULL);

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_ForLoop) {
        fprintf(stderr, "Expected ForLoop, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopCollectionType collection_type = qk_control_flow_loop_collection_type(cf_inst);
    if (collection_type != QkLoopCollectionType_List) {
        fprintf(stderr, "Expected a List collection type, got %u\n", collection_type);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopElements loop_elements = qk_control_flow_loop_elements(cf_inst);
    if (loop_elements.len != 2) {
        fprintf(stderr, "Expected 2 loop elements, got %zu\n", loop_elements.len);
        result = EqualityError;
        goto cleanup;
    }

    if (loop_elements.elements == NULL) {
        fprintf(stderr, "Failed to get loop elements pointer\n");
        result = NullptrError;
        goto cleanup;
    }

    for (size_t i = 0; i < loop_elements.len; i++) {
        if (loop_elements.elements[i] != (ptrdiff_t)i + 1) {
            fprintf(stderr, "Expected loop_elements.elements[%zu] == %zu, got %td\n", i, i + 1,
                    loop_elements.elements[i]);
            result = EqualityError;
            goto cleanup;
        }
    }

    size_t num_blocks = qk_control_flow_num_blocks(cf_inst);
    if (num_blocks != 1) {
        fprintf(stderr, "Expected 1 block for ForLoop, got %zu\n", num_blocks);
        result = EqualityError;
        goto cleanup;
    }

    const QkCircuit *for_block = qk_control_flow_block_circuit(cf_inst, 0);
    if (for_block == NULL) {
        fprintf(stderr, "Failed to get for loop block circuit\n");
        result = NullptrError;
        goto cleanup;
    }

    size_t num_instructions = qk_circuit_num_instructions(for_block);
    if (num_instructions != 5) {
        fprintf(stderr, "Expected 5 instructions in for loop block, got %zu\n", num_instructions);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopParamKind param_kind = qk_control_flow_loop_param_kind(cf_inst);
    if (param_kind != QkLoopParamKind_Parameter) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_Parameter, got %d\n",
                param_kind);
        result = EqualityError;
        goto cleanup;
    }

    QkSymbolInfo symbol_info = qk_control_flow_loop_symbol_info(cf_inst);
    if (symbol_info.ty != QkSymbolType_Standalone) {
        fprintf(stderr, "Expected symbol type to be QkSymbolType_Standalone, got %d\n",
                symbol_info.ty);
        result = EqualityError;
        goto cleanup;
    }

    if (symbol_info.name == NULL) {
        fprintf(stderr, "Expected symbol name to be non-NULL, but it is\n");
        result = NullptrError;
        goto cleanup;
    }

    if (strcmp(symbol_info.name, "x") != 0) {
        fprintf(stderr, "Expected symbol name to be 'x', got %s\n", symbol_info.name);
        qk_str_free(symbol_info.name);
        result = EqualityError;
        goto cleanup;
    }
    qk_str_free(symbol_info.name);

    // Get the if-else instruction (4th instruction, index 3)
    if_inst = qk_circuit_get_control_flow_instruction(for_block, 3, cf_inst);
    if (if_inst == NULL) {
        fprintf(stderr, "Failed to get if-else control flow instruction at index 3\n");
        result = NullptrError;
        goto cleanup;
    }

    kind = qk_control_flow_kind(if_inst);
    if (kind != QkControlFlowKind_IfElse) {
        fprintf(stderr, "Expected IfElse, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    num_blocks = qk_control_flow_num_blocks(if_inst);
    if (num_blocks != 2) {
        fprintf(stderr, "Expected 2 blocks for IfElse, got %zu\n", num_blocks);
        result = EqualityError;
        goto cleanup;
    }

    // Check the then block contains break_loop
    const QkCircuit *then_block = qk_control_flow_block_circuit(if_inst, 0);
    if (then_block == NULL) {
        fprintf(stderr, "Failed to get then block circuit\n");
        result = NullptrError;
        goto cleanup;
    }

    break_inst = qk_circuit_get_control_flow_instruction(then_block, 0, if_inst);
    if (break_inst == NULL) {
        fprintf(stderr, "Failed to get break_loop instruction from then block\n");
        result = NullptrError;
        goto cleanup;
    }

    kind = qk_control_flow_kind(break_inst);
    if (kind != QkControlFlowKind_BreakLoop) {
        fprintf(stderr, "Expected BreakLoop in then block, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    // Check the else block contains continue_loop
    const QkCircuit *else_block = qk_control_flow_block_circuit(if_inst, 1);
    if (else_block == NULL) {
        fprintf(stderr, "Failed to get else block circuit\n");
        result = NullptrError;
        goto cleanup;
    }

    cont_inst = qk_circuit_get_control_flow_instruction(else_block, 0, if_inst);
    if (cont_inst == NULL) {
        fprintf(stderr, "Failed to get continue_loop instruction from else block\n");
        result = NullptrError;
        goto cleanup;
    }

    kind = qk_control_flow_kind(cont_inst);
    if (kind != QkControlFlowKind_ContinueLoop) {
        fprintf(stderr, "Expected ContinueLoop in else block, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cont_inst != NULL) {
        qk_control_flow_instruction_free(cont_inst);
    }
    if (break_inst != NULL) {
        qk_control_flow_instruction_free(break_inst);
    }
    if (if_inst != NULL) {
        qk_control_flow_instruction_free(if_inst);
    }
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test a switch statement on a classical register which is constructed like this in Python:
// qc = QuantumCircuit(2, 2)
// cr = ClassicalRegister(2, 'cr')
// with qc.switch(cr) as case:
//     with case((1 << 80) - 1):
//         qc.x(0)
//     with case(1, 2, 3):
//         qc.h(1)
//     with case((1 << 64) - 1, case.DEFAULT):
//         qc.y(2)
static int test_switch_case_on_register(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 2, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get control flow instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_Switch) {
        fprintf(stderr, "Expected Switch, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    size_t num_cases = qk_control_flow_switch_num_cases(cf_inst);
    if (num_cases != 3) {
        fprintf(stderr, "Expected 3 cases for Switch, got %zu\n", num_cases);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_switch_target_type(cf_inst);
    if (cond_type != QkConditionType_ClReg) {
        fprintf(stderr, "Expected QkConditionType_ClReg, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    const QkClassicalRegister *cond_creg = qk_control_flow_switch_target_register(cf_inst);
    if (cond_creg == NULL) {
        fprintf(stderr, "Failed to get switch target register\n");
        result = NullptrError;
        goto cleanup;
    }

    // Test case 0: single label (1<<80) - 1
    uint64_t bit_width = qk_control_flow_switch_case_labels_bit_width(cf_inst, 0);
    if (bit_width <= 64) {
        fprintf(stderr, "Expected label width to be larger than 64 bits, got %" PRIu64 "\n",
                bit_width);
        result = EqualityError;
        goto cleanup;
    }

    // Test case 1: multiple labels (1, 2, 3)
    QkSwitchCaseLabels case_labels = qk_control_flow_switch_case_labels_uint(cf_inst, 1);
    if (case_labels.num_labels != 3) {
        fprintf(stderr, "Expected 3 labels for case 1, got %zu\n", case_labels.num_labels);
        qk_control_flow_switch_case_labels_clear(&case_labels);
        result = EqualityError;
        goto cleanup;
    }

    for (size_t l = 0; l < 3; l++) {
        if (case_labels.labels[l] != l + 1) {
            fprintf(stderr, "Expected label %zu for case 1, got %" PRIu64 "\n", l + 1,
                    case_labels.labels[l]);
            qk_control_flow_switch_case_labels_clear(&case_labels);
            result = EqualityError;
            goto cleanup;
        }
    }
    qk_control_flow_switch_case_labels_clear(&case_labels);

    // Test case 2: A label and DEFAULT ((1<<64)-1, DEFAULT)
    case_labels = qk_control_flow_switch_case_labels_uint(cf_inst, 2);
    if (case_labels.num_labels != 1) {
        fprintf(stderr, "Expected one label for case 2, got %zu\n", case_labels.num_labels);
        qk_control_flow_switch_case_labels_clear(&case_labels);
        result = EqualityError;
        goto cleanup;
    }
    qk_control_flow_switch_case_labels_clear(&case_labels);

    bool is_default = qk_control_flow_switch_is_case_default(cf_inst, 1);
    if (is_default) {
        fprintf(stderr, "Expected case 1 to not be default, but it is\n");
        result = EqualityError;
        goto cleanup;
    }

    is_default = qk_control_flow_switch_is_case_default(cf_inst, 2);
    if (!is_default) {
        fprintf(stderr, "Expected case 2 to be default, but it is not\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test while loop with condition on classical bit:
// with qc.while_loop((cr[1], False)):
//     qc.x(0)
static int test_while_on_bit(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 3, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get while loop instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_While) {
        fprintf(stderr, "Expected While, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_condition_type(cf_inst);
    if (cond_type != QkConditionType_ClBit) {
        fprintf(stderr, "Expected QkConditionType_ClBit, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionBitInfo bit_info = qk_control_flow_condition_bit_info(cf_inst);
    if (bit_info.clbit != 1) {
        fprintf(stderr, "Expected clbit 1, got %u\n", bit_info.clbit);
        result = EqualityError;
        goto cleanup;
    }

    if (bit_info.condition != false) {
        fprintf(stderr, "Expected condition false, got %d\n", bit_info.condition);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test while loop with condition on classical register:
// with qc.while_loop((cr, 7)):
//     qc.y(0)
static int test_while_on_register(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 4, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get while loop instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_While) {
        fprintf(stderr, "Expected While, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_condition_type(cf_inst);
    if (cond_type != QkConditionType_ClReg) {
        fprintf(stderr, "Expected QkConditionType_ClReg, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    const QkClassicalRegister *reg = qk_control_flow_condition_reg(cf_inst);
    if (reg == NULL) {
        fprintf(stderr, "Expected non-null classical register\n");
        result = NullptrError;
        goto cleanup;
    }

    uint64_t cond_val = qk_control_flow_condition_reg_cond_uint(cf_inst);
    if (cond_val != 7) {
        fprintf(stderr, "Expected condition value 7, got %" PRIu64 "\n", cond_val);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test while loop with condition on expression:
// with qc.while_loop(expr.less(cr, 7)):
//     qc.z(0)
static int test_while_on_expr(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 5, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get while loop instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_While) {
        fprintf(stderr, "Expected While, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_condition_type(cf_inst);
    if (cond_type != QkConditionType_Expr) {
        fprintf(stderr, "Expected QkConditionType_Expr, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    const QkExprNode *expr = qk_control_flow_condition_expr(cf_inst);
    if (expr == NULL) {
        fprintf(stderr, "Expected non-null expression\n");
        result = NullptrError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test a switch statement on a classical bit which is constructed like this in Python:
// with qc.switch(cr[0]) as case:
//     with case(case.DEFAULT):
//         qc.x(0)
static int test_switch_case_on_bit(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 6, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get switch instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_Switch) {
        fprintf(stderr, "Expected Switch, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_switch_target_type(cf_inst);
    if (cond_type != QkConditionType_ClBit) {
        fprintf(stderr, "Expected QkConditionType_ClBit, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    uint32_t condition_bit = qk_control_flow_switch_target_bit(cf_inst);
    if (condition_bit != 0) {
        fprintf(stderr, "Expected condition bit 0, got %u\n", condition_bit);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test a switch statement on an expression which is constructed like this in Python:
// with qc.switch(expr.less(cr, 2)) as case:
//     with case(case.DEFAULT):
//         qc.y(0)
static int test_switch_case_on_expr(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = NULL;

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 7, NULL);
    if (cf_inst == NULL) {
        fprintf(stderr, "Failed to get switch instruction\n");
        result = NullptrError;
        goto cleanup;
    }

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_Switch) {
        fprintf(stderr, "Expected Switch, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkConditionType cond_type = qk_control_flow_switch_target_type(cf_inst);
    if (cond_type != QkConditionType_Expr) {
        fprintf(stderr, "Expected QkConditionType_Expr, got %d\n", cond_type);
        result = EqualityError;
        goto cleanup;
    }

    const QkExprNode *expr = qk_control_flow_switch_target_expr(cf_inst);
    if (expr == NULL) {
        fprintf(stderr, "Expected non-null expression\n");
        result = NullptrError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    return result;
}

// Test a for-loop statement over a range like this in Python:
// with qc.for_loop(range(1,10,3), expr.Var.new("v"), types.Uint(8)):
//      qc.y(0)
static int test_for_loop_over_range(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();

    QkControlFlowInstruction *cf_inst = qk_circuit_get_control_flow_instruction(circuit, 8, NULL);

    QkControlFlowKind kind = qk_control_flow_kind(cf_inst);
    if (kind != QkControlFlowKind_ForLoop) {
        fprintf(stderr, "Expected ForLoop, got %u\n", kind);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopCollectionType collection_type = qk_control_flow_loop_collection_type(cf_inst);
    if (collection_type != QkLoopCollectionType_Range) {
        fprintf(stderr, "Expected a Range collection type, got %u\n", collection_type);
        result = EqualityError;
        goto cleanup;
    }

    int64_t start, stop, step;
    qk_control_flow_loop_range(cf_inst, &start, &stop, &step);
    if (start != 1 || stop != 10 || step != 3) {
        fprintf(stderr,
                "Expected a for-loop over Range(1,10,3), got Range(%" PRIi64 ",%" PRIi64 ",%" PRIi64
                ")\n",
                start, stop, step);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopParamKind param_kind = qk_control_flow_loop_param_kind(cf_inst);
    if (param_kind != QkLoopParamKind_Variable) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_Variable, got %d\n",
                param_kind);
        result = EqualityError;
        goto cleanup;
    }

    const QkVar *loop_var = qk_control_flow_loop_variable(cf_inst);
    char *var_name = qk_var_name(loop_var);
    if (strcmp(var_name, "v") != 0) {
        fprintf(stderr, "Expected var name to be v, got %s\n", var_name);
        result = EqualityError;
        qk_str_free(var_name);
        goto cleanup;
    }
    qk_str_free(var_name);

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    return result;
}

// Test while loop with condition on classical register:
// with qc.while_loop((cr, (1<<80)-1)):
//     qc.Z(0)
static int test_while_on_register_large_condition(void) {
    int result = Ok;
    QkCircuit *circuit = inner_test_control_flow_circuit();
    QkControlFlowInstruction *cf_inst = qk_circuit_get_control_flow_instruction(circuit, 9, NULL);

    uint64_t cond_bit_width = qk_control_flow_condition_reg_cond_bit_width(cf_inst);
    if (cond_bit_width <= 64) {
        fprintf(stderr, "Expected condition width to be larger than 64 bits, got %" PRIu64 "\n",
                cond_bit_width);
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_control_flow_instruction_free(cf_inst);
    qk_circuit_free(circuit);
    return result;
}

int test_control_flow(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_box_and_bit_mapping);
    num_failed += RUN_TEST(test_box_construction);
    num_failed += RUN_TEST(test_box_without_duration);
    num_failed += RUN_TEST(test_box_with_duration_expr);
    num_failed += RUN_TEST(test_box_duration_dt);
    num_failed += RUN_TEST(test_box_index);
    num_failed += RUN_TEST(test_box_errors);
    num_failed += RUN_TEST(test_for_nested_break_continue);
    num_failed += RUN_TEST(test_switch_case_on_register);
    num_failed += RUN_TEST(test_while_on_bit);
    num_failed += RUN_TEST(test_while_on_register);
    num_failed += RUN_TEST(test_while_on_expr);
    num_failed += RUN_TEST(test_switch_case_on_bit);
    num_failed += RUN_TEST(test_switch_case_on_expr);
    num_failed += RUN_TEST(test_for_loop_over_range);
    num_failed += RUN_TEST(test_while_on_register_large_condition);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
