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

// TODO: remove this forward declaration
void inner_expr_free(QkExprNode *);
bool inner_test_add_input_var(QkCircuit *, const QkVar *);
QkExprNode *inner_test_expr_kinds_and_types(QkExprNodeKind, QkExprTypeInfo);

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

// Build a for-loop over a range with a loop parameter, then read it back:
// i = Parameter("i")
// body = QuantumCircuit(2)
// body.rx(i, 0)
// body.cx(0, 1)
// qc = QuantumCircuit(3)
// qc.for_loop(range(0, 5, 1), i, body, [2, 0], [])
static int test_build_for_loop_over_range(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(3, 0);
    QkParam *param = qk_param_new_symbol("i");
    QkCircuit *body = qk_circuit_new(2, 0);
    QkControlFlowInstruction *cf_inst = NULL;

    uint32_t body_qubits[2] = {0, 1};
    const QkParam *rx_params[1] = {param};
    qk_circuit_parameterized_gate(body, QkGate_RX, body_qubits, rx_params);
    qk_circuit_gate(body, QkGate_CX, body_qubits, NULL);

    QkSymbolInfo symbol = {.ty = QkSymbolType_Standalone, .name = "i", .index = 0};
    QkLoopParamValue loop_param = {.parameter = &symbol};

    uint32_t qubits[2] = {2, 0};

    size_t idx = SIZE_MAX;
    QkExitCode exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                                     QkLoopParamKind_Parameter, &loop_param, &idx);
    if (exit_code != QkExitCode_Success || idx != 0) {
        fprintf(stderr,
                "Expected the for-loop to be appended at index 0, got exit code %u and index %zu\n",
                exit_code, idx);
        result = RuntimeError;
        goto cleanup;
    }

    if (qk_circuit_num_instructions(circuit) != 1) {
        fprintf(stderr, "Expected 1 instruction, got %zu\n", qk_circuit_num_instructions(circuit));
        result = EqualityError;
        goto cleanup;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (qk_control_flow_kind(cf_inst) != QkControlFlowKind_ForLoop) {
        fprintf(stderr, "Expected ForLoop, got %u\n", qk_control_flow_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

    const QkCircuit *block = qk_control_flow_block_circuit(cf_inst, 0);
    if (qk_circuit_num_instructions(block) != 2) {
        fprintf(stderr, "Expected the copied body to hold 2 instructions, got %zu\n",
                qk_circuit_num_instructions(block));
        result = EqualityError;
        goto cleanup;
    }

    const uint32_t *qubit_map = qk_control_flow_qubit_map(cf_inst);
    if (qubit_map[0] != 2 || qubit_map[1] != 0) {
        fprintf(stderr, "Expected the body's qubits to map onto [2, 0], got [%u, %u]\n",
                qubit_map[0], qubit_map[1]);
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
    if (start != 0 || stop != 5 || step != 1) {
        fprintf(stderr,
                "Expected a for-loop over Range(0,5,1), got Range(%" PRIi64 ",%" PRIi64 ",%" PRIi64
                ")\n",
                start, stop, step);
        result = EqualityError;
        goto cleanup;
    }

    if (qk_control_flow_loop_param_kind(cf_inst) != QkLoopParamKind_Parameter) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_Parameter, got %d\n",
                qk_control_flow_loop_param_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

    QkSymbolInfo symbol_info = qk_control_flow_loop_symbol_info(cf_inst);
    if (strcmp(symbol_info.name, "i") != 0) {
        fprintf(stderr, "Expected loop parameter name to be i, got %s\n", symbol_info.name);
        result = EqualityError;
    }
    qk_str_free(symbol_info.name);

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    qk_param_free(param);
    return result;
}

// i = Parameter("i")
// body = QuantumCircuit(2)
// body.rx(i, 0)
// body.cx(0, 1)
// qc.for_loop([1, 3, 7], i, body, [0, 1], [])
static int test_build_for_loop_over_elements(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    QkParam *param = qk_param_new_symbol("i");
    QkCircuit *body = qk_circuit_new(2, 0);
    QkControlFlowInstruction *cf_inst = NULL;

    uint32_t body_qubits[2] = {0, 1};
    const QkParam *rx_params[1] = {param};
    qk_circuit_parameterized_gate(body, QkGate_RX, body_qubits, rx_params);
    qk_circuit_gate(body, QkGate_CX, body_qubits, NULL);

    ptrdiff_t values[3] = {1, 3, 7};
    QkLoopElements elements = {values, 3};
    QkSymbolInfo symbol = {.ty = QkSymbolType_Standalone, .name = "i", .index = 0};
    QkLoopParamValue loop_param = {.parameter = &symbol};

    uint32_t qubits[2] = {0, 1};

    QkExitCode exit_code = qk_circuit_for_loop_elements(
        circuit, body, qubits, NULL, elements, QkLoopParamKind_Parameter, &loop_param, NULL);
    if (exit_code != QkExitCode_Success) {
        fprintf(stderr, "Expected the for-loop to be appended, got exit code %u\n", exit_code);
        result = RuntimeError;
        goto cleanup;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (qk_control_flow_kind(cf_inst) != QkControlFlowKind_ForLoop) {
        fprintf(stderr, "Expected ForLoop, got %u\n", qk_control_flow_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

    QkLoopCollectionType collection_type = qk_control_flow_loop_collection_type(cf_inst);
    if (collection_type != QkLoopCollectionType_List) {
        fprintf(stderr, "Expected a List collection type, got %u\n", collection_type);
        result = EqualityError;
        goto cleanup;
    }

    QkLoopElements out_elements = qk_control_flow_loop_elements(cf_inst);
    if (out_elements.len != 3 || out_elements.elements[0] != 1 || out_elements.elements[1] != 3 ||
        out_elements.elements[2] != 7) {
        fprintf(stderr, "Expected the loop to iterate over [1, 3, 7]\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_control_flow_loop_param_kind(cf_inst) != QkLoopParamKind_Parameter) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_Parameter, got %d\n",
                qk_control_flow_loop_param_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    qk_param_free(param);
    return result;
}

// Without a loop parameter the body's own parameter stays free, so it becomes a parameter of the
// outer circuit instead of being bound on each iteration:
// i = Parameter("i")
// body = QuantumCircuit(2)
// body.rx(i, 0)
// qc.for_loop([1, 3, 7], None, body, [0, 1], [])
static int test_build_for_loop_without_loop_param(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    QkParam *param = qk_param_new_symbol("i");
    QkCircuit *body = qk_circuit_new(2, 0);
    QkControlFlowInstruction *cf_inst = NULL;

    uint32_t body_qubits[2] = {0, 1};
    const QkParam *rx_params[1] = {param};
    qk_circuit_parameterized_gate(body, QkGate_RX, body_qubits, rx_params);

    ptrdiff_t values[3] = {1, 3, 7};
    QkLoopElements elements = {values, 3};
    uint32_t qubits[2] = {0, 1};

    QkExitCode exit_code = qk_circuit_for_loop_elements(circuit, body, qubits, NULL, elements,
                                                        QkLoopParamKind_NoLoopParam, NULL, NULL);
    if (exit_code != QkExitCode_Success) {
        fprintf(stderr, "Expected the for-loop to be appended, got exit code %u\n", exit_code);
        result = RuntimeError;
        goto cleanup;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (qk_control_flow_loop_param_kind(cf_inst) != QkLoopParamKind_NoLoopParam) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_NoLoopParam, got %d\n",
                qk_control_flow_loop_param_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_param_symbols(circuit) != 1) {
        fprintf(stderr,
                "Expected the body's parameter to be a parameter of the outer circuit, got %zu\n",
                qk_circuit_num_param_symbols(circuit));
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }
    qk_circuit_free(circuit);
    qk_param_free(param);
    return result;
}

// v = expr.Var.new("test_var", types.Uint(8))
// body = QuantumCircuit(2, inputs=[v])
// body.cx(0, 1)
// qc = QuantumCircuit(3)
// qc.for_loop(range(0, 5, 1), v, body, [2, 0], [])
static int test_build_for_loop_with_loop_variable(void) {
    int result = Ok;
    QkControlFlowInstruction *cf_inst = NULL;
    QkCircuit *circuit = qk_circuit_new(3, 0);
    QkCircuit *body = qk_circuit_new(2, 0);

    uint32_t body_qubits[2] = {0, 1};
    QkExprTypeInfo type_info = {QkExprType_Uint, 8};
    // fixture gives it name "test_var"
    QkExprNode *var_expr = inner_test_expr_kinds_and_types(QkExprNodeKind_Var, type_info);
    const QkVar *var = qk_expr_as_var(var_expr);
    if (!inner_test_add_input_var(body, var)) {
        fprintf(stderr, "Failed to add the loop variable to the body\n");
        result = RuntimeError;
        qk_circuit_free(body);
        body = NULL;
        goto cleanup;
    }

    qk_circuit_gate(body, QkGate_CX, body_qubits, NULL);

    QkLoopParamValue loop_param = {.variable = var};

    uint32_t qubits[2] = {2, 0};

    size_t idx = SIZE_MAX;
    QkExitCode exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                                     QkLoopParamKind_Variable, &loop_param, &idx);
    if (exit_code != QkExitCode_Success || idx != 0) {
        fprintf(stderr,
                "Expected the for-loop to be appended at index 0, got exit code %u and index %zu\n",
                exit_code, idx);
        result = RuntimeError;
        goto cleanup;
    }

    cf_inst = qk_circuit_get_control_flow_instruction(circuit, 0, NULL);
    if (qk_control_flow_loop_param_kind(cf_inst) != QkLoopParamKind_Variable) {
        fprintf(stderr, "Expected loop parameter kind to be QkLoopParamKind_Variable, got %d\n",
                qk_control_flow_loop_param_kind(cf_inst));
        result = EqualityError;
        goto cleanup;
    }

    char *var_name = qk_var_name(qk_control_flow_loop_variable(cf_inst));
    if (strcmp(var_name, "test_var") != 0) {
        fprintf(stderr, "Expected the loop variable to be test_var, got %s\n", var_name);
        result = EqualityError;
    }

    qk_str_free(var_name);

cleanup:
    if (cf_inst != NULL) {
        qk_control_flow_instruction_free(cf_inst);
    }

    inner_expr_free(var_expr);
    qk_circuit_free(circuit);

    return result;
}

// Invalid range and loop-parameter inputs are rejected before the circuit is touched.
static int test_build_for_loop_invalid_input(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    QkCircuit *body = qk_circuit_new(2, 0);

    uint32_t qubits[2] = {0, 1};

    QkExitCode exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 0,
                                                     QkLoopParamKind_NoLoopParam, NULL, NULL);
    if (exit_code != QkExitCode_ZeroLoopStep) {
        fprintf(stderr, "Expected QkExitCode_ZeroLoopStep for a zero step, got exit code %u\n",
                exit_code);
        result = EqualityError;
        goto cleanup;
    }

    body = qk_circuit_new(2, 0);
    QkSymbolInfo symbol = {.ty = QkSymbolType_Standalone, .name = "i", .index = 0};
    QkLoopParamValue loop_param = {.parameter = &symbol};
    exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                          QkLoopParamKind_Parameter, &loop_param, NULL);
    if (exit_code != QkExitCode_ParameterError) {
        fprintf(
            stderr,
            "Expected QkExitCode_ParameterError for a loop parameter that the body does not use, "
            "got exit code %u\n",
            exit_code);
        result = EqualityError;
        goto cleanup;
    }

    body = qk_circuit_new(2, 0);
    symbol = (QkSymbolInfo){.ty = QkSymbolType_Standalone, .name = "\xff", .index = 0};
    loop_param = (QkLoopParamValue){.parameter = &symbol};
    exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                          QkLoopParamKind_Parameter, &loop_param, NULL);
    if (exit_code != QkExitCode_CInputError) {
        fprintf(
            stderr,
            "Expected QkExitCode_CInputError for a loop parameter name that is not valid UTF-8, "
            "got exit code %u\n",
            exit_code);
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_instructions(circuit) != 0) {
        fprintf(stderr, "Expected no instruction to be appended, got %zu\n",
                qk_circuit_num_instructions(circuit));
        result = EqualityError;
    }

cleanup:
    qk_circuit_free(circuit);
    return result;
}

// Invalid loop variables are rejected before the circuit is touched.
static int test_build_for_loop_invalid_loop_variable(void) {
    int result = Ok;
    QkCircuit *circuit = qk_circuit_new(3, 0);
    QkCircuit *body = qk_circuit_new(2, 0);

    QkExprTypeInfo type_info = {QkExprType_Bool, 0};
    QkExprNode *var_expr = inner_test_expr_kinds_and_types(QkExprNodeKind_Var, type_info);
    const QkVar *var = qk_expr_as_var(var_expr);

    QkLoopParamValue loop_param = {.variable = var};
    uint32_t qubits[2] = {2, 0};

    QkExitCode exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                                     QkLoopParamKind_Variable, &loop_param, NULL);
    if (exit_code != QkExitCode_InvalidLoopVariable) {
        fprintf(
            stderr,
            "Expected QkExitCode_InvalidLoopVariable for a bool loop variable, got exit code %u\n",
            exit_code);
        result = EqualityError;
        goto cleanup;
    }

    inner_expr_free(var_expr);
    type_info = (QkExprTypeInfo){QkExprType_Uint, 8};
    var_expr = inner_test_expr_kinds_and_types(QkExprNodeKind_Var, type_info);
    var = qk_expr_as_var(var_expr);
    loop_param = (QkLoopParamValue){.variable = var};
    body = qk_circuit_new(2, 0);
    exit_code = qk_circuit_for_loop_range(circuit, body, qubits, NULL, 0, 5, 1,
                                          QkLoopParamKind_Variable, &loop_param, NULL);
    if (exit_code != QkExitCode_LoopVariableMismatch) {
        fprintf(stderr,
                "Expected QkExitCode_LoopVariableMismatch for a loop variable the body does not "
                "declare, got exit code %u\n",
                exit_code);
        result = EqualityError;
        goto cleanup;
    }

    if (qk_circuit_num_instructions(circuit) != 0) {
        fprintf(stderr, "Expected no instruction to be appended, got %zu\n",
                qk_circuit_num_instructions(circuit));
        result = EqualityError;
    }

cleanup:

    inner_expr_free(var_expr);
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
    num_failed += RUN_TEST(test_for_nested_break_continue);
    num_failed += RUN_TEST(test_switch_case_on_register);
    num_failed += RUN_TEST(test_while_on_bit);
    num_failed += RUN_TEST(test_while_on_register);
    num_failed += RUN_TEST(test_while_on_expr);
    num_failed += RUN_TEST(test_switch_case_on_bit);
    num_failed += RUN_TEST(test_switch_case_on_expr);
    num_failed += RUN_TEST(test_for_loop_over_range);
    num_failed += RUN_TEST(test_while_on_register_large_condition);
    num_failed += RUN_TEST(test_build_for_loop_over_range);
    num_failed += RUN_TEST(test_build_for_loop_over_elements);
    num_failed += RUN_TEST(test_build_for_loop_invalid_input);
    num_failed += RUN_TEST(test_build_for_loop_invalid_loop_variable);
    num_failed += RUN_TEST(test_build_for_loop_without_loop_param);
    num_failed += RUN_TEST(test_build_for_loop_with_loop_variable);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
