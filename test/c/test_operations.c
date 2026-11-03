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
#include <inttypes.h>
#include <math.h>
#include <qiskit.h>
#include <stdio.h>
#include <string.h>

static const char *FOO_NAME = "foo";

struct foo_gate {
    uint32_t num_qubits;
    uint32_t num_clbits;
    uint32_t num_params;
};

static const char *foo_name(const struct foo_gate *gate) {
    (void)gate; // Unused.
    return FOO_NAME;
}
static uint32_t foo_num_qubits(const struct foo_gate *gate) { return gate->num_qubits; }
static uint32_t foo_num_clbits(const struct foo_gate *gate) { return gate->num_clbits; }
static uint32_t foo_num_params(const void *gate) {
    struct foo_gate *self = (struct foo_gate *)gate;
    return self->num_params;
}
static bool foo_eq(const struct foo_gate *gate, const struct foo_gate *other) {
    return (gate->num_qubits == other->num_qubits && gate->num_clbits == other->num_clbits &&
            gate->num_params == other->num_params);
}
static struct foo_gate *foo_clone(const struct foo_gate *gate) {
    struct foo_gate *out = malloc(sizeof(*out));
    memcpy(out, gate, sizeof(*out));
    return out;
}
QkCircuit *foo_definition(const struct foo_gate *gate, QkParam **params) {
    (void)params;
    QkCircuit *def = qk_circuit_new(gate->num_qubits, gate->num_clbits);
    for (uint32_t i = 0; i < gate->num_qubits; i++) {
        qk_circuit_gate(def, QkGate_H, (uint32_t[1]){i}, NULL);
    }
    return def;
}

static QkVtableEntry foo_entries[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
    {.slot = QkCustomOpSlot_Eq, .ptr = foo_eq},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = foo_num_qubits},
    {.slot = QkCustomOpSlot_NumClbits, .ptr = foo_num_clbits},
    {.slot = QkCustomOpSlot_NumParams, .ptr = foo_num_params},
    {.slot = QkCustomOpSlot_Definition, .ptr = foo_definition},
    {.slot = QkCustomOpSlot_Clone, .ptr = foo_clone},
    {.slot = QkCustomOpSlot_Delete, .ptr = free},
    {.slot = -1, .ptr = NULL},
};

static QkVtableEntry incomplete_slots[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
    {.slot = -1, .ptr = NULL},
};

static QkVtableEntry complete_slots_with_null[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = NULL},
    {.slot = -1, .ptr = NULL},
};

struct fee_gate {
    char *label;
};

const char *fee_name(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return FOO_NAME;
}
uint32_t fee_num_qubits(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 2;
}
uint32_t fee_num_clbits(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 0;
}
uint32_t fee_num_params(const void *gate) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    return 1;
}
QkCircuit *fee_definition(const void *gate, const QkParam **params) {
    struct fee_gate *_self = (struct fee_gate *)gate;
    // Void pointer.
    (void)_self;
    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t hgate_args[1] = {0};
    uint32_t cxgate_args[2] = {0, 1};
    uint32_t rzgate_args[2] = {1};
    qk_circuit_gate(circuit, QkGate_H, hgate_args, NULL);
    qk_circuit_gate(circuit, QkGate_CX, cxgate_args, NULL);

    double params_fixed[1] = {qk_param_as_real(params[0])};
    qk_circuit_gate(circuit, QkGate_RZ, rzgate_args, params_fixed);

    return circuit;
}
const char *fee_label(const void *gate) {
    struct fee_gate *self = (struct fee_gate *)gate;
    return self->label;
}

static QkVtableEntry fee_entries[] = {
    {.slot = QkCustomOpSlot_Name, .ptr = fee_name},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = fee_num_qubits},
    {.slot = QkCustomOpSlot_NumClbits, .ptr = fee_num_clbits},
    {.slot = QkCustomOpSlot_NumParams, .ptr = fee_num_params},
    {.slot = QkCustomOpSlot_Definition, .ptr = fee_definition},
    {.slot = QkCustomOpSlot_Label, .ptr = fee_label},
    {.slot = -1, .ptr = NULL},
};

// A controlled operation exercising the optional ``num_ctrl_qubits`` slot (6).
const char *CTRL_NAME = "ctrl";
struct ctrl_gate {
    uint32_t num_qubits;
    uint32_t num_ctrl_qubits;
};

const char *ctrl_name(const void *gate) {
    (void)gate;
    return CTRL_NAME;
}
uint32_t ctrl_num_qubits(const void *gate) { return ((struct ctrl_gate *)gate)->num_qubits; }
uint32_t ctrl_num_clbits(const void *gate) {
    (void)gate;
    return 0;
}
uint32_t ctrl_num_params(const void *gate) {
    (void)gate;
    return 0;
}
bool ctrl_directive(const void *gate) {
    (void)gate;
    return false;
}
bool ctrl_is_unitary(const void *gate) {
    (void)gate;
    return true;
}
uint32_t ctrl_num_ctrl_qubits(const void *gate) {
    return ((struct ctrl_gate *)gate)->num_ctrl_qubits;
}

QkVtableEntry ctrl_entries[8] = {
    {.slot = QkCustomOpSlot_Name, .ptr = ctrl_name},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = ctrl_num_qubits},
    {.slot = QkCustomOpSlot_NumClbits, .ptr = ctrl_num_clbits},
    {.slot = QkCustomOpSlot_NumParams, .ptr = ctrl_num_params},
    {.slot = QkCustomOpSlot_Directive, .ptr = ctrl_directive},
    {.slot = QkCustomOpSlot_IsUnitary, .ptr = ctrl_is_unitary},
    {.slot = QkCustomOpSlot_NumCtrlQubits, .ptr = ctrl_num_ctrl_qubits},
    {.slot = -1, .ptr = NULL},
};

// A directive-style operation (barrier-like): it is a directive and is not
// unitary, unlike ``foo_gate``.
const char *DIR_NAME = "dir";
struct dir_gate {
    uint32_t num_qubits;
};

const char *dir_name(const void *gate) {
    (void)gate;
    return DIR_NAME;
}
uint32_t dir_num_qubits(const void *gate) { return ((struct dir_gate *)gate)->num_qubits; }
uint32_t dir_num_clbits(const void *gate) {
    (void)gate;
    return 0;
}
uint32_t dir_num_params(const void *gate) {
    (void)gate;
    return 0;
}
bool dir_directive(const void *gate) {
    (void)gate;
    return true;
}
bool dir_is_unitary(const void *gate) {
    (void)gate;
    return false;
}

QkVtableEntry dir_entries[7] = {
    {.slot = QkCustomOpSlot_Name, .ptr = dir_name},
    {.slot = QkCustomOpSlot_NumQubits, .ptr = dir_num_qubits},
    {.slot = QkCustomOpSlot_NumClbits, .ptr = dir_num_clbits},
    {.slot = QkCustomOpSlot_NumParams, .ptr = dir_num_params},
    {.slot = QkCustomOpSlot_Directive, .ptr = dir_directive},
    {.slot = QkCustomOpSlot_IsUnitary, .ptr = dir_is_unitary},
    {.slot = -1, .ptr = NULL},
};

/// Test adding a custom operation in the cicuit;
static int test_custom_operation_in_circuit(void) {
    int res = Ok;

    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 0,
        .num_params = 2,
    };
    struct foo_gate test_2q_op = {
        .num_qubits = 2,
        .num_clbits = 1,
        .num_params = 0,
    };

    // Initialize Vtable
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);

    if (foo_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOp *test_3q = qk_custom_operation_new(foo_clone(&test_3q_op), foo_vtable);
    QkCustomOp *test_2q_1c = qk_custom_operation_new(foo_clone(&test_2q_op), foo_vtable);

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};
    uint32_t qubits_2[2] = {1, 2};
    uint32_t clbits_2[1] = {1};
    QkParam *params[2] = {qk_param_from_double(3.14), qk_param_from_double(1.57)};

    qk_circuit_custom_operation(circuit, test_3q, qubits, NULL, params);
    qk_circuit_custom_operation(circuit, test_2q_1c, qubits_2, clbits_2, NULL);

    // Retrieve operation from circuit
    QkCircuitInstruction inst;
    qk_circuit_get_instruction(circuit, 0, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                FOO_NAME, inst.name);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_qubits != test_3q_op.num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_clbits != test_3q_op.num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_3q_op.num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_params, inst.num_params);
        res = EqualityError;
        goto cleanup;
    }

    // Retrieve operation from circuit
    qk_circuit_instruction_clear(&inst);
    qk_circuit_get_instruction(circuit, 1, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                FOO_NAME, inst.name);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_qubits != test_2q_op.num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                inst.name, test_2q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_clbits != test_2q_op.num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                inst.name, test_2q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_2q_op.num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                inst.name, test_2q_op.num_params, inst.num_params);
        res = EqualityError;
        goto cleanup;
    }

    QkOperationKind kind = qk_circuit_instruction_kind(circuit, 0);

    if (kind != 8) {
        fprintf(stderr, "Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8,
                kind);
        res = EqualityError;
        goto cleanup;
    }
    kind = qk_circuit_instruction_kind(circuit, 1);

    if (kind != 8) {
        fprintf(stderr, "Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8,
                kind);
        res = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_circuit_instruction_clear(&inst);
    qk_param_free(params[0]);
    qk_param_free(params[1]);
    qk_circuit_free(circuit);
    qk_custom_operation_vtable_free(foo_vtable);
exit:
    return res;
}

/// Test adding a custom operation in the cicuit;
static int test_custom_operation_in_dag(void) {
    int res = Ok;

    struct foo_gate test_1q_op = {
        .num_qubits = 1,
        .num_clbits = 0,
        .num_params = 1,
    };
    struct foo_gate test_3q_op = {
        .num_qubits = 3,
        .num_clbits = 1,
        .num_params = 0,
    };

    // Initialize Vtable
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);

    if (foo_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOp *test_1q = qk_custom_operation_new(foo_clone(&test_1q_op), foo_vtable);
    QkCustomOp *test_3q_1c = qk_custom_operation_new(foo_clone(&test_3q_op), foo_vtable);

    QkDag *circuit = qk_dag_new();
    QkQuantumRegister *qreg = qk_quantum_register_new(3, "qreg0");
    QkClassicalRegister *creg = qk_classical_register_new(1, "creg0");
    qk_dag_add_quantum_register(circuit, qreg);
    qk_dag_add_classical_register(circuit, creg);

    uint32_t qubits_1[1] = {0};
    uint32_t qubits_3[3] = {0, 1, 2};
    uint32_t clbits_1[1] = {0};
    QkParam *params[1] = {qk_param_from_double(3.14)};

    uint32_t ind1;
    uint32_t ind2;
    if (qk_dag_apply_custom_operation(circuit, test_1q, qubits_1, NULL, params, &ind1, false) !=
        QkExitCode_Success) {
        fprintf(stderr, "Unable to add operation 1q parametric custom operation to dag.");
        res = RuntimeError;
        goto cleanup;
    };
    if (qk_dag_apply_custom_operation(circuit, test_3q_1c, qubits_3, clbits_1, NULL, &ind2,
                                      false) != QkExitCode_Success) {
        fprintf(stderr, "Unable to add operation 3q custom operation to dag.");
        res = RuntimeError;
        goto cleanup;
    };

    // Retrieve operation from circuit
    QkCircuitInstruction inst;
    qk_dag_get_instruction(circuit, ind1, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                FOO_NAME, inst.name);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_qubits != test_1q_op.num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                inst.name, test_1q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_clbits != test_1q_op.num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                inst.name, test_1q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto cleanup;
    }
    if (inst.num_params != test_1q_op.num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                inst.name, test_1q_op.num_params, inst.num_params);
        res = EqualityError;
        goto inst_cleanup;
    }

    QkOperationKind kind = qk_dag_op_node_kind(circuit, ind2);

    if (kind != 8) {
        fprintf(stderr, "Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8,
                kind);
        res = EqualityError;
        goto cleanup;
    }

    // Retrieve operation from circuit
    qk_circuit_instruction_clear(&inst);
    qk_dag_get_instruction(circuit, ind2, &inst);

    if (strcmp(inst.name, FOO_NAME)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                FOO_NAME, inst.name);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_qubits != test_3q_op.num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_qubits, inst.num_qubits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_clbits != test_3q_op.num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_clbits, inst.num_clbits);
        res = EqualityError;
        goto inst_cleanup;
    }
    if (inst.num_params != test_3q_op.num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                inst.name, test_3q_op.num_params, inst.num_params);
        res = EqualityError;
        goto inst_cleanup;
    }

    kind = qk_dag_op_node_kind(circuit, ind2);

    if (kind != 8) {
        fprintf(stderr, "Retrieved incorrect kind for '%s'. Expected %u, got %u.\n", inst.name, 8,
                kind);
        res = EqualityError;
        goto cleanup;
    }
inst_cleanup:
    qk_circuit_instruction_clear(&inst);
cleanup:
    qk_quantum_register_free(qreg);
    qk_classical_register_free(creg);
    qk_param_free(params[0]);
    qk_dag_free(circuit);
    qk_custom_operation_vtable_free(foo_vtable);
exit:
    return res;
}

/**
 * Test passing an incomplete vtable returns NULL and exits gracefully.
 */
static int test_incomplete_vtable(void) {
    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(incomplete_slots);
    int result = Ok;
    if (vtable != NULL) {
        qk_custom_operation_vtable_free(vtable);
        result = EqualityError;
    }
    return result;
}

/**
 * Test passing a vtable with a NULL pointer in a functional slot returns NULL and exits gracefully.
 */
static int test_vtable_with_null(void) {
    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(complete_slots_with_null);
    int result = Ok;
    if (vtable != NULL) {
        qk_custom_operation_vtable_free(vtable);
        result = EqualityError;
    }
    return result;
}

static const char *leaky_name(void *_unused) {
    (void)_unused;
    return "name";
}
static uint32_t leaky_num_qubits(void *_unused) {
    (void)_unused;
    return 0;
}
static int leaky_delete_count;
static void leaky_delete(void *_unused) {
    (void)_unused;
    leaky_delete_count += 1;
}
static QkVtableEntry leaky_slots[] = {
    {QkCustomOpSlot_Name, 0, leaky_name},
    {QkCustomOpSlot_NumQubits, 0, leaky_num_qubits},
    {QkCustomOpSlot_Delete, 0, leaky_delete},
    {-1, 0, NULL},
};

static int test_dtor_calls(void) {
    int ret = Ok;
    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(leaky_slots);
    QkCustomOp *with_data = qk_custom_operation_new((void *)0xDEADBEEFDEADBEEF, vtable);
    QkCustomOp *no_data = qk_custom_operation_new(NULL, vtable);

    leaky_delete_count = 0;
    qk_custom_operation_free(no_data);
    if (leaky_delete_count) {
        ret = EqualityError;
        fprintf(stderr, "%s: dtor called %d times unexpectedly\n", __func__, leaky_delete_count);
        goto cleanup;
    }

    qk_custom_operation_free(with_data);
    if (leaky_delete_count != 1) {
        ret = EqualityError;
        fprintf(stderr, "%s: dtor called %d times, but expected 1\n", __func__, leaky_delete_count);
        goto cleanup;
    }
cleanup:
    qk_custom_operation_vtable_free(vtable);
    return ret;
}

void append_foo(QkCircuit *circuit, const QkCustomOpVtable *vtable, uint32_t num_qubits) {
    struct foo_gate *foo = malloc(sizeof(struct foo_gate));
    foo->num_qubits = num_qubits;
    foo->num_clbits = 0;
    foo->num_params = 0;

    QkCustomOp *foo_op = qk_custom_operation_new(foo, vtable);

    uint32_t *qubits = malloc(num_qubits * sizeof(uint32_t));
    for (uint32_t i = 0; i < num_qubits; i++)
        qubits[i] = i;

    qk_circuit_custom_operation(circuit, foo_op, qubits, NULL, NULL);
    free(qubits);
}

static int test_custom_op_transpile(void) {
    int res = Ok;
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);
    if (foo_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        res = NullptrError;
        goto exit;
    }

    QkCircuit *circuit = qk_circuit_new(10, 0);
    append_foo(circuit, foo_vtable, 3);
    append_foo(circuit, foo_vtable, 5);
    // TODO There currently is a bug in `transpile` which does not correctly unroll custom
    // gates with 2 qubits or less.
    // append_foo(circuit, foo_vtable, 2);

    QkTarget *target = qk_target_new(10);
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_H));
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_CX));

    QkTranspileResult result = {NULL, NULL};
    QkTranspileOptions options = qk_transpiler_default_options();
    options.optimization_level = 0;
    char *error = NULL;
    qk_transpile(circuit, target, &options, &result, &error);
    if (error != NULL) {
        fprintf(stderr, "Transpilation failed with\n%s\n", error);
        qk_str_free(error);
        goto cleanup;
    }
    QkOpCounts counts = qk_circuit_count_ops(result.circuit);
    if (counts.len != 1) {
        fprintf(stderr, "Wrong operation count after transpile.\n");
        res = EqualityError;
        goto cleanup_counts;
    }
    if (strcmp(counts.data[0].name, "h") != 0) {
        fprintf(stderr, "Unexpected gate (%s) after transpile.\n", counts.data[0].name);
        res = EqualityError;
        goto cleanup_counts;
    }
    if (counts.data[0].count != 8) {
        fprintf(stderr, "Expected 8 H gates, got %zu.\n", counts.data[0].count);
        res = EqualityError;
    }

cleanup_counts:
    qk_opcounts_clear(&counts);
cleanup:
    qk_target_free(target);
    qk_circuit_free(circuit);
    qk_circuit_free(result.circuit);
    qk_transpile_layout_free(result.layout);

    qk_custom_operation_vtable_free(foo_vtable);

exit:
    return res;
}

static int test_custom_operation_query(void) {
    int res = Ok;

    struct foo_gate test_3q_op = {
        .num_qubits = 3,
    };
    struct fee_gate test_2q_op = {
        .label = "fee",
    };

    // Initialize Vtable
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);
    // Initialize Vtable
    const QkCustomOpVtable *fee_vtable = qk_custom_operation_vtable_new(fee_entries);

    // Foo type_id
    uint64_t foo_type = (uint64_t)foo_vtable;
    // Fee type_id
    uint64_t fee_type = (uint64_t)fee_vtable;

    if (foo_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer for foo_gate.");
        res = NullptrError;
        goto exit;
    }
    if (fee_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer for fee_gate.");
        res = NullptrError;
        goto exit;
    }

    QkCustomOp *test_3q = qk_custom_operation_new(foo_clone(&test_3q_op), foo_vtable);
    QkCustomOp *test_2q_1c = qk_custom_operation_new(&test_2q_op, fee_vtable);

    QkCircuit *circuit = qk_circuit_new(3, 2);
    uint32_t qubits[3] = {0, 1, 2};
    uint32_t qubits_2[2] = {1, 2};
    uint32_t clbits_2[1] = {1};
    const QkParam *params[1] = {qk_param_from_double(3.14)};

    qk_circuit_custom_operation(circuit, test_3q, qubits, NULL, NULL);
    qk_circuit_custom_operation(circuit, test_2q_1c, qubits_2, clbits_2, (QkParam **)params);

    void *gates[2] = {(void *)&test_3q_op, (void *)&test_2q_op};
    if (qk_circuit_instruction_kind(circuit, 0) != QkOperationKind_Unknown) {
        res = RuntimeError;
        goto cleanup;
    }
    const QkCustomOp *op = qk_circuit_get_custom_operation(circuit, 0);
    void *gate = gates[0];

    char *retrieved_name = qk_custom_operation_name(op);
    const char *orig_name = foo_name(gate);
    if (strcmp(retrieved_name, orig_name)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                orig_name, retrieved_name);
        res = EqualityError;
        goto cleanup_name;
    }
    uint32_t retrieved_num_qubits = qk_custom_operation_num_qubits(op);
    uint32_t orig_num_qubits = foo_num_qubits(gate);
    if (retrieved_num_qubits != orig_num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                retrieved_name, orig_num_qubits, retrieved_num_qubits);
        res = EqualityError;
        goto cleanup_name;
    }
    uint32_t retrieved_num_clbits = qk_custom_operation_num_clbits(op);
    uint32_t orig_num_clbits = foo_num_clbits(gate);
    if (retrieved_num_clbits != orig_num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                retrieved_name, orig_num_clbits, retrieved_num_clbits);
        res = EqualityError;
        goto cleanup_name;
    }
    uint32_t retrieved_num_params = qk_custom_operation_num_params(op);
    uint32_t orig_num_params = foo_num_params(gate);
    if (retrieved_num_params != orig_num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                retrieved_name, orig_num_params, retrieved_num_params);
        res = EqualityError;
        goto cleanup_name;
    }

    if (qk_custom_operation_is_unitary(op) != true) {
        fprintf(stderr, "Unexpected non-unitary instruction for '%s'.\n", retrieved_name);
        res = EqualityError;
        goto cleanup_name;
    }

    QkCircuit *retrieved_definition_foo = qk_custom_operation_definition(op, NULL);
    QkCircuit *orig_definition_foo = foo_definition(gate, NULL);
    char *retrieved_drawing_foo = qk_circuit_draw(retrieved_definition_foo, NULL);
    char *orig_drawing_foo =
        qk_circuit_draw(qk_custom_operation_definition(op, (const QkParam **)params), NULL);
    if (strcmp(retrieved_drawing_foo, orig_drawing_foo) != 0) {
        fprintf(stderr, "Definitions are not simlar for '%s'.\n", retrieved_name);
        fprintf(stderr, "Expected.\n");
        print_circuit(retrieved_definition_foo);
        fprintf(stderr, "Got.\n");
        print_circuit(orig_definition_foo);
        res = EqualityError;
        goto cleanup_definitions;
    }

    uint64_t retreived_foo_type = qk_custom_operation_type_id(op);
    if (retreived_foo_type != foo_type) {
        fprintf(stderr,
                "Unexpected type retrieved for '%s'.\n, expected: %" PRIu64 ", got %" PRIu64 "",
                retrieved_name, foo_type, retreived_foo_type);
        res = EqualityError;
        goto cleanup_definitions;
    }

    // Should result in corrupted data if wrong.
    struct foo_gate *cast_foo = (struct foo_gate *)qk_custom_operation_raw(op);
    if (cast_foo->num_clbits != test_3q_op.num_clbits) {
        fprintf(stderr, "Unexpected num_clbits retrieved for '%s' pointer.\n, expected: %d, got %d",
                retrieved_name, test_3q_op.num_clbits, cast_foo->num_clbits);
        res = EqualityError;
        goto cleanup_definitions;
    }
    if (cast_foo->num_qubits != test_3q_op.num_qubits) {
        fprintf(stderr, "Unexpected num_qubits retrieved for '%s' pointer.\n, expected: %d, got %d",
                retrieved_name, test_3q_op.num_qubits, cast_foo->num_qubits);
        res = EqualityError;
        goto cleanup_definitions;
    }
    if (cast_foo->num_params != test_3q_op.num_params) {
        fprintf(stderr, "Unexpected num_params retrieved for '%s' pointer.\n, expected: %d, got %d",
                retrieved_name, test_3q_op.num_params, cast_foo->num_params);
        res = EqualityError;
        goto cleanup_definitions;
    }

    if (qk_circuit_instruction_kind(circuit, 1) != QkOperationKind_Unknown) {
        res = RuntimeError;
        goto cleanup_definitions;
    }
    op = qk_circuit_get_custom_operation(circuit, 1);
    gate = gates[1];

    char *retrieved_name_1 = qk_custom_operation_name(op);
    const char *orig_name_1 = fee_name(gate);
    if (strcmp(retrieved_name_1, orig_name_1)) {
        fprintf(stderr, "Retrieved incorrect instruction name. Expected '%s', got '%s'.\n",
                orig_name_1, retrieved_name_1);
        res = EqualityError;
        goto cleanup_name_1;
    }
    retrieved_num_qubits = qk_custom_operation_num_qubits(op);
    orig_num_qubits = fee_num_qubits(gate);
    if (retrieved_num_qubits != orig_num_qubits) {
        fprintf(stderr, "Retrieved incorrect num_qubits for '%s'. Expected %u, got %u.\n",
                retrieved_name_1, orig_num_qubits, retrieved_num_qubits);
        res = EqualityError;
        goto cleanup_name_1;
    }
    retrieved_num_clbits = qk_custom_operation_num_clbits(op);
    orig_num_clbits = fee_num_clbits(gate);
    if (retrieved_num_clbits != orig_num_clbits) {
        fprintf(stderr, "Retrieved incorrect num_clbits for '%s'. Expected %u, got %u.\n",
                retrieved_name_1, orig_num_clbits, retrieved_num_clbits);
        res = EqualityError;
        goto cleanup_name_1;
    }
    retrieved_num_params = qk_custom_operation_num_params(op);
    orig_num_params = fee_num_params(gate);
    if (retrieved_num_params != orig_num_params) {
        fprintf(stderr, "Retrieved incorrect num_params for '%s'. Expected %u, got %u.\n",
                retrieved_name_1, orig_num_params, retrieved_num_params);
        res = EqualityError;
        goto cleanup_name_1;
    }

    if (qk_custom_operation_is_unitary(op) != true) {
        fprintf(stderr, "Unexpected non-unitary instruction for '%s'.\n", retrieved_name_1);
        res = EqualityError;
        goto cleanup_name_1;
    }

    char *retrieved_label = qk_custom_operation_label(op);
    const char *orig_label = fee_label(gate);
    if (strcmp(retrieved_label, orig_label)) {
        fprintf(stderr, "Retrieved incorrect instruction label. Expected '%s', got '%s'.\n",
                orig_label, retrieved_label);
        res = EqualityError;
        goto cleanup_label;
    }

    // No definition was made for this operation, therefore it should be NULL
    QkCircuit *retrieved_definition = qk_custom_operation_definition(op, (const QkParam **)params);
    QkCircuit *orig_definition = fee_definition(gate, (const QkParam **)params);
    char *retrieved_drawing = qk_circuit_draw(retrieved_definition, NULL);
    char *orig_drawing =
        qk_circuit_draw(qk_custom_operation_definition(op, (const QkParam **)params), NULL);
    if (strcmp(retrieved_drawing, orig_drawing) != 0) {
        fprintf(stderr, "Definitions are not simlar for '%s'.\n", retrieved_name_1);
        fprintf(stderr, "Expected.\n");
        print_circuit(retrieved_definition);
        fprintf(stderr, "Got.\n");
        print_circuit(orig_definition);
        res = EqualityError;
        goto cleanup_definitions_1;
    }

    uint64_t retreived_fee_type = qk_custom_operation_type_id(op);
    if (retreived_fee_type != fee_type) {
        fprintf(stderr,
                "Unexpected type retrieved for '%s'.\n, expected: %" PRIu64 ", got %" PRIu64 "",
                retrieved_name, fee_type, retreived_fee_type);
        res = EqualityError;
        goto cleanup_definitions_1;
    }

    // Should result in corrupted data if wrong.
    struct fee_gate *cast_fee = (struct fee_gate *)qk_custom_operation_raw(op);
    if (strcmp(cast_fee->label, test_2q_op.label) != 0) {
        fprintf(stderr, "Unexpected label retrieved for '%s' pointer.\n, expected: %s, got %s",
                retrieved_name, test_2q_op.label, cast_fee->label);
        res = EqualityError;
        goto cleanup_definitions_1;
    }

cleanup_definitions_1:
    qk_circuit_free(retrieved_definition);
    qk_circuit_free(orig_definition);
cleanup_label:
    qk_str_free(retrieved_label);
cleanup_name_1:
    qk_str_free(retrieved_name_1);
cleanup_definitions:
    qk_circuit_free(retrieved_definition_foo);
    qk_circuit_free(orig_definition_foo);
cleanup_name:
    qk_str_free(retrieved_name);
cleanup:
    qk_circuit_free(circuit);
exit:
    return res;
}

/// Test that ``qk_custom_operation_eq`` correctly compares two custom
/// operations using a user-supplied ``eq`` slot. Two distinct instances holding
/// equal attributes should compare equal, while instances with differing
/// attributes should not.
static int test_custom_operation_eq(void) {
    int res = Ok;

    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(foo_entries);
    if (vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    // Two distinct instances with identical attributes...
    struct foo_gate a = {.num_qubits = 3, .num_clbits = 0, .num_params = 1};
    struct foo_gate b = {.num_qubits = 3, .num_clbits = 0, .num_params = 1};
    // ...and one with different attributes.
    struct foo_gate c = {.num_qubits = 2, .num_clbits = 1, .num_params = 0};

    QkCustomOp *op_a = qk_custom_operation_new(foo_clone(&a), vtable);
    QkCustomOp *op_b = qk_custom_operation_new(foo_clone(&b), vtable);
    QkCustomOp *op_c = qk_custom_operation_new(foo_clone(&c), vtable);

    if (!qk_custom_operation_eq(op_a, op_b)) {
        fprintf(stderr, "Expected operations with equal attributes to compare equal.\n");
        res = EqualityError;
        goto cleanup;
    }
    // Equality should be reflexive and symmetric.
    if (!qk_custom_operation_eq(op_a, op_a) || !qk_custom_operation_eq(op_b, op_a)) {
        fprintf(stderr, "Expected equality to be reflexive and symmetric.\n");
        res = EqualityError;
        goto cleanup;
    }
    if (qk_custom_operation_eq(op_a, op_c)) {
        fprintf(stderr, "Expected operations with different attributes to compare unequal.\n");
        res = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_custom_operation_free(op_a);
    qk_custom_operation_free(op_b);
    qk_custom_operation_free(op_c);
    return res;
}

/// Test ``qk_custom_operation_type_id``. Operations built from the same vtable
/// must share a type id, while operations built from different vtables must not.
static int test_custom_operation_type_id(void) {
    int res = Ok;

    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);
    const QkCustomOpVtable *fee_vtable = qk_custom_operation_vtable_new(fee_entries);
    if (foo_vtable == NULL || fee_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    struct foo_gate a = {.num_qubits = 3, .num_clbits = 0, .num_params = 0};
    struct foo_gate b = {.num_qubits = 1, .num_clbits = 0, .num_params = 0};
    struct fee_gate f = {.label = "fee"};

    QkCustomOp *op_a = qk_custom_operation_new(foo_clone(&a), foo_vtable);
    QkCustomOp *op_b = qk_custom_operation_new(foo_clone(&b), foo_vtable);
    QkCustomOp *op_f = qk_custom_operation_new(&f, fee_vtable);

    uint64_t id_a = qk_custom_operation_type_id(op_a);
    uint64_t id_b = qk_custom_operation_type_id(op_b);
    uint64_t id_f = qk_custom_operation_type_id(op_f);

    if (id_a != id_b) {
        fprintf(stderr,
                "Expected equal type ids for operations sharing a vtable. Got %" PRIu64
                " and %" PRIu64 ".\n",
                id_a, id_b);
        res = EqualityError;
        goto cleanup;
    }
    if (id_a == id_f) {
        fprintf(stderr, "Expected different type ids for operations with different vtables.\n");
        res = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_custom_operation_free(op_a);
    qk_custom_operation_free(op_b);
    qk_custom_operation_free(op_f);
    return res;
}

/// Test the optional ``num_ctrl_qubits`` slot. An operation that provides slot 6
/// should report its controlled-qubit count, while an operation that omits it
/// should fall back to the default of 0.
static int test_custom_operation_num_ctrl_qubits(void) {
    int res = Ok;

    const QkCustomOpVtable *ctrl_vtable = qk_custom_operation_vtable_new(ctrl_entries);
    const QkCustomOpVtable *foo_vtable = qk_custom_operation_vtable_new(foo_entries);
    if (ctrl_vtable == NULL || foo_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    struct ctrl_gate cg = {.num_qubits = 3, .num_ctrl_qubits = 2};
    struct foo_gate fg = {.num_qubits = 3, .num_clbits = 0, .num_params = 0};

    QkCustomOp *op_ctrl = qk_custom_operation_new(&cg, ctrl_vtable);
    QkCustomOp *op_foo = qk_custom_operation_new(foo_clone(&fg), foo_vtable);

    uint32_t ctrl = qk_custom_operation_num_ctrl_qubits(op_ctrl);
    if (ctrl != cg.num_ctrl_qubits) {
        fprintf(stderr, "Retrieved incorrect num_ctrl_qubits. Expected %u, got %u.\n",
                cg.num_ctrl_qubits, ctrl);
        res = EqualityError;
        goto cleanup;
    }
    // ``foo_gate`` does not provide slot 6, so it should default to 0.
    uint32_t foo_ctrl = qk_custom_operation_num_ctrl_qubits(op_foo);
    if (foo_ctrl != 0) {
        fprintf(stderr, "Expected default num_ctrl_qubits of 0, got %u.\n", foo_ctrl);
        res = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_custom_operation_free(op_ctrl);
    qk_custom_operation_free(op_foo);
    return res;
}

/// Test that a directive, non-unitary custom operation reports its ``directive``
/// and ``is_unitary`` flags correctly, both directly and after being retrieved
/// from a circuit.
static int test_custom_operation_directive(void) {
    int res = Ok;

    const QkCustomOpVtable *dir_vtable = qk_custom_operation_vtable_new(dir_entries);
    if (dir_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    struct dir_gate dg = {.num_qubits = 3};
    QkCustomOp *op = qk_custom_operation_new(&dg, dir_vtable);

    if (!qk_custom_operation_directive(op)) {
        fprintf(stderr, "Expected operation to be a directive.\n");
        qk_custom_operation_free(op);
        return EqualityError;
    }
    if (qk_custom_operation_is_unitary(op)) {
        fprintf(stderr, "Expected directive operation to be non-unitary.\n");
        qk_custom_operation_free(op);
        return EqualityError;
    }

    // The flags should survive a round-trip through a circuit.
    QkCircuit *circuit = qk_circuit_new(3, 0);
    uint32_t qubits[3] = {0, 1, 2};
    // NOTE: this consumes ``op``, so it must not be freed afterwards.
    qk_circuit_custom_operation(circuit, op, qubits, NULL, NULL);

    const QkCustomOp *retrieved = qk_circuit_get_custom_operation(circuit, 0);
    if (retrieved == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a custom operation.\n");
        res = NullptrError;
        goto cleanup;
    }
    if (!qk_custom_operation_directive(retrieved)) {
        fprintf(stderr, "Expected retrieved operation to be a directive.\n");
        res = EqualityError;
        goto cleanup;
    }
    if (qk_custom_operation_is_unitary(retrieved)) {
        fprintf(stderr, "Expected retrieved directive operation to be non-unitary.\n");
        res = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_circuit_free(circuit);
    return res;
}

/// Test that constructing a vtable that is missing a required slot fails and
/// returns a null pointer.
static int test_custom_vtable_missing_required_slot(void) {
    int res = Ok;

    // Omit the required ``num_params`` slot (3).
    QkVtableEntry bad_entries[6] = {
        {.slot = QkCustomOpSlot_Name, .ptr = foo_name},
        {.slot = -1, .ptr = NULL},
    };

    const QkCustomOpVtable *vtable = qk_custom_operation_vtable_new(bad_entries);
    if (vtable != NULL) {
        fprintf(stderr, "Expected a null vtable when a required slot is missing.\n");
        res = RuntimeError;
    }

    return res;
}

/// Test that omitting the optional slots (``num_ctrl_qubits``, ``label``,
/// ``definition``) yields their default values.
static int test_custom_operation_defaults(void) {
    int res = Ok;

    const QkCustomOpVtable *dir_vtable = qk_custom_operation_vtable_new(dir_entries);
    if (dir_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    struct dir_gate dir_g = {.num_qubits = 2};
    QkCustomOp *op = qk_custom_operation_new(&dir_g, dir_vtable);

    if (qk_custom_operation_num_ctrl_qubits(op) != 0) {
        fprintf(stderr, "Expected default num_ctrl_qubits of 0.\n");
        res = EqualityError;
        goto cleanup;
    }
    if (qk_custom_operation_label(op) != NULL) {
        fprintf(stderr, "Expected default label to be NULL.\n");
        res = EqualityError;
        goto cleanup;
    }
    if (qk_custom_operation_definition(op, NULL) != NULL) {
        fprintf(stderr, "Expected default definition to be NULL.\n");
        res = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_custom_operation_free(op);
    return res;
}

/// Test querying a custom operation back from a DAG via
/// ``qk_dag_get_custom_operation``, including its label and definition.
static int test_custom_operation_in_dag_query(void) {
    int res = Ok;

    const QkCustomOpVtable *fee_vtable = qk_custom_operation_vtable_new(fee_entries);
    if (fee_vtable == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a Vtable pointer.");
        return NullptrError;
    }

    struct fee_gate fg = {.label = "fee"};
    QkCustomOp *op = qk_custom_operation_new(&fg, fee_vtable);

    QkDag *dag = qk_dag_new();
    QkQuantumRegister *qreg = qk_quantum_register_new(2, "qreg0");
    qk_dag_add_quantum_register(dag, qreg);

    uint32_t qubits[2] = {0, 1};
    const QkParam *params[1] = {qk_param_from_double(3.14)};

    uint32_t idx;
    // NOTE: this consumes ``op``, so it must not be freed afterwards.
    if (qk_dag_apply_custom_operation(dag, op, qubits, NULL, (QkParam **)params, &idx, false) !=
        QkExitCode_Success) {
        fprintf(stderr, "Unable to add custom operation to dag.\n");
        res = RuntimeError;
        goto cleanup;
    }

    if (qk_dag_op_node_kind(dag, idx) != QkOperationKind_Unknown) {
        fprintf(stderr, "Expected a custom (unknown) operation kind.\n");
        res = EqualityError;
        goto cleanup;
    }

    const QkCustomOp *retrieved = qk_dag_get_custom_operation(dag, idx);
    if (retrieved == NULL) {
        fprintf(stderr, "Retrieved a Null pointer instead of a custom operation.\n");
        res = NullptrError;
        goto cleanup;
    }

    char *retrieved_name = qk_custom_operation_name(retrieved);
    if (strcmp(retrieved_name, fee_name(&fg))) {
        fprintf(stderr, "Retrieved incorrect name. Expected '%s', got '%s'.\n", fee_name(&fg),
                retrieved_name);
        res = EqualityError;
        goto cleanup_name;
    }
    if (qk_custom_operation_num_qubits(retrieved) != fee_num_qubits(&fg)) {
        fprintf(stderr, "Retrieved incorrect num_qubits. Expected %u, got %u.\n",
                fee_num_qubits(&fg), qk_custom_operation_num_qubits(retrieved));
        res = EqualityError;
        goto cleanup_name;
    }
    char *retrieved_label = qk_custom_operation_label(retrieved);
    if (strcmp(retrieved_label, fee_label(&fg))) {
        fprintf(stderr, "Retrieved incorrect label. Expected '%s', got '%s'.\n", fee_label(&fg),
                retrieved_label);
        res = EqualityError;
        goto cleanup_label;
    }

    // Compare the retrieved definition against the reference definition.
    QkCircuit *retrieved_definition =
        qk_custom_operation_definition(retrieved, (const QkParam **)params);
    QkCircuit *orig_definition = fee_definition(&fg, (const QkParam **)params);
    if (retrieved_definition == NULL) {
        fprintf(stderr, "Expected a non-null definition.\n");
        res = NullptrError;
        goto cleanup_definitions;
    }
    char *retrieved_drawing = qk_circuit_draw(retrieved_definition, NULL);
    char *orig_drawing = qk_circuit_draw(orig_definition, NULL);
    if (strcmp(retrieved_drawing, orig_drawing) != 0) {
        fprintf(stderr, "Definitions are not similar for '%s'.\n", retrieved_name);
        fprintf(stderr, "Expected.\n");
        print_circuit(orig_definition);
        fprintf(stderr, "Got.\n");
        print_circuit(retrieved_definition);
        res = EqualityError;
    }

cleanup_definitions:
    qk_circuit_free(retrieved_definition);
    qk_circuit_free(orig_definition);
cleanup_label:
    qk_str_free(retrieved_label);
cleanup_name:
    qk_str_free(retrieved_name);
cleanup:
    qk_quantum_register_free(qreg);
    qk_dag_free(dag);
    return res;
}

int test_operations(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_custom_operation_in_circuit);
    num_failed += RUN_TEST(test_custom_operation_in_dag);
    num_failed += RUN_TEST(test_incomplete_vtable);
    num_failed += RUN_TEST(test_vtable_with_null);
    num_failed += RUN_TEST(test_dtor_calls);
    num_failed += RUN_TEST(test_custom_op_transpile);
    num_failed += RUN_TEST(test_custom_operation_query);
    num_failed += RUN_TEST(test_custom_operation_eq);
    num_failed += RUN_TEST(test_custom_operation_type_id);
    num_failed += RUN_TEST(test_custom_operation_num_ctrl_qubits);
    num_failed += RUN_TEST(test_custom_operation_directive);
    num_failed += RUN_TEST(test_custom_vtable_missing_required_slot);
    num_failed += RUN_TEST(test_custom_operation_defaults);
    num_failed += RUN_TEST(test_custom_operation_in_dag_query);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}
