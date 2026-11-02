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
#include <qiskit.h>
#include <stdio.h>
#include <string.h>

#define UNUSED_VARIABLE(x) (void)(x)

/// A struct to keep the configuration of the RemoveIdentity pass.
typedef struct {
    QkTarget *target;
} RemoveIdentity;

/// The execution function for RemoveIdentity.
QkCircuit *run_remove_identity(RemoveIdentity *self, QkCircuit *ir, QkPassContext *context,
                               QkCompilationError **error) {
    UNUSED_VARIABLE(context);
    UNUSED_VARIABLE(error);

    qk_transpiler_pass_standalone_remove_identity_equivalent(ir, self->target, 1.0);
    return ir;
}

static const QkVtableEntry remove_identity_slots[] = {
    {.slot = QkPassSlot_RunOwned, .flags = 0, .ptr = (void *)&run_remove_identity},
    {.slot = -1, .flags = 0, .ptr = NULL},
};

/// The execution function for a circuit-to-dag pass.
QkDag *run_circuit_to_dag(void *self, QkCircuit *ir, QkPassContext *context,
                          QkCompilationError **error) {
    UNUSED_VARIABLE(self);
    UNUSED_VARIABLE(context);
    UNUSED_VARIABLE(error);

    QkDag *dag = qk_circuit_to_dag(ir);
    qk_circuit_free(ir);
    return dag;
}

static const QkVtableEntry circuit_to_dag_slots[] = {
    {.slot = QkPassSlot_RunOwned, .flags = 0, .ptr = (void *)&run_circuit_to_dag},
    {.slot = -1, .flags = 0, .ptr = NULL}};

/// A pass on `QkCircuit*` that always returns an error message "task successfully failed!"
QkCircuit *always_error(void *self, QkCircuit *ir, QkPassContext *context,
                        QkCompilationError **error) {
    UNUSED_VARIABLE(self);
    UNUSED_VARIABLE(context);

    qk_circuit_free(ir); // we are responsible to free the IR
    *error = qk_compilation_error_new("task successfully failed!");
    return NULL;
}

/// A logger to keep track of delete calls.
typedef struct {
    size_t num_deletes;
} DeleteLogger;

/// A custom integer-based IR.
///
/// This holds an array of uint32_t and is assumed to flip the bits at where there is
/// a 1-bit. For example:
///     [1, 3, 6]
/// would flip apply X gates on
///     [..X, .XX, XX.]
///
/// We're defining a set of passes on this IR, e.g. one that cancels adjacent integers
/// and one to remove specific integers.
///
/// Importantly, this struct is contains data that must be freed (the `*integers` pointer),
/// and keeps a logger to count how often the deconstructor (`free_flips`) is called.
typedef struct {
    size_t capacity;
    size_t len;
    uint32_t *integers;
    DeleteLogger *logger;
} Flips;

Flips *new_flips(size_t capacity, DeleteLogger *logger) {
    Flips *flips = malloc(sizeof(Flips));
    flips->capacity = capacity;
    flips->len = 0;
    flips->integers = malloc(capacity * sizeof(uint32_t));
    flips->logger = logger;
    return flips;
}

/// Free the content *and* the flips pointer.
void free_flips(Flips *flips) {
    if (flips->logger != NULL)
        flips->logger->num_deletes++;

    free(flips->integers);
    free(flips);
}

static const QkVtableEntry flip_methods[] = {
    {.slot = QkIrSlot_Delete, .flags = 0, .ptr = (void *)&free_flips},
    {.slot = -1, .flags = 0, .ptr = NULL},
};

/// Methods for the IR.
int apply_flip(Flips *flips, uint32_t flip) {
    if (flips->capacity <= flips->len)
        return 1; // cannot append

    // i++ increases _after_ reading the value, hence this is correct
    flips->integers[flips->len++] = flip;
    return 0;
}

size_t num_flips(Flips *flips) {
    size_t n = 0;
    for (size_t i = 0; i < flips->len; i++) {
        uint32_t integer = flips->integers[i];
        for (uint32_t b = 0; b < 32; b++) {
            n += (integer >> b) & 1u;
        }
    }
    return n;
}

/// Passes for the flips IR.
Flips *inverse_cancellation(void *self, Flips *ir, QkPassContext *context,
                            QkCompilationError **error) {
    UNUSED_VARIABLE(self);
    UNUSED_VARIABLE(context);
    UNUSED_VARIABLE(error);

    if (ir->len < 2)
        return ir;

    size_t write_index = 0;
    size_t read_index = 0;
    while (read_index < ir->len - 1) {
        // If the next one is not the same, write the current one.
        if (ir->integers[read_index] != ir->integers[read_index + 1]) {
            ir->integers[write_index++] = ir->integers[read_index];
            read_index++;
        } else {
            // .. if they were the same, increase the read index by 2
            read_index += 2;
        }
    }
    // Handle the last element: if it was skipped, then read_index equals ir->len,
    // if not we are one element short.
    if (read_index == ir->len - 1) {
        ir->integers[write_index++] = ir->integers[read_index];
    }
    ir->len = write_index;

    return ir;
}

static const QkVtableEntry inverse_cancellation_slots[] = {
    {.slot = QkPassSlot_RunOwned, .flags = 0, .ptr = (void *)&inverse_cancellation},
    {.slot = -1, .flags = 0, .ptr = NULL},
};

/// A pass to pop specified integers from the Flips IR.
/// This pass needs freeing the data it holds upon destruction.
typedef struct {
    size_t len;
    uint32_t *to_pop;
    DeleteLogger *logger;
} PopFlips;

void delete_pops(PopFlips *this) {
    if (this->logger != NULL) {
        this->logger->num_deletes++;
    }
    free(this->to_pop);
}

Flips *pop_flips(PopFlips *self, Flips *flips, QkPassContext *context, QkCompilationError **error) {
    UNUSED_VARIABLE(context);
    UNUSED_VARIABLE(error);

    size_t write_index = 0;
    for (size_t read_index = 0; read_index < flips->len; read_index++) {
        bool skip = false;
        for (size_t i = 0; i < self->len; i++) {
            if (flips->integers[read_index] == self->to_pop[i]) {
                skip = true;
                break;
            }
        }

        if (!skip) {
            flips->integers[write_index] = flips->integers[read_index];
            write_index++;
        }
    }
    flips->len = write_index;
    return flips;
}

static const QkVtableEntry pop_slots[] = {
    {.slot = QkPassSlot_RunOwned, .flags = 0, .ptr = (void *)&pop_flips},
    {.slot = QkPassSlot_Delete, .flags = 0, .ptr = (void *)&delete_pops},
    {.slot = -1, .flags = 0, .ptr = NULL},
};

/**
 * Test running a single RemoveIdentity pass on a circuit.
 *
 * This is a simple test case on a single builtin IR, without lowering or custom destructors.
 * The pass manager is run twice to check it is re-usable.
 */
static int test_circuit_ir(void) {
    QkIrHandle *ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkPassVtable *vtable = qk_pass_vtable_new("remove_identity", ir, ir, remove_identity_slots);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity this = {target};
    QkPass *pass = qk_pass_new((void *)&this, vtable);

    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;

    if (qk_passmanager_push_pass(pm, pass) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    QkCircuit *circuit1 = qk_circuit_new(10, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit1, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit1, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit1, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit1, QkGate_H, q0, NULL);

    // note: as the passmanager is set up, it takes ownership of the input IR, which no longer
    // needs to be freed -- only the output IR must be freed
    QkCircuit *out = (QkCircuit *)qk_passmanager_run_simple(pm, (void *)circuit1, ir, ir, NULL);

    QkOpCounts counts = qk_circuit_count_ops(out);
    qk_circuit_free(out);

    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "h") == 0) {
            if (count.count != 2) {
                fprintf(stderr, "Expected 2 H gates, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_counts;
            }
        } else if (strcmp(count.name, "rz") == 0) {
            if (count.count != 1) {
                fprintf(stderr, "Expected 1 RZ gate, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_counts;
            }
        } else {
            fprintf(stderr, "Unexpected gate.\n");
            result = EqualityError;
            goto cleanup_counts;
        }
    }
    qk_opcounts_clear(&counts);

    QkCircuit *circuit2 = qk_circuit_new(3, 0);
    uint32_t q01[2] = {0, 1};
    qk_circuit_gate(circuit2, QkGate_CX, q01, NULL);
    qk_circuit_gate(circuit2, QkGate_RY, q0, almost_zero);
    qk_circuit_gate(circuit2, QkGate_CX, q01, NULL);

    out = (QkCircuit *)qk_passmanager_run_simple(pm, (void *)circuit2, ir, ir, NULL);
    counts = qk_circuit_count_ops(out);
    qk_circuit_free(out);

    for (size_t i = 0; i < counts.len; i++) {
        QkOpCount count = counts.data[i];
        if (strcmp(count.name, "cx") == 0) {
            if (count.count != 2) {
                fprintf(stderr, "Expected 2 CX gates, found %zu\n", count.count);
                result = EqualityError;
                goto cleanup_counts;
            }
        } else {
            fprintf(stderr, "Unexpected gate.\n");
            result = EqualityError;
            goto cleanup_counts;
        }
    }

cleanup_counts:
    qk_opcounts_clear(&counts);
cleanup:
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_pass_vtable_free(vtable);
    qk_ir_handle_free(ir);

    return result;
}

/**
 * Test a pass manager lowering from circuit to dag.
 *
 * This tests the IR lowering between two builtin IRs.
 */
static int test_lowering(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);

    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    QkPassVtable *remove_identity_vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkPass *remove_identity = qk_pass_new((void *)&remove_identity_config, remove_identity_vtable);

    QkPassVtable *circuit_to_dag_vtable =
        qk_pass_vtable_new("circuit_to_dag", circuit_ir, dag_ir, circuit_to_dag_slots);
    QkPass *circuit_to_dag = qk_pass_new(NULL, circuit_to_dag_vtable);

    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;
    if (qk_passmanager_push_pass(pm, remove_identity) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing circuit pass.\n");
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_passmanager_push_pass(pm, circuit_to_dag) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing circuit->dag lowering pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    QkCircuit *circuit = qk_circuit_new(10, 0);
    uint32_t q0[1] = {0};
    double almost_zero[1] = {1e-20};
    double nonzero[1] = {1.23};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_RX, q0, almost_zero);
    qk_circuit_gate(circuit, QkGate_RZ, q0, nonzero);
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);

    QkCompilationError *error = NULL;
    QkDag *out =
        (QkDag *)qk_passmanager_run_simple(pm, (void *)circuit, circuit_ir, dag_ir, &error);
    if (error != NULL) {
        fprintf(stderr, "Failed running pass.\n");
        result = RuntimeError;
        qk_compilation_error_free(error);
        goto cleanup;
    }

    // iterate over the DAG and ensure it matches the expected ops
    size_t num_ops = qk_dag_num_op_nodes(out);
    uint32_t *op_indices = malloc(num_ops * sizeof(*op_indices));
    qk_dag_topological_op_nodes(out, op_indices);

    for (size_t i = 0; i < num_ops; i++) {
        QkCircuitInstruction inst;
        qk_dag_get_instruction(out, op_indices[i], &inst);

        if (i == 0 || i == 2) {
            if (strcmp(inst.name, "h") != 0) {
                fprintf(stderr, "Expected h at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto dag_cleanup;
            }
        } else if (i == 1) {
            if (strcmp(inst.name, "rz") != 0) {
                fprintf(stderr, "Expected rz at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto dag_cleanup;
            }
        } else {
            fprintf(stderr, "Unexpected number of operations.\n");
            result = EqualityError;
            qk_circuit_instruction_clear(&inst);
            goto dag_cleanup;
        }
        qk_circuit_instruction_clear(&inst);
    }

dag_cleanup:
    free(op_indices);
    qk_dag_free(out);
cleanup:
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_pass_vtable_free(circuit_to_dag_vtable);
    qk_pass_vtable_free(remove_identity_vtable);
    qk_ir_handle_free(circuit_ir);
    qk_ir_handle_free(dag_ir);

    return result;
}

static int test_custom_ir(void) {
    DeleteLogger logger = {0};
    Flips *program = new_flips(5, &logger);
    apply_flip(program, 1);
    apply_flip(program, 1);
    apply_flip(program, 2);
    apply_flip(program, 3);

    size_t flipped = num_flips(program);

    if (flipped != 5) {
        fprintf(stderr, "Wrong number of initial bitflips, expected 5 got %zu\n", flipped);
        free_flips(program);
        return EqualityError;
    }

    QkPassManager *pm = qk_passmanager_new();

    QkIrHandle *ir = qk_ir_handle_new("flips", flip_methods);
    QkPassVtable *cancellation_vtable =
        qk_pass_vtable_new("cancellation", ir, ir, inverse_cancellation_slots);
    QkPass *cancellation = qk_pass_new(NULL, cancellation_vtable);

    int result = Ok;
    if (qk_passmanager_push_pass(pm, cancellation) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        free_flips(program);
        goto cleanup;
    }

    Flips *out = (Flips *)qk_passmanager_run_simple(pm, program, ir, ir, NULL);
    flipped = num_flips(out);
    size_t num_deletes = logger.num_deletes; // check the number of delete-calls before freeing
    free_flips(out);

    if (num_deletes != 0) {
        fprintf(stderr, "Unwarrented delete of the IR!\n");
        result = RuntimeError;
        goto cleanup;
    }

    if (flipped != 3) {
        fprintf(stderr, "Wrong number of bitflips, expected 3 got %zu\n", flipped);
        result = EqualityError;
    }

cleanup:
    qk_passmanager_free(pm);
    qk_ir_handle_free(ir);
    qk_pass_vtable_free(cancellation_vtable);

    return result;
}

/**
 * Test running an empty PM acts as identity.
 */
static int test_empty_pm(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);

    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t q0[1] = {0};
    uint32_t q01[2] = {0, 1};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_CX, q01, NULL);

    QkPassManager *pm = qk_passmanager_new();
    QkCircuit *out =
        (QkCircuit *)qk_passmanager_run_simple(pm, circuit, circuit_ir, circuit_ir, NULL);

    const size_t num_ops = 2;
    int result = Ok;
    if (qk_circuit_num_instructions(out) != num_ops) {
        fprintf(stderr, "Expected 2 instructions, got %zu\n", qk_circuit_num_instructions(out));
        result = EqualityError;
        goto cleanup;
    }

    for (size_t i = 0; i < num_ops; i++) {
        QkCircuitInstruction inst;
        qk_circuit_get_instruction(out, i, &inst);

        if (i == 0) {
            if (strcmp(inst.name, "h") != 0) {
                fprintf(stderr, "Expected h at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto cleanup;
            }
        } else {
            if (strcmp(inst.name, "cx") != 0) {
                fprintf(stderr, "Expected cx at %zu, but got %s\n", i, inst.name);
                result = EqualityError;
                qk_circuit_instruction_clear(&inst);
                goto cleanup;
            }
        }
        qk_circuit_instruction_clear(&inst);
    }

cleanup:
    qk_circuit_free(out);
    qk_passmanager_free(pm);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test the pipeline being incoherent.
 */
static int test_invalid_pipeline(void) {
    // A pass on Flips->Flips IR.
    QkIrHandle *flip_ir = qk_ir_handle_new("flips", flip_methods);

    QkPassVtable *cancellation_vtable =
        qk_pass_vtable_new("cancellation", flip_ir, flip_ir, inverse_cancellation_slots);
    QkPass *flip_pass = qk_pass_new(NULL, cancellation_vtable);

    // A pass on Circuit->Circuit IR.
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    QkPassVtable *remove_identity_vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkPass *circuit_pass = qk_pass_new((void *)&remove_identity_config, remove_identity_vtable);

    int result = Ok;
    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_pass(pm, flip_pass) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    // Now try pushing the circuit pass onto the flip pass. This should error.
    if (qk_passmanager_push_pass(pm, circuit_pass) != QkExitCode_IncompatibleTypes) {
        fprintf(stderr, "Expected QkExitCode_IncompatibleTypes.\n");
        result = EqualityError;
    }

cleanup:
    qk_passmanager_free(pm);
    qk_target_free(target);
    qk_pass_vtable_free(cancellation_vtable);
    qk_pass_vtable_free(remove_identity_vtable);
    qk_ir_handle_free(flip_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test the input IR not matching the pipeline type.
 *
 * This checks that the input IR is properly freed and `NULL` is returned when the pipeline cannot
 * run.
 */
static int test_mismatching_input(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *flip_ir = qk_ir_handle_new("flip", flip_methods);

    QkPassVtable *vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity this = {target};
    QkPass *pass = qk_pass_new((void *)(&this), vtable);

    // This PM now has a pipeline built on QkCircuit
    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;
    if (qk_passmanager_push_pass(pm, pass) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    // .. and now we call it on the Flip IR. This also needs to call its destructor.
    DeleteLogger logger = {0};
    Flips *program = new_flips(5, &logger);
    apply_flip(program, 1);

    QkCompilationError *error = NULL;
    Flips *out = (Flips *)qk_passmanager_run_simple(pm, (void *)program, flip_ir, flip_ir, &error);

    if (out != NULL) {
        fprintf(stderr, "Expected out pointer to be NULL, but it is not.\n");
        result = EqualityError;
        free_flips(out);
        goto cleanup;
    }

    if (error == NULL) {
        fprintf(stderr, "Expected error to be written, but the pointer is NULL.\n");
        result = EqualityError;
        goto cleanup;
    } else {
        char *error_msg = qk_compilation_error_str(error);
        if (strcmp(error_msg, "failed to cast to expected input type") != 0) {
            fprintf(stderr, "Wrong error message: %s\n", error_msg);
            result = EqualityError;
        }
        qk_str_free(error_msg);
        qk_compilation_error_free(error);
    }

    // at this point we know `out` is NULL (as expected) and no longer need to free it,
    // and the error has been freed. Now we can verify the input IR was freed, too.
    if (logger.num_deletes != 1) {
        fprintf(stderr, "Input IR has not been freed despite faulty pipeline.\n");
        result = RuntimeError;
        goto cleanup;
    }

cleanup:
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_pass_vtable_free(vtable);
    qk_ir_handle_free(flip_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test the ouput IR not matching the pipeline type.
 */
static int test_mismatching_output(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *flip_ir = qk_ir_handle_new("flip", flip_methods);

    // This PM now has a pipeline built on Flips IR
    QkPassVtable *cancellation_vtable =
        qk_pass_vtable_new("cancellation", flip_ir, flip_ir, inverse_cancellation_slots);
    QkPass *cancellation = qk_pass_new(NULL, cancellation_vtable);

    QkPassManager *pm = qk_passmanager_new();
    int result = Ok;
    if (qk_passmanager_push_pass(pm, cancellation) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    DeleteLogger logger = {0};
    Flips *program = new_flips(5, &logger);
    apply_flip(program, 1);
    apply_flip(program, 1);
    apply_flip(program, 2);
    apply_flip(program, 3);

    // .. and now we call the PM but we give the wrong output IR. The pipeline produces
    // a Flip IR but we request `circuit_ir`.
    QkCompilationError *error = NULL;
    QkCircuit *out =
        (QkCircuit *)qk_passmanager_run_simple(pm, (void *)program, flip_ir, circuit_ir, &error);

    if (out != NULL) {
        fprintf(stderr, "Expected out pointer to be NULL, but it is not.\n");
        result = EqualityError;
        goto cleanup;
    }

    // at this point we know `out` is NULL (as expected) and no longer need to free it
    // -- but the input IR should've been freed, so we check this here
    if (logger.num_deletes != 1) {
        fprintf(stderr, "Input IR has not been freed despite faulty return type.\n");
        result = RuntimeError;
        goto cleanup;
    }

    if (error == NULL) {
        fprintf(stderr, "Expected error to be written, but the pointer is NULL.\n");
        result = EqualityError;
        goto cleanup;
    } else {
        char *error_msg = qk_compilation_error_str(error);
        if (strcmp(error_msg, "declared output IR type does not match the pipeline result") != 0) {
            fprintf(stderr, "Wrong error message: %s\n", error_msg);
            result = EqualityError;
        }
        qk_str_free(error_msg);
        qk_compilation_error_free(error);
    }

cleanup:
    qk_passmanager_free(pm);
    qk_pass_vtable_free(cancellation_vtable);
    qk_ir_handle_free(flip_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test the pass' deconstructor is correctly called exactly once when the PM is freed.
 */
static int test_pass_deconstructor_after_run(void) {
    QkIrHandle *ir = qk_ir_handle_new("flip", flip_methods);

    QkPassVtable *pop_vtable = qk_pass_vtable_new("pop", ir, ir, pop_slots);
    size_t len = 2;
    uint32_t *to_pop = malloc(len * sizeof(uint32_t));
    to_pop[0] = 3;
    to_pop[1] = 4;

    DeleteLogger pass_logger = {0};
    PopFlips pops = {len, to_pop, &pass_logger};
    QkPass *pop = qk_pass_new((void *)&pops, pop_vtable);

    int result = Ok;
    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_pass(pm, pop) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        qk_passmanager_free(pm);
        goto cleanup;
    }

    DeleteLogger ir_logger = {0};
    Flips *program = new_flips(5, &ir_logger);
    apply_flip(program, 1);
    apply_flip(program, 1);
    apply_flip(program, 2);
    apply_flip(program, 3);

    Flips *out = (Flips *)qk_passmanager_run_simple(pm, program, ir, ir, NULL);
    if (out == NULL) {
        fprintf(stderr, "Failed running PM.\n");
        result = RuntimeError;
        qk_passmanager_free(pm);
        goto cleanup;
    }

    size_t flipped = num_flips(out);
    size_t num_ir_deletes = ir_logger.num_deletes;
    free_flips(out);

    if (flipped != 3) {
        fprintf(stderr, "Wrong number of final flips, expected 3, got %zu\n", flipped);
        result = EqualityError;
        qk_passmanager_free(pm);
        goto cleanup;
    }
    if (num_ir_deletes != 0) {
        fprintf(stderr, "Unwarrented delete of the IR!\n");
        result = RuntimeError;
        qk_passmanager_free(pm);
        goto cleanup;
    }
    if (pass_logger.num_deletes != 0) {
        fprintf(stderr, "Unwarrented delete of the pass!\n");
        result = RuntimeError;
        qk_passmanager_free(pm);
        goto cleanup;
    }

    qk_passmanager_free(pm);
    if (pass_logger.num_deletes != 1) {
        fprintf(stderr, "Pass not correctly deletes upon pass manager free.\n");
        result = RuntimeError;
    }
cleanup:
    qk_pass_vtable_free(pop_vtable);
    qk_ir_handle_free(ir);

    return result;
}

/**
 * Test the pass' deconstructor is called when failing to push a pass.
 */
static int test_pass_deconstructor_on_failure(void) {
    QkIrHandle *flip_ir = qk_ir_handle_new("flip", flip_methods);
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);

    // A pass on circuit IR.
    QkPassVtable *vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkTarget *target = qk_target_new(10);
    RemoveIdentity this = {target};
    QkPass *circuit_pass = qk_pass_new((void *)&this, vtable);

    // A pass on Flips IR.
    QkPassVtable *pop_vtable = qk_pass_vtable_new("pop", flip_ir, flip_ir, pop_slots);

    int result = Ok;
    QkPassManager *pm = qk_passmanager_new();
    if (qk_passmanager_push_pass(pm, circuit_pass) != QkExitCode_Success) {
        fprintf(stderr, "Failed pushing pass.\n");
        result = RuntimeError;
        goto cleanup;
    }

    size_t len = 2;
    uint32_t *to_pop = malloc(len * sizeof(uint32_t));
    to_pop[0] = 3;
    to_pop[1] = 4;
    DeleteLogger pass_logger = {0};
    PopFlips pops = {len, to_pop, &pass_logger};
    QkPass *flip_pass = qk_pass_new((void *)&pops, pop_vtable);

    // Now try pushing the circuit pass onto the flip pass. This should error.
    // In case this succeeds, we already know from other tests that the pass' deconstructor
    // is correctly called, so we don't have any cleanup to do.
    if (qk_passmanager_push_pass(pm, flip_pass) != QkExitCode_IncompatibleTypes) {
        fprintf(stderr, "Expected QkExitCode_IncompatibleTypes.\n");
        result = EqualityError;
        goto cleanup;
    }

    // .. since the PM took ownership, the pass' deconstructor should've been called.
    if (pass_logger.num_deletes != 1) {
        fprintf(stderr, "Pass deconstructor not called on failed push.\n");
        result = RuntimeError;
    }

cleanup:
    qk_target_free(target);
    qk_passmanager_free(pm);
    qk_pass_vtable_free(vtable);
    qk_pass_vtable_free(pop_vtable);
    qk_ir_handle_free(flip_ir);
    qk_ir_handle_free(circuit_ir);

    return result;
}

/**
 * Test a pipeline where a pass returns an error.
 */
static int test_failing_pass(void) {
    QkIrHandle *circuit_ir = qk_ir_handle_builtin(QkIrBuiltin_Circuit);
    QkIrHandle *dag_ir = qk_ir_handle_builtin(QkIrBuiltin_Dag);

    // Pass 1: a working circuit IR pass
    QkTarget *target = qk_target_new(10);
    RemoveIdentity remove_identity_config = {target};
    QkPassVtable *remove_identity_vtable =
        qk_pass_vtable_new("remove_identity", circuit_ir, circuit_ir, remove_identity_slots);
    QkPass *remove_identity = qk_pass_new((void *)&remove_identity_config, remove_identity_vtable);

    // Pass 2: the failing pass
    const QkVtableEntry failing_slots[2] = {
        {.slot = QkPassSlot_RunOwned, .flags = 0, .ptr = (void *)&always_error},
        {.slot = -1, .flags = 0, .ptr = NULL}};
    QkPassVtable *failing_vtable =
        qk_pass_vtable_new("failing", circuit_ir, circuit_ir, failing_slots);
    QkPass *failing_pass = qk_pass_new(NULL, failing_vtable);

    // Pass 3: a working circuit->dag IR pass
    QkPassVtable *circuit_to_dag_vtable =
        qk_pass_vtable_new("circuit_to_dag", circuit_ir, dag_ir, circuit_to_dag_slots);
    QkPass *circuit_to_dag = qk_pass_new(NULL, circuit_to_dag_vtable);

    QkPassManager *pm = qk_passmanager_new();
    qk_passmanager_push_pass(pm, remove_identity);
    qk_passmanager_push_pass(pm, failing_pass);
    qk_passmanager_push_pass(pm, circuit_to_dag);

    QkCircuit *circuit = qk_circuit_new(2, 0);
    uint32_t q0[1] = {0};
    uint32_t q01[2] = {0, 1};
    qk_circuit_gate(circuit, QkGate_H, q0, NULL);
    qk_circuit_gate(circuit, QkGate_CX, q01, NULL);

    // running the pass should return NULL and set the error message
    QkCompilationError *error = NULL;
    QkDag *out = (QkDag *)qk_passmanager_run_simple(pm, circuit, circuit_ir, dag_ir, &error);

    int result = Ok;
    if (out != NULL) {
        fprintf(stderr, "Expected NULL pointer.\n");
        result = EqualityError;
        goto cleanup;
    }

    if (error == NULL) {
        fprintf(stderr, "Expected error to be set, but it is NULL\n");
        result = EqualityError;
        goto cleanup;
    } else {
        char *error_msg = qk_compilation_error_str(error);
        if (strcmp(error_msg, "task successfully failed!") != 0) {
            fprintf(stderr, "Wrong error message: %s\n", error_msg);
            result = EqualityError;
        }
        qk_str_free(error_msg);
        qk_compilation_error_free(error);
    }

cleanup:
    qk_passmanager_free(pm);
    qk_target_free(target);
    qk_ir_handle_free(circuit_ir);
    qk_ir_handle_free(dag_ir);
    qk_pass_vtable_free(remove_identity_vtable);
    qk_pass_vtable_free(circuit_to_dag_vtable);
    qk_pass_vtable_free(failing_vtable);

    return result;
}

int test_passmanager(void) {
    int num_failed = 0;

    num_failed += RUN_TEST(test_circuit_ir);
    num_failed += RUN_TEST(test_lowering);
    num_failed += RUN_TEST(test_custom_ir);
    num_failed += RUN_TEST(test_empty_pm);
    num_failed += RUN_TEST(test_invalid_pipeline);
    num_failed += RUN_TEST(test_mismatching_input);
    num_failed += RUN_TEST(test_mismatching_output);
    num_failed += RUN_TEST(test_pass_deconstructor_after_run);
    num_failed += RUN_TEST(test_pass_deconstructor_on_failure);
    num_failed += RUN_TEST(test_failing_pass);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests (passmanager): %i\n", num_failed);

    return num_failed;
}
