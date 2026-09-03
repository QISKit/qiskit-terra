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

/**
 * Load a program with the default options, reporting any error message before discarding it.
 */
static QkCircuit *load(const char *program) {
    QkOpenQasm2Options options = qk_openqasm2_default_options();
    char *error = NULL;
    QkCircuit *circuit = qk_circuit_from_openqasm2(program, &options, &error);
    if (error != NULL) {
        fprintf(stderr, "parse error: %s\n", error);
        qk_str_free(error);
    }
    return circuit;
}

/**
 * A program using only `qelib1.inc` gates should match the equivalent hand-built circuit.
 */
static int test_bell_pair(void) {
    QkCircuit *parsed = load("OPENQASM 2.0;\n"
                             "include \"qelib1.inc\";\n"
                             "qreg q[2];\n"
                             "creg c[2];\n"
                             "h q[0];\n"
                             "cx q[0], q[1];\n"
                             "measure q -> c;\n");
    if (parsed == NULL)
        return RuntimeError;

    QkCircuit *expected = qk_circuit_new(2, 2);
    uint32_t h_qubits[1] = {0};
    qk_circuit_gate(expected, QkGate_H, h_qubits, NULL);
    uint32_t cx_qubits[2] = {0, 1};
    qk_circuit_gate(expected, QkGate_CX, cx_qubits, NULL);
    qk_circuit_measure(expected, 0, 0);
    qk_circuit_measure(expected, 1, 1);

    int result = compare_circuits(parsed, expected) ? Ok : EqualityError;
    qk_circuit_free(parsed);
    qk_circuit_free(expected);
    return result;
}

/**
 * An OpenQASM 2 `gate` statement becomes a single custom operation carrying the declared name,
 * rather than being inlined.
 */
static int test_defined_gate(void) {
    QkCircuit *parsed = load("include \"qelib1.inc\";\n"
                             "gate my_h q { h q; }\n"
                             "qreg q[1];\n"
                             "my_h q[0];\n");
    if (parsed == NULL)
        return RuntimeError;

    int result = Ok;
    if (qk_circuit_num_instructions(parsed) != 1) {
        fprintf(stderr, "%s: expected 1 instruction, got %zu\n", __func__,
                qk_circuit_num_instructions(parsed));
        result = EqualityError;
    } else {
        QkOpCounts counts = qk_circuit_count_ops(parsed);
        if (counts.len != 1 || strcmp(counts.data[0].name, "my_h") != 0 ||
            counts.data[0].count != 1) {
            fprintf(stderr, "%s: expected a single \"my_h\", got %zu distinct op(s)\n", __func__,
                    counts.len);
            result = EqualityError;
        }
        qk_opcounts_clear(&counts);
    }
    qk_circuit_free(parsed);
    return result;
}

/**
 * A malformed program returns a null pointer and describes the parse failure.
 */
static int test_parse_error(void) {
    QkOpenQasm2Options options = qk_openqasm2_default_options();
    char *error = NULL;
    QkCircuit *parsed = qk_circuit_from_openqasm2("qreg q[2]; not_a_gate q[0];", &options, &error);
    if (parsed != NULL) {
        fprintf(stderr, "%s: a circuit was returned despite the failure\n", __func__);
        qk_circuit_free(parsed);
        qk_str_free(error);
        return RuntimeError;
    }
    if (error == NULL) {
        fprintf(stderr, "%s: no error message was written\n", __func__);
        return NullptrError;
    }
    qk_str_free(error);
    return Ok;
}

/**
 * The caller can decline the error message, and strict mode is honoured.
 */
static int test_strict_requires_version(void) {
    QkOpenQasm2Options options = qk_openqasm2_default_options();
    options.strict = true;
    // No `OPENQASM 2.0;` header, which only strict mode objects to.  Passing `NULL` for `error`
    // must be accepted.
    QkCircuit *parsed = qk_circuit_from_openqasm2("qreg q[1];", &options, NULL);
    if (parsed != NULL) {
        fprintf(stderr, "%s: strict mode accepted a program with no version statement\n", __func__);
        qk_circuit_free(parsed);
        return EqualityError;
    }
    // The same program is fine without strict mode.
    parsed = load("qreg q[1];");
    if (parsed == NULL)
        return RuntimeError;
    qk_circuit_free(parsed);
    return Ok;
}

/**
 * A null `program` is rejected rather than dereferenced, and a null `options` means the defaults.
 */
static int test_null_arguments(void) {
    // A null `error` is allowed, and the description is simply discarded.
    if (qk_circuit_from_openqasm2(NULL, NULL, NULL) != NULL) {
        fprintf(stderr, "%s: a null program was not rejected\n", __func__);
        return EqualityError;
    }
    // When `error` is given, every failure path must describe itself, not just the parse errors.
    char *error = NULL;
    if (qk_circuit_from_openqasm2(NULL, NULL, &error) != NULL) {
        fprintf(stderr, "%s: a null program was not rejected\n", __func__);
        return EqualityError;
    }
    if (error == NULL) {
        fprintf(stderr, "%s: a null program left no description in `error`\n", __func__);
        return EqualityError;
    }
    qk_str_free(error);
    // Null `options` selects the defaults.
    QkCircuit *parsed = qk_circuit_from_openqasm2("qreg q[1];", NULL, NULL);
    if (parsed == NULL) {
        fprintf(stderr, "%s: null options were not treated as the defaults\n", __func__);
        return RuntimeError;
    }
    qk_circuit_free(parsed);
    return Ok;
}

int test_qasm2(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_bell_pair);
    num_failed += RUN_TEST(test_defined_gate);
    num_failed += RUN_TEST(test_parse_error);
    num_failed += RUN_TEST(test_strict_requires_version);
    num_failed += RUN_TEST(test_null_arguments);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);
    return num_failed;
}
