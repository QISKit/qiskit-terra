// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
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

// A function to run a test function of a given name. This function will also
// post-process the returned `TestResult` to product a minimal info message for
// the developer running the test suite.
int run(const char *name, int (*test_function)(void)) {
    // TODO: we could consider to change the return value of our test functions
    // to be a struct containing the integer return value and a custom error
    // message which could then be included below.
    int result = test_function();
    int did_fail = 1;
    char *msg;
    if (result == Ok) {
        did_fail = 0;
        msg = "Ok";
    } else if (result == EqualityError) {
        msg = "FAILED with an EqualityError";
    } else if (result == RuntimeError) {
        msg = "FAILED with a RuntimeError";
    } else {
        msg = "FAILED with unknown error";
    }
    fprintf(stderr, "--- %-30s: %s\n", name, msg);
    fflush(stderr);

    return did_fail;
}

void print_circuit(const QkCircuit *qc) {
    size_t num_instructions = qk_circuit_num_instructions(qc);
    QkCircuitInstruction inst;
    for (size_t i = 0; i < num_instructions; i++) {
        qk_circuit_get_instruction(qc, i, &inst);
        fprintf(stderr, "%s: qubits: (", inst.name);
        for (uint32_t j = 0; j < inst.num_qubits; j++) {
            fprintf(stderr, "%d,", inst.qubits[j]);
        }
        fprintf(stderr, ")");
        uint32_t num_clbits = inst.num_clbits;
        if (num_clbits > 0) {
            fprintf(stderr, ", clbits: (");
            for (uint32_t j = 0; j < num_clbits; j++) {
                fprintf(stderr, "%d,", inst.clbits[j]);
            }
            fprintf(stderr, ")");
        }
        fprintf(stderr, "\n");
        qk_circuit_instruction_clear(&inst);
    }
}

bool compare_circuits(const QkCircuit *res, const QkCircuit *expected) {
    if (qk_circuit_num_instructions(res) != qk_circuit_num_instructions(expected)) {
        fprintf(stderr, "Number of instructions in circuit is mismatched\n");
        return false;
    }
    QkCircuitInstruction res_inst;
    QkCircuitInstruction expected_inst;
    for (size_t i = 0; i < qk_circuit_num_instructions(res); i++) {
        qk_circuit_get_instruction(res, i, &res_inst);
        qk_circuit_get_instruction(expected, i, &expected_inst);
        int result = strcmp(res_inst.name, expected_inst.name);
        if (result != 0) {
            fprintf(stderr, "Gate %zu have different gates %s was found and expected %s\n", i,
                    res_inst.name, expected_inst.name);
            qk_circuit_instruction_clear(&res_inst);
            qk_circuit_instruction_clear(&expected_inst);
            return false;
        }
        if (res_inst.num_qubits != expected_inst.num_qubits) {
            fprintf(stderr,
                    "Gate %zu have different number of qubits %d was found and expected %d\n", i,
                    res_inst.num_qubits, expected_inst.num_qubits);
            qk_circuit_instruction_clear(&res_inst);
            qk_circuit_instruction_clear(&expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst.num_qubits; j++) {
            if (res_inst.qubits[j] != expected_inst.qubits[j]) {
                fprintf(stderr,
                        "Qubit %d for gate %zu are different %d was found and expected %d\n", j, i,
                        res_inst.qubits[j], expected_inst.qubits[j]);
                fprintf(stderr, "Expected circuit instructions:\n");
                print_circuit(expected);
                fprintf(stderr, "Result circuit:\n");
                print_circuit(res);
                qk_circuit_instruction_clear(&res_inst);
                qk_circuit_instruction_clear(&expected_inst);
                return false;
            }
        }
        if (res_inst.num_clbits != expected_inst.num_clbits) {
            fprintf(stderr,
                    "Gate %zu have different number of clbits %d was found and expected %d\n", i,
                    res_inst.num_clbits, expected_inst.num_clbits);
            qk_circuit_instruction_clear(&res_inst);
            qk_circuit_instruction_clear(&expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst.num_clbits; j++) {
            if (res_inst.clbits[j] != expected_inst.clbits[j]) {
                fprintf(stderr,
                        "Clbit %d for gate %zu are different %d was found and expected %d\n", j, i,
                        res_inst.clbits[j], expected_inst.clbits[j]);
                qk_circuit_instruction_clear(&res_inst);
                qk_circuit_instruction_clear(&expected_inst);
                return false;
            }
        }
        if (res_inst.num_params != expected_inst.num_params) {
            fprintf(stderr,
                    "Gate %zu have different number of params %d was found and expected %d\n", i,
                    res_inst.num_params, expected_inst.num_params);
            qk_circuit_instruction_clear(&res_inst);
            qk_circuit_instruction_clear(&expected_inst);
            return false;
        }
        for (uint32_t j = 0; j < res_inst.num_params; j++) {
            if (!qk_param_equal(res_inst.params[j], expected_inst.params[j])) {
                char *res_str = qk_param_str(res_inst.params[j]);
                char *expected_str = qk_param_str(expected_inst.params[j]);

                fprintf(stderr,
                        "Parameter %d for gate %zu are different %s was found and expected %s\n", j,
                        i, res_str, expected_str);
                qk_str_free(res_str);
                qk_str_free(expected_str);
                qk_circuit_instruction_clear(&res_inst);
                qk_circuit_instruction_clear(&expected_inst);
                return false;
            }
        }
        qk_circuit_instruction_clear(&res_inst);
        qk_circuit_instruction_clear(&expected_inst);
    }
    return true;
}
