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

#include <qiskit.h>
#include <stdint.h>
#include <stdio.h>

QkCircuit *ghz(int n, int layers) {
    QkCircuit *qc = qk_circuit_new(n, n);
    qk_circuit_gate(qc, QkGate_H, (uint32_t[]){0}, NULL);

    for (int i = 0; i < layers; i++) {
        for (int i = 0; i < qk_circuit_num_qubits(qc) - 1; i++) {
            qk_circuit_gate(qc, QkGate_CX, (uint32_t[]){i, i + 1}, NULL);
        }
    }

    for (int i = 0; i < qk_circuit_num_qubits(qc); i++) {
        qk_circuit_measure(qc, i, i);
    }
    return qc;
}

QkObs *create_observable(int num_qubits) {
    uint64_t num_terms = 2; // we have 2 terms: |01><01|, -1 * |+-><+-|
    uint64_t num_bits = 4;  // we have 4 non-identity bits: 0, 1, +, -
    QkComplex64 coeffs = {1, -1};
    QkBitTerm bits[4] = {QkBitTerm_Zero, QkBitTerm_One, QkBitTerm_Plus, QkBitTerm_Minus};

    uint32_t indices[4] = {0, 1, 98, 99}; // <-- e.g. {1, 0, 99, 98} would be invalid
    size_t boundaries[3] = {0, 2, 4};
    QkObs *obs = qk_obs_new(num_qubits, num_terms, num_bits, &coeffs, bits, indices, boundaries);
    return obs;
}

QkTarget *create_target(int num_qubits) {
    // https://github.com/Qiskit/qiskit/blob/main/test/c/test_transpiler.c
    QkTarget *target = qk_target_new(num_qubits);

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

    QkTargetEntry *entry = qk_target_entry_new_measure();
    for (uint32_t i = 0; i < num_qubits; i++) {
        // Measure is a single qubit instruction
        uint32_t qargs[1] = {i};
        qk_target_entry_add_property(entry, qargs, 1, 1.928e-10, 7.9829e-11);
    }
    qk_target_add_instruction(target, entry);

    return target;
}

int num_gates(QkCircuit *qc) {
    int total_num_gates = 0;
    QkOpCounts counts = qk_circuit_count_ops(qc);
    for (int g = 0; g < counts.len; g++) {
        total_num_gates += counts.data[g].count;
    }
    qk_opcounts_clear(&counts);
    return total_num_gates;
}

int main(int argc, char *argv[]) {
    uint32_t num_qubits = 200;

    // create observable
    QkObs *observable = create_observable(num_qubits);

    // Create target
    QkTarget *target = create_target(num_qubits);

    // Create QuantumCircuit
    QkCircuit *ghz_circuit = ghz(num_qubits, 10000);

    printf("Num gates (pre-transpilation): %i\n", num_gates(ghz_circuit));

    // Transpile circuit
    QkTranspileResult transpile_result = {NULL, NULL};
    char *error = NULL;
    QkTranspileOptions options = qk_transpiler_default_options();
    options.seed = 42;
    int result_code = qk_transpile(ghz_circuit, target, &options, &transpile_result, &error);

    printf("Num gates (post-transpilation): %i\n", num_gates(transpile_result.circuit));

    // Free resources
    qk_obs_free(observable);
    qk_target_free(target);
    qk_circuit_free(ghz_circuit);
    qk_circuit_free(transpile_result.circuit);
    qk_transpile_layout_free(transpile_result.layout);

    return 0;
}
