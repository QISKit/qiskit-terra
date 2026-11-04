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
#include <math.h>
#include <qiskit.h>
#include <stdio.h>
#include <string.h>

static int test_all_to_all(void) {
    int res = Ok;
    uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    QkGate gates[3] = {QkGate_RZ, QkGate_SX, QkGate_CX};
    uint32_t num_qargs[3] = {1, 1, 2};
    for (size_t i = 0; i < sizeof(gates) / sizeof(gates[0]); i++) {
        QkTargetEntry *entry = qk_target_entry_new(gates[i]);
        qk_target_entry_add_property(entry, NULL, num_qargs[i], NAN, NAN);
        qk_target_add_instruction(target, entry);
    }
    QkNeighbors neighbors;
    if (!qk_neighbors_from_target(target, &neighbors)) {
        fprintf(stderr, "%s: incorrect return from `qk_neighbors_from_target`\n", __func__);
        res = EqualityError;
        goto cleanup;
    }
    if (!qk_neighbors_is_all_to_all(&neighbors)) {
        fprintf(stderr, "%s: incorrect return from `qk_neighbors_is_all_to_all`\n", __func__);
        res = EqualityError;
        goto cleanup;
    }
    if (neighbors.neighbors || neighbors.partition) {
        fprintf(stderr, "%s: pointers should be null but are (%p, %p)\n", __func__,
                (void *)neighbors.neighbors, (void *)neighbors.partition);
        res = EqualityError;
        goto cleanup;
    }
    if (neighbors.num_qubits != num_qubits) {
        fprintf(stderr, "%s: incorrect num_qubits: %u\n", __func__, neighbors.num_qubits);
        res = EqualityError;
        goto cleanup;
    }
    // This isn't necessary to call since there's no allocations, but let's call it for coverage.
    qk_neighbors_clear(&neighbors);
    if (neighbors.neighbors || neighbors.partition) {
        fprintf(stderr, "%s: `qk_neighbors_free` wrote non-null pointers (%p, %p)\n", __func__,
                (void *)neighbors.neighbors, (void *)neighbors.partition);
        res = EqualityError;
        goto cleanup;
    }
cleanup:
    qk_target_free(target);
    return res;
}

static int test_multiq(void) {
    int res = Ok;
    uint32_t num_qubits = 5;
    QkTarget *target = qk_target_new(num_qubits);
    // Simple line of CZ gates.
    QkTargetEntry *cz_entry = qk_target_entry_new(QkGate_CZ);
    for (uint32_t q = 1; q < num_qubits; q++) {
        uint32_t qargs[2] = {q - 1, q};
        qk_target_entry_add_property(cz_entry, qargs, 2, NAN, NAN);
    }
    qk_target_add_instruction(target, cz_entry);
    // Now let's add a CCX that links {0, 1, 4} so we can check there's no link in the output.
    QkTargetEntry *ccx_entry = qk_target_entry_new(QkGate_CCX);
    uint32_t qargs[3] = {0, 1, 4};
    qk_target_entry_add_property(ccx_entry, qargs, 3, NAN, NAN);
    qk_target_add_instruction(target, ccx_entry);

    QkNeighbors neighbors;
    if (qk_neighbors_from_target(target, &neighbors)) {
        fprintf(stderr, "%s: incorrect return from `qk_neighbors_from_target`\n", __func__);
        res = EqualityError;
        goto cleanup_target;
    }
    if (!neighbors.neighbors || !neighbors.partition) {
        fprintf(stderr, "%s: pointers should be non-null but are (%p, %p)\n", __func__,
                (void *)neighbors.neighbors, (void *)neighbors.partition);
        res = NullptrError;
        goto cleanup_target;
    }
    if (qk_neighbors_is_all_to_all(&neighbors)) {
        fprintf(stderr, "%s: incorrect return from `qk_neighbors_is_all_to_all`\n", __func__);
        res = EqualityError;
        goto cleanup_neighbors;
    }
    if (neighbors.num_qubits != num_qubits) {
        fprintf(stderr, "%s: incorrect num_qubits: %u\n", __func__, neighbors.num_qubits);
        res = EqualityError;
        goto cleanup_neighbors;
    }
    uint32_t expected_neighbors[] = {1, 0, 2, 1, 3, 2, 4, 3};
    size_t expected_partition[] = {0, 1, 3, 5, 7, 8};
    if (memcmp(expected_partition, neighbors.partition, sizeof(expected_partition)) ||
        memcmp(expected_neighbors, neighbors.neighbors, sizeof(expected_neighbors))) {
        fprintf(stderr, "%s: incorrect data in neighbors\n", __func__);
        res = EqualityError;
        goto cleanup_neighbors;
    }
cleanup_neighbors:
    qk_neighbors_clear(&neighbors);
    if (neighbors.neighbors || neighbors.partition) {
        fprintf(stderr, "%s: `qk_neighbors_free` wrote non-null pointers (%p, %p)\n", __func__,
                (void *)neighbors.neighbors, (void *)neighbors.partition);
        res = EqualityError;
        goto cleanup_target;
    }
cleanup_target:
    qk_target_free(target);
    return res;
}

int test_neighbors(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_all_to_all);
    num_failed += RUN_TEST(test_multiq);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    return num_failed;
}
