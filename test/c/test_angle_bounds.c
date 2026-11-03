#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <qiskit.h>

static int test_basic_wrapangles_linkage(void) {
    printf("Running basic WrapAngles linkage test...\n");

    QkCircuit *qc = qk_circuit_new(0, 0);

    QkTarget *target = qk_target_new(0);
    if (!target) {
        fprintf(stderr, "Failed to allocate Target\n");
        qk_circuit_free(qc);
        return 1;
    }

    QkWrapAngleRegistry* registry = qk_wrap_angle_registry_new();
    if (!registry) {
        fprintf(stderr, "Failed to allocate WrapAngleRegistry\n");
        qk_target_free(target);
        qk_circuit_free(qc);
        return 1;
    }

    int rc = qk_transpiler_pass_standalone_wrap_angles(qc, target, registry);
    printf("Return code: %d\n", rc);

    qk_wrap_angle_registry_free(registry);
    qk_target_free(target);
    qk_circuit_free(qc);

    if (rc != 0) {
        fprintf(stderr, "Expected 0, got %d\n", rc);
        return 1;
    }

    return Ok;
}

int test_angle_bounds(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_basic_wrapangles_linkage);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);
    return num_failed;
}
