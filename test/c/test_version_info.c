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
#include <stdlib.h>
#include <string.h>

/**
 * Build the version a string, based on the version numbers.
 */
static char *build_version_string(void) {
    const size_t suffix_len = 16;
    char *suffix = calloc(suffix_len, sizeof(char));
    switch (QISKIT_RELEASE_LEVEL) {
    case QISKIT_RELEASE_LEVEL_DEV:
        snprintf(suffix, suffix_len, "-dev");
        break;
    case QISKIT_RELEASE_LEVEL_BETA:
        snprintf(suffix, suffix_len, "-beta%u", QISKIT_RELEASE_SERIAL);
        break;
    case QISKIT_RELEASE_LEVEL_RC:
        snprintf(suffix, suffix_len, "-rc%u", QISKIT_RELEASE_SERIAL);
        break;
    default:
        // no suffix
        break;
    }

    const size_t version_len = 32;
    char *version = calloc(version_len, sizeof(char));
    snprintf(version, version_len, "%u.%u.%u%s", QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR,
             QISKIT_VERSION_PATCH, suffix);
    free(suffix);
    return version;
}

/**
 * Test the string version.
 */
static int test_version(void) {
    char *ref = build_version_string();
    int result;
    if (strcmp(ref, QISKIT_VERSION) == 0)
        result = Ok;
    else {
        fprintf(stderr, "Qiskit version (%s) didn't match the expectation (%s)", QISKIT_VERSION,
                ref);
        result = EqualityError;
    }

    free(ref);
    return result;
}

/**
 * Test the version macro and HEX version.
 */
static int test_version_macros(void) {
    if (QISKIT_VERSION_MAJOR < 0 || QISKIT_VERSION_MINOR < 0 || QISKIT_VERSION_PATCH < 0) {
        return EqualityError;
    }
    if (QISKIT_GET_VERSION_HEX(QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR, QISKIT_VERSION_PATCH,
                               QISKIT_RELEASE_LEVEL, QISKIT_RELEASE_SERIAL) != QISKIT_VERSION_HEX) {
        fprintf(stderr, "QISKIT_VERSION_HEX does not match QISKIT_GET_VERSION_HEX\n");
        return EqualityError;
    }
    return Ok;
}

/**
 * Test the version in the header matches the version returned from the library.
 */
static int test_header_vs_lib(void) {
    if (qk_api_version() != QISKIT_VERSION_HEX) {
        fprintf(stderr, "%s: QISKIT_VERSION_HEX (%x) does not match qk_api_version() (%x)\n",
                __func__, QISKIT_VERSION_HEX, qk_api_version());
        return EqualityError;
    }
    return Ok;
}

int test_version_info(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_version);
    num_failed += RUN_TEST(test_version_macros);
    num_failed += RUN_TEST(test_header_vs_lib);

    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);
    fflush(stderr);
    return num_failed;
}
