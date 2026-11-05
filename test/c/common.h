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

#include <complex.h>
#include <qiskit.h>
#include <stdio.h>

// These constants are technically extensions beyond the raw C11 spec, but we want to run our tests
// using the most restrictive version of the C spec we require.  It's more convenient to just
// arrange for them to be defined ourselves.
#if !defined(M_PI)
// Prevent MSVC redefining our constants.
#define _MATH_DEFINES_DEFINED
#define M_E 2.7182818284590452354
#define M_LOG2E 1.4426950408889634074
#define M_LOG10E 0.43429448190325182765
#define M_LN2 0.69314718055994530942
#define M_LN10 2.30258509299404568402
#define M_PI 3.14159265358979323846
#define M_PI_2 1.57079632679489661923
#define M_PI_4 0.78539816339744830962
#define M_1_PI 0.31830988618379067154
#define M_2_PI 0.63661977236758134308
#define M_2_SQRTPI 1.12837916709551257390
#define M_SQRT2 1.41421356237309504880
#define M_SQRT1_2 0.70710678118654752440
#endif // !defined(M_PI)

// An enumeration of test results. These should be returned by test functions to
// indicate what kind of error occurred. This will be used to produce more
// helpful messages for the developer running the test suite.
enum TestResult {
    Ok,
    EqualityError,
    RuntimeError,
    NullptrError, // an unexpected null pointer
};

// A macro for running a test function. This calls the run function below with
// the provided function and its name.
#define RUN_TEST(f) run(#f, f)

// A function to run a test function of a given name. This function will also
// post-process the returned `TestResult` to product a minimal info message for
// the developer running the test suite.
int run(const char *name, int (*test_function)(void));

bool compare_circuits(const QkCircuit *res, const QkCircuit *expected);
void print_circuit(const QkCircuit *qc);
