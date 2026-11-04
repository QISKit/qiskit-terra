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
#include <complex.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef _MSC_VER
typedef _Dcomplex ComplexDouble;
ComplexDouble make_complex(double real, double imag) { return (ComplexDouble){real, imag}; }
#else
typedef double complex ComplexDouble;
ComplexDouble make_complex(double real, double imag) { return real + I * imag; }
#endif

/**
 * Test converting a native number to QkComplex64.
 */
static int test_from_native(void) {
    const double real = -1;
    const double imag = 2;
    ComplexDouble native = make_complex(real, imag);
    QkComplex64 value = qk_complex64_from_native(&native);

    if (value.re != real || value.im != imag) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test converting QkComplex64 to native number.
 */
static int test_to_native(void) {
    const double real = 1.65;
    const double imag = -5.21;
    QkComplex64 value = {real, imag};
    ComplexDouble native = qk_complex64_to_native(&value);

    if (creal(native) != real || cimag(native) != imag) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test roundtrips.
 */
int test_qkcomplex_roundtrip(void) {
    const double real = 1.003;
    const double imag = 2.31;
    QkComplex64 value = {real, imag};
    ComplexDouble native = qk_complex64_to_native(&value);
    QkComplex64 value_roundtrip = qk_complex64_from_native(&native);

    if (value.re != value_roundtrip.re || value.im != value_roundtrip.im) {
        return EqualityError;
    }
    return Ok;
}

/**
 * Test roundtrips.
 */
static int test_native_roundtrip(void) {
    const double real = 1.003;
    const double imag = 2.31;
    ComplexDouble native = make_complex(real, imag);
    QkComplex64 value = qk_complex64_from_native(&native);
    ComplexDouble native_roundtrip = qk_complex64_to_native(&value);

    if (creal(native) != creal(native_roundtrip) || cimag(native) != cimag(native_roundtrip)) {
        return EqualityError;
    }
    return Ok;
}

int test_complex(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_from_native);
    num_failed += RUN_TEST(test_to_native);
    num_failed += RUN_TEST(test_qkcomplex_roundtrip);
    num_failed += RUN_TEST(test_native_roundtrip);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
