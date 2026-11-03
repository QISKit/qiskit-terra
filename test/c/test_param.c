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
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/**
 * Test creating a new free symbol and check the name.
 */
static int test_param_new(void) {
    QkParam *p = qk_param_new_symbol("a");
    char *str = qk_param_str(p);
    qk_param_free(p);

    if (strcmp(str, "a") != 0) {
        fprintf(stderr, "The parameter is not a\n");
        qk_str_free(str);
        return EqualityError;
    }
    qk_str_free(str);
    return Ok;
}

/**
 * Test creating a new symbol from values.
 */
static int test_param_new_values(void) {
    QkParam *real = qk_param_from_double(0.1);
    if (real == NULL) {
        return EqualityError;
    }

    QkComplex64 c = {0.4, 1.4};
    QkParam *cmplx = qk_param_from_complex(c);
    if (cmplx == NULL) {
        qk_param_free(real);
        return EqualityError;
    }

    qk_param_free(real);
    qk_param_free(cmplx);
    return Ok;
}

/**
 * Test casting to real values.
 */
static int test_param_to_real(void) {
    QkParam *x = qk_param_new_symbol("x");
    QkParam *cmplx = qk_param_from_complex((QkComplex64){1.0, 2.0});
    QkParam *val = qk_param_from_double(10.0);

    double x_out = qk_param_as_real(x);
    double cmplx_out = qk_param_as_real(cmplx);
    double val_out = qk_param_as_real(val);

    QkParamKind x_kind = qk_param_kind(x);
    QkParamKind cmplx_kind = qk_param_kind(cmplx);
    QkParamKind val_kind = qk_param_kind(val);

    qk_param_free(x);
    qk_param_free(cmplx);
    qk_param_free(val);

    if (x_kind != QkParamKind_ParameterExpression ||
        cmplx_kind != QkParamKind_ParameterExpression || val_kind != QkParamKind_Real) {
        fprintf(stderr, "Unexpected success/failure in qk_param_kind.\n");
        return EqualityError;
    }

    if (!isnan(x_out) || isnan(cmplx_out) || isnan(val_out)) {
        fprintf(stderr, "Unexpected success/failure in qk_param_as_real.\n");
        return EqualityError;
    }

    if (fabs(val_out - 10.0) > 1e-10) {
        fprintf(stderr, "Unexpected extracted value in qk_param_as_real.\n");
        return EqualityError;
    }

    // qk_param_as_real extracts the real part of a complex value
    if (fabs(cmplx_out - 1.0) > 1e-10) {
        fprintf(stderr, "Unexpected extracted value in qk_param_as_real.\n");
        return EqualityError;
    }

    return Ok;
}

/**
 * Test calling all binary operations and verify their string representation.
 */
static int test_param_binary_ops(void) {
    QkParam *a = qk_param_new_symbol("a");
    QkParam *b = qk_param_new_symbol("b");
    QkParam *ret = qk_param_zero();
    char *str;
    int result = Ok;

    // add
    if (qk_param_add(ret, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "a + b") != 0) {
        fprintf(stderr, "qk_param_add is not a + b\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // sub
    if (qk_param_sub(ret, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "a - b") != 0) {
        fprintf(stderr, "qk_param_sub is not a - b\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // mul
    if (qk_param_mul(ret, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "a*b") != 0) {
        fprintf(stderr, "qk_param_mul is not a*b\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // div
    if (qk_param_div(ret, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "a/b") != 0) {
        fprintf(stderr, "qk_param_div is not a/b\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // pow
    if (qk_param_pow(ret, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "a**b") != 0) {
        fprintf(stderr, "qk_param_pow is not a**b\n");
        result = EqualityError;
        goto cleanup_str;
    }

cleanup_str:
    qk_str_free(str);
cleanup:
    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(ret);

    return result;
}

/**
 * Test calling all unary operations and verify their string representation.
 */
static int test_param_unary_ops(void) {
    QkParam *a = qk_param_new_symbol("a");
    QkParam *b = qk_param_new_symbol("b");
    QkParam *c = qk_param_zero();
    QkParam *ret = qk_param_zero();

    int result = Ok;
    if (qk_param_add(c, a, b) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    char *str;

    // sin
    if (qk_param_sin(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "sin(a + b)") != 0) {
        fprintf(stderr, "qk_param_sin is not sin(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // cos
    if (qk_param_cos(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "cos(a + b)") != 0) {
        fprintf(stderr, "qk_param_cos is not cos(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // tan
    if (qk_param_tan(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "tan(a + b)") != 0) {
        fprintf(stderr, "qk_param_tan is not tan(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // asin
    if (qk_param_asin(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "asin(a + b)") != 0) {
        fprintf(stderr, "qk_param_asin is not asin(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // acos
    if (qk_param_acos(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "acos(a + b)") != 0) {
        fprintf(stderr, "qk_param_acos is not acos(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // atan
    if (qk_param_atan(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "atan(a + b)") != 0) {
        fprintf(stderr, "qk_param_atan is not atan(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // log
    if (qk_param_log(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "log(a + b)") != 0) {
        fprintf(stderr, "qk_param_log is not log(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // exp
    if (qk_param_exp(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "exp(a + b)") != 0) {
        fprintf(stderr, "qk_param_exp is not exp(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // abs
    if (qk_param_abs(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "abs(a + b)") != 0) {
        fprintf(stderr, "qk_param_abs is not abs(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // sign
    if (qk_param_sign(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "sign(a + b)") != 0) {
        fprintf(stderr, "qk_param_sign is not sign(a + b)\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // neg
    if (qk_param_neg(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "-a - b") != 0) {
        fprintf(stderr, "qk_param_neg is not -a - b\n");
        result = EqualityError;
        goto cleanup_str;
    }
    qk_str_free(str);

    // conj
    if (qk_param_conjugate(ret, c) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    str = qk_param_str(ret);
    if (strcmp(str, "conj(a) + conj(b)") != 0) {
        fprintf(stderr, "qk_param_conj is not conj(a) + conj(b)\n");
        result = EqualityError;
        goto cleanup_str;
    }

cleanup_str:
    qk_str_free(str);
cleanup:
    qk_param_free(a);
    qk_param_free(b);
    qk_param_free(c);
    qk_param_free(ret);

    return result;
}

/**
 * Test operations with free parameters and fixed values.
 */
static int test_param_with_value(void) {
    QkParam *a = qk_param_new_symbol("a");
    QkParam *v = qk_param_from_double(2.5);
    QkParam *ret = qk_param_zero();
    char *str;
    int result = Ok;

    // add
    if (qk_param_add(ret, a, v) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    str = qk_param_str(ret);
    if (strcmp(str, "2.5 + a") != 0) {
        fprintf(stderr, "qk_param_add is not 2.5 + a\n");
        result = EqualityError;
    }
    qk_str_free(str);

cleanup:
    qk_param_free(a);
    qk_param_free(v);
    qk_param_free(ret);

    return result;
}

/**
 * Test equality.
 */
static int test_param_equal(void) {
    QkParam *x = qk_param_new_symbol("x");
    QkParam *x_imposter = qk_param_new_symbol("x");
    QkParam *y = qk_param_new_symbol("y");
    QkParam *z = qk_param_new_symbol("z");
    QkParam *val = qk_param_from_double(0.2);
    QkParam *sum1 = qk_param_zero();
    QkParam *sum1_clone = qk_param_zero();
    QkParam *sum2 = qk_param_zero();
    QkParam *mul = qk_param_zero();

    int result = Ok;

    if (qk_param_add(sum1, x, y) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_param_add(sum1_clone, x, y) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_param_add(sum2, x, z) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }
    if (qk_param_mul(mul, x, val) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    if (!qk_param_equal(x, x)) {
        fprintf(stderr, "Symbol not equal to itself\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_param_equal(x, x_imposter)) {
        fprintf(stderr, "Symbol equal a new instance with the same name\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_param_equal(x, mul)) {
        fprintf(stderr, "Symbol equals but they differ by a coefficient\n");
        result = EqualityError;
        goto cleanup;
    }

    if (!qk_param_equal(sum1, sum1_clone)) {
        fprintf(stderr, "Expression not equal to the same expression.\n");
        result = EqualityError;
        goto cleanup;
    }

    if (qk_param_equal(sum1, sum2)) {
        fprintf(stderr, "Expression equal to a different sum.\n");
        result = EqualityError;
        goto cleanup;
    }

cleanup:
    qk_param_free(x);
    qk_param_free(x_imposter);
    qk_param_free(y);
    qk_param_free(z);
    qk_param_free(val);
    qk_param_free(sum1);
    qk_param_free(sum1_clone);
    qk_param_free(sum2);
    qk_param_free(mul);

    return result;
}

/**
 * Test copy.
 */
static int test_param_copy(void) {
    QkParam *x = qk_param_new_symbol("x");
    QkParam *y = qk_param_new_symbol("y");
    QkParam *sum = qk_param_zero();

    int result = Ok;

    if (qk_param_add(sum, x, y) != QkExitCode_Success) {
        result = RuntimeError;
        goto cleanup;
    }

    QkParam *copy = qk_param_copy(sum);
    if (!qk_param_equal(sum, copy)) {
        fprintf(stderr, "Copy not equal to original\n");
        result = EqualityError;
    }
    qk_param_free(copy);
cleanup:
    qk_param_free(x);
    qk_param_free(y);
    qk_param_free(sum);

    return result;
}

int test_param(void) {
    int num_failed = 0;
    num_failed += RUN_TEST(test_param_new);
    num_failed += RUN_TEST(test_param_new_values);
    num_failed += RUN_TEST(test_param_to_real);
    num_failed += RUN_TEST(test_param_equal);
    num_failed += RUN_TEST(test_param_copy);
    num_failed += RUN_TEST(test_param_binary_ops);
    num_failed += RUN_TEST(test_param_unary_ops);
    num_failed += RUN_TEST(test_param_with_value);

    fflush(stderr);
    fprintf(stderr, "=== Number of failed subtests: %i\n", num_failed);

    return num_failed;
}
