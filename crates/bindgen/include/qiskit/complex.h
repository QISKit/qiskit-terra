// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef QISKIT__COMPLEX_H
#define QISKIT__COMPLEX_H

/**
 * A complex double.
 *
 * See also ``qk_complex64_to_native`` and ``qk_complex64_from_native`` to convert
 * this struct to (or from) a compiler-native complex number representation.
 */
typedef struct {
    /** The real part. */
    double re;
    /** The imaginary part. */
    double im;
} QkComplex64;

// Complex number typedefs conversions.
#ifdef __cplusplus
#include <complex>
static inline std::complex<double> qk_complex64_to_native(QkComplex64 *value) {
    return std::complex<double>(value->re, value->im);
}
static inline QkComplex64 qk_complex64_from_native(std::complex<double> *value) {
    QkComplex64 ret = {value->real(), value->imag()};
    return ret;
}
#else //__cplusplus
#include <complex.h>

#ifdef _MSC_VER
static inline _Dcomplex qk_complex64_to_native(QkComplex64 *value) {
    return (_Dcomplex){value->re, value->im};
}
static inline QkComplex64 qk_complex64_from_native(_Dcomplex *value) {
    return (QkComplex64){creal(*value), cimag(*value)};
}
#else
static inline double _Complex qk_complex64_to_native(QkComplex64 *value) {
    return value->re + I * value->im;
}
static inline QkComplex64 qk_complex64_from_native(double _Complex *value) {
    return (QkComplex64){creal(*value), cimag(*value)};
}
#endif // _MSC_VER

#endif //__cplusplus

#endif // QISKIT__COMPLEX_H
