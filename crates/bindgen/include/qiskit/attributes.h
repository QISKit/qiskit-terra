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

#ifndef QISKIT_ATTRIBUTES_H
#define QISKIT_ATTRIBUTES_H

// Macros for deprecated functions.
#if defined(__GNUC__) || defined(__clang__)
#define Qk_DEPRECATED_FN __attribute__((deprecated))
#define Qk_DEPRECATED_FN_NOTE(note) __attribute__((deprecated(note)))
#elif defined(_MSC_VER)
#define Qk_DEPRECATED_FN __declspec(deprecated)
#define Qk_DEPRECATED_FN_NOTE(note) __declspec(deprecated(note))
#else
#define Qk_DEPRECATED_FN
#define Qk_DEPRECATED_FN_NOTE(note)
#endif

#endif // QISKIT_ATTRIBUTES_H
