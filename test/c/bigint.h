// This code is part of Qiskit.
//
// (C) Copyright IBM 2026.
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>

/*
 * BigInt helpers for testing.
 */

int biguint_cmp(const QkBigUint a, const QkBigUint b);
bool biguint_eq(const QkBigUint a, const QkBigUint b);
void biguint_debug_print(const QkBigUint *biguint, const char *name);
