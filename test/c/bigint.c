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

#include <inttypes.h>
#include <qiskit.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

int biguint_cmp(const QkBigUint a, const QkBigUint b) {
    if (a.num_limbs < b.num_limbs)
        return -1;
    if (a.num_limbs > b.num_limbs)
        return 1;
    return memcmp(a.limbs, b.limbs, a.num_limbs * sizeof(uint64_t));
}

bool biguint_eq(const QkBigUint a, const QkBigUint b) { return biguint_cmp(a, b) == 0; }

/*
 * Print QkBigUint values (debug helper)
 */
void biguint_debug_print(const QkBigUint *biguint, const char *name) {
    if (name == NULL)
        name = "biguint";
    fprintf(stderr, "%s.num_limbs = %zu\n", name, biguint->num_limbs);
    for (size_t limb_idx = 0; limb_idx < biguint->num_limbs; limb_idx++) {
        fprintf(stderr, "%s[%zu] = [", name, limb_idx);
        for (size_t i = 0; i < sizeof(uint64_t); i++) {
            fprintf(stderr, "%02" PRIX8,
                    ((uint8_t *)biguint->limbs)[sizeof(uint64_t) * limb_idx + i]);
            if (i < sizeof(uint64_t) - 1)
                fprintf(stderr, " ");
        }
        fprintf(stderr, "]\n");
    }
}
