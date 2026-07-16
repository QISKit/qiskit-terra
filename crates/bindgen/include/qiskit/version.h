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

#ifndef QISKIT_VERSION_H
#define QISKIT_VERSION_H

#define QISKIT_RELEASE_LEVEL_DEV 0xA
#define QISKIT_RELEASE_LEVEL_BETA 0xB
#define QISKIT_RELEASE_LEVEL_RC 0xC
#define QISKIT_RELEASE_LEVEL_FINAL 0xF

#define QISKIT_VERSION_MAJOR 2
#define QISKIT_VERSION_MINOR 6
#define QISKIT_VERSION_PATCH 0
#define QISKIT_RELEASE_LEVEL QISKIT_RELEASE_LEVEL_DEV
// For the final release, set the below to 0.
#define QISKIT_RELEASE_SERIAL 0

#define QISKIT_VERSION "2.6.0-dev"

#define QISKIT_GET_VERSION_HEX(major, minor, patch, level, serial)                                 \
    (((major) & 0xff) << 24 | ((minor) & 0xff) << 16 | ((patch) & 0xff) << 8 |                     \
     ((level) & 0xf) << 4 | ((serial) & 0xf))

// HEX version using 4 bytes: 0xMMmmppls. For example, 2.1.0rc1 is 0x020100C1.
#define QISKIT_VERSION_HEX                                                                         \
    QISKIT_GET_VERSION_HEX(QISKIT_VERSION_MAJOR, QISKIT_VERSION_MINOR, QISKIT_VERSION_PATCH,       \
                           QISKIT_RELEASE_LEVEL, QISKIT_RELEASE_SERIAL)

#endif // QISKIT_VERSION_H
