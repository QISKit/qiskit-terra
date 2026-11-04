# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Coupling maps for fake backend generation and transpiler testing."""

# directional
# 16 qubits
MELBOURNE_CMAP = [
    [1, 0],
    [1, 2],
    [2, 3],
    [4, 3],
    [4, 10],
    [5, 4],
    [5, 6],
    [5, 9],
    [6, 8],
    [7, 8],
    [9, 8],
    [9, 10],
    [11, 3],
    [11, 10],
    [11, 12],
    [12, 2],
    [13, 1],
    [13, 12],
]

# 27 qubits
MUMBAI_CMAP = [
    [0, 1],
    [1, 0],
    [1, 2],
    [1, 4],
    [2, 1],
    [2, 3],
    [3, 2],
    [3, 5],
    [4, 1],
    [4, 7],
    [5, 3],
    [5, 8],
    [6, 7],
    [7, 4],
    [7, 6],
    [7, 10],
    [8, 5],
    [8, 9],
    [8, 11],
    [9, 8],
    [10, 7],
    [10, 12],
    [11, 8],
    [11, 14],
    [12, 10],
    [12, 13],
    [12, 15],
    [13, 12],
    [13, 14],
    [14, 11],
    [14, 13],
    [14, 16],
    [15, 12],
    [15, 18],
    [16, 14],
    [16, 19],
    [17, 18],
    [18, 15],
    [18, 17],
    [18, 21],
    [19, 16],
    [19, 20],
    [19, 22],
    [20, 19],
    [21, 18],
    [21, 23],
    [22, 19],
    [22, 25],
    [23, 21],
    [23, 24],
    [24, 23],
    [24, 25],
    [25, 22],
    [25, 24],
    [25, 26],
    [26, 25],
]
