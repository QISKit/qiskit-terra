# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template circuits over Clifford gates.
"""
from .clifford_2_1 import clifford_2_1
from .clifford_2_2 import clifford_2_2
from .clifford_2_3 import clifford_2_3
from .clifford_2_4 import clifford_2_4
from .clifford_3_1 import clifford_3_1
from .clifford_4_1 import clifford_4_1
from .clifford_4_2 import clifford_4_2
from .clifford_4_3 import clifford_4_3
from .clifford_4_4 import clifford_4_4
from .clifford_5_1 import clifford_5_1
from .clifford_6_1 import clifford_6_1
from .clifford_6_2 import clifford_6_2
from .clifford_6_3 import clifford_6_3
from .clifford_6_4 import clifford_6_4
from .clifford_6_5 import clifford_6_5
from .clifford_8_1 import clifford_8_1
from .clifford_8_2 import clifford_8_2
from .clifford_8_3 import clifford_8_3

__all__ = [
    "clifford_2_1",
    "clifford_2_2",
    "clifford_2_3",
    "clifford_2_4",
    "clifford_3_1",
    "clifford_4_1",
    "clifford_4_2",
    "clifford_4_3",
    "clifford_4_4",
    "clifford_5_1",
    "clifford_6_1",
    "clifford_6_2",
    "clifford_6_3",
    "clifford_6_4",
    "clifford_6_5",
    "clifford_8_1",
    "clifford_8_2",
    "clifford_8_3",
]
