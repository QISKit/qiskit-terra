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
Template circuits for X, CX and Toffoli gates.

References:

Maslov, D. and Dueck, G. W. and Miller, D. M.,
Techniques for the synthesis of reversible Toffoli networks, 2007
https://dx.doi.org/10.1145/1278349.1278355
"""
from .template_nct_2a_1 import template_nct_2a_1
from .template_nct_2a_2 import template_nct_2a_2
from .template_nct_2a_3 import template_nct_2a_3
from .template_nct_4a_1 import template_nct_4a_1
from .template_nct_4a_2 import template_nct_4a_2
from .template_nct_4a_3 import template_nct_4a_3
from .template_nct_4b_1 import template_nct_4b_1
from .template_nct_4b_2 import template_nct_4b_2
from .template_nct_5a_1 import template_nct_5a_1
from .template_nct_5a_2 import template_nct_5a_2
from .template_nct_5a_3 import template_nct_5a_3
from .template_nct_5a_4 import template_nct_5a_4
from .template_nct_6a_1 import template_nct_6a_1
from .template_nct_6a_2 import template_nct_6a_2
from .template_nct_6a_3 import template_nct_6a_3
from .template_nct_6a_4 import template_nct_6a_4
from .template_nct_6b_1 import template_nct_6b_1
from .template_nct_6b_2 import template_nct_6b_2
from .template_nct_6c_1 import template_nct_6c_1
from .template_nct_7a_1 import template_nct_7a_1
from .template_nct_7b_1 import template_nct_7b_1
from .template_nct_7c_1 import template_nct_7c_1
from .template_nct_7d_1 import template_nct_7d_1
from .template_nct_7e_1 import template_nct_7e_1
from .template_nct_9a_1 import template_nct_9a_1
from .template_nct_9c_1 import template_nct_9c_1
from .template_nct_9c_2 import template_nct_9c_2
from .template_nct_9c_3 import template_nct_9c_3
from .template_nct_9c_4 import template_nct_9c_4
from .template_nct_9c_5 import template_nct_9c_5
from .template_nct_9c_6 import template_nct_9c_6
from .template_nct_9c_7 import template_nct_9c_7
from .template_nct_9c_8 import template_nct_9c_8
from .template_nct_9c_9 import template_nct_9c_9
from .template_nct_9c_10 import template_nct_9c_10
from .template_nct_9c_11 import template_nct_9c_11
from .template_nct_9c_12 import template_nct_9c_12
from .template_nct_9d_1 import template_nct_9d_1
from .template_nct_9d_2 import template_nct_9d_2
from .template_nct_9d_3 import template_nct_9d_3
from .template_nct_9d_4 import template_nct_9d_4
from .template_nct_9d_5 import template_nct_9d_5
from .template_nct_9d_6 import template_nct_9d_6
from .template_nct_9d_7 import template_nct_9d_7
from .template_nct_9d_8 import template_nct_9d_8
from .template_nct_9d_9 import template_nct_9d_9
from .template_nct_9d_10 import template_nct_9d_10

__all__ = [
    "template_nct_2a_1",
    "template_nct_2a_2",
    "template_nct_2a_3",
    "template_nct_4a_1",
    "template_nct_4a_2",
    "template_nct_4a_3",
    "template_nct_4b_1",
    "template_nct_4b_2",
    "template_nct_5a_1",
    "template_nct_5a_2",
    "template_nct_5a_3",
    "template_nct_5a_4",
    "template_nct_6a_1",
    "template_nct_6a_2",
    "template_nct_6a_3",
    "template_nct_6a_4",
    "template_nct_6b_1",
    "template_nct_6b_2",
    "template_nct_6c_1",
    "template_nct_7a_1",
    "template_nct_7b_1",
    "template_nct_7c_1",
    "template_nct_7d_1",
    "template_nct_7e_1",
    "template_nct_9a_1",
    "template_nct_9c_1",
    "template_nct_9c_2",
    "template_nct_9c_3",
    "template_nct_9c_4",
    "template_nct_9c_5",
    "template_nct_9c_6",
    "template_nct_9c_7",
    "template_nct_9c_8",
    "template_nct_9c_9",
    "template_nct_9c_10",
    "template_nct_9c_11",
    "template_nct_9c_12",
    "template_nct_9d_1",
    "template_nct_9d_2",
    "template_nct_9d_3",
    "template_nct_9d_4",
    "template_nct_9d_5",
    "template_nct_9d_6",
    "template_nct_9d_7",
    "template_nct_9d_8",
    "template_nct_9d_9",
    "template_nct_9d_10",
]
