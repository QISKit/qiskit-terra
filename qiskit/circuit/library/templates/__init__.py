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
A library of template circuits.

Templates are circuits that compute the identity. They find use
in circuit optimization where matching part of the template allows the compiler
to replace the match with the inverse of the remainder from the template.
"""
from .nct.template_nct_2a_1 import template_nct_2a_1
from .nct.template_nct_2a_2 import template_nct_2a_2
from .nct.template_nct_2a_3 import template_nct_2a_3
from .nct.template_nct_4a_1 import template_nct_4a_1
from .nct.template_nct_4a_2 import template_nct_4a_2
from .nct.template_nct_4a_3 import template_nct_4a_3
from .nct.template_nct_4b_1 import template_nct_4b_1
from .nct.template_nct_4b_2 import template_nct_4b_2
from .nct.template_nct_5a_1 import template_nct_5a_1
from .nct.template_nct_5a_2 import template_nct_5a_2
from .nct.template_nct_5a_3 import template_nct_5a_3
from .nct.template_nct_5a_4 import template_nct_5a_4
from .nct.template_nct_6a_1 import template_nct_6a_1
from .nct.template_nct_6a_2 import template_nct_6a_2
from .nct.template_nct_6a_3 import template_nct_6a_3
from .nct.template_nct_6a_4 import template_nct_6a_4
from .nct.template_nct_6b_1 import template_nct_6b_1
from .nct.template_nct_6b_2 import template_nct_6b_2
from .nct.template_nct_6c_1 import template_nct_6c_1
from .nct.template_nct_7a_1 import template_nct_7a_1
from .nct.template_nct_7b_1 import template_nct_7b_1
from .nct.template_nct_7c_1 import template_nct_7c_1
from .nct.template_nct_7d_1 import template_nct_7d_1
from .nct.template_nct_7e_1 import template_nct_7e_1
from .nct.template_nct_9a_1 import template_nct_9a_1
from .nct.template_nct_9c_1 import template_nct_9c_1
from .nct.template_nct_9c_2 import template_nct_9c_2
from .nct.template_nct_9c_3 import template_nct_9c_3
from .nct.template_nct_9c_4 import template_nct_9c_4
from .nct.template_nct_9c_5 import template_nct_9c_5
from .nct.template_nct_9c_6 import template_nct_9c_6
from .nct.template_nct_9c_7 import template_nct_9c_7
from .nct.template_nct_9c_8 import template_nct_9c_8
from .nct.template_nct_9c_9 import template_nct_9c_9
from .nct.template_nct_9c_10 import template_nct_9c_10
from .nct.template_nct_9c_11 import template_nct_9c_11
from .nct.template_nct_9c_12 import template_nct_9c_12
from .nct.template_nct_9d_1 import template_nct_9d_1
from .nct.template_nct_9d_2 import template_nct_9d_2
from .nct.template_nct_9d_3 import template_nct_9d_3
from .nct.template_nct_9d_4 import template_nct_9d_4
from .nct.template_nct_9d_5 import template_nct_9d_5
from .nct.template_nct_9d_6 import template_nct_9d_6
from .nct.template_nct_9d_7 import template_nct_9d_7
from .nct.template_nct_9d_8 import template_nct_9d_8
from .nct.template_nct_9d_9 import template_nct_9d_9
from .nct.template_nct_9d_10 import template_nct_9d_10

from .rzx.rzx_yz import rzx_yz
from .rzx.rzx_xz import rzx_xz
from .rzx.rzx_cy import rzx_cy
from .rzx.rzx_zz1 import rzx_zz1
from .rzx.rzx_zz2 import rzx_zz2
from .rzx.rzx_zz3 import rzx_zz3

from .clifford.clifford_2_1 import clifford_2_1
from .clifford.clifford_2_2 import clifford_2_2
from .clifford.clifford_2_3 import clifford_2_3
from .clifford.clifford_2_4 import clifford_2_4
from .clifford.clifford_3_1 import clifford_3_1
from .clifford.clifford_4_1 import clifford_4_1
from .clifford.clifford_4_2 import clifford_4_2
from .clifford.clifford_4_3 import clifford_4_3
from .clifford.clifford_4_4 import clifford_4_4
from .clifford.clifford_5_1 import clifford_5_1
from .clifford.clifford_6_1 import clifford_6_1
from .clifford.clifford_6_2 import clifford_6_2
from .clifford.clifford_6_3 import clifford_6_3
from .clifford.clifford_6_4 import clifford_6_4
from .clifford.clifford_6_5 import clifford_6_5
from .clifford.clifford_8_1 import clifford_8_1
from .clifford.clifford_8_2 import clifford_8_2
from .clifford.clifford_8_3 import clifford_8_3

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
    "rzx_cy",
    "rzx_xz",
    "rzx_yz",
    "rzx_zz1",
    "rzx_zz2",
    "rzx_zz3",
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
