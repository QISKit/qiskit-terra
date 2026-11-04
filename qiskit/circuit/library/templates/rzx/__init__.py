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
from .rzx_yz import rzx_yz
from .rzx_xz import rzx_xz
from .rzx_cy import rzx_cy
from .rzx_zz1 import rzx_zz1
from .rzx_zz2 import rzx_zz2
from .rzx_zz3 import rzx_zz3

__all__ = [
    "rzx_cy",
    "rzx_xz",
    "rzx_yz",
    "rzx_zz1",
    "rzx_zz2",
    "rzx_zz3",
]
