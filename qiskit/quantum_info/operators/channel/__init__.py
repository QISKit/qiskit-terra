# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum Channel Representations Package

For explanation of terminology and details of operations see Ref. [1]

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from .superop import SuperOp
from .choi import Choi
from .kraus import Kraus
from .stinespring import Stinespring
from .ptm import PTM
from .chi import Chi

__all__ = [
    "PTM",
    "Chi",
    "Choi",
    "Kraus",
    "Stinespring",
    "SuperOp",
]
