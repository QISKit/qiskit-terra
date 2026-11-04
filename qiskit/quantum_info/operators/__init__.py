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

"""Quantum Operators."""

from . import channel, dihedral, symplectic, utils
from .channel import *
from .dihedral import *
from .symplectic import *
from .utils import *
from .measures import average_gate_fidelity, diamond_norm, gate_error, process_fidelity
from .operator import Operator
from .scalar_op import ScalarOp

__all__ = [
    "Operator",
    "ScalarOp",
    "average_gate_fidelity",
    "diamond_norm",
    "gate_error",
    "process_fidelity",
]
__all__ += channel.__all__
__all__ += dihedral.__all__
__all__ += symplectic.__all__
__all__ += utils.__all__
