# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing swap strategies for blocks of commuting gates.

Swap routing is, in general, a hard problem. However, this problem is much simpler if
the gates commute. Many variational algorithms such as QAOA are built with blocks of
commuting gates. Transpiling such circuits with a general purpose SWAP router typically
yields sub optimal results or is costly to run. This module introduces a framework to
transpile blocks of commuting gates by applying layers of a predefined swap strategy.
Further details can also be found here: https://arxiv.org/abs/2202.03459.
"""

from .swap_strategy import SwapStrategy
from .pauli_2q_evolution_commutation import FindCommutingPauliEvolutions
from .commuting_2q_gate_router import Commuting2qGateRouter

__all__ = ["Commuting2qGateRouter", "FindCommutingPauliEvolutions", "SwapStrategy"]
