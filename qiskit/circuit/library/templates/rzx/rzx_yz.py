# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from __future__ import annotations

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterValueType


def rzx_yz(theta: ParameterValueType | None = None):
    """RZX-based template for CX - RYGate - CX.

    .. code-block:: text

                  ┌────────┐     ┌─────────┐┌─────────┐┌──────────┐
        q_0: ──■──┤ RY(-ϴ) ├──■──┤ RX(π/2) ├┤0        ├┤ RX(-π/2) ├
             ┌─┴─┐└────────┘┌─┴─┐└─────────┘│  RZX(ϴ) │└──────────┘
        q_1: ┤ X ├──────────┤ X ├───────────┤1        ├────────────
             └───┘          └───┘           └─────────┘
    """
    if theta is None:
        theta = Parameter("ϴ")

    circ = QuantumCircuit(2)
    circ.cx(0, 1)
    circ.ry(-1 * theta, 0)
    circ.cx(0, 1)
    circ.rx(np.pi / 2, 0)
    circ.rzx(theta, 0, 1)
    circ.rx(-np.pi / 2, 0)

    return circ
