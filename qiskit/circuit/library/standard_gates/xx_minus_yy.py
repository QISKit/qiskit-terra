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

"""Two-qubit XX-YY gate."""

from __future__ import annotations

import math
from cmath import exp


import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit._accelerate.circuit import StandardGate


class XXMinusYYGate(Gate):
    r"""XX-YY interaction gate.

    A 2-qubit parameterized XX-YY interaction. Its action is to induce
    a coherent rotation by some angle between :math:`|00\rangle` and :math:`|11\rangle`.

    Circuit symbol:

    .. code-block:: text

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX-YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{XX-YY}(\theta, \beta) q_0, q_1 =
          RZ_1(\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX-YY}{2}\right) \cdot RZ_1(-\beta) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & 0 & 0 & -i\sin\left(\rotationangle\right)e^{-i\beta} \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                -i\sin\left(\rotationangle\right)e^{i\beta} & 0 & 0 & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.XXMinusYY

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: str | None = "(XX-YY)",
    ):
        """
        Args:
            theta: The rotation angle.
            beta: The phase angle.
            label: The label of the gate.
        """
        super().__init__("xx_minus_yy", 2, [theta, beta], label=label)

    def _define(self):
        """Default definition"""

        #       ┌─────┐  ┌────┐┌───┐     ┌─────────┐      ┌─────┐ ┌──────┐┌───┐
        # q_0: ─┤ Sdg ├──┤ √X ├┤ S ├──■──┤ Ry(θ/2) ├───■──┤ Sdg ├─┤ √Xdg ├┤ S ├
        #      ┌┴─────┴─┐├───┬┘└───┘┌─┴─┐├─────────┴┐┌─┴─┐├─────┤┌┴──────┤└───┘
        # q_1: ┤ Rz(-β) ├┤ S ├──────┤ X ├┤ Ry(-θ/2) ├┤ X ├┤ Sdg ├┤ Rz(β) ├─────
        #      └────────┘└───┘      └───┘└──────────┘└───┘└─────┘└───────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.XXMinusYY._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Inverse gate.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.XXMinusYYGate` with inverse
                parameter values.

        Returns:
            XXMinusYYGate: inverse gate.
        """
        theta, beta = self.params
        return XXMinusYYGate(-theta, beta)

    def __array__(self, dtype=None, copy=None):
        """Gate matrix."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, beta = self.params
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return np.array(
            [
                [cos, 0, 0, -1j * sin * exp(-1j * beta)],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [-1j * sin * exp(1j * beta), 0, 0, cos],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        theta, beta = self.params
        return XXMinusYYGate(exponent * theta, beta)

    def __eq__(self, other):
        if isinstance(other, XXMinusYYGate):
            return self._compare_parameters(other)
        return False
