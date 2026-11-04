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

"""Two-qubit XX+YY gate."""

from __future__ import annotations

import math
from cmath import exp


import numpy

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class XXPlusYYGate(Gate):
    r"""XX+YY interaction gate.

    A 2-qubit parameterized XX+YY interaction, also known as an XY gate. Its action is to induce
    a coherent rotation by some angle between :math:`|01\rangle` and :math:`|10\rangle`.

    Circuit symbol:

    .. code-block:: text

             ┌───────────────┐
        q_0: ┤0              ├
             │  (XX+YY)(θ,β) │
        q_1: ┤1              ├
             └───────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{XX+YY}(\theta, \beta)\ q_0, q_1 =
          RZ_0(-\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX+YY}{2}\right) \cdot RZ_0(\beta) =
            \begin{pmatrix}
                1 & 0 & 0 & 0  \\
                0 & \cos\left(\rotationangle\right) & -i\sin\left(\rotationangle\right)e^{-i\beta} & 0 \\
                0 & -i\sin\left(\rotationangle\right)e^{i\beta} & \cos\left(\rotationangle\right) & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In the above example we apply the gate
        on (q_0, q_1) which results in adding the (optional) phase defined
        by :math:`\beta` on q_0. Instead, if we apply it on (q_1, q_0), the
        phase is added on q_1. If :math:`\beta` is set to its default value
        of :math:`0`, the gate is equivalent in big and little endian.

        .. code-block:: text

                 ┌───────────────┐
            q_0: ┤1              ├
                 │  (XX+YY)(θ,β) │
            q_1: ┤0              ├
                 └───────────────┘

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}

            R_{XX+YY}(\theta, \beta)\ q_0, q_1 =
            RZ_1(-\beta) \cdot \exp\left(-i \frac{\theta}{2} \frac{XX+YY}{2}\right) \cdot RZ_1(\beta) =
                \begin{pmatrix}
                    1 & 0 & 0 & 0  \\
                    0 & \cos\left(\rotationangle\right) &
                    -i\sin\left(\rotationangle\right)e^{i\beta} & 0 \\
                    0 & -i\sin\left(\rotationangle\right)e^{-i\beta} &
                    \cos\left(\rotationangle\right) & 0 \\
                    0 & 0 & 0 & 1
                \end{pmatrix}
    """

    _standard_gate = StandardGate.XXPlusYY

    def __init__(
        self,
        theta: ParameterValueType,
        beta: ParameterValueType = 0,
        label: str | None = "(XX+YY)",
    ):
        """
        Args:
            theta: The rotation angle.
            beta: The phase angle.
            label: The label of the gate.
        """
        super().__init__("xx_plus_yy", 2, [theta, beta], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌───────┐┌───┐      ┌───┐┌──────────┐┌───┐┌─────┐┌────────┐
        # q_0: ┤ Rz(β) ├┤ S ├──────┤ X ├┤ Ry(-θ/2) ├┤ X ├┤ Sdg ├┤ Rz(-β) ├─────
        #      └┬─────┬┘├───┴┐┌───┐└─┬─┘├──────────┤└─┬─┘├─────┤└┬──────┬┘┌───┐
        # q_1: ─┤ Sdg ├─┤ √X ├┤ S ├──■──┤ Ry(-θ/2) ├──■──┤ Sdg ├─┤ √Xdg ├─┤ S ├
        #       └─────┘ └────┘└───┘     └──────────┘     └─────┘ └──────┘ └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.XXPlusYY._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse XX+YY gate (i.e. with the negative rotation angle and same phase angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.XXPlusYYGate` with inverse
                parameter values.

        Returns:
            XXPlusYYGate: inverse gate.
        """
        return XXPlusYYGate(-self.params[0], self.params[1])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the XX+YY gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        half_theta = float(self.params[0]) / 2
        beta = float(self.params[1])
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        return numpy.array(
            [
                [1, 0, 0, 0],
                [0, cos, -1j * sin * exp(-1j * beta), 0],
                [0, -1j * sin * exp(1j * beta), cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        theta, beta = self.params
        return XXPlusYYGate(exponent * theta, beta)

    def __eq__(self, other):
        if isinstance(other, XXPlusYYGate):
            return self._compare_parameters(other)
        return False
