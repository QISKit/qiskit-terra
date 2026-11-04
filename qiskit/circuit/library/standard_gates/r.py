# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around an axis in x-y plane."""

import math
from cmath import exp

import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class RGate(Gate):
    r"""Rotation :math:`\theta` around the :math:`\cos(\phi)x + \sin(\phi)y` axis.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.r` method.

    Circuit symbol:

    .. code-block:: text

               ┌────────┐
        q_0:   ┤ R(θ,ϕ) ├ 
               └────────┘


    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R(\theta, \phi) = e^{-i \rotationangle \left(\cos{\phi} x + \sin{\phi} y\right)} =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & -i e^{-i \phi} \sin\left(\rotationangle\right) \\
                -i e^{i \phi} \sin\left(\rotationangle\right) & \cos\left(\rotationangle\right)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.R

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        label: str | None = None,
    ):
        r"""
        Args:
            theta: The rotation angle :math:`\theta`.
            phi: The angle specifying the rotation axis, given by :math:`\cos(\phi) x + \sin(\phi)y`.
            label: An optional label for the gate.
        """
        super().__init__("r", 1, [theta, phi], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌───────────────────────┐
        # q: ┤ U(θ,-π/2 + φ,π/2 - φ) ├
        #    └───────────────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.R._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Invert this gate as: :math:`R(θ, φ)^{\dagger} = R(-θ, φ)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RGate` with an inverted parameter value.

        Returns:
            RGate: inverse gate.
        """
        return RGate(-self.params[0], self.params[1])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the R gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi = float(self.params[0]), float(self.params[1])
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        exp_m = exp(-1j * phi)
        exp_p = exp(1j * phi)
        return numpy.array([[cos, -1j * exp_m * sin], [-1j * exp_p * sin, cos]], dtype=dtype)

    def power(self, exponent: float, annotated: bool = False):
        theta, phi = self.params
        return RGate(exponent * theta, phi)

    def __eq__(self, other):
        if isinstance(other, RGate):
            return self._compare_parameters(other)
        return False
