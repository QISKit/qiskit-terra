# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Two-qubit ZZ-rotation gate."""

from __future__ import annotations

from cmath import exp

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class RZZGate(Gate):
    r"""A parametric 2-qubit :math:`Z \otimes Z` interaction (rotation about ZZ).

    This gate is symmetric, and is maximally entangling at :math:`\theta = \pi/2`.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rzz` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ───■────
                │zz(θ)
        q_1: ───■────

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        R_{ZZ}(\theta) = \exp\left(-i \rotationangle Z{\otimes}Z\right) =
            \begin{pmatrix}
                e^{-i \rotationangle} & 0 & 0 & 0 \\
                0 & e^{i \rotationangle} & 0 & 0 \\
                0 & 0 & e^{i \rotationangle} & 0 \\
                0 & 0 & 0 & e^{-i \rotationangle}
            \end{pmatrix}

    This is a direct sum of RZ rotations, so this gate is equivalent to a
    uniformly controlled (multiplexed) RZ gate:

    .. math::

        R_{ZZ}(\theta) =
            \begin{pmatrix}
                RZ(\theta) & 0 \\
                0 & RZ(-\theta)
            \end{pmatrix}

    Examples:

    .. math::

        R_{ZZ}(\theta = 0) = I

    .. math::

        R_{ZZ}(\theta = 2\pi) = -I

    .. math::

        R_{ZZ}(\theta = \pi) = - i Z \otimes Z

    .. math::

        R_{ZZ}\left(\theta = \frac{\pi}{2}\right) = \frac{1}{\sqrt{2}}
                                \begin{pmatrix}
                                    1-i & 0 & 0 & 0 \\
                                    0 & 1+i & 0 & 0 \\
                                    0 & 0 & 1+i & 0 \\
                                    0 & 0 & 0 & 1-i
                                \end{pmatrix}
    """

    _standard_gate = StandardGate.RZZ

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        """
        Args:
            theta: The rotation angle.
            label: An optional label for the gate.
        """
        super().__init__("rzz", 2, [theta], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        # q_0: ──■─────────────■──
        #      ┌─┴─┐┌───────┐┌─┴─┐
        # q_1: ┤ X ├┤ Rz(0) ├┤ X ├
        #      └───┘└───────┘└───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.RZZ._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        """Return inverse RZZ gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZZGate` with an inverted parameter value.

        Returns:
            RZZGate: inverse gate.
        """
        return RZZGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the RZZ gate."""
        import numpy

        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        itheta2 = 1j * float(self.params[0]) / 2
        return numpy.array(
            [
                [exp(-itheta2), 0, 0, 0],
                [0, exp(itheta2), 0, 0],
                [0, 0, exp(itheta2), 0],
                [0, 0, 0, exp(-itheta2)],
            ],
            dtype=dtype,
        )

    def power(self, exponent: float, annotated: bool = False):
        (theta,) = self.params
        return RZZGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RZZGate):
            return self._compare_parameters(other)
        return False
