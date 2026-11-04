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

"""One-pulse single-qubit gate."""
from math import sqrt, pi
from cmath import exp

import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class U2Gate(Gate):
    r"""Single-qubit rotation about the X+Z axis.

    Implemented using one X90 pulse on IBM Quantum systems:

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U2(\phi, \lambda) = U\left(\frac{\pi}{2}, \phi, \lambda\right)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.u(pi/2, phi, lambda)

    Circuit symbol:

    .. code-block:: text

             ┌─────────┐
        q_0: ┤ U2(φ,λ) ├
             └─────────┘

    Matrix representation:

    .. math::

        U2(\phi, \lambda) = \frac{1}{\sqrt{2}}
            \begin{pmatrix}
                1          & -e^{i\lambda} \\
                e^{i\phi} & e^{i(\phi+\lambda)}
            \end{pmatrix}

    Examples:

    .. math::

        U2(\phi,\lambda) = e^{i \frac{\phi + \lambda}{2}}RZ(\phi)
        RY\left(\frac{\pi}{2}\right) RZ(\lambda)
        = e^{- i\frac{\pi}{4}} P\left(\frac{\pi}{2} + \phi\right)
        \sqrt{X} P\left(\lambda- \frac{\pi}{2}\right)

    .. math::

        U2(0, \pi) = H

    .. math::

        U2(0, 0) = RY(\pi/2)

    .. math::

        U2(-\pi/2, \pi/2) = RX(\pi/2)

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.
    """

    _standard_gate = StandardGate.U2

    def __init__(
        self,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: str | None = None,
    ):
        r"""
        Args:
            phi: The rotation angle :math:`\phi`.
            lam: The rotation angle :math:`\lambda`.
            label: An optional label for the gate.
        """
        super().__init__("u2", 1, [phi, lam], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌────────────┐
        # q: ┤ U(π/2,φ,λ) ├
        #    └────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.U2._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverted U2 gate.

        :math:`U2(\phi, \lambda)^{\dagger} =U2(-\lambda-\pi, -\phi+\pi))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.U2Gate` with inverse parameter values.

        Returns:
            U2Gate: inverse gate.
        """
        return U2Gate(-self.params[1] - pi, -self.params[0] + pi)

    def __array__(self, dtype=None, copy=None):
        """Return a Numpy.array for the U2 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        isqrt2 = 1 / sqrt(2)
        phi, lam = self.params
        phi, lam = float(phi), float(lam)
        return numpy.array(
            [
                [isqrt2, -exp(1j * lam) * isqrt2],
                [exp(1j * phi) * isqrt2, exp(1j * (phi + lam)) * isqrt2],
            ],
            dtype=dtype or complex,
        )

    def __eq__(self, other):
        return isinstance(other, U2Gate) and self._compare_parameters(other)
