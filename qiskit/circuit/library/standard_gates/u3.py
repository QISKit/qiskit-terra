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

"""Two-pulse single-qubit gate."""

from __future__ import annotations

import math
from cmath import exp

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class U3Gate(Gate):
    r"""Generic single-qubit rotation gate with 3 Euler angles.

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U3(\theta, \phi, \lambda) = U(\theta, \phi, \lambda)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.u(theta, phi, lambda)

    Circuit symbol:

    .. code-block:: text

             ┌───────────┐
        q_0: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        U3(\theta, \phi, \lambda) =
            \begin{pmatrix}
                \cos\left(\rotationangle\right) & -e^{i\lambda}\sin\left(\rotationangle\right) \\
                e^{i\phi}\sin\left(\rotationangle\right) &
                e^{i(\phi+\lambda)}\cos\left(\rotationangle\right)
            \end{pmatrix}

    .. note::

        The matrix representation shown here differs from the `OpenQASM 2.0 specification
        <https://doi.org/10.48550/arXiv.1707.03429>`_ by a global phase of
        :math:`e^{i(\phi+\lambda)/2}`.

    Examples:

    .. math::

        U3(\theta, \phi, \lambda) = P(\phi) R_Y(\theta) P(\lambda)
        = e^{i\frac{\phi + \lambda}{2}} R_Z(\phi) R_Y(\theta) R_Z(\lambda)

    .. math::

        U3\left(\theta, -\frac{\pi}{2}, \frac{\pi}{2}\right) = RX(\theta)

    .. math::

        U3(\theta, 0, 0) = RY(\theta)
    """

    _standard_gate = StandardGate.U3

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: str | None = None,
    ):
        r"""
        Args:
            theta: The angle :math:`\theta` corresponding to the :math:`R_Y(\theta)` rotation.
            phi: The angle :math:`\phi` corresponding to the :math:`R_Z(\phi)` rotation.
            lam: The angle :math:`\lambda` corresponding to the :math:`R_Z(\lambda)` rotation.
            label: An optional label for the gate.
        """
        super().__init__("u3", 1, [theta, phi, lam], label=label)

    def inverse(self, annotated: bool = False):
        r"""Return inverted U3 gate.

        :math:`U3(\theta,\phi,\lambda)^{\dagger} =U3(-\theta,-\lambda,-\phi))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.U3Gate` with inverse parameter values.

        Returns:
            U3Gate: inverse gate.
        """
        return U3Gate(-self.params[0], -self.params[2], -self.params[1])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the U3 gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CU3Gate`,
        regardless of the value of `annotated`.

        For more than one control qubit,
        the controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False``, and as :class:`.AnnotatedOperation` when ``annotated`` is ``True``.
        When ``annotated`` is ``None``, it is interpreted as ``True`` when the gate has free
        parameters (in which case the gate cannot be synthesized at the construction time)
        and as ``False`` otherwise.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
                Ignored if the controlled gate is implemented as an annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation. If ``None``, treated as ``False``

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CU3Gate(*self.params, label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌──────────┐
        # q: ┤ U(θ,φ,λ) ├
        #    └──────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.U3._get_definition(self.params), legacy_qubits=True
        )

    def __array__(self, dtype=None, copy=None):
        """Return a Numpy.array for the U3 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array(
            [
                [cos, -exp(1j * lam) * sin],
                [exp(1j * phi) * sin, exp(1j * (phi + lam)) * cos],
            ],
            dtype=dtype or complex,
        )

    def __eq__(self, other):
        return isinstance(other, U3Gate) and self._compare_parameters(other)


class CU3Gate(ControlledGate):
    r"""Controlled-U3 gate (3-parameter two-qubit gate).

    This is a controlled version of the U3 gate (generic single qubit rotation).
    It is restricted to 3 parameters, and so cannot cover generic two-qubit
    controlled gates).

    .. warning::

       This gate is deprecated. Instead, the :class:`.CUGate` should be used

       .. math::

           CU3(\theta, \phi, \lambda) = CU(\theta, \phi, \lambda, 0)

       .. code-block:: python

          circuit = QuantumCircuit(2)
          gamma = 0
          circuit.cu(theta, phi, lambda, gamma, 0, 1)




    Circuit symbol:

    .. code-block:: text

        q_0: ──────■──────
             ┌─────┴─────┐
        q_1: ┤ U3(ϴ,φ,λ) ├
             └───────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        CU3(\theta, \phi, \lambda)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| +
            U3(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0                   & 0 & 0 \\
                0 & \cos(\rotationangle)           & 0 & -e^{i\lambda}\sin(\rotationangle) \\
                0 & 0                   & 1 & 0 \\
                0 & e^{i\phi}\sin(\rotationangle)  & 0 & e^{i(\phi+\lambda)}\cos(\rotationangle)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌───────────┐
            q_0: ┤ U3(ϴ,φ,λ) ├
                 └─────┬─────┘
            q_1: ──────■──────

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}

            CU3(\theta, \phi, \lambda)\ q_1, q_0 =
                |0\rangle\langle 0| \otimes I +
                |1\rangle\langle 1| \otimes U3(\theta,\phi,\lambda) =
                \begin{pmatrix}
                    1 & 0   & 0                  & 0 \\
                    0 & 1   & 0                  & 0 \\
                    0 & 0   & \cos(\rotationangle)          & -e^{i\lambda}\sin(\rotationangle) \\
                    0 & 0   & e^{i\phi}\sin(\rotationangle) & e^{i(\phi+\lambda)}\cos(\rotationangle)
                \end{pmatrix}
    """

    _standard_gate = StandardGate.CU3

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CU3 gate."""
        super().__init__(
            "cu3",
            2,
            [theta, phi, lam],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U3Gate(theta, phi, lam, label=_base_label),
        )

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌──────────────┐
        # q_0: ┤ P(λ/2 + φ/2) ├──■────────────────────────────■────────────────
        #      ├──────────────┤┌─┴─┐┌──────────────────────┐┌─┴─┐┌────────────┐
        # q_1: ┤ P(λ/2 - φ/2) ├┤ X ├┤ U(-θ/2,0,-λ/2 - φ/2) ├┤ X ├┤ U(θ/2,φ,0) ├
        #      └──────────────┘└───┘└──────────────────────┘└───┘└────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CU3._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverted CU3 gate.

        :math:`CU3(\theta,\phi,\lambda)^{\dagger} =CU3(-\theta,-\phi,-\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CU3Gate` with inverse
                parameter values.

        Returns:
            CU3Gate: inverse gate.
        """
        return CU3Gate(
            -self.params[0], -self.params[2], -self.params[1], ctrl_state=self.ctrl_state
        )

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CU3 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam = self.params
        theta, phi, lam = float(theta), float(phi), float(lam)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        if self.ctrl_state:
            return numpy.array(
                [
                    [1, 0, 0, 0],
                    [0, cos, 0, -exp(1j * lam) * sin],
                    [0, 0, 1, 0],
                    [0, exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos],
                ],
                dtype=dtype or complex,
            )
        else:
            return numpy.array(
                [
                    [cos, 0, -exp(1j * lam) * sin, 0],
                    [0, 1, 0, 0],
                    [exp(1j * phi) * sin, 0, exp(1j * (phi + lam)) * cos, 0],
                    [0, 0, 0, 1],
                ],
                dtype=dtype or complex,
            )

    def __eq__(self, other):
        return (
            isinstance(other, CU3Gate)
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )
