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

import cmath
import math
from cmath import exp

import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class UGate(Gate):
    r"""Generic single-qubit rotation in terms of ZYZ Euler angles.

    The action of this gate can be related to the standard ZYZ Euler decomposition by

    .. math::

        U(\theta, \phi, \lambda) = P(\phi) R_Y(\theta) P(\lambda) 
        = e^{i\frac{\phi + \lambda}{2}} R_Z(\phi) R_Y(\theta) R_Z(\lambda).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.u` method.

    Circuit symbol:

    .. code-block:: text

             ┌──────────┐
        q_0: ┤ U(ϴ,φ,λ) ├
             └──────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        U(\theta, \phi, \lambda) =
        \begin{pmatrix}
            \cos\left(\rotationangle\right) & -e^{i\lambda}\sin\left(\rotationangle\right) \\
            e^{i\phi}\sin\left(\rotationangle\right) & e^{i(\phi+\lambda)}\cos\left(\rotationangle\right)
        \end{pmatrix}

    .. note::

        The matrix representation shown here differs from the `OpenQASM 3.0 specification
        <https://openqasm.com/language/gates.html#built-in-gates>`_ by a global phase of
        :math:`e^{-i\theta/2}`.  It differs from the
        `OpenQASM 2.0 specification <https://doi.org/10.48550/arXiv.1707.03429>`_ 
        by a global phase of :math:`e^{i(\phi+\lambda)/2}`.

    Examples:

    .. math::

        U\left(\theta, -\frac{\pi}{2}, \frac{\pi}{2}\right) = RX(\theta)

    .. math::

        U(\theta, 0, 0) = RY(\theta)
    """

    _standard_gate = StandardGate.U

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
        super().__init__("u", 1, [theta, phi, lam], label=label)

    def inverse(self, annotated: bool = False):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the
                inverse of this gate is always a :class:`.UGate` with inverse parameter values.

        Returns:
            UGate: inverse gate.
        """
        return UGate(-self.params[0], -self.params[2], -self.params[1])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the U gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CUGate`,
        regardless of the value of `annotated`.

        For more than one control qubit,
        the controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False``, and as :class:`.AnnotatedOperation` when ``annotated`` is ``True``.
        When ``annotated`` is ``None``, it is interpreted as ``True`` when the gate has free
        parameters (in which case the gate cannot be synthesized at the construction time),
        and as ``False`` otherwise.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Ignored if the controlled gate is implemented as an
                annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CUGate(
                self.params[0],
                self.params[1],
                self.params[2],
                0,
                label=label,
                ctrl_state=ctrl_state,
            )
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam = (float(param) for param in self.params)
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
        if isinstance(other, UGate):
            return self._compare_parameters(other)
        return False


class U4Gate(Gate):
    r"""Generic single-qubit unitary with global phase.

    This gate represents a generic :math:`U(2)` operation, obtained by adding a global-phase
    parameter to :class:`.UGate`.

    .. math::

        U_4(\theta, \phi, \lambda, \gamma) = e^{i\gamma} U(\theta, \phi, \lambda)

    .. note::

        Currently, :class:`.U4Gate` is only available in Python. It was introduced as to allow
        :class:`.CUGate` to be defined as the controlled version of this gate and have the same
        parameters as the base gate.

    Circuit symbol:

    .. code-block:: text

             ┌─────────────┐
        q_0: ┤ U4(ϴ,φ,λ,γ) ├
             └─────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        U_4(\theta, \phi, \lambda, \gamma) =
        \begin{pmatrix}
            e^{i\gamma} \cos\left(\rotationangle\right) & -e^{i(\gamma + \lambda)}\sin\left(\rotationangle\right) \\
            e^{i(\gamma + \phi)}\sin\left(\rotationangle\right) & e^{i(\gamma + \phi+\lambda)}\cos\left(\rotationangle\right)
        \end{pmatrix}

    """

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        gamma: ParameterValueType,
        label: str | None = None,
    ):
        r"""
        Args:
            theta: The angle :math:`\theta` corresponding to the :math:`R_Y(\theta)` rotation.
            phi: The angle :math:`\phi` corresponding to the :math:`R_Z(\phi)` rotation.
            lam: The angle :math:`\lambda` corresponding to the :math:`R_Z(\lambda)` rotation.
            gamma: The global phase corresponding to :math:`e^{i\gamma}`.
            label: An optional label for the gate.
        """
        super().__init__("u4", 1, [theta, phi, lam, gamma], label=label)

    def inverse(self, annotated: bool = False):
        r"""Return inverted U gate.

        :math:`U4(\theta,\phi,\lambda, \gamma)^{\dagger} =U4(-\theta,-\lambda,-\phi,-\gamma))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the
                inverse of this gate is always a :class:`.U4Gate` with inverse parameter values.

        Returns:
            U4Gate: inverse gate.
        """
        return U4Gate(-self.params[0], -self.params[2], -self.params[1], -self.params[3])

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the U4 gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CUGate`,
        regardless of the value of `annotated`.

        For more than one control qubit,
        the controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False`` or ``None``, and as :class:`.AnnotatedOperation` when ``annotated``
        is ``True``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Ignored if the controlled gate is implemented as an
                annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation.

        Returns:
            A controlled version of this gate.
        """
        if num_ctrl_qubits == 1:
            gate = CUGate(
                self.params[0],
                self.params[1],
                self.params[2],
                self.params[3],
                label=label,
                ctrl_state=ctrl_state,
            )
            gate.base_gate.label = self.label
        else:
            gate = super().control(
                num_ctrl_qubits=num_ctrl_qubits,
                label=label,
                ctrl_state=ctrl_state,
                annotated=annotated,
            )
        return gate

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U4 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam, gamma = (float(param) for param in self.params)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        return numpy.array(
            [
                [exp(1j * gamma) * cos, -exp(1j * (gamma + lam)) * sin],
                [exp(1j * (gamma + phi)) * sin, exp(1j * (gamma + phi + lam)) * cos],
            ],
            dtype=dtype or complex,
        )

    def _define(self):
        """Default definition"""

        # global phase: γ
        #    ┌──────────┐
        # q: ┤ U(ϴ,φ,λ) ├
        #    └──────────┘

        from qiskit.circuit import QuantumCircuit

        definition = QuantumCircuit(1, global_phase=self.params[3])
        definition.u(self.params[0], self.params[1], self.params[2], 0)
        self.definition = definition

    def __eq__(self, other):
        if isinstance(other, U4Gate):
            return self._compare_parameters(other)
        return False


class _UGateWithGlobalPhase(Gate):
    """A one-qubit U gate with an explicit global phase parameter."""

    def __init__(self, theta, phi, lam, gamma, label=None):
        super().__init__("u_global_phase", 1, [theta, phi, lam, gamma], label=label)

    @Gate.params.setter
    def params(self, parameters):
        self._definition = None
        Gate.params.fset(self, parameters)

    def _define(self):
        from qiskit.circuit import QuantumCircuit

        theta, phi, lam, gamma = self.params
        definition = QuantumCircuit(1, name=self.name)
        definition.global_phase = gamma
        definition.append(UGate(theta, phi, lam), [0])
        self.definition = definition

    def inverse(self, annotated: bool = False):
        if annotated:
            return super().inverse(annotated=annotated)
        theta, phi, lam, gamma = self.params
        return _UGateWithGlobalPhase(-theta, -lam, -phi, -gamma)

    def __eq__(self, other):
        return isinstance(other, _UGateWithGlobalPhase) and self._compare_parameters(other)


class CUGate(ControlledGate):
    r"""Controlled-U gate (4-parameter two-qubit gate).

    This is a controlled version of :class:`.U4Gate` (generic single qubit rotation
    including a possible global phase).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cu` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ───────■──────
             ┌──────┴──────┐
        q_1: ┤ U4(ϴ,φ,λ,γ) ├
             └─────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\theta}{2}}

        CU(\theta, \phi, \lambda, \gamma)\ q_0, q_1 =
            I \otimes |0\rangle\langle 0| +
            e^{i\gamma} U(\theta,\phi,\lambda) \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & e^{i\gamma}\cos(\rotationangle) &
                0 & -e^{i(\gamma + \lambda)}\sin(\rotationangle) \\
                0 & 0 & 1 & 0 \\
                0 & e^{i(\gamma+\phi)}\sin(\rotationangle) &
                0 & e^{i(\gamma+\phi+\lambda)}\cos(\rotationangle)
            \end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. code-block:: text

                 ┌────────────┐
            q_0: ┤ U(ϴ,φ,λ,γ) ├
                 └─────┬──────┘
            q_1: ──────■───────

        .. math::

            \newcommand{\rotationangle}{\frac{\theta}{2}}
            CU(\theta, \phi, \lambda, \gamma)\ q_1, q_0 =
            |0\rangle\langle 0| \otimes I +
            e^{i\gamma}|1\rangle\langle 1| \otimes U(\theta,\phi,\lambda) =
            \begin{pmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{i\gamma} \cos(\rotationangle) & -e^{i(\gamma + \lambda)}\sin(\rotationangle) \\
            0 & 0 &
            e^{i(\gamma + \phi)}\sin(\rotationangle) & e^{i(\gamma + \phi+\lambda)}\cos(\rotationangle)
            \end{pmatrix}
    """

    _standard_gate = StandardGate.CU

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        gamma: ParameterValueType,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        *,
        _base_label=None,
    ):
        """Create new CU gate."""
        super().__init__(
            "cu",
            2,
            [theta, phi, lam, gamma],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U4Gate(theta, phi, lam, gamma, label=_base_label),
        )

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #          ┌──────┐    ┌──────────────┐
        # q_0: ────┤ P(γ) ├────┤ P(λ/2 + φ/2) ├──■────────────────────────────■────────────────
        #      ┌───┴──────┴───┐└──────────────┘┌─┴─┐┌──────────────────────┐┌─┴─┐┌────────────┐
        # q_1: ┤ P(λ/2 - φ/2) ├────────────────┤ X ├┤ U(-0/2,0,-λ/2 - φ/2) ├┤ X ├┤ U(0/2,φ,0) ├
        #      └──────────────┘                └───┘└──────────────────────┘└───┘└────────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CU._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverted CU gate.

        :math:`CU(\theta,\phi,\lambda,\gamma)^{\dagger} = CU(-\theta,-\phi,-\lambda,-\gamma))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CUGate` with inverse parameter
                values.

        Returns:
            CUGate: inverse gate.
        """
        return CUGate(
            -self.params[0],
            -self.params[2],
            -self.params[1],
            -self.params[3],
            ctrl_state=self.ctrl_state,
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the CU gate."""
        gate = super().control(
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            annotated=annotated,
        )
        if isinstance(gate, ControlledGate):
            gate.base_gate = _UGateWithGlobalPhase(*self.params, label=gate.base_gate.label)
        return gate

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CU gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta, phi, lam, gamma = (float(param) for param in self.params)
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        a = cmath.exp(1j * gamma) * cos
        b = -cmath.exp(1j * (gamma + lam)) * sin
        c = cmath.exp(1j * (gamma + phi)) * sin
        d = cmath.exp(1j * (gamma + phi + lam)) * cos
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, a, 0, b], [0, 0, 1, 0], [0, c, 0, d]], dtype=dtype
            )
        else:
            return numpy.array(
                [[a, 0, b, 0], [0, 1, 0, 0], [c, 0, d, 0], [0, 0, 0, 1]], dtype=dtype
            )

    def __eq__(self, other):
        return (
            isinstance(other, CUGate)
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )
