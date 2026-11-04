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

"""U1 Gate."""
from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit._utils import _ctrl_state_to_int
from qiskit._accelerate.circuit import StandardGate


class U1Gate(Gate):
    r"""Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           U1(\theta) = P(\theta)= U(0,0,\theta)

       .. code-block:: python

          circuit = QuantumCircuit(1)
          circuit.p(lambda, 0) # or circuit.u(0, 0, lambda, 0)

    Circuit symbol:

    .. code-block:: text

             ┌───────┐
        q_0: ┤ U1(θ) ├
             └───────┘

    Matrix representation:

    .. math::

        U1(\theta) =
            \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\theta}
            \end{pmatrix}

    Examples:

    .. math::

        U1(\theta = \pi) = Z

    .. math::

        U1(\theta = \pi/2) = S

    .. math::

        U1(\theta = \pi/4) = T

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.RZGate`:
        This gate is equivalent to RZ up to a phase factor.

            .. math::

                U1(\theta) = e^{i{\theta}/2} RZ(\theta)

        :class:`~qiskit.circuit.library.standard_gates.U3Gate`:
        U3 is a generalization of U2 that covers all single-qubit rotations,
        using two X90 pulses.

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    _standard_gate = StandardGate.U1

    def __init__(self, theta: ParameterValueType, label: str | None = None):
        """
        Args:
            theta: The rotation angle.
            label: An optional label for the gate.
        """
        super().__init__("u1", 1, [theta], label=label)

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌──────┐
        # q: ┤ P(θ) ├
        #    └──────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.U1._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the U1 gate.

        For a single control qubit, the controlled gate is implemented as :class:`.CU1Gate`.
        For more than one control qubits, the controlled gate is implemented as :class:`.MCU1Gate`.
        In each case, the value of ``annotated`` is ignored.


        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.


        Returns:
            A controlled version of this gate.
        """

        if num_ctrl_qubits == 1:
            gate = CU1Gate(self.params[0], label=label, ctrl_state=ctrl_state)
        else:
            gate = MCU1Gate(self.params[0], num_ctrl_qubits, ctrl_state=ctrl_state, label=label)
        gate.base_gate.label = self.label
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted U1 gate (:math:`U1(\lambda)^{\dagger} = U1(-\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.U1Gate` with inverse parameter values.

        Returns:
            U1Gate: inverse gate.
        """
        return U1Gate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the U1 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        lam = float(self.params[0])
        return numpy.array([[1, 0], [0, numpy.exp(1j * lam)]], dtype=dtype)

    def __eq__(self, other):
        return isinstance(other, U1Gate) and self._compare_parameters(other)


class CU1Gate(ControlledGate):
    r"""Controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    .. warning::

       This gate is deprecated. Instead, the :class:`.CPhaseGate` should be used

       .. math::

           CU1(\lambda) = CP(\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(2)
          circuit.cp(lambda, 0, 1)




    Circuit symbol:

    .. code-block:: text


        q_0: ─■──
              │θ
        q_1: ─■──


    Matrix representation:

    .. math::

        CU1(\theta) =
            I \otimes |0\rangle\langle 0| + U1 \otimes |1\rangle\langle 1| =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & e^{i\theta}
            \end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    _standard_gate = StandardGate.CU1

    def __init__(
        self,
        theta: ParameterValueType,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        *,
        _base_label=None,
    ):
        """Create new CU1 gate."""
        super().__init__(
            "cu1",
            2,
            [theta],
            num_ctrl_qubits=1,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U1Gate(theta, label=_base_label),
        )

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌────────┐
        # q_0: ┤ P(θ/2) ├──■───────────────■────────────
        #      └────────┘┌─┴─┐┌─────────┐┌─┴─┐┌────────┐
        # q_1: ──────────┤ X ├┤ P(-θ/2) ├┤ X ├┤ P(θ/2) ├
        #                └───┘└─────────┘└───┘└────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.CU1._get_definition(self.params), legacy_qubits=True
        )

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the CU1 gate.

        The controlled gate is implemented as :class:`.MCU1Gate`, regardless of
        the value of ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state

        gate = MCU1Gate(
            self.params[0],
            num_ctrl_qubits=num_ctrl_qubits + 1,
            ctrl_state=new_ctrl_state,
            label=label,
        )
        gate.base_gate.label = self.label

        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted CU1 gate (:math:`CU1(\lambda)^{\dagger} = CU1(-\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CU1Gate` with inverse parameter
                values.

        Returns:
            CU1Gate: inverse gate.
        """
        return CU1Gate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the CU1 gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype
            )
        else:
            return numpy.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype
            )

    def __eq__(self, other):
        return (
            isinstance(other, CU1Gate)
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )


class MCU1Gate(ControlledGate):
    r"""Multi-controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the state of the control qubits.

    .. warning::

       This gate is deprecated. Instead, the following replacements should be used

       .. math::

           MCU1(\lambda) = MCP(\lambda)

       .. code-block:: python

          circuit = QuantumCircuit(5)
          circuit.mcp(lambda, list(range(4)), 4)




    Circuit symbol:

    .. code-block:: text

            q_0: ────■────
                     │
                     .
                     │
        q_(n-1): ────■────
                 ┌───┴───┐
            q_n: ┤ U1(λ) ├
                 └───────┘

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CU1Gate`:
        The singly-controlled-version of this gate.
    """

    def __init__(
        self,
        lam: ParameterValueType,
        num_ctrl_qubits: int,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        *,
        _base_label=None,
    ):
        """Create new MCU1 gate."""
        super().__init__(
            "mcu1",
            num_ctrl_qubits + 1,
            [lam],
            num_ctrl_qubits=num_ctrl_qubits,
            label=label,
            ctrl_state=ctrl_state,
            base_gate=U1Gate(lam, label=_base_label),
        )

    def _define(self):

        if self.num_ctrl_qubits == 0:
            definition = U1Gate(self.params[0]).definition
        elif self.num_ctrl_qubits == 1:
            definition = CU1Gate(self.params[0]).definition
        else:
            from .p import MCPhaseGate

            definition = MCPhaseGate(self.params[0], self.num_ctrl_qubits).definition

        self.definition = definition

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: str | int | None = None,
        annotated: bool | None = None,
    ):
        """Return a controlled version of the MCU1 gate.

        The controlled gate is implemented as :class:`.MCU1Gate`, regardless of
        the value of ``annotated``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``
            annotated: Ignored.

        Returns:
            A controlled version of this gate.
        """
        ctrl_state = _ctrl_state_to_int(ctrl_state, num_ctrl_qubits)
        new_ctrl_state = (self.ctrl_state << num_ctrl_qubits) | ctrl_state
        gate = MCU1Gate(
            self.params[0],
            num_ctrl_qubits=num_ctrl_qubits + self.num_ctrl_qubits,
            label=label,
            ctrl_state=new_ctrl_state,
        )
        gate.base_gate.label = self.label
        return gate

    def inverse(self, annotated: bool = False):
        r"""Return inverted MCU1 gate (:math:`MCU1(\lambda)^{\dagger} = MCU1(-\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.MCU1Gate` with inverse
                parameter values.

        Returns:
            MCU1Gate: inverse gate.
        """
        return MCU1Gate(
            -self.params[0], num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state
        )

    def __eq__(self, other):
        return (
            isinstance(other, MCU1Gate)
            and self.num_ctrl_qubits == other.num_ctrl_qubits
            and self.ctrl_state == other.ctrl_state
            and self._compare_parameters(other)
        )
