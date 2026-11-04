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

"""T and Tdg gate."""
import math


import numpy

from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit._utils import with_gate_array
from qiskit._accelerate.circuit import StandardGate


@with_gate_array([[1, 0], [0, (1 + 1j) / math.sqrt(2)]])
class TGate(SingletonGate):
    r"""Single qubit T gate (:math:`\sqrt[4]{Z}`).

    It induces a :math:`\pi/4` phase, and is sometimes called the :math:`\pi/8` gate, because
    it is equivalent to :math:`\exp(i\pi/8~Z)` up to a global phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.t` method.

    Matrix representation:

    .. math::

        T = \begin{pmatrix}
                1 & 0 \\
                0 & e^{i\pi/4}
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌───┐
        q_0: ┤ T ├
             └───┘

    Equivalent to a :math:`\pi/4` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.T

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("t", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌────────┐
        # q: ┤ P(π/4) ├
        #    └────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.T._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverse T gate (i.e. :math:`T^\dagger`).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.TdgGate`.

        Returns:
            TdgGate: inverse of :class:`.TGate`
        """
        return TdgGate()

    def power(self, exponent: float, annotated: bool = False):
        return PhaseGate(0.25 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, TGate)


@with_gate_array([[1, 0], [0, (1 - 1j) / math.sqrt(2)]])
class TdgGate(SingletonGate):
    r"""Single qubit T-adjoint gate (:math:`T^\dagger`).

    It induces a :math:`-\pi/4` phase.

    This is a non-Clifford gate and a fourth-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.tdg` method.

    Matrix representation:

    .. math::

        T^\dagger = \begin{pmatrix}
                1 & 0 \\
                0 & e^{-i\pi/4}
            \end{pmatrix}

    Circuit symbol:

    .. code-block:: text

             ┌─────┐
        q_0: ┤ Tdg ├
             └─────┘

    Equivalent to a :math:`-\pi/4` radian rotation about the Z axis.
    """

    _standard_gate = StandardGate.Tdg

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("tdg", 1, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #    ┌─────────┐
        # q: ┤ P(-π/4) ├
        #    └─────────┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.Tdg._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverse :math:`T^\dagger` gate (i.e. T).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.TGate`.

        Returns:
            TGate: inverse of :class:`.TdgGate`
        """
        return TGate()

    def power(self, exponent: float, annotated: bool = False):
        return PhaseGate(-0.25 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, TdgGate)
