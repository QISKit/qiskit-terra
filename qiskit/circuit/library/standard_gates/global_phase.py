# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Global Phase Gate"""


import numpy

from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit._accelerate.circuit import StandardGate


class GlobalPhaseGate(Gate):
    r"""The global phase gate (:math:`e^{i\theta}`).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`

    Mathematical representation:

    .. math::
        \text{GlobalPhaseGate}\ =
            \begin{pmatrix}
                e^{i\theta}
            \end{pmatrix}
    """

    _standard_gate = StandardGate.GlobalPhase

    def __init__(self, phase: ParameterValueType, label: str | None = None):
        """
        Args:
            phase: The value of phase it takes.
            label: An optional label for the gate.
        """
        super().__init__("global_phase", 0, [phase], label=label)

    def _define(self):

        from qiskit.circuit import QuantumCircuit

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.GlobalPhase._get_definition(self.params), legacy_qubits=True
        )

    def inverse(self, annotated: bool = False):
        r"""Return inverse GlobalPhaseGate gate.

        :math:`\text{GlobalPhaseGate}(\lambda)^{\dagger} = \text{GlobalPhaseGate}(-\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                is always another :class:`.GlobalPhaseGate` with an inverted
                parameter value.

        Returns:
            GlobalPhaseGate: inverse gate.
        """
        return GlobalPhaseGate(-self.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return a numpy.array for the global_phase gate."""
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        theta = self.params[0]
        return numpy.array([[numpy.exp(1j * theta)]], dtype=dtype or complex)

    def __eq__(self, other):
        if isinstance(other, GlobalPhaseGate):
            return self._compare_parameters(other)
        return False
