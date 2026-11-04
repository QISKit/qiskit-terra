# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""iSWAP gate."""


import numpy as np

from qiskit.circuit.singleton import SingletonGate, stdlib_singleton_key
from qiskit.circuit._utils import with_gate_array
from qiskit._accelerate.circuit import StandardGate

from .xx_plus_yy import XXPlusYYGate


@with_gate_array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
class iSwapGate(SingletonGate):
    r"""iSWAP gate.

    A 2-qubit XX+YY interaction.
    This is a Clifford and symmetric gate. Its action is to swap two qubit
    states and phase the :math:`|01\rangle` and :math:`|10\rangle`
    amplitudes by i.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.iswap` method.

    Circuit symbol:

    .. code-block:: text

        q_0: ─⨂─
              │
        q_1: ─⨂─

    Reference implementation:

    .. code-block:: text

             ┌───┐┌───┐     ┌───┐
        q_0: ┤ S ├┤ H ├──■──┤ X ├─────
             ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
        q_1: ┤ S ├─────┤ X ├──■──┤ H ├
             └───┘     └───┘     └───┘

    Matrix representation:

    .. math::

        iSWAP = R_{XX+YY}\left(-\frac{\pi}{2}\right)
          = \exp\left(i \frac{\pi}{4} \left(X{\otimes}X+Y{\otimes}Y\right)\right) =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}

    This gate is equivalent to a SWAP up to a diagonal.

    .. math::

         iSWAP =
            \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
         .  \begin{pmatrix}
                1 & 0 & 0 & 0 \\
                0 & i & 0 & 0 \\
                0 & 0 & i & 0 \\
                0 & 0 & 0 & 1
            \end{pmatrix}
    """

    _standard_gate = StandardGate.ISwap

    def __init__(self, label: str | None = None):
        """
        Args:
            label: An optional label for the gate.
        """
        super().__init__("iswap", 2, [], label=label)

    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """Default definition"""

        from qiskit.circuit import QuantumCircuit

        #      ┌───┐┌───┐     ┌───┐
        # q_0: ┤ S ├┤ H ├──■──┤ X ├─────
        #      ├───┤└───┘┌─┴─┐└─┬─┘┌───┐
        # q_1: ┤ S ├─────┤ X ├──■──┤ H ├
        #      └───┘     └───┘     └───┘

        self.definition = QuantumCircuit._from_circuit_data(
            StandardGate.ISwap._get_definition(self.params), legacy_qubits=True
        )

    def power(self, exponent: float, annotated: bool = False):
        return XXPlusYYGate(-np.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, iSwapGate)
