# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Global Mølmer–Sørensen gate.
"""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.library.standard_gates import RXXGate
from qiskit.circuit.gate import Gate
from qiskit.utils.deprecation import deprecate_func


class GMS(QuantumCircuit):
    r"""Global Mølmer–Sørensen gate.

    Circuit symbol:

    .. code-block:: text

             ┌───────────┐
        q_0: ┤0          ├
             │           │
        q_1: ┤1   GMS    ├
             │           │
        q_2: ┤2          ├
             └───────────┘

    Expanded Circuit:

    .. plot::
       :alt: Diagram illustrating the previously described circuit.

       from qiskit.circuit.library import GMS
       from qiskit.visualization.library import _generate_circuit_library_visualization
       import numpy as np
       circuit = GMS(num_qubits=3, theta=[[0, np.pi/4, np.pi/8],
                                          [0, 0, np.pi/2],
                                          [0, 0, 0]])
       _generate_circuit_library_visualization(circuit.decompose())

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS
    can be applied to multiple ions to entangle multiple qubits simultaneously [1].

    In the two-qubit case, this is equivalent to an XX(theta) interaction,
    and is thus reduced to the RXXGate. The global MS gate is a sum of XX
    interactions on all pairs [2].

    .. math::

        GMS(\chi_{12}, \chi_{13}, ..., \chi_{n-1 n}) =
        exp(-i \sum_{i=1}^{n} \sum_{j=i+1}^{n} X{\otimes}X \frac{\chi_{ij}}{2})

    References:

    [1] Sørensen, A. and Mølmer, K., Multi-particle entanglement of hot trapped ions.
    Physical Review Letters. 82 (9): 1835–1838.
    `arXiv:9810040 <https://arxiv.org/abs/quant-ph/9810040>`_

    [2] Maslov, D. and Nam, Y., Use of global interactions in efficient quantum circuit
    constructions. New Journal of Physics, 20(3), p.033018.
    `arXiv:1707.06356 <https://arxiv.org/abs/1707.06356>`_
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the MSGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(self, num_qubits: int, theta: list[list[float]] | np.ndarray) -> None:
        """
        Args:
            num_qubits: width of gate.
            theta: a num_qubits x num_qubits symmetric matrix of
                interaction angles for each qubit pair. The upper
                triangle is considered.

        """
        super().__init__(num_qubits, name="gms")
        if not isinstance(theta, list):
            theta = [theta] * int((num_qubits**2 - 1) / 2)
        gms = QuantumCircuit(num_qubits, name="gms")
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                gms.append(RXXGate(theta[i][j]), [i, j])
        self.append(gms.to_gate(), self.qubits)


class MSGate(Gate):
    r"""The Mølmer–Sørensen gate.

    The Mølmer–Sørensen gate is native to ion-trap systems. The global MS
    can be applied to multiple ions to entangle multiple qubits simultaneously [1].

    In the two-qubit case, this is equivalent to an XX interaction,
    and is thus reduced to the :class:`.RXXGate`. The global MS gate is a sum of XX
    interactions on all pairs [2].

    .. math::

        MS(\chi_{12}, \chi_{13}, ..., \chi_{n-1 n}) =
        exp(-i \sum_{i=1}^{n} \sum_{j=i+1}^{n} X{\otimes}X \frac{\chi_{ij}}{2})

    Example::

        import numpy as np
        from qiskit.circuit.library import MSGate
        from qiskit.quantum_info import Operator

        gate = MSGate(num_qubits=3, theta=[[0, np.pi/4, np.pi/8],
                                           [0, 0, np.pi/2],
                                           [0, 0, 0]])
        print(Operator(gate))


    References:

    [1] Sørensen, A. and Mølmer, K., Multi-particle entanglement of hot trapped ions.
    Physical Review Letters. 82 (9): 1835–1838.
    `arXiv:9810040 <https://arxiv.org/abs/quant-ph/9810040>`_

    [2] Maslov, D. and Nam, Y., Use of global interactions in efficient quantum circuit
    constructions. New Journal of Physics, 20(3), p.033018.
    `arXiv:1707.06356 <https://arxiv.org/abs/1707.06356>`_
    """

    def __init__(
        self,
        num_qubits: int,
        theta: ParameterValueType | Sequence[Sequence[ParameterValueType]],
        label: str | None = None,
    ):
        """
        Args:
            num_qubits: The number of qubits the MS gate acts on.
            theta: The XX rotation angles. If a single value, the same angle is used on all
                interactions. Alternatively an upper-triangular, square matrix with width
                ``num_qubits`` can be provided with interaction angles for each qubit pair.
            label: A gate label.
        """
        super().__init__("ms", num_qubits, [theta], label=label)

    def _define(self):
        thetas = self.params[0]
        q = QuantumRegister(self.num_qubits, name="q")
        qc = QuantumCircuit(q)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                # if theta is just a single angle, use that, otherwise use the correct index
                theta = thetas if not isinstance(thetas, Sequence) else thetas[i][j]
                qc._append(RXXGate(theta), [q[i], q[j]], [])

        self.definition = qc

    def validate_parameter(self, parameter):
        if isinstance(parameter, Sequence):

            return [
                [super(MSGate, self).validate_parameter(theta) for theta in row]
                for row in parameter
            ]

        return super().validate_parameter(parameter)
