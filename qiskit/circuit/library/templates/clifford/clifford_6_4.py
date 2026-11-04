# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


from math import pi

from qiskit.circuit.quantumcircuit import QuantumCircuit


def clifford_6_4():
    """
    Clifford template 6_4:

    .. code-block:: text

        global phase: 7π/4
           ┌───┐┌───┐┌───┐┌───┐┌───┐┌───┐
        q: ┤ S ├┤ H ├┤ S ├┤ H ├┤ S ├┤ H ├
           └───┘└───┘└───┘└───┘└───┘└───┘

    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    # Add a global phase of -pi/4 to get Operator(qc) == I
    qc.global_phase = -pi / 4
    return qc
