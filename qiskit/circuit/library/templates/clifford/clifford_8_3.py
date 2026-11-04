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


from qiskit.circuit.quantumcircuit import QuantumCircuit


def clifford_8_3():
    """
    Clifford template 8_3:

    .. code-block:: text

        q_0: ─────────────────■───────────────────────■──
             ┌───┐┌───┐┌───┐┌─┴─┐┌─────┐┌───┐┌─────┐┌─┴─┐
        q_1: ┤ S ├┤ H ├┤ S ├┤ X ├┤ SDG ├┤ H ├┤ SDG ├┤ X ├
             └───┘└───┘└───┘└───┘└─────┘└───┘└─────┘└───┘

    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.s(1)
    qc.h(1)
    qc.s(1)
    qc.cx(0, 1)
    qc.sdg(1)
    qc.h(1)
    qc.sdg(1)
    qc.cx(0, 1)
    return qc
