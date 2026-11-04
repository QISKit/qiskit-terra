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


def template_nct_5a_4():
    """
    Template 5a_4:

    .. code-block:: text

                  ┌───┐     ┌───┐
        q_0: ──■──┤ X ├──■──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐├───┤
        q_1: ┤ X ├─────┤ X ├┤ X ├
             └───┘     └───┘└───┘

    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.x(1)
    return qc
