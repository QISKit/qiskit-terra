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


def template_nct_4a_1():
    """
    Template 4a_1:

    .. code-block:: text

        q_0: ───────■─────────■──
                    │         │
        q_1: ──■────┼────■────┼──
               │    │    │    │
        q_2: ──■────■────■────■──
               │  ┌─┴─┐  │  ┌─┴─┐
        q_3: ──┼──┤ X ├──┼──┤ X ├
             ┌─┴─┐└───┘┌─┴─┐└───┘
        q_4: ┤ X ├─────┤ X ├─────
             └───┘     └───┘

    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(5)
    qc.ccx(1, 2, 4)
    qc.ccx(0, 2, 3)
    qc.ccx(1, 2, 4)
    qc.ccx(0, 2, 3)
    return qc
