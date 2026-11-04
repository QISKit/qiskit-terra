# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the CNOTDihedral class.
"""

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import CNOTDihedral


def synth_cnotdihedral_two_qubits(elem: CNOTDihedral) -> QuantumCircuit:
    """Decompose a :class:`.CNOTDihedral` element on a single qubit and two
    qubits into a :class:`.QuantumCircuit`.
    This decomposition has an optimal number of :class:`.CXGate`\\ s.

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    Raises:
        QiskitError: if the element is not 1-qubit or 2-qubit :class:`.CNOTDihedral`.

    References:
        1. Shelly Garion and Andrew W. Cross, *On the structure of the CNOT-Dihedral group*,
           `arXiv:2006.12042 [quant-ph] <https://arxiv.org/abs/2006.12042>`_
    """

    circuit = QuantumCircuit(elem.num_qubits)

    if elem.num_qubits > 2:
        raise QiskitError(
            "Cannot decompose a CNOT-Dihedral element with more than 2 qubits. "
            "use synth_cnotdihedral_full function instead."
        )

    if elem.num_qubits == 1:
        if elem.poly.weight_0 != 0 or elem.linear != [[1]]:
            raise QiskitError("1-qubit element is not CNOT-Dihedral.")
        tpow0 = elem.poly.weight_1[0]
        xpow0 = elem.shift[0]
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow0 == 0 and xpow0 == 0:
            circuit.id(0)
        return circuit

    # case elem.num_qubits == 2:
    if elem.poly.weight_0 != 0:
        raise QiskitError("2-qubit element is not CNOT-Dihedral.")
    weight_1 = elem.poly.weight_1
    weight_2 = elem.poly.weight_2
    linear = elem.linear
    shift = elem.shift

    # CS subgroup
    if (linear == [[1, 0], [0, 1]]).all():
        [xpow0, xpow1] = shift

        # Dihedral class
        if weight_2 == [0]:
            [tpow0, tpow1] = weight_1
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            if tpow0 == 0 and tpow1 == 0 and xpow0 == 0 and xpow1 == 0:
                circuit.id(0)
                circuit.id(1)

        # CS-like class
        if (weight_2 == [2] and xpow0 == xpow1) or (weight_2 == [6] and xpow0 != xpow1):
            tpow0 = (weight_1[0] - 2 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 2 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CS gate is implemented using 2 CX gates
            circuit.p((np.pi / 4), [0])
            circuit.p((np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.p((7 * np.pi / 4), [1])
            circuit.cx(0, 1)

        # CSdg-like class
        if (weight_2 == [6] and xpow0 == xpow1) or (weight_2 == [2] and xpow0 != xpow1):
            tpow0 = (weight_1[0] - 6 * xpow1 - 4 * xpow0 * xpow1) % 8
            tpow1 = (weight_1[1] - 6 * xpow0 - 4 * xpow0 * xpow1) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CSdg gate is implemented using 2 CX gates
            circuit.p((7 * np.pi / 4), [0])
            circuit.p((7 * np.pi / 4), [1])
            circuit.cx(0, 1)
            circuit.p((np.pi / 4), [1])
            circuit.cx(0, 1)

        # CZ-like class
        if weight_2 == [4]:
            tpow0 = (weight_1[0] - 4 * xpow1) % 8
            tpow1 = (weight_1[1] - 4 * xpow0) % 8
            if tpow0 > 0:
                circuit.p((tpow0 * np.pi / 4), [0])
            if xpow0 == 1:
                circuit.x(0)
            if tpow1 > 0:
                circuit.p((tpow1 * np.pi / 4), [1])
            if xpow1 == 1:
                circuit.x(1)
            # CZ gate is implemented using 2 CX gates
            circuit.cz(1, 0)

    # CX01-like class
    if (linear == [[1, 0], [1, 1]]).all():
        xpow0 = shift[0]
        xpow1 = (shift[1] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])

    # CX10-like class
    if (linear == [[1, 1], [0, 1]]).all():
        xpow1 = shift[1]
        xpow0 = (shift[0] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [0])

    # CX01*CX10-like class
    if (linear == [[0, 1], [1, 1]]).all():
        xpow1 = shift[0]
        xpow0 = (shift[1] + xpow1) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])

    # CX10*CX01-like class
    if (linear == [[1, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = (shift[0] + xpow0) % 2
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)
        if m > 0:
            circuit.p((m * np.pi / 4), [0])

    # CX01*CX10*CX01-like class
    if (linear == [[0, 1], [1, 0]]).all():
        xpow0 = shift[1]
        xpow1 = shift[0]
        if xpow0 == xpow1:
            m = ((8 - weight_2[0]) / 2) % 4
            tpow0 = (weight_1[0] - m) % 8
            tpow1 = (weight_1[1] - m) % 8
        else:
            m = (weight_2[0] / 2) % 4
            tpow0 = (weight_1[0] + m) % 8
            tpow1 = (weight_1[1] + m) % 8
        if tpow0 > 0:
            circuit.p((tpow0 * np.pi / 4), [0])
        if xpow0 == 1:
            circuit.x(0)
        if tpow1 > 0:
            circuit.p((tpow1 * np.pi / 4), [1])
        if xpow1 == 1:
            circuit.x(1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        if m > 0:
            circuit.p((m * np.pi / 4), [1])
        circuit.cx(0, 1)

    return circuit
