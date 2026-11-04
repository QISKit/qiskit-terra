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


def synth_cnotdihedral_general(elem: CNOTDihedral) -> QuantumCircuit:
    """Decompose a :class:`.CNOTDihedral` element into a :class:`.QuantumCircuit`.

    Decompose a general :class:`.CNOTDihedral` elements.
    The number of CX gates is not necessarily optimal.
    For a decomposition of a 1-qubit or 2-qubit element, call
    :func:`.synth_cnotdihedral_two_qubits`.

    Args:
        elem: A :class:`.CNOTDihedral` element.

    Returns:
        A circuit implementation of the :class:`.CNOTDihedral` element.

    Raises:
        QiskitError: if the element could not be decomposed into a circuit.

    References:
        1. Andrew W. Cross, Easwar Magesan, Lev S. Bishop, John A. Smolin and Jay M. Gambetta,
           *Scalable randomized benchmarking of non-Clifford gates*,
           npj Quantum Inf 2, 16012 (2016).
    """

    num_qubits = elem.num_qubits
    circuit = QuantumCircuit(num_qubits)

    # Make a copy of the CNOTDihedral element as we are going to
    # reduce it to an identity
    elem_cpy = elem.copy()

    if not np.allclose((np.linalg.det(elem_cpy.linear) % 2), 1):
        raise QiskitError("Linear part is not invertible.")

    # Do x gate for each qubit i where shift[i]=1
    for i in range(num_qubits):
        if elem.shift[i]:
            circuit.x(i)
            elem_cpy._append_x(i)

    # Do Gauss elimination on the linear part by adding cx gates
    for i in range(num_qubits):
        # set i-th element to be 1
        if not elem_cpy.linear[i][i]:
            for j in range(i + 1, num_qubits):
                if elem_cpy.linear[j][i]:  # swap qubits i and j
                    circuit.cx(j, i)
                    circuit.cx(i, j)
                    circuit.cx(j, i)
                    elem_cpy._append_cx(j, i)
                    elem_cpy._append_cx(i, j)
                    elem_cpy._append_cx(j, i)
                    break
        # make all the other elements in column i zero
        for j in range(num_qubits):
            if j != i:
                if elem_cpy.linear[j][i]:
                    circuit.cx(i, j)
                    elem_cpy._append_cx(i, j)

    if (
        not (elem_cpy.shift == np.zeros(num_qubits)).all()
        or not (elem_cpy.linear == np.eye(num_qubits)).all()
    ):
        raise QiskitError("Cannot do Gauss elimination on linear part.")

    # Initialize new_elem to an identity CNOTDihedral element
    new_elem = elem_cpy.copy()
    new_elem.poly.weight_0 = 0
    new_elem.poly.weight_1 = np.zeros(num_qubits, dtype=np.int8)
    new_elem.poly.weight_2 = np.zeros(int(num_qubits * (num_qubits - 1) / 2), dtype=np.int8)
    new_elem.poly.weight_3 = np.zeros(
        int(num_qubits * (num_qubits - 1) * (num_qubits - 2) / 6), dtype=np.int8
    )

    new_circuit = QuantumCircuit(num_qubits)

    # Do cx and phase gates to construct all monomials of weight 3
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            for k in range(j + 1, num_qubits):
                if elem_cpy.poly.get_term([i, j, k]) != 0:
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_elem._append_phase(1, k)
                    new_elem._append_cx(i, k)
                    new_elem._append_cx(j, k)
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)
                    new_circuit.p((np.pi / 4), [k])
                    new_circuit.cx(i, k)
                    new_circuit.cx(j, k)

    # Do cx and phase gates to construct all monomials of weight 2
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            tpow1 = elem_cpy.poly.get_term([i, j])
            tpow2 = new_elem.poly.get_term([i, j])
            tpow = ((tpow2 - tpow1) / 2) % 4
            if tpow != 0:
                new_elem._append_cx(i, j)
                new_elem._append_phase(tpow, j)
                new_elem._append_cx(i, j)
                new_circuit.cx(i, j)
                new_circuit.p((tpow * np.pi / 4), [j])
                new_circuit.cx(i, j)

    # Do phase gates to construct all monomials of weight 1
    for i in range(num_qubits):
        tpow1 = elem_cpy.poly.get_term([i])
        tpow2 = new_elem.poly.get_term([i])
        tpow = (tpow1 - tpow2) % 8
        if tpow != 0:
            new_elem._append_phase(tpow, i)
            new_circuit.p((tpow * np.pi / 4), [i])

    if elem.poly != new_elem.poly:
        raise QiskitError("Could not recover phase polynomial.")

    inv_circuit = circuit.inverse()
    return new_circuit.compose(inv_circuit)
