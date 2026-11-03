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
from qiskit.quantum_info import CNOTDihedral

from qiskit._accelerate.synthesis.cnotdihedral import (
    synth_cnotdihedral_general as synth_cnotdihedral_general_inner,
)


def _dihedral_parts(elem: CNOTDihedral) -> tuple:
    """Unpack a ``CNOTDihedral`` element into the arrays expected by the Rust synthesis
    routines: the binary ``linear`` matrix and ``shift`` vector of the affine part, and
    the phase polynomial coefficients reduced modulo 8."""
    return (
        np.asarray(elem.linear, dtype=bool),
        np.asarray(elem.shift, dtype=bool),
        int(elem.poly.weight_0) % 8,
        np.mod(np.asarray(elem.poly.weight_1), 8).astype(np.uint8),
        np.mod(np.asarray(elem.poly.weight_2), 8).astype(np.uint8),
        np.mod(np.asarray(elem.poly.weight_3), 8).astype(np.uint8),
    )


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
    return QuantumCircuit._from_circuit_data(
        synth_cnotdihedral_general_inner(*_dihedral_parts(elem)), legacy_qubits=True
    )
