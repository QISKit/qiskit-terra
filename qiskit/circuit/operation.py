# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Operation Mixin."""

from abc import ABC, abstractmethod


class Operation(ABC):
    """Quantum operation interface.

    The minimal interface that any object must fulfil in order to be added to a
    :class:`.QuantumCircuit`.

    Concrete instances of this interface include :class:`~qiskit.circuit.Gate`,
    :class:`~qiskit.circuit.Reset`, :class:`~qiskit.circuit.Barrier`,
    :class:`~qiskit.circuit.Measure`, and operators such as :class:`~qiskit.quantum_info.Clifford`.

    The main purpose is to allow abstract mathematical objects to be added directly onto
    abstract circuits, and for the exact syntheses of these to be determined later, during
    compilation.

    Example:

        Add a Clifford and a Toffoli gate to a :class:`QuantumCircuit`.

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Clifford, random_clifford

           qc = QuantumCircuit(3)
           cliff = random_clifford(2)
           qc.append(cliff, [0, 1])
           qc.ccx(0, 1, 2)
           qc.draw('mpl')
    """

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for operation type."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_qubits(self) -> int:
        """Number of qubits."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_clbits(self) -> int:
        """Number of classical bits."""
        raise NotImplementedError
