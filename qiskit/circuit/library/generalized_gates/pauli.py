# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Simulator command to perform multiple pauli gates in a single pass
"""

from __future__ import annotations

import typing
from qiskit.circuit.quantumcircuitdata import CircuitInstruction
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.circuit.library.standard_gates.y import YGate
from qiskit.circuit.library.standard_gates.z import ZGate

from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError

if typing.TYPE_CHECKING:
    from qiskit.quantum_info import SparseObservable


class PauliGate(Gate):
    r"""A multi-qubit Pauli gate.

    This gate exists for optimization purposes for the
    quantum statevector simulation, since applying multiple
    pauli gates to different qubits at once can be done via
    a single pass on the statevector.

    The functionality is equivalent to applying
    the pauli gates sequentially using standard Qiskit gates.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.pauli` method.
    """

    def __init__(self, label):
        super().__init__("pauli", len(label), [label])

    def _define(self):
        """
        gate pauli (p1 a1,...,pn an) { p1 a1; ... ; pn an; }
        """

        from qiskit.circuit import QuantumCircuit, QuantumRegister

        gates = {"X": XGate, "Y": YGate, "Z": ZGate}
        q = QuantumRegister(len(self.params[0]), "q")
        qc = QuantumCircuit(q, name=f"{self.name}({self.params[0]})")

        paulis = self.params[0]
        for i, p in enumerate(reversed(paulis)):
            if p == "I":
                continue
            qc._append(CircuitInstruction(gates[p](), (q[i],), ()))
        self.definition = qc

    def inverse(self, annotated: bool = False):
        r"""Return inverted pauli gate (itself)."""
        return PauliGate(self.params[0])  # self-inverse

    def __array__(self, dtype=None, copy=None):
        """Return a Numpy.array for the pauli gate.
        i.e. tensor product of the paulis"""

        from qiskit.quantum_info.operators import Pauli

        return Pauli(self.params[0]).__array__(dtype=dtype, copy=copy)

    def validate_parameter(self, parameter):
        if isinstance(parameter, str):
            if all(c in ["I", "X", "Y", "Z"] for c in parameter):
                return parameter
            else:
                raise CircuitError(
                    f"Parameter string {parameter} should contain only 'I', 'X', 'Y', 'Z' characters"
                )
        else:
            raise CircuitError(
                f"Parameter {parameter} should be a string of 'I', 'X', 'Y', 'Z' characters"
            )

    def _extract_sparse_observable(self) -> SparseObservable:
        from qiskit.quantum_info import SparseObservable

        # The label was already validated, so we know it only contains IXYZ chars.
        return SparseObservable(self.params[0])
