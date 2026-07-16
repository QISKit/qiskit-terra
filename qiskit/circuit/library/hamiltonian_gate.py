# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Gate described by the time evolution of a Hermitian Hamiltonian operator.
"""

from __future__ import annotations
import math
import typing

from numbers import Number
import numpy as np

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix

from .generalized_gates.unitary import UnitaryGate

if typing.TYPE_CHECKING:
    from qiskit.quantum_info.operators.base_operator import BaseOperator


class HamiltonianGate(Gate):
    r"""Class for representing evolution by a Hamiltonian operator as a gate.

    This gate resolves to a :class:`~.library.UnitaryGate` as :math:`U(t) = \exp(-i t H)`,
    which can be decomposed into basis gates if it is 2 qubits or less, or
    simulated directly in Aer for more qubits.
    """

    def __init__(
        self,
        data: np.ndarray | Gate | BaseOperator,
        time: float | ParameterExpression,
        label: str | None = None,
    ) -> None:
        """
        Args:
            data: A hermitian operator.
            time: Time evolution parameter.
            label: Unitary name for backend [Default: ``None``].

        Raises:
            ValueError: if input data is not an N-qubit unitary operator.
        """
        if hasattr(data, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to np array in case not already an array
        data = np.asarray(data, dtype=complex)
        # Check input is unitary
        if not is_hermitian_matrix(data):
            raise ValueError("Input matrix is not Hermitian.")
        if isinstance(time, Number) and time != np.real(time):
            raise ValueError("Evolution time is not real.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(math.log2(input_dim))
        if input_dim != output_dim or 2**num_qubits != input_dim:
            raise ValueError("Input matrix is not an N-qubit operator.")

        # Store instruction params
        super().__init__("hamiltonian", num_qubits, [data, time], label=label)

    def __eq__(self, other):
        if not isinstance(other, HamiltonianGate):
            return False
        if self.label != other.label:
            return False
        operators_eq = matrix_equal(self.params[0], other.params[0], ignore_phase=False)
        times_eq = self.params[1] == other.params[1]
        return operators_eq and times_eq

    def __array__(self, dtype=None, copy=None):
        """Return matrix for the unitary."""
        import scipy.linalg

        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        try:
            time = float(self.params[1])
        except TypeError as ex:
            raise TypeError(
                f"Unable to generate Unitary matrix for unbound t parameter {self.params[1]}"
            ) from ex
        arr = scipy.linalg.expm(-1j * self.params[0] * time)
        dtype = complex if dtype is None else dtype
        return np.array(arr, dtype=dtype, copy=None)

    def inverse(self, annotated: bool = False):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the conjugate of the Hamiltonian."""
        return HamiltonianGate(np.conj(self.params[0]), -self.params[1])

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return HamiltonianGate(self.params[0], -self.params[1])

    def transpose(self):
        """Return the transpose of the Hamiltonian."""
        return HamiltonianGate(np.transpose(self.params[0]), self.params[1])

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        q = QuantumRegister(self.num_qubits, "q")
        qc = QuantumCircuit(q)
        qc._append(UnitaryGate(self.to_matrix()), q[:], [])
        self.definition = qc

    def validate_parameter(self, parameter):
        """Hamiltonian parameter has to be an ndarray, operator or float."""
        if isinstance(parameter, (float, int, np.ndarray)):
            return parameter
        elif isinstance(parameter, ParameterExpression) and len(parameter.parameters) == 0:
            return float(parameter)
        else:
            raise CircuitError(f"invalid param type {type(parameter)} for gate {self.name}")
