# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Arbitrary unitary circuit instruction."""

from __future__ import annotations
import math

import typing
import numpy

from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import AnnotatedOperation, ControlModifier
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

from .isometry import Isometry

if typing.TYPE_CHECKING:
    from qiskit.quantum_info.operators.base_operator import BaseOperator


class UnitaryGate(Gate):
    """Class for quantum gates specified by a unitary matrix.

    Example:

    We can create a unitary gate from a unitary matrix then add it to a
    quantum circuit. The matrix can also be directly applied to the quantum
    circuit, see :meth:`.QuantumCircuit.unitary`.

    .. plot::
        :include-source:
        :nofigs:

        from qiskit import QuantumCircuit
        from qiskit.circuit.library import UnitaryGate

        matrix = [[0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]]
        gate = UnitaryGate(matrix)

        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
    """

    def __init__(
        self,
        data: numpy.ndarray | Gate | BaseOperator,
        label: str | None = None,
        check_input: bool = True,
        *,
        num_qubits: int | None = None,
    ) -> None:
        """
        Args:
            data: Unitary operator.
            label: Unitary name for backend [Default: ``None``].
            check_input: If set to ``False`` this asserts the input
                is known to be unitary and the checking to validate this will
                be skipped. This should only ever be used if you know the
                input is unitary, setting this to ``False`` and passing in
                a non-unitary matrix will result unexpected behavior and errors.
            num_qubits: If given, the number of qubits in the matrix.  If not given, it is inferred.

        Raises:
            ValueError: If input data is not an N-qubit unitary operator.
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
        # Convert to numpy array in case not already an array
        data = numpy.asarray(data, dtype=complex)
        input_dim, output_dim = data.shape
        num_qubits = num_qubits if num_qubits is not None else int(math.log2(input_dim))
        if check_input:
            # Check input is unitary
            if not is_unitary_matrix(data):
                raise ValueError("Input matrix is not unitary.")
            # Check input is N-qubit matrix
            if input_dim != output_dim or 2**num_qubits != input_dim:
                raise ValueError("Input matrix is not an N-qubit operator.")
        # Store instruction params
        super().__init__("unitary", num_qubits, [data], label=label)

    def __eq__(self, other):
        if not isinstance(other, UnitaryGate):
            return False
        if self.label != other.label:
            return False
        return matrix_equal(self.params[0], other.params[0])

    def __array__(self, dtype=None, copy=None):
        """Return matrix for the unitary."""
        dtype = self.params[0].dtype if dtype is None else dtype
        return numpy.array(self.params[0], dtype=dtype, copy=copy)

    def inverse(self, annotated: bool = False):
        """Return the adjoint of the unitary."""
        return self.adjoint()

    def conjugate(self):
        """Return the complex conjugate of the unitary with respect to the
        representation returned by :meth:`~UnitaryGate.to_matrix`."""
        return UnitaryGate(numpy.conj(self.to_matrix()))

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return self.transpose().conjugate()

    def transpose(self):
        """Return the transpose of the unitary."""
        return UnitaryGate(numpy.transpose(self.to_matrix()))

    def _define(self):
        """Calculate a subcircuit that implements this unitary."""
        if self.num_qubits == 1:
            from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer

            q = QuantumRegister(1, "q")
            qc = QuantumCircuit(q)
            theta, phi, lam, global_phase = OneQubitEulerDecomposer("U").angles_and_phase(
                self.to_matrix()
            )
            qc._append(UGate(theta, phi, lam), [q[0]], [])
            qc.global_phase = global_phase
            self.definition = qc
        elif self.num_qubits == 2:
            from qiskit.synthesis.two_qubit.two_qubit_decompose import (
                two_qubit_cnot_decompose,
            )

            self.definition = two_qubit_cnot_decompose(self.to_matrix())
        else:

            mat = self.to_matrix()

            from qiskit.synthesis.unitary.qsd import qs_decomposition

            try:
                # The Rust code for Quantum Shannon Decomposition may return QiskitError
                # when linear algebra methods (e.g. Schur decomposition) fail.
                mat_def = qs_decomposition(mat)
            except QiskitError:
                mat_def = None

            # If QSD fails or provides numerically imprecise matrices, fallback on the
            # Isometry decomposition (which produces more gates but is more numerically
            # stable)
            from qiskit.quantum_info.operators import Operator

            if (mat_def is None) or not (
                matrix_equal(Operator(mat_def).to_matrix(), mat, atol=1e-7)
            ):
                mat_def = Isometry(self.to_matrix(), 0, 0).definition

            self.definition = mat_def

    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ) -> ControlledGate | AnnotatedOperation:
        """Return a controlled version of itself.

        The controlled gate is implemented as :class:`.ControlledGate` when ``annotated``
        is ``False``, and as :class:`.AnnotatedOperation` when ``annotated`` is ``True``.

        Args:
            num_ctrl_qubits: Number of controls to add. Defaults to ``1``.
            label: Optional gate label. Defaults to ``None``. Ignored if the controlled gate
                is implemented as an annotated operation.
            ctrl_state: The control state of the gate, specified either as an integer or a bitstring
                (e.g. ``"110"``). If ``None``, defaults to the all-ones state ``2**num_ctrl_qubits - 1``.
            annotated: Indicates whether the controlled gate should be implemented as a controlled gate
                or as an annotated operation. If ``None``, treated as ``False``.

        Returns:
            A controlled version of this gate.
        """
        if not annotated:
            mat = self.to_matrix()
            cmat = _compute_control_matrix(mat, num_ctrl_qubits, ctrl_state=None)
            from qiskit.synthesis.unitary.qsd import qs_decomposition

            try:
                # The Rust code for Quantum Shannon Decomposition may return QiskitError
                # when linear algebra methods (e.g. Schur decomposition) fail.
                cmat_def = qs_decomposition(cmat)
            except QiskitError:
                cmat_def = None

            # If QSD fails or provides numerically imprecise matrices, fallback on the
            # Isometry decomposition (which produces more gates but is more numerically
            # stable)
            from qiskit.quantum_info.operators import Operator

            if (cmat_def is None) or not (
                matrix_equal(Operator(cmat_def).to_matrix(), cmat, atol=1e-7)
            ):
                cmat_def = Isometry(cmat, 0, 0).definition

            gate = ControlledGate(
                "c-unitary",
                num_qubits=self.num_qubits + num_ctrl_qubits,
                params=[mat],
                label=label,
                num_ctrl_qubits=num_ctrl_qubits,
                definition=cmat_def,
                ctrl_state=ctrl_state,
                base_gate=self.copy(),
            )
        else:
            gate = AnnotatedOperation(
                self, ControlModifier(num_ctrl_qubits=num_ctrl_qubits, ctrl_state=ctrl_state)
            )
        return gate

    def _qasm_decomposition(self):
        """Return an unparameterized version of ourselves, so the OQ2 exporter doesn't choke on the
        non-standard things in our `params` field."""
        out = self.definition.to_gate()
        out.name = self.name
        return out

    def validate_parameter(self, parameter):
        """Unitary gate parameter has to be an ndarray."""
        if isinstance(parameter, numpy.ndarray):
            return parameter
        else:
            raise CircuitError(f"invalid param type {type(parameter)} in gate {self.name}")
