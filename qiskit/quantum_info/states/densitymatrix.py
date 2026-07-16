# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
DensityMatrix quantum state class.
"""

from __future__ import annotations
import copy as _copy
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import is_positive_semidefinite_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp

from qiskit._accelerate.pauli_expval import density_expval_pauli_no_x, density_expval_pauli_with_x
from qiskit.quantum_info.states.statevector import Statevector

if TYPE_CHECKING:
    from qiskit import circuit


class DensityMatrix(QuantumState, TolerancesMixin):
    """DensityMatrix class"""

    def __init__(
        self,
        data: np.ndarray | list | QuantumCircuit | circuit.instruction.Instruction | QuantumState,
        dims: int | tuple | list | None = None,
    ):
        """Initialize a density matrix object.

        Args:
            data: A statevector, quantum instruction or an object with a ``to_operator`` or
                ``to_matrix`` method from which the density matrix can be constructed.
                If a vector the density matrix is constructed as the projector of that vector.
                If a quantum instruction, the density matrix is constructed by assuming all
                qubits are initialized in the zero state.
            dims: The subsystem dimension of the state (See additional information).

        Raises:
            QiskitError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg can be None, an integer, or an iterable of
            integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` or ``None`` -- the leading dimension of the input matrix
              specifies the total dimension of the density matrix. If it is a
              power of two the state will be initialized as an N-qubit state.
              If it is not a power of two the state will have a single
              d-dimensional subsystem.
        """
        if isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw matrix in either a
            # python list or numpy array format.
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            # If the data is a circuit or an instruction use the classmethod
            # to construct the DensityMatrix object
            self._data = DensityMatrix.from_instruction(data)._data
        elif hasattr(data, "to_operator"):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            op = data.to_operator()
            self._data = op.data
            if dims is None:
                dims = op.output_dims()
        elif hasattr(data, "to_matrix"):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            self._data = np.asarray(data.to_matrix(), dtype=complex)
        else:
            raise QiskitError("Invalid input data format for DensityMatrix")
        # Convert statevector into a density matrix
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim == 2 and shape[0] == shape[1]:
            pass  # We good
        elif ndim == 1:
            self._data = np.outer(self._data, np.conj(self._data))
        elif ndim == 2 and shape[1] == 1:
            self._data = np.reshape(self._data, shape[0])
        else:
            raise QiskitError("Invalid DensityMatrix input: not a square matrix.")
        super().__init__(op_shape=OpShape.auto(shape=self._data.shape, dims_l=dims, dims_r=dims))

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            self._data, other._data, rtol=self.rtol, atol=self.atol
        )

    def __repr__(self):
        prefix = "DensityMatrix("
        pad = len(prefix) * " "
        return (
            f"{prefix}{np.array2string(self._data, separator=', ', prefix=prefix)},\n"
            f"{pad}dims={self._op_shape.dims_l()})"
        )

    @property
    def settings(self):
        """Return settings."""
        return {"data": self.data, "dims": self._op_shape.dims_l()}

    def draw(self, output: str | None = None, **drawer_args):
        """Return a visualization of the density matrix.

        **repr**: ASCII TextMatrix of the state's ``__repr__``.

        **text**: ASCII TextMatrix that can be printed in the console.

        **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

        **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

        **qsphere**: Matplotlib figure, rendering of density matrix using `plot_state_qsphere()`.

        **hinton**: Matplotlib figure, rendering of density matrix using `plot_state_hinton()`.

        **bloch**: Matplotlib figure, rendering of density matrix using `plot_bloch_multivector()`.

        Args:
            output (str): Select the output method to use for drawing the
                state. Valid choices are `repr`, `text`, `latex`, `latex_source`,
                `qsphere`, `hinton`, or `bloch`. Default is `repr`. Default can
                be changed by adding the line ``state_drawer = <default>`` to
                ``~/.qiskit/settings.conf`` under ``[default]``.
            drawer_args: Arguments to be passed directly to the relevant drawing
                function or constructor (`TextMatrix()`, `array_to_latex()`,
                `plot_state_qsphere()`, `plot_state_hinton()` or `plot_bloch_multivector()`).
                See the relevant function under `qiskit.visualization` for that function's
                documentation.

        Returns:
            :class:`matplotlib.Figure` or :class:`str` or
            :class:`TextMatrix` or :class:`IPython.display.Latex`:
            Drawing of the density matrix.

        Raises:
            ValueError: when an invalid output method is selected.
        """

        from qiskit.visualization.state_visualization import state_drawer

        return state_drawer(self, output=output, **drawer_args)

    def _ipython_display_(self):
        out = self.draw()
        if isinstance(out, str):
            print(out)  # noqa: T201
        else:
            from IPython.display import display

            display(out)

    @property
    def data(self):
        """Return data."""
        return self._data

    def is_valid(self, atol=None, rtol=None):
        """Return True if trace 1 and positive semidefinite."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        # Check trace == 1
        if not np.allclose(self.trace(), 1, rtol=rtol, atol=atol):
            return False
        # Check Hermitian
        if not is_hermitian_matrix(self.data, rtol=rtol, atol=atol):
            return False
        # Check positive semidefinite
        return is_positive_semidefinite_matrix(self.data, rtol=rtol, atol=atol)

    def to_operator(self) -> Operator:
        """Convert to Operator"""
        dims = self.dims()
        return Operator(self.data, input_dims=dims, output_dims=dims)

    def conjugate(self):
        """Return the conjugate of the density matrix."""
        return DensityMatrix(np.conj(self.data), dims=self.dims())

    def trace(self):
        """Return the trace of the density matrix."""
        return np.trace(self.data)

    def purity(self):
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return np.trace(np.dot(self.data, self.data))

    def tensor(self, other: DensityMatrix) -> DensityMatrix:
        """Return the tensor product state self ⊗ other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        ret = _copy.copy(self)
        ret._data = np.kron(self._data, other._data)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        return ret

    def expand(self, other: DensityMatrix) -> DensityMatrix:
        """Return the tensor product state other ⊗ self.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        ret = _copy.copy(self)
        ret._data = np.kron(other._data, self._data)
        ret._op_shape = self._op_shape.expand(other._op_shape)
        return ret

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (DensityMatrix): a quantum state object.

        Returns:
            DensityMatrix: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        if not isinstance(other, DensityMatrix):
            other = DensityMatrix(other)
        self._op_shape._validate_add(other._op_shape)
        ret = _copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        """Return the scalar multiplied state other * self.

        Args:
            other (complex): a complex number.

        Returns:
            DensityMatrix: the scalar multiplied state other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = _copy.copy(self)
        ret._data = other * self.data
        return ret

    def evolve(
        self,
        other: Operator | QuantumChannel | circuit.instruction.Instruction | QuantumCircuit,
        qargs: list[int] | None = None,
    ) -> DensityMatrix:
        """Evolve a quantum state by an operator.

        Args:
            other: The operator to evolve by.
            qargs: a list of QuantumState subsystem positions to apply the operator on.

        Returns:
            The output density matrix.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified QuantumState subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        # Evolution by a circuit or instruction
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)

        # Evolution by a QuantumChannel
        # Currently the class that has `to_quantumchannel` is QuantumError of Qiskit Aer, so we can't
        # use QuantumError as a type hint.
        if hasattr(other, "to_quantumchannel"):
            return other.to_quantumchannel()._evolve(self, qargs=qargs)
        if isinstance(other, QuantumChannel):
            return other._evolve(self, qargs=qargs)

        # Unitary evolution by an Operator
        if not isinstance(other, Operator):
            dims = self.dims(qargs=qargs)
            other = Operator(other, input_dims=dims, output_dims=dims)
        return self._evolve_operator(other, qargs=qargs)

    def reverse_qargs(self) -> DensityMatrix:
        r"""Return a DensityMatrix with reversed subsystem ordering.

        For a tensor product state this is equivalent to reversing the order
        of tensor product subsystems. For a density matrix
        :math:`\rho = \rho_{n-1} \otimes ... \otimes \rho_0`
        the returned state will be
        :math:`\rho_0 \otimes ... \otimes \rho_{n-1}`.

        Returns:
            DensityMatrix: the state with reversed subsystem order.
        """
        ret = _copy.copy(self)
        axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
        axes = axes + tuple(len(axes) + i for i in axes)
        ret._data = np.reshape(
            np.transpose(np.reshape(self.data, self._op_shape.tensor_shape), axes),
            self._op_shape.shape,
        )
        ret._op_shape = self._op_shape.reverse()
        return ret

    def _expectation_value_pauli(self, pauli, qargs=None):
        """Compute the expectation value of a Pauli.

        Args:
            pauli (Pauli): a Pauli operator to evaluate expval of.
            qargs (None or list): subsystems to apply operator on.

        Returns:
            complex: the expectation value.
        """
        n_pauli = len(pauli)
        if qargs is None:
            qubits = np.arange(n_pauli)
        else:
            qubits = np.array(qargs)
        x_mask = np.dot(1 << qubits, pauli.x)
        z_mask = np.dot(1 << qubits, pauli.z)
        pauli_phase = (-1j) ** pauli.phase if pauli.phase else 1

        if x_mask + z_mask == 0:
            return pauli_phase * self.trace()

        data = np.ravel(self.data, order="F")
        if x_mask == 0:
            return pauli_phase * density_expval_pauli_no_x(data, self.num_qubits, z_mask)

        x_max = qubits[pauli.x][-1]
        y_phase = (-1j) ** pauli._count_y()
        y_phase = y_phase[0]
        return pauli_phase * density_expval_pauli_with_x(
            data, self.num_qubits, z_mask, x_mask, y_phase, x_max
        )

    def expectation_value(self, oper: Operator, qargs: None | list[int] = None) -> complex:
        """Compute the expectation value of an operator.

        Args:
            oper (Operator): an operator to evaluate expval.
            qargs (None or list): subsystems to apply the operator on.

        Returns:
            complex: the expectation value.
        """
        if isinstance(oper, Pauli):
            return self._expectation_value_pauli(oper, qargs)

        if isinstance(oper, SparsePauliOp):
            return sum(
                coeff * self._expectation_value_pauli(Pauli((z, x)), qargs)
                for z, x, coeff in zip(oper.paulis.z, oper.paulis.x, oper.coeffs)
            )

        if not isinstance(oper, Operator):
            oper = Operator(oper)
        return np.trace(Operator(self).dot(oper, qargs=qargs).data)

    def probabilities(
        self, qargs: None | list[int] = None, decimals: None | int = None
    ) -> np.ndarray:
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.

        Examples:

            Consider a 2-qubit product state :math:`\\rho=\\rho_1\\otimes\\rho_0`
            with :math:`\\rho_1=|+\\rangle\\!\\langle+|`,
            :math:`\\rho_0=|0\\rangle\\!\\langle0|`.

            .. plot::
               :include-source:
               :nofigs:

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('+0')

                # Probabilities for measuring both qubits
                probs = rho.probabilities()
                print('probs: {}'.format(probs))

                # Probabilities for measuring only qubit-0
                probs_qubit_0 = rho.probabilities([0])
                print('Qubit-0 probs: {}'.format(probs_qubit_0))

                # Probabilities for measuring only qubit-1
                probs_qubit_1 = rho.probabilities([1])
                print('Qubit-1 probs: {}'.format(probs_qubit_1))

            .. code-block:: text

                probs: [0.5 0.  0.5 0. ]
                Qubit-0 probs: [1. 0.]
                Qubit-1 probs: [0.5 0.5]

            We can also permute the order of qubits in the ``qargs`` list
            to change the qubit position in the probabilities output

            .. plot::
               :include-source:
               :nofigs:

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('+0')

                # Probabilities for measuring both qubits
                probs = rho.probabilities([0, 1])
                print('probs: {}'.format(probs))

                # Probabilities for measuring both qubits
                # but swapping qubits 0 and 1 in output
                probs_swapped = rho.probabilities([1, 0])
                print('Swapped probs: {}'.format(probs_swapped))

            .. code-block:: text

                probs: [0.5 0.  0.5 0. ]
                Swapped probs: [0.5 0.5 0.  0. ]
        """
        probs = self._subsystem_probabilities(
            np.abs(self.data.diagonal()), self._op_shape.dims_l(), qargs=qargs
        )

        # to account for roundoff errors, we clip
        probs = np.clip(probs, a_min=0, a_max=1)

        if decimals is not None:
            probs = probs.round(decimals=decimals)

        return probs

    def reset(self, qargs: list[int] | None = None) -> DensityMatrix:
        """Reset state or subsystems to the 0-state.

        Args:
            qargs (list or None): subsystems to reset, if None all
                                  subsystems will be reset to their 0-state
                                  (Default: None).

        Returns:
            DensityMatrix: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only some subsystems are reset this
            function will perform evolution by the reset
            :class:`~qiskit.quantum_info.SuperOp` of the reset subsystems.
        """
        if qargs is None:
            # Resetting all qubits does not require sampling or RNG
            ret = _copy.copy(self)
            state = np.zeros(self._op_shape.shape, dtype=complex)
            state[0, 0] = 1
            ret._data = state
            return ret

        # Reset by evolving by reset SuperOp
        dims = self.dims(qargs)
        reset_superop = SuperOp(ScalarOp(dims, coeff=0))
        reset_superop.data[0] = Operator(ScalarOp(dims)).data.ravel()
        return self.evolve(reset_superop, qargs=qargs)

    @classmethod
    def from_label(cls, label: str) -> DensityMatrix:
        r"""Return a tensor product of Pauli X,Y,Z eigenstates.

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label
             - Statevector
           * - ``"0"``
             - :math:`\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}`
           * - ``"1"``
             - :math:`\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}`
           * - ``"+"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}`
           * - ``"-"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}`
           * - ``"r"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & -i \\ i & 1 \end{pmatrix}`
           * - ``"l"``
             - :math:`\frac{1}{2}\begin{pmatrix} 1 & i \\ -i & 1 \end{pmatrix}`

        Args:
            label (string): an eigenstate string ket label (see table for
                            allowed values).

        Returns:
            DensityMatrix: The N-qubit basis state density matrix.

        Raises:
            QiskitError: if the label contains invalid characters, or the length
                         of the label is larger than an explicitly specified num_qubits.
        """

        return DensityMatrix(Statevector.from_label(label))

    @staticmethod
    def from_int(i: int, dims: int | tuple | list) -> DensityMatrix:
        """Return a computational basis state density matrix.

        Args:
            i (int): the basis state element.
            dims (int or tuple or list): The subsystem dimensions of the statevector
                                         (See additional information).

        Returns:
            DensityMatrix: The computational basis state :math:`|i\\rangle\\!\\langle i|`.

        Additional Information:
            The ``dims`` kwarg can be an integer or an iterable of integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` -- the integer specifies the total dimension of the
              state. If it is a power of two the state will be initialized
              as an N-qubit state. If it is not a power of two the state
              will have a single d-dimensional subsystem.
        """
        size = np.prod(dims)
        state = np.zeros((size, size), dtype=complex)
        state[i, i] = 1.0
        return DensityMatrix(state, dims=dims)

    @classmethod
    def from_instruction(
        cls, instruction: circuit.instruction.Instruction | QuantumCircuit
    ) -> DensityMatrix:
        """Return the output density matrix of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of
        the same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction: instruction or circuit

        Returns:
            The final density matrix.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         density matrix simulation.
        """
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        num_qubits = instruction.num_qubits
        init = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        init[0, 0] = 1
        vec = DensityMatrix(init, dims=num_qubits * (2,))
        vec._append_instruction(instruction)
        return vec

    def to_dict(self, decimals: None | int = None) -> dict:
        r"""Convert the density matrix to dictionary form.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            decimals (None or int): the number of decimal places to round
                                    values. If None no rounding is done
                                    (Default: None).

        Returns:
            dict: the dictionary form of the DensityMatrix.

        Examples:

            The ket-form of a 2-qubit density matrix
            :math:`rho = |-\rangle\!\langle -|\otimes |0\rangle\!\langle 0|`

            .. plot::
               :include-source:
               :nofigs:

                from qiskit.quantum_info import DensityMatrix

                rho = DensityMatrix.from_label('-0')
                print(rho.to_dict())

            .. code-block:: text

               {
                   '00|00': (0.4999999999999999+0j),
                   '10|00': (-0.4999999999999999-0j),
                   '00|10': (-0.4999999999999999+0j),
                   '10|10': (0.4999999999999999+0j)
               }

            For non-qubit subsystems the integer range can go from 0 to 9. For
            example in a qutrit system

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit.quantum_info import DensityMatrix

                mat = np.zeros((9, 9))
                mat[0, 0] = 0.25
                mat[3, 3] = 0.25
                mat[6, 6] = 0.25
                mat[-1, -1] = 0.25
                rho = DensityMatrix(mat, dims=(3, 3))
                print(rho.to_dict())

            .. code-block:: text

                {'00|00': (0.25+0j), '10|10': (0.25+0j), '20|20': (0.25+0j), '22|22': (0.25+0j)}

            For large subsystem dimensions delimiters are required. The
            following example is for a 20-dimensional system consisting of
            a qubit and 10-dimensional qudit.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit.quantum_info import DensityMatrix

                mat = np.zeros((2 * 10, 2 * 10))
                mat[0, 0] = 0.5
                mat[-1, -1] = 0.5
                rho = DensityMatrix(mat, dims=(2, 10))
                print(rho.to_dict())

            .. code-block:: text

                {'00|00': (0.5+0j), '91|91': (0.5+0j)}
        """
        return self._matrix_to_dict(
            self.data, self._op_shape.dims_l(), decimals=decimals, string_labels=True
        )

    def _evolve_operator(self, other, qargs=None):
        """Evolve density matrix by an operator"""
        # Get shape of output density matrix
        new_shape = self._op_shape.compose(other._op_shape, qargs=qargs)
        new_shape._dims_r = new_shape._dims_l
        new_shape._num_qargs_r = new_shape._num_qargs_l

        ret = _copy.copy(self)
        if qargs is None:
            # Evolution on full matrix
            op_mat = other.data
            ret._data = np.dot(op_mat, self.data).dot(op_mat.T.conj())
            ret._op_shape = new_shape
            return ret

        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._op_shape.tensor_shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        # Left multiple by mat
        mat = np.reshape(other.data, other._op_shape.tensor_shape)
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        # Right multiply by mat ** dagger
        adj = other.adjoint()
        mat_adj = np.reshape(adj.data, adj._op_shape.tensor_shape)
        tensor = Operator._einsum_matmul(tensor, mat_adj, indices, num_indices, True)
        # Replace evolved dimensions
        ret._data = np.reshape(tensor, new_shape.shape)
        ret._op_shape = new_shape
        return ret

    def _append_instruction(self, other, qargs=None):
        """Update the current Statevector by applying an instruction."""
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier

        # Try evolving by a matrix operator (unitary-like evolution)
        mat = Operator._instruction_to_matrix(other)
        if mat is not None:
            self._data = self._evolve_operator(Operator(mat), qargs=qargs).data
            return

        # Special instruction types
        if isinstance(other, Reset):
            self._data = self.reset(qargs)._data
            return
        if isinstance(other, Barrier):
            return

        # Otherwise try evolving by a Superoperator
        chan = SuperOp._instruction_to_superop(other)
        if chan is not None:
            # Evolve current state by the superoperator
            self._data = chan._evolve(self, qargs=qargs).data
            return
        # If the instruction doesn't have a matrix defined we use its
        # circuit decomposition definition if it exists, otherwise we
        # cannot compose this gate and raise an error.
        if other.definition is None:
            raise QiskitError(f"Cannot apply Instruction: {other.name}")
        if not isinstance(other.definition, QuantumCircuit):
            raise QiskitError(
                f"{other.name} instruction definition is {type(other.definition)};"
                f" expected QuantumCircuit"
            )
        qubit_indices = {bit: idx for idx, bit in enumerate(other.definition.qubits)}
        for instruction in other.definition:
            if instruction.clbits:
                raise QiskitError(
                    f"Cannot apply instruction with classical bits: {instruction.operation.name}"
                )
            # Get the integer position of the flat register
            if qargs is None:
                new_qargs = [qubit_indices[tup] for tup in instruction.qubits]
            else:
                new_qargs = [qargs[qubit_indices[tup]] for tup in instruction.qubits]
            self._append_instruction(instruction.operation, qargs=new_qargs)

    def _evolve_instruction(self, obj, qargs=None):
        """Return a new statevector by applying an instruction."""
        if isinstance(obj, QuantumCircuit):
            obj = obj.to_instruction()
        vec = _copy.copy(self)
        vec._append_instruction(obj, qargs=qargs)
        return vec

    def to_statevector(self, atol: float | None = None, rtol: float | None = None) -> Statevector:
        """Return a statevector from a pure density matrix.

        Args:
            atol (float): Absolute tolerance for checking operation validity.
            rtol (float): Relative tolerance for checking operation validity.

        Returns:
            Statevector: The pure density matrix's corresponding statevector.
                Corresponds to the eigenvector of the only non-zero eigenvalue.

        Raises:
            QiskitError: if the state is not pure.
        """

        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        if not is_hermitian_matrix(self._data, atol=atol, rtol=rtol):
            raise QiskitError("Not a valid density matrix (non-hermitian).")

        evals, evecs = np.linalg.eig(self._data)

        nonzero_evals = evals[abs(evals) > atol]
        if len(nonzero_evals) != 1 or not np.isclose(nonzero_evals[0], 1, atol=atol, rtol=rtol):
            raise QiskitError("Density matrix is not a pure state")

        psi = evecs[:, np.argmax(evals)]  # eigenvectors returned in columns.
        return Statevector(psi)

    def partial_transpose(self, qargs: list[int]) -> DensityMatrix:
        """Return partially transposed density matrix.

        Args:
            qargs (list): The subsystems to be transposed.

        Returns:
            DensityMatrix: The partially transposed density matrix.
        """
        arr = self._data.reshape(self._op_shape.tensor_shape)
        qargs = len(self._op_shape.dims_l()) - 1 - np.array(qargs)
        n = len(self.dims())
        lst = list(range(2 * n))
        for i in qargs:
            lst[i], lst[i + n] = lst[i + n], lst[i]
        rho = np.transpose(arr, lst)
        rho = np.reshape(rho, self._op_shape.shape)
        return DensityMatrix(rho, dims=self.dims())
