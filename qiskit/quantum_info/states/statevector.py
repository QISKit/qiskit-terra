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
Statevector quantum state class.
"""
from __future__ import annotations
import copy as _copy
import math
import re
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.operators.mixins.tolerances import TolerancesMixin
from qiskit.quantum_info.operators.operator import Operator, BaseOperator
from qiskit.quantum_info.operators.symplectic import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.predicates import matrix_equal

from qiskit._accelerate.pauli_expval import (
    expval_pauli_no_x,
    expval_pauli_with_x,
)

if TYPE_CHECKING:
    from qiskit import circuit


class Statevector(QuantumState, TolerancesMixin):
    """Statevector class"""

    def __init__(
        self,
        data: (
            np.ndarray
            | list
            | Statevector
            | Operator
            | QuantumCircuit
            | circuit.instruction.Instruction
        ),
        dims: int | tuple | list | None = None,
    ):
        """Initialize a statevector object.

        Args:
            data: Data from which the statevector can be constructed. This can be either a complex
                vector, another statevector, a ``Operator`` with only one column or a
                ``QuantumCircuit`` or ``Instruction``.  If the data is a circuit or instruction,
                the statevector is constructed by assuming that all qubits are initialized to the
                zero state.
            dims: The subsystem dimension of the state (See additional information).

        Raises:
            QiskitError: if input data is not valid.

        Additional Information:
            The ``dims`` kwarg can be None, an integer, or an iterable of
            integers.

            * ``Iterable`` -- the subsystem dimensions are the values in the list
              with the total number of subsystems given by the length of the list.

            * ``Int`` or ``None`` -- the length of the input vector
              specifies the total dimension of the state. If it is a
              power of two the state will be initialized as an N-qubit state.
              If it is not a power of two the state will have a single
              d-dimensional subsystem.
        """
        if isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw vector in either a
            # python list or numpy array format.
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, Statevector):
            self._data = data._data
            if dims is None:
                dims = data._op_shape._dims_l
        elif isinstance(data, Operator):
            # We allow conversion of column-vector operators to Statevectors
            input_dim, _ = data.dim
            if input_dim != 1:
                raise QiskitError("Input Operator is not a column-vector.")
            self._data = np.ravel(data.data)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._data = Statevector.from_instruction(data).data
        else:
            raise QiskitError("Invalid input data format for Statevector")
        # Check that the input is a numpy vector or column-vector numpy
        # matrix. If it is a column-vector matrix reshape to a vector.
        ndim = self._data.ndim
        shape = self._data.shape
        if ndim != 1:
            if ndim == 2 and shape[1] == 1:
                self._data = np.reshape(self._data, shape[0])
                shape = self._data.shape
            elif ndim != 2 or shape[1] != 1:
                raise QiskitError("Invalid input: not a vector or column-vector.")
        super().__init__(op_shape=OpShape.auto(shape=shape, dims_l=dims, num_qubits_r=0))

    @classmethod
    def from_circuit(cls, circuit: QuantumCircuit, ignore_set_layout: bool = False) -> Statevector:
        """Create a Statevector from a quantum circuit.

        Args:
            circuit (QuantumCircuit): A quantum circuit
            ignore_set_layout (bool): When set to ``True``, if the input ``circuit``
                has a layout set, it will be ignored. Defaults to ``False``.

        Returns:
            The statevector obtained by applying the circuit on the all-zero
            input state.

        Example:
            Create a statevector from a transpiled circuit:

            .. plot::
               :include-source:
               :nofigs:

                from qiskit import QuantumCircuit, transpile
                from qiskit.quantum_info import Statevector
                from qiskit.providers.basic_provider import BasicSimulator

                qc = QuantumCircuit(3)
                qc.h(0)
                qc.cx(0, 1)
                qc.swap(1, 2)

                backend = BasicSimulator()
                transpiled1 = transpile(qc, backend, optimization_level=0)
                transpiled2 = transpile(qc, backend, optimization_level=2)

                # Get statevectors accounting for layout changes
                sv1 = Statevector.from_circuit(transpiled1)
                sv2 = Statevector.from_circuit(transpiled2)

                # These will be equivalent up to global phase
                print(sv1.equiv(sv2))  # True
        """

        # Handle layout extraction
        layout = None
        if not ignore_set_layout:
            layout = circuit.layout
        # Create statevector using from_instruction (which iterates through circuit)
        statevec = cls.from_instruction(circuit)

        # Apply layout permutations if needed (this handles transpiler layout changes)
        if not ignore_set_layout and layout is not None:
            layout_indices = layout.final_index_layout()
            # Apply LSb0 qubit ordering (least-significant-bit is 0)
            num_qubits = statevec.num_qubits
            reversed_perm = [
                num_qubits - 1 - layout_indices[num_qubits - 1 - i] for i in range(num_qubits)
            ]

            # Use NumPy tensor reordering to apply the permutation efficiently
            # Reshape to tensor (one dimension per qubit)
            data_tensor = np.reshape(statevec._data, (2,) * num_qubits)
            # Apply permutation via transpose
            permuted_tensor = np.transpose(data_tensor, reversed_perm)
            # Flatten back to statevector
            statevec._data = np.reshape(permuted_tensor, -1)

        return statevec

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    def __eq__(self, other):
        return super().__eq__(other) and np.allclose(
            self._data, other._data, rtol=self.rtol, atol=self.atol
        )

    def __repr__(self):
        prefix = "Statevector("
        pad = len(prefix) * " "
        return (
            f"{prefix}{np.array2string(self._data, separator=', ', prefix=prefix)},\n{pad}"
            f"dims={self._op_shape.dims_l()})"
        )

    @property
    def settings(self) -> dict:
        """Return settings."""
        return {"data": self._data, "dims": self._op_shape.dims_l()}

    def draw(self, output: str | None = None, **drawer_args):
        """Return a visualization of the Statevector.

        **repr**: ASCII TextMatrix of the state's ``__repr__``.

        **text**: ASCII TextMatrix that can be printed in the console.

        **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

        **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

        **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.

        **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.

        **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.

        **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.

        **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.

        Args:
            output (str): Select the output method to use for drawing the
                state. Valid choices are `repr`, `text`, `latex`, `latex_source`,
                `qsphere`, `hinton`, `bloch`, `city`, or `paulivec`. Default is `repr`.
                Default can be changed by adding the line ``state_drawer = <default>`` to
                ``~/.qiskit/settings.conf`` under ``[default]``.
            drawer_args: Arguments to be passed directly to the relevant drawing
                function or constructor (`TextMatrix()`, `array_to_latex()`,
                `plot_state_qsphere()`, `plot_state_hinton()` or `plot_bloch_multivector()`).
                See the relevant function under `qiskit.visualization` for that function's
                documentation.

        Returns:
            :class:`matplotlib.Figure` or :class:`str` or
            :class:`TextMatrix` or :class:`IPython.display.Latex`:
            Drawing of the Statevector.

        Raises:
            ValueError: when an invalid output method is selected.

        Examples:

            Plot one of the Bell states

            .. plot::
               :alt: Output from the previous code.
               :include-source:

                from numpy import sqrt
                from qiskit.quantum_info import Statevector
                sv=Statevector([1/sqrt(2), 0, 0, -1/sqrt(2)])
                sv.draw(output='hinton')

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

    def __getitem__(self, key: int | str) -> np.complex128:
        """Return Statevector item either by index or binary label
        Args:
            key (int or str): index or corresponding binary label, e.g. '01' = 1.

        Returns:
            numpy.complex128: Statevector item.

        Raises:
            QiskitError: if key is not valid.
        """
        if isinstance(key, str):
            try:
                key = int(key, 2)
            except ValueError:
                raise QiskitError(f"Key '{key}' is not a valid binary string.") from None
        if isinstance(key, int):
            if key >= self.dim:
                raise QiskitError(f"Key {key} is greater than Statevector dimension {self.dim}.")
            if key < 0:
                raise QiskitError(f"Key {key} is not a valid positive value.")
            return self._data[key]
        else:
            raise QiskitError("Key must be int or a valid binary string.")

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)

    @property
    def data(self) -> np.ndarray:
        """Return data."""
        return self._data

    def is_valid(self, atol: float | None = None, rtol: float | None = None) -> bool:
        """Return True if a Statevector has norm 1."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        norm = np.linalg.norm(self.data)
        return np.allclose(norm, 1, rtol=rtol, atol=atol)

    def to_operator(self) -> Operator:
        """Convert state to a rank-1 projector operator"""
        mat = np.outer(self.data, np.conj(self.data))
        return Operator(mat, input_dims=self.dims(), output_dims=self.dims())

    def conjugate(self) -> Statevector:
        """Return the conjugate of the operator."""
        return Statevector(np.conj(self.data), dims=self.dims())

    def trace(self) -> np.float64:
        """Return the trace of the quantum state as a density matrix."""
        return np.sum(np.abs(self.data) ** 2)

    def purity(self) -> np.float64:
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return self.trace() ** 2

    def tensor(self, other: Statevector) -> Statevector:
        """Return the tensor product state self ⊗ other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        ret = _copy.copy(self)
        ret._op_shape = self._op_shape.tensor(other._op_shape)
        ret._data = np.kron(self._data, other._data)
        return ret

    def inner(self, other: Statevector) -> np.complex128:
        r"""Return the inner product of self and other as
        :math:`\langle self| other \rangle`.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            np.complex128: the inner product of self and other, :math:`\langle self| other \rangle`.

        Raises:
            QiskitError: if other is not a quantum state or has different dimension.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        if self.dims() != other.dims():
            raise QiskitError(
                f"Statevector dimensions do not match: {self.dims()} and {other.dims()}."
            )
        inner = np.vdot(self.data, other.data)
        return inner

    def expand(self, other: Statevector) -> Statevector:
        """Return the tensor product state other ⊗ self.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        ret = _copy.copy(self)
        ret._op_shape = self._op_shape.expand(other._op_shape)
        ret._data = np.kron(other._data, self._data)
        return ret

    def _add(self, other):
        """Return the linear combination self + other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        self._op_shape._validate_add(other._op_shape)
        ret = _copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        """Return the scalar multiplied state self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Statevector: the scalar multiplied state other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = _copy.copy(self)
        ret._data = other * self.data
        return ret

    def evolve(
        self, other: Operator | QuantumCircuit | Instruction, qargs: list[int] | None = None
    ) -> Statevector:
        """Evolve a quantum state by the operator.

        Args:
            other (Operator | QuantumCircuit | circuit.Instruction): The operator to evolve by.
            qargs (list): a list of Statevector subsystem positions to apply
                           the operator on.

        Returns:
            Statevector: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified Statevector subsystem dimensions.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        # Get return vector
        ret = _copy.copy(self)

        # Evolution by a circuit or instruction
        if isinstance(other, QuantumCircuit):
            other = other.to_instruction()
        if isinstance(other, Instruction):
            if self.num_qubits is None:
                raise QiskitError("Cannot apply QuantumCircuit to non-qubit Statevector.")
            return self._evolve_instruction(ret, other, qargs=qargs)

        # Evolution by an Operator
        if not isinstance(other, Operator):
            dims = self.dims(qargs=qargs)
            other = Operator(other, input_dims=dims, output_dims=dims)

        # check dimension
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        return Statevector._evolve_operator(ret, other, qargs=qargs)

    def equiv(
        self, other: Statevector, rtol: float | None = None, atol: float | None = None
    ) -> bool:
        """Return True if other is equivalent as a statevector up to global phase.

        .. note::

            If other is not a Statevector, but can be used to initialize a statevector object,
            this will check that Statevector(other) is equivalent to the current statevector up
            to global phase.

        Args:
            other (Statevector): an object from which a ``Statevector`` can be constructed.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if statevectors are equivalent up to global phase.
        """
        if not isinstance(other, Statevector):
            try:
                other = Statevector(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.data, other.data, ignore_phase=True, rtol=rtol, atol=atol)

    def reverse_qargs(self) -> Statevector:
        r"""Return a Statevector with reversed subsystem ordering.

        For a tensor product state this is equivalent to reversing the order
        of tensor product subsystems. For a statevector
        :math:`|\psi \rangle = |\psi_{n-1} \rangle \otimes ... \otimes |\psi_0 \rangle`
        the returned statevector will be
        :math:`|\psi_{0} \rangle \otimes ... \otimes |\psi_{n-1} \rangle`.

        Returns:
            Statevector: the Statevector with reversed subsystem order.
        """
        ret = _copy.copy(self)
        axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
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

        if x_mask == 0:
            return pauli_phase * expval_pauli_no_x(self.data, self.num_qubits, z_mask)

        x_max = qubits[pauli.x][-1]
        y_phase = (-1j) ** pauli._count_y()
        y_phase = y_phase[0]

        return pauli_phase * expval_pauli_with_x(
            self.data, self.num_qubits, z_mask, x_mask, y_phase, x_max
        )

    def expectation_value(
        self, oper: BaseOperator | QuantumCircuit | Instruction, qargs: None | list[int] = None
    ) -> complex:
        """Compute the expectation value of an operator.

        Args:
            oper (Operator): an operator to evaluate expval of.
            qargs (None or list): subsystems to apply operator on.

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

        val = self.evolve(oper, qargs=qargs)
        conj = self.conjugate()
        return np.dot(conj.data, val.data)

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

            Consider a 2-qubit product state
            :math:`|\\psi\\rangle=|+\\rangle\\otimes|0\\rangle`.

            .. plot::
               :include-source:
               :nofigs:

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('+0')

                # Probabilities for measuring both qubits
                probs = psi.probabilities()
                print('probs: {}'.format(probs))

                # Probabilities for measuring only qubit-0
                probs_qubit_0 = psi.probabilities([0])
                print('Qubit-0 probs: {}'.format(probs_qubit_0))

                # Probabilities for measuring only qubit-1
                probs_qubit_1 = psi.probabilities([1])
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

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('+0')

                # Probabilities for measuring both qubits
                probs = psi.probabilities([0, 1])
                print('probs: {}'.format(probs))

                # Probabilities for measuring both qubits
                # but swapping qubits 0 and 1 in output
                probs_swapped = psi.probabilities([1, 0])
                print('Swapped probs: {}'.format(probs_swapped))

            .. code-block:: text

                probs: [0.5 0.  0.5 0. ]
                Swapped probs: [0.5 0.5 0.  0. ]

        """
        probs = self._subsystem_probabilities(
            np.abs(self.data) ** 2, self._op_shape.dims_l(), qargs=qargs
        )

        # to account for roundoff errors, we clip
        probs = np.clip(probs, a_min=0, a_max=1)

        if decimals is not None:
            probs = probs.round(decimals=decimals)

        return probs

    def reset(self, qargs: list[int] | None = None) -> Statevector:
        """Reset state or subsystems to the 0-state.

        Args:
            qargs (list or None): subsystems to reset, if None all
                                  subsystems will be reset to their 0-state
                                  (Default: None).

        Returns:
            Statevector: the reset state.

        Additional Information:
            If all subsystems are reset this will return the ground state
            on all subsystems. If only some subsystems are reset this
            function will perform a measurement on those subsystems and
            evolve the subsystems so that the collapsed post-measurement
            states are rotated to the 0-state. The RNG seed for this
            sampling can be set using the :meth:`seed` method.
        """
        if qargs is None:
            # Resetting all qubits does not require sampling or RNG
            ret = _copy.copy(self)
            state = np.zeros(self._op_shape.shape, dtype=complex)
            state[0] = 1
            ret._data = state
            return ret

        # Sample a single measurement outcome
        dims = self.dims(qargs)
        probs = self.probabilities(qargs)
        sample = self._rng.choice(len(probs), p=probs, size=1)

        # Convert to projector for state update
        proj = np.zeros(len(probs), dtype=complex)
        proj[sample] = 1 / np.sqrt(probs[sample])

        # Rotate outcome to 0
        reset = np.eye(len(probs))
        reset[0, 0] = 0
        reset[sample, sample] = 0
        reset[0, sample] = 1

        # compose with reset projection
        reset = np.dot(reset, np.diag(proj))
        return self.evolve(Operator(reset, input_dims=dims, output_dims=dims), qargs=qargs)

    @classmethod
    def from_label(cls, label: str) -> Statevector:
        """Return a tensor product of Pauli X,Y,Z eigenstates.

        .. list-table:: Single-qubit state labels
           :header-rows: 1

           * - Label
             - Statevector
           * - ``"0"``
             - :math:`[1, 0]`
           * - ``"1"``
             - :math:`[0, 1]`
           * - ``"+"``
             - :math:`[1 / \\sqrt{2},  1 / \\sqrt{2}]`
           * - ``"-"``
             - :math:`[1 / \\sqrt{2},  -1 / \\sqrt{2}]`
           * - ``"r"``
             - :math:`[1 / \\sqrt{2},  i / \\sqrt{2}]`
           * - ``"l"``
             - :math:`[1 / \\sqrt{2},  -i / \\sqrt{2}]`

        Args:
            label (string): an eigenstate string ket label (see table for
                            allowed values).

        Returns:
            Statevector: The N-qubit basis state statevector.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.
        """
        # Check label is valid
        if re.match(r"^[01rl\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # We can prepare Z-eigenstates by converting the computational
        # basis bit-string to an integer and preparing that unit vector
        # However, for X-basis states, we will prepare a Z-eigenstate first
        # then apply Hadamard gates to rotate 0 and 1s to + and -.
        z_label = label
        xy_states = False
        if re.match("^[01]+$", label) is None:
            # We have X or Y eigenstates so replace +,r with 0 and
            # -,l with 1 and prepare the corresponding Z state
            xy_states = True
            z_label = z_label.replace("+", "0")
            z_label = z_label.replace("r", "0")
            z_label = z_label.replace("-", "1")
            z_label = z_label.replace("l", "1")
        # Initialize Z eigenstate vector
        num_qubits = len(label)
        data = np.zeros(1 << num_qubits, dtype=complex)
        pos = int(z_label, 2)
        data[pos] = 1
        state = Statevector(data)
        if xy_states:
            # Apply hadamards to all qubits in X eigenstates
            x_mat = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
            # Apply S.H to qubits in Y eigenstates
            y_mat = np.dot(np.diag([1, 1j]), x_mat)
            for qubit, char in enumerate(reversed(label)):
                if char in ["+", "-"]:
                    state = state.evolve(x_mat, qargs=[qubit])
                elif char in ["r", "l"]:
                    state = state.evolve(y_mat, qargs=[qubit])
        return state

    @staticmethod
    def from_int(i: int, dims: int | tuple | list) -> Statevector:
        """Return a computational basis statevector.

        Args:
            i (int): the basis state element.
            dims (int or tuple or list): The subsystem dimensions of the statevector
                                         (See additional information).

        Returns:
            Statevector: The computational basis state :math:`|i\\rangle`.

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
        state = np.zeros(size, dtype=complex)
        state[i] = 1.0
        return Statevector(state, dims=dims)

    @classmethod
    def from_instruction(cls, instruction: Instruction | QuantumCircuit) -> Statevector:
        """Return the output statevector of an instruction.

        The statevector is initialized in the state :math:`|{0,\\ldots,0}\\rangle` of the
        same number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction (qiskit.circuit.Instruction or QuantumCircuit): instruction or circuit

        Returns:
            Statevector: The final statevector.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
                         the statevector simulation.
        """
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        init = np.zeros(2**instruction.num_qubits, dtype=complex)
        init[0] = 1.0
        vec = Statevector(init, dims=instruction.num_qubits * (2,))
        return Statevector._evolve_instruction(vec, instruction)

    def to_dict(self, decimals: None | int = None) -> dict:
        r"""Convert the statevector to dictionary form.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            decimals (None or int): the number of decimal places to round
                                    values. If None no rounding is done
                                    (Default: None).

        Returns:
            dict: the dictionary form of the Statevector.

        Example:

            The ket-form of a 2-qubit statevector
            :math:`|\psi\rangle = |-\rangle\otimes |0\rangle`

            .. plot::
               :include-source:
               :nofigs:

                from qiskit.quantum_info import Statevector

                psi = Statevector.from_label('-0')
                print(psi.to_dict())

            .. code-block:: text

                {'00': (0.7071067811865475+0j), '10': (-0.7071067811865475+0j)}

            For non-qubit subsystems the integer range can go from 0 to 9. For
            example in a qutrit system

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit.quantum_info import Statevector

                vec = np.zeros(9)
                vec[0] = 1 / np.sqrt(2)
                vec[-1] = 1 / np.sqrt(2)
                psi = Statevector(vec, dims=(3, 3))
                print(psi.to_dict())

            .. code-block:: text

                {'00': (0.7071067811865475+0j), '22': (0.7071067811865475+0j)}

            For large subsystem dimensions delimiters are required. The
            following example is for a 20-dimensional system consisting of
            a qubit and 10-dimensional qudit.

            .. plot::
               :include-source:
               :nofigs:

                import numpy as np
                from qiskit.quantum_info import Statevector

                vec = np.zeros(2 * 10)
                vec[0] = 1 / np.sqrt(2)
                vec[-1] = 1 / np.sqrt(2)
                psi = Statevector(vec, dims=(2, 10))
                print(psi.to_dict())

            .. code-block:: text

                {'00': (0.7071067811865475+0j), '91': (0.7071067811865475+0j)}

        """
        return self._vector_to_dict(
            self.data, self._op_shape.dims_l(), decimals=decimals, string_labels=True
        )

    @staticmethod
    def _evolve_operator(statevec, oper, qargs=None):
        """Evolve a qudit statevector"""
        new_shape = statevec._op_shape.compose(oper._op_shape, qargs=qargs)
        if qargs is None:
            # Full system evolution
            statevec._data = np.dot(oper._data, statevec._data)
            statevec._op_shape = new_shape
            return statevec

        # Get transpose axes
        num_qargs = statevec._op_shape.num_qargs[0]
        indices = [num_qargs - 1 - i for i in reversed(qargs)]
        axes = indices + [i for i in range(num_qargs) if i not in indices]
        axes_inv = np.argsort(axes).tolist()

        # Calculate contraction dimensions
        contract_dim = oper._op_shape.shape[1]
        contract_shape = (contract_dim, statevec._op_shape.shape[0] // contract_dim)

        # Reshape and transpose input array for contraction
        tensor = np.transpose(
            np.reshape(statevec.data, statevec._op_shape.tensor_shape),
            axes,
        )
        tensor_shape = tensor.shape

        # Perform contraction
        tensor = np.reshape(
            np.dot(oper.data, np.reshape(tensor, contract_shape)),
            tensor_shape,
        )

        # Transpose back to  original subsystem spec and flatten
        statevec._data = np.reshape(np.transpose(tensor, axes_inv), new_shape.shape[0])

        # Update dimension
        statevec._op_shape = new_shape
        return statevec

    @staticmethod
    def _evolve_instruction(statevec, obj, qargs=None):
        """Update the current Statevector by applying an instruction."""
        from qiskit.circuit.reset import Reset
        from qiskit.circuit.barrier import Barrier

        # pylint complains about a cyclic import since the following Initialize file
        # imports the StatePreparation, which again requires the Statevector (this file),
        # but as this is a local import, it's not actually an issue and can be ignored

        from qiskit.circuit.library.data_preparation.initializer import Initialize

        mat = Operator._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            return Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)

        # Special instruction types
        if isinstance(obj, Reset):
            statevec._data = statevec.reset(qargs)._data
            return statevec
        if isinstance(obj, Barrier):
            return statevec
        if isinstance(obj, Initialize):
            # state is initialized to labels in the initialize object
            if all(isinstance(param, str) for param in obj.params):
                initialization = Statevector.from_label("".join(obj.params))._data
            # state is initialized to an integer
            # here we're only checking the length as (1) a length-1 object necessarily means the
            # state is described by an integer (as labels were already covered) and (2) the int
            # was cast to a complex and we cannot do an int typecheck anyways
            elif len(obj.params) == 1:
                state = int(np.real(obj.params[0]))
                initialization = Statevector.from_int(state, (2,) * obj.num_qubits)._data
            # state is initialized to the statevector
            else:
                initialization = np.asarray(obj.params, dtype=complex)

            if qargs is None:
                statevec._data = initialization
            else:
                # if we act on a subsystem we first need to reset and then apply the
                # state preparation
                statevec._data = statevec.reset(qargs)._data
                mat = np.zeros((2 ** len(qargs), 2 ** len(qargs)), dtype=complex)
                mat[:, 0] = initialization
                statevec = Statevector._evolve_operator(statevec, Operator(mat), qargs=qargs)

            return statevec

        # If the instruction doesn't have a matrix defined we use its
        # circuit decomposition definition if it exists, otherwise we
        # cannot compose this gate and raise an error.
        if obj.definition is None:
            raise QiskitError(f"Cannot apply Instruction: {obj.name}")
        if not isinstance(obj.definition, QuantumCircuit):
            raise QiskitError(
                f"{obj.name} instruction definition is {type(obj.definition)}; expected QuantumCircuit"
            )

        if obj.definition.global_phase:
            # We do not apply this in-place, just in-case we have a shallow copy of _data.
            statevec._data = statevec._data * np.exp(1j * float(obj.definition.global_phase))
        qubits = {qubit: i for i, qubit in enumerate(obj.definition.qubits)}
        for instruction in obj.definition:
            if instruction.clbits:
                raise QiskitError(
                    f"Cannot apply instruction with classical bits: {instruction.operation.name}"
                )
            # Get the integer position of the flat register
            if qargs is None:
                new_qargs = [qubits[tup] for tup in instruction.qubits]
            else:
                new_qargs = [qargs[qubits[tup]] for tup in instruction.qubits]
            Statevector._evolve_instruction(statevec, instruction.operation, qargs=new_qargs)
        return statevec
