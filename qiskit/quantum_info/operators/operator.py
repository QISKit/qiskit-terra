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
Matrix Operator class.
"""

from __future__ import annotations

import cmath
import copy as _copy
import re
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np

from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal

if TYPE_CHECKING:
    from qiskit.transpiler.layout import Layout


class Operator(LinearOp):
    r"""Matrix operator class

    This represents a matrix operator :math:`M` that will
    :meth:`~Statevector.evolve` a :class:`Statevector` :math:`|\psi\rangle`
    by matrix-vector multiplication

    .. math::

        |\psi\rangle \mapsto M|\psi\rangle,

    and will :meth:`~DensityMatrix.evolve` a :class:`DensityMatrix` :math:`\rho`
    by left and right multiplication

    .. math::

        \rho \mapsto M \rho M^\dagger.

    For example, the following operator :math:`M = X` applied to the zero state
    :math:`|\psi\rangle=|0\rangle (\rho = |0\rangle\langle 0|)` changes it to the
    one state :math:`|\psi\rangle=|1\rangle (\rho = |1\rangle\langle 1|)`:

    .. plot::
       :include-source:
       :nofigs:

        >>> import numpy as np
        >>> from qiskit.quantum_info import Operator
        >>> op = Operator(np.array([[0.0, 1.0], [1.0, 0.0]]))  # Represents Pauli X operator

        >>> from qiskit.quantum_info import Statevector
        >>> sv = Statevector(np.array([1.0, 0.0]))
        >>> sv.evolve(op)
        Statevector([0.+0.j, 1.+0.j],
                    dims=(2,))

        >>> from qiskit.quantum_info import DensityMatrix
        >>> dm = DensityMatrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
        >>> dm.evolve(op)
        DensityMatrix([[0.+0.j, 0.+0.j],
                    [0.+0.j, 1.+0.j]],
                    dims=(2,))

    """

    def __init__(
        self,
        data: QuantumCircuit | Operation | BaseOperator | np.ndarray,
        input_dims: tuple | None = None,
        output_dims: tuple | None = None,
    ):
        """Initialize an operator object.

        Args:
            data: data to initialize operator.
            input_dims: the input subsystem dimensions.
            output_dims: the output subsystem dimensions.

        Raises:
            QiskitError: if input data cannot be initialized as an operator.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
            Note that two operators initialized via this method are only considered equivalent if they
            match up to their canonical qubit order (or: permutation). See :meth:`.Operator.from_circuit`
            to specify a different qubit permutation.
        """
        op_shape = None
        if isinstance(data, (list, np.ndarray)):
            # Default initialization from list or numpy array matrix
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, (QuantumCircuit, Operation)):
            # If the input is a Terra QuantumCircuit or Operation we
            # perform a simulation to construct the unitary operator.
            # This will only work if the circuit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            self._data = self._init_instruction(data).data
        elif hasattr(data, "to_operator"):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            data = data.to_operator()
            self._data = data.data
            op_shape = data._op_shape
        elif hasattr(data, "to_matrix"):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            self._data = np.asarray(data.to_matrix(), dtype=complex)
        else:
            raise QiskitError("Invalid input data format for Operator")

        super().__init__(
            op_shape=op_shape,
            input_dims=input_dims,
            output_dims=output_dims,
            shape=self._data.shape,
        )

    def __array__(self, dtype=None, copy=None):
        dtype = self.data.dtype if dtype is None else dtype
        return np.array(self.data, dtype=dtype, copy=copy)

    def __repr__(self):
        prefix = "Operator("
        pad = len(prefix) * " "
        return (
            f"{prefix}{np.array2string(self.data, separator=', ', prefix=prefix)},\n"
            f"{pad}input_dims={self.input_dims()}, output_dims={self.output_dims()})"
        )

    def __eq__(self, other):
        """Test if two Operators are equal."""
        if not super().__eq__(other):
            return False
        return np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)

    @property
    def data(self):
        """The underlying Numpy array."""
        return self._data

    @property
    def settings(self):
        """Return operator settings."""
        return {
            "data": self._data,
            "input_dims": self.input_dims(),
            "output_dims": self.output_dims(),
        }

    def draw(self, output=None, **drawer_args):
        """Return a visualization of the Operator.

        **repr**: String of the state's ``__repr__``.

        **text**: ASCII TextMatrix that can be printed in the console.

        **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

        **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

        Args:
            output (str): Select the output method to use for drawing the
                state. Valid choices are `repr`, `text`, `latex`, `latex_source`,
                Default is `repr`.
            drawer_args: Arguments to be passed directly to the relevant drawing
                function or constructor (`TextMatrix()`, `array_to_latex()`).
                See the relevant function under `qiskit.visualization` for that function's
                documentation.

        Returns:
            :class:`str` or :class:`TextMatrix` or :class:`IPython.display.Latex`:
            Drawing of the Operator.

        Raises:
            ValueError: when an invalid output method is selected.
            MissingOptionalLibrary: If SymPy isn't installed and ``'latex'`` or
                ``'latex_source'`` is selected for ``output``.

        """

        from qiskit.visualization import array_to_latex

        default_output = "repr"
        if output is None:
            output = default_output

        if output == "repr":
            return self.__repr__()

        elif output == "text":
            from qiskit.visualization.state_visualization import TextMatrix

            return TextMatrix(self, **drawer_args)

        elif output == "latex":
            return array_to_latex(self, **drawer_args)

        elif output == "latex_source":
            return array_to_latex(self, source=True, **drawer_args)

        else:
            raise ValueError(
                f"""'{output}' is not a valid option for drawing {type(self).__name__} objects.
            Please choose from: 'text', 'latex', or 'latex_source'."""
            )

    def _ipython_display_(self):
        out = self.draw()
        if isinstance(out, str):
            print(out)  # noqa: T201
        else:
            from IPython.display import display

            display(out)

    @classmethod
    def from_label(cls, label: str) -> Operator:
        """Return a tensor product of single-qubit operators.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Operator: The N-qubit operator.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.

        Additional Information:
            The labels correspond to the single-qubit matrices:
            'I': [[1, 0], [0, 1]]
            'X': [[0, 1], [1, 0]]
            'Y': [[0, -1j], [1j, 0]]
            'Z': [[1, 0], [0, -1]]
            'H': [[1, 1], [1, -1]] / sqrt(2)
            'S': [[1, 0], [0 , 1j]]
            'T': [[1, 0], [0, (1+1j) / sqrt(2)]]
            '0': [[1, 0], [0, 0]]
            '1': [[0, 0], [0, 1]]
            '+': [[0.5, 0.5], [0.5 , 0.5]]
            '-': [[0.5, -0.5], [-0.5 , 0.5]]
            'r': [[0.5, -0.5j], [0.5j , 0.5]]
            'l': [[0.5, 0.5j], [-0.5j , 0.5]]
        """
        # Check label is valid
        label_mats = {
            "I": IGate().to_matrix(),
            "X": XGate().to_matrix(),
            "Y": YGate().to_matrix(),
            "Z": ZGate().to_matrix(),
            "H": HGate().to_matrix(),
            "S": SGate().to_matrix(),
            "T": TGate().to_matrix(),
            "0": np.array([[1, 0], [0, 0]], dtype=complex),
            "1": np.array([[0, 0], [0, 1]], dtype=complex),
            "+": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
            "-": np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex),
            "r": np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex),
            "l": np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex),
        }
        if re.match(r"^[IXYZHST01rl\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Operator(np.eye(2**num_qubits, dtype=complex))
        for qubit, char in enumerate(reversed(label)):
            if char != "I":
                op = op.compose(label_mats[char], qargs=[qubit])
        return op

    def apply_permutation(self, perm: list, front: bool = False) -> Operator:
        """Modifies operator's data by composing it with a permutation.

        Args:
            perm (list): permutation pattern, describing which qubits
                occupy the positions 0, 1, 2, etc. after applying the permutation.
            front (bool): When set to ``True`` the permutation is applied before the
                operator, when set to ``False`` the permutation is applied after the
                operator.
        Returns:
            Operator: The modified operator.

        Raises:
            QiskitError: if the size of the permutation pattern does not match the
                dimensions of the operator.
        """

        # See https://github.com/Qiskit/qiskit-terra/pull/9403 for the math
        # behind the following code.

        inv_perm = np.argsort(perm)
        raw_shape_l = self._op_shape.dims_l()
        n_dims_l = len(raw_shape_l)
        raw_shape_r = self._op_shape.dims_r()
        n_dims_r = len(raw_shape_r)

        if front:
            # The permutation is applied first, the operator is applied after;
            # however, in terms of matrices, we compute [O][P].

            if len(perm) != n_dims_r:
                raise QiskitError(
                    "The size of the permutation pattern does not match dimensions of the operator."
                )

            # shape: original on left, permuted on right
            shape_l = self._op_shape.dims_l()
            shape_r = tuple(raw_shape_r[n_dims_r - n - 1] for n in reversed(perm))

            # axes order: id on left, inv-permuted on right
            axes_l = tuple(x for x in range(self._op_shape._num_qargs_l))
            axes_r = tuple(self._op_shape._num_qargs_l + x for x in (np.argsort(perm[::-1]))[::-1])

            # updated shape: original on left, permuted on right
            new_shape_l = self._op_shape.dims_l()
            new_shape_r = tuple(raw_shape_r[n_dims_r - n - 1] for n in reversed(inv_perm))

        else:
            # The operator is applied first, the permutation is applied after;
            # however, in terms of matrices, we compute [P][O].

            if len(perm) != n_dims_l:
                raise QiskitError(
                    "The size of the permutation pattern does not match dimensions of the operator."
                )

            # shape: inv-permuted on left, original on right
            shape_l = tuple(raw_shape_l[n_dims_l - n - 1] for n in reversed(inv_perm))
            shape_r = self._op_shape.dims_r()

            # axes order: permuted on left, id on right
            axes_l = tuple((np.argsort(inv_perm[::-1]))[::-1])
            axes_r = tuple(
                self._op_shape._num_qargs_l + x for x in range(self._op_shape._num_qargs_r)
            )

            # updated shape: permuted on left, original on right
            new_shape_l = tuple(raw_shape_l[n_dims_l - n - 1] for n in reversed(perm))
            new_shape_r = self._op_shape.dims_r()

        # Computing the new operator
        split_shape = shape_l + shape_r
        axes_order = axes_l + axes_r
        new_mat = (
            self._data.reshape(split_shape).transpose(axes_order).reshape(self._op_shape.shape)
        )
        new_op = Operator(new_mat, input_dims=new_shape_r, output_dims=new_shape_l)
        return new_op

    @classmethod
    def from_circuit(
        cls,
        circuit: QuantumCircuit,
        ignore_set_layout: bool = False,
        layout: Layout | None = None,
        final_layout: Layout | None = None,
    ) -> Operator:
        """Create a new Operator object from a :class:`.QuantumCircuit`

        While a :class:`~.QuantumCircuit` object can passed directly as ``data``
        to the class constructor this provides no options on how the circuit
        is used to create an :class:`.Operator`. This constructor method lets
        you control how the :class:`.Operator` is created so it can be adjusted
        for a particular use case.

        By default this constructor method will permute the qubits based on a
        configured initial layout (i.e. after it was transpiled). It also
        provides an option to manually provide a :class:`.Layout` object
        directly.

        Args:
            circuit (QuantumCircuit): The :class:`.QuantumCircuit` to create an Operator
                object from.
            ignore_set_layout (bool): When set to ``True`` if the input ``circuit``
                has a layout set it will be ignored
            layout (Layout): If specified this kwarg can be used to specify a
                particular layout to use to permute the qubits in the created
                :class:`.Operator`. If this is specified it will be used instead
                of a layout contained in the ``circuit`` input. If specified
                the virtual bits in the :class:`~.Layout` must be present in the
                ``circuit`` input.
            final_layout (Layout): If specified this kwarg can be used to represent the
                output permutation caused by swap insertions during the routing stage
                of the transpiler.
        Returns:
            Operator: An operator representing the input circuit
        """

        if layout is None:
            if not ignore_set_layout:
                layout = getattr(circuit, "_layout", None)
        else:
            from qiskit.transpiler.layout import TranspileLayout

            layout = TranspileLayout(
                initial_layout=layout,
                input_qubit_mapping={qubit: index for index, qubit in enumerate(circuit.qubits)},
            )

        initial_layout = layout.initial_layout if layout is not None else None

        if final_layout is None:
            if not ignore_set_layout and layout is not None:
                final_layout = getattr(layout, "final_layout", None)

        from qiskit.synthesis.permutation import _inverse_pattern

        op = Operator(circuit)

        if initial_layout is not None:
            input_qubits = [None] * len(layout.input_qubit_mapping)
            for q, p in layout.input_qubit_mapping.items():
                input_qubits[p] = q

            initial_permutation = initial_layout.to_permutation(input_qubits)
            initial_permutation_inverse = _inverse_pattern(initial_permutation)
            op = op.apply_permutation(initial_permutation, True)

            if final_layout is not None:
                final_permutation = final_layout.to_permutation(circuit.qubits)
                final_permutation_inverse = _inverse_pattern(final_permutation)
                op = op.apply_permutation(final_permutation_inverse, False)
            op = op.apply_permutation(initial_permutation_inverse, False)
        elif final_layout is not None:
            final_permutation = final_layout.to_permutation(circuit.qubits)
            final_permutation_inverse = _inverse_pattern(final_permutation)
            op = op.apply_permutation(final_permutation_inverse, False)

        return op

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return is_unitary_matrix(self._data, rtol=rtol, atol=atol)

    def to_operator(self) -> Operator:
        """Convert operator to matrix operator class"""
        return self

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""

        from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate

        return UnitaryGate(self.data)

    def conjugate(self):
        # Make a shallow copy and update array
        ret = _copy.copy(self)
        ret._data = np.conj(self._data)
        return ret

    def transpose(self):
        # Make a shallow copy and update array
        ret = _copy.copy(self)
        ret._data = np.transpose(self._data)
        ret._op_shape = self._op_shape.transpose()
        return ret

    def compose(self, other: Operator, qargs: list | None = None, front: bool = False) -> Operator:
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, Operator):
            other = Operator(other)

        # Validate dimensions are compatible and return the composed
        # operator dimensions
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()

        # Full composition of operators
        if qargs is None:
            if front:
                # Composition self * other
                data = np.dot(self._data, other.data)
            else:
                # Composition other * self
                data = np.dot(other.data, self._data)
            ret = Operator(data, input_dims, output_dims)
            ret._op_shape = new_shape
            return ret

        # Compose with other on subsystem
        num_qargs_l, num_qargs_r = self._op_shape.num_qargs
        if front:
            num_indices = num_qargs_r
            shift = num_qargs_l
            right_mul = True
        else:
            num_indices = num_qargs_l
            shift = 0
            right_mul = False

        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._op_shape.tensor_shape)
        mat = np.reshape(other.data, other._op_shape.tensor_shape)
        indices = [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [int(np.prod(output_dims)), int(np.prod(input_dims))]
        data = np.reshape(
            Operator._einsum_matmul(tensor, mat, indices, shift, right_mul), final_shape
        )
        ret = Operator(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def power(
        self, n: float, branch_cut_rotation=cmath.pi * 1e-12, assume_unitary=False
    ) -> Operator:
        """Return the matrix power of the operator.

        Non-integer powers of operators with an eigenvalue whose complex phase is :math:`\\pi` have
        a branch cut in the complex plane, which makes the calculation of the principal root around
        this cut subject to precision / differences in BLAS implementation.  For example, the square
        root of Pauli Y can return the :math:`\\pi/2` or :math:`-\\pi/2` Y rotation depending on
        whether the -1 eigenvalue is found as ``complex(-1, tiny)`` or ``complex(-1, -tiny)``. Such
        eigenvalues are really common in quantum information, so this function first phase-rotates
        the input matrix to shift the branch cut to a far less common point.  The underlying
        numerical precision issues around the branch-cut point remain, if an operator has an
        eigenvalue close to this phase.  The magnitude of this rotation can be controlled with the
        ``branch_cut_rotation`` parameter.

        The choice of ``branch_cut_rotation`` affects the principal root that is found.  For
        example, the square root of :class:`.ZGate` will be calculated as either :class:`.SGate` or
        :class:`.SdgGate` depending on which way the rotation is done::

            from qiskit.circuit import library
            from qiskit.quantum_info import Operator

            z_op = Operator(library.ZGate())
            assert z_op.power(0.5, branch_cut_rotation=1e-3) == Operator(library.SGate())
            assert z_op.power(0.5, branch_cut_rotation=-1e-3) == Operator(library.SdgGate())

        Args:
            n (float): the power to raise the matrix to.
            branch_cut_rotation (float): The rotation angle to apply to the branch cut in the
                complex plane.  This shifts the branch cut away from the common point of :math:`-1`,
                but can cause a different root to be selected as the principal root.  The rotation
                is anticlockwise, following the standard convention for complex phase.
            assume_unitary (bool): if ``True``, the operator is assumed to be unitary. In this case,
                for fractional powers we employ a faster implementation based on Schur's decomposition.

        Returns:
            Operator: the resulting operator ``O ** n``.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal.

        .. note::
            It is only safe to set the argument ``assume_unitary`` to ``True`` when the operator
            is unitary (or, more generally, normal). Otherwise, the function will return an
            incorrect output.
        """
        if self.input_dims() != self.output_dims():
            raise QiskitError("Can only power with input_dims = output_dims.")
        ret = _copy.copy(self)
        if isinstance(n, int):
            ret._data = np.linalg.matrix_power(self.data, n)
        else:
            import scipy.linalg

            if assume_unitary:
                # Experimentally, for fractional powers this seems to be 3x faster than
                # calling scipy.linalg.fractional_matrix_power(self.data, exponent)
                decomposition, unitary = scipy.linalg.schur(
                    cmath.rect(1, -branch_cut_rotation) * self.data, output="complex"
                )
                decomposition_diagonal = decomposition.diagonal()
                decomposition_power = [pow(element, n) for element in decomposition_diagonal]
                unitary_power = unitary @ np.diag(decomposition_power) @ unitary.conj().T
                ret._data = cmath.rect(1, branch_cut_rotation * n) * unitary_power
            else:
                ret._data = cmath.rect(
                    1, branch_cut_rotation * n
                ) * scipy.linalg.fractional_matrix_power(
                    cmath.rect(1, -branch_cut_rotation) * self.data, n
                )

        return ret

    def tensor(self, other: Operator) -> Operator:
        if not isinstance(other, Operator):
            other = Operator(other)
        return self._tensor(self, other)

    def expand(self, other: Operator) -> Operator:
        if not isinstance(other, Operator):
            other = Operator(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = _copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = np.kron(a.data, b.data)
        return ret

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (Operator): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            Operator: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
                         dimensions.
        """

        from qiskit.quantum_info.operators.scalar_op import ScalarOp

        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, Operator):
            other = Operator(other)

        self._op_shape._validate_add(other._op_shape, qargs)
        other = ScalarOp._pad_with_identity(self, other, qargs)

        ret = _copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        """Return the operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = _copy.copy(self)
        ret._data = other * self._data
        return ret

    def equiv(self, other: Operator, rtol: float | None = None, atol: float | None = None) -> bool:
        """Return True if operators are equivalent up to global phase.

        Args:
            other (Operator): an operator object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if operators are equivalent up to global phase.
        """
        if not isinstance(other, Operator):
            try:
                other = Operator(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.data, other.data, ignore_phase=True, rtol=rtol, atol=atol)

    def reverse_qargs(self) -> Operator:
        r"""Return an Operator with reversed subsystem ordering.

        For a tensor product operator this is equivalent to reversing
        the order of tensor product subsystems. For an operator
        :math:`A = A_{n-1} \otimes ... \otimes A_0`
        the returned operator will be
        :math:`A_0 \otimes ... \otimes A_{n-1}`.

        Returns:
            Operator: the operator with reversed subsystem order.
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

    def to_matrix(self):
        """Convert operator to NumPy matrix."""
        return self.data

    @classmethod
    def _einsum_matmul(cls, tensor, mat, indices, shift=0, right_mul=False):
        """Perform a contraction using Numpy.einsum

        Args:
            tensor (np.array): a vector or matrix reshaped to a rank-N tensor.
            mat (np.array): a matrix reshaped to a rank-2M tensor.
            indices (list): tensor indices to contract with mat.
            shift (int): shift for indices of tensor to contract [Default: 0].
            right_mul (bool): if True right multiply tensor by mat
                              (else left multiply) [Default: False].

        Returns:
            Numpy.ndarray: the matrix multiplied rank-N tensor.

        Raises:
            QiskitError: if mat is not an even rank tensor.
        """
        rank = tensor.ndim
        rank_mat = mat.ndim
        if rank_mat % 2 != 0:
            raise QiskitError("Contracted matrix must have an even number of indices.")
        # Get einsum indices for tensor
        indices_tensor = list(range(rank))
        for j, index in enumerate(indices):
            indices_tensor[index + shift] = rank + j
        # Get einsum indices for mat
        mat_contract = list(reversed(range(rank, rank + len(indices))))
        mat_free = [index + shift for index in reversed(indices)]
        if right_mul:
            indices_mat = mat_contract + mat_free
        else:
            indices_mat = mat_free + mat_contract
        return np.einsum(tensor, indices_tensor, mat, indices_mat)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Operation to an Operator."""
        # Initialize an identity operator of the correct size of the circuit
        if hasattr(instruction, "__array__"):
            return Operator(np.array(instruction, dtype=complex))

        dimension = 2**instruction.num_qubits
        op = Operator(np.eye(dimension))
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_matrix(cls, obj):
        """Return Operator for instruction if defined or None otherwise."""
        # Note: to_matrix() is not a required method for Operations, so for now
        # we do not allow constructing matrices for general Operations.
        # However, for backward compatibility we need to support constructing matrices
        # for Cliffords, which happen to have a to_matrix() method.

        from qiskit.quantum_info import Clifford
        from qiskit.circuit.annotated_operation import AnnotatedOperation

        if not isinstance(obj, (Instruction, Clifford, AnnotatedOperation)):
            raise QiskitError("Input is neither Instruction, Clifford or AnnotatedOperation.")
        mat = None
        if hasattr(obj, "to_matrix"):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                mat = obj.to_matrix()
            except QiskitError:
                pass
        return mat

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        from qiskit.circuit.barrier import Barrier
        from .scalar_op import ScalarOp

        mat = self._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(mat, qargs=qargs)
            self._data = op.data
        elif isinstance(obj, Barrier):
            return
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError(f"Cannot apply Operation: {obj.name}")
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError(
                    f'Operation "{obj.name}" '
                    f"definition is {type(obj.definition)} but expected QuantumCircuit."
                )
            if obj.definition.global_phase:
                dimension = 2**obj.num_qubits
                op = self.compose(
                    ScalarOp(dimension, np.exp(1j * float(obj.definition.global_phase))),
                    qargs=qargs,
                )
                self._data = op.data
            flat_instr = obj.definition
            bit_indices = {
                bit: index
                for bits in [flat_instr.qubits, flat_instr.clbits]
                for index, bit in enumerate(bits)
            }

            for instruction in flat_instr:
                if instruction.clbits:
                    raise QiskitError(
                        "Cannot apply operation with classical bits:"
                        f" {instruction.operation.name}"
                    )
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [bit_indices[tup] for tup in instruction.qubits]
                else:
                    new_qargs = [qargs[bit_indices[tup]] for tup in instruction.qubits]
                self._append_instruction(instruction.operation, qargs=new_qargs)


# Update docstrings for API docs
generate_apidocs(Operator)
