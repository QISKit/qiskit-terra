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

"""A circuit implementing a quadratic form on binary variables."""

from __future__ import annotations


import math
from collections.abc import Sequence

import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterExpression, Gate, CircuitError
from qiskit.utils.deprecation import deprecate_func
from ..basis_change import QFT, QFTGate

_ValueType = int | float | np.integer | np.floating | ParameterExpression


class QuadraticForm(QuantumCircuit):
    r"""Implements a quadratic form on binary variables encoded in qubit registers.

    A quadratic form on binary variables is a quadratic function :math:`Q` acting on a binary
    variable of :math:`n` bits, :math:`x = x_0 ... x_{n-1}`. For an integer matrix :math:`A`,
    an integer vector :math:`b` and an integer :math:`c` the function can be written as

    .. math::

        Q(x) = x^T A x + x^T b + c

    If :math:`A`, :math:`b` or :math:`c` contain scalar values, this circuit computes only
    an approximation of the quadratic form.

    Provided with :math:`m` qubits to encode the value, this circuit computes :math:`Q(x) \mod 2^m`
    in [two's complement](https://stackoverflow.com/questions/1049722/what-is-2s-complement)
    representation.

    .. math::

        |x\rangle_n |0\rangle_m \mapsto |x\rangle_n |(Q(x) + 2^m) \mod 2^m \rangle_m

    Since we use two's complement e.g. the value of :math:`Q(x) = 3` requires 2 bits to represent
    the value and 1 bit for the sign: `3 = '011'` where the first `0` indicates a positive value.
    On the other hand, :math:`Q(x) = -3` would be `-3 = '101'`, where the first `1` indicates
    a negative value and `01` is the two's complement of `3`.

    If the value of :math:`Q(x)` is too large to be represented with `m` qubits, the resulting
    bitstring is :math:`(Q(x) + 2^m) \mod 2^m)`.

    The implementation of this circuit is discussed in [1], Fig. 6.

    References:

    [1] Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
    `arXiv:1912.04088 <https://arxiv.org/pdf/1912.04088.pdf>`_

    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the QuadraticFormGate instead.",
        removal_timeline="Qiskit 3.0",
    )
    def __init__(
        self,
        num_result_qubits: int | None = None,
        quadratic: np.ndarray | list[list[float | ParameterExpression]] | None = None,
        linear: np.ndarray | list[float | ParameterExpression] | None = None,
        offset: float | ParameterExpression | None = None,
        little_endian: bool = True,
    ) -> None:
        r"""
        Args:
            num_result_qubits: The number of qubits to encode the result. Called :math:`m` in
                the class documentation.
            quadratic: A matrix containing the quadratic coefficients, :math:`A`.
            linear: An array containing the linear coefficients, :math:`b`.
            offset: A constant offset, :math:`c`.
            little_endian: Encode the result in little endianness.

        Raises:
            ValueError: If ``linear`` and ``quadratic`` have mismatching sizes.
            ValueError: If ``num_result_qubits`` is unspecified but cannot be determined because
                some values of the quadratic form are parameterized.
        """
        # check inputs match
        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError("Mismatching sizes of quadratic and linear.")

        # temporarily set quadratic and linear to [] instead of None so we can iterate over them
        if quadratic is None:
            quadratic = []

        if linear is None:
            linear = []

        if offset is None:
            offset = 0

        num_input_qubits = np.max([1, len(linear), len(quadratic)])

        # deduce number of result bits if not added
        if num_result_qubits is None:
            # check no value is parameterized
            if (
                any(any(isinstance(q_ij, ParameterExpression) for q_ij in q_i) for q_i in quadratic)
                or any(isinstance(l_i, ParameterExpression) for l_i in linear)
                or isinstance(offset, ParameterExpression)
            ):
                raise ValueError(
                    "If the number of result qubits is not specified, the quadratic "
                    "form matrices/vectors/offset may not be parameterized."
                )
            num_result_qubits = self.required_result_qubits(quadratic, linear, offset)

        qr_input = QuantumRegister(num_input_qubits)
        qr_result = QuantumRegister(num_result_qubits)
        circuit = QuantumCircuit(qr_input, qr_result, name="Q(x)")

        # set quadratic and linear again to None if they were None
        if len(quadratic) == 0:
            quadratic = None

        if len(linear) == 0:
            linear = None

        scaling = np.pi * 2 ** (1 - num_result_qubits)

        # initial QFT (just hadamards)
        circuit.h(qr_result)

        if little_endian:
            qr_result = qr_result[::-1]

        # constant coefficient
        if offset != 0:
            for i, q_i in enumerate(qr_result):
                circuit.p(scaling * 2**i * offset, q_i)

        # the linear part consists of the vector and the diagonal of the
        # matrix, since x_i * x_i = x_i, as x_i is a binary variable
        for j in range(num_input_qubits):
            value = linear[j] if linear is not None else 0
            value += quadratic[j][j] if quadratic is not None else 0
            if value != 0:
                for i, q_i in enumerate(qr_result):
                    circuit.cp(scaling * 2**i * value, qr_input[j], q_i)

        # the quadratic part adds A_ij and A_ji as x_i x_j == x_j x_i
        if quadratic is not None:
            for j in range(num_input_qubits):
                for k in range(j + 1, num_input_qubits):
                    value = quadratic[j][k] + quadratic[k][j]
                    if value != 0:
                        for i, q_i in enumerate(qr_result):
                            circuit.mcp(scaling * 2**i * value, [qr_input[j], qr_input[k]], q_i)

        # add the inverse QFT
        iqft = QFT(num_result_qubits, do_swaps=False).inverse().reverse_bits()
        circuit.compose(iqft, qubits=qr_result[:], inplace=True)

        super().__init__(*circuit.qregs, name="Q(x)")
        self.compose(circuit.to_gate(), qubits=self.qubits, inplace=True)

    @staticmethod
    def required_result_qubits(
        quadratic: np.ndarray | list[list[float]],
        linear: np.ndarray | list[float],
        offset: float,
    ) -> int:
        """Get the number of required result qubits.

        Args:
            quadratic: A matrix containing the quadratic coefficients.
            linear: An array containing the linear coefficients.
            offset: A constant offset.

        Returns:
            The number of qubits needed to represent the value of the quadratic form
            in twos complement.
        """
        return QuadraticFormGate.required_result_qubits(quadratic, linear, offset)


class QuadraticFormGate(Gate):
    r"""Implements a quadratic form on binary variables encoded in qubit registers.

    A quadratic form on binary variables is a quadratic function :math:`Q` acting on a binary
    variable of :math:`n` bits, :math:`x = x_0 ... x_{n-1}`. For an integer matrix :math:`A`,
    an integer vector :math:`b` and an integer :math:`c` the function can be written as

    .. math::

        Q(x) = x^T A x + x^T b + c

    If :math:`A`, :math:`b` or :math:`c` contain scalar values, this circuit computes only
    an approximation of the quadratic form.

    Provided with :math:`m` qubits to encode the value, this circuit computes :math:`Q(x) \mod 2^m`
    in [two's complement](https://stackoverflow.com/questions/1049722/what-is-2s-complement)
    representation.

    .. math::

        |x\rangle_n |0\rangle_m \mapsto |x\rangle_n |(Q(x) + 2^m) \mod 2^m \rangle_m

    Since we use two's complement e.g. the value of :math:`Q(x) = 3` requires 2 bits to represent
    the value and 1 bit for the sign: `3 = '011'` where the first `0` indicates a positive value.
    On the other hand, :math:`Q(x) = -3` would be `-3 = '101'`, where the first `1` indicates
    a negative value and `01` is the two's complement of `3`.

    If the value of :math:`Q(x)` is too large to be represented with `m` qubits, the resulting
    bitstring is :math:`(Q(x) + 2^m) \mod 2^m)`.

    The implementation of this circuit is discussed in [1], Fig. 6.

    References:

    [1] Gilliam et al., Grover Adaptive Search for Constrained Polynomial Binary Optimization.
    `arXiv:1912.04088 <https://arxiv.org/pdf/1912.04088.pdf>`_

    """

    def __init__(
        self,
        num_result_qubits: int | None = None,
        quadratic: Sequence[Sequence[float]] | None = None,
        linear: Sequence[Sequence[float]] | None = None,
        offset: float | None = None,
        label: str = "Q(x)",
    ):
        # check inputs match
        if quadratic is not None and linear is not None:
            if len(quadratic) != len(linear):
                raise ValueError("Mismatching sizes of quadratic and linear.")

        # temporarily set quadratic and linear to [] instead of None so we can iterate over them
        if quadratic is None:
            quadratic = []

        if linear is None:
            linear = []

        if offset is None:
            offset = 0

        self.num_input_qubits = np.max([1, len(linear), len(quadratic)])

        # deduce number of result bits if not added
        if num_result_qubits is None:
            num_result_qubits = self.required_result_qubits(quadratic, linear, offset)

        self.num_result_qubits = num_result_qubits
        self.quadratic = quadratic
        self.linear = linear
        self.offset = offset

        num_qubits = int(self.num_input_qubits + self.num_result_qubits)
        super().__init__("QuadraticForm", num_qubits, [], label=label)

    @staticmethod
    def required_result_qubits(
        quadratic: Sequence[Sequence[float]],
        linear: Sequence[float],
        offset: float,
    ) -> int:
        """Get the number of required result qubits.

        Args:
            quadratic: A matrix containing the quadratic coefficients.
            linear: An array containing the linear coefficients.
            offset: A constant offset.

        Returns:
            The number of qubits needed to represent the value of the quadratic form
            in twos complement.
        """

        bounds = []  # bounds = [minimum value, maximum value]
        for condition in [lambda x: x < 0, lambda x: x > 0]:
            bound = 0.0
            bound += sum(sum(q_ij for q_ij in q_i if condition(q_ij)) for q_i in quadratic)
            bound += sum(l_i for l_i in linear if condition(l_i))
            bound += offset if condition(offset) else 0
            bounds.append(bound)

        # the minimum number of qubits is the number of qubits needed to represent
        # the minimum/maximum value plus one sign qubit
        num_qubits_for_min = math.ceil(math.log2(max(-bounds[0], 1)))
        num_qubits_for_max = math.ceil(math.log2(bounds[1] + 1))
        num_result_qubits = 1 + max(num_qubits_for_min, num_qubits_for_max)

        return num_result_qubits

    def validate_parameter(self, parameter):
        if isinstance(parameter, _ValueType):
            return parameter

        if isinstance(parameter, (np.ndarray, Sequence)):
            if all(isinstance(el, _ValueType) for el in parameter):
                return parameter
            for params in parameter:
                if not all(isinstance(el, _ValueType) for el in params):
                    raise CircuitError(
                        f"Invalid parameter type {type(parameter)} for QuadraticFormGate"
                    )

            return parameter

        return super().validate_parameter(parameter)

    def _define(self):
        quadratic, linear, offset = self.quadratic, self.linear, self.offset

        qr_input = QuantumRegister(self.num_input_qubits)
        qr_result = QuantumRegister(self.num_result_qubits)
        circuit = QuantumCircuit(qr_input, qr_result)

        # set quadratic and linear again to None if they were None
        if len(quadratic) == 0:
            quadratic = None

        if len(linear) == 0:
            linear = None

        scaling = np.pi * 2 ** (1 - self.num_result_qubits)

        # initial QFT
        qft = QFTGate(self.num_result_qubits)
        circuit.append(qft, qr_result)

        # constant coefficient
        if offset != 0:
            for i, q_i in enumerate(qr_result):
                circuit.p(scaling * 2**i * offset, q_i)

        # the linear part consists of the vector and the diagonal of the
        # matrix, since x_i * x_i = x_i, as x_i is a binary variable
        for j in range(self.num_input_qubits):
            value = linear[j] if linear is not None else 0
            value += quadratic[j][j] if quadratic is not None else 0
            if value != 0:
                for i, q_i in enumerate(qr_result):
                    circuit.cp(scaling * 2**i * value, qr_input[j], q_i)

        # the quadratic part adds A_ij and A_ji as x_i x_j == x_j x_i
        if quadratic is not None:
            for j in range(self.num_input_qubits):
                for k in range(j + 1, self.num_input_qubits):
                    value = quadratic[j][k] + quadratic[k][j]
                    if value != 0:
                        for i, q_i in enumerate(qr_result):
                            circuit.mcp(scaling * 2**i * value, [qr_input[j], qr_input[k]], q_i)

        # add the inverse QFT
        iqft = qft.inverse()
        circuit.append(iqft, qr_result)

        self.definition = circuit
