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


"""Polynomially controlled Pauli-rotations."""

from __future__ import annotations

from itertools import product

from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations


def _binomial_coefficients(n):
    """Return a dictionary of binomial coefficients

    Based-on/forked from sympy's binomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """

    data = {(0, n): 1, (n, 0): 1}
    temp = 1
    for k in range(1, n // 2 + 1):
        temp = (temp * (n - k + 1)) // k
        data[k, n - k] = data[n - k, k] = temp
    return data


def _large_coefficients_iter(m, n):
    """Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients_iterator() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if m < 2 * n or n == 1:
        coefficients = _multinomial_coefficients(m, n)
        for key, value in coefficients.items():
            yield (key, value)
    else:
        coefficients = _multinomial_coefficients(n, n)
        coefficients_dict = {}
        for key, value in coefficients.items():
            coefficients_dict[tuple(filter(None, key))] = value
        coefficients = coefficients_dict

        temp = [n] + [0] * (m - 1)
        temp_a = tuple(temp)
        b = tuple(filter(None, temp_a))
        yield (temp_a, coefficients[b])
        if n:
            j = 0  # j will be the leftmost nonzero position
        else:
            j = m
        # enumerate tuples in co-lex order
        while j < m - 1:
            # compute next tuple
            temp_j = temp[j]
            if j:
                temp[j] = 0
                temp[0] = temp_j
            if temp_j > 1:
                temp[j + 1] += 1
                j = 0
            else:
                j += 1
                temp[j] += 1

            temp[0] -= 1
            temp_a = tuple(temp)
            b = tuple(filter(None, temp_a))
            yield (temp_a, coefficients[b])


def _multinomial_coefficients(m, n):
    """Return an iterator of multinomial coefficients

    Based-on/forked from sympy's multinomial_coefficients() function [#]

    .. [#] https://github.com/sympy/sympy/blob/sympy-1.5.1/sympy/ntheory/multinomial.py
    """
    if not m:
        if n:
            return {}
        return {(): 1}
    if m == 2:
        return _binomial_coefficients(n)
    if m >= 2 * n and n > 1:
        return dict(_large_coefficients_iter(m, n))
    if n:
        j = 0
    else:
        j = m
    temp = [n] + [0] * (m - 1)
    res = {tuple(temp): 1}
    while j < m - 1:
        temp_j = temp[j]
        if j:
            temp[j] = 0
            temp[0] = temp_j
        if temp_j > 1:
            temp[j + 1] += 1
            j = 0
            start = 1
            v = 0
        else:
            j += 1
            start = j + 1
            v = res[tuple(temp)]
            temp[j] += 1
        for k in range(start, m):
            if temp[k]:
                temp[k] -= 1
                v += res[tuple(temp)]
                temp[k] += 1
        temp[0] -= 1
        res[tuple(temp)] = (v * temp_j) // (n - temp[0])
    return res


class PolynomialPauliRotations(FunctionalPauliRotations):
    r"""A circuit implementing polynomial Pauli rotations.

    For a polynomial :math:`p`, a basis state :math:`|i\rangle` of the state register,
    and a target qubit initialized in :math:`|0\rangle`, this operator applies a
    Pauli rotation to the target qubit by the angle :math:`p(i)`. For the default
    ``basis="Y"``, this acts as:

    .. math::

        |0\rangle |i\rangle \mapsto
        \cos\left(\frac{p(i)}{2}\right) |0\rangle |i\rangle
        + \sin\left(\frac{p(i)}{2}\right) |1\rangle |i\rangle

    For ``basis="X"``, the action on the target qubit is:

    .. math::

        |0\rangle |i\rangle \mapsto
        \cos\left(\frac{p(i)}{2}\right) |0\rangle |i\rangle
        - i\sin\left(\frac{p(i)}{2}\right) |1\rangle |i\rangle

    For ``basis="Z"``, the action on the target qubit is:

    .. math::

        |0\rangle |i\rangle \mapsto e^{-i p(i) / 2} |0\rangle |i\rangle

    Let :math:`n` be the number of qubits representing the state, :math:`d` the degree of :math:`p`
    and :math:`q_i` the qubits, where :math:`q_0` is the least significant qubit. Then for

    .. math::

        x = \sum_{i=0}^{n-1} 2^i q_i,

    we can write

    .. math::

        p(x) = \sum_{j=0}^{j=d} c_j x^j

    where :math:`c` are the input coefficients, ``coeffs``.
    """

    def __init__(
        self,
        num_state_qubits: int | None = None,
        coeffs: list[float] | None = None,
        basis: str = "Y",
        name: str = "poly",
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits representing the state.
            coeffs: The coefficients of the polynomial. ``coeffs[i]`` is the coefficient of
                :math:`x^i`. Defaults to linear: ``[0, 1]``.
            basis: The type of Pauli rotation (``"X"``, ``"Y"``, ``"Z"``).
            name: The name of the circuit.
        """
        # set default internal parameters
        self._coeffs = coeffs or [0, 1]

        # initialize super (after setting coeffs)
        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def coeffs(self) -> list[float]:
        """The coefficients of the polynomial.

        ``coeffs[i]`` is the coefficient of the i-th power of the function input :math:`x`,
        that means that the rotation angles are based on the coefficients value,
        following the formula

        .. math::

            c_j x^j ,  j=0, ..., d

        where :math:`d` is the degree of the polynomial :math:`p(x)` and :math:`c` are the coefficients
        ``coeffs``.

        Returns:
            The coefficients of the polynomial.
        """
        return self._coeffs

    @coeffs.setter
    def coeffs(self, coeffs: list[float]) -> None:
        """Set the coefficients of the polynomial.

        ``coeffs[i]`` is the coefficient of the i-th power of x.

        Args:
            coeffs: The coefficients of the polynomial.
        """
        self._invalidate()
        self._coeffs = coeffs

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial, equals to the number of coefficients minus 1.

        Returns:
            The degree of the polynomial. If the coefficients have not been set, return 0.
        """
        if self.coeffs:
            return len(self.coeffs) - 1
        return 0

    def _reset_registers(self, num_state_qubits):
        """Reset the registers."""
        if num_state_qubits is not None:
            # set new register of appropriate size
            qr_state = QuantumRegister(num_state_qubits, name="state")
            qr_target = QuantumRegister(1, name="target")

            self.qregs = [qr_state, qr_target]
        else:
            self.qregs = []

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if self.num_state_qubits is None:
            valid = False
            if raise_on_failure:
                raise AttributeError("The number of qubits has not been set.")

        if self.num_qubits < self.num_state_qubits + 1:
            valid = False
            if raise_on_failure:
                raise CircuitError(
                    "Not enough qubits in the circuit, need at least "
                    f"{self.num_state_qubits + 1}."
                )

        return valid

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        gate = PolynomialPauliRotationsGate(self.num_state_qubits, self.coeffs, self.basis)
        self.append(gate, self.qubits)


class PolynomialPauliRotationsGate(Gate):
    r"""A gate implementing polynomial Pauli rotations.

    For a polynomial :math:`p`, a basis state :math:`|i\rangle` of the state register,
    and a target qubit initialized in :math:`|0\rangle`, this operator applies a
    Pauli rotation to the target qubit by the angle :math:`p(i)`. For the default
    ``basis="Y"``, this acts as:

    .. math::

        |0\rangle |i\rangle \mapsto
        \cos\left(\frac{p(i)}{2}\right) |0\rangle |i\rangle
        + \sin\left(\frac{p(i)}{2}\right) |1\rangle |i\rangle

    For ``basis="X"``, the action on the target qubit is:

    .. math::

        |0\rangle |i\rangle \mapsto
        \cos\left(\frac{p(i)}{2}\right) |0\rangle |i\rangle
        - i\sin\left(\frac{p(i)}{2}\right) |1\rangle |i\rangle

    For ``basis="Z"``, the action on the target qubit is:

    .. math::

        |0\rangle |i\rangle \mapsto e^{-i p(i) / 2} |0\rangle |i\rangle

    Let :math:`n` be the number of qubits representing the state, :math:`d` the degree of :math:`p`
    and :math:`q_i` the qubits, where :math:`q_0` is the least significant qubit. Then for

    .. math::

        x = \sum_{i=0}^{n-1} 2^i q_i,

    we can write

    .. math::

        p(x) = \sum_{j=0}^{j=d} c_j x^j

    where :math:`c` are the input coefficients, ``coeffs``.
    """

    def __init__(
        self,
        num_state_qubits: int,
        coeffs: list[float] | None = None,
        basis: str = "Y",
        label: str | None = None,
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits representing the state.
            coeffs: The coefficients of the polynomial. ``coeffs[i]`` is the coefficient of
                :math:`x^i`. Defaults to linear: ``[0, 1]``.
            basis: The type of Pauli rotation (``"X"``, ``"Y"``, ``"Z"``).
            label: A label for the gate.
        """
        self.coeffs = coeffs or [0, 1]
        self.basis = basis.lower()
        super().__init__("PolyPauli", num_state_qubits + 1, [], label=label)

    def _define(self):
        circuit = QuantumCircuit(self.num_qubits)
        qr_state = circuit.qubits[: self.num_qubits - 1]
        qr_target = circuit.qubits[-1]

        rotation_coeffs = self._get_rotation_coefficients()

        if self.basis == "x":
            circuit.rx(self.coeffs[0], qr_target)
        elif self.basis == "y":
            circuit.ry(self.coeffs[0], qr_target)
        else:
            circuit.rz(self.coeffs[0], qr_target)

        for c in rotation_coeffs:
            qr_control = []
            for i, _ in enumerate(c):
                if c[i] > 0:
                    qr_control.append(qr_state[i])

            # apply controlled rotations
            if len(qr_control) > 1:
                if self.basis == "x":
                    circuit.mcrx(rotation_coeffs[c], qr_control, qr_target)
                elif self.basis == "y":
                    circuit.mcry(rotation_coeffs[c], qr_control, qr_target)
                else:
                    circuit.mcrz(rotation_coeffs[c], qr_control, qr_target)

            elif len(qr_control) == 1:
                if self.basis == "x":
                    circuit.crx(rotation_coeffs[c], qr_control[0], qr_target)
                elif self.basis == "y":
                    circuit.cry(rotation_coeffs[c], qr_control[0], qr_target)
                else:
                    circuit.crz(rotation_coeffs[c], qr_control[0], qr_target)

        self.definition = circuit

    def _get_rotation_coefficients(self) -> dict[tuple[int, ...], float]:
        """Compute the coefficient of each monomial.

        Returns:
            A dictionary with pairs ``{control_state: rotation angle}`` where ``control_state``
            is a tuple of ``0`` or ``1`` bits.
        """
        # determine the control states
        num_state_qubits = self.num_qubits - 1
        degree = len(self.coeffs) - 1
        all_combinations = list(product([0, 1], repeat=num_state_qubits))
        valid_combinations = []
        for combination in all_combinations:
            if 0 < sum(combination) <= degree:
                valid_combinations += [combination]

        rotation_coeffs = dict.fromkeys(valid_combinations, 0.0)

        # compute the coefficients for the control states
        for i, coeff in enumerate(self.coeffs[1:]):
            i += 1  # since we skip the first element we need to increase i by one

            # iterate over the multinomial coefficients
            for comb, num_combs in _multinomial_coefficients(num_state_qubits, i).items():
                control_state: tuple[int, ...] = ()
                power = 1
                for j, qubit in enumerate(comb):
                    if qubit > 0:  # means we control on qubit i
                        control_state += (1,)
                        power *= 2 ** (j * qubit)
                    else:
                        control_state += (0,)

                # Add angle
                rotation_coeffs[control_state] += coeff * num_combs * power

        return rotation_coeffs
