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


"""Piecewise-linearly-controlled rotation."""

from __future__ import annotations
import warnings
from collections.abc import Sequence
import numpy as np

from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit, Gate
from qiskit.circuit.exceptions import CircuitError

from .functional_pauli_rotations import FunctionalPauliRotations
from .linear_pauli_rotations import LinearPauliRotations, LinearPauliRotationsGate
from .integer_comparator import IntegerComparator, IntegerComparatorGate


class PiecewiseLinearPauliRotations(FunctionalPauliRotations):
    r"""Piecewise-linearly-controlled Pauli rotations.

    For a piecewise linear (not necessarily continuous) function :math:`f(x)`, which is defined
    through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`, where
    :math:`n` is the number of state qubits. Further on, denote the corresponding slopes and
    offsets by :math:`a_j` and :math:`b_j` respectively.
    Then f(x) is defined as:

    .. math::

        f(x) = \begin{cases}
            0, x < x_0 \\
            a_j (x - x_j) + b_j, x_j \leq x < x_{j+1}
            \end{cases}

    where we implicitly assume :math:`x_{J+1} = 2^n`.
    """

    def __init__(
        self,
        num_state_qubits: int | None = None,
        breakpoints: list[int] | None = None,
        slopes: list[float] | np.ndarray | None = None,
        offsets: list[float] | np.ndarray | None = None,
        basis: str = "Y",
        name: str = "pw_lin",
    ) -> None:
        """
        Args:
            num_state_qubits: The number of qubits representing the state.
            breakpoints: The breakpoints to define the piecewise-linear function.
                Defaults to ``[0]``.
            slopes: The slopes for different segments of the piecewise-linear function.
                Defaults to ``[1]``.
            offsets: The offsets for different segments of the piecewise-linear function.
                Defaults to ``[0]``.
            basis: The type of Pauli rotation (``'X'``, ``'Y'``, ``'Z'``).
            name: The name of the circuit.
        """
        # store parameters
        self._breakpoints = breakpoints if breakpoints is not None else [0]
        self._slopes = slopes if slopes is not None else [1]
        self._offsets = offsets if offsets is not None else [0]

        super().__init__(num_state_qubits=num_state_qubits, basis=basis, name=name)

    @property
    def breakpoints(self) -> list[int]:
        """The breakpoints of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitly is ``2**(num_state_qubits + 1)``.
        """
        return self._breakpoints

    @breakpoints.setter
    def breakpoints(self, breakpoints: list[int]) -> None:
        """Set the breakpoints.

        Args:
            breakpoints: The new breakpoints.
        """
        self._invalidate()
        self._breakpoints = breakpoints

        if self.num_state_qubits and breakpoints:
            self._reset_registers(self.num_state_qubits)

    @property
    def slopes(self) -> list[float] | np.ndarray:
        """The slopes of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitly is ``2**(num_state_qubits + 1)``.
        """
        return self._slopes

    @slopes.setter
    def slopes(self, slopes: list[float]) -> None:
        """Set the slopes.

        Args:
            slopes: The new slopes.
        """
        self._invalidate()
        self._slopes = slopes

    @property
    def offsets(self) -> list[float] | np.ndarray:
        """The offsets of the piecewise linear function.

        The function is linear in the intervals ``[point_i, point_{i+1}]`` where the last
        point implicitly is ``2**(num_state_qubits + 1)``.
        """
        return self._offsets

    @offsets.setter
    def offsets(self, offsets: list[float]) -> None:
        """Set the offsets.

        Args:
            offsets: The new offsets.
        """
        self._invalidate()
        self._offsets = offsets

    @property
    def mapped_slopes(self) -> np.ndarray:
        """The slopes mapped to the internal representation.

        Returns:
            The mapped slopes.
        """
        mapped_slopes = np.zeros_like(self.slopes)
        for i, slope in enumerate(self.slopes):
            mapped_slopes[i] = slope - sum(mapped_slopes[:i])

        return mapped_slopes

    @property
    def mapped_offsets(self) -> np.ndarray:
        """The offsets mapped to the internal representation.

        Returns:
            The mapped offsets.
        """
        mapped_offsets = np.zeros_like(self.offsets)
        for i, (offset, slope, point) in enumerate(
            zip(self.offsets, self.slopes, self.breakpoints)
        ):
            mapped_offsets[i] = offset - slope * point - sum(mapped_offsets[:i])

        return mapped_offsets

    @property
    def contains_zero_breakpoint(self) -> bool | np.bool_:
        """Whether 0 is the first breakpoint.

        Returns:
            True, if 0 is the first breakpoint, otherwise False.
        """
        return np.isclose(0, self.breakpoints[0])

    def evaluate(self, x: float) -> float:
        """Classically evaluate the piecewise linear rotation.

        Args:
            x: Value to be evaluated at.

        Returns:
            Value of piecewise linear function at x.
        """

        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (
                x * self.mapped_slopes[i] + self.mapped_offsets[i]
            )

        return y

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

        if len(self.breakpoints) != len(self.slopes) or len(self.breakpoints) != len(self.offsets):
            valid = False
            if raise_on_failure:
                raise ValueError("Mismatching sizes of breakpoints, slopes and offsets.")

        return valid

    def _reset_registers(self, num_state_qubits: int | None) -> None:
        """Reset the registers."""
        self.qregs = []

        if num_state_qubits is not None:
            qr_state = QuantumRegister(num_state_qubits)
            qr_target = QuantumRegister(1)
            self.qregs = [qr_state, qr_target]

            # add ancillas if required
            if len(self.breakpoints) > 1:
                num_ancillas = num_state_qubits
                qr_ancilla = AncillaRegister(num_ancillas)
                self.add_register(qr_ancilla)

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        circuit = QuantumCircuit(*self.qregs, name=self.name)

        qr_state = circuit.qubits[: self.num_state_qubits]
        qr_target = [circuit.qubits[self.num_state_qubits]]
        qr_ancilla = circuit.ancillas

        # apply comparators and controlled linear rotations
        for i, point in enumerate(self.breakpoints):
            if i == 0 and self.contains_zero_breakpoint:
                # deprecation warning is already triggered upon init, filter the rest
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                    # apply rotation
                    lin_r = LinearPauliRotations(
                        num_state_qubits=self.num_state_qubits,
                        slope=self.mapped_slopes[i],
                        offset=self.mapped_offsets[i],
                        basis=self.basis,
                    )
                circuit.append(lin_r.to_gate(), qr_state[:] + qr_target)

            else:
                qr_compare = [qr_ancilla[0]]
                qr_helper = qr_ancilla[1:]

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                    # apply Comparator
                    comp = IntegerComparator(num_state_qubits=self.num_state_qubits, value=point)
                qr = qr_state[:] + qr_compare[:]  # add ancilla as compare qubit

                circuit.append(comp.to_gate(), qr[:] + qr_helper[: comp.num_ancillas])

                # apply controlled rotation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning, module="qiskit")
                    lin_r = LinearPauliRotations(
                        num_state_qubits=self.num_state_qubits,
                        slope=self.mapped_slopes[i],
                        offset=self.mapped_offsets[i],
                        basis=self.basis,
                    )
                circuit.append(
                    lin_r.to_gate().control(annotated=False),
                    qr_compare[:] + qr_state[:] + qr_target,
                )

                # uncompute comparator
                circuit.append(comp.to_gate(), qr[:] + qr_helper[: comp.num_ancillas])

        self.append(circuit.to_gate(), self.qubits)


class PiecewiseLinearPauliRotationsGate(Gate):
    r"""Piecewise-linearly-controlled Pauli rotations.

    For a piecewise linear (not necessarily continuous) function :math:`f(x)`, which is defined
    through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints :math:`(x_0, ..., x_J)` are a subset of :math:`[0, 2^n-1]`, where
    :math:`n` is the number of state qubits. Further on, denote the corresponding slopes and
    offsets by :math:`a_j` and :math:`b_j` respectively.
    Then f(x) is defined as:

    .. math::

        f(x) = \begin{cases}
            0, x < x_0 \\
            a_j (x - x_j) + b_j, x_j \leq x < x_{j+1}
            \end{cases}

    where we implicitly assume :math:`x_{J+1} = 2^n`.
    """

    def __init__(
        self,
        num_state_qubits: int | None = None,
        breakpoints: list[int] | None = None,
        slopes: Sequence[float] | None = None,
        offsets: Sequence[float] | None = None,
        basis: str = "Y",
        label: str | None = None,
    ) -> None:
        """Construct piecewise-linearly-controlled Pauli rotations.

        Args:
            num_state_qubits: The number of qubits representing the state.
            breakpoints: The breakpoints to define the piecewise-linear function.
                Defaults to ``[0]``.
            slopes: The slopes for different segments of the piecewise-linear function.
                Defaults to ``[1]``.
            offsets: The offsets for different segments of the piecewise-linear function.
                Defaults to ``[0]``.
            basis: The type of Pauli rotation (``'X'``, ``'Y'``, ``'Z'``).
            label: The label of the gate.
        """
        self.breakpoints = breakpoints if breakpoints is not None else [0]
        self.slopes = slopes if slopes is not None else [1]
        self.offsets = offsets if offsets is not None else [0]
        self.basis = basis

        num_compare_bits = 1 if len(self.breakpoints) > 1 else 0
        super().__init__("PwLinPauliRot", num_state_qubits + 1 + num_compare_bits, [], label=label)

    def _define(self):
        circuit = QuantumCircuit(self.num_qubits)

        if len(self.breakpoints) == 1:
            qr_state = circuit.qubits[: self.num_qubits - 1]
            qr_target = [circuit.qubits[-1]]
            qr_compare = []
        else:
            qr_state = circuit.qubits[: self.num_qubits - 2]
            qr_target = [circuit.qubits[-2]]
            qr_compare = [circuit.qubits[-1]]

        num_state_qubits = len(qr_state)

        mapped_slopes = np.zeros_like(self.slopes)
        for i, slope in enumerate(self.slopes):
            mapped_slopes[i] = slope - sum(mapped_slopes[:i])

        mapped_offsets = np.zeros_like(self.offsets)
        for i, (offset, slope, point) in enumerate(
            zip(self.offsets, self.slopes, self.breakpoints)
        ):
            mapped_offsets[i] = offset - slope * point - sum(mapped_offsets[:i])

        # apply comparators and controlled linear rotations
        contains_zero_breakpoint = np.isclose(self.breakpoints[0], 0)
        for i, point in enumerate(self.breakpoints):
            if i == 0 and contains_zero_breakpoint:
                # apply rotation
                lin_r = LinearPauliRotationsGate(
                    num_state_qubits=num_state_qubits,
                    slope=mapped_slopes[i],
                    offset=mapped_offsets[i],
                    basis=self.basis,
                )
                circuit.append(lin_r, qr_state[:] + qr_target)

            else:
                # apply Comparator
                comp = IntegerComparatorGate(num_state_qubits=num_state_qubits, value=point)
                qr = qr_state[:] + qr_compare[:]  # add ancilla as compare qubit

                circuit.append(comp, qr[:])

                # apply controlled rotation
                lin_r = LinearPauliRotationsGate(
                    num_state_qubits=num_state_qubits,
                    slope=mapped_slopes[i],
                    offset=mapped_offsets[i],
                    basis=self.basis,
                )
                circuit.append(
                    lin_r.control(annotated=False), qr_compare[:] + qr_state[:] + qr_target
                )

                # uncompute comparator (which is its self-inverse)
                circuit.append(comp, qr[:])

        self.definition = circuit
