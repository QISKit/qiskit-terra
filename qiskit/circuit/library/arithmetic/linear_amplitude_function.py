# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A class implementing a (piecewise-) linear function on qubit amplitudes."""

from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.utils.deprecation import deprecate_func

from .piecewise_linear_pauli_rotations import (
    PiecewiseLinearPauliRotations,
    PiecewiseLinearPauliRotationsGate,
)


class LinearAmplitudeFunction(QuantumCircuit):
    r"""A circuit implementing a (piecewise) linear function on qubit amplitudes.

    An amplitude function :math:`F` of a function :math:`f` is a mapping

    .. math::

        F|x\rangle|0\rangle = \sqrt{1 - \hat{f}(x)} |x\rangle|0\rangle + \sqrt{\hat{f}(x)}
            |x\rangle|1\rangle.

    for a function :math:`\hat{f}: \{ 0, ..., 2^n - 1 \} \rightarrow [0, 1]`, where
    :math:`|x\rangle` is a :math:`n` qubit state.

    This circuit implements :math:`F` for piecewise linear functions :math:`\hat{f}`.
    In this case, the mapping :math:`F` can be approximately implemented using a Taylor expansion
    and linearly controlled Pauli-Y rotations, see [1, 2] for more detail. This approximation
    uses a ``rescaling_factor`` to determine the accuracy of the Taylor expansion.

    In general, the function of interest :math:`f` is defined from some interval :math:`[a,b]`,
    the ``domain`` to :math:`[c,d]`, the ``image``, instead of :math:`\{ 1, ..., N \}` to
    :math:`[0, 1]`. Using an affine transformation we can rescale :math:`f` to :math:`\hat{f}`:

    .. math::

        \hat{f}(x) = \frac{f(\phi(x)) - c}{d - c}

    with

    .. math::

        \phi(x) = a + \frac{b - a}{2^n - 1} x.

    If :math:`f` is a piecewise linear function on :math:`m` intervals
    :math:`[p_{i-1}, p_i], i \in \{1, ..., m\}` with slopes :math:`\alpha_i` and
    offsets :math:`\beta_i` it can be written as

    .. math::

        f(x) = \sum_{i=1}^m 1_{[p_{i-1}, p_i]}(x) (\alpha_i x + \beta_i)

    where :math:`1_{[a, b]}` is an indicator function that is 1 if the argument is in the interval
    :math:`[a, b]` and otherwise 0. The breakpoints :math:`p_i` can be specified by the
    ``breakpoints`` argument.

    References:

    [1] Woerner, S., & Egger, D. J. (2018). Quantum Risk Analysis.
    `arXiv:1806.06893 <https://arxiv.org/abs/1806.06893>`_

    [2] Gacon, J., Zoufal, C., & Woerner, S. (2020). Quantum-Enhanced Simulation-Based Optimization.
    `arXiv:2005.10780 <https://arxiv.org/abs/2005.10780>`_
    """

    @deprecate_func(
        since="2.2",
        additional_msg="Use the class qiskit.circuit.library.LinearAmplitudeFunctionGate instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_state_qubits: int,
        slope: float | list[float],
        offset: float | list[float],
        domain: tuple[float, float],
        image: tuple[float, float],
        rescaling_factor: float = 1,
        breakpoints: list[float] | None = None,
        name: str = "F",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits used to encode the variable :math:`x`.
            slope: The slope of the linear function. Can be a list of slopes if it is a piecewise
                linear function.
            offset: The offset of the linear function. Can be a list of offsets if it is a piecewise
                linear function.
            domain: The domain of the function as tuple :math:`(x_\min{}, x_\max{})`.
            image: The image of the function as tuple :math:`(f_\min{}, f_\max{})`.
            rescaling_factor: The rescaling factor to adjust the accuracy in the Taylor
                approximation.
            breakpoints: The breakpoints if the function is piecewise linear. If None, the function
                is not piecewise.
            name: Name of the circuit.

        """

        if not hasattr(slope, "__len__"):
            slope = [slope]
        if not hasattr(offset, "__len__"):
            offset = [offset]

        # ensure that the breakpoints include the first point of the domain
        if breakpoints is None:
            breakpoints = [domain[0]]
        elif not np.isclose(breakpoints[0], domain[0]):
            breakpoints = [domain[0]] + breakpoints

        _check_sizes_match(slope, offset, breakpoints)
        _check_sorted_and_in_range(breakpoints, domain)

        self._domain = domain
        self._image = image
        self._rescaling_factor = rescaling_factor

        # do rescaling
        a, b = domain
        c, d = image

        mapped_breakpoints = []
        mapped_slope = []
        mapped_offset = []
        for i, point in enumerate(breakpoints):
            mapped_breakpoint = (point - a) / (b - a) * (2**num_state_qubits - 1)
            mapped_breakpoints += [mapped_breakpoint]

            # factor (upper - lower) / (2^n - 1) is for the scaling of x to [l,u]
            # note that the +l for mapping to [l,u] is already included in
            # the offsets given as parameters
            mapped_slope += [slope[i] * (b - a) / (2**num_state_qubits - 1)]
            mapped_offset += [offset[i]]

        # approximate linear behavior by scaling and contracting around pi/4
        slope_angles = np.zeros(len(breakpoints))
        offset_angles = np.pi / 4 * (1 - rescaling_factor) * np.ones(len(breakpoints))
        for i in range(len(breakpoints)):
            slope_angles[i] = np.pi * rescaling_factor * mapped_slope[i] / 2 / (d - c)
            offset_angles[i] += np.pi * rescaling_factor * (mapped_offset[i] - c) / 2 / (d - c)

        # use PWLPauliRotations to implement the function
        pwl_pauli_rotation = PiecewiseLinearPauliRotations(
            num_state_qubits, mapped_breakpoints, 2 * slope_angles, 2 * offset_angles, name=name
        )

        super().__init__(*pwl_pauli_rotation.qregs, name=name)
        self.append(pwl_pauli_rotation.to_gate(), self.qubits)

    def post_processing(self, scaled_value: float) -> float:
        r"""Map the function value of the approximated :math:`\hat{f}` to :math:`f`.

        Args:
            scaled_value: A function value from the Taylor expansion of :math:`\hat{f}(x)`.

        Returns:
            The ``scaled_value`` mapped back to the domain of :math:`f`, by first inverting
            the transformation used for the Taylor approximation and then mapping back from
            :math:`[0, 1]` to the original domain.
        """
        # revert the mapping applied in the Taylor approximation
        value = scaled_value - 1 / 2 + np.pi / 4 * self._rescaling_factor
        value *= 2 / np.pi / self._rescaling_factor

        # map the value from [0, 1] back to the original domain
        value *= self._image[1] - self._image[0]
        value += self._image[0]

        return value


class LinearAmplitudeFunctionGate(Gate):
    r"""A circuit implementing a (piecewise) linear function on qubit amplitudes.

    An amplitude function :math:`F` of a function :math:`f` is a mapping

    .. math::

        F|x\rangle|0\rangle = \sqrt{1 - \hat{f}(x)} |x\rangle|0\rangle + \sqrt{\hat{f}(x)}
            |x\rangle|1\rangle.

    for a function :math:`\hat{f}: \{ 0, ..., 2^n - 1 \} \rightarrow [0, 1]`, where
    :math:`|x\rangle` is a :math:`n` qubit state.

    This circuit implements :math:`F` for piecewise linear functions :math:`\hat{f}`.
    In this case, the mapping :math:`F` can be approximately implemented using a Taylor expansion
    and linearly controlled Pauli-Y rotations, see [1, 2] for more detail. This approximation
    uses a ``rescaling_factor`` to determine the accuracy of the Taylor expansion.

    In general, the function of interest :math:`f` is defined from some interval :math:`[a,b]`,
    the ``domain`` to :math:`[c,d]`, the ``image``, instead of :math:`\{ 1, ..., N \}` to
    :math:`[0, 1]`. Using an affine transformation we can rescale :math:`f` to :math:`\hat{f}`:

    .. math::

        \hat{f}(x) = \frac{f(\phi(x)) - c}{d - c}

    with

    .. math::

        \phi(x) = a + \frac{b - a}{2^n - 1} x.

    If :math:`f` is a piecewise linear function on :math:`m` intervals
    :math:`[p_{i-1}, p_i], i \in \{1, ..., m\}` with slopes :math:`\alpha_i` and
    offsets :math:`\beta_i` it can be written as

    .. math::

        f(x) = \sum_{i=1}^m 1_{[p_{i-1}, p_i]}(x) (\alpha_i x + \beta_i)

    where :math:`1_{[a, b]}` is an indicator function that is 1 if the argument is in the interval
    :math:`[a, b]` and otherwise 0. The breakpoints :math:`p_i` can be specified by the
    ``breakpoints`` argument.

    References:

    [1] Woerner, S., & Egger, D. J. (2018).
    Quantum Risk Analysis.
    `arXiv:1806.06893 <https://arxiv.org/abs/1806.06893>`_

    [2] Gacon, J., Zoufal, C., & Woerner, S. (2020).
    Quantum-Enhanced Simulation-Based Optimization.
    `arXiv:2005.10780 <https://arxiv.org/abs/2005.10780>`_
    """

    def __init__(
        self,
        num_state_qubits: int,
        slope: float | list[float],
        offset: float | list[float],
        domain: tuple[float, float],
        image: tuple[float, float],
        rescaling_factor: float = 1,
        breakpoints: list[float] | None = None,
        label: str = "F",
    ) -> None:
        r"""
        Args:
            num_state_qubits: The number of qubits used to encode the variable :math:`x`.
            slope: The slope of the linear function. Can be a list of slopes if it is a piecewise
                linear function.
            offset: The offset of the linear function. Can be a list of offsets if it is a piecewise
                linear function.
            domain: The domain of the function as tuple :math:`(x_\min{}, x_\max{})`.
            image: The image of the function as tuple :math:`(f_\min{}, f_\max{})`.
            rescaling_factor: The rescaling factor to adjust the accuracy in the Taylor
                approximation.
            breakpoints: The breakpoints if the function is piecewise linear. If None, the function
                is not piecewise.
            label: A label for the gate.
        """
        if not hasattr(slope, "__len__"):
            slope = [slope]
        if not hasattr(offset, "__len__"):
            offset = [offset]

        # ensure that the breakpoints include the first point of the domain
        if breakpoints is None:
            breakpoints = [domain[0]]
        elif not np.isclose(breakpoints[0], domain[0]):
            breakpoints = [domain[0]] + breakpoints

        _check_sizes_match(slope, offset, breakpoints)
        _check_sorted_and_in_range(breakpoints, domain)

        self.slope = slope
        self.offset = offset
        self.domain = domain
        self.image = image
        self.rescaling_factor = rescaling_factor
        self.breakpoints = breakpoints

        num_compare = int(len(breakpoints) > 1)
        super().__init__("LinFunction", num_state_qubits + num_compare + 1, [], label=label)

    def _define(self):
        num_compare = int(len(self.breakpoints) > 1)
        num_state_qubits = self.num_qubits - num_compare - 1

        # do rescaling
        a, b = self.domain
        c, d = self.image

        mapped_breakpoints = []
        mapped_slope = []
        mapped_offset = []
        for i, point in enumerate(self.breakpoints):
            mapped_breakpoint = (point - a) / (b - a) * (2**num_state_qubits - 1)
            mapped_breakpoints += [mapped_breakpoint]

            # factor (upper - lower) / (2^n - 1) is for the scaling of x to [l,u]
            # note that the +l for mapping to [l,u] is already included in
            # the offsets given as parameters
            mapped_slope += [self.slope[i] * (b - a) / (2**num_state_qubits - 1)]
            mapped_offset += [self.offset[i]]

        # approximate linear behavior by scaling and contracting around pi/4
        slope_angles = np.zeros(len(self.breakpoints))
        offset_angles = np.pi / 4 * (1 - self.rescaling_factor) * np.ones_like(slope_angles)
        for i, (slope_i, offset_i) in enumerate(zip(mapped_slope, mapped_offset)):
            slope_angles[i] = np.pi * self.rescaling_factor * slope_i / 2 / (d - c)
            offset_angles[i] += np.pi * self.rescaling_factor * (offset_i - c) / 2 / (d - c)

        # use PWLPauliRotations to implement the function
        pwl_pauli_rotation = PiecewiseLinearPauliRotationsGate(
            num_state_qubits, mapped_breakpoints, 2 * slope_angles, 2 * offset_angles
        )

        self.definition = QuantumCircuit(pwl_pauli_rotation.num_qubits)
        self.definition.append(pwl_pauli_rotation, self.definition.qubits)

    def post_processing(self, scaled_value: float) -> float:
        r"""Map the function value of the approximated :math:`\hat{f}` to :math:`f`.

        Args:
            scaled_value: A function value from the Taylor expansion of :math:`\hat{f}(x)`.

        Returns:
            The ``scaled_value`` mapped back to the domain of :math:`f`, by first inverting
            the transformation used for the Taylor approximation and then mapping back from
            :math:`[0, 1]` to the original domain.
        """
        # revert the mapping applied in the Taylor approximation
        value = scaled_value - 1 / 2 + np.pi / 4 * self.rescaling_factor
        value *= 2 / np.pi / self.rescaling_factor

        # map the value from [0, 1] back to the original domain
        value *= self.image[1] - self.image[0]
        value += self.image[0]

        return value


def _check_sorted_and_in_range(breakpoints, domain):
    if breakpoints is None:
        return

    # check if sorted
    if not np.all(np.diff(breakpoints) > 0):
        raise ValueError("Breakpoints must be unique and sorted.")

    if breakpoints[0] < domain[0] or breakpoints[-1] > domain[1]:
        raise ValueError("Breakpoints must be included in domain.")


def _check_sizes_match(slope, offset, breakpoints):
    size = len(slope)
    if len(offset) != size:
        raise ValueError(f"Size mismatch of slope ({size}) and offset ({len(offset)}).")
    if breakpoints is not None:
        if len(breakpoints) != size:
            raise ValueError(
                f"Size mismatch of slope ({size}) and breakpoints ({len(breakpoints)})."
            )
