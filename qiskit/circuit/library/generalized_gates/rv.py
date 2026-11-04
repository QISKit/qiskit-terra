# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rotation around an arbitrary axis on the Bloch sphere."""

import math
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.exceptions import CircuitError


class RVGate(Gate):
    r"""Rotation around arbitrary rotation axis :math:`\vec{v}` where :math:`\|\vec{v}\|_2` is
    angle of rotation in radians.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rv` method.

    Circuit symbol:

    .. code-block:: text

             ┌─────────────────┐
        q_0: ┤ RV(v_x,v_y,v_z) ├
             └─────────────────┘

    Matrix representation:

    .. math::

        \newcommand{\rotationangle}{\frac{\|\vec{v}\|_2}{2}}
            R(\vec{v}) = e^{-i \vec{v}\cdot\vec{\sigma} / 2} =
                \begin{pmatrix}
                    \cos\left(\rotationangle\right)
                    -i \frac{v_z}{\|\vec{v}\|_2} \sin\left(\rotationangle\right)
                    & -(i \frac{v_x}{\|\vec{v}\|_2}
                    + \frac{v_y}{\|\vec{v}\|_2}) \sin\left(\rotationangle\right) \\
                    -(i \frac{v_x}{\|\vec{v}\|_2}
                    - \frac{v_y}{\|\vec{v}\|_2}) \sin\left(\rotationangle\right)
                    & \cos\left(\rotationangle\right)
                    + i \frac{v_z}{\|\vec{v}\|_2} \sin\left(\rotationangle\right)
                \end{pmatrix}
    """

    def __init__(
        self,
        v_x: float,
        v_y: float,
        v_z: float,
        basis: str = "U",
        label: str | None = None,
    ):
        """
        Args:
            v_x: x-component
            v_y: y-component
            v_z: z-component
            basis: basis (see
                :class:`~qiskit.synthesis.one_qubit.one_qubit_decompose.OneQubitEulerDecomposer`)
            label: An optional label for the gate.
        """

        from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer

        super().__init__("rv", 1, [v_x, v_y, v_z], label=label)
        self._decomposer = OneQubitEulerDecomposer(basis=basis)

    def _define(self):
        try:
            self.definition = self._decomposer(self.to_matrix())
        except TypeError as ex:
            raise CircuitError(
                f"The {self.name} gate cannot be decomposed with unbound parameters"
            ) from ex

    def inverse(self, annotated: bool = False):
        """Invert this gate."""
        vx, vy, vz = self.params
        return RVGate(-vx, -vy, -vz)

    def to_matrix(self) -> numpy.ndarray:
        """Return a numpy.array for the R(v) gate."""
        v = numpy.asarray(self.params, dtype=float)
        angle = math.sqrt(v.dot(v))
        if angle == 0:
            return numpy.array([[1, 0], [0, 1]])
        nx, ny, nz = v / angle
        sin = math.sin(angle / 2)
        cos = math.cos(angle / 2)
        return numpy.array(
            [
                [cos - 1j * nz * sin, (-ny - 1j * nx) * sin],
                [(ny - 1j * nx) * sin, cos + 1j * nz * sin],
            ]
        )
