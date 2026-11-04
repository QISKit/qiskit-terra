# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A generic implementation of Approximate Quantum Compiler."""
from __future__ import annotations

import typing
from functools import partial

from collections.abc import Callable
from typing import Protocol

import numpy as np

from qiskit.quantum_info import Operator

from .approximate import ApproximateCircuit, ApproximatingObjective

if typing.TYPE_CHECKING:
    import scipy.optimize


class Minimizer(Protocol):
    """Callable Protocol for minimizer.

    This interface is based on `SciPy's optimize module
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`__.

     This protocol defines a callable taking the following parameters:

         fun
             The objective function to minimize.
         x0
             The initial point for the optimization.
         jac
             The gradient of the objective function.
         bounds
             Parameters bounds for the optimization. Note that these might not be supported
             by all optimizers.

     and which returns a SciPy minimization result object.
    """

    def __call__(
        self,
        fun: Callable[[np.ndarray], float],
        x0: np.ndarray,
        jac: Callable[[np.ndarray], np.ndarray] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> scipy.optimize.OptimizeResult:
        """Minimize the objective function.

        This interface is based on `SciPy's optimize module <https://docs.scipy.org/doc
        /scipy/reference/generated/scipy.optimize.minimize.html>`__.

        Args:
            fun: The objective function to minimize.
            x0: The initial point for the optimization.
            jac: The gradient of the objective function.
            bounds: Parameters bounds for the optimization. Note that these might not be supported
                by all optimizers.

        Returns:
             The SciPy minimization result object.
        """
        ...


class AQC:
    """
    A generic implementation of the Approximate Quantum Compiler. This implementation is agnostic of
    the underlying implementation of the approximate circuit, objective, and optimizer. Users may
    pass corresponding implementations of the abstract classes:

    * The *optimizer* is an implementation of the :class:`~.Minimizer` protocol, a callable used to run
      the optimization process. The choice of optimizer may affect overall convergence, required time
      for the optimization process and achieved objective value.

    * The *approximate circuit* represents a template which parameters we want to optimize.  Currently,
      there's only one implementation based on 4-rotations CNOT unit blocks:
      :class:`.CNOTUnitCircuit`. See the paper for more details.

    * The *approximate objective* is tightly coupled with the approximate circuit implementation and
      provides two methods for computing objective function and gradient with respect to approximate
      circuit parameters. This objective is passed to the optimizer. Currently, there are two
      implementations based on 4-rotations CNOT unit blocks: :class:`.DefaultCNOTUnitObjective` and
      its accelerated version :class:`.FastCNOTUnitObjective`. Both implementations share the same
      idea of maximization the Hilbert-Schmidt product between the target matrix and its
      approximation. The former implementation approach should be considered as a baseline one. It
      may suffer from performance issues, and is mostly suitable for a small number of qubits
      (up to 5 or 6), whereas the latter, accelerated one, can be applied to larger problems.

    * One should take into consideration the exponential growth of matrix size with the number of
      qubits because the implementation not only creates a potentially large target matrix, but
      also allocates a number of temporary memory buffers comparable in size to the target matrix.
    """

    def __init__(
        self,
        optimizer: Minimizer | None = None,
        seed: int | None = None,
    ):
        """
        Args:
            optimizer: an optimizer to be used in the optimization procedure of the search for
                the best approximate circuit. By default, the scipy minimizer with the
                ``L-BFGS-B`` method is used with max iterations set to 1000.
            seed: a seed value to be used by a random number generator.
        """
        import scipy.optimize

        super().__init__()
        self._optimizer = optimizer or partial(
            scipy.optimize.minimize, args=(), method="L-BFGS-B", options={"maxiter": 1000}
        )

        self._seed = seed

    def compile_unitary(
        self,
        target_matrix: np.ndarray,
        approximate_circuit: ApproximateCircuit,
        approximating_objective: ApproximatingObjective,
        initial_point: np.ndarray | None = None,
    ) -> None:
        """
        Approximately compiles a circuit represented as a unitary matrix by solving an optimization
        problem defined by ``approximating_objective`` and using ``approximate_circuit`` as a
        template for the approximate circuit.

        Args:
            target_matrix: a unitary matrix to approximate.
            approximate_circuit: a template circuit that will be filled with the parameter values
                obtained in the optimization procedure.
            approximating_objective: a definition of the optimization problem.
            initial_point: initial values of angles/parameters to start optimization from.
        """
        matrix_dim = target_matrix.shape[0]
        # check if it is actually a special unitary matrix
        target_det = np.linalg.det(target_matrix)
        if not np.isclose(target_det, 1):
            su_matrix = target_matrix / np.power(target_det, (1 / matrix_dim), dtype=complex)
            global_phase_required = True
        else:
            su_matrix = target_matrix
            global_phase_required = False

        # set the matrix to approximate in the algorithm
        approximating_objective.target_matrix = su_matrix

        if initial_point is None:
            np.random.seed(self._seed)
            initial_point = np.random.uniform(0, 2 * np.pi, approximating_objective.num_thetas)

        opt_result = self._optimizer(
            fun=approximating_objective.objective,
            x0=initial_point,
            jac=approximating_objective.gradient,
        )

        approximate_circuit.build(opt_result.x)

        approx_matrix = Operator(approximate_circuit).data

        if global_phase_required:
            alpha = np.angle(np.trace(np.dot(approx_matrix.conj().T, target_matrix)))
            approximate_circuit.global_phase = alpha
