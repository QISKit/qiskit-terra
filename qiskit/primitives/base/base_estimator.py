# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base Estimator V1 and V2 classes"""

from __future__ import annotations

from abc import abstractmethod, ABC
from collections.abc import Iterable, Sequence
from copy import copy
from typing import Generic, TypeVar

from qiskit.circuit import QuantumCircuit
from qiskit.providers import JobV1 as Job
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from ..containers import (
    DataBin,
    EstimatorPubLike,
    PrimitiveResult,
    PubResult,
)
from ..containers.estimator_pub import EstimatorPub
from .validation_v1 import _validate_estimator_args
from .base_primitive_v1 import BasePrimitiveV1
from .base_primitive_job import BasePrimitiveJob

T = TypeVar("T", bound=Job)


class BaseEstimatorV2(ABC):
    r"""Base class for ``EstimatorV2`` implementations.

    An estimator calculates expectation values for provided quantum circuit and
    observable combinations. Implementations of this :class:`.BaseEstimatorV2`
    interface must define their own :meth:`.run` method, which is designed to
    take the following inputs:

     * pubs: list of pubs (Primitive Unified Blocs). An estimator pub is a list
        or tuple of two to four elements that define the unit of work for the
        estimator. These are:

        * A single :class:`~qiskit.circuit.QuantumCircuit`, possibly parameterized,
            whose final state we define as :math:`\psi(\theta)`.

        * One or more observables (specified as any :class:`~.ObservablesArrayLike`, including
            :class:`~.quantum_info.Pauli`, :class:`~.SparsePauliOp`, ``str``) that specify which
            expectation values to estimate, denoted :math:`H_j`.

        * A collection parameter value sets to bind the circuit against, :math:`\theta_k`

        * Optionally, the estimation precision.

     * precision: the estimation precision. This specification is optional and will be overridden by
        the pub-wise precision if provided.

    All estimator implementations must implement default value for the ``precision`` in the
    :meth:`.run` method. This default value will be used any time ``precision=None`` is specified, which
    can take place in the :meth:`.run` kwargs or at the pub level.
    """

    @staticmethod
    def _make_data_bin(_: EstimatorPub) -> type[DataBin]:
        # this method is present for backwards compat. new primitive implementations
        # should avoid it.
        return DataBin

    @abstractmethod
    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> BasePrimitiveJob[PrimitiveResult[PubResult]]:
        """Estimate expectation values for each provided pub (Primitive Unified Bloc).

        Args:
            pubs: An iterable of pub-like objects, such as tuples ``(circuit, observables)``
                  or ``(circuit, observables, parameter_values)``.
            precision: The target precision for expectation value estimates of each
                       run Estimator Pub that does not specify its own precision. If None
                       the estimator's default precision value will be used.

        Returns:
            A job object that contains results.
        """


class BaseEstimatorV1(BasePrimitiveV1, Generic[T]):
    r"""Base class for ``EstimatorV1`` implementations.

    Note that the reference estimator in Qiskit follows the ``EstimatorV2``
    interface specifications instead.

    An estimator calculates expectation values for provided quantum circuit and
    observable combinations.

    Implementations of :class:`.BaseEstimatorV1` should define their own
    :meth:`.BaseEstimatorV1._run` method
    that will be called by the public-facing :meth:`qiskit.primitives.BaseEstimatorV1.run`,
    which takes the following inputs:

    * quantum circuits (:math:`\psi_i(\theta)`): list of (parameterized) quantum circuits
      (a list of :class:`~qiskit.circuit.QuantumCircuit` objects).

    * observables (:math:`H_j`): a list of :class:`~qiskit.quantum_info.SparsePauliOp`
      objects.

    * parameter values (:math:`\theta_k`): list of sets of values
      to be bound to the parameters of the quantum circuits
      (list of list of float).

    The method returns a :class:`~qiskit.providers.JobV1` object. Calling
    :meth:`qiskit.providers.JobV1.result()` yields
    a list of expectation values plus optional metadata like confidence intervals for
    the estimation.

    .. math::

        \langle\psi_i(\theta_k)|H_j|\psi_i(\theta_k)\rangle

    Here is an example of how a :class:`.BaseEstimatorV1` would be used:

    .. code-block:: python

        # This is a fictional import path.
        # There are currently no EstimatorV1 implementations in Qiskit.
        from estimator_v1_location import EstimatorV1
        from qiskit.circuit.library import RealAmplitudes
        from qiskit.quantum_info import SparsePauliOp

        psi1 = RealAmplitudes(num_qubits=2, reps=2)
        psi2 = RealAmplitudes(num_qubits=2, reps=3)

        H1 = SparsePauliOp.from_list([("II", 1), ("IZ", 2), ("XI", 3)])
        H2 = SparsePauliOp.from_list([("IZ", 1)])
        H3 = SparsePauliOp.from_list([("ZI", 1), ("ZZ", 1)])

        theta1 = [0, 1, 1, 2, 3, 5]
        theta2 = [0, 1, 1, 2, 3, 5, 8, 13]
        theta3 = [1, 2, 3, 4, 5, 6]

        estimator = EstimatorV1()

        # calculate [ <psi1(theta1)|H1|psi1(theta1)> ]
        job = estimator.run([psi1], [H1], [theta1])
        job_result = job.result() # It will block until the job finishes.
        print(f"The primitive-job finished with result {job_result}")

        # calculate [ <psi1(theta1)|H1|psi1(theta1)>,
        #             <psi2(theta2)|H2|psi2(theta2)>,
        #             <psi1(theta3)|H3|psi1(theta3)> ]
        job2 = estimator.run([psi1, psi2, psi1], [H1, H2, H3], [theta1, theta2, theta3])
        job_result = job2.result()
        print(f"The primitive-job finished with result {job_result}")
    """

    __hash__ = None

    def __init__(
        self,
        *,
        options: dict | None = None,
    ):
        """
        Initialize ``EstimatorV1``.

        Args:
            options: Default options.
        """
        super().__init__(options)

    def run(
        self,
        circuits: Sequence[QuantumCircuit] | QuantumCircuit,
        observables: Sequence[BaseOperator | str] | BaseOperator | str,
        parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None = None,
        **run_options,
    ) -> T:
        """Run the job of the estimation of expectation value(s).

        ``circuits``, ``observables``, and ``parameter_values`` should have the same
        length. The i-th element of the result is the expectation of observable

        .. code-block:: python

            obs = observables[i]

        for the state prepared by

        .. code-block:: python

            circ = circuits[i]

        with bound parameters

        .. code-block:: python

            values = parameter_values[i]

        Args:
            circuits: one or more circuit objects.
            observables: one or more observable objects. Several formats are allowed;
                importantly, ``str`` should follow the string representation format for
                :class:`~qiskit.quantum_info.Pauli` objects.
            parameter_values: concrete parameters to be bound.
            run_options: runtime options used for circuit execution.

        Returns:
            The job object of EstimatorResult.

        Raises:
            TypeError: Invalid argument type given.
            ValueError: Invalid argument values given.
        """
        # Validation
        circuits, observables, parameter_values = _validate_estimator_args(
            circuits, observables, parameter_values
        )

        # Options
        run_opts = copy(self.options)
        run_opts.update_options(**run_options)

        return self._run(
            circuits,
            observables,
            parameter_values,
            **run_opts.__dict__,
        )

    @abstractmethod
    def _run(
        self,
        circuits: tuple[QuantumCircuit, ...],
        observables: tuple[SparsePauliOp, ...],
        parameter_values: tuple[tuple[float, ...], ...],
        **run_options,
    ) -> T:
        raise NotImplementedError("The subclass of BaseEstimatorV1 must implement `_run` method.")
