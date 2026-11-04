# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Statevector Estimator V2 class
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from qiskit.quantum_info import SparsePauliOp

from .base import BaseEstimatorV2
from .containers import DataBin, EstimatorPubLike, PrimitiveResult, PubResult
from .containers.estimator_pub import EstimatorPub
from .primitive_job import PrimitiveJob
from .utils import _statevector_from_circuit


class StatevectorEstimator(BaseEstimatorV2):
    """
    Simple implementation of :class:`BaseEstimatorV2` with full state vector simulation.

    This class is implemented via :class:`~.Statevector` which turns provided circuits into
    pure state vectors. These states are subsequently acted on by :class:`~.SparsePauliOp`,
    which implies that, at present, this implementation is only compatible with Pauli-based
    observables.

    Each tuple of ``(circuit, observables, <optional> parameter values, <optional> precision)``,
    called an estimator primitive unified bloc (PUB), produces its own array-based result. The
    :meth:`~.StatevectorEstimator.run` method can be given a sequence of pubs to run in one call.


    .. note::
        The result of this class is exact if the circuit contains only unitary operations.
        On the other hand, the result could be stochastic if the circuit contains a non-unitary
        operation such as a reset for some subsystems.
        The stochastic result can be made reproducible by setting ``seed``, e.g.,
        ``StatevectorEstimator(seed=123)``.

    .. plot::
        :alt: Output from the previous code.
        :include-source:

        from qiskit.circuit import Parameter, QuantumCircuit
        from qiskit.primitives import StatevectorEstimator
        from qiskit.quantum_info import Pauli, SparsePauliOp

        import matplotlib.pyplot as plt
        import numpy as np

        # Define a circuit with two parameters.
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ry(Parameter("a"), 0)
        circuit.rz(Parameter("b"), 0)
        circuit.cx(0, 1)
        circuit.h(0)

        # Define a sweep over parameter values, where the second axis is over
        # the two parameters in the circuit.
        params = np.vstack([
            np.linspace(-np.pi, np.pi, 100),
            np.linspace(-4 * np.pi, 4 * np.pi, 100)
        ]).T

        # Define three observables. Many formats are supported here including
        # classes such as qiskit.quantum_info.SparsePauliOp. The inner length-1
        # lists cause this array of observables to have shape (3, 1), rather
        # than shape (3,) if they were omitted.
        observables = [
            [SparsePauliOp(["XX", "IY"], [0.5, 0.5])],
            [Pauli("XX")],
            [Pauli("IY")]
        ]

        # Instantiate a new statevector simulation based estimator object.
        estimator = StatevectorEstimator()

        # Estimate the expectation value for all 300 combinations of
        # observables and parameter values, where the pub result will have
        # shape (3, 100). This shape is due to our array of parameter
        # bindings having shape (100,), combined with our array of observables
        # having shape (3, 1)
        pub = (circuit, observables, params)
        job = estimator.run([pub])

        # Extract the result for the 0th pub (this example only has one pub).
        result = job.result()[0]

        # Error-bar information is also available, but the error is 0
        # for this StatevectorEstimator.
        result.data.stds

        # Pull out the array-based expectation value estimate data from the
        # result and plot a trace for each observable.
        for idx, pauli in enumerate(observables):
            plt.plot(result.data.evs[idx], label=pauli)
        plt.legend()
    """

    def __init__(
        self, *, default_precision: float = 0.0, seed: np.random.Generator | int | None = None
    ):
        """
        Args:
            default_precision: The default precision for the estimator if not specified during run.
            seed: The seed or Generator object for random number generation.
                If None, a random seeded default RNG will be used.
        """
        self._default_precision = default_precision
        self._seed = seed

    @property
    def default_precision(self) -> float:
        """Return the default precision"""
        return self._default_precision

    @property
    def seed(self) -> np.random.Generator | int | None:
        """Return the seed or Generator object for random number generation."""
        return self._seed

    def run(
        self, pubs: Iterable[EstimatorPubLike], *, precision: float | None = None
    ) -> PrimitiveJob[PrimitiveResult[PubResult]]:
        if precision is None:
            precision = self._default_precision
        coerced_pubs = [EstimatorPub.coerce(pub, precision) for pub in pubs]

        job = PrimitiveJob(self._run, coerced_pubs)
        job._submit()
        return job

    def _run(self, pubs: list[EstimatorPub]) -> PrimitiveResult[PubResult]:
        return PrimitiveResult([self._run_pub(pub) for pub in pubs], metadata={"version": 2})

    def _run_pub(self, pub: EstimatorPub) -> PubResult:
        rng = np.random.default_rng(self._seed)
        circuit = pub.circuit
        observables = pub.observables
        parameter_values = pub.parameter_values
        precision = pub.precision
        bound_circuits = parameter_values.bind_all(circuit)
        bc_circuits, bc_obs = np.broadcast_arrays(bound_circuits, observables)
        evs = np.zeros_like(bc_circuits, dtype=np.float64)
        stds = np.zeros_like(bc_circuits, dtype=np.float64)
        for index in np.ndindex(*bc_circuits.shape):
            bound_circuit = bc_circuits[index]
            observable = bc_obs[index]
            final_state = _statevector_from_circuit(bound_circuit, rng)
            paulis, coeffs = zip(*observable.items())
            obs = SparsePauliOp(paulis, coeffs)  # TODO: support non Pauli operators
            expectation_value = np.real_if_close(final_state.expectation_value(obs))
            if precision != 0:
                if not np.isreal(expectation_value):
                    raise ValueError("Given operator is not Hermitian and noise cannot be added.")
                expectation_value = rng.normal(expectation_value, precision)
            evs[index] = expectation_value

        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data, metadata={"target_precision": precision, "circuit_metadata": pub.circuit.metadata}
        )
