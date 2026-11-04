# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Estimator Pub class
"""

from __future__ import annotations

from numbers import Real
from collections.abc import Mapping


import numpy as np

from qiskit import QuantumCircuit

from .bindings_array import BindingsArray, BindingsArrayLike
from .observables_array import ObservablesArray, ObservablesArrayLike
from .shape import ShapedMixin

# Public API classes
__all__ = ["EstimatorPubLike"]


class EstimatorPub(ShapedMixin):
    """Primitive Unified Bloc for any Estimator primitive.

    An estimator pub is essentially a tuple ``(circuit, observables, parameter_values, precision)``.

    If precision is provided this should be used for the target precision of an
    estimator, if ``precision=None`` the estimator will determine the target precision.
    """

    __slots__ = ("_circuit", "_observables", "_parameter_values", "_precision", "_shape")

    def __init__(
        self,
        circuit: QuantumCircuit,
        observables: ObservablesArray,
        parameter_values: BindingsArray | None = None,
        precision: float | None = None,
        validate: bool = True,
    ):
        """Initialize an estimator pub.

        Args:
            circuit: A quantum circuit.
            observables: An observables array.
            parameter_values: A bindings array, if the circuit is parametric.
            precision: An optional target precision for expectation value estimates.
            validate: Whether to validate arguments during initialization.

        Raises:
            ValueError: If the ``observables`` and ``parameter_values`` are not broadcastable, that
                is, if their shapes, when right-aligned, do not agree or equal 1.
        """
        super().__init__()
        self._circuit = circuit
        self._observables = observables
        self._parameter_values = parameter_values or BindingsArray()
        self._precision = precision

        # for ShapedMixin
        try:
            # _shape has to be defined to properly be Shaped, so we can't put it in validation
            self._shape = np.broadcast_shapes(self.observables.shape, self.parameter_values.shape)
        except ValueError as ex:
            raise ValueError(
                f"The observables shape {self.observables.shape} and the "
                f"parameter values shape {self.parameter_values.shape} are not broadcastable."
            ) from ex

        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit."""
        return self._circuit

    @property
    def observables(self) -> ObservablesArray:
        """An observables array."""
        return self._observables

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array."""
        return self._parameter_values

    @property
    def precision(self) -> float | None:
        """The target precision for expectation value estimates (optional)."""
        return self._precision

    @classmethod
    def coerce(cls, pub: EstimatorPubLike, precision: float | None = None) -> EstimatorPub:
        """Coerce :class:`~.EstimatorPubLike` into :class:`~.EstimatorPub`.

        Args:
            pub: A compatible object for coercion.
            precision: an optional default precision to use if not
                       already specified by the pub-like object.

        Returns:
            An estimator pub.
        """
        # Validate precision kwarg if provided
        if precision is not None:
            if not isinstance(precision, Real):
                raise TypeError(f"precision must be a real number, not {type(precision)}.")
            if precision < 0:
                raise ValueError("precision must be non-negative")
        if isinstance(pub, EstimatorPub):
            if pub.precision is None and precision is not None:
                return cls(
                    circuit=pub.circuit,
                    observables=pub.observables,
                    parameter_values=pub.parameter_values,
                    precision=precision,
                    validate=False,  # Assume Pub is already validated
                )
            return pub

        if isinstance(pub, QuantumCircuit):
            raise ValueError(
                f"An invalid Estimator pub-like was given ({type(pub)}). "
                "If you want to run a single pub, you need to wrap it with `[]` like "
                "`estimator.run([(circuit, observables, param_values)])` "
                "instead of `estimator.run((circuit, observables, param_values))`."
            )

        if len(pub) not in [2, 3, 4]:
            raise ValueError(
                f"The length of pub must be 2, 3 or 4, but length {len(pub)} is given."
            )
        circuit = pub[0]
        observables = ObservablesArray.coerce(pub[1])

        if len(pub) > 2 and pub[2] is not None:
            values = pub[2]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None

        if len(pub) > 3 and pub[3] is not None:
            precision = pub[3]

        return cls(
            circuit=circuit,
            observables=observables,
            parameter_values=parameter_values,
            precision=precision,
            validate=True,
        )

    def validate(self):
        """Validate the pub."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.observables.validate()
        self.parameter_values.validate()

        if self.precision is not None:
            if not isinstance(self.precision, Real):
                raise TypeError(f"precision must be a real number, not {type(self.precision)}.")
            if self.precision < 0:
                raise ValueError("precision must be non-negative.")

        # Cross validate circuits and observables
        for i, observable in np.ndenumerate(self.observables):
            num_qubits = len(next(iter(observable)))
            if self.circuit.num_qubits != num_qubits:
                raise ValueError(
                    f"The number of qubits of the circuit ({self.circuit.num_qubits}) does "
                    f"not match the number of qubits of the {i}-th observable ({num_qubits})."
                )

        # Cross validate circuits and parameter_values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            raise ValueError(
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )


EstimatorPubLike = (
    EstimatorPub
    | tuple[QuantumCircuit, ObservablesArrayLike]
    | tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike]
    | tuple[QuantumCircuit, ObservablesArrayLike, BindingsArrayLike, Real]
)
"""A Pub (Primitive Unified Bloc) for an Estimator primitive.

A fully specified estimator pub is a tuple ``(circuit, observables, parameter_values, precision)``.

If precision is provided this should be used for the target precision of an
estimator, if ``precision=None`` the estimator will determine the target precision.

.. note::

    An Estimator Pub can also be initialized in the following formats which
    will be converted to the full Pub tuple:

    * ``(circuit, observables)``
    * ``(circuit, observables, parameter_values)``
"""
