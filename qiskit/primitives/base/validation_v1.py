# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Primitive V1 validation methods.

Note that these are not intended to be part of the public API of base primitives
but are here for backward compatibility with BaseSamplerV1 and BaseEstimatorV1 classes.
"""

from __future__ import annotations

from collections.abc import Sequence
import numpy as np

from qiskit.circuit import QuantumCircuit, ControlFlowOp, Measure
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli


def _validate_estimator_args(
    circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    observables: Sequence[BaseOperator | str] | BaseOperator | str,
    parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None = None,
) -> tuple[tuple[QuantumCircuit], tuple[BaseOperator], tuple[tuple[float]]]:
    """Validate run arguments for BaseEstimatorV1.

    Args:
        circuits: one or more circuit objects.
        observables: one or more observable objects.
        parameter_values: concrete parameters to be bound.

    Returns:
        The formatted arguments ``(circuits, observables, parameter_values)``.

    Raises:
        TypeError: If input arguments are invalid types.
        ValueError: if input arguments are invalid values.
    """
    # Singular validation
    circuits = _validate_circuits(circuits)
    observables = _validate_observables(observables)
    parameter_values = _validate_parameter_values(
        parameter_values,
        default=[()] * len(circuits),
    )

    # Cross-validation
    _cross_validate_circuits_parameter_values(circuits, parameter_values)
    _cross_validate_circuits_observables(circuits, observables)

    return circuits, observables, parameter_values


def _validate_sampler_args(
    circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None = None,
) -> tuple[tuple[QuantumCircuit], tuple[BaseOperator], tuple[tuple[float]]]:
    """Validate run arguments for BaseSamplerV1.

    Args:
        circuits: one or more circuit objects.
        parameter_values: concrete parameters to be bound.

    Returns:
        The formatted arguments ``(circuits, parameter_values)``.

    Raises:
        TypeError: If input arguments are invalid types.
        ValueError: if input arguments are invalid values.
    """
    # Singular validation
    circuits = _validate_circuits(circuits, requires_measure=True)
    parameter_values = _validate_parameter_values(
        parameter_values,
        default=[()] * len(circuits),
    )

    # Cross-validation
    _cross_validate_circuits_parameter_values(circuits, parameter_values)

    return circuits, parameter_values


def _init_observable(observable: BaseOperator | str) -> SparsePauliOp:
    """Initialize observable by converting the input to a :class:`~qiskit.quantum_info.SparsePauliOp`.

    Args:
        observable: The observable.

    Returns:
        The observable as :class:`~qiskit.quantum_info.SparsePauliOp`.

    Raises:
        QiskitError: when observable type cannot be converted to SparsePauliOp.
    """

    if isinstance(observable, SparsePauliOp):
        return observable
    elif isinstance(observable, BaseOperator) and not isinstance(observable, BasePauli):
        raise QiskitError(f"observable type not supported: {type(observable)}")
    else:
        if isinstance(observable, PauliList):
            raise QiskitError(f"observable type not supported: {type(observable)}")
        return SparsePauliOp(observable)


def _validate_circuits(
    circuits: Sequence[QuantumCircuit] | QuantumCircuit,
    requires_measure: bool = False,
) -> tuple[QuantumCircuit, ...]:
    if isinstance(circuits, QuantumCircuit):
        circuits = (circuits,)
    elif not isinstance(circuits, Sequence) or not all(
        isinstance(cir, QuantumCircuit) for cir in circuits
    ):
        raise TypeError("Invalid circuits, expected Sequence[QuantumCircuit].")
    elif not isinstance(circuits, tuple):
        circuits = tuple(circuits)
    if len(circuits) == 0:
        raise ValueError("No circuits were provided.")

    if requires_measure:
        for i, circuit in enumerate(circuits):
            if circuit.num_clbits == 0:
                raise ValueError(
                    f"The {i}-th circuit does not have any classical bit. "
                    "Sampler requires classical bits, plus measurements "
                    "on the desired qubits."
                )
            if not _has_measure(circuit):
                raise ValueError(
                    f"The {i}-th circuit does not have Measure instruction. "
                    "Without measurements, the circuit cannot be sampled from."
                )
    return circuits


def _validate_parameter_values(
    parameter_values: Sequence[Sequence[float]] | Sequence[float] | float | None,
    default: Sequence[Sequence[float]] | Sequence[float] | None = None,
) -> tuple[tuple[float, ...], ...]:
    # Allow optional (if default)
    if parameter_values is None:
        if default is None:
            raise ValueError("No default `parameter_values`, optional input disallowed.")
        parameter_values = default

    # Support numpy ndarray
    if isinstance(parameter_values, np.ndarray):
        parameter_values = parameter_values.tolist()
    elif isinstance(parameter_values, Sequence):
        parameter_values = tuple(
            vector.tolist() if isinstance(vector, np.ndarray) else vector
            for vector in parameter_values
        )

    # Allow single value
    if _isreal(parameter_values):
        parameter_values = ((parameter_values,),)
    elif isinstance(parameter_values, Sequence) and not any(
        isinstance(vector, Sequence) for vector in parameter_values
    ):
        parameter_values = (parameter_values,)

    # Validation
    if (
        not isinstance(parameter_values, Sequence)
        or not all(isinstance(vector, Sequence) for vector in parameter_values)
        or not all(all(_isreal(value) for value in vector) for vector in parameter_values)
    ):
        raise TypeError("Invalid parameter values, expected Sequence[Sequence[float]].")

    return tuple(tuple(float(value) for value in vector) for vector in parameter_values)


def _validate_observables(
    observables: Sequence[BaseOperator | str] | BaseOperator | str,
) -> tuple[SparsePauliOp, ...]:
    if isinstance(observables, str) or not isinstance(observables, Sequence):
        observables = (observables,)
    if len(observables) == 0:
        raise ValueError("No observables were provided.")
    return tuple(_init_observable(obs) for obs in observables)


def _cross_validate_circuits_parameter_values(
    circuits: tuple[QuantumCircuit, ...], parameter_values: tuple[tuple[float, ...], ...]
) -> None:
    if len(circuits) != len(parameter_values):
        raise ValueError(
            f"The number of circuits ({len(circuits)}) does not match "
            f"the number of parameter value sets ({len(parameter_values)})."
        )
    for i, (circuit, vector) in enumerate(zip(circuits, parameter_values)):
        if len(vector) != circuit.num_parameters:
            raise ValueError(
                f"The number of values ({len(vector)}) does not match "
                f"the number of parameters ({circuit.num_parameters}) for the {i}-th circuit."
            )


def _cross_validate_circuits_observables(
    circuits: tuple[QuantumCircuit, ...], observables: tuple[BaseOperator, ...]
) -> None:
    if len(circuits) != len(observables):
        raise ValueError(
            f"The number of circuits ({len(circuits)}) does not match "
            f"the number of observables ({len(observables)})."
        )
    for i, (circuit, observable) in enumerate(zip(circuits, observables)):
        if circuit.num_qubits != observable.num_qubits:
            raise ValueError(
                f"The number of qubits of the {i}-th circuit ({circuit.num_qubits}) does "
                f"not match the number of qubits of the {i}-th observable "
                f"({observable.num_qubits})."
            )


def _isint(obj: Sequence[Sequence[float]] | Sequence[float] | float) -> bool:
    """Check if object is int."""
    int_types = (int, np.integer)
    return isinstance(obj, int_types) and not isinstance(obj, bool)


def _isreal(obj: Sequence[Sequence[float]] | Sequence[float] | float) -> bool:
    """Check if object is a real number: int or float except ``±Inf`` and ``NaN``."""
    float_types = (float, np.floating)
    return _isint(obj) or (isinstance(obj, float_types) and float("-Inf") < obj < float("Inf"))


def _has_measure(circuit: QuantumCircuit) -> bool:
    for instruction in reversed(circuit):
        if isinstance(instruction.operation, Measure):
            return True
        elif isinstance(instruction.operation, ControlFlowOp):
            for block in instruction.operation.blocks:
                if _has_measure(block):
                    return True
    return False
