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
Sampler Pub class
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral


from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

from .bindings_array import BindingsArray, BindingsArrayLike
from .shape import ShapedMixin

# Public API classes
__all__ = ["SamplerPubLike"]


class SamplerPub(ShapedMixin):
    """Pub (Primitive Unified Bloc) for a Sampler.

    Pub is composed of tuple (circuit, parameter_values, shots).

    If shots are provided this number of shots will be run with the sampler,
    if ``shots=None`` the number of run shots is determined by the sampler.
    """

    def __init__(
        self,
        circuit: QuantumCircuit,
        parameter_values: BindingsArray | None = None,
        shots: int | None = None,
        validate: bool = True,
    ):
        """Initialize a sampler pub.

        Args:
            circuit: A quantum circuit.
            parameter_values: A bindings array.
            shots: A specific number of shots to run with. This value takes
                precedence over any value owed by or supplied to a sampler.
            validate: If ``True``, the input data is validated during initialization.
        """
        super().__init__()
        self._circuit = circuit
        self._parameter_values = parameter_values or BindingsArray()
        self._shots = shots
        self._shape = self._parameter_values.shape
        if validate:
            self.validate()

    @property
    def circuit(self) -> QuantumCircuit:
        """A quantum circuit."""
        return self._circuit

    @property
    def parameter_values(self) -> BindingsArray:
        """A bindings array."""
        return self._parameter_values

    @property
    def shots(self) -> int | None:
        """A specific number of shots to run with (optional).

        This value takes precedence over any value owed by or supplied to a sampler.
        """
        return self._shots

    @classmethod
    def coerce(cls, pub: SamplerPubLike, shots: int | None = None) -> SamplerPub:
        """Coerce a :class:`~.SamplerPubLike` object into a :class:`~.SamplerPub` instance.

        Args:
            pub: An object to coerce.
            shots: An optional default number of shots to use if not
                   already specified by the pub-like object.

        Returns:
            A coerced sampler pub.
        """
        # Validate shots kwarg if provided
        if shots is not None:
            if not isinstance(shots, Integral) or isinstance(shots, bool):
                raise TypeError("shots must be an integer")
            if shots <= 0:
                raise ValueError("shots must be positive")

        if isinstance(pub, SamplerPub):
            if pub.shots is None and shots is not None:
                return cls(
                    circuit=pub.circuit,
                    parameter_values=pub.parameter_values,
                    shots=shots,
                    validate=False,  # Assume Pub is already validated
                )
            return pub

        if isinstance(pub, QuantumCircuit):
            return cls(circuit=pub, shots=shots, validate=True)

        if isinstance(pub, CircuitInstruction):
            raise ValueError(
                f"An invalid Sampler pub-like was given ({type(pub)}). "
                "If you want to run a single circuit, "
                "you need to wrap it with `[]` like `sampler.run([circuit])` "
                "instead of `sampler.run(circuit)`."
            )

        if len(pub) not in [1, 2, 3]:
            raise ValueError(
                f"The length of pub must be 1, 2 or 3, but length {len(pub)} is given."
            )
        circuit = pub[0]

        if len(pub) > 1 and pub[1] is not None:
            values = pub[1]
            if not isinstance(values, (BindingsArray, Mapping)):
                values = {tuple(circuit.parameters): values}
            parameter_values = BindingsArray.coerce(values)
        else:
            parameter_values = None

        if len(pub) > 2 and pub[2] is not None:
            shots = pub[2]
        return cls(circuit=circuit, parameter_values=parameter_values, shots=shots, validate=True)

    def validate(self):
        """Validate the pub."""
        if not isinstance(self.circuit, QuantumCircuit):
            raise TypeError("circuit must be QuantumCircuit.")

        self.parameter_values.validate()

        if self.shots is not None:
            if not isinstance(self.shots, Integral) or isinstance(self.shots, bool):
                raise TypeError("shots must be an integer")
            if self.shots <= 0:
                raise ValueError("shots must be positive")

        # Cross validate circuits and parameter values
        num_parameters = self.parameter_values.num_parameters
        if num_parameters != self.circuit.num_parameters:
            message = (
                f"The number of values ({num_parameters}) does not match "
                f"the number of parameters ({self.circuit.num_parameters}) for the circuit."
            )
            if num_parameters == 0:
                message += (
                    " Note that if you want to run a single pub, you need to wrap it with `[]` like "
                    "`sampler.run([(circuit, param_values)])` instead of "
                    "`sampler.run((circuit, param_values))`."
                )
            raise ValueError(message)


SamplerPubLike = (
    SamplerPub
    | QuantumCircuit
    | tuple[QuantumCircuit]
    | tuple[QuantumCircuit, BindingsArrayLike]
    | tuple[QuantumCircuit, BindingsArrayLike, Integral | None]
)
"""A Pub (Primitive Unified Bloc) for a Sampler.

A fully specified sample Pub is a tuple ``(circuit, parameter_values, shots)``.

If shots are provided this number of shots will be run with the sampler,
if ``shots=None`` the number of run shots is determined by the sampler.

.. note::

    A Sampler Pub can also be initialized in the following formats which
    will be converted to the full Pub tuple:

    * ``circuit``
    * ``(circuit,)``
    * ``(circuit, parameter_values)``
"""
