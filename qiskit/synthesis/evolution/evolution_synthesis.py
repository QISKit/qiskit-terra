# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Evolution synthesis."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EvolutionSynthesis(ABC):
    """Interface for evolution synthesis algorithms."""

    @abstractmethod
    def synthesize(self, evolution):
        """Synthesize a ``qiskit.circuit.library.PauliEvolutionGate``.

        Args:
            evolution (PauliEvolutionGate): The evolution gate to synthesize.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        raise NotImplementedError

    @property
    def settings(self) -> dict[str, Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: The interface does not implement this method.
        """
        raise NotImplementedError(
            "The settings property is not implemented for the base interface."
        )
