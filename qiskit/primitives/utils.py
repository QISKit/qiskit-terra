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
"""
Utility functions for primitives
"""
from __future__ import annotations

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.quantum_info import Statevector


def _statevector_from_circuit(
    circuit: QuantumCircuit, rng: np.random.Generator | None
) -> Statevector:
    """Generate a statevector from a circuit. Used in StatevectorEstimator class.

    If the input circuit includes any resets for some subsystem,
    :meth:`.Statevector.reset` behaves in a stochastic way in :meth:`.Statevector.evolve`.
    This function sets a random number generator to be reproducible.

    See :meth:`.Statevector.reset` for details.

    Args:
        circuit: The quantum circuit.
        rng: The random number generator or None.
    """
    sv = Statevector.from_int(0, 2**circuit.num_qubits)
    sv.seed(rng)
    return sv.evolve(bound_circuit_to_instruction(circuit))


def bound_circuit_to_instruction(circuit: QuantumCircuit) -> Instruction:
    """Build an :class:`~qiskit.circuit.Instruction` object from
    a :class:`~qiskit.circuit.QuantumCircuit`

    This is a specialized version of :func:`~qiskit.converters.circuit_to_instruction`
    to avoid deep copy. This requires a quantum circuit whose parameters are all bound.
    Because this does not take a copy of the input circuit, this assumes that the input
    circuit won't be modified.

    If https://github.com/Qiskit/qiskit-terra/issues/7983 is resolved,
    we can remove this function.

    Args:
        circuit(QuantumCircuit): Input quantum circuit

    Returns:
        An :class:`~qiskit.circuit.Instruction` object
    """
    if len(circuit.qregs) > 1:
        return circuit.to_instruction()

    # len(circuit.qregs) == 1 -> No need to flatten qregs
    inst = Instruction(
        name=circuit.name,
        num_qubits=circuit.num_qubits,
        num_clbits=circuit.num_clbits,
        params=[],
    )
    inst.definition = circuit
    return inst
