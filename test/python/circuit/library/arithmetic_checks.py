# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Shared checks for arithmetic circuit tests."""

import numpy as np

from qiskit.quantum_info import Statevector


def assert_maps_basis_states(test, circuit, num_input_qubits, func, seed=12345):
    """Assert that ``circuit`` maps ``|i>`` to ``|func(i)>`` for every input ``i``.

    The inputs live on the ``num_input_qubits`` lowest qubits, all other qubits start in ``|0>``.
    A single random superposition of all inputs is evolved and the full output state is compared
    up to a global phase.
    """
    rng = np.random.default_rng(seed)
    dim_in = 2**num_input_qubits
    amplitudes = rng.normal(size=dim_in) + 1j * rng.normal(size=dim_in)
    amplitudes /= np.linalg.norm(amplitudes)
    initial = np.zeros(2**circuit.num_qubits, dtype=complex)
    initial[:dim_in] = amplitudes
    expected = np.zeros_like(initial)
    for index, amplitude in enumerate(amplitudes):
        expected[func(index)] += amplitude
    actual = Statevector(initial).evolve(circuit)
    test.assertTrue(actual.equiv(Statevector(expected)))
