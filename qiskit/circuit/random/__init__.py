# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
==============================================
Random Circuits (:mod:`qiskit.circuit.random`)
==============================================

.. currentmodule:: qiskit.circuit.random

Overview
========

The :mod:`qiskit.circuit.random` module offers functions that can be used for generating
arbitrary circuits with gates randomly selected from a given set of gates.

These circuits can be used for benchmarking existing quantum hardware and estimating
the performance of quantum circuit transpilers and software infrastructure.
The functions below can generate bespoke quantum circuits respecting various properties
such as number of qubits, depth of the circuit, coupling map, gate set, etc.

Generating arbitrary circuits
------------------------------------

.. autofunction:: random_circuit


Generating arbitrary circuits respecting qubit-coupling
--------------------------------------------------------------

.. autofunction:: random_circuit_from_graph


Generating arbitrary circuits with clifford gates
--------------------------------------------------------

.. autofunction:: random_clifford_circuit

"""

from .utils import random_circuit, random_clifford_circuit, random_circuit_from_graph

__all__ = [
    "random_circuit",
    "random_circuit_from_graph",
    "random_clifford_circuit",
]
