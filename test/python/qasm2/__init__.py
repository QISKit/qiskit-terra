# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Common utility function."""

from collections.abc import Iterable
from qiskit.circuit import QuantumCircuit, Parameter

import qiskit.qasm2


def gate_builder(name: str, parameters: Iterable[Parameter], definition: QuantumCircuit):
    """Get a builder for a custom gate.

    Ideally we would just use an eagerly defined `Gate` instance here, but limitations in how
    `QuantumCircuit.__eq__` and `Instruction.__eq__` work mean that we have to ensure we're using
    the same class as the parser for equality checks to work."""

    # Ideally we wouldn't have this at all, but hiding it away in one function is likely the safest
    # and easiest to update if the Python component of the library changes.

    def definer(*arguments):
        # We can supply empty lists for the gates and the bytecode, because we're going to override
        # the definition manually ourselves.
        gate = qiskit.qasm2.parse._DefinedGate(name, definition.num_qubits, arguments, (), ())
        gate._definition = definition.assign_parameters(dict(zip(parameters, arguments)))
        return gate

    return definer
