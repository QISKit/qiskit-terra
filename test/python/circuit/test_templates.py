# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit templates."""

import unittest
from inspect import getmembers, isfunction
from ddt import ddt

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info.operators import Operator
import qiskit.circuit.library.templates as templib
from test import combine
from test import QiskitTestCase


@ddt
class TestTemplates(QiskitTestCase):
    """Tests for the circuit templates."""

    circuits = [o[1]() for o in getmembers(templib) if isfunction(o[1])]

    for circuit in circuits:
        if isinstance(circuit, QuantumCircuit):
            circuit.assign_parameters(dict.fromkeys(circuit.parameters, 0.2), inplace=True)

    @combine(template_circuit=circuits)
    def test_template(self, template_circuit):
        """Test that all templates are exactly equal to the identity (including global phase).

        Uses strict Operator equality rather than equiv() so that a template whose
        gate content carries a residual global phase (i.e. global_phase is not set
        to compensate) is caught as a failure rather than silently accepted.
        """
        target = Operator(template_circuit)
        value = Operator(np.eye(2**template_circuit.num_qubits))
        self.assertEqual(target, value)


if __name__ == "__main__":
    unittest.main()
