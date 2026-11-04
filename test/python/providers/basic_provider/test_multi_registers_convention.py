# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test executing multiple-register circuits on BasicProvider."""

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.providers.basic_provider import BasicProvider
from test import QiskitTestCase


class TestCircuitMultiRegs(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_multi(self):
        """Test circuit multi regs declared at start."""
        qreg0 = QuantumRegister(2, "q0")
        creg0 = ClassicalRegister(2, "c0")
        qreg1 = QuantumRegister(2, "q1")
        creg1 = ClassicalRegister(2, "c1")
        circ = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        circ.x(qreg0[1])
        circ.x(qreg1[0])

        meas = QuantumCircuit(qreg0, qreg1, creg0, creg1)
        meas.measure(qreg0, creg0)
        meas.measure(qreg1, creg1)

        qc = circ.compose(meas)

        backend_sim = BasicProvider().get_backend("basic_simulator")

        result = backend_sim.run(qc).result()
        counts = result.get_counts(qc)
        target = {"01 10": 1024}

        self.assertEqual(counts, target)
