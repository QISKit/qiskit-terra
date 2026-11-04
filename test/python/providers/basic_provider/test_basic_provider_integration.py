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

"""BasicProvider provider integration tests."""

import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.result import Result
from qiskit.providers.basic_provider import BasicProviderError, BasicSimulator
from test import QiskitTestCase


class TestBasicProviderIntegration(QiskitTestCase):
    """Qiskit BasicProvider simulator integration tests."""

    def setUp(self):
        super().setUp()
        qr = QuantumRegister(1)
        cr = ClassicalRegister(1)
        self._qc1 = QuantumCircuit(qr, cr, name="qc1")
        self._qc2 = QuantumCircuit(qr, cr, name="qc2")
        self._qc1.measure(qr[0], cr[0])
        self.backend = BasicSimulator()
        self._result1 = self.backend.run(transpile(self._qc1)).result()

    def test_builtin_simulator_result_fields(self):
        """Test components of a result from a local simulator."""

        self.assertEqual("basic_simulator", self._result1.backend_name)
        self.assertIsInstance(self._result1.job_id, str)
        self.assertEqual(self._result1.status, "COMPLETED")
        self.assertEqual(self._result1.results[0].status, "DONE")

    def test_basicprovider_execute(self):
        """Test Compiler and run."""
        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        job = self.backend.run(qc)
        result = job.result()
        self.assertIsInstance(result, Result)

    def test_basicprovider_execute_two(self):
        """Test Compiler and run."""
        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, clbit_reg, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        job = self.backend.run([qc, qc_extra])
        result = job.result()
        self.assertIsInstance(result, Result)

    def test_basicprovider_num_qubits(self):
        """Test BasicProviderError is raised if num_qubits too large to simulate."""
        qc = QuantumCircuit(50, 1)
        qc.x(0)
        qc.measure(0, 0)
        with self.assertRaises(BasicProviderError):
            self.backend.run(qc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
