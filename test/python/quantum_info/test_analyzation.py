# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit.quantum_info.analysis"""

import unittest

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.quantum_info.analysis.average import average_data
from qiskit.quantum_info.analysis.make_observable import make_dict_observable
from qiskit.quantum_info.analysis import hellinger_fidelity
from test import QiskitTestCase


class TestAnalyzation(QiskitTestCase):
    """Test qiskit.Result API"""

    def test_average_data_dict_observable(self):
        """Test average_data for dictionary observable input"""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr, name="qc")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        shots = 10000
        backend = BasicSimulator()
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts(qc)
        observable = {"00": 1, "11": 1, "01": -1, "10": -1}
        mean_zz = average_data(counts=counts, observable=observable)
        observable = {"00": 1, "11": -1, "01": 1, "10": -1}
        mean_zi = average_data(counts, observable)
        observable = {"00": 1, "11": -1, "01": -1, "10": 1}
        mean_iz = average_data(counts, observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)

    def test_average_data_list_observable(self):
        """Test average_data for list observable input."""
        qr = QuantumRegister(3)
        cr = ClassicalRegister(3)
        qc = QuantumCircuit(qr, cr, name="qc")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[2])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        qc.measure(qr[2], cr[2])
        shots = 10000
        backend = BasicSimulator()
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts(qc)
        observable = [1, -1, -1, 1, -1, 1, 1, -1]
        mean_zzz = average_data(counts=counts, observable=observable)
        observable = [1, 1, 1, 1, -1, -1, -1, -1]
        mean_zii = average_data(counts, observable)
        observable = [1, 1, -1, -1, 1, 1, -1, -1]
        mean_izi = average_data(counts, observable)
        observable = [1, 1, -1, -1, -1, -1, 1, 1]
        mean_zzi = average_data(counts, observable)
        self.assertAlmostEqual(mean_zzz, 0, places=1)
        self.assertAlmostEqual(mean_zii, 0, places=1)
        self.assertAlmostEqual(mean_izi, 0, places=1)
        self.assertAlmostEqual(mean_zzi, 1, places=1)

    def test_average_data_matrix_observable(self):
        """Test average_data for matrix observable input."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr, name="qc")
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.measure(qr[0], cr[0])
        qc.measure(qr[1], cr[1])
        shots = 10000
        backend = BasicSimulator()
        result = backend.run(qc, shots=shots).result()
        counts = result.get_counts(qc)
        observable = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        mean_zz = average_data(counts=counts, observable=observable)
        observable = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
        mean_zi = average_data(counts, observable)
        observable = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
        mean_iz = average_data(counts, observable)
        self.assertAlmostEqual(mean_zz, 1, places=1)
        self.assertAlmostEqual(mean_zi, 0, places=1)
        self.assertAlmostEqual(mean_iz, 0, places=1)

    def test_make_dict_observable(self):
        """Test make_dict_observable."""
        list_in = [1, 1, -1, -1]
        list_out = make_dict_observable(list_in)
        list_expected = {"00": 1, "01": 1, "10": -1, "11": -1}
        matrix_in = [[4, 0, 0, 0], [0, -3, 0, 0], [0, 0, 2, 0], [0, 0, 0, -1]]
        matrix_out = make_dict_observable(matrix_in)
        matrix_expected = {"00": 4, "01": -3, "10": 2, "11": -1}
        long_list_in = [1, 1, -1, -1, -1, -1, 1, 1]
        long_list_out = make_dict_observable(long_list_in)
        long_list_expected = {
            "000": 1,
            "001": 1,
            "010": -1,
            "011": -1,
            "100": -1,
            "101": -1,
            "110": 1,
            "111": 1,
        }
        self.assertEqual(list_out, list_expected)
        self.assertEqual(matrix_out, matrix_expected)
        self.assertEqual(long_list_out, long_list_expected)

    def test_hellinger_fidelity_same(self):
        """Test hellinger fidelity is one for same dist."""
        qc = QuantumCircuit(5, 5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(1, 0)
        qc.measure(range(5), range(5))

        sim = BasicSimulator()

        res = sim.run(qc).result()

        ans = hellinger_fidelity(res.get_counts(), res.get_counts())

        self.assertEqual(ans, 1.0)

    def test_hellinger_fidelity_no_overlap(self):
        """Test hellinger fidelity is zero for no overlap."""

        #                ┌───┐     ┌─┐
        # q_0: ──────────┤ X ├─────┤M├────────────
        #           ┌───┐└─┬─┘     └╥┘┌─┐
        # q_1: ─────┤ X ├──■────────╫─┤M├─────────
        #      ┌───┐└─┬─┘           ║ └╥┘┌─┐
        # q_2: ┤ H ├──■────■────────╫──╫─┤M├──────
        #      └───┘     ┌─┴─┐      ║  ║ └╥┘┌─┐
        # q_3: ──────────┤ X ├──■───╫──╫──╫─┤M├───
        #                └───┘┌─┴─┐ ║  ║  ║ └╥┘┌─┐
        # q_4: ───────────────┤ X ├─╫──╫──╫──╫─┤M├
        #                     └───┘ ║  ║  ║  ║ └╥┘
        # c: 5/═════════════════════╩══╩══╩══╩══╩═
        #                           0  1  2  3  4
        qc = QuantumCircuit(5, 5)
        qc.h(2)
        qc.cx(2, 1)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(1, 0)
        qc.measure(range(5), range(5))

        #                ┌───┐     ┌─┐
        # q_0: ──────────┤ X ├─────┤M├─────────
        #           ┌───┐└─┬─┘     └╥┘┌─┐
        # q_1: ─────┤ X ├──■────────╫─┤M├──────
        #      ┌───┐└─┬─┘┌───┐      ║ └╥┘┌─┐
        # q_2: ┤ H ├──■──┤ Y ├──■───╫──╫─┤M├───
        #      └───┘     └───┘┌─┴─┐ ║  ║ └╥┘┌─┐
        # q_3: ───────────────┤ X ├─╫──╫──╫─┤M├
        #       ┌─┐           └───┘ ║  ║  ║ └╥┘
        # q_4: ─┤M├─────────────────╫──╫──╫──╫─
        #       └╥┘                 ║  ║  ║  ║
        # c: 5/══╩══════════════════╩══╩══╩══╩═
        #        4                  0  1  2  3
        qc2 = QuantumCircuit(5, 5)
        qc2.h(2)
        qc2.cx(2, 1)
        qc2.y(2)
        qc2.cx(2, 3)
        qc2.cx(1, 0)
        qc2.measure(range(5), range(5))

        sim = BasicSimulator()

        res1 = sim.run(qc).result()
        res2 = sim.run(transpile(qc2, sim)).result()

        ans = hellinger_fidelity(res1.get_counts(), res2.get_counts())

        self.assertEqual(ans, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
