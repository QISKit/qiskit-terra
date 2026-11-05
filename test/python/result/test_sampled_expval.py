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

"""Tests for qiskit.quantum_info.analysis"""

import unittest

from qiskit.result import Counts, QuasiDistribution, ProbDistribution, sampled_expectation_value
from qiskit.quantum_info import Pauli, SparsePauliOp, SparseObservable
from test import QiskitTestCase

PROBS = {
    "1000": 0.0022,
    "1001": 0.0045,
    "1110": 0.0081,
    "0001": 0.0036,
    "0010": 0.0319,
    "0101": 0.001,
    "1100": 0.0008,
    "1010": 0.0009,
    "1111": 0.3951,
    "0011": 0.0007,
    "0111": 0.01,
    "0000": 0.4666,
    "1101": 0.0355,
    "1011": 0.0211,
    "0110": 0.0081,
    "0100": 0.0099,
}


class TestSampledExpval(QiskitTestCase):
    """Test sampled expectation values"""

    def test_simple(self):
        """Test that basic exp values work"""

        dist2 = {"00": 0.5, "11": 0.5}
        dist3 = {"000": 0.5, "111": 0.5}
        # ZZ even GHZ is 1.0
        self.assertAlmostEqual(sampled_expectation_value(dist2, "ZZ"), 1.0)
        # ZZ odd GHZ is 0.0
        self.assertAlmostEqual(sampled_expectation_value(dist3, "ZZZ"), 0.0)
        # All id ops goes to 1.0
        self.assertAlmostEqual(sampled_expectation_value(dist3, "III"), 1.0)
        # flipping one to I makes even GHZ 0.0
        self.assertAlmostEqual(sampled_expectation_value(dist2, "IZ"), 0.0)
        self.assertAlmostEqual(sampled_expectation_value(dist2, "ZI"), 0.0)
        # Generic Z on PROBS
        self.assertAlmostEqual(sampled_expectation_value(PROBS, "ZZZZ"), 0.7554)

    def test_same(self):
        """Test that all operators agree with each other for counts input"""
        ans = 0.9356
        counts = Counts(
            {
                "001": 67,
                "110": 113,
                "100": 83,
                "011": 205,
                "111": 4535,
                "101": 100,
                "010": 42,
                "000": 4855,
            }
        )
        oper = "IZZ"
        oper_non_diag = "IXZ"

        exp1 = sampled_expectation_value(counts, oper)
        self.assertAlmostEqual(exp1, ans)

        exp2 = sampled_expectation_value(counts, Pauli(oper))
        self.assertAlmostEqual(exp2, ans)

        spo = SparsePauliOp([oper], coeffs=[1])
        exp3 = sampled_expectation_value(counts, spo)
        self.assertAlmostEqual(exp3, ans)

        so = SparseObservable.from_label(oper)
        exp4 = sampled_expectation_value(counts, so)
        self.assertAlmostEqual(exp4, ans)

        so_non_diag = SparseObservable.from_label(oper_non_diag)
        with self.assertRaisesRegex(ValueError, "Operator string .* contains non-diagonal terms"):
            _ = sampled_expectation_value(counts, so_non_diag)

    def test_asym_ops(self):
        """Test that asymmetric exp values work"""
        dist = QuasiDistribution(PROBS)
        self.assertAlmostEqual(sampled_expectation_value(dist, "0III"), 0.5318)
        self.assertAlmostEqual(sampled_expectation_value(dist, "III0"), 0.5285)
        self.assertAlmostEqual(sampled_expectation_value(dist, "1011"), 0.0211)

    def test_probdist(self):
        """Test that ProbDistro"""
        dist = ProbDistribution(PROBS)
        result = sampled_expectation_value(dist, "IZIZ")
        self.assertAlmostEqual(result, 0.8864)

        result2 = sampled_expectation_value(dist, "00ZI")
        self.assertAlmostEqual(result2, 0.4376)

    def test_complex_coefficient(self):
        """Test that complex coefficients return complex expectation values.

        Regression test for #11393."""
        dist = {"11111": 1}
        complex_coeff = SparsePauliOp(["ZZZZZ"], coeffs=[1j])
        float_coeff = SparsePauliOp(["ZZZZZ"], coeffs=[1])

        self.assertAlmostEqual(-1j, sampled_expectation_value(dist, complex_coeff))
        self.assertAlmostEqual(-1, sampled_expectation_value(dist, float_coeff))


if __name__ == "__main__":
    unittest.main(verbosity=2)
