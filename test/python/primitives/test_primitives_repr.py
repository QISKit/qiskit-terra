# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test __repr__ methods for StatevectorSampler and StatevectorEstimator classes."""

from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from test import QiskitTestCase


class TestStatevectorSamplerRepr(QiskitTestCase):
    """Tests for StatevectorSampler.__repr__ method."""

    def test_statevector_sampler_repr_default(self):
        """Test StatevectorSampler repr with default parameters."""
        sampler = StatevectorSampler()
        result = repr(sampler)
        expected = f"<StatevectorSampler with default_shots={sampler.default_shots}, seed={sampler.seed}>"
        self.assertEqual(result, expected)

    def test_statevector_sampler_repr_with_shots(self):
        """Test StatevectorSampler repr with custom shots."""
        sampler = StatevectorSampler(default_shots=2048)
        result = repr(sampler)
        expected = "<StatevectorSampler with default_shots=2048, seed=None>"
        self.assertEqual(result, expected)

    def test_statevector_sampler_repr_with_seed(self):
        """Test StatevectorSampler repr with seed."""
        sampler = StatevectorSampler(seed=42)
        result = repr(sampler)
        expected = f"<StatevectorSampler with default_shots={sampler.default_shots}, seed=42>"
        self.assertEqual(result, expected)

    def test_statevector_sampler_repr_with_shots_and_seed(self):
        """Test StatevectorSampler repr with both shots and seed."""
        sampler = StatevectorSampler(default_shots=2048, seed=42)
        result = repr(sampler)
        expected = "<StatevectorSampler with default_shots=2048, seed=42>"
        self.assertEqual(result, expected)


class TestStatevectorEstimatorRepr(QiskitTestCase):
    """Tests for StatevectorEstimator.__repr__ method."""

    def test_statevector_estimator_repr_default(self):
        """Test StatevectorEstimator repr with default parameters."""
        estimator = StatevectorEstimator()
        result = repr(estimator)
        expected = f"<StatevectorEstimator with default_precision={estimator.default_precision}, seed={estimator.seed}>"
        self.assertEqual(result, expected)

    def test_statevector_estimator_repr_with_precision(self):
        """Test StatevectorEstimator repr with custom precision."""
        estimator = StatevectorEstimator(default_precision=0.01)
        result = repr(estimator)
        expected = "<StatevectorEstimator with default_precision=0.01, seed=None>"
        self.assertEqual(result, expected)

    def test_statevector_estimator_repr_with_seed(self):
        """Test StatevectorEstimator repr with seed."""
        estimator = StatevectorEstimator(seed=123)
        result = repr(estimator)
        expected = f"<StatevectorEstimator with default_precision={estimator.default_precision}, seed=123>"
        self.assertEqual(result, expected)

    def test_statevector_estimator_repr_with_precision_and_seed(self):
        """Test StatevectorEstimator repr with both precision and seed."""
        estimator = StatevectorEstimator(default_precision=0.01, seed=123)
        result = repr(estimator)
        expected = "<StatevectorEstimator with default_precision=0.01, seed=123>"
        self.assertEqual(result, expected)

