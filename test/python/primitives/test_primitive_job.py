# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for PrimitiveJob."""

import pickle
from test import QiskitTestCase

import numpy as np
from ddt import data, ddt

from qiskit import QuantumCircuit
from qiskit.primitives import PrimitiveJob, StatevectorSampler


@ddt
class TestPrimitiveJob(QiskitTestCase):
    """Tests PrimitiveJob."""

    @data(1, 2, 3)
    def test_serialize(self, size):
        """Test serialize."""
        n = 2
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.measure_all()
        sampler = StatevectorSampler()
        job = sampler.run([qc] * size)
        obj = pickle.dumps(job)
        job2 = pickle.loads(obj)
        self.assertIsInstance(job2, PrimitiveJob)
        self.assertEqual(job.job_id(), job2.job_id())
        self.assertEqual(job.status(), job2.status())
        self.assertEqual(job.metadata, job2.metadata)
        result = job.result()
        result2 = job2.result()
        self.assertEqual(result.metadata, result2.metadata)
        self.assertEqual(len(result), len(result2))
        for sampler_pub in result:
            self.assertEqual(sampler_pub.metadata, sampler_pub.metadata)
            self.assertEqual(sampler_pub.data.keys(), sampler_pub.data.keys())
            np.testing.assert_allclose(sampler_pub.join_data().array, sampler_pub.join_data().array)
