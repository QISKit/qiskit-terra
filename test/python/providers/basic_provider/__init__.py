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

"""Qiskit BasicProvider integration tests."""

from qiskit import transpile


class BasicProviderBackendTestMixin:
    """Test mixins for BasicProvider backend tests."""

    def test_run_circuit(self):
        """Test running a single circuit."""
        transpiled_qc = transpile(self.circuit, self.backend)
        job = self.backend.run(transpiled_qc)
        result = job.result()
        self.assertEqual(result.success, True)
        return result
