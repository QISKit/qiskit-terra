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

"""BasicProvider Backends Test."""

from qiskit.providers.basic_provider.basic_provider import BasicProvider
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from test import QiskitTestCase


class TestBasicProviderBackends(QiskitTestCase):
    """Qiskit BasicProvider Backends (Object) Tests."""

    def setUp(self):
        super().setUp()
        self.provider = BasicProvider()
        self.backend_name = "basic_simulator"

    def test_backends(self):
        """Test the provider has backends."""
        backends = self.provider.backends()
        self.assertTrue(len(backends) > 0)

    def test_get_backend(self):
        """Test getting a backend from the provider."""
        backend = self.provider.get_backend(name=self.backend_name)
        self.assertEqual(backend.name, self.backend_name)

    def test_aliases_fail(self):
        """Test a failing backend lookup."""
        self.assertRaises(QiskitBackendNotFoundError, BasicProvider().get_backend, "bad_name")
