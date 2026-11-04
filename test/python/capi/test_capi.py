# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from pathlib import Path

from qiskit import capi
from test import QiskitTestCase


class TestCAPI(QiskitTestCase):
    def test_includes_exists(self):
        path = Path(capi.get_include())
        self.assertIn("QISKIT_H", (path / "qiskit.h").read_text(encoding="utf-8"))
        self.assertLess(set(), set((path / "qiskit").glob("*.h")))

    def test_library_exists(self):
        self.assertTrue(Path(capi.get_lib()).exists())
