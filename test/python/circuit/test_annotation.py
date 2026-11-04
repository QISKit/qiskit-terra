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

"""Test cases for circuit annotations."""


from qiskit.circuit import annotation
from test import QiskitTestCase

# Most of the tests for this module are elsewhere, such as for OpenQASM 3 serialisation being in the
# OpenQASM 3 tests, and similar for QPY.


class TestAnnotation(QiskitTestCase):
    def test_iter_namespaces(self):
        self.assertEqual(
            list(annotation.iter_namespaces("hello.world")), ["hello.world", "hello", ""]
        )
        self.assertEqual(list(annotation.iter_namespaces("single_word")), ["single_word", ""])
        self.assertEqual(list(annotation.iter_namespaces("")), [""])
