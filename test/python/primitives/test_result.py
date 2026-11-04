# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BasePrimitiveResult."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any
from ddt import data, ddt, unpack

from qiskit.primitives.base.base_result_v1 import _BasePrimitiveResultV1 as BasePrimitiveResult
from test import QiskitTestCase


@dataclass
class Result(BasePrimitiveResult):
    """Dummy result dataclass implementing BasePrimitiveResult."""

    field_1: Collection[Any]
    field_2: Collection[Any]


@ddt
class TestBasePrimitiveResult(QiskitTestCase):
    """Tests BasePrimitiveResult."""

    @data(0, 1.2, True, "sequence", b"sequence", {"name": "value"})
    def test_post_init_type_error(self, field_1):
        """Tests post init type error."""
        self.assertRaises(TypeError, Result, *(field_1, []))

    @data(([1], []), ([], [1]), ([1, 2], []), ([1], [1, 2]))
    @unpack
    def test_post_init_value_error(self, field_1, field_2):
        """Tests post init value error."""
        self.assertRaises(ValueError, Result, *(field_1, field_2))

    def test_field_names(self):
        """Tests field names ("field_1", "field_2")."""
        result = Result([], [])
        self.assertEqual(result._field_names, ("field_1", "field_2"))

    @data(([], []), ([0], [0]), ([0], [1]))
    @unpack
    def test_field_values(self, field_1, field_2):
        """Tests field values ({field_1}, {field_2})."""
        result = Result(field_1, field_2)
        self.assertEqual(result._field_values, (field_1, field_2))
