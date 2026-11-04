# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test for unit conversion functions."""

from ddt import ddt, data

from qiskit.utils import apply_prefix, detach_prefix
from test import QiskitTestCase


@ddt
class TestUnitConversion(QiskitTestCase):
    """Test the unit conversion utilities."""

    def test_apply_prefix(self):
        """Test applying prefix to value."""
        ref_values = [
            ([1.0, "THz"], 1e12),
            ([1.0, "GHz"], 1e9),
            ([1.0, "MHz"], 1e6),
            ([1.0, "kHz"], 1e3),
            ([1.0, "mHz"], 1e-3),
            ([1.0, "µHz"], 1e-6),
            ([1.0, "uHz"], 1e-6),
            ([1.0, "nHz"], 1e-9),
            ([1.0, "pHz"], 1e-12),
        ]

        for args, ref_ret in ref_values:
            self.assertEqual(apply_prefix(*args), ref_ret)

    def test_not_convert_meter(self):
        """Test not apply prefix to meter."""
        self.assertEqual(apply_prefix(1.0, "m"), 1.0)

    def test_detach_prefix(self):
        """Test detach prefix from the value."""
        ref_values = [
            (1e12, (1.0, "T")),
            (1e11, (100.0, "G")),
            (1e10, (10.0, "G")),
            (1e9, (1.0, "G")),
            (1e8, (100.0, "M")),
            (1e7, (10.0, "M")),
            (1e6, (1.0, "M")),
            (1e5, (100.0, "k")),
            (1e4, (10.0, "k")),
            (1e3, (1.0, "k")),
            (100, (100.0, "")),
            (10, (10.0, "")),
            (1.0, (1.0, "")),
            (0.1, (100.0, "m")),
            (0.01, (10.0, "m")),
            (1e-3, (1.0, "m")),
            (1e-4, (100.0, "µ")),
            (1e-5, (10.0, "µ")),
            (1e-6, (1.0, "µ")),
            (1e-7, (100.0, "n")),
            (1e-8, (10.0, "n")),
            (1e-9, (1.0, "n")),
            (1e-10, (100.0, "p")),
            (1e-11, (10.0, "p")),
            (1e-12, (1.0, "p")),
        ]

        for arg, ref_rets in ref_values:
            self.assertTupleEqual(detach_prefix(arg), ref_rets)

    def test_detach_prefix_with_zero(self):
        """Test detach prefix by input zero."""
        self.assertTupleEqual(detach_prefix(0.0), (0.0, ""))

    def test_detach_prefix_with_negative(self):
        """Test detach prefix by input negative values."""
        self.assertTupleEqual(detach_prefix(-1.234e7), (-12.34, "M"))

    def test_detach_prefix_with_value_too_large(self):
        """Test detach prefix by input too large value."""
        with self.assertRaises(Exception):
            self.assertTupleEqual(detach_prefix(1e20), (1e20, ""))

    def test_detach_prefix_with_value_too_small(self):
        """Test detach prefix by input too small value."""
        with self.assertRaises(Exception):
            self.assertTupleEqual(detach_prefix(1e-20), (1e-20, ""))

    def test_rounding(self):
        """Test detach prefix with decimal specification."""
        ret = detach_prefix(999_999.991)
        self.assertTupleEqual(ret, (999.999991, "k"))

        ret = detach_prefix(999_999.991, decimal=4)
        self.assertTupleEqual(ret, (1.0, "M"))

        ret = detach_prefix(999_999.991, decimal=5)
        self.assertTupleEqual(ret, (999.99999, "k"))

    @data(
        -20.791378538739863,
        9.242757760406565,
        2.7366806276451543,
        9.183776167253349,
        7.658091886606501,
        -12.21553566621071,
        8.914055281578145,
        1.2518807770035825,
        -6.652899195646036,
        -4.647159596697976,
    )
    def test_get_same_value_after_attach_detach(self, value: float):
        """Test if same value can be obtained."""
        unit = "Hz"

        for prefix in ["P", "T", "G", "k", "m", "µ", "n", "p", "f"]:
            scaled_val = apply_prefix(value, prefix + unit)
            test_val, ret_prefix = detach_prefix(scaled_val)
            self.assertAlmostEqual(test_val, value)
            self.assertEqual(prefix, ret_prefix)

    def test_get_symbol_mu(self):
        """Test if µ is returned rather than u."""
        _, prefix = detach_prefix(3e-6)
        self.assertEqual(prefix, "µ")
