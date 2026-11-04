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

"""BasePass and generic pass testing"""

import unittest.mock
from ._dummy_passes import DummyAP, DummyTP, PassA_TP_NR_NP, PassD_TP_NR_NP, PassE_AP_NR_NP
from test import QiskitTestCase


class TestGenericPass(QiskitTestCase):
    """Passes have common characteristics defined in BasePass."""

    def test_is_TP_or_AP(self):
        """Passes have is_transformation_pass and is_analysis_pass properties."""
        tp_pass = DummyTP()
        self.assertTrue(tp_pass.is_transformation_pass)
        self.assertFalse(tp_pass.is_analysis_pass)
        ap_pass = DummyAP()
        self.assertFalse(ap_pass.is_transformation_pass)
        self.assertTrue(ap_pass.is_analysis_pass)

    def test_pass_diff_TP_AP(self):
        """Different passes are different"""
        pass1 = DummyAP()
        pass2 = DummyTP()
        self.assertNotEqual(pass1, pass2)

    def test_pass_diff_parent_child(self):
        """Parents are different from their children"""
        pass2 = DummyTP()
        pass1 = PassD_TP_NR_NP()
        self.assertNotEqual(pass1, pass2)

    def test_pass_diff_args(self):
        """Same pass with different arguments are different"""
        pass1 = PassD_TP_NR_NP(argument1=[1, 2])
        pass2 = PassD_TP_NR_NP(argument1=[2, 1])
        self.assertNotEqual(pass1, pass2)

    def test_pass_args_kwargs(self):
        """Same pass if same args and kwargs"""
        pass1 = PassD_TP_NR_NP(argument1=[1, 2])
        pass2 = PassD_TP_NR_NP([1, 2])
        self.assertEqual(pass1, pass2)

    def test_pass_kwargs_out_of_order(self):
        """Passes instances with same arguments (independently of the order) are the same"""
        pass1 = PassD_TP_NR_NP(argument1=1, argument2=2)
        pass2 = PassD_TP_NR_NP(argument2=2, argument1=1)
        self.assertEqual(pass1, pass2)

    def test_pass_kwargs_and_args(self):
        """Passes instances with same arguments (independently if they are named or not) are the
        same"""
        pass1 = PassD_TP_NR_NP(1, 2)
        pass2 = PassD_TP_NR_NP(argument2=2, argument1=1)
        self.assertEqual(pass1, pass2)

    def test_set_identity(self):
        """Two "instances" of the same pass in a set are counted as one."""
        a_set = set()
        a_set.add(PassA_TP_NR_NP())
        a_set.add(PassA_TP_NR_NP())
        self.assertEqual(len(a_set), 1)

    def test_identity_params_same_hash(self):
        """True is 1. They are not the same parameter."""
        self.assertNotEqual(PassE_AP_NR_NP(True), PassE_AP_NR_NP(1))


if __name__ == "__main__":
    unittest.main()
