# This code is part of Qiskit.
#
# (C) Copyright IBM 2022
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the staged passmanager logic"""

from unittest.mock import patch

from ddt import data, ddt

from qiskit.transpiler import PassManager, StagedPassManager
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes import Optimize1qGates, Depth, BasicSwap, BasisTranslator
from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)
from test import QiskitTestCase


@ddt
class TestStagedPassManager(QiskitTestCase):
    def test_default_stages(self):
        spm = StagedPassManager()
        self.assertEqual(
            spm.stages, ("init", "layout", "routing", "translation", "optimization", "scheduling")
        )
        spm = StagedPassManager(
            init=PassManager([Optimize1qGates()]),
            routing=PassManager([BasisTranslator(std_eqlib, ["u", "cx"])]),
            scheduling=PassManager([Depth()]),
        )
        self.assertEqual(
            [x.__class__.__name__ for x in spm.to_flow_controller().tasks],
            ["Optimize1qGates", "BasisTranslator", "Depth"],
        )

    def test_inplace_edit(self):
        spm = StagedPassManager(stages=["single_stage"])
        spm.single_stage = PassManager([Optimize1qGates(), Depth()])
        self.assertEqual(
            [x.__class__.__name__ for x in spm.to_flow_controller().tasks],
            ["Optimize1qGates", "Depth"],
        )
        spm.single_stage.append(BasisTranslator(std_eqlib, ["u"]))
        spm.single_stage.append(Depth())
        self.assertEqual(
            [x.__class__.__name__ for x in spm.to_flow_controller().tasks],
            ["Optimize1qGates", "Depth", "BasisTranslator", "Depth"],
        )

    def test_invalid_stage(self):
        with self.assertRaises(AttributeError):
            StagedPassManager(stages=["init"], translation=PassManager())

    def test_pre_phase_is_valid_stage(self):
        spm = StagedPassManager(stages=["init"], pre_init=PassManager([Depth()]))

        self.assertEqual(
            [x.__class__.__name__ for x in spm.to_flow_controller().tasks],
            ["Depth"],
        )

    def test_append_extend_not_implemented(self):
        spm = StagedPassManager()
        with self.assertRaises(NotImplementedError):
            spm.append(Depth())
        with self.assertRaises(NotImplementedError):
            spm += PassManager()

    def test_invalid_stages(self):
        invalid_stages = [
            "two words",
            "two-words",
            "two+words",
            "two&words",
            "[two_words]",
            "<two_words>",
            "{two_words}",
            "(two_words)",
            "two^words",
            "two_words!",
            "^two_words",
            "@two_words",
            "two~words",
            r"two\words",
            "two/words",
        ]
        all_stages = invalid_stages + ["two_words", "init"]

        with self.assertRaises(ValueError) as err:
            StagedPassManager(all_stages)
        message = str(err.exception)
        for stage in invalid_stages:
            self.assertIn(stage, message)

    def test_repeated_stages(self):
        stages = ["alpha", "omega", "alpha"]
        pre_alpha = PassManager(BasisTranslator(std_eqlib, ["u", "cx"]))
        alpha = PassManager(Depth())
        post_alpha = PassManager(BasicSwap([[0, 1], [1, 2]]))
        omega = PassManager(Optimize1qGates())
        spm = StagedPassManager(
            stages, pre_alpha=pre_alpha, alpha=alpha, post_alpha=post_alpha, omega=omega
        )
        passes = (
            *pre_alpha.to_flow_controller().tasks,
            *alpha.to_flow_controller().tasks,
            *post_alpha.to_flow_controller().tasks,
            *omega.to_flow_controller().tasks,
            *pre_alpha.to_flow_controller().tasks,
            *alpha.to_flow_controller().tasks,
            *post_alpha.to_flow_controller().tasks,
        )
        self.assertEqual(spm.to_flow_controller().tasks, passes)

    def test_edit_stages(self):
        spm = StagedPassManager()
        with self.assertRaises(AttributeError):
            spm.stages = ["init"]
        with self.assertRaises(AttributeError):
            spm.expanded_stages = ["init"]

    @data(None, ["init"], ["init", "schedule"])
    def test_expanded_stages(self, stages):
        spm = StagedPassManager(stages=stages)
        expanded_stages = (stage for stage in spm.expanded_stages)
        for stage in spm.stages:
            self.assertEqual(next(expanded_stages), "pre_" + stage)
            self.assertEqual(next(expanded_stages), stage)
            self.assertEqual(next(expanded_stages), "post_" + stage)

    def test_setattr(self):
        spm = StagedPassManager()
        with self.assertRaises(TranspilerError):
            spm.init = spm
        mock_target = "qiskit.transpiler.passmanager.StagedPassManager._update_passmanager"
        with patch(mock_target, spec=True) as mock:
            spm.max_iteration = spm.max_iteration
            mock.assert_not_called()
            spm.init = None
            mock.assert_called_once()

    def test_getitem(self):
        pm = PassManager(Depth())
        spm = StagedPassManager(init=pm)
        mock_target = "qiskit.transpiler.passmanager.StagedPassManager._update_passmanager"
        with patch(mock_target, spec=True) as mock:
            _ = spm[0]
            mock.assert_called_once()
