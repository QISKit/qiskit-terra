# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test calling passes (passmanager-less)"""

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PropertySet, TransformationPass, AnalysisPass
from ._dummy_passes import PassD_TP_NR_NP, PassE_AP_NR_NP, PassN_AP_NR_NP
from test import QiskitTestCase


def initial_properties(circuit):
    """The properties that are inherent to the start of pass-manager execution."""
    return {
        "original_qubit_indices": {bit: i for i, bit in enumerate(circuit.qubits)},
        "num_input_qubits": circuit.num_qubits,
    }


class TestPassCall(QiskitTestCase):
    """Test calling passes (passmanager-less)."""

    def assertMessageLog(self, context, messages):
        """Checks the log messages"""
        self.assertEqual([record.message for record in context.records], messages)

    def test_transformation_pass(self):
        """Call a transformation pass without a scheduler"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")

        pass_d = PassD_TP_NR_NP(argument1=[1, 2])
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_d(circuit)

        self.assertMessageLog(cm, ["run transformation pass PassD_TP_NR_NP", "argument [1, 2]"])
        self.assertEqual(circuit, result)

    def test_analysis_pass_dict(self):
        """Call an analysis pass without a scheduler (property_set dict)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = {"another_property": "another_value"}

        pass_e = PassE_AP_NR_NP("value")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(cm, ["run analysis pass PassE_AP_NR_NP", "set property as value"])
        self.assertEqual(
            property_set,
            {
                "another_property": "another_value",
                "property": "value",
                **initial_properties(circuit),
            },
        )
        self.assertIsInstance(property_set, dict)
        self.assertEqual(circuit, result)

    def test_analysis_pass_property_set(self):
        """Call an analysis pass without a scheduler (PropertySet dict)"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = PropertySet({"another_property": "another_value"})

        pass_e = PassE_AP_NR_NP("value")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(cm, ["run analysis pass PassE_AP_NR_NP", "set property as value"])
        self.assertEqual(
            property_set,
            PropertySet(
                {
                    "another_property": "another_value",
                    "property": "value",
                    **initial_properties(circuit),
                }
            ),
        )
        self.assertIsInstance(property_set, PropertySet)
        self.assertEqual(circuit, result)

    def test_analysis_pass_remove_property(self):
        """Call an analysis pass that removes a property without a scheduler"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        property_set = {"to remove": "value to remove", "to none": "value to none"}

        pass_e = PassN_AP_NR_NP("to remove", "to none")
        with self.assertLogs("LocalLogger", level="INFO") as cm:
            result = pass_e(circuit, property_set)

        self.assertMessageLog(
            cm,
            [
                "run analysis pass PassN_AP_NR_NP",
                "property to remove deleted",
                "property to none noned",
            ],
        )
        self.assertEqual(
            property_set, PropertySet({"to none": None, **initial_properties(circuit)})
        )
        self.assertIsInstance(property_set, dict)
        self.assertEqual(circuit, result)

    def test_pass_requires(self):
        """The 'requires' field of a pass should be respected when called."""
        name = "my_property"
        value = "hello, world"

        assert_equal = self.assertEqual

        class Analyse(AnalysisPass):
            """Dummy pass to set a property."""

            def run(self, dag):
                self.property_set[name] = value
                return dag

        class Execute(TransformationPass):
            """Dummy pass to assert that its required pass ran."""

            def __init__(self):
                super().__init__()
                self.requires.append(Analyse())

            def run(self, dag):
                assert_equal(self.property_set[name], value)
                return dag

        pass_ = Execute()
        pass_(QuantumCircuit())
        self.assertEqual(pass_.property_set[name], value)
