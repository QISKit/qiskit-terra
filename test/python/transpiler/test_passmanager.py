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


"""Test the passmanager logic"""

import copy

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import U2Gate
from qiskit.converters import circuit_to_dag
from qiskit.passmanager.flow_controllers import (
    FlowControllerLinear,
    ConditionalController,
    DoWhileController,
)
from qiskit.transpiler import PassManager, PropertySet, TransformationPass, AnalysisPass
from qiskit.transpiler.passes import Optimize1qGates, BasisTranslator, ResourceEstimation
from qiskit.circuit.library.standard_gates.equivalence_library import (
    StandardEquivalenceLibrary as std_eqlib,
)
from test import QiskitTestCase


class TestPassManager(QiskitTestCase):
    """Test Pass manager logic."""

    def test_callback(self):
        """Test the callback parameter."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start.append(U2Gate(0, np.pi), [qr[0]])
        expected_start_dag = circuit_to_dag(expected_start)

        base_property_set = PropertySet(
            {
                "original_qubit_indices": {bit: i for i, bit in enumerate(circuit.qubits)},
                "num_input_qubits": circuit.num_qubits,
            }
        )

        expected_end = QuantumCircuit(qr)
        expected_end.append(U2Gate(0, np.pi), [qr[0]])
        expected_end_dag = circuit_to_dag(expected_end)

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict["dag"] = copy.deepcopy(kwargs["dag"])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(BasisTranslator(std_eqlib, ["u2"]))
        passmanager.append(Optimize1qGates())
        passmanager.run(circuit, callback=callback)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(calls[0]), 5)
        self.assertEqual(calls[0]["count"], 0)
        self.assertEqual(calls[0]["pass_"].name(), "BasisTranslator")
        self.assertEqual(expected_start_dag, calls[0]["dag"])
        self.assertIsInstance(calls[0]["time"], float)
        self.assertEqual(calls[0]["property_set"], base_property_set)
        self.assertEqual("MyCircuit", calls[0]["dag"].name)
        self.assertEqual(len(calls[1]), 5)
        self.assertEqual(calls[1]["count"], 1)
        self.assertEqual(calls[1]["pass_"].name(), "Optimize1qGates")
        self.assertEqual(expected_end_dag, calls[1]["dag"])
        self.assertIsInstance(calls[1]["time"], float)
        self.assertEqual(calls[1]["property_set"], base_property_set)
        self.assertEqual("MyCircuit", calls[1]["dag"].name)

    def test_callback_multi_circuits(self):
        """Test callback with multiple circuits."""

        qr = QuantumRegister(1, "qr")

        circuit1 = QuantumCircuit(qr, name="Circuit1")
        circuit1.h(qr[0])
        circuit1.h(qr[0])
        circuit1.h(qr[0])
        expected_start_1 = QuantumCircuit(qr)
        expected_start_1.append(U2Gate(0, np.pi), [qr[0]])
        expected_start_1.append(U2Gate(0, np.pi), [qr[0]])
        expected_start_1.append(U2Gate(0, np.pi), [qr[0]])

        circuit2 = circuit1.copy(name="Circuit2")

        def callback(**kwargs):
            dag = kwargs["dag"]
            dag.name += "_callback"

        passmanager = PassManager()
        passmanager.append(BasisTranslator(std_eqlib, ["u2"]))
        passmanager.append(Optimize1qGates())

        out_circuits = passmanager.run([circuit1, circuit2], callback=callback)

        # Check that callback visibly modified circuit names
        self.assertTrue(out_circuits[0].name.endswith("_callback"))
        self.assertTrue(out_circuits[1].name.endswith("_callback"))

    def test_callback_multi_circuits_lambda(self):
        """Test callback with multiple circuits using lambda function."""

        qr = QuantumRegister(1, "qr")

        circuit1 = QuantumCircuit(qr, name="Circuit1")
        circuit1.h(qr[0])
        circuit1.h(qr[0])
        circuit1.h(qr[0])

        circuit2 = circuit1.copy(name="Circuit2")

        # Run pass manager with lambda callback that modifies the DAG by appending XGate
        passmanager = PassManager()
        passmanager.append(BasisTranslator(std_eqlib, ["u2"]))
        passmanager.append(Optimize1qGates())

        from qiskit.circuit.library import XGate

        out_circuits = passmanager.run(
            [circuit1, circuit2],
            callback=lambda **kwargs: kwargs["dag"].apply_operation_back(
                XGate(), [kwargs["dag"].qubits[0]]
            ),
        )

        # Check that callback visibly modified the circuits by adding an XGate
        self.assertIn("x", out_circuits[0].count_ops())
        self.assertIn("x", out_circuits[1].count_ops())

        # Optional: structure checks for original circuit
        self.assertTrue(out_circuits[0].name.startswith("Circuit1"))
        self.assertTrue(out_circuits[1].name.startswith("Circuit2"))

    def test_callback_with_pass_requires(self):
        """Test the callback with a pass with pass requirements."""
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr, name="MyCircuit")
        circuit.z(qr[0])
        circuit.cx(qr[0], qr[2])
        circuit.z(qr[0])
        expected_start = QuantumCircuit(qr)
        expected_start.z(qr[0])
        expected_start.cx(qr[0], qr[2])
        expected_start.z(qr[0])
        expected_start_dag = circuit_to_dag(expected_start)

        expected_end = QuantumCircuit(qr)
        expected_end.cx(qr[0], qr[2])

        calls = []

        def callback(**kwargs):
            out_dict = kwargs
            out_dict["dag"] = copy.deepcopy(kwargs["dag"])
            calls.append(out_dict)

        passmanager = PassManager()
        passmanager.append(ResourceEstimation())
        passmanager.run(circuit, callback=callback)

        self.assertEqual(len(calls), 7)

        required_passes = [
            "Depth",
            "Width",
            "Size",
            "CountOps",
            "NumTensorFactors",
            "NumQubits",
            "ResourceEstimation",
        ]

        for call_entry in range(7):
            self.assertEqual(len(calls[call_entry]), 5)
            self.assertEqual(calls[call_entry]["count"], call_entry)
            self.assertEqual(calls[call_entry]["pass_"].name(), required_passes[call_entry])
            self.assertEqual(expected_start_dag, calls[call_entry]["dag"])
            self.assertIsInstance(calls[call_entry]["time"], float)
            self.assertIsInstance(calls[call_entry]["property_set"], PropertySet)
            self.assertEqual("MyCircuit", calls[call_entry]["dag"].name)

    def test_to_flow_controller(self):
        """Test that conversion to a `FlowController` works, and the result can be added to a
        circuit and conditioned, with the condition only being called once."""

        class DummyPass(TransformationPass):
            def __init__(self, x):
                super().__init__()
                self.x = x

            def run(self, dag):
                return dag

        def repeat(count):
            def condition(_):
                nonlocal count
                if not count:
                    return False
                count -= 1
                return True

            return condition

        def make_inner(prefix):
            inner = PassManager()
            inner.append(DummyPass(f"{prefix} 1"))
            inner.append(ConditionalController(DummyPass(f"{prefix} 2"), condition=lambda _: False))
            inner.append(ConditionalController(DummyPass(f"{prefix} 3"), condition=lambda _: True))
            inner.append(DoWhileController(DummyPass(f"{prefix} 4"), do_while=repeat(1)))
            return inner.to_flow_controller()

        self.assertIsInstance(make_inner("test"), FlowControllerLinear)

        outer = PassManager()
        outer.append(make_inner("first"))
        outer.append(ConditionalController(make_inner("second"), condition=lambda _: False))
        # The intent of this `condition=repeat(1)` is to ensure that the outer condition is only
        # checked once and not flattened into the inner controllers; an inner pass invalidating the
        # condition should not affect subsequent passes once the initial condition was met.
        outer.append(ConditionalController(make_inner("third"), condition=repeat(1)))

        calls = []

        def callback(pass_, **_):
            self.assertIsInstance(pass_, DummyPass)
            calls.append(pass_.x)

        outer.run(QuantumCircuit(), callback=callback)

        expected = [
            "first 1",
            "first 3",
            # it's a do-while loop, not a while, which is why the `repeat(1)` gives two calls
            "first 4",
            "first 4",
            # If the outer pass-manager condition is called more than once, then only the first of
            # the `third` passes will appear.
            "third 1",
            "third 3",
            "third 4",
            "third 4",
        ]
        self.assertEqual(calls, expected)

    def test_override_initial_property_set(self):
        """Test that the ``property_set`` argument allows seeding the base analysis."""
        input_name = "my_property"
        output_name = "output_property"

        class Analyse(AnalysisPass):
            def run(self, dag):
                self.property_set[output_name] = self.property_set[input_name]
                return dag

        pm = PassManager([Analyse()])
        pm.run(QuantumCircuit(), property_set={input_name: "hello, world"})
        self.assertEqual(pm.property_set[output_name], "hello, world")

        pm.run(QuantumCircuit(), property_set={input_name: "a different string"})
        self.assertEqual(pm.property_set[output_name], "a different string")
