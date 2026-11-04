# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Tests for singleton gate and instruction behavior
"""

import copy
import io
import pickle
import sys
import types
import unittest.mock
import uuid

from qiskit.circuit.library import (
    HGate,
    SXGate,
    CXGate,
    CZGate,
    CSwapGate,
    CHGate,
    XGate,
    C4XGate,
)
from qiskit.circuit import Measure, Reset
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.singleton import SingletonGate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from test.utils.base import QiskitTestCase


class TestSingleton(QiskitTestCase):
    """Qiskit SingletonGate and SingletonInstruction tests."""

    def test_default_singleton(self):
        gate = HGate()
        new_gate = HGate()
        self.assertIs(gate, new_gate)

    def test_base_class(self):
        gate = HGate()
        self.assertIsInstance(gate, HGate)
        self.assertIs(gate.base_class, HGate)

    def test_label_not_singleton(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)

    def test_raise_on_state_mutation(self):
        gate = HGate()
        with self.assertRaises(TypeError):
            gate.label = "foo"

    def test_default_singleton_copy(self):
        gate = HGate()
        copied = gate.copy()
        self.assertIs(gate, copied)

    def test_label_copy(self):
        gate = HGate(label="special")
        copied = gate.copy()
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)

    def test_label_copy_new(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = gate.copy()
        copied_label = label_gate.copy()
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_deepcopy(self):
        gate = HGate()
        copied = copy.deepcopy(gate)
        self.assertIs(gate, copied)

    def test_deepcopy_with_label(self):
        gate = HGate(label="special")
        copied = copy.deepcopy(gate)
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)
        self.assertEqual(copied.label, "special")

    def test_label_deepcopy_new(self):
        gate = HGate()
        label_gate = HGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = copy.deepcopy(gate)
        copied_label = copy.deepcopy(label_gate)
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_control_a_singleton(self):
        singleton_gate = HGate()
        gate = HGate(label="special")
        ch = gate.control(annotated=False, label="my_ch")
        self.assertEqual(ch.base_gate.label, "special")
        self.assertIsNot(ch.base_gate, singleton_gate)

    def test_round_trip_dag_conversion(self):
        qc = QuantumCircuit(1)
        gate = HGate()
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIs(qc.data[0].operation, out.data[0].operation)

    def test_round_trip_dag_conversion_with_label(self):
        gate = HGate(label="special")
        qc = QuantumCircuit(1)
        qc.append(gate, [0])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIsNot(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(out.data[0].operation.label, "special")

    def test_is_mutable(self):
        gate = HGate()
        self.assertFalse(gate.mutable)
        label_gate = HGate(label="foo")
        self.assertTrue(label_gate.mutable)
        self.assertIsNot(gate, label_gate)

    def test_to_mutable(self):
        gate = HGate()
        self.assertFalse(gate.mutable)
        new_gate = gate.to_mutable()
        self.assertTrue(new_gate.mutable)
        self.assertIsNot(gate, new_gate)

    def test_to_mutable_setter(self):
        gate = HGate()
        self.assertFalse(gate.mutable)
        mutable_gate = gate.to_mutable()
        mutable_gate.label = "foo"
        self.assertTrue(mutable_gate.mutable)
        self.assertIsNot(gate, mutable_gate)
        self.assertEqual(mutable_gate.label, "foo")

    def test_to_mutable_of_mutable_instance(self):
        gate = HGate(label="foo")
        mutable_copy = gate.to_mutable()
        self.assertIsNot(gate, mutable_copy)
        self.assertEqual(mutable_copy.label, gate.label)
        mutable_copy.label = "not foo"
        self.assertNotEqual(mutable_copy.label, gate.label)

    def test_set_custom_attr(self):
        gate = SXGate()
        with self.assertRaises(TypeError):
            gate.custom_foo = 12345
        mutable_gate = gate.to_mutable()
        self.assertTrue(mutable_gate.mutable)
        mutable_gate.custom_foo = 12345
        self.assertEqual(12345, mutable_gate.custom_foo)

    def test_positional_label(self):
        gate = SXGate()
        label_gate = SXGate("I am a little label")
        self.assertIsNot(gate, label_gate)
        self.assertEqual(label_gate.label, "I am a little label")

    def test_immutable_pickle(self):
        gate = SXGate()
        self.assertFalse(gate.mutable)
        with io.BytesIO() as fd:
            pickle.dump(gate, fd)
            fd.seek(0)
            copied = pickle.load(fd)
        self.assertFalse(copied.mutable)
        self.assertIs(copied, gate)

    def test_mutable_pickle(self):
        label_gate = SXGate(label="FOO")
        gate = SXGate()
        self.assertIsNot(gate, label_gate)
        self.assertEqual(label_gate.label, "FOO")
        self.assertTrue(label_gate.mutable)
        with io.BytesIO() as fd:
            pickle.dump(label_gate, fd)
            fd.seek(0)
            copied = pickle.load(fd)
        self.assertEqual(copied, label_gate)
        self.assertTrue(copied.mutable)

    def test_uses_default_arguments(self):
        class MyGate(SingletonGate):
            def __init__(self, label="my label"):
                super().__init__("my_gate", 1, [], label=label)

        gate = MyGate()
        self.assertIs(gate, MyGate())
        self.assertFalse(gate.mutable)
        self.assertIs(gate.base_class, MyGate)
        self.assertEqual(gate.label, "my label")

        with self.assertRaisesRegex(TypeError, "immutable"):
            gate.label = None

    def test_suppress_singleton(self):
        # Mostly the test here is that the `class` statement passes; it would raise if it attempted
        # to create a singleton instance since there's no defaults.
        class MyAbstractGate(SingletonGate, create_default_singleton=False):
            def __init__(self, x):
                super().__init__("my_abstract", 1, [])
                self.x = x

        gate = MyAbstractGate(1)
        self.assertTrue(gate.mutable)
        self.assertEqual(gate.x, 1)
        self.assertIsNot(MyAbstractGate(1), MyAbstractGate(1))

    def test_return_type_singleton_instructions(self):
        measure = Measure()
        new_measure = Measure()
        self.assertIs(measure, new_measure)
        self.assertIs(measure.base_class, Measure)
        self.assertIsInstance(measure, Measure)

        reset = Reset()
        new_reset = Reset()
        self.assertIs(reset, new_reset)
        self.assertIs(reset.base_class, Reset)
        self.assertIsInstance(reset, Reset)

    def test_singleton_instruction_integration(self):
        measure = Measure()
        reset = Reset()
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        qc.reset(0)
        self.assertIs(qc.data[0].operation, measure)
        self.assertIs(qc.data[1].operation, reset)

    def test_inherit_singleton_instructions(self):
        class ESPMeasure(Measure):
            pass

        measure_base = Measure()
        esp_measure = ESPMeasure()
        self.assertIs(esp_measure, ESPMeasure())
        self.assertIsNot(esp_measure, measure_base)
        self.assertIs(measure_base.base_class, Measure)
        self.assertIs(esp_measure.base_class, ESPMeasure)

    def test_singleton_with_default(self):
        # Explicitly setting the label to its default.
        gate = HGate(label=None)
        self.assertIs(gate, HGate())
        self.assertIsNot(gate, HGate(label="label"))

    def test_additional_singletons(self):
        additional_inputs = [
            ((1,), {}),
            ((2,), {"label": "x"}),
        ]

        class Discrete(SingletonGate, additional_singletons=additional_inputs):
            def __init__(self, n=0, label=None):
                super().__init__("discrete", 1, [], label=label)
                self.n = n

            @staticmethod
            def _singleton_lookup_key(n=0, label=None):
                # This is an atypical usage - in Qiskit standard gates, the `label` being set
                # not-None should not generate a singleton, so should return a mutable instance.
                return (n, label)

        default = Discrete()
        self.assertIs(default, Discrete())
        self.assertIs(default, Discrete(0, label=None))
        self.assertEqual(default.n, 0)
        self.assertIsNot(default, Discrete(1))

        one = Discrete(1)
        self.assertIs(one, Discrete(1))
        self.assertIs(one, Discrete(1, label=None))
        self.assertEqual(one.n, 1)
        self.assertIs(one.label, None)

        two = Discrete(2, label="x")
        self.assertIs(two, Discrete(2, label="x"))
        self.assertIsNot(two, Discrete(2))
        self.assertEqual(two.n, 2)
        self.assertEqual(two.label, "x")

        # This doesn't match any of the defined singletons, and we're checking that it's not
        # spuriously cached without us asking for it.
        self.assertIsNot(Discrete(2), Discrete(2))

    def test_additional_singletons_copy(self):
        additional_inputs = [
            ((1,), {}),
            ((2,), {"label": "x"}),
        ]

        class Discrete(SingletonGate, additional_singletons=additional_inputs):
            def __init__(self, n=0, label=None):
                super().__init__("discrete", 1, [], label=label)
                self.n = n

            @staticmethod
            def _singleton_lookup_key(n=0, label=None):
                return (n, label)

        default = Discrete()
        one = Discrete(1)
        two = Discrete(2, "x")
        mutable = Discrete(3)

        self.assertIsNot(default, default.to_mutable())
        self.assertEqual(default.n, default.to_mutable().n)
        self.assertIsNot(one, one.to_mutable())
        self.assertEqual(one.n, one.to_mutable().n)
        self.assertIsNot(two, two.to_mutable())
        self.assertEqual(two.n, two.to_mutable().n)
        self.assertIsNot(mutable, mutable.to_mutable())
        self.assertEqual(mutable.n, mutable.to_mutable().n)

        # The equality assertions in the middle are sanity checks that nothing got overwritten.

        self.assertIs(default, copy.copy(default))
        self.assertEqual(default.n, 0)
        self.assertIs(one, copy.copy(one))
        self.assertEqual(one.n, 1)
        self.assertIs(two, copy.copy(two))
        self.assertEqual(two.n, 2)
        self.assertIsNot(mutable, copy.copy(mutable))

        self.assertIs(default, copy.deepcopy(default))
        self.assertEqual(default.n, 0)
        self.assertIs(one, copy.deepcopy(one))
        self.assertEqual(one.n, 1)
        self.assertIs(two, copy.deepcopy(two))
        self.assertEqual(two.n, 2)
        self.assertIsNot(mutable, copy.deepcopy(mutable))

    def test_additional_singletons_pickle(self):
        additional_inputs = [
            ((1,), {}),
            ((2,), {"label": "x"}),
        ]

        class Discrete(SingletonGate, additional_singletons=additional_inputs):
            def __init__(self, n=0, label=None):
                super().__init__("discrete", 1, [], label=label)
                self.n = n

            @staticmethod
            def _singleton_lookup_key(n=0, label=None):
                return (n, label)

        # Pickle needs the class to be importable.  We want the class to only be instantiated inside
        # the test, which means we need a little magic to make it pretend-importable.
        dummy_module = types.ModuleType("_QISKIT_DUMMY_" + str(uuid.uuid4()).replace("-", "_"))
        dummy_module.Discrete = Discrete
        Discrete.__module__ = dummy_module.__name__
        Discrete.__qualname__ = Discrete.__name__

        default = Discrete()
        one = Discrete(1)
        two = Discrete(2, "x")
        mutable = Discrete(3)

        with unittest.mock.patch.dict(sys.modules, {dummy_module.__name__: dummy_module}):
            # The singletons in `additional_singletons` are static; their lifetimes should be tied
            # to the type object itself, so if we don't delete it, it should be eligible to be
            # reloaded from and produce the exact instances.
            self.assertIs(default, pickle.loads(pickle.dumps(default)))
            self.assertEqual(default.n, 0)
            self.assertIs(one, pickle.loads(pickle.dumps(one)))
            self.assertEqual(one.n, 1)
            self.assertIs(two, pickle.loads(pickle.dumps(two)))
            self.assertEqual(two.n, 2)
            self.assertIsNot(mutable, pickle.loads(pickle.dumps(mutable)))


class TestSingletonControlledGate(QiskitTestCase):
    """Qiskit SingletonGate tests."""

    def test_default_singleton(self):
        gate = CXGate()
        new_gate = CXGate()
        self.assertIs(gate, new_gate)

    def test_label_not_singleton(self):
        gate = CXGate()
        label_gate = CXGate(label="special")
        self.assertIsNot(gate, label_gate)

    def test_raise_on_state_mutation(self):
        gate = CSwapGate()
        with self.assertRaises(TypeError):
            gate.label = "foo"

    def test_default_singleton_copy(self):
        gate = CXGate()
        copied = gate.copy()
        self.assertIs(gate, copied)

    def test_label_copy(self):
        gate = CZGate(label="special")
        copied = gate.copy()
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)

    def test_label_copy_new(self):
        gate = CZGate()
        label_gate = CZGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = gate.copy()
        copied_label = label_gate.copy()
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_deepcopy(self):
        gate = CXGate()
        copied = copy.deepcopy(gate)
        self.assertIs(gate, copied)

    def test_deepcopy_with_label(self):
        singleton_gate = CXGate()
        gate = CXGate(label="special")
        copied = copy.deepcopy(gate)
        self.assertIsNot(gate, copied)
        self.assertEqual(gate, copied)
        self.assertEqual(copied.label, "special")
        self.assertTrue(copied.mutable)
        self.assertIsNot(copied, singleton_gate)
        self.assertEqual(singleton_gate, copied)
        self.assertNotEqual(singleton_gate.label, copied.label)

    def test_label_deepcopy_new(self):
        gate = CHGate()
        label_gate = CHGate(label="special")
        self.assertIsNot(gate, label_gate)
        self.assertNotEqual(gate.label, label_gate.label)
        copied = copy.deepcopy(gate)
        copied_label = copy.deepcopy(label_gate)
        self.assertIs(gate, copied)
        self.assertIsNot(copied, label_gate)
        self.assertIsNot(copied_label, gate)
        self.assertIsNot(copied_label, label_gate)
        self.assertNotEqual(copied.label, label_gate.label)
        self.assertEqual(copied_label, label_gate)
        self.assertNotEqual(copied.label, "special")
        self.assertEqual(copied_label.label, "special")

    def test_control_a_singleton(self):
        singleton_gate = CHGate()
        gate = CHGate(label="special")
        ch = gate.control(annotated=False, label="my_ch")
        self.assertEqual(ch.base_gate.label, "special")
        self.assertIsNot(ch.base_gate, singleton_gate)

    def test_round_trip_dag_conversion(self):
        qc = QuantumCircuit(2)
        gate = CHGate()
        qc.append(gate, [0, 1])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIs(qc.data[0].operation, out.data[0].operation)

    def test_round_trip_dag_conversion_with_label(self):
        gate = CHGate(label="special")
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])
        dag = circuit_to_dag(qc)
        out = dag_to_circuit(dag)
        self.assertIsNot(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(qc.data[0].operation, out.data[0].operation)
        self.assertEqual(out.data[0].operation.label, "special")

    def test_is_mutable(self):
        gate = CXGate()
        self.assertFalse(gate.mutable)
        label_gate = CXGate(label="foo")
        self.assertTrue(label_gate.mutable)
        self.assertIsNot(gate, label_gate)

    def test_to_mutable(self):
        gate = CXGate()
        self.assertFalse(gate.mutable)
        new_gate = gate.to_mutable()
        self.assertTrue(new_gate.mutable)
        self.assertIsNot(gate, new_gate)

    def test_to_mutable_setter(self):
        gate = CZGate()
        self.assertFalse(gate.mutable)
        mutable_gate = gate.to_mutable()
        mutable_gate.label = "foo"
        self.assertTrue(mutable_gate.mutable)
        self.assertIsNot(gate, mutable_gate)
        self.assertEqual(mutable_gate.label, "foo")

    def test_to_mutable_of_mutable_instance(self):
        gate = CZGate(label="foo")
        mutable_copy = gate.to_mutable()
        self.assertIsNot(gate, mutable_copy)
        self.assertEqual(mutable_copy.label, gate.label)
        mutable_copy.label = "not foo"
        self.assertNotEqual(mutable_copy.label, gate.label)

    def test_inner_gate_label(self):
        inner_gate = HGate(label="my h gate")
        controlled_gate = inner_gate.control(annotated=False)
        self.assertTrue(controlled_gate.mutable)
        self.assertEqual("my h gate", controlled_gate.base_gate.label)

    def test_inner_gate_label_outer_label_too(self):
        inner_gate = HGate(label="my h gate")
        controlled_gate = inner_gate.control(annotated=False, label="foo")
        self.assertTrue(controlled_gate.mutable)
        self.assertEqual("my h gate", controlled_gate.base_gate.label)
        self.assertEqual("foo", controlled_gate.label)

    def test_inner_outer_label_pickle(self):
        inner_gate = XGate(label="my h gate")
        controlled_gate = inner_gate.control(annotated=False, label="foo")
        self.assertTrue(controlled_gate.mutable)
        self.assertEqual("my h gate", controlled_gate.base_gate.label)
        self.assertEqual("foo", controlled_gate.label)
        with io.BytesIO() as fd:
            pickle.dump(controlled_gate, fd)
            fd.seek(0)
            copied = pickle.load(fd)
        self.assertIsNot(controlled_gate, copied)
        self.assertTrue(copied.mutable)
        self.assertEqual("my h gate", copied.base_gate.label)
        self.assertEqual("foo", copied.label)

    def test_singleton_with_defaults(self):
        self.assertIs(CXGate(), CXGate(label=None))
        self.assertIs(CXGate(), CXGate(_base_label=None))
        self.assertIs(CXGate(), CXGate(label=None, ctrl_state=None))

    def test_singleton_with_equivalent_ctrl_state(self):
        self.assertIs(CXGate(), CXGate(ctrl_state=None))
        self.assertIs(CXGate(), CXGate(ctrl_state=1))
        self.assertIs(CXGate(), CXGate(label=None, ctrl_state=1))
        self.assertIs(CXGate(), CXGate(ctrl_state="1"))
        self.assertIsNot(CXGate(), CXGate(ctrl_state=0))
        self.assertIsNot(CXGate(), CXGate(ctrl_state="0"))

        self.assertIs(C4XGate(), C4XGate(ctrl_state=None))
        self.assertIs(C4XGate(), C4XGate(ctrl_state=15))
        self.assertIs(C4XGate(), C4XGate(ctrl_state="1111"))
        self.assertIsNot(C4XGate(), C4XGate(ctrl_state=0))
