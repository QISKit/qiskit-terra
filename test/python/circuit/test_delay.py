# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test delay instruction for quantum circuits."""
import copy
import pickle

import numpy as np

from qiskit.circuit import Delay, Duration
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.classical import expr
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


class TestDelayClass(QiskitTestCase):
    """Test delay instruction for quantum circuits."""

    def test_keep_units_after_adding_delays_to_circuit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(100, 0)
        qc.delay(200, 0, unit="s")
        qc.delay(300, 0, unit="ns")
        qc.delay(400, 0, unit="dt")
        self.assertEqual(qc.data[1].operation.unit, "dt")
        self.assertEqual(qc.data[2].operation.unit, "s")
        self.assertEqual(qc.data[3].operation.unit, "ns")
        self.assertEqual(qc.data[4].operation.unit, "dt")

    def test_fail_if_non_integer_duration_with_dt_unit_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(0.5, 0, unit="dt")

    def test_fail_if_unknown_unit_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(100, 0, unit="my_unit")

    def test_fail_if_negative_duration_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(-1, 0, unit="dt")
        with self.assertRaises(CircuitError):
            qc.delay(-1.5, 0, unit="s")

    def test_add_delay_on_single_qubit_to_circuit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(100, 0)
        qc.delay(200, [0])
        qc.delay(300, qc.qubits[0])
        self.assertEqual(qc.data[1], CircuitInstruction(Delay(duration=100), qc.qubits, []))
        self.assertEqual(qc.data[2], CircuitInstruction(Delay(duration=200), qc.qubits, []))
        self.assertEqual(qc.data[3], CircuitInstruction(Delay(duration=300), qc.qubits, []))

    def test_add_delay_on_multiple_qubits_to_circuit(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.delay(100)
        qc.delay(200, range(2))
        qc.delay(300, qc.qubits[1:])

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.delay(100, 0)
        expected.delay(100, 1)
        expected.delay(100, 2)
        expected.delay(200, 0)
        expected.delay(200, 1)
        expected.delay(300, 1)
        expected.delay(300, 2)
        self.assertEqual(qc, expected)

    def test_delay_set(self):
        """Test that a set argument to `delay` works."""
        qc = QuantumCircuit(5)
        qc.delay(8, {0, 1, 3, 4})
        expected = QuantumCircuit(5)
        expected.delay(8, 0)
        expected.delay(8, 1)
        expected.delay(8, 3)
        expected.delay(8, 4)
        self.assertEqual(qc, expected)

    def test_to_matrix_return_identity_matrix(self):
        actual = np.array(Delay(100))
        expected = np.array([[1, 0], [0, 1]], dtype=complex)
        self.assertTrue(np.array_equal(actual, expected))

    def test_equality(self):
        # At the time `__eq__` was specialised for `Delay`, the class was undergoing changes and
        # moving to Rust, so we didn't also modify the Python-space semantics to declare equality
        # between (say) 1000ms and 1s.  We could revisit that decision once the data model settles.
        #
        # This test then deliberately doesn't assert about mixed-scale comparisons, only comparisons
        # between the same units, and 'dt' to absolute times.
        def circuit_from(delay):
            out = QuantumCircuit(1)
            out.append(delay, [0], [])
            return out

        a = Parameter("a")
        left_instructions, right_instructions = [], []
        left_circuits, right_circuits = [], []
        for unit in ("s", "ms", "us", "ns", "ps", "dt"):
            for base in (left_instructions, right_instructions):
                base.append(Delay(1, unit))
                base.append(Delay(5.0, unit))
                base.append(Delay(a, unit))
            for base in (left_circuits, right_circuits):
                base.append(circuit_from(Delay(1, unit)))
                base.append(circuit_from(Delay(5.0, unit)))
                base.append(circuit_from(Delay(a, unit)))
        self.assertEqual(left_instructions, right_instructions)
        self.assertEqual(left_circuits, right_circuits)

        # We can't do all the non-equal tests in a single list comparison, since any single list
        # failure would mask any spurious successes.
        for unit in ("s", "ms", "us", "ns", "ps"):
            self.assertNotEqual(Delay(2, unit), Delay(2, "dt"))
            self.assertNotEqual(circuit_from(Delay(2, unit)), circuit_from(Delay(2, "dt")))
            self.assertNotEqual(Delay(2, "dt"), Delay(2, unit))
            self.assertNotEqual(circuit_from(Delay(2, "dt")), circuit_from(Delay(2, unit)))
            self.assertNotEqual(Delay(a, unit), Delay(a, "dt"))
            self.assertNotEqual(circuit_from(Delay(a, unit)), circuit_from(Delay(a, "dt")))

    def test_delay_clone(self):
        """Test that circuits with delays can be copied or pickled."""
        qc = QuantumCircuit(3)
        stretch = qc.add_stretch("a")
        qc.delay(100, qc.qubits[0])
        qc.delay(expr.lift(Duration.us(1)), 0)
        qc.delay(expr.lift(Duration.ns(2)), 0)
        qc.delay(expr.lift(Duration.ms(3)), 0)
        qc.delay(expr.lift(Duration.s(4)), 0)
        qc.delay(expr.lift(Duration.dt(5)), 0)
        qc.delay(expr.lift(Duration.ps(6)), 0)
        qc.delay(stretch, [0, 1])
        self.assertEqual(qc, pickle.loads(pickle.dumps(qc)))
        self.assertEqual(qc, copy.copy(qc))
        self.assertEqual(qc, copy.deepcopy(qc))


class TestParameterizedDelay(QiskitTestCase):
    """Test delay instruction with parameterized duration."""

    def test_can_accept_parameterized_duration(self):
        dur = Parameter("t")
        self.assertEqual(Delay(dur).duration, dur)

    def test_can_append_parameterized_delay(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur)
        self.assertEqual(qc.data[0].operation, Delay(dur))

    def test_can_assign_duration_parameter(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur)
        qc.assign_parameters({dur: 100}, inplace=True)
        self.assertEqual(qc.data[0].operation.duration, 100)

    def test_fail_if_assign_invalid_duration_parameters(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur, unit="dt")
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: 1 + 1j}, inplace=True)
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: 0.5}, inplace=True)
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: -1}, inplace=True)

    def test_can_assign_multiple_duration_parameters(self):
        durs = ParameterVector("dur", 2)
        qc = QuantumCircuit(2)
        qc.delay(durs[0])
        qc.delay(durs[1])
        qc = qc.assign_parameters([0, 1])
        durations = [inst.operation.duration for inst in qc.data]
        self.assertEqual(durations, [0, 0, 1, 1])
