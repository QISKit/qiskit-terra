# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test unitary overlap function"""
import unittest
from ddt import ddt, data
import numpy as np

from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit.circuit.library import EfficientSU2, UnitaryOverlap, unitary_overlap
from qiskit.quantum_info import Statevector
from qiskit.circuit.exceptions import CircuitError
from test import QiskitTestCase


@ddt
class TestUnitaryOverlap(QiskitTestCase):
    """Test the unitary overlap circuit class."""

    @data(True, False)
    def test_identity(self, use_function):
        """Test identity is returned"""
        with self.assertWarns(DeprecationWarning):
            unitary = EfficientSU2(2)
        unitary.assign_parameters(np.random.random(size=unitary.num_parameters), inplace=True)
        if use_function:
            overlap = unitary_overlap(unitary, unitary)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary, unitary)
        self.assertLess(abs(Statevector.from_instruction(overlap)[0] - 1), 1e-12)

    @data(True, False)
    def test_parameterized_identity(self, use_function):
        """Test identity is returned"""
        with self.assertWarns(DeprecationWarning):
            unitary = EfficientSU2(2)

        if use_function:
            overlap = unitary_overlap(unitary, unitary)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary, unitary)

        rands = np.random.random(size=unitary.num_parameters)
        double_rands = np.hstack((rands, rands))
        overlap.assign_parameters(double_rands, inplace=True)
        self.assertLess(abs(Statevector.from_instruction(overlap)[0] - 1), 1e-12)

    @data(True, False)
    def test_two_parameterized_inputs(self, use_function):
        """Test two parameterized inputs"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(2)

        if use_function:
            overlap = unitary_overlap(unitary1, unitary2)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2)
        self.assertEqual(overlap.num_parameters, unitary1.num_parameters + unitary2.num_parameters)

    @data(True, False)
    def test_parameter_prefixes(self, use_function):
        """Test two parameterized inputs"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(2)

        if use_function:
            overlap = unitary_overlap(unitary1, unitary2, prefix1="a", prefix2="b")
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2, prefix1="a", prefix2="b")

        self.assertEqual(overlap.num_parameters, unitary1.num_parameters + unitary2.num_parameters)

        expected_names = [f"a[{i}]" for i in range(unitary1.num_parameters)]
        expected_names += [f"b[{i}]" for i in range(unitary2.num_parameters)]

        self.assertListEqual([p.name for p in overlap.parameters], expected_names)

    @data(True, False)
    def test_partial_parameterized_inputs(self, use_function):
        """Test one parameterized inputs (1)"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        unitary1.assign_parameters(np.random.random(size=unitary1.num_parameters), inplace=True)

        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(2, reps=5)

        if use_function:
            overlap = unitary_overlap(unitary1, unitary2)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2)

        self.assertEqual(overlap.num_parameters, unitary2.num_parameters)

    @data(True, False)
    def test_partial_parameterized_inputs2(self, use_function):
        """Test one parameterized inputs (2)"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(2, reps=5)
        unitary2.assign_parameters(np.random.random(size=unitary2.num_parameters), inplace=True)

        if use_function:
            overlap = unitary_overlap(unitary1, unitary2)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2)

        self.assertEqual(overlap.num_parameters, unitary1.num_parameters)

    @data(True, False)
    def test_barrier(self, use_function):
        """Test that barriers on input circuits are well handled"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(1, reps=0)
        unitary1.barrier()
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(1, reps=1)
        unitary2.barrier()
        if use_function:
            overlap = unitary_overlap(unitary1, unitary2)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2)
        self.assertEqual(overlap.num_parameters, unitary1.num_parameters + unitary2.num_parameters)

    @data(True, False)
    def test_measurements(self, use_function):
        """Test that exception is thrown for measurements"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        unitary1.measure_all()
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(2)

        with self.assertRaises(CircuitError):
            if use_function:
                _ = unitary_overlap(unitary1, unitary2)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = UnitaryOverlap(unitary1, unitary2)

    @data(True, False)
    def test_rest(self, use_function):
        """Test that exception is thrown for rest"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(1, reps=0)
        unitary1.reset(0)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(1, reps=1)

        with self.assertRaises(CircuitError):
            if use_function:
                _ = unitary_overlap(unitary1, unitary2)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = UnitaryOverlap(unitary1, unitary2)

    @data(True, False)
    def test_controlflow(self, use_function):
        """Test that exception is thrown for controlflow"""
        bit = Clbit()
        unitary1 = QuantumCircuit([Qubit(), bit])
        unitary1.h(0)
        with unitary1.if_test((bit, 0)):
            unitary1.x(0)

        unitary2 = QuantumCircuit(1)
        unitary2.rx(0.2, 0)

        with self.assertRaises(CircuitError):
            if use_function:
                _ = unitary_overlap(unitary1, unitary2)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = UnitaryOverlap(unitary1, unitary2)

    @data(True, False)
    def test_mismatching_qubits(self, use_function):
        """Test that exception is thrown for mismatching circuit"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(2)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(1)

        with self.assertRaises(CircuitError):
            if use_function:
                _ = unitary_overlap(unitary1, unitary2)
            else:
                with self.assertWarns(DeprecationWarning):
                    _ = UnitaryOverlap(unitary1, unitary2)

    @data(True, False)
    def test_insert_barrier(self, use_function):
        """Test inserting barrier between circuits"""
        with self.assertWarns(DeprecationWarning):
            unitary1 = EfficientSU2(1, reps=1)
        with self.assertWarns(DeprecationWarning):
            unitary2 = EfficientSU2(1, reps=1)

        if use_function:
            overlap = unitary_overlap(unitary1, unitary2, insert_barrier=True)
        else:
            with self.assertWarns(DeprecationWarning):
                overlap = UnitaryOverlap(unitary1, unitary2, insert_barrier=True)

        self.assertEqual(overlap.count_ops()["barrier"], 1)
        self.assertEqual(
            str(overlap.draw(fold=-1, output="text")).strip(),
            """
   ┌───────────────────────────────────────┐ ░ ┌──────────────────────────────────────────┐
q: ┤ EfficientSU2(p1[0],p1[1],p1[2],p1[3]) ├─░─┤ EfficientSU2_dg(p2[0],p2[1],p2[2],p2[3]) ├
   └───────────────────────────────────────┘ ░ └──────────────────────────────────────────┘
""".strip(),
        )


if __name__ == "__main__":
    unittest.main()
