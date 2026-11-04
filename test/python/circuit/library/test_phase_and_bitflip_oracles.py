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

"""Test the phase and bit-flip oracle circuits."""

import os
import unittest
import tempfile
from ddt import ddt, data, unpack
from numpy import sqrt, isclose

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PhaseOracle, PhaseOracleGate, BitFlipOracleGate
from qiskit.quantum_info import Statevector
from test import QiskitTestCase


@ddt
class TestPhaseOracleAndGate(QiskitTestCase):
    """Test phase oracle and phase oracle gate objects."""

    @data(
        ("x | x", "1", True),
        ("x & x", "0", False),
        ("(x0 & x1 | ~x2) ^ x4", "0110", False),
        ("xx & xxx | ( ~z ^ zz)", "0111", True),
    )
    @unpack
    def test_evaluate_bitstring(self, expression, input_bitstring, expected):
        """PhaseOracle(...).evaluate_bitstring"""
        with self.assertWarns(DeprecationWarning):
            oracle = PhaseOracle(expression)
        result = oracle.evaluate_bitstring(input_bitstring)
        self.assertEqual(result, expected)

    @data(
        ("x | x", "01"),
        ("~x", "10"),
        ("x & y", "0001"),
        ("x & ~y", "0100"),
        ("(x0 | ~x0) & (x1 | ~x1) & (x2 | ~x2)", "11111111"),
        ("(x0 & x1 | ~x2) ^ x4", "1111000100001110"),
        ("x & y ^ ( ~z1 | z2)", "1110000111101110"),
    )
    @unpack
    def test_statevector(self, expression, truth_table):
        """Circuit generation"""
        for use_gate in [True, False]:
            if use_gate:
                oracle = PhaseOracleGate(expression)
            else:
                with self.assertWarns(DeprecationWarning):
                    oracle = PhaseOracle(expression)
            num_qubits = oracle.num_qubits
            circuit = QuantumCircuit(num_qubits)
            circuit.h(range(num_qubits))
            circuit.compose(oracle, inplace=True)
            statevector = Statevector.from_instruction(circuit)

            valid_state = -1 / sqrt(2**num_qubits)
            invalid_state = 1 / sqrt(2**num_qubits)

            states = list(range(2**num_qubits))
            good_states = [i for i in range(len(states)) if truth_table[i] == "1"]
            expected_valid = [state in good_states for state in states]
            result_valid = [isclose(statevector.data[state], valid_state) for state in states]

            expected_invalid = [state not in good_states for state in states]
            result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
            with self.subTest(use_gate=use_gate):
                self.assertListEqual(expected_valid, result_valid)
                self.assertListEqual(expected_invalid, result_invalid)

    @data(
        ("((A & C) | (B & D)) & ~(C & D)", None, [3, 7, 12, 13]),
        ("((A & C) | (B & D)) & ~(C & D)", ["A", "B", "C", "D"], [5, 7, 10, 11]),
    )
    @unpack
    def test_variable_order(self, expression, var_order, good_states):
        """Circuit generation"""
        for use_gate in [True, False]:
            if use_gate:
                oracle = PhaseOracleGate(expression, var_order=var_order)
            else:
                with self.assertWarns(DeprecationWarning):
                    oracle = PhaseOracle(expression, var_order=var_order)
            num_qubits = oracle.num_qubits
            circuit = QuantumCircuit(num_qubits)
            circuit.h(range(num_qubits))
            circuit.compose(oracle, inplace=True)
            statevector = Statevector.from_instruction(circuit)

            valid_state = -1 / sqrt(2**num_qubits)
            invalid_state = 1 / sqrt(2**num_qubits)

            states = list(range(2**num_qubits))
            expected_valid = [state in good_states for state in states]
            result_valid = [isclose(statevector.data[state], valid_state) for state in states]

            expected_invalid = [state not in good_states for state in states]
            result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
            with self.subTest(use_gate=use_gate):
                self.assertListEqual(expected_valid, result_valid)
                self.assertListEqual(expected_invalid, result_invalid)

    def test_from_dimacs_file(self):
        """Initializing from DIMACS file"""
        input_3sat_instance = """
        c DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
        p cnf 3 5
        -1 -2 -3 0
        1 -2 3 0
        1 2 -3 0
        1 -2 -3 0
        -1 2 3 0
        """
        with tempfile.NamedTemporaryFile("wt", suffix=".dimacs", delete=False) as file:
            file.write(input_3sat_instance)
        self.addCleanup(os.remove, file.name)
        for use_gate in [True, False]:
            if use_gate:
                oracle = PhaseOracleGate.from_dimacs_file(file.name)
            else:
                with self.assertWarns(DeprecationWarning):
                    oracle = PhaseOracle.from_dimacs_file(file.name)
            self.assertEqual(oracle.num_qubits, 3)


@ddt
class TestBitFlipOracleGate(QiskitTestCase):
    """Test bit-flip oracle object."""

    @data(
        ("x | x", "01"),
        ("~x", "10"),
        ("x & y", "0001"),
        ("x & ~y", "0100"),
        ("(x0 | ~x0) & (x1 | ~x1) & (x2 | ~x2)", "11111111"),
        ("(x0 & x1 | ~x2) ^ x4", "1111000100001110"),
        ("x & y ^ ( ~z1 | z2)", "1110000111101110"),
    )
    @unpack
    def test_statevector(self, expression, truth_table):
        """Circuit generation"""
        oracle = BitFlipOracleGate(expression)
        num_qubits = oracle.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.h(
            range(num_qubits - 1)
        )  # we keep the result qubit 0 in all the superposition states
        circuit.compose(oracle, inplace=True)
        statevector = Statevector.from_instruction(circuit)
        truth_table_size = 2 ** (num_qubits - 1)
        valid_state = 1 / sqrt(truth_table_size)
        invalid_state = 0
        states = list(range(len(statevector)))
        good_states = [
            i + (truth_table_size if truth_table[i] == "1" else 0) for i in range(truth_table_size)
        ]
        expected_valid = [state in good_states for state in states]
        result_valid = [isclose(statevector.data[state], valid_state) for state in states]

        expected_invalid = [state not in good_states for state in states]
        result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
        self.assertListEqual(expected_valid, result_valid)
        self.assertListEqual(expected_invalid, result_invalid)

    @data(
        ("((A & C) | (B & D)) & ~(C & D)", None, "0001000100001100"),
        ("((A & C) | (B & D)) & ~(C & D)", ["A", "B", "C", "D"], "0000010100110000"),
    )
    @unpack
    def test_variable_order(self, expression, var_order, truth_table):
        """Circuit generation"""
        oracle = BitFlipOracleGate(expression, var_order=var_order)
        num_qubits = oracle.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.h(
            range(num_qubits - 1)
        )  # we keep the result qubit 0 in all the superposition states
        circuit.compose(oracle, inplace=True)
        statevector = Statevector.from_instruction(circuit)

        truth_table_size = 2 ** (num_qubits - 1)
        valid_state = 1 / sqrt(truth_table_size)
        invalid_state = 0

        states = list(range(2**num_qubits))
        good_states = [
            i + (truth_table_size if truth_table[i] == "1" else 0) for i in range(truth_table_size)
        ]
        expected_valid = [state in good_states for state in states]
        result_valid = [isclose(statevector.data[state], valid_state) for state in states]

        expected_invalid = [state not in good_states for state in states]
        result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
        self.assertListEqual(expected_valid, result_valid)
        self.assertListEqual(expected_invalid, result_invalid)

    def test_from_dimacs_file(self):
        """Initializing from DIMACS file"""
        input_3sat_instance = """
        c DIMACS CNF file with 3 satisfying assignments: 1 -2 3, -1 -2 -3, 1 2 -3.
        p cnf 3 5
        -1 -2 -3 0
        1 -2 3 0
        1 2 -3 0
        1 -2 -3 0
        -1 2 3 0
        """
        with tempfile.NamedTemporaryFile("wt", suffix=".dimacs", delete=False) as file:
            file.write(input_3sat_instance)
        self.addCleanup(os.remove, file.name)
        oracle = BitFlipOracleGate.from_dimacs_file(file.name)
        self.assertEqual(oracle.num_qubits, 4)


if __name__ == "__main__":
    unittest.main()
