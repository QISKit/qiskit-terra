# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rust gate definition tests"""

from functools import partial
from math import pi

from test import QiskitTestCase

import numpy as np

from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.library.standard_gates import C3XGate, CU1Gate, CZGate, CCZGate
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.quantum_info import Operator

CUSTOM_NAME_MAPPING = {"mcx": C3XGate()}


class TestRustGateEquivalence(QiskitTestCase):
    """Tests that compile time rust gate definitions is correct."""

    def setUp(self):
        super().setUp()
        self.standard_gates = get_standard_gate_name_mapping()
        self.standard_gates.update(CUSTOM_NAME_MAPPING)
        # Pre-warm gate mapping cache, this is needed so rust -> py conversion is done
        qc = QuantumCircuit(5)
        for gate in self.standard_gates.values():
            if getattr(gate, "_standard_gate", None):
                if gate.params:
                    gate = gate.base_class(*[pi] * len(gate.params))
                qc.append(gate, list(range(gate.num_qubits)))

    def test_gate_cross_domain_conversion(self):
        """Test the rust -> python conversion returns the right class."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # Gate not in rust yet or no constructor method
                continue
            with self.subTest(name=name):
                qc = QuantumCircuit(standard_gate.num_qubits)
                qc._append(
                    CircuitInstruction.from_standard(standard_gate, qc.qubits, gate_class.params)
                )
                self.assertEqual(qc.data[0].operation.base_class, gate_class.base_class)
                self.assertEqual(qc.data[0].operation, gate_class)

    def test_definitions(self):
        """Test definitions are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                params = [0.1 * (i + 1) for i in range(standard_gate._num_params())]
                py_def = gate_class.base_class(*params).definition
                rs_def = standard_gate._get_definition(params)
                if py_def is None:
                    self.assertIsNone(rs_def)
                else:
                    rs_def = QuantumCircuit._from_circuit_data(rs_def)
                    for rs_inst, py_inst in zip(rs_def._data, py_def._data):
                        self.assertEqual(py_inst.operation.name, rs_inst.operation.name)
                        self.assertEqual(rs_inst.operation.params, py_inst.operation.params)
                        self.assertEqual(
                            [py_def.find_bit(x).index for x in py_inst.qubits],
                            [rs_def.find_bit(x).index for x in rs_inst.qubits],
                        )

    def test_matrix(self):
        """Test matrices are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                params = [0.1 * (i + 1) for i in range(standard_gate._num_params())]
                py_def = gate_class.base_class(*params).to_matrix()
                rs_def = standard_gate._to_matrix(params)
                np.testing.assert_allclose(rs_def, py_def)

    def test_inverse(self):
        """Test that the inverse is the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                params = [0.1 * (i + 1) for i in range(standard_gate._num_params())]
                py_def = gate_class.base_class(*params).inverse()
                rs_def = standard_gate._inverse(params)
                if rs_def is not None:
                    self.assertEqual(py_def.name, rs_def[0].name)
                    np.testing.assert_allclose(py_def.params, rs_def[1])

    def test_name(self):
        """Test that the gate name properties match in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(gate_class.name, standard_gate.name)

    def test_num_qubits(self):
        """Test the number of qubits are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(gate_class.num_qubits, standard_gate.num_qubits)

    def test_num_params(self):
        """Test the number of parameters are the same in rust space."""
        for name, gate_class in self.standard_gates.items():
            standard_gate = getattr(gate_class, "_standard_gate", None)
            if standard_gate is None:
                # gate is not in rust yet
                continue

            with self.subTest(name=name):
                self.assertEqual(
                    len(gate_class.params), standard_gate.num_params, msg=f"{name} not equal"
                )

    def test_non_default_controls(self):
        """Test that controlled gates with a non-default ctrl_state
        are not using the standard rust representation."""
        # CZ and CU1 are diagonal matrices with one non-1 term
        # in the diagonal (see op_terms)
        gate_classes = [CZGate, partial(CU1Gate, 0.1)]
        op_terms = [-1, 0.99500417 + 0.09983342j]

        for gate_cls, term in zip(gate_classes, op_terms):
            with self.subTest(name="2q gates"):
                default_op = np.diag([1, 1, 1, term])
                non_default_op = np.diag([1, 1, term, 1])
                state_out_map = {
                    1: default_op,
                    "1": default_op,
                    None: default_op,
                    0: non_default_op,
                    "0": non_default_op,
                }
                for state, op in state_out_map.items():
                    circuit = QuantumCircuit(2)
                    gate = gate_cls(ctrl_state=state)
                    circuit.append(gate, [0, 1])
                    self.assertIsNotNone(getattr(gate, "_standard_gate", None))
                    np.testing.assert_almost_equal(circuit.data[0].operation.to_matrix(), op)

        with self.subTest(name="3q gate"):
            default_op = np.diag([1, 1, 1, 1, 1, 1, 1, -1])
            non_default_op_0 = np.diag([1, 1, 1, 1, -1, 1, 1, 1])
            non_default_op_1 = np.diag([1, 1, 1, 1, 1, -1, 1, 1])
            non_default_op_2 = np.diag([1, 1, 1, 1, 1, 1, -1, 1])
            state_out_map = {
                3: default_op,
                "11": default_op,
                None: default_op,
                0: non_default_op_0,
                1: non_default_op_1,
                "01": non_default_op_1,
                "10": non_default_op_2,
            }
            for state, op in state_out_map.items():
                circuit = QuantumCircuit(3)
                gate = CCZGate(ctrl_state=state)
                circuit.append(gate, [0, 1, 2])
                self.assertIsNotNone(getattr(gate, "_standard_gate", None))
                np.testing.assert_almost_equal(Operator(circuit.data[0].operation).to_matrix(), op)

    def test_extracted_as_standard_gate(self):
        """Test that every gate in the standard library gets correctly extracted as a Rust-space
        `StandardGate` in its default configuration when passed through `append`."""
        standards = set()
        qc = QuantumCircuit(4)
        for name, gate in get_standard_gate_name_mapping().items():
            if gate._standard_gate is None:
                # Not a standard gate.
                continue
            standards.add(name)
            qc.append(gate, qc.qubits[: gate.num_qubits], [])
        # Sanity check: the test should have found at least one standard gate in the mapping.
        self.assertNotEqual(standards, set())

        extracted = {inst.name for inst in qc.data if inst.is_standard_gate()}
        self.assertEqual(standards, extracted)
