# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test serializing ParameterExpressions from qpy."""

import io
from test import QiskitTestCase
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit import qpy
from qiskit.quantum_info import SparseObservable
from qiskit.quantum_info.operators import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate


class TestQpySerializeParameterExpression(QiskitTestCase):
    """QPY serializing ParameterExpression"""

    def test_roundtrip_equal(self):
        """Test serialize deserialize with ParameterExpression in _qpy_replay"""
        a = Parameter("a")
        b = Parameter("b")
        a1 = a * 2
        a2 = a1.subs({a: 3 * b})

        qc = QuantumCircuit(1)
        qc.rz(a2, 0)

        use_symengine = True
        version = 13
        with io.BytesIO() as container:
            qpy.dump(qc, container, version=version, use_symengine=use_symengine)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(qc, qc_from_qpy)

    def test_roundtrip_assigned_values(self):
        """Test that an expression which was assigned a value is correctly loaded."""
        a = Parameter("a")
        b = Parameter("b")
        c = Parameter("c")

        # lhs test
        circuit = QuantumCircuit(1)
        circuit.rz(a.sin() * b, 0)
        circuit = circuit.assign_parameters({a: 3})

        with io.BytesIO() as container:
            qpy.dump(circuit, container)
            container.seek(0)
            loaded = qpy.load(container)[0]

        self.assertEqual(circuit.parameters, loaded.parameters)
        self.assertAlmostEqual(
            float(circuit.data[0].operation.params[0].bind({b: 2})),
            float(loaded.data[0].operation.params[0].bind({b: 2})),
        )

        # rhs test
        circuit = QuantumCircuit(1)
        circuit.rz(b * a.sin(), 0)
        circuit = circuit.assign_parameters({a: 3})

        with io.BytesIO() as container:
            qpy.dump(circuit, container)
            container.seek(0)
            loaded = qpy.load(container)[0]

        self.assertEqual(circuit.parameters, loaded.parameters)
        self.assertAlmostEqual(
            float(circuit.data[0].operation.params[0].bind({b: 2})),
            float(loaded.data[0].operation.params[0].bind({b: 2})),
        )

        # both sides test
        circuit = QuantumCircuit(1)
        circuit.rz(c * b.cos() * a.sin(), 0)
        circuit = circuit.assign_parameters({a: 3, b: 5})

        with io.BytesIO() as container:
            qpy.dump(circuit, container)
            container.seek(0)
            loaded = qpy.load(container)[0]

        self.assertEqual(circuit.parameters, loaded.parameters)
        self.assertAlmostEqual(
            float(circuit.data[0].operation.params[0].bind({c: 7})),
            float(loaded.data[0].operation.params[0].bind({c: 7})),
        )


class TestPauliEvolution(QiskitTestCase):
    """QPY serializing PauliEvolutionGate with SparseObservable and SparsePauliOp"""

    def test_pauli_evolution_sparseobservable(self):
        """Test PauliEvolutionGate with SparseObservable"""
        op = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])

        # build the evolution gate
        evo = PauliEvolutionGate(op)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 17

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)

    def test_pauli_evolution_sparse_pauliop(self):
        """Test PauliEvolutionGate with SparsePauliOp"""
        operator = SparsePauliOp.from_list([("ZZ", 1), ("XI", -0.1)])

        # build the evolution gate
        evo = PauliEvolutionGate(operator, time=0.2)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 16

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)

    def test_pauli_evolution_operator_list(self):
        """Test PauliEvolutionGate with list of operators"""
        op1 = SparseObservable.from_list([("XIX", 0.1), ("ZIZ", 0.3)])
        op2 = SparsePauliOp.from_list([("ZZI", 1), ("XIX", -0.1)])

        # build the evolution gate
        evo = PauliEvolutionGate([op1, op2], time=0.5)
        circuit = QuantumCircuit(evo.num_qubits)
        circuit.append(evo, circuit.qubits)
        version = 17

        with io.BytesIO() as container:
            qpy.dump(circuit, container, version=version)
            qc_qpy_str = container.getvalue()

        with io.BytesIO(qc_qpy_str) as container:
            qc_from_qpy = qpy.load(container)[0]

        self.assertEqual(circuit, qc_from_qpy)
