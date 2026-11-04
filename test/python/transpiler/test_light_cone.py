# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the LightCone pass"""

import unittest
from test import QiskitTestCase

import ddt

from qiskit.circuit import (
    Parameter,
    QuantumCircuit,
)
from qiskit.circuit.library import real_amplitudes
from qiskit.circuit.library.n_local.efficient_su2 import efficient_su2
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp, SparseObservable
from qiskit.transpiler.passes.optimization.light_cone import LightCone
from qiskit.transpiler.passmanager import PassManager


@ddt.ddt
class TestLightConePass(QiskitTestCase):
    """Test the LightCone pass."""

    @ddt.data("X", "Z")
    def test_nonparameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.s(1)
        qc.cy(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_nonparameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli."""
        light_cone = LightCone(bit_terms="Y", indices=[1])
        pm = PassManager([light_cone])

        qc = QuantumCircuit(2)
        qc.x(0)
        qc.s(1)
        qc.cy(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.s(1)

        self.assertEqual(expected, new_circuit)

    @ddt.data("Y", "Z")
    def test_parameterized_noncommuting(self, pauli_label):
        """Test for a non-commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        bit_terms = pauli_label
        light_cone = LightCone(bit_terms=bit_terms, indices=[1])
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = qc

        self.assertEqual(expected, new_circuit)

    def test_parameterized_commuting(self):
        """Test for a commuting, asymmetric, weight-one Pauli on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("XI").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rx(theta, 0)
        qc.ry(theta, 1)
        qc.cx(0, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.ry(theta, 1)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_symmetric(self):
        """Test for a double symmetric `Z` observable on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("ZIIZ").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = real_amplitudes(4, entanglement="pairwise", reps=1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.ry(theta[4], 0)
        expected.ry(theta[7], 3)

        self.assertEqual(expected, new_circuit)

    def test_parameterized_asymmetric(self):
        """Test for a double asymmetric observable on a parameterized circuit."""
        bit_terms, indices, _ = SparsePauliOp("IZIX").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = efficient_su2(4, entanglement="circular", reps=1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        theta = qc.parameters
        expected.ry(theta[0], 0)
        expected.ry(theta[1], 1)
        expected.ry(theta[2], 2)
        expected.ry(theta[3], 3)
        expected.rz(theta[4], 0)
        expected.rz(theta[5], 1)
        expected.rz(theta[6], 2)
        expected.rz(theta[7], 3)
        expected.cx(3, 0)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.cx(2, 3)
        expected.ry(theta[8], 0)
        expected.ry(theta[10], 2)
        expected.rz(theta[14], 2)

        self.assertEqual(expected, new_circuit)

    def test_all_commuting(self):
        """Test for a circuit that fully commutes with an observable."""
        bit_terms, indices, _ = SparsePauliOp("IIIZ").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(4)
        qc.s(0)
        qc.z(0)
        qc.h(2)
        qc.cx(0, 1)
        qc.cx(2, 3)

        new_circuit = pm.run(qc)

        self.assertEqual(sum(new_circuit.count_ops().values()), 0)

    def test_commuting_block(self):
        """Test for a commuting block. Currently, gates are checked
        one by one and commuting blocks are thus ignored."""
        bit_terms, indices, _ = SparsePauliOp("IIXII").to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(5)
        qc.cx(2, 1)
        qc.cx(3, 4)
        qc.cx(1, 0)
        qc.cx(2, 3)
        qc.cx(2, 1)
        qc.x(1)
        qc.cx(2, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(5)
        expected.cx(2, 1)
        expected.cx(3, 4)
        expected.cx(1, 0)
        expected.cx(2, 3)
        expected.cx(2, 1)
        expected.cx(2, 1)

        self.assertEqual(expected, new_circuit)

    def test_measurement_barriers(self):
        """Test for measurement and barriers."""
        light_cone = LightCone()
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2, 1)
        qc.rz(theta, 1)
        qc.ry(theta, 0)
        qc.barrier()
        qc.cx(0, 1)
        qc.barrier()
        qc.rz(theta, 1)
        qc.measure(0, 0)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2, 1)
        expected.rz(theta, 1)
        expected.ry(theta, 0)
        expected.barrier()
        expected.cx(0, 1)
        expected.barrier()
        expected.measure(0, 0)

        self.assertEqual(expected, new_circuit)

    @ddt.data(SparsePauliOp("IX"), SparseObservable("I+"))
    def test_parameter_expression(self, sparse_object):
        """Test for Parameter expressions."""
        bit_terms, indices, _ = sparse_object.to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(2)
        qc.rz(theta + 2, 1)
        qc.ry(theta - 2, 0)
        qc.h(1)
        qc.cz(0, 1)
        qc.h(1)
        qc.rz(theta * 2, 1)
        qc.rz(theta / 2, 1)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(2)
        expected.rz(theta + 2, 1)
        expected.ry(theta - 2, 0)
        expected.h(1)
        expected.cz(0, 1)

        self.assertEqual(expected, new_circuit)

    @ddt.data(
        SparsePauliOp.from_sparse_list([("XY", [119, 2], 1)], 120),
        SparseObservable.from_sparse_list([("-l", [119, 2], 1)], 120),
    )
    def test_big_circuit(self, sparse_object):
        """Test for large circuit and observable."""
        num_qubits = sparse_object.num_qubits
        bit_terms, indices, _ = sparse_object.to_sparse_list()[0]
        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])
        theta = Parameter("θ")

        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            qc.h(i)
        for i in range(0, num_qubits, 2):
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(num_qubits):
            qc.rz(theta, i)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(num_qubits)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(num_qubits - 2)
        expected.h(num_qubits - 1)
        expected.cx(0, 1)
        expected.cx(2, 3)
        expected.cx(1, 2)
        expected.rz(theta, 2)
        expected.cx(num_qubits - 2, num_qubits - 1)
        expected.rz(theta, num_qubits - 1)

        self.assertEqual(expected, new_circuit)

    @ddt.data("IIIIIZXIIIIIIII", "YYYYYZXYYYYYYYY")
    def test_large_observable(self, pauli_string):
        """Test for a large initial observable."""
        op = SparsePauliOp([pauli_string])
        bit_terms, indices, _ = op.to_sparse_list()[0]

        light_cone = LightCone(bit_terms=bit_terms, indices=indices)
        pm = PassManager([light_cone])

        qc = QuantumCircuit(15)
        qc.cx(5, 6)

        new_circuit = pm.run(qc)

        expected = QuantumCircuit(15)

        self.assertEqual(expected, new_circuit)

    def test_raise_error_when_indices_is_empty(self):
        """Test that `ValueError` is raised if bit_terms is given but indices is empty."""
        with self.assertRaises(
            ValueError, msg="`indices` must be non-empty when providing `bit_terms`."
        ):
            _ = LightCone(bit_terms="X", indices=[])

    def test_raise_error_for_invalid_bit_terms_characters(self):
        """Test that `ValueError` is raised if `bit_terms` has characters not in `valid_characters`."""
        with self.assertRaises(
            ValueError, msg="`bit_terms` should contain only characters in {...}."
        ):
            _ = LightCone(bit_terms="AX", indices=[0])  # 'A' is invalid

    def test_raise_error_when_indices_out_of_range(self):
        """Test that `ValueError` is raised if an index is outside the DAG qubit range."""
        qc = QuantumCircuit(1)
        dag = circuit_to_dag(qc)

        light_cone = LightCone(bit_terms="X", indices=[1])  # index 1 doesn't exist
        with self.assertRaises(
            ValueError, msg="`indices` contains values outside the qubit range."
        ):
            light_cone.run(dag)

    def test_raise_error_when_circuit_measurements_and_observable_present(self):
        """Test that `ValueError` is raised if the circuit has measurements
        and `bit_terms` is also given."""
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)  # A measurement on qubit 0
        dag = circuit_to_dag(qc)

        light_cone = LightCone(bit_terms="X", indices=[0])
        with self.assertRaises(
            ValueError, msg="The circuit contains measurements and an observable has been given"
        ):
            light_cone.run(dag)


if __name__ == "__main__":
    unittest.main()
