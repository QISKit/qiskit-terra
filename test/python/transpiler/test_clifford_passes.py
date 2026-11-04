# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test transpiler passes and conversion methods that deal with Cliffords."""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import LinearFunction, PauliGate
from qiskit.converters import dag_to_circuit, circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.passes import HighLevelSynthesis
from qiskit.transpiler.passes import OptimizeCliffords, CollectCliffords
from qiskit.quantum_info.operators import Clifford
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator, random_clifford
from qiskit.compiler.transpiler import transpile
from test import QiskitTestCase

from qiskit.circuit.library import (
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    iSwapGate,
    RXGate,
    RYGate,
    RZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    SwapGate,
    UGate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
    YGate,
    ZGate,
)


class TestCliffordPasses(QiskitTestCase):
    """Tests to verify correctness of transpiler passes and
    conversion methods that deal with Cliffords."""

    def create_cliff1(self):
        """Creates a simple Clifford."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff2(self):
        """Creates another simple Clifford."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def create_cliff3(self):
        """Creates a third Clifford which is the composition of the previous two."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.s(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.cx(1, 2)
        qc.s(2)
        return Clifford(qc)

    def test_circuit_with_cliffords(self):
        """Test that Cliffords get stored natively on a QuantumCircuit,
        and that QuantumCircuit's decompose() replaces Clifford with gates."""

        # Create a circuit with 2 cliffords and four other gates
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(2, 0)
        qc.append(cliff1, [3, 0, 2])
        qc.swap(1, 3)
        qc.append(cliff2, [1, 2, 3])
        qc.h(3)

        # Check that there are indeed two Clifford objects in the circuit,
        # and that these are not gates.
        cliffords = [inst.operation for inst in qc.data if isinstance(inst.operation, Clifford)]
        gates = [inst.operation for inst in qc.data if isinstance(inst.operation, Gate)]
        self.assertEqual(len(cliffords), 2)
        self.assertEqual(len(gates), 4)

        # Check that calling QuantumCircuit's decompose(), no Clifford objects remain
        qc2 = qc.decompose()
        cliffords2 = [inst.operation for inst in qc2.data if isinstance(inst.operation, Clifford)]
        self.assertEqual(len(cliffords2), 0)

    def test_can_construct_operator(self):
        """Test that we can construct an Operator from a circuit that
        contains a Clifford gate."""

        cliff = self.create_cliff1()
        qc = QuantumCircuit(4)
        qc.append(cliff, [3, 1, 2])

        # Create an operator from the decomposition of qc into gates
        op1 = Operator(qc.decompose())

        # Create an operator from qc directly
        op2 = Operator(qc)

        # Check that the two operators are equal
        self.assertTrue(op1.equiv(op2))

    def test_can_combine_cliffords(self):
        """Test that we can combine a pair of Cliffords over the same qubits
        using OptimizeCliffords transpiler pass."""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        cliff3 = self.create_cliff3()

        # Create a circuit with two consecutive cliffords
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 1, 2])
        self.assertEqual(qc1.count_ops()["clifford"], 2)

        # Run OptimizeCliffords pass, and check that only one Clifford remains
        qc1opt = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1opt.count_ops()["clifford"], 1)

        # Create the expected circuit
        qc2 = QuantumCircuit(4)
        qc2.append(cliff3, [3, 1, 2])

        # Check that all possible operators are equal
        self.assertTrue(Operator(qc1).equiv(Operator(qc1.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc1opt.decompose())))
        self.assertTrue(Operator(qc1).equiv(Operator(qc1opt)))
        self.assertTrue(Operator(qc2).equiv(Operator(qc2.decompose())))
        self.assertTrue(Operator(qc1opt).equiv(Operator(qc2)))

    def test_cannot_combine(self):
        """Test that currently we cannot combine a pair of Cliffords.
        The result will be changed after pass is updated"""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        qc1 = QuantumCircuit(4)
        qc1.append(cliff1, [3, 1, 2])
        qc1.append(cliff2, [3, 2, 1])
        qc1 = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc1.count_ops()["clifford"], 2)

    def test_circuit_to_dag_conversion_and_back(self):
        """Test that converting a circuit containing Clifford to a DAG
        and back preserves the Clifford.
        """
        # Create a Clifford
        cliff_circ = QuantumCircuit(3)
        cliff_circ.cx(0, 1)
        cliff_circ.h(0)
        cliff_circ.s(1)
        cliff_circ.swap(1, 2)
        cliff = Clifford(cliff_circ)

        # Add this Clifford to a Quantum Circuit, and check that it remains a Clifford
        circ0 = QuantumCircuit(4)
        circ0.append(cliff, [0, 1, 2])
        circ0_cliffords = [
            inst.operation for inst in circ0.data if isinstance(inst.operation, Clifford)
        ]
        circ0_gates = [inst.operation for inst in circ0.data if isinstance(inst.operation, Gate)]
        self.assertEqual(len(circ0_cliffords), 1)
        self.assertEqual(len(circ0_gates), 0)

        # Check that converting circuit to DAG preserves Clifford.
        dag0 = circuit_to_dag(circ0)
        dag0_cliffords = [
            node
            for node in dag0.topological_nodes()
            if isinstance(node, DAGOpNode) and isinstance(node.op, Clifford)
        ]
        self.assertEqual(len(dag0_cliffords), 1)

        # Check that converted DAG to a circuit also preserves Clifford.
        circ1 = dag_to_circuit(dag0)
        circ1_cliffords = [
            inst.operation for inst in circ1.data if isinstance(inst.operation, Clifford)
        ]
        circ1_gates = [inst.operation for inst in circ1.data if isinstance(inst.operation, Gate)]
        self.assertEqual(len(circ1_cliffords), 1)
        self.assertEqual(len(circ1_gates), 0)

        # However, test that running an unrolling pass on the DAG replaces Clifford
        # by gates.
        dag1 = HighLevelSynthesis().run(dag0)
        dag1_cliffords = [
            node
            for node in dag1.topological_nodes()
            if isinstance(node, DAGOpNode) and isinstance(node.op, Clifford)
        ]
        self.assertEqual(len(dag1_cliffords), 0)

    def test_optimize_cliffords(self):
        """Test OptimizeCliffords pass."""

        rng = np.random.default_rng(1234)
        for _ in range(20):
            # Create several random Cliffords
            cliffs = [random_clifford(3, rng) for _ in range(5)]

            # The first circuit contains these cliffords
            qc1 = QuantumCircuit(5)
            for cliff in cliffs:
                qc1.append(cliff, [4, 0, 2])
            self.assertEqual(qc1.count_ops()["clifford"], 5)

            # The second circuit is obtained by running the OptimizeCliffords pass.
            qc2 = PassManager(OptimizeCliffords()).run(qc1)
            self.assertEqual(qc2.count_ops()["clifford"], 1)

            # The third circuit contains the decompositions of Cliffods.
            qc3 = QuantumCircuit(5)
            for cliff in cliffs:
                qc3.append(cliff.to_circuit(), [4, 0, 2])
            self.assertNotIn("clifford", qc3.count_ops())

            # Check that qc1, qc2 and qc3 and their decompositions are all equivalent.
            self.assertTrue(Operator(qc1).equiv(Operator(qc1.decompose())))
            self.assertTrue(Operator(qc2).equiv(Operator(qc2.decompose())))
            self.assertTrue(Operator(qc3).equiv(Operator(qc3.decompose())))
            self.assertTrue(Operator(qc1).equiv(Operator(qc2)))
            self.assertTrue(Operator(qc1).equiv(Operator(qc3)))

    def test_if_else(self):
        """Test pass recurses into simple if-else."""
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        combined = cliff1.compose(cliff2)

        inner_test = QuantumCircuit(cliff1.num_qubits)
        inner_test.append(cliff1, inner_test.qubits)
        inner_test.append(cliff2, inner_test.qubits)

        inner_expected = QuantumCircuit(combined.num_qubits)
        inner_expected.append(combined, inner_expected.qubits)

        test = QuantumCircuit(cliff1.num_qubits, 1)
        test.measure(0, 0)
        test.if_else((test.clbits[0], True), inner_test.copy(), inner_test.copy(), test.qubits, [])

        expected = QuantumCircuit(combined.num_qubits, 1)
        expected.measure(0, 0)
        expected.if_else(
            (expected.clbits[0], True), inner_expected, inner_expected, expected.qubits, []
        )

        self.assertEqual(OptimizeCliffords()(test), expected)

    def test_nested_control_flow(self):
        """Test pass recurses into nested control flow."""
        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff2()
        combined = cliff1.compose(cliff2)

        inner_test = QuantumCircuit(cliff1.num_qubits)
        inner_test.append(cliff1, inner_test.qubits)
        inner_test.append(cliff2, inner_test.qubits)

        while_test = QuantumCircuit(cliff1.num_qubits, 1)
        while_test.for_loop((0,), None, inner_test.copy(), while_test.qubits, [])

        inner_expected = QuantumCircuit(combined.num_qubits)
        inner_expected.append(combined, inner_expected.qubits)

        while_expected = QuantumCircuit(combined.num_qubits, 1)
        while_expected.for_loop((0,), None, inner_expected, while_expected.qubits, [])

        test = QuantumCircuit(cliff1.num_qubits, 1)
        test.measure(0, 0)
        test.while_loop((test.clbits[0], True), while_test, test.qubits, test.clbits)

        expected = QuantumCircuit(combined.num_qubits, 1)
        expected.measure(0, 0)
        expected.while_loop(
            (expected.clbits[0], True), while_expected, expected.qubits, expected.clbits
        )

        self.assertEqual(OptimizeCliffords()(test), expected)

    def test_topological_ordering(self):
        """Test that Clifford optimization pass optimizes Cliffords across a gate
        on a different qubit."""

        cliff1 = self.create_cliff1()
        cliff2 = self.create_cliff1()

        qc1 = QuantumCircuit(5)
        qc1.append(cliff1, [0, 1, 2])
        qc1.h(4)
        qc1.append(cliff2, [0, 1, 2])

        # The second circuit is obtained by running the OptimizeCliffords pass.
        qc2 = PassManager(OptimizeCliffords()).run(qc1)
        self.assertEqual(qc2.count_ops()["clifford"], 1)

    def test_transpile_level_0(self):
        """Make sure that transpile with optimization_level=0 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=0)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_1(self):
        """Make sure that transpile with optimization_level=1 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=1)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_2(self):
        """Make sure that transpile with optimization_level=2 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=2)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_transpile_level_3(self):
        """Make sure that transpile with optimization_level=3 transpiles
        the Clifford."""
        cliff1 = self.create_cliff1()
        qc = QuantumCircuit(3)
        qc.append(cliff1, [0, 1, 2])
        self.assertIn("clifford", qc.count_ops())
        qc2 = transpile(qc, optimization_level=3)
        self.assertNotIn("clifford", qc2.count_ops())

    def test_collect_cliffords_default(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly."""

        # original circuit (consisting of Clifford gates only)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.x(2)
        qc.cx(0, 1)
        qc.sdg(2)
        qc.swap(2, 1)
        qc.z(0)
        qc.cz(0, 1)
        qc.y(2)
        qc.cy(1, 2)

        # We should end up with a single Clifford object
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(qct.size(), 1)
        self.assertIn("clifford", qct.count_ops().keys())

    def test_collect_cliffords_max_block_width(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly when the option ``max_block_width`` is specified."""

        # original circuit (consisting of Clifford gates only)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.cx(0, 1)
        qc.sdg(0)
        qc.x(1)
        qc.swap(2, 1)
        qc.h(1)
        qc.swap(1, 2)

        # We should end up with two Clifford objects
        qct = PassManager(CollectCliffords(max_block_width=2)).run(qc)
        self.assertEqual(qct.size(), 2)
        self.assertEqual(qct[0].name, "clifford")
        self.assertEqual(len(qct[0].qubits), 2)
        self.assertEqual(qct[1].name, "clifford")
        self.assertEqual(len(qct[1].qubits), 2)

    def test_collect_cliffords_multiple_blocks(self):
        """Make sure that when collecting Clifford gates, non-Clifford gates
        are not collected, and the pass correctly splits disconnected Clifford
        blocks."""

        # original circuit (with one non-Clifford gate in the middle that uniquely
        # separates the circuit)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.x(2)
        qc.cx(0, 1)
        qc.sdg(2)
        qc.swap(2, 1)
        qc.t(1)
        qc.cz(0, 1)
        qc.z(0)
        qc.y(1)

        # We should end up with two Cliffords and one "rx" gate
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(qct.size(), 3)
        self.assertIn("t", qct.count_ops().keys())
        self.assertEqual(qct.count_ops()["clifford"], 2)

        self.assertIsInstance(qct.data[0].operation, Clifford)
        self.assertIsInstance(qct.data[2].operation, Clifford)

        collected_clifford1 = qct.data[0].operation
        collected_clifford2 = qct.data[2].operation

        expected_clifford_circuit1 = QuantumCircuit(3)
        expected_clifford_circuit1.h(0)
        expected_clifford_circuit1.s(1)
        expected_clifford_circuit1.x(2)
        expected_clifford_circuit1.cx(0, 1)
        expected_clifford_circuit1.sdg(2)
        expected_clifford_circuit1.swap(2, 1)
        expected_clifford1 = Clifford(expected_clifford_circuit1)

        expected_clifford_circuit2 = QuantumCircuit(2)
        expected_clifford_circuit2.cz(0, 1)
        expected_clifford_circuit2.z(0)
        expected_clifford_circuit2.y(1)
        expected_clifford2 = Clifford(expected_clifford_circuit2)

        # Check that collected and expected cliffords are equal
        self.assertEqual(collected_clifford1, expected_clifford1)
        self.assertEqual(collected_clifford2, expected_clifford2)

    def test_collect_cliffords_options(self):
        """Test the option split_blocks and min_block_size for collecting Clifford gates."""

        # original circuit (consisting of Clifford gates only)
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.sdg(2)
        qc.x(0)
        qc.z(0)
        qc.y(2)

        # When split_blocks is false and min_block_size=2 (default),
        # we should end up with a single Clifford object.
        qct = PassManager(CollectCliffords(split_blocks=False)).run(qc)
        self.assertEqual(qct.size(), 1)
        self.assertEqual(qct.count_ops()["clifford"], 1)

        # The above code should also work when commutativity analysis is enabled.
        qct = PassManager(CollectCliffords(split_blocks=False, do_commutative_analysis=True)).run(
            qc
        )
        self.assertEqual(qct.size(), 1)
        self.assertEqual(qct.count_ops()["clifford"], 1)

        # When split_blocks is true (default) and min_block_size is 1,
        # we should end up with 3 Cliffords (each over a single qubit).
        qct = PassManager(CollectCliffords(min_block_size=1)).run(qc)
        self.assertEqual(qct.size(), 3)
        self.assertEqual(qct.count_ops()["clifford"], 3)

        # When split_blocks is true (default) and min_block_size is 2,
        # we should end up with 2 Cliffords (the s(1)-gate should not be combined).
        qct = PassManager(CollectCliffords(min_block_size=2)).run(qc)
        self.assertEqual(qct.size(), 3)
        self.assertEqual(qct.count_ops()["clifford"], 2)

        # When split_blocks is true (default) and min_block_size is 3,
        # we should end up with a single Clifford.
        qct = PassManager(CollectCliffords(min_block_size=3)).run(qc)
        self.assertEqual(qct.size(), 4)
        self.assertEqual(qct.count_ops()["clifford"], 1)

        # When split_blocks is true (default) and min_block_size is 4,
        # no Cliffords should be collected.
        qct = PassManager(CollectCliffords(min_block_size=4)).run(qc)
        self.assertEqual(qct.size(), 6)
        self.assertNotIn("clifford", qct.count_ops())

    def test_collect_cliffords_options_multiple_blocks(self):
        """Test the option split_blocks and min_block_size for collecting Clifford
        gates when there are multiple disconnected Clifford blocks."""

        # original circuit (with several non-Clifford gate in the middle that uniquely
        # separates the circuit)
        qc = QuantumCircuit(4)
        qc.z(3)
        qc.cx(0, 2)
        qc.cy(1, 3)
        qc.x(2)
        qc.cx(2, 0)

        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 1)
        qc.rx(np.pi / 4, 2)

        qc.cz(0, 1)
        qc.z(0)

        # When split_blocks is false and min_block_size=2 (default),
        # we should end up with two Clifford object.
        qct = PassManager(CollectCliffords(split_blocks=False)).run(qc)
        self.assertEqual(qct.count_ops()["clifford"], 2)

        # The above code should also work when commutativity analysis is enabled.
        qct = PassManager(CollectCliffords(split_blocks=False, do_commutative_analysis=True)).run(
            qc
        )
        self.assertEqual(qct.count_ops()["clifford"], 2)

        # When split_blocks is true (default)
        # we should end up with 3 Cliffords, as the first Clifford
        # block further splits into two.
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(qct.count_ops()["clifford"], 3)

        # When split_blocks is true (default) and min_block_size is 3,
        # two of the Cliffords above do not get collected, so we should
        # end up with only one Clifford.
        qct = PassManager(CollectCliffords(min_block_size=3)).run(qc)
        self.assertEqual(qct.count_ops()["clifford"], 1)

        # When split_blocks is true (default) and min_block_size is 4,
        # no Cliffords should be collected.
        qct = PassManager(CollectCliffords(min_block_size=4)).run(qc)
        self.assertNotIn("clifford", qct.count_ops())

    def test_collect_from_back_corectness(self):
        """Test the option collect_from_back for collecting Clifford gates."""

        # original circuit (with non-Clifford gate on the first qubit in the middle
        # of the circuit)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(0)
        qc.x(1)
        qc.h(1)
        qc.t(0)
        qc.y(0)
        qc.h(1)

        qct1 = PassManager(CollectCliffords(collect_from_back=False)).run(qc)
        qct2 = PassManager(CollectCliffords(collect_from_back=True)).run(qc)
        self.assertEqual(Operator(qct1), Operator(qct2))

    def test_collect_from_back_as_expected(self):
        """Test the option collect_from_back for collecting Clifford gates."""
        # original circuit (with non-Clifford gate on the first qubit in the middle
        # of the circuit)
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(0)
        qc.x(1)
        qc.h(1)
        qc.t(0)
        qc.y(0)
        qc.h(1)

        qct = PassManager(
            CollectCliffords(collect_from_back=True, split_blocks=False, min_block_size=1)
        ).run(qc)

        self.assertIsInstance(qct.data[0].operation, Clifford)
        self.assertIsInstance(qct.data[2].operation, Clifford)

        collected_clifford1 = qct.data[0].operation
        collected_clifford2 = qct.data[2].operation

        # The first Clifford is over qubit {0}, the second is over qubits {0, 1}.
        expected_clifford_circuit1 = QuantumCircuit(1)
        expected_clifford_circuit1.x(0)
        expected_clifford_circuit1.h(0)

        expected_clifford_circuit2 = QuantumCircuit(2)
        expected_clifford_circuit2.x(1)
        expected_clifford_circuit2.h(1)
        expected_clifford_circuit2.y(0)
        expected_clifford_circuit2.h(1)

        expected_clifford1 = Clifford(expected_clifford_circuit1)
        expected_clifford2 = Clifford(expected_clifford_circuit2)

        # Check that collected and expected cliffords are equal
        self.assertEqual(collected_clifford1, expected_clifford1)
        self.assertEqual(collected_clifford2, expected_clifford2)

    def test_collect_split_layers(self):
        """Test the option split_layers for collecting Clifford gates."""

        # original circuit (consisting of Clifford gates only)
        qc = QuantumCircuit(3)
        qc.y(2)
        qc.z(2)
        qc.cx(0, 1)
        qc.h(0)
        qc.swap(0, 2)

        # When split_layers=True, we should get three layers:
        #   cx(0, 1), y(2)
        #   h(0), z(2)
        #   swap(0, 2)
        qct = PassManager(
            CollectCliffords(
                split_blocks=False,
                min_block_size=1,
                split_layers=True,
                do_commutative_analysis=False,
            )
        ).run(qc)

        self.assertEqual(Operator(qc), Operator(qct))
        self.assertEqual(qct.size(), 3)

        qct = PassManager(
            CollectCliffords(
                split_blocks=False,
                min_block_size=1,
                split_layers=True,
                do_commutative_analysis=True,
            )
        ).run(qc)

        self.assertEqual(Operator(qc), Operator(qct))
        self.assertEqual(qct.size(), 3)

    def test_collect_with_cliffords(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly when the gates include other cliffords."""

        # Create a Clifford over 2 qubits
        cliff_circuit = QuantumCircuit(2)
        cliff_circuit.cx(0, 1)
        cliff_circuit.h(0)
        cliff = Clifford(cliff_circuit)

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.append(cliff, [1, 0])
        qc.cx(1, 2)

        # Collect clifford gates from the circuit (in this case all the gates must be collected).
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(len(qct.data), 1)

        # Make sure that the operator for the initial quantum circuit is equivalent to the
        # operator for the collected clifford.
        op1 = Operator(qc)
        op2 = Operator(qct)
        self.assertTrue(op1.equiv(op2))

    def test_collect_with_linear_functions(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly when the gates include LinearFunctions."""

        # Create a linear function over 2 qubits
        lf = LinearFunction([[0, 1], [1, 0]])

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.append(lf, [1, 0])
        qc.cx(1, 2)

        # Collect clifford gates from the circuit (in this case all the gates must be collected).
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(len(qct.data), 1)

        # Make sure that the operator for the initial quantum circuit is equivalent to the
        # operator for the collected clifford.
        op1 = Operator(qc)
        op2 = Operator(qct)
        self.assertTrue(op1.equiv(op2))

    def test_collect_with_pauli_gates(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly when the gates include PauliGates."""

        # Create a pauli gate over 2 qubits
        pauli_gate = PauliGate("XY")

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.append(pauli_gate, [1, 0])
        qc.cx(1, 2)

        # Collect clifford gates from the circuit (in this case all the gates must be collected).
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(len(qct.data), 1)

        # Make sure that the operator for the initial quantum circuit is equivalent to the
        # operator for the collected clifford.
        op1 = Operator(qc)
        op2 = Operator(qct)
        self.assertTrue(op1.equiv(op2))

    def test_collect_with_all_types(self):
        """Make sure that collecting Clifford gates and replacing them by Clifford
        works correctly when the gates include all possible clifford gate types."""

        cliff_circuit0 = QuantumCircuit(1)
        cliff_circuit0.h(0)
        cliff0 = Clifford(cliff_circuit0)

        cliff_circuit1 = QuantumCircuit(2)
        cliff_circuit1.cz(0, 1)
        cliff_circuit1.s(1)
        cliff1 = Clifford(cliff_circuit1)

        lf1 = LinearFunction([[0, 1], [1, 1]])
        lf2 = LinearFunction([[0, 1, 0], [1, 0, 0], [0, 0, 1]])

        pauli_gate1 = PauliGate("X")
        pauli_gate2 = PauliGate("YZX")

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(cliff0, [1])
        qc.cy(0, 1)

        # not a clifford gate (separating the circuit)
        qc.t(0)

        qc.append(pauli_gate2, [0, 2, 1])
        qc.append(lf2, [2, 1, 0])
        qc.x(0)
        qc.append(pauli_gate1, [1])
        qc.append(lf1, [1, 0])
        qc.h(2)
        qc.append(cliff1, [1, 2])

        # Collect clifford gates from the circuit (we should get two Clifford blocks separated by
        # the RX gate).
        qct = PassManager(CollectCliffords()).run(qc)
        self.assertEqual(len(qct.data), 3)

        # Make sure that the operator for the initial quantum circuit is equivalent to the
        # operator for the circuit with the collected cliffords.
        op1 = Operator(qc)
        op2 = Operator(qct)
        self.assertTrue(op1.equiv(op2))

    def test_collect_all_clifford_gates(self):
        """Assert that CollectClifford collects all basis gates
        (including certain rotation gates with pi/2 angles)"""
        gates_1q = [
            XGate(),
            YGate(),
            ZGate(),
            IGate(),
            HGate(),
            SGate(),
            SdgGate(),
            SXGate(),
            SXdgGate(),
            RXGate(theta=np.pi / 2),
            RYGate(theta=np.pi / 2),
            RZGate(phi=np.pi / 2),
            UGate(np.pi / 2, np.pi / 2, np.pi / 2),
        ]
        gates_2q = [
            CXGate(),
            CYGate(),
            CZGate(),
            DCXGate(),
            ECRGate(),
            SwapGate(),
            iSwapGate(),
            CPhaseGate(theta=np.pi),
            CRXGate(theta=np.pi),
            CRYGate(theta=np.pi),
            CRZGate(theta=np.pi),
            RXXGate(theta=np.pi / 2),
            RYYGate(theta=np.pi / 2),
            RZZGate(theta=np.pi / 2),
            RZXGate(theta=np.pi / 2),
            XXMinusYYGate(theta=np.pi),
            XXPlusYYGate(theta=-np.pi),
        ]

        qc = QuantumCircuit(2)
        for gate in gates_1q:
            qc.append(gate, [0])
        for gate in gates_2q:
            qc.append(gate, [0, 1])

        qct = PassManager(CollectCliffords(matrix_based=True)).run(qc)
        self.assertEqual(qct.count_ops()["clifford"], 1)

    def test_plugin_unfortunate_name(self):
        """Test the synthesis is not triggered for a custom gate with the same name."""
        intruder = QuantumCircuit(2, name="clifford")
        circuit = QuantumCircuit(2)
        circuit.append(intruder.to_gate(), [0, 1])

        hls = HighLevelSynthesis()
        synthesized = hls(circuit)

        self.assertIn("clifford", synthesized.count_ops())


if __name__ == "__main__":
    unittest.main()
