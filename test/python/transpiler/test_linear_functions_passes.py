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

"""Test transpiler passes that deal with linear functions."""

import unittest
from ddt import ddt

from qiskit.circuit import QuantumCircuit, Qubit, Clbit
from qiskit.transpiler.passes.optimization import CollectLinearFunctions
from qiskit.transpiler.passes.synthesis import HighLevelSynthesis, LinearFunctionsToPermutations
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.circuit.library import real_amplitudes
from qiskit.transpiler import PassManager
from qiskit.quantum_info import Operator
from test import QiskitTestCase
from test import combine


@ddt
class TestLinearFunctionsPasses(QiskitTestCase):
    """Tests to verify correctness of the transpiler passes that deal with linear functions:
    the pass that extracts blocks of CX and SWAP gates and replaces these blocks by LinearFunctions,
    the pass that synthesizes LinearFunctions into CX and SWAP gates,
    and the pass that promotes LinearFunctions to Permutations whenever possible.
    """

    # Most of CollectLinearFunctions tests should work correctly both without and with
    # commutativity analysis.

    @combine(do_commutative_analysis=[False, True])
    def test_single_linear_block(self, do_commutative_analysis):
        """Test that when all gates in a circuit are either CX or SWAP,
        we end up with a single LinearFunction."""

        # original circuit
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.swap(2, 3)
        circuit.cx(0, 1)
        circuit.cx(0, 3)

        # new circuit with linear functions extracted using transpiler pass
        optimized_circuit = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit)

        # check that this circuit consists of a single LinearFunction
        self.assertIn("linear_function", optimized_circuit.count_ops().keys())
        self.assertEqual(len(optimized_circuit.data), 1)
        inst1 = optimized_circuit.data[0]
        self.assertIsInstance(inst1.operation, LinearFunction)

        # construct a circuit with linear function directly, without the transpiler pass
        expected_circuit = QuantumCircuit(4)
        expected_circuit.append(LinearFunction(circuit), [0, 1, 2, 3])

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(expected_circuit))

        # now a circuit with linear functions synthesized
        synthesized_circuit = PassManager(HighLevelSynthesis()).run(optimized_circuit)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(optimized_circuit), Operator(synthesized_circuit))

    @combine(do_commutative_analysis=[False, True])
    def test_two_linear_blocks(self, do_commutative_analysis):
        """Test that when we have two blocks of linear gates with one nonlinear gate in the middle,
        we end up with two LinearFunctions."""
        # Create a circuit with a nonlinear gate (h) cleanly separating it into two linear blocks.
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)

        # new circuit with linear functions extracted using transpiler pass
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # We expect to see 3 gates (linear, h, linear)
        self.assertEqual(len(circuit2.data), 3)
        inst1 = circuit2.data[0]
        inst2 = circuit2.data[2]
        self.assertIsInstance(inst1.operation, LinearFunction)
        self.assertIsInstance(inst2.operation, LinearFunction)

        # Check that the first linear function represents the subcircuit before h
        resulting_subcircuit1 = QuantumCircuit(4)
        resulting_subcircuit1.append(inst1)

        expected_subcircuit1 = QuantumCircuit(4)
        expected_subcircuit1.cx(0, 1)
        expected_subcircuit1.cx(0, 2)
        expected_subcircuit1.cx(0, 3)

        self.assertEqual(Operator(resulting_subcircuit1), Operator(expected_subcircuit1))

        # Check that the second linear function represents the subcircuit after h
        resulting_subcircuit2 = QuantumCircuit(4)
        resulting_subcircuit2.append(inst2)

        expected_subcircuit2 = QuantumCircuit(4)
        expected_subcircuit2.swap(2, 3)
        expected_subcircuit2.cx(1, 2)
        expected_subcircuit2.cx(0, 1)
        self.assertEqual(Operator(resulting_subcircuit2), Operator(expected_subcircuit2))

        # now a circuit with linear functions synthesized
        synthesized_circuit = PassManager(HighLevelSynthesis()).run(circuit2)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", synthesized_circuit.count_ops().keys())

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit2), Operator(synthesized_circuit))

    @combine(do_commutative_analysis=[False, True])
    def test_to_permutation(self, do_commutative_analysis):
        """Test that converting linear functions to permutations works correctly."""

        # Original circuit with two linear blocks; the second block happens to be
        # a permutation
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.cx(0, 3)
        circuit1.h(3)
        circuit1.swap(2, 3)
        circuit1.cx(1, 2)
        circuit1.cx(2, 1)
        circuit1.cx(1, 2)

        # new circuit with linear functions extracted
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # check that we have two blocks of linear functions
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # new circuit with linear functions converted to permutations
        # (the first linear function should not be converted, the second should)
        circuit3 = PassManager(LinearFunctionsToPermutations()).run(circuit2)

        # check that there is one linear function and one permutation
        self.assertEqual(circuit3.count_ops()["linear_function"], 1)
        self.assertEqual(circuit3.count_ops()["permutation"], 1)

        # check that the final circuit is still equivalent to the original circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_hidden_identity_block(self, do_commutative_analysis):
        """Test that extracting linear functions and synthesizing them back
        results in an equivalent circuit when a linear block represents
        the identity matrix."""

        # Create a circuit with multiple non-linear blocks
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)
        circuit1.swap(0, 2)
        circuit1.swap(0, 2)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.h(2)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # synthesize linear functions
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_multiple_non_linear_blocks(self, do_commutative_analysis):
        """Test that extracting linear functions and synthesizing them back
        results in an equivalent circuit when there are multiple non-linear blocks."""

        # Create a circuit with multiple non-linear blocks
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.s(1)
        circuit1.h(0)
        circuit1.cx(0, 1)
        circuit1.cx(0, 2)
        circuit1.swap(1, 2)
        circuit1.h(1)
        circuit1.sdg(2)
        circuit1.cx(1, 0)
        circuit1.cx(1, 2)
        circuit1.h(2)
        circuit1.cx(1, 2)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(0, 1)
        circuit1.h(1)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # synthesize linear functions
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit1), Operator(circuit3))

    @combine(do_commutative_analysis=[False, True])
    def test_real_amplitudes_circuit_4q(self, do_commutative_analysis):
        """Test that for the 4-qubit real amplitudes circuit
        extracting linear functions produces the expected number of linear blocks,
        and synthesizing these blocks produces an expected number of CNOTs.
        """
        ansatz = real_amplitudes(4, reps=2)
        circuit1 = ansatz.decompose()

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # synthesize linear functions
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()["cx"], 6)

    @combine(do_commutative_analysis=[False, True])
    def test_real_amplitudes_circuit_5q(self, do_commutative_analysis):
        """Test that for the 5-qubit real amplitudes circuit
        extracting linear functions produces the expected number of linear blocks,
        and synthesizing these blocks produces an expected number of CNOTs.
        """
        ansatz = real_amplitudes(5, reps=2)
        circuit1 = ansatz.decompose()

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

        # synthesize linear functions
        circuit3 = PassManager(HighLevelSynthesis()).run(circuit2)
        self.assertEqual(circuit3.count_ops()["cx"], 8)

    @combine(do_commutative_analysis=[False, True])
    def test_not_collecting_single_gates1(self, do_commutative_analysis):
        """Test that extraction of linear functions does not create
        linear functions out of single gates.
        """
        circuit1 = QuantumCircuit(3)
        circuit1.cx(0, 1)
        circuit1.h(1)
        circuit1.cx(1, 2)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_not_collecting_single_gates2(self, do_commutative_analysis):
        """Test that extraction of linear functions does not create
        linear functions out of single gates.
        """
        circuit1 = QuantumCircuit(3)
        circuit1.h(0)
        circuit1.h(1)
        circuit1.swap(0, 1)
        circuit1.s(1)
        circuit1.swap(1, 2)
        circuit1.h(2)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_disconnected_gates1(self, do_commutative_analysis):
        """Test that extraction of linear functions does not create
        linear functions out of disconnected gates.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(2, 3)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # check that there are no LinearFunctions present in synthesized_circuit
        self.assertNotIn("linear_function", circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_disconnected_gates2(self, do_commutative_analysis):
        """Test that extraction of linear functions does not create
        linear functions out of disconnected gates.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # we expect that the first two CX gates will be combined into
        # a linear function, but the last will not
        self.assertEqual(circuit2.count_ops()["linear_function"], 1)
        self.assertEqual(circuit2.count_ops()["cx"], 1)

    @combine(do_commutative_analysis=[False, True])
    def test_connected_gates(self, do_commutative_analysis):
        """Test that extraction of linear functions combines gates
        which become connected later.
        """
        circuit1 = QuantumCircuit(4)
        circuit1.cx(0, 1)
        circuit1.cx(1, 0)
        circuit1.cx(2, 3)
        circuit1.swap(0, 3)

        # collect linear functions
        circuit2 = PassManager(
            CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        ).run(circuit1)

        # we expect that the first two CX gates will be combined into
        # a linear function, but the last will not
        self.assertEqual(circuit2.count_ops()["linear_function"], 1)
        self.assertNotIn("cx", circuit2.count_ops().keys())
        self.assertNotIn("swap", circuit2.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_if_else(self, do_commutative_analysis):
        """Test that collection recurses into a simple if-else."""
        pass_ = CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)

        circuit = QuantumCircuit(4, 1)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(2, 3)

        test = QuantumCircuit(4, 1)
        test.h(0)
        test.measure(0, 0)
        test.if_else((0, True), circuit.copy(), circuit.copy(), range(4), [0])

        expected = QuantumCircuit(4, 1)
        expected.h(0)
        expected.measure(0, 0)
        expected.if_else((0, True), pass_(circuit), pass_(circuit), range(4), [0])

        self.assertEqual(pass_(test), expected)

    @combine(do_commutative_analysis=[False, True])
    def test_nested_control_flow(self, do_commutative_analysis):
        """Test that collection recurses into nested control flow."""
        pass_ = CollectLinearFunctions(do_commutative_analysis=do_commutative_analysis)
        qubits = [Qubit() for _ in [None] * 4]
        clbit = Clbit()

        circuit = QuantumCircuit(qubits, [clbit])
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(2, 3)

        true_body = QuantumCircuit(qubits, [clbit])
        true_body.while_loop((clbit, True), circuit.copy(), [0, 1, 2, 3], [0])

        test = QuantumCircuit(qubits, [clbit])
        test.for_loop(range(2), None, circuit.copy(), [0, 1, 2, 3], [0])
        test.if_else((clbit, True), true_body, None, [0, 1, 2, 3], [0])

        expected_if_body = QuantumCircuit(qubits, [clbit])
        expected_if_body.while_loop((clbit, True), pass_(circuit), [0, 1, 2, 3], [0])
        expected = QuantumCircuit(qubits, [clbit])
        expected.for_loop(range(2), None, pass_(circuit), [0, 1, 2, 3], [0])
        expected.if_else((clbit, True), pass_(expected_if_body), None, [0, 1, 2, 3], [0])

        self.assertEqual(pass_(test), expected)

    @combine(do_commutative_analysis=[False, True])
    def test_split_blocks(self, do_commutative_analysis):
        """Test that splitting blocks of nodes into sub-blocks works correctly."""

        # original circuit is linear
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)

        # If we do not split block into sub-blocks, we expect a single linear block.
        circuit1 = PassManager(
            CollectLinearFunctions(
                split_blocks=False, do_commutative_analysis=do_commutative_analysis
            )
        ).run(circuit)
        self.assertEqual(circuit1.count_ops()["linear_function"], 1)

        # If we do split block into sub-blocks, we expect two linear blocks:
        # one over qubits {0, 2, 3}, and another over qubits {1, 4}.
        circuit2 = PassManager(
            CollectLinearFunctions(
                split_blocks=True, do_commutative_analysis=do_commutative_analysis
            )
        ).run(circuit)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)

    @combine(do_commutative_analysis=[False, True])
    def test_do_not_split_blocks(self, do_commutative_analysis):
        """Test that splitting blocks of nodes into sub-blocks works correctly."""

        # original circuit is linear
        circuit = QuantumCircuit(5)
        circuit.cx(0, 3)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.swap(4, 2)

        # Check that we have a single linear block
        circuit1 = PassManager(
            CollectLinearFunctions(
                split_blocks=True, do_commutative_analysis=do_commutative_analysis
            )
        ).run(circuit)
        self.assertEqual(circuit1.count_ops()["linear_function"], 1)

    def test_commutative_analysis(self):
        """Test that collecting linear blocks with commutativity analysis can merge blocks
        (if they can be commuted to be next to each other)."""

        # original circuit
        # note that z(0) commutes with cx(0, *) gates, and x(3) commutes with cx(*, 3) gates
        circuit = QuantumCircuit(4)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.z(0)
        circuit.cx(0, 1)
        circuit.cx(0, 2)
        circuit.cx(0, 3)
        circuit.x(3)
        circuit.cx(2, 3)
        circuit.cx(1, 3)

        # circuit with linear functions extracted without commutativity analysis
        # z(0) and x(3) break the circuit into 3 linear blocks
        circuit1 = PassManager(CollectLinearFunctions(do_commutative_analysis=False)).run(circuit)
        self.assertEqual(circuit1.count_ops()["linear_function"], 3)
        self.assertNotIn("cx", circuit1.count_ops().keys())
        self.assertNotIn("swap", circuit1.count_ops().keys())

        # circuit with linear functions extracted with commutativity analysis
        # z(0) and x(3) can be commuted out of the linear block
        circuit2 = PassManager(CollectLinearFunctions(do_commutative_analysis=True)).run(circuit)
        self.assertEqual(circuit2.count_ops()["linear_function"], 1)
        self.assertNotIn("cx", circuit2.count_ops().keys())
        self.assertNotIn("swap", circuit2.count_ops().keys())

    def test_min_block_size(self):
        """Test that the option min_block_size for collecting linear functions works correctly."""

        # original circuit
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.cx(1, 0)
        circuit.cx(0, 1)

        # When min_block_size = 1, we should obtain 3 linear blocks
        circuit1 = PassManager(CollectLinearFunctions(min_block_size=1)).run(circuit)
        self.assertEqual(circuit1.count_ops()["linear_function"], 3)
        self.assertNotIn("cx", circuit1.count_ops().keys())

        # When min_block_size = 2, we should obtain 2 linear blocks
        circuit2 = PassManager(CollectLinearFunctions(min_block_size=2)).run(circuit)
        self.assertEqual(circuit2.count_ops()["linear_function"], 2)
        self.assertEqual(circuit2.count_ops()["cx"], 1)

        # When min_block_size = 3, we should obtain 1 linear block
        circuit3 = PassManager(CollectLinearFunctions(min_block_size=3)).run(circuit)
        self.assertEqual(circuit3.count_ops()["linear_function"], 1)
        self.assertEqual(circuit3.count_ops()["cx"], 3)

        # When min_block_size = 4, we should obtain no linear blocks
        circuit4 = PassManager(CollectLinearFunctions(min_block_size=4)).run(circuit)
        self.assertNotIn("linear_function", circuit4.count_ops().keys())
        self.assertEqual(circuit4.count_ops()["cx"], 6)

    def test_max_block_width(self):
        """Test that the option max_block_width for collecting linear functions works correctly."""
        circuit = QuantumCircuit(6)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.cx(4, 5)

        # When max_block_width = 3, we should obtain 3 linear blocks
        circuit1 = PassManager(CollectLinearFunctions(min_block_size=1, max_block_width=3)).run(
            circuit
        )
        self.assertEqual(circuit1.count_ops()["linear_function"], 3)
        self.assertNotIn("cx", circuit1.count_ops().keys())

        # When max_block_width = 4, we should obtain 2 linear blocks
        circuit1 = PassManager(CollectLinearFunctions(min_block_size=1, max_block_width=4)).run(
            circuit
        )
        self.assertEqual(circuit1.count_ops()["linear_function"], 2)
        self.assertNotIn("cx", circuit1.count_ops().keys())

    @combine(do_commutative_analysis=[False, True])
    def test_collect_from_back_correctness(self, do_commutative_analysis):
        """Test that collecting from the back of the circuit works correctly."""

        # original circuit
        circuit = QuantumCircuit(5)
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.cx(3, 4)
        circuit.h(2)
        circuit.swap(0, 1)
        circuit.swap(1, 2)
        circuit.swap(2, 3)
        circuit.swap(3, 4)

        circuit1 = PassManager(
            CollectLinearFunctions(
                split_blocks=False,
                do_commutative_analysis=do_commutative_analysis,
                collect_from_back=False,
            )
        ).run(circuit)

        circuit2 = PassManager(
            CollectLinearFunctions(
                split_blocks=False,
                do_commutative_analysis=do_commutative_analysis,
                collect_from_back=True,
            )
        ).run(circuit)
        self.assertEqual(Operator(circuit1), Operator(circuit2))

    @combine(do_commutative_analysis=[False, True])
    def test_collect_from_back_as_expected(self, do_commutative_analysis):
        """Test that collecting from the back of the circuit works as expected."""

        # original circuit
        circuit = QuantumCircuit(3)
        circuit.cx(1, 2)
        circuit.cx(1, 0)
        circuit.h(2)
        circuit.cx(1, 2)

        # If we collect from the back, we expect the cx(1, 0) to be part of the second block.
        circuit1 = PassManager(
            CollectLinearFunctions(
                split_blocks=False,
                min_block_size=1,
                do_commutative_analysis=do_commutative_analysis,
                collect_from_back=True,
            )
        ).run(circuit)

        # We expect to see 3 gates (linear, h, linear)
        self.assertEqual(len(circuit1.data), 3)
        inst1 = circuit1.data[0]
        inst2 = circuit1.data[2]
        self.assertIsInstance(inst1.operation, LinearFunction)
        self.assertIsInstance(inst2.operation, LinearFunction)

        resulting_subcircuit1 = QuantumCircuit(3)
        resulting_subcircuit1.append(inst1)
        resulting_subcircuit2 = QuantumCircuit(3)
        resulting_subcircuit2.append(inst2)

        expected_subcircuit1 = QuantumCircuit(3)
        expected_subcircuit1.cx(1, 2)

        expected_subcircuit2 = QuantumCircuit(3)
        expected_subcircuit2.cx(1, 0)
        expected_subcircuit2.cx(1, 2)

        self.assertEqual(Operator(resulting_subcircuit1), Operator(expected_subcircuit1))
        self.assertEqual(Operator(resulting_subcircuit2), Operator(expected_subcircuit2))

    @combine(do_commutative_analysis=[False, True])
    def test_split_layers(self, do_commutative_analysis):
        """Test that splitting blocks of nodes into layers works correctly."""

        # original circuit is linear
        circuit = QuantumCircuit(5)
        circuit.cx(0, 2)
        circuit.cx(1, 4)
        circuit.cx(2, 0)
        circuit.cx(0, 3)
        circuit.swap(3, 2)
        circuit.swap(4, 1)

        circuit2 = PassManager(
            CollectLinearFunctions(
                split_blocks=False,
                min_block_size=1,
                split_layers=True,
                do_commutative_analysis=do_commutative_analysis,
            )
        ).run(circuit)

        # check that we have an equivalent circuit
        self.assertEqual(Operator(circuit), Operator(circuit2))

        # Check that we have the expected number of linear blocks
        self.assertEqual(circuit2.count_ops()["linear_function"], 4)


if __name__ == "__main__":
    unittest.main()
