# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the LookaheadSwap pass"""

import unittest

import ddt
import numpy as np
from numpy import pi

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes import LookaheadSwap, CheckMap
from qiskit.transpiler import CouplingMap, Target
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library import CXGate, PermutationGate
from qiskit.quantum_info import Operator
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, transpile
from test import QiskitTestCase

from ..legacy_cmaps import MELBOURNE_CMAP


def _issue_reproducer():
    """Circuit and coupling map on which LookaheadSwap used to loop forever."""
    coupling_map = CouplingMap([(0, 6), (0, 4), (1, 6), (1, 7), (2, 7), (3, 4), (5, 7)])
    coupling_map.make_symmetric()
    circuit = QuantumCircuit(QuantumRegister(8, "q"))
    for a, b in [(0, 2), (0, 3), (0, 4), (1, 4), (1, 5), (3, 5)]:
        circuit.cz(a, b)
    return circuit, coupling_map


@ddt.ddt
class TestLookaheadSwap(QiskitTestCase):
    """Tests the LookaheadSwap pass."""

    def assertRoutedEquivalent(self, circuit, coupling_map, search_depth=4, search_width=4):
        """Route ``circuit`` and check the output respects ``coupling_map``, keeps every
        non-swap operation and, after undoing the final layout, implements the same unitary."""
        pass_ = LookaheadSwap(coupling_map, search_depth=search_depth, search_width=search_width)
        mapped_dag = pass_.run(circuit_to_dag(circuit))

        check_map = CheckMap(coupling_map)
        check_map.run(mapped_dag)
        self.assertTrue(check_map.property_set["is_swap_mapped"])

        mapped_ops = dict(mapped_dag.count_ops())
        mapped_ops.pop("swap", None)
        self.assertEqual(mapped_ops, dict(circuit.count_ops()))

        mapped = dag_to_circuit(mapped_dag)
        final_layout = pass_.property_set["final_layout"]
        mapped.append(PermutationGate([final_layout[q] for q in mapped.qubits]), mapped.qubits)
        self.assertTrue(Operator(mapped).equiv(Operator(circuit)))
        return mapped_dag

    def test_lookahead_swap_doesnt_modify_mapped_circuit(self):
        """Test that lookahead swap is idempotent.

        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """

        qr = QuantumRegister(3, name="q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        original_dag = circuit_to_dag(circuit)

        # Create coupling map which contains all two-qubit gates in the circuit.
        coupling_map = CouplingMap([[0, 1], [0, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(original_dag)

        self.assertEqual(original_dag, mapped_dag)

        remapped_dag = LookaheadSwap(coupling_map).run(mapped_dag)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_lookahead_swap_should_add_a_single_swap(self):
        """Test that LookaheadSwap will insert a SWAP to match layout.

        For a single cx gate which is not available in the current layout, test
        that the mapper inserts a single swap to enable the gate.
        """

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        self.assertEqual(
            mapped_dag.count_ops().get("swap", 0), dag_circuit.count_ops().get("swap", 0) + 1
        )

    def test_lookahead_swap_finds_minimal_swap_solution(self):
        """Of many valid SWAPs, test that LookaheadSwap finds the cheapest path.

        For a two CNOT circuit: cx q[0],q[2]; cx q[0],q[1]
        on the initial layout: qN -> qN
        (At least) two solutions exist:
        - SWAP q[0],[1], cx q[0],q[2], cx q[0],q[1]
        - SWAP q[1],[2], cx q[0],q[2], SWAP q[1],q[2], cx q[0],q[1]

        Verify that we find the first solution, as it requires fewer SWAPs.
        """

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        self.assertEqual(
            mapped_dag.count_ops().get("swap", 0), dag_circuit.count_ops().get("swap", 0) + 1
        )

    def test_lookahead_swap_maps_measurements(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.

        Create a circuit with measures on q0 and q2, following a swap between q0 and q2.
        Since that swap is not in the coupling, one of the two will be required to move.
        Verify that the mapped measure corresponds to one of the two possible layouts following
        the swap.

        """

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)

        circuit.cx(qr[0], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[2], cr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        mapped_measure_qargs = {op.qargs[0] for op in mapped_dag.named_nodes("measure")}

        self.assertIn(mapped_measure_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_maps_measurements_with_target(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.

        Create a circuit with measures on q0 and q2, following a swap between q0 and q2.
        Since that swap is not in the coupling, one of the two will be required to move.
        Verify that the mapped measure corresponds to one of the two possible layouts following
        the swap.

        """

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)

        circuit.cx(qr[0], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[2], cr[1])

        dag_circuit = circuit_to_dag(circuit)

        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None})

        mapped_dag = LookaheadSwap(target).run(dag_circuit)

        mapped_measure_qargs = {op.qargs[0] for op in mapped_dag.named_nodes("measure")}

        self.assertIn(mapped_measure_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_maps_barriers(self):
        """Verify barrier nodes are updated to re-mapped qregs.

        Create a circuit with a barrier on q0 and q2, following a swap between q0 and q2.
        Since that swap is not in the coupling, one of the two will be required to move.
        Verify that the mapped barrier corresponds to one of the two possible layouts following
        the swap.

        """

        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)

        circuit.cx(qr[0], qr[2])
        circuit.barrier(qr[0], qr[2])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        mapped_barrier_qargs = next(set(op.qargs) for op in mapped_dag.named_nodes("barrier"))

        self.assertIn(mapped_barrier_qargs, [{qr[0], qr[1]}, {qr[1], qr[2]}])

    def test_lookahead_swap_higher_depth_width_is_better(self):
        """Test that lookahead swap finds better circuit with increasing search space.

        Increasing the tree width and depth is expected to yield a better (or same) quality
        circuit, in the form of fewer SWAPs.
        """
        # q_0: ──■───────────────────■───────────────────────────────────────────────»
        #      ┌─┴─┐                 │                 ┌───┐                         »
        # q_1: ┤ X ├──■──────────────┼─────────────────┤ X ├─────────────────────────»
        #      └───┘┌─┴─┐            │                 └─┬─┘┌───┐          ┌───┐     »
        # q_2: ─────┤ X ├──■─────────┼───────────────────┼──┤ X ├──────────┤ X ├──■──»
        #           └───┘┌─┴─┐     ┌─┴─┐                 │  └─┬─┘     ┌───┐└─┬─┘  │  »
        # q_3: ──────────┤ X ├──■──┤ X ├─────────────────┼────┼────■──┤ X ├──┼────┼──»
        #                └───┘┌─┴─┐└───┘          ┌───┐  │    │    │  └─┬─┘  │    │  »
        # q_4: ───────────────┤ X ├──■────────────┤ X ├──┼────■────┼────┼────┼────┼──»
        #                     └───┘┌─┴─┐          └─┬─┘  │         │    │    │    │  »
        # q_5: ────────────────────┤ X ├──■─────────┼────┼─────────┼────■────┼────┼──»
        #                          └───┘┌─┴─┐       │    │         │         │    │  »
        # q_6: ─────────────────────────┤ X ├──■────■────┼─────────┼─────────■────┼──»
        #                               └───┘┌─┴─┐       │       ┌─┴─┐          ┌─┴─┐»
        # q_7: ──────────────────────────────┤ X ├───────■───────┤ X ├──────────┤ X ├»
        #                                    └───┘               └───┘          └───┘»
        # «q_0: ──■───────
        # «       │
        # «q_1: ──┼───────
        # «       │
        # «q_2: ──┼───────
        # «       │
        # «q_3: ──┼───────
        # «       │
        # «q_4: ──┼───────
        # «       │
        # «q_5: ──┼────■──
        # «     ┌─┴─┐  │
        # «q_6: ┤ X ├──┼──
        # «     └───┘┌─┴─┐
        # «q_7: ─────┤ X ├
        # «          └───┘
        qr = QuantumRegister(8, name="q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[6], qr[4])
        circuit.cx(qr[7], qr[1])
        circuit.cx(qr[4], qr[2])
        circuit.cx(qr[3], qr[7])
        circuit.cx(qr[5], qr[3])
        circuit.cx(qr[6], qr[2])
        circuit.cx(qr[2], qr[7])
        circuit.cx(qr[0], qr[6])
        circuit.cx(qr[5], qr[7])
        original_dag = circuit_to_dag(circuit)

        # Create a ring of 8 connected qubits
        coupling_map = CouplingMap.from_grid(num_rows=2, num_columns=4)

        mapped_dag_1 = LookaheadSwap(coupling_map, search_depth=3, search_width=3).run(original_dag)
        mapped_dag_2 = LookaheadSwap(coupling_map, search_depth=5, search_width=5).run(original_dag)

        num_swaps_1 = mapped_dag_1.count_ops().get("swap", 0)
        num_swaps_2 = mapped_dag_2.count_ops().get("swap", 0)

        self.assertLessEqual(num_swaps_2, num_swaps_1)

    def test_lookahead_swap_hang_in_min_case(self):
        """Verify LookaheadSwap does not stall in minimal case."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2171

        qr = QuantumRegister(14, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        dag = circuit_to_dag(qc)

        cmap = CouplingMap(MELBOURNE_CMAP)
        out = LookaheadSwap(cmap, search_depth=4, search_width=4).run(dag)

        self.assertIsInstance(out, DAGCircuit)

    def test_lookahead_swap_hang_full_case(self):
        """Verify LookaheadSwap does not stall in reported case."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/2171

        qr = QuantumRegister(14, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        qc.cx(qr[6], qr[7])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[8], qr[6])
        qc.cx(qr[7], qr[8])
        qc.cx(qr[0], qr[13])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[13], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        cmap = CouplingMap(MELBOURNE_CMAP)

        out = LookaheadSwap(cmap, search_depth=4, search_width=4).run(dag)

        self.assertIsInstance(out, DAGCircuit)

    def test_global_phase_preservation(self):
        """Test that LookaheadSwap preserves global phase"""

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.global_phase = pi / 3
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        self.assertEqual(mapped_dag.global_phase, circuit.global_phase)
        self.assertEqual(
            mapped_dag.count_ops().get("swap", 0), dag_circuit.count_ops().get("swap", 0) + 1
        )

    def test_lookahead_swap_terminates_on_issue_reproducer(self):
        """Test the reported circuit that made LookaheadSwap loop forever is routed correctly.

        The search used to settle in a local minimum of the layout heuristic and return a swap
        followed by its own inverse forever, without ever mapping a gate.
        """
        circuit, coupling_map = _issue_reproducer()
        transpiled = transpile(
            circuit,
            coupling_map=coupling_map,
            layout_method="trivial",
            routing_method="lookahead",
            optimization_level=0,
        )
        self.assertEqual(transpiled.count_ops()["cz"], 6)
        check_map = CheckMap(coupling_map)
        check_map.run(circuit_to_dag(transpiled))
        self.assertTrue(check_map.property_set["is_swap_mapped"])
        self.assertTrue(Operator.from_circuit(transpiled).equiv(Operator(circuit)))

    @ddt.data((1, 1), (2, 2), (4, 4), (5, 6))
    @ddt.unpack
    def test_lookahead_swap_issue_reproducer_search_settings(self, search_depth, search_width):
        """Test the reported circuit is routed correctly for a range of search settings."""
        circuit, coupling_map = _issue_reproducer()
        self.assertRoutedEquivalent(circuit, coupling_map, search_depth, search_width)

    def test_lookahead_swap_escapes_local_minimum_on_grid(self):
        """Test a grid circuit whose lookahead search makes no progress is still routed."""
        circuit = QuantumCircuit(QuantumRegister(9, "q"))
        for a, b in [(2, 8), (5, 2), (8, 5), (6, 1), (5, 6)]:
            circuit.cx(a, b)
        self.assertRoutedEquivalent(circuit, CouplingMap.from_grid(3, 3), 2, 2)

    @ddt.data((1, 1), (2, 2), (4, 4))
    @ddt.unpack
    def test_lookahead_swap_reversing_a_swap(self, search_depth, search_width):
        """Test a circuit whose routing moves a qubit away for one gate and back for the next."""
        circuit = QuantumCircuit(QuantumRegister(4, "q"))
        circuit.cx(0, 1)
        circuit.cx(1, 3)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        circuit.cx(0, 3)
        self.assertRoutedEquivalent(circuit, CouplingMap.from_line(4), search_depth, search_width)

    @ddt.data(
        ("line", CouplingMap.from_line(6)),
        ("ring", CouplingMap.from_ring(8)),
        ("grid", CouplingMap.from_grid(3, 3)),
        ("tree", CouplingMap([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])),
    )
    @ddt.unpack
    def test_lookahead_swap_already_routable_on_topology(self, _, coupling_map):
        """Test a circuit with gates only on coupled qubits needs no swaps on various topologies."""
        circuit = QuantumCircuit(QuantumRegister(coupling_map.size(), "q"))
        for a, b in coupling_map.get_edges():
            circuit.cx(a, b)
        mapped_dag = self.assertRoutedEquivalent(circuit, coupling_map)
        self.assertEqual(mapped_dag, circuit_to_dag(circuit))

    @ddt.idata(
        (name, coupling_map, depth, width, seed)
        for name, coupling_map in [
            ("line", CouplingMap.from_line(6)),
            ("ring", CouplingMap.from_ring(8)),
            ("grid", CouplingMap.from_grid(3, 3)),
            ("tree", CouplingMap([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])),
        ]
        for depth, width in [(2, 2), (4, 4)]
        for seed in range(2)
    )
    @ddt.unpack
    def test_lookahead_swap_random_circuits_on_topology(self, _, coupling_map, depth, width, seed):
        """Test circuits with many blocked gates are routed correctly on various topologies."""
        rng = np.random.default_rng(seed)
        num_qubits = coupling_map.size()
        circuit = QuantumCircuit(QuantumRegister(num_qubits, "q"))
        for i in range(25):
            a, b = rng.choice(num_qubits, 2, replace=False)
            circuit.cx(int(a), int(b))
            if i % 5 == 0:
                circuit.h(int(rng.integers(num_qubits)))
            if i % 10 == 9:
                circuit.barrier()
        self.assertRoutedEquivalent(circuit, coupling_map, depth, width)

    def test_lookahead_swap_disconnected_coupling_map(self):
        """Test routing within the components of a disconnected coupling map."""
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)])
        circuit = QuantumCircuit(QuantumRegister(8, "q"))
        circuit.cx(0, 3)
        circuit.cx(4, 7)
        circuit.cx(1, 3)
        circuit.cx(7, 5)
        circuit.cx(0, 2)
        circuit.cx(6, 4)
        self.assertRoutedEquivalent(circuit, coupling_map, 2, 2)


if __name__ == "__main__":
    unittest.main()
