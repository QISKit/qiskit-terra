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

"""Test the Sabre Swap pass"""

import unittest
import itertools
import pickle
from copy import deepcopy
import io

import ddt
import numpy.random

from qiskit.circuit import Clbit, ControlFlowOp, Qubit
from qiskit.circuit.library import CCXGate, HGate, Measure, SwapGate
from qiskit.circuit.classical import expr, types
from qiskit.circuit.random import random_circuit
from qiskit.compiler.transpiler import transpile
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes import SabreSwap, CheckMap
from qiskit.transpiler.passes.routing.sabre_swap import Heuristic, SetScaling
from qiskit.transpiler import CouplingMap, Layout, PassManager, Target, TranspilerError
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.utils import optionals
from test.utils._canonical import canonicalize_control_flow
from test import QiskitTestCase

from ..legacy_cmaps import MUMBAI_CMAP


def looping_circuit(uphill_swaps=1, additional_local_minimum_gates=0):
    """A circuit that causes SabreSwap to loop infinitely for the legacy 'lookahead' heuristic.

    This looks like (using cz gates to show the symmetry, though we actually output cx for testing
    purposes):

    .. code-block:: text

         q_0: ─■────────────────
               │
         q_1: ─┼──■─────────────
               │  │
         q_2: ─┼──┼──■──────────
               │  │  │
         q_3: ─┼──┼──┼──■───────
               │  │  │  │
         q_4: ─┼──┼──┼──┼─────■─
               │  │  │  │     │
         q_5: ─┼──┼──┼──┼──■──■─
               │  │  │  │  │
         q_6: ─┼──┼──┼──┼──┼────
               │  │  │  │  │
         q_7: ─┼──┼──┼──┼──■──■─
               │  │  │  │     │
         q_8: ─┼──┼──┼──┼─────■─
               │  │  │  │
         q_9: ─┼──┼──┼──■───────
               │  │  │
        q_10: ─┼──┼──■──────────
               │  │
        q_11: ─┼──■─────────────
               │
        q_12: ─■────────────────

    where `uphill_swaps` is the number of qubits separating the inner-most gate (representing how
    many swaps need to be made that all increase the heuristics), and
    `additional_local_minimum_gates` is how many extra gates to add on the outside (these increase
    the size of the region of stability).
    """
    outers = 4 + additional_local_minimum_gates
    n_qubits = 2 * outers + 4 + uphill_swaps
    # This is (most of) the front layer, which is a bunch of outer qubits in the
    # coupling map.
    outer_pairs = [(i, n_qubits - i - 1) for i in range(outers)]
    inner_heuristic_peak = [
        # This gate is completely "inside" all the others in the front layer in
        # terms of the coupling map, so it's the only one that we can in theory
        # make progress towards without making the others worse.
        (outers + 1, outers + 2 + uphill_swaps),
        # These are the only two gates in the extended set, and they both get
        # further apart if you make a swap to bring the above gate closer
        # together, which is the trick that creates the "heuristic hill".
        (outers, outers + 1),
        (outers + 2 + uphill_swaps, outers + 3 + uphill_swaps),
    ]
    qc = QuantumCircuit(n_qubits)
    for pair in outer_pairs + inner_heuristic_peak:
        qc.cx(*pair)
    return qc


@ddt.ddt
class TestSabreSwap(QiskitTestCase):
    """Tests the SabreSwap pass."""

    def test_sabre_swap_pickle(self):
        """Test the pass can be pickled."""
        coupling = CouplingMap.from_ring(5)
        target = Target.from_configuration(["u", "cx"], coupling_map=coupling)
        sabre_swap = SabreSwap(target, "lookahead", seed=42, trials=1024)
        with io.BytesIO() as buf:
            pickle.dump(sabre_swap, buf)
            buf.seek(0)
            output = pickle.load(buf)
        self.assertIsInstance(output, SabreSwap)
        self.assertIsNone(output._routing_target)
        self.assertEqual(sabre_swap.heuristic, output.heuristic)
        self.assertEqual(sabre_swap.trials, output.trials)
        self.assertEqual(sabre_swap.fake_run, output.fake_run)

        test_circuit = QuantumCircuit(5)
        test_circuit.cx(0, 1)
        test_circuit.cx(0, 2)
        test_circuit.cx(0, 3)
        test_circuit.cx(0, 4)
        before_result = sabre_swap(test_circuit)
        with io.BytesIO() as buf:
            pickle.dump(sabre_swap, buf)
            buf.seek(0)
            output = pickle.load(buf)
        self.assertIsInstance(output, SabreSwap)
        self.assertIsNotNone(output._routing_target)
        self.assertEqual(sabre_swap.heuristic, output.heuristic)
        self.assertEqual(sabre_swap.trials, output.trials)
        self.assertEqual(sabre_swap.fake_run, output.fake_run)
        after_result = output(test_circuit)
        self.assertEqual(before_result, after_result)

    def test_sabre_swap_deepcopy(self):
        """Test the pass can be deepcopied."""
        coupling = CouplingMap.from_ring(5)
        target = Target.from_configuration(["u", "cx"], coupling_map=coupling)
        sabre_swap = SabreSwap(target, "lookahead", seed=42, trials=1024)
        output = deepcopy(sabre_swap)
        self.assertIsInstance(output, SabreSwap)
        self.assertIsNone(output._routing_target)
        self.assertEqual(sabre_swap.heuristic, output.heuristic)
        self.assertEqual(sabre_swap.trials, output.trials)
        self.assertEqual(sabre_swap.fake_run, output.fake_run)

        test_circuit = QuantumCircuit(5)
        test_circuit.cx(0, 1)
        test_circuit.cx(0, 2)
        test_circuit.cx(0, 3)
        test_circuit.cx(0, 4)
        before_result = sabre_swap(test_circuit)
        output = deepcopy(sabre_swap)
        self.assertIsInstance(output, SabreSwap)
        self.assertIsNotNone(output._routing_target)
        self.assertEqual(sabre_swap.heuristic, output.heuristic)
        self.assertEqual(sabre_swap.trials, output.trials)
        self.assertEqual(sabre_swap.fake_run, output.fake_run)
        after_result = output(test_circuit)
        self.assertEqual(before_result, after_result)

    def test_trivial_case(self):
        """Test that an already mapped circuit is unchanged.
                  ┌───┐┌───┐
        q_0: ──■──┤ H ├┤ X ├──■──
             ┌─┴─┐└───┘└─┬─┘  │
        q_1: ┤ X ├──■────■────┼──
             └───┘┌─┴─┐       │
        q_2: ──■──┤ X ├───────┼──
             ┌─┴─┐├───┤       │
        q_3: ┤ X ├┤ X ├───────┼──
             └───┘└─┬─┘     ┌─┴─┐
        q_4: ───────■───────┤ X ├
                            └───┘
        """
        coupling = CouplingMap.from_ring(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_2q_barriers_not_routed(self):
        """Test that a 2q barrier is not routed."""
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.barrier(0, 1)
        qc.barrier(0, 2)
        qc.barrier(0, 3)
        qc.barrier(2, 3)
        qc.h(0)
        qc.barrier(1, 2)
        qc.barrier(1, 0)
        qc.barrier(1, 3)
        qc.barrier(1, 4)
        qc.barrier(4, 3)
        qc.barrier(0, 4)

        passmanager = PassManager(SabreSwap(coupling, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_trivial_with_target(self):
        """Test that an already mapped circuit is unchanged with target."""
        coupling = CouplingMap.from_ring(5)
        target = Target(num_qubits=5)
        target.add_instruction(SwapGate(), dict.fromkeys(coupling.get_edges()))

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # F
        qc.cx(1, 0)
        qc.cx(4, 3)  # F
        qc.cx(0, 4)

        passmanager = PassManager(SabreSwap(target, "basic"))
        new_qc = passmanager.run(qc)

        self.assertEqual(new_qc, qc)

    def test_lookahead_mode(self):
        """Test lookahead mode's lookahead finds single SWAP gate.
                  ┌───┐
        q_0: ──■──┤ H ├───────────────
             ┌─┴─┐└───┘
        q_1: ┤ X ├──■────■─────────■──
             └───┘┌─┴─┐  │         │
        q_2: ──■──┤ X ├──┼────■────┼──
             ┌─┴─┐└───┘┌─┴─┐┌─┴─┐┌─┴─┐
        q_3: ┤ X ├─────┤ X ├┤ X ├┤ X ├
             └───┘     └───┘└───┘└───┘
        q_4: ─────────────────────────

        """
        coupling = CouplingMap.from_line(5)

        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(0, 1)  # free
        qc.cx(2, 3)  # free
        qc.h(0)  # free
        qc.cx(1, 2)  # free
        qc.cx(1, 3)  # F
        qc.cx(2, 3)  # E
        qc.cx(1, 3)  # E

        pm = PassManager(SabreSwap(coupling, "lookahead"))
        new_qc = pm.run(qc)

        self.assertEqual(new_qc.num_nonlocal_gates(), 7)

    def test_do_not_change_cm(self):
        """Coupling map should not change.
        See https://github.com/Qiskit/qiskit-terra/issues/5675"""
        cm_edges = [(1, 0), (2, 0), (2, 1), (3, 2), (3, 4), (4, 2)]
        coupling = CouplingMap(cm_edges)

        passmanager = PassManager(SabreSwap(coupling))
        _ = passmanager.run(QuantumCircuit(coupling.size()))

        self.assertEqual(set(cm_edges), set(coupling.get_edges()))

    def test_do_not_reorder_measurements(self):
        """Test that SabreSwap doesn't reorder measurements to the same classical bit.

        With the particular coupling map used in this test and the 3q ccx gate, the routing would
        invariably the measurements if the classical successors are not accurately tracked.
        Regression test of gh-7950."""
        coupling = CouplingMap([(0, 2), (2, 0), (1, 2), (2, 1)])
        qc = QuantumCircuit(3, 1)
        qc.compose(CCXGate().definition, [0, 1, 2], [])  # Unroll CCX to 2q operations.
        qc.h(0)
        qc.barrier()
        qc.measure(0, 0)  # This measure is 50/50 between the Z states.
        qc.measure(1, 0)  # This measure always overwrites with 0.
        passmanager = PassManager(SabreSwap(coupling))
        transpiled = passmanager.run(qc)

        last_h = transpiled.data[-4]
        self.assertIsInstance(last_h.operation, HGate)
        first_measure = transpiled.data[-2]
        second_measure = transpiled.data[-1]
        self.assertIsInstance(first_measure.operation, Measure)
        self.assertIsInstance(second_measure.operation, Measure)
        # Assert that the first measure is on the same qubit that the HGate was applied to, and the
        # second measurement is on a different qubit (though we don't care which exactly - that
        # depends a little on the randomization of the pass).
        self.assertEqual(last_h.qubits, first_measure.qubits)
        self.assertNotEqual(last_h.qubits, second_measure.qubits)

    def test_no_infinite_loop(self):
        """Test that the 'release value' mechanisms allow SabreSwap to make progress even on
        circuits that get stuck in a stable local minimum of the lookahead parameters."""
        # This is an approximation to the "lookahead" heuristic described by the original Sabre
        # paper.  For the particular circuit we're using, the layer-based lookahead Qiskit uses is
        # the same as the max-size depth-agnostic set of the original Sabre paper, since there's
        # only two (disjoint) gates that aren't in the front layer.  We use this here because
        # Qiskit's modern default heuristics don't get stuck with this circuit.
        heuristic = (
            Heuristic(attempt_limit=100)
            # The basic heuristic scaling by size is the problematic bit.
            .with_basic(1.0, SetScaling.Size).with_lookahead([0.5], SetScaling.Size)
        )
        qc = looping_circuit(3, 1)
        qc.measure_all()
        coupling_map = CouplingMap.from_line(qc.num_qubits)
        routing_pass = PassManager(SabreSwap(coupling_map, heuristic))

        # Since all the logic happens in Rust space these days, the best we'll really see here is
        # the test hanging.
        routed = routing_pass.run(qc)

        routed_ops = routed.count_ops()
        del routed_ops["swap"]
        self.assertEqual(routed_ops, qc.count_ops())
        couplings = {
            tuple(routed.find_bit(bit).index for bit in instruction.qubits)
            for instruction in routed.data
            if len(instruction.qubits) == 2
        }
        # Asserting equality to the empty set gives better errors on failure than asserting that
        # `couplings <= coupling_map`.
        self.assertEqual(couplings - set(coupling_map.get_edges()), set())

        # Assert that the same keys are produced by a simulation - this is a test that the inserted
        # swaps route the qubits correctly.
        if not optionals.HAS_AER:
            return

        from qiskit_aer import Aer

        sim = Aer.get_backend("aer_simulator")
        in_results = sim.run(qc, shots=4096).result().get_counts()
        out_results = sim.run(routed, shots=4096).result().get_counts()
        self.assertEqual(set(in_results), set(out_results))

    @ddt.data("basic", "lookahead", "decay")
    def test_deterministic(self, heuristic):
        """Test that the output of the SabreSwap pass is deterministic for a given random seed."""
        width = 40

        # The actual circuit is unimportant, we just need one with lots of scoring degeneracy.
        qc = QuantumCircuit(width)
        for i in range(width // 2):
            qc.cx(i, i + (width // 2))
        for i in range(0, width, 2):
            qc.cx(i, i + 1)
        dag = circuit_to_dag(qc)

        coupling = CouplingMap.from_line(width)
        pass_0 = SabreSwap(coupling, heuristic, seed=0, trials=1)
        pass_1 = SabreSwap(coupling, heuristic, seed=1, trials=1)
        dag_0 = pass_0.run(dag)
        dag_1 = pass_1.run(dag)

        # This deliberately avoids using a topological order, because that introduces an opportunity
        # for the re-ordering to sort the swaps back into a canonical order.
        def normalize_nodes(dag):
            return [(node.op.name, node.qargs, node.cargs) for node in dag.op_nodes()]

        # A sanity check for the test - if unequal seeds don't produce different outputs for this
        # degenerate circuit, then the test probably needs fixing (or Sabre is ignoring the seed).
        self.assertNotEqual(normalize_nodes(dag_0), normalize_nodes(dag_1))

        # Check that a re-run with the same seed produces the same circuit in the exact same order.
        self.assertEqual(normalize_nodes(dag_0), normalize_nodes(pass_0.run(dag)))

    def test_rejects_too_many_qubits(self):
        """Test that a sensible Python-space error message is emitted if the DAG has an incorrect
        number of qubits."""
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(5, "q"))
        with self.assertRaisesRegex(TranspilerError, "More qubits in the circuit"):
            pass_(qc)

    def test_rejects_too_few_qubits(self):
        """Test that a sensible Python-space error message is emitted if the DAG has an incorrect
        number of qubits."""
        pass_ = SabreSwap(CouplingMap.from_line(4))
        qc = QuantumCircuit(QuantumRegister(3, "q"))
        with self.assertRaisesRegex(TranspilerError, "Fewer qubits in the circuit"):
            pass_(qc)


@ddt.ddt
class TestSabreSwapControlFlow(QiskitTestCase):
    """Tests for control flow in sabre swap."""

    def test_shared_block(self):
        """Test multiple control flow ops sharing the same block instance."""
        inner = QuantumCircuit(2)
        inner.cx(0, 1)

        qreg = QuantumRegister(4, "q")
        outer = QuantumCircuit(qreg, ClassicalRegister(1))
        for pair in itertools.permutations(range(outer.num_qubits), 2):
            outer.if_test((outer.cregs[0], 1), inner, pair, [])

        coupling = CouplingMap.from_line(4)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(circuit_to_dag(outer))

        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_blocks_use_registers(self):
        """Test that control flow ops using registers still use registers after routing."""
        num_qubits = 2
        qreg = QuantumRegister(num_qubits, "q")
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(1)
        qc = QuantumCircuit(qreg, cr1, cr2)

        with qc.if_test((cr1, False)):
            qc.cx(0, 1)
            qc.measure(0, cr2[0])
            with qc.if_test((cr2, 0)):
                qc.cx(0, 1)

        coupling = CouplingMap.from_line(num_qubits)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(circuit_to_dag(qc))

        outer_if_op = cdag.op_nodes(ControlFlowOp)[0].op
        self.assertEqual(outer_if_op.condition[0], cr1)

        inner_if_op = circuit_to_dag(outer_if_op.blocks[0]).op_nodes(ControlFlowOp)[0].op
        self.assertEqual(inner_if_op.condition[0], cr2)

    def test_pre_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[2]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[2]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[2]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        etrue_body = QuantumCircuit(qreg, creg[[2]])
        etrue_body.x(3)
        efalse_body = QuantumCircuit(qreg, creg[[2]])
        efalse_body.x(4)
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg, creg[[2]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else_route_post_x(self):
        """test swap with if else controlflow construct; pre-cx and post x"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.measure(2, 2)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(3)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.if_else((creg[2], 0), true_body, false_body, qreg, creg[[0]])
        qc.x(1)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(1, 2)
        new_order = [0, 2, 1, 3, 4]
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.x(3)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.x(4)
        expected.if_else((creg[2], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.x(2)
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_post_if_else_route(self):
        """test swap with if else controlflow construct; post cx"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        true_body.x(0)
        false_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        false_body.x(1)
        qc.barrier(qreg)
        qc.if_else((creg[0], 0), true_body, false_body, qreg[[3, 4]], creg[[0]])
        qc.barrier(qreg)
        qc.cx(0, 2)
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[3, 4]], creg[[0]])
        efalse_body.x(1)
        expected.barrier(qreg)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[3, 4]], creg[[0]])
        expected.barrier(qreg)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_if_else2(self):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg[[0]], creg[[0]])
        true_body.x(0)
        false_body = QuantumCircuit(qreg[[0]], creg[[0]])
        qc.if_else((creg[0], 0), true_body, false_body, qreg[[0]], creg[[0]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg[[0]], creg[[0]])
        etrue_body.x(0)
        efalse_body = QuantumCircuit(qreg[[0]], creg[[0]])
        new_order = [0, 2, 1, 3, 4]
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg[[0]], creg[[0]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[new_order])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_intra_if_else_route(self):
        """test swap with if else controlflow construct"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.swap(1, 2)
        etrue_body.cx(0, 1)
        etrue_body.swap(1, 2)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.swap(0, 1)
        efalse_body.swap(3, 4)
        efalse_body.swap(2, 3)
        efalse_body.cx(1, 2)
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        efalse_body.swap(3, 4)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_intra_if_else(self):
        """test swap with if else controlflow construct; cx in if statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap([(i, i + 1) for i in range(num_qubits - 1)])
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)

        etrue_body.cx(0, 1)

        efalse_body.swap(0, 1)
        efalse_body.swap(3, 4)
        efalse_body.swap(2, 3)
        efalse_body.cx(1, 2)
        efalse_body.swap(0, 1)
        efalse_body.swap(2, 3)
        efalse_body.swap(3, 4)

        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_pre_intra_post_if_else(self):
        """test swap with if else controlflow construct; cx before, in, and after if
        statement"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg[:], creg[[0]])
        true_body.cx(0, 2)
        false_body = QuantumCircuit(qreg[:], creg[[0]])
        false_body.cx(0, 4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.h(3)
        qc.cx(3, 0)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(0, 1)
        expected.cx(1, 2)
        expected.measure(1, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.cx(1, 2)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.swap(1, 2)
        efalse_body.swap(3, 4)
        efalse_body.cx(2, 3)
        efalse_body.swap(1, 2)
        efalse_body.swap(3, 4)

        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.h(3)
        expected.swap(2, 3)
        expected.cx(2, 1)
        expected.barrier()
        expected.measure(qreg[[1, 0, 3, 2, 4]], creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_if_expr(self):
        """Test simple if conditional with an `Expr` condition."""
        coupling = CouplingMap.from_line(4)

        body = QuantumCircuit(4)
        body.cx(0, 1)
        body.cx(0, 2)
        body.cx(0, 3)
        qc = QuantumCircuit(4, 2)
        qc.if_test(expr.logic_and(qc.clbits[0], qc.clbits[1]), body, [0, 1, 2, 3], [])

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=58, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_if_else_expr(self):
        """Test simple if/else conditional with an `Expr` condition."""
        coupling = CouplingMap.from_line(4)

        true = QuantumCircuit(4)
        true.cx(0, 1)
        true.cx(0, 2)
        true.cx(0, 3)
        false = QuantumCircuit(4)
        false.cx(3, 0)
        false.cx(3, 1)
        false.cx(3, 2)
        qc = QuantumCircuit(4, 2)
        qc.if_else(expr.logic_and(qc.clbits[0], qc.clbits[1]), true, false, [0, 1, 2, 3], [])

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=58, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_no_layout_change(self):
        """test controlflow with no layout change needed"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.cx(0, 2)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(2)
        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.x(4)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.barrier(qreg)
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.swap(1, 2)
        expected.cx(0, 1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.x(1)
        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.x(4)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.barrier(qreg)
        expected.measure(qreg, creg[[0, 2, 1, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    @ddt.data(1, 2, 3)
    def test_for_loop(self, nloops):
        """test stochastic swap with for_loop"""
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        for_body = QuantumCircuit(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        qc.for_loop(range(nloops), loop_parameter, for_body, qreg, [])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        efor_body = QuantumCircuit(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        loop_parameter = None
        expected.for_loop(range(nloops), loop_parameter, efor_body, qreg, [])
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_while_loop(self):
        """test while loop"""
        num_qubits = 4
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(len(qreg))
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        while_body = QuantumCircuit(qreg, creg)
        while_body.reset(qreg[2:])
        while_body.h(qreg[2:])
        while_body.cx(0, 3)
        while_body.measure(qreg[3], creg[3])
        qc.while_loop((creg, 0), while_body, qc.qubits, qc.clbits)
        qc.barrier()
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        ewhile_body = QuantumCircuit(qreg, creg)
        ewhile_body.reset(qreg[2:])
        ewhile_body.h(qreg[2:])
        ewhile_body.swap(0, 1)
        ewhile_body.swap(2, 3)
        ewhile_body.cx(1, 2)
        ewhile_body.measure(qreg[2], creg[3])
        ewhile_body.swap(1, 0)
        ewhile_body.swap(3, 2)
        expected.while_loop((creg, 0), ewhile_body, expected.qubits, expected.clbits)
        expected.barrier()
        expected.measure(qreg, creg)
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_while_loop_expr(self):
        """Test simple while loop with an `Expr` condition."""
        coupling = CouplingMap.from_line(4)

        body = QuantumCircuit(4)
        body.cx(0, 1)
        body.cx(0, 2)
        body.cx(0, 3)
        qc = QuantumCircuit(4, 2)
        qc.while_loop(expr.logic_and(qc.clbits[0], qc.clbits[1]), body, [0, 1, 2, 3], [])

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

    def test_switch_implicit_carg_use(self):
        """Test that a switch statement that uses cargs only implicitly via its ``target`` attribute
        and not explicitly in bodies of the cases is routed correctly, with the dependencies
        fulfilled correctly."""
        coupling = CouplingMap.from_line(4)
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)

        body = QuantumCircuit([Qubit()])
        body.x(0)

        # If the classical wire condition isn't respected, then the switch would appear in the front
        # layer and be immediately eligible for routing, which would produce invalid output.
        qc = QuantumCircuit(4, 1)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(0, 2)
        qc.measure(2, 0)
        qc.switch(expr.lift(qc.clbits[0]), [(False, body.copy()), (True, body.copy())], [3], [])

        expected = QuantumCircuit(4, 1)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.swap(2, 1)
        expected.cx(0, 1)
        expected.measure(1, 0)
        expected.switch(
            expr.lift(expected.clbits[0]), [(False, body.copy()), (True, body.copy())], [3], []
        )

        self.assertEqual(pass_(qc), expected)

    def test_switch_single_case(self):
        """Test routing of 'switch' with just a single case."""
        qreg = QuantumRegister(5, "q")
        creg = ClassicalRegister(3, "c")
        qc = QuantumCircuit(qreg, creg)

        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        qc.switch(creg, [(0, case0)], qreg[[0, 1, 2]], creg)

        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)
        test = pass_(qc)

        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        expected.switch(creg, [(0, case0)], qreg[[0, 1, 2]], creg[:])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_nonexhaustive(self):
        """Test routing of 'switch' with several but nonexhaustive cases."""
        qreg = QuantumRegister(5, "q")
        creg = ClassicalRegister(3, "c")

        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.cx(3, 1)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.cx(4, 2)
        qc.switch(creg, [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)

        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)
        test = pass_(qc)

        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.swap(1, 2)
        case1.cx(3, 2)
        case1.swap(1, 2)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.swap(2, 3)
        case2.cx(4, 3)
        case2.swap(2, 3)
        expected.switch(creg, [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_single_case(self):
        """Test routing of 'switch' with an `Expr` target and just a single case."""
        qreg = QuantumRegister(5, "q")
        creg = ClassicalRegister(3, "c")
        qc = QuantumCircuit(qreg, creg)

        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        qc.switch(expr.bit_or(creg, 5), [(0, case0)], qreg[[0, 1, 2]], creg)

        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)
        test = pass_(qc)

        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg[[0, 1, 2]], creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        expected.switch(expr.bit_or(creg, 5), [(0, case0)], qreg[[0, 1, 2]], creg[:])

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_switch_expr_nonexhaustive(self):
        """Test routing of 'switch' with an `Expr` target and several but nonexhaustive cases."""
        qreg = QuantumRegister(5, "q")
        creg = ClassicalRegister(3, "c")

        qc = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.cx(2, 0)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.cx(3, 1)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.cx(4, 2)
        qc.switch(expr.bit_or(creg, 5), [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)

        coupling = CouplingMap.from_line(len(qreg))
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)
        test = pass_(qc)

        check = CheckMap(coupling)
        check(test)
        self.assertTrue(check.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        case0 = QuantumCircuit(qreg, creg[:])
        case0.cx(0, 1)
        case0.cx(1, 2)
        case0.swap(0, 1)
        case0.cx(2, 1)
        case0.swap(0, 1)
        case1 = QuantumCircuit(qreg, creg[:])
        case1.cx(1, 2)
        case1.cx(2, 3)
        case1.swap(1, 2)
        case1.cx(3, 2)
        case1.swap(1, 2)
        case2 = QuantumCircuit(qreg, creg[:])
        case2.cx(2, 3)
        case2.cx(3, 4)
        case2.swap(2, 3)
        case2.cx(4, 3)
        case2.swap(2, 3)
        expected.switch(expr.bit_or(creg, 5), [(0, case0), ((1, 2), case1), (3, case2)], qreg, creg)

        self.assertEqual(canonicalize_control_flow(test), canonicalize_control_flow(expected))

    def test_nested_inner_cnot(self):
        """test swap in nested if else controlflow construct; swap in inner"""
        num_qubits = 3
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.x(0)

        for_body = QuantumCircuit(qreg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(0, 2)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, [])

        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.x(0)

        efor_body = QuantumCircuit(qreg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg, [])

        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg)

        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_nested_outer_cnot(self):
        """test swap with nested if else controlflow construct; swap in outer"""
        num_qubits = 5
        qreg = QuantumRegister(num_qubits, "q")
        creg = ClassicalRegister(num_qubits)
        coupling = CouplingMap.from_line(num_qubits)
        qc = QuantumCircuit(qreg, creg)
        qc.h(0)
        qc.x(1)
        qc.measure(0, 0)
        true_body = QuantumCircuit(qreg, creg[[0]])
        true_body.cx(0, 2)
        true_body.x(0)

        for_body = QuantumCircuit(qreg)
        for_body.delay(10, 0)
        for_body.barrier(qreg)
        for_body.cx(1, 3)
        loop_parameter = None
        true_body.for_loop(range(3), loop_parameter, for_body, qreg, [])

        false_body = QuantumCircuit(qreg, creg[[0]])
        false_body.y(0)
        qc.if_else((creg[0], 0), true_body, false_body, qreg, creg[[0]])
        qc.measure(qreg, creg)

        dag = circuit_to_dag(qc)
        cdag = SabreSwap(coupling, "lookahead", seed=82, trials=1).run(dag)
        check_map_pass = CheckMap(coupling)
        check_map_pass.run(cdag)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qreg, creg)
        expected.h(0)
        expected.x(1)
        expected.measure(0, 0)
        etrue_body = QuantumCircuit(qreg, creg[[0]])
        etrue_body.swap(1, 2)
        etrue_body.cx(0, 1)
        etrue_body.x(0)

        efor_body = QuantumCircuit(qreg)
        efor_body.delay(10, 0)
        efor_body.barrier(qreg)
        efor_body.cx(2, 3)

        etrue_body.for_loop(range(3), loop_parameter, efor_body, qreg[[0, 1, 2, 3, 4]], [])
        etrue_body.swap(1, 2)

        efalse_body = QuantumCircuit(qreg, creg[[0]])
        efalse_body.y(0)
        expected.if_else((creg[0], 0), etrue_body, efalse_body, qreg, creg[[0]])
        expected.measure(qreg, creg[[0, 1, 2, 3, 4]])
        self.assertEqual(dag_to_circuit(cdag), expected)

    def test_disjoint_looping(self):
        """Test looping controlflow on different qubit register"""
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        loop_body = QuantumCircuit(2)
        loop_body.cx(0, 1)
        qc.for_loop((0,), None, loop_body, [0, 2], [])
        cqc = SabreSwap(cm, "lookahead", seed=82, trials=1)(qc)

        expected = QuantumCircuit(qr)
        efor_body = QuantumCircuit(qr[[0, 1, 2]])
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(1, 2)
        expected.for_loop((0,), None, efor_body, [0, 1, 2], [])
        self.assertEqual(cqc, expected)

    def test_disjoint_multiblock(self):
        """Test looping controlflow on different qubit register"""
        num_qubits = 4
        cm = CouplingMap.from_line(num_qubits)
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        true_body = QuantumCircuit(qr[0:3] + cr[:])
        true_body.cx(0, 1)
        false_body = QuantumCircuit(qr[0:3] + cr[:])
        false_body.cx(0, 2)
        qc.if_else((cr[0], 1), true_body, false_body, [0, 1, 2], [0])
        cqc = SabreSwap(cm, "lookahead", seed=82, trials=1)(qc)

        expected = QuantumCircuit(qr, cr)
        etrue_body = QuantumCircuit(qr[[0, 1, 2]], cr[[0]])
        etrue_body.cx(0, 1)
        efalse_body = QuantumCircuit(qr[[0, 1, 2]], cr[[0]])
        efalse_body.swap(1, 2)
        efalse_body.cx(0, 1)
        efalse_body.swap(1, 2)
        expected.if_else((cr[0], 1), etrue_body, efalse_body, [0, 1, 2], cr[[0]])
        self.assertEqual(cqc, expected)

    def test_multiple_ops_per_layer(self):
        """Test circuits with multiple operations per layer"""
        num_qubits = 6
        coupling = CouplingMap.from_line(num_qubits)
        check_map_pass = CheckMap(coupling)
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)
        # This cx and the for_loop are in the same layer.
        qc.cx(0, 2)
        with qc.for_loop((0,)):
            qc.cx(3, 5)
        cqc = SabreSwap(coupling, "lookahead", seed=82, trials=1)(qc)
        check_map_pass(cqc)
        self.assertTrue(check_map_pass.property_set["is_swap_mapped"])

        expected = QuantumCircuit(qr)
        expected.swap(1, 2)
        expected.cx(0, 1)
        efor_body = QuantumCircuit(qr[[3, 4, 5]])
        efor_body.swap(1, 2)
        efor_body.cx(0, 1)
        efor_body.swap(2, 1)
        expected.for_loop((0,), None, efor_body, [3, 4, 5], [])
        self.assertEqual(cqc, expected)

    def test_if_no_else_restores_layout(self):
        """Test that an if block with no else branch restores the initial layout."""
        qc = QuantumCircuit(8, 1)
        with qc.if_test((qc.clbits[0], False)):
            # Just some arbitrary gates with no perfect layout.
            qc.cx(3, 5)
            qc.cx(4, 6)
            qc.cx(1, 4)
            qc.cx(7, 4)
            qc.cx(0, 5)
            qc.cx(7, 3)
            qc.cx(1, 3)
            qc.cx(5, 2)
            qc.cx(6, 7)
            qc.cx(3, 2)
            qc.cx(6, 2)
            qc.cx(2, 0)
            qc.cx(7, 6)
        coupling = CouplingMap.from_line(8)
        pass_ = SabreSwap(coupling, "lookahead", seed=82, trials=1)
        transpiled = pass_(qc)

        # Check the pass claims to have done things right.
        initial_layout = Layout.generate_trivial_layout(*qc.qubits)
        self.assertEqual(initial_layout, pass_.property_set["final_layout"])

        # Check that pass really did do it right.
        inner_block = transpiled.data[0].operation.blocks[0]
        running_layout = initial_layout.copy()
        for instruction in inner_block:
            if instruction.operation.name == "swap":
                running_layout.swap(*instruction.qubits)
        self.assertEqual(initial_layout, running_layout)

    def test_idle_qubit_contraction(self):
        """Incident virtual qubits to a control-flow block should be maintained, even if idle, but
        the blocks shouldn't contain further unnecessary qubits."""
        qc = QuantumCircuit(8)
        with qc.if_test(expr.lift(True)):
            qc.cx(0, 3)
            qc.noop(4)
            # Both of these qubits will have been moved around by the prior necessary layout
            # changes, so this is testing the recursion works for modified layouts.
            with qc.if_test(expr.lift(True)) as else_:
                qc.noop(0)
            with else_:
                qc.noop(3)

        coupling = CouplingMap.from_line(8)

        # With the `decay` heuristic set to penalise re-use of the same qubit swap, this expected
        # circuit should be the only valid output (except for symmetries in the swap operation,
        # which the equality check should handle).
        expected = QuantumCircuit(8)
        with expected.if_test(expr.lift(True)):
            expected.noop(4)
            expected.swap(0, 1)
            expected.swap(2, 3)
            expected.cx(1, 2)
            with expected.if_test(expr.lift(True)) as else_:
                expected.noop(1)
            with else_:
                expected.noop(2)
            # We have to restore the output layout.
            expected.swap(0, 1)
            expected.swap(2, 3)

        pass_ = SabreSwap(coupling, "decay", seed=2025_02_05, trials=1)
        self.assertEqual(pass_(qc), expected)

    def test_nested_vars(self):
        """The Sabre rebuilder shouldn't choke if there is `Var` or `Stretch` usage within a
        control-flow block."""
        qc = QuantumCircuit(4)
        a = qc.add_input("a", types.Bool())
        b = qc.add_var("b", False)
        stretch_0 = qc.add_stretch("c")
        for other in qc.qubits[1:]:
            qc.cx(0, other)
        qc.delay(stretch_0, 0)
        with qc.if_test(a):  # block 1
            d = qc.add_var("d", False)
            stretch_1 = qc.add_stretch("e")
            for other in qc.qubits[1:]:
                qc.cx(0, other)
            with qc.while_loop(expr.logic_and(b, d)):  # block 2
                for other in qc.qubits[1:]:
                    qc.cx(0, other)
                qc.delay(stretch_1)

        # We don't care about the routing, just that the stretches and vars are there.
        out = SabreSwap(CouplingMap.from_line(4), heuristic="basic", seed=0, trials=1)(qc)

        def extract_vars(circuit):
            """Extract the variables and the types of variables from a circuit and contained
            control-flow blocks.  We assume that each block contains at most 1 control-flow block
            with disparate names, just for ease."""

            def extract_local(block):
                return {
                    "inputs": set(block.iter_input_vars()),
                    "captures": set(block.iter_captures()),
                    "local vars": set(block.iter_declared_vars()),
                    "local stretches": set(block.iter_declared_stretches()),
                }

            blocks = (
                ("global", circuit),
                ("if", (if_body := circuit.data[-1].operation.blocks[0])),
                ("while", if_body.data[-1].operation.blocks[0]),
            )
            return {name: extract_local(block) for name, block in blocks}

        self.assertEqual(extract_vars(qc), extract_vars(out))


@ddt.ddt
class TestSabreSwapRandomCircuitValidOutput(QiskitTestCase):
    """Assert the output of a transpilation with stochastic swap is a physical circuit."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.backend = GenericBackendV2(
            num_qubits=27,
            control_flow=True,
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        cls.coupling_edge_set = {tuple(x) for x in cls.backend.coupling_map}
        cls.basis_gates = set(cls.backend.operation_names)

    def assert_valid_circuit(self, transpiled):
        """Assert circuit complies with constraints of backend."""
        self.assertIsInstance(transpiled, QuantumCircuit)
        self.assertIsNotNone(getattr(transpiled, "_layout", None))

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name in {"barrier", "measure"}:
                    continue
                self.assertIn(instruction.operation.name, self.basis_gates)
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                if not isinstance(instruction.operation, ControlFlowOp):
                    if len(qargs) > 2 or len(qargs) < 0:
                        raise RuntimeError("Invalid number of qargs for instruction")
                    if len(qargs) == 2:
                        self.assertIn(qargs, self.coupling_edge_set)
                    else:
                        self.assertLessEqual(qargs[0], 26)
                else:
                    for block in instruction.operation.blocks:
                        self.assertEqual(block.num_qubits, len(instruction.qubits))
                        self.assertEqual(block.num_clbits, len(instruction.clbits))
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        # Assert routing ran.
        _visit_block(
            transpiled,
            qubit_mapping={qubit: index for index, qubit in enumerate(transpiled.qubits)},
        )

    @ddt.data(*range(1, 27))
    def test_random_circuit_no_control_flow(self, size):
        """Test that transpiled random circuits without control flow are physical circuits."""
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        tqc = transpile(
            circuit,
            self.backend,
            routing_method="sabre",
            layout_method="sabre",
            seed_transpiler=12342,
        )
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(1, 27))
    def test_random_circuit_no_control_flow_target(self, size):
        """Test that transpiled random circuits without control flow are physical circuits."""
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        tqc = transpile(
            circuit,
            routing_method="sabre",
            layout_method="sabre",
            seed_transpiler=12342,
            target=GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP, seed=42).target,
        )
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(4, 27))
    def test_random_circuit_for_loop(self, size):
        """Test that transpiled random circuits with nested for loops are physical circuits."""
        circuit = random_circuit(size, 3, measure=False, seed=12342)
        for_block = random_circuit(3, 2, measure=False, seed=12342)
        inner_for_block = random_circuit(2, 1, measure=False, seed=12342)
        with circuit.for_loop((1,)):
            with circuit.for_loop((1,)):
                circuit.append(inner_for_block, [0, 3])
            circuit.append(for_block, [1, 0, 2])
        circuit.measure_all()

        tqc = transpile(
            circuit,
            self.backend,
            basis_gates=list(self.basis_gates),
            routing_method="sabre",
            layout_method="sabre",
            seed_transpiler=12342,
        )
        self.assert_valid_circuit(tqc)

    @ddt.data(*range(6, 27))
    def test_random_circuit_if_else(self, size):
        """Test that transpiled random circuits with if else blocks are physical circuits."""
        circuit = random_circuit(size, 3, measure=True, seed=12342)
        if_block = random_circuit(3, 2, measure=True, seed=12342)
        else_block = random_circuit(2, 1, measure=True, seed=12342)

        rng = numpy.random.default_rng(seed=12342)
        inner_clbit_count = max((if_block.num_clbits, else_block.num_clbits))
        if inner_clbit_count > circuit.num_clbits:
            circuit.add_bits([Clbit() for _ in [None] * (inner_clbit_count - circuit.num_clbits)])
        clbit_indices = list(range(circuit.num_clbits))
        rng.shuffle(clbit_indices)

        with circuit.if_test((circuit.clbits[0], True)) as else_:
            circuit.append(if_block, [0, 2, 1], clbit_indices[: if_block.num_clbits])
        with else_:
            circuit.append(else_block, [2, 5], clbit_indices[: else_block.num_clbits])

        tqc = transpile(
            circuit,
            self.backend,
            basis_gates=list(self.basis_gates),
            routing_method="sabre",
            layout_method="sabre",
            seed_transpiler=12342,
        )
        self.assert_valid_circuit(tqc)


if __name__ == "__main__":
    unittest.main()
