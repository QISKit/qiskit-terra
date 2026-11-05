# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the VF2Layout pass"""

import io
import pickle
import unittest
from math import pi

import ddt
import numpy
import rustworkx

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.circuit import ControlFlowOp, Qubit, library as lib, Parameter
from qiskit.transpiler import CouplingMap, Target, TranspilerError
from qiskit.transpiler.passes import ApplyLayout
from qiskit.transpiler.passes.layout.vf2_layout import VF2Layout, VF2LayoutStopReason
from qiskit._accelerate.error_map import ErrorMap
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit import Measure
from qiskit.circuit.library import GraphStateGate, CXGate, XGate, HGate
from qiskit.transpiler import PassManager, AnalysisPass
from qiskit.transpiler.target import InstructionProperties
from qiskit.transpiler.preset_passmanagers.common import generate_embed_passmanager
from test import QiskitTestCase, combine

from ..legacy_cmaps import TENERIFE_CMAP, RUESCHLIKON_CMAP, MANHATTAN_CMAP, YORKTOWN_CMAP


class LayoutTestCase(QiskitTestCase):
    """VF2Layout assertions"""

    seed = 42

    def assertLayout(self, dag, coupling_map, property_set, strict_direction=False):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND)

        layout = property_set["layout"]
        edges = coupling_map.graph.edge_list()

        def run(dag, wire_map):
            for gate in dag.two_qubit_ops():
                if isinstance(gate.op, ControlFlowOp):
                    continue
                physical_q0 = wire_map[gate.qargs[0]]
                physical_q1 = wire_map[gate.qargs[1]]

                if strict_direction:
                    result = (physical_q0, physical_q1) in edges
                else:
                    result = (physical_q0, physical_q1) in edges or (
                        physical_q1,
                        physical_q0,
                    ) in edges
                self.assertTrue(result)
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    run(circuit_to_dag(block), inner_wire_map)

        run(dag, {bit: layout[bit] for bit in dag.qubits})


@ddt.ddt
class TestVF2LayoutSimple(LayoutTestCase):
    """Tests the VF2Layout pass"""

    def test_1q_component_influence(self):
        """Assert that the 1q component of a connected interaction graph is scored correctly."""
        target = Target()
        target.add_instruction(
            CXGate(),
            {
                (0, 1): InstructionProperties(error=0.0),
                (1, 2): InstructionProperties(error=0.0),
                (2, 3): InstructionProperties(error=0.0),
            },
        )
        target.add_instruction(
            HGate(),
            {
                (0,): InstructionProperties(error=0.0),
                (1,): InstructionProperties(error=0.0),
                (2,): InstructionProperties(error=0.0),
            },
        )
        target.add_instruction(
            Measure(),
            {
                (0,): InstructionProperties(error=0.1),
                (1,): InstructionProperties(error=0.1),
                (2,): InstructionProperties(error=0.9),
            },
        )

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.measure(0, 0)
        qc.measure(1, 1)
        vf2_pass = VF2Layout(target=target, seed=self.seed)
        vf2_pass(qc)
        layout = vf2_pass.property_set["layout"]
        self.assertNotIn(2, layout.get_physical_bits())

    def test_2q_circuit_2q_coupling(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        cmap = CouplingMap([[0, 1]])

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)

    def test_2q_circuit_2q_coupling_sd(self):
        """A simple example, considering the direction
         0  -> 1
        qr1 -> qr0
        """
        cmap = CouplingMap([[0, 1]])

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set, strict_direction=True)

    @ddt.data(True, False)
    def test_2q_circuit_simple_control_flow(self, strict_direction):
        """Test that simple control-flow can be routed on a 2q coupling map."""
        cmap = CouplingMap([(0, 1)])
        circuit = QuantumCircuit(2)
        with circuit.for_loop((1,)):
            circuit.cx(1, 0)
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=strict_direction, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set, strict_direction=strict_direction)

    @ddt.data(True, False)
    def test_2q_circuit_nested_control_flow(self, strict_direction):
        """Test that simple control-flow can be routed on a 2q coupling map."""
        cmap = CouplingMap([(0, 1)])
        circuit = QuantumCircuit(2, 1)
        with circuit.while_loop((circuit.clbits[0], True)):
            with circuit.if_test((circuit.clbits[0], True)) as else_:
                circuit.cx(1, 0)
            with else_:
                circuit.cx(1, 0)
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, strict_direction=strict_direction, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set, strict_direction=strict_direction)

    def test_3q_circuit_3q_coupling_non_induced(self):
        """A simple example, check for non-induced subgraph
            1         qr0 -> qr1 -> qr2
           / \
          0 - 2
        """
        cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)

    def test_3q_circuit_3q_coupling_non_induced_control_flow(self):
        r"""A simple example, check for non-induced subgraph
            1         qr0 -> qr1 -> qr2
           / \
          0 - 2
        """
        cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])

        circuit = QuantumCircuit(3, 1)
        with circuit.for_loop((1,)):
            circuit.cx(0, 1)  # qr0-> qr1
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            pass
        with else_:
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(1, 2)  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap, pass_.property_set)

    def test_call_limit(self):
        """Test that call limit is enforce."""
        cmap = CouplingMap([[0, 1], [1, 2], [2, 0]])

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, call_limit=1)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.NO_SOLUTION_FOUND
        )

    def test_coupling_map_and_target(self):
        """Test that a Target is used instead of a CouplingMap if both are specified."""
        cmap = CouplingMap([[0, 1], [1, 2]])
        target = Target()
        target.add_instruction(CXGate(), {(0, 1): None, (1, 2): None, (1, 0): None})
        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])  # qr0-> qr1
        circuit.cx(qr[1], qr[2])  # qr1-> qr2
        circuit.cx(qr[1], qr[0])  # qr1-> qr0
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap, seed=-1, max_trials=1, target=target)
        pass_.run(dag)
        self.assertLayout(dag, target.build_coupling_map(), pass_.property_set)

    def test_neither_coupling_map_or_target(self):
        """Test that we raise if neither a target or coupling map is specified."""
        vf2_pass = VF2Layout(seed=123, call_limit=1000, time_limit=20, max_trials=7)
        circuit = QuantumCircuit(2)
        dag = circuit_to_dag(circuit)
        with self.assertRaises(TranspilerError):
            vf2_pass.run(dag)

    def test_target_no_error(self):
        """Test that running vf2layout on a pass against a target with no error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2Layout(target=target)
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertLayout(dag, target.build_coupling_map(), vf2_pass.property_set)

    def test_target_some_error(self):
        """Test that running vf2layout on a pass against a target with some error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(
            XGate(), {(i,): InstructionProperties(error=0.00123) for i in range(n_qubits)}
        )
        target.add_instruction(CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2Layout(target=target)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertLayout(dag, target.build_coupling_map(), vf2_pass.property_set)

    def test_determinism_all_1q(self):
        """Test that running vf2layout on a circuit with all single qubit gates is deterministic."""

        circ = QuantumCircuit(3)
        for i in range(3):
            circ.rx(3.14159, i)
        circ.measure_all()

        backend = GenericBackendV2(10, noise_info=True, seed=123456789)
        layouts = []
        for _ in range(10):
            layout_pass = VF2Layout(target=backend.target)
            property_set = {}
            layout_pass(circ, property_set=property_set)
            layouts.append(property_set["layout"])
        self.assertEqual(10, len(layouts), "Expected 10 layouts from 10 pass executions")
        for i, layout in enumerate(layouts):
            self.assertIsNotNone(layout, f"A layout was not found for layout {i}")
            self.assertEqual(
                layouts[0], layout, f"Layout for execution {i} differs from the expected"
            )

    @combine(
        seed=(-1, 12),  # This hits both the "seeded" and "unseeded" paths.
        strict_direction=(True, False),
    )
    def test_complete_layout_with_idle_qubits(self, seed, strict_direction):
        """Test that completely idle qubits are included in the resulting layout."""
        # Use registerless qubits to avoid any register-based shenangigans from adding the bits
        # automatically.
        qc = QuantumCircuit([Qubit() for _ in range(3)])
        qc.cx(0, 1)
        target = Target.from_configuration(
            num_qubits=3, basis_gates=["sx", "rz", "cx"], coupling_map=CouplingMap.from_line(3)
        )
        property_set = {}
        pass_ = VF2Layout(target=target, seed=seed, strict_direction=strict_direction)
        pass_(qc, property_set=property_set)
        unallocated = {i for i, bit in enumerate(qc.qubits) if bit not in property_set["layout"]}
        self.assertEqual(unallocated, set())


@ddt.ddt
class TestVF2LayoutLattice(LayoutTestCase):
    """Fit in 25x25 hexagonal lattice coupling map"""

    cmap25 = CouplingMap.from_hexagonal_lattice(25, 25, bidirectional=False)

    def graph_state_from_pygraph(self, graph):
        """Creates a GraphStateGate circuit from a PyGraph"""
        adjacency_matrix = rustworkx.adjacency_matrix(graph)
        return GraphStateGate(adjacency_matrix).definition

    def test_hexagonal_lattice_graph_20_in_25(self):
        """A 20x20 interaction map in 25x25 coupling map"""
        graph_20_20 = rustworkx.generators.hexagonal_lattice_graph(20, 20)
        circuit = self.graph_state_from_pygraph(graph_20_20)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)

    def test_hexagonal_lattice_graph_9_in_25(self):
        """A 9x9 interaction map in 25x25 coupling map"""
        graph_9_9 = rustworkx.generators.hexagonal_lattice_graph(9, 9)
        circuit = self.graph_state_from_pygraph(graph_9_9)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)

    @ddt.data(True, False)
    def test_hexagonal_lattice_graph_9_in_25_no_trial_limit(self, strict_direction):
        """A 9x9 interaction map in 25x25 coupling map"""
        graph_9_9 = rustworkx.generators.hexagonal_lattice_graph(9, 9)
        circuit = self.graph_state_from_pygraph(graph_9_9)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=-1, max_trials=-1, strict_direction=strict_direction)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)

    @ddt.data(True, False)
    def test_hexagonal_lattice_graph_9_in_25_default_trial_limit(self, strict_direction):
        """A 9x9 interaction map in 25x25 coupling map"""
        graph_9_9 = rustworkx.generators.hexagonal_lattice_graph(9, 9)
        circuit = self.graph_state_from_pygraph(graph_9_9)

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(self.cmap25, seed=-1, max_trials=None, strict_direction=strict_direction)
        pass_.run(dag)
        self.assertLayout(dag, self.cmap25, pass_.property_set)


class TestVF2LayoutBackend(LayoutTestCase):
    """Tests VF2Layout against backends"""

    def test_5q_circuit_Rueschlikon_no_solution(self):
        """5 qubits in Rueschlikon, no solution

        q0[1] ↖     ↗ q0[2]
               q0[0]
        q0[3] ↙     ↘ q0[4]
        """
        cmap16 = RUESCHLIKON_CMAP

        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[0], qr[4])
        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(CouplingMap(cmap16), seed=self.seed, max_trials=1)
        pass_.run(dag)
        layout = pass_.property_set["layout"]
        self.assertIsNone(layout)
        self.assertEqual(
            pass_.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.NO_SOLUTION_FOUND
        )

    def test_9q_circuit_Rueschlikon_sd(self):
        """9 qubits in Rueschlikon, considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9
        """
        cmap16 = CouplingMap(RUESCHLIKON_CMAP)

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap16, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap16, pass_.property_set)

    def test_4q_circuit_Tenerife_loose_nodes(self):
        """4 qubits in Tenerife, with loose nodes

            1
          ↙ ↑
        0 ← 2 ← 3
            ↑ ↙
            4
        """
        cmap5 = CouplingMap(TENERIFE_CMAP)

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set)

    def test_3q_circuit_Tenerife_sd(self):
        """3 qubits in Tenerife, considering the direction
            1                       1
          ↙ ↑                    ↙  ↑
        0 ← 2 ← 3              0 ← qr2 ← qr1
            ↑ ↙                     ↑  ↙
            4                      qr0
        """
        cmap5 = CouplingMap(TENERIFE_CMAP)

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, strict_direction=True, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set, strict_direction=True)

    def test_9q_circuit_Rueschlikon(self):
        """9 qubits in Rueschlikon, without considering the direction

        1 →  2 →  3 →  4 ←  5 ←  6 →  7 ← 8
        ↓    ↑    ↓    ↓    ↑    ↓    ↓   ↑
        0 ← 15 → 14 ← 13 ← 12 → 11 → 10 ← 9

          1 -- q1_0 - q1_1 - 4 --- 5 --  6  - 7 --- q0_1
          |    |      |      |     |     |    |      |
        q1_2 - q1_3 - q0_0 - 13 - q0_3 - 11 - q1_4 - q0_2
        """
        cmap16 = CouplingMap(RUESCHLIKON_CMAP)

        qr0 = QuantumRegister(4, "q0")
        qr1 = QuantumRegister(5, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr0[1], qr0[2])  # q0[1] -> q0[2]
        circuit.cx(qr0[0], qr1[3])  # q0[0] -> q1[3]
        circuit.cx(qr1[4], qr0[2])  # q1[4] -> q0[2]

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap16, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap16, pass_.property_set)

    def test_3q_circuit_Tenerife(self):
        """3 qubits in Tenerife, without considering the direction

            1                    1
          ↙ ↑                 /  |
        0 ← 2 ← 3           0 - qr1 - qr2
            ↑ ↙                 |   /
            4                   qr0
        """
        cmap5 = CouplingMap(TENERIFE_CMAP)

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap5, strict_direction=False, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap5, pass_.property_set)

    def test_3q_circuit_vigo_with_custom_scores(self):
        """Test custom ErrorMap from analysis pass are used for scoring."""
        backend = GenericBackendV2(num_qubits=5, seed=42)
        target = backend.target

        class FakeScore(AnalysisPass):
            """Fake analysis pass with custom scoring."""

            def run(self, dag):
                error_map = ErrorMap(9)
                error_map.add_error((0, 0), 0.1)
                error_map.add_error((0, 1), 0.5)
                error_map.add_error((1, 1), 0.2)
                error_map.add_error((1, 2), 0.8)
                error_map.add_error((1, 3), 0.75)
                error_map.add_error((2, 2), 0.123)
                error_map.add_error((3, 3), 0.333)
                error_map.add_error((3, 4), 0.12345423)
                error_map.add_error((4, 4), 0.2222)
                self.property_set["vf2_avg_error_map"] = error_map

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2

        vf2_pass = VF2Layout(target=target, seed=1234568942)
        property_set = {}
        vf2_pass(circuit, property_set)
        pm = PassManager([FakeScore(), VF2Layout(target=target, seed=1234568942)])
        pm.run(circuit)
        # Assert layout is different from backend properties
        self.assertNotEqual(property_set["layout"], pm.property_set["layout"])
        self.assertLayout(circuit_to_dag(circuit), backend.coupling_map, pm.property_set)

    def test_error_map_pickle(self):
        """Test that the `ErrorMap` Rust structure correctly pickles and depickles."""
        errors = {(0, 1): 0.2, (1, 0): 0.2, (0, 0): 0.05, (1, 1): 0.02}
        error_map = ErrorMap.from_dict(errors)
        with io.BytesIO() as fptr:
            pickle.dump(error_map, fptr)
            fptr.seek(0)
            loaded = pickle.load(fptr)
        self.assertEqual(len(loaded), len(errors))
        self.assertEqual({k: loaded[k] for k in errors}, errors)

    def test_perfect_fit_Manhattan(self):
        """A circuit that fits perfectly in Manhattan (65 qubits)
        See https://github.com/Qiskit/qiskit-terra/issues/5694"""
        cmap65 = CouplingMap(MANHATTAN_CMAP)

        rows = [x[0] for x in MANHATTAN_CMAP]
        cols = [x[1] for x in MANHATTAN_CMAP]

        num_qubits = 65
        adj_matrix = numpy.zeros((num_qubits, num_qubits))
        adj_matrix[rows, cols] = 1

        circuit = GraphStateGate(adj_matrix).definition
        circuit.measure_all()

        dag = circuit_to_dag(circuit)
        pass_ = VF2Layout(cmap65, seed=self.seed, max_trials=1)
        pass_.run(dag)
        self.assertLayout(dag, cmap65, pass_.property_set)


class TestVF2LayoutOther(LayoutTestCase):
    """Other VF2Layout tests"""

    def test_seed(self):
        """Different seeds yield different results"""
        seed_1 = 42
        seed_2 = 45

        cmap5 = TENERIFE_CMAP

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        circuit.cx(qr[0], qr[2])  # qr0 -> qr2
        circuit.cx(qr[1], qr[2])  # qr1 -> qr2
        dag = circuit_to_dag(circuit)

        pass_1 = VF2Layout(CouplingMap(cmap5), seed=seed_1, max_trials=1)
        pass_1.run(dag)
        layout_1 = pass_1.property_set["layout"]
        self.assertEqual(
            pass_1.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND
        )

        pass_2 = VF2Layout(CouplingMap(cmap5), seed=seed_2, max_trials=1)
        pass_2.run(dag)
        layout_2 = pass_2.property_set["layout"]
        self.assertEqual(
            pass_2.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND
        )

        self.assertNotEqual(layout_1, layout_2)

    def test_3_q_gate(self):
        """The pass does not handle gates with more than 2 qubits"""
        seed_1 = 42

        cmap5 = TENERIFE_CMAP

        qr = QuantumRegister(3, "qr")
        circuit = QuantumCircuit(qr)
        circuit.ccx(qr[1], qr[0], qr[2])
        dag = circuit_to_dag(circuit)

        pass_1 = VF2Layout(CouplingMap(cmap5), seed=seed_1, max_trials=1)
        pass_1.run(dag)
        self.assertEqual(
            pass_1.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.MORE_THAN_2Q
        )

    def test_target_without_coupling_map(self):
        """When a target has no coupling_map but it is provided as argument.
        See: https://github.com/Qiskit/qiskit/pull/11585"""

        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)

        target = Target(num_qubits=3)
        target.add_instruction(CXGate())

        vf2_pass = VF2Layout(
            coupling_map=CouplingMap([[0, 2], [1, 2]]), target=target, seed=-1, max_trials=1
        )
        vf2_pass.run(dag)

        self.assertEqual(
            vf2_pass.property_set["VF2Layout_stop_reason"], VF2LayoutStopReason.SOLUTION_FOUND
        )

    def test_error_clipping(self):
        """Out-of-range errors should clip to 0.0 and 1.0, and be safely usable."""
        target = Target(3)
        props_1q = {
            # Too low.
            (0,): InstructionProperties(error=-1e-15),
            # Too high.
            (1,): InstructionProperties(error=1 + 1e-15),
            # Juuuuust right.
            (2,): InstructionProperties(error=1e-5),
        }
        # Set the same 1q on both types of 1q operation because we don't want any averaging to mess
        # with our clipping.
        target.add_instruction(lib.SXGate(), props_1q)
        target.add_instruction(lib.RZGate(Parameter("a")), props_1q)
        target.add_instruction(
            lib.CXGate(),
            {
                (0, 1): InstructionProperties(error=-1e-15),
                (1, 2): InstructionProperties(error=1 + 1e-15),
                (2, 0): InstructionProperties(error=1e-5),
            },
        )

        # This circuit has a symmetric interaction graph, but we break the degeneracy by weighting
        # each edge and node by different amounts, so the resulting layout will be deterministic.
        # One qubit deliberately doesn't have any instructions so we test the zero-weighted path
        # with all the types of clipped error too.
        qc = QuantumCircuit(3)
        qc.cx(1, 0)
        qc.cx(2, 1)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(0, 2)
        qc.cx(0, 2)
        qc.sx(1)
        qc.sx(2)

        qc = PassManager([VF2Layout(target=target), ApplyLayout()]).run(qc)
        self.assertEqual(qc.layout.initial_index_layout(), [1, 2, 0])


class TestMultipleTrials(QiskitTestCase):
    """Test the passes behavior with >1 trial."""

    def test_no_properties(self):
        """Test it finds the lowest degree perfect layout with no properties."""
        vf2_pass = VF2Layout(
            CouplingMap(
                [
                    (0, 1),
                    (0, 2),
                    (0, 3),
                    (1, 0),
                    (1, 2),
                    (1, 3),
                    (2, 0),
                    (2, 1),
                    (2, 3),
                    (3, 0),
                    (3, 1),
                    (3, 2),
                    (4, 0),
                    (0, 4),
                    (5, 1),
                    (1, 5),
                ]
            )
        )
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {4, 5})

    def test_with_properties(self):
        """Test it finds the least noise perfect layout with no properties."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.measure_all()
        cmap = CouplingMap(YORKTOWN_CMAP)
        backend = GenericBackendV2(num_qubits=5, coupling_map=cmap, seed=15)
        vf2_pass = VF2Layout(target=backend.target)
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {1, 3})

    def test_max_trials_exceeded(self):
        """Test it exits when max_trials is reached."""

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.cx(0, 1)
        qc.measure_all()
        cmap = CouplingMap(YORKTOWN_CMAP)
        backend = GenericBackendV2(num_qubits=5, coupling_map=cmap, seed=1)
        vf2_pass = VF2Layout(target=backend.target, seed=-1, max_trials=1)
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {2, 0})

    def test_time_limit_exceeded(self):
        """Test the pass stops after time_limit is reached."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.x(qr)
        qc.cx(0, 1)
        qc.measure_all()
        cmap = CouplingMap(YORKTOWN_CMAP)
        backend = GenericBackendV2(num_qubits=5, coupling_map=cmap, seed=1)
        vf2_pass = VF2Layout(target=backend.target, seed=-1, time_limit=0.0)
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {2, 0})

    def test_reasonable_limits_for_simple_layouts(self):
        """Test that the default trials is set to a reasonable number."""
        backend = GenericBackendV2(27, seed=42)
        qc = QuantumCircuit(5)
        qc.cx(2, 3)
        qc.cx(0, 1)

        # Run without any limits set
        vf2_pass = VF2Layout(target=backend.target, seed=42)
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {26, 11, 14, 7, 10})

    def test_no_limits_with_negative(self):
        """Test that we're not enforcing a trial limit if set to negative."""
        qc = QuantumCircuit(3)
        qc.h(0)
        cmap = CouplingMap(YORKTOWN_CMAP)
        backend = GenericBackendV2(num_qubits=5, coupling_map=cmap, seed=4)

        # Run without any limits set
        vf2_pass = VF2Layout(
            target=backend.target,
            seed=42,
            max_trials=0,
        )
        property_set = {}
        vf2_pass(qc, property_set)
        self.assertEqual(set(property_set["layout"].get_physical_bits()), {3, 2, 0})

    def test_qregs_valid_layout_output(self):
        """Test that vf2 layout doesn't add extra qubits.

        Reproduce from https://github.com/Qiskit/qiskit-terra/issues/8667
        """
        backend = GenericBackendV2(
            basis_gates=["cx", "id", "rz", "sx", "x"], num_qubits=16, seed=42
        )
        qr = QuantumRegister(16, name="qr")
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.rz(pi / 2, qr[0])
        qc.sx(qr[0])
        qc.sx(qr[1])
        qc.rz(-pi / 4, qr[1])
        qc.sx(qr[1])
        qc.rz(pi / 2, qr[1])
        qc.rz(2.8272143, qr[0])
        qc.rz(0.43324854, qr[1])
        qc.sx(qr[1])
        qc.rz(-0.95531662, qr[7])
        qc.sx(qr[7])
        qc.rz(3 * pi / 4, qr[7])
        qc.barrier([qr[1], qr[10], qr[4], qr[0], qr[7]])
        vf2_pass = VF2Layout(
            seed=12345,
            target=backend.target,
        )
        vf2_pass(qc)
        self.assertEqual(len(vf2_pass.property_set["layout"].get_physical_bits()), 16)
        self.assertEqual(len(vf2_pass.property_set["layout"].get_virtual_bits()), 16)
        pm = PassManager(
            [
                VF2Layout(
                    seed=12345,
                    target=backend.target,
                )
            ]
        )
        pm += generate_embed_passmanager(backend.coupling_map)
        res = pm.run(qc)
        self.assertEqual(res.num_qubits, 16)


if __name__ == "__main__":
    unittest.main()
