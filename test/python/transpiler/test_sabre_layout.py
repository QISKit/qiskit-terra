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

"""Test the SabreLayout pass"""

import unittest

import math

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import library as lib, Parameter
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import efficient_su2, quantum_volume
from qiskit.transpiler import CouplingMap, AnalysisPass, PassManager, Target, Layout
from qiskit.transpiler.passes import SabreLayout, DenseLayout, Unroll3qOrMore, BasicSwap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.converters import circuit_to_dag
from qiskit.compiler.transpiler import transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler.passes.layout.sabre_pre_layout import SabrePreLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase, slow_test

from ..legacy_cmaps import ALMADEN_CMAP, MUMBAI_CMAP


class TestSabreLayout(QiskitTestCase):
    """Tests the SabreLayout pass"""

    def setUp(self):
        super().setUp()
        self.cmap20 = ALMADEN_CMAP

    def test_5q_circuit_20q_coupling(self):
        """Test finds layout for 5q circuit on 20q device."""
        #                ┌───┐
        # q_0: ──■───────┤ X ├───────────────
        #        │       └─┬─┘┌───┐
        # q_1: ──┼────■────┼──┤ X ├───────■──
        #      ┌─┴─┐  │    │  ├───┤┌───┐┌─┴─┐
        # q_2: ┤ X ├──┼────┼──┤ X ├┤ X ├┤ X ├
        #      └───┘┌─┴─┐  │  └───┘└─┬─┘└───┘
        # q_3: ─────┤ X ├──■─────────┼───────
        #           └───┘            │
        # q_4: ──────────────────────■───────
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[1], qr[3])
        circuit.cx(qr[3], qr[0])
        circuit.x(qr[2])
        circuit.cx(qr[4], qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[1], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [11, 8, 12, 7, 13])

    def test_6q_circuit_20q_coupling(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        pass_ = SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 11, 12, 13, 6])

    def test_6q_circuit_20q_coupling_with_partial(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        pm = PassManager(
            [
                DensePartialSabreTrial(CouplingMap(self.cmap20)),
                SabreLayout(CouplingMap(self.cmap20), seed=0, swap_trials=32, layout_trials=0),
            ]
        )
        pm.run(circuit)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [1, 3, 5, 2, 6, 0])

    def test_6q_circuit_20q_coupling_with_target(self):
        """Test finds layout for 6q circuit on 20q device."""
        #       ┌───┐┌───┐┌───┐┌───┐┌───┐
        # q0_0: ┤ X ├┤ X ├┤ X ├┤ X ├┤ X ├
        #       └─┬─┘└─┬─┘└─┬─┘└─┬─┘└─┬─┘
        # q0_1: ──┼────■────┼────┼────┼──
        #         │  ┌───┐  │    │    │
        # q0_2: ──┼──┤ X ├──┼────■────┼──
        #         │  └───┘  │         │
        # q1_0: ──■─────────┼─────────┼──
        #            ┌───┐  │         │
        # q1_1: ─────┤ X ├──┼─────────■──
        #            └───┘  │
        # q1_2: ────────────■────────────
        qr0 = QuantumRegister(3, "q0")
        qr1 = QuantumRegister(3, "q1")
        circuit = QuantumCircuit(qr0, qr1)
        circuit.cx(qr1[0], qr0[0])
        circuit.cx(qr0[1], qr0[0])
        circuit.cx(qr1[2], qr0[0])
        circuit.x(qr0[2])
        circuit.cx(qr0[2], qr0[0])
        circuit.x(qr1[1])
        circuit.cx(qr1[1], qr0[0])

        dag = circuit_to_dag(circuit)
        target = GenericBackendV2(num_qubits=20, coupling_map=self.cmap20).target
        pass_ = SabreLayout(target, seed=0, swap_trials=32, layout_trials=32)
        pass_.run(dag)

        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in circuit.qubits], [7, 8, 11, 12, 13, 6])

    def test_layout_with_classical_bits(self):
        """Test sabre layout with classical bits recreate from issue #8635."""
        qc = QuantumCircuit.from_qasm_str(
            """
OPENQASM 2.0;
include "qelib1.inc";
qreg q4833[1];
qreg q4834[6];
qreg q4835[7];
creg c982[2];
creg c983[2];
creg c984[2];
rzz(0) q4833[0],q4834[4];
cu(0,-6.1035156e-05,0,1e-05) q4834[1],q4835[2];
swap q4834[0],q4834[2];
cu(-1.1920929e-07,0,-0.33333333,0) q4833[0],q4834[2];
ccx q4835[2],q4834[5],q4835[4];
measure q4835[4] -> c984[0];
ccx q4835[2],q4835[5],q4833[0];
measure q4835[5] -> c984[1];
measure q4834[0] -> c982[1];
u(10*pi,0,1.9) q4834[5];
measure q4834[3] -> c984[1];
measure q4835[0] -> c982[0];
rz(0) q4835[1];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        res = transpile(
            qc, backend, layout_method="sabre", seed_transpiler=1234, optimization_level=1
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [4, 0, 12, 1, 16, 7, 21, 11, 26, 15, 3, 18, 10, 2]
        )

    def test_layout_many_search_trials(self):
        """Test recreate failure from randomized testing that overflowed."""
        qc = QuantumCircuit.from_qasm_str(
            """
    OPENQASM 2.0;
include "qelib1.inc";
qreg q18585[14];
creg c1423[5];
creg c1424[4];
creg c1425[3];
barrier q18585[4],q18585[5],q18585[12],q18585[1];
cz q18585[11],q18585[3];
cswap q18585[8],q18585[10],q18585[6];
u(-2.00001,6.1035156e-05,-1.9) q18585[2];
barrier q18585[3],q18585[6],q18585[5],q18585[8],q18585[10],q18585[9],q18585[11],q18585[2],q18585[12],q18585[7],q18585[13],q18585[4],q18585[0],q18585[1];
cp(0) q18585[2],q18585[4];
cu(-0.99999,0,0,0) q18585[7],q18585[1];
cu(0,0,0,2.1507119) q18585[6],q18585[3];
barrier q18585[13],q18585[0],q18585[12],q18585[3],q18585[2],q18585[10];
ry(-1.1044662) q18585[13];
barrier q18585[13];
id q18585[12];
barrier q18585[12],q18585[6];
cu(-1.9,1.9,-1.5,0) q18585[10],q18585[0];
barrier q18585[13];
id q18585[8];
barrier q18585[12];
barrier q18585[12],q18585[1],q18585[9];
sdg q18585[2];
rz(-10*pi) q18585[6];
u(0,27.566433,1.9) q18585[1];
barrier q18585[12],q18585[11],q18585[9],q18585[4],q18585[7],q18585[0],q18585[13],q18585[3];
cu(-0.99999,-5.9604645e-08,-0.5,2.00001) q18585[3],q18585[13];
rx(-5.9604645e-08) q18585[7];
p(1.1) q18585[13];
barrier q18585[12],q18585[13],q18585[10],q18585[9],q18585[7],q18585[4];
z q18585[10];
measure q18585[7] -> c1423[2];
barrier q18585[0],q18585[3],q18585[7],q18585[4],q18585[1],q18585[8],q18585[6],q18585[11],q18585[5];
barrier q18585[5],q18585[2],q18585[8],q18585[3],q18585[6];
"""
        )
        backend = GenericBackendV2(
            num_qubits=27,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=MUMBAI_CMAP,
            seed=42,
        )
        res = transpile(
            qc,
            backend,
            layout_method="sabre",
            routing_method="basic",
            seed_transpiler=12345,
            optimization_level=1,
        )
        self.assertIsInstance(res, QuantumCircuit)
        layout = res._layout.initial_layout
        self.assertEqual(
            [layout[q] for q in qc.qubits], [13, 7, 18, 22, 17, 8, 14, 10, 11, 3, 16, 25, 23, 19]
        )

    def test_support_var_with_rust_fastpath(self):
        """Test that the joint layout/embed/routing logic for the Rust-space fast-path works in the
        presence of standalone `Var` nodes."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        out = SabreLayout(CouplingMap.from_line(8), seed=0, swap_trials=2, layout_trials=2)(qc)

        self.assertIsInstance(out, QuantumCircuit)
        self.assertEqual(out.layout.initial_index_layout(), [2, 3, 4, 6, 5, 0, 1, 7])

    def test_support_var_with_explicit_routing_pass(self):
        """Test that the logic works if an explicit routing pass is given."""
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))

        qc = QuantumCircuit(5, inputs=[a])
        qc.add_var(b, 12)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 0)

        cm = CouplingMap.from_line(8)
        pass_ = SabreLayout(cm, seed=0, routing_pass=BasicSwap(cm, fake_run=True))
        _ = pass_(qc)
        layout = pass_.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 4, 2, 5, 1])

    def test_uninitialized_target(self):
        """We shouldn't panic if the target isn't initialized."""
        target = Target(num_qubits=None)
        qc = QuantumCircuit(2)
        pass_ = SabreLayout(target, seed=0)
        with self.assertRaisesRegex(TranspilerError, "not initialized"):
            pass_(qc)

    def test_out_of_range_partials(self):
        """We should safely reject partial layouts that are invalid."""
        pass_ = SabreLayout(
            Target.from_configuration(
                num_qubits=5, coupling_map=CouplingMap.from_line(5), basis_gates=["sx", "rz", "cx"]
            ),
            seed=0,
        )
        qc = QuantumCircuit(2)
        partial = Layout(dict(zip(qc.qubits, [1, 5])))
        with self.assertRaisesRegex(TranspilerError, "out-of-range physical qubits"):
            pass_(qc, property_set={"sabre_starting_layouts": [partial]})

    @slow_test
    def test_release_valve_routes_multiple(self):
        """Test Sabre works if the release valve routes more than 1 operation.

        Regression test of #13081.
        """
        qv = quantum_volume(500, seed=42)
        qv.measure_all()
        qc = Unroll3qOrMore()(qv)

        cmap = CouplingMap.from_heavy_hex(21)
        pm = PassManager(
            [
                SabreLayout(cmap, swap_trials=20, layout_trials=20, max_iterations=4, seed=100),
            ]
        )
        _ = pm.run(qc)
        self.assertIsNotNone(pm.property_set.get("layout"))

    def test_all_to_all(self):
        """An implicitly all-to-all backend should just become physical with the trivial layout."""
        qc = QuantumCircuit(QuantumRegister(5, "virtuals"))
        for target in qc.qubits[1:]:
            qc.cx(qc.qubits[0], target)
        # No qargs in the instruction properties => implicitly all-to-all.
        target = Target(num_qubits=10)
        target.add_instruction(lib.RZGate(Parameter("t")))
        target.add_instruction(lib.SXGate())
        target.add_instruction(lib.CXGate())
        pass_ = SabreLayout(target, seed=0)
        out = pass_(qc)
        self.assertEqual(out.layout.initial_index_layout(), list(range(10)))
        self.assertEqual(out.layout.routing_permutation(), list(range(10)))

        expected = QuantumCircuit(QuantumRegister(10, "q"))
        for target in range(1, qc.num_qubits):
            expected.cx(0, target)
        self.assertEqual(out, expected)


class DensePartialSabreTrial(AnalysisPass):
    """Pass to run dense layout as a sabre trial."""

    def __init__(self, cmap):
        self.dense_pass = DenseLayout(cmap)
        super().__init__()

    def run(self, dag):
        self.dense_pass.run(dag)
        self.property_set["sabre_starting_layouts"] = [self.dense_pass.property_set["layout"]]


class TestDisjointDeviceSabreLayout(QiskitTestCase):
    """Test SabreLayout with a disjoint coupling map."""

    def setUp(self):
        super().setUp()
        self.dual_grid_cmap = CouplingMap(
            [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [5, 8]]
        )

    def test_dual_ghz(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_wide_barrier(self):
        """Test a basic example with 2 circuit components and 2 cmap components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_intermediate_barriers(self):
        """Test dual ghz circuit with intermediate barriers local to each component."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.barrier(4, 5, 6)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dual_ghz_with_intermediate_spanning_barriers(self):
        """Test dual ghz circuit with barrier in the middle across components."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.barrier(0, 1, 2, 4, 5)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        layout_routing_pass(qc)
        layout = layout_routing_pass.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_too_large_components(self):
        """Assert trying to run a circuit with too large a connected component raises."""
        qc = QuantumCircuit(8)
        qc.h(0)
        for i in range(1, 6):
            qc.cx(0, i)
        qc.h(7)
        qc.cx(7, 6)
        layout_routing_pass = SabreLayout(
            self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1
        )
        with self.assertRaises(TranspilerError):
            layout_routing_pass(qc)

    def test_with_partial_layout(self):
        """Test a partial layout with a disjoint connectivity graph."""
        qc = QuantumCircuit(8, name="double dhz")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.h(3)
        qc.cx(3, 4)
        qc.cx(3, 5)
        qc.cx(3, 6)
        qc.cx(3, 7)
        qc.measure_all()
        pm = PassManager(
            [
                DensePartialSabreTrial(self.dual_grid_cmap),
                SabreLayout(self.dual_grid_cmap, seed=123456, swap_trials=1, layout_trials=1),
            ]
        )
        pm.run(qc)
        layout = pm.property_set["layout"]
        self.assertEqual([layout[q] for q in qc.qubits], [3, 2, 1, 5, 4, 7, 6, 8])

    def test_dag_fits_in_one_component(self):
        """Test that the output is valid if the DAG all fits in a single component of a disjoint
        coupling map.."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        disjoint = CouplingMap([(0, 1), (1, 2), (3, 4), (4, 5)])
        layout_routing_pass = SabreLayout(disjoint, seed=2025_02_12, swap_trials=1, layout_trials=1)
        out = layout_routing_pass(qc)
        self.assertEqual(len(out.layout.initial_layout), len(out.layout.final_layout))
        self.assertEqual(out.layout.initial_index_layout(filter_ancillas=False), [4, 5, 3, 0, 1, 2])
        self.assertEqual(out.layout.final_index_layout(filter_ancillas=False), [3, 5, 4, 0, 1, 2])

    def test_sabre_layout_global_1q(self):
        """Test that the pass runs with a globally defined 1q gate."""
        target = Target(num_qubits=3)
        target.add_instruction(lib.XGate())
        target.add_instruction(lib.CXGate(), {(0, 1): None, (1, 2): None})
        layout_routing_pass = SabreLayout(target, swap_trials=1, layout_trials=1, seed=42)
        qc = QuantumCircuit(3)
        qc.cz(0, 2)
        out = layout_routing_pass(qc)
        # sabre maps 2 -> 1, and 0 -> 2 in the layout with no swaps
        self.assertIsNone(out.count_ops().get("swap", None))
        self.assertEqual([out.find_bit(x).index for x in out.data[0].qubits], [2, 1])
        self.assertEqual(len(out.layout.initial_layout), len(out.layout.final_layout))
        self.assertEqual(out.layout.initial_index_layout(filter_ancillas=False), [2, 0, 1])
        self.assertEqual(out.layout.routing_permutation(), [0, 1, 2])
        self.assertEqual(out.layout.final_index_layout(filter_ancillas=False), [2, 0, 1])


class TestSabrePreLayout(QiskitTestCase):
    """Tests the SabreLayout pass with starting layout created by SabrePreLayout."""

    def setUp(self):
        super().setUp()
        circuit = efficient_su2(16, entanglement="circular", reps=6)
        circuit.assign_parameters([math.pi / 2] * len(circuit.parameters), inplace=True)
        circuit.measure_all()
        self.circuit = circuit
        self.coupling_map = CouplingMap.from_heavy_hex(7)

    def test_starting_layout(self):
        """Test that a starting layout is created and looks as expected."""
        pm = PassManager(
            [
                SabrePreLayout(coupling_map=self.coupling_map),
                SabreLayout(self.coupling_map, seed=123456, swap_trials=1, layout_trials=1),
            ]
        )
        pm.run(self.circuit)
        layout = pm.property_set["layout"]
        self.assertEqual(
            [layout[q] for q in self.circuit.qubits],
            [9, 81, 54, 16, 86, 58, 92, 22, 91, 21, 57, 14, 85, 53, 79, 80],
        )

    def test_integration_with_pass_manager(self):
        """Tests SabrePreLayoutIntegration with the rest of PassManager pipeline."""
        backend = GenericBackendV2(num_qubits=20, coupling_map=ALMADEN_CMAP, seed=42)
        pm = generate_preset_pass_manager(
            0, backend, layout_method="sabre", routing_method="sabre", seed_transpiler=0
        )
        pm.pre_layout = PassManager([SabrePreLayout(backend.target)])
        qct = pm.run(self.circuit)
        qct_initial_layout = qct.layout.initial_layout
        self.assertEqual(
            [qct_initial_layout[q] for q in self.circuit.qubits],
            [12, 11, 10, 16, 17, 18, 13, 14, 9, 8, 3, 2, 1, 6, 5, 7],
        )


if __name__ == "__main__":
    unittest.main()
