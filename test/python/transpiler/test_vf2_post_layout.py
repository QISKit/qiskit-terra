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

import ddt

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import ControlFlowOp, Parameter, library as lib
from qiskit.transpiler import Layout, TranspilerError, PassManager, passes
from qiskit.transpiler.passes import TrivialLayout, ApplyLayout
from qiskit.transpiler.passes.layout.vf2_post_layout import VF2PostLayout, VF2PostLayoutStopReason
from qiskit.converters import circuit_to_dag
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.compiler.transpiler import transpile
from qiskit.transpiler.target import Target, InstructionProperties
from test import QiskitTestCase, combine

from ..legacy_cmaps import LIMA_CMAP, YORKTOWN_CMAP, BOGOTA_CMAP


@ddt.ddt
class TestVF2PostLayout(QiskitTestCase):
    """Tests the VF2Layout pass"""

    seed = 42

    def assertLayoutV2(self, dag, target, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        target"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]

        def run(dag, wire_map):
            for gate in dag.two_qubit_ops():
                if isinstance(gate.op, ControlFlowOp):
                    continue
                physical_q0 = wire_map[gate.qargs[0]]
                physical_q1 = wire_map[gate.qargs[1]]
                qargs = (physical_q0, physical_q1)
                self.assertTrue(target.instruction_supported(gate.name, qargs))
            for node in dag.op_nodes(ControlFlowOp):
                for block in node.op.blocks:
                    inner_wire_map = {
                        inner: wire_map[outer] for outer, inner in zip(node.qargs, block.qubits)
                    }
                    run(circuit_to_dag(block), inner_wire_map)

        run(dag, {bit: layout[bit] for bit in dag.qubits if bit in layout})

    def test_no_constraints(self):
        """Test we raise at runtime if no target or coupling graph specified."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout()
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_empty_circuit_v2(self):
        """Test no solution found for empty circuit with v2 backend"""
        qc = QuantumCircuit(2, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_skip_3q_circuit_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_skip_3q_circuit_control_flow_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        with qc.for_loop((1,)):
            qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], coupling_map=LIMA_CMAP, seed=42
        ).target
        vf2_pass = VF2PostLayout(target=target)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.MORE_THAN_2Q
        )

    def test_2q_circuit_5q_backend_max_trials(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        max_trials = 11
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=YORKTOWN_CMAP,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            seed=1,
        )

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, max_trials=max_trials)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_best_mapping_ghz_state_full_device_multiple_qregs(self):
        """Test best mappings with multiple registers"""
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=123,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, backend, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        )

        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2_control_flow(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target

        circuit = QuantumCircuit(2, 1)
        with circuit.for_loop((1,)):
            circuit.cx(1, 0)  # qr1 -> qr0
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            pass
        with else_:
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(1, 0)  # qr1 -> qr0
        initial_layout = Layout(dict(enumerate(circuit.qubits)))
        circuit._layout = initial_layout
        dag = circuit_to_dag(circuit)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertLayoutV2(dag, target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_target_invalid_2q_gate(self):
        """Test that we don't find a solution with a gate outside target."""
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target
        qc = QuantumCircuit(2)
        qc.ecr(0, 1)
        dag = circuit_to_dag(qc)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_SOLUTION_FOUND,
        )

    def test_target_invalid_2q_gate_control_flow(self):
        """Test that we don't find a solution with a gate outside target."""
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=42,
        ).target
        qc = QuantumCircuit(2)
        with qc.for_loop((1,)):
            qc.ecr(0, 1)
        dag = circuit_to_dag(qc)
        pass_ = VF2PostLayout(target=target, seed=self.seed)
        pass_.run(dag)
        self.assertEqual(
            pass_.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_SOLUTION_FOUND,
        )

    def test_target_no_error(self):
        """Test that running vf2layout on a pass against a target with no error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(lib.CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2PostLayout(target=target)
        circuit = QuantumCircuit(2)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertNotIn("post_layout", vf2_pass.property_set)

    def test_target_some_error(self):
        """Test that running vf2layout on a pass against a target with some error rates works."""
        n_qubits = 15
        target = Target()
        target.add_instruction(
            lib.XGate(), {(i,): InstructionProperties(error=0.00123) for i in range(n_qubits)}
        )
        target.add_instruction(lib.CXGate(), {(i, i + 1): None for i in range(n_qubits - 1)})
        vf2_pass = VF2PostLayout(target=target, seed=1234, strict_direction=False)
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        # No layout selected because nothing will beat initial layout
        self.assertNotIn("post_layout", vf2_pass.property_set)

    def test_trivial_layout_is_best(self):
        """Test that vf2postlayout reports no better solution if the trivial layout is the best layout"""
        n_qubits = 4
        trivial_target = Target()
        trivial_target.add_instruction(
            lib.CXGate(),
            {(i, i + 1): InstructionProperties(error=0.001) for i in range(n_qubits - 1)},
        )

        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        vf2_pass = VF2PostLayout(target=trivial_target, seed=self.seed, strict_direction=False)
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_last_qubits_best(self):
        """Test that vf2postlayout determines the best layout when the last qubits have least error"""
        n_qubits = 4
        target_last_qubits_best = Target()
        target_last_qubits_best.add_instruction(
            lib.CXGate(),
            {(i, i + 1): InstructionProperties(error=10**-i) for i in range(n_qubits - 1)},
        )

        circuit = QuantumCircuit(n_qubits)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        vf2_pass = VF2PostLayout(
            target=target_last_qubits_best, seed=self.seed, strict_direction=False
        )
        dag = circuit_to_dag(circuit)
        vf2_pass.run(dag)
        self.assertLayoutV2(dag, target_last_qubits_best, vf2_pass.property_set)

    @combine(
        seed=(-1, 12),  # This hits both the "seeded" and "unseeded" paths.
        strict_direction=(True, False),
    )
    def test_complete_layout_with_idle_qubits(self, seed, strict_direction):
        """Test that completely idle qubits are included in the resulting layout."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        # We need to ensure that VF2Post actually triggers a remapping.
        target = Target(3)
        target.add_instruction(
            lib.CXGate(),
            properties={
                (0, 1): InstructionProperties(error=1e-1),
                (1, 0): InstructionProperties(error=1e-1),
                (2, 1): InstructionProperties(error=1e-8),
            },
        )
        property_set = {}
        pass_ = VF2PostLayout(target=target, seed=seed, strict_direction=strict_direction)
        pass_(qc, property_set=property_set)
        unallocated = {
            i for i, bit in enumerate(qc.qubits) if bit not in property_set["post_layout"]
        }
        self.assertEqual(unallocated, set())

    @ddt.data(-1, 12)
    def test_complete_layout_with_one_uncoupled_qubit(self, seed):
        """Test that a single uncoupled qubit is handled correctly.

        Regression test of https://github.com/Qiskit/qiskit/issues/14997."""
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.cx(1, 2)

        bad = InstructionProperties(error=1e-1)
        good = InstructionProperties(error=1e-5)

        target = Target()
        target.add_instruction(lib.XGate(), {(0,): bad, (1,): good, (2,): good})
        target.add_instruction(lib.CXGate(), {(0, 1): good, (1, 2): bad})

        pm = PassManager(
            [
                passes.TrivialLayout(target),
                passes.ApplyLayout(),
                passes.VF2PostLayout(target, seed=seed, strict_direction=True),
                passes.ApplyLayout(),
            ]
        )
        out = pm.run(qc)
        self.assertEqual(out.layout.final_index_layout(), [2, 0, 1])

    def test_can_match_heterogeneous_target_with_free_qubits(self):
        """Test that we can successfully match onto a heterogeneous target, even if there's
        non-interaction qubits present in the interaction graph."""
        num_qubits = 5
        # We don't care about the 2q links for these purposes, just heterogeneity in the 1q case.
        target = Target()
        target.add_instruction(
            lib.CXGate(),
            properties={
                (i, i + 1): InstructionProperties(error=1e-3) for i in range(num_qubits - 1)
            },
        )
        # X is available on everything, and the error gets worse as the qubit index increases, so
        # ensure we're breaking a symmetry.
        target.add_instruction(
            lib.XGate(),
            properties={
                (i,): InstructionProperties(error=1e-5 * (i + 1)) for i in range(num_qubits)
            },
        )
        # Now qubit 0, the lowest-error X gate, _also_ has RX.
        target.add_instruction(
            lib.RXGate(Parameter("a")), properties={(0,): InstructionProperties(error=1e-5)}
        )

        # The circuit is a 4q line that's all X and CX, so there's two choices for it: qubits 0-3
        # and qubits 1-4, plus a single free RX.  If VF2 tries to do all the interacting qubits
        # first it should reliably pick 0-3, since those are lower errors, but then there'd be no
        # place for the RX.  We'll also reverse the circuit indices, to make sure that there's no
        # cop-out around returning "no improvement".
        qc = QuantumCircuit(num_qubits)
        qc.x([0, 1, 2, 3])
        qc.rx(0.5, 4)
        for pair in [(3, 2), (2, 1), (1, 0)]:
            qc.cx(*pair)

        pass_ = VF2PostLayout(target=target, seed=-1, strict_direction=True)
        pass_(qc)
        layout = pass_.property_set["post_layout"]
        self.assertEqual(layout, Layout(dict(zip(qc.qubits, [4, 3, 2, 1, 0]))))

    @ddt.data(False, True)
    def test_error_clipping(self, strict_direction):
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
                (2, 0): InstructionProperties(error=1e-3),
            },
        )

        # This circuit has a symmetric interaction graph, but we break the degeneracy by weighting
        # each edge and node by different amounts, so the resulting layout will be deterministic.
        # One qubit deliberately doesn't have any instructions so we test the zero-weighted path
        # with all the types of clipped error too.
        qc = QuantumCircuit(3)
        qc.cx(2, 1)
        qc.cx(0, 2)
        qc.cx(0, 2)
        qc.sx(1)
        qc.sx(2)

        qc = PassManager(
            [
                TrivialLayout(target),
                ApplyLayout(),
                VF2PostLayout(target=target, strict_direction=strict_direction),
                ApplyLayout(),
            ]
        ).run(qc)
        # If not `strict_direction`, then we can align the really high-cost 1q and 2q physical parts
        # with the least used virtual ones.  If `strict_direction`, then we can only align one, and
        # it's better to align the 2q.
        expected = [2, 1, 0] if strict_direction else [1, 2, 0]
        self.assertEqual(qc.layout.initial_index_layout(), expected)


class TestVF2PostLayoutUndirected(QiskitTestCase):
    """Tests the VF2Layout pass"""

    seed = 42

    def assertLayout(self, dag, coupling_map, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]
        for gate in dag.two_qubit_ops():
            physical_q0 = layout[gate.qargs[0]]
            physical_q1 = layout[gate.qargs[1]]
            self.assertTrue(coupling_map.graph.has_edge(physical_q0, physical_q1))

    def assertLayoutV2(self, dag, target, property_set):
        """Checks if the circuit in dag was a perfect layout in property_set for the given
        coupling_map"""
        self.assertEqual(
            property_set["VF2PostLayout_stop_reason"], VF2PostLayoutStopReason.SOLUTION_FOUND
        )

        layout = property_set["post_layout"]
        for gate in dag.two_qubit_ops():
            physical_q0 = layout[gate.qargs[0]]
            physical_q1 = layout[gate.qargs[1]]
            qargs = (physical_q0, physical_q1)
            self.assertTrue(target.instruction_supported(gate.name, qargs))

    def test_no_constraints(self):
        """Test we raise at runtime if no target or coupling graph specified."""
        qc = QuantumCircuit(2)
        empty_pass = VF2PostLayout(strict_direction=False)
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_empty_circuit_v2(self):
        """Test no solution found for empty circuit with v2 backend"""
        qc = QuantumCircuit(2, 2)
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        ).target
        vf2_pass = VF2PostLayout(target=target, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.NO_BETTER_SOLUTION_FOUND,
        )

    def test_skip_3q_circuit_v2(self):
        """Test that the pass is a no-op on circuits with >2q gates with a target."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)
        target = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        ).target
        vf2_pass = VF2PostLayout(target=target, strict_direction=False)
        vf2_pass.run(circuit_to_dag(qc))
        self.assertEqual(
            vf2_pass.property_set["VF2PostLayout_stop_reason"],
            VF2PostLayoutStopReason.MORE_THAN_2Q,
        )

    def test_best_mapping_ghz_state_full_device_multiple_qregs(self):
        """Test best mappings with multiple registers"""
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=YORKTOWN_CMAP,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            seed=8,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=BOGOTA_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, layout_method="dense")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_best_mapping_ghz_state_full_device_multiple_qregs_v2(self):
        """Test best mappings with multiple registers"""

        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LIMA_CMAP,
            seed=self.seed,
        )
        qr_a = QuantumRegister(2)
        qr_b = QuantumRegister(3)
        qc = QuantumCircuit(qr_a, qr_b)
        qc.h(qr_a[0])
        qc.cx(qr_a[0], qr_a[1])
        qc.cx(qr_a[0], qr_b[0])
        qc.cx(qr_a[0], qr_b[1])
        qc.cx(qr_a[0], qr_b[2])
        qc.measure_all()
        tqc = transpile(qc, backend, seed_transpiler=self.seed, layout_method="trivial")
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)

    def test_2q_circuit_5q_backend_v2(self):
        """A simple example, without considering the direction
          0 - 1
        qr1 - qr0
        """
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=YORKTOWN_CMAP,
            seed=self.seed,
        )
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[0])  # qr1 -> qr0
        tqc = transpile(circuit, backend, initial_layout=[1, 0])
        initial_layout = tqc._layout
        dag = circuit_to_dag(tqc)
        pass_ = VF2PostLayout(target=backend.target, seed=self.seed, strict_direction=False)
        pass_.run(dag)
        self.assertLayoutV2(dag, backend.target, pass_.property_set)
        self.assertNotEqual(pass_.property_set["post_layout"], initial_layout)
