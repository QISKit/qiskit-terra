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

"""Test Clifford+T transpilation pipeline"""

import unittest

import os
import numpy as np
from ddt import ddt, data, unpack

from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    QFTGate,
    iqp,
    GraphStateGate,
    MCXGate,
    MultiplierGate,
    ModularAdderGate,
    UnitaryGate,
)

from qiskit.transpiler import PassManager, TransformationPass, CouplingMap, Target, TranspilerError
from qiskit.transpiler.passes import SynthesizeRZRotations, CheckGateDirection, SabreLayout
from qiskit.transpiler.preset_passmanagers.plugin import PassManagerStagePluginManager
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
    generate_preset_clifford_t_pass_manager,
)

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import get_clifford_gate_names, Operator
from qiskit.synthesis import gridsynth_rz
from qiskit.dagcircuit import DAGCircuit

from test import QiskitTestCase


_original_get_passmanager_stage = PassManagerStagePluginManager.get_passmanager_stage


class CustomTransformationPass(TransformationPass):
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        return dag


def mock_get_passmanager_stage(
    self, stage_name, plugin_name, pm_config, optimization_level
) -> PassManager:
    """Mock function for get_passmanager_stage."""
    if plugin_name == "custom":
        return PassManager([CustomTransformationPass()])
    return _original_get_passmanager_stage(
        self, stage_name, plugin_name, pm_config, optimization_level
    )


@ddt
class TestCliffordTPassManager(QiskitTestCase):
    """Test Clifford+T transpilation pipeline."""

    @data(0, 1, 2, 3)
    def test_cliffords_1q(self, optimization_level):
        """Clifford+T transpilation of a circuit with single-qubit Clifford gates."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(1)
        qc.h(2)
        qc.sdg(2)
        qc.h(2)
        qc.h(2)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_complex_clifford(self, optimization_level):
        """Clifford+T transpilation of a more complex Clifford circuit."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.s(0)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.s(1)
        qc.sdg(0)
        qc.cx(1, 0)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_rx(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate,
        requiring approximate synthesis of RZ rotations.
        """
        qc = QuantumCircuit(1)
        qc.rx(0.8, 0)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(0, 1, 2, 3)
    def test_rx_pi2(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate
        that corresponds to a Clifford gate.
        """
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 2, 0)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_rx_pi4_all_cliffords(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate
        with a T-like angle and all Clifford gates in the basis set.
        """
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 4, 0)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        transpiled_ops = transpiled.count_ops()
        self.assertLessEqual(set(transpiled_ops), set(basis_gates))

        # We should have exactly 1 T-gate.
        self.assertEqual(_get_t_count(transpiled), 1)

    @data(0, 1, 2, 3)
    def test_rx_pi4_some_cliffords(self, optimization_level):
        """Clifford+T transpilation of a circuit with a single-qubit rotation gate
        with a T-like angle and only some Clifford gates in the basis set.
        """
        qc = QuantumCircuit(1)
        qc.rx(np.pi / 4, 0)

        basis_gates = ["cx", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        transpiled_ops = transpiled.count_ops()

        # All the gates should be within the specified basis set.
        self.assertLessEqual(set(transpiled_ops), set(basis_gates))
        self.assertEqual(transpiled_ops, {"h": 2, "t": 1})

    @data(0, 1, 2, 3)
    def test_qft(self, optimization_level):
        """Clifford+T transpilation of a more complex circuit, requiring approximate
        synthesis of RZ rotations.
        """
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(0, 1, 2, 3)
    def test_multiplier(self, optimization_level):
        """Clifford+T transpilation of a multiplier gate, using different optimization levels."""
        gate = MultiplierGate(4)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            optimization_level=optimization_level, seed_transpiler=0
        )
        transpiled = pm.run(qc)

        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))
        t_count = _get_t_count(transpiled)

        # This is the T-count with optimization level 0.
        # We should not expect to see more T-gates with higher optimization levels
        # (while this is technically possible, it means that Clifford+T transpiler
        # pipeline is not setup correctly).
        expected_t_count = 1085
        self.assertLessEqual(t_count, expected_t_count)

    @data(0, 1, 2, 3)
    def test_iqp(self, optimization_level):
        """Clifford+T transpilation of IQP circuits."""
        interactions = np.array([[6, 5, 1], [5, 4, 3], [1, 3, 2]])
        qc = iqp(interactions)

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The transpiled circuit should be fairly efficient in terms of T/Tdg gates:
        # the circuit contains 3 powers of CZ-gates, each leading to at most 9 T/Tdg gates,
        # and 3 powers of T-gates, each leading to at most 4 T/Tdg gates.
        # Importantly, the transpilation should not make this worse.
        max_t_size = 3 * 9 + 3 * 4
        self.assertLessEqual(_get_t_count(transpiled), max_t_size)

    @data(0, 1, 2, 3)
    def test_graph_state(self, optimization_level):
        """Clifford+T transpilation of graph-state circuits."""
        adjacency_matrix = [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
        graph_state_gate = GraphStateGate(adjacency_matrix)

        qc = QuantumCircuit(5)
        qc.append(graph_state_gate, [0, 1, 2, 3, 4])

        basis_gates = ["cx", "s", "sdg", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)
        transpiled_ops = transpiled.count_ops()
        self.assertLessEqual(set(transpiled_ops), set(basis_gates))
        # The resulting circuit should not have any T/Tdg-gates.
        self.assertEqual(_get_t_count(transpiled), 0)

    @data(0, 1, 2, 3)
    def test_ccx(self, optimization_level):
        """Clifford+T transpilation of the Toffoli circuit."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        basis_gates = ["cx", "h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=optimization_level
        )
        transpiled = pm.run(qc)

        # Should get the efficient decomposition of the Toffoli gates into Clifford+T.
        self.assertEqual(transpiled.count_ops(), {"cx": 6, "t": 4, "tdg": 3, "h": 2})

    def test_t_gates(self):
        """Clifford+T transpilation of a circuit with T/Tdg-gates."""
        qc = QuantumCircuit(3)
        qc.t(0)
        qc.tdg(1)
        qc.t(2)
        qc.tdg(2)

        basis_gates = ["h", "t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates)
        transpiled = pm.run(qc)

        # The single T/Tdg gates on qubits 0 and 1 should remain, the T/Tdg pair on qubit 2
        # should cancel out.
        expected = QuantumCircuit(3)
        expected.t(0)
        expected.tdg(1)

        self.assertEqual(transpiled, expected)

    @data(0, 1, 2, 3)
    def test_gate_direction_remapped(self, optimization_level):
        """Test that gate directions are correct."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        basis_gates = ["cx", "h", "t", "tdg"]
        coupling_map = CouplingMap([[1, 0]])

        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level,
        )

        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # Make sure gate direction is correct
        pass_ = CheckGateDirection(coupling_map=coupling_map)
        pass_.run(circuit_to_dag(transpiled))
        self.assertTrue(pass_.property_set["is_direction_mapped"])

    @data(["t", "h"], ["tdg", "h"], ["t", "sx"], ["t", "sxdg"], ["tdg", "sx"], ["tdg", "sxdg"])
    def test_clifford_t_bases(self, basis_gates):
        """Test transpiling into various Clifford+T basis sets."""
        qc = QuantumCircuit(2)
        qc.rz(0.8, 0)
        pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    def test_target(self):
        """Clifford+T transpilation given Target."""
        qc = QuantumCircuit(2)
        qc.rz(0.8, 0)

        basis_gates = ["cx", "t", "tdg", "h"]
        backend = GenericBackendV2(5, basis_gates=basis_gates)
        pm = generate_preset_clifford_t_pass_manager(target=backend.target, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    def test_get_clifford_gate_names(self):
        """Test transpiling with get_clifford_gate_names."""
        qc = QuantumCircuit(1)
        qc.rx(0.8, 0)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    @data(2, 3, 4, 5, 6, 7)
    def test_mcx_gate_1_clean_ancilla(self, n):
        """Clifford+T transpilation of a circuit with an mcx gate."""
        # Create a circuit with an mcx gate and 1 additional clean ancilla qubit
        gate = MCXGate(n)
        nq = gate.num_qubits
        qc = QuantumCircuit(nq + 1)
        qc.append(gate, qc.qubits[0:nq])

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count
        # provided 1 ancilla qubit is available
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 7, 3: 15, 4: 27, 5: 39, 6: 51, 7: 63}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(2, 3, 4, 5, 6, 7)
    def test_mcx_gate_many_dirty_ancillas(self, n):
        """Clifford+T transpilation of a circuit with an mcx gate."""
        # Create a circuit with an mcx gate and many dirty ancillas
        nqc = 20  # number of qubits in the circuit
        qc = QuantumCircuit(nqc)
        for i in range(nqc):  # make all the qubits dirty
            qc.x(i)
        gate = MCXGate(n)
        nq = gate.num_qubits  # number of qubits in the gate
        qc.append(gate, qc.qubits[0:nq])

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count
        # provided 1 ancilla qubit is available
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 7, 3: 17, 4: 29, 5: 41, 6: 51, 7: 63}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(2, 3, 4, 5, 6, 7)
    def test_multiplier_gate(self, n):
        """Clifford+T transpilation of a circuit with a multiplier gate."""
        # Create a circuit with a multiplier gate
        gate = MultiplierGate(n)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count,
        # except surprisingly for the case n=1 (which is why it is not used in this test)
        t_count = _get_t_count(transpiled)
        expected_t_count = {2: 153, 3: 501, 4: 1114, 5: 2005, 6: 2596, 7: 3850}
        self.assertLessEqual(t_count, expected_t_count[n])

    @data(1, 2, 3, 4, 5, 6, 7)
    def test_modular_adder_gate(self, n):
        """Clifford+T transpilation of a circuit with a modular adder gate."""
        # Create a circuit with a modular adder gate
        gate = ModularAdderGate(n)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_clifford_t_pass_manager(optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count,
        t_count = _get_t_count(transpiled)
        expected_t_count = {1: 0, 2: 8, 3: 16, 4: 24, 5: 32, 6: 40, 7: 48}
        self.assertLessEqual(t_count, expected_t_count[n])

    def test_single_z_rotation(self):
        """Test a single RZ rotation is transpiled with expected overhead."""
        angle = 0.1
        circuit = QuantumCircuit(1)
        circuit.rz(angle, 0)

        # get the expected reference count
        reference = gridsynth_rz(angle, epsilon=0.5e-12)
        t_threshold = _get_t_count(reference)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        with self.subTest(basis_gates=basis_gates):
            pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates)
            disc = pm.run(circuit)
            self.assertLessEqual(_get_t_count(disc), t_threshold)

        basis_gates = ["t", "h", "s", "cx"]
        with self.subTest(basis_gates=basis_gates):
            pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates)
            disc = pm.run(circuit)
            self.assertLessEqual(_get_t_count(disc), t_threshold)

        basis_gates = ["t", "h", "cx"]
        with self.subTest(basis_gates=basis_gates):
            # gridsynth produces only S, X, T, H (and global phase)
            s_overhead = 2 * reference.count_ops().get("s", 0)
            x_overhead = 4 * reference.count_ops().get("x", 0)

            pm = generate_preset_clifford_t_pass_manager(basis_gates=basis_gates)
            disc = pm.run(circuit)
            self.assertLessEqual(_get_t_count(disc), t_threshold + s_overhead + x_overhead)

    @data("diag", "cliff", "collect")
    def test_sequence_collection(self, sequence_kind):
        """Test Clifford+T friendly sequences are not collected into unitaries."""
        qc = QuantumCircuit(2)
        qc.t(1)

        if sequence_kind == "cliff":
            qc.h(1)
            qc.y(1)
            qc.sx(1)
        elif sequence_kind == "diag":
            qc.rz(0.1, 1)
            qc.z(1)
            qc.sdg(1)
            qc.tdg(1)
        else:
            qc.ry(0.4, 1)

        has_unitary = [False]

        def check_for_unitary(**kwargs):
            name = kwargs["pass_"].__class__.__name__
            if name == "ConsolidateBlocks":
                ops = kwargs["dag"].count_ops()
                has_unitary[0] = "unitary" in ops.keys()

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        _ = transpile(qc, basis_gates=basis_gates, callback=check_for_unitary)

        expect_unitary = sequence_kind == "collect"
        self.assertEqual(expect_unitary, has_unitary[0])

    def test_raises_on_non_clifford_t(self):
        """Assert that calling generate_preset_clifford_t_pass_manager raises when
        passed a non Clifford+T set.
        """
        # non-Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg", "rz"]
        with self.assertRaises(TranspilerError):
            _ = generate_preset_clifford_t_pass_manager(
                basis_gates=basis_gates, optimization_level=0
            )

    def test_incomplete_basis_sets(self):
        """
        Test that compiling into incomplete Clifford+T basis sets
        succeeds provided that the translation exists.
        """
        with self.subTest("Only T-gate"):
            basis_gates = ["t"]
            pm = generate_preset_pass_manager(basis_gates=basis_gates)
            qc = QuantumCircuit(2)
            qc.s(0)
            transpiled = pm.run(qc)
            self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        with self.subTest("Only CX-gate"):
            basis_gates = ["cx"]
            pm = generate_preset_pass_manager(basis_gates=basis_gates)
            qc = QuantumCircuit(2)
            qc.swap(0, 1)
            transpiled = pm.run(qc)
            self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    def test_legacy_pass_manager_with_clifford_t(self):
        """Test that calling generate_preset_pass_manager with Clifford+T
        gates also works as expected.
        """
        gate = ModularAdderGate(7)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        # Transpile to a Clifford+T basis set
        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        pm = generate_preset_pass_manager(basis_gates=basis_gates, optimization_level=0)
        transpiled = pm.run(qc)
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

        # The resulting decomposition should be efficient in terms of T-count,
        t_count = _get_t_count(transpiled)
        expected_t_count = 48
        self.assertLessEqual(t_count, expected_t_count)

    def test_target_and_basis_gates(self):
        """Test calling generate_preset_clifford_t_pass_manager with various values
        of the arguments ``target`` and ``basis_gates``.
        """
        gate = ModularAdderGate(4)
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        target = Target.from_configuration(basis_gates=basis_gates)

        # Transpile with explicitly setting ``basis_gates`` but not ``target``.
        transpiled_basis_gates = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, optimization_level=0
        ).run(qc)

        # Transpile with explicitly setting ``target`` but not ``basis_gates``.
        transpiled_target = generate_preset_clifford_t_pass_manager(
            target=target, optimization_level=0
        ).run(qc)

        # Transpile with setting neither ``basis_gates`` nor ``target``, in which case
        # ``basis_gates`` should default to all of Clifford+T gates.
        transpiled_neither = generate_preset_clifford_t_pass_manager(optimization_level=0).run(qc)

        # Transpile setting both, to make sure target overrides basis_gates when both are
        # specified.
        transpiled_both = generate_preset_clifford_t_pass_manager(
            target=target, basis_gates=["cx", "tdg", "h", "s"], optimization_level=0
        ).run(qc)
        self.assertEqual(transpiled_basis_gates, transpiled_target)
        self.assertEqual(transpiled_basis_gates, transpiled_neither)
        self.assertEqual(transpiled_basis_gates, transpiled_both)

    @unittest.mock.patch.object(
        PassManagerStagePluginManager,
        "get_passmanager_stage",
        new=mock_get_passmanager_stage,
    )
    def test_custom_stage_plugins_legacy(self):
        """
        Test that in the legacy Clifford+T pipeline (called from `generate_preset_pass_manager`)
        we can specify custom stage plugins.
        """
        basis_gates = ["cx", "h", "t"]
        coupling_map = CouplingMap.from_line(3)
        with self.subTest("no custom methods"):
            pm = generate_preset_pass_manager(optimization_level=1, basis_gates=basis_gates)
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertNotIn("CustomTransformationPass", passes)

        with self.subTest("custom init method"):
            pm = generate_preset_pass_manager(
                optimization_level=1, basis_gates=basis_gates, init_method="custom"
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)

        with self.subTest("custom layout method"):
            pm = generate_preset_pass_manager(
                optimization_level=1,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                layout_method="custom",
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)

        with self.subTest("custom routing method"):
            pm = generate_preset_pass_manager(
                optimization_level=1,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                routing_method="custom",
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)

        with self.subTest("custom translation method"):
            pm = generate_preset_pass_manager(
                optimization_level=1, basis_gates=basis_gates, translation_method="custom"
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)

        with self.subTest("custom optimization method"):
            pm = generate_preset_pass_manager(
                optimization_level=1, basis_gates=basis_gates, optimization_method="custom"
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)
        with self.subTest("custom scheduling method"):
            pm = generate_preset_pass_manager(
                optimization_level=1, basis_gates=basis_gates, scheduling_method="custom"
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("CustomTransformationPass", passes)

    def test_legacy_pass_manager_with_routing_disabled(self):
        """
        Test that in the legacy Clifford+T pipeline, routing is disabled exactly when
        `routing_method="none"`. In other words, passes that can modify the final layout
        (such as `ElidePermutations`) do not run.
        """
        basis_gates = ["cx", "h", "t"]
        coupling_map = CouplingMap.from_line(3)

        with self.subTest("routing is enabled"):
            pm = generate_preset_pass_manager(
                optimization_level=2, basis_gates=basis_gates, coupling_map=coupling_map
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertIn("ElidePermutations", passes)

        with self.subTest("routing is disabled"):
            pm = generate_preset_pass_manager(
                optimization_level=2,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                routing_method="none",
            )
            passes = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
            self.assertNotIn("ElidePermutations", passes)

    @data(
        (1 - 1e-8, 1e-4, 1e-5),
        (1 - 1e-9, 1e-7, 1e-2),
        (1 - 1e-9, 1e-7, None),
        (1 - 1e-9, None, 1e-7),
        (None, 1e-7, 1e-2),
    )
    @unpack
    def test_rz_config_is_passed(self, approximation_degree, synthesis_error, cache_error):
        """
        Test that the options `rz_synthesis_config` and `approximation_degree` are passed correctly
        to the `SynthesizeRZRotations` transpiler pass.
        """

        qc = QuantumCircuit(1)
        theta = 2.3456
        qc.rz(theta, 0)

        basis_gates = ["cx", "h", "t"]
        rz_synthesis_config = {
            "rz_synthesis_error": synthesis_error,
            "rz_cache_error": cache_error,
        }

        pm = generate_preset_clifford_t_pass_manager(
            rz_synthesis_config=rz_synthesis_config,
            basis_gates=basis_gates,
        )

        # Save the original SynthesizeRZRotations.run method,
        # and create a mock method that calls the original method, but also
        # records the values of synthesis_error and cache_error.
        original_run = SynthesizeRZRotations.run
        run_calls = []

        def mock_run(self, dag):
            run_calls.append([self.approximation_degree, self.synthesis_error, self.cache_error])
            original_run(self, dag)

        with unittest.mock.patch.object(SynthesizeRZRotations, "run", new=mock_run):
            pm = generate_preset_clifford_t_pass_manager(
                rz_synthesis_config=rz_synthesis_config,
                basis_gates=basis_gates,
                approximation_degree=approximation_degree,
            )
            _ = pm.run(qc)
        self.assertEqual(len(run_calls), 1)
        expected_approximation_degree = (
            approximation_degree if approximation_degree is not None else 1.0
        )
        self.assertEqual(
            run_calls[0], [expected_approximation_degree, synthesis_error, cache_error]
        )

    def test_initial_layout(self):
        """Test argument `initial_layout`."""

        qc = QuantumCircuit(2)
        qc.t(0)

        basis_gates = ["t"]

        with self.subTest("initial layout is [0, 1]"):
            pm = generate_preset_clifford_t_pass_manager(
                basis_gates=basis_gates, optimization_level=1, initial_layout=[0, 1]
            )
            qct = pm.run(qc)
            used_qubit = qct[0].qubits[0]
            self.assertEqual(qct.find_bit(used_qubit).index, 0)

        with self.subTest("initial layout is [1, 0]"):
            pm = generate_preset_clifford_t_pass_manager(
                basis_gates=basis_gates, optimization_level=1, initial_layout=[1, 0]
            )
            qct = pm.run(qc)
            used_qubit = qct[0].qubits[0]
            self.assertEqual(qct.find_bit(used_qubit).index, 1)

    def test_optimization_level(self):
        """Test argument `optimization_level`."""
        qc = QuantumCircuit(2)
        qc.t(0)
        qc.tdg(0)
        basis_gates = ["t", "tdg"]
        with self.subTest("optimization level is 0"):
            pm = generate_preset_clifford_t_pass_manager(
                basis_gates=basis_gates, optimization_level=0
            )
            qct = pm.run(qc)
            self.assertEqual(qct, qc)
        with self.subTest("optimization level is 3"):
            pm = generate_preset_clifford_t_pass_manager(
                basis_gates=basis_gates, optimization_level=3
            )
            qct = pm.run(qc)
            expected = QuantumCircuit(2)
            self.assertEqual(qct, expected)

    def test_coupling_map(self):
        """Test argument `coupling_map`."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        basis_gates = {"cx", "t"}
        coupling_map = [[2, 3]]
        pm = generate_preset_clifford_t_pass_manager(
            basis_gates=basis_gates, coupling_map=coupling_map
        )
        qct = pm.run(qc)
        used_qubits = {qct.find_bit(q).index for q in qct[0].qubits}
        self.assertEqual(used_qubits, {2, 3})

    def test_circuit_with_unitaries(self):
        """
        Test that `generate_preset_clifford_t_pass_manager` handles circuits with unitaries.
        """
        # 1q unitary
        qc1 = QuantumCircuit(1)
        qc1.rx(0.2, 0)
        u1 = UnitaryGate(Operator(qc1))

        # 2q unitary
        qc2 = QuantumCircuit(2)
        qc2.rx(0.3, 0)
        qc2.cx(0, 1)
        u2 = UnitaryGate(Operator(qc2))

        # 3q unitary
        qc3 = QuantumCircuit(3)
        qc3.rx(0.3, 0)
        qc3.cx(0, 1)
        qc3.cx(1, 2)
        u3 = UnitaryGate(Operator(qc3))

        qc = QuantumCircuit(3)
        qc.append(u1, [0])
        qc.append(u2, [1, 2])
        qc.append(u3, [2, 0, 1])

        pm = generate_preset_clifford_t_pass_manager()

        transpiled = pm.run(qc)

        basis_gates = get_clifford_gate_names() + ["t", "tdg"]
        self.assertLessEqual(set(transpiled.count_ops()), set(basis_gates))

    def test_parse_seed_transpiler_from_env_var(self):
        """Test that the environment variable QISKIT_TRANSPILER_SEED is passed to the transpiler."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        # Save the original SabreLayout.run method, and create a mock method that calls the original method,
        # but also records the value for seed.
        original_run = SabreLayout.run
        run_calls = []

        def mock_run(self, dag):
            run_calls.append(self.seed)
            original_run(self, dag)

        with unittest.mock.patch.dict(os.environ, {"QISKIT_TRANSPILER_SEED": "9876"}):
            with unittest.mock.patch.object(SabreLayout, "run", new=mock_run):
                _ = transpile(qc, optimization_level=1, coupling_map=CouplingMap.from_line(3))
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0], 9876)


def _get_t_count(qc):
    """Returns the number of T/Tdg gates in a circuit."""
    ops = qc.count_ops()
    return ops.get("t", 0) + ops.get("tdg", 0)
