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

"""Tests preset pass manager API"""

import os
import unittest


from test import combine
from ddt import ddt, data

import numpy as np

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import (
    Qubit,
    Gate,
    ControlFlowOp,
    Measure,
    library as lib,
    Parameter,
    AncillaRegister,
)
from qiskit.compiler import transpile
from qiskit.transpiler import (
    CouplingMap,
    Layout,
    PassManager,
    TranspilerError,
    Target,
    InstructionProperties,
)
from qiskit.circuit.library import U2Gate, U3Gate, quantum_volume, CXGate, CZGate, XGate
from qiskit.transpiler.passes import (
    ALAPScheduleAnalysis,
    PadDynamicalDecoupling,
    RemoveResetInZeroState,
    SabreLayout,
)
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import GraphStateGate, UnitaryGate
from qiskit.quantum_info import random_unitary
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.preset_passmanagers import level0, level1, level2, level3
from qiskit.transpiler.passes import GatesInBasis, TwoQubitPeepholeOptimization
from qiskit.transpiler.passes.scheduling.alignments.check_durations import InstructionDurationCheck
from qiskit.transpiler.preset_passmanagers.builtin_plugins import OptimizationPassManager
from qiskit.transpiler.timing_constraints import TimingConstraints
from test import QiskitTestCase

from ..legacy_cmaps import MELBOURNE_CMAP, RUESCHLIKON_CMAP, LAGOS_CMAP, TOKYO_CMAP, BOGOTA_CMAP


def mock_get_passmanager_stage(
    stage_name,
    plugin_name,
    pm_config,
    optimization_level=None,
) -> PassManager:
    """Mock function for get_passmanager_stage."""
    if stage_name == "translation" and plugin_name == "custom_stage_for_test":
        pm = PassManager([RemoveResetInZeroState()])
        return pm

    elif stage_name == "scheduling" and plugin_name == "custom_stage_for_test":
        dd_sequence = [XGate(), XGate()]
        pm = PassManager(
            [
                ALAPScheduleAnalysis(pm_config.instruction_durations),
                PadDynamicalDecoupling(pm_config.instruction_durations, dd_sequence),
            ]
        )
        return pm
    elif stage_name == "init":
        return PassManager([])
    elif stage_name == "routing":
        return PassManager([])
    elif stage_name == "optimization":
        return OptimizationPassManager().pass_manager(pm_config, optimization_level)
    elif stage_name == "layout":
        return PassManager([])
    else:
        raise RuntimeError("Failure, unexpected stage plugin combo for test")


def emptycircuit():
    """Empty circuit"""
    return QuantumCircuit()


def circuit_2532():
    """See https://github.com/Qiskit/qiskit-terra/issues/2532"""
    circuit = QuantumCircuit(5)
    circuit.cx(2, 4)
    return circuit


@ddt
class TestPresetPassManager(QiskitTestCase):
    """Test preset passmanagers work as expected."""

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_no_coupling_map_with_sabre(self, level):
        """Test that coupling_map can be None with Sabre (level={level})"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.cz(q[0], q[1])
        result = transpile(
            circuit,
            coupling_map=None,
            layout_method="sabre",
            routing_method="sabre",
            optimization_level=level,
        )
        self.assertEqual(result, circuit)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_no_coupling_map(self, level):
        """Test that coupling_map can be None (level={level})"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=["u1", "u2", "u3", "cx"], optimization_level=level)
        self.assertIsInstance(result, QuantumCircuit)
        self.assertEqual(result.num_qubits, circuit.num_qubits)

    def test_layout_3239(self, level=3):
        """Test final layout after preset level3 passmanager does not include diagonal gates
        See: https://github.com/Qiskit/qiskit-terra/issues/3239
        """
        qc = QuantumCircuit(5, 5)
        qc.h(0)
        qc.cx(range(3), range(1, 4))
        qc.z(range(4))
        qc.measure(range(4), range(4))
        result = transpile(
            qc,
            basis_gates=["u1", "u2", "u3", "cx"],
            layout_method="trivial",
            optimization_level=level,
        )

        dag = circuit_to_dag(result)
        op_nodes = [node.name for node in dag.topological_op_nodes()]
        self.assertNotIn("u1", op_nodes)  # Check if the diagonal Z-Gates (u1) were removed

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_no_basis_gates(self, level):
        """Test that basis_gates can be None (level={level})"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=None, optimization_level=level)
        self.assertEqual(result, circuit)

    def test_level0_keeps_reset(self):
        """Test level 0 should keep the reset instructions"""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.reset(q[0])
        circuit.reset(q[0])
        result = transpile(circuit, basis_gates=None, optimization_level=0)
        self.assertEqual(result, circuit)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_unitary_is_preserved_if_in_basis_lala(self, level):
        """Test that a unitary is not synthesized if in the basis."""
        qc = QuantumCircuit(2)
        ugate = UnitaryGate(random_unitary(4, seed=42))
        qc.append(ugate, [0, 1])
        qc.measure_all()
        target = Target.from_configuration(
            num_qubits=2,
            basis_gates=["cx", "u", "unitary", "measure"],
            custom_name_mapping={"unitary": ugate},
        )
        result = transpile(qc, target=target, optimization_level=level)
        self.assertEqual(result, qc)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_unitary_is_preserved_if_basis_is_None(self, level):
        """Test that a unitary is not synthesized if basis is None."""
        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=4242), [0, 1])
        qc.measure_all()
        result = transpile(qc, basis_gates=None, optimization_level=level)
        self.assertEqual(result, qc)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_unitary_is_preserved_if_in_basis_synthesis_translation(self, level):
        """Test that a unitary is not synthesized if in the basis with synthesis translation."""
        qc = QuantumCircuit(2)
        ugate = UnitaryGate(random_unitary(4, seed=424242))
        qc.unitary(ugate, [0, 1])
        qc.measure_all()
        target = Target.from_configuration(
            num_qubits=2,
            basis_gates=["cx", "u", "unitary", "measure"],
            custom_name_mapping={"unitary": ugate},
        )
        result = transpile(
            qc,
            target=target,
            optimization_level=level,
            translation_method="synthesis",
        )
        self.assertEqual(result, qc)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_unitary_is_preserved_if_basis_is_None_synthesis_transltion(self, level):
        """Test that a unitary is not synthesized if basis is None with synthesis translation."""
        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=42424242), [0, 1])
        qc.measure_all()
        result = transpile(
            qc, basis_gates=None, optimization_level=level, translation_method="synthesis"
        )
        self.assertEqual(result, qc)

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_respect_basis(self, level):
        """Test that all levels respect basis"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.cp(np.pi / 8, 0, 1)
        qc.cp(np.pi / 4, 0, 2)
        basis_gates = ["id", "rz", "sx", "x", "cx"]
        result = transpile(
            qc, basis_gates=basis_gates, coupling_map=[[0, 1], [2, 1]], optimization_level=level
        )

        dag = circuit_to_dag(result)
        circuit_ops = {node.name for node in dag.topological_op_nodes()}
        self.assertEqual(circuit_ops.union(set(basis_gates)), set(basis_gates))

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_alignment_constraints_called_with_by_default(self, level):
        """Test that TimeUnitConversion is not called if there is no delay in the circuit."""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.cz(q[0], q[1])
        with unittest.mock.patch("qiskit.transpiler.passes.TimeUnitConversion.run") as mock:
            backend = GenericBackendV2(
                num_qubits=20,
                coupling_map=TOKYO_CMAP,
                basis_gates=["id", "u1", "u2", "u3", "cx"],
                seed=42,
            )
            transpile(circuit, backend=backend, optimization_level=level)
        mock.assert_not_called()

    @combine(level=[0, 1, 2, 3], name="level{level}")
    def test_alignment_constraints_called_with_delay_in_circuit(self, level):
        """Test that TimeUnitConversion is called if there is a delay in the circuit."""
        q = QuantumRegister(2, name="q")
        circuit = QuantumCircuit(q)
        circuit.h(q[0])
        circuit.cz(q[0], q[1])
        circuit.delay(9.5, unit="ns")
        with unittest.mock.patch(
            "qiskit.transpiler.passes.TimeUnitConversion.run", return_value=circuit_to_dag(circuit)
        ) as mock:
            backend = GenericBackendV2(
                num_qubits=20,
                coupling_map=TOKYO_CMAP,
                basis_gates=["id", "u1", "u2", "u3", "cx"],
                seed=42,
            )
            transpile(circuit, backend=backend, optimization_level=level)
        mock.assert_called_once()

    def test_unroll_only_if_not_gates_in_basis(self):
        """Test that the list of passes _unroll only runs if a gate is not in the basis."""
        qcomp = GenericBackendV2(
            num_qubits=5,
            coupling_map=BOGOTA_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        qv_circuit = quantum_volume(3)
        gates_in_basis_true_count = 0
        peephole_count = 0

        def counting_callback_func(pass_, dag, time, property_set, count):
            nonlocal gates_in_basis_true_count
            nonlocal peephole_count
            if isinstance(pass_, GatesInBasis) and property_set["all_gates_in_basis"]:
                gates_in_basis_true_count += 1
            if isinstance(pass_, TwoQubitPeepholeOptimization):
                peephole_count += 1

        transpile(
            qv_circuit,
            backend=qcomp,
            optimization_level=3,
            callback=counting_callback_func,
            translation_method="synthesis",
        )
        self.assertEqual(gates_in_basis_true_count, peephole_count)

    @data(0, 1, 2, 3)
    def test_layout_registers_preserved(self, optimization_level):
        """Test that registers in circuit are preserved in TranspileLayout

        Reproduce from `#15014 <https://github.com/Qiskit/qiskit/issues/15014>`__
        """
        a = QuantumRegister(2, "a")
        b = AncillaRegister(1, "b")
        qc = QuantumCircuit(b, a)
        qc.x(0)

        initial_layout = [0, 1, 2]
        qc_transpiled = transpile(
            qc, initial_layout=initial_layout, optimization_level=optimization_level
        )

        self.assertEqual(qc_transpiled.layout.initial_layout.get_registers(), {a, b})

    @data(0, 1, 2, 3)
    def test_layout_registers_preserved_with_allocated_ancilla(self, optimization_level):
        """Test that allocated ancilla register is preserved in output layout."""
        a = QuantumRegister(3, "a")
        b = AncillaRegister(1, "b")
        qc = QuantumCircuit(b, a)
        qc.x(0)
        qc.h(3)
        qc.cx(3, 0)
        qc.cx(3, 1)
        qc.cx(3, 2)

        target = Target.from_configuration(["u", "cx"], coupling_map=CouplingMap.from_line(10))
        tqc = transpile(qc, optimization_level=optimization_level, target=target)
        out_registers = tqc.layout.initial_layout.get_registers()
        self.assertEqual(len(out_registers), 3)
        self.assertIn(a, out_registers)
        self.assertIn(b, out_registers)
        out_registers.remove(a)
        out_registers.remove(b)
        last = out_registers.pop()
        self.assertTrue(last.name.startswith("ancilla"))


@ddt
class TestTranspileLevels(QiskitTestCase):
    """Test transpiler on fake backend"""

    BACKEND_CONFIGS = {
        "bogota": (5, BOGOTA_CMAP),
        "tokyo": (20, TOKYO_CMAP),
        "none": None,
    }

    @staticmethod
    def _backend_from_name(name):
        config = TestTranspileLevels.BACKEND_CONFIGS[name]
        if config is None:
            return None
        num_qubits, coupling_map = config
        return GenericBackendV2(
            num_qubits=num_qubits,
            coupling_map=coupling_map,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )

    @combine(
        circuit=[emptycircuit, circuit_2532],
        level=[0, 1, 2, 3],
        backend=BACKEND_CONFIGS,
        dsc="Transpiler {circuit.__name__} on {backend} backend at level {level}",
        name="{circuit.__name__}_{backend}_level{level}",
    )
    def test(self, circuit, level, backend):
        """All the levels with all the backends"""
        backend = self._backend_from_name(backend)
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_quantum_volume_function_transpile(self, opt_level):
        """Test quantum_volume transpilation."""
        qc = quantum_volume(10, 10, 12345)
        backend = GenericBackendV2(
            num_qubits=100,
            basis_gates=["cz", "rz", "sx", "x", "id"],
            coupling_map=CouplingMap.from_grid(10, 10),
        )
        pm = generate_preset_pass_manager(opt_level, backend)
        res = pm.run(qc)
        for inst in res.data:
            self.assertTrue(
                backend.target.instruction_supported(
                    inst.operation.name, qargs=tuple(res.find_bit(x).index for x in inst.qubits)
                )
            )


@ddt
class TestPassesInspection(QiskitTestCase):
    """Test run passes under different conditions"""

    def setUp(self):
        """Sets self.callback to set self.passes with the passes that have been executed"""
        super().setUp()
        self.passes = []

        def callback(**kwargs):
            self.passes.append(kwargs["pass_"].__class__.__name__)

        self.callback = callback

    @data(0, 1, 2, 3)
    def test_no_coupling_map(self, level):
        """Without coupling map, no layout selection nor swapper"""
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])

        _ = transpile(qc, optimization_level=level, callback=self.callback)

        self.assertNotIn("SetLayout", self.passes)
        self.assertNotIn("TrivialLayout", self.passes)
        self.assertNotIn("ApplyLayout", self.passes)
        self.assertNotIn("StochasticSwap", self.passes)
        self.assertNotIn("SabreSwap", self.passes)
        self.assertNotIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_backend(self, level):
        """With backend a layout and a swapper is run"""
        qr = QuantumRegister(5, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[2], qr[4])

        backend = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP, seed=42)

        _ = transpile(qc, backend, optimization_level=level, callback=self.callback)

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_symmetric_coupling_map(self, level):
        """Symmetric coupling map does not run CheckGateDirection"""
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])

        coupling_map = [[0, 1], [1, 0]]

        _ = transpile(
            qc,
            coupling_map=coupling_map,
            initial_layout=[0, 1],
            optimization_level=level,
            callback=self.callback,
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertNotIn("CheckGateDirection", self.passes)

    @data(0, 1, 2, 3)
    def test_initial_layout_fully_connected_cm(self, level):
        """Honor initial_layout when coupling_map=None
        See: https://github.com/Qiskit/qiskit-terra/issues/5345
        """
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        transpiled = transpile(
            qc, initial_layout=[0, 1], optimization_level=level, callback=self.callback
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        self.assertEqual(transpiled._layout.initial_layout, Layout.from_qubit_list([qr[0], qr[1]]))

    @data(0, 1, 2, 3)
    def test_partial_layout_fully_connected_cm(self, level):
        """Honor initial_layout (partially defined) when coupling_map=None
        See: https://github.com/Qiskit/qiskit-terra/issues/5345
        """
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])

        transpiled = transpile(
            qc, initial_layout=[4, 2], optimization_level=level, callback=self.callback
        )

        self.assertIn("SetLayout", self.passes)
        self.assertIn("ApplyLayout", self.passes)
        ancilla = QuantumRegister(3, "ancilla")
        self.assertEqual(
            transpiled._layout.initial_layout,
            Layout.from_qubit_list([ancilla[0], ancilla[1], qr[1], ancilla[2], qr[0]]),
        )

    @unittest.mock.patch.object(
        level0.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages(self, _plugin_manager_mock):
        """Test transpile() executes backend specific custom stage."""
        optimization_level = 1

        class TargetBackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = TargetBackend(num_qubits=7, seed=42)
        qr = QuantumRegister(2, "q")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        _ = transpile(qc, target, optimization_level=optimization_level, callback=self.callback)
        self.assertIn("ALAPScheduleAnalysis", self.passes)
        self.assertIn("PadDynamicalDecoupling", self.passes)
        self.assertIn("RemoveResetInZeroState", self.passes)

    def test_level1_runs_vf2post_layout_when_routing_required(self):
        """Test that if we run routing as part of sabre layout VF2PostLayout runs."""
        target = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cy(0, 1)
        qc.cy(0, 2)
        qc.cy(0, 3)
        qc.cy(0, 4)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        # Expected call path for layout and routing is:
        # 1. TrivialLayout (no perfect match)
        # 2. VF2Layout (no perfect match)
        # 3. SabreLayout (heuristic layout and also runs routing)
        # 4. VF2PostLayout (applies a better layout)
        self.assertIn("TrivialLayout", self.passes)
        self.assertIn("VF2Layout", self.passes)
        self.assertIn("SabreLayout", self.passes)
        self.assertIn("VF2PostLayout", self.passes)
        #  Assert we don't run standalone sabre swap
        self.assertNotIn("SabreSwap", self.passes)

    def test_level1_runs_vf2post_layout_when_routing_method_set_and_required(self):
        """Test that if we run routing as part of sabre layout then VF2PostLayout runs."""
        target = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cy(0, 1)
        qc.cy(0, 2)
        qc.cy(0, 3)
        qc.cy(0, 4)
        qc.measure_all()
        _ = transpile(
            qc, target, optimization_level=1, routing_method="sabre", callback=self.callback
        )
        # Expected call path for layout and routing is:
        # 1. TrivialLayout (no perfect match)
        # 2. VF2Layout (no perfect match)
        # 3. SabreLayout (heuristic layout and also runs routing)
        # 4. VF2PostLayout (applies a better layout)
        self.assertIn("TrivialLayout", self.passes)
        self.assertIn("VF2Layout", self.passes)
        self.assertIn("SabreLayout", self.passes)
        self.assertIn("VF2PostLayout", self.passes)

    def test_level1_not_runs_vf2post_layout_when_layout_method_set(self):
        """Test that if we don't run VF2PostLayout with custom layout_method."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
        )
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cy(0, 1)
        qc.cy(0, 2)
        qc.cy(0, 3)
        qc.cy(0, 4)
        qc.measure_all()
        _ = transpile(
            qc, target, optimization_level=1, layout_method="dense", callback=self.callback
        )
        self.assertNotIn("TrivialLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        self.assertIn("DenseLayout", self.passes)
        self.assertIn("SabreSwap", self.passes)

    def test_level1_not_run_vf2post_layout_when_trivial_is_perfect(self):
        """Test that if we find a trivial perfect layout we don't run vf2post."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        self.assertIn("TrivialLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        #  Assert we don't run standalone sabre swap
        self.assertNotIn("SabreSwap", self.passes)

    def test_level1_not_run_vf2post_layout_when_vf2layout_is_perfect(self):
        """Test that if we find a vf2 perfect layout we don't run vf2post."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
        )
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        self.assertIn("TrivialLayout", self.passes)
        self.assertIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        #  Assert we don't run standalone sabre swap
        self.assertNotIn("SabreSwap", self.passes)

    def test_level1_runs_vf2post_layout_when_routing_required_control_flow(self):
        """Test that if we run routing as part of sabre layout VF2PostLayout runs."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cy(0, 1)
        qc.cy(0, 2)
        qc.cy(0, 3)
        qc.cy(0, 4)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        # Expected call path for layout and routing is:
        # 1. TrivialLayout (no perfect match)
        # 2. VF2Layout (no perfect match)
        # 3. SabreLayout (heuristic layout)
        # 4. VF2PostLayout (applies a better layout)
        self.assertIn("TrivialLayout", self.passes)
        self.assertIn("VF2Layout", self.passes)
        self.assertIn("SabreLayout", self.passes)
        self.assertIn("VF2PostLayout", self.passes)

    def test_level1_not_runs_vf2post_layout_when_layout_method_set_control_flow(self):
        """Test that if we don't run VF2PostLayout with custom layout_method."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cy(0, 1)
        qc.cy(0, 2)
        qc.cy(0, 3)
        qc.cy(0, 4)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(
            qc, target, optimization_level=1, layout_method="dense", callback=self.callback
        )
        self.assertNotIn("TrivialLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        self.assertIn("DenseLayout", self.passes)
        self.assertIn("SabreSwap", self.passes)

    def test_level1_not_run_vf2post_layout_when_trivial_is_perfect_control_flow(self):
        """Test that if we find a trivial perfect layout we don't run vf2post."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        self.assertIn("TrivialLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("SabreSwap", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)

    def test_level1_not_run_vf2post_layout_when_vf2layout_is_perfect_control_flow(self):
        """Test that if we find a vf2 perfect layout we don't run vf2post."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(qc, target, optimization_level=1, callback=self.callback)
        self.assertIn("TrivialLayout", self.passes)
        self.assertIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        self.assertNotIn("SabreSwap", self.passes)

    def test_level3_does_not_run_vf2post_layout_when_initial_layout_set(self):
        """Test that level 3 does not run VF2PostLayout when an initial layout is set."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(
            qc, target, optimization_level=1, callback=self.callback, initial_layout=[0, 1, 5, 6]
        )
        self.assertIn("SetLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)

    def test_level3_does_not_run_vf2post_layout_when_layout_method_set(self):
        """Test that level 3 does not run VF2PostLayout when layout_method is set."""
        target = GenericBackendV2(
            num_qubits=7,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=LAGOS_CMAP,
            seed=42,
            control_flow=True,
        )
        qc = QuantumCircuit(4)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with qc.for_loop((1,)):
            qc.cx(0, 1)
        qc.measure_all()
        _ = transpile(
            qc, target, optimization_level=1, callback=self.callback, layout_method="dense"
        )
        self.assertIn("SetLayout", self.passes)
        self.assertNotIn("VF2Layout", self.passes)
        self.assertNotIn("SabreLayout", self.passes)
        self.assertNotIn("VF2PostLayout", self.passes)
        self.assertIn("DenseLayout", self.passes)


@ddt
class TestInitialLayouts(QiskitTestCase):
    """Test transpiling with different layouts"""

    @data(0, 1, 2, 3)
    def test_layout_1711(self, level):
        """Test that a user-given initial layout is respected
        in the output.

        See: https://github.com/Qiskit/qiskit-terra/issues/1711
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(3)
        ancilla = QuantumRegister(13, "ancilla")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[1])
        qc.cx(qr[2], qr[0])
        initial_layout = {0: qr[1], 2: qr[0], 15: qr[2]}
        final_layout = {
            0: qr[1],
            1: ancilla[0],
            2: qr[0],
            3: ancilla[1],
            4: ancilla[2],
            5: ancilla[3],
            6: ancilla[4],
            7: ancilla[5],
            8: ancilla[6],
            9: ancilla[7],
            10: ancilla[8],
            11: ancilla[9],
            12: ancilla[10],
            13: ancilla[11],
            14: ancilla[12],
            15: qr[2],
        }
        backend = GenericBackendV2(num_qubits=16, coupling_map=RUESCHLIKON_CMAP, seed=42)
        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout.initial_layout._p2v, final_layout)

        for inst in qc_b.data:
            if inst.operation.name == "cx":
                self.assertIn(
                    tuple(qc_b.find_bit(bit).index for bit in inst.qubits), backend.coupling_map
                )
                self.assertIn([qc_b.find_bit(bit).index for bit in inst.qubits], [[15, 0], [15, 2]])

    @data(0, 1, 2, 3)
    def test_layout_2532(self, level):
        """Test that a user-given initial layout is respected,
        in the transpiled circuit.

        See: https://github.com/Qiskit/qiskit-terra/issues/2532
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(5, "q")
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(9, "ancilla")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[2], qr[4])
        initial_layout = {
            qr[2]: 11,
            qr[4]: 3,  # map to [11, 3] connection
            qr[0]: 1,
            qr[1]: 5,
            qr[3]: 9,
        }
        final_layout = {
            0: ancilla[0],
            1: qr[0],
            2: ancilla[1],
            3: qr[4],
            4: ancilla[2],
            5: qr[1],
            6: ancilla[3],
            7: ancilla[4],
            8: ancilla[5],
            9: qr[3],
            10: ancilla[6],
            11: qr[2],
            12: ancilla[7],
            13: ancilla[8],
        }
        backend = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP, seed=42)
        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)
        self.assertEqual(qc_b._layout.initial_layout._p2v, final_layout)

        output_qr = qc_b.qregs[0]
        for instruction in qc_b:
            if instruction.operation.name == "cx":
                for qubit in instruction.qubits:
                    self.assertIn(qubit, [output_qr[11], output_qr[3]])

    @data(0, 1, 2, 3)
    def test_layout_2503(self, level):
        """Test that a user-given initial layout is respected,
        even if cnots are not in the coupling map.

        See: https://github.com/Qiskit/qiskit-terra/issues/2503
        """
        # build a circuit which works as-is on the coupling map, using the initial layout
        qr = QuantumRegister(3, "q")
        cr = ClassicalRegister(2)
        ancilla = QuantumRegister(17, "ancilla")

        qc = QuantumCircuit(qr, cr)
        qc.append(U3Gate(0.1, 0.2, 0.3), [qr[0]])
        qc.append(U2Gate(0.4, 0.5), [qr[2]])
        qc.barrier()
        qc.cx(qr[0], qr[2])
        initial_layout = [6, 7, 12]

        final_layout = {
            0: ancilla[0],
            1: ancilla[1],
            2: ancilla[2],
            3: ancilla[3],
            4: ancilla[4],
            5: ancilla[5],
            6: qr[0],
            7: qr[1],
            8: ancilla[6],
            9: ancilla[7],
            10: ancilla[8],
            11: ancilla[9],
            12: qr[2],
            13: ancilla[10],
            14: ancilla[11],
            15: ancilla[12],
            16: ancilla[13],
            17: ancilla[14],
            18: ancilla[15],
            19: ancilla[16],
        }

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        qc_b = transpile(qc, backend, initial_layout=initial_layout, optimization_level=level)

        self.assertEqual(qc_b._layout.initial_layout._p2v, final_layout)

        output_qr = qc_b.qregs[0]
        self.assertIsInstance(qc_b[0].operation, U3Gate)
        self.assertEqual(qc_b[0].qubits[0], output_qr[6])
        self.assertIsInstance(qc_b[1].operation, U2Gate)
        self.assertEqual(qc_b[1].qubits[0], output_qr[12])


@ddt
class TestFinalLayouts(QiskitTestCase):
    """Test final layouts after preset transpilation"""

    @data(0, 1, 2, 3)
    def test_layout_tokyo_2845(self, level):
        """Test that final layout in a Tokyo-like device
         is not the trivial layout for optimization level>0
        See: https://github.com/Qiskit/qiskit-terra/issues/2845
        """
        qr1 = QuantumRegister(3, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qc = QuantumCircuit(qr1, qr2)
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[2], qr2[0])
        qc.cx(qr2[0], qr2[1])

        ancilla = QuantumRegister(15, "ancilla")
        trivial_layout = {
            0: qr1[0],
            1: qr1[1],
            2: qr1[2],
            3: qr2[0],
            4: qr2[1],
            5: ancilla[0],
            6: ancilla[1],
            7: ancilla[2],
            8: ancilla[3],
            9: ancilla[4],
            10: ancilla[5],
            11: ancilla[6],
            12: ancilla[7],
            13: ancilla[8],
            14: ancilla[9],
            15: ancilla[10],
            16: ancilla[11],
            17: ancilla[12],
            18: ancilla[13],
            19: ancilla[14],
        }

        vf2_layout = {
            0: ancilla[0],
            1: ancilla[1],
            2: ancilla[2],
            3: ancilla[3],
            4: ancilla[4],
            5: qr1[2],
            6: qr2[0],
            7: qr2[1],
            8: ancilla[5],
            9: ancilla[6],
            10: qr1[1],
            11: qr1[0],
            12: ancilla[7],
            13: ancilla[8],
            14: ancilla[9],
            15: ancilla[10],
            16: ancilla[11],
            17: ancilla[12],
            18: ancilla[13],
            19: ancilla[14],
        }

        # Trivial layout
        expected_layout_level0 = trivial_layout
        # Dense layout
        expected_layout_level1 = vf2_layout
        # CSP layout
        expected_layout_level2 = vf2_layout
        expected_layout_level3 = vf2_layout

        expected_layouts = [
            expected_layout_level0,
            expected_layout_level1,
            expected_layout_level2,
            expected_layout_level3,
        ]

        backend = GenericBackendV2(num_qubits=20, coupling_map=TOKYO_CMAP, seed=42)
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout.initial_layout._p2v, expected_layouts[level])

    @data(0, 1, 2, 3)
    def test_layout_tokyo_fully_connected_cx(self, level):
        """Test that final layout in a Tokyo-like device
        is a fully connected circuit
        """
        qr = QuantumRegister(5, "qr")
        qc = QuantumCircuit(qr)
        for qubit_target in qr:
            for qubit_control in qr:
                if qubit_control != qubit_target:
                    qc.cx(qubit_control, qubit_target)
        expected_layouts = [
            [0, 1, 2, 3, 4],
            [5, 6, 10, 0, 11],
            [5, 6, 10, 0, 11],
            [5, 6, 10, 0, 11],
        ]
        backend = GenericBackendV2(num_qubits=20, coupling_map=TOKYO_CMAP, seed=42)
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(
            result.layout.initial_index_layout(filter_ancillas=True), expected_layouts[level]
        )

    @data(0, 1)
    def test_low_levels_use_trivial_if_perfect(self, level):
        """Test that we use trivial if it's a perfect match at levels 0 and 1.  Higher levels use
        VF2 instead, which may not always return the trivial layout based on the scoring.

        See: https://github.com/Qiskit/qiskit-terra/issues/5694 for more details.
        """
        backend = GenericBackendV2(num_qubits=20, coupling_map=TOKYO_CMAP, seed=42)

        rows = [x[0] for x in backend.coupling_map]
        cols = [x[1] for x in backend.coupling_map]

        adjacency_matrix = np.zeros((20, 20))
        adjacency_matrix[rows, cols] = 1
        qc = GraphStateGate(adjacency_matrix).definition
        qc.measure_all()
        expected = {
            0: Qubit(QuantumRegister(20, "q"), 0),
            1: Qubit(QuantumRegister(20, "q"), 1),
            2: Qubit(QuantumRegister(20, "q"), 2),
            3: Qubit(QuantumRegister(20, "q"), 3),
            4: Qubit(QuantumRegister(20, "q"), 4),
            5: Qubit(QuantumRegister(20, "q"), 5),
            6: Qubit(QuantumRegister(20, "q"), 6),
            7: Qubit(QuantumRegister(20, "q"), 7),
            8: Qubit(QuantumRegister(20, "q"), 8),
            9: Qubit(QuantumRegister(20, "q"), 9),
            10: Qubit(QuantumRegister(20, "q"), 10),
            11: Qubit(QuantumRegister(20, "q"), 11),
            12: Qubit(QuantumRegister(20, "q"), 12),
            13: Qubit(QuantumRegister(20, "q"), 13),
            14: Qubit(QuantumRegister(20, "q"), 14),
            15: Qubit(QuantumRegister(20, "q"), 15),
            16: Qubit(QuantumRegister(20, "q"), 16),
            17: Qubit(QuantumRegister(20, "q"), 17),
            18: Qubit(QuantumRegister(20, "q"), 18),
            19: Qubit(QuantumRegister(20, "q"), 19),
        }
        trans_qc = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(trans_qc._layout.initial_layout._p2v, expected)

    @data(0)
    def test_trivial_layout(self, level):
        """Test that trivial layout is preferred in level 0
        See: https://github.com/Qiskit/qiskit-terra/pull/3657#pullrequestreview-342012465
        """
        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[6])
        qc.cx(qr[3], qr[8])
        qc.cx(qr[4], qr[9])
        qc.cx(qr[9], qr[8])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[7], qr[6])
        qc.cx(qr[6], qr[5])
        qc.cx(qr[5], qr[0])

        ancilla = QuantumRegister(10, "ancilla")
        trivial_layout = {
            0: qr[0],
            1: qr[1],
            2: qr[2],
            3: qr[3],
            4: qr[4],
            5: qr[5],
            6: qr[6],
            7: qr[7],
            8: qr[8],
            9: qr[9],
            10: ancilla[0],
            11: ancilla[1],
            12: ancilla[2],
            13: ancilla[3],
            14: ancilla[4],
            15: ancilla[5],
            16: ancilla[6],
            17: ancilla[7],
            18: ancilla[8],
            19: ancilla[9],
        }

        expected_layouts = [trivial_layout, trivial_layout]

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        result = transpile(qc, backend, optimization_level=level, seed_transpiler=42)
        self.assertEqual(result._layout.initial_layout._p2v, expected_layouts[level])

    @data(0, 1, 2, 3)
    def test_initial_layout(self, level):
        """When a user provides a layout (initial_layout), it should be used."""
        qr = QuantumRegister(10, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[0], qr[1])
        qc.cx(qr[1], qr[2])
        qc.cx(qr[2], qr[3])
        qc.cx(qr[3], qr[9])
        qc.cx(qr[4], qr[9])
        qc.cx(qr[9], qr[8])
        qc.cx(qr[8], qr[7])
        qc.cx(qr[7], qr[6])
        qc.cx(qr[6], qr[5])
        qc.cx(qr[5], qr[0])

        initial_layout = {
            0: qr[0],
            2: qr[1],
            4: qr[2],
            6: qr[3],
            8: qr[4],
            10: qr[5],
            12: qr[6],
            14: qr[7],
            16: qr[8],
            18: qr[9],
        }

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        result = transpile(
            qc, backend, optimization_level=level, initial_layout=initial_layout, seed_transpiler=42
        )

        for physical, virtual in initial_layout.items():
            self.assertEqual(result._layout.initial_layout._p2v[physical], virtual)


@ddt
class TestTranspileLevelsSwap(QiskitTestCase):
    """Test if swap is in the basis, do not unroll
    See https://github.com/Qiskit/qiskit-terra/pull/3963
    The circuit in combine should require a swap and that swap should exit at the end
    for the transpilation"""

    @combine(
        circuit=[circuit_2532],
        level=[0, 1, 2, 3],
        dsc="circuit: {circuit.__name__}, level: {level}",
        name="{circuit.__name__}_level{level}",
    )
    def test_1(self, circuit, level):
        """Simple coupling map (linear 5 qubits)."""
        basis = ["u1", "u2", "cx", "swap"]
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        result = transpile(
            circuit(),
            optimization_level=level,
            basis_gates=basis,
            coupling_map=coupling_map,
            seed_transpiler=42,
            initial_layout=[0, 1, 2, 3, 4],
        )
        self.assertIsInstance(result, QuantumCircuit)
        resulting_basis = {node.name for node in circuit_to_dag(result).op_nodes()}
        self.assertIn("swap", resulting_basis)

    # Skipping optimization level 3 because the swap gates get absorbed into
    # a unitary block as part of the KAK decomposition optimization passes and
    # optimized away.
    @combine(
        level=[0, 1, 2],
        dsc="If swap in basis, do not decompose it. level: {level}",
        name="level{level}",
    )
    def test_2(self, level):
        """Simple coupling map (linear 5 qubits).
        The circuit requires a swap and that swap should exit at the end
        for the transpilation"""
        basis = ["u1", "u2", "cx", "swap"]
        circuit = QuantumCircuit(5)
        circuit.cx(0, 4)
        circuit.cx(1, 4)
        circuit.cx(2, 4)
        circuit.cx(3, 4)
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        result = transpile(
            circuit,
            optimization_level=level,
            basis_gates=basis,
            coupling_map=coupling_map,
            seed_transpiler=421234242,
        )
        self.assertIsInstance(result, QuantumCircuit)
        resulting_basis = {node.name for node in circuit_to_dag(result).op_nodes()}
        self.assertIn("swap", resulting_basis)


@ddt
class TestOptimizationOnSize(QiskitTestCase):
    """Test the optimization levels for optimization based on
    both size and depth of the circuit.
    See https://github.com/Qiskit/qiskit-terra/pull/7542
    """

    @data(2, 3)
    def test_size_optimization(self, level):
        """Test the levels for optimization based on size of circuit"""
        qc = QuantumCircuit(8)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(5, 4)
        qc.cx(6, 5)
        qc.cx(4, 5)
        qc.cx(3, 4)
        qc.cx(5, 6)
        qc.cx(5, 4)
        qc.cx(3, 4)
        qc.cx(2, 3)
        qc.cx(1, 2)
        qc.cx(6, 7)
        qc.cx(6, 5)
        qc.cx(5, 4)
        qc.cx(7, 6)
        qc.cx(6, 7)

        circ = transpile(qc, optimization_level=level).decompose()

        circ_data = circ.data
        free_qubits = {0, 1, 2, 3}

        # ensure no gates are using qubits - [0,1,2,3]
        for gate in circ_data:
            indices = {circ.find_bit(qubit).index for qubit in gate.qubits}
            common = indices.intersection(free_qubits)
            for common_qubit in common:
                self.assertTrue(common_qubit not in free_qubits)

        self.assertLess(circ.size(), qc.size())
        self.assertLessEqual(circ.depth(), qc.depth())


@ddt
class TestGeneratePresetPassManagers(QiskitTestCase):
    """Test generate_preset_pass_manager function."""

    @data(0, 1, 2, 3)
    def test_with_backend(self, optimization_level):
        """Test a passmanager is constructed when only a backend and optimization level."""
        backend = GenericBackendV2(num_qubits=2)
        pm = generate_preset_pass_manager(optimization_level, backend)
        self.assertIsInstance(pm, PassManager)

    def test_default_optimization_level(self):
        """Test a pass manager is constructed with no optimization level."""
        backend = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP)
        pm = generate_preset_pass_manager(backend=backend)
        self.assertIsInstance(pm, PassManager)

    def test_default_optimization_level_backend_first_pos_arg(self):
        """Test a pass manager is constructed with only a positional backend."""
        backend = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP)
        pm = generate_preset_pass_manager(backend)
        self.assertIsInstance(pm, PassManager)

    def test_default_optimization_level_target_first_pos_arg(self):
        """Test a pass manager is constructed with only a positional target."""
        backend = GenericBackendV2(num_qubits=14, coupling_map=MELBOURNE_CMAP)
        pm = generate_preset_pass_manager(backend.target)
        self.assertIsInstance(pm, PassManager)

    @data(0, 1, 2, 3)
    def test_with_no_backend(self, optimization_level):
        """Test a passmanager is constructed with no backend and optimization level."""
        target = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(
            optimization_level,
            coupling_map=target.coupling_map,
            basis_gates=target.operation_names,
            target=target.target,
        )
        self.assertIsInstance(pm, PassManager)

    @data(0, 1, 2, 3)
    def test_with_no_backend_only_target(self, optimization_level):
        """Test a passmanager is constructed with a manual target and optimization level."""
        target = GenericBackendV2(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(optimization_level, target=target.target)
        self.assertIsInstance(pm, PassManager)

    def test_invalid_optimization_level(self):
        """Assert we fail with an invalid optimization_level."""
        with self.assertRaises(ValueError):
            generate_preset_pass_manager(42)

    @unittest.mock.patch.object(
        level2.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level2(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 2

        class TargetBackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = TargetBackend(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(optimization_level, backend=target)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level1.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level1(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 1

        class TargetBackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = TargetBackend(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(optimization_level, backend=target)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level3.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level3(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 3

        class TargetBackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = TargetBackend(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(optimization_level, backend=target)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [
            x.__class__.__name__ for x in pm.translation.to_flow_controller().tasks
        ]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    @unittest.mock.patch.object(
        level0.PassManagerStagePluginManager,
        "get_passmanager_stage",
        wraps=mock_get_passmanager_stage,
    )
    def test_backend_with_custom_stages_level0(self, _plugin_manager_mock):
        """Test generated preset pass manager includes backend specific custom stages."""
        optimization_level = 0

        class TargetBackend(GenericBackendV2):
            """Fake lagos subclass with custom transpiler stages."""

            def get_scheduling_stage_plugin(self):
                """Custom scheduling stage."""
                return "custom_stage_for_test"

            def get_translation_stage_plugin(self):
                """Custom post translation stage."""
                return "custom_stage_for_test"

        target = TargetBackend(num_qubits=7, coupling_map=LAGOS_CMAP, seed=42)
        pm = generate_preset_pass_manager(optimization_level, backend=target)
        self.assertIsInstance(pm, PassManager)

        pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("PadDynamicalDecoupling", pass_list)
        self.assertIn("ALAPScheduleAnalysis", pass_list)
        post_translation_pass_list = [x.__class__.__name__ for x in pm.to_flow_controller().tasks]
        self.assertIn("RemoveResetInZeroState", post_translation_pass_list)

    def test_generate_preset_pass_manager_with_list_coupling_map(self):
        """Test that generate_preset_pass_manager can handle list-based coupling_map."""

        # Define the coupling map as a list
        coupling_map_list = [[0, 1]]
        coupling_map_object = CouplingMap(coupling_map_list)

        # Circuit that doesn't fit in the coupling map
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.measure_all()

        pm_list = generate_preset_pass_manager(
            optimization_level=0, coupling_map=coupling_map_list, seed_transpiler=42
        )
        pm_object = generate_preset_pass_manager(
            optimization_level=0, coupling_map=coupling_map_object, seed_transpiler=42
        )

        transpiled_circuit_list = pm_list.run(qc)
        transpiled_circuit_object = pm_object.run(qc)

        # Check if both are instances of PassManager
        self.assertIsInstance(pm_list, PassManager)
        self.assertIsInstance(pm_object, PassManager)

        # Ensure the DAGs from both methods are identical
        self.assertEqual(transpiled_circuit_list, transpiled_circuit_object)

    @data(0, 1, 2, 3)
    def test_generate_preset_pass_manager_with_list_initial_layout(self, optimization_level):
        """Test that generate_preset_pass_manager can handle list based initial layouts."""
        coupling_map_list = [[0, 1]]

        # Circuit that doesn't fit in the coupling map
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.measure_all()

        pm_list = generate_preset_pass_manager(
            optimization_level=optimization_level,
            coupling_map=coupling_map_list,
            basis_gates=["u", "cx"],
            seed_transpiler=42,
            initial_layout=[1, 0],
        )
        pm_object = generate_preset_pass_manager(
            optimization_level=optimization_level,
            coupling_map=coupling_map_list,
            basis_gates=["u", "cx"],
            seed_transpiler=42,
            initial_layout=Layout.from_intlist([1, 0], *qc.qregs),
        )
        tqc_list = pm_list.run(qc)
        tqc_obj = pm_list.run(qc)
        self.assertIsInstance(pm_list, PassManager)
        self.assertIsInstance(pm_object, PassManager)
        self.assertEqual(tqc_list, tqc_obj)

    def test_parse_seed_transpiler_raises_value_error(self):
        """Test that seed for transpiler is non-negative integer."""
        with self.assertRaisesRegex(
            ValueError, "Expected non-negative integer as seed for transpiler."
        ):
            generate_preset_pass_manager(optimization_level=1, seed_transpiler=-1)

        with self.assertRaisesRegex(
            ValueError, "Expected non-negative integer as seed for transpiler."
        ):
            generate_preset_pass_manager(seed_transpiler=0.1)

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

        with unittest.mock.patch.dict(os.environ, {"QISKIT_TRANSPILER_SEED": "42"}):
            with unittest.mock.patch.object(SabreLayout, "run", new=mock_run):
                pm = generate_preset_pass_manager(
                    optimization_level=1, coupling_map=CouplingMap.from_line(3)
                )
                _ = pm.run(qc)

        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0], 42)

    @combine(
        optimization_level=[0, 1, 2, 3],
        layout_method=["default", "dense", "sabre"],
        routing_method=["default", "sabre"],
        translation_method=["default", "translator"],
    )
    def test_preserves_circuit_metadata(
        self, optimization_level, layout_method, routing_method, translation_method
    ):
        """Test that basic metadata is preserved."""
        metadata = {"experiment_id": "1234", "execution_number": 4}
        name = "my circuit"
        circuit = QuantumCircuit(4, metadata=metadata.copy(), name=name)
        circuit.h(0)
        circuit.cx(0, 3)

        # Add a custom gate to trigger HighLevelSynthesis
        my_h = QuantumCircuit(1, name="my_h")
        my_h.h(0)
        circuit.append(my_h.to_gate(), (1,))

        num_qubits = 10
        target = Target(num_qubits)
        for inst in (lib.IGate(), lib.PhaseGate(Parameter("t")), lib.SXGate()):
            target.add_instruction(inst, {(i,): None for i in range(num_qubits)})
        target.add_instruction(CXGate(), dict.fromkeys(CouplingMap.from_line(num_qubits)))

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            target=target,
            seed_transpiler=42,
            layout_method=layout_method,
            routing_method=routing_method,
            translation_method=translation_method,
        )
        res = pm.run(circuit)
        self.assertEqual(res.metadata, metadata)
        self.assertEqual(res.name, name)

    def test_timing_constraints_from_target_no_backend(self):
        """Test that timing constrains are obtained from target when no backend is provided."""
        target = Target.from_configuration(
            basis_gates=["sx", "rz", "cz", "measure", "delay"],
            coupling_map=CouplingMap.from_line(4),
            timing_constraints=TimingConstraints(acquire_alignment=2),
        )
        pm = generate_preset_pass_manager(optimization_level=2, target=target)

        timing_constraint_from_target = False

        # Check to ensure that one of the tasks is of the type 'InstructionDurationsCheck'
        for task in pm.scheduling.__dict__["_tasks"]:
            if isinstance(task[0], InstructionDurationCheck):
                self.assertEqual(task[0].acquire_align, 2)
                timing_constraint_from_target = True

        self.assertTrue(timing_constraint_from_target)


@ddt
class TestIntegrationControlFlow(QiskitTestCase):
    """Integration tests for control-flow circuits through the preset pass managers."""

    @data(0, 1, 2, 3)
    def test_default_compilation(self, optimization_level):
        """Test that a simple circuit with each type of control-flow passes a full transpilation
        pipeline with the defaults."""

        class CustomCX(Gate):
            """Custom CX"""

            def __init__(self):
                super().__init__("custom_cx", 2, [])

            def _define(self):
                self._definition = QuantumCircuit(2)
                self._definition.cx(0, 1)

        circuit = QuantumCircuit(6, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.cz(0, 2)
        circuit.append(CustomCX(), [1, 2], [])
        with circuit.for_loop((1,)):
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with else_:
            circuit.cx(3, 4)
            circuit.cz(3, 5)
            circuit.append(CustomCX(), [4, 5], [])
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(3, 4)
                circuit.cz(3, 5)
                circuit.append(CustomCX(), [4, 5], [])

        coupling_map = CouplingMap.from_line(6)

        transpiled = transpile(
            circuit,
            basis_gates=["sx", "rz", "cx", "if_else", "for_loop", "while_loop"],
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_04,
        )
        # Tests of the complete validity of a circuit are mostly done at the individual pass level;
        # here we're just checking that various passes do appear to have run.
        self.assertIsInstance(transpiled, QuantumCircuit)
        # Assert layout ran.
        self.assertIsNot(getattr(transpiled, "_layout", None), None)

        def _visit_block(circuit, stack=None):
            """Assert that every block contains at least one swap to imply that routing has run."""
            if stack is None:
                # List of (instruction_index, block_index).
                stack = ()
            seen_cx = 0
            for i, instruction in enumerate(circuit):
                if isinstance(instruction.operation, ControlFlowOp):
                    for j, block in enumerate(instruction.operation.blocks):
                        _visit_block(block, stack + ((i, j),))
                elif isinstance(instruction.operation, CXGate):
                    seen_cx += 1
                # Assert unrolling ran.
                self.assertNotIsInstance(instruction.operation, CustomCX)
                # Assert translation ran.
                self.assertNotIsInstance(instruction.operation, CZGate)
            # There are three "natural" swaps in each block (one for each 2q operation), so if
            # routing ran, we should see more than that.
            self.assertGreater(seen_cx, 3, msg=f"no swaps in block at indices: {stack}")

        # Assert routing ran.
        _visit_block(transpiled)

    @data(0, 1, 2, 3)
    def test_allow_overriding_defaults(self, optimization_level):
        """Test that the method options can be overridden."""
        circuit = QuantumCircuit(3, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        with circuit.for_loop((1,)):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.cx(1, 2)

        coupling_map = CouplingMap.from_line(3)

        calls = set()

        def callback(pass_, **_):
            calls.add(pass_.name())

        transpiled = transpile(
            circuit,
            basis_gates=["u3", "cx", "if_else", "for_loop", "while_loop"],
            layout_method="trivial",
            translation_method="translator",
            coupling_map=coupling_map,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_04,
            callback=callback,
        )
        self.assertIsInstance(transpiled, QuantumCircuit)
        self.assertIsNot(getattr(transpiled, "_layout", None), None)

        self.assertIn("TrivialLayout", calls)
        self.assertIn("BasisTranslator", calls)
        self.assertNotIn("DenseLayout", calls)
        self.assertNotIn("SabreLayout", calls)

    @data(0, 1, 2, 3)
    def test_invalid_methods_raise_on_control_flow(self, optimization_level):
        """Test that trying to use an invalid method with control flow fails."""
        qc = QuantumCircuit(1)
        with qc.for_loop((1,)):
            qc.x(0)

        with self.assertRaisesRegex(TranspilerError, "Got routing_method="):
            transpile(qc, routing_method="lookahead", optimization_level=optimization_level)
        with self.assertRaisesRegex(TranspilerError, "Got scheduling_method="):
            transpile(qc, scheduling_method="alap", optimization_level=optimization_level)

    @data(0, 1, 2, 3)
    def test_unsupported_basis_gates_raise(self, optimization_level):
        """Test that trying to transpile a control-flow circuit for a backend that doesn't support
        the necessary operations in its `basis_gates` will raise a sensible error."""
        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        qc = QuantumCircuit(1, 1)
        with qc.for_loop((0,)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, backend, optimization_level=optimization_level)

        qc = QuantumCircuit(1, 1)
        with qc.if_test((qc.clbits[0], False)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, backend, optimization_level=optimization_level)

        qc = QuantumCircuit(1, 1)
        with qc.while_loop((qc.clbits[0], False)):
            pass
        with qc.for_loop((0, 1, 2)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, backend, optimization_level=optimization_level)

    @data(0, 1, 2, 3)
    def test_unsupported_targets_raise(self, optimization_level):
        """Test that trying to transpile a control-flow circuit for a backend that doesn't support
        the necessary operations in its `Target` will raise a more sensible error."""
        target = Target(num_qubits=2)
        target.add_instruction(CXGate(), {(0, 1): None})

        qc = QuantumCircuit(1, 1)
        with qc.for_loop((0,)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, target=target, optimization_level=optimization_level)

        qc = QuantumCircuit(1, 1)
        with qc.if_test((qc.clbits[0], False)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, target=target, optimization_level=optimization_level)

        qc = QuantumCircuit(1, 1)
        with qc.while_loop((qc.clbits[0], False)):
            pass
        with qc.for_loop((0, 1, 2)):
            pass
        with self.assertRaisesRegex(TranspilerError, "The control-flow construct.*not supported"):
            transpile(qc, target=target, optimization_level=optimization_level)

    @data(0, 1, 2, 3)
    def test_custom_basis_gates_raise(self, optimization_level):
        """Test that trying to provide a list of custom basis gates to generate_preset_pass_manager
        raises a ValueError."""

        with self.subTest(msg="no error"):
            # check that the warning isn't raised if the basis gates aren't custom
            basis_gates = ["x", "cx"]
            _ = generate_preset_pass_manager(
                optimization_level=optimization_level, basis_gates=basis_gates
            )

        with self.subTest(msg="error only basis gates"):
            # check that the warning is raised if they are custom
            basis_gates = ["my_gate"]
            with self.assertRaises(
                ValueError,
            ):
                _ = generate_preset_pass_manager(
                    optimization_level=optimization_level, basis_gates=basis_gates
                )

        with self.subTest(msg="no error custom basis gates in backend"):
            # check that the warning is not raised if a loose custom gate is found in the backend
            backend = GenericBackendV2(num_qubits=2)
            gate = Gate(name="my_gate", num_qubits=1, params=[])
            backend.target.add_instruction(gate)
            self.assertEqual(
                backend.operation_names,
                ["cx", "id", "rz", "sx", "x", "reset", "delay", "measure", "my_gate"],
            )
            basis_gates = ["my_gate"]
            _ = generate_preset_pass_manager(
                optimization_level=optimization_level, basis_gates=basis_gates, backend=backend
            )

    @data(0, 1, 2, 3)
    def test_custom_measurement_subclass(self, optimization_level):
        """Test that a custom measurement subclass is treated appropriately."""
        backend = GenericBackendV2(num_qubits=2)

        class CustomMeasurement(Measure):
            """A custom specialized measurement."""

            def __init__(self, label=None):
                super().__init__(label=label)
                self.name = "custom_measure"

        backend = GenericBackendV2(num_qubits=2, coupling_map=[[0, 1]], control_flow=True)
        backend.target.add_instruction(
            CustomMeasurement(),
            {
                (0,): InstructionProperties(error=1e-2, duration=1e-8),
                (1,): InstructionProperties(error=1e-2, duration=1e-8),
            },
        )
        pm = generate_preset_pass_manager(backend=backend, optimization_level=optimization_level)
        qc = QuantumCircuit(2, 3)
        qc.h(0)
        qc.cx(0, 1)
        qc.append(CustomMeasurement(), [0], [0])
        qc.append(CustomMeasurement(), [1], [1])
        qc.reset(0)
        with qc.if_test((qc.clbits[0], 1)):
            qc.x(0)
        with qc.if_test((qc.clbits[1], 1)):
            qc.z(0)
        qc.measure(0, 2)
        res = pm.run(qc)
        counts = res.count_ops()
        self.assertIn("custom_measure", counts)
        self.assertEqual(counts["custom_measure"], 2)
        encountered = 0
        for inst in res.data:
            if inst.name == "custom_measure":
                encountered += 1
                self.assertIsInstance(inst.operation, CustomMeasurement)
                self.assertIsInstance(inst.operation, Measure)

        self.assertEqual(
            encountered, 2, "Despite count ops no custom measurements were encountered"
        )
