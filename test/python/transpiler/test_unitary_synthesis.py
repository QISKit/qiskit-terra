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


"""
Tests for the default UnitarySynthesis transpiler pass.
"""

import unittest
import math
import numpy as np
import scipy
from ddt import ddt, data

from qiskit import transpile, generate_preset_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, ParameterVector
from qiskit.circuit.library import quantum_volume
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler.passes import UnitarySynthesis
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info import get_clifford_gate_names
from qiskit.transpiler import PassManager, CouplingMap, Target, InstructionProperties
from qiskit.exceptions import QiskitError
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    Optimize1qGates,
    SabreLayout,
    Unroll3qOrMore,
    CheckMap,
    BarrierBeforeFinalMeasurements,
    SabreSwap,
    TrivialLayout,
)
from qiskit.circuit.library import (
    IGate,
    CXGate,
    CZGate,
    RZGate,
    RXGate,
    SXGate,
    XGate,
    iSwapGate,
    ECRGate,
    UGate,
    ZGate,
    RYYGate,
    RZZGate,
    RXXGate,
    PauliEvolutionGate,
    CPhaseGate,
)
from qiskit.quantum_info import SparsePauliOp, average_gate_fidelity
from qiskit.circuit import Measure
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit import Parameter, Gate
from qiskit.synthesis.unitary.qsd import qs_decomposition

from test import combine
from test import QiskitTestCase
from test.python.providers.fake_mumbai_v2 import (
    FakeMumbaiFractionalCX,
)
from ..legacy_cmaps import YORKTOWN_CMAP


class FakeBackend2QV2(GenericBackendV2):
    """A 2-qubit fake backend"""

    def __init__(self):
        super().__init__(num_qubits=2, basis_gates=["rx", "u"], seed=42)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
        }
        self._target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (1, 0): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        self._target.add_instruction(ECRGate(), ecr_props)


class FakeBackend5QV2(GenericBackendV2):
    """A 5-qubit fake backend"""

    def __init__(self, bidirectional=True):
        super().__init__(num_qubits=5, basis_gates=["u"], seed=42)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
            (3, 4): InstructionProperties(duration=5.23e-7, error=0.00098115),
        }
        if bidirectional:
            cx_props[(1, 0)] = InstructionProperties(duration=6.23e-7, error=0.00099115)
            cx_props[(4, 3)] = InstructionProperties(duration=7.23e-7, error=0.00099115)
        self._target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (1, 2): InstructionProperties(duration=4.52e-9, error=0.0000132115),
            (2, 3): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        if bidirectional:
            ecr_props[(2, 1)] = InstructionProperties(duration=5.52e-9, error=0.0000232115)
            ecr_props[(3, 2)] = InstructionProperties(duration=5.52e-9, error=0.0000232115)


@ddt
class TestUnitarySynthesisBasisGates(QiskitTestCase):
    """Test UnitarySynthesis pass with basis gates."""

    def test_empty_basis_gates(self):
        """Verify when basis_gates is None, we do not synthesize unitaries."""
        qc = QuantumCircuit(3)
        op_1q = random_unitary(2, seed=0)
        op_2q = random_unitary(4, seed=0)
        op_3q = random_unitary(8, seed=0)
        qc.unitary(op_1q.data, [0])
        qc.unitary(op_2q.data, [0, 1])
        qc.unitary(op_3q.data, [0, 1, 2])
        out = UnitarySynthesis(basis_gates=None, min_qubits=2)(qc)
        self.assertEqual(out.count_ops(), {"unitary": 3})

    @data(
        ["u3", "cx"],
        ["u1", "u2", "u3", "cx"],
        ["ry", "rz", "rxx"],
        ["rx", "rz", "rzz"],
        ["rx", "rz", "iswap"],
        ["u3", "rx", "rz", "cz", "iswap"],
        ["rx", "rz", "cz", "rzz"],
    )
    def test_two_qubit_synthesis_to_basis(self, basis_gates):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell_op = Operator(bell)

        qc = QuantumCircuit(2)
        qc.unitary(bell_op, [0, 1])
        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(basis_gates).run(dag)
        self.assertTrue(set(out.count_ops()).issubset(basis_gates))

    @data(True, False, None)
    def test_two_qubit_synthesis_to_directional_cx_from_coupling_map(self, natural_direction):
        """Verify natural cx direction is used when specified in coupling map."""

        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=coupling_map,
            pulse_optimize=True,
            natural_direction=natural_direction,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)

        if natural_direction is False:
            self.assertTrue(
                all((qr[1], qr[0]) == instr.qubits for instr in qc_out.get_instructions("cx"))
            )
        else:
            # the decomposer defaults to the [1, 0] direction but the coupling
            # map specifies a [0, 1] direction. Check that this is respected.
            self.assertTrue(
                all((qr[0], qr[1]) == instr.qubits for instr in qc_out.get_instructions("cx"))
            )
        self.assertEqual(Operator(qc), Operator(qc_out))

    def test_two_qubit_synthesis_not_pulse_optimal(self):
        """Verify not attempting pulse optimal decomposition when pulse_optimize==False."""

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        coupling_map = CouplingMap([[0, 1]])
        pm_nonoptimal = PassManager(
            [
                TrivialLayout(coupling_map),
                UnitarySynthesis(
                    basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
                    coupling_map=coupling_map,
                    pulse_optimize=False,
                    natural_direction=True,
                ),
            ]
        )
        pm_optimal = PassManager(
            [
                TrivialLayout(coupling_map),
                UnitarySynthesis(
                    basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
                    coupling_map=coupling_map,
                    pulse_optimize=True,
                    natural_direction=True,
                ),
            ]
        )
        qc_nonoptimal = pm_nonoptimal.run(qc)
        qc_optimal = pm_optimal.run(qc)
        self.assertGreater(qc_nonoptimal.count_ops()["sx"], qc_optimal.count_ops()["sx"])

    def test_two_qubit_pulse_optimal_true_raises(self):
        """Verify raises if pulse optimal==True but cx is not in the basis."""
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        # this assumes iswap pulse optimal decomposition doesn't exist
        basis_gates = [gate if gate != "cx" else "iswap" for gate in basis_gates]
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            pulse_optimize=True,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        with self.assertRaises(QiskitError):
            pm.run(qc)

    def test_two_qubit_natural_direction_true_gate_length_raises(self):
        """Verify that error is raised if preferred direction cannot be inferred
        from gate lengths/errors.
        """
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            pulse_optimize=True,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        with self.assertRaises(QiskitError):
            pm.run(qc)

    def test_two_qubit_pulse_optimal_none_optimal(self):
        """Verify pulse optimal decomposition when pulse_optimize==None."""
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=coupling_map,
            pulse_optimize=None,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        if isinstance(qc_out, QuantumCircuit):
            num_ops = qc_out.count_ops()
        else:
            num_ops = qc_out[0].count_ops()
        self.assertIn("sx", num_ops)
        self.assertLessEqual(num_ops["sx"], 12)

    def test_two_qubit_pulse_optimal_none_no_raise(self):
        """Verify pulse optimal decomposition when pulse_optimize==None doesn't
        raise when pulse optimal decomposition unknown."""
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        # this assumes iswap pulse optimal decomposition doesn't exist
        basis_gates = [gate if gate != "cx" else "iswap" for gate in basis_gates]
        qr = QuantumRegister(2)
        coupling_map = CouplingMap([[0, 1], [1, 2], [1, 3], [3, 4]])
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            pulse_optimize=None,
            natural_direction=True,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        try:
            qc_out = pm.run(qc)
        except QiskitError:
            self.fail("pulse_optimize=None raised exception unexpectedly")
        if isinstance(qc_out, QuantumCircuit):
            num_ops = qc_out.count_ops()
        else:
            num_ops = qc_out[0].count_ops()
        self.assertIn("sx", num_ops)
        self.assertLessEqual(num_ops["sx"], 14)

    def test_qv_natural(self):
        """Check that quantum volume circuit compiles for natural direction"""
        qv64 = quantum_volume(5, seed=15)

        def construct_passmanager(basis_gates, coupling_map, synthesis_fidelity, pulse_optimize):
            seed = 2
            _map = [SabreLayout(coupling_map, max_iterations=2, seed=seed)]
            _unroll3q = Unroll3qOrMore()
            _swap_check = CheckMap(coupling_map)
            _swap = [
                BarrierBeforeFinalMeasurements(),
                SabreSwap(coupling_map, heuristic="lookahead", seed=seed),
            ]
            _optimize = [
                Collect2qBlocks(),
                ConsolidateBlocks(basis_gates=basis_gates),
                UnitarySynthesis(
                    basis_gates,
                    synthesis_fidelity,
                    coupling_map,
                    pulse_optimize=pulse_optimize,
                    natural_direction=True,
                ),
                Optimize1qGates(basis_gates),
            ]

            pm = PassManager()
            pm.append(_map)  # map to hardware by inserting swaps
            pm.append(_unroll3q)
            pm.append(_swap_check)
            pm.append(_swap)
            pm.append(_optimize)
            return pm

        coupling_map = CouplingMap([[0, 1], [1, 2], [3, 2], [3, 4], [5, 4]])
        basis_gates = ["rz", "sx", "cx"]

        pm1 = construct_passmanager(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            synthesis_fidelity=0.99,
            pulse_optimize=True,
        )
        pm2 = construct_passmanager(
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            synthesis_fidelity=0.99,
            pulse_optimize=False,
        )

        qv64_1 = pm1.run(qv64.decompose())
        qv64_2 = pm2.run(qv64.decompose())
        self.assertLessEqual(
            {
                tuple(qv64_1.qubits.index(qubit) for qubit in instr.qubits)
                for instr in qv64_1.get_instructions("cx")
            },
            {tuple(edge) for edge in coupling_map.get_edges()},
        )
        self.assertEqual(Operator(qv64_1), Operator(qv64_2))

    @data(1, 2, 3)
    def test_coupling_map_transpile(self, opt):
        """test natural_direction works with transpile"""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [0, 1])
        circ_01 = transpile(
            circ, basis_gates=["rz", "sx", "cx"], optimization_level=opt, coupling_map=[[0, 1]]
        )
        circ_10 = transpile(
            circ, basis_gates=["rz", "sx", "cx"], optimization_level=opt, coupling_map=[[1, 0]]
        )
        circ_01_index = {qubit: index for index, qubit in enumerate(circ_01.qubits)}
        circ_10_index = {qubit: index for index, qubit in enumerate(circ_10.qubits)}

        self.assertTrue(
            all(
                (1, 0) == (circ_10_index[instr.qubits[0]], circ_10_index[instr.qubits[1]])
                for instr in circ_10.get_instructions("cx")
            )
        )
        self.assertTrue(
            all(
                (0, 1) == (circ_01_index[instr.qubits[0]], circ_01_index[instr.qubits[1]])
                for instr in circ_01.get_instructions("cx")
            )
        )

    def test_if_simple(self):
        """Test a simple if statement."""
        basis_gates = {"u", "cx"}
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)

        qc_uni = QuantumCircuit(2)
        qc_uni.h(0)
        qc_uni.cx(0, 1)
        qc_uni_mat = Operator(qc_uni)

        qc_true_body = QuantumCircuit(2)
        qc_true_body.unitary(qc_uni_mat, [0, 1])

        qc = QuantumCircuit(qr, cr)
        qc.if_test((cr, 1), qc_true_body, [0, 1], [])
        dag = circuit_to_dag(qc)
        cdag = UnitarySynthesis(basis_gates=basis_gates).run(dag)
        cqc = dag_to_circuit(cdag)
        cbody = cqc.data[0].operation.params[0]
        self.assertEqual(cbody.count_ops().keys(), basis_gates)
        self.assertEqual(qc_uni_mat, Operator(cbody))

    def test_nested_control_flow(self):
        """Test unrolling nested control flow blocks."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(1)
        qc_uni1 = QuantumCircuit(2)
        qc_uni1.swap(0, 1)
        qc_uni1_mat = Operator(qc_uni1)

        qc = QuantumCircuit(qr, cr)
        with qc.for_loop(range(3)):
            with qc.while_loop((cr, 0)):
                qc.unitary(qc_uni1_mat, [0, 1])
        dag = circuit_to_dag(qc)
        cdag = UnitarySynthesis(basis_gates=["u", "cx"]).run(dag)
        cqc = dag_to_circuit(cdag)
        cbody = cqc.data[0].operation.params[2].data[0].operation.params[0]
        self.assertEqual(cbody.count_ops().keys(), {"u", "cx"})
        self.assertEqual(qc_uni1_mat, Operator(cbody))

    def test_default_does_not_fail_on_no_syntheses(self):
        qc = QuantumCircuit(1)
        qc.unitary(np.eye(2), [0])
        pass_ = UnitarySynthesis(["unknown", "gates"])
        self.assertEqual(qc, pass_(qc))

    @data(["unitary"], ["rz"])
    def test_synth_gates_to_basis(self, synth_gates):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        unitary = QuantumCircuit(1)
        unitary.h(0)
        unitary_op = Operator(unitary)

        qc = QuantumCircuit(1)
        qc.unitary(unitary_op, 0)
        qc.rz(0.1, 0)
        dag = circuit_to_dag(qc)

        basis_gates = ["u3"]
        out = UnitarySynthesis(basis_gates=basis_gates, synth_gates=synth_gates).run(dag)
        self.assertTrue(set(out.count_ops()).isdisjoint(synth_gates))

    def test_one_qubit_clifford_t_synthesis(self):
        """Check that when basis_set if of the form Clifford+T, 1-qubit unitaries are
        indeed synthesized to Clifford+T gates.
        """
        qc = QuantumCircuit(1)
        qc.append(random_unitary(2, seed=1), [0])
        dag = circuit_to_dag(qc)

        out = UnitarySynthesis(basis_gates=["cx", "h", "t", "tdg"]).run(dag)
        self.assertTrue(
            set(out.count_ops()).issubset(set(get_clifford_gate_names()).union({"t", "tdg"}))
        )

    @data(
        ["u3", "cx"],
        ["u1", "u2", "u3", "cx"],
        ["ry", "rz", "rxx"],
        ["rx", "rz", "rzz"],
        ["rx", "rz", "iswap"],
        ["u3", "rx", "rz", "cz", "iswap"],
        ["rx", "rz", "cz", "rzz"],
    )
    def test_no_panic_on_parametervector_phase(self, basis):
        pv = ParameterVector("v", 2)
        synth_pass = UnitarySynthesis(basis)
        for i in range(2, 6):
            with self.subTest(width=i):
                qc = QuantumCircuit(i, global_phase=pv[0])
                qc.rz(pv[1], range(i))
                for _ in range(50):
                    for j in range(i):
                        if i - 1 == j:
                            continue
                        qc.h(i - 1)
                        qc.x(j)
                        qc.rzx(math.pi / 4, i - 1, j)
                        qc.x(j)
                        qc.unitary(random_unitary(4, 4206_2026 + i + j), [i - 1, j])
                qc.unitary(random_unitary(2**i, 4206_2026), range(i))
                consolidated = ConsolidateBlocks(basis_gates=basis)(qc)
                result = synth_pass(consolidated)
                qc.assign_parameters([math.pi, -math.pi], inplace=True)
                result.assign_parameters([math.pi, -math.pi], inplace=True)
                self.assertEqual(Operator(result), Operator(qc))


@ddt
class TestUnitarySynthesisTarget(QiskitTestCase):
    """Test UnitarySynthesis pass with target/BackendV2."""

    @combine(
        opt_level=[0, 1, 2, 3],
        bidirectional=[True, False],
        dsc=(
            "test natural_direction works with transpile using opt_level {opt_level} on"
            " target with multiple 2q gates with bidirectional={bidirectional}"
        ),
        name="opt_level_{opt_level}_bidirectional_{bidirectional}",
    )
    def test_coupling_map_transpile_with_backendv2(self, opt_level, bidirectional):
        backend = FakeBackend5QV2(bidirectional)
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [0, 1])
        circ_01 = transpile(
            circ, backend=backend, optimization_level=opt_level, layout_method="trivial"
        )
        circ_01_index = {qubit: index for index, qubit in enumerate(circ_01.qubits)}
        self.assertGreaterEqual(len(circ_01.get_instructions("cx")), 1)
        for instr in circ_01.get_instructions("cx"):
            self.assertEqual(
                (0, 1), (circ_01_index[instr.qubits[0]], circ_01_index[instr.qubits[1]])
            )

    @combine(
        opt_level=[0, 1, 2, 3],
        bidirectional=[True, False],
        dsc=(
            "Test direction with transpile using opt_level {opt_level} on"
            " target with multiple 2q gates with bidirectional={bidirectional}"
            "direction [0, 1] is lower error and should be picked."
        ),
        name="opt_level_{opt_level}_bidirectional_{bidirectional}",
    )
    def test_coupling_unequal_duration_with_backendv2(self, opt_level, bidirectional):
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        backend = FakeBackend5QV2(bidirectional)
        tqc = transpile(
            circ,
            backend=backend,
            optimization_level=opt_level,
            translation_method="synthesis",
            layout_method="trivial",
        )
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("cx")), 1)
        for instr in tqc.get_instructions("cx"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    @combine(
        opt_level=[0, 1, 2, 3],
        dsc=(
            "Test direction with transpile using opt_level {opt_level} on"
            " target with multiple 2q gates"
        ),
        name="opt_level_{opt_level}",
    )
    def test_non_overlapping_kak_gates_with_backendv2(self, opt_level):
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        backend = FakeBackend2QV2()
        tqc = transpile(
            circ,
            backend=backend,
            optimization_level=opt_level,
            translation_method="synthesis",
            layout_method="trivial",
        )
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("ecr")), 1)
        for instr in tqc.get_instructions("ecr"):
            self.assertEqual((1, 0), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    def test_fractional_cx_with_backendv2(self):
        """Test fractional CX gets used if present in target."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [0, 1])
        backend = FakeMumbaiFractionalCX()
        synth_pass = UnitarySynthesis(target=backend.target)
        tqc = synth_pass(circ)
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("rzx")), 1)
        for instr in tqc.get_instructions("rzx"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    @combine(
        opt_level=[0, 1, 2, 3],
        dsc=(
            "Test direction with transpile using opt_level {opt_level} on"
            "target with multiple 2q gates available in reverse direction"
        ),
        name="opt_level_{opt_level}",
    )
    def test_reverse_direction(self, opt_level):
        target = Target(2)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=1.2e-6)})
        target.add_instruction(ECRGate(), {(0, 1): InstructionProperties(error=1.2e-7)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")), {(0,): None, (1,): None}
        )
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        tqc = transpile(
            circ,
            target=target,
            optimization_level=opt_level,
            translation_method="synthesis",
            layout_method="trivial",
        )
        tqc_index = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        self.assertGreaterEqual(len(tqc.get_instructions("ecr")), 1)
        for instr in tqc.get_instructions("ecr"):
            self.assertEqual((0, 1), (tqc_index[instr.qubits[0]], tqc_index[instr.qubits[1]]))

    @combine(
        opt_level=[0, 1, 2, 3],
        dsc=("Test controlled but not supercontrolled basis"),
        name="opt_level_{opt_level}",
    )
    def test_controlled_basis(self, opt_level):
        target = Target(2)
        target.add_instruction(RYYGate(np.pi / 8), {(0, 1): InstructionProperties(error=1.2e-6)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")), {(0,): None, (1,): None}
        )
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.append(random_unitary(4, seed=1), [1, 0])
        tqc = transpile(
            circ,
            target=target,
            optimization_level=opt_level,
            translation_method="synthesis",
            layout_method="trivial",
        )
        self.assertGreaterEqual(len(tqc.get_instructions("ryy")), 1)
        self.assertEqual(Operator(tqc), Operator(circ))

    def test_approximation_controlled(self):
        target = Target(2)
        target.add_instruction(RZZGate(np.pi / 10), {(0, 1): InstructionProperties(error=0.006)})
        target.add_instruction(RXXGate(np.pi / 3), {(0, 1): InstructionProperties(error=0.01)})
        target.add_instruction(
            UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")),
            {(0,): InstructionProperties(error=0.001), (1,): InstructionProperties(error=0.002)},
        )
        circ = QuantumCircuit(2)
        circ.append(random_unitary(4, seed=7), [1, 0])

        dag = circuit_to_dag(circ)
        dag_100 = UnitarySynthesis(target=target, approximation_degree=1.0).run(dag)
        dag_99 = UnitarySynthesis(target=target, approximation_degree=0.99).run(dag)
        self.assertGreaterEqual(dag_100.depth(), dag_99.depth())
        self.assertEqual(Operator(dag_to_circuit(dag_100)), Operator(circ))

    def test_mapping_control_flow(self):
        """Test that inner dags use proper qubit mapping."""
        qr = QuantumRegister(3, "q")
        qc = QuantumCircuit(qr)

        # Create target that supports CX only between 0 and 2.
        fake_target = Target()
        fake_target.add_instruction(CXGate(), {(0, 2): None})
        fake_target.add_instruction(
            UGate(Parameter("t"), Parameter("p"), Parameter("l")),
            {
                (0,): None,
                (1,): None,
                (2,): None,
            },
        )

        qc_uni1 = QuantumCircuit(2)
        qc_uni1.swap(0, 1)
        qc_uni1_mat = Operator(qc_uni1)

        loop_body = QuantumCircuit(2)
        loop_body.unitary(qc_uni1_mat, [0, 1])

        # Loop body uses qubits 0 and 2, mapped to 0 and 1 in the block.
        # If synthesis doesn't handle recursive mapping, it'll incorrectly
        # look for a CX on (0, 1) instead of on (0, 2).
        qc.for_loop((0,), None, loop_body, [0, 2], [])

        dag = circuit_to_dag(qc)
        UnitarySynthesis(basis_gates=["u", "cx"], target=fake_target).run(dag)

    def test_single_qubit_with_target(self):
        """Test input circuit with only 1q works with target."""
        qc = QuantumCircuit(1)
        qc.append(ZGate(), [qc.qubits[0]])
        dag = circuit_to_dag(qc)
        backend = GenericBackendV2(num_qubits=5, seed=42)
        unitary_synth_pass = UnitarySynthesis(target=backend.target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(qc, result_qc)

    def test_single_qubit_identity_with_target(self):
        """Test input single qubit identity works with target."""
        qc = QuantumCircuit(1)
        qc.unitary([[1.0, 0.0], [0.0, 1.0]], 0)
        dag = circuit_to_dag(qc)
        backend = GenericBackendV2(num_qubits=5)
        unitary_synth_pass = UnitarySynthesis(target=backend.target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, QuantumCircuit(1))

    def test_unitary_synthesis_with_ideal_and_variable_width_ops(self):
        """Test unitary synthesis works with a target that contains ideal and variadic ops."""
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4), [0, 1])
        dag = circuit_to_dag(qc)
        target = GenericBackendV2(num_qubits=5).target
        target.add_instruction(IfElseOp, name="if_else")
        target.add_instruction(ZGate())
        target.add_instruction(ECRGate())
        unitary_synth_pass = UnitarySynthesis(target=target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, QuantumCircuit(2))

    def test_unitary_synthesis_custom_gate_target(self):
        qc = QuantumCircuit(2)
        qc.unitary(np.eye(4), [0, 1])
        dag = circuit_to_dag(qc)

        class CustomGate(Gate):
            """Custom Opaque Gate"""

            def __init__(self):
                super().__init__("custom", 2, [])

        target = Target(num_qubits=2)
        target.add_instruction(
            UGate(Parameter("t"), Parameter("p"), Parameter("l")), {(0,): None, (1,): None}
        )
        target.add_instruction(CustomGate(), {(0, 1): None, (1, 0): None})
        unitary_synth_pass = UnitarySynthesis(target=target)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertEqual(result_qc, qc)

    def test_iswap_no_cx_synthesis_succeeds(self):
        """Test basis set with iswap but no cx can synthesize a circuit"""
        target = Target()
        theta = Parameter("theta")

        i_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(IGate(), i_props)
        rz_props = {
            (0,): InstructionProperties(duration=0, error=0),
            (1,): InstructionProperties(duration=0, error=0),
        }
        target.add_instruction(RZGate(theta), rz_props)
        sx_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(SXGate(), sx_props)
        x_props = {
            (0,): InstructionProperties(duration=35.5e-9, error=0.000413),
            (1,): InstructionProperties(duration=35.5e-9, error=0.000502),
        }
        target.add_instruction(XGate(), x_props)
        iswap_props = {
            (0, 1): InstructionProperties(duration=519.11e-9, error=0.01201),
            (1, 0): InstructionProperties(duration=554.66e-9, error=0.01201),
        }
        target.add_instruction(iSwapGate(), iswap_props)
        measure_props = {
            (0,): InstructionProperties(duration=5.813e-6, error=0.0751),
            (1,): InstructionProperties(duration=5.813e-6, error=0.0225),
        }
        target.add_instruction(Measure(), measure_props)

        qc = QuantumCircuit(2)
        cxmat = Operator(CXGate()).to_matrix()
        qc.unitary(cxmat, [0, 1])
        unitary_synth_pass = UnitarySynthesis(target=target)
        dag = circuit_to_dag(qc)
        result_dag = unitary_synth_pass.run(dag)
        result_qc = dag_to_circuit(result_dag)
        self.assertTrue(np.allclose(Operator(result_qc.to_gate()).to_matrix(), cxmat))

    @combine(
        is_random=[True, False], add_kak=[True, False], param_gate=[RXXGate, RZZGate, CPhaseGate]
    )
    def test_parameterized_basis_gate_in_target(self, is_random, add_kak, param_gate):
        """Test synthesis with parameterized RZZ/RXX gate."""
        theta = Parameter("θ")
        lam = Parameter("λ")
        phi = Parameter("ϕ")
        target = Target(num_qubits=2)
        target.add_instruction(RZGate(lam))
        target.add_instruction(RXGate(phi))
        target.add_instruction(param_gate(theta))
        if add_kak:
            target.add_instruction(CZGate())
        qc = QuantumCircuit(2)
        if is_random:
            qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc.cp(np.pi / 2, 0, 1)
        qc_transpiled = transpile(qc, target=target, optimization_level=3, seed_transpiler=42)
        opcount = qc_transpiled.count_ops()
        # should only use the parametrized gate and not the CZ gate
        # regression test for https://github.com/Qiskit/qiskit/issues/13428
        self.assertTrue(set(opcount).issubset({"rz", "rx", param_gate(theta).name}))
        self.assertTrue(np.allclose(Operator(qc_transpiled), Operator(qc)))

    def test_custom_parameterized_gate_in_target(self):
        """Test synthesis with custom parameterized gate in target."""

        class CustomXXGate(RXXGate):
            """Custom RXXGate subclass that's not a standard gate"""

            _standard_gate = None

            def __init__(self, theta, label=None):
                super().__init__(theta, label)
                self.name = "MyCustomXXGate"

        theta = Parameter("θ")
        lam = Parameter("λ")
        phi = Parameter("ϕ")

        target = Target(num_qubits=2)
        target.add_instruction(RZGate(lam))
        target.add_instruction(RXGate(phi))
        target.add_instruction(CustomXXGate(theta))

        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc_transpiled = UnitarySynthesis(target=target)(qc)
        opcount = qc_transpiled.count_ops()
        self.assertTrue(set(opcount).issubset({"rz", "rx", "MyCustomXXGate"}))

        self.assertTrue(np.allclose(Operator(qc_transpiled), Operator(qc)))

    def test_custom_parameterized_gate_in_target_skips(self):
        """Test that synthesis is skipped with custom parameterized
        gate in target that is not RXX equivalent."""

        class CustomXYGate(Gate):
            """Custom Gate subclass that's not a standard gate and not RXX equivalent"""

            _standard_gate = None

            def __init__(self, theta: ParameterValueType, label=None):
                """Create new custom rotstion XY gate."""
                super().__init__("MyCustomXYGate", 2, [theta])

            def __array__(self, dtype=None):
                """Return a Numpy.array for the custom gate."""
                theta = self.params[0]
                cos = math.cos(theta)
                isin = 1j * math.sin(theta)
                return np.array(
                    [[1, 0, 0, 0], [0, cos, -isin, 0], [0, -isin, cos, 0], [0, 0, 0, 1]],
                    dtype=dtype,
                )

            def inverse(self, annotated: bool = False):
                return CustomXYGate(-self.params[0])

        theta = Parameter("θ")
        lam = Parameter("λ")
        phi = Parameter("ϕ")

        target = Target(num_qubits=2)
        target.add_instruction(RZGate(lam))
        target.add_instruction(RXGate(phi))
        target.add_instruction(CustomXYGate(theta))

        qc = QuantumCircuit(2)
        qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc_transpiled = UnitarySynthesis(target=target)(qc)
        opcount = qc_transpiled.count_ops()
        self.assertTrue(set(opcount).issubset({"unitary"}))
        self.assertTrue(np.allclose(Operator(qc_transpiled), Operator(qc)))

    @data(["rx", "ry", "rxx"], ["rx", "rz", "rzz"], ["rx", "rz", "rzz", "cz"])
    def test_parameterized_backend(self, basis_gates):
        """Test synthesis with parameterized backend."""
        backend = GenericBackendV2(3, basis_gates=basis_gates, seed=0)
        qc = QuantumCircuit(3)
        qc.unitary(random_unitary(4, seed=1234), [0, 1])
        qc.unitary(random_unitary(4, seed=4321), [0, 2])
        qc.cp(np.pi / 2, 0, 1)
        qc_transpiled = transpile(qc, backend, optimization_level=3, seed_transpiler=42)
        opcount = qc_transpiled.count_ops()
        self.assertTrue(set(opcount).issubset(basis_gates))
        self.assertTrue(np.allclose(Operator.from_circuit(qc_transpiled), Operator(qc)))

    @data(1, 2, 3)
    def test_qsd(self, opt):
        """Test that the unitary synthesis pass runs qsd successfully with a target."""
        num_qubits = 3
        target = Target(num_qubits=num_qubits)
        target.add_instruction(UGate(Parameter("theta"), Parameter("phi"), Parameter("lam")))
        target.add_instruction(CXGate())
        mat = scipy.stats.ortho_group.rvs(2**num_qubits)
        qc = qs_decomposition(mat)
        qc_transpiled = transpile(qc, target=target, optimization_level=opt)
        self.assertTrue(np.allclose(mat, Operator(qc_transpiled).data))

    def test_3q_with_measure(self):
        """Test 3-qubit synthesis with measurements."""
        backend = FakeBackend5QV2()

        qc = QuantumCircuit(3, 1)
        qc.unitary(np.eye(2**3), range(3))
        qc.measure(0, 0)

        qc_transpiled = transpile(qc, backend)
        self.assertTrue(qc_transpiled.size, 1)
        self.assertTrue(qc_transpiled.count_ops().get("measure", 0), 1)

    def test_3q_series(self):
        """Test a series of 3-qubit blocks."""
        backend = GenericBackendV2(5, basis_gates=["u", "cx"], seed=1)

        x = QuantumCircuit(3)
        x.x(2)
        x_mat = Operator(x)

        qc = QuantumCircuit(3)
        qc.unitary(x_mat, range(3))
        qc.unitary(np.eye(2**3), range(3))

        tqc = transpile(qc, backend, optimization_level=0, initial_layout=[0, 1, 2])

        expected = np.kron(np.eye(2**2), x_mat)
        self.assertEqual(Operator(tqc), Operator(expected))

    def test_3q_measure_all(self):
        """Regression test of #13586."""
        hamiltonian = SparsePauliOp.from_list(
            [("IXX", 1), ("IYY", 1), ("IZZ", 1), ("XXI", 1), ("YYI", 1), ("ZZI", 1)]
        )

        qc = QuantumCircuit(3)
        qc.x([1, 2])
        op = PauliEvolutionGate(hamiltonian, time=1)
        qc.append(op.power(8), [0, 1, 2])
        qc.measure_all()

        backend = GenericBackendV2(5, basis_gates=["u", "cx"], seed=1)
        tqc = transpile(qc, backend)

        ops = tqc.count_ops()
        self.assertIn("u", ops)
        self.assertIn("cx", ops)
        self.assertIn("measure", ops)

    def test_target_with_global_gates(self):
        """Test that 2q decomposition can handle a target with global gates."""
        basis_gates = ["h", "p", "cp", "rz", "cx", "ccx", "swap"]
        target = Target.from_configuration(basis_gates=basis_gates)

        bell = QuantumCircuit(2)
        bell.h(0)
        bell.cx(0, 1)
        bell_op = Operator(bell)
        qc = QuantumCircuit(2)
        qc.unitary(bell_op, [0, 1])

        tqc = transpile(qc, target=target)
        self.assertTrue(set(tqc.count_ops()).issubset(basis_gates))

    def test_determinism(self):
        """Test that the decomposition is deterministic."""
        gate_counts = {"rx": 6, "rz": 12, "iswap": 2}
        basis_gates = ["rx", "rz", "iswap"]
        target = Target.from_configuration(basis_gates=basis_gates)
        pm = generate_preset_pass_manager(target=target, optimization_level=2, seed_transpiler=42)

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        for _ in range(10):
            out = pm.run(qc)
            self.assertTrue(Operator(out).equiv(qc))
            self.assertTrue(set(out.count_ops()).issubset(basis_gates))
            for basis_gate in basis_gates:
                self.assertLessEqual(out.count_ops()[basis_gate], gate_counts[basis_gate])

    def test_determinism_parallelism(self):
        """Test that the decomposition is deterministic in parallel mode with >50 gates to synthesize."""
        gate_counts = {"rx": 800, "rz": 1100, "iswap": 300}
        basis_gates = ["rx", "rz", "iswap"]
        target = Target.from_configuration(basis_gates=basis_gates)

        qc = QuantumCircuit(2)
        for i in range(100):
            qc.unitary(random_unitary(4, seed=2606_42), [0, 1])

        for _ in range(10):
            out = UnitarySynthesis(target=target)(qc)
            self.assertTrue(Operator(out).equiv(qc))
            op_counts = out.count_ops()
            self.assertTrue(set(op_counts).issubset(basis_gates))
            for basis_gate in basis_gates:
                self.assertLessEqual(op_counts[basis_gate], gate_counts[basis_gate])

    @combine(gate=["unitary", "swap"], natural_direction=[True, False])
    def test_two_qubit_synthesis_to_directional_cx_target(self, gate, natural_direction):
        """Verify two qubit unitaries are synthesized to match basis gates."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=YORKTOWN_CMAP,
            seed=1,
        )
        coupling_map = CouplingMap(backend.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        if gate == "unitary":
            qc.unitary(random_unitary(4, seed=12), [0, 1])
        elif gate == "swap":
            qc.swap(qr[0], qr[1])

        unisynth_pass = UnitarySynthesis(
            target=backend.target,
            pulse_optimize=True,
            natural_direction=natural_direction,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))

    @data(True, False)
    def test_two_qubit_synthesis_to_directional_cx_multiple_registers_target(
        self, natural_direction
    ):
        """Verify two qubit unitaries are synthesized to match basis gates
        across multiple registers."""
        # TODO: should make check more explicit e.g. explicitly set gate
        # direction in test instead of using specific fake backend
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            coupling_map=YORKTOWN_CMAP,
            seed=1,
        )
        qr0 = QuantumRegister(1)
        qr1 = QuantumRegister(1)
        coupling_map = CouplingMap(backend.coupling_map)
        triv_layout_pass = TrivialLayout(coupling_map)
        qc = QuantumCircuit(qr0, qr1)
        qc.unitary(random_unitary(4, seed=12), [qr0[0], qr1[0]])
        unisynth_pass = UnitarySynthesis(
            target=backend.target,
            pulse_optimize=True,
            natural_direction=natural_direction,
        )
        pm = PassManager([triv_layout_pass, unisynth_pass])
        qc_out = pm.run(qc)
        self.assertEqual(Operator(qc), Operator(qc_out))

    def test_two_qubit_natural_direction_true_duration_fallback_target(self):
        """Verify fallback path when pulse_optimize==True."""
        basis_gates = ["id", "rz", "sx", "x", "cx", "reset"]
        qr = QuantumRegister(5)
        coupling_map = CouplingMap([[0, 1], [1, 0], [1, 2], [1, 3], [3, 4]])
        backend = GenericBackendV2(
            num_qubits=5, basis_gates=basis_gates, coupling_map=coupling_map, seed=1
        )
        qc = QuantumCircuit(qr)
        qc.unitary(random_unitary(4, seed=12), [0, 1])
        unisynth_pass = UnitarySynthesis(
            target=backend.target,
            pulse_optimize=True,
            natural_direction=True,
        )
        qc_out = unisynth_pass(qc)
        self.assertEqual(
            {
                tuple(qc_out.find_bit(q).index for q in instr.qubits)
                for instr in qc_out.get_instructions("cx")
            },
            {(0, 1)},
        )

    def test_approximate_synthesis(self):
        # Arbitrary Hermitian matrix with a norm known to be sensibly sized (it's about 2.8).
        herm = np.array(
            [
                [-0.742, 0.643 + 0.341j, 0.166 - 0.432j, 0.803 + 0.285j],
                [0.643 - 0.341j, 1.462, -0.477 + 0.0674j, 0.216 + 0.653j],
                [0.166 + 0.432j, -0.477 - 0.0674j, 0.007, -0.231 - 0.226j],
                [0.803 - 0.285j, 0.216 - 0.653j, -0.231 + 0.226j, -0.743],
            ]
        )
        # A unitary perturbation that is a small distance from the identity.  It needs 3 cx to
        # synthesise.
        perturbation = scipy.linalg.expm(-1j * herm * 1e-3)

        target = Target(2)
        target.add_instruction(CXGate(), {(0, 1): InstructionProperties(error=1e-4)})
        target.add_instruction(RZGate(Parameter("a")))
        target.add_instruction(SXGate())

        pass_exact = UnitarySynthesis(target=target, approximation_degree=1.0)
        pass_approximate = UnitarySynthesis(target=target, approximation_degree=None)

        # iSwap can be synthesised with 2 CX.
        near_2cx = iSwapGate().to_matrix() @ perturbation
        qc = QuantumCircuit(2)
        qc.ensure_physical()
        qc.unitary(near_2cx, [0, 1])

        # First, a sanity check: the pass defaults should produce an exact synthesis, and it should
        # have taken 3 cx since we know the perturbation needs that.
        from_default = UnitarySynthesis(target=target)(qc)
        self.assertLess(
            1 - average_gate_fidelity(Operator(near_2cx), Operator(from_default)), 1e-15
        )
        self.assertEqual(from_default.count_ops()["cx"], 3)
        # These two circuits should be exactly identical, since it's the same decomposition.
        self.assertEqual(from_default, pass_exact(qc))
        # ... but now if we allow approximation up to the gate error, we should be able to find the
        # 2-cx synthesis of iSwap (or something else that's nearby).
        self.assertEqual(pass_approximate(qc).count_ops()["cx"], 2)

        # The same applies for gates that are near a 1-cx decomposition...
        near_1cx = CXGate().to_matrix() @ perturbation
        qc = QuantumCircuit(2)
        qc.ensure_physical()
        qc.unitary(near_1cx, [0, 1])
        from_exact = pass_exact(qc)
        self.assertLess(1 - average_gate_fidelity(Operator(near_1cx), Operator(from_exact)), 1e-15)
        self.assertEqual(from_exact.count_ops()["cx"], 3)
        self.assertEqual(pass_approximate(qc).count_ops()["cx"], 1)

        # ... and near a 0q decomposition.
        near_separable = np.kron(XGate().to_matrix(), ZGate().to_matrix()) @ perturbation
        qc = QuantumCircuit(2)
        qc.ensure_physical()
        qc.unitary(near_separable, [0, 1])
        from_exact = pass_exact(qc)
        self.assertLess(
            1 - average_gate_fidelity(Operator(near_separable), Operator(from_exact)), 1e-15
        )
        self.assertEqual(from_exact.count_ops()["cx"], 3)
        self.assertNotIn("cx", pass_approximate(qc).count_ops())

    def test_problematic_unitary(self):
        """Test on a circuit with a special unitary matrix that caused problems for
        Quantum Shannon Decomposition based on nalgebra.

        Based on regression test of #15870.
        """
        unitary_matrix = np.array(
            [
                [0j, 0j, 0j, 0j, 0j, 0j, 0j, (1 + 0j)],
                [0j, 0j, 0j, 0j, 0j, 0j, (1 + 0j), 0j],
                [0j, 0j, 0j, 0j, 0j, (1 + 0j), 0j, 0j],
                [0j, (1 + 0j), 0j, 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, 0j, (1 + 0j), 0j, 0j, 0j],
                [0j, 0j, (1 + 0j), 0j, 0j, 0j, 0j, 0j],
                [0j, 0j, 0j, (1 + 0j), 0j, 0j, 0j, 0j],
                [(1 + 0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j],
            ]
        )

        circuit = QuantumCircuit(3)
        circuit.unitary(unitary_matrix, list(range(3)))

        _ = UnitarySynthesis(basis_gates=["cx", "u"])(circuit)


if __name__ == "__main__":
    unittest.main()
