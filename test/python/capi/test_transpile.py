# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import math

from test import QiskitTestCase, combine

import ddt
import rustworkx as rx

from qiskit import capi
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import CXGate
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Operator

from . import ffi
from ..legacy_cmaps import MELBOURNE_CMAP


@ddt.ddt
class TestTranspile(QiskitTestCase):
    @ddt.data(0, 1, 2, 3)
    def test_empty_transpilation(self, opt_level):
        """Test transpilation of empty circuit."""
        target = ffi.build_homogenous_target(CouplingMap.from_ring(10), ["cx", "u"], 42)
        self.addCleanup(capi.qk_target_free, target)
        circuit = QuantumCircuit(5, 5)
        res = ffi.transpile_from_c(circuit, target, opt_level, 1.0, 42)
        # Remove layout since it's not valid for this comparison it just says a layout of empty
        # qubits was selected with no permutation
        res._layout = None
        expected = QuantumCircuit(10)
        for creg in circuit.cregs:
            expected.add_register(creg)
        self.assertEqual(expected, res)

    @ddt.data(0, 1, 2, 3)
    def test_transpile_qft_grid(self, opt_level):
        """Transpile pipeline can handle 8-qubit QFT on 14-qubit grid."""

        basis_gates = ["cx", "id", "rz", "sx", "x"]

        qr = QuantumRegister(8)
        circuit = QuantumCircuit(qr)
        for i, q in enumerate(qr):
            for j in range(i):
                circuit.cp(math.pi / float(2 ** (i - j)), q, qr[j])
            circuit.h(q)
        target = ffi.build_homogenous_target(CouplingMap(MELBOURNE_CMAP), basis_gates, seed=42)
        self.addCleanup(capi.qk_target_free, target)
        new_circuit = ffi.transpile_from_c(circuit, target, opt_level, 1.0, 42)

        qubit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qubits)}
        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([qubit_indices[x] for x in instruction.qubits], MELBOURNE_CMAP)

    @ddt.data(0, 1, 2, 3)
    def test_translate_ecr_basis(self, optimization_level):
        """Verify that rewriting in ECR basis is efficient."""
        circuit = QuantumCircuit(2)
        circuit.rzx(0.121234, 0, 1)
        circuit.cx(0, 1)
        circuit.swap(0, 1)
        circuit.iswap(0, 1)
        target = ffi.build_homogenous_target(CouplingMap.from_full(2), ["u", "ecr"], 42)
        self.addCleanup(capi.qk_target_free, target)
        res = ffi.transpile_from_c(circuit, target, optimization_level, 1.0, 42)
        # Swap gates get optimized away in opt. level 2, 3
        expected_num_ecr_gates = 2 if optimization_level in (2, 3) else 8
        self.assertEqual(res.count_ops()["ecr"], expected_num_ecr_gates)
        self.assertEqual(Operator(circuit), Operator.from_circuit(res))

    def test_optimize_ecr_basis(self):
        """Test highest optimization level can optimize over ECR."""
        circuit = QuantumCircuit(2)
        circuit.swap(1, 0)
        circuit.iswap(0, 1)
        target = ffi.build_homogenous_target(CouplingMap.from_full(2), ["u", "ecr"], 42)
        self.addCleanup(capi.qk_target_free, target)
        res = ffi.transpile_from_c(circuit, target, 3, 1.0, 42)
        # an iswap gate is equivalent to (swap, CZ) up to single-qubit rotations. Normally, the swap gate
        # in the circuit would cancel with the swap gate of the (swap, CZ), leaving a single CZ gate that
        # can be realized via one ECR gate. However, with the introduction of ElideSwap, the swap gate
        # cancellation can not occur anymore, thus requiring two ECR gates for the iswap gate.
        self.assertEqual(res.count_ops()["ecr"], 2)
        self.assertEqual(Operator(circuit), Operator.from_circuit(res))

    @ddt.data(0, 1, 2, 3)
    def test_target_ideal_gates(self, opt_level):
        """Test that transpile() with a custom ideal sim target works."""
        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        target = ffi.build_homogenous_target(
            CouplingMap.from_full(2), ["u", "cx"], 42, ideal_gates=True
        )
        result = ffi.transpile_from_c(qc, target, opt_level, 1.0, 42)

        self.assertEqual(Operator.from_circuit(result), Operator.from_circuit(qc))

    @combine(opt_level=[0, 1, 2, 3], basis=[["rz", "x"], ["rx", "z"], ["rz", "y"], ["ry", "x"]])
    def test_paulis_to_constrained_1q_basis(self, opt_level, basis):
        """Test that Pauli-gate circuits can be transpiled to constrained 1q bases that do not
        contain any root-Pauli gates."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier()
        qc.y(0)
        qc.barrier()
        qc.z(0)
        target = ffi.build_homogenous_target(CouplingMap.from_line(1), basis, 42, True)
        self.addCleanup(capi.qk_target_free, target)
        transpiled = ffi.transpile_from_c(qc, target, opt_level, 1.0, 42)
        self.assertGreaterEqual(set(basis) | {"barrier"}, transpiled.count_ops().keys())
        self.assertEqual(Operator(qc), Operator(transpiled))

    @ddt.data(0, 1, 2, 3)
    def test_single_qubit_circuit_deterministic_output(self, optimization_level):
        """Test that the transpiler's output is deterministic in a single qubit example.

        Reproduce from `#14729 <https://github.com/Qiskit/qiskit/issues/14729>`__"""
        params = [math.pi, math.pi / 2, math.pi * 2, math.pi / 4, 0]

        circ = QuantumCircuit(len(params))
        for i, par in enumerate(params):
            circ.rx(par, i)
        circ.measure_all()
        target = ffi.build_homogenous_target(
            CouplingMap.from_full(10), ["cx", "rz", "sx", "x"], 123
        )
        self.addCleanup(capi.qk_target_free, target)
        isa_circs = []
        for _ in range(10):
            isa_circs.append(ffi.transpile_from_c(circ, target, optimization_level, 1.0, 123))
        for i in range(10):
            self.assertEqual(isa_circs[0], isa_circs[i])

    @ddt.data(2, 3)
    def test_size_optimization(self, level):
        """Test the levels for optimization based on size of circuit"""
        target = ffi.build_homogenous_target(CouplingMap.from_full(8), ["u3", "cx"], 42, True)
        self.addCleanup(capi.qk_target_free, target)
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
        circ = ffi.transpile_from_c(qc, target, level, 1.0, 123)

        circ_data = circ.data
        free_qubits = {0, 1, 2, 3}

        # ensure no gates are using qubits - [0,1,2,3]
        for gate in circ_data:
            layout = circ.layout.initial_layout
            indices = {layout[circ.find_bit(qubit).index] for qubit in gate.qubits}
            common = indices.intersection(free_qubits)
            for common_qubit in common:
                self.assertTrue(common_qubit not in free_qubits)

        self.assertLess(circ.size(), qc.size())
        self.assertLessEqual(circ.depth(), qc.depth())

    @ddt.data(0, 1, 2, 3)
    def test_single_cx_gate_circuit_on_linear_backend(self, level):
        """Simple coupling map (linear 5 qubits)."""
        basis = ["u1", "u2", "cx", "swap"]
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        circuit = QuantumCircuit(5)
        circuit.cx(2, 4)

        target = ffi.build_homogenous_target(coupling_map, basis, seed=24)
        self.addCleanup(capi.qk_target_free, target)
        result = ffi.transpile_from_c(circuit, target, level, 1.0, 123)

        self.assertIsInstance(result, QuantumCircuit)
        self.assertEqual(Operator.from_circuit(result), Operator(circuit))

    @ddt.data(0, 1, 2, 3)
    def test_multiple_cx_gate_circuit_on_linear_backend(self, level):
        """Simple coupling map (linear 5 qubits)."""
        basis = ["u1", "u2", "cx", "swap"]
        circuit = QuantumCircuit(5)
        circuit.cx(0, 4)
        circuit.cx(1, 4)
        circuit.cx(2, 4)
        circuit.cx(3, 4)
        coupling_map = CouplingMap([(0, 1), (1, 2), (2, 3), (3, 4)])
        target = ffi.build_homogenous_target(coupling_map, basis, seed=24)
        self.addCleanup(capi.qk_target_free, target)
        result = ffi.transpile_from_c(circuit, target, level, 1.0, 123)
        self.assertIsInstance(result, QuantumCircuit)
        self.assertEqual(Operator.from_circuit(result), Operator(circuit))


@ddt.ddt
class TestTranspileMultiChipTarget(QiskitTestCase):
    """Test qk_transpile() with a disjoint coupling map."""

    def setUp(self):
        super().setUp()

        graph = rx.generators.directed_heavy_hex_graph(3)
        edges = []
        for i in range(3):
            for root_edge in graph.edge_list():
                offset = i * len(graph)
                edge = (root_edge[0] + offset, root_edge[1] + offset)
                edges.append(edge)
        self.edge_set = set(edges)
        cmap = CouplingMap(edges)
        self.target = ffi.build_homogenous_target(cmap, ["rz", "x", "sx", "cz"], seed=12345678942)
        self.addCleanup(capi.qk_target_free, self.target)

    @ddt.data(0, 1, 2, 3)
    def test_basic_connected_circuit(self, opt_level):
        """Test basic connected circuit on disjoint backend"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        tqc = ffi.transpile_from_c(qc, self.target, opt_level, 1.0, 123)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            if len(qubits) == 2:
                self.assertIn(qubits, self.edge_set)

    @ddt.data(0, 1, 2, 3)
    def test_triple_circuit(self, opt_level):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()

        if opt_level == 0:
            self.skipTest("Invalid layout for this backend causes a panic in sabre")
        tqc = ffi.transpile_from_c(qc, self.target, opt_level, 1.0, 123)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            if len(qubits) == 2:
                self.assertIn(qubits, self.edge_set)
