# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test the StarPreRouting pass"""

import unittest

from test import QiskitTestCase
import ddt

from qiskit.synthesis.qft import synth_qft_full
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import VF2Layout, ApplyLayout, SabreSwap, SabreLayout
from qiskit.transpiler.passes.routing.star_prerouting import StarPreRouting
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.utils.optionals import HAS_AER


@ddt.ddt
class TestStarPreRouting(QiskitTestCase):
    """Tests the StarPreRouting pass"""

    def test_simple_ghz_dagcircuit(self):
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, range(1, 5))
        dag = circuit_to_dag(qc)
        new_dag = StarPreRouting().run(dag)
        new_qc = dag_to_circuit(new_dag)

        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(0, 1)
        expected.cx(0, 2)
        expected.swap(0, 2)
        expected.cx(2, 3)
        expected.swap(2, 3)
        expected.cx(3, 4)
        # expected.swap(3,4)

        self.assertTrue(Operator(expected).equiv(Operator(new_qc)))

    def test_simple_ghz_dagdependency(self):
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, range(1, 5))

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()

        result = pm.run(qc)

        self.assertTrue(Operator.from_circuit(result).equiv(Operator(qc)))

    def test_double_ghz_dagcircuit(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.cx(0, range(1, 5))
        qc.h(9)
        qc.cx(9, range(8, 4, -1))

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()
        new_qc = pm.run(qc)

        self.assertTrue(Operator.from_circuit(new_qc).equiv(Operator(qc)))

    def test_double_ghz_dagdependency(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.cx(0, range(1, 5))
        qc.h(9)
        qc.cx(9, range(8, 4, -1))
        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()
        new_qc = pm.run(qc)

        self.assertTrue(Operator(qc).equiv(Operator.from_circuit(new_qc)))

    def test_mixed_double_ghz_dagdependency(self):
        """Shows off the power of using commutation analysis."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(0, 2)

        qc.cx(3, 1)
        qc.cx(3, 2)

        qc.cx(0, 1)
        qc.cx(0, 2)

        qc.cx(3, 1)
        qc.cx(3, 2)

        qc.cx(0, 1)
        qc.cx(0, 2)

        qc.cx(3, 1)
        qc.cx(3, 2)

        qc.cx(0, 1)
        qc.cx(0, 2)

        qc.cx(3, 1)
        qc.cx(3, 2)
        # qc.measure_all()

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()

        result = pm.run(qc)

        self.assertTrue(Operator.from_circuit(result).equiv(Operator(qc)))

    def test_double_ghz(self):
        qc = QuantumCircuit(10)
        qc.h(0)
        qc.cx(0, range(1, 5))
        qc.h(9)
        qc.cx(9, range(8, 4, -1))

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()
        result = pm.run(qc)

        self.assertEqual(Operator.from_circuit(result), Operator(qc))

    def test_linear_ghz_no_change(self):
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 4)
        qc.cx(4, 5)

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()

        result = pm.run(qc)

        self.assertEqual(Operator.from_circuit(result), Operator(qc))

    def test_no_star(self):
        qc = QuantumCircuit(6)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(3, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(1, 4)
        qc.cx(2, 1)

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()
        result = pm.run(qc)

        self.assertTrue(Operator.from_circuit(result).equiv(qc))

    def test_10q_bv(self):
        num_qubits = 10
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.barrier()
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        result = StarPreRouting()(qc)

        expected = QuantumCircuit(num_qubits, num_qubits - 1)
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        expected.h(6)
        expected.h(7)
        expected.h(8)
        expected.x(9)
        expected.h(9)
        expected.cx(0, 9)
        expected.cx(1, 9)
        expected.swap(1, 9)
        expected.cx(2, 1)
        expected.swap(2, 1)
        expected.cx(3, 2)
        expected.swap(3, 2)
        expected.cx(4, 3)
        expected.swap(4, 3)
        expected.cx(5, 4)
        expected.swap(5, 4)
        expected.cx(6, 5)
        expected.swap(6, 5)
        expected.cx(7, 6)
        expected.swap(7, 6)
        expected.cx(8, 7)
        expected.barrier()
        expected.h(0)
        expected.h(1)
        expected.h(2)
        expected.h(3)
        expected.h(4)
        expected.h(5)
        expected.h(6)
        expected.h(8)
        expected.h(9)
        expected.measure(0, 0)
        expected.measure(9, 1)
        expected.measure(1, 2)
        expected.measure(2, 3)
        expected.measure(3, 4)
        expected.measure(4, 5)
        expected.measure(5, 6)
        expected.measure(6, 7)
        expected.measure(8, 8)
        self.assertEqual(result, expected)

    # Skip level 3 because of unitary synth introducing non-clifford gates
    @unittest.skipUnless(HAS_AER, "Aer required for clifford simulation")
    @ddt.data(0, 1)
    def test_100q_grid_full_path(self, opt_level):
        from qiskit_aer import AerSimulator

        num_qubits = 100
        coupling_map = CouplingMap.from_grid(10, 10)
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.barrier()
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        pm = generate_preset_pass_manager(
            opt_level, basis_gates=["h", "cx", "x"], coupling_map=coupling_map
        )
        pm.init += StarPreRouting()
        result = pm.run(qc)
        counts_before = AerSimulator().run(qc).result().get_counts()
        counts_after = AerSimulator().run(result).result().get_counts()
        self.assertEqual(counts_before, counts_after)

    def test_10q_bv_no_barrier(self):
        num_qubits = 6
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.h(qc.qubits[:-1])

        pm = generate_preset_pass_manager(optimization_level=3, seed_transpiler=42)
        pm.init += StarPreRouting()

        result = pm.run(qc)
        self.assertTrue(Operator.from_circuit(result).equiv(Operator(qc)))

    # Skip level 3 because of unitary synth introducing non-clifford gates
    @unittest.skipUnless(HAS_AER, "Aer required for clifford simulation")
    @ddt.data(0, 1)
    def test_100q_grid_full_path_no_barrier(self, opt_level):
        from qiskit_aer import AerSimulator

        num_qubits = 100
        coupling_map = CouplingMap.from_grid(10, 10)
        qc = QuantumCircuit(num_qubits, num_qubits - 1)
        qc.x(num_qubits - 1)
        qc.h(qc.qubits)
        for i in range(num_qubits - 1):
            qc.cx(i, num_qubits - 1)
        qc.h(qc.qubits[:-1])
        for i in range(num_qubits - 1):
            qc.measure(i, i)
        pm = generate_preset_pass_manager(
            opt_level, basis_gates=["h", "cx", "x"], coupling_map=coupling_map
        )
        pm.init += StarPreRouting()
        result = pm.run(qc)
        counts_before = AerSimulator().run(qc).result().get_counts()
        counts_after = AerSimulator().run(result).result().get_counts()
        self.assertEqual(counts_before, counts_after)

    def test_hadamard_ordering(self):
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.h(0)
        qc.cx(0, 2)
        qc.h(0)
        qc.cx(0, 3)
        qc.h(0)
        qc.cx(0, 4)
        result = StarPreRouting()(qc)
        expected = QuantumCircuit(5)
        expected.h(0)
        expected.cx(0, 1)
        expected.h(0)
        expected.cx(0, 2)
        expected.swap(0, 2)
        expected.h(2)
        expected.cx(2, 3)
        expected.swap(2, 3)
        expected.h(3)
        expected.cx(3, 4)
        # expected.swap(3, 4)
        self.assertEqual(expected, result)

    def test_count_1_stars_starting_center(self):
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        spr = StarPreRouting()

        star_blocks, _ = spr.determine_star_blocks_processing(circuit_to_dag(qc), min_block_size=2)
        self.assertEqual(len(star_blocks), 1)
        self.assertEqual(len(star_blocks[0].nodes), 5)

    def test_count_1_stars_starting_branch(self):
        qc = QuantumCircuit(6)
        qc.cx(1, 0)
        qc.cx(2, 0)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        spr = StarPreRouting()
        _ = spr(qc)

        star_blocks, _ = spr.determine_star_blocks_processing(circuit_to_dag(qc), min_block_size=2)
        self.assertEqual(len(star_blocks), 1)
        self.assertEqual(len(star_blocks[0].nodes), 5)

    def test_count_2_stars(self):
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        qc.cx(1, 2)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(1, 5)
        spr = StarPreRouting()
        _ = spr(qc)

        star_blocks, _ = spr.determine_star_blocks_processing(circuit_to_dag(qc), min_block_size=2)
        self.assertEqual(len(star_blocks), 2)
        self.assertEqual(len(star_blocks[0].nodes), 5)
        self.assertEqual(len(star_blocks[1].nodes), 4)

    def test_count_3_stars(self):
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)

        qc.cx(1, 2)
        qc.cx(1, 3)
        qc.cx(1, 4)
        qc.cx(1, 5)

        qc.cx(2, 3)
        qc.cx(2, 4)
        qc.cx(2, 5)
        spr = StarPreRouting()
        star_blocks, _ = spr.determine_star_blocks_processing(circuit_to_dag(qc), min_block_size=2)

        self.assertEqual(len(star_blocks), 3)
        self.assertEqual(len(star_blocks[0].nodes), 5)
        self.assertEqual(len(star_blocks[1].nodes), 4)
        self.assertEqual(len(star_blocks[2].nodes), 3)

    def test_count_70_qft_stars(self):
        qft_module = synth_qft_full(10, do_swaps=False)
        qftqc = QuantumCircuit(100)
        for i in range(10):
            qftqc.compose(qft_module, qubits=range(i * 10, (i + 1) * 10), inplace=True)
        spr = StarPreRouting()
        star_blocks, _ = spr.determine_star_blocks_processing(
            circuit_to_dag(qftqc), min_block_size=2
        )

        self.assertEqual(len(star_blocks), 80)
        star_len_list = [len([n for n in b.nodes if len(n.qargs) > 1]) for b in star_blocks]
        expected_star_size = {2, 3, 4, 5, 6, 7, 8, 9}
        self.assertEqual(set(star_len_list), expected_star_size)
        for i in expected_star_size:
            self.assertEqual(star_len_list.count(i), 10)

    def test_count_50_qft_stars(self):
        qft_module = synth_qft_full(10, do_swaps=False)
        qftqc = QuantumCircuit(10)
        for _ in range(10):
            qftqc.compose(qft_module, qubits=range(10), inplace=True)
        spr = StarPreRouting()
        _ = spr(qftqc)

        star_blocks, _ = spr.determine_star_blocks_processing(
            circuit_to_dag(qftqc), min_block_size=2
        )
        self.assertEqual(len(star_blocks), 50)
        star_len_list = [len([n for n in b.nodes if len(n.qargs) > 1]) for b in star_blocks]
        expected_star_size = {9}
        self.assertEqual(set(star_len_list), expected_star_size)

    def test_two_star_routing(self):
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(0, 2)

        qc.cx(2, 3)
        qc.cx(2, 1)

        spr = StarPreRouting()
        res = spr(qc)

        self.assertTrue(Operator.from_circuit(res).equiv(qc))

    def test_detect_two_opposite_stars_barrier(self):
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.barrier()
        qc.cx(5, 1)
        qc.cx(5, 2)
        qc.cx(5, 3)
        qc.cx(5, 4)

        spr = StarPreRouting()
        star_blocks, _ = spr.determine_star_blocks_processing(circuit_to_dag(qc), min_block_size=2)
        self.assertEqual(len(star_blocks), 2)
        self.assertEqual(len(star_blocks[0].nodes), 4)
        self.assertEqual(len(star_blocks[1].nodes), 4)

    def test_routing_after_star_prerouting(self):
        nq = 6
        qc = synth_qft_full(nq, do_swaps=False, insert_barriers=True)
        cm = CouplingMap.from_line(nq)

        pm_preroute = PassManager()
        pm_preroute.append(StarPreRouting())
        pm_preroute.append(VF2Layout(coupling_map=cm, seed=17))
        pm_preroute.append(ApplyLayout())
        pm_preroute.append(SabreSwap(coupling_map=cm, seed=17))

        pm_sabre = PassManager()
        pm_sabre.append(SabreLayout(coupling_map=cm, seed=17))

        res_sabre = pm_sabre.run(qc)
        res_star = pm_sabre.run(qc)

        self.assertTrue(Operator.from_circuit(res_sabre), qc)
        self.assertTrue(Operator.from_circuit(res_star), qc)
        self.assertTrue(Operator.from_circuit(res_star), Operator.from_circuit(res_sabre))

    @ddt.data(4, 8, 16, 32)
    def test_qft_linearization(self, num_qubits):
        """Test the QFT circuit to verify if it is linearized and requires n-2 swaps."""

        qc = synth_qft_full(num_qubits, do_swaps=False, insert_barriers=True)
        dag = circuit_to_dag(qc)
        new_dag = StarPreRouting().run(dag)
        new_qc = dag_to_circuit(new_dag)

        # Check that resulting result has n-2 swaps, where n is the number of cp gates
        swap_count = new_qc.count_ops().get("swap", 0)
        cp_count = new_qc.count_ops().get("cp", 0)
        self.assertEqual(swap_count, cp_count - 2)

        # Confirm linearization by checking that the number of edges is equal to the number of nodes
        def get_edge(node, dag):
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            if len(qubits) != 2:
                return None
            return (qubits[0], qubits[1]) if qubits[0] < qubits[1] else (qubits[1], qubits[0])

        edges = {
            edge for node in new_dag.op_nodes() if (edge := get_edge(node, new_dag)) is not None
        }
        self.assertEqual(len(edges), new_dag.num_qubits() - 1)
