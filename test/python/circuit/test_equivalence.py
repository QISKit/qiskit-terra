# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's EquivalenceLibrary class."""

import unittest

import numpy as np
import rustworkx as rx

from qiskit.circuit import QuantumCircuit, Parameter, Gate
from qiskit.circuit.library import U2Gate
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Qubit
from qiskit.converters import circuit_to_instruction, circuit_to_gate
from qiskit.circuit.equivalence import EquivalenceLibrary, Key, Equivalence, NodeData, EdgeData
from qiskit.utils import optionals
from test import QiskitTestCase

from ..visualization.visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class OneQubitZeroParamGate(Gate):
    """Mock one qubit zero param gate."""

    def __init__(self):
        super().__init__("1q0p", 1, [])


class OneQubitOneParamGate(Gate):
    """Mock one qubit one  param gate."""

    def __init__(self, theta):
        super().__init__("1q1p", 1, [theta])


class OneQubitTwoParamGate(Gate):
    """Mock one qubit two param gate."""

    def __init__(self, phi, lam):
        super().__init__("1q2p", 1, [phi, lam])


class TestEquivalenceLibraryWithoutBase(QiskitTestCase):
    """Test cases for basic EquivalenceLibrary."""

    def test_create_empty_library(self):
        """An empty library should return an empty entry."""
        eq_lib = EquivalenceLibrary()
        self.assertIsInstance(eq_lib, EquivalenceLibrary)

        gate = OneQubitZeroParamGate()

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 0)

    def test_add_single_entry(self):
        """Verify an equivalence added to the library can be retrieved."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit([Qubit()])
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertIsNot(entry[0], equiv)
        self.assertEqual(entry[0], equiv)

    def test_add_double_entry(self):
        """Verify separately added equivalences can be retrieved."""
        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit([Qubit()])
        first_equiv.h(0)

        eq_lib.add_equivalence(gate, first_equiv)

        second_equiv = QuantumCircuit([Qubit()])
        second_equiv.append(U2Gate(0, np.pi), [0])

        eq_lib.add_equivalence(gate, second_equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 2)
        self.assertEqual(entry[0], first_equiv)
        self.assertEqual(entry[1], second_equiv)

    def test_set_entry(self):
        """Verify setting an entry overrides any previously added, without affecting entries that
        depended on the set entry."""
        eq_lib = EquivalenceLibrary()

        gates = {key: Gate(key, 1, []) for key in "abcd"}
        target = Gate("target", 1, [])

        old = QuantumCircuit([Qubit()])
        old.append(gates["a"], [0])
        old.append(gates["b"], [0])
        eq_lib.add_equivalence(target, old)

        outbound = QuantumCircuit([Qubit()])
        outbound.append(target, [0])
        eq_lib.add_equivalence(gates["c"], outbound)

        self.assertEqual(eq_lib.get_entry(target), [old])
        self.assertEqual(eq_lib.get_entry(gates["c"]), [outbound])
        # Assert the underlying graph structure is correct as well.
        gate_indices = {eq_lib.graph[node].key.name: node for node in eq_lib.graph.node_indices()}
        self.assertTrue(eq_lib.graph.has_edge(gate_indices["a"], gate_indices["target"]))
        self.assertTrue(eq_lib.graph.has_edge(gate_indices["b"], gate_indices["target"]))
        self.assertTrue(eq_lib.graph.has_edge(gate_indices["target"], gate_indices["c"]))

        new = QuantumCircuit([Qubit()])
        new.append(gates["d"], [0])
        eq_lib.set_entry(target, [new])

        self.assertEqual(eq_lib.get_entry(target), [new])
        self.assertEqual(eq_lib.get_entry(gates["c"]), [outbound])
        # Assert the underlying graph structure is correct as well.
        gate_indices = {eq_lib.graph[node].key.name: node for node in eq_lib.graph.node_indices()}
        self.assertFalse(eq_lib.graph.has_edge(gate_indices["a"], gate_indices["target"]))
        self.assertFalse(eq_lib.graph.has_edge(gate_indices["b"], gate_indices["target"]))
        self.assertTrue(eq_lib.graph.has_edge(gate_indices["d"], gate_indices["target"]))
        self.assertTrue(eq_lib.graph.has_edge(gate_indices["target"], gate_indices["c"]))

    def test_set_entry_parallel_edges(self):
        """Test that `set_entry` works correctly in the case of parallel wires."""
        eq_lib = EquivalenceLibrary()
        gates = {key: Gate(key, 1, []) for key in "abcd"}
        target = Gate("target", 1, [])

        old_1 = QuantumCircuit([Qubit()], name="a")
        old_1.append(gates["a"], [0])
        old_1.append(gates["b"], [0])
        eq_lib.add_equivalence(target, old_1)

        old_2 = QuantumCircuit([Qubit()], name="b")
        old_2.append(gates["b"], [0])
        old_2.append(gates["a"], [0])
        eq_lib.add_equivalence(target, old_2)

        # This extra rule is so that 'a' still has edges, so we can do an exact isomorphism test.
        # There's not particular requirement for `set_entry` to remove orphan nodes, so we'll just
        # craft a test that doesn't care either way.
        a_to_b = QuantumCircuit([Qubit()])
        a_to_b.append(gates["b"], [0])
        eq_lib.add_equivalence(gates["a"], a_to_b)

        self.assertEqual(sorted(eq_lib.get_entry(target), key=lambda qc: qc.name), [old_1, old_2])

        new = QuantumCircuit([Qubit()], name="c")
        # No more use of 'a', but re-use 'b' and introduce 'c'.
        new.append(gates["b"], [0])
        new.append(gates["c"], [0])
        eq_lib.set_entry(target, [new])

        self.assertEqual(eq_lib.get_entry(target), [new])

        expected = EquivalenceLibrary()
        expected.add_equivalence(gates["a"], a_to_b)
        expected.add_equivalence(target, new)

        def node_fn(left, right):
            return left == right

        def edge_fn(left, right):
            return left.rule == right.rule

        self.assertTrue(rx.is_isomorphic(eq_lib.graph, expected.graph, node_fn, edge_fn))

    def test_raise_if_gate_entry_shape_mismatch(self):
        """Verify we raise if adding a circuit and gate with different shapes."""
        # This could be relaxed in the future to e.g. support ancilla management.

        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit(2)
        equiv.h(0)

        with self.assertRaises(CircuitError):
            eq_lib.add_equivalence(gate, equiv)

    def test_has_entry(self):
        """Verify we find an entry defined in the library."""

        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit([Qubit()])
        equiv.h(0)

        eq_lib.add_equivalence(gate, equiv)

        self.assertTrue(eq_lib.has_entry(gate))
        self.assertTrue(eq_lib.has_entry(OneQubitZeroParamGate()))

    def test_has_not_entry(self):
        """Verify we don't find an entry not defined in the library."""

        eq_lib = EquivalenceLibrary()

        self.assertFalse(eq_lib.has_entry(OneQubitZeroParamGate()))

    def test_equivalence_graph(self):
        """Verify valid graph created by add_equivalence"""

        eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit([Qubit()])
        first_equiv.h(0)
        eq_lib.add_equivalence(gate, first_equiv)

        equiv_copy = eq_lib._get_equivalences(Key(name="1q0p", num_qubits=1))[0].circuit

        egraph = rx.PyDiGraph()
        node_wt = NodeData(
            key=Key(name="1q0p", num_qubits=1), equivs=[Equivalence(params=[], circuit=equiv_copy)]
        )

        egraph.add_node(node_wt)

        node_wt = NodeData(key=Key(name="h", num_qubits=1), equivs=[])
        egraph.add_node(node_wt)

        edge_wt = EdgeData(
            index=0,
            num_gates=1,
            rule=Equivalence(params=[], circuit=equiv_copy),
            source=Key(name="h", num_qubits=1),
        )
        egraph.add_edge(0, 1, edge_wt)

        for node in eq_lib.graph.nodes():
            self.assertTrue(node in egraph.nodes())
            for edge in eq_lib.graph.edges():
                self.assertTrue(edge in egraph.edges())

        self.assertEqual(len(eq_lib.graph.nodes()), len(egraph.nodes()))
        self.assertEqual(len(eq_lib.graph.edges()), len(egraph.edges()))

        keys = {Key(name="1q0p", num_qubits=1): 0, Key(name="h", num_qubits=1): 1}.keys()
        self.assertEqual(keys, eq_lib.keys())


class TestEquivalenceLibraryWithBase(QiskitTestCase):
    """Test cases for EquivalenceLibrary with base library."""

    def test_create_empty_library_with_base(self):
        """Verify retrieving from an empty library returns an empty entry."""
        base = EquivalenceLibrary()

        eq_lib = EquivalenceLibrary(base=base)
        self.assertIsInstance(eq_lib, EquivalenceLibrary)

        gate = OneQubitZeroParamGate()

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 0)

    def test_get_through_empty_library_to_base(self):
        """Verify we find an entry defined only in the base library."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit([Qubit()])
        equiv.h(0)
        base.add_equivalence(gate, equiv)

        eq_lib = EquivalenceLibrary(base=base)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertIsNot(entry[0], equiv)
        self.assertEqual(entry[0], equiv)

    def test_add_equivalence(self):
        """Verify we find all equivalences if a gate is added to top and base."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit([Qubit()])
        first_equiv.h(0)
        base.add_equivalence(gate, first_equiv)

        eq_lib = EquivalenceLibrary(base=base)

        second_equiv = QuantumCircuit([Qubit()])
        second_equiv.append(U2Gate(0, np.pi), [0])

        eq_lib.add_equivalence(gate, second_equiv)

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 2)
        self.assertNotEqual(entry[0], entry[1])
        self.assertTrue(entry[0] in [first_equiv, second_equiv])
        self.assertTrue(entry[1] in [first_equiv, second_equiv])

    def test_set_entry(self):
        """Verify we find only equivalences from top when explicitly set."""
        base = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit([Qubit()])
        first_equiv.h(0)
        base.add_equivalence(gate, first_equiv)

        eq_lib = EquivalenceLibrary(base=base)

        second_equiv = QuantumCircuit([Qubit()])
        second_equiv.append(U2Gate(0, np.pi), [0])

        eq_lib.set_entry(gate, [second_equiv])

        entry = eq_lib.get_entry(gate)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], second_equiv)

    def test_has_entry_in_base(self):
        """Verify we find an entry defined in the base library."""

        base_eq_lib = EquivalenceLibrary()

        gate = OneQubitZeroParamGate()
        equiv = QuantumCircuit([Qubit()])
        equiv.h(0)

        base_eq_lib.add_equivalence(gate, equiv)

        eq_lib = EquivalenceLibrary(base=base_eq_lib)

        self.assertTrue(eq_lib.has_entry(gate))
        self.assertTrue(eq_lib.has_entry(OneQubitZeroParamGate()))

        gate = OneQubitZeroParamGate()
        equiv2 = QuantumCircuit([Qubit()])
        equiv.append(U2Gate(0, np.pi), [0])

        eq_lib.add_equivalence(gate, equiv2)

        self.assertTrue(eq_lib.has_entry(gate))
        self.assertTrue(eq_lib.has_entry(OneQubitZeroParamGate()))

    def test_has_not_entry_in_base(self):
        """Verify we find an entry not defined in the base library."""

        base_eq_lib = EquivalenceLibrary()
        eq_lib = EquivalenceLibrary(base=base_eq_lib)

        self.assertFalse(eq_lib.has_entry(OneQubitZeroParamGate()))


class TestEquivalenceLibraryWithParameters(QiskitTestCase):
    """Test cases for EquivalenceLibrary with gate parameters."""

    def test_raise_if_gate_equiv_parameter_mismatch(self):
        """Verify we raise if adding a circuit and gate with different sets of parameters."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter("theta")
        phi = Parameter("phi")

        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit([Qubit()])
        equiv.p(phi, 0)

        with self.assertRaises(CircuitError):
            eq_lib.add_equivalence(gate, equiv)

        with self.assertRaises(CircuitError):
            eq_lib.set_entry(gate, [equiv])

    def test_parameter_in_parameter_out(self):
        """Verify query parameters will be included in returned entry."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter("theta")

        gate = OneQubitOneParamGate(theta)
        equiv = QuantumCircuit([Qubit()])
        equiv.p(theta, 0)

        eq_lib.add_equivalence(gate, equiv)

        phi = Parameter("phi")
        gate_phi = OneQubitOneParamGate(phi)

        entry = eq_lib.get_entry(gate_phi)

        expected = QuantumCircuit([Qubit()])
        expected.p(phi, 0)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], expected)

    def test_partial_parameter_in_parameter_out(self):
        """Verify numeric query parameters will be included in returned entry."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter("theta")
        phi = Parameter("phi")

        gate = OneQubitTwoParamGate(theta, phi)
        equiv = QuantumCircuit([Qubit()])
        equiv.u(theta, phi, 0, 0)

        eq_lib.add_equivalence(gate, equiv)

        lam = Parameter("lam")
        gate_partial = OneQubitTwoParamGate(lam, 1.59)

        entry = eq_lib.get_entry(gate_partial)

        expected = QuantumCircuit([Qubit()])
        expected.u(lam, 1.59, 0, 0)

        self.assertEqual(len(entry), 1)
        self.assertEqual(entry[0], expected)

    def test_adding_gate_under_different_parameters(self):
        """Verify a gate can be added under different sets of parameters."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter("theta")

        gate_theta = OneQubitOneParamGate(theta)
        equiv_theta = QuantumCircuit([Qubit()])
        equiv_theta.p(theta, 0)

        eq_lib.add_equivalence(gate_theta, equiv_theta)

        phi = Parameter("phi")
        gate_phi = OneQubitOneParamGate(phi)
        equiv_phi = QuantumCircuit([Qubit()])
        equiv_phi.rz(phi, 0)

        eq_lib.add_equivalence(gate_phi, equiv_phi)

        lam = Parameter("lam")
        gate_query = OneQubitOneParamGate(lam)

        entry = eq_lib.get_entry(gate_query)

        first_expected = QuantumCircuit([Qubit()])
        first_expected.p(lam, 0)

        second_expected = QuantumCircuit([Qubit()])
        second_expected.rz(lam, 0)

        self.assertEqual(len(entry), 2)
        self.assertEqual(entry[0], first_expected)
        self.assertEqual(entry[1], second_expected)

    def test_adding_gate_and_partially_specified_gate(self):
        """Verify entries will different numbers of parameters will be returned."""
        eq_lib = EquivalenceLibrary()

        theta = Parameter("theta")
        phi = Parameter("phi")

        # e.g. RGate(theta, phi)
        gate_full = OneQubitTwoParamGate(theta, phi)
        equiv_full = QuantumCircuit([Qubit()])
        equiv_full.append(U2Gate(theta, phi), [0])

        eq_lib.add_equivalence(gate_full, equiv_full)

        gate_partial = OneQubitTwoParamGate(theta, 0)
        equiv_partial = QuantumCircuit([Qubit()])
        equiv_partial.rx(theta, 0)

        eq_lib.add_equivalence(gate_partial, equiv_partial)

        lam = Parameter("lam")
        gate_query = OneQubitTwoParamGate(lam, 0)

        entry = eq_lib.get_entry(gate_query)

        first_expected = QuantumCircuit([Qubit()])
        first_expected.append(U2Gate(lam, 0), [0])

        second_expected = QuantumCircuit([Qubit()])
        second_expected.rx(lam, 0)

        self.assertEqual(len(entry), 2)
        self.assertEqual(entry[0], first_expected)
        self.assertEqual(entry[1], second_expected)


class TestSessionEquivalenceLibrary(QiskitTestCase):
    """Test cases for SessionEquivalenceLibrary."""

    def test_converter_gate_registration(self):
        """Verify converters register gates in session equivalence library."""
        qc_gate = QuantumCircuit([Qubit() for _ in range(2)])
        qc_gate.h(0)
        qc_gate.cx(0, 1)

        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        bell_gate = circuit_to_gate(qc_gate, equivalence_library=sel)

        qc_inst = QuantumCircuit([Qubit() for _ in range(2)])
        qc_inst.h(0)
        qc_inst.cx(0, 1)

        bell_inst = circuit_to_instruction(qc_inst, equivalence_library=sel)

        gate_entry = sel.get_entry(bell_gate)
        inst_entry = sel.get_entry(bell_inst)

        self.assertEqual(len(gate_entry), 1)
        self.assertEqual(len(inst_entry), 1)

        self.assertEqual(gate_entry[0], qc_gate)
        self.assertEqual(inst_entry[0], qc_inst)

    def test_gate_decomposition_properties(self):
        """Verify decompositions are accessible via gate properties."""
        qc = QuantumCircuit([Qubit() for _ in range(2)])
        qc.h(0)
        qc.cx(0, 1)

        from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel

        gate = circuit_to_gate(qc, equivalence_library=sel)

        decomps = gate.decompositions

        self.assertEqual(len(decomps), 1)
        self.assertEqual(decomps[0], qc)

        qc2 = QuantumCircuit([Qubit() for _ in range(2)])
        qc2.h([0, 1])
        qc2.cz(0, 1)
        qc2.h(1)

        gate.add_decomposition(qc2)

        decomps = gate.decompositions

        self.assertEqual(len(decomps), 2)
        self.assertEqual(decomps[0], qc)
        self.assertEqual(decomps[1], qc2)

        gate.decompositions = [qc2]

        decomps = gate.decompositions

        self.assertEqual(len(decomps), 1)
        self.assertEqual(decomps[0], qc2)


class TestEquivalenceLibraryVisualization(QiskitVisualizationTestCase):
    """Test cases for EquivalenceLibrary visualization."""

    @unittest.skipUnless(optionals.HAS_GRAPHVIZ, "Graphviz not installed")
    @unittest.skipUnless(optionals.HAS_PIL, "PIL not installed")
    def test_equivalence_draw(self):
        """Verify EquivalenceLibrary drawing with reference image."""
        sel = EquivalenceLibrary()
        gate = OneQubitZeroParamGate()
        first_equiv = QuantumCircuit([Qubit()])
        first_equiv.h(0)

        sel.add_equivalence(gate, first_equiv)

        second_equiv = QuantumCircuit([Qubit()])
        second_equiv.append(U2Gate(0, np.pi), [0])

        sel.add_equivalence(gate, second_equiv)

        image = sel.draw()
        image_ref = path_to_diagram_reference("equivalence_library.png")
        self.assertImagesAreEqual(image, image_ref, 0.04)
