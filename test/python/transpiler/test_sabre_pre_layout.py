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

"""Test the SabrePreLayout pass"""

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import TranspilerError, CouplingMap, PassManager
from qiskit.transpiler.passes.layout.sabre_pre_layout import SabrePreLayout
from qiskit.converters import circuit_to_dag
from test import QiskitTestCase


class TestSabrePreLayout(QiskitTestCase):
    """Tests the SabrePreLayout pass."""

    def test_no_constraints(self):
        """Test we raise at runtime if no target or coupling graph are provided."""
        qc = QuantumCircuit(2)
        empty_pass = SabrePreLayout(coupling_map=None)
        with self.assertRaises(TranspilerError):
            empty_pass.run(circuit_to_dag(qc))

    def test_starting_layout_created(self):
        """Test the case that no perfect layout exists and SabrePreLayout can find a
        starting layout."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(5)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map)])
        pm.run(qc)

        # SabrePreLayout should discover a single layout.
        self.assertIn("sabre_starting_layouts", pm.property_set)
        layouts = pm.property_set["sabre_starting_layouts"]
        self.assertEqual(len(layouts), 1)
        layout = layouts[0]
        self.assertEqual([layout[q] for q in qc.qubits], [0, 1, 2, 3])

    def test_perfect_layout_exists(self):
        """Test the case that a perfect layout exists."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(4)
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map)])
        pm.run(qc)

        # SabrePreLayout should not create starting layouts when a perfect layout exists.
        self.assertNotIn("sabre_starting_layouts", pm.property_set)

    def test_max_distance(self):
        """Test the ``max_distance`` option to SabrePreLayout."""
        qc = QuantumCircuit(6)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        coupling_map = CouplingMap.from_ring(6)

        # It is not possible to map a star-graph with 5 leaves into a ring with 6 nodes,
        # so that all nodes are distance-2 apart.
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map, max_distance=2)])
        pm.run(qc)
        self.assertNotIn("sabre_starting_layouts", pm.property_set)

        # But possible with distance-3.
        pm = PassManager([SabrePreLayout(coupling_map=coupling_map, max_distance=3)])
        pm.run(qc)
        self.assertIn("sabre_starting_layouts", pm.property_set)

    def test_call_limit_vf2(self):
        """Test the ``call_limit_vf2`` option to SabrePreLayout."""
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.cx(3, 0)
        coupling_map = CouplingMap.from_ring(5)
        pm = PassManager(
            [SabrePreLayout(coupling_map=coupling_map, call_limit_vf2=1, max_distance=3)]
        )
        pm.run(qc)
        self.assertNotIn("sabre_starting_layouts", pm.property_set)
