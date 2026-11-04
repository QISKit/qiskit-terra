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

"""Test the 'clifford' unitary synthesis plugin."""

import math

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.converters import dag_to_circuit
from qiskit.quantum_info import Operator, get_clifford_gate_names
from qiskit.transpiler.passes import UnitarySynthesis, CliffordUnitarySynthesis
from test import QiskitTestCase


class TestCliffordSynthesis(QiskitTestCase):
    """Test the 'clifford' unitary synthesis plugin."""

    def setUp(self):
        super().setUp()
        self._create_unitaries()

    def _create_unitaries(self):
        """
        Creates unitary gates of varying sizes, some of which can be represented
        as Cliffords and some of which cannot.
        """

        # clifford unitary over 1 qubit
        c1 = QuantumCircuit(1)
        c1.h(0)
        c1.s(0)
        self.uc1 = UnitaryGate(Operator(c1).data)

        # clifford unitary over 2 qubits
        c2 = QuantumCircuit(2)
        c2.h(0)
        c2.rz(math.pi / 4, 1)
        c2.rz(math.pi / 4, 1)
        c2.sdg(1)
        self.uc2 = UnitaryGate(Operator(c2).data)

        # clifford unitary over 3 qubits
        c3 = QuantumCircuit(3)
        c3.h(0)
        c3.cx(0, 1)
        c3.cx(1, 2)
        c3.rx(math.pi / 2, 2)
        self.uc3 = UnitaryGate(Operator(c3).data)

        # non-clifford unitary over 1 qubit
        n1 = QuantumCircuit(1)
        n1.h(0)
        n1.rz(0.8, 0)
        self.un1 = UnitaryGate(Operator(n1).data)

        # non-clifford unitary over 2 qubits
        n2 = QuantumCircuit(2)
        n2.h(0)
        n2.rz(math.pi / 4, 1)
        n2.sdg(1)
        self.un2 = UnitaryGate(Operator(n2).data)

        # non-clifford unitary over 3 qubits
        n3 = QuantumCircuit(3)
        n3.h(0)
        n3.cx(0, 1)
        n3.cx(1, 2)
        n3.rx(0.8, 2)
        self.un3 = UnitaryGate(Operator(n3).data)

    def test_plugin(self):
        """Test running CliffordUnitarySynthesis plugin directly."""

        plugin = CliffordUnitarySynthesis()
        clifford_gate_names = set(get_clifford_gate_names())

        for gate in [self.uc1, self.uc2, self.uc3]:
            mat = gate.to_matrix()
            out = plugin.run(mat)
            self.assertLessEqual(set(out.count_ops()), clifford_gate_names)
            self.assertEqual(Operator(dag_to_circuit(out)), Operator(mat))

        for gate in [self.un1, self.un2, self.un3]:
            out = plugin.run(gate.to_matrix())
            self.assertIsNone(out)

    def test_plugin_with_parameters(self):
        """Test that we can pass parameters to the plugin."""

        plugin = CliffordUnitarySynthesis()
        for gate, config in [(self.uc1, {"min_qubits": 2}), (self.uc3, {"max_qubits": 2})]:
            mat = gate.to_matrix()
            out = plugin.run(mat, config=config)
            self.assertIsNone(out)

    def test_unitary_synthesis(self):
        """Test running the plugin from the unitary synthesis transpiler pass."""

        clifford_gate_names = set(get_clifford_gate_names())
        qubits = [2, 0, 1]

        for gate in [self.uc1, self.uc2, self.uc3]:
            qc = QuantumCircuit(3)
            qc.append(gate, qubits[: gate.num_qubits])
            transpiled = UnitarySynthesis(method="clifford")(qc)
            transpiled_ops = transpiled.count_ops()
            self.assertEqual(transpiled_ops.get("unitary", 0), 0)
            self.assertLessEqual(set(transpiled_ops), clifford_gate_names)
            self.assertEqual(Operator(transpiled), Operator(qc))

        for gate in [self.un1, self.un2, self.un3]:
            qc = QuantumCircuit(3)
            qc.append(gate, qubits[: gate.num_qubits])
            # the circuit should be unchanged
            transpiled = UnitarySynthesis(method="clifford")(qc)
            self.assertEqual(qc, transpiled)

    def test_unitary_synthesis_with_parameters(self):
        """Test that we can pass parameters to the plugin via unitary synthesis."""
        qubits = [2, 0, 1]

        for gate, config in [(self.uc1, {"min_qubits": 2}), (self.uc3, {"max_qubits": 2})]:
            qc = QuantumCircuit(3)
            qc.append(gate, qubits[: gate.num_qubits])
            transpiled = UnitarySynthesis(method="clifford", plugin_config=config)(qc)
            # the circuit should be unchanged
            self.assertEqual(qc, transpiled)
