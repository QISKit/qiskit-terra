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

"""Tests PassManager.run()"""

from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit.library import CXGate
from qiskit.transpiler.preset_passmanagers import level_1_pass_manager
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap, Layout, PassManager, PropertySet
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passes import Decompose
from ..legacy_cmaps import ALMADEN_CMAP
from test import QiskitTestCase


class TestPassManagerRun(QiskitTestCase):
    """Test default_pass_manager.run(circuit(s))."""

    def test_bare_pass_manager_single(self):
        """Test that PassManager.run(circuit) returns a single circuit."""
        qc = QuantumCircuit(1)
        pm = PassManager([])
        new_qc = pm.run(qc)
        self.assertIsInstance(new_qc, QuantumCircuit)
        self.assertEqual(qc, new_qc)  # pm has no passes

    def test_bare_pass_manager_single_list(self):
        """Test that PassManager.run([circuit]) returns a list with a single circuit."""
        qc = QuantumCircuit(1)
        pm = PassManager([])
        result = pm.run([qc])
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], QuantumCircuit)
        self.assertEqual(result[0], qc)  # pm has no passes

    def test_bare_pass_manager_multiple(self):
        """Test that PassManager.run(circuits) returns a list of circuits."""
        qc0 = QuantumCircuit(1)
        qc1 = QuantumCircuit(2)

        pm = PassManager([])
        result = pm.run([qc0, qc1])

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)

        for qc, new_qc in zip([qc0, qc1], result):
            self.assertIsInstance(new_qc, QuantumCircuit)
            self.assertEqual(new_qc, qc)  # pm has no passes

    def test_preserve_layout_metadata_from_initial_property_set(self):
        """Test layout metadata in an initial property set is not overwritten."""
        circuit = QuantumCircuit(2)
        circuit.h(0)
        laid_out = transpile(
            circuit,
            coupling_map=CouplingMap.from_line(3),
            initial_layout=[0, 2],
            optimization_level=0,
        )
        property_set = PropertySet()
        laid_out.layout.write_into_property_set(property_set)

        output = PassManager([Decompose()]).run(laid_out, property_set=property_set)

        self.assertEqual(output.layout, laid_out.layout)

    def test_default_pass_manager_single(self):
        """Test default_pass_manager.run(circuit).

        circuit:
        qr0:-[H]--.------------  -> 1
                  |
        qr1:-----(+)--.--------  -> 2
                      |
        qr2:---------(+)--.----  -> 3
                          |
        qr3:-------------(+)---  -> 5

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=ALMADEN_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        pass_manager = level_1_pass_manager(
            PassManagerConfig.from_backend(
                backend,
                initial_layout=Layout.from_qubit_list(initial_layout),
                seed_transpiler=42,
            )
        )
        new_circuit = pass_manager.run(circuit)
        self.assertIsInstance(new_circuit, QuantumCircuit)

        bit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qregs[0])}

        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([bit_indices[x] for x in instruction.qubits], ALMADEN_CMAP)

    def test_default_pass_manager_two(self):
        """Test default_pass_manager.run(circuitS).

        circuit1 and circuit2:
        qr0:-[H]--.------------  -> 1
                  |
        qr1:-----(+)--.--------  -> 2
                      |
        qr2:---------(+)--.----  -> 3
                          |
        qr3:-------------(+)---  -> 5

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        qr = QuantumRegister(4, "qr")
        circuit1 = QuantumCircuit(qr)
        circuit1.h(qr[0])
        circuit1.cx(qr[0], qr[1])
        circuit1.cx(qr[1], qr[2])
        circuit1.cx(qr[2], qr[3])

        circuit2 = QuantumCircuit(qr)
        circuit2.cx(qr[1], qr[2])
        circuit2.cx(qr[0], qr[1])
        circuit2.cx(qr[2], qr[3])

        coupling_map = [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 6],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [3, 8],
            [4, 3],
            [5, 6],
            [5, 10],
            [6, 1],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [7, 12],
            [8, 3],
            [8, 7],
            [8, 9],
            [9, 8],
            [9, 14],
            [10, 5],
            [10, 11],
            [11, 10],
            [11, 12],
            [11, 16],
            [12, 7],
            [12, 11],
            [12, 13],
            [13, 12],
            [13, 14],
            [13, 18],
            [14, 9],
            [14, 13],
            [15, 16],
            [16, 11],
            [16, 15],
            [16, 17],
            [17, 16],
            [17, 18],
            [18, 13],
            [18, 17],
            [18, 19],
            [19, 18],
        ]
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=coupling_map,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            seed=42,
        )

        pass_manager = level_1_pass_manager(
            PassManagerConfig.from_backend(
                backend=backend,
                initial_layout=Layout.from_qubit_list(initial_layout),
                seed_transpiler=42,
            )
        )
        new_circuits = pass_manager.run([circuit1, circuit2])
        self.assertIsInstance(new_circuits, list)
        self.assertEqual(len(new_circuits), 2)

        for new_circuit in new_circuits:
            self.assertIsInstance(new_circuit, QuantumCircuit)
            bit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qregs[0])}

            for instruction in new_circuit.data:
                if isinstance(instruction.operation, CXGate):
                    self.assertIn([bit_indices[x] for x in instruction.qubits], coupling_map)
