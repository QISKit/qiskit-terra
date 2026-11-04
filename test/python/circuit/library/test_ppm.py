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

"""A Test for Pauli Product Measurement instruction."""

import io

from ddt import ddt, data
from qiskit.circuit import QuantumCircuit, CircuitError
from qiskit.circuit.library import PauliProductMeasurement
from qiskit.quantum_info import Pauli, Clifford
from qiskit.qpy import dump, load
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap, Target
from qiskit.transpiler.passes import (
    RemoveDiagonalGatesBeforeMeasure,
    ElidePermutations,
    OptimizeSwapBeforeMeasure,
    ResetAfterMeasureSimplification,
    BarrierBeforeFinalMeasurements,
)
from test import QiskitTestCase


@ddt
class TestPauliProductMeasurement(QiskitTestCase):
    """Test the PauliProductMeasurement Instruction."""

    @data("-XIYZ", "ZIYXY", "-Y", "-ZXII")
    def test_pauli_evolution(self, p):
        """Asserts that the pauli evolution is correct and that
        the circuit around the measure reduces to identity."""
        pauli = Pauli(p)
        num_qubits = pauli.num_qubits
        ppm = PauliProductMeasurement(pauli)
        qc_before_meas = QuantumCircuit(num_qubits)
        qc_no_meas = QuantumCircuit(num_qubits)
        for inst in ppm.definition.data:
            if inst.operation.name != "measure":
                qc_no_meas.append(inst.operation, inst.qubits)
        for inst in ppm.definition.data:
            if inst.operation.name == "measure":
                break
            qc_before_meas.append(inst.operation, inst.qubits)

        ind_z = 0
        for q in pauli:
            if Pauli(q) != Pauli("I"):
                break
            ind_z += 1
        cliff = Clifford(qc_before_meas)
        pauli_z = Pauli((num_qubits - 1 - ind_z) * "I" + "Z" + ind_z * "I")
        self.assertEqual(pauli_z.evolve(cliff), pauli)
        self.assertEqual(Clifford(qc_no_meas), Clifford(QuantumCircuit(num_qubits)))

    @data("-iX", "iZY")
    def test_raises_on_bad_phase(self, p):
        """Test that creating a PauliProductMeasurement instruction
        from a Pauli with phase i or -i raises an error.
        """
        with self.assertRaises(CircuitError):
            _ = PauliProductMeasurement(Pauli(p))

    @data("", "II", "-III")
    def test_raises_on_bad_pauli_label(self, p):
        """Test that creating a PauliProductMeasurement instruction
        from a Pauli with either an empty label or an all-"I" label
        raises an error.
        """
        with self.assertRaises(CircuitError):
            _ = PauliProductMeasurement(Pauli(p))

    def test_inverse_raises(self):
        """Test that the inverse method raises an error."""
        with self.assertRaises(CircuitError):
            _ = PauliProductMeasurement(Pauli("XYZ")).inverse()

    def test_qpy(self):
        """Test qpy for circuits with PauliProductMeasurement instructions."""
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [0])
        qc.append(PauliProductMeasurement(Pauli("ZZ")), [3, 2], [0])
        qc.h(0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_qpy_with_labeled_instructions(self):
        """Test qpy for circuits with PauliProductMeasurement instructions."""
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZ"), label="Alice"), [4, 1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z"), label="Bob"), [2], [0])
        qc.h(0)

        qpy_file = io.BytesIO()
        dump(qc, qpy_file)
        qpy_file.seek(0)
        new_circuit = load(qpy_file)[0]
        self.assertEqual(qc, new_circuit)

    def test_instructions_equal(self):
        """Test checking equality of PauliProductMeasurement instructions."""
        self.assertEqual(PauliProductMeasurement(Pauli("XZ")), PauliProductMeasurement(Pauli("XZ")))
        self.assertNotEqual(
            PauliProductMeasurement(Pauli("XZ")), PauliProductMeasurement(Pauli("XX"))
        )

    def test_add_instruction_to_circuit(self):
        """Test that adding a PauliProductMeasurement instruction to a circuit
        and retrieving it back is equivalent to the original instruction.
        """
        # Note that adding a PauliProductMeasurement instruction to a circuit
        # converts the Python instruction to a Rust instruction.
        # Copying the circuit avoids the gate-caching optimization, so that
        # the queried instruction is a Python instruction faithfully constructed
        # from the Rust instruction.
        gate = PauliProductMeasurement(Pauli("-XZ"))
        qc = QuantumCircuit(3, 2)
        qc.append(gate, [1, 2], [1])
        qc1 = qc.copy()
        gate_on_circuit = qc1[0].operation
        self.assertEqual(gate, gate_on_circuit)

    def test_circuits_with_instructions_equal(self):
        """Test checking equality of circuits with PauliProductMeasurement instructions."""
        qc1 = QuantumCircuit(5, 2)
        qc1.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])

        qc2 = QuantumCircuit(5, 2)
        qc2.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])

        qc3 = QuantumCircuit(5, 2)
        qc3.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [0])

        qc4 = QuantumCircuit(5, 2)
        qc4.append(PauliProductMeasurement(Pauli("ZX")), [4, 1], [1])

        self.assertEqual(qc1, qc2)
        self.assertNotEqual(qc1, qc3)
        self.assertNotEqual(qc1, qc4)

    def test_default_label_preserved(self):
        """
        Test that a default label is created correctly and
        preserved when a PauliProductMeasurement instruction
        is added to a circuit.
        """
        ppm = PauliProductMeasurement(Pauli("-XY"))
        self.assertEqual(ppm.label, "PPM(-XY)")

        qc = QuantumCircuit(2, 1)
        qc.append(ppm, [0, 1], [0])
        ppm_from_circuit = qc[0]

        self.assertEqual(ppm_from_circuit.label, ppm.label)

    def test_custom_label_preserved(self):
        """
        Test that a custom label is created correctly and
        preserved when a PauliProductMeasurement instruction
        is added to a circuit.
        """
        custom_label = "I Will Survive"
        ppm = PauliProductMeasurement(Pauli("-XY"), label=custom_label)
        self.assertEqual(ppm.label, custom_label)

        qc = QuantumCircuit(2, 1)
        qc.append(ppm, [0, 1], [0])
        ppm_from_circuit = qc[0]

        self.assertEqual(ppm_from_circuit.label, custom_label)

    def test_pauli_accessor(self):
        """Check that ``pauli()`` returns the original Pauli."""
        original = Pauli("-XZ")
        ppm = PauliProductMeasurement(original)
        self.assertEqual(ppm.pauli(), original)

    @data(0, 1, 2, 3)
    def test_transpile(self, optimization_level):
        """Check that transpiling circuits with PauliProductMeasurement instructions
        works as expected.
        """
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [0])
        qc.append(PauliProductMeasurement(Pauli("ZZ")), [3, 2], [0])
        qc.h(0)

        basis_gates = ["cx", "u"]
        coupling_map = CouplingMap.from_line(6)

        qct = transpile(
            qc,
            optimization_level=optimization_level,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
        )
        self.assertEqual(set(qct.count_ops()), {"cx", "u", "measure"})

    @data(0, 1, 2, 3)
    def test_transpile_with_target(self, optimization_level):
        """Check that transpiling circuits with PauliProductMeasurement instructions
        works as expected.
        """
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZ")), [4, 1], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [0])
        qc.append(PauliProductMeasurement(Pauli("ZZ")), [3, 2], [0])
        qc.h(0)

        basis_gates = ["cx", "u", "measure"]
        coupling_map = CouplingMap.from_line(6)
        target = Target.from_configuration(
            num_qubits=6, coupling_map=coupling_map, basis_gates=basis_gates
        )

        qct = transpile(qc, optimization_level=optimization_level, target=target)
        self.assertEqual(set(qct.count_ops()), {"cx", "u", "measure"})

    @data(
        RemoveDiagonalGatesBeforeMeasure(),
        ElidePermutations(),
        OptimizeSwapBeforeMeasure(),
        ResetAfterMeasureSimplification(),
        BarrierBeforeFinalMeasurements(),
    )
    def test_transpiler_passes_on_ppms(self, pass_):
        """Check that running various transpiler passes on circuits
        with PauliProductMeasurement instructions does not produce unexpected
        errors.
        """
        qc = QuantumCircuit(6, 2)
        qc.append(PauliProductMeasurement(Pauli("XZY")), [4, 1, 2], [1])
        qc.append(PauliProductMeasurement(Pauli("Z")), [2], [0])
        qc.append(PauliProductMeasurement(Pauli("ZZ")), [3, 2], [0])
        qc.h(0)

        _ = pass_(qc)
