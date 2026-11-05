# This code is part of Qiskit.
#
# (C) Copyright IBM 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test PBC transpilation pipeline."""

from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, Parameter, Clbit, Qubit
from qiskit.circuit.library import (
    QFTGate,
    MCXGate,
    UnitaryGate,
    RXGate,
    RYGate,
    RZGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    TGate,
    TdgGate,
    RXXGate,
    RZZGate,
    RYYGate,
    RZXGate,
    PhaseGate,
    U1Gate,
    PauliProductRotationGate,
    PauliProductMeasurement,
)
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.transpiler import TranspilerError
from qiskit.transpiler.passmanager import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pbc_pass_manager
from qiskit.transpiler.passes import HLSConfig, LitinskiTransformation
from qiskit.quantum_info import Operator, Pauli
from qiskit.converters import circuit_to_dag

from test import QiskitTestCase


@ddt
class TestPBCPassManager(QiskitTestCase):
    """Test PBC transpilation pipeline."""

    @data(0, 1, 2, 3)
    def test_all_standard_gates(self, optimization_level):
        """Test that PBC pipeline handles all standard gates."""
        qc = QuantumCircuit(4)
        for gate in get_standard_gate_name_mapping().values():
            if gate._standard_gate is None:
                continue
            qc.append(gate, qc.qubits[: gate.num_qubits], [])

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertEqual(set(transpiled.count_ops()), {"pauli_product_rotation"})

    def test_measures(self):
        """Test that PBC pipeline converts a measure instruction to a single PPM."""
        qc = QuantumCircuit(3, 1)
        qc.measure(1, 0)

        pm = generate_preset_pbc_pass_manager()
        transpiled = pm.run(qc)
        self.assertEqual(transpiled.count_ops(), {"pauli_product_measurement": 1})

    @data(0, 1, 2, 3)
    def test_qft(self, optimization_level):
        """Test that PBC pipeline handles QFT circuits."""

        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)

        self.assertEqual(set(transpiled.count_ops()), {"pauli_product_rotation"})
        self.assertEqual(Operator(transpiled), Operator(qc))

    @data(0, 1, 2, 3)
    def test_qft_with_measures(self, optimization_level):
        """Test that PBC pipeline handles circuits with barriers and measures."""
        qc = QuantumCircuit(4)
        qc.append(QFTGate(4), [0, 1, 2, 3])
        qc.measure_all()

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)

        self.assertEqual(
            set(transpiled.count_ops()),
            {"pauli_product_rotation", "pauli_product_measurement", "barrier"},
        )

    @data(0, 1, 2, 3)
    def test_reset(self, optimization_level):
        """Test that PBC pipeline handles circuits with resets."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.reset(0)
        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertEqual(set(transpiled.count_ops()), {"pauli_product_rotation", "reset"})

    @data(0, 1, 2, 3)
    def test_delay(self, optimization_level):
        """Test that PBC pipeline handles circuits with delays."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.delay(0)
        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertEqual(set(transpiled.count_ops()), {"pauli_product_rotation", "delay"})

    @data(0, 1, 2, 3)
    def test_nested_control_flow(self, optimization_level):
        """Test that PBC pipeline handles circuits with control flow."""
        for_body = QuantumCircuit(1)
        for_body.h(0)

        while_body = QuantumCircuit(1)
        while_body.rz(0.1, 0)

        qubit = Qubit()
        clbit = Clbit()

        true_body = QuantumCircuit([qubit, clbit])
        true_body.while_loop((clbit, True), while_body, [0], [])

        qc = QuantumCircuit([qubit, clbit])
        qc.for_loop(range(2), None, for_body, [0], [])
        qc.if_else((clbit, True), true_body, None, [0], [0])

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)

        # Counts ops recursively
        ops = circuit_to_dag(transpiled).count_ops()
        self.assertEqual(set(ops), {"for_loop", "if_else", "pauli_product_rotation", "while_loop"})

    @data(0, 1, 2, 3)
    def test_circuit_with_unitaries(self, optimization_level):
        """Test that PBC pipeline handles circuits with unitaries."""

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

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)

        self.assertEqual(set(transpiled.count_ops()), {"pauli_product_rotation"})
        self.assertEqual(Operator(transpiled), Operator(qc))

    @data(0, 1, 2, 3)
    def test_circuits_with_pprs_and_ppms(self, optimization_level):
        """Test that PBC pipeline handles circuits that already contain
        Pauli product rotations and measurements.
        """
        qc = QuantumCircuit(4, 2)
        qc.h(0)
        qc.append(PauliProductRotationGate(Pauli("XYZ"), 0.1), [1, 2, 3])
        qc.cx(1, 2)
        qc.measure(1, 1)
        qc.append(PauliProductMeasurement(Pauli("-XZ")), [1, 2], [1])

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertEqual(
            set(transpiled.count_ops()), {"pauli_product_rotation", "pauli_product_measurement"}
        )

    @data(0, 1, 2, 3)
    def test_pprs_and_ppms_pass_through(self, optimization_level):
        """Test that PBC pipeline does not change Pauli product rotation
        and measurements that are already in the circuit.
        """
        qc = QuantumCircuit(4, 2)
        qc.append(PauliProductRotationGate(Pauli("XYZ"), 0.1), [1, 2, 3])
        qc.append(PauliProductMeasurement(Pauli("-XZ")), [1, 2], [1])

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)
        self.assertEqual(qc, transpiled)

    @data(
        RXGate(0.1),
        RYGate(-0.3),
        RZGate(Parameter("theta")),
        XGate(),
        YGate(),
        ZGate(),
        SXGate(),
        SXdgGate(),
        SGate(),
        SdgGate(),
        TGate(),
        TdgGate(),
        RXXGate(0.2),
        RZXGate(Parameter("alpha")),
        RZZGate(-0.3),
        RYYGate(0.2),
        PhaseGate(0.2),
        U1Gate(-1.2),
    )
    def test_ppr_like_gates(self, gate):
        """Test that PBC pipeline converts each PPR-like rotation to a single PPRs."""
        qc = QuantumCircuit(gate.num_qubits)
        qc.append(gate, qc.qubits)

        pm = generate_preset_pbc_pass_manager()
        transpiled = pm.run(qc)
        self.assertEqual(transpiled.count_ops(), {"pauli_product_rotation": 1})

    @data(0, 1, 2, 3)
    def test_ccx(self, optimization_level):
        """Test that PBC pipeline translates CCX to 7 PPRs."""
        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        pm = generate_preset_pbc_pass_manager(optimization_level=optimization_level)
        transpiled = pm.run(qc)

        self.assertEqual(transpiled.count_ops(), {"pauli_product_rotation": 7})
        self.assertEqual(Operator(transpiled), Operator(qc))

    def test_raises_on_invalid_optimization_level(self):
        """Test that PBC pipeline raises an error when optimization_level is invalid."""
        with self.assertRaises(TranspilerError):
            _ = generate_preset_pbc_pass_manager(optimization_level=-1)

    def test_approximation_degree(self):
        """Test that PBC pipeline takes approximation_degree into account."""
        qc = QuantumCircuit(1)
        qc.rz(1e-4, 0)

        # With default approximation degree and optimization_level=2,
        # the RZ gate should not be removed.
        pm1 = generate_preset_pbc_pass_manager()
        transpiled1 = pm1.run(qc)
        self.assertEqual(transpiled1.count_ops(), {"pauli_product_rotation": 1})

        # With increased approximation degree, the RZ gate should not
        # be removed.
        pm2 = generate_preset_pbc_pass_manager(approximation_degree=1e-6)
        transpiled2 = pm2.run(qc)
        self.assertEqual(transpiled2, QuantumCircuit(1))

    def test_hls_config(self):
        """Test that PBC pipeline takes hls_config into account."""

        qc = QuantumCircuit(6)
        qc.append(MCXGate(4), [0, 1, 2, 3, 4])

        config1 = HLSConfig(mcx=["1_clean_kg24"])
        pm1 = generate_preset_pbc_pass_manager(seed_transpiler=0, hls_config=config1)
        qct1 = pm1.run(qc)
        ops1 = qct1.count_ops()
        self.assertEqual(set(ops1), {"pauli_product_rotation"})

        config2 = HLSConfig(mcx=["1_dirty_kg24"])
        pm2 = generate_preset_pbc_pass_manager(seed_transpiler=0, hls_config=config2)
        qct2 = pm2.run(qc)
        ops2 = qct2.count_ops()
        self.assertEqual(set(ops2), {"pauli_product_rotation"})

        # Specifying two different configs leads to different circuits
        self.assertNotEqual(qct1, qct2)

    def test_qubits_initially_zero(self):
        """Test that PBC pipeline takes test_qubits_initially_zero into account."""

        qc = QuantumCircuit(6)
        qc.append(MCXGate(4), [0, 1, 2, 3, 4])

        config = HLSConfig(mcx=["1_clean_kg24"])

        pm1 = generate_preset_pbc_pass_manager(
            seed_transpiler=0, hls_config=config, qubits_initially_zero=False
        )
        qct1 = pm1.run(qc)
        ops1 = qct1.count_ops()
        self.assertEqual(set(ops1), {"pauli_product_rotation"})

        pm2 = generate_preset_pbc_pass_manager(
            seed_transpiler=0, hls_config=config, qubits_initially_zero=True
        )
        qct2 = pm2.run(qc)
        ops2 = qct2.count_ops()
        self.assertEqual(set(ops2), {"pauli_product_rotation"})

        # Specifying different values for qubits_initially_zero leads  to different circuits
        self.assertNotEqual(qct1, qct2)

    def test_unitary_synthesis_method(self):
        """Test that PBC pipeline takes unitary_synthesis_method into account."""
        with self.assertRaises(TranspilerError):
            _ = generate_preset_pbc_pass_manager(unitary_synthesis_method="not a method")

    def test_unitary_synthesis_plugin_config(self):
        """Test that PBC pipeline takes unitary_synthesis_plugin_config into account."""
        qc1 = QuantumCircuit(1)
        qc1.rx(0.2, 0)
        u1 = UnitaryGate(Operator(qc1))

        qc = QuantumCircuit(3)
        qc.append(u1, [0])

        pm1 = generate_preset_pbc_pass_manager(
            seed_transpiler=0,
            unitary_synthesis_method="gridsynth",
            unitary_synthesis_plugin_config={"epsilon": 1e-6},
        )
        qct1 = pm1.run(qc)
        ops1 = qct1.count_ops()
        self.assertEqual(set(ops1), {"pauli_product_rotation"})

        pm2 = generate_preset_pbc_pass_manager(
            seed_transpiler=0,
            unitary_synthesis_method="gridsynth",
            unitary_synthesis_plugin_config={"epsilon": 1e-8},
        )
        qct2 = pm2.run(qc)
        ops2 = qct2.count_ops()
        self.assertEqual(set(ops2), {"pauli_product_rotation"})

        # Specifying different configs leads  to different circuits
        self.assertNotEqual(qct1, qct2)

    def test_redefine_translation_stage(self):
        """Test that one is able to redefine individual stages of the PBC pipeline."""
        pm = generate_preset_pbc_pass_manager()

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.rz(0.1, 2)

        # Run standard pass manager
        qct1 = pm.run(qc)
        self.assertEqual(qct1.count_ops(), {"pauli_product_rotation": 9})

        # Modify the translation stage to run LitinskiTransformation instead
        # ConvertToPauliRotations
        pm.pbc_translation = PassManager([LitinskiTransformation(use_ppr=True)])
        qct2 = pm.run(qc)
        self.assertEqual(qct2.count_ops(), {"pauli_product_rotation": 1, "h": 1, "cx": 2})
