# This code is part of Qiskit.
#
# (C) Copyright IBM 2023, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of GenericBackendV2 backend"""

import math
import operator
import unittest
from ddt import ddt, data

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap
from qiskit.utils import optionals
from qiskit.exceptions import QiskitError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from test import QiskitTestCase, combine


BACKEND_NUM_QUBITS = (5, 7, 16, 20, 27, 65, 127)


def _generic_backend(num_qubits):
    return GenericBackendV2(
        num_qubits=num_qubits, coupling_map=CouplingMap.from_ring(num_qubits), seed=42
    )


@ddt
class TestGenericBackendV2(QiskitTestCase):
    """Test class for GenericBackendV2 backend"""

    def setUp(self):
        super().setUp()
        self.cmap = CouplingMap(
            [(0, 2), (0, 1), (1, 3), (2, 4), (2, 3), (3, 5), (4, 6), (4, 5), (5, 7), (6, 7)]
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.circuit = QuantumCircuit(2)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.h(0)
        cls.circuit.h(1)
        cls.circuit.x(0)
        cls.circuit.x(1)
        cls.circuit.measure_all()

    def test_supported_basis_gates(self):
        """Test that target raises error if basis_gate not in ``supported_names``."""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=8, basis_gates=["cx", "id", "rz", "sx", "zz"], seed=42)

    def test_cx_1Q(self):
        """Test failing with a backend with single qubit but with a two-qubit basis gate"""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=1, basis_gates=["cx", "id"], seed=42)

    def test_ccx_2Q(self):
        """Test failing with a backend with two qubits but with a three-qubit basis gate"""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=2, basis_gates=["ccx", "id"], seed=42)

    def test_no_noise(self):
        """Test no noise info when parameter is false"""
        backend = GenericBackendV2(
            num_qubits=5, coupling_map=CouplingMap.from_line(5), noise_info=False, seed=42
        )
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 4)
        qc.cx(3, 0)
        qc.cx(2, 4)
        qc_res = generate_preset_pass_manager(optimization_level=2, backend=backend).run(qc)
        self.assertTrue(Operator.from_circuit(qc_res).equiv(qc))
        self.assertEqual(backend.target.qubit_properties, None)

    def test_no_noise_fully_connected(self):
        """Test no noise info when parameter is false"""
        backend = GenericBackendV2(num_qubits=5, noise_info=False, seed=42)
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 4)
        qc.cx(3, 0)
        qc.cx(2, 4)
        qc_res = generate_preset_pass_manager(optimization_level=2, backend=backend).run(qc)
        self.assertTrue(Operator.from_circuit(qc_res).equiv(qc))
        self.assertEqual(backend.target.qubit_properties, None)

    def test_no_info(self):
        """Test no noise info when parameter is false"""
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=CouplingMap.from_line(5),
            noise_info=False,
            seed=42,
        )
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 4)
        qc.cx(3, 0)
        qc.cx(2, 4)
        qc_res = generate_preset_pass_manager(optimization_level=2, backend=backend).run(qc)
        self.assertTrue(Operator.from_circuit(qc_res).equiv(qc))
        self.assertEqual(backend.target.qubit_properties, None)

    def test_operation_names(self):
        """Test that target basis gates include "delay", "measure" and "reset" even
        if not provided by user."""
        target = GenericBackendV2(num_qubits=8, seed=42)
        op_names = list(target.operation_names)
        op_names.sort()
        self.assertEqual(op_names, ["cx", "delay", "id", "measure", "reset", "rz", "sx", "x"])

        target = GenericBackendV2(num_qubits=8, basis_gates=["ecr", "id", "rz", "sx", "x"], seed=42)
        op_names = list(target.operation_names)
        op_names.sort()
        self.assertEqual(op_names, ["delay", "ecr", "id", "measure", "reset", "rz", "sx", "x"])

    def test_incompatible_coupling_map(self):
        """Test that the size of the coupling map must match num_qubits."""
        with self.assertRaises(QiskitError):
            GenericBackendV2(num_qubits=5, coupling_map=self.cmap, seed=42)

    def test_control_flow_operation_names(self):
        """Test that control flow instructions are added to the target if control_flow is True."""
        target = GenericBackendV2(
            num_qubits=8,
            basis_gates=["ecr", "id", "rz", "sx", "x"],
            coupling_map=self.cmap,
            control_flow=True,
            seed=42,
        ).target
        op_names = list(target.operation_names)
        op_names.sort()
        reference = [
            "box",
            "break",
            "continue",
            "delay",
            "ecr",
            "for_loop",
            "id",
            "if_else",
            "measure",
            "reset",
            "rz",
            "switch_case",
            "sx",
            "while_loop",
            "x",
        ]
        self.assertEqual(op_names, reference)

    def test_default_coupling_map(self):
        """Test that fully-connected coupling map is generated correctly."""

        # fmt: off
        reference_cmap = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (0, 4), (4, 0), (1, 2), (2, 1),
                          (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
        # fmt: on
        self.assertEqual(
            list(GenericBackendV2(num_qubits=5, seed=42).coupling_map.get_edges()),
            reference_cmap,
        )

    def test_run(self):
        """Test run method, confirm correct noisy simulation if Aer is installed."""

        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 4):
            qc.cx(qr[0], qr[k])
        qc.measure(qr, cr)

        backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], seed=42)
        tqc = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=42)
        result = backend.run(tqc, seed_simulator=42, shots=1000).result()
        counts = result.get_counts()

        self.assertTrue(math.isclose(counts["00000"], 500, rel_tol=0.1))
        self.assertTrue(math.isclose(counts["01111"], 500, rel_tol=0.1))

    def test_duration_defaults(self):
        """Test that the basis gates are assigned duration defaults within expected ranges."""

        basis_gates = ["cx", "id", "rz", "sx", "x", "sdg", "rxx"]
        expected_durations = {
            "cx": (7.992e-08, 8.99988e-07),
            "id": (2.997e-08, 5.994e-08),
            "rz": (0.0, 0.0),
            "sx": (2.997e-08, 5.994e-08),
            "x": (2.997e-08, 5.994e-08),
            "measure": (6.99966e-07, 1.500054e-06),
            "sdg": (2.997e-08, 5.994e-08),
            "rxx": (7.992e-08, 8.99988e-07),
        }
        for _ in range(20):
            target = GenericBackendV2(num_qubits=2, basis_gates=basis_gates, seed=42).target
            for inst in target:
                for qargs in target.qargs_for_operation_name(inst):
                    duration = target[inst][qargs].duration
                    if inst not in ["delay", "reset"]:
                        self.assertGreaterEqual(duration, expected_durations[inst][0])
                        self.assertLessEqual(duration, expected_durations[inst][1])

    def test_custom_dt(self):
        """Test that the custom dt is respected."""

        ref_backend = GenericBackendV2(num_qubits=2, basis_gates=["cx", "id"], seed=42)
        double_dt_backend = GenericBackendV2(
            num_qubits=2, basis_gates=["cx", "id"], dt=ref_backend.dt * 2, seed=42
        )
        self.assertEqual(ref_backend.dt * 2, double_dt_backend.dt)

    @combine(
        num_qubits=BACKEND_NUM_QUBITS,
        optimization_level=[0, 1, 2, 3],
    )
    def test_circuit_on_generic_backend_v2(self, num_qubits, optimization_level):
        """Test run method."""

        backend = _generic_backend(num_qubits)
        if not optionals.HAS_AER and backend.num_qubits > 20:
            self.skipTest(f"Unable to run generic backend {backend.name} without qiskit-aer")
        job = backend.run(
            transpile(
                self.circuit, backend, seed_transpiler=42, optimization_level=optimization_level
            ),
            seed_simulator=42,
        )
        result = job.result()
        counts = result.get_counts()
        max_count = max(counts.items(), key=operator.itemgetter(1))[0]
        self.assertEqual(max_count, "11")

    @data(*BACKEND_NUM_QUBITS)
    def test_backend_v2_dt(self, num_qubits):
        """Test default dt value is consistent with legacy fake backends."""

        backend = _generic_backend(num_qubits)
        target = backend.target
        if target.dt is not None:
            self.assertLess(target.dt, 1e-6)

    @data(*BACKEND_NUM_QUBITS)
    def test_backend_v2_dtm(self, num_qubits):
        """Test default dtm value is consistent with legacy fake backends."""

        backend = _generic_backend(num_qubits)
        if backend.dtm:
            self.assertLess(backend.dtm, 1e-6)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_generic_noise_model_always_present(self):
        """Test that GenericBackendV2 instances run with noise if Aer installed."""

        backend = GenericBackendV2(num_qubits=5, seed=42)
        backend.set_options(seed_simulator=42)
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.measure_all()

        res = backend.run(qc, shots=1000).result().get_counts()
        # Assert noise was present and result wasn't ideal
        self.assertNotEqual(res, {"1": 1000})
