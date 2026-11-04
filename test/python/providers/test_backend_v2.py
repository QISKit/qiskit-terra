# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import math

from test import combine

from ddt import ddt, data

from numpy.testing import assert_array_max_ulp
from qiskit.circuit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library.standard_gates import (
    CXGate,
    ECRGate,
)
from qiskit.compiler import transpile
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import Operator
from qiskit.transpiler import InstructionProperties
from test import QiskitTestCase

from ..legacy_cmaps import TENERIFE_CMAP
from .fake_mumbai_v2 import FakeMumbaiFractionalCX


@ddt
class TestBackendV2(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.backend = GenericBackendV2(num_qubits=2, basis_gates=["rx", "u"], seed=42)
        cx_props = {
            (0, 1): InstructionProperties(duration=5.23e-7, error=0.00098115),
        }
        self.backend.target.add_instruction(CXGate(), cx_props)
        ecr_props = {
            (1, 0): InstructionProperties(duration=4.52e-9, error=0.0000132115),
        }
        self.backend.target.add_instruction(ECRGate(), ecr_props)
        self.backend.options.set_validator("shots", (1, 4096))

    def assertMatchesTargetConstraints(self, tqc, target):
        qubit_indices = {qubit: index for index, qubit in enumerate(tqc.qubits)}
        for instruction in tqc.data:
            qubits = tuple(qubit_indices[x] for x in instruction.qubits)
            target_set = target[instruction.operation.name].keys()
            self.assertIn(
                qubits,
                target_set,
                f"qargs: {qubits} not found in target for operation {instruction.operation.name}:"
                f" {set(target_set)}",
            )

    def test_qubit_properties(self):
        """Test that qubit properties are returned as expected."""
        props = self.backend.qubit_properties([1, 0])
        assert_array_max_ulp([0.0001697368029059364, 0.00017739560485559633], [x.t1 for x in props])
        assert_array_max_ulp(
            [0.00010941773478876496, 0.00014388784397520525], [x.t2 for x in props]
        )
        assert_array_max_ulp([5487811175.818378, 5429298959.955691], [x.frequency for x in props])

    def test_legacy_qubit_properties(self):
        """Test that qubit props work for backends not using properties in target."""

        class FakeBackendV2LegacyQubitProps(GenericBackendV2):
            """Fake backend that doesn't use qubit properties via the target."""

            def qubit_properties(self, qubit):
                if isinstance(qubit, int):
                    return self.target.qubit_properties[qubit]
                return [self.target.qubit_properties[i] for i in qubit]

        props = FakeBackendV2LegacyQubitProps(num_qubits=2, seed=42).qubit_properties([1, 0])
        assert_array_max_ulp([0.0001697368029059364, 0.00017739560485559633], [x.t1 for x in props])
        assert_array_max_ulp(
            [0.00010941773478876496, 0.00014388784397520525], [x.t2 for x in props]
        )
        assert_array_max_ulp([5487811175.818378, 5429298959.955691], [x.frequency for x in props])

    def test_no_qubit_properties_raises(self):
        """Ensure that if no qubit properties are defined we raise correctly."""
        with self.assertRaises(NotImplementedError):
            BasicSimulator().qubit_properties(0)

    def test_option_bounds(self):
        """Test that option bounds are enforced."""
        with self.assertRaises(ValueError) as cm:
            self.backend.set_options(shots=8192)
        self.assertEqual(
            str(cm.exception),
            "Specified value for 'shots' is not a valid value, must be >=1 and <=4096",
        )

    @data(0, 1, 2, 3)
    def test_transpile(self, opt_level):
        """Test that transpile() works with a BackendV2 backend."""
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.cz(1, 0)
        tqc = transpile(qc, self.backend, optimization_level=opt_level)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    @combine(
        opt_level=[0, 1, 2, 3],
        gate=["cx", "ecr", "cz"],
        bidirectional=[True, False],
        dsc=(
            "Test GHZ circuit with {gate} using opt level {opt_level} on backend "
            "with bidirectional={bidirectional}"
        ),
        name="{gate}_level_{opt_level}_bidirectional_{bidirectional}",
    )
    def test_5q_ghz(self, opt_level, gate, bidirectional):
        if bidirectional:
            backend = GenericBackendV2(num_qubits=5, seed=42)
        else:
            backend = GenericBackendV2(num_qubits=5, coupling_map=TENERIFE_CMAP, seed=42)
        qc = QuantumCircuit(5)
        qc.h(0)
        getattr(qc, gate)(0, 1)
        getattr(qc, gate)(2, 1)
        getattr(qc, gate)(2, 3)
        getattr(qc, gate)(4, 3)
        tqc = transpile(qc, backend, optimization_level=opt_level)
        t_op = Operator.from_circuit(tqc)
        self.assertTrue(t_op.equiv(qc))
        self.assertMatchesTargetConstraints(tqc, backend.target)

    def test_transpile_respects_arg_constraints(self):
        """Test that transpile() respects a heterogenous basis."""
        # Test CX on wrong link
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(1, 0)
        tqc = transpile(qc, self.backend, optimization_level=1)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        # Below is done to check we're decomposing cx(1, 0) with extra
        # rotations to correct for direction. However because of fp
        # differences between windows and other platforms the optimization
        # from the 1q optimization passes differ and the output gates
        # change (while still being equivalent). This relaxes the check to
        # still ensure it's valid but not so specific that it fails on windows
        self.assertEqual(tqc.count_ops().keys(), {"cx", "u"})
        self.assertEqual(tqc.count_ops()["cx"], 1)
        self.assertLessEqual(tqc.count_ops()["u"], 4)
        self.assertMatchesTargetConstraints(tqc, self.backend.target)
        # Test ECR on wrong link
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ecr(0, 1)
        tqc = transpile(qc, self.backend, optimization_level=1)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertEqual(tqc.count_ops(), {"ecr": 1, "u": 4})
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    def test_transpile_relies_on_gate_direction(self):
        """Test that transpile() relies on gate direction pass for 2q."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.ecr(0, 1)
        tqc = transpile(qc, self.backend, optimization_level=1)
        expected = QuantumCircuit(2)
        expected.u(0, 0, -math.pi, 0)
        expected.u(math.pi / 2, 0, 0, 1)
        expected.ecr(1, 0)
        expected.u(math.pi / 2, 0, -math.pi, 0)
        expected.u(math.pi / 2, 0, -math.pi, 1)
        self.assertTrue(Operator.from_circuit(tqc).equiv(qc))
        self.assertEqual(tqc.count_ops(), {"ecr": 1, "u": 4})
        self.assertMatchesTargetConstraints(tqc, self.backend.target)

    def test_transpile_mumbai_target(self):
        """Test that transpile respects a more involved target for a fake mumbai."""
        backend = FakeMumbaiFractionalCX()
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(1, 0)
        qc.measure_all()
        tqc = transpile(qc, backend, optimization_level=1)
        qr = QuantumRegister(27, "q")
        cr = ClassicalRegister(2, "meas")
        expected = QuantumCircuit(qr, cr, global_phase=math.pi / 4)
        expected.rz(math.pi / 2, 0)
        expected.sx(0)
        expected.rz(math.pi / 2, 0)
        expected.cx(1, 0)
        expected.barrier(qr[0], qr[1])
        expected.measure(qr[0], cr[0])
        expected.measure(qr[1], cr[1])
        self.assertEqual(expected, tqc)
