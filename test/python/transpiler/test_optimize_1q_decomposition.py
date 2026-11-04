# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the optimize-1q-gate pass"""

import unittest

import ddt
import numpy as np

from qiskit.circuit import (
    QuantumRegister,
    QuantumCircuit,
    ClassicalRegister,
    Parameter,
    Gate,
    Measure,
)
from qiskit.circuit.library.generalized_gates import UnitaryGate
from qiskit.circuit.library.standard_gates import (
    UGate,
    SXGate,
    PhaseGate,
    U3Gate,
    U2Gate,
    U1Gate,
    RZGate,
    RXGate,
    RYGate,
    HGate,
    SGate,
)
from qiskit.circuit.random import random_circuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, Target, InstructionProperties
from qiskit.transpiler.passes import Optimize1qGatesDecomposition
from qiskit.transpiler.passes import BasisTranslator
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.quantum_info import Operator
from test import QiskitTestCase


theta = Parameter("θ")
phi = Parameter("ϕ")
lambda_ = Parameter("λ")

# a typical target where u1 is cheaper than u2 is cheaper than u3
u1_props = {(0,): InstructionProperties(error=0)}
u2_props = {(0,): InstructionProperties(error=1e-4)}
u3_props = {(0,): InstructionProperties(error=2e-4)}
target_u1_u2_u3 = Target()
target_u1_u2_u3.add_instruction(U1Gate(theta), u1_props, name="u1")
target_u1_u2_u3.add_instruction(U2Gate(theta, phi), u2_props, name="u2")
target_u1_u2_u3.add_instruction(U3Gate(theta, phi, lambda_), u3_props, name="u3")

# a typical target where continuous rz and rx are available; rz is cheaper
rz_props = {(0,): InstructionProperties(duration=0, error=0)}
rx_props = {(0,): InstructionProperties(duration=0.5e-8, error=0.00025)}
target_rz_rx = Target()
target_rz_rx.add_instruction(RZGate(theta), rz_props, name="rz")
target_rz_rx.add_instruction(RXGate(theta), rx_props, name="rx")

# a typical target where continuous rz, and discrete sx are available; rz is cheaper
rz_props = {(0,): InstructionProperties(duration=0, error=0)}
sx_props = {(0,): InstructionProperties(duration=0.5e-8, error=0.00025)}
target_rz_sx = Target()
target_rz_sx.add_instruction(RZGate(theta), rz_props, name="rz")
target_rz_sx.add_instruction(SXGate(), sx_props, name="sx")

# a target with overcomplete basis, rz is cheaper than ry is cheaper than u
rz_props = {(0,): InstructionProperties(duration=0.1e-8, error=0.0001)}
ry_props = {(0,): InstructionProperties(duration=0.5e-8, error=0.0002)}
u_props = {(0,): InstructionProperties(duration=0.9e-8, error=0.0005)}
target_rz_ry_u = Target()
target_rz_ry_u.add_instruction(RZGate(theta), rz_props, name="rz")
target_rz_ry_u.add_instruction(RYGate(theta), ry_props, name="ry")
target_rz_ry_u.add_instruction(UGate(theta, phi, lambda_), u_props, name="u")

# a target with hadamard and phase, we don't yet have an explicit decomposer
# but we can at least recognize circuits that are native for it
h_props = {(0,): InstructionProperties(duration=0.3e-8, error=0.0003)}
p_props = {(0,): InstructionProperties(duration=0, error=0)}
target_h_p = Target()
target_h_p.add_instruction(HGate(), h_props, name="h")
target_h_p.add_instruction(PhaseGate(theta), p_props, name="p")

# a target with rz, ry, and u. Error are not specified so we should prefer
# shorter decompositions.
rz_props = {(0,): None}
ry_props = {(0,): None}
u_props = {(0,): None}
target_rz_ry_u_noerror = Target()
target_rz_ry_u_noerror.add_instruction(RZGate(theta), rz_props, name="rz")
target_rz_ry_u_noerror.add_instruction(RYGate(theta), ry_props, name="ry")
target_rz_ry_u_noerror.add_instruction(UGate(theta, phi, lambda_), u_props, name="u")


@ddt.ddt
class TestOptimize1qGatesDecomposition(QiskitTestCase):
    """Test for 1q gate optimizations."""

    def test_run_pass_in_parallel(self):
        """Test running pass on multiple circuits in parallel."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        passmanager = PassManager([Optimize1qGatesDecomposition(target=target_u1_u2_u3)])
        results = passmanager.run([circuit, circuit])
        for result in results:
            self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(target_u1_u2_u3, target_rz_rx, target_rz_sx, target_rz_ry_u, target_h_p)
    def test_optimize_h_gates_target(self, target):
        """Transpile: qr:--[H]-[H]-[H]--"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target))
        result = passmanager.run(circuit)

        self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(
        target_u1_u2_u3,
        target_rz_rx,
        target_rz_sx,
        target_rz_ry_u,
    )
    def test_optimize_identity_target(self, target):
        """Transpile: qr:--[RY(θ), RY(-θ)]-- to null."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.ry(np.pi / 7, qr[0])
        circuit.ry(-np.pi / 7, qr[0])

        expected = QuantumCircuit(qr)

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target))
        result = passmanager.run(circuit)
        self.assertEqual(expected, result)

    def test_optimize_away_idenity_no_target(self):
        """Test identity run is removed for no target specified."""
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.h(0)
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition())
        result = passmanager.run(circuit)
        self.assertEqual(QuantumCircuit(1), result)

    def test_optimize_error_over_target_basis_choice(self):
        """Euler basis selection must use target error rates, not just gate count.

        Regression test of gh-16460: with rz and rx both available, the pass
        must pick the basis with the lowest total error for the target at
        hand, so flipping which gate is expensive must flip the chosen basis.
        """

        def make_target(rx_error, rz_error):
            target = Target(num_qubits=1)
            target.add_instruction(RXGate(theta), {(0,): InstructionProperties(error=rx_error)})
            target.add_instruction(RZGate(theta), {(0,): InstructionProperties(error=rz_error)})
            return target

        def total_error(circuit, target):
            fidelity = 1.0
            for instruction in circuit.data:
                fidelity *= 1.0 - target[instruction.operation.name][(0,)].error
            return 1.0 - fidelity

        circuit = QuantumCircuit(1)
        circuit.rz(0.2, 0)
        circuit.rx(0.3, 0)
        circuit.rz(0.4, 0)
        circuit.rx(0.5, 0)
        circuit.rz(0.6, 0)

        for rx_error, rz_error in [(1e-2, 1e-6), (1e-6, 1e-2)]:
            with self.subTest(rx_error=rx_error, rz_error=rz_error):
                target = make_target(rx_error, rz_error)
                result = PassManager([Optimize1qGatesDecomposition(target=target)]).run(circuit)
                self.assertEqual(Operator(circuit), Operator(result))
                # the alternative Euler basis (rx<->rz swapped roles) must not
                # have been cheaper than what the pass selected
                swapped = QuantumCircuit(1)
                for instruction in result.data:
                    other = "rx" if instruction.operation.name == "rz" else "rz"
                    getattr(swapped, other)(*instruction.operation.params, 0)
                self.assertLessEqual(total_error(result, target), total_error(swapped, target))

    def test_optimize_basis_choice_no_error_uses_gate_count(self):
        """Without target error rates, basis selection falls back to gate count.

        Companion to test_optimize_error_over_target_basis_choice for the rz/rx
        family: when the target carries no error rates the pass must still return
        a correct, minimal-length decomposition rather than erroring.
        """
        target = Target(num_qubits=1)
        target.add_instruction(RXGate(theta), {(0,): None})
        target.add_instruction(RZGate(theta), {(0,): None})

        circuit = QuantumCircuit(1)
        circuit.rz(0.2, 0)
        circuit.rx(0.3, 0)
        circuit.rz(0.4, 0)
        circuit.rx(0.5, 0)
        circuit.rz(0.6, 0)

        result = PassManager([Optimize1qGatesDecomposition(target=target)]).run(circuit)
        self.assertEqual(Operator(circuit), Operator(result))
        # a generic 1q run is three gates in either rz/rx Euler basis; the pass must
        # not return something longer than the original run when it resynthesizes
        self.assertLessEqual(sum(result.count_ops().values()), len(circuit.data))

    def test_optimize_error_over_target_3(self):
        """U is shorter than RZ-RY-RZ or RY-RZ-RY so use it when no error given."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.u(np.pi / 7, np.pi / 4, np.pi / 3, qr[0])

        target = target_rz_ry_u_noerror
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target))
        result = passmanager.run(circuit)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0].operation, UGate)

    @ddt.data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cx", "u"],
        ["p", "sx", "u", "cx"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
        ["rz", "rx"],
        ["rz", "sx"],
        ["p", "sx"],
        ["r"],
    )
    def test_optimize_h_gates_pass_manager(self, basis):
        """Transpile: qr:--[H]-[H]-[H]--"""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.h(qr[0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        self.assertTrue(Operator(circuit).equiv(Operator(result)))
        self.assertLessEqual(result.depth(), circuit.depth())

    @ddt.data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cx", "u"],
        ["p", "sx", "u", "cx"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
        ["rz", "rx"],
        ["rz", "sx"],
        ["p", "sx"],
        ["r"],
    )
    def test_in_the_back(self, basis):
        """Optimizations can be in the back of the circuit.
        See https://github.com/Qiskit/qiskit-terra/issues/2004.

        qr0:--[U1]-[U1]-[H]--
        """
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        circuit.p(0.3, qr)
        circuit.p(0.4, qr)
        circuit.h(qr)

        expected = QuantumCircuit(qr)
        expected.p(0.7, qr)
        expected.h(qr)

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        self.assertTrue(Operator(circuit).equiv(Operator(result)))

    @ddt.data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cx", "u"],
        ["p", "sx", "u", "cx"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
        ["rz", "sx"],
        ["rz", "rx"],
        ["p", "sx"],
    )
    def test_single_parameterized_circuit(self, basis):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta_p = Parameter("theta")

        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta_p, qr)
        qc.p(0.1, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.assign_parameters({theta_p: 3.14})).equiv(
                Operator(result.assign_parameters({theta_p: 3.14}))
            )
        )

    @ddt.data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cx", "u"],
        ["p", "sx", "u", "cx"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
        ["rz", "rx"],
        ["rz", "sx"],
        ["p", "sx"],
    )
    def test_parameterized_circuits(self, basis):
        """Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta_p = Parameter("theta")

        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta_p, qr)
        qc.p(0.1, qr)
        qc.p(0.2, qr)
        qc.p(theta_p, qr)
        qc.p(0.3, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.assign_parameters({theta_p: 3.14})).equiv(
                Operator(result.assign_parameters({theta_p: 3.14}))
            )
        )

    @ddt.data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cx", "u"],
        ["p", "sx", "u", "cx"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
        ["rz", "rx"],
        ["rz", "sx"],
        ["p", "sx"],
    )
    def test_parameterized_expressions_in_circuits(self, basis):
        """Expressions of Parameters should be treated as opaque gates."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        theta_p = Parameter("theta")
        phi_p = Parameter("phi")

        sum_ = theta_p + phi_p
        product_ = theta_p * phi_p
        qc.p(0.3, qr)
        qc.p(0.4, qr)
        qc.p(theta_p, qr)
        qc.p(phi_p, qr)
        qc.p(sum_, qr)
        qc.p(product_, qr)
        qc.p(0.3, qr)
        qc.p(0.2, qr)

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)

        self.assertTrue(
            Operator(qc.assign_parameters({theta_p: 3.14, phi_p: 10})).equiv(
                Operator(result.assign_parameters({theta_p: 3.14, phi_p: 10}))
            )
        )

    def test_identity_xyx(self):
        """Test lone identity gates in rx ry basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.rx(0, 1)
        circuit.ry(0, 0)
        basis = ["rxx", "rx", "ry"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_zxz(self):
        """Test lone identity gates in rx rz basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.rx(0, 1)
        circuit.rz(0, 0)
        basis = ["cz", "rx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_psx(self):
        """Test lone identity gates in p sx basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.p(0, 0)
        basis = ["cx", "p", "sx"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_u(self):
        """Test lone identity gates in u basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.u(0, 0, 0, 0)
        basis = ["cx", "u"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_u3(self):
        """Test lone identity gates in u3 basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.append(U3Gate(0, 0, 0), [0])
        basis = ["cx", "u3"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_r(self):
        """Test lone identity gates in r basis are removed."""
        circuit = QuantumCircuit(1)
        circuit.r(0, 0, 0)
        basis = ["r"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_identity_rzx(self):
        """Test lone identity gates in rz rx basis are removed."""
        circuit = QuantumCircuit(2)
        circuit.rz(0, 0)
        circuit.rx(0, 1)
        basis = ["cx", "rz", "rx"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual([], result.data)

    def test_overcomplete_basis(self):
        """Test optimization with an overcomplete basis."""
        circuit = random_circuit(3, 3, seed=42)
        basis = ["rz", "rxx", "rx", "ry", "p", "sx", "u", "cx"]
        passmanager = PassManager()
        passmanager.append(BasisTranslator(sel, basis))
        basis_translated = passmanager.run(circuit)
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result_full = passmanager.run(basis_translated)
        self.assertTrue(Operator(circuit).equiv(Operator(result_full)))
        self.assertGreater(basis_translated.depth(), result_full.depth())

    def test_euler_decomposition_worse(self):
        """Ensure we don't decompose to a deeper circuit."""
        circuit = QuantumCircuit(1)
        circuit.rx(-np.pi / 2, 0)
        circuit.rz(-np.pi / 2, 0)
        basis = ["rx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        # decomposition of circuit will result in 3 gates instead of 2
        # assert optimization pass doesn't use it.
        self.assertEqual(circuit, result, f"Circuit:\n{circuit}\nResult:\n{result}")

    def test_euler_decomposition_worse_2(self):
        """Ensure we don't decompose to a deeper circuit in an edge case."""
        circuit = QuantumCircuit(1)
        circuit.rz(0.13, 0)
        circuit.ry(-0.14, 0)
        basis = ["ry", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual(circuit, result, f"Circuit:\n{circuit}\nResult:\n{result}")

    def test_euler_decomposition_zsx(self):
        """Ensure we don't decompose to a deeper circuit in the ZSX basis."""
        circuit = QuantumCircuit(1)
        circuit.rz(0.3, 0)
        circuit.sx(0)
        circuit.rz(0.2, 0)
        circuit.sx(0)

        basis = ["sx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual(circuit, result, f"Circuit:\n{circuit}\nResult:\n{result}")

    def test_euler_decomposition_zsx_2(self):
        """Ensure we don't decompose to a deeper circuit in the ZSX basis."""
        circuit = QuantumCircuit(1)
        circuit.sx(0)
        circuit.rz(0.2, 0)
        circuit.sx(0)
        circuit.rz(0.3, 0)

        basis = ["sx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)
        self.assertEqual(circuit, result, f"Circuit:\n{circuit}\nResult:\n{result}")

    def test_optimize_run_of_u_to_single_u_on_target_no_error(self):
        """U(pi/3, 0, 0) * U(pi/3, 0, 0) * U(pi/3, 0, 0) -> U(pi, 0, 0). See #9701."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        for _ in range(3):
            circuit.append(UGate(np.pi / 3, 0.0, 0.0), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(UGate(np.pi, 0.0, 0.0), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target_rz_ry_u_noerror))
        result = passmanager.run(circuit)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_optimize_run_of_u_to_single_u_no_target(self):
        """U(pi/3, 0, 0) * U(pi/3, 0, 0) * U(pi/3, 0, 0) -> U(pi, 0, 0). See #9701."""
        qr = QuantumRegister(1, "qr")
        circuit = QuantumCircuit(qr)
        for _ in range(3):
            circuit.append(UGate(np.pi / 3, 0.0, 0.0), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(UGate(np.pi, 0.0, 0.0), [qr[0]])

        basis = ["u"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_optimize_u_to_phase_gate(self):
        """U(0, 0, pi/4) ->  p(pi/4). Basis [p, sx]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(PhaseGate(np.pi / 4), [qr[0]])

        basis = ["p", "sx"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_optimize_u_to_p_sx_p(self):
        """U(pi/2, 0, pi/4) ->  p(-pi/4)-sx-p(p/2). Basis [p, sx]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(UGate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr, global_phase=-np.pi / 4)
        expected.append(PhaseGate(-np.pi / 4), [qr[0]])
        expected.append(SXGate(), [qr[0]])
        expected.append(PhaseGate(np.pi / 2), [qr[0]])

        basis = ["p", "sx"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(circuit)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_optimize_u3_to_u1(self):
        """U3(0, 0, pi/4) ->  U1(pi/4). Basis [u1, u2, u3]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(0, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U1Gate(np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target_u1_u2_u3))
        result = passmanager.run(circuit)

        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_optimize_u3_to_u2(self):
        """U3(pi/2, 0, pi/4) ->  U2(0, pi/4). Basis [u1, u2, u3]."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.append(U3Gate(np.pi / 2, 0, np.pi / 4), [qr[0]])

        expected = QuantumCircuit(qr)
        expected.append(U2Gate(0, np.pi / 4), [qr[0]])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(target=target_u1_u2_u3))
        result = passmanager.run(circuit)
        self.assertEqual(expected, result)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_y_simplification_rz_sx_x(self):
        """Test that a y gate gets decomposed to x-zx with ibmq basis."""
        qc = QuantumCircuit(1)
        qc.y(0)
        basis = ["id", "rz", "sx", "x", "cx"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)
        expected = QuantumCircuit(1)
        expected.rz(-np.pi, 0)
        expected.x(0)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_short_string(self):
        """Test that a shorter-than-universal string is still rewritten."""
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(np.pi / 2, 0)
        basis = ["sx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)
        expected = QuantumCircuit(1)
        expected.sx(0)
        expected.sx(0)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_u_rewrites_to_rz(self):
        """Test that a phase-like U-gate gets rewritten into an RZ gate."""
        qc = QuantumCircuit(1)
        qc.u(0, 0, np.pi / 6, 0)
        basis = ["sx", "rz"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)
        expected = QuantumCircuit(1, global_phase=np.pi / 12)
        expected.rz(np.pi / 6, 0)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_u_rewrites_to_phase(self):
        """Test that a phase-like U-gate gets rewritten into an RZ gate."""
        qc = QuantumCircuit(1)
        qc.u(0, 0, np.pi / 6, 0)
        basis = ["sx", "p"]
        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(qc)
        expected = QuantumCircuit(1)
        expected.p(np.pi / 6, 0)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_no_warning_on_hadamard(self):
        """
        Test Hadamards don't trigger the inefficiency warning when they're part of the basis.

        See #6848.
        """
        qc = QuantumCircuit(2)
        qc.cz(0, 1)
        basis = ["h", "cx", "rz", "sx", "x"]
        result = transpile(qc, basis_gates=basis)
        expected = QuantumCircuit(2)
        expected.h(1)
        expected.cx(0, 1)
        expected.h(1)
        msg = f"expected:\n{expected}\nresult:\n{result}"
        self.assertEqual(expected, result, msg=msg)

    def test_if_else(self):
        """Test that the pass recurses in a simple if-else."""
        basis = ["cx", "sx", "u", "rz", "if_else"]
        num_qubits = 4

        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(1)

        test = QuantumCircuit(qr, cr)
        test.h(0)
        test.measure(0, 0)
        test_true = QuantumCircuit(qr, cr)
        test_true.h(qr[0])
        test_true.h(qr[0])
        test_true.h(qr[0])
        test.if_else((0, True), test_true.copy(), None, range(num_qubits), [0])

        expected = QuantumCircuit(qr, cr)
        expected.u(np.pi / 2, 0, -np.pi, 0)
        expected.measure(0, 0)
        expected_true = QuantumCircuit(qr, cr)
        expected_true.u(np.pi / 2, 0, -np.pi, qr[0])
        expected.if_else((0, True), expected_true, None, range(num_qubits), [0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(test)
        self.assertEqual(result, expected)

    def test_nested_control_flow(self):
        """Test that collection recurses into nested control flow."""

        basis = ["cx", "u", "if_else", "for_loop"]
        num_qubits = 4

        qr = QuantumRegister(num_qubits)
        cr = ClassicalRegister(1)

        test = QuantumCircuit(qr, cr)
        test.h(0)
        test.measure(0, 0)
        test_true = QuantumCircuit(qr, cr)
        test_for = QuantumCircuit(qr, cr)
        test_for.h(qr[0])
        test_for.h(qr[0])
        test_for.h(qr[0])
        test_true.for_loop(range(4), body=test_for, qubits=qr, clbits=cr)
        test.if_else((0, True), test_true.copy(), None, range(num_qubits), [0])

        expected = QuantumCircuit(qr, cr)
        expected.u(np.pi / 2, 0, -np.pi, 0)
        expected.measure(0, 0)
        expected_true = QuantumCircuit(qr, cr)
        expected_for = QuantumCircuit(qr, cr)
        expected_for.u(np.pi / 2, 0, -np.pi, qr[0])
        expected_true.for_loop(range(4), body=expected_for, qubits=qr, clbits=cr)
        expected.if_else((0, True), expected_true, None, range(num_qubits), [0])

        passmanager = PassManager()
        passmanager.append(Optimize1qGatesDecomposition(basis))
        result = passmanager.run(test)
        self.assertEqual(result, expected)

    def test_prefer_no_substitution_if_all_ideal(self):
        """Test that gates are not substituted if all our ideal gates in basis."""
        target = Target(num_qubits=1)
        target.add_instruction(HGate(), {(0,): InstructionProperties(error=0)})
        target.add_instruction(
            UGate(Parameter("a"), Parameter("b"), Parameter("c")),
            {(0,): InstructionProperties(error=0)},
        )
        qc = QuantumCircuit(1)
        qc.h(0)
        opt_pass = Optimize1qGatesDecomposition(target)
        res = opt_pass(qc)
        self.assertEqual(res, qc)

    def test_custom_gate(self):
        """Test that pass handles custom single-qubit gates."""

        class CustomGate(Gate):
            """Custom u1 gate."""

            def __init__(self, lam):
                super().__init__("custom_u1", 1, [lam])

            def __array__(self, dtype=None, _copy=None):
                return U1Gate(*self.params).__array__(dtype=dtype)

        qc = QuantumCircuit(1)
        qc.append(CustomGate(0.5), [0])
        result = Optimize1qGatesDecomposition(["cx", "u"])(qc)

        expected = QuantumCircuit(1)
        expected.u(0, 0, 0.5, [0])

        self.assertEqual(result, expected)

    def test_custom_gate_in_target(self):
        """Test reproduce of deadlock from: #16591."""

        class CustomGate(Gate):
            """Custom u1 gate."""

            def __init__(self, lam):
                super().__init__("custom_u1", 1, [lam])

            def __array__(self, dtype=None, _copy=None):
                return U1Gate(*self.params).__array__(dtype=dtype)

        target = Target(num_qubits=2)
        target.add_instruction(CustomGate(Parameter("lam")))
        target.add_instruction(Measure())

        qc = QuantumCircuit(2)
        qc.append(CustomGate(0.5), [0])
        qc.append(CustomGate(0.5), [1])
        qc.measure_all()

        result = Optimize1qGatesDecomposition(target)(qc)
        expected = QuantumCircuit(2)
        expected.append(CustomGate(0.5), [0])
        expected.append(CustomGate(0.5), [1])
        expected.measure_all()
        self.assertEqual(result, expected)

    def test_unitary_gate_row_major(self):
        """
        Test that pass handles unitary single-qubit gates constructed
        using the default row-major ordering.
        """

        qc = QuantumCircuit(1)
        mat = np.asarray(SGate().to_matrix(), dtype=complex)
        qc.append(UnitaryGate(mat), [0])
        result = Optimize1qGatesDecomposition(["cx", "u"])(qc)

        expected = QuantumCircuit(1)
        expected.u(0, 0, np.pi / 2, [0])
        self.assertEqual(result, expected)

    def test_unitary_gate_column_major(self):
        """
        Test that pass handles unitary single-qubit gates constructed
        using the column-major ordering.
        """
        qc = QuantumCircuit(1)
        mat = np.asarray(SGate().to_matrix(), dtype=complex, order="f")
        qc.append(UnitaryGate(mat), [0])
        result = Optimize1qGatesDecomposition(["cx", "u"])(qc)

        expected = QuantumCircuit(1)
        expected.u(0, 0, np.pi / 2, [0])
        self.assertEqual(result, expected)

    def test_target_fixed_angle_1q_gate(self):
        """Test that a fixed angle gate is not used for decomposition."""
        target = Target(num_qubits=1)
        target.add_instruction(UGate(3.14, 0, -3.14), {(0,): None})
        target.add_instruction(SXGate(), {(0,): None})
        target.add_instruction(RZGate(Parameter("t")), {(0,): None})

        opt_pass = Optimize1qGatesDecomposition(target=target)
        qc = QuantumCircuit(1)
        qc.u(1, 2, 3, 0)
        res = opt_pass(qc)
        self.assertNotIn("u", res.count_ops())
        self.assertIn("rz", res.count_ops())
        self.assertIn("sx", res.count_ops())


if __name__ == "__main__":
    unittest.main()
