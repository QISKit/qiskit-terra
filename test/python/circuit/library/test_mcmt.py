# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of multi-controlled multi-target circuits."""

import unittest
from functools import reduce
from itertools import repeat
from ddt import ddt, data, unpack
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from qiskit.circuit.library import (
    MCMT,
    MCMTVChain,
    CHGate,
    XGate,
    ZGate,
    RYGate,
    CXGate,
    CZGate,
    MCMTGate,
    GlobalPhaseGate,
    SwapGate,
)
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.states import state_fidelity
from qiskit.quantum_info.operators.operator_utils import _equal_with_ancillas
from qiskit.transpiler.passes import HighLevelSynthesis, HLSConfig
from qiskit.synthesis.multi_controlled import synth_mcmt_vchain, synth_mcmt_xgate
from qiskit.quantum_info import Operator
from test import QiskitTestCase, combine


@ddt
class TestMCMT(QiskitTestCase):
    """Test the multi-controlled multi-target circuit."""

    @data(MCMT, MCMTVChain)
    def test_mcmt_as_normal_control(self, mcmt_class):
        """Test that the MCMT can act as normal control gate."""
        qc = QuantumCircuit(2)
        with self.assertWarns(DeprecationWarning):
            mcmt = mcmt_class(gate=CHGate(), num_ctrl_qubits=1, num_target_qubits=1)
        qc = qc.compose(mcmt, [0, 1])

        ref = QuantumCircuit(2)
        ref.ch(0, 1)

        self.assertEqual(len(qc.data), 1)

        basis = ["u", "cx"]
        self.assertEqual(transpile(qc, basis_gates=basis), transpile(ref, basis_gates=basis))

    def test_missing_qubits(self):
        """Test that an error is raised if qubits are missing."""
        with self.subTest(msg="no control qubits"):
            with self.assertWarns(DeprecationWarning):
                with self.assertRaises(AttributeError):
                    _ = MCMT(XGate(), num_ctrl_qubits=0, num_target_qubits=1)

        with self.subTest(msg="no target qubits"):
            with self.assertWarns(DeprecationWarning):
                with self.assertRaises(AttributeError):
                    _ = MCMT(ZGate(), num_ctrl_qubits=4, num_target_qubits=0)

    def test_different_gate_types(self):
        """Test the different supported input types for the target gate."""
        x_circ = QuantumCircuit(1)
        x_circ.x(0)
        for input_gate in [x_circ, QuantumCircuit.cx, QuantumCircuit.x, "cx", "x", CXGate()]:
            with self.subTest(input_gate=input_gate):
                with self.assertWarns(DeprecationWarning):
                    mcmt = MCMT(input_gate, 2, 2)
                if isinstance(input_gate, QuantumCircuit):
                    self.assertEqual(mcmt.gate.definition[0].operation, XGate())
                    self.assertEqual(len(mcmt.gate.definition), 1)
                else:
                    self.assertEqual(mcmt.gate, XGate())

    def test_mcmt_v_chain_ancilla_test(self):
        """Test too few and too many ancillas for the MCMT V-chain mode."""
        with self.subTest(msg="insufficient number of auxiliary qubits on gate"):
            qc = QuantumCircuit(5)
            with self.assertWarns(DeprecationWarning):
                mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, range(5))

        with self.subTest(msg="too many auxiliary qubits on gate"):
            qc = QuantumCircuit(9)
            with self.assertWarns(DeprecationWarning):
                mcmt = MCMTVChain(ZGate(), 3, 1)
            with self.assertRaises(QiskitError):
                qc.append(mcmt, range(9))

    @data(
        [CZGate(), 1, 1],
        [CHGate(), 1, 1],
        [CZGate(), 3, 3],
        [CHGate(), 3, 3],
        [CZGate(), 1, 5],
        [CHGate(), 1, 5],
        [CZGate(), 5, 1],
        [CHGate(), 5, 1],
    )
    @unpack
    def test_mcmt_v_chain_simulation(self, cgate, num_controls, num_targets):
        """Test the MCMT V-chain implementation test on a simulation."""
        controls = QuantumRegister(num_controls)
        targets = QuantumRegister(num_targets)

        subsets = [tuple(range(i)) for i in range(num_controls + 1)]
        for subset in subsets:
            qc = QuantumCircuit(targets, controls)
            # Initialize all targets to 1, just to be sure that
            # the generic gate has some effect (f.e. Z gate has no effect
            # on a 0 state)
            qc.x(targets)

            num_ancillas = max(0, num_controls - 1)

            if num_ancillas > 0:
                ancillas = QuantumRegister(num_ancillas)
                qc.add_register(ancillas)
                qubits = controls[:] + targets[:] + ancillas[:]
            else:
                qubits = controls[:] + targets[:]

            for i in subset:
                qc.x(controls[i])

            with self.assertWarns(DeprecationWarning):
                mcmt = MCMTVChain(cgate, num_controls, num_targets)
            qc.compose(mcmt, qubits, inplace=True)

            for i in subset:
                qc.x(controls[i])

            vec = Statevector.from_label("0" * qc.num_qubits).evolve(qc)

            # target register is initially |11...1>, with length equal to 2**(n_targets)
            vec_exp = np.array([0] * (2 ** (num_targets) - 1) + [1])

            if isinstance(cgate, CZGate):
                # Z gate flips the last qubit only if it's applied an odd number of times
                if len(subset) == num_controls and (num_controls % 2) == 1:
                    vec_exp[-1] = -1
            elif isinstance(cgate, CHGate):
                # if all the control qubits have been activated,
                # we repeatedly apply the kronecker product of the Hadamard
                # with itself and then multiply the results for the original
                # state of the target qubits
                if len(subset) == num_controls:
                    h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
                    h_tot = np.array([1])
                    for _ in range(num_targets):
                        h_tot = np.kron(h_tot, h_i)
                    vec_exp = np.dot(h_tot, vec_exp)
            else:
                raise ValueError(f"Test not implement for gate: {cgate}")

            # append the remaining part of the state
            vec_exp = np.concatenate(
                (vec_exp, [0] * (2 ** (num_controls + num_ancillas + num_targets) - vec_exp.size))
            )
            f_i = state_fidelity(vec, vec_exp)
            self.assertAlmostEqual(f_i, 1)

    def test_default_plugin(self):
        """Test the default behavior of the plugin."""
        num_controls = 5
        num_target = 2
        num_vchain_ancillas = num_controls - 1

        gate = ZGate()  # Anything other than XGate as XGate has special handling
        mcmt = MCMTGate(gate, num_controls, num_target)

        # make sure MCX-synthesis does not use ancilla qubits
        config = HLSConfig(mcx=["noaux_v24"])
        hls = HighLevelSynthesis(hls_config=config)

        # test a decomposition without sufficient ancillas for MCMT V-chain
        with self.subTest(msg="insufficient auxiliaries"):
            circuit = QuantumCircuit(num_controls + num_target + num_vchain_ancillas - 1)
            circuit.append(mcmt, range(mcmt.num_qubits))

            synthesized = hls(circuit)
            num_idle = len(list(circuit_to_dag(synthesized).idle_wires()))

            self.assertEqual(num_idle, num_vchain_ancillas - 1)

        # test with enough auxiliary qubits available
        with self.subTest(msg="enough auxiliaries"):
            circuit = QuantumCircuit(num_controls + num_target + num_vchain_ancillas)
            circuit.append(mcmt, range(mcmt.num_qubits))

            synthesized = hls(circuit)
            num_idle = len(list(circuit_to_dag(synthesized).idle_wires()))

            self.assertEqual(num_idle, 0)

    def test_explicit_plugin(self):
        """Test explicitly setting the plugin."""
        num_controls = 5
        num_target = 2
        num_vchain_ancillas = num_controls - 1

        gate = XGate()
        mcmt = MCMTGate(gate, num_controls, num_target)

        circuit = QuantumCircuit(num_controls + num_target + num_vchain_ancillas)
        circuit.append(mcmt, range(mcmt.num_qubits))

        # test a decomposition without sufficient ancillas for MCMT V-chain
        with self.subTest(msg="force default decomposition"):
            config = HLSConfig(mcmt=["noaux"], mcx=["noaux_v24"])

            synthesized = transpile(circuit, hls_config=config)
            num_idle = len(list(circuit_to_dag(synthesized).idle_wires()))

            self.assertEqual(num_idle, num_vchain_ancillas)

        # test with enough auxiliary qubits available
        with self.subTest(msg="enough auxiliaries"):
            config = HLSConfig(mcmt=["vchain"])

            synthesized = transpile(circuit, hls_config=config)
            num_idle = len(list(circuit_to_dag(synthesized).idle_wires()))

            self.assertEqual(num_idle, 0)

    def test_vchain_open_control(self):
        """Test the V-chain with open controls."""
        num_ctrl_qubits = 2
        ctrl_state = None
        gate = XGate()

        synthesized = synth_mcmt_vchain(
            gate, num_ctrl_qubits, num_target_qubits=1, ctrl_state=ctrl_state
        )
        result = Operator(synthesized)

        # Build the expected matrix by adding padding with the identity operator.
        # This could be potentially moved into the _equal_with_ancillas util.
        action = _compute_control_matrix(gate.to_matrix(), num_ctrl_qubits, ctrl_state)
        pad = np.eye(2 ** (num_ctrl_qubits - 1))
        expected = Operator(np.kron(pad, action))

        self.assertTrue(
            _equal_with_ancillas(
                result,
                expected,
                ancilla_qubits=list(range(num_ctrl_qubits + 1, synthesized.num_qubits)),
                ignore_phase=True,
            )
        )

    @combine(num_ctrl=range(1, 3), num_targ=range(2, 4))
    def test_mcmt_x_gate(self, num_ctrl, num_targ):
        """Test the MCMT X gate synthesis."""
        ctrl_state = None

        synthesized = synth_mcmt_xgate(num_ctrl, num_targ, ctrl_state)
        result = Operator(synthesized).data

        x_gate_matrix = XGate().to_matrix()
        target_gate_matrix = reduce(np.kron, repeat(x_gate_matrix, num_targ))

        expected = _compute_control_matrix(target_gate_matrix, num_ctrl, ctrl_state)
        self.assertTrue(np.allclose(result, expected))

    @combine(num_ctrl=range(1, 6), num_targ=range(2, 6))
    def test_mcmt_x_gate_counts(self, num_ctrl, num_targ):
        """Test MCMT gate uses `synth_mcmt_x` for X gate synthesis."""
        mcmt_x_gate = MCMTGate(XGate(), num_ctrl_qubits=num_ctrl, num_target_qubits=num_targ)
        qc = QuantumCircuit(num_ctrl + num_targ)
        qc.append(mcmt_x_gate, range(num_ctrl + num_targ))

        qc_transpiled = transpile(qc, basis_gates=["u", "cx"], qubits_initially_zero=False)
        expected_cx_count = 12 * num_ctrl + 2 * (num_targ - 1)  # 1 MCX + 2(num_targ - 1) CX gates
        self.assertLessEqual(qc_transpiled.count_ops().get("cx", 0), expected_cx_count)

    def test_invalid_base_gate_width(self):
        """Test only 1-qubit base gates are accepted."""
        for gate in [GlobalPhaseGate(0.2), SwapGate()]:
            with self.subTest(gate=gate):
                with self.assertRaises(ValueError):
                    _ = MCMTGate(gate, 10, 2)

    def test_invalid_base_gate_width_synthfun(self):
        """Test only 1-qubit base gates are accepted."""
        for gate in [GlobalPhaseGate(0.2), SwapGate()]:
            with self.subTest(gate=gate):
                with self.assertRaises(ValueError):
                    _ = synth_mcmt_vchain(gate, 10, 2)

    def test_gate_with_parameters_vchain(self):
        """Test a gate with parameters as base gate."""
        theta = Parameter("th")
        gate = RYGate(theta)
        num_target = 3
        circuit = synth_mcmt_vchain(gate, num_ctrl_qubits=10, num_target_qubits=num_target)

        self.assertEqual(circuit.count_ops().get("cry", 0), num_target)
        self.assertEqual(circuit.num_parameters, 1)
        self.assertEqual(circuit.parameters[0], theta)

    def test_mcmt_circuit_as_gate(self):
        """Test the MCMT plugin is only triggered for the gate, not the same-named circuit.

        Regression test of #13563.
        """
        circuit = QuantumCircuit(2)
        gate = RYGate(0.1)
        with self.assertWarns(DeprecationWarning):
            mcmt = MCMT(gate=gate, num_ctrl_qubits=1, num_target_qubits=1)
        circuit.append(mcmt, circuit.qubits)  # append the MCMT circuit as gate called "MCMT"

        transpiled = transpile(circuit, basis_gates=["u", "cx"])
        self.assertEqual(Operator(transpiled), Operator(gate.control(1, annotated=False)))


if __name__ == "__main__":
    unittest.main()
