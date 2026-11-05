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


"""Tests for Clifford class."""

import os
import subprocess
import sys
import textwrap
import unittest
import itertools
import numpy as np
from ddt import ddt

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.random import random_clifford_circuit

from qiskit.circuit.library import (
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CXGate,
    CYGate,
    CZGate,
    DCXGate,
    ECRGate,
    HGate,
    IGate,
    RXGate,
    RYGate,
    RZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    SGate,
    SdgGate,
    SXGate,
    SXdgGate,
    SwapGate,
    XGate,
    XXMinusYYGate,
    XXPlusYYGate,
    YGate,
    ZGate,
    iSwapGate,
    LinearFunction,
    PermutationGate,
    PauliGate,
)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info.operators import Clifford, Operator
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_operation,
    _prepend_operation,
)
from qiskit.synthesis.linear import random_invertible_binary_matrix
from test import QiskitTestCase
from test import combine


@ddt
class TestCliffordGates(QiskitTestCase):
    """Tests for clifford append gate functions."""

    def test_append_1_qubit_gate(self):
        """Tests for append of 1-qubit gates"""

        target_table = {
            "i": np.array([[[True, False], [False, True]]], dtype=bool),
            "id": np.array([[[True, False], [False, True]]], dtype=bool),
            "iden": np.array([[[True, False], [False, True]]], dtype=bool),
            "x": np.array([[[True, False], [False, True]]], dtype=bool),
            "y": np.array([[[True, False], [False, True]]], dtype=bool),
            "z": np.array([[[True, False], [False, True]]], dtype=bool),
            "h": np.array([[[False, True], [True, False]]], dtype=bool),
            "s": np.array([[[True, True], [False, True]]], dtype=bool),
            "sdg": np.array([[[True, True], [False, True]]], dtype=bool),
            "sinv": np.array([[[True, True], [False, True]]], dtype=bool),
            "v": np.array([[[True, True], [True, False]]], dtype=bool),
            "w": np.array([[[False, True], [True, True]]], dtype=bool),
            "sx": np.array([[[True, False], [True, True]]], dtype=bool),
            "sxdg": np.array([[[True, False], [True, True]]], dtype=bool),
        }

        target_phase = {
            "i": np.array([[False, False]], dtype=bool),
            "id": np.array([[False, False]], dtype=bool),
            "iden": np.array([[False, False]], dtype=bool),
            "x": np.array([[False, True]], dtype=bool),
            "y": np.array([[True, True]], dtype=bool),
            "z": np.array([[True, False]], dtype=bool),
            "h": np.array([[False, False]], dtype=bool),
            "s": np.array([[False, False]], dtype=bool),
            "sdg": np.array([[True, False]], dtype=bool),
            "sinv": np.array([[True, False]], dtype=bool),
            "v": np.array([[False, False]], dtype=bool),
            "w": np.array([[False, False]], dtype=bool),
            "sx": np.array([[False, True]], dtype=bool),
            "sxdg": np.array([[False, False]], dtype=bool),
        }

        target_stabilizer = {
            "i": "+Z",
            "id": "+Z",
            "iden": "+Z",
            "x": "-Z",
            "y": "-Z",
            "z": "+Z",
            "h": "+X",
            "s": "+Z",
            "sdg": "+Z",
            "sinv": "+Z",
            "v": "+X",
            "w": "+Y",
            "sx": "-Y",
            "sxdg": "+Y",
        }

        target_destabilizer = {
            "i": "+X",
            "id": "+X",
            "iden": "+X",
            "x": "+X",
            "y": "-X",
            "z": "-X",
            "h": "+Z",
            "s": "+Y",
            "sdg": "-Y",
            "sinv": "-Y",
            "v": "+Y",
            "w": "+Z",
            "sx": "+X",
            "sxdg": "+X",
        }

        for gate_name in (
            "i",
            "id",
            "iden",
            "x",
            "y",
            "z",
            "h",
            "s",
            "sdg",
            "v",
            "w",
            "sx",
            "sxdg",
        ):
            with self.subTest(msg=f"append gate {gate_name}"):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff = _append_operation(cliff, gate_name, [0])
                value_table = cliff.tableau[:, :-1]
                value_phase = cliff.phase
                value_stabilizer = cliff.to_labels(mode="S")
                value_destabilizer = cliff.to_labels(mode="D")
                self.assertTrue(np.all(np.array(value_table == target_table[gate_name])))
                self.assertTrue(np.all(np.array(value_phase == target_phase[gate_name])))
                self.assertTrue(
                    np.all(np.array(value_stabilizer == [target_stabilizer[gate_name]]))
                )
                self.assertTrue(
                    np.all(np.array(value_destabilizer == [target_destabilizer[gate_name]]))
                )

    @combine(hash_seed=["1", "2", "10", "100"])
    def test_random_clifford_circuit_with_same_seed(self, hash_seed):
        random_clifford = random_clifford_circuit(
            num_qubits=10, num_gates=100, gates="all", seed=0
        ).count_ops()

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed

        test_script = textwrap.dedent(
            """
        from qiskit.circuit.random.utils import random_clifford_circuit
        cliff_circuit = random_clifford_circuit(num_qubits=10, num_gates=100, gates="all", seed=0)
        print(cliff_circuit.count_ops(),end="")
        """
        )

        result = subprocess.run(
            [sys.executable, "-c", test_script], env=env, capture_output=True, text=True, check=True
        )

        self.assertEqual(result.stdout, random_clifford.__str__())

    @combine(
        gate=[
            IGate(),
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            SXGate(),
            SXdgGate(),
        ],
        num_qubits=[1, 2, 3],
    )
    def test_append_1_qubit(self, gate, num_qubits):
        """Test _append_operation method for 1-qubit gates"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            for qubit in range(num_qubits):
                circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
                cliff = Clifford(circ)
                cliff = _append_operation(cliff, gate.name, [qubit])
                circ.append(gate, [qubit])
                self.assertEqual(cliff, Clifford(circ))

    @combine(
        gate=[
            IGate(),
            XGate(),
            YGate(),
            ZGate(),
            HGate(),
            SGate(),
            SdgGate(),
            SXGate(),
            SXdgGate(),
        ],
        num_qubits=[1, 2, 3],
    )
    def test_prepend_1_qubit(self, gate, num_qubits):
        """Test _prepend_operation method for 1-qubit gates"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            for qubit in range(num_qubits):
                circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
                cliff = Clifford(circ)
                cliff = _prepend_operation(cliff, gate.name, [qubit])
                circ1 = QuantumCircuit(num_qubits)
                circ1.append(gate, [qubit])
                circ1.append(circ, range(num_qubits))
                self.assertEqual(cliff, Clifford(circ1))

    @combine(
        gate=[CXGate(), CZGate(), CYGate(), SwapGate(), iSwapGate(), ECRGate(), DCXGate()],
        num_qubits=[2, 3],
    )
    def test_append_2_qubits(self, gate, num_qubits):
        """Test _append_operation method for 2-qubit gates"""
        samples = 10
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            for qubits in itertools.combinations(range(num_qubits), 2):
                circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
                cliff = Clifford(circ)
                cliff = _append_operation(cliff, gate.name, qubits)
                circ.append(gate, qubits)
                self.assertEqual(cliff, Clifford(circ))

    @combine(
        gate=[CXGate(), CZGate(), CYGate(), SwapGate(), iSwapGate(), ECRGate(), DCXGate()],
        num_qubits=[2, 3],
    )
    def test_prepend_2_qubits(self, gate, num_qubits):
        """Test _prepend_operation method for 2-qubit gates"""
        samples = 10
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            for qubits in itertools.combinations(range(num_qubits), 2):
                circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
                cliff = Clifford(circ)
                cliff = _prepend_operation(cliff, gate.name, qubits)
                circ1 = QuantumCircuit(num_qubits)
                circ1.append(gate, qubits)
                circ1.append(circ, range(num_qubits))
                self.assertEqual(cliff, Clifford(circ1))

    def test_1_qubit_identity_relations(self):
        """Tests identity relations for 1-qubit gates"""

        for gate_name in ("x", "y", "z", "h"):
            with self.subTest(msg=f"identity for gate {gate_name}"):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_operation(cliff, gate_name, [0])
                cliff = _append_operation(cliff, gate_name, [0])
                self.assertEqual(cliff, cliff1)

        gates = ["s", "s", "v"]
        inv_gates = ["sdg", "sinv", "w"]

        for gate_name, inv_gate in zip(gates, inv_gates):
            with self.subTest(msg=f"identity for gate {gate_name}"):
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_operation(cliff, gate_name, [0])
                cliff = _append_operation(cliff, inv_gate, [0])
                self.assertEqual(cliff, cliff1)

    def test_1_qubit_mult_relations(self):
        """Tests multiplicity relations for 1-qubit gates"""

        rels = [
            "x * y = z",
            "x * z = y",
            "y * z = x",
            "s * s = z",
            "sdg * sdg = z",
            "sinv * sinv = z",
            "sdg * h = v",
            "h * s = w",
        ]

        for rel in rels:
            with self.subTest(msg=f"relation {rel}"):
                split_rel = rel.split()
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_operation(cliff, split_rel[0], [0])
                cliff = _append_operation(cliff, split_rel[2], [0])
                cliff1 = _append_operation(cliff1, split_rel[4], [0])
                self.assertEqual(cliff, cliff1)

    def test_1_qubit_conj_relations(self):
        """Tests conjugation relations for 1-qubit gates"""

        rels = [
            "h * x * h = z",
            "h * y * h = y",
            "s * x * sdg = y",
            "w * x * v = y",
            "w * y * v = z",
            "w * z * v = x",
            "sdg * h * sdg = sx",
            "s * h * s = sxdg",
        ]

        for rel in rels:
            with self.subTest(msg=f"relation {rel}"):
                split_rel = rel.split()
                cliff = Clifford([[1, 0], [0, 1]])
                cliff1 = cliff.copy()
                cliff = _append_operation(cliff, split_rel[0], [0])
                cliff = _append_operation(cliff, split_rel[2], [0])
                cliff = _append_operation(cliff, split_rel[4], [0])
                cliff1 = _append_operation(cliff1, split_rel[6], [0])
                self.assertEqual(cliff, cliff1)

    @combine(gate_name=("cx", "cz", "swap"), qubits=([0, 1], [1, 0]))
    def test_append_2_qubit_gate(self, gate_name, qubits):
        """Tests for append of 2-qubit gate {gate_name} {qubits}."""

        targets_cliffords = {
            "cx [0, 1]": Clifford(
                [
                    [True, True, False, False],
                    [False, True, False, False],
                    [False, False, True, False],
                    [False, False, True, True],
                ]
            ),
            "cx [1, 0]": Clifford(
                [
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, True, True],
                    [False, False, False, True],
                ]
            ),
            "cz [0, 1]": Clifford(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ),
            "cz [1, 0]": Clifford(
                [
                    [True, False, False, True],
                    [False, True, True, False],
                    [False, False, True, False],
                    [False, False, False, True],
                ]
            ),
            "swap [0, 1]": Clifford(
                [
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                ]
            ),
            "swap [1, 0]": Clifford(
                [
                    [False, True, False, False],
                    [True, False, False, False],
                    [False, False, False, True],
                    [False, False, True, False],
                ]
            ),
        }

        gate_qubits = gate_name + " " + str(qubits)
        cliff = _append_operation(Clifford(np.eye(4)), gate_name, qubits)
        target = targets_cliffords[gate_qubits]
        self.assertEqual(target, cliff)

    def test_2_qubit_identity_relations(self):
        """Tests identity relations for 2-qubit gates"""

        for gate_name in ("cx", "cz", "swap"):
            for qubits in ([0, 1], [1, 0]):
                with self.subTest(msg=f"append gate {gate_name} {qubits}"):
                    cliff = Clifford(np.eye(4))
                    cliff1 = cliff.copy()
                    cliff = _append_operation(cliff, gate_name, qubits)
                    cliff = _append_operation(cliff, gate_name, qubits)
                    self.assertEqual(cliff, cliff1)

    def test_2_qubit_relations(self):
        """Tests relations for 2-qubit gates"""

        with self.subTest(msg="relation between cx, h and cz"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "h", [1])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "h", [1])
            cliff = _append_operation(cliff, "cz", [0, 1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and swap"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "cx", [1, 0])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "swap", [0, 1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and x"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "x", [0])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "x", [0])
            cliff = _append_operation(cliff, "x", [1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and z"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "z", [1])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "z", [0])
            cliff = _append_operation(cliff, "z", [1])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and s"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "cx", [1, 0])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "s", [1])
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "cx", [1, 0])
            cliff = _append_operation(cliff, "sdg", [0])
            self.assertEqual(cliff, cliff1)

        with self.subTest(msg="relation between cx and dcx"):
            cliff = Clifford(np.eye(4))
            cliff1 = cliff.copy()
            cliff = _append_operation(cliff, "cx", [0, 1])
            cliff = _append_operation(cliff, "cx", [1, 0])
            cliff1 = _append_operation(cliff1, "dcx", [0, 1])
            self.assertEqual(cliff, cliff1)

    def test_barrier_delay_sim(self):
        """Test barrier and delay instructions can be simulated"""
        target_circ = QuantumCircuit(2)
        target_circ.h(0)
        target_circ.cx(0, 1)
        target = Clifford(target_circ)

        circ = QuantumCircuit(2)
        circ.h(0)
        circ.delay(100, 0)
        circ.barrier([0, 1])
        circ.cx(0, 1)
        value = Clifford(circ)
        self.assertEqual(value, target)

    def test_from_circuit_with_other_clifford(self):
        """Test initialization from circuit containing another clifford."""
        cliff = random_clifford(1, seed=777)
        qc = QuantumCircuit(1)
        qc.append(cliff, [0])
        cliff1 = Clifford(qc)
        self.assertEqual(cliff, cliff1)

    def test_from_circuit_with_multiple_cliffords(self):
        """Test initialization from circuit containing multiple clifford."""
        cliff1 = random_clifford(2, seed=777)
        cliff2 = random_clifford(2, seed=999)

        # Append the two cliffords to circuit and create the clifford from this circuit
        qc1 = QuantumCircuit(3)
        qc1.append(cliff1, [0, 1])
        qc1.append(cliff2, [1, 2])
        expected_cliff1 = Clifford(qc1)

        # Compose the two cliffords directly
        qc2 = QuantumCircuit(3)
        expected_cliff2 = Clifford(qc2)
        expected_cliff2 = Clifford.compose(expected_cliff2, cliff1, qargs=[0, 1], front=False)
        expected_cliff2 = Clifford.compose(expected_cliff2, cliff2, qargs=[1, 2], front=False)
        self.assertEqual(expected_cliff1, expected_cliff2)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_from_linear_function(self, num_qubits):
        """Test initialization from linear function."""
        rng = np.random.default_rng(1234)
        samples = 50
        seeds = rng.integers(100000, size=samples, dtype=np.uint64)
        for seed in seeds:
            mat = random_invertible_binary_matrix(num_qubits, seed=seed)
            lin = LinearFunction(mat)
            cliff = Clifford(lin)
            self.assertTrue(Operator(cliff).equiv(Operator(lin)))

    def test_from_circuit_with_linear_function(self):
        """Test initialization from a quantum circuit that contains a linear function."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        mat = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]
        lin = LinearFunction(mat)
        qc.append(lin, [0, 1, 2, 3])
        qc.h(1)
        cliff = Clifford(qc)
        self.assertTrue(Operator(cliff).equiv(Operator(qc)))

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_from_permutation_gate(self, num_qubits):
        """Test initialization from permutation gate."""
        np.random.seed(1234)
        samples = 50

        for _ in range(samples):
            pat = np.random.permutation(num_qubits)
            perm = PermutationGate(pat)
            cliff = Clifford(perm)
            self.assertTrue(Operator(cliff).equiv(Operator(perm)))

    def test_from_circuit_with_permutation_gate(self):
        """Test initialization from a quantum circuit that contains a permutation gate."""
        qc = QuantumCircuit(5)
        qc.cx(0, 1)
        perm = PermutationGate([2, 1, 0, 3])
        qc.append(perm, [0, 1, 2, 3])
        qc.h(1)
        cliff = Clifford(qc)
        self.assertTrue(Operator(cliff).equiv(Operator(qc)))

    def test_from_circuit_with_all_types(self):
        """Test initialization from circuit containing various Clifford-like objects."""

        # Construct objects that can go onto a Clifford circuit.
        # These include regular clifford gates, linear functions, Pauli gates, other Clifford,
        # and even circuits with other clifford objects.
        linear_function = LinearFunction([[0, 1], [1, 1]])
        pauli_gate = PauliGate("YZ")

        qc_cliff = QuantumCircuit(2)
        qc_cliff.h(0)
        qc_cliff.cx(0, 1)
        cliff = Clifford(qc_cliff)

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.append(random_clifford(1, seed=999), [1])

        # Construct a quantum circuit with these objects and convert it to clifford
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.append(linear_function, [0, 2])
        circuit.cz(0, 1)
        circuit.append(pauli_gate, [2, 1])
        circuit.append(cliff, [0, 1])
        circuit.swap(0, 2)
        circuit.append(qc, [0, 1])

        # Make sure that Clifford can be constructed from such circuit.
        combined_clifford = Clifford(circuit)

        # Additionally, make sure that it produces the correct clifford.
        expected_clifford_dict = {
            "stabilizer": ["-IZX", "+ZYZ", "+XZI"],
            "destabilizer": ["+XZZ", "-XII", "+IXY"],
        }
        expected_clifford = Clifford.from_dict(expected_clifford_dict)
        self.assertEqual(combined_clifford, expected_clifford)


@ddt
class TestCliffordDecomposition(QiskitTestCase):
    """Test Clifford decompositions."""

    @combine(
        gates=[
            ["h", "s"],
            ["h", "s", "i", "x", "y", "z"],
            ["h", "s", "sdg"],
            ["h", "sx", "sxdg"],
            ["s", "sx", "sxdg"],
            ["h", "s", "sdg", "i", "x", "y", "z", "sx", "sxdg"],
        ]
    )
    def test_to_operator_1qubit_gates(self, gates):
        """Test 1-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 10
        seed = 100
        for i in range(samples):
            circ = random_clifford_circuit(1, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(
        gates=[
            ["cx"],
            ["cz"],
            ["cy"],
            ["swap"],
            ["iswap"],
            ["ecr"],
            ["dcx"],
            ["cx", "cz"],
            ["cx", "cz", "cy"],
            ["cx", "swap"],
            ["cz", "swap"],
            ["cx", "cz", "swap"],
            ["cx", "cz", "cy", "swap", "iswap", "ecr", "dcx"],
        ]
    )
    def test_to_operator_2qubit_gates(self, gates):
        """Test 2-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 10
        seed = 200
        for i in range(samples):
            circ = random_clifford_circuit(2, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(
        gates=[["h", "s", "cx"], ["h", "s", "cz"], ["h", "s", "swap"], "all"], num_qubits=[2, 3, 4]
    )
    def test_to_operator_nqubit_gates(self, gates, num_qubits):
        """Test {num_qubits}-qubit circuit with gates {gates}"""
        samples = 10
        num_gates = 20
        seed = 300
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).to_operator()
            target = Operator(circ)
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_to_matrix(self, num_qubits):
        """Test to_matrix method"""
        samples = 10
        num_gates = 10
        seed = 333
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            mat = Clifford(circ).to_matrix()
            self.assertIsInstance(mat, np.ndarray)
            self.assertEqual(mat.shape, 2 * (2**num_qubits,))
            value = Operator(mat)
            target = Operator(circ)
            self.assertTrue(value.equiv(target))

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_to_circuit(self, num_qubits):
        """Test to_circuit method"""
        samples = 10
        num_gates = 10
        seed = 700
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            target = Clifford(circ)
            decomp = target.to_circuit()
            self.assertIsInstance(decomp, QuantumCircuit)
            self.assertEqual(decomp.num_qubits, circ.num_qubits)
            # Convert back to clifford and check it is the same
            self.assertEqual(Clifford(decomp), target)

    def test_to_circuit_manual(self):
        """Test a manual comparison to a known circuit.

        This also tests whether the resulting Clifford circuit has quantum registers, thereby
        regression testing #13041.
        """
        # this is set to a circuit that remains the same under Clifford reconstruction
        circuit = QuantumCircuit(2)
        circuit.z(0)
        circuit.h(0)
        circuit.cx(0, 1)

        cliff = Clifford(circuit)
        reconstructed = cliff.to_circuit()

        self.assertEqual(circuit, reconstructed)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_to_instruction(self, num_qubits):
        """Test to_instruction method"""
        samples = 10
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            target = Clifford(circ)
            decomp = target.to_instruction()
            self.assertIsInstance(decomp, Gate)
            self.assertEqual(decomp.num_qubits, circ.num_qubits)
            # Convert back to clifford and check it is the same
            self.assertEqual(Clifford(decomp), target)


@ddt
class TestCliffordOperators(QiskitTestCase):
    """Test Clifford operator class methods."""

    @combine(num_qubits=[1, 2, 3])
    def test_is_unitary(self, num_qubits):
        """Test is_unitary method"""
        samples = 10
        num_gates = 10
        seed = 700
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).is_unitary()
            self.assertTrue(value)
        # tests a false clifford
        cliff = Clifford([[0, 0], [0, 1]], validate=False)
        value = cliff.is_unitary()
        self.assertFalse(value)

    @combine(num_qubits=[1, 2, 3])
    def test_conjugate(self, num_qubits):
        """Test conjugate method"""
        samples = 10
        num_gates = 10
        seed = 400
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).conjugate().to_operator()
            target = Operator(circ).conjugate()
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_transpose(self, num_qubits):
        """Test transpose method"""
        samples = 10
        num_gates = 1
        seed = 500
        gates = "all"
        for i in range(samples):
            circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            value = Clifford(circ).transpose().to_operator()
            target = Operator(circ).transpose()
            self.assertTrue(target.equiv(value))

    @combine(num_qubits=[1, 2, 3])
    def test_compose_method(self, num_qubits):
        """Test compose method"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.compose(cliff2)
            target = Clifford(circ1.compose(circ2))
            self.assertEqual(target, value)
            value_circ_composed = cliff1.compose(circ2)
            self.assertEqual(target, value_circ_composed)

    @combine(num_qubits=[1, 2, 3])
    def test_dot_method(self, num_qubits):
        """Test dot method"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.dot(cliff2)
            target = Clifford(circ2.compose(circ1))
            self.assertEqual(target, value)
            value_circ_composed = cliff1.dot(circ2)
            self.assertEqual(target, value_circ_composed)

    @combine(num_qubits_1=[1, 2, 3], num_qubits_2=[1, 2, 3])
    def test_tensor_method(self, num_qubits_1, num_qubits_2):
        """Test tensor method"""
        samples = 5
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.tensor(cliff2)
            circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
            circ.append(circ2, range(num_qubits_2))
            circ.append(circ1, range(num_qubits_2, num_qubits_1 + num_qubits_2))
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[1, 2, 3], num_qubits_2=[1, 2, 3])
    def test_expand_method(self, num_qubits_1, num_qubits_2):
        """Test expand method"""
        samples = 5
        num_gates = 10
        seed = 800
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            cliff1 = Clifford(circ1)
            cliff2 = Clifford(circ2)
            value = cliff1.expand(cliff2)
            circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
            circ.append(circ1, range(num_qubits_1))
            circ.append(circ2, range(num_qubits_1, num_qubits_1 + num_qubits_2))
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_compose_subsystem(self, num_qubits_1, num_qubits_2):
        """Test compose method of subsystems"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            circ = circ1.copy()
            circ.append(circ2.to_instruction(), qargs)
            value = Clifford(circ1).compose(Clifford(circ2), qargs)
            target = Clifford(circ)
            self.assertEqual(target, value)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_dot_subsystem(self, num_qubits_1, num_qubits_2):
        """Test dot method of subsystems"""
        samples = 10
        num_gates = 10
        seed = 600
        gates = "all"
        for i in range(samples):
            circ1 = random_clifford_circuit(num_qubits_1, num_gates, gates=gates, seed=seed + i)
            circ2 = random_clifford_circuit(
                num_qubits_2, num_gates, gates=gates, seed=seed + samples + i
            )
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            circ = QuantumCircuit(num_qubits_1)
            circ.append(circ2.to_instruction(), qargs)
            circ.append(circ1.to_instruction(), range(num_qubits_1))
            value = Clifford(circ1).dot(Clifford(circ2), qargs)
            target = Clifford(circ)
            self.assertEqual(target, value)

    def test_to_dict(self):
        """Test to_dict method"""

        with self.subTest(msg="Identity"):
            cliff = Clifford(np.eye(8))
            value = cliff.to_dict()

            keys_value = set(value.keys())
            keys_target = {"destabilizer", "stabilizer"}
            self.assertEqual(keys_value, keys_target)

            stabilizer_value = set(value["stabilizer"])
            stabilizer_target = {"+IIIZ", "+IIZI", "+IZII", "+ZIII"}
            self.assertEqual(stabilizer_value, stabilizer_target)

            destabilizer_value = set(value["destabilizer"])
            destabilizer_target = {"+IIIX", "+IIXI", "+IXII", "+XIII"}
            self.assertEqual(destabilizer_value, destabilizer_target)

        with self.subTest(msg="bell"):
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            cliff = Clifford(qc)
            value = cliff.to_dict()

            keys_value = set(value.keys())
            keys_target = {"destabilizer", "stabilizer"}
            self.assertEqual(keys_value, keys_target)

            stabilizer_value = set(value["stabilizer"])
            stabilizer_target = {"+XX", "+ZZ"}
            self.assertEqual(stabilizer_value, stabilizer_target)

            destabilizer_value = set(value["destabilizer"])
            destabilizer_target = {"+IZ", "+XI"}
            self.assertEqual(destabilizer_value, destabilizer_target)

    def test_from_dict(self):
        """Test from_dict method"""

        with self.subTest(msg="test raises not unitary"):
            cliff_dict = {"stabilizer": ["+XX", "+ZZ"], "destabilizer": ["+IZ", "+IY"]}
            self.assertRaises(QiskitError, Clifford.from_dict, cliff_dict)

        with self.subTest(msg="test raises wrong shape"):
            cliff_dict = {
                "stabilizer": ["+XX", "+ZZ", "+YY"],
                "destabilizer": ["+IZ", "+XI", "+IY"],
            }
            self.assertRaises(QiskitError, Clifford.from_dict, cliff_dict)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_dict_round_trip(self, num_qubits):
        """Test round trip conversion to and from dict"""
        num_gates = 10
        seed = 655
        gates = "all"
        circ = random_clifford_circuit(num_qubits, num_gates, gates=gates, seed=seed + num_qubits)
        target = Clifford(circ)
        value = Clifford.from_dict(target.to_dict())
        self.assertEqual(value, target)

    def test_from_label(self):
        """Test from_label method"""
        label = "IXYZHS"
        CI = Clifford(IGate())
        CX = Clifford(XGate())
        CY = Clifford(YGate())
        CZ = Clifford(ZGate())
        CH = Clifford(HGate())
        CS = Clifford(SGate())
        target = CI.tensor(CX).tensor(CY).tensor(CZ).tensor(CH).tensor(CS)
        self.assertEqual(Clifford.from_label(label), target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_instruction_name(self, num_qubits):
        """Test to verify the correct clifford name is maintained
        after converting to instruction"""
        clifford = random_clifford(num_qubits, seed=777)
        self.assertEqual(clifford.to_instruction().name, str(clifford))

    def test_visualize_does_not_throw_error(self):
        """Test to verify that drawing Clifford does not throw an error"""
        # An error may be thrown if visualization code calls op.condition instead
        # of getattr(op, "condition", None)
        clifford = random_clifford(3, seed=0)
        _ = str(clifford)
        _ = repr(clifford)

    @combine(num_qubits=[1, 2, 3, 4])
    def test_from_matrix_round_trip(self, num_qubits):
        """Test round trip conversion to and from matrix"""
        for i in range(10):
            expected = random_clifford(num_qubits, seed=42 + i)
            actual = Clifford.from_matrix(expected.to_matrix())
            self.assertEqual(expected, actual)

    def test_from_non_clifford_diagonal_operator(self):
        """Test if failing with non-clifford diagonal operator.
        See https://github.com/Qiskit/qiskit/issues/10903"""
        with self.assertRaises(QiskitError):
            Clifford.from_operator(Operator(RZZGate(0.2)))

    @combine(num_qubits=[1, 2, 3, 4])
    def test_from_operator_round_trip(self, num_qubits):
        """Test round trip conversion to and from operator"""
        for i in range(10):
            expected = random_clifford(num_qubits, seed=777 + i)
            actual = Clifford.from_operator(expected.to_operator())
            self.assertEqual(expected, actual)

    @combine(
        gate=[
            RXGate(theta=np.pi / 2),
            RYGate(theta=np.pi / 2),
            RZGate(phi=np.pi / 2),
            CPhaseGate(theta=np.pi),
            CRXGate(theta=np.pi),
            CRYGate(theta=np.pi),
            CRZGate(theta=np.pi),
            CXGate(),
            CYGate(),
            CZGate(),
            ECRGate(),
            RXXGate(theta=np.pi / 2),
            RYYGate(theta=np.pi / 2),
            RZZGate(theta=np.pi / 2),
            RZXGate(theta=np.pi / 2),
            SwapGate(),
            iSwapGate(),
            XXMinusYYGate(theta=np.pi),
            XXPlusYYGate(theta=-np.pi),
        ]
    )
    def test_create_from_gates(self, gate):
        """Test if matrix of Clifford created from gate equals the gate matrix up to global phase"""
        self.assertTrue(
            matrix_equal(Clifford(gate).to_matrix(), gate.to_matrix(), ignore_phase=True)
        )


if __name__ == "__main__":
    unittest.main()
