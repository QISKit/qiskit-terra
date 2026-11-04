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

"""Test library of Pauli feature map circuits."""

import unittest
from test import combine

import numpy as np
from ddt import ddt, data, unpack

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import (
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
    HGate,
    pauli_feature_map,
    z_feature_map,
    zz_feature_map,
)
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from test import QiskitTestCase


@ddt
class TestDataPreparation(QiskitTestCase):
    """Test the data encoding circuits."""

    def test_pauli_empty(self):
        """Test instantiating an empty Pauli expansion."""
        with self.assertWarns(DeprecationWarning):
            encoding = PauliFeatureMap()

        with self.subTest(msg="equal to empty circuit"):
            self.assertTrue(Operator(encoding).equiv(QuantumCircuit()))

        with self.subTest(msg="rotation blocks is H gate"):
            self.assertEqual(len(encoding.rotation_blocks), 1)
            self.assertIsInstance(encoding.rotation_blocks[0].data[0].operation, HGate)

    @data((2, 3, ["X", "YY"]), (5, 2, ["ZZZXZ", "XZ"]))
    @unpack
    def test_num_parameters(self, num_qubits, reps, pauli_strings):
        """Test the number of parameters equals the number of qubits, independent of reps."""
        with self.assertWarns(DeprecationWarning):
            encoding = PauliFeatureMap(num_qubits, paulis=pauli_strings, reps=reps)
        self.assertEqual(encoding.num_parameters, num_qubits)
        self.assertEqual(encoding.num_parameters_settable, num_qubits)

    def test_pauli_evolution(self):
        """Test the generation of Pauli blocks."""
        with self.assertWarns(DeprecationWarning):
            encoding = PauliFeatureMap()
        time = 1.4
        with self.subTest(pauli_string="ZZ"):
            evo = QuantumCircuit(2)
            evo.cx(0, 1)
            evo.p(2 * time, 1)
            evo.cx(0, 1)

            pauli = encoding.pauli_evolution("ZZ", time)
            self.assertTrue(Operator(pauli).equiv(evo))

        with self.subTest(pauli_string="XYZ"):
            # q_0: ────────■────────────────────────■──────────
            #      ┌────┐┌─┴─┐                    ┌─┴─┐┌──────┐
            # q_1: ┤ √X ├┤ X ├──■──────────────■──┤ X ├┤ √Xdg ├
            #      └┬───┤└───┘┌─┴─┐┌────────┐┌─┴─┐├───┤└──────┘
            # q_2: ─┤ H ├─────┤ X ├┤ P(2.8) ├┤ X ├┤ H ├────────
            #       └───┘     └───┘└────────┘└───┘└───┘
            evo = QuantumCircuit(3)
            # X on the most-significant, bottom qubit, Z on the top
            evo.h(2)
            evo.sx(1)
            evo.cx(0, 1)
            evo.cx(1, 2)
            evo.p(2 * time, 2)
            evo.cx(1, 2)
            evo.cx(0, 1)
            evo.sxdg(1)
            evo.h(2)

            pauli = encoding.pauli_evolution("XYZ", time)
            self.assertTrue(Operator(pauli).equiv(evo))

        with self.subTest(pauli_string="I"):
            evo = QuantumCircuit(1)
            pauli = encoding.pauli_evolution("I", time)
            self.assertTrue(Operator(pauli).equiv(evo))

    def test_first_order_circuit(self):
        """Test a first order expansion circuit."""
        times = [0.2, 1, np.pi, -1.2]
        with self.assertWarns(DeprecationWarning):
            encoding = ZFeatureMap(4, reps=3).assign_parameters(times)

        #      ┌───┐ ┌────────┐┌───┐ ┌────────┐┌───┐ ┌────────┐
        # q_0: ┤ H ├─┤ P(0.4) ├┤ H ├─┤ P(0.4) ├┤ H ├─┤ P(0.4) ├
        #      ├───┤ └┬──────┬┘├───┤ └┬──────┬┘├───┤ └┬──────┬┘
        # q_1: ┤ H ├──┤ P(2) ├─┤ H ├──┤ P(2) ├─┤ H ├──┤ P(2) ├─
        #      ├───┤ ┌┴──────┤ ├───┤ ┌┴──────┤ ├───┤ ┌┴──────┤
        # q_2: ┤ H ├─┤ P(2π) ├─┤ H ├─┤ P(2π) ├─┤ H ├─┤ P(2π) ├─
        #      ├───┤┌┴───────┴┐├───┤┌┴───────┴┐├───┤┌┴───────┴┐
        # q_3: ┤ H ├┤ P(-2.4) ├┤ H ├┤ P(-2.4) ├┤ H ├┤ P(-2.4) ├
        #      └───┘└─────────┘└───┘└─────────┘└───┘└─────────┘
        ref = QuantumCircuit(4)
        for _ in range(3):
            ref.h([0, 1, 2, 3])
            for i in range(4):
                ref.p(2 * times[i], i)

        self.assertTrue(Operator(encoding).equiv(ref))

    def test_second_order_circuit(self):
        """Test a second order expansion circuit."""
        times = [0.2, 1, np.pi]
        with self.assertWarns(DeprecationWarning):
            encoding = ZZFeatureMap(3, reps=2).assign_parameters(times)

        def zz_evolution(circuit, qubit1, qubit2):
            time = (np.pi - times[qubit1]) * (np.pi - times[qubit2])
            circuit.cx(qubit1, qubit2)
            circuit.p(2 * time, qubit2)
            circuit.cx(qubit1, qubit2)

        #      ┌───┐┌────────┐                                         ┌───┐┌────────┐»
        # q_0: ┤ H ├┤ P(0.4) ├──■─────────────────■────■────────────■──┤ H ├┤ P(0.4) ├»
        #      ├───┤└┬──────┬┘┌─┴─┐┌───────────┐┌─┴─┐  │            │  └───┘└────────┘»
        # q_1: ┤ H ├─┤ P(2) ├─┤ X ├┤ P(12.599) ├┤ X ├──┼────────────┼────■────────────»
        #      ├───┤┌┴──────┤ └───┘└───────────┘└───┘┌─┴─┐┌──────┐┌─┴─┐┌─┴─┐ ┌──────┐ »
        # q_2: ┤ H ├┤ P(2π) ├────────────────────────┤ X ├┤ P(0) ├┤ X ├┤ X ├─┤ P(0) ├─»
        #      └───┘└───────┘                        └───┘└──────┘└───┘└───┘ └──────┘ »
        # «                                                                              »
        # «q_0: ─────────────────────■─────────────────■────■────────────■───────────────»
        # «          ┌───┐ ┌──────┐┌─┴─┐┌───────────┐┌─┴─┐  │            │               »
        # «q_1: ──■──┤ H ├─┤ P(2) ├┤ X ├┤ P(12.599) ├┤ X ├──┼────────────┼────■──────────»
        # «     ┌─┴─┐├───┤┌┴──────┤└───┘└───────────┘└───┘┌─┴─┐┌──────┐┌─┴─┐┌─┴─┐┌──────┐»
        # «q_2: ┤ X ├┤ H ├┤ P(2π) ├───────────────────────┤ X ├┤ P(0) ├┤ X ├┤ X ├┤ P(0) ├»
        # «     └───┘└───┘└───────┘                       └───┘└──────┘└───┘└───┘└──────┘»
        # «
        # «q_0: ─────
        # «
        # «q_1: ──■──
        # «     ┌─┴─┐
        # «q_2: ┤ X ├
        # «     └───┘
        ref = QuantumCircuit(3)
        for _ in range(2):
            ref.h([0, 1, 2])
            for i in range(3):
                ref.p(2 * times[i], i)
            zz_evolution(ref, 0, 1)
            zz_evolution(ref, 0, 2)
            zz_evolution(ref, 1, 2)

        self.assertTrue(Operator(encoding).equiv(ref))

    @combine(entanglement=["linear", "reverse_linear", "pairwise"])
    def test_zz_entanglement(self, entanglement):
        """Test the ZZ feature map works with pairwise, linear and reverse_linear entanglement."""
        num_qubits = 5
        with self.assertWarns(DeprecationWarning):
            encoding = ZZFeatureMap(num_qubits, entanglement=entanglement, reps=1)
        ops = encoding.decompose().count_ops()
        expected_ops = {"h": num_qubits, "p": 2 * num_qubits - 1, "cx": 2 * (num_qubits - 1)}
        self.assertEqual(ops, expected_ops)

    def test_pauli_alpha(self):
        """Test  Pauli rotation factor (getter, setter)."""
        with self.assertWarns(DeprecationWarning):
            encoding = PauliFeatureMap()
        self.assertEqual(encoding.alpha, 2.0)
        encoding.alpha = 1.4
        self.assertEqual(encoding.alpha, 1.4)

    def test_zzfeaturemap_raises_if_too_small(self):
        """Test the ``ZZFeatureMap`` raises an error if the number of qubits is smaller than 2."""
        with self.assertRaises(ValueError):
            with self.assertWarns(DeprecationWarning):
                _ = ZZFeatureMap(1)

    def test_parameter_prefix(self):
        """Test the Parameter prefix"""
        with self.assertWarns(DeprecationWarning):
            encoding_pauli = PauliFeatureMap(
                feature_dimension=2, reps=2, paulis=["ZY"], parameter_prefix="p"
            )
        with self.assertWarns(DeprecationWarning):
            encoding_z = ZFeatureMap(feature_dimension=2, reps=2, parameter_prefix="q")
        with self.assertWarns(DeprecationWarning):
            encoding_zz = ZZFeatureMap(feature_dimension=2, reps=2, parameter_prefix="r")

        x = ParameterVector("x", 2)
        y = Parameter("y")

        self.assertEqual(
            str(encoding_pauli.parameters),
            "ParameterView([ParameterVectorElement(p[0]), ParameterVectorElement(p[1])])",
        )
        self.assertEqual(
            str(encoding_z.parameters),
            "ParameterView([ParameterVectorElement(q[0]), ParameterVectorElement(q[1])])",
        )
        self.assertEqual(
            str(encoding_zz.parameters),
            "ParameterView([ParameterVectorElement(r[0]), ParameterVectorElement(r[1])])",
        )

        encoding_pauli_param_x = encoding_pauli.assign_parameters(x)
        encoding_z_param_x = encoding_z.assign_parameters(x)
        encoding_zz_param_x = encoding_zz.assign_parameters(x)

        self.assertEqual(
            str(encoding_pauli_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )
        self.assertEqual(
            str(encoding_z_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )
        self.assertEqual(
            str(encoding_zz_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )

        encoding_pauli_param_y = encoding_pauli.assign_parameters([1, y])
        encoding_z_param_y = encoding_z.assign_parameters([1, y])
        encoding_zz_param_y = encoding_zz.assign_parameters([1, y])

        self.assertEqual(str(encoding_pauli_param_y.parameters), "ParameterView([Parameter(y)])")
        self.assertEqual(str(encoding_z_param_y.parameters), "ParameterView([Parameter(y)])")
        self.assertEqual(str(encoding_zz_param_y.parameters), "ParameterView([Parameter(y)])")

    def test_entanglement_as_dictionary(self):
        """Test whether PauliFeatureMap accepts entanglement as a dictionary and generates
        correct feature map circuit"""
        n_qubits = 3
        entanglement = {
            1: [(0,), (2,)],
            2: [(0, 1), (1, 2)],
            3: [(0, 1, 2)],
        }
        params = [np.pi / 4, np.pi / 2, np.pi]

        def z_block(circuit, q1):
            circuit.p(2 * params[q1], q1)

        def zz_block(circuit, q1, q2):
            param = (np.pi - params[q1]) * (np.pi - params[q2])
            circuit.cx(q1, q2)
            circuit.p(2 * param, q2)
            circuit.cx(q1, q2)

        def zzz_block(circuit, q1, q2, q3):
            param = (np.pi - params[q1]) * (np.pi - params[q2]) * (np.pi - params[q3])
            circuit.cx(q1, q2)
            circuit.cx(q2, q3)
            circuit.p(2 * param, q3)
            circuit.cx(q2, q3)
            circuit.cx(q1, q2)

        with self.assertWarns(DeprecationWarning):
            feat_map = PauliFeatureMap(
                n_qubits, reps=2, paulis=["Z", "ZZ", "ZZZ"], entanglement=entanglement
            )
        feat_map.assign_parameters(params, inplace=True)

        qc = QuantumCircuit(n_qubits)
        for _ in range(2):
            qc.h([0, 1, 2])
            for e1 in entanglement[1]:
                z_block(qc, *e1)
            for e2 in entanglement[2]:
                zz_block(qc, *e2)
            for e3 in entanglement[3]:
                zzz_block(qc, *e3)

        self.assertTrue(Operator(feat_map).equiv(qc))

    def test_invalid_entanglement(self):
        """Test if a ValueError is raised when an invalid entanglement is passed"""
        n_qubits = 3
        entanglement = {
            1: [(0, 1), (2,)],
            2: [(0, 1), (1, 2)],
            3: [(0, 1, 2)],
        }

        with self.assertWarns(DeprecationWarning):
            feat_map = PauliFeatureMap(
                n_qubits, reps=2, paulis=["Z", "ZZ", "ZZZ"], entanglement=entanglement
            )
        with self.assertRaises(ValueError):
            feat_map.count_ops()

    def test_entanglement_not_specified(self):
        """Test if an error is raised when entanglement is not explicitly specified for
        all n-qubit pauli blocks"""
        n_qubits = 3
        entanglement = {
            1: [(0, 1), (2,)],
            3: [(0, 1, 2)],
        }
        with self.assertWarns(DeprecationWarning):
            feat_map = PauliFeatureMap(
                n_qubits, reps=2, paulis=["Z", "ZZ", "ZZZ"], entanglement=entanglement
            )
        with self.assertRaises(ValueError):
            feat_map.count_ops()


@ddt
class TestPauliFeatureMap(QiskitTestCase):
    """Test the Pauli feature map."""

    @data((2, 3, ["X", "YY"]), (5, 2, ["ZZZXZ", "XZ"]))
    @unpack
    def test_num_parameters(self, num_qubits, reps, pauli_strings):
        """Test the number of parameters equals the number of qubits, independent of reps."""
        encoding = pauli_feature_map(num_qubits, paulis=pauli_strings, reps=reps)
        self.assertEqual(encoding.num_parameters, num_qubits)

    def test_pauli_zz_with_barriers(self):
        """Test the generation of Pauli blocks."""
        encoding = QuantumCircuit(3)
        encoding.compose(pauli_feature_map(3, paulis=["zz"], insert_barriers=True), inplace=True)

        params = encoding.parameters

        def zz(circuit, i, j):
            circuit.cx(i, j)
            circuit.p(2 * (np.pi - params[i]) * (np.pi - params[j]), j)
            circuit.cx(i, j)

        ref = QuantumCircuit(3)
        for i in range(2):
            ref.h(range(3))
            ref.barrier()
            zz(ref, 0, 1)
            zz(ref, 0, 2)
            zz(ref, 1, 2)
            if i == 0:
                ref.barrier()

        self.assertEqual(ref, encoding)

    def test_pauli_xyz(self):
        """Test the generation of Pauli blocks."""
        encoding = QuantumCircuit(3)
        encoding.compose(pauli_feature_map(3, paulis=["xyz"], reps=1), inplace=True)
        # encoding = PauliFeatureMap(3, paulis=["XYZ"], reps=1).decompose()

        params = encoding.parameters

        # q_0: ────────■────────────────────────■──────────
        #      ┌────┐┌─┴─┐                    ┌─┴─┐┌──────┐
        # q_1: ┤ √X ├┤ X ├──■──────────────■──┤ X ├┤ √Xdg ├
        #      └┬───┤└───┘┌─┴─┐┌────────┐┌─┴─┐├───┤└──────┘
        # q_2: ─┤ H ├─────┤ X ├┤ P(2.8) ├┤ X ├┤ H ├────────
        #       └───┘     └───┘└────────┘└───┘└───┘
        # X on the most-significant, bottom qubit, Z on the top
        ref = QuantumCircuit(3)
        ref.h(range(3))
        ref.h(2)
        ref.sx(1)
        ref.cx(0, 1)
        ref.cx(1, 2)
        ref.p(2 * np.prod([np.pi - p for p in params]), 2)
        ref.cx(1, 2)
        ref.cx(0, 1)
        ref.sxdg(1)
        ref.h(2)

        self.assertEqual(ref, encoding)

    def test_first_order_circuit(self):
        """Test a first order expansion circuit."""
        times = [0.2, 1, np.pi, -1.2]
        encoding = z_feature_map(4, reps=3).assign_parameters(times)

        #      ┌───┐ ┌────────┐┌───┐ ┌────────┐┌───┐ ┌────────┐
        # q_0: ┤ H ├─┤ P(0.4) ├┤ H ├─┤ P(0.4) ├┤ H ├─┤ P(0.4) ├
        #      ├───┤ └┬──────┬┘├───┤ └┬──────┬┘├───┤ └┬──────┬┘
        # q_1: ┤ H ├──┤ P(2) ├─┤ H ├──┤ P(2) ├─┤ H ├──┤ P(2) ├─
        #      ├───┤ ┌┴──────┤ ├───┤ ┌┴──────┤ ├───┤ ┌┴──────┤
        # q_2: ┤ H ├─┤ P(2π) ├─┤ H ├─┤ P(2π) ├─┤ H ├─┤ P(2π) ├─
        #      ├───┤┌┴───────┴┐├───┤┌┴───────┴┐├───┤┌┴───────┴┐
        # q_3: ┤ H ├┤ P(-2.4) ├┤ H ├┤ P(-2.4) ├┤ H ├┤ P(-2.4) ├
        #      └───┘└─────────┘└───┘└─────────┘└───┘└─────────┘
        ref = QuantumCircuit(4)
        for _ in range(3):
            ref.h([0, 1, 2, 3])
            for i in range(4):
                ref.p(2 * times[i], i)

        self.assertTrue(Operator(encoding).equiv(ref))

    def test_second_order_circuit(self):
        """Test a second order expansion circuit."""
        times = [0.2, 1, np.pi]
        encoding = zz_feature_map(3, reps=2).assign_parameters(times)

        def zz_evolution(circuit, qubit1, qubit2):
            time = (np.pi - times[qubit1]) * (np.pi - times[qubit2])
            circuit.cx(qubit1, qubit2)
            circuit.p(2 * time, qubit2)
            circuit.cx(qubit1, qubit2)

        #      ┌───┐┌────────┐                                         ┌───┐┌────────┐»
        # q_0: ┤ H ├┤ P(0.4) ├──■─────────────────■────■────────────■──┤ H ├┤ P(0.4) ├»
        #      ├───┤└┬──────┬┘┌─┴─┐┌───────────┐┌─┴─┐  │            │  └───┘└────────┘»
        # q_1: ┤ H ├─┤ P(2) ├─┤ X ├┤ P(12.599) ├┤ X ├──┼────────────┼────■────────────»
        #      ├───┤┌┴──────┤ └───┘└───────────┘└───┘┌─┴─┐┌──────┐┌─┴─┐┌─┴─┐ ┌──────┐ »
        # q_2: ┤ H ├┤ P(2π) ├────────────────────────┤ X ├┤ P(0) ├┤ X ├┤ X ├─┤ P(0) ├─»
        #      └───┘└───────┘                        └───┘└──────┘└───┘└───┘ └──────┘ »
        # «                                                                              »
        # «q_0: ─────────────────────■─────────────────■────■────────────■───────────────»
        # «          ┌───┐ ┌──────┐┌─┴─┐┌───────────┐┌─┴─┐  │            │               »
        # «q_1: ──■──┤ H ├─┤ P(2) ├┤ X ├┤ P(12.599) ├┤ X ├──┼────────────┼────■──────────»
        # «     ┌─┴─┐├───┤┌┴──────┤└───┘└───────────┘└───┘┌─┴─┐┌──────┐┌─┴─┐┌─┴─┐┌──────┐»
        # «q_2: ┤ X ├┤ H ├┤ P(2π) ├───────────────────────┤ X ├┤ P(0) ├┤ X ├┤ X ├┤ P(0) ├»
        # «     └───┘└───┘└───────┘                       └───┘└──────┘└───┘└───┘└──────┘»
        # «
        # «q_0: ─────
        # «
        # «q_1: ──■──
        # «     ┌─┴─┐
        # «q_2: ┤ X ├
        # «     └───┘
        ref = QuantumCircuit(3)
        for _ in range(2):
            ref.h([0, 1, 2])
            for i in range(3):
                ref.p(2 * times[i], i)
            zz_evolution(ref, 0, 1)
            zz_evolution(ref, 0, 2)
            zz_evolution(ref, 1, 2)

        self.assertTrue(Operator(encoding).equiv(ref))

    @combine(entanglement=["linear", "reverse_linear", "pairwise"])
    def test_zz_entanglement(self, entanglement):
        """Test the ZZ feature map works with pairwise, linear and reverse_linear entanglement."""
        num_qubits = 5
        encoding = zz_feature_map(num_qubits, entanglement=entanglement, reps=1)
        ops = encoding.count_ops()
        expected_ops = {"h": num_qubits, "p": 2 * num_qubits - 1, "cx": 2 * (num_qubits - 1)}
        self.assertEqual(ops, expected_ops)

    def test_pauli_alpha(self):
        """Test  Pauli rotation factor (getter, setter)."""
        alpha = 1.234

        # this is needed as the outcoming Rust circuit has no qreg
        encoding = QuantumCircuit(1)
        encoding.compose(pauli_feature_map(1, alpha=alpha, paulis=["z"], reps=1), inplace=True)

        ref = QuantumCircuit(1)
        ref.h(0)
        ref.p(alpha * encoding.parameters[0], 0)

        self.assertEqual(ref, encoding)

    def test_zzfeaturemap_raises_if_too_small(self):
        """Test the ``ZZFeatureMap`` raises an error if the number of qubits is smaller than 2."""
        with self.assertRaises(QiskitError):
            _ = zz_feature_map(1)

    def test_dict_entanglement(self):
        """Test passing the entanglement as dictionary."""
        entanglement = {1: [(0,), (2,)], 2: [(1, 2)], 3: [(0, 1, 2)]}
        circuit = QuantumCircuit(3)
        circuit.compose(
            pauli_feature_map(3, reps=1, paulis=["z", "xx", "yyy"], entanglement=entanglement),
            inplace=True,
        )
        x = circuit.parameters

        ref = QuantumCircuit(3)
        ref.h(ref.qubits)

        ref.p(2 * x[0], 0)
        ref.p(2 * x[2], 2)

        ref.h([1, 2])
        ref.cx(1, 2)
        ref.p(2 * np.prod([np.pi - xi for xi in [x[1], x[2]]]), 2)
        ref.cx(1, 2)
        ref.h([1, 2])

        ref.sx(range(3))
        ref.cx(0, 1)
        ref.cx(1, 2)
        ref.p(2 * np.prod([np.pi - xi for xi in x]), 2)
        ref.cx(1, 2)
        ref.cx(0, 1)
        ref.sxdg(range(3))

        self.assertEqual(ref, circuit)

    def test_invalid_entanglement(self):
        """Test if a ValueError is raised when an invalid entanglement is passed"""
        n_qubits = 3
        entanglement = {
            1: [(0, 1), (2,)],
            2: [(0, 1), (1, 2)],
            3: [(0, 1, 2)],
        }

        with self.assertRaises(QiskitError):
            _ = pauli_feature_map(
                n_qubits, reps=2, paulis=["Z", "ZZ", "ZZZ"], entanglement=entanglement
            )

    def test_entanglement_not_specified(self):
        """Test if an error is raised when entanglement is not explicitly specified for
        all n-qubit pauli blocks"""
        n_qubits = 3
        entanglement = {
            1: [(0, 1), (2,)],
            3: [(0, 1, 2)],
        }
        with self.assertRaises(QiskitError):
            _ = pauli_feature_map(
                n_qubits, reps=2, paulis=["Z", "ZZ", "ZZZ"], entanglement=entanglement
            )

    def test_parameter_prefix(self):
        """Test the Parameter prefix"""
        encoding_pauli = pauli_feature_map(
            feature_dimension=2, reps=2, paulis=["ZY"], parameter_prefix="p"
        )
        encoding_z = z_feature_map(feature_dimension=2, reps=2, parameter_prefix="q")
        encoding_zz = zz_feature_map(feature_dimension=2, reps=2, parameter_prefix="r")
        x = ParameterVector("x", 2)
        y = Parameter("y")

        self.assertEqual(
            str(encoding_pauli.parameters),
            "ParameterView([ParameterVectorElement(p[0]), ParameterVectorElement(p[1])])",
        )
        self.assertEqual(
            str(encoding_z.parameters),
            "ParameterView([ParameterVectorElement(q[0]), ParameterVectorElement(q[1])])",
        )
        self.assertEqual(
            str(encoding_zz.parameters),
            "ParameterView([ParameterVectorElement(r[0]), ParameterVectorElement(r[1])])",
        )

        encoding_pauli_param_x = encoding_pauli.assign_parameters(x)
        encoding_z_param_x = encoding_z.assign_parameters(x)
        encoding_zz_param_x = encoding_zz.assign_parameters(x)

        self.assertEqual(
            str(encoding_pauli_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )
        self.assertEqual(
            str(encoding_z_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )
        self.assertEqual(
            str(encoding_zz_param_x.parameters),
            "ParameterView([ParameterVectorElement(x[0]), ParameterVectorElement(x[1])])",
        )

        encoding_pauli_param_y = encoding_pauli.assign_parameters([1, y])
        encoding_z_param_y = encoding_z.assign_parameters([1, y])
        encoding_zz_param_y = encoding_zz.assign_parameters([1, y])

        self.assertEqual(str(encoding_pauli_param_y.parameters), "ParameterView([Parameter(y)])")
        self.assertEqual(str(encoding_z_param_y.parameters), "ParameterView([Parameter(y)])")
        self.assertEqual(str(encoding_zz_param_y.parameters), "ParameterView([Parameter(y)])")

    def test_custom_data_mapping(self):
        """Test passing a custom data mapping function."""

        def my_mapping(x):
            return 42 if len(x) == 1 else np.sum(x)

        encoding = QuantumCircuit(2)
        encoding.compose(zz_feature_map(2, reps=1, data_map_func=my_mapping), inplace=True)

        params = encoding.parameters
        ref = QuantumCircuit(2)
        ref.h(range(2))
        ref.p(2 * 42, range(2))
        ref.cx(0, 1)
        ref.p(2 * (params[0] + params[1]), 1)
        ref.cx(0, 1)

        self.assertEqual(ref, encoding)


if __name__ == "__main__":
    unittest.main()
