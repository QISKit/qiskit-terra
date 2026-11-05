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


"""Tests for Pauli operator class."""

import itertools as it
import re
import unittest
from functools import lru_cache
from test import QiskitTestCase, combine

import numpy as np
from ddt import data, ddt, unpack

from qiskit import QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.circuit.random import random_clifford_circuit
from qiskit.circuit.library import (
    CXGate,
    CYGate,
    CZGate,
    ECRGate,
    efficient_su2,
    HGate,
    IGate,
    SdgGate,
    SGate,
    SwapGate,
    iSwapGate,
    DCXGate,
    XGate,
    YGate,
    ZGate,
    SXGate,
    SXdgGate,
    RXGate,
    RYGate,
    RZGate,
    CPhaseGate,
    CRXGate,
    CRYGate,
    CRZGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.compiler.transpiler import transpile
from qiskit.exceptions import QiskitError
from qiskit.primitives import BackendEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info.operators import Operator, Pauli, SparsePauliOp
from qiskit.quantum_info import random_clifford, random_pauli
from qiskit.utils import optionals

LABEL_REGEX = re.compile(r"(?P<coeff>[+-]?1?[ij]?)(?P<pauli>[IXYZ]*)")
PHASE_MAP = {"": 0, "-i": 1, "-": 2, "i": 3}


def _split_pauli_label(label):
    match_ = LABEL_REGEX.fullmatch(label)
    return match_["pauli"], match_["coeff"]


def _phase_from_label(label):
    coeff = LABEL_REGEX.fullmatch(label)["coeff"] or ""
    return PHASE_MAP[coeff.replace("+", "").replace("1", "").replace("j", "i")]


@lru_cache(maxsize=8)
def pauli_group_labels(nq, full_group=True):
    """Generate list of the N-qubit pauli group string labels"""
    labels = ["".join(i) for i in it.product(("I", "X", "Y", "Z"), repeat=nq)]
    if full_group:
        labels = ["".join(i) for i in it.product(("", "-i", "-", "i"), labels)]
    return labels


def operator_from_label(label):
    """Construct operator from full Pauli group label"""
    pauli, coeff = _split_pauli_label(label)
    coeff = (-1j) ** _phase_from_label(coeff)
    return coeff * Operator.from_label(pauli)


@ddt
class TestPauliConversions(QiskitTestCase):
    """Test representation conversions of Pauli"""

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_labels(self, label):
        """Test round trip label conversion"""
        pauli = Pauli(label)
        self.assertEqual(Pauli(str(pauli)), pauli)

    @data("S", "XX-")
    def test_invalid_labels(self, label):
        """Test raise if invalid labels are supplied"""
        with self.assertRaises(QiskitError):
            Pauli(label)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_operator(self, label):
        """Test Pauli operator conversion"""
        value = Operator(Pauli(label))
        target = operator_from_label(label)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_matrix_sparse(self, label):
        """Test Pauli operator conversion"""
        spmat = Pauli(label).to_matrix(sparse=True)
        value = Operator(spmat.todense())
        target = operator_from_label(label)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1), *pauli_group_labels(2))
    def test_to_instruction(self, label):
        """Test Pauli to instruction"""
        pauli = Pauli(label)
        value = Operator(pauli.to_instruction())
        target = Operator(pauli)
        self.assertEqual(value, target)

    @data((IGate(), "I"), (XGate(), "X"), (YGate(), "Y"), (ZGate(), "Z"))
    @unpack
    def test_init_single_pauli_gate(self, gate, label):
        """Test initialization from Pauli basis gates"""
        self.assertEqual(str(Pauli(gate)), label)

    @data("IXYZ", "XXY", "ZYX", "ZI", "Y")
    def test_init_pauli_gate(self, label):
        """Test initialization from Pauli basis gates"""
        pauli = Pauli(PauliGate(label))
        self.assertEqual(str(pauli), label)


@ddt
class TestPauliProperties(QiskitTestCase):
    """Test Pauli properties"""

    @data("I", "XY", "XYZ", "IXYZ", "IXYZX")
    def test_len(self, label):
        """Test __len__ method"""
        self.assertEqual(len(Pauli(label)), len(label))

    @data(*it.product(pauli_group_labels(1, full_group=False), pauli_group_labels(1)))
    @unpack
    def test_equal(self, label1, label2):
        """Test __eq__ method"""
        pauli1 = Pauli(label1)
        pauli2 = Pauli(label2)
        target = (
            np.all(pauli1.z == pauli2.z)
            and np.all(pauli1.x == pauli2.x)
            and pauli1.phase == pauli2.phase
        )
        self.assertEqual(pauli1 == pauli2, target)

    @data(*it.product(pauli_group_labels(1, full_group=False), pauli_group_labels(1)))
    @unpack
    def test_equiv(self, label1, label2):
        """Test equiv method"""
        pauli1 = Pauli(label1)
        pauli2 = Pauli(label2)
        target = np.all(pauli1.z == pauli2.z) and np.all(pauli1.x == pauli2.x)
        self.assertEqual(pauli1.equiv(pauli2), target)

    @data(*pauli_group_labels(1))
    def test_phase(self, label):
        """Test phase attribute"""
        pauli = Pauli(label)
        _, coeff = _split_pauli_label(str(pauli))
        target = _phase_from_label(coeff)
        self.assertEqual(pauli.phase, target)

    @data(*((p, q) for p in ["I", "X", "Y", "Z"] for q in range(4)))
    @unpack
    def test_phase_setter(self, pauli, phase):
        """Test phase setter"""
        pauli = Pauli(pauli)
        pauli.phase = phase
        _, coeff = _split_pauli_label(str(pauli))
        value = _phase_from_label(coeff)
        self.assertEqual(value, phase)

    def test_x_setter(self):
        """Test phase attribute"""
        pauli = Pauli("II")
        pauli.x = True
        self.assertEqual(pauli, Pauli("XX"))

    def test_z_setter(self):
        """Test phase attribute"""
        pauli = Pauli("II")
        pauli.z = True
        self.assertEqual(pauli, Pauli("ZZ"))

    @data(
        *(
            ("IXYZ", i)
            for i in [0, 1, 2, 3, slice(None, None, None), slice(None, 2, None), [0, 3], [2, 1, 3]]
        )
    )
    @unpack
    def test_getitem(self, label, qubits):
        """Test __getitem__"""
        pauli = Pauli(label)
        value = str(pauli[qubits])
        val_array = np.array(list(reversed(label)))[qubits]
        target = "".join(reversed(val_array.tolist()))
        self.assertEqual(value, target, msg=f"indices = {qubits}")

    @data(
        (0, "iY", "iIIY"),
        ([1, 0], "XZ", "IZX"),
        (slice(None, None, None), "XYZ", "XYZ"),
        (slice(None, None, -1), "XYZ", "ZYX"),
    )
    @unpack
    def test_setitem(self, qubits, value, target):
        """Test __setitem__"""
        pauli = Pauli("III")
        pauli[qubits] = value
        self.assertEqual(str(pauli), target)

    def test_insert(self):
        """Test insert method"""
        pauli = Pauli("III")
        pauli = pauli.insert([2, 0, 4], "XYZ")
        self.assertEqual(str(pauli), "IXIZIY")

    def test_delete(self):
        """Test delete method"""
        pauli = Pauli("IXYZ")
        pauli = pauli.delete([])
        self.assertEqual(str(pauli), "IXYZ")
        pauli = pauli.delete([0, 2])
        self.assertEqual(str(pauli), "IY")


@ddt
class TestPauli(QiskitTestCase):
    """Tests for Pauli operator class."""

    @data(*pauli_group_labels(2))
    def test_conjugate(self, label):
        """Test conjugate method."""
        value = Pauli(label).conjugate()
        target = operator_from_label(label).conjugate()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_transpose(self, label):
        """Test transpose method."""
        value = Pauli(label).transpose()
        target = operator_from_label(label).transpose()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_adjoint(self, label):
        """Test adjoint method."""
        value = Pauli(label).adjoint()
        target = operator_from_label(label).adjoint()
        self.assertEqual(Operator(value), target)

    @data(*pauli_group_labels(2))
    def test_inverse(self, label):
        """Test inverse method."""
        pauli = Pauli(label)
        value = pauli.inverse()
        target = pauli.adjoint()
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(2, full_group=False), repeat=2))
    @unpack
    def test_dot(self, label1, label2):
        """Test dot method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.dot(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.dot(op2)
        self.assertEqual(value, target)
        target = op1 @ op2
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1))
    def test_dot_qargs(self, label2):
        """Test dot method with qargs."""
        label1 = "-iXYZ"
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        qargs = [0]
        value = Operator(p1.dot(p2, qargs=qargs))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.dot(op2, qargs=qargs)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(2, full_group=False), repeat=2))
    @unpack
    def test_compose(self, label1, label2):
        """Test compose method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.compose(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.compose(op2)
        self.assertEqual(value, target)

    @data(*pauli_group_labels(1))
    def test_compose_qargs(self, label2):
        """Test compose method with qargs."""
        label1 = "-XYZ"
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        qargs = [0]
        value = Operator(p1.compose(p2, qargs=qargs))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.compose(op2, qargs=qargs)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(1, full_group=False), repeat=2))
    @unpack
    def test_tensor(self, label1, label2):
        """Test tensor method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.tensor(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.tensor(op2)
        self.assertEqual(value, target)

    @data(*it.product(pauli_group_labels(1, full_group=False), repeat=2))
    @unpack
    def test_expand(self, label1, label2):
        """Test expand method."""
        p1 = Pauli(label1)
        p2 = Pauli(label2)
        value = Operator(p1.expand(p2))
        op1 = operator_from_label(label1)
        op2 = operator_from_label(label2)
        target = op1.expand(op2)
        self.assertEqual(value, target)

    @data("II", "XI", "YX", "ZZ", "YZ")
    def test_power(self, label):
        """Test power method."""
        iden = Pauli("II")
        op = Pauli(label)
        self.assertTrue(op**2, iden)

    @data(1, 1.0, -1, -1.0, 1j, -1j)
    def test_multiply(self, val):
        """Test multiply method."""
        op = val * Pauli(([True, True], [False, False], 0))
        phase = (-1j) ** op.phase
        self.assertEqual(phase, val)
        op = Pauli(([True, True], [False, False], 0)) * val
        phase = (-1j) ** op.phase
        self.assertEqual(phase, val)

    def test_multiply_except(self):
        """Test multiply method raises exceptions."""
        op = Pauli("XYZ")
        self.assertRaises(QiskitError, op._multiply, 2)

    @data(0, 1, 2, 3)
    def test_negate(self, phase):
        """Test negate method"""
        op = Pauli(([False], [True], phase))
        neg = -op
        self.assertTrue(op.equiv(neg))
        self.assertEqual(neg.phase, (op.phase + 2) % 4)

    @data(*it.product(pauli_group_labels(1, False), repeat=2))
    @unpack
    def test_commutes(self, p1, p2):
        """Test commutes method"""
        P1 = Pauli(p1)
        P2 = Pauli(p2)
        self.assertEqual(P1.commutes(P2), P1.dot(P2) == P2.dot(P1))

    @data(*it.product(pauli_group_labels(1, False), repeat=2))
    @unpack
    def test_anticommutes(self, p1, p2):
        """Test anticommutes method"""
        P1 = Pauli(p1)
        P2 = Pauli(p2)
        self.assertEqual(P1.anticommutes(P2), P1.dot(P2) == -P2.dot(P1))

    @data(
        *it.product(
            (
                IGate(),
                XGate(),
                YGate(),
                ZGate(),
                HGate(),
                SGate(),
                SdgGate(),
                SXGate(),
                SXdgGate(),
                RXGate(theta=np.pi / 2),
                RYGate(theta=np.pi / 2),
                RZGate(phi=np.pi / 2),
                RZGate(phi=17 * np.pi / 2),
                RXGate(theta=-5 * np.pi / 2),
            ),
            pauli_group_labels(1, False),
        )
    )
    @unpack
    def test_evolve_clifford1(self, gate, label):
        """Test evolve method for 1-qubit Clifford gates."""
        op = Operator(gate)
        pauli = Pauli(label)
        value = Operator(pauli.evolve(gate))
        value_h = Operator(pauli.evolve(gate, frame="h"))
        value_s = Operator(pauli.evolve(gate, frame="s"))
        value_inv = Operator(pauli.evolve(gate.inverse()))
        target = op.adjoint().dot(pauli).dot(op)
        self.assertEqual(value, target)
        self.assertEqual(value, value_h)
        self.assertEqual(value_inv, value_s)

    @data(
        *it.product(
            (
                CXGate(),
                CYGate(),
                CZGate(),
                SwapGate(),
                iSwapGate(),
                ECRGate(),
                DCXGate(),
                CPhaseGate(theta=np.pi),
                CRXGate(theta=np.pi),
                CRYGate(theta=np.pi),
                CRZGate(theta=np.pi),
                RXXGate(theta=np.pi / 2),
                RYYGate(theta=np.pi / 2),
                RZZGate(theta=np.pi / 2),
                RZXGate(theta=np.pi / 2),
                XXMinusYYGate(theta=np.pi),
                XXPlusYYGate(theta=-np.pi),
                RZZGate(theta=7 * np.pi / 2),
                RXXGate(theta=-15 * np.pi / 2),
            ),
            pauli_group_labels(2, False),
        )
    )
    @unpack
    def test_evolve_clifford2(self, gate, label):
        """Test evolve method for 2-qubit Clifford gates."""
        op = Operator(gate)
        pauli = Pauli(label)
        value = Operator(pauli.evolve(gate))
        value_h = Operator(pauli.evolve(gate, frame="h"))
        value_s = Operator(pauli.evolve(gate, frame="s"))
        value_inv = Operator(pauli.evolve(gate.inverse()))
        target = op.adjoint().dot(pauli).dot(op)
        self.assertEqual(value, target)
        self.assertEqual(value, value_h)
        self.assertEqual(value_inv, value_s)

    @data(
        *it.product(
            (
                IGate(),
                XGate(),
                YGate(),
                ZGate(),
                HGate(),
                SGate(),
                SdgGate(),
                SXGate(),
                SXdgGate(),
                CXGate(),
                CYGate(),
                CZGate(),
                SwapGate(),
                iSwapGate(),
                ECRGate(),
                DCXGate(),
            ),
            [int, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64],
        )
    )
    @unpack
    def test_phase_dtype_evolve_clifford(self, gate, dtype):
        """Test phase dtype for evolve method for Clifford gates."""
        z = np.ones(gate.num_qubits, dtype=bool)
        x = np.ones(gate.num_qubits, dtype=bool)
        phase = (np.sum(z & x) % 4).astype(dtype)
        paulis = Pauli((z, x, phase))
        evo = paulis.evolve(gate)
        self.assertEqual(evo.phase.dtype, dtype)

    def test_evolve_clifford_qargs(self):
        """Test evolve method for random Clifford"""
        cliff = random_clifford(3, seed=10)
        op = Operator(cliff)
        pauli = random_pauli(5, seed=10)
        qargs = [3, 0, 1]
        value = Operator(pauli.evolve(cliff, qargs=qargs))
        value_h = Operator(pauli.evolve(cliff, qargs=qargs, frame="h"))
        value_s = Operator(pauli.evolve(cliff, qargs=qargs, frame="s"))
        value_inv = Operator(pauli.evolve(cliff.adjoint(), qargs=qargs))
        target = Operator(pauli).compose(op.adjoint(), qargs=qargs).dot(op, qargs=qargs)
        self.assertEqual(value, target)
        self.assertEqual(value, value_h)
        self.assertEqual(value_inv, value_s)

    def test_evolve_clifford_circuit_qargs(self):
        """Test evolve method for random Clifford circuit"""
        qargs = [2, 4, 1, 0]
        qc = random_clifford_circuit(4, 100, seed=123)
        op = Operator(qc)
        pauli = random_pauli(5, seed=5678)
        target_s = Operator(pauli).compose(op, qargs=qargs).dot(op.adjoint(), qargs=qargs)
        target_h = Operator(pauli).compose(op.adjoint(), qargs=qargs).dot(op, qargs=qargs)
        value_evolve_s = Operator(pauli.evolve(qc, frame="s", qargs=qargs))
        value_evolve_h = Operator(pauli.evolve(qc, frame="h", qargs=qargs))
        self.assertEqual(value_evolve_s, target_s)
        self.assertEqual(value_evolve_h, target_h)

    @data("s", "h")
    def test_evolve_with_misleading_name(self, frame):
        """Test evolve by circuit contents, not by name (fixed bug)."""
        circ = QuantumCircuit(2, name="cx")
        p = Pauli("IX")
        self.assertEqual(p, p.evolve(circ, frame=frame))

    def test_barrier_delay_sim(self):
        """Test barrier and delay instructions can be simulated"""
        target_circ = QuantumCircuit(2)
        target_circ.x(0)
        target_circ.y(1)
        target = Pauli(target_circ)

        circ = QuantumCircuit(2)
        circ.x(0)
        circ.delay(100, 0)
        circ.barrier([0, 1])
        circ.y(1)
        value = Pauli(circ)
        self.assertEqual(value, target)

    @data(("", 0), ("-", 2), ("i", 3), ("-1j", 1))
    @unpack
    def test_zero_qubit_pauli_construction(self, label, phase):
        """Test that Paulis of zero qubits can be constructed."""
        expected = Pauli(label + "X")[0:0]  # Empty slice from a 1q Pauli, which becomes phaseless
        expected.phase = phase
        test = Pauli(label)
        self.assertEqual(expected, test)

    def test_circuit_with_bit(self):
        """Test new-style Bit support when converting from QuantumCircuit"""
        circ = QuantumCircuit([Qubit()])
        circ.x(0)
        value = Pauli(circ)
        target = Pauli("X")

        self.assertEqual(value, target)

    def test_apply_layout_with_transpile(self):
        """Test the apply_layout method with a transpiler layout."""
        psi = efficient_su2(4, reps=4, entanglement="circular")
        op = Pauli("IZZZ")
        backend = GenericBackendV2(num_qubits=7)
        transpiled_psi = transpile(psi, backend, optimization_level=3, seed_transpiler=12345)
        permuted_op = op.apply_layout(transpiled_psi.layout)
        identity_op = Pauli("I" * 7)
        initial_layout = transpiled_psi.layout.initial_index_layout(filter_ancillas=True)
        final_layout = transpiled_psi.layout.routing_permutation()
        qargs = [final_layout[x] for x in initial_layout]
        expected_op = identity_op.compose(op, qargs=qargs)
        self.assertNotEqual(op, permuted_op)
        self.assertEqual(permuted_op, expected_op)

    def test_apply_layout_consistency(self):
        """Test that the Pauli apply_layout() is consistent with the SparsePauliOp apply_layout()."""
        psi = efficient_su2(4, reps=4, entanglement="circular")
        op = Pauli("IZZZ")
        sparse_op = SparsePauliOp(op)
        backend = GenericBackendV2(num_qubits=7)
        transpiled_psi = transpile(psi, backend, optimization_level=3, seed_transpiler=12345)
        permuted_op = op.apply_layout(transpiled_psi.layout)
        permuted_sparse_op = sparse_op.apply_layout(transpiled_psi.layout)
        self.assertEqual(SparsePauliOp(permuted_op), permuted_sparse_op)

    def test_permute_pauli_estimator_example(self):
        """Test using the apply_layout method with an estimator workflow."""
        psi = efficient_su2(4, reps=4, entanglement="circular")
        op = Pauli("XXXI")
        backend = GenericBackendV2(num_qubits=7, seed=0)
        backend.set_options(seed_simulator=123)
        estimator = BackendEstimatorV2(backend=backend)
        thetas = list(range(len(psi.parameters)))
        transpiled_psi = transpile(psi, backend, optimization_level=3)
        permuted_op = op.apply_layout(transpiled_psi.layout)
        job = estimator.run([(transpiled_psi, permuted_op, thetas)])
        res = job.result()[0].data.evs
        if optionals.HAS_AER:
            np.testing.assert_allclose(res, [0.20898438], rtol=0.5, atol=0.2)
        else:
            np.testing.assert_allclose(res, [0.15820312], rtol=0.5, atol=0.2)

    def test_apply_layout_invalid_qubits_list(self):
        """Test that apply_layout with an invalid qubit count raises."""
        op = Pauli("YI")
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 1], 1)

    def test_apply_layout_invalid_layout_list(self):
        """Test that apply_layout with an invalid layout list raises."""
        op = Pauli("YI")
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 3], 2)

    def test_apply_layout_invalid_layout_list_no_num_qubits(self):
        """Test that apply_layout with an invalid layout list raises."""
        op = Pauli("YI")
        with self.assertRaises(QiskitError):
            op.apply_layout([0, 2])

    def test_apply_layout_layout_list_no_num_qubits(self):
        """Test apply_layout with a layout list and no qubit count"""
        op = Pauli("YI")
        res = op.apply_layout([1, 0])
        self.assertEqual(Pauli("IY"), res)

    def test_apply_layout_layout_list_and_num_qubits(self):
        """Test apply_layout with a layout list and qubit count"""
        op = Pauli("YI")
        res = op.apply_layout([4, 0], 5)
        self.assertEqual(Pauli("IIIIY"), res)

    def test_apply_layout_null_layout_no_num_qubits(self):
        """Test apply_layout with a null layout"""
        op = Pauli("IZ")
        res = op.apply_layout(layout=None)
        self.assertEqual(op, res)

    def test_apply_layout_null_layout_and_num_qubits(self):
        """Test apply_layout with a null layout a num_qubits provided"""
        op = Pauli("IZ")
        res = op.apply_layout(layout=None, num_qubits=5)
        # this should expand the operator
        self.assertEqual(Pauli("IIIIZ"), res)

    def test_apply_layout_null_layout_invalid_num_qubits(self):
        """Test apply_layout with a null layout and num_qubits smaller than capable"""
        op = Pauli("IZ")
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=None, num_qubits=1)

    def test_apply_layout_negative_indices(self):
        """Test apply_layout with negative indices"""
        op = Pauli("IZ")
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=[-1, 0], num_qubits=3)

    def test_apply_layout_duplicate_indices(self):
        """Test apply_layout with duplicate indices"""
        op = Pauli("IZ")
        with self.assertRaises(QiskitError):
            op.apply_layout(layout=[0, 0], num_qubits=3)

    @combine(phase=["", "-i", "-", "i"], layout=[None, []])
    def test_apply_layout_zero_qubit(self, phase, layout):
        """Test apply_layout with a zero-qubit operator"""
        op = Pauli(phase)
        res = op.apply_layout(layout=layout, num_qubits=5)
        self.assertEqual(Pauli(phase + "IIIII"), res)


if __name__ == "__main__":
    unittest.main()
