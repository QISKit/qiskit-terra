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

"""Tests for ScalarOp class."""

import unittest

import numpy as np
from ddt import ddt

from qiskit.quantum_info.operators import Operator, ScalarOp
from test import combine
from test import QiskitTestCase


class ScalarOpTestCase(QiskitTestCase):
    """ScalarOp test case base class"""

    def assertOperator(self, operator, dims, target):
        """Checks operator has dims as input_dims and output_dims and matches the target"""
        self.assertTrue(isinstance(operator, Operator))
        self.assertEqual(operator.input_dims(), dims)
        self.assertEqual(operator.output_dims(), dims)
        self.assertEqual(operator, target)

    def assertScalarOp(self, scalarop, dims, target):
        """Checks ScalarOp has dims as input_dims and output_dims and matches the target coeff"""
        self.assertTrue(isinstance(scalarop, ScalarOp))
        self.assertEqual(scalarop.input_dims(), dims)
        self.assertEqual(scalarop.output_dims(), dims)
        self.assertAlmostEqual(scalarop.coeff, target)


@ddt
class TestScalarOpInit(ScalarOpTestCase):
    """Test initialization."""

    @combine(j=range(1, 5))
    def test_init(self, j):
        """Test {j}-qubit automatic dims."""
        dim = 2**j
        op = ScalarOp(dim)
        self.assertEqual(op.dim, (dim, dim))
        self.assertEqual(op.input_dims(), j * (2,))
        self.assertEqual(op.output_dims(), j * (2,))
        self.assertEqual(op.coeff, 1)
        self.assertEqual(op.num_qubits, j)

    def test_custom_dims(self):
        """Test custom dims."""
        dims = (2, 3, 4, 5)
        dim = np.prod(dims)
        op = ScalarOp(dims)
        self.assertEqual(op.dim, (dim, dim))
        self.assertEqual(op.input_dims(), dims)
        self.assertEqual(op.output_dims(), dims)
        self.assertEqual(op.coeff, 1)
        self.assertIsNone(op.num_qubits)

    def test_real_coeff(self):
        """Test real coeff."""
        op = ScalarOp(5, coeff=5.2)
        self.assertEqual(op.coeff, 5.2)

    def test_complex_coeff(self):
        """Test complex coeff."""
        op = ScalarOp(5, coeff=3j)
        self.assertEqual(op.coeff, 3j)


@ddt
class TestScalarOpMethods(ScalarOpTestCase):
    """Test ScalarOp methods"""

    @combine(dims=[2, 4, 5, (2, 3), (3, 2)], coeff=[0, 1, 2.1 - 3.1j])
    def test_to_operator(self, dims, coeff):
        """Test to_matrix and to_operator methods (dims={dims}, coeff={coeff})"""
        dim = np.prod(dims)
        iden = ScalarOp(dims, coeff=coeff)
        target = Operator(coeff * np.eye(dim), input_dims=dims, output_dims=dims)
        with self.subTest(msg="to_operator"):
            self.assertEqual(iden.to_operator(), target)

        with self.subTest(msg="to_matrix"):
            self.assertTrue(np.allclose(iden.to_matrix(), target.data))

    @combine(coeff=[0, 1, 5, -1j, 2.3 - 5.2j])
    def test_base_operator_conjugate(self, coeff):
        """Test basic class conjugate method (coeff={coeff})"""
        op = ScalarOp(4, coeff=coeff).conjugate()
        target = np.conjugate(coeff)
        self.assertTrue(isinstance(op, ScalarOp))
        self.assertEqual(op.coeff, target)

    @combine(coeff=[0, 1, 5, -1j, 2.3 - 5.2j])
    def test_base_operator_transpose(self, coeff):
        """Test basic class transpose method (coeff={coeff})"""
        op = ScalarOp(4, coeff=coeff).transpose()
        self.assertTrue(isinstance(op, ScalarOp))
        self.assertEqual(op.coeff, coeff)

    @combine(coeff=[0, 1, 5, -1j, 2.3 - 5.2j])
    def test_base_operator_adjoint(self, coeff):
        """Test basic class adjoint method (coeff={coeff})"""
        op = ScalarOp(4, coeff=coeff).adjoint()
        target = np.conjugate(coeff)
        self.assertTrue(isinstance(op, ScalarOp))
        self.assertEqual(op.coeff, target)

    @combine(coeff=[0, 1, 5, -1j, 2.3 - 5.2j])
    def test_base_operator_power_0(self, coeff):
        """Test basic class power method (** 0, coeff={coeff})"""
        op = ScalarOp(4, coeff=coeff).power(0)
        target = 1
        self.assertTrue(isinstance(op, ScalarOp))
        self.assertAlmostEqual(op.coeff, target)

    @combine(coeff=[1, 5, -1j, 2.3 - 5.2j], exp=[1, 3.3, -2])
    def test_base_operator_power_exp(self, coeff, exp):
        """Test basic class power method (** {exp}, coeff={coeff})"""
        op = ScalarOp(4, coeff=coeff).power(exp)
        target = coeff**exp
        self.assertTrue(isinstance(op, ScalarOp))
        self.assertAlmostEqual(op.coeff, target)


@ddt
class TestScalarOpLinearMethods(ScalarOpTestCase):
    """Test ScalarOp linear add, sub, mul methods"""

    @combine(coeff=[0, 1, -3.1, 1 + 3j])
    def test_multiply_negate(self, coeff):
        """Test scalar multiplication. Negate. (coeff={coeff})"""
        dims = (3, 2)
        val = -ScalarOp(dims, coeff=coeff)
        target = -1 * coeff
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[1, -1, -5.1 - 2j])
    def test_multiply(self, coeff1, coeff2):
        """Test scalar multiplication. ({coeff1}, {coeff2})"""
        dims = (3, 2)
        val = coeff2 * ScalarOp(dims, coeff=coeff1)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)
        val = ScalarOp(dims, coeff=coeff1) * coeff2
        target = coeff2 * coeff1
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_add(self, coeff1, coeff2):
        """Test add operation with ScalarOp. ({coeff1} + {coeff2})"""
        # Add two ScalarOps
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(6, coeff=coeff2)

        val = op1 + op2
        target = coeff1 + coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j])
    def test_radd(self, coeff1):
        """Test right-side addition with ScalarOp."""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)

        val = op1 + 0
        self.assertScalarOp(val, dims, coeff1)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_sum(self, coeff1, coeff2):
        """Test add operation with ScalarOp. ({coeff1} + {coeff2})"""
        # Add two ScalarOps
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(6, coeff=coeff2)

        val = sum([op1, op2])
        target = coeff1 + coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_subtract(self, coeff1, coeff2):
        """Test add operation with ScalarOp. ({coeff1} - {coeff2})"""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(6, coeff=coeff2)

        val = op1 - op2
        target = coeff1 - coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_rsub(self, coeff1, coeff2):
        """Test right-side subtraction with ScalarOp."""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(dims, coeff=coeff2)

        val = op2.__rsub__(op1)
        target = coeff1 - coeff2
        self.assertScalarOp(val, dims, target)

    @combine(
        coeff1=[0, 1, -3.1, 1 + 3j],
        coeff2=[-1, -5.1 - 2j],
        qargs=[[0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [0, 1, 2], [0, 2, 1]],
    )
    def test_add_qargs(self, coeff1, coeff2, qargs):
        """Test add operation with ScalarOp. ({coeff1} + {coeff2}({qargs}))"""
        # Add two ScalarOps
        full_dims = np.array([2, 3, 4])
        dims1 = tuple(full_dims.tolist())
        dims2 = tuple(full_dims[qargs].tolist())
        op1 = ScalarOp(dims1, coeff=coeff1)
        op2 = ScalarOp(dims2, coeff=coeff2)

        val = op1 + op2(qargs)
        target = coeff1 + coeff2
        self.assertScalarOp(val, dims1, target)

    @combine(
        coeff1=[0, 1, -3.1, 1 + 3j],
        coeff2=[-1, -5.1 - 2j],
        qargs=[[0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [0, 1, 2], [0, 2, 1]],
    )
    def test_subtract_qargs(self, coeff1, coeff2, qargs):
        """Test subtract operation with ScalarOp. ({coeff1} - {coeff2}({qargs}))"""
        # Add two ScalarOps
        full_dims = np.array([2, 3, 4])
        dims1 = tuple(full_dims.tolist())
        dims2 = tuple(full_dims[qargs].tolist())
        op1 = ScalarOp(dims1, coeff=coeff1)
        op2 = ScalarOp(dims2, coeff=coeff2)

        val = op1 - op2(qargs)
        target = coeff1 - coeff2
        self.assertScalarOp(val, dims1, target)

    @combine(coeff=[0, 1, -3.1, 1 + 3j], label=["II", "XX", "YY", "ZZ"])
    def test_add_operator(self, coeff, label):
        """Test add operation with Operator (coeff={coeff}, label={label})"""
        dims = (2, 2)
        iden = ScalarOp(4, coeff=coeff)
        op = Operator.from_label(label)
        with self.subTest(msg=f"{iden} + Operator({label})"):
            val = iden + op
            target = coeff * Operator.from_label("II") + op
            self.assertOperator(val, dims, target)

        with self.subTest(msg=f"Operator({label}) + {iden}"):
            val = op + iden
            target = coeff * Operator.from_label("II") + op
            self.assertOperator(val, dims, target)

    @combine(coeff=[0, 1, -3.1, 1 + 3j], label=["II", "XX", "YY", "ZZ"])
    def test_subtract_operator(self, coeff, label):
        """Test subtract operation with Operator (coeff={coeff}, label={label})"""
        dims = (2, 2)
        iden = ScalarOp(4, coeff=coeff)
        op = Operator.from_label(label)
        with self.subTest(msg=f"{iden} - Operator({label})"):
            val = iden - op
            target = coeff * Operator.from_label("II") - op
            self.assertOperator(val, dims, target)

        with self.subTest(msg=f"Operator({label}) - {iden}"):
            val = op - iden
            target = op - coeff * Operator.from_label("II")
            self.assertOperator(val, dims, target)

    @combine(
        coeff=[0, 1, -3.1, 1 + 3j],
        qargs=[[0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [0, 1, 2], [2, 0, 1], [1, 2, 0]],
    )
    def test_add_operator_qargs(self, coeff, qargs):
        """Test qargs add operation with Operator (coeff={coeff}, qargs={qargs})"""
        # Get labels for qarg addition
        part_array = np.array(["X", "Y", "Z"])[range(len(qargs))]
        label = "".join(part_array)
        full_array = np.array(3 * ["I"])
        inds = [2 - i for i in reversed(qargs)]
        full_array[inds] = part_array
        full_label = "".join(full_array)
        dims = 3 * (2,)
        val = ScalarOp(dims, coeff=coeff) + Operator.from_label(label)(qargs)
        target = (coeff * Operator.from_label(3 * "I")) + Operator.from_label(full_label)
        self.assertOperator(val, dims, target)

    @combine(
        coeff=[0, 1, -3.1, 1 + 3j],
        qargs=[[0], [1], [2], [0, 1], [1, 0], [0, 2], [2, 0], [0, 1, 2], [2, 0, 1], [1, 2, 0]],
    )
    def test_subtract_operator_qargs(self, coeff, qargs):
        """Test qargs subtract operation with Operator (coeff={coeff}, qargs={qargs})"""
        # Get labels for qarg addition
        part_array = np.array(["X", "Y", "Z"])[range(len(qargs))]
        label = "".join(part_array)
        full_array = np.array(3 * ["I"])
        inds = [2 - i for i in reversed(qargs)]
        full_array[inds] = part_array
        full_label = "".join(full_array)
        dims = 3 * (2,)
        val = ScalarOp(dims, coeff=coeff) - Operator.from_label(label)(qargs)
        target = (coeff * Operator.from_label(3 * "I")) - Operator.from_label(full_label)
        self.assertOperator(val, dims, target)


@ddt
class TestScalarOpTensor(ScalarOpTestCase):
    """Test ScalarOp tensor and expand methods."""

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_expand_scalar(self, coeff1, coeff2):
        """Test expand method with two ScalarOp ({coeff1}, {coeff2})"""
        dims1 = (3, 2)
        dims2 = (2, 4)
        op1 = ScalarOp(dims1, coeff=coeff1)
        op2 = ScalarOp(dims2, coeff=coeff2)

        val = op1.expand(op2)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims1 + dims2, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_tensor_scalar(self, coeff1, coeff2):
        """Test tensor method with two ScalarOp. {coeff1}, {coeff2})"""
        dims1 = (3, 2)
        dims2 = (2, 4)
        op1 = ScalarOp(dims1, coeff=coeff1)
        op2 = ScalarOp(dims2, coeff=coeff2)

        val = op1.tensor(op2)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims2 + dims1, target)

    @combine(coeff=[0, 1, -3.1, 1 + 3j], label=["I", "X", "Y", "Z"])
    def test_tensor_operator(self, coeff, label):
        """Test tensor and expand methods with ScalarOp and Operator. ({coeff}, {label})"""
        dim = 3
        iden = ScalarOp(dim, coeff=coeff)
        op = Operator.from_label(label)

        with self.subTest(msg=f"{iden}.expand(Operator({label}))"):
            val = iden.expand(op)
            target = iden.to_operator().expand(op)
            self.assertOperator(val, (3, 2), target)

        with self.subTest(msg=f"Operator({label}).expand({iden})"):
            val = op.expand(iden)
            target = op.expand(iden.to_operator())
            self.assertOperator(val, (2, 3), target)

        with self.subTest(msg=f"{iden}.tensor(Operator({label}))"):
            val = iden.tensor(op)
            target = iden.to_operator().tensor(op)
            self.assertOperator(val, (2, 3), target)

        with self.subTest(msg=f"Operator({label}).tensor({iden})"):
            val = op.tensor(iden)
            target = op.tensor(iden.to_operator())
            self.assertOperator(val, (3, 2), target)


@ddt
class TestScalarOpCompose(ScalarOpTestCase):
    """Test ScalarOp compose and dot methods with another ScalarOp."""

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_compose_scalar(self, coeff1, coeff2):
        """Test compose method with two ScalarOp. ({coeff1}, {coeff2})"""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(dims, coeff=coeff2)

        val = op1.compose(op2)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_dot_scalar(self, coeff1, coeff2):
        """Test dot method with two ScalarOp. ({coeff1}, {coeff2})"""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(dims, coeff=coeff2)

        val = op1.dot(op2)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)

        val = op1 @ op2
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[0, -1, -5.1 - 2j])
    def test_matmul_scalar(self, coeff1, coeff2):
        """Test matmul method with two ScalarOp. ({coeff1}, {coeff2})"""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(dims, coeff=coeff2)
        val = op1 & op2
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_mul_scalar(self, coeff1, coeff2):
        """Test mul method with two ScalarOp. ({coeff1}, {coeff2})"""
        dims = (3, 2)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(dims, coeff=coeff2)
        val = op1.dot(op2)
        target = coeff1 * coeff2
        self.assertScalarOp(val, dims, target)

    @combine(coeff1=[0, 1, -3.1, 1 + 3j], coeff2=[-1, -5.1 - 2j])
    def test_compose_qargs_scalar(self, coeff1, coeff2):
        """Test qargs compose and dot methods with two ScalarOp. ({coeff1}, {coeff2})"""
        dims = (2, 3)
        op1 = ScalarOp(dims, coeff=coeff1)
        op2 = ScalarOp(2, coeff=coeff2)
        op3 = ScalarOp(3, coeff=coeff2)

        with self.subTest(msg=f"{op1}.compose({op2}, qargs=[0])"):
            val = op1.compose(op2, qargs=[0])
            target = coeff1 * coeff2
            self.assertScalarOp(val, dims, target)

        with self.subTest(msg=f"{op1}.compose({op3}, qargs=[1])"):
            val = op1.compose(op3, qargs=[1])
            target = coeff1 * coeff2
            self.assertScalarOp(val, dims, target)

        with self.subTest(msg=f"{op1}.dot({op2}, qargs=[0])"):
            val = op1.dot(op2, qargs=[0])
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)

        with self.subTest(msg=f"{op1}.dot({op3}, qargs=[1])"):
            val = op1.dot(op3, qargs=[1])
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)

        with self.subTest(msg=f"{op1} & {op2}([0])"):
            val = op1 & op2([0])
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)

        with self.subTest(msg=f"{op1} & {op3}([1])"):
            val = op1 & op3([1])
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)

        with self.subTest(msg=f"{op1} * {op2}([0])"):
            val = op1.dot(op2([0]))
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)

        with self.subTest(msg=f"{op1} * {op3}([1])"):
            val = op1.dot(op3([1]))
            target = coeff1 * coeff2
            self.assertTrue(isinstance(val, ScalarOp))
            self.assertEqual(val.input_dims(), dims)
            self.assertEqual(val.output_dims(), dims)
            self.assertAlmostEqual(val.coeff, target)


@ddt
class TestScalarOpComposeOperator(ScalarOpTestCase):
    """Test ScalarOp compose and dot methods with an Operator."""

    @combine(coeff=[0, 1, -3.1, 1 + 3j], label=["II", "XX", "YY", "ZZ"])
    def test_compose_operator(self, coeff, label):
        """Test compose and dot methods with ScalarOp and Operator."""
        dim = 4
        iden = ScalarOp(dim, coeff=coeff)
        op = Operator.from_label(label)

        with self.subTest(msg=f"{iden}.compose(Operator({label}))"):
            val = iden.compose(op)
            target = iden.to_operator().compose(op)
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"Operator({label}).compose({iden})"):
            val = op.compose(iden)
            target = op.compose(iden.to_operator())
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden}.dot(Operator({label}))"):
            val = iden.dot(op)
            target = iden.to_operator().dot(op)
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"Operator({label}).dot({iden})"):
            val = op.dot(iden)
            target = op.dot(iden.to_operator())
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden} & Operator({label})"):
            val = iden & op
            target = iden.to_operator().compose(op)
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"Operator({label}) & {iden}"):
            val = op & iden
            target = op.compose(iden.to_operator())
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

    @combine(coeff=[0, 1, -3.1, 1 + 3j], label=["I", "X", "Y", "Z"])
    def test_compose_qargs_operator(self, coeff, label):
        """Test qargs compose and dot methods with ScalarOp and Operator."""
        iden = ScalarOp((2, 2), coeff=coeff)
        op = Operator.from_label(label)

        with self.subTest(msg=f"{iden}.compose(Operator({label}), qargs=[0])"):
            val = iden.compose(op, qargs=[0])
            target = iden.to_operator().compose(op, qargs=[0])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden}.compose(Operator({label}), qargs=[1])"):
            val = iden.compose(op, qargs=[1])
            target = iden.to_operator().compose(op, qargs=[1])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden}.dot(Operator({label}), qargs=[0])"):
            val = iden.dot(op, qargs=[0])
            target = iden.to_operator().dot(op, qargs=[0])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden}.dot(Operator({label}), qargs=[1])"):
            val = iden.dot(op, qargs=[1])
            target = iden.to_operator().dot(op, qargs=[1])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden} & Operator({label})([0])"):
            val = iden & op([0])
            target = iden.to_operator().compose(op, qargs=[0])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)

        with self.subTest(msg=f"{iden} & Operator({label})([1])"):
            val = iden & op([1])
            target = iden.to_operator().compose(op, qargs=[1])
            self.assertTrue(isinstance(val, Operator))
            self.assertEqual(val.input_dims(), (2, 2))
            self.assertEqual(val.output_dims(), (2, 2))
            self.assertEqual(val, target)


if __name__ == "__main__":
    unittest.main()
