# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's power instruction operation."""

import unittest

import numpy as np
import scipy.linalg
from ddt import data, ddt, unpack

from qiskit.circuit import Gate, QuantumCircuit, AnnotatedOperation
from qiskit.circuit.library import (
    CPhaseGate,
    CSdgGate,
    CSGate,
    CXGate,
    IGate,
    PhaseGate,
    RGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
    SdgGate,
    SGate,
    TdgGate,
    TGate,
    UnitaryGate,
    XXMinusYYGate,
    XXPlusYYGate,
    ZGate,
    iSwapGate,
    SwapGate,
)
from qiskit.quantum_info.operators import Operator
from test import QiskitTestCase, combine


@ddt
class TestPowerSgate(QiskitTestCase):
    """Test Sgate.power() with integer"""

    @data(2, 1, 0, -1, -2)
    def test_sgate_int(self, n):
        """Test Sgate.power(n) method with n as integer."""
        result = SGate().power(n)
        self.assertIsInstance(result, PhaseGate)
        self.assertEqual(Operator(result), Operator(SGate()).power(n))

    results = {
        1.5: np.array([[1, 0], [0, -0.70710678 + 0.70710678j]], dtype=complex),
        0.1: np.array([[1, 0], [0, 0.98768834 + 0.15643447j]], dtype=complex),
        -1.5: np.array([[1, 0], [0, -0.70710678 - 0.70710678j]], dtype=complex),
        -0.1: np.array([[1, 0], [0, 0.98768834 - 0.15643447j]], dtype=complex),
    }

    @data(1.5, 0.1, -1.5, -0.1)
    def test_sgate_float(self, n):
        """Test Sgate.power(<float>) method."""
        result = SGate().power(n)

        expected = self.results[n]
        self.assertIsInstance(result, PhaseGate)
        self.assertEqual(Operator(result), Operator(expected))


@ddt
class TestPowerIntCX(QiskitTestCase):
    """Test CX.power() with integer"""

    results = {
        2: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),
        1: np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex),
        0: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),
        -1: np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex),
        -2: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex),
    }

    @data(2, 1, 0, -2)
    def test_cx_int(self, n):
        """Test CX.power(<int>) method."""
        result = CXGate().power(n)

        self.assertEqual(result.label, "cx^" + str(n))
        self.assertIsInstance(result, UnitaryGate)
        self.assertEqual(Operator(result), Operator(self.results[n]))


class TestGateSqrt(QiskitTestCase):
    """Test square root using Gate.power()"""

    def test_unitary_sqrt(self):
        """Test UnitaryGate.power(1/2) method."""
        expected = np.array([[0.5 + 0.5j, 0.5 + 0.5j], [-0.5 - 0.5j, 0.5 + 0.5j]], dtype=complex)

        result = UnitaryGate([[0, 1j], [-1j, 0]]).power(1 / 2)

        self.assertEqual(result.label, "unitary^0.5")
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertEqual(Operator(result), Operator(expected))

    def test_standard_sqrt(self):
        """Test standard Gate.power(1/2) method."""
        expected = np.array([[1, 0], [0, 0.70710678118 + 0.70710678118j]], dtype=complex)

        result = SGate().power(1 / 2)

        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertEqual(Operator(result), Operator(expected))

    def test_composite_sqrt(self):
        """Test composite Gate.power(1/2) method."""
        circ = QuantumCircuit(1, name="my_gate")

        thetaz = 0.1
        thetax = 0.2
        circ.rz(thetaz, 0)
        circ.rx(thetax, 0)
        gate = circ.to_gate()

        result = gate.power(1 / 2)

        iden = Operator.from_label("I")
        xgen = Operator.from_label("X")
        zgen = Operator.from_label("Z")

        def rzgate(theta):
            return np.cos(0.5 * theta) * iden - 1j * np.sin(0.5 * theta) * zgen

        def rxgate(theta):
            return np.cos(0.5 * theta) * iden - 1j * np.sin(0.5 * theta) * xgen

        rxrz = rxgate(thetax).dot(rzgate(thetaz))

        self.assertEqual(result.label, "my_gate^0.5")
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertEqual(Operator(result) & Operator(result), rxrz)


@ddt
class TestGateFloat(QiskitTestCase):
    """Test power generalization to root calculation"""

    @data(2, 3, 4, 5, 6, 7, 8, 9)
    def test_direct_root(self, degree):
        """Test nth root"""
        result = SGate().power(1 / degree)

        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertEqual(Operator(result).power(degree), Operator(SGate()))

    @data(2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8, 0.2)
    def test_float_gt_one(self, exponent):
        """Test greater-than-one exponents"""
        result = SGate().power(exponent)

        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        # SGate().to_matrix() is diagonal so `**` is equivalent.
        self.assertEqual(Operator(SGate().to_matrix() ** exponent), Operator(result))

    def test_minus_zero_two(self, exponent=-0.2):
        """Test Sgate^(-0.2)"""
        result = SGate().power(exponent)

        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertEqual(
            Operator(np.array([[1, 0], [0, 0.95105652 - 0.30901699j]], dtype=complex)),
            Operator(result),
        )


@ddt
class TestPowerInvariant(QiskitTestCase):
    """Test Gate.power invariants"""

    @data(-3, -2, -1, 1, 2, 3)
    def test_invariant1_int(self, n):
        """Test (op^(1/n))^(n) == op, integer n"""
        result = SGate().power(1 / n).power(n)
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        self.assertTrue(Operator(SGate()), Operator(result))

    @data(-3, -2, -1, 1, 2, 3)
    def test_invariant2(self, n):
        """Test op^(n) * op^(-n) == I"""
        result = Operator(SGate()).power(n) & Operator(SGate()).power(-n)
        expected = Operator(np.eye(2))

        self.assertEqual(len(result.data), len(expected.data))
        self.assertEqual(result, expected)


@ddt
class TestGatePow(QiskitTestCase):
    """Test gate __pow__ method."""

    @data(2, 3, 4, 5)
    def test_gate_pow(self, degree):
        """Test gate __pow__ method."""
        self.assertEqual(SGate() ** (1 / degree), SGate().power(1 / degree))
        self.assertEqual(CXGate() ** (1 / degree), CXGate().power(1 / degree))


@ddt
class TestEfficientGatePowering(QiskitTestCase):
    """Test gate powering is efficient where expected."""

    @data(
        (CPhaseGate(0.1), CPhaseGate),
        (CSdgGate(), CPhaseGate),
        (CSGate(), CPhaseGate),
        (IGate(), IGate),
        (PhaseGate(-0.1), PhaseGate),
        (RGate(0.1, 0.1), RGate),
        (RXGate(0.1), RXGate),
        (RXXGate(-0.1), RXXGate),
        (RYGate(-0.1), RYGate),
        (RYYGate(0.1), RYYGate),
        (RZGate(0.1), RZGate),
        (RZXGate(-0.1), RZXGate),
        (RZZGate(-0.1), RZZGate),
        (SdgGate(), PhaseGate),
        (SGate(), PhaseGate),
        (TGate(), PhaseGate),
        (TdgGate(), PhaseGate),
        (XXMinusYYGate(-0.1, 0.1), XXMinusYYGate),
        (XXPlusYYGate(2.1, 0.1), XXPlusYYGate),
        (ZGate(), PhaseGate),
        (iSwapGate(), XXPlusYYGate),
    )
    @unpack
    def test_efficient_gate_powering(self, gate: Gate, output_gate_type: type[Gate]):
        """Test efficient gate powering."""
        exponents = (-5, -0.5, -0.1, 0, 0.1, 0.5, 5)
        for exponent in exponents:
            result = gate.power(exponent)
            self.assertIsInstance(result, output_gate_type)
            expected = scipy.linalg.fractional_matrix_power(np.array(gate), exponent)
            np.testing.assert_allclose(np.array(result), expected, atol=1e-8)


@ddt
class TestPowerAnnotatedGate(QiskitTestCase):
    """Test raising gates to power using ``annotated`` argument."""

    def test_s_gate(self):
        """Test raising SGate to a power. Since SGate has an efficient ``power``
        method, the result should not be an annotated operation.
        """
        result = SGate().power(1.5, annotated=True)
        self.assertNotIsInstance(result, AnnotatedOperation)

    def test_swap_gate(self):
        """Test raising SwapGate to a power using different methods."""
        # SwapGate has no native power method.
        result1 = SwapGate().power(1.5, annotated=True)
        self.assertIsInstance(result1, AnnotatedOperation)

        result2 = SwapGate().power(1.5, annotated=False)
        self.assertIsInstance(result2, UnitaryGate)

        self.assertEqual(Operator(result1), Operator(result2))


@ddt
class TestControlledGatePower(QiskitTestCase):
    """Test power method preserves ctrl_state on controlled gates."""

    @combine(
        gate=[
            CPhaseGate(0.7, ctrl_state=0),
            CPhaseGate(0.7, ctrl_state=1),
            CSGate(ctrl_state=0),
            CSGate(ctrl_state=1),
            CSdgGate(ctrl_state=0),
            CSdgGate(ctrl_state=1),
        ],
        exponent=[0.5, 1.5, 2, -1, -0.5],
    )
    def test_power_preserves_ctrl_state(self, gate: Gate, exponent: float):
        """Test that power() preserves the control state on CPhase, CS, and CSdg gates."""
        powered = gate.power(exponent)
        self.assertEqual(powered.ctrl_state, gate.ctrl_state)
        self.assertEqual(
            Operator(powered),
            Operator(gate.base_gate.power(exponent).control(1, ctrl_state=gate.ctrl_state)),
        )


if __name__ == "__main__":
    unittest.main()
