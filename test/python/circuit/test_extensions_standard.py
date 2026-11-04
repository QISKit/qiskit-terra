# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import unittest
from inspect import signature
from math import pi
import numpy as np
from scipy.linalg import expm
from ddt import data, ddt, unpack

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit import Gate, ControlledGate
from qiskit.circuit.library import (
    U1Gate,
    U2Gate,
    U3Gate,
    CU1Gate,
    CU3Gate,
    XXMinusYYGate,
    XXPlusYYGate,
    RZGate,
    XGate,
    YGate,
    GlobalPhaseGate,
)
from qiskit.quantum_info import Pauli
from qiskit.quantum_info.operators.predicates import matrix_equal, is_unitary_matrix
from qiskit.quantum_info import Operator
from qiskit import transpile
from test import QiskitTestCase


class TestStandard1Q(QiskitTestCase):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        super().setUp()
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier(self):
        self.circuit.barrier(self.qr[1])
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].operation.name, "barrier")
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_barrier_wires(self):
        self.circuit.barrier(1)
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].operation.name, "barrier")
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_barrier_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.barrier, self.cr[0])
        self.assertRaises(CircuitError, qc.barrier, self.cr)
        self.assertRaises(CircuitError, qc.barrier, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.barrier, 0.0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.qr)
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].operation.name, "barrier")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2]))

    def test_barrier_none(self):
        self.circuit.barrier()
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].operation.name, "barrier")
        self.assertEqual(
            self.circuit[0].qubits,
            (self.qr[0], self.qr[1], self.qr[2], self.qr2[0], self.qr2[1], self.qr2[2]),
        )

    def test_ccx(self):
        self.circuit.ccx(self.qr[0], self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "ccx")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2]))

    def test_ccx_wires(self):
        self.circuit.ccx(0, 1, 2)
        self.assertEqual(self.circuit[0].operation.name, "ccx")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2]))

    def test_ccx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ccx, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.ccx, self.qr[0], self.qr[0], self.qr[2])
        self.assertRaises(CircuitError, qc.ccx, 0.0, self.qr[0], self.qr[2])
        self.assertRaises(CircuitError, qc.ccx, self.cr, self.qr, self.qr)
        self.assertRaises(CircuitError, qc.ccx, "a", self.qr[1], self.qr[2])

    def test_ch(self):
        self.circuit.ch(self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "ch")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_ch_wires(self):
        self.circuit.ch(0, 1)
        self.assertEqual(self.circuit[0].operation.name, "ch")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_ch_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ch, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.ch, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.ch, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.ch, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.ch, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.ch, "a", self.qr[1])

    def test_crz(self):
        self.circuit.crz(1, self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "crz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_cry(self):
        self.circuit.cry(1, self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "cry")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_crx(self):
        self.circuit.crx(1, self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "crx")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_crz_wires(self):
        self.circuit.crz(1, 0, 1)
        self.assertEqual(self.circuit[0].operation.name, "crz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_cry_wires(self):
        self.circuit.cry(1, 0, 1)
        self.assertEqual(self.circuit[0].operation.name, "cry")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_crx_wires(self):
        self.circuit.crx(1, 0, 1)
        self.assertEqual(self.circuit[0].operation.name, "crx")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1]))

    def test_crz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.crz, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.crz, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.crz, 0, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.crz, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.crz, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.crz, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.crz, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.crz, 'a', self.qr[1], self.qr[2])

    def test_cry_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cry, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.cry, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cry, 0, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.cry, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.cry, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cry, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.cry, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.cry, 'a', self.qr[1], self.qr[2])

    def test_crx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.crx, 0, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.crx, 0, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.crx, 0, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.crx, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.crx, 0, self.qr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.crx, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(CircuitError, qc.crx, 0, self.cr, self.qr)
        # TODO self.assertRaises(CircuitError, qc.crx, 'a', self.qr[1], self.qr[2])

    def test_cswap(self):
        self.circuit.cswap(self.qr[0], self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "cswap")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2]))

    def test_cswap_wires(self):
        self.circuit.cswap(0, 1, 2)
        self.assertEqual(self.circuit[0].operation.name, "cswap")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2]))

    def test_cswap_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cswap, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cswap, self.qr[1], self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.qr[1], 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.cr[0], self.cr[1], self.qr[0])
        self.assertRaises(CircuitError, qc.cswap, self.qr[0], self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, 0.0, self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, (self.qr, 3), self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, self.cr, self.qr[0], self.qr[1])
        self.assertRaises(CircuitError, qc.cswap, "a", self.qr[1], self.qr[2])

    def test_cu1(self):
        self.circuit.append(CU1Gate(1), [self.qr[1], self.qr[2]])
        self.assertEqual(self.circuit[0].operation.name, "cu1")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cu1_wires(self):
        self.circuit.append(CU1Gate(1), [1, 2])
        self.assertEqual(self.circuit[0].operation.name, "cu1")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cu3(self):
        self.circuit.append(CU3Gate(1, 2, 3), [self.qr[1], self.qr[2]])
        self.assertEqual(self.circuit[0].operation.name, "cu3")
        self.assertEqual(self.circuit[0].operation.params, [1, 2, 3])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cu3_wires(self):
        self.circuit.append(CU3Gate(1, 2, 3), [1, 2])
        self.assertEqual(self.circuit[0].operation.name, "cu3")
        self.assertEqual(self.circuit[0].operation.params, [1, 2, 3])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cx(self):
        self.circuit.cx(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "cx")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cx_wires(self):
        self.circuit.cx(1, 2)
        self.assertEqual(self.circuit[0].operation.name, "cx")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cx, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cx, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cx, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.cx, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cx, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cx, "a", self.qr[1])

    def test_cy(self):
        self.circuit.cy(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "cy")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cy_wires(self):
        self.circuit.cy(1, 2)
        self.assertEqual(self.circuit[0].operation.name, "cy")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cy_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cy, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cy, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cy, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.cy, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cy, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cy, "a", self.qr[1])

    def test_cz(self):
        self.circuit.cz(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "cz")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cz_wires(self):
        self.circuit.cz(1, 2)
        self.assertEqual(self.circuit[0].operation.name, "cz")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_cz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.cz, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.cz, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.cz, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.cz, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.cz, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.cz, "a", self.qr[1])

    def test_h(self):
        self.circuit.h(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "h")
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_h_wires(self):
        self.circuit.h(1)
        self.assertEqual(self.circuit[0].operation.name, "h")
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_h_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.h, self.cr[0])
        self.assertRaises(CircuitError, qc.h, self.cr)
        self.assertRaises(CircuitError, qc.h, (self.qr, 3))
        self.assertRaises(CircuitError, qc.h, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.h, 0.0)

    def test_h_reg(self):
        instruction_set = self.circuit.h(self.qr)
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "h")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))

    def test_h_reg_inv(self):
        instruction_set = self.circuit.h(self.qr).inverse()
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "h")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))

    def test_iden(self):
        self.circuit.id(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "id")
        self.assertEqual(self.circuit[0].operation.params, [])

    def test_iden_wires(self):
        self.circuit.id(1)
        self.assertEqual(self.circuit[0].operation.name, "id")
        self.assertEqual(self.circuit[0].operation.params, [])

    def test_iden_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.id, self.cr[0])
        self.assertRaises(CircuitError, qc.id, self.cr)
        self.assertRaises(CircuitError, qc.id, (self.qr, 3))
        self.assertRaises(CircuitError, qc.id, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.id, 0.0)

    def test_iden_reg(self):
        instruction_set = self.circuit.id(self.qr)
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "id")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))

    def test_iden_reg_inv(self):
        instruction_set = self.circuit.id(self.qr).inverse()
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "id")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))

    def test_rx(self):
        self.circuit.rx(1, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "rx")
        self.assertEqual(self.circuit[0].operation.params, [1])

    def test_rx_wires(self):
        self.circuit.rx(1, 1)
        self.assertEqual(self.circuit[0].operation.name, "rx")
        self.assertEqual(self.circuit[0].operation.params, [1])

    def test_rx_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rx, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.rx, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.rx, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.rx, 0, 0.0)
        self.assertRaises(CircuitError, qc.rx, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.rx, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.rx, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.rx, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.rx, 0, "a")

    def test_rx_reg(self):
        instruction_set = self.circuit.rx(1, self.qr)
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "rx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_rx_reg_inv(self):
        instruction_set = self.circuit.rx(1, self.qr).inverse()
        self.assertEqual(len(instruction_set), 3)
        self.assertEqual(instruction_set[0].operation.name, "rx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_rx_pi(self):
        qc = self.circuit
        qc.rx(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "rx")
        self.assertEqual(self.circuit[0].operation.params, [pi / 2])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_ry(self):
        self.circuit.ry(1, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "ry")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_ry_wires(self):
        self.circuit.ry(1, 1)
        self.assertEqual(self.circuit[0].operation.name, "ry")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_ry_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.ry, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.ry, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.ry, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.ry, 0, 0.0)
        self.assertRaises(CircuitError, qc.ry, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.ry, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.ry, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.ry, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.ry, 0, "a")

    def test_ry_reg(self):
        instruction_set = self.circuit.ry(1, self.qr)
        self.assertEqual(instruction_set[0].operation.name, "ry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_ry_reg_inv(self):
        instruction_set = self.circuit.ry(1, self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "ry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_ry_pi(self):
        qc = self.circuit
        qc.ry(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "ry")
        self.assertEqual(self.circuit[0].operation.params, [pi / 2])

    def test_rz(self):
        self.circuit.rz(1, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "rz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_rz_wires(self):
        self.circuit.rz(1, 1)
        self.assertEqual(self.circuit[0].operation.name, "rz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_rz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rz, self.cr[0], self.cr[1])
        self.assertRaises(CircuitError, qc.rz, self.qr[1], 0)
        self.assertRaises(CircuitError, qc.rz, 0, self.cr[0])
        self.assertRaises(CircuitError, qc.rz, 0, 0.0)
        self.assertRaises(CircuitError, qc.rz, self.qr[2], self.qr[1])
        self.assertRaises(CircuitError, qc.rz, 0, (self.qr, 3))
        self.assertRaises(CircuitError, qc.rz, 0, self.cr)
        # TODO self.assertRaises(CircuitError, qc.rz, 'a', self.qr[1])
        self.assertRaises(CircuitError, qc.rz, 0, "a")

    def test_rz_reg(self):
        instruction_set = self.circuit.rz(1, self.qr)
        self.assertEqual(instruction_set[0].operation.name, "rz")
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_rz_reg_inv(self):
        instruction_set = self.circuit.rz(1, self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "rz")
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_rz_pi(self):
        self.circuit.rz(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "rz")
        self.assertEqual(self.circuit[0].operation.params, [pi / 2])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_rzz(self):
        self.circuit.rzz(1, self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "rzz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_rzz_wires(self):
        self.circuit.rzz(1, 1, 2)
        self.assertEqual(self.circuit[0].operation.name, "rzz")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_rzz_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.rzz, 1, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.rzz, 1, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.rzz, 1, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.rzz, 1, "a", self.qr[1])
        self.assertRaises(CircuitError, qc.rzz, 0.1, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.rzz, 0.1, self.qr[0], self.qr[0])

    def test_s(self):
        self.circuit.s(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "s")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_s_wires(self):
        self.circuit.s(1)
        self.assertEqual(self.circuit[0].operation.name, "s")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_s_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.s, self.cr[0])
        self.assertRaises(CircuitError, qc.s, self.cr)
        self.assertRaises(CircuitError, qc.s, (self.qr, 3))
        self.assertRaises(CircuitError, qc.s, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.s, 0.0)

    def test_s_reg(self):
        instruction_set = self.circuit.s(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "s")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_s_reg_inv(self):
        instruction_set = self.circuit.s(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "sdg")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_sdg(self):
        self.circuit.sdg(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "sdg")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_sdg_wires(self):
        self.circuit.sdg(1)
        self.assertEqual(self.circuit[0].operation.name, "sdg")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_sdg_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.sdg, self.cr[0])
        self.assertRaises(CircuitError, qc.sdg, self.cr)
        self.assertRaises(CircuitError, qc.sdg, (self.qr, 3))
        self.assertRaises(CircuitError, qc.sdg, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.sdg, 0.0)

    def test_sdg_reg(self):
        instruction_set = self.circuit.sdg(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "sdg")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_sdg_reg_inv(self):
        instruction_set = self.circuit.sdg(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "s")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_swap(self):
        self.circuit.swap(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].operation.name, "swap")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_swap_wires(self):
        self.circuit.swap(1, 2)
        self.assertEqual(self.circuit[0].operation.name, "swap")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1], self.qr[2]))

    def test_swap_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.swap, self.cr[1], self.cr[2])
        self.assertRaises(CircuitError, qc.swap, self.qr[0], self.qr[0])
        self.assertRaises(CircuitError, qc.swap, 0.0, self.qr[0])
        self.assertRaises(CircuitError, qc.swap, (self.qr, 3), self.qr[0])
        self.assertRaises(CircuitError, qc.swap, self.cr, self.qr)
        self.assertRaises(CircuitError, qc.swap, "a", self.qr[1])
        self.assertRaises(CircuitError, qc.swap, self.qr, self.qr2[[1, 2]])
        self.assertRaises(CircuitError, qc.swap, self.qr[:2], self.qr2)

    def test_t(self):
        self.circuit.t(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "t")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_t_wire(self):
        self.circuit.t(1)
        self.assertEqual(self.circuit[0].operation.name, "t")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_t_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.t, self.cr[0])
        self.assertRaises(CircuitError, qc.t, self.cr)
        self.assertRaises(CircuitError, qc.t, (self.qr, 3))
        self.assertRaises(CircuitError, qc.t, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.t, 0.0)

    def test_t_reg(self):
        instruction_set = self.circuit.t(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "t")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_t_reg_inv(self):
        instruction_set = self.circuit.t(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "tdg")
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_tdg(self):
        self.circuit.tdg(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "tdg")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_tdg_wires(self):
        self.circuit.tdg(1)
        self.assertEqual(self.circuit[0].operation.name, "tdg")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_tdg_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.tdg, self.cr[0])
        self.assertRaises(CircuitError, qc.tdg, self.cr)
        self.assertRaises(CircuitError, qc.tdg, (self.qr, 3))
        self.assertRaises(CircuitError, qc.tdg, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.tdg, 0.0)

    def test_tdg_reg(self):
        instruction_set = self.circuit.tdg(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "tdg")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_tdg_reg_inv(self):
        instruction_set = self.circuit.tdg(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "t")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_u1(self):
        self.circuit.append(U1Gate(1), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u1")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u1_wires(self):
        self.circuit.append(U1Gate(1), [1])
        self.assertEqual(self.circuit[0].operation.name, "u1")
        self.assertEqual(self.circuit[0].operation.params, [1])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u1_reg(self):
        instruction_set = self.circuit.append(U1Gate(1), [self.qr])
        self.assertEqual(instruction_set[0].operation.name, "u1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_u1_reg_inv(self):
        instruction_set = self.circuit.append(U1Gate(1), [self.qr]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "u1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_u1_pi(self):
        qc = self.circuit
        qc.append(U1Gate(pi / 2), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u1")
        self.assertEqual(self.circuit[0].operation.params, [pi / 2])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u2(self):
        self.circuit.append(U2Gate(1, 2), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u2")
        self.assertEqual(self.circuit[0].operation.params, [1, 2])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u2_wires(self):
        self.circuit.append(U2Gate(1, 2), [1])
        self.assertEqual(self.circuit[0].operation.name, "u2")
        self.assertEqual(self.circuit[0].operation.params, [1, 2])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u2_reg(self):
        instruction_set = self.circuit.append(U2Gate(1, 2), [self.qr])
        self.assertEqual(instruction_set[0].operation.name, "u2")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [1, 2])

    def test_u2_reg_inv(self):
        instruction_set = self.circuit.append(U2Gate(1, 2), [self.qr]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "u2")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [-pi - 2, -1 + pi])

    def test_u2_pi(self):
        self.circuit.append(U2Gate(pi / 2, 0.3 * pi), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u2")
        self.assertEqual(self.circuit[0].operation.params, [pi / 2, 0.3 * pi])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u3(self):
        self.circuit.append(U3Gate(1, 2, 3), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u3")
        self.assertEqual(self.circuit[0].operation.params, [1, 2, 3])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u3_wires(self):
        self.circuit.append(U3Gate(1, 2, 3), [1])
        self.assertEqual(self.circuit[0].operation.name, "u3")
        self.assertEqual(self.circuit[0].operation.params, [1, 2, 3])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_u3_reg(self):
        instruction_set = self.circuit.append(U3Gate(1, 2, 3), [self.qr])
        self.assertEqual(instruction_set[0].operation.name, "u3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [1, 2, 3])

    def test_u3_reg_inv(self):
        instruction_set = self.circuit.append(U3Gate(1, 2, 3), [self.qr]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "u3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [-1, -3, -2])

    def test_u3_pi(self):
        self.circuit.append(U3Gate(pi, pi / 2, 0.3 * pi), [self.qr[1]])
        self.assertEqual(self.circuit[0].operation.name, "u3")
        self.assertEqual(self.circuit[0].operation.params, [pi, pi / 2, 0.3 * pi])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_x(self):
        self.circuit.x(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "x")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_x_wires(self):
        self.circuit.x(1)
        self.assertEqual(self.circuit[0].operation.name, "x")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_x_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.x, self.cr[0])
        self.assertRaises(CircuitError, qc.x, self.cr)
        self.assertRaises(CircuitError, qc.x, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.x, 0.0)

    def test_x_reg(self):
        instruction_set = self.circuit.x(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "x")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_x_reg_inv(self):
        instruction_set = self.circuit.x(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "x")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_y(self):
        self.circuit.y(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "y")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_y_wires(self):
        self.circuit.y(1)
        self.assertEqual(self.circuit[0].operation.name, "y")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_y_invalid(self):
        qc = self.circuit
        self.assertRaises(CircuitError, qc.y, self.cr[0])
        self.assertRaises(CircuitError, qc.y, self.cr)
        self.assertRaises(CircuitError, qc.y, (self.qr, "a"))
        self.assertRaises(CircuitError, qc.y, 0.0)

    def test_y_reg(self):
        instruction_set = self.circuit.y(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "y")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_y_reg_inv(self):
        instruction_set = self.circuit.y(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "y")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_z(self):
        self.circuit.z(self.qr[1])
        self.assertEqual(self.circuit[0].operation.name, "z")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_z_wires(self):
        self.circuit.z(1)
        self.assertEqual(self.circuit[0].operation.name, "z")
        self.assertEqual(self.circuit[0].operation.params, [])
        self.assertEqual(self.circuit[0].qubits, (self.qr[1],))

    def test_z_reg(self):
        instruction_set = self.circuit.z(self.qr)
        self.assertEqual(instruction_set[0].operation.name, "z")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_z_reg_inv(self):
        instruction_set = self.circuit.z(self.qr).inverse()
        self.assertEqual(instruction_set[0].operation.name, "z")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1],))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_global_phase(self):
        qc = self.circuit
        qc.append(GlobalPhaseGate(0.1), [])
        self.assertEqual(self.circuit[0].operation.name, "global_phase")
        self.assertEqual(self.circuit[0].operation.params, [0.1])
        self.assertEqual(self.circuit[0].qubits, ())

    def test_global_phase_inv(self):
        instruction_set = self.circuit.append(GlobalPhaseGate(0.1), []).inverse()
        self.assertEqual(len(instruction_set), 1)
        self.assertEqual(instruction_set[0].operation.params, [-0.1])

    def test_global_phase_matrix(self):
        """Test global_phase matrix."""
        theta = 0.1
        np.testing.assert_allclose(
            np.array(GlobalPhaseGate(theta)),
            np.array([[np.exp(1j * theta)]], dtype=complex),
            atol=1e-7,
        )

    def test_global_phase_consistency(self):
        """Tests compatibility of GlobalPhaseGate with QuantumCircuit.global_phase"""
        theta = 0.1
        qc1 = QuantumCircuit(0, global_phase=theta)
        qc2 = QuantumCircuit(0)
        qc2.append(GlobalPhaseGate(theta), [])
        np.testing.assert_allclose(
            Operator(qc1),
            Operator(qc2),
            atol=1e-7,
        )

    def test_transpile_global_phase_consistency(self):
        """Tests compatibility of transpiled GlobalPhaseGate with QuantumCircuit.global_phase"""
        qc1 = QuantumCircuit(0, global_phase=0.3)
        qc2 = QuantumCircuit(0, global_phase=0.2)
        qc2.append(GlobalPhaseGate(0.1), [])
        np.testing.assert_allclose(
            Operator(transpile(qc1, basis_gates=["u"])),
            Operator(transpile(qc2, basis_gates=["u"])),
            atol=1e-7,
        )


@ddt
class TestStandard2Q(QiskitTestCase):
    """Standard Extension Test. Gates with two Qubits"""

    def setUp(self):
        super().setUp()
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier_reg_bit(self):
        self.circuit.barrier(self.qr, self.qr2[0])
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].operation.name, "barrier")
        self.assertEqual(self.circuit[0].qubits, (self.qr[0], self.qr[1], self.qr[2], self.qr2[0]))

    def test_ch_reg_reg(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "ch")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_ch_reg_reg_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "ch")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_ch_reg_bit(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "ch")
        self.assertEqual(
            instruction_set[1].qubits,
            (
                self.qr[1],
                self.qr2[1],
            ),
        )
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_ch_reg_bit_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "ch")
        self.assertEqual(
            instruction_set[1].qubits,
            (
                self.qr[1],
                self.qr2[1],
            ),
        )
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_ch_bit_reg(self):
        instruction_set = self.circuit.ch(self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "ch")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_crz_reg_reg(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crz_reg_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_crz_reg_bit(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crz_reg_bit_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_crz_bit_reg(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crz_bit_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cry_reg_reg(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cry_reg_reg_inv(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cry_reg_bit(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cry_reg_bit_inv(self):
        instruction_set = self.circuit.cry(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cry_bit_reg(self):
        instruction_set = self.circuit.cry(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cry_bit_reg_inv(self):
        instruction_set = self.circuit.cry(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cry")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_crx_reg_reg(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crx_reg_reg_inv(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_crx_reg_bit(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crx_reg_bit_inv(self):
        instruction_set = self.circuit.crx(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_crx_bit_reg(self):
        instruction_set = self.circuit.crx(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_crx_bit_reg_inv(self):
        instruction_set = self.circuit.crx(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "crx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cu1_reg_reg(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr, self.qr2])
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cu1_reg_reg_inv(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr, self.qr2]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cu1_reg_bit(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr, self.qr2[1]])
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cu1_reg_bit_inv(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr, self.qr2[1]]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cu1_bit_reg(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr[1], self.qr2])
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1])

    def test_cu1_bit_reg_inv(self):
        instruction_set = self.circuit.append(CU1Gate(1), [self.qr[1], self.qr2]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu1")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1])

    def test_cu3_reg_reg(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr, self.qr2])
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1, 2, 3])

    def test_cu3_reg_reg_inv(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr, self.qr2]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1, -3, -2])

    def test_cu3_reg_bit(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr, self.qr2[1]])
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1, 2, 3])

    def test_cu3_reg_bit_inv(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr, self.qr2[1]]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1, -3, -2])

    def test_cu3_bit_reg(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr[1], self.qr2])
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [1, 2, 3])

    def test_cu3_bit_reg_inv(self):
        instruction_set = self.circuit.append(CU3Gate(1, 2, 3), [self.qr[1], self.qr2]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cu3")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [-1, -3, -2])

    def test_cx_reg_reg(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cx_reg_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cx_reg_bit(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cx_reg_bit_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cx_bit_reg(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cx_bit_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_reg_reg(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_reg_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_reg_bit(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_reg_bit_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_bit_reg(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cy_bit_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cy")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_reg_reg(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_reg_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_reg_bit(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1])
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_reg_bit_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_bit_reg(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cz_bit_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cz")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_swap_reg_reg(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2)
        self.assertEqual(instruction_set[0].operation.name, "swap")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_swap_reg_reg_inv(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set[0].operation.name, "swap")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    @unpack
    @data(
        (0, 0, np.eye(4)),
        (
            np.pi / 2,
            np.pi / 2,
            np.array(
                [
                    [np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                ]
            ),
        ),
        (
            np.pi,
            np.pi / 2,
            np.array([[0, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]),
        ),
        (
            2 * np.pi,
            np.pi / 2,
            np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        ),
        (
            np.pi / 2,
            np.pi,
            np.array(
                [
                    [np.sqrt(2) / 2, 0, 0, 1j * np.sqrt(2) / 2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1j * np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                ]
            ),
        ),
        (4 * np.pi, 0, np.eye(4)),
    )
    def test_xx_minus_yy_matrix(self, theta: float, beta: float, expected: np.ndarray):
        """Test XX-YY matrix."""
        gate = XXMinusYYGate(theta, beta)
        np.testing.assert_allclose(np.array(gate), expected, atol=1e-7)

    def test_xx_minus_yy_exponential_formula(self):
        """Test XX-YY exponential formula."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        gate = XXMinusYYGate(theta, beta)
        x = np.array(XGate())
        y = np.array(YGate())
        xx = np.kron(x, x)
        yy = np.kron(y, y)
        rz1 = np.kron(np.array(RZGate(beta)), np.eye(2))
        np.testing.assert_allclose(
            np.array(gate),
            rz1 @ expm(-0.25j * theta * (xx - yy)) @ rz1.T.conj(),
            atol=1e-7,
        )

    def test_xx_plus_yy_exponential_formula(self):
        """Test XX+YY exponential formula."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        gate = XXPlusYYGate(theta, beta)
        x = np.array(XGate())
        y = np.array(YGate())
        xx = np.kron(x, x)
        yy = np.kron(y, y)
        rz0 = np.kron(np.eye(2), np.array(RZGate(beta)))
        np.testing.assert_allclose(
            np.array(gate),
            rz0.T.conj() @ expm(-0.25j * theta * (xx + yy)) @ rz0,
            atol=1e-7,
        )


class TestStandard3Q(QiskitTestCase):
    """Standard Extension Test. Gates with three Qubits"""

    def setUp(self):
        super().setUp()
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.qr3 = QuantumRegister(3, "s")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.qr3, self.cr)

    def test_ccx_reg_reg_reg(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set[0].operation.name, "ccx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1], self.qr3[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_ccx_reg_reg_inv(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set[0].operation.name, "ccx")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1], self.qr3[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cswap_reg_reg_reg(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set[0].operation.name, "cswap")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1], self.qr3[1]))
        self.assertEqual(instruction_set[2].operation.params, [])

    def test_cswap_reg_reg_inv(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set[0].operation.name, "cswap")
        self.assertEqual(instruction_set[1].qubits, (self.qr[1], self.qr2[1], self.qr3[1]))
        self.assertEqual(instruction_set[2].operation.params, [])


class TestStandardMethods(QiskitTestCase):
    """Standard Extension Test."""

    def test_to_matrix(self):
        """test gates implementing to_matrix generate matrix which matches definition."""
        from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate

        params = [0.1 * (i + 1) for i in range(10)]
        gate_class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        for gate_class in gate_class_list:
            if hasattr(gate_class, "__abstractmethods__"):
                # gate_class is abstract
                continue
            sig = signature(gate_class)
            free_params = len(set(sig.parameters) - {"label", "ctrl_state"})
            try:
                if gate_class == PauliGate:
                    # special case due to PauliGate using string parameters
                    gate = gate_class("IXYZ")
                elif gate_class == PauliEvolutionGate:
                    gate = gate_class(Pauli("XYZ"))
                else:
                    gate = gate_class(*params[0:free_params])
            except (CircuitError, QiskitError, AttributeError, TypeError):
                self.log.info("Cannot init gate with params only. Skipping %s", gate_class)
                continue
            if gate.name in ["U", "CX"]:
                continue
            circ = QuantumCircuit(gate.num_qubits)
            circ.append(gate, range(gate.num_qubits))
            try:
                gate_matrix = gate.to_matrix()
            except CircuitError:
                # gate doesn't implement to_matrix method: skip
                self.log.info('to_matrix method FAILED for "%s" gate', gate.name)
                continue
            definition_unitary = Operator(circ)

            with self.subTest(gate_class):
                # TODO check for exact equality
                self.assertTrue(matrix_equal(definition_unitary, gate_matrix, ignore_phase=True))
                self.assertTrue(is_unitary_matrix(gate_matrix))

    def test_to_matrix_op(self):
        """test gates implementing to_matrix generate matrix which matches
        definition using Operator."""
        from qiskit.circuit.library.generalized_gates.gms import MSGate
        from qiskit.circuit.library.generalized_gates.pauli import PauliGate
        from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate

        params = [0.1 * i for i in range(1, 11)]
        gate_class_list = Gate.__subclasses__() + ControlledGate.__subclasses__()
        for gate_class in gate_class_list:
            if hasattr(gate_class, "__abstractmethods__"):
                # gate_class is abstract
                continue
            sig = signature(gate_class)
            if gate_class == MSGate:
                # due to the signature (num_qubits, theta, *, n_qubits=None) the signature detects
                # 3 arguments but really its only 2. This if can be removed once the deprecated
                # n_qubits argument is no longer supported.
                free_params = 2
            else:
                free_params = len(set(sig.parameters) - {"label", "ctrl_state"})
            try:
                if gate_class == PauliGate:
                    # special case due to PauliGate using string parameters
                    gate = gate_class("IXYZ")
                elif gate_class == PauliEvolutionGate:
                    gate = gate_class(Pauli("XYZ"))
                else:
                    gate = gate_class(*params[0:free_params])
            except (CircuitError, QiskitError, AttributeError, TypeError):
                self.log.info("Cannot init gate with params only. Skipping %s", gate_class)
                continue
            if gate.name in ["U", "CX"]:
                continue
            try:
                gate_matrix = gate.to_matrix()
            except CircuitError:
                # gate doesn't implement to_matrix method: skip
                self.log.info('to_matrix method FAILED for "%s" gate', gate.name)
                continue
            if not hasattr(gate, "definition") or not gate.definition:
                continue
            definition_unitary = Operator(gate.definition).data
            self.assertTrue(matrix_equal(definition_unitary, gate_matrix))
            self.assertTrue(is_unitary_matrix(gate_matrix))


if __name__ == "__main__":
    unittest.main(verbosity=2)
