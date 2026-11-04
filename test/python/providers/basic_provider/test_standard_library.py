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


import unittest

from qiskit import QuantumCircuit
from qiskit.providers.basic_provider import BasicSimulator
import qiskit.circuit.library.standard_gates as lib
from test import QiskitTestCase


class TestStandardGates(QiskitTestCase):
    """Standard gates support in BasicSimulator, up to 3 qubits"""

    def setUp(self):
        super().setUp()
        self.seed = 43
        self.shots = 1
        self.circuit = QuantumCircuit(4)

    def test_barrier(self):
        self.circuit.barrier(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_barrier_none(self):
        self.circuit.barrier()
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_unitary(self):
        matrix = [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
        self.circuit.unitary(matrix, [0, 1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u(self):
        self.circuit.u(0.5, 1.5, 1.5, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u1(self):
        self.circuit.append(lib.U1Gate(0.5), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u2(self):
        self.circuit.append(lib.U2Gate(0.5, 0.5), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_u3(self):
        self.circuit.append(lib.U3Gate(0.5, 0.5, 0.5), [1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ccx(self):
        self.circuit.ccx(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ccz(self):
        self.circuit.ccz(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ch(self):
        self.circuit.ch(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cp(self):
        self.circuit.cp(0, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_crx(self):
        self.circuit.crx(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cry(self):
        self.circuit.cry(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_crz(self):
        self.circuit.crz(1, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cswap(self):
        self.circuit.cswap(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu1(self):
        self.circuit.append(lib.CU1Gate(1), [1, 2])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu3(self):
        self.circuit.append(lib.CU3Gate(1, 2, 3), [1, 2])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cx(self):
        self.circuit.cx(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ecr(self):
        self.circuit.ecr(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cy(self):
        self.circuit.cy(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cz(self):
        self.circuit.cz(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_h(self):
        self.circuit.h(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_id(self):
        self.circuit.id(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rx(self):
        self.circuit.rx(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ry(self):
        self.circuit.ry(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rz(self):
        self.circuit.rz(1, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rxx(self):
        self.circuit.rxx(1, 1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rzx(self):
        self.circuit.rzx(1, 1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_ryy(self):
        self.circuit.ryy(1, 1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rzz(self):
        self.circuit.rzz(1, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_s(self):
        self.circuit.s(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sdg(self):
        self.circuit.sdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sx(self):
        self.circuit.sx(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_sxdg(self):
        self.circuit.sxdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_swap(self):
        self.circuit.swap(1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_iswap(self):
        self.circuit.iswap(1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_p(self):
        self.circuit.p(1, 0)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_r(self):
        self.circuit.r(0.5, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_t(self):
        self.circuit.t(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_tdg(self):
        self.circuit.tdg(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_x(self):
        self.circuit.x(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_y(self):
        self.circuit.y(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_z(self):
        self.circuit.z(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cs(self):
        self.circuit.cs(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_csdg(self):
        self.circuit.csdg(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_csx(self):
        self.circuit.csx(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_cu(self):
        self.circuit.cu(0.5, 0.5, 0.5, 0.5, 0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_dcx(self):
        self.circuit.dcx(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_delay(self):
        self.circuit.delay(0, 1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_reset(self):
        self.circuit.reset(1)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_rcx(self):
        self.circuit.rccx(0, 1, 2)
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_global_phase(self):
        qc = self.circuit
        qc.append(lib.GlobalPhaseGate(0.1), [])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_xx_minus_yy(self):
        self.circuit.append(lib.XXMinusYYGate(0.1, 0.2), [0, 1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)

    def test_xx_plus_yy(self):
        self.circuit.append(lib.XXPlusYYGate(0.1, 0.2), [0, 1])
        self.circuit.measure_all()
        result = (
            BasicSimulator().run(self.circuit, shots=self.shots, seed_simulator=self.seed).result()
        )
        self.assertEqual(result.success, True)


class TestStandardGatesTarget(QiskitTestCase):
    """Standard gates, up to 3 qubits, as a target"""

    def test_target(self):
        target = BasicSimulator().target
        expected = {
            "cz",
            "u3",
            "p",
            "cswap",
            "z",
            "cu1",
            "ecr",
            "reset",
            "ch",
            "cy",
            "dcx",
            "crx",
            "sx",
            "unitary",
            "csdg",
            "rzz",
            "measure",
            "swap",
            "csx",
            "y",
            "s",
            "xx_plus_yy",
            "cs",
            "h",
            "t",
            "u",
            "rxx",
            "cu",
            "rzx",
            "ry",
            "rx",
            "cu3",
            "tdg",
            "u2",
            "xx_minus_yy",
            "global_phase",
            "u1",
            "id",
            "cx",
            "cp",
            "rz",
            "sxdg",
            "x",
            "ryy",
            "sdg",
            "ccz",
            "delay",
            "crz",
            "iswap",
            "ccx",
            "cry",
            "rccx",
            "r",
        }
        self.assertEqual(set(target.operation_names), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
