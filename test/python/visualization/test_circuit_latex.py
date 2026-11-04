# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for visualization of circuit with Latex drawer."""

import os
import unittest
import math
import numpy as np

from qiskit.visualization import circuit_drawer
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.circuit.library import (
    XGate,
    MCXGate,
    RZZGate,
    SwapGate,
    DCXGate,
    HamiltonianGate,
    Isometry,
    iqp,
)
from qiskit.circuit import Parameter, Qubit, Clbit
from qiskit.quantum_info import random_unitary
from qiskit.utils import optionals
from .visualization import QiskitVisualizationTestCase
from ..legacy_cmaps import YORKTOWN_CMAP

pi = np.pi


@unittest.skipUnless(optionals.HAS_PYLATEX, "needs pylatexenc")
class TestLatexSourceGenerator(QiskitVisualizationTestCase):
    """Qiskit latex source generator tests."""

    def _get_resource_path(self, filename):
        reference_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(reference_dir, filename)

    def test_empty_circuit(self):
        """Test draw an empty circuit"""
        filename = self._get_resource_path("test_latex_empty.tex")
        circuit = QuantumCircuit(1)
        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_tiny_circuit(self):
        """Test draw tiny circuit."""
        filename = self._get_resource_path("test_latex_tiny.tex")
        circuit = QuantumCircuit(1)
        circuit.h(0)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_multi_underscore_reg_names(self):
        """Test multi-underscores in register names display properly"""
        filename1 = self._get_resource_path("test_latex_multi_underscore_true.tex")
        filename2 = self._get_resource_path("test_latex_multi_underscore_false.tex")
        q_reg1 = QuantumRegister(1, "q1_re__g__g")
        q_reg3 = QuantumRegister(3, "q3_re_g__g")
        c_reg1 = ClassicalRegister(1, "c1_re_g__g")
        c_reg3 = ClassicalRegister(3, "c3_re_g__g")
        circuit = QuantumCircuit(q_reg1, q_reg3, c_reg1, c_reg3)
        circuit_drawer(circuit, cregbundle=True, filename=filename1, output="latex_source")
        circuit_drawer(circuit, cregbundle=False, filename=filename2, output="latex_source")
        self.assertEqualToReference(filename1)
        self.assertEqualToReference(filename2)

    def test_normal_circuit(self):
        """Test draw normal size circuit."""
        filename = self._get_resource_path("test_latex_normal.tex")
        circuit = QuantumCircuit(5)
        for qubit in range(5):
            circuit.h(qubit)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_deep_circuit(self):
        """Test draw deep circuit."""
        filename = self._get_resource_path("test_latex_deep.tex")
        circuit = QuantumCircuit(1)
        for _ in range(100):
            circuit.h(0)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_huge_circuit(self):
        """Test draw huge circuit."""
        filename = self._get_resource_path("test_latex_huge.tex")
        circuit = QuantumCircuit(40)
        for qubit in range(39):
            circuit.h(qubit)
            circuit.cx(qubit, 39)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_global_phase(self):
        """Test circuit with global phase"""
        filename = self._get_resource_path("test_latex_global_phase.tex")
        circuit = QuantumCircuit(3, global_phase=1.57079632679)
        circuit.h(range(3))

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_no_ops(self):
        """Test circuit with no ops.
        See https://github.com/Qiskit/qiskit-terra/issues/5393"""
        filename = self._get_resource_path("test_latex_no_ops.tex")
        circuit = QuantumCircuit(2, 3)
        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """
        filename = self._get_resource_path("test_latex_long_name.tex")

        # add a register with a very long name
        qr = QuantumRegister(4, "veryLongQuantumRegisterName")
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, "q0")
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_plot_partial_barrier(self):
        """Test plotting of partial barriers."""

        filename = self._get_resource_path("test_latex_plot_partial_barriers.tex")
        # generate a circuit with barrier and other barrier like instructions in
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier(0)
        circuit.h(q[0])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_plot_barriers(self):
        """Test to see that plotting barriers works.
        If it is set to False, no blank columns are introduced"""

        filename1 = self._get_resource_path("test_latex_plot_barriers_true.tex")
        filename2 = self._get_resource_path("test_latex_plot_barriers_false.tex")
        # generate a circuit with barriers and other barrier like instructions in
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier()

        # check for other barrier like commands
        circuit.h(q[1])

        # check the barriers plot properly when plot_barriers= True
        circuit_drawer(circuit, filename=filename1, output="latex_source", plot_barriers=True)

        self.assertEqualToReference(filename1)
        circuit_drawer(circuit, filename=filename2, output="latex_source", plot_barriers=False)

        self.assertEqualToReference(filename2)

    def test_no_barriers_false(self):
        """Generate the same circuit as test_plot_barriers but without the barrier commands
        as this is what the circuit should look like when displayed with plot barriers false"""
        filename = self._get_resource_path("test_latex_no_barriers_false.tex")
        q1 = QuantumRegister(2, "q")
        c1 = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q1, c1)
        circuit.h(q1[0])
        circuit.h(q1[1])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_barrier_label(self):
        """Test the barrier label"""
        filename = self._get_resource_path("test_latex_barrier_label.tex")
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.y(1)
        circuit.barrier()
        circuit.y(0)
        circuit.x(1)
        circuit.barrier(label="End Y/X")

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_barrier_label_truncation(self):
        """Test that long barrier labels are truncated"""
        filename = self._get_resource_path("test_latex_barrier_label_truncation.tex")
        circuit = QuantumCircuit(2)
        circuit.barrier()
        circuit.barrier(label="a" * 20)

        circuit_drawer(circuit, filename=filename, output="latex_source", barrier_label_len=9)

        self.assertEqualToReference(filename)

    def test_big_gates(self):
        """Test large gates with params"""
        filename = self._get_resource_path("test_latex_big_gates.tex")
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(iqp([[6, 5, 3], [5, 4, 5], [3, 5, 1]]), [0, 1, 2])

        desired_vector = [
            1 / math.sqrt(16) * complex(0, 1),
            1 / math.sqrt(8) * complex(1, 0),
            1 / math.sqrt(16) * complex(1, 1),
            0,
            0,
            1 / math.sqrt(8) * complex(1, 2),
            1 / math.sqrt(16) * complex(1, 0),
            0,
        ]

        circuit.initialize(desired_vector, [qr[3], qr[4], qr[5]])
        circuit.unitary([[1, 0], [0, 1]], [qr[0]])
        matrix = np.zeros((4, 4))
        theta = Parameter("theta")
        circuit.append(HamiltonianGate(matrix, theta), [qr[1], qr[2]])
        circuit = circuit.assign_parameters({theta: 1})
        circuit.append(Isometry(np.eye(4, 4), 0, 0), list(range(3, 5)))

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_cnot(self):
        """Test different cnot gates (ccnot, mcx, etc)"""
        filename = self._get_resource_path("test_latex_cnot.tex")
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.append(
            XGate().control(3, ctrl_state="010", annotated=False), [qr[2], qr[3], qr[0], qr[1]]
        )
        circuit.append(MCXGate(num_ctrl_qubits=3, ctrl_state="101"), [qr[0], qr[1], qr[2], qr[4]])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_pauli_clifford(self):
        """Test Pauli(green) and Clifford(blue) gates"""
        filename = self._get_resource_path("test_latex_pauli_clifford.tex")
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.y(0)
        circuit.z(0)
        circuit.id(0)
        circuit.h(1)
        circuit.cx(1, 2)
        circuit.cy(1, 2)
        circuit.cz(1, 2)
        circuit.swap(3, 4)
        circuit.s(3)
        circuit.sdg(3)
        circuit.iswap(3, 4)
        circuit.dcx(3, 4)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_u_gates(self):
        """Test U 1, 2, & 3 gates"""
        filename = self._get_resource_path("test_latex_u_gates.tex")
        from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate

        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.append(U1Gate(3 * pi / 2), [0])
        circuit.append(U2Gate(3 * pi / 2, 2 * pi / 3), [1])
        circuit.append(U3Gate(3 * pi / 2, 4.5, pi / 4), [2])
        circuit.append(CU1Gate(pi / 4), [0, 1])
        circuit.append(U2Gate(pi / 2, 3 * pi / 2).control(1, annotated=False), [2, 3])
        circuit.append(CU3Gate(3 * pi / 2, -3 * pi / 4, -pi / 2), [0, 1])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_creg_initial(self):
        """Test cregbundle and initial state options"""
        filename1 = self._get_resource_path("test_latex_creg_initial_true.tex")
        filename2 = self._get_resource_path("test_latex_creg_initial_false.tex")
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.h(0)
        circuit.x(1)

        circuit_drawer(
            circuit, filename=filename1, output="latex_source", cregbundle=True, initial_state=True
        )

        self.assertEqualToReference(filename1)
        circuit_drawer(
            circuit,
            filename=filename2,
            output="latex_source",
            cregbundle=False,
            initial_state=False,
        )

        self.assertEqualToReference(filename2)

    def test_r_gates(self):
        """Test all R gates"""
        filename = self._get_resource_path("test_latex_r_gates.tex")
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.r(3 * pi / 4, 3 * pi / 8, 0)
        circuit.rx(pi / 2, 1)
        circuit.ry(-pi / 2, 2)
        circuit.rz(3 * pi / 4, 3)
        circuit.rxx(pi / 2, 0, 1)
        circuit.ryy(3 * pi / 4, 2, 3)
        circuit.rzx(-pi / 2, 0, 1)
        circuit.rzz(pi / 2, 2, 3)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_cswap_rzz(self):
        """Test controlled swap and rzz gates"""
        filename = self._get_resource_path("test_latex_cswap_rzz.tex")
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.x(1)
        circuit.cswap(0, 1, 2)
        circuit.append(
            RZZGate(3 * pi / 4).control(3, ctrl_state="010", annotated=False), [2, 1, 4, 3, 0]
        )

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_ghz_to_gate(self):
        """Test controlled GHZ to_gate circuit"""
        filename = self._get_resource_path("test_latex_ghz_to_gate.tex")
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        ghz_circuit = QuantumCircuit(3, name="Ctrl-GHZ Circuit")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2, ctrl_state="10", annotated=False)
        circuit.append(ccghz, [4, 0, 1, 3, 2])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_scale(self):
        """Tests scale
        See: https://github.com/Qiskit/qiskit-terra/issues/4179"""
        filename1 = self._get_resource_path("test_latex_scale_default.tex")
        filename2 = self._get_resource_path("test_latex_scale_half.tex")
        filename3 = self._get_resource_path("test_latex_scale_double.tex")
        circuit = QuantumCircuit(5)
        circuit.unitary(random_unitary(2**5), circuit.qubits)

        circuit_drawer(circuit, filename=filename1, output="latex_source")

        self.assertEqualToReference(filename1)
        circuit_drawer(circuit, filename=filename2, output="latex_source", scale=0.5)

        self.assertEqualToReference(filename2)
        circuit_drawer(circuit, filename=filename3, output="latex_source", scale=2.0)

        self.assertEqualToReference(filename3)

    def test_pi_param_expr(self):
        """Text pi in circuit with parameter expression."""
        filename = self._get_resource_path("test_latex_pi_param_expr.tex")
        x, y = Parameter("x"), Parameter("y")
        circuit = QuantumCircuit(1)
        circuit.rx((pi - x) * (pi - y), 0)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_partial_layout(self):
        """Tests partial_layout
        See: https://github.com/Qiskit/qiskit-terra/issues/4757"""
        filename = self._get_resource_path("test_latex_partial_layout.tex")

        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=YORKTOWN_CMAP,
            basis_gates=["id", "rz", "sx", "x", "cx", "reset"],
            seed=42,
        )

        circuit = QuantumCircuit(3)
        circuit.h(1)
        transpiled = transpile(
            circuit,
            backend=backend,
            optimization_level=0,
            initial_layout=[1, 2, 0],
            seed_transpiler=0,
        )

        circuit_drawer(transpiled, filename=filename, output="latex_source", idle_wires=True)

        self.assertEqualToReference(filename)

    def test_init_reset(self):
        """Test reset and initialize with 1 and 2 qubits"""
        filename = self._get_resource_path("test_latex_init_reset.tex")
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], 0)
        circuit.reset(1)
        circuit.initialize([0, 1, 0, 0], [0, 1])

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_iqx_colors(self):
        """Tests with iqx color scheme"""
        filename = self._get_resource_path("test_latex_iqx.tex")
        circuit = QuantumCircuit(7)
        circuit.h(0)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.swap(0, 1)
        circuit.cswap(0, 1, 2)
        circuit.append(SwapGate().control(2, annotated=False), [0, 1, 2, 3])
        circuit.dcx(0, 1)
        circuit.append(DCXGate().control(1, annotated=False), [0, 1, 2])
        circuit.append(DCXGate().control(2, annotated=False), [0, 1, 2, 3])
        circuit.z(4)
        circuit.s(4)
        circuit.sdg(4)
        circuit.t(4)
        circuit.tdg(4)
        circuit.p(pi / 2, 4)
        circuit.p(pi / 2, 4)
        circuit.cz(5, 6)
        circuit.cp(pi / 2, 5, 6)
        circuit.y(5)
        circuit.rx(pi / 3, 5)
        circuit.rzx(pi / 2, 5, 6)
        circuit.u(pi / 2, pi / 2, pi / 2, 5)
        circuit.barrier(5, 6)
        circuit.reset(5)

        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_reverse_bits(self):
        """Tests reverse_bits parameter"""
        filename = self._get_resource_path("test_latex_reverse_bits.tex")
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ccx(2, 1, 0)

        circuit_drawer(circuit, filename=filename, output="latex_source", reverse_bits=True)

        self.assertEqualToReference(filename)

    def test_inst_with_cbits(self):
        """Test custom instructions with classical bits"""

        filename = self._get_resource_path("test_latex_inst_with_cbits.tex")
        qinst = QuantumRegister(2, "q")
        cinst = ClassicalRegister(2, "c")
        inst = QuantumCircuit(qinst, cinst, name="instruction").to_instruction()

        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(4, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.append(inst, [qr[1], qr[2]], [cr[2], cr[1]])
        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_registerless_one_bit(self):
        """Text circuit with one-bit registers and registerless bits."""
        filename = self._get_resource_path("test_latex_registerless_one_bit.tex")
        qrx = QuantumRegister(2, "qrx")
        qry = QuantumRegister(1, "qry")
        crx = ClassicalRegister(2, "crx")
        circuit = QuantumCircuit(qrx, [Qubit(), Qubit()], qry, [Clbit(), Clbit()], crx)
        circuit_drawer(circuit, filename=filename, output="latex_source")

        self.assertEqualToReference(filename)

    def test_idle_wires_barrier(self):
        """Test that idle_wires False works with barrier"""
        filename = self._get_resource_path("test_latex_idle_wires_barrier.tex")
        circuit = QuantumCircuit(4, 4)
        circuit.x(2)
        circuit.barrier()
        circuit_drawer(circuit, idle_wires=False, filename=filename, output="latex_source")
        self.assertEqualToReference(filename)


if __name__ == "__main__":
    unittest.main(verbosity=2)
