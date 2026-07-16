# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for circuit MPL drawer"""

import unittest

import os
import math
from pathlib import Path
import numpy as np
from numpy import pi

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.visualization.circuit.circuit_visualization import circuit_drawer
from qiskit.circuit.library import (
    XGate,
    MCXGate,
    HGate,
    RZZGate,
    SwapGate,
    DCXGate,
    ZGate,
    SGate,
    SXGate,
    HamiltonianGate,
    Isometry,
    iqp,
)
from qiskit.circuit.library import MCXVChain
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit import Parameter, Qubit, Clbit, IfElseOp, SwitchCaseOp
from qiskit.circuit.classical import expr, types
from qiskit.quantum_info import random_clifford
from qiskit.quantum_info import random_unitary
from qiskit.utils import optionals
from test.visual import VisualTestUtilities
from test import QiskitTestCase
from test.python.legacy_cmaps import (
    TENERIFE_CMAP,
    YORKTOWN_CMAP,
)

if optionals.HAS_MATPLOTLIB:
    from matplotlib.pyplot import close as mpl_close
else:
    raise ImportError('Must have Matplotlib installed. To install, run "pip install matplotlib".')

BASE_DIR = Path(__file__).parent
RESULT_DIR = Path(BASE_DIR) / "circuit_results"
TEST_REFERENCE_DIR = Path(BASE_DIR) / "references"
FAILURE_DIFF_DIR = Path(BASE_DIR).parent / "visual_test_failures"
FAILURE_PREFIX = "circuit_failure_"


class TestCircuitMatplotlibDrawer(QiskitTestCase):
    """Circuit MPL visualization"""

    def setUp(self):
        super().setUp()
        self.threshold = 0.9999
        self.circuit_drawer = VisualTestUtilities.save_data_wrap(
            circuit_drawer, str(self), RESULT_DIR
        )

        if not os.path.exists(FAILURE_DIFF_DIR):
            os.makedirs(FAILURE_DIFF_DIR)

        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

    def tearDown(self):
        super().tearDown()
        mpl_close("all")

    @staticmethod
    def _image_path(image_name):
        return os.path.join(RESULT_DIR, image_name)

    @staticmethod
    def _reference_path(image_name):
        return os.path.join(TEST_REFERENCE_DIR, image_name)

    def test_empty_circuit(self):
        """Test empty circuit"""
        circuit = QuantumCircuit()

        fname = "empty_circut.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_no_ops(self):
        """Test circuit with no ops.
        See https://github.com/Qiskit/qiskit-terra/issues/5393"""
        circuit = QuantumCircuit(2, 3)

        fname = "no_op_circut.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_long_name(self):
        """Test to see that long register names can be seen completely
        As reported in #2605
        """

        # add a register with a very long name
        qr = QuantumRegister(4, "veryLongQuantumRegisterName")
        # add another to make sure adjustments are made based on longest
        qrr = QuantumRegister(1, "q0")
        circuit = QuantumCircuit(qr, qrr)

        # check gates are shifted over accordingly
        circuit.h(qr)
        circuit.h(qr)
        circuit.h(qr)

        fname = "long_name.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_multi_underscore_reg_names(self):
        """Test that multi-underscores in register names display properly"""
        q_reg1 = QuantumRegister(1, "q1_re__g__g")
        q_reg3 = QuantumRegister(3, "q3_re_g__g")
        c_reg1 = ClassicalRegister(1, "c1_re_g__g")
        c_reg3 = ClassicalRegister(3, "c3_re_g__g")
        circuit = QuantumCircuit(q_reg1, q_reg3, c_reg1, c_reg3)

        fname = "multi_underscore_true.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=True, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        fname2 = "multi_underscore_false.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname2)

        ratio2 = VisualTestUtilities._save_diff(
            self._image_path(fname2),
            self._reference_path(fname2),
            fname2,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        self.assertGreaterEqual(ratio, self.threshold)
        self.assertGreaterEqual(ratio2, self.threshold)

    def test_plot_partial_barrier(self):
        """Test plotting of partial barriers."""

        # generate a circuit with barrier and other barrier like instructions in
        q = QuantumRegister(2, "q")
        c = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q, c)

        # check for barriers
        circuit.h(q[0])
        circuit.barrier(0)
        circuit.h(q[0])

        fname = "plot_partial_barrier.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname, plot_barriers=True)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_plot_barriers(self):
        """Test to see that plotting barriers works.
        If it is set to False, no blank columns are introduced"""

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
        fname = "plot_barriers_true.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname, plot_barriers=True)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        fname2 = "plot_barriers_false.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname2, plot_barriers=False)

        ratio2 = VisualTestUtilities._save_diff(
            self._image_path(fname2),
            self._reference_path(fname2),
            fname2,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        self.assertGreaterEqual(ratio, self.threshold)
        self.assertGreaterEqual(ratio2, self.threshold)

    def test_no_barriers_false(self):
        """Generate the same circuit as test_plot_barriers but without the barrier commands
        as this is what the circuit should look like when displayed with plot barriers false"""
        q1 = QuantumRegister(2, "q")
        c1 = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(q1, c1)
        circuit.h(q1[0])
        circuit.h(q1[1])

        fname = "no_barriers.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname, plot_barriers=False)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_fold_minus1(self):
        """Test to see that fold=-1 is no folding"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        fname = "fold_minus1.png"
        self.circuit_drawer(circuit, output="mpl", fold=-1, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_fold_4(self):
        """Test to see that fold=4 is folding"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        for _ in range(3):
            circuit.h(0)
            circuit.x(0)

        fname = "fold_4.png"
        self.circuit_drawer(circuit, output="mpl", fold=4, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_big_gates(self):
        """Test large gates with params"""
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

        fname = "big_gates.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_cnot(self):
        """Test different cnot gates (ccnot, mcx, etc)"""
        qr = QuantumRegister(6, "q")
        circuit = QuantumCircuit(qr)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)
        circuit.append(
            XGate().control(3, ctrl_state="010", annotated=False), [qr[2], qr[3], qr[0], qr[1]]
        )
        circuit.append(
            MCXGate(num_ctrl_qubits=3, ctrl_state="101"),
            [qr[0], qr[1], qr[2], qr[4]],
        )
        with self.assertWarns(DeprecationWarning):
            circuit.append(MCXVChain(3, dirty_ancillas=True), [qr[0], qr[1], qr[2], qr[3], qr[5]])

        fname = "cnot.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_cz(self):
        """Test Z and Controlled-Z Gates"""
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.z(0)
        circuit.cz(0, 1)
        circuit.append(ZGate().control(3, ctrl_state="101", annotated=False), [0, 1, 2, 3])
        circuit.append(ZGate().control(2, annotated=False), [1, 2, 3])
        circuit.append(ZGate().control(1, ctrl_state="0", annotated=False, label="CZ Gate"), [2, 3])

        fname = "cz.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_pauli_clifford(self):
        """Test Pauli(green) and Clifford(blue) gates"""
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

        fname = "pauli_clifford.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_creg_initial(self):
        """Test cregbundle and initial state options"""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.h(0)
        circuit.x(1)

        fname = "creg_initial_true.png"
        self.circuit_drawer(
            circuit, output="mpl", filename=fname, cregbundle=True, initial_state=True
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        fname2 = "creg_initial_false.png"
        self.circuit_drawer(
            circuit, output="mpl", filename=fname2, cregbundle=False, initial_state=False
        )

        ratio2 = VisualTestUtilities._save_diff(
            self._image_path(fname2),
            self._reference_path(fname2),
            fname2,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        self.assertGreaterEqual(ratio, self.threshold)
        self.assertGreaterEqual(ratio2, self.threshold)

    def test_r_gates(self):
        """Test all R gates"""
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

        fname = "r_gates.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_ctrl_labels(self):
        """Test control labels"""
        qr = QuantumRegister(4, "q")
        circuit = QuantumCircuit(qr)
        circuit.cy(1, 0, label="Bottom Y label")
        circuit.cu(pi / 2, pi / 2, pi / 2, 0, 2, 3, label="Top U label")
        circuit.ch(0, 1, label="Top H label")
        circuit.append(
            HGate(label="H gate label").control(
                3, label="H control label", ctrl_state="010", annotated=False
            ),
            [qr[1], qr[2], qr[3], qr[0]],
        )

        fname = "ctrl_labels.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_cswap_rzz(self):
        """Test controlled swap and rzz gates"""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        circuit.cswap(0, 1, 2)
        circuit.append(
            RZZGate(3 * pi / 4).control(3, ctrl_state="010", annotated=False), [2, 1, 4, 3, 0]
        )

        fname = "cswap_rzz.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_ghz_to_gate(self):
        """Test controlled GHZ to_gate circuit"""
        qr = QuantumRegister(5, "q")
        circuit = QuantumCircuit(qr)
        ghz_circuit = QuantumCircuit(3, name="this is a WWWWWWWWWWWide name Ctrl-GHZ Circuit")
        ghz_circuit.h(0)
        ghz_circuit.cx(0, 1)
        ghz_circuit.cx(1, 2)
        ghz = ghz_circuit.to_gate()
        ccghz = ghz.control(2, ctrl_state="10", annotated=False)
        circuit.append(ccghz, [4, 0, 1, 3, 2])

        fname = "ghz_to_gate.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_scale(self):
        """Tests scale
        See: https://github.com/Qiskit/qiskit-terra/issues/4179"""
        circuit = QuantumCircuit(5)
        circuit.unitary(random_unitary(2**5), circuit.qubits)

        fname = "scale_default.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        fname2 = "scale_half.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname2, scale=0.5)

        ratio2 = VisualTestUtilities._save_diff(
            self._image_path(fname2),
            self._reference_path(fname2),
            fname2,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        fname3 = "scale_double.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname3, scale=2)

        ratio3 = VisualTestUtilities._save_diff(
            self._image_path(fname3),
            self._reference_path(fname3),
            fname3,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        self.assertGreaterEqual(ratio, self.threshold)
        self.assertGreaterEqual(ratio2, self.threshold)
        self.assertGreaterEqual(ratio3, self.threshold)

    def test_pi_param_expr(self):
        """Test pi in circuit with parameter expression."""
        x, y = Parameter("x"), Parameter("y")
        circuit = QuantumCircuit(1)
        circuit.rx((pi - x) * (pi - y), 0)

        fname = "pi_in_param_expr.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_partial_layout(self):
        """Tests partial_layout
        See: https://github.com/Qiskit/qiskit-terra/issues/4757"""
        circuit = QuantumCircuit(3)
        circuit.h(1)
        transpiled = transpile(
            circuit,
            backend=GenericBackendV2(5, coupling_map=TENERIFE_CMAP),
            basis_gates=["id", "cx", "rz", "sx", "x"],
            optimization_level=0,
            initial_layout=[1, 2, 0],
            seed_transpiler=0,
        )

        fname = "partial_layout.png"
        self.circuit_drawer(transpiled, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_init_reset(self):
        """Test reset and initialize with 1 and 2 qubits"""
        circuit = QuantumCircuit(2)
        circuit.initialize([0, 1], 0)
        circuit.reset(1)
        circuit.initialize([0, 1, 0, 0], [0, 1])

        fname = "init_reset.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_with_global_phase(self):
        """Tests with global phase"""
        circuit = QuantumCircuit(3, global_phase=1.57079632679)
        circuit.h(range(3))

        fname = "global_phase.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_alternative_colors(self):
        """Tests alternative color schemes"""
        for style in ["iqp", "iqp-dark", "textbook", "clifford"]:
            with self.subTest(style=style):
                circuit = QuantumCircuit(7)
                circuit.h(0)
                circuit.x(0)
                circuit.cx(0, 1)
                circuit.ccx(0, 1, 2)
                circuit.swap(0, 1)
                circuit.iswap(2, 3)
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
                circuit.cz(5, 6)
                circuit.cp(pi / 2, 5, 6)
                circuit.mcp(pi / 5, [0, 1, 2, 3], 4)
                circuit.y(5)
                circuit.rx(pi / 3, 5)
                circuit.rz(pi / 6, 6)
                circuit.rzx(pi / 2, 5, 6)
                circuit.rzz(pi / 4, 5, 6)
                circuit.u(pi / 2, pi / 2, pi / 2, 5)
                circuit.barrier(5, 6)
                circuit.reset(5)

                fname = f"{style}_color.png"
                self.circuit_drawer(circuit, output="mpl", style={"name": style}, filename=fname)

                ratio = VisualTestUtilities._save_diff(
                    self._image_path(fname),
                    self._reference_path(fname),
                    fname,
                    FAILURE_DIFF_DIR,
                    FAILURE_PREFIX,
                )
                self.assertGreaterEqual(ratio, self.threshold)

    def test_reverse_bits(self):
        """Tests reverse_bits parameter"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.ccx(2, 1, 0)

        fname = "reverse_bits.png"
        self.circuit_drawer(circuit, output="mpl", reverse_bits=True, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_bw(self):
        """Tests black and white style parameter"""
        circuit = QuantumCircuit(3, 3)
        circuit.h(0)
        circuit.x(1)
        circuit.sdg(2)
        circuit.cx(0, 1)
        circuit.ccx(2, 1, 0)
        circuit.swap(1, 2)
        circuit.measure_all()

        fname = "bw.png"
        self.circuit_drawer(circuit, output="mpl", style={"name": "bw"}, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_user_style(self):
        """Tests loading a user style"""
        circuit = QuantumCircuit(7)
        circuit.h(0)
        circuit.append(HGate(label="H2"), [1])
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
        circuit.append(SGate(label="S1"), [4])
        circuit.sdg(4)
        circuit.t(4)
        circuit.tdg(4)
        circuit.p(pi / 2, 4)
        circuit.cz(5, 6)
        circuit.cp(pi / 2, 5, 6)
        circuit.y(5)
        circuit.rx(pi / 3, 5)
        circuit.rzx(pi / 2, 5, 6)
        circuit.u(pi / 2, pi / 2, pi / 2, 5)
        circuit.barrier(5, 6)
        circuit.reset(5)

        style = {
            "name": "user_style",
            "displaytext": {"H2": "H_2"},
            "displaycolor": {"H2": ("#EEDD00", "#FF0000")},
        }
        fname = "user_style.png"
        self.circuit_drawer(
            circuit,
            output="mpl",
            style=style,
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )

        with self.subTest(msg="check image"):
            self.assertGreaterEqual(ratio, self.threshold)

        with self.subTest(msg="check style dict unchanged"):
            self.assertEqual(
                style,
                {
                    "name": "user_style",
                    "displaytext": {"H2": "H_2"},
                    "displaycolor": {"H2": ("#EEDD00", "#FF0000")},
                },
            )

    def test_subfont_change(self):
        """Tests changing the subfont size"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.x(0)
        circuit.u(pi / 2, pi / 2, pi / 2, 1)
        circuit.p(pi / 2, 2)
        style = {"name": "iqp", "subfontsize": 11}

        fname = "subfont.png"
        self.circuit_drawer(circuit, output="mpl", style=style, filename=fname)
        self.assertEqual(style, {"name": "iqp", "subfontsize": 11})  # check does not change style

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_style_custom_gates(self):
        """Tests style for custom gates"""

        def cnotnot(gate_label):
            gate_circuit = QuantumCircuit(3, name=gate_label)
            gate_circuit.cx(0, 1)
            gate_circuit.cx(0, 2)
            gate = gate_circuit.to_gate()
            return gate

        q = QuantumRegister(3, name="q")

        circuit = QuantumCircuit(q)

        circuit.append(cnotnot("CNOTNOT"), [q[0], q[1], q[2]])
        circuit.append(cnotnot("CNOTNOT_PRIME"), [q[0], q[1], q[2]])
        circuit.h(q[0])

        fname = "style_custom_gates.png"
        self.circuit_drawer(
            circuit,
            output="mpl",
            style={
                "displaycolor": {"CNOTNOT": ("#000000", "#FFFFFF"), "h": ("#A1A1A1", "#043812")},
                "displaytext": {"CNOTNOT_PRIME": "$\\mathrm{CNOTNOT}'$"},
            },
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_6095(self):
        """Tests controlled-phase gate style
        See https://github.com/Qiskit/qiskit-terra/issues/6095"""
        circuit = QuantumCircuit(2)
        circuit.cp(1.0, 0, 1)
        circuit.h(1)

        fname = "6095.png"
        self.circuit_drawer(
            circuit,
            output="mpl",
            style={"displaycolor": {"cp": ("#A27486", "#000000"), "h": ("#A27486", "#000000")}},
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_instruction_1q_1c(self):
        """Tests q0-cr0 instruction on a circuit"""
        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)
        inst = QuantumCircuit(1, 1, name="Inst").to_instruction()
        circuit.append(inst, [qr[0]], [cr[0]])

        fname = "instruction_1q_1c.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_instruction_3q_3c_circ1(self):
        """Tests q0-q1-q2-cr_20-cr0-cr1 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr, cr, cr2)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[0], qr[1], qr[2]], [cr2[0], cr[0], cr[1]])

        fname = "instruction_3q_3c_circ1.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_instruction_3q_3c_circ2(self):
        """Tests q3-q0-q2-cr0-cr1-cr_20 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr, cr, cr2)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[3], qr[0], qr[2]], [cr[0], cr[1], cr2[0]])

        fname = "instruction_3q_3c_circ2.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_instruction_3q_3c_circ3(self):
        """Tests q3-q1-q2-cr_31-cr1-cr_30 instruction on a circuit"""
        qr = QuantumRegister(4, "qr")
        cr = ClassicalRegister(2, "cr")
        cr2 = ClassicalRegister(1, "cr2")
        cr3 = ClassicalRegister(2, "cr3")
        circuit = QuantumCircuit(qr, cr, cr2, cr3)
        inst = QuantumCircuit(3, 3, name="Inst").to_instruction()
        circuit.append(inst, [qr[3], qr[1], qr[2]], [cr3[1], cr[1], cr3[0]])

        fname = "instruction_3q_3c_circ3.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_overwide_gates(self):
        """Test gates don't exceed width of default fold"""
        circuit = QuantumCircuit(5)
        initial_state = np.zeros(2**5)
        initial_state[5] = 1
        circuit.initialize(initial_state)

        fname = "wide_params.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_one_bit_regs(self):
        """Test registers with only one bit display without number"""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr1 = ClassicalRegister(1, "cr1")
        cr2 = ClassicalRegister(2, "cr2")
        circuit = QuantumCircuit(qr1, qr2, cr1, cr2)
        circuit.h(0)
        circuit.measure(0, 0)

        fname = "one_bit_regs.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_measure_arrows_false(self):
        """Test measure_arrows set to False"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.x(0)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.x(1)
        circuit.h(1)
        circuit.measure(1, 1)
        circuit.x(2)
        circuit.h(2)
        circuit.measure(2, 2)

        fname = "measure_arrows_false.png"
        self.circuit_drawer(circuit, output="mpl", measure_arrows=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_user_ax_subplot(self):
        """Test for when user supplies ax for a subplot"""
        import matplotlib.pyplot as plt

        fig = plt.figure(1, figsize=(6, 4))
        fig.patch.set_facecolor("white")
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot([1, 2, 3])

        circuit = QuantumCircuit(4)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.h(1)
        circuit.cx(1, 2)
        plt.close(fig)

        fname = "user_ax.png"
        self.circuit_drawer(circuit, output="mpl", ax=ax2, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_figwidth(self):
        """Test style dict 'figwidth'"""
        circuit = QuantumCircuit(3)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.x(1)
        circuit.cx(1, 2)
        circuit.x(2)

        fname = "figwidth.png"
        self.circuit_drawer(circuit, output="mpl", style={"figwidth": 5}, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_registerless_one_bit(self):
        """Test circuit with one-bit registers and registerless bits."""
        qrx = QuantumRegister(2, "qrx")
        qry = QuantumRegister(1, "qry")
        crx = ClassicalRegister(2, "crx")
        circuit = QuantumCircuit(qrx, [Qubit(), Qubit()], qry, [Clbit(), Clbit()], crx)

        fname = "registerless_one_bit.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_idle_wires_barrier(self):
        """Test that idle_wires False works with barrier"""
        circuit = QuantumCircuit(4, 4)
        circuit.x(2)
        circuit.barrier()

        fname = "idle_wires_barrier.png"
        self.circuit_drawer(
            circuit, output="mpl", cregbundle=False, filename=fname, idle_wires=False
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_wire_order(self):
        """Test the wire_order option"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(4, "c")
        cr2 = ClassicalRegister(2, "cx")
        circuit = QuantumCircuit(qr, cr, cr2)
        circuit.h(0)
        circuit.h(3)
        circuit.x(1)
        with circuit.if_test((cr, 10)):
            circuit.x(3)

        fname = "wire_order.png"
        self.circuit_drawer(
            circuit,
            output="mpl",
            cregbundle=False,
            wire_order=[2, 1, 3, 0, 6, 8, 9, 5, 4, 7],
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_basic_box(self):
        """Test that drawing a `box` doesn't explode."""
        # The exact output is not important - feel free to change it.  We only care that it doesn't
        # explode when drawing.
        qc = QuantumCircuit(5)
        with qc.box():
            qc.x(0)
        qc.x(1)
        with qc.box():
            qc.cx(2, 3)
            with qc.box():
                qc.noop(4)
        fname = "basic_box.png"
        self.circuit_drawer(qc, output="mpl", filename=fname)
        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_barrier_label(self):
        """Test the barrier label"""
        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.y(1)
        circuit.barrier()
        circuit.y(0)
        circuit.x(1)
        circuit.barrier(label="End Y/X")

        fname = "barrier_label.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_barrier_label_truncation(self):
        """Test that long barrier labels are truncated"""
        circuit = QuantumCircuit(2)
        circuit.barrier()
        circuit.barrier(label="a" * 10)
        circuit.barrier(label="b" * 1000)

        fname = "barrier_label_truncation.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname, barrier_label_len=9)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_op(self):
        """Test the IfElseOp with if only"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        with circuit.if_test((cr[1], 1)):
            circuit.h(0)
            circuit.cx(0, 1)

        fname = "if_op.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_bundle_false(self):
        """Test the IfElseOp with else with cregbundle False"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        with circuit.if_test((cr[1], 1)) as _else:
            circuit.h(0)
            circuit.cx(0, 1)
        with _else:
            circuit.cx(0, 1)

        fname = "if_else_op_false.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_bundle_true(self):
        """Test the IfElseOp with else with cregbundle True"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        with circuit.if_test((cr[1], 1)) as _else:
            circuit.h(0)
            circuit.cx(0, 1)
        with _else:
            circuit.cx(0, 1)

        fname = "if_else_op_true.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=True, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_textbook_style(self):
        """Test the IfElseOp with else in textbook style"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qr, cr)

        with circuit.if_test((cr[1], 1)) as _else:
            circuit.h(0)
            circuit.cx(0, 1)
        with _else:
            circuit.cx(0, 1)

        fname = "if_else_op_textbook.png"
        self.circuit_drawer(
            circuit, output="mpl", style="textbook", cregbundle=False, filename=fname
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_with_body(self):
        """Test the IfElseOp with adding a body manually"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)
        circuit.h(0)
        circuit.h(1)
        circuit.measure(0, 1)
        circuit.measure(1, 2)
        circuit.x(2)
        circuit.x(2, label="XLabel")

        qr2 = QuantumRegister(3, "qr2")
        qc2 = QuantumCircuit(qr2, cr)
        qc2.x(1)
        qc2.y(1)
        qc2.z(0)
        qc2.x(0, label="X1i")

        circuit.if_else((cr[1], 1), qc2, None, [0, 1, 2], [0, 1, 2])
        circuit.x(0, label="X1i")

        fname = "if_else_body.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_nested(self):
        """Test the IfElseOp with complex nested if/else"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        with circuit.if_test((cr[1], 1)) as _else:
            circuit.x(0, label="X")
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)
                circuit.y(1)
                with circuit.if_test((cr[1], 1)):
                    circuit.y(1)
                    circuit.z(2)
                    with circuit.if_test((cr[2], 1)):
                        circuit.cx(0, 1)
                        with circuit.if_test((cr[1], 1)):
                            circuit.h(0)
                            circuit.x(1)
        with _else:
            circuit.y(1)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)
                circuit.x(1)
            inst = QuantumCircuit(2, 2, name="Inst").to_instruction()
            circuit.append(inst, [qr[0], qr[1]], [cr[0], cr[1]])
        circuit.x(0)

        fname = "if_else_op_nested.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=True, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_wire_order(self):
        """Test the IfElseOp with complex nested if/else and wire_order"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        with circuit.if_test((cr[1], 1)) as _else:
            circuit.x(0, label="X")
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)
                circuit.y(1)
                with circuit.if_test((cr[1], 1)):
                    circuit.y(1)
                    circuit.z(2)
                    with circuit.if_test((cr[2], 1)):
                        circuit.cx(0, 1)
                        with circuit.if_test((cr[1], 1)):
                            circuit.h(0)
                            circuit.x(1)
        with _else:
            circuit.y(1)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)
                circuit.x(1)
            inst = QuantumCircuit(2, 2, name="Inst").to_instruction()
            circuit.append(inst, [qr[0], qr[1]], [cr[0], cr[1]])
        circuit.x(0)

        fname = "if_else_op_wire_order.png"
        self.circuit_drawer(
            circuit,
            output="mpl",
            cregbundle=False,
            wire_order=[2, 0, 3, 1, 4, 5, 6],
            filename=fname,
        )

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_op_fold(self):
        """Test the IfElseOp with complex nested if/else and fold"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        with circuit.if_test((cr[1], 1)) as _else:
            circuit.x(0, label="X")
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)
                circuit.y(1)
                with circuit.if_test((cr[1], 1)):
                    circuit.y(1)
                    circuit.z(2)
                    with circuit.if_test((cr[2], 1)):
                        circuit.cx(0, 1)
                        with circuit.if_test((cr[1], 1)):
                            circuit.h(0)
                            circuit.x(1)
        with _else:
            circuit.y(1)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)
                circuit.x(1)
            inst = QuantumCircuit(2, 2, name="Inst").to_instruction()
            circuit.append(inst, [qr[0], qr[1]], [cr[0], cr[1]])
        circuit.x(0)

        fname = "if_else_op_fold.png"
        self.circuit_drawer(circuit, output="mpl", fold=7, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_while_loop_op(self):
        """Test the WhileLoopOp"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.while_loop((cr[0], 0)):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.x(0)

        fname = "while_loop.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_for_loop_op(self):
        """Test the ForLoopOp"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        a = Parameter("a")
        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.for_loop((2, 4, 8, 16), loop_parameter=a):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.rx(pi / a, 1)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)

        fname = "for_loop.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_for_loop_op_range(self):
        """Test the ForLoopOp with a range"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        a = Parameter("a")
        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.for_loop(range(10, 20), loop_parameter=a):
            circuit.h(0)
            circuit.cx(0, 1)
            circuit.rx(pi / a, 1)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)

        fname = "for_loop_range.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_for_loop_op_1_qarg(self):
        """Test the ForLoopOp with 1 qarg"""
        qr = QuantumRegister(4, "q")
        cr = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qr, cr)

        a = Parameter("a")
        circuit.h(0)
        circuit.measure(0, 2)
        with circuit.for_loop((2, 4, 8, 16), loop_parameter=a):
            circuit.h(0)
            circuit.rx(pi / a, 0)
            circuit.measure(0, 0)
            with circuit.if_test((cr[2], 1)):
                circuit.z(0)

        fname = "for_loop_1_qarg.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_switch_case_op(self):
        """Test the SwitchCaseOp"""
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qreg, creg)

        circuit.h([0, 1, 2])
        circuit.measure([0, 1, 2], [0, 1, 2])

        with circuit.switch(creg) as case:
            with case(0, 1, 2):
                circuit.x(0)
            with case(3, 4, 5):
                circuit.y(1)
                circuit.y(0)
                circuit.y(0)
            with case(case.DEFAULT):
                circuit.cx(0, 1)
        circuit.h(0)

        fname = "switch_case.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_switch_case_op_1_qarg(self):
        """Test the SwitchCaseOp with 1 qarg"""
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qreg, creg)

        circuit.h([0, 1, 2])
        circuit.measure([0, 1, 2], [0, 1, 2])

        with circuit.switch(creg) as case:
            with case(0, 1, 2):
                circuit.x(0)
            with case(case.DEFAULT):
                circuit.y(0)
        circuit.h(0)

        fname = "switch_case_1_qarg.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_switch_case_op_empty_default(self):
        """Test the SwitchCaseOp with empty default case"""
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(3, "cr")
        circuit = QuantumCircuit(qreg, creg)

        circuit.h([0, 1, 2])
        circuit.measure([0, 1, 2], [0, 1, 2])

        with circuit.switch(creg) as case:
            with case(0, 1, 2):
                circuit.x(0)
            with case(case.DEFAULT):
                pass
        circuit.h(0)

        fname = "switch_case_empty_default.png"
        self.circuit_drawer(circuit, output="mpl", cregbundle=False, filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_with_expression(self):
        """Test the IfElseOp with an expression"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.if_test(expr.equal(expr.bit_and(cr1, expr.bit_and(cr2, cr3)), 3)):
            circuit.z(0)

        fname = "if_op_expr.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_with_expression_nested(self):
        """Test the IfElseOp with an expression for nested"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.if_test(expr.equal(expr.bit_and(cr1, expr.bit_and(cr2, cr3)), 3)):
            circuit.x(0)
            with circuit.if_test(expr.equal(expr.bit_and(cr3, expr.bit_and(cr1, cr2)), 5)):
                circuit.z(1)

        fname = "if_op_expr_nested.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_switch_with_expression(self):
        """Test the SwitchCaseOp with an expression"""
        qr = QuantumRegister(3, "qr")
        cr = ClassicalRegister(3, "cr")
        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        circuit = QuantumCircuit(qr, cr, cr1, cr2, cr3)

        circuit.h(0)
        with circuit.switch(expr.bit_and(cr1, expr.bit_and(cr2, cr3))) as case:
            with case(0, 1, 2, 3):
                circuit.x(0)
            with case(case.DEFAULT):
                circuit.cx(0, 1)

        fname = "switch_expr.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_control_flow_layout(self):
        """Test control flow with a layout set."""
        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qreg, creg)
        qc.h([0, 1])
        qc.h([0, 1])
        qc.h([0, 1])
        qc.measure([0, 1], [0, 1])
        with qc.switch(creg) as case:
            with case(0):
                qc.z(0)
            with case(1, 2):
                qc.cx(0, 1)
            with case(case.DEFAULT):
                qc.h(0)
        backend = GenericBackendV2(5, coupling_map=YORKTOWN_CMAP, seed=16)
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        tqc = transpile(qc, backend, optimization_level=2, seed_transpiler=671_42)
        fname = "layout_control_flow.png"
        self.circuit_drawer(tqc, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_control_flow_nested_layout(self):
        """Test nested control flow with a layout set."""
        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qreg, creg)
        qc.h([0, 1])
        qc.h([0, 1])
        qc.h([0, 1])
        qc.measure([0, 1], [0, 1])
        with qc.switch(creg) as case:
            with case(0):
                qc.z(0)
            with case(1, 2):
                with qc.if_test((creg[0], 0)):
                    qc.cx(0, 1)
            with case(case.DEFAULT):
                with qc.if_test((creg[1], 0)):
                    qc.h(0)
        backend = GenericBackendV2(5, coupling_map=YORKTOWN_CMAP, seed=0)
        backend.target.add_instruction(SwitchCaseOp, name="switch_case")
        backend.target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, backend, optimization_level=2, seed_transpiler=671_42)

        fname = "nested_layout_control_flow.png"
        self.circuit_drawer(tqc, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_control_flow_with_fold_minus_one(self):
        """Test control flow works with fold=-1. Qiskit issue #12012"""
        qreg = QuantumRegister(2, "qr")
        creg = ClassicalRegister(2, "cr")
        circuit = QuantumCircuit(qreg, creg)
        with circuit.if_test((creg[1], 1)):
            circuit.h(0)
            circuit.cx(0, 1)

        fname = "control_flow_fold_minus_one.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname, fold=-1)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_annotated_operation(self):
        """Test AnnotatedOperations and other non-Instructions."""
        circuit = QuantumCircuit(3)
        cliff = random_clifford(2)
        circuit.append(cliff, [0, 1])
        circuit.x(0)
        circuit.h(1)
        circuit.append(SGate().control(2, ctrl_state=1, annotated=False), [0, 2, 1])
        circuit.ccx(0, 1, 2)
        op1 = AnnotatedOperation(
            SGate(), [InverseModifier(), ControlModifier(2, 1), PowerModifier(3.29)]
        )
        circuit.append(op1, [0, 1, 2])
        circuit.append(SXGate(), [1])
        fname = "annotated.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_no_qreg_names_after_layout(self):
        """Test that full register names are not shown after transpilation.
        See https://github.com/Qiskit/qiskit-terra/issues/11038"""
        backend = GenericBackendV2(5, coupling_map=YORKTOWN_CMAP, seed=42)

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        circuit = transpile(
            qc, backend, basis_gates=["rz", "sx", "cx"], layout_method="sabre", seed_transpiler=15
        )

        fname = "qreg_names_after_layout.png"
        self.circuit_drawer(circuit, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_if_else_standalone_var(self):
        """Test if/else with standalone Var."""
        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit(2, 2, inputs=[a])
        b = qc.add_var("b", False)
        qc.store(a, 128)
        with qc.if_test(expr.logic_not(b)):
            # Mix old-style and new-style.
            with qc.if_test(expr.equal(b, qc.clbits[0])):
                qc.cx(0, 1)
            c = qc.add_var("c", b)
            with qc.if_test(expr.logic_and(c, expr.equal(a, 128))):
                qc.h(0)
        fname = "if_else_standalone_var.png"
        self.circuit_drawer(qc, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)

    def test_switch_standalone_var(self):
        """Test switch with standalone Var."""
        a = expr.Var.new("a", types.Uint(8))
        qc = QuantumCircuit(2, 2, inputs=[a])
        b = qc.add_var("b", expr.lift(5, a.type))
        with qc.switch(expr.bit_not(a)) as case:
            with case(0):
                with qc.switch(b) as case2:
                    with case2(2):
                        qc.cx(0, 1)
                    with case2(case2.DEFAULT):
                        qc.cx(1, 0)
            with case(case.DEFAULT):
                c = qc.add_var("c", expr.equal(a, b))
                with qc.if_test(c):
                    qc.h(0)
        fname = "switch_standalone_var.png"
        self.circuit_drawer(qc, output="mpl", filename=fname)

        ratio = VisualTestUtilities._save_diff(
            self._image_path(fname),
            self._reference_path(fname),
            fname,
            FAILURE_DIFF_DIR,
            FAILURE_PREFIX,
        )
        self.assertGreaterEqual(ratio, self.threshold)


if __name__ == "__main__":
    unittest.main(verbosity=1)
