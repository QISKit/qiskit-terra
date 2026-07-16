# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for CNOTDihedral functions.
"""

import unittest
from test import combine
from ddt import ddt

import numpy as np
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library import (
    IGate,
    XGate,
    YGate,
    ZGate,
    TGate,
    TdgGate,
    SGate,
    SdgGate,
    CXGate,
    CZGate,
    CSGate,
    CSdgGate,
    SwapGate,
    CCZGate,
)
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.dihedral import CNOTDihedral
from qiskit.quantum_info import random_cnotdihedral, random_pauli
from qiskit.synthesis.cnotdihedral import (
    synth_cnotdihedral_general,
    synth_cnotdihedral_full,
    synth_cnotdihedral_two_qubits,
)


def random_cnotdihedral_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo random CNOTDihedral circuit."""

    if gates == "all":
        gates = ["i", "x", "y", "z", "t", "tdg", "s", "sdg"]
        if num_qubits >= 2:
            gates += ["cx", "cz", "cs", "csdg", "swap"]
        if num_qubits >= 3:
            gates += ["ccz"]

    instructions = {
        "i": (IGate(), 1),
        "x": (XGate(), 1),
        "y": (YGate(), 1),
        "z": (ZGate(), 1),
        "s": (SGate(), 1),
        "sdg": (SdgGate(), 1),
        "t": (TGate(), 1),
        "tdg": (TdgGate(), 1),
        "cx": (CXGate(), 2),
        "cz": (CZGate(), 2),
        "cs": (CSGate(), 2),
        "csdg": (CSdgGate(), 2),
        "swap": (SwapGate(), 2),
        "ccz": (CCZGate(), 3),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    samples = rng.choice(gates, num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs)

    return circ


@ddt
class TestCNOTDihedral(unittest.TestCase):
    """
    Test CNOT-dihedral functions
    """

    def test_1_qubit_identities(self):
        """Tests identities for 1-qubit gates"""
        # T*X*T = X
        circ1 = QuantumCircuit(1)
        circ1.t(0)
        circ1.x(0)
        circ1.t(0)
        elem1 = CNOTDihedral(circ1)
        elem = CNOTDihedral(XGate())
        self.assertEqual(elem1, elem, "Error: 1-qubit identity does not hold")

        # X*T*X = Tdg
        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.t(0)
        circ1.x(0)
        elem1 = CNOTDihedral(circ1)
        elem = CNOTDihedral(TdgGate())
        self.assertEqual(elem1, elem, "Error: 1-qubit identity does not hold")

        # X*Tdg*X = T
        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.tdg(0)
        circ1.x(0)
        elem1 = CNOTDihedral(circ1)
        elem = CNOTDihedral(TGate())
        self.assertEqual(elem1, elem, "Error: 1-qubit identity does not hold")

        # X*S*X = Sdg
        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.s(0)
        circ1.x(0)
        elem1 = CNOTDihedral(circ1)
        elem = CNOTDihedral(SdgGate())
        self.assertEqual(elem1, elem, "Error: 1-qubit identity does not hold")

        # X*Sdg*X = S
        circ1 = QuantumCircuit(1)
        circ1.x(0)
        circ1.sdg(0)
        circ1.x(0)
        elem1 = CNOTDihedral(circ1)
        elem = CNOTDihedral(SGate())
        self.assertEqual(elem1, elem, "Error: 1-qubit identity does not hold")

        # T*X*Tdg = S*X
        circ1 = QuantumCircuit(1)
        circ1.t(0)
        circ1.x(0)
        circ1.tdg(0)
        circ2 = QuantumCircuit(1)
        circ2.s(0)
        circ2.x(0)
        elem1 = CNOTDihedral(circ1)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 1-qubit identity does not hold")

    def test_2_qubit_identities(self):
        """Tests identities for 2-qubit gates"""
        # SI * CX * SdgI = CX
        elem = CNOTDihedral(CXGate())
        circ1 = QuantumCircuit(2)
        circ1.s(0)
        circ1.cx(0, 1)
        circ1.sdg(0)
        elem1 = CNOTDihedral(circ1)
        self.assertEqual(elem, elem1, "Error: 2-qubit CX identity does not hold")

        # SI * CZ * SdgI = CZ
        elem = CNOTDihedral(CZGate())
        circ1 = QuantumCircuit(2)
        circ1.s(0)
        circ1.cz(1, 0)
        circ1.sdg(0)
        elem1 = CNOTDihedral(circ1)
        self.assertEqual(elem, elem1, "Error: 2-qubit CZ identity does not hold")

        # SWAP = CX01 * CX10 * CX01 = CX10 * CX01 * CX10
        elem = CNOTDihedral(SwapGate())
        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)
        circ1.cx(0, 1)
        elem1 = CNOTDihedral(circ1)
        self.assertEqual(elem, elem1, "Error: 2-qubit SWAP identity does not hold")

        circ2 = QuantumCircuit(2)
        circ2.cx(1, 0)
        circ2.cx(0, 1)
        circ2.cx(1, 0)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit SWAP identity does not hold")

        # CS01 = CS10 (symmetric)
        #
        #      ┌───┐
        # q_0: ┤ T ├──■───────────■──
        #      ├───┤┌─┴─┐┌─────┐┌─┴─┐
        # q_1: ┤ T ├┤ X ├┤ Tdg ├┤ X ├
        #      └───┘└───┘└─────┘└───┘
        circ1 = QuantumCircuit(2)
        circ1.t(0)
        circ1.t(1)
        circ1.cx(0, 1)
        circ1.tdg(1)
        circ1.cx(0, 1)

        #      ┌───┐┌───┐┌─────┐┌───┐
        # q_0: ┤ T ├┤ X ├┤ Tdg ├┤ X ├
        #      ├───┤└─┬─┘└─────┘└─┬─┘
        # q_1: ┤ T ├──■───────────■──
        #      └───┘
        circ2 = QuantumCircuit(2)
        circ2.t(1)
        circ2.t(0)
        circ2.cx(1, 0)
        circ2.tdg(0)
        circ2.cx(1, 0)
        elem1 = CNOTDihedral(circ1)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit CS identity does not hold")

        circcs01 = QuantumCircuit(2)
        circcs01.cs(0, 1)
        elemcs01 = CNOTDihedral(circcs01)
        self.assertEqual(elem1, elemcs01, "Error: 2-qubit CS identity does not hold")

        circcs10 = QuantumCircuit(2)
        circcs10.cs(1, 0)
        elemcs10 = CNOTDihedral(circcs10)
        self.assertEqual(elem1, elemcs10, "Error: 2-qubit CS identity does not hold")

        circ_cs = CSGate()
        elem_cs = CNOTDihedral(circ_cs)
        self.assertEqual(elem1, elem_cs, "Error: 2-qubit CS identity does not hold")

        # TI*CS*TdgI = CS
        #
        #      ┌───┐┌───┐                 ┌─────┐
        # q_0: ┤ T ├┤ T ├──■───────────■──┤ Tdg ├
        #      ├───┤└───┘┌─┴─┐┌─────┐┌─┴─┐└─────┘
        # q_1: ┤ T ├─────┤ X ├┤ Tdg ├┤ X ├───────
        #      └───┘     └───┘└─────┘└───┘
        circ3 = QuantumCircuit(2)
        circ3.t(0)
        circ3.t(0)
        circ3.t(1)
        circ3.cx(0, 1)
        circ3.tdg(1)
        circ3.cx(0, 1)
        circ3.tdg(0)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CS identity does not hold")

        circ3 = QuantumCircuit(2)
        circ3.t(0)
        circ3.cs(0, 1)
        circ3.tdg(0)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CS identity does not hold")

        # IT*CS*ITdg = CS
        #
        #      ┌───┐
        # q_0: ┤ T ├───────■───────────■─────────
        #      ├───┤┌───┐┌─┴─┐┌─────┐┌─┴─┐┌─────┐
        # q_1: ┤ T ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ Tdg ├
        #      └───┘└───┘└───┘└─────┘└───┘└─────┘
        circ4 = QuantumCircuit(2)
        circ4.t(1)
        circ4.t(0)
        circ4.t(1)
        circ4.cx(0, 1)
        circ4.tdg(1)
        circ4.cx(0, 1)
        circ4.tdg(1)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CS identity does not hold")

        circ4 = QuantumCircuit(2)
        circ4.t(1)
        circ4.cs(0, 1)
        circ4.tdg(1)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CS identity does not hold")

        # XX*CS*XX*SS = CS
        #
        #      ┌───┐┌───┐                 ┌───┐┌───┐
        # q_0: ┤ X ├┤ T ├──■───────────■──┤ X ├┤ S ├
        #      ├───┤├───┤┌─┴─┐┌─────┐┌─┴─┐├───┤├───┤
        # q_1: ┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ X ├┤ S ├
        #      └───┘└───┘└───┘└─────┘└───┘└───┘└───┘
        circ5 = QuantumCircuit(2)
        circ5.x(0)
        circ5.x(1)
        circ5.t(0)
        circ5.t(1)
        circ5.cx(0, 1)
        circ5.tdg(1)
        circ5.cx(0, 1)
        circ5.x(0)
        circ5.x(1)
        circ5.s(0)
        circ5.s(1)
        elem5 = CNOTDihedral(circ5)
        self.assertEqual(elem1, elem5, "Error: 2-qubit CS identity does not hold")

        circ5 = QuantumCircuit(2)
        circ5.x(0)
        circ5.x(1)
        circ5.cs(0, 1)
        circ5.x(0)
        circ5.x(1)
        circ5.s(0)
        circ5.s(1)
        elem5 = CNOTDihedral(circ5)
        self.assertEqual(elem1, elem5, "Error: 2-qubit CS identity does not hold")

        # CSdg01 = CSdg10 (symmetric)
        #
        #      ┌─────┐
        # q_0: ┤ Tdg ├──■─────────■──
        #      ├─────┤┌─┴─┐┌───┐┌─┴─┐
        # q_1: ┤ Tdg ├┤ X ├┤ T ├┤ X ├
        #      └─────┘└───┘└───┘└───┘
        circ1 = QuantumCircuit(2)
        circ1.tdg(0)
        circ1.tdg(1)
        circ1.cx(0, 1)
        circ1.t(1)
        circ1.cx(0, 1)

        #      ┌─────┐┌───┐┌───┐┌───┐
        # q_0: ┤ Tdg ├┤ X ├┤ T ├┤ X ├
        #      ├─────┤└─┬─┘└───┘└─┬─┘
        # q_1: ┤ Tdg ├──■─────────■──
        #      └─────┘
        circ2 = QuantumCircuit(2)
        circ2.tdg(1)
        circ2.tdg(0)
        circ2.cx(1, 0)
        circ2.t(0)
        circ2.cx(1, 0)
        elem1 = CNOTDihedral(circ1)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit CSdg identity does not hold")

        circsdg01 = QuantumCircuit(2)
        circsdg01.csdg(0, 1)
        elemcsdg01 = CNOTDihedral(circsdg01)
        self.assertEqual(elem1, elemcsdg01, "Error: 2-qubit CSdg identity does not hold")

        circsdg10 = QuantumCircuit(2)
        circsdg10.csdg(1, 0)
        elemcsdg10 = CNOTDihedral(circsdg10)
        self.assertEqual(elem1, elemcsdg10, "Error: 2-qubit CSdg identity does not hold")

        circ_csdg = CSdgGate()
        elem_csdg = CNOTDihedral(circ_csdg)
        self.assertEqual(elem1, elem_csdg, "Error: 2-qubit CS identity does not hold")

        # XI*CS*XI*ISdg = CSdg
        #
        #      ┌───┐┌───┐                  ┌───┐
        # q_0: ┤ X ├┤ T ├──■───────────■───┤ X ├─
        #      ├───┤└───┘┌─┴─┐┌─────┐┌─┴─┐┌┴───┴┐
        # q_1: ┤ T ├─────┤ X ├┤ Tdg ├┤ X ├┤ Sdg ├
        #      └───┘     └───┘└─────┘└───┘└─────┘
        circ3 = QuantumCircuit(2)
        circ3.x(0)
        circ3.t(0)
        circ3.t(1)
        circ3.cx(0, 1)
        circ3.tdg(1)
        circ3.cx(0, 1)
        circ3.x(0)
        circ3.sdg(1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CSdg identity does not hold")

        circ3 = QuantumCircuit(2)
        circ3.x(0)
        circ3.cs(0, 1)
        circ3.x(0)
        circ3.sdg(1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem3, elemcsdg01, "Error: 2-qubit CSdg identity does not hold")

        # IX*CS*IX*SdgI = CSdg
        #
        #      ┌───┐                      ┌─────┐
        # q_0: ┤ T ├───────■───────────■──┤ Sdg ├
        #      ├───┤┌───┐┌─┴─┐┌─────┐┌─┴─┐└┬───┬┘
        # q_1: ┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├─┤ X ├─
        #      └───┘└───┘└───┘└─────┘└───┘ └───┘
        circ4 = QuantumCircuit(2)
        circ4.x(1)
        circ4.t(0)
        circ4.t(1)
        circ4.cx(0, 1)
        circ4.tdg(1)
        circ4.cx(0, 1)
        circ4.x(1)
        circ4.sdg(0)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CSdg identity does not hold")

        circ4 = QuantumCircuit(2)
        circ4.x(1)
        circ4.cs(0, 1)
        circ4.x(1)
        circ4.sdg(0)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elemcsdg01, elem4, "Error: 2-qubit CSdg identity does not hold")

        # relations for CZ
        # CZ(0,1) = CZ(1,0)
        elem = CNOTDihedral(CZGate())
        circ1 = QuantumCircuit(2)
        circ1.cz(0, 1)
        elem1 = CNOTDihedral(circ1)
        self.assertEqual(elem, elem1, "Error: 2-qubit CZ identity does not hold")

        circ2 = QuantumCircuit(2)
        circ2.cz(1, 0)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit CZ identity does not hold")

        # CZ = CS * CS
        #
        #      ┌───┐                 ┌───┐
        # q_0: ┤ T ├──■───────────■──┤ T ├──■───────────■──
        #      ├───┤┌─┴─┐┌─────┐┌─┴─┐├───┤┌─┴─┐┌─────┐┌─┴─┐
        # q_1: ┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├
        #      └───┘└───┘└─────┘└───┘└───┘└───┘└─────┘└───┘
        circ3 = QuantumCircuit(2)
        circ3.t(0)
        circ3.t(1)
        circ3.cx(0, 1)
        circ3.tdg(1)
        circ3.cx(0, 1)
        circ3.t(0)
        circ3.t(1)
        circ3.cx(0, 1)
        circ3.tdg(1)
        circ3.cx(0, 1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CZ identity does not hold")

        circ3 = QuantumCircuit(2)
        circ3.cs(0, 1)
        circ3.cs(0, 1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CZ identity does not hold")

        # CZ = CSdg * CSdg
        #
        #      ┌─────┐               ┌─────┐
        # q_0: ┤ Tdg ├──■─────────■──┤ Tdg ├──■─────────■──
        #      ├─────┤┌─┴─┐┌───┐┌─┴─┐├─────┤┌─┴─┐┌───┐┌─┴─┐
        # q_1: ┤ Tdg ├┤ X ├┤ T ├┤ X ├┤ Tdg ├┤ X ├┤ T ├┤ X ├
        #      └─────┘└───┘└───┘└───┘└─────┘└───┘└───┘└───┘
        circ4 = QuantumCircuit(2)
        circ4.tdg(0)
        circ4.tdg(1)
        circ4.cx(0, 1)
        circ4.t(1)
        circ4.cx(0, 1)
        circ4.tdg(0)
        circ4.tdg(1)
        circ4.cx(0, 1)
        circ4.t(1)
        circ4.cx(0, 1)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CZ identity does not hold")

        circ4 = QuantumCircuit(2)
        circ4.csdg(0, 1)
        circ4.csdg(0, 1)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CZ identity does not hold")

        # CZ = TdgTdg * CX * T^2I * CX * TdgTdg
        #
        #      ┌─────┐┌───┐┌───┐┌───┐┌───┐┌─────┐
        # q_0: ┤ Tdg ├┤ X ├┤ T ├┤ T ├┤ X ├┤ Tdg ├
        #      ├─────┤└─┬─┘└───┘└───┘└─┬─┘├─────┤
        # q_1: ┤ Tdg ├──■──────────────■──┤ Tdg ├
        #      └─────┘                    └─────┘
        circ5 = QuantumCircuit(2)
        circ5.tdg(0)
        circ5.tdg(1)
        circ5.cx(1, 0)
        circ5.t(0)
        circ5.t(0)
        circ5.cx(1, 0)
        circ5.tdg(0)
        circ5.tdg(1)
        elem5 = CNOTDihedral(circ5)
        self.assertEqual(elem1, elem5, "Error: 2-qubit CZ identity does not hold")

        # relations for CX
        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        elem1 = CNOTDihedral(circ1)

        # TI*CX*TdgI = CX
        #
        #      ┌───┐     ┌─────┐
        # q_0: ┤ T ├──■──┤ Tdg ├
        #      └───┘┌─┴─┐└─────┘
        # q_1: ─────┤ X ├───────
        #           └───┘
        circ2 = QuantumCircuit(2)
        circ2.t(0)
        circ2.cx(0, 1)
        circ2.tdg(0)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit CX identity does not hold")

        # IZ*CX*ZZ = CX
        #
        #                ┌───┐
        # q_0: ───────■──┤ Z ├
        #      ┌───┐┌─┴─┐├───┤
        # q_1: ┤ Z ├┤ X ├┤ Z ├
        #      └───┘└───┘└───┘
        circ3 = QuantumCircuit(2)
        circ3.z(1)
        circ3.cx(0, 1)
        circ3.z(0)
        circ3.z(1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 2-qubit CX identity does not hold")

        # IX*CX*IX = CX
        #
        # q_0: ───────■───────
        #      ┌───┐┌─┴─┐┌───┐
        # q_1: ┤ X ├┤ X ├┤ X ├
        #      └───┘└───┘└───┘
        circ4 = QuantumCircuit(2)
        circ4.x(1)
        circ4.cx(0, 1)
        circ4.x(1)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 2-qubit CX identity does not hold")

        # XI*CX*XX = CX
        #
        #      ┌───┐     ┌───┐
        # q_0: ┤ X ├──■──┤ X ├
        #      └───┘┌─┴─┐├───┤
        # q_1: ─────┤ X ├┤ X ├
        #           └───┘└───┘
        circ5 = QuantumCircuit(2)
        circ5.x(0)
        circ5.cx(0, 1)
        circ5.x(0)
        circ5.x(1)
        elem5 = CNOTDihedral(circ5)
        self.assertEqual(elem1, elem5, "Error: 2-qubit CX identity does not hold")

        # IT*CX01*CX10*TdgI = CX01*CX10
        #
        #           ┌───┐
        # q_0: ──■──┤ X ├
        #      ┌─┴─┐└─┬─┘
        # q_1: ┤ X ├──■──
        #      └───┘
        circ1 = QuantumCircuit(2)
        circ1.cx(0, 1)
        circ1.cx(1, 0)

        #                ┌───┐┌─────┐
        # q_0: ───────■──┤ X ├┤ Tdg ├
        #      ┌───┐┌─┴─┐└─┬─┘└─────┘
        # q_1: ┤ T ├┤ X ├──■─────────
        #      └───┘└───┘
        circ2 = QuantumCircuit(2)
        circ2.t(1)
        circ2.cx(0, 1)
        circ2.cx(1, 0)
        circ2.tdg(0)
        elem1 = CNOTDihedral(circ1)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 2-qubit CX01*CX10 identity does not hold")

    def test_ccz_identities(self):
        """Tests identities for CCZ gate"""

        # Check the definition of a CCZ gate
        circ_ccz = CCZGate()
        elem_ccz = CNOTDihedral(circ_ccz)

        circ1 = QuantumCircuit(3)
        circ1.ccz(0, 1, 2)
        elem1 = CNOTDihedral(circ1)
        self.assertEqual(elem1, elem_ccz, "Error: 3-qubit CCZ identity does not hold")

        circ2 = QuantumCircuit(3)
        circ2.cx(1, 2)
        circ2.tdg(2)
        circ2.cx(0, 2)
        circ2.t(2)
        circ2.cx(1, 2)
        circ2.t(1)
        circ2.tdg(2)
        circ2.cx(0, 2)
        circ2.cx(0, 1)
        circ2.t(2)
        circ2.t(0)
        circ2.tdg(1)
        circ2.cx(0, 1)
        elem2 = CNOTDihedral(circ2)
        self.assertEqual(elem1, elem2, "Error: 3-qubit CCZ identity does not hold")

        # Check that a CCZ gate is the same when permuting the qubits
        circ3 = QuantumCircuit(3)
        circ3.ccz(0, 2, 1)
        elem3 = CNOTDihedral(circ3)
        self.assertEqual(elem1, elem3, "Error: 3-qubit CCZ identity does not hold")

        circ4 = QuantumCircuit(3)
        circ4.ccz(1, 0, 2)
        elem4 = CNOTDihedral(circ4)
        self.assertEqual(elem1, elem4, "Error: 3-qubit CCZ identity does not hold")

        circ5 = QuantumCircuit(3)
        circ5.ccz(1, 2, 0)
        elem5 = CNOTDihedral(circ5)
        self.assertEqual(elem1, elem5, "Error: 3-qubit CCZ identity does not hold")

        circ6 = QuantumCircuit(3)
        circ6.ccz(2, 0, 1)
        elem6 = CNOTDihedral(circ6)
        self.assertEqual(elem1, elem6, "Error: 3-qubit CCZ identity does not hold")

        circ7 = QuantumCircuit(3)
        circ7.ccz(1, 2, 0)
        elem7 = CNOTDihedral(circ7)
        self.assertEqual(elem1, elem7, "Error: 3-qubit CCZ identity does not hold")

    def test_random_decompose(self):
        """
        Test that random elements are CNOTDihedral
        and to_circuit, to_instruction, _from_circuit, _is_valid methods
        """
        rng = np.random.default_rng(1234)
        samples = 10
        for num_qubits in range(1, 9):
            for _ in range(samples):
                # Test of random_cnotdihedral method
                elem = random_cnotdihedral(num_qubits, seed=rng)
                self.assertIsInstance(
                    elem, CNOTDihedral, "Error: random element is not CNOTDihedral"
                )
                self.assertTrue(elem._is_valid(), "Error: random element is not CNOTDihedral")

                # Test of to_circuit and _from_circuit methods
                test_circ = elem.to_circuit()
                self.assertTrue(
                    test_circ,
                    "Error: cannot decompose a random CNOTDihedral element to a circuit",
                )
                test_elem = CNOTDihedral(test_circ)

                self.assertEqual(
                    elem,
                    test_elem,
                    "Error: decomposed circuit is not equal to the original circuit",
                )
                # Test that _is_valid fails if linear part is wrong
                test_elem.linear = np.zeros((num_qubits, num_qubits))
                value = test_elem._is_valid()
                self.assertFalse(value, "Error: CNOTDihedral _is_valid is not correct.")

                # Test of to_instruction and _from_circuit methods
                test_gates = elem.to_instruction()
                self.assertIsInstance(
                    test_gates,
                    Gate,
                    "Error: cannot decompose a random CNOTDihedral element to a Gate",
                )
                self.assertEqual(
                    test_gates.num_qubits,
                    test_circ.num_qubits,
                    "Error: wrong num_qubits in decomposed gates",
                )
                test_elem1 = CNOTDihedral(test_gates)
                self.assertEqual(
                    elem,
                    test_elem1,
                    "Error: decomposed gates are not equal to the original gates",
                )

    @combine(num_qubits=[1, 2])
    def test_synth_two_qubits(self, num_qubits):
        """Test synthesis for set of {num_qubits}-qubit CNOTDihedral"""
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = CNOTDihedral(circ)
            value = CNOTDihedral(synth_cnotdihedral_two_qubits(target))
            self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_synth_general(self, num_qubits):
        """Test synthesis for set of {num_qubits}-qubit CNOTDihedral"""
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = CNOTDihedral(circ)
            value = CNOTDihedral(synth_cnotdihedral_general(target))
            self.assertEqual(value, target)

    @combine(num_qubits=[1, 2, 3, 4, 5])
    def test_synth_full(self, num_qubits):
        """Test synthesis for set of {num_qubits}-qubit CNOTDihedral"""
        rng = np.random.default_rng(1234)
        samples = 10
        for _ in range(samples):
            circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
            target = CNOTDihedral(circ)
            value = CNOTDihedral(synth_cnotdihedral_full(target))
            self.assertEqual(value, target)

    def test_init_circuit_decompose(self):
        """
        Test initialization from circuit and to_circuit, to_instruction methods
        """
        rng = np.random.default_rng(1234)
        samples = 10
        for num_qubits in range(1, 9):
            for _ in range(samples):
                # Test of to_circuit and _from_circuit methods
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                test_circ = elem.to_circuit()
                test_elem = CNOTDihedral(test_circ)
                self.assertEqual(
                    elem,
                    test_elem,
                    "Error: decomposed gates are not equal to the original gates",
                )

                # Test of to_instruction and _from_circuit methods
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                test_circ = elem.to_instruction()
                test_elem = CNOTDihedral(test_circ)
                self.assertEqual(
                    elem,
                    test_elem,
                    "Error: decomposed gates are not equal to the original gates",
                )

    def test_compose_method(self):
        """Test compose method"""
        samples = 10
        rng = np.random.default_rng(111)
        for num_qubits in range(1, 6):
            for _ in range(samples):
                circ1 = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                circ2 = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem1 = CNOTDihedral(circ1)
                elem2 = CNOTDihedral(circ2)
                value = elem1.compose(elem2)
                target = CNOTDihedral(circ1.compose(circ2))
                self.assertEqual(target, value, "Error: composed circuit is not the same")

    def test_dot_method(self):
        """Test dot method"""
        samples = 10
        rng = np.random.default_rng(222)
        for num_qubits in range(1, 6):
            for _ in range(samples):
                circ1 = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                circ2 = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem1 = CNOTDihedral(circ1)
                elem2 = CNOTDihedral(circ2)
                value = elem1.dot(elem2)
                target = CNOTDihedral(circ2.compose(circ1))
                self.assertEqual(target, value, "Error: composed circuit is not the same")

    def test_tensor_method(self):
        """Test tensor method"""
        samples = 10
        rng = np.random.default_rng(333)
        for num_qubits_1 in range(1, 5):
            for num_qubits_2 in range(1, 5):
                for _ in range(samples):
                    elem1 = random_cnotdihedral(num_qubits_1, seed=rng)
                    elem2 = random_cnotdihedral(num_qubits_2, seed=rng)
                    circ1 = elem1.to_instruction()
                    circ2 = elem2.to_instruction()
                    value = elem1.tensor(elem2)
                    circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
                    qargs = list(range(num_qubits_1))
                    for instruction in circ1.definition:
                        new_qubits = [
                            qargs[circ1.definition.qubits.index(tup)] for tup in instruction.qubits
                        ]
                        circ.append(instruction.operation, new_qubits)
                    qargs = list(range(num_qubits_1, num_qubits_1 + num_qubits_2))
                    for instruction in circ2.definition:
                        new_qubits = [
                            qargs[circ2.definition.qubits.index(tup)] for tup in instruction.qubits
                        ]
                        circ.append(instruction.operation, new_qubits)
                    target = CNOTDihedral(circ)

                    self.assertEqual(target, value, "Error: tensor circuit is not the same")

    def test_expand_method(self):
        """Test expand method"""
        samples = 10
        rng = np.random.default_rng(333)
        for num_qubits_1 in range(1, 5):
            for num_qubits_2 in range(1, 5):
                for _ in range(samples):
                    elem1 = random_cnotdihedral(num_qubits_1, seed=rng)
                    elem2 = random_cnotdihedral(num_qubits_2, seed=rng)
                    circ1 = elem1.to_instruction()
                    circ2 = elem2.to_instruction()
                    value = elem2.expand(elem1)
                    circ = QuantumCircuit(num_qubits_1 + num_qubits_2)
                    qargs = list(range(num_qubits_1))
                    for instruction in circ1.definition:
                        new_qubits = [
                            qargs[circ1.definition.qubits.index(tup)] for tup in instruction.qubits
                        ]
                        circ.append(instruction.operation, new_qubits)
                    qargs = list(range(num_qubits_1, num_qubits_1 + num_qubits_2))
                    for instruction in circ2.definition:
                        new_qubits = [
                            qargs[circ2.definition.qubits.index(tup)] for tup in instruction.qubits
                        ]
                        circ.append(instruction.operation, new_qubits)
                    target = CNOTDihedral(circ)

                    self.assertEqual(target, value, "Error: expand circuit is not the same")

    def test_adjoint(self):
        """Test transpose method"""
        samples = 10
        rng = np.random.default_rng(555)
        for num_qubits in range(1, 5):
            for _ in range(samples):
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                value = elem.adjoint().to_operator()
                target = Operator(circ).adjoint()
                self.assertTrue(target.equiv(value), "Error: adjoint circuit is not the same")

    def test_transpose(self):
        """Test transpose method"""
        samples = 10
        rng = np.random.default_rng(666)
        for num_qubits in range(1, 5):
            for _ in range(samples):
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                value = elem.transpose().to_operator()
                target = Operator(circ).transpose()
                self.assertTrue(target.equiv(value), "Error: transpose circuit is not the same")

    def test_conjugate(self):
        """Test transpose method"""
        samples = 10
        rng = np.random.default_rng(777)
        for num_qubits in range(1, 5):
            for _ in range(samples):
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                value = elem.conjugate().to_operator()
                target = Operator(circ).conjugate()
                self.assertTrue(target.equiv(value), "Error: conjugate circuit is not the same")

    def test_to_matrix(self):
        """Test to_matrix method"""
        samples = 10
        rng = np.random.default_rng(888)
        for num_qubits in range(1, 5):
            for _ in range(samples):
                circ = random_cnotdihedral_circuit(num_qubits, 5 * num_qubits, seed=rng)
                elem = CNOTDihedral(circ)
                mat = elem.to_matrix()
                self.assertIsInstance(mat, np.ndarray)
                self.assertEqual(mat.shape, 2 * (2**num_qubits,))
                value = Operator(mat)
                target = Operator(circ)
                self.assertTrue(value.equiv(target), "Error: matrix of the circuit is not the same")

    def test_init_from_pauli(self):
        """Test initialization from Pauli"""
        samples = 10
        rng = np.random.default_rng(999)
        for num_qubits in range(1, 5):
            for _ in range(samples):
                pauli = random_pauli(num_qubits, seed=rng)
                elem = CNOTDihedral(pauli)
                value = Operator(pauli)
                target = Operator(elem)
                self.assertTrue(value.equiv(target), "Error: Pauli operator is not the same.")

    def test_barrier_delay_sim(self):
        """Test barrier and delay instructions can be simulated"""
        target_circ = QuantumCircuit(2)
        target_circ.x(0)
        target_circ.cx(0, 1)
        target = CNOTDihedral(target_circ)

        circ = QuantumCircuit(2)
        circ.x(0)
        circ.delay(100, 0)
        circ.barrier([0, 1])
        circ.cx(0, 1)
        value = CNOTDihedral(circ)
        self.assertEqual(value, target)


if __name__ == "__main__":
    unittest.main()
