# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


import os

import qiskit


class TranspilerBenchSuite:
    def _build_cx_circuit(self):
        cx_register = qiskit.QuantumRegister(2)
        cx_circuit = qiskit.QuantumCircuit(cx_register)
        cx_circuit.h(cx_register[0])
        cx_circuit.h(cx_register[0])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        cx_circuit.cx(cx_register[0], cx_register[1])
        return cx_circuit

    def _build_single_gate_circuit(self):
        single_register = qiskit.QuantumRegister(1)
        single_gate_circuit = qiskit.QuantumCircuit(single_register)
        single_gate_circuit.h(single_register[0])
        return single_gate_circuit

    def setup(self):
        self.single_gate_circuit = self._build_single_gate_circuit()
        self.cx_circuit = self._build_cx_circuit()
        self.qasm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "qasm"))
        large_qasm_path = os.path.join(self.qasm_path, "test_eoh_qasm.qasm")
        self.large_qasm = qiskit.QuantumCircuit.from_qasm_file(large_qasm_path)
        self.coupling_map = [
            [0, 1],
            [1, 0],
            [1, 2],
            [1, 4],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 5],
            [4, 1],
            [4, 7],
            [5, 3],
            [5, 8],
            [6, 7],
            [7, 4],
            [7, 6],
            [7, 10],
            [8, 5],
            [8, 9],
            [8, 11],
            [9, 8],
            [10, 7],
            [10, 12],
            [11, 8],
            [11, 14],
            [12, 10],
            [12, 13],
            [12, 15],
            [13, 12],
            [13, 14],
            [14, 11],
            [14, 13],
            [14, 16],
            [15, 12],
            [15, 18],
            [16, 14],
            [16, 19],
            [17, 18],
            [18, 15],
            [18, 17],
            [18, 21],
            [19, 16],
            [19, 20],
            [19, 22],
            [20, 19],
            [21, 18],
            [21, 23],
            [22, 19],
            [22, 25],
            [23, 21],
            [23, 24],
            [24, 23],
            [24, 25],
            [25, 22],
            [25, 24],
            [25, 26],
            [26, 25],
        ]
        self.basis = ["id", "rz", "sx", "x", "cx", "reset"]

    def time_single_gate_compile(self):
        qiskit.compiler.transpile(
            self.single_gate_circuit,
            coupling_map=self.coupling_map,
            basis_gates=self.basis,
            seed_transpiler=20220125,
        )

    def time_cx_compile(self):
        qiskit.compiler.transpile(
            self.cx_circuit,
            coupling_map=self.coupling_map,
            basis_gates=self.basis,
            seed_transpiler=20220125,
        )

    def time_compile_from_large_qasm(self):
        qiskit.compiler.transpile(
            self.large_qasm,
            coupling_map=self.coupling_map,
            basis_gates=self.basis,
            seed_transpiler=20220125,
        )
