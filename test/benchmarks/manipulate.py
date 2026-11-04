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

from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.circuit import pauli_twirl_2q_gates
from qiskit.circuit.library import quantum_volume
from qiskit.passmanager import PropertySet
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from .utils import multi_control_circuit


class TestCircuitManipulate:
    def setup(self):
        qasm_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qasm")
        self.qft_qasm = os.path.join(qasm_dir, "dtc_100_cx_12345.qasm")
        self.qft_qc = QuantumCircuit.from_qasm_file(self.qft_qasm)

        qv_circuit = quantum_volume(100, seed=2025_12345)
        qv_circuit.measure_all()
        backend = GenericBackendV2(100, coupling_map=CouplingMap.from_line(100), seed=2025_12345)
        self.qv_qc = transpile(
            qv_circuit, backend, optimization_level=0, seed_transpiler=2025_12345
        )

        self.dtc_qasm = os.path.join(qasm_dir, "dtc_100_cx_12345.qasm")
        self.dtc_qc = QuantumCircuit.from_qasm_file(self.dtc_qasm)
        self.translate = generate_preset_pass_manager(1, basis_gates=["rx", "ry", "rz", "cz"])

    def time_DTC100_twirling(self):
        """Perform Pauli-twirling on a 100Q QV
        circuit
        """
        out = pauli_twirl_2q_gates(self.dtc_qc, seed=12345678942)
        return out

    def time_multi_control_decompose(self):
        """Decompose a multi-control gate into the
        basis [rx, ry, rz, cz]
        """
        circ = multi_control_circuit(16)
        self.translate.property_set = PropertySet()
        out = self.translate.run(circ)
        return out

    def time_QV100_basis_change(self):
        """Change a QV100 circuit basis from [rx, ry, rz, cx]
        to [sx, x, rz, cz]
        """
        self.translate.property_set = PropertySet()
        out = self.translate.run(self.qv_qc)
        return out
