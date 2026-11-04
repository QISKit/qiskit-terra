# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring
# pylint: disable=attribute-defined-outside-init

from qiskit.transpiler import Target
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import get_clifford_gate_names
from .utils import create_ft_circuit


class TranspilerCliffordRZBenchmarks:
    circuit_names = ["qft", "trotter", "qaoa", "grover", "mcx", "multiplier", "modular_adder"]
    num_qubits = [8, 16, 32, 64, 128]
    optimization_level = [0, 1, 2, 3]
    params = (circuit_names, num_qubits, optimization_level)
    param_names = ["circuit_name", "n_qubits", "optimization_level"]
    timeout = 300

    def setup(self, circuit_name, n_qubits, optimization_level):
        # List of slow tests that we want to exclude for now
        slow_tests = {
            ("qaoa", 64, 3),
            ("qaoa", 128, 2),
            ("qaoa", 128, 3),
            ("multiplier", 64, 2),
            ("multiplier", 64, 3),
            ("multiplier", 128, 2),
            ("multiplier", 128, 3),
        }

        if (circuit_name, n_qubits, optimization_level) in slow_tests:
            raise NotImplementedError

        self.circuit = create_ft_circuit(circuit_name, n_qubits)

        target = Target.from_configuration(["rz", "measure"] + get_clifford_gate_names(), n_qubits)
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level, target=target, seed_transpiler=0
        )

    def time_transpile(self, _, __, ___):
        self.pm.run(self.circuit)

    def track_rz_count(self, _, __, ___):
        res = self.pm.run(self.circuit)
        return res.count_ops().get("rz", 0)


class TranspilerCliffordTBenchmarks:
    # Note: QAOA circuits are not included because they are parametric, and
    # for Trotter circuits we decrease the number of reps to 2.
    circuit_names = ["qft", "trotter", "grover", "mcx", "multiplier", "modular_adder"]
    num_qubits = [4, 8]
    optimization_level = [0, 1, 2, 3]
    params = (circuit_names, num_qubits, optimization_level)
    param_names = ["circuit_name", "n_qubits", "optimization_level"]
    timeout = 300

    def setup(self, circuit_name, n_qubits, optimization_level):
        slow_tests = {
            ("grover", 8, 1),
            ("grover", 8, 2),
            ("grover", 8, 3),
        }

        if (circuit_name, n_qubits, optimization_level) in slow_tests:
            raise NotImplementedError

        self.circuit = create_ft_circuit(circuit_name, n_qubits, reps=2)
        target = Target.from_configuration(
            ["t", "tdg", "measure"] + get_clifford_gate_names(), n_qubits
        )
        self.pm = generate_preset_pass_manager(
            optimization_level=optimization_level,
            target=target,
            seed_transpiler=0,
        )

    def time_transpile(self, _, __, ___):
        self.pm.run(self.circuit)

    def track_t_count(self, _, __, ___):
        res = self.pm.run(self.circuit)
        ops = res.count_ops()
        return ops.get("t", 0) + ops.get("tdg", 0)
