#!/usr/bin/env python3

# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Script to generate 'utility scale' load for profiling in a PGO context"""

import os
from qiskit import qasm2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit.circuit.library import quantum_volume
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

QASM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "test",
    "benchmarks",
    "qasm",
)


def _main():
    cmap = CouplingMap.from_heavy_hex(9)
    cz_backend = GenericBackendV2(
        cmap.size(),
        ["rz", "x", "sx", "cz", "id"],
        coupling_map=cmap,
        control_flow=True,
        seed=12345678942,
    )
    ecr_backend = GenericBackendV2(
        cmap.size(),
        ["rz", "x", "sx", "ecr", "id"],
        coupling_map=cmap,
        control_flow=True,
        seed=12345678942,
    )
    cx_backend = GenericBackendV2(
        cmap.size(),
        ["rz", "x", "sx", "cx", "id"],
        coupling_map=cmap,
        control_flow=True,
        seed=12345678942,
    )
    cz_pm = generate_preset_pass_manager(2, cz_backend)
    ecr_pm = generate_preset_pass_manager(2, ecr_backend)
    cx_pm = generate_preset_pass_manager(2, cx_backend)
    qft_circ = qasm2.load(
        os.path.join(QASM_DIR, "qft_N100.qasm"),
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
        strict=False,
    )
    qft_circ.name = "qft_N100"
    square_heisenberg_circ = qasm2.load(
        os.path.join(QASM_DIR, "square_heisenberg_N100.qasm"),
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
        strict=False,
    )
    square_heisenberg_circ.name = "square_heisenberg_N100"
    qaoa_circ = qasm2.load(
        os.path.join(QASM_DIR, "qaoa_barabasi_albert_N100_3reps.qasm"),
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
        strict=False,
    )
    qaoa_circ.name = "qaoa_barabasi_albert_N100_3reps"
    qv_circ = quantum_volume(100, seed=123456789)
    qv_circ.measure_all()
    qv_circ.name = "QV1267650600228229401496703205376"
    hwb_circ = qasm2.load(
        os.path.join(QASM_DIR, "hwb12.qasm"),
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
        strict=False,
    )
    hwb_circ.name = "hwb12"

    for pm in [cz_pm, ecr_pm, cx_pm]:
        for circ in [qft_circ, square_heisenberg_circ, qaoa_circ, qv_circ, hwb_circ]:
            print(f"Compiling: {circ.name}")
            pm.run(circ)


if __name__ == "__main__":
    _main()
