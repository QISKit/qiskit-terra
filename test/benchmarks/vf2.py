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


import dataclasses

import numpy as np

from qiskit.circuit import Measure, Parameter, QuantumCircuit, library as lib
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import Target, CouplingMap, InstructionProperties
from qiskit.transpiler.passes import VF2Layout

HEAVY_HEX_DISTANCE = {57: 5, 115: 7, 1081: 21}
GRID_SIZE = {49: 7, 121: 11, 256: 16}
DAG_LINE = (10, 20, 57)


@dataclasses.dataclass
class State:
    heavy_hex: dict[int, Target]
    grid: dict[int, Target]
    dag: dict[tuple[str, int], DAGCircuit]


def target_from_coupling(coupling, rng):
    target = Target(coupling.size())
    error_1q = 10 ** rng.normal(-4.0, 0.2, (coupling.size(),))
    error_2q = 10 ** rng.normal(-2.0, 0.5, (len(coupling.get_edges()),))
    error_measure = 10 ** rng.normal(-3.0, 0.4, (coupling.size(),))
    target.add_instruction(
        lib.RZGate(Parameter("a")),
        {(i,): InstructionProperties(error=0.0) for i in coupling.physical_qubits},
    )
    target.add_instruction(
        lib.SXGate(),
        {
            (i,): InstructionProperties(error=error)
            for i, error in zip(coupling.physical_qubits, error_1q)
        },
    )
    target.add_instruction(
        Measure(),
        {
            (i,): InstructionProperties(error=error)
            for i, error in zip(coupling.physical_qubits, error_measure)
        },
    )
    target.add_instruction(
        lib.CXGate(),
        {pair: InstructionProperties(error=error) for pair, error in zip(coupling, error_2q)},
    )
    return target


def exact_dag_from_coupling(coupling, repeats=1):
    qc = QuantumCircuit(coupling.size(), coupling.size())
    for _ in range(repeats):
        for left, right in coupling:
            qc.sx(left)
            qc.rz(0.5, left)
            qc.sx(left)
            qc.sx(right)
            qc.rz(1.25, right)
            qc.sx(right)
            qc.cx(left, right)
    qc.measure(qc.qubits, qc.clbits)
    return circuit_to_dag(qc, copy_operations=False)


def setup_cache():
    rng = lambda: np.random.default_rng(2025_04_08)
    out = State({}, {}, {})
    for num_qubits, distance in HEAVY_HEX_DISTANCE.items():
        cm = CouplingMap.from_heavy_hex(distance, bidirectional=False)
        assert num_qubits == cm.size()
        out.heavy_hex[num_qubits] = target_from_coupling(cm, rng())
        out.dag["heavy hex", num_qubits] = exact_dag_from_coupling(cm)
    for num_qubits, side in GRID_SIZE.items():
        cm = CouplingMap.from_grid(side, side, bidirectional=False)
        assert num_qubits == cm.size()
        out.grid[num_qubits] = target_from_coupling(cm, rng())
        out.dag["grid", num_qubits] = exact_dag_from_coupling(cm)
    for num_qubits in DAG_LINE:
        out.dag["line", num_qubits] = exact_dag_from_coupling(
            CouplingMap.from_line(num_qubits, bidirectional=False), repeats=5
        )
    return out


class VF2LayoutSuite:
    # Set the timeout low so that we can maintain the same benchmark suite for older commits that
    # are still slow, without exploding the total runtime.
    timeout = 20.0

    # Heavy hex stuff.

    def time_heavy_hex_line(
        self,
        state: State,
        num_physical_qubits,
        directional,
        line_qubits,
        call_limit,
    ):
        pass_ = VF2Layout(
            seed=-1,
            target=state.heavy_hex[num_physical_qubits],
            strict_direction=directional,
            max_trials=0,
            call_limit=call_limit,
        )
        pass_.run(state.dag["line", line_qubits])

    time_heavy_hex_line.params = ((115, 1081), (False, True), (10, 57), (None, 1_000_000))
    time_heavy_hex_line.param_names = (
        "num_physical_qubits",
        "directional",
        "line_qubits",
        "call_limit",
    )

    def time_heavy_hex_trivial(self, state: State, num_physical_qubits):
        pass_ = VF2Layout(
            seed=-1,
            target=state.heavy_hex[num_physical_qubits],
            strict_direction=False,
            max_trials=0,
        )
        pass_.run(state.dag["heavy hex", num_physical_qubits])

    time_heavy_hex_trivial.params = ((57,),)
    time_heavy_hex_trivial.param_names = ("num_physical_qubits",)

    def time_heavy_hex_impossible(self, state: State, num_physical_qubits, call_limit):
        pass_ = VF2Layout(
            seed=-1,
            target=state.heavy_hex[num_physical_qubits],
            strict_direction=False,
            max_trials=0,
            call_limit=call_limit,
        )
        pass_.run(state.dag["line", num_physical_qubits])

    time_heavy_hex_impossible.params = ((57,), (None, 1_000_000))
    time_heavy_hex_impossible.param_names = ("num_physical_qubits", "call_limit")

    # Grid stuff.

    def time_grid_line(
        self, state: State, num_physical_qubits, directional, line_qubits, call_limit
    ):
        pass_ = VF2Layout(
            seed=-1,
            target=state.grid[num_physical_qubits],
            strict_direction=directional,
            max_trials=0,
            call_limit=call_limit,
        )
        pass_.run(state.dag["line", line_qubits])

    time_grid_line.params = ((121, 256), (False, True), (10, 20), (None, 1_000_000))
    time_grid_line.param_names = ("num_physical_qubits", "directional", "line_qubits", "call_limit")

    def time_grid_trivial(self, state: State, num_physical_qubits):
        pass_ = VF2Layout(
            seed=-1,
            target=state.grid[num_physical_qubits],
            strict_direction=False,
            max_trials=0,
        )
        pass_.run(state.dag["grid", num_physical_qubits])

    time_grid_trivial.params = ((49,),)
    time_grid_trivial.param_names = ("num_physical_qubits",)
