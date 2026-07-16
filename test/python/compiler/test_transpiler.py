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

"""Tests basic functionality of the transpile function"""

import copy
import io
import itertools
import math
import os
import sys
import random
from logging import StreamHandler, getLogger
from unittest.mock import patch
import numpy as np
from numpy import pi
import rustworkx as rx
from ddt import data, idata, ddt, unpack

from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    qasm3,
    qpy,
)
from qiskit.circuit import (
    Annotation,
    Clbit,
    ControlFlowOp,
    ForLoopOp,
    Gate,
    IfElseOp,
    BoxOp,
    Parameter,
    ParameterVector,
    Qubit,
    SwitchCaseOp,
    WhileLoopOp,
    Duration,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.annotated_operation import (
    AnnotatedOperation,
    InverseModifier,
    ControlModifier,
    PowerModifier,
)
from qiskit.circuit.delay import Delay
from qiskit.circuit.measure import Measure
from qiskit.circuit.reset import Reset
from qiskit.circuit.library import (
    CXGate,
    CZGate,
    ECRGate,
    HGate,
    IGate,
    PermutationGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SGate,
    SXGate,
    SXdgGate,
    SdgGate,
    U2Gate,
    UGate,
    XGate,
    ZGate,
    XXPlusYYGate,
    RZZGate,
)
from qiskit.compiler import transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGOutNode, DAGCircuit
from qiskit.exceptions import QiskitError
from qiskit.providers.backend import BackendV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.providers.options import Options
from qiskit.quantum_info import Operator, random_unitary
from qiskit.utils import should_run_in_parallel
from qiskit.transpiler import CouplingMap, Layout, PassManager, passes
from qiskit.transpiler.exceptions import TranspilerError, CircuitTooWideForTarget
from qiskit.transpiler.passes import (
    BarrierBeforeFinalMeasurements,
    GateDirection,
    VF2PostLayout,
    WrapAngles,
    SabreLayout,
)

from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager, level_0_pass_manager
from qiskit.transpiler.target import InstructionProperties, Target
from qiskit.transpiler.timing_constraints import TimingConstraints
from qiskit.transpiler import WrapAngleRegistry

from test import QiskitTestCase, combine, slow_test

from ..legacy_cmaps import MELBOURNE_CMAP, RUESCHLIKON_CMAP, TOKYO_CMAP, MUMBAI_CMAP


class CustomCX(Gate):
    """Custom CX gate representation."""

    def __init__(self):
        super().__init__("custom_cx", 2, [])

    def _define(self):
        self._definition = QuantumCircuit(2)
        self._definition.cx(0, 1)


class AlignmentBackend(BackendV2):
    """A backend with arbitrary alignment constraints."""

    def __init__(self, num_qubits, control_flow=False):
        super().__init__()
        self._target = Target.from_configuration(
            basis_gates=["rz", "sx", "cx", "delay", "measure"],
            coupling_map=CouplingMap.from_line(num_qubits),
            timing_constraints=TimingConstraints(
                granularity=2, min_length=4, pulse_alignment=4, acquire_alignment=4
            ),
        )
        if control_flow:
            self._target.add_instruction(IfElseOp, name="if_else")
            self._target.add_instruction(ForLoopOp, name="for_loop")
            self._target.add_instruction(WhileLoopOp, name="while_loop")
            self._target.add_instruction(SwitchCaseOp, name="switch_case")

    @property
    def target(self):
        return self._target

    @property
    def max_circuits(self):
        return 1

    @classmethod
    def _default_options(cls):
        return Options()

    def run(self, run_input, **_options):
        pass


def connected_qubits(physical: int, coupling_map: CouplingMap) -> set:
    """Get the physical qubits that have a connection to this one in the coupling map."""
    for component in coupling_map.connected_components():
        if physical in (qubits := set(component.graph.nodes())):
            return qubits
    raise ValueError(f"physical qubit {physical} is not in the coupling map")


@ddt
class TestTranspile(QiskitTestCase):
    """Test transpile function."""

    def test_empty_transpilation(self):
        """Test that transpiling an empty list is a no-op.  Regression test of gh-7287."""
        self.assertEqual(transpile([], seed_transpiler=42), [])

    def test_pass_manager_none(self):
        """Test passing the default (None) pass manager to the transpiler.

        It should perform the default qiskit flow:
        unroll, swap_mapper, cx_direction, cx_cancellation, optimize_1q_gates
        and should be equivalent to using tools.compile
        """
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        coupling_map = [[1, 0]]
        basis_gates = ["u1", "u2", "u3", "cx", "id"]

        backend = BasicSimulator()
        circuit2 = transpile(
            circuit,
            backend=backend,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            seed_transpiler=42,
        )

        circuit3 = transpile(
            circuit,
            backend=backend,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            seed_transpiler=42,
        )
        self.assertEqual(circuit2, circuit3)

    @data(0, 1, 2, 3)
    def test_num_processes_kwarg_concurrent_default(self, num_processes):
        """Test that num_processes kwarg works when the system default parallel is false"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = GenericBackendV2(
            num_qubits=27, coupling_map=CouplingMap.from_line(27), seed=42
        ).target
        res = transpile([qc] * 3, target=target, num_processes=num_processes)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    def test_transpile_basis_gates_no_backend_no_coupling_map(self):
        """Verify transpile() works with no coupling_map or backend."""
        qr = QuantumRegister(2, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        basis_gates = ["u1", "u2", "u3", "cx", "id"]
        circuit2 = transpile(
            circuit, basis_gates=basis_gates, optimization_level=0, seed_transpiler=42
        )
        resources_after = circuit2.count_ops()
        self.assertEqual({"u2": 2, "cx": 4}, resources_after)

    def test_transpile_non_adjacent_layout(self):
        """Transpile pipeline can handle manual layout on non-adjacent qubits.

        circuit:

        .. code-block:: text

                  ┌───┐
            qr_0: ┤ H ├──■──────────── -> 1
                  └───┘┌─┴─┐
            qr_1: ─────┤ X ├──■─────── -> 2
                       └───┘┌─┴─┐
            qr_2: ──────────┤ X ├──■── -> 3
                            └───┘┌─┴─┐
            qr_3: ───────────────┤ X ├ -> 5
                                 └───┘

        device:
        0  -  1  -  2  -  3  -  4  -  5  -  6

              |     |     |     |     |     |

              13 -  12  - 11 -  10 -  9  -  8  -   7
        """
        cmap = [
            [0, 1],
            [0, 14],
            [1, 0],
            [1, 2],
            [1, 13],
            [2, 1],
            [2, 3],
            [2, 12],
            [3, 2],
            [3, 4],
            [3, 11],
            [4, 3],
            [4, 5],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 5],
            [6, 8],
            [7, 8],
            [8, 6],
            [8, 7],
            [8, 9],
            [9, 5],
            [9, 8],
            [9, 10],
            [10, 4],
            [10, 9],
            [10, 11],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [12, 11],
            [12, 13],
            [13, 1],
            [13, 12],
            [13, 14],
            [14, 0],
            [14, 13],
        ]

        qr = QuantumRegister(4, "qr")
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])

        backend = GenericBackendV2(
            num_qubits=15, basis_gates=["ecr", "id", "rz", "sx", "x"], coupling_map=cmap, seed=42
        )
        initial_layout = [None, qr[0], qr[1], qr[2], None, qr[3]]

        new_circuit = transpile(
            circuit,
            basis_gates=backend.operation_names,
            coupling_map=backend.coupling_map,
            initial_layout=initial_layout,
            seed_transpiler=42,
        )

        qubit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qubits)}

        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([qubit_indices[x] for x in instruction.qubits], backend.coupling_map)

    def test_transpile_qft_grid(self):
        """Transpile pipeline can handle 8-qubit QFT on 14-qubit grid."""

        basis_gates = ["cx", "id", "rz", "sx", "x"]

        qr = QuantumRegister(8)
        circuit = QuantumCircuit(qr)
        for i, q in enumerate(qr):
            for j in range(i):
                circuit.cp(math.pi / float(2 ** (i - j)), q, qr[j])
            circuit.h(q)

        new_circuit = transpile(
            circuit, basis_gates=basis_gates, coupling_map=MELBOURNE_CMAP, seed_transpiler=42
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_circuit.qubits)}
        for instruction in new_circuit.data:
            if isinstance(instruction.operation, CXGate):
                self.assertIn([qubit_indices[x] for x in instruction.qubits], MELBOURNE_CMAP)

    def test_already_mapped_1(self):
        """Circuit not remapped if matches topology.

        See: https://github.com/Qiskit/qiskit-terra/issues/342
        """

        backend = GenericBackendV2(num_qubits=16, coupling_map=RUESCHLIKON_CMAP, seed=42)
        coupling_map = backend.coupling_map
        basis_gates = backend.operation_names

        qr = QuantumRegister(16, "qr")
        cr = ClassicalRegister(16, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.cx(qr[3], qr[14])
        qc.cx(qr[5], qr[4])
        qc.h(qr[9])
        qc.cx(qr[9], qr[8])
        qc.x(qr[11])
        qc.cx(qr[3], qr[4])
        qc.cx(qr[12], qr[11])
        qc.cx(qr[13], qr[4])
        qc.measure(qr, cr)

        new_qc = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            initial_layout=Layout.generate_trivial_layout(qr),
            seed_transpiler=42,
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
        cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
        cx_qubits_physical = [
            [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
        ]
        self.assertEqual(
            sorted(cx_qubits_physical), [[3, 4], [3, 14], [5, 4], [9, 8], [12, 11], [13, 4]]
        )

    def test_already_mapped_via_layout(self):
        """Test that a manual layout that satisfies a coupling map does not get altered.

        See: https://github.com/Qiskit/qiskit-terra/issues/2036

        circuit:

        .. code-block:: text

                  ┌───┐                  ┌───┐ ░ ┌─┐
            qn_0: ┤ H ├──■────────────■──┤ H ├─░─┤M├─── -> 9
                  └───┘  │            │  └───┘ ░ └╥┘
            qn_1: ───────┼────────────┼────────░──╫──── -> 6
                         │            │        ░  ║
            qn_2: ───────┼────────────┼────────░──╫──── -> 5
                         │            │        ░  ║
            qn_3: ───────┼────────────┼────────░──╫──── -> 0
                         │            │        ░  ║
            qn_4: ───────┼────────────┼────────░──╫──── -> 1
                  ┌───┐┌─┴─┐┌──────┐┌─┴─┐┌───┐ ░  ║ ┌─┐
            qn_5: ┤ H ├┤ X ├┤ P(2) ├┤ X ├┤ H ├─░──╫─┤M├ -> 4
                  └───┘└───┘└──────┘└───┘└───┘ ░  ║ └╥┘
            cn: 2/════════════════════════════════╩══╩═
                                                  0  1

        device:
        0 -- 1 -- 2 -- 3 -- 4
        |                   |
        5 -- 6 -- 7 -- 8 -- 9
        |                   |
        10 - 11 - 12 - 13 - 14
        |                   |
        15 - 16 - 17 - 18 - 19
        """
        basis_gates = ["u1", "u2", "u3", "cx", "id"]
        coupling_map = [
            [0, 1],
            [0, 5],
            [1, 0],
            [1, 2],
            [2, 1],
            [2, 3],
            [3, 2],
            [3, 4],
            [4, 3],
            [4, 9],
            [5, 0],
            [5, 6],
            [5, 10],
            [6, 5],
            [6, 7],
            [7, 6],
            [7, 8],
            [7, 12],
            [8, 7],
            [8, 9],
            [9, 4],
            [9, 8],
            [9, 14],
            [10, 5],
            [10, 11],
            [10, 15],
            [11, 10],
            [11, 12],
            [12, 7],
            [12, 11],
            [12, 13],
            [13, 12],
            [13, 14],
            [14, 9],
            [14, 13],
            [14, 19],
            [15, 10],
            [15, 16],
            [16, 15],
            [16, 17],
            [17, 16],
            [17, 18],
            [18, 17],
            [18, 19],
            [19, 14],
            [19, 18],
        ]

        q = QuantumRegister(6, name="qn")
        c = ClassicalRegister(2, name="cn")
        qc = QuantumCircuit(q, c)
        qc.h(q[0])
        qc.h(q[5])
        qc.cx(q[0], q[5])
        qc.p(2, q[5])
        qc.cx(q[0], q[5])
        qc.h(q[0])
        qc.h(q[5])
        qc.barrier(q)
        qc.measure(q[0], c[0])
        qc.measure(q[5], c[1])

        initial_layout = [
            q[3],
            q[4],
            None,
            None,
            q[5],
            q[2],
            q[1],
            None,
            None,
            q[0],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        new_qc = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            initial_layout=initial_layout,
            seed_transpiler=42,
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_qc.qubits)}
        cx_qubits = [instr.qubits for instr in new_qc.data if instr.operation.name == "cx"]
        cx_qubits_physical = [
            [qubit_indices[ctrl], qubit_indices[tgt]] for [ctrl, tgt] in cx_qubits
        ]
        self.assertEqual(sorted(cx_qubits_physical), [[9, 4], [9, 4]])

    def test_transpile_bell(self):
        """Test Transpile Bell.

        If all correct some should exists.
        """
        backend = BasicSimulator()

        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile(qc, backend, seed_transpiler=42)
        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_bell_discrete_basis(self):
        """Test that it's possible to transpile a very simple circuit to a discrete stabilizer-like
        basis.  In general, we do not make any guarantees about the possibility or quality of
        transpilation in these situations, but this is at least useful as a check that stuff that
        _could_ be possible remains so."""

        target = Target(num_qubits=2)
        for one_q in [XGate(), SXGate(), SXdgGate(), SGate(), SdgGate(), ZGate()]:
            target.add_instruction(one_q, {(0,): None, (1,): None})
        # This is only in one direction, and not the direction we're going to attempt to lay it out
        # onto, so we can test the basis translation.
        target.add_instruction(ECRGate(), {(1, 0): None})

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        # Try with the initial layout in both directions to ensure we're dealing with the basis
        # having only a single direction.

        # Use optimization level=1 because the synthesis that runs as part of optimization at
        # higher optimization levels will create intermediate gates that the transpiler currently
        # lacks logic to translate to a discrete basis.
        self.assertIsInstance(
            transpile(
                qc, target=target, initial_layout=[0, 1], seed_transpiler=42, optimization_level=1
            ),
            QuantumCircuit,
        )
        self.assertIsInstance(
            transpile(
                qc, target=target, initial_layout=[1, 0], seed_transpiler=42, optimization_level=1
            ),
            QuantumCircuit,
        )

    def test_transpile_one(self):
        """Test transpile a single circuit.

        Check that the top-level `transpile` function returns
        a single circuit."""
        backend = BasicSimulator()

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuit = transpile(qc, backend, seed_transpiler=42)
        self.assertIsInstance(circuit, QuantumCircuit)

    def test_transpile_two(self):
        """Test transpile two circuits.

        Check that the transpiler returns a list of two circuits.
        """
        backend = BasicSimulator()

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qubit_reg2 = QuantumRegister(2)
        clbit_reg2 = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)
        qc_extra = QuantumCircuit(qubit_reg, qubit_reg2, clbit_reg, clbit_reg2, name="extra")
        qc_extra.measure(qubit_reg, clbit_reg)
        circuits = transpile([qc, qc_extra], backend, seed_transpiler=42)
        self.assertIsInstance(circuits, list)
        self.assertEqual(len(circuits), 2)

        for circuit in circuits:
            self.assertIsInstance(circuit, QuantumCircuit)

    def test_transpile_singleton(self):
        """Test transpile a single-element list with a circuit.

        Check that `transpile` returns a single-element list.

        See https://github.com/Qiskit/qiskit-terra/issues/5260
        """
        backend = BasicSimulator()

        qubit_reg = QuantumRegister(2)
        clbit_reg = ClassicalRegister(2)
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        circuits = transpile([qc], backend, seed_transpiler=42)
        self.assertIsInstance(circuits, list)
        self.assertEqual(len(circuits), 1)
        self.assertIsInstance(circuits[0], QuantumCircuit)

    def test_transpiler_layout_from_intlist(self):
        """A list of ints gives layout to correctly map circuit.
        virtual  physical
         q1_0  -  4   ---[H]---
         q2_0  -  5
         q2_1  -  6   ---[H]---
         q3_0  -  8
         q3_1  -  9
         q3_2  -  10  ---[H]---

        """
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        qr3 = QuantumRegister(3, "qr3")
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        new_circ = transpile(
            qc,
            backend=None,
            coupling_map=cmap,
            basis_gates=["u2"],
            initial_layout=layout,
            seed_transpiler=42,
        )
        qubit_indices = {bit: idx for idx, bit in enumerate(new_circ.qubits)}
        mapped_qubits = []

        for instruction in new_circ.data:
            mapped_qubits.append(qubit_indices[instruction.qubits[0]])

        self.assertEqual(mapped_qubits, [4, 6, 10])

    def test_mapping_multi_qreg(self):
        """Test mapping works for multiple qregs."""
        backend = GenericBackendV2(num_qubits=8, coupling_map=CouplingMap.from_line(8), seed=42)
        qr = QuantumRegister(3, name="qr")
        qr2 = QuantumRegister(1, name="qr2")
        qr3 = QuantumRegister(4, name="qr3")
        cr = ClassicalRegister(3, name="cr")
        qc = QuantumCircuit(qr, qr2, qr3, cr)
        qc.h(qr[0])
        qc.cx(qr[0], qr2[0])
        qc.cx(qr[1], qr3[2])
        qc.measure(qr, cr)

        circuits = transpile(qc, backend, seed_transpiler=42)

        self.assertIsInstance(circuits, QuantumCircuit)

    def test_transpile_circuits_diff_registers(self):
        """Transpile list of circuits with different qreg names."""
        backend = GenericBackendV2(num_qubits=4, seed=42)
        circuits = []
        for _ in range(2):
            qr = QuantumRegister(2)
            cr = ClassicalRegister(2)
            circuit = QuantumCircuit(qr, cr)
            circuit.h(qr[0])
            circuit.cx(qr[0], qr[1])
            circuit.measure(qr, cr)
            circuits.append(circuit)

        circuits = transpile(circuits, backend)
        self.assertIsInstance(circuits[0], QuantumCircuit)

    def test_wrong_initial_layout(self):
        """Test transpile with a bad initial layout."""
        backend = GenericBackendV2(num_qubits=4, seed=42)

        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])
        qc.measure(qubit_reg, clbit_reg)

        bad_initial_layout = [
            QuantumRegister(3, "q")[0],
            QuantumRegister(3, "q")[1],
            QuantumRegister(3, "q")[2],
        ]

        with self.assertRaises(TranspilerError):
            transpile(qc, backend, initial_layout=bad_initial_layout)

    def test_parameterized_circuit_for_simulator(self):
        """Verify that a parameterized circuit can be transpiled for a simulator backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        qc.rz(theta, qr[0])

        transpiled_qc = transpile(qc, backend=BasicSimulator())

        expected_qc = QuantumCircuit(qr)
        expected_qc.append(RZGate(theta), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameterized_circuit_for_device(self):
        """Verify that a parameterized circuit can be transpiled for a device backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        qc.p(theta, qr[0])
        backend = GenericBackendV2(num_qubits=4, seed=42)

        transpiled_qc = transpile(
            qc,
            backend=backend,
            initial_layout=Layout.generate_trivial_layout(qr),
        )

        qr = QuantumRegister(backend.num_qubits, "q")
        expected_qc = QuantumCircuit(qr, global_phase=theta / 2.0)
        expected_qc.append(RZGate(theta), [qr[0]])

        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_simulator(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a simulator backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        square = theta * theta
        qc.rz(square, qr[0])

        transpiled_qc = transpile(qc, backend=BasicSimulator())

        expected_qc = QuantumCircuit(qr)
        expected_qc.append(RZGate(square), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_parameter_expression_circuit_for_device(self):
        """Verify that a circuit including expressions of parameters can be
        transpiled for a device backend."""
        qr = QuantumRegister(2, name="qr")
        qc = QuantumCircuit(qr)

        theta = Parameter("theta")
        square = theta * theta
        qc.rz(square, qr[0])

        backend = GenericBackendV2(num_qubits=4, seed=42)
        transpiled_qc = transpile(
            qc,
            backend=backend,
            initial_layout=Layout.generate_trivial_layout(qr),
        )

        qr = QuantumRegister(backend.num_qubits, "q")
        expected_qc = QuantumCircuit(qr)
        expected_qc.append(RZGate(square), [qr[0]])
        self.assertEqual(expected_qc, transpiled_qc)

    def test_final_measurement_barrier_for_devices(self):
        """Verify BarrierBeforeFinalMeasurements pass is called in default pipeline for devices."""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "example.qasm"))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        orig_pass = BarrierBeforeFinalMeasurements()

        with patch.object(BarrierBeforeFinalMeasurements, "run", wraps=orig_pass.run) as mock_pass:
            transpile(
                circ,
                coupling_map=RUESCHLIKON_CMAP,
                initial_layout=layout,
            )
            self.assertTrue(mock_pass.called)

    def test_do_not_run_gatedirection_with_symmetric_cm(self):
        """When the coupling map is symmetric, do not run GateDirection."""
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "example.qasm"))
        layout = Layout.generate_trivial_layout(*circ.qregs)
        coupling_map = CouplingMap.from_line(16, bidirectional=True)
        orig_pass = GateDirection(coupling_map)
        with patch.object(GateDirection, "run", wraps=orig_pass.run) as mock_pass:
            transpile(circ, coupling_map=coupling_map, initial_layout=layout)
            self.assertFalse(mock_pass.called)

    def test_do_not_run_elide_permutations_no_routing(self):
        """Test the ElidePermutations pass doesn't run if we disable routing

        See https://github.com/Qiskit/qiskit/issues/13144 for the details and
        reproduce in this test
        """
        circuit_routed = QuantumCircuit(4)
        circuit_routed.cx(0, 1)
        circuit_routed.h(1)
        circuit_routed.swap(1, 2)
        circuit_routed.cx(2, 3)
        pm = generate_preset_pass_manager(
            basis_gates=["cx", "sx", "rz"], routing_method="none", optimization_level=2
        )
        circuit_basis = pm.run(circuit_routed)
        cx_gate_qubits = []
        for instruction in circuit_basis.data:
            if instruction.name == "cx":
                cx_gate_qubits.append(instruction.qubits)
        # If we did not Elide the existing swaps then the swap should be
        # decomposed into 3 cx between 1 and 2 and there are no gates between
        # 1 and 3
        self.assertIn((circuit_basis.qubits[1], circuit_basis.qubits[2]), cx_gate_qubits)
        self.assertIn((circuit_basis.qubits[2], circuit_basis.qubits[1]), cx_gate_qubits)
        self.assertNotIn((circuit_basis.qubits[1], circuit_basis.qubits[3]), cx_gate_qubits)
        self.assertNotIn((circuit_basis.qubits[3], circuit_basis.qubits[1]), cx_gate_qubits)

    def test_optimize_to_nothing(self):
        """Optimize gates up to fixed point in the default pipeline
        See https://github.com/Qiskit/qiskit-terra/issues/2035
        """
        #       ┌───┐     ┌───┐┌───┐┌───┐     ┌───┐
        # q0_0: ┤ H ├──■──┤ X ├┤ Y ├┤ Z ├──■──┤ H ├──■────■──
        #       └───┘┌─┴─┐└───┘└───┘└───┘┌─┴─┐└───┘┌─┴─┐┌─┴─┐
        # q0_1: ─────┤ X ├───────────────┤ X ├─────┤ X ├┤ X ├
        #            └───┘               └───┘     └───┘└───┘
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr)
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.x(qr[0])
        circ.y(qr[0])
        circ.z(qr[0])
        circ.cx(qr[0], qr[1])
        circ.h(qr[0])
        circ.cx(qr[0], qr[1])
        circ.cx(qr[0], qr[1])

        after = transpile(circ, coupling_map=[[0, 1], [1, 0]], basis_gates=["u3", "u2", "u1", "cx"])

        expected = QuantumCircuit(QuantumRegister(2, "q"), global_phase=-np.pi / 2)
        msg = f"after:\n{after}\nexpected:\n{expected}"
        self.assertEqual(after, expected, msg=msg)

    def test_pass_manager_empty(self):
        """Test passing an empty PassManager() to the transpiler.

        It should perform no transformations on the circuit.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        resources_before = circuit.count_ops()

        pass_manager = PassManager()
        out_circuit = pass_manager.run(circuit)
        resources_after = out_circuit.count_ops()

        self.assertDictEqual(resources_before, resources_after)

    def test_move_measurements(self):
        """Measurements applied AFTER swap mapping."""
        cmap = CouplingMap.from_line(16)
        qasm_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qasm")
        circ = QuantumCircuit.from_qasm_file(os.path.join(qasm_dir, "move_measurements.qasm"))

        lay = [0, 1, 15, 2, 14, 3, 13, 4, 12, 5, 11, 6]
        out = transpile(circ, initial_layout=lay, coupling_map=cmap, routing_method="sabre")
        out_dag = circuit_to_dag(out)
        meas_nodes = out_dag.named_nodes("measure")
        for meas_node in meas_nodes:
            is_last_measure = all(
                isinstance(after_measure, DAGOutNode)
                for after_measure in out_dag.quantum_successors(meas_node)
            )
            self.assertTrue(is_last_measure)

    @data(0, 1, 2, 3)
    def test_init_resets_kept_preset_passmanagers(self, optimization_level):
        """Test initial resets kept at all preset transpilation levels"""
        num_qubits = 5
        qc = QuantumCircuit(num_qubits)
        qc.reset(range(num_qubits))
        qc.h(range(num_qubits))

        num_resets = transpile(qc, optimization_level=optimization_level).count_ops()["reset"]
        self.assertEqual(num_resets, num_qubits)

    @data(0, 1, 2, 3)
    def test_initialize_reset_is_not_removed(self, optimization_level):
        """The reset in front of initializer should NOT be removed at beginning"""
        qr = QuantumRegister(1, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)], [qr[0]])
        qc.initialize([1.0 / math.sqrt(2), -1.0 / math.sqrt(2)], [qr[0]])

        after = transpile(qc, basis_gates=["reset", "u3"], optimization_level=optimization_level)
        self.assertEqual(after.count_ops()["reset"], 2, msg=f"{after}\n does not have 2 resets.")

    def test_initialize_FakeMelbourne(self):
        """Test that the zero-state resets are remove in a device not supporting them."""
        desired_vector = [1 / math.sqrt(2), 0, 0, 0, 0, 0, 0, 1 / math.sqrt(2)]
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.initialize(desired_vector, [qr[0], qr[1], qr[2]])

        out = transpile(qc, backend=GenericBackendV2(num_qubits=4, seed=42))
        out_dag = circuit_to_dag(out)
        reset_nodes = out_dag.named_nodes("reset")

        self.assertEqual(len(reset_nodes), 3)

    def test_non_standard_basis(self):
        """Test a transpilation with a non-standard basis"""
        qr1 = QuantumRegister(1, "q1")
        qr2 = QuantumRegister(2, "q2")
        qr3 = QuantumRegister(3, "q3")
        qc = QuantumCircuit(qr1, qr2, qr3)
        qc.h(qr1[0])
        qc.h(qr2[1])
        qc.h(qr3[2])
        layout = [4, 5, 6, 8, 9, 10]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        circuit = transpile(
            qc, backend=None, coupling_map=cmap, basis_gates=["h"], initial_layout=layout
        )

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({"h": 3}, resources_after)

    def test_hadamard_to_rot_gates(self):
        """Test a transpilation from H to Rx, Ry gates"""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.h(0)

        expected = QuantumCircuit(qr, global_phase=np.pi / 2)
        expected.append(RYGate(theta=np.pi / 2), [0])
        expected.append(RXGate(theta=np.pi), [0])

        circuit = transpile(qc, basis_gates=["rx", "ry"], optimization_level=0)
        self.assertEqual(circuit, expected)

    def test_basis_subset(self):
        """Test a transpilation with a basis subset of the standard basis"""
        qr = QuantumRegister(1, "q1")
        qc = QuantumCircuit(qr)
        qc.h(qr[0])
        qc.x(qr[0])
        qc.t(qr[0])

        layout = [4]

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        circuit = transpile(
            qc, backend=None, coupling_map=cmap, basis_gates=["u3"], initial_layout=layout
        )

        dag_circuit = circuit_to_dag(circuit)
        resources_after = dag_circuit.count_ops()
        self.assertEqual({"u3": 1}, resources_after)

    def test_check_circuit_width(self):
        """Verify transpilation of circuit with virtual qubits greater than
        physical qubits raises error"""
        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        qc = QuantumCircuit(15, 15)

        with self.assertRaises(CircuitTooWideForTarget):
            transpile(qc, coupling_map=cmap)

    @data(0, 1, 2, 3)
    def test_ccx_routing_method_none(self, optimization_level):
        """CCX without routing method."""

        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)

        out = transpile(
            qc,
            routing_method="none",
            basis_gates=["u", "cx"],
            initial_layout=[0, 1, 2],
            seed_transpiler=0,
            coupling_map=[[0, 1], [1, 2]],
            optimization_level=optimization_level,
        )

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_ccx_routing_method_none_failed(self, optimization_level):
        """CCX without routing method cannot be routed."""

        qc = QuantumCircuit(3)
        qc.ccx(0, 1, 2)

        with self.assertRaises(TranspilerError):
            transpile(
                qc,
                routing_method="none",
                basis_gates=["u", "cx"],
                initial_layout=[0, 1, 2],
                seed_transpiler=0,
                coupling_map=[[0, 1], [1, 2]],
                optimization_level=optimization_level,
            )

    @data(0, 1, 2, 3)
    def test_ms_unrolls_to_cx(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit transpile to a U3,CX target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(
            qc, basis_gates=["u3", "cx"], optimization_level=optimization_level, seed_transpiler=42
        )

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_ms_can_target_ms(self, optimization_level):
        """Verify a Rx,Ry,Rxx circuit can transpile to an Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)

        out = transpile(
            qc,
            basis_gates=["rx", "ry", "rxx"],
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_cx_can_target_ms(self, optimization_level):
        """Verify a U3,CX circuit can transpiler to a Rx,Ry,Rxx target."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.rz(math.pi / 4, [0, 1])

        out = transpile(
            qc,
            basis_gates=["rx", "ry", "rxx"],
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        self.assertTrue(Operator(qc).equiv(out))

    @data(0, 1, 2, 3)
    def test_measure_doesnt_unroll_ms(self, optimization_level):
        """Verify a measure doesn't cause an Rx,Ry,Rxx circuit to unroll to U3,CX."""

        qc = QuantumCircuit(2, 2)
        qc.rx(math.pi / 2, 0)
        qc.ry(math.pi / 4, 1)
        qc.rxx(math.pi / 4, 0, 1)
        qc.measure([0, 1], [0, 1])
        out = transpile(
            qc,
            basis_gates=["rx", "ry", "rxx"],
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        self.assertEqual(qc, out)

    @data(
        ["cx", "u3"],
        ["cz", "u3"],
        ["cz", "rx", "rz"],
        ["rxx", "rx", "ry"],
        ["iswap", "rx", "rz"],
    )
    def test_block_collection_runs_for_non_cx_bases(self, basis_gates):
        """Verify block collection is run when a single two qubit gate is in the basis."""
        twoq_gate, *_ = basis_gates

        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.cx(1, 0)
        qc.cx(0, 1)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3, seed_transpiler=42)

        self.assertLessEqual(out.count_ops()[twoq_gate], 2)

    @unpack
    @data(
        (["u3", "cx"], {"u3": 1, "cx": 1}),
        (["rx", "rz", "iswap"], {"rx": 6, "rz": 12, "iswap": 2}),
        (["rx", "ry", "rxx"], {"rx": 6, "ry": 5, "rxx": 1}),
    )
    def test_block_collection_reduces_1q_gate(self, basis_gates, gate_counts):
        """For synthesis to non-U3 bases, verify we minimize 1q gates."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        out = transpile(qc, basis_gates=basis_gates, optimization_level=3, seed_transpiler=42)

        self.assertTrue(Operator(out).equiv(qc))
        self.assertTrue(set(out.count_ops()).issubset(basis_gates))
        for basis_gate in basis_gates:
            self.assertLessEqual(out.count_ops()[basis_gate], gate_counts[basis_gate])

    @combine(
        optimization_level=[0, 1, 2, 3],
        basis_gates=[
            ["u3", "cx"],
            ["rx", "rz", "iswap"],
            ["ry", "rz", "rxx"],
        ],
    )
    def test_translation_method_synthesis(self, optimization_level, basis_gates):
        """Verify translation_method='synthesis' gets to the basis."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        out = transpile(
            qc,
            translation_method="synthesis",
            basis_gates=basis_gates,
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        self.assertTrue(Operator(out).equiv(qc))
        self.assertTrue(set(out.count_ops()).issubset(basis_gates))

    @data(0, 1, 2, 3)
    def test_circuit_with_delay(self, optimization_level):
        """Verify a circuit with delay can transpile to a scheduled circuit."""

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(500, 1)
        qc.cx(0, 1)

        target = Target(num_qubits=2, dt=1e-9)
        target.add_instruction(
            HGate(), {(i,): InstructionProperties(duration=200 * 1e-9) for i in range(2)}
        )
        target.add_instruction(
            CXGate(),
            {(0, 1): InstructionProperties(duration=700 * 1e-9)},
        )
        target.add_instruction(Delay(Parameter("t")), {(i,): None for i in range(2)})
        out = transpile(
            qc,
            scheduling_method="alap",
            target=target,
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(out.unit, "dt")
            self.assertEqual(out.duration, 1200)

    @data(0, 1, 2, 3)
    def test_circuit_with_delay_expr_duration(self, optimization_level):
        """Verify a circuit with delay with a duration of type types.Duration
        can transpile to a scheduled circuit."""

        # This resolves to 500dt
        delay_expr = expr.add(
            expr.mul(expr.mul(Duration.dt(400), 2.0), expr.div(Duration.dt(200), Duration.dt(400))),
            Duration.dt(100),
        )

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(delay_expr, 1)
        qc.cx(0, 1)

        target = Target(num_qubits=2, dt=1e-9)
        target.add_instruction(
            HGate(), {(i,): InstructionProperties(duration=200 * 1e-9) for i in range(2)}
        )
        target.add_instruction(
            CXGate(),
            {(0, 1): InstructionProperties(duration=700 * 1e-9)},
        )
        target.add_instruction(Delay(Parameter("t")), {(i,): None for i in range(2)})

        out = transpile(
            qc,
            scheduling_method="alap",
            target=target,
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        with self.assertWarns(DeprecationWarning):
            self.assertEqual(out.unit, "dt")
            self.assertEqual(out.duration, 1200)

    def test_delay_converts_to_dt(self):
        """Test that a delay instruction is converted to units of dt given a backend."""
        qc = QuantumCircuit(2)
        qc.delay(1000, [0], unit="us")

        backend = GenericBackendV2(num_qubits=4)
        backend.target.dt = 0.5e-6
        out = transpile([qc, qc], backend, seed_transpiler=42)
        self.assertEqual(out[0].data[0].operation.unit, "dt")
        self.assertEqual(out[1].data[0].operation.unit, "dt")

        out = transpile(qc, dt=1e-9, seed_transpiler=42)
        self.assertEqual(out.data[0].operation.unit, "dt")

    def test_delay_converts_to_seconds(self):
        """Test that a delay instruction is converted to units of seconds when there is no dt."""
        qc = QuantumCircuit(2)
        qc.delay(1000, [0], unit="us")
        qc.x(0)

        # No backend
        out = transpile([qc, qc], seed_transpiler=42)
        self.assertEqual(out[0].data[0].operation.unit, "s")
        self.assertEqual(out[1].data[0].operation.unit, "s")
        self.assertEqual(out[0].data[0].operation.params[0], 1e-3)
        self.assertEqual(out[1].data[0].operation.params[0], 1e-3)

        # Backend without dt
        backend = GenericBackendV2(num_qubits=4)
        backend.target.dt = None
        out = transpile([qc, qc], backend, seed_transpiler=42)
        self.assertEqual(out[0].data[0].operation.unit, "s")
        self.assertEqual(out[1].data[0].operation.unit, "s")
        self.assertEqual(out[0].data[0].operation.params[0], 1e-3)
        self.assertEqual(out[1].data[0].operation.params[0], 1e-3)

    def test_delay_converts_expr_to_dt(self):
        """Test that a delay instruction with a duration expression of type Duration
        is converted to units of dt given a backend."""
        qc = QuantumCircuit(2)
        qc.delay(expr.lift(Duration.us(1000)), [0])

        backend = GenericBackendV2(num_qubits=4)
        backend.target.dt = 0.5e-6
        out = transpile([qc, qc], backend, seed_transpiler=42)
        self.assertEqual(out[0].data[0].operation.unit, "dt")
        self.assertEqual(out[1].data[0].operation.unit, "dt")

        out = transpile(qc, dt=1e-9, seed_transpiler=42)
        self.assertEqual(out.data[0].operation.unit, "dt")

    def test_delay_converts_expr_to_dt_with_rounding(self):
        """Test that converting to 'dt' from wall-time correctly rounds to nearest
        integer."""
        qc = QuantumCircuit(2)
        qc.delay(expr.lift(Duration.ns(1234560)), [0])

        backend = GenericBackendV2(num_qubits=4)
        backend.target.dt = 5e-7

        with self.assertWarnsRegex(UserWarning, "Duration is rounded"):
            out = transpile(qc, backend, seed_transpiler=42)

        self.assertEqual(out.data[0].operation.unit, "dt")
        self.assertEqual(type(out.data[0].operation.duration), int)
        self.assertEqual(out.data[0].operation.duration, round(float(1234560) / 1e9 / 5e-7))

    def test_delay_expr_evaluation_dt(self):
        """Test that a delay instruction with a complex duration expression
        of type Duration is evaluated to 'dt' properly."""
        # 500dt - 200dt = 300dt
        delay_expr = expr.sub(
            # 400dt + 100dt = 500dt
            expr.add(
                # 800dt * 0.5 = 400dt
                expr.mul(
                    # 400dt * 2 = 800dt
                    expr.mul(Duration.s(0.0002), 2.0),
                    # 200dt / 400dt = 0.5
                    expr.div(Duration.ms(0.1), Duration.us(200)),
                ),
                Duration.dt(100),
            ),
            Duration.ns(100_000),
        )

        qc = QuantumCircuit(2)
        qc.delay(delay_expr, 1)

        backend = GenericBackendV2(num_qubits=2)
        backend.target.dt = 5e-7
        out = transpile(
            qc,
            backend=backend,
            seed_transpiler=42,
        )

        self.assertEqual(out.data[0].operation.unit, "dt")
        self.assertTrue(math.isclose(out.data[0].operation.duration, 300, rel_tol=1e-07))

    def test_delay_expr_evaluation_seconds(self):
        """Test that a delay instruction with a complex duration expression
        of type Duration is evaluated to seconds properly when the target 'dt'
        is absent."""
        # .00025s - .0001s = .00015s
        delay_expr = expr.sub(
            # .0002s + .00005s = .00025s
            expr.add(
                # .0004s * 0.5 = .0002s
                expr.mul(
                    # .0002s * 2 = .0004s
                    expr.mul(Duration.s(0.0002), 2.0),
                    # .0001s / .0002s = 0.5
                    expr.div(Duration.ms(0.1), Duration.us(200)),
                ),
                Duration.s(0.00005),
            ),
            Duration.ns(100_000),
        )

        qc = QuantumCircuit(2)
        qc.delay(delay_expr, 1)

        backend = GenericBackendV2(num_qubits=2)
        backend.target.dt = None
        out = transpile(
            qc,
            backend=backend,
            seed_transpiler=42,
        )

        self.assertEqual(out.data[0].operation.unit, "s")
        self.assertTrue(math.isclose(out.data[0].operation.duration, 0.00015, rel_tol=1e-07))

    def test_delay_expr_evaluation_dt_without_target_dt(self):
        """Test that a delay expression with only 'dt' is evaluated properly
        even when the target doesn't specify a 'dt'."""
        delay_expr = expr.sub(
            expr.add(
                expr.mul(
                    expr.mul(Duration.dt(400), 2.0),
                    expr.div(Duration.dt(200), Duration.dt(400)),
                ),
                Duration.dt(100),
            ),
            Duration.dt(200),
        )

        qc = QuantumCircuit(2)
        qc.delay(delay_expr, 1)

        target = Target(num_qubits=2, dt=None)
        target.add_instruction(Delay(Parameter("t")), {(i,): None for i in range(2)})

        out = transpile(
            qc,
            target=target,
            seed_transpiler=42,
        )

        self.assertEqual(out.data[0].operation.unit, "dt")
        self.assertTrue(math.isclose(out.data[0].operation.duration, 300, rel_tol=1e-07))

    def test_rejects_negative_delay_expr(self):
        """Test that a delay instruction with an expression duration is rejected
        when the duration resolves to a negative number."""
        negative_delay = expr.sub(Duration.dt(100), Duration.dt(200))
        qc = QuantumCircuit(2)
        qc.delay(negative_delay, 1)

        with self.assertRaisesRegex(TranspilerError, ".*negative duration"):
            transpile(
                qc,
                backend=GenericBackendV2(num_qubits=2),
                seed_transpiler=42,
            )

    def test_rejects_mixed_units_delay_without_target_dt(self):
        """Test that delay instructions with SI and dt units are rejected without dt."""
        qc = QuantumCircuit(2)
        qc.delay(10, 1, unit="dt")
        qc.delay(10, 1, unit="ns")

        backend = GenericBackendV2(num_qubits=2)
        backend.target.dt = None
        with self.assertRaisesRegex(TranspilerError, ".*SI units and dt unit must not be mixed"):
            transpile(
                qc,
                backend=backend,
                seed_transpiler=42,
            )

    def test_rejects_mixed_units_delay_expr_without_target_dt(self):
        """Test that a delay instruction with wall time and cycles without target DT
        is rejected."""
        mixed_delay = expr.sub(Duration.dt(100), Duration.s(200))
        qc = QuantumCircuit(2)
        qc.delay(mixed_delay, 1)

        backend = GenericBackendV2(num_qubits=2)
        backend.target.dt = None
        with self.assertRaisesRegex(TranspilerError, ".*SI units and dt unit must not be mixed"):
            transpile(
                qc,
                backend=backend,
                seed_transpiler=42,
            )

    @data(0, 1, 2, 3)
    def test_circuit_with_delay_expr_stretch(self, optimization_level):
        """Verify a circuit with delay with a duration of type types.Duration
        can pass through the transpiler without generating an error."""

        qc = QuantumCircuit(2)
        a = qc.add_stretch("a")
        qc.h(0)
        qc.delay(a, 1)
        qc.cx(0, 1)
        with qc.box(duration=a):
            pass

        out = transpile(
            qc,
            backend=GenericBackendV2(
                num_qubits=2, basis_gates=["cx", "h"], control_flow=True, seed=0
            ),
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        self.assertEqual(qc, out)

    @idata(itertools.product([0, 1, 2, 3], ["alap", "asap"]))
    @unpack
    def test_scheduling_with_delay_stretch_fails(self, optimization_level, scheduling_method):
        """Scheduling should fail with an appropriate error message if it is attempted
        on a circuit containing delays with stretch expressions.
        """
        qc = QuantumCircuit(2)
        a = qc.add_stretch("a")
        qc.h(0)
        qc.delay(a, 1)
        qc.cx(0, 1)

        with self.assertRaisesRegex(TranspilerError, "Scheduling cannot run.*stretch"):
            transpile(
                qc,
                backend=GenericBackendV2(num_qubits=2),
                optimization_level=optimization_level,
                scheduling_method=scheduling_method,
                seed_transpiler=42,
            )

    def test_scheduling_backend_v2(self):
        """Test that scheduling method works with Backendv2."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        out = transpile(
            [qc, qc],
            backend=GenericBackendV2(num_qubits=4),
            scheduling_method="alap",
            seed_transpiler=42,
        )
        self.assertIn("delay", out[0].count_ops())
        self.assertIn("delay", out[1].count_ops())

    def test_scheduling_instruction_constraints_backend(self):
        """Test that scheduling-related loose transpile constraints
        work with BackendV2."""

        backend = GenericBackendV2(
            2,
            coupling_map=[[0, 1]],
            basis_gates=["cx", "h"],
            seed=42,
        )
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(0.000001, 1, "s")
        qc.cx(0, 1)

        # update cx to 2 seconds
        backend.target.update_instruction_properties("cx", (0, 1), InstructionProperties(0.000001))

        scheduled = transpile(
            qc,
            backend=backend,
            scheduling_method="alap",
            layout_method="trivial",
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, 9010)

    def test_scheduling_instruction_constraints(self):
        """Test that scheduling-related loose transpile constraints work with target."""
        target = GenericBackendV2(
            2,
            coupling_map=[[0, 1]],
            basis_gates=["cx", "h"],
            seed=42,
        ).target
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.delay(0.000001, 1, "s")
        qc.cx(0, 1)

        # update cx to 2 seconds
        target.update_instruction_properties("cx", (0, 1), InstructionProperties(0.000001))

        scheduled = transpile(
            qc,
            target=target,
            scheduling_method="alap",
            layout_method="trivial",
        )
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, 9010)

    def test_scheduling_dt_constraints(self):
        """Test that scheduling-related loose transpile constraints
        work with BackendV2."""

        original_dt = 2.2222222222222221e-10
        backend_v2 = GenericBackendV2(num_qubits=2, dt=original_dt, seed=3)
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.measure(0, 0)
        scheduled = transpile(qc, backend=backend_v2, scheduling_method="asap")
        with self.assertWarns(DeprecationWarning):
            original_duration = scheduled.duration

        # halve dt in sec = double duration in dt
        scheduled = transpile(qc, backend=backend_v2, scheduling_method="asap", dt=original_dt / 2)
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(scheduled.duration, original_duration * 2)

    @data(1, 2, 3)
    def test_no_infinite_loop(self, optimization_level):
        """Verify circuit cost always descends and optimization does not flip flop indefinitely."""
        qc = QuantumCircuit(1)
        qc.ry(0.2, 0)

        out = transpile(
            qc,
            basis_gates=["id", "p", "sx", "cx"],
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        if optimization_level == 1:
            # Expect a -pi/2 global phase for the U3 to RZ/SX conversion, and
            # a -0.5 * theta phase for RZ to P twice, once at theta, and once at 3 pi
            # for the second and third RZ gates in the U3 decomposition.
            expected = QuantumCircuit(
                1, global_phase=-np.pi / 2 - 0.5 * (-0.2 + np.pi) - 0.5 * 3 * np.pi
            )
            expected.p(-np.pi, 0)
            expected.sx(0)
            expected.p(np.pi - 0.2, 0)
            expected.sx(0)
        else:
            expected = QuantumCircuit(1, global_phase=(15 * np.pi - 1) / 10)
            expected.sx(0)
            expected.p(1.0 / 5.0 + np.pi, 0)
            expected.sx(0)
            expected.p(3 * np.pi, 0)

        error_message = (
            f"\nOutput circuit:\n{out!s}\n{Operator(out).data}\n"
            f"Expected circuit:\n{expected!s}\n{Operator(expected).data}"
        )
        self.assertEqual(Operator(qc), Operator(out))
        self.assertEqual(out, expected, error_message)

    @data(0, 1, 2, 3)
    def test_transpile_preserves_circuit_metadata(self, optimization_level):
        """Verify that transpile preserves circuit metadata in the output."""
        metadata = {"experiment_id": "1234", "execution_number": 4}
        name = "my circuit"
        circuit = QuantumCircuit(2, metadata=metadata.copy(), name=name)
        circuit.h(0)
        circuit.cx(0, 1)

        cmap = [
            [1, 0],
            [1, 2],
            [2, 3],
            [4, 3],
            [4, 10],
            [5, 4],
            [5, 6],
            [5, 9],
            [6, 8],
            [7, 8],
            [9, 8],
            [9, 10],
            [11, 3],
            [11, 10],
            [11, 12],
            [12, 2],
            [13, 1],
            [13, 12],
        ]

        res = transpile(
            circuit,
            basis_gates=["id", "p", "sx", "cx"],
            coupling_map=cmap,
            optimization_level=optimization_level,
            seed_transpiler=42,
        )
        self.assertEqual(res.metadata, metadata)
        self.assertEqual(res.name, name)

        target = Target(14)
        for inst in (IGate(), PhaseGate(Parameter("t")), SXGate()):
            target.add_instruction(inst, {(i,): None for i in range(14)})
        target.add_instruction(CXGate(), {tuple(pair): None for pair in cmap})

        res = transpile(
            circuit,
            target=target,
            optimization_level=optimization_level,
            seed_transpiler=42,
        )
        self.assertEqual(res.metadata, metadata)
        self.assertEqual(res.name, name)

    @data(0, 1, 2, 3)
    def test_transpile_optional_registers(self, optimization_level):
        """Verify transpile accepts circuits without registers end-to-end."""

        qubits = [Qubit() for _ in range(3)]
        clbits = [Clbit() for _ in range(3)]

        qc = QuantumCircuit(qubits, clbits)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)

        qc.measure(qubits, clbits)
        backend = GenericBackendV2(num_qubits=4)

        out = transpile(
            qc, backend=backend, optimization_level=optimization_level, seed_transpiler=42
        )

        self.assertEqual(len(out.qubits), backend.num_qubits)
        self.assertEqual(len(out.clbits), len(clbits))

    @data(0, 1, 2, 3)
    def test_translate_ecr_basis(self, optimization_level):
        """Verify that rewriting in ECR basis is efficient."""
        circuit = QuantumCircuit(2)
        circuit.append(random_unitary(4, seed=1), [0, 1])
        circuit.barrier()
        circuit.cx(0, 1)
        circuit.barrier()
        circuit.swap(0, 1)
        circuit.barrier()
        circuit.iswap(0, 1)

        res = transpile(
            circuit,
            basis_gates=["u", "ecr"],
            optimization_level=optimization_level,
            seed_transpiler=42,
        )

        # Swap gates get optimized away in opt. level 2, 3
        expected_num_ecr_gates = 6 if optimization_level in (2, 3) else 9
        self.assertEqual(res.count_ops()["ecr"], expected_num_ecr_gates)
        self.assertEqual(Operator(circuit), Operator.from_circuit(res))

    def test_optimize_ecr_basis(self):
        """Test highest optimization level can optimize over ECR."""
        circuit = QuantumCircuit(2)
        circuit.swap(1, 0)
        circuit.iswap(0, 1)

        res = transpile(circuit, basis_gates=["u", "ecr"], optimization_level=3, seed_transpiler=42)

        # an iswap gate is equivalent to (swap, CZ) up to single-qubit rotations. Normally, the swap gate
        # in the circuit would cancel with the swap gate of the (swap, CZ), leaving a single CZ gate that
        # can be realized via one ECR gate. However, with the introduction of ElideSwap, the swap gate
        # cancellation can not occur anymore, thus requiring two ECR gates for the iswap gate.
        self.assertEqual(res.count_ops()["ecr"], 2)
        self.assertEqual(Operator(circuit), Operator.from_circuit(res))

    def test_approximation_degree_invalid(self):
        """Test invalid approximation degree raises."""
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        with self.assertRaises(QiskitError):
            transpile(
                circuit, basis_gates=["u", "cz"], approximation_degree=1.1, seed_transpiler=42
            )

    def test_approximation_degree(self):
        """Test more approximation can give lower-cost circuit."""
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        circuit.h(0)
        circ_10 = transpile(
            circuit,
            basis_gates=["u", "cx"],
            translation_method="synthesis",
            approximation_degree=0.1,
            seed_transpiler=42,
            optimization_level=1,
        )
        circ_90 = transpile(
            circuit,
            basis_gates=["u", "cx"],
            translation_method="synthesis",
            approximation_degree=0.9,
            seed_transpiler=42,
            optimization_level=1,
        )
        self.assertLess(circ_10.depth(), circ_90.depth())

    @data(0, 1, 2, 3)
    def test_synthesis_translation_method_with_single_qubit_gates(self, optimization_level):
        """Test that synthesis basis translation works for solely 1q circuit"""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        res = transpile(
            qc,
            basis_gates=["id", "rz", "x", "sx", "cx"],
            translation_method="synthesis",
            optimization_level=optimization_level,
            seed_transpiler=42,
        )
        expected = QuantumCircuit(3, global_phase=3 * np.pi / 4)
        expected.rz(np.pi / 2, 0)
        expected.rz(np.pi / 2, 1)
        expected.rz(np.pi / 2, 2)
        expected.sx(0)
        expected.sx(1)
        expected.sx(2)
        expected.rz(np.pi / 2, 0)
        expected.rz(np.pi / 2, 1)
        expected.rz(np.pi / 2, 2)
        self.assertEqual(res, expected)

    @data(0, 1, 2, 3)
    def test_synthesis_translation_method_with_gates_outside_basis(self, optimization_level):
        """Test that synthesis translation works for circuits with single gates outside basis"""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        res = transpile(
            qc,
            basis_gates=["id", "rz", "x", "sx", "cx"],
            translation_method="synthesis",
            optimization_level=optimization_level,
            seed_transpiler=42,
        )
        if optimization_level not in {2, 3}:
            self.assertTrue(Operator(qc).equiv(res))
            self.assertNotIn("swap", res.count_ops())
        else:
            # Optimization level 2 and 3 eliminates the swap by permuting the
            # qubits
            self.assertEqual(res, QuantumCircuit(2))

    @data(0, 1, 2, 3)
    def test_target_ideal_gates(self, opt_level):
        """Test that transpile() with a custom ideal sim target works."""
        theta = Parameter("θ")
        phi = Parameter("ϕ")
        lam = Parameter("λ")
        target = Target(num_qubits=2)
        target.add_instruction(UGate(theta, phi, lam), {(0,): None, (1,): None})
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(Measure(), {(0,): None, (1,): None})
        qubit_reg = QuantumRegister(2, name="q")
        clbit_reg = ClassicalRegister(2, name="c")
        qc = QuantumCircuit(qubit_reg, clbit_reg, name="bell")
        qc.h(qubit_reg[0])
        qc.cx(qubit_reg[0], qubit_reg[1])

        result = transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)

        self.assertEqual(Operator.from_circuit(result), Operator.from_circuit(qc))

    @data(0, 1, 2, 3)
    def test_target_as_positional_backend_arg(self, opt_level):
        """Test that a Target can be passed as the second positional argument to transpile(),
        consistent with the behaviour of generate_preset_pass_manager()."""
        theta = Parameter("θ")
        phi = Parameter("ϕ")
        lam = Parameter("λ")
        target = Target(num_qubits=2)
        target.add_instruction(UGate(theta, phi, lam), {(0,): None, (1,): None})
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(Measure(), {(0,): None, (1,): None})
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        result = transpile(qc, target, optimization_level=opt_level, seed_transpiler=42)

        self.assertEqual(Operator.from_circuit(result), Operator.from_circuit(qc))

    def test_target_as_positional_backend_arg_conflicts_with_target_kwarg(self):
        """Test that passing a Target positionally as backend and also as target= raises TypeError."""
        theta = Parameter("θ")
        phi = Parameter("ϕ")
        lam = Parameter("λ")
        target = Target(num_qubits=2)
        target.add_instruction(UGate(theta, phi, lam), {(0,): None, (1,): None})
        target.add_instruction(CXGate(), {(0, 1): None})
        target.add_instruction(Measure(), {(0,): None, (1,): None})
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        with self.assertRaises(TypeError):
            transpile(qc, target, target=target)

    @data(0, 1, 2, 3)
    def test_transpile_control_flow_no_backend(self, opt_level):
        """Test `transpile` with control flow and no specified hardware constraints."""
        qc = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "c"))
        qc.h(0)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], False)):
            qc.x(0)
        with qc.while_loop((qc.clbits[0], True)):
            qc.x(0)
        with qc.for_loop(range(2)):
            qc.x(0)
        with qc.switch(qc.cregs[0]) as case:
            with case(case.DEFAULT):
                qc.x(0)
        qc.measure(0, 0)

        transpiled = transpile(qc, optimization_level=opt_level)
        # There's nothing that can be optimized here.
        self.assertEqual(qc, transpiled)

    @data(0, 1, 2, 3)
    def test_transpile_with_custom_control_flow_target(self, opt_level):
        """Test transpile() with a target and control flow ops."""
        target = GenericBackendV2(
            num_qubits=8, coupling_map=CouplingMap.from_line(8), control_flow=True
        ).target

        circuit = QuantumCircuit(6, 1)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(0, 1)
        circuit.cz(0, 2)
        circuit.append(CustomCX(), [1, 2], [])
        with circuit.for_loop((1,)):
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with circuit.if_test((circuit.clbits[0], True)) as else_:
            circuit.cx(0, 1)
            circuit.cz(0, 2)
            circuit.append(CustomCX(), [1, 2], [])
        with else_:
            circuit.cx(3, 4)
            circuit.cz(3, 5)
            circuit.append(CustomCX(), [4, 5], [])
            with circuit.while_loop((circuit.clbits[0], True)):
                circuit.cx(3, 4)
                circuit.cz(3, 5)
                circuit.append(CustomCX(), [4, 5], [])
        with circuit.switch(circuit.cregs[0]) as case_:
            with case_(0):
                circuit.cx(0, 1)
                circuit.cz(0, 2)
                circuit.append(CustomCX(), [1, 2], [])
            with case_(1):
                circuit.cx(1, 2)
                circuit.cz(1, 3)
                circuit.append(CustomCX(), [2, 3], [])
        transpiled = transpile(
            circuit, optimization_level=opt_level, target=target, seed_transpiler=12434
        )
        # Tests of the complete validity of a circuit are mostly done at the individual pass level;
        # here we're just checking that various passes do appear to have run.
        self.assertIsInstance(transpiled, QuantumCircuit)
        # Assert layout ran.
        self.assertIsNot(getattr(transpiled, "_layout", None), None)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)
                # Assert unrolling ran.
                self.assertNotIsInstance(instruction.operation, CustomCX)
                # Assert translation ran.
                self.assertNotIsInstance(instruction.operation, CZGate)

        # Assert routing ran.
        _visit_block(
            transpiled,
            qubit_mapping={qubit: index for index, qubit in enumerate(transpiled.qubits)},
        )

    @data(1, 2, 3)
    def test_transpile_identity_circuit_no_target(self, opt_level):
        """Test circuit equivalent to identity is optimized away for all optimization levels >0.

        Reproduce taken from https://github.com/Qiskit/qiskit-terra/issues/9217
        """
        qr1 = QuantumRegister(3, "state")
        qr2 = QuantumRegister(2, "ancilla")
        cr = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.h(qr1[0])
        qc.cx(qr1[0], qr1[1])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[1], qr1[2])
        qc.cx(qr1[0], qr1[1])
        qc.h(qr1[0])

        empty_qc = QuantumCircuit(qr1, qr2, cr)
        result = transpile(qc, optimization_level=opt_level, seed_transpiler=42)
        self.assertEqual(empty_qc, result)

    @data(0, 1, 2, 3)
    def test_initial_layout_with_loose_qubits(self, opt_level):
        """Regression test of gh-10125."""
        qc = QuantumCircuit([Qubit(), Qubit()])
        qc.cx(0, 1)
        transpiled = transpile(
            qc, initial_layout=[1, 0], optimization_level=opt_level, seed_transpiler=42
        )
        self.assertIsNotNone(transpiled.layout)
        self.assertEqual(
            transpiled.layout.initial_layout, Layout({0: qc.qubits[1], 1: qc.qubits[0]})
        )

    @data(0, 1, 2, 3)
    def test_initial_layout_with_overlapping_qubits(self, opt_level):
        """Regression test of gh-10125."""
        qr1 = QuantumRegister(2, "qr1")
        qr2 = QuantumRegister(bits=qr1[:])
        qc = QuantumCircuit(qr1, qr2)
        qc.cx(0, 1)
        transpiled = transpile(
            qc, initial_layout=[1, 0], optimization_level=opt_level, seed_transpiler=42
        )
        self.assertIsNotNone(transpiled.layout)
        self.assertEqual(
            transpiled.layout.initial_layout, Layout({0: qc.qubits[1], 1: qc.qubits[0]})
        )

    @combine(opt_level=[0, 1, 2, 3], basis=[["rz", "x"], ["rx", "z"], ["rz", "y"], ["ry", "x"]])
    def test_paulis_to_constrained_1q_basis(self, opt_level, basis):
        """Test that Pauli-gate circuits can be transpiled to constrained 1q bases that do not
        contain any root-Pauli gates."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.barrier()
        qc.y(0)
        qc.barrier()
        qc.z(0)
        transpiled = transpile(
            qc, basis_gates=basis, optimization_level=opt_level, seed_transpiler=42
        )
        self.assertGreaterEqual(set(basis) | {"barrier"}, transpiled.count_ops().keys())
        self.assertEqual(Operator(qc), Operator(transpiled))

    @data(0, 1, 2, 3)
    def test_barrier_not_output(self, opt_level):
        """Test that barriers added as part internal transpiler operations do not leak out."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))
        tqc = transpile(
            qc,
            initial_layout=[1, 4],
            coupling_map=[[1, 2], [2, 3], [3, 4]],
            optimization_level=opt_level,
        )
        self.assertNotIn("barrier", tqc.count_ops())

    @data(0, 1, 2, 3)
    def test_barrier_not_output_input_preserved(self, opt_level):
        """Test that barriers added as part internal transpiler operations do not leak out."""
        qc = QuantumCircuit(2, 2)
        qc.cx(0, 1)
        qc.measure_all()
        tqc = transpile(
            qc,
            initial_layout=[1, 4],
            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
            optimization_level=opt_level,
        )
        op_counts = tqc.count_ops()
        self.assertEqual(op_counts["barrier"], 1)
        for inst in tqc.data:
            if inst.operation.name == "barrier":
                self.assertEqual(len(inst.qubits), 2)

    @combine(opt_level=[0, 1, 2, 3])
    def test_transpile_annotated_ops(self, opt_level):
        """Test transpilation of circuits with annotated operations."""
        qc = QuantumCircuit(3)
        qc.append(AnnotatedOperation(SGate(), InverseModifier()), [0])
        qc.append(AnnotatedOperation(XGate(), ControlModifier(1)), [1, 2])
        qc.append(AnnotatedOperation(HGate(), PowerModifier(3)), [2])
        expected = QuantumCircuit(3)
        expected.sdg(0)
        expected.cx(1, 2)
        expected.h(2)
        transpiled = transpile(qc, optimization_level=opt_level, seed_transpiler=42)
        self.assertNotIn("annotated", transpiled.count_ops().keys())
        self.assertEqual(Operator(qc), Operator(transpiled))
        self.assertEqual(Operator(qc), Operator(expected))

    @combine(opt_level=[0, 1, 2, 3])
    def test_transpile_annotated_ops_with_backend(self, opt_level):
        """Test transpilation of circuits with annotated operations given a backend."""
        qc = QuantumCircuit(3)
        qc.append(AnnotatedOperation(SGate(), InverseModifier()), [0])
        qc.append(AnnotatedOperation(XGate(), ControlModifier(1)), [1, 2])
        qc.append(AnnotatedOperation(HGate(), PowerModifier(3)), [2])

        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
        )
        transpiled = transpile(
            qc, optimization_level=opt_level, backend=backend, seed_transpiler=42
        )
        self.assertLessEqual(set(transpiled.count_ops().keys()), {"u1", "u2", "u3", "cx"})

    @data(1, 2, 3)
    def test_optimize_decomposition_around_control_flow(self, level):
        """Test that we successfully optimise away idle wires from control flow."""
        qc = QuantumCircuit(5, 1)
        # This cz(0, 1) can't cancel with its friend on the other side until the data dependency is
        # removed from the `if` block.  Similarly, the sx(2) needs the two x(2) in the `if` to go.
        qc.cz(0, 1)
        qc.sx(2)
        qc.cz(3, 4)
        with qc.if_test((qc.clbits[0], False)):
            # The `(0, 4)` data dependencies should be removed before routing, so we don't see any
            # swaps in here.
            qc.cz(0, 4)
            qc.x(2)
            qc.x(2)
            qc.x(3)
            qc.cz(0, 4)
        qc.cz(0, 1)
        qc.sxdg(2)
        qc.cz(3, 4)

        expected = qc.copy_empty_like()
        expected.cz(3, 4)
        with expected.if_test((expected.clbits[0], False)):
            expected.x(3)
        expected.cz(3, 4)

        target = Target(5)
        target.add_instruction(XGate(), {(i,): None for i in range(5)})
        target.add_instruction(SXGate(), {(i,): None for i in range(5)})
        target.add_instruction(RZGate(Parameter("a")), {(i,): None for i in range(5)})
        target.add_instruction(CZGate(), dict.fromkeys(CouplingMap.from_line(5)))
        target.add_instruction(IfElseOp, name="if_else")

        self.assertEqual(
            transpile(qc, target=target, optimization_level=level, initial_layout=[0, 1, 2, 3, 4]),
            expected,
        )

    @data(0, 1, 2, 3)
    def test_no_cancelling_around_box(self, level):
        """Test that operations aren't cancelled through the walls of a 'box'."""
        # In linear operation, we do cz(0,1) - cz(0,1) - cx(1,2) - cx(1,2), so without the `box`,
        # the circuit would optimise to the identity.  We want to be sure that the box itself is
        # treated as atomic, though.
        qc = QuantumCircuit(3)
        qc.cz(0, 1)
        with qc.box():
            qc.cz(0, 1)
            qc.cx(1, 2)
        qc.cx(1, 2)

        target = Target(3)
        target.add_instruction(SXGate(), {(i,): None for i in range(3)})
        target.add_instruction(RZGate(Parameter("a")), {(i,): None for i in range(3)})
        target.add_instruction(CZGate(), dict.fromkeys(CouplingMap.from_line(3)))
        target.add_instruction(CXGate(), dict.fromkeys(CouplingMap.from_line(3)))
        target.add_instruction(BoxOp, name="box")

        out = transpile(qc, target=target, optimization_level=level, initial_layout=[0, 1, 2])
        self.assertEqual(out, qc)

    @data(0, 1, 2, 3)
    def test_no_contraction_of_wires_in_box(self, level):
        """Test that no-ops in boxes are not contracted."""
        qc = QuantumCircuit(3)
        with qc.box():
            qc.cz(0, 1)
            # This qubit should stay used; optimisation must not remove it from the `box`.
            qc.noop(2)

        target = Target(3)
        target.add_instruction(SXGate(), {(i,): None for i in range(3)})
        target.add_instruction(RZGate(Parameter("a")), {(i,): None for i in range(3)})
        target.add_instruction(CZGate(), dict.fromkeys(CouplingMap.from_line(3)))
        target.add_instruction(BoxOp, name="box")

        out = transpile(qc, target=target, optimization_level=level, initial_layout=[0, 1, 2])
        self.assertEqual(out, qc)

    @data(0, 1, 2, 3)
    def test_no_contraction_of_wires_in_routed_box(self, level):
        """Test that no-ops in boxes are not contracted, even if routing happens."""
        num_qubits = 10
        qc = QuantumCircuit(num_qubits)
        with qc.box():
            # This is long range, and we force routing to engage by requiring the trivial layout.
            qc.cz(0, 4)
            # This qubit should stay used, even though routing will be engaged to sort out the
            # long-range `cz`.
            qc.noop(8)

        target = Target(num_qubits)
        target.add_instruction(SXGate(), {(i,): None for i in range(num_qubits)})
        target.add_instruction(RZGate(Parameter("a")), {(i,): None for i in range(num_qubits)})
        target.add_instruction(CZGate(), dict.fromkeys(CouplingMap.from_line(num_qubits)))
        target.add_instruction(BoxOp, name="box")

        out = transpile(
            qc, target=target, optimization_level=level, initial_layout=list(range(num_qubits))
        )
        self.assertIsInstance(out.data[0].operation, BoxOp)
        body = out.data[0].operation.blocks[0]
        qubit_map = dict(zip(body.qubits, (out.find_bit(bit).index for bit in out.data[0].qubits)))

        # It must use the initial indices (because of the trivial layout), and more for routing.
        self.assertGreater(set(qubit_map.values()), {0, 4, 8})

        # Index 8 must be idle still; there's no reason for routing to have engaged it.  If this
        # fails, the test isn't valid---we want to test that the no-op marker _alone_ is sufficient
        # to prevent contraction of the wires.
        active_indices = {qubit_map[bit] for instruction in body for bit in instruction.qubits}
        self.assertNotIn(8, active_indices)

    def test_custom_dt_preserves_properties(self):
        """Test that setting the `dt` parameter with a `backend` doesn't affect the target properties
        and vf2 runs as expected.
        """

        coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "sx", "x", "cx", "rz"],
            coupling_map=coupling_map,
            seed=123,
        )
        qubits = 3
        qc = QuantumCircuit(qubits)
        for i in range(5):
            qc.cx(i % qubits, int(i + qubits / 2) % qubits)

        # The point of this test is to test that error rates are still checked when there's a custom
        # `dt` set.  For that test to work, we need to check that compiling explicitly without
        # errors produces a different layout to transpiling with them, so we can then later check
        # that the "custom dt" transpile matches the "with errors" case.
        seed_transpiler = 2025_07_07
        tqc_no_error = transpile(qc, coupling_map=coupling_map, seed_transpiler=seed_transpiler)
        # transpile with gate errors
        tqc_no_dt = transpile(qc, backend=backend, seed_transpiler=seed_transpiler)
        # confirm that the output layouts are different. This can fail spurious if the built-in
        # layout pass happened to choose the same initial layout that the noise-aware remapping
        # converges to.  In that case, you can bump the transpiler seed.
        self.assertNotEqual(
            tqc_no_dt.layout.final_index_layout(), tqc_no_error.layout.final_index_layout()
        )
        # now modify dt with gate errors
        tqc_dt = transpile(qc, backend=backend, seed_transpiler=seed_transpiler, dt=backend.dt * 2)
        # confirm that dt doesn't affect layout.
        self.assertEqual(tqc_no_dt.layout.final_index_layout(), tqc_dt.layout.final_index_layout())

    @combine(optimization_level=[0, 1, 2, 3], control_flow=[False, True])
    def test_stretch_integration_with_alignment(self, optimization_level, control_flow):
        """Test that `stretch`es can pass all the way through default transpilation, even when the
        backend has alignment constraints.  We treat the presence of a `stretch` as meaning
        "something else will schedule this", so we don't need to reschedule in this case."""
        backend = AlignmentBackend(4, control_flow=control_flow)
        qc = QuantumCircuit(3, 3)
        a = qc.add_stretch("a")
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(1, 2)
        qc.delay(a, 0)
        qc.delay(expr.mul(2, a), 1)
        qc.measure([0, 1, 2], [0, 1, 2])
        if control_flow:
            with qc.if_test((qc.clbits[0], False)):
                qc.delay(a, 2)
        _ = transpile(qc, backend, optimization_level=optimization_level)
        # No meaningful assertions; this is a simple regression test for "stretches don't explode
        # backends with alignments" more than anything.

    @data(0, 1, 2, 3)
    def test_single_qubit_circuit_deterministic_output(self, optimization_level):
        """Test that the transpiler's output is deterministic in a single qubit example.

        Reproduce from `#14729 <https://github.com/Qiskit/qiskit/issues/14729>`__"""
        params = ParameterVector("θ", length=3)

        circ = QuantumCircuit(3)
        for i, par in enumerate(params):
            circ.rx(par, i)
        circ.measure_all()
        backend = GenericBackendV2(10, noise_info=True, seed=123)
        isa_circs = []
        for _ in range(10):
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level, target=backend.target, seed_transpiler=123
            )
            isa_circs.append(pm.run(circ))
        for i in range(10):
            self.assertEqual(isa_circs[0], isa_circs[i])

    @data(0, 1, 2, 3)
    def test_reuse_on_differing_circuits(self, optimization_level):
        """Test that re-using the same `PassManager` instance on two different circuits is the same
        as using new instances of the `PassManager`."""

        def make_pm():
            num_qubits = 12
            target = Target()
            target.add_instruction(
                SXGate(),
                {(i,): InstructionProperties(error=1e-5 * (i + 1)) for i in range(num_qubits)},
            )
            target.add_instruction(
                RZGate(Parameter("a")), {(i,): InstructionProperties() for i in range(num_qubits)}
            )
            target.add_instruction(
                Measure(),
                {
                    (i,): InstructionProperties(error=1e-3 * (num_qubits - i))
                    for i in range(num_qubits)
                },
            )
            target.add_instruction(
                CZGate(),
                {
                    pair: InstructionProperties(error=1e-3 * sum(pair))
                    for pair in CouplingMap.from_line(num_qubits)
                },
            )
            return generate_preset_pass_manager(
                optimization_level=optimization_level, target=target, seed_transpiler=2025_10_27
            )

        def ghz(num_qubits):
            qc = QuantumCircuit(num_qubits, num_qubits)
            qc.h(0)
            for i in range(1, num_qubits):
                qc.cx(0, i)
            qc.measure(qc.qubits, qc.clbits)
            return qc

        circuits = [ghz(5), ghz(8), ghz(10)]
        shared_pm = make_pm()
        shared_circuits = [shared_pm.run(circuit) for circuit in circuits]
        own_circuits = [make_pm().run(circuit) for circuit in circuits]
        self.assertEqual(shared_circuits, own_circuits)
        together_circuits = make_pm().run(circuits)
        self.assertEqual(together_circuits, own_circuits)

    def test_parse_seed_transpiler_from_env_var(self):
        """Test that the environment variable QISKIT_TRANSPILER_SEED is passed to the transpiler."""
        qc = QuantumCircuit(3)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)

        # Save the original SabreLayout.run method, and create a mock method that calls the original method,
        # but also records the value for seed.
        original_run = SabreLayout.run
        run_calls = []

        def mock_run(self, dag):
            run_calls.append(self.seed)
            original_run(self, dag)

        with patch.dict(os.environ, {"QISKIT_TRANSPILER_SEED": "1234"}):
            with patch.object(SabreLayout, "run", new=mock_run):
                _ = transpile(qc, optimization_level=1, coupling_map=CouplingMap.from_line(3))
        self.assertEqual(len(run_calls), 1)
        self.assertEqual(run_calls[0], 1234)


@ddt
class TestPostTranspileIntegration(QiskitTestCase):
    """Test that the output of `transpile` is usable in various other integration contexts."""

    def _regular_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.measure(0, 0)
        base.cx(0, 1)
        base.cz(0, 2)
        base.cz(0, 3)
        base.cz(1, 4)
        base.cx(1, 5)
        base.measure(1, 1)
        base.append(CustomCX(), [3, 6])
        base.append(CustomCX(), [5, 4])
        base.append(CustomCX(), [5, 3])
        with base.if_test((base.cregs[0], 3)):
            base.append(CustomCX(), [2, 4])
        base.ry(a, 4)
        base.measure(4, 2)
        return base

    def _control_flow_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.measure(0, 0)
        with base.if_test((base.cregs[0], 1)) as else_:
            base.cx(0, 1)
            base.cz(0, 2)
            base.cz(0, 3)
        with else_:
            base.cz(1, 4)
            with base.for_loop((1, 2)):
                base.cx(1, 5)
        base.measure(2, 2)
        with base.while_loop((2, False)):
            base.append(CustomCX(), [3, 6])
            base.append(CustomCX(), [5, 4])
            base.append(CustomCX(), [5, 3])
            base.append(CustomCX(), [2, 4])
            base.ry(a, 4)
            base.measure(4, 2)
        with base.switch(base.cregs[0]) as case_:
            with case_(0, 1):
                base.cz(3, 5)
            with case_(case_.DEFAULT):
                base.cz(1, 4)
                base.append(CustomCX(), [2, 4])
                base.append(CustomCX(), [3, 4])
        return base

    def _control_flow_expr_circuit(self):
        a = Parameter("a")
        regs = [
            QuantumRegister(2, name="q0"),
            QuantumRegister(3, name="q1"),
            ClassicalRegister(2, name="c0"),
        ]
        bits = [Qubit(), Qubit(), Clbit()]
        base = QuantumCircuit(*regs, bits)
        base.h(0)
        base.delay(expr.lift(Duration.ps(1234)), 0)
        base.measure(0, 0)
        with base.if_test(expr.equal(base.cregs[0], 1)) as else_:
            base.cx(0, 1)
            base.cz(0, 2)
            base.cz(0, 3)
        with else_:
            base.cz(1, 4)
            with base.for_loop((1, 2)):
                base.cx(1, 5)
        base.measure(2, 2)
        with base.while_loop(expr.logic_not(bits[2])):
            base.append(CustomCX(), [3, 6])
            base.append(CustomCX(), [5, 4])
            base.append(CustomCX(), [5, 3])
            base.append(CustomCX(), [2, 4])
            base.ry(a, 4)
            base.measure(4, 2)
        with base.switch(expr.bit_and(base.cregs[0], 2)) as case_:
            with case_(0, 1):
                base.cz(3, 5)
            with case_(case_.DEFAULT):
                base.cz(1, 4)
                base.append(CustomCX(), [2, 4])
                base.append(CustomCX(), [3, 4])
        with base.if_test(expr.less(1.0, 2.0)):
            base.cx(0, 1)
        with base.if_test(
            expr.logic_and(
                expr.logic_and(
                    expr.equal(Duration.dt(1), Duration.ns(2)),
                    expr.equal(Duration.us(3), Duration.ms(4)),
                ),
                expr.equal(Duration.s(5), Duration.dt(6)),
            )
        ):
            base.cx(0, 1)
        with base.if_test(
            expr.logic_and(
                expr.logic_and(
                    expr.equal(expr.mul(Duration.dt(1), 2.0), expr.div(Duration.ns(2), 2.0)),
                    expr.equal(
                        expr.add(Duration.us(3), Duration.us(4)),
                        expr.sub(Duration.ms(5), Duration.ms(6)),
                    ),
                ),
                expr.logic_and(
                    expr.equal(expr.mul(expr.lift(1.0), 2.0), expr.div(4.0, 2.0)),
                    expr.equal(expr.add(3.0, 4.0), expr.sub(10.5, expr.lift(4.3, types.Float()))),
                ),
            )
        ):
            base.cx(0, 1)
        return base

    def _standalone_var_circuit(self):
        a = expr.Var.new("a", types.Bool())
        b = expr.Var.new("b", types.Uint(8))
        c = expr.Var.new("c", types.Uint(8))
        d = expr.Stretch.new("d")

        qc = QuantumCircuit(5, 5, inputs=[a])
        qc.add_var(b, 12)
        qc.add_stretch(d)
        qc.h(0)
        qc.delay(expr.add(Duration.dt(1000), d), 0)
        qc.cx(0, 1)
        qc.measure([0, 1], [0, 1])
        qc.store(a, expr.bit_xor(qc.clbits[0], qc.clbits[1]))
        with qc.if_test(a) as else_:
            qc.cx(2, 3)
            qc.cx(3, 4)
            qc.cx(4, 2)
        with else_:
            qc.add_var(c, 12)
        with qc.while_loop(a):
            with qc.while_loop(a):
                qc.add_var(c, 12)
                qc.cz(1, 0)
                qc.cz(4, 1)
                qc.store(a, False)
        with qc.switch(expr.bit_and(b, 7)) as case:
            with case(0):
                qc.cz(0, 1)
                qc.cx(1, 2)
                qc.cy(2, 0)
            with case(case.DEFAULT):
                qc.store(b, expr.bit_and(b, 7))
        return qc

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip(self, optimization_level):
        """Test that the output of a transpiled circuit can be round-tripped through QPY."""
        transpiled = transpile(
            self._regular_circuit(),
            backend=GenericBackendV2(
                num_qubits=8,
                coupling_map=CouplingMap.from_line(8),
                control_flow=True,
                seed=2025_05_28,
            ),
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow can be round-tripped
        through QPY."""
        if optimization_level == 3 and sys.platform == "win32":
            self.skipTest(
                "This test case triggers a bug in the eigensolver routine on windows. "
                "See #10345 for more details."
            )

        backend = GenericBackendV2(
            num_qubits=8, coupling_map=CouplingMap.from_line(8), control_flow=True, seed=2025_05_28
        )
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=backend,
            basis_gates=backend.operation_names,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # Round-tripping the layout is out-of-scope for QPY while it's a private attribute.
        transpiled._layout = None
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_control_flow_expr(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow including `Expr` nodes can
        be round-tripped through QPY."""
        backend = GenericBackendV2(
            num_qubits=27,
            coupling_map=CouplingMap.from_line(27),
            seed=2025_05_28,
            control_flow=True,
        )
        transpiled = transpile(
            self._control_flow_expr_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2023_07_26,
        )
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qpy_roundtrip_standalone_var(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow including standalone `Var`
        nodes can be round-tripped through QPY."""
        backend = GenericBackendV2(
            num_qubits=11,
            coupling_map=CouplingMap.from_line(11),
            control_flow=True,
            seed=2025_05_28,
        )
        transpiled = transpile(
            self._standalone_var_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2024_05_01,
        )
        buffer = io.BytesIO()
        qpy.dump(transpiled, buffer)
        buffer.seek(0)
        round_tripped = qpy.load(buffer)[0]
        self.assertEqual(round_tripped, transpiled)

    @data(0, 1, 2, 3)
    def test_qasm3_output(self, optimization_level):
        """Test that the output of a transpiled circuit can be dumped into OpenQASM 3."""
        backend = GenericBackendV2(
            num_qubits=20,
            coupling_map=TOKYO_CMAP,
            basis_gates=["id", "u1", "u2", "u3", "cx"],
            control_flow=True,
        )

        transpiled = transpile(
            self._regular_circuit(),
            backend=backend,
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(qasm3.dumps(transpiled).strip(), str)

    @data(0, 1, 2, 3)
    def test_qasm3_output_control_flow(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow can be dumped into
        OpenQASM 3."""
        transpiled = transpile(
            self._control_flow_circuit(),
            backend=GenericBackendV2(
                num_qubits=8,
                coupling_map=CouplingMap.from_line(8),
                control_flow=True,
                seed=2025_05_28,
            ),
            optimization_level=optimization_level,
            seed_transpiler=2022_10_17,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(
            qasm3.dumps(transpiled, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1).strip(),
            str,
        )

    @data(0, 1, 2, 3)
    def test_qasm3_output_control_flow_expr(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow and `Expr` nodes can be
        dumped into OpenQASM 3."""
        transpiled = transpile(
            self._control_flow_expr_circuit(),
            backend=GenericBackendV2(
                num_qubits=27, coupling_map=MUMBAI_CMAP, control_flow=True, seed=2025_05_28
            ),
            optimization_level=optimization_level,
            seed_transpiler=2023_07_26,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(
            qasm3.dumps(transpiled, experimental=qasm3.ExperimentalFeatures.SWITCH_CASE_V1).strip(),
            str,
        )

    @data(0, 1, 2, 3)
    def test_qasm3_output_standalone_var(self, optimization_level):
        """Test that the output of a transpiled circuit with control flow and standalone `Var` nodes
        can be dumped into OpenQASM 3."""
        transpiled = transpile(
            self._standalone_var_circuit(),
            backend=GenericBackendV2(
                num_qubits=13,
                coupling_map=CouplingMap.from_line(13),
                control_flow=True,
                seed=2025_05_28,
            ),
            optimization_level=optimization_level,
            seed_transpiler=2024_05_01,
        )
        # TODO: There's not a huge amount we can sensibly test for the output here until we can
        # round-trip the OpenQASM 3 back into a Terra circuit.  Mostly we're concerned that the dump
        # itself doesn't throw an error, though.
        self.assertIsInstance(qasm3.dumps(transpiled), str)

    @data(0, 1, 2, 3)
    def test_transpile_target_no_measurement_error(self, opt_level):
        """Test that transpile with a target which contains ideal measurement works

        Reproduce from https://github.com/Qiskit/qiskit-terra/issues/8969
        """
        target = Target()
        target.add_instruction(Measure(), {(0,): None})
        qc = QuantumCircuit(1, 1)
        qc.measure(0, 0)
        res = transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)
        self.assertEqual(qc, res)

    def test_transpile_final_layout_updated_with_post_layout(self):
        """Test that the final layout is correctly set when vf2postlayout runs.

        Reproduce from #10457
        """

        def _get_index_layout(transpiled_circuit: QuantumCircuit, num_source_qubits: int):
            """Return the index layout of a transpiled circuit"""
            layout = transpiled_circuit.layout
            if layout is None:
                return list(range(num_source_qubits))

            pos_to_virt = {v: k for k, v in layout.input_qubit_mapping.items()}
            qubit_indices = []
            for index in range(num_source_qubits):
                qubit_idx = layout.initial_layout[pos_to_virt[index]]
                if layout.final_layout is not None:
                    qubit_idx = layout.final_layout[transpiled_circuit.qubits[qubit_idx]]
                qubit_indices.append(qubit_idx)
            return qubit_indices

        vf2_post_layout_called = False

        def callback(pass_, property_set, **_):
            nonlocal vf2_post_layout_called
            if isinstance(pass_, VF2PostLayout):
                vf2_post_layout_called = True
                # If this error indicates "no better solution found", the pass probably still
                # completely successfully, but it's not longer testing what this test is actually
                # trying to make assertions about; we need VF2PostLayout to _update_ the layout.
                self.assertIsNotNone(
                    property_set["post_layout"],
                    f"Stop reason: {property_set['VF2PostLayout_stop_reason']}",
                )

        coupling_map = [[0, 1], [1, 0], [1, 2], [1, 3], [2, 1], [3, 1], [3, 4], [4, 3]]
        backend = GenericBackendV2(
            num_qubits=5,
            basis_gates=["id", "sx", "x", "cx", "rz"],
            coupling_map=coupling_map,
            seed=2025_08_08,
        )
        qubits = 3
        qc = QuantumCircuit(qubits)
        for i in range(5):
            qc.cx(i % qubits, int(i + qubits / 2) % qubits)

        tqc = transpile(qc, backend=backend, seed_transpiler=2025_07_07, callback=callback)
        self.assertTrue(vf2_post_layout_called)
        self.assertEqual([1, 4, 3], _get_index_layout(tqc, qubits))

    @data("sabre", "lookahead", "basic")
    def test_final_layout_combined_correctly(self, routing):
        """Test that multiple `final_layout`s are combined correctly."""
        generators = {
            "sabre": lambda cmap, seed: passes.SabreSwap(cmap, seed=seed, trials=1),
            "lookahead": lambda cmap, _seed: passes.LookaheadSwap(cmap),
            "basic": lambda cmap, seed: passes.BasicSwap(cmap),
        }
        make_routing_pass = generators[routing]

        def random_line(num_qubits, rng):
            line = list(range(num_qubits))
            rng.shuffle(line)
            out = CouplingMap([[a, b] for a, b in zip(line[:-1], line[1:])])
            out.make_symmetric()
            return out

        rng = random.Random(0)
        num_qubits = 5

        # This is just loads of stars, so routing has to work for it.
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits):
            for j in range(num_qubits):
                if i == j:
                    continue
                qc.cx(i, j)
        # ... and the mirror, so the circuit implements the identity.
        qc.barrier()
        qc.compose(qc.inverse(), qc.qubits, inplace=True)

        # Routing needs a layout set, and we don't want qubit relabelling to mess with our test.
        pm = PassManager([passes.SetLayout(list(range(num_qubits))), passes.ApplyLayout()])
        # Now we route the circuit to several different coupling maps in a row, so we generate lots
        # of routing permutations, and each pass needs to combine them.
        pm += PassManager([make_routing_pass(random_line(num_qubits, rng), i) for i in range(5)])
        out = pm.run(qc)

        # The invariant of `routing_permutation` is that you're supposed to be able to append it as
        # a permutation and it will "undo" the effects.  We already know our circuit implements the
        # identity.
        out.append(PermutationGate(out.layout.routing_permutation()), out.qubits)
        self.assertEqual(Operator(out), Operator(np.eye(2**num_qubits)))

    @data(0, 1, 2, 3)
    def test_annotations_survive(self, optimization_level):
        """Test that custom annotations survive a full transpile."""

        # When the annotation framework expands to have more semantics, the test here might need to
        # expand to mark the custom annotations as being safe under any circuit transformation.
        class Custom(Annotation):
            namespace = "custom"

            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, Custom) and self.value == other.value

        qc = QuantumCircuit(4, 4)
        qc.h(0)
        outer_annotations = [Custom("hello"), Custom("world")]
        with qc.box(outer_annotations.copy()):
            qc.cx(0, 1)
            qc.cx(0, 2)
            qc.cx(0, 3)
            with qc.box([Custom("inner hello"), Custom("inner world")]):
                qc.cx(0, 1)
                qc.cx(0, 2)
                qc.cx(0, 3)
        qc.measure(qc.qubits, qc.clbits)
        backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=CouplingMap.from_line(5),
            basis_gates=["rz", "sx", "cx"],
            control_flow=True,
            seed=0,
        )
        out = transpile(
            qc, backend, optimization_level=optimization_level, seed_transpiler=2025_06_05
        )

        def get_first_box_op(qc):
            return next(inst.operation for inst in qc.data if inst.name == "box")

        outer_box_qc = get_first_box_op(qc)
        outer_box_out = get_first_box_op(out)
        self.assertEqual(outer_box_qc.annotations, outer_annotations)  # Check for no mutation.
        self.assertEqual(outer_box_qc.annotations, outer_box_out.annotations)
        inner_box_qc = get_first_box_op(outer_box_qc.blocks[0])
        inner_box_out = get_first_box_op(outer_box_out.blocks[0])
        self.assertEqual(inner_box_qc.annotations, inner_box_out.annotations)


class StreamHandlerRaiseException(StreamHandler):
    """Handler class that will raise an exception on formatting errors."""

    def handleError(self, record):
        raise sys.exc_info()


class TestLogTranspile(QiskitTestCase):
    """Testing the log_transpile option."""

    def setUp(self):
        super().setUp()
        logger = getLogger()
        self.addCleanup(logger.setLevel, logger.level)
        logger.setLevel("DEBUG")
        self.output = io.StringIO()
        logger.addHandler(StreamHandlerRaiseException(self.output))
        self.circuit = QuantumCircuit(QuantumRegister(1))

    def assertTranspileLog(self, log_msg):
        """Runs the transpiler and check for logs containing specified message"""
        transpile(self.circuit, seed_transpiler=42)
        self.output.seek(0)
        # Filter unrelated log lines
        output_lines = self.output.readlines()
        transpile_log_lines = [x for x in output_lines if log_msg in x]
        self.assertTrue(len(transpile_log_lines) > 0)

    def test_transpile_log_time(self):
        """Check Total Transpile Time is logged"""
        self.assertTranspileLog("Total Transpile Time")


class TestTranspileCustomPM(QiskitTestCase):
    """Test transpile function with custom pass manager"""

    def test_custom_multiple_circuits(self):
        """Test transpiling with custom pass manager and multiple circuits.
        This tests created a deadlock, so it needs to be monitored for timeout.
        See: https://github.com/Qiskit/qiskit-terra/issues/3925
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)

        pm_conf = PassManagerConfig(
            initial_layout=None,
            basis_gates=["u1", "u2", "u3", "cx"],
            coupling_map=CouplingMap([[0, 1]]),
            seed_transpiler=1,
        )
        passmanager = level_0_pass_manager(pm_conf)

        transpiled = passmanager.run([qc, qc])

        expected = QuantumCircuit(QuantumRegister(2, "q"))
        expected.append(U2Gate(0, math.pi), [0])
        expected.cx(0, 1)

        self.assertEqual(len(transpiled), 2)
        self.assertEqual(transpiled[0], expected)
        self.assertEqual(transpiled[1], expected)


@ddt
class TestTranspileParallel(QiskitTestCase):
    """Test transpile() in parallel."""

    def setUp(self):
        super().setUp()

        # Force parallel execution to True to test multiprocessing for this class
        cm = should_run_in_parallel.override(True)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    @data(0, 1, 2, 3)
    def test_parallel_multiprocessing(self, opt_level):
        """Test parallel dispatch works with multiprocessing."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        pm = generate_preset_pass_manager(opt_level, backend=GenericBackendV2(num_qubits=4))
        res = pm.run([qc, qc])
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_with_target(self, opt_level):
        """Test that parallel dispatch works with a manual target."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = GenericBackendV2(num_qubits=4).target
        res = transpile([qc] * 3, target=target, optimization_level=opt_level, seed_transpiler=42)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_num_processes_kwarg(self, num_processes):
        """Test that num_processes kwarg works when the system default parallel is true"""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        target = GenericBackendV2(num_qubits=27, coupling_map=MUMBAI_CMAP, seed=2025_05_28).target
        res = transpile([qc] * 3, target=target, num_processes=num_processes)
        self.assertIsInstance(res, list)
        for circ in res:
            self.assertIsInstance(circ, QuantumCircuit)

    @data(0, 1, 2, 3)
    def test_parallel_dispatch(self, opt_level):
        """Test that transpile in parallel works for all optimization levels."""
        backend = GenericBackendV2(num_qubits=5, basis_gates=["cx", "id", "rz", "sx", "x"], seed=42)
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0])
        for k in range(1, 4):
            qc.cx(qr[0], qr[k])
        qc.measure(qr, cr)
        qlist = [qc for k in range(15)]
        tqc = transpile(
            qlist, backend=backend, optimization_level=opt_level, seed_transpiler=424242
        )
        result = backend.run(tqc, seed_simulator=4242424242, shots=1000).result()
        counts = result.get_counts()
        for count in counts:
            self.assertTrue(math.isclose(count["00000"], 500, rel_tol=0.1))
            self.assertTrue(math.isclose(count["01111"], 500, rel_tol=0.1))

    @data(0, 1, 2, 3)
    def test_backendv2_and_basis_gates(self, opt_level):
        """Test transpile() with BackendV2 and basis_gates set."""
        backend = GenericBackendV2(num_qubits=6)
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.cz(0, 3)
        qc.cz(0, 4)
        qc.measure_all()
        tqc = transpile(
            qc,
            backend=backend,
            basis_gates=["u", "cz"],
            optimization_level=opt_level,
            seed_transpiler=12345678942,
        )
        op_count = set(tqc.count_ops())
        self.assertEqual({"u", "cz", "measure", "barrier"}, op_count)
        for inst in tqc.data:
            if inst.operation.name not in {"u", "cz"}:
                continue
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            self.assertIn(qubits, backend.target.qargs)

    @data(0, 1, 2, 3)
    def test_backendv2_and_coupling_map(self, opt_level):
        """Test transpile() with custom coupling map."""

        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cz(0, 1)
        qc.cz(0, 2)
        qc.cz(0, 3)
        qc.cz(0, 4)
        qc.measure_all()
        num_qubits = 5
        cmap = CouplingMap.from_line(num_qubits, bidirectional=False)
        tqc = transpile(
            qc,
            backend=GenericBackendV2(num_qubits=num_qubits),
            coupling_map=cmap,
            optimization_level=opt_level,
            seed_transpiler=12345678942,
        )
        op_count = set(tqc.count_ops())
        self.assertTrue({"rz", "sx", "x", "cx", "measure", "barrier"}.issuperset(op_count))
        for inst in tqc.data:
            if len(inst.qubits) == 2:
                qubit_0 = tqc.find_bit(inst.qubits[0]).index
                qubit_1 = tqc.find_bit(inst.qubits[1]).index
                self.assertEqual(qubit_1, qubit_0 + 1)

    def test_transpile_with_multiple_coupling_maps(self):
        """Test passing a different coupling map for every circuit"""
        backend = GenericBackendV2(num_qubits=4)

        qc = QuantumCircuit(3)
        qc.cx(0, 2)

        # Add a connection between 0 and 2 so that transpile does not change
        # the gates
        cmap = CouplingMap.from_line(7)
        cmap.add_edge(0, 2)

        with self.assertRaisesRegex(TranspilerError, "Only a single input coupling"):
            # Initial layout needed to prevent transpiler from relabeling
            # qubits to avoid doing the swap
            transpile(
                [qc] * 2,
                backend,
                coupling_map=[backend.coupling_map, cmap],
                initial_layout=(0, 1, 2),
                seed_transpiler=42,
            )

    @data(0, 1, 2, 3)
    def test_angle_bounds_respected(self, opt_level):
        """Test that angle bounds in the target are respected."""
        qc = QuantumCircuit(2)
        qc.append(XXPlusYYGate(np.pi, -np.pi / 2), qc.qubits)
        rzz_outside_bounds = QuantumCircuit(2)
        rzz_outside_bounds.h(0)
        rzz_outside_bounds.h(1)
        rzz_outside_bounds.rzz(pi, 0, 1)
        rzz_outside_bounds.z(0)
        rzz_outside_bounds.z(1)
        rzz_outside_bounds.rzz(pi / 4, 1, 0)
        rzz_outside_bounds.x(0)
        rzz_outside_bounds.x(1)
        rzz_outside_bounds.rzz(2 * pi, 0, 1)
        rzz_outside_bounds.y(0)
        rzz_outside_bounds.y(1)
        rzz_outside_bounds.rzz(3 * pi / 2, 0, 1)
        rzz_outside_bounds.h(0)
        rzz_outside_bounds.h(1)
        circs = [qc, rzz_outside_bounds]

        def fold_rzz(angles, _qubits):
            angle = angles[0]
            if 0 <= angle <= pi / 2:
                return None
            wrap_angle = np.angle(np.exp(1j * angle))
            qubits = [Qubit(), Qubit()]
            new_dag = DAGCircuit()
            new_dag.add_qubits(qubits)
            if 0 <= wrap_angle <= pi / 2:
                new_dag.apply_operation_back(
                    RZZGate(wrap_angle),
                    qargs=qubits,
                    check=False,
                )
            elif pi / 2 < wrap_angle <= pi:
                new_dag.global_phase = pi / 2
                new_dag.apply_operation_back(
                    RZGate(pi),
                    qargs=(qubits[0],),
                    cargs=(),
                    check=False,
                )
                new_dag.apply_operation_back(
                    RZGate(pi),
                    qargs=(qubits[1],),
                    check=False,
                )
                if not np.isclose(new_angle := (pi - wrap_angle), 0.0):
                    new_dag.apply_operation_back(
                        XGate(),
                        qargs=(qubits[0],),
                        check=False,
                    )
                    new_dag.apply_operation_back(
                        RZZGate(new_angle),
                        qargs=qubits,
                        check=False,
                    )
                    new_dag.apply_operation_back(
                        XGate(),
                        qargs=(qubits[0],),
                        check=False,
                    )
            elif -pi <= wrap_angle <= -pi / 2:
                new_dag.global_phase = -pi / 2
                new_dag.apply_operation_back(
                    RZGate(pi),
                    qargs=(qubits[0],),
                    check=False,
                )
                new_dag.apply_operation_back(
                    RZGate(pi),
                    qargs=(qubits[1],),
                    check=False,
                )
                if not np.isclose(new_angle := (pi - np.abs(wrap_angle)), 0.0):
                    new_dag.apply_operation_back(
                        RZZGate(new_angle),
                        qargs=qubits,
                        check=False,
                    )
            elif -pi / 2 < wrap_angle < 0:
                new_dag.apply_operation_back(
                    XGate(),
                    qargs=(qubits[0],),
                    check=False,
                )
                new_dag.apply_operation_back(
                    RZZGate(abs(wrap_angle)),
                    qargs=qubits,
                    check=False,
                )
                new_dag.apply_operation_back(
                    XGate(),
                    qargs=(qubits[0],),
                    check=False,
                )
            else:
                raise RuntimeError

            if pi < angle % (4 * pi) < 3 * pi:
                new_dag.global_phase += pi
            return new_dag

        theta = Parameter("theta")
        target = Target(num_qubits=2)
        target.add_instruction(
            SXGate(),
            {(0,): InstructionProperties(error=1e-5), (1,): InstructionProperties(error=3e-6)},
        )
        target.add_instruction(
            XGate(),
            {(0,): InstructionProperties(error=2e-5), (1,): InstructionProperties(error=6e-6)},
        )
        target.add_instruction(
            RZGate(theta),
            {(0,): InstructionProperties(error=0), (1,): InstructionProperties(error=0)},
        )
        target.add_instruction(
            RXGate(theta),
            {(0,): InstructionProperties(error=2e-5), (1,): InstructionProperties(error=6e-6)},
        )
        target.add_instruction(
            RZZGate(theta),
            {(0, 1): InstructionProperties(error=5e-3), (1, 0): InstructionProperties(error=5e-3)},
            angle_bounds=[(0, pi / 2)],
        )
        target.add_instruction(
            CZGate(),
            {(0, 1): InstructionProperties(error=5e-3), (1, 0): InstructionProperties(error=5e-3)},
        )
        registry = WrapAngleRegistry()
        registry.add_wrapper("rzz", fold_rzz)
        with patch.object(WrapAngles, "DEFAULT_REGISTRY", registry):
            transpiled = transpile(
                circs, target=target, optimization_level=opt_level, seed_transpiler=1234567890
            )

        self.assertTrue(Operator.from_circuit(transpiled[0]).equiv(Operator.from_circuit(circs[0])))
        self.assertTrue(Operator.from_circuit(transpiled[1]).equiv(Operator.from_circuit(circs[1])))
        for circ in transpiled:
            for inst in circ.data:
                self.assertTrue(
                    target.instruction_supported(
                        inst.name,
                        tuple(circ.find_bit(x).index for x in inst.qubits),
                        parameters=inst.operation.params,
                        check_angle_bounds=True,
                    )
                )


@ddt
class TestTranspileMultiChipTarget(QiskitTestCase):
    """Test transpile() with a disjoint coupling map."""

    def setUp(self):
        super().setUp()

        class FakeMultiChip(BackendV2):
            """Fake multi chip backend."""

            def __init__(self):
                super().__init__()
                graph = rx.generators.directed_heavy_hex_graph(3)
                num_qubits = len(graph) * 3
                rng = np.random.default_rng(seed=12345678942)
                rz_props = {}
                x_props = {}
                sx_props = {}
                measure_props = {}
                delay_props = {}
                self._target = Target("Fake multi-chip backend", num_qubits=num_qubits)
                for i in range(num_qubits):
                    qarg = (i,)
                    rz_props[qarg] = InstructionProperties(error=0.0, duration=0.0)
                    x_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-6, 1e-4), duration=rng.uniform(1e-8, 9e-7)
                    )
                    sx_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-6, 1e-4), duration=rng.uniform(1e-8, 9e-7)
                    )
                    measure_props[qarg] = InstructionProperties(
                        error=rng.uniform(1e-3, 1e-1), duration=rng.uniform(1e-8, 9e-7)
                    )
                    delay_props[qarg] = None
                self._target.add_instruction(XGate(), x_props)
                self._target.add_instruction(SXGate(), sx_props)
                self._target.add_instruction(RZGate(Parameter("theta")), rz_props)
                self._target.add_instruction(Measure(), measure_props)
                self._target.add_instruction(Delay(Parameter("t")), delay_props)
                cz_props = {}
                for i in range(3):
                    for root_edge in graph.edge_list():
                        offset = i * len(graph)
                        edge = (root_edge[0] + offset, root_edge[1] + offset)
                        cz_props[edge] = InstructionProperties(
                            error=rng.uniform(1e-5, 5e-3), duration=rng.uniform(1e-8, 9e-7)
                        )
                self._target.add_instruction(CZGate(), cz_props)

            @property
            def target(self):
                return self._target

            @property
            def max_circuits(self):
                return None

            @classmethod
            def _default_options(cls):
                return Options(shots=1024)

            def run(self, circuit, **kwargs):  # pylint:disable=arguments-renamed
                raise NotImplementedError

        self.backend = FakeMultiChip()

    @data(0, 1, 2, 3)
    def test_basic_connected_circuit(self, opt_level):
        """Test basic connected circuit on disjoint backend"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_triple_circuit(self, opt_level):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()

        if opt_level == 0:
            with self.assertRaises(TranspilerError):
                tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
            return

        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    def test_disjoint_control_flow(self):
        """Test control flow circuit on disjoint coupling map."""
        qc = QuantumCircuit(6, 1)
        qc.h(0)
        qc.ecr(0, 1)
        qc.cx(0, 2)
        qc.measure(0, 0)
        with qc.if_test((qc.clbits[0], True)):
            qc.reset(0)
            qc.cz(1, 0)
        qc.h(3)
        qc.cz(3, 4)
        qc.cz(3, 5)
        target = self.backend.target
        target.add_instruction(Reset(), {(i,): None for i in range(target.num_qubits)})
        target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, target=target, seed_transpiler=42)
        edges = set(target.build_coupling_map().graph.edge_list())

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)
                elif len(qargs) == 2:
                    self.assertIn(qargs, edges)
                self.assertIn(instruction.operation.name, target)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    def test_disjoint_control_flow_shared_classical(self):
        """Test circuit with classical data dependency between connected components."""
        creg = ClassicalRegister(19)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        qc.h(0)
        for i in range(18):
            qc.cx(0, i + 1)
        for i in range(18):
            qc.measure(i, creg[i])
        with qc.if_test((creg, 0)):
            qc.h(20)
            qc.ecr(20, 21)
            qc.ecr(20, 22)
            qc.ecr(20, 23)
            qc.ecr(20, 24)
        target = self.backend.target
        target.add_instruction(Reset(), {(i,): None for i in range(target.num_qubits)})
        target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, target=target, seed_transpiler=42)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(target.instruction_supported(instruction.operation.name, qargs))
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    @slow_test
    @data(2, 3)
    def test_six_component_circuit(self, opt_level):
        """Test input circuit with more than 1 component per backend component."""
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    def test_six_component_circuit_level_1(self):
        """Test input circuit with more than 1 component per backend component."""
        opt_level = 1
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_shared_classical_between_components_condition(self, opt_level):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(19)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        qc.h(0)
        for i in range(18):
            qc.cx(0, i + 1)
        for i in range(18):
            qc.measure(i, creg[i])

        with qc.if_test((creg, 0)):
            qc.ecr(20, 21)
        self.backend.target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=42)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )

    @data(0, 1, 2, 3)
    def test_shared_classical_between_components_condition_large_to_small(self, opt_level):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        # Component 0
        qc.h(24)
        qc.cx(24, 23)
        qc.measure(24, creg[0])
        qc.measure(23, creg[1])
        # Component 1
        with qc.if_test((creg, 0)):
            qc.h(0)
        for i in range(18):
            with qc.if_test((creg, 0)):
                qc.ecr(0, i + 1)
        self.backend.target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=123456789)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[23]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in [23, 24]}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(19)}, components[1])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        # The first node should be a measurement
        self.assertIsInstance(first_meas_node.op, Measure)
        # This should be in the first component
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while isinstance(op_node, DAGOpNode):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]

    @data(1, 2, 3)
    def test_shared_classical_between_components_condition_large_to_small_reverse_index(
        self, opt_level
    ):
        """Test a condition sharing classical bits between components."""
        creg = ClassicalRegister(2)
        qc = QuantumCircuit(25)
        qc.add_register(creg)
        # Component 0
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(0, creg[0])
        qc.measure(1, creg[1])
        # Component 1
        with qc.if_test((creg, 0)):
            qc.h(24)
        for i in range(23, 5, -1):
            with qc.if_test((creg, 0)):
                qc.ecr(24, i)
        self.backend.target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=2023)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[6]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(2)}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(6, 25)}, components[1])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        # The first node should be a measurement
        self.assertIsInstance(first_meas_node.op, Measure)
        # This should be in the first component
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while isinstance(op_node, DAGOpNode):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]

    @data(1, 2, 3)
    def test_chained_data_dependency(self, opt_level):
        """Test 3 component circuit with shared clbits between each component."""
        creg = ClassicalRegister(1)
        qc = QuantumCircuit(30)
        qc.add_register(creg)
        # Component 0
        qc.h(0)
        for i in range(9):
            qc.cx(0, i + 1)
        measure_op = Measure()
        qc.append(measure_op, [9], [creg[0]])
        # Component 1
        with qc.if_test((creg, 0)):
            qc.h(10)
        for i in range(11, 20):
            with qc.if_test((creg, 0)):
                qc.ecr(10, i)
        measure_op = Measure()
        qc.append(measure_op, [19], [creg[0]])
        # Component 2
        with qc.if_test((creg, 0)):
            qc.h(20)
        for i in range(21, 30):
            with qc.if_test((creg, 0)):
                qc.cz(20, i)
        measure_op = Measure()
        qc.append(measure_op, [29], [creg[0]])
        self.backend.target.add_instruction(IfElseOp, name="if_else")
        tqc = transpile(qc, self.backend, optimization_level=opt_level, seed_transpiler=2023)

        def _visit_block(circuit, qubit_mapping=None):
            for instruction in circuit:
                if instruction.operation.name == "barrier":
                    continue
                qargs = tuple(qubit_mapping[x] for x in instruction.qubits)
                self.assertTrue(
                    self.backend.target.instruction_supported(instruction.operation.name, qargs)
                )
                if isinstance(instruction.operation, ControlFlowOp):
                    for block in instruction.operation.blocks:
                        new_mapping = {
                            inner: qubit_mapping[outer]
                            for outer, inner in zip(instruction.qubits, block.qubits)
                        }
                        _visit_block(block, new_mapping)

        _visit_block(
            tqc,
            qubit_mapping={qubit: index for index, qubit in enumerate(tqc.qubits)},
        )
        # Check that virtual qubits that interact with each other via quantum links are placed into
        # the same component of the coupling map.
        initial_layout = tqc.layout.initial_layout
        coupling_map = self.backend.target.build_coupling_map()
        components = [
            connected_qubits(initial_layout[qc.qubits[0]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[10]], coupling_map),
            connected_qubits(initial_layout[qc.qubits[20]], coupling_map),
        ]
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(10)}, components[0])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(10, 20)}, components[1])
        self.assertLessEqual({initial_layout[qc.qubits[i]] for i in range(20, 30)}, components[2])

        # Check clbits are in order.
        # Traverse the output dag over the sole clbit, checking that the qubits of the ops
        # go in order between the components. This is a sanity check to ensure that routing
        # doesn't reorder a classical data dependency between components. Inside a component
        # we have the dag ordering so nothing should be out of order within a component.
        tqc_dag = circuit_to_dag(tqc)
        qubit_map = {qubit: index for index, qubit in enumerate(tqc_dag.qubits)}
        input_node = tqc_dag.input_map[tqc_dag.clbits[0]]
        first_meas_node = tqc_dag._find_successors_by_edge(
            input_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        self.assertIsInstance(first_meas_node.op, Measure)
        self.assertIn(qubit_map[first_meas_node.qargs[0]], components[0])
        op_node = tqc_dag._find_successors_by_edge(
            first_meas_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while not isinstance(op_node.op, Measure):
            self.assertIn(qubit_map[op_node.qargs[0]], components[1])
            op_node = tqc_dag._find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]
        self.assertIn(qubit_map[op_node.qargs[0]], components[1])
        op_node = tqc_dag._find_successors_by_edge(
            op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
        )[0]
        while not isinstance(op_node.op, Measure):
            self.assertIn(qubit_map[op_node.qargs[0]], components[2])
            op_node = tqc_dag._find_successors_by_edge(
                op_node._node_id, lambda edge_data: isinstance(edge_data, Clbit)
            )[0]
        self.assertIn(qubit_map[op_node.qargs[0]], components[2])

    @data("sabre", "basic", "lookahead")
    def test_basic_connected_circuit_dense_layout(self, routing_method):
        """Test basic connected circuit on disjoint backend"""
        qc = QuantumCircuit(5)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    # Lookahead swap skipped for performance
    @data("sabre", "basic")
    def test_triple_circuit_dense_layout(self, routing_method):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data("sabre", "basic", "lookahead")
    def test_triple_circuit_invalid_layout(self, routing_method):
        """Test a split circuit with one circuit component per chip."""
        qc = QuantumCircuit(30)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.measure_all()

        with self.assertRaises(TranspilerError):
            with self.assertWarnsRegex(
                DeprecationWarning,
                expected_regex="The `target` parameter should be used instead",
            ):
                if routing_method == "stochastic":
                    with self.assertWarnsRegex(
                        DeprecationWarning,
                        expected_regex="The StochasticSwap transpilation pass is a suboptimal",
                    ):
                        transpile(
                            qc,
                            self.backend,
                            layout_method="trivial",
                            routing_method=routing_method,
                            seed_transpiler=42,
                        )
                else:
                    transpile(
                        qc,
                        self.backend,
                        layout_method="trivial",
                        routing_method=routing_method,
                        seed_transpiler=42,
                    )

    # Lookahead swap skipped for performance reasons, stochastic moved to new test due to deprecation
    @data("sabre", "basic")
    def test_six_component_circuit_dense_layout(self, routing_method):
        """Test input circuit with more than 1 component per backend component."""
        qc = QuantumCircuit(42)
        qc.h(0)
        qc.h(10)
        qc.h(20)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        qc.cx(0, 4)
        qc.cx(0, 5)
        qc.cx(0, 6)
        qc.cx(0, 7)
        qc.cx(0, 8)
        qc.cx(0, 9)
        qc.ecr(10, 11)
        qc.ecr(10, 12)
        qc.ecr(10, 13)
        qc.ecr(10, 14)
        qc.ecr(10, 15)
        qc.ecr(10, 16)
        qc.ecr(10, 17)
        qc.ecr(10, 18)
        qc.ecr(10, 19)
        qc.cy(20, 21)
        qc.cy(20, 22)
        qc.cy(20, 23)
        qc.cy(20, 24)
        qc.cy(20, 25)
        qc.cy(20, 26)
        qc.cy(20, 27)
        qc.cy(20, 28)
        qc.cy(20, 29)
        qc.h(30)
        qc.cx(30, 31)
        qc.cx(30, 32)
        qc.cx(30, 33)
        qc.h(34)
        qc.cx(34, 35)
        qc.cx(34, 36)
        qc.cx(34, 37)
        qc.h(38)
        qc.cx(38, 39)
        qc.cx(39, 40)
        qc.cx(39, 41)
        qc.measure_all()
        tqc = transpile(
            qc,
            self.backend,
            layout_method="dense",
            routing_method=routing_method,
            seed_transpiler=42,
        )
        for inst in tqc.data:
            qubits = tuple(tqc.find_bit(x).index for x in inst.qubits)
            op_name = inst.operation.name
            if op_name == "barrier":
                continue
            self.assertIn(qubits, self.backend.target[op_name])

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops(self, opt_level):
        """Test qubits without operations aren't ever used."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(3)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        tqc = transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)
        invalid_qubits = {3, 4}
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_with_routing(self, opt_level):
        """Test qubits without operations aren't ever used."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(4)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(4)})
        target.add_instruction(
            CXGate(),
            {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0), (2, 3)]},
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 3)
        tqc = transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)
        invalid_qubits = {
            4,
        }
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_circuit_too_large(self, opt_level):
        """Test qubits without operations aren't ever used and error if circuit needs them."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(0, 3)
        with self.assertRaises(TranspilerError):
            transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)

    @data(0, 1, 2, 3)
    def test_transpile_target_with_qubits_without_ops_circuit_too_large_disconnected(
        self, opt_level
    ):
        """Test qubits without operations aren't ever used if a disconnected circuit needs them."""
        target = Target(num_qubits=5)
        target.add_instruction(XGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(HGate(), {(i,): InstructionProperties(error=0.5) for i in range(3)})
        target.add_instruction(
            CXGate(), {edge: InstructionProperties(error=0.5) for edge in [(0, 1), (1, 2), (2, 0)]}
        )
        qc = QuantumCircuit(5)
        qc.x(0)
        qc.x(1)
        qc.x(3)
        qc.x(4)
        with self.assertRaises(TranspilerError):
            transpile(qc, target=target, optimization_level=opt_level, seed_transpiler=42)

    @data(0, 1, 2, 3)
    def test_barrier_no_leak_disjoint_connectivity(self, opt_level):
        """Test that we don't leak an internal labelled barrier from disjoint layout processing."""
        cmap = CouplingMap([(0, 1), (1, 2), (3, 4)])
        qc = QuantumCircuit(cmap.size(), cmap.size())
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 0)
        qc.cx(3, 4)
        qc.measure(qc.qubits, qc.clbits)
        out = transpile(qc, coupling_map=cmap, optimization_level=opt_level)
        self.assertNotIn("barrier", out.count_ops())

    @data(0, 1, 2, 3)
    def test_transpile_does_not_affect_backend_coupling(self, opt_level):
        """Test that transpiliation of a circuit does not mutate the `CouplingMap` stored by a V2
        backend.  Regression test of gh-9997."""
        qc = QuantumCircuit(127)
        for i in range(1, 127):
            qc.ecr(0, i)
        backend = GenericBackendV2(
            num_qubits=130, coupling_map=CouplingMap.from_line(130, bidirectional=False)
        )
        original_map = copy.deepcopy(backend.coupling_map)
        transpile(qc, backend, optimization_level=opt_level, seed_transpiler=42)
        self.assertEqual(original_map, backend.coupling_map)

    @combine(
        optimization_level=[0, 1, 2, 3],
        scheduling_method=["asap", "alap"],
    )
    def test_transpile_target_with_qubits_without_delays_with_scheduling(
        self, optimization_level, scheduling_method
    ):
        """Test qubits without operations aren't ever used."""
        no_delay_qubits = [1, 3, 4]
        target = Target(num_qubits=5, dt=1)
        target.add_instruction(
            XGate(), {(i,): InstructionProperties(duration=160) for i in range(4)}
        )
        target.add_instruction(
            HGate(), {(i,): InstructionProperties(duration=160) for i in range(4)}
        )
        target.add_instruction(
            CXGate(),
            {
                edge: InstructionProperties(duration=800)
                for edge in [(0, 1), (1, 2), (2, 0), (2, 3)]
            },
        )
        target.add_instruction(
            Delay(Parameter("t")), {(i,): None for i in range(4) if i not in no_delay_qubits}
        )
        qc = QuantumCircuit(4)
        qc.x(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.cx(1, 3)
        qc.cx(0, 3)
        tqc = transpile(
            qc,
            target=target,
            optimization_level=optimization_level,
            scheduling_method=scheduling_method,
            seed_transpiler=42,
        )
        invalid_qubits = {
            4,
        }
        self.assertEqual(tqc.num_qubits, 5)
        for inst in tqc.data:
            for bit in inst.qubits:
                self.assertNotIn(tqc.find_bit(bit).index, invalid_qubits)
                if isinstance(inst.operation, Delay):
                    self.assertNotIn(tqc.find_bit(bit).index, no_delay_qubits)
