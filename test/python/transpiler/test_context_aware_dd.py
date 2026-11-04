# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the context-aware dynamical decoupling passes."""

import math
import itertools
import contextlib
from ddt import ddt, data, unpack
import numpy as np
import rustworkx as rx

from qiskit.circuit import QuantumCircuit, Delay, Reset
from qiskit.circuit.library import SXGate, CXGate, XGate, CZGate, ECRGate
from qiskit.transpiler import PassManager, Target, InstructionProperties
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import (
    SetLayout,
    FullAncillaAllocation,
    EnlargeWithAncilla,
    ContextAwareDynamicalDecoupling,
    ALAPScheduleAnalysis,
    ASAPScheduleAnalysis,
    ApplyLayout,
)
from qiskit.providers.fake_provider import GenericBackendV2

from test import QiskitTestCase


class Mock127q(GenericBackendV2):
    """A fake 127-qubit backend.

    The node number equals ``ibm_sherbrooke`` though the directions of the ECR gates might
    not match.

    Used instead of the GenericBackendV2 to ensure the 2-qubit gate lengths are larger
    than 2 X gates, and fixes the seed.
    """

    def __init__(self):
        coupling_map = []
        line_lengths = [13] + 5 * [14] + [13]
        lines = []
        start = 0
        for length in line_lengths:
            line = list(range(start, start + length + 1))
            coupling_map += list(zip(line[:-1], line[1:]))
            lines.append(line)
            start += length + 5  # 4 nodes are in between the lines

        for i, (prev, after) in enumerate(zip(lines[:-1], lines[1:])):
            from_front = i % 2 == 0
            for j in range(4):
                node = prev[-1] + j + 1 if from_front else prev[-1] + 4 - j  # node number
                index = 4 * j if from_front else ~(4 * j)
                coupling_map.append((prev[index], node))
                coupling_map.append((node, after[index]))

        basis_gates = ["id", "rz", "sx", "x", "ecr"]

        super().__init__(num_qubits=127, coupling_map=coupling_map, basis_gates=basis_gates, seed=9)

    def _get_noise_defaults(self, name, num_qubits):
        # gate durations and errors as (min duration, max duration, min error, max error)
        if name == "ecr":
            return (341e-9, 881e-9, 1e-3, 3e-3)
        elif name == "x":
            return (1e-10, 1.1e-10, 9e-5, 6e-3)
        elif name == "sx":
            return (5e-11, 5.5e-11, 9e-5, 6e-3)
        return super()._get_noise_defaults(name, num_qubits)


@ddt
class TestContextAwareDD(QiskitTestCase):
    """Test context-aware dynamical decoupling.

    See the reference: https://arxiv.org/abs/2403.06852v2.
    """

    def setUp(self):
        super().setUp()

        # gate times in terms of dt
        self.dt = 1e-9
        self.t_cx = 1e3
        self.t_sx = 10
        self.t_x = 20

    @contextlib.contextmanager
    def assertMultiDelayInserted(self, expected_blocks):
        """Assert that multi-delays are inserted.

        These are to be given as [(start1, end1, qubits1), (start2, end2, qubits2), ...]
        per adjacent block.
        """
        module = "qiskit.transpiler.passes.scheduling.padding.context_aware_dynamical_decoupling"
        try:
            with self.assertLogs(module, level="DEBUG") as logs:
                yield logs
        finally:
            template = "MultiDelay({}:{} on {})"  # MultiDelay(start:end on qubits)
            for delay_block in expected_blocks:
                message = f"DEBUG:{module}:Split adjacent delay block into " + "\n".join(
                    [template.format(start, end, qubits) for start, end, qubits in delay_block]
                )

                self.assertIn(message, logs.output)

    def test_full(self):
        """Test the full workflow on a simple circuit and a concrete reference."""
        circuit = QuantumCircuit(5)
        circuit.sx(circuit.qubits)
        circuit.barrier()
        circuit.cx(1, 2)
        circuit.barrier()
        circuit.cx(0, 1)
        circuit.cx(3, 4)

        target = get_toy_target(num_qubits=5, t_cx=self.t_cx)
        durations = target.durations()

        pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        pm.append(ContextAwareDynamicalDecoupling(target))

        ref = QuantumCircuit(5)
        ref.sx(ref.qubits)
        ref.barrier()
        apply_delay_sequence(ref, 0, self.t_cx, durations, order=1)  # ctrl sequence
        ref.cx(1, 2)
        apply_delay_sequence(ref, 3, self.t_cx, durations, order=0)  # tgt sequence
        apply_delay_sequence(ref, 4, self.t_cx, durations, order=1)  # lowest seq. orthogonal to tgt

        ref.barrier()
        ref.cx(0, 1)
        apply_delay_sequence(ref, 2, self.t_cx, durations, order=2)  # ctrl+tgt sequence
        ref.cx(3, 4)

        circuit = pm.run(circuit)
        self.assertEqual(circuit, ref)

    @data(True, False)
    def test_skip_initial_delays(self, skip_initial):
        """Test initial delays are skipped.

           delays here since not after a reset
                    v
        q_0: ──■───────
             ┌─┴─┐
        q_1: ┤ X ├──■──
             └───┘┌─┴─┐
        q_2: ─────┤ X ├
               ^  └───┘
             no delay since after qubit initialization

        """
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        target = get_toy_target(num_qubits=5)
        pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        pm.append(ContextAwareDynamicalDecoupling(target, skip_reset_qubits=skip_initial))

        circuit = pm.run(circuit)

        # a single ctrl-specific decoupling sequence with 2 X gates if initial delays are skipped,
        # otherwise we get an additional 5 sequences à 2 X gates due to the delay blocks on the
        # additional idle qubits of the 5-qubit target
        expected_x = 2 if skip_initial else 12

        self.assertEqual(circuit.count_ops().get("x", 0), expected_x)

    def test_min_duration_default(self):
        """Test the default value for minimum durations."""
        circuit = QuantumCircuit(3)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        # target = get_toy_target(num_qubits=5)
        target, max_diff = get_varying_target(5)
        dd = ContextAwareDynamicalDecoupling(target)

        self.assertAlmostEqual(2 * max_diff / target.dt, dd._min_duration)

    def test_2q_gate_combos(self):
        """Test the ctrl/tgt specific behavior for CX/ECR and default for others (like CZ).

        There are specific sequence for control/target/control+target spectator qubits for
        CX and ECR. Other gates do not get special sequences.

                     ┌────┐ ░      ┌──────┐
            q_0 -> 0 ┤ √X ├─░──────┤0     ├──────────
                     ├────┤ ░      │   ?  │
            q_1 -> 1 ┤ √X ├─░──────┤1     ├──────────
                     ├────┤ ░      └──────┘
            q_2 -> 2 ┤ √X ├─░── test this sequence ──
                     ├────┤ ░      ┌──────┐
            q_3 -> 3 ┤ √X ├─░──────┤0     ├──────────
                     ├────┤ ░      │   ?  │
            q_4 -> 4 ┤ √X ├─░──────┤1     ├──────────
                     └────┘ ░      └──────┘

        """
        target = get_toy_target(num_qubits=5, t_cx=self.t_cx)
        pm = _get_schedule_pm(target, list(range(5)))
        pm.append(ContextAwareDynamicalDecoupling(target))

        gates = ["cx", "ecr", "cz"]
        for top in gates:
            for bottom in gates:
                if top in ["cx", "ecr"]:
                    if bottom in ["cx", "ecr"]:
                        # squeezed between control and target
                        order = 2
                    else:
                        # other gate (CZ) has no specific behavior, use target-specific sequence
                        order = 0
                elif bottom in ["cx", "ecr"]:
                    # use ctrl-specific sequence
                    order = 1
                else:  # both gates do not have special behavior, use 0
                    order = 0

                # construct circuit with specified gates
                circuit = QuantumCircuit(5)
                circuit.sx(circuit.qubits)
                circuit.barrier()
                getattr(circuit, top)(0, 1)
                getattr(circuit, bottom)(3, 4)
                circuit = pm.run(circuit)

                # compute reference
                ref = QuantumCircuit(5)
                ref.sx(ref.qubits)
                ref.barrier()
                getattr(ref, top)(0, 1)
                apply_delay_sequence(ref, 2, self.t_cx, target.durations(), order)
                getattr(ref, bottom)(3, 4)

                with self.subTest(top=top, bottom=bottom):
                    self.assertEqual(circuit, ref)

    def test_threshold_skipping(self):
        """Test skipping of delays that are below the threshold."""
        skip_threshold = 0.6  # arbitrary threshold below 1

        target = get_toy_target(num_qubits=5, t_x=self.t_x)
        pm = _get_schedule_pm(target, [0])
        pm.append(ContextAwareDynamicalDecoupling(target, skip_dd_threshold=skip_threshold))

        epsilon = 0.01  # ask a mathematician for a formal definition of epsilon
        dd_duration = 2 * self.t_x
        for exceed_threshold in [True, False]:
            # go above threshold once and below once
            if exceed_threshold:
                delay = math.floor(dd_duration / (skip_threshold + epsilon))
            else:
                delay = math.ceil(dd_duration / (skip_threshold - epsilon))

            circuit = QuantumCircuit(1)
            circuit.sx(0)
            circuit.delay(delay, 0)
            circuit.sx(0)

            circuit = pm.run(circuit)

            with self.subTest(exceed_threshold=exceed_threshold):
                expected_x = 0 if exceed_threshold else 2
                self.assertEqual(circuit.count_ops().get("x", 0), expected_x)

    @data(4, 20)  # X gate length is integer multiple of these values
    def test_pulse_alignment(self, alignment):
        """Test setting the pulse alignment.

        Pulses should only start at integer multiples of the allowed pulse alignment.
        """
        target = get_toy_target(num_qubits=5)
        pm = _get_schedule_pm(target, [0])
        pm.append(
            ContextAwareDynamicalDecoupling(
                target, pulse_alignment=alignment, skip_reset_qubits=False
            )
        )

        circuit = QuantumCircuit(1)
        circuit.delay(100, 0)
        circuit = pm.run(circuit)

        x_times = [
            time for gate, time in pm.property_set["node_start_time"].items() if gate.op.name == "x"
        ]

        self.assertTrue((np.asarray(x_times) % alignment == 0).all())

    def test_invalid_pulse_alignment(self):
        """Test an error is raised if the X gate length is not compatible with the pulse alignment."""
        target = get_toy_target(num_qubits=5, t_x=self.t_x)

        pm = _get_schedule_pm(target, [0])
        pm.append(
            ContextAwareDynamicalDecoupling(
                target, pulse_alignment=self.t_x + 1, skip_reset_qubits=False
            )
        )

        circuit = QuantumCircuit(1)
        circuit.delay(100, 0)

        with self.assertRaises(TranspilerError):
            _ = pm.run(circuit)

    def test_orthogonal_sequences(self):
        """Check orthogonality of sequences up to the supported order."""
        target = get_toy_target(5)
        cadd = ContextAwareDynamicalDecoupling(target)

        # get the DD sequences and check for the smallest time frame
        dd_sequences = [np.asarray(cadd.get_orthogonal_sequence(order)[0]) for order in range(6)]
        # we must exclude 0, which is sometimes used as padding in the end
        smallest_unit = np.min([np.min(row[np.where(row > 0)]) for row in dd_sequences])

        # expand the DD times to vectors with an entry +1 if the qubit is in + state
        # and a -1 if the qubit is in - state
        sign_vectors = [expand_dd_sequence(sequence, smallest_unit) for sequence in dd_sequences]

        # check pairwise orthogonality
        for i, row_i in enumerate(sign_vectors):
            for j, row_j in enumerate(sign_vectors[i + 1 :]):
                with self.subTest(i=i, j=j):
                    self.assertAlmostEqual(np.dot(row_i, row_j), 0)

    def test_exceed_default_highest_order(self):
        """Check an error is raised if we query a sequence with order too high."""
        order_threshold = 7  # pick an order above the default max order (5)
        kranka = Target(num_qubits=order_threshold, dt=self.dt)  # a target with scary connectivity
        coupling_web = itertools.combinations(list(range(order_threshold)), 2)
        cx_props = {
            connection: InstructionProperties(duration=self.t_cx * self.dt, error=1e-2)
            for connection in coupling_web
        }
        x_props = {
            (i,): InstructionProperties(duration=self.t_x * self.dt, error=1e-4)
            for i in range(kranka.num_qubits)
        }
        kranka.add_instruction(CXGate(), cx_props)
        kranka.add_instruction(XGate(), x_props)
        kranka.add_instruction(Delay(1), x_props)

        circuit = QuantumCircuit(order_threshold)
        circuit.x(circuit.qubits)
        circuit.barrier()
        circuit.delay(1000)  # note this should be long enough to fit the DD sequences

        pm = _get_schedule_pm(kranka, list(range(kranka.num_qubits)))
        pm.append(ContextAwareDynamicalDecoupling(kranka, skip_reset_qubits=False))

        circuit = pm.run(circuit)

        # number of flips in the first 7 Walsh-Hadamard sequences
        num_dd_x = 2 + 2 + 4 + 4 + 6 + 6 + 8
        num_x = circuit.num_qubits  # X layer
        self.assertEqual(num_x + num_dd_x, circuit.count_ops().get("x", 0))

    def test_collecting_pyramid(self):
        """Test collecting the largest adjacent blocks for a pyramid structure.

        The delays form the following circuit (just on more qubits in the test):

            q0: ------████-
            q1: ----██████-
            q2: -█████████-
            q3: ----██████-
            q4: ------████-

        The logs contain the multi-delay instructions, which we check here.
        """
        num_qubits = 9
        circuit = QuantumCircuit(num_qubits)
        center = num_qubits // 2
        circuit.sx(center)
        for i in range(num_qubits // 2):
            circuit.cx(center - i, center - (i + 1))
            circuit.cx(center + i, center + i + 1)

        target = get_toy_target(num_qubits=num_qubits, t_sx=self.t_sx, t_cx=self.t_cx)
        pm = _get_schedule_pm(target, list(range(num_qubits)))
        pm.append(ContextAwareDynamicalDecoupling(target))

        first_delay = [(1010, 2010, (center - 1,))]
        pyramid_delays = [
            (2010, 3010, (center,)),
            (3010, 4010, tuple(range(center - 1, center + 2))),
            (4010, 5010, tuple(range(center - 2, center + 3))),
        ]
        expected_blocks = [first_delay, pyramid_delays]

        with self.assertMultiDelayInserted(expected_blocks):
            _ = pm.run(circuit)

    def test_no_scheduling(self):
        """Test a sensible error is raised if no scheduling is performed prior to CA-DD."""
        circuit = QuantumCircuit(1)
        pass_ = ContextAwareDynamicalDecoupling(get_toy_target(1))

        with self.assertRaisesRegex(TranspilerError, "scheduling passes"):
            _ = pass_(circuit)

    def test_collecting_diamond_with_initial(self):
        """Test collecting delays in form of a diamond, including initial delays.

        This tests delay blocks are correctly merged when they get smaller again.

            q0: --████--
            q1: -██████-
            q2: --████--

        """
        circuit = QuantumCircuit(8)
        circuit.cx(2, 3)
        circuit.cx(1, 2)
        circuit.cx(0, 1)
        circuit.cx(0, 1)
        circuit.cx(1, 2)

        circuit.cx(4, 5)
        circuit.cx(5, 6)
        circuit.cx(6, 7)
        circuit.cx(6, 7)
        circuit.cx(5, 6)

        target = get_toy_target(num_qubits=circuit.num_qubits, t_sx=self.t_sx, t_cx=self.t_cx)
        pm = PassManager(
            [
                ASAPScheduleAnalysis(target.durations(), target),
                ContextAwareDynamicalDecoupling(target, skip_reset_qubits=False),
            ]
        )
        top_left = [(0, 1000, (0, 1)), (1000, 2000, (0,))]
        bottom_left = [
            (0, 1000, (6, 7)),
            (1000, 2000, (7,)),
        ]
        diamond = [(1000, 2000, (3, 4)), (2000, 4000, (2, 3, 4, 5)), (4000, 5000, (3, 4))]
        top_right = [(4000, 5000, (0,))]
        bottom_right = [(4000, 5000, (7,))]
        expected_blocks = [top_left, top_right, bottom_left, bottom_right, diamond]

        with self.assertMultiDelayInserted(expected_blocks):
            _ = pm.run(circuit)

    def test_collecting_donut(self):
        """Test collecting delays in a donut shape."""
        circuit = QuantumCircuit(3)
        circuit.delay(2 * self.t_cx, 0)
        circuit.delay(self.t_cx, 1)
        circuit.delay(2 * self.t_cx, 2)

        circuit.x(1)
        circuit.delay(self.t_cx - self.t_x, 1)

        target = get_toy_target(num_qubits=3, t_cx=self.t_cx, t_x=self.t_x)
        pm = PassManager(
            [
                ASAPScheduleAnalysis(target.durations(), target=target),
                ContextAwareDynamicalDecoupling(target, skip_reset_qubits=False),
            ]
        )

        expected_block = [
            (0, 1000, (0, 1, 2)),
            (1000, 1020, (0,)),
            (1000, 1020, (2,)),
            (1020, 2000, (0, 1, 2)),
        ]

        with self.assertMultiDelayInserted([expected_block]):
            _ = pm.run(circuit)

    def test_skip_after_reset(self):
        """Test delays after a reset are ignored.

        We're testing on a circuit looking like

            q0: -███-X-███-X-|0>-███-X-███
                  ^               ^
         ignored (initial)   ignored (after reset)

        """
        dt = 100
        circuit = QuantumCircuit(1)
        circuit.delay(dt, 0)
        circuit.x(0)
        circuit.delay(dt, 0)
        circuit.x(0)
        circuit.reset(0)
        circuit.delay(dt, 0)
        circuit.x(0)
        circuit.delay(dt, 0)

        target = get_toy_target(1)
        schedule_pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        dd = PassManager(
            [
                ContextAwareDynamicalDecoupling(target, skip_reset_qubits=True),
            ]
        )
        pm = schedule_pm + dd
        circuit = pm.run(circuit)

        reference = QuantumCircuit(1)
        durations = target.durations()
        reference.delay(dt, 0)  # initial is ignored
        reference.x(0)
        apply_delay_sequence(reference, 0, dt, durations, order=0)
        reference.x(0)
        reference.reset(0)
        reference.delay(dt, 0)  # after reset is ignored
        reference.x(0)
        apply_delay_sequence(reference, 0, dt, durations, order=0)

        self.assertEqual(reference, circuit)

    def test_127q(self):
        """Test running on a larger backend with 100+ qubits.

        Since this is a bipartite graph, the coloring problem can be solved optimally, which
        in this case means the maximum order is 1.
        """
        backend = Mock127q()
        target = backend.target

        snake = (
            list(range(14))[::-1]
            + [14]
            + list(range(18, 33))
            + [36]
            + list(range(37, 52))[::-1]
            + [52]
            + list(range(56, 71))
            + [74]
            + list(range(75, 90))[::-1]
            + [90]
            + list(range(94, 109))
            + [112]
            + list(range(113, 127))[::-1]
        )

        # build a circuit with staircase ECR gates in native directions, then we'll get
        # len(snake) - 2 concurrent delay blocks, which we can test for
        circuit = QuantumCircuit(backend.num_qubits)
        circuit.sx(snake)
        for ctrl, tgt in zip(snake[:-1], snake[1:]):
            if target.instruction_supported("ecr", (ctrl, tgt)):
                circuit.ecr(ctrl, tgt)
            else:
                circuit.ecr(tgt, ctrl)

        pre = generate_preset_pass_manager(
            optimization_level=0, target=target, initial_layout=list(range(127))
        )
        dd = PassManager(
            [
                ALAPScheduleAnalysis(target.durations(), target=target),
                ContextAwareDynamicalDecoupling(target, min_duration=0),
            ]
        )
        circuit = dd.run(pre.run(circuit))

        # compute the number of expected X gates via the number of DD sequences, which is in
        # the upper triangular part of the circuit, and each sequence has 2 X gates
        num_dd_sequences = int((len(snake) - 2) * (len(snake) - 1) / 2)
        num_x = 2 * num_dd_sequences

        # this can fail if the coloring is not solved optimally (i.e. we have colors >= 2)
        # or if there are more than len(snake) - 2 layers of multi delay operations
        self.assertEqual(num_x, circuit.count_ops().get("x", 0))

    def test_min_duration(self):
        """Test cutting off short peaks below the joinable duration.

        delay is too short
                   v
          q0: -----█------      q0: ------------
          q1: -██████████-  ->  q1: -██████████-
          q2: -----██████-      q2: -----██████-

        """
        circuit = QuantumCircuit(3)
        _ = [circuit.x([0, 2]) for _ in range(4)]
        circuit.delay(self.t_x, 0)  # this one should be ignored
        circuit.delay(10 * self.t_x, 1)
        circuit.delay(6 * self.t_x, 2)
        _ = [circuit.x(0) for _ in range(5)]

        num_x = circuit.count_ops()["x"]  # number of original X gates
        num_x += 2 + 2 + 2  # we expect 3 DD sequences -- without the cutoff, short delay

        target = get_toy_target(num_qubits=3, t_x=self.t_x)
        pm = PassManager(
            [
                ASAPScheduleAnalysis(target.durations(), target=target),
                ContextAwareDynamicalDecoupling(
                    target,
                    skip_reset_qubits=False,
                    min_duration=self.t_x + 1,
                ),
            ]
        )
        circuit = pm.run(circuit)

        self.assertEqual(circuit.count_ops().get("x", 0), num_x)

    @data(
        (rx.ColoringStrategy.Degree, 8),
        (rx.ColoringStrategy.IndependentSet, 10),
        (rx.ColoringStrategy.Saturation, 8),
    )
    @unpack
    def test_coloring_strategy(self, strategy, num_x):
        """Test setting different coloring strategies.

        Not all perform the same, e.g. IndependentSet is suboptimal on this qubit line
        with a boundary condition of the first idle wire being a target spectator.
        """
        circuit = QuantumCircuit(6)
        circuit.cx(0, 1)

        target = get_toy_target(circuit.num_qubits)
        pm = _get_schedule_pm(target, list(range(circuit.num_qubits)))
        pm.append(
            ContextAwareDynamicalDecoupling(
                target, skip_reset_qubits=False, coloring_strategy=strategy
            )
        )

        circuit = pm.run(circuit)

        self.assertEqual(num_x, circuit.count_ops().get("x", 0))


def _get_schedule_pm(target, initial_layout):
    durations = target.durations()

    schedule_pm = PassManager(
        [
            SetLayout(initial_layout),
            FullAncillaAllocation(target),
            EnlargeWithAncilla(),
            ApplyLayout(),
            ALAPScheduleAnalysis(durations, target),
        ]
    )

    return schedule_pm


def apply_delay_sequence(circuit, qubit, timespan, durations, order):
    """Tool to apply a delay sequence to a circuit."""
    if order == 0:
        dt = (timespan - 2 * durations.get("x", qubit)) / 2
        for _ in range(2):
            circuit.delay(dt, qubit)
            circuit.x(qubit)
    elif order == 1:
        reduced_timespan = timespan - 2 * durations.get("x", qubit)
        circuit.delay(reduced_timespan / 4, qubit)
        circuit.x(qubit)
        circuit.delay(reduced_timespan / 2, qubit)
        circuit.x(qubit)
        circuit.delay(reduced_timespan / 4, qubit)
    elif order == 2:
        dt = (timespan - 4 * durations.get("x", qubit)) / 4
        for _ in range(4):
            circuit.delay(dt, qubit)
            circuit.x(qubit)

    return circuit


def expand_dd_sequence(sequence, unit):
    """Expand durations in terms of the unit.

    E.g. in the unit is 1/8 and sequence is [1/4, 1/2, 1/4] expand to [++----++].
    """
    signs = []
    sign = 1

    for timespan in sequence:
        num_items = int(timespan / unit)
        signs += num_items * [sign]
        sign *= -1

    return signs


def get_toy_target(num_qubits, dt=1e-9, t_cx=1e3, t_sx=10, t_x=20):
    """Get a toy target."""

    # set up an idealistic target to test context-aware DD in a clean setting
    # (if I don't also add realistic settings I should be scolded)
    target = Target(num_qubits=num_qubits, dt=dt)
    # bidirectional linear next neighbor
    linear_topo = [(i, i + 1) for i in range(num_qubits - 1)]
    linear_topo += [tuple(reversed(connection)) for connection in linear_topo]
    # CX, SX and X gate durations (somewhat sensible durations and errors chosen)
    cx_props = {
        connection: InstructionProperties(duration=t_cx * dt, error=1e-2)
        for connection in linear_topo
    }
    sx_props = {
        (i,): InstructionProperties(duration=t_sx * dt, error=1e-4) for i in range(num_qubits)
    }
    x_props = {
        (i,): InstructionProperties(duration=t_x * dt, error=1e-4) for i in range(num_qubits)
    }
    target.add_instruction(CXGate(), cx_props)
    target.add_instruction(ECRGate(), cx_props)  # re-use CX props for ECR
    target.add_instruction(CZGate(), cx_props)  # re-use CX props for CZ
    target.add_instruction(SXGate(), sx_props)
    target.add_instruction(XGate(), x_props)
    target.add_instruction(Delay(1), sx_props)  # support delays, duration does not matter here
    target.add_instruction(Reset(), x_props)  # support resets, duration does not matter here

    return target


def get_varying_target(
    num_qubits, dt=1e-9, t_cx_range=(5e2, 5e3), t_sx_range=(10, 80), t_x_range=(20, 160)
):
    """Get a toy target."""

    # set up an idealistic target to test context-aware DD in a clean setting
    # (if I don't also add realistic settings I should be scolded)
    target = Target(num_qubits=num_qubits, dt=dt)
    # bidirectional linear next neighbor
    linear_topo = [(i, i + 1) for i in range(num_qubits - 1)]
    linear_topo += [tuple(reversed(connection)) for connection in linear_topo]
    # CX, SX and X gate durations (somewhat sensible durations and errors chosen)
    cx_props = {
        connection: InstructionProperties(duration=dt * np.random.randint(*t_cx_range), error=1e-2)
        for connection in linear_topo
    }
    sx_props = {
        (i,): InstructionProperties(duration=dt * np.random.randint(*t_sx_range), error=1e-4)
        for i in range(num_qubits)
    }
    x_props = {
        (i,): InstructionProperties(duration=dt * np.random.randint(*t_x_range), error=1e-4)
        for i in range(num_qubits)
    }
    target.add_instruction(CXGate(), cx_props)
    target.add_instruction(ECRGate(), cx_props)  # re-use CX props for ECR
    target.add_instruction(CZGate(), cx_props)  # re-use CX props for CZ
    target.add_instruction(SXGate(), sx_props)
    target.add_instruction(XGate(), x_props)
    target.add_instruction(Delay(1), sx_props)  # support delays, duration does not matter here
    target.add_instruction(Reset(), x_props)  # support resets, duration does not matter here

    # get the maximal time difference of the 2q gates, this is used for verification
    cx_durations = [prop.duration for prop in cx_props.values()]
    max_cx_diff = max(cx_durations) - min(cx_durations)

    return target, max_cx_diff
