# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Context-aware dynamical decoupling."""

from __future__ import annotations
from enum import Enum

import itertools
import logging

from pprint import pformat
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import rustworkx as rx

from qiskit.circuit import QuantumCircuit, Qubit, Reset, Gate
from qiskit.circuit.library import CXGate, ECRGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode, DAGInNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError

from .pad_delay import PadDelay

logger = logging.getLogger(__name__)


class ContextAwareDynamicalDecoupling(TransformationPass):
    """Implement an X-sequence dynamical decoupling considering the gate- and qubit-context.

    This pass implements a context-aware dynamical decoupling (DD) [1], which ensures that

        (1) simultaneously occurring DD sequences on device-adjacent qubits are mutually orthogonal, and
        (2) DD sequences on spectator qubits of ECR and CX gates are orthogonal to the echo
            pulses on the neighboring target/control qubits.

    The mutually orthogonal DD sequences are currently Walsh-Hadamard sequences, consisting of only
    X gates. In some cases it might therefore be beneficial to use :class:`.PadDynamicalDecoupling`
    with more generic sequences, such as XY4.

    This pass performs best if the two-qubit interactions have the same durations on the
    device, as it allows to align delay sequences and take into account potential control and target
    operations on neighboring qubits. However, it is still valid if this is not the case.

    .. note::

        If this pass is run within a pass manager (as in the example below), it will
        automatically run :class:`.PadDelay` to allocate the delays. If instead it is run as
        standalone (not inside a :class:`.PassManager`), the delays must already be inserted.


    Example::

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import QFTGate
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import ALAPScheduleAnalysis, ContextAwareDynamicalDecoupling
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke

        num_qubits = 10
        circuit = QuantumCircuit(num_qubits)
        circuit.append(QFTGate(num_qubits), circuit.qubits)
        circuit.measure_all()

        target = FakeSherbrooke().target

        pm = generate_preset_pass_manager(optimization_level=2, target=target)
        dd = PassManager([
            ALAPScheduleAnalysis(target=target),
            ContextAwareDynamicalDecoupling(target=target),
        ])

        transpiled = pm.run(circuit)
        with_dd = dd.run(transpiled)

        print(with_dd.draw(idle_wires=False))

    References:

        [1] A. Seif et al. (2024). Suppressing Correlated Noise in Quantum Computers via
            Context-Aware Compiling, `arXiv:2403.06852 <https://arxiv.org/abs/2403.06852>`_.

    """

    def __init__(
        self,
        target: Target,
        *,
        min_duration: int | None = None,
        skip_reset_qubits: bool = True,
        skip_dd_threshold: float = 1.0,
        pulse_alignment: int | None = None,
        coloring_strategy: rx.ColoringStrategy = rx.ColoringStrategy.Saturation,
    ) -> None:
        """
        Args:
            target: The :class:`.Target` of the device to run the circuit.
            min_duration: Minimal delay duration (in ``dt``) to insert a DD sequence. This
                can be useful, e.g. if a big delay block would be interrupted and split into
                smaller blocks due to a very short, adjacent delay. If ``None``, this is set
                to be at least twice the difference of the longest/shortest CX or ECR gate.
            skip_reset_qubits: Skip initial delays and delays after a reset.
            skip_dd_threshold: Skip dynamical decoupling on an idle qubit, if the duration of
                the decoupling sequence exceeds this fraction of the idle window. For example, to
                skip a DD sequence if it would take up more than 95% of the idle time, set this
                value to 0.95. A value of 1. means that the DD sequence is applied if it fits into
                the window.
            pulse_alignment: The hardware constraints (in ``dt``) for gate timing allocation.
                If provided, the inserted gates will only be executed on integer multiples of
                this value. This is usually provided on ``backend.configuration().timing_constraints``.
                If ``None``, this is extracted from the ``target``.
            coloring_strategy: The coloring strategy used for ``rx.greedy_graph_color``.
                Defaults to a saturation strategy, which is optimal on bipartite graphs,
                see Section 1.2.2.8 of [2].

        References:

            [2] A. Kosowski, and K. Manuszewski, Classical Coloring of Graphs, Graph Colorings,
                2-19, 2004. ISBN 0-8218-3458-4.

        """
        super().__init__()

        if not 0 <= skip_dd_threshold <= 1:
            raise ValueError(f"skip_dd_threshold must be in [0, 1], but is {skip_dd_threshold}")

        if min_duration is None:
            if target.dt is None:
                min_duration = 0
            else:
                min_duration = 2 * _gate_length_variance(target) / target.dt

        self._min_duration = min_duration
        self._skip_reset_qubits = skip_reset_qubits
        self._skip_dd_threshold = skip_dd_threshold
        self._target = target
        self._coupling_map = target.build_coupling_map()  # build once and reuse for performance
        self._pulse_alignment = (
            target.pulse_alignment if pulse_alignment is None else pulse_alignment
        )
        self._coloring_strategy = coloring_strategy
        self._sequence_generator = WalshHadamardSequence()

        # Use PadDelay to insert Delay operations into the DAG before running this pass.
        # This could be integrated into this pass as well, saving one iteration over the DAG,
        # but this would repeat logic and is currently no bottleneck.
        self.requires = [PadDelay(target=target)]

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # check the schedule analysis was run
        if "node_start_time" not in self.property_set:
            raise RuntimeError(
                "node_start_time not found in the pass manager's property set. "
                "Please run a scheduling pass before calling ContextAwareDynamicalDecoupling."
            )

        # 1) Find all delay instructions that qualify. We split them into explicit
        # "begin" and "end" events and sort them.
        sorted_delay_events = self._get_sorted_delays(dag)

        # 2) Identify adjacent delays by overlap in time and adjacency on device.
        qubit_map = {bit: idx for idx, bit in enumerate(dag.qubits)}  # map qubits to their indices
        adjacent_delay_blocks = self._collect_adjacent_delay_blocks(sorted_delay_events, qubit_map)
        logger.debug("adjacent delay blocks: %s", pformat(adjacent_delay_blocks))

        # 3) Split the blocks into multi-qubit delay layers
        merged_delays = []
        for block in adjacent_delay_blocks:
            merged_delays_in_block = self._split_adjacent_block(block, qubit_map)
            merged_delays += merged_delays_in_block

        # 4) For each multi-qubit delay, find a coloring (that takes into account neighboring
        # CX or ECR gates) and get the DD sequence
        all_delays = set()  # keep a list of all delays to iterate over them easily later

        for merged_delay in merged_delays:
            all_delays.update(merged_delay.ops)

            # get coloring inside the n-qubit delay
            coloring = self._get_wire_coloring(dag, merged_delay)

            # get the DD sequence on the qubit
            duration = merged_delay.end - merged_delay.start
            checked_durations_cache = set()
            for op in merged_delay.ops:
                dd_circuit, start_times = self._get_dd_sequence(
                    coloring[op.index],
                    op.index,
                    duration,
                    checked_durations_cache,
                )
                op.replacement.compose(dd_circuit, inplace=True, copy=False)
                if len(op.start_times) == 0:
                    op.start_times += start_times
                else:
                    start_times = [op.start_times[-1] + time for time in start_times]

                op.start_times += start_times

        # 5) Replace each delay operation with its individual DD sequence
        qubit_map = dict(enumerate(dag.qubits))
        for delay in all_delays:
            # replace it with its stored replacement
            as_dag = circuit_to_dag(delay.replacement)
            node_ids = [node._node_id for node in as_dag.topological_op_nodes()]
            id_map = dag.substitute_node_with_dag(
                delay.op, as_dag, {as_dag.qubits[0]: qubit_map[delay.index]}
            )

            # update node start times
            for node_id, start_time in zip(node_ids, delay.start_times):
                # Start times represent a list of times in dt times, so cast to 'int' when needed
                self.property_set["node_start_time"][id_map[node_id]] = int(start_time)

        return dag

    def get_orthogonal_sequence(self, order: int) -> tuple[list[float], list[Gate]]:
        """Return a DD sequence of given order, where different orders are orthogonal."""
        spacing = self._sequence_generator.get_sequence(order)
        return spacing, [XGate() for _ in spacing[:-1]]

    def _get_sorted_delays(
        self,
        dag: DAGCircuit,
    ) -> list[DelayEvent]:
        """Get sorted DelayEvent objects for all eligible delay operations."""

        def is_after_reset(node):
            if not self._skip_reset_qubits:
                return False  # if we do not skip reset qubits, we always want to use the delay

            predecessor = next(dag.predecessors(node))  # a delay has one 1 predecessor
            return isinstance(predecessor, DAGInNode) or isinstance(predecessor.op, Reset)

        qubit_map = {bit: index for index, bit in enumerate(dag.qubits)}

        eligible_delays = [
            DelayEvent(
                event_type,
                (
                    start_time
                    if event_type == EventType.BEGIN
                    else start_time + self._duration(node, qubit_map)
                ),
                node,
            )
            for node, start_time in self.property_set["node_start_time"].items()
            if (
                node.name == "delay"
                and not is_after_reset(node)
                and self._duration(node, qubit_map) > self._min_duration
            )
            for event_type in (EventType.BEGIN, EventType.END)
        ]

        sorted_events = sorted(eligible_delays, key=DelayEvent.sort_key)
        logger.debug("Sorted delay events: %s", sorted_events)

        return sorted_events

    def _get_wire_coloring(self, dag: DAGCircuit, merged_delay: MultiDelayOp) -> dict[int, int]:
        """Find a wire coloring for a multi-delay operation.

        This function returns a dictionary that includes the coloring (as int) for the indices in the
        ``merged_delay`` as ``{index: color}`` pairs. Spectator qubits are handled by assigning
        neighboring qubits with a CX or ECR a color (0 if control, 1 if target) and including them
        in the coloring problem.
        """
        # get neighboring wires, for which we will give initial colors
        neighbors = set()
        index_map = dict(enumerate(dag.qubits))
        qubit_map = {bit: index for index, bit in enumerate(dag.qubits)}

        for delay_op in merged_delay.ops:
            # use coupling_map.graph.neighbors_undirected once Qiskit/rustworkx#1254 is in a release
            new_neighbors = {
                i
                for i in range(dag.num_qubits())
                if self._coupling_map.distance(i, delay_op.index) == 1
            }
            # do not add indices that are already in the merged delay
            neighbors.update(new_neighbors.difference(merged_delay.indices))

        # build a subgraph we will apply the coloring function on
        wires = sorted(neighbors.union(merged_delay.indices))
        subgraph = self._coupling_map.graph.subgraph(list(wires)).to_undirected()
        glob2loc = dict(zip(wires, subgraph.node_indices()))
        preset_coloring = dict.fromkeys(subgraph.node_indices())

        # find the neighbor wires and check if ctrl/tgt spectator
        for wire in neighbors:
            for op_node in dag.nodes_on_wire(dag.qubits[wire], only_ops=True):
                # check if the operation occurs during the delay
                op_start = self.property_set["node_start_time"][op_node]
                op_end = op_start + self._duration(op_node, qubit_map)
                if (
                    isinstance(op_node.op, (CXGate, ECRGate))
                    and op_start < merged_delay.end
                    and op_end > merged_delay.start
                ):
                    # set coloring to 0 if ctrl, and to 1 if tgt
                    ctrl, tgt = op_node.qargs
                    if index_map[wire] == ctrl:
                        preset_coloring[glob2loc[wire]] = 0
                    if index_map[wire] == tgt:
                        preset_coloring[glob2loc[wire]] = 1

        local_coloring = rx.graph_greedy_color(
            subgraph,
            preset_color_fn=lambda loc: preset_coloring[loc],
            strategy=self._coloring_strategy,
        )

        # map the local indices of the subgraph back to our DAG indices
        loc2glob = dict(zip(subgraph.node_indices(), wires))
        coloring = {loc2glob[loc]: color for loc, color in local_coloring.items()}

        # for debugging purposes, print the coloring of each block
        logger.debug("Coloring for block %s: %s", merged_delay, coloring)

        return coloring

    def _get_dd_sequence(
        self,
        order: int,
        index: int,
        duration: int,
        checked_durations_cache: set[int],
    ) -> tuple[QuantumCircuit, list[int]]:
        """Get a DD sequence of specified order on qubit with a given index.

        Takes the gate durations, the pulse alignment and a set to cache, which gate lengths
        we've already checked to match the pulse alignment. Returns the DD sequence and the
        node start times as list.
        """
        instruction_durations = self._target.durations()
        # check the X gate on the active qubit is compatible with pulse alignment
        if index not in checked_durations_cache:
            x_duration = instruction_durations.get("x", index)
            if x_duration % self._pulse_alignment != 0:
                raise TranspilerError(
                    f"X gate length on qubit with index {index} is {x_duration} which is not "
                    f"an integer multiple of the pulse alignment {self._pulse_alignment}."
                )
            checked_durations_cache.add(index)

        spacing, dd_sequence = self.get_orthogonal_sequence(order=order)

        # check if DD can be applied or if there is not enough time
        dd_sequence_duration = sum(instruction_durations.get(gate, index) for gate in dd_sequence)
        slack = duration - dd_sequence_duration
        slack_fraction = slack / duration
        if 1 - slack_fraction >= self._skip_dd_threshold:  # dd doesn't fit
            seq = QuantumCircuit(1)
            seq.delay(duration, 0)
            return seq, [0]

        # compute actual spacings in between the delays, taking into account
        # the pulse alignment restriction of the hardware
        taus = self._constrain_spacing(spacing, slack)

        # apply the DD gates
        # tau has one more entry than the gate sequence
        start_times = []
        time = 0  # track the node start time
        seq = QuantumCircuit(1)  # if the DD sequence has a global phase, add it here
        for tau, gate in itertools.zip_longest(taus, dd_sequence):
            if tau > 0:
                seq.delay(tau, 0)
                start_times.append(time)
                time += tau
            if gate is not None:
                seq.append(gate, [0])
                start_times.append(time)
                time += instruction_durations.get(gate, index)

        return seq, start_times

    def _constrain_spacing(self, spacing, slack):
        def _constrained_length(values):
            return self._pulse_alignment * np.floor(values / self._pulse_alignment)

        taus = _constrained_length(slack * np.asarray(spacing))
        unused_slack = slack - sum(taus)  # unused, due to rounding to int multiples of dt
        middle_index = int((len(taus) - 1) / 2)  # arbitrary: redistribute to middle
        to_middle = _constrained_length(unused_slack)
        taus[middle_index] += to_middle  # now we add up to original delay duration
        if unused_slack - to_middle:
            taus[-1] += unused_slack - to_middle

        return taus

    def _collect_adjacent_delay_blocks(
        self,
        sorted_delay_events: list[DelayEvent],
        qubit_map: dict[Qubit, int],
    ) -> list[AdjacentDelayBlock]:
        """Collect delay events into adjacent blocks.

        Events in an adjacent block are overlapping in time and adjacent on the device.
        See also the dataclass ``AdjacentDelayBlock`` for more information.

        Args:
            sorted_delay_events: All eligible delay operations, sorted by time and type.
            qubit_map: A map from qubit instance to qubit index.

        Returns:
            A list of adjacent delay blocks.
        """
        open_delay_blocks = []
        closed_delay_blocks = []

        def _open_delay_block(delay_event):
            open_delay_blocks.append(
                AdjacentDelayBlock(
                    events=[delay_event], active_qubits=set(delay_event.op_node.qargs)
                )
            )
            return open_delay_blocks[-1]

        def _update_delay_block(open_delay_block, delay_event):
            """Add another delay event to an existing block to either extend or close it."""
            open_delay_block.events.append(delay_event)

            # at this point we know that delay_event.op_node.qargs is active
            open_delay_block.active_qubits -= set(delay_event.op_node.qargs)

            if not open_delay_block.active_qubits:
                open_delay_blocks.remove(open_delay_block)
                closed_delay_blocks.append(open_delay_block)

        def _combine_delay_blocks(delay_blocks):
            survivor, *doomed = delay_blocks

            for doomed_delay_group in doomed:
                # Add events and qubits from doomed block to survivor.
                if logger.isEnabledFor(logging.DEBUG):
                    if survivor.active_qubits.intersection(doomed_delay_group.active_qubits):
                        logger.debug("More than one open delay on a qubit?")

                survivor.events.extend(doomed_delay_group.events)
                survivor.active_qubits.update(doomed_delay_group.active_qubits)

                open_delay_blocks.remove(doomed_delay_group)
                survivor.events.sort(key=DelayEvent.sort_key)  # Maintain sorted event order

        for delay_event in sorted_delay_events:
            # This could be avoided by keeping a map of device qubit to open
            # block and only considering neighbors of current event.
            # use coupling_map.graph.neighbors_undirected once Qiskit/rustworkx#1254 is in a release
            adjacent_open_delay_blocks = [
                open_delay
                for open_delay in open_delay_blocks
                if any(
                    self._coupling_map.distance(
                        qubit_map[delay_event.op_node.qargs[0]], qubit_map[open_delay_qubit]
                    )
                    <= 1
                    for open_delay_qubit in open_delay.active_qubits
                )
            ]

            if delay_event.type == EventType.BEGIN:
                # If crossing a begin edge, check if there are any open delays that are adjacent.
                # If so, add current event to that group.

                if len(adjacent_open_delay_blocks) == 0:
                    # Make a new delay block
                    _open_delay_block(delay_event)
                else:
                    # Make a new block and combine that with adjacent open blocks
                    new_block = _open_delay_block(delay_event)
                    _combine_delay_blocks(adjacent_open_delay_blocks + [new_block])

            else:
                if logger.isEnabledFor(logging.DEBUG):
                    # If crossing a end edge, remove this qubit from the actively delaying qubits"
                    if len(adjacent_open_delay_blocks) != 1:
                        logger.debug("Closing edge w/o an open delay?")

                _update_delay_block(adjacent_open_delay_blocks[0], delay_event)

        # log the open delays and the number of closed delays
        logger.debug("-- post collect")
        logger.debug("open delays: %s", open_delay_blocks)
        logger.debug("len(closed delays): %s", len(closed_delay_blocks))

        # validate the results, there should be no open delays and all active qubits
        # should be accounted for
        if logger.isEnabledFor(logging.DEBUG):
            if len(open_delay_blocks) > 0:
                logger.debug("Failed to close all open delays.")

            for closed_delay in closed_delay_blocks:
                if len(closed_delay.active_qubits) > 0:
                    logger.debug("Failed to remove active qubits on closed delay %s.", closed_delay)

                closed_delay.validate()

        return closed_delay_blocks

    def _split_adjacent_block(
        self,
        adjacent_block: AdjacentDelayBlock,
        qubit_map: dict[Qubit, int],
    ) -> list[MultiDelayOp]:
        """Split adjacent delay blocks in concurrent layers of maximum width.

        This code performs the following steps:

            1) Find the atomic time windows where delays can begin and end.
            2) Find which connected qubit components are jointly idle during the windows.
            3) Merge time-adjacent qubit components.

        """
        # 1) Find times at which a new delay event is happening and we might have to break up
        # the adjacent delays
        breakpoints = []
        # a dictionary of {qubit_index: [delay_op1, delay_op2, ...]}
        all_delays = defaultdict(list)

        for event in adjacent_block.events:
            index = qubit_map[event.op_node.qargs[0]]  # delays are 1q ops
            if event.type == EventType.BEGIN:
                delay = DelayOp(start=event.time, index=index, op=event.op_node)
                all_delays[index].append(delay)
            else:  # find which delay to close
                all_delays[index][-1].end = event.time

            if len(breakpoints) == 0 or event.time > breakpoints[-1]:
                breakpoints.append(event.time)

        # 2) Find which delays are active during which time windows, where a time window
        # is defined as (breakpoints[i], breakpoints[i + 1])
        active_delays = {}  # {window: [group1, group2]} where each group = (index1, index2, ..)
        op_map = {}  # {(qubit_index, window): delay operation as DAGOpNode}
        windows = list(zip(breakpoints[:-1], breakpoints[1:]))
        for window in windows:
            active = []

            # check which delays are active during the window
            # this could be e.g. [0, 1, 2, 5, 6, 9]
            for index, delays in all_delays.items():
                for delay in delays:
                    # since the windows are atomic, we have three cases:
                    # (1) the delay starts at the time window, (2) it ends with it, (3) it contains it
                    if delay.start <= window[0] and delay.end >= window[1]:
                        active.append(index)
                        delay.add_window(window)
                        op_map[(index, window)] = delay

            # check which are adjacent
            # on a linear topology, we would get [[0, 1, 2], [5, 6], [9]]
            visited = defaultdict(lambda: False)
            grouped = []
            for start_index in active:
                if visited[start_index]:
                    continue

                active_neighbors = {start_index}
                _dfs(start_index, self._coupling_map, active_neighbors, active)

                for index in active_neighbors:
                    visited[index] = True

                group = tuple(sorted(active_neighbors))  # must be sorted to merge later
                grouped.append(group)

            # sanity check: groups must be disjoint
            if logger.isEnabledFor(logging.DEBUG):
                for i, g1 in enumerate(grouped):
                    for g2 in grouped[i + 1 :]:
                        if len(set(g1).intersection(g2)) > 0:
                            logger.debug("Groups not disjoint: %s and %s.", g1, g2)

            active_delays[window] = grouped

        # 3) Merge time-adjacent active delays
        merged_delays = []
        open_groups = {}
        for window in windows:
            next_groups = active_delays[window]

            # check which opened groups are still active in this window
            next_open_groups = {}
            for open_group, delay in open_groups.items():
                if open_group in next_groups:
                    # extend the delay operation and remove a breakpoint
                    delay.end = window[1]
                    for op in delay.ops:
                        op.breakpoints.remove(window[0])

                    if logger.isEnabledFor(logging.DEBUG):
                        involved_ops = {op_map[(index, window)] for index in open_group}
                        if len(involved_ops.difference(delay.ops)) > 0:
                            logger.debug("Not all involved operations are part of the joint delay.")

                    next_open_groups[open_group] = delay
                else:
                    # the group has not been extended, close it!
                    merged_delays.append(delay)

            # add the new open groups
            for group in next_groups:
                if group not in open_groups:
                    involved_ops = {op_map[(index, window)] for index in group}
                    next_open_groups[group] = MultiDelayOp(
                        start=window[0], end=window[1], indices=group, ops=involved_ops
                    )

            open_groups = next_open_groups

        # add all from the final layer
        for open_group, delay in open_groups.items():
            merged_delays.append(delay)

        logger.debug("Split adjacent delay block into %s", "\n".join(map(str, merged_delays)))

        return merged_delays

    def _duration(self, node: DAGOpNode, qubit_map: dict[Qubit, int]) -> float:
        # this is cached on the target, so we can repeatedly call it w/o penalty
        instruction_durations = self._target.durations()
        indices = [qubit_map[bit] for bit in node.qargs]

        return instruction_durations.get(node.op, indices)


class EventType(Enum):
    """Delay event type, which is either begin or end."""

    BEGIN = 0
    END = 1


@dataclass
class DelayEvent:
    """Represent a single-qubit delay event, which is either begin or end of a delay instruction."""

    type: EventType
    time: int  # Time in this circuit, in dt
    op_node: DAGOpNode  # The node for the circuit delay

    @staticmethod
    def sort_key(event: DelayEvent) -> tuple(int, int):
        """Sort events, first by time then by type ('end' events come before 'begin' events)."""
        return (
            event.time,  # Sort by event time
            0 if event.type == EventType.END else 1,  # With 'end' events before 'begin'
        )


@dataclass
class DelayOp:
    """Represent a delay operation."""

    start: int
    index: int
    op: DAGOpNode  # the circuit op node this represents
    end: int | None = None  # None means currently unknown
    breakpoints: list[int] = field(
        default_factory=list
    )  # timepoints at which the delay op is split
    # circuit with which we will replace this delay
    start_times: list[int] = field(default_factory=list)
    replacement: QuantumCircuit = field(default_factory=lambda: QuantumCircuit(1))

    def __hash__(self):
        return hash(self.op)

    def add_window(self, window: tuple[int, int]):
        """Add a time window to the delay op.

        This means the delay is active during this window and we add a potential breakpoint.
        """
        if self.end is None:
            raise ValueError("Cannot add a window if DelayOp.end is None. Please set it.")

        start, end = window
        if self.start < start and start not in self.breakpoints:
            self.breakpoints.append(start)
        if self.end > end and end not in self.breakpoints:
            self.breakpoints.append(end)

        self.breakpoints = sorted(self.breakpoints)


@dataclass
class MultiDelayOp:
    """A multi-qubit delay operation."""

    start: int
    end: int
    indices: list[int]
    ops: set[DelayOp]
    replacing: set[DAGOpNode] = field(default_factory=set)

    def __str__(self) -> str:
        return f"MultiDelay({self.start}:{self.end} on {self.indices})"


@dataclass
class AdjacentDelayBlock:
    """Group of circuit delays which are collectively adjacent in time and on device.

    For example, here the 3 delay operations on q0, q1 and q2 form an adjacent delay block.

        q0: -██████---------  |  qubits q0,q1,q2 have adjacent delay
        q1: ------███████---  |  operations, since the delay operations
        q2: --█████████-----  |  all overlap
        q3: -----------████-  -> this delay starts when delay on q2 ends, so they have no overlap
        q4: ----████--------  -> this clearly has no overlap with something else

    """

    events: list[DelayEvent]
    active_qubits: set[Qubit]

    def validate(self, log: bool = True) -> None:
        """Validate the list of delay events in the adjacent block.

        Args:
            log: If ``True`` log invalid blocks on DEBUG level. Otherwise raise an error if the
                block is invalid.

        Raises:
            RuntimeError: If the blocks are not ordered by time and event type.
        """

        def notify(msg, *args):
            if log:
                logger.debug(msg, *args)
            else:
                raise RuntimeError(msg.format(*args))

        for idx, event in enumerate(self.events[:-1]):
            if event.time > self.events[idx + 1].time:
                notify("adjacent_delay_block.events not ordered by time")

            if event.time == self.events[idx + 1].time:
                # At same time, can either be ('begin', 'begin'), ('end', 'begin') or ('end', 'end')
                if (event.type, self.events[idx + 1].type) == (EventType.BEGIN, EventType.END):
                    notify(
                        "Events in the AdjacentDelayBlock are not correctly sorted by "
                        "event type. At same time, we can have either of (begin, begin), "
                        "(end, begin) or (end, end). This happened at time %s.",
                        event.time,
                    )


def _dfs(qubit, cmap: CouplingMap, visited, active_qubits):
    """Depth-first search to get the widest group of idle qubits during a given time frame."""
    # use coupling_map.graph.neighbors_undirected once Qiskit/rustworkx#1254 is in a release
    neighbors = {other for other in active_qubits if cmap.distance(qubit, other) == 1}
    for neighbor in neighbors:
        if neighbor in active_qubits and neighbor not in visited:
            visited.add(neighbor)
            _dfs(neighbor, cmap, visited, active_qubits)


class WalshHadamardSequence:
    """Get Walsh-Hadamard sequences for DD up to arbitrary order."""

    def __init__(self, max_order: int = 5):
        """
        Args:
            max_order: The maximal order for which the sequences are computed.
        """
        # these are set in set_max_order
        self.sequences = None
        self.max_order = None

        self.set_max_order(max_order)

    def set_max_order(self, max_order: int) -> None:
        """Set the maximal available order."""
        if self.max_order is not None:
            if max_order <= self.max_order:
                return

        # get the dimension of the transformation matrix we need,
        # this is given by the smallest power of 2 that includes max_order
        num_krons = int(np.ceil(np.log2(max_order + 1)))
        self.max_order = 2**num_krons - 1

        rows = _get_transformation_matrix(num_krons).tolist()
        distances = [_bitflips_to_timings(row) for row in rows]
        num_flips = [len(distance) - 1 for distance in distances]

        # sort by the number of flips and throw out the first one,
        # which corresponds to the 000... bit-sequence, i.e., no flip
        indices = np.argsort(num_flips)[1:]

        self.sequences = [distances[i] for i in indices]

    def get_sequence(self, order: int) -> list[float]:
        """Get the Walsh-Hadamard sequence of given order (starts at 0)."""
        if order > self.max_order:
            self.set_max_order(order)

        return self.sequences[order]


def _get_transformation_matrix(n):
    """Get a 2^n x 2^n Walsh-Hadamard matrix with elements in [0, 1]."""

    from qiskit.circuit.library import HGate

    had = np.array(HGate()).real * np.sqrt(2)

    def recurse(matrix, m):
        # we build the matrix recursively, adding one Hadamard kronecker product per recursion
        if m == 1:
            # finally, map to [0, 1] by the mapping (1 - H) / 2
            return ((1 - matrix) / 2).astype(int)

        return recurse(np.kron(had, matrix), m - 1)

    return recurse(had, n)


def _bitflips_to_timings(row):
    num = len(row)
    distances = []
    count = 0
    last = 0  # start in no flip state
    for el in row:
        if el == last:
            count += 1
        else:
            distances.append(count / num)
            last = el
            count = 1

    distances.append(count / num)

    if len(distances) % 2 == 0:
        return distances + [0]

    return distances


def _gate_length_variance(target: Target) -> float:
    max_length, min_length = None, None

    for gate, properties in target.items():
        if gate not in ["cx", "cz", "ecr"]:
            continue

        for prop in properties.values():
            duration = prop.duration
            if max_length is None or max_length < duration:
                max_length = duration
            if min_length is None or min_length > duration:
                min_length = duration

    # it could be that there are no 2q gates available, in this
    # case we just return 0, which will mean we join all idle times
    if max_length is None or min_length is None:
        return 0

    return max_length - min_length
