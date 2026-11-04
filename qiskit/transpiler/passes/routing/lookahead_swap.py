# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Map input circuit onto a backend topology via insertion of SWAPs."""

import collections
import copy
import logging
import math

from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.dagcircuit import DAGOpNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils

logger = logging.getLogger(__name__)


_Step = collections.namedtuple("_Step", ("state", "swaps_added", "gates_mapped", "gates_remaining"))
"""Describes one possible step in the lookahead process.

The fields are:

    state (_SystemState): The current state of the system, including its virtual-to-physical layout.
    swaps_added (list): List of qargs of swap gates introduced.
    gates_mapped (list): Gates that were mapped, including added SWAPs.
    gates_remaining (list): Gates that could not be mapped.
"""

_SystemState = collections.namedtuple(
    "_SystemState",
    ("layout", "coupling_map", "register", "swaps"),
    # The None default applies to the right-most element, i.e. `swaps`.
    defaults=(None,),
)


class LookaheadSwap(TransformationPass):
    """Map input circuit onto a backend topology via insertion of SWAPs.

    Implementation of Sven Jandura's swap mapper submission for the 2018 Qiskit
    Developer Challenge, adapted to integrate into the transpiler architecture.

    The role of the swapper pass is to modify the starting circuit to be compatible
    with the target device's topology (the set of two-qubit gates available on the
    hardware.) To do this, the pass will insert SWAP gates to relocate the virtual
    qubits for each upcoming gate onto a set of coupled physical qubits. However, as
    SWAP gates are particularly lossy, the goal is to accomplish this remapping while
    introducing the fewest possible additional SWAPs.

    This algorithm searches through the available combinations of SWAP gates by means
    of a narrowed best first/beam search, described as follows:

    - Start with a layout of virtual qubits onto physical qubits.
    - Find any gates in the input circuit which can be performed with the current
      layout and mark them as mapped.
    - For all possible SWAP gates, calculate the layout that would result from their
      application and rank them according to the distance of the resulting layout
      over upcoming gates (see _calc_layout_distance.)
    - For the four (search_width) highest-ranking SWAPs, repeat the above process on
      the layout that would be generated if they were applied.
    - Repeat this process down to a depth of four (search_depth) SWAPs away from the
      initial layout, for a total of 256 (search_width^search_depth) prospective
      layouts.
    - Choose the layout which maximizes the number of two-qubit which could be
      performed. Add its mapped gates, including the SWAPs generated, to the
      output circuit.
    - Repeat the above until all gates from the initial circuit are mapped.

    For more details on the algorithm, see Sven's blog post:
    https://medium.com/qiskit/improving-a-quantum-compiler-48410d7a7084
    """

    def __init__(self, coupling_map, search_depth=4, search_width=4, fake_run=False):
        """LookaheadSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): CouplingMap of the target backend.
            search_depth (int): lookahead tree depth when ranking best SWAP options.
            search_width (int): lookahead tree width when ranking best SWAP options.
            fake_run (bool): if true, it will only pretend to do routing, i.e., no
                swap is effectively added.
        """

        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.search_depth = search_depth
        self.search_width = search_width
        self.fake_run = fake_run

    def run(self, dag):
        """Run the LookaheadSwap pass on `dag`.

        Args:
            dag (DAGCircuit): the directed acyclic graph to be mapped
        Returns:
            DAGCircuit: A dag mapped to be compatible with the coupling_map in
                the property_set.
        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the coupling_map=None
        """

        if self.coupling_map is None:
            raise TranspilerError("LookaheadSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Lookahead swap runs on physical circuits only")

        number_of_available_qubits = len(self.coupling_map.physical_qubits)
        if len(dag.qubits) > number_of_available_qubits:
            raise TranspilerError(
                f"The number of DAG qubits ({len(dag.qubits)}) is greater than the number of "
                f"available device qubits ({number_of_available_qubits})."
            )
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )

        register = dag.qregs["q"]
        current_state = _SystemState(
            Layout.generate_trivial_layout(register), self.coupling_map, register
        )

        mapped_gates = []
        gates_remaining = list(dag.serial_layers())

        while gates_remaining:
            logger.debug("Top-level routing step: %d gates remaining.", len(gates_remaining))

            best_step = _search_forward_n_swaps(
                current_state,
                gates_remaining,
                self.search_depth,
                self.search_width,
            )

            if best_step is None:
                raise TranspilerError(
                    "Lookahead failed to find a swap which mapped gates or improved layout score."
                )

            logger.debug(
                "Found best step: mapped %d gates. Added swaps: %s.",
                len(best_step.gates_mapped),
                best_step.swaps_added,
            )

            current_state = best_step.state
            gates_mapped = best_step.gates_mapped
            gates_remaining = best_step.gates_remaining

            mapped_gates.extend(gates_mapped)

        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_state.layout
        else:
            # The "final layout" can be thought of as a "comes from" permutation that you apply at
            # the end of the circuit to invert the routing.  So if there's an existing one, what we
            # apply at the end of the circuit needs to set the circuit qubits so they "come from"
            # the previous one, then those "come from" the one we've just added.
            self.property_set["final_layout"] = self.property_set["final_layout"].compose(
                current_state.layout, dag.qubits
            )

        if self.fake_run:
            return dag

        # Preserve input DAG's name, regs, wire_map, etc. but replace the graph.
        mapped_dag = dag.copy_empty_like()
        for node in mapped_gates:
            mapped_dag.apply_operation_back(node.op, node.qargs, node.cargs, check=False)

        return mapped_dag


def _search_forward_n_swaps(state, gates, depth, width):
    """Search for SWAPs which allow for application of largest number of gates.

    Args:
        state (_SystemState): The ``namedtuple`` collection containing the state of the physical
            system.  This includes the current layout, the coupling map, the canonical register and
            the possible swaps available.
        gates (list): Gates to be mapped.
        depth (int): Number of SWAP layers to search before choosing a result.
        width (int): Number of SWAPs to consider at each layer.
    Returns:
        Optional(_Step): Describes the solution step found.  If ``None``, no swaps leading to an
        improvement were found.
    """
    if state.swaps is None:
        # Include symmetric couplings (e.g [0,1] and [1,0]) as one swap.
        state = state._replace(
            swaps={((a, b) if a < b else (b, a)) for a, b in state.coupling_map.get_edges()}
        )
    gates_mapped, gates_remaining = _map_free_gates(state, gates)
    base_step = _Step(state, [], gates_mapped, gates_remaining)

    if not gates_remaining or depth == 0:
        return base_step

    ranked_swaps = sorted(
        (_score_state_with_swap(swap, state, gates) for swap in state.swaps),
        key=lambda x: x[0],
    )
    logger.debug(
        "At depth %d, ranked candidate swaps: %s...",
        depth,
        [(swap, score) for score, swap, _ in ranked_swaps[: width * 2]],
    )

    best_swap, best_step, best_score = None, None, -math.inf
    for rank, (_, swap, new_state) in enumerate(ranked_swaps):
        next_step = _search_forward_n_swaps(new_state, gates_remaining, depth - 1, width)

        if next_step is None:
            continue

        next_score = _score_step(next_step)
        # ranked_swaps already sorted by distance, so distance is the tie-breaker.
        if next_score > best_score:
            logger.debug(
                "At depth %d, updating best step: %s (score: %f).",
                depth,
                [swap] + next_step.swaps_added,
                next_score,
            )
            best_swap, best_step, best_score = swap, next_step, next_score

        if (
            rank >= min(width, len(ranked_swaps) - 1)
            and best_step is not None
            and (
                len(best_step.gates_mapped) > depth
                or len(best_step.gates_remaining) < len(gates_remaining)
                or (
                    _calc_layout_distance(best_step.gates_remaining, best_step.state)
                    < _calc_layout_distance(gates_remaining, new_state)
                )
            )
        ):
            # Once we've examined either $WIDTH swaps, or all available swaps,
            # return the best-scoring swap provided it leads to an improvement
            # in either the number of gates mapped, number of gates left to be
            # mapped, or in the score of the ending layout.
            break
    else:
        return None

    best_swap_gate = _swap_ops_from_edge(best_swap, state)
    out = _Step(
        best_step.state,
        [best_swap] + best_step.swaps_added,
        gates_mapped + best_swap_gate + best_step.gates_mapped,
        best_step.gates_remaining,
    )
    logger.debug("At depth %d, best_swap set: %s.", depth, out.swaps_added)
    return out


def _map_free_gates(state, gates):
    """Map all gates that can be executed with the current layout.

    Args:
        state (_SystemState): The physical characteristics of the system, including its current
            layout and the coupling map.
        gates (list): Gates to be mapped.

    Returns:
        tuple:
            mapped_gates (list): ops for gates that can be executed, mapped onto layout.
            remaining_gates (list): gates that cannot be executed on the layout.
    """
    blocked_qubits = set()

    mapped_gates = []
    remaining_gates = []
    layout_map = state.layout._v2p

    for gate in gates:
        # Gates without a partition (barrier, snapshot, save, load, noise) may
        # still have associated qubits. Look for them in the qargs.
        if not gate["partition"]:
            qubits = _first_op_node(gate["graph"]).qargs

            if not qubits:
                continue

            if blocked_qubits.intersection(qubits):
                blocked_qubits.update(qubits)
                remaining_gates.append(gate)
            else:
                mapped_gate = _transform_gate_for_system(gate, state)
                mapped_gates.append(mapped_gate)
            continue

        qubits = gate["partition"][0]

        if blocked_qubits.intersection(qubits):
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)
        elif len(qubits) == 1:
            mapped_gate = _transform_gate_for_system(gate, state)
            mapped_gates.append(mapped_gate)
        elif state.coupling_map.distance(layout_map[qubits[0]], layout_map[qubits[1]]) == 1:
            mapped_gate = _transform_gate_for_system(gate, state)
            mapped_gates.append(mapped_gate)
        else:
            blocked_qubits.update(qubits)
            remaining_gates.append(gate)

    return mapped_gates, remaining_gates


def _calc_layout_distance(gates, state, max_gates=None):
    """Return the sum of the distances of two-qubit pairs in each CNOT in gates
    according to the layout and the coupling.
    """
    if max_gates is None:
        max_gates = 50 + 10 * len(state.coupling_map.physical_qubits)

    layout_map = state.layout._v2p
    out = 0
    for gate in gates[:max_gates]:
        if not gate["partition"]:
            continue
        qubits = gate["partition"][0]
        if len(qubits) == 2:
            out += state.coupling_map.distance(layout_map[qubits[0]], layout_map[qubits[1]])
    return out


def _score_state_with_swap(swap, state, gates):
    """Calculate the relative score for a given SWAP.

    Returns:
        float: the score of the given swap.
        Tuple[int, int]: the input swap that should be performed.
        _SystemState: an updated system state with the new layout contained.
    """
    trial_layout = state.layout.copy()
    trial_layout.swap(*swap)
    new_state = state._replace(layout=trial_layout)
    return _calc_layout_distance(gates, new_state), swap, new_state


def _score_step(step):
    """Count the mapped two-qubit gates, less the number of added SWAPs."""
    # Each added swap will add 3 ops to gates_mapped, so subtract 3.
    return len([g for g in step.gates_mapped if len(g.qargs) == 2]) - 3 * len(step.swaps_added)


def _transform_gate_for_system(gate, state):
    """Return op implementing a virtual gate on given layout."""
    mapped_op_node = copy.copy(_first_op_node(gate["graph"]))

    device_qreg = state.register
    layout_map = state.layout._v2p
    mapped_op_node.qargs = tuple(device_qreg[layout_map[a]] for a in mapped_op_node.qargs)

    return mapped_op_node


def _swap_ops_from_edge(edge, state):
    """Generate list of ops to implement a SWAP gate along a coupling edge."""
    device_qreg = state.register
    qreg_edge = tuple(device_qreg[i] for i in edge)

    # TODO shouldn't be making other nodes not by the DAG!!
    return [DAGOpNode(op=SwapGate(), qargs=qreg_edge, cargs=())]


def _first_op_node(dag):
    """Get the first op node from a DAG."""
    # This doesn't use `DAGCircuit.op_nodes` because that function always consumes the entire
    # iterator to create a list, whereas we only need the first element.
    return next(node for node in dag.nodes() if isinstance(node, DAGOpNode))
