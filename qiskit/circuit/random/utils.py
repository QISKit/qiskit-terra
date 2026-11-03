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

"""Utility functions for generating random circuits."""
import numpy as np
from rustworkx import PyDiGraph

from qiskit.circuit import (
    ClassicalRegister,
    QuantumCircuit,
    CircuitInstruction,
)
from qiskit.circuit import Reset
from qiskit.circuit.library import standard_gates
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _BASIS_1Q, _BASIS_2Q

# (Gate class, number of qubits, number of parameters)
gates_1q_data = [
    (standard_gates.IGate, 1, 0),
    (standard_gates.SXGate, 1, 0),
    (standard_gates.XGate, 1, 0),
    (standard_gates.RZGate, 1, 1),
    (standard_gates.RGate, 1, 2),
    (standard_gates.HGate, 1, 0),
    (standard_gates.PhaseGate, 1, 1),
    (standard_gates.RXGate, 1, 1),
    (standard_gates.RYGate, 1, 1),
    (standard_gates.SGate, 1, 0),
    (standard_gates.SdgGate, 1, 0),
    (standard_gates.SXdgGate, 1, 0),
    (standard_gates.TGate, 1, 0),
    (standard_gates.TdgGate, 1, 0),
    (standard_gates.UGate, 1, 3),
    (standard_gates.U1Gate, 1, 1),
    (standard_gates.U2Gate, 1, 2),
    (standard_gates.U3Gate, 1, 3),
    (standard_gates.YGate, 1, 0),
    (standard_gates.ZGate, 1, 0),
]

gates_2q_data = [
    (standard_gates.CXGate, 2, 0),
    (standard_gates.DCXGate, 2, 0),
    (standard_gates.CHGate, 2, 0),
    (standard_gates.CPhaseGate, 2, 1),
    (standard_gates.CRXGate, 2, 1),
    (standard_gates.CRYGate, 2, 1),
    (standard_gates.CRZGate, 2, 1),
    (standard_gates.CSXGate, 2, 0),
    (standard_gates.CUGate, 2, 4),
    (standard_gates.CU1Gate, 2, 1),
    (standard_gates.CU3Gate, 2, 3),
    (standard_gates.CYGate, 2, 0),
    (standard_gates.CZGate, 2, 0),
    (standard_gates.RXXGate, 2, 1),
    (standard_gates.RYYGate, 2, 1),
    (standard_gates.RZZGate, 2, 1),
    (standard_gates.RZXGate, 2, 1),
    (standard_gates.XXMinusYYGate, 2, 2),
    (standard_gates.XXPlusYYGate, 2, 2),
    (standard_gates.ECRGate, 2, 0),
    (standard_gates.CSGate, 2, 0),
    (standard_gates.CSdgGate, 2, 0),
    (standard_gates.SwapGate, 2, 0),
    (standard_gates.iSwapGate, 2, 0),
]


def random_circuit_from_graph(
    interaction_graph,
    min_2q_gate_per_edge=1,
    max_operands=2,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
    insert_1q_oper=True,
    prob_conditional=0.1,
    prob_reset=0.1,
):
    """Generate random circuit of arbitrary size and form which strictly respects the interaction
    graph passed as argument. Interaction Graph is a graph G=(V, E) where V are the qubits in the
    circuit, and, E is the set of two-qubit gate interactions between two particular qubits in the
    circuit.

    This function will generate a random circuit by randomly selecting 1Q and 2Q gates from the set
    of standard gates in :mod:`qiskit.circuit.library.standard_gates`. The user can attach a numerical
    value as a metadata to the edge of the graph indicating the edge weight for that particular edge,
    These edge weights would be normalized to the probabilities of the edges getting selected.
    If all the edge weights are passed as `None`, then the probability of each qubit-pair of getting
    selected is set to 1/N, where `N` is the number of edges in the `interaction_graph` passed in,
    i.e each edge is drawn uniformly. If any weight of an edge is set as zero, that particular
    edge will not be included in the output circuit.

    Passing a list of tuples of control qubit, target qubit and associated probability is also
    acceptable as a way to specify an interaction graph.

    If numerical values are present as probabilities but some/any are None, or negative, when
    `max_operands=2` this will raise a ValueError.

    If `max_operands` is set to 1, then there are no 2Q operations, so no need to take care
    of the edges, in such case the function will return a circuit from the `random_circuit` function,
    which would be passed with the `max_operands` as 1.

    If `max_operands` is set to 2, then in every while-iteration 2Q gates are chosen randomly, and
    qubit-pairs which exist in the input interaction graph are also chosen randomly based on the
    probability attached to the qubit-pair. Then in a for-iteration 2Q gates are applied to the
    randomly chosen qubit-pairs with the aim to reach the count of 2Q on any qubit-pair to
    `min_2q_gate_per_edge` criteria as soon as possible within a particular iteration. Now, if
    some qubits are still idle after applying 2Q gates for that particular iteration, then randomly
    chosen 1Q gates are applied to those idle qubits if `insert_1q_oper` is set to True.

    Example:

    .. plot::
       :alt: Pass in interaction graph and minimum 2Q gate per edge as a bare minimum.
       :include-source:

       from qiskit.circuit.random.utils import random_circuit_from_graph
       import rustworkx as rx
       pydi_graph = rx.PyDiGraph()
       pydi_graph.add_nodes_from(range(7))
       cp_map = [(0, 1, 0.18), (1, 2, 0.15), (2, 3, 0.15), (3, 4, 0.22), (4, 5, 0.13), (5, 6, 0.17)]
       pydi_graph.add_edges_from(cp_map)
       # cp_map can be passed in directly as interaction_graph
       qc = random_circuit_from_graph(pydi_graph, min_2q_gate_per_edge=2, seed=12345)
       qc.draw(output='mpl')

    Args:
        interaction_graph (PyGraph | PyDiGraph | List[Tuple[int, int, float]]): Interaction Graph
        min_2q_gate_per_edge (int): Minimum number of times every qubit-pair must be used
            in the random circuit. (optional, default:1)
        max_operands (int): maximum qubit operands of each gate(should be 1 or 2)
            (optional, default:2)
        measure (bool): if True, measure all qubits at the end. (optional, default: False)
        conditional (bool): if True, insert middle measurements and conditionals.
            (optional, default: False)
        reset (bool): if True, insert middle resets. (optional, default: False)
        seed (int): sets random seed. (If `None`, a random seed is chosen) (optional)
        insert_1q_oper (bool): Insert 1Q operations to the circuit. (optional, default: True)
        prob_conditional (float): Probability less than 1.0, this is used to control the occurrence
            of conditionals in the circuit. (optional, default: 0.1)
        prob_reset (float): Probability less than 1.0, this is used to control the occurrence of
            reset in the circuit. (optional, default: 0.1)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: When `max_operands` is not 1 or 2.
        CircuitError: When `max_operands` is set to 1, but no 1Q operations are allowed by setting
            `insert_1q_oper` to false.
        CircuitError: When the interaction graph has no edges, so only 1Q gates are possible in
            the circuit, but `insert_1q_oper` is set to False.
        CircuitError: When an invalid interaction graph object is passed.
        ValueError: Given `max_operands=2`, when any edge have probability `None` but not all, or any
            of the probabilities are negative.
    """

    # max_operands should be 1 or 2
    if max_operands not in {1, 2}:
        raise CircuitError("`max_operands` should be either 1 or 2")

    if max_operands == 1 and not insert_1q_oper:
        raise CircuitError(
            "`max_operands` of 1 means only 1Q gates are allowed, but `insert_1q_oper` is False"
        )

    # Declaring variables so lint doesn't complaint.
    num_qubits = 0
    num_edges = None
    edge_list = None
    edges_probs = None

    if isinstance(interaction_graph, list):
        num_edges = len(interaction_graph)
        edge_list = []
        edges_probs = []
        for ctrl, trgt, prob in interaction_graph:
            edge_list.append((ctrl, trgt))
            edges_probs.append(prob)

            num_qubits = max(num_qubits, ctrl)

            num_qubits = max(num_qubits, trgt)

        num_qubits += 1  # ctrl, trgt are qubit indices.
        edge_list = np.array(edge_list)
        edges_probs = np.array(edges_probs)
    elif isinstance(interaction_graph, PyDiGraph):
        num_qubits = interaction_graph.num_nodes()
        num_edges = interaction_graph.num_edges()
        edge_list = np.array(interaction_graph.edge_list())
        edges_probs = np.array(interaction_graph.edges())
    else:
        raise CircuitError("Invalid interaction graph object has been passed")

    if num_qubits == 0:
        return QuantumCircuit()

    max_operands = max_operands if num_qubits > max_operands else num_qubits

    if num_edges == 0 and not insert_1q_oper:
        raise CircuitError(
            "There are no edges in the `interaction_graph` so, there could be only 1Q gates, "
            "however `insert_1q_oper` is set to `False`"
        )

    if num_edges == 0 or max_operands == 1:
        # If there is no edge then there could be no 2Q operation
        # or, if only 1Q operations are allowed, then there is no
        # point in considering edges.
        return random_circuit(
            num_qubits=num_qubits,
            depth=min_2q_gate_per_edge,
            max_operands=1,
            measure=measure,
            conditional=conditional,
            reset=reset,
            seed=seed,
        )

    # If any edge weight is zero, just remove that edge from the edge_list
    if 0 in edges_probs:
        _mask = edges_probs != 0
        edges_probs = edges_probs[_mask]
        edge_list = edge_list[_mask]

        # Update 'num_edges'
        num_edges = len(edge_list)

    # Now, zeros are filtered out in above if-block.
    # Now, If none of the edges_probs are `None`
    if all(edges_probs):

        # edge weights in interaction_graph must be positive
        for prob in edges_probs:
            if prob < 0:
                raise ValueError("Probabilities cannot be negative")

        # Normalize edge weights if not already normalized.
        probs_sum = edges_probs.sum()
        if not np.isclose(probs_sum, 1.000, rtol=0.001):
            edges_probs = edges_probs / probs_sum
    # If any of the values of the probability is None, then it would raise an error.
    elif any(edges_probs):
        raise ValueError(
            "Some of the probabilities of a qubit-pair getting selected is `None`"
            " It should either be all `None` or all positive numerical weights. "
        )

    # If all edge weights are none, assume the weight of each edge to be 1/N.
    elif None in edges_probs:
        edges_probs = np.ones(num_edges) / num_edges

    gates_2q = np.array(
        gates_2q_data,
        dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)],
    )

    if insert_1q_oper:
        gates_1q = np.array(gates_1q_data, dtype=gates_2q.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    edges_used = {tuple(edge): 0 for edge in edge_list}

    # Declaring variables, so that lint doesn't complaint.
    reset_2q = None
    cond_val_2q = None
    cond_val_1q = None

    # If conditional is not required, there is no need to calculate random numbers.
    # But, since the variables are required in the for-loop so let's get an array of false.
    if not conditional:
        cond_1q = np.zeros(num_qubits, dtype=bool)
        cond_2q = np.zeros(num_edges, dtype=bool)

    # Similarly, if resets are not required, then since, the variable is required
    # in the for-loop, let's get an array of false.
    if not reset:
        reset_2q = np.zeros(num_edges, dtype=bool)

    # This loop will keep on applying gates to qubits until every qubit-pair
    # has 2Q operations applied at-least `min_2q_gate_per_edge` times.
    while edges_used:

        # For any given while-iteration, this is a set of qubits not having any 2Q gates.
        # Qubits in this set will have 1Q gates, if `insert_1q_oper` is True.
        qubit_idx_idle = set(range(num_qubits))

        # normalized edge weights represent the probability with which each qubit-pair
        # is inserted into the circuit.
        edge_choices = rng.choice(
            edge_list,
            size=num_edges,
            replace=True,
            p=edges_probs,
        )
        gate_choices = rng.choice(gates_2q, size=num_edges, replace=True)

        cumsum_params = np.cumsum(gate_choices["num_params"], dtype=np.int64)
        parameters = rng.uniform(0, 2 * np.pi, size=cumsum_params[-1])

        # If reset is required, then, generating a random boolean matrix of
        # num_edges x 2, corresponding to probable reset on both control and
        # target qubits of the edge from the edge_list.
        if reset:
            reset_2q = rng.random(size=(num_edges, 2)) < prob_reset

        if conditional:
            cond_2q = rng.random(size=len(gate_choices)) < prob_conditional
            cond_val_2q = rng.integers(0, 1 << min(num_qubits, 63), size=np.count_nonzero(cond_2q))
            clbit_2q_idx = 0

        for gate, num_gate_params, edge, is_cond_2q, is_rst in zip(
            gate_choices["class"],
            gate_choices["num_params"],
            edge_choices,
            cond_2q,
            reset_2q,
        ):

            control_qubit, target_qubit = tuple(edge)

            # For every edge there are two probabilistically generated boolean values corresponding
            # to control, target qubits of the edge
            # an idle qubit for a particular iteration on which reset is applied is considered idle.

            if reset:
                is_rst_control, is_rst_target = is_rst
                rst_oper = Reset()
                if is_rst_control:
                    qc._append(
                        CircuitInstruction(
                            operation=rst_oper,
                            qubits=[qubits[control_qubit]],
                        )
                    )

                if is_rst_target:
                    qc._append(
                        CircuitInstruction(
                            operation=rst_oper,
                            qubits=[qubits[target_qubit]],
                        )
                    )

            params = parameters[:num_gate_params]
            parameters = parameters[num_gate_params:]
            current_instr = gate(*params)

            if is_cond_2q:
                qc.measure(qc.qubits, cr)
                # The condition values are required to be bigints, not Numpy's fixed-width type.
                with qc.if_test((cr, int(cond_val_2q[clbit_2q_idx]))):
                    clbit_2q_idx += 1
                    qc.append(
                        CircuitInstruction(
                            operation=current_instr,
                            qubits=[qubits[control_qubit], qubits[target_qubit]],
                        )
                    )
            else:
                qc._append(
                    CircuitInstruction(
                        operation=current_instr,
                        qubits=[qubits[control_qubit], qubits[target_qubit]],
                    )
                )

            # Removing the qubits that have been applied with 2Q gates from the
            # set of idle qubits for that while-iteration.
            qubit_idx_idle = qubit_idx_idle - set(edge)

            # Update the number of occurrences of the edge in the circuit.
            _temp_edge = (control_qubit, target_qubit)
            if _temp_edge in edges_used:
                edges_used[_temp_edge] += 1
                if edges_used[_temp_edge] >= min_2q_gate_per_edge:
                    del edges_used[_temp_edge]

        if insert_1q_oper:
            num_unused_qubits = len(qubit_idx_idle)
            if not num_unused_qubits == 0:

                # Calculating for conditionals to make even the 1Q operations
                # probabilistically conditional.
                if conditional:
                    cond_1q = rng.random(size=num_unused_qubits) < prob_conditional
                    cond_val_1q = rng.integers(
                        0, 1 << min(num_qubits, 63), size=np.count_nonzero(cond_1q)
                    )
                    clbit_1q_idx = 0

                # Some extra 1Q Gate in to fill qubits which are still idle for this
                # particular while iteration.
                extra_1q_gates = rng.choice(gates_1q, size=num_unused_qubits, replace=True)

                cumsum_params = np.cumsum(extra_1q_gates["num_params"], dtype=np.int64)
                parameters_1q = rng.uniform(0, 2 * np.pi, size=cumsum_params[-1])

                for gate_1q, num_gate_params, qubit_idx, is_cond_1q in zip(
                    extra_1q_gates["class"],
                    extra_1q_gates["num_params"],
                    qubit_idx_idle,
                    cond_1q,
                ):
                    params_1q = parameters_1q[:num_gate_params]
                    parameters_1q = parameters_1q[num_gate_params:]
                    current_instr = gate_1q(*params_1q)

                    if is_cond_1q:
                        qc.measure(qc.qubits, cr)
                        # The condition values are required to be bigints, not Numpy's fixed-width type.
                        with qc.if_test((cr, int(cond_val_1q[clbit_1q_idx]))):
                            clbit_1q_idx += 1
                            qc.append(
                                CircuitInstruction(
                                    operation=current_instr,
                                    qubits=[qubits[qubit_idx]],
                                )
                            )
                    else:
                        qc._append(
                            CircuitInstruction(
                                operation=current_instr,
                                qubits=[qubits[qubit_idx]],
                            )
                        )

    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def random_circuit(
    num_qubits,
    depth,
    max_operands=4,
    measure=False,
    conditional=False,
    reset=False,
    seed=None,
    num_operand_distribution: dict | None = None,
):
    """Generate random circuit of arbitrary size and form.

    This function will generate a random circuit by randomly selecting gates
    from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.random import random_circuit

       circ = random_circuit(2, 2, measure=True)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires
        depth (int): layers of operations (i.e. critical path length).
            Note -  that due to the stochastic nature of the method, some of the qubits
            in some layers may end up not being assigned to any gate, leading to a
            circuit with a shorter critical path than specified by `depth`.
        max_operands (int): maximum qubit operands of each gate (between 1 and 4)
        measure (bool): if True, measure all qubits at the end
        conditional (bool): if True, insert middle measurements and conditionals
        reset (bool): if True, insert middle resets
        seed (int): sets random seed (optional)
        num_operand_distribution (dict): a distribution of gates that specifies the ratio
            of 1-qubit, 2-qubit, 3-qubit, ..., n-qubit gates in the random circuit. Expect a
            deviation from the specified ratios that depends on the size of the requested
            random circuit. (optional)

    Returns:
        QuantumCircuit: constructed circuit

    Raises:
        CircuitError: when invalid options given
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(seed)

    if num_operand_distribution:
        if min(num_operand_distribution.keys()) < 1 or max(num_operand_distribution.keys()) > 4:
            raise CircuitError("'num_operand_distribution' must have keys between 1 and 4")
        for key, prob in num_operand_distribution.items():
            if key > num_qubits and prob != 0.0:
                raise CircuitError(
                    f"'num_operand_distribution' cannot have {key}-qubit gates"
                    f" for circuit with {num_qubits} qubits"
                )
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    if not num_operand_distribution and max_operands:
        if max_operands < 1 or max_operands > 4:
            raise CircuitError("max_operands must be between 1 and 4")
        max_operands = max_operands if num_qubits > max_operands else num_qubits
        rand_dist = rng.dirichlet(
            np.ones(max_operands)
        )  # This will create a random distribution that sums to 1
        num_operand_distribution = {i + 1: rand_dist[i] for i in range(max_operands)}
        num_operand_distribution = dict(sorted(num_operand_distribution.items()))

    # Here we will use np.isclose() because very rarely there might be floating
    # point precision errors
    if not np.isclose(sum(num_operand_distribution.values()), 1):
        raise CircuitError("The sum of all the values in 'num_operand_distribution' is not 1.")

    if num_qubits == 0:
        return QuantumCircuit()

    gates_3q = [
        (standard_gates.CCXGate, 3, 0),
        (standard_gates.CSwapGate, 3, 0),
        (standard_gates.CCZGate, 3, 0),
        (standard_gates.RCCXGate, 3, 0),
    ]

    gates_4q = [
        (standard_gates.C3SXGate, 4, 0),
        (standard_gates.RC3XGate, 4, 0),
    ]

    gates_1q = np.array(
        gates_1q_data + [(Reset, 1, 0)] if reset else gates_1q_data,
        dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)],
    )

    gates_2q = np.array(gates_2q_data, dtype=gates_1q.dtype)
    gates_3q = np.array(gates_3q, dtype=gates_1q.dtype)
    gates_4q = np.array(gates_4q, dtype=gates_1q.dtype)

    all_gate_lists = [gates_1q, gates_2q, gates_3q, gates_4q]

    # Here we will create a list 'gates_to_consider' that will have a
    # subset of different n-qubit gates and will also create a list for
    # ratio (or probability) for each gates
    gates_to_consider = []
    distribution = []
    for n_qubits, ratio in num_operand_distribution.items():
        gate_list = all_gate_lists[n_qubits - 1]
        gates_to_consider.extend(gate_list)
        distribution.extend([ratio / len(gate_list)] * len(gate_list))

    gates = np.array(gates_to_consider, dtype=gates_1q.dtype)

    qc = QuantumCircuit(num_qubits)

    if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

    qubits = np.array(qc.qubits, dtype=object, copy=True)

    # Counter to keep track of number of different gate types
    counter = np.zeros(len(all_gate_lists) + 1, dtype=np.int64)
    total_gates = 0

    # Apply arbitrary random operations in layers across all qubits.
    for layer_number in range(depth):
        # We generate all the randomness for the layer in one go, to avoid many separate calls to
        # the randomization routines, which can be fairly slow.

        # This reliably draws too much randomness, but it's less expensive than looping over more
        # calls to the rng. After, trim it down by finding the point when we've used all the qubits.

        # Due to the stochastic nature of generating a random circuit, the resulting ratios
        # may not precisely match the specified values from `num_operand_distribution`. Expect
        # greater deviations from the target ratios in quantum circuits with fewer qubits and
        # shallower depths, and smaller deviations in larger and deeper quantum circuits.
        # For more information on how the distribution changes with number of qubits and depth
        # refer to the pull request #12483 on Qiskit GitHub.

        gate_specs = rng.choice(gates, size=len(qubits), p=distribution)
        cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)

        # Efficiently find the point in the list where the total gates would use as many as
        # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
        # it with 1q gates.
        max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
        gate_specs = gate_specs[:max_index]

        slack = num_qubits - cumulative_qubits[max_index - 1]

        # Updating the counter for 1-qubit, 2-qubit, 3-qubit and 4-qubit gates
        gate_qubits = gate_specs["num_qubits"]
        counter += np.bincount(gate_qubits, minlength=len(all_gate_lists) + 1)

        total_gates += len(gate_specs)

        # Slack handling loop, this loop will add gates to fill
        # the slack while respecting the 'num_operand_distribution'
        while slack > 0:
            gate_added_flag = False

            for key, dist in sorted(num_operand_distribution.items(), reverse=True):
                if slack >= key and counter[key] / total_gates < dist:
                    gate_to_add = np.array(
                        all_gate_lists[key - 1][rng.integers(0, len(all_gate_lists[key - 1]))]
                    )
                    gate_specs = np.hstack((gate_specs, gate_to_add))
                    counter[key] += 1
                    total_gates += 1
                    slack -= key
                    gate_added_flag = True
            if not gate_added_flag:
                break

        # For efficiency in the Python loop, this uses Numpy vectorization to pre-calculate the
        # indices into the lists of qubits and parameters for every gate, and then suitably
        # randomizes those lists.
        q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
        q_indices[0] = p_indices[0] = 0
        np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
        np.cumsum(gate_specs["num_params"], out=p_indices[1:])
        parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
        rng.shuffle(qubits)

        # We've now generated everything we're going to need.  Now just to add everything.  The
        # conditional check is outside the two loops to make the more common case of no conditionals
        # faster, since in Python we don't have a compiler to do this for us.
        if conditional and layer_number != 0:
            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])

                if is_cond:
                    qc.measure(qc.qubits, cr)
                    # The condition values are required to be bigints, not Numpy's fixed-width type.
                    with qc.if_test((cr, int(condition_values[c_ptr]))):
                        c_ptr += 1
                        qc.append(
                            CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end])
                        )
                else:
                    qc._append(
                        CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end])
                    )
        else:
            for gate, q_start, q_end, p_start, p_end in zip(
                gate_specs["class"], q_indices[:-1], q_indices[1:], p_indices[:-1], p_indices[1:]
            ):
                operation = gate(*parameters[p_start:p_end])
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))
    if measure:
        qc.measure(qc.qubits, cr)

    return qc


def random_clifford_circuit(num_qubits, num_gates, gates="all", seed=None):
    """Generate a pseudo-random Clifford circuit.

    This function will generate a Clifford circuit by randomly selecting the chosen amount of Clifford
    gates from the set of standard gates in :mod:`qiskit.circuit.library.standard_gates`. For example:

    .. plot::
       :alt: Circuit diagram output by the previous code.
       :include-source:

       from qiskit.circuit.random import random_clifford_circuit

       circ = random_clifford_circuit(num_qubits=2, num_gates=6)
       circ.draw(output='mpl')

    Args:
        num_qubits (int): number of quantum wires.
        num_gates (int): number of gates in the circuit.
        gates (list[str]): optional list of Clifford gate names to randomly sample from.
            If ``"all"`` (default), use all Clifford gates in the standard library.
        seed (int | np.random.Generator): sets random seed/generator (optional).

    Returns:
        QuantumCircuit: constructed circuit
    """

    gates_1q = [gate for gate in _BASIS_1Q if gate not in {"v", "w", "id", "iden", "sinv"}]
    gates_2q = list(_BASIS_2Q.keys())

    if gates == "all":
        if num_qubits == 1:
            gates = gates_1q
        else:
            gates = gates_1q + gates_2q

    instructions = {
        "i": (standard_gates.IGate(), 1),
        "x": (standard_gates.XGate(), 1),
        "y": (standard_gates.YGate(), 1),
        "z": (standard_gates.ZGate(), 1),
        "h": (standard_gates.HGate(), 1),
        "s": (standard_gates.SGate(), 1),
        "sdg": (standard_gates.SdgGate(), 1),
        "sx": (standard_gates.SXGate(), 1),
        "sxdg": (standard_gates.SXdgGate(), 1),
        "cx": (standard_gates.CXGate(), 2),
        "cy": (standard_gates.CYGate(), 2),
        "cz": (standard_gates.CZGate(), 2),
        "swap": (standard_gates.SwapGate(), 2),
        "iswap": (standard_gates.iSwapGate(), 2),
        "ecr": (standard_gates.ECRGate(), 2),
        "dcx": (standard_gates.DCXGate(), 2),
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
        circ.append(gate, qargs, copy=False)

    return circ
