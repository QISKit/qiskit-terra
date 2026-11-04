# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Template matching in the backward direction, it takes an initial match, a
configuration of qubit, both circuit and template as inputs and the list
obtained from forward match. The result is a list of matches between the
template and the circuit.


**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_

"""
import heapq

from qiskit.circuit.controlledgate import ControlledGate


class Match:
    """
    Object to represent a match and its qubit configurations.
    """

    def __init__(self, match, qubit, clbit):
        """
        Create a Match class with necessary arguments.
        Args:
            match (list): list of matched gates.
            qubit (list): list of qubits configuration.
            clbit (list): list of clbits configuration.

        """
        # Match list
        self.match = match
        # Qubits list for circuit
        self.qubit = [qubit]
        # Clbits for template
        self.clbit = [clbit]


class MatchingScenarios:
    """
    Class to represent a matching scenario.
    """

    def __init__(
        self, circuit_matched, circuit_blocked, template_matched, template_blocked, matches, counter
    ):
        """
        Create a MatchingScenarios class with necessary arguments.
        Args:
            circuit_matched (list): list of matchedwith attributes in the circuit.
            circuit_blocked (list): list of isblocked attributes in the circuit.
            template_matched (list): list of matchedwith attributes in the template.
            template_blocked (list): list of isblocked attributes in the template.
            matches (list): list of matches.
            counter (int): counter of the number of circuit gates already considered.
        """
        self.circuit_matched = circuit_matched
        self.template_matched = template_matched
        self.circuit_blocked = circuit_blocked
        self.template_blocked = template_blocked
        self.matches = matches
        self.counter = counter


class MatchingScenariosList:
    """
    Object to define a list of MatchingScenarios, with method to append
    and pop elements.
    """

    def __init__(self):
        """
        Create an empty MatchingScenariosList.
        """
        self.matching_scenarios_list = []

    def append_scenario(self, matching):
        """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
        self.matching_scenarios_list.append(matching)

    def pop_scenario(self):
        """
        Pop the first scenario of the list.
        Returns:
            MatchingScenarios: a scenario of match.
        """
        # Pop the first MatchingScenario and returns it
        first = self.matching_scenarios_list[0]
        self.matching_scenarios_list.pop(0)
        return first


class BackwardMatch:
    """
    Class BackwardMatch allows to run backward direction part of template
    matching algorithm.
    """

    def __init__(
        self,
        circuit_dag_dep,
        template_dag_dep,
        forward_matches,
        node_id_c,
        node_id_t,
        qubits,
        clbits=None,
        heuristics_backward_param=None,
    ):
        """
        Create a ForwardMatch class with necessary arguments.
        Args:
            circuit_dag_dep (DAGDependency): circuit in the dag dependency form.
            template_dag_dep (DAGDependency): template in the dag dependency form.
            forward_matches (list): list of match obtained in the forward direction.
            node_id_c (int): index of the first gate matched in the circuit.
            node_id_t (int): index of the first gate matched in the template.
            qubits (list): list of considered qubits in the circuit.
            clbits (list): list of considered clbits in the circuit.
            heuristics_backward_param (list): list that contains the two parameters for
            applying the heuristics (length and survivor).
        """
        self.circuit_dag_dep = circuit_dag_dep.copy()
        self.template_dag_dep = template_dag_dep.copy()
        self.qubits = qubits
        self.clbits = clbits if clbits is not None else []
        self.node_id_c = node_id_c
        self.node_id_t = node_id_t
        self.forward_matches = forward_matches
        self.match_final = []
        self.heuristics_backward_param = (
            heuristics_backward_param if heuristics_backward_param is not None else []
        )
        self.matching_list = MatchingScenariosList()

    def _gate_indices(self):
        """
        Function which returns the list of gates that are not match and not
        blocked for the first scenario.
        Returns:
            list: list of gate id.
        """
        gate_indices = []

        current_dag = self.circuit_dag_dep

        for node in current_dag.get_nodes():
            if (not node.matchedwith) and (not node.isblocked):
                gate_indices.append(node.node_id)
        gate_indices.reverse()
        return gate_indices

    def _find_backward_candidates(self, template_blocked, matches):
        """
        Function which returns the list possible backward candidates in the template dag.
        Args:
            template_blocked (list): list of attributes isblocked in the template circuit.
            matches (list): list of matches.
        Returns:
            list: list of backward candidates (id).
        """
        template_block = []

        for node_id in range(self.node_id_t, self.template_dag_dep.size()):
            if template_blocked[node_id]:
                template_block.append(node_id)

        matches_template = sorted(match[0] for match in matches)

        successors = self.template_dag_dep.get_node(self.node_id_t).successors
        potential = []
        for index in range(self.node_id_t + 1, self.template_dag_dep.size()):
            if (index not in successors) and (index not in template_block):
                potential.append(index)

        candidates_indices = list(set(potential) - set(matches_template))
        candidates_indices = sorted(candidates_indices)
        candidates_indices.reverse()

        return candidates_indices

    def _update_qarg_indices(self, qarg):
        """
        Change qubits indices of the current circuit node in order to
        be comparable the indices of the template qubits list.
        Args:
            qarg (list): list of qubits indices from the circuit for a given gate.
        Returns:
            list: circuit indices update for qubits.
        """
        qarg_indices = []
        for q in qarg:
            if q in self.qubits:
                qarg_indices.append(self.qubits.index(q))
        if len(qarg) != len(qarg_indices):
            qarg_indices = []
        return qarg_indices

    def _update_carg_indices(self, carg):
        """
        Change clbits indices of the current circuit node in order to
        be comparable the indices of the template qubits list.
        Args:
            carg (list): list of clbits indices from the circuit for a given gate.
        Returns:
            list: circuit indices update for clbits.
        """
        carg_indices = []
        if carg:
            for q in carg:
                if q in self.clbits:
                    carg_indices.append(self.clbits.index(q))
            if len(carg) != len(carg_indices):
                carg_indices = []
        return carg_indices

    def _is_same_op(self, node_circuit, node_template):
        """
        Check if two instructions are the same.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
        Returns:
            bool: True if the same, False otherwise.
        """
        return node_circuit.op.soft_compare(node_template.op)

    def _is_same_q_conf(self, node_circuit, node_template, qarg_circuit):
        """
        Check if the qubits configurations are compatible.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
            qarg_circuit (list): qubits configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """
        # If the gate is controlled, then the control qubits have to be compared as sets.
        if isinstance(node_circuit.op, ControlledGate):

            c_template = node_template.op.num_ctrl_qubits

            if c_template == 1:
                return qarg_circuit == node_template.qindices

            else:
                control_qubits_template = node_template.qindices[:c_template]
                control_qubits_circuit = qarg_circuit[:c_template]

                if set(control_qubits_circuit) == set(control_qubits_template):

                    target_qubits_template = node_template.qindices[c_template::]
                    target_qubits_circuit = qarg_circuit[c_template::]

                    if node_template.op.base_gate.name in [
                        "rxx",
                        "ryy",
                        "rzz",
                        "swap",
                        "iswap",
                        "ms",
                    ]:
                        return set(target_qubits_template) == set(target_qubits_circuit)
                    else:
                        return target_qubits_template == target_qubits_circuit
                else:
                    return False
        # For non controlled gates, the qubits indices for symmetric gates can be compared as sets
        # But for non-symmetric gates the qubits indices have to be compared as lists.
        elif node_template.op.name in ["rxx", "ryy", "rzz", "swap", "iswap", "ms"]:
            return set(qarg_circuit) == set(node_template.qindices)
        else:
            return qarg_circuit == node_template.qindices

    def _is_same_c_conf(self, node_circuit, node_template, carg_circuit):
        """
        Check if the clbits configurations are compatible.
        Args:
            node_circuit (DAGDepNode): node in the circuit.
            node_template (DAGDepNode): node in the template.
            carg_circuit (list): clbits configuration for the Instruction in the circuit.
        Returns:
            bool: True if possible, False otherwise.
        """
        if (
            node_circuit.type == "op"
            and getattr(node_circuit.op, "_condition", None)
            and node_template.type == "op"
            and getattr(node_template.op, "_condition", None)
        ):
            if set(carg_circuit) != set(node_template.cindices):
                return False
            if (
                getattr(node_circuit.op, "_condition", None)[1]
                != getattr(node_template.op, "_condition", None)[1]
            ):
                return False
        return True

    def _init_matched_blocked_list(self):
        """
        Initialize the list of blocked and matchedwith attributes.
        Returns:
            Tuple[list, list, list, list]:
            First list contains the attributes matchedwith in the circuit,
            second list contains the attributes isblocked in the circuit,
            third list contains the attributes matchedwith in the template,
            fourth list contains the attributes isblocked in the template.
        """
        circuit_matched = []
        circuit_blocked = []

        for node in self.circuit_dag_dep.get_nodes():
            circuit_matched.append(node.matchedwith)
            circuit_blocked.append(node.isblocked)

        template_matched = []
        template_blocked = []

        for node in self.template_dag_dep.get_nodes():
            template_matched.append(node.matchedwith)
            template_blocked.append(node.isblocked)

        return circuit_matched, circuit_blocked, template_matched, template_blocked

    def _backward_heuristics(self, gate_indices, length, survivor):
        """
        Heuristics to cut the tree in the backward match algorithm
        Args:
            gate_indices (list): list of candidates in the circuit.
            length (int): depth for cutting the tree, cutting operation is repeated every length.
            survivor (int): number of survivor branches.
        """
        # Set the list of the counter for the different scenarios.
        list_counter = []

        for scenario in self.matching_list.matching_scenarios_list:
            list_counter.append(scenario.counter)

        metrics = []
        # If all scenarios have the same counter and the counter is divisible by length.
        if list_counter.count(list_counter[0]) == len(list_counter) and list_counter[0] <= len(
            gate_indices
        ):
            if (list_counter[0] - 1) % length == 0:
                # The list metrics contains metric results for each scenarios.
                for scenario in self.matching_list.matching_scenarios_list:
                    metrics.append(self._backward_metrics(scenario))
                # Select only the scenarios with higher metrics for the given number of survivors.
                largest = heapq.nlargest(survivor, range(len(metrics)), key=lambda x: metrics[x])
                self.matching_list.matching_scenarios_list = [
                    i
                    for j, i in enumerate(self.matching_list.matching_scenarios_list)
                    if j in largest
                ]

    def _backward_metrics(self, scenario):
        """
        Heuristics to cut the tree in the backward match algorithm.
        Args:
            scenario (MatchingScenarios): scenario for the given match.
        Returns:
            int: length of the match for the given scenario.
        """
        return len(scenario.matches)

    def run_backward_match(self):
        """
        Apply the forward match algorithm and returns the list of matches given an initial match
        and a circuit qubits configuration.

        """
        match_store_list = []

        counter = 1

        # Initialize the list of attributes matchedwith and isblocked.
        (
            circuit_matched,
            circuit_blocked,
            template_matched,
            template_blocked,
        ) = self._init_matched_blocked_list()

        # First Scenario is stored in the MatchingScenariosList().
        first_match = MatchingScenarios(
            circuit_matched,
            circuit_blocked,
            template_matched,
            template_blocked,
            self.forward_matches,
            counter,
        )

        self.matching_list = MatchingScenariosList()
        self.matching_list.append_scenario(first_match)

        # Set the circuit indices that can be matched.
        gate_indices = self._gate_indices()

        number_of_gate_to_match = (
            self.template_dag_dep.size() - (self.node_id_t - 1) - len(self.forward_matches)
        )

        # While the scenario stack is not empty.
        while self.matching_list.matching_scenarios_list:

            # If parameters are given, the heuristics is applied.
            if self.heuristics_backward_param:
                self._backward_heuristics(
                    gate_indices,
                    self.heuristics_backward_param[0],
                    self.heuristics_backward_param[1],
                )

            scenario = self.matching_list.pop_scenario()

            circuit_matched = scenario.circuit_matched
            circuit_blocked = scenario.circuit_blocked
            template_matched = scenario.template_matched
            template_blocked = scenario.template_blocked
            matches_scenario = scenario.matches
            counter_scenario = scenario.counter

            # Part of the match list coming from the backward match.
            match_backward = [
                match for match in matches_scenario if match not in self.forward_matches
            ]

            # Matches are stored if the counter is bigger than the length of the list of
            # candidates in the circuit. Or if number of gate left to match is the same as
            # the length of the backward part of the match.
            if (
                counter_scenario > len(gate_indices)
                or len(match_backward) == number_of_gate_to_match
            ):
                matches_scenario.sort(key=lambda x: x[0])
                match_store_list.append(Match(matches_scenario, self.qubits, self.clbits))
                continue

            # First circuit candidate.
            circuit_id = gate_indices[counter_scenario - 1]
            node_circuit = self.circuit_dag_dep.get_node(circuit_id)

            # If the circuit candidate is blocked, only the counter is changed.
            if circuit_blocked[circuit_id]:
                matching_scenario = MatchingScenarios(
                    circuit_matched,
                    circuit_blocked,
                    template_matched,
                    template_blocked,
                    matches_scenario,
                    counter_scenario + 1,
                )
                self.matching_list.append_scenario(matching_scenario)
                continue

            # The candidates in the template.
            candidates_indices = self._find_backward_candidates(template_blocked, matches_scenario)
            # Update of the qubits/clbits indices in the circuit in order to be
            # comparable with the one in the template.
            qarg1 = node_circuit.qindices
            carg1 = node_circuit.cindices

            qarg1 = self._update_qarg_indices(qarg1)
            carg1 = self._update_carg_indices(carg1)

            global_match = False
            global_broken = []

            # Loop over the template candidates.
            for template_id in candidates_indices:

                node_template = self.template_dag_dep.get_node(template_id)
                qarg2 = self.template_dag_dep.get_node(template_id).qindices

                # Necessary but not sufficient conditions for a match to happen.
                if (
                    len(qarg1) != len(qarg2)
                    or set(qarg1) != set(qarg2)
                    or node_circuit.name != node_template.name
                ):
                    continue

                # Check if the qubit, clbit configuration are compatible for a match,
                # also check if the operation are the same.
                if (
                    self._is_same_q_conf(node_circuit, node_template, qarg1)
                    and self._is_same_c_conf(node_circuit, node_template, carg1)
                    and self._is_same_op(node_circuit, node_template)
                ):

                    # If there is a match the attributes are copied.
                    circuit_matched_match = circuit_matched.copy()
                    circuit_blocked_match = circuit_blocked.copy()

                    template_matched_match = template_matched.copy()
                    template_blocked_match = template_blocked.copy()

                    matches_scenario_match = matches_scenario.copy()

                    block_list = []
                    broken_matches_match = []

                    # Loop to check if the match is not connected, in this case
                    # the successors matches are blocked and unmatched.
                    for potential_block in self.template_dag_dep.successors(template_id):
                        if not template_matched_match[potential_block]:
                            template_blocked_match[potential_block] = True
                            block_list.append(potential_block)
                            for block_id in block_list:
                                for succ_id in self.template_dag_dep.successors(block_id):
                                    template_blocked_match[succ_id] = True
                                    if template_matched_match[succ_id]:
                                        new_id = template_matched_match[succ_id][0]
                                        circuit_matched_match[new_id] = []
                                        template_matched_match[succ_id] = []
                                        broken_matches_match.append(succ_id)

                    if broken_matches_match:
                        global_broken.append(True)
                    else:
                        global_broken.append(False)

                    new_matches_scenario_match = [
                        elem
                        for elem in matches_scenario_match
                        if elem[0] not in broken_matches_match
                    ]

                    condition = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_match:
                            condition = False
                            break

                    # First option greedy match.
                    if ([self.node_id_t, self.node_id_c] in new_matches_scenario_match) and (
                        condition or not match_backward
                    ):
                        template_matched_match[template_id] = [circuit_id]
                        circuit_matched_match[circuit_id] = [template_id]
                        new_matches_scenario_match.append([template_id, circuit_id])

                        new_matching_scenario = MatchingScenarios(
                            circuit_matched_match,
                            circuit_blocked_match,
                            template_matched_match,
                            template_blocked_match,
                            new_matches_scenario_match,
                            counter_scenario + 1,
                        )
                        self.matching_list.append_scenario(new_matching_scenario)

                        global_match = True

            if global_match:
                circuit_matched_block_s = circuit_matched.copy()
                circuit_blocked_block_s = circuit_blocked.copy()

                template_matched_block_s = template_matched.copy()
                template_blocked_block_s = template_blocked.copy()

                matches_scenario_block_s = matches_scenario.copy()

                circuit_blocked_block_s[circuit_id] = True

                broken_matches = []

                # Second option, not a greedy match, block all successors (push the gate
                # to the right).
                for succ in self.circuit_dag_dep.get_node(circuit_id).successors:
                    circuit_blocked_block_s[succ] = True
                    if circuit_matched_block_s[succ]:
                        broken_matches.append(succ)
                        new_id = circuit_matched_block_s[succ][0]
                        template_matched_block_s[new_id] = []
                        circuit_matched_block_s[succ] = []

                new_matches_scenario_block_s = [
                    elem for elem in matches_scenario_block_s if elem[1] not in broken_matches
                ]

                condition_not_greedy = True

                for back_match in match_backward:
                    if back_match not in new_matches_scenario_block_s:
                        condition_not_greedy = False
                        break

                if ([self.node_id_t, self.node_id_c] in new_matches_scenario_block_s) and (
                    condition_not_greedy or not match_backward
                ):
                    new_matching_scenario = MatchingScenarios(
                        circuit_matched_block_s,
                        circuit_blocked_block_s,
                        template_matched_block_s,
                        template_blocked_block_s,
                        new_matches_scenario_block_s,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(new_matching_scenario)

                # Third option: if blocking the successors breaks a match, we consider
                # also the possibility to block all predecessors (push the gate to the left).
                if broken_matches and all(global_broken):

                    circuit_matched_block_p = circuit_matched.copy()
                    circuit_blocked_block_p = circuit_blocked.copy()

                    template_matched_block_p = template_matched.copy()
                    template_blocked_block_p = template_blocked.copy()

                    matches_scenario_block_p = matches_scenario.copy()

                    circuit_blocked_block_p[circuit_id] = True

                    for pred in self.circuit_dag_dep.get_node(circuit_id).predecessors:
                        circuit_blocked_block_p[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched_block_p,
                        circuit_blocked_block_p,
                        template_matched_block_p,
                        template_blocked_block_p,
                        matches_scenario_block_p,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

            # If there is no match then there are three options.
            if not global_match:

                circuit_blocked[circuit_id] = True

                following_matches = []

                successors = self.circuit_dag_dep.get_node(circuit_id).successors
                for succ in successors:
                    if circuit_matched[succ]:
                        following_matches.append(succ)

                # First option, the circuit gate is not disturbing because there are no
                # following match and no predecessors.
                predecessors = self.circuit_dag_dep.get_node(circuit_id).predecessors

                if not predecessors or not following_matches:

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                else:

                    circuit_matched_nomatch = circuit_matched.copy()
                    circuit_blocked_nomatch = circuit_blocked.copy()

                    template_matched_nomatch = template_matched.copy()
                    template_blocked_nomatch = template_blocked.copy()

                    matches_scenario_nomatch = matches_scenario.copy()

                    # Second option, all predecessors are blocked (circuit gate is
                    # moved to the left).
                    for pred in predecessors:
                        circuit_blocked[pred] = True

                    matching_scenario = MatchingScenarios(
                        circuit_matched,
                        circuit_blocked,
                        template_matched,
                        template_blocked,
                        matches_scenario,
                        counter_scenario + 1,
                    )
                    self.matching_list.append_scenario(matching_scenario)

                    # Third option, all successors are blocked (circuit gate is
                    # moved to the right).

                    broken_matches = []

                    successors = self.circuit_dag_dep.get_node(circuit_id).successors

                    for succ in successors:
                        circuit_blocked_nomatch[succ] = True
                        if circuit_matched_nomatch[succ]:
                            broken_matches.append(succ)
                            circuit_matched_nomatch[succ] = []

                    new_matches_scenario_nomatch = [
                        elem for elem in matches_scenario_nomatch if elem[1] not in broken_matches
                    ]

                    condition_block = True

                    for back_match in match_backward:
                        if back_match not in new_matches_scenario_nomatch:
                            condition_block = False
                            break

                    if ([self.node_id_t, self.node_id_c] in matches_scenario_nomatch) and (
                        condition_block or not match_backward
                    ):
                        new_matching_scenario = MatchingScenarios(
                            circuit_matched_nomatch,
                            circuit_blocked_nomatch,
                            template_matched_nomatch,
                            template_blocked_nomatch,
                            new_matches_scenario_nomatch,
                            counter_scenario + 1,
                        )
                        self.matching_list.append_scenario(new_matching_scenario)

        length = max(len(m.match) for m in match_store_list)

        # Store the matches with maximal length.
        for scenario in match_store_list:
            if (len(scenario.match) == length) and not any(
                scenario.match == x.match for x in self.match_final
            ):
                self.match_final.append(scenario)
