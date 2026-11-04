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
Template matching substitution, given a list of maximal matches it substitutes
them in circuit and creates a new optimized dag version of the circuit.
"""
import collections
import copy
import itertools

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.dagcircuit.dagdependency import DAGDependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.utils import optionals as _optionals


class SubstitutionConfig:
    """
    Class to store the configuration of a given match substitution, which circuit
    gates, template gates, qubits, and clbits and predecessors of the match
    in the circuit.
    """

    def __init__(
        self,
        circuit_config,
        template_config,
        pred_block,
        qubit_config,
        template_dag_dep,
        clbit_config=None,
    ):
        self.template_dag_dep = template_dag_dep
        self.circuit_config = circuit_config
        self.template_config = template_config
        self.qubit_config = qubit_config
        self.clbit_config = clbit_config if clbit_config is not None else []
        self.pred_block = pred_block

    def has_parameters(self):
        """Ensure that the template does not have parameters."""
        for node in self.template_dag_dep.get_nodes():
            for param in node.op.params:
                if isinstance(param, ParameterExpression):
                    return True

        return False


class TemplateSubstitution:
    """
    Class to run the substitution algorithm from the list of maximal matches.

    .. warning::

        This pass internally relies on using SymPy and will pass strings from the
        :class:`.DAGCircuit` the pass is run on to SymPy. This is inherently unsafe as SymPy
        heavily relies on Python's :func:`.eval` function for the processing of strings. This
        pass should not be used with untrusted input.

    """

    def __init__(self, max_matches, circuit_dag_dep, template_dag_dep, user_cost_dict=None):
        """
        Initialize TemplateSubstitution with necessary arguments.
        Args:
            max_matches (list): list of maximal matches obtained from the running
             the template matching algorithm.
            circuit_dag_dep (DAGDependency): circuit in the dag dependency form.
            template_dag_dep (DAGDependency): template in the dag dependency form.
            user_cost_dict (Optional[dict]): user provided cost dictionary that will override
                the default cost dictionary.
        """

        self.match_stack = max_matches
        self.circuit_dag_dep = circuit_dag_dep
        self.template_dag_dep = template_dag_dep

        self.substitution_list = []
        self.unmatched_list = []
        self.dag_dep_optimized = DAGDependency()
        self.dag_optimized = DAGCircuit()

        if user_cost_dict is not None:
            self.cost_dict = dict(user_cost_dict)
        else:
            self.cost_dict = {
                "id": 0,
                "x": 1,
                "y": 1,
                "z": 1,
                "h": 1,
                "t": 1,
                "tdg": 1,
                "s": 1,
                "sdg": 1,
                "u1": 1,
                "u2": 2,
                "u3": 2,
                "rx": 1,
                "ry": 1,
                "rz": 1,
                "r": 2,
                "cx": 2,
                "cy": 4,
                "cz": 4,
                "ch": 8,
                "swap": 6,
                "iswap": 8,
                "rxx": 9,
                "ryy": 9,
                "rzz": 5,
                "rzx": 7,
                "ms": 9,
                "cu3": 10,
                "crx": 10,
                "cry": 10,
                "crz": 10,
                "ccx": 21,
                "rccx": 12,
                "c3x": 96,
                "rc3x": 24,
                "c4x": 312,
                "p": 1,
            }

    def _pred_block(self, circuit_sublist, index):
        """
        It returns the predecessors of a given part of the circuit.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            index (int): Index of the group of matches.
        Returns:
            list: List of predecessors of the current match circuit configuration.
        """
        predecessors = set()
        for node_id in circuit_sublist:
            predecessors = predecessors | set(self.circuit_dag_dep.get_node(node_id).predecessors)

        exclude = set()
        for elem in self.substitution_list[:index]:
            exclude = exclude | set(elem.circuit_config) | set(elem.pred_block)

        pred = list(predecessors - set(circuit_sublist) - exclude)
        pred.sort()

        return pred

    def _quantum_cost(self, left, right):
        """
        Compare the two parts of the template and returns True if the quantum cost is reduced.
        Args:
            left (list): list of matched nodes in the template.
            right (list): list of nodes to be replaced.
        Returns:
            bool: True if the quantum cost is reduced
        """
        cost_left = 0
        for i in left:
            cost_left += self.cost_dict[self.template_dag_dep.get_node(i).name]

        cost_right = 0
        for j in right:
            cost_right += self.cost_dict[self.template_dag_dep.get_node(j).name]

        return cost_left > cost_right

    def _rules(self, circuit_sublist, template_sublist, template_complement):
        """
        Set of rules to decide whether the match is to be substitute or not.
        Args:
            circuit_sublist (list): list of the gates matched in the circuit.
            template_sublist (list): list of matched nodes in the template.
            template_complement (list): list of gates not matched in the template.
        Returns:
            bool: True if the match respects the given rule for replacement, False otherwise.
        """
        if self._quantum_cost(template_sublist, template_complement):
            for elem in circuit_sublist:
                for config in self.substitution_list:
                    if any(elem == x for x in config.circuit_config):
                        return False
            return True
        else:
            return False

    def _template_inverse(self, template_list, template_sublist, template_complement):
        """
        The template circuit realizes the identity operator, then given the list of
        matches in the template, it returns the inverse part of the template that
        will be replaced.
        Args:
            template_list (list): list of all gates in the template.
            template_sublist (list): list of the gates matched in the circuit.
            template_complement  (list): list of gates not matched in the template.
        Returns:
            list: the template inverse part that will substitute the circuit match.
        """
        inverse = template_complement
        left = []
        right = []

        pred = set()
        for index in template_sublist:
            pred = pred | set(self.template_dag_dep.get_node(index).predecessors)
        pred = list(pred - set(template_sublist))

        succ = set()
        for index in template_sublist:
            succ = succ | set(self.template_dag_dep.get_node(index).successors)
        succ = list(succ - set(template_sublist))

        comm = list(set(template_list) - set(pred) - set(succ))

        for elem in inverse:
            if elem in pred:
                left.append(elem)
            elif elem in succ:
                right.append(elem)
            elif elem in comm:
                right.append(elem)

        left.sort()
        right.sort()

        left.reverse()
        right.reverse()

        total = left + right
        return total

    def _substitution_sort(self):
        """
        Sort the substitution list.
        """
        ordered = False
        while not ordered:
            ordered = self._permutation()

    def _permutation(self):
        """
        Permute two groups of matches if first one has predecessors in the second one.
        Returns:
            bool: True if the matches groups are in the right order, False otherwise.
        """
        for scenario in self.substitution_list:
            predecessors = set()
            for match in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag_dep.get_node(match).predecessors)
            predecessors = predecessors - set(scenario.circuit_config)
            index = self.substitution_list.index(scenario)
            for scenario_b in self.substitution_list[index::]:
                if set(scenario_b.circuit_config) & predecessors:

                    index1 = self.substitution_list.index(scenario)
                    index2 = self.substitution_list.index(scenario_b)

                    scenario_pop = self.substitution_list.pop(index2)
                    self.substitution_list.insert(index1, scenario_pop)
                    return False
        return True

    def _remove_impossible(self):
        """
        Remove matched groups if they both have predecessors in the other one, they are not
        compatible.
        """
        list_predecessors = []
        remove_list = []

        # Initialize predecessors for each group of matches.
        for scenario in self.substitution_list:
            predecessors = set()
            for index in scenario.circuit_config:
                predecessors = predecessors | set(self.circuit_dag_dep.get_node(index).predecessors)
            list_predecessors.append(predecessors)

        # Check if two groups of matches are incompatible.
        for scenario_a in self.substitution_list:
            if scenario_a in remove_list:
                continue
            index_a = self.substitution_list.index(scenario_a)
            circuit_a = scenario_a.circuit_config
            for scenario_b in self.substitution_list[index_a + 1 : :]:
                if scenario_b in remove_list:
                    continue
                index_b = self.substitution_list.index(scenario_b)
                circuit_b = scenario_b.circuit_config
                if (set(circuit_a) & list_predecessors[index_b]) and (
                    set(circuit_b) & list_predecessors[index_a]
                ):
                    remove_list.append(scenario_b)

        # Remove the incompatible groups from the list.
        if remove_list:
            self.substitution_list = [
                scenario for scenario in self.substitution_list if scenario not in remove_list
            ]

    def _substitution(self):
        """
        From the list of maximal matches, it chooses which one will be used and gives the necessary
        details for each substitution(template inverse, predecessors of the match).
        """

        while self.match_stack:

            # Get the first match scenario of the list
            current = self.match_stack.pop(0)

            current_match = current.match
            current_qubit = current.qubit
            current_clbit = current.clbit

            template_sublist = [x[0] for x in current_match]
            circuit_sublist = [x[1] for x in current_match]
            circuit_sublist.sort()

            # Fake bind any parameters in the template
            template = self._attempt_bind(template_sublist, circuit_sublist)
            if template is None or self._incr_num_parameters(template):
                continue

            template_list = range(self.template_dag_dep.size())
            template_complement = list(set(template_list) - set(template_sublist))

            # If the match obey the rule then it is added to the list.
            if self._rules(circuit_sublist, template_sublist, template_complement):
                template_sublist_inverse = self._template_inverse(
                    template_list, template_sublist, template_complement
                )

                config = SubstitutionConfig(
                    circuit_sublist,
                    template_sublist_inverse,
                    [],
                    current_qubit,
                    template,
                    current_clbit,
                )
                self.substitution_list.append(config)

        # Remove incompatible matches.
        self._remove_impossible()

        # First sort the matches according to the smallest index in the matches (circuit).
        self.substitution_list.sort(key=lambda x: x.circuit_config[0])

        # Change position of the groups due to predecessors of other groups.
        self._substitution_sort()

        for scenario in self.substitution_list:
            index = self.substitution_list.index(scenario)
            scenario.pred_block = self._pred_block(scenario.circuit_config, index)

        circuit_list = []
        for elem in self.substitution_list:
            circuit_list = circuit_list + elem.circuit_config + elem.pred_block

        # Unmatched gates that are not predecessors of any group of matches.
        self.unmatched_list = sorted(set(range(self.circuit_dag_dep.size())) - set(circuit_list))

    def run_dag_opt(self):
        """
        It runs the substitution algorithm and creates the optimized DAGCircuit().
        """
        self._substitution()

        dag_dep_opt = DAGDependency()

        dag_dep_opt.name = self.circuit_dag_dep.name
        dag_dep_opt.global_phase = self.circuit_dag_dep.global_phase

        qregs = list(self.circuit_dag_dep.qregs.values())
        cregs = list(self.circuit_dag_dep.cregs.values())

        for register in qregs:
            dag_dep_opt.add_qreg(register)

        for register in cregs:
            dag_dep_opt.add_creg(register)

        already_sub = []

        if self.substitution_list:
            # Loop over the different matches.
            for group in self.substitution_list:

                circuit_sub = group.circuit_config
                template_inverse = group.template_config

                pred = group.pred_block

                qubit = group.qubit_config[0]

                if group.clbit_config:
                    clbit = group.clbit_config[0]
                else:
                    clbit = []

                # First add all the predecessors of the given match.
                for elem in pred:
                    node = self.circuit_dag_dep.get_node(elem)
                    inst = node.op.copy()
                    dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)
                    already_sub.append(elem)

                already_sub = already_sub + circuit_sub

                # Then add the inverse of the template.
                for index in template_inverse:
                    all_qubits = self.circuit_dag_dep.qubits
                    qarg_t = group.template_dag_dep.get_node(index).qindices
                    qarg_c = [qubit[x] for x in qarg_t]
                    qargs = [all_qubits[x] for x in qarg_c]

                    all_clbits = self.circuit_dag_dep.clbits
                    carg_t = group.template_dag_dep.get_node(index).cindices

                    if all_clbits and clbit:
                        carg_c = [clbit[x] for x in carg_t]
                        cargs = [all_clbits[x] for x in carg_c]
                    else:
                        cargs = []
                    node = group.template_dag_dep.get_node(index)
                    inst = node.op.copy()
                    dag_dep_opt.add_op_node(inst.inverse(), qargs, cargs)

                # Update global phase to account for the template's phase.
                dag_dep_opt.global_phase -= group.template_dag_dep.global_phase

            # Add the unmatched gates.
            for node_id in self.unmatched_list:
                node = self.circuit_dag_dep.get_node(node_id)
                inst = node.op.copy()
                dag_dep_opt.add_op_node(inst, node.qargs, node.cargs)

            dag_dep_opt._add_predecessors()
            dag_dep_opt._add_successors()
        # If there is no valid match, it returns the original dag.
        else:
            dag_dep_opt = self.circuit_dag_dep

        self.dag_dep_optimized = dag_dep_opt
        self.dag_optimized = dagdependency_to_dag(dag_dep_opt)

    @_optionals.HAS_SYMPY.require_in_call("Bind parameters in templates")
    def _attempt_bind(self, template_sublist, circuit_sublist):
        """
        Copies the template and attempts to bind any parameters,
        i.e. attempts to solve for a valid parameter assignment.
        template_sublist and circuit_sublist match up to the
        assignment of the parameters. For example the template

        .. code-block:: text

                 ┌───────────┐                  ┌────────┐
            q_0: ┤ P(-1.0*β) ├──■────────────■──┤0       ├
                 ├───────────┤┌─┴─┐┌──────┐┌─┴─┐│  CZ(β) │
            q_1: ┤ P(-1.0*β) ├┤ X ├┤ P(β) ├┤ X ├┤1       ├
                 └───────────┘└───┘└──────┘└───┘└────────┘

        should only maximally match once in the circuit

        .. code-block:: text

                 ┌───────┐
            q_0: ┤ P(-2) ├──■────────────■────────────────────────────
                 ├───────┤┌─┴─┐┌──────┐┌─┴─┐┌──────┐
            q_1: ┤ P(-2) ├┤ X ├┤ P(2) ├┤ X ├┤ P(3) ├──■────────────■──
                 └┬──────┤└───┘└──────┘└───┘└──────┘┌─┴─┐┌──────┐┌─┴─┐
            q_2: ─┤ P(3) ├──────────────────────────┤ X ├┤ P(3) ├┤ X ├
                  └──────┘                          └───┘└──────┘└───┘

        However, up until attempt bind is called, the soft matching
        will have found two matches due to the parameters.
        The first match can be satisfied with β=2. However, the
        second match would imply both β=3 and β=-3 which is impossible.
        Attempt bind detects inconsistencies by solving a system of equations
        given by the parameter expressions in the sub-template and the
        value of the parameters in the gates of the sub-circuit. If a
        solution is found then the match is valid and the parameters
        are assigned. If not, None is returned.

        In order to resolve the conflict of the same parameter names in the
        circuit and template, each variable in the template sublist is
        re-assigned to a new dummy parameter with a completely separate name
        if it clashes with one that exists in an input circuit.

        Args:
            template_sublist (list): part of the matched template.
            circuit_sublist (list): part of the matched circuit.

        Returns:
            DAGDependency: A deep copy of the template with
                the parameters bound. If no binding satisfies the
                parameter constraints, returns None.
        """
        import sympy as sym

        circuit_params, template_params = [], []
        # Set of all parameter names that are present in the circuits to be optimized.
        circuit_params_set = set()

        template_dag_dep = copy.deepcopy(self.template_dag_dep)

        # add parameters from circuit to circuit_params
        for idx, _ in enumerate(template_sublist):
            qc_idx = circuit_sublist[idx]
            parameters = self.circuit_dag_dep.get_node(qc_idx).op.params
            circuit_params += parameters
            for parameter in parameters:
                if isinstance(parameter, ParameterExpression):
                    circuit_params_set.update(x.name for x in parameter.parameters)

        _dummy_counter = itertools.count()

        def dummy_parameter():
            # Strictly not _guaranteed_ to avoid naming clashes, but if someone's calling their
            # parameters this then that's their own fault.
            return Parameter(f"_qiskit_template_dummy_{next(_dummy_counter)}")

        # Substitutions for parameters that have clashing names between the input circuits and the
        # defined templates.
        template_clash_substitutions = collections.defaultdict(dummy_parameter)

        # add parameters from template to template_params, replacing parameters with names that
        # clash with those in the circuit.
        for t_idx in template_sublist:
            node = template_dag_dep.get_node(t_idx)
            sub_node_params = []
            for t_param_exp in node.op.params:
                if isinstance(t_param_exp, ParameterExpression):
                    for t_param in t_param_exp.parameters:
                        if t_param.name in circuit_params_set:
                            new_param = template_clash_substitutions[t_param.name]
                            t_param_exp = t_param_exp.assign(t_param, new_param)
                sub_node_params.append(t_param_exp)
                template_params.append(t_param_exp)
            if not node.op.mutable:
                node.op = node.op.to_mutable()
            node.op.params = sub_node_params

        for node in template_dag_dep.get_nodes():
            sub_node_params = []
            for param_exp in node.op.params:
                if isinstance(param_exp, ParameterExpression):
                    for param in param_exp.parameters:
                        if param.name in template_clash_substitutions:
                            param_exp = param_exp.assign(
                                param, template_clash_substitutions[param.name]
                            )
                sub_node_params.append(param_exp)

            if not node.op.mutable:
                node.op = node.op.to_mutable()
            node.op.params = sub_node_params

        # Create the fake binding dict and check
        equations, circ_dict, temp_symbols = [], {}, {}
        for circuit_param, template_param in zip(circuit_params, template_params):
            if isinstance(template_param, ParameterExpression):
                if isinstance(circuit_param, ParameterExpression):
                    circ_param_sym = circuit_param.sympify()
                else:
                    # if it's not a ParameterExpression we're a float
                    circ_param_sym = sym.Float(circuit_param)
                equations.append(sym.Eq(template_param.sympify(), circ_param_sym))

                for param in template_param.parameters:
                    temp_symbols[param] = param.sympify()

                if isinstance(circuit_param, ParameterExpression):
                    for param in circuit_param.parameters:
                        circ_dict[param.name] = param
            elif template_param != circuit_param:
                # Both are numeric parameters, but aren't equal.
                return None

        if not temp_symbols:
            return template_dag_dep

        # Check compatibility by solving the resulting equation.  `dict=True` (surprisingly) forces
        # the output to always be a list, even if there's exactly one solution.
        sym_sol = sym.solve(equations, set(temp_symbols.values()), dict=True)
        if not sym_sol:
            # No solutions.
            return None
        # If there's multiple solutions, arbitrarily pick the first one.
        sol = {
            param.name: ParameterExpression(circ_dict, str(expr))
            for param, expr in sym_sol[0].items()
        }
        fake_bind = {key: sol[key.name] for key in temp_symbols}

        for node in template_dag_dep.get_nodes():
            bound_params = []
            for param_exp in node.op.params:
                if isinstance(param_exp, ParameterExpression):
                    for param in param_exp.parameters:
                        if param in fake_bind:
                            if fake_bind[param] not in bound_params:
                                param_exp = param_exp.assign(param, fake_bind[param])
                else:
                    param_exp = float(param_exp)
                bound_params.append(param_exp)

            if not node.op.mutable:
                node.op = node.op.to_mutable()
            node.op.params = bound_params

        return template_dag_dep

    def _incr_num_parameters(self, template):
        """
        Checks if template substitution would increase the number of
        parameters in the circuit.
        """
        template_params = set()
        for param_list in (node.op.params for node in template.get_nodes()):
            for param_exp in param_list:
                if isinstance(param_exp, ParameterExpression):
                    template_params.update(param_exp.parameters)

        circuit_params = set()
        for param_list in (node.op.params for node in self.circuit_dag_dep.get_nodes()):
            for param_exp in param_list:
                if isinstance(param_exp, ParameterExpression):
                    circuit_params.update(param_exp.parameters)

        return len(template_params) > len(circuit_params)
