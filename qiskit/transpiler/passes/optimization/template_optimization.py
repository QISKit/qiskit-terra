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
Given a template and a circuit: it applies template matching and substitutes
all compatible maximal matches that reduces the size of the circuit.

**Reference:**

[1] Iten, R., Moyard, R., Metger, T., Sutter, D. and Woerner, S., 2020.
Exact and practical pattern matching for quantum circuit optimization.
`arXiv:1909.05270 <https://arxiv.org/abs/1909.05270>`_
"""
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGDependency
from qiskit.converters.circuit_to_dagdependency import circuit_to_dagdependency
from qiskit.converters.dagdependency_to_circuit import dagdependency_to_circuit
from qiskit.converters.dag_to_dagdependency import dag_to_dagdependency
from qiskit.converters.dagdependency_to_dag import dagdependency_to_dag
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit.library.templates import template_nct_2a_1, template_nct_2a_2, template_nct_2a_3
from qiskit.quantum_info.operators.operator import Operator
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.optimization.template_matching import (
    TemplateMatching,
    TemplateSubstitution,
    MaximalMatches,
)


class TemplateOptimization(TransformationPass):
    """
    Class for the template optimization pass.
    """

    def __init__(
        self,
        template_list=None,
        heuristics_qubits_param=None,
        heuristics_backward_param=None,
        user_cost_dict=None,
    ):
        """
        Args:
            template_list (list[QuantumCircuit()]): list of the different template circuit to apply.
            heuristics_backward_param (list[int]): [length, survivor] Those are the parameters for
                applying heuristics on the backward part of the algorithm. This part of the
                algorithm creates a tree of matching scenario. This tree grows exponentially. The
                heuristics evaluate which scenarios have the longest match and keep only those.
                The length is the interval in the tree for cutting it and survivor is the number
                of scenarios that are kept. We advise to use l=3 and s=1 to have serious time
                advantage. We remind that the heuristics implies losing a part of the maximal
                matches. Check reference for more details.
            heuristics_qubits_param (list[int]): [length] The heuristics for the qubit choice make
                guesses from the dag dependency of the circuit in order to limit the number of
                qubit configurations to explore. The length is the number of successors or not
                predecessors that will be explored in the dag dependency of the circuit, each
                qubits of the nodes are added to the set of authorized qubits. We advise to use
                length=1. Check reference for more details.
            user_cost_dict (Dict[str, int]): quantum cost dictionary passed to TemplateSubstitution
                to configure its behavior. This will override any default values if None
                is not given. The key is the name of the gate and the value its quantum cost.
        """
        super().__init__()
        # If no template is given; the templates are set as x-x, cx-cx, ccx-ccx.
        if template_list is None:
            template_list = [template_nct_2a_1(), template_nct_2a_2(), template_nct_2a_3()]
        self.template_list = template_list
        self.heuristics_qubits_param = (
            heuristics_qubits_param if heuristics_qubits_param is not None else []
        )
        self.heuristics_backward_param = (
            heuristics_backward_param if heuristics_backward_param is not None else []
        )

        self.user_cost_dict = user_cost_dict

    def run(self, dag):
        """
        Args:
            dag(DAGCircuit): DAG circuit.
        Returns:
            DAGCircuit: optimized DAG circuit.
        Raises:
            TranspilerError: If the template has not the right form or
             if the output circuit acts differently as the input circuit.
        """
        circuit_dag = dag
        circuit_dag_dep = dag_to_dagdependency(circuit_dag)

        for template in self.template_list:
            if not isinstance(template, (QuantumCircuit, DAGDependency)):
                raise TranspilerError("A template is a QuantumCircuit or a DAGDependency.")

            if len(template.qubits) > len(circuit_dag_dep.qubits):
                continue

            identity = np.identity(2 ** len(template.qubits), dtype=complex)
            try:
                if isinstance(template, DAGDependency):
                    data = Operator(dagdependency_to_circuit(template)).data
                else:
                    data = Operator(template).data

                comparison = np.allclose(data, identity)

                if not comparison:
                    raise TranspilerError(
                        "A template is a QuantumCircuit() that performs the identity."
                    )
            except TypeError:
                pass

            if isinstance(template, QuantumCircuit):
                template_dag_dep = circuit_to_dagdependency(template)
            else:
                template_dag_dep = template

            template_m = TemplateMatching(
                circuit_dag_dep,
                template_dag_dep,
                self.heuristics_qubits_param,
                self.heuristics_backward_param,
            )

            template_m.run_template_matching()

            matches = template_m.match_list

            if matches:
                maximal = MaximalMatches(matches)
                maximal.run_maximal_matches()
                max_matches = maximal.max_match_list

                substitution = TemplateSubstitution(
                    max_matches,
                    template_m.circuit_dag_dep,
                    template_m.template_dag_dep,
                    self.user_cost_dict,
                )
                substitution.run_dag_opt()

                circuit_dag_dep = substitution.dag_dep_optimized
            else:
                continue
        circuit_dag = dagdependency_to_dag(circuit_dag_dep)
        return circuit_dag
