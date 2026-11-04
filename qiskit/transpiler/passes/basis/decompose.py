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

"""Expand a gate in a circuit using its decomposition rules."""

from __future__ import annotations

from collections.abc import Sequence
from fnmatch import fnmatch

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.dagcircuit.dagnode import DAGOpNode
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.circuit.instruction import Instruction

from ..synthesis import HighLevelSynthesis


class Decompose(TransformationPass):
    """Expand a gate in a circuit using its decomposition rules."""

    def __init__(
        self,
        gates_to_decompose: (
            str | type[Instruction] | Sequence[str | type[Instruction]] | None
        ) = None,
        apply_synthesis: bool = False,
    ) -> None:
        """
        Args:
            gates_to_decompose: optional subset of gates to be decomposed,
                identified by gate label, name or type. Defaults to all gates.
            apply_synthesis: If ``True``, run :class:`.HighLevelSynthesis` to synthesize operations
                that do not have a definition attached.
        """
        super().__init__()
        self.gates_to_decompose = gates_to_decompose
        self.apply_synthesis = apply_synthesis

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Decompose pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            output dag where ``gate`` was expanded.
        """
        # We might use HLS to synthesize objects that do not have a definition
        hls = HighLevelSynthesis(qubits_initially_zero=False) if self.apply_synthesis else None

        # Walk through the DAG and expand each non-basis node
        for node in dag.op_nodes():
            # Check in self.gates_to_decompose if the operation should be decomposed
            if not self._should_decompose(node):
                continue

            if node.is_control_flow():
                decomposition = control_flow.map_blocks(self.run, node.op)
                dag.substitute_node(node, decomposition, inplace=True)

            elif getattr(node.op, "definition", None) is None:
                # if we try to synthesize, turn the node into a DAGCircuit and run HLS
                if self.apply_synthesis:
                    node_as_dag = _node_to_dag(node)
                    synthesized = hls.run(node_as_dag)
                    dag.substitute_node_with_dag(node, synthesized)

                # else: no definition and synthesis not enabled, so we do nothing
            else:
                rule = node.op.definition.data

                if (
                    len(rule) == 1
                    and len(node.qargs) == len(rule[0].qubits) == 1  # to preserve gate order
                    and len(node.cargs) == len(rule[0].clbits) == 0
                ):
                    if node.op.definition.global_phase:
                        dag.global_phase += node.op.definition.global_phase
                    dag.substitute_node(node, rule[0].operation, inplace=True)
                else:
                    decomposition = circuit_to_dag(node.op.definition)
                    dag.substitute_node_with_dag(node, decomposition)

        return dag

    def _should_decompose(self, node: DAGOpNode) -> bool:
        """Call a decomposition pass on this circuit to decompose one level (shallow decompose)."""
        if self.gates_to_decompose is None:  # check if no gates given
            return True

        if not isinstance(self.gates_to_decompose, list):
            gates = [self.gates_to_decompose]
        else:
            gates = self.gates_to_decompose

        strings_list = [s for s in gates if isinstance(s, str)]
        gate_type_list = [g for g in gates if isinstance(g, type)]

        if (
            getattr(node.op, "label", None) is not None
            and node.op.label != ""
            and (  # check if label or label wildcard is given
                node.op.label in gates or any(fnmatch(node.op.label, p) for p in strings_list)
            )
        ):
            return True
        elif node.name in gates or any(  # check if name or name wildcard is given
            fnmatch(node.name, p) for p in strings_list
        ):
            return True
        elif any(isinstance(node.op, op) for op in gate_type_list):  # check if Gate type given
            return True
        else:
            return False


def _node_to_dag(node: DAGOpNode) -> DAGCircuit:
    # create new dag and apply the operation
    dag = DAGCircuit()
    dag.add_qubits(node.qargs)
    dag.add_clbits(node.cargs)
    dag.apply_operation_back(node.op, node.qargs, node.cargs)

    return dag
