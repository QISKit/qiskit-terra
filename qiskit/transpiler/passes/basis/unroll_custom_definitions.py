# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unrolls instructions with custom definitions."""

from qiskit.exceptions import QiskitError
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.circuit import ControlledGate, ControlFlowOp
from qiskit.converters.circuit_to_dag import circuit_to_dag


class UnrollCustomDefinitions(TransformationPass):
    """Unrolls instructions with custom definitions."""

    def __init__(self, equivalence_library, basis_gates=None, target=None, min_qubits=0):
        """Unrolls instructions with custom definitions.

        Args:
            equivalence_library (EquivalenceLibrary): The equivalence library
                which will be used by the BasisTranslator pass. (Instructions in
                this library will not be unrolled by this pass.)
            basis_gates (Optional[list[str]]): Target basis names to unroll to, e.g. ``['u3', 'cx']``.
                Ignored if ``target`` is also specified.
            target (Optional[Target]): The :class:`~.Target` object corresponding to the compilation
                target. When specified, any argument specified for ``basis_gates`` is ignored.
            min_qubits (int): The minimum number of qubits for operations in the input
                dag to translate.
        """

        super().__init__()
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._target = target
        self._min_qubits = min_qubits

    def run(self, dag):
        """Run the UnrollCustomDefinitions pass on `dag`.

        Args:
            dag (DAGCircuit): input dag

        Raises:
            QiskitError: if unable to unroll given the basis due to undefined
            decomposition rules (such as a bad basis) or excessive recursion.

        Returns:
            DAGCircuit: output unrolled dag
        """

        if self._basis_gates is None and self._target is None:
            return dag

        device_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
        if self._target is None:
            device_insts |= set(self._basis_gates)

        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                dag.substitute_node(
                    node,
                    control_flow.map_blocks(self.run, node.op),
                )
                continue

            if getattr(node.op, "_directive", False):
                continue

            if len(node.qargs) < self._min_qubits:
                continue

            controlled_gate_open_ctrl = isinstance(node.op, ControlledGate) and node.op._open_ctrl
            if not controlled_gate_open_ctrl:
                if self._target is not None:
                    inst_supported = self._target.instruction_supported(
                        operation_name=node.op.name,
                        qargs=tuple(dag.find_bit(x).index for x in node.qargs),
                    )
                else:
                    inst_supported = node.name in device_insts

                if inst_supported or self._equiv_lib.has_entry(node.op):
                    continue
            try:
                unrolled = getattr(node.op, "definition", None)
            except TypeError as err:
                raise QiskitError(f"Error decomposing node {node.name}: {err}") from err

            if unrolled is None:
                # opaque node
                raise QiskitError(
                    f"Cannot unroll the circuit to the given basis, {self._basis_gates!s}. "
                    f"Instruction {node.op.name} not found in equivalence library "
                    "and no rule found to expand."
                )

            decomposition = circuit_to_dag(unrolled, copy_operations=False)
            unrolled_dag = self.run(decomposition)
            dag.substitute_node_with_dag(node, unrolled_dag)

        return dag
