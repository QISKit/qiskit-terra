# This code is part of Qiskit.
#
# (C) Copyright IBM 2025
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Contract control-flow operations that contain idle wires."""

from qiskit.circuit import Qubit, Clbit, QuantumCircuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import TransformationPass


class ContractIdleWiresInControlFlow(TransformationPass):
    """Remove idle qubits from control-flow operations of a :class:`.DAGCircuit`."""

    def run(self, dag):
        # `control_flow_op_nodes` is eager and doesn't borrow; we're mutating the DAG in the loop.
        for node in dag.control_flow_op_nodes() or []:
            inst = node._to_circuit_instruction()
            new_inst = _contract_control_flow(inst)
            if new_inst is inst:
                # No top-level contraction; nothing to do.
                continue
            replacement = DAGCircuit()
            # Dictionaries to retain insertion order for reproducibility, and because we can
            # then reuse them as mapping dictionaries.
            qubits, clbits, vars_ = {}, {}, {}
            for _, _, wire in dag.edges(node):
                if isinstance(wire, Qubit):
                    qubits[wire] = wire
                elif isinstance(wire, Clbit):
                    clbits[wire] = wire
                else:
                    vars_[wire] = wire
            replacement.add_qubits(list(qubits))
            replacement.add_clbits(list(clbits))
            for var in vars_:
                replacement.add_captured_var(var)
            replacement._apply_op_node_back(DAGOpNode.from_instruction(new_inst))
            # The replacement DAG is defined over all the same qubits, but with the correct
            # qubits now explicitly marked as idle, so everything gets linked up correctly.
            dag.substitute_node_with_dag(node, replacement, wires=qubits | clbits | vars_)
        return dag


def _contract_control_flow(inst):
    """Contract a `CircuitInstruction` containing a control-flow operation.

    Returns the input object by the same reference if there's no contraction to be done at the call
    site, though nested control-flow ops may have been contracted in place."""
    op = inst.operation
    idle = set(inst.qubits)
    for block in op.blocks:
        qubit_map = dict(zip(block.qubits, inst.qubits))
        for i, inner in enumerate(block.data):
            if inner.is_control_flow():
                # In `QuantumCircuit` it's easy to replace an instruction with a narrower one, so it
                # doesn't matter much if this is replacing it with itself.
                block.data[i] = inner = _contract_control_flow(inner)
            for qubit in inner.qubits:
                idle.discard(qubit_map[qubit])
    # If a box, we still want the prior side-effect of contracting any internal control-flow
    # operations (optimisations are still valid _within_ a box), but we don't want to contract the
    # box itself.  If there's no idle qubits, we're also done here.
    if not idle or inst.name == "box":
        return inst

    def contract(block):
        out = QuantumCircuit(
            name=block.name,
            global_phase=block.global_phase,
            metadata=block.metadata,
            captures=block.iter_captures(),
        )
        out.add_bits(
            [
                block_qubit
                for (block_qubit, inst_qubit) in zip(block.qubits, inst.qubits)
                if inst_qubit not in idle
            ]
        )
        out.add_bits(block.clbits)
        for creg in block.cregs:
            out.add_register(creg)
        # Control-flow ops can only have captures and locals, and we already added the captures.
        for var in block.iter_declared_vars():
            out.add_uninitialized_var(var)
        for stretch in block.iter_declared_stretches():
            out.add_stretch(stretch)
        for inner in block:
            out._append(inner)
        return out

    return inst.replace(
        operation=op.replace_blocks(contract(block) for block in op.blocks),
        qubits=[qubit for qubit in inst.qubits if qubit not in idle],
    )
