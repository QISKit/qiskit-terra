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

"""Optimize annotated operations on a circuit."""


from qiskit.circuit.controlflow import CONTROL_FLOW_OP_NAMES
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.annotated_operation import AnnotatedOperation, _canonicalize_modifiers
from qiskit.circuit import (
    QuantumCircuit,
    Instruction,
    EquivalenceLibrary,
    ControlledGate,
    Operation,
    ControlFlowOp,
)
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passes.utils import control_flow
from qiskit.transpiler.target import Target
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.exceptions import TranspilerError


class OptimizeAnnotated(TransformationPass):
    """Optimization pass on circuits with annotated operations.

    Implemented optimizations:

    * For each annotated operation, converting the list of its modifiers to a canonical form.
      For example, consecutively applying ``inverse()``, ``control(2)`` and ``inverse()``
      is equivalent to applying ``control(2)``.

    * Removing annotations when possible.
      For example, ``AnnotatedOperation(SwapGate(), [InverseModifier(), InverseModifier()])``
      is equivalent to ``SwapGate()``.

    * Recursively combining annotations.
      For example, if ``g1 = AnnotatedOperation(SwapGate(), InverseModifier())`` and
      ``g2 = AnnotatedOperation(g1, ControlModifier(2))``, then ``g2`` can be replaced with
      ``AnnotatedOperation(SwapGate(), [InverseModifier(), ControlModifier(2)])``.

    * Applies conjugate reduction to annotated operations. As an example,
      ``control - [P -- Q -- P^{-1}]`` can be rewritten as ``P -- control - [Q] -- P^{-1}``,
      that is, only the middle part needs to be controlled. This also works for inverse
      and power modifiers.

    """

    def __init__(
        self,
        target: Target | None = None,
        equivalence_library: EquivalenceLibrary | None = None,
        basis_gates: list[str] | None = None,
        recurse: bool = True,
        do_conjugate_reduction: bool = True,
    ):
        """
        OptimizeAnnotated initializer.

        Args:
            target:  the backend target to use for this pass.
            equivalence_library: The equivalence library used
                (instructions in this library will not be optimized by this pass).
            basis_gates:  target basis names to unroll to, e.g. `['u3', 'cx']`
                (instructions in this list will not be optimized by this pass).
                Ignored if ``target`` is also specified.
            recurse: By default, when either ``target`` or ``basis_gates`` is specified,
                the pass recursively descends into gate definitions (and the recursion is
                not applied when neither is specified since such objects do not need to
                be synthesized). Setting this value to ``False`` precludes the recursion in
                every case.
            do_conjugate_reduction: controls whether conjugate reduction should be performed.
        """
        super().__init__()

        self._target = target
        self._equiv_lib = equivalence_library
        self._basis_gates = basis_gates
        self._do_conjugate_reduction = do_conjugate_reduction

        self._top_level_only = not recurse or (self._basis_gates is None and self._target is None)

        if not self._top_level_only and self._target is None:
            basic_insts = {"measure", "reset", "barrier", "snapshot", "delay", "store"}
            self._device_insts = basic_insts | set(self._basis_gates)

    def run(self, dag: DAGCircuit):
        """Run the OptimizeAnnotated pass on `dag`.

        Args:
            dag: input dag.

        Returns:
            Output dag with higher-level operations optimized.

        Raises:
            TranspilerError: when something goes wrong.

        """
        dag, _ = self._run_inner(dag)
        return dag

    def _run_inner(self, dag) -> tuple[DAGCircuit, bool]:
        """
        Optimizes annotated operations.
        Returns True if did something.
        """
        # Fast return
        if self._top_level_only:
            op_names = dag.count_ops(recurse=False)
            if "annotated" not in op_names and not CONTROL_FLOW_OP_NAMES.intersection(op_names):
                return dag, False

        # Handle control-flow
        for node in dag.op_nodes():
            if isinstance(node.op, ControlFlowOp):
                dag.substitute_node(
                    node,
                    control_flow.map_blocks(self.run, node.op),
                )

        # First, optimize every node in the DAG.
        dag, opt1 = self._canonicalize(dag)

        opt2 = False
        if not self._top_level_only:
            # Second, recursively descend into definitions.
            # Note that it is important to recurse only after the optimization methods have been run,
            # as they may remove annotated gates.
            dag, opt2 = self._recurse(dag)

        opt3 = False
        if not self._top_level_only and self._do_conjugate_reduction:
            dag, opt3 = self._conjugate_reduction(dag)

        return dag, opt1 or opt2 or opt3

    def _canonicalize(self, dag) -> tuple[DAGCircuit, bool]:
        """
        Combines recursive annotated operations and canonicalizes modifiers.
        Returns True if did something.
        """

        did_something = False
        for node in dag.op_nodes(op=AnnotatedOperation):
            modifiers = []
            cur = node.op
            while isinstance(cur, AnnotatedOperation):
                modifiers = cur.modifiers + modifiers
                cur = cur.base_op
            canonical_modifiers = _canonicalize_modifiers(modifiers)
            if len(canonical_modifiers) > 0:
                # this is still an annotated operation
                node.op.base_op = cur
                node.op.modifiers = canonical_modifiers
            else:
                # no need for annotated operations
                dag.substitute_node(node, cur)
            did_something = True
        return dag, did_something

    def _conjugate_decomposition(
        self, dag: DAGCircuit
    ) -> tuple[DAGCircuit, DAGCircuit, DAGCircuit] | None:
        """
        Decomposes a circuit ``A`` into 3 sub-circuits ``P``, ``Q``, ``R`` such that
        ``A = P -- Q -- R`` and ``R = P^{-1}``.

        This is accomplished by iteratively finding inverse nodes at the front and at the back of the
        circuit.
        """

        front_block = []  # nodes collected from the front of the circuit (aka P)
        back_block = []  # nodes collected from the back of the circuit (aka R)

        # Stores in- and out- degree for each node. These degrees are computed at the start of this
        # function and are updated when nodes are collected into front_block or into back_block.
        in_degree = {}
        out_degree = {}

        # We use dicts to track for each qubit a DAG node at the front of the circuit that involves
        # this qubit and a DAG node at the end of the circuit that involves this qubit (when exist).
        # Note that for the DAGCircuit structure for each qubit there can be at most one such front
        # and such back node.
        # This allows for an efficient way to find an inverse pair of gates (one from the front and
        # one from the back of the circuit).
        # A qubit that was never examined does not appear in these dicts, and a qubit that was examined
        # but currently is not involved at the front (resp. at the back) of the circuit has the value of
        # None.
        front_node_for_qubit = {}
        back_node_for_qubit = {}

        # Keep the set of nodes that have been moved either to front_block or to back_block
        processed_nodes = set()

        # Keep the set of qubits that are involved in nodes at the front or at the back of the circuit.
        # When looking for inverse pairs of gates we will only iterate over these qubits.
        active_qubits = set()

        # Keep pairs of nodes for which the inverse check was performed and the nodes
        # were found to be not inverse to each other (memoization).
        checked_node_pairs = set()

        # compute in- and out- degree for every node
        # also update information for nodes at the start and at the end of the circuit
        for node in dag.op_nodes():
            preds = list(dag.op_predecessors(node))
            in_degree[node] = len(preds)
            if len(preds) == 0:
                for q in node.qargs:
                    front_node_for_qubit[q] = node
                    active_qubits.add(q)
            succs = list(dag.op_successors(node))
            out_degree[node] = len(succs)
            if len(succs) == 0:
                for q in node.qargs:
                    back_node_for_qubit[q] = node
                    active_qubits.add(q)

        # iterate while there is a possibility to find more inverse pairs
        while len(active_qubits) > 0:
            to_check = active_qubits.copy()
            active_qubits.clear()

            # For each qubit q, check whether the gate at the front of the circuit that involves q
            # and the gate at the end of the circuit that involves q are inverse
            for q in to_check:

                if (front_node := front_node_for_qubit.get(q, None)) is None:
                    continue
                if (back_node := back_node_for_qubit.get(q, None)) is None:
                    continue

                # front_node or back_node could be already collected when considering other qubits
                if front_node in processed_nodes or back_node in processed_nodes:
                    continue

                # it is possible that the same node is both at the front and at the back,
                # it should not be collected
                if front_node == back_node:
                    continue

                # have been checked before
                if (front_node, back_node) in checked_node_pairs:
                    continue

                # fast check based on the arguments
                if front_node.qargs != back_node.qargs or front_node.cargs != back_node.cargs:
                    continue

                # in the future we want to include a more precise check whether a pair
                # of nodes are inverse
                if front_node.op == back_node.op.inverse():
                    # update front_node_for_qubit and back_node_for_qubit
                    for q in front_node.qargs:
                        front_node_for_qubit[q] = None
                    for q in back_node.qargs:
                        back_node_for_qubit[q] = None

                    # see which other nodes become at the front and update information
                    for node in dag.op_successors(front_node):
                        if node not in processed_nodes:
                            in_degree[node] -= 1
                            if in_degree[node] == 0:
                                for q in node.qargs:
                                    front_node_for_qubit[q] = node
                                    active_qubits.add(q)

                    # see which other nodes become at the back and update information
                    for node in dag.op_predecessors(back_node):
                        if node not in processed_nodes:
                            out_degree[node] -= 1
                            if out_degree[node] == 0:
                                for q in node.qargs:
                                    back_node_for_qubit[q] = node
                                    active_qubits.add(q)

                    # collect and mark as processed
                    front_block.append(front_node)
                    back_block.append(back_node)
                    processed_nodes.add(front_node)
                    processed_nodes.add(back_node)

                else:
                    checked_node_pairs.add((front_node, back_node))

        # if nothing is found, return None
        if len(front_block) == 0:
            return None

        # create the output DAGs
        front_circuit = dag.copy_empty_like()
        middle_circuit = dag.copy_empty_like()
        back_circuit = dag.copy_empty_like()
        front_circuit.global_phase = 0
        back_circuit.global_phase = 0

        for node in front_block:
            front_circuit.apply_operation_back(node.op, node.qargs, node.cargs)

        for node in back_block:
            back_circuit.apply_operation_front(node.op, node.qargs, node.cargs)

        for node in dag.op_nodes():
            if node not in processed_nodes:
                middle_circuit.apply_operation_back(node.op, node.qargs, node.cargs)

        return front_circuit, middle_circuit, back_circuit

    def _conjugate_reduce_op(
        self, op: AnnotatedOperation, base_decomposition: tuple[DAGCircuit, DAGCircuit, DAGCircuit]
    ) -> Operation:
        """
        We are given an annotated-operation ``op = M [ B ]`` (where ``B`` is the base operation and
        ``M`` is the list of modifiers) and the "conjugate decomposition" of the definition of ``B``,
        i.e. ``B = P * Q * R``, with ``R = P^{-1}`` (with ``P``, ``Q`` and ``R`` represented as
        ``DAGCircuit`` objects).

        Let ``IQ`` denote a new custom instruction with definitions ``Q``.

        We return the operation ``op_new`` which a new custom instruction with definition
        ``P * A * R``, where ``A`` is a new annotated-operation with modifiers ``M`` and
        base gate ``IQ``.
        """
        p_dag, q_dag, r_dag = base_decomposition

        q_instr = Instruction(
            name="iq", num_qubits=op.base_op.num_qubits, num_clbits=op.base_op.num_clbits, params=[]
        )
        q_instr.definition = dag_to_circuit(q_dag)

        op_new = Instruction(
            "optimized", num_qubits=op.num_qubits, num_clbits=op.num_clbits, params=[]
        )
        num_control_qubits = op.num_qubits - op.base_op.num_qubits

        circ = QuantumCircuit(op.num_qubits, op.num_clbits)
        qubits = circ.qubits
        circ.compose(
            dag_to_circuit(p_dag), qubits[num_control_qubits : op.num_qubits], inplace=True
        )
        circ.append(
            AnnotatedOperation(base_op=q_instr, modifiers=op.modifiers), range(op.num_qubits)
        )
        circ.compose(
            dag_to_circuit(r_dag), qubits[num_control_qubits : op.num_qubits], inplace=True
        )
        op_new.definition = circ
        return op_new

    def _conjugate_reduction(self, dag) -> tuple[DAGCircuit, bool]:
        """
        Looks for annotated operations whose base operation has a nontrivial conjugate decomposition.
        In such cases, the modifiers of the annotated operation can be moved to the "middle" part of
        the decomposition.

        Returns the modified DAG and whether it did something.
        """
        did_something = False
        for node in dag.op_nodes(op=AnnotatedOperation):
            base_op = node.op.base_op
            if not self._skip_definition(base_op):
                base_dag = circuit_to_dag(base_op.definition, copy_operations=False)
                base_decomposition = self._conjugate_decomposition(base_dag)
                if base_decomposition is not None:
                    new_op = self._conjugate_reduce_op(node.op, base_decomposition)
                    dag.substitute_node(node, new_op)
                    did_something = True
        return dag, did_something

    def _skip_definition(self, op: Operation) -> bool:
        """
        Returns True if we should not recurse into a gate's definition.
        """
        # Similar to HighLevelSynthesis transpiler pass, we do not recurse into a gate's
        # `definition` for a gate that is supported by the target or in equivalence library.

        controlled_gate_open_ctrl = isinstance(op, ControlledGate) and op._open_ctrl
        if not controlled_gate_open_ctrl:
            inst_supported = (
                self._target.instruction_supported(operation_name=op.name)
                if self._target is not None
                else op.name in self._device_insts
            )
            if inst_supported or (self._equiv_lib is not None and self._equiv_lib.has_entry(op)):
                return True
        return False

    def _recursively_process_definitions(self, op: Operation) -> bool:
        """
        Recursively applies optimizations to op's definition (or to op.base_op's
        definition if op is an annotated operation).
        Returns True if did something.
        """

        # If op is an annotated operation, we descend into its base_op
        if isinstance(op, AnnotatedOperation):
            return self._recursively_process_definitions(op.base_op)

        if self._skip_definition(op):
            return False

        try:
            # extract definition
            definition = op.definition
        except TypeError as err:
            raise TranspilerError(
                f"OptimizeAnnotated was unable to extract definition for {op.name}: {err}"
            ) from err
        except AttributeError:
            # definition is None
            definition = None

        if definition is None:
            raise TranspilerError(f"OptimizeAnnotated was unable to optimize {op}.")

        definition_dag = circuit_to_dag(definition, copy_operations=False)
        definition_dag, opt = self._run_inner(definition_dag)

        if opt:
            # We only update a gate's definition if it was actually changed.
            # This is important to preserve non-annotated singleton gates.
            op.definition = dag_to_circuit(definition_dag)

        return opt

    def _recurse(self, dag) -> tuple[DAGCircuit, bool]:
        """
        Recursively handles gate definitions.
        Returns True if did something.
        """
        did_something = False

        for node in dag.op_nodes():
            opt = self._recursively_process_definitions(node.op)
            did_something = did_something or opt

        return dag, did_something
