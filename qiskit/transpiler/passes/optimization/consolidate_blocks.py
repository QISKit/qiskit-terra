# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Replace each block of consecutive gates by a single Unitary node."""
from __future__ import annotations

import typing

from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer, TwoQubitControlledUDecomposer
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CZGate,
    iSwapGate,
    ECRGate,
    RXXGate,
    RYYGate,
    RZZGate,
    RZXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CPhaseGate,
)

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.passmanager import PassManager
from qiskit._accelerate.consolidate_blocks import consolidate_blocks

from .collect_1q_runs import Collect1qRuns
from .collect_2q_blocks import Collect2qBlocks


if typing.TYPE_CHECKING:
    from qiskit.transpiler.target import Target
    from qiskit.circuit.gate import Gate

KAK_GATE_NAMES = {
    "cz": CZGate(),
    "cx": CXGate(),
    "iswap": iSwapGate(),
    "ecr": ECRGate(),
}

KAK_GATE_PARAM_NAMES = {
    "rxx": RXXGate,
    "rzz": RZZGate,
    "ryy": RYYGate,
    "rzx": RZXGate,
    "cphase": CPhaseGate,
    "crx": CRXGate,
    "cry": CRYGate,
    "crz": CRZGate,
}


class ConsolidateBlocks(TransformationPass):
    """Replace each block of consecutive gates by a single Unitary node.

    Pass to consolidate sequences of uninterrupted gates acting on
    the same qubits into a :class:`.UnitaryGate` node, to be resynthesized later,
    to a potentially more optimal subcircuit. The typical mode of operation of this pass is to run
    the analysis of the input :class:`.DAGCircuit` to find all the two qubit blocks in the circuit
    and then determine based on an internal heuristic whether that block should be consolidated to
    a :class:`.UnitaryGate` or not. However if either ``block_list`` or ``run_list`` are set in the
    property set then this pass will not do its own analysis of the DAG.

    There are two legacy modes of operation for this pass based on whether either ``block_list`` or
    ``run_list`` is set in the property set when this pass is run. ``block_list`` should contain a
    list of lists of node indices where each inner list represents a collection of blocks to be
    potentially consolidated into a :class:`.UnitaryGate`. These blocks can be any number of qubits
    but in previous Qiskit releases' preset pass managers it was typically two and set by the
    :class:`.Collect2qBlocks` pass. There is a also the :class:`.CollectMultiQBlocks` transpiler pass
    which will set ``blocks_list`` with blocks found for an arbitrary number of qubits. The other property
    set key ``run_list`` is a list of lists of "runs" which are single qubit blocks to consolidate. This
    was potentially set by the :class:`.Collect1qRuns` transpiler pass.
    This functionality for "runs" has been mostly superseded by the :class:`.Optimize1qGatesDecomposition`
    transpiler pass and that should typically be used instead for a more thorough and performant
    method of simplifying single qubit runs.

    This pass reads the :class:`.PropertySet` key ``ConsolidateBlocks_qubit_map`` which it uses to
    communicate with recursive worker instances of itself for control-flow operations.  The key
    should never be observable in a user-facing :class:`.PassManager` pipeline (it is only set in
    internal :class:`.PassManager` instances), but the pass may return incorrect results or error if
    another pass sets this key.
    """

    _QUBIT_MAP_KEY = "ConsolidateBlocks_qubit_map"

    def __init__(
        self,
        kak_basis_gate: Gate | None = None,
        force_consolidate: bool = False,
        basis_gates: list[str] | None = None,
        approximation_degree: float | None = 1.0,
        target: Target | None = None,
    ):
        """ConsolidateBlocks initializer.

        If ``kak_basis_gate`` is not ``None`` it will be used as the basis gate for KAK decomposition.
        Otherwise, if ``basis_gates`` is not ``None`` a basis gate will be chosen from this list.
        Otherwise, the basis gate will be :class:`.CXGate`.

        Args:
            kak_basis_gate: Basis gate for KAK decomposition.
            force_consolidate: Force block consolidation.
            basis_gates: Basis gates from which to choose a KAK gate.
            approximation_degree: a float between :math:`[0.0, 1.0]`. Lower approximates more.
            target: The target object for the compilation target backend.
        """
        super().__init__()
        self.basis_gates = None
        self.basis_gate_name = None
        self._approximation_degree = approximation_degree
        # Bypass target if it doesn't contain any basis gates (i.e. it's a _FakeTarget), as this
        # not part of the official target model.
        self.target = target if target is not None and len(target.operation_names) > 0 else None
        if basis_gates is not None:
            self.basis_gates = set(basis_gates)
        self.force_consolidate = force_consolidate
        if kak_basis_gate is not None:
            self.decomposer = TwoQubitBasisDecomposer(kak_basis_gate)
            self.basis_gate_name = kak_basis_gate.name
        elif basis_gates is not None:
            kak_gate = next((gate for gate in KAK_GATE_NAMES if gate in basis_gates), None)
            kak_param_gate = next(
                (gate for gate in KAK_GATE_PARAM_NAMES if gate in basis_gates), None
            )
            if kak_param_gate is not None:
                self.decomposer = TwoQubitControlledUDecomposer(
                    KAK_GATE_PARAM_NAMES[kak_param_gate]
                )
                self.basis_gate_name = kak_param_gate
            elif kak_gate is not None:
                self.decomposer = TwoQubitBasisDecomposer(
                    KAK_GATE_NAMES[kak_gate],
                    basis_fidelity=approximation_degree or 1.0,
                )
                self.basis_gate_name = kak_gate
            else:
                self.decomposer = None
                self.basis_gate_name = "cx"
        elif not force_consolidate:
            self.decomposer = TwoQubitBasisDecomposer(CXGate())
            self.basis_gate_name = "cx"
        else:
            self.decomposer = None
            self.basis_gate_name = "cx"

    def run(self, dag):
        """Run the ConsolidateBlocks pass on `dag`.

        Iterate over each block and replace it with an equivalent Unitary
        on the same wires.
        """
        if not self.force_consolidate and self.decomposer is None:
            return dag

        blocks = self.property_set["block_list"]
        if blocks is not None:
            blocks = [[node._node_id for node in block] for block in blocks]
        runs = self.property_set["run_list"]
        if runs is not None:
            runs = [[node._node_id for node in run] for run in runs]

        qubit_map = self.property_set.get(self._QUBIT_MAP_KEY, None)
        if qubit_map is None:
            qubit_map = list(range(dag.num_qubits()))
        if self.decomposer is not None:
            decomposer = self.decomposer._inner_decomposer
        else:
            decomposer = None
        consolidate_blocks(
            dag,
            decomposer,
            self.basis_gate_name,
            self.force_consolidate,
            target=self.target,
            basis_gates=self.basis_gates,
            blocks=blocks,
            runs=runs,
            qubit_map=qubit_map,
            approximation_degree=self._approximation_degree,
        )
        dag = self._handle_control_flow_ops(dag, qubit_map)

        # Clear collected blocks and runs as they are no longer valid after consolidation
        if "run_list" in self.property_set:
            del self.property_set["run_list"]
        if "block_list" in self.property_set:
            del self.property_set["block_list"]

        return dag

    def _handle_control_flow_ops(self, dag, qubit_map):
        """
        This is similar to transpiler/passes/utils/control_flow.py except that the
        collect blocks is redone for the control flow blocks.
        """

        pass_manager = PassManager()
        if "run_list" in self.property_set:
            pass_manager.append(Collect1qRuns())
            pass_manager.append(Collect2qBlocks())
        pass_manager.append(self)

        for node in dag.control_flow_op_nodes():
            inner_qubit_map = [qubit_map[dag.find_bit(q).index] for q in node.qargs]
            new_op = node.op.replace_blocks(
                pass_manager.run(block, property_set={self._QUBIT_MAP_KEY: inner_qubit_map})
                for block in node.op.blocks
            )
            dag.substitute_node(node, new_op)
        return dag
