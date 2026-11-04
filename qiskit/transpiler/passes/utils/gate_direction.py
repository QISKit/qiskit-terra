# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rearrange the direction of the 2-qubit gate nodes to match the directed coupling map."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit._accelerate.gate_direction import fix_gate_direction_coupling, fix_gate_direction_target


class GateDirection(TransformationPass):
    """Modify asymmetric gates to match the hardware coupling direction.

    This pass supports replacements for the `cx`, `cz`, `ecr`, `swap`, `rzx`, `rxx`, `ryy` and
    `rzz` gates, using the following identities::

                             ┌───┐┌───┐┌───┐
        q_0: ──■──      q_0: ┤ H ├┤ X ├┤ H ├
             ┌─┴─┐  =        ├───┤└─┬─┘├───┤
        q_1: ┤ X ├      q_1: ┤ H ├──■──┤ H ├
             └───┘           └───┘     └───┘


                          global phase: 3π/2
             ┌──────┐           ┌───┐ ┌────┐┌─────┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ─┤ S ├─┤ √X ├┤ Sdg ├┤1     ├┤ H ├
             │  ECR │  =       ┌┴───┴┐├────┤└┬───┬┘│  Ecr │├───┤
        q_1: ┤1     ├     q_1: ┤ Sdg ├┤ √X ├─┤ S ├─┤0     ├┤ H ├
             └──────┘          └─────┘└────┘ └───┘ └──────┘└───┘
        Note: This is done in terms of less-efficient S/SX/Sdg gates instead of the more natural
        `RY(pi /2)` so we have a chance for basis translation to keep things in a discrete basis
        during resynthesis, if that's what's being asked for.


             ┌──────┐          ┌───┐┌──────┐┌───┐
        q_0: ┤0     ├     q_0: ┤ H ├┤1     ├┤ H ├
             │  RZX │  =       ├───┤│  RZX │├───┤
        q_1: ┤1     ├     q_1: ┤ H ├┤0     ├┤ H ├
             └──────┘          └───┘└──────┘└───┘

        cz, swap, rxx, ryy and rzz directions are fixed by reversing their qargs order.

    This pass assumes that the positions of the qubits in the :attr:`.DAGCircuit.qubits` attribute
    are the physical qubit indices. For example if ``dag.qubits[0]`` is qubit 0 in the
    :class:`.CouplingMap` or :class:`.Target`.
    """

    def __init__(self, coupling_map, target=None):
        """GateDirection pass.

        Args:
            coupling_map (CouplingMap): Directed graph representing a coupling map.
            target (Target): The backend target to use for this pass. If this is specified
                it will be used instead of the coupling map
        """
        super().__init__()
        self.coupling_map = coupling_map
        self.target = target

    def run(self, dag):
        """Run the GateDirection pass on `dag`.

        Flips the cx nodes to match the directed coupling map. Modifies the
        input dag.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: The rearranged dag for the coupling map

        Raises:
            TranspilerError: If the circuit cannot be mapped just by flipping the
                cx nodes.
        """
        # Only use "fix_gate_direction_target" if a target exists and target.operation_names
        # is not empty, else use "fix_gate_direction_coupling".
        if self.target is None:
            fix_gate_direction_coupling(dag, set(self.coupling_map.get_edges()))
        elif len(self.target.operation_names) == 0:
            # A  _FakeTarget path, no basis gates, just use the coupling map
            fix_gate_direction_coupling(dag, set(self.target.build_coupling_map().get_edges()))
        else:
            fix_gate_direction_target(dag, self.target)

        return dag
