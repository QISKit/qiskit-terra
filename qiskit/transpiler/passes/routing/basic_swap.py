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

"""Map (with minimum effort) a DAGCircuit onto a ``coupling_map`` adding swap gates."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.layout import Layout
from qiskit.circuit.library.standard_gates import SwapGate
from qiskit.transpiler.target import Target
from qiskit.transpiler.passes.layout import disjoint_utils


class BasicSwap(TransformationPass):
    """Map (with minimum effort) a DAGCircuit onto a ``coupling_map`` adding swap gates.

    The basic mapper is a minimum effort to insert swap gates to map the DAG onto
    a coupling map. When a cx is not in the coupling map possibilities, it inserts
    one or more swaps in front to make it compatible.
    """

    def __init__(self, coupling_map, fake_run=False):
        """BasicSwap initializer.

        Args:
            coupling_map (Union[CouplingMap, Target]): Directed graph representing a coupling map.
            fake_run (bool): if true, it will only pretend to do routing, i.e., no
                swap is effectively added.
        """
        super().__init__()
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        self.fake_run = fake_run

    def run(self, dag):
        """Run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG, or if the ``coupling_map=None``.
        """
        if self.fake_run:
            return self._fake_run(dag)

        new_dag = dag.copy_empty_like()

        if self.coupling_map is None:
            raise TranspilerError("BasicSwap cannot run with coupling_map=None")

        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")
        disjoint_utils.require_layout_isolated_to_component(
            dag, self.coupling_map if self.target is None else self.target
        )
        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]

            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(
                            SwapGate(), (qubit_1, qubit_2), cargs=(), check=False
                        )

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)

        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            # The "final layout" can be thought of as a "comes from" permutation that you apply at
            # the end of the circuit to invert the routing.  So if there's an existing one, what we
            # apply at the end of the circuit needs to set the circuit qubits so they "come from"
            # the previous one, then those "come from" the one we've just added.
            self.property_set["final_layout"] = self.property_set["final_layout"].compose(
                current_layout, dag.qubits
            )

        return new_dag

    def _fake_run(self, dag):
        """Do a fake run the BasicSwap pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to improve initial layout.

        Returns:
            DAGCircuit: The same DAG.

        Raises:
            TranspilerError: if the coupling map or the layout are not
            compatible with the DAG.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("Basic swap runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        canonical_register = dag.qregs["q"]
        trivial_layout = Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer["graph"]
            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])

        if self.property_set["final_layout"] is None:
            self.property_set["final_layout"] = current_layout
        else:
            # The "final layout" can be thought of as a "comes from" permutation that you apply at
            # the end of the circuit to invert the routing.  So if there's an existing one, what we
            # apply at the end of the circuit needs to set the circuit qubits so they "come from"
            # the previous one, then those "come from" the one we've just added.
            self.property_set["final_layout"] = self.property_set["final_layout"].compose(
                current_layout, dag.qubits
            )
        return dag
