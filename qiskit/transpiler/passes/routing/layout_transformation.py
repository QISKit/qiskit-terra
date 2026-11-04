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
from __future__ import annotations

import numpy as np

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.coupling import CouplingMap
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import Layout
from qiskit.transpiler.passes.routing.algorithms import ApproximateTokenSwapper
from qiskit.transpiler.target import Target


class LayoutTransformation(TransformationPass):
    """Adds a Swap circuit for a given (partial) permutation to the circuit.

    This circuit is found by a 4-approximation algorithm for Token Swapping.
    More details are available in the routing code.
    """

    def __init__(
        self,
        coupling_map: CouplingMap | Target | None,
        from_layout: Layout | str,
        to_layout: Layout | str,
        seed: int | np.random.Generator | None = None,
        trials=4,
    ):
        """LayoutTransformation initializer.

        Args:
            coupling_map:
                Directed graph representing a coupling map.

            from_layout (Union[Layout, str]):
                The starting layout of qubits onto physical qubits.
                If the type is str, look up `property_set` when this pass runs.

            to_layout (Union[Layout, str]):
                The final layout of qubits on physical qubits.
                If the type is str, look up ``property_set`` when this pass runs.

            seed (Union[int, np.random.default_rng]):
                Seed to use for random trials.

            trials (int):
                How many randomized trials to perform, taking the best circuit as output.
        """
        super().__init__()
        self.from_layout = from_layout
        self.to_layout = to_layout
        if isinstance(coupling_map, Target):
            self.target = coupling_map
            self.coupling_map = self.target.build_coupling_map()
        else:
            self.target = None
            self.coupling_map = coupling_map
        if self.coupling_map is None:
            self.coupling_map = CouplingMap.from_full(len(to_layout))
        graph = self.coupling_map.graph.to_undirected()
        self.token_swapper = ApproximateTokenSwapper(graph, seed)
        self.trials = trials

    def run(self, dag):
        """Apply the specified partial permutation to the circuit.

        Args:
            dag (DAGCircuit): DAG to transform the layout of.

        Returns:
            DAGCircuit: The DAG with transformed layout.

        Raises:
            TranspilerError: if the coupling map or the layout are not compatible with the DAG.
                Or if either of string from/to_layout is not found in `property_set`.
        """
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("LayoutTransform runs on physical circuits only")

        if len(dag.qubits) > len(self.coupling_map.physical_qubits):
            raise TranspilerError("The layout does not match the amount of qubits in the DAG")

        from_layout = self.from_layout
        if isinstance(from_layout, str):
            try:
                from_layout = self.property_set[from_layout]
            except Exception as ex:
                raise TranspilerError(f"No {from_layout} (from_layout) in property_set.") from ex

        to_layout = self.to_layout
        if isinstance(to_layout, str):
            try:
                to_layout = self.property_set[to_layout]
            except Exception as ex:
                raise TranspilerError(f"No {to_layout} (to_layout) in property_set.") from ex

        # Find the permutation between the initial physical qubits and final physical qubits.
        permutation = {
            pqubit: to_layout.get_virtual_bits()[vqubit]
            for vqubit, pqubit in from_layout.get_virtual_bits().items()
        }

        perm_circ = self.token_swapper.permutation_circuit(permutation, self.trials)

        qubits = [dag.qubits[i[0]] for i in sorted(perm_circ.inputmap.items(), key=lambda x: x[0])]
        dag.compose(perm_circ.circuit, qubits=qubits)
        return dag
