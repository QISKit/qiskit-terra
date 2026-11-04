# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Transform a circuit with virtual qubits into a circuit with physical qubits."""

from __future__ import annotations

from qiskit._accelerate.apply_layout import apply_layout, update_layout
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.layout import TranspileLayout


class ApplyLayout(TransformationPass):
    """Apply or update the mapping of virtual qubits to physical qubits in the :class:`.DAGCircuit`.

    A "layout" in Qiskit is a mapping of virtual qubits (as the user typically creates circuits in
    terms of) to the physical qubits used to represent them on hardware.

    This pass has two modes of operation, depending on the state of the ``layout`` and
    ``post_layout`` keys in the :class:`.PropertySet`:

    1. Standard operation: ``post_layout`` is not set.  This takes in a :class:`.DAGCircuit` defined
       over virtual qubits, and rewrites it in terms of physical qubits.  In this case, the
       ``layout`` field must have been chosen by a layout pass (for example :class:`.SetLayout` or
       :class:`.VF2Layout`), and both it and the :class:`.DAGCircuit` must have been expanded with
       ancillas (see :class:`.EnlargeWithAncilla` and :class:`.FullAncillaAllocation`).

    2. Improving a layout: ``post_layout`` is set (such as by :class:`.VF2PostLayout`).  In this
       case, the ``post_layout`` must already be the correct size.  It is interpreted as an
       _additional_ relabelling on top of the relabelling that is already applied to the input
       :class:`.DAGCircuit`.

       After the pass runs, the ``layout`` field will be updated to represent the composition of the
       two relabellings, as will the :class:`.DAGCircuit` and any final permutation.  The
       ``post_layout`` field will be removed.
    """

    def run(self, dag):
        """Run the ApplyLayout pass on ``dag``.

        Args:
            dag (DAGCircuit): DAG to map.

        Returns:
            DAGCircuit: A mapped DAG (with physical qubits).

        Raises:
            TranspilerError: if no layout is found in ``property_set`` or no full physical qubits.
        """
        if (post_layout := self.property_set.pop("post_layout", None)) is not None:
            return self._apply_post_layout(dag, post_layout)
        return self._apply_layout(dag)

    def _apply_layout(self, dag):
        # We're going to put this layout back later, potentially after ancilla expansion, we just
        # need it gone to be able to use `TranspileLayout.from_property_set` to normalise any
        # `virtual_permutation_layout` and `final_layout` that might be there.
        layout = self.property_set.pop("layout", None)
        if layout is None:
            raise TranspilerError(
                "No 'layout' is found in property_set. Please run a Layout pass in advance."
            )
        if len(layout) != (1 + max(layout.get_physical_bits(), default=-1)):
            raise TranspilerError("The 'layout' must be full (with ancilla).")
        num_physical_qubits = len(layout)

        prev_layout = TranspileLayout.from_property_set(dag, self.property_set)
        if prev_layout is None:
            permutation = None
            num_virtual_qubits = self.property_set.get("num_input_qubits", None)
        else:
            permutation = prev_layout.routing_permutation()
            num_virtual_qubits = prev_layout._input_qubit_count
        if num_virtual_qubits is None:
            num_virtual_qubits = dag.num_qubits()
        transpile_layout = apply_layout(
            dag,
            num_virtual_qubits,
            num_physical_qubits,
            [layout[qubit] for qubit in dag.qubits],
            permutation,
        )
        transpile_layout.write_into_property_set(self.property_set)
        return dag

    def _apply_post_layout(self, dag, post_layout):
        transpile_layout = TranspileLayout.from_property_set(dag, self.property_set)
        transpile_layout = update_layout(
            dag, transpile_layout, [post_layout[q] for q in dag.qubits]
        )
        transpile_layout.write_into_property_set(self.property_set)
        return dag
