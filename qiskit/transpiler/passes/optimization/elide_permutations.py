# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Remove any swap gates in the circuit by pushing it through into a qubit permutation."""

import logging

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.layout import Layout
from qiskit._accelerate import elide_permutations as elide_permutations_rs

logger = logging.getLogger(__name__)


class ElidePermutations(TransformationPass):
    r"""Remove permutation operations from a pre-layout circuit

    This pass is intended to be run before a layout (mapping virtual qubits
    to physical qubits) is set during the transpilation pipeline. This
    pass iterates over the :class:`~.DAGCircuit` and when a :class:`~.SwapGate`
    or :class:`~.PermutationGate` are encountered it permutes the virtual qubits in
    the circuit and removes the swap gate. This will effectively remove any
    :class:`~SwapGate`\s or :class:`~PermutationGate` in the circuit prior to running
    layout. If this pass is run after a layout has been set it will become a no-op
    (and log a warning) as this optimization is not sound after physical qubits are
    selected and there are connectivity constraints to adhere to.

    For tracking purposes this pass sets 3 values in the property set if there
    are any :class:`~.SwapGate` or :class:`~.PermutationGate` objects in the circuit
    and the pass isn't a no-op.

    * ``original_layout``: The trivial :class:`~.Layout` for the input to this pass being run
    * ``original_qubit_indices``: The mapping of qubit objects to positional indices for the state
        of the circuit as input to this pass.
    * ``virtual_permutation_layout``: A :class:`~.Layout` object mapping input qubits to the output
        state after eliding permutations.

    These three properties are needed for the transpiler to track the permutations in the out
    :attr:`.QuantumCircuit.layout` attribute. The elision of permutations is equivalent to a
    ``final_layout`` set by routing and all three of these attributes are needed in the case
    """

    def run(self, dag):
        """Run the ElidePermutations pass on ``dag``.

        Args:
            dag (DAGCircuit): the DAG to be optimized.

        Returns:
            DAGCircuit: the optimized DAG.
        """
        if self.property_set["layout"] is not None:
            logger.warning(
                "ElidePermutations is not valid after a layout has been set. This indicates "
                "an invalid pass manager construction."
            )
            return dag

        result = elide_permutations_rs.run(dag)

        # If the pass did not do anything, the result is None
        if result is None:
            return dag

        # Otherwise, the result is a pair consisting of the rewritten DAGCircuit and the
        # qubit mapping.
        (new_dag, qubit_mapping) = result

        input_qubit_mapping = {qubit: index for index, qubit in enumerate(dag.qubits)}
        self.property_set["original_layout"] = Layout(input_qubit_mapping)
        if self.property_set["original_qubit_indices"] is None:
            self.property_set["original_qubit_indices"] = input_qubit_mapping

        new_layout = Layout({dag.qubits[out]: idx for idx, out in enumerate(qubit_mapping)})
        if current_layout := self.property_set["virtual_permutation_layout"]:
            self.property_set["virtual_permutation_layout"] = new_layout.compose(
                current_layout, dag.qubits
            )
        else:
            self.property_set["virtual_permutation_layout"] = new_layout
        return new_dag
