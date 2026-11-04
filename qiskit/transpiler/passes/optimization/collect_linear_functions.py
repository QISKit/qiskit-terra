# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Replace each sequence of CX and SWAP gates by a single LinearFunction gate."""

from functools import partial

from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collect_using_filter_function,
    collapse_to_operation,
)


class CollectLinearFunctions(CollectAndCollapse):
    """Collect blocks of linear gates (:class:`.CXGate` and :class:`.SwapGate` gates)
    and replaces them by linear functions (:class:`.LinearFunction`)."""

    def __init__(
        self,
        do_commutative_analysis=False,
        split_blocks=True,
        min_block_size=2,
        split_layers=False,
        collect_from_back=False,
        max_block_width=None,
    ):
        """CollectLinearFunctions initializer.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
            split_blocks (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            min_block_size (int): specifies the minimum number of gates in the block
                for the block to be collected.
            split_layers (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            collect_from_back (bool): specifies if blocks should be collected started
                from the end of the circuit.
            max_block_width (int | None): specifies the maximum width of the block
                (that is, the number of qubits over which the block is defined)
                for the block to be collected.
        """

        collect_function = partial(
            collect_using_filter_function,
            filter_function=_is_linear_gate,
            split_blocks=split_blocks,
            min_block_size=min_block_size,
            split_layers=split_layers,
            collect_from_back=collect_from_back,
            max_block_width=max_block_width,
        )
        collapse_function = partial(
            collapse_to_operation, collapse_function=_collapse_to_linear_function
        )

        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )


def _is_linear_gate(node):
    """Specifies whether a node holds a linear gate."""
    return node.name in ("cx", "swap") and getattr(node, "condition", None) is None


def _collapse_to_linear_function(circuit):
    """Specifies how to construct a ``LinearFunction`` from a quantum circuit (that must
    consist of linear gates only)."""
    return LinearFunction(circuit)
