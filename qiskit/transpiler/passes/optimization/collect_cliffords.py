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


"""Replace each sequence of Clifford gates by a single Clifford gate."""

from functools import partial

from qiskit.exceptions import QiskitError
from qiskit.transpiler.passes.optimization.collect_and_collapse import (
    CollectAndCollapse,
    collect_using_filter_function,
    collapse_to_operation,
)

from qiskit.quantum_info.operators import Clifford
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _BASIS_1Q, _BASIS_2Q


class CollectCliffords(CollectAndCollapse):
    """Collects blocks of Clifford gates and replaces them by a :class:`~qiskit.quantum_info.Clifford`
    object.
    """

    def __init__(
        self,
        do_commutative_analysis=False,
        split_blocks=True,
        min_block_size=2,
        split_layers=False,
        collect_from_back=False,
        matrix_based=False,
        max_block_width=None,
    ):
        """CollectCliffords initializer.

        Args:
            do_commutative_analysis (bool): if True, exploits commutativity relations
                between nodes.
            split_blocks (bool): if True, splits collected blocks into sub-blocks
                over disjoint qubit subsets.
            min_block_size (int): specifies the minimum number of gates in the block
                for the block to be collected.
            split_layers (bool): if True, splits collected blocks into layers of
                non-overlapping instructions (depth-1 sub-blocks).
            collect_from_back (bool): specifies if blocks should be collected started
                from the end of the circuit.
            matrix_based (bool): specifies whether to collect unitary gates
               which are Clifford gates only for certain parameters (based on their unitary matrix).
            max_block_width (int | None): specifies the maximum width of the block
                (that is, the number of qubits over which the block is defined)
                for the block to be collected.
        """

        collect_function = partial(
            collect_using_filter_function,
            filter_function=partial(_is_clifford_gate, matrix_based=matrix_based),
            split_blocks=split_blocks,
            min_block_size=min_block_size,
            split_layers=split_layers,
            collect_from_back=collect_from_back,
            max_block_width=max_block_width,
        )
        collapse_function = partial(collapse_to_operation, collapse_function=_collapse_to_clifford)

        super().__init__(
            collect_function=collect_function,
            collapse_function=collapse_function,
            do_commutative_analysis=do_commutative_analysis,
        )


clifford_gate_names = (
    list(_BASIS_1Q.keys())
    + list(_BASIS_2Q.keys())
    + ["clifford", "linear_function", "pauli", "permutation"]
)


def _is_clifford_gate(node, matrix_based=False):
    """Specifies whether a node holds a clifford gate."""
    if getattr(node.op, "_condition", None) is not None:
        return False
    if node.op.name in clifford_gate_names:
        return True

    if not matrix_based:
        return False

    try:
        Clifford(node.op)
        return True
    except QiskitError:
        return False


def _collapse_to_clifford(circuit):
    """Specifies how to construct a ``Clifford`` from a quantum circuit (that must
    consist of Clifford gates only)."""
    return Clifford(circuit)
