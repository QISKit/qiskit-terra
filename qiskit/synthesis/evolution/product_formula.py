# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A product formula base for decomposing non-commuting operator exponentials."""

from __future__ import annotations

import warnings
import itertools
from collections.abc import Callable, Sequence
from collections import defaultdict
from itertools import combinations
import typing
import numpy as np
import rustworkx as rx

from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.circuit.quantumcircuit import ParameterValueType
from qiskit.quantum_info import SparsePauliOp, SparseObservable
from qiskit._accelerate.circuit_library import pauli_evolution
import qiskit.quantum_info

from .evolution_synthesis import EvolutionSynthesis

if typing.TYPE_CHECKING:
    from qiskit.circuit.library import PauliEvolutionGate

SparsePauliLabel = tuple[str, list[int], ParameterValueType]


class ProductFormula(EvolutionSynthesis):
    """Product formula base class for the decomposition of non-commuting operator exponentials.

    :obj:`.LieTrotter` and :obj:`.SuzukiTrotter` inherit from this class.
    """

    def __init__(
        self,
        order: int,
        reps: int = 1,
        insert_barriers: bool = False,
        cx_structure: str = "chain",
        atomic_evolution: (
            Callable[[QuantumCircuit, qiskit.quantum_info.Pauli | SparsePauliOp, float], None]
            | None
        ) = None,
        wrap: bool = False,
        preserve_order: bool = True,
        *,
        atomic_evolution_sparse_observable: bool = False,
    ) -> None:
        r"""
        Args:
            order: The order of the product formula.
            reps: The number of time steps.
            insert_barriers: Whether to insert barriers between the atomic evolutions.
            cx_structure: How to arrange the CX gates for the Pauli evolutions, can be
                ``"chain"``, where next neighbor connections are used, or ``"fountain"``,
                where all qubits are connected to one. This only takes effect when
                ``atomic_evolution is None``.
            atomic_evolution: A function to apply the evolution of a single
                :class:`~.quantum_info.Pauli`, or :class:`.SparsePauliOp` of only commuting terms,
                to a circuit. The function takes in three arguments: the circuit to append the
                evolution to, the Pauli operator to evolve, and the evolution time. By default, a
                single Pauli evolution is decomposed into a chain of ``CX`` gates and a single
                ``RZ`` gate.
            wrap: Whether to wrap the atomic evolutions into custom gate objects. Note that setting
                this to ``True`` is slower than ``False``. This only takes effect when
                ``atomic_evolution is None``.
            preserve_order: If ``False``, allows reordering the terms of the operator to
                potentially yield a shallower evolution circuit. Not relevant
                when synthesizing operator with a single term.
            atomic_evolution_sparse_observable: If a custom ``atomic_evolution`` is passed,
                which does not yet support :class:`.SparseObservable`\ s as input, set this
                argument to ``False`` to automatically apply a conversion to :class:`.SparsePauliOp`.
                This argument is supported until Qiskit 2.2, at which point all atomic evolutions
                are required to support :class:`.SparseObservable`\ s as input.
        """
        super().__init__()
        self.order = order
        self.reps = reps
        self.insert_barriers = insert_barriers
        self.preserve_order = preserve_order

        # user-provided atomic evolution, stored for serialization
        self._atomic_evolution = atomic_evolution

        if cx_structure not in ["chain", "fountain"]:
            raise ValueError(f"Unsupported CX structure: {cx_structure}")

        self._cx_structure = cx_structure
        self._wrap = wrap

        # if atomic evolution is not provided, set a default
        if atomic_evolution is None:
            self.atomic_evolution = None
        else:
            self.atomic_evolution = wrap_custom_atomic_evolution(
                atomic_evolution, atomic_evolution_sparse_observable
            )

    def expand(
        self, evolution: PauliEvolutionGate
    ) -> list[tuple[str, tuple[int], ParameterValueType]]:
        """Apply the product formula to expand the Hamiltonian in the evolution gate.

        Args:
            evolution: The :class:`.PauliEvolutionGate`, whose Hamiltonian we expand.

        Returns:
            A list of Pauli rotations in a sparse format, where each element is
            ``(paulistring, qubits, coefficient)``. For example, the Lie-Trotter expansion
            of ``H = XI + ZZ`` would return ``[("X", [1], 1), ("ZZ", [0, 1], 1)]``.
        """
        raise NotImplementedError(
            f"The method ``expand`` is not implemented for {self.__class__}. Implement it to "
            f"automatically enable the call to {self.__class__}.synthesize."
        )

    def synthesize(self, evolution: PauliEvolutionGate) -> QuantumCircuit:
        """Synthesize a :class:`.PauliEvolutionGate`.

        Args:
            evolution: The evolution gate to synthesize.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        pauli_rotations = self.expand(evolution)
        num_qubits = evolution.num_qubits

        if self._wrap or self._atomic_evolution is not None:
            # this is the slow path, where each Pauli evolution is constructed in Rust
            # separately and then wrapped into a gate object
            circuit = self._custom_evolution(num_qubits, pauli_rotations)
        else:
            # this is the fast path, where the whole evolution is constructed Rust-side
            cx_fountain = self._cx_structure == "fountain"
            data = pauli_evolution(num_qubits, pauli_rotations, self.insert_barriers, cx_fountain)
            circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True)

        return circuit

    @property
    def settings(self) -> dict[str, typing.Any]:
        """Return the settings in a dictionary, which can be used to reconstruct the object.

        Returns:
            A dictionary containing the settings of this product formula.

        Raises:
            NotImplementedError: If a custom atomic evolution is set, which cannot be serialized.
        """
        if self._atomic_evolution is not None:
            raise NotImplementedError(
                "Cannot serialize a product formula with a custom atomic evolution."
            )

        return {
            "order": self.order,
            "reps": self.reps,
            "insert_barriers": self.insert_barriers,
            "cx_structure": self._cx_structure,
            "wrap": self._wrap,
            "preserve_order": self.preserve_order,
        }

    def _custom_evolution(self, num_qubits, pauli_rotations):
        """Implement the evolution for the non-standard path.

        This is either because a user-defined atomic evolution is given, or because the evolution
        of individual Paulis needs to be wrapped in gates.
        """
        circuit = QuantumCircuit(num_qubits)
        cx_fountain = self._cx_structure == "fountain"

        num_paulis = len(pauli_rotations)
        for i, pauli_rotation in enumerate(pauli_rotations):
            if self._atomic_evolution is not None:
                # use the user-provided evolution with a global operator
                operator = SparseObservable.from_sparse_list([pauli_rotation], num_qubits)
                self.atomic_evolution(circuit, operator, time=1)  # time is inside the Pauli coeff

            else:  # this means self._wrap is True
                # we create a local sparse Pauli representation such that the operator
                # does not span over all qubits of the circuit
                pauli_string, qubits, coeff = pauli_rotation
                local_pauli = (pauli_string, list(range(len(qubits))), coeff)

                # build the circuit Rust-side
                data = pauli_evolution(
                    len(qubits),
                    [local_pauli],
                    False,
                    cx_fountain,
                )
                evo = QuantumCircuit._from_circuit_data(data)

                # and append it to the circuit with the correct label
                gate = evo.to_gate(label=f"exp(it {pauli_string})")
                circuit.append(gate, qubits)

            if self.insert_barriers and i < num_paulis - 1:
                circuit.barrier()

        return circuit


def real_or_fail(value, tol=100):
    """Return real if close, otherwise fail. Unbound parameters are left unchanged.

    Based on NumPy's ``real_if_close``, i.e. ``tol`` is in terms of machine precision for float.
    """
    if isinstance(value, ParameterExpression):
        return value

    abstol = tol * np.finfo(float).eps
    if abs(np.imag(value)) < abstol:
        return np.real(value)

    raise ValueError(f"Encountered complex value {value}, but expected real.")


def reorder_paulis(
    paulis: Sequence[SparsePauliLabel],
    strategy: rx.ColoringStrategy = rx.ColoringStrategy.Saturation,
) -> list[SparsePauliLabel]:
    r"""
    Creates an equivalent operator by reordering terms in order to yield a
    shallower circuit after evolution synthesis. The original operator remains
    unchanged.

    This method works in three steps. First, a graph is constructed, where the
    nodes are the terms of the operator and where two nodes are connected if
    their terms act on the same qubit (for example, the terms :math:`IXX` and
    :math:`IYI` would be connected, but not :math:`IXX` and :math:`YII`). Then,
    the graph is colored.  Two terms with the same color thus do not act on the
    same qubit, and in particular, their evolution subcircuits can be run in
    parallel in the greater evolution circuit of ``paulis``.

    This method is deterministic and invariant under permutation of the Pauli
    term in ``paulis``.

    Args:
        paulis: The operator whose terms to reorder.
        strategy: The coloring heuristic to use, see ``ColoringStrategy`` [#].
            Default is ``ColoringStrategy.Saturation``.

    .. [#] https://www.rustworkx.org/apiref/rustworkx.ColoringStrategy.html#coloringstrategy

    """

    def _term_sort_key(term: SparsePauliLabel) -> typing.Any:
        # sort by index, then by pauli
        return (term[1], term[0])

    # Do nothing in trivial cases
    if len(paulis) <= 1:
        return paulis

    terms = sorted(paulis, key=_term_sort_key)
    graph = rx.PyGraph()
    graph.add_nodes_from(terms)
    indexed_nodes = list(enumerate(graph.nodes()))
    for (idx1, (_, ind1, _)), (idx2, (_, ind2, _)) in combinations(indexed_nodes, 2):
        # Add an edge between two terms if they touch the same qubit
        if len(set(ind1).intersection(ind2)) > 0:
            graph.add_edge(idx1, idx2, None)

    # rx.graph_greedy_color is supposed to be deterministic
    coloring = rx.graph_greedy_color(graph, strategy=strategy)
    terms_by_color = defaultdict(list)

    for term_idx, color in sorted(coloring.items()):
        term = graph.nodes()[term_idx]
        terms_by_color[color].append(term)

    terms = list(itertools.chain(*terms_by_color.values()))
    return terms


def wrap_custom_atomic_evolution(atomic_evolution, support_sparse_observable):
    r"""Wrap a custom atomic evolution into compatible format for the product formula.

    This includes an inplace action, i.e. the signature is (circuit, operator, time) and
    ensuring that ``SparseObservable``\ s are supported.
    """
    # next, enable backward compatible use of atomic evolutions, that did not support
    # SparseObservable inputs
    if support_sparse_observable is False:
        warnings.warn(
            "The atomic_evolution should support SparseObservables as operator input. "
            "Until Qiskit 2.2, an automatic conversion to SparsePauliOp is done, which can "
            "be turned off by passing the argument atomic_evolution_sparse_observable=True.",
            category=PendingDeprecationWarning,
            stacklevel=2,
        )

        def sparseobs_atomic_evolution(output, operator, time):
            if isinstance(operator, SparseObservable):
                operator = SparsePauliOp.from_sparse_observable(operator)

            atomic_evolution(output, operator, time)

    else:
        sparseobs_atomic_evolution = atomic_evolution

    return sparseobs_atomic_evolution
