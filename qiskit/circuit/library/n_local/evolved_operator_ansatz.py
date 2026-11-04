# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The evolved operator ansatz."""

from __future__ import annotations
from collections.abc import Sequence

import typing
import warnings
import itertools
import numpy as np


from qiskit.circuit.library.hamiltonian_gate import HamiltonianGate
from qiskit.circuit.library.pauli_evolution import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli, SparsePauliOp, SparseObservable
from qiskit.quantum_info.operators.base_operator import BaseOperator

from qiskit._accelerate.circuit_library import pauli_evolution

from .n_local import NLocal

if typing.TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis


def evolved_operator_ansatz(
    operators: BaseOperator | Sequence[BaseOperator],
    reps: int = 1,
    evolution: EvolutionSynthesis | None = None,
    insert_barriers: bool = False,
    name: str = "EvolvedOps",
    parameter_prefix: str | Sequence[str] = "t",
    remove_identities: bool = True,
    flatten: bool | None = None,
) -> QuantumCircuit:
    r"""Construct an ansatz out of operator evolutions.

    For a set of operators :math:`[O_1, ..., O_J]` and :math:`R` repetitions (``reps``), this circuit
    is defined as

    .. math::

        \prod_{r=1}^{R} \left( \prod_{j=J}^1 e^{-i\theta_{j, r} O_j} \right)

    where the exponentials :math:`exp(-i\theta O_j)` are expanded using the product formula
    specified by ``evolution``.

    Examples:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.circuit.library import evolved_operator_ansatz
        from qiskit.quantum_info import Pauli

        ops = [Pauli("ZZI"), Pauli("IZZ"), Pauli("IXI")]
        ansatz = evolved_operator_ansatz(ops, reps=3, insert_barriers=True)
        ansatz.draw("mpl")

    Args:
        operators: The operators to evolve. Can be a single operator or a sequence thereof.
        reps: The number of times to repeat the evolved operators.
        evolution: A specification of which evolution synthesis to use for the
            :class:`.PauliEvolutionGate`. Defaults to first order Trotterization. Note, that
            operators of type :class:`.Operator` are evolved using the :class:`.HamiltonianGate`,
            as there are no Hamiltonian terms to expand in Trotterization.
        insert_barriers: Whether to insert barriers in between each evolution.
        name: The name of the circuit.
        parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
            will be used for each parameter. Can also be a list to specify a prefix per
            operator.
        remove_identities: If ``True``, ignore identity operators (note that we do not check
            :class:`.Operator` inputs). This will also remove parameters associated with identities.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.
    """
    from qiskit.synthesis.evolution import LieTrotter
    from qiskit.synthesis.evolution.product_formula import real_or_fail

    if reps < 0:
        raise ValueError("reps must be a non-negative integer.")

    if isinstance(operators, (BaseOperator, SparseObservable)):
        operators = [operators]
    elif len(operators) == 0:
        return QuantumCircuit()

    num_operators = len(operators)
    if not isinstance(parameter_prefix, str):
        if num_operators != len(parameter_prefix):
            raise ValueError(
                f"Mismatching number of operators ({len(operators)}) and parameter_prefix "
                f"({len(parameter_prefix)})."
            )

    num_qubits = operators[0].num_qubits
    if remove_identities:
        operators, parameter_prefix = _remove_identities(operators, parameter_prefix)
        # After removing identities, update num_operators to reflect the actual count
        num_operators = len(operators)
        if num_operators == 0:
            return QuantumCircuit(num_qubits, name=name)

    if any(op.num_qubits != num_qubits for op in operators):
        raise ValueError("Inconsistent numbers of qubits in the operators.")

    # get the total number of parameters
    if isinstance(parameter_prefix, str):
        parameters = ParameterVector(parameter_prefix, reps * num_operators)
        param_iter = iter(parameters)
    else:
        # this creates the parameter vectors per operator, e.g.
        #    [[a0, a1, a2, ...], [b0, b1, b2, ...], [c0, c1, c2, ...]]
        # and turns them into an iterator
        #    a0 -> b0 -> c0 -> a1 -> b1 -> c1 -> a2 -> ...
        per_operator = [ParameterVector(prefix, reps).params for prefix in parameter_prefix]
        param_iter = itertools.chain.from_iterable(zip(*per_operator))

    # fast, Rust-path
    if (
        flatten is not False  # captures None and True
        and evolution is None
        and all(isinstance(op, (SparsePauliOp, SparseObservable)) for op in operators)
    ):
        sparse_labels = [op.to_sparse_list() for op in operators]
        expanded_paulis = []
        for _ in range(reps):
            for term in sparse_labels:
                param = next(param_iter)
                expanded_paulis += [
                    (pauli, indices, 2 * real_or_fail(coeff) * param)
                    for pauli, indices, coeff in term
                ]

        data = pauli_evolution(num_qubits, expanded_paulis, insert_barriers, False)
        circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True)
        circuit.name = name

        return circuit

    # slower, Python-path
    if evolution is None:
        evolution = LieTrotter(insert_barriers=insert_barriers)

    circuit = QuantumCircuit(num_qubits, name=name)

    for rep in range(reps):
        for i, op in enumerate(operators):
            if isinstance(op, Operator):
                gate = HamiltonianGate(op, next(param_iter))
                if flatten:
                    warnings.warn(
                        "Cannot flatten the evolution of an Operator, flatten is set to "
                        "False for this operator."
                    )
                flatten_operator = False

            elif isinstance(op, (BaseOperator, SparseObservable)):
                gate = PauliEvolutionGate(op, next(param_iter), synthesis=evolution)
                flatten_operator = flatten is True or flatten is None
            else:
                raise ValueError(f"Unsupported operator type: {type(op)}")

            if flatten_operator:
                circuit.compose(gate.definition, inplace=True)
            else:
                circuit.append(gate, circuit.qubits)

            if insert_barriers and (rep < reps - 1 or i < num_operators - 1):
                circuit.barrier()

    return circuit


def hamiltonian_variational_ansatz(
    hamiltonian: SparsePauliOp | Sequence[SparsePauliOp],
    reps: int = 1,
    insert_barriers: bool = False,
    name: str = "HVA",
    parameter_prefix: str = "t",
) -> QuantumCircuit:
    r"""Construct a Hamiltonian variational ansatz.

    For a Hamiltonian :math:`H = \sum_{k=1}^K H_k` where the terms :math:`H_k` consist of only
    commuting Paulis, but the terms do not commute among each other :math:`[H_k, H_{k'}] \neq 0`, the
    Hamiltonian variational ansatz (HVA) is

    .. math::

        \prod_{r=1}^{R} \left( \prod_{k=K}^1 e^{-i\theta_{k, r} H_k} \right)

    where the exponentials :math:`exp(-i\theta H_k)` are implemented exactly [1, 2]. Note that this
    differs from :func:`.evolved_operator_ansatz`, where no assumptions on the structure of the
    operators are done.

    The Hamiltonian can be passed as :class:`.SparsePauliOp`, in which case we split the Hamiltonian
    into commuting terms :math:`\{H_k\}_k`. Note, that this may not be optimal and if the
    minimal set of commuting terms is known it can be passed as sequence into this function.

    Examples:

    A single operator will be split into commuting terms automatically:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import hamiltonian_variational_ansatz

        # this Hamiltonian will be split into the two terms [ZZI, IZZ] and [IXI]
        hamiltonian = SparsePauliOp(["ZZI", "IZZ", "IXI"])
        ansatz = hamiltonian_variational_ansatz(hamiltonian, reps=2)
        ansatz.draw("mpl")

    Alternatively, we can directly provide the terms:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import hamiltonian_variational_ansatz

        zz = SparsePauliOp(["ZZI", "IZZ"])
        x = SparsePauliOp(["IXI"])
        ansatz = hamiltonian_variational_ansatz([zz, x], reps=2)
        ansatz.draw("mpl")


    Args:
        hamiltonian: The Hamiltonian to evolve. If given as single operator, it will be split into
            commuting terms. If a sequence of :class:`.SparsePauliOp`, then it is assumed that
            each element consists of commuting terms, but the elements do not commute among each
            other.
        reps: The number of times to repeat the evolved operators.
        insert_barriers: Whether to insert barriers in between each evolution.
        name: The name of the circuit.
        parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
            will be used for each parameter. Can also be a list to specify a prefix per
            operator.

    References:

    [1] D. Wecker et al. Progress towards practical quantum variational algorithms (2015)
    `Phys Rev A 92, 042303 <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.042303>`__

    [2] R. Wiersema et al. Exploring entanglement and optimization within the Hamiltonian
    Variational Ansatz (2020) `arXiv:2008.02941 <https://arxiv.org/abs/2008.02941>`__

    """
    # If a single operator is given, check if it is a sum of operators (a SparsePauliOp),
    # and split it into commuting terms. Otherwise, treat it as single operator.
    if isinstance(hamiltonian, SparsePauliOp):
        hamiltonian = hamiltonian.group_commuting()

    return evolved_operator_ansatz(
        hamiltonian,
        reps=reps,
        evolution=None,
        insert_barriers=insert_barriers,
        name=name,
        parameter_prefix=parameter_prefix,
        flatten=True,
    )


class EvolvedOperatorAnsatz(NLocal):
    """The evolved operator ansatz."""

    def __init__(
        self,
        operators=None,
        reps: int = 1,
        evolution=None,
        insert_barriers: bool = False,
        name: str = "EvolvedOps",
        parameter_prefix: str | Sequence[str] = "t",
        initial_state: QuantumCircuit | None = None,
        flatten: bool | None = None,
    ):
        """
        Args:
            operators (BaseOperator | QuantumCircuit | list | None): The operators
                to evolve. If a circuit is passed, we assume it implements an already evolved
                operator and thus the circuit is not evolved again. Can be a single operator
                (circuit) or a list of operators (and circuits).
            reps: The number of times to repeat the evolved operators.
            evolution (EvolutionBase | EvolutionSynthesis | None):
                A specification of which evolution synthesis to use for the
                :class:`.PauliEvolutionGate`.
                Defaults to first order Trotterization.
            insert_barriers: Whether to insert barriers in between each evolution.
            name: The name of the circuit.
            parameter_prefix: Set the names of the circuit parameters. If a string, the same prefix
                will be used for each parameter. Can also be a list to specify a prefix per
                operator.
            initial_state: A :class:`.QuantumCircuit` object to prepend to the circuit.
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
        """
        super().__init__(
            initial_state=initial_state,
            parameter_prefix=parameter_prefix,
            reps=reps,
            insert_barriers=insert_barriers,
            name=name,
            flatten=flatten,
        )
        self._operators = None

        if operators is not None:
            self.operators = operators

        self._evolution = evolution

        # a list of which operators are parameterized, used for internal settings
        self._ops_are_parameterized = None

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        if not super()._check_configuration(raise_on_failure):
            return False

        if self.operators is None:
            if raise_on_failure:
                raise ValueError("The operators are not set.")
            return False

        return True

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        if self.operators is None:
            return 0

        if isinstance(self.operators, list):
            if len(self.operators) == 0:
                return 0
            return self.operators[0].num_qubits

        return self.operators.num_qubits

    @property
    def evolution(self):
        """The evolution converter used to compute the evolution.

        Returns:
            EvolutionSynthesis: The evolution converter used to compute the evolution.
        """

        return self._evolution

    @evolution.setter
    def evolution(self, evol) -> None:
        """Sets the evolution converter used to compute the evolution.

        Args:
            evol (EvolutionSynthesis): An evolution synthesis object
        """
        self._invalidate()
        self._evolution = evol

    @property
    def operators(self):
        """The operators that are evolved in this circuit.

        Returns:
            list: The operators to be evolved (and circuits) contained in this ansatz.
        """
        return self._operators

    @operators.setter
    def operators(self, operators=None) -> None:
        """Set the operators to be evolved.

        operators (Optional[Union[QuantumCircuit, list]]): The operators to evolve.
            If a circuit is passed, we assume it implements an already evolved operator and thus
            the circuit is not evolved again. Can be a single operator (circuit) or a list of
            operators (and circuits).
        """
        operators = _validate_operators(operators)
        self._invalidate()
        self._operators = operators
        if self.num_qubits == 0:
            self.qregs = []
        else:
            self.qregs = [QuantumRegister(self.num_qubits, name="q")]

    # TODO: the `preferred_init_points`-implementation can (and should!) be improved!
    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            # If an initial state was set by the user, then we want to make sure that the VQE does
            # not start from a random point. Thus, we return an all-zero initial point for the
            # optimizer which is used (unless it gets overwritten by a higher-priority setting at
            # runtime of the VQE).
            # However, in order to determine the correct length, we must build the QuantumCircuit
            # first, because otherwise the operators may not be set yet.
            self._build()
            return np.zeros(self.reps * len(self.operators), dtype=float)

    def _evolve_operator(self, operator, time):

        # if the operator is specified as matrix use exact matrix exponentiation
        if isinstance(operator, Operator):
            gate = HamiltonianGate(operator, time)
        # otherwise, use the PauliEvolutionGate
        else:
            from qiskit.synthesis.evolution import LieTrotter

            evolution = LieTrotter() if self._evolution is None else self._evolution
            gate = PauliEvolutionGate(operator, time, synthesis=evolution)

        evolved = QuantumCircuit(operator.num_qubits)
        if not self.flatten:
            evolved.append(gate, evolved.qubits)
        else:
            evolved.compose(gate.definition, evolved.qubits, inplace=True)
        return evolved

    def _build(self):
        if self._is_built:
            return

        # need to check configuration here to ensure the operators are not None
        self._check_configuration()

        coeff = Parameter("c")
        circuits = []

        for op in self.operators:
            # if the operator is already the evolved circuit just append it
            if isinstance(op, QuantumCircuit):
                circuits.append(op)
            else:
                # check if the operator is just the identity, if yes, skip it
                if _is_pauli_identity(op):
                    continue

                evolved = self._evolve_operator(op, coeff)
                circuits.append(evolved)

        self.rotation_blocks = []
        self.entanglement_blocks = circuits

        super()._build()


def _validate_operators(operators):
    if not isinstance(operators, list):
        operators = [operators]

    if len(operators) > 1:
        num_qubits = operators[0].num_qubits
        if any(operators[i].num_qubits != num_qubits for i in range(1, len(operators))):
            raise ValueError("All operators must act on the same number of qubits.")

    return operators


def _validate_prefix(parameter_prefix, operators):
    if isinstance(parameter_prefix, str):
        return len(operators) * [parameter_prefix]
    if len(parameter_prefix) != len(operators):
        raise ValueError("The number of parameter prefixes must match the operators.")

    return parameter_prefix


def _is_pauli_identity(operator):
    if isinstance(operator, SparseObservable):
        return operator.num_terms == 1 and len(operator[0].bit_labels()) == 0
    if isinstance(operator, SparsePauliOp):
        if len(operator.paulis) == 1:
            operator = operator.paulis[0]  # check if the single Pauli is identity below
        else:
            return False
    if isinstance(operator, Pauli):
        return not np.any(np.logical_or(operator.x, operator.z))
    return False


def _remove_identities(operators, prefixes):
    identity_ops = {index for index, op in enumerate(operators) if _is_pauli_identity(op)}

    if len(identity_ops) == 0:
        return operators, prefixes

    cleaned_ops = [op for i, op in enumerate(operators) if i not in identity_ops]
    # Handle both string and list prefixes
    if isinstance(prefixes, str):
        # If it's a string, keep it as a string (it will be used for all remaining operators)
        cleaned_prefix = prefixes
    else:
        # If it's a list, remove the corresponding entries
        cleaned_prefix = [prefix for i, prefix in enumerate(prefixes) if i not in identity_ops]

    return cleaned_ops, cleaned_prefix
