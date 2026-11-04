# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A generalized QAOA quantum circuit with a support of custom initial states and mixers."""


from __future__ import annotations

import numpy as np

from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .evolved_operator_ansatz import (
    EvolvedOperatorAnsatz,
    _is_pauli_identity,
    evolved_operator_ansatz,
)


def qaoa_ansatz(
    cost_operator: BaseOperator,
    reps: int = 1,
    initial_state: QuantumCircuit | None = None,
    mixer_operator: BaseOperator | None = None,
    insert_barriers: bool = False,
    name: str = "QAOA",
    flatten: bool = True,
) -> QuantumCircuit:
    r"""A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    Examples:

    To define the QAOA ansatz we require a cost Hamiltonian, encoding the classical
    optimization problem:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.quantum_info import SparsePauliOp
        from qiskit.circuit.library import qaoa_ansatz

        cost_operator = SparsePauliOp(["ZZII", "IIZZ", "ZIIZ"])
        ansatz = qaoa_ansatz(cost_operator, reps=3, insert_barriers=True)
        ansatz.draw("mpl")

    Args:
        cost_operator: The operator representing the cost of the optimization problem, denoted as
            :math:`U(C, \gamma)` in [1].
        reps: The integer determining the depth of the circuit, called :math:`p` in [1].
        initial_state: An optional initial state to use, which defaults to a layer of
            Hadamard gates preparing the :math:`|+\rangle^{\otimes n}` state.
            If a custom mixer is chosen, this circuit should be set to prepare its ground state,
            to appropriately fulfill the annealing conditions.
        mixer_operator: An optional custom mixer, which defaults to global Pauli-:math:`X`
            rotations. This is denoted as :math:`U(B, \beta)` in [1]. If this is set,
            the ``initial_state`` might also require modification.
        insert_barriers: Whether to insert barriers in-between the cost and mixer operators.
        name: The name of the circuit.
        flatten: If ``True``, a flat circuit is returned instead of nesting it inside multiple
            layers of gate objects. Setting this to ``False`` is significantly less performant,
            especially for parameter binding, but can be desirable for a cleaner visualization.

    References:

    [1] Farhi et al., A Quantum Approximate Optimization Algorithm.
    `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """
    num_qubits = cost_operator.num_qubits

    if initial_state is None:
        initial_state = QuantumCircuit(num_qubits)
        initial_state.h(range(num_qubits))

    if mixer_operator is None:
        mixer_operator = SparsePauliOp.from_sparse_list(
            [("X", [i], 1) for i in range(num_qubits)], num_qubits
        )

    parameter_prefix = ["γ", "β"]

    return initial_state.compose(
        evolved_operator_ansatz(
            [cost_operator, mixer_operator],
            reps=reps,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            name=name,
            flatten=flatten,
        ),
        copy=False,
    )


class QAOAAnsatz(EvolvedOperatorAnsatz):
    """A generalized QAOA quantum circuit with a support of custom initial states and mixers.

    References:

    [1] Farhi et al., A Quantum Approximate Optimization Algorithm.
    `arXiv:1411.4028 <https://arxiv.org/pdf/1411.4028>`_
    """

    def __init__(
        self,
        cost_operator=None,
        reps: int = 1,
        initial_state: QuantumCircuit | None = None,
        mixer_operator=None,
        name: str = "QAOA",
        flatten: bool | None = None,
    ):
        r"""
        Args:
            cost_operator (BaseOperator or OperatorBase, optional): The operator
                representing the cost of the optimization problem, denoted as :math:`U(C, \gamma)`
                in the original paper. Must be set either in the constructor or via property setter.
            reps (int): The integer parameter p, which determines the depth of the circuit,
                as specified in the original paper, default is 1.
            initial_state (QuantumCircuit, optional): An optional initial state to use.
                If `None` is passed then a set of Hadamard gates is applied as an initial state
                to all qubits.
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): An optional
                custom mixer to use instead of the global X-rotations, denoted as :math:`U(B, \beta)`
                in the original paper. Can be an operator or an optionally parameterized quantum
                circuit.
            name (str): A name of the circuit, default 'qaoa'
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
        """
        super().__init__(reps=reps, name=name, flatten=flatten)

        self._cost_operator = None
        self._reps = reps
        self._initial_state: QuantumCircuit | None = initial_state
        self._mixer = mixer_operator

        # set this circuit as a not-built circuit
        self._bounds: list[tuple[float | None, float | None]] | None = None

        # store cost operator and set the registers if the operator is not None
        self.cost_operator = cost_operator

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the current configuration is valid."""
        valid = True

        if not super()._check_configuration(raise_on_failure):
            return False

        if self.cost_operator is None:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    "The operator representing the cost of the optimization problem is not set"
                )

        if self.initial_state is not None and self.initial_state.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    f"The number of qubits of the initial state {self.initial_state.num_qubits}"
                    " does not match "
                    f"the number of qubits of the cost operator {self.num_qubits}"
                )

        if self.mixer_operator is not None and self.mixer_operator.num_qubits != self.num_qubits:
            valid = False
            if raise_on_failure:
                raise ValueError(
                    f"The number of qubits of the mixer {self.mixer_operator.num_qubits}"
                    f" does not match "
                    f"the number of qubits of the cost operator {self.num_qubits}"
                )

        return valid

    @property
    def parameter_bounds(self) -> list[tuple[float | None, float | None]] | None:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If None is returned, problem is fully
            unbounded.
        """
        if self._bounds is not None:
            return self._bounds

        # if the mixer is a circuit, we set no bounds
        if isinstance(self.mixer_operator, QuantumCircuit):
            return None

        # default bounds: None for gamma (cost operator), [0, 2pi] for gamma (mixer operator)
        beta_bounds = (0, 2 * np.pi)
        gamma_bounds = (None, None)
        bounds: list[tuple[float | None, float | None]] = []

        if not _is_pauli_identity(self.mixer_operator):
            bounds += self.reps * [beta_bounds]

        if not _is_pauli_identity(self.cost_operator):
            bounds += self.reps * [gamma_bounds]

        return bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float | None, float | None]] | None) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    @property
    def operators(self) -> list:
        """The operators that are evolved in this circuit.

        Returns:
             List[Union[BaseOperator, OperatorBase, QuantumCircuit]]: The operators to be evolved
                (and circuits) in this ansatz.
        """
        return [self.cost_operator, self.mixer_operator]

    @property
    def cost_operator(self):
        """Returns an operator representing the cost of the optimization problem.

        Returns:
            BaseOperator or OperatorBase: cost operator.
        """
        return self._cost_operator

    @cost_operator.setter
    def cost_operator(self, cost_operator) -> None:
        """Sets cost operator.

        Args:
            cost_operator (BaseOperator or OperatorBase, optional): cost operator to set.
        """
        self._cost_operator = cost_operator
        self.qregs = [QuantumRegister(self.num_qubits, name="q")]
        self._invalidate()

    @property
    def reps(self) -> int:
        """Returns the `reps` parameter, which determines the depth of the circuit."""
        return self._reps

    @reps.setter
    def reps(self, reps: int) -> None:
        """Sets the `reps` parameter."""
        self._reps = reps
        self._invalidate()

    @property
    def initial_state(self) -> QuantumCircuit | None:
        """Returns an optional initial state as a circuit"""
        if self._initial_state is not None:
            return self._initial_state

        # if no initial state is passed and we know the number of qubits, then initialize it.
        if self.num_qubits > 0:
            initial_state = QuantumCircuit(self.num_qubits)
            initial_state.h(range(self.num_qubits))
            return initial_state

        # otherwise we cannot provide a default
        return None

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit | None) -> None:
        """Sets initial state."""
        self._initial_state = initial_state
        self._invalidate()

    # we can't directly specify OperatorBase as a return type, it causes a circular import
    # and pylint objects if return type is not documented
    @property
    def mixer_operator(self):
        """Returns an optional mixer operator expressed as an operator or a quantum circuit.

        Returns:
            BaseOperator or OperatorBase or QuantumCircuit, optional: mixer operator or circuit.
        """
        if self._mixer is not None:
            return self._mixer

        # if no mixer is passed and we know the number of qubits, then initialize it.
        if self.cost_operator is not None:
            # local imports to avoid circular imports
            num_qubits = self.cost_operator.num_qubits

            # Mixer is just a sum of single qubit X's on each qubit. Evolving by this operator
            # will simply produce rx's on each qubit.
            mixer_terms = [
                ("I" * left + "X" + "I" * (num_qubits - left - 1), 1) for left in range(num_qubits)
            ]
            mixer = SparsePauliOp.from_list(mixer_terms)
            return mixer

        # otherwise we cannot provide a default
        return None

    @mixer_operator.setter
    def mixer_operator(self, mixer_operator) -> None:
        """Sets mixer operator.

        Args:
            mixer_operator (BaseOperator or OperatorBase or QuantumCircuit, optional): mixer
                operator or circuit to set.
        """
        self._mixer = mixer_operator
        self._invalidate()

    @property
    def num_qubits(self) -> int:
        if self._cost_operator is None:
            return 0
        return self._cost_operator.num_qubits

    def _build(self):
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        # keep old parameter order: first cost operator, then mixer operators
        num_cost = 0 if _is_pauli_identity(self.cost_operator) else 1
        if isinstance(self.mixer_operator, QuantumCircuit):
            num_mixer = self.mixer_operator.num_parameters
        else:
            num_mixer = 0 if _is_pauli_identity(self.mixer_operator) else 1

        betas = ParameterVector("β", self.reps * num_mixer)
        gammas = ParameterVector("γ", self.reps * num_cost)

        # Create a permutation to take us from (cost_1, mixer_1, cost_2, mixer_2, ...)
        # to (cost_1, cost_2, ..., mixer_1, mixer_2, ...), or if the mixer is a circuit
        # with more than 1 parameters, from (cost_1, mixer_1a, mixer_1b, cost_2, ...)
        # to (cost_1, cost_2, ..., mixer_1a, mixer_1b, mixer_2a, mixer_2b, ...)
        reordered = []
        for rep in range(self.reps):
            reordered.extend(gammas[rep * num_cost : (rep + 1) * num_cost])
            reordered.extend(betas[rep * num_mixer : (rep + 1) * num_mixer])

        self.assign_parameters(dict(zip(self.ordered_parameters, reordered)), inplace=True)
