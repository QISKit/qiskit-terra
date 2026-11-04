# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The Grover operator."""

from __future__ import annotations

import numpy

from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister, AncillaQubit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Statevector, Operator, DensityMatrix
from qiskit.utils.deprecation import deprecate_func
from .standard_gates import MCXGate
from .generalized_gates import DiagonalGate


def grover_operator(
    oracle: QuantumCircuit | Statevector,
    state_preparation: QuantumCircuit | None = None,
    zero_reflection: QuantumCircuit | DensityMatrix | Operator | None = None,
    reflection_qubits: list[int] | None = None,
    insert_barriers: bool = False,
    name: str = "Q",
):
    r"""Construct the Grover operator.

    Grover's search algorithm [1, 2] consists of repeated applications of the so-called
    Grover operator used to amplify the amplitudes of the desired output states.
    This operator, :math:`\mathcal{Q}`, consists of the phase oracle, :math:`\mathcal{S}_f`,
    zero phase-shift or zero reflection, :math:`\mathcal{S}_0`, and an
    input state preparation :math:`\mathcal{A}`:

    .. math::
        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f

    In the standard Grover search we have :math:`\mathcal{A} = H^{\otimes n}`:

    .. math::
        \mathcal{Q} = H^{\otimes n} \mathcal{S}_0 H^{\otimes n} \mathcal{S}_f
                    = D \mathcal{S_f}

    The operation :math:`D = H^{\otimes n} \mathcal{S}_0 H^{\otimes n}` is also referred to as
    diffusion operator. In this formulation we can see that Grover's operator consists of two
    steps: first, the phase oracle multiplies the good states by -1 (with :math:`\mathcal{S}_f`)
    and then the whole state is reflected around the mean (with :math:`D`).

    This class allows setting a different state preparation, as in quantum amplitude
    amplification (a generalization of Grover's algorithm), :math:`\mathcal{A}` might not be
    a layer of Hadamard gates [3].

    The action of the phase oracle :math:`\mathcal{S}_f` is defined as

    .. math::
        \mathcal{S}_f: |x\rangle \mapsto (-1)^{f(x)}|x\rangle

    where :math:`f(x) = 1` if :math:`x` is a good state and 0 otherwise. To highlight the fact
    that this oracle flips the phase of the good states and does not flip the state of a result
    qubit, we call :math:`\mathcal{S}_f` a phase oracle.

    Note that you can easily construct a phase oracle from a bitflip oracle by sandwiching the
    controlled X gate on the result qubit by a X and H gate. For instance

    .. parsed-literal::

        Bitflip oracle     Phaseflip oracle
        q_0: ──■──         q_0: ────────────■────────────
             ┌─┴─┐              ┌───┐┌───┐┌─┴─┐┌───┐┌───┐
        out: ┤ X ├         out: ┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├
             └───┘              └───┘└───┘└───┘└───┘└───┘

    There is some flexibility in defining the oracle and :math:`\mathcal{A}` operator. Before the
    Grover operator is applied in Grover's algorithm, the qubits are first prepared with one
    application of the :math:`\mathcal{A}` operator (or Hadamard gates in the standard formulation).
    Thus, we always have operation of the form
    :math:`\mathcal{A} \mathcal{S}_f \mathcal{A}^\dagger`. Therefore it is possible to move
    bitflip logic into :math:`\mathcal{A}` and leaving the oracle only to do phaseflips via Z gates
    based on the bitflips. One possible use-case for this are oracles that do not uncompute the
    state qubits.

    The zero reflection :math:`\mathcal{S}_0` is usually defined as

    .. math::
        \mathcal{S}_0 = 2 |0\rangle^{\otimes n} \langle 0|^{\otimes n} - \mathbb{I}_n

    where :math:`\mathbb{I}_n` is the identity on :math:`n` qubits.
    By default, this class implements the negative version
    :math:`2 |0\rangle^{\otimes n} \langle 0|^{\otimes n} - \mathbb{I}_n`, since this can simply
    be implemented with a multi-controlled Z sandwiched by X gates on the target qubit and the
    introduced global phase does not matter for Grover's algorithm.

    Examples:

    We can construct a Grover operator just from the phase oracle:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context:

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import grover_operator

        oracle = QuantumCircuit(2)
        oracle.z(0)  # good state = first qubit is |1>
        grover_op = grover_operator(oracle, insert_barriers=True)
        grover_op.draw("mpl")

    We can also modify the state preparation:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        oracle = QuantumCircuit(1)
        oracle.z(0)  # the qubit state |1> is the good state
        state_preparation = QuantumCircuit(1)
        state_preparation.ry(0.2, 0)  # non-uniform state preparation
        grover_op = grover_operator(oracle, state_preparation)
        grover_op.draw("mpl")

    In addition, we can also mark which qubits the zero reflection should act on. This
    is useful in case that some qubits are just used as scratch space but should not affect
    the oracle:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        oracle = QuantumCircuit(4)
        oracle.z(3)
        reflection_qubits = [0, 3]
        state_preparation = QuantumCircuit(4)
        state_preparation.cry(0.1, 0, 3)
        state_preparation.ry(0.5, 3)
        grover_op = grover_operator(oracle, state_preparation, reflection_qubits=reflection_qubits)
        grover_op.draw("mpl")


    The oracle and the zero reflection can also be passed as :mod:`qiskit.quantum_info`
    objects:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        from qiskit.quantum_info import Statevector, DensityMatrix, Operator

        mark_state = Statevector.from_label("011")
        reflection = 2 * DensityMatrix.from_label("000") - Operator.from_label("III")
        grover_op = grover_operator(oracle=mark_state, zero_reflection=reflection)
        grover_op.draw("mpl")

    For a large number of qubits, the multi-controlled X gate used for the zero-reflection
    can be synthesized in different fashions. Depending on the number of available qubits,
    the compiler will choose a different implementation:

    .. code-block:: python

        from qiskit import transpile, Qubit
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import grover_operator

        oracle = QuantumCircuit(10)
        oracle.z(oracle.qubits)
        grover_op = grover_operator(oracle)

        # without extra qubit space, the MCX synthesis is expensive
        basis_gates = ["u", "cx"]
        tqc = transpile(grover_op, basis_gates=basis_gates)
        is_2q = lambda inst: len(inst.qubits) == 2
        print("2q depth w/o scratch qubits:", tqc.depth(filter_function=is_2q))  # > 350

        # add extra bits that can be used as scratch space
        grover_op.add_bits([Qubit() for _ in range(num_qubits)])
        tqc = transpile(grover_op, basis_gates=basis_gates)
        print("2q depth w/ scratch qubits:", tqc.depth(filter_function=is_2q)) # < 100

    Args:
        oracle: The phase oracle implementing a reflection about the bad state. Note that this
            is not a bitflip oracle, see the docstring for more information.
        state_preparation: The operator preparing the good and bad state.
            For Grover's algorithm, this is a n-qubit Hadamard gate and for amplitude
            amplification or estimation the operator :math:`\mathcal{A}`.
        zero_reflection: The reflection about the zero state, :math:`\mathcal{S}_0`.
        reflection_qubits: Qubits on which the zero reflection acts on.
        insert_barriers: Whether barriers should be inserted between the reflections and A.
        name: The name of the circuit.

    References:

    [1] L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
    `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.

    [2] I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
    Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.

    [3] Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum Amplitude Amplification and Estimation.
    `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
    """
    # We inherit the ancillas/qubits structure from the oracle, if it is given as circuit.
    if isinstance(oracle, QuantumCircuit):
        circuit = oracle.copy_empty_like(name=name, vars_mode="drop")
    else:
        circuit = QuantumCircuit(oracle.num_qubits, name=name)

    # (1) Add the oracle.
    # If the oracle is given as statevector, turn it into a circuit that implements the
    # reflection about the state.
    if isinstance(oracle, Statevector):
        diagonal = DiagonalGate((-1) ** oracle.data)
        circuit.append(diagonal, circuit.qubits)
    else:
        circuit.compose(oracle, inplace=True)

    if insert_barriers:
        circuit.barrier()

    # (2) Add the inverse state preparation.
    # For this we need to know the target qubits that we apply the zero reflection to.
    # If the reflection qubits are not given, we assume they are the qubits that are not
    # of type ``AncillaQubit`` in the oracle.
    if reflection_qubits is None:
        reflection_qubits = [
            i for i, qubit in enumerate(circuit.qubits) if not isinstance(qubit, AncillaQubit)
        ]

    if state_preparation is None:
        circuit.h(reflection_qubits)  # H is self-inverse
    else:
        circuit.compose(state_preparation.inverse(), inplace=True)

    if insert_barriers:
        circuit.barrier()

    # (3) Add the zero reflection.
    if zero_reflection is None:
        num_reflection = len(reflection_qubits)

        circuit.x(reflection_qubits)
        if num_reflection == 1:
            circuit.z(
                reflection_qubits[0]
            )  # MCX does not support 0 controls, hence this is separate
        else:
            mcx = MCXGate(num_reflection - 1)

            circuit.h(reflection_qubits[-1])
            circuit.append(mcx, reflection_qubits)
            circuit.h(reflection_qubits[-1])
        circuit.x(reflection_qubits)

    elif isinstance(zero_reflection, (Operator, DensityMatrix)):
        diagonal = DiagonalGate(zero_reflection.data.diagonal())
        circuit.append(diagonal, circuit.qubits)

    else:
        circuit.compose(zero_reflection, inplace=True)

    if insert_barriers:
        circuit.barrier()

    # (4) Add the state preparation.
    if state_preparation is None:
        circuit.h(reflection_qubits)
    else:
        circuit.compose(state_preparation, inplace=True)

    # minus sign
    circuit.global_phase = numpy.pi

    return circuit


class GroverOperator(QuantumCircuit):
    r"""The Grover operator.

    Grover's search algorithm [1, 2] consists of repeated applications of the so-called
    Grover operator used to amplify the amplitudes of the desired output states.
    This operator, :math:`\mathcal{Q}`, consists of the phase oracle, :math:`\mathcal{S}_f`,
    zero phase-shift or zero reflection, :math:`\mathcal{S}_0`, and an
    input state preparation :math:`\mathcal{A}`:

    .. math::
        \mathcal{Q} = \mathcal{A} \mathcal{S}_0 \mathcal{A}^\dagger \mathcal{S}_f

    In the standard Grover search we have :math:`\mathcal{A} = H^{\otimes n}`:

    .. math::
        \mathcal{Q} = H^{\otimes n} \mathcal{S}_0 H^{\otimes n} \mathcal{S}_f
                    = D \mathcal{S_f}

    The operation :math:`D = H^{\otimes n} \mathcal{S}_0 H^{\otimes n}` is also referred to as
    diffusion operator. In this formulation we can see that Grover's operator consists of two
    steps: first, the phase oracle multiplies the good states by -1 (with :math:`\mathcal{S}_f`)
    and then the whole state is reflected around the mean (with :math:`D`).

    This class allows setting a different state preparation, as in quantum amplitude
    amplification (a generalization of Grover's algorithm), :math:`\mathcal{A}` might not be
    a layer of Hadamard gates [3].

    The action of the phase oracle :math:`\mathcal{S}_f` is defined as

    .. math::
        \mathcal{S}_f: |x\rangle \mapsto (-1)^{f(x)}|x\rangle

    where :math:`f(x) = 1` if :math:`x` is a good state and 0 otherwise. To highlight the fact
    that this oracle flips the phase of the good states and does not flip the state of a result
    qubit, we call :math:`\mathcal{S}_f` a phase oracle.

    Note that you can easily construct a phase oracle from a bitflip oracle by sandwiching the
    controlled X gate on the result qubit by a X and H gate. For instance

    .. code-block:: text

        Bitflip oracle     Phaseflip oracle
        q_0: ──■──         q_0: ────────────■────────────
             ┌─┴─┐              ┌───┐┌───┐┌─┴─┐┌───┐┌───┐
        out: ┤ X ├         out: ┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├
             └───┘              └───┘└───┘└───┘└───┘└───┘

    There is some flexibility in defining the oracle and :math:`\mathcal{A}` operator. Before the
    Grover operator is applied in Grover's algorithm, the qubits are first prepared with one
    application of the :math:`\mathcal{A}` operator (or Hadamard gates in the standard formulation).
    Thus, we always have operation of the form
    :math:`\mathcal{A} \mathcal{S}_f \mathcal{A}^\dagger`. Therefore it is possible to move
    bitflip logic into :math:`\mathcal{A}` and leaving the oracle only to do phaseflips via Z gates
    based on the bitflips. One possible use-case for this are oracles that do not uncompute the
    state qubits.

    The zero reflection :math:`\mathcal{S}_0` is usually defined as

    .. math::
        \mathcal{S}_0 = 2 |0\rangle^{\otimes n} \langle 0|^{\otimes n} - \mathbb{I}_n

    where :math:`\mathbb{I}_n` is the identity on :math:`n` qubits.
    By default, this class implements the negative version
    :math:`2 |0\rangle^{\otimes n} \langle 0|^{\otimes n} - \mathbb{I}_n`, since this can simply
    be implemented with a multi-controlled Z sandwiched by X gates on the target qubit and the
    introduced global phase does not matter for Grover's algorithm.

    Examples:
        >>> from qiskit.circuit import QuantumCircuit
        >>> from qiskit.circuit.library import GroverOperator
        >>> oracle = QuantumCircuit(2)
        >>> oracle.z(0)  # good state = first qubit is |1>
        >>> grover_op = GroverOperator(oracle, insert_barriers=True)
        >>> grover_op.decompose().draw()
                 ┌───┐ ░ ┌───┐ ░ ┌───┐          ┌───┐      ░ ┌───┐
        state_0: ┤ Z ├─░─┤ H ├─░─┤ X ├───────■──┤ X ├──────░─┤ H ├
                 └───┘ ░ ├───┤ ░ ├───┤┌───┐┌─┴─┐├───┤┌───┐ ░ ├───┤
        state_1: ──────░─┤ H ├─░─┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├─░─┤ H ├
                       ░ └───┘ ░ └───┘└───┘└───┘└───┘└───┘ ░ └───┘

        >>> oracle = QuantumCircuit(1)
        >>> oracle.z(0)  # the qubit state |1> is the good state
        >>> state_preparation = QuantumCircuit(1)
        >>> state_preparation.ry(0.2, 0)  # non-uniform state preparation
        >>> grover_op = GroverOperator(oracle, state_preparation)
        >>> grover_op.decompose().draw()
                 ┌───┐┌──────────┐┌───┐┌───┐┌───┐┌─────────┐
        state_0: ┤ Z ├┤ RY(-0.2) ├┤ X ├┤ Z ├┤ X ├┤ RY(0.2) ├
                 └───┘└──────────┘└───┘└───┘└───┘└─────────┘

        >>> oracle = QuantumCircuit(4)
        >>> oracle.z(3)
        >>> reflection_qubits = [0, 3]
        >>> state_preparation = QuantumCircuit(4)
        >>> state_preparation.cry(0.1, 0, 3)
        >>> state_preparation.ry(0.5, 3)
        >>> grover_op = GroverOperator(oracle, state_preparation,
        ... reflection_qubits=reflection_qubits)
        >>> grover_op.decompose().draw()
                                              ┌───┐          ┌───┐
        state_0: ──────────────────────■──────┤ X ├───────■──┤ X ├──────────■────────────────
                                       │      └───┘       │  └───┘          │
        state_1: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                                       │                  │                 │
        state_2: ──────────────────────┼──────────────────┼─────────────────┼────────────────
                 ┌───┐┌──────────┐┌────┴─────┐┌───┐┌───┐┌─┴─┐┌───┐┌───┐┌────┴────┐┌─────────┐
        state_3: ┤ Z ├┤ RY(-0.5) ├┤ RY(-0.1) ├┤ X ├┤ H ├┤ X ├┤ H ├┤ X ├┤ RY(0.1) ├┤ RY(0.5) ├
                 └───┘└──────────┘└──────────┘└───┘└───┘└───┘└───┘└───┘└─────────┘└─────────┘

        >>> mark_state = Statevector.from_label('011')
        >>> diffuse_operator = 2 * DensityMatrix.from_label('000') - Operator.from_label('III')
        >>> grover_op = GroverOperator(oracle=mark_state, zero_reflection=diffuse_operator)
        >>> grover_op.decompose().draw(fold=70)
                 ┌─────────────────┐      ┌───┐                          »
        state_0: ┤0                ├──────┤ H ├──────────────────────────»
                 │                 │┌─────┴───┴─────┐     ┌───┐          »
        state_1: ┤1 UCRZ(0,pi,0,0) ├┤0              ├─────┤ H ├──────────»
                 │                 ││  UCRZ(pi/2,0) │┌────┴───┴────┐┌───┐»
        state_2: ┤2                ├┤1              ├┤ UCRZ(-pi/4) ├┤ H ├»
                 └─────────────────┘└───────────────┘└─────────────┘└───┘»
        «         ┌─────────────────┐      ┌───┐
        «state_0: ┤0                ├──────┤ H ├─────────────────────────
        «         │                 │┌─────┴───┴─────┐    ┌───┐
        «state_1: ┤1 UCRZ(pi,0,0,0) ├┤0              ├────┤ H ├──────────
        «         │                 ││  UCRZ(pi/2,0) │┌───┴───┴────┐┌───┐
        «state_2: ┤2                ├┤1              ├┤ UCRZ(pi/4) ├┤ H ├
        «         └─────────────────┘└───────────────┘└────────────┘└───┘

    .. seealso::

        The :func:`.grover_operator` implements the same functionality but keeping the
        :class:`.MCXGate` abstract, such that the compiler may choose the optimal decomposition.
        We recommend using :func:`.grover_operator` for performance reasons, which does not
        wrap the circuit into an opaque gate.

    References:

    [1] L. K. Grover (1996), A fast quantum mechanical algorithm for database search,
    `arXiv:quant-ph/9605043 <https://arxiv.org/abs/quant-ph/9605043>`_.

    [2] I. Chuang & M. Nielsen, Quantum Computation and Quantum Information,
    Cambridge: Cambridge University Press, 2000. Chapter 6.1.2.

    [3] Brassard, G., Hoyer, P., Mosca, M., & Tapp, A. (2000).
    Quantum Amplitude Amplification and Estimation.
    `arXiv:quant-ph/0005055 <https://arxiv.org/abs/quant-ph/0005055>`_.
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use qiskit.circuit.library.grover_operator instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        oracle: QuantumCircuit | Statevector,
        state_preparation: QuantumCircuit | None = None,
        zero_reflection: QuantumCircuit | DensityMatrix | Operator | None = None,
        reflection_qubits: list[int] | None = None,
        insert_barriers: bool = False,
        mcx_mode: str = "noancilla",
        name: str = "Q",
    ) -> None:
        r"""
        Args:
            oracle: The phase oracle implementing a reflection about the bad state. Note that this
                is not a bitflip oracle, see the docstring for more information.
            state_preparation: The operator preparing the good and bad state.
                For Grover's algorithm, this is a n-qubit Hadamard gate and for amplitude
                amplification or estimation the operator :math:`\mathcal{A}`.
            zero_reflection: The reflection about the zero state, :math:`\mathcal{S}_0`.
            reflection_qubits: Qubits on which the zero reflection acts on.
            insert_barriers: Whether barriers should be inserted between the reflections and A.
            mcx_mode: The mode to use for building the default zero reflection.
            name: The name of the circuit.

        """
        super().__init__(name=name)

        # store inputs
        if isinstance(oracle, Statevector):
            from qiskit.circuit.library import Diagonal

            oracle = Diagonal((-1) ** oracle.data)
        self._oracle = oracle

        if isinstance(zero_reflection, (Operator, DensityMatrix)):
            from qiskit.circuit.library import Diagonal

            zero_reflection = Diagonal(zero_reflection.data.diagonal())
        self._zero_reflection = zero_reflection

        self._reflection_qubits = reflection_qubits
        self._state_preparation = state_preparation
        self._insert_barriers = insert_barriers
        self._mcx_mode = mcx_mode

        # build circuit
        self._build()

    @property
    def reflection_qubits(self):
        """Reflection qubits, on which S0 is applied (if S0 is not user-specified)."""
        if self._reflection_qubits is not None:
            return self._reflection_qubits

        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        return list(range(num_state_qubits))

    @property
    def zero_reflection(self) -> QuantumCircuit:
        """The subcircuit implementing the reflection about 0."""
        if self._zero_reflection is not None:
            return self._zero_reflection

        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        return _zero_reflection(num_state_qubits, self.reflection_qubits, self._mcx_mode)

    @property
    def state_preparation(self) -> QuantumCircuit:
        """The subcircuit implementing the A operator or Hadamards."""
        if self._state_preparation is not None:
            return self._state_preparation

        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        hadamards = QuantumCircuit(num_state_qubits, name="H")
        # apply Hadamards only on reflection qubits, rest will cancel out
        hadamards.h(self.reflection_qubits)
        return hadamards

    @property
    def oracle(self):
        """The oracle implementing a reflection about the bad state."""
        return self._oracle

    def _build(self):
        num_state_qubits = self.oracle.num_qubits - self.oracle.num_ancillas
        circuit = QuantumCircuit(QuantumRegister(num_state_qubits, name="state"), name="Q")
        num_ancillas = numpy.max(
            [
                self.oracle.num_ancillas,
                self.zero_reflection.num_ancillas,
                self.state_preparation.num_ancillas,
            ]
        )
        if num_ancillas > 0:
            circuit.add_register(AncillaRegister(num_ancillas, name="ancilla"))

        circuit.compose(self.oracle, list(range(self.oracle.num_qubits)), inplace=True)
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(
            self.state_preparation.inverse(),
            list(range(self.state_preparation.num_qubits)),
            inplace=True,
        )
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(
            self.zero_reflection, list(range(self.zero_reflection.num_qubits)), inplace=True
        )
        if self._insert_barriers:
            circuit.barrier()
        circuit.compose(
            self.state_preparation, list(range(self.state_preparation.num_qubits)), inplace=True
        )

        # minus sign
        circuit.global_phase = numpy.pi

        self.add_register(*circuit.qregs)
        try:
            circuit_wrapped = circuit.to_gate()
        except QiskitError:
            circuit_wrapped = circuit.to_instruction()

        self.compose(circuit_wrapped, qubits=self.qubits, inplace=True)


# TODO use the oracle compiler or the bit string oracle
def _zero_reflection(
    num_state_qubits: int, qubits: list[int], mcx_mode: str | None = None
) -> QuantumCircuit:
    qr_state = QuantumRegister(num_state_qubits, "state")
    reflection = QuantumCircuit(qr_state, name="S_0")

    num_ancillas = MCXGate.get_num_ancilla_qubits(len(qubits) - 1, mcx_mode)
    if num_ancillas > 0:
        qr_ancilla = AncillaRegister(num_ancillas, "ancilla")
        reflection.add_register(qr_ancilla)
    else:
        qr_ancilla = AncillaRegister(0)

    reflection.x(qubits)
    if len(qubits) == 1:
        reflection.z(0)  # MCX does not allow 0 control qubits, therefore this is separate
    else:
        reflection.h(qubits[-1])
        reflection.mcx(qubits[:-1], qubits[-1], qr_ancilla[:], mode=mcx_mode)
        reflection.h(qubits[-1])
    reflection.x(qubits)

    return reflection
