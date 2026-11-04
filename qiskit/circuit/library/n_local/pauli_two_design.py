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

"""The Random Pauli circuit class."""

from __future__ import annotations
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate, CZGate
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.circuit_library import Block, py_n_local
from .two_local import TwoLocal


def pauli_two_design(
    num_qubits: int,
    reps: int = 3,
    seed: int | None = None,
    insert_barriers: bool = False,
    parameter_prefix: str = "θ",
    name: str = "PauliTwoDesign",
) -> QuantumCircuit:
    r"""Construct a Pauli 2-design ansatz.

    This class implements a particular form of a 2-design circuit [1], which is frequently studied
    in quantum machine learning literature, such as, e.g., the investigation of Barren plateaus in
    variational algorithms [2].

    The circuit consists of alternating rotation and entanglement layers with
    an initial layer of :math:`\sqrt{H} = RY(\pi/4)` gates.
    The rotation layers contain single qubit Pauli rotations, where the axis is chosen uniformly
    at random to be X, Y or Z. The entanglement layers are comprised of pairwise CZ gates
    with a total depth of 2.

    For instance, the circuit could look like this:

    .. parsed-literal::

             ┌─────────┐┌──────────┐       ░ ┌──────────┐       ░  ┌──────────┐
        q_0: ┤ RY(π/4) ├┤ RZ(θ[0]) ├─■─────░─┤ RY(θ[4]) ├─■─────░──┤ RZ(θ[8]) ├
             ├─────────┤├──────────┤ │     ░ ├──────────┤ │     ░  ├──────────┤
        q_1: ┤ RY(π/4) ├┤ RZ(θ[1]) ├─■──■──░─┤ RY(θ[5]) ├─■──■──░──┤ RX(θ[9]) ├
             ├─────────┤├──────────┤    │  ░ ├──────────┤    │  ░ ┌┴──────────┤
        q_2: ┤ RY(π/4) ├┤ RX(θ[2]) ├─■──■──░─┤ RY(θ[6]) ├─■──■──░─┤ RX(θ[10]) ├
             ├─────────┤├──────────┤ │     ░ ├──────────┤ │     ░ ├───────────┤
        q_3: ┤ RY(π/4) ├┤ RZ(θ[3]) ├─■─────░─┤ RX(θ[7]) ├─■─────░─┤ RY(θ[11]) ├
             └─────────┘└──────────┘       ░ └──────────┘       ░ └───────────┘

    Examples:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:

        from qiskit.circuit.library import pauli_two_design
        circuit = pauli_two_design(4, reps=2, seed=5, insert_barriers=True)
        circuit.draw("mpl")

    Args:
        num_qubits: The number of qubits of the Pauli Two-Design circuit.
        reps: Specifies how often a block consisting of a rotation layer and entanglement
            layer is repeated.
        seed: The seed for randomly choosing the axes of the Pauli rotations.
        parameter_prefix: The prefix used for the rotation parameters.
        insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
            no barriers are inserted. Defaults to ``False``.
        name: The circuit name.

    Returns:
        A Pauli 2-design circuit.

    References:

    [1] Nakata et al., Unitary 2-designs from random X- and Z-diagonal unitaries.
    `arXiv:1502.07514 <https://arxiv.org/pdf/1502.07514.pdf>`_

    [2] McClean et al., Barren plateaus in quantum neural network training landscapes.
    `arXiv:1803.11173 <https://arxiv.org/pdf/1803.11173.pdf>`_
    """
    rng = np.random.default_rng(seed)
    random_block = Block.from_callable(1, 1, lambda params: _random_pauli_builder(params, rng))
    entanglement_block = [Block.from_standard_gate(CZGate._standard_gate)] if num_qubits > 1 else []

    data = py_n_local(
        num_qubits=num_qubits,
        reps=reps,
        rotation_blocks=[random_block],
        entanglement_blocks=entanglement_block,
        entanglement=["pairwise"],
        insert_barriers=insert_barriers,
        skip_final_rotation_layer=False,
        skip_unentangled_qubits=False,
        parameter_prefix=parameter_prefix,
    )

    two_design = QuantumCircuit._from_circuit_data(data)

    circuit = QuantumCircuit(num_qubits, name=name)
    circuit.ry(np.pi / 4, circuit.qubits)
    circuit.compose(two_design, inplace=True, copy=False)

    return circuit


def _random_pauli_builder(params, rng):
    gate_cls = rng.choice([RXGate, RYGate, RZGate])
    gate = gate_cls(params[0])
    return gate, gate.params


class PauliTwoDesign(TwoLocal):
    r"""The Pauli Two-Design ansatz.

    This class implements a particular form of a 2-design circuit [1], which is frequently studied
    in quantum machine learning literature, such as the investigation of barren plateaus in
    variational algorithms [2].

    The circuit consists of alternating rotation and entanglement layers with
    an initial layer of :math:`\sqrt{H} = RY(\pi/4)` gates.
    The rotation layers contain single qubit Pauli rotations, where the axis is chosen uniformly
    at random to be X, Y or Z. The entanglement layers are comprised of pairwise CZ gates
    with a total depth of 2.

    For instance, the circuit could look like this (but note that choosing a different seed
    yields different Pauli rotations).

    .. code-block:: text

             ┌─────────┐┌──────────┐       ░ ┌──────────┐       ░  ┌──────────┐
        q_0: ┤ RY(π/4) ├┤ RZ(θ[0]) ├─■─────░─┤ RY(θ[4]) ├─■─────░──┤ RZ(θ[8]) ├
             ├─────────┤├──────────┤ │     ░ ├──────────┤ │     ░  ├──────────┤
        q_1: ┤ RY(π/4) ├┤ RZ(θ[1]) ├─■──■──░─┤ RY(θ[5]) ├─■──■──░──┤ RX(θ[9]) ├
             ├─────────┤├──────────┤    │  ░ ├──────────┤    │  ░ ┌┴──────────┤
        q_2: ┤ RY(π/4) ├┤ RX(θ[2]) ├─■──■──░─┤ RY(θ[6]) ├─■──■──░─┤ RX(θ[10]) ├
             ├─────────┤├──────────┤ │     ░ ├──────────┤ │     ░ ├───────────┤
        q_3: ┤ RY(π/4) ├┤ RZ(θ[3]) ├─■─────░─┤ RX(θ[7]) ├─■─────░─┤ RY(θ[11]) ├
             └─────────┘└──────────┘       ░ └──────────┘       ░ └───────────┘

    Examples:

        .. plot::
           :alt: Circuit diagram output by the previous code.
           :include-source:

           from qiskit.circuit.library import PauliTwoDesign
           circuit = PauliTwoDesign(4, reps=2, seed=5, insert_barriers=True)
           circuit.draw('mpl')

    .. seealso::

        The :func:`.pauli_two_design` function constructs the functionally same circuit, but faster.

    References:

        [1]: Nakata et al., Unitary 2-designs from random X- and Z-diagonal unitaries.
            `arXiv:1502.07514 <https://arxiv.org/pdf/1502.07514.pdf>`_

        [2]: McClean et al., Barren plateaus in quantum neural network training landscapes.
             `arXiv:1803.11173 <https://arxiv.org/pdf/1803.11173.pdf>`_
    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the function qiskit.circuit.library.pauli_two_design instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int | None = None,
        reps: int = 3,
        seed: int | None = None,
        insert_barriers: bool = False,
        name: str = "PauliTwoDesign",
    ):
        """
        Args:
            num_qubits: The number of qubits of the Pauli Two-Design circuit.
            reps: Specifies how often a block consisting of a rotation layer and entanglement
                layer is repeated.
            seed: The seed for randomly choosing the axes of the Pauli rotations.
            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
                no barriers are inserted. Defaults to ``False``.
            name: The name to use for the generated circuit.

        """
        # store a random number generator
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # store a dict to keep track of the random gates
        self._gates: dict[int, list[str]] = {}

        super().__init__(
            num_qubits,
            reps=reps,
            entanglement_blocks="cz",
            entanglement="pairwise",
            insert_barriers=insert_barriers,
            name=name,
        )

        # set the initial layer
        self._prepended_blocks = [RYGate(np.pi / 4)]
        self._prepended_entanglement = ["linear"]

    def _invalidate(self):
        """Invalidate the circuit and reset the random number."""
        self._rng = np.random.default_rng(self._seed)  # reset number generator
        super()._invalidate()

    def _build_rotation_layer(self, circuit, param_iter, i):
        """Build a rotation layer."""
        layer = QuantumCircuit(*self.qregs)
        qubits = range(self.num_qubits)

        # if no gates for this layer were generated, generate them
        if i not in self._gates:
            self._gates[i] = list(self._rng.choice(["rx", "ry", "rz"], self.num_qubits))
        # if not enough gates exist, add more
        elif len(self._gates[i]) < self.num_qubits:
            num_missing = self.num_qubits - len(self._gates[i])
            self._gates[i] += list(self._rng.choice(["rx", "ry", "rz"], num_missing))

        for j in qubits:
            getattr(layer, self._gates[i][j])(next(param_iter), j)

        # add the layer to the circuit
        circuit.compose(layer, inplace=True)

    @property
    def num_parameters_settable(self) -> int:
        """Return the number of settable parameters.

        Returns:
            The number of possibly distinct parameters.
        """
        return (self.reps + 1) * self.num_qubits
