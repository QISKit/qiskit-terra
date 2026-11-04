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

"""The ExcitationPreserving 2-local circuit."""

from __future__ import annotations
from collections.abc import Callable, Iterable
from numpy import pi

from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import RZGate, XXPlusYYGate
from qiskit.utils.deprecation import deprecate_func
from .n_local import n_local, BlockEntanglement
from .two_local import TwoLocal


def excitation_preserving(
    num_qubits: int,
    mode: str = "iswap",
    entanglement: (
        BlockEntanglement
        | Iterable[BlockEntanglement]
        | Callable[[int], BlockEntanglement | Iterable[BlockEntanglement]]
    ) = "full",
    reps: int = 3,
    skip_unentangled_qubits: bool = False,
    skip_final_rotation_layer: bool = False,
    parameter_prefix: str = "θ",
    insert_barriers: bool = False,
    name: str = "ExcitationPreserving",
) -> QuantumCircuit:
    r"""The heuristic excitation-preserving wave function ansatz.

    The ``excitation_preserving`` circuit preserves the ratio of :math:`|00\rangle`,
    :math:`|01\rangle + |10\rangle` and :math:`|11\rangle` states. To this end, this circuit
    uses two-qubit interactions of the form

    .. math::

        \newcommand{\rotationangle}{\theta/2}

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \cos\left(\rotationangle\right) & -i\sin\left(\rotationangle\right) & 0 \\
        0 & -i\sin\left(\rotationangle\right) & \cos\left(\rotationangle\right) & 0 \\
        0 & 0 & 0 & e^{-i\phi}
        \end{pmatrix}

    for the mode ``"fsim"`` or with :math:`e^{-i\phi} = 1` for the mode ``"iswap"``.

    Note that other wave functions, such as UCC-ansatzes, are also excitation preserving.
    However these can become complex quickly, while this heuristically motivated circuit follows
    a simpler pattern.

    This trial wave function consists of layers of :math:`Z` rotations with 2-qubit entanglements.
    The entanglement is created using :math:`XX+YY` rotations and optionally a controlled-phase
    gate for the mode ``"fsim"``.

    Examples:

    With linear entanglement, this circuit is given by:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        from qiskit.circuit.library import excitation_preserving

        ansatz = excitation_preserving(3, reps=1, insert_barriers=True, entanglement="linear")
        ansatz.draw("mpl")

    The entanglement structure can be explicitly specified with the ``entanglement``
    argument. The ``"fsim"`` mode includes an additional parameterized :class:`.CPhaseGate`
    in each block:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context:

        ansatz = excitation_preserving(3, reps=1, mode="fsim", entanglement=[[0, 2]])
        ansatz.draw("mpl")

    Args:
        num_qubits: The number of qubits.
        mode: Choose the entangler mode, can be `"iswap"` or `"fsim"`.
        reps: Specifies how often the structure of a rotation layer followed by an entanglement
            layer is repeated.
        entanglement: The indices specifying on which qubits the input blocks act.
            See :func:`.n_local` for detailed information.
        skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.
        skip_unentangled_qubits: If ``True``, the rotation gates act only on qubits that
            are entangled. If ``False``, the rotation gates act on all qubits.
        parameter_prefix: The name of the free parameters.
        insert_barriers: If True, barriers are inserted in between each layer. If False,
            no barriers are inserted.
        name: The name of the circuit.

    Returns:
        An excitation-preserving circuit.
    """
    supported_modes = ["iswap", "fsim"]
    if mode not in supported_modes:
        raise ValueError(f"Unsupported mode {mode}, choose one of {supported_modes}")

    theta = Parameter("θ")
    if num_qubits > 1:
        swap = QuantumCircuit(2, name="Interaction")
        swap.append(XXPlusYYGate(2 * theta), [0, 1])
        if mode == "fsim":
            phi = Parameter("φ")
            swap.cp(phi, 0, 1)
        entanglement_blocks = [swap.to_gate()]
    else:
        entanglement_blocks = []

    return n_local(
        num_qubits,
        ["rz"],
        entanglement_blocks,
        entanglement,
        reps,
        insert_barriers,
        parameter_prefix,
        True,
        skip_final_rotation_layer,
        skip_unentangled_qubits,
        name,
    )


class ExcitationPreserving(TwoLocal):
    r"""The heuristic excitation-preserving wave function ansatz.

    The ``ExcitationPreserving`` circuit preserves the ratio of :math:`|00\rangle`,
    :math:`|01\rangle + |10\rangle` and :math:`|11\rangle` states. To this end, this circuit
    uses two-qubit interactions of the form

    .. math::

        \newcommand{\rotationangle}{\theta/2}

        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & \cos\left(\rotationangle\right) & -i\sin\left(\rotationangle\right) & 0 \\
        0 & -i\sin\left(\rotationangle\right) & \cos\left(\rotationangle\right) & 0 \\
        0 & 0 & 0 & e^{-i\phi}
        \end{pmatrix}

    for the mode ``'fsim'`` or with :math:`e^{-i\phi} = 1` for the mode ``'iswap'``.

    Note that other wave functions, such as UCC-ansatzes, are also excitation preserving.
    However these can become complex quickly, while this heuristically motivated circuit follows
    a simpler pattern.

    This trial wave function consists of layers of :math:`Z` rotations with 2-qubit entanglements.
    The entanglement is created using :math:`XX+YY` rotations and optionally a controlled-phase
    gate for the mode ``'fsim'``.

    See :class:`~qiskit.circuit.library.RealAmplitudes` for more detail on the possible arguments
    and options such as skipping unentangled qubits, which apply here too.

    The rotations of the ExcitationPreserving ansatz can be written as

    Examples:

        >>> ansatz = ExcitationPreserving(3, reps=1, insert_barriers=True, entanglement='linear')
        >>> print(ansatz.decompose())  # show the circuit
             ┌──────────┐ ░ ┌────────────┐┌────────────┐                             ░ ┌──────────┐
        q_0: ┤ RZ(θ[0]) ├─░─┤0           ├┤0           ├─────────────────────────────░─┤ RZ(θ[5]) ├
             ├──────────┤ ░ │  RXX(θ[3]) ││  RYY(θ[3]) │┌────────────┐┌────────────┐ ░ ├──────────┤
        q_1: ┤ RZ(θ[1]) ├─░─┤1           ├┤1           ├┤0           ├┤0           ├─░─┤ RZ(θ[6]) ├
             ├──────────┤ ░ └────────────┘└────────────┘│  RXX(θ[4]) ││  RYY(θ[4]) │ ░ ├──────────┤
        q_2: ┤ RZ(θ[2]) ├─░─────────────────────────────┤1           ├┤1           ├─░─┤ RZ(θ[7]) ├
             └──────────┘ ░                             └────────────┘└────────────┘ ░ └──────────┘

        >>> ansatz = ExcitationPreserving(2, reps=1, flatten=True)
        >>> qc = QuantumCircuit(2)  # create a circuit and append the RY variational form
        >>> qc.cry(0.2, 0, 1)  # do some previous operation
        >>> qc.compose(ansatz, inplace=True)  # add the excitation-preserving
        >>> qc.draw()
                        ┌──────────┐┌────────────┐┌────────────┐┌──────────┐
        q_0: ─────■─────┤ RZ(θ[0]) ├┤0           ├┤0           ├┤ RZ(θ[3]) ├
             ┌────┴────┐├──────────┤│  RXX(θ[2]) ││  RYY(θ[2]) │├──────────┤
        q_1: ┤ RY(0.2) ├┤ RZ(θ[1]) ├┤1           ├┤1           ├┤ RZ(θ[4]) ├
             └─────────┘└──────────┘└────────────┘└────────────┘└──────────┘

        >>> ansatz = ExcitationPreserving(3, reps=1, mode='fsim', entanglement=[[0,2]],
        ... insert_barriers=True, flatten=True)
        >>> print(ansatz.decompose())
             ┌──────────┐ ░ ┌────────────┐┌────────────┐        ░ ┌──────────┐
        q_0: ┤ RZ(θ[0]) ├─░─┤0           ├┤0           ├─■──────░─┤ RZ(θ[5]) ├
             ├──────────┤ ░ │            ││            │ │      ░ ├──────────┤
        q_1: ┤ RZ(θ[1]) ├─░─┤  RXX(θ[3]) ├┤  RYY(θ[3]) ├─┼──────░─┤ RZ(θ[6]) ├
             ├──────────┤ ░ │            ││            │ │θ[4]  ░ ├──────────┤
        q_2: ┤ RZ(θ[2]) ├─░─┤1           ├┤1           ├─■──────░─┤ RZ(θ[7]) ├
             └──────────┘ ░ └────────────┘└────────────┘        ░ └──────────┘

    .. seealso::

        The :func:`.excitation_preserving` function constructs a functionally equivalent circuit,
        but faster.

    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the function qiskit.circuit.library.excitation_preserving instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int | None = None,
        mode: str = "iswap",
        entanglement: str | list[list[int]] | Callable[[int], list[int]] = "full",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: QuantumCircuit | None = None,
        name: str = "ExcitationPreserving",
        flatten: bool | None = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits of the ExcitationPreserving circuit.
            mode: Choose the entangler mode, can be `'iswap'` or `'fsim'`.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            entanglement: Specifies the entanglement structure. Can be a string ('full', 'linear'
                or 'sca'), a list of integer-pairs specifying the indices of qubits
                entangled with one another, or a callable returning such a list provided with
                the index of the entanglement layer.
                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
                detail.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If True, a rotation layer is added at the end of the
                ansatz. If False, no rotation layer is added. Defaults to True.
            parameter_prefix: The parameterized gates require a parameter to be defined, for which
                we use :class:`~qiskit.circuit.ParameterVector`.
            insert_barriers: If True, barriers are inserted in between each layer. If False,
                no barriers are inserted.
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.
            name: Name of the circuit.

        Raises:
            ValueError: If the selected mode is not supported.
        """
        supported_modes = ["iswap", "fsim"]
        if mode not in supported_modes:
            raise ValueError(f"Unsupported mode {mode}, choose one of {supported_modes}")

        theta = Parameter("θ")
        swap = QuantumCircuit(2, name="Interaction")
        swap.append(XXPlusYYGate(2 * theta), [0, 1])
        if mode == "fsim":
            phi = Parameter("φ")
            swap.cp(phi, 0, 1)

        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=RZGate,
            entanglement_blocks=swap,
            entanglement=entanglement,
            reps=reps,
            skip_unentangled_qubits=skip_unentangled_qubits,
            skip_final_rotation_layer=skip_final_rotation_layer,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            initial_state=initial_state,
            name=name,
            flatten=flatten,
        )

    @property
    def parameter_bounds(self) -> list[tuple[float, float]]:
        """Return the parameter bounds.

        Returns:
            The parameter bounds.
        """
        return self.num_parameters * [(-pi, pi)]
