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

"""The EfficientSU2 2-local circuit."""

from __future__ import annotations
import typing
from collections.abc import Callable, Iterable

from numpy import pi

from qiskit.circuit import QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import RYGate, RZGate, CXGate
from qiskit.utils.deprecation import deprecate_func
from .n_local import n_local, BlockEntanglement
from .two_local import TwoLocal

if typing.TYPE_CHECKING:
    import qiskit


def efficient_su2(
    num_qubits: int,
    su2_gates: str | Gate | Iterable[str | Gate] | None = None,
    entanglement: (
        BlockEntanglement
        | Iterable[BlockEntanglement]
        | Callable[[int], BlockEntanglement | Iterable[BlockEntanglement]]
    ) = "reverse_linear",
    reps: int = 3,
    skip_unentangled_qubits: bool = False,
    skip_final_rotation_layer: bool = False,
    parameter_prefix: str = "θ",
    insert_barriers: bool = False,
    name: str = "EfficientSU2",
) -> QuantumCircuit:
    r"""The hardware-efficient :math:`SU(2)` 2-local circuit.

    The ``efficient_su2`` circuit consists of layers of single qubit operations spanned by
    :math:`SU(2)` and CX entanglements. This is a heuristic pattern that can be used to prepare trial
    wave functions for variational quantum algorithms or classification circuit for machine learning.

    :math:`SU(2)` is the special unitary group of degree 2, its elements are :math:`2 \times 2`
    unitary matrices with determinant 1, such as the Pauli rotation gates.

    On 3 qubits and using the Pauli :math:`Y` and :math:`Z` rotations as single qubit gates,
    this circuit is represented by:

    .. parsed-literal::

        ┌──────────┐┌──────────┐ ░            ░       ░ ┌───────────┐┌───────────┐
        ┤ RY(θ[0]) ├┤ RZ(θ[3]) ├─░────────■───░─ ... ─░─┤ RY(θ[12]) ├┤ RZ(θ[15]) ├
        ├──────────┤├──────────┤ ░      ┌─┴─┐ ░       ░ ├───────────┤├───────────┤
        ┤ RY(θ[1]) ├┤ RZ(θ[4]) ├─░───■──┤ X ├─░─ ... ─░─┤ RY(θ[13]) ├┤ RZ(θ[16]) ├
        ├──────────┤├──────────┤ ░ ┌─┴─┐└───┘ ░       ░ ├───────────┤├───────────┤
        ┤ RY(θ[2]) ├┤ RZ(θ[5]) ├─░─┤ X ├──────░─ ... ─░─┤ RY(θ[14]) ├┤ RZ(θ[17]) ├
        └──────────┘└──────────┘ ░ └───┘      ░       ░ └───────────┘└───────────┘

    Examples:

    Per default, the ``"reverse_linear"`` entanglement is used, which, in the case of
    CX gates, is equivalent to an all-to-all entanglement:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context:

        from qiskit.circuit.library import efficient_su2

        circuit = efficient_su2(3, reps=1)
        circuit.draw("mpl")

    To specify which SU(2) gates should be used in the rotation layer, we can set the
    ``su2_gates`` argument. In addition, we can change the entanglement structure.
    For example:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        circuit = efficient_su2(4, su2_gates=["rx", "y"], entanglement="circular", reps=1)
        circuit.draw("mpl")

    Args:
        num_qubits: The number of qubits.
        su2_gates: The :math:`SU(2)` single qubit gates to apply in single qubit gate layers.
            If only one gate is provided, the same gate is applied to each qubit.
            If a list of gates is provided, all gates are applied to each qubit in the provided
            order.
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
        An efficient-SU(2) circuit.
    """
    if su2_gates is None:
        su2_gates = ["ry", "rz"]

    # Set entanglement_blocks to None when num_qubits == 1
    entanglement_blocks = ["cx"] if num_qubits > 1 else []

    return n_local(
        num_qubits,
        su2_gates,
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


class EfficientSU2(TwoLocal):
    r"""The hardware efficient SU(2) 2-local circuit.

    The ``EfficientSU2`` circuit consists of layers of single qubit operations spanned by SU(2)
    and :math:`CX` entanglements. This is a heuristic pattern that can be used to prepare trial wave
    functions for variational quantum algorithms or classification circuit for machine learning.

    SU(2) stands for special unitary group of degree 2, its elements are :math:`2 \times 2`
    unitary matrices with determinant 1, such as the Pauli rotation gates.

    On 3 qubits and using the Pauli :math:`Y` and :math:`Z` su2_gates as single qubit gates, the
    hardware efficient SU(2) circuit is represented by:

    .. code-block:: text

        ┌──────────┐┌──────────┐ ░            ░       ░ ┌───────────┐┌───────────┐
        ┤ RY(θ[0]) ├┤ RZ(θ[3]) ├─░────────■───░─ ... ─░─┤ RY(θ[12]) ├┤ RZ(θ[15]) ├
        ├──────────┤├──────────┤ ░      ┌─┴─┐ ░       ░ ├───────────┤├───────────┤
        ┤ RY(θ[1]) ├┤ RZ(θ[4]) ├─░───■──┤ X ├─░─ ... ─░─┤ RY(θ[13]) ├┤ RZ(θ[16]) ├
        ├──────────┤├──────────┤ ░ ┌─┴─┐└───┘ ░       ░ ├───────────┤├───────────┤
        ┤ RY(θ[2]) ├┤ RZ(θ[5]) ├─░─┤ X ├──────░─ ... ─░─┤ RY(θ[14]) ├┤ RZ(θ[17]) ├
        └──────────┘└──────────┘ ░ └───┘      ░       ░ └───────────┘└───────────┘

    See :class:`~qiskit.circuit.library.RealAmplitudes` for more detail on the possible arguments
    and options such as skipping unentangled qubits, which apply here too.

    Examples:

        >>> circuit = EfficientSU2(3, reps=1)
        >>> print(circuit.decompose())
             ┌──────────┐┌──────────┐          ┌──────────┐┌──────────┐
        q_0: ┤ RY(θ[0]) ├┤ RZ(θ[3]) ├──■────■──┤ RY(θ[6]) ├┤ RZ(θ[9]) ├─────────────
             ├──────────┤├──────────┤┌─┴─┐  │  └──────────┘├──────────┤┌───────────┐
        q_1: ┤ RY(θ[1]) ├┤ RZ(θ[4]) ├┤ X ├──┼───────■──────┤ RY(θ[7]) ├┤ RZ(θ[10]) ├
             ├──────────┤├──────────┤└───┘┌─┴─┐   ┌─┴─┐    ├──────────┤├───────────┤
        q_2: ┤ RY(θ[2]) ├┤ RZ(θ[5]) ├─────┤ X ├───┤ X ├────┤ RY(θ[8]) ├┤ RZ(θ[11]) ├
             └──────────┘└──────────┘     └───┘   └───┘    └──────────┘└───────────┘

        >>> ansatz = EfficientSU2(4, su2_gates=['rx', 'y'], entanglement='circular', reps=1,
        ... flatten=True)
        >>> qc = QuantumCircuit(4)  # create a circuit and append the RY variational form
        >>> qc.compose(ansatz, inplace=True)
        >>> qc.draw()
             ┌──────────┐┌───┐┌───┐     ┌──────────┐   ┌───┐
        q_0: ┤ RX(θ[0]) ├┤ Y ├┤ X ├──■──┤ RX(θ[4]) ├───┤ Y ├─────────────────────
             ├──────────┤├───┤└─┬─┘┌─┴─┐└──────────┘┌──┴───┴───┐   ┌───┐
        q_1: ┤ RX(θ[1]) ├┤ Y ├──┼──┤ X ├─────■──────┤ RX(θ[5]) ├───┤ Y ├─────────
             ├──────────┤├───┤  │  └───┘   ┌─┴─┐    └──────────┘┌──┴───┴───┐┌───┐
        q_2: ┤ RX(θ[2]) ├┤ Y ├──┼──────────┤ X ├─────────■──────┤ RX(θ[6]) ├┤ Y ├
             ├──────────┤├───┤  │          └───┘       ┌─┴─┐    ├──────────┤├───┤
        q_3: ┤ RX(θ[3]) ├┤ Y ├──■──────────────────────┤ X ├────┤ RX(θ[7]) ├┤ Y ├
             └──────────┘└───┘                         └───┘    └──────────┘└───┘

    .. seealso::

        The :func:`.efficient_su2` function constructs a functionally equivalent circuit, but faster.

    """

    @deprecate_func(
        since="2.1",
        additional_msg="Use the function qiskit.circuit.library.efficient_su2 instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int | None = None,
        su2_gates: (
            str
            | type
            | qiskit.circuit.Instruction
            | QuantumCircuit
            | list[str | type | qiskit.circuit.Instruction | QuantumCircuit]
            | None
        ) = None,
        entanglement: str | list[list[int]] | Callable[[int], list[int]] = "reverse_linear",
        reps: int = 3,
        skip_unentangled_qubits: bool = False,
        skip_final_rotation_layer: bool = False,
        parameter_prefix: str = "θ",
        insert_barriers: bool = False,
        initial_state: QuantumCircuit | None = None,
        name: str = "EfficientSU2",
        flatten: bool | None = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits of the EfficientSU2 circuit.
            reps: Specifies how often the structure of a rotation layer followed by an entanglement
                layer is repeated.
            su2_gates: The SU(2) single qubit gates to apply in single qubit gate layers.
                If only one gate is provided, the same gate is applied to each qubit.
                If a list of gates is provided, all gates are applied to each qubit in the provided
                order.
            entanglement: Specifies the entanglement structure. Can be a string
                ('full', 'linear', 'reverse_linear', 'pairwise', 'circular', or 'sca'),
                a list of integer-pairs specifying the indices of qubits entangled with one another,
                or a callable returning such a list provided with the index of the entanglement layer.
                Defaults to 'reverse_linear' entanglement.
                Note that 'reverse_linear' entanglement provides the same unitary as 'full'
                with fewer entangling gates.
                See the Examples section of :class:`~qiskit.circuit.library.TwoLocal` for more
                detail.
            initial_state: A `QuantumCircuit` object to prepend to the circuit.
            skip_unentangled_qubits: If True, the single qubit gates are only applied to qubits
                that are entangled with another qubit. If False, the single qubit gates are applied
                to each qubit in the Ansatz. Defaults to False.
            skip_final_rotation_layer: If False, a rotation layer is added at the end of the
                ansatz. If True, no rotation layer is added.
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

        """
        if su2_gates is None:
            su2_gates = [RYGate, RZGate]
        super().__init__(
            num_qubits=num_qubits,
            rotation_blocks=su2_gates,
            entanglement_blocks=CXGate,
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
