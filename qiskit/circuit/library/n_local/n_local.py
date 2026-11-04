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

"""The n-local circuit class."""

from __future__ import annotations

import collections
import itertools
import typing
from collections.abc import Callable, Mapping, Sequence, Iterable

import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit, ParameterValueType
from qiskit.circuit.parametervector import ParameterVector, ParameterVectorElement
from qiskit.circuit import QuantumRegister
from qiskit.circuit import (
    Instruction,
    Parameter,
    ParameterExpression,
    CircuitInstruction,
)
from qiskit.exceptions import QiskitError
from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
from qiskit.utils.deprecation import deprecate_func

from qiskit._accelerate.circuit_library import (
    Block,
    py_n_local,
    get_entangler_map as fast_entangler_map,
)

from ..blueprintcircuit import BlueprintCircuit


if typing.TYPE_CHECKING:
    import qiskit

# entanglement for an individual block, e.g. if the block is CXGate() and we have
# 3 qubits, this could be [(0, 1), (1, 2), (2, 0)]
BlockEntanglement = str | Iterable[Iterable[int]]


def n_local(
    num_qubits: int,
    rotation_blocks: str | Gate | Iterable[str | Gate],
    entanglement_blocks: str | Gate | Iterable[str | Gate],
    entanglement: (
        BlockEntanglement
        | Iterable[BlockEntanglement]
        | Callable[[int], BlockEntanglement | Iterable[BlockEntanglement]]
    ) = "full",
    reps: int = 3,
    insert_barriers: bool = False,
    parameter_prefix: str = "θ",
    overwrite_block_parameters: bool = True,
    skip_final_rotation_layer: bool = False,
    skip_unentangled_qubits: bool = False,
    name: str | None = "nlocal",
) -> QuantumCircuit:
    r"""Construct an n-local variational circuit.

    The structure of the n-local circuit are alternating rotation and entanglement layers.
    In both layers, parameterized circuit-blocks act on the circuit in a defined way.
    In the rotation layer, the blocks are applied stacked on top of each other, while in the
    entanglement layer according to the ``entanglement`` strategy.
    The circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the
    circuit). Each layer is repeated ``reps`` times, and by default a final rotation layer is
    appended.

    For instance, a rotation block on 2 qubits and an entanglement block on 4 qubits using
    ``"linear"`` entanglement yields the following circuit.

    .. parsed-literal::

        ┌──────┐ ░ ┌──────┐                      ░ ┌──────┐
        ┤0     ├─░─┤0     ├──────────────── ... ─░─┤0     ├
        │  Rot │ ░ │      │┌──────┐              ░ │  Rot │
        ┤1     ├─░─┤1     ├┤0     ├──────── ... ─░─┤1     ├
        ├──────┤ ░ │  Ent ││      │┌──────┐      ░ ├──────┤
        ┤0     ├─░─┤2     ├┤1     ├┤0     ├ ... ─░─┤0     ├
        │  Rot │ ░ │      ││  Ent ││      │      ░ │  Rot │
        ┤1     ├─░─┤3     ├┤2     ├┤1     ├ ... ─░─┤1     ├
        ├──────┤ ░ └──────┘│      ││  Ent │      ░ ├──────┤
        ┤0     ├─░─────────┤3     ├┤2     ├ ... ─░─┤0     ├
        │  Rot │ ░         └──────┘│      │      ░ │  Rot │
        ┤1     ├─░─────────────────┤3     ├ ... ─░─┤1     ├
        └──────┘ ░                 └──────┘      ░ └──────┘

        |                                 |
        +---------------------------------+
               repeated reps times

    Entanglement:

    The entanglement describes the connections of the gates in the entanglement layer.
    For a two-qubit gate for example, the entanglement contains pairs of qubits on which the
    gate should acts, e.g. ``[[ctrl0, target0], [ctrl1, target1], ...]``.
    A set of default entanglement strategies is provided and can be selected by name:

    * ``"full"`` entanglement is where each qubit is entangled with all the others.
    * ``"linear"`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
        for all :math:`i \in \{0, 1, ... , n - 2\}`, where :math:`n` is the total number of qubits.
    * ``"reverse_linear"`` entanglement is qubit :math:`i` entangled with qubit :math:`i + 1`,
        for all :math:`i \in \{n-2, n-3, ... , 1, 0\}`, where :math:`n` is the total number of qubits.
        Note that if ``entanglement_blocks=="cx"`` then this option provides the same unitary as
        ``"full"`` with fewer entangling gates.
    * ``"pairwise"`` entanglement is one layer where qubit :math:`i` is entangled with qubit
        :math:`i + 1`, for all even values of :math:`i`, and then a second layer where qubit :math:`i`
        is entangled with qubit :math:`i + 1`, for all odd values of :math:`i`.
    * ``"circular"`` entanglement is linear entanglement but with an additional entanglement of the
        first and last qubit before the linear part.
    * ``"sca"`` (shifted-circular-alternating) entanglement is a generalized and modified version
        of the proposed circuit 14 in `Sim et al. <https://arxiv.org/abs/1905.10876>`__.
        It consists of circular entanglement where the "long" entanglement connecting the first with
        the last qubit is shifted by one each block.  Furthermore the role of control and target
        qubits are swapped every block (therefore alternating).

    If an entanglement layer contains multiple blocks, then the entanglement should be
    given as list of entanglements for each block. For example::

        entanglement_blocks = ["rxx", "ryy"]
        entanglement = ["full", "linear"]  # full for rxx and linear for ryy

    or::

        structure_rxx = [[0, 1], [2, 3]]
        structure_ryy = [[0, 2]]
        entanglement = [structure_rxx, structure_ryy]

    Finally, the entanglement can vary in each repetition of the circuit. For this, we
    support passing a callable that takes as input the layer index and returns the entanglement
    for the layer in the above format. See the examples below for a concrete example.

    Examples:

    The rotation and entanglement gates can be specified via single strings, if they
    are made up of a single block per layer:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context:

        from qiskit.circuit.library import n_local

        circuit = n_local(3, "ry", "cx", "linear", reps=2, insert_barriers=True)
        circuit.draw("mpl")

    Multiple gates per layer can be set by passing a list. Here, for example, we use
    Pauli-Y and Pauli-Z rotations in the rotation layer:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        circuit = n_local(3, ["ry", "rz"], "cz", "full", reps=1, insert_barriers=True)
        circuit.draw("mpl")

    To omit rotation or entanglement layers, the block can be set to an empty list:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        circuit = n_local(4, [], "cry", reps=2)
        circuit.draw("mpl")

    The entanglement can be set explicitly via the ``entanglement`` argument:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        entangler_map = [[0, 1], [2, 0]]
        circuit = n_local(3, "x", "crx", entangler_map, reps=2)
        circuit.draw("mpl")

    We can set different entanglements per layer, by specifying a callable that takes
    as input the current layer index, and returns the entanglement structure. For example,
    the following uses different entanglements for odd and even layers:

    .. plot::
        :alt: Circuit diagram output by the previous code.
        :include-source:
        :context: close-figs

        def entanglement(layer_index):
            if layer_index % 2 == 0:
                return [[0, 1], [0, 2]]
            return [[1, 2]]

        circuit = n_local(3, "x", "cx", entanglement, reps=3, insert_barriers=True)
        circuit.draw("mpl")


    Args:
        num_qubits: The number of qubits of the circuit.
        rotation_blocks: The blocks used in the rotation layers. If multiple are passed,
            these will be applied one after another (like new sub-layers).
        entanglement_blocks: The blocks used in the entanglement layers. If multiple are passed,
            these will be applied one after another.
        entanglement: The indices specifying on which qubits the input blocks act. This is
            specified by string describing an entanglement strategy (see the additional info)
            or a list of qubit connections.
            If a list of entanglement blocks is passed, different entanglement for each block can
            be specified by passing a list of entanglements. To specify varying entanglement for
            each repetition, pass a callable that takes as input the layer and returns the
            entanglement for that layer.
            Defaults to ``"full"``, meaning an all-to-all entanglement structure.
        reps: Specifies how often the rotation blocks and entanglement blocks are repeated.
        insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
            no barriers are inserted.
        parameter_prefix: The prefix used if default parameters are generated.
        overwrite_block_parameters: If the parameters in the added blocks should be overwritten.
            If ``False``, the parameters in the blocks are not changed.
        skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.
        skip_unentangled_qubits: If ``True``, the rotation gates act only on qubits that
            are entangled. If ``False``, the rotation gates act on all qubits.
        name: The name of the circuit.

    Returns:
        An n-local circuit.
    """
    if reps < 0:
        # this is an important check, since we cast this to an unsigned integer Rust-side
        raise ValueError(f"reps must be non-negative, but is {reps}")

    supported_gates = get_standard_gate_name_mapping()
    rotation_blocks = _normalize_blocks(
        rotation_blocks, supported_gates, overwrite_block_parameters
    )
    entanglement_blocks = _normalize_blocks(
        entanglement_blocks, supported_gates, overwrite_block_parameters
    )

    entanglement = _normalize_entanglement(entanglement, len(entanglement_blocks))

    data = py_n_local(
        num_qubits=num_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=insert_barriers,
        parameter_prefix=parameter_prefix,
        skip_final_rotation_layer=skip_final_rotation_layer,
        skip_unentangled_qubits=skip_unentangled_qubits,
    )
    circuit = QuantumCircuit._from_circuit_data(data, legacy_qubits=True, name=name)

    return circuit


class NLocal(BlueprintCircuit):
    """The n-local circuit class.

    The structure of the n-local circuit are alternating rotation and entanglement layers.
    In both layers, parameterized circuit-blocks act on the circuit in a defined way.
    In the rotation layer, the blocks are applied stacked on top of each other, while in the
    entanglement layer according to the ``entanglement`` strategy.
    The circuit blocks can have arbitrary sizes (smaller equal to the number of qubits in the
    circuit). Each layer is repeated ``reps`` times, and by default a final rotation layer is
    appended.

    For instance, a rotation block on 2 qubits and an entanglement block on 4 qubits using
    ``'linear'`` entanglement yields the following circuit.

    .. code-block:: text

        ┌──────┐ ░ ┌──────┐                      ░ ┌──────┐
        ┤0     ├─░─┤0     ├──────────────── ... ─░─┤0     ├
        │  Rot │ ░ │      │┌──────┐              ░ │  Rot │
        ┤1     ├─░─┤1     ├┤0     ├──────── ... ─░─┤1     ├
        ├──────┤ ░ │  Ent ││      │┌──────┐      ░ ├──────┤
        ┤0     ├─░─┤2     ├┤1     ├┤0     ├ ... ─░─┤0     ├
        │  Rot │ ░ │      ││  Ent ││      │      ░ │  Rot │
        ┤1     ├─░─┤3     ├┤2     ├┤1     ├ ... ─░─┤1     ├
        ├──────┤ ░ └──────┘│      ││  Ent │      ░ ├──────┤
        ┤0     ├─░─────────┤3     ├┤2     ├ ... ─░─┤0     ├
        │  Rot │ ░         └──────┘│      │      ░ │  Rot │
        ┤1     ├─░─────────────────┤3     ├ ... ─░─┤1     ├
        └──────┘ ░                 └──────┘      ░ └──────┘

        |                                 |
        +---------------------------------+
               repeated reps times

    If specified, barriers can be inserted in between every block.
    If an initial state object is provided, it is added in front of the NLocal.

    .. seealso::

        The :func:`.n_local` function constructs a functionally equivalent circuit, but faster.

    """

    @deprecate_func(
        since="2.1",
        additional_msg="This applies to NLocal subclasses too. Use the corresponding function "
        "from the module qiskit.circuit.library.n_local instead.",
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        num_qubits: int | None = None,
        rotation_blocks: (
            QuantumCircuit
            | list[QuantumCircuit]
            | qiskit.circuit.Instruction
            | list[qiskit.circuit.Instruction]
            | None
        ) = None,
        entanglement_blocks: (
            QuantumCircuit
            | list[QuantumCircuit]
            | qiskit.circuit.Instruction
            | list[qiskit.circuit.Instruction]
            | None
        ) = None,
        entanglement: list[int] | list[list[int]] | None = None,
        reps: int = 1,
        insert_barriers: bool = False,
        parameter_prefix: str = "θ",
        overwrite_block_parameters: bool | list[list[Parameter]] = True,
        skip_final_rotation_layer: bool = False,
        skip_unentangled_qubits: bool = False,
        initial_state: QuantumCircuit | None = None,
        name: str | None = "nlocal",
        flatten: bool | None = None,
    ) -> None:
        """
        Args:
            num_qubits: The number of qubits of the circuit.
            rotation_blocks: The blocks used in the rotation layers. If multiple are passed,
                these will be applied one after another (like new sub-layers).
            entanglement_blocks: The blocks used in the entanglement layers. If multiple are passed,
                these will be applied one after another. To use different entanglements for
                the sub-layers, see :meth:`get_entangler_map`.
            entanglement: The indices specifying on which qubits the input blocks act. If ``None``, the
                entanglement blocks are applied at the top of the circuit.
            reps: Specifies how often the rotation blocks and entanglement blocks are repeated.
            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
                no barriers are inserted.
            parameter_prefix: The prefix used if default parameters are generated.
            overwrite_block_parameters: If the parameters in the added blocks should be overwritten.
                If ``False``, the parameters in the blocks are not changed.
            skip_final_rotation_layer: Whether a final rotation layer is added to the circuit.
            skip_unentangled_qubits: If ``True``, the rotation gates act only on qubits that
                are entangled. If ``False``, the rotation gates act on all qubits.
            initial_state: A :class:`.QuantumCircuit` object which can be used to describe an initial
                state prepended to the NLocal circuit.
            name: The name of the circuit.
            flatten: Set this to ``True`` to output a flat circuit instead of nesting it inside multiple
                layers of gate objects. By default currently the contents of
                the output circuit will be wrapped in nested objects for
                cleaner visualization. However, if you're using this circuit
                for anything besides visualization its **strongly** recommended
                to set this flag to ``True`` to avoid a large performance
                overhead for parameter binding.

        Raises:
            ValueError: If ``reps`` parameter is less than or equal to 0.
            TypeError: If ``reps`` parameter is not an int value.
        """
        super().__init__(name=name)

        self._num_qubits: int | None = None
        self._insert_barriers = insert_barriers
        self._reps = reps
        self._entanglement_blocks: list[QuantumCircuit] = []
        self._rotation_blocks: list[QuantumCircuit] = []
        self._prepended_blocks: list[QuantumCircuit] = []
        self._prepended_entanglement: list[list[list[int]] | str] = []
        self._appended_blocks: list[QuantumCircuit] = []
        self._appended_entanglement: list[list[list[int]] | str] = []
        self._entanglement = None
        self._entangler_maps = None
        self._ordered_parameters: ParameterVector | list[Parameter] = ParameterVector(
            name=parameter_prefix
        )
        self._overwrite_block_parameters = overwrite_block_parameters
        self._skip_final_rotation_layer = skip_final_rotation_layer
        self._skip_unentangled_qubits = skip_unentangled_qubits
        self._initial_state: QuantumCircuit | None = None
        self._initial_state_circuit: QuantumCircuit | None = None
        self._bounds: list[tuple[float | None, float | None]] | None = None
        self._flatten = flatten

        # During the build, if a subclass hasn't overridden our parametrization methods, we can use
        # a newer fast-path method to parametrise the rotation and entanglement blocks if internally
        # those are just simple stdlib gates that have been promoted to circuits.  We don't
        # precalculate the fast-path layers themselves because there's far too much that can be
        # overridden between object construction and build, and far too many subclasses of `NLocal`
        # that override bits and bobs of the internal private methods, so it'd be too hard to keep
        # everything in sync.
        self._allow_fast_path_parametrization = (
            getattr(self._parameter_generator, "__func__", None) is NLocal._parameter_generator
        )

        if int(reps) != reps:
            raise TypeError("The value of reps should be int")

        if reps < 0:
            raise ValueError("The value of reps should be larger than or equal to 0")

        if num_qubits is not None:
            self.num_qubits = num_qubits

        if entanglement_blocks is not None:
            self.entanglement_blocks = entanglement_blocks

        if rotation_blocks is not None:
            self.rotation_blocks = rotation_blocks

        if entanglement is not None:
            self.entanglement = entanglement

        if initial_state is not None:
            self.initial_state = initial_state

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the n-local circuit.

        Args:
            num_qubits: The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def flatten(self) -> bool:
        """Returns whether the circuit is wrapped in nested gates/instructions or flattened."""
        return bool(self._flatten)

    @flatten.setter
    def flatten(self, flatten: bool) -> None:
        self._invalidate()
        self._flatten = flatten

    def _convert_to_block(self, layer: typing.Any) -> QuantumCircuit:
        """Try to convert ``layer`` to a QuantumCircuit.

        Args:
            layer: The object to be converted to an NLocal block / Instruction.

        Returns:
            The layer converted to a circuit.

        Raises:
            TypeError: If the input cannot be converted to a circuit.
        """
        if isinstance(layer, QuantumCircuit):
            return layer

        if isinstance(layer, Instruction):
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer, list(range(layer.num_qubits)))
            return circuit

        try:
            circuit = QuantumCircuit(layer.num_qubits)
            circuit.append(layer.to_instruction(), list(range(layer.num_qubits)))
            return circuit
        except AttributeError:
            pass

        raise TypeError(f"Adding a {type(layer)} to an NLocal is not supported.")

    @property
    def rotation_blocks(self) -> list[QuantumCircuit]:
        """The blocks in the rotation layers.

        Returns:
            The blocks in the rotation layers.
        """
        return self._rotation_blocks

    @rotation_blocks.setter
    def rotation_blocks(
        self, blocks: QuantumCircuit | list[QuantumCircuit] | Instruction | list[Instruction]
    ) -> None:
        """Set the blocks in the rotation layers.

        Args:
            blocks: The new blocks for the rotation layers.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._rotation_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement_blocks(self) -> list[QuantumCircuit]:
        """The blocks in the entanglement layers.

        Returns:
            The blocks in the entanglement layers.
        """
        return self._entanglement_blocks

    @entanglement_blocks.setter
    def entanglement_blocks(
        self, blocks: QuantumCircuit | list[QuantumCircuit] | Instruction | list[Instruction]
    ) -> None:
        """Set the blocks in the entanglement layers.

        Args:
            blocks: The new blocks for the entanglement layers.
        """
        # cannot check for the attribute ``'__len__'`` because a circuit also has this attribute
        if not isinstance(blocks, (list, numpy.ndarray)):
            blocks = [blocks]

        self._invalidate()
        self._entanglement_blocks = [self._convert_to_block(block) for block in blocks]

    @property
    def entanglement(
        self,
    ) -> (
        str
        | list[str]
        | list[list[str]]
        | list[int]
        | list[list[int]]
        | list[list[list[int]]]
        | list[list[list[list[int]]]]
        | Callable[[int], str]
        | Callable[[int], list[list[int]]]
    ):
        """Get the entanglement strategy.

        Returns:
            The entanglement strategy, see :meth:`get_entangler_map` for more detail on how the
            format is interpreted.
        """
        return self._entanglement

    @entanglement.setter
    def entanglement(
        self,
        entanglement: (
            str
            | list[str]
            | list[list[str]]
            | list[int]
            | list[list[int]]
            | list[list[list[int]]]
            | list[list[list[list[int]]]]
            | Callable[[int], str]
            | Callable[[int], list[list[int]]]
            | None
        ),
    ) -> None:
        """Set the entanglement strategy.

        Args:
            entanglement: The entanglement strategy. See :meth:`get_entangler_map` for more detail
                on the supported formats.
        """
        self._invalidate()
        self._entanglement = entanglement

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the n-local circuit.

        Returns:
            The number of layers in the circuit.
        """
        return 2 * self._reps + int(not self._skip_final_rotation_layer)

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the NLocal class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the blocks are not set.
            ValueError: If the number of repetitions is not set.
            ValueError: If the qubit indices are not set.
            ValueError: If the number of qubit indices does not match the number of blocks.
            ValueError: If an index in the repetitions list exceeds the number of blocks.
            ValueError: If the number of repetitions does not match the number of block-wise
                parameters.
            ValueError: If a specified qubit index is larger than the (manually set) number of
                qubits.
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        # check no needed parameters are None
        if self.entanglement_blocks is None and self.rotation_blocks is None:
            valid = False
            if raise_on_failure:
                raise ValueError("The blocks are not set.")

        return valid

    @property
    def ordered_parameters(self) -> list[Parameter]:
        """The parameters used in the underlying circuit.

        This includes float values and duplicates.

        Examples:

            >>> # prepare circuit ...
            >>> print(nlocal)
                 ┌───────┐┌──────────┐┌──────────┐┌──────────┐
            q_0: ┤ Ry(1) ├┤ Ry(θ[1]) ├┤ Ry(θ[1]) ├┤ Ry(θ[3]) ├
                 └───────┘└──────────┘└──────────┘└──────────┘
            >>> nlocal.parameters
            {Parameter(θ[1]), Parameter(θ[3])}
            >>> nlocal.ordered_parameters
            [1, Parameter(θ[1]), Parameter(θ[1]), Parameter(θ[3])]

        Returns:
            The parameters objects used in the circuit.
        """
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.num_parameters_settable)
            return list(self._ordered_parameters)

        return self._ordered_parameters

    @ordered_parameters.setter
    def ordered_parameters(self, parameters: ParameterVector | list[Parameter]) -> None:
        """Set the parameters used in the underlying circuit.

        Args:
            parameters: The parameters to be used in the underlying circuit.

        Raises:
            ValueError: If the length of ordered parameters does not match the number of
                parameters in the circuit and they are not a ``ParameterVector`` (which could
                be resized to fit the number of parameters).
        """
        if (
            not isinstance(parameters, ParameterVector)
            and len(parameters) != self.num_parameters_settable
        ):
            raise ValueError(
                "The length of ordered parameters must be equal to the number of "
                f"settable parameters in the circuit ({self.num_parameters_settable}),"
                f" but is {len(parameters)}"
            )
        self._ordered_parameters = parameters
        self._invalidate()

    @property
    def insert_barriers(self) -> bool:
        """If barriers are inserted in between the layers or not.

        Returns:
            ``True``, if barriers are inserted in between the layers, ``False`` if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers should be inserted in between the layers or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        # if insert_barriers changes, we have to invalidate the circuit definition,
        # if it is the same as before we can leave the NLocal instance as it is
        if insert_barriers is not self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    def get_unentangled_qubits(self) -> set[int]:
        """Get the indices of unentangled qubits in a set.

        Returns:
            The unentangled qubits.
        """
        entangled_qubits = set()
        for i in range(self._reps):
            for j, block in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                entangled_qubits.update([idx for indices in entangler_map for idx in indices])
        unentangled_qubits = set(range(self.num_qubits)) - entangled_qubits

        return unentangled_qubits

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        This does not change when the parameters are bound or exchanged for same parameters,
        and therefore is different from ``num_parameters`` which counts the number of unique
        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.

        Returns:
            The number of parameters originally available in the circuit.

        Note:
            This quantity does not require the circuit to be built yet.
        """
        num = 0

        for i in range(self._reps):
            for j, block in enumerate(self.entanglement_blocks):
                entangler_map = self.get_entangler_map(i, j, block.num_qubits)
                num += len(entangler_map) * len(get_parameters(block))

        if self._skip_unentangled_qubits:
            unentangled_qubits = self.get_unentangled_qubits()

        num_rot = 0
        for block in self.rotation_blocks:
            block_indices = [
                list(range(j * block.num_qubits, (j + 1) * block.num_qubits))
                for j in range(self.num_qubits // block.num_qubits)
            ]
            if self._skip_unentangled_qubits:
                block_indices = [
                    indices
                    for indices in block_indices
                    if set(indices).isdisjoint(unentangled_qubits)
                ]
            num_rot += len(block_indices) * len(get_parameters(block))

        num += num_rot * (self._reps + int(not self._skip_final_rotation_layer))

        return num

    @property
    def reps(self) -> int:
        """The number of times rotation and entanglement block are repeated.

        Returns:
            The number of repetitions.
        """
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        """Set the repetitions.

        If the repetitions are `0`, only one rotation layer with no entanglement
        layers is applied (unless ``self.skip_final_rotation_layer`` is set to ``True``).

        Args:
            repetitions: The new repetitions.

        Raises:
            ValueError: If reps setter has parameter repetitions < 0.
        """
        if repetitions < 0:
            raise ValueError("The repetitions should be larger than or equal to 0")
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    def print_settings(self) -> str:
        """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
        ret = f"NLocal: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    @property
    def preferred_init_points(self) -> list[float] | None:
        """The initial points for the parameters. Can be stored as initial guess in optimization.

        Returns:
            The initial values for the parameters, or None, if none have been set.
        """
        return None

    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
    ) -> Sequence[Sequence[int]]:
        """Get the entangler map for in the repetition ``rep_num`` and the block ``block_num``.

        The entangler map for the current block is derived from the value of ``self.entanglement``.
        Below the different cases are listed, where ``i`` and ``j`` denote the repetition number
        and the block number, respectively, and ``n`` the number of qubits in the block.

        =================================== ========================================================
        entanglement type                   entangler map
        =================================== ========================================================
        ``None``                            ``[[0, ..., n - 1]]``
        ``str`` (e.g ``'full'``)            the specified connectivity on ``n`` qubits
        ``List[int]``                       [``entanglement``]
        ``List[List[int]]``                 ``entanglement``
        ``List[List[List[int]]]``           ``entanglement[i]``
        ``List[List[List[List[int]]]]``     ``entanglement[i][j]``
        ``List[str]``                       the connectivity specified in ``entanglement[i]``
        ``List[List[str]]``                 the connectivity specified in ``entanglement[i][j]``
        ``Callable[int, str]``              same as ``List[str]``
        ``Callable[int, List[List[int]]]``  same as ``List[List[List[int]]]``
        =================================== ========================================================


        Note that all indices are to be taken modulo the length of the array they act on, i.e.
        no out-of-bounds index error will be raised but we re-iterate from the beginning of the
        list.

        Args:
            rep_num: The current repetition we are in.
            block_num: The block number within the entanglement layers.
            num_block_qubits: The number of qubits in the block.

        Returns:
            The entangler map for the current block in the current repetition.

        Raises:
            ValueError: If the value of ``entanglement`` could not be cast to a corresponding
                entangler map.
        """
        i, j, n = rep_num, block_num, num_block_qubits
        entanglement = self._entanglement

        # entanglement is None
        if entanglement is None:
            return [list(range(n))]

        # entanglement is callable
        if callable(entanglement):
            entanglement = entanglement(i)

        # entanglement is str
        if isinstance(entanglement, str):
            return get_entangler_map(n, self.num_qubits, entanglement, offset=i)

        # check if entanglement is list of something
        if not isinstance(entanglement, (tuple, list)):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")
        num_i = len(entanglement)

        # entanglement is List[str]
        if all(isinstance(en, str) for en in entanglement):
            return get_entangler_map(n, self.num_qubits, entanglement[i % num_i], offset=i)

        # entanglement is List[int]
        if all(isinstance(en, (int, numpy.integer)) for en in entanglement):
            return [[int(en) for en in entanglement]]

        # check if entanglement is List[List]
        if not all(isinstance(en, (tuple, list)) for en in entanglement):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")
        num_j = len(entanglement[i % num_i])

        # entanglement is List[List[str]]
        if all(isinstance(e2, str) for en in entanglement for e2 in en):
            return get_entangler_map(
                n, self.num_qubits, entanglement[i % num_i][j % num_j], offset=i
            )

        # entanglement is List[List[int]]
        if all(isinstance(e2, (int, numpy.int32, numpy.int64)) for en in entanglement for e2 in en):
            for ind, en in enumerate(entanglement):
                entanglement[ind] = tuple(map(int, en))
            return entanglement

        # check if entanglement is List[List[List]]
        if not all(isinstance(e2, (tuple, list)) for en in entanglement for e2 in en):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")

        # entanglement is List[List[List[int]]]
        if all(
            isinstance(e3, (int, numpy.int32, numpy.int64))
            for en in entanglement
            for e2 in en
            for e3 in e2
        ):
            for en in entanglement:
                for ind, e2 in enumerate(en):
                    en[ind] = tuple(map(int, e2))
            return entanglement[i % num_i]

        # check if entanglement is List[List[List[List]]]
        if not all(isinstance(e3, (tuple, list)) for en in entanglement for e2 in en for e3 in e2):
            raise ValueError(f"Invalid value of entanglement: {entanglement}")

        # entanglement is List[List[List[List[int]]]]
        if all(
            isinstance(e4, (int, numpy.int32, numpy.int64))
            for en in entanglement
            for e2 in en
            for e3 in e2
            for e4 in e3
        ):
            for en in entanglement:
                for e2 in en:
                    for ind, e3 in enumerate(e2):
                        e2[ind] = tuple(map(int, e3))
            return entanglement[i % num_i][j % num_j]

        raise ValueError(f"Invalid value of entanglement: {entanglement}")

    @property
    def initial_state(self) -> QuantumCircuit:
        """Return the initial state that is added in front of the n-local circuit.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Set the initial state.

        Args:
            initial_state: The new initial state.

        Raises:
            ValueError: If the number of qubits has been set before and the initial state
                does not match the number of qubits.
        """
        self._initial_state = initial_state
        self._invalidate()

    @property
    def parameter_bounds(self) -> list[tuple[float, float]] | None:
        """The parameter bounds for the unbound parameters in the circuit.

        Returns:
            A list of pairs indicating the bounds, as (lower, upper). None indicates an unbounded
            parameter in the corresponding direction. If ``None`` is returned, problem is fully
            unbounded.
        """
        if not self._is_built:
            self._build()
        return self._bounds

    @parameter_bounds.setter
    def parameter_bounds(self, bounds: list[tuple[float, float]]) -> None:
        """Set the parameter bounds.

        Args:
            bounds: The new parameter bounds.
        """
        self._bounds = bounds

    def add_layer(
        self,
        other: QuantumCircuit | qiskit.circuit.Instruction,
        entanglement: list[int] | str | list[list[int]] | None = None,
        front: bool = False,
    ) -> NLocal:
        """Append another layer to the NLocal.

        Args:
            other: The layer to compose, can be another NLocal, an Instruction or Gate,
                or a QuantumCircuit.
            entanglement: The entanglement or qubit indices.
            front: If True, ``other`` is appended to the front, else to the back.

        Returns:
            self, such that chained composes are possible.

        Raises:
            TypeError: If `other` is not compatible, i.e. is no Instruction and does not have a
                `to_instruction` method.
        """
        block = self._convert_to_block(other)

        if entanglement is None:
            entanglement = [list(range(block.num_qubits))]
        elif isinstance(entanglement, list) and not isinstance(entanglement[0], list):
            entanglement = [entanglement]
        if front:
            self._prepended_blocks += [block]
            self._prepended_entanglement += [entanglement]
        else:
            self._appended_blocks += [block]
            self._appended_entanglement += [entanglement]

        if isinstance(entanglement, list):
            num_qubits = 1 + max(max(indices) for indices in entanglement)
            if num_qubits > self.num_qubits:
                self._invalidate()  # rebuild circuit
                self.num_qubits = num_qubits

        # modify the circuit accordingly
        if front is False and self._is_built:
            if self._insert_barriers and len(self.data) > 0:
                self.barrier()

            if isinstance(entanglement, str):
                entangler_map: Sequence[Sequence[int]] = get_entangler_map(
                    block.num_qubits, self.num_qubits, entanglement
                )
            else:
                entangler_map = entanglement

            for i in entangler_map:
                params = self.ordered_parameters[-len(get_parameters(block)) :]
                parameterized_block = self._parameterize_block(block, params=params)
                self.compose(parameterized_block, i, inplace=True, copy=False)
        else:
            # cannot prepend a block currently, just rebuild
            self._invalidate()

        return self

    def assign_parameters(
        self,
        parameters: (
            Mapping[Parameter, ParameterExpression | float] | Sequence[ParameterExpression | float]
        ),
        inplace: bool = False,
        **kwargs,
    ) -> QuantumCircuit | None:
        """Assign parameters to the n-local circuit.

        This method also supports passing a list instead of a dictionary. If a list
        is passed, the list must have the same length as the number of unbound parameters in
        the circuit. The parameters are assigned in the order of the parameters in
        :meth:`ordered_parameters`.

        Returns:
            A copy of the NLocal circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
        if parameters is None or len(parameters) == 0:
            return self

        if not self._is_built:
            self._build()

        return super().assign_parameters(parameters, inplace=inplace, **kwargs)

    def _parameterize_block(
        self, block, param_iter=None, rep_num=None, block_num=None, indices=None, params=None
    ):
        """Convert ``block`` to a circuit of correct width and parameterized using the iterator."""
        if self._overwrite_block_parameters:
            # check if special parameters should be used

            if params is None:
                params = self._parameter_generator(rep_num, block_num, indices)
            if params is None:
                params = [next(param_iter) for _ in range(len(get_parameters(block)))]

            update = dict(zip(block.parameters, params))
            return block.assign_parameters(update)

        return block.copy()

    def _build_rotation_layer(self, circuit, param_iter, i):
        """Build a rotation layer."""
        # if the unentangled qubits are skipped, compute the set of qubits that are not entangled
        if self._skip_unentangled_qubits:
            skipped_qubits = self.get_unentangled_qubits()
        else:
            skipped_qubits = set()

        target_qubits = circuit.qubits

        # iterate over all rotation blocks
        for j, block in enumerate(self.rotation_blocks):
            skipped_blocks = {qubit // block.num_qubits for qubit in skipped_qubits}
            if (
                self._allow_fast_path_parametrization
                and (simple_block := _stdlib_gate_from_simple_block(block)) is not None
            ):
                all_qubits = (
                    tuple(target_qubits[k * block.num_qubits : (k + 1) * block.num_qubits])
                    for k in range(self.num_qubits // block.num_qubits)
                    if k not in skipped_blocks
                )
                for qubits in all_qubits:
                    instr = CircuitInstruction(
                        simple_block.gate(*itertools.islice(param_iter, simple_block.num_params)),
                        qubits,
                    )
                    circuit._append(instr)
            else:
                block_indices = [
                    list(range(k * block.num_qubits, (k + 1) * block.num_qubits))
                    for k in range(self.num_qubits // block.num_qubits)
                    if k not in skipped_blocks
                ]
                # apply the operations in the layer
                for indices in block_indices:
                    parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                    circuit.compose(parameterized_block, indices, inplace=True, copy=False)

    def _build_entanglement_layer(self, circuit, param_iter, i):
        """Build an entanglement layer."""
        # iterate over all entanglement blocks
        target_qubits = circuit.qubits
        for j, block in enumerate(self.entanglement_blocks):
            entangler_map = self.get_entangler_map(i, j, block.num_qubits)
            if (
                self._allow_fast_path_parametrization
                and (simple_block := _stdlib_gate_from_simple_block(block)) is not None
            ):
                for indices in entangler_map:
                    # It's actually nontrivially faster to use a listcomp and pass that to `tuple`
                    # than to pass a generator expression directly.

                    instr = CircuitInstruction(
                        simple_block.gate(*itertools.islice(param_iter, simple_block.num_params)),
                        tuple([target_qubits[i] for i in indices]),
                    )
                    circuit._append(instr)
            else:
                # apply the operations in the layer
                for indices in entangler_map:
                    parameterized_block = self._parameterize_block(block, param_iter, i, j, indices)
                    circuit.compose(parameterized_block, indices, inplace=True, copy=False)

    def _build_additional_layers(self, circuit, which):
        if which == "appended":
            blocks = self._appended_blocks
            entanglements = self._appended_entanglement
        elif which == "prepended":
            blocks = reversed(self._prepended_blocks)
            entanglements = reversed(self._prepended_entanglement)
        else:
            raise ValueError("`which` must be either `appended` or `prepended`.")

        for block, ent in zip(blocks, entanglements):
            if isinstance(ent, str):
                ent = get_entangler_map(block.num_qubits, self.num_qubits, ent)
            for indices in ent:
                circuit.compose(block, indices, inplace=True, copy=False)

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        if self.num_qubits == 0:
            return

        if not self._flatten:
            circuit = QuantumCircuit(*self.qregs, name=self.name)
        else:
            circuit = self

        # use the initial state as starting circuit, if it is set
        if self.initial_state:
            circuit.compose(self.initial_state.copy(), inplace=True, copy=False)

        param_iter = iter(self.ordered_parameters)

        # build the prepended layers
        self._build_additional_layers(circuit, "prepended")

        # main loop to build the entanglement and rotation layers
        for i in range(self.reps):
            # insert barrier if specified and there is a preceding layer
            if self._insert_barriers and (i > 0 or len(self._prepended_blocks) > 0):
                circuit.barrier()

            # build the rotation layer
            self._build_rotation_layer(circuit, param_iter, i)

            # barrier in between rotation and entanglement layer
            if self._insert_barriers and len(self._rotation_blocks) > 0:
                circuit.barrier()

            # build the entanglement layer
            self._build_entanglement_layer(circuit, param_iter, i)

        # add the final rotation layer
        if not self._skip_final_rotation_layer:
            if self.insert_barriers and self.reps > 0:
                circuit.barrier()
            self._build_rotation_layer(circuit, param_iter, self.reps)

        # add the appended layers
        self._build_additional_layers(circuit, "appended")

        # cast global phase to float if it has no free parameters
        if isinstance(circuit.global_phase, ParameterExpression):
            try:
                circuit.global_phase = float(circuit.global_phase)
            except TypeError:
                # expression contains free parameters
                pass

        if not self._flatten:
            try:
                block = circuit.to_gate()
            except QiskitError:
                block = circuit.to_instruction()

            self.append(block, self.qubits, copy=False)

    def _parameter_generator(self, rep: int, block: int, indices: list[int]) -> Parameter | None:
        """If certain blocks should use certain parameters this method can be overridden."""
        return None


def get_parameters(block: QuantumCircuit | Instruction) -> list[Parameter]:
    """Return the list of Parameters objects inside a circuit or instruction.

    This is required since, in a standard gate the parameters are not necessarily Parameter
    objects (e.g. U3Gate(0.1, 0.2, 0.3).params == [0.1, 0.2, 0.3]) and instructions and
    circuits do not have the same interface for parameters.
    """
    if isinstance(block, QuantumCircuit):
        return list(block.parameters)
    else:
        return [p for p in block.params if isinstance(p, ParameterExpression)]


def get_entangler_map(
    num_block_qubits: int, num_circuit_qubits: int, entanglement: str, offset: int = 0
) -> Sequence[tuple[int, ...]]:
    """Get an entangler map for an arbitrary number of qubits.

    Args:
        num_block_qubits: The number of qubits of the entangling block.
        num_circuit_qubits: The number of qubits of the circuit.
        entanglement: The entanglement strategy.
        offset: The block offset, can be used if the entanglements differ per block.
            See mode ``sca`` for instance.

    Returns:
        The entangler map using mode ``entanglement`` to scatter a block of ``num_block_qubits``
        qubits on ``num_circuit_qubits`` qubits.

    Raises:
        ValueError: If the entanglement mode is not supported.
    """
    try:
        return fast_entangler_map(num_circuit_qubits, num_block_qubits, entanglement, offset)
    except Exception as exc:
        # need this as Rust is now raising a QiskitError, where this function was raising ValueError
        raise ValueError("Something went wrong in Rust space, here's the error:") from exc


_StdlibGateResult = collections.namedtuple("_StdlibGateResult", ("gate", "num_params"))
_STANDARD_GATE_MAPPING = get_standard_gate_name_mapping()


def _stdlib_gate_from_simple_block(block: QuantumCircuit) -> _StdlibGateResult | None:
    if block.global_phase != 0.0 or len(block) != 1:
        return None
    instruction = block.data[0]
    # If the single instruction isn't a standard-library gate that spans the full width of the block
    # in the correct order, we're not simple.  If the gate isn't fully parametrized with pure,
    # unique `Parameter` instances (expressions are too complex) that are in order, we're not
    # simple.
    if (
        instruction.clbits
        or tuple(instruction.qubits) != tuple(block.qubits)
        or (
            getattr(_STANDARD_GATE_MAPPING.get(instruction.operation.name), "base_class", None)
            is not instruction.operation.base_class
        )
        or tuple(instruction.operation.params) != tuple(block.parameters)
    ):
        return None
    return _StdlibGateResult(instruction.operation.base_class, len(instruction.operation.params))


def _normalize_entanglement(
    entanglement: (
        BlockEntanglement
        | Iterable[BlockEntanglement]
        | Callable[[int], BlockEntanglement | Iterable[BlockEntanglement]]
    ),
    num_entanglement_blocks: int,
) -> list[str | list[tuple[int]]] | Callable[[int], list[str | list[tuple[int]]]]:
    """If the entanglement is Iterable[Iterable], normalize to list[tuple]."""
    if isinstance(entanglement, str):
        return [entanglement] * num_entanglement_blocks

    if callable(entanglement):
        return lambda offset: _normalize_entanglement(entanglement(offset), num_entanglement_blocks)

    # here, entanglement is an Iterable
    if len(entanglement) == 0:
        # handle edge cases when entanglement is set to an empty list
        return [[]]

    # if the entanglement is Iterable[Iterable[int]], normalize to Iterable[Iterable[Iterable[int]]]
    try:
        # if users e.g. gave Iterable[int] this in invalid and will raise a TypeError
        if isinstance(entanglement[0][0], (int, numpy.integer)):
            entanglement = [entanglement]
    except TypeError as exc:
        raise TypeError(f"Invalid entanglement type: {entanglement}.") from exc

    # ensure the number of block entanglements matches the number of blocks
    if len(entanglement) != num_entanglement_blocks:
        raise QiskitError(
            f"Number of block-entanglements ({len(entanglement)}) must match number of "
            f"entanglement blocks ({num_entanglement_blocks})!"
        )

    # normalize the data: str remains, and Iterable[Iterable[int]] becomes list[tuple[int]]
    normalized = []
    for block in entanglement:
        if isinstance(block, str):
            normalized.append(block)
        else:
            normalized.append([tuple(connections) for connections in block])

    return normalized


def _normalize_blocks(
    blocks: str | Gate | Iterable[str | Gate],
    supported_gates: dict[str, Gate],
    overwrite_block_parameters: bool,
) -> list[Block]:
    # normalize the input into an iterable -- we add an extra check for a circuit as
    # courtesy to the users, since the NLocal class used to accept circuits
    if isinstance(blocks, (str, Gate, QuantumCircuit)):
        blocks = [blocks]

    normalized = []
    for block in blocks:
        # since the NLocal circuit accepted circuits as inputs, we raise a warning here
        # to simplify the transition (even though, strictly speaking, quantum circuits are
        # not a supported input type)
        if isinstance(block, QuantumCircuit):
            raise ValueError(
                "The blocks should be of type Gate or str, but you passed a QuantumCircuit. "
                "You can call .to_gate() on the circuit to turn it into a Gate object."
            )

        is_standard = False
        if isinstance(block, str):
            if block not in supported_gates:
                raise ValueError(f"Unsupported gate: {block}")
            block = supported_gates[block]
            is_standard = True
        elif isinstance(block, Gate) and getattr(block, "_standard_gate", None) is not None:
            if len(block.params) == 0:
                is_standard = True
            # the fast path will always overwrite block parameters
            elif overwrite_block_parameters:
                # if all parameters are plain Parameter objects, this is a plain
                # standard gate we do not need to propagate parameterizations for
                is_standard = all(isinstance(p, Parameter) for p in block.params)

        if is_standard:
            block = Block.from_standard_gate(block._standard_gate)
        else:
            if overwrite_block_parameters:
                num_parameters, builder = _get_gate_builder(block)
            else:
                num_parameters, builder = _trivial_builder(block)

            block = Block.from_callable(block.num_qubits, num_parameters, builder)

        normalized.append(block)

    return normalized


def _trivial_builder(
    gate: Gate,
) -> tuple[int, Callable[list[Parameter], tuple[Gate, list[ParameterValueType]]]]:

    def builder(_):
        copied = gate.copy()
        return copied, copied.params

    return 0, builder


def _get_gate_builder(
    gate: Gate,
) -> tuple[int, Callable[list[Parameter], tuple[Gate, list[ParameterValueType]]]]:
    """Construct a callable that handles parameter-rebinding.

    For a given gate, this return the number of free parameters and a callable that can be
    used to obtain a re-parameterized version of the gate. For example::

        x, y = Parameter("x"), Parameter("y")
        gate = CUGate(x, 2 * y, 0.5, 0.)

        num_parameters, builder = _build_gate(gate)
        print(num_parameters)  # prints 2

        a, b = Parameter("a"), Parameter("b")
        new_gate, new_params = builder([a, b])
        print(new_gate)  # CUGate(a, 2 * b, 0.5, 0)
        print(new_params)  # [a, 2 * b, 0.5, 0]

    """
    free_parameters = set()
    for p in gate.params:
        if isinstance(p, ParameterExpression):
            free_parameters |= set(p.parameters)

    num_parameters = len(free_parameters)

    sorted_parameters = _sort_parameters(free_parameters)

    def builder(new_parameters):
        out = gate.copy()

        # re-bind the ``Gate.params`` attribute
        param_dict = dict(zip(sorted_parameters, new_parameters))
        bound_params = gate.params.copy()
        for i, expr in enumerate(gate.params):
            if isinstance(expr, ParameterExpression):
                for parameter in expr.parameters:
                    expr = expr.assign(parameter, param_dict[parameter])
                bound_params[i] = expr

        out.params = bound_params

        # if the definition exists, rebind it
        if out._definition is not None:
            out._definition.assign_parameters(param_dict, inplace=True)

        return out, bound_params

    return num_parameters, builder


def _sort_parameters(parameters):
    """Sort a list of Parameter objects."""

    def key(parameter):
        if isinstance(parameter, ParameterVectorElement):
            return (parameter.vector.name, parameter.index)
        return (parameter.name,)

    return sorted(parameters, key=key)
