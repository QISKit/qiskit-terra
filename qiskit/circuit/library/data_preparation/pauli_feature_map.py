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

"""The Pauli expansion circuit module."""

from __future__ import annotations

from collections.abc import Sequence, Mapping

from collections.abc import Callable
from functools import reduce
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression
from qiskit.circuit.library.standard_gates import HGate
from qiskit.utils.deprecation import deprecate_func
from qiskit._accelerate.circuit_library import pauli_feature_map as _fast_map

from ..n_local.n_local import NLocal


def _normalize_entanglement(
    entanglement: str | Mapping[int, Sequence[Sequence[int]]],
) -> str | dict[int, list[tuple[int]]]:
    if isinstance(entanglement, str):
        return entanglement

    return {
        num_paulis: [tuple(connections) for connections in ent]
        for num_paulis, ent in entanglement.items()
    }


def pauli_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: (
        str
        | Mapping[int, Sequence[Sequence[int]]]
        | Callable[[int], str | Mapping[int, Sequence[Sequence[int]]]]
    ) = "full",
    alpha: float = 2.0,
    paulis: list[str] | None = None,
    data_map_func: Callable[[Parameter], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str = "PauliFeatureMap",
) -> QuantumCircuit:
    r"""The Pauli expansion circuit.

    The Pauli expansion circuit is a data encoding circuit that transforms input data
    :math:`\vec{x} \in \mathbb{R}^n`, where :math:`n` is the ``feature_dimension``, as

    .. math::

        U_{\Phi(\vec{x})}=\exp\left(i\sum_{S \in \mathcal{I}}
        \phi_S(\vec{x})\prod_{i\in S} P_i\right).

    Here, :math:`S` is a set of qubit indices that describes the connections in the feature map,
    :math:`\mathcal{I}` is a set containing all these index sets, and
    :math:`P_i \in \{I, X, Y, Z\}`. Per default the data-mapping
    :math:`\phi_S` is

    .. math::

        \phi_S(\vec{x}) = \begin{cases}
            x_i \text{ if } S = \{i\} \\
            \prod_{j \in S} (\pi - x_j) \text{ if } |S| > 1
            \end{cases}.

    The possible connections can be set using the ``entanglement`` and ``paulis`` arguments.
    For example, for single-qubit :math:`Z` rotations and two-qubit :math:`YY` interactions
    between all qubit pairs, we can set::


        circuit = pauli_feature_map(..., paulis=["Z", "YY"], entanglement="full")

    which will produce blocks of the form

    .. code-block:: text

        ┌───┐┌─────────────┐┌──────────┐                                            ┌───────────┐
        ┤ H ├┤ P(2.0*x[0]) ├┤ RX(pi/2) ├──■──────────────────────────────────────■──┤ RX(-pi/2) ├
        ├───┤├─────────────┤├──────────┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐├───────────┤
        ┤ H ├┤ P(2.0*x[1]) ├┤ RX(pi/2) ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ RX(-pi/2) ├
        └───┘└─────────────┘└──────────┘└───┘└────────────────────────────────┘└───┘└───────────┘

    The circuit contains ``reps`` repetitions of this transformation.

    Please refer to :func:`.z_feature_map` for the case of single-qubit Pauli-:math:`Z` rotations
    and to :func:`.zz_feature_map` for the single- and two-qubit Pauli-:math:`Z` rotations.

    Args:
        feature_dimension: Number of qubits in the circuit.
        reps: The number of times the evolution layers are repeated.
        entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
            ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``) or can be a
            dictionary where the keys represent the number of qubits and the values are list
            of integer-pairs specifying the indices of qubits that are entangled with one
            another, for example: ``{1: [(0,), (2,)], 2: [(0,1), (2,0)]}`` or can be a
            ``Callable[[int], Union[str | Dict[...]]]`` to return an entanglement specific for
            a repetition.
        alpha: The Pauli rotation factor, multiplicative to the pauli rotations.
        paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``
            will be used.
        data_map_func: A mapping function for the data ``x`` which can be supplied to override the
            default mapping.
        parameter_prefix: The prefix used if default parameters are generated.
        insert_barriers: If ``True``, barriers are inserted in between the evolution instructions
            and Hadamard layers.
        name: The name of the circuit.

    Returns:
        A quantum circuit implementing the Pauli feature map.

    Examples:

        >>> prep = pauli_feature_map(2, reps=1, paulis=["ZZ"])
        >>> print(prep)
             ┌───┐
        q_0: ┤ H ├──■──────────────────────────────────────■──
             ├───┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└───┘└────────────────────────────────┘└───┘

        >>> prep = pauli_feature_map(2, reps=1, paulis=["Z", "XX"])
        >>> print(prep)
             ┌───┐┌─────────────┐┌───┐                                            ┌───┐
        q_0: ┤ H ├┤ P(2.0*x[0]) ├┤ H ├──■──────────────────────────────────────■──┤ H ├
             ├───┤├─────────────┤├───┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐├───┤
        q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ H ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ H ├
             └───┘└─────────────┘└───┘└───┘└────────────────────────────────┘└───┘└───┘

        >>> prep = pauli_feature_map(2, reps=1, paulis=["ZY"])
        >>> print(prep)
             ┌───┐┌──────────┐                                            ┌───────────┐
        q_0: ┤ H ├┤ RX(pi/2) ├──■──────────────────────────────────────■──┤ RX(-pi/2) ├
             ├───┤└──────────┘┌─┴─┐┌────────────────────────────────┐┌─┴─┐└───────────┘
        q_1: ┤ H ├────────────┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├─────────────
             └───┘            └───┘└────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import efficient_su2
        >>> prep = pauli_feature_map(3, reps=3, paulis=["Z", "YY", "ZXZ"])
        >>> wavefunction = efficient_su2(3)
        >>> classifier = prep.compose(wavefunction)
        >>> classifier.num_parameters
        27
        >>> classifier.count_ops()
        OrderedDict([('cx', 39), ('rx', 36), ('u1', 21), ('h', 15), ('ry', 12), ('rz', 12)])

    References:

    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.
    """
    # create parameter vector used in the Pauli feature map
    parameters = ParameterVector(parameter_prefix, feature_dimension)

    # the Rust implementation expects the entanglement to be a str or list[tuple[int]] (or the
    # callable to return these types), therefore we normalize the entanglement here
    if callable(entanglement):
        normalized = lambda offset: _normalize_entanglement(entanglement(offset))
    else:
        normalized = _normalize_entanglement(entanglement)

    # construct from Rust
    circuit = QuantumCircuit._from_circuit_data(
        _fast_map(
            feature_dimension,
            paulis=paulis,
            entanglement=normalized,
            reps=reps,
            parameters=parameters,
            data_map_func=data_map_func,
            alpha=alpha,
            insert_barriers=insert_barriers,
        ),
        name=name,
    )

    return circuit


def z_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: (
        str | Sequence[Sequence[int]] | Callable[[int], str | Sequence[Sequence[int]]]
    ) = "full",
    alpha: float = 2.0,
    data_map_func: Callable[[Parameter], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str = "ZFeatureMap",
) -> QuantumCircuit:
    """The first order Pauli Z-evolution circuit.

    On 3 qubits and with 2 repetitions the circuit is represented by:

    .. code-block:: text

        ┌───┐┌─────────────┐┌───┐┌─────────────┐
        ┤ H ├┤ P(2.0*x[0]) ├┤ H ├┤ P(2.0*x[0]) ├
        ├───┤├─────────────┤├───┤├─────────────┤
        ┤ H ├┤ U(2.0*x[1]) ├┤ H ├┤ P(2.0*x[1]) ├
        ├───┤├─────────────┤├───┤├─────────────┤
        ┤ H ├┤ P(2.0*x[2]) ├┤ H ├┤ P(2.0*x[2]) ├
        └───┘└─────────────┘└───┘└─────────────┘

    This is a sub-class of :class:`~qiskit.circuit.library.PauliFeatureMap` where the Pauli
    strings are fixed as `['Z']`. As a result the first order expansion will be a circuit without
    entangling gates.

    Args:
        feature_dimension: Number of qubits in the circuit.
        reps: The number of times the evolution layers are repeated.
        entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
            ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``), a list of
            integer-tuples, or a callable returning these types for each repetition.
        alpha: The Pauli rotation factor, multiplicative to the pauli rotations.
        data_map_func: A mapping function for the data ``x`` which can be supplied to override the
            default mapping.
        parameter_prefix: The prefix used if default parameters are generated.
        insert_barriers: If ``True``, barriers are inserted in between the evolution instructions
            and Hadamard layers.
        name: The name of the circuit.

    Returns:
        A quantum circuit implementing the Z feature map.

    Examples:

        >>> from qiskit.circuit.library import z_feature_map
        >>> prep = z_feature_map(3, reps=3, insert_barriers=True)
        >>> print(prep)
             ┌───┐ ░ ┌─────────────┐ ░ ┌───┐ ░ ┌─────────────┐ ░ ┌───┐ ░ ┌─────────────┐
        q_0: ┤ H ├─░─┤ P(2.0*x[0]) ├─░─┤ H ├─░─┤ P(2.0*x[0]) ├─░─┤ H ├─░─┤ P(2.0*x[0]) ├
             ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤
        q_1: ┤ H ├─░─┤ P(2.0*x[1]) ├─░─┤ H ├─░─┤ P(2.0*x[1]) ├─░─┤ H ├─░─┤ P(2.0*x[1]) ├
             ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤ ░ ├───┤ ░ ├─────────────┤
        q_2: ┤ H ├─░─┤ P(2.0*x[2]) ├─░─┤ H ├─░─┤ P(2.0*x[2]) ├─░─┤ H ├─░─┤ P(2.0*x[2]) ├
             └───┘ ░ └─────────────┘ ░ └───┘ ░ └─────────────┘ ░ └───┘ ░ └─────────────┘

        >>> data_map = lambda x: x[0]*x[0] + 1  # note: input is an array
        >>> prep = z_feature_map(3, reps=1, data_map_func=data_map)
        >>> print(prep)
             ┌───┐┌──────────────────────┐
        q_0: ┤ H ├┤ P(2.0*x[0]**2 + 2.0) ├
             ├───┤├──────────────────────┤
        q_1: ┤ H ├┤ P(2.0*x[1]**2 + 2.0) ├
             ├───┤├──────────────────────┤
        q_2: ┤ H ├┤ P(2.0*x[2]**2 + 2.0) ├
             └───┘└──────────────────────┘

        >>> from qiskit.circuit.library import n_local
        >>> circuit = n_local(3, "ry", "cz", reps=1).decompose()
        >>> classifier = z_feature_map(3, reps=1)
        >>> classifier.append(circuit, list(range(classifier.num_qubits)))
        >>> print(classifier)
             ┌───┐┌─────────────┐┌──────────┐      ┌──────────┐
        q_0: ┤ H ├┤ P(2.0*x[0]) ├┤ RY(θ[0]) ├─■──■─┤ RY(θ[3]) ├────────────
             ├───┤├─────────────┤├──────────┤ │  │ └──────────┘┌──────────┐
        q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ RY(θ[1]) ├─■──┼──────■──────┤ RY(θ[4]) ├
             ├───┤├─────────────┤├──────────┤    │      │      ├──────────┤
        q_2: ┤ H ├┤ P(2.0*x[2]) ├┤ RY(θ[2]) ├────■──────■──────┤ RY(θ[5]) ├
             └───┘└─────────────┘└──────────┘                  └──────────┘

    """
    return pauli_feature_map(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
        alpha=alpha,
        paulis=["z"],
        data_map_func=data_map_func,
        parameter_prefix=parameter_prefix,
        insert_barriers=insert_barriers,
        name=name,
    )


def zz_feature_map(
    feature_dimension: int,
    reps: int = 2,
    entanglement: (
        str | Sequence[Sequence[int]] | Callable[[int], str | Sequence[Sequence[int]]]
    ) = "full",
    alpha: float = 2.0,
    data_map_func: Callable[[Parameter], ParameterExpression] | None = None,
    parameter_prefix: str = "x",
    insert_barriers: bool = False,
    name: str = "ZZFeatureMap",
) -> QuantumCircuit:
    r"""Second-order Pauli-Z evolution circuit.

    For 3 qubits and 1 repetition and linear entanglement the circuit is represented by:

    .. code-block:: text

        ┌───┐┌────────────────┐
        ┤ H ├┤ P(2.0*φ(x[0])) ├──■───────────────────────────■───────────────────────────────────
        ├───┤├────────────────┤┌─┴─┐┌─────────────────────┐┌─┴─┐
        ┤ H ├┤ P(2.0*φ(x[1])) ├┤ X ├┤ P(2.0*φ(x[0],x[1])) ├┤ X ├──■───────────────────────────■──
        ├───┤├────────────────┤└───┘└─────────────────────┘└───┘┌─┴─┐┌─────────────────────┐┌─┴─┐
        ┤ H ├┤ P(2.0*φ(x[2])) ├─────────────────────────────────┤ X ├┤ P(2.0*φ(x[1],x[2])) ├┤ X ├
        └───┘└────────────────┘                                 └───┘└─────────────────────┘└───┘

    Here, :math:`\varphi` is a classical non-linear function, which defaults to
    :math:`\varphi(x) = x` if :math:`|S| = 1` and
    :math:`\varphi(x,y) = (\pi - x)(\pi - y)` if :math:`|S| > 1`, and
    :math:`S` is the set of qubit indices describing the connections in the feature map.
    See the docstring of :func:`pauli_feature_map` for more detail.

    Args:
        feature_dimension: Number of qubits in the circuit.
        reps: The number of times the evolution layers are repeated.
        entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
            ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``), a list of
            integer-tuples, or a callable returning these types for each repetition.
        alpha: The Pauli rotation factor, multiplicative to the pauli rotations.
        data_map_func: A mapping function for the data ``x`` which can be supplied to override the
            default mapping.
        parameter_prefix: The prefix used if default parameters are generated.
        insert_barriers: If ``True``, barriers are inserted in between the evolution instructions
            and Hadamard layers.
        name: The name of the circuit.

    Returns:
        A quantum circuit implementing the ZZ feature map.

    Examples:

        >>> from qiskit.circuit.library import zz_feature_map
        >>> prep = zz_feature_map(2, reps=1)
        >>> print(prep)
             ┌───┐┌─────────────┐
        q_0: ┤ H ├┤ P(2.0*x[0]) ├──■──────────────────────────────────────■──
             ├───┤├─────────────┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└─────────────┘└───┘└────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import efficient_su2
        >>> classifier = zz_feature_map(3).compose(efficient_su2(3))
        >>> classifier.num_parameters
        27
        >>> classifier.parameters  # 'x' for the data preparation, 'θ' for the SU2 parameters
        ParameterView([
            ParameterVectorElement(x[0]), ParameterVectorElement(x[1]),
            ParameterVectorElement(x[2]), ParameterVectorElement(θ[0]),
            ParameterVectorElement(θ[1]), ParameterVectorElement(θ[2]),
            ParameterVectorElement(θ[3]), ParameterVectorElement(θ[4]),
            ParameterVectorElement(θ[5]), ParameterVectorElement(θ[6]),
            ParameterVectorElement(θ[7]), ParameterVectorElement(θ[8]),
            ParameterVectorElement(θ[9]), ParameterVectorElement(θ[10]),
            ParameterVectorElement(θ[11]), ParameterVectorElement(θ[12]),
            ParameterVectorElement(θ[13]), ParameterVectorElement(θ[14]),
            ParameterVectorElement(θ[15]), ParameterVectorElement(θ[16]),
            ParameterVectorElement(θ[17]), ParameterVectorElement(θ[18]),
            ParameterVectorElement(θ[19]), ParameterVectorElement(θ[20]),
            ParameterVectorElement(θ[21]), ParameterVectorElement(θ[22]),
            ParameterVectorElement(θ[23])
        ])
    """
    return pauli_feature_map(
        feature_dimension=feature_dimension,
        reps=reps,
        entanglement=entanglement,
        alpha=alpha,
        paulis=["z", "zz"],
        data_map_func=data_map_func,
        parameter_prefix=parameter_prefix,
        insert_barriers=insert_barriers,
        name=name,
    )


class PauliFeatureMap(NLocal):
    r"""The Pauli Expansion circuit.

    The Pauli Expansion circuit is a data encoding circuit that transforms input data
    :math:`\vec{x} \in \mathbb{R}^n`, where `n` is the ``feature_dimension``, as

    .. math::

        U_{\Phi(\vec{x})}=\exp\left(i\sum_{S \in \mathcal{I}}
        \phi_S(\vec{x})\prod_{i\in S} P_i\right).

    Here, :math:`S` is a set of qubit indices that describes the connections in the feature map,
    :math:`\mathcal{I}` is a set containing all these index sets, and
    :math:`P_i \in \{I, X, Y, Z\}`. Per default the data-mapping
    :math:`\phi_S` is

    .. math::

        \phi_S(\vec{x}) = \begin{cases}
            x_i \text{ if } S = \{i\} \\
            \prod_{j \in S} (\pi - x_j) \text{ if } |S| > 1
            \end{cases}.

    The possible connections can be set using the ``entanglement`` and ``paulis`` arguments.
    For example, for single-qubit :math:`Z` rotations and two-qubit :math:`YY` interactions
    between all qubit pairs, we can set::


        feature_map = PauliFeatureMap(..., paulis=["Z", "YY"], entanglement="full")

    which will produce blocks of the form

    .. code-block:: text

        ┌───┐┌─────────────┐┌──────────┐                                            ┌───────────┐
        ┤ H ├┤ P(2.0*x[0]) ├┤ RX(pi/2) ├──■──────────────────────────────────────■──┤ RX(-pi/2) ├
        ├───┤├─────────────┤├──────────┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐├───────────┤
        ┤ H ├┤ P(2.0*x[1]) ├┤ RX(pi/2) ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ RX(-pi/2) ├
        └───┘└─────────────┘└──────────┘└───┘└────────────────────────────────┘└───┘└───────────┘

    The circuit contains ``reps`` repetitions of this transformation.

    Please refer to :class:`.ZFeatureMap` for the case of single-qubit Pauli-:math:`Z` rotations
    and to :class:`.ZZFeatureMap` for the single- and two-qubit Pauli-:math:`Z` rotations.

    Examples:

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZZ'])
        >>> print(prep.decompose())
             ┌───┐
        q_0: ┤ H ├──■──────────────────────────────────────■──
             ├───┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐
        q_1: ┤ H ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├
             └───┘└───┘└────────────────────────────────┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['Z', 'XX'])
        >>> print(prep.decompose())
             ┌───┐┌─────────────┐┌───┐                                            ┌───┐
        q_0: ┤ H ├┤ P(2.0*x[0]) ├┤ H ├──■──────────────────────────────────────■──┤ H ├
             ├───┤├─────────────┤├───┤┌─┴─┐┌────────────────────────────────┐┌─┴─┐├───┤
        q_1: ┤ H ├┤ P(2.0*x[1]) ├┤ H ├┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├┤ H ├
             └───┘└─────────────┘└───┘└───┘└────────────────────────────────┘└───┘└───┘

        >>> prep = PauliFeatureMap(2, reps=1, paulis=['ZY'])
        >>> print(prep.decompose())
             ┌───┐┌──────────┐                                            ┌───────────┐
        q_0: ┤ H ├┤ RX(pi/2) ├──■──────────────────────────────────────■──┤ RX(-pi/2) ├
             ├───┤└──────────┘┌─┴─┐┌────────────────────────────────┐┌─┴─┐└───────────┘
        q_1: ┤ H ├────────────┤ X ├┤ P(2.0*(pi - x[0])*(pi - x[1])) ├┤ X ├─────────────
             └───┘            └───┘└────────────────────────────────┘└───┘

        >>> from qiskit.circuit.library import EfficientSU2
        >>> prep = PauliFeatureMap(3, reps=3, paulis=['Z', 'YY', 'ZXZ'])
        >>> wavefunction = EfficientSU2(3)
        >>> classifier = prep.compose(wavefunction)
        >>> classifier.num_parameters
        27
        >>> classifier.count_ops()
        OrderedDict([('cx', 39), ('rx', 36), ('u1', 21), ('h', 15), ('ry', 12), ('rz', 12)])

    References:

    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.
    """

    @deprecate_func(
        since="2.1",
        additional_msg=(
            "Use the pauli_feature_map function as a replacement. Note that this will no longer "
            "return a BlueprintCircuit, but just a plain QuantumCircuit."
        ),
        removal_timeline="in Qiskit 3.0",
    )
    def __init__(
        self,
        feature_dimension: int | None = None,
        reps: int = 2,
        entanglement: (
            str | dict[int, list[tuple[int]]] | Callable[[int], str | dict[int, list[tuple[int]]]]
        ) = "full",
        alpha: float = 2.0,
        paulis: list[str] | None = None,
        data_map_func: Callable[[np.ndarray], float] | None = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
        name: str = "PauliFeatureMap",
    ) -> None:
        """
        Args:
            feature_dimension: Number of qubits in the circuit.
            reps: The number of times the evolution layers are repeated.
            entanglement: Specifies the entanglement structure. Can be a string (``'full'``,
                ``'linear'``, ``'reverse_linear'``, ``'circular'`` or ``'sca'``) or can be a
                dictionary where the keys represent the number of qubits and the values are list
                of integer-pairs specifying the indices of qubits that are entangled with one
                another, for example: ``{1: [(0,), (2,)], 2: [(0,1), (2,0)]}`` or can be a
                ``Callable[[int], Union[str | Dict[...]]]`` to return an entanglement specific for
                a repetition
            alpha: The Pauli rotation factor, multiplicative to the pauli rotations
            paulis: A list of strings for to-be-used paulis. If None are provided, ``['Z', 'ZZ']``
                will be used.
            data_map_func: A mapping function for the data ``x`` which can be supplied to override the
                default mapping.
            parameter_prefix: The prefix used if default parameters are generated.
            insert_barriers: If ``True``, barriers are inserted in between the evolution instructions
                and Hadamard layers.
            name: Name of the circuit.

        """

        super().__init__(
            num_qubits=feature_dimension,
            reps=reps,
            rotation_blocks=HGate(),
            entanglement=entanglement,
            parameter_prefix=parameter_prefix,
            insert_barriers=insert_barriers,
            skip_final_rotation_layer=True,
            name=name,
        )

        self._prefix = parameter_prefix
        self._data_map_func = data_map_func or self_product
        self._paulis = paulis or ["Z", "ZZ"]
        self._alpha = alpha

    def _parameter_generator(
        self, rep: int, block: int, indices: list[int]
    ) -> list[Parameter] | None:
        """If certain blocks should use certain parameters this method can be overridden."""
        params = [self.ordered_parameters[i] for i in indices]
        return params

    @property
    def num_parameters_settable(self):
        """The number of distinct parameters."""
        return self.feature_dimension

    @property
    def paulis(self) -> list[str]:
        """The Pauli strings used in the entanglement of the qubits.

        Returns:
            The Pauli strings as list.
        """
        return self._paulis

    @paulis.setter
    def paulis(self, paulis: list[str]) -> None:
        """Set the pauli strings.

        Args:
            paulis: The new pauli strings.
        """
        self._invalidate()
        self._paulis = paulis

    @property
    def alpha(self) -> float:
        """The Pauli rotation factor (alpha).

        Returns:
            The Pauli rotation factor.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float) -> None:
        """Set the Pauli rotation factor (alpha).

        Args:
            alpha: Pauli rotation factor
        """
        self._invalidate()
        self._alpha = alpha

    @property
    def entanglement_blocks(self):
        """The blocks in the entanglement layers.

        Returns:
            The blocks in the entanglement layers.
        """
        return [self.pauli_block(pauli) for pauli in self._paulis]

    @entanglement_blocks.setter
    def entanglement_blocks(self, entanglement_blocks):
        self._entanglement_blocks = entanglement_blocks

    @property
    def feature_dimension(self) -> int:
        """Returns the feature dimension (which is equal to the number of qubits).

        Returns:
            The feature dimension of this feature map.
        """
        return self.num_qubits

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension.
        """
        self.num_qubits = feature_dimension

    def _extract_data_for_rotation(self, pauli, x):
        where_non_i = np.where(np.asarray(list(pauli[::-1])) != "I")[0]
        x = np.asarray(x)
        return x[where_non_i]

    def pauli_block(self, pauli_string):
        """Get the Pauli block for the feature map circuit."""
        params = ParameterVector("_", length=len(pauli_string))
        time = self._data_map_func(np.asarray(params))
        return self.pauli_evolution(pauli_string, time)

    def pauli_evolution(self, pauli_string, time):
        """Get the evolution block for the given pauli string."""
        # for some reason this is in reversed order
        pauli_string = pauli_string[::-1]

        # trim the pauli string if identities are included
        trimmed = []
        indices = []
        for i, pauli in enumerate(pauli_string):
            if pauli != "I":
                trimmed += [pauli]
                indices += [i]

        evo = QuantumCircuit(len(pauli_string))

        if len(trimmed) == 0:
            return evo

        def basis_change(circuit, inverse=False):
            for i, pauli in enumerate(pauli_string):
                if pauli == "X":
                    circuit.h(i)
                elif pauli == "Y":
                    if inverse:
                        circuit.sxdg(i)
                    else:
                        circuit.sx(i)

        def cx_chain(circuit, inverse=False):
            num_cx = len(indices) - 1
            for i in reversed(range(num_cx)) if inverse else range(num_cx):
                circuit.cx(indices[i], indices[i + 1])

        basis_change(evo)
        cx_chain(evo)
        evo.p(self.alpha * time, indices[-1])
        cx_chain(evo, inverse=True)
        basis_change(evo, inverse=True)
        return evo

    def get_entangler_map(
        self, rep_num: int, block_num: int, num_block_qubits: int
    ) -> Sequence[Sequence[int]]:

        # if entanglement is a Callable[[int], Union[str | Dict[...]]]
        if callable(self._entanglement):
            entanglement = self._entanglement(rep_num)
        else:
            entanglement = self._entanglement

        # entanglement is Dict[int, List[List[int]]]
        if isinstance(entanglement, dict):
            if all(
                isinstance(e2, (int, np.int32, np.int64))
                for key in entanglement.keys()
                for en in entanglement[key]
                for e2 in en
            ):
                for qb, ent in entanglement.items():
                    for en in ent:
                        if len(en) != qb:
                            raise ValueError(
                                f"For num_qubits = {qb}, entanglement must be a "
                                f"tuple of length {qb}. You specified {en}."
                            )

            # Check if the entanglement is specified for all the pauli blocks being used
            for pauli in self.paulis:
                if len(pauli) not in entanglement.keys():
                    raise ValueError(f"No entanglement specified for {pauli} pauli.")

            return entanglement[num_block_qubits]

        else:
            # if the entanglement is not Dict[int, List[List[int]]] or
            # Dict[int, List[Tuple[int]]] then we fall back on the original
            # `get_entangler_map()` method from NLocal
            return super().get_entangler_map(rep_num, block_num, num_block_qubits)


def self_product(x: np.ndarray) -> float:
    """
    Define a function map from R^n to R.

    Args:
        x: data

    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, np.pi - x)
    return coeff
