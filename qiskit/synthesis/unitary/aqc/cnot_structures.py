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
"""
These are the CNOT structure methods: anything that you need for creating CNOT structures.
"""
import logging
import math

import numpy as np

_NETWORK_LAYOUTS = ["sequ", "spin", "cart", "cyclic_spin", "cyclic_line"]
_CONNECTIVITY_TYPES = ["full", "line", "star"]


logger = logging.getLogger(__name__)


def _lower_limit(num_qubits: int) -> int:
    """
    Returns lower limit on the number of CNOT units that guarantees exact representation of
    a unitary operator by quantum gates.

    Args:
        num_qubits: number of qubits.

    Returns:
        lower limit on the number of CNOT units.
    """
    num_cnots = math.ceil((4**num_qubits - 3 * num_qubits - 1) / 4.0)
    return num_cnots


def make_cnot_network(
    num_qubits: int,
    network_layout: str = "spin",
    connectivity_type: str = "full",
    depth: int = 0,
) -> np.ndarray:
    """
    Generates a network consisting of building blocks each containing a CNOT gate and possibly some
    single-qubit ones. This network models a quantum operator in question. Note, each building
    block has 2 input and outputs corresponding to a pair of qubits. What we actually return here
    is a chain of indices of qubit pairs shared by every building block in a row.

    Args:
        num_qubits: number of qubits.
        network_layout: type of network geometry, ``{"sequ", "spin", "cart", "cyclic_spin",
            "cyclic_line"}``.
        connectivity_type: type of inter-qubit connectivity, ``{"full", "line", "star"}``.
        depth: depth of the CNOT-network, i.e. the number of layers, where each layer consists of
            a single CNOT-block; default value will be selected, if ``L <= 0``.

    Returns:
        A matrix of size ``(2, N)`` matrix that defines layers in cnot-network, where ``N``
            is either equal ``L``, or defined by a concrete type of the network.

    Raises:
         ValueError: if unsupported type of CNOT-network layout or number of qubits or combination
            of parameters are passed.
    """
    if num_qubits < 2:
        raise ValueError("Number of qubits must be greater or equal to 2")

    if depth <= 0:
        new_depth = _lower_limit(num_qubits)
        logger.debug(
            "Number of CNOT units chosen as the lower limit: %d, got a non-positive value: %d",
            new_depth,
            depth,
        )
        depth = new_depth

    if network_layout == "sequ":
        links = _get_connectivity(num_qubits=num_qubits, connectivity=connectivity_type)
        return _sequential_network(num_qubits=num_qubits, links=links, depth=depth)

    elif network_layout == "spin":
        return _spin_network(num_qubits=num_qubits, depth=depth)

    elif network_layout == "cart":
        cnots = _cartan_network(num_qubits=num_qubits)
        logger.debug(
            "Optimal lower bound: %d; Cartan CNOTs: %d", _lower_limit(num_qubits), cnots.shape[1]
        )
        return cnots

    elif network_layout == "cyclic_spin":
        if connectivity_type != "full":
            raise ValueError(f"'{network_layout}' layout expects 'full' connectivity")

        return _cyclic_spin_network(num_qubits, depth)

    elif network_layout == "cyclic_line":
        if connectivity_type != "line":
            raise ValueError(f"'{network_layout}' layout expects 'line' connectivity")

        return _cyclic_line_network(num_qubits, depth)
    else:
        raise ValueError(
            f"Unknown type of CNOT-network layout, expects one of {_NETWORK_LAYOUTS}, "
            f"got {network_layout}"
        )


def _get_connectivity(num_qubits: int, connectivity: str) -> dict:
    """
    Generates connectivity structure between qubits.

    Args:
        num_qubits: number of qubits.
        connectivity: type of connectivity structure, ``{"full", "line", "star"}``.

    Returns:
        dictionary of allowed links between qubits.

    Raises:
         ValueError: if unsupported type of CNOT-network layout is passed.
    """
    if num_qubits == 1:
        links = {0: [0]}

    elif connectivity == "full":
        # Full connectivity between qubits.
        links = {i: list(range(num_qubits)) for i in range(num_qubits)}

    elif connectivity == "line":
        # Every qubit is connected to its immediate neighbors only.
        links = {i: [i - 1, i, i + 1] for i in range(1, num_qubits - 1)}

        # first qubit
        links[0] = [0, 1]

        # last qubit
        links[num_qubits - 1] = [num_qubits - 2, num_qubits - 1]

    elif connectivity == "star":
        # Every qubit is connected to the first one only.
        links = {i: [0, i] for i in range(1, num_qubits)}

        # first qubit
        links[0] = list(range(num_qubits))

    else:
        raise ValueError(
            f"Unknown connectivity type, expects one of {_CONNECTIVITY_TYPES}, got {connectivity}"
        )
    return links


def _sequential_network(num_qubits: int, links: dict, depth: int) -> np.ndarray:
    """
    Generates a sequential network.

    Args:
        num_qubits: number of qubits.
        links: dictionary of connectivity links.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of ``(2, N)`` that defines layers in qubit network.
    """
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(num_qubits - 1):
            for j in range(i + 1, num_qubits):
                if j in links[i]:
                    cnots[0, layer] = i
                    cnots[1, layer] = j
                    layer += 1
                    if layer >= depth:
                        return cnots


def _spin_network(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a spin-like network.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    """
    layer = 0
    cnots = np.zeros((2, depth), dtype=int)
    while True:
        for i in range(0, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots

        for i in range(1, num_qubits - 1, 2):
            cnots[0, layer] = i
            cnots[1, layer] = i + 1
            layer += 1
            if layer >= depth:
                return cnots


def _cartan_network(num_qubits: int) -> np.ndarray:
    """
    Cartan decomposition in a recursive way, starting from n = 3.

    Args:
        num_qubits: number of qubits.

    Returns:
        2xN matrix that defines layers in qubit network, where N is the
             depth of Cartan decomposition.

    Raises:
        ValueError: if number of qubits is less than 3.
    """
    n = num_qubits
    if n > 3:
        cnots = np.asarray([[0, 0, 0], [1, 1, 1]])
        mult = np.asarray([[n - 2, n - 3, n - 2, n - 3], [n - 1, n - 1, n - 1, n - 1]])
        for _ in range(n - 2):
            cnots = np.hstack((np.tile(np.hstack((cnots, mult)), 3), cnots))
            mult[0, -1] -= 1
            mult = np.tile(mult, 2)
    elif n == 3:
        cnots = np.asarray(
            [
                [0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1],
            ]
        )
    else:
        raise ValueError(f"The number of qubits must be >= 3, got {n}.")

    return cnots


def _cyclic_spin_network(num_qubits: int, depth: int) -> np.ndarray:
    """
    Same as in the spin-like network, but the first and the last qubits are also connected.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    """

    cnots = np.zeros((2, depth), dtype=int)
    z = 0
    while True:
        for i in range(0, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            if z >= depth:
                return cnots

        for i in range(1, num_qubits, 2):
            if i + 1 <= num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = i + 1
                z += 1
            elif i == num_qubits - 1:
                cnots[0, z] = i
                cnots[1, z] = 0
                z += 1
            if z >= depth:
                return cnots


def _cyclic_line_network(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a line based CNOT structure.

    Args:
        num_qubits: number of qubits.
        depth: depth of the network (number of layers of building blocks).

    Returns:
        A matrix of size ``2 x L`` that defines layers in qubit network.
    """

    cnots = np.zeros((2, depth), dtype=int)
    for i in range(depth):
        cnots[0, i] = (i + 0) % num_qubits
        cnots[1, i] = (i + 1) % num_qubits
    return cnots
