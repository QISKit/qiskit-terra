# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A module for visualizing device coupling maps"""

import math

import numpy as np
import rustworkx as rx
from rustworkx.visualization import graphviz_draw

from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.transpiler.coupling import CouplingMap
from .exceptions import VisualizationError


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_gate_map(
    backend,
    figsize=None,
    plot_directed=False,
    label_qubits=True,
    qubit_size=None,
    line_width=4,
    font_size=None,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    font_color="white",
    ax=None,
    filename=None,
    qubit_coordinates=None,
):
    """Plots the gate map of a device.

    Args:
        backend (Backend): The backend instance that will be used to plot the device
            gate map.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        filename (str): file path to save image to.
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence must match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        QiskitError: If you tried to pass a simulator or the backend is None,
            but one of num_qubits, mpl_data, or cmap is None.
        MissingOptionalLibraryError: If matplotlib not installed.

    Example:

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           from qiskit.providers.fake_provider import GenericBackendV2
           from qiskit.visualization import plot_gate_map

           backend = GenericBackendV2(num_qubits=5)

           plot_gate_map(backend)
    """
    qubit_coordinates_map = {}

    qubit_coordinates_map[5] = [[1, 0], [0, 1], [1, 1], [1, 2], [2, 1]]

    qubit_coordinates_map[7] = [[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 1], [2, 2]]

    qubit_coordinates_map[20] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [3, 0],
        [3, 1],
        [3, 2],
        [3, 3],
        [3, 4],
    ]

    qubit_coordinates_map[15] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 7],
        [1, 6],
        [1, 5],
        [1, 4],
        [1, 3],
        [1, 2],
        [1, 1],
        [1, 0],
    ]

    qubit_coordinates_map[16] = [
        [1, 0],
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [0, 3],
        [1, 3],
        [3, 3],
        [4, 3],
        [1, 4],
        [3, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [1, 6],
    ]

    qubit_coordinates_map[27] = [
        [1, 0],
        [1, 1],
        [2, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [0, 3],
        [1, 3],
        [3, 3],
        [4, 3],
        [1, 4],
        [3, 4],
        [1, 5],
        [2, 5],
        [3, 5],
        [1, 6],
        [3, 6],
        [0, 7],
        [1, 7],
        [3, 7],
        [4, 7],
        [1, 8],
        [3, 8],
        [1, 9],
        [2, 9],
        [3, 9],
        [3, 10],
    ]

    qubit_coordinates_map[28] = [
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 2],
        [1, 6],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 0],
        [3, 4],
        [3, 8],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
    ]

    qubit_coordinates_map[53] = [
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [1, 2],
        [1, 6],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [3, 0],
        [3, 4],
        [3, 8],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [5, 2],
        [5, 6],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [7, 0],
        [7, 4],
        [7, 8],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [9, 2],
        [9, 6],
    ]

    qubit_coordinates_map[65] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [1, 0],
        [1, 4],
        [1, 8],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [3, 2],
        [3, 6],
        [3, 10],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [5, 0],
        [5, 4],
        [5, 8],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [7, 2],
        [7, 6],
        [7, 10],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
    ]

    qubit_coordinates_map[127] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
    ]

    qubit_coordinates_map[433] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [0, 5],
        [0, 6],
        [0, 7],
        [0, 8],
        [0, 9],
        [0, 10],
        [0, 11],
        [0, 12],
        [0, 13],
        [0, 14],
        [0, 15],
        [0, 16],
        [0, 17],
        [0, 18],
        [0, 19],
        [0, 20],
        [0, 21],
        [0, 22],
        [0, 23],
        [0, 24],
        [0, 25],
        [1, 0],
        [1, 4],
        [1, 8],
        [1, 12],
        [1, 16],
        [1, 20],
        [1, 24],
        [2, 0],
        [2, 1],
        [2, 2],
        [2, 3],
        [2, 4],
        [2, 5],
        [2, 6],
        [2, 7],
        [2, 8],
        [2, 9],
        [2, 10],
        [2, 11],
        [2, 12],
        [2, 13],
        [2, 14],
        [2, 15],
        [2, 16],
        [2, 17],
        [2, 18],
        [2, 19],
        [2, 20],
        [2, 21],
        [2, 22],
        [2, 23],
        [2, 24],
        [2, 25],
        [2, 26],
        [3, 2],
        [3, 6],
        [3, 10],
        [3, 14],
        [3, 18],
        [3, 22],
        [3, 26],
        [4, 0],
        [4, 1],
        [4, 2],
        [4, 3],
        [4, 4],
        [4, 5],
        [4, 6],
        [4, 7],
        [4, 8],
        [4, 9],
        [4, 10],
        [4, 11],
        [4, 12],
        [4, 13],
        [4, 14],
        [4, 15],
        [4, 16],
        [4, 17],
        [4, 18],
        [4, 19],
        [4, 20],
        [4, 21],
        [4, 22],
        [4, 23],
        [4, 24],
        [4, 25],
        [4, 26],
        [5, 0],
        [5, 4],
        [5, 8],
        [5, 12],
        [5, 16],
        [5, 20],
        [5, 24],
        [6, 0],
        [6, 1],
        [6, 2],
        [6, 3],
        [6, 4],
        [6, 5],
        [6, 6],
        [6, 7],
        [6, 8],
        [6, 9],
        [6, 10],
        [6, 11],
        [6, 12],
        [6, 13],
        [6, 14],
        [6, 15],
        [6, 16],
        [6, 17],
        [6, 18],
        [6, 19],
        [6, 20],
        [6, 21],
        [6, 22],
        [6, 23],
        [6, 24],
        [6, 25],
        [6, 26],
        [7, 2],
        [7, 6],
        [7, 10],
        [7, 14],
        [7, 18],
        [7, 22],
        [7, 26],
        [8, 0],
        [8, 1],
        [8, 2],
        [8, 3],
        [8, 4],
        [8, 5],
        [8, 6],
        [8, 7],
        [8, 8],
        [8, 9],
        [8, 10],
        [8, 11],
        [8, 12],
        [8, 13],
        [8, 14],
        [8, 15],
        [8, 16],
        [8, 17],
        [8, 18],
        [8, 19],
        [8, 20],
        [8, 21],
        [8, 22],
        [8, 23],
        [8, 24],
        [8, 25],
        [8, 26],
        [9, 0],
        [9, 4],
        [9, 8],
        [9, 12],
        [9, 16],
        [9, 20],
        [9, 24],
        [10, 0],
        [10, 1],
        [10, 2],
        [10, 3],
        [10, 4],
        [10, 5],
        [10, 6],
        [10, 7],
        [10, 8],
        [10, 9],
        [10, 10],
        [10, 11],
        [10, 12],
        [10, 13],
        [10, 14],
        [10, 15],
        [10, 16],
        [10, 17],
        [10, 18],
        [10, 19],
        [10, 20],
        [10, 21],
        [10, 22],
        [10, 23],
        [10, 24],
        [10, 25],
        [10, 26],
        [11, 2],
        [11, 6],
        [11, 10],
        [11, 14],
        [11, 18],
        [11, 22],
        [11, 26],
        [12, 0],
        [12, 1],
        [12, 2],
        [12, 3],
        [12, 4],
        [12, 5],
        [12, 6],
        [12, 7],
        [12, 8],
        [12, 9],
        [12, 10],
        [12, 11],
        [12, 12],
        [12, 13],
        [12, 14],
        [12, 15],
        [12, 16],
        [12, 17],
        [12, 18],
        [12, 19],
        [12, 20],
        [12, 21],
        [12, 22],
        [12, 23],
        [12, 24],
        [12, 25],
        [12, 26],
        [13, 0],
        [13, 4],
        [13, 8],
        [13, 12],
        [13, 16],
        [13, 20],
        [13, 24],
        [14, 0],
        [14, 1],
        [14, 2],
        [14, 3],
        [14, 4],
        [14, 5],
        [14, 6],
        [14, 7],
        [14, 8],
        [14, 9],
        [14, 10],
        [14, 11],
        [14, 12],
        [14, 13],
        [14, 14],
        [14, 15],
        [14, 16],
        [14, 17],
        [14, 18],
        [14, 19],
        [14, 20],
        [14, 21],
        [14, 22],
        [14, 23],
        [14, 24],
        [14, 25],
        [14, 26],
        [15, 2],
        [15, 6],
        [15, 10],
        [15, 14],
        [15, 18],
        [15, 22],
        [15, 26],
        [16, 0],
        [16, 1],
        [16, 2],
        [16, 3],
        [16, 4],
        [16, 5],
        [16, 6],
        [16, 7],
        [16, 8],
        [16, 9],
        [16, 10],
        [16, 11],
        [16, 12],
        [16, 13],
        [16, 14],
        [16, 15],
        [16, 16],
        [16, 17],
        [16, 18],
        [16, 19],
        [16, 20],
        [16, 21],
        [16, 22],
        [16, 23],
        [16, 24],
        [16, 25],
        [16, 26],
        [17, 0],
        [17, 4],
        [17, 8],
        [17, 12],
        [17, 16],
        [17, 20],
        [17, 24],
        [18, 0],
        [18, 1],
        [18, 2],
        [18, 3],
        [18, 4],
        [18, 5],
        [18, 6],
        [18, 7],
        [18, 8],
        [18, 9],
        [18, 10],
        [18, 11],
        [18, 12],
        [18, 13],
        [18, 14],
        [18, 15],
        [18, 16],
        [18, 17],
        [18, 18],
        [18, 19],
        [18, 20],
        [18, 21],
        [18, 22],
        [18, 23],
        [18, 24],
        [18, 25],
        [18, 26],
        [19, 2],
        [19, 6],
        [19, 10],
        [19, 14],
        [19, 18],
        [19, 22],
        [19, 26],
        [20, 0],
        [20, 1],
        [20, 2],
        [20, 3],
        [20, 4],
        [20, 5],
        [20, 6],
        [20, 7],
        [20, 8],
        [20, 9],
        [20, 10],
        [20, 11],
        [20, 12],
        [20, 13],
        [20, 14],
        [20, 15],
        [20, 16],
        [20, 17],
        [20, 18],
        [20, 19],
        [20, 20],
        [20, 21],
        [20, 22],
        [20, 23],
        [20, 24],
        [20, 25],
        [20, 26],
        [21, 0],
        [21, 4],
        [21, 8],
        [21, 12],
        [21, 16],
        [21, 20],
        [21, 24],
        [22, 0],
        [22, 1],
        [22, 2],
        [22, 3],
        [22, 4],
        [22, 5],
        [22, 6],
        [22, 7],
        [22, 8],
        [22, 9],
        [22, 10],
        [22, 11],
        [22, 12],
        [22, 13],
        [22, 14],
        [22, 15],
        [22, 16],
        [22, 17],
        [22, 18],
        [22, 19],
        [22, 20],
        [22, 21],
        [22, 22],
        [22, 23],
        [22, 24],
        [22, 25],
        [22, 26],
        [23, 2],
        [23, 6],
        [23, 10],
        [23, 14],
        [23, 18],
        [23, 22],
        [23, 26],
        [24, 1],
        [24, 2],
        [24, 3],
        [24, 4],
        [24, 5],
        [24, 6],
        [24, 7],
        [24, 8],
        [24, 9],
        [24, 10],
        [24, 11],
        [24, 12],
        [24, 13],
        [24, 14],
        [24, 15],
        [24, 16],
        [24, 17],
        [24, 18],
        [24, 19],
        [24, 20],
        [24, 21],
        [24, 22],
        [24, 23],
        [24, 24],
        [24, 25],
        [24, 26],
    ]

    num_qubits = backend.num_qubits
    coupling_map = backend.coupling_map
    name = backend.name
    if qubit_coordinates is None and ("ibm" in name or "fake" in name):
        qubit_coordinates = qubit_coordinates_map.get(num_qubits, None)

    if qubit_coordinates:
        if len(qubit_coordinates) != num_qubits:
            raise QiskitError(
                f"The number of specified qubit coordinates {len(qubit_coordinates)} "
                f"does not match the device number of qubits: {num_qubits}"
            )
    return plot_coupling_map(
        num_qubits,
        qubit_coordinates,
        coupling_map.get_edges(),
        figsize,
        plot_directed,
        label_qubits,
        qubit_size,
        line_width,
        font_size,
        qubit_color,
        qubit_labels,
        line_color,
        font_color,
        ax,
        filename,
        planar=rx.is_planar(coupling_map.graph.to_undirected(multigraph=False)),
    )


@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_GRAPHVIZ.require_in_call
def plot_coupling_map(
    num_qubits: int,
    qubit_coordinates: list[list[int]],
    coupling_map: list[list[int]],
    figsize=None,
    plot_directed=False,
    label_qubits=True,
    qubit_size=None,
    line_width=4,
    font_size=None,
    qubit_color=None,
    qubit_labels=None,
    line_color=None,
    font_color="white",
    ax=None,
    filename=None,
    *,
    planar=True,
):
    """Plots an arbitrary coupling map of qubits (embedded in a plane).

    .. warning::
        This function will call the system Graphviz tool on a file involving user-controllable
        strings (such as qubit labels).  It is recommended to only call this function on trusted
        input.

    Args:
        num_qubits (int): The number of qubits defined and plotted.
        qubit_coordinates (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the planar coordinates in a 0-based square grid where each qubit is located.
        coupling_map (List[List[int]]): A list of two-element lists, with entries of each nested
            list being the qubit numbers of the bonds to be plotted.
        figsize (tuple): Output figure size (wxh) in inches.
        plot_directed (bool): Plot directed coupling map.
        label_qubits (bool): Label the qubits.
        qubit_size (float): Size of qubit marker.
        line_width (float): Width of lines.
        font_size (int): Font size of qubit labels.
        qubit_color (list): A list of colors for the qubits
        qubit_labels (list): A list of qubit labels
        line_color (list): A list of colors for each line from coupling_map.
        font_color (str): The font color for the qubit labels.
        ax (Axes): A Matplotlib axes instance.
        filename (str): file path to save image to.
        planar (bool): If the coupling map is planar or not. Default: ``True`` (i.e. it is planar)

    Returns:
        Figure: A Matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: If matplotlib or graphviz is not installed.
        QiskitError: If the length of qubit labels does not match the number of qubits.

    Example:

        .. plot::
           :alt: Output from the previous code.
           :include-source:

            from qiskit.visualization import plot_coupling_map

            num_qubits = 8
            qubit_coordinates = [[0, 1], [1, 1], [1, 0], [1, 2], [2, 0], [2, 2], [2, 1], [3, 1]]
            coupling_map = [[0, 1], [1, 2], [2, 3], [3, 5], [4, 5], [5, 6], [2, 4], [6, 7]]
            plot_coupling_map(num_qubits, qubit_coordinates, coupling_map)
    """
    import matplotlib.pyplot as plt
    from .utils import matplotlib_close_if_inline

    input_axes = False
    if ax:
        input_axes = True

    if qubit_size is None:
        qubit_size = 30

    if qubit_labels is None:
        qubit_labels = list(range(num_qubits))
    elif len(qubit_labels) != num_qubits:
        raise QiskitError("Length of qubit labels does not equal number of qubits.")

    if not label_qubits:
        qubit_labels = [""] * num_qubits

    # set coloring
    if qubit_color is None:
        qubit_color = ["#648fff"] * num_qubits
    if line_color is None:
        line_color = ["#648fff"] * len(coupling_map)

    if num_qubits == 1:
        graph = rx.PyDiGraph()
        graph.add_node(0)
    else:
        graph = CouplingMap(coupling_map).graph

    if not plot_directed:
        line_color_map = dict(zip(graph.edge_list(), line_color))
        graph = graph.to_undirected(multigraph=False)
        line_color = [line_color_map[edge] for edge in graph.edge_list()]

    for node in graph.node_indices():
        graph[node] = node

    for edge_index in graph.edge_indices():
        graph.update_edge_by_index(edge_index, edge_index)

    # pixel-to-inch conversion
    px = 1.15 / plt.rcParams["figure.dpi"]

    if qubit_coordinates:
        qubit_coordinates = [coordinates[::-1] for coordinates in qubit_coordinates]

    if font_size is None:
        max_characters = max(1, max(len(str(x)) for x in qubit_labels))
        if max_characters == 1:
            font_size = 20
        elif max_characters == 2:
            font_size = 14
        elif max_characters == 3:
            font_size = 12
        else:
            font_size = 1

    def color_node(node):
        if qubit_coordinates:
            out_dict = {
                "label": str(qubit_labels[node]),
                "color": f'"{qubit_color[node]}"',
                "fillcolor": f'"{qubit_color[node]}"',
                "pos": f'"{qubit_coordinates[node][0]},{qubit_coordinates[node][1]}"',
                "pin": "True",
            }
        else:
            out_dict = {
                "label": str(qubit_labels[node]),
                "color": f'"{qubit_color[node]}"',
                "fillcolor": f'"{qubit_color[node]}"',
            }
        out_dict["style"] = "filled"
        out_dict["shape"] = "circle"
        out_dict["fontcolor"] = f'"{font_color}"'
        out_dict["fontsize"] = f'"{font_size!s}!"'
        out_dict["height"] = str(qubit_size * px)
        out_dict["fixedsize"] = "True"
        out_dict["fontname"] = '"DejaVu Sans"'
        return out_dict

    def color_edge(edge):
        out_dict = {
            "color": f'"{line_color[edge]}"',
            "fillcolor": f'"{line_color[edge]}"',
            "penwidth": str(line_width),
        }
        return out_dict

    graph_attributes = None
    if not qubit_coordinates:
        if planar:
            graph_attributes = {
                "overlap_scaling": "-7",
                "overlap": "prism",
                "model": "subset",
            }
        else:
            graph_attributes = {
                "overlap": "true",
            }
    plot = graphviz_draw(
        graph,
        method="neato",
        graph_attr=graph_attributes,
        node_attr_fn=color_node,
        edge_attr_fn=color_edge,
        filename=filename,
    )

    if filename:
        return None

    if not input_axes:
        if figsize is None:
            width, height = plot.size
            figsize = (width * px, height * px)
        fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.imshow(plot)

    if not input_axes:
        matplotlib_close_if_inline(fig)
        return fig


def plot_circuit_layout(circuit, backend, view="virtual", qubit_coordinates=None):
    """Plot the layout of a circuit transpiled for a given
    target backend.

    Args:
        circuit (QuantumCircuit): Input quantum circuit.
        backend (Backend): Target backend.
        view (str): How to label qubits in the layout. Options:

          - ``"virtual"``: Label each qubit with the index of the virtual qubit that
            mapped to it.
          - ``"physical"``: Label each qubit with the index of the physical qubit that it
            corresponds to on the device.

        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence must match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A matplotlib figure showing layout.

    Raises:
        QiskitError: Invalid view type given.
        VisualizationError: Circuit has no layout attribute.

    Example:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            from qiskit import QuantumCircuit, transpile
            from qiskit.providers.fake_provider import GenericBackendV2
            from qiskit.visualization import plot_circuit_layout

            ghz = QuantumCircuit(3, 3)
            ghz.h(0)
            for idx in range(1,3):
                ghz.cx(0,idx)
            ghz.measure(range(3), range(3))

            backend = GenericBackendV2(num_qubits=5)
            new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
            plot_circuit_layout(new_circ_lv3, backend)
    """
    if circuit._layout is None:
        raise QiskitError("Circuit has no layout. Perhaps it has not been transpiled.")

    num_qubits = backend.num_qubits
    cmap = backend.coupling_map
    cmap_len = cmap.graph.num_edges()

    qubits = []
    qubit_labels = [""] * num_qubits

    bit_locations = {
        bit: {"register": register, "index": index}
        for register in circuit._layout.initial_layout.get_registers()
        for index, bit in enumerate(register)
    }
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {"register": None, "index": index}

    if view == "virtual":
        for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
            bit_register = bit_locations[key]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(val)
                qubit_labels[val] = str(bit_locations[key]["index"])

    elif view == "physical":
        for key, val in circuit._layout.initial_layout.get_physical_bits().items():
            bit_register = bit_locations[val]["register"]
            if bit_register is None or bit_register.name != "ancilla":
                qubits.append(key)
                qubit_labels[key] = str(key)

    else:
        raise VisualizationError("Layout view must be 'virtual' or 'physical'.")

    qcolors = ["#648fff"] * num_qubits
    for k in qubits:
        qcolors[k] = "black"

    lcolors = ["#648fff"] * cmap_len

    for idx, edge in enumerate(cmap):
        if edge[0] in qubits and edge[1] in qubits:
            lcolors[idx] = "black"

    fig = plot_gate_map(
        backend,
        qubit_color=qcolors,
        qubit_labels=qubit_labels,
        line_color=lcolors,
        qubit_coordinates=qubit_coordinates,
    )
    return fig


@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_error_map(backend, figsize=(15, 12), show_title=True, qubit_coordinates=None):
    """Plots the error map of a given backend.

    Args:
        backend (Backend): Given backend.
        figsize (tuple): Figure size in inches.
        show_title (bool): Show the title or not.
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence must match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A matplotlib figure showing error map.

    Raises:
        VisualizationError: The backend does not provide gate errors for the 'sx' gate.
        MissingOptionalLibraryError: If matplotlib or seaborn is not installed.

    Example:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            from qiskit.visualization import plot_error_map
            from qiskit.providers.fake_provider import GenericBackendV2

            backend = GenericBackendV2(num_qubits=5)
            plot_error_map(backend)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import gridspec, ticker
    import seaborn as sns
    from .utils import matplotlib_close_if_inline

    color_map = sns.cubehelix_palette(reverse=True, as_cmap=True)

    backend_name = backend.name
    num_qubits = backend.num_qubits
    cmap = backend.coupling_map
    two_q_error_map = {}
    single_gate_errors = [0] * num_qubits
    read_err = [0] * num_qubits
    cx_errors = []
    for gate, prop_dict in backend.target.items():
        if prop_dict is None or None in prop_dict:
            continue
        for qargs, inst_props in prop_dict.items():
            if inst_props is None:
                continue
            if gate == "measure":
                if inst_props.error is not None:
                    read_err[qargs[0]] = inst_props.error
            elif len(qargs) == 1:
                if inst_props.error is not None:
                    single_gate_errors[qargs[0]] = max(
                        single_gate_errors[qargs[0]], inst_props.error
                    )
            elif len(qargs) == 2:
                if inst_props.error is not None:
                    two_q_error_map[qargs] = max(two_q_error_map.get(qargs, 0), inst_props.error)
    if cmap:
        directed = False
        if num_qubits < 20:
            for edge in cmap:
                if [edge[1], edge[0]] not in cmap:
                    directed = True
                    break
        for line in cmap.get_edges():
            err = two_q_error_map.get(tuple(line), 0)
            cx_errors.append(err)

    # Convert to percent
    single_gate_errors = 100 * np.asarray(single_gate_errors)
    avg_1q_err = np.mean(single_gate_errors)

    single_norm = matplotlib.colors.Normalize(
        vmin=min(single_gate_errors), vmax=max(single_gate_errors)
    )
    q_colors = [matplotlib.colors.to_hex(color_map(single_norm(err))) for err in single_gate_errors]

    directed = False
    line_colors = []
    if cmap:

        # Convert to percent
        cx_errors = 100 * np.asarray(cx_errors)
        avg_cx_err = np.mean(cx_errors)

        cx_norm = matplotlib.colors.Normalize(vmin=min(cx_errors), vmax=max(cx_errors))
        line_colors = [matplotlib.colors.to_hex(color_map(cx_norm(err))) for err in cx_errors]

    read_err = 100 * np.asarray(read_err)
    avg_read_err = np.mean(read_err)
    max_read_err = np.max(read_err)

    fig = plt.figure(figsize=figsize)
    gridspec.GridSpec(nrows=2, ncols=3)

    grid_spec = gridspec.GridSpec(
        12, 12, height_ratios=[1] * 11 + [0.5], width_ratios=[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]
    )

    left_ax = plt.subplot(grid_spec[2:10, :1])
    main_ax = plt.subplot(grid_spec[:11, 1:11])
    right_ax = plt.subplot(grid_spec[2:10, 11:])
    bleft_ax = plt.subplot(grid_spec[-1, :5])
    if cmap:
        bright_ax = plt.subplot(grid_spec[-1, 7:])

    qubit_size = 28
    if num_qubits <= 5:
        qubit_size = 20
    plot_gate_map(
        backend,
        qubit_color=q_colors,
        line_color=line_colors,
        qubit_size=qubit_size,
        line_width=5,
        plot_directed=directed,
        ax=main_ax,
        qubit_coordinates=qubit_coordinates,
    )

    main_ax.axis("off")
    main_ax.set_aspect(1)
    if cmap:
        single_cb = matplotlib.colorbar.ColorbarBase(
            bleft_ax, cmap=color_map, norm=single_norm, orientation="horizontal"
        )
        tick_locator = ticker.MaxNLocator(nbins=5)
        single_cb.locator = tick_locator
        single_cb.update_ticks()
        single_cb.update_ticks()
        bleft_ax.set_title(f"H error rate (%) [Avg. = {round(avg_1q_err, 3)}]")

    if cmap is None:
        bleft_ax.axis("off")
        bleft_ax.set_title(f"H error rate (%) = {round(avg_1q_err, 3)}")

    if cmap:
        cx_cb = matplotlib.colorbar.ColorbarBase(
            bright_ax, cmap=color_map, norm=cx_norm, orientation="horizontal"
        )
        tick_locator = ticker.MaxNLocator(nbins=5)
        cx_cb.locator = tick_locator
        cx_cb.update_ticks()
        bright_ax.set_title(f"CNOT error rate (%) [Avg. = {round(avg_cx_err, 3)}]")

    if num_qubits < 10:
        num_left = num_qubits
        num_right = 0
    else:
        num_left = math.ceil(num_qubits / 2)
        num_right = num_qubits - num_left

    left_ax.barh(range(num_left), read_err[:num_left], align="center", color="#DDBBBA")
    left_ax.axvline(avg_read_err, linestyle="--", color="#212121")
    left_ax.set_yticks(range(num_left))
    left_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
    left_ax.set_yticklabels([str(kk) for kk in range(num_left)], fontsize=12)
    left_ax.invert_yaxis()
    left_ax.set_title("Readout Error (%)", fontsize=12)

    for spine in left_ax.spines.values():
        spine.set_visible(False)

    if num_right:
        right_ax.barh(
            range(num_left, num_qubits), read_err[num_left:], align="center", color="#DDBBBA"
        )
        right_ax.axvline(avg_read_err, linestyle="--", color="#212121")
        right_ax.set_yticks(range(num_left, num_qubits))
        right_ax.set_xticks([0, round(avg_read_err, 2), round(max_read_err, 2)])
        right_ax.set_yticklabels([str(kk) for kk in range(num_left, num_qubits)], fontsize=12)
        right_ax.invert_yaxis()
        right_ax.invert_xaxis()
        right_ax.yaxis.set_label_position("right")
        right_ax.yaxis.tick_right()
        right_ax.set_title("Readout Error (%)", fontsize=12)
    else:
        right_ax.axis("off")

    for spine in right_ax.spines.values():
        spine.set_visible(False)

    if show_title:
        fig.suptitle(f"{backend_name} Error Map", fontsize=24, y=0.9)
    matplotlib_close_if_inline(fig)
    return fig
