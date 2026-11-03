# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
Visualization functions for quantum states.
"""

import math

from functools import reduce
import colorsys

import numpy as np
from qiskit import user_config
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import PauliList, SparsePauliOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.utils import optionals as _optionals
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import HGate, SdgGate

from .array import _num_to_latex, array_to_latex
from .utils import matplotlib_close_if_inline
from .exceptions import VisualizationError


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_hinton(state, title="", figsize=None, ax_real=None, ax_imag=None, *, filename=None):
    """Plot a hinton diagram for the density matrix of a quantum state.

    The hinton diagram represents the values of a matrix using
    squares, whose size indicate the magnitude of their corresponding value
    and their color, its sign. A white square means the value is positive and
    a black one means negative.

    Args:
        state (Statevector or DensityMatrix or ndarray): An N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        filename (str | None): The optional file path to save image to. If not specified
            no file is created for the visualization. If this is set the return
            from this function will be ``None``.
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if
            neither ax_real or ax_imag is set.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: Input is not a valid N-qubit state.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            import numpy as np
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import DensityMatrix
            from qiskit.visualization import plot_state_hinton

            qc = QuantumCircuit(2)
            qc.h([0, 1])
            qc.cz(0,1)
            qc.ry(np.pi/3 , 0)
            qc.rx(np.pi/5, 1)

            state = DensityMatrix(qc)
            plot_state_hinton(state, title="New Hinton Plot")

    """
    from matplotlib import pyplot as plt

    # Figure data
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    max_weight = 2 ** math.ceil(math.log2(np.abs(rho.data).max()))
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    if figsize is None:
        figsize = (8, 5)
    if not ax_real and not ax_imag:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        if ax_real:
            fig = ax_real.get_figure()
        else:
            fig = ax_imag.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    # Reversal is to account for Qiskit's endianness.
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)][::-1]
    ly, lx = datareal.shape
    # Real
    if ax1:
        ax1.patch.set_facecolor("gray")
        ax1.set_aspect("equal", "box")
        ax1.xaxis.set_major_locator(plt.NullLocator())
        ax1.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(datareal):
            # Convert from matrix coordinates to plot coordinates.
            plot_x, plot_y = y, lx - x - 1
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax1.add_patch(rect)

        ax1.set_xticks(0.5 + np.arange(lx))
        ax1.set_yticks(0.5 + np.arange(ly))
        ax1.set_xlim([0, lx])
        ax1.set_ylim([0, ly])
        ax1.set_yticklabels(row_names, fontsize=14)
        ax1.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax1.set_title("Re[$\\rho$]", fontsize=14)
    # Imaginary
    if ax2:
        ax2.patch.set_facecolor("gray")
        ax2.set_aspect("equal", "box")
        ax2.xaxis.set_major_locator(plt.NullLocator())
        ax2.yaxis.set_major_locator(plt.NullLocator())

        for (x, y), w in np.ndenumerate(dataimag):
            # Convert from matrix coordinates to plot coordinates.
            plot_x, plot_y = y, lx - x - 1
            color = "white" if w > 0 else "black"
            size = np.sqrt(np.abs(w) / max_weight)
            rect = plt.Rectangle(
                [0.5 + plot_x - size / 2, 0.5 + plot_y - size / 2],
                size,
                size,
                facecolor=color,
                edgecolor=color,
            )
            ax2.add_patch(rect)

        ax2.set_xticks(0.5 + np.arange(lx))
        ax2.set_yticks(0.5 + np.arange(ly))
        ax2.set_xlim([0, lx])
        ax2.set_ylim([0, ly])
        ax2.set_yticklabels(row_names, fontsize=14)
        ax2.set_xticklabels(column_names, fontsize=14, rotation=90)
        ax2.set_title("Im[$\\rho$]", fontsize=14)
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontsize=16)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_vector(
    bloch, title="", ax=None, figsize=None, coord_type="cartesian", font_size=None
):
    """Plot the Bloch sphere.

    Plot a Bloch sphere with the specified coordinates, that can be given in both
    cartesian and spherical systems.

    Args:
        bloch (tuple[float, float, float]): tuple of three elements where (<x>, <y>, <z>) (Cartesian)
            or (<r>, <theta>, <phi>) (spherical in radians)
            <theta> is inclination angle from +z direction
            <phi> is azimuth from +x direction
        title (str): a string that represents the plot title
        ax (matplotlib.axes.Axes): An Axes to use for rendering the bloch
            sphere
        figsize (tuple): Figure size in inches. Has no effect if passing ``ax``.
        coord_type (Literal["cartesian", "spherical"]): Either ``"cartesian"`` or ``"spherical"``
            depending on whether the input is given in Cartesian or spherical coordinates.
        font_size (float): Font size.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` : A matplotlib figure instance if ``ax = None``.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

           from qiskit.visualization import plot_bloch_vector

           plot_bloch_vector([0,1,0], title="New Bloch Sphere")

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           import numpy as np
           from qiskit.visualization import plot_bloch_vector

           # You can use spherical coordinates instead of cartesian.

           plot_bloch_vector([1, np.pi/2, np.pi/3], coord_type='spherical')

    """
    from .bloch import Bloch

    if figsize is None:
        figsize = (5, 5)
    B = Bloch(axes=ax, font_size=font_size)
    if coord_type == "spherical":
        r, theta, phi = bloch
        bloch = (
            r * math.sin(theta) * math.cos(phi),
            r * math.sin(theta) * math.sin(phi),
            r * math.cos(theta),
        )
    B.add_vectors(bloch)
    B.render(title=title)
    if ax is None:
        fig = B.fig
        fig.set_size_inches(figsize[0], figsize[1])
        matplotlib_close_if_inline(fig)
        return fig
    return None


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_bloch_multivector(
    state,
    title="",
    figsize=None,
    *,
    reverse_bits=False,
    filename=None,
    font_size=None,
    title_font_size=None,
    title_pad=1,
):
    r"""Plot a Bloch sphere for each qubit.

    Each component :math:`(x,y,z)` of the Bloch sphere labeled as 'qubit i' represents the expected
    value of the corresponding Pauli operator acting only on that qubit, that is, the expected value
    of :math:`I_{N-1} \otimes\dotsb\otimes I_{i+1}\otimes P_i \otimes I_{i-1}\otimes\dotsb\otimes
    I_0`, where :math:`N` is the number of qubits, :math:`P\in \{X,Y,Z\}` and :math:`I` is the
    identity operator.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): size of each individual Bloch sphere figure, in inches.
        reverse_bits (bool): If True, plots qubits following Qiskit's convention [Default:False].
        font_size (float): Font size for the Bloch ball figures.
        title_font_size (float): Font size for the title.
        title_pad (float): Padding for the title (suptitle ``y`` position is ``0.98``
        and the image height will be extended by ``1 + title_pad/100``).
        filename (str | None): The optional file path to save image to. If not specified
            no file is created for the visualization. If this is set the return
            from this function will be ``None``.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            A matplotlib figure instance.

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            from qiskit.visualization import plot_bloch_multivector

            qc = QuantumCircuit(2)
            qc.h(0)
            qc.x(1)

            state = Statevector(qc)
            plot_bloch_multivector(state)

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_bloch_multivector

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.x(1)

           # You can reverse the order of the qubits.

           from qiskit.quantum_info import DensityMatrix

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.t(1)
           qc.s(0)
           qc.cx(0,1)

           matrix = DensityMatrix(qc)
           plot_bloch_multivector(matrix, title='My Bloch Spheres', reverse_bits=True)

    """
    from matplotlib import pyplot as plt

    # Data
    bloch_data = (
        _bloch_multivector_data(state)[::-1] if reverse_bits else _bloch_multivector_data(state)
    )
    num = len(bloch_data)
    if figsize is not None:
        width, height = figsize
        width *= num
    else:
        width, height = plt.figaspect(1 / num)
    if len(title) > 0:
        height += 1 + title_pad / 100  # additional space for the title
    default_title_font_size = font_size if font_size is not None else 16
    title_font_size = title_font_size if title_font_size is not None else default_title_font_size
    fig = plt.figure(figsize=(width, height))
    for i in range(num):
        pos = num - 1 - i if reverse_bits else i
        ax = fig.add_subplot(1, num, i + 1, projection="3d")
        plot_bloch_vector(
            bloch_data[i], "qubit " + str(pos), ax=ax, figsize=figsize, font_size=font_size
        )
    fig.suptitle(title, fontsize=title_font_size, y=0.98)
    matplotlib_close_if_inline(fig)
    if filename is None:
        try:
            fig.tight_layout()
        except AttributeError:
            pass
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_city(
    state,
    title="",
    figsize=None,
    color=None,
    alpha=1,
    ax_real=None,
    ax_imag=None,
    *,
    filename=None,
):
    """Plot the cityscape of quantum state.

    Plot two 3d bar graphs (two dimensional) of the real and imaginary
    part of the density matrix rho.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list): A list of len=2 giving colors for real and
            imaginary components of matrix elements.
        alpha (float): Transparency value for bars
        ax_real (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_imag only the real component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        ax_imag (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. If this is specified without an
            ax_real only the imaginary component plot will be generated.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        filename (str | None): The optional file path to save image to. If not specified
            no file is created for the visualization. If this is set the return
            from this function will be ``None``.

    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if the
            ``ax_real`` and ``ax_imag`` kwargs are not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        ValueError: When 'color' is not a list of len=2.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

           # You can choose different colors for the real and imaginary parts of the density matrix.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import DensityMatrix
           from qiskit.visualization import plot_state_city

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = DensityMatrix(qc)
           plot_state_city(state, color=['midnightblue', 'crimson'], title="New State City")

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           # You can make the bars more transparent to better see the ones that are behind
           # if they overlap.

           import numpy as np
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_city
           from qiskit import QuantumCircuit

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)


           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0,1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           state = Statevector(qc)
           plot_state_city(state, alpha=0.6)

    """
    import matplotlib.colors as mcolors
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")

    # get the real and imag parts of rho
    datareal = np.real(rho.data)
    dataimag = np.imag(rho.data)

    # get the labels
    column_names = [bin(i)[2:].zfill(num) for i in range(2**num)]
    row_names = [bin(i)[2:].zfill(num) for i in range(2**num)]

    ly, lx = datareal.shape[:2]
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)  # width of bars
    dy = dx.copy()
    dzr = datareal.flatten()
    dzi = dataimag.flatten()

    if color is None:
        real_color, imag_color = "#648fff", "#648fff"
    else:
        if len(color) != 2:
            raise ValueError("'color' must be a list of len=2.")
        real_color = "#648fff" if color[0] is None else color[0]
        imag_color = "#648fff" if color[1] is None else color[1]
    if ax_real is None and ax_imag is None:
        # set default figure size
        if figsize is None:
            figsize = (16, 8)

        fig = plt.figure(figsize=figsize, facecolor="w")
        ax1 = fig.add_subplot(1, 2, 1, projection="3d", computed_zorder=False)
        ax2 = fig.add_subplot(1, 2, 2, projection="3d", computed_zorder=False)

    elif ax_real is not None:
        fig = ax_real.get_figure()
        ax1 = ax_real
        ax2 = ax_imag
    else:
        fig = ax_imag.get_figure()
        ax1 = None
        ax2 = ax_imag

    fig.tight_layout()

    max_dzr = np.max(dzr)
    max_dzi = np.max(dzi)

    # Figure scaling variables since fig.tight_layout won't work
    fig_width, fig_height = fig.get_size_inches()
    max_plot_size = min(fig_width / 2.25, fig_height)
    max_font_size = int(3 * max_plot_size)
    max_zoom = 10 / (10 + np.sqrt(max_plot_size))

    for ax, dz, col, zlabel in (
        (ax1, dzr, real_color, "Real"),
        (ax2, dzi, imag_color, "Imaginary"),
    ):

        if ax is None:
            continue

        max_dz = np.max(dz)
        min_dz = np.min(dz)

        if isinstance(col, str) and col.startswith("#"):
            col = mcolors.to_rgba_array(col)

        dzn = dz < 0
        if np.any(dzn):
            fc = generate_facecolors(
                xpos[dzn], ypos[dzn], zpos[dzn], dx[dzn], dy[dzn], dz[dzn], col
            )
            negative_bars = ax.bar3d(
                xpos[dzn],
                ypos[dzn],
                zpos[dzn],
                dx[dzn],
                dy[dzn],
                dz[dzn],
                alpha=alpha,
                zorder=0.625,
            )
            negative_bars.set_facecolor(fc)

        if min_dz < 0 < max_dz:
            xlim, ylim = [0, lx], [0, ly]
            verts = [list(zip(xlim + xlim[::-1], np.repeat(ylim, 2), [0] * 4))]
            plane = Poly3DCollection(verts, alpha=0.25, facecolor="k", linewidths=1)
            plane.set_zorder(0.75)
            ax.add_collection3d(plane)

        dzp = dz >= 0
        if np.any(dzp):
            fc = generate_facecolors(
                xpos[dzp], ypos[dzp], zpos[dzp], dx[dzp], dy[dzp], dz[dzp], col
            )
            positive_bars = ax.bar3d(
                xpos[dzp],
                ypos[dzp],
                zpos[dzp],
                dx[dzp],
                dy[dzp],
                dz[dzp],
                alpha=alpha,
                zorder=0.875,
            )
            positive_bars.set_facecolor(fc)

        ax.set_title(f"{zlabel} Amplitude (ρ)", fontsize=max_font_size)

        ax.set_xticks(np.arange(0.5, lx + 0.5, 1))
        ax.set_yticks(np.arange(0.5, ly + 0.5, 1))
        if max_dz != min_dz:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-9, max_dzi))
        elif min_dz == 0:
            ax.axes.set_zlim3d(min_dz, max(max_dzr + 1e-9, max_dzi))
        else:
            ax.axes.set_zlim3d(auto=True)
        ax.get_autoscalez_on()

        ax.xaxis.set_ticklabels(
            row_names, fontsize=max_font_size, rotation=45, ha="right", va="top"
        )
        ax.yaxis.set_ticklabels(
            column_names, fontsize=max_font_size, rotation=-22.5, ha="left", va="center"
        )

        for tick in ax.zaxis.get_major_ticks():
            tick.label1.set_fontsize(max_font_size)
            tick.label1.set_horizontalalignment("left")
            tick.label1.set_verticalalignment("bottom")

        ax.set_box_aspect(aspect=(4, 4, 4), zoom=max_zoom)
        ax.set_xmargin(0)
        ax.set_ymargin(0)

    fig.suptitle(title, fontsize=max_font_size * 1.25)
    fig.subplots_adjust(top=0.9, bottom=0, left=0, right=1, hspace=0, wspace=0)
    if ax_real is None and ax_imag is None:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def plot_state_paulivec(state, title="", figsize=None, color=None, ax=None, *, filename=None):
    r"""Plot the Pauli-vector representation of a quantum state as bar graph.

    The Pauli-vector of a density matrix :math:`\rho` is defined by the expectation of each
    possible tensor product of single-qubit Pauli operators (including the identity), that is

    .. math ::

        \rho = \frac{1}{2^n} \sum_{\sigma \in \{I, X, Y, Z\}^{\otimes n}}
               \mathrm{Tr}(\sigma \rho) \sigma.

    This function plots the coefficients :math:`\mathrm{Tr}(\sigma\rho)` as bar graph.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        title (str): a string that represents the plot title
        figsize (tuple): Figure size in inches.
        color (list or str): Color of the coefficient value bars.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        filename (str | None): The optional file path to save image to. If not specified
            no file is created for the visualization. If this is set the return
            from this function will be ``None``.

    Returns:
         :class:`matplotlib:matplotlib.figure.Figure` :
            The matplotlib.Figure of the visualization if the
            ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: if input is not a valid N-qubit state.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

           # You can set a color for all the bars.

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_paulivec

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector(qc)
           plot_state_paulivec(state, color='midnightblue', title="New PauliVec plot")

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           # If you introduce a list with less colors than bars, the color of the bars will
           # alternate following the sequence from the list.

           import numpy as np
           from qiskit.quantum_info import DensityMatrix
           from qiskit import QuantumCircuit
           from qiskit.visualization import plot_state_paulivec

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0, 1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)

           matrix = DensityMatrix(qc)
           plot_state_paulivec(matrix, color=['crimson', 'midnightblue', 'seagreen'])
    """
    from matplotlib import pyplot as plt

    labels, values = _paulivec_data(state)
    numelem = len(values)

    if figsize is None:
        figsize = (7, 5)
    if color is None:
        color = "#648fff"

    ind = np.arange(numelem)  # the x locations for the groups
    width = 0.5  # the width of the bars
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()
    ax.grid(zorder=0, linewidth=1, linestyle="--")
    ax.bar(ind, values, width, color=color, zorder=2)
    ax.axhline(linewidth=1, color="k")
    # add some text for labels, title, and axes ticks
    ax.set_ylabel("Coefficients", fontsize=14)
    ax.set_xticks(ind)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(labels, fontsize=14, rotation=70)
    ax.set_xlabel("Pauli", fontsize=14)
    ax.set_ylim([-1, 1])
    ax.set_facecolor("#eeeeee")
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    ax.set_title(title, fontsize=16)
    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        try:
            fig.tight_layout()
        except AttributeError:
            pass
        return fig
    else:
        return fig.savefig(filename)


def n_choose_k(n, k):
    """Return the number of combinations for n choose k.

    Args:
        n (int): the total number of options .
        k (int): The number of elements.

    Returns:
        int: returns the binomial coefficient
    """
    if n == 0:
        return 0
    return reduce(lambda x, y: x * y[0] / y[1], zip(range(n - k + 1, n + 1), range(1, k + 1)), 1)


def lex_index(n, k, lst):
    """Return  the lex index of a combination..

    Args:
        n (int): the total number of options .
        k (int): The number of elements.
        lst (list): list

    Returns:
        int: returns int index for lex order

    Raises:
        VisualizationError: if length of list is not equal to k
    """
    if len(lst) != k:
        raise VisualizationError("list should have length k")
    comb = [n - 1 - x for x in lst]
    dualm = sum(n_choose_k(comb[k - 1 - i], i + 1) for i in range(k))
    return int(dualm)


def bit_string_index(s):
    """Return the index of a string of 0s and 1s."""
    n = len(s)
    k = s.count("1")
    if s.count("0") != n - k:
        raise VisualizationError("s must be a string of 0 and 1")
    ones = [pos for pos, char in enumerate(s) if char == "1"]
    return lex_index(n, k, ones)


def phase_to_rgb(complex_number):
    """Map a phase of a complex number to a color in (r,g,b).

    complex_number is phase is first mapped to angle in the range
    [0, 2pi] and then to the HSL color wheel
    """
    angles = (np.angle(complex_number) + (np.pi * 5 / 4)) % (np.pi * 2)
    rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
    return rgb


@_optionals.HAS_MATPLOTLIB.require_in_call
@_optionals.HAS_SEABORN.require_in_call
def plot_state_qsphere(
    state,
    figsize=None,
    ax=None,
    show_state_labels=True,
    show_state_phases=False,
    use_degrees=False,
    *,
    filename=None,
):
    """Plot the qsphere representation of a quantum state.
    Here, the size of the points is proportional to the probability
    of the corresponding term in the state and the color represents
    the phase.

    Args:
        state (Statevector or DensityMatrix or ndarray): an N-qubit quantum state.
        figsize (tuple): Figure size in inches.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        show_state_labels (bool): An optional boolean indicating whether to
            show labels for each basis state.
        show_state_phases (bool): An optional boolean indicating whether to
            show the phase for each basis state.
        use_degrees (bool): An optional boolean indicating whether to use
            radians or degrees for the phase values in the plot.
        filename (str | None): The optional file path to save image to. If not specified
            no file is created for the visualization. If this is set the return
            from this function will be ``None``.


    Returns:
        :class:`matplotlib:matplotlib.figure.Figure` :
            A matplotlib figure instance if the ``ax`` kwarg is not set

    Raises:
        MissingOptionalLibraryError: Requires matplotlib.
        VisualizationError: Input is not a valid N-qubit state.

        QiskitError: Input statevector does not have valid dimensions.

    Examples:
        .. plot::
           :alt: Output from the previous code.
           :include-source:

           from qiskit import QuantumCircuit
           from qiskit.quantum_info import Statevector
           from qiskit.visualization import plot_state_qsphere

           qc = QuantumCircuit(2)
           qc.h(0)
           qc.cx(0, 1)

           state = Statevector(qc)
           plot_state_qsphere(state)

        .. plot::
           :alt: Output from the previous code.
           :include-source:

           # You can show the phase of each state and use
           # degrees instead of radians

           from qiskit.quantum_info import DensityMatrix
           import numpy as np
           from qiskit import QuantumCircuit
           from qiskit.visualization import plot_state_qsphere

           qc = QuantumCircuit(2)
           qc.h([0, 1])
           qc.cz(0,1)
           qc.ry(np.pi/3, 0)
           qc.rx(np.pi/5, 1)
           qc.z(1)

           matrix = DensityMatrix(qc)
           plot_state_qsphere(matrix,
                show_state_phases = True, use_degrees = True)
    """
    from matplotlib import gridspec
    from matplotlib import pyplot as plt
    from matplotlib.patches import Circle
    import seaborn as sns
    from scipy import linalg
    from .bloch import Arrow3D

    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    # get the eigenvectors and eigenvalues
    eigvals, eigvecs = linalg.eigh(rho.data)

    if figsize is None:
        figsize = (7, 7)

    if ax is None:
        return_fig = True
        fig = plt.figure(figsize=figsize)
    else:
        return_fig = False
        fig = ax.get_figure()

    gs = gridspec.GridSpec(nrows=3, ncols=3)

    ax = fig.add_subplot(gs[0:3, 0:3], projection="3d")
    ax.axes.set_xlim3d(-1.0, 1.0)
    ax.axes.set_ylim3d(-1.0, 1.0)
    ax.axes.set_zlim3d(-1.0, 1.0)
    ax.axes.grid(False)
    ax.view_init(elev=5, azim=275)

    # Force aspect ratio
    # MPL 3.2 or previous do not have set_box_aspect
    if hasattr(ax.axes, "set_box_aspect"):
        ax.axes.set_box_aspect((1, 1, 1))

    # start the plotting
    # Plot semi-transparent sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(
        x, y, z, rstride=1, cstride=1, color=plt.rcParams["grid.color"], alpha=0.2, linewidth=0
    )

    # Get rid of the panes
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # traversing the eigvals/vecs backward as sorted low->high
    for idx in range(eigvals.shape[0] - 1, -1, -1):
        if eigvals[idx] > 0.001:
            # get the max eigenvalue
            state = eigvecs[:, idx]
            # Rounding to 13 decimals ignores machine epsilon noise (~1e-16)
            # from the solver, ensuring 'argmax' finds the true analytical winner.
            loc = np.round(np.absolute(state), decimals=13).argmax()
            # remove the global phase from max element
            angles = (np.angle(state[loc]) + 2 * np.pi) % (2 * np.pi)
            angleset = np.exp(-1j * angles)
            state = angleset * state

            d = num
            for i in range(2**num):
                # get x,y,z points
                element = bin(i)[2:].zfill(num)
                weight = element.count("1")
                zvalue = -2 * weight / d + 1
                number_of_divisions = n_choose_k(d, weight)
                weight_order = bit_string_index(element)
                angle = (float(weight) / d) * (np.pi * 2) + (
                    weight_order * 2 * (np.pi / number_of_divisions)
                )

                if (weight > d / 2) or (
                    (weight == d / 2) and (weight_order >= number_of_divisions / 2)
                ):
                    angle = np.pi - angle - (2 * np.pi / number_of_divisions)

                xvalue = np.sqrt(1 - zvalue**2) * np.cos(angle)
                yvalue = np.sqrt(1 - zvalue**2) * np.sin(angle)

                # get prob and angle - prob will be shade and angle color
                prob = np.real(np.dot(state[i], state[i].conj()))
                prob = min(prob, 1)  # See https://github.com/Qiskit/qiskit-terra/issues/4666
                colorstate = phase_to_rgb(state[i])

                alfa = 1
                if yvalue >= 0.1:
                    alfa = 1.0 - yvalue

                if not np.isclose(prob, 0) and show_state_labels:
                    rprime = 1.3
                    angle_theta = np.arctan2(np.sqrt(1 - zvalue**2), zvalue)
                    xvalue_text = rprime * np.sin(angle_theta) * np.cos(angle)
                    yvalue_text = rprime * np.sin(angle_theta) * np.sin(angle)
                    zvalue_text = rprime * np.cos(angle_theta)
                    element_text = "$\\vert" + element + "\\rangle$"
                    if show_state_phases:
                        element_angle = (np.angle(state[i]) + (np.pi * 4)) % (np.pi * 2)
                        if use_degrees:
                            element_text += f"\n${element_angle * 180 / np.pi:.1f}^\\circ$"
                        else:
                            element_angle = pi_check(element_angle, ndigits=3).replace("pi", "\\pi")
                            element_text += f"\n${element_angle}$"
                    ax.text(
                        xvalue_text,
                        yvalue_text,
                        zvalue_text,
                        element_text,
                        ha="center",
                        va="center",
                        size=12,
                    )

                ax.plot(
                    [xvalue],
                    [yvalue],
                    [zvalue],
                    markerfacecolor=colorstate,
                    markeredgecolor=colorstate,
                    marker="o",
                    markersize=np.sqrt(prob) * 30,
                    alpha=alfa,
                )

                a = Arrow3D(
                    [0, xvalue],
                    [0, yvalue],
                    [0, zvalue],
                    mutation_scale=20,
                    alpha=prob,
                    arrowstyle="-",
                    color=colorstate,
                    lw=2,
                )
                ax.add_artist(a)

            # add weight lines
            for weight in range(d + 1):
                theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
                z = -2 * weight / d + 1
                r = np.sqrt(1 - z**2)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, z, color=(0.5, 0.5, 0.5), lw=1, ls=":", alpha=0.5)

            # add center point
            ax.plot(
                [0],
                [0],
                [0],
                markerfacecolor=(0.5, 0.5, 0.5),
                markeredgecolor=(0.5, 0.5, 0.5),
                marker="o",
                markersize=3,
                alpha=1,
            )
        else:
            break

    n = 64
    theta = np.ones(n)
    colors = sns.hls_palette(n)

    ax2 = fig.add_subplot(gs[2:, 2:])
    ax2.pie(theta, colors=colors[5 * n // 8 :] + colors[: 5 * n // 8], radius=0.75)
    ax2.add_artist(Circle((0, 0), 0.5, color="white", zorder=1))
    offset = 0.95  # since radius of sphere is one.

    if use_degrees:
        labels = ["Phase\n(Deg)", "0", "90", "180   ", "270"]
    else:
        labels = ["Phase", "$0$", "$\\pi/2$", "$\\pi$", "$3\\pi/2$"]

    ax2.text(0, 0, labels[0], horizontalalignment="center", verticalalignment="center", fontsize=14)
    ax2.text(
        offset, 0, labels[1], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, offset, labels[2], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        -offset, 0, labels[3], horizontalalignment="center", verticalalignment="center", fontsize=14
    )
    ax2.text(
        0, -offset, labels[4], horizontalalignment="center", verticalalignment="center", fontsize=14
    )

    if return_fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)


@_optionals.HAS_MATPLOTLIB.require_in_call
def generate_facecolors(x, y, z, dx, dy, dz, color):
    """Generates shaded facecolors for shaded bars.

    This is here to work around a Matplotlib bug
    where alpha does not work in Bar3D.

    Args:
        x (array_like): The x- coordinates of the anchor point of the bars.
        y (array_like): The y- coordinates of the anchor point of the bars.
        z (array_like): The z- coordinates of the anchor point of the bars.
        dx (array_like): Width of bars.
        dy (array_like): Depth of bars.
        dz (array_like): Height of bars.
        color (array_like): sequence of valid color specifications, optional
    Returns:
        list: Shaded colors for bars.
    Raises:
        MissingOptionalLibraryError: If matplotlib is not installed
    """
    import matplotlib.colors as mcolors

    cuboid = np.array(
        [
            # -z
            (
                (0, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (1, 0, 0),
            ),
            # +z
            (
                (0, 0, 1),
                (1, 0, 1),
                (1, 1, 1),
                (0, 1, 1),
            ),
            # -y
            (
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 1),
                (0, 0, 1),
            ),
            # +y
            (
                (0, 1, 0),
                (0, 1, 1),
                (1, 1, 1),
                (1, 1, 0),
            ),
            # -x
            (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 1),
                (0, 1, 0),
            ),
            # +x
            (
                (1, 0, 0),
                (1, 1, 0),
                (1, 1, 1),
                (1, 0, 1),
            ),
        ]
    )

    # indexed by [bar, face, vertex, coord]
    polys = np.empty(x.shape + cuboid.shape)
    # handle each coordinate separately
    for i, p, dp in [(0, x, dx), (1, y, dy), (2, z, dz)]:
        p = p[..., np.newaxis, np.newaxis]
        dp = dp[..., np.newaxis, np.newaxis]
        polys[..., i] = p + dp * cuboid[..., i]

    # collapse the first two axes
    polys = polys.reshape((-1,) + polys.shape[2:])

    facecolors = []
    if len(color) == len(x):
        # bar colors specified, need to expand to number of faces
        for c in color:
            facecolors.extend([c] * 6)
    else:
        # a single color specified, or face colors specified explicitly
        facecolors = list(mcolors.to_rgba_array(color))
        if len(facecolors) < len(x):
            facecolors *= 6 * len(x)

    normals = _generate_normals(polys)
    return _shade_colors(facecolors, normals)


def _generate_normals(polygons):
    """Takes a list of polygons and return an array of their normals.

    Normals point towards the viewer for a face with its vertices in
    counterclockwise order, following the right hand rule.
    Uses three points equally spaced around the polygon.
    This normal of course might not make sense for polygons with more than
    three points not lying in a plane, but it's a plausible and fast
    approximation.

    Args:
        polygons (list): list of (M_i, 3) array_like, or (..., M, 3) array_like
            A sequence of polygons to compute normals for, which can have
            varying numbers of vertices. If the polygons all have the same
            number of vertices and array is passed, then the operation will
            be vectorized.
    Returns:
        normals: (..., 3) array_like
            A normal vector estimated for the polygon.
    """
    if isinstance(polygons, np.ndarray):
        # optimization: polygons all have the same number of points, so can
        # vectorize
        n = polygons.shape[-2]
        i1, i2, i3 = 0, n // 3, 2 * n // 3
        v1 = polygons[..., i1, :] - polygons[..., i2, :]
        v2 = polygons[..., i2, :] - polygons[..., i3, :]
    else:
        # The subtraction doesn't vectorize because polygons is jagged.
        v1 = np.empty((len(polygons), 3))
        v2 = np.empty((len(polygons), 3))
        for poly_i, ps in enumerate(polygons):
            n = len(ps)
            i1, i2, i3 = 0, n // 3, 2 * n // 3
            v1[poly_i, :] = ps[i1, :] - ps[i2, :]
            v2[poly_i, :] = ps[i2, :] - ps[i3, :]

    return np.cross(v1, v2)


def _shade_colors(color, normals, lightsource=None):
    """
    Shade *color* using normal vectors given by *normals*.
    *color* can also be an array of the same length as *normals*.
    """
    from matplotlib.colors import Normalize, LightSource
    import matplotlib.colors as mcolors

    if lightsource is None:
        # chosen for backwards-compatibility
        lightsource = LightSource(azdeg=225, altdeg=19.4712)

    def mod(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)

    shade = np.array(
        [np.dot(n / mod(n), lightsource.direction) if mod(n) else np.nan for n in normals]
    )
    mask = ~np.isnan(shade)

    if mask.any():
        norm = Normalize(min(shade[mask]), max(shade[mask]))
        shade[~mask] = min(shade[mask])
        color = mcolors.to_rgba_array(color)
        # shape of color should be (M, 4) (where M is number of faces)
        # shape of shade should be (M,)
        # colors should have final shape of (M, 4)
        alpha = color[:, 3]
        colors = (0.5 + norm(shade)[:, np.newaxis] * 0.5) * color
        colors[:, 3] = alpha
    else:
        colors = np.asanyarray(color).copy()

    return colors


def _ket_basis_gates_and_labels(ket_basis: str) -> tuple[list[Gate], tuple[str, str]]:
    """Return the single-qubit basis-change gates and the ket labels for a given basis.

    Applying the gates in order to every qubit of a state yields its coefficients
    :math:`\\langle b | \\psi \\rangle` in the requested basis.

    Args:
        ket_basis: The ket basis. Must be one of ``'z'``, ``'x'`` or ``'y'``.

    Returns:
        A tuple ``(gates, labels)``, where ``gates`` is the sequence of single-qubit gates to apply
        to each qubit (empty if no change of basis is needed) and ``labels`` is the ``(zero, one)``
        pair of ket symbols for that basis.

    Raises:
        VisualizationError: If the ket basis is not supported.
    """
    match ket_basis:
        case "z":
            return [], ("0", "1")
        case "x":
            # <+|psi>, <-|psi> are the rows of H.
            return [HGate()], ("+", "-")
        case "y":
            # <+i|psi>, <-i|psi> are the rows of H @ S^dagger, i.e. apply S^dagger, then H.
            return [SdgGate(), HGate()], ("+i", "-i")
        case _:
            raise VisualizationError(
                f"Ket basis {ket_basis!r} is not supported. Only 'x', 'y' and 'z' are supported."
            )


def state_to_latex(
    state: Statevector | DensityMatrix,
    dims: bool | None = None,
    convention: str = "ket",
    ket_basis: str = "z",
    **args,
) -> str:
    r"""Return a LaTeX representation of a quantum state.

    For qubit statevectors using the default ``convention='ket'``, this generates
    a LaTeX string in ket notation (e.g. :math:`\frac{1}{\sqrt{2}}|0\rangle + \frac{1}{\sqrt{2}}|1\rangle`).
    Otherwise, or when ``convention='vector'``, it formats the state as an array using
    :func:`~qiskit.visualization.array_to_latex`. Subsystem dimensions can optionally
    be displayed alongside the representation.

    Args:
        state: State to be drawn.
        dims: Whether to display the state's subsystem dimensions.
            Defaults to ``True`` for non-qubit systems and ``False`` for qubits.
        convention: Either ``'ket'`` or ``'vector'``. For ``'ket'`` (default for
            qubit statevectors), represents the state in ket notation. For ``'vector'``,
            draws as a column vector.
        ket_basis: The single-qubit basis the kets are written in. This only applies to the
            ket convention for qubit statevectors. One of ``'z'`` (default), the computational
            basis :math:`|0\rangle, |1\rangle`; ``'x'``, the Hadamard basis
            :math:`|+\rangle, |-\rangle`; or ``'y'``, the basis
            :math:`|{+}i\rangle, |{-}i\rangle`.
        **args: Additional keyword arguments passed to the underlying formatting function:

            * When ``convention='ket'`` (qubit statevectors):
              ``max_size`` (int, default: 12) defines the maximum number of non-zero terms
              displayed before truncating with an ellipsis (``\ldots``). Set to a larger
              integer or ``math.inf`` to show all terms. ``prefix`` (str, default: ``""``)
              is prepended to the expression. ``decimals`` (int, default: 10) controls
              rounding precision.
            * When ``convention='vector'``: arguments are forwarded to
              :func:`~qiskit.visualization.array_to_latex` (such as ``max_size``,
              ``precision``, and ``prefix``).

    Returns:
        str: LaTeX representation of the state.

    Raises:
        MissingOptionalLibraryError: If SymPy is not installed when formatting via
            :func:`~qiskit.visualization.array_to_latex`.
        VisualizationError: If ``ket_basis`` is not supported.

    Examples:
        Write :math:`|0\rangle` in the Hadamard basis:

        .. code-block:: python

            from qiskit.quantum_info import Statevector
            from qiskit.visualization.state_visualization import state_to_latex

            state_to_latex(Statevector.from_label("0"), ket_basis="x")
            # \frac{\sqrt{2}}{2} |+\rangle+\frac{\sqrt{2}}{2} |-\rangle
    """
    if dims is None:  # show dims if state is not only qubits
        if set(state.dims()) == {2}:
            dims = False
        else:
            dims = True

    prefix = ""
    suffix = ""
    if dims:
        prefix = "\\begin{align}\n"
        dims_str = state._op_shape.dims_l()
        suffix = f"\\\\\n\\text{{dims={dims_str}}}\n\\end{{align}}"

    operator_shape = state._op_shape
    # we only use the ket convention for qubit statevectors
    # this means the operator shape should have no input dimensions and all output dimensions equal to 2
    is_qubit_statevector = len(operator_shape.dims_r()) == 0 and set(operator_shape.dims_l()) == {2}
    if convention == "ket" and is_qubit_statevector:
        gates, ket_labels = _ket_basis_gates_and_labels(ket_basis)
        if gates:
            # Apply the basis-changing gates to every qubit in one call to evolve.
            basis_change = QuantumCircuit(state.num_qubits)
            for qubit in range(state.num_qubits):
                for gate in gates:
                    basis_change.append(gate, [qubit])
            state = state.evolve(basis_change)
        latex_str = _state_to_latex_ket(state._data, ket_labels=ket_labels, **args)
    else:
        latex_str = array_to_latex(state._data, source=True, **args)
    return prefix + latex_str + suffix


def _numbers_to_latex_terms(numbers: list[complex], decimals: int = 10) -> list[str]:
    """Convert a list of numbers to latex formatted terms

    The first non-zero term is treated differently. For this term a leading + is suppressed.

    Args:
        numbers: List of numbers to format
        decimals: Number of decimal places to round to (default: 10).
    Returns:
        List of formatted terms
    """
    first_term = True
    terms = []
    for number in numbers:
        term = _num_to_latex(number, decimals=decimals, first_term=first_term, coefficient=True)
        terms.append(term)
        first_term = False
    return terms


def _state_to_latex_ket(
    data: list[complex],
    max_size: int = 12,
    prefix: str = "",
    decimals: int = 10,
    ket_labels: tuple[str, str] = ("0", "1"),
) -> str:
    """Convert state vector to latex representation

    Args:
        data: State vector
        max_size: Maximum number of non-zero terms in the expression. If the number of
                 non-zero terms is larger than the max_size, then the representation is truncated.
        prefix: Latex string to be prepended to the latex, intended for labels.
        decimals: Number of decimal places to round to (default: 10).
        ket_labels: The ``(zero, one)`` single-qubit symbols used to render each ket.

    Returns:
        String with LaTeX representation of the state vector
    """
    num = int(math.log2(len(data)))
    zero_label, one_label = ket_labels

    def ket_name(i):
        return "".join(one_label if bit == "1" else zero_label for bit in bin(i)[2:].zfill(num))

    data = np.around(data, decimals)
    nonzero_indices = np.where(data != 0)[0].tolist()
    if len(nonzero_indices) > max_size:
        nonzero_indices = (
            nonzero_indices[: max_size // 2] + [0] + nonzero_indices[-max_size // 2 + 1 :]
        )
        latex_terms = _numbers_to_latex_terms(data[nonzero_indices], decimals)
        nonzero_indices[max_size // 2] = None
    else:
        latex_terms = _numbers_to_latex_terms(data[nonzero_indices], decimals)

    latex_str = ""
    for idx, ket_idx in enumerate(nonzero_indices):
        if ket_idx is None:
            latex_str += r" + \ldots "
        else:
            term = latex_terms[idx]
            ket = ket_name(ket_idx)
            latex_str += f"{term} |{ket}\\rangle"
    return prefix + latex_str


class TextMatrix:
    """Text representation of an array, with `__str__` method so it
    displays nicely in Jupyter notebooks"""

    def __init__(self, state, max_size=8, dims=None, prefix="", suffix=""):
        self.state = state
        self.max_size = max_size
        if dims is None:  # show dims if state is not only qubits
            if (isinstance(state, (Statevector, DensityMatrix)) and set(state.dims()) == {2}) or (
                isinstance(state, Operator)
                and len(state.input_dims()) == len(state.output_dims())
                and set(state.input_dims()) == set(state.output_dims()) == {2}
            ):
                dims = False
            else:
                dims = True
        self.dims = dims
        self.prefix = prefix
        self.suffix = suffix
        if isinstance(max_size, int):
            self.max_size = max_size
        elif isinstance(state, DensityMatrix):
            # density matrices are square, so threshold for
            # summarization is shortest side squared
            self.max_size = min(max_size) ** 2
        else:
            self.max_size = max_size[0]

    def __str__(self):
        threshold = self.max_size
        data = np.array2string(
            self.state._data, prefix=self.prefix, threshold=threshold, separator=","
        )
        dimstr = ""
        if self.dims:
            data += ",\n"
            dimstr += " " * len(self.prefix)
            if isinstance(self.state, (Statevector, DensityMatrix)):
                dimstr += f"dims={self.state._op_shape.dims_l()}"
            else:
                dimstr += f"input_dims={self.state.input_dims()}, "
                dimstr += f"output_dims={self.state.output_dims()}"

        return self.prefix + data + dimstr + self.suffix

    def __repr__(self):
        return self.__str__()


def state_drawer(state, output=None, **drawer_args):
    """Returns a visualization of the state.

    **repr**: ASCII TextMatrix of the state's ``_repr_``.

    **text**: ASCII TextMatrix that can be printed in the console.

    **latex**: An IPython Latex object for displaying in Jupyter Notebooks.

    **latex_source**: Raw, uncompiled ASCII source to generate array using LaTeX.

    **qsphere**: Matplotlib figure, rendering of statevector using `plot_state_qsphere()`.

    **hinton**: Matplotlib figure, rendering of statevector using `plot_state_hinton()`.

    **bloch**: Matplotlib figure, rendering of statevector using `plot_bloch_multivector()`.

    **city**: Matplotlib figure, rendering of statevector using `plot_state_city()`.

    **paulivec**: Matplotlib figure, rendering of statevector using `plot_state_paulivec()`.

    Args:
        state: State to be drawn
        output (str): Select the output method to use for drawing the
            circuit. Valid choices are ``text``, ``latex``, ``latex_source``,
            ``qsphere``, ``hinton``, ``bloch``, ``city`` or ``paulivec``.
            Default is `'text`'.
        drawer_args: Arguments to be passed to the relevant drawer. For
            ``'latex'`` and ``'latex_source'``, see :func:`~qiskit.visualization.state_to_latex`.
            For ``'text'``, arguments such as ``max_size`` (default: 8) can be passed to
            control the display threshold. For Matplotlib figures, see the corresponding
            plotting function under :mod:`qiskit.visualization`.

    Returns:
        :class:`matplotlib.figure` or :class:`str` or
        :class:`TextMatrix` or :class:`IPython.display.Latex`:
        Drawing of the state.

    Raises:
        MissingOptionalLibraryError: when `output` is `latex` and IPython is not installed.
            or if SymPy isn't installed and ``'latex'`` or ``'latex_source'`` is selected for
            ``output``.

        ValueError: when `output` is not a valid selection.
    """
    config = user_config.get_config()
    # Get default 'output' from config file else use 'repr'
    default_output = "repr"
    if output is None:
        if config:
            default_output = config.get("state_drawer", "repr")
        output = default_output
    output = output.lower()

    # Choose drawing backend:
    drawers = {
        "text": TextMatrix,
        "latex_source": state_to_latex,
        "qsphere": plot_state_qsphere,
        "hinton": plot_state_hinton,
        "bloch": plot_bloch_multivector,
        "city": plot_state_city,
        "paulivec": plot_state_paulivec,
    }
    if output == "latex":
        _optionals.HAS_IPYTHON.require_now("state_drawer")
        from IPython.display import Latex

        draw_func = drawers["latex_source"]
        return Latex(f"$${draw_func(state, **drawer_args)}$$")

    if output == "repr":
        return state.__repr__()

    try:
        draw_func = drawers[output]
        return draw_func(state, **drawer_args)
    except KeyError as err:
        raise ValueError(
            f"""'{output}' is not a valid option for drawing {type(state).__name__}
             objects. Please choose from:
            'text', 'latex', 'latex_source', 'qsphere', 'hinton',
            'bloch', 'city' or 'paulivec'."""
        ) from err


def _bloch_multivector_data(state):
    """Return list of Bloch vectors for each qubit

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        list: list of Bloch vectors (x, y, z) for each qubit.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = DensityMatrix(state)
    num = rho.num_qubits
    if num is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    pauli_singles = PauliList(["X", "Y", "Z"])
    bloch_data = []
    for i in range(num):
        if num > 1:
            paulis = PauliList.from_symplectic(
                np.zeros((3, (num - 1)), dtype=bool), np.zeros((3, (num - 1)), dtype=bool)
            ).insert(i, pauli_singles, qubit=True)
        else:
            paulis = pauli_singles
        bloch_state = [np.real(np.trace(np.dot(mat, rho.data))) for mat in paulis.matrix_iter()]
        bloch_data.append(bloch_state)
    return bloch_data


def _paulivec_data(state):
    """Return paulivec data for plotting.

    Args:
        state (DensityMatrix or Statevector): an N-qubit state.

    Returns:
        tuple: (labels, values) for Pauli vector.

    Raises:
        VisualizationError: if input is not an N-qubit state.
    """
    rho = SparsePauliOp.from_operator(DensityMatrix(state))
    if rho.num_qubits is None:
        raise VisualizationError("Input is not a multi-qubit quantum state.")
    return rho.paulis.to_labels(), np.real(rho.coeffs * 2**rho.num_qubits)
