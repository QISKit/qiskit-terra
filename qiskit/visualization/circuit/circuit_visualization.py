# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Module for the primary interface to the circuit drawers.

This module contains the end user facing API for drawing quantum circuits.
There are 3 available drawer backends:

 0. ASCII art
 1. LaTeX
 2. Matplotlib

This provides a single function entry point to drawing a circuit object with
any of the backends.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import typing
from warnings import warn

from qiskit import user_config
from qiskit.circuit import ControlFlowOp, Measure
from qiskit.utils import optionals as _optionals

from ..exceptions import VisualizationError
from ..utils import _trim as trim_image
from . import _utils
from . import latex as _latex
from . import matplotlib as _matplotlib
from . import text as _text

if typing.TYPE_CHECKING:
    from typing import Any
    from qiskit.circuit import QuantumCircuit


logger = logging.getLogger(__name__)


def circuit_drawer(
    circuit: QuantumCircuit,
    scale: float | None = None,
    filename: str | None = None,
    style: dict | str | None = None,
    output: str | None = None,
    interactive: bool = False,
    plot_barriers: bool = True,
    reverse_bits: bool | None = None,
    justify: str | None = None,
    vertical_compression: str | None = "medium",
    idle_wires: bool | str | None = None,
    with_layout: bool = True,
    fold: int | None = None,
    # The type of ax is matplotlib.axes.Axes, but this is not a fixed dependency, so cannot be
    # safely forward-referenced.
    ax: Any | None = None,
    initial_state: bool = False,
    cregbundle: bool | None = None,
    wire_order: list[int] | None = None,
    expr_len: int = 30,
    measure_arrows: bool | None = None,
    barrier_label_len: int = 16,
):
    r"""Draw the quantum circuit. Use the output parameter to choose the drawing format:

    ``text``
        ASCII art TextDrawing that can be printed in the console.

    ``mpl``
        Images with color rendered purely in Python using matplotlib.

    ``latex``
        High-quality images compiled via LaTeX.

        .. warning::
            This will call an installed system version of ``pdflatex`` on arbitrary user input by
            design (such as to render custom code in :attr:`.Instruction.label`), so should only be
            used on trusted data.

    ``latex_source``
        Raw uncompiled LaTeX output.  This is the source of what would be rendered by the
        ``latex`` drawer.

    .. warning::

        Support for :class:`~.expr.Expr` nodes in conditions and :attr:`.SwitchCaseOp.target`
        fields is preliminary and incomplete.  The ``text`` and ``mpl`` drawers will make a
        best-effort attempt to show data dependencies, but the LaTeX-based drawers will skip
        these completely.

    Args:
        circuit: The circuit to visualize.
        scale: Scale of image to draw (shrink if ``< 1.0``). Only used by
            the ``mpl``, ``latex`` and ``latex_source`` outputs. Defaults to ``1.0``.
        filename: File path to save image to. Defaults to ``None`` (result not saved in a file).
        style: Style name, file name of style JSON file, or a dictionary specifying the style.

            * The supported style names are ``"iqp"`` (default), ``"iqp-dark"``, ``"clifford"``,
                ``"textbook"`` and ``"bw"``.
            * If given a JSON file, e.g. ``my_style.json`` or ``my_style`` (the ``.json``
                extension may be omitted), this function attempts to load the style dictionary
                from that location. Note, that the JSON file must completely specify the
                visualization specifications. The file is searched for in
                ``qiskit/visualization/circuit/styles``, the current working directory, and
                the location specified in ``~/.qiskit/settings.conf``.
            * If a dictionary, every entry overrides the default configuration. If the
                ``"name"`` key is given, the default configuration is given by that style.
                For example, ``{"name": "textbook", "subfontsize": 5}`` loads the ``"textbook"``
                style and sets the subfontsize (e.g. the gate angles) to ``5``.
            * If ``None`` the default style ``"iqp"`` is used or, if given, the default style
                specified in ``~/.qiskit/settings.conf``.

        output: Select the output method to use for drawing the circuit.
            Valid choices are ``text``, ``mpl``, ``latex``, ``latex_source``.
            By default, the ``text`` drawer is used unless the user config file
            (usually ``~/.qiskit/settings.conf``) has an alternative backend set
            as the default. For example, ``circuit_drawer = latex``. If the output
            kwarg is set, that backend will always be used over the default in
            the user config file.
        interactive: When set to ``True``, show the circuit in a new window
            (for ``mpl`` this depends on the matplotlib backend being used
            supporting this). Note when used with either the `text` or the
            ``latex_source`` output type this has no effect and will be silently
            ignored. Defaults to ``False``.
        reverse_bits: When set to ``True``, reverse the bit order inside
            registers for the output visualization. Defaults to ``False`` unless the
            user config file (usually ``~/.qiskit/settings.conf``) has an
            alternative value set. For example, ``circuit_reverse_bits = True``.
        plot_barriers: Enable/disable drawing barriers in the output
            circuit. Defaults to ``True``.
        justify: Options are ``"left"``, ``"right"`` or ``"none"`` (str).
            If anything else is supplied, left justified will be used instead.
            It refers to where gates should be placed in the output circuit if
            there is an option. ``none`` results in each gate being placed in
            its own column. Defaults to ``left``.
        vertical_compression: ``high``, ``medium`` or ``low``. It
            merges the lines generated by the `text` output so the drawing
            will take less vertical room.  Default is ``medium``. Only used by
            the ``text`` output, will be silently ignored otherwise.
        idle_wires: Include (or not) idle wires (wires with no circuit elements)
            in output visualization. The string ``"auto"`` is also possible, in which
            case idle wires are show except that the circuit has a layout attached.
            Default is ``"auto"`` unless the
            user config file (usually ``~/.qiskit/settings.conf``) has an
            alternative value set. For example, ``circuit_idle_wires = False``.
        with_layout: Include layout information, with labels on the
            physical layout. Default is ``True``.
        fold: Sets pagination. It can be disabled using -1. In ``text``,
            sets the length of the lines. This is useful when the drawing does
            not fit in the console. If None (default), it will try to guess the
            console width using ``shutil.get_terminal_size()``. However, if
            running in jupyter, the default line length is set to 80 characters.
            In ``mpl``, it is the number of (visual) layers before folding.
            Default is 25.
        ax: Only used by the `mpl` backend. An optional ``matplotlib.axes.Axes``
            object to be used for the visualization output. If none is
            specified, a new matplotlib Figure will be created and used.
            Additionally, if specified there will be no returned Figure since
            it is redundant.
        initial_state: Adds :math:`|0\rangle` in the beginning of the qubit wires and
            :math:`0` to classical wires. Default is ``False``.
        cregbundle: If set to ``True``, bundle classical registers.
            Default is ``True``, except for when ``output`` is set to  ``"text"``.
        wire_order: A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (``num_qubits`` + ``num_clbits``).
        expr_len: The number of characters to display if an :class:`~.expr.Expr`
            is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.
        measure_arrows: If True, draw an arrow from each measure box down to the classical bit
            or register where the measure value is placed. If False, do not draw arrow, but
            instead place the name of the bit or register in the measure box.
            Default is ``True`` unless the user config file (usually ``~/.qiskit/settings.conf``)
            has an alternative value set. For example, ``circuit_measure_arrows = False``.
        barrier_label_len: The number of characters to display for
            :class:`.Barrier` labels in the output circuit. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.

    Returns:
        :class:`.TextDrawing` or :class:`matplotlib.figure` or :class:`PIL.Image` or
        :class:`str`:

        * ``TextDrawing`` (if ``output='text'``)
            A drawing that can be printed as ascii art.
        * ``matplotlib.figure.Figure`` (if ``output='mpl'``)
            A matplotlib figure object for the circuit diagram.
        * ``PIL.Image`` (if ``output='latex``')
            An in-memory representation of the image of the circuit diagram.
        * ``str`` (if ``output='latex_source'``)
            The LaTeX source code for visualizing the circuit diagram.

    Raises:
        VisualizationError: when an invalid output method is selected
        ImportError: when the output methods requires non-installed libraries.

    Example:
        .. plot::
            :alt: Circuit diagram output by the previous code.
            :include-source:

            from qiskit import QuantumCircuit
            from qiskit.visualization import circuit_drawer
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            qc.measure(0, 0)
            circuit_drawer(qc, output="mpl", style={"backgroundcolor": "#EEEEEE"})
    """
    image = None
    expr_len = max(expr_len, 0)
    barrier_label_len = max(barrier_label_len, 0)
    config = user_config.get_config()
    # Get default from config file else use text
    default_output = "text"
    default_reverse_bits = False
    default_idle_wires = config.get("circuit_idle_wires", "auto")
    default_measure_arrows = config.get("circuit_measure_arrows", True)
    if config:
        default_output = config.get("circuit_drawer", "text")
        if default_output == "auto":
            if _optionals.HAS_MATPLOTLIB:
                default_output = "mpl"
            else:
                default_output = "text"
        if wire_order is None:
            default_reverse_bits = config.get("circuit_reverse_bits", False)
    if output is None:
        output = default_output

    if reverse_bits is None:
        reverse_bits = default_reverse_bits

    if idle_wires is None:
        idle_wires = default_idle_wires
    if isinstance(idle_wires, str):
        if idle_wires == "auto":
            idle_wires = hasattr(circuit, "_layout") and circuit._layout is None
        else:
            raise VisualizationError(f"Parameter idle_wires={idle_wires} unrecognized.")

    if measure_arrows is None:
        measure_arrows = default_measure_arrows

    if wire_order is not None and reverse_bits:
        raise VisualizationError(
            "The wire_order option cannot be set when the reverse_bits option is True."
        )

    complete_wire_order = wire_order
    if wire_order is not None:
        wire_order_len = len(wire_order)
        total_wire_len = circuit.num_qubits + circuit.num_clbits
        if wire_order_len not in [circuit.num_qubits, total_wire_len]:
            raise VisualizationError(
                f"The wire_order list (length {wire_order_len}) should as long as "
                f"the number of qubits ({circuit.num_qubits}) or the "
                f"total numbers of qubits and classical bits {total_wire_len}."
            )

        if len(set(wire_order)) != len(wire_order):
            raise VisualizationError("The wire_order list should not have repeated elements.")

        if wire_order_len == circuit.num_qubits:
            complete_wire_order = wire_order + list(range(circuit.num_qubits, total_wire_len))

    if (
        circuit.clbits
        and (reverse_bits or wire_order is not None)
        and not set(wire_order or []).issubset(set(range(circuit.num_qubits)))
    ):
        if cregbundle:
            warn(
                "cregbundle set to False since either reverse_bits or wire_order "
                "(over classical bit) has been set.",
                RuntimeWarning,
                2,
            )
        cregbundle = False

    def check_clbit_in_inst(circuit, cregbundle):
        if cregbundle is False:
            return False
        for inst in circuit.data:
            if isinstance(inst.operation, ControlFlowOp):
                for block in inst.operation.blocks:
                    if check_clbit_in_inst(block, cregbundle) is False:
                        return False
            elif inst.clbits and not isinstance(inst.operation, Measure):
                if cregbundle is not False:
                    warn(
                        "Cregbundle set to False since an instruction needs to refer"
                        " to individual classical wire",
                        RuntimeWarning,
                        3,
                    )
                return False

        return True

    cregbundle = check_clbit_in_inst(circuit, cregbundle)

    if output == "text":
        return _text_circuit_drawer(
            circuit,
            filename=filename,
            reverse_bits=reverse_bits,
            plot_barriers=plot_barriers,
            justify=justify,
            vertical_compression=vertical_compression,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=complete_wire_order,
            expr_len=expr_len,
            barrier_label_len=barrier_label_len,
            measure_arrows=measure_arrows,
        )
    elif output == "latex":
        image = _latex_circuit_drawer(
            circuit,
            filename=filename,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=complete_wire_order,
            barrier_label_len=barrier_label_len,
        )
    elif output == "latex_source":
        return _generate_latex_source(
            circuit,
            filename=filename,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=complete_wire_order,
            barrier_label_len=barrier_label_len,
        )
    elif output == "mpl":
        image = _matplotlib_circuit_drawer(
            circuit,
            scale=scale,
            filename=filename,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            fold=fold,
            ax=ax,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=complete_wire_order,
            expr_len=expr_len,
            measure_arrows=measure_arrows,
            barrier_label_len=barrier_label_len,
        )
    else:
        raise VisualizationError(
            f"Invalid output type {output} selected. The only valid choices "
            "are text, latex, latex_source, and mpl"
        )
    if image and interactive:
        image.show()
    return image


# -----------------------------------------------------------------------------
# _text_circuit_drawer
# -----------------------------------------------------------------------------


def _text_circuit_drawer(
    circuit,
    filename=None,
    reverse_bits=False,
    plot_barriers=True,
    justify=None,
    vertical_compression="high",
    idle_wires=True,
    with_layout=True,
    fold=None,
    initial_state=True,
    cregbundle=None,
    encoding=None,
    wire_order=None,
    expr_len=30,
    measure_arrows=True,
    barrier_label_len=16,
):
    """Draws a circuit using ascii art.

    Args:
        circuit (QuantumCircuit): Input circuit
        filename (str): Optional filename to write the result
        reverse_bits (bool): Rearrange the bits in reverse order.
        plot_barriers (bool): Draws the barriers when they are there.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        vertical_compression (string): `high`, `medium`, or `low`. It merges the
            lines so the drawing will take less vertical room. Default is `high`.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information with labels on the physical
            layout. Default: True
        fold (int): Optional. Breaks the circuit drawing to this length. This
            is useful when the drawing does not fit in the console. If
            None (default), it will try to guess the console width using
            `shutil.get_terminal_size()`. If you don't want pagination
            at all, set `fold=-1`.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
            Default: ``True``.
        encoding (str): Optional. Sets the encoding preference of the output.
            Default: ``sys.stdout.encoding``.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        expr_len (int): Optional. The number of characters to display if an :class:`~.expr.Expr`
            is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.
        measure_arrows: If True, draw an arrow from each measure box down to the classical bit
            or register where the measure value is placed. If False, do not draw arrow, but
            instead place the name of the bit or register in the measure box.
        barrier_label_len (int): Optional. The number of characters to display for
            :class:`.Barrier` labels. If this number is exceeded, the string will be truncated.

    Returns:
        TextDrawing: An instance that, when printed, draws the circuit in ascii art.

    Raises:
        VisualizationError: When the filename extension is not .txt.
    """
    qubits, clbits, nodes = _utils._get_layered_instructions(
        circuit,
        reverse_bits=reverse_bits,
        justify=justify,
        idle_wires=idle_wires,
        wire_order=wire_order,
        measure_arrows=measure_arrows,
    )
    text_drawing = _text.TextDrawing(
        qubits,
        clbits,
        nodes,
        circuit,
        reverse_bits=reverse_bits,
        initial_state=initial_state,
        cregbundle=cregbundle,
        encoding=encoding,
        with_layout=with_layout,
        expr_len=expr_len,
        barrier_label_len=barrier_label_len,
        measure_arrows=measure_arrows,
    )
    text_drawing.plotbarriers = plot_barriers
    text_drawing.line_length = fold
    text_drawing.vertical_compression = vertical_compression

    if filename:
        text_drawing.dump(filename, encoding=encoding)
    return text_drawing


# -----------------------------------------------------------------------------
# latex_circuit_drawer
# -----------------------------------------------------------------------------


@_optionals.HAS_PDFLATEX.require_in_call("LaTeX circuit drawing")
@_optionals.HAS_PDFTOCAIRO.require_in_call("LaTeX circuit drawing")
@_optionals.HAS_PIL.require_in_call("LaTeX circuit drawing")
def _latex_circuit_drawer(
    circuit,
    scale=0.7,
    style=None,
    filename=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    initial_state=False,
    cregbundle=None,
    wire_order=None,
    barrier_label_len=16,
):
    """Draw a quantum circuit based on latex (Qcircuit package)

    Requires version >=2.6.0 of the qcircuit LaTeX package.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): file path to save image to
        reverse_bits (bool): When set to True reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.  On by default, if
            this is possible for the given circuit, otherwise off.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        barrier_label_len (int): Optional. The number of characters to display for
            :class:`.Barrier` labels. If this number is exceeded, the string will be truncated.

    Returns:
        PIL.Image: an in-memory representation of the circuit diagram

    Raises:
        MissingOptionalLibraryError: if pillow, pdflatex, or poppler are not installed
        VisualizationError: if one of the conversion utilities failed for some internal or
            file-access reason.
    """
    from PIL import Image

    tmpfilename = "circuit"
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmppath = os.path.join(tmpdirname, tmpfilename + ".tex")
        _generate_latex_source(
            circuit,
            filename=tmppath,
            scale=scale,
            style=style,
            plot_barriers=plot_barriers,
            reverse_bits=reverse_bits,
            justify=justify,
            idle_wires=idle_wires,
            with_layout=with_layout,
            initial_state=initial_state,
            cregbundle=cregbundle,
            wire_order=wire_order,
            barrier_label_len=barrier_label_len,
        )

        try:
            subprocess.run(
                [
                    "pdflatex",
                    "-halt-on-error",
                    f"-output-directory={tmpdirname}",
                    f"{tmpfilename + '.tex'}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except OSError as exc:
            # OSError should generally not occur, because it's usually only triggered if `pdflatex`
            # doesn't exist as a command, but we've already checked that.
            raise VisualizationError("`pdflatex` command could not be run.") from exc
        except subprocess.CalledProcessError as exc:
            with open("latex_error.log", "wb") as error_file:
                error_file.write(exc.stdout)
            logger.warning(
                "Unable to compile LaTeX. Perhaps you are missing the `qcircuit` package."
                " The output from the `pdflatex` command is in `latex_error.log`."
            )
            raise VisualizationError(
                "`pdflatex` call did not succeed: see `latex_error.log`."
            ) from exc
        base = os.path.join(tmpdirname, tmpfilename)
        try:
            subprocess.run(
                ["pdftocairo", "-singlefile", "-png", "-q", base + ".pdf", base],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            message = "`pdftocairo` failed to produce an image."
            logger.warning(message)
            raise VisualizationError(message) from exc
        image = Image.open(base + ".png")
        image = trim_image(image)
        if filename:
            if filename.endswith(".pdf"):
                shutil.move(base + ".pdf", filename)
            else:
                try:
                    image.save(filename)
                except (ValueError, OSError) as exc:
                    raise VisualizationError(
                        f"Pillow could not write the image file '{filename}'."
                    ) from exc
        return image


def _generate_latex_source(
    circuit,
    filename=None,
    scale=0.7,
    style=None,
    reverse_bits=False,
    plot_barriers=True,
    justify=None,
    idle_wires=True,
    with_layout=True,
    initial_state=False,
    cregbundle=None,
    wire_order=None,
    barrier_label_len=16,
):
    """Convert QuantumCircuit to LaTeX string.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        style (dict or str): dictionary of style or file name of style file
        filename (str): optional filename to write latex
        reverse_bits (bool): When set to True reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str) : `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True, bundle classical registers.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        barrier_label_len (int): Optional. The number of characters to display for
            :class:`.Barrier` labels. If this number is exceeded, the string will be truncated.

    Returns:
        str: Latex string appropriate for writing to file.
    """
    qubits, clbits, nodes = _utils._get_layered_instructions(
        circuit,
        reverse_bits=reverse_bits,
        justify=justify,
        idle_wires=idle_wires,
        wire_order=wire_order,
    )
    qcimg = _latex.QCircuitImage(
        qubits,
        clbits,
        nodes,
        scale,
        style=style,
        reverse_bits=reverse_bits,
        plot_barriers=plot_barriers,
        initial_state=initial_state,
        cregbundle=cregbundle,
        with_layout=with_layout,
        circuit=circuit,
        barrier_label_len=barrier_label_len,
    )
    latex = qcimg.latex()
    if filename:
        with open(filename, "w") as latex_file:
            latex_file.write(latex)

    return latex


# -----------------------------------------------------------------------------
# matplotlib_circuit_drawer
# -----------------------------------------------------------------------------


def _matplotlib_circuit_drawer(
    circuit,
    scale=None,
    filename=None,
    style=None,
    plot_barriers=True,
    reverse_bits=False,
    justify=None,
    idle_wires=True,
    with_layout=True,
    fold=None,
    ax=None,
    initial_state=False,
    cregbundle=None,
    wire_order=None,
    expr_len=30,
    measure_arrows=None,
    barrier_label_len=16,
):
    """Draw a quantum circuit based on matplotlib.
    If `%matplotlib inline` is invoked in a Jupyter notebook, it visualizes a circuit inline.
    We recommend `%config InlineBackend.figure_format = 'svg'` for the inline visualization.

    Args:
        circuit (QuantumCircuit): a quantum circuit
        scale (float): scaling factor
        filename (str): file path to save image to
        style (dict or str): dictionary of style or file name of style file
        reverse_bits (bool): When set to True, reverse the bit order inside
            registers for the output visualization.
        plot_barriers (bool): Enable/disable drawing barriers in the output
            circuit. Defaults to True.
        justify (str): `left`, `right` or `none`. Defaults to `left`. Says how
            the circuit should be justified.
        idle_wires (bool): Include idle wires. Default is True.
        with_layout (bool): Include layout information, with labels on the physical
            layout. Default: True.
        fold (int): Number of vertical layers allowed before folding. Default is 25.
        ax (matplotlib.axes.Axes): An optional Axes object to be used for
            the visualization output. If none is specified, a new matplotlib
            Figure will be created and used. Additionally, if specified there
            will be no returned Figure since it is redundant.
        initial_state (bool): Optional. Adds |0> in the beginning of the line.
            Default: `False`.
        cregbundle (bool): Optional. If set True bundle classical registers.
            Default: ``True``.
        wire_order (list): Optional. A list of integers used to reorder the display
            of the bits. The list must have an entry for every bit with the bits
            in the range 0 to (num_qubits + num_clbits).
        expr_len (int): Optional. The number of characters to display if an :class:`~.expr.Expr`
            is used for the condition in a :class:`.ControlFlowOp`. If this number is exceeded,
            the string will be truncated at that number and '...' added to the end.
        measure_arrows: If True, draw an arrow from each measure box down to the classical bit
            or register where the measure value is placed. If False, do not draw arrow, but
            instead place the name of the bit or register in the measure box.
        barrier_label_len (int): Optional. The number of characters to display for
            :class:`.Barrier` labels. If this number is exceeded, the string will be truncated.

    Returns:
        matplotlib.figure: a matplotlib figure object for the circuit diagram
            if the ``ax`` kwarg is not set.
    """

    qubits, clbits, nodes = _utils._get_layered_instructions(
        circuit,
        reverse_bits=reverse_bits,
        justify=justify,
        idle_wires=idle_wires,
        wire_order=wire_order,
        measure_arrows=measure_arrows,
    )
    if fold is None:
        fold = 25

    qcd = _matplotlib.MatplotlibDrawer(
        qubits,
        clbits,
        nodes,
        circuit,
        scale=scale,
        style=style,
        reverse_bits=reverse_bits,
        plot_barriers=plot_barriers,
        fold=fold,
        ax=ax,
        initial_state=initial_state,
        cregbundle=cregbundle,
        with_layout=with_layout,
        expr_len=expr_len,
        barrier_label_len=barrier_label_len,
        measure_arrows=measure_arrows,
    )
    return qcd.draw(filename)
