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


"""mpl circuit visualization backend."""

import collections
import itertools
import re
from io import StringIO

import numpy as np

from qiskit.circuit import (
    QuantumCircuit,
    Qubit,
    Clbit,
    ClassicalRegister,
    ControlledGate,
    Measure,
    ControlFlowOp,
    BoxOp,
    WhileLoopOp,
    IfElseOp,
    ForLoopOp,
    SwitchCaseOp,
    CircuitError,
)
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.classical import expr
from qiskit.circuit.annotated_operation import _canonicalize_modifiers, ControlModifier
from qiskit.circuit.library import Initialize
from qiskit.circuit.library.standard_gates import (
    SwapGate,
    RZZGate,
    U1Gate,
    PhaseGate,
    XGate,
    ZGate,
)
from qiskit.qasm3 import ast
from qiskit.qasm3.exporter import _ExprBuilder
from qiskit.qasm3.printer import BasicPrinter

from qiskit.circuit.tools.pi_check import pi_check
from qiskit.utils import optionals as _optionals

from qiskit.visualization.style import load_style
from qiskit.visualization.circuit.qcstyle import MPLDefaultStyle, MPLStyleDict
from ._utils import (
    get_gate_ctrl_text,
    get_param_str,
    get_wire_map,
    get_bit_register,
    get_bit_reg_index,
    get_wire_label,
    get_condition_label_val,
    _get_layered_instructions,
)
from ..utils import matplotlib_close_if_inline

# Default gate width and height
WID = 0.65
HIG = 0.65

# Z dimension order for different drawing types
PORDER_REGLINE = 1
PORDER_FLOW = 3
PORDER_MASK = 4
PORDER_LINE = 6
PORDER_LINE_PLUS = 7
PORDER_BARRIER = 8
PORDER_GATE = 10
PORDER_GATE_PLUS = 11
PORDER_TEXT = 13

INFINITE_FOLD = 10000000


@_optionals.HAS_MATPLOTLIB.require_in_instance
@_optionals.HAS_PYLATEX.require_in_instance
class MatplotlibDrawer:
    """Matplotlib drawer class called from circuit_drawer"""

    _mathmode_regex = re.compile(r"(?<!\\)\$(.*)(?<!\\)\$")

    def __init__(
        self,
        qubits,
        clbits,
        nodes,
        circuit,
        scale=None,
        style=None,
        reverse_bits=False,
        plot_barriers=True,
        fold=25,
        ax=None,
        initial_state=False,
        cregbundle=None,
        with_layout=False,
        expr_len=30,
        measure_arrows=None,
        barrier_label_len=16,
    ):
        self._circuit = circuit
        self._qubits = qubits
        self._clbits = clbits
        self._nodes = nodes
        self._scale = 1.0 if scale is None else scale

        self._style = style

        self._plot_barriers = plot_barriers
        self._reverse_bits = reverse_bits
        if with_layout:
            if self._circuit._layout:
                self._layout = self._circuit._layout.initial_layout
            else:
                self._layout = None
        else:
            self._layout = None

        self._fold = fold
        if self._fold < 2:
            self._fold = -1

        self._ax = ax

        self._initial_state = initial_state
        self._global_phase = self._circuit.global_phase
        self._expr_len = expr_len
        self._barrier_label_len = barrier_label_len
        self._cregbundle = cregbundle
        self._measure_arrows = measure_arrows

        self._lwidth1 = 1.0
        self._lwidth15 = 1.5
        self._lwidth2 = 2.0
        self._lwidth3 = 3.0
        self._lwidth4 = 4.0

        # Class instances of MatplotlibDrawer for each flow gate - If/Else, For, While, Switch
        self._flow_drawers = {}

        # Set if gate is inside a flow gate
        self._flow_parent = None
        self._flow_wire_map = {}

        # _char_list for finding text_width of names, labels, and params
        self._char_list = {
            " ": (0.0958, 0.0583),
            "!": (0.1208, 0.0729),
            '"': (0.1396, 0.0875),
            "#": (0.2521, 0.1562),
            "$": (0.1917, 0.1167),
            "%": (0.2854, 0.1771),
            "&": (0.2333, 0.1458),
            "'": (0.0833, 0.0521),
            "(": (0.1167, 0.0729),
            ")": (0.1167, 0.0729),
            "*": (0.15, 0.0938),
            "+": (0.25, 0.1562),
            ",": (0.0958, 0.0583),
            "-": (0.1083, 0.0667),
            ".": (0.0958, 0.0604),
            "/": (0.1021, 0.0625),
            "0": (0.1875, 0.1167),
            "1": (0.1896, 0.1167),
            "2": (0.1917, 0.1188),
            "3": (0.1917, 0.1167),
            "4": (0.1917, 0.1188),
            "5": (0.1917, 0.1167),
            "6": (0.1896, 0.1167),
            "7": (0.1917, 0.1188),
            "8": (0.1896, 0.1188),
            "9": (0.1917, 0.1188),
            ":": (0.1021, 0.0604),
            ";": (0.1021, 0.0604),
            "<": (0.25, 0.1542),
            "=": (0.25, 0.1562),
            ">": (0.25, 0.1542),
            "?": (0.1583, 0.0979),
            "@": (0.2979, 0.1854),
            "A": (0.2062, 0.1271),
            "B": (0.2042, 0.1271),
            "C": (0.2083, 0.1292),
            "D": (0.2312, 0.1417),
            "E": (0.1875, 0.1167),
            "F": (0.1708, 0.1062),
            "G": (0.2312, 0.1438),
            "H": (0.225, 0.1396),
            "I": (0.0875, 0.0542),
            "J": (0.0875, 0.0542),
            "K": (0.1958, 0.1208),
            "L": (0.1667, 0.1042),
            "M": (0.2583, 0.1604),
            "N": (0.225, 0.1396),
            "O": (0.2354, 0.1458),
            "P": (0.1812, 0.1125),
            "Q": (0.2354, 0.1458),
            "R": (0.2083, 0.1292),
            "S": (0.1896, 0.1188),
            "T": (0.1854, 0.1125),
            "U": (0.2208, 0.1354),
            "V": (0.2062, 0.1271),
            "W": (0.2958, 0.1833),
            "X": (0.2062, 0.1271),
            "Y": (0.1833, 0.1125),
            "Z": (0.2042, 0.1271),
            "[": (0.1167, 0.075),
            "\\": (0.1021, 0.0625),
            "]": (0.1167, 0.0729),
            "^": (0.2521, 0.1562),
            "_": (0.1521, 0.0938),
            "`": (0.15, 0.0938),
            "a": (0.1854, 0.1146),
            "b": (0.1917, 0.1167),
            "c": (0.1646, 0.1021),
            "d": (0.1896, 0.1188),
            "e": (0.1854, 0.1146),
            "f": (0.1042, 0.0667),
            "g": (0.1896, 0.1188),
            "h": (0.1896, 0.1188),
            "i": (0.0854, 0.0521),
            "j": (0.0854, 0.0521),
            "k": (0.1729, 0.1083),
            "l": (0.0854, 0.0521),
            "m": (0.2917, 0.1812),
            "n": (0.1896, 0.1188),
            "o": (0.1833, 0.1125),
            "p": (0.1917, 0.1167),
            "q": (0.1896, 0.1188),
            "r": (0.125, 0.0771),
            "s": (0.1562, 0.0958),
            "t": (0.1167, 0.0729),
            "u": (0.1896, 0.1188),
            "v": (0.1771, 0.1104),
            "w": (0.2458, 0.1521),
            "x": (0.1771, 0.1104),
            "y": (0.1771, 0.1104),
            "z": (0.1562, 0.0979),
            "{": (0.1917, 0.1188),
            "|": (0.1, 0.0604),
            "}": (0.1896, 0.1188),
        }

    def draw(self, filename=None):
        """Main entry point to 'matplotlib' ('mpl') drawer. Called from
        ``visualization.circuit_drawer`` and from ``QuantumCircuit.draw`` through circuit_drawer.
        """

        # Import matplotlib and load all the figure, window, and style info
        from matplotlib import patches
        from matplotlib import pyplot as plt

        # glob_data contains global values used throughout, "n_lines", "x_offset", "next_x_index",
        # "patches_mod", "subfont_factor"
        glob_data = {}

        glob_data["patches_mod"] = patches
        plt_mod = plt

        self._style, def_font_ratio = load_style(
            self._style,
            style_dict=MPLStyleDict,
            default_style=MPLDefaultStyle(),
            user_config_opt="circuit_mpl_style",
            user_config_path_opt="circuit_mpl_style_path",
        )

        # If font/subfont ratio changes from default, have to scale width calculations for
        # subfont. Font change is auto scaled in the mpl_figure.set_size_inches call in draw()
        glob_data["subfont_factor"] = self._style["sfs"] * def_font_ratio / self._style["fs"]

        # if no user ax, setup default figure. Else use the user figure.
        if self._ax is None:
            is_user_ax = False
            mpl_figure = plt.figure()
            mpl_figure.patch.set_facecolor(color=self._style["bg"])
            self._ax = mpl_figure.add_subplot(111)
        else:
            is_user_ax = True
            mpl_figure = self._ax.get_figure()
        self._ax.axis("off")
        self._ax.set_aspect("equal")
        self._ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

        # All information for the drawing is first loaded into node_data for the gates and into
        # qubits_dict, clbits_dict, and wire_map for the qubits, clbits, and wires,
        # followed by the coordinates for each gate.

        # load the wire map
        wire_map = get_wire_map(self._circuit, self._qubits + self._clbits, self._cregbundle)

        # node_data per node filled with class NodeData attributes
        node_data = {}

        # dicts for the names and locations of register/bit labels
        qubits_dict = {}
        clbits_dict = {}

        # load the _qubit_dict and _clbit_dict with register info
        self._set_bit_reg_info(wire_map, qubits_dict, clbits_dict, glob_data)

        # get layer widths - flow gates are initialized here
        layer_widths = self._get_layer_widths(node_data, wire_map, self._circuit, glob_data)

        # load the coordinates for each top level gate and compute number of folds.
        # coordinates for flow gates are loaded before draw_ops
        max_x_index = self._get_coords(
            node_data, wire_map, self._circuit, layer_widths, qubits_dict, clbits_dict, glob_data
        )
        num_folds = max(0, max_x_index - 1) // self._fold if self._fold > 0 else 0

        # The window size limits are computed, followed by one of the four possible ways
        # of scaling the drawing.

        # compute the window size
        if max_x_index > self._fold > 0:
            xmax = self._fold + glob_data["x_offset"] + 0.1
            ymax = (num_folds + 1) * (glob_data["n_lines"] + 1) - 1
        else:
            x_incr = 0.4 if not self._nodes else 0.9
            xmax = max_x_index + 1 + glob_data["x_offset"] - x_incr
            ymax = glob_data["n_lines"]

        xl = -self._style["margin"][0]
        xr = xmax + self._style["margin"][1]
        yb = -ymax - self._style["margin"][2] + 0.5
        yt = self._style["margin"][3] + 0.5
        self._ax.set_xlim(xl, xr)
        self._ax.set_ylim(yb, yt)

        # update figure size and, for backward compatibility,
        # need to scale by a default value equal to (self._style["fs"] * 3.01 / 72 / 0.65)
        base_fig_w = (xr - xl) * 0.8361111
        base_fig_h = (yt - yb) * 0.8361111
        scale = self._scale

        # if user passes in an ax, this size takes priority over any other settings
        if is_user_ax:
            # from stackoverflow #19306510, get the bbox size for the ax and then reset scale
            bbox = self._ax.get_window_extent().transformed(mpl_figure.dpi_scale_trans.inverted())
            scale = bbox.width / base_fig_w / 0.8361111

        # if scale not 1.0, use this scale factor
        elif self._scale != 1.0:
            mpl_figure.set_size_inches(base_fig_w * self._scale, base_fig_h * self._scale)

        # if "figwidth" style param set, use this to scale
        elif self._style["figwidth"] > 0.0:
            # in order to get actual inches, need to scale by factor
            adj_fig_w = self._style["figwidth"] * 1.282736
            mpl_figure.set_size_inches(adj_fig_w, adj_fig_w * base_fig_h / base_fig_w)
            scale = adj_fig_w / base_fig_w

        # otherwise, display default size
        else:
            mpl_figure.set_size_inches(base_fig_w, base_fig_h)

        # drawing will scale with 'set_size_inches', but fonts and linewidths do not
        if scale != 1.0:
            self._style["fs"] *= scale
            self._style["sfs"] *= scale
            self._lwidth1 = 1.0 * scale
            self._lwidth15 = 1.5 * scale
            self._lwidth2 = 2.0 * scale
            self._lwidth3 = 3.0 * scale
            self._lwidth4 = 4.0 * scale

        # Once the scaling factor has been determined, the global phase, register names
        # and numbers, wires, and gates are drawn
        if self._global_phase:
            plt_mod.text(xl, yt, f"Global Phase: {pi_check(self._global_phase, output='mpl')}")
        self._draw_regs_wires(num_folds, xmax, max_x_index, qubits_dict, clbits_dict, glob_data)
        self._draw_ops(
            self._nodes,
            node_data,
            wire_map,
            self._circuit,
            layer_widths,
            qubits_dict,
            clbits_dict,
            glob_data,
        )
        if filename:
            mpl_figure.savefig(
                filename,
                dpi=self._style["dpi"],
                bbox_inches="tight",
                facecolor=mpl_figure.get_facecolor(),
            )
        if not is_user_ax:
            matplotlib_close_if_inline(mpl_figure)
            return mpl_figure

    def _get_layer_widths(self, node_data, wire_map, outer_circuit, glob_data):
        """Compute the layer_widths for the layers"""

        layer_widths = {}
        for layer_num, layer in enumerate(self._nodes):
            widest_box = WID
            for i, node in enumerate(layer):
                # Put the layer_num in the first node in the layer and put -1 in the rest
                # so that layer widths are not counted more than once
                if i != 0:
                    layer_num = -1
                layer_widths[node] = [1, layer_num, self._flow_parent]

                op = node.op
                node_data[node] = NodeData()
                node_data[node].width = WID
                num_ctrl_qubits = getattr(op, "num_ctrl_qubits", 0)
                if (
                    getattr(op, "_directive", False) and (not op.label or not self._plot_barriers)
                ) or (self._measure_arrows and isinstance(op, Measure)):
                    node_data[node].raw_gate_text = op.name
                    continue

                base_type = getattr(op, "base_gate", None)
                gate_text, ctrl_text, raw_gate_text = get_gate_ctrl_text(
                    op, "mpl", style=self._style
                )
                # Truncate directive labels to match the rendered length so the layer
                # width doesn't account for text that will never be displayed.
                if getattr(op, "_directive", False) and len(gate_text) > self._barrier_label_len:
                    gate_text = gate_text[: self._barrier_label_len] + "..."
                node_data[node].gate_text = gate_text
                node_data[node].ctrl_text = ctrl_text
                # Measure doesn't use raw_gate_text since it displays a dial
                if not isinstance(op, Measure):
                    node_data[node].raw_gate_text = raw_gate_text
                node_data[node].param_text = ""

                # if single qubit, no params, and no labels, layer_width is 1
                if (
                    (len(node.qargs) - num_ctrl_qubits) == 1
                    and len(gate_text) < 3
                    and len(getattr(op, "params", [])) == 0
                    and ctrl_text is None
                ):
                    continue

                if isinstance(op, SwapGate) or isinstance(base_type, SwapGate):
                    continue

                # small increments at end of the 3 _get_text_width calls are for small
                # spacing adjustments between gates
                ctrl_width = (
                    self._get_text_width(ctrl_text, glob_data, fontsize=self._style["sfs"]) - 0.05
                )
                # get param_width, but 0 for gates with array params or circuits in params
                if (
                    len(getattr(op, "params", [])) > 0
                    and not any(isinstance(param, np.ndarray) for param in op.params)
                    and not any(isinstance(param, QuantumCircuit) for param in op.params)
                ):
                    param_text = get_param_str(op, "mpl", ndigits=3)
                    if isinstance(op, Initialize):
                        param_text = f"$[{param_text.replace('$', '')}]$"
                    node_data[node].param_text = param_text
                    raw_param_width = self._get_text_width(
                        param_text, glob_data, fontsize=self._style["sfs"], param=True
                    )
                    param_width = raw_param_width + 0.08
                else:
                    param_width = raw_param_width = 0.0

                # get gate_width for sidetext symmetric gates
                if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                    if isinstance(base_type, PhaseGate):
                        gate_text = "P"
                    raw_gate_width = (
                        self._get_text_width(
                            gate_text + " ()", glob_data, fontsize=self._style["sfs"]
                        )
                        + raw_param_width
                    )
                    gate_width = (raw_gate_width + 0.08) * 1.58

                # Check if a ControlFlowOp - node_data load for these gates is done here
                elif isinstance(node.op, ControlFlowOp):
                    self._flow_drawers[node] = []
                    node_data[node].width = []
                    node_data[node].nest_depth = 0
                    gate_width = 0.0
                    expr_width = 0.0

                    if (isinstance(op, SwitchCaseOp) and isinstance(op.target, expr.Expr)) or (
                        getattr(op, "condition", None) and isinstance(op.condition, expr.Expr)
                    ):

                        def lookup_var(var):
                            """Look up a classical-expression variable or register/bit in our
                            internal symbol table, and return an OQ3-like identifier."""
                            # We don't attempt to disambiguate anything like register/var naming
                            # collisions; we already don't really show classical variables.
                            if isinstance(var, expr.Var):
                                return ast.Identifier(var.name)
                            if isinstance(var, ClassicalRegister):
                                return ast.Identifier(var.name)
                            # Single clbit.  This is not actually the correct way to lookup a bit on
                            # the circuit (it doesn't handle bit bindings fully), but the mpl
                            # drawer doesn't completely track inner-outer _bit_ bindings, only
                            # inner-indices, so we can't fully recover the information losslessly.
                            # Since most control-flow uses the control-flow builders, we should
                            # decay to something usable most of the time.
                            try:
                                register, bit_index, reg_index = get_bit_reg_index(
                                    outer_circuit, var
                                )
                            except CircuitError:
                                # We failed to find the bit due to binding problems - fall back to
                                # something that's probably wrong, but at least disambiguating.
                                return ast.Identifier(f"bit{wire_map[var]}")
                            if register is None:
                                return ast.Identifier(f"bit{bit_index}")
                            return ast.SubscriptedIdentifier(
                                register.name, ast.IntegerLiteral(reg_index)
                            )

                        condition = op.target if isinstance(op, SwitchCaseOp) else op.condition
                        stream = StringIO()
                        BasicPrinter(stream, indent="  ").visit(
                            condition.accept(_ExprBuilder(lookup_var))
                        )
                        expr_text = stream.getvalue()
                        # Truncate expr_text so that first gate is no more than about 3 x_index's over
                        if len(expr_text) > self._expr_len:
                            expr_text = expr_text[: self._expr_len] + "..."
                        node_data[node].expr_text = expr_text

                        expr_width = self._get_text_width(
                            node_data[node].expr_text, glob_data, fontsize=self._style["sfs"]
                        )
                        node_data[node].expr_width = int(expr_width)

                    # Get the list of circuits to iterate over from the blocks
                    circuit_list = list(node.op.blocks)

                    # params is [indexset, loop_param, circuit] for for_loop,
                    # op.cases_specifier() returns jump tuple and circuit for switch/case
                    if isinstance(op, ForLoopOp):
                        node_data[node].indexset = op.params[0]
                    elif isinstance(op, SwitchCaseOp):
                        node_data[node].jump_values = []
                        cases = list(op.cases_specifier())

                        # Create an empty circuit at the head of the circuit_list if a Switch box
                        circuit_list.insert(0, cases[0][1].copy_empty_like())
                        for jump_values, _ in cases:
                            node_data[node].jump_values.append(jump_values)

                    # Now process the circuits inside the ControlFlowOps
                    for circ_num, circuit in enumerate(circuit_list):
                        # Only add expr_width for if, while, and switch
                        raw_gate_width = expr_width if circ_num == 0 else 0.0

                        # Depth of nested ControlFlowOp used for color of box
                        if self._flow_parent is not None:
                            node_data[node].nest_depth = node_data[self._flow_parent].nest_depth + 1

                        # Build the wire_map to be used by this flow op
                        flow_wire_map = wire_map.copy()
                        flow_wire_map.update(
                            {
                                inner: wire_map[outer]
                                for outer, inner in zip(node.qargs, circuit.qubits)
                            }
                        )
                        for outer, inner in zip(node.cargs, circuit.clbits):
                            if self._cregbundle and (
                                (in_reg := get_bit_register(outer_circuit, inner)) is not None
                            ):
                                out_reg = get_bit_register(outer_circuit, outer)
                                flow_wire_map.update({in_reg: wire_map[out_reg]})
                            else:
                                flow_wire_map.update({inner: wire_map[outer]})

                        # Get the layered node lists and instantiate a new drawer class for
                        # the circuit inside the ControlFlowOp.
                        qubits, clbits, flow_nodes = _get_layered_instructions(
                            circuit, wire_map=flow_wire_map, measure_arrows=self._measure_arrows
                        )
                        flow_drawer = MatplotlibDrawer(
                            qubits,
                            clbits,
                            flow_nodes,
                            circuit,
                            style=self._style,
                            plot_barriers=self._plot_barriers,
                            fold=self._fold,
                            cregbundle=self._cregbundle,
                        )

                        # flow_parent is the parent of the new class instance
                        flow_drawer._flow_parent = node
                        flow_drawer._flow_wire_map = flow_wire_map
                        self._flow_drawers[node].append(flow_drawer)

                        # Recursively call _get_layer_widths for the circuit inside the ControlFlowOp
                        flow_widths = flow_drawer._get_layer_widths(
                            node_data, flow_wire_map, outer_circuit, glob_data
                        )
                        layer_widths.update(flow_widths)

                        for flow_layer in flow_nodes:
                            for flow_node in flow_layer:
                                node_data[flow_node].circ_num = circ_num

                        # Add up the width values of the same flow_parent that are not -1
                        # to get the raw_gate_width
                        for width, layer_num, flow_parent in flow_widths.values():
                            if layer_num != -1 and flow_parent == flow_drawer._flow_parent:
                                raw_gate_width += width
                                # This is necessary to prevent 1 being added to the width of a
                                # BoxOp in layer_widths at the end of this method
                                if isinstance(node.op, BoxOp):
                                    raw_gate_width -= 0.001

                        # Need extra incr of 1.0 for else and case boxes
                        gate_width += raw_gate_width + (1.0 if circ_num > 0 else 0.0)

                        # Minor adjustment so else and case section gates align with indexes
                        if circ_num > 0:
                            raw_gate_width += 0.045

                        # If expr_width has a value, remove the decimal portion from raw_gate_widthl
                        if not isinstance(op, ForLoopOp) and circ_num == 0:
                            node_data[node].width.append(raw_gate_width - (expr_width % 1))
                        else:
                            node_data[node].width.append(raw_gate_width)

                # If measure_arrows is False, this section gets the layer width for a measure
                # based on the width of register_bit and puts it into the param_width. If the
                # register_bit is small enough, the gate will just use the WID width.
                elif not self._measure_arrows and isinstance(op, Measure):
                    register, _, reg_index = get_bit_reg_index(outer_circuit, node.cargs[0])
                    if register is not None:
                        param_text = f"{register.name}_{reg_index}"
                    else:
                        param_text = f"{reg_index}"
                    raw_param_width = self._get_text_width(
                        param_text, glob_data, fontsize=self._style["sfs"], param=True
                    )
                    param_width = raw_param_width
                    raw_gate_width = gate_width = ctrl_width = 0.0

                # Otherwise, standard gate or multiqubit gate
                else:
                    raw_gate_width = self._get_text_width(
                        gate_text, glob_data, fontsize=self._style["fs"]
                    )
                    gate_width = raw_gate_width + 0.10
                    # add .21 for the qubit numbers on the left of the multibit gates
                    if len(node.qargs) - num_ctrl_qubits > 1:
                        gate_width += 0.21

                box_width = max(gate_width, ctrl_width, param_width, WID)
                widest_box = max(widest_box, box_width)
                if not isinstance(node.op, ControlFlowOp):
                    node_data[node].width = max(raw_gate_width, raw_param_width)
            for node in layer:
                layer_widths[node][0] = int(widest_box) + 1

        return layer_widths

    def _set_bit_reg_info(self, wire_map, qubits_dict, clbits_dict, glob_data):
        """Get all the info for drawing bit/reg names and numbers"""

        longest_wire_label_width = 0
        glob_data["n_lines"] = 0
        initial_qbit = r" $|0\rangle$" if self._initial_state else ""
        initial_cbit = " 0" if self._initial_state else ""

        idx = 0
        pos = y_off = -len(self._qubits) + 1
        for ii, wire in enumerate(wire_map):
            # if it's a creg, register is the key and just load the index
            if isinstance(wire, ClassicalRegister):
                # If wire came from ControlFlowOp and not in clbits, don't draw it
                if wire[0] not in self._clbits:
                    continue
                register = wire
                index = wire_map[wire]

            # otherwise, get the register from find_bit and use bit_index if
            # it's a bit, or the index of the bit in the register if it's a reg
            else:
                # If wire came from ControlFlowOp and not in qubits or clbits, don't draw it
                if wire not in self._qubits + self._clbits:
                    continue
                register, bit_index, reg_index = get_bit_reg_index(self._circuit, wire)
                index = bit_index if register is None else reg_index

            wire_label = get_wire_label(
                "mpl", register, index, layout=self._layout, cregbundle=self._cregbundle
            )
            initial_bit = initial_qbit if isinstance(wire, Qubit) else initial_cbit

            # for cregs with cregbundle on, don't use math formatting, which means
            # no italics
            if isinstance(wire, Qubit) or register is None or not self._cregbundle:
                wire_label = "$" + wire_label + "$"
            wire_label += initial_bit

            reg_size = (
                0 if register is None or isinstance(wire, ClassicalRegister) else register.size
            )
            reg_remove_under = 0 if reg_size < 2 else 1
            text_width = (
                self._get_text_width(
                    wire_label, glob_data, self._style["fs"], reg_remove_under=reg_remove_under
                )
                * 1.15
            )
            longest_wire_label_width = max(longest_wire_label_width, text_width)

            if isinstance(wire, Qubit):
                pos = -ii
                qubits_dict[ii] = {
                    "y": pos,
                    "wire_label": wire_label,
                }
                glob_data["n_lines"] += 1
            else:
                if (
                    not self._cregbundle
                    or register is None
                    or (self._cregbundle and isinstance(wire, ClassicalRegister))
                ):
                    glob_data["n_lines"] += 1
                    idx += 1

                pos = y_off - idx
                clbits_dict[ii] = {
                    "y": pos,
                    "wire_label": wire_label,
                    "register": register,
                }
        glob_data["x_offset"] = -1.2 + longest_wire_label_width

    def _get_coords(
        self,
        node_data,
        wire_map,
        outer_circuit,
        layer_widths,
        qubits_dict,
        clbits_dict,
        glob_data,
        flow_parent=None,
    ):
        """Load all the coordinate info needed to place the gates on the drawing."""

        prev_x_index = -1
        for layer in self._nodes:
            curr_x_index = prev_x_index + 1
            l_width = []
            for node in layer:
                # For gates inside a flow op set the x_index and if it's an else or case,
                # increment by if/switch width. If more cases increment by width of previous cases.
                if flow_parent is not None:
                    node_data[node].inside_flow = True
                    # front_space provides a space for 'If', 'While', etc. which is not
                    # necessary for a BoxOp
                    front_space = 0 if isinstance(flow_parent.op, BoxOp) else 1
                    node_data[node].x_index = (
                        node_data[flow_parent].x_index + curr_x_index + front_space
                    )

                    # If an else or case
                    if node_data[node].circ_num > 0:
                        for width in node_data[flow_parent].width[: node_data[node].circ_num]:
                            node_data[node].x_index += int(width) + 1
                        x_index = node_data[node].x_index
                    # Add expr_width to if, while, or switch if expr used
                    else:
                        x_index = node_data[node].x_index + node_data[flow_parent].expr_width
                else:
                    node_data[node].inside_flow = False
                    x_index = curr_x_index

                # get qubit indexes
                q_indxs = []
                for qarg in node.qargs:
                    if qarg in self._qubits:
                        q_indxs.append(wire_map[qarg])

                # get clbit indexes
                c_indxs = []
                for carg in node.cargs:
                    if carg in self._clbits:
                        if self._cregbundle:
                            register = get_bit_register(outer_circuit, carg)
                            if register is not None:
                                c_indxs.append(wire_map[register])
                            else:
                                c_indxs.append(wire_map[carg])
                        else:
                            c_indxs.append(wire_map[carg])

                flow_op = isinstance(node.op, ControlFlowOp)

                # qubit coordinates
                node_data[node].q_xy = [
                    self._plot_coord(
                        x_index,
                        qubits_dict[ii]["y"],
                        layer_widths[node][0],
                        glob_data,
                        flow_op,
                    )
                    for ii in q_indxs
                ]
                # clbit coordinates
                node_data[node].c_xy = [
                    self._plot_coord(
                        x_index,
                        clbits_dict[ii]["y"],
                        layer_widths[node][0],
                        glob_data,
                        flow_op,
                    )
                    for ii in c_indxs
                ]

                # update index based on the value from plotting
                if flow_parent is None:
                    curr_x_index = glob_data["next_x_index"]
                l_width.append(layer_widths[node][0])
                node_data[node].x_index = x_index

                # Special case of default case with no ops in it, need to push end
                # of switch op one extra x_index
                if isinstance(node.op, SwitchCaseOp):
                    if len(node.op.blocks[-1]) == 0:
                        curr_x_index += 1

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self._plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = (
                    -1 if all(getattr(nd.op, "_directive", False) for nd in layer) else 0
                )
            max_lwidth = max(l_width) if l_width else 0
            prev_x_index = curr_x_index + max_lwidth + barrier_offset - 1

        return prev_x_index + 1

    def _get_text_width(self, text, glob_data, fontsize, param=False, reg_remove_under=None):
        """Compute the width of a string in the default font"""

        from pylatexenc.latex2text import LatexNodes2Text

        if not text:
            return 0.0

        math_mode_match = self._mathmode_regex.search(text)
        num_underscores = 0
        num_carets = 0
        if math_mode_match:
            math_mode_text = math_mode_match.group(1)
            num_underscores = math_mode_text.count("_")
            num_carets = math_mode_text.count("^")
        text = LatexNodes2Text().latex_to_text(text.replace("$$", ""))

        # If there are subscripts or superscripts in mathtext string
        # we need to account for that spacing by manually removing
        # from text string for text length

        # if it's a register and there's a subscript at the end,
        # remove 1 underscore, otherwise don't remove any
        if reg_remove_under is not None:
            num_underscores = reg_remove_under
        if num_underscores:
            text = text.replace("_", "", num_underscores)
        if num_carets:
            text = text.replace("^", "", num_carets)

        # This changes hyphen to + to match width of math mode minus sign.
        if param:
            text = text.replace("-", "+")

        f = 0 if fontsize == self._style["fs"] else 1
        sum_text = 0.0
        for c in text:
            try:
                sum_text += self._char_list[c][f]
            except KeyError:
                # if non-ASCII char, use width of 'c', an average size
                sum_text += self._char_list["c"][f]
        if f == 1:
            sum_text *= glob_data["subfont_factor"]
        return sum_text

    def _draw_regs_wires(self, num_folds, xmax, max_x_index, qubits_dict, clbits_dict, glob_data):
        """Draw the register names and numbers, wires, and vertical lines at the ends"""

        for fold_num in range(num_folds + 1):
            # quantum registers
            for qubit in qubits_dict.values():
                qubit_label = qubit["wire_label"]
                y = qubit["y"] - fold_num * (glob_data["n_lines"] + 1)
                self._ax.text(
                    glob_data["x_offset"] - 0.2,
                    y,
                    qubit_label,
                    ha="right",
                    va="center",
                    fontsize=1.25 * self._style["fs"],
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                # draw the qubit wire
                self._line([glob_data["x_offset"], y], [xmax, y], zorder=PORDER_REGLINE)

            # classical registers
            this_clbit_dict = {}
            for clbit in clbits_dict.values():
                y = clbit["y"] - fold_num * (glob_data["n_lines"] + 1)
                if y not in this_clbit_dict:
                    this_clbit_dict[y] = {
                        "val": 1,
                        "wire_label": clbit["wire_label"],
                        "register": clbit["register"],
                    }
                else:
                    this_clbit_dict[y]["val"] += 1

            for y, this_clbit in this_clbit_dict.items():
                # cregbundle
                if self._cregbundle and this_clbit["register"] is not None:
                    self._ax.plot(
                        [glob_data["x_offset"] + 0.2, glob_data["x_offset"] + 0.3],
                        [y - 0.1, y + 0.1],
                        color=self._style["cc"],
                        zorder=PORDER_REGLINE,
                    )
                    self._ax.text(
                        glob_data["x_offset"] + 0.1,
                        y + 0.1,
                        str(this_clbit["register"].size),
                        ha="left",
                        va="bottom",
                        fontsize=0.8 * self._style["fs"],
                        color=self._style["tc"],
                        clip_on=True,
                        zorder=PORDER_TEXT,
                    )
                self._ax.text(
                    glob_data["x_offset"] - 0.2,
                    y,
                    this_clbit["wire_label"],
                    ha="right",
                    va="center",
                    fontsize=1.25 * self._style["fs"],
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
                # draw the clbit wire
                self._line(
                    [glob_data["x_offset"], y],
                    [xmax, y],
                    lc=self._style["cc"],
                    ls=self._style["cline"],
                    zorder=PORDER_REGLINE,
                )

            # lf vertical line at either end
            feedline_r = num_folds > 0 and num_folds > fold_num
            feedline_l = fold_num > 0
            if feedline_l or feedline_r:
                xpos_l = glob_data["x_offset"] - 0.01
                xpos_r = self._fold + glob_data["x_offset"] + 0.1
                ypos1 = -fold_num * (glob_data["n_lines"] + 1)
                ypos2 = -(fold_num + 1) * (glob_data["n_lines"]) - fold_num + 1
                if feedline_l:
                    self._ax.plot(
                        [xpos_l, xpos_l],
                        [ypos1, ypos2],
                        color=self._style["lc"],
                        linewidth=self._lwidth15,
                        zorder=PORDER_REGLINE,
                    )
                if feedline_r:
                    self._ax.plot(
                        [xpos_r, xpos_r],
                        [ypos1, ypos2],
                        color=self._style["lc"],
                        linewidth=self._lwidth15,
                        zorder=PORDER_REGLINE,
                    )
            # Mask off any lines or boxes in the bit label area to clean up
            # from folding for ControlFlow and other wrapping gates
            box = glob_data["patches_mod"].Rectangle(
                xy=(glob_data["x_offset"] - 0.1, -fold_num * (glob_data["n_lines"] + 1) + 0.5),
                width=-25.0,
                height=-(fold_num + 1) * (glob_data["n_lines"] + 1),
                fc=self._style["bg"],
                ec=self._style["bg"],
                linewidth=self._lwidth15,
                zorder=PORDER_MASK,
            )
            self._ax.add_patch(box)

        # draw index number
        if self._style["index"]:
            for layer_num in range(max_x_index):
                if self._fold > 0:
                    x_coord = layer_num % self._fold + glob_data["x_offset"] + 0.53
                    y_coord = -(layer_num // self._fold) * (glob_data["n_lines"] + 1) + 0.65
                else:
                    x_coord = layer_num + glob_data["x_offset"] + 0.53
                    y_coord = 0.65
                self._ax.text(
                    x_coord,
                    y_coord,
                    str(layer_num + 1),
                    ha="center",
                    va="center",
                    fontsize=self._style["sfs"],
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )

    def _add_nodes_and_coords(
        self,
        nodes,
        node_data,
        wire_map,
        outer_circuit,
        layer_widths,
        qubits_dict,
        clbits_dict,
        glob_data,
    ):
        """Add the nodes from ControlFlowOps and their coordinates to the main circuit"""
        for flow_drawers in self._flow_drawers.values():
            for flow_drawer in flow_drawers:
                nodes += flow_drawer._nodes
                flow_drawer._get_coords(
                    node_data,
                    flow_drawer._flow_wire_map,
                    outer_circuit,
                    layer_widths,
                    qubits_dict,
                    clbits_dict,
                    glob_data,
                    flow_parent=flow_drawer._flow_parent,
                )
                # Recurse for ControlFlowOps inside the flow_drawer
                flow_drawer._add_nodes_and_coords(
                    nodes,
                    node_data,
                    wire_map,
                    outer_circuit,
                    layer_widths,
                    qubits_dict,
                    clbits_dict,
                    glob_data,
                )

    def _draw_ops(
        self,
        nodes,
        node_data,
        wire_map,
        outer_circuit,
        layer_widths,
        qubits_dict,
        clbits_dict,
        glob_data,
    ):
        """Draw the gates in the circuit"""

        # Add the nodes from all the ControlFlowOps and their coordinates to the main nodes
        self._add_nodes_and_coords(
            nodes,
            node_data,
            wire_map,
            outer_circuit,
            layer_widths,
            qubits_dict,
            clbits_dict,
            glob_data,
        )
        prev_x_index = -1
        for layer in nodes:
            l_width = []
            curr_x_index = prev_x_index + 1

            # draw the gates in this layer
            for node in layer:
                op = node.op

                self._get_colors(node, node_data)

                # add conditional
                if getattr(op, "condition", None) or isinstance(op, SwitchCaseOp):
                    cond_xy = [
                        self._plot_coord(
                            node_data[node].x_index,
                            clbits_dict[ii]["y"],
                            layer_widths[node][0],
                            glob_data,
                            isinstance(op, ControlFlowOp),
                        )
                        for ii in clbits_dict
                    ]
                    self._condition(node, node_data, wire_map, outer_circuit, cond_xy, glob_data)

                # AnnotatedOperation with ControlModifier
                mod_control = None
                if getattr(op, "modifiers", None):
                    canonical_modifiers = _canonicalize_modifiers(op.modifiers)
                    for modifier in canonical_modifiers:
                        if isinstance(modifier, ControlModifier):
                            mod_control = modifier
                            break

                # draw measure
                if isinstance(op, Measure):
                    self._measure(node, node_data, outer_circuit, glob_data)

                # draw barriers, snapshots, etc.
                elif getattr(op, "_directive", False):
                    if self._plot_barriers:
                        self._barrier(node, node_data, glob_data)

                # draw the box for control flow circuits
                elif isinstance(op, ControlFlowOp):
                    self._flow_op_gate(node, node_data, glob_data)

                # draw single qubit gates
                elif len(node_data[node].q_xy) == 1 and not node.cargs:
                    self._gate(node, node_data, glob_data)

                # draw controlled gates
                elif isinstance(op, ControlledGate) or mod_control:
                    self._control_gate(node, node_data, glob_data, mod_control)

                # draw multi-qubit gate as final default
                else:
                    self._multiqubit_gate(node, node_data, glob_data)

                # Determine the max width of the circuit only at the top level
                if not node_data[node].inside_flow:
                    l_width.append(layer_widths[node][0])

            # adjust the column if there have been barriers encountered, but not plotted
            barrier_offset = 0
            if not self._plot_barriers:
                # only adjust if everything in the layer wasn't plotted
                barrier_offset = (
                    -1 if all(getattr(nd.op, "_directive", False) for nd in layer) else 0
                )
            prev_x_index = curr_x_index + (max(l_width) if l_width else 0) + barrier_offset - 1

    def _get_colors(self, node, node_data):
        """Get all the colors needed for drawing the circuit"""

        op = node.op
        base_name = getattr(getattr(op, "base_gate", None), "name", None)
        color = None
        if node_data[node].raw_gate_text in self._style["dispcol"]:
            color = self._style["dispcol"][node_data[node].raw_gate_text]
        elif op.name in self._style["dispcol"]:
            color = self._style["dispcol"][op.name]
        if color is not None:
            # Backward compatibility for style dict using 'displaycolor' with
            # gate color and no text color, so test for str first
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        # Treat special case of classical gates in iqx style by making all
        # controlled gates of x, dcx, and swap the classical gate color
        elif self._style["name"] in ["iqp", "iqx", "iqp-dark", "iqx-dark"] and base_name in [
            "x",
            "dcx",
            "swap",
        ]:
            color = self._style["dispcol"][base_name]
            if isinstance(color, str):
                fc = color
                gt = self._style["gt"]
            else:
                fc = color[0]
                gt = color[1]
        else:
            fc = self._style["gc"]
            gt = self._style["gt"]

        if self._style["name"] == "bw":
            ec = self._style["ec"]
            lc = self._style["lc"]
        else:
            ec = fc
            lc = fc
        # Subtext needs to be same color as gate text
        sc = gt
        node_data[node].fc = fc
        node_data[node].ec = ec
        node_data[node].gt = gt
        node_data[node].tc = self._style["tc"]
        node_data[node].sc = sc
        node_data[node].lc = lc

    def _condition(self, node, node_data, wire_map, outer_circuit, cond_xy, glob_data):
        """Add a conditional to a gate"""

        # For SwitchCaseOp convert the target to a fully closed Clbit or register
        # in condition format
        if isinstance(node.op, SwitchCaseOp):
            if isinstance(node.op.target, expr.Expr):
                condition = node.op.target
            elif isinstance(node.op.target, Clbit):
                condition = (node.op.target, 1)
            else:
                condition = (node.op.target, 2 ** (node.op.target.size) - 1)
        else:
            condition = node.op.condition

        override_fc = False
        first_clbit = len(self._qubits)
        cond_pos = []

        if isinstance(condition, expr.Expr):
            # If fixing this, please update the docstrings of `QuantumCircuit.draw` and
            # `visualization.circuit_drawer` to remove warnings.

            condition_bits = condition_resources(condition).clbits
            label = "[expr]"
            override_fc = True
            registers = collections.defaultdict(list)
            for bit in condition_bits:
                registers[get_bit_register(outer_circuit, bit)].append(bit)
            # Registerless bits don't care whether cregbundle is set.
            cond_pos.extend(cond_xy[wire_map[bit] - first_clbit] for bit in registers.pop(None, ()))
            if self._cregbundle:
                cond_pos.extend(cond_xy[wire_map[register] - first_clbit] for register in registers)
            else:
                cond_pos.extend(
                    cond_xy[wire_map[bit] - first_clbit]
                    for bit in itertools.chain.from_iterable(registers.values())
                )
            val_bits = ["1"] * len(cond_pos)
        else:
            label, val_bits = get_condition_label_val(condition, self._circuit, self._cregbundle)
            cond_bit_reg = condition[0]
            cond_bit_val = int(condition[1])
            override_fc = (
                cond_bit_val != 0
                and self._cregbundle
                and isinstance(cond_bit_reg, ClassicalRegister)
            )

            # In the first case, multiple bits are indicated on the drawing. In all
            # other cases, only one bit is shown.
            if not self._cregbundle and isinstance(cond_bit_reg, ClassicalRegister):
                for idx in range(cond_bit_reg.size):
                    cond_pos.append(cond_xy[wire_map[cond_bit_reg[idx]] - first_clbit])

            # If it's a register bit and cregbundle, need to use the register to find the location
            elif self._cregbundle and isinstance(cond_bit_reg, Clbit):
                register = get_bit_register(outer_circuit, cond_bit_reg)
                if register is not None:
                    cond_pos.append(cond_xy[wire_map[register] - first_clbit])
                else:
                    cond_pos.append(cond_xy[wire_map[cond_bit_reg] - first_clbit])
            else:
                cond_pos.append(cond_xy[wire_map[cond_bit_reg] - first_clbit])

        xy_plot = []
        for val_bit, xy in zip(val_bits, cond_pos):
            fc = self._style["lc"] if override_fc or val_bit == "1" else self._style["bg"]
            box = glob_data["patches_mod"].Circle(
                xy=xy,
                radius=WID * 0.15,
                fc=fc,
                ec=self._style["lc"],
                linewidth=self._lwidth15,
                zorder=PORDER_GATE,
            )
            self._ax.add_patch(box)
            xy_plot.append(xy)

        if not xy_plot:
            # Expression that's only on new-style `expr.Var` nodes, and doesn't need any vertical
            # line drawing.
            return

        qubit_b = min(node_data[node].q_xy, key=lambda xy: xy[1])
        clbit_b = min(xy_plot, key=lambda xy: xy[1])

        # For IfElseOp, WhileLoopOp or SwitchCaseOp, place the condition line
        # near the left edge of the box
        if isinstance(node.op, (IfElseOp, WhileLoopOp, SwitchCaseOp)):
            qubit_b = (qubit_b[0], qubit_b[1] - (0.5 * HIG + 0.14))

        # display the label at the bottom of the lowest conditional and draw the double line
        xpos, ypos = clbit_b
        if isinstance(node.op, Measure):
            xpos += 0.3
        self._ax.text(
            xpos,
            ypos - 0.3 * HIG,
            label,
            ha="center",
            va="top",
            fontsize=self._style["sfs"],
            color=self._style["tc"],
            clip_on=True,
            zorder=PORDER_TEXT,
        )
        self._line(qubit_b, clbit_b, lc=self._style["cc"], ls=self._style["cline"])

    def _measure(self, node, node_data, outer_circuit, glob_data):
        """Draw the measure symbol and the line to the clbit"""
        qx, qy = node_data[node].q_xy[0]
        cx, cy = node_data[node].c_xy[0]
        register, _, reg_index = get_bit_reg_index(outer_circuit, node.cargs[0])

        # draw gate box
        self._gate(node, node_data, glob_data)

        # add measure symbol
        qy_adj1 = 0.15 if self._measure_arrows else 0.05
        arc = glob_data["patches_mod"].Arc(
            xy=(qx, qy - qy_adj1 * HIG),
            width=WID * 0.7,
            height=HIG * 0.7,
            theta1=0,
            theta2=180,
            fill=False,
            ec=node_data[node].gt,
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(arc)
        qy_adj2 = 0.2 if self._measure_arrows else 0.3
        self._ax.plot(
            [qx, qx + 0.35 * WID],
            [qy - qy_adj1 * HIG, qy + qy_adj2 * HIG],
            color=node_data[node].gt,
            linewidth=self._lwidth2,
            zorder=PORDER_GATE,
        )
        # If measure_arrows, draw the down arrow to the clbit
        if self._measure_arrows:
            self._line(
                node_data[node].q_xy[0],
                [cx, cy + 0.35 * WID],
                lc=self._style["cc"],
                ls=self._style["cline"],
            )
            arrowhead = glob_data["patches_mod"].Polygon(
                (
                    (cx - 0.20 * WID, cy + 0.35 * WID),
                    (cx + 0.20 * WID, cy + 0.35 * WID),
                    (cx, cy + 0.04),
                ),
                fc=self._style["cc"],
                ec=None,
            )
            self._ax.add_artist(arrowhead)
            # target
            if self._cregbundle and register is not None:
                self._ax.text(
                    cx + 0.25,
                    cy + 0.1,
                    str(reg_index),
                    ha="left",
                    va="bottom",
                    fontsize=0.8 * self._style["fs"],
                    color=self._style["tc"],
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
        else:
            # If not measure_arrows, write the reg_bit into the measure box
            if register is not None:
                label = f"{register.name}_{reg_index}"
            else:
                label = f"{reg_index}"
            self._ax.text(
                qx,
                qy - 0.42 * HIG,
                label,
                ha="center",
                va="bottom",
                fontsize=self._style["sfs"],
                color=self._style["tc"],
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _barrier(self, node, node_data, glob_data):
        """Draw a barrier"""
        for i, xy in enumerate(node_data[node].q_xy):
            xpos, ypos = xy
            # For the topmost barrier, reduce the rectangle if there's a label to allow for the text.
            if i == 0 and node.op.label is not None:
                ypos_adj = -0.35
            else:
                ypos_adj = 0.0
            self._ax.plot(
                [xpos, xpos],
                [ypos + 0.5 + ypos_adj, ypos - 0.5],
                linewidth=self._lwidth1,
                linestyle="dashed",
                color=self._style["lc"],
                zorder=PORDER_TEXT,
            )
            box = glob_data["patches_mod"].Rectangle(
                xy=(xpos - (0.3 * WID), ypos - 0.5),
                width=0.6 * WID,
                height=1.0 + ypos_adj,
                fc=self._style["bc"],
                ec=None,
                alpha=0.6,
                linewidth=self._lwidth15,
                zorder=PORDER_BARRIER,
            )
            self._ax.add_patch(box)

            # display the barrier label at the top if there is one
            if i == 0 and node.op.label is not None:
                label = node.op.label
                if len(label) > self._barrier_label_len:
                    label = label[: self._barrier_label_len] + "..."
                dir_ypos = ypos + 0.65 * HIG
                self._ax.text(
                    xpos,
                    dir_ypos,
                    label,
                    ha="center",
                    va="top",
                    fontsize=self._style["fs"],
                    color=node_data[node].tc,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )

    def _gate(self, node, node_data, glob_data, xy=None):
        """Draw a 1-qubit gate"""
        if xy is None:
            xy = node_data[node].q_xy[0]
        xpos, ypos = xy
        wid = max(node_data[node].width, WID)

        box = glob_data["patches_mod"].Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=HIG,
            fc=node_data[node].fc,
            ec=node_data[node].ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        if node_data[node].gate_text:
            gate_ypos = ypos
            if node_data[node].param_text:
                gate_ypos = ypos + 0.15 * HIG
                self._ax.text(
                    xpos,
                    ypos - 0.3 * HIG,
                    node_data[node].param_text,
                    ha="center",
                    va="center",
                    fontsize=self._style["sfs"],
                    color=node_data[node].sc,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            self._ax.text(
                xpos,
                gate_ypos,
                node_data[node].gate_text,
                ha="center",
                va="center",
                fontsize=self._style["fs"],
                color=node_data[node].gt,
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _multiqubit_gate(self, node, node_data, glob_data, xy=None):
        """Draw a gate covering more than one qubit"""
        op = node.op
        if xy is None:
            xy = node_data[node].q_xy

        # Swap gate
        if isinstance(op, SwapGate):
            self._swap(xy, node_data[node].lc)
            return

        # RZZ Gate
        elif isinstance(op, RZZGate):
            self._symmetric_gate(node, node_data, RZZGate, glob_data)
            return

        c_xy = node_data[node].c_xy
        xpos = min(x[0] for x in xy)
        ypos = min(y[1] for y in xy)
        ypos_max = max(y[1] for y in xy)
        if c_xy:
            cxpos = min(x[0] for x in c_xy)
            cypos = min(y[1] for y in c_xy)
            ypos = min(ypos, cypos)

        wid = max(node_data[node].width + 0.21, WID)
        qubit_span = abs(ypos) - abs(ypos_max)
        height = HIG + qubit_span

        box = glob_data["patches_mod"].Rectangle(
            xy=(xpos - 0.5 * wid, ypos - 0.5 * HIG),
            width=wid,
            height=height,
            fc=node_data[node].fc,
            ec=node_data[node].ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # annotate inputs
        for bit, y in enumerate([x[1] for x in xy]):
            self._ax.text(
                xpos + 0.07 - 0.5 * wid,
                y,
                str(bit),
                ha="left",
                va="center",
                fontsize=self._style["fs"],
                color=node_data[node].gt,
                clip_on=True,
                zorder=PORDER_TEXT,
            )
        if c_xy:
            # annotate classical inputs
            for bit, y in enumerate([x[1] for x in c_xy]):
                self._ax.text(
                    cxpos + 0.07 - 0.5 * wid,
                    y,
                    str(bit),
                    ha="left",
                    va="center",
                    fontsize=self._style["fs"],
                    color=node_data[node].gt,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
        if node_data[node].gate_text:
            gate_ypos = ypos + 0.5 * qubit_span
            if node_data[node].param_text:
                gate_ypos = ypos + 0.4 * height
                self._ax.text(
                    xpos + 0.11,
                    ypos + 0.2 * height,
                    node_data[node].param_text,
                    ha="center",
                    va="center",
                    fontsize=self._style["sfs"],
                    color=node_data[node].sc,
                    clip_on=True,
                    zorder=PORDER_TEXT,
                )
            self._ax.text(
                xpos + 0.11,
                gate_ypos,
                node_data[node].gate_text,
                ha="center",
                va="center",
                fontsize=self._style["fs"],
                color=node_data[node].gt,
                clip_on=True,
                zorder=PORDER_TEXT,
            )

    def _flow_op_gate(self, node, node_data, glob_data):
        """Draw the box for a flow op circuit"""
        xy = node_data[node].q_xy
        xpos = min(x[0] for x in xy)
        ypos = min(y[1] for y in xy)
        ypos_max = max(y[1] for y in xy)

        # If a BoxOp, bring the right side back tight against the gates to allow for
        # better spacing
        if_width = node_data[node].width[0] + (WID if not isinstance(node.op, BoxOp) else -0.19)
        box_width = if_width
        # Add the else and case widths to the if_width
        for ewidth in node_data[node].width[1:]:
            if ewidth > 0.0:
                box_width += ewidth + WID + 0.3

        qubit_span = abs(ypos) - abs(ypos_max)
        height = HIG + qubit_span

        # Cycle through box colors based on depth.
        # Default - blue, purple, green, black
        colors = [
            self._style["dispcol"]["h"][0],
            self._style["dispcol"]["u"][0],
            self._style["dispcol"]["x"][0],
            self._style["cc"],
        ]
        # To fold box onto next lines, draw it repeatedly, shifting
        # it left by x_shift and down by y_shift
        fold_level = 0
        end_x = xpos + box_width

        while end_x > 0.0:
            x_shift = fold_level * self._fold
            y_shift = fold_level * (glob_data["n_lines"] + 1)
            end_x = xpos + box_width - x_shift if self._fold > 0 else 0.0

            if isinstance(node.op, IfElseOp):
                flow_text = "  If"
            elif isinstance(node.op, WhileLoopOp):
                flow_text = " While"
            elif isinstance(node.op, ForLoopOp):
                flow_text = " For"
            elif isinstance(node.op, SwitchCaseOp):
                flow_text = "Switch"
            elif isinstance(node.op, BoxOp):
                flow_text = ""
            else:
                raise RuntimeError(f"unhandled control-flow op: {node.name}")

            # Some spacers. op_spacer moves 'Switch' back a bit for alignment,
            # expr_spacer moves the expr over to line up with 'Switch' and
            # empty_default_spacer makes the switch box longer if the default
            # case is empty so text doesn't run past end of box.
            if isinstance(node.op, SwitchCaseOp):
                op_spacer = 0.04
                expr_spacer = 0.0
                empty_default_spacer = 0.3 if len(node.op.blocks[-1]) == 0 else 0.0
            elif isinstance(node.op, BoxOp):
                # Move the X start position back for a BoxOp, since there is no
                # leading text. This tightens the BoxOp with other ops.
                xpos -= 0.15
                op_spacer = 0.0
                expr_spacer = 0.0
                empty_default_spacer = 0.0
            else:
                op_spacer = 0.08
                expr_spacer = 0.02
                empty_default_spacer = 0.0

            # FancyBbox allows rounded corners
            box = glob_data["patches_mod"].FancyBboxPatch(
                xy=(xpos - x_shift, ypos - 0.5 * HIG - y_shift),
                width=box_width + empty_default_spacer,
                height=height,
                boxstyle="round, pad=0.1",
                fc="none",
                ec=colors[node_data[node].nest_depth % 4],
                linewidth=self._lwidth3,
                zorder=PORDER_FLOW,
            )
            self._ax.add_patch(box)

            # Indicate type of ControlFlowOp and if expression used, print below
            self._ax.text(
                xpos - x_shift - op_spacer,
                ypos_max + 0.2 - y_shift,
                flow_text,
                ha="left",
                va="center",
                fontsize=self._style["fs"],
                color=node_data[node].tc,
                clip_on=True,
                zorder=PORDER_FLOW,
            )
            self._ax.text(
                xpos - x_shift + expr_spacer,
                ypos_max + 0.2 - y_shift - 0.4,
                node_data[node].expr_text,
                ha="left",
                va="center",
                fontsize=self._style["sfs"],
                color=node_data[node].tc,
                clip_on=True,
                zorder=PORDER_FLOW,
            )
            if isinstance(node.op, ForLoopOp):
                idx_set = str(node_data[node].indexset)
                # If a range was used display 'range' and grab the range value
                # to be displayed below
                if "range" in idx_set:
                    idx_set = "r(" + idx_set[6:-1] + ")"
                else:
                    # If a tuple, show first 4 elements followed by '...'
                    idx_set = str(node_data[node].indexset)[1:-1].split(",")[:5]
                    if len(idx_set) > 4:
                        idx_set[4] = "..."
                    idx_set = f"{','.join(idx_set)}"
                y_spacer = 0.2 if len(node.qargs) == 1 else 0.5
                self._ax.text(
                    xpos - x_shift - 0.04,
                    ypos_max - y_spacer - y_shift,
                    idx_set,
                    ha="left",
                    va="center",
                    fontsize=self._style["sfs"],
                    color=node_data[node].tc,
                    clip_on=True,
                    zorder=PORDER_FLOW,
                )
            # If there's an else or a case draw the vertical line and the name
            else_case_text = "Else" if isinstance(node.op, IfElseOp) else "Case"
            ewidth_incr = if_width
            for circ_num, ewidth in enumerate(node_data[node].width[1:]):
                if ewidth > 0.0:
                    self._ax.plot(
                        [xpos + ewidth_incr + 0.3 - x_shift, xpos + ewidth_incr + 0.3 - x_shift],
                        [ypos - 0.5 * HIG - 0.08 - y_shift, ypos + height - 0.22 - y_shift],
                        color=colors[node_data[node].nest_depth % 4],
                        linewidth=3.0,
                        linestyle="solid",
                        zorder=PORDER_FLOW,
                    )
                    self._ax.text(
                        xpos + ewidth_incr + 0.4 - x_shift,
                        ypos_max + 0.2 - y_shift,
                        else_case_text,
                        ha="left",
                        va="center",
                        fontsize=self._style["fs"],
                        color=node_data[node].tc,
                        clip_on=True,
                        zorder=PORDER_FLOW,
                    )
                    if isinstance(node.op, SwitchCaseOp):
                        jump_val = node_data[node].jump_values[circ_num]
                        # If only one value, e.g. (0,)
                        if len(str(jump_val)) == 4:
                            jump_text = str(jump_val)[1]
                        elif "default" in str(jump_val):
                            jump_text = "default"
                        else:
                            # If a tuple, show first 4 elements followed by '...'
                            jump_text = str(jump_val)[1:-1].replace(" ", "").split(",")[:5]
                            if len(jump_text) > 4:
                                jump_text[4] = "..."
                            jump_text = f"{', '.join(jump_text)}"
                        y_spacer = 0.2 if len(node.qargs) == 1 else 0.5
                        self._ax.text(
                            xpos + ewidth_incr + 0.4 - x_shift,
                            ypos_max - y_spacer - y_shift,
                            jump_text,
                            ha="left",
                            va="center",
                            fontsize=self._style["sfs"],
                            color=node_data[node].tc,
                            clip_on=True,
                            zorder=PORDER_FLOW,
                        )
                ewidth_incr += ewidth + 1

            fold_level += 1

    def _control_gate(self, node, node_data, glob_data, mod_control):
        """Draw a controlled gate"""
        op = node.op
        xy = node_data[node].q_xy
        base_type = getattr(op, "base_gate", None)
        qubit_b = min(xy, key=lambda xy: xy[1])
        qubit_t = max(xy, key=lambda xy: xy[1])
        num_ctrl_qubits = mod_control.num_ctrl_qubits if mod_control else op.num_ctrl_qubits
        num_qargs = len(xy) - num_ctrl_qubits
        ctrl_state = mod_control.ctrl_state if mod_control else op.ctrl_state
        self._set_ctrl_bits(
            ctrl_state,
            num_ctrl_qubits,
            xy,
            glob_data,
            ec=node_data[node].ec,
            tc=node_data[node].tc,
            text=node_data[node].ctrl_text,
            qargs=node.qargs,
        )
        self._line(qubit_b, qubit_t, lc=node_data[node].lc)

        if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, ZGate, RZZGate)):
            self._symmetric_gate(node, node_data, base_type, glob_data)

        elif num_qargs == 1 and isinstance(base_type, XGate):
            tgt_color = self._style["dispcol"]["target"]
            tgt = tgt_color if isinstance(tgt_color, str) else tgt_color[0]
            self._x_tgt_qubit(xy[num_ctrl_qubits], glob_data, ec=node_data[node].ec, ac=tgt)

        elif num_qargs == 1:
            self._gate(node, node_data, glob_data, xy[num_ctrl_qubits:][0])

        elif isinstance(base_type, SwapGate):
            self._swap(xy[num_ctrl_qubits:], node_data[node].lc)

        else:
            self._multiqubit_gate(node, node_data, glob_data, xy[num_ctrl_qubits:])

    def _set_ctrl_bits(
        self, ctrl_state, num_ctrl_qubits, qbit, glob_data, ec=None, tc=None, text="", qargs=None
    ):
        """Determine which qubits are controls and whether they are open or closed"""
        # place the control label at the top or bottom of controls
        if text:
            qlist = [self._circuit.find_bit(qubit).index for qubit in qargs]
            ctbits = qlist[:num_ctrl_qubits]
            qubits = qlist[num_ctrl_qubits:]
            max_ctbit = max(ctbits)
            min_ctbit = min(ctbits)
            top = min(qubits) > min_ctbit

        # display the control qubits as open or closed based on ctrl_state
        cstate = f"{ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]
        for i in range(num_ctrl_qubits):
            fc_open_close = ec if cstate[i] == "1" else self._style["bg"]
            text_top = None
            if text:
                if top and qlist[i] == min_ctbit:
                    text_top = True
                elif not top and qlist[i] == max_ctbit:
                    text_top = False
            self._ctrl_qubit(
                qbit[i], glob_data, fc=fc_open_close, ec=ec, tc=tc, text=text, text_top=text_top
            )

    def _ctrl_qubit(self, xy, glob_data, fc=None, ec=None, tc=None, text="", text_top=None):
        """Draw a control circle and if top or bottom control, draw control label"""
        xpos, ypos = xy
        box = glob_data["patches_mod"].Circle(
            xy=(xpos, ypos),
            radius=WID * 0.15,
            fc=fc,
            ec=ec,
            linewidth=self._lwidth15,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # adjust label height according to number of lines of text
        label_padding = 0.7
        if text is not None:
            text_lines = text.count("\n")
            if not text.endswith("(cal)\n"):
                for _ in range(text_lines):
                    label_padding += 0.3

        if text_top is None:
            return

        # display the control label at the top or bottom if there is one
        ctrl_ypos = ypos + label_padding * HIG if text_top else ypos - 0.3 * HIG
        self._ax.text(
            xpos,
            ctrl_ypos,
            text,
            ha="center",
            va="top",
            fontsize=self._style["sfs"],
            color=tc,
            clip_on=True,
            zorder=PORDER_TEXT,
        )

    def _x_tgt_qubit(self, xy, glob_data, ec=None, ac=None):
        """Draw the cnot target symbol"""
        linewidth = self._lwidth2
        xpos, ypos = xy
        box = glob_data["patches_mod"].Circle(
            xy=(xpos, ypos),
            radius=HIG * 0.35,
            fc=ec,
            ec=ec,
            linewidth=linewidth,
            zorder=PORDER_GATE,
        )
        self._ax.add_patch(box)

        # add '+' symbol
        self._ax.plot(
            [xpos, xpos],
            [ypos - 0.2 * HIG, ypos + 0.2 * HIG],
            color=ac,
            linewidth=linewidth,
            zorder=PORDER_GATE_PLUS,
        )
        self._ax.plot(
            [xpos - 0.2 * HIG, xpos + 0.2 * HIG],
            [ypos, ypos],
            color=ac,
            linewidth=linewidth,
            zorder=PORDER_GATE_PLUS,
        )

    def _symmetric_gate(self, node, node_data, base_type, glob_data):
        """Draw symmetric gates for cz, cu1, cp, and rzz"""
        op = node.op
        xy = node_data[node].q_xy
        qubit_b = min(xy, key=lambda xy: xy[1])
        qubit_t = max(xy, key=lambda xy: xy[1])
        base_type = getattr(op, "base_gate", None)
        ec = node_data[node].ec
        tc = node_data[node].tc
        lc = node_data[node].lc

        # cz and mcz gates
        if not isinstance(op, ZGate) and isinstance(base_type, ZGate):
            num_ctrl_qubits = op.num_ctrl_qubits
            self._ctrl_qubit(xy[-1], glob_data, fc=ec, ec=ec, tc=tc)
            self._line(qubit_b, qubit_t, lc=lc, zorder=PORDER_LINE_PLUS)

        # cu1, cp, rzz, and controlled rzz gates (sidetext gates)
        elif isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
            num_ctrl_qubits = 0 if isinstance(op, RZZGate) else op.num_ctrl_qubits
            gate_text = "P" if isinstance(base_type, PhaseGate) else node_data[node].gate_text

            self._ctrl_qubit(xy[num_ctrl_qubits], glob_data, fc=ec, ec=ec, tc=tc)
            if not isinstance(base_type, (U1Gate, PhaseGate)):
                self._ctrl_qubit(xy[num_ctrl_qubits + 1], glob_data, fc=ec, ec=ec, tc=tc)

            self._sidetext(
                node,
                node_data,
                qubit_b,
                tc=tc,
                text=f"{gate_text} ({node_data[node].param_text})",
            )
            self._line(qubit_b, qubit_t, lc=lc)

    def _swap(self, xy, color=None):
        """Draw a Swap gate"""
        self._swap_cross(xy[0], color=color)
        self._swap_cross(xy[1], color=color)
        self._line(xy[0], xy[1], lc=color)

    def _swap_cross(self, xy, color=None):
        """Draw the Swap cross symbol"""
        xpos, ypos = xy

        self._ax.plot(
            [xpos - 0.20 * WID, xpos + 0.20 * WID],
            [ypos - 0.20 * WID, ypos + 0.20 * WID],
            color=color,
            linewidth=self._lwidth2,
            zorder=PORDER_LINE_PLUS,
        )
        self._ax.plot(
            [xpos - 0.20 * WID, xpos + 0.20 * WID],
            [ypos + 0.20 * WID, ypos - 0.20 * WID],
            color=color,
            linewidth=self._lwidth2,
            zorder=PORDER_LINE_PLUS,
        )

    def _sidetext(self, node, node_data, xy, tc=None, text=""):
        """Draw the sidetext for symmetric gates"""
        xpos, ypos = xy

        # 0.11 = the initial gap, add 1/2 text width to place on the right
        xp = xpos + 0.11 + node_data[node].width / 2
        self._ax.text(
            xp,
            ypos + HIG,
            text,
            ha="center",
            va="top",
            fontsize=self._style["sfs"],
            color=tc,
            clip_on=True,
            zorder=PORDER_TEXT,
        )

    def _line(self, xy0, xy1, lc=None, ls=None, zorder=PORDER_LINE):
        """Draw a line from xy0 to xy1"""
        x0, y0 = xy0
        x1, y1 = xy1
        linecolor = self._style["lc"] if lc is None else lc
        linestyle = "solid" if ls is None else ls

        if linestyle == "doublet":
            theta = np.arctan2(np.abs(x1 - x0), np.abs(y1 - y0))
            dx = 0.05 * WID * np.cos(theta)
            dy = 0.05 * WID * np.sin(theta)
            self._ax.plot(
                [x0 + dx, x1 + dx],
                [y0 + dy, y1 + dy],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle="solid",
                zorder=zorder,
            )
            self._ax.plot(
                [x0 - dx, x1 - dx],
                [y0 - dy, y1 - dy],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle="solid",
                zorder=zorder,
            )
        else:
            self._ax.plot(
                [x0, x1],
                [y0, y1],
                color=linecolor,
                linewidth=self._lwidth2,
                linestyle=linestyle,
                zorder=zorder,
            )

    def _plot_coord(self, x_index, y_index, gate_width, glob_data, flow_op=False):
        """Get the coord positions for an index"""

        # Check folding
        fold = self._fold if self._fold > 0 else INFINITE_FOLD
        h_pos = x_index % fold + 1

        # Don't fold flow_ops here, only gates inside the flow_op
        if not flow_op and h_pos + (gate_width - 1) > fold:
            x_index += fold - (h_pos - 1)
        x_pos = x_index % fold + glob_data["x_offset"] + 0.04
        if not flow_op:
            x_pos += 0.5 * gate_width
        else:
            x_pos += 0.25
        y_pos = y_index - (x_index // fold) * (glob_data["n_lines"] + 1)

        # x_index could have been updated, so need to store
        glob_data["next_x_index"] = x_index
        return x_pos, y_pos


class NodeData:
    """Class containing drawing data on a per node basis"""

    def __init__(self):
        # Node data for positioning
        self.width = 0.0
        self.x_index = 0
        self.q_xy = []
        self.c_xy = []

        # Node data for text
        self.gate_text = ""
        self.raw_gate_text = ""
        self.ctrl_text = ""
        self.param_text = ""

        # Node data for color
        self.fc = self.ec = self.lc = self.sc = self.gt = self.tc = 0

        # Special values stored for ControlFlowOps
        self.nest_depth = 0
        self.expr_width = 0.0
        self.expr_text = ""
        self.inside_flow = False
        self.indexset = ()  # List of indices used for ForLoopOp
        self.jump_values = []  # List of jump values used for SwitchCaseOp
        self.circ_num = 0  # Which block is it in op.blocks
