# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""latex visualization backend."""

import io
import itertools
import math
import re
from warnings import warn

import numpy as np
from qiskit.circuit import Clbit, Qubit, ClassicalRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.library.standard_gates import SwapGate, XGate, ZGate, RZZGate, U1Gate, PhaseGate
from qiskit.circuit.measure import Measure
from qiskit.circuit.tools.pi_check import pi_check


from qiskit.visualization.style import load_style
from qiskit.visualization.circuit.qcstyle import MPLDefaultStyle, MPLStyleDict
from ._utils import (
    get_gate_ctrl_text,
    get_param_str,
    get_wire_map,
    get_bit_register,
    get_bit_reg_index,
    get_wire_label,
    generate_latex_label,
    get_condition_label_val,
)


class QCircuitImage:
    """This class contains methods to create \\LaTeX circuit images.

    The class targets the \\LaTeX package Q-circuit
    (https://arxiv.org/pdf/quant-ph/0406003).

    Thanks to Eric Sabo for the initial implementation for Qiskit.
    """

    def __init__(
        self,
        qubits,
        clbits,
        nodes,
        scale,
        style=None,
        reverse_bits=False,
        plot_barriers=True,
        initial_state=False,
        cregbundle=None,
        with_layout=False,
        circuit=None,
        barrier_label_len=16,
    ):
        """QCircuitImage initializer.

        Args:
            qubits (list[Qubit]): list of qubits
            clbits (list[Clbit]): list of clbits
            nodes (list[list[DAGNode]]): list of circuit instructions, grouped by layer
            scale (float): image scaling
            style (dict or str): dictionary of style or file name of style file
            reverse_bits (bool): when True, reverse the bit ordering of the registers
            plot_barriers (bool): Enable/disable drawing barriers in the output
               circuit. Defaults to True.
            initial_state (bool): Optional. Adds |0> in the beginning of the line. Default: `False`.
            cregbundle (bool): Optional. If set True bundle classical registers.
            with_layout (bool): Optional. If set to True display the layout in the circuit.
            circuit (QuantumCircuit): the circuit that's being displayed
            barrier_label_len (int): Optional. The number of characters to display for
               barrier labels. If this number is exceeded, the string will be truncated.
        Raises:
            ImportError: If pylatexenc is not installed
        """

        self._circuit = circuit
        self._qubits = qubits
        self._clbits = clbits
        self._barrier_label_len = barrier_label_len

        # list of lists corresponding to layers of the circuit
        self._nodes = nodes

        # image scaling
        self._scale = 1.0 if scale is None else scale

        # Map of cregs to sizes
        self._cregs = {}

        # Array to hold the \\LaTeX commands to generate a circuit image.
        self._latex = []

        # Variable to hold image depth (width)
        self._img_depth = 0

        # Variable to hold image width (height)
        self._img_width = 0

        # Variable to hold total circuit depth
        self._sum_column_widths = 0

        # Variable to hold total circuit width
        self._sum_wire_heights = 0

        # em points of separation between circuit columns
        self._column_separation = 1

        # em points of separation between circuit wire
        self._wire_separation = 0

        # presence of "box" or "target" determines wire spacing
        self._has_box = False
        self._has_target = False

        self._plot_barriers = plot_barriers
        self._reverse_bits = reverse_bits
        if with_layout:
            if self._circuit._layout:
                self._layout = self._circuit._layout.initial_layout
            else:
                self._layout = None
        else:
            self._layout = None

        self._initial_state = initial_state
        self._global_phase = circuit.global_phase

        # If there is any custom instruction that uses classical bits
        # then cregbundle is forced to be False.
        for node in itertools.chain.from_iterable(self._nodes):
            if node.cargs and node.op.name != "measure":
                if cregbundle:
                    warn(
                        "Cregbundle set to False since an instruction needs to refer"
                        " to individual classical wire",
                        RuntimeWarning,
                        2,
                    )
                self._cregbundle = False
                break
        else:
            self._cregbundle = True if cregbundle is None else cregbundle

        self._wire_map = get_wire_map(circuit, qubits + clbits, self._cregbundle)
        self._img_width = len(self._wire_map)

        self._style, _ = load_style(
            style,
            style_dict=MPLStyleDict,
            default_style=MPLDefaultStyle(),
            user_config_opt="circuit_mpl_style",
            user_config_path_opt="circuit_mpl_style_path",
        )

    def latex(self):
        """Return LaTeX string representation of circuit."""

        self._initialize_latex_array()
        self._build_latex_array()
        header_1 = r"\documentclass[border=2px]{standalone}" + "\n"

        header_2 = r"""
\usepackage[braket, qm]{qcircuit}
\usepackage{graphicx}

\begin{document}
"""
        header_scale = f"\\scalebox{{{self._scale}}}" + "{"

        qcircuit_line = r"""
\Qcircuit @C=%.1fem @R=%.1fem @!R { \\
"""
        output = io.StringIO()
        output.write(header_1)
        output.write(header_2)
        output.write(header_scale)
        if self._global_phase:
            output.write(
                f"{{$\\mathrm{{global\\,phase:\\,}} \\mathrm{{{pi_check(self._global_phase, output='latex')}}}$}}"
            )
        output.write(qcircuit_line % (self._column_separation, self._wire_separation))
        for i in range(self._img_width):
            output.write("\t \t")
            for j in range(self._img_depth + 1):
                output.write(self._latex[i][j])
                if j != self._img_depth:
                    output.write(" & ")
                else:
                    output.write(r"\\" + "\n")
        output.write(r"\\ " + "}}\n")
        output.write("\\end{document}")
        contents = output.getvalue()
        output.close()
        return contents

    def _initialize_latex_array(self):
        """Initialize qubit and clbit labels and set wire separation"""
        self._img_depth, self._sum_column_widths = self._get_image_depth()
        self._sum_wire_heights = self._img_width
        # choose the most compact wire spacing, while not squashing them
        if self._has_box:
            self._wire_separation = 0.2
        elif self._has_target:
            self._wire_separation = 0.8
        else:
            self._wire_separation = 1.0
        self._latex = [
            ["\\qw" if isinstance(wire, Qubit) else "\\cw" for _ in range(self._img_depth + 1)]
            for wire in self._wire_map
        ]
        self._latex.append([" "] * (self._img_depth + 1))

        # display the bit/register labels
        for wire, index in self._wire_map.items():
            if isinstance(wire, ClassicalRegister):
                register = wire
                wire_label = get_wire_label(
                    "latex", register, index, layout=self._layout, cregbundle=self._cregbundle
                )
            else:
                register, bit_index, reg_index = get_bit_reg_index(self._circuit, wire)
                wire_label = get_wire_label(
                    "latex",
                    register,
                    bit_index if register is None else reg_index,
                    layout=self._layout,
                    cregbundle=self._cregbundle,
                )

            wire_label += " : "
            if self._initial_state:
                wire_label += "\\ket{{0}}" if isinstance(wire, Qubit) else "0"
            wire_label += " }"

            if not isinstance(wire, (Qubit)) and self._cregbundle and register is not None:
                pos = self._wire_map[register]
                self._latex[pos][1] = "\\lstick{/_{_{" + str(register.size) + "}}} \\cw"
                wire_label = f"\\mathrm{{{wire_label}}}"
            else:
                pos = index
            self._latex[pos][0] = "\\nghost{" + wire_label + " & " + "\\lstick{" + wire_label

    def _get_image_depth(self):
        """Get depth information for the circuit."""

        # wires in the beginning and end
        columns = 2
        if self._cregbundle and (
            self._nodes
            and self._nodes[0]
            and (
                self._nodes[0][0].op.name == "measure"
                or getattr(self._nodes[0][0].op, "condition", None)
            )
        ):
            columns += 1

        # Determine wire spacing before image depth
        max_column_widths = []
        for layer in self._nodes:
            column_width = 1
            current_max = 0
            for node in layer:
                op = node.op
                # useful information for determining wire spacing
                boxed_gates = [
                    "u1",
                    "u2",
                    "u3",
                    "u",
                    "p",
                    "x",
                    "y",
                    "z",
                    "h",
                    "s",
                    "sdg",
                    "t",
                    "tdg",
                    "sx",
                    "sxdg",
                    "rx",
                    "ry",
                    "rz",
                    "ch",
                    "cy",
                    "crz",
                    "cu2",
                    "cu3",
                    "cu",
                    "id",
                ]
                target_gates = ["cx", "ccx", "cu1", "cp", "rzz"]
                if op.name in boxed_gates:
                    self._has_box = True
                elif op.name in target_gates:
                    self._has_target = True
                elif isinstance(op, ControlledGate):
                    self._has_box = True

                arg_str_len = 0
                # the wide gates
                for arg in op.params:
                    if not any(isinstance(param, np.ndarray) for param in op.params):
                        arg_str = re.sub(r"[-+]?\d*\.\d{2,}|\d{2,}", self._truncate_float, str(arg))
                        arg_str_len += len(arg_str)

                # the width of the column is the max of all the gates in the column
                current_max = max(arg_str_len, current_max)

                # all gates take up 1 column except from those with side labels (ie cu1, cp, rzz)
                # which take 4 columns
                base_type = None if not hasattr(op, "base_gate") else op.base_gate
                if isinstance(op, RZZGate) or isinstance(base_type, (U1Gate, PhaseGate, RZZGate)):
                    column_width = 4
            max_column_widths.append(current_max)
            columns += column_width

        # every 3 characters is roughly one extra 'unit' of width in the cell
        # the gate name is 1 extra 'unit'
        # the qubit/cbit labels plus initial states is 2 more
        # the wires poking out at the ends is 2 more
        sum_column_widths = sum(1 + v / 3 for v in max_column_widths)

        max_wire_name = 3
        for wire in self._wire_map:
            if isinstance(wire, (Qubit, Clbit)):
                register = get_bit_register(self._circuit, wire)
                name = register.name if register is not None else ""
            else:
                name = wire.name
            max_wire_name = max(max_wire_name, len(name))

        sum_column_widths += 5 + max_wire_name / 3

        # could be a fraction so ceil
        return columns, math.ceil(sum_column_widths)

    def _get_beamer_page(self):
        """Get height, width & scale attributes for the beamer page."""

        # PIL python package limits image size to around a quarter gigabyte
        # this means the beamer image should be limited to < 50000
        # if you want to avoid a "warning" too, set it to < 25000
        pil_limit = 40000

        # the beamer latex template limits each dimension to < 19 feet
        # (i.e. 575cm)
        beamer_limit = 550

        # columns are roughly twice as big as wires
        aspect_ratio = self._sum_wire_heights / self._sum_column_widths

        # choose a page margin so circuit is not cropped
        margin_factor = 1.5
        height = min(self._sum_wire_heights * margin_factor, beamer_limit)
        width = min(self._sum_column_widths * margin_factor, beamer_limit)

        # if too large, make it fit
        if height * width > pil_limit:
            height = min(np.sqrt(pil_limit * aspect_ratio), beamer_limit)
            width = min(np.sqrt(pil_limit / aspect_ratio), beamer_limit)

        # if too small, give it a minimum size
        height = max(height, 10)
        width = max(width, 10)

        return (height, width, self._scale)

    def _build_latex_array(self):
        """Returns an array of strings containing \\LaTeX for this circuit."""

        column = 1
        # Leave a column to display number of classical registers if needed
        if self._cregbundle and (
            self._nodes
            and self._nodes[0]
            and (
                self._nodes[0][0].op.name == "measure"
                or getattr(self._nodes[0][0].op, "condition", None)
            )
        ):
            column += 1

        for layer in self._nodes:
            num_cols_layer = 1

            for node in layer:
                op = node.op
                num_cols_op = 1
                wire_list = [self._wire_map[qarg] for qarg in node.qargs if qarg in self._qubits]
                if getattr(op, "condition", None):
                    if isinstance(op.condition, expr.Expr):
                        warn("ignoring expression condition, which is not supported yet")
                    else:
                        self._add_condition(op, wire_list, column)

                if isinstance(op, Measure):
                    self._build_measure(node, column)

                elif getattr(op, "_directive", False):  # barrier, snapshot, etc.
                    self._build_barrier(node, column)

                else:
                    gate_text, _, _ = get_gate_ctrl_text(op, "latex", style=self._style)
                    gate_text += get_param_str(op, "latex", ndigits=4)
                    gate_text = generate_latex_label(gate_text)
                    if node.cargs:
                        cwire_list = [
                            self._wire_map[carg] for carg in node.cargs if carg in self._clbits
                        ]
                    else:
                        cwire_list = []

                    if len(wire_list) == 1 and not node.cargs:
                        self._latex[wire_list[0]][column] = f"\\gate{{{gate_text}}}"

                    elif isinstance(op, ControlledGate):
                        num_cols_op = self._build_ctrl_gate(op, gate_text, wire_list, column)
                    else:
                        num_cols_op = self._build_multi_gate(
                            op, gate_text, wire_list, cwire_list, column
                        )

                num_cols_layer = max(num_cols_layer, num_cols_op)

            column += num_cols_layer

    def _build_multi_gate(self, op, gate_text, wire_list, cwire_list, col):
        """Add a multiple wire gate to the _latex list"""
        cwire_start = len(self._qubits)
        num_cols_op = 1
        if isinstance(op, (SwapGate, RZZGate)):
            num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
        else:
            wire_min = min(wire_list)
            wire_max = max(wire_list)
            if cwire_list and not self._cregbundle:
                wire_max = max(cwire_list)
            wire_ind = wire_list.index(wire_min)
            self._latex[wire_min][col] = (
                f"\\multigate{{{wire_max - wire_min}}}{{{gate_text}}}_"
                + "<" * (len(str(wire_ind)) + 2)
                + f"{{{wire_ind}}}"
            )
            for wire in range(wire_min + 1, wire_max + 1):
                if wire < cwire_start:
                    ghost_box = f"\\ghost{{{gate_text}}}"
                    if wire in wire_list:
                        wire_ind = wire_list.index(wire)
                else:
                    ghost_box = f"\\cghost{{{gate_text}}}"
                    if wire in cwire_list:
                        wire_ind = cwire_list.index(wire)
                if wire in wire_list + cwire_list:
                    self._latex[wire][col] = (
                        ghost_box + "_" + "<" * (len(str(wire_ind)) + 2) + f"{{{wire_ind}}}"
                    )
                else:
                    self._latex[wire][col] = ghost_box
        return num_cols_op

    def _build_ctrl_gate(self, op, gate_text, wire_list, col):
        """Add a gate with multiple controls to the _latex list"""
        num_cols_op = 1
        num_ctrl_qubits = op.num_ctrl_qubits
        wireqargs = wire_list[num_ctrl_qubits:]
        ctrlqargs = wire_list[:num_ctrl_qubits]
        wire_min = min(wireqargs)
        wire_max = max(wireqargs)
        ctrl_state = f"{op.ctrl_state:b}".rjust(num_ctrl_qubits, "0")[::-1]

        # First do single qubit target gates
        if len(wireqargs) == 1:
            self._add_controls(wire_list, ctrlqargs, ctrl_state, col)

            # Check for cx, cz, cu1 and cp first, then do standard gate
            if isinstance(op.base_gate, XGate):
                self._latex[wireqargs[0]][col] = "\\targ"
            elif isinstance(op.base_gate, ZGate):
                self._latex[wireqargs[0]][col] = "\\control\\qw"
            elif isinstance(op.base_gate, (U1Gate, PhaseGate)):
                num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
            else:
                self._latex[wireqargs[0]][col] = f"\\gate{{{gate_text}}}"
        # Treat special cases of swap and rzz gates
        elif isinstance(op.base_gate, (SwapGate, RZZGate)):
            self._add_controls(wire_list, ctrlqargs, ctrl_state, col)
            num_cols_op = self._build_symmetric_gate(op, gate_text, wire_list, col)
        else:
            # If any controls appear in the span of the multiqubit
            # gate just treat the whole thing as a big gate
            for ctrl in ctrlqargs:
                if ctrl in range(wire_min, wire_max):
                    wireqargs = wire_list
                    break
            else:
                self._add_controls(wire_list, ctrlqargs, ctrl_state, col)

            self._build_multi_gate(op, gate_text, wireqargs, [], col)
        return num_cols_op

    def _build_symmetric_gate(self, op, gate_text, wire_list, col):
        """Add symmetric gates for cu1, cp, swap, and rzz"""
        wire_max = max(wire_list)
        # The last and next to last in the wire list are the gate wires without added controls
        wire_next_last = wire_list[-2]
        wire_last = wire_list[-1]
        base_op = None if not hasattr(op, "base_gate") else op.base_gate

        if isinstance(op, SwapGate) or (base_op and isinstance(base_op, SwapGate)):
            self._latex[wire_next_last][col] = "\\qswap"
            self._latex[wire_last][col] = "\\qswap \\qwx[" + str(wire_next_last - wire_last) + "]"
            return 1  # num_cols

        if isinstance(op, RZZGate) or (base_op and isinstance(base_op, RZZGate)):
            ctrl_bit = "1"
        else:
            ctrl_bit = f"{op.ctrl_state:b}".rjust(1, "0")[::-1]

        control = "\\ctrlo" if ctrl_bit == "0" else "\\ctrl"
        self._latex[wire_next_last][col] = f"{control}" + (
            "{" + str(wire_last - wire_next_last) + "}"
        )
        self._latex[wire_last][col] = "\\control \\qw"
        # Put side text to the right between bottom wire in wire_list and the one above it
        self._latex[wire_max - 1][col + 1] = f"\\dstick{{\\hspace{{2.0em}}{gate_text}}} \\qw"
        return 4  # num_cols for side text gates

    def _build_measure(self, node, col):
        """Build a meter and the lines to the creg"""
        wire1 = self._wire_map[node.qargs[0]]
        self._latex[wire1][col] = "\\meter"

        idx_str = ""
        cond_offset = 1.5 if getattr(node.op, "condition", None) else 0.0
        if self._cregbundle:
            register = get_bit_register(self._circuit, node.cargs[0])
            if register is not None:
                wire2 = self._wire_map[register]
                idx_str = str(self._circuit.find_bit(node.cargs[0]).registers[0][1])
            else:
                wire2 = self._wire_map[node.cargs[0]]
            self._latex[wire2][col] = (
                f"\\dstick{{_{{_{{\\hspace{{{cond_offset}em}}{idx_str}}}}}}} "
                f"\\cw \\ar @{{<=}} [-{wire2 - wire1!s},0]"
            )
        else:
            wire2 = self._wire_map[node.cargs[0]]
            self._latex[wire2][col] = "\\cw \\ar @{<=} [-" + str(wire2 - wire1) + ",0]"

    def _build_barrier(self, node, col):
        """Build a partial or full barrier if plot_barriers set"""
        if self._plot_barriers:
            indexes = [self._wire_map[qarg] for qarg in node.qargs if qarg in self._qubits]
            indexes.sort()
            first = last = indexes[0]
            for index in indexes[1:]:
                if index - 1 == last:
                    last = index
                else:
                    pos = self._wire_map[self._qubits[first]]
                    self._latex[pos][col - 1] += " \\barrier[0em]{" + str(last - first) + "}"
                    self._latex[pos][col] = "\\qw"
                    first = last = index
            pos = self._wire_map[self._qubits[first]]
            self._latex[pos][col - 1] += " \\barrier[0em]{" + str(last - first) + "}"
            if node.op.label is not None:
                pos = indexes[0]
                label = node.op.label
                if len(label) > self._barrier_label_len:
                    label = label[: self._barrier_label_len] + "..."
                label = label.replace(" ", "\\,")  # \, is a LaTeX thin space
                self._latex[pos][col] = f"\\cds{{0}}{{^{{\\mathrm{{{label}}}}}}}"

    def _add_controls(self, wire_list, ctrlqargs, ctrl_state, col):
        """Add one or more controls to a gate"""
        for index, ctrl_item in enumerate(zip(ctrlqargs, ctrl_state)):
            pos = ctrl_item[0]
            nxt = wire_list[index]
            if wire_list[index] > wire_list[-1]:
                nxt -= 1
                while nxt not in wire_list:
                    nxt -= 1
            else:
                nxt += 1
                while nxt not in wire_list:
                    nxt += 1

            # ctrl_item[1] is ctrl_state for this bit
            control = "\\ctrlo" if ctrl_item[1] == "0" else "\\ctrl"
            self._latex[pos][col] = f"{control}" + "{" + str(nxt - wire_list[index]) + "}"

    def _add_condition(self, op, wire_list, col):
        """Add a condition to the _latex list"""
        # cwire - the wire number for the first wire for the condition register
        #         or if cregbundle, wire number of the condition register itself
        # gap - the number of wires from cwire to the bottom gate qubit
        label, val_bits = get_condition_label_val(op.condition, self._circuit, self._cregbundle)
        cond_is_bit = isinstance(op.condition[0], Clbit)
        cond_reg = op.condition[0]
        if cond_is_bit:
            register = get_bit_register(self._circuit, op.condition[0])
            if register is not None:
                cond_reg = register
        meas_offset = -0.3 if isinstance(op, Measure) else 0.0

        # If condition is a bit or cregbundle true, print the condition value
        # at the bottom and put bullet on creg line
        if cond_is_bit or self._cregbundle:
            cwire = (
                self._wire_map[cond_reg] if self._cregbundle else self._wire_map[op.condition[0]]
            )
            gap = cwire - max(wire_list)
            control = "\\control" if op.condition[1] else "\\controlo"
            self._latex[cwire][
                col
            ] = f"{control} \\cw^({meas_offset}){{^{{\\mathtt{{{label}}}}}}} \\cwx[-{gap!s}]"

        # If condition is a register and cregbundle is false
        else:
            # First sort the val_bits in the order of the register bits in the circuit
            cond_wires = []
            cond_bits = []
            for wire, index in self._wire_map.items():
                reg, _, reg_index = get_bit_reg_index(self._circuit, wire)
                if reg == cond_reg:
                    cond_bits.append(reg_index)
                    cond_wires.append(index)

            gap = cond_wires[0] - max(wire_list)
            prev_wire = cond_wires[0]
            val_bits_sorted = [bit for _, bit in sorted(zip(cond_bits, val_bits))]

            # Iterate over the wire values for the bits in the register
            for i, wire in enumerate(cond_wires[:-1]):
                if i > 0:
                    gap = wire - prev_wire
                control = "\\control" if val_bits_sorted[i] == "1" else "\\controlo"
                self._latex[wire][col] = f"{control} \\cw \\cwx[-" + str(gap) + "]"
                prev_wire = wire

            # Add (hex condition value) below the last cwire
            if len(cond_wires) == 1:  # Only one bit in register
                gap = cond_wires[0] - max(wire_list)
            else:
                gap = cond_wires[-1] - prev_wire
            control = "\\control" if val_bits_sorted[len(cond_wires) - 1] == "1" else "\\controlo"
            self._latex[cond_wires[-1]][col] = (
                f"{control}" + " \\cw^(%s){^{\\mathtt{%s}}} \\cwx[-%s]"
            ) % (
                meas_offset,
                label,
                str(gap),
            )

    def _truncate_float(self, matchobj, ndigits=4):
        """Truncate long floats."""
        if matchobj.group(0):
            return f"%.{ndigits}g" % float(matchobj.group(0))
        return ""
