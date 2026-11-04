# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Matplotlib circuit visualization style."""

from __future__ import annotations

import json
from pathlib import Path
from qiskit.visualization.circuit.qcstyle import StyleDict, DefaultStyle


class DAGStyleDict(StyleDict):
    """A dictionary for graphviz styles.

    Defines additional abbreviations for key accesses, such as allowing
    ``"ec"`` instead of writing ``"edgecolor"``.
    """

    VALID_FIELDS = {
        "name",
        "fontsize",
        "bgcolor",
        "dpi",
        "pad",
        "nodecolor",
        "inputnodecolor",
        "inputnodefontcolor",
        "outputnodecolor",
        "outputnodefontcolor",
        "opnodecolor",
        "opnodefontcolor",
        "edgecolor",
        "qubitedgecolor",
        "clbitedgecolor",
    }

    ABBREVIATIONS = {
        "nc": "nodecolor",
        "ic": "inputnodecolor",
        "if": "inputnodefontcolor",
        "oc": "outputnodecolor",
        "of": "outputnodefontcolor",
        "opc": "opnodecolor",
        "opf": "opnodefontcolor",
        "ec": "edgecolor",
        "qec": "qubitedgecolor",
        "clc": "clbitedgecolor",
    }


class DAGDefaultStyle(DefaultStyle):
    """Creates a Default Style dictionary

    The style dict contains numerous options that define the style of the
    output circuit visualization. The style dict is used by the `graphviz`
    output. The options available in the style dict are defined below:

    Attributes:
        name (str): The name of the style.
        fontsize (str): The font size to use for text.
        bgcolor (str): The color name to use for the background ('red', 'green', etc.).
        nodecolor (str): The color to use for all nodes.
        dpi (int): The DPI to use for the output image.
        pad (int): A number to adjust padding around output
            graph.
        inputnodecolor (str): The color to use for incoming wire nodes. Overrides
            nodecolor for those nodes.
        inputnodefontcolor (str): The font color to use for incoming wire nodes.
            Overrides nodecolor for those nodes.
        outputnodecolor (str): The color to use for output wire nodes. Overrides
            nodecolor for those nodes.
        outputnodefontcolor (str): The font color to use for output wire nodes.
            Overrides nodecolor for those nodes.
        opnodecolor (str): The color to use for Instruction nodes. Overrides
            nodecolor for those nodes.
        opnodefontcolor (str): The font color to use for Instruction nodes.
            Overrides nodecolor for those nodes.

        qubitedgecolor (str): The edge color for qubits. Overrides edgecolor for these edges.
        clbitedgecolor (str): The edge color for clbits. Overrides edgecolor for these edges.
    """

    DEFAULT_STYLE_NAME = "plain"
    DEFAULT_STYLE_PATH = Path(__file__).parent / "styles"

    def __init__(self):
        path = self.DEFAULT_STYLE_PATH / Path(self.DEFAULT_STYLE_NAME).with_suffix(".json")

        with open(path) as infile:
            default_style = json.load(infile)

        # set shortcuts, such as "ec" for "edgecolor"
        self.style = DAGStyleDict(**default_style)
