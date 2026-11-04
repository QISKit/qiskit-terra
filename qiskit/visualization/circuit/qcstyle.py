# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
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

from qiskit.visualization.style import StyleDict, DefaultStyle


class MPLStyleDict(StyleDict):
    """A dictionary for matplotlib styles.

    Defines additional abbreviations for key accesses, such as allowing
    ``"ec"`` instead of writing ``"edgecolor"``.
    """

    VALID_FIELDS = {
        "name",
        "textcolor",
        "gatetextcolor",
        "subtextcolor",
        "linecolor",
        "creglinecolor",
        "gatefacecolor",
        "barrierfacecolor",
        "backgroundcolor",
        "edgecolor",
        "fontsize",
        "subfontsize",
        "showindex",
        "figwidth",
        "dpi",
        "margin",
        "creglinestyle",
        "displaytext",
        "displaycolor",
    }

    ABBREVIATIONS = {
        "tc": "textcolor",
        "gt": "gatetextcolor",
        "sc": "subtextcolor",
        "lc": "linecolor",
        "cc": "creglinecolor",
        "gc": "gatefacecolor",
        "bc": "barrierfacecolor",
        "bg": "backgroundcolor",
        "ec": "edgecolor",
        "fs": "fontsize",
        "sfs": "subfontsize",
        "index": "showindex",
        "cline": "creglinestyle",
        "disptex": "displaytext",
        "dispcol": "displaycolor",
    }

    NESTED_ATTRS = {"displaycolor", "displaytext"}


class MPLDefaultStyle(DefaultStyle):
    """Creates a Default Style dictionary

    The style dict contains numerous options that define the style of the
    output circuit visualization. The style dict is used by the `mpl` or
    `latex` output. The options available in the style dict are defined below:

    Attributes:
        name (str): The name of the style. The name can be set to ``iqp``,
            ``iqp-dark``, ``textbook``, ``bw``, ``clifford``, or the name of a
            user-created json file. This overrides the setting in the user config
            file (usually ``~/.qiskit/settings.conf``).
        textcolor (str): the color code to use for all text not inside a gate.
        subtextcolor (str): the color code to use for subtext.
        linecolor (str): the color code to use for lines.
        creglinecolor (str): The color code to use for classical register lines.
        gatetextcolor (str): The color code to use for gate text.
        gatefacecolor (str): The color code to use for a gate if no color
            specified in the 'displaycolor' dict.
        barrierfacecolor (str): The color code to use for barriers.
        backgroundcolor (str): The color code to use for the background.
        edgecolor (str): The color code to use for gate edges when using the
            `bw` style.
        fontsize (int): The font size to use for text.
        subfontsize (int): The font size to use for subtext.
        showindex (bool): If set to True, show the index numbers at the top.
        figwidth (int): The maximum width (in inches) for the output figure.
            If set to -1, the maximum displayable width will be used.
        dpi (int): The DPI to use for the output image.
        margin (list): A list of margin values to adjust spacing around output
            image. Takes a list of 4 ints: [x left, x right, y bottom, y top].
        creglinestyle (str): The style of line to use for classical registers.
            Choices are ``solid``, ``doublet``, or any valid matplotlib
            `linestyle` kwarg value.
        displaytext (dict): a dictionary of the text to use for certain element
            types in the output visualization. These items allow the use of
            LaTeX formatting for gate names. The 'displaytext' dict can contain
            any number of elements. User created names and labels may be used as
            keys, which allow these to have Latex formatting.
        displaycolor (dict): the color codes to use for each circuit element in
            the form (gate_color, text_color). Colors can also be entered without
            the text color, such as 'u1': '#FA74A6', in which case the text color
            will always be `gatetextcolor`. The `displaycolor` dict can contain
            any number of elements. User names and labels may be used as keys,
            which allows for custom colors for user-created gates.
    """

    DEFAULT_STYLE_NAME = "iqp"
    DEFAULT_STYLE_PATH = Path(__file__).parent / "styles"

    def __init__(self):
        path = self.DEFAULT_STYLE_PATH / Path(self.DEFAULT_STYLE_NAME).with_suffix(".json")

        with open(path) as infile:
            default_style = json.load(infile)

        # set shortcuts, such as "ec" for "edgecolor"
        self.style = MPLStyleDict(**default_style)
