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

"""
Generic style visualization library.
"""

import json
import os
from typing import Any
from warnings import warn
from pathlib import Path

from qiskit import user_config

from .exceptions import VisualizationError


class StyleDict(dict):
    """
    Attributes:
        VALID_FIELDS (set): Set of valid field inputs to a function that supports a style parameter
        ABBREVIATIONS (dict): Mapping of abbreviation:field for abbreviated inputs to VALID_FIELDS
            (must exist in VALID FIELDS)
        NESTED_ATTRS (set): Set of fields that are dictionaries, and need to be updated with .update
    """

    VALID_FIELDS = set()
    ABBREVIATIONS = {}
    NESTED_ATTRS = set()

    def __setitem__(self, key: Any, value: Any) -> None:
        # allow using field ABBREVIATIONS
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        if key not in self.VALID_FIELDS:
            warn(
                f"style option ({key}) is not supported",
                UserWarning,
                2,
            )
        return super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        # allow using field ABBREVIATIONS
        if key in self.ABBREVIATIONS:
            key = self.ABBREVIATIONS[key]

        return super().__getitem__(key)

    def update(self, other):
        # the attributes "displaycolor" and "displaytext" are dictionaries
        # themselves, therefore we need to propagate the update down to them
        for attr in self.NESTED_ATTRS.intersection(other.keys()):
            if attr in self.keys():
                self[attr].update(other[attr])
            else:
                self[attr] = other[attr]

        super().update((key, value) for key, value in other.items() if key not in self.NESTED_ATTRS)


class DefaultStyle:
    """
    Attributes:
        DEFAULT_STYLE_NAME (str): style name for the default style
        STYLE_PATH: file path where DEFAULT_STYLE_NAME.json is located
    """

    DEFAULT_STYLE_NAME = None
    DEFAULT_STYLE_PATH = None

    def __init__(self):
        raise NotImplementedError


def load_style(
    style: dict | str | None,
    style_dict: type[StyleDict],
    default_style: DefaultStyle,
    user_config_opt: str,
    user_config_path_opt: str,
    raise_error_if_not_found: bool = False,
) -> tuple[StyleDict, float]:
    """Utility function to load style from json files.

    Args:
        style: Depending on the type, this acts differently:

            * If a string, it can specify a supported style name (such
              as "iqp" or "clifford"). It can also specify the name of
              a custom color scheme stored as JSON file. This JSON file
              _must_ specify a complete set of colors.
            * If a dictionary, it may specify the style name via a
              ``{"name": "<desired style>"}`` entry. If this is not given,
              the default style will be used. The remaining entries in the
              dictionary can be used to override certain specs.
              E.g. ``{"name": "iqp", "ec": "#FF0000"}`` will use the ``"iqp"``
              color scheme but set the edgecolor to red.
        style_dict: The class used to define the options for loading styles
        default_style: DefaultStyle dictionary definition and documentation
        user_config_opt: User config field in the Qiskit User Configuration File
            used to define the style loaded
        user_config_path_opt: User config field in the Qiskit User Configuration File
            used to define the path to the style loaded
        raise_error_if_not_found: When True, load_style will throw a VisualizationError
            if the style parameter file is not found. When False, load_style will load
            the style passed in by the default_style parameter.


    Returns:
        A tuple containing the style as dictionary and the default font ratio.
    """

    default = default_style.DEFAULT_STYLE_NAME

    # if the style is not given, try to load the configured default (if set),
    # or use the default style
    config = user_config.get_config()
    if style is None:
        if config:
            style = config.get(user_config_opt, default)
        else:
            style = default

    # determine the style name which could also be inside a dictionary, like
    # style={"name": "clifford", <other settings...>}
    if isinstance(style, dict):
        style_name = style.get("name", default)
    elif isinstance(style, str):
        if style.endswith(".json"):
            style_name = style[:-5]
        else:
            style_name = style
    else:
        warn(
            f'Unsupported style parameter "{style}" of type {type(style)}. '
            "Will use the default style.",
            UserWarning,
            2,
        )
        style_name = default

    if style_name in [default]:
        current_style = default_style.style
    else:
        # Search for file in 'styles' dir, then config_path, and finally the current directory
        style_name = style_name + ".json"
        style_paths = []

        default_path = default_style.DEFAULT_STYLE_PATH / style_name
        style_paths.append(default_path)

        # check configured paths, if there are any
        if config:
            config_path = config.get(user_config_path_opt, "")
            if config_path:
                for path in config_path:
                    style_paths.append(Path(path) / style_name)

        # check current directory
        cwd_path = Path("") / style_name
        style_paths.append(cwd_path)

        for path in style_paths:
            # expand ~ to the user directory and check if the file exists
            exp_user = path.expanduser()
            if os.path.isfile(exp_user):
                try:
                    with open(exp_user) as infile:
                        json_style = json.load(infile)

                    current_style = style_dict(json_style)
                    break
                except json.JSONDecodeError as err:
                    warn(
                        f"Could not decode JSON in file '{path}': {err!s}. "
                        "Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
                except (OSError, FileNotFoundError):
                    warn(
                        f"Error loading JSON file '{path}'. Will use default style.",
                        UserWarning,
                        2,
                    )
                    break
        else:
            if raise_error_if_not_found:
                raise VisualizationError(f"Invalid style {style_name}")

            warn(
                f"Style JSON file '{style_name}' not found in any of these locations: "
                f"{', '.join(map(str, style_paths))}. "
                "Will use default style.",
                UserWarning,
                2,
            )
            current_style = default_style.style

    # if the style is a dictionary, update the defaults with the new values
    # this _needs_ to happen after loading by name to cover cases like
    #   style = {"name": "bw", "edgecolor": "#FF0000"}
    if isinstance(style, dict):
        current_style.update(style)

    # this is the default font ratio
    # if the font- or subfont-sizes are changed, the new size is based on this ratio
    def_font_ratio = 13 / 8

    return current_style, def_font_ratio
