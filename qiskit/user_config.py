# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utils for reading a user preference config files."""

import configparser
import os
from warnings import warn

from qiskit import exceptions

DEFAULT_FILENAME = os.path.join(os.path.expanduser("~"), ".qiskit", "settings.conf")


class UserConfig:
    """Class representing a user config file

    The config file format should look like:

    [default]
    circuit_drawer = mpl
    circuit_mpl_style = default
    circuit_mpl_style_path = ~/.qiskit:<default location>
    circuit_reverse_bits = True
    circuit_idle_wires = False
    transpile_optimization_level = 1
    transpiler_seed = 42
    parallel = False
    num_processes = 4
    sabre_all_threads = true
    min_qpy_version = 13
    """

    def __init__(self, filename=None):
        """Create a UserConfig

        Args:
            filename (str): The path to the user config file. If one isn't
                specified, ~/.qiskit/settings.conf is used.
        """
        if filename is None:
            self.filename = DEFAULT_FILENAME
        else:
            self.filename = filename
        self.settings = {}
        self.config_parser = configparser.ConfigParser()

    def read_config_file(self):
        """Read config file and parse the contents into the settings attr."""
        if not os.path.isfile(self.filename):
            return
        self.config_parser.read(self.filename)
        if "default" in self.config_parser.sections():
            # Parse circuit_drawer
            circuit_drawer = self.config_parser.get("default", "circuit_drawer", fallback=None)
            if circuit_drawer:
                if circuit_drawer not in ["text", "mpl", "latex", "latex_source", "auto"]:
                    raise exceptions.QiskitUserConfigError(
                        f"{circuit_drawer} is not a valid circuit drawer backend. Must be "
                        "either 'text', 'mpl', 'latex', 'latex_source', or "
                        "'auto'."
                    )
                self.settings["circuit_drawer"] = circuit_drawer

            # Parse state_drawer
            state_drawer = self.config_parser.get("default", "state_drawer", fallback=None)
            if state_drawer:
                valid_state_drawers = [
                    "repr",
                    "text",
                    "latex",
                    "latex_source",
                    "qsphere",
                    "hinton",
                    "bloch",
                ]
                if state_drawer not in valid_state_drawers:
                    valid_choices_string = "', '".join(c for c in valid_state_drawers)
                    raise exceptions.QiskitUserConfigError(
                        f"'{state_drawer}' is not a valid state drawer backend. "
                        f"Choose from: '{valid_choices_string}'"
                    )
                self.settings["state_drawer"] = state_drawer

            # Parse circuit_mpl_style
            circuit_mpl_style = self.config_parser.get(
                "default", "circuit_mpl_style", fallback=None
            )
            if circuit_mpl_style:
                if not isinstance(circuit_mpl_style, str):
                    warn(
                        f"{circuit_mpl_style} is not a valid mpl circuit style. Must be "
                        "a text string. Will not load style.",
                        UserWarning,
                        2,
                    )
                self.settings["circuit_mpl_style"] = circuit_mpl_style

            # Parse circuit_mpl_style_path
            circuit_mpl_style_path = self.config_parser.get(
                "default", "circuit_mpl_style_path", fallback=None
            )
            if circuit_mpl_style_path:
                cpath_list = circuit_mpl_style_path.split(":")
                for path in cpath_list:
                    if not os.path.exists(os.path.expanduser(path)):
                        warn(
                            f"{path} is not a valid circuit mpl style path."
                            " Correct the path in ~/.qiskit/settings.conf.",
                            UserWarning,
                            2,
                        )
                self.settings["circuit_mpl_style_path"] = cpath_list

            # Parse circuit_reverse_bits
            try:
                circuit_reverse_bits = self.config_parser.getboolean(
                    "default", "circuit_reverse_bits", fallback=None
                )
            except ValueError as err:
                raise exceptions.QiskitUserConfigError(
                    f"Value assigned to circuit_reverse_bits is not valid. {err!s}"
                )
            if circuit_reverse_bits is not None:
                self.settings["circuit_reverse_bits"] = circuit_reverse_bits

            # Parse circuit_idle_wires
            try:
                circuit_idle_wires = self.config_parser.getboolean(
                    "default", "circuit_idle_wires", fallback=None
                )
            except ValueError as err:
                raise exceptions.QiskitUserConfigError(
                    f"Value assigned to circuit_idle_wires is not valid. {err!s}"
                )
            if circuit_idle_wires is not None:
                self.settings["circuit_idle_wires"] = circuit_idle_wires

            # Parse circuit_measure_arrows
            try:
                circuit_measure_arrows = self.config_parser.getboolean(
                    "default", "circuit_measure_arrows", fallback=None
                )
            except ValueError as err:
                raise exceptions.QiskitUserConfigError(
                    f"Value assigned to circuit_measure_arrows is not valid. {err!s}"
                )
            if circuit_measure_arrows is not None:
                self.settings["circuit_measure_arrows"] = circuit_measure_arrows

            # Parse transpile_optimization_level
            transpile_optimization_level = self.config_parser.getint(
                "default", "transpile_optimization_level", fallback=-1
            )
            if transpile_optimization_level != -1:
                if transpile_optimization_level < 0 or transpile_optimization_level > 3:
                    raise exceptions.QiskitUserConfigError(
                        "%s is not a valid optimization level. Must be 0, 1, 2, or 3."
                    )
                self.settings["transpile_optimization_level"] = transpile_optimization_level

            # Parse parallel
            parallel_enabled = self.config_parser.getboolean("default", "parallel", fallback=None)
            if parallel_enabled is not None:
                self.settings["parallel_enabled"] = parallel_enabled

            # Parse num_processes
            num_processes = self.config_parser.getint("default", "num_processes", fallback=-1)
            if num_processes != -1:
                if num_processes <= 0:
                    raise exceptions.QiskitUserConfigError(
                        "%s is not a valid number of processes. Must be greater than 0"
                    )
                self.settings["num_processes"] = num_processes

            # Parse sabre_all_threads
            sabre_all_threads = self.config_parser.getboolean(
                "default", "sabre_all_threads", fallback=None
            )
            if sabre_all_threads is not None:
                self.settings["sabre_all_threads"] = sabre_all_threads

            # Parse min_qpy_version
            try:
                min_qpy_version = self.config_parser.getint(
                    "default", "min_qpy_version", fallback=None
                )
            except ValueError as ve:
                raise exceptions.QiskitUserConfigError(
                    "min_qpy_version is not a valid QPY version."
                ) from ve
            if min_qpy_version:
                if min_qpy_version < 0:
                    raise exceptions.QiskitUserConfigError(
                        f"min_qpy_version {min_qpy_version} is not a valid QPY version."
                    )
                self.settings["min_qpy_version"] = min_qpy_version

            # Parse transpiler_seed
            try:
                transpiler_seed = self.config_parser.getint(
                    "default",
                    "transpiler_seed",
                    fallback=None,
                )
            except ValueError as ve:
                raise exceptions.QiskitUserConfigError(
                    "transpiler_seed in the user configuration file is not a valid seed value. It "
                    "must be a non-negative integer."
                ) from ve
            if transpiler_seed is not None:
                if transpiler_seed < 0:
                    raise exceptions.QiskitUserConfigError(
                        "transpiler_seed in the user configuration file is a negative int. It "
                        "must be a non-negative integer."
                    )
                self.settings["transpiler_seed"] = transpiler_seed


def set_config(key, value, section=None, file_path=None):
    """Adds or modifies a user configuration

    It will add configuration to the currently configured location
    or the value of file argument.

    Only valid user config can be set in 'default' section. Custom
    user config can be added in any other sections.

    Changes to the existing config file will not be reflected in
    the current session since the config file is parsed at import time.

    Args:
        key (str): name of the config
        value (obj): value of the config
        section (str, optional): if not specified, adds it to the
            `default` section of the config file.
        file_path (str, optional): the file to which config is added.
            If not specified, adds it to the default config file or
            if set, the value of `QISKIT_SETTINGS` env variable.

    Raises:
        QiskitUserConfigError: if the config is invalid
    """
    filename = file_path or os.getenv("QISKIT_SETTINGS", DEFAULT_FILENAME)
    section = "default" if section is None else section

    if not isinstance(key, str):
        raise exceptions.QiskitUserConfigError("Key must be string type")

    valid_config = {
        "circuit_drawer",
        "circuit_mpl_style",
        "circuit_mpl_style_path",
        "circuit_reverse_bits",
        "circuit_idle_wires",
        "transpile_optimization_level",
        "parallel",
        "num_processes",
        "sabre_all_threads",
        "min_qpy_version",
        "transpiler_seed",
    }

    if section in [None, "default"]:
        if key not in valid_config:
            raise exceptions.QiskitUserConfigError(f"{key} is not a valid user config.")

    config = configparser.ConfigParser()
    config.read(filename)

    if section not in config.sections():
        config.add_section(section)

    config.set(section, key, str(value))

    try:
        with open(filename, "w") as cfgfile:
            config.write(cfgfile)
    except OSError as ex:
        raise exceptions.QiskitUserConfigError(
            f"Unable to load the config file {filename}. Error: '{ex!s}'"
        )

    # validates config
    user_config = UserConfig(filename)
    user_config.read_config_file()


def get_config():
    """Read the config file from the default location or env var.

    It will read a config file at the location specified by the ``QISKIT_SETTINGS`` environment
    variable if set, or ``$HOME/.qiskit/settings.conf`` if not.

    If the environment variable ``QISKIT_IGNORE_USER_SETTINGS`` is set to the string ``TRUE``, this
    will return an empty configuration, regardless of all other variables.

    Returns:
        dict: The settings dict from the parsed config file.
    """
    if os.getenv("QISKIT_IGNORE_USER_SETTINGS", "false").lower() == "true":
        return {}
    filename = os.getenv("QISKIT_SETTINGS", DEFAULT_FILENAME)
    if not os.path.isfile(filename):
        return {}
    user_config = UserConfig(filename)
    user_config.read_config_file()
    return user_config.settings
