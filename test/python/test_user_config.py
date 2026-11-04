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


import os
import configparser as cp
from uuid import uuid4
from unittest import mock

from qiskit import exceptions
from qiskit import user_config
from test import QiskitTestCase


class TestUserConfig(QiskitTestCase):
    def setUp(self):
        super().setUp()
        self.file_path = f"test_{uuid4()}.conf"

    def test_empty_file_read(self):
        config = user_config.UserConfig(self.file_path)
        config.read_config_file()
        self.assertEqual({}, config.settings)

    def test_invalid_optimization_level(self):
        test_config = """
        [default]
        transpile_optimization_level = 76
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_circuit_drawer(self):
        test_config = """
        [default]
        circuit_drawer = MSPaint
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_circuit_drawer_valid(self):
        test_config = """
        [default]
        circuit_drawer = latex
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"circuit_drawer": "latex"}, config.settings)

    def test_invalid_circuit_reverse_bits(self):
        test_config = """
        [default]
        circuit_reverse_bits = Neither
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_circuit_reverse_bits_valid(self):
        test_config = """
        [default]
        circuit_reverse_bits = false
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"circuit_reverse_bits": False}, config.settings)

    def test_invalid_circuit_idle_wires(self):
        test_config = """
        [default]
        circuit_idle_wires = Neither
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_circuit_idle_wires_valid(self):
        test_config = """
        [default]
        circuit_idle_wires = true
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"circuit_idle_wires": True}, config.settings)

    def test_optimization_level_valid(self):
        test_config = """
        [default]
        transpile_optimization_level = 1
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"transpile_optimization_level": 1}, config.settings)

    def test_invalid_num_processes(self):
        test_config = """
        [default]
        num_processes = -256
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_valid_num_processes(self):
        test_config = """
        [default]
        num_processes = 31
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"num_processes": 31}, config.settings)

    def test_valid_parallel(self):
        test_config = """
        [default]
        parallel = False
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"parallel_enabled": False}, config.settings)

    def test_all_options_valid(self):
        test_config = """
        [default]
        circuit_drawer = latex
        circuit_mpl_style = default
        circuit_mpl_style_path = ~:~/.qiskit
        circuit_reverse_bits = false
        circuit_idle_wires = true
        transpile_optimization_level = 3
        suppress_packaging_warnings = true
        parallel = false
        num_processes = 15
        transpiler_seed = 42
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()

        self.assertEqual(
            {
                "circuit_drawer": "latex",
                "circuit_mpl_style": "default",
                "circuit_mpl_style_path": ["~", "~/.qiskit"],
                "circuit_reverse_bits": False,
                "circuit_idle_wires": True,
                "transpile_optimization_level": 3,
                "num_processes": 15,
                "parallel_enabled": False,
                "transpiler_seed": 42,
            },
            config.settings,
        )

    def test_set_config_all_options_valid(self):
        self.addCleanup(os.remove, self.file_path)

        user_config.set_config("circuit_drawer", "latex", file_path=self.file_path)
        user_config.set_config("circuit_mpl_style", "default", file_path=self.file_path)
        user_config.set_config("circuit_mpl_style_path", "~:~/.qiskit", file_path=self.file_path)
        user_config.set_config("circuit_reverse_bits", "false", file_path=self.file_path)
        user_config.set_config("circuit_idle_wires", "true", file_path=self.file_path)
        user_config.set_config("transpile_optimization_level", "3", file_path=self.file_path)
        user_config.set_config("parallel", "false", file_path=self.file_path)
        user_config.set_config("num_processes", "15", file_path=self.file_path)
        user_config.set_config("min_qpy_version", "10", file_path=self.file_path)
        user_config.set_config("transpiler_seed", "42", file_path=self.file_path)

        config_settings = None
        with mock.patch.dict(os.environ, {"QISKIT_SETTINGS": self.file_path}, clear=True):
            config_settings = user_config.get_config()

        self.assertEqual(
            {
                "circuit_drawer": "latex",
                "circuit_mpl_style": "default",
                "circuit_mpl_style_path": ["~", "~/.qiskit"],
                "circuit_reverse_bits": False,
                "circuit_idle_wires": True,
                "transpile_optimization_level": 3,
                "num_processes": 15,
                "parallel_enabled": False,
                "min_qpy_version": 10,
                "transpiler_seed": 42,
            },
            config_settings,
        )

    def test_set_config_multiple_sections(self):
        self.addCleanup(os.remove, self.file_path)

        user_config.set_config("circuit_drawer", "latex", file_path=self.file_path)
        user_config.set_config("circuit_mpl_style", "default", file_path=self.file_path)
        user_config.set_config("transpile_optimization_level", "3", file_path=self.file_path)

        user_config.set_config("circuit_drawer", "latex", section="test", file_path=self.file_path)
        user_config.set_config("parallel", "false", section="test", file_path=self.file_path)
        user_config.set_config("num_processes", "15", section="test", file_path=self.file_path)

        config = cp.ConfigParser()
        config.read(self.file_path)

        self.assertEqual(config.sections(), ["default", "test"])

        self.assertEqual(
            {
                "circuit_drawer": "latex",
                "circuit_mpl_style": "default",
                "transpile_optimization_level": "3",
            },
            dict(config.items("default")),
        )

    def test_valid_min_qpy_version(self):
        """Test parsing a valid integer min_qpy_version."""
        test_config = """
        [default]
        min_qpy_version = 10
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
        self.assertEqual({"min_qpy_version": 10}, config.settings)

    def test_empty_min_qpy_version(self):
        """Test that empty min_qpy_version is treated as wrong."""
        test_config = """
        [default]
        min_qpy_version =
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
        self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_min_qpy_version_non_integer(self):
        """Test that non-integer min_qpy_version raises QiskitUserConfigError."""
        test_config = """
        [default]
        min_qpy_version = 2.0
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_min_qpy_version_negative(self):
        """Test that negative min_qpy_version raises QiskitUserConfigError."""
        test_config = """
        [default]
        min_qpy_version = -1
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_min_qpy_version_string(self):
        """Test that non-numeric min_qpy_version raises QiskitUserConfigError."""
        test_config = """
        [default]
        min_qpy_version = abc
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_valid_transpiler_seed(self):
        test_config = """
        [default]
        transpiler_seed = 42
        """
        self.addClassCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            config.read_config_file()
            self.assertEqual({"transpiler_seed": 42}, config.settings)

    def test_empty_seed_transpiler(self):
        test_config = """
        [default]
        transpiler_seed =
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
        self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_transpiler_seed_non_integer(self):
        test_config = """
        [default]
        transpiler_seed = 2.0
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_transpiler_seed_string(self):
        test_config = """
        [default]
        transpiler_seed = xyz
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)

    def test_invalid_transpiler_seed_negative(self):
        test_config = """
        [default]
        transpiler_seed = -42
        """
        self.addCleanup(os.remove, self.file_path)
        with open(self.file_path, "w") as file:
            file.write(test_config)
            file.flush()
            config = user_config.UserConfig(self.file_path)
            self.assertRaises(exceptions.QiskitUserConfigError, config.read_config_file)
