# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/tools/parallel"""
import os
import subprocess
import sys
import tempfile
from unittest import mock

from qiskit.utils import local_hardware_info, should_run_in_parallel, parallel_map
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from test import QiskitTestCase


def _parfunc(x):
    """Function for testing parallel_map"""
    return x


def _build_simple_circuit(_):
    qreg = QuantumRegister(2)
    creg = ClassicalRegister(2)
    qc = QuantumCircuit(qreg, creg)
    return qc


class TestParallel(QiskitTestCase):
    """A class for testing parallel_map functionality."""

    def test_parallel(self):
        """Test parallel_map"""
        ans = parallel_map(_parfunc, list(range(10)))
        self.assertEqual(ans, list(range(10)))

    def test_parallel_circuit_names(self):
        """Verify unique circuit names in parallel"""
        out_circs = parallel_map(_build_simple_circuit, list(range(10)))
        names = [circ.name for circ in out_circs]
        self.assertEqual(len(names), len(set(names)))


class TestUtilities(QiskitTestCase):
    """Tests for parallel utilities."""

    def test_local_hardware_five_cpu_count(self):
        """Test cpu count is half when sched affinity is 5"""
        with mock.patch.object(os, "sched_getaffinity", return_value=set(range(5)), create=True):
            self.assertEqual(2, local_hardware_info()["cpus"])

    def test_local_hardware_sixty_four_cpu_count(self):
        """Test cpu count is 32 when sched affinity is 64"""
        with mock.patch.object(os, "sched_getaffinity", return_value=set(range(64)), create=True):
            self.assertEqual(32, local_hardware_info()["cpus"])

    def test_local_hardware_no_cpu_count(self):
        """Test cpu count fallback to 1 when true value can't be determined"""
        with mock.patch.object(os, "sched_getaffinity", return_value=set(), create=True):
            self.assertEqual(1, local_hardware_info()["cpus"])

    def test_local_hardware_no_sched_five_count(self):
        """Test cpu could if sched affinity method is missing and cpu count is 5."""
        with (
            mock.patch.object(os, "sched_getaffinity", None, create=True),
            mock.patch.object(os, "cpu_count", return_value=5),
        ):
            self.assertEqual(2, local_hardware_info()["cpus"])

    def test_local_hardware_no_sched_sixty_four_count(self):
        """Test cpu could if sched affinity method is missing and cpu count is 64."""
        with (
            mock.patch.object(os, "sched_getaffinity", None, create=True),
            mock.patch.object(os, "cpu_count", return_value=64),
        ):
            self.assertEqual(32, local_hardware_info()["cpus"])

    def test_local_hardware_no_sched_no_count(self):
        """Test cpu count fallback to 1 when no sched getaffinity available."""
        with (
            mock.patch.object(os, "sched_getaffinity", None, create=True),
            mock.patch.object(os, "cpu_count", return_value=None),
        ):
            self.assertEqual(1, local_hardware_info()["cpus"])

    def test_should_run_in_parallel_override(self):
        """Test that the context managers allow overriding the default value."""
        natural = should_run_in_parallel(8)
        with should_run_in_parallel.override(True):
            self.assertTrue(should_run_in_parallel(8))
        self.assertEqual(should_run_in_parallel(8), natural)
        with should_run_in_parallel.override(False):
            self.assertFalse(should_run_in_parallel(8))
        self.assertEqual(should_run_in_parallel(8), natural)

    def test_should_run_in_parallel_ignore_user_settings(self):
        """Test that the context managers allow overriding the user settings."""
        # This is a nasty one, because much of the user settings are read statically at `import
        # qiskit`, which we're obviously already past.  We want to override that, so we need a
        # subprocess whose environment we control completely.

        # Windows is picky about opening files that are already opened for writing.  Ideally we'd
        # use a context manager with `delete_on_close=False` so we close the file, launch our
        # subprocess and let the CM clean up on exit, but that argument only arrived in Python 3.12.

        # We're deliberately writing out to a temporary file.
        settings_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf8", delete=False)
        settings_file.write(
            """\
[DEFAULT]
parallel = true
"""
        )
        settings_file.close()
        self.addCleanup(os.remove, settings_file.name)

        # Pass on all our environment, except for our own configuration, which we override with our
        # custom settings file,
        env = {key: value for key, value in os.environ.items() if not key.startswith("QISKIT")}
        env["QISKIT_SETTINGS"] = settings_file.name
        env["QISKIT_IN_PARALLEL"] = "FALSE"
        env["QISKIT_PARALLEL"] = "TRUE"
        env["QISKIT_IGNORE_USER_SETTINGS"] = "FALSE"

        script = """\
import multiprocessing
from unittest.mock import patch
from qiskit.utils import should_run_in_parallel

print(should_run_in_parallel(8))
with (
    patch.object(multiprocessing, "get_start_method", return_value="forkserver"),
    should_run_in_parallel.ignore_user_settings(),
):
    print(should_run_in_parallel(8))
with (
    patch.object(multiprocessing, "get_start_method", return_value="spawn"),
    should_run_in_parallel.ignore_user_settings(),
):
    print(should_run_in_parallel(8))
"""
        result = subprocess.run(
            sys.executable,
            input=script,
            encoding="utf8",
            text=True,
            env=env,
            check=True,
            capture_output=True,
        )
        user_settings, forkserver_default, spawn_default = result.stdout.splitlines()
        self.assertEqual(
            (user_settings, forkserver_default, spawn_default), ("True", "True", "False")
        )
