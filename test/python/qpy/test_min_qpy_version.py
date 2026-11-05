"""Tests for min_qpy_version option"""

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

import struct
from unittest import mock
import unittest
from uuid import uuid4
import io

from qiskit.qpy import load, formats, QpyError
from qiskit.qpy.common import QPY_RUST_READ_MIN_VERSION


class TestMinQpyVersion(unittest.TestCase):
    """Tests for min_qpy_version validation."""

    def setUp(self):
        super().setUp()
        self.file_path = f"test_{uuid4()}.conf"

    @mock.patch("qiskit.user_config.get_config")
    def test_enforce_min_qpy_version(self, mock_get_config):
        """Test that QPY file below min_qpy_version raises QiskitError."""
        mock_get_config.return_value = {"min_qpy_version": 12}
        with io.BytesIO() as buf:
            buf.write(struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", 9, 1, 0, 0, 0))
            buf.seek(0)
            with self.assertRaises(QpyError) as cm:
                load(buf)
        self.assertIn("is lower than the configured minimum version", str(cm.exception))

    @mock.patch("qiskit.user_config.get_config")
    def test_enforce_min_qpy_version_with_rust(self, mock_get_config):
        """Test that QPY file below min_qpy_version raises QiskitError."""
        mock_get_config.return_value = {"min_qpy_version": QPY_RUST_READ_MIN_VERSION + 1}
        with io.BytesIO() as buf:
            buf.write(
                struct.pack(
                    formats.FILE_HEADER_PACK, b"QISKIT", QPY_RUST_READ_MIN_VERSION, 1, 0, 0, 0
                )
            )
            buf.seek(0)
            with self.assertRaises(QpyError) as cm:
                load(buf)
        self.assertIn("is lower than the configured minimum version", str(cm.exception))

    @mock.patch("qiskit.user_config.get_config")
    def test_matches_min_qpy_version(self, mock_get_config):
        """Test that QPY file matches min_qpy_version."""
        mock_get_config.return_value = {"min_qpy_version": 4}
        with io.BytesIO() as buf:
            buf.write(struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", 4, 1, 0, 0, 0))
            buf.seek(0)
            no_circs = load(buf)
        self.assertEqual(no_circs, [])

    @mock.patch("qiskit.user_config.get_config")
    def test_skip_enforcement_if_unset(self, mock_get_config):
        """Test that load proceeds if min_qpy_version is unset."""
        mock_get_config.return_value = {"min_qpy_version": None}
        with io.BytesIO() as buf:
            buf.write(struct.pack(formats.FILE_HEADER_PACK, b"QISKIT", 4, 1, 0, 0, 0))
            buf.seek(0)
            no_circs = load(buf)
        self.assertEqual(no_circs, [])
