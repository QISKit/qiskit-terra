# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Pass manager test cases."""

import contextlib
import logging
import re
from itertools import zip_longest
from logging import getLogger

from test import QiskitTestCase


class BasePassManagerTestCase(QiskitTestCase):
    """Test case for the pass manager module."""

    @contextlib.contextmanager
    def assertLogContains(self, expected_lines):
        """A context manager that capture pass manager log.

        Args:
            expected_lines (List[str]): Expected logs. Each element can be regular expression.
        """
        try:
            logger = getLogger()
            with self.assertLogs(logger=logger, level=logging.DEBUG) as cm:
                yield cm
        finally:
            recorded_lines = cm.output
            for i, (expected, recorded) in enumerate(zip_longest(expected_lines, recorded_lines)):
                expected = expected or ""
                recorded = recorded or ""
                if not re.search(expected, recorded):
                    raise AssertionError(
                        f"Log didn't match. Mismatch found at line #{i}.\n\n"
                        f"Expected:\n{self._format_log(expected_lines)}\n"
                        f"Recorded:\n{self._format_log(recorded_lines)}"
                    )

    def _format_log(self, lines):
        out = ""
        for i, line in enumerate(lines):
            out += f"#{i:02d}: {line}\n"
        return out
