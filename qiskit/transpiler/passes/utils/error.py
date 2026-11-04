# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Error pass to be called when an error happens."""

import logging
import string
import warnings

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError


class Error(AnalysisPass):
    """Error pass to be called when an error happens."""

    def __init__(self, msg=None, action="raise"):
        """Error pass.

        Args:
            msg (str | Callable[[PropertySet], str]): Error message, if not provided a generic error
                will be used.  This can be either a raw string, or a callback function that accepts
                the current ``property_set`` and returns the desired message.
            action (str): the action to perform. Default: 'raise'. The options are:
              * ``'raise'``: Raises a ``TranspilerError`` exception with msg
              * ``'warn'``: Raises a non-fatal warning with msg
              * ``'log'``: logs in ``logging.getLogger(__name__)``

        Raises:
            TranspilerError: if action is not valid.
        """
        super().__init__()
        self.msg = msg
        if action in ["raise", "warn", "log"]:
            self.action = action
        else:
            raise TranspilerError(f"Unknown action: {action}")

    def run(self, _):
        """Run the Error pass on `dag`."""
        if self.msg is None:
            msg = "An error occurred while the pass manager was running."
        elif isinstance(self.msg, str):
            prop_names = [
                tup[1] for tup in string.Formatter().parse(self.msg) if tup[1] is not None
            ]
            properties = {prop_name: self.property_set[prop_name] for prop_name in prop_names}
            msg = self.msg.format(**properties)
        else:
            msg = self.msg(self.property_set)

        if self.action == "raise":
            raise TranspilerError(msg)
        if self.action == "warn":
            warnings.warn(msg, Warning)
        elif self.action == "log":
            logger = logging.getLogger(__name__)
            logger.info(msg)
        else:
            raise TranspilerError(f"Unknown action: {self.action}")
