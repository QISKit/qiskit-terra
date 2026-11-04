# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A custom python-constraint solver used by the :class:`~.CSPLayout` pass"""
from time import time

from qiskit.utils import optionals as _optionals

# This isn't ideal usage because we will import constraint at import time
# but to ensure the CustomSolver class is defined we need to do this.
# If constraint is not installed this will not raise a missing library
# exception until CSPLayout is initialized
if _optionals.HAS_CONSTRAINT:
    from constraint import RecursiveBacktrackingSolver

    class CustomSolver(RecursiveBacktrackingSolver):
        """A wrap to RecursiveBacktrackingSolver to support ``call_limit``"""

        def __init__(self, call_limit=None, time_limit=None):
            self.call_limit = call_limit
            self.time_limit = time_limit
            self.call_current = None
            self.time_start = None
            self.time_current = None
            super().__init__()

        def limit_reached(self):
            """Checks if a limit is reached."""
            if self.call_current is not None:
                self.call_current += 1
                if self.call_current > self.call_limit:
                    return True
            if self.time_start is not None:
                self.time_current = time() - self.time_start
                if self.time_current > self.time_limit:
                    return True
            return False

        def getSolution(self, domains, constraints, vconstraints):
            """Wrap RecursiveBacktrackingSolver.getSolution to add the limits."""
            if self.call_limit is not None:
                self.call_current = 0
            if self.time_limit is not None:
                self.time_start = time()
            return super().getSolution(domains, constraints, vconstraints)

        def recursiveBacktracking(self, solutions, domains, vconstraints, assignments, single):
            """Like ``constraint.RecursiveBacktrackingSolver.recursiveBacktracking`` but
            limited in the amount of calls by ``self.call_limit``"""
            if self.limit_reached():
                return None
            return super().recursiveBacktracking(
                solutions, domains, vconstraints, assignments, single
            )
