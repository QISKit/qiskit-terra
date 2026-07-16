# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if a property reached a fixed point."""

from __future__ import annotations

from copy import deepcopy
import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit
    from qiskit.passmanager.compilation_status import PropertySet

from qiskit.transpiler.basepasses import AnalysisPass


class FixedPoint(AnalysisPass):
    """Check if a property reached a fixed point.

    A dummy analysis pass that checks if a property reached a fixed point.
    The result is saved in ``property_set['<property_to_check>_fixed_point']``
    as a boolean.

    An optional ``getter`` callable can be provided to derive a value from the
    property set instead of reading a single key directly. This is useful
    when convergence should be checked on a combination of properties (e.g. a
    tuple consisting of size and t-count)::

        def get_size_and_t_count(property_set):
            return (
                property_set["size"],
                property_set["count_ops"].get("t", 0) + property_set["count_ops"].get("tdg", 0)
            )

        FixedPoint("size_and_t_count", get_size_and_t_count)
    """

    def __init__(
        self, property_to_check: str, getter: Callable[[PropertySet], Any] | None = None
    ) -> None:
        """
        Args:
            property_to_check: The name used to key the fixed-point result in the property
                set under ``property_set['<property_to_check>_fixed_point']``.
                When ``getter`` is ``None``, this name is also used as a key to
                read the corresponding value from the property set.
            getter: Optional callable that takes the property set as input and
                returns the value to track. When ``None`` (default), the value
                ``property_set[property_to_check]`` is used for tracking purposes.
        """
        super().__init__()
        self._property = property_to_check
        self._getter = getter if getter is not None else operator.itemgetter(property_to_check)

    def run(self, dag: DAGCircuit) -> None:
        """Run the FixedPoint pass on ``dag``."""
        current_value = self._getter(self.property_set)
        fixed_point_previous_property = f"_fixed_point_previous_{self._property}"
        if self.property_set[fixed_point_previous_property] is None:
            self.property_set[f"{self._property}_fixed_point"] = False
        else:
            fixed_point_reached = self.property_set[fixed_point_previous_property] == current_value
            self.property_set[f"{self._property}_fixed_point"] = fixed_point_reached

        self.property_set[fixed_point_previous_property] = deepcopy(current_value)
