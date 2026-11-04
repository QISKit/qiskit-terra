# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Check if the DAG contains a specific instruction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.basepasses import AnalysisPass


class ContainsInstruction(AnalysisPass):
    """An analysis pass to detect if the DAG contains a specific instruction.

    This pass takes in a single instruction name for example ``'delay'`` and
    will set the property set ``contains_delay`` to ``True`` if the DAG contains
    that instruction and ``False`` if it does not.
    """

    def __init__(self, instruction_name: str | Iterable[str], recurse: bool = True) -> None:
        """
        Args:
            instruction_name: The instruction or instructions to check are in
                the DAG. The output in the property set is set to ``contains_`` prefixed on each
                value for this parameter.
            recurse: if ``True`` (default), then recurse into control-flow operations.
        """
        super().__init__()
        self._instruction_names = (
            {instruction_name} if isinstance(instruction_name, str) else set(instruction_name)
        )
        self._recurse = recurse

    def run(self, dag: DAGCircuit) -> None:
        """Run the ContainsInstruction pass on ``dag``."""
        names = dag.count_ops(recurse=self._recurse)
        for name in self._instruction_names:
            self.property_set[f"contains_{name}"] = name in names
