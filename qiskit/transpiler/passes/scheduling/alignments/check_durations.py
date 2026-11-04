# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""A pass to check if input circuit requires reschedule."""

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.target import Target
from qiskit._accelerate.instruction_duration_check import run_instruction_duration_check


class InstructionDurationCheck(AnalysisPass):
    """Duration validation pass for reschedule.

    This pass investigates the input quantum circuit and checks if the circuit requires
    rescheduling for execution. Note that this pass can be triggered without scheduling.
    This pass only checks the duration of delay instructions,
    which report duration values without pre-scheduling.

    This pass assumes backend supported instructions, i.e. basis gates, have no violation
    of the hardware alignment constraints, which is true in general.
    """

    def __init__(self, acquire_alignment: int = 1, pulse_alignment: int = 1, target: Target = None):
        """Create new duration validation pass.

        The alignment values depend on the control electronics of your quantum processor.

        Args:
            acquire_alignment: Integer number representing the minimum time resolution to
                trigger acquisition instruction in units of ``dt``.
            pulse_alignment: Integer number representing the minimum time resolution to
                trigger gate instruction in units of ``dt``.
            target: The :class:`~.Target` representing the target backend, if
                ``target`` is specified then this argument will take
                precedence and ``acquire_alignment`` and ``pulse_alignment`` will be ignored.
        """
        super().__init__()
        self.acquire_align = acquire_alignment
        self.pulse_align = pulse_alignment
        if target is not None:
            self.acquire_align = target.acquire_alignment
            self.pulse_align = target.pulse_alignment

    def run(self, dag: DAGCircuit):
        """Run duration validation passes.

        Args:
            dag: DAG circuit to check instruction durations.
        """
        self.property_set["reschedule_required"] = run_instruction_duration_check(
            dag, self.acquire_align, self.pulse_align
        )
