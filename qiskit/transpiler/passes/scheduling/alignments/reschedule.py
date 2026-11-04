# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Rescheduler pass to adjust node start times."""
from __future__ import annotations
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.target import Target
from qiskit._accelerate.constrained_reschedule import constrained_reschedule


class ConstrainedReschedule(AnalysisPass):
    """Rescheduler pass that updates node start times to conform to the hardware alignments.

    This pass shifts DAG node start times previously scheduled with one of
    the scheduling passes, e.g. :class:`ASAPScheduleAnalysis` or :class:`ALAPScheduleAnalysis`,
    so that every instruction start time satisfies alignment constraints.

    Examples:

        We assume executing the following circuit on a backend with 16 dt of acquire alignment.

        .. code-block:: text

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(100[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

        Note that delay of 100 dt induces a misalignment of 4 dt at the measurement.
        This pass appends an extra 12 dt time shift to the input circuit.

        .. code-block:: text

                 ┌───┐┌────────────────┐┌─┐
            q_0: ┤ X ├┤ Delay(112[dt]) ├┤M├
                 └───┘└────────────────┘└╥┘
            c: 1/════════════════════════╩═
                                         0

    Notes:

        Your backend may execute circuits violating these alignment constraints.
        However, you may obtain erroneous measurement result because of the
        untracked phase originating in the instruction misalignment.
    """

    def __init__(
        self,
        acquire_alignment: int = 1,
        pulse_alignment: int = 1,
        target: Target = None,
    ):
        """Create new rescheduler pass.

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
        self.target = target
        if target is not None:
            self.durations = target.durations()
            self.acquire_align = target.acquire_alignment
            self.pulse_align = target.pulse_alignment

    def run(self, dag: DAGCircuit):
        """Run rescheduler.

        This pass should perform rescheduling to satisfy:

            - All DAGOpNode nodes (except for compiler directives) are placed at start time
              satisfying hardware alignment constraints.
            - The end time of a node does not overlap with the start time of successor nodes.

        Assumptions:

            - Topological order and absolute time order of DAGOpNode are consistent.
            - All bits in either qargs or cargs associated with node synchronously start.
            - Start time of qargs and cargs may different due to I/O latency.

        Based on the configurations above, the rescheduler pass takes the following strategy:

        1. The nodes are processed in the topological order, from the beginning of
            the circuit (i.e. from left to right). For every node (including compiler
            directives), the function ``_push_node_back`` performs steps 2 and 3.
        2. If the start time of the node violates the alignment constraint,
            the start time is increased to satisfy the constraint.
        3. Each immediate successor whose start_time overlaps the node's end_time is
            pushed backwards (towards the end of the wire). Note that at this point
            the shifted successor does not need to satisfy the constraints, but this
            will be taken care of when that successor node itself is processed.
        4. After every node is processed, all misalignment constraints will be resolved,
            and there will be no overlap between the nodes.

        Args:
            dag: DAG circuit to be rescheduled with constraints.

        Raises:
            TranspilerError: If circuit is not scheduled.
        """

        if "node_start_time" not in self.property_set:
            raise TranspilerError(
                f"The input circuit {dag.name} is not scheduled. Call one of scheduling passes "
                f"before running the {self.__class__.__name__} pass."
            )
        node_start_time = self.property_set["node_start_time"]
        clbit_write_latency = self.property_set.get("clbit_write_latency", 0)
        self.property_set["node_start_time"] = constrained_reschedule(
            dag,
            node_start_time,
            clbit_write_latency,
            self.acquire_align,
            self.pulse_align,
            self.target,
        )
