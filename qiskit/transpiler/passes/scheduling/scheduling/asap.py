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

"""ASAP Scheduling."""
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.passes.scheduling.scheduling.base_scheduler import BaseScheduler
from qiskit._accelerate.asap_schedule_analysis import asap_schedule_analysis
from qiskit._accelerate.scheduling import NodeDurations


class ASAPScheduleAnalysis(BaseScheduler):
    """ASAP Scheduling pass, which schedules the start time of instructions as early as possible.

    See the :ref:`transpiler-scheduling-description` section in the :mod:`qiskit.transpiler`
    module documentation for a more detailed description.
    """

    def run(self, dag):
        """Run the ASAPSchedule pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to schedule.

        Returns:
            DAGCircuit: A scheduled DAG.

        Raises:
            TranspilerError: if the circuit is not mapped on physical qubits.
            TranspilerError: if conditional bit is added to non-supported instruction.
        """

        if self.property_set["time_unit"] == "stretch":
            raise TranspilerError("Scheduling cannot run on circuits with stretch durations.")

        node_durations = NodeDurations(
            {node: self._get_node_duration(node, dag) for node in dag.topological_op_nodes()}
        )
        clbit_write_latency = self.property_set.get("clbit_write_latency", 0)
        self.property_set["node_start_time"] = asap_schedule_analysis(
            dag, clbit_write_latency, node_durations
        )
