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
"""Set classical IO latency information to circuit."""

from qiskit.transpiler.basepasses import AnalysisPass
from qiskit.dagcircuit import DAGCircuit


class SetIOLatency(AnalysisPass):
    """Set IOLatency information to the input circuit.

    The ``clbit_write_latency`` and ``conditional_latency`` are added to
    the property set of pass manager. This information can be shared among the passes
    that perform scheduling on instructions acting on classical registers.

    Once these latencies are added to the property set, this information
    is also copied to the output circuit object as protected attributes,
    so that it can be utilized outside the transpilation,
    for example, the timeline visualization can use latency to accurately show
    time occupation by instructions on the classical registers.
    """

    def __init__(
        self,
        clbit_write_latency: int = 0,
        conditional_latency: int = 0,
    ):
        """Create pass with latency information.

        Args:
            clbit_write_latency: A control flow constraint. Because standard superconducting
                quantum processor implement dispersive QND readout, the actual data transfer
                to the clbit happens after the round-trip stimulus signal is buffered
                and discriminated into quantum state.
                The interval ``[t0, t0 + clbit_write_latency]`` is regarded as idle time
                for clbits associated with the measure instruction.
                This defaults to 0 dt which is identical to Qiskit Pulse scheduler.
            conditional_latency: A control flow constraint. This value represents
                a latency of reading a classical register for the conditional operation.
                The gate operation occurs after this latency. This appears as a delay
                in front of the DAGOpNode of the gate.
                This defaults to 0 dt.
        """
        super().__init__()
        self._conditional_latency = conditional_latency
        self._clbit_write_latency = clbit_write_latency

    def run(self, dag: DAGCircuit):
        """Add IO latency information.

        Args:
            dag: Input DAG circuit.
        """
        self.property_set["conditional_latency"] = self._conditional_latency
        self.property_set["clbit_write_latency"] = self._clbit_write_latency
