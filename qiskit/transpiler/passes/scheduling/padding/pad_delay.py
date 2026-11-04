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

"""Padding pass to insert Delay to the empty slots."""

from qiskit.circuit import Qubit
from qiskit.circuit.delay import Delay
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOutNode
from qiskit.transpiler.target import Target
from qiskit.transpiler.instruction_durations import InstructionDurations

from .base_padding import BasePadding


class PadDelay(BasePadding):
    """Padding idle time with Delay instructions.

    Consecutive delays will be merged in the output of this pass.

    .. plot::
       :include-source:
       :nofigs:

        from qiskit import QuantumCircuit
        from qiskit.transpiler import InstructionDurations

        durations = InstructionDurations([("x", None, 160), ("cx", None, 800)])

        qc = QuantumCircuit(2)
        qc.delay(100, 0)
        qc.x(1)
        qc.cx(0, 1)

    The ASAP-scheduled circuit output may become

    .. code-block:: text

             ┌────────────────┐
        q_0: ┤ Delay(160[dt]) ├──■──
             └─────┬───┬──────┘┌─┴─┐
        q_1: ──────┤ X ├───────┤ X ├
                   └───┘       └───┘

    Note that the additional idle time of 60dt on the ``q_0`` wire coming from the duration difference
    between ``Delay`` of 100dt (``q_0``) and ``XGate`` of 160 dt (``q_1``) is absorbed in
    the delay instruction on the ``q_0`` wire, i.e. in total 160 dt.

    See :class:`BasePadding` pass for details.
    """

    def __init__(
        self,
        fill_very_end: bool = True,
        target: Target = None,
        durations: InstructionDurations = None,
    ):
        """Create new padding delay pass.

        Args:
            fill_very_end: Set ``True`` to fill the end of circuit with delay.
            target: The :class:`~.Target` representing the target backend.
                If it is supplied and does not support delay instruction on a qubit,
                padding passes do not pad any idle time of the qubit.
            durations: The instruction durations. This is mostly for legacy applications without
                a :class:`.Target`. The ``target`` argument should typically be used instead of
                this and if both are specified ``target`` will supersede this argument.
        """
        super().__init__(target=target, durations=durations)
        self.fill_very_end = fill_very_end

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        if not self.fill_very_end and isinstance(next_node, DAGOutNode):
            return

        time_interval = t_end - t_start
        self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag._unit), qubit)
