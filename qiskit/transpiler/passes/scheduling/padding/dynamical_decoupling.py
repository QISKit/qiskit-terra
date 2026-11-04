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

"""Dynamical Decoupling insertion pass."""
from __future__ import annotations

import logging
import numpy as np

from qiskit.circuit import Gate, ParameterExpression, Qubit
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.standard_gates import IGate, UGate, U3Gate
from qiskit.circuit.reset import Reset
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGInNode, DAGOpNode, DAGOutNode
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.passes.optimization import Optimize1qGates
from qiskit.transpiler.passes.scheduling.padding.base_padding import BasePadding
from qiskit.transpiler.target import Target


logger = logging.getLogger(__name__)


class PadDynamicalDecoupling(BasePadding):
    """Dynamical decoupling insertion pass.

    This pass works on a scheduled, physical circuit. It scans the circuit for
    idle periods of time (i.e. those containing delay instructions) and inserts
    a DD sequence of gates in those spots. These gates amount to the identity,
    so do not alter the logical action of the circuit, but have the effect of
    mitigating decoherence in those idle periods.

    As a special case, the pass allows a length-1 sequence (e.g. ``[XGate()]``).
    In this case the DD insertion happens only when the gate inverse can be
    absorbed into a neighboring gate in the circuit (so we would still be
    replacing Delay with something that is equivalent to the identity).
    This can be used, for instance, as a Hahn echo.

    This pass ensures that the inserted sequence preserves the circuit exactly
    (including global phase).

    .. plot::
       :alt: Output from the previous code.
       :include-source:

        import numpy as np
        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import XGate
        from qiskit.transpiler import PassManager, InstructionDurations, Target, CouplingMap
        from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling
        from qiskit.visualization import timeline_drawer

        circ = QuantumCircuit(4)
        circ.h(0)
        circ.cx(0, 1)
        circ.cx(1, 2)
        circ.cx(2, 3)
        circ.measure_all()
        durations = InstructionDurations(
            [("h", 0, 50), ("cx", [0, 1], 700), ("reset", None, 10),
             ("cx", [1, 2], 200), ("cx", [2, 3], 300),
             ("x", None, 50), ("measure", None, 1000)],
            dt=1e-7
        )
        target = Target.from_configuration(
            ["h", "cx", "reset", "x", "measure"],
            num_qubits=4,
            coupling_map=CouplingMap.from_line(4, bidirectional=False),
            instruction_durations=durations,
            dt=1e-7,
        )

        # balanced X-X sequence on all qubits
        dd_sequence = [XGate(), XGate()]
        pm = PassManager([ALAPScheduleAnalysis(durations),
                          PadDynamicalDecoupling(durations, dd_sequence)])
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd, target=target)

        # Uhrig sequence on qubit 0
        n = 8
        dd_sequence = [XGate()] * n
        def uhrig_pulse_location(k):
            return np.sin(np.pi * (k + 1) / (2 * n + 2)) ** 2
        spacing = []
        for k in range(n):
            spacing.append(uhrig_pulse_location(k) - sum(spacing))
        spacing.append(1 - sum(spacing))
        pm = PassManager(
            [
                ALAPScheduleAnalysis(durations),
                PadDynamicalDecoupling(durations, dd_sequence, qubits=[0], spacing=spacing),
            ]
        )
        circ_dd = pm.run(circ)
        timeline_drawer(circ_dd, target=target)

    .. note::

        You may need to call alignment pass before running dynamical decoupling to guarantee
        your circuit satisfies acquisition alignment constraints.
    """

    def __init__(
        self,
        durations: InstructionDurations = None,
        dd_sequence: list[Gate] | None = None,
        qubits: list[int] | None = None,
        spacing: list[float] | None = None,
        skip_reset_qubits: bool = True,
        pulse_alignment: int = 1,
        extra_slack_distribution: str = "middle",
        target: Target = None,
    ):
        """Dynamical decoupling initializer.

        Args:
            durations: Durations of instructions to be used in scheduling.
            dd_sequence: Sequence of gates to apply in idle spots.
            qubits: Physical qubits on which to apply DD.
                If None, all qubits will undergo DD (when possible).
            spacing: A list of spacings between the DD gates.
                The available slack will be divided according to this.
                The list length must be one more than the length of dd_sequence,
                and the elements must sum to 1. If None, a balanced spacing
                will be used [d/2, d, d, ..., d, d, d/2].
            skip_reset_qubits: If True, does not insert DD on idle periods that
                immediately follow initialized/reset qubits
                (as qubits in the ground state are less susceptible to decoherence).
            pulse_alignment: The hardware constraints for gate timing allocation.
                This is usually provided from ``backend.configuration().timing_constraints``.
                If provided, the delay length, i.e. ``spacing``, is implicitly adjusted to
                satisfy this constraint.
            extra_slack_distribution: The option to control the behavior of DD sequence generation.
                The duration of the DD sequence should be identical to an idle time in the
                scheduled quantum circuit, however, the delay in between gates comprising the sequence
                should be integer number in units of dt, and it might be further truncated
                when ``pulse_alignment`` is specified. This sometimes results in the duration of
                the created sequence being shorter than the idle time
                that you want to fill with the sequence, i.e. `extra slack`.
                This option takes following values.

                    - "middle": Put the extra slack to the interval at the middle of the sequence.
                    - "edges": Divide the extra slack as evenly as possible into
                      intervals at beginning and end of the sequence.
            target: The :class:`~.Target` representing the target backend.
                Target takes precedence over other arguments when they can be inferred from target.
                Therefore specifying target as well as other arguments like ``durations`` or
                ``pulse_alignment`` will cause those other arguments to be ignored.

        Raises:
            TranspilerError: When invalid DD sequence is specified.
            TranspilerError: When pulse gate with the duration which is
                non-multiple of the alignment constraint value is found.
            TypeError: If ``dd_sequence`` is not specified
        """
        super().__init__(target=target, durations=durations)
        self._durations = durations
        if dd_sequence is None:
            raise TypeError("required argument 'dd_sequence' is not specified")
        self._dd_sequence = dd_sequence
        self._qubits = qubits
        self._skip_reset_qubits = skip_reset_qubits
        self._alignment = pulse_alignment
        self._spacing = spacing
        self._extra_slack_distribution = extra_slack_distribution

        self._no_dd_qubits: set[int] = set()
        self._dd_sequence_lengths: dict[Qubit, list[int]] = {}
        self._sequence_phase = 0
        if target is not None:
            # The priority order for instruction durations is: target > standalone.
            self._durations = target.durations()
            self._alignment = target.pulse_alignment
            for gate in dd_sequence:
                if gate.name not in target.operation_names:
                    raise TranspilerError(
                        f"{gate.name} in dd_sequence is not supported in the target"
                    )

    def _pre_runhook(self, dag: DAGCircuit):
        super()._pre_runhook(dag)

        durations = InstructionDurations()
        if self._durations is not None:
            durations.update(self._durations, getattr(self._durations, "dt", None))
        num_pulses = len(self._dd_sequence)

        # Check if physical circuit is given
        if len(dag.qregs) != 1 or dag.qregs.get("q", None) is None:
            raise TranspilerError("DD runs on physical circuits only.")

        # Set default spacing otherwise validate user input
        if self._spacing is None:
            mid = 1 / num_pulses
            end = mid / 2
            self._spacing = [end] + [mid] * (num_pulses - 1) + [end]
        elif sum(self._spacing) != 1 or any(a < 0 for a in self._spacing):
            raise TranspilerError(
                "The spacings must be given in terms of fractions "
                "of the slack period and sum to 1."
            )

        # Check if DD sequence is identity
        if num_pulses != 1:
            if num_pulses % 2 != 0:
                raise TranspilerError("DD sequence must contain an even number of gates (or 1).")
            noop = np.eye(2)
            for gate in self._dd_sequence:
                noop = noop.dot(gate.to_matrix())
            if not matrix_equal(noop, IGate().to_matrix(), ignore_phase=True):
                raise TranspilerError("The DD sequence does not make an identity operation.")
            self._sequence_phase = np.angle(noop[0][0])

        # Compute no DD qubits on which any gate in dd_sequence is not supported in the target
        for qarg, _ in enumerate(dag.qubits):
            for gate in self._dd_sequence:
                if not self.__gate_supported(gate, qarg):
                    self._no_dd_qubits.add(qarg)
                    logger.debug(
                        "No DD on qubit %d as gate %s is not supported on it", qarg, gate.name
                    )
                    break
        # Precompute qubit-wise DD sequence length for performance
        for physical_index, qubit in enumerate(dag.qubits):
            if not self.__is_dd_qubit(physical_index):
                continue

            sequence_lengths = []
            for index, gate in enumerate(self._dd_sequence):
                gate_length = durations.get(gate, physical_index)
                sequence_lengths.append(gate_length)
                # Update gate duration. This is necessary for current timeline drawer, i.e. scheduled.
                gate = gate.to_mutable()
                self._dd_sequence[index] = gate
                gate.duration = gate_length
            self._dd_sequence_lengths[qubit] = sequence_lengths

    def __gate_supported(self, gate: Gate, qarg: int) -> bool:
        """A gate is supported on the qubit (qarg) or not."""
        if self.target is None or self.target.instruction_supported(gate.name, qargs=(qarg,)):
            return True
        return False

    def __is_dd_qubit(self, qubit_index: int) -> bool:
        """DD can be inserted in the qubit or not."""
        if (qubit_index in self._no_dd_qubits) or (
            self._qubits and qubit_index not in self._qubits
        ):
            return False
        return True

    def _pad(
        self,
        dag: DAGCircuit,
        qubit: Qubit,
        t_start: int,
        t_end: int,
        next_node: DAGNode,
        prev_node: DAGNode,
    ):
        # This routine takes care of the pulse alignment constraint for the DD sequence.
        # Note that the alignment constraint acts on the t0 of the DAGOpNode.
        # Now this constrained scheduling problem is simplified to the problem of
        # finding a delay amount which is a multiple of the constraint value by assuming
        # that the duration of every DAGOpNode is also a multiple of the constraint value.
        #
        # For example, given the constraint value of 16 and XY4 with 160 dt gates.
        # Here we assume current interval is 992 dt.
        #
        # relative spacing := [0.125, 0.25, 0.25, 0.25, 0.125]
        # slack = 992 dt - 4 x 160 dt = 352 dt
        #
        # unconstraind sequence: 44dt-X1-88dt-Y2-88dt-X3-88dt-Y4-44dt
        # constrained sequence  : 32dt-X1-80dt-Y2-80dt-X3-80dt-Y4-32dt + extra slack 48 dt
        #
        # Now we evenly split extra slack into start and end of the sequence.
        # The distributed slack should be multiple of 16.
        # Start = +16, End += 32
        #
        # final sequence       : 48dt-X1-80dt-Y2-80dt-X3-80dt-Y4-64dt / in total 992 dt
        #
        # Now we verify t0 of every node starts from multiple of 16 dt.
        #
        # X1:  48 dt (3 x 16 dt)
        # Y2:  48 dt + 160 dt + 80 dt = 288 dt (18 x 16 dt)
        # Y3: 288 dt + 160 dt + 80 dt = 528 dt (33 x 16 dt)
        # Y4: 368 dt + 160 dt + 80 dt = 768 dt (48 x 16 dt)
        #
        # As you can see, constraints on t0 are all satisfied without explicit scheduling.
        time_interval = t_end - t_start
        if time_interval % self._alignment != 0:
            raise TranspilerError(
                f"Time interval {time_interval} is not divisible by alignment {self._alignment} "
                f"between {_format_node(prev_node)} and {_format_node(next_node)}."
            )

        if not self.__is_dd_qubit(dag.qubits.index(qubit)):
            # Target physical qubit is not the target of this DD sequence.
            self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag._unit), qubit)
            return

        if self._skip_reset_qubits and (
            isinstance(prev_node, DAGInNode) or isinstance(prev_node.op, Reset)
        ):
            # Previous node is the start edge or reset, i.e. qubit is ground state.
            self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag._unit), qubit)
            return

        slack = time_interval - np.sum(self._dd_sequence_lengths[qubit])
        sequence_gphase = self._sequence_phase

        if slack <= 0:
            # Interval too short.
            self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag._unit), qubit)
            return

        if len(self._dd_sequence) == 1:
            # Special case of using a single gate for DD
            u_inv = self._dd_sequence[0].inverse().to_matrix()
            theta, phi, lam, phase = OneQubitEulerDecomposer().angles_and_phase(u_inv)
            if isinstance(next_node, DAGOpNode) and isinstance(next_node.op, (UGate, U3Gate)):
                # Absorb the inverse into the successor (from left in circuit)
                op = next_node.op
                theta_r, phi_r, lam_r = op.params
                op.params = Optimize1qGates.compose_u3(theta_r, phi_r, lam_r, theta, phi, lam)
                next_node.op = op
                sequence_gphase += phase
            elif isinstance(prev_node, DAGOpNode) and isinstance(prev_node.op, (UGate, U3Gate)):
                # Absorb the inverse into the predecessor (from right in circuit)
                op = prev_node.op
                theta_l, phi_l, lam_l = op.params
                op.params = Optimize1qGates.compose_u3(theta, phi, lam, theta_l, phi_l, lam_l)
                new_prev_node = dag.substitute_node(prev_node, op)
                start_time = self.property_set["node_start_time"].pop(prev_node)
                if start_time is not None:
                    self.property_set["node_start_time"][new_prev_node] = start_time
                sequence_gphase += phase
            else:
                # Don't do anything if there's no single-qubit gate to absorb the inverse
                self._apply_scheduled_op(dag, t_start, Delay(time_interval, dag._unit), qubit)
                return

        def _constrained_length(values):
            return self._alignment * np.floor(values / self._alignment)

        # (1) Compute DD intervals satisfying the constraint
        taus = _constrained_length(slack * np.asarray(self._spacing))
        extra_slack = slack - np.sum(taus)

        # (2) Distribute extra slack
        if self._extra_slack_distribution == "middle":
            mid_ind = int((len(taus) - 1) / 2)
            to_middle = _constrained_length(extra_slack)
            taus[mid_ind] += to_middle
            if extra_slack - to_middle:
                # If to_middle is not a multiple value of the pulse alignment,
                # it is truncated to the nearlest multiple value and
                # the rest of slack is added to the end.
                taus[-1] += extra_slack - to_middle
        elif self._extra_slack_distribution == "edges":
            to_begin_edge = _constrained_length(extra_slack / 2)
            taus[0] += to_begin_edge
            taus[-1] += extra_slack - to_begin_edge
        else:
            raise TranspilerError(
                f"Option extra_slack_distribution = {self._extra_slack_distribution} is invalid."
            )

        # (3) Construct DD sequence with delays
        num_elements = max(len(self._dd_sequence), len(taus))
        idle_after = t_start
        for dd_ind in range(num_elements):
            if dd_ind < len(taus):
                tau = taus[dd_ind]
                if tau > 0:
                    self._apply_scheduled_op(dag, idle_after, Delay(tau, dag._unit), qubit)
                    # Cast tau to int from np.float64 to avoid type changes
                    idle_after += int(tau)
            if dd_ind < len(self._dd_sequence):
                gate = self._dd_sequence[dd_ind]
                gate_length = self._dd_sequence_lengths[qubit][dd_ind]
                self._apply_scheduled_op(dag, idle_after, gate, qubit)
                idle_after += gate_length
        dag.global_phase = dag.global_phase + sequence_gphase

    @staticmethod
    def _resolve_params(gate: Gate) -> tuple:
        """Return gate params with any bound parameters replaced with floats"""
        params = []
        for p in gate.params:
            if isinstance(p, ParameterExpression) and not p.parameters:
                params.append(float(p))
            else:
                params.append(p)
        return tuple(params)


def _format_node(node: DAGNode) -> str:
    """Util to format the DAGNode, DAGInNode, and DAGOutNode."""
    if isinstance(node, (DAGInNode, DAGOutNode)):
        return f"{node.__class__.__name__} on qarg {node.wire}"
    return f"DAGNode {node.name} on qargs {node.qargs}"
