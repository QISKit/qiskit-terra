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

"""Unify time unit in circuit for scheduling and following passes."""

from qiskit.circuit import Delay, Duration
from qiskit.circuit.classical import expr
from qiskit.circuit.duration import duration_in_dt
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.transpiler.instruction_durations import InstructionDurations
from qiskit.transpiler.target import Target
from qiskit.utils import apply_prefix


class TimeUnitConversion(TransformationPass):
    """Choose a time unit to be used in the following time-aware passes,
    and make all circuit time units consistent with that.

    This pass will add a :attr:`.Instruction.duration` metadata to each op whose duration is known
    which will be used by subsequent scheduling passes for scheduling.

    If ``dt`` (in seconds) is known to transpiler, the unit ``'dt'`` is chosen. Otherwise,
    the unit to be selected depends on what units are used in delays and instruction durations:

    * ``'s'``: if they are all in SI units.
    * ``'dt'``: if they are all in the unit ``'dt'``.
    * raise error: if they are a mix of SI units and ``'dt'``.
    """

    def __init__(self, inst_durations: InstructionDurations = None, target: Target = None):
        """TimeUnitAnalysis initializer.

        Args:
            inst_durations (InstructionDurations): A dictionary of durations of instructions.
            target: The :class:`~.Target` representing the target backend, if both
                  ``inst_durations`` and ``target`` are specified then this argument will take
                  precedence and ``inst_durations`` will be ignored.


        """
        super().__init__()
        self.inst_durations = inst_durations or InstructionDurations()
        if target is not None:
            # The priority order for instruction durations is: target > standalone.
            self.inst_durations = target.durations()
        self._durations_provided = inst_durations is not None or target is not None

    def run(self, dag: DAGCircuit):
        """Run the TimeUnitAnalysis pass on `dag`.

        Args:
            dag (DAGCircuit): DAG to be checked.

        Returns:
            DAGCircuit: DAG with consistent timing and op nodes annotated with duration.

        Raises:
            TranspilerError: if the units are not unifiable
        """

        inst_durations = InstructionDurations()
        if self._durations_provided:
            inst_durations.update(self.inst_durations, getattr(self.inst_durations, "dt", None))

        # The float-value converted units for delay expressions, either all in 'dt'
        # or all in seconds.
        expression_durations = {}

        # Choose unit
        has_dt = False
        has_si = False

        # We _always_ need to traverse duration expressions to convert them to
        # a float. But we also use the opportunity to note if they intermix cycles
        # and wall-time, in case we don't have a `dt` to use to unify all instruction
        # durations.
        for node in dag.op_nodes(op=Delay):
            if isinstance(node.op.duration, expr.Expr):
                if any(
                    isinstance(x, expr.Stretch) for x in expr.iter_identifiers(node.op.duration)
                ):
                    # If any of the delays use a stretch expression, we can't run scheduling
                    # passes anyway, so we bail out. In theory, we _could_ still traverse
                    # through the stretch expression and replace any Duration value nodes it may
                    # contain with ones of the same units, but it'd be complex and probably useless.
                    self.property_set["time_unit"] = "stretch"
                    return dag

                visitor = _EvalDurationImpl(inst_durations.dt)
                duration = node.op.duration.accept(visitor)
                if visitor.in_cycles():
                    has_dt = True
                    # We need to round in case the expression evaluated to a non-integral 'dt'.
                    duration = duration_in_dt(duration, 1.0)
                else:
                    has_si = True
                if duration < 0:
                    raise TranspilerError(
                        f"Expression '{node.op.duration}' resolves to a negative duration!"
                    )
                expression_durations[node._node_id] = duration
            elif node.op.unit == "dt":
                has_dt = True
            else:
                has_si = True
            if inst_durations.dt is None and has_dt and has_si:
                raise TranspilerError(
                    "Fail to unify time units in delays. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )

        if inst_durations.dt is None:
            # Check what units are used in other instructions: dt or SI or mixed
            units_other = inst_durations.units_used()
            unified_unit = self._unified(units_other)
            has_si = has_si or unified_unit in {"SI", "mixed"}
            has_dt = has_dt or unified_unit in {"dt", "mixed"}
            if has_si and has_dt:
                raise TranspilerError(
                    "Fail to unify time units. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )
            if has_si:
                time_unit = "s"
            else:
                # Either dt units were used or no units were used and we
                # default to dt
                time_unit = "dt"
        else:
            time_unit = "dt"

        # Make instructions with local durations consistent.
        for node in dag.op_nodes(Delay):
            op = node.op.to_mutable()
            if node._node_id in expression_durations:
                op.duration = expression_durations[node._node_id]
            else:
                op.duration = inst_durations._convert_unit(op.duration, op.unit, time_unit)
            op.unit = time_unit
            dag.substitute_node(node, op)

        self.property_set["time_unit"] = time_unit
        return dag

    @staticmethod
    def _unified(unit_set: set[str]) -> str:
        if not unit_set:
            return "none"

        if len(unit_set) == 1 and "dt" in unit_set:
            return "dt"

        all_si = True
        for unit in unit_set:
            if not unit.endswith("s"):
                all_si = False
                break

        if all_si:
            return "SI"

        return "mixed"


class _EvalDurationImpl(expr.ExprVisitor[float]):
    """Evaluates the expression to a single float result.

    If ``dt`` is provided or all durations are already in ``dt``, the result is in ``dt``.
    Otherwise, the result will be in seconds, and all durations MUST be in wall-time (SI).
    """

    __slots__ = ("dt", "has_dt", "has_si")

    def __init__(self, dt=None):
        self.dt = dt
        self.has_dt = False
        self.has_si = False

    def in_cycles(self):
        """Returns ``True`` if units are 'dt' after visit."""
        return self.has_dt or self.dt is not None

    def visit_value(self, node, /) -> float:
        if isinstance(node.value, float):
            return node.value
        if isinstance(node.value, Duration.dt):
            if self.has_si and self.dt is None:
                raise TranspilerError(
                    "Fail to unify time units in delays. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )
            self.has_dt = True
            return node.value[0]
        if isinstance(node.value, Duration):
            if self.has_dt and self.dt is None:
                raise TranspilerError(
                    "Fail to unify time units in delays. SI units "
                    "and dt unit must not be mixed when dt is not supplied."
                )
            self.has_si = True
            # Setting 'divisor' to 1 when there's no 'dt' is just to simplify
            # the logic (we don't need to divide).
            divisor = self.dt if self.dt is not None else 1
            if isinstance(node.value, Duration.s):
                return node.value[0] / divisor
            from_unit = node.value.unit()
            return apply_prefix(node.value[0], from_unit) / divisor
        raise TranspilerError(f"invalid duration expression: {node}")

    def visit_binary(self, node, /) -> float:
        left = node.left.accept(self)
        right = node.right.accept(self)
        if node.op == expr.Binary.Op.ADD:
            return left + right
        if node.op == expr.Binary.Op.SUB:
            return left - right
        if node.op == expr.Binary.Op.MUL:
            return left * right
        if node.op == expr.Binary.Op.DIV:
            return left / right
        raise TranspilerError(f"invalid duration expression: {node}")

    def visit_cast(self, node, /) -> float:
        return node.operand.accept(self)
