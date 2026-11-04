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

"""Private utility functions that are used by the builder interfaces."""

from __future__ import annotations

import dataclasses
from typing import TypeVar, TYPE_CHECKING
from collections.abc import Iterable

from qiskit.circuit import (
    ClassicalRegister,
    Clbit,
    QuantumRegister,
)
from qiskit.circuit.classical import expr, types
from qiskit.circuit.exceptions import CircuitError

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit, Register

_ConditionT = TypeVar(
    "_ConditionT", bound=tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr
)


def validate_condition(condition: _ConditionT) -> _ConditionT:
    """Validate that a condition is in a valid format and return it, but raise if it is invalid.

    Args:
        condition: the condition to be tested for validity.  Must be either the legacy 2-tuple
            format, or a :class:`~.expr.Expr` that has `Bool` type.

    Raises:
        CircuitError: if the condition is not in a valid format.

    Returns:
        The same condition as passed, if it was valid.
    """
    if isinstance(condition, expr.Expr):
        if condition.type.kind is not types.Bool:
            raise CircuitError(
                "Classical conditions must be expressions with the type 'Bool()',"
                f" not '{condition.type}'."
            )
        return condition
    try:
        bits, value = condition
        if isinstance(bits, (ClassicalRegister, Clbit)) and isinstance(value, int):
            return (bits, value)
    except (TypeError, ValueError):
        pass
    raise CircuitError(
        "A classical condition should be a 2-tuple of `(ClassicalRegister | Clbit, int)`,"
        f" but received '{condition!r}'."
    )


@dataclasses.dataclass
class LegacyResources:
    """A pair of the :class:`.Clbit` and :class:`.ClassicalRegister` resources used by some other
    object (such as a legacy condition or :class:`.expr.Expr` node)."""

    clbits: tuple[Clbit, ...]
    cregs: tuple[ClassicalRegister, ...]


def node_resources(node: expr.Expr) -> LegacyResources:
    """Get the legacy classical resources (:class:`.Clbit` and :class:`.ClassicalRegister`)
    referenced by an :class:`~.expr.Expr`."""
    # It's generally convenient for us to ensure that the resources are returned in some
    # deterministic order.  This uses the ordering of 'dict' objects to fake out an ordered set.
    clbits = {}
    cregs = {}
    for var in expr.iter_vars(node):
        if isinstance(var.var, Clbit):
            clbits[var.var] = None
        elif isinstance(var.var, ClassicalRegister):
            clbits.update((bit, None) for bit in var.var)
            cregs[var.var] = None
    return LegacyResources(tuple(clbits), tuple(cregs))


def condition_resources(
    condition: tuple[ClassicalRegister, int] | tuple[Clbit, int] | expr.Expr,
) -> LegacyResources:
    """Get the legacy classical resources (:class:`.Clbit` and :class:`.ClassicalRegister`)
    referenced by a legacy condition or an :class:`~.expr.Expr`."""
    if isinstance(condition, expr.Expr):
        return node_resources(condition)
    target, _ = condition
    if isinstance(target, ClassicalRegister):
        return LegacyResources(tuple(target), (target,))
    return LegacyResources((target,), ())


def partition_registers(
    registers: Iterable[Register],
) -> tuple[set[QuantumRegister], set[ClassicalRegister]]:
    """Partition a sequence of registers into its quantum and classical registers."""
    qregs = set()
    cregs = set()
    for register in registers:
        if isinstance(register, QuantumRegister):
            qregs.add(register)
        elif isinstance(register, ClassicalRegister):
            cregs.add(register)
        else:
            # Purely defensive against Terra expansion.
            raise CircuitError(f"Unknown register: {register}.")
    return qregs, cregs


def unify_circuit_resources(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    """
    Ensure that all the given ``circuits`` have all the same qubits, clbits and registers, and
    that they are defined in the same order.  The order is important for binding when the bodies are
    used in the 3-tuple :obj:`.Instruction` context.

    This function will preferentially try to mutate its inputs if they share an ordering, but if
    not, it will rebuild two new circuits.  This is to avoid coupling too tightly to the inner
    class; there is no real support for deleting or re-ordering bits within a :obj:`.QuantumCircuit`
    context, and we don't want to rely on the *current* behavior of the private APIs, since they
    are very liable to change.  No matter the method used, circuits with unified bits and registers
    are returned.
    """
    circuits = tuple(circuits)
    if len(circuits) < 2:
        return circuits
    qubits = []
    clbits = []
    for circuit in circuits:
        if circuit.qubits[: len(qubits)] != qubits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.clbits[: len(qubits)] != clbits:
            return _unify_circuit_resources_rebuild(circuits)
        if circuit.num_qubits > len(qubits):
            qubits = list(circuit.qubits)
        if circuit.num_clbits > len(clbits):
            clbits = list(circuit.clbits)
    for circuit in circuits:
        circuit.add_bits(qubits[circuit.num_qubits :])
        circuit.add_bits(clbits[circuit.num_clbits :])
    return _unify_circuit_registers(circuits)


def _unify_circuit_resources_rebuild(  # (it's too long?!)
    circuits: tuple[QuantumCircuit, ...],
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Ensure that all the given circuits have all the same qubits and clbits, and that they
    are defined in the same order.  The order is important for binding when the bodies are used in
    the 3-tuple :obj:`.Instruction` context.

    This function will always rebuild the objects into new :class:`.QuantumCircuit` instances.
    """

    from qiskit.circuit import QuantumCircuit

    qubits, clbits = set(), set()
    for circuit in circuits:
        qubits.update(circuit.qubits)
        clbits.update(circuit.clbits)
    qubits, clbits = list(qubits), list(clbits)

    # We use the inner `_append` method because everything is already resolved in the builders.
    out_circuits = []
    for circuit in circuits:
        out = QuantumCircuit(
            qubits,
            clbits,
            *circuit.qregs,
            *circuit.cregs,
            global_phase=circuit.global_phase,
            inputs=circuit.iter_input_vars(),
            captures=circuit.iter_captures(),
        )
        for var in circuit.iter_declared_vars():
            out.add_uninitialized_var(var)
        for stretch in circuit.iter_declared_stretches():
            out.add_stretch(stretch)
        for instruction in circuit.data:
            out._append(instruction)
        out_circuits.append(out)
    return _unify_circuit_registers(out_circuits)


def _unify_circuit_registers(circuits: Iterable[QuantumCircuit]) -> Iterable[QuantumCircuit]:
    """
    Ensure that ``true_body`` and ``false_body`` have the same registers defined within them.  These
    do not need to be in the same order between circuits.  The two input circuits are returned,
    mutated to have the same registers.
    """
    circuits = tuple(circuits)
    total_registers = set()
    for circuit in circuits:
        total_registers.update(circuit.qregs)
        total_registers.update(circuit.cregs)
    for circuit in circuits:
        for register in total_registers - set(circuit.qregs) - set(circuit.cregs):
            circuit.add_register(register)
    return circuits
