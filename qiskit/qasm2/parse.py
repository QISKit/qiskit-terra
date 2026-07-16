# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Python-space bytecode interpreter for the output of the main Rust parser logic."""
import dataclasses
import math
from collections.abc import Iterable, Callable
from typing_extensions import Unpack

import numpy as np

from qiskit.circuit import (
    Barrier,
    CircuitInstruction,
    ClassicalRegister,
    Delay,
    Gate,
    Instruction,
    Measure,
    QuantumCircuit,
    QuantumRegister,
    Qubit,
    Reset,
    library as lib,
)
from qiskit.quantum_info import Operator
from qiskit._accelerate.qasm2 import (
    OpCode,
    UnaryOpCode,
    BinaryOpCode,
    CustomClassical,
    ExprConstant,
    ExprArgument,
    ExprUnary,
    ExprBinary,
    ExprCustom,
)
from .exceptions import QASM2ParseError

# Constructors of the form `*params -> Gate` for the special 'qelib1.inc' include.  This is
# essentially a pre-parsed state for the file as given in the arXiv paper defining OQ2.
QELIB1 = (
    lib.U3Gate,
    lib.U2Gate,
    lib.U1Gate,
    lib.CXGate,
    # IGate in Terra < 0.24 is defined as a single-cycle delay, so is not strictly an identity.
    # Instead we use a true no-op stand-in.
    lambda: lib.UGate(0, 0, 0),
    lib.XGate,
    lib.YGate,
    lib.ZGate,
    lib.HGate,
    lib.SGate,
    lib.SdgGate,
    lib.TGate,
    lib.TdgGate,
    lib.RXGate,
    lib.RYGate,
    lib.RZGate,
    lib.CZGate,
    lib.CYGate,
    lib.CHGate,
    lib.CCXGate,
    lib.CRZGate,
    lib.CU1Gate,
    lib.CU3Gate,
)


@dataclasses.dataclass(frozen=True)
class CustomInstruction:
    """Information about a custom instruction that should be defined during the parse.

    The ``name``, ``num_params`` and ``num_qubits`` fields are self-explanatory.  The
    ``constructor`` field should be a callable object with signature ``*args -> Instruction``, where
    each of the ``num_params`` ``args`` is a floating-point value.  Most of the built-in Qiskit gate
    classes have this form.

    There is a final ``builtin`` field.  This is optional, and if set true will cause the
    instruction to be defined and available within the parsing, even if there is no definition in
    any included OpenQASM 2 file.

    Examples:

        Instruct the importer to use Qiskit's :class:`.ECRGate` and :class:`.RZXGate` objects to
        interpret ``gate`` statements that are known to have been created from those same objects
        during OpenQASM 2 export::

            from qiskit import qasm2
            from qiskit.circuit import QuantumCircuit, library

            qc = QuantumCircuit(2)
            qc.ecr(0, 1)
            qc.rzx(0.3, 0, 1)
            qc.rzx(0.7, 1, 0)
            qc.rzx(1.5, 0, 1)
            qc.ecr(1, 0)

            # This output string includes `gate ecr q0, q1 { ... }` and `gate rzx(p) q0, q1 { ... }`
            # statements, since `ecr` and `rzx` are neither built-in gates nor in ``qelib1.inc``.
            dumped = qasm2.dumps(qc)

            # Tell the importer how to interpret the `gate` statements, which we know are safe
            # because we controlled the input OpenQASM 2 source.
            custom = [
                qasm2.CustomInstruction("ecr", 0, 2, library.ECRGate),
                qasm2.CustomInstruction("rzx", 1, 2, library.RZXGate),
            ]

            loaded = qasm2.loads(dumped, custom_instructions=custom)
    """

    name: str
    num_params: int
    num_qubits: int
    constructor: Callable[[Unpack[tuple[float, ...]]], Instruction]
    builtin: bool = False


def _generate_delay(time: float):
    # This wrapper is just to ensure that the correct type of exception gets emitted; it would be
    # unnecessarily spaghetti-ish to check every emitted instruction in Rust for integer
    # compatibility, but only if its name is `delay` _and_ its constructor wraps Qiskit's `Delay`.
    if int(time) != time:
        raise QASM2ParseError("the custom 'delay' instruction can only accept an integer parameter")
    return Delay(int(time), unit="dt")


class _U0Gate(Gate):
    def __init__(self, count):
        if int(count) != count:
            raise QASM2ParseError("the number of single-qubit delay lengths must be an integer")
        super().__init__("u0", 1, [int(count)])

    def _define(self):
        self._definition = QuantumCircuit(1)
        for _ in [None] * self.params[0]:
            self._definition.id(0)


LEGACY_CUSTOM_INSTRUCTIONS = (
    CustomInstruction("u3", 3, 1, lib.U3Gate),
    CustomInstruction("u2", 2, 1, lib.U2Gate),
    CustomInstruction("u1", 1, 1, lib.U1Gate),
    CustomInstruction("cx", 0, 2, lib.CXGate),
    # The Qiskit parser emits IGate for 'id', even if that is not strictly accurate in Terra < 0.24.
    CustomInstruction("id", 0, 1, lib.IGate),
    CustomInstruction("u0", 1, 1, _U0Gate, builtin=True),
    CustomInstruction("u", 3, 1, lib.UGate, builtin=True),
    CustomInstruction("p", 1, 1, lib.PhaseGate, builtin=True),
    CustomInstruction("x", 0, 1, lib.XGate),
    CustomInstruction("y", 0, 1, lib.YGate),
    CustomInstruction("z", 0, 1, lib.ZGate),
    CustomInstruction("h", 0, 1, lib.HGate),
    CustomInstruction("s", 0, 1, lib.SGate),
    CustomInstruction("sdg", 0, 1, lib.SdgGate),
    CustomInstruction("t", 0, 1, lib.TGate),
    CustomInstruction("tdg", 0, 1, lib.TdgGate),
    CustomInstruction("rx", 1, 1, lib.RXGate),
    CustomInstruction("ry", 1, 1, lib.RYGate),
    CustomInstruction("rz", 1, 1, lib.RZGate),
    CustomInstruction("sx", 0, 1, lib.SXGate, builtin=True),
    CustomInstruction("sxdg", 0, 1, lib.SXdgGate, builtin=True),
    CustomInstruction("cz", 0, 2, lib.CZGate),
    CustomInstruction("cy", 0, 2, lib.CYGate),
    CustomInstruction("swap", 0, 2, lib.SwapGate, builtin=True),
    CustomInstruction("ch", 0, 2, lib.CHGate),
    CustomInstruction("ccx", 0, 3, lib.CCXGate),
    CustomInstruction("cswap", 0, 3, lib.CSwapGate, builtin=True),
    CustomInstruction("crx", 1, 2, lib.CRXGate, builtin=True),
    CustomInstruction("cry", 1, 2, lib.CRYGate, builtin=True),
    CustomInstruction("crz", 1, 2, lib.CRZGate),
    CustomInstruction("cu1", 1, 2, lib.CU1Gate),
    CustomInstruction("cp", 1, 2, lib.CPhaseGate, builtin=True),
    CustomInstruction("cu3", 3, 2, lib.CU3Gate),
    CustomInstruction("csx", 0, 2, lib.CSXGate, builtin=True),
    CustomInstruction("cu", 4, 2, lib.CUGate, builtin=True),
    CustomInstruction("rxx", 1, 2, lib.RXXGate, builtin=True),
    CustomInstruction("rzz", 1, 2, lib.RZZGate, builtin=True),
    CustomInstruction("rccx", 0, 3, lib.RCCXGate, builtin=True),
    CustomInstruction("rc3x", 0, 4, lib.RC3XGate, builtin=True),
    CustomInstruction("c3x", 0, 4, lib.C3XGate, builtin=True),
    CustomInstruction("c3sqrtx", 0, 4, lib.C3SXGate, builtin=True),
    CustomInstruction("c4x", 0, 5, lib.C4XGate, builtin=True),
    CustomInstruction("delay", 1, 1, _generate_delay),
)

LEGACY_CUSTOM_CLASSICAL = tuple(CustomClassical.builtins())


def from_bytecode(bytecode, custom_instructions: Iterable[CustomInstruction]):
    """Loop through the Rust bytecode iterator `bytecode` producing a
    :class:`~qiskit.circuit.QuantumCircuit` instance from it.  All the hard work is done in Rust
    space where operations are faster; here, we're just about looping through the instructions as
    fast as possible, doing as little calculation as we can in Python space.  The Python-space
    components are the vast majority of the runtime.

    The "bytecode", and so also this Python function, is very tightly coupled to the output of the
    Rust parser.  The bytecode itself is largely defined by Rust; from Python space, the iterator is
    over essentially a 2-tuple of `(opcode, operands)`.  The `operands` are fixed by Rust, and
    assumed to be correct by this function.

    The Rust code is responsible for all validation.  If this function causes any errors to be
    raised by Qiskit (except perhaps for some symbolic manipulations of `Parameter` objects), we
    should consider that a bug in the Rust code."""
    # The method `QuantumCircuit._append` is a semi-public method, so isn't really subject to
    # "protected access".

    qc = QuantumCircuit()
    qubits = []
    clbits = []
    gates = []
    has_u, has_cx = False, False
    for custom in custom_instructions:
        gates.append(custom.constructor)
        if custom.name == "U":
            has_u = True
        elif custom.name == "CX":
            has_cx = True
    if not has_u:
        gates.append(lib.UGate)
    if not has_cx:
        gates.append(lib.CXGate)
    # Pull this out as an explicit iterator so we can manually advance the loop in `DeclareGate`
    # contexts easily.
    bc = iter(bytecode)
    for op in bc:
        # We have to check `op.opcode` so many times, it's worth pulling out the extra attribute
        # access.  We should check the opcodes in order of their likelihood to be in the OQ2 program
        # for speed.  Gate applications are by far the most common for long programs.  This function
        # is deliberately long and does not use hashmaps or function lookups for speed in
        # Python-space.
        opcode = op.opcode
        # `OpCode` is an `enum` in Rust, but its instances don't have the same singleton property as
        # Python `enum.Enum` objects.
        if opcode == OpCode.Gate:
            gate_id, parameters, op_qubits = op.operands
            qc._append(
                CircuitInstruction(gates[gate_id](*parameters), [qubits[q] for q in op_qubits])
            )
        elif opcode == OpCode.ConditionedGate:
            gate_id, parameters, op_qubits, creg, value = op.operands
            with qc.if_test((qc.cregs[creg], value)):
                gate = gates[gate_id](*parameters)
                qc.append(gate, [qubits[q] for q in op_qubits])
        elif opcode == OpCode.Measure:
            qubit, clbit = op.operands
            qc._append(CircuitInstruction(Measure(), (qubits[qubit],), (clbits[clbit],)))
        elif opcode == OpCode.ConditionedMeasure:
            qubit, clbit, creg, value = op.operands
            with qc.if_test((qc.cregs[creg], value)):
                measure = Measure()
                qc.append(measure, (qubits[qubit],), (clbits[clbit],))
        elif opcode == OpCode.Reset:
            qc._append(CircuitInstruction(Reset(), (qubits[op.operands[0]],)))
        elif opcode == OpCode.ConditionedReset:
            qubit, creg, value = op.operands
            with qc.if_test((qc.cregs[creg], value)):
                reset = Reset()
                qc.append(reset, (qubits[qubit],))
        elif opcode == OpCode.Barrier:
            op_qubits = op.operands[0]
            qc._append(CircuitInstruction(Barrier(len(op_qubits)), [qubits[q] for q in op_qubits]))
        elif opcode == OpCode.DeclareQreg:
            name, size = op.operands
            register = QuantumRegister(size, name)
            qubits += register[:]
            qc.add_register(register)
        elif opcode == OpCode.DeclareCreg:
            name, size = op.operands
            register = ClassicalRegister(size, name)
            clbits += register[:]
            qc.add_register(register)
        elif opcode == OpCode.SpecialInclude:
            # Including `qelib1.inc` is pretty much universal, and we treat its gates as having
            # special relationships to the Qiskit ones, so we don't actually parse it; we just
            # short-circuit to add its pre-calculated content to our state.
            (indices,) = op.operands
            for index in indices:
                gates.append(QELIB1[index])
        elif opcode == OpCode.DeclareGate:
            name, num_qubits = op.operands
            # This inner loop advances the iterator of the outer loop as well, since `bc` is a
            # manually created iterator, rather than an implicit one from the first loop.
            inner_bc = []
            for inner_op in bc:
                if inner_op.opcode == OpCode.EndDeclareGate:
                    break
                inner_bc.append(inner_op)
            # Technically there's a quadratic dependency in the number of gates here, which could be
            # fixed by just sharing a reference to `gates` rather than recreating a new object.
            # Gates can't ever be removed from the list, so it wouldn't get out-of-date, though
            # there's a minor risk of somewhere accidentally mutating it instead, and in practice
            # the cost shouldn't really matter.
            gates.append(_gate_builder(name, num_qubits, tuple(gates), inner_bc))
        elif opcode == OpCode.DeclareOpaque:
            name, num_qubits = op.operands
            gates.append(_opaque_builder(name, num_qubits))
        else:
            raise ValueError(f"invalid operation: {op}")
    return qc


class _DefinedGate(Gate):
    """A gate object defined by a `gate` statement in an OpenQASM 2 program.  This object lazily
    binds its parameters to its definition, so it is only synthesized when required."""

    def __init__(self, name, num_qubits, params, gates, bytecode):
        self._gates = gates
        self._bytecode = bytecode
        super().__init__(name, num_qubits, list(params))

    def _define(self):
        # This is a stripped-down version of the bytecode interpreter; there's very few opcodes that
        # we actually need to handle within gate bodies.

        qubits = [Qubit() for _ in [None] * self.num_qubits]
        qc = QuantumCircuit(qubits)
        for op in self._bytecode:
            if op.opcode == OpCode.Gate:
                gate_id, args, op_qubits = op.operands
                qc._append(
                    CircuitInstruction(
                        self._gates[gate_id](*(_evaluate_argument(a, self.params) for a in args)),
                        [qubits[q] for q in op_qubits],
                    )
                )
            elif op.opcode == OpCode.Barrier:
                op_qubits = op.operands[0]
                qc._append(
                    CircuitInstruction(Barrier(len(op_qubits)), [qubits[q] for q in op_qubits])
                )
            else:
                raise ValueError(f"received invalid bytecode to build gate: {op}")
        self._definition = qc

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.asarray(Operator(self.definition), dtype=dtype)

    # It's fiddly to implement pickling for PyO3 types (the bytecode stream), so instead if we need
    # to pickle ourselves, we just eagerly create the definition and pickle that.

    def __getstate__(self):
        return (self.name, self.num_qubits, self.params, self.definition)

    def __setstate__(self, state):
        name, num_qubits, params, definition = state
        super().__init__(name, num_qubits, params)
        self._gates = ()
        self._bytecode = ()
        self._definition = definition


def _gate_builder(name, num_qubits, known_gates, bytecode):
    """Create a gate-builder function of the signature `*params -> Gate` for a gate with a given
    `name`.  This produces a `_DefinedGate` class, whose `_define` method runs through the given
    `bytecode` using the current list of `known_gates` to interpret the gate indices.

    The indirection here is mostly needed to correctly close over `known_gates` and `bytecode`."""

    def definer(*params):
        return _DefinedGate(name, num_qubits, params, known_gates, tuple(bytecode))

    return definer


def _opaque_builder(name, num_qubits):
    """Create a gate-builder function of the signature `*params -> Gate` for an opaque gate with a
    given `name`, which takes the given numbers of qubits."""

    def definer(*params):
        return Gate(name, num_qubits, params)

    return definer


# The natural way to reduce returns in this function would be to use a lookup table for the opcodes,
# but the PyO3 enum entities aren't (currently) hashable.
def _evaluate_argument(expr, parameters):
    """Inner recursive function to calculate the value of a mathematical expression given the
    concrete values in the `parameters` field."""
    if isinstance(expr, ExprConstant):
        return expr.value
    if isinstance(expr, ExprArgument):
        return parameters[expr.index]
    if isinstance(expr, ExprUnary):
        inner = _evaluate_argument(expr.argument, parameters)
        opcode = expr.opcode
        if opcode == UnaryOpCode.Negate:
            return -inner
        if opcode == UnaryOpCode.Cos:
            return math.cos(inner)
        if opcode == UnaryOpCode.Exp:
            return math.exp(inner)
        if opcode == UnaryOpCode.Ln:
            return math.log(inner)
        if opcode == UnaryOpCode.Sin:
            return math.sin(inner)
        if opcode == UnaryOpCode.Sqrt:
            return math.sqrt(inner)
        if opcode == UnaryOpCode.Tan:
            return math.tan(inner)
        raise ValueError(f"unhandled unary opcode: {opcode}")
    if isinstance(expr, ExprBinary):
        left = _evaluate_argument(expr.left, parameters)
        right = _evaluate_argument(expr.right, parameters)
        opcode = expr.opcode
        if opcode == BinaryOpCode.Add:
            return left + right
        if opcode == BinaryOpCode.Subtract:
            return left - right
        if opcode == BinaryOpCode.Multiply:
            return left * right
        if opcode == BinaryOpCode.Divide:
            return left / right
        if opcode == BinaryOpCode.Power:
            return left**right
        raise ValueError(f"unhandled binary opcode: {opcode}")
    if isinstance(expr, ExprCustom):
        return expr.call(*(_evaluate_argument(x, parameters) for x in expr.arguments))
    raise ValueError(f"unhandled expression type: {expr}")
