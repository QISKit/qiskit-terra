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

"""Export tools for OpenQASM 2."""

from __future__ import annotations

__all__ = ["dump", "dumps"]

import collections.abc
import io
import itertools
import os
import re
import string

from qiskit.circuit import (
    QuantumCircuit,
    Instruction,
    QuantumRegister,
    ClassicalRegister,
    Qubit,
    Clbit,
    Parameter,
    library as lib,
)
from qiskit.circuit.classical import expr
from qiskit.circuit.tools import pi_check
from .exceptions import QASM2ExportError

_EXISTING_GATE_NAMES = frozenset(
    [
        "barrier",
        "measure",
        "reset",
        "u3",
        "u2",
        "u1",
        "cx",
        "id",
        "u0",
        "u",
        "p",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "sx",
        "sxdg",
        "cz",
        "cy",
        "swap",
        "ch",
        "ccx",
        "cswap",
        "crx",
        "cry",
        "crz",
        "cu1",
        "cp",
        "cu3",
        "csx",
        "cu",
        "rxx",
        "rzz",
        "rccx",
        "rc3x",
        "c3x",
        "c3sx",  # This is the Qiskit gate name, but the qelib1.inc name is 'c3sqrtx'.
        "c4x",
    ]
)

_RESERVED = frozenset(
    [
        "OPENQASM",
        "qreg",
        "creg",
        "include",
        "gate",
        "opaque",
        "U",
        "CX",
        "measure",
        "reset",
        "if",
        "barrier",
    ]
)


def dump(circuit: QuantumCircuit, filename_or_stream: os.PathLike | io.TextIOBase, /):
    """Dump a circuit as an OpenQASM 2 program to a file or stream.

    Args:
        circuit: the :class:`.QuantumCircuit` to be exported.
        filename_or_stream: either a path-like object (likely a :class:`str` or
            :class:`pathlib.Path`), or an already opened text-mode stream.

    Raises:
        QASM2ExportError: if the circuit cannot be represented by OpenQASM 2.
    """
    if isinstance(filename_or_stream, io.TextIOBase):
        print(dumps(circuit), file=filename_or_stream)
        return
    with open(filename_or_stream, "w") as stream:
        print(dumps(circuit), file=stream)


def dumps(circuit: QuantumCircuit, /) -> str:
    """Export a circuit to an OpenQASM 2 program in a string.

    Args:
        circuit: the :class:`.QuantumCircuit` to be exported.

    Returns:
        An OpenQASM 2 string representing the circuit.

    Raises:
        QASM2ExportError: if the circuit cannot be represented by OpenQASM 2.
    """
    if circuit.num_parameters > 0:
        raise QASM2ExportError("Cannot represent circuits with unbound parameters in OpenQASM 2.")

    # Mapping of instruction name to a pair of the source for a definition, and an OQ2 string
    # that includes the `gate` or `opaque` statement that defines the gate.
    gates_to_define: collections.OrderedDict[str, tuple[Instruction, str]] = (
        collections.OrderedDict()
    )

    regless_qubits = [bit for bit in circuit.qubits if not circuit.find_bit(bit).registers]
    regless_clbits = [bit for bit in circuit.clbits if not circuit.find_bit(bit).registers]
    dummy_registers: list[QuantumRegister | ClassicalRegister] = []
    if regless_qubits:
        dummy_registers.append(QuantumRegister(name="qregless", bits=regless_qubits))
    if regless_clbits:
        dummy_registers.append(ClassicalRegister(name="cregless", bits=regless_clbits))
    register_escaped_names: dict[str, QuantumRegister | ClassicalRegister] = {}
    name_from_register: dict[QuantumRegister | ClassicalRegister, str] = {}
    for regs in (circuit.qregs, circuit.cregs, dummy_registers):
        for reg in regs:
            name = _make_unique(_escape_name(reg.name, "reg_"), register_escaped_names)
            register_escaped_names[name] = reg
            name_from_register[reg] = name
    bit_labels: dict[Qubit | Clbit, str] = {
        bit: f"{name}[{idx}]"
        for name, register in register_escaped_names.items()
        for (idx, bit) in enumerate(register)
    }
    register_definitions_qasm = "\n".join(
        f"{'qreg' if isinstance(reg, QuantumRegister) else 'creg'} {name}[{reg.size}];"
        for name, reg in register_escaped_names.items()
    )

    def qasm_line(instruction) -> str | None:
        if instruction.is_control_flow():
            operation = instruction.operation
            if operation.name != "if_else":
                raise QASM2ExportError(
                    "OpenQASM 2 does not support control-flow constructs outside if_else"
                )
            if isinstance(operation.condition, expr.Expr) or isinstance(
                operation.condition[0], Clbit
            ):
                # This could be relaxed (somewhat) in the future by attempting to resolve the `Expr`
                # into a simple "register-equality" form; `Expr` is a superset of allowed conditions
                raise QASM2ExportError("OpenQASM 2 only supports register-equality conditions")
            if len(operation.blocks) != 1:
                raise QASM2ExportError("OpenQASM 2 does not support 'else' statements")
            body = operation.blocks[0]
            if len(body.data) != 1:
                # This could be relaxed in the future by "unrolling" the block and putting
                # individual `if` conditions on each instruction.
                raise QASM2ExportError(
                    "OpenQASM 2 only supports conditionals on single instructions"
                )
            inner_instruction = body.data[0]
            if inner_instruction.name == "if_else":
                raise QASM2ExportError("OpenQASM 2 does not support nested conditionals")
            if inner_instruction.name == "barrier":
                raise QASM2ExportError("barriers cannot be conditional in OpenQASM 2")
            reg, value = operation.condition
            return f"if ({name_from_register[reg]} == {int(value)}) {qasm_line(inner_instruction)}"
        if instruction.name == "measure":
            qubit = instruction.qubits[0]
            clbit = instruction.clbits[0]
            return f"measure {bit_labels[qubit]} -> {bit_labels[clbit]};"
        if instruction.name == "reset":
            return f"reset {bit_labels[instruction.qubits[0]]};"
        if instruction.name == "barrier":
            if not instruction.qubits:
                # Barriers with no operands are invalid in (strict) OQ2.  Qiskit doesn't track
                # "global" barriers specially, so that case is handled by explicitly listing all the
                # qubits already.
                return None
            qargs = ",".join(bit_labels[q] for q in instruction.qubits)
            return f"barrier {qargs};"
        return _custom_operation_statement(instruction, gates_to_define, bit_labels)

    instruction_calls = [
        line for instruction in circuit._data if (line := qasm_line(instruction)) is not None
    ]
    instructions_qasm = "\n".join(f"{call}" for call in instruction_calls)
    gate_definitions_qasm = "\n".join(f"{qasm}" for _, qasm in gates_to_define.values())

    return "\n".join(
        part
        for part in (
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            gate_definitions_qasm,
            register_definitions_qasm,
            instructions_qasm,
        )
        if part
    )


def _escape_name(name: str, prefix: str) -> str:
    """Returns a valid OpenQASM 2.0 identifier, using `prefix` as a prefix if necessary.  `prefix`
    must itself be a valid identifier."""
    # Replace all non-ASCII-word characters (letters, digits, underscore) with the underscore.
    escaped_name = re.sub(r"\W", "_", name, flags=re.ASCII)
    if (
        not escaped_name
        or escaped_name[0] not in string.ascii_lowercase
        or escaped_name in _RESERVED
    ):
        escaped_name = prefix + escaped_name
    return escaped_name


def _make_unique(name: str, already_defined: collections.abc.Set[str]) -> str:
    """Generate a name by suffixing the given stem that is unique within the defined set."""
    if name not in already_defined:
        return name
    used = {in_use[len(name) :] for in_use in already_defined if in_use.startswith(name)}
    characters = (string.digits + string.ascii_letters) if name else string.ascii_letters
    for parts in itertools.chain.from_iterable(
        itertools.product(characters, repeat=n) for n in itertools.count(1)
    ):
        suffix = "".join(parts)
        if suffix not in used:
            return name + suffix
    # This isn't actually reachable because the above loop is infinite.
    return name


def _rename_operation(operation):
    """Returns the operation with a new name following this pattern: {operation name}_{operation id}"""
    new_name = f"{operation.name}_{id(operation)}"
    updated_operation = operation.copy(name=new_name)
    return updated_operation


def _instruction_call_site(operation):
    """Return an OpenQASM 2 string for the instruction."""
    if operation.name == "c3sx":
        qasm2_call = "c3sqrtx"
    else:
        qasm2_call = operation.name
    if operation.params:
        params = ",".join([pi_check(i, output="qasm", eps=1e-12) for i in operation.params])
        qasm2_call = f"{qasm2_call}({params})"
    return qasm2_call


# Just needs to have enough parameters to support the largest standard (non-controlled) gate in our
# standard library.  We have to use the same `Parameter` instances each time so the equality
# comparisons will work.
_FIXED_PARAMETERS = [Parameter("param0"), Parameter("param1"), Parameter("param2")]


def _custom_operation_statement(instruction, gates_to_define, bit_labels):
    operation = _define_custom_operation(instruction.operation, gates_to_define)
    # Insert qasm representation of the original instruction
    if instruction.clbits:
        bits = itertools.chain(instruction.qubits, instruction.clbits)
    else:
        bits = instruction.qubits
    bits_qasm = ",".join(bit_labels[j] for j in bits)
    return f"{_instruction_call_site(operation)} {bits_qasm};"


def _define_custom_operation(operation, gates_to_define):
    """Extract a custom definition from the given operation, and append any necessary additional
    subcomponents' definitions to the ``gates_to_define`` ordered dictionary.

    Returns a potentially new :class:`.Instruction`, which should be used for the
    :meth:`~.Instruction.qasm` call (it may have been renamed)."""
    if operation.name in _EXISTING_GATE_NAMES:
        return operation

    # Check instructions names or label are valid
    escaped = _escape_name(operation.name, "gate_")
    if escaped != operation.name:
        operation = operation.copy(name=escaped)

    # These are built-in gates that are known to be safe to construct by passing the correct number
    # of `Parameter` instances positionally, and have no other information.  We can't guarantee that
    # if they've been subclassed, though.  This is a total hack; ideally we'd be able to inspect the
    # "calling" signatures of Qiskit `Gate` objects to know whether they're safe to re-parameterise.
    known_good_parameterized = {
        lib.PhaseGate,
        lib.RGate,
        lib.RXGate,
        lib.RXXGate,
        lib.RYGate,
        lib.RYYGate,
        lib.RZGate,
        lib.RZXGate,
        lib.RZZGate,
        lib.XXMinusYYGate,
        lib.XXPlusYYGate,
        lib.UGate,
        lib.U1Gate,
        lib.U2Gate,
        lib.U3Gate,
    }

    # In known-good situations we want to use a manually parametrized object as the source of the
    # definition, but still continue to return the given object as the call-site object.
    if operation.base_class in known_good_parameterized:
        parameterized_operation = type(operation)(*_FIXED_PARAMETERS[: len(operation.params)])
    elif hasattr(operation, "_qasm_decomposition"):
        new_op = operation._qasm_decomposition()
        parameterized_operation = operation = new_op.copy(name=_escape_name(new_op.name, "gate_"))
    else:
        parameterized_operation = operation

    # If there's an _equal_ operation in the existing circuits to be defined, then our job is done.
    previous_definition_source, _ = gates_to_define.get(operation.name, (None, None))
    if parameterized_operation == previous_definition_source:
        return operation

    # Otherwise, if there's a naming clash, we need a unique name.
    if operation.name in gates_to_define:
        operation = _rename_operation(operation)

    new_name = operation.name

    if parameterized_operation.params:
        parameters_qasm = (
            "(" + ",".join(f"param{i}" for i in range(len(parameterized_operation.params))) + ")"
        )
    else:
        parameters_qasm = ""

    if operation.num_qubits == 0:
        raise QASM2ExportError(
            f"OpenQASM 2 cannot represent '{operation.name}', which acts on zero qubits."
        )
    if operation.num_clbits != 0:
        raise QASM2ExportError(
            f"OpenQASM 2 cannot represent '{operation.name}', which acts on {operation.num_clbits}"
            " classical bits."
        )

    qubits_qasm = ",".join(f"q{i}" for i in range(parameterized_operation.num_qubits))
    parameterized_definition = getattr(parameterized_operation, "definition", None)
    if parameterized_definition is None:
        gates_to_define[new_name] = (
            parameterized_operation,
            f"opaque {new_name}{parameters_qasm} {qubits_qasm};",
        )
    else:
        qubit_labels = {bit: f"q{i}" for i, bit in enumerate(parameterized_definition.qubits)}
        body_qasm = " ".join(
            _custom_operation_statement(instruction, gates_to_define, qubit_labels)
            for instruction in parameterized_definition.data
        )

        # if an inner operation has the same name as the actual operation, it needs to be renamed
        if operation.name in gates_to_define:
            operation = _rename_operation(operation)
            new_name = operation.name

        definition_qasm = f"gate {new_name}{parameters_qasm} {qubits_qasm} {{ {body_qasm} }}"
        gates_to_define[new_name] = (parameterized_operation, definition_qasm)
    return operation
