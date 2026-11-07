# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
QPY Type keys for several namespace.
"""

import uuid
from abc import abstractmethod
from enum import Enum, IntEnum, IntFlag

import numpy as np

from qiskit.circuit import (
    Gate,
    Instruction,
    QuantumCircuit,
    ControlledGate,
    CASE_DEFAULT,
    Clbit,
    ClassicalRegister,
    Duration,
)
from qiskit.circuit.annotated_operation import AnnotatedOperation, Modifier
from qiskit.circuit.classical import expr, types
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.qpy import exceptions


class TypeKeyBase(bytes, Enum):
    """Abstract baseclass for type key Enums."""

    @classmethod
    @abstractmethod
    def assign(cls, obj):
        """Assign type key to given object.

        Args:
            obj (any): Arbitrary object to evaluate.

        Returns:
            TypeKey: Corresponding key object.
        """

    @classmethod
    @abstractmethod
    def retrieve(cls, type_key):
        """Get a class from given type key.

        Args:
            type_key (bytes): Object type key.

        Returns:
            any: Corresponding class.
        """


class Value(TypeKeyBase):
    """Type key enum for value object."""

    INTEGER = b"i"
    FLOAT = b"f"
    COMPLEX = b"c"
    CASE_DEFAULT = b"d"
    REGISTER = b"R"
    NUMPY_OBJ = b"n"
    PARAMETER = b"p"
    PARAMETER_VECTOR = b"v"
    PARAMETER_EXPRESSION = b"e"
    STRING = b"s"
    NULL = b"z"
    EXPRESSION = b"x"
    MODIFIER = b"m"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, int):
            return cls.INTEGER
        if isinstance(obj, float):
            return cls.FLOAT
        if isinstance(obj, complex):
            return cls.COMPLEX
        if isinstance(obj, (np.integer, np.floating, np.complexfloating, np.ndarray)):
            return cls.NUMPY_OBJ
        if isinstance(obj, ParameterVectorElement):
            return cls.PARAMETER_VECTOR
        if isinstance(obj, Parameter):
            return cls.PARAMETER
        if isinstance(obj, ParameterExpression):
            return cls.PARAMETER_EXPRESSION
        if isinstance(obj, str):
            return cls.STRING
        if isinstance(obj, (Clbit, ClassicalRegister)):
            return cls.REGISTER
        if obj is None:
            return cls.NULL
        if obj is CASE_DEFAULT:
            return cls.CASE_DEFAULT
        if isinstance(obj, expr.Expr):
            return cls.EXPRESSION
        if isinstance(obj, Modifier):
            return cls.MODIFIER

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class Condition(IntEnum):
    """Type keys for the ``conditional_key`` field of the INSTRUCTION struct."""

    # This class is deliberately raw integers and not in terms of ASCII characters for backwards
    # compatibility in the form as an old Boolean value was expanded; `NONE` and `TWO_TUPLE` must
    # have the enumeration values 0 and 1.

    NONE = 0
    TWO_TUPLE = 1
    EXPRESSION = 2


class InstructionExtraFlags(IntFlag):
    """If an instruction has extra payloads associated with it."""

    HAS_ANNOTATIONS = 0b1000_0000


class Container(TypeKeyBase):
    """Type key enum for container-like object."""

    RANGE = b"r"
    TUPLE = b"t"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, range):
            return cls.RANGE
        if isinstance(obj, tuple):
            return cls.TUPLE

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class CircuitInstruction(TypeKeyBase):
    """Type key enum for circuit instruction object."""

    INSTRUCTION = b"i"
    GATE = b"g"
    PAULI_EVOL_GATE = b"p"
    CONTROLLED_GATE = b"c"
    ANNOTATED_OPERATION = b"a"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, PauliEvolutionGate):
            return cls.PAULI_EVOL_GATE
        if isinstance(obj, ControlledGate):
            return cls.CONTROLLED_GATE
        if isinstance(obj, AnnotatedOperation):
            return cls.ANNOTATED_OPERATION
        if isinstance(obj, Gate):
            return cls.GATE
        if isinstance(obj, Instruction):
            return cls.INSTRUCTION

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ScheduleOperand(TypeKeyBase):
    """Type key enum for schedule instruction operand object.

    Note: This class is kept post pulse-removal to allow reading of
    legacy payloads containing pulse gates without breaking the entire
    load flow.
    """

    WAVEFORM = b"w"
    SYMBOLIC_PULSE = b"s"
    CHANNEL = b"c"
    KERNEL = b"k"
    DISCRIMINATOR = b"d"

    # We need to have own string type definition for operands of schedule instruction.
    # Note that string type is already defined in the Value namespace,
    # but its key "s" conflicts with the SYMBOLIC_PULSE in the ScheduleOperand namespace.
    # New in QPY version 7.
    OPERAND_STR = b"o"

    @classmethod
    def assign(cls, _):
        raise NotImplementedError

    @classmethod
    def retrieve(cls, _):
        raise NotImplementedError


class Program(TypeKeyBase):
    """Type key enum for program that QPY supports."""

    CIRCUIT = b"q"
    # This is left for backward compatibility, for identifying payloads of type `ScheduleBlock`
    # and raising accordingly. `ScheduleBlock` support has been removed in Qiskit 2.0 as part
    # of the pulse package removal in that version.
    SCHEDULE_BLOCK = b"s"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, QuantumCircuit):
            return cls.CIRCUIT

        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class Expression(TypeKeyBase):
    """Type keys for the ``EXPRESSION`` QPY item."""

    VAR = b"x"
    STRETCH = b"s"
    VALUE = b"v"
    CAST = b"c"
    UNARY = b"u"
    BINARY = b"b"
    INDEX = b"i"
    RANGE = b"r"

    @classmethod
    def assign(cls, obj):
        if (
            isinstance(obj, expr.Expr)
            and (key := getattr(cls, obj.__class__.__name__.upper(), None)) is not None
        ):
            return key
        raise exceptions.QpyError(f"Object '{obj}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprVarDeclaration(TypeKeyBase):
    """Type keys for the ``EXPR_VAR_DECLARATION`` QPY item."""

    INPUT = b"I"
    CAPTURE = b"C"
    LOCAL = b"L"
    STRETCH_CAPTURE = b"A"
    STRETCH_LOCAL = b"O"

    @classmethod
    def assign(cls, obj):
        raise NotImplementedError

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprType(TypeKeyBase):
    """Type keys for the ``EXPR_TYPE`` QPY item."""

    BOOL = b"b"
    UINT = b"u"
    FLOAT = b"f"
    DURATION = b"d"

    @classmethod
    def assign(cls, obj):
        if (
            isinstance(obj, types.Type)
            and (key := getattr(cls, obj.__class__.__name__.upper(), None)) is not None
        ):
            return key
        raise exceptions.QpyError(f"Object '{obj}' is not supported in {cls.__name__} namespace.")

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprVar(TypeKeyBase):
    """Type keys for the ``EXPR_VAR`` QPY item."""

    CLBIT = b"C"
    REGISTER = b"R"
    UUID = b"U"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, uuid.UUID):
            return cls.UUID
        if isinstance(obj, Clbit):
            return cls.CLBIT
        if isinstance(obj, ClassicalRegister):
            return cls.REGISTER
        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class ExprValue(TypeKeyBase):
    """Type keys for the ``EXPR_VALUE`` QPY item."""

    BOOL = b"b"
    INT = b"i"
    FLOAT = b"f"
    DURATION = b"t"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, bool):
            return cls.BOOL
        if isinstance(obj, int):
            return cls.INT
        if isinstance(obj, float):
            return cls.FLOAT
        if isinstance(obj, Duration):
            return cls.DURATION
        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class CircuitDuration(TypeKeyBase):
    """Type keys for the ``DURATION`` QPY item."""

    DT = b"t"
    PS = b"p"
    NS = b"n"
    US = b"u"
    MS = b"m"
    S = b"s"

    @classmethod
    def assign(cls, obj):
        if isinstance(obj, Duration):
            unit = obj.unit()
            if unit == "dt":
                return cls.DT
            if unit == "ps":
                return cls.PS
            if unit == "ns":
                return cls.NS
            if unit == "us":
                return cls.US
            if unit == "ms":
                return cls.MS
            if unit == "s":
                return cls.S
        raise exceptions.QpyError(
            f"Object type '{type(obj)}' is not supported in {cls.__name__} namespace."
        )

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError


class SymExprEncoding(TypeKeyBase):
    """Type keys for the symbolic encoding field in the file header."""

    SYMPY = b"p"
    SYMENGINE = b"e"

    @classmethod
    def assign(cls, obj):
        if obj:
            return cls.SYMENGINE
        else:
            return cls.SYMPY

    @classmethod
    def retrieve(cls, type_key):
        raise NotImplementedError
