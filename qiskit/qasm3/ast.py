# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""QASM3 AST Nodes"""

from __future__ import annotations

import enum

from collections.abc import Iterable, Sequence


class ASTNode:
    """Base abstract class for AST nodes"""

    __slots__ = ()


class Statement(ASTNode):
    """
    statement
        : expressionStatement
        | assignmentStatement
        | classicalDeclarationStatement
        | branchingStatement
        | loopStatement
        | endStatement
        | aliasStatement
        | quantumStatement
    """

    __slots__ = ()


class Pragma(ASTNode):
    """
    pragma
        : '#pragma' LBRACE statement* RBRACE  // match any valid openqasm statement
    """

    __slots__ = ("content",)

    def __init__(self, content):
        self.content = content


class Annotation(ASTNode):
    """An annotation."""

    __slots__ = ("namespace", "payload")

    def __init__(self, namespace: str, payload: str):
        self.namespace = namespace
        self.payload = payload


class CalibrationGrammarDeclaration(Statement):
    """
    calibrationGrammarDeclaration
        : 'defcalgrammar' calibrationGrammar SEMICOLON
    """

    __slots__ = ("name",)

    def __init__(self, name):
        self.name = name


class Program(ASTNode):
    """
    program
        : header (globalStatement | statement)*
    """

    __slots__ = ("header", "statements")

    def __init__(self, header, statements=()):
        self.header = header
        self.statements = statements


class Header(ASTNode):
    """
    header
        : version? include*
    """

    __slots__ = ("includes", "version")

    def __init__(self, version, includes):
        self.version = version
        self.includes = includes


class Include(ASTNode):
    """
    include
        : 'include' StringLiteral SEMICOLON
    """

    __slots__ = ("filename",)

    def __init__(self, filename):
        self.filename = filename


class Version(ASTNode):
    """
    version
        : 'OPENQASM'(Integer | RealNumber) SEMICOLON
    """

    __slots__ = ("version_number",)

    def __init__(self, version_number):
        self.version_number = version_number


class QuantumInstruction(ASTNode):
    """
    quantumInstruction
        : quantumGateCall
        | quantumPhase
        | quantumMeasurement
        | quantumReset
        | quantumBarrier
    """

    __slots__ = ()


class ClassicalType(ASTNode):
    """Information about a classical type.  This is just an abstract base for inheritance tests."""

    __slots__ = ()


class FloatType(ClassicalType, enum.Enum):
    """Allowed values for the width of floating-point types."""

    HALF = 16
    SINGLE = 32
    DOUBLE = 64
    QUAD = 128
    OCT = 256


class BoolType(ClassicalType):
    """Type information for a Boolean."""

    __slots__ = ()


class IntType(ClassicalType):
    """Type information for a signed integer."""

    __slots__ = ("size",)

    def __init__(self, size: int | None = None):
        self.size = size


class UintType(ClassicalType):
    """Type information for an unsigned integer."""

    __slots__ = ("size",)

    def __init__(self, size: int | None = None):
        self.size = size


class BitType(ClassicalType):
    """Type information for a single bit."""

    __slots__ = ()


class DurationType(ClassicalType):
    """Type information for a duration."""

    __slots__ = ()


class BitArrayType(ClassicalType):
    """Type information for a sized number of classical bits."""

    __slots__ = ("size",)

    def __init__(self, size: int):
        self.size = size


class Expression(ASTNode):
    __slots__ = ()


class StringifyAndPray(Expression):
    __slots__ = ("obj",)

    # This is not a real AST node, yet is somehow very common. It's used when there are
    # `ParameterExpression` instances; instead of actually visiting the Sympy expression tree into
    # an OQ3 AST, we just convert it to a string, cross our fingers, and hope.
    def __init__(self, obj):
        self.obj = obj


class Range(Expression):
    __slots__ = ("end", "start", "step")

    def __init__(
        self,
        start: Expression | None = None,
        end: Expression | None = None,
        step: Expression | None = None,
    ):
        self.start = start
        self.step = step
        self.end = end


class Identifier(Expression):
    __slots__ = ("string",)

    def __init__(self, string):
        self.string = string


class SubscriptedIdentifier(Identifier):
    """An identifier with subscripted access."""

    __slots__ = ("subscript",)

    def __init__(self, string: str, subscript: Range | Expression):
        super().__init__(string)
        self.subscript = subscript


class Constant(Expression, enum.Enum):
    """A constant value defined by the QASM 3 spec."""

    PI = enum.auto()
    EULER = enum.auto()
    TAU = enum.auto()


class IntegerLiteral(Expression):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class FloatLiteral(Expression):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class BooleanLiteral(Expression):
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


class BitstringLiteral(Expression):
    __slots__ = ("value", "width")

    def __init__(self, value, width):
        self.value = value
        self.width = width


class DurationUnit(enum.Enum):
    """Valid values for the unit of durations."""

    NANOSECOND = "ns"
    MICROSECOND = "us"
    MILLISECOND = "ms"
    SECOND = "s"
    SAMPLE = "dt"


class DurationLiteral(Expression):
    __slots__ = ("unit", "value")

    def __init__(self, value: float, unit: DurationUnit):
        self.value = value
        self.unit = unit


class Unary(Expression):
    __slots__ = ("op", "operand")

    class Op(enum.Enum):
        LOGIC_NOT = "!"
        BIT_NOT = "~"
        NEGATE = "-"

    def __init__(self, op: Op, operand: Expression):
        self.op = op
        self.operand = operand


class Binary(Expression):
    __slots__ = ("left", "op", "right")

    class Op(enum.Enum):
        BIT_AND = "&"
        BIT_OR = "|"
        BIT_XOR = "^"
        LOGIC_AND = "&&"
        LOGIC_OR = "||"
        LESS = "<"
        LESS_EQUAL = "<="
        GREATER = ">"
        GREATER_EQUAL = ">="
        EQUAL = "=="
        NOT_EQUAL = "!="
        SHIFT_LEFT = "<<"
        SHIFT_RIGHT = ">>"
        ADD = "+"
        SUB = "-"
        MUL = "*"
        DIV = "/"

    def __init__(self, op: Op, left: Expression, right: Expression):
        self.op = op
        self.left = left
        self.right = right


class Cast(Expression):
    __slots__ = ("operand", "type")

    def __init__(self, type: ClassicalType, operand: Expression):
        self.type = type
        self.operand = operand


class Index(Expression):
    __slots__ = ("index", "target")

    def __init__(self, target: Expression, index: Expression):
        self.target = target
        self.index = index


class IndexSet(ASTNode):
    """
    A literal index set of values::

        { Expression (, Expression)* }
    """

    __slots__ = ("values",)

    def __init__(self, values: list[Expression]):
        self.values = values


class QuantumMeasurement(ASTNode):
    """
    quantumMeasurement
        : 'measure' indexIdentifierList
    """

    __slots__ = ("identifierList",)

    def __init__(self, identifierList: list[Identifier]):
        self.identifierList = identifierList


class QuantumMeasurementAssignment(Statement):
    """
    quantumMeasurementAssignment
        : quantumMeasurement ARROW indexIdentifierList
        | indexIdentifier EQUALS quantumMeasurement  # eg: bits = measure qubits;
    """

    __slots__ = ("identifier", "quantumMeasurement")

    def __init__(self, identifier: Identifier, quantumMeasurement: QuantumMeasurement):
        self.identifier = identifier
        self.quantumMeasurement = quantumMeasurement


class Designator(ASTNode):
    """
    designator
        : LBRACKET expression RBRACKET
    """

    __slots__ = ("expression",)

    def __init__(self, expression: Expression):
        self.expression = expression


class ClassicalDeclaration(Statement):
    """Declaration of a classical type, optionally initializing it to a value."""

    __slots__ = ("identifier", "initializer", "type")

    def __init__(self, type_: ClassicalType, identifier: Identifier, initializer=None):
        self.type = type_
        self.identifier = identifier
        self.initializer = initializer


class StretchDeclaration(Statement):
    """Declaration of a stretch variable, optionally with a lower bound
    expression."""

    __slots__ = ("bound", "identifier")

    def __init__(self, identifier: Identifier, bound=None):
        self.identifier = identifier
        self.bound = bound


class AssignmentStatement(Statement):
    """Assignment of an expression to an l-value."""

    __slots__ = ("lvalue", "rvalue")

    def __init__(self, lvalue: SubscriptedIdentifier, rvalue: Expression):
        self.lvalue = lvalue
        self.rvalue = rvalue


class QuantumDeclaration(ASTNode):
    """
    quantumDeclaration
        : 'qreg' Identifier designator? |   # NOT SUPPORTED
         'qubit' designator? Identifier
    """

    __slots__ = ("designator", "identifier")

    def __init__(self, identifier: Identifier, designator=None):
        self.identifier = identifier
        self.designator = designator


class AliasStatement(ASTNode):
    """
    aliasStatement
        : 'let' Identifier EQUALS indexIdentifier SEMICOLON
    """

    __slots__ = ("identifier", "value")

    def __init__(self, identifier: Identifier, value: Expression):
        self.identifier = identifier
        self.value = value


class QuantumGateModifierName(enum.Enum):
    """The names of the allowed modifiers of quantum gates."""

    CTRL = enum.auto()
    NEGCTRL = enum.auto()
    INV = enum.auto()
    POW = enum.auto()


class QuantumGateModifier(ASTNode):
    """A modifier of a gate. For example, in ``ctrl @ x $0``, the ``ctrl @`` is the modifier."""

    __slots__ = ("argument", "modifier")

    def __init__(self, modifier: QuantumGateModifierName, argument: Expression | None = None):
        self.modifier = modifier
        self.argument = argument


class QuantumGateCall(QuantumInstruction):
    """
    quantumGateCall
        : quantumGateModifier* quantumGateName ( LPAREN expressionList? RPAREN )? indexIdentifierList
    """

    __slots__ = ("indexIdentifierList", "modifiers", "parameters", "quantumGateName")

    def __init__(
        self,
        quantumGateName: Identifier,
        indexIdentifierList: list[Identifier],
        parameters: Sequence[Expression] = (),
        modifiers: Sequence[QuantumGateModifier] = (),
    ):
        self.quantumGateName = quantumGateName
        self.indexIdentifierList = indexIdentifierList
        self.parameters = parameters
        self.modifiers = modifiers


class DefcalCallStatement(Statement):
    """A quantum-like call that may have an assignment location."""

    __slots__ = ("ident", "lvalue", "parameters", "qubits")

    def __init__(
        self,
        ident: Identifier,
        parameters: Sequence[Expression] = (),
        qubits: Sequence[Expression] = (),
        lvalue: Expression | None = None,
    ):
        self.ident = ident
        self.parameters = tuple(parameters)
        self.qubits = tuple(qubits)
        self.lvalue = lvalue


class QuantumBarrier(QuantumInstruction):
    """
    quantumBarrier
        : 'barrier' indexIdentifierList
    """

    __slots__ = ("indexIdentifierList",)

    def __init__(self, indexIdentifierList: list[Identifier]):
        self.indexIdentifierList = indexIdentifierList


class QuantumReset(QuantumInstruction):
    """A built-in ``reset q0;`` statement."""

    __slots__ = ("identifier",)

    def __init__(self, identifier: Identifier):
        self.identifier = identifier


class QuantumDelay(QuantumInstruction):
    """A built-in ``delay[duration] q0;`` statement."""

    __slots__ = ("duration", "qubits")

    def __init__(self, duration: Expression, qubits: list[Identifier]):
        self.duration = duration
        self.qubits = qubits


class ProgramBlock(ASTNode):
    """
    programBlock
        : statement | controlDirective
        | LBRACE(statement | controlDirective) * RBRACE
    """

    __slots__ = ("statements",)

    def __init__(self, statements: list[Statement]):
        self.statements = statements


class ReturnStatement(ASTNode):  # TODO probably should be a subclass of ControlDirective
    """
    returnStatement
        : 'return' ( expression | quantumMeasurement )? SEMICOLON;
    """

    __slots__ = ("expression",)

    def __init__(self, expression=None):
        self.expression = expression


class QuantumBlock(ProgramBlock):
    """
    quantumBlock
        : LBRACE ( quantumStatement | quantumLoop )* RBRACE
    """

    __slots__ = ()


class SubroutineBlock(ProgramBlock):
    """
    subroutineBlock
        : LBRACE statement* returnStatement? RBRACE
    """

    __slots__ = ()


class QuantumGateDefinition(Statement):
    """
    quantumGateDefinition
        : 'gate' quantumGateSignature quantumBlock
    """

    __slots__ = ("body", "name", "params", "qubits")

    def __init__(
        self,
        name: Identifier,
        params: tuple[Identifier, ...],
        qubits: tuple[Identifier, ...],
        body: QuantumBlock,
    ):
        self.name = name
        self.params = params
        self.qubits = qubits
        self.body = body


class SubroutineDefinition(Statement):
    """
    subroutineDefinition
        : 'def' Identifier LPAREN anyTypeArgumentList? RPAREN
        returnSignature? subroutineBlock
    """

    __slots__ = ("arguments", "identifier", "subroutineBlock")

    def __init__(
        self,
        identifier: Identifier,
        subroutineBlock: SubroutineBlock,
        arguments=(),  # [ClassicalArgument]
    ):
        self.identifier = identifier
        self.arguments = arguments
        self.subroutineBlock = subroutineBlock


class CalibrationArgument(ASTNode):
    """
    calibrationArgumentList
        : classicalArgumentList | expressionList
    """

    __slots__ = ()


class CalibrationDefinition(Statement):
    """
    calibrationDefinition
        : 'defcal' Identifier
        ( LPAREN calibrationArgumentList? RPAREN )? identifierList
        returnSignature? LBRACE .*? RBRACE  // for now, match anything inside body
        ;
    """

    __slots__ = ("calibrationArgumentList", "identifierList", "name")

    def __init__(
        self,
        name: Identifier,
        identifierList: list[Identifier],
        calibrationArgumentList: Sequence[CalibrationArgument] = (),
    ):
        self.name = name
        self.identifierList = identifierList
        self.calibrationArgumentList = calibrationArgumentList


class BranchingStatement(Statement):
    """
    branchingStatement
        : 'if' LPAREN booleanExpression RPAREN programBlock ( 'else' programBlock )?
    """

    __slots__ = ("condition", "false_body", "true_body")

    def __init__(self, condition: Expression, true_body: ProgramBlock, false_body=None):
        self.condition = condition
        self.true_body = true_body
        self.false_body = false_body


class ForLoopStatement(Statement):
    """
    AST node for ``for`` loops.

    ::

        ForLoop: "for" Identifier "in" SetDeclaration ProgramBlock
        SetDeclaration:
            | Identifier
            | "{" Expression ("," Expression)* "}"
            | "[" Range "]"
    """

    __slots__ = ("body", "indexset", "parameter", "type")

    def __init__(
        self,
        indexset: Identifier | IndexSet | Range,
        parameter: Identifier,
        body: ProgramBlock,
        type_: ClassicalType,
    ):
        self.indexset = indexset
        self.parameter = parameter
        self.body = body
        self.type = type_


class WhileLoopStatement(Statement):
    """
    AST node for ``while`` loops.

    ::

        WhileLoop: "while" "(" Expression ")" ProgramBlock
    """

    __slots__ = ("body", "condition")

    def __init__(self, condition: Expression, body: ProgramBlock):
        self.condition = condition
        self.body = body


class BoxStatement(Statement):
    """Like ``box[duration] { statements* }``."""

    # TODO: the `annotations` field maybe should move to `Statement` if it becomes more generally
    # supported.
    __slots__ = ("annotations", "body", "duration")

    def __init__(
        self,
        body: ProgramBlock,
        duration: Expression | None = None,
        annotations: Sequence[Annotation] = (),
    ):
        self.body = body
        self.duration = duration
        self.annotations = tuple(annotations)


class BreakStatement(Statement):
    """AST node for ``break`` statements.  Has no associated information."""

    __slots__ = ()


class ContinueStatement(Statement):
    """AST node for ``continue`` statements.  Has no associated information."""

    __slots__ = ()


class IOModifier(enum.Enum):
    """IO Modifier object"""

    INPUT = enum.auto()
    OUTPUT = enum.auto()


class IODeclaration(ClassicalDeclaration):
    """A declaration of an IO variable."""

    __slots__ = ("modifier",)

    def __init__(self, modifier: IOModifier, type_: ClassicalType, identifier: Identifier):
        super().__init__(type_, identifier)
        self.modifier = modifier


class DefaultCase(Expression):
    """An object representing the `default` special label in switch statements."""

    __slots__ = ()


class SwitchStatementPreview(Statement):
    """AST node for the proposed 'switch-case' extension to OpenQASM 3, before the syntax was
    stabilized.  This corresponds to the :attr:`.ExperimentalFeatures.SWITCH_CASE_V1` logic.

    The stabilized form of the syntax instead uses :class:`.SwitchStatement`."""

    __slots__ = ("cases", "target")

    def __init__(
        self, target: Expression, cases: Iterable[tuple[Iterable[Expression], ProgramBlock]]
    ):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]


class SwitchStatement(Statement):
    """AST node for the stable 'switch' statement of OpenQASM 3.

    The only real difference from an AST form is that the default is required to be separate; it
    cannot be joined with other cases (even though that's meaningless, the V1 syntax permitted it).
    """

    __slots__ = ("cases", "default", "target")

    def __init__(
        self,
        target: Expression,
        cases: Iterable[tuple[Iterable[Expression], ProgramBlock]],
        default: ProgramBlock | None = None,
    ):
        self.target = target
        self.cases = [(tuple(values), case) for values, case in cases]
        self.default = default
