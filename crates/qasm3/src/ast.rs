// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::fmt::{self, Debug, Display, Formatter};

#[allow(dead_code)]
pub enum Node<'a> {
    Program(&'a Program),
    Header(&'a Header),
    Include(&'a Include),
    Version(&'a Version),
    Expression(&'a Expression),
    ProgramBlock(&'a ProgramBlock),
    QuantumBlock(&'a QuantumBlock),
    QuantumMeasurement(&'a QuantumMeasurement),
    QuantumGateModifier(&'a QuantumGateModifier),
    QuantumGateSignature(&'a QuantumGateSignature),
    ClassicalType(&'a ClassicalType),
    Statement(&'a Statement),
    IndexSet(&'a IndexSet),
}

#[derive(Debug)]
pub struct Program {
    pub header: Header,
    pub statements: Vec<Statement>,
}

#[derive(Debug)]
pub struct Header {
    pub version: Option<Version>,
    pub includes: Vec<Include>,
}

#[derive(Debug)]
pub struct Include {
    pub filename: String,
}

#[derive(Debug)]
pub struct Version {
    pub version_number: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Expression {
    Constant(Constant),
    Parameter(Parameter),
    Range(Range),
    IdentifierOrSubscripted(IdentifierOrSubscripted),
    IntegerLiteral(IntegerLiteral),
    BooleanLiteral(BooleanLiteral),
    BitstringLiteral(BitstringLiteral),
    DurationLiteral(DurationLiteral),
    Unary(Unary),
    Binary(Binary),
    Cast(Cast),
    Index(Index),
    IndexSet(IndexSet),
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum Constant {
    PI,
    Euler,
    Tau,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub obj: String,
}

#[derive(Debug, Clone)]
pub struct Range {
    pub start: Option<Box<Expression>>,
    pub end: Option<Box<Expression>>,
    pub step: Option<Box<Expression>>,
}

#[derive(Debug, Clone)]
pub enum IdentifierOrSubscripted {
    Identifier(Identifier),
    Subscripted(SubscriptedIdentifier),
}

#[derive(Debug, Clone)]
pub struct Identifier {
    pub string: String,
}

#[derive(Debug, Clone)]
pub struct SubscriptedIdentifier {
    pub string: String,
    pub subscript: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct IntegerLiteral(pub(crate) i32);

#[derive(Debug, Clone)]
pub struct BooleanLiteral(pub(crate) bool);

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BitstringLiteral {
    pub value: String,
    pub width: u32,
}

#[derive(Debug, Clone)]
pub struct DurationLiteral {
    pub value: f64,
    pub unit: DurationUnit,
}

#[derive(Debug, Clone)]
pub enum DurationUnit {
    Nanosecond,
    Microsecond,
    Millisecond,
    Second,
    Sample,
}

impl Display for DurationUnit {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let unit_str = match self {
            DurationUnit::Nanosecond => "ns",
            DurationUnit::Microsecond => "us",
            DurationUnit::Millisecond => "ms",
            DurationUnit::Second => "s",
            DurationUnit::Sample => "dt",
        };
        write!(f, "{unit_str}")
    }
}

#[derive(Debug, Clone)]
pub struct Unary {
    pub op: UnaryOp,
    pub operand: Box<Expression>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum UnaryOp {
    LogicNot,
    BitNot,
    Default,
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let op_str = match self {
            UnaryOp::LogicNot => "!",
            UnaryOp::BitNot => "~",
            UnaryOp::Default => "",
        };
        write!(f, "{op_str}")
    }
}

#[derive(Debug, Clone)]
pub struct Binary {
    pub op: BinaryOp,
    pub left: Box<Expression>,
    pub right: Box<Expression>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum BinaryOp {
    BitAnd,
    BitOr,
    BitXor,
    LogicAnd,
    LogicOr,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    ShiftLeft,
    ShiftRight,
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let op_str = match self {
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::LogicAnd => "&&",
            BinaryOp::LogicOr => "||",
            BinaryOp::Less => "<",
            BinaryOp::LessEqual => "<=",
            BinaryOp::Greater => ">",
            BinaryOp::GreaterEqual => ">=",
            BinaryOp::Equal => "==",
            BinaryOp::NotEqual => "!=",
            BinaryOp::ShiftLeft => "<<",
            BinaryOp::ShiftRight => ">>",
        };
        write!(f, "{op_str}")
    }
}

#[derive(Debug, Clone)]
pub struct Cast {
    pub type_: ClassicalType,
    pub operand: Box<Expression>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum ClassicalType {
    Float(Float),
    Bool,
    Int(Int),
    Uint(Uint),
    Bit,
    BitArray(BitArray),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Float {
    Half = 16,
    Single = 32,
    Double = 64,
    Quad = 128,
    Oct = 256,
}

impl Float {
    pub fn iter() -> impl Iterator<Item = Float> {
        [
            Float::Half,
            Float::Single,
            Float::Double,
            Float::Quad,
            Float::Oct,
        ]
        .iter()
        .copied()
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let float_str = match self {
            Float::Half => "16",
            Float::Single => "32",
            Float::Double => "64",
            Float::Quad => "128",
            Float::Oct => "256",
        };
        write!(f, "{float_str}")
    }
}

#[derive(Debug, Clone)]
pub struct Int {
    pub size: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct Uint {
    pub size: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct BitArray(pub(crate) u32);

#[derive(Debug, Clone)]
pub struct Index {
    pub target: Box<Expression>,
    pub index: Box<Expression>,
}

#[derive(Debug, Clone)]
pub struct IndexSet {
    pub values: Vec<Expression>,
}

#[derive(Debug)]
pub struct ProgramBlock {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct QuantumBlock {
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct QuantumMeasurement {
    pub identifier_list: Vec<IdentifierOrSubscripted>,
}

#[derive(Debug, Clone)]
pub struct QuantumGateModifier {
    pub modifier: QuantumGateModifierName,
    pub argument: Option<Expression>,
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum QuantumGateModifierName {
    Ctrl,
    Negctrl,
    Inv,
    Pow,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Statement {
    QuantumDeclaration(QuantumDeclaration),
    ClassicalDeclaration(ClassicalDeclaration),
    IODeclaration(IODeclaration),
    QuantumInstruction(QuantumInstruction),
    QuantumMeasurementAssignment(QuantumMeasurementAssignment),
    Assignment(Assignment),
    QuantumGateDefinition(QuantumGateDefinition),
    Alias(Alias),
    Break(Break),
    Continue(Continue),
}

#[derive(Debug, Clone)]
pub struct QuantumDeclaration {
    pub identifier: Identifier,
    pub designator: Option<Designator>,
}

#[derive(Debug, Clone)]
pub struct Designator {
    pub expression: Expression,
}

#[derive(Debug, Clone)]
pub struct ClassicalDeclaration {
    pub type_: ClassicalType,
    pub identifier: Identifier,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct IODeclaration {
    pub modifier: IOModifier,
    pub type_: ClassicalType,
    pub identifier: Identifier,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum IOModifier {
    Input,
    Output,
}

#[derive(Debug, Clone)]
pub enum QuantumInstruction {
    GateCall(GateCall),
    Reset(Reset),
    Barrier(Barrier),
    Delay(Delay),
}

#[derive(Debug, Clone)]
pub struct GateCall {
    pub quantum_gate_name: Identifier,
    pub index_identifier_list: Vec<IdentifierOrSubscripted>,
    pub parameters: Vec<Expression>,
    pub modifiers: Option<Vec<QuantumGateModifier>>,
}

#[derive(Debug, Clone)]
pub struct Reset {
    pub identifier: IdentifierOrSubscripted,
}

#[derive(Debug, Clone)]
pub struct Barrier {
    pub index_identifier_list: Vec<IdentifierOrSubscripted>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Delay {
    pub duration: DurationLiteral,
    pub qubits: Vec<IdentifierOrSubscripted>,
}

#[derive(Debug, Clone)]
pub struct QuantumMeasurementAssignment {
    pub identifier: IdentifierOrSubscripted,
    pub quantum_measurement: QuantumMeasurement,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    pub lvalue: Identifier,
    pub rvalue: Vec<Identifier>,
}

#[derive(Debug, Clone)]
pub struct QuantumGateDefinition {
    pub quantum_gate_signature: QuantumGateSignature,
    pub quantum_block: QuantumBlock,
}

#[derive(Debug, Clone)]
pub struct QuantumGateSignature {
    pub name: Identifier,
    pub qarg_list: Vec<Identifier>,
    pub params: Option<Vec<Expression>>,
}

#[derive(Debug, Clone)]
pub struct Alias {
    pub identifier: Identifier,
    pub value: Expression,
}

#[derive(Debug, Clone)]
pub struct Break {}

#[derive(Debug, Clone)]
pub struct Continue {}

#[derive(Debug, Hash, Eq, PartialEq)]
pub enum OP<'a> {
    UnaryOp(&'a UnaryOp),
    BinaryOp(&'a BinaryOp),
}
