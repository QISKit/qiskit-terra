// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! An operator-precedence subparser used by the main parser for handling parameter expressions.
//! Instances of this subparser are intended to only live for as long as it takes to parse a single
//! parameter.

use core::f64;

use hashbrown::HashMap;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::ops::ControlFlow;

use crate::error::{
    Position, QASM2ParseError, message_bad_eof, message_generic, message_incorrect_requirement,
};
use crate::lex::{Token, TokenContext, TokenStream, TokenType};
use crate::parse::{GateSymbol, GlobalSymbol, ParamId};
use crate::{ClassicalCallableExt, bytecode};

/// Enum representation of the builtin OpenQASM 2 functions.  The built-in Qiskit parser adds the
/// inverse trigonometric functions, but these are an extension to the version as given in the
/// arXiv paper describing OpenQASM 2.  This enum is essentially just a subset of the [TokenType]
/// enum, to allow for better pattern-match checking in the Rust compiler.
pub enum Function {
    Cos,
    Exp,
    Ln,
    Sin,
    Sqrt,
    Tan,
}

impl From<TokenType> for Function {
    fn from(value: TokenType) -> Self {
        match value {
            TokenType::Cos => Function::Cos,
            TokenType::Exp => Function::Exp,
            TokenType::Ln => Function::Ln,
            TokenType::Sin => Function::Sin,
            TokenType::Sqrt => Function::Sqrt,
            TokenType::Tan => Function::Tan,
            _ => panic!(),
        }
    }
}

impl From<Function> for bytecode::UnaryOpCode {
    fn from(value: Function) -> Self {
        match value {
            Function::Cos => Self::Cos,
            Function::Exp => Self::Exp,
            Function::Ln => Self::Ln,
            Function::Sin => Self::Sin,
            Function::Sqrt => Self::Sqrt,
            Function::Tan => Self::Tan,
        }
    }
}

/// An operator symbol used in the expression parsing.  This is essentially just a subset of the
/// [TokenType] enum (albeit with resolved names) to allow for better pattern-match semantics in
/// the Rust compiler.
#[derive(Clone, Copy)]
enum Op {
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
}

impl Op {
    fn text(&self) -> &'static str {
        match self {
            Self::Plus => "+",
            Self::Minus => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Power => "^",
        }
    }
}

impl From<TokenType> for Op {
    fn from(value: TokenType) -> Self {
        match value {
            TokenType::Plus => Op::Plus,
            TokenType::Minus => Op::Minus,
            TokenType::Asterisk => Op::Multiply,
            TokenType::Slash => Op::Divide,
            TokenType::Caret => Op::Power,
            _ => panic!(),
        }
    }
}

/// An atom of the operator-precedence expression parsing.  This is a stripped-down version of the
/// [Token] and [TokenType] used in the main parser.  We can use a data enum here because we do not
/// need all the expressive flexibility in expecting and accepting many different token types as
/// we do in the main parser; it does not significantly harm legibility to simply do
///
/// ```rust
/// match atom {
///     Atom::Const(val) => (),
///     Atom::Parameter(index) => (),
///     // ...
/// }
/// ```
///
/// where required.
enum Atom {
    LParen,
    RParen,
    Function(Function),
    CustomFunction(ClassicalCallableExt),
    Op(Op),
    Const(f64),
    Parameter(ParamId),
}

/// A tree representation of parameter expressions in OpenQASM 2.  The expression
/// operator-precedence parser will do complete constant folding on operations that only involve
/// floating-point numbers, so these will simply be evaluated into a `Constant` variant rather than
/// represented in full tree form.  For references to the gate parameters, we just store the index
/// of which parameter it is.
pub enum Expr {
    Constant(f64),
    Parameter(ParamId),
    Negate(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Subtract(Box<Expr>, Box<Expr>),
    Multiply(Box<Expr>, Box<Expr>),
    Divide(Box<Expr>, Box<Expr>),
    Power(Box<Expr>, Box<Expr>),
    Function(Function, Box<Expr>),
    CustomFunction(ClassicalCallableExt, Vec<Expr>),
}

impl<'py> IntoPyObject<'py> for Expr {
    type Target = PyAny; // the Python type
    type Output = Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(match self {
            Expr::Constant(value) => bytecode::ExprConstant { value }
                .into_pyobject(py)?
                .into_any(),
            Expr::Parameter(index) => bytecode::ExprArgument { index }
                .into_pyobject(py)?
                .into_any(),
            Expr::Negate(expr) => bytecode::ExprUnary {
                opcode: bytecode::UnaryOpCode::Negate,
                argument: expr.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Add(left, right) => bytecode::ExprBinary {
                opcode: bytecode::BinaryOpCode::Add,
                left: left.into_pyobject(py)?.unbind(),
                right: right.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Subtract(left, right) => bytecode::ExprBinary {
                opcode: bytecode::BinaryOpCode::Subtract,
                left: left.into_pyobject(py)?.unbind(),
                right: right.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Multiply(left, right) => bytecode::ExprBinary {
                opcode: bytecode::BinaryOpCode::Multiply,
                left: left.into_pyobject(py)?.unbind(),
                right: right.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Divide(left, right) => bytecode::ExprBinary {
                opcode: bytecode::BinaryOpCode::Divide,
                left: left.into_pyobject(py)?.unbind(),
                right: right.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Power(left, right) => bytecode::ExprBinary {
                opcode: bytecode::BinaryOpCode::Power,
                left: left.into_pyobject(py)?.unbind(),
                right: right.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::Function(func, expr) => bytecode::ExprUnary {
                opcode: func.into(),
                argument: expr.into_pyobject(py)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
            Expr::CustomFunction(callable, exprs) => bytecode::ExprCustom {
                callable,
                arguments: PyTuple::new(py, exprs)?.unbind(),
            }
            .into_pyobject(py)?
            .into_any(),
        })
    }
}

/// Calculate the binding power of an [Op] when used in a prefix position.  Returns [None] if the
/// operation cannot be used in the prefix position.  The binding power is on the same scale as
/// those returned by [binary_power].
fn prefix_power(op: Op) -> Option<u8> {
    match op {
        Op::Plus | Op::Minus => Some(5),
        _ => None,
    }
}

/// Calculate the binding power of an [Op] when used in an infix position.  The differences between
/// left- and right-binding powers represent the associativity of the operation.
fn binary_power(op: Op) -> (u8, u8) {
    // For new binding powers, use the odd number as the "base" and the even number one larger than
    // it to represent the associativity.  Left-associative operators bind more strongly to the
    // operand on their right (i.e. in `a + b + c`, the first `+` binds to the `b` more tightly
    // than the second, so we get the left-associative form), and right-associative operators bind
    // more strongly to the operand of their left.  The separation of using the odd--even pair is
    // so there's no clash between different operator levels, even accounting for the associativity
    // distinction.
    //
    // All powers should be greater than zero; we need zero free to be the base case in the
    // entry-point to the precedence parser.
    match op {
        Op::Plus | Op::Minus => (1, 2),
        Op::Multiply | Op::Divide => (3, 4),
        Op::Power => (8, 7),
    }
}

enum State {
    Paren,
    Function(Function),
    CustomFunction {
        callable: ClassicalCallableExt,
        arguments: Vec<Expr>,
    },
    Prefix(Op),
    Infix {
        lhs: Expr,
        op: Op,
    },
}
struct EvalState {
    power_min: u8,
    state: State,
    token: Token,
}

/// A subparser used to do the operator-precedence part of the parsing for individual parameter
/// expressions.  The main parser creates a new instance of this struct for each expression it
/// expects, and the instance lives only as long as is required to parse that expression, because
/// it takes temporary responsibility for the [TokenStream] that backs the main parser.
pub struct ExprParser<'a> {
    pub tokens: &'a mut Vec<TokenStream>,
    pub context: &'a mut TokenContext,
    pub gate_symbols: &'a HashMap<String, GateSymbol>,
    pub global_symbols: &'a HashMap<String, GlobalSymbol>,
    pub strict: bool,
}

impl ExprParser<'_> {
    /// Get the next token available in the stack of token streams, popping and removing any
    /// complete streams, except the base case.  Will only return `None` once all streams are
    /// exhausted.
    fn next_token(&mut self) -> PyResult<Option<Token>> {
        let mut pointer = self.tokens.len() - 1;
        while pointer > 1 {
            let out = self.tokens[pointer].next(self.context)?;
            if out.is_some() {
                return Ok(out);
            }
            self.tokens.pop();
            pointer -= 1;
        }
        self.tokens[0].next(self.context)
    }

    /// Peek the next token in the stack of token streams.  This does not remove any complete
    /// streams yet.  Will only return `None` once all streams are exhausted.
    fn peek_token(&mut self) -> PyResult<Option<&Token>> {
        let mut pointer = self.tokens.len() - 1;
        while pointer > 1 && self.tokens[pointer].peek(self.context)?.is_none() {
            pointer -= 1;
        }
        self.tokens[pointer].peek(self.context)
    }

    /// Get the filename associated with the currently active token stream.
    fn current_filename(&self) -> &std::ffi::OsStr {
        &self.tokens[self.tokens.len() - 1].filename
    }

    /// Expect a token of the correct [TokenType].  This is a direct analogue of
    /// [parse::State::expect].  The error variant of the result contains a suitable error message
    /// if the expectation is violated.
    fn expect(&mut self, expected: TokenType, required: &str, cause: &Token) -> PyResult<Token> {
        let token = match self.next_token()? {
            None => {
                return Err(QASM2ParseError::new_err(message_bad_eof(
                    Some(&Position::new(
                        self.current_filename(),
                        cause.line,
                        cause.col,
                    )),
                    required,
                )));
            }
            Some(token) => token,
        };
        if token.ttype == expected {
            Ok(token)
        } else {
            Err(QASM2ParseError::new_err(message_incorrect_requirement(
                required,
                &token,
                self.current_filename(),
            )))
        }
    }

    /// Peek the next token from the stream, and consume and return it only if it has the correct
    /// type.
    fn accept(&mut self, acceptable: TokenType) -> PyResult<Option<Token>> {
        match self.peek_token()? {
            Some(Token { ttype, .. }) if *ttype == acceptable => self.next_token(),
            _ => Ok(None),
        }
    }

    /// Apply a prefix [Op] to the current [expression][Expr].  If the current expression is a
    /// constant floating-point value the application will be eagerly constant-folded, otherwise
    /// the resulting [Expr] will have a tree structure.
    fn apply_prefix(&mut self, prefix: Op, expr: Expr) -> PyResult<Expr> {
        match prefix {
            Op::Plus => Ok(expr),
            Op::Minus => match expr {
                Expr::Constant(val) => Ok(Expr::Constant(-val)),
                _ => Ok(Expr::Negate(Box::new(expr))),
            },
            _ => panic!(),
        }
    }

    /// Apply a binary infix [Op] to the current [expression][Expr].  If both operands have
    /// constant floating-point values the application will be eagerly constant-folded, otherwise
    /// the resulting [Expr] will have a tree structure.
    fn apply_infix(&mut self, infix: Op, lhs: Expr, rhs: Expr, op_token: &Token) -> PyResult<Expr> {
        if let (Expr::Constant(val), Op::Divide) = (&rhs, infix) {
            if *val == 0.0 {
                return Err(QASM2ParseError::new_err(message_generic(
                    Some(&Position::new(
                        self.current_filename(),
                        op_token.line,
                        op_token.col,
                    )),
                    "cannot divide by zero",
                )));
            }
        };
        if let (Expr::Constant(val_l), Expr::Constant(val_r)) = (&lhs, &rhs) {
            // Eagerly constant-fold if possible.
            match infix {
                Op::Plus => Ok(Expr::Constant(val_l + val_r)),
                Op::Minus => Ok(Expr::Constant(val_l - val_r)),
                Op::Multiply => Ok(Expr::Constant(val_l * val_r)),
                Op::Divide => Ok(Expr::Constant(val_l / val_r)),
                Op::Power => Ok(Expr::Constant(val_l.powf(*val_r))),
            }
        } else {
            // If not, we have to build a tree.
            let id_l = Box::new(lhs);
            let id_r = Box::new(rhs);
            match infix {
                Op::Plus => Ok(Expr::Add(id_l, id_r)),
                Op::Minus => Ok(Expr::Subtract(id_l, id_r)),
                Op::Multiply => Ok(Expr::Multiply(id_l, id_r)),
                Op::Divide => Ok(Expr::Divide(id_l, id_r)),
                Op::Power => Ok(Expr::Power(id_l, id_r)),
            }
        }
    }

    /// Apply a "scientific calculator" built-in function to an [expression][Expr].  If the operand
    /// is a constant, the function will be constant-folded to produce a new constant expression,
    /// otherwise a tree-form [Expr] is returned.
    fn apply_function(&mut self, func: Function, expr: Expr, token: &Token) -> PyResult<Expr> {
        match expr {
            Expr::Constant(val) => match func {
                Function::Cos => Ok(Expr::Constant(val.cos())),
                Function::Exp => Ok(Expr::Constant(val.exp())),
                Function::Ln => {
                    if val > 0.0 {
                        Ok(Expr::Constant(val.ln()))
                    } else {
                        Err(QASM2ParseError::new_err(message_generic(
                            Some(&Position::new(
                                self.current_filename(),
                                token.line,
                                token.col,
                            )),
                            &format!(
                                "failure in constant folding: cannot take ln of non-positive {val}"
                            ),
                        )))
                    }
                }
                Function::Sin => Ok(Expr::Constant(val.sin())),
                Function::Sqrt => {
                    if val >= 0.0 {
                        Ok(Expr::Constant(val.sqrt()))
                    } else {
                        Err(QASM2ParseError::new_err(message_generic(
                            Some(&Position::new(
                                self.current_filename(),
                                token.line,
                                token.col,
                            )),
                            &format!(
                                "failure in constant folding: cannot take sqrt of negative {val}"
                            ),
                        )))
                    }
                }
                Function::Tan => Ok(Expr::Constant(val.tan())),
            },
            _ => Ok(Expr::Function(func, Box::new(expr))),
        }
    }

    fn apply_custom_function(
        &mut self,
        callable: &ClassicalCallableExt,
        exprs: Vec<Expr>,
        token: &Token,
    ) -> PyResult<Expr> {
        if exprs.len() != callable.num_params() {
            return Err(QASM2ParseError::new_err(message_generic(
                Some(&self.cur_position_of(token)),
                &format!(
                    "custom function argument-count mismatch: expected {}, saw {}",
                    callable.num_params(),
                    exprs.len(),
                ),
            )));
        }
        let as_f64 = exprs
            .iter()
            .map(|expr| match expr {
                Expr::Constant(val) => Some(*val),
                _ => None,
            })
            .collect::<Option<Vec<f64>>>();
        let Some(floats) = as_f64 else {
            return Ok(Expr::CustomFunction(callable.clone(), exprs));
        };
        match callable {
            ClassicalCallableExt::Builtin(builtin) => {
                Ok(Expr::Constant(builtin.call(&floats).expect(
                    "caller of apply_custom_function should validate the number of parameters",
                )))
            }
            ClassicalCallableExt::Py { ob, num_params: _ } => Python::attach(|py| {
                let chain_exception = |msg, base| {
                    let pos = Position::new(self.current_filename(), token.line, token.col);
                    let err = QASM2ParseError::new_err(message_generic(Some(&pos), msg));
                    err.set_cause(py, Some(base));
                    err
                };
                ob.bind(py)
                    .call1(PyTuple::new(py, floats)?)
                    .map_err(|inner| {
                        chain_exception(
                            "caught exception when constant folding with user-defined function",
                            inner,
                        )
                    })?
                    .extract::<f64>()
                    .map_err(|inner| {
                        chain_exception(
                            "user-defined function returned non-float during constant folding",
                            inner,
                        )
                    })
                    .map(Expr::Constant)
            }),
        }
    }

    /// If in `strict` mode, and we have a trailing comma, emit a suitable error message.
    fn check_trailing_comma(&self, comma: Option<&Token>) -> PyResult<()> {
        match (self.strict, comma) {
            (true, Some(token)) => Err(QASM2ParseError::new_err(message_generic(
                Some(&Position::new(
                    self.current_filename(),
                    token.line,
                    token.col,
                )),
                "[strict] trailing commas in parameter and qubit lists are forbidden",
            ))),
            _ => Ok(()),
        }
    }

    /// Convert the given general [Token] into the expression-parser-specific [Atom], if possible.
    /// Not all [Token]s have a corresponding [Atom]; if this is the case, the return value is
    /// `Ok(None)`.  The error variant is returned if the next token is grammatically valid, but
    /// not semantically, such as an identifier for a value of an incorrect type.
    fn try_atom_from_token(&self, token: &Token) -> PyResult<Option<Atom>> {
        match token.ttype {
            TokenType::LParen => Ok(Some(Atom::LParen)),
            TokenType::RParen => Ok(Some(Atom::RParen)),
            TokenType::Minus
            | TokenType::Plus
            | TokenType::Asterisk
            | TokenType::Slash
            | TokenType::Caret => Ok(Some(Atom::Op(token.ttype.into()))),
            TokenType::Cos
            | TokenType::Exp
            | TokenType::Ln
            | TokenType::Sin
            | TokenType::Sqrt
            | TokenType::Tan => Ok(Some(Atom::Function(token.ttype.into()))),
            // This deliberately parses an _integer_ token as a float, since all OpenQASM 2.0
            // integers can be interpreted as floats, and doing that allows us to gracefully handle
            // cases where a huge float would overflow a `usize`.  Never mind that in such a case,
            // there's almost certainly precision loss from the floating-point representing
            // having insufficient mantissa digits to faithfully represent the angle mod 2pi;
            // that's not our fault in the parser.
            TokenType::Real | TokenType::Integer => Ok(Some(Atom::Const(token.real(self.context)))),
            TokenType::Pi => Ok(Some(Atom::Const(f64::consts::PI))),
            TokenType::Id => {
                let id = token.text(self.context);
                match self.gate_symbols.get(id) {
                    Some(GateSymbol::Parameter { index }) => Ok(Some(Atom::Parameter(*index))),
                    Some(GateSymbol::Qubit { .. }) => {
                        Err(QASM2ParseError::new_err(message_generic(
                            Some(&Position::new(
                                self.current_filename(),
                                token.line,
                                token.col,
                            )),
                            &format!("'{id}' is a gate qubit, not a parameter"),
                        )))
                    }
                    None => match self.global_symbols.get(id) {
                        Some(GlobalSymbol::Classical(callable)) => {
                            Ok(Some(Atom::CustomFunction(callable.clone())))
                        }
                        _ => Err(QASM2ParseError::new_err(message_generic(
                            Some(&Position::new(
                                self.current_filename(),
                                token.line,
                                token.col,
                            )),
                            &format!(
                                "'{id}' is not a parameter or custom instruction defined in this scope",
                            ),
                        ))),
                    },
                }
            }
            _ => Ok(None),
        }
    }

    /// Peek at the next [Atom] (and backing [Token]) if the next token exists and can be converted
    /// into a valid [Atom].  If it can't, or if we are at the end of the input, the `None` variant
    /// is returned.
    fn peek_atom(&mut self) -> PyResult<Option<(Atom, Token)>> {
        if let Some(&token) = self.peek_token()? {
            if let Ok(Some(atom)) = self.try_atom_from_token(&token) {
                Ok(Some((atom, token)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    fn next_atom(&mut self, power_min: u8, cause: &Token) -> PyResult<(Token, Atom)> {
        let description = if power_min == 0 {
            "an expression"
        } else {
            "a missing operand"
        };
        let token = self.next_token()?.ok_or_else(|| {
            QASM2ParseError::new_err(message_bad_eof(
                Some(&self.cur_position_of(cause)),
                description,
            ))
        })?;
        let atom = self.try_atom_from_token(&token)?.ok_or_else(|| {
            QASM2ParseError::new_err(message_incorrect_requirement(
                description,
                &token,
                self.current_filename(),
            ))
        })?;
        Ok((token, atom))
    }

    #[inline]
    fn cur_position_of(&self, token: &Token) -> Position<'_> {
        Position::new(self.current_filename(), token.line, token.col)
    }

    /// Parse an expected expression part, allowing the logic to `Continue` if we can.
    ///
    /// If we need to break and look for a new subexpression component, return `Break` with the
    /// stack update instead.
    #[inline]
    fn expect_expression_initial(
        &mut self,
        power_min: u8,
        cause: &Token,
    ) -> PyResult<ControlFlow<(EvalState, u8), Expr>> {
        let (token, atom) = self.next_atom(power_min, cause)?;
        let break_at = |state: State, new_power: u8| {
            let state = EvalState {
                power_min,
                token,
                state,
            };
            ControlFlow::Break((state, new_power))
        };
        match atom {
            Atom::LParen => Ok(break_at(State::Paren, 0)),
            Atom::RParen => {
                let msg = if power_min == 0 {
                    "did not find an expected expression"
                } else {
                    "the parenthesis closed, but there was a missing operand"
                };
                let pos = self.cur_position_of(&token);
                Err(QASM2ParseError::new_err(message_generic(Some(&pos), msg)))
            }
            Atom::Function(func) => {
                self.expect(TokenType::LParen, "an opening parenthesis", &token)?;
                Ok(break_at(State::Function(func), 0))
            }
            Atom::CustomFunction(callable) => {
                self.expect(TokenType::LParen, "an opening parenthesis", &token)?;
                match self.accept(TokenType::RParen)? {
                    Some(_) => self
                        .apply_custom_function(&callable, Vec::new(), &token)
                        .map(ControlFlow::Continue),
                    None => Ok(break_at(
                        State::CustomFunction {
                            arguments: Vec::with_capacity(callable.num_params()),
                            callable,
                        },
                        0,
                    )),
                }
            }
            Atom::Op(op) => {
                let Some(power) = prefix_power(op) else {
                    return Err(QASM2ParseError::new_err(message_generic(
                        Some(&self.cur_position_of(&token)),
                        &format!("'{}' is not a valid unary operator", op.text()),
                    )));
                };
                Ok(break_at(State::Prefix(op), power))
            }
            Atom::Const(val) => Ok(ControlFlow::Continue(Expr::Constant(val))),
            Atom::Parameter(val) => Ok(ControlFlow::Continue(Expr::Parameter(val))),
        }
    }

    /// Parse the expected terminators of the partial expression stored in `EvalState`.
    ///
    /// Returns the evaluated expression if possible (Continue), or the new stack frame to push if
    /// not (Break).
    #[inline]
    fn expect_expression_terminator(
        &mut self,
        expr: Expr,
        eval_state: EvalState,
    ) -> PyResult<ControlFlow<(EvalState, u8), (Expr, u8)>> {
        let EvalState {
            state,
            token,
            power_min,
        } = eval_state;
        let continue_from = |expr| ControlFlow::Continue((expr, power_min));
        match state {
            State::Paren => {
                self.expect(TokenType::RParen, "a closing parenthesis", &token)?;
                Ok(continue_from(expr))
            }
            State::Function(func) => {
                let comma = self.accept(TokenType::Comma)?;
                self.check_trailing_comma(comma.as_ref())?;
                self.expect(TokenType::RParen, "a closing parenthesis", &token)?;
                self.apply_function(func, expr, &token).map(continue_from)
            }
            State::CustomFunction {
                callable,
                mut arguments,
            } => {
                arguments.push(expr);
                let comma = self.accept(TokenType::Comma)?;
                if comma.is_none() {
                    self.expect(TokenType::RParen, "a closing parenthesis", &token)?;
                } else if self.accept(TokenType::RParen)?.is_some() {
                    self.check_trailing_comma(comma.as_ref())?;
                } else {
                    let state = EvalState {
                        power_min,
                        state: State::CustomFunction {
                            callable,
                            arguments,
                        },
                        token,
                    };
                    return Ok(ControlFlow::Break((state, 0)));
                };
                self.apply_custom_function(&callable, arguments, &token)
                    .map(continue_from)
            }
            State::Prefix(op) => self.apply_prefix(op, expr).map(continue_from),
            State::Infix { lhs, op } => self.apply_infix(op, lhs, expr, &token).map(continue_from),
        }
    }

    /// Parse a single expression completely. This is the only public entry point to the
    /// operator-precedence parser.
    ///
    /// .. note::
    ///
    ///     This evaluates in a floating-point context, including evaluating integer tokens, since
    ///     the only places that expressions are valid in OpenQASM 2 is during gate applications.
    pub fn parse_expression(&mut self, cause: &Token) -> PyResult<Expr> {
        // We don't store the "root" case of expression parsing as a stack entry so that in the
        // happy (and _massively_ most common) case of a floating-point literal, there's no heap
        // allocation at all to manage the state.
        let mut stack = Vec::new();
        let mut power_min: u8 = 0;

        'expr: loop {
            // The entry point to this loop represents a parse state where we are starting a new
            // (sub)expression.  Simply put: it's wherever a recursive-descent parser would call
            // `parse_expression` recursively.
            let mut expr = match self.expect_expression_initial(power_min, cause)? {
                ControlFlow::Break((state, power)) => {
                    stack.push(state);
                    power_min = power;
                    continue 'expr;
                }
                ControlFlow::Continue(expr) => expr,
            };

            'infix: loop {
                // We've reached something that _might_ be a complete expression, but it also might
                // just be the left-hand side of an infix operator.  Let's see.
                if let Some((Atom::Op(op), peeked_token)) = self.peek_atom()? {
                    self.next_token()?; // It matched, so consume it.
                    let (power_l, power_r) = binary_power(op);
                    // While the operator (if any) on the left binds tighter than `op`, we now know
                    // it's complete and so can eagerly evaluate it.  This isn't necessary for
                    // correctness in the parse, but doing it stops fully associative expressions
                    // like `1.0 + 1.0 + 1.0 + ...` from causing stack growth (and our
                    // `State::Infix` variant is written expecting the eager evaluation).
                    while power_min > power_l {
                        // If nothing else, the root of the stack should have `power_min == 0`,
                        // and the left-binding power of our operator can't be lower than that.
                        let prev = stack
                            .pop()
                            .expect("tight binding requires a partial operation");
                        (expr, power_min) = match self.expect_expression_terminator(expr, prev)? {
                            ControlFlow::Break(_) => {
                                panic!("tight binding requires a partial operation")
                            }
                            ControlFlow::Continue((expr, power)) => (expr, power),
                        };
                    }
                    // The new binding power is tighter than whatever's remaining to the left of us,
                    // so now we have to ask for a new expression to complete our right-hand side.
                    stack.push(EvalState {
                        power_min,
                        state: State::Infix { lhs: expr, op },
                        token: peeked_token,
                    });
                    power_min = power_r;
                    continue 'expr;
                }

                // We've reached the right-hand edge of a complete expression; the closest left-most
                // subexpression indicator (an operator, a comma, a bracket, etc) binds more tightly
                // than anything to us from the right, so we need to pop from the stack, evaluate
                // its terminators, then check again for an infix against the new power.
                let Some(prev) = stack.pop() else {
                    // If the stack is exhausted, we've entirely evaluated the expression.
                    return Ok(expr);
                };
                (expr, power_min) = match self.expect_expression_terminator(expr, prev)? {
                    ControlFlow::Break((state, power)) => {
                        stack.push(state);
                        power_min = power;
                        continue 'expr;
                    }
                    ControlFlow::Continue((expr, power)) => (expr, power),
                };
                // This statement is actually a no-op, but the loop is long so let's be explicit.
                continue 'infix;
            }
        }
    }
}
