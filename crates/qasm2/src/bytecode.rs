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

use num_bigint::BigUint;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

use crate::error::QASM2ParseError;
use crate::expr::Expr;
use crate::parse::{ClbitId, CregId, GateId, ParamId, QubitId};
use crate::{ClassicalCallableExt, CustomClassical, CustomInstruction, lex, parse};

/// The Rust parser produces an iterator of these `Bytecode` instructions, which comprise an opcode
/// integer for operation distinction, and a free-form tuple containing the operands.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct Bytecode {
    #[pyo3(get)]
    opcode: OpCode,
    #[pyo3(get)]
    operands: Py<PyAny>,
}

/// The operations that are represented by the "bytecode" passed to Python.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, Eq, PartialEq)]
pub enum OpCode {
    // There is only a `Gate` here, not a `GateInBasis`, because in Python space we don't have the
    // same strict typing requirements to satisfy.
    Gate,
    ConditionedGate,
    Measure,
    ConditionedMeasure,
    Reset,
    ConditionedReset,
    Barrier,
    DeclareQreg,
    DeclareCreg,
    DeclareGate,
    EndDeclareGate,
    DeclareOpaque,
    SpecialInclude,
}

// The following structs, with `Expr` or `OpCode` in the name (but not the top-level `OpCode`
// above) build up the tree of symbolic expressions for the parameter applications within gate
// bodies.  We choose to store this in the gate classes that the Python component emits, so it can
// lazily create definitions as required, rather than eagerly binding them as the file is parsed.
//
// In Python space we would usually have the classes inherit from some shared subclass, but doing
// that makes things a little fiddlier with PyO3, and there's no real benefit for our uses.

/// A (potentially folded) floating-point constant value as part of an expression.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprConstant {
    #[pyo3(get)]
    pub value: f64,
}

/// A reference to one of the arguments to the gate.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprArgument {
    #[pyo3(get)]
    pub index: ParamId,
}

/// A unary operation acting on some other part of the expression tree.  This includes the `+` and
/// `-` unary operators, but also any of the built-in scientific-calculator functions.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprUnary {
    #[pyo3(get)]
    pub opcode: UnaryOpCode,
    #[pyo3(get)]
    pub argument: Py<PyAny>,
}

/// A binary operation acting on two other parts of the expression tree.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprBinary {
    #[pyo3(get)]
    pub opcode: BinaryOpCode,
    #[pyo3(get)]
    pub left: Py<PyAny>,
    #[pyo3(get)]
    pub right: Py<PyAny>,
}

/// Some custom callable Python function that the user told us about.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct ExprCustom {
    pub callable: ClassicalCallableExt,
    #[pyo3(get)]
    pub arguments: Py<PyTuple>,
}
#[pymethods]
impl ExprCustom {
    #[pyo3(signature = (*args))]
    fn call(&self, args: Bound<PyTuple>) -> PyResult<f64> {
        match &self.callable {
            ClassicalCallableExt::Builtin(builtin) => builtin
                .call(args.extract::<Vec<f64>>()?.as_slice())
                .map_err(|actual| {
                    PyTypeError::new_err(format!(
                        "argument mismatch: expected {}, got {}",
                        builtin.num_params(),
                        actual
                    ))
                }),
            ClassicalCallableExt::Py { ob, num_params: _ } => {
                ob.bind(args.py()).call1(args).and_then(|ob| ob.extract())
            }
        }
    }
}

/// Discriminator for the different types of unary operator.  We could have a separate class for
/// each of these, but this way involves fewer imports in Python, and also serves to split up the
/// option tree at the top level, so we don't have to test every unary operator before testing
/// other operations.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum UnaryOpCode {
    Negate,
    Cos,
    Exp,
    Ln,
    Sin,
    Sqrt,
    Tan,
}

/// Discriminator for the different types of binary operator.  We could have a separate class for
/// each of these, but this way involves fewer imports in Python, and also serves to split up the
/// option tree at the top level, so we don't have to test every binary operator before testing
/// other operations.
#[pyclass(module = "qiskit._accelerate.qasm2", frozen, eq, skip_from_py_object)]
#[derive(Clone, PartialEq, Eq)]
pub enum BinaryOpCode {
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
}

/// An internal representation of the bytecode that will later be converted to the more free-form
/// [Bytecode] Python-space objects.  This is fairly tightly coupled to Python space; the intent is
/// just to communicate to Python as concisely as possible what it needs to do.  We want to have as
/// little work to do in Python space as possible, since everything is slower there.
///
/// In various enumeration items, we use zero-indexed numeric keys to identify the object rather
/// than its name.  This is much more efficient in Python-space; rather than needing to build and
/// lookup things in a hashmap, we can just build Python lists and index them directly, which also
/// has the advantage of not needing to pass strings to Python for each gate.  It also gives us
/// consistency with how qubits and clbits are tracked; there is no need to track both the register
/// name and the index separately when we can use a simple single index.
pub enum InternalBytecode {
    Gate {
        id: GateId,
        arguments: Vec<f64>,
        qubits: Vec<QubitId>,
    },
    ConditionedGate {
        id: GateId,
        arguments: Vec<f64>,
        qubits: Vec<QubitId>,
        creg: CregId,
        value: BigUint,
    },
    Measure {
        qubit: QubitId,
        clbit: ClbitId,
    },
    ConditionedMeasure {
        qubit: QubitId,
        clbit: ClbitId,
        creg: CregId,
        value: BigUint,
    },
    Reset {
        qubit: QubitId,
    },
    ConditionedReset {
        qubit: QubitId,
        creg: CregId,
        value: BigUint,
    },
    Barrier {
        qubits: Vec<QubitId>,
    },
    DeclareQreg {
        name: String,
        size: usize,
    },
    DeclareCreg {
        name: String,
        size: usize,
    },
    DeclareGate {
        name: String,
        num_qubits: usize,
    },
    GateInBody {
        id: GateId,
        arguments: Vec<Expr>,
        qubits: Vec<QubitId>,
    },
    EndDeclareGate {},
    DeclareOpaque {
        name: String,
        num_qubits: usize,
    },
    SpecialInclude {
        indices: Vec<usize>,
    },
}

impl<'py> IntoPyObject<'py> for InternalBytecode {
    type Target = Bytecode;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    /// Convert the internal bytecode representation to a Python-space one.
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(
            py,
            match self {
                InternalBytecode::Gate {
                    id,
                    arguments,
                    qubits,
                } => Bytecode {
                    opcode: OpCode::Gate,
                    operands: (id, arguments, qubits)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::ConditionedGate {
                    id,
                    arguments,
                    qubits,
                    creg,
                    value,
                } => Bytecode {
                    opcode: OpCode::ConditionedGate,
                    operands: (id, arguments, qubits, creg, value)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::Measure { qubit, clbit } => Bytecode {
                    opcode: OpCode::Measure,
                    operands: (qubit, clbit).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::ConditionedMeasure {
                    qubit,
                    clbit,
                    creg,
                    value,
                } => Bytecode {
                    opcode: OpCode::ConditionedMeasure,
                    operands: (qubit, clbit, creg, value)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::Reset { qubit } => Bytecode {
                    opcode: OpCode::Reset,
                    operands: (qubit,).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::ConditionedReset { qubit, creg, value } => Bytecode {
                    opcode: OpCode::ConditionedReset,
                    operands: (qubit, creg, value).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::Barrier { qubits } => Bytecode {
                    opcode: OpCode::Barrier,
                    operands: (qubits,).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareQreg { name, size } => Bytecode {
                    opcode: OpCode::DeclareQreg,
                    operands: (name, size).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareCreg { name, size } => Bytecode {
                    opcode: OpCode::DeclareCreg,
                    operands: (name, size).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareGate { name, num_qubits } => Bytecode {
                    opcode: OpCode::DeclareGate,
                    operands: (name, num_qubits).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::GateInBody {
                    id,
                    arguments,
                    qubits,
                } => Bytecode {
                    // In Python space, we don't have to be worried about the types of the
                    // parameters changing here, so we can just use `OpCode::Gate` unlike in the
                    // internal bytecode.
                    opcode: OpCode::Gate,
                    operands: (id, arguments.into_pyobject(py)?, qubits)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::EndDeclareGate {} => Bytecode {
                    opcode: OpCode::EndDeclareGate,
                    operands: ().into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareOpaque { name, num_qubits } => Bytecode {
                    opcode: OpCode::DeclareOpaque,
                    operands: (name, num_qubits).into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::SpecialInclude { indices } => Bytecode {
                    opcode: OpCode::SpecialInclude,
                    operands: (indices,).into_pyobject(py)?.into_any().unbind(),
                },
            },
        )
    }
}

/// The custom iterator object that is returned up to Python space for iteration through the
/// bytecode stream.  This is never constructed on the Python side; it is built in Rust space
/// by Python calls to [bytecode_from_string] and [bytecode_from_file].
#[pyclass]
pub struct BytecodeIterator {
    parser_state: parse::State,
    buffer: Vec<Option<InternalBytecode>>,
    buffer_used: usize,
}

impl BytecodeIterator {
    pub fn new(
        tokens: lex::TokenStream,
        include_path: Vec<std::path::PathBuf>,
        custom_instructions: &[CustomInstruction],
        custom_classical: &[CustomClassical],
        strict: bool,
    ) -> PyResult<Self> {
        Ok(BytecodeIterator {
            parser_state: parse::State::new(
                tokens,
                include_path,
                custom_instructions,
                custom_classical,
                strict,
            )
            .map_err(QASM2ParseError::new_err)?,
            buffer: vec![],
            buffer_used: 0,
        })
    }
}

#[pymethods]
impl BytecodeIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Bytecode>> {
        if self.buffer_used >= self.buffer.len() {
            self.buffer.clear();
            self.buffer_used = 0;
            self.parser_state.parse_next(&mut self.buffer)?;
        }
        if self.buffer.is_empty() {
            Ok(None)
        } else {
            self.buffer_used += 1;
            Ok(self.buffer[self.buffer_used - 1]
                .take()
                .map(|bytecode| bytecode.into_pyobject(py))
                .transpose()?
                .map(|x| x.get().clone()))
        }
    }
}
