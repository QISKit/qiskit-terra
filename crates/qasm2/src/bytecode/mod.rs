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
#[cfg(feature = "py")]
use pyo3::prelude::*;

use crate::error::ParseError;
use crate::expr::Expr;
use crate::ext::ClassicalEvaluator;
use crate::parse::{ClbitId, CregId, GateId, QubitId, State};

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
        num_params: usize,
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
        num_params: usize,
    },
    SpecialInclude {
        indices: Vec<usize>,
    },
}

/// Hands out one statement's bytecode at a time; `parse_next` leaves unused slots as `None`.
pub(crate) struct Iter {
    state: State,
    buffer: Vec<Option<InternalBytecode>>,
    handed_out: usize,
    evaluator: ClassicalEvaluator<'static>,
    exhausted: bool,
}

impl Iter {
    pub(crate) fn new(state: State) -> Self {
        Self {
            state,
            buffer: Vec::new(),
            handed_out: 0,
            evaluator: ClassicalEvaluator::detached(),
            exhausted: false,
        }
    }
}

impl Iterator for Iter {
    type Item = Result<InternalBytecode, ParseError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.handed_out < self.buffer.len() {
                let instruction = self.buffer[self.handed_out].take();
                self.handed_out += 1;
                if let Some(instruction) = instruction {
                    return Some(Ok(instruction));
                }
            }
            if self.exhausted {
                return None;
            }
            self.buffer.clear();
            self.handed_out = 0;
            match self.state.parse_next(&mut self.buffer, self.evaluator) {
                Ok(Some(_)) => (),
                Ok(None) => self.exhausted = true,
                Err(err) => {
                    self.buffer.clear();
                    self.exhausted = true;
                    return Some(Err(err));
                }
            }
        }
    }
}

#[cfg(feature = "py")]
mod py;
#[cfg(feature = "py")]
pub use py::*;

#[cfg(feature = "py")]
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
                // `num_params` isn't forwarded: `parse.py` infers it from the call site.
                InternalBytecode::DeclareGate {
                    name,
                    num_qubits,
                    num_params: _,
                } => Bytecode {
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
                    operands: (id, py::GateBodyArguments::new(arguments), qubits)
                        .into_pyobject(py)?
                        .into_any()
                        .unbind(),
                },
                InternalBytecode::EndDeclareGate {} => Bytecode {
                    opcode: OpCode::EndDeclareGate,
                    operands: ().into_pyobject(py)?.into_any().unbind(),
                },
                InternalBytecode::DeclareOpaque {
                    name,
                    num_qubits,
                    num_params: _,
                } => Bytecode {
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
