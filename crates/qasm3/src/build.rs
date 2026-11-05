// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use pyo3::prelude::*;
use pyo3::types::{PySequence, PyTuple};

use foldhash::fast::RandomState;

use hashbrown::HashMap;
use qiskit_util::IndexMap;

use oq3_semantics::asg;
use oq3_semantics::symbols::{SymbolId, SymbolTable, SymbolType};
use oq3_semantics::types::{ArrayDims, Type};

use crate::circuit::{PyCircuit, PyCircuitModule, PyClassicalRegister, PyGate, PyQuantumRegister};
use crate::error::QASM3ImporterError;
use crate::expr;

/// Our internal symbol table mapping base symbols to the Python-space object that represents them.
#[derive(Default)]
pub struct PySymbolTable {
    /// Gate-constructor objects.
    pub gates: HashMap<SymbolId, PyGate>,
    /// Scalar `Qubit` objects.
    pub qubits: HashMap<SymbolId, Py<PyAny>>,
    /// Scalar `Clbit` objects.
    pub clbits: HashMap<SymbolId, Py<PyAny>>,
    /// `QuantumRegister` objects.
    pub qregs: HashMap<SymbolId, PyQuantumRegister>,
    /// `ClassicalRegister` objects.
    pub cregs: HashMap<SymbolId, PyClassicalRegister>,
}

struct BuilderState {
    /// The base circuit under construction.
    qc: PyCircuit,
    /// Symbol table mapping AST symbols into typed Python / Rust objects.  This is owned state; we
    /// mutate it and build it up as we parse the AST.
    symbols: PySymbolTable,
    /// Handle to the constructor object for Python-space objects.
    module: PyCircuitModule,
    /// Constructors for gate objects.
    pygates: HashMap<String, PyGate>,
}

impl BuilderState {
    fn declare_classical(
        &mut self,
        py: Python,
        ast_symbols: &SymbolTable,
        decl: &asg::DeclareClassical,
    ) -> PyResult<()> {
        let name_id = decl
            .name()
            .as_ref()
            .map_err(|err| QASM3ImporterError::new_err(format!("internal error: {err:?}")))?;
        let name_symbol = &ast_symbols[name_id];
        match name_symbol.symbol_type() {
            Type::Bit(is_const) => {
                if is_const.clone().into() {
                    Err(QASM3ImporterError::new_err("cannot handle consts"))
                } else if decl.initializer().is_some() {
                    Err(QASM3ImporterError::new_err(
                        "cannot handle initialized bits",
                    ))
                } else {
                    self.add_clbit(py, name_id.clone())
                }
            }
            Type::BitArray(dims, is_const) => {
                if is_const.clone().into() {
                    Err(QASM3ImporterError::new_err("cannot handle consts"))
                } else if decl.initializer().is_some() {
                    Err(QASM3ImporterError::new_err(
                        "cannot handle initialized registers",
                    ))
                } else {
                    match dims {
                        ArrayDims::D1(size) => {
                            self.add_creg(py, name_id.clone(), name_symbol.name(), *size)
                        }
                        _ => Err(QASM3ImporterError::new_err(
                            "cannot handle classical registers with more than one dimension",
                        )),
                    }
                }
            }
            ty => Err(QASM3ImporterError::new_err(format!(
                "unhandled classical type: {ty:?}",
            ))),
        }
    }

    fn declare_quantum(
        &mut self,
        py: Python,
        ast_symbols: &SymbolTable,
        decl: &asg::DeclareQuantum,
    ) -> PyResult<()> {
        let name_id = decl
            .name()
            .as_ref()
            .map_err(|err| QASM3ImporterError::new_err(format!("internal error: {err:?}")))?;
        let name_symbol = &ast_symbols[name_id];
        match name_symbol.symbol_type() {
            Type::Qubit => self.add_qubit(py, name_id.clone()),
            Type::QubitArray(dims) => match dims {
                ArrayDims::D1(size) => {
                    self.add_qreg(py, name_id.clone(), name_symbol.name(), *size)
                }
                _ => Err(QASM3ImporterError::new_err(
                    "cannot handle quantum registers with more than one dimension",
                )),
            },
            _ => unreachable!(),
        }
    }

    fn call_gate(
        &mut self,
        py: Python,
        ast_symbols: &SymbolTable,
        call: &asg::GateCall,
    ) -> PyResult<()> {
        if !call.modifiers().is_empty() {
            return Err(QASM3ImporterError::new_err(
                "gate modifiers not currently handled",
            ));
        }
        let gate_id = call
            .name()
            .as_ref()
            .map_err(|err| QASM3ImporterError::new_err(format!("internal error: {err:?}")))?;
        let gate = self.symbols.gates.get(gate_id).ok_or_else(|| {
            QASM3ImporterError::new_err(format!("internal error: unknown gate {gate_id:?}"))
        })?;
        let params = PyTuple::new(
            py,
            call.params()
                .as_ref()
                .map(|params| params as &[asg::TExpr])
                .unwrap_or_default()
                .iter()
                .map(|param| expr::eval_gate_param(py, &self.symbols, ast_symbols, param))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        let qargs = call.qubits();
        if params.len() != gate.num_params() {
            return Err(QASM3ImporterError::new_err(format!(
                "incorrect number of params to '{}': expected {}, got {}",
                gate.name(),
                gate.num_params(),
                params.len(),
            )));
        }
        if qargs.len() != gate.num_qubits() {
            return Err(QASM3ImporterError::new_err(format!(
                "incorrect number of quantum arguments to '{}': expected {}, got {}",
                gate.name(),
                gate.num_qubits(),
                qargs.len(),
            )));
        }
        let gate_instance = gate.construct(py, params)?;
        for qubits in expr::broadcast_qubits(py, &self.symbols, ast_symbols, qargs)? {
            self.qc.append(
                py,
                self.module
                    .new_instruction(py, gate_instance.clone_ref(py), qubits, ())?,
            )?;
        }
        Ok(())
    }

    fn apply_barrier(
        &mut self,
        py: Python,
        ast_symbols: &SymbolTable,
        barrier: &asg::Barrier,
    ) -> PyResult<()> {
        let qubits = if let Some(asg_qubits) = barrier.qubits().as_ref() {
            // We want any deterministic order for easier circuit reproducibility in Python space,
            // and to include each seen qubit once.  This simply maintains insertion order.
            let mut qubits =
                IndexMap::<*const ::pyo3::ffi::PyObject, Py<PyAny>>::with_capacity_and_hasher(
                    asg_qubits.len(),
                    RandomState::default(),
                );
            for qarg in asg_qubits.iter() {
                let qarg = expr::expect_gate_operand(qarg)?;
                match expr::eval_qarg(py, &self.symbols, ast_symbols, qarg)? {
                    expr::BroadcastItem::Bit(bit) => {
                        let _ = qubits.insert(bit.as_ptr(), bit);
                    }
                    expr::BroadcastItem::Register(register) => {
                        register.into_iter().for_each(|bit| {
                            let _ = qubits.insert(bit.as_ptr(), bit);
                        })
                    }
                }
            }
            PyTuple::new(py, qubits.values())?
        } else {
            // If there's no qargs (represented in the ASG with a `None` rather than an empty
            // vector), it's a barrier over all in-scope qubits, which is all qubits, unless we're
            // in a gate/subroutine body.
            self.qc
                .inner(py)
                .getattr("qubits")?
                .cast::<PySequence>()?
                .to_tuple()?
        };
        let instruction = self.module.new_instruction(
            py,
            self.module.new_barrier(py, qubits.len())?,
            qubits,
            (),
        )?;
        self.qc.append(py, instruction).map(|_| ())
    }

    // Map gates in the symbol table to Qiskit gates in the standard library.
    // Encountering any gates not in the standard library results in raising an exception.
    // Gates mapped via CustomGates will not raise an exception.
    fn map_gate_ids(&mut self, _py: Python, ast_symbols: &SymbolTable) -> PyResult<()> {
        for (name, name_id, defined_num_params, defined_num_qubits) in ast_symbols.gates() {
            let pygate = self.pygates.get(name).ok_or_else(|| {
                QASM3ImporterError::new_err(format!("can't handle non-built-in gate: '{name}'"))
            })?;
            if pygate.num_params() != defined_num_params {
                return Err(QASM3ImporterError::new_err(format!(
                    "given constructor for '{}' expects {} parameters, but is defined as taking {}",
                    pygate.name(),
                    pygate.num_params(),
                    defined_num_params,
                )));
            }
            if pygate.num_qubits() != defined_num_qubits {
                return Err(QASM3ImporterError::new_err(format!(
                    "given constructor for '{}' expects {} qubits, but is defined as taking {}",
                    pygate.name(),
                    pygate.num_qubits(),
                    defined_num_qubits,
                )));
            }
            self.symbols.gates.insert(name_id.clone(), pygate.clone());
        }
        Ok(())
    }

    fn assign(
        &mut self,
        py: Python,
        ast_symbols: &SymbolTable,
        assignment: &asg::Assignment,
    ) -> PyResult<()> {
        // Only handling measurements in this first pass.
        let qarg = match assignment.rvalue().expression() {
            asg::Expr::MeasureExpression(target) => expr::eval_qarg(
                py,
                &self.symbols,
                ast_symbols,
                expr::expect_gate_operand(target.operand())?,
            ),
            expr => Err(QASM3ImporterError::new_err(format!(
                "only measurement assignments are currently supported, not {expr:?}",
            ))),
        }?;
        let carg = expr::eval_measure_carg(py, &self.symbols, ast_symbols, assignment.lvalue())?;
        for (qubits, clbits) in expr::broadcast_measure(py, &qarg, &carg)? {
            self.qc.append(
                py,
                self.module
                    .new_instruction(py, self.module.measure(py), qubits, clbits)?,
            )?
        }
        Ok(())
    }

    fn add_qubit(&mut self, py: Python, ast_symbol: SymbolId) -> PyResult<()> {
        let qubit = self.module.new_qubit(py)?;
        if self
            .symbols
            .qubits
            .insert(ast_symbol, qubit.clone_ref(py))
            .is_some()
        {
            Err(QASM3ImporterError::new_err(
                "internal error: attempted to add the same qubit multiple times",
            ))
        } else {
            self.qc.add_qubit(py, qubit)
        }
    }

    fn add_clbit(&mut self, py: Python, ast_symbol: SymbolId) -> PyResult<()> {
        let clbit = self.module.new_clbit(py)?;
        if self
            .symbols
            .clbits
            .insert(ast_symbol, clbit.clone_ref(py))
            .is_some()
        {
            Err(QASM3ImporterError::new_err(
                "internal error: attempted to add the same clbit multiple times",
            ))
        } else {
            self.qc.add_clbit(py, clbit)
        }
    }

    fn add_qreg<'a, T: IntoPyObject<'a>>(
        &'a mut self,
        py: Python<'a>,
        ast_symbol: SymbolId,
        name: T,
        size: usize,
    ) -> PyResult<()> {
        let qreg = self.module.new_qreg(py, name, size)?;
        self.qc.add_qreg(py, &qreg)?;
        if self.symbols.qregs.insert(ast_symbol, qreg).is_some() {
            Err(QASM3ImporterError::new_err(
                "internal error: attempted to add the same register multiple times",
            ))
        } else {
            Ok(())
        }
    }

    fn add_creg<'py, T: IntoPyObject<'py>>(
        &mut self,
        py: Python<'py>,
        ast_symbol: SymbolId,
        name: T,
        size: usize,
    ) -> PyResult<()> {
        let creg = self.module.new_creg(py, name, size)?;
        self.qc.add_creg(py, &creg)?;
        if self.symbols.cregs.insert(ast_symbol, creg).is_some() {
            Err(QASM3ImporterError::new_err(
                "internal error: attempted to add the same register multiple times",
            ))
        } else {
            Ok(())
        }
    }
}

pub fn convert_asg(
    py: Python,
    program: &asg::Program,
    ast_symbols: &SymbolTable,
    gate_constructors: HashMap<String, PyGate>,
) -> PyResult<PyCircuit> {
    let module = PyCircuitModule::import(py)?;
    let mut state = BuilderState {
        qc: module.new_circuit(py)?,
        symbols: Default::default(),
        pygates: gate_constructors,
        module,
    };

    state.map_gate_ids(py, ast_symbols)?;

    for statement in program.stmts().iter() {
        match statement {
            asg::Stmt::GateCall(call) => state.call_gate(py, ast_symbols, call)?,
            asg::Stmt::DeclareClassical(decl) => state.declare_classical(py, ast_symbols, decl)?,
            asg::Stmt::DeclareQuantum(decl) => state.declare_quantum(py, ast_symbols, decl)?,
            // We ignore gate definitions because the only information we can currently use
            // from them is extracted with `SymbolTable::gates` via `map_gate_ids`.
            asg::Stmt::GateDefinition(_) => (),
            asg::Stmt::Barrier(barrier) => state.apply_barrier(py, ast_symbols, barrier)?,
            asg::Stmt::Assignment(assignment) => state.assign(py, ast_symbols, assignment)?,
            asg::Stmt::Alias(_)
            | asg::Stmt::AnnotatedStmt(_)
            | asg::Stmt::Block(_)
            | asg::Stmt::Box
            | asg::Stmt::Break
            | asg::Stmt::Cal
            | asg::Stmt::Continue
            | asg::Stmt::DeclareHardwareQubit(_)
            | asg::Stmt::DefCal
            | asg::Stmt::DefStmt(_)
            | asg::Stmt::Delay(_)
            | asg::Stmt::End
            | asg::Stmt::ExprStmt(_)
            | asg::Stmt::Extern
            | asg::Stmt::ForStmt(_)
            | asg::Stmt::GPhaseCall(_)
            | asg::Stmt::If(_)
            | asg::Stmt::Include(_)
            | asg::Stmt::InputDeclaration(_)
            | asg::Stmt::ModifiedGPhaseCall(_)
            | asg::Stmt::NullStmt
            | asg::Stmt::OldStyleDeclaration
            | asg::Stmt::OutputDeclaration(_)
            | asg::Stmt::Pragma(_)
            | asg::Stmt::Reset(_)
            | asg::Stmt::SwitchCaseStmt(_)
            | asg::Stmt::While(_) => {
                return Err(QASM3ImporterError::new_err(format!(
                    "this statement is not yet handled during OpenQASM 3 import: {statement:?}"
                )));
            }
        }
    }
    Ok(state.qc)
}
