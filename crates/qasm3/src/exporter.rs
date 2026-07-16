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

use std::sync::Arc;

use crate::ast::{
    Alias, Barrier, BitArray, Break, ClassicalDeclaration, ClassicalType, Continue, Delay,
    Designator, DurationLiteral, DurationUnit, Expression, Float, GateCall, Header, IODeclaration,
    IOModifier, Identifier, IdentifierOrSubscripted, Include, IndexSet, IntegerLiteral, Node,
    Parameter, Program, QuantumBlock, QuantumDeclaration, QuantumGateDefinition,
    QuantumGateSignature, QuantumInstruction, QuantumMeasurement, QuantumMeasurementAssignment,
    Reset, Statement, SubscriptedIdentifier, Version,
};
use std::io::Write;

use crate::printer::BasicPrinter;
use hashbrown::{HashMap, HashSet};
use pyo3::Python;
use pyo3::prelude::*;
use qiskit_circuit::bit::{
    ClassicalRegister, QuantumRegister, Register, ShareableClbit, ShareableQubit,
};
use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{DelayUnit, StandardInstruction};
use qiskit_circuit::operations::{Operation, Param};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr;
use qiskit_util::IndexMap;
use thiserror::Error;

use lazy_static::lazy_static;
use regex::Regex;

type ExporterResult<T> = Result<T, QASM3ExporterError>;

// These are the prefixes used for the loose qubit and bit names.
lazy_static! {
    static ref BIT_PREFIX: &'static str = "_bit";
}
lazy_static! {
    static ref QUBIT_PREFIX: &'static str = "_qubit";
}

// These are the prefixes used for the gate parameters and qubits.
lazy_static! {
    static ref GATE_PARAM_PREFIX: &'static str = "_gate_p";
}
lazy_static! {
    static ref GATE_QUBIT_PREFIX: &'static str = "_gate_q";
}

// These are the gates that are defined by the standard library.
lazy_static! {
    static ref GATES_DEFINED_BY_STDGATES: HashSet<&'static str> = [
        "p", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "rx", "ry", "rz", "cx", "cy", "cz",
        "rzz", "cp", "crx", "cry", "crz", "ch", "swap", "ccx", "cswap", "cu", "CX", "phase",
        "cphase", "id", "u1", "u2", "u3",
    ]
    .into_iter()
    .collect();
}

// These are the reserved keywords in QASM3.
lazy_static! {
    static ref RESERVED_KEYWORDS: HashSet<&'static str> = [
        "OPENQASM",
        "angle",
        "array",
        "barrier",
        "bit",
        "bool",
        "box",
        "break",
        "cal",
        "complex",
        "const",
        "continue",
        "creg",
        "ctrl",
        "def",
        "defcal",
        "defcalgrammar",
        "delay",
        "duration",
        "durationof",
        "else",
        "end",
        "extern",
        "float",
        "for",
        "gate",
        "gphase",
        "if",
        "in",
        "include",
        "input",
        "int",
        "inv",
        "let",
        "measure",
        "mutable",
        "negctrl",
        "output",
        "pow",
        "qreg",
        "qubit",
        "reset",
        "return",
        "sizeof",
        "stretch",
        "uint",
        "while"
    ]
    .into_iter()
    .collect();
}

lazy_static! {
    static ref VALID_IDENTIFIER: Regex = Regex::new(r"(^[\w][\w\d]*$|^\$\d+$)").unwrap();
}

lazy_static! {
    static ref _BAD_IDENTIFIER_CHARACTERS: Regex = Regex::new(r"[^\w\d]").unwrap();
}

lazy_static! {
    static ref _VALID_HARDWARE_QUBIT: Regex = Regex::new(r"\$\d+").unwrap();
}

#[derive(Error, Debug)]
pub enum QASM3ExporterError {
    #[error("Error: {0}")]
    Error(String),
    #[error("PyError: {0}")]
    PyErr(PyErr),
}

impl From<PyErr> for QASM3ExporterError {
    fn from(err: PyErr) -> Self {
        QASM3ExporterError::PyErr(err)
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
enum BitType {
    ShareableQubit(ShareableQubit),
    ShareableClbit(ShareableClbit),
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
enum RegisterType {
    QuantumRegister(QuantumRegister),
    ClassicalRegister(ClassicalRegister),
}

impl RegisterType {
    fn name(&self) -> &str {
        match self {
            RegisterType::QuantumRegister(q) => q.name(),
            RegisterType::ClassicalRegister(c) => c.name(),
        }
    }

    fn bits(&self) -> Vec<BitType> {
        match self {
            RegisterType::QuantumRegister(quantum_register) => quantum_register
                .bits()
                .map(BitType::ShareableQubit)
                .collect(),
            RegisterType::ClassicalRegister(classical_register) => classical_register
                .bits()
                .map(BitType::ShareableClbit)
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
struct Counter {
    current: usize,
}

impl Counter {
    fn new() -> Self {
        Self { current: 0 }
    }
}

impl Iterator for Counter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.current;
        self.current = self.current.wrapping_add(1);
        Some(n)
    }
}

struct BuildScope {
    circuit_data: CircuitData,
    bit_map: HashMap<BitType, BitType>,
}

impl BuildScope {
    fn new(
        circuit_data: &CircuitData,
        qubits: &[ShareableQubit],
        clbits: &[ShareableClbit],
    ) -> Self {
        let mut bit_map: HashMap<BitType, BitType> = HashMap::new();
        for q in qubits.iter() {
            bit_map.insert(
                BitType::ShareableQubit(q.clone()),
                BitType::ShareableQubit(q.clone()),
            );
        }

        for c in clbits.iter() {
            bit_map.insert(
                BitType::ShareableClbit(c.clone()),
                BitType::ShareableClbit(c.clone()),
            );
        }
        Self {
            circuit_data: circuit_data.clone(),
            bit_map,
        }
    }

    fn with_mappings(circuit_data: CircuitData, bit_map: HashMap<BitType, BitType>) -> Self {
        Self {
            circuit_data,
            bit_map,
        }
    }
}

struct SymbolTable {
    symbols: Vec<HashMap<String, Identifier>>,
    bitinfo: Vec<HashMap<BitType, IdentifierOrSubscripted>>,
    reginfo: Vec<HashMap<RegisterType, IdentifierOrSubscripted>>,
    gates: IndexMap<String, QuantumGateDefinition>,
    stdgates: HashSet<String>,
    _counter: Counter,
}

impl SymbolTable {
    fn new() -> Self {
        let symbols = vec![HashMap::new()];
        let bitinfo = vec![HashMap::new()];
        let reginfo = vec![HashMap::new()];
        Self {
            symbols,
            bitinfo,
            reginfo,
            gates: IndexMap::default(),
            stdgates: HashSet::new(),
            _counter: Counter::new(),
        }
    }

    fn bind(&mut self, name: &str) -> ExporterResult<()> {
        if let Some(symbols) = self.symbols.last() {
            if !symbols.contains_key(name) {
                self.bind_no_check(name);
            }
        } else {
            return Err(QASM3ExporterError::Error(format!(
                "Symbol table is empty, cannot bind '{name}'"
            )));
        }

        Ok(())
    }

    fn bind_no_check(&mut self, name: &str) {
        let id = Identifier {
            string: name.to_string(),
        };
        if let Some(last) = self.symbols.last_mut() {
            last.insert(name.to_string(), id);
        }
    }

    fn contains_name(&self, name: &str) -> bool {
        match self.symbols.last() {
            Some(symbols) => symbols.contains_key(name),
            None => false,
        }
    }

    fn symbol_defined(&self, name: &str) -> bool {
        RESERVED_KEYWORDS.contains(name)
            || self.gates.contains_key(name)
            || self
                .symbols
                .iter()
                .rev()
                .any(|symbol| symbol.contains_key(name))
    }

    fn can_shadow_symbol(&self, name: &str) -> bool {
        !self.symbols.last().unwrap().contains_key(name)
            && !self.gates.contains_key(name)
            && !RESERVED_KEYWORDS.contains(name)
    }

    fn escaped_declarable_name(
        &mut self,
        mut name: String,
        allow_rename: bool,
        unique: bool,
    ) -> ExporterResult<String> {
        let name_allowed = if unique {
            |n: &str, this: &SymbolTable| !this.symbol_defined(n)
        } else {
            |n: &str, this: &SymbolTable| this.can_shadow_symbol(n)
        };
        if allow_rename {
            if !VALID_IDENTIFIER.is_match(&name) {
                name = format!("_{}", _BAD_IDENTIFIER_CHARACTERS.replace_all(&name, "_"));
            }
            while !name_allowed(&name, self) {
                name = format!("{}{}", name, self._counter.next().unwrap());
            }
            return Ok(name);
        }
        if !VALID_IDENTIFIER.is_match(&name) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use '{name}' as a name; it is not a valid identifier"
            )));
        }

        if RESERVED_KEYWORDS.contains(name.as_str()) {
            return Err(QASM3ExporterError::Error(format!(
                "cannot use the keyword '{name}' as a variable name"
            )));
        }

        if !name_allowed(&name, self) {
            if self.gates.contains_key(name.as_str()) {
                return Err(QASM3ExporterError::Error(format!(
                    "cannot shadow variable '{name}', as it is already defined as a gate"
                )));
            }

            for scope in self.symbols.iter().rev() {
                if let Some(other) = scope.get(&name) {
                    return Err(QASM3ExporterError::Error(format!(
                        "cannot shadow variable '{name}', as it is already defined as '{other:?}'"
                    )));
                }
            }

            return Err(QASM3ExporterError::Error(format!(
                "internal error: could not locate unshadowable '{name}'"
            )));
        }

        Ok(name.to_string())
    }

    fn add_standard_library_gates(&mut self) {
        for gate in GATES_DEFINED_BY_STDGATES.iter() {
            self.stdgates.insert(gate.to_string());
        }
    }

    fn set_bitinfo(&mut self, id: IdentifierOrSubscripted, bit: BitType) {
        if self.bitinfo.is_empty() {
            self.bitinfo.push(HashMap::new());
        }
        if let Some(last) = self.bitinfo.last_mut() {
            last.insert(bit, id);
        }
    }

    fn get_bitinfo(&self, bit: &BitType) -> Option<&IdentifierOrSubscripted> {
        for info in self.bitinfo.iter().rev() {
            if let Some(id) = info.get(bit) {
                return Some(id);
            }
        }
        None
    }

    fn set_reginfo(&mut self, id: IdentifierOrSubscripted, reg: RegisterType) {
        if self.reginfo.is_empty() {
            self.reginfo.push(HashMap::new());
        }
        if let Some(last) = self.reginfo.last_mut() {
            last.insert(reg, id);
        }
    }

    fn register_gate(
        &mut self,
        op_name: String,
        params_def: Vec<Identifier>,
        qubits: Vec<Identifier>,
        body: QuantumBlock,
    ) -> ExporterResult<()> {
        // Changing the name is not allowed when defining new gates.
        let name = self.escaped_declarable_name(op_name.clone(), false, false)?;
        let _ = self.bind(&name);
        self.gates.insert(
            op_name,
            QuantumGateDefinition {
                quantum_gate_signature: QuantumGateSignature {
                    name: Identifier { string: name },
                    qarg_list: qubits,
                    params: Some(
                        params_def
                            .into_iter()
                            .map(|id| {
                                Expression::IdentifierOrSubscripted(
                                    IdentifierOrSubscripted::Identifier(id),
                                )
                            })
                            .collect(),
                    ),
                },
                quantum_block: body,
            },
        );
        Ok(())
    }

    fn register_bits(
        &mut self,
        name: String,
        bit: &BitType,
        allow_rename: bool,
        allow_hardware_qubit: bool,
    ) -> ExporterResult<Identifier> {
        if allow_hardware_qubit && _VALID_HARDWARE_QUBIT.is_match(&name) {
            if self.symbol_defined(&name) {
                return Err(QASM3ExporterError::Error(format!(
                    "internal error: cannot redeclare hardware qubit {name}"
                )));
            }
        } else {
            self.escaped_declarable_name(name.clone(), allow_rename, false)?;
        }
        let identifier = Identifier {
            string: name.clone(),
        };
        let _ = self.bind(&name);
        self.set_bitinfo(
            IdentifierOrSubscripted::Identifier(identifier.clone()),
            (*bit).to_owned(),
        );
        Ok(identifier)
    }

    fn register_registers(
        &mut self,
        name: String,
        register: &RegisterType,
    ) -> ExporterResult<Identifier> {
        let name = self.escaped_declarable_name(name.clone(), true, false)?;
        let identifier = Identifier {
            string: name.clone(),
        };
        let _ = self.bind(&name);
        self.set_reginfo(
            IdentifierOrSubscripted::Identifier(identifier.clone()),
            (*register).to_owned(),
        );
        Ok(identifier)
    }

    fn push_scope(&mut self) {
        self.symbols.push(HashMap::new());
        self.bitinfo.push(HashMap::new());
        self.reginfo.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.symbols.pop();
        self.bitinfo.pop();
        self.reginfo.pop();
    }

    fn new_context(&mut self) -> Self {
        let mut new_table = SymbolTable::new();
        new_table.gates.clone_from(&self.gates);
        new_table.stdgates.clone_from(&self.stdgates);
        new_table
    }
}

pub struct Exporter {
    includes: Vec<String>,
    basis_gates: Vec<String>,
    disable_constants: bool,
    allow_aliasing: bool,
    indent: String,
}

impl Exporter {
    pub fn new(
        includes: Vec<String>,
        basis_gates: Vec<String>,
        disable_constants: bool,
        allow_aliasing: bool,
        indent: String,
    ) -> Self {
        Self {
            includes,
            basis_gates,
            disable_constants,
            allow_aliasing,
            indent,
        }
    }

    pub fn dumps(&self, circuit_data: &CircuitData, islayout: bool) -> ExporterResult<String> {
        let mut builder = QASM3Builder::new(
            circuit_data,
            islayout,
            self.includes.clone(),
            self.basis_gates.clone(),
            self.disable_constants,
            self.allow_aliasing,
        );
        match builder.build_program() {
            Ok(program) => {
                let mut output = String::new();
                BasicPrinter::new(&mut output, self.indent.to_string(), false)
                    .visit(&Node::Program(&program));
                Ok(output)
            }
            Err(e) => Err(QASM3ExporterError::Error(e.to_string())),
        }
    }

    pub fn dump<W: Write>(
        &self,
        circuit_data: &CircuitData,
        islayout: bool,
        writer: &mut W,
    ) -> ExporterResult<()> {
        let mut builder = QASM3Builder::new(
            circuit_data,
            islayout,
            self.includes.clone(),
            self.basis_gates.clone(),
            self.disable_constants,
            self.allow_aliasing,
        );

        match builder.build_program() {
            Ok(program) => {
                let mut output = String::new();
                let mut printer = BasicPrinter::new(&mut output, self.indent.to_string(), false);
                printer.visit(&Node::Program(&program));
                drop(printer);
                let _ = writer.write_all(output.as_bytes());
                Ok(())
            }
            Err(e) => Err(QASM3ExporterError::Error(e.to_string())),
        }
    }
}

pub struct QASM3Builder {
    _builtin_instr: HashSet<&'static str>,
    loose_bit_prefix: &'static str,
    loose_qubit_prefix: &'static str,
    _gate_param_prefix: &'static str,
    _gate_qubit_prefix: &'static str,
    circuit_scope: BuildScope,
    is_layout: bool,
    symbol_table: SymbolTable,
    global_io_decls: Vec<IODeclaration>,
    includes: Vec<String>,
    basis_gates: Vec<String>,
    disable_constants: bool,
    allow_aliasing: bool,
}

impl<'a> QASM3Builder {
    pub fn new(
        circuit_data: &'a CircuitData,
        is_layout: bool,
        includes: Vec<String>,
        basis_gates: Vec<String>,
        disable_constants: bool,
        allow_aliasing: bool,
    ) -> Self {
        Self {
            _builtin_instr: [
                "barrier",
                "measure",
                "reset",
                "delay",
                "break_loop",
                "continue_loop",
                "store",
            ]
            .into_iter()
            .collect(),
            loose_bit_prefix: &BIT_PREFIX,
            loose_qubit_prefix: &QUBIT_PREFIX,
            _gate_param_prefix: &GATE_PARAM_PREFIX,
            _gate_qubit_prefix: &GATE_QUBIT_PREFIX,
            circuit_scope: BuildScope::new(
                circuit_data,
                circuit_data.qubits().objects(),
                circuit_data.clbits().objects(),
            ),
            is_layout,
            symbol_table: SymbolTable::new(),
            global_io_decls: Vec::new(),
            includes,
            basis_gates,
            disable_constants,
            allow_aliasing,
        }
    }

    #[allow(dead_code)]
    fn new_scope<F>(
        &mut self,
        circuit_data: &CircuitData,
        qubits: Vec<BitType>,
        clbits: Vec<BitType>,
        f: F,
    ) -> ExporterResult<()>
    where
        F: FnOnce(&mut QASM3Builder) -> ExporterResult<()>,
    {
        let current_bitmap = &self.circuit_scope.bit_map;
        let new_qubits: Vec<BitType> = qubits.iter().map(|q| current_bitmap[q].clone()).collect();
        let new_clbits: Vec<BitType> = clbits.iter().map(|c| current_bitmap[c].clone()).collect();

        if circuit_data.num_qubits() != new_qubits.len() {
            return Err(QASM3ExporterError::Error(format!(
                "Tried to push a scope whose circuit needs {} qubits, but only provided {}.",
                circuit_data.num_qubits(),
                new_qubits.len()
            )));
        }

        if circuit_data.num_clbits() != new_clbits.len() {
            return Err(QASM3ExporterError::Error(format!(
                "Tried to push a scope whose circuit needs {} clbits, but only provided {}.",
                circuit_data.num_clbits(),
                new_clbits.len()
            )));
        }

        let mut new_bit_map = HashMap::new();

        for q in circuit_data.qubits().objects().iter() {
            new_bit_map.insert(
                BitType::ShareableQubit(q.clone()),
                BitType::ShareableQubit(q.clone()),
            );
        }
        for c in circuit_data.clbits().objects().iter() {
            new_bit_map.insert(
                BitType::ShareableClbit(c.clone()),
                BitType::ShareableClbit(c.clone()),
            );
        }

        self.symbol_table.push_scope();
        let mut old_scope = std::mem::replace(
            &mut self.circuit_scope,
            BuildScope::with_mappings(circuit_data.clone(), new_bit_map),
        );

        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);

        self.symbol_table.pop_scope();

        result
    }

    fn new_context<F>(&mut self, body: &'a CircuitData, f: F) -> ExporterResult<QuantumBlock>
    where
        F: FnOnce(&mut QASM3Builder) -> QuantumBlock,
    {
        let mut bit_map = HashMap::new();

        for q in body.qubits().objects().iter() {
            bit_map.insert(
                BitType::ShareableQubit(q.clone()),
                BitType::ShareableQubit(q.clone()),
            );
        }
        for c in body.clbits().objects().iter() {
            bit_map.insert(
                BitType::ShareableClbit(c.clone()),
                BitType::ShareableClbit(c.clone()),
            );
        }

        let new_table = self.symbol_table.new_context();
        let mut old_symbol_table = std::mem::replace(&mut self.symbol_table, new_table);
        let mut old_scope = std::mem::replace(
            &mut self.circuit_scope,
            BuildScope::with_mappings(body.clone(), bit_map),
        );
        let result = f(self);
        std::mem::swap(&mut self.circuit_scope, &mut old_scope);
        std::mem::swap(&mut self.symbol_table, &mut old_symbol_table);
        self.symbol_table.gates = old_symbol_table.gates;

        Ok(result)
    }

    fn lookup_bit(&self, bit: &BitType) -> ExporterResult<&IdentifierOrSubscripted> {
        let qubit_ref = self.circuit_scope.bit_map.get(bit).ok_or_else(|| {
            QASM3ExporterError::Error(format!("Bit mapping not found for {bit:?}"))
        })?;
        let id = self
            .symbol_table
            .get_bitinfo(qubit_ref)
            .ok_or_else(|| QASM3ExporterError::Error(format!("Bit not found: {bit:?}")))?;
        Ok(id)
    }

    pub fn build_program(&mut self) -> ExporterResult<Program> {
        self.register_basis_gates();
        let header = self.build_header();

        self.hoist_global_params();
        let classical_decls = self.hoist_classical_bits()?;
        let qubit_decls = self.build_qubit_decls()?;
        let main_stmts = self.build_top_level_stmts()?;

        let mut all_stmts = Vec::new();
        for decl in &self.global_io_decls {
            all_stmts.push(Statement::IODeclaration(decl.clone()));
        }
        for gate in self.symbol_table.gates.values() {
            all_stmts.push(Statement::QuantumGateDefinition(gate.clone()));
        }
        for decl in classical_decls {
            all_stmts.push(decl);
        }
        for decl in qubit_decls {
            all_stmts.push(decl);
        }
        all_stmts.extend(main_stmts);

        Ok(Program {
            header,
            statements: all_stmts,
        })
    }

    fn register_basis_gates(&mut self) {
        for gate in &self.basis_gates {
            let _ = self.symbol_table.bind(gate);
        }
    }

    fn build_header(&mut self) -> Header {
        let includes = self
            .includes
            .iter()
            .map(|fname| {
                if *fname == "stdgates.inc" {
                    self.symbol_table.add_standard_library_gates();
                }
                Include {
                    filename: fname.to_string(),
                }
            })
            .collect();
        Header {
            version: Some(Version {
                version_number: "3.0".to_string(),
            }),
            includes,
        }
    }

    fn hoist_global_params(&mut self) {
        for param in self.circuit_scope.circuit_data.parameters() {
            let identifier = Identifier {
                string: param.name().to_string(),
            };
            let _ = self.symbol_table.bind(param.name());
            self.global_io_decls.push(IODeclaration {
                modifier: IOModifier::Input,
                type_: ClassicalType::Float(Float::Double),
                identifier,
            });
        }
    }

    fn hoist_classical_bits(&mut self) -> ExporterResult<Vec<Statement>> {
        let clbit_indices = self.circuit_scope.circuit_data.clbit_indices();
        let clbits = self.circuit_scope.circuit_data.clbits().objects();
        let has_multiple_registers = clbits.iter().any(|clbit| {
            let bit_info = self
                .circuit_scope
                .circuit_data
                .clbit_indices()
                .get(clbit)
                .unwrap();
            bit_info.registers().len() > 1
        });

        let mut decls = Vec::new();
        if has_multiple_registers {
            if !self.allow_aliasing {
                return Err(QASM3ExporterError::Error(
                    "Some Classical registers in this circuit overlap.".to_string(),
                ));
            }
            for (i, clbit) in clbits.iter().enumerate() {
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", self.loose_bit_prefix, i),
                    &BitType::ShareableClbit(clbit.clone()),
                    true,
                    false,
                )?;
                decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                    type_: ClassicalType::Bit,
                    identifier,
                }));
            }
            let registers: Vec<_> = self.circuit_scope.circuit_data.cregs().to_vec();
            for register in registers {
                let aliased =
                    self.build_aliases(&RegisterType::ClassicalRegister(register.clone()))?;
                decls.push(Statement::Alias(aliased));
            }
            return Ok(decls);
        }
        for (i, clbit) in clbits.iter().enumerate() {
            if clbit_indices
                .get(clbit)
                .is_none_or(|bit_info| bit_info.registers().is_empty())
            {
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", self.loose_bit_prefix, i),
                    &BitType::ShareableClbit(clbit.clone()),
                    true,
                    false,
                )?;

                decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                    type_: ClassicalType::Bit,
                    identifier,
                }));
            }
        }
        for creg in self.circuit_scope.circuit_data.cregs() {
            let identifier = self.symbol_table.register_registers(
                creg.name().to_string(),
                &RegisterType::ClassicalRegister(creg.clone()),
            )?;

            for (i, clbit) in creg.bits().enumerate() {
                self.symbol_table.set_bitinfo(
                    IdentifierOrSubscripted::Subscripted(SubscriptedIdentifier {
                        string: identifier.string.to_string(),
                        subscript: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i32))),
                    }),
                    BitType::ShareableClbit(clbit),
                )
            }
            decls.push(Statement::ClassicalDeclaration(ClassicalDeclaration {
                type_: ClassicalType::BitArray(BitArray(creg.len() as u32)),
                identifier,
            }))
        }
        Ok(decls)
    }

    fn build_qubit_decls(&mut self) -> ExporterResult<Vec<Statement>> {
        let qubits = self.circuit_scope.circuit_data.qubits().objects().clone();
        let has_multiple_registers = qubits.iter().any(|qubit| {
            let bit_info = self
                .circuit_scope
                .circuit_data
                .qubit_indices()
                .get(qubit)
                .unwrap();
            bit_info.registers().len() > 1
        });

        let mut decls: Vec<Statement> = Vec::new();

        if self.is_layout {
            self.loose_qubit_prefix = "$";
            for (i, qubit) in qubits.iter().enumerate() {
                self.symbol_table.register_bits(
                    format!("${i}"),
                    &BitType::ShareableQubit(qubit.clone()),
                    false,
                    true,
                )?;
            }
            return Ok(decls);
        }

        if has_multiple_registers {
            if !self.allow_aliasing {
                return Err(QASM3ExporterError::Error(
                    "Some quantum registers in this circuit overlap.".to_string(),
                ));
            }

            for (i, qubit) in qubits.iter().enumerate() {
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", self.loose_qubit_prefix, i),
                    &BitType::ShareableQubit(qubit.clone()),
                    true,
                    false,
                )?;
                decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                    identifier,
                    designator: None,
                }));
            }
            let registers: Vec<_> = self.circuit_scope.circuit_data.qregs().to_vec();
            for register in registers {
                let aliased =
                    self.build_aliases(&RegisterType::QuantumRegister(register.clone()))?;
                decls.push(Statement::Alias(aliased));
            }
            return Ok(decls);
        }

        let qubit_indices = self.circuit_scope.circuit_data.qubit_indices();

        for (i, qubit) in qubits.iter().enumerate() {
            if qubit_indices
                .get(qubit)
                .is_none_or(|bit_info| bit_info.registers().is_empty())
            {
                let identifier = self.symbol_table.register_bits(
                    format!("{}{}", self.loose_qubit_prefix, i),
                    &BitType::ShareableQubit(qubit.clone()),
                    true,
                    false,
                )?;
                decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                    identifier,
                    designator: None,
                }));
            }
        }
        for qreg in self.circuit_scope.circuit_data.qregs() {
            let identifier = self.symbol_table.register_registers(
                qreg.name().to_string(),
                &RegisterType::QuantumRegister(qreg.clone()),
            )?;
            for (i, qubit) in qreg.bits().enumerate() {
                self.symbol_table.set_bitinfo(
                    IdentifierOrSubscripted::Subscripted(SubscriptedIdentifier {
                        string: identifier.string.to_string(),
                        subscript: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i32))),
                    }),
                    BitType::ShareableQubit(qubit),
                )
            }
            decls.push(Statement::QuantumDeclaration(QuantumDeclaration {
                identifier,
                designator: Some(Designator {
                    expression: Expression::IntegerLiteral(IntegerLiteral(qreg.len() as i32)),
                }),
            }))
        }
        Ok(decls)
    }

    fn build_aliases(&mut self, register: &RegisterType) -> ExporterResult<Alias> {
        let name = self
            .symbol_table
            .register_registers(register.name().to_string(), register)?;
        let mut elements = Vec::new();
        for (i, bit) in register.bits().iter().enumerate() {
            let id = {
                let temp_id = self.lookup_bit(bit)?;
                temp_id.clone()
            };
            let id2 = IdentifierOrSubscripted::Subscripted(SubscriptedIdentifier {
                string: name.string.clone(),
                subscript: Box::new(Expression::IntegerLiteral(IntegerLiteral(i as i32))),
            });
            self.symbol_table.set_bitinfo(id2, bit.clone());
            elements.push(Expression::IdentifierOrSubscripted(id.clone()));
        }
        Ok(Alias {
            identifier: name,
            value: Expression::IndexSet(IndexSet { values: elements }),
        })
    }

    fn build_top_level_stmts(&mut self) -> ExporterResult<Vec<Statement>> {
        let mut stmts = Vec::new();
        let data = self.circuit_scope.circuit_data.data().to_vec();
        for instr in data {
            self.build_instruction(&instr, &mut stmts)?;
        }
        Ok(stmts)
    }

    fn build_instruction(
        &mut self,
        instruction: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let name = instruction.op.name();

        if instruction.op.try_control_flow().is_some() {
            Err(QASM3ExporterError::Error(format!(
                "Control flow {name} is not supported"
            )))
        } else {
            match name {
                "barrier" => self.handle_barrier(instruction, stmts),
                "measure" => self.handle_measure(instruction, stmts),
                "reset" => self.handle_reset(instruction, stmts),
                "delay" => self.handle_delay(instruction, stmts),
                "break_loop" => {
                    stmts.push(Statement::Break(Break {}));
                    Ok(())
                }
                "continue_loop" => {
                    stmts.push(Statement::Continue(Continue {}));
                    Ok(())
                }
                "store" => {
                    panic!("Store is not yet supported");
                }
                _ => {
                    let gate_call = self.build_gate_call(instruction)?;
                    stmts.push(Statement::QuantumInstruction(QuantumInstruction::GateCall(
                        gate_call,
                    )));
                    Ok(())
                }
            }
        }
    }

    fn handle_barrier(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubit_ids = Vec::new();
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_bit(&BitType::ShareableQubit(
                qubits_registry.get(*q).unwrap().clone(),
            ))?;
            qubit_ids.push(id.to_owned());
        }
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Barrier(
            Barrier {
                index_identifier_list: qubit_ids,
            },
        )));
        Ok(())
    }

    fn handle_measure(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let mut qubits = Vec::new();
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_bit(&BitType::ShareableQubit(
                qubits_registry.get(*q).unwrap().clone(),
            ))?;
            qubits.push(id.to_owned());
        }
        let measurement = QuantumMeasurement {
            identifier_list: qubits,
        };

        let cargs = self
            .circuit_scope
            .circuit_data
            .cargs_interner()
            .get(instr.clbits);
        let clbits_registry = self.circuit_scope.circuit_data.clbits();
        let id = self.lookup_bit(&BitType::ShareableClbit(
            clbits_registry.get(cargs[0]).unwrap().clone(),
        ))?;
        stmts.push(Statement::QuantumMeasurementAssignment(
            QuantumMeasurementAssignment {
                identifier: id.to_owned(),
                quantum_measurement: measurement,
            },
        ));
        Ok(())
    }

    fn handle_reset(
        &mut self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_bit(&BitType::ShareableQubit(
                qubits_registry.get(*q).unwrap().clone(),
            ))?;

            stmts.push(Statement::QuantumInstruction(QuantumInstruction::Reset(
                Reset {
                    identifier: id.to_owned(),
                },
            )));
        }
        Ok(())
    }

    fn handle_delay(
        &self,
        instr: &PackedInstruction,
        stmts: &mut Vec<Statement>,
    ) -> ExporterResult<()> {
        if instr.op.num_clbits() > 0 {
            return Err(QASM3ExporterError::Error(
                "Delay cannot have classical bits".to_string(),
            ));
        }
        let delay = self.build_delay(instr)?;
        stmts.push(Statement::QuantumInstruction(QuantumInstruction::Delay(
            delay,
        )));
        Ok(())
    }

    fn build_delay(&self, instr: &PackedInstruction) -> ExporterResult<Delay> {
        let standard_instr = instr.op.standard_instruction();
        let delay_unit = if let StandardInstruction::Delay(delay) = standard_instr {
            delay
        } else {
            return Err(QASM3ExporterError::Error(
                "Expected Delay instruction, but got wrong instruction".to_string(),
            ));
        };
        let param = &instr.params_view()[0];
        let duration: f64 = Python::attach(|py| match param {
            Param::Float(val) => *val,
            Param::ParameterExpression(p) => match p.try_to_value(true) {
                Ok(symbol_expr::Value::Real(val)) => val,
                _ => {
                    panic!("Failed to parse parameter value")
                }
            },
            Param::Obj(obj) => {
                let py_obj = obj.bind(py);
                let py_str = py_obj.str().expect("Failed to call str() on Parameter");
                let name = py_str
                    .str()
                    .expect("Failed to convert PyString to &str")
                    .to_string();
                match name.parse::<f64>() {
                    Ok(val) => val,
                    Err(_) => panic!("Failed to parse parameter value"),
                }
            }
        });

        let mut map = HashMap::new();
        map.insert(DelayUnit::NS, DurationUnit::Nanosecond);
        map.insert(DelayUnit::US, DurationUnit::Microsecond);
        map.insert(DelayUnit::MS, DurationUnit::Millisecond);
        map.insert(DelayUnit::S, DurationUnit::Second);
        map.insert(DelayUnit::DT, DurationUnit::Sample);

        let duration_literal: DurationLiteral = match map.get(&delay_unit) {
            Some(found) => DurationLiteral {
                value: duration,
                unit: found.clone(),
            },
            None => {
                if delay_unit == DelayUnit::PS {
                    DurationLiteral {
                        value: duration / 1000.0,
                        unit: DurationUnit::Nanosecond,
                    }
                } else {
                    return Err(QASM3ExporterError::Error(format!(
                        "Unknown delay unit: {delay_unit}"
                    )));
                }
            }
        };

        let mut qubits = Vec::new();
        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();

        for q in qargs {
            let id = self.lookup_bit(&BitType::ShareableQubit(
                qubits_registry.get(*q).unwrap().clone(),
            ))?;
            qubits.push(id.to_owned());
        }
        Ok(Delay {
            duration: duration_literal,
            qubits,
        })
    }

    fn build_gate_call(&mut self, instr: &PackedInstruction) -> ExporterResult<GateCall> {
        let mut op_name = instr.op.name();
        if op_name == "u" {
            op_name = "U";
        }
        if !self.symbol_table.contains_name(op_name)
            && !self.symbol_table.stdgates.contains(op_name)
        {
            self.define_gate(instr)?;
        }
        let params = if self.disable_constants {
            Python::attach(|_py| {
                instr
                    .params_view()
                    .iter()
                    .map(|param| match param {
                        Param::Float(val) => Expression::Parameter(Parameter {
                            obj: val.to_string(),
                        }),
                        Param::ParameterExpression(p) => {
                            let name = p.to_string();
                            Expression::Parameter(Parameter { obj: name })
                        }
                        Param::Obj(_) => panic!("Objects not supported yet"),
                    })
                    .collect::<Vec<_>>()
            })
        } else {
            return Err(QASM3ExporterError::Error(
                "Constant parameters not supported yet".to_string(),
            ));
        };

        let qargs = self
            .circuit_scope
            .circuit_data
            .qargs_interner()
            .get(instr.qubits);
        let qubits_registry = self.circuit_scope.circuit_data.qubits();
        let mut qubit_ids = Vec::new();
        for q in qargs {
            let id = self.lookup_bit(&BitType::ShareableQubit(
                qubits_registry.get(*q).unwrap().clone(),
            ))?;
            qubit_ids.push(id.to_owned());
        }
        Ok(GateCall {
            quantum_gate_name: Identifier {
                string: op_name.to_string(),
            },
            index_identifier_list: qubit_ids,
            parameters: params,
            modifiers: None,
        })
    }

    #[allow(dead_code)]
    fn define_gate(&mut self, instr: &PackedInstruction) -> ExporterResult<()> {
        let operation = &instr.op;
        let params: Vec<Param> = (0..instr.op.num_params())
            .map(|i| {
                let name = format!("{}_{}", self._gate_param_prefix, i);
                // TODO this need to be achievable more easily
                let symbol = symbol_expr::Symbol::standalone(name, None);
                let expr = ParameterExpression::from_symbol(symbol);
                Param::ParameterExpression(Arc::new(expr))
            })
            .collect();
        if let Some(instruction) = instr.try_definition() {
            let params_def = params
                .iter()
                .enumerate()
                .map(|(i, _p)| {
                    let name = format!("{}_{}", self._gate_param_prefix, i);
                    Identifier {
                        string: name.clone(),
                    }
                })
                .collect::<Vec<_>>();
            let qubits = (0..instruction.num_qubits())
                .map(|i| {
                    let name = format!("{}_{}", self._gate_qubit_prefix, i);
                    Identifier {
                        string: name.clone(),
                    }
                })
                .collect::<Vec<_>>();

            let body = self.new_context(&instruction, |builder| {
                for param in &params_def {
                    let _ = builder.symbol_table.bind(&param.string);
                }
                for (i, q) in instruction.qubits().objects().iter().enumerate() {
                    let name = format!("{}_{}", builder._gate_qubit_prefix, i);
                    let qid = Identifier {
                        string: name.clone(),
                    };
                    let _ = builder.symbol_table.bind(&qid.string);
                    builder.symbol_table.set_bitinfo(
                        IdentifierOrSubscripted::Identifier(qid.clone()),
                        BitType::ShareableQubit(q.to_owned()),
                    );
                }

                let mut stmts_tmp = Vec::new();
                for instr in instruction.data() {
                    let _ = builder.build_instruction(instr, &mut stmts_tmp);
                }
                QuantumBlock {
                    statements: stmts_tmp,
                }
            })?;

            let _ = self.symbol_table.register_gate(
                operation.name().to_string(),
                params_def,
                qubits,
                body,
            );
            Ok(())
        } else {
            Err(QASM3ExporterError::Error(format!(
                "Failed to get definition for this gate: {}",
                operation.name()
            )))
        }
    }
}
