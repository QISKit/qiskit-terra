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

use hashbrown::HashMap;

use std::fmt::Write;

use crate::ast::{
    Alias, Assignment, Barrier, Binary, BinaryOp, BitArray, BooleanLiteral, Cast,
    ClassicalDeclaration, ClassicalType, Constant, Delay, DurationLiteral, Expression, Float,
    GateCall, Header, Identifier, IdentifierOrSubscripted, Include, Index, IndexSet, Int,
    IntegerLiteral, Node, OP, Parameter, Program, ProgramBlock, QuantumBlock, QuantumDeclaration,
    QuantumGateDefinition, QuantumGateModifier, QuantumGateModifierName, QuantumGateSignature,
    QuantumInstruction, QuantumMeasurement, QuantumMeasurementAssignment, Range, Reset, Statement,
    SubscriptedIdentifier, Uint, Unary, UnaryOp, Version,
};

#[derive(Debug)]
struct BindingPower {
    left: u8,
    right: u8,
}

impl BindingPower {
    fn new(left: u8, right: u8) -> Self {
        BindingPower { left, right }
    }
}

pub struct BasicPrinter<'a> {
    stream: &'a mut String,
    indent: String,
    current_indent: usize,
    _chain_else_if: bool,
    constant_lookup: HashMap<Constant, &'static str>,
    modifier_lookup: HashMap<QuantumGateModifierName, &'static str>,
    float_width_lookup: HashMap<Float, String>,
    binding_power: HashMap<OP<'a>, BindingPower>,
}

impl<'a> BasicPrinter<'a> {
    pub fn new(stream: &'a mut String, indent: String, _chain_else_if: bool) -> Self {
        let mut constant_lookup = HashMap::new();
        constant_lookup.insert(Constant::PI, "pi");
        constant_lookup.insert(Constant::Euler, "euler");
        constant_lookup.insert(Constant::Tau, "tau");

        let mut modifier_lookup = HashMap::new();
        modifier_lookup.insert(QuantumGateModifierName::Ctrl, "ctrl");
        modifier_lookup.insert(QuantumGateModifierName::Negctrl, "negctrl");
        modifier_lookup.insert(QuantumGateModifierName::Inv, "inv");
        modifier_lookup.insert(QuantumGateModifierName::Pow, "pow");

        let float_width_lookup = Float::iter().map(|t| (t, t.to_string())).collect();

        let mut binding_power = HashMap::new();
        binding_power.insert(OP::UnaryOp(&UnaryOp::LogicNot), BindingPower::new(0, 22));
        binding_power.insert(OP::UnaryOp(&UnaryOp::BitNot), BindingPower::new(0, 22));
        binding_power.insert(
            OP::BinaryOp(&BinaryOp::ShiftLeft),
            BindingPower::new(15, 16),
        );
        binding_power.insert(
            OP::BinaryOp(&BinaryOp::ShiftRight),
            BindingPower::new(15, 16),
        );
        binding_power.insert(OP::BinaryOp(&BinaryOp::Less), BindingPower::new(13, 14));
        binding_power.insert(
            OP::BinaryOp(&BinaryOp::LessEqual),
            BindingPower::new(13, 14),
        );
        binding_power.insert(OP::BinaryOp(&BinaryOp::Greater), BindingPower::new(13, 14));
        binding_power.insert(
            OP::BinaryOp(&BinaryOp::GreaterEqual),
            BindingPower::new(13, 14),
        );
        binding_power.insert(OP::BinaryOp(&BinaryOp::Equal), BindingPower::new(11, 12));
        binding_power.insert(OP::BinaryOp(&BinaryOp::NotEqual), BindingPower::new(11, 12));
        binding_power.insert(OP::BinaryOp(&BinaryOp::BitAnd), BindingPower::new(9, 10));
        binding_power.insert(OP::BinaryOp(&BinaryOp::BitXor), BindingPower::new(7, 8));
        binding_power.insert(OP::BinaryOp(&BinaryOp::BitOr), BindingPower::new(5, 6));
        binding_power.insert(OP::BinaryOp(&BinaryOp::LogicAnd), BindingPower::new(3, 4));
        binding_power.insert(OP::BinaryOp(&BinaryOp::LogicOr), BindingPower::new(1, 2));

        BasicPrinter {
            stream,
            indent,
            current_indent: 0,
            _chain_else_if,
            constant_lookup,
            modifier_lookup,
            float_width_lookup,
            binding_power,
        }
    }

    pub fn visit(&mut self, node: &Node) {
        match node {
            Node::Program(node) => self.visit_program(node),
            Node::Header(node) => self.visit_header(node),
            Node::Include(node) => self.visit_include(node),
            Node::Version(node) => self.visit_version(node),
            Node::Expression(node) => self.visit_expression(node),
            Node::ProgramBlock(node) => self.visit_program_block(node),
            Node::QuantumBlock(node) => self.visit_quantum_block(node),
            Node::QuantumMeasurement(node) => self.visit_quantum_measurement(node),
            Node::QuantumGateModifier(node) => self.visit_quantum_gate_modifier(node),
            Node::QuantumGateSignature(node) => self.visit_quantum_gate_signature(node),
            Node::ClassicalType(node) => self.visit_classical_type(node),
            Node::Statement(node) => self.visit_statement(node),
            Node::IndexSet(node) => self.visit_index_set(node),
        }
    }

    fn start_line(&mut self) {
        write!(self.stream, "{}", self.indent.repeat(self.current_indent)).unwrap();
    }

    fn end_statement(&mut self) {
        writeln!(self.stream, ";").unwrap();
    }

    fn end_line(&mut self) {
        writeln!(self.stream).unwrap();
    }

    fn write_statement(&mut self, line: &str) {
        self.start_line();
        write!(self.stream, "{line}").unwrap();
        self.end_statement();
    }

    fn visit_program(&mut self, node: &Program) {
        self.visit(&Node::Header(&node.header));
        for statement in node.statements.iter() {
            self.visit_statement(statement);
        }
    }

    fn visit_header(&mut self, node: &Header) {
        if let Some(version) = &node.version {
            self.visit(&Node::Version(version))
        };
        for include in node.includes.iter() {
            self.visit(&Node::Include(include));
        }
    }

    fn visit_include(&mut self, node: &Include) {
        self.write_statement(&format!("include \"{}\"", node.filename));
    }

    fn visit_version(&mut self, node: &Version) {
        self.write_statement(&format!("OPENQASM {}", node.version_number));
    }

    fn visit_expression(&mut self, node: &Expression) {
        match node {
            Expression::Constant(expression) => self.visit_constant(expression),
            Expression::Parameter(expression) => self.visit_parameter(expression),
            Expression::Range(expression) => self.visit_range(expression),
            Expression::IdentifierOrSubscripted(expression) => match expression {
                IdentifierOrSubscripted::Identifier(identifier) => {
                    self.visit_identifier(identifier)
                }
                IdentifierOrSubscripted::Subscripted(subscripted_identifier) => {
                    self.visit_subscript_identifier(subscripted_identifier)
                }
            },
            Expression::IntegerLiteral(expression) => self.visit_integer_literal(expression),
            Expression::BooleanLiteral(expression) => self.visit_boolean_literal(expression),
            Expression::BitstringLiteral(_) => {
                panic!("BasicPrinter: BitStringLiteral has not been supported yet.")
            }
            Expression::DurationLiteral(expression) => self.visit_duration_literal(expression),
            Expression::Unary(expression) => self.visit_unary(expression),
            Expression::Binary(expression) => self.visit_binary(expression),
            Expression::Cast(expression) => self.visit_cast(expression),
            Expression::Index(expression) => self.visit_index(expression),
            Expression::IndexSet(index_set) => self.visit_index_set(index_set),
        }
    }

    fn visit_constant(&mut self, expression: &Constant) {
        write!(self.stream, "{}", self.constant_lookup[expression]).unwrap();
    }

    fn visit_parameter(&mut self, expression: &Parameter) {
        write!(self.stream, "{}", expression.obj).unwrap();
    }

    fn visit_range(&mut self, expression: &Range) {
        if let Some(start) = &expression.start {
            self.visit_expression(start);
        }
        write!(self.stream, ":").unwrap();
        if let Some(step) = &expression.step {
            self.visit_expression(step);
            write!(self.stream, ":").unwrap();
        }
        if let Some(end) = &expression.end {
            self.visit_expression(end);
        }
    }

    fn visit_identifier(&mut self, expression: &Identifier) {
        write!(self.stream, "{}", expression.string).unwrap();
    }

    fn visit_subscript_identifier(&mut self, expression: &SubscriptedIdentifier) {
        write!(self.stream, "{}", expression.string).unwrap();
        write!(self.stream, "[").unwrap();
        self.visit_expression(&expression.subscript);
        write!(self.stream, "]").unwrap();
    }

    fn visit_integer_literal(&mut self, expression: &IntegerLiteral) {
        write!(self.stream, "{}", expression.0).unwrap();
    }

    fn visit_boolean_literal(&mut self, expression: &BooleanLiteral) {
        write!(
            self.stream,
            "{}",
            if expression.0 { "true" } else { "false" }
        )
        .unwrap();
    }

    fn visit_modifier_sequence(
        &mut self,
        nodes: &[QuantumGateModifier],
        start: &str,
        end: &str,
        separator: &str,
    ) {
        if !start.is_empty() {
            write!(self.stream, "{start}").unwrap();
        }
        for node in nodes.iter().take(nodes.len() - 1) {
            self.visit_quantum_gate_modifier(node);
            write!(self.stream, "{separator}").unwrap();
        }
        if let Some(last) = nodes.last() {
            self.visit_quantum_gate_modifier(last);
        }
        if !end.is_empty() {
            write!(self.stream, "{end}").unwrap();
        }
    }

    fn visit_duration_literal(&mut self, expression: &DurationLiteral) {
        write!(self.stream, "{}", expression).unwrap();
    }

    fn visit_unary(&mut self, expression: &Unary) {
        write!(self.stream, "{}", expression.op).unwrap();
        let op = OP::UnaryOp(&expression.op);
        if matches!(
            *expression.operand,
            Expression::Unary(_) | Expression::Binary(_)
        ) && self.binding_power[&op].left < self.binding_power[&op].right
        {
            write!(self.stream, "(").unwrap();
            self.visit_expression(&expression.operand);
            write!(self.stream, ")").unwrap();
        } else {
            self.visit_expression(&expression.operand);
        }
    }

    fn visit_binary(&mut self, expression: &Binary) {
        let op = OP::BinaryOp(&expression.op);
        if matches!(
            *expression.left,
            Expression::Unary(_) | Expression::Binary(_)
        ) && self.binding_power[&op].left < self.binding_power[&op].right
        {
            write!(self.stream, "(").unwrap();
            self.visit_expression(&expression.left);
            write!(self.stream, ")").unwrap();
        } else {
            self.visit_expression(&expression.left);
        }
        write!(self.stream, "{}", expression.op).unwrap();
        if matches!(
            *expression.right,
            Expression::Unary(_) | Expression::Binary(_)
        ) && self.binding_power[&op].left < self.binding_power[&op].right
        {
            write!(self.stream, "(").unwrap();
            self.visit_expression(&expression.right);
            write!(self.stream, ")").unwrap();
        } else {
            self.visit_expression(&expression.right);
        }
    }

    fn visit_cast(&mut self, expression: &Cast) {
        self.visit_classical_type(&expression.type_);
        write!(self.stream, "(").unwrap();
        self.visit_expression(&expression.operand);
        write!(self.stream, ")").unwrap();
    }

    fn visit_index(&mut self, expression: &Index) {
        if matches!(
            *expression.target,
            Expression::Unary(_) | Expression::Binary(_)
        ) {
            write!(self.stream, "(").unwrap();
            self.visit_expression(&expression.target);
            write!(self.stream, ")").unwrap();
        } else {
            self.visit_expression(&expression.target);
        }
        write!(self.stream, "[").unwrap();
        self.visit_expression(&expression.index);
        write!(self.stream, "]").unwrap();
    }

    fn visit_index_set(&mut self, node: &IndexSet) {
        self.visit_expression_sequence(&node.values, "{", "}", ", ");
    }

    fn visit_program_block(&mut self, node: &ProgramBlock) {
        writeln!(self.stream, "{{").unwrap();
        self.current_indent += 1;
        for statement in &node.statements {
            self.visit_statement(statement);
        }
        self.current_indent -= 1;
        self.start_line();
        write!(self.stream, "}}").unwrap();
    }

    fn visit_quantum_block(&mut self, node: &QuantumBlock) {
        writeln!(self.stream, "{{").unwrap();
        self.current_indent += 1;
        for statement in &node.statements {
            self.visit_statement(statement);
        }
        self.current_indent -= 1;
        self.start_line();
        write!(self.stream, "}}").unwrap();
    }

    fn visit_quantum_measurement(&mut self, node: &QuantumMeasurement) {
        write!(self.stream, "measure ").unwrap();
        let identifier_vec: Vec<Expression> = node
            .identifier_list
            .iter()
            .cloned()
            .map(Expression::IdentifierOrSubscripted)
            .collect();
        let identifier_list = &identifier_vec;
        self.visit_expression_sequence(identifier_list, "", "", ", ");
    }

    fn visit_expression_sequence(
        &mut self,
        nodes: &[Expression],
        start: &str,
        end: &str,
        separator: &str,
    ) {
        if !start.is_empty() {
            write!(self.stream, "{start}").unwrap();
        }
        for node in nodes.iter().take(nodes.len() - 1) {
            self.visit_expression(node);
            write!(self.stream, "{separator}").unwrap();
        }
        if let Some(last) = nodes.last() {
            self.visit_expression(last);
        }
        if !end.is_empty() {
            write!(self.stream, "{end}").unwrap();
        }
    }

    fn visit_quantum_gate_modifier(&mut self, statement: &QuantumGateModifier) {
        write!(self.stream, "{}", self.modifier_lookup[&statement.modifier]).unwrap();
        if let Some(argument) = &statement.argument {
            write!(self.stream, "(").unwrap();
            self.visit_expression(argument);
            write!(self.stream, ")").unwrap();
        }
    }

    fn visit_quantum_gate_signature(&mut self, node: &QuantumGateSignature) {
        self.visit_identifier(&node.name);
        if let Some(params) = &node.params
            && !params.is_empty()
        {
            self.visit_expression_sequence(params, "(", ")", ", ");
        }
        write!(self.stream, " ").unwrap();
        let qarg_list: Vec<Expression> = node
            .qarg_list
            .iter()
            .map(|qarg| {
                Expression::IdentifierOrSubscripted(IdentifierOrSubscripted::Identifier(
                    qarg.to_owned(),
                ))
            })
            .collect();
        self.visit_expression_sequence(&qarg_list, "", "", ", ");
    }

    fn visit_classical_type(&mut self, node: &ClassicalType) {
        match node {
            ClassicalType::Float(type_) => self.visit_float_type(type_),
            ClassicalType::Bool => self.visit_bool_type(),
            ClassicalType::Int(type_) => self.visit_int_type(type_),
            ClassicalType::Uint(type_) => self.visit_uint_type(type_),
            ClassicalType::Bit => self.visit_bit_type(),
            ClassicalType::BitArray(type_) => self.visit_bit_array_type(type_),
        }
    }

    fn visit_float_type(&mut self, type_: &Float) {
        write!(self.stream, "float[{}]", self.float_width_lookup[type_]).unwrap()
    }

    fn visit_bool_type(&mut self) {
        write!(self.stream, "bool").unwrap()
    }

    fn visit_int_type(&mut self, type_: &Int) {
        write!(self.stream, "int").unwrap();
        if let Some(size) = type_.size {
            write!(self.stream, "[{size}]").unwrap();
        }
    }

    fn visit_uint_type(&mut self, type_: &Uint) {
        write!(self.stream, "uint").unwrap();
        if let Some(size) = type_.size {
            write!(self.stream, "[{size}]").unwrap();
        }
    }

    fn visit_bit_type(&mut self) {
        write!(self.stream, "bit").unwrap()
    }

    fn visit_bit_array_type(&mut self, type_: &BitArray) {
        write!(self.stream, "bit[{}]", type_.0).unwrap()
    }

    fn visit_statement(&mut self, statement: &Statement) {
        match statement {
            Statement::QuantumDeclaration(statement) => self.visit_quantum_declaration(statement),
            Statement::ClassicalDeclaration(statement) => {
                self.visit_classical_declaration(statement)
            }
            Statement::IODeclaration(_iodeclaration) => todo!(),
            Statement::QuantumInstruction(statement) => self.visit_quantum_instruction(statement),
            Statement::QuantumMeasurementAssignment(statement) => {
                self.visit_quantum_measurement_assignment(statement)
            }
            Statement::Assignment(statement) => self.visit_assignment_statement(statement),
            Statement::QuantumGateDefinition(statement) => {
                self.visit_quantum_gate_definition(statement)
            }
            Statement::Alias(statement) => self.visit_alias_statement(statement),
            Statement::Break(_) => self.visit_break_statement(),
            Statement::Continue(_) => self.visit_continue_statement(),
        }
    }

    fn visit_quantum_declaration(&mut self, statement: &QuantumDeclaration) {
        self.start_line();
        write!(self.stream, "qubit").unwrap();
        if let Some(designator) = &statement.designator {
            write!(self.stream, "[").unwrap();
            self.visit_expression(&designator.expression);
            write!(self.stream, "]").unwrap();
        }
        write!(self.stream, " ").unwrap();
        self.visit_identifier(&statement.identifier);
        self.end_statement();
    }

    fn visit_classical_declaration(&mut self, statement: &ClassicalDeclaration) {
        self.start_line();
        self.visit_classical_type(&statement.type_);
        write!(self.stream, " ").unwrap();
        self.visit_identifier(&statement.identifier);
        self.end_statement();
    }

    fn visit_quantum_instruction(&mut self, instruction: &QuantumInstruction) {
        match instruction {
            QuantumInstruction::GateCall(instruction) => self.visit_quantum_gate_call(instruction),
            QuantumInstruction::Reset(instruction) => self.visit_quantum_reset(instruction),
            QuantumInstruction::Barrier(instruction) => self.visit_quantum_barrier(instruction),
            QuantumInstruction::Delay(instruction) => self.visit_quantum_delay(instruction),
        }
    }

    fn visit_quantum_gate_call(&mut self, instruction: &GateCall) {
        self.start_line();
        if let Some(modifiers) = &instruction.modifiers {
            self.visit_modifier_sequence(modifiers, "", " @ ", " @ ");
        }
        self.visit_identifier(&instruction.quantum_gate_name);
        if !instruction.parameters.is_empty() {
            self.visit_expression_sequence(&instruction.parameters, "(", ")", ", ");
        }
        write!(self.stream, " ").unwrap();
        let index_identifier_list: Vec<Expression> = instruction
            .index_identifier_list
            .iter()
            .cloned()
            .map(Expression::IdentifierOrSubscripted)
            .collect();
        self.visit_expression_sequence(&index_identifier_list, "", "", ", ");
        self.end_statement();
    }

    fn visit_quantum_reset(&mut self, instruction: &Reset) {
        self.start_line();
        write!(self.stream, "reset ").unwrap();
        match &instruction.identifier {
            IdentifierOrSubscripted::Identifier(id) => self.visit_identifier(id),
            IdentifierOrSubscripted::Subscripted(sub_id) => self.visit_subscript_identifier(sub_id),
        }
        self.end_statement();
    }

    fn visit_quantum_barrier(&mut self, instruction: &Barrier) {
        self.start_line();
        write!(self.stream, "barrier ").unwrap();
        let index_identifier_vec: Vec<Expression> = instruction
            .index_identifier_list
            .iter()
            .cloned()
            .map(Expression::IdentifierOrSubscripted)
            .collect();
        let index_identifier_list: &[Expression] = &index_identifier_vec;
        self.visit_expression_sequence(index_identifier_list, "", "", ", ");
        self.end_statement();
    }

    fn visit_quantum_delay(&mut self, instruction: &Delay) {
        self.start_line();
        write!(self.stream, "delay[").unwrap();
        self.visit_duration_literal(&instruction.duration);
        write!(self.stream, "] ").unwrap();
        for qubit in &instruction.qubits {
            match qubit {
                IdentifierOrSubscripted::Identifier(id) => {
                    self.visit_identifier(id);
                }
                IdentifierOrSubscripted::Subscripted(sub_id) => {
                    self.visit_subscript_identifier(sub_id);
                }
            }
        }
        self.end_statement();
    }

    fn visit_quantum_measurement_assignment(&mut self, node: &QuantumMeasurementAssignment) {
        self.start_line();
        match &node.identifier {
            IdentifierOrSubscripted::Identifier(id) => self.visit_identifier(id),
            IdentifierOrSubscripted::Subscripted(sub_id) => self.visit_subscript_identifier(sub_id),
        }
        write!(self.stream, " = ").unwrap();
        self.visit_quantum_measurement(&node.quantum_measurement);
        self.end_statement();
    }

    fn visit_assignment_statement(&mut self, statement: &Assignment) {
        self.start_line();
        self.visit_identifier(&statement.lvalue);
        write!(self.stream, " = ").unwrap();
        self.visit_identifier_for_switch(&statement.rvalue);
        self.end_statement();
    }

    fn visit_quantum_gate_definition(&mut self, statement: &QuantumGateDefinition) {
        self.start_line();
        write!(self.stream, "gate ").unwrap();
        self.visit_quantum_gate_signature(&statement.quantum_gate_signature);
        write!(self.stream, " ").unwrap();
        self.visit_quantum_block(&statement.quantum_block);
        self.end_line();
    }

    fn visit_alias_statement(&mut self, statement: &Alias) {
        self.start_line();
        write!(self.stream, "let ").unwrap();
        self.visit_identifier(&statement.identifier);
        write!(self.stream, " = ").unwrap();
        self.visit_expression(&statement.value);
        self.end_statement();
    }

    fn visit_break_statement(&mut self) {
        self.write_statement("break");
    }

    fn visit_continue_statement(&mut self) {
        self.write_statement("continue");
    }

    fn visit_identifier_for_switch(&mut self, identifiers: &Vec<Identifier>) {
        if identifiers.len() > 1 {
            write!(self.stream, "[").unwrap();
        }
        for identifier in identifiers {
            self.visit_identifier(identifier);
            write!(self.stream, ",").unwrap();
        }
        if identifiers.len() > 1 {
            write!(self.stream, "]").unwrap();
        }
    }
}
