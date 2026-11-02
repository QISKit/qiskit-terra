// This code is part of Qiskit.
//
// (C) Copyright IBM 2026
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#![allow(dead_code)] // These definitions are used when QPY 19 support is enabled.

use crate::bytes::Bytes;
use crate::value::{
    Complex64MatrixPack, Complex64Pack, StringU16Pack, pack_biguint, unpack_biguint,
};
use binrw::binrw;
use num_bigint::BigUint;
use num_complex::Complex64;
use qiskit_circuit::operations::DelayUnit;

use crate::formats::{
    DurationPack, InstructionsAnnotationPack, ModifierPack, PackedExpression,
    ParameterExpressionPack, ParameterSymbolPack, ParameterVectorElementPack, PauliDataPack,
    QPYCircuit, RegisterPack,
};

#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import (version: u8))]
pub struct CircuitHeaderV19Pack {
    // global circuit data
    #[brw(args(version))]
    pub global_phase: GlobalPhasePack,
    pub num_qubits: u32,
    pub num_clbits: u32,
    pub num_instructions: u64,
    pub num_vars: u32,

    // register data
    #[bw(calc = registers.len() as u32)]
    pub num_registers: u32,
    #[br(count = num_registers, args { inner: (version,) })]
    pub registers: Vec<RegisterPack>,

    // interner data
    #[bw(calc = qubit_interner.len() as u32)]
    pub qubit_interner_size: u32,
    #[bw(calc = clbit_interner.len() as u32)]
    pub clbit_interner_size: u32,
    #[br(count = qubit_interner_size as usize)]
    pub qubit_interner: Vec<InternerEntry>,
    #[br(count = clbit_interner_size as usize)]
    pub clbit_interner: Vec<InternerEntry>,

    pub circuit_name: StringU16Pack,
    // byte-encoded metadata from an external source
    #[bw(calc = metadata.len() as u64)]
    pub metadata_size: u64,
    #[br(count = metadata_size)]
    pub metadata: Bytes,
}

// The data for a specific instruction in the circuit, for QPY version 19 and higher
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub struct CircuitInstructionV19Pack {
    pub operation: CircuitOperationType,
    // Interner index
    pub qargs: u32,
    // Interner index
    pub cargs: u32,

    #[br(args(operation, version))]
    #[bw(args(version))]
    pub operation_data: OperationData,

    // Get param size from OperationData during decoding (it's either static from rust definition
    // or dynamic in the body)
    #[bw(calc = params.len() as u16)]
    pub num_parameters: u16,
    #[br(count = num_parameters as usize, args { inner: (version,) })]
    #[bw(args(version))]
    pub params: Vec<ParamDataPack>,

    // Whether the following optional fields are present.
    #[bw(calc =
        if annotations.is_some() { extra_fields_flag_parts::ANNOTATIONS } else { 0 }
        | if label.is_some() { extra_fields_flag_parts::LABEL } else { 0 }
    )]
    pub extra_fields_flag: u8,
    #[br(if(has_v19_annotations(extra_fields_flag)))]
    pub annotations: Option<InstructionsAnnotationPack>,
    #[br(if(has_label(extra_fields_flag)))]
    pub label: Option<StringU16Pack>,
}

/// Bit masks for optional fields in a QPY 19 circuit instruction.
pub mod extra_fields_flag_parts {
    pub const ANNOTATIONS: u8 = 0b1000_0000;
    pub const LABEL: u8 = 0b0100_0000;
}

fn has_v19_annotations(extra_fields_flag: u8) -> bool {
    extra_fields_flag & extra_fields_flag_parts::ANNOTATIONS != 0
}

fn has_label(extra_fields_flag: u8) -> bool {
    extra_fields_flag & extra_fields_flag_parts::LABEL != 0
}

#[binrw]
#[brw(big)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[brw(repr = u8)]
pub enum CircuitOperationType {
    StandardGate = 0,
    StandardInstruction = 1,
    Custom = 2,
    FromPython = 3,
    UnitaryGate = 4,
    Controlled = 5,
    ControlFlow = 6,
    PauliEvolution = 7,
    PauliProductMeasurement = 8,
    PauliProductRotation = 9,
    Store = 10,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
#[br(import(op_type: CircuitOperationType, version: u8))]
#[bw(import(version: u8))]
pub enum OperationData {
    // The value of the gate from qiskit_circuit::standard_gate::StandardGate
    #[br(pre_assert(op_type == CircuitOperationType::StandardGate))]
    StandardGate(u8),
    // The value of the instruction from qiskit_circuit::operations::StandardInstruction
    #[br(pre_assert(op_type == CircuitOperationType::StandardInstruction))]
    StandardInstruction(StandardInstructionData),
    // Index into custom gate table
    #[br(pre_assert(op_type == CircuitOperationType::Custom))]
    Custom(u64),
    // Store gate class name like is done now for Python defined operations in Qiskit
    #[br(pre_assert(op_type == CircuitOperationType::FromPython))]
    FromPython(#[brw(args(version))] FromPythonPack),
    // Store the raw npy bytes of the underlying array
    #[br(pre_assert(op_type == CircuitOperationType::UnitaryGate))]
    UnitaryGate(Complex64MatrixPack),
    // Store the base gate and then the extra control metadata
    #[br(pre_assert(op_type == CircuitOperationType::Controlled))]
    Controlled(#[brw(args(version))] ControlledGatePack),
    // Store the circuit bodies and the condition explicitly in the pack
    #[br(pre_assert(op_type == CircuitOperationType::ControlFlow))]
    ControlFlow(ControlFlowPack),
    // Store Pauli operators and synthesis settings directly; evolution time remains a parameter.
    #[br(pre_assert(op_type == CircuitOperationType::PauliEvolution))]
    PauliEvolution(#[brw(args(version))] PauliEvolutionGatePack),
    #[br(pre_assert(op_type == CircuitOperationType::PauliProductMeasurement))]
    PauliProductMeasurement(PauliProductMeasurementPack),
    #[br(pre_assert(op_type == CircuitOperationType::PauliProductRotation))]
    PauliProductRotation(PauliProductRotationPack),
    #[br(pre_assert(op_type == CircuitOperationType::Store))]
    Store(PackedExpression, PackedExpression),
}

/// A bit-packed boolean vector.  Bits are stored least-significant-bit first in each byte.
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct BoolVectorPack {
    pub num_bits: u32,
    #[br(count = (num_bits as usize).div_ceil(8))]
    pub data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct PauliProductMeasurementPack {
    pub z: BoolVectorPack,
    pub x: BoolVectorPack,
    pub neg: u8,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct PauliProductRotationPack {
    pub z: BoolVectorPack,
    pub x: BoolVectorPack,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct StandardInstructionData {
    /// The value of `qiskit_circuit::operations::StandardInstructionType`.
    pub discriminant: u8,
    #[br(if(discriminant == 1))]
    #[bw(if(*discriminant == 1))]
    pub delay_unit: Option<DelayUnitPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DelayUnitPack {
    #[brw(magic = b'n')]
    NS,
    #[brw(magic = b'p')]
    PS,
    #[brw(magic = b'u')]
    US,
    #[brw(magic = b'm')]
    MS,
    #[brw(magic = b's')]
    S,
    #[brw(magic = b't')]
    DT,
    #[brw(magic = b'e')]
    Expr,
}

impl From<DelayUnit> for DelayUnitPack {
    fn from(value: DelayUnit) -> Self {
        match value {
            DelayUnit::NS => Self::NS,
            DelayUnit::PS => Self::PS,
            DelayUnit::US => Self::US,
            DelayUnit::MS => Self::MS,
            DelayUnit::S => Self::S,
            DelayUnit::DT => Self::DT,
            DelayUnit::EXPR => Self::Expr,
        }
    }
}

impl From<DelayUnitPack> for DelayUnit {
    fn from(value: DelayUnitPack) -> Self {
        match value {
            DelayUnitPack::NS => Self::NS,
            DelayUnitPack::PS => Self::PS,
            DelayUnitPack::US => Self::US,
            DelayUnitPack::MS => Self::MS,
            DelayUnitPack::S => Self::S,
            DelayUnitPack::DT => Self::DT,
            DelayUnitPack::Expr => Self::EXPR,
        }
    }
}

// this is a non generic version of the `GenericDataPack`
// avoiding length storage where it is not required and the need to pre-serialize the data
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub enum ParamDataPack {
    #[brw(magic = b'b')]
    Bool(u8), // TODO: make this an actual boolean

    #[brw(magic = b'i')]
    Int64(i64),

    #[brw(magic = b'I')]
    BigInt(
        #[br(map = unpack_biguint)]
        #[bw(map = pack_biguint)]
        BigUint,
    ),

    #[brw(magic = b'f')]
    Float64(f64),

    #[brw(magic = b'c')]
    Complex64(
        #[br(map = |value: Complex64Pack| Complex64::new(value.re, value.im))]
        #[bw(map = |value| Complex64Pack { re: value.re, im: value.im })]
        Complex64,
    ),

    #[brw(magic = b'D')]
    CaseDefault,

    #[brw(magic = b'r')]
    Range { start: i64, stop: i64, step: i64 },

    #[brw(magic = b'n')]
    NumpyObject {
        #[bw(calc = data.len() as u64)]
        data_length: u64,
        #[br(count = data_length)]
        data: Bytes,
    }, // this should be avoided if possible

    #[brw(magic = b'T')]
    Tuple {
        #[bw(calc = elements.len() as u64)]
        num_elements: u64,
        #[br(count = num_elements, args { inner: (version,) })]
        #[bw(args(version))]
        elements: Vec<ParamDataPack>,
    },

    #[brw(magic = b'p')]
    Parameter(ParameterSymbolPack),

    #[brw(magic = b'v')]
    ParameterVectorElement(#[brw(args(version))] ParameterVectorElementPack),

    #[brw(magic = b'e')]
    ParameterExpression(#[brw(args(version))] ParameterExpressionPack),

    #[brw(magic = b's')]
    String(StringU16Pack),

    #[brw(magic = b'z')]
    Null,

    #[brw(magic = b'x')]
    Expression(ExpressionPack),

    #[brw(magic = b'm')]
    Modifier(ModifierPack),

    #[brw(magic = b'd')]
    Duration(DurationPack),

    #[brw(magic = b'R')]
    Register(ParamDataRegisterPack),

    #[brw(magic = b'q')]
    Circuit(#[brw(args(version))] Box<QPYCircuit>),
}

/// A register-valued instruction parameter in QPY 19 and newer.
///
/// Unlike [`ParamRegisterPack`], this is self-delimiting because a [`ParamDataPack`] does not
/// carry a payload length.  Register names therefore use [`StringU16Pack`], while clbits retain
/// the compact circuit-local index used by the legacy representation.
#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ParamDataRegisterPack {
    #[brw(magic = 1u8)]
    Register(StringU16Pack),

    #[brw(magic = 0u8)]
    Clbit(u32),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub struct FromPythonPack {
    pub class_name: StringU16Pack,
    pub op_name: StringU16Pack,
    #[bw(calc = init_params.len() as u16)]
    pub num_init_params: u16,
    #[br(count = num_init_params, args { inner: (version,) })]
    #[bw(args(version))]
    pub init_params: Vec<NamedParamDataPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub struct NamedParamDataPack {
    pub name: StringU16Pack,
    #[brw(args(version))]
    pub value: ParamDataPack,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub struct ControlledGatePack {
    #[brw(args(version))]
    pub from_python: FromPythonPack,
    pub num_ctrl_qubits: u32,
    pub ctrl_state: u32,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ControlFlowPack {
    #[brw(magic = b'b')]
    Box(BoxDurationPack),
    #[brw(magic = b'k')]
    BreakLoop,
    #[brw(magic = b'c')]
    ContinueLoop,
    #[brw(magic = b'f')]
    ForLoop(ForCollectionPack, LoopParamPack),
    #[brw(magic = b'i')]
    IfElse(ConditionV19Pack),
    #[brw(magic = b's')]
    Switch(SwitchTargetPack, CaseSpecPack),
    #[brw(magic = b'w')]
    While(ConditionV19Pack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ConditionV19Pack {
    #[brw(magic = b'b')]
    Bit(u32, u8),
    #[brw(magic = b'r')]
    Register(
        StringU16Pack,
        #[br(map = unpack_biguint)]
        #[bw(map = pack_biguint)]
        BigUint,
    ),
    #[brw(magic = b'e')]
    Expression(ExpressionPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum BoxDurationPack {
    #[brw(magic = b'n')]
    None,
    #[brw(magic = b'd')]
    Duration(DurationPack),
    #[brw(magic = b'e')]
    Expression(ExpressionPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum ForCollectionPack {
    #[brw(magic = b'l')]
    List {
        #[bw(calc = values.len() as u64)]
        size: u64,
        #[br(count = size)]
        values: Vec<i64>,
    },
    #[brw(magic = b'r')]
    Range(i64, i64, i64),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum LoopParamPack {
    #[brw(magic = b'n')]
    None,
    #[brw(magic = b'p')]
    Parameter(ParameterSymbolPack),
    #[brw(magic = b'v')]
    Variable,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum SwitchTargetPack {
    #[brw(magic = b'b')]
    Bit(u32),
    #[brw(magic = b'r')]
    Register(StringU16Pack),
    #[brw(magic = b'e')]
    Expression(ExpressionPack),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CaseSpecPack {
    #[bw(calc = labels.len() as u32)]
    num_cases: u32,
    #[br(count = num_cases)]
    pub labels: Vec<CaseLabelsPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct CaseLabelsPack {
    #[bw(calc = labels.len() as u32)]
    num_labels: u32,
    #[br(count = num_labels)]
    pub labels: Vec<CaseSpecifierPack>,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum CaseSpecifierPack {
    #[brw(magic = b'd')]
    Default,
    #[brw(magic = b'i')]
    Uint(
        #[br(map = unpack_biguint)]
        #[bw(map = pack_biguint)]
        BigUint,
    ),
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub enum InternerEntry {
    /// An interner slot retained only to preserve the indices of later, live entries.
    #[brw(magic = b'u')]
    Unused,
    #[brw(magic = b's')]
    Single(u32),
    #[brw(magic = b'd')]
    Double(u32, u32),
    #[brw(magic = b't')]
    Triple(u32, u32, u32),
    #[brw(magic = b'v')]
    VariableSize {
        #[bw(try_calc = u16::try_from(bits.len()))]
        num_bits: u16,
        #[br(count = num_bits)]
        bits: Vec<u32>,
    },
    #[brw(magic = b'a')]
    All, // for gates that operate on all qubits at once, e.g. barriers can do this
}

// The global phase is either a float or a parameter
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub enum GlobalPhasePack {
    #[brw(magic = b'f')]
    Float(f64),

    #[brw(magic = b'p')]
    Parameter(ParameterSymbolPack),

    #[brw(magic = b'v')]
    ParameterVectorElement(#[brw(args(version))] ParameterVectorElementPack),

    #[brw(magic = b'e')]
    ParameterExpression(#[brw(args(version))] ParameterExpressionPack),
}

/// QPY 19 operation payload for a Pauli-evolution gate.  Unlike the legacy custom-operation
/// definition, the evolution time is stored in the instruction's ordinary parameter list.
#[binrw]
#[brw(big)]
#[derive(Debug)]
#[brw(import(version: u8))]
pub struct PauliEvolutionGatePack {
    #[bw(calc = pauli_data.len() as u64)]
    pub operator_size: u64,
    pub standalone_op: u8,
    #[bw(calc = synth_data.len() as u64)]
    pub synth_method_size: u64,
    #[br(count = operator_size, args { inner: (version,) })]
    #[bw(args(version))]
    pub pauli_data: Vec<PauliDataPack>,
    #[br(count = synth_method_size)]
    pub synth_data: Bytes,
}

#[binrw]
#[brw(big)]
#[derive(Debug)]
pub struct ExpressionPack {
    pub expression: PackedExpression,
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::{BinWrite, Endian};
    use std::io::Cursor;

    const VERSION: u8 = 19;

    fn encoded<T>(value: &T, args: T::Args<'_>) -> Vec<u8>
    where
        T: BinWrite,
    {
        let mut result = Cursor::new(Vec::new());
        let write_result = value.write_options(&mut result, Endian::Big, args);
        assert!(
            write_result.is_ok(),
            "test value should serialize: {write_result:?}"
        );
        result.into_inner()
    }

    fn string(value: &str) -> StringU16Pack {
        StringU16Pack {
            value: value.to_owned(),
        }
    }

    fn symbol(name: &str, byte: u8) -> ParameterSymbolPack {
        ParameterSymbolPack {
            uuid: [byte; 16],
            name: name.to_owned(),
        }
    }

    #[test]
    fn circuit_header_layout() {
        let header = CircuitHeaderV19Pack {
            global_phase: GlobalPhasePack::Float(1.5),
            num_qubits: 45,
            num_clbits: 3,
            num_instructions: 2,
            num_vars: 1,
            registers: vec![],
            qubit_interner: vec![InternerEntry::Single(7)],
            clbit_interner: vec![InternerEntry::Double(8, 9)],
            circuit_name: string("foo"),
            metadata: Bytes::from(&b"{}"[..]),
        };
        let mut expected = vec![b'f'];
        expected.extend_from_slice(&1.5f64.to_be_bytes());
        expected.extend_from_slice(&45u32.to_be_bytes());
        expected.extend_from_slice(&3u32.to_be_bytes());
        expected.extend_from_slice(&2u64.to_be_bytes());
        expected.extend_from_slice(&1u32.to_be_bytes());
        expected.extend_from_slice(&0u32.to_be_bytes()); // num_registers
        expected.extend_from_slice(&1u32.to_be_bytes()); // qubit_interner_size
        expected.extend_from_slice(&1u32.to_be_bytes()); // clbit_interner_size
        expected.extend_from_slice(b"s\0\0\0\x07d\0\0\0\x08\0\0\0\x09");
        expected.extend_from_slice(b"\0\x03foo");
        expected.extend_from_slice(&2u64.to_be_bytes()); // metadata_size
        expected.extend_from_slice(b"{}");
        assert_eq!(encoded(&header, (VERSION,)), expected);
    }

    #[test]
    fn circuit_instruction_layout_and_extra_field_flags() {
        let instruction = CircuitInstructionV19Pack {
            operation: CircuitOperationType::StandardGate,
            qargs: 0x0102_0304,
            cargs: 5,
            operation_data: OperationData::StandardGate(17),
            params: vec![ParamDataPack::Bool(1), ParamDataPack::Int64(-2)],
            annotations: None,
            label: Some(string("x")),
        };
        let mut expected = vec![0, 1, 2, 3, 4, 0, 0, 0, 5, 17, 0, 2, b'b', 1, b'i'];
        expected.extend_from_slice(&(-2i64).to_be_bytes());
        expected.extend_from_slice(&[extra_fields_flag_parts::LABEL, 0, 1, b'x']);
        assert_eq!(encoded(&instruction, (VERSION,)), expected);
    }

    #[test]
    fn boolean_and_pauli_product_layouts() {
        let bits = BoolVectorPack {
            num_bits: 9,
            data: Bytes::from(&[0x81, 0x01][..]),
        };
        assert_eq!(encoded(&bits, ()), vec![0, 0, 0, 9, 0x81, 0x01]);

        let measurement = PauliProductMeasurementPack {
            z: BoolVectorPack {
                num_bits: 1,
                data: vec![1].into(),
            },
            x: BoolVectorPack {
                num_bits: 1,
                data: vec![0].into(),
            },
            neg: 1,
        };
        assert_eq!(
            encoded(&measurement, ()),
            vec![0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1]
        );

        let rotation = PauliProductRotationPack {
            z: BoolVectorPack {
                num_bits: 2,
                data: vec![2].into(),
            },
            x: BoolVectorPack {
                num_bits: 2,
                data: vec![1].into(),
            },
        };
        assert_eq!(encoded(&rotation, ()), vec![0, 0, 0, 2, 2, 0, 0, 0, 2, 1]);
    }

    #[test]
    fn standard_instruction_and_delay_unit_layouts() {
        let delay = StandardInstructionData {
            discriminant: 1,
            delay_unit: Some(DelayUnitPack::NS),
        };
        assert_eq!(encoded(&delay, ()), b"\x01n");
        let ordinary = StandardInstructionData {
            discriminant: 7,
            delay_unit: None,
        };
        assert_eq!(encoded(&ordinary, ()), b"\x07");
        assert_eq!(encoded(&DelayUnitPack::Expr, ()), b"e");
    }

    #[test]
    fn parameter_data_scalar_and_container_layouts() {
        assert_eq!(
            encoded(&ParamDataPack::Float64(1.5), (VERSION,)),
            [vec![b'f'], 1.5f64.to_be_bytes().to_vec()].concat()
        );
        assert_eq!(
            encoded(
                &ParamDataPack::Complex64(Complex64::new(1.0, -2.0)),
                (VERSION,)
            ),
            [
                vec![b'c'],
                1.0f64.to_be_bytes().to_vec(),
                (-2.0f64).to_be_bytes().to_vec()
            ]
            .concat()
        );
        assert_eq!(
            encoded(
                &ParamDataPack::Range {
                    start: -1,
                    stop: 4,
                    step: 2
                },
                (VERSION,)
            ),
            [
                vec![b'r'],
                (-1i64).to_be_bytes().to_vec(),
                4i64.to_be_bytes().to_vec(),
                2i64.to_be_bytes().to_vec()
            ]
            .concat()
        );
        assert_eq!(
            encoded(
                &ParamDataPack::NumpyObject {
                    data: vec![3, 4].into()
                },
                (VERSION,)
            ),
            vec![b'n', 0, 0, 0, 0, 0, 0, 0, 2, 3, 4]
        );
        assert_eq!(
            encoded(
                &ParamDataPack::Tuple {
                    elements: vec![ParamDataPack::Null, ParamDataPack::CaseDefault]
                },
                (VERSION,)
            ),
            vec![b'T', 0, 0, 0, 0, 0, 0, 0, 2, b'z', b'D']
        );
        assert_eq!(
            encoded(&ParamDataPack::String(string("hi")), (VERSION,)),
            b"s\0\x02hi"
        );
    }

    #[test]
    fn parameter_register_layouts() {
        assert_eq!(
            encoded(&ParamDataRegisterPack::Register(string("cr")), ()),
            b"\x01\0\x02cr"
        );
        assert_eq!(
            encoded(&ParamDataRegisterPack::Clbit(0x0102_0304), ()),
            b"\0\x01\x02\x03\x04"
        );
    }

    #[test]
    fn python_operation_struct_layouts() {
        let named = NamedParamDataPack {
            name: string("p"),
            value: ParamDataPack::Bool(1),
        };
        assert_eq!(encoded(&named, (VERSION,)), b"\0\x01pb\x01");

        let from_python = FromPythonPack {
            class_name: string("C"),
            op_name: string("op"),
            init_params: vec![NamedParamDataPack {
                name: string("p"),
                value: ParamDataPack::Null,
            }],
        };
        assert_eq!(
            encoded(&from_python, (VERSION,)),
            b"\0\x01C\0\x02op\0\x01\0\x01pz"
        );

        let controlled = ControlledGatePack {
            from_python: FromPythonPack {
                class_name: string("C"),
                op_name: string("x"),
                init_params: vec![],
            },
            num_ctrl_qubits: 2,
            ctrl_state: 1,
        };
        assert_eq!(
            encoded(&controlled, (VERSION,)),
            b"\0\x01C\0\x01x\0\0\0\0\0\x02\0\0\0\x01"
        );
    }

    #[test]
    fn control_flow_component_layouts() {
        assert_eq!(
            encoded(&ConditionV19Pack::Bit(3, 1), ()),
            b"b\0\0\0\x03\x01"
        );
        assert_eq!(encoded(&BoxDurationPack::None, ()), b"n");
        assert_eq!(
            encoded(
                &ForCollectionPack::List {
                    values: vec![-1, 2]
                },
                ()
            ),
            [
                vec![b'l'],
                2u64.to_be_bytes().to_vec(),
                (-1i64).to_be_bytes().to_vec(),
                2i64.to_be_bytes().to_vec()
            ]
            .concat()
        );
        assert_eq!(
            encoded(&LoopParamPack::Parameter(symbol("p", 0xaa)), ()),
            [vec![b'p', 0, 1], vec![0xaa; 16], vec![b'p']].concat()
        );
        assert_eq!(
            encoded(&SwitchTargetPack::Register(string("c")), ()),
            b"r\0\x01c"
        );
    }

    #[test]
    fn case_struct_layouts() {
        let labels = CaseLabelsPack {
            labels: vec![
                CaseSpecifierPack::Default,
                CaseSpecifierPack::Uint(BigUint::from(258u16)),
            ],
        };
        assert_eq!(encoded(&labels, ()), b"\0\0\0\x02di\x02\x01\x02");
        let cases = CaseSpecPack {
            labels: vec![CaseLabelsPack {
                labels: vec![CaseSpecifierPack::Default],
            }],
        };
        assert_eq!(encoded(&cases, ()), b"\0\0\0\x01\0\0\0\x01d");
    }

    #[test]
    fn interner_entry_layouts() {
        assert_eq!(encoded(&InternerEntry::Unused, ()), b"u");
        assert_eq!(
            encoded(&InternerEntry::Triple(1, 2, 3), ()),
            b"t\0\0\0\x01\0\0\0\x02\0\0\0\x03"
        );
        assert_eq!(
            encoded(&InternerEntry::VariableSize { bits: vec![4, 5] }, ()),
            b"v\0\x02\0\0\0\x04\0\0\0\x05"
        );
        assert_eq!(encoded(&InternerEntry::All, ()), b"a");
    }

    #[test]
    fn global_phase_layouts() {
        assert_eq!(
            encoded(&GlobalPhasePack::Float(-0.5), (VERSION,)),
            [vec![b'f'], (-0.5f64).to_be_bytes().to_vec()].concat()
        );
        assert_eq!(
            encoded(&GlobalPhasePack::Parameter(symbol("a", 1)), (VERSION,)),
            [vec![b'p', 0, 1], vec![1; 16], vec![b'a']].concat()
        );
    }

    #[test]
    fn pauli_evolution_struct_layout() {
        let value = PauliEvolutionGatePack {
            standalone_op: 1,
            pauli_data: vec![],
            synth_data: vec![0xaa, 0xbb].into(),
        };
        assert_eq!(
            encoded(&value, (VERSION,)),
            b"\0\0\0\0\0\0\0\0\x01\0\0\0\0\0\0\0\x02\xaa\xbb"
        );
    }
}
