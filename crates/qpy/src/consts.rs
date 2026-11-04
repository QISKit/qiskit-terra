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

use qiskit_circuit::operations::StandardGate;

pub fn standard_gate_from_gate_class_name(name: &str) -> Option<StandardGate> {
    match name {
        "HGate" => Some(StandardGate::H),
        "IGate" => Some(StandardGate::I),
        "XGate" => Some(StandardGate::X),
        "YGate" => Some(StandardGate::Y),
        "ZGate" => Some(StandardGate::Z),
        "PhaseGate" => Some(StandardGate::Phase),
        "RGate" => Some(StandardGate::R),
        "RXGate" => Some(StandardGate::RX),
        "RYGate" => Some(StandardGate::RY),
        "RZGate" => Some(StandardGate::RZ),
        "SGate" => Some(StandardGate::S),
        "SdgGate" => Some(StandardGate::Sdg),
        "SXGate" => Some(StandardGate::SX),
        "SXdgGate" => Some(StandardGate::SXdg),
        "TGate" => Some(StandardGate::T),
        "TdgGate" => Some(StandardGate::Tdg),
        "UGate" => Some(StandardGate::U),
        "U1Gate" => Some(StandardGate::U1),
        "U2Gate" => Some(StandardGate::U2),
        "U3Gate" => Some(StandardGate::U3),
        "CHGate" => Some(StandardGate::CH),
        "CXGate" => Some(StandardGate::CX),
        "CYGate" => Some(StandardGate::CY),
        "CZGate" => Some(StandardGate::CZ),
        "DCXGate" => Some(StandardGate::DCX),
        "ECRGate" => Some(StandardGate::ECR),
        "SwapGate" => Some(StandardGate::Swap),
        "iSwapGate" => Some(StandardGate::ISwap),
        "CPhaseGate" => Some(StandardGate::CPhase),
        "CRXGate" => Some(StandardGate::CRX),
        "CRYGate" => Some(StandardGate::CRY),
        "CRZGate" => Some(StandardGate::CRZ),
        "CSGate" => Some(StandardGate::CS),
        "CSdgGate" => Some(StandardGate::CSdg),
        "CSXGate" => Some(StandardGate::CSX),
        "CUGate" => Some(StandardGate::CU),
        "CU1Gate" => Some(StandardGate::CU1),
        "CU3Gate" => Some(StandardGate::CU3),
        "RXXGate" => Some(StandardGate::RXX),
        "RYYGate" => Some(StandardGate::RYY),
        "RZZGate" => Some(StandardGate::RZZ),
        "RZXGate" => Some(StandardGate::RZX),
        "XXMinusYYGate" => Some(StandardGate::XXMinusYY),
        "XXPlusYYGate" => Some(StandardGate::XXPlusYY),
        "CCXGate" => Some(StandardGate::CCX),
        "CCZGate" => Some(StandardGate::CCZ),
        "CSwapGate" => Some(StandardGate::CSwap),
        "RCCXGate" => Some(StandardGate::RCCX),
        "C3XGate" => Some(StandardGate::C3X),
        "C3SXGate" => Some(StandardGate::C3SX),
        "RC3XGate" => Some(StandardGate::RC3X),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use qiskit_circuit::imports::get_std_gate_class_name;
    use qiskit_circuit::operations::STANDARD_GATE_SIZE;
    #[test]
    fn test_name_coverage() {
        for gate in 1..STANDARD_GATE_SIZE as u8 {
            let gate: StandardGate =
                ::bytemuck::checked::try_cast::<_, StandardGate>(gate).unwrap();
            let gate_name = get_std_gate_class_name(&gate).clone();
            assert!(
                standard_gate_from_gate_class_name(gate_name.as_str()).is_some(),
                "For gate name {gate_name} could not get StandardGate {gate:?}"
            );
        }
    }
}
