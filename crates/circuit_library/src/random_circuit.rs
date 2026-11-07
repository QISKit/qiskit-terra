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

#[cfg(feature = "cache_pygates")]
use std::sync::OnceLock;

use pyo3::prelude::*;

use rand::prelude::*;
use rand::rngs::SysRng;
use rand_distr::{StandardNormal, Uniform, multi::Dirichlet};
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::bit::{ClassicalRegister, QuantumRegister};
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError, PyCircuitData};
use qiskit_circuit::operations::{
    Condition, ControlFlow, ControlFlowInstruction, Operation, Param, StandardGate,
    StandardInstruction,
};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Clbit, Qubit};
use qiskit_util::IndexMap;

static ONE_Q_GATES: [StandardGate; 20] = [
    StandardGate::H,
    StandardGate::I,
    StandardGate::X,
    StandardGate::Y,
    StandardGate::Z,
    StandardGate::Phase,
    StandardGate::R,
    StandardGate::RX,
    StandardGate::RY,
    StandardGate::RZ,
    StandardGate::S,
    StandardGate::Sdg,
    StandardGate::SX,
    StandardGate::SXdg,
    StandardGate::T,
    StandardGate::Tdg,
    StandardGate::U,
    StandardGate::U1,
    StandardGate::U2,
    StandardGate::U3,
];

static TWO_Q_GATES: [StandardGate; 24] = [
    StandardGate::CH,
    StandardGate::CX,
    StandardGate::CY,
    StandardGate::CZ,
    StandardGate::DCX,
    StandardGate::ECR,
    StandardGate::Swap,
    StandardGate::ISwap,
    StandardGate::CPhase,
    StandardGate::CRX,
    StandardGate::CRY,
    StandardGate::CRZ,
    StandardGate::CS,
    StandardGate::CSdg,
    StandardGate::CSX,
    StandardGate::CU,
    StandardGate::CU1,
    StandardGate::CU3,
    StandardGate::RXX,
    StandardGate::RYY,
    StandardGate::RZZ,
    StandardGate::RZX,
    StandardGate::XXMinusYY,
    StandardGate::XXPlusYY,
];

static THREE_Q_GATES: [StandardGate; 4] = [
    StandardGate::CCX,
    StandardGate::CCZ,
    StandardGate::CSwap,
    StandardGate::RCCX,
];

static FOUR_Q_GATES: [StandardGate; 3] =
    [StandardGate::C3X, StandardGate::C3SX, StandardGate::RC3X];

static ALL_GATES: [&[StandardGate]; 4] =
    [&ONE_Q_GATES, &TWO_Q_GATES, &THREE_Q_GATES, &FOUR_Q_GATES];

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "random_circuit", signature=(num_qubits, depth, max_operands=4, measure=false, conditional=false, reset=false, seed=None, distribution=None))]
pub fn py_random_circuit(
    num_qubits: u32,
    depth: usize,
    max_operands: u8,
    measure: bool,
    conditional: bool,
    reset: bool,
    seed: Option<u64>,
    distribution: Option<IndexMap<u8, f64>>,
) -> PyResult<PyCircuitData> {
    Ok(random_circuit(
        num_qubits,
        depth,
        max_operands,
        measure,
        conditional,
        reset,
        seed,
        distribution,
    )?
    .into())
}

#[allow(clippy::too_many_arguments)]
pub fn random_circuit(
    num_qubits: u32,
    depth: usize,
    max_operands: u8,
    measure: bool,
    conditional: bool,
    reset: bool,
    seed: Option<u64>,
    distribution: Option<IndexMap<u8, f64>>,
) -> Result<CircuitData, CircuitDataError> {
    if num_qubits == 0 {
        return CircuitData::new(None, None, Param::Float(0.));
    }
    let mut rng = match seed {
        Some(seed) => Pcg64Mcg::seed_from_u64(seed),
        None => Pcg64Mcg::try_from_rng(&mut SysRng).unwrap(),
    };

    let operand_dist = match distribution {
        Some(dist) => dist,
        None => {
            let max_operands = if num_qubits > max_operands as u32 {
                max_operands
            } else {
                num_qubits as u8
            };
            let ones = vec![1.0; 2.max(max_operands as usize)];
            let distr = rng.sample(Dirichlet::new(&ones).unwrap());
            (0..max_operands)
                .map(|i| (i + 1, distr[i as usize]))
                .collect()
        }
    };

    let qreg = QuantumRegister::new_owning("q", num_qubits);
    let creg = if measure || conditional {
        Some(ClassicalRegister::new_owning("c", num_qubits))
    } else {
        None
    };
    let mut circuit = CircuitData::new(None, None, Param::Float(0.))?;
    circuit.add_qreg(qreg, true)?;
    if let Some(ref creg) = creg {
        circuit.add_creg(creg.clone(), true)?;
    }
    let mut distribution = Vec::with_capacity(51);
    let one_q_gates = if reset {
        let mut out = Vec::with_capacity(ONE_Q_GATES.len() + 1);
        out.extend_from_slice(&ONE_Q_GATES);
        // Use GlobalPhaseGate as a sentinel for reset
        out.push(StandardGate::GlobalPhase);
        out
    } else {
        Vec::new()
    };

    // Here we will create a list 'gates_to_consider' that will have a
    // subset of different n-qubit gates and will also create a list for
    //ratio (or probability) for each gates
    let gates_to_consider: Vec<(usize, StandardGate)> = operand_dist
        .iter()
        .flat_map(|(num_qubits, ratio)| {
            let gate_list = ALL_GATES[*num_qubits as usize - 1];
            if reset && *num_qubits == 1 {
                let val = ratio / (gate_list.len() + 1) as f64;
                for _ in 0..gate_list.len() + 1 {
                    distribution.push(val);
                }
                one_q_gates.as_slice()
            } else {
                let val = ratio / gate_list.len() as f64;
                for _ in 0..gate_list.len() {
                    distribution.push(val);
                }
                gate_list
            }
        })
        .copied()
        .enumerate()
        .collect();
    let mut counter: [usize; 4] = [0, 0, 0, 0];
    let mut total_gates: usize = 0;
    let mut culmulative_qubits: Vec<u32> = Vec::new();
    let mut qubits: Vec<u32> = Vec::new();
    let mut culmulative_params: Vec<u32> = Vec::new();
    let mut params: Vec<Param> = Vec::new();
    let param_distr = Uniform::new_inclusive(0., 2. * std::f64::consts::PI).unwrap();
    let mut circuit_qubits: Vec<Qubit> = (0..num_qubits).map(Qubit).collect();
    let mut selected_gates: Vec<StandardGate> = Vec::new();
    let mut extra_gates: Vec<StandardGate> = Vec::new();
    let mut conditional_gates: Vec<bool> = Vec::new();
    let mut conditional_values: Vec<u64> = Vec::new();

    // Apply arbitrary random operations in layers across all qubits.
    for layer_number in 0..depth {
        qubits.clear();
        culmulative_qubits.clear();
        params.clear();
        culmulative_params.clear();
        selected_gates.clear();
        extra_gates.clear();
        // We generate all the randomness for the layer in one go, to avoid many separate calls to
        // the randomization routines, which can be fairly slow.

        // This reliably draws too much randomness, but it's less expensive than looping over more
        // calls to the rng. After, trim it down by finding the point when we've used all the qubits.

        // Due to the stochastic nature of generating a random circuit, the resulting ratios
        // may not precisely match the specified values from `num_operand_distribution`. Expect
        // greater deviations from the target ratios in quantum circuits with fewer qubits and
        // shallower depths, and smaller deviations in larger and deeper quantum circuits.
        // For more information on how the distribution changes with number of qubits and depth
        // refer to the pull request #12483 on Qiskit GitHub.
        selected_gates.extend(
            gates_to_consider
                .sample_weighted(&mut rng, num_qubits as usize, |gate| distribution[gate.0])
                .unwrap()
                .map(|x| x.1),
        );
        culmulative_qubits.extend(selected_gates.iter().map(|x| {
            let n_qubits = x.num_qubits();
            if n_qubits == 0 { 1 } else { n_qubits }
        }));
        for i in 1..culmulative_qubits.len() {
            // SAFETY: All indices are in bounds based on loop.
            let (left, right) = unsafe { culmulative_qubits.split_at_mut_unchecked(i) };
            right[0] += left[i - 1];
        }
        // Efficiently find the point in the list where the total gates would use as many as
        // possible of, but not more than, the number of qubits in the layer. If there's slack, fill
        // it with extra gates according to the provided distribution.
        let max_index = match culmulative_qubits.binary_search(&num_qubits) {
            Ok(index) => index,
            Err(index) => index,
        };
        let gate_specs = &selected_gates[..max_index];
        let mut slack = num_qubits - culmulative_qubits[max_index.saturating_sub(1)];
        for gate in gate_specs {
            let mut n_qubits = gate.num_qubits();
            if n_qubits == 0 {
                n_qubits = 1;
            }
            counter[n_qubits as usize - 1] += 1;
        }

        // Slack handling loop, this loop will add gates to fill
        // the slack while respecting the 'num_operand_distribution'
        while slack > 0 {
            let mut gate_added_flag = false;

            for i in (0..operand_dist.len()).rev() {
                let (key, dist) = operand_dist.get_index(i).unwrap();
                let frac = counter[*key as usize - 1] as f64 / total_gates as f64;
                if slack >= *key as u32 && frac < *dist {
                    let gates = ALL_GATES[*key as usize - 1];
                    let gate = gates.choose(&mut rng).unwrap();
                    extra_gates.push(*gate);
                    counter[*key as usize - 1] += 1;
                    total_gates += 1;
                    slack -= *key as u32;
                    gate_added_flag = true;
                }
            }
            if !gate_added_flag {
                break;
            }
        }
        total_gates += gate_specs.len();
        culmulative_qubits.clear();
        culmulative_qubits.extend(gate_specs.iter().chain(extra_gates.iter()).map(|x| {
            let n_qubits = x.num_qubits();
            if n_qubits == 0 { 1 } else { n_qubits }
        }));
        for i in 1..culmulative_qubits.len() {
            // SAFETY: All indices are in bounds based on loop.
            let (left, right) = unsafe { culmulative_qubits.split_at_mut_unchecked(i) };
            right[0] += left[i - 1];
        }
        qubits.extend(
            [0].iter()
                .chain(culmulative_qubits.iter().take(max_index))
                .copied(),
        );
        culmulative_params.extend(gate_specs.iter().map(|x| x.num_params()));
        for i in 1..culmulative_params.len() {
            // SAFETY: All indices are in bounds based on loop.
            let (left, right) = unsafe { culmulative_params.split_at_mut_unchecked(i) };
            right[0] += left[i - 1];
        }
        params.extend(
            param_distr
                .map(Param::Float)
                .sample_iter(&mut rng)
                .take(*culmulative_params.last().unwrap_or(&0) as usize),
        );
        params.shuffle(&mut rng);
        circuit_qubits.shuffle(&mut rng);
        let mut qargs = [Qubit(u32::MAX); 4];

        let mut add_gate = |circuit: &mut CircuitData,
                            params: &mut Vec<Param>,
                            idx: usize,
                            gate: &StandardGate|
         -> Result<(), CircuitDataError> {
            let start = idx;
            let end = idx + 1;
            for (j, qubit) in (qubits[start]..qubits[end]).enumerate() {
                qargs[j] = circuit_qubits[qubit as usize];
            }
            let gate_params = (0..gate.num_params())
                .map(|_| params.pop().unwrap())
                .collect();
            let inst = if reset && *gate == StandardGate::GlobalPhase {
                PackedInstruction {
                    op: StandardInstruction::Reset.into(),
                    qubits: circuit.add_qargs(&qargs[..1_usize]),
                    clbits: Default::default(),
                    params: None,
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                }
            } else {
                PackedInstruction::from_standard_gate(
                    *gate,
                    Some(Box::new(gate_params)),
                    circuit.add_qargs(&qargs[..gate.num_qubits() as usize]),
                )
            };
            circuit.push(inst)
        };

        if conditional && layer_number != 0 {
            conditional_gates.clear();
            conditional_values.clear();

            conditional_gates.extend(
                (&mut rng)
                    .sample_iter(StandardNormal)
                    .take(gate_specs.len())
                    .map(|x: f64| x < 0.1),
            );

            conditional_values.extend(
                conditional_gates
                    .iter()
                    .filter(|x| **x)
                    .map(|_| rng.random_range(0..1 << 63.min(num_qubits))),
            );
            for (idx, gate) in gate_specs.iter().enumerate() {
                if conditional_gates[idx] {
                    let mut qargs = [Qubit(u32::MAX); 4];
                    let cond_val = conditional_values.pop().unwrap();
                    let start = idx;
                    let end = idx + 1;
                    for (j, qubit) in (qubits[start]..qubits[end]).enumerate() {
                        qargs[j] = circuit_qubits[qubit as usize];
                    }
                    let gate_params = (0..gate.num_params())
                        .map(|_| params.pop().unwrap())
                        .collect();

                    let mut block =
                        CircuitData::with_capacity(gate.num_qubits(), 0, 1, Param::Float(0.))?;
                    let qargs_idx = (0..block.num_qubits() as u32)
                        .map(Qubit)
                        .collect::<Vec<_>>();
                    let inst = if reset && *gate == StandardGate::GlobalPhase {
                        PackedInstruction {
                            op: StandardInstruction::Reset.into(),
                            qubits: circuit.add_qargs(&qargs_idx),
                            clbits: Default::default(),
                            params: None,
                            label: None,
                            #[cfg(feature = "cache_pygates")]
                            py_op: OnceLock::new(),
                        }
                    } else {
                        PackedInstruction::from_standard_gate(
                            *gate,
                            Some(Box::new(gate_params)),
                            circuit.add_qargs(&qargs_idx),
                        )
                    };
                    block.push(inst)?;
                    let block_idx = circuit.add_block(block);
                    PackedInstruction::from_control_flow(
                        ControlFlowInstruction {
                            control_flow: ControlFlow::IfElse {
                                condition: Condition::Register(
                                    creg.clone().unwrap(),
                                    cond_val.into(),
                                ),
                            },
                            num_qubits: gate.num_qubits(),
                            num_clbits: 0,
                        },
                        vec![block_idx],
                        circuit.add_qargs(&qargs[..gate.num_qubits() as usize]),
                        Default::default(),
                        None,
                    );
                } else {
                    add_gate(&mut circuit, &mut params, idx, gate)?;
                }
            }
        } else {
            for (idx, gate) in gate_specs.iter().enumerate() {
                add_gate(&mut circuit, &mut params, idx, gate)?;
            }
        }

        if measure {
            for bit in 0..num_qubits {
                let inst = PackedInstruction {
                    op: StandardInstruction::Measure.into(),
                    qubits: circuit.add_qargs(&[Qubit(bit)]),
                    clbits: circuit.add_cargs(&[Clbit(bit)]),
                    params: None,
                    label: None,
                    #[cfg(feature = "cache_pygates")]
                    py_op: OnceLock::new(),
                };
                circuit.push(inst)?;
            }
        }
    }
    Ok(circuit)
}
