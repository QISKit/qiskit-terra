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

use pyo3::{
    Python, intern,
    types::{PyNone, PyString},
};
use qiskit_circuit::{
    Clbit, NoBlocks, Qubit,
    circuit_instruction::OperationFromPython,
    operations::{Param, StandardGate, multiply_param},
    packed_instruction::PackedOperation,
};
use smallvec::{SmallVec, smallvec};

// custom type for a more readable code
pub type Instruction = (
    PackedOperation,
    SmallVec<[Param; 3]>,
    Vec<Qubit>,
    Vec<Clbit>,
);

/// Return instructions to implement evolution of Pauli terms and projectors.
///
/// Args:
///     term: The term to evolve as string, e.g. "IXYZ" or "++II".
///     indices: The qubit indices the term acts on, e.g. if given as [0, 1, 2, 3] with the
///         Pauli "IXYZ", then the correspondence is I_0 X_1 Y_2 Z_3.
///     time: The rotation angle. Note that this will directly be used as input of the
///         rotation gate and not be multiplied by a factor of 2 (that should be done before so
///         that this function can remain Rust-only).
///     phase_gate_for_paulis: If ``true``, use the [StandardGate::Phase] instead of
///         [StandardGate::RZ] single-qubit Pauli rotations.
///     do_fountain: If ``true``, implement the CX propagation as "fountain" shape, where each
///         CX uses the top qubit as target. If ``false``, uses a "chain" shape, where CX in between
///         neighboring qubits are used.
///
/// Returns:
///     A pointer to an iterator over standard instructions.
pub fn sparse_term_evolution(
    pauli: &str,
    indices: Vec<u32>,
    time: Param,
    phase_gate_for_paulis: bool,
    do_fountain: bool,
) -> Box<dyn Iterator<Item = Instruction>> {
    // ensure the Pauli has no identity terms
    let binding = pauli.to_lowercase(); // lowercase for convenience
    let active = binding
        .as_str()
        .chars()
        .zip(indices)
        .filter(|(pauli, _)| *pauli != 'i');
    let (paulis, indices): (Vec<char>, Vec<u32>) = active.unzip();

    match (phase_gate_for_paulis, indices.len()) {
        (_, 0) => Box::new(std::iter::empty()),
        (false, 1) => Box::new(single_qubit_evolution(paulis[0], indices[0], time)),
        (false, 2) => two_qubit_evolution(&paulis, &indices, time),
        _ => Box::new(multi_qubit_evolution(
            &paulis,
            &indices,
            time,
            phase_gate_for_paulis,
            do_fountain,
        )),
    }
}

/// Implement a single-qubit Pauli evolution of a Pauli given as char, on a given index and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
fn single_qubit_evolution(
    pauli: char,
    index: u32,
    time: Param,
) -> Box<dyn Iterator<Item = Instruction>> {
    let qubit = vec![Qubit(index)];

    // We don't need to explicitly cover the |1><1| projector case (which is the Phase gate),
    // which will be handled by the multi-qubit evolution.
    match pauli {
        'x' => Box::new(std::iter::once((
            StandardGate::RX.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        'y' => Box::new(std::iter::once((
            StandardGate::RY.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        'z' => Box::new(std::iter::once((
            StandardGate::RZ.into(),
            smallvec![time],
            qubit,
            vec![],
        ))),
        _ => Box::new(multi_qubit_evolution(
            &[pauli],
            &[index],
            time,
            false,
            false,
        )),
    }
}

/// Implement a 2-qubit Pauli evolution of a Pauli string, on a given indices and
/// for given time. Note that the time here equals the angle of the rotation and is not
/// multiplied by a factor of 2.
///
/// If possible, Qiskit's native 2-qubit Pauli rotations are used. Otherwise, the general
/// multi-qubit evolution is called.
fn two_qubit_evolution<'a>(
    pauli: &[char],
    indices: &[u32],
    time: Param,
) -> Box<dyn Iterator<Item = Instruction> + 'a> {
    let qubits = vec![Qubit(indices[0]), Qubit(indices[1])];
    let paulistring: String = pauli.iter().collect();

    // We don't need to explicitly cover the |11><11| projector case (which is the CPhase gate),
    // which will be handled by the multi-qubit evolution. The Paulis need special treatment here
    // since the generic code would use CX-RZ-CX instead of the two-qubit Pauli standard gates.
    match paulistring.as_str() {
        "xx" => Box::new(std::iter::once((
            StandardGate::RXX.into(),
            smallvec![time],
            qubits,
            vec![],
        ))),
        "zx" => Box::new(std::iter::once((
            StandardGate::RZX.into(),
            smallvec![time],
            qubits,
            vec![],
        ))),
        "xz" => Box::new(std::iter::once((
            StandardGate::RZX.into(),
            smallvec![time],
            vec![qubits[1], qubits[0]],
            vec![],
        ))),
        "yy" => Box::new(std::iter::once((
            StandardGate::RYY.into(),
            smallvec![time],
            qubits,
            vec![],
        ))),
        "zz" => Box::new(std::iter::once((
            StandardGate::RZZ.into(),
            smallvec![time],
            qubits,
            vec![],
        ))),
        // Note: the CX modes (do_fountain=true/false) give the same circuit for a 2-qubit
        // Pauli, so we just set it to false here
        _ => Box::new(multi_qubit_evolution(pauli, indices, time, false, false)),
    }
}

/// Implement a multi-qubit evolution. See ``sparse_term_evolution`` detailed docs.
fn multi_qubit_evolution(
    pauli: &[char],
    indices: &[u32],
    time: Param,
    phase_gate_for_paulis: bool,
    do_fountain: bool,
) -> impl Iterator<Item = Instruction> + use<> {
    let mut control_qubits: Vec<Qubit> = Vec::new(); // indices of projectors
    let mut control_states: Vec<bool> = Vec::new(); // +1 projector (true) or -1 projector (false)
    let mut pauli_qubits: Vec<Qubit> = Vec::new(); // indices of Paulis
    let mut basis_change: Vec<Instruction> = Vec::new(); // basis changes for all

    let paulis = ['x', 'y', 'z'];
    let positive = ['+', 'r', '0']; // the +1 eigenstate projectors
    let empty_clbits: Vec<Clbit> = Vec::new(); // convenience def we can clone later on

    // We iterate over the sparse term representation and store the basis-changing Clifford
    // and whether the index is a Pauli or a projector. If the latter, we also store the control
    // state (0 for +1 eigenvalue projector, 1 for -1 eigenvalue projector).
    for (bit_term, index) in pauli.iter().zip(indices.iter()) {
        let q = Qubit(*index);
        match bit_term {
            'x' | '+' | '-' => basis_change.push((
                StandardGate::H.into(),
                smallvec![],
                vec![q],
                empty_clbits.clone(),
            )),
            'y' | 'r' | 'l' => basis_change.push((
                StandardGate::SX.into(),
                smallvec![],
                vec![q],
                empty_clbits.clone(),
            )),
            _ => {}
        };

        if paulis.contains(bit_term) {
            pauli_qubits.push(q);
        } else {
            control_qubits.push(q);
            control_states.push(positive.contains(bit_term));
        }
    }

    // get the inverse basis change: H -> H and SX -> SXdg
    let inverse_basis_change: Vec<Instruction> = basis_change
        .iter()
        .map(|(gate, _, qubit, _)| match gate.standard_gate() {
            StandardGate::H => (
                StandardGate::H.into(),
                smallvec![],
                qubit.clone(),
                empty_clbits.clone(),
            ),
            StandardGate::SX => (
                StandardGate::SXdg.into(),
                smallvec![],
                qubit.clone(),
                empty_clbits.clone(),
            ),
            _ => unreachable!("Invalid basis-changing Clifford."),
        })
        .collect();

    // for the Pauli evolution get the CX propagation up to the first qubit, and down
    let (chain_up, chain_down) = match do_fountain {
        true => (
            cx_fountain(pauli_qubits.clone()),
            cx_fountain(pauli_qubits.clone()).rev(),
        ),
        false => (
            cx_chain(pauli_qubits.clone()),
            cx_chain(pauli_qubits.clone()).rev(),
        ),
    };

    // Get the Z/phase rotation. If we have more than a single rotation qubit, each projector
    // is implemented as open/closed control of the rotation.
    let rotation = if !pauli_qubits.is_empty() {
        // Here we have at least one Pauli rotation, hence the rotation is an RZ rotation
        // per default (unless the user specified otherwise)
        let params: SmallVec<[Param; 3]> = smallvec![time];
        let base_gate = if phase_gate_for_paulis {
            StandardGate::Phase
        } else {
            StandardGate::RZ
        };

        let (packed, qubits) = if control_qubits.is_empty() {
            // simple case: no projectors, so we're done
            let gate: PackedOperation = base_gate.into();
            (gate, vec![pauli_qubits[0]])
        } else {
            // if we have projectors, add controls to the target rotation
            let controlled = add_control(base_gate, &params, &control_states);

            // in our control convention the
            control_qubits.reverse();
            control_qubits.push(pauli_qubits[0]);
            (controlled, control_qubits)
        };
        vec![(packed, params, qubits, empty_clbits.clone())]
    } else {
        // Here we purely have projectors, meaning the target rotation is a phase gate. Remember
        // we have to adjust the rotation angle to account for the different conventions;
        // RZ(t) = exp(-i t/2 Z) vs. P(t) = diag(1, exp(i t)).
        let params: SmallVec<[Param; 3]> = smallvec![multiply_param(&time, -0.5)];
        let (packed, qubits) = if control_qubits.len() == 1 {
            let gate: PackedOperation = StandardGate::Phase.into();
            (gate, vec![control_qubits[0]])
        } else {
            let controlled = add_control(StandardGate::Phase, &params, &control_states[1..]);
            control_qubits.reverse();
            (controlled, control_qubits)
        };
        let inst: Instruction = (packed, params, qubits.clone(), empty_clbits.clone());

        if control_states[0] {
            // sandwich in X gates for the correct projector
            let x: Instruction = (
                StandardGate::X.into(),
                smallvec![],
                vec![*qubits.last().unwrap()],
                empty_clbits.clone(),
            );
            vec![x.clone(), inst, x]
        } else {
            vec![inst]
        }
    };

    // and finally chain everything together
    basis_change
        .into_iter()
        .chain(chain_down)
        .chain(rotation)
        .chain(chain_up)
        .chain(inverse_basis_change)
}

/// Build a CX chain over the active qubits. E.g. with q_1 inactive, this would return
///
///                    ┌───┐
///     q_0: ──────────┤ X ├
///                    └─┬─┘
///     q_1: ────────────┼──
///               ┌───┐  │
///     q_2: ─────┤ X ├──■──
///          ┌───┐└─┬─┘
///     q_3: ┤ X ├──■───────
///          └─┬─┘
///     q_4: ──■────────────
///
fn cx_chain(qubits: Vec<Qubit>) -> Box<dyn DoubleEndedIterator<Item = Instruction>> {
    let num_terms = qubits.len();
    if num_terms < 2 {
        return Box::new(std::iter::empty());
    }

    Box::new(
        (0..num_terms - 1)
            .map(move |i| (qubits[i], qubits[i + 1]))
            .map(|(target, ctrl)| {
                (
                    StandardGate::CX.into(),
                    smallvec![],
                    vec![ctrl, target],
                    vec![],
                )
            }),
    )
}

/// Build a CX fountain over the active qubits. E.g. with q_1 inactive, this would return
///
///         ┌───┐┌───┐┌───┐
///    q_0: ┤ X ├┤ X ├┤ X ├
///         └─┬─┘└─┬─┘└─┬─┘
///    q_1: ──┼────┼────┼──
///           │    │    │
///    q_2: ──■────┼────┼──
///                │    │
///    q_3: ───────■────┼──
///                     │
///    q_4: ────────────■──
///
fn cx_fountain(qubits: Vec<Qubit>) -> Box<dyn DoubleEndedIterator<Item = Instruction>> {
    let num_terms = qubits.len();
    if num_terms < 2 {
        return Box::new(std::iter::empty());
    }

    let first_qubit = qubits[0];
    Box::new((1..num_terms).rev().map(move |i| {
        let ctrl = qubits[i];
        (
            StandardGate::CX.into(),
            smallvec![],
            vec![ctrl, first_qubit],
            vec![],
        )
    }))
}

/// Add controls to a standard gate with a specified control state.
fn add_control(gate: StandardGate, params: &[Param], control_state: &[bool]) -> PackedOperation {
    // This function does not return a PyResult to keep the evolution functions free from PyO3.
    // We know that all calls here should be valid and unwrap eagerly.
    Python::attach(|py| {
        let pygate = gate
            .create_py_op(py, Some(params.iter().cloned().collect()), None)
            .expect("Failed to create Py version of standard gate.");
        let num_controls = control_state.len();
        let py_control_state = PyString::new(
            py,
            control_state
                .iter()
                .map(|is_open| is_open.then(|| '0').unwrap_or('1'))
                .collect::<String>()
                .as_str(),
        );
        let label = PyNone::get(py);
        let controlled_gate = pygate
            .call_method1(
                py,
                intern!(py, "control"),
                (num_controls, label, py_control_state, false),
            )
            .expect("Failed to call .control()")
            .extract::<OperationFromPython<NoBlocks>>(py)
            .expect("The control state should be valid and match the number of controls.");

        controlled_gate.operation
    })
}
