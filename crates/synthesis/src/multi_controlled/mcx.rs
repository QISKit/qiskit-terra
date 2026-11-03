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

use pyo3::types::PyAnyMethods;
use pyo3::{PyResult, Python};
use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
use qiskit_circuit::operations::{
    Operation, OperationRef, Param, PyInstruction, PyOpKind, StandardGate, multiply_param,
};
use qiskit_circuit::{BlocksMode, imports};
use qiskit_circuit::{Clbit, Qubit, VarsMode};
use smallvec::SmallVec;

use qiskit_circuit::instruction::Parameters;
use std::f64::consts::PI;

use crate::QiskitError;
const PI2: f64 = PI / 2.0;

/// Definition circuit for CCX.
pub fn ccx() -> CircuitData {
    StandardGate::CCX.definition(&[]).unwrap()
}

/// Definition circuit for C3X.
pub fn c3x() -> CircuitData {
    StandardGate::C3X.definition(&[]).unwrap()
}

/// Definition circuit for RCCX.
fn rccx() -> CircuitData {
    StandardGate::RCCX.definition(&[]).unwrap()
}

/// Definition circuit for RC3X.
fn rc3x() -> CircuitData {
    StandardGate::RC3X.definition(&[]).unwrap()
}

/// Definition circuit for C3SX.
fn c3sx() -> CircuitData {
    StandardGate::C3SX.definition(&[]).unwrap()
}

/// Convenience methods to add gates to the circuit.
///
/// This trait is **not** intended to be user-facing. It defines utility functions
/// that make the code easier to read and that are used only for synthesis.
trait CircuitDataForSynthesis {
    /// Appends H to the circuit.
    fn h(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends X to the circuit.
    #[allow(dead_code)]
    fn x(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends T to the circuit.
    fn t(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends Tdg to the circuit.
    fn tdg(&mut self, q: u32) -> Result<(), CircuitDataError>;

    /// Appends Phase to the circuit.
    #[allow(dead_code)]
    fn p(&mut self, theta: impl Into<Param>, q: u32) -> Result<(), CircuitDataError>;

    /// Appends CX to the circuit.
    fn cx(&mut self, q1: u32, q2: u32) -> Result<(), CircuitDataError>;

    /// Appends CPhase to the circuit.
    fn cp(&mut self, theta: impl Into<Param>, q1: u32, q2: u32) -> Result<(), CircuitDataError>;

    /// Appends CCPhase to the circuit.
    fn ccp(&mut self, theta: f64, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Appends CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError>;

    /// Appends CRX to the circuit.
    fn crx(&mut self, theta: f64, q1: u32, q2: u32) -> Result<(), CircuitDataError>;

    /// Compose ``other`` into ``self``, while remapping the qubits
    /// over which ``other`` is defined. The operations are added in-place.
    fn compose(
        &mut self,
        other: &Self,
        qargs_map: &[Qubit],
        cargs_map: &[Clbit],
    ) -> Result<(), CircuitDataError>;

    /// Construct the inverse circuit
    fn inverse(&self) -> Result<CircuitData, CircuitDataError>;
}

impl CircuitDataForSynthesis for CircuitData {
    /// Appends H to the circuit.
    #[inline]
    fn h(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::H, &[], &[Qubit(q)])
    }

    /// Appends X to the circuit.
    #[inline]
    fn x(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::X, &[], &[Qubit(q)])
    }

    /// Appends T to the circuit.
    #[inline]
    fn t(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::T, &[], &[Qubit(q)])
    }

    /// Appends Tdg to the circuit.
    #[inline]
    fn tdg(&mut self, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::Tdg, &[], &[Qubit(q)])
    }

    /// Appends Phase to the circuit.
    #[inline]
    fn p(&mut self, theta: impl Into<Param>, q: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::Phase, &[theta.into()], &[Qubit(q)])
    }

    /// Appends CX to the circuit.
    #[inline]
    fn cx(&mut self, q1: u32, q2: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(StandardGate::CX, &[], &[Qubit(q1), Qubit(q2)])
    }

    /// Appends CPhase to the circuit.
    #[inline]
    fn cp(&mut self, theta: impl Into<Param>, q1: u32, q2: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(
            StandardGate::CPhase,
            &[theta.into()],
            &[Qubit(q1), Qubit(q2)],
        )
    }

    /// Appends CRX to the circuit.
    #[inline]
    fn crx(&mut self, theta: f64, q1: u32, q2: u32) -> Result<(), CircuitDataError> {
        self.push_standard_gate(
            StandardGate::CRX,
            &[Param::Float(theta)],
            &[Qubit(q1), Qubit(q2)],
        )
    }

    fn ccp(&mut self, theta: f64, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.cx(q1, q3)?;
        self.p(-theta / 4., q3)?;
        self.cx(q2, q3)?;
        self.p(theta / 4., q3)?;
        self.cx(q1, q3)?;
        self.p(-theta / 4., q3)?;
        self.cx(q2, q3)?;
        self.p(theta / 4., q3)?;
        self.p(theta / 4., q1)?;
        self.p(theta / 4., q2)?;
        self.cx(q1, q2)?;
        self.p(-theta / 4., q2)?;
        self.cx(q1, q2)?;
        Ok(())
    }

    /// Appends the decomposition of the CCX to the circuit.
    fn ccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.compose(&ccx(), &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Appends RCCX to the circuit.
    fn rccx(&mut self, q1: u32, q2: u32, q3: u32) -> Result<(), CircuitDataError> {
        self.compose(&rccx(), &[Qubit(q1), Qubit(q2), Qubit(q3)], &[])
    }

    /// Compose ``other`` into ``self``, while remapping the qubits over which ``other`` is defined.
    /// The operations are added in-place.
    fn compose(
        &mut self,
        other: &Self,
        qargs_map: &[Qubit],
        cargs_map: &[Clbit],
    ) -> Result<(), CircuitDataError> {
        if other.num_qubits() > self.num_qubits() {
            panic!("Cannot compose a larger circuit onto a smaller circuit.");
        }

        for inst in other.data() {
            let remapped_qubits: Vec<Qubit> = other
                .get_qargs(inst.qubits)
                .iter()
                .map(|q| qargs_map[q.index()])
                .collect();
            let remapped_clbits: Vec<Clbit> = other
                .get_cargs(inst.clbits)
                .iter()
                .map(|c| cargs_map[c.index()])
                .collect();

            self.push_packed_operation(
                inst.op.clone(),
                inst.params.as_deref().cloned(),
                &remapped_qubits,
                &remapped_clbits,
            )?;
        }

        self.add_global_phase(other.global_phase())?;
        Ok(())
    }

    /// Construct the inverse circuit
    fn inverse(&self) -> Result<CircuitData, CircuitDataError> {
        let inverse_global_phase = multiply_param(self.global_phase(), -1.0);

        let mut inverse_circuit =
            CircuitData::copy_empty_like(self, VarsMode::Alike, BlocksMode::Keep)?;
        inverse_circuit.set_global_phase_param(inverse_global_phase)?;

        let data = self.data();

        for i in 0..data.len() {
            let inst = &data[data.len() - 1 - i];

            let inverse_inst: Option<(StandardGate, SmallVec<[Param; 3]>)> = match &inst.op.view() {
                OperationRef::StandardGate(gate) => gate.inverse(inst.params_view()),
                _ => None,
            };

            if inverse_inst.is_none() {
                panic!(
                    "The circuit cannot be inverted: {} is not a standard gate.",
                    inst.op.name()
                );
            }

            let (inverse_op, inverse_op_params) = inverse_inst.unwrap();

            inverse_circuit.push_packed_operation(
                inverse_op.into(),
                Some(Parameters::Params(inverse_op_params)),
                self.get_qargs(inst.qubits),
                self.get_cargs(inst.clbits),
            )?;
        }
        Ok(inverse_circuit)
    }
}

/// Efficient synthesis for 4-controlled X-gate.
pub fn c4x() -> CircuitData {
    c4x_inner().expect("c4x is a fixed circuit with valid qubit indices and float parameters")
}

fn c4x_inner() -> Result<CircuitData, CircuitDataError> {
    let mut circuit = CircuitData::with_capacity(5, 0, 0, Param::Float(0.0))?;
    circuit.h(4)?;
    circuit.cp(PI2, 3, 4)?;
    circuit.h(4)?;
    circuit.compose(&rc3x(), &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)], &[])?;
    circuit.h(4)?;
    circuit.cp(-PI2, 3, 4)?;
    circuit.h(4)?;
    circuit.compose(
        &rc3x().inverse()?,
        &[Qubit(0), Qubit(1), Qubit(2), Qubit(3)],
        &[],
    )?;
    circuit.compose(&c3sx(), &[Qubit(0), Qubit(1), Qubit(2), Qubit(4)], &[])?;
    Ok(circuit)
}

/// Adds gates of the "action gadget" to the circuit
fn add_action_gadget(
    circuit: &mut CircuitData,
    q0: u32,
    q1: u32,
    q2: u32,
) -> Result<(), CircuitDataError> {
    circuit.h(q2)?;
    circuit.t(q2)?;
    circuit.cx(q0, q2)?;
    circuit.tdg(q2)?;
    circuit.cx(q1, q2)
}

/// Adds gates of the "reset gadget" to the circuit
fn add_reset_gadget(
    circuit: &mut CircuitData,
    q0: u32,
    q1: u32,
    q2: u32,
) -> Result<(), CircuitDataError> {
    circuit.cx(q1, q2)?;
    circuit.t(q2)?;
    circuit.cx(q0, q2)?;
    circuit.tdg(q2)?;
    circuit.h(q2)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Iten et al. [1].
///
/// For :math:`k\ge 4` the method uses :math:`k - 2` dirty ancillary qubits, producing a circuit
/// with :math:`2 * k - 1` qubits and at most :math:`8 * k - 6` CX gates. For :math:`k\le 3`
/// explicit efficient circuits are used instead.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - relative_phase: when set to `true`, the method applies the optimized multi-controlled
///   X gate up to a relative phase, in a way that the relative phases of the `action part`
///   cancel out with the relative phases of the `reset part`.
/// - action_only: when set to `true`, the methods applies only the `action part`.
///
/// # References
///
/// 1. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
///    [arXiv:1501.06911] (https://arxiv.org/abs/1501.06911).
pub fn synth_mcx_n_dirty_i15(
    num_controls: usize,
    relative_phase: bool,
    action_only: bool,
) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 2 || (num_controls == 3 && !relative_phase) {
        synth_mcx_explicit(num_controls)
    } else {
        let num_ancillas = num_controls - 2;
        let num_qubits = num_controls + 1 + num_ancillas;
        let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, 0, Param::Float(0.0))?;

        let controls: Vec<u32> = (0..num_controls).map(|q| q as u32).collect();
        let target = num_controls as u32;
        let ancillas: Vec<u32> = ((num_controls + 1)..num_qubits).map(|q| q as u32).collect();

        for j in 0..2 {
            if !relative_phase {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            } else if j == 0 {
                add_action_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            } else if j == 1 {
                add_reset_gadget(
                    &mut circuit,
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
            }

            // action part
            for i in (0..num_controls - 3).rev() {
                add_action_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1])?;
            }

            circuit.rccx(controls[0], controls[1], ancillas[0])?;

            // reset part
            for i in 0..num_controls - 3 {
                add_reset_gadget(&mut circuit, controls[i + 2], ancillas[i], ancillas[i + 1])?;
            }

            if action_only {
                circuit.ccx(
                    controls[num_controls - 1],
                    ancillas[num_controls - 3],
                    target,
                )?;
                break;
            }
        }
        Ok(circuit)
    }
}

/// Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
///
/// Together with the caller's initial rccx(control_0, control_1, ancilla), this implements
/// the "up" and "down" Toffoli ladders of Step-1 and Step-2 in Fig. 3a of [1].
///
/// # Arguments
/// - num_controls: the number of qubits involved in the ladder operation.
///
/// # Returns
///
/// A tuple consisting of the linear-depth ladder circuit and the index of the control qubit
/// to apply to the final CCX gate.
///
/// # Errors
///
/// Returns an error if `num_controls < 3`.
///
/// # References
///
/// 1. Khattar and Gidney, *Rise of conditionally clean ancillae for optimizing quantum circuits*,
///    [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
fn linear_depth_ladder_ops(num_controls: u32) -> Result<(CircuitData, u32), CircuitDataError> {
    let k = num_controls;

    if k < 3 {
        return Err(QiskitError::new_err("linear_depth_ladder_ops requires >= 3 controls.").into());
    }

    // At most k-2 rungs, each RCCX (9 gates) + X (1 gate); pre-size to avoid reallocation.
    let mut circuit =
        CircuitData::with_capacity(k, 0, 10 * k.saturating_sub(2) as usize, Param::Float(0.0))?;

    // Fold all k controls into a running partial-AND via RCCX+X pairs. RCCX is used
    // instead of CCX because its relative phase cancels when composed with its inverse.
    // The trailing X after each RCCX prepares the written qubit as a control for the next rung.

    // Up-sweep: fold controls into qubit 1 two at a time, walking toward the middle.
    for i in (1..k - 2).step_by(2) {
        circuit.rccx(i + 1, i + 2, i)?;
        circuit.x(i)?;
    }

    // Peak: where the up-sweep and down-sweep meet. Parity of k determines which qubits
    // participate. target < 0 means no distinct peak (up-sweep already reaches the end).
    let has_peak = if k.is_multiple_of(2) { k >= 6 } else { k >= 5 };

    if has_peak {
        let (a, b, peak) = if k.is_multiple_of(2) {
            (k - 3, k - 5, k - 6)
        } else {
            (k - 1, k - 4, k - 5)
        };
        circuit.rccx(a, b, peak)?;
        circuit.x(peak)?;

        // Down-sweep: mirror of the up-sweep, walking back toward qubit 1.
        for i in (2..=peak).rev().step_by(2) {
            circuit.rccx(i, i - 1, i - 2)?;
            circuit.x(i - 2)?;
        }
    }
    // final_ctrl holds the AND of the controls not covered by the ladder (qubits 0..1).
    // The caller pairs it with the ancilla in one final CCX.
    let final_ctrl = 5u32.saturating_sub(k);
    Ok((circuit, final_ctrl))
}

/// Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`1` ancillary qubit.
///
/// The construction is described in Section 5 of [1]. For :math:`k\le 2`, the returned circuit
/// consists of a single X, CX, or CCX gate (corresponding to :math:`k = 0, 1, 2`, respectively)
/// and uses no ancillary qubits.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - clean: if `true`, the ancilla is clean; if `false`, the ancilla is dirty.
///
/// # References
///
/// 1. Khattar and Gidney, *Rise of conditionally clean ancillae for optimizing quantum circuits*,
///    [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
pub fn synth_mcx_1_kg24(num_controls: usize, clean: bool) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 2 {
        synth_mcx_explicit(num_controls)
    } else {
        // --- General case: k >= 3 controls, 1 ancilla ---
        let k = num_controls as u32;
        let target = k;
        let ancilla = k + 1;

        let (ladder, final_ctrl) = linear_depth_ladder_ops(k)?;

        // Precompute once; the dirty-ancilla case reuses it for the second pass.
        let ladder_inv = ladder.inverse()?;

        // num_passes=1 for clean ancilla, 2 for dirty (repeat to cancel initial-state dependence).
        // Fixed costs: 2 RCCX (9 gates each) + num_passes * (2 * ladder + 1 CCX (15 gates)).
        let num_passes = if clean { 1 } else { 2 };
        let instruction_capacity = 2 * 9 + num_passes * (2 * ladder.data().len() + 15);
        let mut circuit =
            CircuitData::with_capacity(k + 2, 0, instruction_capacity, Param::Float(0.0))?;

        let controls_map: Vec<Qubit> = (0..k).map(Qubit).collect();

        // The steps below follow the base (clean) construction of Fig. 3a in [1].
        // Step 1 (up ladder), part 1: turn the ancilla into a "conditionally clean" qubit
        // holding AND(control_0, control_1) — the first gate of Fig. 3a's "up" ladder.
        // RCCX is used (rather than CCX) because its stray relative phase is harmless:
        // it will cancel against the same RCCX's inverse later.
        circuit.rccx(0, 1, ancilla)?;
        // Step 1 (up ladder), part 2, and Step 2 (down ladder): fold in the remaining
        // controls so that `final_ctrl` ends up holding AND(control_2, ..., control_{k-1}).
        circuit.compose(&ladder, &controls_map, &[])?;
        // Step 3: the actual MCX action — flip the target iff the ancilla AND
        // final_ctrl are both set, i.e. iff AND(control_0, ..., control_{k-1}) holds.
        circuit.ccx(ancilla, final_ctrl, target)?;
        // Step 4: undo Steps 1-2, restoring every control qubit and the ancilla
        // to their original state (the ancilla ends back at |0> if it started there).
        circuit.compose(&ladder_inv, &controls_map, &[])?;
        circuit.rccx(0, 1, ancilla)?;

        if !clean {
            // Dirty ancilla: repeat the compute/uncompute sandwich above once more
            // (toggle-detection) so that dependence on the ancilla's unknown initial
            // state cancels out.
            circuit.compose(&ladder, &controls_map, &[])?;
            circuit.ccx(ancilla, final_ctrl, target)?;
            circuit.compose(&ladder_inv, &controls_map, &[])?;
        }

        Ok(circuit)
    }
}

/// Builds the log-depth AND-folding ladder (Step 2 in Fig. 4b of [1]).
///
/// Reduces `num_controls` controls to a set of AND-flag qubits using a
/// doubling-width tree of parallel X+RCCX gates. Qubits 0 and 1 must have
/// already been primed by the caller via `rccx(0, 1, ancilla0)`.
///
/// Returns `(ladder_circuit, leftover_ctrls)`: the ladder gates and the qubit
/// indices to feed into the final CCX or 1-ancilla MCX. Order of
/// `leftover_ctrls` is not significant (MCX is symmetric in its controls).
///
/// # References
///
/// 1. Khattar and Gidney, *Rise of conditionally clean ancillae for optimizing quantum circuits*,
///    [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
fn log_depth_ladder_ops(num_controls: u32) -> Result<(CircuitData, Vec<u32>), CircuitDataError> {
    if num_controls < 3 {
        return Err(QiskitError::new_err("log_depth_ladder_ops requires >= 3 controls.").into());
    }

    // Rough upper bound: at most (num_controls-1) X+RCCX pairs across all rounds, 10 gates each.
    let mut qc = CircuitData::with_capacity(
        num_controls,
        0,
        10 * (num_controls - 1) as usize,
        Param::Float(0.0),
    )?;

    // Qubits 0 and 1 seed the pool: the caller's priming RCCX already consumed them,
    // so they are free to be overwritten as AND-flag targets.
    let mut ancilla_pool: Vec<u32> = vec![0, 1];
    // Controls not yet folded into the AND-tree.
    let mut unprocessed: Vec<u32> = (2..num_controls).collect();
    // Single survivors from each round — passed to the caller's final gate.
    let mut leftover_ctrls: Vec<u32> = Vec::new();
    // Hoisted out of the outer loop to reuse the allocation across rounds.
    let mut newly_freed: Vec<u32> = Vec::new();

    // --- Outer loop: one round per batch of unprocessed controls -----------
    // Each round pulls a batch whose size fits the current ancilla pool (+1),
    // folds it entirely via the inner loop, then doubles the pool.
    // Loop exits when at most one unprocessed control remains (handled below).
    while unprocessed.len() > 1 {
        // Batch size = pool + 1 ensures the inner tree fully collapses to one survivor.
        let batch_size = (ancilla_pool.len() + 1).min(unprocessed.len());
        let mut batch: Vec<u32> = unprocessed.drain(..batch_size).collect();

        // Track controls consumed by this batch; they rejoin the pool after the inner loop.
        newly_freed.clear();

        // --- Inner loop: parallel X+RCCX tree, halving `batch` each step ---
        // Each step pairs up the elements of `batch` and writes their AND-flags
        // into the last `pair_count` entries of `ancilla_pool`. After the step,
        // `batch` is updated to hold the AND-flag target qubits (plus the
        // odd-one-out if any), ready to be paired again in the next step.
        while batch.len() > 1 {
            let pair_count = batch.len() / 2;
            // `leftover`: 0 or 1 element that cannot be paired this step.
            let leftover = batch.len() % 2;
            let pool_len = ancilla_pool.len();

            let ctrl_a = &batch[leftover..leftover + pair_count];
            let ctrl_b = &batch[leftover + pair_count..];
            let targets = &ancilla_pool[pool_len - pair_count..];

            // Phase 1: X on every target (|0⟩ → |1⟩) so the RCCX relative phase
            // is correct for the "conditionally clean ancilla" construction.
            for &t in targets {
                qc.x(t)?;
            }
            // Phase 2: RCCX(aᵢ, bᵢ, tᵢ) in parallel — writes AND(aᵢ, bᵢ) into tᵢ.
            for i in 0..pair_count {
                qc.rccx(ctrl_a[i], ctrl_b[i], targets[i])?;
            }

            // Record consumed controls; they'll rejoin ancilla_pool after this inner loop.
            newly_freed.extend_from_slice(&batch[leftover..]);

            batch[leftover..leftover + pair_count].copy_from_slice(targets);
            batch.truncate(leftover + pair_count);
            batch.sort_unstable();
            ancilla_pool.truncate(pool_len - pair_count);
        }

        // `batch` is now one element: the AND-flag for all controls in this batch.
        // Return consumed controls to the pool (order irrelevant — AND is commutative).
        ancilla_pool.extend_from_slice(&newly_freed);
        ancilla_pool.sort_unstable();
        leftover_ctrls.extend_from_slice(&batch);
    }

    // Odd leftover control (if any) passes directly to the caller's final gate.
    leftover_ctrls.extend_from_slice(&unprocessed);
    leftover_ctrls.sort_unstable();
    Ok((qc, leftover_ctrls))
}

/// Flips `target` iff AND(ancilla0, leftover_ctrls...) holds (Step 3 of Fig. 4b in [1]).
///
/// Uses a single CCX when `leftover_ctrls` has one element, or a precomputed
/// `synth_mcx_1_kg24` circuit + qubit map (`mid_mcx`) for two or more.
/// Called once per pass (twice in the dirty-ancilla case).
///
/// [1]: Khattar and Gidney, arXiv:2407.17966
fn synth_mcx_2_finish(
    circuit: &mut CircuitData,
    ancilla0: u32,
    target: u32,
    leftover_ctrls: &[u32],
    mid_mcx: Option<(&CircuitData, &[Qubit])>,
) -> Result<(), CircuitDataError> {
    if leftover_ctrls.len() == 1 {
        // Single leftover: CCX(ancilla0, leftover_ctrl, target).
        circuit.ccx(ancilla0, leftover_ctrls[0], target)
    } else {
        let (mid_mcx, qubits_map) =
            mid_mcx.expect("mid_mcx must be precomputed when leftover_ctrls.len() > 1");
        circuit.compose(mid_mcx, qubits_map, &[])
    }
}

/// Synthesize a multi-controlled X gate with :math:`k\ge 3` controls using :math:`2`
/// ancillary qubits, producing a circuit with depth :math:`O(\log k)`, as described in
/// Sec. 5.2/5.4 of [1]. For :math:`k\le 2`, the returned circuit consists of a single X,
/// CX, or CCX gate (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no
/// ancillary qubits.
///
/// # Arguments
/// - num_controls: the number of control qubits.
/// - clean: if `true`, both ancillas are clean; if `false`, both are dirty.
///
/// # References
///
/// 1. Khattar and Gidney, *Rise of conditionally clean ancillae for optimizing quantum circuits*,
///    [arXiv:2407.17966](https://arxiv.org/abs/2407.17966).
pub fn synth_mcx_2_kg24(num_controls: usize, clean: bool) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 2 {
        synth_mcx_explicit(num_controls)
    } else {
        // --- General case: k >= 3 controls, 2 ancillas ---
        let k = num_controls as u32;
        let target = k;
        let ancilla0 = k + 1;
        let ancilla1 = k + 2;

        let controls_map: Vec<Qubit> = (0..k).map(Qubit).collect();

        let (ladder, leftover_ctrls) = log_depth_ladder_ops(k)?;
        // Precompute once; the dirty-ancilla case reuses it for the second pass.
        let ladder_inv = ladder.inverse()?;

        // Precompute mid-MCX and its qubit map together once; reused for both
        // passes in the dirty case. clean=true: ancilla1 is fully uncomputed
        // per pass. None → a single CCX suffices instead.
        let mid_mcx_and_map: Option<(CircuitData, Vec<Qubit>)> = if leftover_ctrls.len() > 1 {
            let circuit = synth_mcx_1_kg24(leftover_ctrls.len() + 1, true)?;
            let mut qubits_map: Vec<Qubit> = Vec::with_capacity(leftover_ctrls.len() + 3);
            qubits_map.push(Qubit(ancilla0));
            qubits_map.extend(leftover_ctrls.iter().map(|&c| Qubit(c)));
            qubits_map.push(Qubit(target));
            qubits_map.push(Qubit(ancilla1));
            Some((circuit, qubits_map))
        } else {
            None
        };
        let mid_mcx_for_finish = mid_mcx_and_map
            .as_ref()
            .map(|(circuit, qubits_map)| (circuit, qubits_map.as_slice()));

        // num_passes=1 for clean ancilla, 2 for dirty (repeat to cancel initial-state dependence).
        // Fixed costs: 2 RCCX (9 gates each) + num_passes * (2 * ladder + finish step).
        // finish_len = 15 when leftover_ctrls.len() == 1 (one CCX).
        let finish_len = mid_mcx_and_map.as_ref().map_or(15, |(m, _)| m.data().len());
        let num_passes = if clean { 1 } else { 2 };
        let instruction_capacity = 2 * 9 + num_passes * (2 * ladder.data().len() + finish_len);
        let mut circuit =
            CircuitData::with_capacity(k + 3, 0, instruction_capacity, Param::Float(0.0))?;

        // The steps below follow Fig. 4b in [1].
        // Step 1: prime -- turn ancilla0 into a conditionally clean qubit holding
        // AND(control_0, control_1). RCCX is used (rather than CCX) because its stray
        // relative phase is harmless: it will cancel against the same RCCX's inverse
        // in step 5.
        circuit.rccx(0, 1, ancilla0)?;
        // Step 2: fold -- log-depth AND-folding ladder over the remaining controls.
        circuit.compose(&ladder, &controls_map, &[])?;
        // Step 3: finish -- flip the target iff AND(all k controls) holds.
        synth_mcx_2_finish(
            &mut circuit,
            ancilla0,
            target,
            &leftover_ctrls,
            mid_mcx_for_finish,
        )?;
        // Step 4: unfold -- undo step 2, restoring all control qubits.
        circuit.compose(&ladder_inv, &controls_map, &[])?;
        // Step 5: unprime -- undo step 1, restoring ancilla0 to its initial state.
        circuit.rccx(0, 1, ancilla0)?;

        if !clean {
            // Dirty ancilla: repeat the compute/uncompute sandwich (toggle-detection) to
            // cancel dependence on the ancillas' initial state. Prime/unprime happen only
            // once (matches synth_mcx_1_kg24's dirty-ancilla pattern).
            circuit.compose(&ladder, &controls_map, &[])?;
            synth_mcx_2_finish(
                &mut circuit,
                ancilla0,
                target,
                &leftover_ctrls,
                mid_mcx_for_finish,
            )?;
            circuit.compose(&ladder_inv, &controls_map, &[])?;
        }

        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate with :math:`k` controls using no auxiliary qubits via the relation
/// MCX = H · MCP(π) · H.
/// For details on MCP synthesis see Python's `synth_mcp_noaux_default` in `qiskit/synthesis/multi_controlled/mcp_synthesis.py`.
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits.
///
#[allow(dead_code)]
pub fn synth_mcx_mcp_noaux(
    py: Python,
    num_controls: usize,
) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 4 {
        synth_mcx_explicit(num_controls)
    } else {
        let num_qubits = (num_controls + 1) as u32;
        let target = num_controls as u32;

        let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
        circuit.h(target)?;

        let mcphase_cls = imports::MCPHASE_GATE.get_bound(py);
        let mcphase_gate = mcphase_cls
            .call1((PI, num_controls))
            .map_err(CircuitDataError::ErrorFromPython)?;

        let inst = PyInstruction {
            qubits: num_qubits,
            clbits: 0,
            params: 1,
            op_name: "mcphase".to_string(),
            ob: mcphase_gate.into(),
            kind: PyOpKind::Gate,
        };
        circuit.push_packed_operation(
            inst.into(),
            Some(Parameters::Params(smallvec::smallvec![Param::Float(PI)])),
            &(0..num_qubits).map(Qubit).collect::<Vec<Qubit>>(),
            &[],
        )?;

        circuit.h(target)?;

        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate with :math:`k` controls using a single clean
/// ancillary qubit, by Barenco et al. [1] and Iten et al. [2].
///
/// For :math:`k \ge 5` the method uses 1 clean ancillary qubit, producing a circuit with
/// :math:`k + 2` qubits and at most :math:`16 * k - 24` CX gates. For :math:`k \le 4`
/// explicit efficient circuits that require no ancillary qubits are used instead.
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// The synthesized quantum circuit.
///
/// # References
///
/// 1. Barenco et al., *Elementary gates for quantum computation*, Phys. Rev. A52 3457 (1995),
///    [arXiv:quant-ph/9503016] (https://arxiv.org/abs/quant-ph/9503016).
/// 2. Iten et al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
///    [arXiv:1501.06911] (https://arxiv.org/abs/1501.06911).
pub fn synth_mcx_1_clean_b95(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 4 {
        synth_mcx_explicit(num_controls)
    } else {
        // k >= 5: 1-clean-ancilla construction (Barenco et al. 1995, Lemma 7.3)
        // decompose the gate into two halves and add 2 qubits, target and ancilla
        let nc = num_controls as u32;
        let num_qubits = nc + 2;
        let q_ancilla = num_qubits - 1;
        let q_target = num_qubits - 2;
        let middle = (nc + 1).div_ceil(2);
        let nc2: u32 = nc - middle + 1; // second half, plus the ancilla

        // Qubit layout (num_controls + 2 qubits total):
        //   nc = num_controls
        //   [0 .. middle-1]   controls1 (first half)
        //   [middle .. nc-1]  controls2 (second half)
        //   [nc]              target
        //   [nc+1]            ancilla (clean)
        //
        // mcx1: controls1 -> ancilla , up to relative phase
        //   drives the ancilla using the first-half controls;
        //   borrows second-half control qubits as dirty ancillas.
        //
        // mcx2: controls2, ancilla -> target, exact
        //   drives the target using second-half controls + ancilla;
        //   borrows first-half control qubits as dirty ancillas.

        let mcx1 = synth_mcx_n_dirty_i15(middle as usize, true, false)?;
        let mcx2 = synth_mcx_n_dirty_i15(nc2 as usize, false, false)?;

        let num_dirty1 = (mcx1.num_qubits() as u32) - middle - 1;
        let qubits1: Vec<Qubit> = (0..middle)
            .chain([q_ancilla])
            .chain(middle..middle + num_dirty1)
            .map(Qubit)
            .collect();

        let num_dirty2 = (mcx2.num_qubits() as u32) - nc2 - 1;
        let qubits2: Vec<Qubit> = (middle..nc)
            .chain([q_ancilla, q_target])
            .chain(0..num_dirty2)
            .map(Qubit)
            .collect();

        // Compose pattern: mcx1 · mcx2 · mcx1† · mcx2  (Lemma 7.3 [1], Lemma 9 [2]).
        // mcx1/mcx1† are synthesized up to relative phase (Lemma 7 [2]): the relative
        // phase commutes with mcx2 and cancels between mcx1 and mcx1†. mcx2 must be exact.
        let mut circuit: CircuitData =
            CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;
        let mcx1_inv = mcx1.inverse()?;
        circuit.compose(&mcx1, &qubits1, &[])?;
        circuit.compose(&mcx2, &qubits2, &[])?;
        circuit.compose(&mcx1_inv, &qubits1, &[])?;
        circuit.compose(&mcx2, &qubits2, &[])?;

        Ok(circuit)
    }
}

/// The short relative-phase Toffoli ``RTS`` from Maslov (2016), Figure 3, gates 2--6.
fn rts() -> Result<CircuitData, CircuitDataError> {
    let mut circuit = CircuitData::with_capacity(3, 0, 5, Param::Float(0.0))?;
    add_action_gadget(&mut circuit, 1, 0, 2)?;
    Ok(circuit)
}

/// The short special-form relative-phase Toffoli ``SRTS`` from Maslov (2016),
/// circuit (3), dashed.
fn srts() -> Result<CircuitData, CircuitDataError> {
    let mut circuit = CircuitData::with_capacity(3, 0, 9, Param::Float(0.0))?;
    circuit.h(2)?;
    circuit.cx(2, 1)?;
    circuit.tdg(1)?;
    circuit.cx(0, 1)?;
    circuit.t(1)?;
    circuit.cx(2, 1)?;
    circuit.tdg(1)?;
    circuit.cx(0, 1)?;
    circuit.t(1)?;
    Ok(circuit)
}

/// The short relative-phase four-qubit Toffoli ``RT4S`` from Maslov (2016),
/// Figure 4, dashed.
fn rt4s() -> Result<CircuitData, CircuitDataError> {
    let mut circuit = CircuitData::with_capacity(4, 0, 10, Param::Float(0.0))?;
    circuit.h(3)?;
    circuit.t(3)?;
    circuit.cx(2, 3)?;
    circuit.tdg(3)?;
    circuit.h(3)?;
    circuit.cx(0, 3)?;
    circuit.t(3)?;
    circuit.cx(1, 3)?;
    circuit.tdg(3)?;
    circuit.cx(0, 3)?;
    Ok(circuit)
}

/// Synthesize a multi-controlled X gate with :math:`k\ge 3` controls, following
/// Proposition 4 of Maslov (2016) [1].
///
/// The method uses :math:`\lceil(k - 2) / 2\rceil` clean ancillary qubits and produces a
/// circuit with :math:`k + 1 + \lceil(k - 2) / 2\rceil` qubits, :math:`8 * k - 9` T gates,
/// and :math:`6 * k - 6` CX gates.
/// For :math:`k\le 2`, the returned circuit consists of a single X, CX or CCX gate
/// (corresponding to :math:`k = 0, 1, 2`, respectively) and uses no ancillary qubits.
///
/// # Arguments
/// - num_controls: the number of control qubits.
///
/// # Returns
/// The synthesized quantum circuit.
///
/// # References
///
/// 1. D. Maslov, Phys. Rev. A 93, 022311 (2016), "Advantages of using
///    relative-phase Toffoli gates with an application to multiple control Toffoli optimization",
///    [arXiv:1508.03273](https://arxiv.org/abs/1508.03273).
pub fn synth_mcx_n_clean_m15(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 2 {
        synth_mcx_explicit(num_controls)
    } else {
        let num_ancillas = (num_controls - 1) / 2;
        let num_qubits = num_controls + 1 + num_ancillas;
        let num_instructions = 18 * num_controls - 21;
        let mut circuit =
            CircuitData::with_capacity(num_qubits as u32, 0, num_instructions, Param::Float(0.0))?;
        let target = num_controls as u32;
        let first_ancilla = (num_controls + 1) as u32;
        let rccx_gate = rccx();
        let rc3x_gate = rc3x();
        let rc3x_inverse = rc3x_gate.inverse()?;

        // Proposition 4 starts with RT3 for an odd number of controls and RT4 for an
        // even number.
        let first_new_control = if num_controls % 2 == 1 {
            circuit.compose(&rccx_gate, &[Qubit(0), Qubit(1), Qubit(first_ancilla)], &[])?;
            2
        } else {
            circuit.compose(
                &rc3x_gate,
                &[Qubit(0), Qubit(1), Qubit(2), Qubit(first_ancilla)],
                &[],
            )?;
            3
        };

        // Each subsequent RT4 absorbs two more controls into one clean ancilla.
        for i in 1..num_ancillas {
            let control = first_new_control + 2 * (i - 1);
            circuit.compose(
                &rc3x_gate,
                &[
                    Qubit((num_controls + i) as u32),
                    Qubit(control as u32),
                    Qubit((control + 1) as u32),
                    Qubit((num_controls + 1 + i) as u32),
                ],
                &[],
            )?;
        }

        let final_control = first_new_control + 2 * (num_ancillas - 1);
        debug_assert_eq!(final_control, num_controls - 1);
        circuit.ccx(
            (num_controls + num_ancillas) as u32,
            final_control as u32,
            target,
        )?;

        for i in (1..num_ancillas).rev() {
            let control = first_new_control + 2 * (i - 1);
            circuit.compose(
                &rc3x_inverse,
                &[
                    Qubit((num_controls + i) as u32),
                    Qubit(control as u32),
                    Qubit((control + 1) as u32),
                    Qubit((num_controls + 1 + i) as u32),
                ],
                &[],
            )?;
        }

        if num_controls % 2 == 1 {
            // Inverse RCCX (RCCX is self-inverse).
            circuit.compose(&rccx_gate, &[Qubit(0), Qubit(1), Qubit(first_ancilla)], &[])?;
        } else {
            circuit.compose(
                &rc3x_inverse,
                &[Qubit(0), Qubit(1), Qubit(2), Qubit(first_ancilla)],
                &[],
            )?;
        }

        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate using dirty ancillary qubits, following
/// circuit (5) for three controls and Proposition 5 for four or more controls in
/// Maslov (2016).
///
/// For three controls the construction uses one dirty ancilla and has 16 T gates and 14 CX gates.
/// For :math:`k \ge 4` controls it uses :math:`\lceil(k - 2) / 2\rceil` dirty ancillas and has
/// :math:`8k - 8` T gates and :math:`8k - 12` CX gates.
///
/// # References
///
/// 1. D. Maslov, *Advantages of using relative-phase Toffoli gates with an application to
///    multiple control Toffoli optimization*, Phys. Rev. A 93, 022311 (2016),
///    [arXiv:1508.03273](https://arxiv.org/abs/1508.03273).
pub fn synth_mcx_n_dirty_m15(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    if num_controls <= 2 {
        return synth_mcx_explicit(num_controls);
    }

    let srts_gate = srts()?;
    let srts_inverse = srts_gate.inverse()?;
    let rtl_gate = rccx();

    if num_controls == 3 {
        let mut circuit = CircuitData::with_capacity(5, 0, 36, Param::Float(0.0))?;
        circuit.compose(&srts_gate, &[Qubit(2), Qubit(4), Qubit(3)], &[])?;
        circuit.compose(&rtl_gate, &[Qubit(0), Qubit(1), Qubit(4)], &[])?;
        circuit.compose(&srts_inverse, &[Qubit(2), Qubit(4), Qubit(3)], &[])?;
        // Inverse RCCX (RCCX is self-inverse).
        circuit.compose(&rtl_gate, &[Qubit(0), Qubit(1), Qubit(4)], &[])?;
        return Ok(circuit);
    }

    // Maslov numbers the n-1 controls 1..n-1, the n-3 initial ancillas n..2n-4,
    // and the target 2n-3. The RT4 replacements eliminate ancillas n, n+2, ...;
    // compact the remaining paper indices into Qiskit's controls-target-ancillas order.
    let n = num_controls + 1;
    let map_qubit_idx = |original: usize| {
        let compact = if original < n {
            original - 1
        } else if original == 2 * n - 3 {
            num_controls
        } else {
            num_controls + 1 + (original - n) / 2
        };
        Qubit(compact as u32)
    };

    let num_qubits = num_controls + 1 + (num_controls - 1) / 2;
    let mut circuit = CircuitData::with_capacity(num_qubits as u32, 0, 0, Param::Float(0.0))?;
    let rts_gate = rts()?;
    let rts_inverse = rts_gate.inverse()?;
    let rt4s_gate = rt4s()?;
    let rt4s_inverse = rt4s_gate.inverse()?;
    let rt4l_gate = rc3x();
    let rt4l_inverse = rt4l_gate.inverse()?;

    circuit.compose(
        &srts_gate,
        &[
            map_qubit_idx(n - 1),
            map_qubit_idx(2 * n - 4),
            map_qubit_idx(2 * n - 3),
        ],
        &[],
    )?;

    for forward_pass in [true, false] {
        // Items 2 and 6 of Proposition 5, with adjacent RTS pairs replaced by RT4S.
        let first_unpaired = if n.is_multiple_of(2) {
            circuit.compose(
                &rts_gate,
                &[
                    map_qubit_idx(2 * n - 5),
                    map_qubit_idx(n - 2),
                    map_qubit_idx(2 * n - 4),
                ],
                &[],
            )?;
            2
        } else {
            1
        };
        for k in (first_unpaired..n - 4).step_by(2) {
            circuit.compose(
                &rt4s_gate,
                &[
                    map_qubit_idx(2 * n - 5 - k),
                    map_qubit_idx(n - 2 - k),
                    map_qubit_idx(n - 1 - k),
                    map_qubit_idx(2 * n - 3 - k),
                ],
                &[],
            )?;
        }

        circuit.compose(
            if forward_pass {
                &rt4l_gate
            } else {
                &rt4l_inverse
            },
            &[
                map_qubit_idx(1),
                map_qubit_idx(2),
                map_qubit_idx(3),
                map_qubit_idx(n + 1),
            ],
            &[],
        )?;

        // Items 4 and 8, using inverses of the same short relative-phase blocks.
        for k in (2..n - 4).step_by(2) {
            circuit.compose(
                &rt4s_inverse,
                &[
                    map_qubit_idx(n - 1 + k),
                    map_qubit_idx(k + 2),
                    map_qubit_idx(k + 3),
                    map_qubit_idx(n + k + 1),
                ],
                &[],
            )?;
        }
        if n.is_multiple_of(2) {
            let k = n - 4;
            circuit.compose(
                &rts_inverse,
                &[
                    map_qubit_idx(n - 1 + k),
                    map_qubit_idx(k + 2),
                    map_qubit_idx(n + k),
                ],
                &[],
            )?;
        }

        if forward_pass {
            circuit.compose(
                &srts_inverse,
                &[
                    map_qubit_idx(n - 1),
                    map_qubit_idx(2 * n - 4),
                    map_qubit_idx(2 * n - 3),
                ],
                &[],
            )?;
        }
    }

    Ok(circuit)
}

// The following synth_mcx_noaux_hp24 algorithm is based on the work by Huang and Palsberg.
//
// # References
//
// 1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using Helper Qubits*, PLDI (2024),
// https://dl.acm.org/doi/10.1145/3656436.
// 2. The supplementary material for [1] that can be downloaded from the link above.
// 3. Python implementation, available at https://github.com/Keli-Huang/Qulin_Large_Toffoli

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// The construction appears in Fig. 18 in the supplementary material [2].
///
/// Best suitable when n is large.
fn increment_n_dirty_large(n: u32) -> PyResult<CircuitData> {
    // U_x^3-gate from Fig. 22 in [2].
    fn ux(circuit: &mut CircuitData, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        circuit.cx(q1, q3)?;
        circuit.cx(q1, q2)?;
        circuit.ccx(q2, q3, q1)?;
        Ok(())
    }

    // U_z^3-gate from Fig. 24 in [2].
    fn uz(circuit: &mut CircuitData, q1: u32, q2: u32, q3: u32) -> PyResult<()> {
        circuit.ccx(q2, q3, q1)?;
        circuit.cx(q1, q2)?;
        circuit.cx(q2, q3)?;
        Ok(())
    }

    let mut circuit = CircuitData::with_capacity(2 * n, 0, 0, Param::Float(0.0))?;
    let qubits: Vec<u32> = (0..n).collect();
    let ancillas: Vec<u32> = (n..2 * n).collect();

    circuit.x(ancillas[0])?;
    for q in qubits.iter() {
        circuit.cx(ancillas[0], *q)?;
    }
    circuit.x(ancillas[0])?;

    // This implements U^{n}_{z+y+x} in Fig.19 and Fig.23 from the supplementary material for [1].
    for i in 0..n - 1 {
        ux(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }

    circuit.cx(ancillas[0], qubits[(n - 1) as usize])?;
    for i in (0..n - 1).rev() {
        uz(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }

    for i in 0..n - 1 {
        circuit.x(ancillas[(i + 1) as usize])?;
    }

    // This implements U^{n}_{z+y+x} in Fig.19 and Fig.23 from the supplementary material for [1].
    for i in 0..n - 1 {
        ux(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }
    circuit.cx(ancillas[0], qubits[(n - 1) as usize])?;
    for i in (0..n - 1).rev() {
        uz(
            &mut circuit,
            ancillas[0],
            ancillas[(i + 1) as usize],
            qubits[i as usize],
        )?;
    }
    for i in 0..n - 1 {
        circuit.x(ancillas[(i + 1) as usize])?;
    }

    circuit.x(qubits[(n - 1) as usize])?;
    circuit.x(ancillas[0])?;
    for q in qubits.iter() {
        circuit.cx(ancillas[0], *q)?;
    }
    circuit.x(ancillas[0])?;

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// The construction appears in Fig. 10 in the main paper [1].
///
/// Best suitable for when n is small.
fn increment_n_dirty_small(n: u32) -> PyResult<CircuitData> {
    let mut circuit = CircuitData::with_capacity(2 * n, 0, 0, Param::Float(0.0))?;

    for k in (1..n).rev() {
        let k_mcx = synth_mcx_n_dirty_i15(k as usize, false, false)?;
        let k_mcx_qubits: Vec<Qubit> = (0..k + 1).chain(n + 1..2 * n).map(Qubit).collect();
        circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;
    }
    circuit.x(0)?;

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment gate using :math:`n` dirty ancilla qubits.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment gate.
///
/// # Returns
///
/// A quantum circuit with :math:`2 * n` qubits.
fn increment_n_dirty(n: u32) -> PyResult<CircuitData> {
    if n <= 10 {
        increment_n_dirty_small(n)
    } else {
        increment_n_dirty_large(n)
    }
}

/// Synthesize a relative MCX gate without any auxiliary qubits.
///
/// The construction appears as Fig. 10 in the main paper [1].
///
/// Best suitable for when `num_controls` is small.
fn synth_relative_mcx(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    let num_qubits = (num_controls + 1) as u32;
    let target = num_controls as u32;
    let mut circuit = CircuitData::with_capacity(num_qubits, 0, 0, Param::Float(0.0))?;

    match num_controls {
        0 => {
            panic!("synth_relative_mcx requires at least 1 control qubit.");
        }
        1 => {
            circuit.cx(0, 1)?;
        }
        2 => {
            circuit.rccx(0, 1, 2)?;
        }
        3.. => {
            // splits the control qubits into 3 blocks of approximately equal size
            let num3 = num_controls / 3;
            let num2 = (num_controls - num3) / 2;
            let num1 = num_controls - num3 - num2;

            let qubits1: Vec<Qubit> = (0..num1 as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();
            let qubits2: Vec<Qubit> = ((num1 as u32)..(num1 + num2) as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();
            let qubits3: Vec<Qubit> = ((num1 + num2) as u32..num_controls as u32)
                .chain(std::iter::once(target))
                .map(Qubit)
                .collect();

            let circuit1 = synth_relative_mcx(num1)?;
            let circuit2 = synth_relative_mcx(num2)?;
            let circuit3 = synth_relative_mcx(num3)?;

            circuit.h(target)?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit2, &qubits2, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit1, &qubits1, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit2, &qubits2, &[])?;
            circuit.p(PI / 8., target)?;
            circuit.compose(&circuit3, &qubits3, &[])?;
            circuit.p(-PI / 8., target)?;
            circuit.compose(&circuit1, &qubits1, &[])?;
            circuit.h(target)?;
        }
    }
    Ok(circuit)
}

/// Synthesize a relative MCX gate using up to `num_controls` dirty ancilla qubits.
fn synth_relative_mcx_n_dirty(num_controls: usize) -> Result<CircuitData, CircuitDataError> {
    // For small values of num_controls, it is more efficient to use a relative MCX
    // gate that does not require any auxiliary qubits, while for large values it is
    // not efficient to construct the true MCX gate that uses num_controls ancillas.
    // An interesting question is whether there are relative-MCX implementations that
    // use ancilla qubits.
    if num_controls < 11 {
        synth_relative_mcx(num_controls)
    } else {
        synth_mcx_n_dirty_i15(num_controls, false, false)
    }
}

/// Synthesize the :math:`n`-qubit increment/decrement gate using :math:`1` dirty
/// ancilla qubit for odd values of :math:`n`.
///
/// Returns an error when :math:`n` is even.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment/decrement gate.
/// - flag_add: whether to increment by 1 or to decrement by 1.
///
/// # Returns
///
/// A quantum circuit with :math:`n+1` qubits.
fn increment_1_dirty(n: u32, flag_add: bool) -> PyResult<CircuitData> {
    if n.is_multiple_of(2) {
        return Err(QiskitError::new_err(
            "increment_1_dirty_large requires an odd number of qubits.",
        ));
    }

    let k = n.div_ceil(2);

    // This construction is described in Fig. 6 of [1].
    //
    // For n = 2k-1 odd, it reduces an n-increment gate with 1 ancilla to
    // * three k-increment gates that can use up to k ancillas, and
    // * two relative-MCX(k) gates that can use up to k-1 ancillas.

    // We say that qubits 0..k are "first half" qubits, qubits k+1..n are "second-half" qubits.
    let ancilla = n;

    let mut circuit = CircuitData::with_capacity(n + 1, 0, 0, Param::Float(0.0))?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    let k_incrementer = increment_n_dirty(k)?;

    // The first two instances of k_incrementer are defined over ancilla + "second-half" qubits,
    // and use "first-half" qubits as own ancillas
    let k12_incrementer_qubits: Vec<Qubit> = std::iter::once(ancilla)
        .chain(k..n)
        .chain(0..k)
        .map(Qubit)
        .collect();

    circuit.compose(&k_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla)?;

    for q in k..n {
        circuit.cx(ancilla, q)?;
    }

    let k_mcx = synth_relative_mcx_n_dirty(k as usize)?;

    // The two instances of the relative MCX gate have "first half" qubits as controls, ancilla as target, and
    // "second-half qubits" as own ancillas
    let k_mcx_qubits: Vec<Qubit> = (0..k)
        .chain(std::iter::once(ancilla))
        .chain(k..n)
        .map(Qubit)
        .collect();
    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    circuit.compose(&k_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla)?;

    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    for q in k..n {
        circuit.cx(ancilla, q)?;
    }

    let k3_incrementer = increment_n_dirty(k)?;

    // The instance of k3_incrementer is defined over "first-half" qubits,
    // and use "first-half" qubits as own ancillas
    let k3_incrementer_qubits: Vec<Qubit> = (0..k)
        .chain(k..n)
        .chain(std::iter::once(ancilla))
        .map(Qubit)
        .collect();

    circuit.compose(&k3_incrementer, &k3_incrementer_qubits, &[])?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    Ok(circuit)
}

/// Synthesize the :math:`n`-qubit increment/decrement gate using :math:`2` dirty
/// ancilla qubit both for odd and even values of :math:`n`.
///
/// Returns an error when :math:`n` is even.
///
/// # Arguments
/// - :math:`n`: the number of qubits in the increment/decrement gate.
/// - flag_add: whether to increment by 1 or to decrement by 1.
///
/// # Returns
///
/// A quantum circuit with :math:`n+2` qubits.
fn increment_2_dirty(n: u32, flag_add: bool) -> PyResult<CircuitData> {
    let k = (n + 2) / 2; // same as (n+1)/2 for n odd, and (n+2)/2 for n even

    // This construction is a slight modification of the construction described
    // in Fig. 6 of [1].
    //
    // For n = 2k-1 odd, it reduces an n-increment gate with 2 ancillas to
    // * three k-increment gates with k ancillas, and
    // * two relative-MCX(k) gates with k-1 ancillas;
    //
    // For n = 2k-2 even, it reduces an n-increment gate with 2 ancillas to
    // * two (k-1)-increment gates with k ancillas,
    // * one k-increment gate with k-1 ancillas, and
    // * two relative-MCX gates with k-1 ancillas.

    // We say that qubits 0..k are "first half" qubits, qubits k+1..n are "second-half" qubits.
    let ancilla1 = n;
    let ancilla2 = n + 1;

    let mut circuit = CircuitData::with_capacity(n + 2, 0, 0, Param::Float(0.0))?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    let k12_incrementer = increment_n_dirty(1 + n - k)?;

    // The two instances of k12_incrementer are defined over ancilla1 + "second-half" qubits,
    // and use "first-half" qubits + ancilla2 as own ancillas
    let k12_incrementer_qubits: Vec<Qubit> = std::iter::once(ancilla1)
        .chain(k..n)
        .chain(0..k)
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();

    circuit.compose(&k12_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla1)?;

    for q in k..n {
        circuit.cx(ancilla1, q)?;
    }

    let k_mcx = synth_relative_mcx_n_dirty(k as usize)?;

    // The two instances of the relative MCX gate have "first half" qubits as controls, ancilla1 as target, and
    // "second-half qubits" + ancilla2 as own ancillas
    let k_mcx_qubits: Vec<Qubit> = (0..k)
        .chain(std::iter::once(ancilla1))
        .chain(k..n)
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();
    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    circuit.compose(&k12_incrementer, &k12_incrementer_qubits, &[])?;

    circuit.x(ancilla1)?;

    circuit.compose(&k_mcx, &k_mcx_qubits, &[])?;

    for q in k..n {
        circuit.cx(ancilla1, q)?;
    }

    let k3_incrementer = increment_n_dirty(k)?;

    // The instance of k3_incrementer is defined over "first-half" qubits,
    // and use "first-half" qubits + ancilla1 + ancilla2 as own ancillas
    let k3_incrementer_qubits: Vec<Qubit> = (0..k)
        .chain(k..n)
        .chain(std::iter::once(ancilla1))
        .chain(std::iter::once(ancilla2))
        .map(Qubit)
        .collect();

    circuit.compose(&k3_incrementer, &k3_incrementer_qubits, &[])?;

    if !flag_add {
        for i in 0..n {
            circuit.x(i)?;
        }
    }

    Ok(circuit)
}

/// Synthesize a multi-controlled X gate with :math:`k` controls based on the paper
/// by Huang and Palsberg [1].
///
/// # Arguments
///
/// - num_controls: the number of control qubits.
///
/// # Returns
///
/// A quantum circuit with :math:`k + 1` qubits.
/// The number of CX-gates is linear in :math:`k`.
///
/// # References:
///
/// 1. Huang and Palsberg, *Compiling Conditional Quantum Gates without Using Helper Qubits*, PLDI (2024),
///    https://dl.acm.org/doi/10.1145/3656436.
pub fn synth_mcx_noaux_hp24(num_controls: usize) -> PyResult<CircuitData> {
    // Handle small cases explicitly.
    if num_controls <= 1 {
        synth_mcx_explicit(num_controls).map_err(Into::into)
    } else {
        let n: usize = num_controls + 1;
        let mut circuit = CircuitData::with_capacity(n as u32, 0, 0, Param::Float(0.0))?;
        circuit.h(num_controls as u32)?;

        // The construction described in Fig.7 of the paper only works for even values of n.
        // The construction described in Fig.8 of the paper works for all values of n and is better than the one
        // in Fig.7 when n<23.

        if n.is_multiple_of(2) && (n >= 23) {
            // This implements C^{n-1}(V) in Fig.7.

            // These implement U^{n-1}_{+1} and U^{n-1}_{-1} (last qubit is ancilla)
            let increment_plus_1 = increment_1_dirty((n - 1) as u32, true)?;
            let increment_minus_1 = increment_1_dirty((n - 1) as u32, false)?;
            let increment_qubits: Vec<Qubit> = (0..n).map(|q| Qubit(q as u32)).collect();

            circuit.compose(&increment_plus_1, &increment_qubits, &[])?;

            // This implements U^{n-1}_{V_{n-1}}(-1)
            let mut phi = -PI;
            for q in (1..n - 1).rev() {
                phi /= 2.;
                circuit.cp(phi, q as u32, (n - 1) as u32)?;
            }

            circuit.compose(&increment_minus_1, &increment_qubits, &[])?;

            // This implements U^{n-1}_{V_{n-1}}(-1)
            let mut phi = PI;
            for q in (1..n - 1).rev() {
                phi /= 2.;
                circuit.cp(phi, q as u32, (n - 1) as u32)?;
            }

            circuit.cp(phi, 0, (n - 1) as u32)?;
        } else {
            // This implements C^{n-1}(V) in Fig.8.

            // These implement U^{n-1}_{+1} and U^{n-1}_{-1} (last qubit is ancilla)
            let increment_plus_1 = increment_2_dirty((n - 2) as u32, true)?;
            let increment_minus_1 = increment_2_dirty((n - 2) as u32, false)?;
            let increment_qubits: Vec<Qubit> = (0..n).map(|q| Qubit(q as u32)).collect();

            circuit.compose(&increment_plus_1, &increment_qubits, &[])?;

            // This implements C(U^{n-2}_{V_{n-2}}(-1)) in Fig.8.
            let mut phi = -PI;
            for q in (1..n - 2).rev() {
                phi /= 2.;
                circuit.ccp(phi, q as u32, (n - 2) as u32, (n - 1) as u32)?;
            }

            circuit.compose(&increment_minus_1, &increment_qubits, &[])?;

            // This implements C(U^{n-2}_{V_{n-2}}(1)) in Fig.8.
            let mut phi = PI;
            for q in (1..n - 2).rev() {
                phi /= 2.;
                circuit.ccp(phi, q as u32, (n - 2) as u32, (n - 1) as u32)?;
            }

            circuit.ccp(phi, 0, (n - 2) as u32, (n - 1) as u32)?;
        }

        circuit.h(num_controls as u32)?;
        Ok(circuit)
    }
}

/// Synthesize a multi-controlled X gate with :math:`k` controls using the relation
/// MCX = H · MCP(π) · H, where MCP is synthesized via `synth_mcp_noaux_sp22` [1][2].
///
/// Produces a quantum circuit with :math:`k + 1` qubits.
/// The number of CX-gates is quadratic in :math:`k`.
///
/// # Arguments
///
/// - num_ctrl_qubits: The number of control qubits.
///
/// # Returns
///
/// The synthesized quantum circuit.
///
/// # References
/// 1. A. J. da Silva and D. K. Park, *Linear-depth quantum circuits for multiqubit controlled gates*,
///    [Phys. Rev. A 106, 042602](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602).
///
/// 2. <https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py>
pub fn synth_mcx_noaux_sp22(num_ctrl_qubits: usize) -> Result<CircuitData, CircuitDataError> {
    if num_ctrl_qubits <= 4 {
        synth_mcx_explicit(num_ctrl_qubits)
    } else {
        // 2n^2-2n+1 instructions from synth_mcp_noaux_sp22, plus 2 H gates wrapping it.
        let num_instructions = 2 * num_ctrl_qubits * num_ctrl_qubits - 2 * num_ctrl_qubits + 3;
        let mut circuit = CircuitData::with_capacity(
            (num_ctrl_qubits + 1) as u32,
            0,
            num_instructions,
            Param::Float(0.0),
        )?;
        circuit.h(num_ctrl_qubits as u32)?;
        let pi_phase = Param::Float(PI);
        // Adding MCP(π) on `circuit` which has target qubit at index `num_ctrl_qubits`.
        // Instead of calling `synth_mcp_noaux_sp22`, we call its step functions directly. This avoids
        // allocating and copying an intermediate circuit and is ~2x faster.
        sp22::step_1(&mut circuit, &pi_phase, num_ctrl_qubits)?;
        sp22::step_2(&mut circuit, &pi_phase, num_ctrl_qubits)?;
        sp22::step_3(&mut circuit, num_ctrl_qubits)?;
        sp22::step_4(&mut circuit, num_ctrl_qubits)?;
        circuit.h(num_ctrl_qubits as u32)?;
        Ok(circuit)
    }
}

/// Synthesize multi-controlled X explicit gates with up to 4 control qubits.
///
/// # Arguments
///
/// - num_ctrl_qubits: The number of control qubits.
///
/// Panics if called with more than 4 control qubits.
///
/// # Returns
///
/// The synthesized quantum circuit.
pub fn synth_mcx_explicit(num_ctrl_qubits: usize) -> Result<CircuitData, CircuitDataError> {
    assert!(
        num_ctrl_qubits <= 4,
        "synth_mcx_explicit called with num_ctrl_qubits = {num_ctrl_qubits}, expected <= 4"
    );
    match num_ctrl_qubits {
        0 => {
            let mut circuit = CircuitData::with_capacity(1, 0, 1, Param::Float(0.0))?;
            circuit.x(0)?;
            Ok(circuit)
        }
        1 => {
            let mut circuit = CircuitData::with_capacity(2, 0, 1, Param::Float(0.0))?;
            circuit.cx(0, 1)?;
            Ok(circuit)
        }
        2 => Ok(ccx()),
        3 => Ok(c3x()),
        4 => Ok(c4x()),
        _ => unreachable!(),
    }
}

// The following code (synth_mcp_noaux_sp22 and mod sp22) is a derivative work of qclib
// (https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py).
// Copyright 2021 qclib project. Licensed under the Apache License, Version 2.0.

/// Synthesize a multi-controlled phase gate with :math:`n` controls based on the paper
/// by da Silva et al. [1] and the implementation in qclib [2].
///
/// For :math:`n \ge 2`, the method produces a circuit with :math:`4n^2-4n+2` CX gates
/// and requires :math:`O(n)` depth.
///
/// # Arguments
///
/// - num_ctrl_qubits: the number of control qubits.
/// - phase: the phase angle for the multi-controlled phase gate.
///
/// # Returns
///
/// A quantum circuit implementing the multi-controlled phase gate with :math:`4n^2-4n+2` CX gates.
///
/// # References
///
/// 1. A. J. da Silva and D. K. Park, *Linear-depth quantum circuits for multiqubit controlled gates*,
///    [Phys. Rev. A 106, 042602](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602).
///
/// 2. <https://github.com/qclib/qclib/blob/master/qclib/gates/ldmcu.py>
pub fn synth_mcp_noaux_sp22(
    num_ctrl_qubits: usize,
    phase: Param,
) -> Result<CircuitData, CircuitDataError> {
    // For n>1 steps 1-4 emit 2n-1 CP gates and 2(n-1)^2 CRX gates, i.e. 2n^2-2n+1 instructions
    // in total (later each of which decomposes into two CX gates, giving total of 4n^2-4n+2 of the
    // doc comment).
    let num_instructions = if num_ctrl_qubits < 2 {
        1
    } else {
        2 * num_ctrl_qubits * num_ctrl_qubits - 2 * num_ctrl_qubits + 1
    };
    let mut circuit = CircuitData::with_capacity(
        (num_ctrl_qubits + 1) as u32,
        0,
        num_instructions,
        Param::Float(0.0),
    )?;

    if num_ctrl_qubits == 0 {
        circuit.p(phase, 0)?;
    } else if num_ctrl_qubits == 1 {
        circuit.cp(phase, 0, 1)?;
    } else {
        sp22::step_1(&mut circuit, &phase, num_ctrl_qubits)?;
        sp22::step_2(&mut circuit, &phase, num_ctrl_qubits)?;
        sp22::step_3(&mut circuit, num_ctrl_qubits)?;
        sp22::step_4(&mut circuit, num_ctrl_qubits)?;
    }
    Ok(circuit)
}

mod sp22 {
    use std::f64::consts::PI;

    use qiskit_circuit::circuit_data::{CircuitData, CircuitDataError};
    use qiskit_circuit::operations::{Param, multiply_param};

    use super::CircuitDataForSynthesis;

    /// Returns all `(control, target)` index pairs with `0 <= control < target < n_qubits`,
    /// sorted by `control + target` descending.
    fn pairs_high_sum_first(n_qubits: usize) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = (0..n_qubits)
            .flat_map(|target| (0..target).map(move |control| (control, target)))
            .collect();
        pairs.sort_by_key(|a| std::cmp::Reverse(a.0 + a.1));
        pairs
    }

    /// Returns all `(control, target)` index pairs with `1 <= control < target < n_qubits`,
    /// sorted by `control + target` ascending.
    fn pairs_low_sum_first(n_qubits: usize) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = (1..n_qubits)
            .flat_map(|target| (1..target).map(move |control| (control, target)))
            .collect();
        pairs.sort_by_key(|a| a.0 + a.1);
        pairs
    }

    /// Returns `2^exponent` where `exponent = (target - control) - 1` if `control == 0`,
    /// or `(target - control)` otherwise. This is the divisor of the fractional angle
    /// (`φ / divisor` for CP gates, `π / divisor` for CRX gates) applied to each qubit pair.
    /// Uses `f64::exp2` rather than a bit shift to avoid overflow for `n > 64` controls.
    /// Note that target > control by design, no need to assert for.
    fn angle_divisor(control: usize, target: usize) -> f64 {
        let exponent = if control == 0 {
            target - control - 1
        } else {
            target - control
        };
        (exponent as f64).exp2()
    }

    /// SP22 Step 1 — forward phase accumulation on controls and target.
    ///
    /// Iterates over all pairs `0 <= c < t <= num_ctrl_qubits`, high-sum first.
    /// Emits CP(+φ/divisor, c, t) when `t` is the target qubit, CRX(+π/divisor, c, t) otherwise.
    /// The CP gates accumulate phase on the target; the CRX gates entangle the controls
    /// so that the phases cancel for all input states except |11…1⟩.
    /// Gate count: :math:`n` CP and :math:`n(n-1)/2` CRX.
    pub fn step_1(
        circuit: &mut CircuitData,
        phi: &Param,
        num_ctrl_qubits: usize,
    ) -> Result<(), CircuitDataError> {
        for (control, target) in pairs_high_sum_first(num_ctrl_qubits + 1) {
            let divisor = angle_divisor(control, target);
            if target == num_ctrl_qubits {
                circuit.cp(
                    multiply_param(phi, 1.0 / divisor),
                    control as u32,
                    target as u32,
                )?;
            } else {
                circuit.crx(PI / divisor, control as u32, target as u32)?;
            }
        }
        Ok(())
    }

    /// SP22 Step 2 — reverse sweep on controls and target.
    ///
    /// Iterates over all pairs `1 <= c < t <= num_ctrl_qubits`, low-sum first.
    /// Emits CP(−φ/divisor, c, t) when `t` is the target qubit, CRX(−π/divisor, c, t) otherwise.
    /// Mirrors step 1 in reverse order to complete the phase accumulation pattern.
    /// Gate count: :math:`n-1` CP and :math:`(n-1)(n-2)/2` CRX.
    pub fn step_2(
        circuit: &mut CircuitData,
        phi: &Param,
        num_ctrl_qubits: usize,
    ) -> Result<(), CircuitDataError> {
        for (control, target) in pairs_low_sum_first(num_ctrl_qubits + 1) {
            let divisor = angle_divisor(control, target);
            if target == num_ctrl_qubits {
                circuit.cp(
                    multiply_param(phi, -1.0 / divisor),
                    control as u32,
                    target as u32,
                )?;
            } else {
                circuit.crx(-PI / divisor, control as u32, target as u32)?;
            }
        }
        Ok(())
    }

    /// SP22 Step 3 — uncomputation on controls only, forward order.
    ///
    /// Iterates over all pairs `0 <= c < t < num_ctrl_qubits`, high-sum first.
    /// Emits CRX(−π/divisor, c, t) when `c == 0`, CRX(+π/divisor, c, t) otherwise.
    /// The sign alternation here and in step 4 cancels the CRX entanglement from steps 1 and 2.
    /// Gate count: :math:`n(n-1)/2` CRX.
    pub fn step_3(
        circuit: &mut CircuitData,
        num_ctrl_qubits: usize,
    ) -> Result<(), CircuitDataError> {
        for (control, target) in pairs_high_sum_first(num_ctrl_qubits) {
            let divisor = angle_divisor(control, target);
            let sign = if control == 0 { -1.0 } else { 1.0 };
            circuit.crx(sign * PI / divisor, control as u32, target as u32)?;
        }
        Ok(())
    }

    /// SP22 Step 4 — uncomputation on controls only, reverse order.
    ///
    /// Iterates over all pairs `1 <= c < t < num_ctrl_qubits`, low-sum first.
    /// Emits CRX(−π/divisor, c, t) for all pairs (`c == 0` never occurs in this range).
    /// Mirrors step 3 in reverse order to complete the uncomputation.
    /// Gate count: :math:`(n-1)(n-2)/2` CRX.
    pub fn step_4(
        circuit: &mut CircuitData,
        num_ctrl_qubits: usize,
    ) -> Result<(), CircuitDataError> {
        for (control, target) in pairs_low_sum_first(num_ctrl_qubits) {
            let divisor = angle_divisor(control, target);
            circuit.crx(-PI / divisor, control as u32, target as u32)?;
        }
        Ok(())
    }
}

#[cfg(all(test, not(miri)))]
mod test {
    use crate::matrix::sim::sim_unitary_circuit;
    use approx::abs_diff_eq;
    use qiskit_circuit::circuit_data::CircuitData;

    use super::{
        increment_n_dirty_large, increment_n_dirty_small, synth_mcx_n_clean_m15,
        synth_mcx_n_dirty_m15,
    };

    fn assert_mcx(circuit: &CircuitData, num_controls: usize, clean_ancillas: bool) {
        let unitary = sim_unitary_circuit(circuit).unwrap();
        let control_mask = (1 << num_controls) - 1;
        let target_mask = 1 << num_controls;
        let columns = if clean_ancillas {
            1 << (num_controls + 1)
        } else {
            unitary.ncols()
        };

        for column in 0..columns {
            let expected_row = if column & control_mask == control_mask {
                column ^ target_mask
            } else {
                column
            };
            for row in 0..unitary.nrows() {
                let expected = if row == expected_row { 1.0 } else { 0.0 };
                assert!(abs_diff_eq!(
                    unitary[(row, column)].re,
                    expected,
                    epsilon = 1e-10
                ));
                assert!(abs_diff_eq!(
                    unitary[(row, column)].im,
                    0.0,
                    epsilon = 1e-10
                ));
            }
        }
    }

    #[test]
    fn test_synth_mcx_n_clean_m15() {
        // Exercise odd and even control counts, including an RT4 ladder with two ancillas.
        for num_controls in 3..=6 {
            let circuit = synth_mcx_n_clean_m15(num_controls).unwrap();
            assert_mcx(&circuit, num_controls, true);
        }
    }

    #[test]
    fn test_synth_mcx_n_dirty_m15() {
        // Exercise both the three-control special case and the general RT4 construction.
        for num_controls in 3..=4 {
            let circuit = synth_mcx_n_dirty_m15(num_controls).unwrap();
            assert_mcx(&circuit, num_controls, false);
        }
    }

    #[test]
    fn test_increment_n_dirty() {
        // Check that both methods to implement the :math:`n`-qubit increment gate using
        // :math:`n` dirty ancilla qubits produce the same matrix (for small number of qubits).
        for nq in 1..4 {
            let circuit1 = increment_n_dirty_small(nq).unwrap();
            let mat1 = sim_unitary_circuit(&circuit1).unwrap();

            let circuit2 = increment_n_dirty_large(nq).unwrap();
            let mat2 = sim_unitary_circuit(&circuit2).unwrap();

            const EPS: f64 = 1e-10;
            let close = abs_diff_eq!(mat1, mat2, epsilon = EPS);
            assert!(close);
        }
    }
}
