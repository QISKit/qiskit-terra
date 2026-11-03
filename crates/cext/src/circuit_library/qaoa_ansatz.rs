// This code is part of Qiskit.
//
// (C) Copyright IBM 2025
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

//! QAOA ansatz circuit construction.
//!
//! This module builds a QAOA circuit as [`CircuitData`] for a cost Hamiltonian expressed as a
//! [`SparseObservable`].
//!
//! The circuit is constructed as:
//! 1. An initial state preparation (default: `H` on all qubits, i.e. `|+>^{⊗n}`),
//! 2. `reps` alternating applications of:
//!    - cost evolution `exp(-i * γ_k * C)`
//!    - mixer evolution `exp(-i * β_k * B)`
//!
//! Cost and (optional) custom mixer evolutions are compiled term-by-term using
//! [`crate::pauli_evolution::sparse_term_evolution`], which synthesizes a Pauli-string evolution
//! into standard gates (basis change + CX propagation + rotation + uncompute).
//!
//! # Semantics note: term-by-term product decomposition
//! For an observable `H = Σ_k c_k P_k`, this implementation builds the evolution block as a product
//! of term evolutions `Π_k exp(-i * t * c_k P_k)` in the iteration order of [`SparseObservable`].
//! This is exact when all terms commute; otherwise it corresponds to a first-order product-formula
//! decomposition with the chosen ordering.
//!
//! # Scope (v1)
//! - Supports Pauli terms (X/Y/Z) with real coefficients.
//! - Rejects projector bit-terms (`+/-/r/l/0/1`).
//! - Exposes only a single knob: `insert_barriers`.

use std::sync::Arc;

use num_complex::Complex64;
use smallvec::smallvec;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::operations::{self, Param, StandardGate, multiply_param, radd_param};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::parameter::parameter_expression::ParameterExpression;
use qiskit_circuit::parameter::symbol_expr::Symbol;
use qiskit_circuit::{Clbit, Qubit};

use qiskit_circuit_library::pauli_evolution::{Instruction, sparse_term_evolution};
use qiskit_quantum_info::sparse_observable::{BitTerm, SparseObservable};

/// Errors returned by QAOA circuit construction.
#[derive(Debug)]
pub enum QaoaError {
    /// `reps` must be at least 1 for this builder API.
    RepsMustBePositive,
    /// Mixer qubit count mismatch.
    NumQubitsMismatch { cost: u32, mixer: u32 },
    /// A term has an imaginary coefficient; supports only real coefficients.
    NonRealCoefficient { term_index: usize, coeff: Complex64 },
    /// A term contains a non-Pauli bit-term (e.g., projector symbols).
    UnsupportedBitTerm {
        term_index: usize,
        bit_term: BitTerm,
    },
    /// CircuitData construction failed (internal error).
    CircuitDataBuildFailed { err: String },
}

impl std::fmt::Display for QaoaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QaoaError::RepsMustBePositive => write!(f, "reps must be >= 1"),
            QaoaError::NumQubitsMismatch { cost, mixer } => write!(
                f,
                "mixer num_qubits ({mixer}) does not match cost num_qubits ({cost})."
            ),
            QaoaError::NonRealCoefficient { term_index, coeff } => write!(
                f,
                "term {term_index} has non-real coefficient {coeff:?} (only real coefficient supported.)"
            ),
            QaoaError::UnsupportedBitTerm {
                term_index,
                bit_term,
            } => write!(
                f,
                "term {term_index} contains unsupported BitTerm {bit_term:?} (projectors not supported.)"
            ),
            QaoaError::CircuitDataBuildFailed { err } => {
                write!(f, "failed to build CircuitData: {err}")
            }
        }
    }
}
impl std::error::Error for QaoaError {}

/// Construct a *parameterized* QAOA ansatz circuit (parameters created internally).
///
/// The circuit is returned with symbolic parameters `γ[k]` and `β[k]` for `k = 0..reps-1`.
pub fn qaoa_ansatz(
    cost: &SparseObservable,
    reps: usize,
    insert_barriers: bool,
    mixer: Option<&SparseObservable>,
) -> Result<CircuitData, QaoaError> {
    if reps == 0 {
        return Err(QaoaError::RepsMustBePositive);
    }

    // Validate observables are Pauli-only and coefficients are real.
    validate_pauli_only(cost)?;
    if let Some(m) = mixer {
        validate_pauli_only(m)?;
    }

    let (gammas, betas) = default_layer_parameters(reps);
    build_qaoa_with_params(cost, reps, &gammas, &betas, insert_barriers, mixer)
}

/// Internal builder that constructs the QAOA circuit from supplied per-layer parameters.
fn build_qaoa_with_params(
    cost: &SparseObservable,
    reps: usize,
    gammas: &[Param],
    betas: &[Param],
    insert_barriers: bool,
    mixer: Option<&SparseObservable>, // if None, use dafault X-mixer
) -> Result<CircuitData, QaoaError> {
    let n = cost.num_qubits();

    if let Some(m) = mixer {
        let m_n = m.num_qubits();
        if m_n != n {
            return Err(QaoaError::NumQubitsMismatch {
                cost: n,
                mixer: m_n,
            });
        }
    }

    let empty_clbits: Vec<Clbit> = Vec::new();
    let mut global_phase = Param::Float(0.0);

    let barrier = barrier_instruction(n, &empty_clbits);

    let mut insts: Vec<Instruction> = Vec::new();

    // 1) Initial state: |+>^n (H on all qubits).
    append_default_initial_state(&mut insts, n, &empty_clbits);

    // 2) QAOA layers: [cost(gamma_k), mixer(beta_k)] repeated `reps` times.
    for layer in 0..reps {
        // Cost block: exp(-i gamma_k * cost)
        append_observable_evolution(&mut insts, cost, gammas[layer].clone(), &mut global_phase);

        if insert_barriers {
            insts.push(barrier.clone());
        }

        // Mixer block:
        // - default is Σ_i X_i  -> RX(2*beta_k) on each qubit.
        // - if custom mixer provided, compile exp(-i beta_k * mixer) term-by-term.
        match mixer {
            None => {
                append_default_mixer(&mut insts, n, &betas[layer], &empty_clbits);
            }
            Some(m) => {
                append_observable_evolution(&mut insts, m, betas[layer].clone(), &mut global_phase);
            }
        }

        if insert_barriers && layer + 1 < reps {
            insts.push(barrier.clone());
        }
    }

    let iter = insts.into_iter().map(Ok);

    let circuit = CircuitData::from_packed_operations(n, 0, iter, global_phase)
        .map_err(|e| QaoaError::CircuitDataBuildFailed { err: e.to_string() })?;
    Ok(circuit)
}

/// Create default symbolic parameters `(gammas, betas)` for `reps` layers.
fn default_layer_parameters(reps: usize) -> (Vec<Param>, Vec<Param>) {
    let mut gammas = Vec::with_capacity(reps);
    let mut betas = Vec::with_capacity(reps);

    for k in 0..reps {
        gammas.push(indexed_symbol_param("γ", k));
        betas.push(indexed_symbol_param("β", k));
    }

    (gammas, betas)
}

/// Construct a parameter `prefix[k]` as a [`Param::ParameterExpression`].
#[inline]
fn indexed_symbol_param(prefix: &str, index: usize) -> Param {
    let sym = Symbol::new(prefix, None, Some(index as u32));
    let expr = ParameterExpression::from_symbol(sym);
    Param::ParameterExpression(Arc::new(expr))
}

/// Append the default QAOA initial state `|+>^{⊗n}` (`H` on all qubits).
#[inline]
fn append_default_initial_state(out: &mut Vec<Instruction>, n: u32, empty_clbits: &[Clbit]) {
    out.extend((0..n).map(|q| {
        (
            StandardGate::H.into(),
            smallvec![],
            vec![Qubit(q)],
            empty_clbits.to_vec(),
        )
    }));
}

/// Append the default QAOA mixer block for one layer.
///
/// The default mixer Hamiltonian is `B = Σ_i X_i`. Using the convention
/// `RX(θ) = exp(-i θ/2 X)`, we implement `exp(-i β X)` as `RX(2β)` on each qubit.
#[inline]
fn append_default_mixer(out: &mut Vec<Instruction>, n: u32, beta: &Param, empty_clbits: &[Clbit]) {
    let angle = multiply_param(beta, 2.0);
    out.extend((0..n).map(|q| {
        (
            StandardGate::RX.into(),
            smallvec![angle.clone()],
            vec![Qubit(q)],
            empty_clbits.to_vec(),
        )
    }));
}

/// Append a compiled time-evolution block for `obs` under parameter `t`.
///
/// This compiles `exp(-i * t * obs)` by iterating over all terms `c_k P_k` in `obs` and appending
/// an evolution for each term using [`sparse_term_evolution`].
///
/// # Parameter convention
/// Qiskit’s Pauli-rotation gates satisfy `R_P(θ) = exp(-i θ/2 P)`. Therefore, to implement
/// `exp(-i * t * c_k * P_k)`, we pass `θ = 2 * t * c_k` into the term synthesis.
///
/// # Global phase
/// Identity-only terms contribute only a global phase and do not emit gates. These contributions
/// are accumulated into `global_phase`.
fn append_observable_evolution(
    out: &mut Vec<Instruction>,
    obs: &SparseObservable,
    t: Param,
    global_phase: &mut Param,
) {
    for term in obs.iter() {
        let coeff = term.coeff;

        if coeff.re == 0.0 {
            continue;
        }

        // Rotation angle used by the gates:" for Pauli rotations, Qiskit uses
        // R{P}(θ) = exp(-i θ/2 * P). We want exp(-i t * coeff * P),
        // so set θ = 2 * t * coeff.
        let angle = multiply_param(&t, 2.0 * coeff.re);

        // Identity term: contributes only global phase.
        // We add (-0.5 * angle) = -(t * coeff), matching pauli_evolution.rs convention.
        if term.indices.is_empty() {
            *global_phase = radd_param(global_phase.clone(), multiply_param(&angle, -0.5));
            continue;
        }

        // Convert the term's bit_terms into a Pauli label string like "ZZX".
        // Indices are already the qubit positions for those non-identity operators.
        let pauli: String = term
            .bit_terms
            .iter()
            .map(|bt| match bt {
                BitTerm::X => 'X',
                BitTerm::Y => 'Y',
                BitTerm::Z => 'Z',
                _ => unreachable!("validate_pauli_only ensures only X/Y/Z bit terms"),
            })
            .collect();

        let indices: Vec<u32> = term.indices.to_vec();

        let evo = sparse_term_evolution(
            pauli.as_str(),
            indices,
            angle,
            /*phase_gate_for_paulis=*/ false,
            /*do_fountain=*/ false,
        );
        out.extend(evo);
    }
}

/// Validate that `obs` contains only Pauli X/Y/Z bit-terms with real coefficients.
///
/// # Errors
/// Returns [`QaoaError::NonRealCoefficient`] if any term has an imaginary coefficient, and
/// [`QaoaError::UnsupportedBitTerm`] if any term contains a non-Pauli bit-term (e.g. projector
/// symbols).
fn validate_pauli_only(obs: &SparseObservable) -> Result<(), QaoaError> {
    for (term_index, term) in obs.iter().enumerate() {
        let coeff = term.coeff;
        if coeff.im != 0.0 {
            return Err(QaoaError::NonRealCoefficient { term_index, coeff });
        }
        for &bt in term.bit_terms.iter() {
            match bt {
                BitTerm::X | BitTerm::Y | BitTerm::Z => {}
                other => {
                    return Err(QaoaError::UnsupportedBitTerm {
                        term_index,
                        bit_term: other,
                    });
                }
            }
        }
    }
    Ok(())
}

/// Construct a reusable barrier instruction over all `n` qubits.
#[inline]
fn barrier_instruction(n: u32, empty_clbits: &[Clbit]) -> Instruction {
    (
        PackedOperation::from_standard_instruction(operations::StandardInstruction::Barrier(n)),
        smallvec![],
        (0..n).map(Qubit).collect(),
        empty_clbits.to_vec(),
    )
}

// ===== C-API glue (cext) =====

/// @ingroup QkCircuitLibrary
/// Build a *parameterized* QAOA ansatz circuit for a given cost Hamiltonian.
///
/// This creates symbolic parameters internally (`γ[k]`, `β[k]`) and returns a circuit
/// with unbound parameters.
///
/// @param cost  Pointer to a valid `SparseObservable` cost Hamiltonian. Must be non-NULL.
/// @param reps  Number of QAOA layers (p). Must be >= 1.
/// @param insert_barriers  If true, insert barriers between cost/mixer blocks and between layers.
/// @param mixer  Optional mixer Hamiltonian. May be NULL to use the default `Σ_i X_i`.
///
/// @return A newly allocated `QkCircuit*` (caller must free with `qk_circuit_free`),
///         or NULL on invalid input or if circuit construction fails.
///
/// # Safety
/// - `cost` must be a valid pointer to a `SparseObservable` created by the Rust library
///   and alive for the duration of this call.
/// - If `mixer` is non-NULL, it must also be a valid pointer to a `SparseObservable`.
#[unsafe(no_mangle)]
#[cfg(feature = "cbinding")]
pub unsafe extern "C" fn qk_circuit_library_qaoa_ansatz(
    cost: *const SparseObservable,
    reps: u32,
    insert_barriers: bool,
    mixer: *const SparseObservable,
) -> *mut CircuitData {
    if reps == 0 {
        return std::ptr::null_mut();
    }

    // SAFETY: `cost` is a raw pointer from C. We check for NULL and only borrow it for this call.
    // If non-NULL, it must point to a valid SparseObservable allocated by Rust and alive for this call.
    let cost = match unsafe { cost.as_ref() } {
        Some(c) => c,
        None => return std::ptr::null_mut(),
    };

    // SAFETY: `mixer` may be NULL (maps to None). If non-NULL, it must be valid for this call.
    let mixer = unsafe { mixer.as_ref() }; // Option<&SparseObservable>

    match qaoa_ansatz(cost, reps as usize, insert_barriers, mixer) {
        Ok(circ) => Box::into_raw(Box::new(circ)),
        Err(_err) => std::ptr::null_mut(),
    }
}
