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

use rand::prelude::*;
use rand::rngs::SysRng;
use rand_pcg::Pcg64Mcg;

use qiskit_circuit::circuit_data::CircuitData;
use qiskit_circuit::dag_circuit::DAGCircuit;
use qiskit_circuit::{PhysicalQubit, Qubit};
use qiskit_transpiler::passes::sabre::heuristic;
use qiskit_transpiler::passes::sabre::sabre_layout_and_routing;
use qiskit_transpiler::target::Target;
use qiskit_transpiler::transpile_layout::TranspileLayout;

/// The options for running ``qk_transpiler_pass_standalone_sabre_layout``. This struct is used
/// as an input to control the behavior of the layout and routing algorithms.
#[repr(C)]
pub struct SabreLayoutOptions {
    /// The number of forward-backward iterations in the sabre routing algorithm
    max_iterations: usize,
    /// The number of trials to run of the sabre routing algorithm for each iteration. When > 1 the
    /// trial that routing trial that results in the output with the fewest swap gates will be
    /// selected.
    num_swap_trials: usize,
    /// The number of random layout trials to run. The trial that results in the output with the
    /// fewest swap gates will be selected.
    num_random_trials: usize,
    /// A seed value for the pRNG used internally.
    seed: u64,
}

/// @ingroup QkSabreLayoutOptions
///
/// Build a default sabre layout options object. This builds a sabre layout with ``max_iterations``
/// set to 4, both ``num_swap_trials`` and ``num_random_trials`` set to 20, and the seed selected
/// by a RNG seeded from system entropy.
///
/// @return A ``QkSabreLayoutOptions`` object with default settings.
#[unsafe(no_mangle)]
pub extern "C" fn qk_sabre_layout_options_default() -> SabreLayoutOptions {
    SabreLayoutOptions {
        max_iterations: 4,
        num_swap_trials: 20,
        num_random_trials: 20,
        seed: Pcg64Mcg::try_from_rng(&mut SysRng).unwrap().random(),
    }
}

/// @ingroup QkTranspilerPassesStandalone
/// Run the SabreLayout transpiler pass on a circuit.
///
/// The SabreLayout pass chooses a layout via an iterative bidirectional routing of the input
/// circuit.
///
/// Starting with a random initial Layout, the algorithm does a full routing of the circuit to end up with a final_layout.
/// This final_layout is then used as the initial_layout for routing the reverse circuit. The algorithm iterates a number
/// of times until it finds an initial_layout that reduces full routing cost.
///
/// This method exploits the reversibility of quantum circuits, and tries to include global circuit information in the
/// choice of initial_layout.
///
/// This pass will run both layout and routing and will transform the circuit so that the layout is applied to the input
/// (meaning that the output circuit will have ancilla qubits allocated for unused qubits on the coupling map and the
/// qubits will be reordered to match the mapped physical qubits) and then routing will be applied. This is done because
/// the pass will run parallel seed trials with different random seeds for selecting the random initial layout and then
/// selecting the routed output which results in the least number of swap gates needed. This final
/// swap calculation is the same as performing a final routing, so it's more efficient to apply it
/// after computing it.
///
/// This function is multithreaded and will launch a thread pool with threads equal to the number
/// of CPUs by default. You can tune the number of threads with the ``RAYON_NUM_THREADS``
/// environment variable. For example, setting ``RAYON_NUM_THREADS=4`` would limit the thread pool
/// to 4 threads.
///
/// # References
///
/// [1] Henry Zou and Matthew Treinish and Kevin Hartman and Alexander Ivrii and Jake Lishman.
/// "LightSABRE: A Lightweight and Enhanced SABRE Algorithm"
/// [arXiv:2409.08368](https://doi.org/10.48550/arXiv.2409.08368)
///
/// [2] Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem
/// for NISQ-era quantum devices." ASPLOS 2019.
/// [arXiv:1809.02573](https://arxiv.org/pdf/1809.02573.pdf)
///
/// @param circuit A pointer to the circuit to run SabreLayout on. The circuit
///     is modified in place and the original circuit's allocations are freed by this function.
/// @param target A pointer to the target to run SabreLayout on
/// @param options A pointer to the options for SabreLayout
///
/// @return The transpile layout that describes the layout and output permutation caused
///     by the pass
///
/// # Safety
///
/// Behavior is undefined if ``circuit`` or ``target`` is not a valid, non-null pointer to a ``QkCircuit`` and ``QkTarget``.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qk_transpiler_pass_standalone_sabre_layout(
    circuit: *mut CircuitData,
    target: *const Target,
    options: *const SabreLayoutOptions,
) -> *mut TranspileLayout {
    // SAFETY: Per documentation, the pointer is non-null and aligned.
    let circuit = unsafe { &mut *circuit };
    let target = unsafe { &*target };
    let options = unsafe { &*options };
    let mut dag = DAGCircuit::from_circuit_data(circuit, false, None, None)
        .unwrap_or_else(|_| panic!("Internal circuit to DAG conversion failed."));
    let heuristic = heuristic::Heuristic::new(
        Some(heuristic::BasicHeuristic::new(
            1.0,
            heuristic::SetScaling::Constant,
        )),
        Some(
            heuristic::LookaheadHeuristic::new(
                vec![0.5 / target.num_qubits.unwrap_or(20) as f64],
                heuristic::SetScaling::Constant,
            )
            .expect("number of layers should be valid"),
        ),
        Some(heuristic::DecayHeuristic::new(0.001, 5)),
        Some(10 * target.num_qubits.unwrap() as usize),
        1e-10,
    );
    let (result, initial_layout, final_layout) = sabre_layout_and_routing(
        &mut dag,
        target,
        &heuristic,
        options.max_iterations,
        options.num_swap_trials,
        options.num_random_trials,
        Some(options.seed),
        Vec::new(),
        false,
    )
    .unwrap_or_else(|_| panic!("Sabre layout failed."));
    let out_circuit =
        CircuitData::from_dag_ref(&result).expect("Internal DAG to circuit conversion failed");
    let num_input_qubits = circuit.num_qubits() as u32;
    *circuit = out_circuit;
    let out_permutation = (0..result.num_qubits() as u32)
        .map(|ref q| {
            Qubit(
                final_layout
                    .virtual_to_physical(initial_layout.physical_to_virtual(PhysicalQubit(*q)))
                    .0,
            )
        })
        .collect();

    Box::into_raw(Box::new(TranspileLayout::new(
        Some(initial_layout),
        Some(out_permutation),
        result.qubits().objects().clone(),
        num_input_qubits,
        result.qregs().to_vec(),
    )))
}
