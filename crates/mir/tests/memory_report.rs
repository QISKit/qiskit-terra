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

//! Memory report for the MIR instruction list, `LineGraph<MirInstruction>`.
//!
//! Shows how much memory the list takes per instruction as it grows to millions of instructions.
//!
//! Run with:
//! ```bash
//! cargo test --test memory_report -p qiskit-mir --features mem_dbg
//! ```

#![allow(clippy::print_stdout)]

use mem_dbg::{DbgFlags, MemDbg, MemSize, SizeFlags};
use qiskit_mir::instruction::MirInstruction;
use qiskit_mir::line_graph::LineGraph;
use qiskit_mir::ops::MirOp;
use qiskit_mir::ops::pbc::{AngleType, Pauli, PauliRotation, PauliString};
use qiskit_mir::qid::{Qid, QidList};
use qiskit_mir::symbol_table::SymbolIdList;

/// Build a 2-qubit `PauliRotation` instruction.
fn pauli_rotation_instruction(i: u32) -> MirInstruction {
    let op = MirOp::PauliRotation(PauliRotation::new(
        PauliString::new(vec![Pauli::X; 2]),
        AngleType::new(0.5),
    ));
    let qids = QidList::new(vec![Qid::virt(i), Qid::virt(i + 1)]);
    MirInstruction::new(op, qids, SymbolIdList::default(), SymbolIdList::default())
}

fn main() {
    println!("=== MIR instruction list: LineGraph<MirInstruction> (2-qubit PauliRotation) ===");
    println!();
    println!("--- 1. Scaling instruction list with the number of instructions ---");
    println!("used:      slots in use + instruction heap data");
    println!("allocated: used + spare Vec capacity");
    println!();
    println!(
        "{:>10}  {:>14}  {:>14}  {:>11}  {:>12}",
        "N", "used", "allocated", "used/node", "alloc/node"
    );
    let mut last = None;
    for n in [1_000_u32, 10_000, 100_000, 1_000_000] {
        let mut graph = LineGraph::new();
        for i in 0..n {
            graph.push_back(pauli_rotation_instruction(i));
        }
        let used = graph.mem_size(SizeFlags::default());
        let allocated = graph.mem_size(SizeFlags::CAPACITY);
        println!(
            "{:>10}  {:>12} B  {:>12} B  {:>9.2} B  {:>10.2} B",
            n,
            used,
            allocated,
            used as f64 / f64::from(n),
            allocated as f64 / f64::from(n),
        );
        last = Some((n, graph));
    }

    let (n, graph) = last.expect("at least one size");
    println!();
    println!("--- 2. LineGraph<MirInstruction> layout ({n} instructions) ---");
    graph
        .mem_dbg(DbgFlags::default() | DbgFlags::CAPACITY | DbgFlags::RUST_LAYOUT)
        .expect("failed to print mem_dbg tree");

    // `mem_dbg` prints a `Vec` as a single leaf, so the tree above stops at `slots`. Break one
    // instruction down instead: this shows where the per-node heap data comes from.
    let instruction = pauli_rotation_instruction(0);
    println!();
    println!("--- 3. MirInstruction layout ---");
    instruction
        .mem_dbg(DbgFlags::default())
        .expect("failed to print mem_dbg tree");
}
