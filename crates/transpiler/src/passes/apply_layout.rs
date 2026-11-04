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

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use hashbrown::HashSet;

use qiskit_circuit::bit::{QuantumRegister, Register};
use qiskit_circuit::dag_circuit::{DAGCircuit, PyDAGCircuit};
use qiskit_circuit::nlayout::NLayout;
use qiskit_circuit::{PhysicalQubit, Qubit, VirtualQubit};

use crate::transpile_layout::TranspileLayout;

/// Map the qubits of a DAG defined over virtual qubits into physical qubits.
///
/// This modifies the [DAGCircuit] and [TranspileLayout] in place, turning it into a "physical" DAG
/// (including using the canonical physical-qubit register).  This includes ancilla expansion, if
/// the number of physical qubits is greater than the number of virtual qubits.  Ancillas get
/// implicit virtual-qubit indices that are in the range `dag.num_qubits..num_physical_qubits`.
///
/// # Panics
///
/// * if the DAG is too large to fit into the given number of physical qubits
/// * if the `layout_fn` returns a `PhysicalQubit` that is out of range
/// * if `cur_layout` already has an initial layout set (perhaps you meant [update_layout])
pub fn apply_layout(
    dag: &mut DAGCircuit,
    cur_layout: &mut TranspileLayout,
    num_physical_qubits: u32,
    mut layout_fn: impl FnMut(VirtualQubit) -> PhysicalQubit,
) {
    if cur_layout.initial_physical_layout(false).is_some() {
        panic!("cannot apply a layout when one is already set");
    }

    let num_virtual_qubits = dag.num_qubits() as u32;
    let mut virtuals = vec![VirtualQubit::MAX; num_physical_qubits as usize];
    let mut physicals = Vec::with_capacity(num_physical_qubits as usize);
    for virt in (0..num_virtual_qubits).map(VirtualQubit::new) {
        let phys = layout_fn(virt);
        let old_virt = ::std::mem::replace(&mut virtuals[phys.index()], virt);
        assert_eq!(
            old_virt,
            VirtualQubit::MAX,
            "both {virt:?} and {old_virt:?} were assigned to {phys:?}",
        );
        physicals.push(phys);
    }
    // First, an iterator over the virtual indices corresponding to ancillas...
    (num_virtual_qubits..num_physical_qubits)
        .zip(virtuals.iter_mut().enumerate().filter_map(|(i, virt)| {
            // ... zipped with an iterator over the "slots" in `virtuals` that haven't been filled
            // in yet, and the physical qubits they correspond to.
            (*virt == VirtualQubit::MAX).then_some((virt, PhysicalQubit::new(i as u32)))
        }))
        .for_each(|(ancilla, (slot, phys))| {
            physicals.push(phys);
            *slot = VirtualQubit::new(ancilla);
        });
    let initial_layout = NLayout::from_vecs_unchecked(physicals, virtuals);

    let final_permutation = cur_layout.output_permutation().map(|previous| {
        let mut permutation = (0..num_physical_qubits as usize)
            .map(Qubit::new)
            .collect::<Vec<_>>();
        for virt in (0..num_virtual_qubits).map(VirtualQubit::new) {
            permutation[virt.to_phys(&initial_layout).index()] =
                VirtualQubit::from(previous[virt.index()])
                    .to_phys(&initial_layout)
                    .into();
        }
        permutation
    });
    let mut virtual_qubits = dag.qubits().objects().to_vec();
    let num_ancillas = num_physical_qubits - num_virtual_qubits;
    if num_ancillas > 0 {
        let reg_name = unique_ancilla_register_name(dag.qregs());
        virtual_qubits.extend(QuantumRegister::new_owning(reg_name, num_ancillas).bits())
    }

    dag.make_physical(num_physical_qubits as usize);
    dag.reindex_qargs(|virt| VirtualQubit::from(virt).to_phys(&initial_layout).into());
    *cur_layout = TranspileLayout::new(
        Some(initial_layout),
        final_permutation,
        virtual_qubits,
        cur_layout.num_input_qubits(),
        cur_layout.input_registers().to_vec(),
    );
}

/// Permute the qubit indices of the [DAGCircuit] and the [TranspileLayout]
///
/// This is typically called to improve a previously set layout.
///
/// # Panics
///
/// * if the `layout_fn` returns a `PhysicalQubit` that is out of range.
pub fn update_layout(
    dag: &mut DAGCircuit,
    cur_layout: &mut TranspileLayout,
    mut layout_fn: impl FnMut(Qubit) -> Qubit,
) {
    dag.reindex_qargs(&mut layout_fn);
    cur_layout.relabel_initial_layout(|q| layout_fn(q.into()).into());
}

fn unique_ancilla_register_name(qregs: &[QuantumRegister]) -> String {
    let base = "ancilla";
    if !qregs.iter().any(|qreg| qreg.name() == base) {
        // Most likely we'll return out of here, and can avoid constructing a hash set.
        return base.to_owned();
    }
    let names = qregs.iter().map(|qreg| qreg.name()).collect::<HashSet<_>>();
    let mut i = 0;
    loop {
        let name = format!("{base}_{i}");
        if !names.contains(name.as_str()) {
            return name;
        }
        i += 1;
    }
}

/// Apply a layout to the :class:`.DAGCircuit`.
///
/// Args:
///     dag: the circuit.
///     num_virtual_qubits: the original number of qubits in the input circuit.  When running in
///         legacy mode, this might be less than the number of qubits in the :class:`.DAGCircuit`.
///         In "normal" mode, where :class:`.ApplyLayout` is also embedded and expanding the virtual
///         layout, this would be the same as the number of qubits in the :class:`.DAGCircuit`.
///     num_physical_qubits: the number of physical qubits in the target we're compiling for.
///     physical_from_virtual: the layout to apply.
///     permutation: the current output permutation of the circuit.
#[pyfunction]
#[pyo3(name = "apply_layout")]
fn py_apply_layout<'py>(
    py: Python<'py>,
    dag: &mut PyDAGCircuit,
    num_virtual_qubits: u32,
    num_physical_qubits: u32,
    physical_from_virtual: Vec<PhysicalQubit>,
    permutation: Option<Vec<Qubit>>,
) -> PyResult<Bound<'py, PyAny>> {
    let dag_mut = dag.try_write()?;
    let num_dag_qubits = dag_mut.num_qubits() as u32;
    if num_dag_qubits > num_physical_qubits {
        return Err(PyValueError::new_err(format!(
            "More qubits in DAG ({num_dag_qubits}) than physical qubits ({num_physical_qubits})"
        )));
    }
    if num_virtual_qubits > num_dag_qubits {
        return Err(PyValueError::new_err(format!(
            "More original input qubits ({num_virtual_qubits}) than in the DAG ({num_dag_qubits})"
        )));
    }
    if physical_from_virtual.len() != dag_mut.num_qubits() {
        return Err(PyValueError::new_err(format!(
            "Layout has different number of qubits ({}) than the DAG ({})",
            physical_from_virtual.len(),
            num_dag_qubits
        )));
    }
    match permutation.as_ref().map(Vec::len) {
        Some(len) if len != dag_mut.num_qubits() => {
            return Err(PyValueError::new_err(format!(
                "Given permutation has difference number of qubits ({len}) than the DAG ({num_dag_qubits})"
            )));
        }
        _ => (),
    }
    if physical_from_virtual.len() != dag_mut.num_qubits() {
        return Err(PyValueError::new_err(format!(
            "Layout has different number of qubits ({}) than the DAG ({})",
            physical_from_virtual.len(),
            num_dag_qubits
        )));
    }
    let mut seen = vec![false; num_physical_qubits as usize];
    for qubit in physical_from_virtual.iter() {
        if qubit.index() >= num_physical_qubits as usize {
            return Err(PyValueError::new_err(format!(
                "Layout contains out-of-bounds qubit {}",
                qubit.index()
            )));
        }
        if ::std::mem::replace(&mut seen[qubit.index()], true) {
            return Err(PyValueError::new_err(format!(
                "Layout contains duplicate qubit {}",
                qubit.index()
            )));
        }
    }

    // TODO: while Rust-space and Python-space `TranspileLayout` are two different objects, Python
    // gets the short end of the "unnecessary conversion" stick, and we have to make a new object to
    // return to Python.  We don't accept the Python-space object into this function because it
    // can't represent the state of "initial layout is not yet applied", so we may as well just make
    // it ourselves.
    let mut cur_layout = TranspileLayout::new(
        None,
        permutation,
        dag_mut.qubits().objects().to_vec(),
        // We had to take `num_virtual_qubits` by value because the DAG might already have been
        // expanded with ancillas in the legacy mode.
        num_virtual_qubits,
        dag_mut.qregs().to_vec(),
    );
    apply_layout(dag_mut, &mut cur_layout, num_physical_qubits, |v| {
        physical_from_virtual[v.index()]
    });
    cur_layout.to_py_native(py, dag_mut.qubits().objects())
}

#[pyfunction]
#[pyo3(name = "update_layout")]
fn py_update_layout<'py>(
    dag: &mut PyDAGCircuit,
    py_layout: &Bound<'py, PyAny>,
    reorder: Vec<Qubit>,
) -> PyResult<Bound<'py, PyAny>> {
    let dag_mut = dag.try_write()?;
    if reorder.len() != dag_mut.num_qubits() {
        return Err(PyValueError::new_err(format!(
            "Updated layout has different number of qubits ({}) to the DAG ({})",
            reorder.len(),
            dag_mut.num_qubits()
        )));
    }
    let mut seen = vec![false; dag_mut.num_qubits()];
    for qubit in reorder.iter() {
        if qubit.index() >= dag_mut.num_qubits() {
            return Err(PyValueError::new_err(format!(
                "Layout contains out-of-bounds qubit {}",
                qubit.index()
            )));
        }
        if ::std::mem::replace(&mut seen[qubit.index()], true) {
            return Err(PyValueError::new_err(format!(
                "Layout contains duplicate qubit {}",
                qubit.index()
            )));
        }
    }

    let mut cur_layout = TranspileLayout::from_py_native(py_layout)?;
    update_layout(dag_mut, &mut cur_layout, |q| reorder[q.index()]);
    cur_layout.to_py_native(py_layout.py(), dag_mut.qubits().objects())
}

pub fn apply_layout_mod(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(py_apply_layout))?;
    m.add_wrapped(wrap_pyfunction!(py_update_layout))?;
    Ok(())
}
