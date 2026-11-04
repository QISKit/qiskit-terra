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

use qiskit_circuit::Qubit;
use qiskit_circuit::bit::QuantumRegister;
use qiskit_circuit::bit::ShareableQubit;
use qiskit_circuit::nlayout::{NLayout, PhysicalQubit, VirtualQubit};

// TODO: Conditionally compile these imports for Python builds
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PySet};
use qiskit_circuit::imports::LAYOUT;
use qiskit_circuit::imports::TRANSPILE_LAYOUT;

/// The "layout" caused by transpilation
///
/// In general Qiskit's transpiler is unitary-preserving up to the initial layout
/// and routing permutations. The initial layout permutation is caused by
/// setting and applying the initial layout (the mapping from virtual circuit
/// qubits to physical qubits on the target) and the routing permtuations are
/// caused by swap gate insertion or permutation elision prior to the initial
/// layout. This struct tracks these details and provide an interface to reason
/// about these permutations.
pub struct TranspileLayout {
    /// The initial layout which is mapping the virtual qubits in the input circuit to the
    /// transpiler to the physical qubits used on the transpilation target
    initial_layout: Option<NLayout>,
    /// The optional routing permutation that
    /// represents the permutation caused by routing or permutation elision during
    /// transpilation. This vector maps the qubits at the start of the circuit to their
    /// final position/physical qubit at the end of the circuit.
    output_permutation: Option<Vec<Qubit>>,
    /// The virtual qubits [`ShareableQubit`] objects from the input circuit to the transpiler.
    /// This vec should be arranged in order as in the original circuit, the index of the `Vec`
    /// corresponds to the [`VirtualQubit`] in the `initial_layout` attribute. This should include
    /// any allocated ancilla qubits so that the length of `virtual_qubits` matches the length of
    /// `initial_layout`, `num_input_qubits` is used to determine which qubit objects were in the
    /// original circuit, and which were allocated ancillas.
    virtual_qubits: Vec<ShareableQubit>,
    /// The number of input qubits in the original input circuit
    num_input_qubits: u32,
    input_registers: Vec<QuantumRegister>,
}

impl TranspileLayout {
    /// Construct a new [`TranspileLayout`] object
    ///
    /// # Args
    ///
    /// - `initial_layout` - The initial layout which is mapping from the virtual qubits in the
    ///   input circuit to the transpiler to the physical qubits used on the transpilation target
    /// - `output_permutation` - The optional routing permutation that
    ///   represents the permutation caused by routing or permutation elision during
    ///   transpilation. This vector maps the qubits at the start of the circuit to their
    ///   final position/physical qubit at the end of the circuit.
    /// - `virtual_qubits` - The virtual qubits [`ShareableQubit`] objects from the input circuit to the transpiler.
    ///   his `Vec` should be arranged in order as in the original circuit, the index of the `Vec`
    ///   corresponds to the [`VirtualQubit`] in the `initial_layout` attribute. This should include
    ///   any allocated ancilla qubits so that the length of `virtual_qubits` matches the length of
    ///   initial_layout`, `num_input_qubits` is used to determine which qubit objects were in the
    ///   original circuit, and which were allocated ancillas.
    /// - `num_input_qubits` - The number of qubits in the original input circuit
    pub fn new(
        initial_layout: Option<NLayout>,
        output_permutation: Option<Vec<Qubit>>,
        virtual_qubits: Vec<ShareableQubit>,
        num_input_qubits: u32,
        registers: Vec<QuantumRegister>,
    ) -> Self {
        TranspileLayout {
            initial_layout,
            output_permutation,
            virtual_qubits,
            num_input_qubits,
            input_registers: registers,
        }
    }

    /// Create a [TranspileLayout] object from a pair of layouts, rather than the natural
    /// constituent parts.
    ///
    /// # Panics
    ///
    /// If the two layouts and the number of virtual qubits are not all the same size, or if the
    /// number of input qubits is larger than the sizes of the other objects.
    pub fn from_layouts(
        initial_layout: NLayout,
        final_layout: &NLayout,
        virtual_qubits: Vec<ShareableQubit>,
        num_input_qubits: u32,
        input_registers: Vec<QuantumRegister>,
    ) -> Self {
        assert!(
            initial_layout.num_qubits() == virtual_qubits.len(),
            "number of virtual qubit objects did not match the given layouts"
        );
        assert!(
            num_input_qubits as usize <= virtual_qubits.len(),
            "cannot have more input qubits than ancilla-expanded virtual qubits"
        );
        Self {
            output_permutation: Some(Self::permutation_from_layouts(
                &initial_layout,
                final_layout,
            )),
            initial_layout: Some(initial_layout),
            virtual_qubits,
            num_input_qubits,
            input_registers,
        }
    }

    /// Create a permutation vector in the same convention as [TranspileLayout] that represents the
    /// same permutation as between two explicit layouts.
    pub fn permutation_from_layouts(
        initial_layout: &NLayout,
        final_layout: &NLayout,
    ) -> Vec<Qubit> {
        assert!(
            initial_layout.num_qubits() == final_layout.num_qubits(),
            "layout objects had mismatched numbers of qubits"
        );
        (0..initial_layout.num_qubits())
            .map(|i| {
                PhysicalQubit::new(i as u32)
                    .to_virt(initial_layout)
                    .to_phys(final_layout)
                    .into()
            })
            .collect()
    }

    pub fn initial_layout(&self) -> Option<&NLayout> {
        self.initial_layout.as_ref()
    }

    /// The number of input circuit qubits
    pub fn num_input_qubits(&self) -> u32 {
        self.num_input_qubits
    }

    pub fn input_registers(&self) -> &[QuantumRegister] {
        &self.input_registers
    }

    /// The number of output circuit qubits
    pub fn num_output_qubits(&self) -> u32 {
        if let Some(ref initial_layout) = self.initial_layout {
            initial_layout.num_qubits() as u32
        } else if let Some(ref permutation) = self.output_permutation {
            permutation.len() as u32
        } else {
            self.virtual_qubits.len() as u32
        }
    }

    /// Generate an initial layout as an array of Qubit indices.
    ///
    /// # Args
    ///     `filter_ancillas` - If set to `true` any ancilla qubits added to
    ///     the circuit by the transpiler will not be included in the output
    ///     array.
    pub fn initial_physical_layout(&self, filter_ancillas: bool) -> Option<Vec<PhysicalQubit>> {
        self.initial_layout.as_ref().map(|layout| {
            if filter_ancillas {
                (0..self.num_input_qubits())
                    .map(|x| VirtualQubit::new(x).to_phys(layout))
                    .collect()
            } else {
                (0..self.num_output_qubits())
                    .map(|x| VirtualQubit::new(x).to_phys(layout))
                    .collect()
            }
        })
    }

    /// Return the routing permutation
    ///
    /// This method returns an option slice, if it is `Some` then there
    /// was a permutation introduced by the transpiler typically caused
    /// by either routing or permutation elision. The slice contained
    /// in the return represents the mapping of the qubits at the start
    /// of the circuit for each index to their final position
    /// at the end of the circuit.
    ///
    /// If you would instead prefer to represent no permutation case with
    /// an explicit trivial permutation (e.g. `[0, 1, 2, 3]`) then you can
    /// use [`explicit_output_permutation`] which always returns a slice.
    pub fn output_permutation(&self) -> Option<&[Qubit]> {
        self.output_permutation.as_deref()
    }

    /// Return the routing permutation explicitly
    ///
    /// This method returns a slice, of the permutation introduced by the
    /// transpiler typically caused by either routing or permutation elision.
    /// The slice contained in the return represents the mapping of the qubits
    /// at the start of the circuit for each index to their final position
    /// at the end of the circuit.
    ///
    /// If you would instead prefer to represent no permutation case with
    /// a `None` then you can use [`output_permutation`] which returns an
    /// `Option<&[Qubit]>`.
    pub fn explicit_output_permutation(&self) -> std::borrow::Cow<'_, [Qubit]> {
        match self.output_permutation {
            Some(ref perm) => std::borrow::Cow::Borrowed(perm),
            None => std::borrow::Cow::Owned(
                (0..self.num_output_qubits() as usize)
                    .map(Qubit::new)
                    .collect(),
            ),
        }
    }

    /// Generate the final layout as an array of PhysicalQubits
    ///
    /// This method will generate an array of final positions for each qubit in the input circuit.
    /// For example, if you had an input circuit like:
    ///
    /// ```python
    /// qc = QuantumCircuit(3)
    /// qc.h(0)
    /// qc.cx(0, 1)
    /// qc.cx(0, 2)
    /// ```
    ///
    /// and then the output from the transpiler was:
    ///
    /// ```python
    /// tqc = QuantumCircuit(3)
    /// tqc.h(2)
    /// tqc.cx(2, 1)
    /// tqc.swap(0, 1)
    /// tqc.cx(2, 1)
    /// ```
    ///
    /// then the `final_index_layout` method returns:
    ///
    /// ```python
    /// [2, 0, 1]
    /// ```
    ///
    /// This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
    /// in the output circuit during the layout stage, which is mapped to qubit 2 during the
    /// routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
    /// circuit during the layout stage, which is mapped to qubit 0 during the routing
    /// stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
    /// during the layout stage, which is mapped to qubit 1 during the routing stage.
    /// The output list length will be as wide as the input circuit's number of qubits,
    /// as the output list from this method is for tracking the permutation of qubits in the
    /// original circuit caused by the transpiler.
    ///
    /// # Args
    ///     `filter_ancillas` - If set to `true` any ancilla qubits added to
    ///     the circuit by the transpiler will not be included in the output
    ///     array.
    pub fn final_index_layout(&self, filter_ancillas: bool) -> Vec<PhysicalQubit> {
        let qubit_range = if filter_ancillas {
            0..self.num_input_qubits()
        } else {
            0..self.num_output_qubits()
        };
        qubit_range
            .map(|idx| {
                let mut qubit_idx = self
                    .initial_layout
                    .as_ref()
                    .map(|x| x.virtual_to_physical(VirtualQubit::new(idx)))
                    .unwrap_or_else(|| PhysicalQubit::new(idx));
                if let Some(ref output_permutation) = self.output_permutation {
                    qubit_idx = PhysicalQubit::new(output_permutation[qubit_idx.index()].0)
                }
                qubit_idx
            })
            .collect()
    }

    /// Generate the final layout
    ///
    /// This method will generate the final layout which is the mapping of the
    /// the virtual qubits in the input circuit to the transpiler to the
    /// physical qubits with that virtual qubit's state at the end of the circuit.
    /// For example, if you had an input circuit like:
    ///
    /// ```python
    /// qc = QuantumCircuit(3)
    /// qc.h(0)
    /// qc.cx(0, 1)
    /// qc.cx(0, 2)
    /// ```
    ///
    /// and then the output from the transpiler was:
    ///
    /// ```python
    /// tqc = QuantumCircuit(3)
    /// tqc.h(2)
    /// tqc.cx(2, 1)
    /// tqc.swap(0, 1)
    /// tqc.cx(2, 1)
    /// ```
    ///
    /// then the `final_layout` method returns:
    ///
    /// | Virtual Qubit | Physical Qunit |
    /// |---------------|----------------|
    /// | 0             | 2              |
    /// | 1             | 0              |
    /// | 2             | 1              |
    ///
    /// This can be seen as follows. Qubit 0 in the original circuit is mapped to qubit 2
    /// in the output circuit during the layout stage, which is mapped to qubit 2 during the
    /// routing stage. Qubit 1 in the original circuit is mapped to qubit 1 in the output
    /// circuit during the layout stage, which is mapped to qubit 0 during the routing
    /// stage. Qubit 2 in the original circuit is mapped to qubit 0 in the output circuit
    /// during the layout stage, which is mapped to qubit 1 during the routing stage.
    /// The output list length will be as wide as the input circuit's number of qubits,
    /// as the output list from this method is for tracking the permutation of qubits in the
    /// original circuit caused by the transpiler.
    ///
    /// # Returns
    ///
    /// An [`NLayout`] object. This will always include all ancilla qubits
    /// allocated by the transpiler because an `NLayout` must have an equal
    /// number of [`VirtualQubit`] and [`PhysicalQubit`]. If you want a view
    /// with ancillas filtered you should use [`final_index_layout`] instead
    /// which returns the layout as a `Vec<PhysicalQubit>`.
    pub fn final_layout(&self) -> NLayout {
        NLayout::from_virtual_to_physical(self.final_index_layout(false)).unwrap()
    }

    /// Update the initial layout by permuting the output-space qubit indices from the input of the
    /// `relabel_fn` to its output.
    ///
    /// For example, if virtual qubit 0 is previously mapped to physical qubit 2, and this function
    /// is called with a mapping that sends physical qubit 2 to physical qubit 4, the new layout
    /// state will map virtual qubit 0 to physical qubit 4.
    ///
    /// This includes updating all of the other tracked objects.  If no `initial_layout` is set, it
    /// is treated as equivalent to the identity mapping.
    ///
    /// # Panics
    ///
    /// If `relabel_fn` returns and out-of-bounds [PhysicalQubit], or maps more than one
    /// [PhysicalQubit] to the same new value.
    pub fn relabel_initial_layout(
        &mut self,
        mut relabel_fn: impl FnMut(PhysicalQubit) -> PhysicalQubit,
    ) {
        let initial_layout = match self.initial_layout.as_ref() {
            Some(layout) => NLayout::from_virtual_to_physical(
                (0..self.num_output_qubits())
                    .map(|q| relabel_fn(VirtualQubit::new(q).to_phys(layout)))
                    .collect(),
            )
            .expect("all qubits should be in bounds and not duplicates"),
            None => NLayout::from_virtual_to_physical(
                (0..self.num_output_qubits())
                    .map(|q| relabel_fn(PhysicalQubit::new(q)))
                    .collect(),
            )
            .expect("all qubits should be in bounds and not duplicates"),
        };
        let output_permutation = self.output_permutation.as_ref().map(|permutation| {
            let mut new_permutation = vec![Qubit::MAX; permutation.len()];
            for old in 0..permutation.len() as u32 {
                let old = PhysicalQubit::new(old);
                let new = relabel_fn(old);
                if new_permutation[new.index()] != Qubit::MAX {
                    panic!("layout function returned duplicate qubit");
                }
                new_permutation[new.index()] = relabel_fn(permutation[old.index()].into()).into();
            }
            new_permutation
        });

        self.initial_layout = Some(initial_layout);
        self.output_permutation = output_permutation;
    }

    /// Add an extra "undoing" permutation that comes inbetween the [DAGCircuit] and any permutation
    /// already stored in this layout.
    ///
    /// This is typically what transpiler passes should call if they are mutating a
    /// [TranspileLayout] they received, to add new information; if you mutate a [DAGCircuit]
    /// inducing an additional permutation to track, then you call this method with the new
    /// permutation to add it to the existing layout tracking.
    ///
    /// If the combination of this layout and the DAG originally formed a triple like:
    /// ```text
    /// (initial layout, DAG, output permutation)
    /// ```
    /// and a transpiler passes splits the DAG into
    /// ```text
    /// (DAG, new permutation)
    /// ```
    /// it would then call this function with the new permutation in order to combine it into the
    /// existing layout (whether or not the initial layout and prior output permutation were set).
    ///
    /// The function is a "gets set to" permutation, just like as returned by [output_permutation].
    ///
    /// The `permutation` function will be called once for each qubit index in the "output" space
    /// (in other words, the range from 0 up to but excluding [num_output_qubits]).
    pub fn add_permutation_inside(&mut self, mut permutation: impl FnMut(Qubit) -> Qubit) {
        let new_permutation = match self.output_permutation.as_ref() {
            Some(previous) => previous.iter().map(|q| permutation(*q)).collect(),
            None => (0..self.num_output_qubits())
                .map(|q| permutation(Qubit(q)))
                .collect(),
        };
        self.output_permutation = Some(new_permutation);
    }

    /// Add an extra "undoing" permutation that applied to the virtual qubits of the input circuit
    /// before the the rest of this layout was split out from the DAG.
    ///
    /// Say a pass like `ElidePermutations` had run, and so the current state of the transpiler IR
    /// was
    /// ```text
    /// (virtual DAG, virtual permutation)
    /// ```
    /// Then, a layout and/or routing pass run on "virtual DAG", ignoring the virtual permutation,
    /// so we now have a situation where we hold
    /// ```text
    /// ((initial layout, physical DAG, routing permutation), virtual permutation)
    /// ```
    /// and this [TranspileLayout] corresponds to the inner 3-tuple.  In order to combine the
    /// previous virtual permutation into this layout, you would call this function.
    ///
    /// The function is a "gets set to" permutation, just like as returned by [output_permutation].
    ///
    /// The `permutation` function will be called once for each qubit index in the "input" space
    /// (in other words, the range from 0 up to but excluding [num_input_qubits]).
    ///
    /// # Panics
    ///
    /// If this layout does not have a set [initial_layout].
    pub fn add_permutation_outside(
        &mut self,
        mut permutation: impl FnMut(VirtualQubit) -> VirtualQubit,
    ) {
        let initial = self
            .initial_layout
            .as_ref()
            .expect("it only makes sense to call this function when the layout is set");
        // These are logically physical qubits, but we want to be able to assign it directly.
        let mut physical_permutation = vec![Qubit::MAX; self.num_output_qubits() as usize];
        for virt in (0..self.num_input_qubits()).map(VirtualQubit) {
            let phys = virt.to_phys(initial);
            physical_permutation[phys.index()] = permutation(virt).to_phys(initial).into();
        }
        for ancilla in (self.num_input_qubits()..self.num_output_qubits()).map(VirtualQubit) {
            physical_permutation[ancilla.to_phys(initial).index()] =
                ancilla.to_phys(initial).into();
        }
        let new_permutation = match self.output_permutation.as_ref() {
            Some(current) => (0..self.num_output_qubits())
                .map(|q| current[physical_permutation[q as usize].index()])
                .collect(),
            None => physical_permutation,
        };
        self.output_permutation = Some(new_permutation);
    }

    // TODO: Conditionally compile this method so we don't depend on symbols from Python
    /// Return a Python space `TranspileLayout` object built from this rust space `TranspileLayout`
    ///
    /// # Args
    ///
    /// - `py`: Python token for gil access
    /// - `output_qubits`: A slice of the qubit objects from the output circuit's qubits
    pub fn to_py_native<'py>(
        &self,
        py: Python<'py>,
        output_qubits: &[ShareableQubit],
    ) -> PyResult<Bound<'py, PyAny>> {
        let initial_layout_dict = PyDict::new(py);
        let input_qubit_mapping = PyDict::new(py);
        for (idx, virtual_bit) in self.virtual_qubits.iter().enumerate() {
            initial_layout_dict.set_item(
                virtual_bit,
                self.initial_layout
                    .as_ref()
                    .map(|x| x.virtual_to_physical(VirtualQubit::new(idx as u32))),
            )?;
            input_qubit_mapping.set_item(virtual_bit, idx)?;
        }
        let initial_layout = LAYOUT.get_bound(py).call1((initial_layout_dict,))?;
        for reg in &self.input_registers {
            initial_layout.call_method1(intern!(py, "add_register"), (reg,))?;
        }
        let final_layout: Option<Bound<PyAny>> = self
            .output_permutation()
            .map(|perm| -> PyResult<Bound<PyAny>> {
                let final_layout_dict = PyDict::new(py);
                for (idx, bit) in output_qubits.iter().enumerate() {
                    final_layout_dict.set_item(bit, perm[idx].0)?;
                }
                LAYOUT.get_bound(py).call1((final_layout_dict,))
            })
            .transpose()?;
        TRANSPILE_LAYOUT.get_bound(py).call1((
            initial_layout,
            input_qubit_mapping,
            final_layout,
            self.num_input_qubits(),
            output_qubits,
        ))
    }

    // TODO: Conditionally compile this method so we don't depend on symbols from Python
    /// Build a rust space `TranspileLayout` from a python space `TranspileLayout
    ///
    /// # Args
    ///
    /// - `py_layout`: A Python `TranspileLayout` object
    pub fn from_py_native(py_layout: &Bound<PyAny>) -> PyResult<Self> {
        let py = py_layout.py();
        let initial_index_layout: Option<Vec<PhysicalQubit>> = py_layout
            .call_method1(intern!(py, "initial_index_layout"), (false,))?
            .extract()?;
        let initial_layout = initial_index_layout
            .map(NLayout::from_virtual_to_physical)
            .transpose()?;
        let output_permutation = if !py_layout.getattr(intern!(py, "final_layout"))?.is_none() {
            let permutation: Vec<Qubit> = py_layout
                .call_method0(intern!(py, "routing_permutation"))?
                .extract()?;
            Some(permutation)
        } else {
            None
        };
        let index_map = py_layout
            .getattr(intern!(py, "input_qubit_mapping"))?
            .cast::<PyDict>()?
            .iter()
            .map(|(k, v)| -> PyResult<(usize, ShareableQubit)> {
                let index: usize = v.extract()?;
                let value: ShareableQubit = k.extract()?;
                Ok((index, value))
            })
            .collect::<PyResult<HashMap<usize, ShareableQubit>>>()?;
        let num_input_qubits = py_layout
            .getattr(intern!(py, "_input_qubit_count"))?
            .extract::<Option<u32>>()?
            .unwrap_or_else(|| index_map.len() as u32);
        let virtual_qubits: Vec<ShareableQubit> = if let Some(ref initial_layout) = initial_layout {
            (0..initial_layout.num_qubits())
                .map(|x| index_map[&x].clone())
                .collect()
        } else {
            (0..num_input_qubits as usize)
                .map(|x| index_map[&x].clone())
                .collect()
        };
        let input_registers: Vec<QuantumRegister> = py_layout
            .getattr(intern!(py, "initial_layout"))?
            .call_method0(intern!(py, "get_registers"))?
            .cast::<PySet>()?
            .iter()
            .map(|x| x.extract::<QuantumRegister>().map_err(PyErr::from))
            .collect::<PyResult<Vec<QuantumRegister>>>()?;
        Ok(Self::new(
            initial_layout,
            output_permutation,
            virtual_qubits,
            num_input_qubits,
            input_registers,
        ))
    }
}

#[cfg(test)]
mod test_transpile_layout {
    use super::TranspileLayout;
    use qiskit_circuit::Qubit;
    use qiskit_circuit::bit::ShareableQubit;
    use qiskit_circuit::nlayout::{NLayout, PhysicalQubit, VirtualQubit};

    #[test]
    fn test_final_index_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![Qubit(1), Qubit(0), Qubit(2)];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.final_index_layout(false);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(0), PhysicalQubit(1)],
            result
        );
    }

    #[test]
    fn test_final_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![Qubit(1), Qubit(0), Qubit(2)];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.final_layout();
        assert_eq!(
            NLayout::from_virtual_to_physical(vec![
                PhysicalQubit(2),
                PhysicalQubit(0),
                PhysicalQubit(1)
            ])
            .unwrap(),
            result
        );
    }

    #[test]
    fn test_initial_layout() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![Qubit(1), Qubit(0), Qubit(2)];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.initial_physical_layout(false);
        assert_eq!(
            Some(vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)]),
            result
        );
    }

    #[test]
    fn test_output_permutation() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![Qubit(1), Qubit(0), Qubit(2)];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.output_permutation();
        assert_eq!(Some([Qubit(1), Qubit(0), Qubit(2)].as_slice()), result);
    }

    #[test]
    fn test_output_permutation_explicit() {
        let initial_layout_vec = vec![PhysicalQubit(2), PhysicalQubit(1), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![Qubit(1), Qubit(0), Qubit(2)];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.explicit_output_permutation();
        assert_eq!(
            std::borrow::Cow::Borrowed(&[Qubit(1), Qubit(0), Qubit(2)]),
            result
        );
    }

    #[test]
    fn test_final_index_layout_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.final_index_layout(true);
        assert_eq!(
            vec![PhysicalQubit(3), PhysicalQubit(5), PhysicalQubit(2)],
            result
        )
    }

    #[test]
    fn test_initial_layout_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.initial_physical_layout(true);
        assert_eq!(
            Some(vec![PhysicalQubit(9), PhysicalQubit(4), PhysicalQubit(0)]),
            result
        )
    }

    #[test]
    fn test_output_permutation_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let expected = output_permutation.clone();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.output_permutation();
        assert_eq!(Some(expected.as_slice()), result)
    }

    #[test]
    fn test_output_permutation_with_ancillas_explicit() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let expected_vec = output_permutation.clone();
        let expected: std::borrow::Cow<_> = expected_vec.as_slice().into();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.explicit_output_permutation();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_final_index_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.final_index_layout(false);
        let expected = vec![
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
        ];
        assert_eq!(expected, result)
    }

    #[test]
    fn test_final_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.final_layout();
        let expected = NLayout::from_virtual_to_physical(vec![
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(2),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(4),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
            PhysicalQubit(9),
        ])
        .unwrap();
        assert_eq!(expected, result)
    }

    #[test]
    fn test_initial_layout_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(9),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(2),
            PhysicalQubit(3),
            PhysicalQubit(5),
            PhysicalQubit(6),
            PhysicalQubit(7),
            PhysicalQubit(8),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let output_permutation = vec![
            Qubit(2),
            Qubit(0),
            Qubit(1),
            Qubit(4),
            Qubit(5),
            Qubit(6),
            Qubit(7),
            Qubit(8),
            Qubit(9),
            Qubit(3),
        ];
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 10];
        let layout = TranspileLayout::new(
            Some(initial_layout),
            Some(output_permutation),
            initial_qubits,
            3,
            vec![],
        );
        let result = layout.initial_physical_layout(false);
        assert_eq!(Some(expected), result)
    }

    #[test]
    fn test_final_index_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.final_index_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let expected = initial_layout.clone();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.final_layout();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_initial_layout_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.initial_physical_layout(false);
        assert_eq!(Some(expected), result);
    }

    #[test]
    fn test_output_permutation_no_routing() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.output_permutation();
        assert_eq!(None, result);
    }

    #[test]
    fn test_output_permutation_no_routing_explicit() {
        let initial_layout_vec = vec![PhysicalQubit(1), PhysicalQubit(2), PhysicalQubit(0)];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 3];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.explicit_output_permutation();
        let expected: std::borrow::Cow<[Qubit]> = (0..3).map(Qubit::new).collect();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_index_layout_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.final_index_layout(true);
        assert_eq!(
            vec![PhysicalQubit(2), PhysicalQubit(4), PhysicalQubit(0)],
            result
        );
    }

    #[test]
    fn test_initial_layout_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.initial_physical_layout(true);
        assert_eq!(
            Some(vec![PhysicalQubit(2), PhysicalQubit(4), PhysicalQubit(0)]),
            result
        );
    }

    #[test]
    fn test_output_permutation_no_routing_with_ancillas() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.output_permutation();
        assert_eq!(None, result);
    }

    #[test]
    fn test_output_permutation_no_routing_with_ancillas_explicit() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.explicit_output_permutation();
        let expected = std::borrow::Cow::from_iter((0..5).map(Qubit::new));
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_index_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.final_index_layout(false);
        assert_eq!(expected, result);
    }

    #[test]
    fn test_final_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let expected = initial_layout.clone();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.final_layout();
        assert_eq!(expected, result);
    }

    #[test]
    fn test_initial_layout_no_routing_with_ancillas_no_filter() {
        let initial_layout_vec = vec![
            PhysicalQubit(2),
            PhysicalQubit(4),
            PhysicalQubit(0),
            PhysicalQubit(1),
            PhysicalQubit(3),
        ];
        let expected = initial_layout_vec.clone();
        let initial_layout = NLayout::from_virtual_to_physical(initial_layout_vec).unwrap();
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 5];
        let layout = TranspileLayout::new(Some(initial_layout), None, initial_qubits, 3, vec![]);
        let result = layout.initial_physical_layout(false);
        assert_eq!(Some(expected), result);
    }

    #[test]
    fn test_compose() {
        let first = vec![Qubit(0), Qubit(3), Qubit(1), Qubit(2)];
        let second = [2, 3, 1, 0].map(VirtualQubit);
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 4];
        let mut layout = TranspileLayout::new(
            Some(NLayout::generate_trivial_layout(4)),
            Some(first),
            initial_qubits,
            4,
            vec![],
        );
        layout.add_permutation_outside(|q| second[q.index()]);
        let result = layout.output_permutation();
        let expected = Some([Qubit(1), Qubit(2), Qubit(3), Qubit(0)].as_slice());
        assert_eq!(expected, result);

        let first = vec![Qubit(1), Qubit(2), Qubit(3), Qubit(0)];
        let second = [0, 2, 1, 3].map(Qubit);
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 4];
        let mut layout = TranspileLayout::new(None, Some(first), initial_qubits, 4, vec![]);
        layout.add_permutation_inside(|q| second[q.index()]);
        let result = layout.output_permutation();
        let expected = Some([Qubit(2), Qubit(1), Qubit(3), Qubit(0)].as_slice());
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compose_no_permutation_original() {
        let second = [2, 3, 1, 0].map(Qubit);
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 4];
        let mut layout = TranspileLayout::new(None, None, initial_qubits, 4, vec![]);
        layout.add_permutation_inside(|q| second[q.index()]);
        let result = layout.output_permutation();
        let expected = Some([Qubit(2), Qubit(3), Qubit(1), Qubit(0)].as_slice());
        assert_eq!(expected, result);
    }

    #[test]
    fn test_compose_no_permutation_second() {
        let second = [2, 3, 1, 0].map(VirtualQubit);
        let initial_qubits = vec![ShareableQubit::new_anonymous(); 4];
        let mut layout = TranspileLayout::new(
            Some(NLayout::generate_trivial_layout(4)),
            None,
            initial_qubits,
            4,
            vec![],
        );
        layout.add_permutation_outside(|q| second[q.index()]);
        let result = layout.output_permutation();
        let expected = Some([Qubit(2), Qubit(3), Qubit(1), Qubit(0)].as_slice());
        assert_eq!(expected, result);
    }
}
