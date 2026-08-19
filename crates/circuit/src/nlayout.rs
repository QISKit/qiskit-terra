// This code is part of Qiskit.
//
// (C) Copyright IBM 2022
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::num::TryFromIntError;

#[cfg(feature = "py")]
use pyo3::IntoPyObjectExt;
#[cfg(feature = "py")]
use pyo3::prelude::*;
#[cfg(feature = "py")]
use pyo3::types::PyList;

use hashbrown::HashMap;

/// A newtype for the different categories of qubits used within layouts.  This is to enforce
/// significantly more type safety when dealing with mixtures of physical and virtual qubits, as we
/// typically are when dealing with layouts.  In Rust space, `NLayout` only works in terms of the
/// correct newtype, meaning that it's not possible to accidentally pass the wrong type of qubit to
/// a lookup.  We can't enforce the same rules on integers in Python space without runtime
/// overhead, so we just allow conversion to and from any valid `PyLong`.
macro_rules! qubit_newtype {
    ($id: ident) => {
        #[repr(transparent)]
        #[cfg_attr(feature = "py", derive(IntoPyObject, IntoPyObjectRef))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
        pub struct $id(pub u32);

        impl $id {
            /// The maximum storable index.
            pub const MAX: Self = $id($crate::Qubit::MAX.index() as _);

            #[inline]
            pub fn new(val: u32) -> Self {
                Self(val)
            }
            #[inline]
            pub fn index(&self) -> usize {
                self.0 as usize
            }

            /// Specialise a slice of `Qubit` values into a slice of this type's values.
            pub fn lift_slice(slice: &[$crate::Qubit]) -> &[$id] {
                // SAFETY: `Qubit` and this type share the exact same set of valid bit patterns and
                // alignments.
                unsafe { ::std::slice::from_raw_parts(slice.as_ptr() as *const $id, slice.len()) }
            }
        }

        impl From<$id> for $crate::Qubit {
            fn from(val: $id) -> Self {
                $crate::Qubit(val.0)
            }
        }
        impl From<$crate::Qubit> for $id {
            fn from(val: $crate::Qubit) -> Self {
                Self(val.0)
            }
        }

        #[cfg(feature = "py")]
        impl<'a, 'py> ::pyo3::FromPyObject<'a, 'py> for $id {
            type Error = <u32 as FromPyObject<'a, 'py>>::Error;

            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
                ob.extract().map(Self)
            }
        }

        #[cfg(feature = "py")]
        unsafe impl numpy::Element for $id {
            const IS_COPY: bool = true;

            fn get_dtype(py: Python<'_>) -> Bound<'_, numpy::PyArrayDescr> {
                u32::get_dtype(py)
            }

            fn clone_ref(&self, _py: Python<'_>) -> Self {
                *self
            }
        }

        unsafe impl ::rustworkx_core::petgraph::graph::IndexType for $id {
            fn new(x: usize) -> Self {
                Self::new(x as u32)
            }
            fn index(&self) -> usize {
                self.0 as usize
            }
            fn max() -> Self {
                Self::MAX
            }
        }
    };
}

qubit_newtype!(PhysicalQubit);
impl PhysicalQubit {
    /// Get the virtual qubit that currently corresponds to this index of physical qubit in the
    /// given layout.
    pub fn to_virt(self, layout: &NLayout) -> VirtualQubit {
        layout.phys_to_virt[self.index()]
    }
}

qubit_newtype!(VirtualQubit);
impl VirtualQubit {
    /// Get the physical qubit that currently corresponds to this index of virtual qubit in the
    /// given layout.
    pub fn to_phys(self, layout: &NLayout) -> PhysicalQubit {
        layout.virt_to_phys[self.index()]
    }
}

/// An unsigned integer Vector based layout class
///
/// This class tracks the layout (or mapping between virtual qubits in the the
/// circuit and physical qubits on the physical device) efficiently
///
/// Args:
///     qubit_indices (dict): A dictionary mapping the virtual qubit index in the circuit to the
///         physical qubit index on the coupling graph.
///     logical_qubits (int): The number of logical qubits in the layout
///     physical_qubits (int): The number of physical qubits in the layout
#[cfg_attr(feature = "py", pyclass(module = "qiskit._accelerate.nlayout"))]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NLayout {
    virt_to_phys: Vec<PhysicalQubit>,
    phys_to_virt: Vec<VirtualQubit>,
}

#[cfg(feature = "py")]
#[pymethods]
impl NLayout {
    #[new]
    fn py_new(
        qubit_indices: HashMap<VirtualQubit, PhysicalQubit>,
        virtual_qubits: usize,
        physical_qubits: usize,
    ) -> Self {
        Self::new(qubit_indices, virtual_qubits, physical_qubits)
    }

    fn __reduce__(&self, py: Python) -> PyResult<Py<PyAny>> {
        (
            py.get_type::<Self>().getattr("from_virtual_to_physical")?,
            (self.virt_to_phys.clone().into_pyobject(py)?,),
        )
            .into_py_any(py)
    }

    /// Return the layout mapping.
    ///
    /// .. note::
    ///
    ///     This copies the data from Rust to Python and has linear overhead based on the number of
    ///     qubits.
    ///
    /// Returns:
    ///     list: A list of 2 element lists in the form ``[(virtual_qubit, physical_qubit), ...]``,
    ///     where the virtual qubit is the index in the qubit index in the circuit.
    ///
    #[pyo3(text_signature = "(self, /)")]
    fn layout_mapping(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        PyList::new(py, self.iter_virtual()).map(|x| x.unbind())
    }

    /// Get physical bit from virtual bit
    #[pyo3(text_signature = "(self, virtual, /)", name = "virtual_to_physical")]
    pub fn py_virtual_to_physical(&self, r#virtual: VirtualQubit) -> PhysicalQubit {
        self.virtual_to_physical(r#virtual)
    }

    /// Get virtual bit from physical bit
    #[pyo3(text_signature = "(self, physical, /)", name = "physical_to_virtual")]
    pub fn py_physical_to_virtual(&self, physical: PhysicalQubit) -> VirtualQubit {
        self.physical_to_virtual(physical)
    }

    /// Swap the specified virtual qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)", name = "swap_virtual")]
    pub fn py_swap_virtual(&mut self, bit_a: VirtualQubit, bit_b: VirtualQubit) {
        self.swap_virtual(bit_a, bit_b)
    }

    /// Swap the specified physical qubits
    #[pyo3(text_signature = "(self, bit_a, bit_b, /)", name = "swap_physical")]
    pub fn py_swap_physical(&mut self, bit_a: PhysicalQubit, bit_b: PhysicalQubit) {
        self.swap_physical(bit_a, bit_b)
    }

    pub fn copy(&self) -> NLayout {
        self.clone()
    }

    #[staticmethod]
    #[pyo3(name = "generate_trivial_layout")]
    pub fn py_generate_trivial_layout(num_qubits: u32) -> Self {
        Self::generate_trivial_layout(num_qubits)
    }

    #[staticmethod]
    #[pyo3(name = "from_virtual_to_physical")]
    pub fn py_from_virtual_to_physical(
        virt_to_phys: Vec<PhysicalQubit>,
    ) -> Result<Self, TryFromIntError> {
        Self::from_virtual_to_physical(virt_to_phys)
    }

    #[staticmethod]
    #[pyo3(name = "from_physical_to_virtual")]
    pub fn py_from_physical_to_virtual(phys_to_virt: Vec<VirtualQubit>) -> PyResult<Self> {
        Ok(Self::from_physical_to_virtual(phys_to_virt)?)
    }
}

impl NLayout {
    fn new(
        qubit_indices: HashMap<VirtualQubit, PhysicalQubit>,
        virtual_qubits: usize,
        physical_qubits: usize,
    ) -> Self {
        let mut res = NLayout {
            virt_to_phys: vec![PhysicalQubit(u32::MAX); virtual_qubits],
            phys_to_virt: vec![VirtualQubit(u32::MAX); physical_qubits],
        };
        for (virt, phys) in qubit_indices {
            res.virt_to_phys[virt.index()] = phys;
            res.phys_to_virt[phys.index()] = virt;
        }
        res
    }

    /// Get physical bit from virtual bit
    pub fn virtual_to_physical(&self, r#virtual: VirtualQubit) -> PhysicalQubit {
        self.virt_to_phys[r#virtual.index()]
    }

    /// Get virtual bit from physical bit
    pub fn physical_to_virtual(&self, physical: PhysicalQubit) -> VirtualQubit {
        self.phys_to_virt[physical.index()]
    }

    /// Swap the specified virtual qubits
    pub fn swap_virtual(&mut self, bit_a: VirtualQubit, bit_b: VirtualQubit) {
        self.virt_to_phys.swap(bit_a.index(), bit_b.index());
        self.phys_to_virt[self.virt_to_phys[bit_a.index()].index()] = bit_a;
        self.phys_to_virt[self.virt_to_phys[bit_b.index()].index()] = bit_b;
    }

    /// Swap the specified physical qubits
    pub fn swap_physical(&mut self, bit_a: PhysicalQubit, bit_b: PhysicalQubit) {
        self.phys_to_virt.swap(bit_a.index(), bit_b.index());
        self.virt_to_phys[self.phys_to_virt[bit_a.index()].index()] = bit_a;
        self.virt_to_phys[self.phys_to_virt[bit_b.index()].index()] = bit_b;
    }

    pub fn generate_trivial_layout(num_qubits: u32) -> Self {
        NLayout {
            virt_to_phys: (0..num_qubits).map(PhysicalQubit).collect(),
            phys_to_virt: (0..num_qubits).map(VirtualQubit).collect(),
        }
    }

    pub fn from_virtual_to_physical(
        virt_to_phys: Vec<PhysicalQubit>,
    ) -> Result<Self, TryFromIntError> {
        let mut phys_to_virt = vec![VirtualQubit(u32::MAX); virt_to_phys.len()];
        for (virt, phys) in virt_to_phys.iter().enumerate() {
            phys_to_virt[phys.index()] = VirtualQubit(virt.try_into()?);
        }
        Ok(NLayout {
            virt_to_phys,
            phys_to_virt,
        })
    }

    pub fn from_physical_to_virtual(
        phys_to_virt: Vec<VirtualQubit>,
    ) -> Result<Self, TryFromIntError> {
        let mut virt_to_phys = vec![PhysicalQubit(u32::MAX); phys_to_virt.len()];
        for (phys, virt) in phys_to_virt.iter().enumerate() {
            virt_to_phys[virt.index()] = PhysicalQubit(phys.try_into()?);
        }
        Ok(NLayout {
            virt_to_phys,
            phys_to_virt,
        })
    }

    /// Iterator of `(VirtualQubit, PhysicalQubit)` pairs, in order of the `VirtualQubit` indices.
    pub fn iter_virtual(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = (VirtualQubit, PhysicalQubit)> + '_ {
        self.virt_to_phys
            .iter()
            .enumerate()
            .map(|(v, p)| (VirtualQubit::new(v as u32), *p))
    }
    /// Iterator of `(PhysicalQubit, VirtualQubit)` pairs, in order of the `PhysicalQubit` indices.
    pub fn iter_physical(
        &'_ self,
    ) -> impl ExactSizeIterator<Item = (PhysicalQubit, VirtualQubit)> + '_ {
        self.phys_to_virt
            .iter()
            .enumerate()
            .map(|(p, v)| (PhysicalQubit::new(p as u32), *v))
    }

    pub fn num_qubits(&self) -> usize {
        self.virt_to_phys.len()
    }

    /// Destructure `self` into the two backing arrays.
    pub fn take(self) -> (Vec<PhysicalQubit>, Vec<VirtualQubit>) {
        (self.virt_to_phys, self.phys_to_virt)
    }

    /// Directly create a new layout without sanity checking the coherence of the inputs.
    ///
    /// This is the opposite of [take].
    pub fn from_vecs_unchecked(
        virt_to_phys: Vec<PhysicalQubit>,
        phys_to_virt: Vec<VirtualQubit>,
    ) -> Self {
        Self {
            virt_to_phys,
            phys_to_virt,
        }
    }
}
impl ::std::ops::Index<VirtualQubit> for NLayout {
    type Output = PhysicalQubit;
    fn index(&self, index: VirtualQubit) -> &PhysicalQubit {
        &self.virt_to_phys[index.index()]
    }
}
impl ::std::ops::Index<PhysicalQubit> for NLayout {
    type Output = VirtualQubit;
    fn index(&self, index: PhysicalQubit) -> &VirtualQubit {
        &self.phys_to_virt[index.index()]
    }
}

#[cfg(feature = "py")]
pub fn nlayout(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<NLayout>()?;
    Ok(())
}
