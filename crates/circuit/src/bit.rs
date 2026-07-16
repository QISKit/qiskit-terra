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

//! Definitions of the shareable bit types and registers.

use std::{
    fmt::Debug,
    hash::Hash,
    sync::Arc,
    sync::atomic::{AtomicU32, AtomicU64, Ordering},
};

use hashbrown::HashSet;
use pyo3::prelude::*;
use pyo3::{
    IntoPyObjectExt, PyTypeInfo,
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    types::{PyList, PyType},
};

use crate::circuit_data::CircuitError;
use crate::dag_circuit::PyBitLocations;
use qiskit_util::py::{PySequenceIndex, SequenceIndex};

/// Describes a relationship between a bit and all the registers it belongs to
#[derive(Debug, Clone)]
pub struct BitLocations<R: Register> {
    pub(crate) index: u32,
    registers: Vec<(R, usize)>,
}

impl<R: Register + PartialEq> BitLocations<R> {
    /// Creates new instance of [BitLocations]
    pub fn new<T: IntoIterator<Item = (R, usize)>>(index: u32, registers: T) -> Self {
        Self {
            index,
            registers: registers.into_iter().collect(),
        }
    }

    /// Adds a register entry
    pub fn add_register(&mut self, register: R, index: usize) {
        self.registers.push((register, index))
    }

    /// Removes a register location in `O(n)`` time, where N is the number of
    /// registers in this entry.
    pub fn remove_register(&mut self, register: &R, index: usize) -> Option<(R, usize)> {
        for (idx, reg) in self.registers.iter().enumerate() {
            if (&reg.0, &reg.1) == (register, &index) {
                let res = self.registers.remove(idx);
                return Some(res);
            }
        }
        None
    }

    pub fn index(&self) -> u32 {
        self.index
    }

    pub fn registers(&self) -> &[(R, usize)] {
        &self.registers
    }
}

impl<'py, R> IntoPyObject<'py> for BitLocations<R>
where
    R: Debug + Clone + Register + for<'a> IntoPyObject<'a>,
{
    type Target = PyBitLocations;
    type Output = Bound<'py, PyBitLocations>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PyBitLocations::new(
            self.index as usize,
            self.registers
                .into_pyobject(py)?
                .cast_into::<PyList>()?
                .unbind(),
        )
        .into_pyobject(py)
    }
}

impl<'a, 'py, R> FromPyObject<'a, 'py> for BitLocations<R>
where
    R: Debug
        + Clone
        + Register
        + IntoPyObject<'py>
        + for<'b> FromPyObject<'b, 'py, Error: Into<PyErr>>,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, PyErr> {
        let ob_down = ob.cast::<PyBitLocations>()?.borrow();
        Ok(Self {
            index: ob_down.index as u32,
            registers: ob_down.registers.bind(ob.py()).extract()?,
        })
    }
}

/// Main representation of the inner properties of a shareable `Bit` object.
///
/// This is supplemented by an extra marker type to encode additional subclass information for
/// communication with Python space, which is used to distinguish an AncillaQubit from a Qubit
/// which is a Python domain construct only for backwards compatibility, but we don't need in
/// rust.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
enum BitInfo<B> {
    Owned {
        register: Arc<OwningRegisterInfo<B>>,
        index: u32,
    },
    Anonymous {
        /// Unique id for bit, derives from [ShareableBit::anonymous_instance_count]
        uid: u64,
        subclass: B,
    },
}
impl<B> BitInfo<B> {
    /// Which subclass the bit is.
    #[inline]
    fn subclass(&self) -> &B {
        match self {
            BitInfo::Owned { register, .. } => &register.subclass,
            BitInfo::Anonymous { subclass, .. } => subclass,
        }
    }
}

// The trait bounds aren't _strictly_ necessary, but they simplify a lot of later bounds.
pub trait ShareableBit: Clone + Eq + Hash + Debug {
    type Subclass: Copy + Eq + Hash + Debug + Default;
    // A description of the bit class which is used for error messages
    const DESCRIPTION: &'static str;
}
// An internal trait to let `RegisterInfo` manifest full `ShareableBit` instances from `BitInfo`
// structs without leaking that implementation detail into the public.
trait ManifestableBit: ShareableBit {
    fn from_info(val: BitInfo<<Self as ShareableBit>::Subclass>) -> Self;
    fn info(&self) -> &BitInfo<<Self as ShareableBit>::Subclass>;
}

/// Implement a generic bit.
///
/// .. note::
///     This class cannot be instantiated directly. Its only purpose is to allow generic type
///     checking for :class:`.Clbit` and :class:`.Qubit`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pyclass(
    subclass,
    name = "Bit",
    module = "qiskit.circuit",
    frozen,
    skip_from_py_object
)]
pub struct PyBit;
/// Implement a generic register.
///
/// .. note::
///     This class cannot be instantiated directly.  Its only purpose is to allow generic type
///     checking for :class:`~.ClassicalRegister` and :class:`~.QuantumRegister`.
#[pyclass(
    name = "Register",
    module = "qiskit.circuit",
    subclass,
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct PyRegister;

/// Contains the information for a register that owns the bits it contains.
///
/// This is separate to the full [RegisterInfo] because owned bits also need to store a
/// backreference to their owning register.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct OwningRegisterInfo<S> {
    name: String,
    size: u32,
    subclass: S,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RegisterInfo<B: ShareableBit> {
    Owning(Arc<OwningRegisterInfo<<B as ShareableBit>::Subclass>>),
    Alias {
        name: String,
        bits: Vec<B>,
        subclass: <B as ShareableBit>::Subclass,
    },
}
impl<B: ShareableBit> RegisterInfo<B> {
    /// The name of the register.
    pub fn name(&self) -> &str {
        match self {
            Self::Owning(reg) => &reg.name,
            Self::Alias { name, .. } => name,
        }
    }
    /// The length of the register.
    pub fn len(&self) -> usize {
        match self {
            Self::Owning(reg) => reg.size as usize,
            Self::Alias { bits, .. } => bits.len(),
        }
    }
    /// Is the register empty?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Which subclass does the register correspond to?
    fn subclass(&self) -> &<B as ShareableBit>::Subclass {
        match self {
            Self::Owning(reg) => &reg.subclass,
            Self::Alias { subclass, .. } => subclass,
        }
    }
}
impl<B: ManifestableBit> RegisterInfo<B> {
    /// What is the index of this bit in the register, if any?
    pub fn index_of(&self, bit: &B) -> Option<usize> {
        match self {
            Self::Owning(our_reg) => {
                let BitInfo::Owned {
                    register: bit_reg,
                    index,
                } = bit.info()
                else {
                    return None;
                };
                (*our_reg == *bit_reg).then_some(*index as usize)
            }
            Self::Alias { bits, .. } => bits.iter().position(|other| *other == *bit),
        }
    }
    /// Does this register contain this bit?
    pub fn contains(&self, bit: &B) -> bool {
        self.index_of(bit).is_some()
    }
    /// Get the bit at the given index, if in range.
    fn get(&self, index: usize) -> Option<B> {
        match self {
            RegisterInfo::Owning(reg) => (index < (reg.size as usize)).then(|| {
                B::from_info(BitInfo::Owned {
                    register: reg.clone(),
                    index: index as u32,
                })
            }),
            RegisterInfo::Alias { bits, .. } => bits.get(index).cloned(),
        }
    }
    /// Iterate over the bits in the register.
    fn iter(&self) -> RegisterInfoIter<'_, B> {
        RegisterInfoIter {
            base: self,
            index: 0,
        }
    }
}

struct RegisterInfoIter<'a, B: ManifestableBit> {
    base: &'a RegisterInfo<B>,
    index: usize,
}
impl<B: ManifestableBit> Iterator for RegisterInfoIter<'_, B> {
    type Item = B;
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.base.get(self.index);
        if out.is_some() {
            self.index += 1;
        }
        out
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.base.len().saturating_sub(self.index);
        (rem, Some(rem))
    }
}
impl<B: ManifestableBit> ExactSizeIterator for RegisterInfoIter<'_, B> {}

pub trait Register {
    /// The type of bit stored by the [Register]
    type Bit;

    /// Returns the size of the [Register].
    fn len(&self) -> usize;
    /// Checks if the [Register] is empty.
    fn is_empty(&self) -> bool;
    /// Returns the name of the [Register].
    fn name(&self) -> &str;
    /// Checks if a bit exists within the [Register].
    fn contains(&self, bit: &Self::Bit) -> bool;
    /// Return an iterator over all the bits in the register
    fn bits(&self) -> impl ExactSizeIterator<Item = Self::Bit>;
    /// Gets a bit by index
    fn get(&self, index: usize) -> Option<Self::Bit>;
    /// Checks if the register is owning or alias
    fn is_owning(&self) -> bool;
}

/// Create a (Bit, Register) pair, and the associated Python objects.
///
/// # Args
///
/// * `bit_struct` - Identifier for new bit struct, for example, `ShareableQubit`.
/// * `subclass_ty` - Subclass type used for determining parent bit type, typically an enum to
///   select between different bit types (for qubit this is Qubit vs AncillaQubit). See
///   [QubitSubclass] and [ClbitSubclass] for typical types used here.
/// * `pybit_struct` - Identifier for python bit struct name, for example `PyQubit`.
/// * `pybit_name` - Python space class name for `pybit_struct`, for example `"Qubit"`.
/// * `bit_desc` - &'static str used as a name for describing bits, typically only used in error
///   messages to describe the bit. For example,   "qubit",
/// * `reg_struct` - Identifier for rust register struct name, for example `QuantumRegister`
/// * `pyreg_struct` - Identifier for python register struct, for example `PyQuantumRegister`
/// * `pyreg_name` - Python space class name for `pyreg_struct. For example, `"QuantumRegister"`.
/// * `pyreg_prefix` - String prefix for python space registers. Normally only `"q"` or `"c"`.
/// * `bit_counter_name` - Identifier to use for global static atomic counter of anonymous bits
///   created. For example, `QUBIT_INSTANCES`.
/// * `reg_counter_name` - Identifier to use for global static atomic counter of anonymous
///   registers create. For example, `QUANTUM_REGISTER_INSTANCES`.
macro_rules! create_bit_object {
    (
        $bit_struct:ident,
        $subclass_ty:ty,
        $pybit_struct:ident,
        $pybit_name:expr,
        $bit_desc:expr,
        $reg_struct:ident,
        $pyreg_struct:ident,
        $pyreg_name:expr,
        $pyreg_prefix:expr,
        $bit_counter_name:ident,
        $reg_counter_name:ident
    ) => {
        /// Global counter for the number of anonymous bits created.
        static $bit_counter_name: AtomicU64 = AtomicU64::new(0);
        /// Global counter for the number of anonymous register created.
        static $reg_counter_name: AtomicU32 = AtomicU32::new(0);

        /// A representation of a bit that can be shared between circuits, and allows linking
        /// corresponding bits between two circuits.
        ///
        /// These objects are comparable in a global sense, unlike the lighter [Qubit] or [Clbit]
        /// index-like objects used only _within_ a circuit.  We use these objects when comparing
        /// two circuits to each other, and resolving Python objects, but within the context of a
        /// circuit, we just use the simple indices.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $bit_struct(BitInfo<$subclass_ty>);
        impl $bit_struct {
            #[inline]
            fn anonymous_instance_count() -> &'static AtomicU64 {
                &$bit_counter_name
            }

            /// Which subclass the bit is.
            #[inline]
            pub fn subclass(&self) -> &$subclass_ty {
                self.0.subclass()
            }

            /// Create a new anonymous bit.
            #[inline]
            pub fn new_anonymous() -> Self {
                Self(BitInfo::Anonymous {
                    uid: Self::anonymous_instance_count().fetch_add(1, Ordering::Relaxed),
                    subclass: Default::default(),
                })
            }

            /// Return the register owning the bit, if it exists
            pub fn owning_register(&self) -> Option<$reg_struct> {
                match &self.0 {
                    BitInfo::Owned { register, .. } => Some($reg_struct(Arc::new(
                        RegisterInfo::Owning(register.clone()),
                    ))),
                    BitInfo::Anonymous { .. } => None,
                }
            }

            /// Return the index of the bit in its owning register, if it exists
            pub fn owning_register_index(&self) -> Option<u32> {
                match &self.0 {
                    BitInfo::Owned { index, .. } => Some(*index),
                    BitInfo::Anonymous { .. } => None,
                }
            }

            /// Create `count` new anonymous bits, returned as an iterator.
            ///
            /// This is a bit more efficient than creating the bits individually, since we only have
            /// to update the underlying atomic instance count once.
            #[inline]
            pub fn iter_anonymous(count: u32) -> impl ExactSizeIterator<Item = Self> {
                let base = Self::anonymous_instance_count()
                    .fetch_add(count as u64, Ordering::Relaxed);
                (0..count).map(move |offset| {
                    Self(BitInfo::Anonymous {
                        uid: base + offset as u64,
                        subclass: Default::default(),
                    })
                })
            }
        }

        impl ShareableBit for $bit_struct {
            type Subclass = $subclass_ty;
            const DESCRIPTION: &'static str = $bit_desc;
        }
        impl ManifestableBit for $bit_struct {
            fn from_info(val: BitInfo<<$bit_struct as ShareableBit>::Subclass>) -> Self {
                Self(val)
            }
            fn info(&self) -> &BitInfo<<$bit_struct as ShareableBit>::Subclass> {
                &self.0
            }
        }

        #[doc = concat!("A ", $bit_desc, ", which can be compared between different circuits.")]
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        #[pyclass(subclass, name=$pybit_name, module="qiskit.circuit", extends=PyBit, frozen, eq, hash, skip_from_py_object)]
        pub struct $pybit_struct($bit_struct);
        #[pymethods]
        impl $pybit_struct {
            /// Create a new bit.
            #[new]
            #[pyo3(signature=(register=None, index=None))]
            fn new(
                register: Option<Bound<$pyreg_struct>>,
                index: Option<u32>,
            ) -> PyResult<PyClassInitializer<Self>> {
                let bit = match (register, index) {
                    (Some(register), Some(index)) => {
                        let register = &register.borrow().0;
                        register.get(index as usize).ok_or_else(|| {
                            PyIndexError::new_err(format!(
                                "index {} out of range for size {}",
                                index,
                                register.len()
                            ))
                        })?
                    }
                    (None, None) => $bit_struct::new_anonymous(),
                    _ => return Err(PyTypeError::new_err(
                        "either both 'register' and 'index' are provided, or neither are",
                    )),
                };
                Ok(PyClassInitializer::from(PyBit).add_subclass(Self(bit)))
            }

            fn __repr__(slf: Bound<Self>) -> PyResult<String> {
                let ob = &slf.borrow().0;
                let name = slf.get_type().qualname()?;
                match &ob.0 {
                    BitInfo::Owned { register, index } => Ok(format!(
                        "<{} register=({}, \"{}\"), index={}>",
                        name, register.size, &register.name, index
                    )),
                    BitInfo::Anonymous { uid, .. } => Ok(format!("<{name} uid={uid}>")),
                }
            }
            fn __copy__(slf: PyRef<Self>) -> PyRef<Self> {
                // Bits are immutable.
                slf
            }
            fn __deepcopy__<'py>(
                slf: PyRef<'py, Self>,
                _memo: Bound<'py, PyAny>,
            ) -> PyRef<'py, Self> {
                // Everything a bit contains is immutable.
                slf
            }

            fn __reduce__(slf: Bound<Self>) -> PyResult<Bound<PyAny>> {
                // This is deliberately type-erasing up top, so `AncillaQubit` can override the
                // constructor methods.
                let ty = slf.get_type();
                match &slf.borrow().0.0 {
                    BitInfo::Owned { register, index } => (
                        ty.getattr("_from_owned")?,
                        (register.name.to_owned(), register.size, index),
                    )
                        .into_bound_py_any(slf.py()),
                    // Don't need to examine the subclass, because it's handled by the overrides of
                    // the `_from_anonymous` and `_from_owned` methods.
                    BitInfo::Anonymous { uid, .. } => {
                        (ty.getattr("_from_anonymous")?, (uid,)).into_bound_py_any(slf.py())
                    }
                }
            }
            // Used by pickle, overridden in subclasses.
            #[staticmethod]
            fn _from_anonymous(py: Python, uid: u64) -> PyResult<Bound<PyAny>> {
                Ok(Bound::new(
                    py,
                    PyClassInitializer::from((
                        Self($bit_struct(BitInfo::Anonymous {
                            uid,
                            // Fine to do this, because `AncillaRegister` overrides the method.
                            subclass: Default::default(),
                        })),
                        PyBit,
                    )),
                )?
                .into_any())
            }
            #[staticmethod]
            fn _from_owned(
                py: Python,
                reg_name: String,
                reg_size: u32,
                index: u32,
            ) -> PyResult<Bound<PyAny>> {
                // This doesn't feel like the most efficient way - in a big list of owned qubits,
                // we'll pickle the register information many times - but good enough for now.
                let register = Arc::new(OwningRegisterInfo {
                    name: reg_name,
                    size: reg_size,
                    // Fine to do this, because `AncillaRegister` overrides the method.
                    subclass: Default::default(),
                });
                Ok(Bound::new(
                    py,
                    PyClassInitializer::from((
                        Self($bit_struct(BitInfo::Owned { register, index })),
                        PyBit,
                    )),
                )?
                .into_any())
            }

            // Legacy getters to keep Python-space QPY happy.
            #[getter]
            fn _register(&self) -> Option<$reg_struct> {
                match &self.0.0 {
                    BitInfo::Owned { register, .. } => Some($reg_struct(Arc::new(
                        RegisterInfo::Owning(register.clone()),
                    ))),
                    BitInfo::Anonymous { .. } => None,
                }
            }
            #[getter]
            fn _index(&self) -> Option<u32> {
                match &self.0.0 {
                    BitInfo::Owned { index, .. } => Some(*index),
                    BitInfo::Anonymous { .. } => None,
                }
            }
        }

        impl<'a, 'py> FromPyObject<'a, 'py> for $bit_struct {
            type Error = ::pyo3::CastError<'a, 'py>;

            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
                ob.cast::<$pybit_struct>().map(|ob| ob.borrow().0.clone())
            }
        }
        // The owning impl of `IntoPyObject` needs to be done manually, to better handle
        // subclassing.
        impl<'a, 'py> IntoPyObject<'py> for &'a $bit_struct {
            type Target = <$bit_struct as IntoPyObject<'py>>::Target;
            type Output = <$bit_struct as IntoPyObject<'py>>::Output;
            type Error = <$bit_struct as IntoPyObject<'py>>::Error;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                self.clone().into_pyobject(py)
            }
        }

        /// A Rust-space register object.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $reg_struct(Arc<RegisterInfo<$bit_struct>>);
        impl $reg_struct {
            #[inline]
            fn anonymous_instance_count() -> &'static AtomicU32 {
                &$reg_counter_name
            }

            /// Which subclass the register is.
            #[inline]
            pub fn subclass(&self) -> &<$bit_struct as ShareableBit>::Subclass {
                self.0.subclass()
            }

            /// Create a new owning register.
            #[inline]
            pub fn new_owning<S: Into<String>>(name: S, size: u32) -> Self {
                Self(Arc::new(RegisterInfo::Owning(Arc::new(
                    OwningRegisterInfo {
                        name: name.into(),
                        size,
                        subclass: Default::default(),
                    },
                ))))
            }

            /// Create a new aliasing register.
            #[inline]
            pub fn new_alias(name: Option<String>, bits: Vec<$bit_struct>) -> Self {
                let name = name.unwrap_or_else(|| {
                    format!(
                        "{}{}",
                        $pyreg_prefix,
                        $reg_struct::anonymous_instance_count().fetch_add(1, Ordering::Relaxed)
                    )
                });
                Self(Arc::new(RegisterInfo::Alias {
                    name,
                    bits,
                    // This assumes that `B::default()` returns the base class.
                    subclass: Default::default(),
                }))
            }

            /// Get the name of the register.
            #[inline]
            pub fn name(&self) -> &str {
                self.0.name()
            }
            /// Get the length of the register.
            #[inline]
            pub fn len(&self) -> usize {
                self.0.len()
            }
            /// Is the register empty?
            #[inline]
            pub fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
            /// Get the bit at the given index, if in range.
            #[inline]
            pub fn get(&self, index: usize) -> Option<$bit_struct> {
                self.0.get(index)
            }
            /// Is this bit in this register?
            #[inline]
            pub fn contains(&self, bit: &$bit_struct) -> bool {
                self.0.contains(bit)
            }
            /// What of the index of this bit in this register, if any?
            #[inline]
            pub fn index_of(&self, bit: &$bit_struct) -> Option<usize> {
                self.0.index_of(bit)
            }
            /// Iterate over the bits in the register.
            #[inline]
            pub fn iter(&self) -> impl ExactSizeIterator<Item = $bit_struct> + '_ {
                self.0.iter()
            }
        }

        impl Register for $reg_struct {
            type Bit = $bit_struct;

            fn len(&self) -> usize {
                self.0.len()
            }
            fn is_empty(&self) -> bool {
                self.0.is_empty()
            }
            fn name(&self) -> &str {
                self.0.name()
            }
            fn contains(&self, bit: &Self::Bit) -> bool {
                self.0.contains(bit)
            }
            fn bits(&self) -> impl ExactSizeIterator<Item = $bit_struct> {
                self.iter()
            }
            fn get(&self, index: usize) -> Option<Self::Bit> {
                self.0.get(index)
            }
            fn is_owning(&self) -> bool {
                match self.0.as_ref() {
                    RegisterInfo::Owning(_) => true,
                    RegisterInfo::Alias { .. } => false,
                }
            }
        }

        impl<'a, 'py> FromPyObject<'a, 'py> for $reg_struct {
            type Error = ::pyo3::CastError<'a, 'py>;

            fn extract(ob: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
                ob.cast::<$pyreg_struct>().map(|ob| ob.borrow().0.clone())
            }
        }
        // The owning impl of `IntoPyObject` needs to be done manually, to better handle
        // subclassing.
        impl<'a, 'py> IntoPyObject<'py> for &'a $reg_struct {
            type Target = <$reg_struct as IntoPyObject<'py>>::Target;
            type Output = <$reg_struct as IntoPyObject<'py>>::Output;
            type Error = <$reg_struct as IntoPyObject<'py>>::Error;

            fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
                self.clone().into_pyobject(py)
            }
        }

        /// Implement a register.
        #[pyclass(subclass, name=$pyreg_name, module="qiskit.circuit", extends=PyRegister, frozen, eq, hash, sequence, skip_from_py_object)]
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub struct $pyreg_struct($reg_struct);

        #[pymethods]
        impl $pyreg_struct {
            /// Create a new register.
            ///
            /// Either the ``size`` or the ``bits`` argument must be provided. If
            /// ``size`` is not None, the register will be pre-populated with bits of the
            /// correct type.
            ///
            /// Args:
            ///     size (int): Optional. The number of bits to include in the register.
            ///     name (str): Optional. The name of the register. If not provided, a
            ///         unique name will be auto-generated from the register type.
            ///     bits (list[Bit]): Optional. A list of :class:`.Bit` instances to be used to
            ///         populate the register.
            ///
            /// Raises:
            ///     CircuitError: if any of:
            ///
            ///         * both the ``size`` and ``bits`` arguments are provided, or if neither are.
            ///         * ``size`` is not valid.
            ///         * ``bits`` contained duplicated bits.
            ///         * ``bits`` contained bits of an incorrect type.
            ///         * ``bits`` exceeds the possible capacity for a register.
            #[pyo3(signature=(size=None, name=None, bits=None))]
            #[new]
            fn py_new(
                size: Option<isize>,
                name: Option<String>,
                bits: Option<Vec<$bit_struct>>,
            ) -> PyResult<PyClassInitializer<Self>> {
                let name = name.unwrap_or_else(|| {
                    format!(
                        "{}{}",
                        Self::prefix(),
                        $reg_struct::anonymous_instance_count().fetch_add(1, Ordering::Relaxed)
                    )
                });
                let register = match (size, bits) {
                    (None, None) | (Some(_), Some(_)) => return Err(CircuitError::new_err(
                        "Exactly one of the size or bits arguments can be provided.",
                    )),
                    (Some(size), None) => {
                        if size < 0 {
                            return Err(CircuitError::new_err(
                                "Register size must be non-negative.",
                            ));
                        }
                        let Ok(size) = size.try_into() else {
                            return Err(CircuitError::new_err("Register size too large."));
                        };
                        $reg_struct::new_owning(name, size)
                    }
                    (None, Some(bits)) => {
                        if bits.iter().cloned().collect::<HashSet<_>>().len() != bits.len() {
                            return Err(CircuitError::new_err(
                                "Register bits must not be duplicated.",
                            ));
                        }
                        $reg_struct::new_alias(Some(name), bits)
                    }
                };
                Ok(PyClassInitializer::from(PyRegister).add_subclass(Self(register)))
            }

            /// The name of the register.
            #[getter]
            fn get_name(&self) -> &str {
                self.0.name()
            }
            /// The size of the register.
            #[getter]
            fn get_size(&self) -> usize {
                self.0.len()
            }

            fn __repr__(&self) -> String {
                format!("{}({}, '{}')", $pyreg_name, self.0.len(), self.0.name())
            }

            fn __contains__(&self, bit: &$pybit_struct) -> bool {
                self.0.contains(&bit.0)
            }

            fn __getnewargs__(&self) -> (Option<isize>, Option<String>, Option<Vec<$bit_struct>>) {
                match &*self.0.0 {
                    RegisterInfo::Owning(reg) => {
                        (Some(reg.size as isize), Some(reg.name.to_owned()), None)
                    }
                    RegisterInfo::Alias { name, bits, .. } => {
                        (None, Some(name.clone()), Some(bits.clone()))
                    }
                }
            }

            // We rely on `__len__` and `__getitem__` for implicit iteration - it's easier than
            // defining a new struct ourselves.
            fn __len__(&self) -> usize {
                self.0.len()
            }
            fn __getitem__<'py>(&self, ob: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                let get_inner = |idx| {
                    self.0
                        .get(idx)
                        .expect("PySequenceIndex always returns valid indices")
                };
                if let Ok(sequence) = ob.extract::<PySequenceIndex>() {
                    match sequence.with_len(self.0.len())? {
                        SequenceIndex::Int(idx) => get_inner(idx).into_bound_py_any(ob.py()),
                        s => Ok(PyList::new(ob.py(), s.into_iter().map(get_inner))?.into_any()),
                    }
                } else if let Ok(list) = ob.cast::<PyList>() {
                    let out = PyList::empty(ob.py());
                    for item in list.iter() {
                        out.append(get_inner(PySequenceIndex::convert_idx(
                            item.extract()?,
                            self.0.len(),
                        )?))?;
                    }
                    Ok(out.into_any())
                } else {
                    Err(PyTypeError::new_err("index must be int, slice or list"))
                }
            }

            /// The index of the given bit in the register.
            fn index(&self, bit: Bound<$pybit_struct>) -> PyResult<usize> {
                let bit_inner = bit.borrow();
                self.0
                    .index_of(&bit_inner.0)
                    .ok_or_else(|| match bit.repr() {
                        Ok(repr) => {
                            PyValueError::new_err(format!("Bit {repr} not found in register."))
                        }
                        Err(err) => err,
                    })
            }

            /// Allows for the creation of a new register with a temporary prefix and the
            /// same instance counter.
            #[pyo3(signature=(size=None, name=None, bits=None))]
            #[staticmethod]
            fn _new_with_prefix(
                py: Python,
                size: Option<isize>,
                name: Option<String>,
                bits: Option<Vec<$bit_struct>>,
            ) -> PyResult<Py<Self>> {
                let name = format!(
                    "{}{}",
                    name.unwrap_or(Self::prefix().to_string()),
                    $reg_struct::anonymous_instance_count().fetch_add(1, Ordering::Relaxed)
                );
                Py::new(py, Self::py_new(size, Some(name), bits)?)
            }

            #[classattr]
            fn prefix() -> &'static str {
                $pyreg_prefix
            }
            #[classattr]
            fn bit_type(py: Python) -> Bound<PyType> {
                $pybit_struct::type_object(py)
            }
            #[classattr]
            fn instances_count() -> u32 {
                $reg_struct::anonymous_instance_count().load(Ordering::Relaxed)
            }
        }

        impl ::std::ops::Deref for $pyreg_struct {
            type Target = $reg_struct;
            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
    };
}

/// Which subclass a [ShareableQubit] is.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, Default)]
pub enum QubitSubclass {
    /// The base Python-space ``Qubit`` class.
    #[default]
    QUBIT,
    /// The subclass ``AncillaQubit``.
    ANCILLA,
}
create_bit_object!(
    ShareableQubit,
    QubitSubclass,
    PyQubit,
    "Qubit",
    "qubit",
    QuantumRegister,
    PyQuantumRegister,
    "QuantumRegister",
    "q",
    QUBIT_INSTANCES,
    QUANTUM_REGISTER_INSTANCES
);
impl ShareableQubit {
    /// Create a new anonymous ancilla qubit.
    ///
    /// Qubits owned by registers can only be created *by* registers.
    pub fn new_anonymous_ancilla() -> Self {
        Self(BitInfo::Anonymous {
            uid: Self::anonymous_instance_count().fetch_add(1, Ordering::Relaxed),
            subclass: QubitSubclass::ANCILLA,
        })
    }

    /// Is this qubit an ancilla?
    #[inline]
    pub fn is_ancilla(&self) -> bool {
        *self.subclass() == QubitSubclass::ANCILLA
    }
}

/// A qubit used as an ancilla.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[pyclass(name="AncillaQubit", module="qiskit.circuit", extends=PyQubit, frozen, skip_from_py_object)]
pub struct PyAncillaQubit;
#[pymethods]
impl PyAncillaQubit {
    /// Create a new anonymous ancilla qubit.
    #[new]
    fn new() -> PyClassInitializer<Self> {
        PyClassInitializer::from(PyBit)
            .add_subclass(PyQubit(ShareableQubit::new_anonymous_ancilla()))
            .add_subclass(Self)
    }

    fn __hash__(slf: PyRef<Self>) -> PyResult<isize> {
        let py = slf.py();
        let qubit: &PyQubit = &slf.into_super();
        Bound::new(py, (qubit.clone(), PyBit))?.hash()
    }

    fn __eq__(slf: PyRef<Self>, other: PyRef<Self>) -> bool {
        slf.as_super().eq(other.as_super())
    }

    // Pickle overrides.
    #[staticmethod]
    fn _from_anonymous(py: Python, uid: u64) -> PyResult<Bound<PyAny>> {
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyBit)
                .add_subclass(PyQubit(ShareableQubit(BitInfo::Anonymous {
                    uid,
                    subclass: QubitSubclass::ANCILLA,
                })))
                .add_subclass(PyAncillaQubit),
        )?
        .into_any())
    }
    #[staticmethod]
    fn _from_owned(
        py: Python,
        reg_name: String,
        reg_size: u32,
        index: u32,
    ) -> PyResult<Bound<PyAny>> {
        // This doesn't feel like the most efficient way - in a big list of owned qubits,
        // we'll pickle the register information many times - but good enough for now.
        let register = Arc::new(OwningRegisterInfo {
            name: reg_name,
            size: reg_size,
            subclass: QubitSubclass::ANCILLA,
        });
        Ok(Bound::new(
            py,
            PyClassInitializer::from(PyBit)
                .add_subclass(PyQubit(ShareableQubit(BitInfo::Owned { register, index })))
                .add_subclass(PyAncillaQubit),
        )?
        .into_any())
    }
}
impl<'py> IntoPyObject<'py> for ShareableQubit {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        if self.is_ancilla() {
            Ok(Bound::new(
                py,
                PyClassInitializer::from(PyBit)
                    .add_subclass(PyQubit(self))
                    .add_subclass(PyAncillaQubit),
            )?
            .into_any())
        } else {
            Ok(Bound::new(py, (PyQubit(self), PyBit))?.into_any())
        }
    }
}

impl QuantumRegister {
    /// Create a new ancilla register that owns its qubits.
    pub fn new_ancilla_owning(name: String, size: u32) -> Self {
        Self(Arc::new(RegisterInfo::Owning(Arc::new(
            OwningRegisterInfo {
                name,
                size,
                subclass: QubitSubclass::ANCILLA,
            },
        ))))
    }

    /// Create a new ancilla register that aliases other bits.
    ///
    /// Returns `None` if not all the bits are ancillas.
    pub fn new_ancilla_alias(name: String, bits: Vec<ShareableQubit>) -> Option<Self> {
        bits.iter().all(|bit| bit.is_ancilla()).then(|| {
            Self(Arc::new(RegisterInfo::Alias {
                name,
                bits,
                subclass: QubitSubclass::ANCILLA,
            }))
        })
    }

    /// Is this an ancilla register?
    ///
    /// Non-ancilla registers can still contain ancilla qubits, but not the other way around.
    pub fn is_ancilla(&self) -> bool {
        *self.subclass() == QubitSubclass::ANCILLA
    }
}
// This isn't intended for use from Rust space.
/// Implement an ancilla register.
#[pyclass(
    name = "AncillaRegister",
    module = "qiskit.circuit",
    frozen,
    extends=PyQuantumRegister,
    skip_from_py_object,
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PyAncillaRegister;
#[pymethods]
impl PyAncillaRegister {
    // Most of the methods are inherited from `QuantumRegister` in Python space.

    #[pyo3(signature=(size=None, name=None, bits=None))]
    #[new]
    fn py_new(
        size: Option<isize>,
        name: Option<String>,
        bits: Option<Vec<ShareableQubit>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let name = name.unwrap_or_else(|| {
            format!(
                "{}{}",
                Self::prefix(),
                QuantumRegister::anonymous_instance_count().fetch_add(1, Ordering::Relaxed)
            )
        });
        let reg = match (size, bits) {
            (None, None) | (Some(_), Some(_)) => {
                return Err(CircuitError::new_err(
                    "Exactly one of the size or bits arguments can be provided.",
                ));
            }
            (Some(size), None) => {
                if size < 0 {
                    return Err(CircuitError::new_err("Register size must be non-negative."));
                }
                let Ok(size) = size.try_into() else {
                    return Err(CircuitError::new_err("Register size too large."));
                };
                QuantumRegister::new_ancilla_owning(name, size)
            }
            (None, Some(bits)) => {
                if bits.iter().cloned().collect::<HashSet<_>>().len() != bits.len() {
                    return Err(CircuitError::new_err(
                        "Register bits must not be duplicated.",
                    ));
                }
                QuantumRegister::new_ancilla_alias(name, bits)
                    .ok_or_else(|| PyTypeError::new_err("all bits must be AncillaQubit"))?
            }
        };
        Ok(PyClassInitializer::from(PyRegister)
            .add_subclass(PyQuantumRegister(reg))
            .add_subclass(Self))
    }

    fn __hash__(slf: PyRef<Self>) -> PyResult<isize> {
        let py = slf.py();
        let qreg: &PyQuantumRegister = &slf.into_super();
        Bound::new(py, (qreg.clone(), PyRegister))?.hash()
    }

    fn __eq__(slf: PyRef<Self>, other: PyRef<Self>) -> bool {
        slf.as_super().eq(other.as_super())
    }

    #[classattr]
    fn prefix() -> &'static str {
        "a"
    }

    #[classattr]
    fn bit_type(py: Python) -> Bound<PyType> {
        PyAncillaQubit::type_object(py)
    }
}
impl<'py> IntoPyObject<'py> for QuantumRegister {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        if self.is_ancilla() {
            Ok(Bound::new(
                py,
                PyClassInitializer::from(PyRegister)
                    .add_subclass(PyQuantumRegister(self))
                    .add_subclass(PyAncillaRegister),
            )?
            .into_any())
        } else {
            Ok(Bound::new(py, (PyQuantumRegister(self), PyRegister))?.into_any())
        }
    }
}

/// Which subclass a [ShareableClbit] is.
///
/// This carries no real data; there's no subclasses of ``Clbit``.  It's mostly used as a data
/// marker and a way to define traits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ClbitSubclass {
    #[default]
    CLBIT,
}
create_bit_object!(
    ShareableClbit,
    ClbitSubclass,
    PyClbit,
    "Clbit",
    "clbit",
    ClassicalRegister,
    PyClassicalRegister,
    "ClassicalRegister",
    "c",
    CLBIT_INSTANCES,
    CLASSICAL_REGISTER_INSTANCES
);
impl<'py> IntoPyObject<'py> for ShareableClbit {
    type Target = PyClbit;
    type Output = Bound<'py, PyClbit>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(py, (PyClbit(self), PyBit))
    }
}

impl<'py> IntoPyObject<'py> for ClassicalRegister {
    type Target = PyClassicalRegister;
    type Output = Bound<'py, PyClassicalRegister>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Bound::new(py, (PyClassicalRegister(self), PyRegister))
    }
}
