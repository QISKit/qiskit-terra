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

//! Dynamic runtime type-information systems.
//!
//! This module contains definitions of Qiskit's system for defining and managing dynamic runtime
//! types that may come from languages or runtimes outside Rust, such as C extensions or Python.

use std::{any, borrow, hash};

/// An identifier for a type that may have additional runtime-defined components in it, such as a
/// dynamic trait implementer that comes from C or Python.
#[derive(Copy, Clone, Debug)]
pub struct DynTypeId<'a> {
    static_id: any::TypeId,
    static_name: &'static str,
    /// The dynamic components of the type information.
    ///
    /// The payload and name are logically tied to some object that creates them.  The pointer, if
    /// used, is valid for the same lifetime as `'a`.
    dynamic: Option<(*mut (), &'a str)>,
}
impl DynTypeId<'_> {
    /// Produce a representation of a [`DynTypeId`] for a type whose implementation is completely
    /// known at Rust compile time.
    ///
    /// Use [`Self::with_dynamic`] to add dynamic context afterwards.
    pub fn of<T: 'static>() -> Self {
        Self {
            static_id: any::TypeId::of::<T>(),
            static_name: any::type_name::<T>(),
            dynamic: None,
        }
    }

    /// Discard the dynamic components of the type, leaving only the static part.
    pub fn to_static(self) -> DynTypeId<'static> {
        DynTypeId {
            dynamic: None,
            ..self
        }
    }

    /// A key object that subsets the fields to define equality and hashing.
    #[inline]
    fn compare_key(&self) -> impl Eq + hash::Hash {
        (self.static_id, self.dynamic.map(|(addr, _)| addr))
    }

    /// Describe this type.
    pub fn describe(&self) -> borrow::Cow<'_, str> {
        match self.dynamic {
            Some((_addr, dyn_name)) => {
                borrow::Cow::Owned(format!("{}[{}]", self.static_name, dyn_name))
            }
            None => borrow::Cow::Borrowed(self.static_name),
        }
    }

    /// A short description of this type.
    ///
    /// This differs from [`Self::describe`] in that it includes fewer qualifiers. When the
    /// type is dynamic, it omits the description of the static type as a wrapper, for example,
    /// leads to `CustomIR` rather than `qiskit_pyext::passmanager::PyIr[CustomIR]` for a
    /// Python-defined IR. When the type is static, it omits the crate path. This method is
    /// used in debugging printouts and error messages and its conventions are therefore
    /// subject to change without notice.
    pub fn short_describe(&self) -> &str {
        match self.dynamic {
            Some((_addr, dyn_name)) => dyn_name,
            None => {
                // Only the leading path, up to the first generic parameter, is certain to be a
                // module path. Stopping there keeps the split from landing somewhere meaningless,
                // at the cost of leaving any path within the generic parameters qualified.
                let head = self
                    .static_name
                    .split_once('<')
                    .map_or(self.static_name, |(head, _)| head);
                match head.rfind("::") {
                    Some(split) => &self.static_name[split + 2..],
                    None => self.static_name,
                }
            }
        }
    }
}
impl<'a> DynTypeId<'a> {
    /// Set the dynamic components of the type identifier.
    ///
    /// The combination of the Rust type `T` and the address of `payload` is what uniquely defines
    /// the "dynamic type".  If you are using this object to represent a pure type from (say)
    /// Python, you might want to use the pointer to the object's Python `type`.  If you are
    /// representing a dynamic implementation of some trait coming in from C, you probably want to
    /// use a pointer to the vtable of the trait methods.
    ///
    /// Note that the `name` is purely for human inspectability and plays no part in hashing or
    /// comparisons.
    pub fn with_dynamic(self, payload: *mut (), name: &'a str) -> Self {
        Self {
            dynamic: Some((payload, name)),
            ..self
        }
    }
}
impl PartialEq for DynTypeId<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.compare_key() == other.compare_key()
    }
}
impl Eq for DynTypeId<'_> {}
impl hash::Hash for DynTypeId<'_> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        self.compare_key().hash(state)
    }
}

/// Trait for types that interact with the Qiskit dynamic-typing system ([`DynTyped`]) as static
/// Rust objects.
///
/// This trait is not object safe; use the blanket implementation of [`DynTyped`] for that.
pub trait StaticDynTyped {
    fn static_dyn_type_id() -> DynTypeId<'static>;
}
/// Declare a static Rust type as directly usable with the Qiskit dynamic-typing system.
///
/// This should only be done for objects that are exposed in `dyn Any`-derived handles as direct
/// pointers.  For example, `DAGCircuit` is valid for use with this, but structs in `qiskit-cext`
/// that wrap
#[macro_export]
macro_rules! static_dyn_typed {
    ($ty:ty) => {
        impl $crate::dyn_types::StaticDynTyped for $ty {
            fn static_dyn_type_id() -> $crate::dyn_types::DynTypeId<'static> {
                $crate::dyn_types::DynTypeId::of::<$ty>()
            }
        }
    };
}
pub use static_dyn_typed;

/// Objects that can interact with Qiskit's dynamic-typing subsystem.
///
/// There are two components to the system: the static Rust type that backs the object, and any
/// additional dynamic typing on top of that, tracked by the wrapper object.
///
/// First-class objects defined in Rust can implement [`StaticDynTyped`] and use the blanket
/// implementation that provides this object-safe variant.
pub trait DynTyped: any::Any {
    /// The dynamic type identifier.
    fn dyn_type_id(&self) -> DynTypeId<'_>;
}
impl<T: StaticDynTyped + 'static> DynTyped for T {
    fn dyn_type_id(&self) -> DynTypeId<'_> {
        T::static_dyn_type_id()
    }
}
