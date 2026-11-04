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

use std::sync::LazyLock;

// This has to be `pub(crate)` so that the `export_fn` macro can find it, but nothing in here is
// supposed to actually be used other than as internal implementation details.
#[doc(hidden)]
pub(crate) mod inner {
    #[derive(Copy, Clone, Debug)]
    pub struct ExportedFunctionPartial {
        pub name: &'static str,
        #[cfg(feature = "addr")]
        pub addr: usize,
    }

    pub fn last_element(path: &str) -> &str {
        path.rfind(":")
            .map(|index| path.split_at(index + 1).1)
            .unwrap_or(path)
    }
}

/// An exported C API function, along with a slot to place it in a function-pointer lookup table.
#[derive(Copy, Clone, Debug)]
pub struct ExportedFunction {
    /// The name of the function.
    pub name: &'static str,
    /// Which slot the function pointer should be assigned to, in its appropriate table.
    pub slot: usize,
    /// A pointer to the function, type erased to a pointer-width integer.
    ///
    /// In general, these derive from a type that looks like
    /// ```
    /// unsafe extern "C" fn(T0, T1, ...) -> TRet
    /// ```
    /// for some number of arguments and some (maybe void) return type.
    #[cfg(feature = "addr")]
    pub addr: usize,
}

/// Maximum number of children of any single [`ExportedFunctions`] object.
///
/// Upping this causes more static memory use, but it shouldn't be too onerous.  You can nest
/// [`ExportedFunctions`] objects at any depth without trouble.
pub const MAX_CHILDREN: usize = 8;

/// A compile-time list of exported functions, including potential subgroups of functions.
///
/// When creating one of these, you almost certainly want to assign it to a `static` variable; all
/// the data it is supposed to represent is static
pub struct ExportedFunctions {
    /// The amount of space reserved for the leaves.  It is a panic to reserve less space than
    /// required, but it's fine (and encouraged) to reserve as much space as you think you'll expand
    /// to, in any given set of [ExportedFunctions].
    leaves_reserve: usize,
    /// The calculated total length of reserved space (though there may be internal gaps that aren't
    /// technically reserved within it).  This is calculated at compile time, mostly for the
    /// purposes of causing compile-time errors of this crate if the requested reservations don't
    /// fit together properly.
    len: usize,
    /// The leaf functions owned by this set of [ExportedFunctions].  This has to be constructed
    /// lazily because the function-pointer values can't (in general) be calculated until the
    /// compiled artifact is loaded into a process's memory space.
    ///
    /// This shouldn't be used directly; use [Self::get_leaves] to build it to ensure the `assert`
    /// code is called too.
    leaves: LazyLock<Vec<Option<inner::ExportedFunctionPartial>>>,
    /// The offsets and references to each child owned by this object.  The funky static-sized array
    /// of maybe-uninitialized references is to make this all work at compile time.  The array is
    /// guaranteed to be zero or more `Some` values and all the remainder are `None`.
    children: [Option<(usize, &'static ExportedFunctions)>; MAX_CHILDREN],
}
impl ExportedFunctions {
    /// Create a new (lazy) list of exported functions.
    ///
    /// The first argument is how much space to reserve for the leaf nodes.  It must be at least as
    /// large as the vector of leaves, or this will panic when trying to access the functions.  The
    /// second is a non-capturing closure that produces a vector of items defined by `export_fn`
    /// (or `None`).
    ///
    /// The second argument has to be a lazy closure because the addresses of functions generally
    /// aren't set until the fully compiled binary has been loaded up into a process; they can't be
    /// set at compile time.
    ///
    /// You can then append children with [`add_child`][Self::add_child].  If you don't need any
    /// leaf functions, use [`empty`][Self::empty].
    pub const fn leaves(
        reserve: usize,
        slots: fn() -> Vec<Option<inner::ExportedFunctionPartial>>,
    ) -> Self {
        Self {
            leaves_reserve: reserve,
            len: reserve,
            leaves: LazyLock::new(slots),
            children: [None; MAX_CHILDREN],
        }
    }
    /// Create a new empty list of exported functions.
    ///
    /// You can then append children with [`add_child`][Self::add_child].
    pub const fn empty() -> Self {
        Self::leaves(0, Vec::new)
    }

    /// Add a group of exported functions as a child of this set.
    ///
    /// You must add children in offset order, or the compile-time checks on validity will fail.
    ///
    /// # Panics
    ///
    /// If there are already [`MAX_CHILDREN`] children attached to this set of functions, or if the
    /// base `offset` is less than the maximum current reservation.
    pub const fn add_child(mut self, offset: usize, fns: &'static ExportedFunctions) -> Self {
        if offset < self.len {
            panic!("offset is less than previously reserved space; don't fill in holes");
        }
        let mut i = 0;
        while self.children[i].is_some() {
            i += 1;
            if i == MAX_CHILDREN {
                // We'd panic even without this catch, but this just makes sure the dev sees a
                // clearer message about what's gone wrong.
                panic!("too many children; consider using deeper nesting");
            }
        }
        // There isn't actually a value to throw away here, but we had to do this little dance with
        // the iteration and `replace` to keep things safely `const`
        self.children[i].replace((offset, fns));
        self.len = offset + fns.len;
        self
    }

    /// The total length of the reservation
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn get_leaves(&self) -> &[Option<inner::ExportedFunctionPartial>] {
        let slots = &self.leaves;
        assert!(slots.len() <= self.leaves_reserve);
        slots
    }

    /// Iterate through all the exported functions, filling in their complete slot information from
    /// a base offset.
    ///
    /// The order of iteration is not defined with respect to the slots; they are not guaranteed to
    /// be in sorted order.
    ///
    /// Requiring a `'static` lifetime on `self` is mostly just laziness in defining this (it lets
    /// us safely do it with iterator combinators rather than producing a custom "walker" class
    /// while avoiding recursive types), but one that shouldn't actually affect use of this, since
    /// all [`ExportedFunctions`] objects are expected to be defined as `static`s.
    pub fn exports(&'static self, offset: usize) -> Box<dyn Iterator<Item = ExportedFunction>> {
        Box::new(
            self.get_leaves()
                .iter()
                .enumerate()
                .filter_map(move |(i, func)| {
                    func.as_ref().map(move |func| ExportedFunction {
                        name: func.name,
                        slot: offset + i,
                        #[cfg(feature = "addr")]
                        addr: func.addr,
                    })
                })
                .chain(
                    self.children
                        .iter()
                        .filter_map(move |funcs| {
                            funcs
                                .as_ref()
                                .map(move |(inner, funcs)| funcs.exports(offset + inner))
                        })
                        .flatten(),
                ),
        )
    }

    pub fn slots(&'static self) -> Vec<Option<ExportedFunction>> {
        let mut out = vec![None; self.len];
        for export in self.exports(0) {
            out[export.slot] = Some(export);
        }
        out
    }
}

/// Create an entry in an `ExportedFunctions` table.
///
/// The first argument to the macro is the path to export, which should resolve to some object
/// declared like
/// ```
/// #[unsafe(no_mangle)]
/// pub unsafe extern "C" fn qk_my_function() {}
/// ```
/// (or just `pub extern` - the `unsafe` is not important).
///
/// If the function is only defined when certain features are active, you can follow the path with a
/// comma-separated list of `feature = "my-feature"` items, such as
/// ```
/// export_fn!(path::to::qk_my_function, feature = "python_binding", feature = "cool_stuff");
/// ```
macro_rules! export_fn {
    ($fn:path) => {
        Some($crate::impl_::inner::ExportedFunctionPartial {
            name: $crate::impl_::inner::last_element(stringify!($fn)),
            #[cfg(feature = "addr")]
            addr: ($fn as *const ()).addr(),
        })
    };
    ($fn:path, $(feature = $feat:tt),+) => {{
        #[cfg(all($(feature = $feat),+))]
        let out = $crate::impl_::export_fn!($fn);
        #[cfg(not(all($(feature = $feat),+)))]
        let out = None::<$crate::impl_::inner::ExportedFunctionPartial>;
        out
    }};
}
pub(crate) use export_fn;

/// Helper module to made exports easier.  This should contain everything that modules need to
/// define their exports.
pub(crate) mod prelude {
    pub(crate) use super::{ExportedFunctions, export_fn};
}
