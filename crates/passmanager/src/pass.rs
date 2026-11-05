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

use std::{
    any::{self, Any},
    fmt, marker,
};

use thiserror::Error;

use crate::{IR, PassContext};
use qiskit_util::dyn_types::DynTypeId;

/// The base behavior for compiler passes written in first-party Rust code.
///
/// This is the component that pass authors actually need to implement.
pub trait StaticPass<In, Out = In>: Send + Sync + Sized + 'static
where
    In: IR,
    Out: IR,
{
    /// Run the pass.
    fn run(&self, ir: Box<In>, context: &mut PassContext) -> anyhow::Result<Box<Out>>;

    /// Turn this object into the full type-erased version.
    ///
    /// If the base structure implements [`StaticPass`] for more than one pair of input and output
    /// types, you might need to call this as something like
    /// ```ignore
    /// <MyImplementer as StaticPass<In, Out>>::into_pass(ob)
    /// ```
    fn into_pass(self) -> Box<dyn Pass> {
        Box::new(StaticPassOb {
            ob: self,
            phantom: marker::PhantomData,
        })
    }
}

/// Type-system wrapper object to move the `In` and `Out` IR types of `StaticPass` into a concrete
/// object.
///
/// Storing the "associated types" as separate phantom markers in this object lets us have base Rust
/// structs that implement [`StaticPass`] for more than one input/output pair.  We still need to
/// have a fully monomorphised object to hold the type parameters when we do `impl Pass for
/// SomeObject`, because otherwise that implementation would overlap for _all_ the `StaticPass`
/// implementations of the base object.
struct StaticPassOb<T, In, Out = In> {
    ob: T,
    phantom: marker::PhantomData<(In, Out)>,
}
impl<P, In, Out> Pass for StaticPassOb<P, In, Out>
where
    In: IR,
    Out: IR,
    P: StaticPass<In, Out>,
{
    fn ir_id_in(&self) -> DynTypeId<'_> {
        DynTypeId::of::<In>()
    }
    fn ir_id_out(&self) -> DynTypeId<'_> {
        DynTypeId::of::<Out>()
    }
    fn name(&self) -> &str {
        any::type_name::<P>()
    }
    fn run(&self, ir: Box<dyn IR>, context: &mut PassContext) -> Result<Box<dyn IR>, PassError> {
        let ir = (ir as Box<dyn Any>)
            .downcast::<In>()
            .map_err(|_| PassError::Conversion)?;
        self.ob
            .run(ir, context)
            .map(|out| out as Box<dyn IR>)
            .map_err(PassError::Runtime)
    }
}

/// Errors returned by individual pass implementations.
#[derive(Error, Debug)]
pub enum PassError {
    /// The given input type failed to cast to the right type dynamically.
    #[error("failed to cast to expected input type")]
    Conversion,
    /// An arbitrary error during processing of the pass.
    #[error(transparent)]
    Runtime(#[from] anyhow::Error),
}

/// The trait for objects that can be called as transformation [`Task`](super::Task)s.
pub trait Pass: Any + Send + Sync {
    /// Return the type ID of the IR expected on input.
    fn ir_id_in(&self) -> DynTypeId<'_>;
    /// Return the type ID of the IR that is emitted by the pass.
    fn ir_id_out(&self) -> DynTypeId<'_>;
    /// A human-readable name for the pass.
    ///
    /// This is primarily for debugging purposes and may generally depend on the crate
    /// the pass is defined in. No part of the name should be relied on as being stable.
    fn name(&self) -> &str;
    /// Run the pass.
    ///
    /// In general, the [`PassManager`](crate::PassManager) construction logic will have validated
    /// the pipeline, so `ir` should typically cast correctly into the desired object.  However,
    /// badly behaved passes might have lied about their output types, or this trait may be called
    /// outside the context of the [`PassManager`](crate::PassManager).
    fn run(&self, ir: Box<dyn IR>, context: &mut PassContext) -> Result<Box<dyn IR>, PassError>;
}

impl fmt::Debug for dyn Pass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name().fmt(f)
    }
}
