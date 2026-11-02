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

//! Defines the contract of atomic units within a quantum program.

use crate::tensor::{Tensor, TensorType};

/// The [`ProgramOp::namespace`] of every op Qiskit defines.
pub const QISKIT: &str = "qiskit";

/// Destructure an op's operands, panicking if the count is wrong.
///
/// The panic is unreachable for an op held by a [`ProgramFunction`](crate::ProgramFunction), which
/// checks the count as the op is added.
#[macro_export]
macro_rules! unpack_operands {
    ($op:expr, $operands:expr, [$($name:ident),+ $(,)?]) => {
        let operands = $operands;
        let [$($name),+] = operands else {
            panic!(
                "{} expects {} operand(s), got {}",
                $op.full_name(),
                $op.arity(),
                operands.len()
            )
        };
    };
}

/// An atomic operation in a quantum program: a typed mapping from tensors to tensors.
///
/// An op declares how many operands it takes ([`Self::arity`]) and how its result types follow from
/// prospective operand types ([`Self::infer_output_types`]). Operands and results are flat and
/// positional, and inference is monomorphic: operand types the op accepts determine its result
/// types exactly.
///
/// An op may have a payload of its own, such as "which axis" information in
/// [`Mean`](crate::ops::Mean), quantum circuit instances in [`ShotLoop`](crate::ops::ShotLoop), or
/// a tensor in [`Constant`](crate::ops::Constant).
///
/// An op may implement [`Self::eval`] to perform the tensor manipulation it represents, and
/// [`Self::has_builtin_eval`] reports whether it does. An op without one is evaluated by a backend.
///
/// An op defined outside this crate lives in its own [`Self::namespace`] and is treated like
/// any other.
pub trait ProgramOp {
    /// The error this op reports for a rejected operand type or a failed evaluation.
    type Error;

    /// Return the name of this op within its namespace, `add` for instance.
    fn name(&self) -> &str;

    /// Return the namespace this op belongs to.
    fn namespace(&self) -> &str;

    /// Return the namespace and name as one string.
    fn full_name(&self) -> String {
        format!("{}.{}", self.namespace(), self.name())
    }

    /// Return the number of operand tensors this op consumes.
    fn arity(&self) -> usize;

    /// Return whether [`Self::eval`] contains an implementation.
    fn has_builtin_eval(&self) -> bool;

    /// Infer the types of this op's results from the types of its operands.
    ///
    /// This runs when the op is added to a program function, so that every program and every
    /// function it owns is well-defined at all times. The types returned here are the ones later
    /// ops are checked against.
    ///
    /// # Panics
    ///
    /// May panic if `inputs.len()` is not [`Self::arity`].
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, Self::Error>;

    /// Evaluate this op on `args`, returning one tensor per result.
    ///
    /// The returned tensors match, in count and type, what [`Self::infer_output_types`] returns
    /// the corresponding operand types. An error is a last resort: for example, a division op returns non-finite
    /// values for a zero divisor rather than failing, so that the rest of the program's data stays
    /// usable. An op whose [`Self::has_builtin_eval`] is false always returns an error.
    ///
    /// # Panics
    ///
    /// May panic if `args` does not correspond with a non-erroring input to
    /// [`Self::infer_output_types`].
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, Self::Error>;
}

/// The error of an op whose type inference or evaluation fails, type-erased.
pub type BoxedOpError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// An owned [`ErasedProgramOp`].
pub type BoxedProgramOp = Box<dyn ErasedProgramOp>;

/// A type-erased [`ProgramOp`].
pub trait ErasedProgramOp: std::any::Any + Send + Sync + sealed::Clonable {
    fn name(&self) -> &str;
    fn namespace(&self) -> &str;
    fn full_name(&self) -> String;
    fn arity(&self) -> usize;
    fn has_builtin_eval(&self) -> bool;
    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, BoxedOpError>;
    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedOpError>;
}

impl<O> ErasedProgramOp for O
where
    O: ProgramOp + Clone + Send + Sync + 'static,
    O::Error: std::error::Error + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        ProgramOp::name(self)
    }
    fn namespace(&self) -> &str {
        ProgramOp::namespace(self)
    }

    fn full_name(&self) -> String {
        ProgramOp::full_name(self)
    }

    fn arity(&self) -> usize {
        ProgramOp::arity(self)
    }

    fn has_builtin_eval(&self) -> bool {
        ProgramOp::has_builtin_eval(self)
    }

    fn infer_output_types(&self, inputs: &[TensorType]) -> Result<Vec<TensorType>, BoxedOpError> {
        ProgramOp::infer_output_types(self, inputs).map_err(|error| Box::new(error) as BoxedOpError)
    }

    fn eval(&self, args: &[Tensor]) -> Result<Vec<Tensor>, BoxedOpError> {
        ProgramOp::eval(self, args).map_err(|error| Box::new(error) as BoxedOpError)
    }
}

impl dyn ErasedProgramOp + 'static {
    /// Downcast a type-erased program op to the concrete op it holds.
    pub fn downcast_ref<O: ProgramOp + 'static>(&self) -> Option<&O> {
        (self as &dyn std::any::Any).downcast_ref()
    }
}

impl ToOwned for dyn ErasedProgramOp {
    type Owned = BoxedProgramOp;

    fn to_owned(&self) -> Self::Owned {
        self.clone_dyn()
    }
}

mod sealed {
    use super::{BoxedProgramOp, ProgramOp};

    /// Copying an op through a trait object.
    ///
    /// [`Clone`] is not dyn-compatible, because it returns `Self`. This trait is sealed and blanket
    /// implemented, so an implementor supplies only `Clone`.
    #[diagnostic::on_unimplemented(
        message = "Clone is required to store {Self} in a program function",
        note = "Consider annotating {Self} with `#[derive(Clone)]`"
    )]
    pub trait Clonable {
        fn clone_dyn(&self) -> BoxedProgramOp;
    }

    impl<O> Clonable for O
    where
        O: ProgramOp + Clone + Send + Sync + 'static,
        O::Error: std::error::Error + Send + Sync + 'static,
    {
        fn clone_dyn(&self) -> BoxedProgramOp {
            Box::new(self.clone())
        }
    }
}

#[cfg(test)]
mod test {
    use crate::ops::{Add, ProgramOp};
    use crate::tensor::Tensor;

    #[test]
    #[should_panic(expected = "qiskit.add expects 2 operand(s), got 1")]
    fn test_unpack_operands_names_the_op_and_the_arity_it_declares() {
        let _ = Add.eval(&[Tensor::from([1.0_f64])]);
    }
}
