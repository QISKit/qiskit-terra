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

//! A [`Tensor`] and its staticly introspectable data-less counterpart, a [`TensorType`].
//!
//! A value is a dense array over one of a fixed set of [`DType`]s. Its type pairs a dtype with a
//! shape whose axes are each either a fixed size or a size bounded above ([`Dim`]).

mod broadcast;
mod dtype;
mod error;
mod tensor_type;
mod value;

pub use broadcast::broadcast_shape;
pub use dtype::{DType, promotion};
pub use error::TensorError;
pub use tensor_type::{Dim, TensorType, broadcast_dims, broadcast_dims_to, require_static};
pub use value::Tensor;
