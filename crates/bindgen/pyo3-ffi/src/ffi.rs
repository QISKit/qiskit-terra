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

// This is a dummy file that's overwritten by the crate-generation script.  It's just here as a base
// test that the macro works, and to suppress clippy/rust-analyzer warnings while editing the
// template crate.

use crate::declare_fn;

// This is just an example of what the generated file produces.
declare_fn!(crate::QK_FFI_CIRCUIT[0]; qk_api_version() -> u32);
