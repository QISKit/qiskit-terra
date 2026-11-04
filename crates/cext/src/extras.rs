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

//! The purpose of this module is to have "dummy" definitions of objects that Qiskit and `cext` use,
//! but aren't otherwise visible to `cbindgen.`  This is principally types that are macro-generated.

#[allow(dead_code)] // used by cbindgen
pub struct QuantumRegister;
#[allow(dead_code)] // used by cbindgen
pub struct ClassicalRegister;
