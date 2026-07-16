// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

mod extras;
mod pointers;
#[cfg(feature = "python_binding")]
mod py;

pub mod circuit;
pub mod circuit_library;
pub mod classical_expr;
pub mod control_flow;
pub mod dag;
pub mod exit_codes;
pub mod param;
pub mod sparse_observable;
pub mod transpiler;

pub use exit_codes::ExitCode;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Get the C API version of the loaded library.
///
/// If you are dynamically linking against Qiskit, in either a stand-alone or Python-extension
/// build, this function can be useful to check the version of the library actually loaded at
/// runtime.  The header-file macro `QISKIT_VERSION_HEX` is the equivalent of this function, but
/// for the version of the headers used at build time.
///
/// @return The version of the compiled Qiskit C API library, in the same format as the
///     `QISKIT_VERSION_HEX` header-file macro.
#[unsafe(no_mangle)]
pub extern "C" fn qk_api_version() -> u32 {
    const VERSION: u32 = {
        const fn parse_int(s: &[u8]) -> u32 {
            let mut i = 0;
            let mut out = 0;
            while i < s.len() {
                if s[i] < b'0' || s[i] > b'9' {
                    panic!("not a positive integer");
                }
                out *= 10;
                out += (s[i] - b'0') as u32;
                i += 1;
            }
            out
        }

        let mut v = (parse_int(env!("CARGO_PKG_VERSION_MAJOR").as_bytes()) << 24)
            | (parse_int(env!("CARGO_PKG_VERSION_MINOR").as_bytes()) << 16)
            | (parse_int(env!("CARGO_PKG_VERSION_PATCH").as_bytes()) << 8);
        let pre = env!("CARGO_PKG_VERSION_PRE");
        let pre = pre.as_bytes();
        if let Some((b"dev", serial)) = pre.split_at_checked(3) {
            v |= 0xa0 | parse_int(serial);
        } else if let Some((b"beta", serial)) = pre.split_at_checked(4) {
            v |= 0xb0 | parse_int(serial);
        } else if let Some((b"rc", serial)) = pre.split_at_checked(2) {
            v |= 0xc0 | parse_int(serial);
        } else if pre.is_empty() {
            v |= 0xf0;
        } else {
            panic!("could not parse 'CARGO_PKG_VERSION_PRE'");
        }
        v
    };
    VERSION
}
