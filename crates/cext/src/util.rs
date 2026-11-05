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

use std::ffi::CString;

/// Convert the `utf8` string into an owned [`CString`], replacing any interior nul bytes with a
/// literal two-byte `\0` marker.
pub fn cstring_from_utf8_lossy(utf8: impl Into<String>) -> CString {
    #[expect(clippy::disallowed_methods, reason = "this _is_ the helper")]
    match CString::new(utf8.into()) {
        Ok(out) => out,
        Err(e) => {
            // SAFETY: `e.into_vec()` reproduces `utf8.into()`, which was a `String` and therefore
            // valid UTF-8.
            let msg = unsafe { String::from_utf8_unchecked(e.into_vec()) };
            // SAFETY: the `msg.replace` call replaces all interior nul bytes with `\\0`.
            unsafe { CString::from_vec_unchecked(msg.replace("\0", "\\0").into_bytes()) }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cstring_from_string_with_interior_nuls() {
        assert_eq!(c"hello\\0world", &*cstring_from_utf8_lossy("hello\0world"),);
        assert_eq!(
            c"\\0how\\0are\\0we\\0doing?\\0",
            &*cstring_from_utf8_lossy("\0how\0are\0we\0doing?\0")
        );
    }
}
