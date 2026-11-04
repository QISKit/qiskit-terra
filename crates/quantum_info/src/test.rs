// This code is part of Qiskit.
//
// (C) Copyright IBM 2023
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

/// Helper for tests that involve calling Rayon code from within Miri.  This runs the given
/// function in a scoped threadpool, which is then immediately dropped.  This means that Miri will
/// not complain about the global (static) threads that are not joined when the process exits,
/// which is deliberate.
pub fn in_scoped_thread_pool<F, T>(worker: F) -> Result<T, ::rayon::ThreadPoolBuildError>
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    ::rayon::ThreadPoolBuilder::new()
        .build_scoped(::rayon::ThreadBuilder::run, |pool| pool.install(worker))
}
