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

// This is a stub error file for distributions of the C API libraries that do not include the
// generated access to Python extension modules.  It is overwritten by build scripts of the full
// Python package.
//
// The top-level `qiskit.h` defines all the C API functions in a different manner if
// `QISKIT_PYTHON_EXTENSION` is defined.  In Python-aware builds/distributions of the C API, this
// file is replaced by one that is safe to use in those situations.  We leave this file in place
// with an explicit stub to provide a better explanations to users who have a version of the C API
// without Python support; the alternative is a preprocessor "file not found" error.

#error This Qiskit distribution does not include the ability to define Python extension modules. \
    Use `qiskit` as a Python-package build dependency.
