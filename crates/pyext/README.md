# `qiskit-pyext`

This is the Rust crate that actually builds the `qiskit._accelerate` Python extension module.

See the README at the `crates` top level for more information on the structure.
Any self-contained submodule crates (e.g. `qasm2`) can define their own `pymodule`, but they should compile only as an rlib, and this crate should then build them into the top-level `qiskit._accelerate`.
This module is also responsible for rewrapping all of the bits-and-pieces parts of `crates/accelerate` into the `qiskit._accelerate` extension module.
