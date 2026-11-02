## Crate structure

All the crates in here are called `qiskit-*`, and are stored in directories that omit the `qiskit-`.

This crate structure currently serves the purpose of building only a single Python extension module, which still separates out some of the Rust code into separate logical chunks.
The intention is that (much) longer term, we might be wanting to expose more of Qiskit functionality directly for other languages to interact with without going through Python.

We use a structure with several crates in it for a couple of reasons:

* logical separation of code
* faster incremental compile times

When we're doing Rust/Python interaction, though, we have to be careful.
Pure-Rust FFI with itself over dynamic-library boundaries (like a Python C extension) isn't very natural, since Rust heavily prefers static linking.
If we had more than one Python C extension, it would be very hard to interact between the code in them.
This would be a particular problem for defining the circuit object and using it in other places, which is something we absolutely need to do.

### Libraries

* `qiskit-pyext` is the only crate that actually builds a Python C extension library.
  This is kind of like the parent crate of all the others, from an FFI perspective; others define `pyclass`es and `pyfunction`s and the like, but it's `qiskit-pyext` that builds the C extension.
  Our C extension is built as `qiskit._accelerate` in Python space.
* `qiskit-cext` is the crate that defines the C FFI for Qiskit. It defines the C API to work with the Rust code directly. It has two modes of operation: a standalone mode
  that compiles to a C dynamic library without any runtime dependency on the Python interpreter, and an embedded mode where the API is re-exported from `qiskit-pyext`
  and used to accelerate Python workflows when writing compiled extensions that interact with Qiskit.
* `qiskit-bindgen` is an internal library that contains the logic for constructing header files to access the C API defined in `qiskit-cext`.
* `qiskit-accelerate` is a catch-all crate for one-off accelerators. This should be the end of the dependency tree and only be in the dependency list for the public
  interface crates: `qiskit-pyext` or `qiskit-cext`. If what you're working on is small and largely self-contained, and doesn't fit in another crate you probably just
  want to put it in here, then bind it to the C extension module in `qiskit-pyext`.
* `qiskit-circuit` is a base crate defining the Rust-space circuit objects.
  This is one of the lowest points of the stack; everything that builds or works with circuits from Rust space depends on this.
* `qiskit-transpiler` is the crate defining the transpiler functionality.
* `qiskit-qasm2` is the OpenQASM 2 parser.
  This depends on `qiskit-circuit`, but is otherwise pretty standalone, and it's unlikely that other things will need to interact with it.
* `qiskit-qasm3` is the Qiskit-specific side of the OpenQASM 3 importer.
  The actual parser lives at https://github.com/Qiskit/openqasm3_parser, and is its own set of Rust-only crates.
* `qiskit-quantum-info` contains definitions and operations on quantum-information-related objects,
  without reference to circuit-level objects.  This is a base crate.  Interactions between
  circuit-level objects and quantum-info objects are either in `qiskit-circuit` (if they are part of
  core IR definitions) or `qiskit_synthesis::quantum_info` (if not).
* `qiskit-synthesis` is the crate for synthesis functionality.
* `qiskit-circuit-library` is the crate for circuit library functions. It contains constructors or other
  circuit functionality that builds on the core circuit data model defined in `qiskit-circuit`.
* `qiskit-providers` defines the quantum program representation and the backend interface that consumes it. A
  quantum program contains the semantics of classical processing that is near-time to the QPU in a quantum computer.
* `qiskit-util` is for small utility functions and data structures that are independent of all
  Qiskit-specific objects, so it can be depended-on by both `quantum-info` and `circuit`.

### Binaries

* `qiskit-bindgen-cli` is a toolkit for working with the code generation tooling around C header files
  and vtables, including linting them for consistency.  See the crate's `README` for more.

## Developer notes

### Beware of initialization order

The Qiskit C extension `qiskit._accelerate` needs to be initialized in a single go.
It is the lowest part of the Python package stack, so it cannot rely on importing other parts of the Python library at initialization time (except for exceptions through PyO3's `import_exception!` mechanism).
This is because, unlike pure-Python modules, the initialization of `_accelerate` cannot be done partially, and many components of Qiskit import their accelerators from `_accelerate`.

In general, this should not be too onerous a requirement, but if you violate it, you might see Rust panics on import, and PyO3 should wrap that up into an exception.
You might be able to track down the Rust source of the import cycle by running the import with the environment variable `RUST_BACKTRACE=full`.


### Tests

Most of our functionality is tested through the Python-space tests of the `qiskit` Python package, since the Rust implementations are (to begin with) just private implementation details.
However, where it's more useful, Rust crates can also include Rust-space tests within themselves.

Each of the Rust crates disables `doctest` because our documentation tends to be written in Sphinx's rST rather than Markdown, so `rustdoc` has a habit of thinking various random bits of syntax are codeblocks.
We can revisit that if we start writing crates that we intend to publish.

#### Running the tests

You need to make sure that the tests can build in standalone mode, i.e. that they link in `libpython`.
By default, we activate PyO3's `extension-module` feature in `crates/pyext`, so you have to run with the tests with that disabled:

```bash
cargo test --no-default-features
```

On Linux, you might find that the dynamic linker still can't find `libpython`.
If so, you can extend the environment variable `LD_LIBRARY_PATH` to include it:

```bash
export LD_LIBRARY_PATH="$(python -c 'import sys; print(sys.base_prefix)')/lib:$LD_LIBRARY_PATH"
```
