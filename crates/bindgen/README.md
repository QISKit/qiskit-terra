# `qiskit-bindgen`

The logic to parse `qiskit-cext` and produce suitable C header files to access the functions in the
library produced by that crate.
This is an internal library only used as part of the build and distribution process of Qiskit.

This crate owns all parts of the stand-alone header-file generation logic, including the
custom-written include files (the `include` directory), the Rust template crate (the `pyo3-ffi`
directory), and the installation logic of both.

## Usage

This library is designed to be used by its consuming binary, `qiskit-bindgen-cli`, and (in the future)
the Python-extension build process in `qiskit-pyext`.

This encapsulates all the custom logic needed to parse the `qiskit-cext` source files into a
structured set of bindings to output.  Mostly, the work is done by `cbdingen`, though we have some
scraping setup logic to the configuration.

The library may, at some point, provide a wrapper around the `cbindgen` output to better expose the
functions and types that will be written out, so downstream libraries (including our own Python
extension) can generate raw language bindings to the C API from structured data, rather than having
to re-parse the generated C header files.
