# `qiskit-cext-vtable`

This crate defines the machinery to specify ABI-stable vtables of function pointers, and provides
concrete vtables for the functions within `cext`.

This vtable can be compiled into dependencies on `cext` to include the actual function-pointers and
a complete vtable (when using the `addr` feature, in which case it depends on `cext`), or, if not
using the `addr` feature, then the tables can be built solely in terms of the function names, which
is used by build scripts to generate accessor files.

This exists as a separate module to `cext` because language-bindings generators typically do not
want to, or _cannot_ compile against `cext` fully.  For example, the build script of `pyext` cannot
depend on `cext` itself, because that would trigger a complete second compilation of Qiskit and
require the build script to link against `libpython` simply to run, both of which are highly
problematic for the build process.

If you add a new `pub extern "C" fn` in `qiskit-cext`, you will also need to give it a slot
in this crate.
