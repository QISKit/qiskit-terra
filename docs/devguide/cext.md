# Contributing: C API

Qiskit was fundamentally developed as a Python library, yet it has continued to
evolve into a more universal tool. As quantum computing steadily expands its
userbase, the need for different entry points to use Qiskit will lead to the
development of new pipelines, which ultimately resulted in the development of
the C API. 

**Use Cases**

- Wrappers like [Qiskit.jl](https://github.com/Qiskit/Qiskit.jl) and
  [qiskit-cpp](https://github.com/Qiskit/qiskit-cpp).
- Projects with high-performance computing (HPC) requirements.
- Python extension modules.

This document defines guidelines for C API development while helping address
stability, consistency, and familiarity concerns. The examples are written with
Rust and `cbindgen` in mind.

## Structures

C structs are either transparent or opaque. A transparent struct is fully
defined in the header file, allowing callers to access members and allocate
instances themselves. An opaque struct is only forward-declared; callers cannot
access members or allocate instances. `cbindgen` generates transparent structs
when the layout is `repr(C)`, otherwise opaque.

*Don't conflate transparency with Rust's `repr(transparent)`, which has a
completely different meaning.*

Use opaque structs when runtime invariants need enforcement, complex behavior
needs encapsulation, fields are likely to change.

```rust
pub struct Qubit {
    a: Complex64, 
    b: Complex64,
}
```

*Note that `repr(Rust)` is implicit.*

Use transparent structs when encoding plain data without complex behavior or
fields are unlikely to change.

```rust
#[repr(C)]
pub struct QubitArray {
    pub data: *mut *mut Qubit,
    pub len: usize,
}
```

*The `pub` keyword has no effect across the foreign function interface (FFI).
Though, it's best practice to include `pub` for documentation purposes.*

## Enumerations

C enums are named integer constants. Qiskit uses fixed-width enums with
`repr(u*)` for consistency across compilers. Since
`sizeof(enum t) == sizeof(int)` on most modern platforms, `repr(u32)` is a solid
default. Use `repr(u8)` for space-critical enums within tightly packed
structures. Explicit integer values discourage accidental breaking changes.

```rust
#[repr(u32)]
pub enum QubitType {
    Unknown = 0,
    Physical = 1,
    Logical = 2,
}
```

*The `Unknown` variant, often having the value `0`, is useful for error
handling.*

## Functions

C FFI functions need `extern "C"` and the unsafe `no_mangle` attribute.
`no_mangle` turns off name mangling so that callers have a well defined
symbol to link. The `pub` keyword serves as documentation.

```rust
#[unsafe(no_mangle)]
pub extern "C" fn qk_obs_zero(num_qubits: u32) -> *mut SparseObservable {
    let obs = SparseObservable::zero(num_qubits);
    Box::into_raw(Box::new(obs))
}
```

Use `&T` and `&mut T` instead of `*const T` and `*mut T` when the documented
contract is non-null. Thin references have the same ABI as raw pointers.
References better express the non-null contract in function signature. Also,
references avoid manually dereferencing with `&*` and `&mut *` in the function
body.

```rust
#[unsafe(no_mangle)]
pub extern "C" fn qk_obs_num_qubits(obs: &SparseObservable) -> u32 {
    obs.num_qubits()
}
```

Use `Option<&T>` and `Option<&mut T>` when passing `NULL` has defined behavior.
The optional explicitly states the nullable contract. The compiler guarentees
that `T` cannot be used without checking the `NULL` case.

Use `NonNull<T>` instead of `&mut T` when taking ownership of `T`. The former
accurately describes taking ownership of a raw pointer. `NonNull<T>` avoids
manually casting `&mut T` as `*mut T` in the function body.

The types `Option<&T>`, `Option<&mut T>`, and `NonNull<T>` have the same ABI as
raw pointers.

Both idioms are useful in combination for free-like functions. The call
`free(NULL)` is a no-op in C. Callers expect this behavior because it makes
memory management less cumbersome. Friendly cleanup functions take ownership of
`T` and handle the `NULL` case gracefully.

```rust
#[unsafe(no_mangle)]
pub extern "C" fn qk_obs_free(obs: Option<NonNull<SparseObservable>) {
    if let Some(obs) = obs {
        drop(Box::from_raw(obs.as_ptr()))
    }
}
```

*Note that `Option<T>` forces programmers to handle the `NULL` case before
interacting with `T`.*

## Integer Types

Prefer fixed-width integers like `i32` and `u64`. Avoid imprecise-width integers
like `c_int` and `c_long`. `cbindgen` generates headers using fixed-width
integers found in `inttypes.h`. For example, `i32` is mapped to `int32_t`.

## Namespacing

C has no namepaces, so every identifier with external linkage shares one flat
namespace across the entire program. If two header files define a function with
the same name, the linker will report a duplicate symbol error. Library authors
work around this by prefixing identifiers. Qiskit prefixes all identifiers with
`qk`, `QK`, or `Qk`. In some cases, identifiers are prefixed automatically by
`cbindgen`. Functions are named manually. The pattern
`qk_<GROUP>_<VERB>_<EXTRA>` is a useful mental model for naming functions.

- `qk_obs_new`
- `qk_obs_num_qubits`
- `qk_obs_free`
- `qk_circuit_new`
- `qk_circuit_add_register`
- `qk_circuit_free`

Memory management functions benefit from short, common verbs. Verbs like `new`
and `free` imply heap allocation and deallocation. Likewise, `init` and `clear`
suggest operating on storage the caller owns. The verb `copy` indicates that the
function creates a duplicate of an existing object. The caller is responsible
for the new copy's lifecycle.

## Errors

### Sentinels

Choose a sentinel value when functions fail trivially. Sentinels must fall
outside of the return value's domain. `-1` and `0` are poor sentinels if either
value has additional meaning. `-1` is suitable for index-returning functioons
because indicies are non-negative. `0` only works when the domain excludes zero.
`NAN` is problematic because `NAN != NAN`. Constants like `u64::MAX` require
callers to cast `-1` or include `limits.h` to handle errors without compiler
warnings. Prefer returning signed types for positive domains that won't exceed
$2^{63}$.

**Sentinel Examples**

| Type (Rust) | Type (C)       | Sentinel    | Domain |
| ----------- | -------------- | ----------- | ------ |
| `*mut T`    | `T *`          | `NULL`      |        |
| `enum T`    | `enum t`       | `T_UNKNOWN` |        |
| `isize`     | `intptr_t`     | `-1`        | `>= 0` |
| `u64`       | `uint64_t`     | `0`         | `>= 1` |

```rust
pub extern "C" fn qk_matrix_type(matrix: Option<&Matrix>) -> MatrixType {
    if let Some(matrix) = matrix {
        matrix.kind.into()        
    } else {
        MatrixType::Unknown
    }
}
```

### Error Codes

Return `ExitCode` for non-trivial functions with multiple failure points. These
functions often write to an `out` parameter for the success case. Given an
`ExitCode`, the callers should have access to a human readable error message.

```rust
pub extern "C" fn qk_obs_matrix(
    obs: &QkObs,
    out: &mut *mut Matrix,
) -> ExitCode {
    match obs.to_matrix() {
        Ok(matrix) => {
            *out = Box::into_raw(Box::new(matrix));
            ExitCode::Success
        },
        Err(error) => {
            *out = ptr::null_mut();
            error.into()
        },
    }
}
```

### Better Messages

`ExitCode` isn't always enough. To remedy this, some Qiskit functions accept
`char **`, optionally allocating a message on error. For example,
`qk_transpile` follows this pattern. The caller is responsible for calling
`qk_str_free` on the allocated message.

### Don't Panic!

You can almost always do better than panicking based on caller input. For
short-lived, experimental programs, panic could be acceptable. However,
consider that some long-running, production program could depend on Qiskit.
If a small part of it uses Qiskit incorrectly, it would be very unfriendly to
crash the entire program! Your code may panic without you realizing. For
example, the operation `a[i]` panics in the out-of-bounds case. 

## Versioning & Backwards Compatibility

C and Python share the same version number, following
[SemVer](https://semver.org/). Breaking changes are prohibited without
incrementing the major version number.

**Breaking Changes**

- Removing or renaming public functions, types, macros, constants, and headers.
- If declared exhausitive, adding an enum variant.
- Changing transparent struct fields.
- Tightening preconditions.
- Changing function signatures (including `const`-ness changes).

**Non-breaking Changes**

- Creating public functions, types, macros, constants, and headers.
- Adding enum variant(s) without changing integer values.
- Adding bit flags without changing the previous meanings.
- Changing the size and layout of opaque structs.
- Relaxing preconditions.
- Fixing bugs that bring behavior in line with the documentation.

## Platform-specific Features

It's best practice to avoid compiler and platform dependent features because
callers could be targeting multiple platforms. For example, `__int128_t` and
`_Complex double` are potentially relevant in the quantum domain, but those
types are only available in GCC. In this case, we would be better off using the
platform independent complex number utilities defined in `complex.h` since C99.
For reference, Qiskit targets C11 as the minimum C standard.

## Documentation

The documentation pipeline uses Doxygen, Breathe, and Sphinx, orchestrated by
the `tox -e docs` command.

### Doxygen

C API documentation is generated from outer doc comments written with
[Doxygen](https://www.doxygen.nl/manual/commands.html) syntax.

```rust
/// @ingroup QkFoo
///
/// Describe what the function does. Then, describe memory management responsibilities.
///
/// @param arg1 Describe `arg1`, including the contract (e.g. non-null).
/// @param arg2 Describe `arg2`, including the contract (e.g. non-null).
/// 
/// @return Describe the meaning of each return value, if any. Omit this section if the function
///         returns `void`.
#[unsafe(no_mangle)]
pub extern "C" fn qk_foo(arg1: *mut c_void, arg2: *mut c_void) -> i32;
```

*The optional `@ingroup` command groups related items.*

### Groups

C API documentation is organized into groups, defined in `docs/cdoc/index.h`
with the `@defgroup` command. The `@ingroup` command assigns items to a group.
Structs exposed directly from the data model must be manually added to an `.rst`
file.

### Ownership

C syntax cannot express ownership semantics, so the ownership contract must be
documented explicitly. Phrases like "takes ownership" or "consumes" aren't
meaningful to every reader. Alongside those phrases, state what the caller
_must not_ do for each outcome.

In the example below, `qk_target_add_instruction` takes ownership of `entry`
only when the instruction is successfully added to the target. The `entry`
parameter is typed `*mut` because the function may take ownership of it.

```rust
/// @ingroup QkTarget
/// 
/// Append an instruction to the ``QkTarget`` through a ``QkTargetEntry``. The caller must not use
/// or free `entry` if `QkExitCode_Success` is returned.
/// 
/// @param target A non-null pointer to the ``QkTarget``.
/// @param entry A non-null pointer to the ``QkTargetEntry``.
///
/// @return Returns `QkExitCode_Success` if the instruction is added.
extern "C" fn qk_target_add_instruction(target: *mut Target, entry: *mut TargetEntry) -> ExitCode;
```

### Safety

`unsafe` enables operations that are otherwise prohibited in safe Rust. When
writing FFI code, we often dereference raw pointers or call unsafe functions
within an `unsafe` block. These operations are unsafe because the compiler
cannot verify their validity statically. As the programmer, you are asserting
that you've checked the necessary conditions yourself. Document those conditions
in a **safety comment** so that future readers understand the assumptions you
made.

```rust
pub unsafe extern "C" fn qk_circuit_free(circuit: *mut Circuit) {
    if !circuit.is_null() {
        // SAFETY: `circuit` is non-null in this block.
        let circuit = unsafe { Box::from_raw(circuit) };
        drop(circuit)
    }
}
```

### Release Notes

Link C objects in release notes using Sphinx syntax in the `c` domain.

```
:c:struct:`QkFoo`
:c:type:`QkFoo`
:c:enum:`QkFoo`
:c:enumerator:`QkFoo_Bar`
:c:func:`qk_foo`
```

## Python Extensions

This section needs some more thought. Qiskit provides functions for extracting a
C-native pointer from a `PyObject *`. The working theory is that, in sequential
programming contexts, only 1 thread will access the pointer at a given instant.
We assume this is true for both GIL-attached and free- threaded Python. Thus, a
simple function that returns a data model pointer given a `PyObject *` should be
sufficient.

*See [this PR](https://github.com/Qiskit/qiskit/pull/17011) for details.*
