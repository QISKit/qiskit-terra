# `qiskit-cext`

This crate contains the bindings for Qiskit's C API.
This crate is responsible for the C API symbols only; the header files needed to access it are
created by `qiskit-bindgen` by parsing the source of this crate.

If you add a new `pub extern "C" fn` in this crate, you will also need to give it a slot
in `qiskit-cext-vtable`.

## Building the library

The default build mode of `qiskit-cext` is as an `rlib` so that it can be included in `qiskit-pyext`
and re-exposed through there.

To build the library in standalone mode for distribution and direct use by other C programs, you
need to build the crate as a `cdylib` instead.  The easiest way to do that, which will also produce
a complete "distribution" directory in `<repo>/dist/c`, along with the header files, is to run
```bash
make c
```

To build only the library object (which is what this crate is actually responsible for), you can use
the subrecipe
```bash
make clib
```

## Example C usage

The following example uses the header to build a 100-qubit observable:
```c
#include <complex.h>
#include <qiskit.h>
#include <stdint.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    // build a 100-qubit empty observable
    uint32_t num_qubits = 100;
    QkObs *obs = qk_obs_zero(num_qubits);

    // add the term 2 * (X0 Y1 Z2) to the observable
    complex double coeff = 2;
    QkBitTerm bit_terms[3] = {QkBitTerm_X, QkBitTerm_Y, QkBitTerm_Z};
    uint32_t indices[3] = {0, 1, 2};
    QkObsTerm term = {coeff, 3, bit_terms, indices, num_qubits};
    qk_obs_add_term(obs, &term);

    // print some properties
    printf("num_qubits: %u\n", qk_obs_num_qubits(obs));
    printf("num_terms: %lu\n", qk_obs_num_terms(obs));

    // free the memory allocated for the observable
    qk_obs_free(obs);

    return 0;
}
```
Refer to the C API documentation for more information and examples.

## Compiling

The above program can be compiled by including the header and linking to the `qiskit` library, which
are located in the standard directory configuration whose root is `dist/c`.
```bash
make c
gcc program.c -I<repo>/dist/c/include -lqiskit -L<repo>/dist/c/lib
```

The example program will then output
```bash
./a.out
```
```text
num_qubits: 100
num_terms: 1
```
