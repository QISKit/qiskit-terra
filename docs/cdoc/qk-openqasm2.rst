.. _capi-openqasm2:

===========
QkOpenQasm2
===========

Functionality for building a :c:type:`QkCircuit` from an OpenQASM 2 program.  The importer is
written natively in Rust and does not call into Python; unlike :func:`qiskit.qasm2.loads`, which
parses in Rust but builds the circuit in Python.

.. code-block:: c

    #include <qiskit.h>

    QkOpenQasm2Options *options = qk_openqasm2_options_new();
    QkCircuit *qc = NULL;
    char *error = NULL;
    QkExitCode code = qk_openqasm2_load(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0], q[1];",
        options, &qc, &error);
    qk_openqasm2_options_free(options);
    if (code != QkExitCode_Success) {
        printf("failed to load: %s\n", error);
        qk_str_free(error);
    } else {
        qk_circuit_free(qc);
    }

Data Types
==========

.. doxygentypedef:: QkOpenQasm2Options

Functions
=========

.. doxygengroup:: QkOpenQasm2
    :members:
    :content-only:
