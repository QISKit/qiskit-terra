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
    char *error = NULL;
    QkCircuit *qc = qk_circuit_from_openqasm2(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0], q[1];",
        options, &error);
    qk_openqasm2_options_free(options);

Data Types
==========

.. doxygentypedef:: QkOpenQasm2Options

Functions
=========

.. doxygengroup:: QkOpenQasm2
    :members:
    :content-only:
