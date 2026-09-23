.. _capi-openqasm2:

===========
QkOpenQasm2
===========

Functionality for building a :c:type:`QkCircuit` from an OpenQASM 2 program, using the same
native Rust importer as :func:`qiskit.qasm2.loads`.

.. code-block:: c

    #include <qiskit.h>

    QkOpenQasm2Options options = qk_openqasm2_default_options();
    char *error = NULL;
    QkCircuit *qc = qk_circuit_from_openqasm2(
        "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; h q[0]; cx q[0], q[1];",
        &options, &error);

Data Types
==========

.. doxygenstruct:: QkOpenQasm2Options
    :members:

Functions
=========

.. doxygengroup:: QkOpenQasm2
    :members:
    :content-only:
