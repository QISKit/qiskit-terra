.. _capi-classical-register:

===================
QkClassicalRegister
===================

.. code-block:: c

   typedef struct QkClassicalRegister QkClassicalRegister

A common way to instantiate several bits at once is to create a register. A
register is a named collection of bits. Creating a register enables giving a
collection of bits a name which can be use as metadata around specific bits
in a circuit. This name will also typically be preserved when exporting the
circuit to interchange languages.

You can create a register by calling ``qk_classical_register_new()``, for
example:

.. code-block:: c

    #include <qiskit.h>

    QkClassicalRegister *creg = qk_classical_register_new(5, "my_creg");

Which creates a new 5 classical bit register named ``"my_creg"``.

Then to add the register to a circuit you use the
``qk_circuit_add_classical_register()`` function:

.. code-block:: c

    QkCircuit *qc = qk_circuit_new(0, 0);
    qk_circuit_add_classical_register(qc, creg);
    uint32_t num_qubits = qk_circuit_num_qubits(qc); // 5

While circuits track registers, the registers themselves impart almost no behavioral
differences on circuits.

Functions
=========

.. doxygengroup:: QkClassicalRegister
   :members:
   :content-only:
