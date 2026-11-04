.. _capi-quantum-register:

=================
QkQuantumRegister
=================

.. code-block:: c

   typedef struct QkQuantumRegister QkQuantumRegister

A common way to instantiate several bits at once is to create a register. A
register is a named collection of bits. Creating a register enables giving a
collection of bits a name which can be use as metadata around specific bits
in a circuit. This name will also typically be preseserved when exporting the
circuit to interchange languages.

You can create a register by calling ``qk_quantum_register_new()``, for example:

.. code-block:: c

    #include <qiskit.h>

    QkQuantumRegister *qreg = qk_quantum_register_new(5, "my_qreg");

Which creates a new 5 qubit register named ``"my_qreg"``.

Then to add the register to a circuit you use the
``qk_circuit_add_quantum_register()`` function:

.. code-block:: c

    QkCircuit *qc = qk_circuit_new(0, 0);
    qk_circuit_add_quantum_register(qc, qreg);
    uint32_t num_qubits = qk_circuit_num_qubits(qc); // 5

While circuits track registers, the registers themselves impart almost no behavioral
differences on circuits.

Functions
=========

.. doxygengroup:: QkQuantumRegister
   :members:
   :content-only:
