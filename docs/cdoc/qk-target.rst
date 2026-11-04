.. _capi-target:

========
QkTarget
========

.. code-block:: c

    typedef struct QkTarget QkTarget

A mapping of instructions and properties representing the particular constraints
of a backend. Its purpose is to provide the compiler with information that allows it
to compile an input circuit into another that is optimized taking in consideration the
``QkTarget``'s specifications. This structure represents a low level interface to the main
Target data structure in Rust, which represents the base class for its Python
counterpart, :class:`.Target`.

Here's an example of how this structure works:

.. code-block:: c

    #include <qiskit.h>
    #include <math.h>

    // Create a Target with 2 qubits
    QkTarget *target = qk_target_new(2);
    // Create an entry for a CX Gate with qargs (0, 1) and properties
    // duration = 0.0123
    // error = NaN
    uint32_t qargs[2] = {0, 1};
    QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);
    qk_target_entry_add_property(entry, qargs, 2, 0.0123, NAN);

    // Add a CX Gate to the target
    qk_target_add_instruction(target, entry);

    // Add a global H gate
    qk_target_add_instruction(target, qk_target_entry_new(QkGate_H));

    // Free the created target.
    qk_target_free(target);

The Target C API currently only supports additions of ``QkGate`` instances
with either no parameters or fixed parameters. Support for regular parameters
will be added in the future. The functionality will keep expanding over time
as we improve our Rust data model capabilities.

Data Types
==========

.. doxygenstruct:: QkInstructionProperties
   :members:

.. doxygenstruct:: QkTargetOp
   :members:

Functions
=========

.. doxygengroup:: QkTarget
    :members:
    :content-only:
