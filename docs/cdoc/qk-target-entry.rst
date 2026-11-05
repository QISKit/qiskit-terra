.. _capi-target-entry:

=============
QkTargetEntry
=============

.. code-block:: c

    typedef struct QkTargetEntry QkTargetEntry

A mapping of qubit arguments and properties representing gate map of the
Target. This feature is used due to not having valid native mappings available
from C.

Here's an example of how this structure works:

.. code-block:: c

    #include <qiskit.h>
    #include <math.h>

    // Create a Target Entry for a CX Gate
    QkTargetEntry *entry = qk_target_entry_new(QkGate_CX);

    // Set a name for the entry
    qk_target_entry_set_name(entry, "cx_gate");

    // Add mapping between (0,1) and properties duration of 10e-9 and unknown error.
    uint32_t qargs[2] = {0, 1};
    qk_target_entry_add_property(entry, qargs, 2, 10e-9, NAN);

    // Add mapping between Global, and no set duration NAN and error of 0.003)
    qk_target_entry_add_property(entry, NULL, 0, NAN, 0.003);


Functions
=========

.. doxygengroup:: QkTargetEntry
    :members:
    :content-only:
