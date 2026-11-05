.. _capi-operations:

=================
QkCustomOp
=================

.. code-block:: c

   typedef struct QkCustomOp QkCustomOp

A representation of an operation that has been defined in C.

This operation object contains the minimal functionality an object
should adhere to in order operate on a ``QkCircuit``.

Any object that can be implemented using ``QkCustomOp`` will be
dynamically dispatched to be added to the circuit. In other words,
the circuit is unaware of the type of object it is accepting, but
it will work with it as long as it has the functionality expected
from any operation.

To achieve this, an operation is defined by two parts:
- The original pointer to the operation struct.
- The pointer to a vtable with the function slots that define
the functionality of this operation. See :c:func:`qk_custom_op_vtable_new`
for more details.

Here's a quick example of what that looks like:
   
.. code-block:: c

   // Define an operation with a single attribute.
   struct foo_gate {
      uint32_t num_qubits;
   };
   // Represents the name of the operation.
   const char *FOO_NAME = "foo";
   // Design the required methods for the vtable.
   const char *foo_name(const void *gate) {
      // Cast void to original pointer.
      struct foo_gate *_self = (struct foo_gate *)gate;
      // Cast once more to consume it
      (void)_self;
      return FOO_NAME;
   }
   uint32_t foo_num_qubits(const void *gate) {
      struct foo_gate *self = (struct foo_gate *)gate;
      // Used stored attirbute as return value.
      return self->num_qubits;
   }
   // Use same logic below for required methods that have
   // fixed values.
   uint32_t foo_num_clbits(const void *gate) {
      struct foo_gate *_self = (struct foo_gate *)gate;
      (void)_self;
      return 0;
   }
   // Implement all required methods.
   // Build list of entries for the vtable (at least 7 required entries)
   QkCustomOpVtableEntry entries[7] = {
      {.slot = 0, .func = foo_name},
      {.slot = 1, .func = foo_num_qubits},
      {.slot = 2, .func = foo_num_clbits},
      // ...
      // End with sentinel value
      {.slot = -1, .func = NULL},
   };
   // Create a vtable
   QkCustomOpVtable *foo_vtable = qk_custom_op_vtable_new(entries);
   // Declare a sample instance
   struct foo_gate foo_3q = {
      .num_qubits = 3,
   };
   // Create the custom operation
   QkCustomOp *foo_3q_custom = qk_custom_op_new(&foo_3q, foo_vtable);
   // Add to a circuit
   QkCircuit *circuit = qk_circuit_new(3, 0);
   uint32_t qubits[3] = {0, 1, 2};
   qk_circuit_custom_operation(circuit, foo_3q_custom, qubits, NULL, NULL);
   

Data Types
==========

.. doxygenenum:: QkCustomOpSlot

Functions
=========

.. doxygengroup:: QkCustomOp
   :content-only:
