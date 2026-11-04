.. _capi-config:

=============
Configuration
=============

The Qiskit C API has minimal configuration.

.. c:macro:: QISKIT_PYTHON_EXTENSION

   If defined before including ``qiskit.h``, the header files will define all of symbols in a mode
   safe for use with Python extension modules.  This means that all API functions will evaluate to
   function-pointer dereferences from a lookup table.

   This is a user-defined macro and not defined by Qiskit itself.

.. c:function:: int qk_import(void)

   Import the Qiskit C API from the Python-space :py:mod:`qiskit` package.

   You must call this once per compilation unit, before attempting to call any C API functions.
   Failure to do so will typically result in null-pointer dereferences at runtime.

   You typically will want to do this inside your ``PyInit_*`` module initialization function.  For
   example, in ``my_extension.c``:

   .. code-block:: c

      #define QISKIT_PYTHON_EXTENSION
      #include <Python.h>
      #include <qiskit.h>

      static struct PyModuleDef my_extension_mod = {
         .m_base = PyModuleDef_HEAD_INIT,
         .m_name = "my_extension",
      };

      PyMODINIT_FUNC PyInit_my_extension(void) {
         if (qk_import() < 0) {
            return NULL;
         }
         return PyModuleDef_Init(&my_extension_mod);
      }

   This function is only defined when :c:macro:`QISKIT_PYTHON_EXTENSION` was defined prior to
   including ``qiskit.h``.

   :return: 0 on success, negative on failure.
