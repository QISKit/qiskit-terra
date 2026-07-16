.. _capi-complex64:

===========
QkComplex64
===========

A complex, double-precision number representation. This data type is used to safely
use complex numbers within Qiskit, but is not necessarily designed for easy manipulation
on user-side. Instead, we provide functions to convert from compiler-native representation
(e.g. ``complex double`` for GNU or Clang compilers), which allow for ergonomic handling,
to Qiskit's ``QkComplex64`` representation, which is meant for passing into Qiskit functions
and structs.

Explicitly, Qiskit assumes the compiler-native complex number type in C to be
``_Dcomplex`` for Windows MSVC (if ``_MSC_VER`` is defined) and ``double _Complex`` otherwise.
In C++ (if ``__cplusplus`` is defined), the complex number type is always ``std::complex<double>``.


----------
Data types
----------

.. doxygenstruct:: QkComplex64
   :members:
   :undoc-members:


---------
Functions
---------

.. c:function:: static CMPLX_DOUBLE qk_complex64_to_native(QkComplex64 *value)

   Convert a ``QkComplex64`` to a compiler-native complex number representation.
   Note that ``CMPLX_DOUBLE`` is a placeholder for the compiler-native complex number type,
   see the top of the page for more details.

Example
-------

Assuming a GNU/clang compiler with ``complex double`` as native complex number, we have

.. code-block:: c

   QkComplex64 qk_value = {1, 1}; // represents 1 + i
   complex double value = qk_complex64_to_native(&qk_value);

Safety
------

Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkComplex64``.

:Parameters:
    **value** – A pointer to the ``QkComplex64`` to convert.

:Returns:
    A native representation of the complex number.


.. c:function:: static QkComplex64 qk_complex64_from_native(CMPLX_DOUBLE *value)

   Convert a compiler-native complex number to a ``QkComplex64``.
   Note that ``CMPLX_DOUBLE`` is a placeholder for the compiler-native complex number type,
   see the top of the page for more details.

Example
-------

Assuming a GNU/clang compiler with ``complex double`` as native complex number, we have

.. code-block:: c

   complex double value = 1 + I;
   QkComplex64 qk_value = qk_complex64_from_native(&value);

Safety
------

Behavior is undefined if ``value`` is not a valid, non-null pointer to a ``QkComplex64``.

:Parameters:
    **value** – A pointer to the native complex number.

:Returns:
    A ``QkComplex64`` representation of the complex number.

