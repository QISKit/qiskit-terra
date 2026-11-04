.. _capi-pass-manager:

===========================
Pass-manager infrastructure
===========================

Qiskit contains generic infrastructure for defining pass managers, passes and IRs dynamically.
Qiskit itself contains certain types that are used as compilation IRs in different situations, but
extensions are free to define their own as well.

.. doxygengroup:: pass-manager
   :members:
   :content-only:
